"""
data_cache.py — Persistent Atomic Downloader v11.49
===================================================
Robustly manages downloading, parsing, and persisting yfinance data.

Callers must invoke configure_data_cache() before using module functions.

BUG FIXES (murder board):
- FIX-MB-GROWWADJ: GrowwProvider._build_adj_close_from_batches uses only ffill
  (no bfill) when aligning the adjustment ratio, preventing look-ahead
  contamination across holiday gaps.
- FIX-MB-GROWW429: GrowwProvider._fetch_candles_chunk returns _RATE_LIMITED
  sentinel on HTTP 429, enabling exponential backoff in the outer loop.
- FIX-MB-DC-02: _fetch_full_history now caps consecutive rate-limit retries at
  _MAX_RATE_LIMIT_RETRIES (5). After the cap is hit, a warning is logged and the
  history fetch aborts for that symbol rather than blocking indefinitely.
- FIX-MB-DC-01: _recover_from_stale_cache now updates the manifest entry for
  successfully recovered symbols so that subsequent load_or_fetch calls do not
  repeatedly trigger download attempts for stale-but-available parquets.
- BUG-FIX-FALLBACK: _process_chunk now returns the set of validated-and-saved
  tickers. The fallback chain uses this ground-truth set instead of raw column
  presence, so NaN-only provider responses no longer block SecondaryProvider.
- BUG-FIX-DOTENV-DC: shared load_dotenv_safe() strips inline comments before
  removing quotes, and is used across cache/optimizer/fallback modules.
- BUG-FIX-MANIFEST-IO: _save_manifest moved outside the for-chunk loop;
  one write per load_or_fetch call instead of one per chunk.
- BUG-FIX-FILLNA: GrowwProvider._build_adj_close_from_batches replaces
  fillna(1.0) on leading NaN ratio with bfill, preventing a price
  discontinuity at the oldest known adjustment boundary.
- BUG-FIX-STALE-RETRY: _recover_from_stale_cache now writes retry_after=tomorrow
  to the manifest entry, and the staleness check honours this cooldown.  Previously
  every run re-downloaded stale symbols, failed all providers, and recovered from
  the stale cache again — an infinite hammer-and-recover loop.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from abc import ABC, abstractmethod
from pathlib import Path

from log_config import load_dotenv_safe
from shared_constants import COLUMN_ADJ_CLOSE, COLUMN_STOCK_SPLITS, TIMEZONE_IST

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
_IST_TZ = TIMEZONE_IST

# Sentinel returned by _fetch_candles_chunk to signal rate-limit hit
_RATE_LIMITED = object()

# Maximum consecutive 429 retries per symbol before aborting
_MAX_RATE_LIMIT_RETRIES = 5
_MANIFEST_LOCK_POLL_SEC = 0.1
_MANIFEST_LOCK_TIMEOUT_SEC = 30.0
_SUSPENSION_GAP_DAYS = 7

_DEFAULT_CACHE_DIR = Path(os.getenv("DATA_CACHE_DIR", "data/cache"))
CACHE_DIR: Path | None = _DEFAULT_CACHE_DIR
MANIFEST_FILE: Path | None = _DEFAULT_CACHE_DIR / "_manifest.json"
_MANIFEST_LOCK_DIR: Path | None = _DEFAULT_CACHE_DIR / "_manifest.lock"
_CACHE_CONFIGURED = False


def _safe_yf_download(*args, **kwargs) -> pd.DataFrame:
    """
    Run yf.download while suppressing yfinance's noisy stdout/stderr prints.

    yfinance writes per-symbol "possibly delisted" and "Failed downloads"
    messages directly to stderr/stdout. When pulling large universes this can
    flood logs and hide real errors. We capture those streams and keep only a
    compact debug summary.
    """
    out_buf = StringIO()
    err_buf = StringIO()
    yfinance_logger_names = {"yfinance"}
    yfinance_logger_names.update(
        name for name in logging.Logger.manager.loggerDict
        if name.startswith("yfinance.")
    )

    previous_logger_states: dict[str, tuple[int, bool, bool]] = {}
    for name in yfinance_logger_names:
        logger_obj = logging.getLogger(name)
        previous_logger_states[name] = (
            logger_obj.level,
            logger_obj.propagate,
            logger_obj.disabled,
        )
        # yfinance emits per-symbol fetch failures at ERROR level which can
        # flood structured logs for large universes. Suppress these temporary
        # noisy logs here and rely on our compact cache-level summaries.
        logger_obj.setLevel(logging.CRITICAL + 1)
        logger_obj.propagate = False

    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            data = yf.download(*args, **kwargs)
    finally:
        for name, (level, propagate, disabled) in previous_logger_states.items():
            logger_obj = logging.getLogger(name)
            logger_obj.setLevel(level)
            logger_obj.propagate = propagate
            logger_obj.disabled = disabled

    noisy_lines: list[str] = []
    for text in (out_buf.getvalue(), err_buf.getvalue()):
        if not text:
            continue
        noisy_lines.extend(
            ln.strip() for ln in text.splitlines()
            if ln.strip()
        )

    if noisy_lines:
        logger.debug(
            "[Cache] Suppressed %d yfinance warning line(s). First line: %s",
            len(noisy_lines),
            noisy_lines[0],
        )
    return data

# Large multi-ticker yfinance requests are brittle for mixed universes that
# include newly listed/suspended symbols. Smaller chunks reduce the chance that
# one malformed ticker poisons the whole batch and improves completion rates on
# constrained retail connections.
_DOWNLOAD_CHUNK_SIZE = 25


def configure_data_cache(
    cache_dir: Optional[Path] = None,
    dotenv_path: Optional[Path] = None,
) -> None:
    """Initialize data-cache environment and filesystem paths.

    Callers must invoke configure_data_cache() once before using module
    functions so cache paths and optional local env vars are initialized.
    """
    global CACHE_DIR, MANIFEST_FILE, _MANIFEST_LOCK_DIR, _CACHE_CONFIGURED
    if dotenv_path is None and cache_dir is not None:
        candidate = Path(cache_dir)
        if candidate.name == ".env" or candidate.suffix == ".env":
            dotenv_path = candidate
            cache_dir = None
    load_dotenv_safe(dotenv_path)
    # After load_dotenv_safe, honor DATA_CACHE_DIR from .env if set
    resolved_cache_dir = cache_dir or os.environ.get("DATA_CACHE_DIR") or _DEFAULT_CACHE_DIR
    CACHE_DIR = Path(resolved_cache_dir)
    MANIFEST_FILE = CACHE_DIR / "_manifest.json"
    _MANIFEST_LOCK_DIR = CACHE_DIR / "_manifest.lock"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_CONFIGURED = True


_GROWW_BASE_URL           = "https://api.groww.in/v1"
_GROWW_DAILY_INTERVAL     = 1440
_GROWW_MAX_DAYS_PER_REQUEST = 1080
_GROWW_CHUNK_DAYS         = 1080
_GROWW_INDEX_PREFIXES     = ("^",)


def _ensure_cache_paths_configured() -> None:
    global _CACHE_CONFIGURED
    if not _CACHE_CONFIGURED:
        configure_data_cache()


class _ManifestProcessFileLock:
    """Cross-process lock implemented using an atomic lock directory with stale-lock cleanup."""

    def __init__(self, lock_dir: Path, timeout_s: float = _MANIFEST_LOCK_TIMEOUT_SEC) -> None:
        self._lock_dir = lock_dir
        self._timeout_s = timeout_s
        self._owner_file = lock_dir / "owner"

    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process with the given PID is alive."""
        if pid <= 0:
            return False
        try:
            # os.kill with signal 0 checks process existence without sending a signal
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            # Process doesn't exist
            return False
        except PermissionError:
            # Process exists but we don't have permission to signal it
            return True
        except OSError as e:
            # Distinguish between ESRCH (no such process) and EPERM (permission denied)
            import errno
            if e.errno == errno.ESRCH:
                # Process doesn't exist
                return False
            elif e.errno == errno.EPERM:
                # Process exists but can't be signaled
                return True
            else:
                # Other OS errors - assume process might be alive to be safe
                return True
        except Exception:
            # Fallback: assume process might be alive on unexpected errors
            return True

    def _is_lock_stale(self) -> bool:
        """
        Detect if the current lock is stale by checking:
        1. Owner file existence and parsability
        2. Process liveness via os.kill(pid, 0)
        3. Lock age against a stale threshold (2x timeout)

        Returns True if the lock is stale and can be safely removed.
        """
        if not self._owner_file.exists():
            # Owner file missing but lock dir exists - might be in-progress acquisition.
            # Wait briefly for owner file to appear to avoid race with a fresh lock.
            # But first check if lock dir was removed - fast-path out if so.
            if not self._lock_dir.exists():
                # Lock dir was already removed by another process - not stale, just gone
                return False

            max_wait = 1.0  # Wait up to 1 second for owner file to appear
            wait_step = 0.05
            elapsed = 0.0
            while elapsed < max_wait:
                # Check if lock dir was removed during our wait
                if not self._lock_dir.exists():
                    # Lock dir removed - not stale, just gone
                    return False

                time.sleep(wait_step)
                elapsed += wait_step
                if self._owner_file.exists():
                    # Owner file appeared, proceed with normal staleness check
                    break
            else:
                # Owner file still missing after wait - check lock dir age
                # But first verify lock dir still exists
                if not self._lock_dir.exists():
                    # Lock dir removed during wait - not stale, just gone
                    return False

                try:
                    lock_dir_stat = self._lock_dir.stat()
                    lock_dir_age = time.time() - lock_dir_stat.st_mtime
                    # Only treat as stale if directory is older than our wait window
                    if lock_dir_age < max_wait:
                        # Directory is fresh, likely still being set up
                        return False
                    # Directory is old with no owner file - treat as stale
                    return True
                except Exception:
                    # Can't stat lock dir, treat as stale to be safe
                    return True

        try:
            # Read and parse owner file
            content = self._owner_file.read_text(encoding="utf-8").strip()
            pid = None
            for part in content.split():
                if part.startswith("pid="):
                    try:
                        pid = int(part.split("=", 1)[1])
                        break
                    except (ValueError, IndexError):
                        pass

            # If we can't parse the PID, check file age only
            if pid is None:
                logger.debug("[Lock] Could not parse PID from owner file, checking age only")
                # Fall through to age-based check below
            else:
                # Check if the owning process is still alive
                if self._is_process_alive(pid):
                    # Process is alive - lock is NOT stale regardless of age
                    logger.debug("[Lock] Owner PID %d is alive, lock is valid", pid)
                    return False
                else:
                    # Process is dead - lock IS stale regardless of age
                    logger.debug("[Lock] Owner PID %d is dead, lock is stale", pid)
                    return True

            # Fallback: if PID couldn't be parsed, use file age as safety net
            stat_info = self._owner_file.stat()
            lock_age = time.time() - stat_info.st_mtime
            stale_threshold = max(60.0, self._timeout_s * 2.0)

            return lock_age >= stale_threshold

        except FileNotFoundError:
            # Owner file disappeared between exists check and read
            return True
        except Exception as exc:
            # On unexpected errors reading/parsing, check age as fallback
            logger.debug("[Lock] Error checking lock staleness: %s", exc)
            try:
                stat_info = self._owner_file.stat()
                lock_age = time.time() - stat_info.st_mtime
                stale_threshold = max(60.0, self._timeout_s * 2.0)
                return lock_age >= stale_threshold
            except Exception:
                # Can't determine staleness reliably, assume not stale to be safe
                return False

    def _remove_stale_lock(self) -> bool:
        """
        Attempt to remove a stale lock directory and owner file.
        Returns True if removal succeeded, False otherwise.
        Uses try/except to guard against races with live lockers.
        """
        try:
            # Try to remove owner file first
            if self._owner_file.exists():
                self._owner_file.unlink()
            # Then remove the lock directory
            if self._lock_dir.exists():
                self._lock_dir.rmdir()
            logger.debug("[Lock] Successfully removed stale lock at %s", self._lock_dir)
            return True
        except FileNotFoundError:
            # Lock already removed by another process
            return True
        except OSError as exc:
            # Directory not empty or permission denied - another process may have claimed it
            logger.debug("[Lock] Could not remove stale lock: %s", exc)
            return False
        except Exception as exc:
            logger.debug("[Lock] Unexpected error removing stale lock: %s", exc)
            return False

    def __enter__(self) -> "_ManifestProcessFileLock":
        deadline = time.monotonic() + max(0.0, float(self._timeout_s))
        stale_check_attempted = False

        while True:
            try:
                self._lock_dir.mkdir(parents=False, exist_ok=False)
                self._owner_file.write_text(
                    f"pid={os.getpid()} token={uuid.uuid4().hex}\n",
                    encoding="utf-8",
                )
                return self
            except FileExistsError:
                # Before waiting, check if the existing lock is stale
                # Only attempt stale cleanup once to avoid repeated overhead
                if not stale_check_attempted:
                    stale_check_attempted = True
                    if self._is_lock_stale():
                        logger.info(
                            "[Lock] Detected stale lock at %s, attempting cleanup",
                            self._lock_dir
                        )
                        if self._remove_stale_lock():
                            # Successfully removed stale lock, retry acquisition immediately
                            stale_check_attempted = False  # Allow another check on next collision
                            continue

                # Check timeout
                if time.monotonic() >= deadline:
                    # Perform one final stale check before raising TimeoutError
                    # to handle case where owner was alive but just exited
                    if self._is_lock_stale():
                        logger.info(
                            "[Lock] Final check found stale lock at %s before timeout",
                            self._lock_dir
                        )
                        if self._remove_stale_lock():
                            # Reset and allow one more acquisition attempt
                            stale_check_attempted = False
                            continue
                    raise TimeoutError(
                        f"Timed out waiting for manifest lock {self._lock_dir}"
                    )
                time.sleep(_MANIFEST_LOCK_POLL_SEC)

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            # Atomically rename the lock directory to prevent race conditions
            # where waiters see a missing owner file before the directory is removed
            # Use unique name to avoid collision with leftover released directories
            base_name = f"{self._lock_dir.name}_released"
            released_dir = self._lock_dir.parent / f"{base_name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"

            # Ensure uniqueness even if PID+uuid somehow collides
            attempt = 0
            while released_dir.exists() and attempt < 100:
                released_dir = self._lock_dir.parent / f"{base_name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
                attempt += 1

            try:
                self._lock_dir.rename(released_dir)
            except FileNotFoundError:
                # Lock already removed by another process
                return

            # Now clean up the renamed directory and owner file
            renamed_owner_file = released_dir / "owner"
            try:
                if renamed_owner_file.exists():
                    renamed_owner_file.unlink()
            except FileNotFoundError:
                pass

            try:
                released_dir.rmdir()
            except FileNotFoundError:
                pass
        except OSError as os_exc:
            # Tolerate benign OS errors (directory not empty, permission issues)
            logger.debug(
                "[Lock] Non-fatal error releasing lock %s: %s",
                self._lock_dir,
                os_exc,
            )
        except Exception as cleanup_exc:
            # Log unexpected errors during cleanup to avoid hiding programming errors
            logger.error(
                "[Lock] Unexpected error releasing lock %s: %s",
                self._lock_dir,
                cleanup_exc,
                exc_info=True,
            )


class DataProvider(ABC):
    @abstractmethod
    def download(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError


# ─── Groww Provider ───────────────────────────────────────────────────────────

class GrowwProvider(DataProvider):
    """
    Fetches daily OHLCV candles from the Groww API for NSE equity tickers.

    FIX-MB-GROWWADJ: Only ffill applied when aligning adj ratio — no bfill.
    FIX-MB-GROWW429: 429 returns _RATE_LIMITED sentinel for exponential backoff.
    FIX-MB-DC-02: Exponential backoff capped at _MAX_RATE_LIMIT_RETRIES to
      prevent indefinite blocking when a token is revoked or account suspended.

    Recommended usage: `with GrowwProvider(...) as provider:` so HTTP sessions
    are always released deterministically.
    """

    _GROWW_SLEEP_SECS = 0.2

    def __init__(self, api_token: Optional[str] = None) -> None:
        self.api_token = api_token or os.getenv("GROWW_API_TOKEN", "").strip()
        self._session: Optional[requests.Session] = None

    def close(self) -> None:
        # FIX-GROWW-SESSION-LEAK: close the session deterministically so pooled
        # sockets are released promptly in long-running processes/tests.
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "GrowwProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # FIX-GROWW-SESSION-LEAK: always close on context-manager exit.
        self.close()

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_token}",
                "Accept":        "application/json",
                "X-API-VERSION": "1.0",
            })
        return self._session

    @staticmethod
    def _to_groww_symbol(ticker: str) -> Optional[str]:
        if any(ticker.startswith(p) for p in _GROWW_INDEX_PREFIXES):
            return None
        for sfx in (".NS", ".BO", ".BSE"):
            if ticker.upper().endswith(sfx):
                return ticker[:-len(sfx)].upper()
        return ticker.upper()

    def _fetch_candles_chunk(
        self,
        groww_symbol: str,
        chunk_start: str,
        chunk_end: str,
    ):
        """
        Fetch one chunk of daily candles from Groww.
        Returns list of raw candle arrays, [] on normal failure, or
        _RATE_LIMITED sentinel on HTTP 429 so the caller can back off.
        """
        session = self._get_session()
        params: Dict[str, str | int] = {
            "exchange":           "NSE",
            "segment":            "CASH",
            "groww_symbol":       f"NSE-{groww_symbol}",
            "start_time":         f"{chunk_start} 09:15:00",
            "end_time":           f"{chunk_end} 15:30:00",
            "interval_in_minutes": _GROWW_DAILY_INTERVAL,
        }
        try:
            resp = session.get(
                f"{_GROWW_BASE_URL}/historical/candles",
                params=params,
                timeout=20,
            )
            if resp.status_code == 404:
                logger.debug("[Groww] Symbol %s not found (404), skipping.", groww_symbol)
                return []
            if resp.status_code == 429:
                logger.warning("[Groww] Rate limited for %s.", groww_symbol)
                return _RATE_LIMITED
            resp.raise_for_status()
            payload = resp.json()
            status = payload.get("status", "")
            if status != "SUCCESS":
                logger.debug("[Groww] Non-success status for %s: %s", groww_symbol, status)
                return []
            candles = payload.get("payload", {}).get("candles", [])
            return candles if isinstance(candles, list) else []
        except Exception as exc:
            logger.debug("[Groww] Fetch failed for %s (%s → %s): %s", groww_symbol, chunk_start, chunk_end, exc)
            return []

    def _fetch_full_history(self, groww_symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Fetch complete daily OHLCV history for groww_symbol over [start, end].

        FIX-MB-DC-02: Caps consecutive rate-limit retries at _MAX_RATE_LIMIT_RETRIES.
        When the cap is reached, logs a warning and aborts the fetch for this symbol
        rather than blocking indefinitely (e.g. when a token is revoked).
        """
        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end)

        all_candles: List[list] = []
        cursor = start_ts
        consecutive_rate_limits = 0

        while cursor <= end_ts:
            chunk_end = min(cursor + pd.Timedelta(days=_GROWW_CHUNK_DAYS), end_ts)
            result = self._fetch_candles_chunk(
                groww_symbol,
                cursor.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )

            if result is _RATE_LIMITED:
                consecutive_rate_limits += 1

                # FIX-MB-DC-02: abort after too many consecutive 429s to prevent
                # infinite blocking (e.g. revoked token, suspended account).
                if consecutive_rate_limits >= _MAX_RATE_LIMIT_RETRIES:
                    logger.warning(
                        "[Groww] %s: hit rate-limit %d times consecutively "
                        "(max %d). Aborting fetch — token may be revoked or "
                        "account suspended. Falling back to yfinance.",
                        groww_symbol, consecutive_rate_limits, _MAX_RATE_LIMIT_RETRIES,
                    )
                    return None

                backoff = min(5.0 * (2 ** (consecutive_rate_limits - 1)), 60.0)
                backoff += random.uniform(0, backoff * 0.2)
                logger.warning(
                    "[Groww] Rate limit hit #%d for %s — backing off %.0fs.",
                    consecutive_rate_limits, groww_symbol, backoff,
                )
                time.sleep(backoff)
                continue

            consecutive_rate_limits = 0
            all_candles.extend(result)
            cursor = chunk_end + pd.Timedelta(days=1)
            time.sleep(self._GROWW_SLEEP_SECS)

        if not all_candles:
            return None

        rows = []
        for c in all_candles:
            if not isinstance(c, list) or len(c) < 6:
                continue
            try:
                ts    = pd.Timestamp(str(c[0]))
                open_ = float(c[1])
                high  = float(c[2])
                low   = float(c[3])
                close = float(c[4])
                vol   = float(c[5]) if c[5] is not None else 0.0
                rows.append({
                    "Date":   ts.normalize(),
                    "Open":   open_,
                    "High":   high,
                    "Low":    low,
                    "Close":  close,
                    "Volume": vol,
                })
            except Exception:
                continue

        if not rows:
            return None

        df = pd.DataFrame(rows).set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    @staticmethod
    def _extract_batch_series(
        batch_df: pd.DataFrame,
        field: str,
        ticker: str,
    ) -> Optional[pd.Series]:
        if batch_df is None or batch_df.empty or field not in batch_df.columns:
            return None
        try:
            field_df = batch_df[field]
            if isinstance(field_df, pd.DataFrame):
                if ticker not in field_df.columns:
                    return None
                series = field_df[ticker].copy()
            else:
                series = field_df.copy()
        except Exception:
            return None

        if hasattr(series.index, "tz") and series.index.tz is not None:
            series.index = series.index.tz_convert(_IST_TZ).tz_localize(None)
        series.index = pd.DatetimeIndex(series.index).normalize()
        return pd.to_numeric(series, errors="coerce")

    def _build_adj_close_from_batches(
        self,
        raw_close: pd.Series,
        ns_ticker: str,
        yf_raw: pd.DataFrame,
    ) -> pd.Series:
        """
        Back-adjust raw Groww Close using split/dividend metadata from a single
        yfinance call (auto_adjust=False, actions=True).

        FIX-MB-GROWWADJ: Only ffill is applied when aligning ratio to Groww
        dates. bfill was removed because it propagates a future split ratio
        backward across holiday gaps (look-ahead contamination).
        """
        try:
            adj_yf     = self._extract_batch_series(yf_raw, COLUMN_ADJ_CLOSE, ns_ticker)
            raw_yf_cls = self._extract_batch_series(yf_raw, "Close",     ns_ticker)

            if adj_yf is None or raw_yf_cls is None:
                return raw_close

            common = adj_yf.index.intersection(raw_yf_cls.index)
            if common.empty:
                return raw_close

            ratio = (adj_yf.loc[common] / raw_yf_cls.loc[common]).dropna()
            if ratio.empty:
                return raw_close

            # FIX-MB-GROWWADJ: ffill only (no future look-ahead) for dates
            # AFTER the first known ratio bar.
            # BUG-FIX-FILLNA: replace fillna(1.0) with bfill limited to the
            # leading-NaN region only (dates BEFORE the first known ratio bar).
            # fillna(1.0) caused a price discontinuity: if YFinance ratio data
            # starts in 2018 but Groww data goes back to 2015, and a 10:1 split
            # happened in 2016, the post-2018 ratio is ~0.1.  fillna(1.0) gave
            # pre-2018 bars ratio=1.0, making those prices 10x higher than
            # post-2018 prices — a phantom discontinuity in the adjusted series.
            # Correct fix: extend the EARLIEST known ratio backward (bfill only
            # on the leading NaN region), which preserves proportionality across
            # the entire series.  This is safe because no future information is
            # used — we are filling BACKWARD from the oldest known data point.
            ratio_reindexed = ratio.reindex(raw_close.index)
            # FIX-BFILL-SCOPE: limit bfill strictly to the leading-NaN region
            # (dates before the first known ratio bar).  The previous
            # ratio_reindexed.ffill().bfill() applied bfill unconditionally
            # across the entire series, which would fill trailing NaN dates
            # (Groww bars after the last YF ratio bar) backward from a future
            # ratio — minor look-ahead contamination.  The correct approach:
            # 1. ffill() propagates each known ratio forward across internal
            #    gaps (holidays); ratio is stable between corporate actions.
            # 2. For the leading NaN region (before the first known ratio bar),
            #    extend the oldest known ratio backward — this is safe because
            #    we use the earliest available data point, not a future one.
            _first_valid_idx = ratio_reindexed.first_valid_index()
            if _first_valid_idx is None:
                # ratio is completely empty for this date range — return raw
                return raw_close
            ratio_aligned = ratio_reindexed.ffill()
            _leading_mask = ratio_aligned.index < _first_valid_idx
            if _leading_mask.any():
                ratio_aligned[_leading_mask] = ratio_aligned[~_leading_mask].iloc[0]
            if ratio_aligned.isna().any():
                # trailing NaN after last YF bar — fill with last known ratio
                ratio_aligned = ratio_aligned.ffill()
            if ratio_aligned.isna().any():
                # ratio is completely empty for this date range — return raw
                return raw_close
            adj_groww = raw_close * ratio_aligned
            return adj_groww

        except Exception as exc:
            logger.debug("[Groww] Adj Close build failed for %s: %s", ns_ticker, exc)
            return raw_close

    def _extract_actions_from_batches(
        self,
        index: pd.DatetimeIndex,
        ns_ticker: str,
        yf_raw: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        dividends = self._extract_batch_series(yf_raw, "Dividends", ns_ticker)
        splits = self._extract_batch_series(yf_raw, COLUMN_STOCK_SPLITS, ns_ticker)

        div_series = pd.Series(0.0, index=index, dtype=float)
        split_series = pd.Series(0.0, index=index, dtype=float)

        if dividends is not None and not dividends.empty:
            div_series = dividends.reindex(index).fillna(0.0).astype(float)
        if splits is not None and not splits.empty:
            split_series = splits.reindex(index).fillna(0.0).astype(float)

        return div_series, split_series

    def download(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        if not self.api_token:
            return None

        frames: Dict[str, pd.DataFrame] = {}

        valid_groww_tickers = [
            t for t in tickers
            if not any(t.startswith(p) for p in _GROWW_INDEX_PREFIXES)
        ]

        yf_start = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        yf_end   = (pd.Timestamp(end) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

        # Single yfinance call: auto_adjust=False, actions=True — both raw Close
        # and Adj Close come from the same snapshot, eliminating ratio divergence.
        yf_raw = pd.DataFrame()
        if valid_groww_tickers:
            try:
                yf_raw = _safe_yf_download(
                    valid_groww_tickers,
                    start=yf_start,
                    end=yf_end,
                    auto_adjust=False,
                    actions=True,
                    progress=False,
                    threads=True,
                )
            except Exception as exc:
                logger.warning("[Groww] YF batch fetch failed: %s", exc)
                yf_raw = pd.DataFrame()

        for ns_ticker in tickers:
            if any(ns_ticker.startswith(p) for p in _GROWW_INDEX_PREFIXES):
                continue

            groww_sym = self._to_groww_symbol(ns_ticker)
            if groww_sym is None:
                continue

            raw_df = self._fetch_full_history(groww_sym, start, end)
            if raw_df is None or raw_df.empty:
                continue

            adj_close = self._build_adj_close_from_batches(raw_df["Close"], ns_ticker, yf_raw)
            raw_df[COLUMN_ADJ_CLOSE] = adj_close

            dividends, splits = self._extract_actions_from_batches(raw_df.index, ns_ticker, yf_raw)
            raw_df["Dividends"] = dividends
            raw_df[COLUMN_STOCK_SPLITS] = splits

            frames[ns_ticker] = raw_df

        if not frames:
            return None

        combined = pd.concat(frames, axis=1)
        # Build the expected MultiIndex directly as (Ticker, Price) so downstream
        # extractors can access per-symbol frames via level-0 ticker keys.
        combined.columns = pd.MultiIndex.from_tuples(
            [(ticker, field) for ticker, field in combined.columns],
            names=["Ticker", "Price"],
        )
        combined = combined.sort_index(axis=1)
        return _normalize_history_index(combined)


# ─── YFinance Provider ────────────────────────────────────────────────────────

class YFinanceProvider(DataProvider):
    def download(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        result = self._download_batch(tickers, start, end)
        if result is None:
            return None

        if len(tickers) > 1 and result.empty:
            recovered = self._download_individual(tickers, start, end)
            if recovered is not None and not recovered.empty:
                return recovered

        if result.empty:
            return result
        if len(tickers) > 1 and not isinstance(result.columns, pd.MultiIndex):
            logger.warning(
                "[Cache] Batch yfinance response for %d tickers had flat columns; retrying individually.",
                len(tickers),
            )
            return self._download_individual(tickers, start, end)
        return result

    def _download_batch(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        try:
            return _safe_yf_download(
                tickers,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                actions=True,
            )
        except Exception as exc:
            logger.warning("[Cache] yfinance batch download failed for %d tickers: %s", len(tickers), exc)
            return None

    def _download_individual(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        frames: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            single = self._download_batch([ticker], start, end)
            if single is None or single.empty:
                continue

            frame = _extract_ticker_frame(single, ticker, is_single_request=True)
            if frame is None or frame.empty:
                continue
            frames[ticker] = frame

        if not frames:
            return None

        combined = pd.concat(frames, axis=1)
        combined.columns = pd.MultiIndex.from_tuples(
            [(col, ticker) for ticker, col in combined.columns],
            names=["Price", "Ticker"],
        )
        return _normalize_history_index(combined.swaplevel(0, 1, axis=1).sort_index(axis=1))


class SecondaryProvider(DataProvider):
    """AlphaVantage fallback provider with basic rate-limit throttling."""

    _URL = "https://www.alphavantage.co/query"

    def __init__(self) -> None:
        self.api_key      = os.getenv("FALLBACK_API_KEY", "").strip()
        self.min_interval = float(os.getenv("FALLBACK_MIN_INTERVAL_SEC", "12"))

    def download(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        if not self.api_key:
            # This path is retained for direct unit use of SecondaryProvider.
            # In production load_or_fetch, _build_provider_chain excludes this
            # provider when FALLBACK_API_KEY is absent, preventing repeated
            # per-chunk warnings.
            logger.debug("[Cache][Fallback] FALLBACK_API_KEY not set; skipping secondary provider.")
            return None

        last_call_ts = 0.0
        ticker_frames: Dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            av_symbol = self._map_symbol(ticker)
            elapsed = time.time() - last_call_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            try:
                frame = self._download_single(av_symbol, self.api_key, start=start, end=end)
                last_call_ts = time.time()
            except Exception as exc:
                logger.warning("[Cache][Fallback] %s fetch failed: %s", ticker, exc)
                frame = None

            if frame is None:
                continue

            ticker_frames[ticker] = frame

        if not ticker_frames:
            return None

        # BUG-FIX-MULTIINDEX: The previous bypass returned a flat DataFrame
        # when exactly one ticker succeeded from a multi-symbol chunk.
        # _extract_ticker_frame received is_single_request=False (because the
        # original chunk had >1 symbol), expected a MultiIndex, found none,
        # and returned None — silently discarding the only recovered data.
        #
        # Fix: use pd.concat when the chunk requested multiple tickers,
        # ensuring a MultiIndex even if only one ticker succeeded.  When the
        # chunk is genuinely single-ticker (len(tickers)==1), return flat so
        # the caller's is_single_request=True path works correctly.
        if len(tickers) == 1:
            only = next(iter(ticker_frames.values()))
            return _normalize_history_index(only)

        combined = pd.concat(ticker_frames, axis=1)
        return _normalize_history_index(combined)

    def _download_single(self, symbol: str, api_key: str, start: str, end: str) -> Optional[pd.DataFrame]:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol":   symbol,
            "outputsize": "full",
            "apikey":   api_key,
        }
        response = requests.get(self._URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()

        if "Note" in payload or "Information" in payload:
            msg = payload.get("Note") or payload.get("Information")
            logger.warning("[Cache][Fallback] Rate-limit/notice for %s: %s", symbol, msg)
            return None

        series = payload.get("Time Series (Daily)")
        if not isinstance(series, dict) or not series:
            err = payload.get("Error Message", "empty time series")
            logger.warning("[Cache][Fallback] Invalid payload for %s: %s", symbol, err)
            return None

        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end)
        rows = []
        for d, item in series.items():
            date = pd.Timestamp(d)
            if date < start_ts or date >= end_ts:
                continue
            rows.append({
                "Date":      date,
                "Open":      float(item.get("1. open",             np.nan)),
                "High":      float(item.get("2. high",             np.nan)),
                "Low":       float(item.get("3. low",              np.nan)),
                "Close":     float(item.get("4. close",            np.nan)),
                COLUMN_ADJ_CLOSE: float(item.get("5. adjusted close",   np.nan)),
                "Volume":    float(item.get("6. volume",           np.nan)),
            })

        if not rows:
            return None

        df = pd.DataFrame(rows).set_index("Date").sort_index()
        return _ensure_price_columns(_normalize_history_index(df))

    @staticmethod
    def _map_symbol(ticker: str) -> str:
        if ticker.startswith("^"):
            return ticker
        bare = ticker[:-3] if ticker.endswith(".NS") else ticker
        return f"{bare}.NSE"


# ─── Index normalization ──────────────────────────────────────────────────────

def _normalize_history_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index, utc=True)
        except Exception:
            pass

    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is not None:
            out.index = out.index.tz_convert(_IST_TZ).tz_localize(None)
        out.index = out.index.normalize()

        if out.index.duplicated().any():
            out = out[~out.index.duplicated(keep="last")]

        out = out.sort_index()

    return out


def _extract_ticker_frame(
    raw_data: pd.DataFrame,
    ticker: str,
    *,
    is_single_request: bool = False,
) -> Optional[pd.DataFrame]:
    if raw_data is None or raw_data.empty:
        return None

    if isinstance(raw_data.columns, pd.MultiIndex):
        valid_mask = [
            (a is not None and b is not None)
            for a, b in raw_data.columns
        ]
        if not all(valid_mask):
            raw_data = raw_data.loc[:, valid_mask].copy()
        if raw_data.empty:
            return None

        level0 = set(raw_data.columns.get_level_values(0))
        level1 = set(raw_data.columns.get_level_values(1))

        if ticker in level0:
            df = raw_data[ticker].copy()
        elif ticker in level1:
            df = raw_data.xs(ticker, level=1, axis=1).copy()
        else:
            return None

        df = df.loc[:, df.columns.notna()]
        if df.empty:
            return None
        return _ensure_price_columns(_normalize_history_index(df))

    if is_single_request:
        return _ensure_price_columns(_normalize_history_index(raw_data.copy()))
    return None


def _download_with_timeout(
    tickers: List[str],
    start: str,
    end: str,
    provider: Optional[DataProvider] = None,
) -> Optional[pd.DataFrame]:
    max_retries = 3
    errors: list = []
    for attempt in range(max_retries):
        try:
            active_provider = provider or YFinanceProvider()
            df = active_provider.download(tickers, start, end)
            return df
        except Exception as exc:
            errors.append(exc)
            logger.debug("[Cache] Download attempt %d failed: %s", attempt + 1, exc)
            if attempt == max_retries - 1:
                logger.error("[Cache] Provider failed after %d retries.", max_retries, exc_info=True)
                if len(errors) > 1:
                    raise errors[-1] from errors[0]
                raise errors[-1]
            time.sleep((2 ** attempt) + random.random())

    return None


def _load_manifest() -> dict:
    _ensure_cache_paths_configured()
    default_manifest = {"schema_version": 1, "entries": {}}
    assert MANIFEST_FILE is not None
    if not MANIFEST_FILE.exists():
        return default_manifest

    try:
        with MANIFEST_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
            if not isinstance(data, dict):
                logger.warning("[Cache] Manifest root must be a JSON object; starting fresh.")
                return default_manifest
            if "schema_version" in data:
                return data
            else:
                return {"schema_version": 1, "entries": data}
    except Exception as exc:
        logger.warning("[Cache] Manifest corrupted or unreadable. Starting fresh. Error: %s", exc)
        return default_manifest


def _save_manifest(manifest_data: dict) -> None:
    _ensure_cache_paths_configured()
    assert CACHE_DIR is not None and MANIFEST_FILE is not None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = MANIFEST_FILE.with_name(MANIFEST_FILE.name + ".tmp")
    try:
        with temp_file.open("w", encoding="utf-8") as file:
            json.dump(manifest_data, file, indent=2)
            # FIX-BUG-12: flush userspace buffer then fsync to kernel before the
            # atomic rename.  Without this a crash between write() and rename()
            # can leave a zero-byte or partially-written manifest.  Matches the
            # hardened pattern already used in save_portfolio_state().
            file.flush()
            os.fsync(file.fileno())
        temp_file.replace(MANIFEST_FILE)
    except Exception as exc:
        logger.error("[Cache] Failed to save manifest: %s", exc, exc_info=True)


def invalidate_cache() -> None:
    _ensure_cache_paths_configured()
    assert MANIFEST_FILE is not None
    if MANIFEST_FILE.exists():
        try:
            MANIFEST_FILE.unlink()
            logger.info("[Cache] Market data cache invalidated.")
        except OSError as e:
            logger.error("[Cache] Failed to invalidate cache: %s", e, exc_info=True)


def _is_valid_dataframe(df: pd.DataFrame, ticker: Optional[str] = None, cfg=None) -> bool:
    if cfg is not None:
        history_gate_raw = getattr(cfg, "HISTORY_GATE", None)
        min_rows = int(history_gate_raw) if history_gate_raw is not None else 5
    else:
        min_rows = 5
    if df is None or df.empty or len(df) < min_rows:
        return False
    if not isinstance(df.index, pd.DatetimeIndex):
        return False
    if not df.index.is_unique:
        return False
    if not df.index.is_monotonic_increasing:
        return False

    if "Close" not in df.columns or df["Close"].isnull().all():
        return False

    if COLUMN_ADJ_CLOSE not in df.columns or df[COLUMN_ADJ_CLOSE].isnull().all():
        return False

    is_index_ticker = bool(ticker) and str(ticker).startswith("^")
    if (not is_index_ticker) and ("Volume" not in df.columns or df["Volume"].isnull().all()):
        return False

    return True


def _ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[:, out.columns.notna()]

    numeric_cols = [
        "Open", "High", "Low", "Close", COLUMN_ADJ_CLOSE,
        "Volume", "Dividends", COLUMN_STOCK_SPLITS,
    ]

    def _coerce_numeric_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series
        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("INR", "", regex=False)
            .str.strip()
        )
        cleaned = cleaned.mask(cleaned.isin(["", "nan", "None"]))
        return pd.to_numeric(cleaned, errors="coerce")

    for col in numeric_cols:
        if col in out.columns:
            out[col] = _coerce_numeric_series(out[col])

    if COLUMN_ADJ_CLOSE not in out.columns:
        if "Close" in out.columns:
            out[COLUMN_ADJ_CLOSE] = out["Close"]
    else:
        if "Close" in out.columns:
            out[COLUMN_ADJ_CLOSE] = out[COLUMN_ADJ_CLOSE].fillna(out["Close"])

    for col in ["Dividends", COLUMN_STOCK_SPLITS]:
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(0.0)

    return out


def _latest_business_day() -> str:
    """
    Return the most recent Mon–Fri business day before today as 'YYYY-MM-DD'.

    Uses pandas_market_calendars 'NSE' when available to respect exchange
    holidays (not just Mon–Fri weekdays). Falls back to pandas BDay logic on
    any calendar import/runtime failure.
    """
    today = pd.Timestamp.now(tz=_IST_TZ).normalize()
    try:
        import pandas_market_calendars as mcal

        nse_calendar = mcal.get_calendar("NSE")
        valid_days = nse_calendar.valid_days(
            start_date=today - pd.Timedelta(days=366),
            end_date=today - pd.Timedelta(days=1),
        )
        if len(valid_days) > 0:
            last_valid = pd.Timestamp(valid_days[-1])
            if last_valid.tzinfo is None:
                last_valid = last_valid.tz_localize(_IST_TZ)
            else:
                last_valid = last_valid.tz_convert(_IST_TZ)
            return last_valid.strftime("%Y-%m-%d")
    except Exception as exc:
        logger.debug("Falling back to BDay due to NSE calendar error: %s", exc, exc_info=True)

    latest_bday_ts = today - pd.offsets.BDay(1)
    return latest_bday_ts.strftime("%Y-%m-%d")


def _build_provider_chain(cfg=None) -> List[DataProvider]:
    chain: List[DataProvider] = []

    groww_token = os.getenv("GROWW_API_TOKEN", "").strip()
    if groww_token:
        chain.append(GrowwProvider(api_token=groww_token))
        logger.info("[Cache] GrowwProvider enabled as primary data source.")
    else:
        logger.info("[Cache] GROWW_API_TOKEN not set — YFinanceProvider is primary.")

    chain.append(YFinanceProvider())

    provider = SecondaryProvider()
    if provider.api_key:
        chain.append(provider)
        logger.info("[Cache] SecondaryProvider enabled as tertiary fallback.")
    else:
        logger.info("[Cache] SecondaryProvider disabled (FALLBACK_API_KEY not set).")

    return chain


def load_or_fetch(
    tickers: List[str],
    required_start: str,
    required_end: str,
    force_refresh: bool = False,
    cfg=None,
) -> Dict[str, pd.DataFrame]:
    _ensure_cache_paths_configured()
    assert CACHE_DIR is not None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    entries  = manifest["entries"]

    def _clean_ticker(t: str) -> str:
        if t is None:
            raise ValueError("ticker list contains None")
        t_str = str(t).strip()
        if not t_str:
            raise ValueError("ticker list contains empty string")
        if t_str.startswith("^"):
            return t_str
        upper_t = t_str.upper()
        # Strip only terminal exchange suffixes. Order matters: .NSE must be
        # checked before .NS so ".NSE" is removed as one suffix (not partial).
        if upper_t.endswith(".NSE"):
            upper_t = upper_t[:-4]
        elif upper_t.endswith(".NS"):
            upper_t = upper_t[:-3]
        return upper_t + ".NS"

    standardized_tickers = list(dict.fromkeys(_clean_ticker(t) for t in tickers))

    if not required_start:
        raise ValueError("required_start must be provided")
    if not required_end:
        raise ValueError("required_end must be provided")

    cfg_lookback = int(getattr(cfg, "CVAR_LOOKBACK", 200) or 200)
    dynamic_padding_days = max(400, cfg_lookback * 2)
    padded_start = (
        pd.Timestamp(required_start) - timedelta(days=dynamic_padding_days)
    ).strftime("%Y-%m-%d")
    yf_end = (pd.Timestamp(required_end) + timedelta(days=1)).strftime("%Y-%m-%d")

    latest_bday = _latest_business_day()

    tickers_to_download = []
    market_data: Dict[str, pd.DataFrame] = {}

    for ticker in standardized_tickers:
        entry        = entries.get(ticker, {})
        parquet_path = CACHE_DIR / f"{ticker}.parquet"

        is_stale     = entry.get("last_date", "") < latest_bday
        missing_file = not parquet_path.exists()
        # BUG-FIX-STALE-RETRY: honour retry_after cooldown set by
        # _recover_from_stale_cache.  Without this, a symbol whose providers
        # all failed would be re-downloaded (and fail again) on every single
        # run, causing infinite hammer-and-recover cycles.
        retry_after  = entry.get("retry_after", "")
        in_cooldown  = bool(retry_after) and latest_bday < retry_after

        if force_refresh or (is_stale and not in_cooldown) or missing_file:
            tickers_to_download.append(ticker)
        else:
            try:
                df = _normalize_history_index(pd.read_parquet(parquet_path))
                market_data[ticker] = df
            except Exception as exc:
                logger.warning("[Cache] Corrupted parquet for %s: %s", ticker, exc)
                tickers_to_download.append(ticker)

    if not tickers_to_download:
        logger.debug(
            "[Cache] All %d requested symbols served from cache (no downloads needed).",
            len(standardized_tickers),
        )

    providers = _build_provider_chain(cfg)
    try:
        def _retry_unresolved_individually(unresolved: List[str]) -> List[str]:
            """
            Retry unresolved symbols one-by-one through the full provider chain.

            A chunk-level provider failure can drop many otherwise valid symbols
            at once (for example when one problematic symbol poisons a batched
            response). This best-effort pass limits blast radius by retrying
            each unresolved symbol in isolation before stale-cache recovery.
            Returns the still-missing tickers after individual retries.
            """
            still_missing: List[str] = []
            if not unresolved:
                return still_missing

            for ticker in unresolved:
                resolved = False
                for provider in providers:
                    try:
                        raw_single = _download_with_timeout(
                            [ticker], padded_start, yf_end, provider=provider
                        )
                    except Exception as exc:
                        logger.warning(
                            "[Cache] Provider %s failed on individual retry for %s: %s",
                            type(provider).__name__,
                            ticker,
                            exc,
                        )
                        raw_single = None

                    if raw_single is None or raw_single.empty:
                        continue

                    saved = _process_chunk(
                        [ticker],
                        raw_single,
                        entries,
                        market_data,
                        cfg,
                        provider_name=type(provider).__name__,
                    )
                    if ticker in saved:
                        resolved = True
                        break

                if not resolved:
                    still_missing.append(ticker)
            return still_missing

        if tickers_to_download:
            logger.info("[Cache] Initiating download for %d missing/stale symbols.", len(tickers_to_download))
            chunks = [
                tickers_to_download[i:i + _DOWNLOAD_CHUNK_SIZE]
                for i in range(0, len(tickers_to_download), _DOWNLOAD_CHUNK_SIZE)
            ]

            for chunk in chunks:
                raw_data = None
                for provider in providers:
                    try:
                        raw_data = _download_with_timeout(chunk, padded_start, yf_end, provider=provider)
                    except Exception as exc:
                        logger.warning(
                            "[Cache] Provider %s failed for chunk starting with %s: %s",
                            type(provider).__name__, chunk[0], exc,
                        )
                        raw_data = None

                    if raw_data is not None and not raw_data.empty:
                        # BUG-FIX-FALLBACK: use validated-and-saved tickers from
                        # _process_chunk, not raw column presence.  A provider that
                        # returns a column full of NaN values causes that ticker to
                        # appear in raw column names but fail _is_valid_dataframe
                        # and never be saved.  Using column names would mark it as
                        # "served" and block the SecondaryProvider fallback.
                        saved = _process_chunk(
                            chunk,
                            raw_data,
                            entries,
                            market_data,
                            cfg,
                            provider_name=type(provider).__name__,
                        )
                        missing_from_provider = [t for t in chunk if t not in saved]

                        if not missing_from_provider:
                            break

                        chunk = missing_from_provider
                        raw_data = None
                        continue

                # FIX-BUG-8: guard recovery on whether `chunk` still has unresolved
                # tickers, not on the state of `raw_data`.  The old check
                #   `if raw_data is None or raw_data.empty`
                # missed the case where the last provider returned a non-empty
                # DataFrame but every ticker in it failed _is_valid_dataframe:
                # raw_data was truthy so _recover_from_stale_cache was never called,
                # silently dropping those tickers with no fallback and no WARNING.
                # After the provider loop `chunk` always reflects the residual set of
                # unresolved tickers (reassigned to missing_from_provider each pass),
                # so a non-empty chunk is the correct and complete recovery signal.
                if chunk:
                    unresolved_after_retry = _retry_unresolved_individually(chunk)
                    if unresolved_after_retry:
                        _recover_from_stale_cache(unresolved_after_retry, entries, market_data, cfg=cfg)

            # BUG-FIX-MANIFEST-IO: save manifest once after ALL chunks complete,
            # not once per chunk. Previously 300 tickers (4 chunks) triggered 4
            # full JSON serialisations and atomic renames. One write is sufficient
            # because the manifest is only read at startup and the per-chunk
            # entries have already been mutated in-place in `manifest`.
            # Note: _save_manifest is intentionally called only on download/recovery
            # paths. Any future code path that mutates `entries` without downloading
            # must explicitly call _save_manifest() to keep manifest state in sync.
            assert _MANIFEST_LOCK_DIR is not None
            with _ManifestProcessFileLock(_MANIFEST_LOCK_DIR):
                live_manifest = _load_manifest()
                live_entries = live_manifest.setdefault("entries", {})
                for ticker in tickers_to_download:
                    entry = entries.get(ticker)
                    if entry is not None:
                        live_entries[ticker] = entry
                _save_manifest(live_manifest)
    finally:
        for provider in providers:
            close_fn = getattr(provider, "close", None)
            if callable(close_fn):
                close_fn()

    # FIX-BUG-10: only clip data at required_end, not at a computed start bound.
    # The old logic computed effective_start = min(df.index[0], padded_start_ts):
    # when cached data predates padded_start (df.index[0] < padded_start_ts),
    # effective_start correctly equals df.index[0] and the slice is harmless.
    # But when df.index[0] > padded_start_ts (the cache doesn't go back far enough),
    # effective_start = padded_start_ts and df.loc[padded_start_ts:required_end]
    # returns an *empty* leading slice because padded_start_ts predates the data —
    # silently discarding all available history.  The intent is purely to drop
    # future rows beyond required_end, so trim only the end.
    for t, df in list(market_data.items()):
        if df.empty:
            continue
        market_data[t] = df.loc[:pd.Timestamp(required_end)]

    return market_data


def _process_chunk(
    chunk: List[str],
    raw_data: pd.DataFrame,
    entries: dict,
    market_data: Dict[str, pd.DataFrame],
    cfg,
    *,
    provider_name: str = "unknown",
) -> set:
    """Process a downloaded chunk and return the set of successfully saved tickers.

    BUG-FIX-FALLBACK: Previously returned None, so the caller inferred served
    tickers from raw column names — which includes NaN-only columns that pass
    the column-presence check but fail _is_valid_dataframe.  Those tickers were
    added to `served`, blocking the SecondaryProvider fallback even though no
    valid data was actually saved.  Returning the validated set here lets the
    caller use ground-truth instead of inferred membership.
    """
    _ensure_cache_paths_configured()
    assert CACHE_DIR is not None
    manifest_entries = entries
    saved_tickers: set = set()

    for ticker in chunk:
        try:
            df = _extract_ticker_frame(
                raw_data,
                ticker,
                is_single_request=(len(chunk) == 1),
            )
            if df is None or df.empty:
                # FIX-NEW-DC-02: log at WARNING rather than silently skipping so
                # callers can distinguish a provider returning no data for a ticker
                # (possible fetch failure) from a ticker genuinely absent in the
                # response.  Silent skips make it impossible to diagnose partial
                # provider outages or symbol delisting from logs alone.
                logger.warning(
                    "[Cache] No usable data extracted for %s "
                    "(provider=%s; check symbol name, market hours, or provider status).",
                    ticker,
                    provider_name,
                )
                continue

            df.dropna(how="all", inplace=True)
            if df.empty:
                continue

            if not _is_valid_dataframe(df, ticker=ticker, cfg=cfg):
                logger.warning(
                    "[Cache] Structural validation failed for %s "
                    "(non-monotonic index, duplicate dates, or null OHLC). Skipping.",
                    ticker,
                )
                continue

            parquet_path = CACHE_DIR / f"{ticker}.parquet"
            # Use unique temp file per writer to avoid race conditions
            tmp_path = CACHE_DIR / f"{ticker}.parquet.tmp.{uuid.uuid4().hex}"
            try:
                df.to_parquet(tmp_path)
                # Atomically replace the final parquet file
                tmp_path.replace(parquet_path)
            except Exception:
                # Clean up temp file on failure
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

            gap_series   = df.index.to_series().diff().dt.days
            max_gap      = gap_series.max()
            max_gap_days = int(max_gap) if pd.notna(max_gap) else 0

            manifest_entries[ticker] = {
                "fetched_at":    pd.Timestamp.now(tz=_IST_TZ).isoformat(),
                "rows":          len(df),
                "last_date":     df.index[-1].strftime("%Y-%m-%d"),
                "suspended":     max_gap_days > _SUSPENSION_GAP_DAYS,
                "max_gap_days":  max_gap_days,
            }
            market_data[ticker] = df
            saved_tickers.add(ticker)

        except Exception as exc:
            logger.error("[Cache] Failed processing downloaded dataframe for %s: %s", ticker, exc, exc_info=True)

    return saved_tickers


def _recover_from_stale_cache(
    chunk: List[str],
    entries: dict,
    market_data: Dict[str, pd.DataFrame],
    cfg=None,
) -> None:
    """
    Load stale-but-present parquets as a fallback when all providers fail.

    FIX-MB-DC-01: Update the manifest entry for each successfully recovered
    symbol so that subsequent load_or_fetch calls see a valid (if stale) entry
    and do not repeatedly trigger download attempts. The last_date is preserved
    from the parquet itself rather than being set to today, so the entry
    accurately reflects that the data is stale and will be refreshed when the
    provider recovers.
    """
    _ensure_cache_paths_configured()
    assert CACHE_DIR is not None
    recovered = 0
    recovered_symbols: List[str] = []
    for ticker in chunk:
        parquet_path = CACHE_DIR / f"{ticker}.parquet"
        if not parquet_path.exists():
            continue
        try:
            fallback_df = _normalize_history_index(pd.read_parquet(parquet_path))
            if fallback_df is None or fallback_df.empty:
                continue
            # Recovery must enforce the same structural/row-count thresholds as
            # the primary download path; otherwise very short stale frames can
            # sneak in and poison downstream signals.
            if not _is_valid_dataframe(fallback_df, ticker=ticker, cfg=cfg):
                continue
            market_data[ticker] = fallback_df
            recovered += 1
            recovered_symbols.append(ticker)

            # FIX-MB-DC-01: update manifest with the actual last_date from the
            # parquet so subsequent calls don't re-trigger downloads on every run.
            existing = entries.get(ticker, {})
            # BUG-FIX-STALE-RETRY: set retry_after to tomorrow so the
            # staleness check does not re-trigger a download on every run.
            # Previously last_date was preserved as the parquet's actual last
            # date, which is always < latest_bday, so is_stale was always True
            # and every subsequent run would hammer all providers, fail, and
            # fall back to the stale cache again — infinite retry loop.
            # retry_after acts as a cooldown: we do not attempt a re-download
            # until at least the next business day.  last_date is preserved
            # accurately so callers know the true data freshness.
            _tomorrow = (pd.Timestamp.now(tz=_IST_TZ) + timedelta(days=1)).strftime("%Y-%m-%d")
            entries[ticker] = {
                "fetched_at":   existing.get("fetched_at", pd.Timestamp.now(tz=_IST_TZ).isoformat()),
                "rows":         len(fallback_df),
                "last_date":    fallback_df.index[-1].strftime("%Y-%m-%d"),
                "suspended":    existing.get("suspended", False),
                "max_gap_days": existing.get("max_gap_days", 0),
                "stale_recovery": True,
                "retry_after":  _tomorrow,
            }

            # Keep per-symbol details at debug level to avoid flooding stdout
            # during large universe runs.
            logger.debug(
                "[Cache] Using stale cached parquet for %s after all providers failed "
                "(last_date=%s).",
                ticker, entries[ticker]["last_date"],
            )
        except Exception as exc:
            logger.warning(
                "[Cache] Failed loading stale parquet fallback for %s: %s", ticker, exc,
            )

    missing = len(chunk) - recovered
    if recovered > 0:
        sample = ", ".join(recovered_symbols[:5])
        suffix = " ..." if recovered > 5 else ""
        logger.warning(
            "[Cache] Recovered %d symbol(s) from stale local cache after provider failures "
            "(sample: %s%s).",
            recovered,
            sample,
            suffix,
        )
    if missing > 0:
        logger.error(
            "[Cache] Skipping %d symbols from chunk starting with %s after all providers failed.",
            missing, chunk[0],
        )


def get_cache_summary() -> dict:
    manifest = _load_manifest()
    entries  = manifest.get("entries", {})
    groww_token = os.getenv("GROWW_API_TOKEN", "").strip()
    return {
        "total_symbols":     len(entries),
        "schema_version":    manifest.get("schema_version", 1),
        "suspended_symbols": sum(1 for v in entries.values() if v.get("suspended")),
        "stale_recovered":   sum(1 for v in entries.values() if v.get("stale_recovery")),
        "groww_enabled":     bool(groww_token),
    }
