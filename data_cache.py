"""
data_cache.py — Persistent Atomic Downloader
=============================================
Parquet-backed cache with atomic writes, manifest tracking,
and ThreadPoolExecutor isolation for multiprocessing safety on all platforms.

ASM/GSM handling (FIX G1)
--------------------------
NSE stocks placed under Additional or Graded Surveillance Measures (ASM/GSM)
frequently move to periodic call auctions, creating 30–90 day gaps in their
daily OHLCV series. The original code hard-rejected any file with a gap
exceeding 30 days, permanently excluding a stock even after its suspension
lifted. The corrected logic forward-fills the gap with the last-traded price
and records a `suspended` flag plus `max_gap_days` in the manifest, so
downstream consumers can inspect suspension history without losing the data.

Chunked downloading (FIX D1)
-----------------------------
A single yf.download() call for 500+ tickers reliably exceeds any reasonable
wall-clock timeout (observed: 120 s exhausted with 0 tickers returned).
load_or_fetch now splits `to_download` into chunks of _DOWNLOAD_CHUNK_SIZE
tickers, each with its own per-chunk timeout (_CHUNK_TIMEOUT_S). The manifest
is committed after each successful chunk so a mid-run interruption preserves
all already-downloaded data. A short inter-chunk sleep avoids Yahoo Finance
rate-limiting.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

logger         = logging.getLogger(__name__)

# FIX (Bug #1): Constants defined BEFORE the yf TZ-cache block that references them.
CACHE_DIR      = "data/cache"
MANIFEST_FILE  = os.path.join(CACHE_DIR, "_manifest.json")
SCHEMA_VERSION = 1

# ── Chunked-download tunables ─────────────────────────────────────────────────
# Empirically, yfinance handles ~50-100 NSE tickers reliably within 90 s.
# Larger batches cause Yahoo's connection pool to stall mid-transfer.
_DOWNLOAD_CHUNK_SIZE  = 75    # tickers per yf.download() call
_CHUNK_TIMEOUT_S      = 90.0  # per-chunk wall-clock timeout (seconds)
_INTER_CHUNK_SLEEP_S  = 2.0   # polite pause between chunks

# Calendar-day gap threshold above which a stock is flagged as likely suspended.
_SUSPENSION_GAP_DAYS = 30

# FIX (Bug #1 continued): CACHE_DIR is now defined above, so this block works.
try:
    _YF_TZ_CACHE = os.path.join(CACHE_DIR, "_yf_tz_cache")
    os.makedirs(_YF_TZ_CACHE, exist_ok=True)
    yf.set_tz_cache_location(_YF_TZ_CACHE)
except Exception:
    pass  # Non-fatal: yfinance will fall back to in-memory TZ lookup.


# ─── Worker ───────────────────────────────────────────────────────────────────

def _yf_fetch_worker(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Downloads OHLCV data via yfinance with exponential-backoff retry on
    transient network failures. Raises on the third consecutive failure.

    Note on 'database is locked': yfinance's SQLite TZ cache can raise this
    when a previous run crashed mid-write. Pointing the cache at our own
    directory (see _YF_TZ_CACHE above) isolates the file; if the lock persists
    across reboots, delete data/cache/_yf_tz_cache/ and retry.
    """
    for attempt in range(3):
        try:
            return yf.download(
                tickers, start=start, end=end,
                group_by="ticker", progress=False, auto_adjust=True,
            )
        except Exception as exc:
            if attempt == 2:
                raise exc
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.debug("[Cache] Download attempt %d failed (%s); retrying in %.1fs.",
                         attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError("_yf_fetch_worker: exhausted all 3 retries without returning or raising.")


# ─── Manifest helpers ─────────────────────────────────────────────────────────

def _load_manifest() -> dict:
    """Loads the manifest with backward compatibility for flat legacy structures."""
    if not os.path.exists(MANIFEST_FILE):
        return {"schema_version": SCHEMA_VERSION, "entries": {}}
    try:
        with open(MANIFEST_FILE) as f:
            m = json.load(f)
            if "schema_version" not in m:
                return {"schema_version": SCHEMA_VERSION, "entries": m}
            return m
    except Exception:
        return {"schema_version": SCHEMA_VERSION, "entries": {}}


def _save_manifest(m: dict) -> None:
    """Atomic write for the JSON manifest using a temp-file swap."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = MANIFEST_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=2)
    os.replace(tmp, MANIFEST_FILE)


# ─── Data Validation Gate ─────────────────────────────────────────────────────

def _is_valid_dataframe(df: Optional[pd.DataFrame]) -> bool:
    """
    Validation gate blocking ingestion of structurally corrupt OHLCV data.

    FIX G1: The 30-day gap check has been removed. Trading gaps caused by
    NSE ASM/GSM surveillance suspensions are handled in _ingest_raw via
    forward-fill rather than hard-rejection. Only structural corruption
    (bad index, missing Close, zero variance) is rejected here.
    """
    if df is None or df.empty or len(df) < 5:
        return False
    if not (df.index.is_unique and df.index.is_monotonic_increasing):
        return False
    if "Close" not in df.columns or df["Close"].isnull().all():
        return False
    if df["Close"].nunique() <= 1:
        return False
    return True


# ─── Suspension Detection & Repair ────────────────────────────────────────────

def _repair_suspension_gaps(
    df: pd.DataFrame,
    ticker: str,
) -> Tuple[pd.DataFrame, bool, int]:
    """
    Detects trading gaps longer than _SUSPENSION_GAP_DAYS and forward-fills them.

    Returns
    -------
    df          : Repaired DataFrame; gaps are filled with last-traded price.
    suspended   : True if any gap exceeding the threshold was detected.
    max_gap     : Largest calendar-day gap observed (0 if index has < 2 rows).

    Note: pd.bdate_range uses a fixed Mon–Fri calendar. NSE has additional
    holidays that yfinance elides from raw data, so a small number of synthetic
    rows may land on Indian market holidays. This is benign for signal math but
    inflates the `rows` count in the manifest slightly.
    """
    if len(df) < 2:
        return df, False, 0

    gap_days  = df.index.to_series().diff().dt.days
    max_gap   = int(gap_days.max()) if not gap_days.empty else 0
    suspended = max_gap > _SUSPENSION_GAP_DAYS

    if suspended:
        logger.warning(
            "[Cache] %s: trading gap of %d calendar days detected "
            "(likely ASM/GSM suspension). Forward-filling to preserve series.",
            ticker, max_gap,
        )
        bday_idx = pd.bdate_range(df.index[0], df.index[-1])
        df = df.reindex(bday_idx).ffill()

    return df, suspended, max_gap


# ─── Downloader Implementation ────────────────────────────────────────────────

def _download_with_timeout(
    tickers: List[str], start: str, end: str, timeout: float
) -> pd.DataFrame:
    """
    Executes the yfinance download in a background thread with a hard timeout.

    Uses ThreadPoolExecutor rather than ProcessPoolExecutor: yfinance releases
    the GIL during network I/O so threading is efficient, and it avoids the
    Windows multiprocessing 'spawn' requirement that causes recursive import
    errors when called outside a __main__ guard.
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_yf_fetch_worker, tickers, start, end)
            return future.result(timeout=timeout)
    except TimeoutError:
        logger.error(
            "[Cache] Thread pool fetch timed out after %.1fs for %d tickers. "
            "Thread may still be running in background.",
            timeout, len(tickers),
        )
        return pd.DataFrame()
    except Exception as exc:
        logger.error("[Cache] Thread pool fetch failed: %s", exc)
        return pd.DataFrame()


def _extract_ticker_df(
    raw: pd.DataFrame, ticker: str, is_multi: bool
) -> Optional[pd.DataFrame]:
    """Extracts a single ticker's OHLCV DataFrame from a yf.download multi-index result."""
    if not is_multi:
        return raw.copy()
    try:
        levels = raw.columns.get_level_values
        if ticker in levels(0):
            return raw[ticker].copy()
        if ticker in levels(1):
            return raw.xs(ticker, axis=1, level=1).copy()
    except Exception as exc:
        logger.debug("[Cache] Extraction error for %s: %s", ticker, exc)
    return None


def _ingest_raw(raw: pd.DataFrame, to_download: List[str], fetch_start: str) -> dict:
    """
    Validates each ticker's data and writes individual parquets atomically.

    FIX G1: Gaps > _SUSPENSION_GAP_DAYS are forward-filled (not rejected) and
    the manifest entry is annotated with `suspended: bool` and `max_gap_days: int`
    so callers can inspect suspension history without data loss.

    Returns a sub-manifest of successfully written tickers (staged commit).
    """
    sub_manifest = {}
    if raw is None or raw.empty:
        return sub_manifest

    is_multi = isinstance(raw.columns, pd.MultiIndex)
    now      = datetime.now().isoformat()

    for t in to_download:
        try:
            df = _extract_ticker_df(raw, t, is_multi)

            if not _is_valid_dataframe(df):
                logger.debug("[Cache] Validation failed for %s.", t)
                continue

            if getattr(df.index, "tz", None):
                df.index = df.index.tz_convert(None)
            df = df.dropna(how="all")

            # FIX G1: Repair ASM/GSM suspension gaps before writing to disk.
            df, suspended, max_gap = _repair_suspension_gaps(df, t)

            path     = os.path.join(CACHE_DIR, f"{t}.parquet")
            tmp_path = path + ".tmp"
            df.to_parquet(tmp_path)
            os.replace(tmp_path, path)

            sub_manifest[t] = {
                "fetched_at":    now,
                "rows":          len(df),
                "first_date":    df.index[0].strftime("%Y-%m-%d"),
                "last_date":     df.index[-1].strftime("%Y-%m-%d"),
                "covered_start": fetch_start,
                "suspended":     suspended,
                "max_gap_days":  max_gap,
            }
        except Exception as exc:
            logger.warning("[Cache] Failed to process %s: %s", t, exc)

    return sub_manifest


# ─── Chunked batch downloader ─────────────────────────────────────────────────

def _download_chunk(
    chunk: List[str],
    fetch_start: str,
    required_end: str,
    chunk_timeout: float,
    adv_timeout: float,
    chunk_idx: int,
    total_chunks: int,
) -> dict:
    """
    Download and ingest a single chunk of tickers. Returns a sub-manifest
    dict for the tickers that succeeded. Performs one surgical retry pass
    for any missed tickers in the chunk before returning.
    """
    logger.info(
        "[Cache] Chunk %d/%d: downloading %d tickers ...",
        chunk_idx, total_chunks, len(chunk),
    )
    raw = _download_with_timeout(chunk, fetch_start, required_end, chunk_timeout)
    sub = _ingest_raw(raw, chunk, fetch_start)

    missed = [t for t in chunk if t not in sub]
    if missed and len(missed) <= 10:
        logger.info("[Cache] Chunk %d/%d: surgical retry for %d missed ticker(s).",
                    chunk_idx, total_chunks, len(missed))
        for mt in missed:
            raw_single = _download_with_timeout([mt], fetch_start, required_end, adv_timeout)
            sub.update(_ingest_raw(raw_single, [mt], fetch_start))
    elif missed:
        logger.warning(
            "[Cache] Chunk %d/%d: %d tickers missed and exceed surgical cap — skipping.",
            chunk_idx, total_chunks, len(missed),
        )

    logger.info(
        "[Cache] Chunk %d/%d: %d/%d tickers written to disk.",
        chunk_idx, total_chunks, len(sub), len(chunk),
    )
    return sub


# ─── Public Interface ─────────────────────────────────────────────────────────

def load_or_fetch(
    tickers:        List[str],
    required_start: str,
    required_end:   str,
    force_refresh:  bool = False,
    cfg=None,
) -> Dict[str, pd.DataFrame]:
    """
    Main entry point for retrieving market data.

    FIX D1: Large `to_download` lists are now split into chunks of
    _DOWNLOAD_CHUNK_SIZE tickers. Each chunk is downloaded with its own
    per-chunk timeout (_CHUNK_TIMEOUT_S), ingested, and committed to the
    manifest before the next chunk starts. This replaces the previous single
    giant batch call that reliably timed out for 300+ ticker universes.

    The manifest is committed after each successful chunk so a mid-run
    interruption (Ctrl-C, crash, network drop) preserves all already-
    downloaded data and the next run will only fetch what is missing.
    """
    chunk_timeout = getattr(cfg, "YF_CHUNK_TIMEOUT",  _CHUNK_TIMEOUT_S)
    adv_timeout   = getattr(cfg, "YF_ADV_TIMEOUT",    60.0)
    os.makedirs(CACHE_DIR, exist_ok=True)

    full_manifest    = _load_manifest()
    manifest_entries = full_manifest["entries"]

    standard_tickers = list(dict.fromkeys(
        t if (t.endswith(".NS") or t.startswith("^")) else t + ".NS"
        for t in tickers
    ))

    if not required_start or str(required_start).strip() == "":
        required_start = "2020-01-01"

    fetch_start = (pd.Timestamp(required_start) - timedelta(days=400)).strftime("%Y-%m-%d")
    today_bday  = (pd.Timestamp.today().normalize() - pd.offsets.BDay(1)).strftime("%Y-%m-%d")

    to_download: List[str]               = []
    market_data: Dict[str, pd.DataFrame] = {}

    for t in standard_tickers:
        entry         = manifest_entries.get(t, {})
        fetched_at    = entry.get("fetched_at", "2000-01-01")
        covered_start = entry.get("covered_start", entry.get("first_date", "2099-01-01"))
        last_date     = entry.get("last_date", "2000-01-01")

        stale_time   = (datetime.now() - datetime.fromisoformat(fetched_at)) > timedelta(hours=20)
        stale_bday   = last_date < today_bday
        parquet_path = os.path.join(CACHE_DIR, f"{t}.parquet")

        needs_download = (
            force_refresh
            or stale_time
            or stale_bday
            or not os.path.exists(parquet_path)
            or pd.Timestamp(covered_start) > pd.Timestamp(fetch_start)
        )

        if needs_download:
            to_download.append(t)
        else:
            try:
                market_data[t] = pd.read_parquet(parquet_path)
            except Exception:
                to_download.append(t)

    if to_download:
        # FIX D1: Split into manageable chunks so each yf.download() call
        # completes within the per-chunk timeout even for large universes.
        chunks = [
            to_download[i : i + _DOWNLOAD_CHUNK_SIZE]
            for i in range(0, len(to_download), _DOWNLOAD_CHUNK_SIZE)
        ]
        total_chunks = len(chunks)

        logger.info(
            "[Cache] %d tickers to download → %d chunk(s) of ≤%d "
            "(%s → %s, timeout %.0fs/chunk).",
            len(to_download), total_chunks, _DOWNLOAD_CHUNK_SIZE,
            fetch_start, required_end, chunk_timeout,
        )

        for chunk_idx, chunk in enumerate(chunks, start=1):
            sub = _download_chunk(
                chunk, fetch_start, required_end,
                chunk_timeout, adv_timeout,
                chunk_idx, total_chunks,
            )

            # Commit each chunk to the manifest immediately so that a crash
            # or keyboard-interrupt mid-run preserves all completed work.
            if sub:
                manifest_entries.update(sub)
                _save_manifest(full_manifest)
                for t in sub:
                    try:
                        market_data[t] = pd.read_parquet(
                            os.path.join(CACHE_DIR, f"{t}.parquet")
                        )
                    except Exception:
                        pass

            # Polite inter-chunk pause to avoid Yahoo Finance rate-limiting.
            if chunk_idx < total_chunks:
                time.sleep(_INTER_CHUNK_SLEEP_S)

        total_written = sum(1 for t in to_download if t in market_data)
        logger.info(
            "[Cache] Download complete: %d/%d tickers available.",
            total_written, len(to_download),
        )

    return market_data


def get_cache_summary() -> pd.DataFrame:
    """
    Returns a DataFrame summarising cached tickers for diagnostic purposes.
    Includes `suspended` and `max_gap_days` columns to surface ASM/GSM history.
    """
    m       = _load_manifest()
    entries = m.get("entries", {})
    if not entries:
        return pd.DataFrame(
            columns=["ticker", "fetched_at", "rows", "last_date", "suspended", "max_gap_days"]
        )
    return pd.DataFrame([{"ticker": k, **v} for k, v in entries.items()])


def invalidate_cache() -> None:
    """Removes all cached parquet files and the manifest."""
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
