"""
data_cache.py — Persistent Atomic Downloader v11.49
===================================================
Robustly manages downloading, parsing, and persisting yfinance data.

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
- BUG-FIX-DOTENV-DC: _load_local_env_file now strips inline comments before
  removing quotes, matching the fix applied to optimizer.py.
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
import time
from abc import ABC, abstractmethod
from pathlib import Path

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Sentinel returned by _fetch_candles_chunk to signal rate-limit hit
_RATE_LIMITED = object()

# Maximum consecutive 429 retries per symbol before aborting
_MAX_RATE_LIMIT_RETRIES = 5


def _load_local_env_file(env_path: Path = Path('.env')) -> None:
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            # BUG-FIX-DOTENV-DC: strip inline comments before removing quotes.
            # "FALLBACK_API_KEY=mykey # comment" left " # comment" in the value.
            if value and value[0] in ('"', "'"):
                q = value[0]
                if value.endswith(q) and len(value) >= 2:
                    value = value[1:-1]
            else:
                for _sep in (' #', '\t#'):
                    if _sep in value:
                        value = value[:value.index(_sep)].rstrip()
                        break
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:
        logger.debug('[Cache] Could not parse .env file at %s: %s', env_path, exc)


_load_local_env_file()

CACHE_DIR     = Path("data/cache")
MANIFEST_FILE = CACHE_DIR / "_manifest.json"
_DOWNLOAD_CHUNK_SIZE = 75

_GROWW_BASE_URL           = "https://api.groww.in/v1"
_GROWW_DAILY_INTERVAL     = 1440
_GROWW_MAX_DAYS_PER_REQUEST = 1080
_GROWW_CHUNK_DAYS         = 1080
_GROWW_INDEX_PREFIXES     = ("^",)


class DataFetchError(RuntimeError):
    """Raised when one or more requested ticker chunks cannot be fetched."""


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
    """

    _GROWW_SLEEP_SECS = 0.2

    def __init__(self, api_token: Optional[str] = None) -> None:
        self.api_token = api_token or os.getenv("GROWW_API_TOKEN", "").strip()
        self._session: Optional[requests.Session] = None

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
        params = {
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
            series.index = series.index.tz_convert("Asia/Kolkata").tz_localize(None)
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
            adj_yf     = self._extract_batch_series(yf_raw, "Adj Close", ns_ticker)
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
        splits = self._extract_batch_series(yf_raw, "Stock Splits", ns_ticker)

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
                yf_raw = yf.download(
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
            raw_df["Adj Close"] = adj_close

            dividends, splits = self._extract_actions_from_batches(raw_df.index, ns_ticker, yf_raw)
            raw_df["Dividends"] = dividends
            raw_df["Stock Splits"] = splits

            frames[ns_ticker] = raw_df

        if not frames:
            return None

        combined = pd.concat(frames, axis=1)
        combined.columns = pd.MultiIndex.from_tuples(
            [(col, tkr) for tkr, col in combined.columns],
            names=["Price", "Ticker"],
        )
        combined = combined.swaplevel(0, 1, axis=1).sort_index(axis=1)
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
            return None
        return result

    def _download_batch(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        try:
            return yf.download(
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
            logger.warning("[Cache][Fallback] FALLBACK_API_KEY not set; skipping secondary provider.")
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
                "Adj Close": float(item.get("5. adjusted close",   np.nan)),
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
            out.index = out.index.tz_convert("Asia/Kolkata").tz_localize(None)
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
                logger.error("[Cache] Provider failed after %d retries.", max_retries)
                raise errors[-1] from errors[0]
            time.sleep((2 ** attempt) + random.random())

    return None


def _load_manifest() -> dict:
    default_manifest = {"schema_version": 1, "entries": {}}
    if not MANIFEST_FILE.exists():
        return default_manifest

    try:
        with MANIFEST_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
            if "schema_version" in data:
                return data
            else:
                return {"schema_version": 1, "entries": data}
    except Exception as exc:
        logger.warning("[Cache] Manifest corrupted or unreadable. Starting fresh. Error: %s", exc)
        return default_manifest


def _save_manifest(manifest_data: dict) -> None:
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
        logger.error("[Cache] Failed to save manifest: %s", exc)


def invalidate_cache() -> None:
    if MANIFEST_FILE.exists():
        try:
            MANIFEST_FILE.unlink()
            logger.info("[Cache] Market data cache invalidated.")
        except OSError as e:
            logger.error("[Cache] Failed to invalidate cache: %s", e)


def _is_valid_dataframe(df: pd.DataFrame, ticker: Optional[str] = None, cfg=None) -> bool:
    min_rows = int(getattr(cfg, "HISTORY_GATE", 5)) if cfg is not None else 5
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

    if "Adj Close" not in df.columns or df["Adj Close"].isnull().all():
        return False

    is_index_ticker = bool(ticker) and str(ticker).startswith("^")
    if (not is_index_ticker) and ("Volume" not in df.columns or df["Volume"].isnull().all()):
        return False

    return True


def _ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[:, out.columns.notna()]

    numeric_cols = [
        "Open", "High", "Low", "Close", "Adj Close",
        "Volume", "Dividends", "Stock Splits",
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

    if "Adj Close" not in out.columns:
        if "Close" in out.columns:
            out["Adj Close"] = out["Close"]
    else:
        if "Close" in out.columns:
            out["Adj Close"] = out["Adj Close"].fillna(out["Close"])

    for col in ["Dividends", "Stock Splits"]:
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(0.0)

    return out


def _latest_business_day() -> str:
    """
    Return the most recent Mon–Fri business day before today as 'YYYY-MM-DD'.

    Note: uses pandas BDay (Mon–Fri only) rather than an NSE-specific holiday
    calendar.  On the trading day immediately following an NSE market holiday
    (e.g. Holi, Diwali) this will return the holiday date — a day with no NSE
    data — causing the staleness check to mark every symbol as stale and trigger
    spurious re-downloads.  The downloads themselves succeed (providers return
    the data up to the last real trading day), so this is a performance/noise
    issue rather than a correctness bug.  A future improvement would integrate
    a pandas_market_calendars 'NSE' calendar here.
    """
    today = pd.Timestamp.today().normalize()
    latest_bday_ts = today - pd.offsets.BDay(1)
    return latest_bday_ts.strftime("%Y-%m-%d")


def _build_provider_chain(cfg=None) -> List[DataProvider]:
    chain: List[DataProvider] = []

    groww_token = os.getenv("GROWW_API_TOKEN", "").strip()
    if groww_token:
        chain.append(GrowwProvider(api_token=groww_token))
        logger.debug("[Cache] GrowwProvider enabled as primary data source.")
    else:
        logger.debug(
            "[Cache] GROWW_API_TOKEN not set — using YFinanceProvider only."
        )

    chain.append(YFinanceProvider())
    chain.append(SecondaryProvider())
    return chain


def load_or_fetch(
    tickers: List[str],
    required_start: str,
    required_end: str,
    force_refresh: bool = False,
    cfg=None,
) -> Dict[str, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    entries  = manifest["entries"]

    def _clean_ticker(t: str) -> str:
        t_str = str(t).strip()
        if t_str.startswith("^"):
            return t_str
        return t_str.replace(".NSE", "").replace(".NS", "") + ".NS"

    standardized_tickers = list(dict.fromkeys(_clean_ticker(t) for t in tickers))

    cfg_lookback = int(getattr(cfg, "CVAR_LOOKBACK", 200) or 200)
    dynamic_padding_days = max(400, cfg_lookback * 2)
    padded_start = (
        pd.Timestamp(required_start or "2020-01-01") - timedelta(days=dynamic_padding_days)
    ).strftime("%Y-%m-%d")

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
                logger.debug("[Cache] Corrupted parquet for %s: %s", ticker, exc)
                tickers_to_download.append(ticker)

    providers = _build_provider_chain(cfg)

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
                    _yf_end = (pd.Timestamp(required_end) + timedelta(days=1)).strftime("%Y-%m-%d")
                    raw_data = _download_with_timeout(chunk, padded_start, _yf_end, provider=provider)
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
                    saved = _process_chunk(chunk, raw_data, entries, market_data, cfg)
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
                _recover_from_stale_cache(chunk, entries, market_data)

        # BUG-FIX-MANIFEST-IO: save manifest once after ALL chunks complete,
        # not once per chunk. Previously 300 tickers (4 chunks) triggered 4
        # full JSON serialisations and atomic renames. One write is sufficient
        # because the manifest is only read at startup and the per-chunk
        # entries have already been mutated in-place in `manifest`.
        _save_manifest(manifest)

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
        market_data[t] = df.loc[:required_end]

    return market_data


def _process_chunk(
    chunk: List[str],
    raw_data: pd.DataFrame,
    entries: dict,
    market_data: Dict[str, pd.DataFrame],
    cfg,
) -> set:
    """Process a downloaded chunk and return the set of successfully saved tickers.

    BUG-FIX-FALLBACK: Previously returned None, so the caller inferred served
    tickers from raw column names — which includes NaN-only columns that pass
    the column-presence check but fail _is_valid_dataframe.  Those tickers were
    added to `served`, blocking the SecondaryProvider fallback even though no
    valid data was actually saved.  Returning the validated set here lets the
    caller use ground-truth instead of inferred membership.
    """
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
                    "(provider returned None or empty frame for this symbol; "
                    "check symbol name, market hours, or provider status).",
                    ticker,
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
            df.to_parquet(parquet_path)

            gap_series   = df.index.to_series().diff().dt.days
            max_gap      = gap_series.max()
            max_gap_days = int(max_gap) if pd.notna(max_gap) else 0
            expected_gap = 7

            manifest_entries[ticker] = {
                "fetched_at":    datetime.now().isoformat(),
                "rows":          len(df),
                "last_date":     df.index[-1].strftime("%Y-%m-%d"),
                "suspended":     max_gap_days > expected_gap,
                "max_gap_days":  max_gap_days,
            }
            market_data[ticker] = df
            saved_tickers.add(ticker)

        except Exception as exc:
            logger.error("[Cache] Failed processing downloaded dataframe for %s: %s", ticker, exc)

    return saved_tickers


def _recover_from_stale_cache(
    chunk: List[str],
    entries: dict,
    market_data: Dict[str, pd.DataFrame],
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
    recovered = 0
    for ticker in chunk:
        parquet_path = CACHE_DIR / f"{ticker}.parquet"
        if not parquet_path.exists():
            continue
        try:
            fallback_df = _normalize_history_index(pd.read_parquet(parquet_path))
            if fallback_df is None or fallback_df.empty:
                continue
            market_data[ticker] = fallback_df
            recovered += 1

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
            _tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            entries[ticker] = {
                "fetched_at":   existing.get("fetched_at", datetime.now().isoformat()),
                "rows":         len(fallback_df),
                "last_date":    fallback_df.index[-1].strftime("%Y-%m-%d"),
                "suspended":    existing.get("suspended", False),
                "max_gap_days": existing.get("max_gap_days", 0),
                "stale_recovery": True,
                "retry_after":  _tomorrow,
            }

            logger.warning(
                "[Cache] Using stale cached parquet for %s after all providers failed "
                "(last_date=%s).",
                ticker, entries[ticker]["last_date"],
            )
        except Exception as exc:
            logger.warning(
                "[Cache] Failed loading stale parquet fallback for %s: %s", ticker, exc,
            )

    missing = len(chunk) - recovered
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
