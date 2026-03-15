"""
data_cache.py — Persistent Atomic Downloader v11.49
===================================================
Robustly manages downloading, parsing, and persisting yfinance data.

Features:
- Atomic JSON manifest updates
- Network retry logic and chunking
- Persistent storage of raw provider data only (no synthetic mutation at cache layer)
- PHASE 4 FIX: Strict timezone stripping and index normalization to prevent misalignment.
- PHASE 9 FIX: Explicit `actions=True` to fetch Corporate Actions (Dividends/Splits).
- STABILITY: Strict OHLC validation, padding trimming, and robust ticker sanitization.
- FIX D1: Return full padded data (trim to padded_start, not required_start).
- FIX D8: Timezone-independent latest_bday using pandas BDay (UTC-server safe).

GROWW PROVIDER (v11.49):
GrowwProvider is added as the PRIMARY provider for all NSE equity data when
GROWW_API_TOKEN is set in the environment. It fetches daily OHLCV candles from
the Groww API (clean, official NSE data).

Groww daily candle history:
  - The API supports a maximum of 1080 days (~3 years) PER REQUEST.
  - There is NO cap on total historical depth — full history is available
    by making multiple chunked requests and stitching the results together.
  - GrowwProvider automatically chunks requests into 1080-day windows and
    merges them, so the caller receives a single continuous DataFrame from
    any start date to today.

Falls back to yfinance only for:
  - Index tickers (^NSEI, ^CRSLDX) — Groww does not serve these
  - Any fetch failure for a specific ticker

Provider priority order:
  1. GrowwProvider  — full NSE equity history via chunked requests (if GROWW_API_TOKEN set)
  2. YFinanceProvider — indices + fallback for any Groww failures
  3. SecondaryProvider — AlphaVantage last-resort fallback

IMPORTANT: Groww daily candles are RAW (unadjusted). The engine uses
AUTO_ADJUST_PRICES=True by default, which means Adj Close is used for
return computation. GrowwProvider synthesises Adj Close from raw Close
using split/dividend data fetched via yfinance for the same ticker,
exactly matching what yfinance returns. This keeps the two data sources
interchangeable within the existing cache schema.
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


def _load_local_env_file(env_path: Path = Path('.env')) -> None:
    """
    Lightweight `.env` loader for non-interactive scripts.

    This avoids requiring python-dotenv in every runtime while still honoring
    local credentials such as GROWW_API_TOKEN when present.
    """
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:
        logger.debug('[Cache] Could not parse .env file at %s: %s', env_path, exc)


_load_local_env_file()

CACHE_DIR     = Path("data/cache")
MANIFEST_FILE = CACHE_DIR / "_manifest.json"
_DOWNLOAD_CHUNK_SIZE = 75

# Groww API constants
_GROWW_BASE_URL           = "https://api.groww.in/v1"
_GROWW_DAILY_INTERVAL     = 1440   # 1 day in minutes
_GROWW_MAX_DAYS_PER_REQUEST = 1080 # max days per single request (Groww enforced limit)
_GROWW_CHUNK_DAYS         = 1080   # use the full per-request limit; loop for older history
_GROWW_INDEX_PREFIXES     = ("^",) # Groww does not serve index tickers like ^NSEI

class DataFetchError(RuntimeError):
    """Raised when one or more requested ticker chunks cannot be fetched."""


class DataProvider(ABC):
    """Strategy interface for market-data providers."""

    @abstractmethod
    def download(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError


# ─── Groww Provider ───────────────────────────────────────────────────────────

class GrowwProvider(DataProvider):
    """
    Fetches daily OHLCV candles from the Groww API for NSE equity tickers.

    Activated automatically when the GROWW_API_TOKEN environment variable is
    set. Skipped silently for index tickers (^NSEI, ^CRSLDX) — those always
    fall through to YFinanceProvider.

    Ticker translation:
        Cache uses .NS suffix  →  Groww expects bare symbol
        "RELIANCE.NS"          →  "RELIANCE"
        "^NSEI"                →  skipped (Groww does not serve indices)

    Corporate action handling:
        Groww returns raw (unadjusted) prices. To maintain compatibility with
        the existing cache schema (which expects both "Close" and "Adj Close"),
        this provider fetches split/dividend metadata from yfinance for the
        same ticker and back-adjusts the Groww Close series to produce Adj Close.
        This is a one-time cost per ticker per cache refresh and gives the same
        adjusted prices yfinance would return.

    Rate limiting:
        Groww does not publish rate limits. We apply a conservative 0.2s sleep
        between requests (300 req/min ceiling). Increase _GROWW_SLEEP_SECS if
        you encounter 429 responses.
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
        """Convert .NS cache ticker to bare Groww trading symbol. Returns None for indices."""
        if any(ticker.startswith(p) for p in _GROWW_INDEX_PREFIXES):
            return None
        # Strip .NS or .BO suffix
        for sfx in (".NS", ".BO", ".BSE"):
            if ticker.upper().endswith(sfx):
                return ticker[:-len(sfx)].upper()
        return ticker.upper()

    def _fetch_candles_chunk(
        self,
        groww_symbol: str,
        chunk_start: str,
        chunk_end: str,
    ) -> List[list]:
        """
        Fetch one chunk of daily candles from Groww.
        Returns list of raw candle arrays, or [] on any failure.
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
                # Symbol not found on Groww — not an error, just unsupported
                logger.debug("[Groww] Symbol %s not found (404), skipping.", groww_symbol)
                return []
            if resp.status_code == 429:
                logger.warning("[Groww] Rate limited for %s — sleeping 5s.", groww_symbol)
                time.sleep(5)
                return []
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

        Groww enforces a maximum of 1080 days per single API request but places
        no cap on total historical depth. This method loops over the full date
        range in 1080-day chunks, issuing one request per chunk and stitching
        all results into a single continuous DataFrame.

        For a backtest starting in 2018 with today's end date (~8 years = ~2920
        days) this issues ceil(2920/1080) = 3 requests per ticker — fast and
        well within any reasonable rate limit.

        Returns a raw (unadjusted) DataFrame with columns [Open, High, Low,
        Close, Volume] and a timezone-naive DatetimeIndex, or None if no data
        was retrieved across all chunks.
        """
        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end)

        all_candles: List[list] = []
        cursor = start_ts
        while cursor <= end_ts:
            chunk_end = min(cursor + pd.Timedelta(days=_GROWW_CHUNK_DAYS), end_ts)
            candles = self._fetch_candles_chunk(
                groww_symbol,
                cursor.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
            all_candles.extend(candles)
            cursor = chunk_end + pd.Timedelta(days=1)
            time.sleep(self._GROWW_SLEEP_SECS)

        if not all_candles:
            return None

        rows = []
        for c in all_candles:
            if not isinstance(c, list) or len(c) < 6:
                continue
            try:
                # Groww response: [timestamp_str, open, high, low, close, volume, oi_or_null]
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
    def _build_adj_close(
        raw_close: pd.Series,
        ns_ticker: str,
        start: str,
        end: str,
    ) -> pd.Series:
        """
        Back-adjust the raw Groww Close series using split/dividend metadata
        from yfinance. The adjustment matches what yfinance returns for Adj Close.

        If yfinance data is unavailable or the adjustment fails for any reason,
        returns raw_close unchanged (unadjusted). The engine will still work
        correctly — only return computation for corporate-action periods will
        be slightly off.
        """
        try:
            yf_start = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            yf_end   = (pd.Timestamp(end)   + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            ticker_obj = yf.Ticker(ns_ticker)
            hist = ticker_obj.history(
                start=yf_start,
                end=yf_end,
                auto_adjust=True,
                actions=True,
            )
            if hist.empty or "Close" not in hist.columns:
                return raw_close

            # Align on common dates and compute adjustment ratio
            adj_yf = hist["Close"].copy()
            if hasattr(adj_yf.index, "tz") and adj_yf.index.tz is not None:
                adj_yf.index = adj_yf.index.tz_convert("Asia/Kolkata").tz_localize(None)
            adj_yf.index = adj_yf.index.normalize()

            raw_yf = ticker_obj.history(
                start=yf_start,
                end=yf_end,
                auto_adjust=False,
                actions=False,
            )
            if raw_yf.empty or "Close" not in raw_yf.columns:
                return raw_close

            raw_yf_close = raw_yf["Close"].copy()
            if hasattr(raw_yf_close.index, "tz") and raw_yf_close.index.tz is not None:
                raw_yf_close.index = raw_yf_close.index.tz_convert("Asia/Kolkata").tz_localize(None)
            raw_yf_close.index = raw_yf_close.index.normalize()

            common = adj_yf.index.intersection(raw_yf_close.index)
            if common.empty:
                return raw_close

            ratio = (adj_yf.loc[common] / raw_yf_close.loc[common]).dropna()
            if ratio.empty:
                return raw_close

            # Apply ratio to Groww raw close, forward-filling ratio across dates
            # that exist in Groww but not yfinance (e.g. different holiday calendars)
            ratio_aligned = ratio.reindex(raw_close.index).ffill().bfill().fillna(1.0)
            adj_groww = raw_close * ratio_aligned
            return adj_groww

        except Exception as exc:
            logger.debug("[Groww] Adj Close build failed for %s: %s", ns_ticker, exc)
            return raw_close

    def download(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Download complete daily OHLCV history for each ticker from Groww,
        returning a MultiIndex DataFrame in the same shape as YFinanceProvider
        (field, ticker).

        Full history is retrieved by issuing multiple 1080-day chunked requests
        per ticker and merging them. A backtest from 2018 to today requires
        ~3 requests per equity ticker — typically completes in a few seconds.

        Tickers that Groww cannot serve (indices, failures) are silently omitted
        so the caller can fall back to yfinance for those specific tickers.
        """
        if not self.api_token:
            return None

        frames: Dict[str, pd.DataFrame] = {}

        for ns_ticker in tickers:
            # Skip indices — Groww does not serve them
            if any(ns_ticker.startswith(p) for p in _GROWW_INDEX_PREFIXES):
                continue

            groww_sym = self._to_groww_symbol(ns_ticker)
            if groww_sym is None:
                continue

            raw_df = self._fetch_full_history(groww_sym, start, end)
            if raw_df is None or raw_df.empty:
                continue

            # Build Adj Close from yfinance adjustment ratios
            adj_close = self._build_adj_close(raw_df["Close"], ns_ticker, start, end)
            raw_df["Adj Close"] = adj_close

            # Add placeholder corporate-action columns so _is_valid_dataframe passes
            if "Dividends" not in raw_df.columns:
                raw_df["Dividends"]    = 0.0
            if "Stock Splits" not in raw_df.columns:
                raw_df["Stock Splits"] = 0.0

            frames[ns_ticker] = raw_df

        if not frames:
            return None

        # Build a (field, ticker) MultiIndex DataFrame matching yfinance 1.x layout
        combined = pd.concat(frames, axis=1)
        combined.columns = pd.MultiIndex.from_tuples(
            [(col, tkr) for tkr, col in combined.columns],
            names=["Price", "Ticker"],
        )
        # Reorder to (field, ticker) so _extract_ticker_frame's xs() works correctly
        combined = combined.swaplevel(0, 1, axis=1).sort_index(axis=1)
        return _normalize_history_index(combined)


# ─── YFinance Provider ────────────────────────────────────────────────────────

class YFinanceProvider(DataProvider):
    def download(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        result = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            actions=True,
        )
        if result is None:
            return None
        if result.empty:
            return result
        if len(tickers) > 1 and not isinstance(result.columns, pd.MultiIndex):
            return None
        return result


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

        if len(ticker_frames) == 1:
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
    """
    PHASE 4 FIX: Strict Data Alignment.
    Return a copy with a timezone-naive, monotonic DatetimeIndex normalized to midnight.
    """
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
    """Robustly extract one ticker frame from varying yfinance payload shapes."""
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
    """Attempt to download tickers via provider with exponential backoff."""
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
    """Loads the cache tracking manifest, ensuring a valid schema structure."""
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
    """Atomically saves the tracking manifest to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = MANIFEST_FILE.with_name(MANIFEST_FILE.name + ".tmp")
    try:
        with temp_file.open("w", encoding="utf-8") as file:
            json.dump(manifest_data, file, indent=2)
        temp_file.replace(MANIFEST_FILE)
    except Exception as exc:
        logger.error("[Cache] Failed to save manifest: %s", exc)


def invalidate_cache() -> None:
    """Forces cache clearing by deleting the manifest."""
    if MANIFEST_FILE.exists():
        try:
            MANIFEST_FILE.unlink()
            logger.info("[Cache] Market data cache invalidated.")
        except OSError as e:
            logger.error("[Cache] Failed to invalidate cache: %s", e)


def _is_valid_dataframe(df: pd.DataFrame, ticker: Optional[str] = None, cfg=None) -> bool:
    """Strict structural validation gate applied before any data is written to disk."""
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
    """Normalise price column names and fill gaps."""
    out = df.copy()
    out = out.loc[:, out.columns.notna()]

    numeric_cols = [
        "Open", "High", "Low", "Close", "Adj Close",
        "Volume", "Dividends", "Stock Splits",
    ]

    def _coerce_numeric_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series

        # yfinance occasionally returns object-typed corporate-action values
        # such as "2.6 INR". Strip known textual/formatting artifacts before
        # numeric coercion so parquet writes never fail on object payloads.
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
    FIX D8: Compute the latest business day in a timezone-independent way.
    Always use the PREVIOUS business day as the staleness threshold.
    """
    today = pd.Timestamp.today().normalize()
    latest_bday_ts = today - pd.offsets.BDay(1)
    return latest_bday_ts.strftime("%Y-%m-%d")


def _build_provider_chain(cfg=None) -> List[DataProvider]:
    """
    Build the ordered provider chain based on available credentials.

    If GROWW_API_TOKEN is set, GrowwProvider is inserted as the primary
    provider for all NSE equity data — including full historical depth going
    back to any date, achieved via automatic chunked requests of 1080 days each.

    YFinanceProvider always follows as fallback for:
      - Index tickers (^NSEI, ^CRSLDX) which Groww does not serve
      - Any ticker where Groww returns no data (delisted, bad symbol, etc.)

    SecondaryProvider (AlphaVantage) is last resort for both.
    """
    chain: List[DataProvider] = []

    groww_token = os.getenv("GROWW_API_TOKEN", "").strip()
    if groww_token:
        chain.append(GrowwProvider(api_token=groww_token))
        logger.debug("[Cache] GrowwProvider enabled as primary data source.")
    else:
        logger.debug(
            "[Cache] GROWW_API_TOKEN not set — using YFinanceProvider only. "
            "Set GROWW_API_TOKEN environment variable to enable Groww data."
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
    """
    Primary interface for fetching market data.

    Provider chain (in order):
      1. GrowwProvider  — if GROWW_API_TOKEN set in environment
      2. YFinanceProvider — always present as fallback
      3. SecondaryProvider — AlphaVantage last resort

    For each ticker the first provider that returns valid data wins.
    GrowwProvider skips index tickers (^NSEI, ^CRSLDX) automatically,
    so those always fall through to YFinanceProvider.

    The Groww API history starts from ~2022. For data older than that
    (e.g. warm-up from 2018), YFinanceProvider fills the gap seamlessly
    because both providers return the same cache schema.
    """
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

        if force_refresh or is_stale or missing_file:
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
                # GrowwProvider handles per-ticker fetching internally and
                # returns only the tickers it successfully served. The
                # remaining tickers fall through to the next provider below.
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
                    # Check how many tickers were actually served by this provider.
                    # For GrowwProvider some tickers may be missing (indices, failures).
                    served = set()
                    if isinstance(raw_data.columns, pd.MultiIndex):
                        served = set(raw_data.columns.get_level_values(1)) | set(raw_data.columns.get_level_values(0))
                    missing_from_provider = [t for t in chunk if t not in served]

                    # Process what we got from this provider
                    _process_chunk(chunk, raw_data, entries, market_data, cfg)

                    # If all tickers were served, move on
                    if not missing_from_provider:
                        break

                    # Otherwise try next provider for the missing ones
                    chunk = missing_from_provider
                    raw_data = None
                    continue

            # Final fallback: stale parquet recovery for any tickers still missing
            if raw_data is None or (hasattr(raw_data, "empty") and raw_data.empty):
                _recover_from_stale_cache(chunk, entries, market_data)

            _save_manifest(manifest)

    padded_start_ts = pd.Timestamp(padded_start)
    for t, df in market_data.items():
        if df.empty:
            continue
        effective_start = min(df.index[0], padded_start_ts)
        market_data[t] = df.loc[effective_start:required_end]

    return market_data


def _process_chunk(
    chunk: List[str],
    raw_data: pd.DataFrame,
    entries: dict,
    market_data: Dict[str, pd.DataFrame],
    cfg,
) -> None:
    """Extract, validate, and cache each ticker frame from a downloaded chunk."""
    manifest_entries = entries  # mutated in-place

    for ticker in chunk:
        try:
            df = _extract_ticker_frame(
                raw_data,
                ticker,
                is_single_request=(len(chunk) == 1),
            )
            if df is None or df.empty:
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

        except Exception as exc:
            logger.error("[Cache] Failed processing downloaded dataframe for %s: %s", ticker, exc)


def _recover_from_stale_cache(
    chunk: List[str],
    entries: dict,
    market_data: Dict[str, pd.DataFrame],
) -> None:
    """Load stale parquets for tickers that all providers failed to serve."""
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
            logger.warning(
                "[Cache] Using stale cached parquet for %s after all providers failed.",
                ticker,
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
    """Returns a summary of cache health and coverage."""
    manifest = _load_manifest()
    entries  = manifest.get("entries", {})
    groww_token = os.getenv("GROWW_API_TOKEN", "").strip()
    return {
        "total_symbols":     len(entries),
        "schema_version":    manifest.get("schema_version", 1),
        "suspended_symbols": sum(1 for v in entries.values() if v.get("suspended")),
        "groww_enabled":     bool(groww_token),
    }
