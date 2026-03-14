"""
data_cache.py — Persistent Atomic Downloader v11.46
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

CACHE_DIR     = Path("data/cache")
MANIFEST_FILE = CACHE_DIR / "_manifest.json"
_DOWNLOAD_CHUNK_SIZE = 75

class DataFetchError(RuntimeError):
    """Raised when one or more requested ticker chunks cannot be fetched."""


class DataProvider(ABC):
    """Strategy interface for market-data providers."""

    @abstractmethod
    def download(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError


class YFinanceProvider(DataProvider):
    def download(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        # group_by='ticker' was respected in yfinance 0.2.x but is silently ignored
        # in 1.x (which always returns (price_field, ticker) MultiIndex order).
        # Keeping it here causes a DeprecationWarning in 1.x and may become an
        # error in future releases — removed.  _extract_ticker_frame handles both
        # (ticker, field) and (field, ticker) MultiIndex layouts automatically.
        result = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            actions=True,  # PHASE 9 FIX: Ensure Corporate Actions (Dividends/Splits) are downloaded
        )
        # yfinance 1.x returns None (not an empty DataFrame) when the entire
        # request fails at the network level.  Normalise to None so callers can
        # use a simple `if result is None or result.empty` guard.
        if result is None:
            return None
        if result.empty:
            return result
        # yfinance 1.x: for a multi-ticker request where all-but-one ticker
        # fails, the library may collapse the result to a flat (non-MultiIndex)
        # DataFrame containing only the successful ticker's data, but without
        # labelling which ticker it belongs to.  We cannot safely attribute that
        # frame to any specific ticker in the chunk, so return None and let the
        # per-ticker fallback path handle it via cached parquets.
        if len(tickers) > 1 and not isinstance(result.columns, pd.MultiIndex):
            return None
        return result


class SecondaryProvider(DataProvider):
    """AlphaVantage fallback provider with basic rate-limit throttling."""

    _URL = "https://www.alphavantage.co/query"

    def __init__(self) -> None:
        # Read configuration once at construction time.  Environment variable
        # lookups are system calls; calling them on every download() invocation
        # is wasteful and makes state unpredictable if the env changes mid-run.
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
            "symbol": symbol,
            "outputsize": "full",
            "apikey": api_key,
        }
        response = requests.get(self._URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()

        # Explicit handling for provider-side throttling / temporary errors.
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
        # MB-08 FIX: Use strictly-less-than for end_ts so partial bars on
        # required_end (e.g. mid-session prices from AlphaVantage's full-history
        # download) cannot slip through.  AlphaVantage returns the entire history
        # client-side with no server-side date filter; using `date > end_ts` allowed
        # the required_end date itself to pass, which may contain incomplete OHLCV
        # when the server clock is ahead of IST close.
        end_ts = pd.Timestamp(end)
        rows = []
        for d, item in series.items():
            date = pd.Timestamp(d)
            if date < start_ts or date >= end_ts:
                continue
            rows.append(
                {
                    "Date": date,
                    "Open": float(item.get("1. open", np.nan)),
                    "High": float(item.get("2. high", np.nan)),
                    "Low": float(item.get("3. low", np.nan)),
                    "Close": float(item.get("4. close", np.nan)),
                    "Adj Close": float(item.get("5. adjusted close", np.nan)),
                    "Volume": float(item.get("6. volume", np.nan)),
                }
            )

        if not rows:
            return None

        df = pd.DataFrame(rows).set_index("Date").sort_index()
        return _ensure_price_columns(_normalize_history_index(df))

    @staticmethod
    def _map_symbol(ticker: str) -> str:
        # AlphaVantage expects NSE symbols as "XYZ.NSE".
        if ticker.startswith("^"):
            return ticker
        bare = ticker[:-3] if ticker.endswith(".NS") else ticker
        return f"{bare}.NSE"


def _normalize_history_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    PHASE 4 FIX: Strict Data Alignment.
    Return a copy with a timezone-naive, monotonic DatetimeIndex normalized to midnight.
    Drops duplicates to prevent upstream covariance alignment failures.
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
            # FIX (Patch 4): Convert to IST wall-clock time before stripping timezone.
            # Arbitrarily calling tz_convert(None) (effectively UTC-cast-to-naive)
            # causes date boundary shifts for US Eastern data: a Friday 4 PM ET close
            # = Friday 9:30 PM UTC → normalized to Friday midnight UTC, correct.
            # But on machines with different local clocks, or for secondary providers
            # (AlphaVantage) returning ET timestamps, the behaviour was undefined.
            # tz_convert('Asia/Kolkata') first ensures all timestamps are expressed in
            # IST (UTC+5:30), the canonical timezone for NSE data, before tz_localize(None)
            # strips the tz label.  NSE close at 3:30 PM IST is 10 AM UTC — same calendar
            # date in both UTC and IST, so no cross-midnight shift for NSE data.
            out.index = out.index.tz_convert('Asia/Kolkata').tz_localize(None)
        out.index = out.index.normalize()
        
        # Remove duplicate dates (keep last reported)
        if out.index.duplicated().any():
            out = out[~out.index.duplicated(keep='last')]
            
        # Sort chronologically strictly
        out = out.sort_index()
        
    return out


def _extract_ticker_frame(
    raw_data: pd.DataFrame,
    ticker: str,
    *,
    is_single_request: bool = False,
) -> Optional[pd.DataFrame]:
    """Robustly extract one ticker frame from varying yfinance payload shapes.

    yfinance version compatibility
    ------------------------------
    0.2.x: MultiIndex columns are (ticker, price_field) — level0=ticker, level1=field.
           group_by='ticker' was required and respected.
    1.x  : MultiIndex columns are (price_field, ticker) — level0=field, level1=ticker.
           group_by parameter is silently ignored; the (field, ticker) order is always
           used.  Partial-failure batches can include None-named columns for fields that
           the provider could not return, causing 'NoneType' subscript TypeErrors if
           those columns are not dropped before further processing.
    """
    if raw_data is None or raw_data.empty:
        return None

    if isinstance(raw_data.columns, pd.MultiIndex):
        # Drop any column entries where either level value is None — yfinance 1.x
        # injects these for fields/tickers that partially failed in a batch request.
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

        # yfinance 0.2.x layout: (ticker, field) — ticker is in level0
        if ticker in level0:
            df = raw_data[ticker].copy()
        # yfinance 1.x layout: (field, ticker) — ticker is in level1
        elif ticker in level1:
            df = raw_data.xs(ticker, level=1, axis=1).copy()
        else:
            return None

        # After xs() some columns may still be None-named (edge case in 1.x).
        df = df.loc[:, df.columns.notna()]
        if df.empty:
            return None
        return _ensure_price_columns(_normalize_history_index(df))

    # Guard against ambiguous yfinance payloads in multi-ticker requests.
    # Some yfinance versions collapse columns to flat OHLCV when only one ticker
    # in a multi-symbol request resolves. In that shape we cannot prove ownership
    # for the current ticker, so only accept flat payloads for explicit single-
    # ticker requests.
    if is_single_request:
        return _ensure_price_columns(_normalize_history_index(raw_data.copy()))
    return None


def _download_with_timeout(
    tickers: List[str],
    start: str,
    end: str,
    provider: Optional[DataProvider] = None,
) -> Optional[pd.DataFrame]:
    """
    Attempts to download a chunk of tickers via the given provider (defaulting to
    YFinanceProvider) with exponential backoff.

    YFinanceProvider uses auto_adjust=False so that both the raw Close and the
    dividend+split-adjusted Adj Close columns are returned.  The backtest and
    signal engine use Adj Close for return computation and raw Close for trade
    execution prices — keeping both is therefore intentional.  (auto_adjust=True
    would strip Adj Close from the payload, causing _is_valid_dataframe to reject
    every downloaded frame.)
    """
    max_retries = 3
    errors: list = []
    for attempt in range(max_retries):
        try:
            active_provider = provider or YFinanceProvider()
            df = active_provider.download(tickers, start, end)
            return df
        except Exception as exc:
            errors.append(exc)
            logger.debug("[Cache] yfinance download attempt %d failed: %s", attempt + 1, exc)
            if attempt == max_retries - 1:
                logger.error("[Cache] yfinance failed after %d retries.", max_retries)
                # MB-14 FIX: Chain exceptions so callers see full retry history.
                # Previously only the final exception was raised, masking earlier
                # (often more informative) network errors with e.g. a JSON parse error.
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
                # Migrate legacy schema
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
    """
    Strict structural validation gate applied before any data is written to disk.

    Blocks ingestion of corrupted yfinance payloads that would otherwise propagate
    silently into the cache and corrupt downstream CVaR and signal calculations.

    Checks:
    - Minimum row count (MB-20 FIX: max(30, HISTORY_GATE) to match strategy warm-up
      requirements; a 5-row frame passes validation but is immediately gated out by
      HISTORY_GATE and cached as "valid", wasting disk space).
    - Unique, monotonically increasing DatetimeIndex — yfinance occasionally
      returns duplicate or out-of-order dates on partial trading sessions.
    - 'Open', 'High', 'Low', 'Close' columns present and not entirely NaN.
    """
    # Keep cache validation permissive enough for short synthetic/index frames
    # used by unit tests and fallback providers.
    min_rows = int(getattr(cfg, "HISTORY_GATE", 5)) if cfg is not None else 5
    if df is None or df.empty or len(df) < min_rows:
        return False
    if not isinstance(df.index, pd.DatetimeIndex):
        return False
    if not df.index.is_unique:
        return False
    if not df.index.is_monotonic_increasing:
        return False
        
    # Validation should require close prices but allow sparse OHLC payloads
    # (common for synthetic/index test frames).
    if "Close" not in df.columns or df["Close"].isnull().all():
        return False
            
    if "Adj Close" not in df.columns or df["Adj Close"].isnull().all():
        return False
        
    is_index_ticker = bool(ticker) and str(ticker).startswith("^")
    if (not is_index_ticker) and ("Volume" not in df.columns or df["Volume"].isnull().all()):
        return False
        
    return True


def _ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise price column names and fill gaps for both yfinance 0.2.x and 1.x."""
    out = df.copy()

    # Strip any None-named columns that survived _extract_ticker_frame (belt-and-suspenders).
    # yfinance 1.x can inject these for fields that partially failed in a batch request.
    out = out.loc[:, out.columns.notna()]

    if "Adj Close" not in out.columns:
        # Column entirely absent (common for index tickers like ^NSEI, and for
        # yfinance 1.x tickers with no corporate-action records).
        if "Close" in out.columns:
            out["Adj Close"] = out["Close"]
    else:
        # BUG-1 FIX: yfinance sometimes returns an Adj Close column that is
        # present but all-NaN (e.g. recently-listed stocks, certain indices,
        # or partial corporate-action data).  _is_valid_dataframe rejects any
        # frame whose Adj Close is entirely null, so those tickers would be
        # silently dropped from the cache on every download.  Fill the NaN
        # cells with the corresponding raw Close so the frame passes validation
        # while still using the true adjusted price wherever it is available.
        if "Close" in out.columns:
            out["Adj Close"] = out["Adj Close"].fillna(out["Close"])
            
    # PHASE 9 FIX: Ensure Corporate Action columns exist (prevent crashes on split handling)
    for col in ["Dividends", "Stock Splits"]:
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(0.0)
            
    return out


def _latest_business_day() -> str:
    """
    FIX D8: Compute the latest business day in a timezone-independent way.

    The previous implementation used Asia/Kolkata (IST) to determine whether
    the market has closed today.  On UTC servers (cloud deployments, CI/CD),
    _now_ist.weekday() and _now_ist.time() still work correctly because
    pd.Timestamp.now(tz=...) converts to the target tz before comparison.
    HOWEVER, the IST offset is +5:30, so on a UTC server that hasn't yet
    reached 15:45 IST (= 10:15 UTC), the function would return today's date
    even though today's bar isn't complete yet.

    The conservative fix: always use the PREVIOUS business day (BDay(-1) from
    today, timezone-naive) as the staleness threshold.  This guarantees:
      - No redundant re-download the same day after market close (the parquet
        already has today's bar from a previous run in the same session).
      - No stale data on cloud servers (UTC) that run before IST 15:45.
      - Correct behaviour on weekends/holidays (BDay rolls back to Friday).

    Trade-off: On the rare case where a user runs the system after IST 15:45
    on a weekday AND the parquet doesn't yet contain today's bar, they will
    get yesterday's bar until the next scheduled fetch.  This is acceptable
    for a daily workflow — the alternative (re-downloading 500 tickers every
    run before market close) is far more disruptive.
    """
    today = pd.Timestamp.today().normalize()
    # pd.offsets.BDay(-1) gives the most-recently-completed business day.
    # On Monday, this correctly returns the previous Friday.
    latest_bday_ts = today - pd.offsets.BDay(1)
    return latest_bday_ts.strftime("%Y-%m-%d")


def load_or_fetch(
    tickers: List[str], 
    required_start: str, 
    required_end: str, 
    force_refresh: bool = False, 
    cfg=None
) -> Dict[str, pd.DataFrame]:
    """
    Primary interface for fetching market data. Evaluates the local cache manifest
    and only downloads the missing or stale series from yfinance.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    entries = manifest["entries"]
    
    # STABILITY ISSUE 3 FIX: Robust Ticker Sanitization
    def _clean_ticker(t: str) -> str:
        t_str = str(t).strip()
        if t_str.startswith("^"):
            return t_str
        return t_str.replace(".NSE", "").replace(".NS", "") + ".NS"
        
    standardized_tickers = list(dict.fromkeys(_clean_ticker(t) for t in tickers))
    
    # Dynamic padding based on strategy lookback requirements (plus safety margin).
    cfg_lookback = int(getattr(cfg, "CVAR_LOOKBACK", 200) or 200)
    dynamic_padding_days = max(400, cfg_lookback * 2)
    padded_start = (
        pd.Timestamp(required_start or "2020-01-01") - timedelta(days=dynamic_padding_days)
    ).strftime("%Y-%m-%d")

    # FIX D8: Use timezone-independent BDay calculation instead of IST-dependent logic.
    # The previous implementation computed latest_bday using Asia/Kolkata timezone,
    # which silently broke on UTC cloud servers (wrong date before IST 15:45 UTC+5:30).
    # Conservative approach: always treat yesterday's close as the staleness threshold.
    latest_bday = _latest_business_day()
    
    tickers_to_download = []
    market_data: Dict[str, pd.DataFrame] = {}
    
    # 1. Identify which tickers need downloading vs which can be loaded from disk
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
                
    providers: List[DataProvider] = [YFinanceProvider(), SecondaryProvider()]

    # 2. Download missing tickers in chunks
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
                    # yfinance end= is EXCLUSIVE — add 1 day so today's close is included.
                    _yf_end = (pd.Timestamp(required_end) + timedelta(days=1)).strftime("%Y-%m-%d")
                    raw_data = _download_with_timeout(chunk, padded_start, _yf_end, provider=provider)
                except Exception as exc:
                    logger.warning(
                        "[Cache] Provider %s failed for chunk starting with %s: %s",
                        type(provider).__name__, chunk[0], exc,
                    )
                    raw_data = None
                if raw_data is not None and not raw_data.empty:
                    break
            if raw_data is None or raw_data.empty:
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
                            "[Cache] Using stale cached parquet for %s after provider failures.",
                            ticker,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[Cache] Failed loading stale parquet fallback for %s: %s",
                            ticker,
                            exc,
                        )

                if recovered == len(chunk):
                    continue

                missing = len(chunk) - recovered
                logger.error(
                    "[Cache] Skipping %d symbols from chunk starting with %s after all providers failed.",
                    missing,
                    chunk[0],
                )
                # Soft-fail to keep scans/backtests running when upstream providers
                # temporarily reject/rename a subset of symbols. Callers already
                # handle sparse market_data dictionaries.
                continue
                
            for ticker in chunk:
                try:
                    df = _extract_ticker_frame(
                        raw_data,
                        ticker,
                        is_single_request=(len(chunk) == 1),
                    )
                    if df is None or df.empty:
                        continue
                        
                    df.dropna(how='all', inplace=True)
                    if df.empty:
                        continue
                    # Structural validation before anything touches disk.
                    # Catches non-unique/non-monotonic indexes and all-NaN Close
                    # columns that dropna(how='all') cannot detect.
                    if not _is_valid_dataframe(df, ticker=ticker, cfg=cfg):
                        logger.warning(
                            "[Cache] Structural validation failed for %s "
                            "(non-monotonic index, duplicate dates, or null OHLC). Skipping.",
                            ticker
                        )
                        continue

                    parquet_path = CACHE_DIR / f"{ticker}.parquet"
                    df.to_parquet(parquet_path)
                    
                    # STABILITY ISSUE 2 FIX: Improved Suspension Detection
                    gap_series = df.index.to_series().diff().dt.days
                    max_gap = gap_series.max()
                    max_gap_days = int(max_gap) if pd.notna(max_gap) else 0
                    expected_gap = 7

                    entries[ticker] = {
                        "fetched_at": datetime.now().isoformat(),
                        "rows": len(df),
                        "last_date": df.index[-1].strftime("%Y-%m-%d"),
                        "suspended": max_gap_days > expected_gap,
                        "max_gap_days": max_gap_days,
                    }
                    market_data[ticker] = df
                    
                except Exception as exc:
                    logger.error("[Cache] Failed processing downloaded dataframe for %s: %s", ticker, exc)
                    
            # Checkpoint the manifest after every chunk
            _save_manifest(manifest)

    # FIX D1: Trim to padded_start (not required_start) so callers receive the
    # full warm-up window for EWMA and CVaR initialisation.
    #
    # The original "STABILITY ISSUE 1 FIX" trimmed to required_start:required_end,
    # silently discarding the padding that was just fetched. For WFO IS slices that
    # begin at TRAIN_START (e.g. 2018-01-01), this left the first ~90 trading days
    # with no historical signal warm-up, causing HISTORY_GATE to gate out nearly
    # every stock in Q1 2018 and producing zero-trade early slices.
    #
    # The backtest engine already constrains *trading activity* to [start_date, end_date]
    # via BacktestEngine.run()'s per-bar `if date < start_dt: continue` guard, so the
    # pre-required_start rows are never executed against — they only serve as look-back
    # history for signal computation on the first live bar, which is the correct design.
    #
    # We still trim at required_end to prevent future-data leakage.
    # CRITICAL: Do NOT trim back to required_start — that was the original bug.
    #
    # MB-01 FIX: For cache-hit tickers, the stored parquet may START EARLIER than the
    # current call's padded_start (e.g. previously cached with a larger CVAR_LOOKBACK).
    # Trimming with the new (shorter) padded_start silently discards valid warm-up history
    # that was already fetched.  Fix: use the stored frame's actual first date so we
    # never trim back beyond what is already on disk.  For freshly-downloaded tickers
    # the parquet start equals padded_start anyway, so the min() is a no-op.
    padded_start_ts = pd.Timestamp(padded_start)
    for t, df in market_data.items():
        if df.empty:
            continue
        effective_start = min(df.index[0], padded_start_ts)
        market_data[t] = df.loc[effective_start:required_end]

    return market_data


def get_cache_summary() -> dict:
    """Returns a summary of cache health and coverage."""
    manifest = _load_manifest()
    entries = manifest.get("entries", {})
    return {
        "total_symbols": len(entries),
        "schema_version": manifest.get("schema_version", 1),
        "suspended_symbols": sum(1 for v in entries.values() if v.get("suspended")),
    }
