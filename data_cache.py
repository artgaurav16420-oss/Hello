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
from datetime import datetime, time as dt_time, timedelta
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
        return yf.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            progress=False,
            auto_adjust=False,
            actions=True,  # PHASE 9 FIX: Ensure Corporate Actions (Dividends/Splits) are downloaded
        )


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
        end_ts = pd.Timestamp(end)
        rows = []
        for d, item in series.items():
            date = pd.Timestamp(d)
            if date < start_ts or date > end_ts:
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
            out.index = out.index.tz_convert(None)
        out.index = out.index.normalize()
        
        # Remove duplicate dates (keep last reported)
        if out.index.duplicated().any():
            out = out[~out.index.duplicated(keep='last')]
            
        # Sort chronologically strictly
        out = out.sort_index()
        
    return out


def _extract_ticker_frame(raw_data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Robustly extract one ticker frame from varying yfinance payload shapes."""
    if raw_data is None or raw_data.empty:
        return None

    if isinstance(raw_data.columns, pd.MultiIndex):
        level0 = set(raw_data.columns.get_level_values(0))
        level1 = set(raw_data.columns.get_level_values(1))

        if ticker in level0:
            df = raw_data[ticker].copy()
        elif ticker in level1:
            df = raw_data.xs(ticker, level=1, axis=1).copy()
        else:
            return None
        return _ensure_price_columns(_normalize_history_index(df))

    # Single ticker payloads often come as flat OHLCV columns
    return _ensure_price_columns(_normalize_history_index(raw_data.copy()))


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
    for attempt in range(max_retries):
        try:
            active_provider = provider or YFinanceProvider()
            df = active_provider.download(tickers, start, end)
            return df
        except Exception as exc:
            logger.debug("[Cache] yfinance download attempt %d failed: %s", attempt + 1, exc)
            if attempt == max_retries - 1:
                logger.error("[Cache] yfinance failed after %d retries.", max_retries)
                raise exc
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


def _is_valid_dataframe(df: pd.DataFrame, ticker: Optional[str] = None) -> bool:
    """
    Strict structural validation gate applied before any data is written to disk.

    Blocks ingestion of corrupted yfinance payloads that would otherwise propagate
    silently into the cache and corrupt downstream CVaR and signal calculations.

    Checks:
    - Minimum row count (5) to exclude stub responses.
    - Unique, monotonically increasing DatetimeIndex — yfinance occasionally
      returns duplicate or out-of-order dates on partial trading sessions.
    - 'Open', 'High', 'Low', 'Close' columns present and not entirely NaN.
    """
    if df is None or df.empty or len(df) < 5:
        return False
    if not isinstance(df.index, pd.DatetimeIndex):
        return False
    if not df.index.is_unique:
        return False
    if not df.index.is_monotonic_increasing:
        return False
        
    # STABILITY ISSUE 4 FIX: Ensure all execution-critical OHLC columns exist
    required_cols = ["Open", "High", "Low", "Close"]
    for col in required_cols:
        if col not in df.columns or df[col].isnull().all():
            return False
            
    if "Adj Close" not in df.columns or df["Adj Close"].isnull().all():
        return False
        
    is_index_ticker = bool(ticker) and str(ticker).startswith("^")
    if (not is_index_ticker) and ("Volume" not in df.columns or df["Volume"].isnull().all()):
        return False
        
    return True


def _ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Adj Close" not in out.columns:
        # Column entirely absent (common for index tickers like ^NSEI).
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
    
    # Latest valid business day
    # Use today as the staleness threshold if:
    #   (a) today is a weekday, AND
    #   (b) current IST time is past 15:45 (NSE close + 15 min settlement buffer)
    # Otherwise fall back to the previous business day so pre-open runs
    # don't force a redundant full re-download.
    _now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    _market_closed_today = (
        _now_ist.weekday() < 5  # Mon–Fri
        and _now_ist.time() >= dt_time(15, 45)
    )
    if _market_closed_today:
        latest_bday = _now_ist.strftime("%Y-%m-%d")
    else:
        latest_bday = (_now_ist - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    
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

                raise DataFetchError(
                    f"Failed to fetch data for chunk starting with {chunk[0]} after all providers."
                )
                
            for ticker in chunk:
                try:
                    df = _extract_ticker_frame(raw_data, ticker)
                    if df is None or df.empty:
                        continue
                        
                    df.dropna(how='all', inplace=True)
                    if df.empty:
                        continue
                    # Structural validation before anything touches disk.
                    # Catches non-unique/non-monotonic indexes and all-NaN Close
                    # columns that dropna(how='all') cannot detect.
                    if not _is_valid_dataframe(df, ticker=ticker):
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
    for t, df in market_data.items():
        market_data[t] = df.loc[padded_start:required_end]

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