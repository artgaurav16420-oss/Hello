"""
data_cache.py — Persistent Atomic Downloader v11.46
===================================================
Robustly manages downloading, parsing, and persisting yfinance data.

Features:
- Atomic JSON manifest updates
- Network retry logic and chunking
- Persistent storage of raw provider data only (no synthetic mutation at cache layer)
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
        )


class SecondaryProvider(DataProvider):
    """AlphaVantage fallback provider with basic rate-limit throttling."""

    _URL = "https://www.alphavantage.co/query"

    def __init__(self) -> None:
        # Read configuration once at construction time.  Environment variable
        # lookups are system calls; calling them on every download() invocation
        # is wasteful and makes state unpredictable if the env changes mid-run.
        self.api_key     = os.getenv("FALLBACK_API_KEY", "").strip()
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
    """Return a copy with a timezone-naive, monotonic DatetimeIndex."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_localize(None)
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
    - 'Close' column present and not entirely NaN — the only column the entire
      engine is guaranteed to use; a fully-null Close is unusable.
    """
    if df is None or df.empty or len(df) < 5:
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
    if "Adj Close" not in out.columns:
        # Column entirely absent (common for index tickers like ^NSEI).
        out["Adj Close"] = out["Close"]
    else:
        # BUG-1 FIX: yfinance sometimes returns an Adj Close column that is
        # present but all-NaN (e.g. recently-listed stocks, certain indices,
        # or partial corporate-action data).  _is_valid_dataframe rejects any
        # frame whose Adj Close is entirely null, so those tickers would be
        # silently dropped from the cache on every download.  Fill the NaN
        # cells with the corresponding raw Close so the frame passes validation
        # while still using the true adjusted price wherever it is available.
        out["Adj Close"] = out["Adj Close"].fillna(out["Close"])
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
    
    # Ensure standard NSE suffix formatting
    standardized_tickers = list(dict.fromkeys(
        t if (t.endswith(".NS") or t.startswith("^")) else t + ".NS" 
        for t in tickers
    ))
    
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
        latest_bday = (pd.Timestamp.today() - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    
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
                # BUG-3 FIX: The previous code caught TypeError to handle a
                # hypothetical missing 'provider' kwarg — but _download_with_timeout
                # has always accepted that kwarg, so TypeError was never raised from
                # the call itself.  Worse, the fallback path re-called the function
                # WITHOUT provider, silently retrying YFinance instead of advancing
                # to AlphaVantage.  Any non-TypeError exception (ConnectionError,
                # Timeout, etc.) also propagated uncaught and crashed the entire
                # load_or_fetch call for the chunk.
                # Fix: catch all exceptions from a provider attempt, log them, and
                # move on to the next provider in the chain.
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
                            "(non-monotonic index, duplicate dates, or null Close). Skipping.",
                            ticker
                        )
                        continue

                    parquet_path = CACHE_DIR / f"{ticker}.parquet"
                    df.to_parquet(parquet_path)
                    
                    entries[ticker] = {
                        "fetched_at": datetime.now().isoformat(),
                        "rows": len(df),
                        "last_date": df.index[-1].strftime("%Y-%m-%d"),
                        "suspended": False,
                        "max_gap_days": 0,
                    }
                    market_data[ticker] = df
                    
                except Exception as exc:
                    logger.error("[Cache] Failed processing downloaded dataframe for %s: %s", ticker, exc)
                    
            # Checkpoint the manifest after every chunk
            _save_manifest(manifest)
            
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
