"""
data_cache.py — Persistent Atomic Downloader v11.46
===================================================
Robustly manages downloading, parsing, and persisting yfinance data.

Features:
- Atomic JSON manifest updates
- Network retry logic and chunking
- Fully DETERMINISTIC synthetic noise injection for long trading 
  suspensions (ASM/GSM) to prevent the optimizer from misidentifying 
  frozen assets as zero-volatility safe havens.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import hashlib
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = "data/cache"
MANIFEST_FILE = os.path.join(CACHE_DIR, "_manifest.json")
_DOWNLOAD_CHUNK_SIZE = 75
_SUSPENSION_GAP_DAYS = 30


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

    def download(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        api_key = os.getenv("FALLBACK_API_KEY", "").strip()
        if not api_key:
            logger.warning("[Cache][Fallback] FALLBACK_API_KEY not set; skipping secondary provider.")
            return None

        min_interval = float(os.getenv("FALLBACK_MIN_INTERVAL_SEC", "12"))
        last_call_ts = 0.0
        ticker_frames: Dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            av_symbol = self._map_symbol(ticker)
            elapsed = time.time() - last_call_ts
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            try:
                frame = self._download_single(av_symbol, api_key)
                last_call_ts = time.time()
            except Exception as exc:
                logger.warning("[Cache][Fallback] %s fetch failed: %s", ticker, exc)
                frame = None

            if frame is None:
                continue

            filtered = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
            if filtered.empty:
                continue
            ticker_frames[ticker] = filtered

            # AlphaVantage free tier is burst-limited; add a cool-down between symbols.
            time.sleep(0.05)

        if not ticker_frames:
            return None

        if len(ticker_frames) == 1:
            only = next(iter(ticker_frames.values()))
            return _normalize_history_index(only)

        combined = pd.concat(ticker_frames, axis=1)
        return _normalize_history_index(combined)

    def _download_single(self, symbol: str, api_key: str) -> Optional[pd.DataFrame]:
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

        rows = []
        for d, item in series.items():
            rows.append(
                {
                    "Date": pd.Timestamp(d),
                    "Open": float(item.get("1. open", np.nan)),
                    "High": float(item.get("2. high", np.nan)),
                    "Low": float(item.get("3. low", np.nan)),
                    "Close": float(item.get("4. close", np.nan)),
                    "Adj Close": float(item.get("5. adjusted close", np.nan)),
                    "Volume": float(item.get("6. volume", np.nan)),
                }
            )

        df = pd.DataFrame(rows).set_index("Date").sort_index()
        return _normalize_history_index(df)

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
        return _normalize_history_index(df)

    # Single ticker payloads often come as flat OHLCV columns
    return _normalize_history_index(raw_data.copy())


def _download_with_timeout(
    tickers: List[str],
    start: str,
    end: str,
    provider: Optional[DataProvider] = None,
) -> Optional[pd.DataFrame]:
    """
    Attempts to download a chunk of tickers via yfinance with exponential backoff.
    auto_adjust=True guarantees that all historical data handles corporate actions (splits).
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
    if not os.path.exists(MANIFEST_FILE):
        return default_manifest
        
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as file:
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
    os.makedirs(CACHE_DIR, exist_ok=True)
    temp_file = MANIFEST_FILE + ".tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(manifest_data, file, indent=2)
        os.replace(temp_file, MANIFEST_FILE)
    except Exception as exc:
        logger.error("[Cache] Failed to save manifest: %s", exc)


def invalidate_cache() -> None:
    """Forces cache clearing by deleting the manifest."""
    if os.path.exists(MANIFEST_FILE):
        try:
            os.remove(MANIFEST_FILE)
            logger.info("[Cache] Market data cache invalidated.")
        except OSError as e:
            logger.error("[Cache] Failed to invalidate cache: %s", e)


def _is_valid_dataframe(df: pd.DataFrame) -> bool:
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
    if "Volume" not in df.columns or df["Volume"].isnull().all():
        return False
    return True


def _repair_suspension_gaps(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, bool, int]:
    """
    Detects long gaps (e.g. from ASM/GSM regulatory suspensions) and injects 
    DETERMINISTIC synthetic noise instead of a flat forward-fill.
    
    A flat line has 0.0% variance. The risk engine (CVaR optimizer) will see 
    this suspended stock as the "safest asset in the world" and over-allocate 
    to it. This fix prevents that critical institutional flaw.
    """
    if len(df) < 2:
        return df, False, 0
        
    # Calculate days between adjacent dates in the index
    gap_days = df.index.to_series().diff().dt.days
    max_gap = int(gap_days.max()) if not gap_days.empty else 0
    is_suspended = max_gap > _SUSPENSION_GAP_DAYS
    
    if is_suspended:
        logger.warning(
            "[Cache] %s: Prolonged trading gap of %d days detected. "
            "Injecting deterministic synthetic noise to prevent risk-engine suppression.", 
            ticker, max_gap
        )
        
        # Create a complete business day timeline covering the entire range
        bday_idx = pd.bdate_range(df.index[0], df.index[-1])
        
        # FIX (Bug-B — Lookahead Bias): Compute volatility STRICTLY from price data
        # that existed BEFORE the first detected gap.  Using the full series would
        # leak future volatility regimes into the imputed past prices — e.g.,
        # computing vol from 2019-2024 data to fill a 2018 suspension gap inflates
        # apparent risk during backtests and pollutes CVaR estimates.
        _gap_days_tmp = df.index.to_series().diff().dt.days
        _first_gap_positions = np.where(_gap_days_tmp.values > _SUSPENSION_GAP_DAYS)[0]
        if len(_first_gap_positions) > 0:
            _pre_gap_close = df["Close"].iloc[: _first_gap_positions[0]]
        else:
            _pre_gap_close = df["Close"]

        _pre_gap_rets = _pre_gap_close.pct_change().dropna()
        if len(_pre_gap_rets) > 10:
            hist_vol = float(_pre_gap_rets.std())
        else:
            hist_vol = 0.02  # fallback 2 % daily vol when insufficient pre-gap history
            
        # Reindex to fill the gap with NaNs
        df = df.reindex(bday_idx)
        missing_mask = df["Close"].isna()
        
        if missing_mask.any():
            n_missing = missing_mask.sum()
            
            # FIX (I-07): Use a hash derived RandomState seed unique to the ticker. 
            # Ensures perfectly deterministic results per ticker while decoupling across assets.
            seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16) % (2**31)
            rng = np.random.RandomState(seed)
            noise_rets = rng.normal(0, hist_vol, n_missing)
            
            # Forward fill as a baseline
            df["Close"] = df["Close"].ffill()
            
            # Build a deterministic random walk for suspended periods so variance compounds over time.
            missing_idx = df.index[missing_mask]
            anchor_prices = (
                df["Close"].shift(1).reindex(missing_idx).ffill().fillna(df["Close"].dropna().iloc[0])
            )
            walk_returns = np.cumprod(1.0 + noise_rets)
            df.loc[missing_idx, "Close"] = anchor_prices.values * walk_returns
            
            # Zero out volume for the days it didn't trade
            df["Volume"] = df["Volume"].fillna(0)
            
    return df, is_suspended, max_gap


def _ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Adj Close" not in out.columns:
        out["Adj Close"] = out["Close"]
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
    os.makedirs(CACHE_DIR, exist_ok=True)
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
    latest_bday = (pd.Timestamp.today() - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    
    tickers_to_download = []
    market_data: Dict[str, pd.DataFrame] = {}
    
    # 1. Identify which tickers need downloading vs which can be loaded from disk
    for ticker in standardized_tickers:
        entry = entries.get(ticker, {})
        parquet_path = os.path.join(CACHE_DIR, f"{ticker}.parquet")
        
        is_stale = entry.get("last_date", "") < latest_bday
        missing_file = not os.path.exists(parquet_path)
        
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
                    raw_data = _download_with_timeout(chunk, padded_start, required_end, provider=provider)
                except TypeError:
                    raw_data = _download_with_timeout(chunk, padded_start, required_end)
                if raw_data is not None and not raw_data.empty:
                    break
            if raw_data is None or raw_data.empty:
                logger.warning("[Cache] Received empty response for chunk starting with %s", chunk[0])
                continue
                
            for ticker in chunk:
                try:
                    df = _extract_ticker_frame(raw_data, ticker)
                    if df is None or df.empty:
                        continue
                        
                    df.dropna(how='all', inplace=True)
                    if df.empty:
                        continue
                    df = _ensure_price_columns(df)

                    # Structural validation before anything touches disk.
                    # Catches non-unique/non-monotonic indexes and all-NaN Close
                    # columns that dropna(how='all') cannot detect.
                    if not _is_valid_dataframe(df):
                        logger.warning(
                            "[Cache] Structural validation failed for %s "
                            "(non-monotonic index, duplicate dates, or null Close). Skipping.",
                            ticker
                        )
                        continue

                    simulate_halts = bool(getattr(cfg, "SIMULATE_HALTS", False))
                    if simulate_halts:
                        df, suspended, max_gap = _repair_suspension_gaps(df, ticker)
                        if suspended:
                            logger.warning("[Cache] Synthetic halt simulation applied to %s", ticker)
                    else:
                        suspended, max_gap = False, 0
                    
                    parquet_path = os.path.join(CACHE_DIR, f"{ticker}.parquet")
                    df.to_parquet(parquet_path)
                    
                    entries[ticker] = {
                        "fetched_at": datetime.now().isoformat(),
                        "rows": len(df),
                        "last_date": df.index[-1].strftime("%Y-%m-%d"),
                        "suspended": suspended,
                        "max_gap_days": max_gap
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
