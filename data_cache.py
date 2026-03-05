"""
data_cache.py — Persistent Atomic Downloader v11.45
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
from concurrent.futures import ThreadPoolExecutor, TimeoutError
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


def _download_with_timeout(tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Attempts to download a chunk of tickers via yfinance with exponential backoff.
    auto_adjust=True guarantees that all historical data handles corporate actions (splits).
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = yf.download(
                tickers, 
                start=start, 
                end=end, 
                group_by="ticker", 
                progress=False, 
                auto_adjust=True
            )
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
        
        # Capture historical volatility before the gap if possible
        daily_rets = df["Close"].pct_change().dropna()
        if len(daily_rets) > 10:
            hist_vol = daily_rets.std()
        else:
            hist_vol = 0.02 # fallback to 2% daily volatility
            
        # Reindex to fill the gap with NaNs
        df = df.reindex(bday_idx)
        missing_mask = df["Close"].isna()
        
        if missing_mask.any():
            n_missing = missing_mask.sum()
            
            # FIX: Use seeded RandomState for exact determinism across live/backtest
            rng = np.random.RandomState(42)
            noise_rets = rng.normal(0, hist_vol, n_missing)
            
            # Forward fill as a baseline
            df["Close"] = df["Close"].ffill()
            
            # Perturb the baseline with the synthetic noise so the price isn't perfectly flat.
            df.loc[missing_mask, "Close"] *= (1.0 + noise_rets)
            
            # Zero out volume for the days it didn't trade
            df["Volume"] = df["Volume"].fillna(0)
            
    return df, is_suspended, max_gap


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
    
    # We always pad the required start date back by ~400 days to guarantee
    # we have enough history for 200-day SMAs and long-term CVaR lookbacks.
    padded_start = (pd.Timestamp(required_start or "2020-01-01") - timedelta(days=400)).strftime("%Y-%m-%d")
    
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
                df = pd.read_parquet(parquet_path)
                market_data[ticker] = df
            except Exception as exc:
                logger.debug("[Cache] Corrupted parquet for %s: %s", ticker, exc)
                tickers_to_download.append(ticker)
                
    # 2. Download missing tickers in chunks
    if tickers_to_download:
        logger.info("[Cache] Initiating download for %d missing/stale symbols.", len(tickers_to_download))
        chunks = [
            tickers_to_download[i:i + _DOWNLOAD_CHUNK_SIZE] 
            for i in range(0, len(tickers_to_download), _DOWNLOAD_CHUNK_SIZE)
        ]
        
        for chunk in chunks:
            raw_data = _download_with_timeout(chunk, padded_start, required_end)
            if raw_data is None or raw_data.empty:
                logger.warning("[Cache] Received empty response for chunk starting with %s", chunk[0])
                continue
                
            is_multi_index = isinstance(raw_data.columns, pd.MultiIndex)
            
            for ticker in chunk:
                try:
                    if is_multi_index:
                        # yfinance multi-index slicing
                        df = raw_data[ticker].copy()
                    else:
                        # if the chunk only contained one valid ticker
                        df = raw_data.copy()
                        
                    if df is None or df.empty:
                        continue
                        
                    df.dropna(how='all', inplace=True)
                    if df.empty:
                        continue
                        
                    # Fix: Handle suspension gaps with deterministic noise
                    df, suspended, max_gap = _repair_suspension_gaps(df, ticker)
                    
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