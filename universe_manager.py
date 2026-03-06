"""
universe_manager.py — Universe Fetching & Caching v11.48
========================================================
Robust fetching of NSE/Nifty 500 universes, sector mappings, and
point-in-time historical constituents to eliminate backtest survivorship bias.
Now strictly enforces operator awareness if historical data is missing
and robustly handles PyArrow/FastParquet list deserialization quirks.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR            = "data/cache"
UNIVERSE_CACHE_FILE  = os.path.join(CACHE_DIR, "_universe_cache.json")
UNIVERSE_CACHE_TTL_H = 72
_ADV_CHUNK_SIZE      = 75
_ADV_MAX_WORKERS     = 1

class UniverseFetchError(RuntimeError):
    """Raised when primary and secondary universe data sources fail."""
    def __init__(self, message: str):
        super().__init__(message)
        self.fallback_universe: List[str] = []

# Hardcoded fallback list if all network and cache layers fail.
_HARD_FLOOR_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "HINDUNILVR",
    "ITC", "SBIN", "LTIM", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA",
    "ADANIENT", "KOTAKBANK", "TITAN", "ONGC", "TATAMOTORS", "NTPC", "AXISBANK",
    "ADANIPORTS", "ASIANPAINT", "COALINDIA", "BAJAJFINSV", "JSWSTEEL",
    "M&M", "POWERGRID", "TATASTEEL", "ULTRACEMCO", "GRASIM", "HINDALCO", "NESTLEIND",
    "INDUSINDBK", "TECHM", "WIPRO", "CIPLA", "HDFCLIFE", "SBILIFE", "DRREDDY",
    "HEROMOTOCO", "EICHERMOT", "BPCL", "BAJAJ-AUTO", "BRITANNIA", "APOLLOHOSP",
    "DIVISLAB", "TATACONSUM",
]

# Static sector map for tier-1 liquid NSE names used as a deterministic
# fallback when network lookups are unavailable.
STATIC_NSE_SECTORS: Dict[str, str] = {
    "RELIANCE": "Energy",
    "TCS": "Information Technology",
    "HDFCBANK": "Financial Services",
    "ICICIBANK": "Financial Services",
    "INFY": "Information Technology",
    "BHARTIARTL": "Telecommunications",
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "SBIN": "Financial Services",
    "LTIM": "Information Technology",
    "BAJFINANCE": "Financial Services",
    "HCLTECH": "Information Technology",
    "MARUTI": "Automobile and Auto Components",
    "SUNPHARMA": "Healthcare",
    "ADANIENT": "Industrials",
    "KOTAKBANK": "Financial Services",
    "TITAN": "Consumer Durables",
    "ONGC": "Energy",
    "TATAMOTORS": "Automobile and Auto Components",
    "NTPC": "Power",
    "AXISBANK": "Financial Services",
    "ADANIPORTS": "Services",
    "ASIANPAINT": "Consumer Durables",
    "COALINDIA": "Energy",
    "BAJAJFINSV": "Financial Services",
    "JSWSTEEL": "Metals & Mining",
    "M&M": "Automobile and Auto Components",
    "POWERGRID": "Power",
    "TATASTEEL": "Metals & Mining",
    "ULTRACEMCO": "Construction Materials",
    "GRASIM": "Construction Materials",
    "HINDALCO": "Metals & Mining",
    "NESTLEIND": "FMCG",
    "INDUSINDBK": "Financial Services",
    "TECHM": "Information Technology",
    "WIPRO": "Information Technology",
    "CIPLA": "Healthcare",
    "HDFCLIFE": "Financial Services",
    "SBILIFE": "Financial Services",
    "DRREDDY": "Healthcare",
    "HEROMOTOCO": "Automobile and Auto Components",
    "EICHERMOT": "Automobile and Auto Components",
    "BPCL": "Energy",
    "BAJAJ-AUTO": "Automobile and Auto Components",
    "BRITANNIA": "FMCG",
    "APOLLOHOSP": "Healthcare",
    "DIVISLAB": "Healthcare",
    "TATACONSUM": "FMCG",
}

# ─── Historical Universe Logic (Survivorship Bias Fix) ────────────────────────


def _load_pit_universe_from_csv(universe_type: str, date: pd.Timestamp) -> List[str]:
    csv_path = f"data/historical_{universe_type}.csv"
    if not os.path.exists(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    date_col = "date" if "date" in df.columns else ("snapshot_date" if "snapshot_date" in df.columns else None)
    tick_col = "ticker" if "ticker" in df.columns else ("symbol" if "symbol" in df.columns else None)
    if date_col is None or tick_col is None:
        return []
    d = pd.Timestamp(date).normalize()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    subset = df[df[date_col] <= d]
    if subset.empty:
        return []
    last_d = subset[date_col].max()
    tickers = subset.loc[subset[date_col] == last_d, tick_col].dropna().astype(str).str.strip()
    return sorted({t for t in tickers if t})

# Module-level flags so each warning fires at most once per process,
# preventing thousands of identical lines from flooding optimizer output.
_MISSING_PARQUET_WARNED: Dict[str, bool] = {}
_NO_RECORD_WARNED: Dict[str, bool] = {}
_SECTOR_MAP_CACHE_LOCK = threading.Lock()


def get_historical_universe(universe_type: str, date: pd.Timestamp) -> List[str]:
    """
    Attempts to load the exact constituents for a specific historical date.

    Load order:
    1) Historical parquet snapshots.
    2) Point-in-time CSV snapshots (from historical_builder.py).

    If neither source has a valid record, returns an empty list and issues
    a survivorship-bias warning.
    """
    hist_file = f"data/historical_{universe_type}.parquet"

    # Warn once per missing parquet file (not once per date) to keep optimizer
    # output readable. The warning is still ERROR level so it's never silent.
    if not os.path.exists(hist_file):
        if not _MISSING_PARQUET_WARNED.get(universe_type):
            logger.error(
                "HISTORICAL PARQUET MISSING: %s — attempting point-in-time CSV fallback "
                "for %s. Run historical_builder.py to regenerate parquet snapshots.",
                hist_file, universe_type,
            )
            _MISSING_PARQUET_WARNED[universe_type] = True
    else:
        try:
            df = pd.read_parquet(hist_file)
            
            # Find the closest available manifest date preceding the requested date
            available_dates = df.index.unique()
            valid_dates = available_dates[available_dates <= date]
            
            if len(valid_dates) > 0:
                target_date = valid_dates.max()
                constituents = df.loc[target_date, "tickers"]

                def _coerce_members(value) -> List[str]:
                    if isinstance(value, str):
                        return [value]
                    if hasattr(value, "tolist"):
                        converted = value.tolist()
                        return converted if isinstance(converted, list) else [converted]
                    if isinstance(value, (list, tuple, set)):
                        return list(value)
                    return [value]

                # Handle duplicate snapshot rows by unioning all ticker lists deterministically.
                if isinstance(constituents, pd.Series):
                    merged: List[str] = []
                    for cell in constituents.values:
                        merged.extend(_coerce_members(cell))
                    return sorted({str(t).strip() for t in merged if str(t).strip()})

                return sorted({str(t).strip() for t in _coerce_members(constituents) if str(t).strip()})
            else:
                logger.warning(
                    "[Universe] No historical data prior to %s found in %s.", 
                    date.strftime("%Y-%m-%d"), hist_file
                )
        except Exception as exc:
            logger.error(
                "[Universe] Historical load failed for %s on %s: %s", 
                universe_type, date.strftime("%Y-%m-%d"), exc
            )
    
    # Fallback to point-in-time local CSV snapshot; never fallback to current constituents.
    csv_members = _load_pit_universe_from_csv(universe_type, date)
    if csv_members:
        return csv_members

    # Warn once only when both parquet and CSV are unavailable for the date.
    if not _NO_RECORD_WARNED.get(universe_type):
        logger.warning(
            "[Universe] %s: No point-in-time historical record found in parquet/CSV. "
            "Returning empty universe (survivorship bias risk if caller falls back).",
            universe_type,
        )
        _NO_RECORD_WARNED[universe_type] = True
    return []

# ─── Cache Management ─────────────────────────────────────────────────────────

def _load_universe_cache() -> dict:
    if not os.path.exists(UNIVERSE_CACHE_FILE):
        return {}
    try:
        with open(UNIVERSE_CACHE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        logger.warning("[Universe] Cache load failed, starting fresh: %s", exc)
        return {}

def _save_universe_cache(data: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    temp_file = UNIVERSE_CACHE_FILE + ".tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        os.replace(temp_file, UNIVERSE_CACHE_FILE)
    except Exception as exc:
        logger.error("[Universe] Failed to save cache: %s", exc)

def invalidate_universe_cache() -> None:
    """Force clears the local universe JSON cache."""
    if os.path.exists(UNIVERSE_CACHE_FILE):
        try:
            os.remove(UNIVERSE_CACHE_FILE)
            logger.info("[Universe] Cache invalidated.")
        except OSError as e:
            logger.error("[Universe] Failed to invalidate cache: %s", e)

# ─── Network Fetchers ─────────────────────────────────────────────────────────

def _fetch_csv_with_headers(url: str, timeout: float = 15.0) -> pd.DataFrame:
    """Helper to fetch NSE CSVs masking as a standard browser."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/csv,application/csv",
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))

def fetch_nse_equity_universe(cfg=None) -> List[str]:
    """Fetches the entire NSE actively traded equity list, minus illiquid names."""
    cache = _load_universe_cache()
    entry = cache.get("total_equity", {})
    
    if entry:
        fetched_time = datetime.fromisoformat(entry["fetched_at"])
        if datetime.now() - fetched_time < timedelta(hours=UNIVERSE_CACHE_TTL_H):
            return entry["tickers"]
            
    try:
        logger.info("[Universe] Fetching fresh NSE total equity master...")
        df = _fetch_csv_with_headers("https://archives.nseindia.com/content/equities/EQUITY_L.csv")
        df.columns = [col.strip().upper() for col in df.columns]
        
        # Filter for active Equity series only (exclude ETFs, bonds, etc)
        equity_df = df[df["SERIES"] == "EQ"]
        tickers = equity_df["SYMBOL"].unique().tolist()
        
        # Apply strict ADV liquidity filter
        from signals import _apply_adv_filter
        logger.info("[Universe] Applying liquidity filters to %d symbols...", len(tickers))
        liquid_tickers = _apply_adv_filter(tickers, cfg)
        
        cache["total_equity"] = {
            "fetched_at": datetime.now().isoformat(),
            "tickers": liquid_tickers
        }
        _save_universe_cache(cache)
        return liquid_tickers
        
    except Exception as exc:
        logger.error("[Universe] NSE master fetch failed: %s", exc)
        if entry:
            logger.warning("[Universe] Using stale cache for NSE Total Equity.")
            return entry["tickers"]
            
        error = UniverseFetchError("Failed to fetch NSE Total Equity from origin.")
        error.fallback_universe = list(_HARD_FLOOR_UNIVERSE)
        raise error

def get_nifty500() -> List[str]:
    """Fetches the current Nifty 500 index constituents."""
    cache = _load_universe_cache()
    entry = cache.get("nifty500", {})
    
    if entry:
        fetched_time = datetime.fromisoformat(entry["fetched_at"])
        if datetime.now() - fetched_time < timedelta(hours=UNIVERSE_CACHE_TTL_H):
            return entry["tickers"]
            
    try:
        logger.info("[Universe] Fetching fresh Nifty 500 constituents...")
        df = _fetch_csv_with_headers("https://archives.nseindia.com/content/indices/ind_nifty500list.csv")
        df.columns = [col.strip().upper() for col in df.columns]
        
        tickers = df["SYMBOL"].unique().tolist()
        
        cache["nifty500"] = {
            "fetched_at": datetime.now().isoformat(),
            "tickers": tickers
        }
        _save_universe_cache(cache)
        return tickers
        
    except Exception as exc:
        logger.error("[Universe] Nifty 500 fetch failed: %s", exc)
        if entry:
            logger.warning("[Universe] Using stale cache for Nifty 500.")
            return entry["tickers"]
            
        error = UniverseFetchError("Failed to fetch Nifty 500 from origin.")
        error.fallback_universe = list(_HARD_FLOOR_UNIVERSE)
        raise error

def get_sector_map(tickers: List[str], use_cache: bool = True, cfg=None) -> Dict[str, str]:
    """
    Retrieves sector classifications for a list of tickers.
    Uses static fallback mapping, then local cache, then threads out to yfinance.
    """
    resolved_map = {}
    missing_tickers = []
    
    # 1. Resolve via static hardcoded map
    for ticker in tickers:
        bare_ticker = ticker.replace(".NS", "")
        if bare_ticker in STATIC_NSE_SECTORS:
            resolved_map[bare_ticker] = STATIC_NSE_SECTORS[bare_ticker]
        else:
            missing_tickers.append(bare_ticker)
            
    # 2. Resolve via JSON cache
    if missing_tickers and use_cache:
        cache = _load_universe_cache()
        sector_cache = cache.get("sector_map", {}).get("sectors", {})
        
        still_missing = []
        for bare_ticker in missing_tickers:
            if bare_ticker in sector_cache:
                resolved_map[bare_ticker] = sector_cache[bare_ticker]
            else:
                still_missing.append(bare_ticker)
        missing_tickers = still_missing
        
    # 3. Resolve via yfinance network fetch
    if missing_tickers:
        logger.info("[Universe] Fetching sector data for %d missing tickers...", len(missing_tickers))
        import yfinance as yf
        
        def _fetch_single_sector(sym: str) -> Tuple[str, str]:
            try:
                ns_sym = sym + ".NS"
                ticker_obj = yf.Ticker(ns_sym)
                info = ticker_obj.info
                # Some assets are ETFs or missing standard equity fields
                sector = info.get("sector", "Unknown")
                return sym, sector
            except Exception as e:
                logger.debug("Failed to fetch sector for %s: %s", sym, e)
                return sym, "Unknown"
                
        # Threaded fetch to overcome network latency
        with ThreadPoolExecutor(max_workers=8) as pool:
            future_to_sym = {pool.submit(_fetch_single_sector, sym): sym for sym in missing_tickers}
            for future in as_completed(future_to_sym):
                sym, sector = future.result()
                resolved_map[sym] = sector
                
        # Update cache with newly found sectors
        if use_cache:
            with _SECTOR_MAP_CACHE_LOCK:
                cache = _load_universe_cache()
                existing_sector_cache = cache.get("sector_map", {}).get("sectors", {})
                existing_sector_cache.update({sym: resolved_map[sym] for sym in missing_tickers})

                cache["sector_map"] = {
                    "fetched_at": datetime.now().isoformat(),
                    "sectors": existing_sector_cache
                }
                _save_universe_cache(cache)
            
    # Format the return dictionary to match exactly the requested input tickers
    final_map = {}
    for ticker in tickers:
        bare_ticker = ticker.replace(".NS", "")
        final_map[ticker] = resolved_map.get(bare_ticker, "Unknown")
        
    return final_map
