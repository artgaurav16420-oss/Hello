"""
universe_manager.py — Universe Fetching & Caching v11.46
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR             = Path("data")
CACHE_DIR            = DATA_DIR / "cache"
UNIVERSE_CACHE_FILE  = CACHE_DIR / "_universe_cache.json"
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
    csv_path = DATA_DIR / f"historical_{universe_type}.csv"
    if not csv_path.exists():
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
_HISTORICAL_UNIVERSE_DF_CACHE: Dict[Path, Tuple[float, pd.DataFrame]] = {}


def _load_historical_universe_df(hist_file: Path) -> pd.DataFrame:
    """Load historical universe parquet with mtime-based in-memory caching."""
    mtime = hist_file.stat().st_mtime
    cached = _HISTORICAL_UNIVERSE_DF_CACHE.get(hist_file)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    df = pd.read_parquet(hist_file)
    _HISTORICAL_UNIVERSE_DF_CACHE[hist_file] = (mtime, df)
    return df


def get_historical_universe(universe_type: str, date: pd.Timestamp) -> List[str]:
    """
    Attempts to load the exact constituents for a specific historical date.

    Load order:
    1) Historical parquet snapshots.
    2) Point-in-time CSV snapshots (from historical_builder.py).

    If neither source has a valid record, returns an empty list and issues
    a survivorship-bias warning.
    """
    # ── PHASE 3 FIX: Survivorship Bias Prevention ──
    # Do not allow present-day custom screeners to be mapped backwards in time.
    if universe_type.lower() == "custom":
        raise ValueError(
            f"Survivorship Bias Guard: Cannot historically backtest a 'custom' screener list "
            f"at {date.strftime('%Y-%m-%d')} without explicit historical constituent maps. "
            "The current custom list only contains assets that survived to present day, "
            "which would introduce severe survivorship bias into the simulation."
        )

    hist_file = DATA_DIR / f"historical_{universe_type}.parquet"

    # Warn once per missing parquet file (not once per date) to keep optimizer
    # output readable. The warning is still ERROR level so it's never silent.
    if not hist_file.exists():
        if not _MISSING_PARQUET_WARNED.get(universe_type):
            logger.error(
                "HISTORICAL PARQUET MISSING: %s — attempting point-in-time CSV fallback "
                "for %s. Run historical_builder.py to regenerate parquet snapshots.",
                hist_file, universe_type,
            )
            _MISSING_PARQUET_WARNED[universe_type] = True
    else:
        try:
            df = _load_historical_universe_df(hist_file)
            
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

    logger.error(
        "[Universe] %s: No point-in-time historical record found in parquet/CSV. "
        "Returning empty universe (survivorship bias risk if caller falls back).",
        universe_type,
    )
    _NO_RECORD_WARNED[universe_type] = True
    return []

# ─── Cache Management ─────────────────────────────────────────────────────────

def _load_universe_cache() -> dict:
    if not UNIVERSE_CACHE_FILE.exists():
        return {}
    try:
        with UNIVERSE_CACHE_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        logger.warning("[Universe] Cache load failed, starting fresh: %s", exc)
        return {}

def _save_universe_cache(data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = UNIVERSE_CACHE_FILE.with_name(UNIVERSE_CACHE_FILE.name + ".tmp")
    try:
        with temp_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        temp_file.replace(UNIVERSE_CACHE_FILE)
    except Exception as exc:
        logger.error("[Universe] Failed to save cache: %s", exc)

def invalidate_universe_cache() -> None:
    """Force clears the local universe JSON cache."""
    if UNIVERSE_CACHE_FILE.exists():
        try:
            UNIVERSE_CACHE_FILE.unlink()
            logger.info("[Universe] Cache invalidated.")
        except OSError as e:
            logger.error("[Universe] Failed to invalidate cache: %s", e)

# ─── ADV Liquidity Filter ─────────────────────────────────────────────────────

def _apply_adv_filter(tickers: List[str], cfg=None) -> List[str]:
    """
    Filter a raw list of tickers down to those meeting the minimum ADV
    liquidity threshold defined in cfg.MIN_ADV_CRORES.

    Moved here from signals.py — this is a universe curation concern, not a
    signal computation concern.  Keeping it alongside the other universe
    management utilities avoids a conceptual mismatch and removes the
    asymmetric import (signals → universe_manager was already the wrong
    direction; the correct dependency edge is universe_manager → signals for
    compute_single_adv).

    BUG FIX (bare-symbol return):
    The previous implementation looked up market data with to_ns(symbol) —
    correctly obtaining a ".NS"-suffixed key — but then appended the original
    bare symbol to filtered_tickers.  Callers expecting ".NS"-suffixed tickers
    back (e.g. the cache's manifest-keyed entries) received bare names like
    "RELIANCE" instead of "RELIANCE.NS", causing silent lookup misses
    downstream.  This version always appends the normalised ns_sym.
    """
    from momentum_engine import UltimateConfig, to_ns
    from data_cache import load_or_fetch
    from signals import compute_single_adv

    if cfg is None:
        cfg = UltimateConfig()

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=max(150, int(cfg.ADV_LOOKBACK) * 2))).strftime("%Y-%m-%d")

    chunk_size  = _ADV_CHUNK_SIZE   # FIX M2: use module constant; hardcoded duplicate ignored it
    chunks      = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

    filtered_tickers: List[str] = []
    min_adv_volume = cfg.MIN_ADV_CRORES * 1e7

    logger.info(
        "[Universe] Filtering %d tickers against ₹%dCr ADV minimum...",
        len(tickers), cfg.MIN_ADV_CRORES,
    )

    # Sequential loop intentionally retained — parallel yfinance calls on a
    # large universe trigger Yahoo Finance rate-limiting (HTTP 429 / 401
    # "Invalid Crumb"), causing entire chunks to return empty and incorrectly
    # disqualifying hundreds of valid liquid stocks.  Sequential processing
    # (~30s for 500 tickers) produces reliable, complete ADV results.
    chunk_failures: List[Dict[str, object]] = []

    for chunk_idx, chunk in enumerate(chunks):
        try:
            data = load_or_fetch(chunk, start_date, end_date, cfg=cfg)
            for symbol in chunk:
                # to_ns() is idempotent — handles both "RELIANCE" and
                # "RELIANCE.NS" inputs without producing a double-suffix.
                ns_sym = to_ns(symbol)
                if ns_sym in data:
                    adv = compute_single_adv(data[ns_sym], cfg=cfg)
                    if adv >= min_adv_volume:
                        # FIX: append ns_sym (the normalised ".NS" key), NOT
                        # the original bare symbol.  The previous code appended
                        # `symbol` here, which silently returned bare ticker
                        # strings to callers that expected ".NS"-suffixed names,
                        # causing downstream cache-key mismatches.
                        filtered_tickers.append(ns_sym)
        except Exception as exc:
            failure = {
                "chunk_index": chunk_idx,
                "symbols": list(chunk),
                "error": str(exc),
            }
            chunk_failures.append(failure)
            logger.error(
                "[Universe] Error processing ADV chunk %d (size=%d): %s",
                chunk_idx,
                len(chunk),
                exc,
            )

    if chunk_failures:
        failed_symbol_preview = [
            symbol
            for failure in chunk_failures
            for symbol in failure["symbols"][:3]
        ][:6]
        preview_txt = ", ".join(failed_symbol_preview) if failed_symbol_preview else "n/a"
        raise UniverseFetchError(
            "ADV filter failed for "
            f"{len(chunk_failures)} chunk(s); sample symbols: {preview_txt}"
        )

    return list(dict.fromkeys(filtered_tickers))


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
        
        # Apply strict ADV liquidity filter — defined in this module.
        logger.info("[Universe] Applying liquidity filters to %d symbols...", len(tickers))
        liquid_tickers = _apply_adv_filter(tickers, cfg)
        
        cache["total_equity"] = {
            "fetched_at": datetime.now().isoformat(),
            "tickers": liquid_tickers
        }
        _save_universe_cache(cache)
        return liquid_tickers
        
    except UniverseFetchError as exc:
        logger.error("[Universe] NSE master fetch failed during ADV filtering: %s", exc)
        if entry:
            logger.warning("[Universe] Using stale cache for NSE Total Equity.")
            return entry["tickers"]

        exc.fallback_universe = list(_HARD_FLOOR_UNIVERSE)
        raise
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

        try:
            batch = yf.Tickers(" ".join(f"{sym}.NS" for sym in missing_tickers))
            ticker_objs = getattr(batch, "tickers", {}) or {}
            for bare_sym in missing_tickers:
                ns_sym = f"{bare_sym}.NS"
                ticker_obj = ticker_objs.get(ns_sym)
                if ticker_obj is None:
                    ticker_obj = yf.Ticker(ns_sym)
                sector = "Unknown"
                try:
                    info = getattr(ticker_obj, "info", {}) or {}
                    sector = str(info.get("sector", "Unknown") or "Unknown")
                except Exception as e:
                    logger.debug("Failed to fetch sector for %s: %s", bare_sym, e)
                resolved_map[bare_sym] = sector
        except Exception as exc:
            logger.warning("[Universe] Batch sector fetch failed (%s). Falling back to threaded lookup.", exc)

            def _fetch_single_sector(sym: str) -> Tuple[str, str]:
                try:
                    info = (yf.Ticker(sym + ".NS").info) or {}
                    return sym, str(info.get("sector", "Unknown") or "Unknown")
                except Exception as e:
                    logger.debug("Failed to fetch sector for %s: %s", sym, e)
                    return sym, "Unknown"

            with ThreadPoolExecutor(max_workers=min(8, max(1, len(missing_tickers)))) as pool:
                for sym, sector in pool.map(_fetch_single_sector, missing_tickers):
                    resolved_map[sym] = sector
                
        # Update cache with newly found sectors
        if use_cache:
            with _SECTOR_MAP_CACHE_LOCK:
                cache = _load_universe_cache()
                existing_sector_cache = dict(cache.get("sector_map", {}).get("sectors", {}))
                existing_sector_cache.update({sym: resolved_map[sym] for sym in missing_tickers})

                cache["sector_map"] = {
                    "fetched_at": datetime.now().isoformat(),
                    "sectors": existing_sector_cache,
                }
                _save_universe_cache(cache)
            
    # Format the return dictionary to match exactly the requested input tickers
    final_map = {}
    for ticker in tickers:
        bare_ticker = ticker.replace(".NS", "")
        final_map[ticker] = resolved_map.get(bare_ticker, "Unknown")
        
    return final_map