"""
universe_manager.py — Universe Fetching & Caching v11.46
========================================================
Robust fetching of NSE/Nifty 500 universes, sector mappings, and
point-in-time historical constituents to eliminate backtest survivorship bias.

BUG FIXES (murder board):
- FIX-MB-SECTORTOCTOU: get_sector_map cache read inside lock on update path.
- FIX-MB-UM-01: Threaded sector fallback _fetch_single_sector now respects
  cfg.SECTOR_FETCH_TIMEOUT via a module-level default. Previously the yfinance
  call used no timeout, allowing 30-second blocks per symbol in degraded
  network conditions (8 threads × 30s = 240s hang).
- FIX-MB-UM-02: get_historical_universe logs a warning when it detects a
  duplicate-index parquet (constituents is a pd.Series) so operators are
  aware of the structural issue rather than it being silently handled.
- FIX-MB-M-03: _apply_adv_filter relaxed from all-or-nothing to symbol-level
  ADV=0 fallback. Missing symbols (delisted / no history / provider gap) are
  treated as ADV=0 and logged at DEBUG rather than failing the entire chunk.
  UniverseFetchError is only raised when a chunk returns zero symbols total.
- BUG-FIX-SESSION-LEAK: _fetch_single_sector now uses `with requests.Session()`
  context manager, ensuring urllib3 connection pools and sockets are released
  immediately rather than being abandoned to the GC in TIME_WAIT state.
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

# Default timeout for individual yfinance sector info calls (seconds).
# Overridden by cfg.SECTOR_FETCH_TIMEOUT when available.
_DEFAULT_SECTOR_FETCH_TIMEOUT = 8.0

class UniverseFetchError(RuntimeError):
    """Raised when primary and secondary universe data sources fail."""
    def __init__(self, message: str):
        super().__init__(message)
        self.fallback_universe: List[str] = []

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

# ─── Historical Universe Logic ────────────────────────────────────────────────

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

_MISSING_PARQUET_WARNED: Dict[str, bool] = {}
_NO_RECORD_WARNED: Dict[str, bool] = {}
_SECTOR_MAP_CACHE_LOCK = threading.Lock()
_HISTORICAL_UNIVERSE_DF_CACHE: Dict[Path, Tuple[float, pd.DataFrame]] = {}
_HISTORICAL_UNIVERSE_DATES_CACHE: Dict[Path, pd.DatetimeIndex] = {}
_UNIVERSE_LOOKUP_CACHE: Dict[tuple[str, pd.Timestamp], List[str]] = {}


def _clear_historical_universe_caches(hist_file: Path) -> None:
    universe_type = hist_file.stem.removeprefix("historical_")
    _HISTORICAL_UNIVERSE_DF_CACHE.pop(hist_file, None)
    _HISTORICAL_UNIVERSE_DATES_CACHE.pop(hist_file, None)
    stale_keys = [
        key for key in _UNIVERSE_LOOKUP_CACHE
        if key[0] == universe_type
    ]
    for key in stale_keys:
        _UNIVERSE_LOOKUP_CACHE.pop(key, None)


def _coerce_historical_members(value) -> List[str]:
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _normalize_historical_members(values) -> List[str]:
    return sorted({str(t).strip() for t in values if str(t).strip()})


def _is_cache_entry_fresh(fetched_at: str | None, ttl_hours: int = UNIVERSE_CACHE_TTL_H) -> bool:
    if not fetched_at:
        return False
    try:
        fetched_time = datetime.fromisoformat(fetched_at)
    except (TypeError, ValueError):
        return False
    return datetime.now() - fetched_time < timedelta(hours=ttl_hours)


def _normalize_sector_cache_entry(
    entry,
    *,
    fallback_fetched_at: str | None = None,
) -> tuple[str | None, str | None]:
    if isinstance(entry, dict):
        sector = str(entry.get("sector", "Unknown") or "Unknown")
        fetched_at = entry.get("fetched_at") or fallback_fetched_at
        return sector, fetched_at
    if isinstance(entry, str):
        return entry or "Unknown", fallback_fetched_at
    return None, None


def _load_historical_universe_df(hist_file: Path) -> pd.DataFrame:
    """Load historical universe parquet with mtime-based in-memory caching."""
    mtime = hist_file.stat().st_mtime
    cached = _HISTORICAL_UNIVERSE_DF_CACHE.get(hist_file)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    if cached is not None and cached[0] != mtime:
        _clear_historical_universe_caches(hist_file)

    # FIX-NEW-UM-02: the parquet files written by historical_builder store list-
    # valued cells in the "tickers" column.  pyarrow preserves Python lists
    # natively via its list<string> type; fastparquet serialises them differently
    # and may fail on read or return strings instead of lists.  Pin engine="pyarrow"
    # so the roundtrip is deterministic regardless of which parquet engine is
    # installed as the default.  If pyarrow is unavailable fall back to the
    # default engine with a warning so the problem is surfaced at import time.
    try:
        df = pd.read_parquet(hist_file, engine="pyarrow")
    except Exception:
        logger.warning(
            "[Universe] pyarrow not available or failed; falling back to default "
            "parquet engine.  List-valued tickers column may not round-trip correctly."
        )
        df = pd.read_parquet(hist_file)
    _HISTORICAL_UNIVERSE_DF_CACHE[hist_file] = (mtime, df)
    _HISTORICAL_UNIVERSE_DATES_CACHE[hist_file] = pd.DatetimeIndex(df.index.unique()).sort_values()
    return df


def get_historical_universe(universe_type: str, date: pd.Timestamp) -> List[str]:
    """
    Attempts to load the exact constituents for a specific historical date.

    Load order:
    1) Historical parquet snapshots.
    2) Point-in-time CSV snapshots.

    FIX-MB-UM-02: Logs a warning when constituents is returned as a pd.Series
    (indicating a duplicate-index parquet) so operators are aware rather than
    the structural issue being silently absorbed.
    """
    if universe_type.lower() == "custom":
        logger.warning(
            "SURVIVORSHIP BIAS WARNING: Backtesting 'custom' screener universe at %s "
            "uses the CURRENT constituent list only (stocks that survived to today). "
            "Historical members that were delisted or demoted are excluded. "
            "Results will overstate true historical performance. "
            "Use 'nifty500' or 'nse_total' for survivorship-safe backtesting.",
            date.strftime("%Y-%m-%d"),
        )
        return []

    hist_file = DATA_DIR / f"historical_{universe_type}.parquet"

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
            lookup_date = pd.Timestamp(date).normalize()
            df = _load_historical_universe_df(hist_file)
            available_dates = _HISTORICAL_UNIVERSE_DATES_CACHE.get(hist_file)
            if available_dates is None:
                available_dates = pd.DatetimeIndex(df.index.unique()).sort_values()
                _HISTORICAL_UNIVERSE_DATES_CACHE[hist_file] = available_dates

            target_pos = available_dates.searchsorted(lookup_date, side="right") - 1

            if target_pos >= 0:
                target_date = pd.Timestamp(available_dates[target_pos])
                cache_key = (universe_type, target_date)
                cached_members = _UNIVERSE_LOOKUP_CACHE.get(cache_key)
                if cached_members is not None:
                    return cached_members

                constituents = df.loc[target_date, "tickers"]

                if isinstance(constituents, pd.Series):
                    # FIX-MB-UM-02: warn about duplicate-index parquet so operators
                    # can investigate and rebuild if necessary.
                    logger.warning(
                        "[Universe] %s parquet has duplicate index rows for date %s "
                        "(constituents returned as Series). This indicates a structural "
                        "issue — run verify_parquet() and consider rebuilding. "
                        "Merging all rows for this date.",
                        universe_type, target_date.date(),
                    )
                    merged: List[str] = []
                    for cell in constituents.values:
                        merged.extend(_coerce_historical_members(cell))
                    result = _normalize_historical_members(merged)
                else:
                    result = _normalize_historical_members(_coerce_historical_members(constituents))

                _UNIVERSE_LOOKUP_CACHE[cache_key] = result
                return result

            logger.warning(
                "[Universe] No historical data prior to %s found in %s.",
                lookup_date.strftime("%Y-%m-%d"), hist_file
            )
        except Exception as exc:
            logger.error(
                "[Universe] Historical load failed for %s on %s: %s",
                universe_type, date.strftime("%Y-%m-%d"), exc
            )

    csv_members = _load_pit_universe_from_csv(universe_type, date)
    if csv_members:
        return csv_members

    if not _NO_RECORD_WARNED.get(universe_type):
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
    if UNIVERSE_CACHE_FILE.exists():
        try:
            UNIVERSE_CACHE_FILE.unlink()
            logger.info("[Universe] Cache invalidated.")
        except OSError as e:
            logger.error("[Universe] Failed to invalidate cache: %s", e)

# ─── ADV Liquidity Filter ─────────────────────────────────────────────────────

def _apply_adv_filter(tickers: List[str], cfg=None) -> List[str]:
    from momentum_engine import UltimateConfig, to_ns
    from data_cache import load_or_fetch

    if cfg is None:
        cfg = UltimateConfig()

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=max(150, int(cfg.ADV_LOOKBACK) * 2))).strftime("%Y-%m-%d")

    chunk_size  = _ADV_CHUNK_SIZE
    chunks      = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

    filtered_tickers: List[str] = []
    min_adv_volume = cfg.MIN_ADV_CRORES * 1e7

    logger.info(
        "[Universe] Filtering %d tickers against ₹%dCr ADV minimum...",
        len(tickers), cfg.MIN_ADV_CRORES,
    )

    chunk_failures: List[Dict[str, object]] = []

    for chunk_idx, chunk in enumerate(chunks):
        try:
            data = load_or_fetch(chunk, start_date, end_date, cfg=cfg)
            # FIX-MB-M-03: relax the all-or-nothing policy introduced in
            # FIX-NEW-UM-01.  The previous guard raised RuntimeError whenever
            # ANY symbol in a 75-symbol chunk had no data, breaking the entire
            # ADV filter for universes that contain even one recently-listed,
            # delisted, or provider-unavailable stock.  The Nifty 500 routinely
            # has a handful of such stocks.
            #
            # New policy:
            #   - Missing symbols are treated as ADV=0 (they fail the liquidity
            #     gate anyway) and logged at DEBUG.
            #   - Only a TOTAL chunk failure (zero symbols returned) is an error
            #     worthy of appending to chunk_failures and eventually raising
            #     UniverseFetchError.
            expected_ns = {to_ns(s) for s in chunk}
            returned_ns = set(data.keys())
            missing_ns  = expected_ns - returned_ns
            if missing_ns:
                logger.debug(
                    "[Universe] ADV chunk %d: %d symbol(s) not returned by "
                    "load_or_fetch (no history / delisted / provider gap): %s. "
                    "Treating as ADV=0 (will fail liquidity gate).",
                    chunk_idx, len(missing_ns), sorted(missing_ns)[:5],
                )
            if not data:
                # Total fetch failure — no symbols returned at all.
                raise RuntimeError(
                    f"load_or_fetch returned empty dict for entire chunk "
                    f"(size={len(chunk)}): {list(chunk)[:3]}"
                )
            for symbol in chunk:
                ns_sym = to_ns(symbol)
                if ns_sym in data:
                    frame = data[ns_sym]
                    if "Close" in frame.columns and "Volume" in frame.columns:
                        notional = (frame["Close"] * frame["Volume"]).clip(lower=0)
                        lookback = int(getattr(cfg, "ADV_LOOKBACK", 20))
                        adv = float(notional.tail(lookback).mean()) if notional.notna().any() else 0.0
                    else:
                        adv = 0.0
                    if np.isfinite(adv) and adv >= min_adv_volume:
                        filtered_tickers.append(ns_sym)
                # Missing symbols (not in data) contribute ADV=0 implicitly
                # by being absent from filtered_tickers.
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
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/csv,application/csv",
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))

def fetch_nse_equity_universe(cfg=None) -> List[str]:
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

        equity_df = df[df["SERIES"] == "EQ"]
        tickers = equity_df["SYMBOL"].unique().tolist()

        logger.info("[Universe] Cached %d raw EQ constituents (unfiltered).", len(tickers))
        cache["total_equity"] = {
            "fetched_at": datetime.now().isoformat(),
            "tickers":    tickers,
        }
        _save_universe_cache(cache)
        return tickers

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

    FIX-MB-SECTORTOCTOU: Cache read on update path moved inside the lock.
    FIX-MB-UM-01: Threaded fallback _fetch_single_sector now passes a timeout
      to yfinance (cfg.SECTOR_FETCH_TIMEOUT, defaulting to 8s) to prevent
      long per-symbol hangs in degraded network conditions.
    """
    # Resolve timeout from cfg if provided
    sector_timeout = float(
        getattr(cfg, "SECTOR_FETCH_TIMEOUT", _DEFAULT_SECTOR_FETCH_TIMEOUT)
    ) if cfg is not None else _DEFAULT_SECTOR_FETCH_TIMEOUT

    resolved_map = {}
    missing_tickers = []

    for ticker in tickers:
        bare_ticker = ticker.replace(".NS", "")
        if bare_ticker in STATIC_NSE_SECTORS:
            resolved_map[bare_ticker] = STATIC_NSE_SECTORS[bare_ticker]
        else:
            missing_tickers.append(bare_ticker)

    if missing_tickers and use_cache:
        cache = _load_universe_cache()
        sector_map_cache = cache.get("sector_map", {})
        sector_cache = sector_map_cache.get("sectors", {})
        sector_cache_fetched_at = sector_map_cache.get("fetched_at")

        still_missing = []
        for bare_ticker in missing_tickers:
            if bare_ticker in sector_cache:
                cached_sector, fetched_at = _normalize_sector_cache_entry(
                    sector_cache[bare_ticker],
                    fallback_fetched_at=sector_cache_fetched_at,
                )
                if cached_sector is not None and _is_cache_entry_fresh(fetched_at):
                    resolved_map[bare_ticker] = cached_sector
                    continue
            still_missing.append(bare_ticker)
        missing_tickers = still_missing

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
                    # FIX-NEW-UM-03: an empty info dict usually means yfinance hit
                    # a rate limit or the symbol is delisted.  Log at DEBUG so
                    # operators can correlate missing sector data with provider
                    # throttling without flooding production logs.
                    if not info:
                        logger.debug(
                            "[Universe] Empty info dict for %s — "
                            "possible rate limit or delisted symbol; defaulting to 'Unknown'.",
                            bare_sym,
                        )
                    sector = str(info.get("sector", "Unknown") or "Unknown")
                except Exception as e:
                    logger.debug("Failed to fetch sector for %s: %s", bare_sym, e)
                resolved_map[bare_sym] = sector
        except Exception as exc:
            logger.warning("[Universe] Batch sector fetch failed (%s). Falling back to threaded lookup.", exc)

            # FIX-MB-UM-01: pass sector_timeout to each individual yfinance call
            # to prevent indefinite hangs (previously no timeout was set, allowing
            # up to 30s per symbol × 8 workers = 240s total wall-time hang).
            def _fetch_single_sector(sym: str) -> Tuple[str, str]:
                # BUG-FIX-SIGNAL: The previous implementation used signal.alarm()
                # (SIGALRM) inside this worker function.  Python's signal module
                # strictly requires that signals are set and handled only in the
                # main thread.  Calling signal.signal() or signal.alarm() from a
                # ThreadPoolExecutor worker raises ValueError instantly, crashing
                # the entire fallback sector fetch.
                #
                # Fix: attach a requests.Session with a socket-level timeout to
                # the yfinance Ticker object.  The timeout is enforced by the
                # urllib3 socket layer and works correctly in any thread.
                try:
                    # requests is already imported at module level;
                    # use it directly rather than re-importing inside the worker.
                    _timeout = max(1.0, float(sector_timeout))
                    # BUG-FIX-SESSION-LEAK: use context manager so the
                    # urllib3 connection pool and underlying sockets are
                    # released immediately when the function returns.
                    # Without this, 100 missing tickers = 100 Sessions =
                    # ~200 sockets abandoned to the GC in TIME_WAIT state.
                    with requests.Session() as _session:
                        # Wrap Session.request so every call carries the timeout,
                        # regardless of which yfinance code path invokes it.
                        _orig_request = _session.request
                        _session.request = lambda method, url, **kwargs: _orig_request(
                            method, url,
                            timeout=kwargs.pop("timeout", _timeout),
                            **kwargs,
                        )
                        ticker_obj = yf.Ticker(sym + ".NS", session=_session)
                        info = (ticker_obj.info) or {}
                        result_sector = str(info.get("sector", "Unknown") or "Unknown")
                    return sym, result_sector
                except Exception as e:
                    logger.debug("Failed to fetch sector for %s: %s", sym, e)
                    return sym, "Unknown"

            with ThreadPoolExecutor(max_workers=min(8, max(1, len(missing_tickers)))) as pool:
                for sym, sector in pool.map(_fetch_single_sector, missing_tickers):
                    resolved_map[sym] = sector

        # FIX-MB-SECTORTOCTOU: read cache INSIDE lock before merging results.
        if use_cache:
            with _SECTOR_MAP_CACHE_LOCK:
                current_cache = _load_universe_cache()
                existing_sector_cache = dict(current_cache.get("sector_map", {}).get("sectors", {}))
                fetched_at = datetime.now().isoformat()
                existing_sector_cache.update({
                    sym: {
                        "sector": resolved_map[sym],
                        "fetched_at": fetched_at,
                    }
                    for sym in missing_tickers
                })

                current_cache["sector_map"] = {
                    "fetched_at": fetched_at,
                    "sectors": existing_sector_cache,
                }
                _save_universe_cache(current_cache)

    final_map = {}
    for ticker in tickers:
        bare_ticker = ticker.replace(".NS", "")
        final_map[ticker] = resolved_map.get(bare_ticker, "Unknown")

    return final_map
