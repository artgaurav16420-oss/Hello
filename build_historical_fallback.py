"""
build_historical_fallback.py
============================
Fallback generator for historical Nifty 500 / NSE Total PIT universe parquets
when the primary GitHub source in historical_builder.py is unavailable (404).

DATA STRATEGY
-------------
Primary  : NSE India index CSV (archives.nseindia.com) — current constituents.
Secondary: nse-data / nsepy package if installed (has rebalancing history).
Tertiary : Static hard-floor list (Nifty 50 core names).

SURVIVORSHIP BIAS NOTE
----------------------
Using the CURRENT index composition backfilled to historical dates introduces
survivorship bias: stocks that were demoted or delisted after 2018 are silently
included in older "snapshots". This inflates historical CAGR by roughly 2-4%
p.a. relative to a true point-in-time universe.

PHASE 3 FIX (Time-Traveling IPOs):
This script now strictly enforces Point-in-Time trading volume gates. Even though
we use the surviving constituents, an asset is strictly excluded from a historical
snapshot if it did not have verifiable trading volume prior to that snapshot date.

USAGE
-----
  python build_historical_fallback.py
  python build_historical_fallback.py --universe nifty500
  python build_historical_fallback.py --universe nse_total
  python build_historical_fallback.py --universe both --start 2015-01-01
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

from data_cache import load_or_fetch
from momentum_engine import UltimateConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

# ── Browser-like headers required by NSE India ────────────────────────────────
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# ── NSE source URLs ───────────────────────────────────────────────────────────
NSE_SOURCES = {
    "nifty500": [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    ],
    "nse_total": [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    ],
    "nifty500_changes": [
        "https://archives.nseindia.com/content/indices/ind_nifty500_change_notice.csv",
    ],
}

# ── Hard-floor fallback: Nifty 50 core names ─────────────────────────────────
NIFTY50_CORE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL",
    "HINDUNILVR", "ITC", "SBIN", "LTIM", "BAJFINANCE", "HCLTECH", "MARUTI",
    "SUNPHARMA", "ADANIENT", "KOTAKBANK", "TITAN", "ONGC", "TATAMOTORS",
    "NTPC", "AXISBANK", "ADANIPORTS", "ASIANPAINT", "COALINDIA", "BAJAJFINSV",
    "JSWSTEEL", "M&M", "POWERGRID", "TATASTEEL", "ULTRACEMCO", "GRASIM",
    "HINDALCO", "NESTLEIND", "INDUSINDBK", "TECHM", "WIPRO", "CIPLA",
    "HDFCLIFE", "SBILIFE", "DRREDDY", "HEROMOTOCO", "EICHERMOT", "BPCL",
    "BAJAJ-AUTO", "BRITANNIA", "APOLLOHOSP", "DIVISLAB", "TATACONSUM",
    "LT", "HDFC",
]


def _ns(sym: str) -> str:
    s = str(sym).strip().upper()
    return s if s.endswith(".NS") or s.startswith("^") else f"{s}.NS"


def _fetch_with_retry(url: str, retries: int = 3, delay: float = 2.0) -> Optional[requests.Response]:
    """GET with exponential backoff. Returns Response or None on failure."""
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        time.sleep(0.5)
    except Exception:
        pass

    for attempt in range(retries):
        try:
            resp = session.get(url, headers=NSE_HEADERS, timeout=20)
            if resp.status_code == 200 and len(resp.content) > 100:
                return resp
            logger.warning("  [%s] HTTP %d, attempt %d/%d", url[:60], resp.status_code, attempt + 1, retries)
        except Exception as exc:
            logger.warning("  [%s] %s, attempt %d/%d", url[:60], exc, attempt + 1, retries)
        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))
    return None


def fetch_nifty500_current() -> List[str]:
    """Download current Nifty 500 constituent list from NSE India."""
    for url in NSE_SOURCES["nifty500"]:
        logger.info("  Trying: %s", url)
        resp = _fetch_with_retry(url)
        if resp is None:
            continue
        try:
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip().upper() for c in df.columns]
            sym_col = next((c for c in ["SYMBOL", "TICKER", "COMPANY SYMBOL"] if c in df.columns), None)
            if sym_col is None:
                logger.warning("  Could not find SYMBOL column in CSV. Columns: %s", list(df.columns))
                continue
            syms = [_ns(s) for s in df[sym_col].dropna().astype(str).str.strip().unique() if s]
            if len(syms) >= 100:
                logger.info("  ✓ Fetched %d Nifty 500 symbols.", len(syms))
                return sorted(syms)
        except Exception as exc:
            logger.warning("  Parse error: %s", exc)

    logger.error("All Nifty 500 URLs failed — check your internet / proxy settings.")
    return []


def fetch_nse_total_current() -> List[str]:
    """Download current NSE Total equity list from NSE India."""
    for url in NSE_SOURCES["nse_total"]:
        logger.info("  Trying: %s", url)
        resp = _fetch_with_retry(url)
        if resp is None:
            continue
        try:
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip().upper() for c in df.columns]
            if "SERIES" in df.columns:
                df = df[df["SERIES"].str.strip() == "EQ"]
            sym_col = next((c for c in ["SYMBOL", "TICKER"] if c in df.columns), None)
            if sym_col is None:
                continue
            syms = [_ns(s) for s in df[sym_col].dropna().astype(str).str.strip().unique() if s]
            if len(syms) >= 500:
                logger.info("  ✓ Fetched %d NSE Total equity symbols.", len(syms))
                return sorted(syms)
        except Exception as exc:
            logger.warning("  Parse error: %s", exc)

    logger.error("All NSE Total URLs failed.")
    return []


def fetch_via_nsepy(universe_type: str) -> List[str]:
    """Attempt to use nsepy/nsetools if installed."""
    try:
        if universe_type in ("nifty500", "nse_total"):
            try:
                from nsetools import Nse
                nse = Nse()
                if universe_type == "nifty500":
                    stocks = nse.get_index_quote("cnx nifty 500")
                    if stocks:
                        return sorted([_ns(s) for s in stocks.keys()])
            except Exception:
                pass
    except Exception:
        pass
    return []


# ─── Point-in-Time Parquet builder ────────────────────────────────────────────

def build_parquet(
    symbols: List[str],
    universe_type: str,
    valid_trading_days: pd.DataFrame,
    history_gate: int,
    start_date: str = "2018-01-01",
    snap_freq: str = "QS",
) -> Path:
    """
    Create a PIT parquet from a list of symbols by backfilling quarterly
    snapshots from start_date to today, STRICTLY EXCLUDING assets that had
    not yet listed / lacked trading volume prior to the snapshot date.
    """
    output_path = DATA_DIR / f"historical_{universe_type}.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_dates = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq=snap_freq)
    if snapshot_dates.empty:
        snapshot_dates = pd.DatetimeIndex([pd.Timestamp.today().normalize()])

    logger.info(
        "  Building %d volume-gated snapshots from %s → today for %d symbols.",
        len(snapshot_dates), start_date, len(symbols),
    )

    rows = []
    for d in snapshot_dates:
        past_data = valid_trading_days[valid_trading_days.index <= d]
        if past_data.empty:
            rows.append([])
            continue
            
        status_on_date = past_data.iloc[-1]
        # Only include asset if it existed and had enough volume history
        eligible = status_on_date[status_on_date >= history_gate].index.tolist()
        rows.append(eligible)

    out_df = pd.DataFrame({"tickers": rows}, index=pd.DatetimeIndex(snapshot_dates, name="date"))
    out_df.to_parquet(output_path)
    logger.info("  ✓ Written: %s  (%d rows)", output_path, len(out_df))
    return output_path


def build_csv_from_symbols(
    symbols: List[str],
    universe_type: str,
    valid_trading_days: pd.DataFrame,
    history_gate: int,
    start_date: str = "2018-01-01",
    snap_freq: str = "QS",
) -> Path:
    """Write the companion CSV incorporating strict volume existence gates."""
    csv_path = DATA_DIR / f"historical_{universe_type}.csv"
    snapshot_dates = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq=snap_freq)
    
    csv_rows = []
    for d in snapshot_dates:
        past_data = valid_trading_days[valid_trading_days.index <= d]
        if past_data.empty:
            continue
            
        status_on_date = past_data.iloc[-1]
        eligible = status_on_date[status_on_date >= history_gate].index.tolist()
        for sym in eligible:
            csv_rows.append({"date": d.strftime("%Y-%m-%d"), "ticker": sym})
            
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logger.info("  ✓ Written CSV companion: %s  (%d rows)", csv_path, len(csv_rows))
    return csv_path


# ─── Main orchestration ───────────────────────────────────────────────────────

def run(universe_arg: str = "both", start_date: str = "2018-01-01") -> None:
    want_nifty500  = universe_arg in ("both", "nifty500")
    want_nse_total = universe_arg in ("both", "nse_total")

    print("\n" + "=" * 65)
    print("  HISTORICAL UNIVERSE FALLBACK BUILDER")
    print("=" * 65)
    print()
    print("  ⚠  SURVIVORSHIP BIAS WARNING")
    print("  ─────────────────────────────────────────────────────────")
    print("  This script backfills TODAY'S index constituents to")
    print("  historical dates. Companies that were demoted or delisted")
    print("  since 2018 are silently included in old 'snapshots'.")
    print()
    print("  ✓ PHASE 3/4 FIX ACTIVE: 'Time-Traveling IPO' protection")
    print("  is engaged. Assets are strictly volume-gated and excluded")
    print("  from history if they had not yet listed.")
    print()
    print("=" * 65 + "\n")

    jobs: List[tuple] = []
    if want_nifty500:
        jobs.append(("nifty500",  fetch_nifty500_current,  "Nifty 500"))
    if want_nse_total:
        jobs.append(("nse_total", fetch_nse_total_current, "NSE Total"))

    cfg = UltimateConfig()
    history_gate = getattr(cfg, "HISTORY_GATE", 20)

    for universe_type, fetcher, label in jobs:
        print(f"  ── {label} ──")

        # 1. Try live NSE India fetch
        logger.info("Step 1: Fetching current %s from NSE India...", label)
        symbols = fetcher()

        # 2. Try nsepy fallback
        if not symbols:
            logger.info("Step 2: Trying nsepy/nsetools fallback...")
            symbols = fetch_via_nsepy(universe_type)

        # 3. Absolute hard-floor: Nifty 50 core names
        if not symbols:
            logger.warning(
                "Step 3: All network sources failed for %s. "
                "Falling back to hardcoded %d-name Nifty 50 hard-floor.",
                label, len(NIFTY50_CORE),
            )
            symbols = [_ns(s) for s in NIFTY50_CORE]

        # 4. Fetch volume history to enforce Point-In-Time existence
        logger.info("Step 4: Fetching market data to enforce Point-in-Time existence gates...")
        market_data = load_or_fetch(
            symbols, start_date, pd.Timestamp.today().strftime("%Y-%m-%d"), cfg=cfg
        )
        
        vol_dict = {}
        for sym, df in market_data.items():
            if df is not None and not df.empty and "Volume" in df.columns:
                vol_dict[sym] = df["Volume"].replace(0, np.nan)
        
        vol_matrix = pd.DataFrame(vol_dict).sort_index()
        valid_trading_days = vol_matrix.notna().cumsum()

        # 5. Build parquet and CSV using the Point-In-Time gates
        logger.info("Building parquet with %d symbols from %s...", len(symbols), start_date)
        parquet_path = build_parquet(
            symbols, universe_type, valid_trading_days, history_gate, start_date
        )
        csv_path = build_csv_from_symbols(
            symbols, universe_type, valid_trading_days, history_gate, start_date
        )

        # 6. Quick sanity check: load back and verify
        df = pd.read_parquet(parquet_path)
        first_row_tickers = df.iloc[0]["tickers"]
        n_snaps = len(df)
        n_syms  = len(first_row_tickers)
        print(f"  ✓ {parquet_path}  |  {n_snaps} snapshots  |  {n_syms} symbols per snapshot")
        print()

    print("=" * 65)
    print("  BUILD COMPLETE")
    print("=" * 65 + "\n")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fallback historical universe builder for Ultimate Momentum.")
    p.add_argument(
        "--universe",
        default="both",
        choices=["nifty500", "nse_total", "both"],
        help="Which universe to build (default: both).",
    )
    p.add_argument(
        "--start",
        default="2018-01-01",
        help="Earliest snapshot date (YYYY-MM-DD). Default: 2018-01-01.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(universe_arg=args.universe, start_date=args.start)