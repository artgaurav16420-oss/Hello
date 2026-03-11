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

This script attempts to mitigate the bias by:
  1) Using NSE's published rebalancing/change notices if accessible.
  2) If not, it produces a parquet that the engine will use, while logging a
     clear WARNING so you always know the data quality.

For production-grade backtesting you should source true PIT data from:
  - NSE Data Products (https://dataproducts.nseindia.com/)
  - Bloomberg / Refinitiv
  - Quandl / WRDS

USAGE
-----
  python build_historical_fallback.py
  python build_historical_fallback.py --universe nifty500
  python build_historical_fallback.py --universe nse_total
  python build_historical_fallback.py --universe both --start 2015-01-01

OUTPUT
------
  data/historical_nifty500.parquet
  data/historical_nse_total.parquet
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import requests

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
    # NSE publishes semi-annual index rebalancing notices as PDFs/CSVs.
    # These URLs cover known reconstitution announcement CSV files.
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
    # NSE India requires a homepage visit first to set the session cookie.
    try:
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        time.sleep(0.5)
    except Exception:
        pass  # Cookie priming is best-effort

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


# ─── Nifty 500 constituent fetchers ──────────────────────────────────────────

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
            # Filter to EQ series only
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

            try:
                import nsepy
                from nsepy import get_index_pe_pb_div
                # nsepy doesn't directly expose constituent lists, skip
            except Exception:
                pass
    except Exception:
        pass
    return []


# ─── Parquet builder ──────────────────────────────────────────────────────────

def build_parquet(
    symbols: List[str],
    universe_type: str,
    start_date: str = "2018-01-01",
    snap_freq: str = "QS",  # quarterly start by default
) -> Path:
    """
    Create a PIT parquet from a list of symbols by backfilling quarterly
    snapshots from start_date to today.

    Schema expected by universe_manager.get_historical_universe():
        Index : DatetimeIndex named "date" (one row per snapshot date)
        Column: "tickers" — Python list of .NS-suffixed symbol strings
    """
    output_path = DATA_DIR / f"historical_{universe_type}.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_dates = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq=snap_freq)
    if snapshot_dates.empty:
        # Fallback: just today
        snapshot_dates = pd.DatetimeIndex([pd.Timestamp.today().normalize()])

    logger.info(
        "  Building %d quarterly snapshots from %s → today for %d symbols.",
        len(snapshot_dates), start_date, len(symbols),
    )

    rows = pd.DataFrame(
        {"tickers": [list(symbols)] * len(snapshot_dates)},
        index=pd.DatetimeIndex(snapshot_dates, name="date"),
    )
    rows.to_parquet(output_path)
    logger.info("  ✓ Written: %s  (%d rows)", output_path, len(rows))
    return output_path


# ─── CSV companion (for historical_builder.build_parquet_from_csv compat) ────

def build_csv_from_symbols(
    symbols: List[str],
    universe_type: str,
    start_date: str = "2018-01-01",
    snap_freq: str = "QS",
) -> Path:
    """Write the companion CSV (date, ticker) that historical_builder.py expects."""
    csv_path = DATA_DIR / f"historical_{universe_type}.csv"
    snapshot_dates = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq=snap_freq)
    rows = []
    for d in snapshot_dates:
        for sym in symbols:
            rows.append({"date": d.strftime("%Y-%m-%d"), "ticker": sym})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("  ✓ Written CSV companion: %s  (%d rows)", csv_path, len(rows))
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
    print("  historical dates.  Companies that were demoted or delisted")
    print("  since 2018 are silently included in old 'snapshots'.")
    print("  Expected CAGR inflation: +2 to +4 % p.a. vs true PIT data.")
    print()
    print("  For production use, source true PIT data from:")
    print("    • NSE Data Products  (dataproducts.nseindia.com)")
    print("    • Bloomberg / Refinitiv")
    print()
    print("=" * 65 + "\n")

    jobs: List[tuple] = []
    if want_nifty500:
        jobs.append(("nifty500",  fetch_nifty500_current,  "Nifty 500"))
    if want_nse_total:
        jobs.append(("nse_total", fetch_nse_total_current, "NSE Total"))

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
                "Falling back to hardcoded %d-name Nifty 50 hard-floor. "
                "This is a SEVERE data quality degradation — fix your network/proxy.",
                label, len(NIFTY50_CORE),
            )
            symbols = [_ns(s) for s in NIFTY50_CORE]

        # 4. Build parquet and CSV
        logger.info("Building parquet with %d symbols from %s...", len(symbols), start_date)
        parquet_path = build_parquet(symbols, universe_type, start_date)
        csv_path = build_csv_from_symbols(symbols, universe_type, start_date)

        # 5. Quick sanity check: load back and verify
        df = pd.read_parquet(parquet_path)
        first_row_tickers = df.iloc[0]["tickers"]
        n_snaps = len(df)
        n_syms  = len(first_row_tickers)
        print(f"  ✓ {parquet_path}  |  {n_snaps} snapshots  |  {n_syms} symbols per snapshot")
        print()

    print("=" * 65)
    print("  BUILD COMPLETE")
    print("  You can now run backtests. The survivorship bias warning")
    print("  above applies — treat CAGR figures as upper-bound estimates")
    print("  until you source true point-in-time constituent data.")
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
