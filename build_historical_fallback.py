"""
build_historical_fallback.py
============================
Generate point-in-time (PIT) universe parquets for survivorship-safe backtests.

BUG FIXES (murder board):
- FIX-MB-DUPNS: The module previously defined _ns() twice at module level. The
  second definition (line ~130) silently shadowed the first. Both were identical
  in the original code, but any future edit to one would not affect the other.
  Fixed by removing the duplicate; a single definition is kept at the top of
  the module and referenced everywhere.
- FIX-MB-HYBRIDMERGE: The vol-gate row retention condition in the hybrid Wayback
  merge had an off-by-one: vol-gate rows exactly on the last Wayback date were
  kept by both branches (the "before all Wayback dates" branch checked <=, not <)
  creating a duplicate row. The condition is now strictly:
    - Keep if strictly before the FIRST Wayback date (no Wayback anchor yet).
    - Keep if strictly after the LAST Wayback date (future coverage).
    - Drop otherwise (covered by a Wayback anchor).
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import io as _io_mod
import time as _time_mod

import numpy as np
import pandas as pd
import requests


def _load_env_file_fallback(env_path: Path = Path('.env')) -> None:
    """Minimal `.env` parser used when python-dotenv is unavailable."""
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:
        logger = logging.getLogger(__name__)
        logger.debug('[Env] Could not parse .env fallback at %s: %s', env_path, exc)


def _bootstrap_env() -> None:
    """Load env vars from `.env` with optional python-dotenv dependency."""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        _load_env_file_fallback()


_bootstrap_env()

from data_cache import load_or_fetch
from momentum_engine import UltimateConfig

# ── Wayback Machine constants ─────────────────────────────────────────────────
_WBM_CDX_URL          = "https://web.archive.org/cdx/search/cdx"
_WBM_FETCH_TPL        = "https://web.archive.org/web/{ts}if_/{url}"
_WBM_MIN_SNAPSHOTS    = 2
_WBM_SLEEP_SECS       = 0.3

_NSE_N500_CSV_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www1.nseindia.com/content/indices/ind_nifty500list.csv",
]
_NSE_N500_CSV_URL = _NSE_N500_CSV_URLS[0]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

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


# FIX-MB-DUPNS: Single canonical definition of _ns(). The original module had
# two definitions; the second shadowed the first. Removed the duplicate.
def _ns(sym: str) -> str:
    """Return sym with exactly one '.NS' suffix, upper-cased."""
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


# ─── Wayback Machine PIT fetch ───────────────────────────────────────────────

def _wbm_cdx_timestamps(nse_url: str, start_year: int = 2015) -> list[str]:
    params = {
        "url":      nse_url,
        "output":   "json",
        "fl":       "timestamp,statuscode",
        "filter":   "statuscode:200",
        "collapse": "timestamp:6",
        "from":     str(start_year),
        "limit":    "500",
    }
    try:
        resp = requests.get(
            _WBM_CDX_URL, params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        resp.raise_for_status()
        rows = resp.json()
        ts_list = [r[0] for r in rows[1:] if r[1] == "200"]
        logger.info("[Wayback] CDX found %d monthly snapshots for %s", len(ts_list), nse_url)
        return ts_list
    except Exception as exc:
        logger.warning("[Wayback] CDX query failed: %s", exc)
        return []


def _wbm_fetch_csv(timestamp: str, original_url: str) -> pd.DataFrame | None:
    """Fetch a single Wayback snapshot and parse as CSV. Returns None on failure."""
    url = _WBM_FETCH_TPL.format(ts=timestamp, url=original_url)
    try:
        resp = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
        )
        resp.raise_for_status()
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(_io_mod.BytesIO(resp.content), encoding=enc)
                if not df.empty and len(df.columns) >= 3:
                    return df
            except Exception:
                continue
    except Exception as exc:
        logger.debug("[Wayback] fetch failed (ts=%s): %s", timestamp, exc)
    return None


def _symbols_from_nse_csv(df: pd.DataFrame) -> list[str]:
    """Extract .NS-suffixed tickers from an ind_nifty500list.csv DataFrame."""
    import re
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ("symbol", "symbols", "ticker", "nse symbol", "nse_symbol"):
        if col in df.columns:
            raw = df[col].dropna().astype(str).str.strip()
            tickers = [_ns(s) for s in raw if s and s.upper() not in ("SYMBOL", "TICKER", "")]
            if tickers:
                return sorted(set(tickers))
    pattern = re.compile(r"^[A-Z][A-Z0-9&\-]{1,14}$")
    found: set[str] = set()
    skip = {"SERIES", "ISIN", "INDUSTRY", "NAME", "SYMBOL", "TICKER"}
    for col in df.columns:
        for val in df[col].dropna().astype(str):
            v = val.strip().upper()
            if pattern.match(v) and v not in skip:
                found.add(_ns(v))
    return sorted(found)


def fetch_nifty500_wayback(start_year: int = 2015) -> tuple[list[tuple[str, list[str]]], bool]:
    """
    Download all monthly Wayback snapshots across ALL 3 known NSE CSV URLs.
    Returns merged (date_str, [tickers]) pairs sorted by date.
    """
    all_snapshots: dict[str, list[str]] = {}

    for nse_url in _NSE_N500_CSV_URLS:
        domain = nse_url.split("/")[2]
        timestamps = _wbm_cdx_timestamps(nse_url, start_year=start_year)
        if not timestamps:
            logger.info("[Wayback] No snapshots found for %s, skipping.", domain)
            continue

        new_dates = [
            ts for ts in timestamps
            if f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" not in all_snapshots
        ]
        total = len(timestamps)
        new   = len(new_dates)
        print(f"[Wayback] {domain}: {total} snapshots ({new} new dates to download)...")

        failed = 0
        for i, ts in enumerate(new_dates, 1):
            snap_date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"

            df = _wbm_fetch_csv(ts, nse_url)
            if df is None:
                failed += 1
                _time_mod.sleep(_WBM_SLEEP_SECS * 0.5)
                continue

            tickers = _symbols_from_nse_csv(df)
            if not tickers:
                failed += 1
                continue

            all_snapshots[snap_date] = tickers
            print(f"  [{i:3d}/{new}]  {snap_date}  →  {len(tickers)} tickers  ({domain})")
            _time_mod.sleep(_WBM_SLEEP_SECS)

        logger.info(
            "[Wayback] %s: %d/%d new downloads failed.",
            domain, failed, new,
        )

    snapshots = sorted(all_snapshots.items())
    success   = len(snapshots) >= _WBM_MIN_SNAPSHOTS

    logger.info(
        "[Wayback] Total unique PIT snapshots collected: %d across %d URLs. success=%s",
        len(snapshots), len(_NSE_N500_CSV_URLS), success,
    )
    return snapshots, success


def build_parquet_from_wayback(
    universe_type: str,
    snapshots: list[tuple[str, list[str]]],
) -> Path:
    """Write a PIT parquet from Wayback-sourced snapshots."""
    output_path = DATA_DIR / f"historical_{universe_type}.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows: dict[pd.Timestamp, list[str]] = {}
    for date_str, tickers in snapshots:
        ts = pd.Timestamp(date_str)
        existing = rows.get(ts, [])
        rows[ts] = sorted(set(existing) | set(tickers))

    idx    = pd.DatetimeIndex(sorted(rows.keys()), name="date")
    series = pd.Series([rows[d] for d in idx], index=idx, name="tickers")
    out_df = pd.DataFrame({"tickers": series})
    out_df.to_parquet(output_path)

    csv_path = DATA_DIR / f"historical_{universe_type}.csv"
    csv_rows = []
    for date_str, tickers in snapshots:
        for tkr in tickers:
            csv_rows.append({"date": date_str, "ticker": tkr})
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    logger.info(
        "[Wayback] Wrote %d TRUE PIT snapshots → %s  (+ companion CSV)",
        len(out_df), output_path,
    )
    return output_path


# ─── Current constituent fetchers ─────────────────────────────────────────────

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
    universe_type: str,
    valid_trading_days: pd.DataFrame,
    history_gate: int,
    start_date: str = "2018-01-01",
    snap_freq: str = "QS",
) -> Path:
    """
    Create a PIT parquet from valid_trading_days by backfilling quarterly
    snapshots from start_date to today.
    """
    assert all(
        s.endswith(".NS") or s.startswith("^") for s in valid_trading_days.columns
    ), "MB-15: all columns in valid_trading_days must be .NS-suffixed or index tickers"
    output_path = DATA_DIR / f"historical_{universe_type}.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_dates = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq=snap_freq)
    if snapshot_dates.empty:
        snapshot_dates = pd.DatetimeIndex([pd.Timestamp.today().normalize()])

    logger.info(
        "  Building %d volume-gated snapshots from %s → today for %d symbols.",
        len(snapshot_dates), start_date, len(valid_trading_days.columns),
    )

    rows = []
    for d in snapshot_dates:
        past_data = valid_trading_days[valid_trading_days.index <= d]
        if past_data.empty:
            rows.append([])
            continue

        status_on_date = past_data.iloc[-1]
        eligible = status_on_date[status_on_date >= history_gate].index.tolist()
        rows.append(eligible)

    out_df = pd.DataFrame({"tickers": rows}, index=pd.DatetimeIndex(snapshot_dates, name="date"))
    out_df.to_parquet(output_path)
    logger.info("  ✓ Written: %s  (%d rows)", output_path, len(out_df))
    return output_path


def build_csv_from_symbols(
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


def _build_adnv_ranked_snapshots(
    market_data: dict[str, pd.DataFrame],
    start_date: str,
    top_n: int = 500,
    lookback_days: int = 126,
    min_trading_days: int = 60,
    snap_freq: str = "QS",
) -> pd.DataFrame:
    """Build quarterly PIT rows using ADNV ranking over available candidate symbols."""
    snapshot_dates = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq=snap_freq)
    rows: list[list[str]] = []

    for d in snapshot_dates:
        ref = pd.Timestamp(d)
        scores: dict[str, float] = {}
        for sym, df in market_data.items():
            if df is None or df.empty or "Close" not in df.columns or "Volume" not in df.columns:
                continue
            try:
                hist = df.loc[:ref].tail(lookback_days)
                if len(hist) < min_trading_days:
                    continue
                close = pd.to_numeric(hist["Close"], errors="coerce")
                volume = pd.to_numeric(hist["Volume"], errors="coerce").replace(0, np.nan)
                adnv = float((close * volume).mean())
                if pd.isna(adnv) or adnv <= 0:
                    continue
                scores[sym] = adnv
            except Exception:
                continue

        ranked = sorted(scores, key=scores.get, reverse=True)[:top_n]
        rows.append(ranked)

    return pd.DataFrame({"tickers": rows}, index=pd.DatetimeIndex(snapshot_dates, name="date"))


def _write_snapshot_outputs(universe_type: str, snapshot_df: pd.DataFrame) -> Path:
    """Write parquet+CSV outputs from a DataFrame with a `tickers` list column."""
    output_path = DATA_DIR / f"historical_{universe_type}.parquet"
    csv_path = DATA_DIR / f"historical_{universe_type}.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_df.to_parquet(output_path)

    csv_rows = []
    for d, tickers in snapshot_df["tickers"].items():
        for tkr in (tickers if isinstance(tickers, list) else list(tickers)):
            csv_rows.append({"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "ticker": tkr})
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    logger.info("  ✓ Written: %s  (%d rows)", output_path, len(snapshot_df))
    logger.info("  ✓ Written CSV companion: %s  (%d rows)", csv_path, len(csv_rows))
    return output_path


# ─── Main orchestration ───────────────────────────────────────────────────────

def run(universe_arg: str = "both", start_date: str = "2018-01-01") -> None:
    want_nifty500  = universe_arg in ("both", "nifty500")
    want_nse_total = universe_arg in ("both", "nse_total")

    print("\n" + "=" * 65)
    print("  HISTORICAL UNIVERSE BUILDER")
    print("=" * 65)

    groww_enabled = bool(os.getenv("GROWW_API_TOKEN", "").strip())
    provider_msg = "Groww primary + yfinance fallback" if groww_enabled else "yfinance primary (set GROWW_API_TOKEN in .env to enable Groww)"
    print(f"  Data provider mode: {provider_msg}")

    # ── Nifty 500 — try Wayback Machine first ────────────────────────────────
    if want_nifty500:
        print()
        print("  ── Nifty 500 ──")
        print()
        print("  Step 1/2: Attempting TRUE PIT build via Wayback Machine…")
        print("  (downloads ~100 monthly NSE snapshots — takes ~2 min)")

        snapshots, wbm_ok = fetch_nifty500_wayback(start_year=2015)

        cfg = UltimateConfig()
        history_gate = getattr(cfg, "HISTORY_GATE", 20)

        logger.info("Building volume-gate baseline (quarterly, current survivors)...")
        symbols = fetch_nifty500_current()
        if not symbols:
            symbols = fetch_via_nsepy("nifty500")
        if not symbols:
            logger.warning("All network sources failed — using Nifty 50 hard-floor.")
            symbols = [_ns(s) for s in NIFTY50_CORE]

        market_data = load_or_fetch(
            symbols, start_date, pd.Timestamp.today().strftime("%Y-%m-%d"), cfg=cfg
        )
        vol_dict = {}
        for sym, df in market_data.items():
            if df is not None and not df.empty and "Volume" in df.columns:
                vol_dict[sym] = df["Volume"].replace(0, np.nan)
        vol_matrix = pd.DataFrame(vol_dict).sort_index()
        valid_trading_days = vol_matrix.notna().cumsum()

        parquet_path = build_parquet(
            "nifty500", valid_trading_days, history_gate, start_date
        )
        build_csv_from_symbols(
            "nifty500", valid_trading_days, history_gate, start_date
        )

        if snapshots:
            print()
            n_wbm = len(snapshots)
            wbm_mode = "TRUE PIT" if wbm_ok else "PARTIAL PIT (sparse archive)"
            print(f"  ✓ Injecting {n_wbm} Wayback anchor(s) → {wbm_mode}")

            wbm_dates = sorted(pd.Timestamp(d) for d, _ in snapshots)
            wfo_years = [2019, 2020, 2021, 2022]
            print()
            print("  WFO year coverage after hybrid merge:")
            for yr in wfo_years:
                ref = pd.Timestamp(f"{yr}-01-01")
                eligible = [d for d in wbm_dates if d <= ref]
                if eligible:
                    anchor = max(eligible)
                    src = "Wayback TRUE PIT"
                    wbm_dict = {pd.Timestamp(d): t for d, t in snapshots}
                    n_m = len(wbm_dict.get(anchor, []))
                else:
                    src = "volume-gate (survivors-only)"
                    n_m = 0
                print(f"    OOS {yr}  ← {src}  ({anchor.date() if eligible else 'none'}, {n_m if n_m else '~380-410'} members)")

            df_existing = pd.read_parquet(parquet_path)
            wbm_rows: dict[pd.Timestamp, list] = {}
            for date_str, tickers in snapshots:
                ts = pd.Timestamp(date_str)
                wbm_rows[ts] = sorted(set(tickers))

            wbm_dates_sorted = sorted(wbm_rows.keys())
            first_wbm = min(wbm_dates_sorted)
            last_wbm  = max(wbm_dates_sorted)

            # FIX-MB-HYBRIDMERGE: Retention condition corrected to strict
            # inequalities so vol-gate rows on the exact Wayback boundary dates
            # are not double-included (creating duplicate parquet rows).
            # Rule:
            #   Keep if date < first_wbm  → no Wayback anchor exists yet for this period.
            #   Keep if date > last_wbm   → beyond all Wayback coverage (future data).
            #   Drop otherwise            → Wayback anchor covers this period.
            kept_vg_dates = []
            for d in df_existing.index:
                if d < first_wbm:
                    kept_vg_dates.append(d)   # before any Wayback anchor — keep
                elif d > last_wbm:
                    kept_vg_dates.append(d)   # after last Wayback anchor — keep for future coverage
                # else: within Wayback coverage — drop (Wayback row is authoritative)

            all_dates = sorted(set(kept_vg_dates) | set(wbm_rows.keys()))
            merged_tickers = []
            for d in all_dates:
                if d in wbm_rows:
                    merged_tickers.append(wbm_rows[d])
                else:
                    row = df_existing.loc[d, "tickers"]
                    merged_tickers.append(list(row) if not isinstance(row, list) else row)

            idx = pd.DatetimeIndex(all_dates, name="date")
            out_df = pd.DataFrame({"tickers": merged_tickers}, index=idx)
            out_df.to_parquet(parquet_path)

            csv_path = DATA_DIR / "historical_nifty500.csv"
            csv_rows = []
            for d, tickers in zip(all_dates, merged_tickers):
                for tkr in tickers:
                    csv_rows.append({"date": d.strftime("%Y-%m-%d"), "ticker": tkr})
            pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

            logger.info(
                # FIX-MB2-WBMMERGE: Previously used len(df_existing) for the
                # volume-gate count, but after FIX-MB-HYBRIDMERGE only a subset of
                # df_existing rows are kept (those strictly outside the Wayback range).
                # The correct count is len(kept_vg_dates).
                "[Hybrid] Merged parquet: %d total snapshots "
                "(%d Wayback + %d volume-gate kept). Written → %s",
                len(out_df), len(wbm_rows), len(kept_vg_dates), parquet_path,
            )
        else:
            print()
            print("  ⚠ No Wayback snapshots available.")
            print("    Falling back to ADNV-ranked approximation.")

            adnv_df = _build_adnv_ranked_snapshots(
                market_data=market_data,
                start_date=start_date,
                top_n=500,
            )
            non_empty = int(sum(bool(x) for x in adnv_df["tickers"]))
            if non_empty > 0:
                parquet_path = _write_snapshot_outputs("nifty500", adnv_df)
                logger.warning(
                    "[Nifty500] Wayback unavailable: wrote ADNV approximation with %d/%d non-empty snapshots.",
                    non_empty,
                    len(adnv_df),
                )
            else:
                logger.warning(
                    "[Nifty500] ADNV approximation produced no eligible rows; keeping volume-gate baseline output."
                )

        df_check = pd.read_parquet(parquet_path)
        n_snaps  = len(df_check)
        first_row = df_check.iloc[0]["tickers"]
        n_syms    = len(first_row) if isinstance(first_row, list) else len(list(first_row))
        print()
        print(f"  ✓ {parquet_path}  |  {n_snaps} snapshots  |  ~{n_syms} symbols/first-snapshot")

        late_joiners = {"ZOMATO.NS", "NYKAA.NS", "PAYTM.NS", "IREDA.NS"}
        pre_2021 = df_check[df_check.index < pd.Timestamp("2021-07-01")]
        if not pre_2021.empty:
            pre_members: set = set()
            for v in pre_2021["tickers"]:
                pre_members.update(list(v) if not isinstance(v, list) else v)
            found = late_joiners & pre_members
            if found:
                logger.warning(
                    "[Verify] Pre-2021 snapshot contains post-2021 IPOs: %s "
                    "(Wayback may have served a stale/cached page for that date)",
                    found,
                )
            else:
                print("  ✓ Verified: no time-traveling IPOs in pre-2021 snapshots.")

    # ── NSE Total — volume-gate only ─────────────────────────────────────────
    if want_nse_total:
        print()
        print("  ── NSE Total ──")
        print("  No Wayback source for NSE Total universe.")
        print("  Using current equity list + volume-gate method.")
        print()

        cfg = UltimateConfig()
        history_gate = getattr(cfg, "HISTORY_GATE", 20)

        logger.info("Fetching current NSE Total equity list...")
        symbols = fetch_nse_total_current()

        if not symbols:
            symbols = [_ns(s) for s in NIFTY50_CORE]
            logger.warning("NSE Total fetch failed — using Nifty 50 hard-floor.")

        logger.info("Fetching market data for volume-gate PIT construction...")
        market_data = load_or_fetch(
            symbols, start_date, pd.Timestamp.today().strftime("%Y-%m-%d"), cfg=cfg
        )

        vol_dict = {}
        for sym, df in market_data.items():
            if df is not None and not df.empty and "Volume" in df.columns:
                vol_dict[sym] = df["Volume"].replace(0, np.nan)

        vol_matrix = pd.DataFrame(vol_dict).sort_index()
        valid_trading_days = vol_matrix.notna().cumsum()

        parquet_path = build_parquet(
            "nse_total", valid_trading_days, history_gate, start_date
        )
        build_csv_from_symbols(
            "nse_total", valid_trading_days, history_gate, start_date
        )

        df_check = pd.read_parquet(parquet_path)
        first_row = df_check.iloc[0]["tickers"]
        n_syms = len(first_row) if isinstance(first_row, list) else len(list(first_row))
        print(f"  ✓ {parquet_path}  |  {len(df_check)} snapshots  |  ~{n_syms} symbols/snapshot")

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
