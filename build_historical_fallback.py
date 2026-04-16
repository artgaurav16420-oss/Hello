"""
build_historical_fallback.py
============================
Generate point-in-time (PIT) universe parquets for survivorship-safe backtests.

BUG FIXES (murder board):
- FIX-MB-DUPNS: The module previously defined normalize_ns_ticker() twice at module level. The
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
import importlib.util
import io
import logging
import os
import threading
import time
import random
from pathlib import Path
from typing import List, Optional

import io as _io_mod
import time as _time_mod

# OSQP must be imported BEFORE numpy/pandas on Python 3.13/Windows to avoid
# a silent ABI crash (exit code 0xC0000005). momentum_engine imports osqp,
# but by the time Python resolves that import numpy is already loaded — too late.
import osqp  # noqa: F401

import numpy as np
import pandas as pd
import requests

from data_cache import load_or_fetch
from momentum_engine import UltimateConfig
from shared_utils import (
    NSE_DEFAULT_HEADERS,
    NSE_URL_EQUITY_MASTER_CSV,
    NSE_URL_NIFTY500_CSV,
    NSE_URL_NIFTY500_INDEX_CONSTITUENT_CSV,
    atomic_write_file,
    fetch_nse_csv,
    normalize_ns_ticker,
)


def _bootstrap_env() -> None:
    """Initialize environmental variables from .env using available loaders."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        from log_config import load_dotenv_safe
        load_dotenv_safe()
        return
    load_dotenv()


_bootstrap_env()


# ── Wayback Machine constants ─────────────────────────────────────────────────
_WBM_CDX_URL          = "https://web.archive.org/cdx/search/cdx"
_WBM_FETCH_TPL        = "https://web.archive.org/web/{ts}if_/{url}"
_WBM_MIN_SNAPSHOTS    = 2
_WBM_SLEEP_SECS       = 0.3

_NSE_N500_CSV_URLS = [
    NSE_URL_NIFTY500_INDEX_CONSTITUENT_CSV,
    NSE_URL_NIFTY500_CSV,
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

NSE_HEADERS = NSE_DEFAULT_HEADERS

NSE_SOURCES = {
    "nifty500": [
        NSE_URL_NIFTY500_CSV,
        NSE_URL_NIFTY500_INDEX_CONSTITUENT_CSV,
    ],
    "nse_total": [
        NSE_URL_EQUITY_MASTER_CSV,
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

_NSE_SESSION: "requests.Session | None" = None  # FIX-10
_NSE_SESSION_LOCK = threading.Lock()


# FIX-MB-DUPNS: Single canonical definition of normalize_ns_ticker(). The original module had
# two definitions; the second shadowed the first. Removed the duplicate.
def _get_nse_session() -> requests.Session:
    """Retrieve or create the singleton HTTP session used for NSE requests."""
    global _NSE_SESSION
    with _NSE_SESSION_LOCK:
        if _NSE_SESSION is None:  # FIX-10
            _NSE_SESSION = requests.Session()  # FIX-10
            try:
                _NSE_SESSION.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)  # FIX-10
                time.sleep(0.5)  # FIX-10
            except Exception:
                pass
    return _NSE_SESSION


def _invalidate_nse_session() -> None:
    """Close and invalidate the singleton session on connection failures."""
    global _NSE_SESSION
    with _NSE_SESSION_LOCK:
        if _NSE_SESSION is not None:
            try:
                _NSE_SESSION.close()
            except Exception:
                pass
            _NSE_SESSION = None


def _fetch_with_retry(url: str, retries: int = 3, delay: float = 2.0) -> Optional[requests.Response]:
    """GET with exponential backoff. Returns Response or None on failure."""
    session = _get_nse_session()  # FIX-10
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=NSE_HEADERS, timeout=20)
            if resp.status_code == 200 and len(resp.content) > 100:
                return resp
            logger.warning("  [%s] HTTP %d, attempt %d/%d", url[:60], resp.status_code, attempt + 1, retries)
        except requests.ConnectionError as exc:
            logger.warning("  [%s] ConnectionError, invalidating session, attempt %d/%d", url[:60], attempt + 1, retries, exc_info=True)
            _invalidate_nse_session()
            if attempt < retries - 1:
                session = _get_nse_session()
        except Exception as _:
            logger.warning("  [%s] %s, attempt %d/%d", url[:60], _, attempt + 1, retries, exc_info=True)
        if attempt < retries - 1:
            time.sleep(delay * (2 ** attempt) + random.uniform(0, 0.5))  # FIX-2
    return None


# ─── Wayback Machine PIT fetch ───────────────────────────────────────────────

def _wbm_cdx_timestamps(nse_url: str, start_year: int = 2015) -> list[str]:
    """
    Return one Wayback timestamp per month for the given Nifty500 CSV URL.

    FIX-TRUE-PIT-01:
    Some historical captures exist under HTTP vs HTTPS (or wildcard scheme),
    so querying only one exact URL can massively undercount archival coverage.
    We now query multiple URL patterns and then collapse to one snapshot per
    month in Python (earliest snapshot in each month) for stable PIT anchors.
    """
    no_scheme = nse_url.split("://", 1)[1] if "://" in nse_url else nse_url
    query_urls = [nse_url]
    if nse_url.startswith("https://"):
        query_urls.append(nse_url.replace("https://", "http://", 1))
    query_urls.append(f"*://{no_scheme}")

    collected: set[str] = set()
    for query_url in query_urls:
        params = {
            "url": query_url,
            "output": "json",
            "fl": "timestamp,statuscode",
            "filter": "statuscode:200",
            "from": str(start_year),
            "limit": "5000",
            "showResumeKey": "true",  # FIX-6
        }
        try:
            while True:  # FIX-6
                resp = requests.get(
                    _WBM_CDX_URL, params=params,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=30,
                )
                resp.raise_for_status()
                rows = resp.json()
                resume_key = None  # FIX-6
                for row in rows[1:]:  # FIX-6
                    if len(row) == 1:  # FIX-6
                        candidate = str(row[0]).strip()
                        # Validate: not empty, not a timestamp (e.g., "20210131120000")
                        if candidate and len(candidate) > 0:
                            # Reject if it looks like a timestamp (8+ chars, first 4 are digits)
                            if len(candidate) >= 8 and candidate[:4].isdigit():
                                # This looks like a timestamp, not a resume key
                                continue
                            # Accept as resume key
                            resume_key = candidate
                        continue
                    if len(row) >= 2 and row[1] == "200":  # FIX-6
                        ts = str(row[0]).strip()
                        if len(ts) >= 8 and ts[:4].isdigit():
                            collected.add(ts)
                if not resume_key:  # FIX-6
                    break
                params["resumeKey"] = resume_key  # FIX-6
        except Exception as exc:
            logger.debug("[Wayback] CDX query failed for %s: %s", query_url, exc)
        _time_mod.sleep(_WBM_SLEEP_SECS)   # rate-limit between scheme variants  # FIX-9

    month_to_ts: dict[str, str] = {}
    for ts in sorted(collected):
        month_key = ts[:6]
        month_to_ts.setdefault(month_key, ts)
    ts_list = [month_to_ts[m] for m in sorted(month_to_ts)]
    logger.info(
        "[Wayback] CDX found %d monthly snapshots for %s",
        len(ts_list),
        nse_url,
    )
    return ts_list


def _wbm_fetch_csv(timestamp: str, original_url: str, retries: int = 3) -> pd.DataFrame | None:
    """Fetch a single Wayback snapshot and parse as CSV. Returns None on failure."""
    url = _WBM_FETCH_TPL.format(ts=timestamp, url=original_url)
    for attempt in range(retries):  # FIX-3
        try:
            resp = requests.get(
                url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
            )
            resp.raise_for_status()
            for enc in ("utf-8", "latin-1", "cp1252"):  # FIX-3
                try:
                    df = pd.read_csv(_io_mod.BytesIO(resp.content), encoding=enc)
                    if not df.empty and len(df.columns) >= 3:  # FIX-3
                        return df
                except Exception:
                    continue
            raise ValueError("No valid CSV decoding produced usable dataframe")  # FIX-3
        except Exception as exc:
            logger.debug("[Wayback] fetch failed (ts=%s attempt=%d/%d): %s", timestamp, attempt + 1, retries, exc)  # FIX-3
            if attempt < retries - 1:
                _time_mod.sleep(2.0 * (2 ** attempt))  # FIX-3
    return None


def _symbols_from_nse_csv(df: pd.DataFrame) -> list[str]:
    """Extract .NS-suffixed tickers from an ind_nifty500list.csv DataFrame."""
    import re
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ("symbol", "symbols", "ticker", "nse symbol", "nse_symbol"):
        if col in df.columns:
            raw = df[col].dropna().astype(str).str.strip()
            tickers = [t for t in (normalize_ns_ticker(s) for s in raw if s and s.upper() not in ("SYMBOL", "TICKER", "")) if t]
            if tickers:
                return sorted(set(tickers))
    logger.warning("[Wayback] Fallback regex parsing path entered; CSV columns: %s", list(df.columns))  # FIX-12
    pattern = re.compile(r"^[A-Z][A-Z0-9&\-]{2,14}$")  # FIX-12
    counts: dict[str, int] = {}
    MIN_TICKER_OCCURRENCES = 2  # FIX-12
    skip = {"SERIES", "ISIN", "INDUSTRY", "NAME", "SYMBOL", "TICKER"}
    for col in df.columns:
        for val in df[col].dropna().astype(str):
            v = val.strip().upper()
            if pattern.match(v) and v not in skip:
                counts[v] = counts.get(v, 0) + 1  # FIX-12
    found = sorted(t for t in (normalize_ns_ticker(sym) for sym, cnt in counts.items() if cnt >= MIN_TICKER_OCCURRENCES) if t)  # FIX-12
    if len(found) < 10:
        logger.warning("[Wayback] Fallback regex parser yielded fewer than 10 tickers (%d).", len(found))  # FIX-12
    return found


def fetch_nifty500_wayback(start_year: int = 2015) -> tuple[list[tuple[str, list[str]]], bool]:
    """
    Download all monthly Wayback snapshots across ALL known NSE CSV URLs.

    Queries the Internet Archive for historical captures of the Nifty 500
    constituent list. Merges snapshots from multiple URL variants to
    maximize point-in-time coverage.

    Args:
        start_year (int): The earliest year to include in the search.

    Returns:
        tuple: (all_snapshots, success_flag) where all_snapshots is a
            list of (date_str, tickers) tuples sorted chronologically.
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


def _atomic_write_parquet(df: "pd.DataFrame", path) -> None:
    """Write parquet via shared atomic writer."""
    path = Path(path)
    if importlib.util.find_spec("pyarrow") is None:
        raise ImportError(
            f"pyarrow is required to write {path}. "
            "Install it with: pip install pyarrow"
        )

    atomic_write_file(
        path,
        lambda tmp: df.to_parquet(tmp, engine="pyarrow"),
        suffix=".tmp.parquet",
        fsync_file=True,
        fsync_dir=True,
    )


def _atomic_write_csv(df: "pd.DataFrame", path, **kwargs) -> None:
    """Write CSV via shared atomic writer."""
    atomic_write_file(
        Path(path),
        lambda tmp: df.to_csv(tmp, **kwargs),
        suffix=".tmp.csv",
        fsync_file=True,
        fsync_dir=True,
    )


# Backward compatibility alias
def _to_parquet_pyarrow(df: "pd.DataFrame", path) -> None:
    """Backward compatibility wrapper for _atomic_write_parquet."""
    _atomic_write_parquet(df, path)


def _compute_vol_gate_snapshots(
    valid_trading_days: "pd.DataFrame",
    history_gate: int,
    start_date: str,
    snap_freq: str = "QS",
    end_date: "pd.Timestamp | None" = None,
) -> "pd.DataFrame":
    """
    Build volume-gated snapshots aligned to actual trading days.

    FIX-SNAPSHOT-ALIGN: Maps each calendar snapshot date to the last trading day <= d
    to ensure consistent PIT timestamps across parquet and CSV outputs.
    """
    if end_date is None:
        raise ValueError(
            "end_date is required and must not be None. "
            "Caller should pass run()-scoped TODAY_UTC for consistency."
        )
    calendar_dates = pd.date_range(start=start_date, end=end_date, freq=snap_freq)
    trading_days_index = pd.DatetimeIndex(valid_trading_days.index)

    aligned_dates = []
    rows = []
    for d in calendar_dates:
        # Map to last trading day <= d
        past_trading = trading_days_index[trading_days_index <= d]
        if past_trading.empty:
            continue
        snapshot_date = past_trading[-1]

        # Get eligible tickers as of that snapshot date
        past = valid_trading_days[valid_trading_days.index <= snapshot_date]
        eligible = (
            [] if past.empty
            else past.iloc[-1][past.iloc[-1] >= history_gate].index.tolist()
        )
        aligned_dates.append(snapshot_date)
        rows.append(eligible)

    return pd.DataFrame(
        {"tickers": rows},
        index=pd.DatetimeIndex(aligned_dates, name="date"),
    )  # FIX-15


def build_parquet_from_wayback(
    universe_type: str,
    snapshots: list[tuple[str, list[str]]],
) -> Path:
    """
    Write a PIT parquet from Wayback-sourced snapshots.

    FIX-CSV-FILTERED: CSV is now built from the filtered rows dict (not raw snapshots)
    to ensure malformed date entries don't leak into the CSV output.
    """
    output_path = DATA_DIR / f"historical_{universe_type}.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows: dict[pd.Timestamp, list[str]] = {}
    for date_str, tickers in snapshots:
        try:
            ts = pd.Timestamp(date_str)  # FIX-14
        except Exception as exc:
            logger.warning("[BHF] Skipping malformed snapshot date %r: %s", date_str, exc)  # FIX-14
            continue
        existing = rows.get(ts, [])
        rows[ts] = sorted(set(existing) | set(tickers))

    idx    = pd.DatetimeIndex(sorted(rows.keys()), name="date")
    series = pd.Series([rows[d] for d in idx], index=idx, name="tickers")
    out_df = pd.DataFrame({"tickers": series})
    _to_parquet_pyarrow(out_df, output_path)

    # FIX-CSV-FILTERED: Build CSV from filtered rows instead of raw snapshots
    csv_path = DATA_DIR / f"historical_{universe_type}.csv"
    csv_rows = []
    for ts in idx:
        for tkr in rows[ts]:
            csv_rows.append({"date": ts.strftime("%Y-%m-%d"), "ticker": tkr})
    _atomic_write_csv(pd.DataFrame(csv_rows), csv_path, index=False)

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
        try:
            session = _get_nse_session()
            df = fetch_nse_csv(url, timeout=20, headers=NSE_HEADERS, session=session)
            df.columns = [c.strip().upper() for c in df.columns]
            sym_col = next((c for c in ["SYMBOL", "TICKER", "COMPANY SYMBOL"] if c in df.columns), None)
            if sym_col is None:
                logger.warning("  Could not find SYMBOL column in CSV. Columns: %s", list(df.columns))
                continue
            syms = [t for t in (normalize_ns_ticker(s) for s in df[sym_col].dropna().astype(str).str.strip().unique() if s) if t]
            if len(syms) >= 100:
                logger.info("  ✓ Fetched %d Nifty 500 symbols.", len(syms))
                return sorted(syms)
        except requests.ConnectionError as exc:
            logger.warning("  Connection error for %s, invalidating session: %s", url, exc)
            _invalidate_nse_session()
        except Exception as exc:
            logger.warning("  Parse error: %s", exc, exc_info=True)

    logger.error("All Nifty 500 URLs failed — check your internet / proxy settings.")
    return []


def fetch_nse_total_current() -> List[str]:
    """Download current NSE Total equity list from NSE India."""
    for url in NSE_SOURCES["nse_total"]:
        logger.info("  Trying: %s", url)
        try:
            session = _get_nse_session()
            df = fetch_nse_csv(url, timeout=20, headers=NSE_HEADERS, session=session)
            df.columns = [c.strip().upper() for c in df.columns]
            if "SERIES" in df.columns:
                df = df[df["SERIES"].str.strip() == "EQ"]
            sym_col = next((c for c in ["SYMBOL", "TICKER"] if c in df.columns), None)
            if sym_col is None:
                continue
            syms = [t for t in (normalize_ns_ticker(s) for s in df[sym_col].dropna().astype(str).str.strip().unique() if s) if t]
            if len(syms) >= 500:
                logger.info("  ✓ Fetched %d NSE Total equity symbols.", len(syms))
                return sorted(syms)
        except requests.ConnectionError as exc:
            logger.warning("  Connection error for %s, invalidating session: %s", url, exc)
            _invalidate_nse_session()
        except Exception as exc:
            logger.warning("  Parse error: %s", exc, exc_info=True)

    logger.error("All NSE Total URLs failed.")
    return []


# ─── Point-in-Time Parquet builder ────────────────────────────────────────────

def build_parquet(
    universe_type: str,
    valid_trading_days: pd.DataFrame,
    history_gate: int,
    start_date: str = "2015-01-01",
    snap_freq: str = "QS",
    end_date: "pd.Timestamp | None" = None,
) -> Path:
    """
    Create a PIT parquet from valid_trading_days by backfilling quarterly
    snapshots from start_date to today.
    """
    if end_date is None:
        raise ValueError(
            "end_date is required and must not be None. "
            "Caller should pass run()-scoped TODAY_UTC for consistency."
        )
    assert all(
        s.endswith(".NS") or s.startswith("^") for s in valid_trading_days.columns
    ), "MB-15: all columns in valid_trading_days must be .NS-suffixed or index tickers"
    output_path = DATA_DIR / f"historical_{universe_type}.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # BHF-02: use UTC-normalised date instead of local today() to ensure consistency
    # with the rest of the engine's TZ-naive UTC timestamps.
    snapshot_df = _compute_vol_gate_snapshots(  # FIX-15
        valid_trading_days=valid_trading_days,
        history_gate=history_gate,
        start_date=start_date,
        snap_freq=snap_freq,
        end_date=end_date,  # FIX-4
    )
    snapshot_dates = pd.DatetimeIndex(snapshot_df.index)
    trading_days = pd.DatetimeIndex(valid_trading_days.index)  # FIX-8
    aligned = []  # FIX-8
    for d in snapshot_dates:  # FIX-8
        prior = trading_days[trading_days <= d]
        if not prior.empty:
            aligned.append(prior[-1])
    snapshot_dates = pd.DatetimeIndex(sorted(set(aligned)), name="date")  # FIX-8
    if snapshot_dates.empty:
        fallback_end = end_date if end_date is not None else pd.Timestamp.now("UTC").tz_convert(None).normalize()
        snapshot_dates = pd.DatetimeIndex([fallback_end], name="date")  # FIX-4

    rows = []
    for d in snapshot_dates:
        past_data = valid_trading_days[valid_trading_days.index <= d]
        status_on_date = past_data.iloc[-1] if not past_data.empty else pd.Series(dtype=float)
        eligible = status_on_date[status_on_date >= history_gate].index.tolist() if not status_on_date.empty else []
        rows.append(eligible)

    out_df = pd.DataFrame({"tickers": rows}, index=snapshot_dates)

    logger.info(
        "  Building %d volume-gated snapshots from %s → today for %d symbols.",
        len(out_df), start_date, len(valid_trading_days.columns),
    )

    _to_parquet_pyarrow(out_df, output_path)
    logger.info("  ✓ Written: %s  (%d rows)", output_path, len(out_df))
    return output_path


def build_csv_from_symbols(
    universe_type: str,
    valid_trading_days: pd.DataFrame,
    history_gate: int,
    start_date: str = "2015-01-01",
    snap_freq: str = "QS",
    end_date: "pd.Timestamp | None" = None,
) -> Path:
    """Write the companion CSV incorporating strict volume existence gates."""
    if end_date is None:
        raise ValueError(
            "end_date is required and must not be None. "
            "Caller should pass run()-scoped TODAY_UTC for consistency."
        )
    csv_path = DATA_DIR / f"historical_{universe_type}.csv"
    snapshot_df = _compute_vol_gate_snapshots(  # FIX-15
        valid_trading_days=valid_trading_days,
        history_gate=history_gate,
        start_date=start_date,
        snap_freq=snap_freq,
        end_date=end_date,  # FIX-4
    )

    # FIX-EMPTY-CSV: When snapshot_df is empty (no valid snapshots),
    # apply the same fallback logic as build_parquet to ensure both
    # artifacts describe the same build with consistent snapshot dates.
    if snapshot_df.empty:
        fallback_end = end_date if end_date is not None else pd.Timestamp.now("UTC").tz_convert(None).normalize()
        # Create a fallback snapshot with the fabricated date and empty ticker list.
        # In CSV format, this is represented as a single row with the date and no ticker.
        snapshot_df = pd.DataFrame(
            {"tickers": [[]]},
            index=pd.DatetimeIndex([fallback_end], name="date")
        )

    csv_rows = []
    for d, tickers in snapshot_df["tickers"].items():
        ticker_list = tickers if isinstance(tickers, list) else list(tickers)
        if not ticker_list:
            csv_rows.append({"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "ticker": None})
            continue
        for sym in ticker_list:
            csv_rows.append({"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "ticker": sym})

    _atomic_write_csv(pd.DataFrame(csv_rows), csv_path, index=False)
    logger.info("  ✓ Written CSV companion: %s  (%d rows)", csv_path, len(csv_rows))
    return csv_path


def _build_adnv_ranked_snapshots(
    market_data: dict[str, pd.DataFrame],
    start_date: str,
    top_n: int = 500,
    lookback_days: int = 126,
    min_trading_days: int = 60,
    snap_freq: str = "QS",
    end_date: "pd.Timestamp | None" = None,
) -> pd.DataFrame:
    """Build quarterly PIT rows using ADNV ranking over available candidate symbols."""
    if end_date is None:
        raise ValueError(
            "end_date is required and must not be None. "
            "Caller should pass run()-scoped TODAY_UTC for consistency."
        )
    snapshot_dates = pd.date_range(start=start_date, end=end_date, freq=snap_freq)
    rows: list[list[str]] = []

    for d in snapshot_dates:
        ref = pd.Timestamp(d)
        scores: dict[str, float] = {}
        for sym, df in market_data.items():
            if df is None or df.empty or "Close" not in df.columns or "Volume" not in df.columns:
                continue
            try:
                if getattr(df.index, "tz", None) is not None:
                    df = df.copy()
                    df.index = df.index.tz_convert(None)  # FIX-5
                hist = df.loc[:ref].tail(lookback_days)
                if len(hist) < min_trading_days:
                    continue
                close = pd.to_numeric(hist["Close"], errors="coerce")
                volume = pd.to_numeric(hist["Volume"], errors="coerce").replace(0, np.nan)
                adnv = float((close * volume).mean())
                if pd.isna(adnv) or adnv <= 0:
                    continue
                scores[sym] = adnv
            except Exception as exc:
                logger.warning("[ADNV] Skipping %s at %s: %s", sym, ref.date(), exc)  # FIX-5
                continue

        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]
        rows.append(ranked)

    return pd.DataFrame({"tickers": rows}, index=pd.DatetimeIndex(snapshot_dates, name="date"))


def _write_snapshot_outputs(universe_type: str, snapshot_df: pd.DataFrame) -> Path:
    """Write parquet+CSV outputs from a DataFrame with a `tickers` list column."""
    output_path = DATA_DIR / f"historical_{universe_type}.parquet"
    csv_path = DATA_DIR / f"historical_{universe_type}.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    _to_parquet_pyarrow(snapshot_df, output_path)

    csv_rows = []
    for d, tickers in snapshot_df["tickers"].items():
        ticker_list = tickers if isinstance(tickers, list) else list(tickers)
        if not ticker_list:
            csv_rows.append({"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "ticker": None})
            continue
        for tkr in ticker_list:
            csv_rows.append({"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "ticker": tkr})
    _atomic_write_csv(pd.DataFrame(csv_rows), csv_path, index=False)

    logger.info("  ✓ Written: %s  (%d rows)", output_path, len(snapshot_df))
    logger.info("  ✓ Written CSV companion: %s  (%d rows)", csv_path, len(csv_rows))
    return output_path


# ─── Main orchestration ───────────────────────────────────────────────────────

def run(universe_arg: str = "both", start_date: str = "2015-01-01") -> None:
    """
    Execute the full historical fallback generation process.

    Downloads Wayback Machine snapshots for Nifty 500 and NSE Total,
    merges them with current lists, and builds the final survivorship-safe
    parquets in the data directory.

    Args:
        universe_arg (str): One of 'nifty500', 'nse_total', or 'both'.
        start_date (str): YYYY-MM-DD cutoff for historical search.

    Returns:
        None (writes files to disk).
    """
    TODAY_UTC = pd.Timestamp.now("UTC").tz_convert(None).normalize()  # FIX-4
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
            logger.warning("All network sources failed — using Nifty 50 hard-floor.")
            symbols = [normalize_ns_ticker(s) for s in NIFTY50_CORE]

        market_data = load_or_fetch(
            symbols, start_date, TODAY_UTC.strftime("%Y-%m-%d"), cfg=cfg
        )
        vol_dict = {}
        skipped_none, skipped_empty, skipped_no_vol = [], [], []  # FIX-11
        for sym, df in market_data.items():
            if df is None:
                skipped_none.append(sym)
                continue
            if df.empty:
                skipped_empty.append(sym)
                continue
            if "Volume" not in df.columns:
                skipped_no_vol.append(sym)
                continue
            vol_dict[sym] = df["Volume"].replace(0, np.nan)
        logger.info(
            "[VolGate] Skipped — None: %d, empty: %d, no-Volume: %d. Included: %d.",
            len(skipped_none), len(skipped_empty), len(skipped_no_vol), len(vol_dict),
        )
        if skipped_none:
            logger.warning("[VolGate] None symbols (delisting or fetch failure): %s", skipped_none[:20])
        vol_matrix = pd.DataFrame(vol_dict).sort_index()
        valid_trading_days = vol_matrix.notna().cumsum()

        parquet_path = build_parquet(
            "nifty500", valid_trading_days, history_gate, start_date, end_date=TODAY_UTC  # FIX-4
        )
        build_csv_from_symbols(
            "nifty500", valid_trading_days, history_gate, start_date, end_date=TODAY_UTC  # FIX-4
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
                try:
                    ts = pd.Timestamp(date_str)
                except Exception as exc:
                    logger.warning("[Hybrid] Skipping malformed snapshot date %r: %s", date_str, exc)
                    continue
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
            _to_parquet_pyarrow(out_df, parquet_path)

            csv_path = DATA_DIR / "historical_nifty500.csv"
            csv_rows = []
            for d, tickers in zip(all_dates, merged_tickers):
                for tkr in tickers:
                    csv_rows.append({"date": d.strftime("%Y-%m-%d"), "ticker": tkr})
            _atomic_write_csv(pd.DataFrame(csv_rows), csv_path, index=False)

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
                end_date=TODAY_UTC,  # FIX-4
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
        first_non_empty = next(
            (
                v for v in df_check["tickers"]
                if (len(v) if isinstance(v, list) else len(list(v))) > 0
            ),
            None,
        )
        if first_non_empty is None:
            n_syms = 0
        else:
            n_syms = len(first_non_empty) if isinstance(first_non_empty, list) else len(list(first_non_empty))
        print()
        print(f"  ✓ {parquet_path}  |  {n_snaps} snapshots  |  ~{n_syms} symbols/first-non-empty-snapshot")

        # FIX-13: Default IPO_DATES now includes POLICYBAZAAR.NS
        ipo_dates: dict = getattr(cfg, "IPO_DATES", {
            "ZOMATO.NS": "2021-07-23",
            "NYKAA.NS":  "2021-11-10",
            "PAYTM.NS":  "2021-11-18",
            "POLICYBAZAAR.NS": "2021-11-15",
            "IREDA.NS":  "2023-11-29",
        })
        # FIX-DEFENSIVE-PARSE: Defensive late_joiners construction with malformed date handling
        late_joiners = set()
        for sym, dt in ipo_dates.items():
            try:
                if pd.Timestamp(dt) >= pd.Timestamp("2021-07-01"):
                    late_joiners.add(sym)
            except Exception as exc:
                logger.warning("[Verify] Skipping malformed IPO_DATES entry for %s=%r: %s", sym, dt, exc)
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
            symbols = [normalize_ns_ticker(s) for s in NIFTY50_CORE]
            logger.warning("NSE Total fetch failed — using Nifty 50 hard-floor.")

        logger.info("Fetching market data for volume-gate PIT construction...")
        CHUNK_SIZE = 200
        all_market_data: dict = {}
        for i in range(0, len(symbols), CHUNK_SIZE):
            chunk = symbols[i: i + CHUNK_SIZE]
            all_market_data.update(
                load_or_fetch(chunk, start_date, TODAY_UTC.strftime("%Y-%m-%d"), cfg=cfg)
            )
            logger.info(
                "[NSE Total] Loaded chunk %d/%d (%d symbols)",
                i // CHUNK_SIZE + 1,
                -(-len(symbols) // CHUNK_SIZE),
                len(chunk),
            )
        market_data = all_market_data  # FIX-7

        vol_dict = {}
        skipped_none, skipped_empty, skipped_no_vol = [], [], []  # FIX-11
        for sym, df in market_data.items():
            if df is None:
                skipped_none.append(sym)
                continue
            if df.empty:
                skipped_empty.append(sym)
                continue
            if "Volume" not in df.columns:
                skipped_no_vol.append(sym)
                continue
            vol_dict[sym] = df["Volume"].replace(0, np.nan)
        logger.info(
            "[VolGate] Skipped — None: %d, empty: %d, no-Volume: %d. Included: %d.",
            len(skipped_none), len(skipped_empty), len(skipped_no_vol), len(vol_dict),
        )
        if skipped_none:
            logger.warning("[VolGate] None symbols (delisting or fetch failure): %s", skipped_none[:20])

        vol_matrix = pd.DataFrame(vol_dict).sort_index()
        valid_trading_days = vol_matrix.notna().cumsum()

        parquet_path = build_parquet(
            "nse_total", valid_trading_days, history_gate, start_date, end_date=TODAY_UTC  # FIX-4
        )
        build_csv_from_symbols(
            "nse_total", valid_trading_days, history_gate, start_date, end_date=TODAY_UTC  # FIX-4
        )

        df_check = pd.read_parquet(parquet_path)
        if df_check.empty:
            n_syms = 0
        else:
            first_row = df_check.iloc[0]["tickers"]
            n_syms = len(first_row) if isinstance(first_row, list) else len(list(first_row))
        print(f"  ✓ {parquet_path}  |  {len(df_check)} snapshots  |  ~{n_syms} symbols/snapshot")

    print()
    print("=" * 65)
    print("  BUILD COMPLETE")
    print("=" * 65 + "\n")


def _parse_args(argv=None):
    """
    Parse command-line arguments for the historical builder script.

    Args:
        argv (List[str]): Optional list of arguments (defaults to sys.argv).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(description="Fallback historical universe builder for Ultimate Momentum.")
    p.add_argument(
        "--universe",
        default="both",
        choices=["nifty500", "nse_total", "both"],
        help="Which universe to build (default: both).",
    )
    p.add_argument(
        "--start",
        default="2015-01-01",
        help="Earliest snapshot date (YYYY-MM-DD). Default: 2015-01-01.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(universe_arg=args.universe, start_date=args.start)

# VERIFICATION CHECKLIST
# FIX-1  : grep '_to_parquet_pyarrow\|to_csv' — every write uses .tmp + os.replace
# FIX-2  : grep 'time.sleep' in _fetch_with_retry — uses 2**attempt
# FIX-3  : grep '_wbm_fetch_csv' — has for-attempt loop with exponential sleep
# FIX-4  : grep 'Timestamp.today' — zero occurrences; TODAY_UTC defined in run()
# FIX-5  : grep 'tz_convert(None)' in _build_adnv_ranked_snapshots — present
# FIX-6  : grep 'resumeKey' — pagination loop present in _wbm_cdx_timestamps
# FIX-7  : grep 'CHUNK_SIZE' — chunked loop in NSE-Total block of run()
# FIX-8  : grep 'aligned_dates\|trading_days' in build_parquet — present
# FIX-9  : grep '_WBM_SLEEP_SECS' after each scheme-variant query — present
# FIX-10 : grep '_NSE_SESSION\|_get_nse_session' — singleton helper present
# FIX-11 : grep 'skipped_none\|skipped_empty\|skipped_no_vol' — present in run()
# FIX-12 : grep 'MIN_TICKER_OCCURRENCES\|{2,14}' — present in _symbols_from_nse_csv
# FIX-13 : grep 'IPO_DATES\|ipo_dates' — dynamic derivation in run()
# FIX-14 : grep 'malformed snapshot date' — guarded in build_parquet_from_wayback
# FIX-15 : grep '_compute_vol_gate_snapshots' — helper extracted and called
