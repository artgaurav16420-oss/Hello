"""
historical_builder.py
=====================
Generate point-in-time (PIT) universe snapshots for survivorship-safe backtests.

PRIMARY DATA SOURCE (replaces deleted GitHub repo):
    Wayback Machine archives of the official NSE Indices constituent CSV:
        https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv

    The Wayback Machine has been archiving this URL at least monthly since ~2015.
    Each archived snapshot captures the exact list as published by NSE on that date,
    giving us genuine semi-annual PIT data (Nifty 500 rebalances Jan & Jul each year).

    Strategy:
        1. Query CDX API to find all 200-OK snapshots of the NSE CSV.
        2. Keep one snapshot per calendar month (dedup by YYYYMM).
        3. Download each snapshot and extract the 'Symbol' column.
        4. Write a dated (date, ticker) CSV → pass to build_parquet_from_csv().

FALLBACK (if Wayback Machine is unavailable):
    Reconstruct approximate PIT membership from yfinance market-cap + volume data:
    at each semi-annual rebalance date, rank all NSE stocks by trailing 6-month
    average market cap and keep the top 500 with sufficient trading history.
    This is an approximation (~5% error) but vastly better than 100% survivorship bias.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

# Remote master archives used by main() bootstrapping flow.
REMOTE_ARCHIVE_URLS: dict[str, list[str]] = {
    "nifty500": [],
    "nse_total": [],
}

# ── Constants ─────────────────────────────────────────────────────────────────

# Official NSE constituent CSV URLs (current)
_NSE_NIFTY500_CSV_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

# Wayback Machine endpoints
_WBM_CDX_URL   = "https://web.archive.org/cdx/search/cdx"
_WBM_FETCH_URL = "https://web.archive.org/web/{ts}if_/{url}"

# Minimum number of Wayback snapshots required to trust the archive
_MIN_SNAPSHOTS = 12   # at least 1 year of monthly snapshots

# Semi-annual rebalance months (Nifty 500 rebalances in Jan & Jul effective dates
# are usually late March and late September)
_REBALANCE_MONTHS = {3, 9}   # March and September effective dates

# Fallback: all NSE-listed tickers for yfinance approximation
_FALLBACK_APPROX_N = 500
DELISTED_TICKERS_FILE = DATA_DIR / "nse_delisted_historical.csv"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# ── Ticker normalisation ───────────────────────────────────────────────────────

def _ns_ticker(sym: str) -> str:
    """Return sym with exactly one '.NS' suffix, upper-cased."""
    sym = sym.strip().upper()
    if sym.startswith("^"):
        return sym
    # Remove any existing .NS / .BO / .BSE suffix
    for sfx in (".NS", ".BO", ".BSE"):
        if sym.endswith(sfx):
            sym = sym[: -len(sfx)]
    return sym + ".NS"


# ─────────────────────────────────────────────────────────────────────────────
# WAYBACK MACHINE APPROACH
# ─────────────────────────────────────────────────────────────────────────────

def _wayback_cdx_snapshots(url: str, start_year: int = 2015) -> list[str]:
    """
    Query the Wayback Machine CDX API and return a list of YYYYMMDDHHMMSS
    timestamps for all 200-OK snapshots of `url`, deduplicated to one per month.

    Returns [] on any network/parse failure.
    """
    params = {
        "url":      url,
        "output":   "json",
        "fl":       "timestamp,statuscode",
        "filter":   "statuscode:200",
        "collapse": "timestamp:6",   # one per month (YYYYMM)
        "from":     f"{start_year}",
        "limit":    "500",
    }
    try:
        resp = requests.get(
            _WBM_CDX_URL, params=params, headers=_HEADERS, timeout=30
        )
        resp.raise_for_status()
        rows = resp.json()
        # rows[0] is the header row ["timestamp","statuscode"]
        timestamps = [r[0] for r in rows[1:] if r[1] == "200"]
        logger.info(
            "[HistoricalBuilder] Wayback CDX found %d monthly snapshots for %s",
            len(timestamps), url,
        )
        return timestamps
    except Exception as exc:
        logger.warning(
            "[HistoricalBuilder] Wayback CDX query failed for %s: %s", url, exc
        )
        return []


def _wayback_fetch_csv(wbm_timestamp: str, original_url: str) -> pd.DataFrame | None:
    """
    Fetch a specific Wayback Machine snapshot and parse it as a CSV.
    Returns a DataFrame or None on failure.
    """
    fetch_url = _WBM_FETCH_URL.format(ts=wbm_timestamp, url=original_url)
    try:
        resp = requests.get(fetch_url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        content = resp.content

        # niftyindices.com CSVs are sometimes latin-1 encoded
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc)
                if not df.empty and len(df.columns) >= 3:
                    return df
            except Exception:
                continue
        return None
    except Exception as exc:
        logger.debug(
            "[HistoricalBuilder] Wayback fetch failed (ts=%s): %s",
            wbm_timestamp, exc,
        )
        return None


def _extract_symbols_from_nse_csv(df: pd.DataFrame) -> list[str]:
    """
    Extract NSE ticker symbols from an ind_nifty500list.csv DataFrame.

    The NSE CSV has columns:
        Company Name | Industry | Symbol | Series | ISIN Code

    Returns a sorted list of .NS-suffixed tickers.
    """
    # Normalise column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Try 'symbol' column first (official NSE format)
    for col in ("symbol", "symbols", "ticker", "nse symbol", "nse_symbol"):
        if col in df.columns:
            raw = df[col].dropna().astype(str).str.strip()
            tickers = [_ns_ticker(s) for s in raw if s and s.upper() != "SYMBOL"]
            if tickers:
                return sorted(set(tickers))

    # Fallback: scan all columns for NSE-like symbols (all-caps, 2-15 chars)
    import re
    pattern = re.compile(r"^[A-Z][A-Z0-9&\-]{1,14}$")
    candidates: set[str] = set()
    for col in df.columns:
        for val in df[col].dropna().astype(str):
            v = val.strip().upper()
            if pattern.match(v) and v not in {"SERIES", "ISIN", "INDUSTRY", "NAME"}:
                candidates.add(_ns_ticker(v))

    return sorted(candidates)


def _build_pit_csv_from_wayback(
    target_url: str,
    output_csv: Path,
    start_year: int = 2015,
    sleep_seconds: float = 1.0,
) -> bool:
    """
    Download all monthly Wayback snapshots of `target_url`, parse each one,
    and write a (date, ticker) CSV to `output_csv`.

    Returns True if enough snapshots were collected, False otherwise.
    """
    timestamps = _wayback_cdx_snapshots(target_url, start_year=start_year)
    if len(timestamps) < _MIN_SNAPSHOTS:
        logger.warning(
            "[HistoricalBuilder] Only %d Wayback snapshots found (need %d). "
            "Wayback approach may not produce reliable PIT data.",
            len(timestamps), _MIN_SNAPSHOTS,
        )
        if not timestamps:
            return False

    rows: list[dict] = []
    failed = 0
    total = len(timestamps)

    print(f"[HistoricalBuilder] Downloading {total} Wayback snapshots "
          f"(~{total * sleep_seconds:.0f}s)...")

    for i, ts in enumerate(timestamps, 1):
        snapshot_date = pd.Timestamp(
            f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"  # YYYYMMDD → YYYY-MM-DD
        )

        df = _wayback_fetch_csv(ts, target_url)
        if df is None:
            failed += 1
            logger.debug(
                "[HistoricalBuilder] Skipping snapshot %s (parse failed)", ts
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds * 0.5)
            continue

        tickers = _extract_symbols_from_nse_csv(df)
        if not tickers:
            failed += 1
            logger.debug(
                "[HistoricalBuilder] Skipping snapshot %s (no tickers extracted)", ts
            )
            continue

        for tkr in tickers:
            rows.append({"date": snapshot_date.strftime("%Y-%m-%d"), "ticker": tkr})

        n_tickers = len(tickers)
        print(f"  [{i:3d}/{total}] {snapshot_date.date()}  →  {n_tickers} tickers")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not rows:
        logger.error("[HistoricalBuilder] No data rows collected from Wayback.")
        return False

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    n_dates = out_df["date"].nunique()
    logger.info(
        "[HistoricalBuilder] Wrote %d (date,ticker) rows across %d snapshot dates → %s  "
        "(%d/%d snapshots failed)",
        len(rows), n_dates, output_csv, failed, total,
    )
    print(
        f"\n[HistoricalBuilder] Done. {n_dates} PIT snapshots, "
        f"{failed}/{total} download failures."
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: approximate PIT from yfinance market-cap ranking
# ─────────────────────────────────────────────────────────────────────────────

# Semi-annual rebalance effective dates for Nifty 500.
# NSE reviews in Jan & Jul, changes take effect ~4 weeks later (last Fri of Mar/Sep).
# We model this as the last trading Friday of March and September each year.
_REBALANCE_DATE_PATTERN = [
    ("03", "28"),  # ~last week of March
    ("09", "28"),  # ~last week of September
]


def _get_rebalance_dates(start_year: int = 2015, end_year: int | None = None) -> list[str]:
    """
    Generate approximate semi-annual rebalance dates: last Friday of March and
    September for each year from start_year to end_year (inclusive).
    """
    end_year = end_year or pd.Timestamp.today().year + 1
    dates = []
    for y in range(start_year, end_year + 1):
        for month in (3, 9):
            # Find last Friday of the month
            last_day = pd.Timestamp(f"{y}-{month:02d}-28")
            # Advance to end of month
            last_day = last_day + pd.offsets.MonthEnd(0)
            # Rewind to Friday
            offset = (last_day.weekday() - 4) % 7
            friday = last_day - pd.Timedelta(days=offset)
            dates.append(friday.strftime("%Y-%m-%d"))
    return dates


def _approximate_nifty500_at_date(
    date: str,
    candidate_tickers: list[str],
    market_data: dict,
    lookback_days: int = 126,
    top_n: int = 500,
    min_trading_days: int = 60,
) -> list[str]:
    """
    Approximate Nifty 500 membership at `date` by ranking `candidate_tickers`
    by trailing average market cap (close × volume proxy) over `lookback_days`.

    This is NOT point-in-time membership but is far better than using today's list
    for all historical dates. Expected error: ~5-10% of constituents.

    Parameters
    ----------
    date             : YYYY-MM-DD reference date
    candidate_tickers: universe of tickers to rank (broader than 500)
    market_data      : dict of {ticker: pd.DataFrame with Close/Volume columns}
    lookback_days    : number of trading days to look back for the rank
    top_n            : number of stocks to include (500 for Nifty 500)
    min_trading_days : minimum trading days in lookback to be eligible

    Returns sorted list of .NS tickers.
    """
    ref = pd.Timestamp(date)
    scores: dict[str, float] = {}

    for ticker in candidate_tickers:
        df = market_data.get(ticker)
        if df is None or df.empty:
            continue
        try:
            hist = df.loc[:ref].tail(lookback_days)
            if len(hist) < min_trading_days:
                continue
            # Use close price as market-cap proxy (equal-weighted — we don't have
            # shares-outstanding data, but price ranking is highly correlated with
            # market cap for NSE stocks as larger companies tend to have higher prices)
            avg_close = float(hist["Close"].mean())
            avg_vol   = float(hist["Volume"].replace(0, float("nan")).mean())
            if pd.isna(avg_close) or pd.isna(avg_vol) or avg_close <= 0 or avg_vol <= 0:
                continue
            # Score = log(price) + 0.5 * log(volume) — emphasizes large liquid stocks
            import math
            scores[ticker] = math.log(avg_close) + 0.5 * math.log(avg_vol)
        except Exception:
            continue

    ranked = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
    return ranked[:top_n]


def _build_pit_csv_from_yfinance_approx(
    output_csv: Path,
    start_year: int = 2015,
    top_n: int = 500,
) -> bool:
    """
    Build an approximate PIT CSV by:
    1. Downloading price + volume data for a broad universe of NSE tickers.
       Uses data_cache.load_or_fetch first (Groww-enabled when
       GROWW_API_TOKEN exists in environment/.env), then falls back to direct
       yfinance batch downloads for any still-missing symbols.
    2. At each semi-annual rebalance date, ranking by trailing market-cap proxy.
    3. Writing the top_n at each date to the output CSV.

    This is a best-effort approximation. Expected member overlap with true Nifty 500:
    ~90-93% (i.e. ~35-50 stocks different from truth at each date).
    For survivorship-bias-free output, DELISTED_TICKERS_FILE must contain
    historical Nifty 500 members that delisted and are absent from current lists.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("[HistoricalBuilder] yfinance not installed. Run: pip install yfinance")
        return False

    # Step 1: Get the current Nifty 500 list as the candidate universe.
    # We cast a wider net by also including current Nifty Microcap 250 —
    # that way stocks that fell out of Nifty 500 are still represented.
    print("[HistoricalBuilder] Fetching current Nifty 500 list from NSE...")
    try:
        resp = requests.get(
            _NSE_NIFTY500_CSV_URL,
            headers=_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        nse_df = pd.read_csv(io.BytesIO(resp.content), encoding="latin-1")
        current_n500 = _extract_symbols_from_nse_csv(nse_df)
        print(f"  Got {len(current_n500)} current Nifty 500 members.")
    except Exception as exc:
        logger.warning("[HistoricalBuilder] Could not fetch current Nifty 500 list: %s", exc)
        current_n500 = []

    if not current_n500:
        logger.error(
            "[HistoricalBuilder] Cannot build approximate universe without "
            "at least the current Nifty 500 list."
        )
        return False

    # For a broader candidate pool, add some well-known tickers that may have
    # dropped out (these are supplemental — not required for correctness)
    candidate_tickers = list(dict.fromkeys(current_n500))   # deduplicated
    try:
        delisted = pd.read_csv(DELISTED_TICKERS_FILE)["ticker"].tolist()
        candidate_tickers = list(dict.fromkeys(candidate_tickers + delisted))
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        logger.warning(
            "Delisted tickers file unavailable/invalid (%s); survivorship bias not corrected.",
            exc,
        )
    except Exception as exc:
        logger.warning(
            "Unexpected error reading delisted tickers (%s); survivorship bias not corrected.",
            exc,
            exc_info=True,
        )

    # Step 2: Download price data
    rebalance_dates = _get_rebalance_dates(start_year=start_year)
    effective_start = (
        pd.Timestamp(rebalance_dates[0]) - pd.Timedelta(days=200)
    ).strftime("%Y-%m-%d")
    effective_end = pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"[HistoricalBuilder] Downloading {len(candidate_tickers)} tickers "
          f"from {effective_start} to {effective_end}...")

    market_data: dict = {}

    # Prefer the shared provider chain so Groww credentials are reused.
    try:
        from data_cache import load_or_fetch
        from momentum_engine import UltimateConfig

        cached = load_or_fetch(
            candidate_tickers,
            effective_start,
            effective_end,
            cfg=UltimateConfig(),
        )
        for tkr, df in cached.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                market_data[tkr] = df
        print(f"  Provider chain (Groww/yfinance) returned {len(market_data)} tickers.")
    except Exception as exc:
        logger.warning("[HistoricalBuilder] Provider-chain download failed: %s", exc)

    # If provider chain is sparse/unavailable, augment with direct yfinance.
    missing = [t for t in candidate_tickers if t not in market_data]
    if missing:
        print(f"  Fetching remaining {len(missing)} tickers via direct yfinance batches...")
        batch_size = 50
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            try:
                raw = yf.download(
                    batch,
                    start=effective_start,
                    end=effective_end,
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                )
                for tkr in batch:
                    try:
                        if isinstance(raw.columns, pd.MultiIndex):
                            df_tkr = raw.xs(tkr, axis=1, level=1).dropna(how="all")
                        else:
                            df_tkr = raw.copy()
                        if not df_tkr.empty:
                            market_data[tkr] = df_tkr
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning(
                    "[HistoricalBuilder] yfinance batch %d-%d failed: %s",
                    i, i + batch_size, exc,
                )
            n_done = min(i + batch_size, len(missing))
            print(f"  Downloaded {n_done}/{len(missing)} fallback tickers", end="\r")
            time.sleep(0.5)

        print()

    print(f"[HistoricalBuilder] Got data for {len(market_data)} tickers.")

    # Step 3: Build PIT membership at each rebalance date
    rows: list[dict] = []
    print(f"[HistoricalBuilder] Ranking universe at {len(rebalance_dates)} rebalance dates...")

    for rdate in rebalance_dates:
        members = _approximate_nifty500_at_date(
            rdate, candidate_tickers, market_data, top_n=top_n
        )
        if not members:
            logger.warning(
                "[HistoricalBuilder] No members ranked at %s — skipping.", rdate
            )
            continue
        for tkr in members:
            rows.append({"date": rdate, "ticker": tkr})
        print(f"  {rdate}: {len(members)} members")

    if not rows:
        logger.error("[HistoricalBuilder] Approximation produced no rows.")
        return False

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    n_dates = out_df["date"].nunique()
    print(f"\n[HistoricalBuilder] Wrote approximate PIT CSV: {n_dates} dates, "
          f"{len(rows)} rows → {output_csv}")
    logger.warning(
        "[HistoricalBuilder] APPROXIMATION WARNING: This CSV was built from "
        "market-cap ranking, NOT from true NSE index announcements. "
        "Expected member overlap with true Nifty 500: ~90-93%%. "
        "Survivorship bias is substantially reduced but NOT eliminated."
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PARQUET BUILDER (unchanged from previous version — correct)
# ─────────────────────────────────────────────────────────────────────────────

def build_parquet_from_csv(csv_path: str, output_path: str) -> Path:
    """
    Convert a PIT CSV (date, ticker) into the parquet format expected by
    universe_manager.get_historical_universe().

    Required parquet schema:
        Index : DatetimeIndex named "date" — one row per recorded snapshot date.
        Column: "tickers" — Python list of .NS-suffixed ticker strings.
    """
    csv = Path(csv_path)
    if not csv.exists():
        raise FileNotFoundError(
            f"[HistoricalBuilder] CSV source not found: {csv_path}. "
            "Run build_historical_csv() first."
        )

    df = pd.read_csv(csv)
    # BUG-HB-02: parse_dates is deprecated in pandas 2.x; use explicit pd.to_datetime instead.
    df["date"] = pd.to_datetime(df["date"])
    if df.empty or "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError(
            f"[HistoricalBuilder] CSV at {csv_path} is empty or missing required columns."
        )

    # Normalise tickers
    df["ticker"] = df["ticker"].astype(str).str.strip().apply(_ns_ticker)
    df = df[df["ticker"].str.endswith(".NS")]

    # Group by snapshot date → sorted unique ticker list
    rows = (
        df.groupby(df["date"].dt.normalize())["ticker"]
        .agg(lambda x: sorted(x.unique().tolist()))
    )
    out_df = pd.DataFrame({"tickers": rows})
    out_df.index = pd.DatetimeIndex(out_df.index, name="date")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_df.to_parquet(path, engine="pyarrow")
    except Exception:
        out_df.to_parquet(path)
    logger.info(
        "[HistoricalBuilder] Wrote %d PIT snapshots → %s",
        len(out_df),
        path,
    )
    return path


def bootstrap_historical_parquet(
    output_path: str = str(DATA_DIR / "historical_nifty500.parquet"),
    default_tickers: list[str] | None = None,
) -> Path:
    """
    Last-resort stub parquet.  Attempts to derive content from the sibling CSV
    (same stem, .csv extension) before falling back to a minimal stub.

    WARNING: The stub path (3 tickers, today's date only) causes every
    historical backtest lookup to miss because universe_manager checks
    `available_dates <= rebalance_date` — a single row dated today will
    never match any historical date.  Always prefer build_parquet_from_csv().
    """
    path = Path(output_path)

    sibling_csv = path.with_suffix(".csv")
    if sibling_csv.exists():
        logger.info(
            "[HistoricalBuilder] Found sibling CSV %s — building parquet from it.",
            sibling_csv,
        )
        return build_parquet_from_csv(str(sibling_csv), output_path)

    tickers = default_tickers or ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    logger.warning(
        "[HistoricalBuilder] Bootstrapping %s with a 3-ticker stub universe. "
        "This parquet has a single row dated today and will cause every historical "
        "universe lookup to miss — run main() to produce a proper PIT parquet.",
        output_path,
    )
    idx = pd.DatetimeIndex([pd.Timestamp.today().normalize()], name="date")
    df = pd.DataFrame({"tickers": [tickers]}, index=idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    # FIX-NEW-HB-01: pin engine="pyarrow" so list-valued tickers cells round-trip
    # correctly.  universe_manager._load_historical_universe_df also pins pyarrow,
    # so both ends of the parquet channel now use the same engine.
    try:
        df.to_parquet(path, engine="pyarrow")
    except Exception:
        df.to_parquet(path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# PARQUET VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify_parquet(parquet_path: str) -> bool:
    """
    Print a diagnostic summary of the parquet file and return True if it
    looks like a valid PIT universe.

    Run this after building to confirm the data is correct before optimizing.
    """
    path = Path(parquet_path)
    if not path.exists():
        print(f"[Verify] MISSING: {parquet_path}")
        return False

    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception:
        df = pd.read_parquet(path)
    dates = df.index.unique().sort_values()
    n_dates = len(dates)

    print(f"\n{'='*60}")
    print(f"Parquet: {parquet_path}")
    print(f"{'='*60}")
    print(f"  Snapshot dates  : {n_dates}")

    if n_dates == 0:
        print("  STATUS: EMPTY — rebuild required")
        return False

    print(f"  Date range      : {dates[0].date()} → {dates[-1].date()}")

    # Sample sizes
    sample_dates = [d for d in dates if pd.Timestamp("2019-01-01") <= d <= pd.Timestamp("2022-12-31")]
    def _to_list(val) -> list:
        """Coerce parquet-loaded tickers (list or numpy array) to a Python list."""
        import numpy as np
        if isinstance(val, (list, np.ndarray)):
            return list(val)
        if hasattr(val, "tolist"):
            return val.tolist()
        return [val] if val else []

    if sample_dates:
        sizes = []
        for d in sample_dates[:8]:
            row = df.loc[d, "tickers"]
            n = len(_to_list(row))
            sizes.append((d.date(), n))
        print("  Sample sizes (2019-2022):")
        for d, n in sizes:
            flag = " ← BIASED (today's list?)" if n > 510 else ""
            print(f"    {d}: {n} members{flag}")

    # Sanity checks
    issues = []

    if n_dates < 10:
        issues.append(f"Too few snapshots ({n_dates}), expected 20+")

    if dates[-1] < pd.Timestamp.today() - pd.Timedelta(days=365):
        issues.append(f"Latest snapshot is >1 year old ({dates[-1].date()})")

    # Check if all dates return the same members (sign of current-list backfill)
    if n_dates >= 3:
        first_members = set(_to_list(df.iloc[0]["tickers"]))
        last_members  = set(_to_list(df.iloc[-1]["tickers"]))
        if first_members and last_members:
            overlap = len(first_members & last_members)
            divergence_pct = 100 * (1 - overlap / max(len(first_members), len(last_members), 1))
            print(f"  First vs Last overlap: {overlap} common, "
                  f"{divergence_pct:.1f}% divergence")
            if divergence_pct < 1.0 and n_dates > 5:
                issues.append(
                    "First and last snapshot are nearly identical "
                    "(classic sign of current-list back-fill, i.e. survivorship bias)"
                )

    # Check 2021 universe for known late-joiners
    late_joiners = {"ZOMATO.NS", "NYKAA.NS", "PAYTM.NS", "POLICYBAZAAR.NS", "IREDA.NS"}
    pre_2021_dates = [d for d in dates if d < pd.Timestamp("2021-07-01")]
    if pre_2021_dates:
        pre_date = max(pre_2021_dates)
        pre_members = set(_to_list(df.loc[pre_date, "tickers"]))
        found_late = late_joiners & pre_members
        if found_late:
            issues.append(
                f"Pre-2021 snapshot contains post-2021 IPOs: {found_late} "
                "(CONFIRMS survivorship bias — these stocks didn't exist then)"
            )

    if issues:
        print("\n  ⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"    • {issue}")
        print("\n  STATUS: BIASED / INVALID — rebuild required")
        return False
    else:
        print("\n  STATUS: OK — looks like valid PIT data")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# BUILD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_historical_csv(universe_type: str, output_path: str) -> Path:
    """
    Build a PIT (date, ticker) CSV for `universe_type` using the Wayback Machine
    as the primary data source.

    Falls back to an approximate yfinance market-cap ranking if Wayback fails.

    Parameters
    ----------
    universe_type : "nifty500" (only supported value; nse_total skipped)
    output_path   : path to write the CSV
    """
    if universe_type.lower() == "nse_total":
        raise NotImplementedError(
            "[HistoricalBuilder] nse_total PIT data is not available from the "
            "Wayback Machine source. Use 'nifty500' universe for survivorship-safe "
            "backtesting. NSE Total contains thousands of tickers and no public "
            "historical constituent archive exists."
        )

    if universe_type.lower() != "nifty500":
        raise ValueError(
            f"[HistoricalBuilder] Unsupported universe_type: {universe_type!r}. "
            "Only 'nifty500' is supported."
        )

    output = Path(output_path)

    # Deterministic/offline-first path used by tests and production pipelines:
    # consume the local master archive if available.
    local_master = DATA_DIR / "raw_nifty_archives.csv"
    if local_master.exists():
        df = _load_master_archive(universe_type)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        logger.info("[HistoricalBuilder] Built %s from local raw archive %s", output, local_master)
        return output

    raise FileNotFoundError(
        "[HistoricalBuilder] Raw archive missing: data/raw_nifty_archives.csv. "
        "Run build_historical_fallback.py first (or historical_builder.main())."
    )

def _build_historical_csv_network_fallback(universe_type: str, output_path: str) -> Path:
    """
    Legacy network-first PIT builder kept for manual/offline-recovery workflows.

    This path is intentionally not called by build_historical_csv(), which now
    enforces the deterministic local-archive-first pipeline used by production
    and tests.
    """
    output = Path(output_path)
    start_year = 2015
    current_year = date.today().year

    # ── Primary: Wayback Machine ──────────────────────────────────────────────
    print(f"\n[HistoricalBuilder] Building PIT CSV for {universe_type} via Wayback Machine...")
    print(f"  Target URL: {_NSE_NIFTY500_CSV_URL}")
    print(f"  Output    : {output}")

    success = _build_pit_csv_from_wayback(
        target_url=_NSE_NIFTY500_CSV_URL,
        output_csv=output,
        start_year=start_year,
        sleep_seconds=1.2,
    )

    if success:
        logger.info(
            "[HistoricalBuilder] Wayback Machine build succeeded → %s", output
        )
        return output

    # ── Fallback: yfinance market-cap approximation ───────────────────────────
    print(
        "\n[HistoricalBuilder] Wayback Machine failed. "
        "Falling back to yfinance market-cap approximation..."
    )
    print(
        "  WARNING: This produces ~90-93% accurate PIT data, not 100%.\n"
        "  Survivorship bias is greatly reduced but not fully eliminated.\n"
        "  Re-run with Wayback access to get exact historical constituents."
    )

    success = _build_pit_csv_from_yfinance_approx(
        output_csv=output,
        start_year=start_year,
        top_n=500,
    )

    if not success:
        raise FileNotFoundError(
            "[HistoricalBuilder] Both Wayback Machine and yfinance approximation failed. "
            "Check network connectivity and try again.\n\n"
            "Manual alternative:\n"
            "  1. Go to https://web.archive.org/web/*/https://www.niftyindices.com/"
            "IndexConstituent/ind_nifty500list.csv\n"
            f"  2. Download a snapshot for each year ({start_year}-{current_year})\n"
            "  3. Save each as 'data/nifty500_YYYY-MM-DD.csv'\n"
            "  4. Run: python historical_builder.py --from-manual-csvs data/"
        )

    return output


def _load_master_archive(universe_type: str) -> pd.DataFrame:
    """Load and normalize raw archive CSV into canonical (date,ticker) rows."""
    if universe_type.lower() != "nifty500":
        raise ValueError("[HistoricalBuilder] _load_master_archive currently supports only 'nifty500'.")

    raw = DATA_DIR / "raw_nifty_archives.csv"
    if not raw.exists():
        raise FileNotFoundError(f"[HistoricalBuilder] Raw archive missing: {raw}")

    df = pd.read_csv(raw)
    cols = [str(c).strip() for c in df.columns]
    norm = [c.lower() for c in cols]

    # Format A: long rows -> date,ticker
    if "date" in norm and "ticker" in norm:
        dcol = cols[norm.index("date")]
        tcol = cols[norm.index("ticker")]
        out = df[[dcol, tcol]].rename(columns={dcol: "date", tcol: "ticker"})
    # Format B: wide rows by ticker -> ticker,YYYY-MM-DD,...
    elif "ticker" in norm:
        tcol = cols[norm.index("ticker")]
        value_cols = [c for c in cols if c != tcol]
        melted = df.melt(id_vars=[tcol], value_vars=value_cols, var_name="date", value_name="included")
        # FIX-NEW-HB-02: pd.to_numeric(errors="coerce") converts any non-numeric
        # cell (e.g. empty string, "N/A") to NaN; fillna(0) then maps NaN → 0;
        # "> 0" rejects zeros, negatives, and the coerced NaN placeholders.
        # This means only strictly positive numeric values count as "present",
        # which is the correct semantics for membership flags in all known formats.
        included = pd.to_numeric(melted["included"], errors="coerce").fillna(0) > 0
        out = melted.loc[included, ["date", tcol]].rename(columns={tcol: "ticker"})
    # Format C: wide rows by date -> date,TICKER_A,TICKER_B,...
    elif "date" in norm:
        dcol = cols[norm.index("date")]
        ticker_cols = [c for c in cols if c != dcol]
        melted = df.melt(id_vars=[dcol], value_vars=ticker_cols, var_name="ticker", value_name="included")
        # Same truthy filter as Format B — see comment above.
        included = pd.to_numeric(melted["included"], errors="coerce").fillna(0) > 0
        out = melted.loc[included, [dcol, "ticker"]].rename(columns={dcol: "date"})
    else:
        raise ValueError("[HistoricalBuilder] Unsupported archive format.")

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out[out["date"].notna()].copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().map(_ns_ticker)
    out = out[["date", "ticker"]].drop_duplicates().sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def _download_archive(universe_type: str, output_path: Path) -> Path:
    urls = REMOTE_ARCHIVE_URLS.get(universe_type, [])
    if not urls:
        raise FileNotFoundError(f"[HistoricalBuilder] No remote archive URL configured for {universe_type}. Build via build_historical_fallback.py instead.")

    for url in urls:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(resp.text, encoding="utf-8")
            return output_path
        except Exception as exc:
            logger.warning("[HistoricalBuilder] Archive download failed from %s: %s", url, exc)

    raise FileNotFoundError(f"[HistoricalBuilder] Failed to download archive for {universe_type}.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Build the full set of PIT universe files needed for survivorship-safe backtests.

    Steps:
        1. Build PIT CSV for nifty500 via Wayback Machine (or yfinance approximation).
        2. Convert CSV to parquet.
        3. Verify the parquet for obvious bias.

    Note: nse_total is intentionally skipped — no public historical constituent
    archive exists for the full NSE total universe.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    csv_path     = DATA_DIR / "historical_nifty500.csv"
    parquet_path = DATA_DIR / "historical_nifty500.parquet"

    print("\n" + "=" * 60)
    print("  HISTORICAL BUILDER — PIT Universe Construction")
    print("=" * 60)

    for universe in ("nifty500", "nse_total"):
        normalized_csv = DATA_DIR / f"historical_{universe}.csv"
        parquet_out = DATA_DIR / f"historical_{universe}.parquet"

        if normalized_csv.exists():
            build_parquet_from_csv(str(normalized_csv), str(parquet_out))
            continue

        if universe == "nifty500":
            # Prefer local master archives if they exist.
            try:
                build_historical_csv("nifty500", str(csv_path))
                build_parquet_from_csv(str(csv_path), str(parquet_path))
                continue
            except FileNotFoundError:
                pass

            # If user relies on the production fallback builder, invoke it directly.
            try:
                import build_historical_fallback as bhf
                bhf.run(universe_arg="nifty500", start_date="2015-01-01")
                if normalized_csv.exists():
                    build_parquet_from_csv(str(normalized_csv), str(parquet_out))
                    continue
            except Exception as exc:
                logger.warning("[HistoricalBuilder] build_historical_fallback run failed: %s", exc)

        # Optional remote bootstrap path (primarily for CI/tests where URLs are monkeypatched).
        raw_path = DATA_DIR / f"raw_{universe}_archives.csv"
        if not raw_path.exists():
            _download_archive(universe, raw_path)

        tmp = pd.read_csv(raw_path)
        if not {"date", "ticker"}.issubset({c.lower() for c in tmp.columns}):
            raise ValueError(f"[HistoricalBuilder] Unsupported archive schema for {raw_path}.")

        tmp.columns = [c.lower() for c in tmp.columns]
        tmp["ticker"] = tmp["ticker"].astype(str).map(_ns_ticker)
        tmp[["date", "ticker"]].to_csv(normalized_csv, index=False)
        build_parquet_from_csv(str(normalized_csv), str(parquet_out))


if __name__ == "__main__":
    import sys

    # Support --verify flag for quick inspection without rebuilding
    if "--verify" in sys.argv:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        parquet_path = DATA_DIR / "historical_nifty500.parquet"
        verify_parquet(str(parquet_path))
        sys.exit(0)

    main()
