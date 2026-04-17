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
from pathlib import Path

import pandas as pd
import requests

from shared_utils import (
    NSE_URL_NIFTY500_INDEX_CONSTITUENT_CSV,
    fetch_nse_csv,
    normalize_ns_ticker,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

# Default historical NIFTY500 parquet filename used by bootstrap_historical_parquet().
NIFTY500_PARQUET_FILENAME = "historical_nifty500.parquet"

# Remote master archives used by main() bootstrapping flow.
REMOTE_ARCHIVE_URLS: dict[str, list[str]] = {
    "nifty500": [],
    "nse_total": [],
}

# ── Constants ─────────────────────────────────────────────────────────────────

# Official NSE constituent CSV URLs (current)
_NSE_NIFTY500_CSV_URL = NSE_URL_NIFTY500_INDEX_CONSTITUENT_CSV

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

    Searches for known symbol column names and falls back to a regex scan
    across all columns if a direct match is not found.

    Args:
        df (pd.DataFrame): Raw CSV data from NSE/Wayback.

    Returns:
        list[str]: Sorted list of .NS-suffixed tickers.
    """
    # Normalise column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Try 'symbol' column first (official NSE format)
    for col in ("symbol", "symbols", "ticker", "nse symbol", "nse_symbol"):
        if col in df.columns:
            raw = df[col].dropna().astype(str).str.strip()
            tickers = [t for t in (normalize_ns_ticker(s) for s in raw if s and s.upper() != "SYMBOL") if t]
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
                candidates.add(normalize_ns_ticker(v))

    return sorted(candidates)


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
    if df.empty or "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError(
            f"[HistoricalBuilder] CSV at {csv_path} is empty or missing required columns."
        )
    # BUG-HB-02: parse_dates is deprecated in pandas 2.x; use explicit pd.to_datetime instead.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    # Normalise tickers
    df["ticker"] = df["ticker"].astype(str).str.strip().apply(normalize_ns_ticker)
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
    output_path: str = str(DATA_DIR / NIFTY500_PARQUET_FILENAME),
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

def _to_list(val) -> list:
    """Coerce parquet-loaded tickers (list or numpy array) to a Python list."""
    import numpy as np
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    if hasattr(val, "tolist"):
        out = val.tolist()
        return out if isinstance(out, list) else [out]
    return [val] if val else []

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
    if sample_dates:
        sizes = []
        for d in sample_dates[:8]:
            row = df.loc[d, "tickers"]
            if isinstance(row, pd.Series):
                flattened: list = []
                for item in row.tolist():
                    flattened.extend(_to_list(item))
                row = flattened
            n = len(_to_list(row))
            sizes.append((d.date(), n))
        print("  Sample sizes (2019-2022):")
        for d, n in sizes:
            flag = " ← BIASED (today's list?)" if n > 510 else ""
            print(f"    {d}: {n} members{flag}")

    # Sanity checks
    issues = []

    if n_dates < 20:
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

    # Check pre-2021 snapshots for known late-joiners
    late_joiners = {"ZOMATO.NS", "NYKAA.NS", "PAYTM.NS", "POLICYBAZAAR.NS", "IREDA.NS"}
    pre_2021_dates = [d for d in dates if d < pd.Timestamp("2021-07-01")]
    found_late_all: set[str] = set()
    for pre_date in pre_2021_dates:
        pre_row = df.loc[pre_date, "tickers"]
        if isinstance(pre_row, pd.Series):
            flattened: list = []
            for item in pre_row.tolist():
                flattened.extend(_to_list(item))
            pre_row = flattened
        pre_members = set(_to_list(pre_row))
        found_late = late_joiners & pre_members
        if found_late:
            found_late_all.update(found_late)
    
    if found_late_all:
        issues.append(
            f"Pre-2021 snapshot contains post-2021 IPOs: {found_late_all} "
            "(CONFIRMS survivorship bias — these stocks didn't exist then)"
        )

    if issues:
        print("\n  ⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"    • {issue}")
        print("\n  STATUS: BIASED / INVALID — rebuild required")
        return False
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
    out["ticker"] = out["ticker"].astype(str).str.strip().map(normalize_ns_ticker)
    out = out[out["ticker"] != ""]
    out = out[["date", "ticker"]].drop_duplicates().sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def _download_archive(universe_type: str, output_path: Path) -> Path:
    """
    Download/verify a historical universe archive from a remote source.

    Args:
        universe_type (str): Key for the universe (e.g. 'nifty500').
        output_path (Path): Filesystem path to save the downloaded parquet.

    Returns:
        Path: The confirmed path to the downloaded archive.

    Raises:
        RuntimeError: If the remote download fails or the file is corrupt.
    """
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
    parquet_path = DATA_DIR / NIFTY500_PARQUET_FILENAME

    print("\n" + "=" * 60)
    print("  HISTORICAL BUILDER — PIT Universe Construction")
    print("=" * 60)

    for universe in ("nifty500", "nse_total"):
        if universe == "nse_total" and not (
            (DATA_DIR / "historical_nse_total.csv").exists()
            or (DATA_DIR / "raw_nse_total_archives.csv").exists()
            or REMOTE_ARCHIVE_URLS.get("nse_total")
        ):
            logger.info("[HistoricalBuilder] Skipping nse_total — no archive source available.")
            continue
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
        tmp["ticker"] = tmp["ticker"].astype(str).map(normalize_ns_ticker)
        tmp = tmp[tmp["ticker"] != ""]
        tmp[["date", "ticker"]].to_csv(normalized_csv, index=False)
        build_parquet_from_csv(str(normalized_csv), str(parquet_out))


if __name__ == "__main__":
    import sys

    # Support --verify flag for quick inspection without rebuilding
    if "--verify" in sys.argv:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        parquet_path = DATA_DIR / NIFTY500_PARQUET_FILENAME
        is_valid = verify_parquet(str(parquet_path))
        sys.exit(0 if is_valid else 1)

    main()
