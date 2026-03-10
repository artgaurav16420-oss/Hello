"""
historical_builder.py
=====================
Generate point-in-time (PIT) universe snapshots for survivorship-safe backtests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# All paths in this module are rooted here.  Change DATA_DIR to relocate
# the entire historical-data directory without touching individual functions.
DATA_DIR = Path("data")


def _ns_ticker(sym: str) -> str:
    s = str(sym).strip().upper()
    if not s:
        return ""
    if s.startswith("^"):
        return s
    return s if s.endswith(".NS") else f"{s}.NS"


def _load_master_archive(universe_type: str) -> pd.DataFrame:
    """
    Load local archives if available.

    Supported paths (first match wins):
    - data/raw_nifty_archives.csv
    - data/raw_{universe_type}_archives.csv
    """
    candidates = [
        DATA_DIR / "raw_nifty_archives.csv",
        DATA_DIR / f"raw_{universe_type}_archives.csv",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        return pd.DataFrame(columns=["date", "ticker"])

    df = pd.read_csv(src)
    cols = {c.lower().strip(): c for c in df.columns}

    # long-form preferred: date,ticker
    if "date" in cols and "ticker" in cols:
        out = df[[cols["date"], cols["ticker"]]].rename(columns={cols["date"]: "date", cols["ticker"]: "ticker"})
    elif "snapshot_date" in cols and "symbol" in cols:
        out = df[[cols["snapshot_date"], cols["symbol"]]].rename(columns={cols["snapshot_date"]: "date", cols["symbol"]: "ticker"})
    else:
        # Wide format fallback, supporting both common layouts:
        # 1) first column=date, remaining columns=ticker buckets
        # 2) first column=ticker, remaining columns=date buckets
        first_col = df.columns[0]
        first_col_str = df[first_col].astype(str).str.strip()
        first_col_date_like = first_col_str.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$").mean() >= 0.7
        other_cols_are_dates = pd.Index(df.columns[1:]).astype(str).str.match(
            r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$"
        ).mean() >= 0.7
        first_col_is_date = first_col_date_like

        if first_col_is_date or not other_cols_are_dates:
            melted = df.melt(id_vars=[first_col], value_name="ticker").drop(columns=["variable"])
            out = melted.rename(columns={first_col: "date"})
        else:
            melted = df.melt(id_vars=[first_col], var_name="date", value_name="member")
            member = melted["member"].astype(str).str.strip().str.lower()
            valid_member = (
                melted["member"].notna()
                & member.ne("")
                & ~member.isin({"0", "false", "no", "n", "nan"})
            )
            out = melted.loc[valid_member, ["date", first_col]].rename(columns={first_col: "ticker"})

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["ticker"] = out["ticker"].map(_ns_ticker)
    out = out.dropna(subset=["date", "ticker"])
    out = out[out["ticker"] != ""]
    return out[["date", "ticker"]].drop_duplicates().sort_values(["date", "ticker"])


def build_historical_csv(universe_type: str, output_path: str) -> Path:
    """
    Build a PIT CSV containing exact recorded PIT snapshots.

    Output columns:
    - date (YYYY-MM-DD)
    - ticker (with .NS suffix)
    """
    raw = _load_master_archive(universe_type)

    if raw.empty:
        raise FileNotFoundError(
            "[HistoricalBuilder] Raw archive missing. Cannot build PIT universe without "
            f"historical source data for {universe_type}."
        )

    raw_dt = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.assign(_date=raw_dt.dt.normalize()).dropna(subset=["_date"])

    out_rows: list[dict[str, str]] = []
    for snapshot_date, group in raw.groupby("_date"):
        members = sorted(set(group["ticker"]))
        for t in members:
            out_rows.append({"date": snapshot_date.strftime("%Y-%m-%d"), "ticker": _ns_ticker(t)})

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise ValueError(
            f"[HistoricalBuilder] Raw archive for {universe_type} did not contain any usable date/ticker rows."
        )

    out = out.dropna(subset=["date", "ticker"])
    out["ticker"] = out["ticker"].map(_ns_ticker)
    out = out[out["ticker"].str.endswith(".NS") | out["ticker"].str.startswith("^")]
    out = out.drop_duplicates().sort_values(["date", "ticker"]).reset_index(drop=True)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def build_parquet_from_csv(csv_path: str, output_path: str) -> Path:
    """
    Convert a PIT CSV (date, ticker) produced by build_historical_csv into the
    parquet format expected by universe_manager.get_historical_universe().

    Required parquet schema:
        Index : DatetimeIndex named "date" — one row per recorded snapshot date.
        Column: "tickers" — Python list of .NS-suffixed ticker strings.

    This is the canonical way to produce the parquet files.  Never call
    bootstrap_historical_parquet() to create production parquets; always call
    this function after build_historical_csv() has run successfully.
    """
    csv = Path(csv_path)
    if not csv.exists():
        raise FileNotFoundError(
            f"[HistoricalBuilder] CSV source not found: {csv_path}. "
            "Run build_historical_csv() first."
        )

    df = pd.read_csv(csv, parse_dates=["date"])
    if df.empty or "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError(
            f"[HistoricalBuilder] CSV at {csv_path} is empty or missing required columns."
        )

    # Group by snapshot date → list of tickers (already .NS-suffixed from build_historical_csv)
    rows = (
        df.groupby(df["date"].dt.normalize())["ticker"]
        .apply(list)
    )
    out_df = pd.DataFrame({"tickers": rows})
    out_df.index = pd.DatetimeIndex(out_df.index, name="date")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
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

    # Derive the matching CSV path from the parquet path stem.
    # e.g. data/historical_nifty500.parquet → data/historical_nifty500.csv
    sibling_csv = path.with_suffix(".csv")
    if sibling_csv.exists():
        logger.info(
            "[HistoricalBuilder] Found sibling CSV %s — building parquet from it "
            "instead of creating a stub.",
            sibling_csv,
        )
        return build_parquet_from_csv(str(sibling_csv), output_path)

    # True fallback: stub with today's date only.  This is only useful for
    # smoke-testing imports; it will produce survivorship bias in backtests.
    tickers = default_tickers or ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    logger.warning(
        "[HistoricalBuilder] Bootstrapping %s with a 3-ticker stub universe only (%s). "
        "This parquet has a single row dated today and will cause every historical "
        "universe lookup to miss — run build_historical_csv() then "
        "build_parquet_from_csv() to produce a proper PIT parquet.",
        output_path,
        ", ".join(tickers),
    )
    idx = pd.DatetimeIndex([pd.Timestamp.today().normalize()], name="date")
    df = pd.DataFrame({"tickers": [tickers]}, index=idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def main() -> None:
    """
    Build the full set of PIT universe files needed for survivorship-safe backtests.

    Order matters:
      1. Build CSVs first  (recorded PIT snapshots from source archives)
      2. Convert CSVs to parquets  (schema required by universe_manager)

    Never call bootstrap_historical_parquet() here — it produces a single-row
    stub dated today which makes every historical lookup miss.
    """
    print("[HistoricalBuilder] Step 1/2 — Building PIT CSV snapshots...")
    csv_nifty   = build_historical_csv("nifty500",  str(DATA_DIR / "historical_nifty500.csv"))
    csv_total   = build_historical_csv("nse_total", str(DATA_DIR / "historical_nse_total.csv"))
    print(f"  [+] {csv_nifty}")
    print(f"  [+] {csv_total}")

    print("[HistoricalBuilder] Step 2/2 — Converting CSVs to parquet...")
    pq_nifty  = build_parquet_from_csv(
        str(DATA_DIR / "historical_nifty500.csv"),  str(DATA_DIR / "historical_nifty500.parquet")
    )
    pq_total  = build_parquet_from_csv(
        str(DATA_DIR / "historical_nse_total.csv"), str(DATA_DIR / "historical_nse_total.parquet")
    )
    print(f"  [+] {pq_nifty}")
    print(f"  [+] {pq_total}")

    print("\n[HistoricalBuilder] All PIT files ready. Backtests will now use "
          "correct point-in-time constituents without survivorship bias.")


if __name__ == "__main__":
    main()
