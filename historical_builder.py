"""
historical_builder.py
=====================
Generate point-in-time (PIT) universe snapshots for survivorship-safe backtests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from universe_manager import get_nifty500, fetch_nse_equity_universe

logger = logging.getLogger(__name__)


def _monthly_dates(start: str = "2018-01-01", end: str | None = None) -> pd.DatetimeIndex:
    end_ts = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)
    return pd.date_range(pd.Timestamp(start), end_ts, freq="MS")


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
        Path("data/raw_nifty_archives.csv"),
        Path(f"data/raw_{universe_type}_archives.csv"),
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
        # Wide format fallback: first column is date, remaining cols are ticker buckets.
        first_col = df.columns[0]
        melted = df.melt(id_vars=[first_col], value_name="ticker").drop(columns=["variable"])
        out = melted.rename(columns={first_col: "date"})

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["ticker"] = out["ticker"].map(_ns_ticker)
    out = out.dropna(subset=["date", "ticker"])
    out = out[out["ticker"] != ""]
    return out[["date", "ticker"]].drop_duplicates().sort_values(["date", "ticker"])


def _fallback_members(universe_type: str) -> list[str]:
    if universe_type == "nse_total":
        base = fetch_nse_equity_universe()
    else:
        base = get_nifty500()
    return sorted({_ns_ticker(x) for x in base if _ns_ticker(x)})


def build_historical_csv(universe_type: str, output_path: str) -> Path:
    """
    Build a PIT CSV containing monthly snapshots from 2018 to present.

    Output columns:
    - date (YYYY-MM-DD)
    - ticker (with .NS suffix)
    """
    month_grid = _monthly_dates()
    raw = _load_master_archive(universe_type)

    if raw.empty:
        logger.warning(
            "[HistoricalBuilder] No local archive found for %s; using stable fallback constituents for monthly grid.",
            universe_type,
        )
        fallback = _fallback_members(universe_type)
        rows = [{"date": d.strftime("%Y-%m-%d"), "ticker": t} for d in month_grid for t in fallback]
        out = pd.DataFrame(rows)
    else:
        raw_dt = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.assign(_date=raw_dt.dt.normalize()).dropna(subset=["_date"])

        # Snap each monthly point to the latest available historical snapshot at or before that month.
        out_rows: list[dict[str, str]] = []
        for d in month_grid:
            eligible = raw[raw["_date"] <= d]
            if eligible.empty:
                continue
            anchor = eligible["_date"].max()
            members = sorted(set(eligible.loc[eligible["_date"] == anchor, "ticker"]))
            for t in members:
                out_rows.append({"date": d.strftime("%Y-%m-%d"), "ticker": _ns_ticker(t)})

        out = pd.DataFrame(out_rows)

        if out.empty:
            fallback = _fallback_members(universe_type)
            rows = [{"date": d.strftime("%Y-%m-%d"), "ticker": t} for d in month_grid for t in fallback]
            out = pd.DataFrame(rows)

    out = out.dropna(subset=["date", "ticker"])
    out["ticker"] = out["ticker"].map(_ns_ticker)
    out = out[out["ticker"].str.endswith(".NS") | out["ticker"].str.startswith("^")]
    out = out.drop_duplicates().sort_values(["date", "ticker"]).reset_index(drop=True)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def bootstrap_historical_parquet(
    output_path: str = "data/historical_nifty500.parquet",
    default_tickers: list[str] | None = None,
) -> Path:
    """Create a minimal historical parquet with DatetimeIndex + tickers list column."""
    tickers = default_tickers or ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    idx = pd.DatetimeIndex([pd.Timestamp.today().normalize()], name="date")
    df = pd.DataFrame({"tickers": [tickers]}, index=idx)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def main() -> None:
    outputs = [
        bootstrap_historical_parquet("data/historical_nifty500.parquet"),
        bootstrap_historical_parquet("data/historical_nse_total.parquet"),
        build_historical_csv("nifty500", "data/historical_nifty500.csv"),
        build_historical_csv("nse_total", "data/historical_nse_total.csv"),
    ]
    for path in outputs:
        print(f"[+] Bootstrap complete: {path}")


if __name__ == "__main__":
    main()
