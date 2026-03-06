"""
historical_builder.py
=====================
Bootstrap utility to initialize a historical universe parquet so first-time
users can run backtests without hitting missing-file errors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def bootstrap_historical_parquet(
    output_path: str = "data/historical_nifty500.parquet",
    default_tickers: list[str] | None = None,
) -> Path:
    """Create a minimal historical parquet with DatetimeIndex + tickers list column."""
    tickers = default_tickers or ["RELIANCE", "TCS", "HDFCBANK"]
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
    ]
    for path in outputs:
        print(f"[+] Bootstrap complete: {path}")


if __name__ == "__main__":
    main()
