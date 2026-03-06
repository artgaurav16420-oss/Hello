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
    tickers: list[str] | None = None,
) -> Path:
    """Create a minimal historical parquet with DatetimeIndex + tickers list column."""
    default_tickers = tickers or ["RELIANCE", "TCS", "HDFCBANK"]
    idx = pd.DatetimeIndex([pd.Timestamp.today().normalize()], name="date")
    df = pd.DataFrame({"tickers": [default_tickers]}, index=idx)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def main() -> None:
    path = bootstrap_historical_parquet()
    print(f"[+] Bootstrap complete: {path}")


if __name__ == "__main__":
    main()
