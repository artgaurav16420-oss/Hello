from __future__ import annotations

import os

import build_historical_fallback as bhf


def test_load_env_file_fallback_sets_missing_keys_only(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GROWW_API_TOKEN=token_from_file\nEXISTING_KEY=from_file\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("GROWW_API_TOKEN", raising=False)
    monkeypatch.setenv("EXISTING_KEY", "already_set")

    bhf._load_env_file_fallback(env_file)

    assert os.getenv("GROWW_API_TOKEN") == "token_from_file"
    assert os.getenv("EXISTING_KEY") == "already_set"


import pandas as pd


def test_build_adnv_ranked_snapshots_prefers_notional_liquidity(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=90, freq="D")
    market_data = {
        "HIGHPRICE_LOWVOL.NS": pd.DataFrame({
            "Close": [1000.0] * len(idx),
            "Volume": [1.0] * len(idx),
        }, index=idx),
        "LOWPRICE_HIGHVOL.NS": pd.DataFrame({
            "Close": [100.0] * len(idx),
            "Volume": [100.0] * len(idx),
        }, index=idx),
    }

    out = bhf._build_adnv_ranked_snapshots(
        market_data=market_data,
        start_date="2024-03-31",
        top_n=2,
        lookback_days=60,
        min_trading_days=30,
    )

    assert not out.empty
    first = out.iloc[0]["tickers"]
    assert first[0] == "LOWPRICE_HIGHVOL.NS"


def test_write_snapshot_outputs_writes_parquet_and_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(bhf, "DATA_DIR", tmp_path / "data")
    idx = pd.DatetimeIndex(["2024-03-31"], name="date")
    snapshots = pd.DataFrame({"tickers": [["AAA.NS", "BBB.NS"]]}, index=idx)

    out = bhf._write_snapshot_outputs("nifty500", snapshots)

    assert out.exists()
    csv_path = tmp_path / "data" / "historical_nifty500.csv"
    assert csv_path.exists()
    csv = pd.read_csv(csv_path)
    assert set(csv["ticker"]) == {"AAA.NS", "BBB.NS"}
