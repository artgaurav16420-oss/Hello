from __future__ import annotations

import pandas as pd

import historical_builder as hb


def test_build_historical_csv_from_local_master(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw = data_dir / "raw_nifty_archives.csv"
    raw.write_text(
        "date,ticker\n2018-01-01,RELIANCE\n2018-01-01,TCS\n2018-02-01,INFY\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    out_path = hb.build_historical_csv("nifty500", "data/historical_nifty500.csv")
    out = pd.read_csv(out_path)

    assert set(out.columns) == {"date", "ticker"}
    assert out["date"].min() == "2018-01-01"
    assert out["ticker"].str.endswith(".NS").all()


def test_build_historical_csv_uses_fallback_when_master_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hb, "get_nifty500", lambda: ["RELIANCE", "TCS"])
    monkeypatch.setattr(hb, "fetch_nse_equity_universe", lambda: ["SBIN", "INFY"])

    out_path = hb.build_historical_csv("nifty500", "data/historical_nifty500.csv")
    out = pd.read_csv(out_path)

    assert not out.empty
    assert out["date"].nunique() > 12
    assert out["ticker"].str.endswith(".NS").all()
