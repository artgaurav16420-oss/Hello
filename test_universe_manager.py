import logging
import pandas as pd
import universe_manager as um


def test_get_historical_universe_uses_csv_without_survivorship_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "historical_nifty500.csv").write_text(
        "date,ticker\n2020-01-01,RELIANCE.NS\n2020-01-01,INFY.NS\n",
        encoding="utf-8",
    )

    um._MISSING_PARQUET_WARNED.clear()
    um._NO_RECORD_WARNED.clear()

    caplog.set_level(logging.WARNING)
    members = um.get_historical_universe("nifty500", pd.Timestamp("2020-02-01"))

    assert members == ["INFY.NS", "RELIANCE.NS"]
    assert "survivorship bias" not in caplog.text.lower()


def test_get_historical_universe_warns_when_no_parquet_or_csv(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    um._MISSING_PARQUET_WARNED.clear()
    um._NO_RECORD_WARNED.clear()

    caplog.set_level(logging.WARNING)
    members = um.get_historical_universe("nifty500", pd.Timestamp("2020-02-01"))

    assert members == []
    assert "survivorship bias risk" in caplog.text.lower()
