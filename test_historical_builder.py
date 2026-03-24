from __future__ import annotations

import logging
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
    assert out["date"].nunique() == 2
    assert out["ticker"].str.endswith(".NS").all()


def test_build_historical_csv_raises_when_master_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    try:
        hb.build_historical_csv("nifty500", "data/historical_nifty500.csv")
        assert False, "Expected FileNotFoundError when raw archive is absent"
    except FileNotFoundError as exc:
        assert "Raw archive missing" in str(exc)


def test_load_master_archive_supports_wide_ticker_rows(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw = data_dir / "raw_nifty_archives.csv"
    raw.write_text(
        "ticker,2020-01-31,2020-02-28\nRELIANCE,1,1\nTCS,0,1\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    out = hb._load_master_archive("nifty500")

    assert set(out.columns) == {"date", "ticker"}
    assert set(out["date"]) == {"2020-01-31", "2020-02-28"}
    assert set(out["ticker"]) == {"RELIANCE.NS", "TCS.NS"}


def test_bootstrap_historical_parquet_warns_stub_content(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.WARNING)

    out = hb.bootstrap_historical_parquet("data/historical_nifty500.parquet")

    assert out.exists()
    assert "3-ticker stub universe" in caplog.text


def test_load_master_archive_supports_wide_date_rows_truthy_filter(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw = data_dir / "raw_nifty_archives.csv"
    raw.write_text(
        "date,RELIANCE,TCS\n2020-01-31,1,0\n2020-02-28,1,1\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    out = hb._load_master_archive("nifty500")

    assert set(out.columns) == {"date", "ticker"}
    assert set(out["date"]) == {"2020-01-31", "2020-02-28"}
    jan = out[out["date"] == "2020-01-31"]["ticker"].tolist()
    assert jan == ["RELIANCE.NS"]


def test_main_downloads_archives_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    sample_by_universe = {
        "nifty500": "date,ticker\n2020-01-31,RELIANCE\n2020-01-31,TCS\n",
        "nse_total": "date,ticker\n2020-01-31,INFY\n2020-01-31,SBIN\n",
    }

    class _Resp:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    def _fake_get(url, headers=None, timeout=20):
        if "nifty" in url:
            return _Resp(sample_by_universe["nifty500"])
        return _Resp(sample_by_universe["nse_total"])

    monkeypatch.setattr(hb, "REMOTE_ARCHIVE_URLS", {
        "nifty500": ["https://example.com/nifty500.csv"],
        "nse_total": ["https://example.com/nse_total.csv"],
    })
    monkeypatch.setattr(hb.requests, "get", _fake_get)

    hb.main()

    assert (tmp_path / "data" / "raw_nifty500_archives.csv").exists()
    assert (tmp_path / "data" / "raw_nse_total_archives.csv").exists()
    assert (tmp_path / "data" / "historical_nifty500.parquet").exists()
    assert (tmp_path / "data" / "historical_nse_total.parquet").exists()


def test_main_invokes_fallback_builder_from_2015_when_needed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    called: dict[str, str] = {}

    def _fake_run(universe_arg="both", start_date=""):
        called["universe_arg"] = universe_arg
        called["start_date"] = start_date
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "historical_nifty500.csv").write_text(
            "date,ticker\n2015-01-01,RELIANCE.NS\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(hb, "build_historical_csv", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")))
    monkeypatch.setattr(hb, "build_parquet_from_csv", lambda *args, **kwargs: None)
    def _fake_download_archive(universe_type: str, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "date,ticker\n2015-01-01,RELIANCE\n",
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr(hb, "_download_archive", _fake_download_archive)
    monkeypatch.setattr(hb.pd, "read_csv", lambda *args, **kwargs: pd.DataFrame({"date": ["2015-01-01"], "ticker": ["RELIANCE"]}))
    monkeypatch.setattr(hb, "verify_parquet", lambda *args, **kwargs: None)
    monkeypatch.setattr("build_historical_fallback.run", _fake_run)

    hb.main()

    assert called["universe_arg"] == "nifty500"
    assert called["start_date"] == "2015-01-01"
