import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import logging

import historical_builder as hb


def test_build_historical_csv_from_local_master(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw = data_dir / "raw_nifty_archives.csv"
    raw.write_text(
        """date,ticker
2018-01-01,RELIANCE
2018-01-01,TCS
2018-02-01,INFY
""",
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
        """ticker,2020-01-31,2020-02-28
RELIANCE,1,1
TCS,0,1
""",
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
        """date,RELIANCE,TCS
2020-01-31,1,0
2020-02-28,1,1
""",
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
        "nifty500": """date,ticker
2020-01-31,RELIANCE
2020-01-31,TCS
""",
        "nse_total": """date,ticker
2020-01-31,INFY
2020-01-31,SBIN
""",
    }

    class _Resp:
        def __init__(self, text):
            self.text = text
            self.content = text.encode("utf-8")
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            import json
            # If it's the Wayback CDX JSON format, it should be a list of lists
            if self.text.startswith("["):
                return json.loads(self.text)
            return {}

    def _fake_get(url, *args, **kwargs):
        # Handle Wayback CDX query
        if "cdx" in url:
            return _Resp('[["timestamp","statuscode"], ["20200131120000","200"]]')
        if "nifty" in url:
            return _Resp(sample_by_universe["nifty500"])
        if "nse_total" in url:
            return _Resp(sample_by_universe["nse_total"])
        return _Resp("")

    class _FakeSession:
        def get(self, url, *args, **kwargs):
            return _fake_get(url, *args, **kwargs)
        def close(self):
            pass

    monkeypatch.setattr(hb, "REMOTE_ARCHIVE_URLS", {
        "nifty500": ["https://example.com/nifty500.csv"],
        "nse_total": ["https://example.com/nse_total.csv"],
    })
    import requests
    monkeypatch.setattr(requests, "get", _fake_get)
    monkeypatch.setattr(requests, "Session", _FakeSession)

    # Mock out the other build paths so it falls through to _download_archive
    def _fail(*args, **kwargs):
        raise FileNotFoundError("Mock fail")
    
    monkeypatch.setattr(hb, "build_historical_csv", _fail)
    
    # Mock bhf.run to do nothing or raise to fall through
    import build_historical_fallback as bhf
    monkeypatch.setattr(bhf, "run", lambda *args, **kwargs: None)

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
            """date,ticker
2015-01-01,RELIANCE.NS
""",
            encoding="utf-8",
        )

    monkeypatch.setattr(hb, "build_historical_csv", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")))
    monkeypatch.setattr(hb, "build_parquet_from_csv", lambda *args, **kwargs: None)
    def _fake_download_archive(universe_type: str, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            """date,ticker
2015-01-01,RELIANCE
""",
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


class TestHistoricalBuilder:
    def test_build_parquet_from_csv_successful_conversion(self, tmp_path):
        # Create a dummy CSV file
        csv_content = """date,ticker
2023-01-01,TCS
2023-01-01,RELIANCE
2023-01-02,TCS
2023-01-02,INFY"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)

        parquet_path = tmp_path / "output.parquet"

        # Call the function
        result_path = hb.build_parquet_from_csv(str(csv_path), str(parquet_path))

        # Assertions
        assert result_path == parquet_path
        assert parquet_path.exists()

        # Read the parquet file and verify its content and schema
        df = pd.read_parquet(parquet_path)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "date"
        assert "tickers" in df.columns
        assert isinstance(df.loc[pd.Timestamp("2023-01-01").normalize(), "tickers"], np.ndarray)

        # Verify specific content
        expected_data = {
            pd.Timestamp("2023-01-01").normalize(): ["RELIANCE.NS", "TCS.NS"],
            pd.Timestamp("2023-01-02").normalize(): ["INFY.NS", "TCS.NS"],
        }
        assert len(df) == len(expected_data)
        for date, expected_tickers in expected_data.items():
            actual_tickers = df.loc[date, "tickers"].tolist() # Convert numpy array to list for comparison
            assert sorted(actual_tickers) == sorted(expected_tickers)


    def test_build_parquet_from_csv_handles_datetime_errors_gracefully(self, tmp_path):
        csv_content = """date,ticker
2023-01-01,TCS
INVALID-DATE,RELIANCE
2023-01-02,INFY"""
        csv_path = tmp_path / "test_invalid_date.csv"
        csv_path.write_text(csv_content)

        parquet_path = tmp_path / "output_invalid_date.parquet"

        result_path = hb.build_parquet_from_csv(str(csv_path), str(parquet_path))

        # Assertions
        assert result_path == parquet_path
        assert parquet_path.exists()

        df = pd.read_parquet(parquet_path)

        # Only the valid dates should be present
        expected_data = {
            pd.Timestamp("2023-01-01").normalize(): ["TCS.NS"],
            pd.Timestamp("2023-01-02").normalize(): ["INFY.NS"],
        }
        assert len(df) == len(expected_data)
        for date, expected_tickers in expected_data.items():
            actual_tickers = list(df.loc[date, "tickers"]) # Use the helper for consistency
            assert sorted(actual_tickers) == sorted(expected_tickers)

    def test_build_parquet_from_csv_raises_file_not_found_error(self, tmp_path):
        non_existent_csv_path = tmp_path / "non_existent.csv"
        parquet_path = tmp_path / "output.parquet"

        with pytest.raises(FileNotFoundError, match="CSV source not found"):
            hb.build_parquet_from_csv(str(non_existent_csv_path), str(parquet_path))

    def test_build_parquet_from_csv_raises_value_error_for_empty_csv(self, tmp_path):
        empty_csv_path = tmp_path / "empty.csv"
        empty_csv_path.write_text("""date,ticker
""")
        parquet_path = tmp_path / "output.parquet"

        with pytest.raises(ValueError, match="CSV at .* is empty or missing required columns"):
            hb.build_parquet_from_csv(str(empty_csv_path), str(parquet_path))

    def test_build_parquet_from_csv_raises_value_error_for_missing_columns(self, tmp_path):
        missing_col_csv_path = tmp_path / "missing_col.csv"
        missing_col_csv_path.write_text("""date_only,ticker_only
2023-01-01,TCS""")
        parquet_path = tmp_path / "output.parquet"

        with pytest.raises(ValueError, match="CSV at .* is empty or missing required columns"):
            hb.build_parquet_from_csv(str(missing_col_csv_path), str(parquet_path))

    def test_verify_parquet_missing_file(self, tmp_path, capsys):
        non_existent_parquet_path = tmp_path / "non_existent.parquet"
        
        result = hb.verify_parquet(str(non_existent_parquet_path))

        assert result is False
        captured = capsys.readouterr()
        assert f"[Verify] MISSING: {non_existent_parquet_path}" in captured.out

    def test_verify_parquet_empty_file(self, tmp_path, capsys):
        empty_parquet_path = tmp_path / "empty.parquet"
        
        # Create an empty parquet file
        df = pd.DataFrame({"tickers": []}, index=pd.DatetimeIndex([], name="date"))
        df.to_parquet(empty_parquet_path, engine="pyarrow")

        result = hb.verify_parquet(str(empty_parquet_path))

        assert result is False
        captured = capsys.readouterr()
        assert "STATUS: EMPTY — rebuild required" in captured.out

    def test_verify_parquet_valid_pit_data(self, tmp_path, capsys):
        valid_parquet_path = tmp_path / "valid.parquet"
        
        # Create a DataFrame that should pass verification
        dates = pd.to_datetime(pd.date_range(start="2018-01-01", periods=20, freq="6ME"))
        
        # Manually create more diverse tickers over time
        all_tickers = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS", "ICICIBANK.NS", "BAJFINANCE.NS"]
        df_rows = []
        for i, date in enumerate(dates):
            # Simulate changing universe
            current_tickers = sorted(list(set(all_tickers[:3+i]) - set(all_tickers[i:i+1]))) # Simple change
            if not current_tickers: # Ensure at least one ticker
                current_tickers = ["TCS.NS"] 
            if len(current_tickers) < 3: # Ensure enough tickers for diverse universe
                current_tickers.append(all_tickers[3+i] if 3+i < len(all_tickers) else "NEWTICKER.NS")
            df_rows.append({"date": date, "tickers": sorted(list(set(current_tickers)))})

        df_valid = pd.DataFrame(df_rows).set_index("date")
        df_valid.to_parquet(valid_parquet_path, engine="pyarrow")

        result = hb.verify_parquet(str(valid_parquet_path))

        assert result is True
        captured = capsys.readouterr()
        assert "STATUS: OK — looks like valid PIT data" in captured.out

    def test_verify_parquet_biased_too_few_snapshots(self, tmp_path, capsys):
        biased_parquet_path = tmp_path / "few_snapshots.parquet"
        
        dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=5, freq="D")) # Only 5 snapshots
        df_biased = pd.DataFrame({"tickers": [["TCS.NS"]] * 5}, index=dates)
        df_biased.index.name = "date"
        df_biased.to_parquet(biased_parquet_path, engine="pyarrow")

        result = hb.verify_parquet(str(biased_parquet_path))

        assert result is False
        captured = capsys.readouterr()
        assert "Too few snapshots (5), expected 20+" in captured.out
        assert "STATUS: BIASED / INVALID — rebuild required" in captured.out

    def test_verify_parquet_biased_low_divergence(self, tmp_path, capsys):
        biased_parquet_path = tmp_path / "low_divergence.parquet"
        
        # Create a DataFrame where first and last snapshots are nearly identical
        dates = pd.to_datetime(pd.date_range(start="2018-01-01", periods=30, freq="ME")) # Changed M to ME
        base_tickers = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS"]

        df_rows = []
        # First snapshot
        df_rows.append({"date": dates[0], "tickers": sorted(base_tickers)})

        # Intermediate snapshots can have minor, transient changes
        for i in range(1, len(dates) - 1):
            modified_tickers = base_tickers[:]
            # Introduce a temporary change that reverts
            if i % 2 == 0 and len(modified_tickers) > 1:
                modified_tickers[0], modified_tickers[1] = modified_tickers[1], modified_tickers[0]
            df_rows.append({"date": dates[i], "tickers": sorted(list(set(modified_tickers)))})

        # Last snapshot: make it identical to the first. This will result in 0.0% divergence.
        df_rows.append({"date": dates[-1], "tickers": sorted(base_tickers)})

        df_biased = pd.DataFrame(df_rows).set_index("date")
        df_biased.to_parquet(biased_parquet_path, engine="pyarrow")

        result = hb.verify_parquet(str(biased_parquet_path))

        assert result is False
        captured = capsys.readouterr()
        assert "First and last snapshot are nearly identical" in captured.out
        assert "STATUS: BIASED / INVALID — rebuild required" in captured.out

    def test_verify_parquet_biased_late_joiners(self, tmp_path, capsys):
        biased_parquet_path = tmp_path / "late_joiners.parquet"

        # Create a DataFrame with enough snapshots and recent end date, but with a late-joiner in pre-2021
        # Use more dates to avoid "Too few snapshots" issue
        start_date = pd.Timestamp("2015-01-01")
        end_date = pd.Timestamp.today().normalize()
        # Generate enough dates to avoid "too few snapshots", ensuring ME frequency
        dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq="6ME"))

        df_rows = []
        base_tickers = ["TCS.NS", "RELIANCE.NS", "INFY.NS"]
        late_joiner = "ZOMATO.NS" # Known late-joiner
        # First snapshot: only base tickers
        df_rows.append({"date": dates[0], "tickers": sorted(base_tickers)})

        # Intermediate snapshots
        for i in range(1, len(dates) - 1):
            current_tickers = list(base_tickers)
            if dates[i] < pd.Timestamp("2021-07-01"):
                # Introduce late_joiner in all pre-2021 snapshots to trigger the bias
                current_tickers.append(late_joiner)
            df_rows.append({"date": dates[i], "tickers": sorted(list(set(current_tickers)))})

        # Last snapshot: make it significantly different
        last_snapshot_tickers = list(base_tickers)
        last_snapshot_tickers.append(late_joiner)
        last_snapshot_tickers.append("LASTADDITION.NS")
        df_rows.append({"date": dates[-1], "tickers": sorted(list(set(last_snapshot_tickers)))})

        df_biased = pd.DataFrame(df_rows).set_index("date")
        df_biased.to_parquet(biased_parquet_path, engine="pyarrow")

        result = hb.verify_parquet(str(biased_parquet_path))

        assert result is False
        captured = capsys.readouterr()
        assert "Pre-2021 snapshot contains post-2021 IPOs: {'ZOMATO.NS'}" in captured.out
        assert "STATUS: BIASED / INVALID — rebuild required" in captured.out

    def test_verify_parquet_biased_oversized_samples(self, tmp_path, capsys):
        biased_parquet_path = tmp_path / "oversized_samples.parquet"
        
        dates = pd.to_datetime(pd.date_range(start="2019-01-01", periods=10, freq="6ME"))
        
        # Create oversized sample (e.g., 550 tickers) for all dates
        oversized_tickers = [f"TICKER{i}.NS" for i in range(550)]
        
        df_rows = [{"date": date, "tickers": oversized_tickers} for date in dates]
        
        df_biased = pd.DataFrame(df_rows).set_index("date")
        df_biased.to_parquet(biased_parquet_path, engine="pyarrow")

        result = hb.verify_parquet(str(biased_parquet_path))

        assert result is False
        captured = capsys.readouterr()
        assert "← BIASED (today's list?)" in captured.out # Checks if the flag for oversized is present
        assert "STATUS: BIASED / INVALID — rebuild required" in captured.out

    def test_verify_parquet_biased_latest_snapshot_too_old(self, tmp_path, capsys):
        biased_parquet_path = tmp_path / "too_old.parquet"
        
        # Create a DataFrame where the latest snapshot is >1 year old
        old_date = pd.Timestamp.today() - pd.Timedelta(days=366) # More than 1 year ago
        dates = pd.to_datetime(pd.date_range(end=old_date, periods=5, freq="D"))
        df_biased = pd.DataFrame({"tickers": [["TCS.NS"]] * 5}, index=dates)
        df_biased.index.name = "date"
        df_biased.to_parquet(biased_parquet_path, engine="pyarrow")

        result = hb.verify_parquet(str(biased_parquet_path))

        assert result is False
        captured = capsys.readouterr()
        assert f"Latest snapshot is >1 year old ({old_date.date()})" in captured.out
        assert "STATUS: BIASED / INVALID — rebuild required" in captured.out