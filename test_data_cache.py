from __future__ import annotations

import os
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import data_cache
from data_cache import DataProvider
from momentum_engine import UltimateConfig


# Mock implementation of DataProvider for testing purposes
class MockDataProvider(DataProvider):
    def __init__(self, data_frames: dict[str, pd.DataFrame], should_fail=False):
        self.data_frames = data_frames
        self.should_fail = should_fail
        self.download_calls = []

    def download(self, tickers: list[str], start: str, end: str) -> pd.DataFrame | None:
        self.download_calls.append({"tickers": tickers, "start": start, "end": end})
        if self.should_fail:
            return None

        # Simulate yfinance's multi-index output for multiple tickers
        if len(tickers) > 1:
            combined_df = pd.DataFrame()
            for ticker in tickers:
                if ticker in self.data_frames:
                    df = self.data_frames[ticker].copy()
                    df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
                    combined_df = pd.concat([combined_df, df], axis=1)
            return combined_df if not combined_df.empty else None
        else: # Single ticker request
            ticker = tickers[0]
            if ticker in self.data_frames:
                return self.data_frames[ticker].copy()
            return None


class TestDataCache:
    @pytest.fixture(autouse=True)
    def setup_method(self, monkeypatch, tmp_path):
        self.temp_cache_dir = tmp_path / "test_cache"
        self.temp_cache_dir.mkdir()

        monkeypatch.setattr(data_cache, "CACHE_DIR", self.temp_cache_dir)
        monkeypatch.setattr(data_cache, "MANIFEST_FILE", self.temp_cache_dir / "_manifest.json")
        monkeypatch.setattr(data_cache, "_MANIFEST_LOCK_DIR", self.temp_cache_dir / "_manifest.lockfile")
        monkeypatch.setattr(data_cache, "_CACHE_CONFIGURED", True)

        self.mock_manifest_data = {"schema_version": 1, "entries": {}}

        def mock_load_manifest():
            return self.mock_manifest_data

        def mock_save_manifest(manifest_data):
            self.mock_manifest_data = manifest_data

        monkeypatch.setattr(data_cache, "_load_manifest", mock_load_manifest)
        monkeypatch.setattr(data_cache, "_save_manifest", mock_save_manifest)

        # Mock _latest_business_day to return a fixed date for consistent staleness checks
        self.mock_latest_bday = "2024-07-20"
        monkeypatch.setattr(data_cache, "_latest_business_day", lambda: self.mock_latest_bday)

        # Mock the UltimateConfig
        self.mock_cfg = MagicMock(spec=UltimateConfig)
        self.mock_cfg.CVAR_LOOKBACK = 200

        # Helper to create a dummy DataFrame
        def create_dummy_df(start_date, end_date):
            dates = pd.date_range(start=start_date, end=end_date)
            return pd.DataFrame({
                "Open": 100, "High": 105, "Low": 98, "Close": 102,
                "Adj Close": 102, "Volume": 1000,
                "Dividends": 0.0, "Stock Splits": 0.0
            }, index=dates)
        self.create_dummy_df = create_dummy_df

    def test_load_or_fetch_retrieves_fresh_data_from_cache(self, monkeypatch):
        ticker = "TEST.NS"
        # Ensure fresh_data covers up to mock_latest_bday to avoid staleness
        fresh_data = self.create_dummy_df("2024-07-01", self.mock_latest_bday)

        # Simulate initial download and save to cache
        mock_provider = MockDataProvider({ticker: fresh_data})
        monkeypatch.setattr(data_cache, "_build_provider_chain", lambda cfg: [mock_provider])

        # First call: data should be downloaded and cached
        result1 = data_cache.load_or_fetch(
            tickers=[ticker],
            required_start="2024-07-01",
            required_end=self.mock_latest_bday, # Match required_end to fresh_data's end
            cfg=self.mock_cfg
        )

        assert ticker in result1
        pd.testing.assert_frame_equal(result1[ticker], fresh_data)
        assert len(mock_provider.download_calls) == 1
        assert (self.temp_cache_dir / f"{ticker}.parquet").exists()
        assert self.mock_manifest_data["entries"][ticker]["last_date"] == fresh_data.index[-1].strftime("%Y-%m-%d")

        # Clear download calls to check subsequent cache hit
        mock_provider.download_calls = []

        # Second call: data should be retrieved from cache, no new download
        result2 = data_cache.load_or_fetch(
            tickers=[ticker],
            required_start="2024-07-01",
            required_end=self.mock_latest_bday, # Match required_end to fresh_data's end
            cfg=self.mock_cfg
        )

        assert ticker in result2
        pd.testing.assert_frame_equal(result2[ticker], fresh_data)
        assert len(mock_provider.download_calls) == 0 # No new download calls

    def test_load_or_fetch_refreshes_stale_data(self, monkeypatch):
        ticker = "STALE.NS"
        stale_data = self.create_dummy_df("2024-07-01", "2024-07-10") # Older than mock_latest_bday
        updated_data = self.create_dummy_df("2024-07-01", "2024-07-19")
        # Normalize updated_data so it matches the output of data_cache functions
        expected_updated_data = data_cache._ensure_price_columns(data_cache._normalize_history_index(updated_data.copy()))

        # Manually create a stale cache file and manifest entry
        stale_data.to_parquet(self.temp_cache_dir / f"{ticker}.parquet")
        self.mock_manifest_data["entries"][ticker] = {
            "last_date": "2024-07-10",
            "fetched_at": "2024-07-10T00:00:00+05:30",
            "rows": len(stale_data),
            "suspended": False,
            "max_gap_days": 0,
        }

        mock_provider = MockDataProvider({ticker: updated_data})
        monkeypatch.setattr(data_cache, "_build_provider_chain", lambda cfg: [mock_provider])

        result = data_cache.load_or_fetch(
            tickers=[ticker],
            required_start="2024-07-01",
            required_end="2024-07-19",
            cfg=self.mock_cfg
        )

        assert ticker in result
        pd.testing.assert_frame_equal(result[ticker], expected_updated_data)
        assert len(mock_provider.download_calls) == 1 # Should trigger a re-download
        assert self.mock_manifest_data["entries"][ticker]["last_date"] == "2024-07-19" # Manifest updated

    def test_load_or_fetch_downloads_new_data(self, monkeypatch):
        ticker = "NEW.NS"
        new_data = self.create_dummy_df("2024-07-01", "2024-07-19")
        # Normalize new_data so it matches the output of data_cache functions
        expected_new_data = data_cache._ensure_price_columns(data_cache._normalize_history_index(new_data.copy()))


        # Ensure no cache file or manifest entry exists
        assert not (self.temp_cache_dir / f"{ticker}.parquet").exists()
        assert ticker not in self.mock_manifest_data["entries"]

        mock_provider = MockDataProvider({ticker: new_data})
        monkeypatch.setattr(data_cache, "_build_provider_chain", lambda cfg: [mock_provider])

        result = data_cache.load_or_fetch(
            tickers=[ticker],
            required_start="2024-07-01",
            required_end="2024-07-19",
            cfg=self.mock_cfg
        )

        assert ticker in result
        pd.testing.assert_frame_equal(result[ticker], expected_new_data)
        assert len(mock_provider.download_calls) == 1
        assert (self.temp_cache_dir / f"{ticker}.parquet").exists()
        assert self.mock_manifest_data["entries"][ticker]["last_date"] == "2024-07-19"

    def test_load_or_fetch_force_refresh_bypasses_cache(self, monkeypatch):
        ticker = "FORCE.NS"
        cached_data = self.create_dummy_df("2024-07-01", "2024-07-19") # Fresh data
        new_data_on_refresh = self.create_dummy_df("2024-07-01", "2024-07-20")
        # Normalize new_data_on_refresh so it matches the output of data_cache functions
        expected_new_data_on_refresh = data_cache._ensure_price_columns(data_cache._normalize_history_index(new_data_on_refresh.copy()))

        # Manually create a fresh cache file and manifest entry
        cached_data.to_parquet(self.temp_cache_dir / f"{ticker}.parquet")
        self.mock_manifest_data["entries"][ticker] = {
            "last_date": "2024-07-19", # Not stale
            "fetched_at": "2024-07-19T00:00:00+05:30",
            "rows": len(cached_data),
            "suspended": False,
            "max_gap_days": 0,
        }

        mock_provider = MockDataProvider({ticker: new_data_on_refresh})
        monkeypatch.setattr(data_cache, "_build_provider_chain", lambda cfg: [mock_provider])

        result = data_cache.load_or_fetch(
            tickers=[ticker],
            required_start="2024-07-01",
            required_end="2024-07-20",
            force_refresh=True, # Force refresh
            cfg=self.mock_cfg
        )

        assert ticker in result
        pd.testing.assert_frame_equal(result[ticker], new_data_on_refresh.loc[:"2024-07-20"])
        assert len(mock_provider.download_calls) == 1 # Should re-download
        assert self.mock_manifest_data["entries"][ticker]["last_date"] == "2024-07-20" # Manifest updated

    def test_load_or_fetch_re_downloads_corrupted_cache(self, monkeypatch):
        ticker = "CORRUPT.NS"
        good_data = self.create_dummy_df("2024-07-01", "2024-07-19")
        # Normalize good_data so it matches the output of data_cache functions
        expected_good_data = data_cache._ensure_price_columns(data_cache._normalize_history_index(good_data.copy()))

        # Create a "corrupted" cache file by writing invalid content
        corrupted_path = self.temp_cache_dir / f"{ticker}.parquet"
        corrupted_path.write_text("this is not valid parquet data")

        # Add an entry to manifest, as if it was valid before corruption
        self.mock_manifest_data["entries"][ticker] = {
            "last_date": "2024-07-19",
            "fetched_at": "2024-07-19T00:00:00+05:30",
            "rows": len(good_data),
            "suspended": False,
            "max_gap_days": 0,
        }

        mock_provider = MockDataProvider({ticker: good_data})
        monkeypatch.setattr(data_cache, "_build_provider_chain", lambda cfg: [mock_provider])

        result = data_cache.load_or_fetch(
            tickers=[ticker],
            required_start="2024-07-01",
            required_end="2024-07-19",
            cfg=self.mock_cfg
        )

        assert ticker in result
        pd.testing.assert_frame_equal(result[ticker], expected_good_data)
        assert len(mock_provider.download_calls) == 1 # Should re-download due to corruption
        # Verify the corrupted file was replaced with valid data
        re_read_df = pd.read_parquet(corrupted_path)
        # Normalize the re-read data before comparison
        re_read_df_normalized = data_cache._ensure_price_columns(data_cache._normalize_history_index(re_read_df.copy()))
        pd.testing.assert_frame_equal(re_read_df_normalized, expected_good_data)
        assert self.mock_manifest_data["entries"][ticker]["last_date"] == "2024-07-19"
