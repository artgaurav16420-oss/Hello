from __future__ import annotations

import os
import pandas as pd

import data_cache
from momentum_engine import UltimateConfig


def test_load_or_fetch_uses_dynamic_padding_from_cfg(monkeypatch):
    captured = {}

    monkeypatch.setattr(data_cache, "_load_manifest", lambda: {"schema_version": 1, "entries": {}})
    monkeypatch.setattr(data_cache, "_save_manifest", lambda _manifest: None)

    def _fake_download_with_timeout(tickers, start, end, **kwargs):
        captured["start"] = start
        captured["end"] = end
        return pd.DataFrame()

    monkeypatch.setattr(data_cache, "_download_with_timeout", _fake_download_with_timeout)

    cfg = UltimateConfig(CVAR_LOOKBACK=500)
    try:
        data_cache.load_or_fetch(
            tickers=["ABC"],
            required_start="2024-01-01",
            required_end="2024-12-31",
            force_refresh=True,
            cfg=cfg,
        )
    except data_cache.DataFetchError:
        pass

    assert captured["start"] == "2021-04-06"
    assert captured["end"] == "2025-01-01"


def test_secondary_provider_parses_alpha_vantage_payload(monkeypatch):
    payload = {
        "Time Series (Daily)": {
            "2024-01-03": {
                "1. open": "100",
                "2. high": "110",
                "3. low": "95",
                "4. close": "105",
                "5. adjusted close": "104",
                "6. volume": "1000",
            }
        }
    }

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setenv("FALLBACK_API_KEY", "k")
    monkeypatch.setattr(data_cache.requests, "get", lambda *args, **kwargs: _Resp())

    sp = data_cache.SecondaryProvider()
    out = sp.download(["ABC.NS"], "2024-01-01", "2024-01-31")

    assert out is not None
    assert {"Open", "High", "Low", "Close", "Adj Close", "Volume"}.issubset(set(out.columns))


def test_secondary_provider_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("FALLBACK_API_KEY", raising=False)
    sp = data_cache.SecondaryProvider()
    out = sp.download(["ABC.NS"], "2024-01-01", "2024-01-31")
    assert out is None


def test_load_or_fetch_skips_symbols_on_chunk_failure(monkeypatch):
    monkeypatch.setattr(data_cache, "_load_manifest", lambda: {"schema_version": 1, "entries": {}})
    monkeypatch.setattr(data_cache, "_save_manifest", lambda _manifest: None)
    monkeypatch.setattr(data_cache, "_download_with_timeout", lambda *args, **kwargs: pd.DataFrame())

    out = data_cache.load_or_fetch(
        tickers=["ABC"],
        required_start="2024-01-01",
        required_end="2024-01-31",
        force_refresh=True,
    )

    assert out == {}




def test_extract_ticker_frame_rejects_flat_payload_for_multi_ticker_chunk():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100, 101, 102],
            "Adj Close": [100, 101, 102],
            "Volume": [1, 1, 1],
        },
        index=idx,
    )

    assert data_cache._extract_ticker_frame(raw, "MISSING.NS", is_single_request=False) is None


def test_extract_ticker_frame_accepts_flat_payload_for_single_ticker_chunk():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100, 101, 102],
            "Adj Close": [100, 101, 102],
            "Volume": [1, 1, 1],
        },
        index=idx,
    )

    out = data_cache._extract_ticker_frame(raw, "ABC.NS", is_single_request=True)
    assert out is not None
    assert out["Close"].iloc[-1] == 102

def test_extract_ticker_frame_fills_adj_close_for_multiindex_payload():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    raw = pd.DataFrame(
        {
            ("ABC.NS", "Close"): [100, 101, 102, 103, 104, 105],
            ("ABC.NS", "Adj Close"): [None, None, None, None, None, None],
            ("ABC.NS", "Volume"): [1, 1, 1, 1, 1, 1],
        },
        index=idx,
    )
    out = data_cache._extract_ticker_frame(raw, "ABC.NS")
    assert out is not None
    assert (out["Adj Close"] == out["Close"]).all()


def test_is_valid_dataframe_allows_index_ticker_with_nan_volume():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "Close": [100, 101, 102, 103, 104, 105],
            "Adj Close": [100, 101, 102, 103, 104, 105],
            "Volume": [None, None, None, None, None, None],
        },
        index=idx,
    )

    assert data_cache._is_valid_dataframe(df, ticker="^NSEI")


def test_is_valid_dataframe_rejects_non_index_ticker_with_nan_volume():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "Close": [100, 101, 102, 103, 104, 105],
            "Adj Close": [100, 101, 102, 103, 104, 105],
            "Volume": [None, None, None, None, None, None],
        },
        index=idx,
    )

    assert not data_cache._is_valid_dataframe(df, ticker="ABC.NS")


def test_load_local_env_file_sets_missing_keys_only(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GROWW_API_TOKEN=from_file\nEXISTING_KEY=from_file\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("GROWW_API_TOKEN", raising=False)
    monkeypatch.setenv("EXISTING_KEY", "already_set")

    data_cache._load_local_env_file(env_file)

    assert os.getenv("GROWW_API_TOKEN") == "from_file"
    assert os.getenv("EXISTING_KEY") == "already_set"


def test_ensure_price_columns_coerces_object_dividends_and_prices():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            "Close": ["100", "101.5", "102"],
            "Adj Close": ["100", "101.5", "102"],
            "Volume": ["1,000", "2,000", "3,000"],
            "Dividends": ["0", "2.6 INR", None],
            "Stock Splits": ["0", "0", "0"],
        },
        index=idx,
    )

    out = data_cache._ensure_price_columns(raw)

    assert out["Dividends"].dtype.kind in "fc"
    assert out["Dividends"].iloc[1] == 2.6
    assert out["Volume"].iloc[0] == 1000


def test_recover_from_stale_cache_logs_summary_not_per_symbol(tmp_path, monkeypatch, caplog):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(data_cache, "CACHE_DIR", cache_dir)

    idx = pd.date_range("2026-03-17", periods=6, freq="D")
    for ticker in ("AAA.NS", "BBB.NS"):
        df = pd.DataFrame(
            {
                "Open": [100.0] * len(idx),
                "High": [101.0] * len(idx),
                "Low": [99.0] * len(idx),
                "Close": [100.0] * len(idx),
                "Adj Close": [100.0] * len(idx),
                "Volume": [1000.0] * len(idx),
            },
            index=idx,
        )
        df.to_parquet(cache_dir / f"{ticker}.parquet")

    entries: dict = {}
    market_data: dict = {}
    caplog.set_level("WARNING")

    data_cache._recover_from_stale_cache(["AAA.NS", "BBB.NS"], entries, market_data)

    assert "Recovered 2 symbol(s) from stale local cache" in caplog.text
    assert "Using stale cached parquet for AAA.NS" not in caplog.text
    assert set(market_data.keys()) == {"AAA.NS", "BBB.NS"}
