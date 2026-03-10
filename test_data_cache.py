from __future__ import annotations

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


def test_load_or_fetch_raises_on_chunk_failure(monkeypatch):
    monkeypatch.setattr(data_cache, "_load_manifest", lambda: {"schema_version": 1, "entries": {}})
    monkeypatch.setattr(data_cache, "_save_manifest", lambda _manifest: None)
    monkeypatch.setattr(data_cache, "_download_with_timeout", lambda *args, **kwargs: pd.DataFrame())

    try:
        data_cache.load_or_fetch(
            tickers=["ABC"],
            required_start="2024-01-01",
            required_end="2024-01-31",
            force_refresh=True,
        )
    except data_cache.DataFetchError:
        return

    raise AssertionError("Expected DataFetchError for an empty chunk response")


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
    assert out["Adj Close"].equals(out["Close"])


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
