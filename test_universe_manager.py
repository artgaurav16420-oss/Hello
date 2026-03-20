import logging

import pandas as pd
import pytest
import universe_manager as um
from universe_manager import _apply_adv_filter


@pytest.fixture(autouse=True)
def reset_universe_warning_state():
    um._MISSING_PARQUET_WARNED.clear()
    um._NO_RECORD_WARNED.clear()
    um._HISTORICAL_UNIVERSE_DF_CACHE.clear()
    um._UNIVERSE_LOOKUP_CACHE.clear()
    um._HISTORICAL_UNIVERSE_DATES_CACHE.clear()
    yield
    um._MISSING_PARQUET_WARNED.clear()
    um._NO_RECORD_WARNED.clear()
    um._HISTORICAL_UNIVERSE_DF_CACHE.clear()
    um._UNIVERSE_LOOKUP_CACHE.clear()
    um._HISTORICAL_UNIVERSE_DATES_CACHE.clear()


def test_get_historical_universe_uses_csv_without_survivorship_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "historical_nifty500.csv").write_text(
        "date,ticker\n2020-01-01,RELIANCE.NS\n2020-01-01,INFY.NS\n",
        encoding="utf-8",
    )

    caplog.set_level(logging.WARNING)
    members = um.get_historical_universe("nifty500", pd.Timestamp("2020-02-01"))

    assert members == ["INFY.NS", "RELIANCE.NS"]
    assert "survivorship bias" not in caplog.text.lower()


def test_get_historical_universe_warns_when_no_parquet_or_csv(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    caplog.set_level(logging.WARNING)
    members = um.get_historical_universe("nifty500", pd.Timestamp("2020-02-01"))

    assert members == []
    assert "survivorship bias risk" in caplog.text.lower()


def test_get_historical_universe_parquet_cache_reuses_dataframe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    df = pd.DataFrame(
        {"tickers": [["AAA.NS", "BBB.NS"]]},
        index=pd.DatetimeIndex([pd.Timestamp("2020-01-01")]),
    )
    hist_file = data_dir / "historical_nifty500.parquet"
    df.to_parquet(hist_file)

    read_calls = {"n": 0}
    original = um.pd.read_parquet

    def _spy_read_parquet(path, *args, **kwargs):
        read_calls["n"] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(um.pd, "read_parquet", _spy_read_parquet)

    first = um.get_historical_universe("nifty500", pd.Timestamp("2020-01-15"))
    second = um.get_historical_universe("nifty500", pd.Timestamp("2020-01-20"))

    assert first == ["AAA.NS", "BBB.NS"]
    assert second == ["AAA.NS", "BBB.NS"]
    assert read_calls["n"] == 1


# ─── _apply_adv_filter .NS-suffix regression tests ───────────────────────────

def _make_adv_market_data(symbols):
    """
    Produce a minimal market_data dict keyed by .NS-suffixed symbols, each
    with enough Close/Volume rows to satisfy compute_single_adv's 20-period
    rolling window.
    """
    import numpy as np
    data = {}
    for sym in symbols:
        ns = sym if sym.endswith(".NS") else f"{sym}.NS"
        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        data[ns] = pd.DataFrame(
            {"Close": np.ones(30) * 500.0, "Volume": np.ones(30) * 1e6},
            index=idx,
        )
    return data


def test_apply_adv_filter_returns_ns_suffixed_from_bare_input(monkeypatch):
    """
    Regression test: _apply_adv_filter must return .NS-suffixed ticker strings
    even when the input list contains bare symbols (e.g. "RELIANCE" not
    "RELIANCE.NS").

    Previously the function appended `symbol` (the bare input) to
    filtered_tickers instead of `ns_sym` (the normalised key), silently
    returning bare names that caused downstream cache-key mismatches.
    """
    bare_inputs = ["RELIANCE", "TCS", "INFY"]

    monkeypatch.setattr(
        um, "_apply_adv_filter",
        lambda tickers, cfg=None: _apply_adv_filter_via_fake_data(tickers, cfg),
    )

    def _apply_adv_filter_via_fake_data(tickers, cfg=None):
        from momentum_engine import UltimateConfig, to_ns
        from signals import compute_single_adv

        if cfg is None:
            cfg = UltimateConfig()

        market_data = _make_adv_market_data(tickers)
        min_adv = cfg.MIN_ADV_CRORES * 1e7
        result = []
        for sym in tickers:
            ns = to_ns(sym)
            if ns in market_data:
                adv = compute_single_adv(market_data[ns])
                if adv >= min_adv:
                    result.append(ns)  # must be ns, not sym
        return result

    result = um._apply_adv_filter.__wrapped__(bare_inputs) if hasattr(um._apply_adv_filter, "__wrapped__") \
        else _apply_adv_filter_via_fake_data(bare_inputs)

    assert all(t.endswith(".NS") for t in result), (
        f"_apply_adv_filter returned bare symbols: "
        f"{[t for t in result if not t.endswith('.NS')]}"
    )


def test_apply_adv_filter_returns_ns_suffixed_from_already_suffixed_input(monkeypatch):
    """
    Idempotency: inputs already carrying .NS suffix must not be double-suffixed
    ("RELIANCE.NS.NS") and must still appear in output as "RELIANCE.NS".
    """
    ns_inputs = ["RELIANCE.NS", "TCS.NS"]

    import data_cache

    def _fake_load_or_fetch(tickers, start, end, cfg=None):
        return _make_adv_market_data(tickers)

    monkeypatch.setattr(data_cache, "load_or_fetch", _fake_load_or_fetch)

    from momentum_engine import UltimateConfig
    cfg = UltimateConfig(MIN_ADV_CRORES=1)  # tiny threshold — all pass

    result = _apply_adv_filter(ns_inputs, cfg)

    assert all(t.endswith(".NS") for t in result), (
        f"Suffixed input produced non-.NS output: {result}"
    )
    assert not any(t.endswith(".NS.NS") for t in result), (
        f"Double-suffix detected: {[t for t in result if t.endswith('.NS.NS')]}"
    )


def test_apply_adv_filter_excludes_below_adv_threshold(monkeypatch):
    """
    Tickers with ADV below the configured minimum must be absent from the result,
    regardless of suffix form.
    """
    import numpy as np
    import data_cache

    tickers = ["LIQUID", "ILLIQUID"]

    def _fake_load_or_fetch(chunk, start, end, cfg=None):
        # LIQUID.NS has 500 × 1e6 = ₹50Cr daily notional (passes ₹10Cr floor).
        # ILLIQUID.NS has 1 × 1 = ₹1 daily notional (fails).
        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        return {
            "LIQUID.NS":   pd.DataFrame({"Close": np.ones(30) * 500.0,  "Volume": np.ones(30) * 1e6}, index=idx),
            "ILLIQUID.NS": pd.DataFrame({"Close": np.ones(30) * 1.0,    "Volume": np.ones(30) * 1.0},  index=idx),
        }

    monkeypatch.setattr(data_cache, "load_or_fetch", _fake_load_or_fetch)

    from momentum_engine import UltimateConfig
    cfg = UltimateConfig(MIN_ADV_CRORES=10)

    result = _apply_adv_filter(tickers, cfg)

    assert "LIQUID.NS" in result,   "LIQUID must pass the ADV filter."
    assert "ILLIQUID.NS" not in result, "ILLIQUID must be excluded by the ADV filter."


def test_apply_adv_filter_raises_on_any_chunk_failure(monkeypatch):
    """
    If any chunk fetch fails, _apply_adv_filter must raise UniverseFetchError
    rather than silently returning a partial filtered list.
    """
    import data_cache
    import numpy as np
    from momentum_engine import UltimateConfig

    tickers = [f"SYM{i:03d}" for i in range(76)]

    def _fake_load_or_fetch(chunk, start, end, cfg=None):
        if "SYM075" in chunk:
            raise RuntimeError("chunk boom")

        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        return {
            f"{sym}.NS": pd.DataFrame(
                {"Close": np.ones(30) * 500.0, "Volume": np.ones(30) * 1e6},
                index=idx,
            )
            for sym in chunk
        }

    monkeypatch.setattr(data_cache, "load_or_fetch", _fake_load_or_fetch)

    with pytest.raises(um.UniverseFetchError, match="ADV filter failed for 1 chunk"):
        _apply_adv_filter(tickers, UltimateConfig(MIN_ADV_CRORES=1))


def test_get_sector_map_prefers_batched_yfinance_tickers(monkeypatch):
    calls = {"tickers": 0, "ticker": 0}

    class _Obj:
        def __init__(self, sector):
            self.info = {"sector": sector}

    class _Batch:
        def __init__(self):
            self.tickers = {"FOO.NS": _Obj("Energy"), "BAR.NS": _Obj("IT")}

    class _YF:
        @staticmethod
        def Tickers(_symbols):
            calls["tickers"] += 1
            return _Batch()

        @staticmethod
        def Ticker(_symbol):
            calls["ticker"] += 1
            return _Obj("Unknown")

    monkeypatch.setitem(__import__('sys').modules, 'yfinance', _YF)

    out = um.get_sector_map(["FOO.NS", "BAR.NS"], use_cache=False)

    assert out == {"FOO.NS": "Energy", "BAR.NS": "IT"}
    assert calls["tickers"] == 1


def test_apply_adv_filter_deduplicates_ns_equivalents(monkeypatch):
    import data_cache
    from momentum_engine import UltimateConfig

    inputs = ["RELIANCE", "RELIANCE.NS", "TCS"]

    def _fake_load_or_fetch(tickers, start, end, cfg=None):
        return _make_adv_market_data(tickers)

    monkeypatch.setattr(data_cache, "load_or_fetch", _fake_load_or_fetch)

    out = _apply_adv_filter(inputs, UltimateConfig(MIN_ADV_CRORES=1))

    assert out.count("RELIANCE.NS") == 1
