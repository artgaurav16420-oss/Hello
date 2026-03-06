from __future__ import annotations

import pandas as pd

import data_cache
from momentum_engine import UltimateConfig


def test_load_or_fetch_uses_dynamic_padding_from_cfg(monkeypatch):
    captured = {}

    monkeypatch.setattr(data_cache, "_load_manifest", lambda: {"schema_version": 1, "entries": {}})
    monkeypatch.setattr(data_cache, "_save_manifest", lambda _manifest: None)

    def _fake_download_with_timeout(tickers, start, end):
        captured["start"] = start
        captured["end"] = end
        return pd.DataFrame()

    monkeypatch.setattr(data_cache, "_download_with_timeout", _fake_download_with_timeout)

    cfg = UltimateConfig(CVAR_LOOKBACK=500)
    data_cache.load_or_fetch(
        tickers=["ABC"],
        required_start="2024-01-01",
        required_end="2024-12-31",
        force_refresh=True,
        cfg=cfg,
    )

    assert captured["start"] == "2021-04-06"
    assert captured["end"] == "2024-12-31"
