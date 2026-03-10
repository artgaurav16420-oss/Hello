from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import daily_workflow as dw
from momentum_engine import PortfolioState, UltimateConfig


def test_prompt_menu_choice_retries_until_valid(monkeypatch):
    responses = iter(["bad", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))

    choice = dw._prompt_menu_choice("Choice: ", ["1", "2", "q"])

    assert choice == "2"


def test_load_optimized_config_ignores_unknown_fields(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "optimal_cfg.json").write_text(
        json.dumps({"HALFLIFE_FAST": 34, "DOES_NOT_EXIST": 1}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    cfg = dw.load_optimized_config()

    assert cfg.HALFLIFE_FAST == 34
    assert not hasattr(cfg, "DOES_NOT_EXIST")


def test_get_custom_universe_uses_local_fallback_when_confirmed(tmp_path: Path, monkeypatch):
    fallback = tmp_path / "custom_screener.txt"
    fallback.write_text("ABC\n123456\nXYZ\nABC\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dw, "_scrape_screener", lambda _url: [])
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    tickers = dw._get_custom_universe()

    assert tickers == ["ABC", "XYZ"]


def test_run_scan_returns_early_when_universe_has_no_data(monkeypatch):
    captured = {}

    def _fake_load_or_fetch(tickers, required_start, required_end, cfg=None):
        captured["tickers"] = tickers
        captured["required_start"] = required_start
        captured["required_end"] = required_end
        captured["cfg"] = cfg
        return {"^NSEI": pd.DataFrame({"Close": [100.0]}, index=pd.date_range("2024-01-01", periods=1))}

    monkeypatch.setattr(dw, "load_or_fetch", _fake_load_or_fetch)
    monkeypatch.setattr(dw, "_print_stage_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "detect_and_apply_splits", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)

    cfg = UltimateConfig(CVAR_LOOKBACK=260)
    state = PortfolioState()

    returned_state, market_data = dw._run_scan(["ABC"], state, "TEST", cfg_override=cfg)

    assert returned_state is state
    assert "^NSEI" in market_data
    assert captured["cfg"] is cfg


def test_load_portfolio_state_raises_when_all_backups_corrupted(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    base = data_dir / "portfolio_state_main.json"
    base.write_text("{not json", encoding="utf-8")
    (data_dir / "portfolio_state_main.json.bak.0").write_text("{also bad", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="all discovered state files are corrupted"):
        dw.load_portfolio_state("main")
