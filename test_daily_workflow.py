from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import daily_workflow as dw
from momentum_engine import PortfolioState, UltimateConfig, execute_rebalance


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


def test_detect_and_apply_splits_runs_on_raw_prices_even_when_auto_adjust_flag_true():
    cfg = UltimateConfig(AUTO_ADJUST_PRICES=True, SPLIT_TOLERANCE=0.01)
    state = PortfolioState(
        shares={"ABC": 10},
        entry_prices={"ABC": 1000.0},
        last_known_prices={"ABC": 1000.0},
        cash=0.0,
    )
    idx = pd.date_range("2024-01-01", periods=2)
    market_data = {
        "ABC.NS": pd.DataFrame({"Close": [1000.0, 100.0], "Dividends": [0.0, 0.0]}, index=idx)
    }

    adjusted = dw.detect_and_apply_splits(state, market_data, cfg)

    assert adjusted == ["ABC"]
    assert state.shares["ABC"] == 100
    assert state.entry_prices["ABC"] == 100.0


def test_run_scan_cadence_gate_skips_rebalance(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=6)
    md = {
        "ABC.NS": pd.DataFrame({"Close": [100, 101, 102, 103, 104, 105], "Dividends": [0, 0, 0, 0, 0, 0]}, index=idx),
        "^NSEI": pd.DataFrame({"Close": [100] * 6}, index=idx),
        "^CRSLDX": pd.DataFrame({"Close": [100] * 6}, index=idx),
    }
    monkeypatch.setattr(dw, "load_or_fetch", lambda *_args, **_kwargs: md)
    monkeypatch.setattr(dw, "_print_stage_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "detect_and_apply_splits", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)

    called = {"n": 0}

    def _boom(*_args, **_kwargs):
        called["n"] += 1
        raise AssertionError("execute_rebalance should not be called when cadence gate blocks")

    monkeypatch.setattr(dw, "execute_rebalance", _boom)

    state = PortfolioState(
        shares={"ABC": 10},
        entry_prices={"ABC": 100.0},
        last_known_prices={"ABC": 100.0},
        last_rebalance_date=datetime.today().strftime("%Y-%m-%d"),
    )
    cfg = UltimateConfig(REBALANCE_FREQ="W-FRI")

    out_state, _ = dw._run_scan(["ABC"], state, "TEST", cfg_override=cfg)

    assert called["n"] == 0
    assert out_state.absent_periods == {}


def test_run_scan_increments_absent_periods_when_symbol_missing(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=6)
    md = {
        "^NSEI": pd.DataFrame({"Close": [100] * 6}, index=idx),
        "^CRSLDX": pd.DataFrame({"Close": [100] * 6}, index=idx),
    }
    monkeypatch.setattr(dw, "load_or_fetch", lambda *_args, **_kwargs: md)
    monkeypatch.setattr(dw, "_print_stage_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "detect_and_apply_splits", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)

    state = PortfolioState(
        shares={"MISSING": 5},
        last_known_prices={"MISSING": 10.0},
        absent_periods={"MISSING": 2},
        last_rebalance_date=datetime.today().strftime("%Y-%m-%d"),
    )
    cfg = UltimateConfig(REBALANCE_FREQ="W-FRI", MAX_ABSENT_PERIODS=10)

    out_state, _ = dw._run_scan(["ABC"], state, "TEST", cfg_override=cfg)

    assert out_state.absent_periods["MISSING"] == 3


def test_execute_rebalance_initializes_dividend_marker_on_new_position():
    cfg = UltimateConfig()
    state = PortfolioState(cash=10_000.0)
    execute_rebalance(
        state=state,
        target_weights=pd.Series([1.0]).values,
        prices=pd.Series([100.0]).values,
        active_symbols=["ABC"],
        cfg=cfg,
        date_context=pd.Timestamp("2025-01-15"),
    )

    assert state.shares.get("ABC", 0) > 0
    assert state.dividend_ledger["ABC"].startswith("2025-01-15:")
