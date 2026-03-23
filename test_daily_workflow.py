from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import daily_workflow as dw
from momentum_engine import (
    OptimizationError,
    OptimizationErrorType,
    PortfolioState,
    UltimateConfig,
    execute_rebalance,
)
from log_config import current_correlation_id


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


def test_run_scan_keeps_context_active_and_flushes_dead_letter_without_trades(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=6)
    md = {
        "ABC.NS": pd.DataFrame({"Close": [float("nan")] * 6, "Dividends": [0.0] * 6}, index=idx),
        "^NSEI": pd.DataFrame({"Close": [100.0] * 6}, index=idx),
        "^CRSLDX": pd.DataFrame({"Close": [100.0] * 6}, index=idx),
    }
    captured = {"engine_corr": None, "flush_corr": None, "flush_calls": 0}

    class _FakeDeadLetterTracker:
        def __init__(self, threshold=10):
            self.entries = []

        def add(self, symbol, reason, detail=""):
            self.entries.append((symbol, reason, detail))

        def flush(self, logger_name="dead_letter"):
            captured["flush_calls"] += 1
            captured["flush_corr"] = current_correlation_id()

    class _Engine:
        def __init__(self, _cfg):
            captured["engine_corr"] = current_correlation_id()

    monkeypatch.setattr(dw, "DeadLetterTracker", _FakeDeadLetterTracker)
    monkeypatch.setattr(dw, "InstitutionalRiskEngine", _Engine)
    monkeypatch.setattr(dw, "load_or_fetch", lambda *_args, **_kwargs: md)
    monkeypatch.setattr(dw, "_print_stage_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "detect_and_apply_splits", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)

    state = PortfolioState(
        shares={"ABC": 10},
        entry_prices={"ABC": 100.0},
        last_known_prices={"ABC": 99.0},
        last_rebalance_date=datetime.today().strftime("%Y-%m-%d"),
    )
    cfg = UltimateConfig(REBALANCE_FREQ="W-FRI")

    out_state, _ = dw._run_scan(["ABC"], state, "TEST", cfg_override=cfg)

    assert out_state is state
    assert captured["engine_corr"] is not None
    assert captured["flush_calls"] == 1
    assert captured["flush_corr"] == captured["engine_corr"]




def test_run_scan_uses_last_known_price_when_live_quote_is_all_nan(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=6)
    md = {
        "BAD.NS": pd.DataFrame({"Close": [float("nan")] * 6, "Dividends": [0.0] * 6}, index=idx),
        "GOOD.NS": pd.DataFrame({"Close": [100, 101, 102, 103, 104, 105], "Dividends": [0.0] * 6}, index=idx),
        "^NSEI": pd.DataFrame({"Close": [100] * 6}, index=idx),
        "^CRSLDX": pd.DataFrame({"Close": [100] * 6}, index=idx),
    }

    monkeypatch.setattr(dw, "load_or_fetch", lambda *_args, **_kwargs: md)
    monkeypatch.setattr(dw, "_print_stage_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "detect_and_apply_splits", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(dw, "compute_book_cvar", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(dw, "compute_adv", lambda *_args, **_kwargs: __import__("numpy").array([1e9, 1e9]))
    monkeypatch.setattr(dw, "get_sector_map", lambda syms, cfg=None: {s: "Unknown" for s in syms})
    monkeypatch.setattr(dw, "execute_rebalance", lambda *args, **kwargs: 0.0)

    def _fake_generate_signals(*_args, **_kwargs):
        import numpy as np
        # FIX-MB2-TESTGATENAMES: Use renamed keys from FIX-MB-GATENAMES.
        # "history_gated"/"adv_gated"/"knife_gated" → "history_failed"/"adv_failed"/"knife_failed"
        return np.array([0.0, 0.01]), np.array([0.0, 1.0]), [1], {
            "total": 2, "history_failed": 0, "adv_failed": 0, "knife_failed": 1, "selected": 1
        }

    monkeypatch.setattr(dw, "generate_signals", _fake_generate_signals)

    class _Engine:
        def __init__(self, _cfg):
            pass

        def optimize(self, **kwargs):
            import numpy as np
            assert np.isfinite(kwargs["portfolio_value"])
            assert kwargs["portfolio_value"] > 0
            return __import__("numpy").array([1.0])

    monkeypatch.setattr(dw, "InstitutionalRiskEngine", _Engine)

    state = PortfolioState(cash=10_000.0)
    state.shares = {"BAD": 10}
    state.entry_prices = {"BAD": 100.0}
    state.last_known_prices = {"BAD": 99.0}

    dw._run_scan(["BAD", "GOOD"], state, "TEST", cfg_override=UltimateConfig())

def test_load_portfolio_state_returns_safe_default_when_all_backups_corrupted(
    tmp_path: Path, monkeypatch
):
    """PROD-FIX-1: corrupted state returns safe zero-position default, does not raise."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    base = data_dir / "portfolio_state_main.json"
    base.write_text("{not json", encoding="utf-8")
    (data_dir / "portfolio_state_main.json.bak.0").write_text("{also bad", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    state = dw.load_portfolio_state("main")
    assert state.shares == {}
    assert state.cash == 1_000_000.0


def test_detect_and_apply_splits_requires_explicit_stock_splits_signal():
    cfg = UltimateConfig(AUTO_ADJUST_PRICES=True)
    state = PortfolioState(
        shares={"ABC": 10},
        entry_prices={"ABC": 1000.0},
        last_known_prices={"ABC": 1000.0},
        cash=0.0,
    )
    idx = pd.date_range("2024-01-01", periods=2)
    market_data = {
        "ABC.NS": pd.DataFrame(
            {
                "Close": [1000.0, 100.0],
                "Dividends": [0.0, 0.0],
                "Stock Splits": [0.0, 0.0],
            },
            index=idx,
        )
    }

    adjusted = dw.detect_and_apply_splits(state, market_data, cfg)

    assert adjusted == []
    assert state.shares["ABC"] == 10


def test_detect_and_apply_splits_applies_when_stock_splits_column_marks_event():
    cfg = UltimateConfig(AUTO_ADJUST_PRICES=True)
    state = PortfolioState(
        shares={"ABC": 10},
        entry_prices={"ABC": 1000.0},
        last_known_prices={"ABC": 1000.0},
        cash=0.0,
    )
    idx = pd.date_range("2024-01-01", periods=2)
    market_data = {
        "ABC.NS": pd.DataFrame(
            {
                "Close": [1000.0, 500.0],
                "Dividends": [0.0, 0.0],
                "Stock Splits": [0.0, 2.0],
            },
            index=idx,
        )
    }

    adjusted = dw.detect_and_apply_splits(state, market_data, cfg)

    assert adjusted == ["ABC"]
    assert state.shares["ABC"] == 20
    assert state.entry_prices["ABC"] == 500.0
    assert state.cash == pytest.approx(0.0)

def test_detect_and_apply_splits_first_run_uses_position_marker_not_full_history():
    cfg = UltimateConfig(AUTO_ADJUST_PRICES=True)
    state = PortfolioState(
        shares={"ABC": 10},
        entry_prices={"ABC": 1000.0},
        last_known_prices={"ABC": 1000.0},
        dividend_ledger={"ABC": "2024-01-03:0.00000000"},
        cash=0.0,
    )
    idx = pd.date_range("2024-01-01", periods=5)
    market_data = {
        "ABC.NS": pd.DataFrame(
            {
                "Close": [1000.0, 500.0, 505.0, 252.5, 255.0],
                "Dividends": [0.0] * 5,
                "Stock Splits": [0.0, 2.0, 0.0, 2.0, 0.0],
            },
            index=idx,
        )
    }

    adjusted = dw.detect_and_apply_splits(state, market_data, cfg)

    assert adjusted == ["ABC"]
    assert state.shares["ABC"] == 20
    assert state.entry_prices["ABC"] == 500.0


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


def test_run_scan_stale_prices_block_decay_rebalance(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=6)
    stale_idx = pd.date_range("2024-01-01", periods=3)
    md = {
        "ABC.NS": pd.DataFrame({"Close": [100.0] * len(stale_idx), "Dividends": [0.0] * len(stale_idx)}, index=stale_idx),
        "FRESH.NS": pd.DataFrame({"Close": [100.0] * len(idx), "Dividends": [0.0] * len(idx)}, index=idx),
        "^NSEI": pd.DataFrame({"Close": [100.0] * len(idx)}, index=idx),
        "^CRSLDX": pd.DataFrame({"Close": [100.0] * len(idx)}, index=idx),
    }
    monkeypatch.setattr(dw, "load_or_fetch", lambda *_args, **_kwargs: md)
    monkeypatch.setattr(dw, "_print_stage_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "detect_and_apply_splits", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(dw, "compute_book_cvar", lambda *_args, **_kwargs: UltimateConfig().CVAR_DAILY_LIMIT + 0.01)
    monkeypatch.setattr(dw, "compute_adv", lambda *_args, **_kwargs: __import__("numpy").array([1e9, 1e9]))
    monkeypatch.setattr(dw, "get_sector_map", lambda syms, cfg=None: {s: "Unknown" for s in syms})

    def _fake_generate_signals(*_args, **_kwargs):
        import numpy as np
        return np.array([0.01, 0.0]), np.array([1.0, 0.0]), [0], {
            "total": 2, "history_failed": 0, "adv_failed": 0, "knife_failed": 1, "selected": 1
        }

    monkeypatch.setattr(dw, "generate_signals", _fake_generate_signals)

    class _Engine:
        def __init__(self, _cfg):
            pass

        def optimize(self, **_kwargs):
            raise OptimizationError("solver failed", OptimizationErrorType.NUMERICAL)

    monkeypatch.setattr(dw, "InstitutionalRiskEngine", _Engine)

    called = {"n": 0}

    def _boom(*_args, **_kwargs):
        called["n"] += 1
        raise AssertionError("execute_rebalance should not be called when held prices are stale")

    monkeypatch.setattr(dw, "execute_rebalance", _boom)

    state = PortfolioState(
        shares={"ABC": 10},
        entry_prices={"ABC": 100.0},
        last_known_prices={"ABC": 100.0},
        last_rebalance_date="2024-01-01",
    )

    out_state, _ = dw._run_scan(["ABC", "FRESH"], state, "TEST", cfg_override=UltimateConfig())

    assert called["n"] == 0
    assert out_state.shares == {"ABC": 10}


def test_run_scan_hard_cvar_breach_overrides_cadence_gate(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=6)
    md = {
        "ABC.NS": pd.DataFrame({"Close": [100] * 6, "Dividends": [0] * 6}, index=idx),
        "^NSEI": pd.DataFrame({"Close": [100] * 6}, index=idx),
        "^CRSLDX": pd.DataFrame({"Close": [100] * 6}, index=idx),
    }
    monkeypatch.setattr(dw, "load_or_fetch", lambda *_args, **_kwargs: md)
    monkeypatch.setattr(dw, "_print_stage_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "detect_and_apply_splits", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(dw, "compute_book_cvar", lambda *_args, **_kwargs: UltimateConfig().CVAR_DAILY_LIMIT * 2.0)

    called = {"n": 0}

    def _track(*args, **kwargs):
        called["n"] += 1
        return execute_rebalance(*args, **kwargs)

    monkeypatch.setattr(dw, "execute_rebalance", _track)

    state = PortfolioState(
        shares={"ABC": 10},
        entry_prices={"ABC": 100.0},
        last_known_prices={"ABC": 100.0},
        last_rebalance_date=datetime.today().strftime("%Y-%m-%d"),
    )
    cfg = UltimateConfig(REBALANCE_FREQ="W-FRI")

    out_state, _ = dw._run_scan(["ABC"], state, "TEST", cfg_override=cfg)

    assert called["n"] == 1
    assert out_state.shares == {}


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


def test_detect_and_apply_splits_handles_tz_aware_split_index():
    cfg = UltimateConfig(AUTO_ADJUST_PRICES=False)
    state = PortfolioState(
        shares={"ABC": 10},
        entry_prices={"ABC": 1000.0},
        last_known_prices={"ABC": 1000.0},
        cash=0.0,
        last_rebalance_date="2024-01-01",
    )
    idx = pd.date_range("2024-01-01", periods=3, tz="Asia/Kolkata")
    market_data = {
        "ABC.NS": pd.DataFrame(
            {
                "Close": [1000.0, 500.0, 505.0],
                "Dividends": [0.0, 0.0, 0.0],
                "Stock Splits": [0.0, 2.0, 0.0],
            },
            index=idx,
        )
    }

    adjusted = dw.detect_and_apply_splits(state, market_data, cfg)

    assert adjusted == ["ABC"]
    assert state.shares["ABC"] == 20


def test_detect_and_apply_splits_fractional_cash_applies_one_way_slippage():
    cfg = UltimateConfig(AUTO_ADJUST_PRICES=False, ROUND_TRIP_SLIPPAGE_BPS=10.0)
    state = PortfolioState(
        shares={"ABC": 3},
        entry_prices={"ABC": 100.0},
        last_known_prices={"ABC": 100.0},
        cash=0.0,
    )
    market_data = {
        "ABC.NS": pd.DataFrame(
            {
                "Close": [200.0],
                "Dividends": [0.0],
                "Stock Splits": [0.5],
            }
        )
    }

    adjusted = dw.detect_and_apply_splits(state, market_data, cfg)

    assert adjusted == ["ABC"]
    assert state.shares["ABC"] == 1
    assert state.cash == pytest.approx(99.95)
