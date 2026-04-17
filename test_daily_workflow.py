from __future__ import annotations
import osqp_preimport  # noqa: F401

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
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

from collections import namedtuple

# Define a mock Trade object for testing purposes
MockTrade = namedtuple("MockTrade", ["delta_shares", "exec_price", "direction", "symbol"])


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
    monkeypatch.setattr(dw, "_load_pending_sentinel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "_try_claim_pending_sentinel", lambda *_args, **_kwargs: True)
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
    monkeypatch.setattr(dw, "_load_pending_sentinel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "_try_claim_pending_sentinel", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(dw, "compute_book_cvar", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(dw, "compute_adv", lambda market_data, active, *args, **kwargs: __import__("numpy").array([1e9] * len(active)))
    monkeypatch.setattr(dw, "get_sector_map", lambda syms, cfg=None: {s: 0 for s in syms})
    monkeypatch.setattr("momentum_engine.execute_rebalance", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(dw, "_load_pending_sentinel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "_try_claim_pending_sentinel", lambda *_args, **_kwargs: True)
    def _fake_generate_signals(*_args, **_kwargs):
        import numpy as np
        # FIX-MB2-TESTGATENAMES: Use renamed keys from FIX-MB-GATENAMES.
        # "history_gated"/"adv_gated"/"knife_gated" → "history_failed"/"adv_failed"/"knife_failed"
        return np.array([0.01]), np.array([1.0]), [0], {
            "total": 1, "history_failed": 0, "adv_failed": 0, "knife_failed": 0, "selected": 1
        }

    monkeypatch.setattr("signals.generate_signals", _fake_generate_signals)

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


def test_run_scan_forward_fills_after_union_index_alignment(monkeypatch):
    fri = pd.Timestamp("2026-03-27")
    sat = pd.Timestamp("2026-03-28")
    idx_fri = pd.DatetimeIndex([fri])
    idx_weekend = pd.DatetimeIndex([fri, sat])
    md = {
        "AAA.NS": pd.DataFrame({"Close": [100.0], "Dividends": [0.0]}, index=idx_fri),
        "BBB.NS": pd.DataFrame({"Close": [200.0, np.nan], "Dividends": [0.0, 0.0]}, index=idx_weekend),
        "^NSEI": pd.DataFrame({"Close": [100.0]}, index=idx_fri),
        "^CRSLDX": pd.DataFrame({"Close": [100.0]}, index=idx_fri),
    }
    captured = {"prices": None, "active": None}

    monkeypatch.setattr(dw, "load_or_fetch", lambda *_args, **_kwargs: md)
    monkeypatch.setattr(dw, "_print_stage_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "detect_and_apply_splits", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(dw, "_load_pending_sentinel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "_try_claim_pending_sentinel", lambda *args, **kwargs: True)

    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(dw, "compute_book_cvar", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(dw, "compute_adv", lambda market_data, active, *args, **kwargs: np.array([1e9] * len(active)))
    monkeypatch.setattr(dw, "get_sector_map", lambda syms, cfg=None: {s: 0 for s in syms})

    def _fake_generate_signals(*_args, **_kwargs):
        return np.array([0.01, 0.02]), np.array([0.0, 0.0]), [0, 1], {
            "total": 2, "history_failed": 0, "adv_failed": 0, "knife_failed": 0, "selected": 2
        }

    monkeypatch.setattr("signals.generate_signals", _fake_generate_signals)

    class _Engine:
        def __init__(self, _cfg):
            pass

        def optimize(self, **_kwargs):
            return np.array([0.5, 0.5])

    monkeypatch.setattr(dw, "InstitutionalRiskEngine", _Engine)

    def _capture_rebalance(state, weights, prices, active, cfg, **kwargs):
        captured["prices"] = np.array(prices, dtype=float)
        captured["active"] = list(active)
        return 0.0

    monkeypatch.setattr("momentum_engine.execute_rebalance", _capture_rebalance)

    state = PortfolioState(cash=10_000.0)
    dw._run_scan(["AAA", "BBB"], state, "TEST", cfg_override=UltimateConfig())

    assert captured["active"] == ["AAA", "BBB"]
    assert captured["prices"] is not None
    assert np.all(np.isfinite(captured["prices"]))
    assert captured["prices"].tolist() == pytest.approx([100.0, 200.0])


def test_run_scan_excludes_symbol_with_long_price_gap_after_bounded_fill(monkeypatch):
    old_date = pd.Timestamp("2026-03-01")
    recent_idx = pd.date_range("2026-03-24", periods=5, freq="D")
    md = {
        "STALE.NS": pd.DataFrame({"Close": [100.0], "Dividends": [0.0]}, index=pd.DatetimeIndex([old_date])),
        "FRESH.NS": pd.DataFrame(
            {"Close": [200.0, 201.0, 202.0, 203.0, 204.0], "Dividends": [0.0] * 5},
            index=recent_idx,
        ),
        "^NSEI": pd.DataFrame({"Close": [100.0] * 5}, index=recent_idx),
        "^CRSLDX": pd.DataFrame({"Close": [100.0] * 5}, index=recent_idx),
    }
    captured = {"prices": None, "active": None}

    monkeypatch.setattr(dw, "load_or_fetch", lambda *_args, **_kwargs: md)
    monkeypatch.setattr(dw, "_print_stage_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "detect_and_apply_splits", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(dw, "_load_pending_sentinel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "_try_claim_pending_sentinel", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(dw, "compute_book_cvar", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(dw, "compute_adv", lambda *_args, **_kwargs: np.array([1e9]))
    monkeypatch.setattr(dw, "get_sector_map", lambda syms, cfg=None: {s: 0 for s in syms})
    def _fake_generate_signals(*_args, **_kwargs):
        return np.array([0.02]), np.array([0.0]), [0], {
            "total": 1, "history_failed": 0, "adv_failed": 0, "knife_failed": 0, "selected": 1
        }

    monkeypatch.setattr("signals.generate_signals", _fake_generate_signals)

    class _Engine:
        def __init__(self, _cfg):
            pass

        def optimize(self, **_kwargs):
            return np.array([1.0])

    monkeypatch.setattr(dw, "InstitutionalRiskEngine", _Engine)

    def _capture_rebalance(state, weights, prices, active, cfg, **kwargs):
        captured["prices"] = np.array(prices, dtype=float)
        captured["active"] = list(active)
        return 0.0

    monkeypatch.setattr("momentum_engine.execute_rebalance", _capture_rebalance)

    state = PortfolioState(cash=10_000.0)
    dw._run_scan(["STALE", "FRESH"], state, "TEST", cfg_override=UltimateConfig())

    assert captured["active"] == ["FRESH"]
    assert captured["prices"] is not None
    assert captured["prices"].tolist() == pytest.approx([204.0])


def test_print_status_uses_last_non_nan_close(capsys):
    state = PortfolioState(
        shares={"AAA": 10},
        entry_prices={"AAA": 90.0},
        last_known_prices={"AAA": 90.0},
        cash=0.0,
    )
    market_data = {
        "AAA.NS": pd.DataFrame(
            {"Close": [100.0, np.nan]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-03-27"), pd.Timestamp("2026-03-28")]),
        )
    }

    dw._print_status(state, "TEST", market_data, cfg=UltimateConfig())
    out = capsys.readouterr().out

    assert "nan" not in out.lower()
    assert "100.00" in out

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


def _split_state() -> PortfolioState:
    return PortfolioState(
        shares={"ABC": 10},
        entry_prices={"ABC": 1000.0},
        last_known_prices={"ABC": 1000.0},
        cash=0.0,
    )


def _split_market_data(split_series, close_series=None):
    idx = pd.date_range("2024-01-01", periods=len(split_series))
    closes = close_series or [1000.0, 500.0][: len(split_series)]
    return {
        "ABC.NS": pd.DataFrame(
            {
                "Close": closes,
                "Dividends": [0.0] * len(split_series),
                "Stock Splits": split_series,
            },
            index=idx,
        )
    }


def test_detect_and_apply_splits_applies_when_stock_splits_column_marks_event():
    cfg = UltimateConfig(AUTO_ADJUST_PRICES=True)
    state = _split_state()
    market_data = _split_market_data([0.0, 2.0], close_series=[1000.0, 500.0])

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
        dividend_ledger={"split:ABC": "2024-01-03:0.00000000"},
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

def test_detect_and_apply_splits_without_anchor_compounds_visible_split_events():
    cfg = UltimateConfig(AUTO_ADJUST_PRICES=True)
    state = _split_state()
    market_data = _split_market_data(
        [0.0, 2.0, 0.0, 2.0, 0.0],
        close_series=[1000.0, 500.0, 505.0, 252.5, 255.0],
    )

    adjusted = dw.detect_and_apply_splits(state, market_data, cfg)

    assert adjusted == ["ABC"]
    assert state.shares["ABC"] == 40
    assert state.entry_prices["ABC"] == 250.0


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
    monkeypatch.setattr(dw, "_load_pending_sentinel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "_try_claim_pending_sentinel", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)

    called = {"n": 0}

    def _boom(*_args, **_kwargs):
        called["n"] += 1
        raise AssertionError("execute_rebalance should not be called when cadence gate blocks")

    monkeypatch.setattr("momentum_engine.execute_rebalance", _boom)
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
    monkeypatch.setattr(dw, "_load_pending_sentinel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "_try_claim_pending_sentinel", lambda *_args, **_kwargs: True)
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
    monkeypatch.setattr(dw, "_load_pending_sentinel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "_try_claim_pending_sentinel", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(dw, "compute_book_cvar", lambda *_args, **_kwargs: UltimateConfig().CVAR_DAILY_LIMIT + 0.01)
    monkeypatch.setattr(dw, "compute_adv", lambda market_data, active, *args, **kwargs: __import__("numpy").array([1e9] * len(active)))
    monkeypatch.setattr(dw, "get_sector_map", lambda syms, cfg=None: {s: 0 for s in syms})

    def _fake_generate_signals(*_args, **_kwargs):
        import numpy as np
        return np.array([0.01, 0.0]), np.array([1.0, 0.0]), [0], {
            "total": 2, "history_failed": 0, "adv_failed": 0, "knife_failed": 1, "selected": 1
        }

    monkeypatch.setattr("signals.generate_signals", _fake_generate_signals)

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

    monkeypatch.setattr("momentum_engine.execute_rebalance", _boom)

    state = PortfolioState(
        shares={"ABC": 10},
        entry_prices={"ABC": 100.0},
        last_known_prices={"ABC": 100.0},
        last_rebalance_date="2024-01-01",
        consecutive_failures=2,
    )

    out_state, _ = dw._run_scan(["ABC", "FRESH"], state, "TEST", cfg_override=UltimateConfig())

    assert called["n"] == 0
    assert out_state.shares == {"ABC": 10}


def test_run_scan_cadence_stale_gate_does_not_emit_duplicate_rebalance_warning(monkeypatch, caplog):
    idx = pd.date_range("2024-01-01", periods=6)
    stale_idx = pd.date_range("2024-01-01", periods=3)
    md = {
        "ABC.NS": pd.DataFrame(
            {"Close": [100.0] * len(stale_idx), "Dividends": [0.0] * len(stale_idx)},
            index=stale_idx,
        ),
        "FRESH.NS": pd.DataFrame(
            {"Close": [100.0] * len(idx), "Dividends": [0.0] * len(idx)},
            index=idx,
        ),
        "^NSEI": pd.DataFrame({"Close": [100.0] * len(idx)}, index=idx),
        "^CRSLDX": pd.DataFrame({"Close": [100.0] * len(idx)}, index=idx),
    }
    monkeypatch.setattr(dw, "load_or_fetch", lambda *_args, **_kwargs: md)
    monkeypatch.setattr(dw, "_print_stage_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "detect_and_apply_splits", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(dw, "_load_pending_sentinel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "_try_claim_pending_sentinel", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr("daily_workflow.compute_book_cvar", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(dw, "compute_adv", lambda market_data, active, *args, **kwargs: __import__("numpy").array([1e9] * len(active)))
    monkeypatch.setattr(dw, "get_sector_map", lambda syms, cfg=None: {s: 0 for s in syms})

    def _fake_generate_signals(*_args, **_kwargs):
        import numpy as np

        return np.array([0.01, 0.0]), np.array([1.0, 0.0]), [0], {
            "total": 2,
            "history_failed": 0,
            "adv_failed": 0,
            "knife_failed": 1,
            "selected": 1,
        }

    monkeypatch.setattr("signals.generate_signals", _fake_generate_signals)

    class _Engine:
        def __init__(self, _cfg):
            pass

        def optimize(self, **_kwargs):
            import numpy as np

            return np.array([1.0])

    monkeypatch.setattr(dw, "InstitutionalRiskEngine", _Engine)
    monkeypatch.setattr(
        "momentum_engine.execute_rebalance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("execute_rebalance should not be called when cadence staleness gate fires")
        ),
    )

    state = PortfolioState(
        shares={"ABC": 10},
        entry_prices={"ABC": 100.0},
        last_known_prices={"ABC": 100.0},
        last_rebalance_date="2024-01-01",
    )

    with caplog.at_level("WARNING"):
        out_state, _ = dw._run_scan(["ABC", "FRESH"], state, "TEST", cfg_override=UltimateConfig())

    assert out_state.shares == {"ABC": 10}
    assert sum("stale price data" in record.message for record in caplog.records) == 1
    assert sum("REBALANCE SUPPRESSED" in record.message for record in caplog.records) == 0


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
    monkeypatch.setattr(dw, "_load_pending_sentinel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "_try_claim_pending_sentinel", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(dw, "get_sector_map", lambda syms, cfg=None: {s: 0 for s in syms})
    monkeypatch.setattr(dw, "compute_book_cvar", lambda *_args, **_kwargs: UltimateConfig().CVAR_DAILY_LIMIT * 2.0)
    monkeypatch.setattr("momentum_engine.compute_book_cvar", lambda *_args, **_kwargs: UltimateConfig().CVAR_DAILY_LIMIT * 2.0)

    called = {"n": 0}

    def _track(*args, **kwargs):
        called["n"] += 1
        return execute_rebalance(*args, **kwargs)

    monkeypatch.setattr("momentum_engine.execute_rebalance", _track)

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


def test_pending_sentinel_helpers_roundtrip(tmp_path: Path, monkeypatch):
    import hashlib

    monkeypatch.chdir(tmp_path)
    token = "tok-1"
    date_str = "2026-03-30"
    path = dw._write_pending_sentinel("nifty", token, date_str)
    assert path.exists()
    payload = dw._load_pending_sentinel("nifty")
    assert payload is not None
    assert payload["date"] == date_str
    assert payload["token_hash"] == hashlib.sha256(token.encode("utf-8")).hexdigest()
    dw._clear_pending_sentinel("nifty")
    assert dw._load_pending_sentinel("nifty") is None


def test_circuit_breaker_concurrent_increment_and_reset():
    breaker = dw.CircuitBreaker()

    def _worker():
        for _ in range(200):
            breaker.increment()

    threads = [threading.Thread(target=_worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert breaker.count == 1000
    breaker.reset()
    assert breaker.count == 0


def test_circuit_breaker_save_logs_error_and_load_lock_fallback(tmp_path: Path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    breaker = dw.CircuitBreaker(count=3)
    monkeypatch.setattr(dw.os, "replace", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk fail")))
    with caplog.at_level(logging.ERROR):
        breaker.save("data/circuit_breaker.json")
    assert "Failed to persist circuit breaker count" in caplog.text
    assert Path("data/circuit_breaker.lock").exists()

    loader = dw.CircuitBreaker()
    loader.load("data/circuit_breaker.json")
    assert loader.count == 1


def test_run_scan_skips_rebalance_when_pending_sentinel_exists_for_today(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=6)
    md = {
        "ABC.NS": pd.DataFrame({"Close": [100.0] * 6, "Dividends": [0.0] * 6}, index=idx),
        "^NSEI": pd.DataFrame({"Close": [100.0] * 6}, index=idx),
        "^CRSLDX": pd.DataFrame({"Close": [100.0] * 6}, index=idx),
    }
    monkeypatch.setattr(dw, "load_or_fetch", lambda *_args, **_kwargs: md)
    monkeypatch.setattr(dw, "_print_stage_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "detect_and_apply_splits", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(dw, "_load_pending_sentinel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dw, "_try_claim_pending_sentinel", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(dw, "compute_regime_score", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(dw, "compute_book_cvar", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(dw, "compute_adv", lambda *_args, **_kwargs: np.array([1e9]))
    monkeypatch.setattr(dw, "get_sector_map", lambda syms, cfg=None: {s: 0 for s in syms})
    monkeypatch.setattr("signals.generate_signals", lambda *_a, **_k: (np.array([0.01]), np.array([0.0]), [0], {"total": 1, "history_failed": 0, "adv_failed": 0, "knife_failed": 0, "selected": 1}))
    monkeypatch.setattr(dw, "_load_pending_sentinel", lambda name: {"token": "x", "date": datetime.today().strftime("%Y-%m-%d")})

    called = {"rebalance": 0}

    class _Engine:
        def __init__(self, _cfg):
            pass

        def optimize(self, **_kwargs):
            return np.array([1.0])

    monkeypatch.setattr(dw, "InstitutionalRiskEngine", _Engine)

    def _count_rebalance(*_args, **_kwargs):
        called["rebalance"] += 1
        return 0.0

    monkeypatch.setattr("momentum_engine.execute_rebalance", _count_rebalance)
    state = PortfolioState(cash=10_000.0)
    dw._run_scan(["ABC"], state, "TEST", cfg_override=UltimateConfig(), name="nifty")
    assert called["rebalance"] == 0


def test_save_portfolio_state_clears_pending_sentinel_in_paper_mode(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dw, "PAPER_MODE", True)
    state = PortfolioState()
    dw._write_pending_sentinel("nifty", "tok", "2026-03-30")
    assert dw._load_pending_sentinel("nifty") is not None
    dw.save_portfolio_state(state, "nifty")
    assert dw._load_pending_sentinel("nifty") is None


class TestDailyWorkflow:
    @pytest.fixture(autouse=True)
    def cd_tmp_path(self, tmp_path: Path, monkeypatch):
        """Ensure all tests run in a clean temporary directory."""
        monkeypatch.chdir(tmp_path)
        # Mock the circuit breaker to prevent file I/O during tests
        monkeypatch.setattr(dw, "_circuit_breaker", dw.CircuitBreaker())


    @pytest.fixture
    def mock_dependencies(self, monkeypatch):
        """Central fixture for mocking external and slow dependencies."""
        monkeypatch.setattr(dw, "_print_stage_status", lambda *args, **kwargs: None)
        monkeypatch.setattr(dw, "load_optimized_config", UltimateConfig)
        monkeypatch.setattr(dw, "get_sector_map", lambda syms, cfg=None: {s: 0 for s in syms})
        monkeypatch.setattr(dw, "fetch_nse_equity_universe", lambda: ["RELIANCE", "TCS"])
        monkeypatch.setattr(dw, "_check_and_prompt_initial_capital", lambda a, b, c: None)
        monkeypatch.setattr(dw, "_try_claim_pending_sentinel", lambda *args, **kwargs: True)

        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        mock_market_data = {
            "RELIANCE.NS": pd.DataFrame({"Close": np.linspace(2800, 2900, 10), "Open": np.linspace(2800, 2900, 10), "Dividends": 0, "Stock Splits": 0}, index=idx),
            "TCS.NS": pd.DataFrame({"Close": np.linspace(3800, 3900, 10), "Open": np.linspace(3800, 3900, 10), "Dividends": 0, "Stock Splits": 0}, index=idx),
            "^NSEI": pd.DataFrame({"Close": np.linspace(21000, 22000, 10)}, index=idx),
            "^CRSLDX": pd.DataFrame({"Close": np.linspace(15000, 16000, 10)}, index=idx),
        }
        monkeypatch.setattr(dw, "load_or_fetch", lambda *args, **kwargs: mock_market_data)
        
        from momentum_engine import RebalancePipelineResult, RebalanceContext
        
        call_log = {}
        def mock_rebalance_pipeline(ctx: RebalanceContext):
            call_log['ctx'] = ctx
            # Simulate a rebalance by updating weights and shares
            ctx.state.weights = {"RELIANCE": 0.5, "TCS": 0.5}
            ctx.state.shares = {"RELIANCE": 17, "TCS": 13}
            ctx.state.cash = 1000.0
            if ctx.trade_log is not None:
                ctx.trade_log.append(cast(Any, MockTrade(delta_shares=1, exec_price=100, direction="BUY", symbol="RELIANCE")))
            return RebalancePipelineResult(
                optimization_succeeded=True, 
                applied_decay=False,
                total_slippage=0.0,
                soft_cvar_breach=False,
                target_weights=np.array([0.5, 0.5])
            )

        monkeypatch.setattr(dw, "run_rebalance_pipeline", mock_rebalance_pipeline)
        return call_log


    def test_run_scan_integration_with_user_confirmation(self, mock_dependencies, monkeypatch):
        """
        Tests the full `_run_scan` flow followed by `_preview_scan_and_maybe_save`
        where the user confirms the rebalance.
        """
        # Arrange
        states = {"nse_total": PortfolioState(cash=1_000_000.0)}
        mkt_cache = {}
        
        # Mock user input to confirm the save
        monkeypatch.setattr("builtins.input", lambda prompt: "y")
        
        # Mock save_portfolio_state to track calls
        save_calls = []
        def mock_save(state, name):
            save_calls.append((state, name))
        monkeypatch.setattr(dw, "save_portfolio_state", mock_save)

        # Act
        dw._handle_nse_total_scan(states, mkt_cache)

        # Assert
        # Check that the state was updated
        final_state = states["nse_total"]
        assert final_state.shares == {"RELIANCE": 17, "TCS": 13}
        assert final_state.weights == {"RELIANCE": 0.5, "TCS": 0.5}
        assert final_state.cash == 1000.0
        
        # Check that save was called correctly
        assert len(save_calls) == 1
        saved_state, saved_name = save_calls[0]
        assert saved_name == "nse_total"
        assert saved_state.shares == {"RELIANCE": 17, "TCS": 13}
        
        # Check that market cache is populated
        assert "nse_total" in mkt_cache
        assert "RELIANCE.NS" in mkt_cache["nse_total"]

        # Check that run_rebalance_pipeline was called with correct context
        assert "ctx" in mock_dependencies
        ctx = mock_dependencies['ctx']
        assert ctx.cfg is not None
        assert ctx.pv == pytest.approx(1_000_000.0)


    def test_run_scan_integration_with_user_rejection(self, mock_dependencies, monkeypatch):
        """
        Tests the full `_run_scan` flow followed by `_preview_scan_and_maybe_save`
        where the user rejects the rebalance.
        """
        # Arrange
        initial_state = PortfolioState(cash=1_000_000.0)
        states = {"nse_total": initial_state}
        mkt_cache = {}
        
        # Mock user input to reject the save
        monkeypatch.setattr("builtins.input", lambda prompt: "n")
        
        save_calls = []
        def mock_save(state, name):
            save_calls.append((state, name))
        monkeypatch.setattr(dw, "save_portfolio_state", mock_save)

        # Act
        dw._handle_nse_total_scan(states, mkt_cache)

        # Assert
        # Check that the state was NOT updated with rebalance results
        final_state = states["nse_total"]
        assert final_state.shares == {}
        assert final_state.weights == {}
        assert final_state.cash == 1_000_000.0
        
        # Check that save was still called to persist risk metadata
        assert len(save_calls) == 1
        saved_state, saved_name = save_calls[0]
        assert saved_name == "nse_total"
        # The saved state should be the original state, with potentially updated risk metadata
        assert saved_state.shares == {} 
        assert saved_state.cash == 1_000_000.0
        assert saved_state is final_state

        # Check that market cache is populated even on rejection
        assert "nse_total" in mkt_cache
        assert "RELIANCE.NS" in mkt_cache["nse_total"]
