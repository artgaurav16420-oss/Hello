import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")
optimizer = pytest.importorskip("optimizer")

from momentum_engine import InstitutionalRiskEngine, UltimateConfig


def test_save_optimal_config_allows_plain_filename(tmp_path: Path):
    output_path = tmp_path / "optimal_cfg.json"

    optimizer.save_optimal_config({"HALFLIFE_FAST": 21}, str(output_path))

    with output_path.open("r") as fh:
        payload = json.load(fh)

    assert payload["HALFLIFE_FAST"] == 21


def test_run_optimization_raises_when_no_completed_trials(monkeypatch):
    monkeypatch.setattr(optimizer, "N_TRIALS", 1)
    monkeypatch.setattr(optimizer, "pre_load_data", lambda universe_type: {})
    monkeypatch.setattr(
        optimizer,
        "run_backtest",
        lambda **kwargs: (_ for _ in ()).throw(optimizer.OptimizationError("boom")),
    )

    with pytest.raises(RuntimeError, match="no completed trials"):
        optimizer.run_optimization()


def test_optimizer_logger_does_not_duplicate_handlers_on_reload():
    before = len(optimizer.logger.handlers)
    importlib.reload(optimizer)
    after = len(optimizer.logger.handlers)

    assert after == before


def test_objective_returns_zero_when_max_drawdown_is_zero(monkeypatch):
    class _Result:
        metrics = {"cagr": 0.0, "max_dd": 0.0, "turnover": 0.0}
        rebal_log = None

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    objective = optimizer.MomentumObjective(market_data={}, universe_type="nifty500")
    trial = optuna.trial.FixedTrial(
        {
            "HALFLIFE_FAST": 21,
            "HALFLIFE_SLOW": 63,
            "CONTINUITY_BONUS": 0.15,
            "RISK_AVERSION": 5.0,
            "CVAR_DAILY_LIMIT": 0.04,
        }
    )

    assert objective(trial) == 0.0


def test_objective_prunes_trial_on_optimization_error(monkeypatch):
    monkeypatch.setattr(
        optimizer,
        "run_backtest",
        lambda **kwargs: (_ for _ in ()).throw(
            optimizer.OptimizationError("Solver failed")
        ),
    )

    objective = optimizer.MomentumObjective(market_data={}, universe_type="nifty500")
    trial = optuna.trial.FixedTrial(
        {
            "HALFLIFE_FAST": 21,
            "HALFLIFE_SLOW": 63,
            "CONTINUITY_BONUS": 0.15,
            "RISK_AVERSION": 5.0,
            "CVAR_DAILY_LIMIT": 0.04,
        }
    )

    with pytest.raises(optuna.TrialPruned):
        objective(trial)


def test_objective_logs_unexpected_errors_as_warning_and_prunes(monkeypatch, caplog):
    monkeypatch.setattr(
        optimizer,
        "run_backtest",
        lambda **kwargs: (_ for _ in ()).throw(TypeError("bad type")),
    )

    objective = optimizer.MomentumObjective(market_data={}, universe_type="nifty500")
    trial = optuna.trial.FixedTrial(
        {
            "HALFLIFE_FAST": 21,
            "HALFLIFE_SLOW": 63,
            "CONTINUITY_BONUS": 0.15,
            "RISK_AVERSION": 5.0,
            "CVAR_DAILY_LIMIT": 0.04,
        }
    )

    with caplog.at_level("WARNING", logger="Optimizer"):
        with pytest.raises(optuna.TrialPruned):
            objective(trial)

    assert "Trial failed due to internal error" in caplog.text


def test_objective_returns_numeric_score_without_hard_drawdown_prune(monkeypatch):
    class _Result:
        metrics = {"cagr": 10.0, "max_dd": 30.0, "turnover": 0.0}
        rebal_log = None

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    objective = optimizer.MomentumObjective(market_data={}, universe_type="nifty500")
    trial = optuna.trial.FixedTrial(
        {
            "HALFLIFE_FAST": 21,
            "HALFLIFE_SLOW": 63,
            "CONTINUITY_BONUS": 0.15,
            "RISK_AVERSION": 5.0,
            "CVAR_DAILY_LIMIT": 0.04,
        }
    )

    assert isinstance(objective(trial), float)


def test_save_optimal_config_replaces_existing_file_atomically(tmp_path: Path):
    output_path = tmp_path / "optimal_cfg.json"
    output_path.write_text('{"old": 1}', encoding="utf-8")

    optimizer.save_optimal_config({"HALFLIFE_FAST": 34}, str(output_path))

    with output_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    assert payload == {"HALFLIFE_FAST": 34}


def test_pre_load_data_deduplicates_inputs_and_appends_crsldx_index(monkeypatch):
    monkeypatch.setattr(optimizer, "TRAIN_START", "2020-01-01")
    monkeypatch.setattr(optimizer, "TEST_END", "2020-12-31")
    monkeypatch.setattr(optimizer, "fetch_nse_equity_universe", lambda: ["ABC", "^NSEI", "ABC"])

    captured = {}

    def _fake_load_or_fetch(*, tickers, required_start, required_end):
        captured["tickers"] = tickers
        captured["required_start"] = required_start
        captured["required_end"] = required_end
        return {"ok": True}

    monkeypatch.setattr(optimizer, "load_or_fetch", _fake_load_or_fetch)

    result = optimizer.pre_load_data("  NSE_TOTAL ")

    assert result == {"ok": True}
    assert captured["required_start"] == "2020-01-01"
    assert captured["required_end"] == "2020-12-31"
    assert captured["tickers"].count("ABC") == 1
    assert "^NSEI" in captured["tickers"]
    assert "^CRSLDX" in captured["tickers"]


def test_build_sampler_returns_tpe_sampler(monkeypatch):
    monkeypatch.setattr(optimizer, "OPTUNA_SEED", None)
    sampler_unseeded = optimizer._build_sampler()

    monkeypatch.setattr(optimizer, "OPTUNA_SEED", "123")
    sampler_seeded = optimizer._build_sampler()

    assert isinstance(sampler_unseeded, optimizer.TPESampler)
    assert isinstance(sampler_seeded, optimizer.TPESampler)




def test_objective_suggests_cvar_lookback_for_non_fixed_trials(monkeypatch):
    class _Result:
        metrics = {"cagr": 10.0, "max_dd": 10.0, "turnover": 0.0}
        rebal_log = None

    class _DummyTrial:
        params = {}

        def __init__(self):
            self.suggested = []

        def suggest_int(self, name, low, high, step=1):
            self.suggested.append(name)
            return low

        def suggest_float(self, name, low, high, step=None):
            self.suggested.append(name)
            return low

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    objective = optimizer.MomentumObjective(market_data={}, universe_type="nifty500")
    trial = _DummyTrial()

    objective(trial)

    assert "CVAR_LOOKBACK" in trial.suggested


def test_objective_uses_configurable_search_space(monkeypatch):
    class _Result:
        metrics = {"cagr": 10.0, "max_dd": 5.0, "turnover": 0.0}
        rebal_log = None

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    custom_space = {
        "HALFLIFE_FAST": (15, 15),
        "HALFLIFE_SLOW": (70, 70),
        "CONTINUITY_BONUS": (0.1, 0.1, 0.01),
        "RISK_AVERSION": (3.0, 3.0, 0.5),
        "CVAR_DAILY_LIMIT": (0.03, 0.03, 0.005),
    }
    objective = optimizer.MomentumObjective(
        market_data={}, universe_type="nifty500", search_space=custom_space
    )
    trial = optuna.trial.FixedTrial(
        {
            "HALFLIFE_FAST": 15,
            "HALFLIFE_SLOW": 70,
            "CONTINUITY_BONUS": 0.1,
            "RISK_AVERSION": 3.0,
            "CVAR_DAILY_LIMIT": 0.03,
        }
    )

    assert round(objective(trial), 6) == round((10.0 / 6.0) - 0.5, 6)


def test_run_optimization_passes_parallel_jobs_to_optuna(monkeypatch):
    monkeypatch.setattr(optimizer, "N_TRIALS", 1)
    monkeypatch.setattr(optimizer, "N_JOBS", 3)
    monkeypatch.setattr(optimizer, "pre_load_data", lambda universe_type: {})
    monkeypatch.setattr(optimizer, "save_optimal_config", lambda best_params: None)

    class _Result:
        metrics = {"final": 1.0, "cagr": 1.0, "max_dd": 1.0, "calmar": 1.1}

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    captured = {}

    class _Study:
        best_trials = [object()]
        best_params = {
            "HALFLIFE_FAST": 21,
            "HALFLIFE_SLOW": 63,
            "CONTINUITY_BONUS": 0.15,
            "RISK_AVERSION": 5.0,
            "CVAR_DAILY_LIMIT": 0.04,
        }
        best_value = 1.23

        def optimize(self, objective, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(optimizer.optuna, "create_study", lambda **kwargs: _Study())

    optimizer.run_optimization()

    assert captured["n_jobs"] == 3


def test_run_optimization_uses_selected_universe(monkeypatch):
    monkeypatch.setattr(optimizer, "N_TRIALS", 1)
    monkeypatch.setattr(optimizer, "save_optimal_config", lambda best_params: None)

    captured = {}

    def _fake_pre_load_data(universe_type):
        captured["pre_load_universe"] = universe_type
        return {}

    monkeypatch.setattr(optimizer, "pre_load_data", _fake_pre_load_data)

    class _Result:
        metrics = {"final": 1.0, "cagr": 1.0, "max_dd": 1.0, "calmar": 1.1}

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    class _Study:
        best_trials = [object()]
        best_params = {
            "HALFLIFE_FAST": 21,
            "HALFLIFE_SLOW": 63,
            "CONTINUITY_BONUS": 0.15,
            "RISK_AVERSION": 5.0,
            "CVAR_DAILY_LIMIT": 0.04,
        }
        best_value = 1.23

        def optimize(self, objective, **kwargs):
            pass

    monkeypatch.setattr(optimizer.optuna, "create_study", lambda **kwargs: _Study())

    optimizer.run_optimization(universe_type="nse_total")

    assert captured["pre_load_universe"] == "nse_total"


def test_parse_args_accepts_universe_override():
    args = optimizer._parse_args(["--universe", "nse_total"])

    assert args.universe == "nse_total"


def test_parse_args_in_memory_flag_defaults_false():
    args = optimizer._parse_args([])
    assert args.in_memory is False


def test_parse_args_in_memory_flag_sets_true():
    args = optimizer._parse_args(["--in-memory"])
    assert args.in_memory is True


def test_run_optimization_in_memory_uses_memory_storage_and_uncapped_n_jobs(monkeypatch):
    """
    --in-memory must route to :memory: storage and not apply the SQLite n_jobs=1 cap,
    regardless of whether OPTUNA_N_JOBS is set in the environment.
    """
    monkeypatch.setattr(optimizer, "N_TRIALS", 1)
    monkeypatch.setattr(optimizer, "pre_load_data", lambda universe_type: {})
    monkeypatch.setattr(optimizer, "save_optimal_config", lambda best_params: None)

    class _Result:
        metrics = {"final": 1.0, "cagr": 1.0, "max_dd": 1.0, "calmar": 1.1}

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    captured = {}

    class _Study:
        best_trials = [object()]
        best_params = {
            "HALFLIFE_FAST": 21,
            "HALFLIFE_SLOW": 63,
            "CONTINUITY_BONUS": 0.15,
            "RISK_AVERSION": 5.0,
            "CVAR_DAILY_LIMIT": 0.04,
        }
        best_value = 1.23

        def optimize(self, objective, **kwargs):
            captured.update(kwargs)

    def _fake_create_study(**kwargs):
        captured["storage"] = kwargs.get("storage")
        return _Study()

    monkeypatch.setattr(optimizer.optuna, "create_study", _fake_create_study)
    # Ensure OPTUNA_N_JOBS is absent so the default -1 path is exercised
    monkeypatch.delenv("OPTUNA_N_JOBS", raising=False)

    optimizer.run_optimization(in_memory=True)

    assert captured["storage"] == ":memory:", (
        f"in_memory=True must use ':memory:' storage, got: {captured['storage']!r}"
    )


def test_objective_allows_equal_halflife_values(monkeypatch):
    class _Result:
        metrics = {"cagr": 10.0, "max_dd": 5.0, "turnover": 0.0}
        rebal_log = None

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    objective = optimizer.MomentumObjective(market_data={}, universe_type="nifty500")
    trial = optuna.trial.FixedTrial(
        {
            "HALFLIFE_FAST": 21,
            "HALFLIFE_SLOW": 21,
            "CONTINUITY_BONUS": 0.15,
            "RISK_AVERSION": 5.0,
            "CVAR_DAILY_LIMIT": 0.04,
        }
    )

    assert objective(trial) != 0.0


def test_stdout_supports_rupee_with_utf8_stream():
    class _Stream:
        encoding = "utf-8"
        errors = "strict"

    assert optimizer._stdout_supports_rupee(_Stream()) is True


def test_stdout_supports_rupee_falls_back_for_cp1252_stream():
    class _Stream:
        encoding = "cp1252"
        errors = "strict"

    assert optimizer._stdout_supports_rupee(_Stream()) is False


def test_stdout_supports_rupee_false_when_stdout_missing(monkeypatch):
    monkeypatch.setattr(optimizer.sys, "stdout", None)

    assert optimizer._stdout_supports_rupee() is False


def test_optimizer_uses_higher_turnover_penalty_for_illiquid_name(monkeypatch):
    cfg = UltimateConfig()
    engine = InstitutionalRiskEngine(cfg)

    captured = {}

    class _FakeResInfo:
        status = "solved"

    class _FakeRes:
        def __init__(self, n_vars):
            self.info = _FakeResInfo()
            self.x = np.zeros(n_vars, dtype=float)

    class _FakeOSQP:
        def setup(self, P, q, A, l, u, **kwargs):
            captured["q"] = np.array(q, dtype=float)
            captured["n_vars"] = len(q)

        def solve(self):
            return _FakeRes(captured["n_vars"])

    monkeypatch.setattr("momentum_engine.osqp.OSQP", _FakeOSQP)

    expected_returns = np.array([0.01, 0.01], dtype=float)
    prices = np.array([100.0, 100.0], dtype=float)
    adv_shares = np.array([1e9, 1e5], dtype=float)
    hist = pd.DataFrame(
        np.array([
            [0.0010, 0.0015],
            [0.0005, -0.0002],
            [-0.0003, 0.0001],
            [0.0008, -0.0004],
            [0.0001, 0.0002],
            [0.0004, -0.0001],
            [0.0006, 0.0003],
            [-0.0002, 0.0005],
            [0.0003, -0.0002],
            [0.0007, 0.0004],
        ]),
        columns=["LIQUID", "ILLIQUID"],
    )

    engine.optimize(
        expected_returns=expected_returns,
        historical_returns=hist,
        adv_shares=adv_shares,
        prices=prices,
        portfolio_value=1_000_000.0,
        prev_w=np.array([0.0, 0.0], dtype=float),
    )

    turnover_q = captured["q"][2:4]
    assert turnover_q[1] > turnover_q[0]


def test_optimizer_turnover_penalty_respects_execution_floor_and_cap(monkeypatch):
    cfg = UltimateConfig(IMPACT_COEFF=1.0, ROUND_TRIP_SLIPPAGE_BPS=20.0)
    engine = InstitutionalRiskEngine(cfg)

    captured = {}

    class _FakeResInfo:
        status = "solved"

    class _FakeRes:
        def __init__(self, n_vars):
            self.info = _FakeResInfo()
            self.x = np.zeros(n_vars, dtype=float)

    class _FakeOSQP:
        def setup(self, P, q, A, l, u, **kwargs):
            captured["q"] = np.array(q, dtype=float)
            captured["n_vars"] = len(q)

        def solve(self):
            return _FakeRes(captured["n_vars"])

    monkeypatch.setattr("momentum_engine.osqp.OSQP", _FakeOSQP)

    expected_returns = np.array([0.01, 0.01], dtype=float)
    prices = np.array([100.0, 100.0], dtype=float)
    adv_shares = np.array([1e12, 1.0], dtype=float)
    hist = pd.DataFrame(np.tile([0.001, -0.001], (10, 1)), columns=["A", "B"])

    engine.optimize(
        expected_returns=expected_returns,
        historical_returns=hist,
        adv_shares=adv_shares,
        prices=prices,
        portfolio_value=1_000_000.0,
        prev_w=np.array([0.0, 0.0], dtype=float),
    )

    turnover_q = captured["q"][2:4]
    base_one_way = cfg.ROUND_TRIP_SLIPPAGE_BPS / 20_000.0
    assert turnover_q[0] == pytest.approx(base_one_way)
    assert turnover_q[1] == pytest.approx(0.05)
