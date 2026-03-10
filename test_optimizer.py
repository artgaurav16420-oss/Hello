import importlib
import json
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

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


def test_optimizer_excludes_insufficient_history_symbols_without_overweight():
    cfg = UltimateConfig()
    cfg.HISTORY_GATE = 60
    cfg.DIMENSIONALITY_MULTIPLIER = 1

    engine = InstitutionalRiskEngine(cfg)

    idx = pd.date_range("2023-01-02", periods=80, freq="B")
    rng = np.random.default_rng(123)

    mature_a = rng.normal(0.0010, 0.01, len(idx))
    mature_b = rng.normal(0.0008, 0.01, len(idx))
    ipo_sparse = np.full(len(idx), np.nan)
    ipo_sparse[-12:] = rng.normal(0.0015, 0.01, 12)

    historical_returns = pd.DataFrame(
        {"MATURE_A": mature_a, "IPO_NEW": ipo_sparse, "MATURE_B": mature_b},
        index=idx,
    )

    expected_returns = np.array([0.010, 0.030, 0.011])
    prices = np.array([100.0, 150.0, 120.0])
    adv_shares = np.array([2e8, 2e8, 2e8])
    prev_w = np.array([0.20, 0.10, 0.20])
    sector_labels = np.array([0, 1, 0])

    weights = engine.optimize(
        expected_returns=expected_returns,
        historical_returns=historical_returns,
        adv_shares=adv_shares,
        prices=prices,
        portfolio_value=1_000_000.0,
        prev_w=prev_w,
        exposure_multiplier=1.0,
        sector_labels=sector_labels,
    )

    assert weights.shape == (3,)
    assert weights[1] == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(weights).all()
    assert float(np.sum(weights)) > 0.0


def test_optimizer_logs_insufficient_history_exclusions(caplog):
    cfg = UltimateConfig()
    cfg.HISTORY_GATE = 40
    cfg.DIMENSIONALITY_MULTIPLIER = 1

    engine = InstitutionalRiskEngine(cfg)

    idx = pd.date_range("2023-01-02", periods=50, freq="B")
    stable = np.linspace(-0.01, 0.01, len(idx))
    sparse = np.full(len(idx), np.nan)
    sparse[-5:] = np.linspace(-0.005, 0.005, 5)

    historical_returns = pd.DataFrame(
        {"STABLE": stable, "SPARSE": sparse},
        index=idx,
    )

    with caplog.at_level("INFO", logger="momentum_engine"):
        weights = engine.optimize(
            expected_returns=np.array([0.01, 0.02]),
            historical_returns=historical_returns,
            adv_shares=np.array([1e8, 1e8]),
            prices=np.array([100.0, 100.0]),
            portfolio_value=1_000_000.0,
        )

    assert weights.shape == (2,)
    assert weights[1] == pytest.approx(0.0, abs=1e-12)
    assert "reason=insufficient_history" in caplog.text
