import importlib
import json
from pathlib import Path

import pytest

optuna = pytest.importorskip("optuna")
optimizer = pytest.importorskip("optimizer")


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
        metrics = {"cagr": 0.0, "max_dd": 0.0}

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


def test_objective_prunes_trial_when_drawdown_exceeds_cap(monkeypatch):
    class _Result:
        metrics = {"cagr": 10.0, "max_dd": 30.0}

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

    with pytest.raises(optuna.TrialPruned):
        objective(trial)


def test_save_optimal_config_replaces_existing_file_atomically(tmp_path: Path):
    output_path = tmp_path / "optimal_cfg.json"
    output_path.write_text('{"old": 1}', encoding="utf-8")

    optimizer.save_optimal_config({"HALFLIFE_FAST": 34}, str(output_path))

    with output_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    assert payload == {"HALFLIFE_FAST": 34}
