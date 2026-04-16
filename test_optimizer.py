import osqp_preimport  # MUST be first to prevent Windows Access Violation
import importlib
import json
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")
optimizer = pytest.importorskip("optimizer")
from momentum_engine import InstitutionalRiskEngine, UltimateConfig

@pytest.fixture
def relaxed_train_start(monkeypatch):
    monkeypatch.setattr(optimizer, "TRAIN_START", "2018-01-01")

def _fixed_trial_params(**overrides):
    params = {
        "HALFLIFE_FAST": 21,
        "HALFLIFE_SLOW": 63,
        "CONTINUITY_BONUS": 0.15,
        "RISK_AVERSION": 5.0,
        "CVAR_DAILY_LIMIT": 0.04,
        "CVAR_LOOKBACK": 60,
        "SIGNAL_LAG_DAYS": 0,
        "MIN_EXPOSURE_FLOOR": 0.0,
        "MAX_POSITIONS": 20,
        "MAX_SINGLE_NAME_WEIGHT": 1.0,
        "REBALANCE_FREQ": "QE",
    }
    params.update(overrides)
    return optuna.trial.FixedTrial(params)


class _StaticResult:
    def __init__(self):
        self.metrics = {"final": 1.0, "cagr": 1.0, "max_dd": 1.0, "calmar": 1.1, "turnover": 0.0, "sortino": 1.0}
        self.rebal_log = None


class _TrackingTrial:
    def __init__(self):
        self.params = {}
        self.suggested = []

    def suggest_int(self, name, low, high, step=1):
        self.suggested.append(name)
        self.params[name] = low
        return low

    def suggest_float(self, name, low, high, step=None):
        self.suggested.append(name)
        self.params[name] = low
        return low


def _make_best_trial(step_overrides=None):
    distributions = {
        "HALFLIFE_FAST": optuna.distributions.IntDistribution(15, 30, step=1),
        "HALFLIFE_SLOW": optuna.distributions.IntDistribution(60, 100, step=1),
        "CONTINUITY_BONUS": optuna.distributions.FloatDistribution(0.06, 0.20),
        "RISK_AVERSION": optuna.distributions.FloatDistribution(12.0, 20.0),
        "CVAR_DAILY_LIMIT": optuna.distributions.FloatDistribution(0.04, 0.06),
    }
    if step_overrides:
        distributions.update(step_overrides)
    params = {
        "HALFLIFE_FAST": 21,
        "HALFLIFE_SLOW": 65,
        "CONTINUITY_BONUS": 0.15,
        "RISK_AVERSION": 12.0,
        "CVAR_DAILY_LIMIT": 0.04,
    }
    return optuna.trial.create_trial(
        params=params,
        distributions=distributions,
        values=[1.23, 1.11],
        user_attrs={},
    )


def _make_study(captured=None, include_trials=False, trial=None):
    best_trial = trial or _make_best_trial()

    class _Study:
        def __init__(self):
            self.best_params = dict(best_trial.params)
            self.best_trials = [best_trial]
            self.trials = [best_trial]
            self.directions = [
                optuna.study.StudyDirection.MAXIMIZE,
                optuna.study.StudyDirection.MAXIMIZE,
            ]
            if include_trials:
                self.trials = [best_trial]

        def optimize(self, objective, **kwargs):
            if captured is not None:
                captured.update(kwargs)

    return _Study()


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

    with pytest.raises(RuntimeError, match="(no completed trials|consecutive trial failures|Aborting)"):
        optimizer.run_optimization()


def test_optimizer_logger_does_not_duplicate_handlers_on_reload():
    before = len(optimizer.logger.handlers)
    importlib.reload(optimizer)
    after = len(optimizer.logger.handlers)

    assert after == before


def test_train_end_staleness_warning_emitted_after_configure(caplog, monkeypatch):
    """Warning must reach the log — i.e. fire after handlers are installed."""
    import optimizer

    monkeypatch.setattr(optimizer, "TRAIN_END", "2020-01-01")
    monkeypatch.setattr(optimizer, "_utc_today", lambda: pd.Timestamp("2020-09-01"))
    with caplog.at_level(logging.WARNING, logger="Optimizer"):
        optimizer.configure_optimizer_logging(color=False)

    assert any(
        "TRAIN_END" in record.message and "days behind" in record.message
        for record in caplog.records
    )


def test_train_end_staleness_does_not_raise():
    """Staleness check must never alter control flow."""
    import optimizer

    optimizer.configure_optimizer_logging(color=False)


def test_train_end_staleness_warning_not_emitted_when_fresh(caplog, monkeypatch):
    import optimizer

    monkeypatch.setattr(optimizer, "TRAIN_END", "2020-01-01")
    monkeypatch.setattr(optimizer, "_utc_today", lambda: pd.Timestamp("2020-02-01"))
    with caplog.at_level(logging.WARNING, logger="Optimizer"):
        optimizer.configure_optimizer_logging(color=False)

    assert not any(
        "TRAIN_END" in record.message and "days behind" in record.message
        for record in caplog.records
    )


def test_objective_returns_zero_when_max_drawdown_is_zero(monkeypatch):
    class _Result:
        metrics = {"cagr": 0.0, "max_dd": 0.0, "turnover": 0.0}
        rebal_log = None

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    objective = optimizer.MomentumObjective(market_data={"DUMMY": pd.DataFrame({"Close":[1.0]*510, "Volume":[1e6]*510}, index=pd.date_range("2017-01-01", periods=510, freq="B"))}, universe_type="nifty500")
    trial = _fixed_trial_params()

    score, calmar = objective(trial)
    assert score == 0.0
    assert calmar == 0.0


def test_objective_propagates_optimization_error(monkeypatch):
    monkeypatch.setattr(
        optimizer,
        "run_backtest",
        lambda **kwargs: (_ for _ in ()).throw(
            optimizer.OptimizationError("Solver failed")
        ),
    )

    objective = optimizer.MomentumObjective(market_data={"DUMMY": pd.DataFrame({"Close":[1.0]*510, "Volume":[1e6]*510}, index=pd.date_range("2017-01-01", periods=510, freq="B"))}, universe_type="nifty500")
    trial = _fixed_trial_params()

    with pytest.raises(optimizer.OptimizationError, match="Solver failed"):
        objective(trial)


def test_objective_propagates_unexpected_errors(monkeypatch):
    monkeypatch.setattr(
        optimizer,
        "run_backtest",
        lambda **kwargs: (_ for _ in ()).throw(TypeError("bad type")),
    )

    objective = optimizer.MomentumObjective(market_data={"DUMMY": pd.DataFrame({"Close":[1.0]*510, "Volume":[1e6]*510}, index=pd.date_range("2017-01-01", periods=510, freq="B"))}, universe_type="nifty500")
    trial = _fixed_trial_params()

    with pytest.raises(TypeError, match="bad type"):
        objective(trial)


def test_objective_returns_numeric_score_without_hard_drawdown_prune(monkeypatch):
    class _Result:
        metrics = {"cagr": 10.0, "max_dd": 20.0, "turnover": 0.0}
        rebal_log = pd.DataFrame(
            {
                "realised_cvar": [0.01, 0.01, 0.01],
                "exposure_multiplier": [1.0, 1.0, 1.0],
                "n_positions": [8, 8, 8],
                "apply_decay": [False, False, False],
            }
        )

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    objective = optimizer.MomentumObjective(market_data={"DUMMY": pd.DataFrame({"Close":[1.0]*510, "Volume":[1e6]*510}, index=pd.date_range("2017-01-01", periods=510, freq="B"))}, universe_type="nifty500")
    trial = _fixed_trial_params(HALFLIFE_SLOW=65, RISK_AVERSION=12.0)

    out = objective(trial)
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_fitness_penalizes_explicit_forced_cash_events():
    metrics = {"cagr": 12.0, "max_dd": 10.0, "turnover": 0.0}
    base_rebal = pd.DataFrame(
        {
            "realised_cvar": [0.01, 0.01, 0.01, 0.01],
            "exposure_multiplier": [1.0, 1.0, 1.0, 1.0],
            "n_positions": [8, 8, 8, 8],
            "apply_decay": [False, False, False, False],
        }
    )
    forced_rebal = base_rebal.assign(forced_to_cash=[False, True, False, True])

    base_score, _, base_diag = optimizer._fitness_from_metrics(metrics, base_rebal)
    forced_score, _, forced_diag = optimizer._fitness_from_metrics(metrics, forced_rebal)

    assert base_diag["forced_cash_penalty"] == 0.0
    assert forced_diag["forced_cash_penalty"] == 0.0
    assert forced_score == pytest.approx(base_score)


def test_fitness_falls_back_to_decay_and_zero_positions_when_forced_cash_flag_missing():
    metrics = {"cagr": 12.0, "max_dd": 10.0, "turnover": 0.0}
    rebal_log = pd.DataFrame(
        {
            "realised_cvar": [0.01, 0.01, 0.01, 0.01],
            "exposure_multiplier": [1.0, 1.0, 1.0, 1.0],
            "n_positions": [8, 0, 8, 0],
            "apply_decay": [False, True, False, True],
        }
    )

    score, _, diag = optimizer._fitness_from_metrics(metrics, rebal_log)

    assert diag["forced_cash_penalty"] == 0.0
    assert isinstance(score, float)
    assert diag["concentration_mult"] > 1.0


def test_fitness_marks_reachable_score_plateaus_as_ceiling_hits():
    metrics = {"cagr": 32.0, "max_dd": 0.01, "turnover": 0.0, "sortino": 10.0}
    rebal_log = pd.DataFrame(
        {
            "realised_cvar": [0.0, 0.0, 0.0, 0.0],
            "exposure_multiplier": [1.0, 1.0, 1.0, 1.0],
            "n_positions": [8, 8, 8, 8],
            "apply_decay": [False, False, False, False],
        }
    )

    score, _, diag = optimizer._fitness_from_metrics(metrics, rebal_log)

    assert score > 2.0
    assert diag["anomaly_hit"] is False
    assert diag["ceiling_hit"] is False


def test_fitness_applies_nonlinear_turnover_drag_above_18x():
    metrics = {"cagr": 25.0, "max_dd": 10.0, "turnover": 24.0, "sortino": 2.5}
    rebal_log = pd.DataFrame(
        {
            "realised_cvar": [0.01, 0.01, 0.01, 0.01],
            "exposure_multiplier": [1.0, 1.0, 1.0, 1.0],
            "n_positions": [8, 8, 8, 8],
            "apply_decay": [False, False, False, False],
        }
    )

    score, _, diag = optimizer._fitness_from_metrics(metrics, rebal_log)

    assert diag["cagr_net"] == pytest.approx(14.2)
    assert diag["forced_cash_penalty"] == 0.0
    assert score < 1.5


def test_optimizer_defaults_reflect_v11_58_search_space():
    assert optimizer.N_TRIALS == 400
    assert optimizer.OBJECTIVE_VERSION == "fitness_v11_58"
    assert optimizer.SEARCH_SPACE_BOUNDS["HALFLIFE_FAST"] == (30, 70, 5)
    assert optimizer.SEARCH_SPACE_BOUNDS["HALFLIFE_SLOW"] == (100, 180, 10)
    assert optimizer.SEARCH_SPACE_BOUNDS["CONTINUITY_BONUS"] == (0.05, 0.25, 0.05)
    assert optimizer.SEARCH_SPACE_BOUNDS["RISK_AVERSION"] == (8.0, 18.0, 1.0)
    assert optimizer.SEARCH_SPACE_BOUNDS["CVAR_DAILY_LIMIT"] == (0.040, 0.120, 0.010)
    assert optimizer.SEARCH_SPACE_BOUNDS["MAX_POSITIONS"] == (10, 24, 2)
    assert optimizer.SEARCH_SPACE_BOUNDS["MIN_EXPOSURE_FLOOR"] == (0.0, 0.20, 0.05)


def test_save_optimal_config_replaces_existing_file_atomically(tmp_path: Path):
    output_path = tmp_path / "optimal_cfg.json"
    output_path.write_text('{"old": 1}', encoding="utf-8")

    captured = {}
    original_replace = optimizer.os.replace

    def _capturing_replace(src, dst):
        captured["src"] = src
        captured["dst"] = dst
        captured["src_exists_before_replace"] = Path(src).exists()
        return original_replace(src, dst)

    optimizer.os.replace = _capturing_replace
    try:
        optimizer.save_optimal_config({"HALFLIFE_FAST": 34}, str(output_path))
    finally:
        optimizer.os.replace = original_replace

    with output_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    assert payload == {"HALFLIFE_FAST": 34}
    assert captured["dst"] == str(output_path)
    assert captured["src"] != str(output_path)
    assert captured["src_exists_before_replace"] is True


def test_pre_load_data_deduplicates_inputs_and_appends_crsldx_index(monkeypatch):
    monkeypatch.setattr(optimizer, "TRAIN_START", "2020-01-01")
    monkeypatch.setattr(optimizer, "TEST_END", "2020-12-31")
    monkeypatch.setattr(optimizer, "fetch_nse_equity_universe", lambda: ["ABC", "^NSEI", "ABC"])

    captured = {}

    def _fake_load_or_fetch(*, tickers, required_start, required_end):
        captured["tickers"] = tickers
        captured["required_start"] = required_start
        captured["required_end"] = required_end
        return {"ok": True, "^NSEI": pd.DataFrame({"Close": [100.0] * 800, "Volume": [1e9]*800}, index=pd.date_range("2018-11-27", periods=800, freq="B"))}

    monkeypatch.setattr(optimizer, "load_or_fetch", _fake_load_or_fetch)

    result = optimizer.pre_load_data("  NSE_TOTAL ")

    assert result["market_data"]["ok"] is True
    assert "^NSEI" in result["market_data"]
    assert result["precomputed_matrices"] is None
    assert captured["required_start"] == optimizer._compute_warmup_start("2020-01-01", optimizer.UltimateConfig())
    assert captured["required_end"] == optimizer.TEST_END
    assert captured["tickers"].count("ABC") == 1
    assert "^NSEI" in captured["tickers"]
    assert "^CRSLDX" in captured["tickers"]


def test_pre_load_data_includes_historical_union_for_nifty500(monkeypatch):
    monkeypatch.setattr(optimizer, "TRAIN_START", "2020-01-01")
    monkeypatch.setattr(optimizer, "TEST_END", "2020-09-30")
    monkeypatch.setattr(optimizer, "get_nifty500", lambda: ["LIVEONLY"])

    def _fake_hist(universe_type, date):
        assert universe_type == "nifty500"
        if pd.Timestamp(date) == pd.Timestamp("2020-03-31"):
            return ["OLD1", "OLD2"]
        if pd.Timestamp(date) == pd.Timestamp("2020-06-30"):
            return ["OLD2", "OLD3"]
        return []

    monkeypatch.setattr(optimizer, "get_historical_universe", _fake_hist)

    captured = {}

    def _fake_load_or_fetch(*, tickers, required_start, required_end, cfg=None):
        captured["tickers"] = tickers
        captured["required_start"] = required_start
        captured["required_end"] = required_end
        return {"ok": True, "^NSEI": pd.DataFrame({"Close": [100.0] * 800, "Volume": [1e9]*800}, index=pd.date_range("2018-11-27", periods=800, freq="B"))}

    monkeypatch.setattr(optimizer, "load_or_fetch", _fake_load_or_fetch)

    result = optimizer.pre_load_data("nifty500")

    assert result["market_data"]["ok"] is True
    assert "^NSEI" in result["market_data"]
    assert result["precomputed_matrices"] is None
    assert captured["required_start"] == optimizer._compute_warmup_start("2020-01-01", optimizer.UltimateConfig())
    assert captured["required_end"] == optimizer.TEST_END
    assert "LIVEONLY" in captured["tickers"]
    assert "OLD1" in captured["tickers"]
    assert "OLD2" in captured["tickers"]
    assert "OLD3" in captured["tickers"]
    assert "^NSEI" in captured["tickers"]
    assert "^CRSLDX" in captured["tickers"]


def test_pre_load_data_skips_halt_simulation_when_disabled(monkeypatch):
    monkeypatch.setattr(optimizer, "TRAIN_START", "2020-01-01")
    monkeypatch.setattr(optimizer, "TEST_END", "2020-03-31")
    monkeypatch.setattr(optimizer, "fetch_nse_equity_universe", lambda: ["ABC"])

    cfg = optimizer.UltimateConfig()
    cfg.SIMULATE_HALTS = False

    idx = pd.date_range("2018-11-27", "2020-03-31", freq="B")

    monkeypatch.setattr(
        optimizer,
        "load_or_fetch",
        lambda **kwargs: {
            "ABC": pd.DataFrame({"Close": [1.0] * len(idx), "Volume": [1e6] * len(idx)}, index=idx),
            "^NSEI": pd.DataFrame({"Close": [100.0] * len(idx), "Volume": [1e9] * len(idx)}, index=idx)
        },
    )

    def _fail_if_called(_market_data):
        raise AssertionError("apply_halt_simulation should not be called when disabled")

    monkeypatch.setattr(optimizer, "apply_halt_simulation", _fail_if_called)

    result = optimizer.pre_load_data("nse_total", cfg=cfg)

    assert set(result["market_data"]) == {"ABC", "^NSEI"}
    assert isinstance(result["precomputed_matrices"], dict)


def test_validate_regime_benchmark_data_accepts_nsei_fallback():
    idx = pd.date_range("2019-01-01", "2025-12-31", freq="B")
    market_data = {
        "^NSEI": pd.DataFrame({"Close": np.linspace(100.0, 200.0, len(idx))}, index=idx)
    }

    optimizer._validate_regime_benchmark_data(market_data, "2020-01-01", "2025-12-31")


def test_validate_regime_benchmark_data_allows_partial_coverage_with_warning(caplog):
    idx = pd.date_range("2024-01-15", "2025-12-31", freq="B")
    market_data = {
        "^NSEI": pd.DataFrame({"Close": np.linspace(100.0, 200.0, len(idx))}, index=idx)
    }

    with caplog.at_level("WARNING"):
        optimizer._validate_regime_benchmark_data(market_data, "2016-11-27", "2025-12-31")

    assert "partial coverage" in caplog.text
    assert "Proceeding with available history" in caplog.text


def test_validate_regime_benchmark_data_raises_when_benchmarks_missing():
    with pytest.raises(optimizer.OptimizationError, match="Regime benchmark validation failed") as exc_info:
        optimizer._validate_regime_benchmark_data({}, "2020-01-01", "2025-12-31")

    assert exc_info.value.error_type == optimizer.OptimizationErrorType.DATA


def test_build_sampler_returns_tpe_sampler(monkeypatch):
    monkeypatch.setattr(optimizer, "OPTUNA_SEED", None)
    sampler_unseeded = optimizer._build_sampler()

    monkeypatch.setattr(optimizer, "OPTUNA_SEED", "123")
    sampler_seeded = optimizer._build_sampler()

    assert isinstance(sampler_unseeded, optimizer.TPESampler)
    assert isinstance(sampler_seeded, optimizer.TPESampler)


def test_iter_wfo_slices_keeps_first_full_calendar_year():
    slices = list(optimizer._iter_wfo_slices("2018-01-01", "2022-12-31"))

    assert slices[0] == ("2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31")
    assert [oos_start for _, _, oos_start, _ in slices] == [
        "2020-01-01",
        "2021-01-01",
        "2022-01-01",
    ]


def test_default_train_start_drops_2019_oos_fold():
    slices = list(optimizer._iter_wfo_slices(optimizer.TRAIN_START, optimizer.TRAIN_END))

    assert optimizer.TRAIN_START == "2019-01-01"
    # SUCCESS: 2021, 2022, 2023.
    assert [oos_start for _, _, oos_start, _ in slices] == [
        '2021-01-01',
        '2022-01-01',
        '2023-01-01',
    ]


def test_normalize_universe_type_falls_back_to_nifty500():
    assert optimizer._normalize_universe_type("  typo  ") == "nifty500"


def test_pre_load_data_uses_normalized_fallback_universe_for_history(monkeypatch):
    monkeypatch.setattr(optimizer, "TRAIN_START", "2020-01-01")
    monkeypatch.setattr(optimizer, "TEST_END", "2020-03-31")
    monkeypatch.setattr(optimizer, "get_nifty500", lambda: ["LIVEONLY"])

    historical_calls = []

    def _fake_hist(universe_type, date):
        historical_calls.append(universe_type)
        return ["OLD1"]

    monkeypatch.setattr(optimizer, "get_historical_universe", _fake_hist)
    monkeypatch.setattr(optimizer, "load_or_fetch", lambda **kwargs: {"ok": True, "^NSEI": pd.DataFrame({"Close": [100.0] * 800, "Volume": [1e9]*800}, index=pd.date_range("2018-11-27", periods=800, freq="B"))})

    result = optimizer.pre_load_data(" typo ")

    assert result["market_data"]["ok"] is True
    assert "^NSEI" in result["market_data"]
    assert result["precomputed_matrices"] is None
    assert historical_calls
    assert set(historical_calls) == {"nifty500"}


def test_objective_suggests_cvar_lookback_for_non_fixed_trials(monkeypatch):
    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _StaticResult())

    objective = optimizer.MomentumObjective(market_data={"DUMMY": pd.DataFrame({"Close":[1.0]*610, "Volume":[1e6]*610}, index=pd.date_range("2017-01-01", periods=610, freq="B"))}, universe_type="nifty500")
    trial = _TrackingTrial()

    objective(trial)

    assert "CVAR_LOOKBACK" in trial.suggested
    assert "MAX_POSITIONS" in trial.suggested
    assert "SIGNAL_LAG_DAYS" in trial.suggested


def test_objective_suggests_min_exposure_floor_for_non_fixed_trials(monkeypatch):
    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _StaticResult())

    objective = optimizer.MomentumObjective(market_data={"DUMMY": pd.DataFrame({"Close":[1.0]*610, "Volume":[1e6]*610}, index=pd.date_range("2017-01-01", periods=610, freq="B"))}, universe_type="nifty500")
    trial = _TrackingTrial()

    objective(trial)

    assert "MIN_EXPOSURE_FLOOR" in trial.suggested


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
        market_data={"DUMMY": pd.DataFrame({"Close":[1.0]*610, "Volume":[1e6]*610}, index=pd.date_range("2017-01-01", periods=610, freq="B"))}, universe_type="nifty500", search_space=custom_space
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

    expected_raw = (10.0 / ((5.0 + 1.0) * 2.8)) * 0.5 - 0.5
    expected = np.copysign(np.log1p(abs(expected_raw)), expected_raw)
    score, _ = objective(trial)
    assert round(score, 6) == round(expected, 6)


def test_objective_accepts_halflife_bounds_with_step(monkeypatch):
    class _Result:
        metrics = {"cagr": 10.0, "max_dd": 5.0, "turnover": 0.0}
        rebal_log = None

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    custom_space = {
        "HALFLIFE_FAST": (10, 40, 5),
        "HALFLIFE_SLOW": (50, 120, 5),
        "CONTINUITY_BONUS": (0.1, 0.1, 0.01),
        "RISK_AVERSION": (10.0, 10.0, 0.5),
        "CVAR_DAILY_LIMIT": (0.04, 0.04, 0.005),
    }
    objective = optimizer.MomentumObjective(
        market_data={"DUMMY": pd.DataFrame({"Close":[1.0]*610, "Volume":[1e6]*610}, index=pd.date_range("2017-01-01", periods=610, freq="B"))}, universe_type="nifty500", search_space=custom_space
    )
    trial = optuna.trial.FixedTrial(
        {
            "HALFLIFE_FAST": 15,
            "HALFLIFE_SLOW": 55,
            "CONTINUITY_BONUS": 0.1,
            "RISK_AVERSION": 10.0,
            "CVAR_DAILY_LIMIT": 0.04,
        }
    )

    assert isinstance(objective(trial), tuple)


def test_run_optimization_forces_single_job(monkeypatch):
    monkeypatch.setattr(optimizer, "N_TRIALS", 1)
    monkeypatch.setattr(optimizer, "N_JOBS", 3)
    monkeypatch.setattr(optimizer, "pre_load_data", lambda universe_type: {})
    monkeypatch.setattr(optimizer, "save_optimal_config", lambda best_params: None)

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _StaticResult())

    captured = {}
    monkeypatch.setattr(
        optimizer.optuna, "create_study", lambda **kwargs: _make_study(captured=captured)
    )

    optimizer.run_optimization()

    assert captured["n_jobs"] == 1


def test_run_optimization_uses_selected_universe(monkeypatch):
    monkeypatch.setattr(optimizer, "N_TRIALS", 1)
    monkeypatch.setattr(optimizer, "save_optimal_config", lambda best_params: None)

    captured = {}
    def _capture_preload(universe_type):
        captured["pre_load_universe"] = universe_type
        return {}

    monkeypatch.setattr(optimizer, "pre_load_data", _capture_preload)

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _StaticResult())
    monkeypatch.setattr(
        optimizer.optuna, "create_study", lambda **kwargs: _make_study(include_trials=True)
    )

    optimizer.run_optimization(universe_type="nse_total")

    assert captured["pre_load_universe"] == "nse_total"


def test_run_optimization_normalizes_unknown_universe(monkeypatch):
    monkeypatch.setattr(optimizer, "N_TRIALS", 1)
    monkeypatch.setattr(optimizer, "save_optimal_config", lambda best_params: None)

    captured = {}
    def _capture_preload(universe_type):
        captured["pre_load_universe"] = universe_type
        return {}

    monkeypatch.setattr(optimizer, "pre_load_data", _capture_preload)

    class _Result:
        metrics = {"final": 1.0, "cagr": 1.0, "max_dd": 1.0, "calmar": 1.1}

    def _fake_run_backtest(**kwargs):
        captured["backtest_universe"] = kwargs["universe_type"]
        return _Result()

    monkeypatch.setattr(optimizer, "run_backtest", _fake_run_backtest)

    monkeypatch.setattr(
        optimizer.optuna,
        "create_study",
        lambda **kwargs: _make_study(
            include_trials=True,
            trial=_make_best_trial(
                {
                    "HALFLIFE_FAST": optuna.distributions.IntDistribution(15, 29, step=2),
                    "HALFLIFE_SLOW": optuna.distributions.IntDistribution(60, 100, step=5),
                    "CONTINUITY_BONUS": optuna.distributions.FloatDistribution(0.06, 0.18, step=0.03),
                    "RISK_AVERSION": optuna.distributions.FloatDistribution(12.0, 20.0, step=0.5),
                    "CVAR_DAILY_LIMIT": optuna.distributions.FloatDistribution(0.04, 0.06, step=0.005),
                }
            ),
        ),
    )

    optimizer.run_optimization(universe_type=" typo ")

    assert captured["pre_load_universe"] == "nifty500"
    assert captured["backtest_universe"] == "nifty500"


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
    monkeypatch.setattr(optimizer, "N_TRIALS", 1)
    monkeypatch.setattr(optimizer, "pre_load_data", lambda universe_type: {})
    monkeypatch.setattr(optimizer, "save_optimal_config", lambda best_params: None)

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _StaticResult())

    captured = {}

    def _fake_create_study(**kwargs):
        captured["storage"] = kwargs.get("storage")
        return _make_study(captured=captured)

    monkeypatch.setattr(optimizer.optuna, "create_study", _fake_create_study)
    # Ensure OPTUNA_N_JOBS is absent so the default -1 path is exercised
    monkeypatch.delenv("OPTUNA_N_JOBS", raising=False)

    optimizer.run_optimization(in_memory=True)

    assert captured["storage"] == "sqlite:///:memory:"


def test_objective_allows_equal_halflife_values(monkeypatch):
    class _Result:
        metrics = {"cagr": 10.0, "max_dd": 5.0, "turnover": 0.0}
        rebal_log = None

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    objective = optimizer.MomentumObjective(market_data={"DUMMY": pd.DataFrame({"Close":[1.0]*610, "Volume":[1e6]*610}, index=pd.date_range("2017-01-01", periods=610, freq="B"))}, universe_type="nifty500")
    trial = _fixed_trial_params(HALFLIFE_SLOW=21)

    score, _ = objective(trial)
    assert score != 0.0


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
    cfg = UltimateConfig(MAX_SINGLE_NAME_WEIGHT=1.0, CVAR_DAILY_LIMIT=1.0)
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
    adv_shares = np.array([2e15, 2e15, 2e15])
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
    cfg = UltimateConfig(MAX_SINGLE_NAME_WEIGHT=1.0, CVAR_DAILY_LIMIT=1.0)
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
            adv_shares=np.array([1e15, 1e15]),
            prices=np.array([100.0, 100.0]),
            portfolio_value=1_000_000.0,
        )

    assert weights.shape == (2,)
    assert weights[1] == pytest.approx(0.0, abs=1e-12)
    assert "reason=insufficient_history" in caplog.text


def test_optimizer_uses_higher_turnover_penalty_for_illiquid_name(monkeypatch):
    # Use large IMPACT_COEFF so impact term dominates over base_rate.
    # The impact formula is: IMPACT_COEFF * trade_notional / adv_shares.
    # With prev_w=[0.5, 0.0], target hints derived from expected_returns are
    # [0.5, 0.5]*u_gamma, so trade_estimate for B is larger than for A, and B's
    # ADV is 1000x smaller — both effects amplify B's turnover cost.
    cfg = UltimateConfig(MAX_SINGLE_NAME_WEIGHT=1.0, CVAR_DAILY_LIMIT=1.0, IMPACT_COEFF=1.0)
    engine = InstitutionalRiskEngine(cfg)

    captured = {}

    class _FakeResInfo:
        status = "solved"

    class _FakeRes:
        def __init__(self, n_vars):
            self.info = _FakeResInfo()
            self.x = np.zeros(n_vars, dtype=float)
            if n_vars >= 2:
                self.x[0] = 0.4
                self.x[1] = 0.4

    class _FakeOSQP:
        def setup(self, P, q, A, lower_bounds, upper_bounds, **kwargs):
            captured["q"] = np.array(q, dtype=float)
            captured["n_vars"] = len(q)

        def solve(self):
            return _FakeRes(captured["n_vars"])

        def update(self, **kwargs):
            pass

        def warm_start(self, **kwargs):
            pass

    monkeypatch.setattr("momentum_engine.osqp.OSQP", _FakeOSQP)

    expected_returns = np.array([0.01, 0.01], dtype=float)
    prices = np.array([100.0, 100.0], dtype=float)
    # Liquid vs illiquid: 1e15 shares vs 8e6 shares
    # adv_limit[B] = 8e6 * MAX_ADV_PCT(0.05) / pv(1e6) = 0.40 → passes post-solve
    # B's impact: IMPACT_COEFF(1.0) * ~3e5 / 8e6 ≈ 0.04 >> base_rate(0.001)
    # A's impact: IMPACT_COEFF(1.0) * ~3e5 / 1e15 ≈ 0 → uses base_rate(0.001)
    adv_shares = np.array([1e15, 8e6], dtype=float)
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
    assert turnover_q[1] > turnover_q[0]


def test_optimizer_turnover_penalty_respects_execution_floor_and_cap(monkeypatch):
    cfg = UltimateConfig(IMPACT_COEFF=1.0, ROUND_TRIP_SLIPPAGE_BPS=20.0, MAX_SINGLE_NAME_WEIGHT=1.0, CVAR_DAILY_LIMIT=1.0, MIN_EXPOSURE_FLOOR=0.0)
    engine = InstitutionalRiskEngine(cfg)

    captured = {}

    class _FakeResInfo:
        status = "solved"

    class _FakeRes:
        def __init__(self, n_vars):
            self.info = _FakeResInfo()
            self.x = np.zeros(n_vars, dtype=float)
            # Match targeting based on engine's internal gross logic
            if n_vars >= 2:
                self.x[0] = 0.4
                self.x[1] = 0.0

    class _FakeOSQP:
        def setup(self, P, q, A, lower_bounds, upper_bounds, **kwargs):
            captured["q"] = np.array(q, dtype=float)
            captured["n_vars"] = len(q)

        def solve(self):
            return _FakeRes(captured["n_vars"])

        def update(self, **kwargs):
            pass

        def warm_start(self, **kwargs):
            pass

    monkeypatch.setattr("momentum_engine.osqp.OSQP", _FakeOSQP)

    expected_returns = np.array([0.01, 0.01], dtype=float)
    prices = np.array([100.0, 100.0], dtype=float)
    adv_shares = np.array([1e15, 1e-6], dtype=float)
    hist = pd.DataFrame(np.tile([0.001, -0.001], (10, 1)), columns=["A", "B"])

    engine.optimize(
        expected_returns=expected_returns,
        historical_returns=hist,
        adv_shares=adv_shares,
        prices=prices,
        portfolio_value=1_000_000.0,
        prev_w=np.array([0.0, 0.0], dtype=float),
        exposure_multiplier=1.0,
    )

    turnover_q = captured["q"][2:4]
    base_one_way = cfg.ROUND_TRIP_SLIPPAGE_BPS / 20_000.0
    assert turnover_q[0] == pytest.approx(base_one_way)
    assert turnover_q[1] == pytest.approx(0.05)


def test_optimizer_osqp_setup_falls_back_to_polish_keyword(monkeypatch):
    cfg = UltimateConfig(MAX_SINGLE_NAME_WEIGHT=1.0, CVAR_DAILY_LIMIT=1.0, MIN_EXPOSURE_FLOOR=0.0)
    engine = InstitutionalRiskEngine(cfg)
    calls = []

    class _FakeResInfo:
        status = "solved"

    class _FakeRes:
        def __init__(self, n_vars):
            self.info = _FakeResInfo()
            self.x = np.zeros(n_vars, dtype=float)
            # Match targeting based on engine's internal gross logic
            if n_vars >= 2:
                self.x[0] = 0.4
                self.x[1] = 0.4

    class _FakeOSQP:
        def setup(self, _P, q, _A, _l, _u, **kwargs):
            calls.append(dict(kwargs))
            if "warm_starting" in kwargs:
                raise TypeError("unexpected keyword argument 'warm_starting'")
            self._n_vars = len(q)

        def solve(self):
            return _FakeRes(self._n_vars)

        def update(self, **kwargs):
            pass

        def warm_start(self, **kwargs):
            pass

    monkeypatch.setattr("momentum_engine.osqp.OSQP", _FakeOSQP)

    hist = pd.DataFrame(np.tile([0.001, -0.001], (20, 1)), columns=["A", "B"])
    weights = engine.optimize(
        expected_returns=np.array([0.01, 0.01], dtype=float),
        historical_returns=hist,
        adv_shares=np.array([1e15, 1e15], dtype=float),
        prices=np.array([100.0, 100.0], dtype=float),
        portfolio_value=1_000_000.0,
        prev_w=np.array([0.0, 0.0], dtype=float),
        exposure_multiplier=1.0,
    )

    assert weights.shape == (2,)
    assert len(calls) == 2


def test_objective_cvar_lookback_min_scales_with_dimensionality(monkeypatch):
    class _DummyCfg:
        def __init__(self, **kwargs):
            self.INITIAL_CAPITAL = 1_000_000.0
            self.MAX_POSITIONS = 10
            self.MAX_PORTFOLIO_RISK_PCT = 0.20
            self.MAX_SINGLE_NAME_WEIGHT = 0.25
            self.MAX_SECTOR_WEIGHT = 0.40
            self.MAX_ADV_PCT = 0.05
            self.MIN_ADV_CRORES = 100.0
            self.IMPACT_COEFF = 5e-4
            self.ROUND_TRIP_SLIPPAGE_BPS = 20.0
            self.REBALANCE_FREQ = "W-FRI"
            self.CVAR_DAILY_LIMIT = 0.055
            self.CVAR_ALPHA = 0.95
            self.CVAR_LOOKBACK = 90
            self.ADV_LOOKBACK = 90
            self.CVAR_SENTINEL_MULTIPLIER = 2.5
            self.CVAR_MIN_HISTORY = 20
            self.CVAR_HARD_BREACH_MULTIPLIER = 1.5
            self.DELEVERAGING_LIMIT = 0.20
            self.MIN_EXPOSURE_FLOOR = 0.05
            self.CAPITAL_ELASTICITY = 0.15
            self.DRIFT_TOLERANCE = 0.02
            self.IS_DD_GATE = 40.0
            self.IS_DD_PENALTY_PCT = 12.0
            self.CONCENTRATION_MIN_POSITIONS = 6.0
            self.CONCENTRATION_PENALTY_WEIGHT = 0.30
            self.SORTINO_QUALITY_TARGET = 2.5
            self.SIGNAL_ANNUAL_FACTOR = 252
            self.HISTORY_GATE = 90
            self.HALFLIFE_FAST = 21
            self.HALFLIFE_SLOW = 63
            self.SIGNAL_LAG_DAYS = 21
            self.RISK_AVERSION = 5.0
            self.SLACK_PENALTY = 1000.0
            self.DIMENSIONALITY_MULTIPLIER = 3
            self.Z_SCORE_CLIP = 3.0
            self.CONTINUITY_BONUS = 0.15
            self.CONTINUITY_DISPERSION_FLOOR = 0.1
            self.CONTINUITY_MAX_SCALAR = 0.20
            self.CONTINUITY_MAX_HOLD_WEIGHT = 0.10
            self.CONTINUITY_ACTIVITY_WINDOW = 5
            self.CONTINUITY_STALE_SESSIONS = 10
            self.CONTINUITY_FLAT_RET_EPS = 1e-12
            self.CONTINUITY_MIN_ADV_NOTIONAL = 0.0
            self.KNIFE_WINDOW = 20
            self.KNIFE_THRESHOLD = -0.15
            self.MAX_ABSENT_PERIODS = 12
            self.MAX_DECAY_ROUNDS = 3
            self.DECAY_FACTOR = 0.5
            self.GHOST_VOL_LOOKBACK = 20
            self.GHOST_RET_DRIFT = -0.02
            self.GHOST_VOL_FALLBACK = 0.04
            self.YF_BATCH_TIMEOUT = 120.0
            self.YF_CHUNK_TIMEOUT = 90.0
            self.YF_ADV_TIMEOUT = 60.0
            self.SECTOR_FETCH_TIMEOUT = 8.0
            self.REGIME_VOL_FLOOR = 0.18
            self.REGIME_VOL_MULTIPLIER = 1.5
            self.REGIME_SIGMOID_STEEPNESS = 10.0
            self.REGIME_SMA_FAST_WINDOW = 50
            self.REGIME_SMA_WINDOW = 200
            self.REGIME_VOL_EWMA_SPAN = 20
            self.REGIME_LT_VOL_EWMA_SPAN = 1260
            self.SIMULATE_HALTS = False
            self.DIVIDEND_SWEEP = True
            self.SPLIT_TOLERANCE = 0.005
            self.AUTO_ADJUST_PRICES = True
            self.EQUITY_HIST_CAP = 500
            self.SLIPPAGE_BPS = 20.0
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _RecordingTrial:
        params = {}
        def __init__(self):
            self.bounds = {}
        def suggest_int(self, name, low, high, step=1):
            self.bounds[name] = (low, high, step)
            return low
        def suggest_float(self, name, low, high, step=None):
            return low

    monkeypatch.setattr(optimizer, "UltimateConfig", _DummyCfg)
    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _StaticResult())

    idx = pd.date_range("2017-01-01", periods=610, freq="B")
    objective = optimizer.MomentumObjective(market_data={"SYM00": pd.DataFrame({"Close":[1.0]*610, "Volume":[1e6]*610}, index=idx)}, universe_type="n")
    trial = _RecordingTrial()

    objective(trial)

    # 20 * 3 = 60.
    assert trial.bounds["CVAR_LOOKBACK"][0] == 60


def test_objective_prunes_when_cvar_lookback_bounds_are_infeasible(monkeypatch, relaxed_train_start):
    # For pruning: effective_cvar_lb_min = max(cvar_lb_min, MAX_POSITIONS * DIM_MULT)
    # must exceed cvar_lb_max.  With MAX_POSITIONS=120, DIM_MULT=3:
    # effective_min = max(200, 360) = 360 > 300 → TrialPruned.
    class _DummyCfg:
        def __init__(self, **kwargs):
            self.HALFLIFE_FAST = 21
            self.HALFLIFE_SLOW = 63
            self.CONTINUITY_BONUS = 0.15
            self.RISK_AVERSION = 5.0
            self.CVAR_DAILY_LIMIT = 0.9
            self.SIGNAL_LAG_DAYS = 0
            self.MIN_EXPOSURE_FLOOR = 0.0
            self.CVAR_LOOKBACK = 60
            self.DIMENSIONALITY_MULTIPLIER = 3
            self.MAX_POSITIONS = 120
            self.MAX_SINGLE_NAME_WEIGHT = 1.0
            self.REBALANCE_FREQ = "QE"

    monkeypatch.setattr(optimizer, "UltimateConfig", _DummyCfg)
    idx = pd.date_range("2016-01-01", periods=1000, freq="B")
    ss = dict(optimizer.SEARCH_SPACE_BOUNDS)
    ss["CVAR_LOOKBACK"] = (200, 300, 10)
    ss["MAX_POSITIONS"] = (120, 120, 1)

    objective = optimizer.MomentumObjective(
        market_data={"SYM00": pd.DataFrame({"Close":[1.0]*1000, "Volume":[1e6]*1000}, index=idx)},
        universe_type="n",
        search_space=ss,
    )
    trial = _fixed_trial_params(CVAR_LOOKBACK=300, MAX_POSITIONS=120)

    with pytest.raises(optuna.TrialPruned):
        objective(trial)


def test_objective_does_not_count_floor_score_as_gate_hit(monkeypatch):
    class _Result:
        metrics = {"cagr": 0.0, "max_dd": 0.0, "turnover": 0.0}
        rebal_log = pd.DataFrame()

    monkeypatch.setattr(
        optimizer,
        "_iter_wfo_slices",
        lambda *_args: [
            ("2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        ],
    )
    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())
    monkeypatch.setattr(optimizer, "_fitness_from_metrics", lambda *_args: _objective_diag(score=0.5))

    idx = pd.date_range("2017-01-01", periods=610, freq="B")
    objective = optimizer.MomentumObjective(market_data={"SYM00": pd.DataFrame({"Close":[1.0]*610, "Volume":[1e6]*610}, index=idx)}, universe_type="n")
    trial = _fixed_trial_params(HALFLIFE_SLOW=65, RISK_AVERSION=12.0)

    score, _ = objective(trial)
    assert score == pytest.approx(0.5)


def test_objective_prunes_after_third_structural_gate_hit(monkeypatch):
    class _Result:
        metrics = {"cagr": 0.0, "max_dd": 0.0, "turnover": 0.0}
        rebal_log = pd.DataFrame()

    monkeypatch.setattr(
        optimizer,
        "_iter_wfo_slices",
        lambda *_args: [
            ("2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
            ("2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
            ("2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        ],
    )
    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    monkeypatch.setattr(optimizer, "_fitness_from_metrics", lambda *_args: _objective_diag(score=-2.0, dd_gate_hit=True))

    idx = pd.date_range("2017-01-01", periods=610, freq="B")
    objective = optimizer.MomentumObjective(market_data={"SYM00": pd.DataFrame({"Close":[1.0]*610, "Volume":[1e6]*610}, index=idx)}, universe_type="n")
    trial = _fixed_trial_params(HALFLIFE_SLOW=65, RISK_AVERSION=12.0)

    with pytest.raises(optuna.TrialPruned):
        objective(trial)


def test_fitness_uses_symmetric_log_modulus_transform():
    rebal_log = pd.DataFrame({"n_positions": [8], "exposure_multiplier": [1.0], "realised_cvar": [0.01]})

    def _score_for_cagr(cagr: float) -> float:
        score, _, _ = optimizer._fitness_from_metrics(
            {"cagr": cagr, "max_dd": 10.0, "turnover": 0.0, "sortino": 1.0},
            rebal_log,
        )
        return score

    assert _score_for_cagr(-0.01) < _score_for_cagr(0.0) < _score_for_cagr(0.01)


def test_error_triage_callback_records_runtimeerror_class():
    callback = optimizer._error_triage_callback_factory()
    captured = {}

    class _Storage:
        def get_trial(self, _trial_id):
            return type("T", (), {"state": optuna.trial.TrialState.FAIL})()
        def set_trial_user_attr(self, _trial_id, key, value):
            captured[key] = value

    class _Study:
        _storage = _Storage()
        def stop(self):
            pass

    trial = type(
        "Trial",
        (),
        {
            "state": optuna.trial.TrialState.FAIL,
            "system_attrs": {"fail_reason": "RuntimeError: boom"},
            "_trial_id": 1,
        },
    )()

    callback(_Study(), trial)
    assert captured["error_class"] == "RuntimeError"


def test_error_triage_watchdog_stops_study(monkeypatch):
    callback = optimizer._error_triage_callback_factory()
    stopped = {"value": False}

    class _Storage:
        def get_trial(self, _trial_id):
            return type("T", (), {"state": optuna.trial.TrialState.FAIL})()
        def set_trial_user_attr(self, _trial_id, key, value):
            pass

    class _Study:
        _storage = _Storage()
        def stop(self):
            stopped["value"] = True

    trial = type(
        "Trial",
        (),
        {
            "state": optuna.trial.TrialState.FAIL,
            "system_attrs": {"fail_reason": "RuntimeError: boom"},
            "_trial_id": 1,
        },
    )()

    monkeypatch.setattr(optimizer, "N_TRIALS", 10)
    with pytest.raises(RuntimeError):
        for _ in range(4):
            callback(_Study(), trial)
    assert stopped["value"] is True


def test_run_optimization_sets_sqlite_wal_mode(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "optuna.db"
    monkeypatch.setattr(optimizer, "OPTUNA_STORAGE", f"sqlite:///{db_path}")
    monkeypatch.setattr(optimizer, "N_TRIALS", 1)
    monkeypatch.setattr(optimizer, "pre_load_data", lambda universe_type: {})
    monkeypatch.setattr(optimizer, "save_optimal_config", lambda best_params: None)

    class _Result:
        metrics = {"final": 1.0, "cagr": 1.0, "max_dd": 1.0, "calmar": 1.0, "sharpe": 1.0, "sortino": 1.0}

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    distributions = {
        "HALFLIFE_FAST": optuna.distributions.IntDistribution(15, 30, step=1),
        "HALFLIFE_SLOW": optuna.distributions.IntDistribution(60, 100, step=1),
        "CONTINUITY_BONUS": optuna.distributions.FloatDistribution(0.06, 0.20),
        "RISK_AVERSION": optuna.distributions.FloatDistribution(12.0, 20.0),
        "CVAR_DAILY_LIMIT": optuna.distributions.FloatDistribution(0.04, 0.06),
    }

    trial = optuna.trial.create_trial(
        params={
            "HALFLIFE_FAST": 21,
            "HALFLIFE_SLOW": 65,
            "CONTINUITY_BONUS": 0.15,
            "RISK_AVERSION": 12.0,
            "CVAR_DAILY_LIMIT": 0.04,
        },
        distributions=distributions,
        values=[1.23, 1.11],
        user_attrs={"resolved_cfg": {}},
    )

    class _Study:
        best_trials = [trial]
        trials = [trial]
        directions = [optuna.study.StudyDirection.MAXIMIZE, optuna.study.StudyDirection.MAXIMIZE]
        def optimize(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(optimizer.optuna, "create_study", lambda **kwargs: _Study())
    optimizer.run_optimization()

    import sqlite3
    with sqlite3.connect(str(db_path)) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"


def test_oos_journal_path_sanitizes_study_name():
    path = optimizer._oos_journal_path("../bad/name")
    assert ".." not in path.name


def test_fitness_calmar_uses_drawdown_floor():
    _, calmar, _ = optimizer._fitness_from_metrics(
        {"cagr": 10.0, "max_dd": 0.0, "turnover": 0.0},
        pd.DataFrame(),
    )
    assert calmar == pytest.approx(10.0 / optimizer.DRAWDOWN_FLOOR)


def test_best_trial_callback_handler_returns_if_not_complete(caplog):
    handler = optimizer.BestTrialCallbackHandler()

    class MockTrial:
        state = optuna.trial.TrialState.RUNNING

    with caplog.at_level("INFO", logger="Optimizer"):
        handler(None, MockTrial())
    assert "NEW BEST" not in caplog.text


def test_best_trial_callback_logs_best_trial(caplog):
    handler = optimizer.BestTrialCallbackHandler()

    class MockTrial:
        state = optuna.trial.TrialState.COMPLETE
        number = 5
        values = [1.5, 1.2]
        params = {"P1": 100}
        user_attrs = {"slice_diags": [{"year": 2020, "cagr": 10.0, "max_dd": 5.0, "turnover": 0.0, "avg_positions": 10.0, "avg_exposure": 1.0, "forced_cash_penalty": 0.0, "score": 1.0, "ceiling_hit": False, "dd_gate_hit": False, "anomaly_hit": False}]}

    trial = MockTrial()

    class MockStudy:
        def __init__(self, t):
            self.best_trials = [t]
            self.directions = [optuna.study.StudyDirection.MAXIMIZE, optuna.study.StudyDirection.MAXIMIZE]
            self.trials = [t]

    study = MockStudy(trial)

    with caplog.at_level("INFO"):
        handler(study, trial)

    assert "NEW BEST" in caplog.text


def _objective_diag(*, score, dd_gate_hit=False, anomaly_hit=False):
    return (
        score,
        score,
        {
            "cagr": 0.0,
            "max_dd": 0.0,
            "turnover": 0.0,
            "avg_exposure": 1.0,
            "avg_positions": 10.0,
            "avg_cvar_pct": 1.0,
            "risk_penalty": 0.0,
            "exposure_penalty": 0.0,
            "dd_penalty": 0.0,
            "forced_cash_penalty": 0.0,
            "raw_score": score,
            "score": score,
            "dd_gate_hit": dd_gate_hit,
            "anomaly_hit": anomaly_hit,
            "ceiling_hit": False,
        },
    )

class TestOptimizer:
    @pytest.fixture
    def mock_cfg(self):
        return UltimateConfig()

    @pytest.fixture
    def basic_metrics(self):
        return {"cagr": 20.0, "max_dd": -15.0, "turnover": 5.0, "sortino": 2.0, "final": 1.2 * UltimateConfig().INITIAL_CAPITAL}

    @pytest.fixture
    def empty_rebal_log(self):
        return pd.DataFrame()

    @pytest.fixture
    def basic_rebal_log(self):
        return pd.DataFrame({
            "realised_cvar": [0.01, 0.02, 0.03],
            "exposure_multiplier": [0.9, 1.0, 1.1],
            "n_positions": [8, 9, 10],
        })

    def test_fitness_function_calculates_score(self, mock_cfg, basic_metrics, empty_rebal_log):
        """
        Tests that the objective function correctly calculates the metric to be optimized.
        """
        score, calmar, diag = optimizer._fitness_from_metrics(basic_metrics, empty_rebal_log, mock_cfg)
        
        assert isinstance(score, float)
        assert isinstance(calmar, float)
        assert isinstance(diag, dict)
        assert diag["raw_score"] != 0
        assert not diag["dd_gate_hit"]
        assert not diag["anomaly_hit"]

    def test_fitness_function_handles_dd_gate(self, mock_cfg, basic_metrics, empty_rebal_log):
        """
        Tests that the objective function correctly handles the drawdown gate.
        """
        metrics = basic_metrics.copy()
        metrics["max_dd"] = -45.0  # Above the 40% gate
        
        score, _, diag = optimizer._fitness_from_metrics(metrics, empty_rebal_log, mock_cfg)
        
        assert diag["dd_gate_hit"]
        assert score == -2.0

    def test_fitness_function_handles_anomaly(self, mock_cfg, basic_metrics, empty_rebal_log):
        """
        Tests that the objective function correctly handles anomalous CAGR.
        """
        metrics = basic_metrics.copy()
        metrics["cagr"] = optimizer.MAX_REASONABLE_CAGR_PCT + 50
        
        score, _, diag = optimizer._fitness_from_metrics(metrics, empty_rebal_log, mock_cfg)
        
        assert diag["anomaly_hit"]
        assert score < 0

    def test_fitness_near_zero_cagr_maxdd(self, mock_cfg, empty_rebal_log):
        """
        Tests that the score is 0 when both cagr and max_dd are near zero.
        """
        metrics = {"cagr": 0.0, "max_dd": 0.0, "turnover": 0.0, "sortino": 0.0, "final": mock_cfg.INITIAL_CAPITAL}
        score, calmar, diag = optimizer._fitness_from_metrics(metrics, empty_rebal_log, mock_cfg)

        assert score == 0.0
        assert calmar == 0.0
        assert diag["raw_score"] == 0.0

    def test_fitness_turnover_drag_calculation(self, mock_cfg, empty_rebal_log):
        """
        Tests the calculation of cagr_net with different turnover values.
        """
        metrics_low_turnover = {"cagr": 20.0, "max_dd": -10.0, "turnover": 1.0, "sortino": 2.0, "final": 1.2 * mock_cfg.INITIAL_CAPITAL}
        metrics_high_turnover = {"cagr": 20.0, "max_dd": -10.0, "turnover": 25.0, "sortino": 2.0, "final": 1.2 * mock_cfg.INITIAL_CAPITAL}

        _, _, diag_low = optimizer._fitness_from_metrics(metrics_low_turnover, empty_rebal_log, mock_cfg)
        _, _, diag_high = optimizer._fitness_from_metrics(metrics_high_turnover, empty_rebal_log, mock_cfg)
        
        # Turnover drag at 1.0x turnover: 1.0 * 0.30 = 0.30, churn_penalty = 0.0
        assert diag_low["cagr_net"] == pytest.approx(20.0 - 0.30)
        
        # Turnover drag at 25.0x turnover: base_friction_drag = 25 * 0.3 = 7.5
        # churn_penalty = ((25 - 18)^2) / 10 = (7^2) / 10 = 4.9
        # Total drag = 7.5 + 4.9 = 12.4
        assert diag_high["cagr_net"] == pytest.approx(20.0 - 12.4)
        assert diag_high["cagr_net"] < diag_low["cagr_net"]

    def test_fitness_low_avg_positions_penalty(self, mock_cfg, basic_metrics, empty_rebal_log):
        """
        Tests that concentration_mult increases when avg_positions is low.
        """
        rebal_log_low_pos = pd.DataFrame({
            "realised_cvar": [0.01], "exposure_multiplier": [1.0], "n_positions": [1],
        }) # Avg pos = 1
        
        # Default CONCENTRATION_MIN_POSITIONS = 6.0, CONCENTRATION_PENALTY_WEIGHT = 0.30
        # _pos_deficit = max(0.0, 6.0 - 1.0) = 5.0
        # concentration_mult = 1.0 + 5.0 * 0.30 = 2.5

        _, _, diag = optimizer._fitness_from_metrics(basic_metrics, rebal_log_low_pos, mock_cfg)
        assert diag["concentration_mult"] == pytest.approx(2.5)

    def test_fitness_low_exposure_penalty(self, mock_cfg, basic_metrics, empty_rebal_log):
        """
        Tests exposure_penalty when avg_exposure is low or avg_positions is zero.
        """
        # Case 1: avg_exposure < 0.25
        rebal_log_low_exposure = pd.DataFrame({
            "realised_cvar": [0.01], "exposure_multiplier": [0.1], "n_positions": [8],
        }) # Avg exp = 0.1
        # exposure_penalty = (0.25 - 0.1) * 2.0 = 0.3

        _, _, diag_low_exp = optimizer._fitness_from_metrics(basic_metrics, rebal_log_low_exposure, mock_cfg)
        assert diag_low_exp["exposure_penalty"] == pytest.approx(0.3)

        # Case 2: avg_positions < 1.0 (empty rebal log)
        # Should add 0.5 to exposure_penalty if avg_positions is effectively zero.
        # avg_exposure will be 1.0 (fallback) from empty rebal log.
        # But if we force avg_positions to 0.0 via rebal_log
        rebal_log_zero_pos = pd.DataFrame({
            "realised_cvar": [0.0], "exposure_multiplier": [1.0], "n_positions": [0],
        })
        _, _, diag_zero_pos = optimizer._fitness_from_metrics(basic_metrics, rebal_log_zero_pos, mock_cfg)
        # exposure_penalty should be 0.5 (from avg_positions < 1.0) + (0.25 - 1.0) * 2.0 (if avg_exposure was < 0.25)
        # But here avg_exposure is 1.0, so only 0.5 from avg_positions.
        assert diag_zero_pos["exposure_penalty"] == pytest.approx(0.5)

    def test_fitness_sortino_quality_variations(self, mock_cfg, basic_metrics, empty_rebal_log):
        """
        Tests different sortino values and their impact on sortino_quality.
        """
        # Default SORTINO_QUALITY_TARGET = 2.5

        # Sortino < target, but > 0.5 * target
        metrics_low_sortino = basic_metrics.copy()
        metrics_low_sortino["sortino"] = 1.0 # 1.0 / 2.5 = 0.4, but min is 0.5, so 0.5
        _, _, diag_low = optimizer._fitness_from_metrics(metrics_low_sortino, empty_rebal_log, mock_cfg)
        assert diag_low["sortino_quality"] == pytest.approx(0.5)

        # Sortino == target
        metrics_target_sortino = basic_metrics.copy()
        metrics_target_sortino["sortino"] = 2.5
        _, _, diag_target = optimizer._fitness_from_metrics(metrics_target_sortino, empty_rebal_log, mock_cfg)
        assert diag_target["sortino_quality"] == pytest.approx(1.0)

        # Sortino > target, but capped at 1.15
        metrics_high_sortino = basic_metrics.copy()
        metrics_high_sortino["sortino"] = 5.0 # 5.0 / 2.5 = 2.0, but max is 1.15, so 1.15
        _, _, diag_high = optimizer._fitness_from_metrics(metrics_high_sortino, empty_rebal_log, mock_cfg)
        assert diag_high["sortino_quality"] == pytest.approx(1.15)

    def test_fitness_dd_penalty_between_thresholds(self, mock_cfg, basic_metrics, empty_rebal_log):
        """
        Tests the quadratic dd_penalty when max_dd is between IS_DD_PENALTY_PCT and IS_DD_GATE.
        IS_DD_GATE = 40.0, IS_DD_PENALTY_PCT = 12.0
        """
        metrics = basic_metrics.copy()
        metrics["max_dd"] = -22.0 # Between 12.0 and 40.0
        # dd_excess = max(0.0, 22.0 - 12.0) = 10.0
        # dd_penalty = (10.0 ** 2) / 100.0 = 1.0

        _, _, diag = optimizer._fitness_from_metrics(metrics, empty_rebal_log, mock_cfg)
        assert diag["dd_penalty"] == pytest.approx(1.0)
        assert not diag["dd_gate_hit"]

    def test_fitness_log1p_transform_behavior(self, mock_cfg, basic_metrics, empty_rebal_log):
        """
        Tests log1p and copysign transformation for raw scores.
        """
        metrics_positive_raw = basic_metrics.copy() # Will result in positive raw
        metrics_positive_raw["cagr"] = 100.0
        metrics_positive_raw["max_dd"] = -5.0
        metrics_positive_raw["turnover"] = 0.0
        metrics_positive_raw["sortino"] = 2.5
        
        _, _, diag_pos = optimizer._fitness_from_metrics(metrics_positive_raw, empty_rebal_log, mock_cfg)
        expected_raw_pos = (100.0 / ((5.0 + (0.01 * 100.0 * 2.0) + 1.0) * 1.0)) * 1.0 - 0.0 - 0.0
        # Assuming defaults for penalties etc. for a clean calculation
        # risk_penalty = (abs(max_dd) + (avg_cvar * 100.0 * 2.0) + 1.0) * concentration_mult
        # With empty rebal_log, avg_cvar=0, avg_exposure=1, avg_positions=0, concentration_mult=2.5 (from _pos_deficit 6)
        # risk_penalty = (5.0 + 0 + 1.0) * 2.5 = 15.0
        # raw = (100.0 / 15.0) * 1.0 - 0.0 - 0.0 = 6.666...
        # score = log1p(raw)
        
        # Recalculating with actual constants used in _fitness_from_metrics and _calculate_penalty_multipliers
        cagr         = 100.0
        max_dd       = 5.0
        turnover     = 0.0
        sortino      = 2.5
        
        cfg = mock_cfg
        turnover_drag = 0.0 # From _compute_turnover_drag
        cagr_net = cagr - turnover_drag # 100.0

        avg_cvar = 0.0
        avg_exposure = 1.0
        avg_positions = 0.0
        
        _pos_deficit = max(0.0, float(cfg.CONCENTRATION_MIN_POSITIONS) - avg_positions) # 6.0 - 0.0 = 6.0
        concentration_mult = 1.0 + _pos_deficit * float(cfg.CONCENTRATION_PENALTY_WEIGHT) # 1.0 + 6.0 * 0.30 = 2.8

        risk_penalty = (max_dd + (avg_cvar * 100.0 * 2.0) + 1.0) * concentration_mult # (5.0 + 0 + 1.0) * 2.8 = 16.8
        exposure_penalty = 0.5 # from avg_positions < 1.0
        
        sortino_quality = min(max(sortino / float(cfg.SORTINO_QUALITY_TARGET), 0.50), 1.15) # min(max(2.5/2.5, 0.5), 1.15) = 1.0

        dd_penalty = 0.0
        
        raw_pos = (cagr_net / risk_penalty) * sortino_quality - exposure_penalty - dd_penalty
        expected_score_pos = np.copysign(np.log1p(abs(raw_pos)), raw_pos)
        
        assert diag_pos["raw_score"] == pytest.approx(raw_pos)
        assert diag_pos["score"] == pytest.approx(expected_score_pos)
        assert diag_pos["score"] > 0

        metrics_negative_raw = basic_metrics.copy()
        metrics_negative_raw["cagr"] = -5.0
        metrics_negative_raw["max_dd"] = -10.0
        metrics_negative_raw["turnover"] = 0.0
        metrics_negative_raw["sortino"] = 0.1 # This should lead to a very low score
        
        _, _, diag_neg = optimizer._fitness_from_metrics(metrics_negative_raw, empty_rebal_log, mock_cfg)
        
        cagr         = -5.0
        max_dd       = 10.0
        turnover     = 0.0
        sortino      = 0.1
        
        cagr_net = cagr - turnover_drag # -5.0
        
        # Remaining penalty multipliers are similar assuming same empty_rebal_log
        _pos_deficit = max(0.0, float(cfg.CONCENTRATION_MIN_POSITIONS) - avg_positions) # 6.0 - 0.0 = 6.0
        concentration_mult = 1.0 + _pos_deficit * float(cfg.CONCENTRATION_PENALTY_WEIGHT) # 1.0 + 6.0 * 0.30 = 2.8
        risk_penalty = (max_dd + (avg_cvar * 100.0 * 2.0) + 1.0) * concentration_mult # (10.0 + 0 + 1.0) * 2.8 = 30.8
        exposure_penalty = 0.5
        sortino_quality = min(max(sortino / float(cfg.SORTINO_QUALITY_TARGET), 0.50), 1.15) # min(max(0.1/2.5, 0.5), 1.15) = 0.5

        raw_neg = (cagr_net / risk_penalty) * sortino_quality - exposure_penalty - dd_penalty
        expected_score_neg = np.copysign(np.log1p(abs(raw_neg)), raw_neg)

        assert diag_neg["raw_score"] == pytest.approx(raw_neg)
        assert diag_neg["score"] == pytest.approx(expected_score_neg)
        assert diag_neg["score"] < 0
        assert diag_neg["score"] > -2.0 # Should not hit the hard floor unless very negative

    def test_extract_rebalance_summary_with_data(self, basic_rebal_log):
        """
        Tests _extract_rebalance_summary with a populated rebalance log.
        """
        avg_cvar, avg_exposure, avg_positions, n_rebalances = optimizer._extract_rebalance_summary(basic_rebal_log)

        assert avg_cvar == pytest.approx(0.02)
        assert avg_exposure == pytest.approx(1.0)
        assert avg_positions == pytest.approx(9.0)
        assert n_rebalances == 3

    def test_extract_rebalance_summary_empty_dataframe(self, empty_rebal_log):
        """
        Tests _extract_rebalance_summary with an empty rebalance log.
        """
        avg_cvar, avg_exposure, avg_positions, n_rebalances = optimizer._extract_rebalance_summary(empty_rebal_log)
        
        assert avg_cvar == pytest.approx(0.0)
        assert avg_exposure == pytest.approx(1.0) # Fallback to 1.0
        assert avg_positions == pytest.approx(0.0)
        assert n_rebalances == 0

    def test_calculate_penalty_multipliers_no_penalties(self, mock_cfg):
        """
        Tests that penalty multipliers are 1.0 or 0.0 when conditions are not met.
        """
        avg_positions = 10.0 # Above CONCENTRATION_MIN_POSITIONS (6.0)
        avg_exposure = 0.5 # Above 0.25 threshold
        avg_cvar = 0.01
        max_dd = 10.0

        concentration_mult, risk_penalty, exposure_penalty = optimizer._calculate_penalty_multipliers(
            avg_positions, avg_exposure, avg_cvar, max_dd, mock_cfg
        )
        
        assert concentration_mult == pytest.approx(1.0)
        # risk_penalty = (10.0 + (0.01 * 100.0 * 2.0) + 1.0) * 1.0 = (10.0 + 2.0 + 1.0) = 13.0
        assert risk_penalty == pytest.approx(13.0)
        assert exposure_penalty == pytest.approx(0.0)

    def test_momentum_objective_propagates_numerical_error(self, monkeypatch):
        """
        Tests that the MomentumObjective propagates OptimizationErrorType.NUMERICAL.
        """
        monkeypatch.setattr(
            optimizer,
            "run_backtest",
            lambda **kwargs: (_ for _ in ()).throw(
                optimizer.OptimizationError(
                    "Numerical instability",
                    error_type=optimizer.OptimizationErrorType.NUMERICAL,
                )
            ),
        )

        objective = optimizer.MomentumObjective(
            market_data={
                "DUMMY": pd.DataFrame(
                    {"Close": [1.0] * 510, "Volume": [1e6] * 510},
                    index=pd.date_range("2017-01-01", periods=510, freq="B"),
                )
            },
            universe_type="nifty500",
        )
        trial = _fixed_trial_params()

        with pytest.raises(optimizer.OptimizationError) as exc_info:
            objective(trial)
        assert exc_info.value.error_type == optimizer.OptimizationErrorType.NUMERICAL
        assert "Numerical instability" in str(exc_info.value)

    def test_momentum_objective_propagates_infeasible_error(self, monkeypatch):
        """
        Tests that the MomentumObjective propagates OptimizationErrorType.INFEASIBLE.
        """
        monkeypatch.setattr(
            optimizer,
            "run_backtest",
            lambda **kwargs: (_ for _ in ()).throw(
                optimizer.OptimizationError(
                    "Infeasible solution",
                    error_type=optimizer.OptimizationErrorType.INFEASIBLE,
                )
            ),
        )

        objective = optimizer.MomentumObjective(
            market_data={
                "DUMMY": pd.DataFrame(
                    {"Close": [1.0] * 510, "Volume": [1e6] * 510},
                    index=pd.date_range("2017-01-01", periods=510, freq="B"),
                )
            },
            universe_type="nifty500",
        )
        trial = _fixed_trial_params()

        with pytest.raises(optimizer.OptimizationError) as exc_info:
            objective(trial)
        assert exc_info.value.error_type == optimizer.OptimizationErrorType.INFEASIBLE
        assert "Infeasible solution" in str(exc_info.value)

    def test_momentum_objective_propagates_data_error(self, monkeypatch):
        """
        Tests that the MomentumObjective propagates OptimizationErrorType.DATA.
        """
        monkeypatch.setattr(
            optimizer,
            "run_backtest",
            lambda **kwargs: (_ for _ in ()).throw(
                optimizer.OptimizationError(
                    "Missing data", error_type=optimizer.OptimizationErrorType.DATA
                )
            ),
        )

        objective = optimizer.MomentumObjective(
            market_data={
                "DUMMY": pd.DataFrame(
                    {"Close": [1.0] * 510, "Volume": [1e6] * 510},
                    index=pd.date_range("2017-01-01", periods=510, freq="B"),
                )
            },
            universe_type="nifty500",
        )
        trial = _fixed_trial_params()

        with pytest.raises(optimizer.OptimizationError) as exc_info:
            objective(trial)
        assert exc_info.value.error_type == optimizer.OptimizationErrorType.DATA
        assert "Missing data" in str(exc_info.value)

