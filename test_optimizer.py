import importlib
import json
import numbers
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")
optimizer = pytest.importorskip("optimizer")
from momentum_engine import InstitutionalRiskEngine, UltimateConfig


VALID_OPTUNA_TRIAL_PARAMS = {
    "HALFLIFE_FAST": 20,
    "HALFLIFE_SLOW": 60,
    "CONTINUITY_BONUS": 0.15,
    "RISK_AVERSION": 10.0,
    "CVAR_DAILY_LIMIT": 0.04,
}


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
    trial = optuna.trial.FixedTrial(VALID_OPTUNA_TRIAL_PARAMS)

    assert objective(trial) == 0.0


def test_objective_propagates_optimization_error(monkeypatch):
    monkeypatch.setattr(
        optimizer,
        "run_backtest",
        lambda **kwargs: (_ for _ in ()).throw(
            optimizer.OptimizationError("Solver failed")
        ),
    )

    objective = optimizer.MomentumObjective(market_data={}, universe_type="nifty500")
    trial = optuna.trial.FixedTrial(VALID_OPTUNA_TRIAL_PARAMS)

    with pytest.raises(optimizer.OptimizationError, match="Solver failed"):
        objective(trial)


def test_objective_propagates_unexpected_errors(monkeypatch):
    monkeypatch.setattr(
        optimizer,
        "run_backtest",
        lambda **kwargs: (_ for _ in ()).throw(TypeError("bad type")),
    )

    objective = optimizer.MomentumObjective(market_data={}, universe_type="nifty500")
    trial = optuna.trial.FixedTrial(VALID_OPTUNA_TRIAL_PARAMS)

    with pytest.raises(TypeError, match="bad type"):
        objective(trial)


def test_objective_returns_numeric_score_without_hard_drawdown_prune(monkeypatch):
    class _Result:
        metrics = {"cagr": 10.0, "max_dd": 30.0, "turnover": 0.0}
        rebal_log = None

    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    objective = optimizer.MomentumObjective(market_data={}, universe_type="nifty500")
    trial = optuna.trial.FixedTrial(VALID_OPTUNA_TRIAL_PARAMS)

    assert isinstance(objective(trial), numbers.Real)


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
    # apply_halt_simulation is a no-op for this test — it would crash because the
    # fake load_or_fetch returns {"ok": True} (not real DataFrames).
    monkeypatch.setattr(optimizer, "apply_halt_simulation", lambda md: md)

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


def test_pre_load_data_includes_historical_union_for_nifty500(monkeypatch):
    monkeypatch.setattr(optimizer, "TRAIN_START", "2020-01-01")
    monkeypatch.setattr(optimizer, "TEST_END", "2020-03-31")
    monkeypatch.setattr(optimizer, "get_nifty500", lambda: ["LIVEONLY"])
    monkeypatch.setattr(optimizer, "apply_halt_simulation", lambda md: md)

    def _fake_hist(universe_type, date):
        assert universe_type == "nifty500"
        if pd.Timestamp(date) == pd.Timestamp("2020-01-31"):
            return ["OLD1", "OLD2"]
        if pd.Timestamp(date) == pd.Timestamp("2020-02-29"):
            return ["OLD2", "OLD3"]
        return []

    monkeypatch.setattr(optimizer, "get_historical_universe", _fake_hist)

    captured = {}

    def _fake_load_or_fetch(*, tickers, required_start, required_end, cfg=None):
        captured["tickers"] = tickers
        captured["required_start"] = required_start
        captured["required_end"] = required_end
        return {"ok": True}

    monkeypatch.setattr(optimizer, "load_or_fetch", _fake_load_or_fetch)

    result = optimizer.pre_load_data("nifty500")

    assert result == {"ok": True}
    assert captured["required_start"] == "2020-01-01"
    assert captured["required_end"] == "2020-03-31"
    assert "LIVEONLY" in captured["tickers"]
    assert "OLD1" in captured["tickers"]
    assert "OLD2" in captured["tickers"]
    assert "OLD3" in captured["tickers"]
    assert "^NSEI" in captured["tickers"]
    assert "^CRSLDX" in captured["tickers"]


def test_build_sampler_returns_tpe_sampler(monkeypatch):
    monkeypatch.setattr(optimizer, "OPTUNA_SEED", None)
    sampler_unseeded = optimizer._build_sampler()

    monkeypatch.setattr(optimizer, "OPTUNA_SEED", "123")
    sampler_seeded = optimizer._build_sampler()

    assert isinstance(sampler_unseeded, optimizer.TPESampler)
    assert isinstance(sampler_seeded, optimizer.TPESampler)


def test_fitness_from_metrics_handles_missing_rebalance_columns_without_crash():
    metrics = {
        "cagr": 12.0,
        "max_dd": 10.0,
        "turnover": 5.0,
        "sortino": 1.2,
        "final": optimizer.BASE_INITIAL_CAPITAL * 1.1,
    }
    # Missing realised_cvar / exposure_multiplier / n_positions columns used in
    # diagnostics extraction. Function must use a Series fallback and not raise.
    rebal_log = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2, freq="D")})

    score, diag = optimizer._fitness_from_metrics(metrics, rebal_log)

    assert isinstance(score, float)
    assert diag["avg_cvar_pct"] == pytest.approx(0.0)
    assert diag["avg_exposure"] == pytest.approx(0.0)
    assert diag["avg_positions"] == pytest.approx(0.0)




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

    # With no rebalance log: avg_positions=0 triggers concentration and min sortino quality.
    # raw = (cagr / ((max_dd + 1) * concentration_mult)) * sortino_quality - exposure_penalty
    #     = (10 / ((5 + 1) * 2.8)) * 0.5 - 0.5
    expected = (10.0 / ((5.0 + 1.0) * 2.8)) * 0.5 - 0.5
    assert round(objective(trial), 6) == round(expected, 6)


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
        market_data={}, universe_type="nifty500", search_space=custom_space
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

    assert isinstance(objective(trial), numbers.Real)


def test_run_optimization_forces_single_job(monkeypatch):
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

    assert captured["n_jobs"] == 1


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

    custom_space = {
        "HALFLIFE_FAST": (20, 20, 5),
        "HALFLIFE_SLOW": (20, 20, 5),
        "CONTINUITY_BONUS": (0.15, 0.15, 0.01),
        "RISK_AVERSION": (10.0, 10.0, 0.5),
        "CVAR_DAILY_LIMIT": (0.04, 0.04, 0.005),
    }
    objective = optimizer.MomentumObjective(
        market_data={}, universe_type="nifty500", search_space=custom_space
    )
    trial = optuna.trial.FixedTrial(
        {
            "HALFLIFE_FAST": 20,
            "HALFLIFE_SLOW": 20,
            "CONTINUITY_BONUS": 0.15,
            "RISK_AVERSION": 10.0,
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


def test_objective_cvar_lookback_min_scales_with_dimensionality(monkeypatch):
    class _Result:
        metrics = {"cagr": 10.0, "max_dd": 10.0, "turnover": 0.0}
        rebal_log = None

    class _DummyCfg:
        HALFLIFE_FAST = 21
        HALFLIFE_SLOW = 63
        CONTINUITY_BONUS = 0.15
        RISK_AVERSION = 5.0
        CVAR_DAILY_LIMIT = 0.04
        CVAR_LOOKBACK = 60
        DIMENSIONALITY_MULTIPLIER = 3
        MAX_POSITIONS = 30

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
    monkeypatch.setattr(optimizer, "run_backtest", lambda **kwargs: _Result())

    objective = optimizer.MomentumObjective(market_data={}, universe_type="nifty500")
    trial = _RecordingTrial()

    objective(trial)

    assert trial.bounds["CVAR_LOOKBACK"][0] == 90


def test_objective_prunes_when_cvar_lookback_bounds_are_infeasible(monkeypatch):
    class _DummyCfg:
        HALFLIFE_FAST = 21
        HALFLIFE_SLOW = 63
        CONTINUITY_BONUS = 0.15
        RISK_AVERSION = 5.0
        CVAR_DAILY_LIMIT = 0.04
        CVAR_LOOKBACK = 60
        DIMENSIONALITY_MULTIPLIER = 3
        MAX_POSITIONS = 60

    monkeypatch.setattr(optimizer, "UltimateConfig", _DummyCfg)

    objective = optimizer.MomentumObjective(
        market_data={},
        universe_type="nifty500",
        search_space={
            "HALFLIFE_FAST": (10, 40),
            "HALFLIFE_SLOW": (50, 120),
            "CONTINUITY_BONUS": (0.05, 0.30, 0.01),
            "RISK_AVERSION": (5.0, 15.0, 0.5),
            "CVAR_DAILY_LIMIT": (0.04, 0.09, 0.005),
            "CVAR_LOOKBACK": (60, 150, 10),
        },
    )
    trial = optuna.trial.FixedTrial(
        {
            "HALFLIFE_FAST": 21,
            "HALFLIFE_SLOW": 63,
            "CONTINUITY_BONUS": 0.15,
            "RISK_AVERSION": 5.0,
            "CVAR_DAILY_LIMIT": 0.04,
            "CVAR_LOOKBACK": 150,
        }
    )

    with pytest.raises(optuna.TrialPruned):
        objective(trial)
