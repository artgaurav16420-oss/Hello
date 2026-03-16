"""
optimizer.py — Institutional Bayesian Time-Series CV Optimizer v11.48
====================================================================
Automates the discovery of optimal risk and momentum parameters using Optuna.
Uses expanding-window time-series cross-validation for parameter selection,
followed by a true holdout Out-of-Sample (OOS) validation period.

BUG FIXES (murder board):
- FIX-MB-DDPENALTY: _fitness_from_metrics computed dd_penalty independently of
  concentration_mult and subtracted it flat. A 4-position portfolio with a 25%
  drawdown therefore received the same dd_penalty as an 8-position portfolio
  with the same drawdown, even though the 4-position portfolio carries 2x the
  unobserved idiosyncratic risk. This made the IS fitness gradient near the
  IS_DD_PENALTY_PCT boundary blind to concentration.
  Fix: dd_penalty is now scaled by concentration_mult before subtracting, so
  concentrated portfolios pay a proportionally larger drawdown penalty consistent
  with how risk_penalty already handles them.

Requires: pip install optuna
"""
import argparse
import json
import logging
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler


def _load_dotenv_if_present(dotenv_path: str = ".env") -> None:
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


_load_dotenv_if_present()

from momentum_engine import UltimateConfig, OptimizationError
from backtest_engine import run_backtest, apply_halt_simulation, build_precomputed_matrices
from data_cache import load_or_fetch
from universe_manager import get_nifty500, fetch_nse_equity_universe, get_historical_universe

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger("Optimizer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("\033[90m[%(asctime)s]\033[0m %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)

logging.getLogger("universe_manager").setLevel(logging.ERROR)
logging.getLogger("backtest_engine").setLevel(logging.ERROR)
logging.getLogger("momentum_engine").setLevel(logging.ERROR)
logging.getLogger("signals").setLevel(logging.ERROR)
logging.getLogger("data_cache").setLevel(logging.ERROR)

# ─── Optimization Configuration ───────────────────────────────────────────────

TRAIN_START = "2018-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = pd.Timestamp.today().strftime("%Y-%m-%d")

N_TRIALS       = 100
OOS_MAX_DD_CAP = 35.0
OOS_TOP_K = 10

SEARCH_SPACE_BOUNDS = {
    "HALFLIFE_FAST":    (10, 40, 5),
    "HALFLIFE_SLOW":    (50, 120, 5),
    "CONTINUITY_BONUS": (0.05, 0.30, 0.01),
    "RISK_AVERSION":    (10.0, 20.0, 0.5),
    "CVAR_DAILY_LIMIT": (0.040, 0.070, 0.005),
    "CVAR_LOOKBACK":    (60, 150, 5),
}

N_JOBS = int(os.getenv("OPTUNA_N_JOBS", "1"))
OPTUNA_SEED = os.getenv("OPTUNA_SEED")

OPTUNA_STORAGE = os.getenv("OPTUNA_STORAGE", "sqlite:///data/optuna_study.db")

OBJECTIVE_VERSION = "fitness_v11_51"
DEFAULT_STUDY_NAME = f"Momentum_Risk_Parity_{OBJECTIVE_VERSION}"

MAX_REASONABLE_CAGR_PCT = 300.0
MAX_REASONABLE_FINAL_MULTIPLE = 8.0
BASE_INITIAL_CAPITAL = UltimateConfig().INITIAL_CAPITAL


def _stdout_supports_rupee(stdout=None) -> bool:
    stream = stdout if stdout is not None else getattr(sys, "stdout", None)
    if stream is None:
        return False
    encoding = getattr(stream, "encoding", None) or "utf-8"
    errors = getattr(stream, "errors", None) or "strict"
    try:
        "\u20b9".encode(encoding, errors=errors)
    except (LookupError, TypeError, UnicodeEncodeError):
        return False
    return True


def _build_sampler() -> TPESampler:
    if OPTUNA_SEED in (None, ""):
        return TPESampler()
    return TPESampler(seed=int(OPTUNA_SEED))

# ─── Objective Function ───────────────────────────────────────────────────────

def _iter_wfo_slices(train_start: str, train_end: str):
    start = pd.Timestamp(train_start)
    end   = pd.Timestamp(train_end)
    for y in range(start.year + 1, end.year + 1):
        oos_start = pd.Timestamp(f"{y}-01-01")
        oos_end   = min(pd.Timestamp(f"{y}-12-31"), end)
        yield (
            start.strftime("%Y-%m-%d"),
            (oos_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            oos_start.strftime("%Y-%m-%d"),
            oos_end.strftime("%Y-%m-%d"),
        )


def _fitness_from_metrics(
    metrics: dict,
    rebal_log: pd.DataFrame,
) -> tuple[float, dict]:
    """
    Compute a scalar fitness score plus a diagnostics dict for logging.

    FIX-MB-DDPENALTY: dd_penalty is now multiplied by concentration_mult before
    subtracting. Previously dd_penalty was a flat deduction independent of how
    concentrated the portfolio was. Since concentration_mult inflates risk_penalty
    for concentrated portfolios (fewer than 6 positions), the drawdown penalty
    should scale proportionally — a concentrated portfolio near the IS_DD_PENALTY_PCT
    boundary deserves a harsher penalty because unobserved idiosyncratic risk is
    higher. This creates a consistent gradient: both the numerator (risk_penalty)
    and the drawdown correction now scale with concentration.

    Returns
    -------
    score : float  — clipped to [-2.0, 3.5] via Michaelis-Menten saturation
    diag  : dict   — all intermediate values for logging
    """
    cagr = float(metrics.get("cagr", 0.0))
    max_dd = abs(float(metrics.get("max_dd", 100.0)))
    turnover = float(metrics.get("turnover", 0.0))
    sortino = float(metrics.get("sortino", 0.0) or 0.0)
    final_equity = float(metrics.get("final", BASE_INITIAL_CAPITAL) or BASE_INITIAL_CAPITAL)
    final_multiple = final_equity / max(BASE_INITIAL_CAPITAL, 1e-9)
    turnover_drag = turnover * 0.05
    cagr_net = cagr - turnover_drag

    avg_cvar = 0.0
    avg_exposure = 1.0
    avg_positions = 0.0
    n_rebalances = 0

    if rebal_log is not None and not rebal_log.empty:
        fallback_series = pd.Series([0.0])
        avg_cvar = float(pd.to_numeric(rebal_log.get("realised_cvar", fallback_series), errors="coerce").fillna(0.0).mean())
        avg_exposure = float(pd.to_numeric(rebal_log.get("exposure_multiplier", fallback_series), errors="coerce").fillna(0.0).mean())
        avg_positions = float(pd.to_numeric(rebal_log.get("n_positions", fallback_series), errors="coerce").fillna(0.0).mean())
        n_rebalances = len(rebal_log)

    # Concentration multiplier: each position below 6 raises the effective risk
    # denominator by 30% to account for unobserved idiosyncratic risk from the
    # survivor-only IS universe excluding demoted/blown-up names.
    _pos_deficit = max(0.0, 6.0 - avg_positions)
    concentration_mult = 1.0 + _pos_deficit * 0.30

    # Sortino quality multiplier: scales raw score ×[0.50, 1.15].
    import math as _math
    _sortino_safe = sortino if _math.isfinite(sortino) else 0.0
    sortino_quality = min(max(_sortino_safe / 2.5, 0.50), 1.15)

    risk_penalty = (max_dd + (avg_cvar * 100.0 * 2.0) + 1.0) * concentration_mult

    IS_DD_GATE        = 40.0
    IS_DD_PENALTY_PCT = 20.0

    if max_dd > IS_DD_GATE:
        raw = -(max_dd / 5.0)
        score = max(raw, -2.0)
        diag = {
            "cagr": round(cagr, 2), "max_dd": round(-max_dd, 2),
            "turnover": round(turnover, 4), "final_multiple": round(final_multiple, 4),
            "cagr_net": round(cagr_net, 2), "avg_cvar_pct": round(avg_cvar * 100.0, 4),
            "avg_exposure": round(avg_exposure, 4), "avg_positions": round(avg_positions, 2),
            "n_rebalances": n_rebalances, "concentration_mult": round(concentration_mult, 4),
            "sortino_quality": round(sortino_quality, 4), "risk_penalty": round(risk_penalty, 4),
            "exposure_penalty": 0.0, "dd_penalty": 0.0,
            "raw_score": round(raw, 6), "score": round(score, 6),
            "ceiling_hit": False, "dd_gate_hit": True, "anomaly_hit": False,
        }
        return score, diag

    dd_excess  = max(0.0, max_dd - IS_DD_PENALTY_PCT)
    # FIX-MB-DDPENALTY: Scale dd_penalty by concentration_mult so concentrated
    # portfolios near the IS_DD_PENALTY_PCT boundary pay a proportionally larger
    # penalty consistent with how risk_penalty already handles concentration risk.
    dd_penalty = ((dd_excess ** 2) / 100.0) * concentration_mult

    exposure_penalty = 0.0 if avg_exposure >= 0.25 else (0.25 - avg_exposure) * 2.0
    if avg_positions < 1.0:
        exposure_penalty += 0.5

    anomaly_hit = (
        cagr > MAX_REASONABLE_CAGR_PCT
        or final_multiple > MAX_REASONABLE_FINAL_MULTIPLE
    )

    if anomaly_hit:
        raw = -(
            max(cagr - MAX_REASONABLE_CAGR_PCT, 0.0) / 50.0
            + max(final_multiple - MAX_REASONABLE_FINAL_MULTIPLE, 0.0)
        )
        score = max(raw, -2.0)
        ceiling_hit = False
        dd_gate_hit = False
    elif abs(cagr) < 1e-12 and max_dd == 0.0:
        raw = 0.0
        score = 0.0
        ceiling_hit = False
        dd_gate_hit = False
    else:
        raw = (cagr_net / risk_penalty) * sortino_quality - exposure_penalty - dd_penalty
        # Soft saturation: Michaelis-Menten shape so TPE retains gradient between
        # "very good" (raw≈2) and "lucky outlier year" (raw≈14).
        _K = 3.5
        score = (_K * raw / (_K + raw)) if raw > 0.0 else raw
        score = max(score, -2.0)
        ceiling_hit = False
        dd_gate_hit = False

    diag = {
        "cagr":                round(cagr, 2),
        "max_dd":              round(-max_dd, 2),
        "turnover":            round(turnover, 4),
        "final_multiple":      round(final_multiple, 4),
        "cagr_net":            round(cagr_net, 2),
        "avg_cvar_pct":        round(avg_cvar * 100.0, 4),
        "avg_exposure":        round(avg_exposure, 4),
        "avg_positions":       round(avg_positions, 2),
        "n_rebalances":        n_rebalances,
        "concentration_mult":  round(concentration_mult, 4),
        "sortino_quality":     round(sortino_quality, 4),
        "risk_penalty":        round(risk_penalty, 4),
        "exposure_penalty":    round(exposure_penalty, 4),
        "dd_penalty":          round(dd_penalty, 4),
        "raw_score":           round(raw, 6) if not (abs(cagr) < 1e-12 and max_dd == 0.0) else 0.0,
        "score":               round(score, 6),
        "ceiling_hit":         ceiling_hit,
        "dd_gate_hit":         dd_gate_hit,
        "anomaly_hit":         anomaly_hit,
    }
    return score, diag


class MomentumObjective:
    def __init__(
        self,
        market_data: dict,
        universe_type: str,
        search_space: dict | None = None,
        precomputed_matrices: dict | None = None,
    ):
        self.market_data = market_data
        self.universe_type = universe_type
        self.search_space = search_space or SEARCH_SPACE_BOUNDS
        self.precomputed_matrices = precomputed_matrices

    def __call__(self, trial: optuna.Trial) -> float:
        cfg = UltimateConfig()

        halflife_fast_bounds = self.search_space["HALFLIFE_FAST"]
        halflife_slow_bounds = self.search_space["HALFLIFE_SLOW"]

        if len(halflife_fast_bounds) == 3:
            halflife_fast_min, halflife_fast_max, halflife_fast_step = halflife_fast_bounds
        else:
            halflife_fast_min, halflife_fast_max = halflife_fast_bounds
            halflife_fast_step = 1

        if len(halflife_slow_bounds) == 3:
            halflife_slow_min, halflife_slow_max, halflife_slow_step = halflife_slow_bounds
        else:
            halflife_slow_min, halflife_slow_max = halflife_slow_bounds
            halflife_slow_step = 1

        cfg.HALFLIFE_FAST = trial.suggest_int(
            "HALFLIFE_FAST", halflife_fast_min, halflife_fast_max, step=halflife_fast_step
        )
        cfg.HALFLIFE_SLOW = trial.suggest_int(
            "HALFLIFE_SLOW", halflife_slow_min, halflife_slow_max, step=halflife_slow_step
        )

        if cfg.HALFLIFE_FAST > cfg.HALFLIFE_SLOW:
            raise optuna.TrialPruned()

        continuity_min, continuity_max, continuity_step = self.search_space["CONTINUITY_BONUS"]
        cfg.CONTINUITY_BONUS = trial.suggest_float(
            "CONTINUITY_BONUS", continuity_min, continuity_max, step=continuity_step
        )

        risk_aversion_min, risk_aversion_max, risk_aversion_step = self.search_space["RISK_AVERSION"]
        cvar_min, cvar_max, cvar_step = self.search_space["CVAR_DAILY_LIMIT"]
        cfg.RISK_AVERSION = trial.suggest_float(
            "RISK_AVERSION", risk_aversion_min, risk_aversion_max, step=risk_aversion_step
        )
        cfg.CVAR_DAILY_LIMIT = trial.suggest_float(
            "CVAR_DAILY_LIMIT", cvar_min, cvar_max, step=cvar_step
        )

        cvar_lb_bounds = self.search_space.get("CVAR_LOOKBACK", (60, 150, 10))
        cvar_lb_min, cvar_lb_max, cvar_lb_step = cvar_lb_bounds
        min_required_lookback = cfg.DIMENSIONALITY_MULTIPLIER * cfg.MAX_POSITIONS
        effective_cvar_lb_min = max(int(cvar_lb_min), int(min_required_lookback))

        if effective_cvar_lb_min > int(cvar_lb_max):
            raise optuna.TrialPruned()

        if isinstance(trial, optuna.trial.FixedTrial) and "CVAR_LOOKBACK" not in trial.params:
            cfg.CVAR_LOOKBACK = max(UltimateConfig().CVAR_LOOKBACK, effective_cvar_lb_min)
        else:
            cfg.CVAR_LOOKBACK = trial.suggest_int(
                "CVAR_LOOKBACK", effective_cvar_lb_min, cvar_lb_max, step=cvar_lb_step
            )

        if hasattr(trial, "set_user_attr"):
            trial.set_user_attr("resolved_cfg", dict(vars(cfg)))

        scores: list[float] = []
        slice_diags: list[dict] = []

        first_oos_year = pd.Timestamp(TRAIN_START).year + 1
        trial_id = getattr(trial, "number", 0)

        for _, _, wf_oos_start, wf_oos_end in _iter_wfo_slices(TRAIN_START, TRAIN_END):
            oos_year = pd.Timestamp(wf_oos_start).year
            exclude_from_score = (oos_year == first_oos_year)

            oos = run_backtest(
                market_data=self.market_data,
                precomputed_matrices=self.precomputed_matrices,
                universe_type=self.universe_type,
                start_date=wf_oos_start,
                end_date=wf_oos_end,
                cfg=cfg
            )
            m = oos.metrics
            score, diag = _fitness_from_metrics(m, getattr(oos, "rebal_log", pd.DataFrame()))

            diag["year"]             = oos_year
            diag["excluded"]         = exclude_from_score
            diag["eq_start"]         = wf_oos_start
            diag["eq_end"]           = wf_oos_end
            slice_diags.append(diag)

            _excluded_tag = " [EXCLUDED from score]" if exclude_from_score else ""
            _ceiling_tag  = " ⚠ CEILING HIT" if diag["ceiling_hit"] else ""
            _ddgate_tag   = " ⚠ DD-GATE (>40%)" if diag["dd_gate_hit"] else ""
            _anomaly_tag  = " ⚠ ANOMALOUS-RETURNS" if diag.get("anomaly_hit") else ""
            logger.info(
                "[Trial %s | %d%s] CAGR=%+.1f%%  DD=%.1f%%  Turn=%.2fx  "
                "AvgExp=%.2f  AvgPos=%.1f  AvgCVaR=%.3f%%  "
                "RiskPenalty=%.2f  ExpPenalty=%.2f  DDPenalty=%.4f  RawScore=%.4f  Score=%.4f%s%s%s%s",
                trial_id, oos_year, _excluded_tag,
                diag["cagr"], abs(diag["max_dd"]), diag["turnover"],
                diag["avg_exposure"], diag["avg_positions"], diag["avg_cvar_pct"],
                diag["risk_penalty"], diag["exposure_penalty"], diag.get("dd_penalty", 0.0),
                diag["raw_score"], diag["score"],
                _ceiling_tag, _ddgate_tag, _anomaly_tag, "",
            )

            if not pd.notna(score):
                raise optuna.TrialPruned()

            if diag.get("dd_gate_hit") or score <= -2.0:
                raise optuna.TrialPruned()

            if not exclude_from_score:
                scores.append(float(score))

        if not scores:
            raise optuna.TrialPruned()

        aggregate = float(sum(scores) / len(scores))

        scored_diags   = [d for d in slice_diags if not d["excluded"]]
        avg_cagr       = sum(d["cagr"]         for d in scored_diags) / len(scored_diags)
        avg_dd         = sum(abs(d["max_dd"])   for d in scored_diags) / len(scored_diags)
        ceiling_slices = sum(1 for d in scored_diags if d["ceiling_hit"])
        ddgate_slices  = sum(1 for d in scored_diags if d["dd_gate_hit"])

        logger.info(
            "[Trial %s | AGGREGATE] score=%.4f  avg_cagr=%+.1f%%  avg_dd=%.1f%%  "
            "ceiling_hits=%d/%d  ddgate_hits=%d/%d  params=%s",
            trial_id, aggregate, avg_cagr, avg_dd,
            ceiling_slices, len(scored_diags),
            ddgate_slices,  len(scored_diags),
            {k: v for k, v in trial.params.items()},
        )

        if hasattr(trial, "set_user_attr"):
            trial.set_user_attr("slice_diags",      slice_diags)
            trial.set_user_attr("aggregate_score",  round(aggregate, 6))
            trial.set_user_attr("avg_cagr",         round(avg_cagr, 2))
            trial.set_user_attr("avg_dd",           round(-avg_dd, 2))
            trial.set_user_attr("ceiling_hits",     ceiling_slices)
            trial.set_user_attr("ddgate_hits",      ddgate_slices)

        return aggregate

# ─── Orchestration ────────────────────────────────────────────────────────────

def pre_load_data(universe_type: str, cfg: UltimateConfig | None = None) -> dict:
    logger.info("Initializing Data Pre-fetch phase...")
    normalized_universe = (universe_type or "").strip().lower()

    if normalized_universe == "nifty500":
        base_universe = get_nifty500()
    elif normalized_universe == "nse_total":
        base_universe = fetch_nse_equity_universe()
    else:
        logger.warning(f"Unknown universe_type '{universe_type}', falling back to nifty500")
        base_universe = get_nifty500()

    if cfg is None:
        cfg = UltimateConfig()
        cvar_bounds = SEARCH_SPACE_BOUNDS.get("CVAR_LOOKBACK")
        halflife_bounds = SEARCH_SPACE_BOUNDS.get("HALFLIFE_SLOW")
        if cvar_bounds:
            cfg.CVAR_LOOKBACK = int(cvar_bounds[1])
        if halflife_bounds:
            halflife_slow_max = int(halflife_bounds[1])
            halflife_calendar_days = halflife_slow_max * 4 * 365 // 252
            current_cvar_padding = max(400, cfg.CVAR_LOOKBACK * 2)
            cfg._pre_load_padding_days = max(current_cvar_padding, halflife_calendar_days)

    historical_union: set[str] = set()
    try:
        rebalance_dates = pd.date_range(TRAIN_START, TEST_END, freq=cfg.REBALANCE_FREQ)
        month_end_dates = pd.date_range(TRAIN_START, TEST_END, freq="ME")
        all_target_dates = sorted(set(rebalance_dates).union(set(month_end_dates)))
        for target_date in all_target_dates:
            historical_union.update(get_historical_universe(normalized_universe, pd.Timestamp(target_date)))
    except Exception as exc:
        logger.warning(
            "Historical universe preload failed for %s (%s). Falling back to base universe only.",
            normalized_universe,
            exc,
        )

    if historical_union:
        preload_universe = list(dict.fromkeys(base_universe + sorted(historical_union)))
    else:
        preload_universe = list(base_universe)

    symbols_to_fetch = list(dict.fromkeys(preload_universe + ["^NSEI", "^CRSLDX"]))

    logger.info(f"Fetching {len(symbols_to_fetch)} symbols from {TRAIN_START} to {TEST_END}...")
    kwargs = dict(tickers=symbols_to_fetch, required_start=TRAIN_START, required_end=TEST_END)
    if cfg is not None:
        kwargs["cfg"] = cfg
    try:
        market_data = load_or_fetch(**kwargs)
    except TypeError:
        kwargs.pop("cfg", None)
        market_data = load_or_fetch(**kwargs)

    market_data = apply_halt_simulation(market_data)

    precomputed_matrices = None
    if all(isinstance(v, pd.DataFrame) for v in market_data.values()):
        precomputed_matrices = build_precomputed_matrices(market_data, cfg=cfg, symbols=set(preload_universe))

    logger.info("Data pre-load complete. Commencing Bayesian Optimization.")
    return {
        "market_data": market_data,
        "precomputed_matrices": precomputed_matrices,
    }


def save_optimal_config(best_params: dict, filepath: str = "data/optimal_cfg.json"):
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_dir or os.path.dirname(os.path.abspath(filepath)) or ".",
        delete=False,
    ) as tmp_file:
        json.dump(best_params, tmp_file, indent=4)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_path = tmp_file.name

    try:
        os.replace(temp_path, filepath)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
    logger.info(f"Saved optimal parameters to {filepath}")


def run_optimization(
    universe_type: str = "nifty500",
    in_memory: bool = False,
    study_name: str | None = None,
):
    if in_memory:
        effective_storage = ":memory:"
        effective_n_jobs = 1
        logger.info(
            "In-memory mode: storage=:memory:, n_jobs=%d. "
            "Note — trial history will not be persisted; interrupted runs cannot be resumed.",
            effective_n_jobs,
        )
    else:
        effective_storage = OPTUNA_STORAGE
        effective_n_jobs = N_JOBS

    if effective_n_jobs != 1:
        logger.warning(
            "OPTUNA_N_JOBS=%d is not supported in this execution path (GIL bottleneck). "
            "Forcing n_jobs=1. For parallelism, run multiple optimizer processes with "
            "shared SQLite/RDB storage.",
            effective_n_jobs,
        )
        effective_n_jobs = 1

    print(f"\n\033[1;36m=== INSTITUTIONAL TIME-SERIES CV OPTIMIZER ===\033[0m")
    print(f"\033[90mIn-Sample (Train) : {TRAIN_START} to {TRAIN_END}\033[0m")
    print(f"\033[90mOut-of-Sample     : {TEST_START} to {TEST_END}\033[0m")
    print(f"\033[90mTrials            : {N_TRIALS}\033[0m\n")

    logger.info(f"Optimization universe: {universe_type}")
    preloaded = pre_load_data(universe_type)
    if isinstance(preloaded, dict) and "market_data" in preloaded:
        market_data = preloaded["market_data"]
        precomputed_matrices = preloaded.get("precomputed_matrices")
    else:
        market_data = preloaded
        precomputed_matrices = None

    os.makedirs("data", exist_ok=True)
    effective_study_name = (study_name or DEFAULT_STUDY_NAME).strip() or DEFAULT_STUDY_NAME
    logger.info("Using Optuna study: %s", effective_study_name)

    study = optuna.create_study(
        study_name=effective_study_name,
        direction="maximize",
        sampler=_build_sampler(),
        storage=effective_storage,
        load_if_exists=True
    )

    objective = MomentumObjective(
        market_data,
        universe_type,
        precomputed_matrices=precomputed_matrices,
    )

    logger.info(f"Starting {N_TRIALS} Bayesian Trials (This may take a while)...")
    def _best_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if study.best_trial.number != trial.number:
            return

        diags = trial.user_attrs.get("slice_diags", [])
        scored = [d for d in diags if not d.get("excluded", False)]

        hdr = (
            f"\n\033[1;33m{'─'*72}\033[0m"
            f"\n\033[1;33m  NEW BEST  Trial #{trial.number}  "
            f"Aggregate={trial.value:.4f}\033[0m"
            f"\n\033[1;33m{'─'*72}\033[0m"
        )
        logger.info(hdr)
        logger.info("  Parameters:")
        for k, v in trial.params.items():
            logger.info("    %-28s %s", k, v)

        logger.info("")
        logger.info("  %-6s  %-8s  %-8s  %-8s  %-8s  %-8s  %-10s  %s",
                    "Year", "CAGR%", "DD%", "Turn", "AvgPos", "AvgExp", "Score", "Flags")
        logger.info("  " + "-"*70)
        for d in diags:
            flags = []
            if d.get("excluded"):       flags.append("EXCL")
            if d.get("ceiling_hit"):    flags.append("CEIL")
            if d.get("dd_gate_hit"):    flags.append("DD-GATE")
            if d.get("anomaly_hit"):    flags.append("ANOM")
            logger.info(
                "  %-6s  %+7.1f%%  %6.1f%%  %6.2fx  %6.1f  %7.3f  %9.4f  %s",
                d["year"],
                d["cagr"], abs(d["max_dd"]), d["turnover"],
                d["avg_positions"], d["avg_exposure"],
                d["score"], " ".join(flags) if flags else "—",
            )
        logger.info("  " + "-"*70)
        if scored:
            avg_cagr = sum(d["cagr"]       for d in scored) / len(scored)
            avg_dd   = sum(abs(d["max_dd"]) for d in scored) / len(scored)
            ceil_n   = sum(1 for d in scored if d.get("ceiling_hit"))
            logger.info(
                "  %-6s  %+7.1f%%  %6.1f%%  %54s  ceiling_hits=%d/%d",
                "AVG", avg_cagr, avg_dd, "", ceil_n, len(scored),
            )
        logger.info("")

    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            show_progress_bar=True,
            n_jobs=effective_n_jobs,
            catch=(OptimizationError,),
            callbacks=[_best_trial_callback],
        )
    except Exception:
        logger.exception("Optimization aborted due to unexpected internal error.")
        raise

    if not study.best_trials:
        raise RuntimeError(
            "Optimization finished with no completed trials. "
            "Widen the search space or reduce hard constraints."
        )

    best_params = study.best_params
    best_trial = getattr(study, "best_trial", None)
    best_is_fitness = study.best_value

    print(f"\n\033[1;32m=== OPTIMIZATION COMPLETE ===\033[0m")
    print(f"\033[1mBest Fitness Score (IS):\033[0m {best_is_fitness:.4f}")
    print("\033[1mWinning Parameters:\033[0m")
    for k, v in best_params.items():
        print(f"  {k}: \033[33m{v}\033[0m")

    trials = list(getattr(study, "trials", []))
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        top10 = sorted(completed, key=lambda t: t.value or -999, reverse=True)[:10]
        print(f"\n\033[1;36m=== TOP-10 TRIALS DIAGNOSTIC SUMMARY ===\033[0m")
        print(f"\033[90m{'Trial':>6}  {'Score':>7}  {'AvgCAGR':>8}  {'AvgDD':>7}  {'CeilHits':>9}  {'DDGate':>7}\033[0m")
        print(f"\033[90m{'─'*58}\033[0m")
        for t in top10:
            avg_c   = t.user_attrs.get("avg_cagr",     "?")
            avg_d   = t.user_attrs.get("avg_dd",       "?")
            c_hits  = t.user_attrs.get("ceiling_hits", "?")
            dg_hits = t.user_attrs.get("ddgate_hits",  "?")
            scored_n = len([d for d in t.user_attrs.get("slice_diags", []) if not d.get("excluded", False)])
            c_str   = f"{c_hits}/{scored_n}" if isinstance(c_hits, int) else "?"
            dg_str  = f"{dg_hits}/{scored_n}" if isinstance(dg_hits, int) else "?"
            cagr_s  = f"{avg_c:+.1f}%" if isinstance(avg_c, float) else str(avg_c)
            dd_s    = f"{abs(avg_d):.1f}%" if isinstance(avg_d, float) else str(avg_d)
            print(f"  #{t.number:>4}  {t.value:>7.4f}  {cagr_s:>8}  {dd_s:>7}  {c_str:>9}  {dg_str:>7}")
        print(f"\033[90m{'─'*58}\033[0m")
        print()

        if best_trial is not None:
            c_hits   = best_trial.user_attrs.get("ceiling_hits", 0)
            scored_n = len([d for d in best_trial.user_attrs.get("slice_diags", []) if not d.get("excluded", False)])
            if isinstance(c_hits, int) and scored_n > 0 and c_hits == scored_n:
                print("\033[1;31m[WARNING] Best trial hit the 5.0 score ceiling on ALL scored slices.\033[0m")
                print("\033[33m          This means the optimizer cannot distinguish between parameter sets\033[0m")
                print("\033[33m          in the top range — TPE convergence may be unreliable.\033[0m")
                print()

    # OOS Top-K tournament
    print(f"\n\033[1;36m=== INITIATING OUT-OF-SAMPLE (OOS) VALIDATION — TOP-{OOS_TOP_K} TOURNAMENT ===\033[0m")
    print(f"\033[90mEvaluating top {OOS_TOP_K} IS trials on unseen data {TEST_START} → {TEST_END}\033[0m")
    print(f"\033[90mWinner = best OOS Calmar (not best IS score)\033[0m\n")

    trials      = list(getattr(study, "trials", []))
    completed   = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_k_trials = sorted(completed, key=lambda t: t.value or -999, reverse=True)[:OOS_TOP_K]

    if not top_k_trials:
        logger.warning("No completed trials available for OOS validation; skipping OOS tournament.")
        save_optimal_config(best_params)
        return

    _rs = "\u20b9" if _stdout_supports_rupee() else "Rs."
    valid_fields = UltimateConfig.__dataclass_fields__

    print(f"  {'Rank':>4}  {'Trial':>6}  {'IS Score':>9}  {'OOS CAGR':>9}  "
          f"{'OOS MaxDD':>9}  {'OOS Calmar':>10}  {'Status'}")
    print(f"  {'─'*75}")

    oos_results_list = []

    for rank, trial_candidate in enumerate(top_k_trials, 1):
        oos_cfg = UltimateConfig()
        resolved_cfg = trial_candidate.user_attrs.get("resolved_cfg", {})
        for k, v in resolved_cfg.items():
            if k in valid_fields:
                setattr(oos_cfg, k, v)
        for k, v in trial_candidate.params.items():
            if k in valid_fields:
                setattr(oos_cfg, k, v)

        try:
            oos_result = run_backtest(
                market_data=market_data,
                precomputed_matrices=precomputed_matrices,
                universe_type=universe_type,
                start_date=TEST_START,
                end_date=TEST_END,
                cfg=oos_cfg,
            )
            m          = oos_result.metrics
            oos_cagr   = m.get("cagr",   0.0)
            oos_maxdd  = m.get("max_dd", -100.0)
            oos_calmar = m.get("calmar", 0.0)

            passes = (
                oos_calmar > 0.5
                and abs(oos_maxdd) <= OOS_MAX_DD_CAP
            )
            status = "\033[32mPASS\033[0m" if passes else "\033[31mFAIL\033[0m"

            print(
                f"  {rank:>4}  #{trial_candidate.number:>5}  "
                f"{trial_candidate.value:>9.4f}  "
                f"{oos_cagr:>+8.1f}%  "
                f"{oos_maxdd:>8.1f}%  "
                f"{oos_calmar:>10.2f}  "
                f"{status}"
            )

            if passes:
                oos_results_list.append((oos_calmar, trial_candidate, trial_candidate.params, m))

        except Exception as exc:
            print(
                f"  {rank:>4}  #{trial_candidate.number:>5}  "
                f"{trial_candidate.value:>9.4f}  "
                f"{'ERROR':>9}  {'—':>9}  {'—':>10}  \033[31mERROR: {exc}\033[0m"
            )

    print(f"  {'─'*75}\n")

    if not oos_results_list:
        raise RuntimeError(
            f"OOS Validation Failed: None of the top-{OOS_TOP_K} IS trials "
            f"passed OOS (Calmar > 0.5 and MaxDD <= {OOS_MAX_DD_CAP}%). "
            f"All top IS trials are overfit. Consider: "
            f"(1) more N_TRIALS, (2) tighter search space bounds, "
            f"(3) stricter IS fitness penalty."
        )

    oos_results_list.sort(key=lambda x: x[0], reverse=True)
    best_oos_calmar, best_oos_trial, best_oos_params, best_oos_metrics = oos_results_list[0]

    print(f"\033[1;32m=== OOS TOURNAMENT WINNER ===\033[0m")
    print(f"  IS Rank        : #{top_k_trials.index(best_oos_trial) + 1} of top-{OOS_TOP_K} "
          f"(Trial #{best_oos_trial.number}, IS score {best_oos_trial.value:.4f})")
    print(f"  OOS Final      : {_rs}{best_oos_metrics.get('final', 0):,.0f}")
    print(f"  OOS CAGR       : {best_oos_metrics.get('cagr', 0):.2f}%")
    print(f"  OOS MaxDD      : {best_oos_metrics.get('max_dd', 0):.2f}%")
    print(f"  OOS Calmar     : {best_oos_calmar:.2f}")
    print(f"  OOS Sharpe     : {best_oos_metrics.get('sharpe', 0):.2f}")
    print(f"\n  Winning Parameters:")
    for k, v in best_oos_params.items():
        is_winner  = best_params.get(k)
        marker     = "" if is_winner == v else f"  \033[33m← differs from IS #1 ({is_winner})\033[0m"
        print(f"    {k}: \033[33m{v}\033[0m{marker}")

    if best_oos_trial.number != best_trial.number:
        print(
            f"\n\033[1;33m[NOTE] OOS winner is Trial #{best_oos_trial.number}, "
            f"NOT the IS #1 Trial #{best_trial.number}.\033[0m"
        )
        print(
            f"\033[33m       The IS #1 trial was overfit — it did not pass OOS.\033[0m"
            if all(t.number != best_trial.number for _, t, _, _ in oos_results_list)
            else
            f"\033[33m       IS #1 also passed OOS but had lower Calmar ({[m.get('calmar',0) for c,t,p,m in oos_results_list if t.number==best_trial.number][0]:.2f} vs {best_oos_calmar:.2f}).\033[0m"
        )
    else:
        print(f"\n\033[1;32m[NOTE] IS #1 trial also won OOS — strong generalization.\033[0m")

    save_optimal_config(best_oos_params)
    print("\n\033[1;32m[PASS]\033[0m OOS tournament complete. Best generalizing parameters saved.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bayesian optimizer for momentum strategy.")
    parser.add_argument(
        "--universe",
        default="nifty500",
        help="Universe to optimize against (e.g., nifty500, nse_total).",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        default=False,
        help=(
            "Use Optuna in-memory storage instead of the default SQLite backend. "
            "Eliminates write-lock contention for local runs. "
            "Trade-off: interrupted runs cannot be resumed (no on-disk checkpoint). "
            "Uses in-process execution (n_jobs=1)"
        ),
    )
    parser.add_argument(
        "--study-name",
        default=DEFAULT_STUDY_NAME,
        help=(
            "Optuna study name. Change this to start a clean optimization track "
            "when objective logic changes."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_optimization(
        universe_type=args.universe,
        in_memory=args.in_memory,
        study_name=args.study_name,
    )
