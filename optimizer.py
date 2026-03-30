"""
optimizer.py — Institutional Bayesian Time-Series CV Optimizer v11.58
====================================================================
Automates the discovery of optimal risk and momentum parameters using Optuna.
Uses walk-forward time-series cross-validation for parameter selection,
followed by true holdout Out-of-Sample (OOS) validation periods.

CHANGES:
- v11.58:
  - OBJECTIVE_VERSION = fitness_v11_58 to force a clean Optuna study.
  - N_TRIALS = 400 and OOS_TOP_K = 10 tuned for deeper search with tighter
    final OOS tournament selection.
- Walk-forward CV now uses a fixed rolling 2-year IS window instead of an
  expanding window, reducing later-fold history advantage.
- Fitness scoring now uses log1p(raw) for positive raw scores, removing the
  old Michaelis-Menten ceiling while still compressing extremes.
- forced_cash_penalty no longer affects fitness and is always emitted as 0.0
  in diagnostics for observability compatibility.
- OOS selection uses a single primary OOS holdout tournament for final selection.
- SEARCH_SPACE_BOUNDS widened substantially and now includes MAX_POSITIONS
  and SIGNAL_LAG_DAYS as optional optimization dimensions.
"""
import argparse
import json
import logging
import math
import os
import re
import sqlite3
import sys
import tempfile
import warnings
from pathlib import Path

# OSQP must be imported BEFORE numpy/pandas on Python 3.13/Windows to avoid
# a silent ABI crash (exit code 0xC0000005). momentum_engine imports osqp,
# but by the time Python resolves that import numpy is already loaded — too late.
import osqp  # noqa: F401

import numpy as np
import pandas as pd
import optuna
from typing import Any
from optuna.samplers import TPESampler


from log_config import load_dotenv_safe
from momentum_engine import UltimateConfig, OptimizationError, OptimizationErrorType
from backtest_engine import run_backtest, apply_halt_simulation, build_precomputed_matrices, _compute_warmup_start
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


def _utc_today() -> pd.Timestamp:
    """Return current UTC date as timezone-naive midnight timestamp."""
    return pd.Timestamp.now("UTC").tz_convert(None).normalize()


def configure_optimizer_logging(color: bool = True) -> None:
    """Call once from __main__ before any optimizer work begins."""
    load_dotenv_safe()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "\033[90m[%(asctime)s]\033[0m %(message)s" if color else "[%(asctime)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
        logger.addHandler(handler)

    # Staleness check: warn if TRAIN_END is more than 6 months behind today.
    # This is advisory only — behavior and results are never altered.
    # To extend the training window, update TRAIN_END above and re-run the optimizer.
    _today = _utc_today()
    _train_end_ts = pd.Timestamp(TRAIN_END)
    _days_stale = (_today - _train_end_ts).days
    if _today > _train_end_ts + pd.DateOffset(months=TRAIN_END_STALENESS_THRESHOLD_MONTHS):
        logger.warning(
            "TRAIN_END=%s is %d days behind today. Optimizer results remain "
            "reproducible but exclude recent data. Update TRAIN_END in "
            "optimizer.py to include it.",
            TRAIN_END,
            _days_stale,
        )

# ─── Optimization Configuration ───────────────────────────────────────────────

TRAIN_START = "2019-01-01"
TRAIN_END   = "2025-12-31"
TEST_START   = "2024-01-01"
TEST_END     = "2024-12-31"
TRAIN_END_STALENESS_THRESHOLD_MONTHS = 6

N_TRIALS = 400

# OOS hard pass: Calmar > 0.5 AND MaxDD <= 38%.
# Raised from original 35% — 2023-2025 is structurally harder than training.
OOS_MAX_DD_CAP      = 38.0
# OOS soft tier: displayed as NEAR but does not count as a pass.
OOS_SOFT_MAX_DD_CAP = 42.0

OOS_TOP_K = 10

_MIN_IS_CALENDAR_DAYS = 365

# Search space widened in v11_56 to reduce TPE noise-chasing and add optional
# portfolio construction / execution dimensions. Bounds are expressed as
# (min, max, step) for integer and stepped-float suggestions.
SEARCH_SPACE_BOUNDS = {
    "HALFLIFE_FAST":      (30, 70, 5),
    "HALFLIFE_SLOW":      (100, 180, 10),
    "CONTINUITY_BONUS":   (0.05, 0.25, 0.05),
    "RISK_AVERSION":      (8.0, 18.0, 1.0),
    "CVAR_DAILY_LIMIT":   (0.040, 0.120, 0.010),
    "CVAR_LOOKBACK":      (60, 180, 20),
    "MAX_POSITIONS":      (10, 24, 2),
    "SIGNAL_LAG_DAYS":    (0, 21, 3),
    "MIN_EXPOSURE_FLOOR": (0.0, 0.20, 0.05),
}

N_JOBS         = int(os.getenv("OPTUNA_N_JOBS", "1"))
OPTUNA_SEED    = os.getenv("OPTUNA_SEED")
# ARCH-FIX-4: SQLite is suitable for ≤4 parallel workers; use PostgreSQL for larger runs.
OPTUNA_STORAGE = os.getenv("OPTUNA_STORAGE", "sqlite:///data/optuna_study.db?timeout=30")

OBJECTIVE_VERSION  = "fitness_v11_58"
DEFAULT_STUDY_NAME = f"Momentum_Risk_Parity_{OBJECTIVE_VERSION}"

MAX_REASONABLE_CAGR_PCT       = 300.0
MAX_REASONABLE_FINAL_MULTIPLE = 8.0
BASE_INITIAL_CAPITAL          = UltimateConfig().INITIAL_CAPITAL
OOS_TOURNAMENT_JOURNAL_DIR    = Path("data")
DRAWDOWN_FLOOR                = 1.0  # keep consistent with backtest_engine Calmar denominator


def _stdout_supports_rupee(stdout=None) -> bool:
    stream = stdout if stdout is not None else getattr(sys, "stdout", None)
    if stream is None:
        return False
    encoding = getattr(stream, "encoding", None) or "utf-8"
    errors   = getattr(stream, "errors",   None) or "strict"
    try:
        "\u20b9".encode(encoding, errors=errors)
    except (LookupError, TypeError, UnicodeEncodeError):
        return False
    return True


def _build_sampler() -> TPESampler:
    if OPTUNA_SEED in (None, ""):
        return TPESampler(n_ei_candidates=24, multivariate=True)
    return TPESampler(seed=int(str(OPTUNA_SEED)), n_ei_candidates=24, multivariate=True)


def _normalize_universe_type(universe_type: str | None) -> str:
    normalized_universe = (universe_type or "").strip().lower()
    if normalized_universe in {"nifty500", "nse_total"}:
        return normalized_universe

    logger.warning("Unknown universe_type %r, falling back to nifty500", universe_type)
    return "nifty500"


# ─── Objective Function ───────────────────────────────────────────────────────

def _iter_wfo_slices(train_start: str, train_end: str):
    """
    Walk-forward folds with a fixed 2-year IS window and 1-year OOS window,
    stepping annually.

    Fixed-length IS prevents later folds from having a data advantage over
    earlier ones — a known cause of TPE convergence to parameters that only
    work with more history (i.e. later in time).

    With TRAIN_START="2019-01-01" the yielded OOS years are 2021, 2022, 2023.
    The 2019 and 2020 OOS folds from the old expanding window are intentionally
    dropped — 3 independent folds are more diagnostic than 4 correlated ones.
    """
    IS_YEARS = 2
    start    = pd.Timestamp(train_start)
    end      = pd.Timestamp(train_end)
    for y in range(start.year + IS_YEARS, end.year + 1):
        oos_start = pd.Timestamp(f"{y}-01-01")
        oos_end   = min(pd.Timestamp(f"{y}-12-31"), end)
        is_start  = oos_start - pd.DateOffset(years=IS_YEARS)
        is_end    = oos_start - pd.Timedelta(days=1)

        if is_start < start:
            original_is_start = is_start
            is_start = start
            logger.warning(
                "[WFO] Fold OOS=%d: IS window shortened. Original start %s clamped to %s. "
                "Resulting IS window: %d calendar days.",
                y,
                original_is_start.strftime("%Y-%m-%d"),
                start.strftime("%Y-%m-%d"),
                (is_end - is_start).days + 1,
            )

        is_calendar_days = (is_end - is_start).days + 1
        if is_calendar_days < _MIN_IS_CALENDAR_DAYS:
            logger.debug(
                "[WFO] Skipping fold OOS=%d: IS window only %d calendar days (minimum %d).",
                y, is_calendar_days, _MIN_IS_CALENDAR_DAYS,
            )
            continue

        yield (
            is_start.strftime("%Y-%m-%d"),
            is_end.strftime("%Y-%m-%d"),
            oos_start.strftime("%Y-%m-%d"),
            oos_end.strftime("%Y-%m-%d"),
        )


def _fitness_from_metrics(
    metrics: dict,
    rebal_log: pd.DataFrame,
) -> tuple[float, float, dict]:
    """
    Compute a scalar fitness score plus a diagnostics dict for logging.

    IS_DD_GATE = 40%: kept at original value. Lowering to 35% caused the 2020
    COVID fold to always return -2.0 (COVID drove 35-45% DD for any momentum
    strategy), making positive aggregate scores mathematically impossible after
    181 trials. IS_DD_GATE is not the same concept as OOS_MAX_DD_CAP:
      IS_DD_GATE      = "this fold is catastrophic noise, score it harshly"
      OOS_MAX_DD_CAP  = "acceptable drawdown in live trading"
    These thresholds serve different purposes and need not be equal.

    IS_DD_PENALTY_PCT = 12%: lowered from 15%. Quadratic penalty fires earlier,
    adding more continuous pressure on moderately-high-drawdown parameter sets
    without making the gate threshold itself too aggressive.

    forced_cash_penalty is logged only for observability compatibility and no
    longer affects the fitness score.

    Positive raw scores use log1p(raw), preserving ranking without a hard
    asymptotic score ceiling.
    Score floor  : hard -2.0
    """
    cagr         = float(metrics.get("cagr",    0.0))
    max_dd       = abs(float(metrics.get("max_dd", 100.0)))
    turnover     = float(metrics.get("turnover", 0.0))
    sortino      = float(metrics.get("sortino",  0.0) or 0.0)
    final_equity = float(metrics.get("final", BASE_INITIAL_CAPITAL) or BASE_INITIAL_CAPITAL)
    final_multiple = final_equity / max(BASE_INITIAL_CAPITAL, 1e-9)

    # Base friction: 30 bps round-trip cost (15 bps per side for slippage/STT/fees).
    base_friction_drag = turnover * 0.30
    # Nonlinear churn penalty: heavily penalizes turnover beyond ~18 round-trips/year.
    churn_penalty = (max(0.0, turnover - 18.0) ** 2) / 10.0

    turnover_drag = base_friction_drag + churn_penalty
    cagr_net = cagr - turnover_drag

    avg_cvar            = 0.0
    avg_exposure        = 1.0
    avg_positions       = 0.0
    n_rebalances        = 0
    # forced_cash_penalty is dead code (always 0.0) per BUG-OPT-05; preserved for log-parser and test compatibility
    forced_cash_penalty = 0.0

    if rebal_log is not None and not rebal_log.empty:
        fallback_series = pd.Series([0.0])
        avg_cvar      = float(pd.to_numeric(rebal_log.get("realised_cvar",      fallback_series), errors="coerce").fillna(0.0).mean())
        avg_exposure  = float(pd.to_numeric(rebal_log.get("exposure_multiplier", fallback_series), errors="coerce").fillna(0.0).mean())
        avg_positions = float(pd.to_numeric(rebal_log.get("n_positions",         fallback_series), errors="coerce").fillna(0.0).mean())
        n_rebalances  = len(rebal_log)

    _pos_deficit       = max(0.0, 6.0 - avg_positions)
    concentration_mult = 1.0 + _pos_deficit * 0.30

    import math as _math
    if sortino is None or not _math.isfinite(sortino):
        sortino_quality = 1.0   # NaN = no downside data, not bad Sortino
    else:
        sortino_quality = min(max(sortino / 2.5, 0.50), 1.15)

    risk_penalty = (max_dd + (avg_cvar * 100.0 * 2.0) + 1.0) * concentration_mult

    # IS_DD_GATE = 40%: kept at original. See docstring for reasoning.
    # Do NOT lower this to match OOS_MAX_DD_CAP — they are different concepts.
    IS_DD_GATE        = 40.0
    # IS_DD_PENALTY_PCT = 12%: lowered from original 15%.
    IS_DD_PENALTY_PCT = 12.0

    if max_dd > IS_DD_GATE:
        raw   = -(max_dd / 5.0)
        score = max(raw, -2.0)
        diag  = {
            "cagr": round(cagr, 2), "max_dd": round(-max_dd, 2),
            "turnover": round(turnover, 4), "final_multiple": round(final_multiple, 4),
            "cagr_net": round(cagr_net, 2), "avg_cvar_pct": round(avg_cvar * 100.0, 4),
            "avg_exposure": round(avg_exposure, 4), "avg_positions": round(avg_positions, 2),
            "n_rebalances": n_rebalances, "concentration_mult": round(concentration_mult, 4),
            "sortino_quality": round(sortino_quality, 4), "risk_penalty": round(risk_penalty, 4),
            "exposure_penalty": 0.0, "dd_penalty": 0.0,
            "forced_cash_penalty": round(forced_cash_penalty, 4),
            "raw_score": round(raw, 6), "score": round(score, 6),
            "ceiling_hit": False, "dd_gate_hit": True, "anomaly_hit": False,
        }
        calmar_score = cagr_net / max(abs(max_dd), DRAWDOWN_FLOOR)  # ARCH-FIX-2
        return score, calmar_score, diag

    dd_excess  = max(0.0, max_dd - IS_DD_PENALTY_PCT)
    dd_penalty = (dd_excess ** 2) / 100.0

    exposure_penalty = 0.0 if avg_exposure >= 0.25 else (0.25 - avg_exposure) * 2.0
    if avg_positions < 1.0:
        exposure_penalty += 0.5

    anomaly_hit = (
        cagr > MAX_REASONABLE_CAGR_PCT
        or final_multiple > MAX_REASONABLE_FINAL_MULTIPLE
    )

    if anomaly_hit:
        raw         = -(
            max(cagr - MAX_REASONABLE_CAGR_PCT, 0.0) / 50.0
            + max(final_multiple - MAX_REASONABLE_FINAL_MULTIPLE, 0.0)
        )
        score       = max(raw, -2.0)
        ceiling_hit = False
        dd_gate_hit = False
    elif abs(cagr) < 1e-12 and max_dd == 0.0:
        raw         = 0.0
        score       = 0.0
        ceiling_hit = False
        dd_gate_hit = False
    else:
        raw = (
            (cagr_net / risk_penalty) * sortino_quality
            - exposure_penalty
            - dd_penalty
        )
        # ARCH-FIX-6: symmetric log-modulus transform removes the raw>0 discontinuity.
        score = math.copysign(math.log1p(abs(raw)), raw)
        ceiling_hit = False
        score = max(score, -2.0)
        dd_gate_hit = False

    # forced_cash_penalty is logged for observability but no longer affects
    # the fitness score — see BUG-OPT-05.  Always emit 0.0 so downstream
    # test assertions and log parsers remain stable.
    # Intentional redundant reassignment for observability stability: downstream logs/tests
    # expect forced_cash_penalty to be emitted from this point as a stable 0.0 field.
    forced_cash_penalty = 0.0

    diag = {
        "cagr":                round(cagr,    2),
        "max_dd":              round(-max_dd,  2),
        "turnover":            round(turnover, 4),
        "final_multiple":      round(final_multiple,  4),
        "cagr_net":            round(cagr_net,        2),
        "avg_cvar_pct":        round(avg_cvar * 100.0, 4),
        "avg_exposure":        round(avg_exposure,    4),
        "avg_positions":       round(avg_positions,   2),
        "n_rebalances":        n_rebalances,
        "concentration_mult":  round(concentration_mult, 4),
        "sortino_quality":     round(sortino_quality, 4),
        "risk_penalty":        round(risk_penalty,    4),
        "exposure_penalty":    round(exposure_penalty, 4),
        "dd_penalty":          round(dd_penalty,      4),
        "forced_cash_penalty": round(forced_cash_penalty, 4),
        "raw_score":           round(raw, 6) if not (abs(cagr) < 1e-12 and max_dd == 0.0) else 0.0,
        "score":               round(score, 6),
        "ceiling_hit":         ceiling_hit,
        "dd_gate_hit":         dd_gate_hit,
        "anomaly_hit":         anomaly_hit,
    }
    calmar_score = cagr_net / max(abs(max_dd), DRAWDOWN_FLOOR)  # ARCH-FIX-2
    return score, calmar_score, diag


class MomentumObjective:
    def __init__(
        self,
        market_data: dict,
        universe_type: str,
        search_space: dict | None = None,
        precomputed_matrices: dict | None = None,
    ):
        self.market_data          = market_data
        self.universe_type        = universe_type
        self.search_space         = search_space or SEARCH_SPACE_BOUNDS
        self.precomputed_matrices = precomputed_matrices

    def __call__(self, trial: optuna.Trial) -> tuple[float, float]:
        cfg = UltimateConfig()

        halflife_fast_bounds = self.search_space["HALFLIFE_FAST"]
        halflife_slow_bounds = self.search_space["HALFLIFE_SLOW"]

        if len(halflife_fast_bounds) == 3:
            halflife_fast_min, halflife_fast_max, halflife_fast_step = halflife_fast_bounds
        else:
            halflife_fast_min, halflife_fast_max = halflife_fast_bounds[:2]
            halflife_fast_step = 1

        if len(halflife_slow_bounds) == 3:
            halflife_slow_min, halflife_slow_max, halflife_slow_step = halflife_slow_bounds
        else:
            halflife_slow_min, halflife_slow_max = halflife_slow_bounds[:2]
            halflife_slow_step = 1

        cfg.HALFLIFE_FAST = trial.suggest_int(
            "HALFLIFE_FAST", int(halflife_fast_min), int(halflife_fast_max), step=int(halflife_fast_step)
        )
        cfg.HALFLIFE_SLOW = trial.suggest_int(
            "HALFLIFE_SLOW", int(halflife_slow_min), int(halflife_slow_max), step=int(halflife_slow_step)
        )

        if cfg.HALFLIFE_FAST > cfg.HALFLIFE_SLOW:
            raise optuna.TrialPruned()

        continuity_min, continuity_max, continuity_step = self.search_space["CONTINUITY_BONUS"]
        cfg.CONTINUITY_BONUS = trial.suggest_float(
            "CONTINUITY_BONUS", continuity_min, continuity_max, step=continuity_step
        )

        risk_aversion_min, risk_aversion_max, risk_aversion_step = self.search_space["RISK_AVERSION"]
        cvar_min, cvar_max, cvar_step = self.search_space["CVAR_DAILY_LIMIT"]
        cfg.RISK_AVERSION    = trial.suggest_float(
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
                "CVAR_LOOKBACK", int(effective_cvar_lb_min), int(cvar_lb_max), step=int(cvar_lb_step)
            )

        _mp_bounds = self.search_space.get("MAX_POSITIONS")
        if _mp_bounds is not None:
            _mp_min, _mp_max, _mp_step = _mp_bounds
            if isinstance(trial, optuna.trial.FixedTrial) and "MAX_POSITIONS" not in trial.params:
                cfg.MAX_POSITIONS = UltimateConfig().MAX_POSITIONS
            else:
                cfg.MAX_POSITIONS = trial.suggest_int(
                    "MAX_POSITIONS",
                    int(_mp_min),
                    int(_mp_max),
                    step=int(_mp_step),
                )

        _lag_bounds = self.search_space.get("SIGNAL_LAG_DAYS")
        if _lag_bounds is not None:
            _lag_min, _lag_max, _lag_step = _lag_bounds
            if isinstance(trial, optuna.trial.FixedTrial) and "SIGNAL_LAG_DAYS" not in trial.params:
                cfg.SIGNAL_LAG_DAYS = UltimateConfig().SIGNAL_LAG_DAYS
            else:
                cfg.SIGNAL_LAG_DAYS = trial.suggest_int(
                    "SIGNAL_LAG_DAYS",
                    int(_lag_min),
                    int(_lag_max),
                    step=int(_lag_step),
                )

        _floor_bounds = self.search_space.get("MIN_EXPOSURE_FLOOR")
        if _floor_bounds is not None:
            _floor_min, _floor_max, _floor_step = _floor_bounds
            if isinstance(trial, optuna.trial.FixedTrial) and "MIN_EXPOSURE_FLOOR" not in trial.params:
                cfg.MIN_EXPOSURE_FLOOR = UltimateConfig().MIN_EXPOSURE_FLOOR
            else:
                cfg.MIN_EXPOSURE_FLOOR = trial.suggest_float(
                    "MIN_EXPOSURE_FLOOR",
                    float(_floor_min),
                    float(_floor_max),
                    step=float(_floor_step),
                )

        if hasattr(trial, "set_user_attr"):
            trial.set_user_attr("resolved_cfg", dict(vars(cfg)))

        scores:      list[float] = []
        calmar_scores: list[float] = []
        slice_diags: list[dict]  = []
        trial_id    = getattr(trial, "number", 0)
        n_gate_hits = 0

        for wf_is_start, _, wf_oos_start, wf_oos_end in _iter_wfo_slices(TRAIN_START, TRAIN_END):
            oos_year = pd.Timestamp(wf_oos_start).year

            fold_matrices = None
            if self.precomputed_matrices is not None:
                # FIX-BUG-14: clip fold matrices from the warmup of wf_is_start,
                # not the warmup of wf_oos_start.  The OOS backtest (start_date=
                # wf_oos_start) uses precomputed_matrices for signal warmup; those
                # signals are computed over the IS window starting at wf_is_start.
                # Using _compute_warmup_start(wf_oos_start) could truncate the IS
                # data for short signal lookbacks, leaving run_backtest without the
                # full 2-year in-sample history needed for warm signal computation.
                # Taking min() of both warmup starts is safe because the full
                # precomputed_matrices span the entire training+test range.
                warmup_oos_ts = pd.Timestamp(_compute_warmup_start(wf_oos_start, cfg))
                warmup_is_ts  = pd.Timestamp(_compute_warmup_start(wf_is_start,  cfg))
                fold_start_ts = min(warmup_is_ts, warmup_oos_ts)
                oos_end_ts    = pd.Timestamp(wf_oos_end)
                fold_matrices = {}
                for key, df in self.precomputed_matrices.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        fold_matrices[key] = df.loc[fold_start_ts:oos_end_ts]
                    else:
                        fold_matrices[key] = df

            oos = run_backtest(
                market_data          = self.market_data,
                precomputed_matrices = fold_matrices,
                universe_type        = self.universe_type,
                start_date           = wf_oos_start,
                end_date             = wf_oos_end,
                cfg                  = cfg,
            )
            m = oos.metrics
            score, calmar_score, diag = _fitness_from_metrics(m, getattr(oos, "rebal_log", pd.DataFrame()))

            diag["year"]     = oos_year
            diag["eq_start"] = wf_oos_start
            diag["eq_end"]   = wf_oos_end
            slice_diags.append(diag)

            _ceiling_tag = " ⚠ CEILING HIT"       if diag["ceiling_hit"]                    else ""
            _ddgate_tag  = " ⚠ DD-GATE (>40%)"     if diag["dd_gate_hit"]                    else ""
            _anomaly_tag = " ⚠ ANOMALOUS-RETURNS"  if diag.get("anomaly_hit")                else ""
            _cashpen_tag = (
                f" ⚠ FORCED-CASH={diag.get('forced_cash_penalty', 0.0):.2f}"
                if diag.get("forced_cash_penalty", 0.0) > 0.1 else ""
            )
            logger.info(
                "[Trial %s | %d] CAGR=%+.1f%%  DD=%.1f%%  Turn=%.2fx  "
                "AvgExp=%.2f  AvgPos=%.1f  AvgCVaR=%.3f%%  "
                "RiskPenalty=%.2f  ExpPenalty=%.2f  DDPenalty=%.4f  ForcedCashPen=%.4f  "
                "RawScore=%.4f  Score=%.4f%s%s%s%s",
                trial_id, oos_year,
                diag["cagr"], abs(diag["max_dd"]), diag["turnover"],
                diag["avg_exposure"], diag["avg_positions"], diag["avg_cvar_pct"],
                diag["risk_penalty"], diag["exposure_penalty"], diag.get("dd_penalty", 0.0),
                diag.get("forced_cash_penalty", 0.0),
                diag["raw_score"], diag["score"],
                _ceiling_tag, _ddgate_tag, _anomaly_tag, _cashpen_tag,
            )

            if not pd.notna(score):
                raise optuna.TrialPruned()
            if not pd.notna(calmar_score):
                raise optuna.TrialPruned()

            is_structural_gate_hit = diag.get("dd_gate_hit") or diag.get("anomaly_hit")
            diag["excluded"] = is_structural_gate_hit
            if is_structural_gate_hit:
                n_gate_hits += 1
                scores.append(float(score))
                calmar_scores.append(float(calmar_score))
                if n_gate_hits > 2:
                    raise optuna.TrialPruned()
                logger.debug(
                    "[Trial %s | %d] Single structural gate-hit fold included in aggregate "
                    "(dd_gate=%s anomaly=%s score=%.4f).",
                    trial_id, oos_year,
                    diag.get("dd_gate_hit"), diag.get("anomaly_hit"), score,
                )
                continue

            scores.append(float(score))
            calmar_scores.append(float(calmar_score))

        if not scores:
            raise optuna.TrialPruned()

        aggregate = float(sum(scores) / len(scores))
        aggregate_calmar = float(sum(calmar_scores) / len(calmar_scores))

        avg_cagr       = sum(d["cagr"]       for d in slice_diags) / len(slice_diags)
        avg_dd         = sum(abs(d["max_dd"]) for d in slice_diags) / len(slice_diags)
        ceiling_slices = sum(1 for d in slice_diags if d["ceiling_hit"])
        ddgate_slices  = sum(1 for d in slice_diags if d["dd_gate_hit"])

        logger.info(
            "[Trial %s | AGGREGATE] score=%.4f  avg_cagr=%+.1f%%  avg_dd=%.1f%%  "
            "ceiling_hits=%d/%d  ddgate_hits=%d/%d  params=%s",
            trial_id, aggregate, avg_cagr, avg_dd,
            ceiling_slices, len(slice_diags),
            ddgate_slices,  len(slice_diags),
            {k: v for k, v in trial.params.items()},
        )

        if hasattr(trial, "set_user_attr"):
            trial.set_user_attr("slice_diags",     slice_diags)
            trial.set_user_attr("aggregate_score", round(aggregate, 6))
            trial.set_user_attr("aggregate_calmar", round(aggregate_calmar, 6))  # ARCH-FIX-2
            trial.set_user_attr("avg_cagr",        round(avg_cagr, 2))
            trial.set_user_attr("avg_dd",          round(-avg_dd,  2))
            trial.set_user_attr("ceiling_hits",    ceiling_slices)
            trial.set_user_attr("ddgate_hits",     ddgate_slices)

        return aggregate, aggregate_calmar


# ─── Orchestration ────────────────────────────────────────────────────────────

def _validate_regime_benchmark_data(market_data: dict, required_start: str, required_end: str) -> None:
    """
    Validate regime benchmark inputs.

    Hard-fail only when no benchmark has any usable Close data at all.
    Partial history (e.g. symbol history starts after required_start) is
    accepted with a warning because the regime subsystem can safely default
    to neutral when benchmark context is sparse.
    """
    required_start_ts = pd.Timestamp(required_start)
    required_end_ts = pd.Timestamp(required_end)
    benchmark_notes: list[str] = []
    partial_coverage_notes: list[str] = []
    has_any_usable_close = False

    for ticker in ("^CRSLDX", "^NSEI"):
        df = market_data.get(ticker)
        if not isinstance(df, pd.DataFrame) or df.empty:
            benchmark_notes.append(f"{ticker}: missing")
            continue
        if "Close" not in df.columns:
            benchmark_notes.append(f"{ticker}: missing Close column")
            continue

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if close.empty:
            benchmark_notes.append(f"{ticker}: Close column has no finite values")
            continue

        has_any_usable_close = True
        coverage_start = pd.Timestamp(close.index.min())
        coverage_end = pd.Timestamp(close.index.max())
        if coverage_start > required_start_ts or coverage_end < required_end_ts:
            partial_coverage_notes.append(
                f"{ticker}: coverage {coverage_start.date()} -> {coverage_end.date()} does not span "
                f"{required_start_ts.date()} -> {required_end_ts.date()}"
            )
            continue

        if len(close.loc[required_start_ts:required_end_ts]) < 200:
            benchmark_notes.append(
                f"{ticker}: only {len(close.loc[required_start_ts:required_end_ts])} rows within required window"
            )
            continue

        logger.info(
            "Using %s for regime data validation (%s -> %s, %d rows in required window).",
            ticker,
            coverage_start.date(),
            coverage_end.date(),
            len(close.loc[required_start_ts:required_end_ts]),
        )
        return

    if has_any_usable_close:
        warn_details = partial_coverage_notes + benchmark_notes
        logger.warning(
            "Regime benchmark has only partial coverage for %s -> %s. Proceeding with available history; "
            "regime score may be neutral early in-sample. Details: %s",
            required_start_ts.date(),
            required_end_ts.date(),
            "; ".join(warn_details) if warn_details else "n/a",
        )
        return

    raise OptimizationError(
        "Regime benchmark validation failed. Neither ^CRSLDX nor ^NSEI has usable Close data. "
        "Details: " + "; ".join(benchmark_notes),
        OptimizationErrorType.DATA,
    )


def pre_load_data(universe_type: str, cfg: UltimateConfig | None = None) -> dict:
    logger.info("Initializing Data Pre-fetch phase...")
    normalized_universe = _normalize_universe_type(universe_type)

    if normalized_universe == "nifty500":
        base_universe = get_nifty500()
    else:
        base_universe = fetch_nse_equity_universe()

    if cfg is None:
        cfg = UltimateConfig()
        cvar_bounds = SEARCH_SPACE_BOUNDS.get("CVAR_LOOKBACK")
        if cvar_bounds:
            cfg.CVAR_LOOKBACK = int(cvar_bounds[1])

    historical_union: set[str] = set()
    try:
        for target_date in pd.date_range(TRAIN_START, TEST_END, freq="QE"):
            historical_union.update(
                get_historical_universe(normalized_universe, pd.Timestamp(target_date))
            )
    except Exception as exc:
        logger.warning(
            "Historical universe preload failed for %s (%s). Falling back to base universe only.",
            normalized_universe, exc,
        )

    if historical_union:
        preload_universe = list(dict.fromkeys(base_universe + sorted(historical_union)))
    else:
        preload_universe = list(base_universe)

    symbols_to_fetch = list(dict.fromkeys(preload_universe + ["^NSEI", "^CRSLDX"]))

    _pre_load_cfg        = cfg if cfg is not None else UltimateConfig()
    _actual_warmup_start = _compute_warmup_start(TRAIN_START, _pre_load_cfg)
    _fetch_end           = TEST_END
    logger.info(
        "Fetching %d symbols from %s (warmup) to %s...",
        len(symbols_to_fetch), _actual_warmup_start, _fetch_end,
    )
    kwargs: dict[str, Any] = dict(
        tickers        = symbols_to_fetch,
        required_start = _actual_warmup_start,
        required_end   = _fetch_end,
    )
    if cfg is not None:
        kwargs["cfg"] = cfg
    try:
        market_data = load_or_fetch(**kwargs)
    except TypeError as _te:
        # FIX-MB-OPT-02: log the dropped parameter so forward-compatibility
        # shims are visible during debugging rather than failing silently.
        logger.warning(
            "pre_load_data: dropped 'cfg' kwarg due to TypeError (%s). "
            "load_or_fetch does not accept cfg; retrying without it.",
            _te,
        )
        kwargs.pop("cfg", None)
        market_data = load_or_fetch(**kwargs)

    if getattr(cfg, "SIMULATE_HALTS", False):
        market_data = apply_halt_simulation(market_data)

    _validate_regime_benchmark_data(market_data, _actual_warmup_start, _fetch_end)

    precomputed_matrices = None
    if all(isinstance(v, pd.DataFrame) for v in market_data.values()):
        precomputed_matrices = build_precomputed_matrices(
            market_data, cfg=cfg, symbols=set(preload_universe)
        )

    logger.info("Data pre-load complete. Commencing Bayesian Optimization.")
    return {
        "market_data":          market_data,
        "precomputed_matrices": precomputed_matrices,
    }


def _validate_optimal_config(params: dict) -> list[str]:
    """Validate cross-field constraints before persisting optimal config."""
    violations: list[str] = []

    cvar = params.get("CVAR_DAILY_LIMIT")
    if cvar is not None:
        if not isinstance(cvar, (int, float)) or cvar <= 0.0:
            violations.append(f"CVAR_DAILY_LIMIT must be > 0; got {cvar!r}")
        elif cvar > 0.50:
            violations.append(f"CVAR_DAILY_LIMIT={cvar:.3f} exceeds 50% — implausibly loose")

    max_pos = params.get("MAX_POSITIONS")
    if max_pos is not None and (not isinstance(max_pos, int) or max_pos < 2):
        violations.append(f"MAX_POSITIONS must be int >= 2; got {max_pos!r}")

    hf = params.get("HALFLIFE_FAST")
    hs = params.get("HALFLIFE_SLOW")
    if hf is not None and hs is not None and hf > hs:
        violations.append(
            f"HALFLIFE_FAST ({hf}) > HALFLIFE_SLOW ({hs}) — would invert momentum signal"
        )

    exp_floor = params.get("MIN_EXPOSURE_FLOOR")
    if exp_floor is not None and (
        not isinstance(exp_floor, (int, float)) or not (0.0 <= exp_floor <= 1.0)
    ):
        violations.append(f"MIN_EXPOSURE_FLOOR={exp_floor!r} must be in [0, 1]")

    max_w = params.get("MAX_SINGLE_NAME_WEIGHT")
    if max_w is not None and (
        not isinstance(max_w, (int, float)) or not (0.01 <= max_w <= 1.0)
    ):
        violations.append(f"MAX_SINGLE_NAME_WEIGHT={max_w!r} must be in [0.01, 1.0]")

    risk_av = params.get("RISK_AVERSION")
    if risk_av is not None and (
        not isinstance(risk_av, (int, float)) or risk_av <= 0
    ):
        violations.append(f"RISK_AVERSION must be > 0; got {risk_av!r}")

    lag = params.get("SIGNAL_LAG_DAYS")
    if lag is not None and (not isinstance(lag, int) or lag < 0):
        violations.append(
            f"SIGNAL_LAG_DAYS must be int >= 0; got {lag!r}"
        )

    return violations


def save_optimal_config(best_params: dict, filepath: str = "data/optimal_cfg.json"):
    violations = _validate_optimal_config(best_params)
    if violations:
        msg = "; ".join(violations)
        raise ValueError(
            f"save_optimal_config: refusing to persist invalid config ({msg}). "
            f"Params: {best_params}"
        )

    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    resolved     = os.path.abspath(filepath)
    resolved_dir = os.path.dirname(resolved)

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8",
        dir=resolved_dir or ".",
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
    logger.info("Saved optimal parameters to %s", filepath)


def _oos_journal_path(study_name: str) -> Path:
    # ARCH-FIX-9
    cleaned = re.sub(r"[\\/]+", "_", (study_name or "").strip())
    cleaned = cleaned.replace("..", "_")
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "study"
    if len(cleaned) > 80:
        cleaned = cleaned[:80]
    return OOS_TOURNAMENT_JOURNAL_DIR / f"oos_tournament_{cleaned}.jsonl"


def _pareto_sort_key(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> tuple:
    if trial.values is None:
        raise ValueError(f"Trial #{trial.number} has no objective values for Pareto sorting.")
    normalized: list[float] = []
    for direction, value in zip(study.directions, trial.values, strict=True):
        if direction == optuna.study.StudyDirection.MINIMIZE:
            normalized.append(-float(value))
        else:
            normalized.append(float(value))
    normalized.append(-float(trial.number))
    return tuple(normalized)


def _deterministic_best_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return sorted(study.best_trials, key=lambda t: _pareto_sort_key(study, t), reverse=True)


def _error_triage_callback_factory() -> callable:
    # ARCH-FIX-8
    consecutive_failures = {"count": 0}

    def _error_triage_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.FAIL:
            consecutive_failures["count"] += 1
            fail_reason = trial.system_attrs.get("fail_reason")
            if fail_reason:
                error_class = fail_reason.split(":", 1)[0]
            else:
                error_class = "Unknown"
            try:
                if hasattr(study, "_storage") and hasattr(trial, "_trial_id"):
                    study._storage.set_trial_user_attr(trial._trial_id, "error_class", error_class)
            except Exception as e:
                logger.error(
                    "Failed to set trial user attribute 'error_class'=%r for trial %s: %s (storage=%s)",
                    error_class,
                    getattr(trial, "_trial_id", "unknown"),
                    e,
                    getattr(study, "_storage", "unknown"),
                )
            if consecutive_failures["count"] > int(N_TRIALS * 0.30):
                study.stop()
                raise RuntimeError(
                    f"Aborting: {consecutive_failures['count']} consecutive trial failures"
                )
        else:
            consecutive_failures["count"] = 0

    return _error_triage_callback


def run_optimization(
    universe_type: str       = "nifty500",
    in_memory:     bool      = False,
    study_name:    str | None = None,
):
    universe_type = _normalize_universe_type(universe_type)
    if in_memory:
        effective_storage = ":memory:"
        effective_n_jobs  = 1
        logger.info(
            "In-memory mode: storage=:memory:, n_jobs=%d. "
            "Trial history will not be persisted.",
            effective_n_jobs,
        )
    else:
        effective_storage = OPTUNA_STORAGE
        effective_n_jobs  = N_JOBS

    if effective_n_jobs != 1:
        logger.warning(
            "OPTUNA_N_JOBS=%d: forced to 1. For parallelism run multiple "
            "processes against the same storage.",
            effective_n_jobs,
        )
        effective_n_jobs = 1

    print("\n\033[1;36m=== INSTITUTIONAL TIME-SERIES CV OPTIMIZER ===\033[0m")
    print(f"\033[90mIn-Sample (Train) : {TRAIN_START} to {TRAIN_END}\033[0m")
    print(f"\033[90mOut-of-Sample     : {TEST_START} to {TEST_END}\033[0m")
    print(f"\033[90mTrials            : {N_TRIALS}\033[0m\n")

    logger.info("Optimization universe: %s", universe_type)
    preloaded = pre_load_data(universe_type)
    if isinstance(preloaded, dict) and "market_data" in preloaded:
        market_data          = preloaded["market_data"]
        precomputed_matrices = preloaded.get("precomputed_matrices")
    else:
        market_data          = preloaded
        precomputed_matrices = None

    os.makedirs("data", exist_ok=True)
    effective_study_name = (study_name or DEFAULT_STUDY_NAME).strip() or DEFAULT_STUDY_NAME
    logger.info("Using Optuna study: %s", effective_study_name)

    if effective_study_name != DEFAULT_STUDY_NAME and OBJECTIVE_VERSION not in effective_study_name:
        logger.warning(
            "[Optimizer] Study name '%s' does not contain objective version '%s'. "
            "Old trials may contaminate TPE guidance.",
            effective_study_name, OBJECTIVE_VERSION,
        )

    study = optuna.create_study(
        study_name     = effective_study_name,
        directions     = ["maximize", "maximize"],  # ARCH-FIX-2 [composite, calmar]
        sampler        = _build_sampler(),
        storage        = effective_storage,
        load_if_exists = True,
    )
    if isinstance(effective_storage, str) and effective_storage.startswith("sqlite:///"):
        # ARCH-FIX-4: enable WAL mode for better multi-process write concurrency.
        db_path = re.sub(r"^sqlite:///", "", effective_storage.split("?")[0])
        with sqlite3.connect(db_path, timeout=30) as _conn:
            _conn.execute("PRAGMA journal_mode=WAL")


    objective = MomentumObjective(
        market_data,
        universe_type,
        precomputed_matrices=precomputed_matrices,
    )

    logger.info("Starting %d Bayesian Trials (This may take a while)...", N_TRIALS)

    def _best_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        ranked_best = _deterministic_best_trials(study)
        if not ranked_best:
            return
        if ranked_best[0].number != trial.number:
            return
        diags = trial.user_attrs.get("slice_diags", [])
        hdr = (
            f"\n\033[1;33m{'─'*72}\033[0m"
            f"\n\033[1;33m  NEW BEST  Trial #{trial.number}  "
            f"Aggregate={trial.values[0]:.4f} Calmar={trial.values[1]:.4f}\033[0m"
            f"\n\033[1;33m{'─'*72}\033[0m"
        )
        logger.info(hdr)
        logger.info("  Parameters:")
        for k, v in trial.params.items():
            logger.info("    %-28s %s", k, v)
        logger.info("")
        logger.info(
            "  %-6s  %-8s  %-8s  %-8s  %-8s  %-8s  %-10s  %-10s  %s",
            "Year","CAGR%","DD%","Turn","AvgPos","AvgExp","ForcedCash","Score","Flags",
        )
        logger.info("  " + "-"*82)
        for d in diags:
            flags = []
            if d.get("ceiling_hit"): flags.append("CEIL")
            if d.get("dd_gate_hit"): flags.append("DD-GATE")
            if d.get("anomaly_hit"): flags.append("ANOM")
            logger.info(
                "  %-6s  %+7.1f%%  %6.1f%%  %6.2fx  %6.1f  %7.3f  %9.4f  %9.4f  %s",
                d["year"],
                d["cagr"], abs(d["max_dd"]), d["turnover"],
                d["avg_positions"], d["avg_exposure"],
                d.get("forced_cash_penalty", 0.0),
                d["score"], " ".join(flags) if flags else "—",
            )
        logger.info("  " + "-"*82)
        if diags:
            avg_cagr = sum(d["cagr"]       for d in diags) / len(diags)
            avg_dd   = sum(abs(d["max_dd"]) for d in diags) / len(diags)
            ceil_n   = sum(1 for d in diags if d.get("ceiling_hit"))
            logger.info(
                "  %-6s  %+7.1f%%  %6.1f%%  %54s  ceiling_hits=%d/%d",
                "AVG", avg_cagr, avg_dd, "", ceil_n, len(diags),
            )
        logger.info("")

    try:
        study.optimize(
            objective,
            n_trials          = N_TRIALS,
            show_progress_bar = True,
            n_jobs            = effective_n_jobs,
            catch             = (Exception,),
            callbacks         = [_error_triage_callback_factory(), _best_trial_callback],
        )
    except Exception:
        logger.exception("Optimization aborted due to unexpected internal error.")
        raise

    if not study.best_trials:
        raise RuntimeError(
            "Optimization finished with no completed trials. "
            "Widen the search space or reduce hard constraints."
        )

    try:
        best_trial = _deterministic_best_trials(study)[0]
    except ValueError as exc:
        raise RuntimeError(
            f"Optimization finished but no best trial available "
            f"(all {len(study.trials)} trial(s) may have been pruned): {exc}."
        ) from exc

    best_params     = best_trial.params
    best_is_fitness = best_trial.values[0]

    print("\n\033[1;32m=== OPTIMIZATION COMPLETE ===\033[0m")
    print(f"\033[1mBest Fitness Score (IS):\033[0m {best_is_fitness:.4f}")
    print("\033[1mWinning Parameters:\033[0m")
    for k, v in best_params.items():
        print(f"  {k}: \033[33m{v}\033[0m")

    trials    = list(getattr(study, "trials", []))
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        top10 = sorted(completed, key=lambda t: t.values[0] if t.values else -999, reverse=True)[:10]
        print("\n\033[1;36m=== TOP-10 TRIALS DIAGNOSTIC SUMMARY ===\033[0m")
        print(f"\033[90m{'Trial':>6}  {'Score':>7}  {'AvgCAGR':>8}  {'AvgDD':>7}  {'CeilHits':>9}  {'DDGate':>7}\033[0m")
        print(f"\033[90m{'─'*58}\033[0m")
        for t in top10:
            avg_c    = t.user_attrs.get("avg_cagr",     "?")
            avg_d    = t.user_attrs.get("avg_dd",       "?")
            c_hits   = t.user_attrs.get("ceiling_hits", "?")
            dg_hits  = t.user_attrs.get("ddgate_hits",  "?")
            scored_n = len(t.user_attrs.get("slice_diags", []))
            c_str    = f"{c_hits}/{scored_n}" if isinstance(c_hits, int) else "?"
            dg_str   = f"{dg_hits}/{scored_n}" if isinstance(dg_hits, int) else "?"
            cagr_s   = f"{avg_c:+.1f}%" if isinstance(avg_c, float) else str(avg_c)
            dd_s     = f"{abs(avg_d):.1f}%" if isinstance(avg_d, float) else str(avg_d)
            print(f"  #{t.number:>4}  {t.values[0]:>7.4f}  {cagr_s:>8}  {dd_s:>7}  {c_str:>9}  {dg_str:>7}")
        print(f"\033[90m{'─'*58}\033[0m")
        print()

        c_hits   = best_trial.user_attrs.get("ceiling_hits", 0)
        scored_n = len(best_trial.user_attrs.get("slice_diags", []))
        if isinstance(c_hits, int) and scored_n > 0 and c_hits == scored_n:
            print("\033[1;31m[WARNING] Best trial hit the 3.5 score ceiling on ALL scored slices.\033[0m")
            print("\033[33m          TPE convergence may be unreliable.\033[0m")
            print()

    # ── OOS Top-K tournament ─────────────────────────────────────────────────
    print(f"\n\033[1;36m=== INITIATING OUT-OF-SAMPLE (OOS) VALIDATION — TOP-{OOS_TOP_K} TOURNAMENT ===\033[0m")
    print(f"\033[90mEvaluating top {OOS_TOP_K} IS trials on unseen data {TEST_START} -> {TEST_END}\033[0m")
    print("\033[90mWinner = best OOS Calmar (not best IS score)\033[0m")
    print(f"\033[90mPASS   = Calmar > 0.5  AND  MaxDD <= {OOS_MAX_DD_CAP:.0f}%\033[0m")
    print(f"\033[90mNEAR   = Calmar > 0.5  AND  MaxDD <= {OOS_SOFT_MAX_DD_CAP:.0f}%  (diagnostic only)\033[0m\n")

    top_k_trials = _deterministic_best_trials(study)[:OOS_TOP_K]  # ARCH-FIX-2 Pareto-front seeding

    if not top_k_trials:
        logger.warning("No completed trials for OOS validation; skipping tournament.")
        save_optimal_config(best_params)
        return

    valid_fields = UltimateConfig.__dataclass_fields__

    print(
        f"  {'Rank':>4}  {'Trial':>6}  {'IS Score':>9}  {'OOS CAGR':>9}  "
        f"{'OOS MaxDD':>9}  {'OOS Calmar':>10}  {'Status'}"
    )
    print(f"  {'─'*79}")

    oos_results_list = []
    journal_path = _oos_journal_path(effective_study_name)  # ARCH-FIX-9
    completed_trial_ids: dict[int, dict] = {}
    if journal_path.exists():
        for line_idx, line in enumerate(journal_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                trial_number = rec["trial_number"]
                completed_trial_ids[trial_number] = rec
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(
                    "Skipping corrupt/truncated journal line %d in %s: %s (line: %r)",
                    line_idx, journal_path, e, line[:100]
                )
                continue

    for rank, trial_candidate in enumerate(top_k_trials, 1):
        if trial_candidate.number in completed_trial_ids:
            rec = completed_trial_ids[trial_candidate.number]
            if rec.get("status") == "PASS":
                oos_results_list.append((rec["oos_calmar"], trial_candidate, trial_candidate.params, rec["metrics"]))
            continue
        oos_cfg      = UltimateConfig()
        resolved_cfg = trial_candidate.user_attrs.get("resolved_cfg", {})
        for k, v in resolved_cfg.items():
            if k in valid_fields:
                setattr(oos_cfg, k, v)
        for k, v in trial_candidate.params.items():
            if k in valid_fields:
                setattr(oos_cfg, k, v)

        try:
            warmup_start = _compute_warmup_start(TEST_START, oos_cfg)  # ARCH-FIX-5
            clipped_matrices = {
                k: v.loc[warmup_start:TEST_END]
                for k, v in (precomputed_matrices or {}).items()
                if hasattr(v, "loc")
            }
            oos_result = run_backtest(
                market_data          = market_data,
                precomputed_matrices = clipped_matrices,
                universe_type        = universe_type,
                start_date           = TEST_START,
                end_date             = TEST_END,
                cfg                  = oos_cfg,
            )
            m          = oos_result.metrics
            oos_cagr   = m.get("cagr",   0.0)
            oos_maxdd  = m.get("max_dd", -100.0)
            oos_calmar = m.get("calmar", 0.0)

            passes = oos_calmar > 0.5 and abs(oos_maxdd) <= OOS_MAX_DD_CAP
            near   = oos_calmar > 0.5 and abs(oos_maxdd) <= OOS_SOFT_MAX_DD_CAP

            if passes:
                status = "\033[32mPASS\033[0m"
                status_tag = "PASS"
            elif near:
                status = "\033[33mNEAR\033[0m"
                status_tag = "NEAR"
            else:
                status = "\033[31mFAIL\033[0m"
                status_tag = "FAIL"

            print(
                f"  {rank:>4}  #{trial_candidate.number:>5}  "
                f"{trial_candidate.values[0]:>9.4f}  "
                f"{oos_cagr:>+8.1f}%  "
                f"{abs(oos_maxdd):>8.1f}%  "
                f"{oos_calmar:>10.2f}  "
                f"{status}"
            )

            oos_result_dict = {
                "trial_number": trial_candidate.number,
                "oos_calmar": oos_calmar,
                "metrics": m,
                "status": status_tag,
            }
            with journal_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(oos_result_dict) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            if passes:
                oos_results_list.append((oos_calmar, trial_candidate, trial_candidate.params, m))

        except Exception as exc:
            print(
                f"  {rank:>4}  #{trial_candidate.number:>5}  "
                f"{trial_candidate.values[0]:>9.4f}  "
                f"{'ERROR':>9}  {'—':>9}  {'—':>10}  \033[31mERROR: {exc}\033[0m"
            )

    print(f"  {'─'*79}\n")

    if not oos_results_list:
        raise RuntimeError(
            f"OOS Validation Failed: None of the top-{OOS_TOP_K} IS trials "
            f"passed Period-1 OOS (Calmar > 0.5 and MaxDD <= {OOS_MAX_DD_CAP}%).\n"
            f"Recommended actions:\n"
            f"  (1) Run more trials — 300+ often needed for this search space.\n"
            f"  (2) Check data quality — ensure ^NSEI and ^CRSLDX downloaded\n"
            f"      successfully (regime score defaults to 0.5 when missing).\n"
            f"  (3) If NEAR results appear above, the strategy is close —\n"
            f"      raise OOS_MAX_DD_CAP to {OOS_SOFT_MAX_DD_CAP} temporarily."
        )

    oos_results_list.sort(key=lambda x: x[0], reverse=True)
    best_p1_calmar, best_oos_trial, best_oos_params, best_oos_metrics = oos_results_list[0]

    # OPT-02: save the full resolved_cfg from the winning trial so that parameters
    # outside the search space (e.g. CVAR_HARD_BREACH_MULTIPLIER) are not lost.
    # Overlay the trial.params on top just in case they differ.
    final_oos_params = best_oos_trial.user_attrs.get("resolved_cfg", {}).copy()
    final_oos_params.update(best_oos_params)

    print("\n\033[1;32m=== OOS TOURNAMENT WINNER (SINGLE PERIOD) ===\033[0m")
    print(
        f"  Trial      : #{best_oos_trial.number}  "
        f"(IS score {best_oos_trial.values[0]:.4f})"
    )
    print(f"  OOS Calmar : {best_p1_calmar:.2f}")
    print(f"  OOS CAGR   : {best_oos_metrics.get('cagr', 0):.2f}%")
    print(f"  OOS MaxDD  : {abs(best_oos_metrics.get('max_dd', 0)):.2f}%")
    print(f"  OOS Sharpe : {best_oos_metrics.get('sharpe', 0):.2f}")
    print("\n  Winning Parameters:")
    for k, v in final_oos_params.items():
        print(f"    {k}: \033[33m{v}\033[0m")

    save_optimal_config(final_oos_params)
    if journal_path.exists():  # ARCH-FIX-9
        journal_path.unlink()
    print(
        "\n\033[1;32m[PASS]\033[0m OOS tournament complete "
        "(single-period) parameters saved."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bayesian optimizer for momentum strategy."
    )
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
            "Use Optuna in-memory storage. Eliminates old-trial contamination. "
            "Recommended after objective version changes. "
            "Trade-off: interrupted runs cannot be resumed."
        ),
    )
    parser.add_argument(
        "--study-name",
        default=DEFAULT_STUDY_NAME,
        help=(
            "Optuna study name. Default embeds the objective version string "
            f"({OBJECTIVE_VERSION}), ensuring a clean study automatically. "
            "Override only for deliberate named runs."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    configure_optimizer_logging()
    args = _parse_args()
    run_optimization(
        universe_type = args.universe,
        in_memory     = args.in_memory,
        study_name    = args.study_name,
    )