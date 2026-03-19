"""
optimizer.py — Institutional Bayesian Time-Series CV Optimizer v11.48
====================================================================
Automates the discovery of optimal risk and momentum parameters using Optuna.
Uses expanding-window time-series cross-validation for parameter selection,
followed by a true holdout Out-of-Sample (OOS) validation period.

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
            value = value.strip()
            # BUG-FIX-DOTENV: strip inline comments before removing quotes.
            # Previously "API_KEY=abc123 # comment" left " # comment" in the value
            # because strip(chr(34)) only removes surrounding quote chars.
            # Quoted values (double or single) preserve literal # characters.
            if value and value[0] in ('"', "'"):
                # Quoted: remove surrounding matching quotes only.
                q = value[0]
                if value.endswith(q) and len(value) >= 2:
                    value = value[1:-1]
            else:
                # Unquoted: strip trailing inline comment.
                for _sep in (" #", "\t#"):
                    if _sep in value:
                        value = value[:value.index(_sep)].rstrip()
                        break
            if key:
                os.environ.setdefault(key, value)

_load_dotenv_if_present()

from momentum_engine import UltimateConfig, OptimizationError
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
# FIX-OOS-CUTOFF: TEST_END is now a fixed date, not today().
# Using today() means every optimizer run silently expands the OOS window:
# a run on Monday and a run on Friday test against different periods, making
# "OOS" results incomparable and increasingly in-sample over time.
# Set this once, leave it until you have 6+ months of live performance data,
# then advance it deliberately to a new fixed date. The env-var override
# allows CI / exploratory runs to use a different cutoff without code changes.
TEST_END = os.environ.get("OPTIMIZER_OOS_CUTOFF", "2025-12-31")

N_TRIALS       = 200
OOS_MAX_DD_CAP = 35.0
OOS_TOP_K = 10

# Minimum IS calendar days a fold must have before the OOS window.
# 252 trading days * 1.4 calendar-day multiplier ≈ 353 days; use 365 for safety.
_MIN_IS_CALENDAR_DAYS = 365

SEARCH_SPACE_BOUNDS = {
    "HALFLIFE_FAST":    (10, 30, 2),
    "HALFLIFE_SLOW":    (40, 100, 5),
    "CONTINUITY_BONUS": (0.06, 0.30, 0.03),
    "RISK_AVERSION":    (10.0, 20.0, 0.5),
    "CVAR_DAILY_LIMIT": (0.040, 0.070, 0.005),
    "CVAR_LOOKBACK":    (50, 120, 10),
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
        return TPESampler(n_ei_candidates=24, multivariate=True)
    return TPESampler(seed=int(OPTUNA_SEED), n_ei_candidates=24, multivariate=True)

# ─── Objective Function ───────────────────────────────────────────────────────

def _iter_wfo_slices(train_start: str, train_end: str):
    """
    Yield (is_start, is_end, oos_start, oos_end) tuples for walk-forward folds.

    FIX-MB-OPT-01: Folds where the IS window is shorter than _MIN_IS_CALENDAR_DAYS
    are skipped. The first fold (year == TRAIN_START.year + 1) often has only ~1
    calendar year of IS data, which is too thin when CVAR_LOOKBACK is at its upper
    search-space bound (150 days), causing near-zero position counts and depressed
    fitness scores that bias TPE toward low-lookback configurations.
    """
    start = pd.Timestamp(train_start)
    end   = pd.Timestamp(train_end)
    for y in range(start.year + 1, end.year + 1):
        oos_start = pd.Timestamp(f"{y}-01-01")
        oos_end   = min(pd.Timestamp(f"{y}-12-31"), end)
        is_start  = start
        is_end    = oos_start - pd.Timedelta(days=1)

        # FIX-MB-OPT-01: skip folds where IS window is too thin
        is_calendar_days = (is_end - is_start).days
        if is_calendar_days < _MIN_IS_CALENDAR_DAYS:
            logger.debug(
                "[WFO] Skipping fold OOS=%d: IS window only %d calendar days "
                "(minimum %d). Too thin for upper CVAR_LOOKBACK bound.",
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
) -> tuple[float, dict]:
    """
    Compute a scalar fitness score plus a diagnostics dict for logging.

    FIX-MB-DDPENALTY / FIX-MB2-DDPENALTY2: dd_penalty is a flat quadratic
    correction independent of concentration_mult. concentration risk is fully
    covered by risk_penalty (which scales with concentration_mult).

    Returns
    -------
    score : float  — clipped to [-2.0, 3.5 asymptote] via Michaelis-Menten saturation
    diag  : dict   — all intermediate values for logging

    Note on score ceiling: the Michaelis-Menten formula (_K * raw) / (_K + raw)
    approaches _K=3.5 asymptotically as raw→∞. The upper bound is an asymptote,
    not a hard clip — the docstring previously said "clipped to 3.5" which was
    imprecise. The lower bound IS a hard floor at -2.0.
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

    _pos_deficit = max(0.0, 6.0 - avg_positions)
    concentration_mult = 1.0 + _pos_deficit * 0.30

    # FIX-NEW-OPT-01: when sortino is NaN (no downside returns in the IS window,
    # e.g. an all-positive bull-year fold), the previous code mapped NaN → 0.0
    # via the "or 0.0" default, then clamped to quality=0.50 — the worst-case
    # floor.  This penalised the best IS folds and biased TPE toward noisier
    # strategies that happen to produce some losses and thus a finite Sortino.
    # NaN means "insufficient downside data", not "bad Sortino" — treat it as
    # neutral (1.0) so it neither rewards nor penalises the fold.
    import math as _math
    if sortino is None or not _math.isfinite(sortino):
        sortino_quality = 1.0
    else:
        sortino_quality = min(max(sortino / 2.5, 0.50), 1.15)

    risk_penalty = (max_dd + (avg_cvar * 100.0 * 2.0) + 1.0) * concentration_mult

    IS_DD_GATE        = 40.0
    IS_DD_PENALTY_PCT = 15.0

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
    # FIX-MB-DDPENALTY / FIX-MB2-DDPENALTY2: flat quadratic, no concentration scaling.
    dd_penalty = (dd_excess ** 2) / 100.0

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

        trial_id = getattr(trial, "number", 0)

        # FIX-MB-M-02: accumulate bad-fold counts rather than pruning on the
        # first dd_gate_hit or anomaly_hit.  A strategy that performs well in
        # all years except one recessionary year with a 41% drawdown was
        # previously eliminated entirely, even if its aggregate fitness over
        # 4 healthy years beat every alternative.  We now collect all fold
        # scores and prune only if MORE THAN ONE fold triggers a gate.
        n_gate_hits = 0

        for wf_is_start, _, wf_oos_start, wf_oos_end in _iter_wfo_slices(TRAIN_START, TRAIN_END):
            oos_year = pd.Timestamp(wf_oos_start).year

            # FIX-MB2-WFOSLICE / FIX-MB-H-01: slice precomputed_matrices to
            # [is_start, oos_end] so each fold's signal history is bounded on
            # both ends.  The original code only applied an upper bound
            # (:oos_end_ts), leaving the lower bound unconstrained — meaning
            # folds for OOS year 2022 could see 2018-2021 data that was the
            # OOS period of prior folds.
            fold_matrices = None
            if self.precomputed_matrices is not None:
                is_start_ts = pd.Timestamp(wf_is_start)
                oos_end_ts  = pd.Timestamp(wf_oos_end)
                fold_matrices = {}
                for key, df in self.precomputed_matrices.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        fold_matrices[key] = df.loc[is_start_ts:oos_end_ts]
                    else:
                        fold_matrices[key] = df

            oos = run_backtest(
                market_data=self.market_data,
                precomputed_matrices=fold_matrices,
                universe_type=self.universe_type,
                start_date=wf_oos_start,
                end_date=wf_oos_end,
                cfg=cfg
            )
            m = oos.metrics
            score, diag = _fitness_from_metrics(m, getattr(oos, "rebal_log", pd.DataFrame()))

            diag["year"]     = oos_year
            diag["excluded"] = False  # FIX-MB-OPT-01: thin folds skipped at iterator level
            diag["eq_start"] = wf_oos_start
            diag["eq_end"]   = wf_oos_end
            slice_diags.append(diag)

            _ceiling_tag  = " ⚠ CEILING HIT" if diag["ceiling_hit"] else ""
            _ddgate_tag   = " ⚠ DD-GATE (>40%)" if diag["dd_gate_hit"] else ""
            _anomaly_tag  = " ⚠ ANOMALOUS-RETURNS" if diag.get("anomaly_hit") else ""
            logger.info(
                "[Trial %s | %d] CAGR=%+.1f%%  DD=%.1f%%  Turn=%.2fx  "
                "AvgExp=%.2f  AvgPos=%.1f  AvgCVaR=%.3f%%  "
                "RiskPenalty=%.2f  ExpPenalty=%.2f  DDPenalty=%.4f  RawScore=%.4f  Score=%.4f%s%s%s",
                trial_id, oos_year,
                diag["cagr"], abs(diag["max_dd"]), diag["turnover"],
                diag["avg_exposure"], diag["avg_positions"], diag["avg_cvar_pct"],
                diag["risk_penalty"], diag["exposure_penalty"], diag.get("dd_penalty", 0.0),
                diag["raw_score"], diag["score"],
                _ceiling_tag, _ddgate_tag, _anomaly_tag,
            )

            if not pd.notna(score):
                raise optuna.TrialPruned()

            # FIX-MB-M-02: count gate hits; prune only when MORE THAN ONE fold
            # triggers dd_gate_hit or anomaly_hit.  A single bad fold is
            # tolerated as market noise; only pervasively bad strategies prune.
            # NaN-score and score <= -2.0 remain immediate hard prunes.
            is_gate_hit = diag.get("dd_gate_hit") or diag.get("anomaly_hit") or score <= -2.0
            if is_gate_hit:
                n_gate_hits += 1
                if n_gate_hits > 1:
                    raise optuna.TrialPruned()
                # Single gate hit: log and skip this fold's score from aggregate.
                logger.debug(
                    "[Trial %s | %d] Single gate-hit fold tolerated "
                    "(dd_gate=%s anomaly=%s score=%.4f); excluding from aggregate.",
                    trial_id, oos_year,
                    diag.get("dd_gate_hit"), diag.get("anomaly_hit"), score,
                )
                continue

            scores.append(float(score))

        if not scores:
            raise optuna.TrialPruned()

        aggregate = float(sum(scores) / len(scores))

        avg_cagr       = sum(d["cagr"]         for d in slice_diags) / len(slice_diags)
        avg_dd         = sum(abs(d["max_dd"])   for d in slice_diags) / len(slice_diags)
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
        if cvar_bounds:
            cfg.CVAR_LOOKBACK = int(cvar_bounds[1])
        # FIX-MB2-CFGCOPY: Removed dead cfg._pre_load_padding_days assignment.

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

    # FIX-WARMUP-FETCH: pre_load_data must fetch from the computed warmup start,
    # not from TRAIN_START.  _compute_warmup_start requests data hundreds of
    # calendar days before TRAIN_START so that slow EMAs (HALFLIFE_SLOW up to 120)
    # and Ledoit-Wolf covariance matrices are fully initialised for the first WFO
    # fold.  Without this, the first fold's fitness score is subtly biased by
    # cold-start EMA values — overweighting strategies that happen to look good
    # on thin early history.
    _pre_load_cfg = cfg if cfg is not None else UltimateConfig()
    _actual_warmup_start = _compute_warmup_start(TRAIN_START, _pre_load_cfg)
    logger.info(
        f"Fetching {len(symbols_to_fetch)} symbols from {_actual_warmup_start} "        f"(warmup) to {TEST_END}..."
    )
    kwargs = dict(tickers=symbols_to_fetch, required_start=_actual_warmup_start, required_end=TEST_END)
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


def _validate_optimal_config(params: dict) -> list[str]:
    """
    PROD-FIX-2: Validate cross-field constraints before persisting optimal config.

    Returns a list of violation messages (empty = valid).  Callers should refuse
    to persist configs that contain violations, since load_optimized_config applies
    them at startup and a bad value could put the live system in an unsafe state
    (e.g. CVAR_DAILY_LIMIT below the regulatory floor, MAX_POSITIONS=1 creating
    a fully-concentrated portfolio, HALFLIFE_FAST > HALFLIFE_SLOW inverting the
    signal).
    """
    violations: list[str] = []

    cvar = params.get("CVAR_DAILY_LIMIT")
    if cvar is not None:
        if not isinstance(cvar, (int, float)) or cvar <= 0.0:
            violations.append(f"CVAR_DAILY_LIMIT must be > 0; got {cvar!r}")
        elif cvar > 0.50:
            violations.append(f"CVAR_DAILY_LIMIT={cvar:.3f} exceeds 50% — implausibly loose risk limit")

    max_pos = params.get("MAX_POSITIONS")
    if max_pos is not None and (not isinstance(max_pos, int) or max_pos < 2):
        violations.append(f"MAX_POSITIONS must be an integer >= 2; got {max_pos!r}")

    hf = params.get("HALFLIFE_FAST")
    hs = params.get("HALFLIFE_SLOW")
    if hf is not None and hs is not None:
        if hf > hs:
            violations.append(
                f"HALFLIFE_FAST ({hf}) > HALFLIFE_SLOW ({hs}) — would invert momentum signal"
            )

    exp_floor = params.get("MIN_EXPOSURE_FLOOR")
    if exp_floor is not None:
        if not isinstance(exp_floor, (int, float)) or not (0.0 <= exp_floor <= 1.0):
            violations.append(f"MIN_EXPOSURE_FLOOR={exp_floor!r} must be in [0, 1]")

    max_w = params.get("MAX_SINGLE_NAME_WEIGHT")
    if max_w is not None:
        if not isinstance(max_w, (int, float)) or not (0.01 <= max_w <= 1.0):
            violations.append(f"MAX_SINGLE_NAME_WEIGHT={max_w!r} must be in [0.01, 1.0]")

    risk_av = params.get("RISK_AVERSION")
    if risk_av is not None and (not isinstance(risk_av, (int, float)) or risk_av <= 0):
        violations.append(f"RISK_AVERSION must be > 0; got {risk_av!r}")

    return violations


def save_optimal_config(best_params: dict, filepath: str = "data/optimal_cfg.json"):
    # PROD-FIX-2: Validate before writing.  A bad config persisted here will be
    # applied at next startup via load_optimized_config; blocking unsafe values
    # here is the last line of defence before they reach a live rebalance.
    violations = _validate_optimal_config(best_params)
    if violations:
        msg = "; ".join(violations)
        raise ValueError(
            f"save_optimal_config: refusing to persist invalid config ({msg}).  "
            f"Params: {best_params}"
        )

    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # PROD-FIX-L03: resolve output path before splitting to avoid ambiguity when
    # filepath has no directory component (os.path.dirname returns '' which the
    # NamedTemporaryFile dir kwarg may resolve to a different location than CWD).
    resolved = os.path.abspath(filepath)
    resolved_dir = os.path.dirname(resolved)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
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
            "OPTUNA_N_JOBS=%d: for this single-process invocation n_jobs is forced to 1. "
            "For true parallelism, run multiple optimizer processes pointing at the same "
            "shared SQLite/RDB storage — each process uses n_jobs=1.",
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

    # FIX-MB-OPT-09 WARNING: warn when a custom --study-name is passed that does
    # not embed OBJECTIVE_VERSION, because old trials from a different objective
    # version will silently contaminate the new optimization run.
    if effective_study_name != DEFAULT_STUDY_NAME and OBJECTIVE_VERSION not in effective_study_name:
        logger.warning(
            "[Optimizer] Study name '%s' does not contain the current objective "
            "version string '%s'.  If this study was created with a different "
            "objective function, previously completed trials will bias TPE guidance "            "toward parameters calibrated for the old objective.  Rename the study "
            "or use the default name to start a clean optimization track.",
            effective_study_name, OBJECTIVE_VERSION,
        )

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

    # FIX-MB-OPT-06: re-read best_trial AFTER study.optimize() returns so that
    # the OOS tournament references the true best trial, not a pre-run capture
    # that may have been None or from a prior loaded study.
    # FIX-NEW-OPT-02: study.best_trial raises ValueError when every trial was
    # pruned (possible when all folds trigger anomaly_hit or IS_DD_GATE).  The
    # study.best_trials guard above catches the "no completed trials" case, but
    # best_trial itself can still raise if Optuna considers pruned trials
    # "complete" in some storage backends.  Wrap with an explicit try/except so
    # the error message is actionable rather than a bare ValueError traceback.
    try:
        best_trial = study.best_trial
    except ValueError as exc:
        raise RuntimeError(
            "Optimization finished but no best trial is available "
            f"(all {len(study.trials)} trial(s) may have been pruned): {exc}. "
            "Widen the search space, reduce CVAR_DAILY_LIMIT lower bound, "
            "or check that the training data covers the full warmup window."
        ) from exc
    best_params = best_trial.params
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
            scored_n = len(t.user_attrs.get("slice_diags", []))
            c_str   = f"{c_hits}/{scored_n}" if isinstance(c_hits, int) else "?"
            dg_str  = f"{dg_hits}/{scored_n}" if isinstance(dg_hits, int) else "?"
            cagr_s  = f"{avg_c:+.1f}%" if isinstance(avg_c, float) else str(avg_c)
            dd_s    = f"{abs(avg_d):.1f}%" if isinstance(avg_d, float) else str(avg_d)
            print(f"  #{t.number:>4}  {t.value:>7.4f}  {cagr_s:>8}  {dd_s:>7}  {c_str:>9}  {dg_str:>7}")
        print(f"\033[90m{'─'*58}\033[0m")
        print()

        c_hits   = best_trial.user_attrs.get("ceiling_hits", 0)
        scored_n = len(best_trial.user_attrs.get("slice_diags", []))
        if isinstance(c_hits, int) and scored_n > 0 and c_hits == scored_n:
            print("\033[1;31m[WARNING] Best trial hit the 3.5 score ceiling on ALL scored slices.\033[0m")
            print("\033[33m          This means the optimizer cannot distinguish between parameter sets\033[0m")
            print("\033[33m          in the top range — TPE convergence may be unreliable.\033[0m")
            print()

    # OOS Top-K tournament
    print(f"\n\033[1;36m=== INITIATING OUT-OF-SAMPLE (OOS) VALIDATION — TOP-{OOS_TOP_K} TOURNAMENT ===\033[0m")
    print(f"\033[90mEvaluating top {OOS_TOP_K} IS trials on unseen data {TEST_START} → {TEST_END}\033[0m")
    print(f"\033[90mWinner = best OOS Calmar (not best IS score)\033[0m\n")

    top_k_trials = sorted(completed, key=lambda t: t.value or -999, reverse=True)[:OOS_TOP_K]

    if not top_k_trials:
        logger.warning("No completed trials available for OOS validation; skipping OOS tournament.")
        save_optimal_config(best_params)
        return

    _rs = "\u20b9" if _stdout_supports_rupee() else "Rs."
    valid_fields = UltimateConfig.__dataclass_fields__

    # FIX-MB-OPT-03: print abs(oos_maxdd) so the MaxDD column shows a positive
    # percentage consistent with the column header "OOS MaxDD".
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

            # FIX-MB-OPT-03: display abs(oos_maxdd) for positive readability
            print(
                f"  {rank:>4}  #{trial_candidate.number:>5}  "
                f"{trial_candidate.value:>9.4f}  "
                f"{oos_cagr:>+8.1f}%  "
                f"{abs(oos_maxdd):>8.1f}%  "
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
    print(f"  OOS MaxDD      : {abs(best_oos_metrics.get('max_dd', 0)):.2f}%")
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
