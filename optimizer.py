"""
optimizer.py — Institutional Bayesian Time-Series CV Optimizer v11.48
====================================================================
Automates the discovery of optimal risk and momentum parameters using Optuna.
Uses expanding-window time-series cross-validation for parameter selection,
followed by a true holdout Out-of-Sample (OOS) validation period.

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
    """Load simple KEY=VALUE pairs from a local .env file if present."""
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

# Local imports from your v11.48 architecture
from momentum_engine import UltimateConfig, OptimizationError
from backtest_engine import run_backtest, apply_halt_simulation
from data_cache import load_or_fetch
from universe_manager import get_nifty500, fetch_nse_equity_universe, get_historical_universe

# Suppress solver/sklearn warnings during the thousands of iterations
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ensure stdout can handle Unicode characters (₹, etc.) on Windows where the
# default 'charmap' (cp1252) codec raises UnicodeEncodeError for U+20B9.
# errors='replace' is a safe fallback for environments that truly cannot
# represent the character; reconfigure() is a no-op on UTF-8 terminals.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # TTY doesn't support reconfigure (e.g. redirected pipe)

# Configure local logger
logger = logging.getLogger("Optimizer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("\033[90m[%(asctime)s]\033[0m %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)

# Silence noisy sub-module loggers that fire on every rebalance.
# The Optimizer logger itself stays at INFO so per-trial diagnostics are visible.
logging.getLogger("universe_manager").setLevel(logging.ERROR)
logging.getLogger("backtest_engine").setLevel(logging.ERROR)
logging.getLogger("momentum_engine").setLevel(logging.ERROR)
logging.getLogger("signals").setLevel(logging.ERROR)
logging.getLogger("data_cache").setLevel(logging.ERROR)

# ─── Optimization Configuration ───────────────────────────────────────────────

TRAIN_START = "2018-01-01"
TRAIN_END   = "2022-12-31"   # 5-year IS window generates 3 WFO slices:
                              #   2020 — COVID crash + V-recovery (black swan)
                              #   2021 — bull market (trending)
                              #   2022 — Ukraine/rate-hike bear (drawdown stress)
                              #
                              # WHY NOT 2019: The 2019 NBFC/IL&FS contagion slice
                              # always scores -7 to -8 (long-only momentum cannot
                              # survive a 50%+ mid-cap bear market). That single
                              # slice dominated the aggregate score by 40×, making
                              # Bayesian optimisation equivalent to random search —
                              # TPE had no gradient to exploit across the 2020/2021
                              # dimensions. The winning trial in the broken run (trial
                              # 75) scored highest because it produced ZERO trades in
                              # 2019 (all-cash), not because it found good parameters.
                              #
                              # WHY 2022 IN IS: The 2022 Indian bear market is the
                              # correct stress test for a momentum strategy. Including
                              # it in IS means the optimizer learns to survive it.
TEST_START  = "2023-01-01"   # True OOS: 2023-present is fully unseen data.
                              # Spans a sustained bull market — validates CAGR.
                              # 2022 bear is now the final IS slice, providing
                              # bear-market discipline without poisoning scoring.
TEST_END    = pd.Timestamp.today().strftime("%Y-%m-%d")

N_TRIALS       = 100   # 100 trials: TPE needs ~20+ warm-up then ~80 exploitation
                        # rounds across 6 dimensions to converge reliably.
# OOS window is 2023-2026 (bull market): 35% DD cap appropriate.
# A well-tuned momentum strategy should not exceed -35% in a bull regime.
OOS_MAX_DD_CAP = 35.0  # Tightened: forces optimizer to find parameters with lower drawdown

# Number of top IS trials to evaluate on OOS.
# Instead of blindly deploying the #1 IS trial (which is most likely to be
# overfit), we run all top-K candidates on the true holdout and pick the one
# with the best OOS Calmar ratio. K=10 adds ~10x the OOS compute but catches
# overfit #1 trials that collapse on unseen data while a more robust #3 or #7
# survives. The final saved config is always the OOS winner, not the IS winner.
OOS_TOP_K = 10

# Search space bounds are configurable to support high-risk/high-turnover variants.
SEARCH_SPACE_BOUNDS = {
    "HALFLIFE_FAST":    (10, 40, 5),
    "HALFLIFE_SLOW":    (50, 120, 5),
    "CONTINUITY_BONUS": (0.05, 0.30, 0.01),
    # Floor raised 5.0 → 10.0: at RA < 10 the OSQP solver concentrates into
    # 4 positions at the MAX_SINGLE_NAME_WEIGHT cap (25% × 4 = 100% allocation).
    # On the survivor-only IS universe those slots fill with genuine 2021-2022
    # outliers (Tata Elxsi +340%, NDTV +456%) — real returns, but unreproducible
    # live because concentration magnifies idiosyncratic risk and the universe
    # excludes the blow-ups that would have balanced the sample.
    # RA ≥ 10 produces 6-8 positions at 12-16% each — institutional-grade range.
    # Ceiling raised 15.0 → 20.0 to keep the high-RA region explorable.
    "RISK_AVERSION":    (10.0, 20.0, 0.5),
    # Ceiling lowered 0.090 → 0.070. At 9% daily CVaR the limit almost never
    # fires during normal NSE volatility (~1.5% avg daily vol), making it a dead
    # parameter the optimizer exploits by pinning to the upper boundary.
    # 7% allows a ~4.5σ single-day loss before triggering — still generous.
    "CVAR_DAILY_LIMIT": (0.040, 0.070, 0.005),
    # CVAR_LOOKBACK is now tunable. Shorter windows keep CVaR responsive to the
    # current regime rather than dragging crash returns for months afterward.
    "CVAR_LOOKBACK":    (60, 150, 5),
}

# Runtime knobs: optimization runs in-process (`n_jobs=1`) by default because
# Optuna's threaded `n_jobs>1` does not scale CPU-bound backtests under the GIL.
# For parallel speedup, run multiple python processes against shared storage.
N_JOBS = int(os.getenv("OPTUNA_N_JOBS", "1"))
OPTUNA_SEED = os.getenv("OPTUNA_SEED")

# Storage backend for Optuna study persistence (SQLite by default).
OPTUNA_STORAGE = os.getenv("OPTUNA_STORAGE", "sqlite:///data/optuna_study.db")

# Study naming/versioning.
# The current objective is a clipped fitness score in [-2.0, 5.0]. Reusing an
# old study (same name) from a previous objective can show impossible best
# values (e.g., >>5) and surface stale high-risk parameter sets.
OBJECTIVE_VERSION = "fitness_v11_50"  # IS DD gate + quadratic penalty added
DEFAULT_STUDY_NAME = f"Momentum_Risk_Parity_{OBJECTIVE_VERSION}"

# Plausibility guardrails: protect optimizer scoring from data/pathology-driven
# equity explosions that can otherwise dominate TPE with meaningless extremes.
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
    """
    Yield (is_start, is_end, oos_start, oos_end) tuples for each calendar year
    within the training window. The IS window bounds are yielded for documentation
    purposes and logging; the actual backtest in MomentumObjective.__call__ always
    uses start_date=oos_start so that each evaluation covers one OOS year only.
    Signal warm-up history (all pre-OOS data in market_data) is available to the
    backtest engine via BacktestEngine._run_rebalance's hist_log_rets slice.

    With TRAIN_START=2018 / TRAIN_END=2022 this yields 4 slices:
        IS window    OOS window   Regime
        2018         2019         NBFC/IL&FS stress  (score excluded — see NOTE)
        2018-2019    2020         COVID crash + V-recovery
        2018-2020    2021         bull market
        2018-2021    2022         Ukraine/rate-hike bear

    NOTE: The first slice (OOS 2019) is intentionally EXCLUDED from the fitness
    aggregate. The NBFC/IL&FS contagion made 2019 a structural bear for long-only
    momentum strategies. Including it caused it to dominate the aggregate by 40×
    (score ≈ -7 vs the other slices' ≈ ±2), turning TPE into random search.

    The 2019 OOS backtest IS still run so exceptions propagate — parameters that
    crash during that regime are pruned — but its score is not appended to `scores`.
    """
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

    Returns
    -------
    score : float
        Clipped to [-2.0, 5.0].
    diag  : dict
        All intermediate values so callers can log exactly why a score
        came out the way it did.
    """
    cagr = float(metrics.get("cagr", 0.0))
    max_dd = abs(float(metrics.get("max_dd", 100.0)))
    turnover = float(metrics.get("turnover", 0.0))
    sortino = float(metrics.get("sortino", 0.0) or 0.0)
    final_equity = float(metrics.get("final", BASE_INITIAL_CAPITAL) or BASE_INITIAL_CAPITAL)
    final_multiple = final_equity / max(BASE_INITIAL_CAPITAL, 1e-9)
    turnover_drag = turnover * 0.15  # 15 bps = 0.15% per 1x annual turnover
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

    # ── Concentration multiplier ──────────────────────────────────────────────
    # The survivor-only IS universe excludes stocks that were demoted/blown up.
    # A 4-stock portfolio has ~2× unobserved idiosyncratic risk vs an 8-stock one,
    # but the missing losers make measured CVaR/DD look artificially safe.
    # Each position below 6 raises the effective risk denominator by 30%,
    # applied multiplicatively so it cannot be absorbed by a score ceiling.
    _pos_deficit = max(0.0, 6.0 - avg_positions)
    concentration_mult = 1.0 + _pos_deficit * 0.30   # 4 pos → 1.60×, 3 pos → 1.90×

    # ── Sortino quality multiplier ────────────────────────────────────────────
    # Scales the raw score ×[0.50, 1.15].  A strategy that earns its CAGR
    # through one lucky spike (Sortino ≈ 0.5) is discounted 50%; a consistently
    # downside-safe strategy (Sortino ≈ 2.5) earns a 15% bonus.
    import math as _math
    _sortino_safe = sortino if _math.isfinite(sortino) else 0.0
    sortino_quality = min(max(_sortino_safe / 2.5, 0.50), 1.15)

    risk_penalty = (max_dd + (avg_cvar * 100.0 * 5.0) + 1.0) * concentration_mult

    # ── IS Drawdown gate ──────────────────────────────────────────────────────
    # IS trials show ~21% MaxDD but blow out to ~42% OOS. The 2x gap is caused
    # by survivorship bias in the IS universe making IS drawdowns look mild.
    # Hard gate at 30% IS DD: score collapses so TPE avoids these params.
    # Quadratic penalty above 20% creates a smooth gradient toward lower DD.
    # A param set surviving 2020+2022 with <20% IS DD is far more likely to
    # survive unseen regimes without breaching the 35% OOS DD cap.
    IS_DD_GATE        = 30.0   # hard fail above this IS MaxDD
    IS_DD_PENALTY_PCT = 20.0   # quadratic penalty kicks in above this

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
            "exposure_penalty": 0.0, "raw_score": round(raw, 6), "score": round(score, 6),
            "ceiling_hit": False, "dd_gate_hit": True, "anomaly_hit": False,
        }
        return score, diag

    # Quadratic IS DD penalty: 0 at 20%, 0.2 at 25%, 0.8 at 30%
    dd_excess  = max(0.0, max_dd - IS_DD_PENALTY_PCT)
    dd_penalty = (dd_excess ** 2) / 50.0

    exposure_penalty = 0.0 if avg_exposure >= 0.60 else (0.60 - avg_exposure) * 3.0
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
    elif max_dd > OOS_MAX_DD_CAP:
        raw = -(max_dd / 10.0)
        score = max(raw, -2.0)
        ceiling_hit = False
        dd_gate_hit = True
    else:
        raw = (cagr_net / risk_penalty) * sortino_quality - exposure_penalty - dd_penalty
        # ── Soft saturation replaces hard clip at 5.0 ────────────────────────
        # Michaelis-Menten: score approaches 3.5 asymptotically so TPE retains
        # gradient between "very good" (raw≈2) and "lucky outlier year" (raw≈14).
        # Hard clip mapped both to 5.0, erasing the signal needed for convergence.
        _K = 3.5
        score = (_K * raw / (_K + raw)) if raw > 0.0 else raw
        score = max(score, -2.0)
        ceiling_hit = False   # saturation replaces hard ceiling
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
    def __init__(self, market_data: dict, universe_type: str, search_space: dict | None = None):
        self.market_data = market_data
        self.universe_type = universe_type
        self.search_space = search_space or SEARCH_SPACE_BOUNDS

    def __call__(self, trial: optuna.Trial) -> float:
        # 1. Base Configuration
        cfg = UltimateConfig()

        # 2. Define the Search Space (Group A: Alpha & Turnover)
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
        
        # Logical Constraint: Fast must be strictly faster than slow.
        # FIX O4: With current bounds (FAST max=40, SLOW min=50) this branch is
        # unreachable — max(FAST)=40 can never exceed min(SLOW)=50. It is kept as
        # a safety net: if HALFLIFE_FAST bounds are ever widened past 50 (e.g. to
        # test longer fast windows), the constraint will automatically activate and
        # prune nonsensical FAST > SLOW combinations without needing a separate edit.
        # Do NOT remove this check when widening search bounds.
        if cfg.HALFLIFE_FAST > cfg.HALFLIFE_SLOW:
            raise optuna.TrialPruned()
            
        continuity_min, continuity_max, continuity_step = self.search_space["CONTINUITY_BONUS"]
        cfg.CONTINUITY_BONUS = trial.suggest_float(
            "CONTINUITY_BONUS", continuity_min, continuity_max, step=continuity_step
        )

        # 3. Define the Search Space (Group B: Risk Matrix)
        risk_aversion_min, risk_aversion_max, risk_aversion_step = self.search_space["RISK_AVERSION"]
        cvar_min, cvar_max, cvar_step = self.search_space["CVAR_DAILY_LIMIT"]
        cfg.RISK_AVERSION = trial.suggest_float(
            "RISK_AVERSION", risk_aversion_min, risk_aversion_max, step=risk_aversion_step
        )
        cfg.CVAR_DAILY_LIMIT = trial.suggest_float(
            "CVAR_DAILY_LIMIT", cvar_min, cvar_max, step=cvar_step
        )

        # 4. Define the Search Space (Group C: CVaR Lookback Window)
        # Shorter lookbacks keep CVaR responsive to the current regime instead of
        # dragging crash returns forward for months after the event has passed.
        cvar_lb_bounds = self.search_space.get("CVAR_LOOKBACK", (60, 150, 10))
        cvar_lb_min, cvar_lb_max, cvar_lb_step = cvar_lb_bounds
        min_required_lookback = cfg.DIMENSIONALITY_MULTIPLIER * cfg.MAX_POSITIONS
        effective_cvar_lb_min = max(int(cvar_lb_min), int(min_required_lookback))

        # No feasible CVAR_LOOKBACK exists inside the configured bounds.
        if effective_cvar_lb_min > int(cvar_lb_max):
            raise optuna.TrialPruned()

        if isinstance(trial, optuna.trial.FixedTrial) and "CVAR_LOOKBACK" not in trial.params:
            # Backward-compatible fallback for manually constructed FixedTrial
            # objects that predate the CVAR_LOOKBACK search dimension.
            cfg.CVAR_LOOKBACK = max(UltimateConfig().CVAR_LOOKBACK, effective_cvar_lb_min)
        else:
            cfg.CVAR_LOOKBACK = trial.suggest_int(
                "CVAR_LOOKBACK", effective_cvar_lb_min, cvar_lb_max, step=cvar_lb_step
            )

        if hasattr(trial, "set_user_attr"):
            trial.set_user_attr("resolved_cfg", dict(vars(cfg)))

        # 5. Expanding-window time-series CV evaluation
        scores: list[float] = []
        slice_diags: list[dict] = []   # per-slice diagnostics stored in trial user_attrs

        first_oos_year = pd.Timestamp(TRAIN_START).year + 1

        trial_id = getattr(trial, "number", 0)

        for _, _, wf_oos_start, wf_oos_end in _iter_wfo_slices(TRAIN_START, TRAIN_END):
            oos_year = pd.Timestamp(wf_oos_start).year

            # EXCLUDE 2019 from scoring (see _iter_wfo_slices docstring).
            # Still run the OOS backtest so exceptions propagate (parameters
            # that produce NaN signals in 2019 should be pruned), but do not
            # append its score to the aggregate.
            exclude_from_score = (oos_year == first_oos_year)

            oos = run_backtest(
                # MB-03 FIX (v11.48): market_data is pre-repaired by apply_halt_simulation()
                # in pre_load_data — no per-trial copy needed.  Frames are never written
                # back to in the post-repair path (close_d etc. are new Series derived from
                # .ffill() on the frame values, not in-place mutations of the frames).
                market_data=self.market_data,
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

            # ── Per-slice diagnostic log ──────────────────────────────────────
            _excluded_tag = " [EXCLUDED from score]" if exclude_from_score else ""
            _ceiling_tag  = " ⚠ CEILING HIT" if diag["ceiling_hit"] else ""
            _ddgate_tag   = " ⚠ DD-GATE (>40%)" if diag["dd_gate_hit"] else ""
            _anomaly_tag  = " ⚠ ANOMALOUS-RETURNS" if diag.get("anomaly_hit") else ""
            logger.info(
                "[Trial %s | %d%s] CAGR=%+.1f%%  DD=%.1f%%  Turn=%.2fx  "
                "AvgExp=%.2f  AvgPos=%.1f  AvgCVaR=%.3f%%  "
                "RiskPenalty=%.2f  ExpPenalty=%.2f  RawScore=%.4f  Score=%.4f%s%s%s%s",
                trial_id, oos_year, _excluded_tag,
                diag["cagr"], abs(diag["max_dd"]), diag["turnover"],
                diag["avg_exposure"], diag["avg_positions"], diag["avg_cvar_pct"],
                diag["risk_penalty"], diag["exposure_penalty"],
                diag["raw_score"], diag["score"],
                _ceiling_tag, _ddgate_tag, _anomaly_tag, "",
            )

            if not pd.notna(score):
                raise optuna.TrialPruned()

            if not exclude_from_score:
                scores.append(float(score))

        if not scores:
            raise optuna.TrialPruned()

        aggregate = float(sum(scores) / len(scores))

        # ── Per-trial summary log ─────────────────────────────────────────────
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

        # Persist per-slice diagnostics in the trial so they survive to the
        # study dashboard / CSV export and can be inspected post-run.
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
    """Loads data into RAM once to accelerate thousands of backtests."""
    logger.info("Initializing Data Pre-fetch phase...")
    normalized_universe = (universe_type or "").strip().lower()
    
    # FIX (I-06): Replaced hardcoded fallback branch to ensure the full equity 
    # universe loads accurately when targeted by the optimization wrapper.
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
        # MB-17 FIX: Also account for HALFLIFE_SLOW upper bound when computing
        # the pre-fetch padding.  EWMA needs ~4x halflife to converge; previously
        # only CVAR_LOOKBACK was considered, leaving slow-halflife trials with
        # unconverged signals at the start of the first OOS slice.
        if halflife_bounds:
            halflife_slow_max = int(halflife_bounds[1])
            halflife_calendar_days = halflife_slow_max * 4 * 365 // 252
            current_cvar_padding = max(400, cfg.CVAR_LOOKBACK * 2)
            cfg._pre_load_padding_days = max(current_cvar_padding, halflife_calendar_days)

    # MB-18 FIX: Build the optimization preload from the SAME point-in-time
    # historical membership snapshots that run_backtest uses.
    #
    # Root cause fixed:
    # pre_load_data previously fetched only today's base universe. During
    # run_backtest, the engine dynamically switches members by historical
    # rebalance date. Any historical member absent from today's list had no
    # market_data frame and was logged as "missing data", causing noisy logs
    # and optimizer OOS vs daily_workflow backtest mismatches.
    #
    # By unioning historical members across TRAIN_START→TEST_END at the exact
    # rebalance cadence, optimizer and daily_workflow now hydrate the same
    # symbol set and produce consistent replay conditions.
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

    # Ensure index data is present for regime scoring
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

    # MB-03 FIX: Pre-apply gap repair once here so the per-trial objective
    # receives already-repaired frames and needs zero per-trial allocation.
    # _repair_suspension_gaps is deterministic (ticker-hash seed), so pre-computing
    # is equivalent to re-computing on every trial while eliminating
    # 500 tickers × 300 trials × 4 slices = 600,000 DataFrame allocations per run.
    market_data = apply_halt_simulation(market_data)

    logger.info("Data pre-load complete. Commencing Bayesian Optimization.")
    return market_data


def save_optimal_config(best_params: dict, filepath: str = "data/optimal_cfg.json"):
    """Exports the winning configuration so daily_workflow.py can use it."""
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write atomically to avoid partial/truncated JSON if the process is interrupted.
    # FIX O2: Wrap os.replace in try/finally so the temp file is deleted on failure.
    # Without this, a failed os.replace (e.g. Windows file-lock on the target) leaves
    # a NamedTemporaryFile(delete=False) orphan in the data/ directory indefinitely.
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
    # ── Storage & parallelism resolution ──────────────────────────────────────
    # Priority: explicit in_memory flag > OPTUNA_STORAGE env var > SQLite default.
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
        # FIX O5: OPTUNA_N_JOBS is always forced to 1 here. CPU-bound backtests under
        # the GIL do not benefit from Optuna's thread-based n_jobs > 1.
        # For real parallelism: launch N separate `python optimizer.py` processes that
        # all point to the same SQLite or RDB storage. Optuna's study coordination
        # handles concurrent trial allocation automatically. The N_JOBS env var is
        # intentionally preserved for forward-compatibility if this path is later
        # migrated to a process-pool backend.
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
    market_data = pre_load_data(universe_type)

    # 1. Setup Optuna Study
    os.makedirs("data", exist_ok=True) # Ensure the data folder exists
    effective_study_name = (study_name or DEFAULT_STUDY_NAME).strip() or DEFAULT_STUDY_NAME
    logger.info("Using Optuna study: %s", effective_study_name)

    study = optuna.create_study(
        study_name=effective_study_name,
        direction="maximize",
        sampler=_build_sampler(),
        storage=effective_storage,      # :memory: or SQLite depending on flag/env
        load_if_exists=True
    )
     
    objective = MomentumObjective(market_data, universe_type)

    # 2. Run In-Sample Optimization
    logger.info(f"Starting {N_TRIALS} Bayesian Trials (This may take a while)...")
    def _best_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Fires after every completed trial. Prints a summary table when a new best is found."""
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if study.best_trial.number != trial.number:
            return  # not the new best

        diags = trial.user_attrs.get("slice_diags", [])
        scored = [d for d in diags if not d.get("excluded", False)]

        hdr = (
            f"\n\033[1;33m{'─'*72}\033[0m"
            f"\n\033[1;33m  NEW BEST  Trial #{trial.number}  "
            f"Aggregate={trial.value:.4f}\033[0m"
            f"\n\033[1;33m{'─'*72}\033[0m"
        )
        logger.info(hdr)

        # Parameter table
        logger.info("  Parameters:")
        for k, v in trial.params.items():
            logger.info("    %-28s %s", k, v)

        # Per-slice table
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
        # Suggest a high-conviction seed so TPE starts from a known robust region.
        """
        if hasattr(study, "enqueue_trial"):
            study.enqueue_trial({
                "HALFLIFE_FAST": 40,
                "HALFLIFE_SLOW": 110,
                "CONTINUITY_BONUS": 0.06,
                "RISK_AVERSION": 10,
                "CVAR_DAILY_LIMIT": 0.05,
                "CVAR_LOOKBACK": 150,
            })
        """
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
    best_is_fitness = study.best_value  # composite fitness score, NOT a Calmar ratio

    print(f"\n\033[1;32m=== OPTIMIZATION COMPLETE ===\033[0m")
    # FIX O3: Previous label "Best In-Sample Calmar Ratio" was wrong.
    # study.best_value is the output of _fitness_from_metrics() — a composite of
    # (cagr_net / risk_penalty) - exposure_penalty — not CAGR / |MaxDD|.
    print(f"\033[1mBest Fitness Score (IS):\033[0m {best_is_fitness:.4f}")
    print("\033[1mWinning Parameters:\033[0m")
    for k, v in best_params.items():
        print(f"  {k}: \033[33m{v}\033[0m")

    # ── Diagnostic summary: top-10 trials by score ────────────────────────────
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
        # Warn if the best trial hit the ceiling on most/all slices
        if best_trial is not None:
            c_hits   = best_trial.user_attrs.get("ceiling_hits", 0)
            scored_n = len([d for d in best_trial.user_attrs.get("slice_diags", []) if not d.get("excluded", False)])
            if isinstance(c_hits, int) and scored_n > 0 and c_hits == scored_n:
                print(
                    "\033[1;31m[WARNING] Best trial hit the 5.0 score ceiling on ALL scored slices.\033[0m"
                )
                print(
                    "\033[33m          This means the optimizer cannot distinguish between parameter sets\033[0m"
                )
                print(
                    "\033[33m          in the top range — TPE convergence may be unreliable.\033[0m"
                )
                print(
                    "\033[33m          Consider reducing CVAR_DAILY_LIMIT upper bound or adding a\033[0m"
                )
                print(
                    "\033[33m          Sharpe/Sortino term to the fitness to create more score separation.\033[0m"
                )
                print()


    # 3. Out-of-Sample Validation — Top-K tournament
    # ─────────────────────────────────────────────────────────────────────────
    # The IS winner is the trial that scored highest on the 2020-2022 training
    # window. But the IS universe has survivorship bias and limited regime
    # diversity, so the IS #1 trial is often overfit. Instead of deploying it
    # blindly we run the top OOS_TOP_K IS trials on the true holdout (2023 →
    # today) and pick the winner by OOS Calmar ratio. This is standard model
    # selection: IS ranking narrows the shortlist, OOS ranking picks the winner.
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

    # Results table header
    print(f"  {'Rank':>4}  {'Trial':>6}  {'IS Score':>9}  {'OOS CAGR':>9}  "
          f"{'OOS MaxDD':>9}  {'OOS Calmar':>10}  {'Status'}")
    print(f"  {'─'*75}")

    oos_results_list = []   # (oos_calmar, trial, params, metrics)

    for rank, trial_candidate in enumerate(top_k_trials, 1):
        # Build config from this trial's parameters
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

    # Pick the OOS winner — highest Calmar among passing trials
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

    # Note if the OOS winner differs from the IS #1 trial
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

    # Save the OOS winner (not the IS winner) as optimal config
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
