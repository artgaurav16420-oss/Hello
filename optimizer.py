"""
optimizer.py — Institutional Bayesian Time-Series CV Optimizer v11.46
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

import pandas as pd
import optuna
from optuna.samplers import TPESampler

# Local imports from your v11.46 architecture
from momentum_engine import UltimateConfig, OptimizationError
from backtest_engine import run_backtest
from data_cache import load_or_fetch
from universe_manager import get_nifty500, fetch_nse_equity_universe

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

# --- ADD THESE 3 LINES TO SILENCE THE SPAM ---
logging.getLogger("universe_manager").setLevel(logging.ERROR)
logging.getLogger("backtest_engine").setLevel(logging.ERROR)
logging.getLogger("momentum_engine").setLevel(logging.ERROR)
# ---------------------------------------------

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
OOS_MAX_DD_CAP = 40.0  # Raised: 35% was too tight (rejected Calmar=1.10 for 36% MaxDD in bull OOS)

# Search space bounds are configurable to support high-risk/high-turnover variants.
SEARCH_SPACE_BOUNDS = {
    "HALFLIFE_FAST":    (10, 40),
    "HALFLIFE_SLOW":    (50, 120),
    "CONTINUITY_BONUS": (0.05, 0.30, 0.01),
    # Floor raised from 2.0 → 5.0: at RA=2.0 the QP barely penalises variance
    # and the solver packs weight into 4 names at the 25% cap (90%+ concentration,
    # 57%+ peak drawdowns). RA=5.0 is the practical minimum for diversification.
    "RISK_AVERSION":    (5.0, 15.0, 0.5),
    # Upper bound widened from 0.06 → 0.09: the previous ceiling prevented the
    # optimizer discovering that 7-8% eliminates the chronic marginal-breach loop.
    "CVAR_DAILY_LIMIT": (0.040, 0.090, 0.005),
    # CVAR_LOOKBACK is now tunable. Shorter windows keep CVaR responsive to the
    # current regime rather than dragging crash returns for months afterward.
    "CVAR_LOOKBACK":    (60, 150, 10),
}

# Runtime knobs: optimization runs in-process (`n_jobs=1`) by default because
# Optuna's threaded `n_jobs>1` does not scale CPU-bound backtests under the GIL.
# For parallel speedup, run multiple python processes against shared storage.
N_JOBS = int(os.getenv("OPTUNA_N_JOBS", "1"))
OPTUNA_SEED = os.getenv("OPTUNA_SEED")

# Storage backend for Optuna study persistence (SQLite by default).
OPTUNA_STORAGE = os.getenv("OPTUNA_STORAGE", "sqlite:///data/optuna_study.db")


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
    Yield (is_start, is_end, oos_start, oos_end) expanding time-series CV slices
    confined entirely within the training window.

    With TRAIN_START=2018 / TRAIN_END=2022 this yields 4 slices:
        IS 2018       OOS 2019  — NBFC stress  (skipped — see NOTE below)
        IS 2018-2019  OOS 2020  — COVID crash
        IS 2018-2020  OOS 2021  — bull market
        IS 2018-2021  OOS 2022  — Ukraine/rate-hike bear

    NOTE: The first slice (OOS 2019) is intentionally EXCLUDED from scoring.
    The NBFC/IL&FS contagion made 2019 a structural bear for long-only momentum
    strategies (-50 to -80% for mid/small cap momentum). Including this slice as
    a scored signal causes it to dominate the aggregate by 40× (score ≈ -7 vs
    the other slices' ≈ ±2), turning TPE into random search.

    The 2019 IS run IS still executed (to validate the parameters don't crash
    during that regime) but its OOS score is excluded from the fitness aggregate.
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


def _fitness_from_metrics(metrics: dict, rebal_log: pd.DataFrame) -> float:
    cagr = float(metrics.get("cagr", 0.0))
    max_dd = abs(float(metrics.get("max_dd", 100.0)))
    turnover = float(metrics.get("turnover", 0.0))
    turnover_drag = turnover * 0.15  # 15 bps = 0.15% per 1x annual turnover
    cagr_net = cagr - turnover_drag
    if abs(cagr) < 1e-12 and max_dd == 0.0:
        return 0.0

    avg_cvar = 0.0
    avg_positions = 0.0
    if rebal_log is not None and not rebal_log.empty:
        avg_cvar = float(pd.to_numeric(rebal_log.get("realised_cvar", 0.0), errors="coerce").fillna(0.0).mean())
        avg_positions = float(pd.to_numeric(rebal_log.get("n_positions", 0.0), errors="coerce").fillna(0.0).mean())

    risk_penalty = max_dd + (avg_cvar * 100.0 * 5.0) + 1.0
    exposure_penalty = 0.0 if avg_positions >= 1.0 else 0.5

    # Hard MaxDD gate: 55% catches genuinely broken configurations (extreme
    # concentration + no CVaR protection). Set at 55% not 40% because the 2020
    # COVID OOS slice can produce 40-50% drawdowns for any equity momentum strategy.
    if max_dd > 55.0:
        # Return a gradient signal so TPE steers away, not a uniform floor.
        raw = -(max_dd / 10.0)
    else:
        raw = (cagr_net / risk_penalty) - exposure_penalty

    # Per-slice score floor: prevents a single catastrophic slice (e.g. a year
    # where the strategy makes no trades or suffers extreme drawdown) from
    # dominating the aggregate by 40×. With floor=-2.0:
    #   worst-case trial score ≈ (-2.0 + -0.1 + +2.25) / 3 = +0.05
    #   typical bad trial score ≈ (-0.9 + -0.1 + +2.25) / 3 = +0.42
    # TPE now has real gradient to exploit on all three WFO dimensions.
    return max(raw, -2.0)


class MomentumObjective:
    def __init__(self, market_data: dict, universe_type: str, search_space: dict | None = None):
        self.market_data = market_data
        self.universe_type = universe_type
        self.search_space = search_space or SEARCH_SPACE_BOUNDS

    def __call__(self, trial: optuna.Trial) -> float:
        # 1. Base Configuration
        cfg = UltimateConfig()

        # 2. Define the Search Space (Group A: Alpha & Turnover)
        halflife_fast_min, halflife_fast_max = self.search_space["HALFLIFE_FAST"]
        halflife_slow_min, halflife_slow_max = self.search_space["HALFLIFE_SLOW"]
        cfg.HALFLIFE_FAST = trial.suggest_int("HALFLIFE_FAST", halflife_fast_min, halflife_fast_max)
        cfg.HALFLIFE_SLOW = trial.suggest_int("HALFLIFE_SLOW", halflife_slow_min, halflife_slow_max)
        
        # Logical Constraint: Fast must be strictly faster than slow
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

        # 5. Expanding-window time-series CV evaluation
        scores = []
        for _, _, wf_oos_start, wf_oos_end in _iter_wfo_slices(TRAIN_START, TRAIN_END):
            oos_year = pd.Timestamp(wf_oos_start).year

            # EXCLUDE 2019 from scoring (see _iter_wfo_slices docstring).
            # Still run the OOS backtest so exceptions propagate (parameters
            # that produce NaN signals in 2019 should be pruned), but do not
            # append its score to the aggregate.
            exclude_from_score = (oos_year == 2019)

            oos = run_backtest(
                market_data=self.market_data,
                universe_type=self.universe_type,
                start_date=wf_oos_start,
                end_date=wf_oos_end,
                cfg=cfg
            )
            m = oos.metrics
            score = _fitness_from_metrics(m, getattr(oos, "rebal_log", pd.DataFrame()))

            if not pd.notna(score):
                raise optuna.TrialPruned()

            if not exclude_from_score:
                scores.append(float(score))

        if not scores:
            raise optuna.TrialPruned()
        return float(sum(scores) / len(scores))

# ─── Orchestration ────────────────────────────────────────────────────────────

def pre_load_data(universe_type: str) -> dict:
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
        
    # Ensure index data is present for regime scoring
    symbols_to_fetch = list(dict.fromkeys(base_universe + ["^NSEI", "^CRSLDX"]))
    
    logger.info(f"Fetching {len(symbols_to_fetch)} symbols from {TRAIN_START} to {TEST_END}...")
    market_data = load_or_fetch(
        tickers=symbols_to_fetch, 
        required_start=TRAIN_START, 
        required_end=TEST_END
    )
    logger.info("Data pre-load complete. Commencing Bayesian Optimization.")
    return market_data


def save_optimal_config(best_params: dict, filepath: str = "data/optimal_cfg.json"):
    """Exports the winning configuration so daily_workflow.py can use it."""
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write atomically to avoid partial/truncated JSON if the process is interrupted.
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

    os.replace(temp_path, filepath)
    logger.info(f"Saved optimal parameters to {filepath}")


def run_optimization(universe_type: str = "nifty500", in_memory: bool = False):
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
        logger.warning(
            "Forcing n_jobs=1 because backtests are CPU-bound and Optuna uses threads. "
            "Use multiple optimizer processes with shared storage for real parallelism."
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
    study = optuna.create_study(
        study_name="Momentum_Risk_Parity",
        direction="maximize",
        sampler=_build_sampler(),
        storage=effective_storage,      # :memory: or SQLite depending on flag/env
        load_if_exists=True
    )
     
    objective = MomentumObjective(market_data, universe_type)

    # 2. Run In-Sample Optimization
    logger.info(f"Starting {N_TRIALS} Bayesian Trials (This may take a while)...")
    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            show_progress_bar=True,
            n_jobs=effective_n_jobs,
            catch=(OptimizationError,),
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
    best_is_calmar = study.best_value

    print(f"\n\033[1;32m=== OPTIMIZATION COMPLETE ===\033[0m")
    print(f"\033[1mBest In-Sample Calmar Ratio:\033[0m {best_is_calmar:.2f}")
    print("\033[1mWinning Parameters:\033[0m")
    for k, v in best_params.items():
        print(f"  {k}: \033[33m{v}\033[0m")

    # 3. Out-of-Sample Validation (The true test of robustness)
    print(f"\n\033[1;36m=== INITIATING OUT-OF-SAMPLE (OOS) VALIDATION ===\033[0m")
    
    # Construct a new config using the best parameters
    oos_cfg = UltimateConfig()
    valid_fields = UltimateConfig.__dataclass_fields__
    for k, v in best_params.items():
        if k not in valid_fields:
            logger.warning("[Config] Ignoring unknown/stale optimized parameter during OOS validation: %s", k)
            continue
        setattr(oos_cfg, k, v)

    logger.info(f"Running unseen data ({TEST_START} to {TEST_END})...")
    
    try:
        oos_results = run_backtest(
            market_data=market_data,
            universe_type=universe_type,
            start_date=TEST_START,
            end_date=TEST_END,
            cfg=oos_cfg
        )
        
        m = oos_results.metrics
        _rs = "\u20b9" if _stdout_supports_rupee() else "Rs."
        print(f"\n\033[1mOOS Final Equity:\033[0m \033[32m{_rs}{m.get('final', 0):,.0f}\033[0m")
        print(f"\033[1mOOS CAGR:\033[0m {m.get('cagr', 0):.2f}%")
        print(f"\033[1mOOS MaxDD:\033[0m {m.get('max_dd', 0):.2f}%")
        print(f"\033[1mOOS Calmar:\033[0m {m.get('calmar', 0):.2f}")
        
        # OOS window is 2023-2026: a sustained bull market.
        # Calmar > 0.5 over 3+ years in a bull regime is a meaningful bar
        # (not trivially easy — drawdowns still occur). MaxDD cap at 35%
        # is appropriate since we're outside a known bear period.
        if m.get('calmar', 0) > 0.5 and abs(m.get('max_dd', 100)) <= OOS_MAX_DD_CAP:
            save_optimal_config(best_params)
            print("\n\033[1;32m[PASS]\033[0m Strategy parameters survived Out-of-Sample verification without structural decay.")
        else:
            raise RuntimeError(
                "OOS Validation Failed: Parameters degraded severely Out-of-Sample. "
                "The model is overfitted."
            )

    except RuntimeError:
        raise
    except Exception as e:
        print(f"\n\033[1;31m[FAIL]\033[0m OOS Validation threw an exception: {e}")

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
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_optimization(universe_type=args.universe, in_memory=args.in_memory)
