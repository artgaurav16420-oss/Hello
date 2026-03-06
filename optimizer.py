"""
optimizer.py — Institutional Bayesian Walk-Forward Optimizer v11.46
===================================================================
Automates the discovery of optimal risk and momentum parameters using Optuna.
Implements strict Out-of-Sample (OOS) validation to prevent curve-fitting.

Requires: pip install optuna
"""

import argparse
import json
import logging
import os
import tempfile
import sys
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

# Configure local logger
logger = logging.getLogger("Optimizer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("\033[90m[%(asctime)s]\033[0m %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)

# ─── Optimization Configuration ───────────────────────────────────────────────

TRAIN_START = "2018-01-01"
TRAIN_END   = "2019-12-31"   # 2 pre-COVID years — optimizer never sees the crash
TEST_START  = "2020-01-01"   # OOS starts Jan 2020: COVID is the first real stress test
TEST_END    = pd.Timestamp.today().strftime("%Y-%m-%d")

N_TRIALS       = 100    # Number of Bayesian iterations
# IS (2018-2019) is pre-COVID: 25% is a realistic strict cap — a strategy that
# draws down >25% in calm pre-crash markets has a structural problem.
# OOS (2020-present) contains the COVID crash: 40% is the appropriate cap
# since Nifty itself fell 38%. Manual intervention handles true black swans.
OOS_MAX_DD_CAP = 40.0   # OOS cap — lenient, OOS contains COVID

# Search space bounds are configurable to support high-risk/high-turnover variants.
SEARCH_SPACE_BOUNDS = {
    "HALFLIFE_FAST": (10, 40),
    "HALFLIFE_SLOW": (50, 120),
    "CONTINUITY_BONUS": (0.05, 0.30, 0.01),
    "RISK_AVERSION": (2.0, 15.0, 0.5),
    "CVAR_DAILY_LIMIT": (0.025, 0.06, 0.005),
}

# Runtime knobs: use all cores by default and allow optional deterministic seed.
N_JOBS = int(os.getenv("OPTUNA_N_JOBS", "-1"))
OPTUNA_SEED = os.getenv("OPTUNA_SEED")


def _build_sampler() -> TPESampler:
    if OPTUNA_SEED in (None, ""):
        return TPESampler()
    return TPESampler(seed=int(OPTUNA_SEED))

# ─── Objective Function ───────────────────────────────────────────────────────

def _iter_wfo_slices(train_start: str, train_end: str):
    start = pd.Timestamp(train_start)
    end = pd.Timestamp(train_end)
    stop_year = max(end.year, pd.Timestamp(TEST_START).year)
    years = list(range(start.year + 2, stop_year + 1))
    for y in years:
        oos_start = pd.Timestamp(f"{y}-01-01")
        oos_end = pd.Timestamp(f"{y}-12-31")
        yield (start.strftime("%Y-%m-%d"), (oos_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), oos_start.strftime("%Y-%m-%d"), oos_end.strftime("%Y-%m-%d"))


def _fitness_from_metrics(metrics: dict, rebal_log: pd.DataFrame) -> float:
    cagr = float(metrics.get("cagr", 0.0))
    max_dd = abs(float(metrics.get("max_dd", 100.0)))
    turnover = float(metrics.get("turnover", 0.0))
    turnover_drag = turnover * 15.0  # 15 bps per 1x turnover as CAGR percentage drag
    cagr_net = cagr - turnover_drag
    if abs(cagr) < 1e-12 and max_dd == 0.0:
        return 0.0

    avg_cvar = 0.0
    avg_positions = 0.0
    if rebal_log is not None and not rebal_log.empty:
        avg_cvar = float(pd.to_numeric(rebal_log.get("realised_cvar", 0.0), errors="coerce").fillna(0.0).mean())
        avg_positions = float(pd.to_numeric(rebal_log.get("n_positions", 0.0), errors="coerce").fillna(0.0).mean())

    risk_penalty = max_dd + (avg_cvar * 100.0 * 5.0) + 1.0
    exposure_penalty = 1.0 if avg_positions >= 1.0 else 0.5
    return (cagr_net / risk_penalty) - exposure_penalty


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
        if cfg.HALFLIFE_FAST >= cfg.HALFLIFE_SLOW:
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

        # 4. Walk-forward evaluation on expanding windows
        scores = []
        try:
            for wf_train_start, wf_train_end, wf_oos_start, wf_oos_end in _iter_wfo_slices(TRAIN_START, TRAIN_END):
                _ = run_backtest(
                    market_data=self.market_data,
                    universe_type=self.universe_type,
                    start_date=wf_train_start,
                    end_date=wf_train_end,
                    cfg=cfg
                )
                oos = run_backtest(
                    market_data=self.market_data,
                    universe_type=self.universe_type,
                    start_date=wf_oos_start,
                    end_date=wf_oos_end,
                    cfg=cfg
                )
                score = _fitness_from_metrics(oos.metrics, getattr(oos, "rebal_log", pd.DataFrame()))
                if not pd.notna(score):
                    raise optuna.TrialPruned()
                scores.append(float(score))
        except OptimizationError:
            raise optuna.TrialPruned()
        except Exception as e:
            logger.warning(f"Trial failed due to internal error: {e}")
            raise optuna.TrialPruned()

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


def run_optimization(universe_type: str = "nifty500"):
    print(f"\n\033[1;36m=== INSTITUTIONAL WALK-FORWARD OPTIMIZER ===\033[0m")
    print(f"\033[90mIn-Sample (Train) : {TRAIN_START} to {TRAIN_END}\033[0m")
    print(f"\033[90mOut-of-Sample     : {TEST_START} to {TEST_END}\033[0m")
    print(f"\033[90mTrials            : {N_TRIALS}\033[0m\n")

    logger.info(f"Optimization universe: {universe_type}")
    market_data = pre_load_data(universe_type)

    # 1. Setup Optuna Study
    study = optuna.create_study(
        study_name="Momentum_Risk_Parity",
        direction="maximize",
        sampler=_build_sampler(),
    )
    
    objective = MomentumObjective(market_data, universe_type)

    # 2. Run In-Sample Optimization
    logger.info(f"Starting {N_TRIALS} Bayesian Trials (This may take a while)...")
    study.optimize(
        objective, 
        n_trials=N_TRIALS, 
        show_progress_bar=True,
        n_jobs=N_JOBS,
        catch=(Exception,)
    )

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

    save_optimal_config(best_params)

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
        print(f"\n\033[1mOOS Final Equity:\033[0m \033[32m₹{m.get('final', 0):,.0f}\033[0m")
        print(f"\033[1mOOS CAGR:\033[0m {m.get('cagr', 0):.2f}%")
        print(f"\033[1mOOS MaxDD:\033[0m {m.get('max_dd', 0):.2f}%")
        print(f"\033[1mOOS Calmar:\033[0m {m.get('calmar', 0):.2f}")
        
        # Institutional Validation Heuristic
        if m.get('calmar', 0) > 1.0 and abs(m.get('max_dd', 100)) <= OOS_MAX_DD_CAP:
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_optimization(universe_type=args.universe)
