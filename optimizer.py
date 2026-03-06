"""
optimizer.py — Institutional Bayesian Walk-Forward Optimizer v11.46
===================================================================
Automates the discovery of optimal risk and momentum parameters using Optuna.
Implements strict Out-of-Sample (OOS) validation to prevent curve-fitting.

Requires: pip install optuna
"""

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
MAX_DD_CAP     = 25.0   # In-sample hard cap — strict, IS is calm regime
OOS_MAX_DD_CAP = 40.0   # OOS cap — lenient, OOS contains COVID

# ─── Objective Function ───────────────────────────────────────────────────────

class MomentumObjective:
    def __init__(self, market_data: dict, universe_type: str):
        self.market_data = market_data
        self.universe_type = universe_type

    def __call__(self, trial: optuna.Trial) -> float:
        # 1. Base Configuration
        cfg = UltimateConfig()

        # 2. Define the Search Space (Group A: Alpha & Turnover)
        cfg.HALFLIFE_FAST = trial.suggest_int("HALFLIFE_FAST", 10, 40)
        cfg.HALFLIFE_SLOW = trial.suggest_int("HALFLIFE_SLOW", 50, 120)
        
        # Logical Constraint: Fast must be strictly faster than slow
        if cfg.HALFLIFE_FAST >= cfg.HALFLIFE_SLOW:
            raise optuna.TrialPruned()
            
        cfg.CONTINUITY_BONUS = trial.suggest_float("CONTINUITY_BONUS", 0.05, 0.30, step=0.01)

        # 3. Define the Search Space (Group B: Risk Matrix)
        cfg.RISK_AVERSION    = trial.suggest_float("RISK_AVERSION", 2.0, 15.0, step=0.5)
        cfg.CVAR_DAILY_LIMIT = trial.suggest_float("CVAR_DAILY_LIMIT", 0.025, 0.06, step=0.005)

        # 4. Execute the In-Sample Backtest
        try:
            results = run_backtest(
                market_data=self.market_data,
                universe_type=self.universe_type,
                start_date=TRAIN_START,
                end_date=TRAIN_END,
                cfg=cfg
            )
        except OptimizationError:
            # If the solver structurally fails on these params, prune the trial
            raise optuna.TrialPruned()
        except Exception as e:
            logger.debug(f"Trial failed unexpectedly: {e}")
            raise optuna.TrialPruned()

        metrics = results.metrics
        cagr = metrics.get("cagr", 0.0)
        max_dd = abs(metrics.get("max_dd", 100.0))

        # 6. Hard prune if IS drawdown exceeds the pre-COVID cap.
        # IS window (2018-2019) has no crash, so >25% DD is a real structural fail.
        if max_dd > MAX_DD_CAP:
            raise optuna.TrialPruned()

        # 5. Objective Calculation (Calmar Ratio)
        if max_dd == 0:
            return 0.0

        calmar = cagr / max_dd

        return calmar

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


def run_optimization():
    print(f"\n\033[1;36m=== INSTITUTIONAL WALK-FORWARD OPTIMIZER ===\033[0m")
    print(f"\033[90mIn-Sample (Train) : {TRAIN_START} to {TRAIN_END}\033[0m")
    print(f"\033[90mOut-of-Sample     : {TEST_START} to {TEST_END}\033[0m")
    print(f"\033[90mTrials            : {N_TRIALS}\033[0m\n")

    universe_type = "nifty500"
    market_data = pre_load_data(universe_type)

    # 1. Setup Optuna Study
    study = optuna.create_study(
        study_name="Momentum_Risk_Parity",
        direction="maximize",
        sampler=TPESampler(seed=42) # Seeded for reproducibility
    )
    
    objective = MomentumObjective(market_data, universe_type)

    # 2. Run In-Sample Optimization
    logger.info(f"Starting {N_TRIALS} Bayesian Trials (This may take a while)...")
    study.optimize(
        objective, 
        n_trials=N_TRIALS, 
        show_progress_bar=True,
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
    for k, v in best_params.items():
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
            print("\n\033[1;31m[FAIL]\033[0m Parameters degraded severely Out-of-Sample. The model is overfitted to the training data.")

    except Exception as e:
        print(f"\n\033[1;31m[FAIL]\033[0m OOS Validation threw an exception: {e}")

if __name__ == "__main__":
    run_optimization()
