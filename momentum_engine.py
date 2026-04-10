"""
momentum_engine.py — Institutional Risk Engine v11.48
=====================================================
CVaR-constrained Mean-Variance Optimizer with full Transaction Cost formulation.

BUG FIXES (murder board):
- FIX-MB-SLIP: Double-slippage on buys eliminated.
- FIX-MB-VOL: compute_book_cvar writes last_known_volatility only under the
  canonical key present in state.shares.
- FIX-MB-OSQP: OSQP solver cache is invalidated on any solve() exception.
- FIX-MB-SECTOR: max_possible_weight correctly caps known-sector groups.
- FIX-MB-SETTER: SLIPPAGE_BPS setter chains original exception.
- FIX-MB2-GHOSTPV: compute_book_cvar reindexes with fill_value=np.nan.
- FIX-MB-ME-02: compute_one_way_slip_rate now accepts an optional trade_notional
  parameter and uses it as the impact numerator when provided. The portfolio_value
  parameter is kept for backward compatibility but impact is computed against
  trade_notional when available, preventing overestimation for small trades and
  underestimation for large rebalances.
- FIX-MB-ME-03: PortfolioState.update_exposure re-evaluates the breach condition
  after cooldown expiry so that a sustained CVaR breach spanning the entire
  cooldown period re-arms the override on the step cooldown reaches zero, rather
  than allowing one unprotected rebalance.
- FIX-MB-ME-01: execute_rebalance two-phase PV accounting is documented with
  inline comments explaining why force-close proceeds flow through pv_exec
  rather than actual_notional.
- FIX-MB-C-01: Deleted duplicate decay-CVaR check block that appeared twice
  in execute_rebalance. The first (incomplete) copy lacked the inner logic and
  was a merge artifact; only the second (complete) block is retained.
- FIX-MB-C-03: Removed force-close pv_exec double-add from the decay
  liquidation branch. Phase 2 of execute_rebalance unconditionally adds
  force-close proceeds; adding them again inside the liquidation loop inflated
  state.cash by up to 2x the force-closed notional.
- FIX-MB-M-05: override_cooldown re-arm now uses max(current, 4) so any
  longer externally-set cooldown is preserved rather than silently shortened.
- PROD-FIX-EXEC-1: execute_rebalance buy path now sizes shares at raw price;
  slippage is charged as a separate cash deduction in the accounting loop.
  Previously effective_buy_price (raw*(1+slip)) was used for share-count
  sizing, causing a clean 2% intended weight move to be reduced to 1.5% in
  shares and then incorrectly blocked by the drift gate.
- PROD-FIX-EXEC-2: drift gate now compares the intended weight change
  (|target_weight - current_weight|) rather than the slippage-adjusted share
  delta, so the gate correctly reflects manager intent independent of the
  execution slippage model.
- FIX-GHOST-SEED: compute_book_cvar ghost synthesis no longer uses a single
  SeedSequence(row_seeds).spawn(...) fan-out, which mixed row seeds into a
  derived entropy tree and broke strict (symbol,date)->sample reproducibility.
  Each row now builds its own Generator with default_rng(int(row_seed)).
"""

from __future__ import annotations

# OSQP must be imported BEFORE numpy/pandas on some Windows/3.13/NumPy2.x builds
# to avoid silent process termination due to ABI initialisation ordering.
import osqp

import hashlib
import logging
import threading
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.covariance import LedoitWolf

from shared_utils import normalize_ns_ticker

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
logger = logging.getLogger(__name__)
EPSILON = 1e-6
DEFAULT_MAX_ABSENT_PERIODS = 12
DEFAULT_MAX_DECAY_ROUNDS = 3
DEFAULT_GHOST_VOL_FALLBACK = 0.04

# ─── Ghost synthesis determinism cache ───────────────────────────────────────
# Benign race note: concurrent first-write races can occur when two threads
# compute the same symbol seed at once, but the value is deterministic from
# SHA-256(sym), so both writers store the same integer.
_GHOST_SEED_CACHE: Dict[str, int] = {}


def _ghost_seed_for(sym: str) -> int:
    """Return a deterministic integer seed for a given symbol, cached per process."""
    if sym not in _GHOST_SEED_CACHE:
        # FIX-MB-ME-05: 63-bit seed (vs original 31-bit) reduces cross-symbol
        # collision probability; capped at 2**63 to stay within np.int64 range.
        _GHOST_SEED_CACHE[sym] = int(hashlib.sha256(sym.encode()).hexdigest()[:16], 16) % (2 ** 63)
    return _GHOST_SEED_CACHE[sym]


# ─── Symbol helpers ───────────────────────────────────────────────────────────

def to_ns(sym: str) -> str:
    """
    Standardize a ticker to the National Stock Exchange of India (NSE) format.
    Alias for shared_utils.normalize_ns_ticker.

    Args:
        sym (str): The raw ticker string.

    Returns:
        str: Symbol suffixed with '.NS'.
    """
    return normalize_ns_ticker(sym)


def to_bare(sym: str) -> str:
    """
    Remove the exchange suffix from a standardized ticker.

    Args:
        sym (str): Standardized symbol (e.g., 'RELIANCE.NS').

    Returns:
        str: Bare symbol name (e.g., 'RELIANCE').
    """
    return sym[:-3] if sym.endswith(".NS") else sym


def absent_symbol_effective_price(last_known_price: float, absent_periods: int, max_absent_periods: int) -> float:
    """
    Calculate the effective price for a symbol that has been absent from market data.
    Applies a linear haircut to the last known price based on how long it has been missing.
    If absent_periods >= max_absent_periods, the effective price becomes 0.0.

    Args:
        last_known_price (float): The final valid price recorded for the symbol.
        absent_periods (int): Number of consecutive periods the symbol has been missing.
        max_absent_periods (int): Maximum allowed absence before the price is zeroed.

    Returns:
        float: The haircut-adjusted price.
    """
    px = float(last_known_price)
    if not np.isfinite(px) or px <= 0:
        return 0.0
    n_absent = max(0, int(absent_periods))
    max_absent = max(1, int(max_absent_periods))
    haircut = max(0.0, 1.0 - (n_absent / max_absent))
    return px * haircut


# ─── Enumerations & exceptions ────────────────────────────────────────────────

class OptimizationErrorType(Enum):
    """
    Categories of failures during the portfolio optimization process.
    """
    NUMERICAL  = auto()  # Solver failed due to numerical instability or ill-conditioned matrices.
    INFEASIBLE = auto()  # No solution exists that satisfies all provided constraints.
    DATA       = auto()  # Missing or invalid input data (covariances, returns, etc.).


class OptimizationError(Exception):
    """OptimizationError type used by the backtesting system."""
    def __init__(
        self,
        message:    str,
        error_type: OptimizationErrorType = OptimizationErrorType.NUMERICAL,
    ):
        """
        Initialize an optimization error.

        Args:
            message (str): Descriptive error message.
            error_type (OptimizationErrorType): The failure category.
        """
        super().__init__(message)
        self.error_type = error_type


# ─── Value objects ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """
    Detailed record of an executed transaction within a rebalance.

    Attributes:
        symbol (str): NSE ticker symbol (without exchange suffix).
        date (pd.Timestamp): Execution date/time.
        delta_shares (int): Absolute change in share count.
        exec_price (float): Final matched price after haircut/slippage.
        slip_cost (float): Estimated notional impact of the trade.
        direction (str): 'BUY' or 'SELL'.
    """
    symbol:       str
    date:         pd.Timestamp
    delta_shares: int
    exec_price:   float
    slip_cost:    float
    direction:    str          # "BUY" | "SELL"


@dataclass
class SolverDiagnostics:
    """
    Metadata and convergence statistics from the OSQP solver execution.
    Used for monitoring solver health and identifying binding constraints.

    Attributes:
        status (str): Solver exit status (e.g., 'solved', 'infeasible').
        gamma_intent (float): The targeted portfolio exposure.
        actual_weight (float): The sum of optimized weights.
        l_gamma/u_gamma: Bounds on total exposure.
        cvar_value (float): Realized CVaR of the optimized portfolio.
        slack_value (float): Magnitude of slack variables used to satisfy CVaR.
        adv_binding_count (int): Number of symbols hitting the ADV liquidity cap.
        ridge_applied (float): Amount of Tikhonov regularization added to the covariance matrix.
    """
    status:            str
    gamma_intent:      float
    actual_weight:     float
    l_gamma:           float
    u_gamma:           float
    cvar_value:        float
    slack_value:       float
    sum_adv_limit:     float
    adv_binding_count: int
    ridge_applied:     float
    cond_number:       float
    t_cvar:            int

    @property
    def budget_utilisation(self) -> float:
        """Percentage of the upper exposure bound utilized by the solver."""
        return self.actual_weight / self.u_gamma if self.u_gamma > 0 else 0.0


# ─── Matrix Building Helper ───────────────────────────────────────────────────

class _ConstraintBuilder:
    """
    Utility to assemble multiple linear constraints into the sparse format required by OSQP.
    Stack matrices vertically and flattens bound vectors.
    """
    def __init__(self, n_vars: int):
        """Initialize the builder for a fixed number of optimization variables."""
        self.n_vars = n_vars
        self.A_parts: list = []
        self.l_parts: list = []
        self.u_parts: list = []

    def add_constraint(self, A_matrix, lower_bound, upper_bound):
        """Append a new constraint block (A, l, u) to the system."""
        self.A_parts.append(A_matrix)
        self.l_parts.append(lower_bound)
        self.u_parts.append(upper_bound)

    def build(self) -> Tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
        """Finalize and return the vertically stacked constraint system."""
        A = sp.vstack(self.A_parts, format="csc")
        lower = np.concatenate([np.atleast_1d(b) for b in self.l_parts]).astype(float, copy=False)
        upper = np.concatenate([np.atleast_1d(b) for b in self.u_parts]).astype(float, copy=False)
        return A, lower, upper


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class UltimateConfig:
    """
    Global configuration schema for the Institutional Risk Engine.
    Controls sizing constraints, risk thresholds, and execution modeling.
    Kept as a flat dataclass to ensure native compatibility with YAML/JSON 
    serialization and Optuna hyperparameter tracking.
    """

    # --- Core & Sizing Constraints ---
    INITIAL_CAPITAL:          float = 1_000_000.0  # Starting cash balance for the portfolio.
    MAX_POSITIONS:            int   = 10           # Maximum active line items allowed.
    MAX_PORTFOLIO_RISK_PCT:   float = 0.20         # Maximum aggregate CVaR allowed.
    MAX_SINGLE_NAME_WEIGHT:   float = 0.25         # Hard cap on individual stock weights.
    MAX_SECTOR_WEIGHT:        float = 0.35         # Limit any single sector to 35% of gross exposure so OSQP rows (0 ≤ Σw_sector ≤ 0.35) can bind.

    # --- Liquidity & Execution ---
    MAX_ADV_PCT:              float = 0.05         # Maximum participation rate of Average Daily Volume.
    MIN_ADV_CRORES:           float = 100.0        # Minimum ADV limit for universe inclusion.
    IMPACT_COEFF:             float = 5e-4         # Coefficient scaling quadratic market impact.
    ROUND_TRIP_SLIPPAGE_BPS:  float = 20.0         # Estimated round-trip friction in basis points.
    REBALANCE_FREQ:           str   = "W-FRI"      # Calendar frequency for scheduled rebalances.

    # --- Risk & Drawdown Management (CVaR) ---
    CVAR_DAILY_LIMIT:            float = 0.055     # Daily expected shortfall ceiling.
    CVAR_ALPHA:                  float = 0.95      # Confidence level for CVaR computation (e.g. 95%).
    CVAR_LOOKBACK:               int   = 90        # Lookback window for covariance and risk models.
    ADV_LOOKBACK:                int   = 90        # Lookback window for average daily volume.
    CVAR_SENTINEL_MULTIPLIER:    float = 2.5       # Multiplier identifying anomalous volatility spikes.
    CVAR_MIN_HISTORY:            int   = 20        # Minimum trading days required for risk assessment.
    CVAR_HARD_BREACH_MULTIPLIER: float = 1.5       # Triggers strict deleveraging when current CVaR exceeds limit by this multiplier.

    # --- Exposure Management ---
    DELEVERAGING_LIMIT:          float = 0.20      # Max forced cash raise fraction per interval.
    MIN_EXPOSURE_FLOOR:          float = 0.05      # Minimum cash usage unless fully deleveraged.
    CAPITAL_ELASTICITY:          float = 0.15      # Sensitivity of target weight to current cash.
    DRIFT_TOLERANCE:             float = 0.02      # Tolerable weight drift before forcing trades.

    # --- Signal Generation & Optimization ---
    SIGNAL_ANNUAL_FACTOR:     int   = 252          # Trading days in a year for signal scaling.
    HISTORY_GATE:             int   = 90           # Minimum history to permit strategy inclusion.
    HALFLIFE_FAST:            int   = 21           # Fast EWMA momentum halflife.
    HALFLIFE_SLOW:            int   = 63           # Slow EWMA momentum halflife.
    SIGNAL_LAG_DAYS:          int   = 21           # Lag applied between fast/slow signals.
    RISK_AVERSION:            float = 5.0          # Target lambda parameter in objective function.
    SLACK_PENALTY:            float = 1000.0       # Penalty parameter for OSQP soft constraints.
    DIMENSIONALITY_MULTIPLIER:int   = 3            # Factor determining history requirements vs positions.

    # --- Continuity & Gate Logic ---
    Z_SCORE_CLIP:             float = 3.0          # Caps normalized signals to prevent extreme outliers.
    CONTINUITY_BONUS:         float = 0.15         # Reward factor for maintaining existing positions.
    CONTINUITY_DISPERSION_FLOOR: float = 0.1       # Minimum dispersion threshold to apply continuity.
    CONTINUITY_MAX_SCALAR:    float = 0.20         # Maximum applicable continuity bonus score.
    CONTINUITY_MAX_HOLD_WEIGHT: float = 0.10       # Maximum weight limit for applying continuity.
    CONTINUITY_ACTIVITY_WINDOW: int = 5            # Days checked for recent trading activity.
    CONTINUITY_STALE_SESSIONS: int = 10            # Inactivity periods triggering stale status.
    CONTINUITY_FLAT_RET_EPS: float = 1e-12         # Epsilon rounding to identify dead zeroes.
    CONTINUITY_MIN_ADV_NOTIONAL: float = 0.0       # Minimal required ADV for continuity to apply.
    KNIFE_WINDOW:             int   = 20           # Window for detecting falling knife anomalies.
    KNIFE_THRESHOLD:          float = -0.15        # Negative return threshold for falling knife.

    # --- Data Edge Cases & Decay ---
    MAX_ABSENT_PERIODS:       int   = 12           # Consecutive missing periods before force-closing a position.
    MAX_DECAY_ROUNDS:         int   = 3            # Allowable decay attempts before forced liquidation.
    DECAY_FACTOR:             float = 0.5          # multiplier applied to weight during partial liquidation.
    GHOST_VOL_LOOKBACK:       int   = 20           # Lookback for ghost asset pseudo-volatility.
    GHOST_RET_DRIFT:          float = -0.02        # Artificial drift added to untradable ghost stock.
    GHOST_VOL_FALLBACK:       float = 0.04         # Baseline volatility for missing data assets.

    # --- Network & Data Fetching ---
    YF_BATCH_TIMEOUT:         float = 120.0        # Global timeout for YF ticker batch requests.
    YF_CHUNK_TIMEOUT:         float = 90.0         # Block timeout for specific YF thread chunks.
    YF_ADV_TIMEOUT:           float = 60.0         # Timeout for concurrent ADV calculations.
    SECTOR_FETCH_TIMEOUT:     float = 8.0          # Max waiting time for sector definitions.

    # --- Regime Modeling (Market Environment) ---
    REGIME_VOL_FLOOR:         float = 0.18         # Default floor assumed for market volatility.
    REGIME_VOL_MULTIPLIER:    float = 1.5          # Shock multiplier identifying high-stress regimes.
    REGIME_SIGMOID_STEEPNESS: float = 10.0         # Steepness of regime transfer function.
    REGIME_SMA_FAST_WINDOW:   int   = 50           # Short term moving average.
    REGIME_SMA_WINDOW:        int   = 200          # Long term moving average for market trend.
    REGIME_VOL_EWMA_SPAN:     int   = 20           # Time span for fast volatility decay.
    REGIME_LT_VOL_EWMA_SPAN:  int   = 1260         # Rolling baseline parameter for deep historical volatility.

    # --- Institutional Safety & Corporate Actions ---
    SIMULATE_HALTS:           bool  = False        # If True, randomly simulate trading halts.
    DIVIDEND_SWEEP:           bool  = True         # Auto-reinvest detected dividends into cash ledger.
    SPLIT_TOLERANCE:          float = 0.005        # Acceptable error margin for auto-reconstructing splits.
    AUTO_ADJUST_PRICES:       bool  = True         # Enables implicit backward price adjustment.
    EQUITY_HIST_CAP:          int   = 500          # Hard cap on number of periods maintained in historical equity curves to prevent O(N) CVaR scale-outs.

    SLIPPAGE_BPS:             float = field(default=20.0, init=False)  # Backward-compatible alias for round-trip friction in basis points.

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep slippage aliases synchronized and validated."""
        if name in {"SLIPPAGE_BPS", "ROUND_TRIP_SLIPPAGE_BPS"}:
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not np.isfinite(parsed) or parsed < 0:
                raise ValueError(
                    f"{name} must be a finite non-negative number, got {parsed}"
                )
            object.__setattr__(self, "ROUND_TRIP_SLIPPAGE_BPS", parsed)
            object.__setattr__(self, "SLIPPAGE_BPS", parsed)
            return
        object.__setattr__(self, name, value)


@dataclass
class PortfolioState:
    """
    Representation of the current live state of the trading portfolio.
    Tracks cash, shares, trade history, and risk control parameters.
    """
    RISK_CONTROL_FIELDS: ClassVar[Tuple[str, ...]] = (
        "override_active",
        "override_cooldown",
        "consecutive_failures",
        "decay_rounds",
    )
    weights:              Dict[str, float] = field(default_factory=dict)
    shares:               Dict[str, int]   = field(default_factory=dict)
    entry_prices:         Dict[str, float] = field(default_factory=dict)
    equity_hist:          List[float]      = field(default_factory=list)
    universe:             List[str]        = field(default_factory=list)
    cash:                 float            = 1_000_000.0
    exposure_multiplier:  float            = 1.0
    override_active:      bool             = False
    override_cooldown:    int              = 0
    consecutive_failures: int              = 0
    # Presence-aware optional fields: None means "missing from persisted state",
    # allowing callers to apply UltimateConfig defaults only when absent.
    equity_hist_cap:      Optional[int]    = None
    max_absent_periods:   Optional[int]    = None
    absent_periods:       Dict[str, int]   = field(default_factory=dict)
    last_known_prices:    Dict[str, float] = field(default_factory=dict)
    last_known_volatility:Dict[str, float] = field(default_factory=dict)
    vol_hist:             Dict[str, deque] = field(default_factory=dict)
    decay_rounds:         int              = 0
    dividend_ledger:      Dict[str, str]   = field(default_factory=dict)
    last_rebalance_date:  str              = ""

    def __post_init__(self) -> None:
        """Initialize transient private state and default caps after dataclass instantiation."""
        self._initial_cash = float(self.cash)
        self._initial_exposure_multiplier = float(self.exposure_multiplier)
        if self.equity_hist_cap is None:
            self._equity_hist_cap = max(500, int(UltimateConfig().CVAR_LOOKBACK * 2))
        else:
            self._equity_hist_cap = int(self.equity_hist_cap)

    def update_exposure(
        self,
        regime_score:   float,
        realized_cvar:  float,
        cfg:            UltimateConfig,
        gross_exposure: float = 1.0,
    ) -> None:
        """
        Dynamically adjust the portfolio's exposure multiplier based on regime and risk.
        Implements a sigmoid regime response and an immediate override (halving) if CVaR limits are breached.

        Args:
            regime_score (float): Probability of a favorable market regime [0, 1].
            realized_cvar (float): The current portfolio's realized Conditional Value-at-Risk.
            cfg (UltimateConfig): Configuration for risk limits and deleveraging speeds.
            gross_exposure (float): Current total proportional exposure (sum of |weights|).
        """
        target   = 1.0 / (1.0 + np.exp(-cfg.REGIME_SIGMOID_STEEPNESS * (regime_score - 0.5)))
        new_mult = self.exposure_multiplier + float(
            np.clip(target - self.exposure_multiplier, -cfg.DELEVERAGING_LIMIT, cfg.DELEVERAGING_LIMIT)
        )

        normalised_cvar = realized_cvar / max(float(gross_exposure), 0.05)
        # FIX: Align breach detection with the daily ceiling used by the risk engine
        risk_threshold = max(cfg.CVAR_DAILY_LIMIT, cfg.MAX_PORTFOLIO_RISK_PCT)
        if cfg.CVAR_DAILY_LIMIT > 0:
             risk_threshold = cfg.CVAR_DAILY_LIMIT
        breach = normalised_cvar > risk_threshold

        if gross_exposure < 0.01:
            self.exposure_multiplier = float(
                np.clip(new_mult, cfg.MIN_EXPOSURE_FLOOR, 1.0)
            )
            # FIX: Only decrement cooldown if there is no ongoing risk breach
            if self.override_cooldown > 0 and not breach:
                self.override_cooldown -= 1
            if self.override_cooldown == 0 and self.override_active:
                self.override_active = False
            return

        # FIX-MB-ME-03: Decrement cooldown, but evaluate the breach condition
        # AFTER clearing override_active so that a sustained breach spanning the
        # entire cooldown period re-arms the override immediately on the step
        # cooldown reaches zero, rather than allowing one unprotected rebalance.
        # FIX-MB-ME-06: save original cooldown before decrement so the re-arm
        # max() below can correctly preserve a longer externally-set cooldown.
        # Previously override_cooldown was already 0 at the max() call, making
        # max(0, 4) always resolve to 4 and defeating cooldown preservation.
        original_cooldown = self.override_cooldown
        if self.override_cooldown > 0:
            self.override_cooldown -= 1

        # Clear override once cooldown has fully expired
        if self.override_cooldown == 0 and self.override_active:
            self.override_active = False

        # Re-check breach after override is cleared.  If cooldown just expired
        # AND we are still in breach, re-arm immediately (no free step).
        if breach and not self.override_active and self.override_cooldown == 0:
            # FIX-NEW-ME-02: when already at MIN_EXPOSURE_FLOOR, halving again
            # produces a sub-floor value that the np.clip below corrects, but the
            # intermediate assignment is confusing.  Clamp override_mult so it
            # never falls below the floor before the clip, keeping intent clear.
            halved = self.exposure_multiplier * 0.5
            override_mult            = max(cfg.MIN_EXPOSURE_FLOOR, halved)
            self.exposure_multiplier = min(new_mult, override_mult)
            self.override_active     = True
            self.override_cooldown   = max(original_cooldown, 4)
        else:
            self.exposure_multiplier = new_mult

        self.exposure_multiplier = float(
            np.clip(self.exposure_multiplier, cfg.MIN_EXPOSURE_FLOOR, 1.0)
        )

    def realised_cvar(self, min_obs: int = 30) -> float:
        """Return realised portfolio CVaR from recent equity history.

        CVaR is computed over a rolling window capped at `equity_hist_cap` bars.
        History older than this cap is discarded.
        """
        n = len(self.equity_hist)
        if n < min_obs:
            if n > 1:
                logger.debug(
                    "CVaR computed on only %d observations (min %d for stability); "
                    "returning 0.0 during warm-up.", n, min_obs,
                )
            return 0.0

        rets  = np.log1p(
            pd.Series(self.equity_hist).pct_change(fill_method=None).clip(lower=-0.99)
        ).dropna()
        if rets.empty:
            return 0.0
        var_q = rets.quantile(0.05)
        tail  = rets[rets <= var_q]
        return round(max(0.0, -float(tail.mean())), 10) if not tail.empty else 0.0

    def record_eod(self, prices: Dict[str, float]) -> None:
        """
        Update the portfolio's equity history at the end of a trading session.
        Calculates Net Asset Value (NAV) using current prices and handles missing data.

        Args:
            prices (Dict[str, float]): Map of symbol to current closing price.
        """
        max_absent_periods = (
            self.max_absent_periods
            if self.max_absent_periods is not None
            else DEFAULT_MAX_ABSENT_PERIODS
        )
        pv = self.cash
        for sym, n_shares in self.shares.items():
            px = prices.get(sym)
            if px is not None:
                px = float(px)
                self.last_known_prices[sym] = px
            else:
                last_px = self.last_known_prices.get(sym)
                if last_px is None:
                    logger.warning(
                        "record_eod: no price for %s and no last known price; treating as ₹0.", sym
                    )
                    px = 0.0
                else:
                    absent_n = int(self.absent_periods.get(sym, 0))
                    px = absent_symbol_effective_price(last_px, absent_n, max_absent_periods)
            pv += n_shares * float(px or 0.0)

        pv_rounded = round(float(pv), 10)
        self.equity_hist.append(pv_rounded)
        if self.equity_hist_cap is None:
            self._equity_hist_cap = max(500, int(UltimateConfig().CVAR_LOOKBACK * 2))
        else:
            self._equity_hist_cap = int(self.equity_hist_cap)
        if self._equity_hist_cap > 0 and len(self.equity_hist) > self._equity_hist_cap:
            self.equity_hist = self.equity_hist[-self._equity_hist_cap:]

    def record_volatility(self, current_date: pd.Timestamp, vol_by_symbol: Dict[str, float], cap: int) -> None:
        """Record realized volatility for symbols to be used in CVaR estimates."""
        effective_cap = int(cap)
        for sym, vol in vol_by_symbol.items():
            vol_value = float(vol)
            self.last_known_volatility[sym] = vol_value
            hist = self.vol_hist.get(sym)
            if hist is None:
                hist = deque(maxlen=effective_cap)
                self.vol_hist[sym] = hist
            elif hist.maxlen != effective_cap:
                hist = deque(hist, maxlen=effective_cap)
                self.vol_hist[sym] = hist
            hist.append((pd.Timestamp(current_date), vol_value))

    def reset(self) -> None:
        """Clear all holdings and history, resetting capital to the initial balance."""
        initial_cash = float(getattr(self, "_initial_cash", self.cash))
        self.shares = {}
        self.cash = initial_cash
        self.equity_hist = []
        self.weights = {}
        self.entry_prices = {}
        self.last_known_prices = {}
        self.last_known_volatility = {}
        self.vol_hist = {}
        self.absent_periods = {}
        self.exposure_multiplier = float(getattr(self, "_initial_exposure_multiplier", 1.0))
        self.override_active = False
        self.override_cooldown = 0
        self.consecutive_failures = 0
        self.decay_rounds = 0

    def to_dict(self) -> dict:
        """Serialize the entire portfolio state into a primitive dictionary for storage."""
        def _r(v):
            """Recursively round float values for stable JSON serialization.

            Args:
                v: Arbitrary nested value from portfolio state.

            Returns:
                Any: Value with floats rounded and mappings/lists normalized.

            Raises:
                Exception: Propagates unexpected recursion/type errors.
            """
            if isinstance(v, float):
                return round(v, 10)
            if isinstance(v, dict):
                return {k: _r(val) for k, val in sorted(v.items())}
            if isinstance(v, list):
                return [_r(x) for x in v]
            return v
        return {
            "weights":              _r(self.weights),
            "shares":               dict(sorted(self.shares.items())),
            "entry_prices":         _r(self.entry_prices),
            "equity_hist":          _r(self.equity_hist),
            "universe":             sorted(self.universe),
            "cash":                 _r(self.cash),
            "exposure_multiplier":  _r(self.exposure_multiplier),
            "override_active":      self.override_active,
            "override_cooldown":    self.override_cooldown,
            "consecutive_failures": self.consecutive_failures,
            "equity_hist_cap":      self.equity_hist_cap,
            "max_absent_periods":   self.max_absent_periods,
            "absent_periods":       dict(sorted(self.absent_periods.items())),
            "last_known_prices":    _r(self.last_known_prices),
            "last_known_volatility":_r(self.last_known_volatility),
            "vol_hist":             {
                k: {
                    "maxlen": vals.maxlen,
                    "data": [(pd.Timestamp(d).replace(tzinfo=None).isoformat(), float(v)) for d, v in vals],
                }
                for k, vals in sorted(self.vol_hist.items())
            },
            "decay_rounds":         self.decay_rounds,
            "dividend_ledger":      dict(sorted(self.dividend_ledger.items())),
            "last_rebalance_date":  self.last_rebalance_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioState":
        """
        Deserialize and restore a PortfolioState from a dictionary.
        Performs type validation and applies defaults for missing fields.
        """
        ps = cls()
        errors: List[str] = []
        risk_errors: List[str] = []

        def fetch(key: str, coercer: Any, default: Any) -> Any:
            return _deserialize_field(
                d, key, coercer, default, cls.RISK_CONTROL_FIELDS, errors, risk_errors
            )

        ps.weights = fetch("weights", _to_float_dict, {})
        ps.shares = fetch("shares", _to_int_dict, {})
        ps.entry_prices = fetch("entry_prices", _to_float_dict, {})
        ps.equity_hist = fetch("equity_hist", _to_float_list, [])
        ps.universe = fetch("universe", list, [])
        ps.cash = fetch("cash", float, ps.cash)
        ps.exposure_multiplier = fetch("exposure_multiplier", float, 1.0)
        ps.override_active = fetch("override_active", _as_bool_flag, False)
        ps.override_cooldown = fetch("override_cooldown", _as_nonneg_int, 0)
        ps.consecutive_failures = fetch("consecutive_failures", _as_nonneg_int, 0)
        ps.equity_hist_cap = _load_equity_hist_cap(d)
        ps.max_absent_periods = fetch("max_absent_periods", _as_optional_nonneg_int, None)
        ps.absent_periods = fetch("absent_periods", _to_int_dict, {})
        ps.last_known_prices = fetch("last_known_prices", _to_float_dict, {})
        ps.last_known_volatility = fetch("last_known_volatility", _to_float_dict, {})
        ps.vol_hist = fetch("vol_hist", _deserialize_vol_hist, {})
        ps.decay_rounds = fetch("decay_rounds", _as_nonneg_int, 0)
        ps.dividend_ledger = fetch("dividend_ledger", _to_str_dict, {})
        ps.last_rebalance_date = fetch("last_rebalance_date", str, "")
        ps._initial_cash = float(ps.cash)
        ps._initial_exposure_multiplier = float(ps.exposure_multiplier)
        if ps.equity_hist_cap is None:
            ps._equity_hist_cap = max(500, int(UltimateConfig().CVAR_LOOKBACK * 2))
        else:
            ps._equity_hist_cap = int(ps.equity_hist_cap)
        if ps._equity_hist_cap > 0 and len(ps.equity_hist) > ps._equity_hist_cap:
            ps.equity_hist = ps.equity_hist[-ps._equity_hist_cap:]

        if errors:
            logger.error(
                "PortfolioState.from_dict: %d field(s) reset to defaults: %s", len(errors), errors
            )
        if risk_errors:
            logger.critical(
                "PortfolioState.from_dict: %d risk-control field(s) reset to defaults: %s",
                len(risk_errors),
                risk_errors,
            )
        return ps


def _as_bool_flag(value: Any) -> bool:
    """Safely cast various input types (str, int) to a boolean flag."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"cannot parse bool from '{value}'")
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"integer bool flag must be 0/1, got {value}")
    raise TypeError(f"unsupported bool type: {type(value).__name__}")


def _as_nonneg_int(value: Any) -> int:
    """Parse a non-negative integer, raising ValueError if negative or unparseable."""
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"value must be non-negative, got {parsed}")
    return parsed


def _as_optional_nonneg_int(value: Any) -> Optional[int]:
    """Parse an optional non-negative integer (handles None)."""
    if value is None:
        return None
    return _as_nonneg_int(value)


def _to_float_dict(v: Any) -> Dict[str, float]:
    """Coerce a dictionary to {str: float}."""
    return {str(k): float(x) for k, x in v.items()}


def _to_int_dict(v: Any) -> Dict[str, int]:
    """Coerce a dictionary to {str: int}."""
    return {str(k): int(x) for k, x in v.items()}


def _to_str_dict(v: Any) -> Dict[str, str]:
    """Coerce a dictionary to {str: str}."""
    return {str(k): str(x) for k, x in v.items()}


def _to_float_list(v: Any) -> List[float]:
    """Coerce an iterable to [float]."""
    return [float(x) for x in v]


def _load_equity_hist_cap(payload: dict) -> Optional[int]:
    """Helper to extract and validate the history cap from a state payload."""
    if "equity_hist_cap" not in payload:
        return None
    value = payload.get("equity_hist_cap")
    if value is None:
        return None
    try:
        return _as_optional_nonneg_int(value)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        logger.warning(
            "PortfolioState.from_dict: invalid non-critical field 'equity_hist_cap' (%s); using None.",
            exc,
        )
        return None


def _deserialize_vol_hist(value: Any) -> Dict[str, deque]:
    """Deserialize volatility history deques from a list-based representation."""
    default_maxlen = max(500, int(UltimateConfig().CVAR_LOOKBACK * 2))
    out: Dict[str, deque] = {}
    for k, vals in value.items():
        try:
            rows = vals.get("data", []) if isinstance(vals, dict) else vals
            maxlen_raw = vals.get("maxlen", default_maxlen) if isinstance(vals, dict) else default_maxlen
            maxlen = int(maxlen_raw) if maxlen_raw is not None else default_maxlen
            out[str(k)] = deque(
                ((pd.Timestamp(d), float(vol)) for d, vol in rows),
                maxlen=maxlen,
            )
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            logger.warning(
                "PortfolioState.from_dict: skipping malformed vol_hist entry for %s (%s)",
                k,
                exc,
            )
    return out


def _deserialize_field(
    payload: dict,
    key: str,
    converter: Any,
    default: Any,
    risk_control_fields: Tuple[str, ...] = (),
    errors: Optional[List[str]] = None,
    risk_control_errors: Optional[List[str]] = None,
) -> Any:
    """
    Robust field extractor for state dictionary.
    Tracks whether errors occurred in sensitive risk-control fields.
    """
    try:
        if key not in payload:
            return default
        return converter(payload[key])
    except Exception as exc:
        # Broad catch is intentional: converter is caller-provided and may raise
        # arbitrary exception types depending on the target field.
        logger.warning(
            "PortfolioState.from_dict: field '%s' failed conversion (%s); using default.",
            key,
            exc,
        )
        if errors is not None or risk_control_errors is not None:
            msg = f"{key}: {exc}"
            if key in risk_control_fields and risk_control_errors is not None:
                risk_control_errors.append(msg)
            elif errors is not None:
                errors.append(msg)
        return default


def activate_override_on_stress(state: PortfolioState, cfg: UltimateConfig) -> None:
    """Manually trigger a risk override, halving exposure and starting a cooldown period."""
    state.override_active = True
    cooldown = int(cfg.OVERRIDE_COOLDOWN_PERIODS) if cfg and hasattr(cfg, "OVERRIDE_COOLDOWN_PERIODS") else 4
    state.override_cooldown = max(state.override_cooldown, cooldown)
    state.exposure_multiplier = float(max(cfg.MIN_EXPOSURE_FLOOR, state.exposure_multiplier * 0.5))


def compute_one_way_slip_rate(
    cfg: UltimateConfig,
    portfolio_value: float,
    adv_notional: Optional[float],
    trade_notional: Optional[float] = None,
) -> float:
    """
    Compute per-name one-way slippage rate including flat commission and market impact.
    """
    if adv_notional is None or not np.isfinite(adv_notional) or adv_notional <= 0:
        return cfg.ROUND_TRIP_SLIPPAGE_BPS / 20_000.0
    trade_value = (
        trade_notional
        if (trade_notional is not None and np.isfinite(trade_notional) and trade_notional > 0)
        else portfolio_value
    )
    return _compute_one_way_slip_rate_from_trade_value(cfg, float(adv_notional), float(trade_value))


def _compute_one_way_slip_rate_from_trade_value(
    cfg: UltimateConfig,
    adv_notional: Union[float, np.ndarray],
    trade_value: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute one-way slippage rate from already-resolved trade notional(s)."""
    base_rate = cfg.ROUND_TRIP_SLIPPAGE_BPS / 20_000.0
    adv_arr = np.asarray(adv_notional, dtype=float)
    trade_arr = np.asarray(trade_value, dtype=float)
    valid_adv = np.isfinite(adv_arr) & (adv_arr > 0)
    impact = np.zeros_like(trade_arr, dtype=float)
    np.divide(cfg.IMPACT_COEFF * trade_arr, adv_arr, out=impact, where=valid_adv)
    impact = np.clip(impact, 0.0, 0.05)
    rates = np.where(valid_adv, np.maximum(base_rate, impact), base_rate)
    if np.ndim(rates) == 0:
        return float(rates)
    return rates


def _resolve_trade_notional(
    portfolio_value: float,
    trade_notional: Optional[Union[float, np.ndarray]],
) -> Union[float, np.ndarray]:
    """Return provided trade notional when valid/positive, otherwise fall back to portfolio value."""
    if trade_notional is None:
        return float(portfolio_value)
    trade_arr = np.asarray(trade_notional, dtype=float)
    resolved = np.where(np.isfinite(trade_arr) & (trade_arr > 0), trade_arr, float(portfolio_value))
    if np.ndim(resolved) == 0:
        return float(resolved)
    return resolved


def _compute_one_way_slip_rate_vectorized(
    cfg: UltimateConfig,
    portfolio_value: float,
    adv_notional: np.ndarray,
    trade_notional: np.ndarray,
) -> np.ndarray:
    """
    Vectorized version of slippage calculation for bulk optimization constraints.

    Args:
        cfg (UltimateConfig): Configuration for slippage parameters.
        portfolio_value (float): Total portfolio value.
        adv_notional (np.ndarray): Matrix/array of symbol ADV values.
        trade_notional (np.ndarray): Matrix/array of intended trade sizes.

    Returns:
        np.ndarray: Matrix of calculated one-way slippage rates.
    """
    trade_values = _resolve_trade_notional(portfolio_value, trade_notional)
    return _compute_one_way_slip_rate_from_trade_value(cfg, adv_notional, trade_values)


# ─── Execution ────────────────────────────────────────────────────────────────

def _compute_pv_exec(
    state: PortfolioState,
    prices: np.ndarray,
    active_symbols: List[str],
    cfg: UltimateConfig,
    symbols_to_force_close: Set[str],
) -> Tuple[float, float]:
    """
    Calculate the execution-time Portfolio Value (PV).
    Determines both current PV and T-1 PV using execution prices.

    Args:
        state (PortfolioState): Current portfolio holdings.
        prices (np.ndarray): Execution prices for active symbols.
        active_symbols (List[str]): Symbols in the current optimization matrix.
        cfg (UltimateConfig): Config for absence modeling.
        symbols_to_force_close (Set[str]): Symbols being liquidated.

    Returns:
        Tuple: (pv_exec, pv_t1).
    """
    active_idx = {sym: i for i, sym in enumerate(active_symbols)}
    absent_snapshot: Dict[str, int] = dict(state.absent_periods)
    t1_price_snapshot: Dict[str, float] = dict(state.last_known_prices)
    pv_t1 = state.cash
    pv_exec = state.cash

    for sym, n_shares in state.shares.items():
        if sym in active_idx:
            px_exec = float(prices[active_idx[sym]])
            px_t1 = float(t1_price_snapshot.get(sym, 0.0))
            pv_exec += n_shares * px_exec
            pv_t1 += n_shares * px_t1
        elif sym not in symbols_to_force_close:
            px_exec = absent_symbol_effective_price(
                state.last_known_prices.get(sym, 0.0),
                state.absent_periods.get(sym, 0),
                cfg.MAX_ABSENT_PERIODS,
            )
            px_t1 = absent_symbol_effective_price(
                state.last_known_prices.get(sym, 0.0),
                absent_snapshot.get(sym, 0),
                cfg.MAX_ABSENT_PERIODS,
            )
            pv_exec += n_shares * px_exec
            pv_t1 += n_shares * px_t1
    return pv_exec, pv_t1


def _compute_desired_shares(
    target_weights: np.ndarray,
    prices: np.ndarray,
    pv_exec: float,
    adv_shares: Optional[np.ndarray],
    cfg: UltimateConfig,
    active_symbols: Optional[List[str]] = None,
    current_shares: Optional[Dict[str, int]] = None,
    conviction_scores: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, int], List[Tuple[int, str, float, float]]]:
    """
    Convert optimized target weights into integer share counts.
    Applies ADV caps and filters for valid targets (non-zero conviction).

    Args:
        target_weights (np.ndarray): Vector of floating point weights.
        prices (np.ndarray): Current execution prices.
        pv_exec (float): Portfolio value for sizing.
        adv_shares (Optional[np.ndarray]): Per-symbol liquidity limits.
        cfg (UltimateConfig): Configuration schema.
        active_symbols (Optional[List]): Symbol names.
        current_shares (Optional[Dict]): Existing holdings.
        conviction_scores (Optional[np.ndarray]): Alpha/signal scores for ranking.

    Returns:
        Tuple: (Map of symbol to shares, List of (index, symbol, price, score) for buy allocation).
    """
    desired_shares: Dict[str, int] = {}
    valid_targets: List[Tuple[int, str, float, float]] = []
    symbols = active_symbols if active_symbols is not None else [str(i) for i in range(len(target_weights))]
    current_shares = current_shares or {}
    for i, sym in enumerate(symbols):
        w = round(float(target_weights[i]), 10)
        if not np.isfinite(w):
            w = 0.0
        price = max(float(prices[i]), 1e-6)
        old_s = current_shares.get(sym, 0)
        current_notional = old_s * price
        s = _desired_shares_from_weight(w, pv_exec, current_notional, old_s, price)
        s = _apply_adv_share_cap(s, old_s, w, current_notional, pv_exec, price, adv_shares, i, cfg)

        desired_shares[sym] = s
        if s > 0:
            score = float(conviction_scores[i]) if conviction_scores is not None and i < len(conviction_scores) else w
            valid_targets.append((i, sym, price, score))
    return desired_shares, valid_targets


def _desired_shares_from_weight(
    target_weight: float,
    pv_exec: float,
    current_notional: float,
    old_shares: int,
    price: float,
) -> int:
    """Calculate the integer share count required to reach a target weight."""
    if target_weight * pv_exec > current_notional:
        buy_notional = max(0.0, target_weight * pv_exec - current_notional)
        return old_shares + int(np.floor(buy_notional / max(price, 1e-9)))
    return int(np.floor(target_weight * pv_exec / price))


def _apply_adv_share_cap(
    shares: int,
    old_shares: int,
    target_weight: float,
    current_notional: float,
    pv_exec: float,
    price: float,
    adv_shares: Optional[np.ndarray],
    index: int,
    cfg: UltimateConfig,
) -> int:
    """Enforce liquidity constraints by capping total holdings at a percentage of ADV."""
    if adv_shares is None or index >= len(adv_shares):
        return shares
    adv_notional = float(adv_shares[index])
    if not np.isfinite(adv_notional) or adv_notional <= 0:
        return shares
    max_adv_shares = int(np.floor((adv_notional * cfg.MAX_ADV_PCT) / price))
    capped = min(shares, max_adv_shares)
    current_weight = current_notional / max(pv_exec, 1.0)
    if capped < old_shares and target_weight >= current_weight:
        return old_shares
    return capped


def _apply_drift_gate(
    desired_shares: Dict[str, int],
    current_shares: Dict[str, int],
    target_weights: np.ndarray,
    prices: np.ndarray,
    pv_exec: float,
    cfg: UltimateConfig,
    active_symbols: List[str],
) -> Tuple[Dict[str, int], set[str]]:
    """
    Apply a drift tolerance gate to minimize unnecessary small trades.
    If the intended weight change is below a threshold, holdings are kept as-is.

    Args:
        desired_shares (Dict[str, int]): Proposed share counts.
        current_shares (Dict[str, int]): Existing share counts.
        target_weights (np.ndarray): Target weight vector.
        prices (np.ndarray): Asset prices.
        pv_exec (float): Portfolio value.
        cfg (UltimateConfig): Config for DRIFT_TOLERANCE.
        active_symbols (List[str]): List of active symbol names.

    Returns:
        Tuple: (Updated share counts, set of symbols that were gated).
    """
    drift_threshold = cfg.DRIFT_TOLERANCE
    drift_gated_syms: set[str] = set()
    active_idx = {sym: i for i, sym in enumerate(active_symbols)}
    for sym in active_symbols:
        if sym not in desired_shares:
            continue
        old_s = current_shares.get(sym, 0)
        s = desired_shares.get(sym, 0)
        i = active_idx[sym]
        price = max(float(prices[i]), 1e-6)
        if old_s > 0 and s > 0 and s != old_s:
            w_target = float(target_weights[i]) if np.isfinite(target_weights[i]) else 0.0
            w_current = (old_s * price) / max(pv_exec, 1.0)
            intended_change = abs(w_target - w_current)
            if intended_change < drift_threshold:
                desired_shares[sym] = old_s
                drift_gated_syms.add(sym)
    return desired_shares, drift_gated_syms


def _allocate_residual_cash(
    residual_budget: float,
    valid_targets: List[Tuple[int, str, float, float]],
    conviction_scores: Optional[np.ndarray],
    prices: np.ndarray,
    cfg: UltimateConfig,
) -> Dict[str, int]:
    """
    Distribute unallocated cash among buy candidates based on conviction scores.
    Maximizes capital efficiency by filling fractional weight gaps.

    Args:
        residual_budget (float): Remaining cash to deploy.
        valid_targets (List): Candidates for additional buying.
        conviction_scores (Optional[np.ndarray]): Ranking scores.
        prices (np.ndarray): Asset prices.
        cfg (UltimateConfig): Config overrides.

    Returns:
        Dict: Incremental shares to add for each symbol.
    """
    if residual_budget <= 0 or not valid_targets:
        return {}
    allocations: Dict[str, int] = {}
    total_score = 0.0
    for i, sym, _, score in valid_targets:
        score_i = float(conviction_scores[i]) if conviction_scores is not None and i < len(conviction_scores) else float(score)
        total_score += max(score_i, 0.0)
        allocations[sym] = 0
    if total_score <= 0:
        return allocations
    for i, sym, price, score in valid_targets:
        score_i = float(conviction_scores[i]) if conviction_scores is not None and i < len(conviction_scores) else float(score)
        budget = residual_budget * (max(score_i, 0.0) / total_score)
        allocations[sym] = int(budget // max(float(price), 1e-9))
    return allocations


def _collect_force_close_symbols(
    state: PortfolioState,
    prices: Dict[str, float],
    cfg: UltimateConfig,
    absence_threshold: int,
) -> set[str]:
    """Collect symbols to force-close after prolonged universe absence."""
    to_close: set[str] = set()
    for sym in list(state.shares.keys()):
        if sym in prices:
            continue
        count = state.absent_periods.get(sym, 0) + 1
        state.absent_periods[sym] = count
        if count >= absence_threshold:
            import logging
            logging.getLogger(__name__).warning(
                "execute_rebalance: %s absent for %d consecutive periods "
                "(≥ MAX_ABSENT_PERIODS=%d) — treating as delisted; closing position.",
                sym, count, absence_threshold,
            )
            to_close.add(sym)
    return to_close

def execute_rebalance(
    state:          PortfolioState,
    target_weights: np.ndarray,
    prices:         np.ndarray,
    active_symbols: List[str],
    cfg:            UltimateConfig,
    adv_shares:     Optional[np.ndarray] = None,
    date_context=None,
    trade_log:      Optional[List[Trade]] = None,
    apply_decay:    bool = False,
    scenario_losses: Optional[np.ndarray] = None,
    conviction_scores: Optional[np.ndarray] = None,
    force_rebalance_trades: bool = False,
) -> float:
    """
    Execute a portfolio rebalance, updating state in-place and logging trades.
    Implements a two-phase PV accounting strategy to handle force-closes and slippage.


    Args:
        state (PortfolioState): Mutable portfolio state to be updated.
        target_weights (np.ndarray): Optimized target weights (0.0 to 1.0).
        prices (np.ndarray): Execution prices for active symbols.
        active_symbols (List[str]): Symbols corresponding to the price/weight vectors.
        cfg (UltimateConfig): Configuration for slippage and caps.
        adv_shares (Optional[np.ndarray]): Average daily volume in shares.
        date_context (Any): Current simulation date index for logging.
        trade_log (Optional[List[Trade]]): List to append executed trades to.
        apply_decay (bool): If True, indicates a risk-reduction liquidation round.
        scenario_losses (Optional[np.ndarray]): Historical loss scenarios for CVaR check.
        conviction_scores (Optional[np.ndarray]): Signal/Alpha ranking for residual cash filling.
        force_rebalance_trades (bool): If True, bypasses the drift tolerance gate.

    Returns:
        float: Total slippage cost incurred during the rebalance.

    Raises:
        Exception: Propagates state-access or numeric errors during trading logic.
    """
    active_idx = {sym: i for i, sym in enumerate(active_symbols)}

    def _refresh_prices_and_absence_marks() -> np.ndarray:
        """Refresh local prices and clear absence counters for active symbols.

        Args:
            None.

        Returns:
            np.ndarray: Active-symbol prices with finite fallbacks applied.

        Raises:
            Exception: Propagates unexpected runtime or numeric conversion errors.
        """
        local_prices_out = np.array(prices, dtype=float, copy=True)
        for sym, i in active_idx.items():
            px = float(local_prices_out[i])
            if np.isfinite(px) and px > 0:
                state.last_known_prices[sym] = px
            else:
                px = float(state.last_known_prices.get(sym, 0.0))
            local_prices_out[i] = px
            state.absent_periods.pop(sym, None)
        return local_prices_out

    local_prices = _refresh_prices_and_absence_marks()
    price_dict = {sym: local_prices[i] for sym, i in active_idx.items()}
    symbols_to_force_close = _collect_force_close_symbols(state, price_dict, cfg, cfg.MAX_ABSENT_PERIODS)

    # Phase 1: build pv_exec excluding force-close candidates (see docstring)
    force_close_set = symbols_to_force_close
    pv_exec, pv_t1 = _compute_pv_exec(state, local_prices, active_symbols, cfg, force_close_set)

    if apply_decay:
        state.decay_rounds += 1
        logger.info(
            "execute_rebalance: decay round %d/%d — executing caller gate-filtered targets.",
            state.decay_rounds, cfg.MAX_DECAY_ROUNDS,
        )

    target_weights = np.clip(
        np.where(np.isfinite(target_weights), target_weights, 0.0),
        0.0,
        cfg.MAX_SINGLE_NAME_WEIGHT,
    )

    new_weights:      Dict[str, float] = {}
    new_shares:       Dict[str, int]   = {}
    new_entry_prices: Dict[str, float] = dict(state.entry_prices)
    total_slippage = actual_notional = 0.0

    if apply_decay and scenario_losses is not None:
        decay_check_w = np.maximum(target_weights[:len(active_symbols)], 0.0).astype(float)
        if symbols_to_force_close:
            force_weights = []
            for sym in symbols_to_force_close:
                n_shares = state.shares.get(sym, 0)
                px = float(state.last_known_prices.get(sym, 0.0))
                force_weights.append((n_shares * px) / max(pv_exec, 1.0))
        gross_w = float(np.sum(decay_check_w))
        if symbols_to_force_close and force_weights:
            gross_w += float(np.sum(force_weights))

        if gross_w > 1e-6 and scenario_losses.shape[1] == len(active_symbols):
            # FIX-NEW-ME-05: normalise decay_check_w to unit sum before computing
            # portfolio_losses so the CVaR is a pure return fraction.  Previously
            # gross_w could be < 1 when force-close positions inflate pv_exec but
            # are not represented in decay_check_w, causing scenario_losses @ w
            # to understate losses as a fraction of portfolio value and potentially
            # allow deployment when the hard limit should have triggered liquidation.
            normalised_w = decay_check_w / gross_w
            portfolio_losses = scenario_losses @ normalised_w
            T_sc      = len(portfolio_losses)
            tail_n    = max(1, int(np.floor(T_sc * (1.0 - cfg.CVAR_ALPHA))))
            tail_mean = float(np.mean(np.sort(portfolio_losses)[-tail_n:]))

            hard_limit = cfg.CVAR_DAILY_LIMIT * getattr(cfg, "CVAR_HARD_BREACH_MULTIPLIER", 1.5)

            if tail_mean > hard_limit + EPSILON:
                logger.error(
                    "execute_rebalance: POST-DECAY CVaR %.4f%% exceeds hard limit %.4f%%. "
                    "Liquidating all positions to cash.",
                    tail_mean * 100, hard_limit * 100,
                )
                exit_slip_rate = cfg.ROUND_TRIP_SLIPPAGE_BPS / 20000.0
                for sym, n_shares in state.shares.items():
                    if sym in symbols_to_force_close:
                        px_exec = absent_symbol_effective_price(
                            float(state.last_known_prices.get(sym, 0.0)),
                            state.absent_periods.get(sym, 0),
                            cfg.MAX_ABSENT_PERIODS,
                        )
                    elif sym in active_idx:
                        px_exec = float(local_prices[active_idx[sym]])
                    else:
                        # GHOST or off-universe symbol: Preserve unless explicitly force-closed
                        continue

                    if px_exec > 0 and n_shares > 0:
                        # FIX-MB-C-03: full-liquidation returns before Phase 2,
                        # so any force-close candidate excluded from Phase 1
                        # must have its proceeds restored here exactly once.
                        # Active/retained names are already included in pv_exec;
                        # only symbols explicitly excluded from Phase 1 need this.
                        if sym in symbols_to_force_close:
                            pv_exec += n_shares * px_exec
                        slip            = n_shares * px_exec * exit_slip_rate
                        total_slippage += slip
                        if trade_log is not None:
                            tdate = pd.Timestamp(date_context) if date_context is not None else pd.Timestamp.now("UTC").replace(tzinfo=None)
                            trade_log.append(
                                Trade(sym, tdate, -n_shares, px_exec, slip, "SELL")
                            )
                state.weights      = {}
                state.shares       = {}
                state.entry_prices = {}
                state.cash         = max(0.0, round(pv_exec - total_slippage, 10))
                state.decay_rounds = 0
                state.consecutive_failures = 0
                return round(total_slippage, 10)

    desired_shares, valid_targets = _compute_desired_shares(
        target_weights=target_weights,
        prices=local_prices,
        pv_exec=pv_exec,
        adv_shares=adv_shares,
        cfg=cfg,
        active_symbols=active_symbols,
        current_shares=state.shares,
        conviction_scores=conviction_scores,
    )

    if not force_rebalance_trades:
        # Collect gated symbols so we can purge them from valid_targets below.
        desired_shares, drift_gated_syms = _apply_drift_gate(
            desired_shares=desired_shares,
            current_shares=state.shares,
            target_weights=target_weights,
            prices=local_prices,
            pv_exec=pv_exec,
            cfg=cfg,
            active_symbols=active_symbols,
        )

        # FIX-DRIFT-GATE-RESIDUAL: Remove gated symbols from valid_targets so
        # the residual cash allocator cannot issue tiny top-up trades that the
        # drift gate explicitly blocked.  Without this purge a gated symbol with
        # weight 1.5% (gated; target was 2%) still appears in eligible and
        # receives a cash_entitlement, producing a small slippage-incurring buy
        # that negates the gate's purpose.
        if drift_gated_syms:
            valid_targets = [vt for vt in valid_targets if vt[1] not in drift_gated_syms]

    base_notional = sum(
        desired_shares.get(sym, 0) * max(float(local_prices[i]), 1e-6)
        for i, sym in enumerate(active_symbols)
    )

    residual_cash = max(0.0, pv_exec - base_notional)
    # Canonical implementation: keep residual-cash allocation inline so
    # ADV caps, concentration limits, and multi-pass budget tracking stay in
    # one place with execution-time state.

    if valid_targets and residual_cash > 0:
        eligible = {
            sym: {"i": i, "price": price, "w": float(target_weights[i])}
            for i, sym, price, _ in valid_targets
            if np.isfinite(target_weights[i]) and target_weights[i] > 0
        }

        while eligible and residual_cash > 0:
            total_eligible_w = sum(data["w"] for data in eligible.values())
            if total_eligible_w < 1e-8:
                break

            shares_bought_this_pass = 0
            to_remove = []
            eligible_syms = list(eligible.keys())
            prices_arr = np.array([eligible[s]["price"] for s in eligible_syms], dtype=float)
            weights_arr = np.array([eligible[s]["w"] for s in eligible_syms], dtype=float)
            idx_arr = np.array([int(eligible[s]["i"]) for s in eligible_syms], dtype=int)
            cash_entitlements = (weights_arr / total_eligible_w) * residual_cash
            adv_limits_arr = np.full(len(eligible_syms), np.iinfo(np.int64).max, dtype=np.int64)
            if adv_shares is not None:
                adv_notional_arr = adv_shares[idx_arr].astype(float)
            else:
                adv_notional_arr = np.full(len(eligible_syms), np.nan, dtype=float)
            slip_rates = _compute_one_way_slip_rate_vectorized(
                cfg=cfg,
                portfolio_value=pv_exec,
                adv_notional=adv_notional_arr,
                trade_notional=cash_entitlements,
            )
            effective_prices = prices_arr * (1.0 + slip_rates)
            extra_shares = np.floor(cash_entitlements / np.maximum(effective_prices, 1e-9)).astype(int)
            current_shares_arr = np.array([desired_shares[s] for s in eligible_syms], dtype=int)
            headroom_notional = cfg.MAX_SINGLE_NAME_WEIGHT * pv_exec - current_shares_arr * prices_arr
            headroom_arr = np.maximum(0, np.floor(headroom_notional / np.maximum(effective_prices, 1e-9)).astype(int))
            if adv_shares is not None:
                valid_adv_mask = np.isfinite(adv_notional_arr) & (adv_notional_arr > 0)
                max_adv_total = np.zeros(len(eligible_syms), dtype=int)
                max_adv_total[valid_adv_mask] = np.floor(
                    (adv_notional_arr[valid_adv_mask] * cfg.MAX_ADV_PCT) / np.maximum(effective_prices[valid_adv_mask], 1e-9)
                ).astype(int)
                adv_limits_arr = np.maximum(0, max_adv_total - current_shares_arr)
            capped_extra = np.minimum(extra_shares, np.minimum(headroom_arr, adv_limits_arr))
            capped_extra = np.maximum(capped_extra, 0)

            for pos, sym in enumerate(eligible_syms):
                actual_extra = int(capped_extra[pos])
                eff_px = float(effective_prices[pos])
                if actual_extra > 0:
                    desired_shares[sym] += actual_extra
                    residual_cash -= actual_extra * eff_px
                    shares_bought_this_pass += actual_extra
                if actual_extra >= int(min(headroom_arr[pos], adv_limits_arr[pos])) or eff_px > residual_cash:
                    to_remove.append(sym)

            for sym in to_remove:
                eligible.pop(sym, None)

            if shares_bought_this_pass == 0:
                break

    for i, sym in enumerate(active_symbols):
        w = round(float(target_weights[i]), 10)
        if not np.isfinite(w):
            w = 0.0
        price = max(float(local_prices[i]), 1e-6)
        old_s = state.shares.get(sym, 0)
        s = desired_shares.get(sym, 0)

        if s > 0 or old_s > 0:
            delta = s - old_s
            trade_not = abs(delta) * price

            slip_rate = compute_one_way_slip_rate(
                cfg=cfg,
                portfolio_value=pv_exec,
                adv_notional=float(adv_shares[i]) if adv_shares is not None else None,
                trade_notional=trade_not,
            )

            # FIX-MB-SLIP: slip cost is the one-way cost on the traded notional,
            # deducted from cash once. effective_buy_price was used only for
            # share-count sizing above, not for this cash deduction.
            slip = abs(delta) * price * slip_rate
            total_slippage += slip
            actual_notional += s * price

            if s > 0:
                new_weights[sym] = (s * price) / max(pv_exec, 1.0)
                new_shares[sym]  = s
                if delta > 0:
                    if old_s == 0:
                        new_entry_prices[sym] = price * (1.0 + slip_rate)
                        marker_date = (
                            pd.Timestamp(date_context).strftime("%Y-%m-%d")
                            if date_context is not None else pd.Timestamp.now("UTC").replace(tzinfo=None).strftime("%Y-%m-%d")
                        )
                        state.dividend_ledger[sym] = f"{marker_date}:0.00000000"
                    else:
                        old_basis = new_entry_prices.get(sym, price)
                        new_entry_prices[sym] = (old_basis * old_s + price * (1.0 + slip_rate) * delta) / s

            if delta != 0 and trade_log is not None:
                tdate = pd.Timestamp(date_context) if date_context is not None else pd.Timestamp.now("UTC").replace(tzinfo=None)
                trade_log.append(
                    Trade(sym, tdate, delta, price, slip, "BUY" if delta > 0 else "SELL")
                )

    for sym in state.shares:
        if sym not in active_idx and sym not in symbols_to_force_close:
            new_shares[sym]       = state.shares[sym]
            new_weights[sym]      = state.weights.get(sym, 0.0)
            new_entry_prices[sym] = state.entry_prices.get(sym, 0.0)
            actual_notional      += new_shares[sym] * absent_symbol_effective_price(
                state.last_known_prices.get(sym, 0.0),
                state.absent_periods.get(sym, 0),
                cfg.MAX_ABSENT_PERIODS,
            )

    # Phase 2: force-close positions — proceeds added to pv_exec (see docstring)
    # Execute the terminal close at the symbol's last known price so cash and
    # trade logs reflect the actual forced liquidation level at the absence
    # threshold, while earlier mark-to-market bars continue to use the haircut.
    for sym in symbols_to_force_close:
        close_price = float(state.last_known_prices.get(sym, 0.0))
        n_shares    = state.shares.get(sym, 0)
        if n_shares > 0:
            if close_price > 0:
                slip            = n_shares * close_price * (cfg.ROUND_TRIP_SLIPPAGE_BPS / 20000.0)
                total_slippage += slip
                pv_exec        += n_shares * close_price
                if trade_log is not None:
                    tdate = pd.Timestamp(date_context) if date_context is not None else pd.Timestamp.now("UTC").replace(tzinfo=None)
                    trade_log.append(Trade(sym, tdate, -n_shares, close_price, slip, "SELL"))
            else:
                logger.error(
                    "execute_rebalance: force-close of %s (%d shares) has no last "
                    "known price — position removed at ₹0.",
                    sym, n_shares,
                )
                if trade_log is not None:
                    tdate = pd.Timestamp(date_context) if date_context is not None else pd.Timestamp.now("UTC").replace(tzinfo=None)
                    trade_log.append(Trade(sym, tdate, -n_shares, 0.0, 0.0, "SELL"))
        state.absent_periods.pop(sym, None)

    for sym in list(new_entry_prices):
        if sym not in new_shares:
            del new_entry_prices[sym]

    state.weights      = new_weights
    state.shares       = new_shares
    state.entry_prices = new_entry_prices
    state.consecutive_failures = 0
    # FIX-MB-CASH-FLOOR: Validate cash doesn't go negative
    raw_cash = pv_exec - actual_notional - total_slippage
    # FIX-BUG-4: use raw_cash in round() rather than re-evaluating the expression.
    # The original code computed pv_exec - actual_notional - total_slippage twice —
    # raw_cash was computed but then ignored in favour of a duplicate inline expression.
    state.cash         = max(0.0, round(raw_cash, 10))
    if raw_cash < -1e-6:
        logger.warning(
            "execute_rebalance: Slippage overshoot; raw cash would be %.4f",
            raw_cash
        )
    return round(total_slippage, 10)


# ─── Risk & Target Helpers ────────────────────────────────────────────────────

def compute_book_cvar(
    state:          PortfolioState,
    prices:         np.ndarray,
    active_symbols: List[str],
    hist_log_rets:  pd.DataFrame,
    cfg:            UltimateConfig,
) -> float:
    """
    Calculate the current portfolio's Conditional Value-at-Risk (CVaR).
    Uses a historical simulation approach with ghost synthesis for symbols with insufficient history.

    Args:
        state (PortfolioState): Current holdings and volatility history.
        prices (np.ndarray): Mark-to-market prices for active symbols.
        active_symbols (List[str]): List of symbol names in the pricing vector.
        hist_log_rets (pd.DataFrame): Historical log returns for market assets.
        cfg (UltimateConfig): Configuration for lookbacks and confidence levels.

    Returns:
        float: Expected loss in the worst (1 - alpha)% of scenarios.

    Raises:
        Exception: Propagates numeric errors during ghost synthesis or tail calculation.
    """
    active_idx = {sym: i for i, sym in enumerate(active_symbols)}
    mtm_weights, pv = _build_mtm_weights_and_pv(state, prices, active_idx)

    if not np.isfinite(pv):
        return float("inf")
    if pv <= 1e-6:
        return 0.0

    held_syms = list(mtm_weights.keys())
    T_cvar = min(len(hist_log_rets), cfg.CVAR_LOOKBACK)
    rets = _prepare_cvar_returns(hist_log_rets, held_syms, T_cvar)
    _record_current_volatility(state, hist_log_rets, cfg)
    _apply_ghost_return_synthesis(state, rets, held_syms, active_idx, cfg)

    rets = rets.fillna(0.0)

    if len(rets) < 5:
        return 0.0

    w = np.array([mtm_weights[s] / pv for s in held_syms], dtype=float)
    portfolio_losses = -(rets.values @ w)

    sorted_losses = np.sort(portfolio_losses)
    tail_n        = max(1, int(np.floor(T_cvar * (1.0 - cfg.CVAR_ALPHA))))
    tail_mean     = float(np.mean(sorted_losses[-tail_n:]))

    return tail_mean


def _build_mtm_weights_and_pv(
    state: PortfolioState,
    prices: np.ndarray,
    active_idx: Dict[str, int],
) -> Tuple[Dict[str, float], float]:
    """
    Compute current mark-to-market valuations for the held portfolio.

    Args:
        state (PortfolioState): Current shares and last known prices.
        prices (np.ndarray): Vector of latest market prices.
        active_idx (Dict[str, int]): Index mapping symbols to price vector positions.

    Returns:
        Tuple[Dict[str, float], float]: (symbol_notionals_dict, total_portfolio_value).
    """
    mtm_weights: Dict[str, float] = {}
    pv = state.cash
    for sym, n_shares in state.shares.items():
        if sym in active_idx:
            px = float(prices[active_idx[sym]])
        else:
            px = state.last_known_prices.get(sym, 0.0)
        notional = n_shares * px
        mtm_weights[sym] = notional
        pv += notional
    return mtm_weights, pv


def _prepare_cvar_returns(hist_log_rets: pd.DataFrame, held_syms: List[str], t_cvar: int) -> pd.DataFrame:
    """
    Slice and fill historical log returns for CVaR evaluation.

    Args:
        hist_log_rets (pd.DataFrame): Wide matrix of historical log returns.
        held_syms (List[str]): Symbols currently in the portfolio or target universe.
        t_cvar (int): Tail lookback window size.

    Returns:
        pd.DataFrame: Sliced returns matrix with inf/nan handled.
    """
    rets = hist_log_rets.reindex(columns=held_syms, fill_value=np.nan)
    return rets.replace([np.inf, -np.inf], np.nan).ffill().iloc[-t_cvar:].copy()


def _record_current_volatility(state: PortfolioState, hist_log_rets: pd.DataFrame, cfg: UltimateConfig) -> None:
    """Update volatility history buffers with latest rolling standard deviations."""
    vol_window = max(5, cfg.GHOST_VOL_LOOKBACK)
    if hist_log_rets.empty:
        return
    rolling_vol = (
        hist_log_rets.replace([np.inf, -np.inf], np.nan)
        .iloc[-vol_window:]
        .std()
        .dropna()
    )
    state.record_volatility(
        current_date=pd.Timestamp(hist_log_rets.index[-1]),
        vol_by_symbol={str(sym_key): float(max(vol, 1e-4)) for sym_key, vol in rolling_vol.items()},
        cap=cfg.CVAR_LOOKBACK,
    )


def _row_seeds_from_index(index: pd.Index, sym_base_seed: int) -> np.ndarray:
    """Generate daily-varying seeds for deterministic Monte Carlo synthesis."""
    if hasattr(index, "asi8"):
        idx = pd.DatetimeIndex(index)
        if idx.tz is None:
            idx_utc = idx.tz_localize("UTC")
        else:
            idx_utc = idx.tz_convert("UTC")
        days_since_epoch = (idx_utc.normalize().asi8 // np.int64(86_400 * 10 ** 9)).astype(np.int64)
    else:
        days_since_epoch = np.arange(len(index), dtype=np.int64)
    return np.uint64(sym_base_seed) ^ days_since_epoch.astype(np.uint64)


def _ghost_vol_series_for_symbol(
    state: PortfolioState,
    sym: str,
    index: pd.Index,
    cfg: UltimateConfig,
) -> np.ndarray:
    """Construct a volatility time-series for a ghost asset using last known values."""
    vol_series = np.full(len(index), float(cfg.GHOST_VOL_FALLBACK), dtype=float)
    sym_hist = list(state.vol_hist.get(sym, deque()))
    if not sym_hist:
        return vol_series
    hist_dates = [pd.Timestamp(d) for d, _ in sym_hist]
    hist_vals = [float(v) for _, v in sym_hist]
    for idx_row, row_date in enumerate(pd.DatetimeIndex(index)):
        nearest = None
        for j, d_hist in enumerate(hist_dates):
            if d_hist <= row_date:
                nearest = j
            else:
                break
        if nearest is not None:
            vol_series[idx_row] = max(hist_vals[nearest], cfg.GHOST_VOL_FALLBACK)
    return vol_series


def _apply_ghost_return_synthesis(
    state: PortfolioState,
    rets: pd.DataFrame,
    held_syms: List[str],
    active_idx: Dict[str, int],
    cfg: UltimateConfig,
) -> None:
    """Fill missing return data with synthetic normally distributed noise (Ghosting)."""
    ghost_mask = np.array([(s not in active_idx) or (s in rets.columns and rets[s].isna().all()) for s in held_syms])
    if not ghost_mask.any():
        return
    ghost_cols = sorted(s for s, is_ghost in zip(held_syms, ghost_mask, strict=True) if is_ghost)
    for sym in ghost_cols:
        daily_drift = float(cfg.GHOST_RET_DRIFT) / 252.0
        row_seeds = _row_seeds_from_index(rets.index, _ghost_seed_for(sym))
        vol_series = _ghost_vol_series_for_symbol(state, sym, rets.index, cfg)
        raw = np.array([np.random.default_rng(int(seed)).standard_normal() for seed in row_seeds], dtype=float)
        rets.loc[:, sym] = daily_drift + vol_series * raw


def compute_decay_targets(
    state:          PortfolioState,
    sel_idx:        List[int],
    active_symbols: List[str],
    cfg:            UltimateConfig,
    current_prices: Optional[np.ndarray] = None,
    pv:             Optional[float] = None,
) -> np.ndarray:
    """Compute decayed target weights for the gate-passing symbols.

    Prefer true mark-to-market weights derived from current prices and
    portfolio value rather than stale state.weights. For backward-compatible
    test and utility call sites that omit current_prices / pv, fall back to
    the persisted state.weights snapshot.
    """
    targets = np.zeros(len(active_symbols))
    sel_set = set(sel_idx)
    use_mtm = current_prices is not None and pv is not None

    for i, sym in enumerate(active_symbols):
        if i not in sel_set:
            continue

        if use_mtm:
            assert current_prices is not None
            assert pv is not None
            shares = state.shares.get(sym, 0)
            price = max(float(current_prices[i]), 1e-6)
            pre_decay_weight = (shares * price) / max(float(pv), 1.0)
        else:
            pre_decay_weight = float(state.weights.get(sym, 0.0))

        pre_decay_weight = min(max(pre_decay_weight, 0.0), cfg.MAX_SINGLE_NAME_WEIGHT)
        targets[i] = pre_decay_weight * cfg.DECAY_FACTOR

    return targets


# ─── Optimizer ────────────────────────────────────────────────────────────────

def raw_rets_empty_after_sanitization(historical_returns: pd.DataFrame) -> bool:
    """Check if the return dataframe is empty or entirely non-finite."""
    return historical_returns.replace([np.inf, -np.inf], np.nan).empty


def _validate_optimizer_input_shapes(
    expected_returns: np.ndarray,
    historical_returns: pd.DataFrame,
    adv_shares: np.ndarray,
    prices: np.ndarray,
    prev_w: Optional[np.ndarray],
    sector_labels: Optional[np.ndarray],
    portfolio_value: float,
) -> None:
    """Perform structural and sanity checks on all optimizer inputs to prevent logic crashes."""
    original_m = len(expected_returns)
    if len(prices) != original_m or len(adv_shares) != original_m:
        raise OptimizationError(
            "Input length mismatch across expected_returns/prices/adv_shares.",
            OptimizationErrorType.DATA,
        )
    if prev_w is not None and len(prev_w) != original_m:
        raise OptimizationError(
            "prev_w length must match expected_returns length.",
            OptimizationErrorType.DATA,
        )
    if sector_labels is not None and len(sector_labels) != original_m:
        raise OptimizationError(
            "Sector mapping array length does not match expected_returns length.",
            OptimizationErrorType.DATA,
        )
    if not np.all(np.isfinite(expected_returns)):
        raise OptimizationError("expected_returns contains non-finite values.", OptimizationErrorType.DATA)
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise OptimizationError("prices must be finite and strictly positive.", OptimizationErrorType.DATA)
    if not np.all(np.isfinite(adv_shares)) or np.any(adv_shares < 0):
        raise OptimizationError("adv_shares must be finite and non-negative.", OptimizationErrorType.DATA)
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise OptimizationError(
            "portfolio_value must be finite and strictly positive.",
            OptimizationErrorType.DATA,
        )
    if raw_rets_empty_after_sanitization(historical_returns):
        raise OptimizationError("historical_returns is empty after sanitisation.", OptimizationErrorType.DATA)
    if historical_returns.shape[1] != original_m:
        raise OptimizationError(
            "historical_returns columns must align with expected_returns length.",
            OptimizationErrorType.DATA,
        )


def _apply_history_gate(
    raw_rets: pd.DataFrame,
    cfg: UltimateConfig,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], int, int]:
    """Excludes assets from optimization if they lack sufficient valid historical data points."""
    lookback = min(max(int(cfg.HISTORY_GATE), 1), len(raw_rets))
    min_valid_ratio = 0.70
    required_count = max(1, int(np.ceil(lookback * min_valid_ratio)))
    valid_counts = raw_rets.tail(lookback).notna().sum()
    keep_mask = valid_counts >= required_count
    kept_indices = np.flatnonzero(keep_mask.to_numpy())
    excluded_symbols = valid_counts.index[~keep_mask].tolist()
    clean_rets = raw_rets.iloc[:, kept_indices].ffill()
    return clean_rets, kept_indices, excluded_symbols, lookback, required_count


def _fill_missing_returns(clean_rets: pd.DataFrame) -> pd.DataFrame:
    """Fill non-systemic NaNs with cross-sectional row means (market-neutral proxy)."""
    row_means = clean_rets.mean(axis=1)
    return clean_rets.apply(lambda col: col.fillna(row_means)).fillna(0.0)


def _drop_zero_volatility_columns(
    clean_rets: pd.DataFrame,
    kept_indices: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Remove assets with zero variance to avoid singular covariance matrix issues."""
    col_stds = clean_rets.std()
    valid_vol_mask = col_stds >= 1e-10
    if not valid_vol_mask.any():
        return clean_rets, kept_indices
    if not valid_vol_mask.all():
        clean_rets = clean_rets.loc[:, valid_vol_mask]
        kept_indices = kept_indices[valid_vol_mask.to_numpy()]
    return clean_rets, kept_indices


def _compute_exposure_bounds(
    cfg: UltimateConfig,
    exposure_multiplier: float,
    adv_limit: np.ndarray,
    sector_labels: Optional[np.ndarray],
) -> Tuple[float, float, float]:
    """Calculate the lower and upper bounds for total portfolio gross exposure."""
    gamma = float(np.clip(exposure_multiplier, cfg.MIN_EXPOSURE_FLOOR, 1.0))
    l_gamma = max(cfg.MIN_EXPOSURE_FLOOR, gamma * (1.0 - cfg.CAPITAL_ELASTICITY))
    u_gamma = min(1.0, gamma)

    max_possible_weight = 0.0
    if sector_labels is not None:
        labels = np.asarray(sector_labels, dtype=int)
        for sec_id in np.unique(labels):
            mask = labels == sec_id
            sec_adv_sum = float(np.sum(adv_limit[mask]))
            if sec_id == -1:
                max_possible_weight += sec_adv_sum
            else:
                max_possible_weight += min(sec_adv_sum, cfg.MAX_SECTOR_WEIGHT)
    else:
        max_possible_weight = float(np.sum(adv_limit))

    u_gamma = min(u_gamma, max_possible_weight)
    if max_possible_weight < l_gamma:
        l_gamma = max_possible_weight * 0.99
    if np.sum(adv_limit) < l_gamma:
        l_gamma = np.sum(adv_limit) * 0.99
    u_gamma = max(u_gamma, l_gamma)
    return gamma, l_gamma, u_gamma


class InstitutionalRiskEngine:
    """Risk optimizer wrapper with reusable solver cache.

    When reusing an instance across backtests with different date ranges,
    call reset_solver() between runs to clear cached solver state.
    """
    # NOT thread-safe across optimize() calls without _solver_lock.
    def __init__(self, cfg: UltimateConfig):
        """
        Initialize the institutional risk engine.

        Args:
            cfg (UltimateConfig): Configuration schema for limits and penalties.
        """
        self.cfg:       UltimateConfig              = cfg
        self.last_diag: Optional[SolverDiagnostics] = None
        self._solver:       Optional[Any] = None
        self._solver_shape: Optional[tuple]  = None
        self._solver_nnz:   Optional[tuple]  = None
        self._solver_struct: Optional[tuple] = None
        self._solver_lock = threading.Lock()

    def reset_solver(self) -> None:
        """
        Invalidate the cached OSQP solver and its structural metadata.
        Forces a full setup (A, P, l, u matrices) on the next optimize() call.
        """
        with self._solver_lock:
            self._solver = None
            self._solver_shape = None
            self._solver_nnz = None
            self._solver_struct = None

    def _preprocess_optimization_inputs(
        self,
        expected_returns: np.ndarray,
        historical_returns: pd.DataFrame,
        adv_shares: np.ndarray,
        prices: np.ndarray,
        portfolio_value: float,
        prev_w: Optional[np.ndarray],
        exposure_multiplier: float,
        sector_labels: Optional[np.ndarray],
        execution_date: Optional[pd.Timestamp],
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int, int]:
        original_m = len(expected_returns)
        if execution_date is not None and not historical_returns.empty:
            if historical_returns.index.max() > pd.Timestamp(execution_date):
                raise OptimizationError(
                    "T-1 violation: historical_returns include execution_date.",
                    OptimizationErrorType.DATA,
                )

        _validate_optimizer_input_shapes(
            expected_returns=expected_returns,
            historical_returns=historical_returns,
            adv_shares=adv_shares,
            prices=prices,
            prev_w=prev_w,
            sector_labels=sector_labels,
            portfolio_value=portfolio_value,
        )

        raw_rets = historical_returns.replace([np.inf, -np.inf], np.nan)
        clean_rets, kept_indices, excluded_symbols, lookback, required_count = _apply_history_gate(raw_rets, self.cfg)

        if excluded_symbols:
            logger.info(
                "[OptimizerGuard] excluded_symbols=%d reason=insufficient_history "
                "lookback=%d required_non_nan=%d symbols=%s",
                len(excluded_symbols),
                lookback,
                required_count,
                excluded_symbols,
            )

        if len(kept_indices) == 0:
            raise OptimizationError(
                "No symbols passed optimizer minimum-history gate.",
                OptimizationErrorType.DATA,
            )

        clean_rets = _fill_missing_returns(clean_rets)

        col_stds = clean_rets.std()
        valid_vol_mask = col_stds >= 1e-10
        if not valid_vol_mask.all():
            zero_cols = valid_vol_mask[~valid_vol_mask].index.tolist()
            logger.warning(
                "[Optimizer] Detected %d zero-volatility asset(s): %s",
                len(zero_cols), zero_cols,
            )
        clean_rets, kept_indices = _drop_zero_volatility_columns(clean_rets, kept_indices)

        expected_returns = expected_returns[kept_indices]
        prices = prices[kept_indices]
        adv_shares = adv_shares[kept_indices]
        if prev_w is not None:
            prev_w = prev_w[kept_indices]
        if sector_labels is not None:
            sector_labels = np.asarray(sector_labels)[kept_indices]

        m = len(kept_indices)
        T          = len(clean_rets)
        min_rows   = self.cfg.DIMENSIONALITY_MULTIPLIER * m
        if T < min_rows:
            raise OptimizationError(
                f"Insufficient history: {T} rows for {m} assets.", OptimizationErrorType.DATA
            )

        return clean_rets, expected_returns, prices, adv_shares, prev_w, sector_labels, kept_indices, m, T

    def _build_optimization_constraints(
        self,
        clean_rets: pd.DataFrame,
        expected_returns: np.ndarray,
        adv_shares: np.ndarray,
        portfolio_value: float,
        prev_w: Optional[np.ndarray],
        exposure_multiplier: float,
        sector_labels: Optional[np.ndarray],
        m: int,
        T: int,
    ) -> tuple[Any, np.ndarray, Any, list, list, int, float, float, float, np.ndarray, Any]:
        import scipy.sparse as sp
        from sklearn.covariance import LedoitWolf

        simple_rets = np.expm1(clean_rets)
        lw = LedoitWolf()
        lw.fit(simple_rets)
        Sigma_reg = lw.covariance_
        ridge     = 0.0

        gamma = float(np.clip(exposure_multiplier, self.cfg.MIN_EXPOSURE_FLOOR, 1.0))

        adv_w           = adv_shares / np.maximum(adv_shares.sum(), 1e-9)
        adv_w_series    = pd.Series(adv_w, index=clean_rets.columns)
        aligned_w       = adv_w_series.reindex(clean_rets.columns).fillna(0.0).values
        aligned_w       = aligned_w / np.maximum(aligned_w.sum(), 1e-9)
        adv_weighted_rets = pd.Series(
            simple_rets.values.dot(aligned_w), index=simple_rets.index
        )
        var_95   = adv_weighted_rets.quantile(1 - self.cfg.CVAR_ALPHA)
        ew_cvar  = (
            -float(adv_weighted_rets[adv_weighted_rets <= var_95].mean())
            if not adv_weighted_rets.empty else 0.0
        )
        sentinel = self.cfg.CVAR_DAILY_LIMIT * self.cfg.CVAR_SENTINEL_MULTIPLIER

        if ew_cvar > sentinel + EPSILON:
            logger.warning(
                "Selection ADV-weighted CVaR %.2f%% exceeds sentinel %.2f%%. "
                "Forcing 50%% exposure reduction.",
                ew_cvar * 100, sentinel * 100,
            )
            gamma *= 0.5
            gamma = max(self.cfg.MIN_EXPOSURE_FLOOR, gamma)

        adv_limit = np.clip((adv_shares * self.cfg.MAX_ADV_PCT) / portfolio_value, 1e-9, 0.40)
        adv_limit = np.minimum(adv_limit, self.cfg.MAX_SINGLE_NAME_WEIGHT)

        gamma, l_gamma, u_gamma = _compute_exposure_bounds(
            self.cfg,
            gamma,
            adv_limit,
            sector_labels,
        )

        impact     = np.clip(
            self.cfg.IMPACT_COEFF * portfolio_value
            / np.maximum(adv_shares, 1.0),
            0.0, 1e4,
        )
        T_cvar     = min(T, self.cfg.CVAR_LOOKBACK)
        losses     = -simple_rets.iloc[-T_cvar:].values
        n_vars     = 2 * m + 1 + T_cvar + 1
        prev_w_arr = prev_w if prev_w is not None else np.zeros(m)

        P_w   = 2.0 * (self.cfg.RISK_AVERSION * Sigma_reg + np.diag(impact))
        P_aux = sp.eye(n_vars - m, format="csc") * 1e-6
        P     = sp.block_diag([sp.csc_matrix(P_w), P_aux], format="csc")

        target_weight_hint = np.clip(expected_returns, 0.0, None)
        total_hint = float(np.sum(target_weight_hint))
        if total_hint > 0:
            target_weight_hint = target_weight_hint / total_hint
        else:
            target_weight_hint = np.zeros_like(expected_returns, dtype=float)
        scaled_target_weight_hint = target_weight_hint * float(u_gamma)
        trade_estimate_notionals = np.abs(scaled_target_weight_hint - prev_w_arr) * float(portfolio_value)

        turnover_costs = _compute_one_way_slip_rate_vectorized(
            cfg=self.cfg,
            portfolio_value=portfolio_value,
            adv_notional=np.asarray(adv_shares, dtype=float),
            trade_notional=np.asarray(trade_estimate_notionals, dtype=float),
        )

        q        = np.zeros(n_vars)
        q[:m]    = -expected_returns - 2.0 * impact * prev_w_arr
        q[m:2*m] = turnover_costs
        q[-1]    = self.cfg.SLACK_PENALTY

        builder = _ConstraintBuilder(n_vars)

        budget_row = sp.csc_matrix(
            (np.ones(m), (np.zeros(m, int), np.arange(m))), shape=(1, n_vars)
        )
        builder.add_constraint(budget_row, [l_gamma], [u_gamma])

        A_scen = sp.lil_matrix((T_cvar, n_vars))
        A_scen[:, :m]  = losses
        A_scen[:, 2*m] = -1.0
        for i in range(T_cvar):
            A_scen[i, 2*m + 1 + i] = -1.0
        builder.add_constraint(A_scen.tocsc(), [-np.inf] * T_cvar, [0.0] * T_cvar)

        scen_c = 1.0 / (T_cvar * (1.0 - self.cfg.CVAR_ALPHA))
        lim    = sp.lil_matrix((1, n_vars))
        lim[0, 2*m]                 = 1.0
        lim[0, 2*m+1:2*m+1+T_cvar] = scen_c
        lim[0, -1]                  = -1.0
        builder.add_constraint(lim.tocsc(), [-np.inf], [self.cfg.CVAR_DAILY_LIMIT])

        lb, ub = np.full(n_vars, -np.inf), np.full(n_vars, np.inf)
        lb[:m], ub[:m] = 0.0, adv_limit
        lb[m:2*m], lb[2*m+1:2*m+1+T_cvar], lb[-1] = 0.0, 0.0, 0.0
        builder.add_constraint(sp.eye(n_vars, format="csc"), lb.tolist(), ub.tolist())

        if sector_labels is not None:
            labels = np.asarray(sector_labels, dtype=int)
            for sec_id in np.unique(labels):
                if sec_id == -1:
                    continue
                mask = labels == sec_id
                sec_row = sp.lil_matrix((1, n_vars))
                sec_row[0, np.where(mask)[0]] = 1.0
                builder.add_constraint(sec_row.tocsc(), [0.0], [self.cfg.MAX_SECTOR_WEIGHT])

        tc = sp.lil_matrix((2 * m, n_vars))
        for i in range(m):
            tc[2 * i, i] = 1.0
            tc[2 * i, m + i] = -1.0
            tc[2 * i + 1, i] = -1.0
            tc[2 * i + 1, m + i] = -1.0

        tc_u = []
        for p in prev_w_arr:
            tc_u.extend([p, -p])
        builder.add_constraint(tc.tocsc(), [-np.inf] * (2 * m), tc_u)

        A, lower, upper = builder.build()
        P_upper = sp.triu(P, format="csc")

        return P_upper, q, A, lower, upper, T_cvar, gamma, l_gamma, u_gamma, adv_limit, Sigma_reg

    def _invoke_solver(
        self,
        P_upper: Any,
        q: np.ndarray,
        A: Any,
        lower: list,
        upper: list,
        m: int,
        prev_w: Optional[np.ndarray],
        portfolio_value: float,
        adv_shares: np.ndarray,
        T_cvar: int,
    ) -> Any:
        current_shape = (m, T_cvar)
        current_nnz   = (P_upper.nnz, A.nnz)

        with self._solver_lock:
            is_same_structure = False
            if (self._solver is not None
                    and self._solver_shape == current_shape
                    and self._solver_nnz == current_nnz
                    and self._solver_struct is not None):
                P_ind, P_ptr, A_ind, A_ptr = self._solver_struct
                is_same_structure = (
                    np.array_equal(P_upper.indices, P_ind)
                    and np.array_equal(P_upper.indptr, P_ptr)
                    and np.array_equal(A.indices, A_ind)
                    and np.array_equal(A.indptr, A_ptr)
                )

            if not is_same_structure:
                self._solver = osqp.OSQP()
                setup_kwargs = dict(
                    verbose=False,
                    eps_abs=1e-4,
                    eps_rel=1e-4,
                    adaptive_rho=True,
                    max_iter=50000,
                )
                try:
                    self._solver.setup(
                        P_upper, q, A, lower, upper,
                        polishing=True,
                        warm_starting=True,
                        **setup_kwargs,
                    )
                except TypeError as exc:
                    msg = str(exc)
                    if ("polishing" not in msg) and ("warm_starting" not in msg):
                        raise
                    self._solver.setup(
                        P_upper, q, A, lower, upper,
                        polish=True,
                        warm_start=True,
                        **setup_kwargs,
                    )
                self._solver_shape = current_shape
                self._solver_nnz = current_nnz
                self._solver_struct = (
                    P_upper.indices.copy(), P_upper.indptr.copy(),
                    A.indices.copy(), A.indptr.copy(),
                )
            else:
                assert self._solver is not None
                self._solver.update(
                    q=q, l=lower, u=upper,
                    Px=P_upper.data, Ax=A.data,
                )

            res = self._handle_solver_fallback(lambda: self._solver.solve(), "first-pass")
            
            # Turnover iteration
            w_opt = np.maximum(res.x[:m], 0.0)
            prev_w_arr = prev_w if prev_w is not None else np.zeros(m)
            actual_deltas = np.abs(w_opt - prev_w_arr) * float(portfolio_value)
            turnover_costs = _compute_one_way_slip_rate_vectorized(
                cfg=self.cfg,
                portfolio_value=portfolio_value,
                adv_notional=np.asarray(adv_shares, dtype=float),
                trade_notional=np.asarray(actual_deltas, dtype=float),
            )
            q[m:2*m] = turnover_costs

            assert self._solver is not None
            self._solver.update(q=q)
            self._solver.warm_start(x=res.x)

            res = self._handle_solver_fallback(lambda: self._solver.solve(), "second-pass")
        return res

    def _handle_solver_fallback(self, solve_func: callable, stage: str) -> Any:
        try:
            res = solve_func()
        except Exception as exc:
            logger.error(
                "[Optimizer] OSQP %s solve() raised an exception: %s — "
                "invalidating solver cache to force fresh setup on next call.", stage, exc
            )
            self._solver = None
            self._solver_shape = None
            self._solver_nnz = None
            self._solver_struct = None
            raise OptimizationError(
                f"OSQP {stage} solve() failed with exception: {exc}",
                OptimizationErrorType.NUMERICAL,
            ) from exc
        
        if res.info.status not in ("solved", "solved inaccurate", "solved_inaccurate"):
            self._solver = None
            self._solver_shape = None
            self._solver_nnz = None
            self._solver_struct = None
            raise OptimizationError(f"OSQP status: {res.info.status}", OptimizationErrorType.NUMERICAL)
        return res

    def _extract_optimization_results(
        self,
        res: Any,
        clean_rets: pd.DataFrame,
        m: int,
        T_cvar: int,
        gamma: float,
        l_gamma: float,
        u_gamma: float,
        adv_limit: np.ndarray,
        Sigma_reg: Any,
        ridge: float,
        kept_indices: np.ndarray,
        original_m: int,
    ) -> np.ndarray:
        if res.info.status in ("solved inaccurate", "solved_inaccurate"):
            logger.warning(
                "[Optimizer] OSQP returned '%s' — KKT conditions not strictly satisfied. "
                "Proceeding to physical CVaR verification.",
                res.info.status,
            )

        w_opt = np.maximum(res.x[:m], 0.0)

        simple_rets = np.expm1(clean_rets)
        losses     = -simple_rets.iloc[-T_cvar:].values
        portfolio_losses  = losses @ w_opt
        sorted_losses     = np.sort(portfolio_losses)
        tail_cutoff       = int(np.floor(T_cvar * (1.0 - self.cfg.CVAR_ALPHA)))
        tail_cutoff       = max(1, tail_cutoff)
        tail_losses       = sorted_losses[-tail_cutoff:]
        physical_cvar     = float(np.mean(tail_losses)) if tail_losses.size else 0.0

        eta           = res.x[2*m]
        z_vec         = res.x[2*m+1: 2*m+1+T_cvar]
        solver_cvar   = float(eta + np.sum(z_vec) / (T_cvar * (1.0 - self.cfg.CVAR_ALPHA)))
        slack_value   = float(res.x[-1])

        adv_binding_count = int(np.sum(w_opt >= adv_limit - 1e-6))

        self.last_diag = SolverDiagnostics(
            status            = res.info.status,
            gamma_intent      = gamma,
            actual_weight     = float(np.sum(w_opt)),
            l_gamma           = l_gamma,
            u_gamma           = u_gamma,
            cvar_value        = physical_cvar,
            slack_value       = slack_value,
            sum_adv_limit     = float(np.sum(adv_limit)),
            adv_binding_count = adv_binding_count,
            ridge_applied     = ridge,
            cond_number       = float(np.linalg.cond(Sigma_reg)),
            t_cvar            = T_cvar,
        )

        POST_SOLVE_TOL = 1e-4

        if physical_cvar > self.cfg.CVAR_DAILY_LIMIT + POST_SOLVE_TOL:
            raise OptimizationError(
                f"Physical CVaR {physical_cvar:.4%} exceeds hard limit "
                f"{self.cfg.CVAR_DAILY_LIMIT:.4%} (solver reported {solver_cvar:.4%}, "
                f"slack={slack_value:.6f}). Refusing to deploy.",
                OptimizationErrorType.NUMERICAL,
            )

        lower_hard = float(np.min(w_opt)) < -POST_SOLVE_TOL
        upper_hard = bool(np.any(w_opt > (adv_limit + POST_SOLVE_TOL)))
        gross = float(np.sum(w_opt))
        gross_low_hard = gross < (l_gamma - POST_SOLVE_TOL)
        gross_high_hard = gross > (u_gamma + POST_SOLVE_TOL)

        near_tol = 1e-7
        if float(np.min(w_opt)) < near_tol:
            logger.warning("[Optimizer] Post-check near lower bound: min(w)=%.9f", float(np.min(w_opt)))
        if bool(np.any(w_opt > (adv_limit - near_tol))):
            logger.warning("[Optimizer] Post-check near ADV bound for one or more names.")
        if abs(gross - l_gamma) < near_tol or abs(gross - u_gamma) < near_tol:
            logger.warning(
                "[Optimizer] Post-check near gross boundary: gross=%.9f l=%.9f u=%.9f",
                gross,
                float(l_gamma),
                float(u_gamma),
            )

        if lower_hard or upper_hard or gross_low_hard or gross_high_hard:
            raise OptimizationError(
                "Post-solve constraint verification failed: "
                f"min_w={float(np.min(w_opt)):.9g}, "
                f"max_excess={float(np.max(w_opt - adv_limit)):.9g}, "
                f"gross={gross:.9g}, "
                f"bounds=[{float(l_gamma):.9g}, {float(u_gamma):.9g}]",
                OptimizationErrorType.NUMERICAL,
            )

        full_w_opt = np.zeros(original_m)
        full_w_opt[kept_indices] = np.round(w_opt, 10)
        return full_w_opt

    def optimize(
        self,
        expected_returns:    np.ndarray,
        historical_returns:  pd.DataFrame,
        adv_shares:          np.ndarray,
        prices:              np.ndarray,
        portfolio_value:     float,
        prev_w:              Optional[np.ndarray] = None,
        exposure_multiplier: float                = 1.0,
        sector_labels:       Optional[np.ndarray] = None,
        execution_date:      Optional[pd.Timestamp] = None,
    ) -> np.ndarray:
        original_m = len(expected_returns)
        if original_m == 0:
            return np.array([])

        clean_rets, expected_returns_sub, prices_sub, adv_shares_sub, prev_w_sub, sector_labels_sub, kept_indices, m, T = self._preprocess_optimization_inputs(
            expected_returns, historical_returns, adv_shares, prices, portfolio_value,
            prev_w, exposure_multiplier, sector_labels, execution_date
        )

        P_upper, q, A, lower, upper, T_cvar, gamma, l_gamma, u_gamma, adv_limit, Sigma_reg = self._build_optimization_constraints(
            clean_rets, expected_returns_sub, adv_shares_sub, portfolio_value, prev_w_sub,
            exposure_multiplier, sector_labels_sub, m, T
        )

        res = self._invoke_solver(
            P_upper, q, A, lower, upper, m, prev_w_sub, portfolio_value, adv_shares_sub, T_cvar
        )

        full_w_opt = self._extract_optimization_results(
            res, clean_rets, m, T_cvar, gamma, l_gamma, u_gamma, adv_limit,
            Sigma_reg, 0.0, kept_indices, original_m
        )

        return full_w_opt
