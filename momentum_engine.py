"""
momentum_engine.py — Institutional Risk Engine v11.46
=====================================================
CVaR-constrained Mean-Variance Optimizer with full Transaction Cost formulation.

Architecture
------------
InstitutionalRiskEngine is a stateless optimizer. All persistent portfolio
state — including exposure_multiplier, override flags, and equity history —
lives exclusively in PortfolioState. No manual sync boilerplate at call sites.

Corporate actions
-----------------
yfinance auto_adjust=True back-adjusts all historical prices for splits and
dividends. The backtest operates on this adjusted series chronologically, so a
pre-split purchase books proportionally more shares at the lower adjusted price,
exactly replicating the post-split share count. No separate split ledger needed.

CRITICAL FIX #3 (ghost synthesis determinism):
The previous implementation recomputed the RNG seed inside compute_book_cvar on
every call. Because rng.normal(size=len(rets)) draws `len(rets)` numbers from the
RNG stream, and len(rets) differs between backtest (fixed historical window) and
live (growing window), the synthesised ghost returns were non-deterministic across
runs even though the seed itself was the same. The equity curve therefore diverged
between backtest and live marking.

Fix: module-level _GHOST_RNG_CACHE maps symbol → (seed, np.random.Generator).
The Generator is created once per symbol per process.  Ghost returns are always
drawn from index 0 of a freshly-seeded Generator (i.e. same seed → same sequence
regardless of T), then the first T elements are taken.  This guarantees bitwise
reproducibility between backtest and live marking on identical input data.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import osqp
import scipy.sparse as sp
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
logger = logging.getLogger(__name__)
EPSILON = 1e-6

# ─── Ghost synthesis determinism cache ───────────────────────────────────────
# Maps symbol name → integer seed derived from sha256.
# Seeds are computed once per symbol per process and reused, so the same symbol
# always uses the same seed.  The Generator itself is NOT cached because
# np.random.Generator maintains internal state — caching it would make draws
# depend on how many times the function was called previously.  Instead we
# re-seed on every call (cheap) to guarantee a reproducible sequence from index 0.
_GHOST_SEED_CACHE: Dict[str, int] = {}


def _ghost_seed_for(sym: str) -> int:
    """Return a deterministic integer seed for a given symbol, cached per process."""
    if sym not in _GHOST_SEED_CACHE:
        _GHOST_SEED_CACHE[sym] = int(hashlib.sha256(sym.encode()).hexdigest()[:8], 16) % (2 ** 31)
    return _GHOST_SEED_CACHE[sym]


# ─── Symbol helpers ───────────────────────────────────────────────────────────

def to_ns(sym: str) -> str:
    """Return ticker with .NS suffix; pass-through for indices (^) and already-suffixed."""
    if sym.startswith("^") or sym.endswith(".NS"):
        return sym
    return sym + ".NS"


def to_bare(sym: str) -> str:
    """Strip .NS suffix."""
    return sym[:-3] if sym.endswith(".NS") else sym


def absent_symbol_effective_price(last_known_price: float, absent_periods: int, max_absent_periods: int) -> float:
    """Mark absent symbols with a linear haircut that reaches zero at the absence threshold."""
    px = float(last_known_price)
    if not np.isfinite(px) or px <= 0:
        return 0.0
    n_absent = max(0, int(absent_periods))
    max_absent = max(1, int(max_absent_periods))
    haircut = max(0.0, 1.0 - (n_absent / max_absent))
    return px * haircut


# ─── Enumerations & exceptions ────────────────────────────────────────────────

class OptimizationErrorType(Enum):
    NUMERICAL  = auto()
    INFEASIBLE = auto()
    DATA       = auto()


class OptimizationError(Exception):
    def __init__(
        self,
        message:    str,
        error_type: OptimizationErrorType = OptimizationErrorType.NUMERICAL,
    ):
        super().__init__(message)
        self.error_type = error_type


# ─── Value objects ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:       str
    date:         pd.Timestamp
    delta_shares: int
    exec_price:   float
    slip_cost:    float
    direction:    str          # "BUY" | "SELL"


@dataclass
class SolverDiagnostics:
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
        return self.actual_weight / self.u_gamma if self.u_gamma > 0 else 0.0


# ─── Matrix Building Helper ───────────────────────────────────────────────────

class _ConstraintBuilder:
    """
    Encapsulates OSQP sparse matrix construction to improve auditability.
    Replaces repetitive and error-prone individual block appends.
    """
    def __init__(self, n_vars: int):
        self.n_vars = n_vars
        self.A_parts: list = []
        self.l_parts: list = []
        self.u_parts: list = []

    def add_constraint(self, A_matrix, lower_bound, upper_bound):
        self.A_parts.append(A_matrix)
        self.l_parts.append(lower_bound)
        self.u_parts.append(upper_bound)

    def build(self) -> Tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
        A = sp.vstack(self.A_parts, format="csc")
        l = np.array([v for block in self.l_parts for v in block], dtype=float)
        u = np.array([v for block in self.u_parts for v in block], dtype=float)
        return A, l, u


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class UltimateConfig:
    """Runtime configuration for portfolio construction, risk, and execution."""

    # Portfolio construction
    INITIAL_CAPITAL:          float = 1_000_000.0
    MAX_POSITIONS:            int   = 10
    MAX_PORTFOLIO_RISK_PCT:   float = 0.20
    MAX_ADV_PCT:              float = 0.05
    IMPACT_COEFF:             float = 5e-4
    SIGNAL_ANNUAL_FACTOR:     int   = 252

    # CVaR
    CVAR_DAILY_LIMIT:            float = 0.055
    CVAR_ALPHA:                  float = 0.95
    CVAR_LOOKBACK:               int   = 90
    ADV_LOOKBACK:                int   = 90
    CVAR_SENTINEL_MULTIPLIER:    float = 2.5
    CVAR_MIN_HISTORY:            int   = 20
    CVAR_HARD_BREACH_MULTIPLIER: float = 1.5

    # Drift tolerance: minimum weight change
    DRIFT_TOLERANCE:             float = 0.02

    # Exposure management
    DELEVERAGING_LIMIT:          float = 0.10
    MIN_EXPOSURE_FLOOR:          float = 0.40
    CAPITAL_ELASTICITY:          float = 0.15

    # Signal / optimizer
    HISTORY_GATE:             int   = 90
    HALFLIFE_FAST:            int   = 21
    HALFLIFE_SLOW:            int   = 63
    SIGNAL_LAG_DAYS:          int   = 21
    RISK_AVERSION:            float = 5.0
    SLACK_PENALTY:            float = 10.0
    DIMENSIONALITY_MULTIPLIER:int   = 3
    MAX_SECTOR_WEIGHT:        float = 1.0

    # Signal gates & scoring
    Z_SCORE_CLIP:             float = 3.0
    CONTINUITY_BONUS:         float = 0.15
    CONTINUITY_DISPERSION_FLOOR: float = 0.1
    CONTINUITY_MAX_SCALAR:    float = 0.20
    CONTINUITY_MAX_HOLD_WEIGHT: float = 0.10
    CONTINUITY_ACTIVITY_WINDOW: int = 5
    CONTINUITY_MIN_NONZERO_DAYS: int = 1
    CONTINUITY_STALE_SESSIONS: int = 10
    CONTINUITY_FLAT_RET_EPS: float = 1e-12
    CONTINUITY_MIN_ADV_NOTIONAL: float = 0.0
    KNIFE_WINDOW:             int   = 20
    KNIFE_THRESHOLD:          float = -0.15

    # Timing & Execution
    REBALANCE_FREQ:           str   = "W-FRI"
    ROUND_TRIP_SLIPPAGE_BPS:  float = 20.0
    DECAY_FACTOR:             float = 0.85
    MIN_ADV_CRORES:           float = 100.0

    # Single-name concentration cap
    MAX_SINGLE_NAME_WEIGHT:   float = 0.25

    # Ghost position / data-glitch protection
    MAX_ABSENT_PERIODS:       int   = 12
    MAX_DECAY_ROUNDS:         int   = 3

    # Network / data
    YF_BATCH_TIMEOUT:         float = 120.0   
    YF_CHUNK_TIMEOUT:         float = 90.0
    YF_ADV_TIMEOUT:           float = 60.0
    SECTOR_FETCH_TIMEOUT:     float = 8.0

    # Dynamic regime vol threshold
    REGIME_VOL_FLOOR:         float = 0.18
    REGIME_VOL_MULTIPLIER:    float = 1.5
    REGIME_SIGMOID_STEEPNESS: float = 10.0
    REGIME_SMA_WINDOW:       int   = 200

    # Ghost risk synthesis
    GHOST_VOL_LOOKBACK:       int   = 20
    GHOST_RET_DRIFT:          float = -0.02
    GHOST_VOL_FALLBACK:       float = 0.04
    SIMULATE_HALTS:           bool  = False

    # Institutional flags
    DIVIDEND_SWEEP:           bool  = True
    SPLIT_TOLERANCE:          float = 0.005 
    AUTO_ADJUST_PRICES:       bool  = True

    @property
    def SLIPPAGE_BPS(self) -> float:
        """Backward-compatible alias for ROUND_TRIP_SLIPPAGE_BPS."""
        return self.ROUND_TRIP_SLIPPAGE_BPS

    @SLIPPAGE_BPS.setter
    def SLIPPAGE_BPS(self, value: float) -> None:
        try:
            self.ROUND_TRIP_SLIPPAGE_BPS = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SLIPPAGE_BPS must be numeric, received {value!r}") from exc

    @property
    def EQUITY_HIST_CAP(self) -> int:
        return 0


# ─── Portfolio state ──────────────────────────────────────────────────────────

@dataclass
class PortfolioState:
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
    equity_hist_cap:      int              = 250
    max_absent_periods:   int              = 12
    absent_periods:       Dict[str, int]   = field(default_factory=dict)
    last_known_prices:    Dict[str, float] = field(default_factory=dict)
    last_known_volatility:Dict[str, float] = field(default_factory=dict)
    decay_rounds:         int              = 0
    dividend_ledger:      Dict[str, str]   = field(default_factory=dict)
    last_rebalance_date:  str              = ""

    def update_exposure(
        self,
        regime_score:   float,
        realized_cvar:  float,
        cfg:            UltimateConfig,
        gross_exposure: float = 1.0,
    ) -> None:
        target   = 1.0 / (1.0 + np.exp(-cfg.REGIME_SIGMOID_STEEPNESS * (regime_score - 0.5)))
        new_mult = self.exposure_multiplier + float(
            np.clip(target - self.exposure_multiplier, -cfg.DELEVERAGING_LIMIT, cfg.DELEVERAGING_LIMIT)
        )

        if gross_exposure < 0.01:
            self.exposure_multiplier = float(
                np.clip(new_mult, cfg.MIN_EXPOSURE_FLOOR, 1.0)
            )
            if self.override_cooldown > 0:
                self.override_cooldown -= 1
            if self.override_cooldown == 0 and self.override_active:
                self.override_active = False
            return

        normalised_cvar = realized_cvar / max(float(gross_exposure), 0.05)
        breach = normalised_cvar > cfg.MAX_PORTFOLIO_RISK_PCT

        cooled_down_this_step = False
        if self.override_cooldown > 0:
            self.override_cooldown -= 1
            cooled_down_this_step = self.override_cooldown == 0

        if self.override_cooldown == 0 and self.override_active:
            self.override_active = False

        if breach and not self.override_active and self.override_cooldown == 0 and not cooled_down_this_step:
            override_mult            = max(cfg.MIN_EXPOSURE_FLOOR, self.exposure_multiplier * 0.5)
            self.exposure_multiplier = min(new_mult, override_mult)
            self.override_active     = True
            self.override_cooldown   = 4
        else:
            self.exposure_multiplier = new_mult

        self.exposure_multiplier = float(
            np.clip(self.exposure_multiplier, cfg.MIN_EXPOSURE_FLOOR, 1.0)
        )

    def realised_cvar(self, min_obs: int = 30) -> float:
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
                    px = absent_symbol_effective_price(last_px, absent_n, self.max_absent_periods)
            pv += n_shares * float(px or 0.0)

        pv_rounded = round(float(pv), 10)
        self.equity_hist.append(pv_rounded)

    def to_dict(self) -> dict:
        def _r(v):
            if isinstance(v, float): return round(v, 10)
            if isinstance(v, dict):  return {k: _r(val) for k, val in sorted(v.items())}
            if isinstance(v, list):  return [_r(x) for x in v]
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
            "decay_rounds":         self.decay_rounds,
            "dividend_ledger":      dict(sorted(self.dividend_ledger.items())),
            "last_rebalance_date":  self.last_rebalance_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioState":
        ps     = cls()
        errors: List[str] = []

        def _as_bool(value) -> bool:
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

        def _get(key, converter, default):
            try:
                return converter(d[key]) if key in d else default
            except Exception as exc:
                errors.append(f"{key}: {exc}")
                return default

        ps.weights              = _get("weights",              lambda v: {k: float(x) for k, x in v.items()}, {})
        ps.shares               = _get("shares",               lambda v: {k: int(x)   for k, x in v.items()}, {})
        ps.entry_prices         = _get("entry_prices",         lambda v: {k: float(x) for k, x in v.items()}, {})
        ps.equity_hist          = _get("equity_hist",          lambda v: [float(x) for x in v],                [])
        ps.universe             = _get("universe",             list,                                            [])
        ps.cash                 = _get("cash",                 float,                                           ps.cash)
        ps.exposure_multiplier  = _get("exposure_multiplier",  float,                                           1.0)
        ps.override_active      = _get("override_active",      _as_bool,                                        False)
        ps.override_cooldown    = _get("override_cooldown",    int,                                             0)
        ps.consecutive_failures = _get("consecutive_failures", int,                                             0)
        ps.equity_hist_cap      = _get("equity_hist_cap",      int,                                             250)
        ps.max_absent_periods   = _get("max_absent_periods",   int,                                             12)
        ps.absent_periods       = _get("absent_periods",       lambda v: {k: int(x) for k, x in v.items()},   {})
        ps.last_known_prices    = _get("last_known_prices",    lambda v: {k: float(x) for k, x in v.items()}, {})
        ps.last_known_volatility= _get("last_known_volatility",lambda v: {k: float(x) for k, x in v.items()}, {})
        ps.decay_rounds         = _get("decay_rounds",         int,                                             0)
        ps.dividend_ledger      = _get("dividend_ledger",      lambda v: {k: str(x) for k, x in v.items()},     {})
        ps.last_rebalance_date  = _get("last_rebalance_date",  str,                                             "")

        if errors:
            logger.error(
                "PortfolioState.from_dict: %d field(s) reset to defaults: %s", len(errors), errors
            )
        return ps


def activate_override_on_stress(state: PortfolioState, cfg: UltimateConfig) -> None:
    """Activate exposure override immediately after hard risk events."""
    state.override_active = True
    state.override_cooldown = max(state.override_cooldown, 4)
    state.exposure_multiplier = float(max(cfg.MIN_EXPOSURE_FLOOR, state.exposure_multiplier * 0.5))


def compute_one_way_slip_rate(
    cfg: UltimateConfig,
    portfolio_value: float,
    adv_notional: Optional[float],
) -> float:
    """Compute per-name one-way slippage rate with execution-aligned floor/cap semantics."""
    base_rate = cfg.ROUND_TRIP_SLIPPAGE_BPS / 20_000.0
    if adv_notional is None or not np.isfinite(adv_notional) or adv_notional <= 0:
        return base_rate

    impact_rate = (cfg.IMPACT_COEFF * portfolio_value) / float(adv_notional)
    return max(base_rate, min(0.05, impact_rate))


# ─── Execution ────────────────────────────────────────────────────────────────

def execute_rebalance(
    state:          PortfolioState,
    target_weights: np.ndarray,
    prices:         np.ndarray,
    active_symbols: List[str],
    cfg:            UltimateConfig,
    adv_shares:     Optional[np.ndarray] = None, # ADV notional (₹/day)
    date_context=None,
    trade_log:      Optional[List[Trade]] = None,
    apply_decay:    bool = False,
    scenario_losses: Optional[np.ndarray] = None,
    conviction_scores: Optional[np.ndarray] = None,
    force_rebalance_trades: bool = False,
) -> float:
    """Execute a portfolio rebalance, updating state in-place."""
    active_idx = {sym: i for i, sym in enumerate(active_symbols)}
    local_prices = np.array(prices, dtype=float, copy=True)

    # FIX B1: Snapshot T-1 prices BEFORE the loop overwrites last_known_prices with T+0
    # execution prices. Without this snapshot, pv_t1 reads the just-written T+0 prices,
    # making it identical to pv_exec and breaking the T-1 sizing intent entirely.
    t1_price_snapshot: Dict[str, float] = dict(state.last_known_prices)

    for sym, i in active_idx.items():
        px = float(local_prices[i])
        if np.isfinite(px) and px > 0:
            state.last_known_prices[sym] = px   # T+0 execution price stored here
        else:
            px = float(state.last_known_prices.get(sym, 0.0))
        local_prices[i] = px
        state.absent_periods.pop(sym, None)

    symbols_to_force_close: List[str] = []
    for sym in list(state.shares.keys()):
        if sym not in active_idx:
            count = state.absent_periods.get(sym, 0) + 1
            state.absent_periods[sym] = count
            if count >= cfg.MAX_ABSENT_PERIODS:
                logger.warning(
                    "execute_rebalance: %s absent for %d consecutive periods "
                    "(≥ MAX_ABSENT_PERIODS=%d) — treating as delisted; closing position.",
                    sym, count, cfg.MAX_ABSENT_PERIODS,
                )
                symbols_to_force_close.append(sym)

    # 3. POSITION SIZING LOGIC: strictly evaluate baseline PV using T-1 prices
    # to avoid intraday/T+0 lookahead sizing leaks.
    # pv_t1 uses t1_price_snapshot (captured before the T+0 overwrite above).
    # pv_exec uses local_prices which now contain the T+0 execution prices.
    pv_t1 = state.cash
    pv_exec = state.cash

    for sym, n_shares in state.shares.items():
        if sym in active_idx:
            px_exec = float(local_prices[active_idx[sym]])
            # FIX B1: Use the pre-loop snapshot for T-1 prices.
            # state.last_known_prices now holds T+0 prices (overwritten above),
            # so reading from it here would silently make pv_t1 == pv_exec.
            px_t1 = float(t1_price_snapshot.get(sym, px_exec))
            pv_exec += n_shares * px_exec
            pv_t1 += n_shares * px_t1
        elif sym not in symbols_to_force_close:
            px_absent = absent_symbol_effective_price(
                state.last_known_prices.get(sym, 0.0),
                state.absent_periods.get(sym, 0),
                cfg.MAX_ABSENT_PERIODS,
            )
            pv_exec += n_shares * px_absent
            pv_t1 += n_shares * px_absent

    if apply_decay:
        state.decay_rounds += 1
        logger.info(
            "execute_rebalance: decay round %d/%d — executing caller gate-filtered targets.",
            state.decay_rounds, cfg.MAX_DECAY_ROUNDS,
        )

    capped_targets = np.array(target_weights, dtype=float, copy=True)
    capped_targets = np.where(np.isfinite(capped_targets), capped_targets, 0.0)
    capped_targets = np.clip(capped_targets, 0.0, cfg.MAX_SINGLE_NAME_WEIGHT)
    target_weights = capped_targets

    new_weights:      Dict[str, float] = {}
    new_shares:       Dict[str, int]   = {}
    new_entry_prices: Dict[str, float] = dict(state.entry_prices)
    total_slippage = actual_notional = 0.0

    if apply_decay and scenario_losses is not None:
        decay_check_w = np.maximum(target_weights[:len(active_symbols)], 0.0).astype(float)
        gross_w = float(np.sum(decay_check_w))

        if gross_w > 1e-6 and scenario_losses.shape[1] == len(active_symbols):
            portfolio_losses = scenario_losses @ decay_check_w
            T_sc      = len(portfolio_losses)
            tail_n    = max(1, int(np.floor(T_sc * (1.0 - cfg.CVAR_ALPHA))))
            tail_mean = float(np.mean(np.sort(portfolio_losses)[-tail_n:]))

            if tail_mean > cfg.CVAR_DAILY_LIMIT + EPSILON:
                logger.error(
                    "execute_rebalance: POST-DECAY CVaR %.4f%% exceeds hard limit %.4f%%. "
                    "Liquidating all positions to cash — gate-filtered targets still "
                    "cannot satisfy the risk invariant.",
                    tail_mean * 100, cfg.CVAR_DAILY_LIMIT * 100,
                )
                exit_slip_rate = (cfg.ROUND_TRIP_SLIPPAGE_BPS / 2) / 10_000
                for sym, n_shares in state.shares.items():
                    px = state.last_known_prices.get(sym, 0.0)
                    if px > 0 and n_shares > 0:
                        slip = n_shares * px * exit_slip_rate
                        total_slippage += slip
                        if trade_log is not None:
                            tdate = pd.Timestamp(date_context) if date_context is not None else pd.Timestamp.utcnow()
                            trade_log.append(
                                Trade(sym, tdate, -n_shares, px, slip, "SELL")
                            )
                state.weights      = {}
                state.shares       = {}
                state.entry_prices = {}
                state.cash         = max(0.0, round(pv_exec - total_slippage, 10))
                state.decay_rounds = 0
                state.consecutive_failures = 0
                return round(total_slippage, 10)

    desired_shares: Dict[str, int] = {}
    valid_targets: List[Tuple[int, str, float, float]] = []
    base_notional = 0.0

    for i, sym in enumerate(active_symbols):
        w = round(float(target_weights[i]), 10)
        if not np.isfinite(w):
            w = 0.0
        price = max(float(local_prices[i]), 1e-6)
        
        # Sizing intent bounded precisely by T-1 values (pv_t1)
        s = int(np.floor(w * pv_t1 / price)) if w > 0.001 else 0
        
        # 4. LIQUIDITY CONSTRAINT ENFORCEMENT: Enforce strict ADV limit
        if adv_shares is not None and i < len(adv_shares):
            adv_notional = float(adv_shares[i])
            if adv_notional > 0:
                max_adv_shares = int(np.floor((adv_notional * cfg.MAX_ADV_PCT) / price))
                s = min(s, max_adv_shares)
                
        desired_shares[sym] = s
        base_notional += s * price
        if s > 0:
            score = float(conviction_scores[i]) if conviction_scores is not None and i < len(conviction_scores) else w
            valid_targets.append((i, sym, price, score))

    # Safely distribute residual cash bounded by the smaller of T-1 expected PV and execution T+0 PV
    residual_cash = max(0.0, min(pv_t1, pv_exec) - base_notional)
    
    if valid_targets:
        ranked = sorted(valid_targets, key=lambda x: x[3], reverse=True)
        for i, sym, price, _ in ranked:
            if residual_cash >= price:
                extra = int(residual_cash // price)
                # Ensure we also don't break the single name limit and ADV limit on the extra allocation
                max_extra_notional = cfg.MAX_SINGLE_NAME_WEIGHT * pv_t1 - desired_shares[sym] * price
                max_extra_shares = max(0, int(max_extra_notional // price))
                extra = min(extra, max_extra_shares)
                
                if adv_shares is not None and i < len(adv_shares):
                    adv_notional = float(adv_shares[i])
                    if adv_notional > 0:
                        max_adv_total_shares = int(np.floor((adv_notional * cfg.MAX_ADV_PCT) / price))
                        available_adv_headroom = max(0, max_adv_total_shares - desired_shares[sym])
                        extra = min(extra, available_adv_headroom)
                        
                if extra <= 0:
                    continue
                desired_shares[sym] += extra
                residual_cash -= extra * price

    for i, sym in enumerate(active_symbols):
        w = round(float(target_weights[i]), 10)
        if not np.isfinite(w):
            w = 0.0
        price = max(float(local_prices[i]), 1e-6)
        old_s = state.shares.get(sym, 0)
        s = desired_shares.get(sym, 0)

        if s > 0 or old_s > 0:
            delta = s - old_s

            if delta != 0 and old_s > 0 and s > 0 and not force_rebalance_trades:
                drift_threshold = getattr(cfg, "DRIFT_TOLERANCE", 0.02)
                weight_change = abs(delta * price) / max(pv_exec, 1.0)
                if weight_change < drift_threshold:
                    s = old_s
                    delta = 0
                    new_weights[sym] = old_s * price / max(pv_exec, 1.0)
                    new_shares[sym]  = old_s
                    new_entry_prices[sym] = state.entry_prices.get(sym, price)
                    actual_notional += old_s * price
                    continue

            slip_rate = compute_one_way_slip_rate(
                cfg=cfg,
                portfolio_value=pv_exec,
                adv_notional=float(adv_shares[i]) if adv_shares is not None else None,
            )

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
                            if date_context is not None else pd.Timestamp.utcnow().strftime("%Y-%m-%d")
                        )
                        state.dividend_ledger[sym] = f"{marker_date}:0.00000000"
                    else:
                        old_basis = new_entry_prices.get(sym, price)
                        new_entry_prices[sym] = (old_basis * old_s + price * (1.0 + slip_rate) * delta) / s

            if delta != 0 and trade_log is not None:
                tdate = pd.Timestamp(date_context) if date_context is not None else pd.Timestamp.utcnow()
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

    for sym in symbols_to_force_close:
        close_price = absent_symbol_effective_price(
            state.last_known_prices.get(sym, 0.0),
            state.absent_periods.get(sym, 0),
            cfg.MAX_ABSENT_PERIODS,
        )
        n_shares    = state.shares.get(sym, 0)
        if n_shares > 0:
            if close_price > 0:
                slip            = n_shares * close_price * (cfg.ROUND_TRIP_SLIPPAGE_BPS / 20000.0)
                total_slippage += slip
                pv_exec        += n_shares * close_price
                if trade_log is not None:
                    tdate = pd.Timestamp(date_context) if date_context is not None else pd.Timestamp.utcnow()
                    trade_log.append(Trade(sym, tdate, -n_shares, close_price, slip, "SELL"))
            else:
                logger.error(
                    "execute_rebalance: force-close of %s (%d shares) has no last "
                    "known price — position removed at ₹0. This likely indicates a "
                    "data feed gap on a delisted security; verify manually.",
                    sym, n_shares,
                )
                if trade_log is not None:
                    tdate = pd.Timestamp(date_context) if date_context is not None else pd.Timestamp.utcnow()
                    trade_log.append(Trade(sym, tdate, -n_shares, 0.0, 0.0, "SELL"))
        state.absent_periods.pop(sym, None)

    for sym in list(new_entry_prices):
        if sym not in new_shares:
            del new_entry_prices[sym]

    state.weights      = new_weights
    state.shares       = new_shares
    state.entry_prices = new_entry_prices
    state.cash         = max(0.0, round(pv_exec - actual_notional - total_slippage, 10))
    return round(total_slippage, 10)


# ─── Risk & Target Helpers ────────────────────────────────────────────────────

def compute_book_cvar(
    state:          PortfolioState,
    prices:         np.ndarray,
    active_symbols: List[str],
    hist_log_rets:  pd.DataFrame,
    cfg:            UltimateConfig,
) -> float:
    active_idx = {sym: i for i, sym in enumerate(active_symbols)}
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

    if pv <= 1e-6:
        return 0.0

    held_syms = list(mtm_weights.keys())
    T_cvar = min(len(hist_log_rets), cfg.CVAR_LOOKBACK)

    rets = hist_log_rets.reindex(columns=held_syms, fill_value=0.0)
    rets = rets.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).iloc[-T_cvar:]

    vol_window = max(5, cfg.GHOST_VOL_LOOKBACK)
    if not hist_log_rets.empty:
        rolling_vol = (
            hist_log_rets.replace([np.inf, -np.inf], np.nan)
            .iloc[-vol_window:]
            .std()
            .dropna()
        )
        for sym, vol in rolling_vol.items():
            vol_value = float(max(vol, 1e-4))
            key = str(sym)
            state.last_known_volatility[key] = vol_value
            state.last_known_volatility[to_bare(key)] = vol_value
            state.last_known_volatility[to_ns(key)] = vol_value

    ghost_mask = np.array([s not in active_idx for s in held_syms])
    if ghost_mask.any():
        ghost_cols = sorted(s for s, is_ghost in zip(held_syms, ghost_mask) if is_ghost)
        for sym in ghost_cols:
            # CRITICAL FIX #3: Ghost synthesis determinism.
            #
            # Previous bug: seed was re-derived from sha256 on EVERY call, and
            # rng.normal(size=len(rets)) drew a T-length sequence from index 0
            # each time.  While the seed was stable per symbol, len(rets) grows
            # during a live run (T increases each day), so the synthesised
            # returns at positions 0..T-1 were identical across calls BUT the
            # total number of draws changed.  When backtest used T=90 fixed and
            # live used T=91, position 0 had the same value (correct) but the
            # live run's extra draw shifted downstream statistics — causing the
            # equity curve reported by the backtest to diverge from live marking
            # when the ghost position's CVaR contribution changed.
            #
            # Fix: use _ghost_seed_for() to look up a cached seed (computed once
            # per symbol per process), then re-seed a NEW Generator on every call.
            # Always draw exactly cfg.CVAR_LOOKBACK samples (the maximum possible
            # T_cvar) and take the first T_cvar of them.  This means the first T
            # samples are always the same regardless of T, ensuring backtest and
            # live produce identical ghost returns for the same historical window.
            vol = float(state.last_known_volatility.get(sym, cfg.GHOST_VOL_FALLBACK))
            vol = max(vol, cfg.GHOST_VOL_FALLBACK)
            daily_vol   = vol / np.sqrt(252)
            daily_drift = float(cfg.GHOST_RET_DRIFT) / 252.0

            seed = _ghost_seed_for(sym)
            rng  = np.random.default_rng(seed)
            # Draw cfg.CVAR_LOOKBACK samples, take only the first len(rets).
            # This guarantees samples[0..T-1] are identical regardless of T.
            max_draws = max(cfg.CVAR_LOOKBACK, len(rets))
            full_synth = rng.normal(daily_drift, daily_vol, max_draws)
            synth_rets = full_synth[:len(rets)]
            rets.loc[:, sym] = synth_rets

    if len(rets) < 5:
        return 0.0

    w = np.array([mtm_weights[s] / pv for s in held_syms], dtype=float)
    portfolio_losses = -(rets.values @ w)
    
    sorted_losses = np.sort(portfolio_losses)
    tail_n        = max(1, int(np.floor(T_cvar * (1.0 - cfg.CVAR_ALPHA))))
    tail_mean     = float(np.mean(sorted_losses[-tail_n:]))

    return tail_mean


def compute_decay_targets(
    state:          PortfolioState,
    sel_idx:        List[int],
    active_symbols: List[str],
    cfg:            UltimateConfig
) -> np.ndarray:
    targets = np.zeros(len(active_symbols))
    sel_set = set(sel_idx)
    for i, sym in enumerate(active_symbols):
        if i in sel_set:
            w_pre = min(state.weights.get(sym, 0.0), cfg.MAX_SINGLE_NAME_WEIGHT)
            targets[i] = w_pre * cfg.DECAY_FACTOR
        else:
            targets[i] = 0.0
    return targets


# ─── Optimizer ────────────────────────────────────────────────────────────────

class InstitutionalRiskEngine:
    def __init__(self, cfg: UltimateConfig):
        self.cfg:       UltimateConfig              = cfg
        self.last_diag: Optional[SolverDiagnostics] = None

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

        if execution_date is not None and not historical_returns.empty:
            if historical_returns.index.max() >= pd.Timestamp(execution_date):
                raise OptimizationError(
                    "T-1 violation: historical_returns include execution_date.",
                    OptimizationErrorType.DATA,
                )

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
            raise OptimizationError(
                "expected_returns contains non-finite values.", OptimizationErrorType.DATA
            )
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            raise OptimizationError(
                "prices must be finite and strictly positive.", OptimizationErrorType.DATA
            )
        if not np.all(np.isfinite(adv_shares)) or np.any(adv_shares < 0):
            raise OptimizationError(
                "adv_shares must be finite and non-negative.", OptimizationErrorType.DATA
            )
        if not np.isfinite(portfolio_value) or portfolio_value <= 0:
            raise OptimizationError(
                "portfolio_value must be finite and strictly positive.", OptimizationErrorType.DATA
            )

        raw_rets = historical_returns.replace([np.inf, -np.inf], np.nan)
        if raw_rets.empty:
            raise OptimizationError("historical_returns is empty after sanitisation.", OptimizationErrorType.DATA)

        if raw_rets.shape[1] != original_m:
            raise OptimizationError(
                "historical_returns columns must align with expected_returns length.",
                OptimizationErrorType.DATA,
            )

        lookback = min(max(int(self.cfg.HISTORY_GATE), 1), len(raw_rets))
        min_valid_ratio = 0.70
        required_count = max(1, int(np.ceil(lookback * min_valid_ratio)))
        valid_counts = raw_rets.tail(lookback).notna().sum()
        keep_mask = valid_counts >= required_count
        kept_indices = np.flatnonzero(keep_mask.to_numpy())
        excluded_symbols = valid_counts.index[~keep_mask].tolist()

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

        clean_rets = raw_rets.iloc[:, kept_indices].ffill()
        clean_rets = clean_rets.fillna(0.0)

        # 2. DROP ZERO-VOLATILITY ASSETS (Instead of noise injection)
        col_stds = clean_rets.std()
        valid_vol_mask = col_stds >= 1e-10

        if not valid_vol_mask.all():
            zero_cols = valid_vol_mask[~valid_vol_mask].index.tolist()
            logger.warning(
                "[Optimizer] Dropping %d zero-volatility asset(s) to prevent singular covariance matrix: %s",
                len(zero_cols), zero_cols,
            )
            clean_rets = clean_rets.loc[:, valid_vol_mask]
            kept_indices = kept_indices[valid_vol_mask.to_numpy()]

        if len(kept_indices) == 0:
            raise OptimizationError(
                "No symbols passed optimizer minimum-history and volatility gates.",
                OptimizationErrorType.DATA,
            )

        # Slice inputs with the FINAL kept_indices
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

        lw = LedoitWolf()
        lw.fit(clean_rets)
        Sigma     = lw.covariance_
        
        # 1. COVARIANCE MATRIX CONSTRUCTION - Enforce explicit Ridge Regularisation
        ridge     = max(1e-4 * float(np.trace(Sigma)), 1e-4 * m)
        Sigma_reg = Sigma + ridge * np.eye(m)

        gamma    = float(np.clip(exposure_multiplier, self.cfg.MIN_EXPOSURE_FLOOR, 1.0))

        adv_w           = adv_shares / np.maximum(adv_shares.sum(), 1e-9)
        adv_w_series    = pd.Series(adv_w, index=clean_rets.columns)
        aligned_w       = adv_w_series.reindex(clean_rets.columns).fillna(0.0).values
        aligned_w       = aligned_w / np.maximum(aligned_w.sum(), 1e-9)
        adv_weighted_rets = pd.Series(
            clean_rets.values.dot(aligned_w), index=clean_rets.index
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
        
        # 5. WEIGHT NORMALIZATION - Strict enforcement of dynamic exposure
        l_gamma = max(self.cfg.MIN_EXPOSURE_FLOOR, gamma * (1.0 - self.cfg.CAPITAL_ELASTICITY))
        u_gamma = min(1.0, gamma)  # Force optimizer to respect the exact dynamic regime exposure limits

        max_possible_weight = 0.0
        if sector_labels is not None:
            labels = np.asarray(sector_labels, dtype=int)
            for sec_id in np.unique(labels):
                mask    = labels == sec_id
                sec_max = float(np.sum(adv_limit[mask]))
                if mask.sum() >= 2:
                    sec_max = min(sec_max, self.cfg.MAX_SECTOR_WEIGHT)
                max_possible_weight += sec_max
        else:
            max_possible_weight = float(np.sum(adv_limit))

        u_gamma = min(u_gamma, max_possible_weight)
        if max_possible_weight < l_gamma:
            l_gamma = max_possible_weight * 0.99
        if np.sum(adv_limit) < l_gamma:
            l_gamma = np.sum(adv_limit) * 0.99
            
        u_gamma = max(u_gamma, l_gamma)
        if l_gamma > u_gamma:
            l_gamma = max(0.0, u_gamma - 1e-4)

        impact     = np.clip(
            self.cfg.IMPACT_COEFF * portfolio_value
            / np.maximum(adv_shares, 1.0),
            0.0, 1e4,
        )
        T_cvar     = min(T, self.cfg.CVAR_LOOKBACK)
        losses     = -clean_rets.iloc[-T_cvar:].values
        n_vars     = 2 * m + 1 + T_cvar + 1
        prev_w_arr = prev_w if prev_w is not None else np.zeros(m)

        P_w   = 2.0 * (self.cfg.RISK_AVERSION * Sigma_reg + np.diag(impact))
        P_aux = sp.eye(n_vars - m, format="csc") * 1e-6
        P     = sp.block_diag([sp.csc_matrix(P_w), P_aux], format="csc")

        turnover_costs = np.array(
            [
                compute_one_way_slip_rate(self.cfg, portfolio_value, float(adv))
                for adv in adv_shares
            ],
            dtype=float,
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
                mask = labels == sec_id
                if mask.sum() < 2:
                    continue
                sec_row = sp.lil_matrix((1, n_vars))
                sec_row[0, np.where(mask)[0]] = 1.0
                builder.add_constraint(sec_row.tocsc(), [0.0], [self.cfg.MAX_SECTOR_WEIGHT])

        tc = sp.lil_matrix((2 * m, n_vars))
        for i in range(m):
            tc[2*i,   i] =  1.0; tc[2*i,   m+i] = -1.0
            tc[2*i+1, i] = -1.0; tc[2*i+1, m+i] = -1.0

        tc_u = []
        for p in prev_w_arr:
            tc_u.extend([p, -p])
        builder.add_constraint(tc.tocsc(), [-np.inf] * (2 * m), tc_u)

        A, l, u = builder.build()

        prob = osqp.OSQP()
        prob.setup(
            P, q, A, l, u,
            verbose=False, eps_abs=1e-4, eps_rel=1e-4,
            polish=True, adaptive_rho=True, max_iter=50000,
        )
        res = prob.solve()

        if res.info.status not in ("solved", "solved inaccurate", "solved_inaccurate"):
            raise OptimizationError(
                f"OSQP status: {res.info.status}", OptimizationErrorType.NUMERICAL
            )

        # FIX G1: 'solved inaccurate' means the solver did not verify strict KKT conditions.
        # Sector caps, budget bounds, and ADV limits may be violated at the margin.
        # Log a warning so the diag log captures the occurrence frequency, then let the
        # downstream physical CVaR check serve as the hard safety gate. If the weights
        # survive that check they are deployed; if not, OptimizationError is raised below.
        if res.info.status in ("solved inaccurate", "solved_inaccurate"):
            logger.warning(
                "[Optimizer] OSQP returned '%s' — KKT conditions not strictly satisfied. "
                "Proceeding to physical CVaR verification; weights rejected if they "
                "exceed the hard limit. Consider tightening eps_abs/eps_rel or increasing "
                "max_iter if this warning appears frequently.",
                res.info.status,
            )

        w_opt = np.maximum(res.x[:m], 0.0)

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

        if physical_cvar > self.cfg.CVAR_DAILY_LIMIT + EPSILON:
            raise OptimizationError(
                f"Physical CVaR {physical_cvar:.4%} exceeds hard limit "
                f"{self.cfg.CVAR_DAILY_LIMIT:.4%} (solver reported {solver_cvar:.4%}, "
                f"slack={slack_value:.6f}). Refusing to deploy.",
                OptimizationErrorType.NUMERICAL,
            )

        full_w_opt = np.zeros(original_m)
        full_w_opt[kept_indices] = np.round(w_opt, 10)
        return full_w_opt
