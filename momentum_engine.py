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
_GHOST_SEED_CACHE: Dict[str, int] = {}


def _ghost_seed_for(sym: str) -> int:
    """Return a deterministic integer seed for a given symbol, cached per process."""
    if sym not in _GHOST_SEED_CACHE:
        _GHOST_SEED_CACHE[sym] = int(hashlib.sha256(sym.encode()).hexdigest()[:8], 16) % (2 ** 31)
    return _GHOST_SEED_CACHE[sym]


# ─── Symbol helpers ───────────────────────────────────────────────────────────

def to_ns(sym: str) -> str:
    if sym.startswith("^") or sym.endswith(".NS"):
        return sym
    return sym + ".NS"


def to_bare(sym: str) -> str:
    return sym[:-3] if sym.endswith(".NS") else sym


def absent_symbol_effective_price(last_known_price: float, absent_periods: int, max_absent_periods: int) -> float:
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
    SLACK_PENALTY:            float = 1000.0
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
    REGIME_SMA_WINDOW:        int   = 200
    REGIME_VOL_EWMA_SPAN:     int   = 20
    REGIME_LT_VOL_EWMA_SPAN:  int   = 1260

    # Ghost risk synthesis
    GHOST_VOL_LOOKBACK:       int   = 20
    GHOST_RET_DRIFT:          float = -0.02
    GHOST_VOL_FALLBACK:       float = 0.04
    SIMULATE_HALTS:           bool  = False

    # Institutional flags
    DIVIDEND_SWEEP:           bool  = True
    SPLIT_TOLERANCE:          float = 0.005
    AUTO_ADJUST_PRICES:       bool  = True

    # Equity history buffer cap.
    # FIX-NEW-CC-01 / NEW-CC-01: Although this field appears after the @property
    # definitions, Python's dataclass machinery processes ALL class-level
    # annotations as fields regardless of ordering — EQUITY_HIST_CAP is present
    # in __dataclass_fields__ and is therefore visible to load_optimized_config's
    # JSON loader.  Confirmed via: 'EQUITY_HIST_CAP' in UltimateConfig.__dataclass_fields__
    # → True.  Kept here to avoid a large diff; moved above the property block
    # in comments for readability.
    EQUITY_HIST_CAP: int = 500

    @property
    def SLIPPAGE_BPS(self) -> float:
        return self.ROUND_TRIP_SLIPPAGE_BPS

    @SLIPPAGE_BPS.setter
    def SLIPPAGE_BPS(self, value: float) -> None:
        try:
            self.ROUND_TRIP_SLIPPAGE_BPS = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SLIPPAGE_BPS must be numeric, received {value!r}") from exc


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

        # FIX-MB-ME-03: Decrement cooldown, but evaluate the breach condition
        # AFTER clearing override_active so that a sustained breach spanning the
        # entire cooldown period re-arms the override immediately on the step
        # cooldown reaches zero, rather than allowing one unprotected rebalance.
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
            # FIX-MB-M-05: use max(current, 4) so any longer externally-set
            # cooldown (e.g. from activate_override_on_stress) is not shortened
            # back to 4 by this re-arm.
            self.override_cooldown   = max(self.override_cooldown, 4)
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
        cap = self.equity_hist_cap
        if cap > 0 and len(self.equity_hist) > cap:
            self.equity_hist = self.equity_hist[-cap:]

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
    state.override_active = True
    state.override_cooldown = max(state.override_cooldown, 4)
    state.exposure_multiplier = float(max(cfg.MIN_EXPOSURE_FLOOR, state.exposure_multiplier * 0.5))


def compute_one_way_slip_rate(
    cfg: UltimateConfig,
    portfolio_value: float,
    adv_notional: Optional[float],
    trade_notional: Optional[float] = None,
) -> float:
    """
    Compute per-name one-way slippage rate.

    FIX-MB-ME-02: Market impact is now scaled against trade_notional (the delta
    notional being traded) rather than portfolio_value. Using portfolio_value as
    the impact numerator caused systematic overestimation for small trades and
    underestimation for large rebalances, because a 1-share trade in a large
    portfolio received the same impact rate as a full-position rebalance.

    When trade_notional is not provided (backward-compatible callers), falls back
    to portfolio_value to preserve existing behaviour.

    One-way cost = half of round-trip (ROUND_TRIP_SLIPPAGE_BPS / 2).
    The rate returned here is applied ONCE per trade side.
    """
    base_rate = cfg.ROUND_TRIP_SLIPPAGE_BPS / 20_000.0
    if adv_notional is None or not np.isfinite(adv_notional) or adv_notional <= 0:
        return base_rate

    # FIX-MB-ME-02: prefer trade_notional for impact scaling; fall back to
    # portfolio_value when trade_notional is unavailable (legacy call sites).
    numerator = trade_notional if (trade_notional is not None and np.isfinite(trade_notional) and trade_notional > 0) \
        else portfolio_value

    impact_rate = (cfg.IMPACT_COEFF * numerator) / float(adv_notional)
    return max(base_rate, min(0.05, impact_rate))


# ─── Execution ────────────────────────────────────────────────────────────────

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
    """Execute a portfolio rebalance, updating state in-place.

    FIX-MB-ME-01: Two-phase PV accounting explanation:
    Phase 1 — pv_exec is built from cash + retained positions + ghost positions.
              Force-close candidates are EXCLUDED here (sym not in symbols_to_force_close).
    Phase 2 — Force-close positions are sold: proceeds (n_shares * close_price)
              are added to pv_exec, and the sell trade is logged.
    Settlement — state.cash = pv_exec - actual_notional - total_slippage.
              actual_notional covers only retained (new_shares) positions, not
              force-closed ones. The force-close proceeds were added to pv_exec
              in Phase 2, so cash correctly reflects their liquidation without
              needing to subtract them via actual_notional.
    """
    active_idx = {sym: i for i, sym in enumerate(active_symbols)}
    local_prices = np.array(prices, dtype=float, copy=True)

    t1_price_snapshot: Dict[str, float] = dict(state.last_known_prices)

    for sym, i in active_idx.items():
        px = float(local_prices[i])
        if np.isfinite(px) and px > 0:
            state.last_known_prices[sym] = px
        else:
            px = float(state.last_known_prices.get(sym, 0.0))
        local_prices[i] = px
        state.absent_periods.pop(sym, None)

    symbols_to_force_close: List[str] = []
    absent_snapshot: Dict[str, int] = dict(state.absent_periods)
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

    # Phase 1: build pv_exec excluding force-close candidates (see docstring)
    pv_t1 = state.cash
    pv_exec = state.cash

    for sym, n_shares in state.shares.items():
        if sym in active_idx:
            px_exec = float(local_prices[active_idx[sym]])
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
                exit_slip_rate = (cfg.ROUND_TRIP_SLIPPAGE_BPS / 2) / 10_000
                for sym, n_shares in state.shares.items():
                    if sym in active_idx:
                        px_exec = float(local_prices[active_idx[sym]])
                    else:
                        px_exec = absent_symbol_effective_price(
                            state.last_known_prices.get(sym, 0.0),
                            state.absent_periods.get(sym, 0),
                            cfg.MAX_ABSENT_PERIODS,
                        )
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
                            tdate = pd.Timestamp(date_context) if date_context is not None else pd.Timestamp.utcnow()
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

    desired_shares: Dict[str, int] = {}
    valid_targets: List[Tuple[int, str, float, float]] = []
    base_notional = 0.0

    for i, sym in enumerate(active_symbols):
        w = round(float(target_weights[i]), 10)
        if not np.isfinite(w):
            w = 0.0
        price = max(float(local_prices[i]), 1e-6)
        old_s = state.shares.get(sym, 0)
        current_notional = old_s * price

        # FIX-MB-ME-02: compute trade notional for impact calculation.
        # Share count is sized at raw price; slippage is a separate cash cost
        # charged in the accounting loop below via the slip_rate on the actual
        # traded notional.  Using effective_buy_price (raw * (1+slip)) for share
        # count caused a systematic under-allocation: a 2% intended weight move
        # was reduced to 1.5% in shares and then blocked by the drift gate, even
        # though the manager clearly intended to trade.
        # FIX-MAGIC-THRESHOLD: removed the hard 0.001 (0.1%) weight floor.
        # The previous guard forced any target weight below 0.1% to zero shares,
        # which incorrectly liquidates valid optimizer positions in broad-market
        # configurations (e.g. MAX_POSITIONS=100+ where average weight is <0.5%).
        # The natural floor is already enforced by integer share flooring:
        # floor(w * pv_exec / price) = 0 whenever the notional is less than one
        # share's price — no explicit threshold needed.
        if w * pv_exec > current_notional:
            buy_notional = max(0.0, w * pv_exec - current_notional)
            # Size at raw price; slip is charged in the accounting loop.
            s = old_s + int(np.floor(buy_notional / max(price, 1e-9)))
            trade_not = buy_notional
        else:
            s = int(np.floor(w * pv_exec / price))
            trade_not = abs(s - old_s) * price

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

    if not force_rebalance_trades:
        drift_threshold = getattr(cfg, "DRIFT_TOLERANCE", 0.02)
        # Collect gated symbols so we can purge them from valid_targets below.
        drift_gated_syms: set = set()
        for i, sym in enumerate(active_symbols):
            old_s = state.shares.get(sym, 0)
            s = desired_shares.get(sym, 0)
            price = max(float(local_prices[i]), 1e-6)

            if old_s > 0 and s > 0 and s != old_s:
                # FIX: compare the INTENDED weight change (target_weight vs
                # current weight at raw price) not the slippage-adjusted share
                # delta.  Using slippage-adjusted shares means a clean 2% intended
                # move that slippage reduces to 1.5% in shares gets incorrectly
                # blocked.  The drift gate should ask "does the manager intend to
                # move by more than the threshold?" which is a function of the
                # target weight, not of execution slippage.
                w_target  = float(target_weights[i]) if np.isfinite(target_weights[i]) else 0.0
                w_current = (old_s * price) / max(pv_exec, 1.0)
                intended_change = abs(w_target - w_current)
                if intended_change < drift_threshold:
                    desired_shares[sym] = old_s
                    drift_gated_syms.add(sym)

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

    # FIX-SLIP-RESERVE: pre-estimate the slippage cost of base trades and
    # subtract it from the residual budget before the allocation loop starts.
    # Without this reserve, the residual loop spends cash that is already
    # committed to paying the broker for base-trade slippage.  The final
    # settlement (state.cash = pv_exec - actual_notional - total_slippage)
    # then goes slightly negative, silently absorbed by the max(0.0,...) floor
    # as phantom margin money.  At the default 0.1% slip the leakage is small
    # (~0.08% of portfolio per rebalance) but compounds over many rebalances
    # and grows proportionally with illiquidity-driven impact costs.
    _base_slip_reserve = 0.0
    for _i, _sym in enumerate(active_symbols):
        _old_s = state.shares.get(_sym, 0)
        _s = desired_shares.get(_sym, 0)
        if _s > 0 or _old_s > 0:
            _px = max(float(local_prices[_i]), 1e-6)
            _delta_not = abs(_s - _old_s) * _px
            _sr = compute_one_way_slip_rate(
                cfg=cfg,
                portfolio_value=pv_exec,
                adv_notional=float(adv_shares[_i]) if adv_shares is not None and _i < len(adv_shares) else None,
                trade_notional=_delta_not,
            )
            _base_slip_reserve += _delta_not * _sr

    residual_cash = max(0.0, pv_exec - base_notional - _base_slip_reserve)

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

            for sym, data in eligible.items():
                price = data["price"]
                i     = data["i"]
                w_i   = data["w"]
                cash_entitlement = (w_i / total_eligible_w) * residual_cash
                slip_rate = compute_one_way_slip_rate(
                    cfg=cfg,
                    portfolio_value=pv_exec,
                    adv_notional=float(adv_shares[i]) if adv_shares is not None else None,
                    trade_notional=cash_entitlement,
                )
                effective_buy_price = price * (1.0 + slip_rate)

                extra = int(cash_entitlement // max(effective_buy_price, 1e-9))

                if extra <= 0:
                    if effective_buy_price > residual_cash:
                        to_remove.append(sym)
                    continue

                headroom_notional = cfg.MAX_SINGLE_NAME_WEIGHT * pv_exec - desired_shares[sym] * price
                max_extra_weight  = max(0, int(headroom_notional // max(effective_buy_price, 1e-9)))

                max_extra_adv = extra
                if adv_shares is not None and i < len(adv_shares):
                    adv_notional = float(adv_shares[i])
                    if adv_notional > 0:
                        max_adv_total = int(np.floor((adv_notional * cfg.MAX_ADV_PCT) / max(effective_buy_price, 1e-9)))
                        max_extra_adv = max(0, max_adv_total - desired_shares[sym])

                cap_limit    = min(max_extra_weight, max_extra_adv)
                actual_extra = min(extra, cap_limit)

                if actual_extra > 0:
                    desired_shares[sym] += actual_extra
                    # FIX-RESIDUAL-BUDGET: deduct the EFFECTIVE (slipped) cost
                    # from the residual_cash spending budget.
                    #
                    # The previous FIX-NEW-ME-01 comment claimed raw deduction
                    # was necessary to avoid double-charging slippage.  That
                    # reasoning was wrong: residual_cash is a LOCAL loop budget
                    # only — it is never used in the final cash settlement.  The
                    # final accounting always computes:
                    #   state.cash = pv_exec - actual_notional - total_slippage
                    # where actual_notional = sum(shares * raw_price) and
                    # total_slippage = sum(|delta| * raw_price * slip_rate).
                    # Neither term reads residual_cash, so there is no
                    # double-deduction regardless of what we subtract here.
                    #
                    # What raw deduction DID cause: when slip_rate is large
                    # (e.g. 5% market-impact cap on an illiquid small-cap),
                    # the loop could authorise more shares than the portfolio
                    # can actually afford.  The final settlement would then
                    # drive state.cash negative, silently covered by the
                    # max(0.0, ...) floor — phantom margin money.
                    #
                    # Correct fix: deduct effective_buy_price so the budget
                    # reflects the true all-in cost of each purchase, keeping
                    # total authorised spending within the available residual.
                    residual_cash       -= actual_extra * effective_buy_price
                    shares_bought_this_pass += actual_extra

                if actual_extra >= cap_limit or effective_buy_price > residual_cash:
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
                    tdate = pd.Timestamp(date_context) if date_context is not None else pd.Timestamp.utcnow()
                    trade_log.append(Trade(sym, tdate, -n_shares, close_price, slip, "SELL"))
            else:
                logger.error(
                    "execute_rebalance: force-close of %s (%d shares) has no last "
                    "known price — position removed at ₹0.",
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
    # FIX-MB-CASH-FLOOR: Validate cash doesn't go negative
    raw_cash = pv_exec - actual_notional - total_slippage
    state.cash         = max(0.0, round(pv_exec - actual_notional - total_slippage, 10))
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

    # FIX-MB2-GHOSTPV: fill_value=np.nan so absent symbols get ghost synthesis
    # rather than silent zero-fill which understates CVaR.
    rets = hist_log_rets.reindex(columns=held_syms, fill_value=np.nan)
    rets = rets.replace([np.inf, -np.inf], np.nan).ffill().iloc[-T_cvar:]

    vol_window = max(5, cfg.GHOST_VOL_LOOKBACK)
    if not hist_log_rets.empty:
        rolling_vol = (
            hist_log_rets.replace([np.inf, -np.inf], np.nan)
            .iloc[-vol_window:]
            .std()
            .dropna()
        )
        # FIX-MB-VOL: write only under the canonical key as it appears in
        # state.shares; do not write aliased bare/.NS keys to avoid overwriting
        # frozen vol for absent ghost symbols.
        for sym_key, vol in rolling_vol.items():
            vol_value = float(max(vol, 1e-4))
            state.last_known_volatility[str(sym_key)] = vol_value

    ghost_mask = np.array([
        (s not in active_idx) or (s in rets.columns and rets[s].isna().all())
        for s in held_syms
    ])
    if ghost_mask.any():
        ghost_cols = sorted(s for s, is_ghost in zip(held_syms, ghost_mask) if is_ghost)

        for sym in ghost_cols:
            vol = float(state.last_known_volatility.get(sym, cfg.GHOST_VOL_FALLBACK))
            vol = max(vol, cfg.GHOST_VOL_FALLBACK)
            daily_vol   = vol
            daily_drift = float(cfg.GHOST_RET_DRIFT) / 252.0

            sym_base_seed = _ghost_seed_for(sym)

            # FIX-NEW-ME-03: derive row seeds from the absolute calendar date
            # (nanoseconds-since-epoch → days-since-epoch) rather than from
            # rets.index, which may have different depth across calls depending
            # on how much history was forward-filled into the slice.  Using the
            # calendar date means the same synthetic return is generated for a
            # given (symbol, date) pair regardless of how long the surrounding
            # window is, giving reproducible CVaR estimates across slice depths.
            if hasattr(rets.index, 'asi8'):
                # DatetimeIndex: convert ns timestamps to integer day numbers
                days_since_epoch = (rets.index.asi8 // np.int64(86_400 * 10 ** 9)).astype(np.int64)
            else:
                # Fallback for non-datetime index: use positional integers
                days_since_epoch = np.arange(len(rets), dtype=np.int64)

            row_seeds = (
                np.int64(sym_base_seed) ^ days_since_epoch
            ) & np.int64(0x7FFFFFFF)

            ss = np.random.SeedSequence(row_seeds.tolist())
            rngs = [np.random.default_rng(s) for s in ss.spawn(len(row_seeds))]
            synth_rets = np.array([r.normal(daily_drift, daily_vol) for r in rngs])
            rets.loc[:, sym] = synth_rets

    rets = rets.fillna(0.0)

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
            shares = state.shares.get(sym, 0)
            price = max(float(current_prices[i]), 1e-6)
            pre_decay_weight = (shares * price) / max(float(pv), 1.0)
        else:
            pre_decay_weight = float(state.weights.get(sym, 0.0))

        pre_decay_weight = min(max(pre_decay_weight, 0.0), cfg.MAX_SINGLE_NAME_WEIGHT)
        targets[i] = pre_decay_weight * cfg.DECAY_FACTOR

    return targets


# ─── Optimizer ────────────────────────────────────────────────────────────────

class InstitutionalRiskEngine:
    def __init__(self, cfg: UltimateConfig):
        self.cfg:       UltimateConfig              = cfg
        self.last_diag: Optional[SolverDiagnostics] = None
        self._solver:       Optional[object] = None
        self._solver_shape: Optional[tuple]  = None
        self._solver_nnz:   Optional[tuple]  = None
        self._solver_struct: Optional[tuple] = None

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
            # FIX-NEW-ME-04: use strict > so that a same-day last bar
            # (historical_returns.index.max() == execution_date) does not
            # spuriously raise.  The signal slice in _run_rebalance already
            # excludes the execution date via loc[:signal_date], so an equality
            # hit means the last bar IS the signal date — that is valid.
            # Only a bar that is strictly AFTER execution_date is look-ahead.
            if historical_returns.index.max() > pd.Timestamp(execution_date):
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
        # FIX-ZERO-FILL: replace zeros with the cross-sectional row mean so that
        # pre-IPO NaN cells (after ffill exhausts real data) are treated as
        # 'market-average return that day' rather than 'risk-free zero return'.
        # Filling with 0.0 causes the LedoitWolf covariance estimator to see the
        # stock as uncorrelated with all other assets during the pre-IPO period,
        # slightly underestimating its covariance and making it appear as a
        # partial hedge — a mild but systematic bias toward newer listings.
        # Cross-sectional mean fill is standard practice: it assumes the stock
        # would have moved with the market-average on days it didn't exist,
        # producing a more conservative (realistic) covariance estimate.
        # The history gate already ensures at most ~30% of any column is NaN,
        # so row means are anchored by the majority of real observations.
        _row_means = clean_rets.mean(axis=1)
        # FIX-SYSTEMIC-NAN: chain a final .fillna(0.0) to handle systemic
        # market-closure days where every stock has NaN (e.g. a total provider
        # gap or unexpected exchange suspension).  On such days _row_means is
        # NaN, so .fillna(_row_means) leaves those cells as NaN, which would
        # crash LedoitWolf with 'Input contains NaN'.  Filling with 0.0 treats
        # a total market closure as a flat day — the most conservative
        # assumption and consistent with the price ffill applied upstream.
        clean_rets = clean_rets.apply(lambda col: col.fillna(_row_means)).fillna(0.0)

        col_stds = clean_rets.std()
        valid_vol_mask = col_stds >= 1e-10

        if not valid_vol_mask.all():
            zero_cols = valid_vol_mask[~valid_vol_mask].index.tolist()
            logger.warning(
                "[Optimizer] Detected %d zero-volatility asset(s): %s",
                len(zero_cols), zero_cols,
            )
            if valid_vol_mask.any():
                clean_rets = clean_rets.loc[:, valid_vol_mask]
                kept_indices = kept_indices[valid_vol_mask.to_numpy()]

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

        simple_rets = np.expm1(clean_rets)
        lw = LedoitWolf()
        lw.fit(simple_rets)
        Sigma_reg = lw.covariance_
        ridge     = 0.0

        gamma    = float(np.clip(exposure_multiplier, self.cfg.MIN_EXPOSURE_FLOOR, 1.0))

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

        l_gamma = max(self.cfg.MIN_EXPOSURE_FLOOR, gamma * (1.0 - self.cfg.CAPITAL_ELASTICITY))
        u_gamma = min(1.0, gamma)

        # FIX-MB-SECTOR: known-sector groups capped at MAX_SECTOR_WEIGHT;
        # unknown-sector (sentinel -1) stocks governed only by ADV limits.
        max_possible_weight = 0.0
        if sector_labels is not None:
            labels = np.asarray(sector_labels, dtype=int)
            for sec_id in np.unique(labels):
                mask    = labels == sec_id
                sec_adv_sum = float(np.sum(adv_limit[mask]))
                if sec_id == -1:
                    max_possible_weight += sec_adv_sum
                else:
                    # FIX-BUDGET-MISMATCH: always apply MAX_SECTOR_WEIGHT cap here,
                    # matching the constraint builder which now adds a sector
                    # constraint for ALL sector sizes including single-stock sectors
                    # (FIX-SECTOR-SINGLE).  The previous 'mask.sum() >= 2' guard
                    # left single-stock sectors uncapped in the budget, inflating
                    # max_possible_weight and therefore l_gamma above the actual
                    # feasible maximum enforced by the OSQP constraint matrix —
                    # causing guaranteed primal infeasibility when
                    # MAX_SECTOR_WEIGHT < MAX_SINGLE_NAME_WEIGHT.
                    max_possible_weight += min(sec_adv_sum, self.cfg.MAX_SECTOR_WEIGHT)
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
        losses     = -simple_rets.iloc[-T_cvar:].values
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
                if sec_id == -1:
                    continue
                mask = labels == sec_id
                # FIX-SECTOR-SINGLE: removed the 'mask.sum() < 2' guard that
                # skipped sector constraints for single-stock sectors.
                # When MAX_SECTOR_WEIGHT < MAX_SINGLE_NAME_WEIGHT (e.g. a user
                # enforcing tighter sector diversification than per-name limits),
                # a lone stock in its sector could reach MAX_SINGLE_NAME_WEIGHT
                # unchecked, silently violating the sector cap.
                # A single-row sector constraint (w_i <= MAX_SECTOR_WEIGHT) is
                # perfectly valid in OSQP — it simply tightens the effective per-
                # stock bound to min(adv_limit, MAX_SECTOR_WEIGHT) as intended.
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

        P_upper = sp.triu(P, format="csc")
        current_shape = (m, T_cvar)
        current_nnz   = (P_upper.nnz, A.nnz)

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
            self._solver.setup(
                P_upper, q, A, l, u,
                verbose=False, eps_abs=1e-4, eps_rel=1e-4,
                polish=True, adaptive_rho=True, max_iter=50000,
                warm_starting=True,
            )
            self._solver_shape = current_shape
            self._solver_nnz   = current_nnz
            self._solver_struct = (
                P_upper.indices.copy(), P_upper.indptr.copy(),
                A.indices.copy(), A.indptr.copy(),
            )
        else:
            self._solver.update(
                q=q, l=l, u=u,
                Px=P_upper.data, Ax=A.data,
            )

        # FIX-MB-OSQP: invalidate solver cache on any exception so the next
        # call gets a fresh setup rather than reusing a broken solver state.
        try:
            res = self._solver.solve()
        except Exception as exc:
            logger.error(
                "[Optimizer] OSQP solve() raised an exception: %s — "
                "invalidating solver cache to force fresh setup on next call.", exc
            )
            self._solver        = None
            self._solver_shape  = None
            self._solver_nnz    = None
            self._solver_struct = None
            raise OptimizationError(
                f"OSQP solve() failed with exception: {exc}",
                OptimizationErrorType.NUMERICAL,
            ) from exc

        if res.info.status not in ("solved", "solved inaccurate", "solved_inaccurate"):
            self._solver        = None
            self._solver_shape  = None
            self._solver_nnz    = None
            self._solver_struct = None
            raise OptimizationError(
                f"OSQP status: {res.info.status}", OptimizationErrorType.NUMERICAL
            )

        if res.info.status in ("solved inaccurate", "solved_inaccurate"):
            logger.warning(
                "[Optimizer] OSQP returned '%s' — KKT conditions not strictly satisfied. "
                "Proceeding to physical CVaR verification.",
                res.info.status,
            )

        w_opt = np.maximum(res.x[:m], 0.0)

        if res.info.status in ("solved inaccurate", "solved_inaccurate"):
            logger.warning(
                "[Optimizer] OSQP returned '%s'. Normalizing weights to γ=%.4f.",
                res.info.status, gamma,
            )
            w_sum = float(np.sum(w_opt))
            if w_sum > 1e-9:
                w_opt = np.minimum(w_opt, adv_limit)
                clipped_sum = float(np.sum(w_opt))
                if clipped_sum > 1e-9:
                    w_opt = w_opt * min(gamma / clipped_sum, 1.0)

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
