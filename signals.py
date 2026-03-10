"""
signals.py — Deterministic Regime & Momentum Kernel v11.46
=========================================================
Generates momentum Z-scores, handles liquidity filtering, calculates
macro regime penalties, and implements the Dispersion-Normalized Continuity Bonus.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from momentum_engine import UltimateConfig

logger = logging.getLogger(__name__)

class SignalGenerationError(ValueError):
    """Raised when signal generation cannot proceed due to invalid input data."""


def compute_regime_score(
    idx_hist: Optional[pd.DataFrame],
    cfg: Optional['UltimateConfig'] = None,
    universe_close_hist: Optional[pd.DataFrame] = None,
) -> float:
    """
    Computes a macroeconomic regime score bounded [0, 1].
    
    High score -> Risk-on (upward trend, low volatility)
    Low score -> Risk-off (downward trend, high volatility)
    """
    if idx_hist is None or idx_hist.empty:
        logger.debug("[Signals] Missing index history for regime. Defaulting to 0.5")
        return 0.5
        
    close_series = idx_hist["Close"]
    
    # 1. Trend component (Distance from 200-day SMA)
    sma_window = int(getattr(cfg, "REGIME_SMA_WINDOW", 200)) if cfg else 200
    if len(close_series) >= sma_window:
        sma200 = float(close_series.rolling(window=sma_window).mean().iloc[-1])
    else:
        sma200 = float(close_series.expanding(min_periods=20).mean().iloc[-1])
    last_price = float(close_series.iloc[-1])
    
    if sma200 <= 0 or not np.isfinite(sma200):
        return 0.5
        
    trend_deviation = (last_price / sma200) - 1.0
    # Sigmoid function maps deviation to [0, 1] bounded score
    trend_steepness = float(getattr(cfg, "REGIME_SIGMOID_STEEPNESS", 10.0)) if cfg else 10.0
    base_score = 1.0 / (1.0 + np.exp(-trend_steepness * trend_deviation))
    
    # 2. Volatility penalty component
    returns_20d = close_series.pct_change(fill_method=None).tail(20)
    
    vol_component = 0.5
    if len(returns_20d) == 20:
        vol_20d = float(returns_20d.std() * np.sqrt(252))

        vol_floor = float(getattr(cfg, "REGIME_VOL_FLOOR", 0.18)) if cfg else 0.18
        vol_mult = float(getattr(cfg, "REGIME_VOL_MULTIPLIER", 1.5)) if cfg else 1.5

        all_returns = close_series.pct_change(fill_method=None).dropna()
        if len(all_returns) >= 252:
            long_term_vol = float(all_returns.tail(252).std() * np.sqrt(252))
        else:
            long_term_vol = vol_floor

        dynamic_threshold = max(vol_floor, long_term_vol * vol_mult)
        if vol_20d > dynamic_threshold:
            logger.debug("[Signals] Regime Volatility Spike detected (%.2f > %.2f). Applying penalty.", vol_20d, dynamic_threshold)
            base_score *= 0.85
        vol_component = float(np.clip(1.0 - (vol_20d / max(dynamic_threshold * 1.5, 1e-6)), 0.0, 1.0))

    breadth_component = 0.5
    _sma_win = int(getattr(cfg, "REGIME_SMA_WINDOW", 200)) if cfg else 200
    if universe_close_hist is not None and not universe_close_hist.empty:
        if len(universe_close_hist) >= _sma_win:
            recent = universe_close_hist.iloc[-_sma_win:]
            min_obs = max(1, int(np.ceil(_sma_win * 0.8)))
            obs_count = recent.notna().sum()
            sma_vals = recent.mean()
        else:
            min_obs = 20
            obs_count = universe_close_hist.notna().sum()
            sma_vals = universe_close_hist.expanding(min_periods=min_obs).mean().iloc[-1]
        last = universe_close_hist.iloc[-1]
        valid = (obs_count >= min_obs) & (sma_vals > 0) & sma_vals.notna() & last.notna()
        if valid.any():
            breadth_component = float((last[valid] > sma_vals[valid]).mean())

    composite = 0.5 * base_score + 0.3 * breadth_component + 0.2 * vol_component
    return round(float(np.clip(composite, 0.0, 1.0)), 10)


def compute_single_adv(df: pd.DataFrame, cfg: Optional['UltimateConfig'] = None) -> float:
    """
    Robust calculation of Average Daily Notional Volume (ADV) for a single asset.
    Handles NaN padding and computes configurable MA of (Close * Volume)
    Fixes the I-10 asymmetric floor bug and unit incoherence.
    """
    try:
        if "Close" not in df.columns or "Volume" not in df.columns:
            return 0.0
            
        notional = df["Close"] * df["Volume"]
        if notional.empty:
            return 0.0
            
        adv_lookback = int(getattr(cfg, "ADV_LOOKBACK", 20)) if cfg else 20
        # Take configurable moving average to ensure unit coherence against limit bounds.
        adv_val = float(notional.rolling(adv_lookback, min_periods=1).mean().iloc[-1])
        return adv_val if np.isfinite(adv_val) else 0.0
    except Exception as exc:
        logger.debug("[Signals] ADV calculation failed: %s", exc)
        return 0.0


def compute_adv(market_data: dict, active_symbols: List[str], cfg: Optional['UltimateConfig'] = None) -> np.ndarray:
    """
    Compute Average Daily Notional Volume for every symbol in a single
    vectorized pass.

    Builds a (T × N) notional DataFrame (Close × Volume) for all symbols
    simultaneously, applies one rolling(20).mean() across all columns at
    once, and extracts the last row.  This replaces the previous per-symbol
    loop — which allocated a new Pandas Series per ticker — with a single
    2-D matrix operation that is significantly faster inside the backtest
    inner loop and during Bayesian optimisation trials.

    Symbols absent from market_data receive a value of 0.0.
    Lookback defaults to 20 days when cfg is not supplied.
    """
    from momentum_engine import to_ns

    notional_cols: Dict[str, pd.Series] = {}
    for symbol in active_symbols:
        ns_sym = to_ns(symbol)
        df = market_data.get(ns_sym)
        if df is not None and "Close" in df.columns and "Volume" in df.columns:
            notional_cols[symbol] = df["Close"] * df["Volume"]

    if not notional_cols:
        return np.zeros(len(active_symbols), dtype=float)

    # Single rolling mean across the entire matrix — O(T·N) instead of N × O(T).
    notional_df = pd.DataFrame(notional_cols)
    notional_df.fillna(0.0, inplace=True)
    adv_lookback = int(getattr(cfg, "ADV_LOOKBACK", 20)) if cfg else 20
    adv_last_row = notional_df.rolling(adv_lookback, min_periods=1).mean().iloc[-1]

    def _safe_adv(sym: str) -> float:
        # Inline helper keeps `x` strictly scoped to this call frame.
        # Replaces the `v :=` walrus expression that previously leaked `v`
        # into the enclosing function scope after the list comprehension ran
        # (PEP 572 — assignment expressions in comprehensions deliberately
        # leak into the nearest enclosing non-comprehension scope, creating a
        # latent state-pollution footgun for any future edit that references
        # `v` after the return statement).
        x = adv_last_row.get(sym, 0.0)
        return float(x) if np.isfinite(x) else 0.0

    return np.array([_safe_adv(sym) for sym in active_symbols], dtype=float)


# _apply_adv_filter has been moved to universe_manager.py.
# It is a universe curation function, not a signal computation function,
# and belongs alongside the other universe management utilities.
# Import it from there: from universe_manager import _apply_adv_filter

def generate_signals(
    log_rets:     pd.DataFrame,
    adv_arr:      np.ndarray,
    cfg:          'UltimateConfig',
    prev_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Core momentum generation engine.
    Computes blended fast/slow exponentially weighted returns, applies historical
    and liquidity gates, and attaches a dispersion-normalized continuity bonus
    to reduce portfolio turnover friction.
    """
    if log_rets.empty:
        raise SignalGenerationError("no valid data: log_rets dataframe is empty")
        
    active_symbols = list(log_rets.columns)
    
    # 1. Base Signal Calculation (Blended EWMA)
    signal_lag_days = max(int(getattr(cfg, "SIGNAL_LAG_DAYS", 0)), 0)
    if signal_lag_days > 0 and len(log_rets) > signal_lag_days:
        signal_log_rets = log_rets.iloc[:-signal_lag_days]
    else:
        signal_log_rets = log_rets

    fast_ema = signal_log_rets.ewm(halflife=cfg.HALFLIFE_FAST).mean().iloc[-1].values
    slow_ema = signal_log_rets.ewm(halflife=cfg.HALFLIFE_SLOW).mean().iloc[-1].values

    raw_daily_momentum = 0.5 * fast_ema + 0.5 * slow_ema

    # Cross-sectional Z-score standardization
    mu_cross = np.nanmean(raw_daily_momentum)
    std_cross = max(np.nanstd(raw_daily_momentum), 1e-8)

    adj_scores = np.clip(
        (raw_daily_momentum - mu_cross) / std_cross,
        -cfg.Z_SCORE_CLIP,
        cfg.Z_SCORE_CLIP
    )

    # 2. Hard Gates (Disqualification)
    for i, sym in enumerate(active_symbols):
        # Gate A: Minimum History Requirement
        valid_history_days = int(log_rets[sym].notna().sum())
        if valid_history_days < cfg.HISTORY_GATE:
            adj_scores[i] = -np.inf
            
        # Gate B: Liquidity / ADV Requirement
        if not np.isfinite(adv_arr[i]) or adv_arr[i] <= 0:
            adj_scores[i] = -np.inf
            
    # Gate C: Falling Knife Protection
    if len(log_rets) >= cfg.KNIFE_WINDOW:
        recent_simple = np.expm1(log_rets.iloc[-cfg.KNIFE_WINDOW:])
        recent_cumulative_returns = ((1.0 + recent_simple).prod(min_count=1) - 1.0).values
        for i, cumulative_ret in enumerate(recent_cumulative_returns):
            if np.isfinite(cumulative_ret) and cumulative_ret < cfg.KNIFE_THRESHOLD:
                adj_scores[i] = -np.inf

    # 3. FIX: Dispersion-Normalized Continuity Bonus
    # Standardizing the bonus against the current cross-sectional dispersion
    # ensures it isn't over-dominant in tight low-vol markets or invisible in wide high-vol ones.
    valid_mask = np.isfinite(adj_scores)
    
    if prev_weights and valid_mask.any():
        # Guard: nanstd on a single-element array returns NaN (0/0 in ddof=1 mode),
        # which would poison base_bonus.  Default strictly to the floor when fewer
        # than 2 finite scores are available (e.g. extreme crash states where all but
        # one asset is gated out).
        if valid_mask.sum() >= 2:
            current_dispersion = max(np.nanstd(adj_scores[valid_mask]), cfg.CONTINUITY_DISPERSION_FLOOR)
        else:
            current_dispersion = cfg.CONTINUITY_DISPERSION_FLOOR
        cap = float(getattr(cfg, "CONTINUITY_MAX_SCALAR", 0.20))
        base_bonus = min(cfg.CONTINUITY_BONUS, cap) * current_dispersion

        activity_window = max(int(getattr(cfg, "CONTINUITY_ACTIVITY_WINDOW", 5)), 1)
        stale_sessions = max(int(getattr(cfg, "CONTINUITY_STALE_SESSIONS", 10)), 1)
        min_nonzero_days = max(int(getattr(cfg, "CONTINUITY_MIN_NONZERO_DAYS", 1)), 1)
        flat_ret_eps = float(getattr(cfg, "CONTINUITY_FLAT_RET_EPS", 1e-12))
        continuity_min_adv = float(getattr(cfg, "CONTINUITY_MIN_ADV_NOTIONAL", 0.0))

        stale_denied = 0
        liquidity_denied = 0

        for i, sym in enumerate(active_symbols):
            prev_w = float(prev_weights.get(sym, 0.0))
            if valid_mask[i] and prev_w > 0.001:
                recent_rets = log_rets[sym].tail(activity_window)
                nonzero_days = int((recent_rets.abs() > flat_ret_eps).sum()) if len(recent_rets) else 0
                has_recent_activity = nonzero_days >= min_nonzero_days

                stale_rets = log_rets[sym].tail(stale_sessions)
                is_stale = (
                    len(stale_rets) == stale_sessions
                    and stale_rets.notna().all()
                    and bool((stale_rets.abs() <= flat_ret_eps).all())
                )

                passes_continuity_liquidity = np.isfinite(adv_arr[i]) and adv_arr[i] >= continuity_min_adv
                continuity_eligible = has_recent_activity or passes_continuity_liquidity

                if is_stale:
                    stale_denied += 1
                if not passes_continuity_liquidity:
                    liquidity_denied += 1
                if is_stale or not passes_continuity_liquidity or not continuity_eligible:
                    continue

                decay = float(np.clip(prev_w / max(getattr(cfg, "CONTINUITY_MAX_HOLD_WEIGHT", 0.10), 1e-6), 0.25, 1.0))
                adj_scores[i] += base_bonus * decay

        if stale_denied or liquidity_denied:
            logger.debug(
                "[Signals] Continuity denied for %d stale and %d illiquid symbols.",
                stale_denied,
                liquidity_denied,
            )

    # 4. Final Selection
    # Sort and pick the top N elements (ignoring those assigned -inf)
    sorted_indices = np.argsort(adj_scores)
    top_n_indices = sorted_indices[-cfg.MAX_POSITIONS:]
    
    selected_indices = [
        int(idx) for idx in top_n_indices 
        if adj_scores[idx] > -np.inf
    ]
    
    return raw_daily_momentum, adj_scores, selected_indices
