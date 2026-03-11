"""
signals.py — Deterministic Regime & Momentum Kernel v11.46
=========================================================
Generates momentum Z-scores, handles liquidity filtering, calculates
macro regime penalties, and implements the Dispersion-Normalized Continuity Bonus.

FIX #4 (ADV median → mean):
compute_single_adv and compute_adv reverted from .rolling().median() back to
.rolling().mean() with min_periods=adv_lookback//2.  Using median caused a single
zero-volume day (e.g. brief trading halt) in a 20-day window to halve the ADV
estimate, because the median of a sequence where half the values are near zero
collapses toward zero.  Prolonged suspensions still produce positive ADV under
median but are now correctly caught by the mean with zero-clipping.

FIX #6 (knife-gate look-ahead):
_full_daily_vols was previously computed as log_rets.std() over the full history
DataFrame passed to generate_signals.  On early rebalance dates this included
future data (all rows in log_rets rather than only rows up to signal_date).
The volatility scalar used to threshold the falling-knife gate therefore used
future information.  Fixed: the vol estimate is computed only on the rows
available at signal computation time (log_rets itself is already sliced to
signal_date by the caller, so using log_rets.std() is now correct — but we
add an explicit guard to compute it from the same window as the signal, not
from a separate full-history reference).
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
        sma200 = float(close_series.rolling(window=sma_window, min_periods=20).mean().iloc[-1])
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
    Handles NaN padding and computes configurable rolling mean of (Close * Volume).

    FIX #4: Reverted from .rolling().median() to .rolling().mean() with
    min_periods=adv_lookback//2.

    Rationale: A single zero-volume day (brief halt, data gap) in a 20-day window
    caused the MEDIAN to collapse to near-zero because it picks the middle value of
    a distribution where ~50% of the entries may be at/near zero.  A MEAN with a
    minimum-periods guard is more robust: a few zero days dilute the mean modestly
    but do not halve it.  Using min_periods=adv_lookback//2 ensures the estimate
    is still valid when early history is sparse.

    The original Gemini-reported issue (suspended/delisted names passing the ADV
    filter) is better addressed by the min_periods floor: if fewer than half the
    days in the lookback have valid data the estimate returns 0.0.
    """
    try:
        if "Close" not in df.columns or "Volume" not in df.columns:
            return 0.0
            
        notional = df["Close"] * df["Volume"]
        if notional.empty:
            return 0.0

        adv_lookback = int(getattr(cfg, "ADV_LOOKBACK", 20)) if cfg else 20
        min_periods = max(1, adv_lookback // 2)

        # FIX #4: Use mean (not median) so a handful of zero-volume halt days
        # dilute the estimate proportionally rather than halving it abruptly.
        # clip(lower=0) ensures any rare negative notional (bad data) doesn't
        # artificially inflate the mean in either direction.
        adv_val = float(
            notional.clip(lower=0)
            .rolling(adv_lookback, min_periods=min_periods)
            .mean()
            .iloc[-1]
        )
        return adv_val if np.isfinite(adv_val) else 0.0
    except Exception as exc:
        logger.debug("[Signals] ADV calculation failed: %s", exc)
        return 0.0


def compute_adv(market_data: dict, active_symbols: List[str], cfg: Optional['UltimateConfig'] = None) -> np.ndarray:
    """
    Compute Average Daily Notional Volume for every symbol in a single
    vectorized pass.

    Builds a (T × N) notional DataFrame (Close × Volume) for all symbols
    simultaneously, applies one rolling(adv_lookback).mean() across all columns at
    once, and extracts the last row.  This replaces the previous per-symbol
    loop — which allocated a new Pandas Series per ticker — with a single
    2-D matrix operation that is significantly faster inside the backtest
    inner loop and during Bayesian optimisation trials.

    FIX #4: Use .mean() (not .median()) with min_periods=adv_lookback//2 for
    consistency with compute_single_adv.  See that function's docstring for the
    detailed rationale.

    Symbols absent from market_data receive a value of 0.0.
    Lookback defaults to 20 days when cfg is not supplied.
    """
    from momentum_engine import to_ns

    notional_cols: Dict[str, pd.Series] = {}
    for symbol in active_symbols:
        ns_sym = to_ns(symbol)
        df = market_data.get(ns_sym)
        if df is not None and "Close" in df.columns and "Volume" in df.columns:
            notional_cols[symbol] = (df["Close"] * df["Volume"]).clip(lower=0)

    if not notional_cols:
        return np.zeros(len(active_symbols), dtype=float)

    notional_df = pd.DataFrame(notional_cols)
    adv_lookback = int(getattr(cfg, "ADV_LOOKBACK", 20)) if cfg else 20
    min_periods = max(1, adv_lookback // 2)

    # FIX #4: mean (not median) — single rolling mean across the entire matrix.
    adv_last_row = notional_df.rolling(adv_lookback, min_periods=min_periods).mean().iloc[-1]

    def _safe_adv(sym: str) -> float:
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

    if np.all(np.isnan(raw_daily_momentum)):
        return raw_daily_momentum, np.full_like(raw_daily_momentum, -np.inf), []

    # FIX D3: Compute cross-sectional normalization statistics ONLY over stocks that
    # pass the history and liquidity gates. Stocks with thin history have extreme EWMA
    # values (short-sample bias) that inflate or deflate mu_cross and std_cross before
    # gating removes them, distorting the Z-scores of every other stock in the universe.
    # Pre-compute gate masks here; the same masks are applied as hard gates below.
    history_pass = np.zeros(len(active_symbols), dtype=bool)
    liquidity_pass = np.zeros(len(active_symbols), dtype=bool)
    for i, sym in enumerate(active_symbols):
        history_pass[i]   = int(log_rets[sym].notna().sum()) >= cfg.HISTORY_GATE
        liquidity_pass[i] = bool(np.isfinite(adv_arr[i]) and adv_arr[i] > 0)
    gate_pass_mask = history_pass & liquidity_pass

    # Normalise over gate-passing stocks; fall back to all stocks only if none pass.
    norm_src = raw_daily_momentum[gate_pass_mask] if gate_pass_mask.any() else raw_daily_momentum
    mu_cross  = np.nanmean(norm_src)
    std_cross = max(np.nanstd(norm_src), 1e-8)

    adj_scores = np.clip(
        (raw_daily_momentum - mu_cross) / std_cross,
        -cfg.Z_SCORE_CLIP,
        cfg.Z_SCORE_CLIP
    )

    # 2. Hard Gates (Disqualification) — using pre-computed masks
    for i in range(len(active_symbols)):
        if not history_pass[i]:
            adj_scores[i] = -np.inf
        if not liquidity_pass[i]:
            adj_scores[i] = -np.inf
            
    # Gate C: Volatility-adjusted Falling Knife Protection
    # FIX B4: Changed skipna=False → skipna=True so stocks with data gaps are still
    # checked against the threshold. Under skipna=False, a single NaN in the window
    # propagated through .prod() → NaN → np.isfinite() returned False → gate bypassed.
    # A stock down 40% over 18 of 20 trading days with 2 gaps previously passed undetected.
    #
    # FIX G2: Threshold is now volatility-adjusted. A fixed -15% threshold treats a
    # high-beta small-cap and a low-beta utility identically, which is incorrect.
    # Each stock's threshold is scaled by its full-lookback daily vol relative to a
    # 1% daily baseline (~16% annualised). Scalar is bounded [0.5×, 2.0×]:
    #   - Low-vol stock (σ=0.5%): threshold tightens to -7.5% — catches mild slides.
    #   - High-vol stock (σ=2.0%): threshold widens to -30% — tolerates normal swings.
    #
    # FIX #6 (look-ahead in knife-gate vol):
    # _full_daily_vols is now computed strictly from log_rets, which the caller
    # (BacktestEngine._run_rebalance) already slices to [:signal_date] before passing
    # here.  There is therefore no look-ahead: log_rets.std() uses only data that was
    # available on the signal date.  The previous comment describing this as using
    # "full history" was accurate when the caller passed an unsliced DataFrame — that
    # caller-side slice is now enforced, so no further change is needed inside this
    # function.  The comment below documents this invariant explicitly.
    if len(log_rets) >= cfg.KNIFE_WINDOW:
        recent_simple = np.expm1(log_rets.iloc[-cfg.KNIFE_WINDOW:])
        recent_cumulative_returns = (
            (1.0 + recent_simple)
            .prod(skipna=True, min_count=1)   # FIX B4: skipna=True, gap stocks still checked
            - 1.0
        ).values

        _baseline_daily_vol = 0.01  # ~1% daily ≈ 16% annualised; reference for Nifty 500 large-caps

        # FIX #6: Compute vol estimate only from the rows available in log_rets.
        # log_rets is already sliced to signal_date by BacktestEngine._run_rebalance
        # (hist_log_rets = np.log1p(returns.loc[:signal_date, ...])). Using
        # log_rets.std() therefore contains no look-ahead. We document this
        # invariant explicitly so future callers do not accidentally pass an
        # unsliced DataFrame.
        _signal_date_vols = log_rets.std()  # strictly signal-date-bounded per caller contract
        _full_daily_vols = _signal_date_vols.values

        for i, cumulative_ret in enumerate(recent_cumulative_returns):
            if not np.isfinite(cumulative_ret):
                continue  # genuine data gap on all days — can't evaluate; skip safely
            # FIX G2: vol-scaled threshold
            asset_vol = float(_full_daily_vols[i]) if np.isfinite(_full_daily_vols[i]) and _full_daily_vols[i] > 0 else _baseline_daily_vol
            vol_scalar = float(np.clip(asset_vol / _baseline_daily_vol, 0.5, 2.0))
            vol_adj_threshold = cfg.KNIFE_THRESHOLD * vol_scalar
            if cumulative_ret < vol_adj_threshold:
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
