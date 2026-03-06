"""
signals.py — Deterministic Regime & Momentum Kernel v11.46
=========================================================
Generates momentum Z-scores, handles liquidity filtering, calculates
macro regime penalties, and implements the Dispersion-Normalized Continuity Bonus.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from momentum_engine import UltimateConfig

logger = logging.getLogger(__name__)

class SignalGenerationError(ValueError):
    """Raised when signal generation cannot proceed due to invalid input data."""


def compute_regime_score(idx_hist: Optional[pd.DataFrame], cfg: Optional['UltimateConfig'] = None) -> float:
    """
    Computes a macroeconomic regime score bounded [0, 1].
    
    High score -> Risk-on (upward trend, low volatility)
    Low score -> Risk-off (downward trend, high volatility)
    """
    if idx_hist is None or len(idx_hist) < 253:
        logger.debug("[Signals] Insufficient index history for regime. Defaulting to 0.5")
        return 0.5
        
    close_series = idx_hist["Close"]
    
    # 1. Trend component (Distance from 200-day SMA)
    sma200 = float(close_series.rolling(window=200).mean().iloc[-1])
    last_price = float(close_series.iloc[-1])
    
    if sma200 <= 0 or not np.isfinite(sma200):
        return 0.5
        
    trend_deviation = (last_price / sma200) - 1.0
    # Sigmoid function maps deviation to [0, 1] bounded score
    trend_steepness = float(getattr(cfg, "REGIME_SIGMOID_STEEPNESS", 10.0)) if cfg else 10.0
    base_score = 1.0 / (1.0 + np.exp(-trend_steepness * trend_deviation))
    
    # 2. Volatility penalty component
    returns_20d = close_series.pct_change(fill_method=None).tail(20)
    
    if len(returns_20d) == 20:
        vol_20d = float(returns_20d.std() * np.sqrt(252))
        
        vol_floor = float(getattr(cfg, "REGIME_VOL_FLOOR", 0.18)) if cfg else 0.18
        vol_mult = float(getattr(cfg, "REGIME_VOL_MULTIPLIER", 1.5)) if cfg else 1.5
        
        all_returns = close_series.pct_change(fill_method=None).dropna()
        if len(all_returns) >= 252:
            long_term_vol = float(all_returns.tail(252).std() * np.sqrt(252))
        else:
            long_term_vol = vol_floor
            
        # If current 20-day volatility spikes above the dynamic threshold, apply penalty
        dynamic_threshold = max(vol_floor, long_term_vol * vol_mult)
        if vol_20d > dynamic_threshold:
            logger.debug("[Signals] Regime Volatility Spike detected (%.2f > %.2f). Applying penalty.", vol_20d, dynamic_threshold)
            base_score *= 0.85
            
    return round(float(base_score), 10)


def compute_single_adv(df: pd.DataFrame) -> float:
    """
    Robust calculation of Average Daily Notional Volume (ADV) for a single asset.
    Handles NaN padding and computes 20-day MA of (Close * Volume) 
    Fixes the I-10 asymmetric floor bug and unit incoherence.
    """
    try:
        if "Close" not in df.columns or "Volume" not in df.columns:
            return 0.0
            
        notional = (df["Close"] * df["Volume"]).replace(0, np.nan).ffill().fillna(0)
        if notional.empty:
            return 0.0
            
        # Take 20-day moving average strictly to ensure unit coherence against limit bounds
        adv_val = float(notional.rolling(20, min_periods=1).mean().iloc[-1])
        return adv_val if np.isfinite(adv_val) else 0.0
    except Exception as exc:
        logger.debug("[Signals] ADV calculation failed: %s", exc)
        return 0.0


def compute_adv(market_data: dict, active_symbols: List[str]) -> np.ndarray:
    """Vectorized application of compute_single_adv across active universe."""
    from momentum_engine import to_ns
    
    adv_list = []
    for symbol in active_symbols:
        ns_sym = to_ns(symbol)
        if ns_sym in market_data:
            adv_val = compute_single_adv(market_data[ns_sym])
            adv_list.append(adv_val)
        else:
            adv_list.append(0.0)
            
    return np.array(adv_list, dtype=float)


def _apply_adv_filter(tickers: List[str], cfg) -> List[str]:
    """
    Helper for Universe Manager.
    Filters a raw list of tickers down to those meeting the minimum ADV liquidity threshold.
    """
    from momentum_engine import UltimateConfig
    from data_cache import load_or_fetch
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from universe_manager import _ADV_MAX_WORKERS
    
    if cfg is None:
        cfg = UltimateConfig()
        
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=40)).strftime("%Y-%m-%d")
    
    chunk_size = 75
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    filtered_tickers = []
    min_adv_volume = cfg.MIN_ADV_CRORES * 1e7
    
    def process_chunk(chunk: List[str]) -> List[str]:
        valid_in_chunk = []
        try:
            # Load short history for liquidity validation
            data = load_or_fetch(chunk, start_date, end_date, cfg=cfg)
            for symbol in chunk:
                ns_sym = symbol + ".NS"
                if ns_sym in data:
                    df = data[ns_sym]
                    adv = compute_single_adv(df)
                    if adv >= min_adv_volume:
                        valid_in_chunk.append(symbol)
        except Exception as exc:
            logger.error("[Signals] Error processing ADV chunk: %s", exc)
        return valid_in_chunk

    logger.info("[Signals] Filtering %d tickers against ₹%dCr ADV minimum...", len(tickers), cfg.MIN_ADV_CRORES)

    with ThreadPoolExecutor(max_workers=max(1, int(_ADV_MAX_WORKERS))) as pool:
        futures = {pool.submit(process_chunk, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            filtered_tickers.extend(future.result())
            
    return filtered_tickers


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
    fast_ema = log_rets.ewm(halflife=cfg.HALFLIFE_FAST).mean().iloc[-1].values
    slow_ema = log_rets.ewm(halflife=cfg.HALFLIFE_SLOW).mean().iloc[-1].values
    
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
        # Sum of log returns represents total cumulative return over the window
        recent_cumulative_returns = log_rets.iloc[-cfg.KNIFE_WINDOW:].sum(min_count=1).values
        for i, cumulative_ret in enumerate(recent_cumulative_returns):
            if np.isfinite(cumulative_ret) and cumulative_ret < cfg.KNIFE_THRESHOLD:
                adj_scores[i] = -np.inf

    # 3. FIX: Dispersion-Normalized Continuity Bonus
    # Standardizing the bonus against the current cross-sectional dispersion
    # ensures it isn't over-dominant in tight low-vol markets or invisible in wide high-vol ones.
    valid_mask = np.isfinite(adj_scores)
    
    if prev_weights and valid_mask.any():
        # Calculate standard deviation only among assets that survived the gates
        current_dispersion = max(np.nanstd(adj_scores[valid_mask]), cfg.CONTINUITY_DISPERSION_FLOOR)
        normalized_bonus = cfg.CONTINUITY_BONUS * current_dispersion
        
        for i, sym in enumerate(active_symbols):
            if valid_mask[i] and prev_weights.get(sym, 0.0) > 0.001:
                adj_scores[i] += normalized_bonus

    # 4. Final Selection
    # Sort and pick the top N elements (ignoring those assigned -inf)
    sorted_indices = np.argsort(adj_scores)
    top_n_indices = sorted_indices[-cfg.MAX_POSITIONS:]
    
    selected_indices = [
        int(idx) for idx in top_n_indices 
        if adj_scores[idx] > -np.inf
    ]
    
    return raw_daily_momentum, adj_scores, selected_indices
