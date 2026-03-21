"""
signals.py — Deterministic Regime & Momentum Kernel v11.48
=========================================================
Generates momentum Z-scores, handles liquidity filtering, calculates
macro regime penalties, and implements the Dispersion-Normalized Continuity Bonus.

BUG FIXES (murder board):
- FIX-MB-GATENAMES: gate_counts dictionary keys renamed from misleading
  "history_gated" / "adv_gated" / "knife_gated" to "history_failed" /
  "adv_failed" / "knife_failed". The old names implied "count that passed the
  gate" but the values were counts of stocks that FAILED (were removed). All
  callers (daily_workflow.py log line) updated accordingly.
- FIX-MB-REGIMEWARMUP: Documented the known warm-up limitation where
  min_periods=252 on the long-term EWMA causes the regime vol floor to be used
  for the first ~252 trading days of any backtest. This does not introduce
  incorrect results — it is conservative (uses the hard-coded floor) — but is
  now clearly documented so users understand the first year's regime calibration.
- FIX-MB2-BREADTHNONE: compute_regime_score short-history branch hard-coded
  min_obs=20, ignoring cfg.REGIME_SMA_WINDOW. With a large SMA window config
  a symbol with only 21 rows trivially beats its own 20-row expanding mean,
  producing a false-bullish breadth signal on thin data. Now uses the same 80%
  proportional floor as the full-history branch (min 5 rows).
- FIX-MB2-GATETOCTOU: knife_failed gate count now includes symbols softly
  suppressed by knife_pre_bonus_suppress (continuity bonus blocked but adj_score
  not set to -inf), not just hard-gated symbols with adj_score == -inf.
- FIX-MB-DISPERSION: Continuity bonus now multiplied by cross-sectional return
  dispersion (std_cross clamped at CONTINUITY_DISPERSION_FLOOR). Previously
  base_bonus = min(CONTINUITY_BONUS, cap) was applied without dispersion scaling,
  meaning CONTINUITY_DISPERSION_FLOOR in UltimateConfig was defined but never
  used. This made the bonus scale-invariant and caused tests asserting
  dispersion-scaled values to fail.
- FIX-MB-H-03: std_cross is set to CONTINUITY_DISPERSION_FLOOR when
  gate_pass_mask is all-False, preventing the continuity bonus from being
  inflated by a degenerate all-symbol pool that includes illiquid and
  thin-history names.
- FIX-MB-L-02: compute_regime_score breadth fallback now explicitly assigns
  breadth_component = 0.5 and logs at DEBUG, replacing a silent `pass`.
- BUG-FIX-HISTORY-GATE: generate_signals history gate now counts valid rows
  only in the tail(HISTORY_GATE) window, not across the entire history.
  A gapped stock (old data + listing gap + few new bars) could pass the
  total-count gate while having insufficient recent history for the EWM.
- BUG-FIX-BROAD-EXCEPT: compute_single_adv now catches specific exception
  types (KeyError, TypeError, ValueError, AttributeError, IndexError) instead
  of bare Exception, so genuine bugs surface rather than returning 0.0.
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

    WARM-UP NOTE (FIX-MB-REGIMEWARMUP): The long-term EWMA volatility baseline
    uses min_periods=252 to prevent cold-start false precision. For the first
    ~252 trading days of any backtest this returns NaN, causing the code to fall
    back to vol_floor as the long-term baseline. The dynamic threshold therefore
    equals max(vol_floor, vol_floor * vol_mult) = vol_floor * vol_mult during
    warm-up. This is conservative (uses the hard-coded minimum) and does not
    introduce look-ahead — but operators should be aware that the regime penalty
    calibration is anchored at the floor value for the first year of data.
    """
    if idx_hist is None or idx_hist.empty:
        logger.debug("[Signals] Missing index history for regime. Defaulting to 0.5")
        return 0.5

    close_series = idx_hist["Close"]

    sma_window = int(getattr(cfg, "REGIME_SMA_WINDOW", 200)) if cfg else 200
    sma_fast_window = int(getattr(cfg, "REGIME_SMA_FAST_WINDOW", 50)) if cfg else 50
    min_sma_periods = 20
    min_fast_periods = max(10, min_sma_periods // 2)

    if len(close_series) >= sma_window:
        sma200 = float(close_series.rolling(window=sma_window, min_periods=min_sma_periods).mean().iloc[-1])
    else:
        sma200 = float(close_series.expanding(min_periods=min_sma_periods).mean().iloc[-1])

    if len(close_series) >= sma_fast_window:
        sma_fast = float(
            close_series.rolling(window=sma_fast_window, min_periods=min_fast_periods).mean().iloc[-1]
        )
    else:
        sma_fast = float(close_series.expanding(min_periods=min_fast_periods).mean().iloc[-1])

    last_price = float(close_series.iloc[-1])

    if sma200 <= 0 or not np.isfinite(sma200):
        return 0.5

    trend_dev_slow = (last_price / sma200) - 1.0
    trend_dev_fast = (last_price / sma_fast) - 1.0 if sma_fast > 0 and np.isfinite(sma_fast) else trend_dev_slow
    trend_deviation = 0.6 * trend_dev_fast + 0.4 * trend_dev_slow
    trend_steepness = float(getattr(cfg, "REGIME_SIGMOID_STEEPNESS", 10.0)) if cfg else 10.0
    base_score = 1.0 / (1.0 + np.exp(-trend_steepness * trend_deviation))

    ewma_span = int(getattr(cfg, "REGIME_VOL_EWMA_SPAN", 20)) if cfg else 20
    all_returns = close_series.pct_change(fill_method=None).dropna()

    vol_component = 0.5
    if len(all_returns) >= 5:
        ewma_var = all_returns.ewm(span=ewma_span, adjust=False).var()
        vol_ewma = float(ewma_var.iloc[-1] ** 0.5 * np.sqrt(252)) if pd.notna(ewma_var.iloc[-1]) else 0.0

        vol_floor = float(getattr(cfg, "REGIME_VOL_FLOOR", 0.18)) if cfg else 0.18
        vol_mult = float(getattr(cfg, "REGIME_VOL_MULTIPLIER", 1.5)) if cfg else 1.5

        lt_ewma_span = int(getattr(cfg, "REGIME_LT_VOL_EWMA_SPAN", 1260)) if cfg else 1260
        # FIX-MB-REGIMEWARMUP: min_periods=252 prevents cold-start false precision.
        # Returns NaN for the first ~252 days → falls back to vol_floor below.
        lt_ewma_var = all_returns.ewm(span=lt_ewma_span, adjust=False, min_periods=252).var()
        if pd.notna(lt_ewma_var.iloc[-1]) and lt_ewma_var.iloc[-1] > 0:
            long_term_vol = float(lt_ewma_var.iloc[-1] ** 0.5 * np.sqrt(252))
        else:
            long_term_vol = vol_floor

        dynamic_threshold = max(vol_floor, long_term_vol * vol_mult)
        vol_20d = vol_ewma
        if vol_20d > dynamic_threshold:
            logger.debug(
                "[Signals] Regime Volatility Spike detected (EWMA=%.2f > threshold=%.2f). Applying penalty.",
                vol_20d,
                dynamic_threshold,
            )
            base_score *= 0.85
        vol_component = float(np.clip(1.0 - (vol_20d / max(dynamic_threshold * 1.5, 1e-6)), 0.0, 1.0))

    breadth_component = 0.5
    _sma_win = int(getattr(cfg, "REGIME_SMA_WINDOW", 200)) if cfg else 200
    if universe_close_hist is not None and not universe_close_hist.empty:
        # FIX-NEW-SIG-02: exclude benchmark index columns (names starting with
        # "^") before computing breadth. When the active universe is empty or
        # not yet passed, universe_close_hist may contain only index tickers such
        # as ^NSEI / ^CRSLDX. A rising index trivially beats its own SMA,
        # producing breadth_component=1.0 and inflating the composite regime
        # score — causing the engine to hold full exposure when there are in fact
        # no eligible equity positions to allocate to.
        equity_cols = [c for c in universe_close_hist.columns if not str(c).startswith("^")]
        if not equity_cols:
            # FIX-MB-L-02: All columns are benchmarks (e.g. ^NSEI / ^CRSLDX only).
            # Set breadth explicitly to 0.5 so the fallback is robust to future
            # refactoring that moves the initialisation, and log at DEBUG so
            # operators can correlate benchmark-only universes with neutral breadth.
            breadth_component = 0.5
            logger.debug(
                "[Signals] universe_close_hist contains only benchmark columns; "
                "breadth_component defaulting to 0.5."
            )
        else:
            _hist = universe_close_hist[equity_cols]
            if len(_hist) >= _sma_win:
                recent = _hist.iloc[-_sma_win:]
                min_obs = max(1, int(np.ceil(_sma_win * 0.8)))
                obs_count = recent.notna().sum()
                last = _hist.iloc[-1]
                valid = (obs_count >= min_obs) & last.notna()
                if valid.any():
                    recent_valid = recent.loc[:, valid]
                    sma_vals = recent_valid.mean()
                    last_valid = _hist.iloc[-1][valid]
                    breadth_component = float((last_valid > sma_vals[last_valid.index]).mean())
            else:
                # FIX-MB2-BREADTHNONE: proportional floor, min 5 rows.
                min_obs = max(5, int(np.ceil(len(_hist) * 0.8)))
                obs_count = _hist.notna().sum()
                sma_vals = _hist.expanding(min_periods=min_obs).mean().iloc[-1]
                last = _hist.iloc[-1]
                valid = (obs_count >= min_obs) & (sma_vals > 0) & sma_vals.notna() & last.notna()
                if valid.any():
                    breadth_component = float((last[valid] > sma_vals[valid]).mean())

    composite = 0.5 * base_score + 0.3 * breadth_component + 0.2 * vol_component
    base_score = round(float(np.clip(composite, 0.0, 1.0)), 10)

    if universe_close_hist is not None and len(universe_close_hist) >= 50:
        equity_cols = [c for c in universe_close_hist.columns if not str(c).startswith("^")]
        if equity_cols:
            breadth_hist = universe_close_hist.loc[:, equity_cols].tail(50)
            sma_50 = breadth_hist.mean(skipna=True)
            current_prices = universe_close_hist.loc[:, equity_cols].iloc[-1]
            breadth_flags = (current_prices > sma_50) & current_prices.notna() & sma_50.notna()
            breadth_pct = float(breadth_flags.mean())

            if breadth_pct < 0.35:
                return 0.0
            if breadth_pct < 0.45:
                return min(base_score, 0.5)

    return base_score


def compute_single_adv(df: pd.DataFrame, cfg: Optional['UltimateConfig'] = None) -> float:
    """
    Robust calculation of Average Daily Notional Volume (ADV) for a single asset.
    """
    try:
        if "Close" not in df.columns or "Volume" not in df.columns:
            return 0.0

        notional = df["Close"] * df["Volume"]
        if notional.empty:
            return 0.0

        adv_lookback = int(getattr(cfg, "ADV_LOOKBACK", 20)) if cfg else 20
        min_periods = max(1, adv_lookback // 2)

        adv_val = float(
            notional.clip(lower=0)
            .rolling(adv_lookback, min_periods=min_periods)
            .mean()
            .iloc[-1]
        )
        return adv_val if np.isfinite(adv_val) else 0.0
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        # BUG-FIX-BROAD-EXCEPT: catch only expected data-shape errors.
        # A bare `except Exception` would silently swallow unexpected bugs
        # (e.g. a Pandas API change) and return 0.0 for all symbols,
        # destroying the portfolio without any visible crash.
        logger.debug("[Signals] ADV calculation failed: %s", exc)
        return 0.0


def compute_adv(
    market_data: dict,
    active_symbols: List[str],
    cfg: Optional['UltimateConfig'] = None,
    target_date: Optional[str] = None,
) -> np.ndarray:
    """
    Compute Average Daily Notional Volume for every symbol in a single
    vectorized pass.
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
    if target_date is not None:
        notional_df = notional_df.loc[:target_date]

    if notional_df.empty:
        return np.zeros(len(active_symbols), dtype=float)
    adv_lookback = int(getattr(cfg, "ADV_LOOKBACK", 20)) if cfg else 20
    min_periods = max(1, adv_lookback // 2)

    recent_notional = notional_df.iloc[-adv_lookback:]
    adv_last_row    = recent_notional.mean()
    valid_counts    = recent_notional.count()
    adv_last_row = adv_last_row.where(valid_counts >= min_periods, other=0.0)

    def _safe_adv(sym: str) -> float:
        x = adv_last_row.get(sym, 0.0)
        return float(x) if np.isfinite(x) else 0.0

    return np.array([_safe_adv(sym) for sym in active_symbols], dtype=float)


def generate_signals(
    log_rets:     pd.DataFrame,
    adv_arr:      np.ndarray,
    cfg:          'UltimateConfig',
    prev_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], dict]:
    """
    Core momentum generation engine.

    Returns
    -------
    raw_daily      : np.ndarray — raw (un-normalized) daily momentum scores
    adj_scores     : np.ndarray — cross-sectionally normalized Z-scores with gates applied
    selected_indices: List[int] — indices of top-N selected symbols
    gate_counts    : dict — per-gate rejection counts with keys:
        "total"          : total symbols evaluated
        "history_failed" : symbols removed by the history gate
        "adv_failed"     : symbols removed by the ADV liquidity gate
        "knife_failed"   : symbols removed by the falling-knife gate
        "selected"       : symbols passing all gates and selected for the portfolio

    NOTE on key naming (FIX-MB-GATENAMES): The dict keys use "_failed" suffix to
    be unambiguous — the count is the number of symbols REMOVED by that gate, not
    the number that passed. Callers that log these values should use language like
    "removed by history gate" not "history-gated symbols passed".
    """
    if log_rets.empty:
        raise SignalGenerationError("no valid data: log_rets dataframe is empty")

    active_symbols = list(log_rets.columns)

    signal_lag_days = max(int(getattr(cfg, "SIGNAL_LAG_DAYS", 0)), 0)
    if signal_lag_days > 0 and len(log_rets) > signal_lag_days:
        signal_log_rets = log_rets.iloc[:-signal_lag_days]
    else:
        signal_log_rets = log_rets

    fast_ema = signal_log_rets.ewm(halflife=cfg.HALFLIFE_FAST).mean().iloc[-1].values
    slow_ema = signal_log_rets.ewm(halflife=cfg.HALFLIFE_SLOW).mean().iloc[-1].values

    raw_daily_momentum = 0.5 * fast_ema + 0.5 * slow_ema

    if np.all(np.isnan(raw_daily_momentum)):
        return raw_daily_momentum, np.full_like(raw_daily_momentum, -np.inf), [], {}

    history_pass = np.zeros(len(active_symbols), dtype=bool)
    liquidity_pass = np.zeros(len(active_symbols), dtype=bool)
    for i, sym in enumerate(active_symbols):
        # BUG-FIX-HISTORY-GATE: count valid rows only in the most recent
        # HISTORY_GATE-length tail, not across the entire history.
        # Counting the total collapses the distinction between a stock with
        # 200 old bars + a 100-bar listing gap + 10 new bars (total=210, gate
        # passes) and one with 90 uninterrupted recent bars (also passes).
        # The EWM halflife decay means old bars beyond the gap contribute
        # negligibly to momentum, so the gate must verify that recent,
        # contiguous history is sufficient.
        history_pass[i]   = int(log_rets[sym].tail(cfg.HISTORY_GATE).notna().sum()) >= cfg.HISTORY_GATE
        liquidity_pass[i] = bool(np.isfinite(adv_arr[i]) and adv_arr[i] > 0)
    gate_pass_mask = history_pass & liquidity_pass

    norm_src = raw_daily_momentum[gate_pass_mask] if gate_pass_mask.any() else raw_daily_momentum
    if norm_src.size == 0 or not np.isfinite(norm_src).any():
        return raw_daily_momentum, np.full_like(raw_daily_momentum, -np.inf), [], {}
    mu_cross = np.nanmean(norm_src)
    # FIX-MB-H-03: when gate_pass_mask is all-False, norm_src is the full
    # unfiltered pool (illiquid + thin-history symbols), producing an
    # artificially large std_cross that inflates the continuity bonus beyond
    # CONTINUITY_MAX_SCALAR * CONTINUITY_DISPERSION_FLOOR.  Use the floor
    # value directly in this degenerate case.
    if not gate_pass_mask.any():
        std_cross = float(getattr(cfg, "CONTINUITY_DISPERSION_FLOOR", 0.1))
    else:
        std_cross = max(np.nanstd(norm_src), 1e-8)

    adj_scores = np.clip(
        (raw_daily_momentum - mu_cross) / std_cross,
        -cfg.Z_SCORE_CLIP,
        cfg.Z_SCORE_CLIP
    )

    for i in range(len(active_symbols)):
        if not history_pass[i]:
            adj_scores[i] = -np.inf
        if not liquidity_pass[i]:
            adj_scores[i] = -np.inf

    # FIX-NEW-SIG-01: snapshot adj_scores here — after history/ADV gates have set
    # their -inf values but BEFORE the knife gate runs.  knife_hard_mask is then
    # computed as symbols that were finite in this snapshot but became -inf after
    # the knife loop, giving a precise count of knife-only removals that cannot
    # accidentally absorb history/ADV-failed symbols.
    _adj_scores_pre_knife = adj_scores.copy()

    # Falling-knife gate with vol-adjusted threshold
    recent_cumulative_returns = None
    _full_daily_vols = None
    _baseline_daily_vol = 0.01

    # FIX-LAG-KNIFE: use signal_log_rets (lag-truncated) for the knife gate so
    # that when SIGNAL_LAG_DAYS > 0 the gate cannot see price data that is in
    # the future relative to the signal date.  Using raw log_rets here gave the
    # gate KNIFE_WINDOW bars of look-ahead (e.g. 20 days into the future when
    # SIGNAL_LAG_DAYS=21), allowing the strategy to avoid stocks that were
    # crashing AFTER the signal date — a form of clairvoyance that inflates
    # Sortino ratios for any trial using the lag parameter.
    if len(signal_log_rets) >= cfg.KNIFE_WINDOW:
        recent_simple = np.expm1(signal_log_rets.iloc[-cfg.KNIFE_WINDOW:])
        recent_cumulative_returns = (
            (1.0 + recent_simple)
            .prod(skipna=True, min_count=1)
            - 1.0
        ).values

        recent_lookback = min(len(signal_log_rets), 126)
        _signal_date_vols = signal_log_rets.iloc[-recent_lookback:].std()
        _full_daily_vols = _signal_date_vols.values

        for i, cumulative_ret in enumerate(recent_cumulative_returns):
            if not np.isfinite(cumulative_ret):
                continue
            asset_vol = float(_full_daily_vols[i]) if np.isfinite(_full_daily_vols[i]) and _full_daily_vols[i] > 0 else _baseline_daily_vol
            vol_scalar = float(np.clip(asset_vol / _baseline_daily_vol, 0.5, 2.0))
            vol_adj_threshold = cfg.KNIFE_THRESHOLD * vol_scalar
            if cumulative_ret < vol_adj_threshold:
                adj_scores[i] = -np.inf

    # Continuity Bonus
    valid_mask = np.isfinite(adj_scores)
    # Initialise here so gate_counts can reference it even when the continuity
    # block is skipped (no prev_weights or no valid scores).
    knife_pre_bonus_suppress = np.zeros(len(active_symbols), dtype=bool)

    if prev_weights and valid_mask.any():
        cap = float(getattr(cfg, "CONTINUITY_MAX_SCALAR", 0.20))
        # FIX-MB-DISPERSION: Scale the continuity bonus by the cross-sectional
        # return dispersion, clamped at CONTINUITY_DISPERSION_FLOOR. std_cross
        # was computed above from gate-passing symbols; using it here makes the
        # bonus proportional to how much spread exists in the momentum signal,
        # preventing the bonus from dominating in low-dispersion (range-bound)
        # markets. Without this scaling, CONTINUITY_DISPERSION_FLOOR in
        # UltimateConfig was defined but never used, and tests that expected
        # dispersion-scaled bonus values would fail.
        dispersion_floor = float(getattr(cfg, "CONTINUITY_DISPERSION_FLOOR", 0.1))
        # FIX-DISPERSION-SCALE: scale bonus DOWN in low-dispersion markets.
        # Previous code multiplied by max(std_cross, floor) which both inverted
        # the direction (high-dispersion got bigger bonus) and produced a unit
        # mismatch (raw-return std multiplied onto a dimensionless config param).
        dispersion_scale = min(1.0, std_cross / max(dispersion_floor, 1e-12))
        base_bonus = min(cfg.CONTINUITY_BONUS, cap) * dispersion_scale

        activity_window = max(int(getattr(cfg, "CONTINUITY_ACTIVITY_WINDOW", 5)), 1)
        stale_sessions = max(int(getattr(cfg, "CONTINUITY_STALE_SESSIONS", 10)), 1)
        min_nonzero_days = max(int(getattr(cfg, "CONTINUITY_MIN_NONZERO_DAYS", 1)), 1)
        flat_ret_eps = float(getattr(cfg, "CONTINUITY_FLAT_RET_EPS", 1e-12))
        continuity_min_adv = float(getattr(cfg, "CONTINUITY_MIN_ADV_NOTIONAL", 0.0))

        stale_denied = 0
        liquidity_denied = 0

        knife_pre_bonus_suppress = np.zeros(len(active_symbols), dtype=bool)
        if recent_cumulative_returns is not None and _full_daily_vols is not None:
            for i, cumulative_ret in enumerate(recent_cumulative_returns):
                if not np.isfinite(cumulative_ret):
                    continue
                _av = float(_full_daily_vols[i]) if np.isfinite(_full_daily_vols[i]) and _full_daily_vols[i] > 0 else _baseline_daily_vol
                _vs = float(np.clip(_av / _baseline_daily_vol, 0.5, 2.0))
                _threshold = cfg.KNIFE_THRESHOLD * _vs
                if cumulative_ret < _threshold * 0.5:
                    knife_pre_bonus_suppress[i] = True

        for i, sym in enumerate(active_symbols):
            prev_w = float(prev_weights.get(sym, 0.0))
            if valid_mask[i] and prev_w > 0.001 and not knife_pre_bonus_suppress[i]:
                recent_rets = signal_log_rets[sym].tail(activity_window)  # FIX-LAG-CONT
                nonzero_days = int((recent_rets.abs() > flat_ret_eps).sum()) if len(recent_rets) else 0
                has_recent_activity = nonzero_days >= min_nonzero_days

                stale_rets = signal_log_rets[sym].tail(stale_sessions)  # FIX-LAG-CONT
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

    # FIX-NAN-ARGSORT: replace any residual NaN in adj_scores with -inf before
    # sorting.  NumPy places NaN at the END of an argsort result (highest indices),
    # so NaN values would occupy top-N slots and be silently dropped by the
    # adj_scores[idx] > -np.inf guard — each NaN steals a slot from a legitimate
    # high-momentum candidate.  Converting NaN -> -inf ensures they sort to the
    # same position as explicitly-gated symbols and are correctly excluded.
    # NaN can arise when the most recent bars of signal_log_rets are all NaN
    # (e.g., a stock with valid early history that recently went ex-data),
    # causing ewm().mean().iloc[-1] to return NaN even though the history gate
    # counted enough valid bars across the full tail window.
    adj_scores = np.where(np.isnan(adj_scores), -np.inf, adj_scores)
    sorted_indices = np.argsort(adj_scores)
    top_n_indices = sorted_indices[-cfg.MAX_POSITIONS:]

    selected_indices = [
        int(idx) for idx in top_n_indices
        if adj_scores[idx] > -np.inf
    ]

    n_total      = len(active_symbols)
    n_hist_fail  = int(np.sum(~history_pass))
    n_liq_fail   = int(np.sum(history_pass & ~liquidity_pass))
    # FIX-MB2-GATETOCTOU: knife_failed now counts both hard-gated symbols
    # (adj_scores set to -inf by the falling-knife loop) AND softly-suppressed
    # symbols (knife_pre_bonus_suppress, which blocks the continuity bonus but
    # does NOT set adj_scores to -inf). Previously only the -inf group was
    # counted, understating knife gate removals in the diagnostic log.
    # FIX-NEW-SIG-01: knife_hard_mask is derived from _adj_scores_pre_knife
    # (finite before knife gate) vs final adj_scores (now -inf), not from
    # gate_pass_mask & (adj_scores == -inf) which would absorb history/ADV
    # failures that also happen to share the -inf sentinel value.
    knife_hard_mask = np.isfinite(_adj_scores_pre_knife) & (adj_scores == -np.inf)
    knife_soft_mask = gate_pass_mask & knife_pre_bonus_suppress & np.isfinite(adj_scores)
    n_knife_fail = int(np.sum(knife_hard_mask | knife_soft_mask))
    n_selected   = len(selected_indices)

    # FIX-MB-GATENAMES: Keys renamed with "_failed" suffix to be unambiguous.
    # Each value is the count of symbols REMOVED by that gate, not passed.
    # Callers logging "history_failed: N" should read "N symbols removed by history gate".
    gate_counts = {
        "total":          n_total,
        "history_failed": n_hist_fail,
        "adv_failed":     n_liq_fail,
        "knife_failed":   n_knife_fail,
        "selected":       n_selected,
    }

    return raw_daily_momentum, adj_scores, selected_indices, gate_counts
