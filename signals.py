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
    idx_hist must contain only fully completed trading sessions. It must NOT include any partial intraday bar.

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

    close_series = (
        idx_hist["Close"].iloc[:-1]
        if idx_hist.index[-1].date() == pd.Timestamp.today().date()
        else idx_hist["Close"]
    )

    sma_window = int(cfg.REGIME_SMA_WINDOW) if cfg else 200
    sma_fast_window = int(cfg.REGIME_SMA_FAST_WINDOW) if cfg else 50
    min_sma_periods = max(20, int(sma_window * 0.8))
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
    trend_steepness = float(cfg.REGIME_SIGMOID_STEEPNESS) if cfg else 10.0
    base_score = 1.0 / (1.0 + np.exp(-trend_steepness * trend_deviation))

    ewma_span = int(cfg.REGIME_VOL_EWMA_SPAN) if cfg else 20
    all_returns = close_series.pct_change(fill_method=None).dropna()

    vol_component = 0.5
    if len(all_returns) >= 5:
        ewma_var = all_returns.ewm(span=ewma_span, adjust=False).var()
        _raw_var = ewma_var.iloc[-1]
        vol_ewma = float(max(float(_raw_var), 0.0) ** 0.5 * np.sqrt(252)) if pd.notna(_raw_var) else 0.0

        vol_floor = float(cfg.REGIME_VOL_FLOOR) if cfg else 0.18
        vol_mult = float(cfg.REGIME_VOL_MULTIPLIER) if cfg else 1.5

        lt_ewma_span = int(cfg.REGIME_LT_VOL_EWMA_SPAN) if cfg else 1260
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
    _sma_win = int(cfg.REGIME_SMA_WINDOW) if cfg else 200
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
                recent = _hist.iloc[-(_sma_win + 1):-1]  # exclude current bar from SMA to avoid look-ahead
                min_obs = max(1, int(np.ceil(_sma_win * 0.8)))
                obs_count = recent.notna().sum()
                last = _hist.iloc[-1]
                valid = (obs_count >= min_obs) & last.notna()
                if valid.any():
                    recent_valid = recent.loc[:, valid]
                    sma_vals = recent_valid.mean()
                    last_valid = _hist.iloc[-1][valid]
                    breadth_component = float((last_valid > sma_vals.reindex(last_valid.index, fill_value=np.nan)).mean())
                else:
                    # S-02: symmetric DEBUG log for full-history path when no
                    # symbols pass the validity filter (mirrors FIX-MB-L-02 for
                    # the short-history branch). Allows operators to correlate a
                    # neutral breadth signal with data gaps in the close matrix.
                    logger.debug(
                        "[Signals] Full-history breadth: no symbols passed validity filter "
                        "(obs_count >= %d && last.notna()); breadth_component defaulting to 0.5.",
                        min_obs,
                    )
            else:
                # FIX-MB2-BREADTHNONE: proportional floor, min 5 rows.
                min_obs = max(5, int(np.ceil(len(_hist) * 0.8)))
                obs_count = _hist.notna().sum()
                sma_vals = _hist.iloc[:-1].expanding(min_periods=min_obs).mean().iloc[-1]  # exclude current bar to match long-history branch (no look-ahead)
                last = _hist.iloc[-1]
                valid = (obs_count >= min_obs) & (sma_vals > 0) & sma_vals.notna() & last.notna()
                if valid.any():
                    breadth_component = float(
    (last[valid] > sma_vals.reindex(last[valid].index, fill_value=np.nan)).mean()  # defensive reindex for column-alignment safety
)

    composite = 0.5 * base_score + 0.3 * breadth_component + 0.2 * vol_component
    regime_score = round(float(np.clip(composite, 0.0, 1.0)), 10)

    crash_override = _check_market_crash(universe_close_hist, cfg)
    if crash_override is not None:
        return min(regime_score, crash_override) if crash_override == 0.5 else crash_override

    return regime_score


def _check_market_crash(
    close_hist: Optional[pd.DataFrame],
    cfg: Optional['UltimateConfig'] = None,
) -> Optional[float]:
    """Crash override from 15-day breadth history.

    Returns 0.0 for crash conditions, 0.5 for early-warning cap, and None
    when there is no crash override.
    """
    if close_hist is None or close_hist.empty or len(close_hist) < 65:
        return None

    equity_cols = [c for c in close_hist.columns if not str(c).startswith("^")]
    if not equity_cols:
        return None

    px_slice = close_hist.loc[:, equity_cols].tail(65)
    rolling_sma50 = px_slice.rolling(window=50, min_periods=20).mean().shift(1).tail(15)

    # BUG-SIG-04: if rolling_sma50 is entirely NaN (e.g. insufficient
    # history during warmup), breadth_flags.mean() would return 0.0,
    # triggering a false bear signal. Skip the breadth override entirely.
    if not rolling_sma50.notna().any().any():
        logger.debug(
            "[Signals] rolling_sma50 is entirely NaN — skipping breadth override."
        )
        return None

    recent_px = px_slice.tail(15)
    valid_mask = recent_px.notna() & rolling_sma50.notna()
    numerator = ((recent_px > rolling_sma50) & valid_mask).sum(axis=1)
    denominator = valid_mask.sum(axis=1)
    breadth_history = numerator.div(denominator.replace(0, np.nan))
    breadth_history = breadth_history.fillna(0.0)

    current_breadth = float(breadth_history.iloc[-1])
    min_recent_breadth = float(breadth_history.min())
    if current_breadth < 0.35:
        return 0.0
    # S-03: when breadth recently pierced 0.35 but has recovered to [0.35, 0.50),
    # return 0.5 (early-warning cap) rather than 0.0 (full shutdown).  A single
    # historical spike below 0.35 in the 15-day window previously kept the
    # portfolio at zero exposure even after breadth fully recovered, because the
    # condition returned 0.0 regardless of the recovery trajectory.  Using 0.5
    # here matches the intent of the early-warning tier: the market is healing
    # but should remain cautious, not fully liquidated.
    if min_recent_breadth < 0.35 and current_breadth < 0.50:
        return 0.5
    if current_breadth < 0.45:
        return 0.5
    return None


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

        adv_lookback = int(cfg.ADV_LOOKBACK) if cfg else 20
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
        notional_df = notional_df.sort_index()
        notional_df = notional_df.loc[:pd.Timestamp(target_date)]

    if notional_df.empty:
        return np.zeros(len(active_symbols), dtype=float)
    adv_lookback = int(cfg.ADV_LOOKBACK) if cfg else 20
    min_periods = max(1, adv_lookback // 2)

    recent_notional = notional_df.iloc[-adv_lookback:]
    adv_last_row    = recent_notional.mean()
    valid_counts    = recent_notional.count()
    adv_last_row = adv_last_row.where(valid_counts >= min_periods, other=0.0)

    def _safe_adv(sym: str) -> float:
        # S-01: adv_last_row.get() can return pd.NA for symbols that were never
        # in notional_cols; pd.NA is not recognised as a float by np.isfinite
        # (raises TypeError on older NumPy).  Cast to float first, treating NA
        # as 0.0 so the ADV gate treats the symbol as illiquid.
        raw = adv_last_row.get(sym, 0.0)
        try:
            x = float(raw)
        except (TypeError, ValueError):
            return 0.0
        return x if np.isfinite(x) else 0.0

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

    signal_lag_days = max(cfg.SIGNAL_LAG_DAYS, 0)
    if signal_lag_days > 0 and len(log_rets) >= signal_lag_days:
        if len(log_rets) == signal_lag_days:
            logger.warning(
                "[Signals] Signal lag of %d days equals available rows (%d); "
                "signal_log_rets will be empty.",
                signal_lag_days, len(log_rets),
            )
        signal_log_rets = log_rets.iloc[:-signal_lag_days]
    else:
        if signal_lag_days > 0:
            logger.warning(
                "[Signals] Signal lag of %d days cannot be fully applied: "
                "only %d rows available.",
                signal_lag_days, len(log_rets),
            )
        signal_log_rets = log_rets

    # [PHASE 2 FIX] C-01: Guard against empty signal_log_rets after lag
    # truncation.  When SIGNAL_LAG_DAYS >= len(log_rets) the slice produces an
    # empty DataFrame.  Without this guard, the .iloc[-1] call below crashes
    # with IndexError ("single positional indexer is out-of-bounds").
    # Raise SignalGenerationError so callers' existing except handlers
    # (backtest_engine line 364, daily_workflow line 1047) treat this as a
    # recoverable data error and freeze state rather than crash the process.
    if signal_log_rets.empty:
        raise SignalGenerationError(
            f"no valid data after lag truncation: SIGNAL_LAG_DAYS={signal_lag_days} "
            f"consumed all {len(log_rets)} available rows"
        )

    fast_ema = signal_log_rets.ewm(halflife=cfg.HALFLIFE_FAST).mean().iloc[-1].values
    slow_ema = signal_log_rets.ewm(halflife=cfg.HALFLIFE_SLOW).mean().iloc[-1].values

    raw_daily_momentum = 0.5 * fast_ema + 0.5 * slow_ema

    if np.all(np.isnan(raw_daily_momentum)):
        return raw_daily_momentum, np.full_like(raw_daily_momentum, -np.inf), [], {
            "total": len(active_symbols),
            "history_failed": 0,
            "adv_failed": 0,
            "knife_failed": 0,
            "selected": 0,
        }

    history_pass = (signal_log_rets[active_symbols].tail(cfg.HISTORY_GATE).notna().sum(axis=0) >= cfg.HISTORY_GATE).values
    liquidity_pass = np.isfinite(adv_arr) & (adv_arr > 0)
    gate_pass_mask = history_pass & liquidity_pass

    norm_src = raw_daily_momentum[gate_pass_mask] if gate_pass_mask.any() else raw_daily_momentum
    if norm_src.size == 0 or not np.isfinite(norm_src).any():
        return raw_daily_momentum, np.full_like(raw_daily_momentum, -np.inf), [], {
            "total": len(active_symbols),
            "history_failed": int(np.sum(~history_pass)),
            "adv_failed": int(np.sum(history_pass & ~liquidity_pass)),
            "knife_failed": 0,
            "selected": 0,
        }
    mu_cross = np.nanmean(norm_src)
    # FIX-MB-H-03: when gate_pass_mask is all-False, norm_src is the full
    # unfiltered pool (illiquid + thin-history symbols), producing an
    # artificially large std_cross that inflates the continuity bonus beyond
    # CONTINUITY_MAX_SCALAR * CONTINUITY_DISPERSION_FLOOR.  Use the floor
    # value directly in this degenerate case.
    if not gate_pass_mask.any():
        std_cross = cfg.CONTINUITY_DISPERSION_FLOOR
    else:
        std_cross = max(np.nanstd(norm_src), 1e-8)

    adj_scores = np.clip(
        (raw_daily_momentum - mu_cross) / std_cross,
        -cfg.Z_SCORE_CLIP,
        cfg.Z_SCORE_CLIP
    )

    adj_scores[~gate_pass_mask] = -np.inf

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
    if cfg.KNIFE_WINDOW > 0 and len(signal_log_rets) >= cfg.KNIFE_WINDOW:
        _knife_min_count = max(1, cfg.KNIFE_WINDOW // 2)
        recent_cumulative_returns = np.expm1(
            signal_log_rets.iloc[-cfg.KNIFE_WINDOW:]
            .sum(skipna=True, min_count=_knife_min_count)
        ).values

        recent_lookback = min(len(signal_log_rets), 126)
        _signal_date_vols = signal_log_rets.iloc[-recent_lookback:].std()
        _full_daily_vols = _signal_date_vols.values

        safe_vols = np.where(
            np.isfinite(_full_daily_vols) & (_full_daily_vols > 0),
            _full_daily_vols,
            _baseline_daily_vol
        )
        vol_scalars = np.clip(safe_vols / _baseline_daily_vol, 0.5, 2.0)
        thresholds = cfg.KNIFE_THRESHOLD * vol_scalars
        finite_mask = np.isfinite(recent_cumulative_returns)
        knife_hard_mask = gate_pass_mask & finite_mask & (recent_cumulative_returns < thresholds)
        adj_scores = np.where(knife_hard_mask, -np.inf, adj_scores)

    # Continuity Bonus
    valid_mask = np.isfinite(adj_scores)
    # Initialise here so gate_counts can reference it even when the continuity
    # block is skipped (no prev_weights or no valid scores).
    # NOTE (FIX-MB-SIG-01): When prev_weights is None (first scan),
    # knife_pre_bonus_suppress is all-zeros by design — no prior positions means
    # no continuity bonus to suppress.  Consequently knife_failed counts only
    # hard-gated symbols on the first scan; this is correct expected behaviour.
    knife_pre_bonus_suppress = np.zeros(len(active_symbols), dtype=bool)

    if prev_weights and valid_mask.any():
        cap = cfg.CONTINUITY_MAX_SCALAR
        # FIX-MB-DISPERSION: Scale the continuity bonus by the cross-sectional
        # return dispersion, clamped at CONTINUITY_DISPERSION_FLOOR. std_cross
        # was computed above from gate-passing symbols; using it here makes the
        # bonus proportional to how much spread exists in the momentum signal,
        # preventing the bonus from dominating in low-dispersion (range-bound)
        # markets. Without this scaling, CONTINUITY_DISPERSION_FLOOR in
        # UltimateConfig was defined but never used, and tests that expected
        # dispersion-scaled bonus values would fail.
        dispersion_floor = cfg.CONTINUITY_DISPERSION_FLOOR
        # FIX-DISPERSION-SCALE: scale bonus DOWN in low-dispersion markets.
        # Previous code multiplied by max(std_cross, floor) which both inverted
        # the direction (high-dispersion got bigger bonus) and produced a unit
        # mismatch (raw-return std multiplied onto a dimensionless config param).
        dispersion_scale = min(1.0, std_cross / max(dispersion_floor, 1e-12))
        base_bonus = min(cfg.CONTINUITY_BONUS, cap) * dispersion_scale

        activity_window = max(cfg.CONTINUITY_ACTIVITY_WINDOW, 1)
        stale_sessions = max(cfg.CONTINUITY_STALE_SESSIONS, 1)
        min_nonzero_days = max(cfg.CONTINUITY_MIN_NONZERO_DAYS, 1)
        flat_ret_eps = cfg.CONTINUITY_FLAT_RET_EPS
        continuity_min_adv = cfg.CONTINUITY_MIN_ADV_NOTIONAL

        stale_denied = 0
        liquidity_denied = 0

        knife_pre_bonus_suppress = np.zeros(len(active_symbols), dtype=bool)
        if recent_cumulative_returns is not None and _full_daily_vols is not None:
            for i, cumulative_ret in enumerate(recent_cumulative_returns):
                if not np.isfinite(cumulative_ret):
                    continue
                vol_adj_threshold = thresholds[i]
                if cumulative_ret < vol_adj_threshold * 0.5:
                    knife_pre_bonus_suppress[i] = True

        _max_win = max(activity_window, stale_sessions)
        _recent_window_df = signal_log_rets[active_symbols].tail(_max_win)
        # --- Vectorized continuity bonus (replaces serial for-loop) ---
        recent_rets_full = _recent_window_df.tail(activity_window)
        has_recent_activity = (recent_rets_full.abs() > flat_ret_eps).sum(axis=0).values >= min_nonzero_days  # noqa: F841 (retained for future use)

        stale_rets_full = _recent_window_df.tail(stale_sessions)
        # is_stale: all stale_sessions rows are present, non-NaN, and flat
        if len(stale_rets_full) == stale_sessions:
            is_stale = (
                stale_rets_full.notna().all(axis=0)
                & (stale_rets_full.abs() <= flat_ret_eps).all(axis=0)
            ).values
        else:
            is_stale = np.zeros(len(active_symbols), dtype=bool)

        passes_continuity_liquidity = np.isfinite(adv_arr) & (adv_arr >= continuity_min_adv)

        prev_w_arr = np.array(
            [float(prev_weights.get(sym, 0.0)) for sym in active_symbols], dtype=float
        )

        candidate_mask = valid_mask & (prev_w_arr > 0.001) & ~knife_pre_bonus_suppress

        stale_denied    = int(np.sum(candidate_mask & is_stale))
        liquidity_denied = int(np.sum(candidate_mask & ~passes_continuity_liquidity))

        bonus_mask = candidate_mask & ~is_stale & passes_continuity_liquidity
        decay = np.clip(
            prev_w_arr / max(cfg.CONTINUITY_MAX_HOLD_WEIGHT, 1e-6), 0.25, 1.0
        )
        adj_scores[bonus_mask] += base_bonus * decay[bonus_mask]
        # --- End vectorized continuity bonus ---

        if stale_denied or liquidity_denied:
            logger.debug(
                "[Signals] Continuity denied for %d stale and %d illiquid symbols.",
                stale_denied,
                liquidity_denied,
            )

    _adj_scores_post_knife = adj_scores.copy()

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
    _n_pos = int(cfg.MAX_POSITIONS)
    top_n_indices = sorted_indices[-_n_pos:] if _n_pos > 0 else np.array([], dtype=int)

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
    knife_soft_mask = gate_pass_mask & knife_pre_bonus_suppress & np.isfinite(_adj_scores_post_knife)
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
