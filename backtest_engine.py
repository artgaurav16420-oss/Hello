"""
backtest_engine.py — Deterministic Walk-Forward Engine v11.48
=============================================================
Weekly rebalance cadence with full equity ledger, CVaR risk management,
and sector-diversified portfolio construction.

BUG FIXES (murder board):
- FIX-MB-SNAP: Custom-universe branch now applies the same holiday-snap
  re-keying as the parquet branch.
- FIX-MB-ADVDTYPE: _build_adv_vector uses np.issubdtype(pos.dtype, np.bool_).
- FIX-MB-HITRATE: _compute_metrics hit-rate calculation correctly handles
  unmatched sell shares.
- FIX-MB2-SORTEDUNIV: union_universe is sorted before indexing matrices.
- FIX-MB-BE-01: run_backtest now uses the warmup start date returned by
  _compute_warmup_start when building matrices from scratch, ensuring that
  standalone backtest calls get the full required warm-up history.
- FIX-MB-BE-02: build_precomputed_matrices returns column is now derived from
  the same series used for close (valuation_series) rather than always using
  close_adj. When AUTO_ADJUST_PRICES=False, signals and execution prices are
  now consistent.
- FIX-MB-BE-03: _compute_metrics CAGR now uses elapsed calendar time instead
  of a bar-count proxy, eliminating the prior off-by-one overstatement and
  making annualisation consistent across irregular trading calendars.
- FIX-MB-C-02: run_backtest now slices precomputed_matrices to [warmup_start,
  end_date] on the time axis before handing them to BacktestEngine.run.
  Previously any caller that pre-built a full TRAIN_START→TEST_END matrix
  gave signal generation look-ahead access to data beyond end_date.
- FIX-MB-H-02: _repair_suspension_gaps refactored to process all gaps against
  the original df before concatenating, eliminating incremental-frame
  contamination across multiple gaps and ensuring each gap's noise array is
  deterministically length-consistent with its seed.
- FIX-MB-M-01: _compute_metrics non-datetime turnover fallback now clamps
  years to at least 1/252 to prevent near-zero values from inflating
  annualised turnover by orders of magnitude in short test series.
- FIX-MB-M-04: Holiday-snap collision logging elevated from DEBUG to WARNING.
- FIX-MB-M-05: turnover metric documentation now explicitly states that
  1.0 turnover = one annualized full-portfolio round-trip, matching the
  optimizer friction model.
- BUG-FIX-FRAC-SLIP: Fractional share liquidations on stock splits now deduct
  one-way slippage from the cash credit, matching the treatment of all other
  sell-side transactions.
"""

from __future__ import annotations

import logging
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from momentum_engine import (
    InstitutionalRiskEngine,
    UltimateConfig,
    DEFAULT_MAX_ABSENT_PERIODS,
    OptimizationError,
    OptimizationErrorType,
    PortfolioState,
    execute_rebalance,
    compute_book_cvar,
    compute_decay_targets,
    Trade,
    activate_override_on_stress,
    absent_symbol_effective_price,
)
from signals import (
    generate_signals,
    compute_regime_score,
    SignalGenerationError,
)
from universe_manager import get_historical_universe
from shared_constants import (
    COLUMN_OPEN,
    COLUMN_HIGH,
    COLUMN_LOW,
    COLUMN_CLOSE,
    COLUMN_ADJ_CLOSE,
    COLUMN_VOLUME,
    COLUMN_STOCK_SPLITS,
)

logger = logging.getLogger(__name__)
_REBALANCE_SNAP_WINDOW_DAYS = 5
_SUSPENSION_GAP_DAYS = 30


# ─── Results container ────────────────────────────────────────────────────────

@dataclass
class BacktestResults:
    """
    Container for backtest execution results.

    Attributes:
        equity_curve (pd.Series): Time series of portfolio equity values, typically at rebalance frequency.
        trades (List[Trade]): List of all executed trades during the backtest.
        metrics (Dict): Dictionary of computed performance metrics (CAGR, Sharpe, etc.).
        rebal_log (pd.DataFrame): Log of rebalance events including regime scores and exposures.
    """
    equity_curve: pd.Series
    trades:       List[Trade]
    metrics:      Dict
    rebal_log:    pd.DataFrame


# ─── Engine ───────────────────────────────────────────────────────────────────

class BacktestEngine:
    """BacktestEngine type used by the backtesting system."""
    def __init__(self, engine: InstitutionalRiskEngine, initial_cash: float = 1_000_000):
        """
        Initialize the backtest engine with a risk engine and initial capital.

        Args:
            engine (InstitutionalRiskEngine): The core risk/optimization engine to use.
            initial_cash (float): Initial cash balance for the portfolio. Defaults to 1,000,000.
        """
        self.engine              = engine
        self.state               = PortfolioState(cash=initial_cash)
        self.state.equity_hist_cap = engine.cfg.EQUITY_HIST_CAP
        self.state.max_absent_periods = engine.cfg.MAX_ABSENT_PERIODS
        self.trades:  List[Trade]  = []
        self._eq_dates: list       = []
        self._eq_vals:  list       = []
        self._rebal_rows: list     = []

    def _reset_run_state(self) -> None:
        """
        Reset the engine's internal state to prepare for a fresh backtest run.
        Clears trade history, equity tracking, and resets the solver.
        """
        self.state.reset()
        self.engine.reset_solver()
        self._eq_dates = []
        self._eq_vals = []
        self._rebal_rows = []
        self.trades = []

    def _compute_portfolio_value_and_gross_exposure(
        self,
        valuation_close: pd.Series,
        cfg: UltimateConfig,
    ) -> tuple[float, float]:
        """Compute current portfolio value and gross exposure ratio."""
        invested_notional = sum(
            self.state.shares.get(sym, 0) * (
                float(valuation_close[sym])
                if (sym in valuation_close.index and pd.notna(valuation_close[sym]))
                else _ffill_price(self.state, sym, cfg)
            )
            for sym in self.state.shares
        )
        pv = self.state.cash + invested_notional
        gross_exposure = invested_notional / max(pv, 1e-6)
        return pv, gross_exposure

    def run(
        self,
        close:           pd.DataFrame,
        volume:          pd.DataFrame,
        returns:         pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
        start_date:      str,
        end_date:        Optional[str] = None,
        idx_df:          Optional[pd.DataFrame] = None,
        sector_map:      Optional[dict]         = None,
        open_px:         Optional[pd.DataFrame] = None,
        high_px:         Optional[pd.DataFrame] = None,
        low_px:          Optional[pd.DataFrame] = None,
        dividends:       Optional[pd.DataFrame] = None,
        splits:          Optional[pd.DataFrame] = None,
        universe_by_rebalance_date: Optional[Dict[pd.Timestamp, set[str]]] = None,
        log_rets_arr:    Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        # FIX-BE-STATE-RESET: BacktestEngine instances can be reused across
        # runs; without clearing per-run buffers, equity/trade/rebalance state
        # accumulates and contaminates subsequent outputs.
        """
        Execute the main backtest loop across the specified date range.

        Args:
            close (pd.DataFrame): Daily close prices matrix (Date x Symbol).
            volume (pd.DataFrame): Daily volume matrix (Date x Symbol).
            returns (pd.DataFrame): Daily returns matrix (Date x Symbol).
            rebalance_dates (pd.DatetimeIndex): Exact dates to perform rebalances.
            start_date (str): Backtest evaluation start date (YYYY-MM-DD).
            end_date (Optional[str]): Backtest evaluation end date (YYYY-MM-DD).
            idx_df (Optional[pd.DataFrame]): Benchmark index data for regime detection.
            sector_map (Optional[dict]): Mapping of symbols to their respective sectors.
            open_px (Optional[pd.DataFrame]): Daily open prices for execution modeling.
            high_px (Optional[pd.DataFrame]): Daily high prices.
            low_px (Optional[pd.DataFrame]): Daily low prices.
            dividends (Optional[pd.DataFrame]): Dividend cash payments (Date x Symbol).
            splits (Optional[pd.DataFrame]): Stock split ratios (Date x Symbol).
            universe_by_rebalance_date (Optional[Dict]): Point-in-time universe membership.
            log_rets_arr (Optional[np.ndarray]): Pre-computed log returns for CVaR logic.

        Returns:
            pd.DataFrame: A dataframe containing the daily 'equity' curve.
        """
        self._reset_run_state()
        # Guard against stale values left by a prior state.from_dict()
        # deserialization when reusing the same BacktestEngine instance.
        self.state.equity_hist_cap = self.engine.cfg.EQUITY_HIST_CAP
        self.state.max_absent_periods = self.engine.cfg.MAX_ABSENT_PERIODS

        if close.empty:
            logger.warning("[Backtest] Received empty close dataframe; skipping run.")
            return pd.DataFrame({"equity": pd.Series(dtype=float)})
        if log_rets_arr is None:
            log_rets_arr = np.log1p(returns).replace([np.inf, -np.inf], np.nan).values

        start_dt = pd.Timestamp(start_date)
        end_dt   = pd.Timestamp(end_date) if end_date else close.index[-1]
        symbols  = close.columns.tolist()
        close_arr = close.values
        date_to_pos = {d: i for i, d in enumerate(close.index)}

        rebal_set  = set(rebalance_dates)
        active_idx = {sym: i for i, sym in enumerate(symbols)}
        pending_splits: Dict[str, float] = {}

        for date in close.index:
            if date < start_dt or date > end_dt:
                continue

            date_pos = date_to_pos[date]
            close_t = close_arr[date_pos]
            prices_t = close_t.astype(float)

            if splits is not None and not self.engine.cfg.AUTO_ADJUST_PRICES and (date in splits.index or pending_splits):
                if date in splits.index:
                    split_row = splits.loc[date]
                    for sym in split_row.index:
                        split_val = split_row[sym]
                        if pd.notna(split_val) and float(split_val) > 0:
                            pending_splits[sym] = float(split_val)
                for sym, old_shares in list(self.state.shares.items()):
                    if old_shares <= 0 or sym not in pending_splits:
                        continue
                    split_ratio = float(pending_splits[sym])
                    if split_ratio <= 0:
                        continue

                    theoretical_new = old_shares * split_ratio
                    new_shares = int(np.floor(theoretical_new + 1e-12))
                    fractional_shares = max(0.0, theoretical_new - new_shares)
                    sym_idx = active_idx.get(sym)
                    sym_px = close_t[sym_idx] if sym_idx is not None else np.nan
                    price_now = float(sym_px) if pd.notna(sym_px) else 0.0
                    if price_now <= 0:
                        logger.warning(
                            "[Backtest] Split adjustment deferred for %s on %s due to invalid price %.4f.",
                            sym, date, price_now,
                        )
                        continue
                    if price_now > 0 and fractional_shares > 0:
                        # BUG-FIX-FRAC-SLIP: deduct one-way slippage on the
                        # fractional-share liquidation.  Previously the cash
                        # credit was at the raw price with zero transaction cost,
                        # slightly overstating CAGR in split-heavy environments.
                        _frac_slip_rate = (self.engine.cfg.ROUND_TRIP_SLIPPAGE_BPS / 2) / 10_000
                        _frac_proceeds = fractional_shares * price_now * (1.0 - _frac_slip_rate)
                        self.state.cash = round(self.state.cash + _frac_proceeds, 10)

                    self.state.shares[sym] = new_shares
                    split_denom = max(split_ratio, 1e-12)
                    if sym in self.state.entry_prices:
                        old_entry = float(self.state.entry_prices[sym])
                        self.state.entry_prices[sym] = round(old_entry / split_denom, 4)
                    else:
                        logger.warning(
                            "[Backtest] Missing entry price for %s during split adjustment on %s; "
                            "using current price as fallback instead of inferring a pre-split entry.",
                            sym,
                            date,
                        )
                        self.state.entry_prices[sym] = round(float(price_now), 4)
                    pending_splits.pop(sym, None)

            # BUG-BE-06: Rebalance runs before dividend sweep so T+0 reinvestment
            # is prevented. Dividends are credited to cash before record_eod, so
            # same-day equity includes the dividend cash.
            if date in rebal_set:
                self._run_rebalance(
                    date, close, volume, returns, symbols, prices_t,
                    idx_df, sector_map, open_px=open_px, high_px=high_px, low_px=low_px,
                    member_universe=(universe_by_rebalance_date or {}).get(pd.Timestamp(date)),
                    date_pos=date_pos, log_rets_arr=log_rets_arr,
                )

            if (
                dividends is not None
                and date in dividends.index
                and self.engine.cfg.DIVIDEND_SWEEP
                and not self.engine.cfg.AUTO_ADJUST_PRICES
            ):
                div_row = dividends.loc[date]
                for sym, shares in self.state.shares.items():
                    if shares <= 0 or sym not in div_row.index:
                        continue
                    div_val = div_row[sym]
                    if pd.notna(div_val) and float(div_val) > 0:
                        self.state.cash = round(self.state.cash + float(div_val) * shares, 10)

            valid_mask = np.isfinite(close_t)
            valid_syms = [s for s, v in zip(symbols, valid_mask) if v]
            price_dict = {s: prices_t[active_idx[s]] for s in valid_syms}
            self.state.record_eod(price_dict)
            post_pv = self.state.equity_hist[-1] if self.state.equity_hist else self.state.cash

            self._eq_dates.append(date)
            self._eq_vals.append(post_pv)

        return pd.DataFrame({"equity": pd.Series(self._eq_vals, index=self._eq_dates)})

    def _run_rebalance(
        self,
        date:       pd.Timestamp,
        close:      pd.DataFrame,
        volume:     pd.DataFrame,
        returns:    pd.DataFrame,
        symbols:    List[str],
        prices_t:   np.ndarray,
        idx_df:     Optional[pd.DataFrame],
        sector_map: Optional[dict],
        open_px:    Optional[pd.DataFrame] = None,
        high_px:    Optional[pd.DataFrame] = None,
        low_px:     Optional[pd.DataFrame] = None,
        member_universe: Optional[set[str]] = None,
        date_pos: Optional[int] = None,
        log_rets_arr: Optional[np.ndarray] = None,
    ) -> None:
        """
        Internal rebalance handler called for each rebalance date.
        Generates signals, optimizes weights, and executes trades while managing
        risk breaches and decay liquidations.

        Args:
            date (pd.Timestamp): The rebalance effective date.
            close (pd.DataFrame): Full close prices history.
            volume (pd.DataFrame): Full volume history for ADV computation.
            returns (pd.DataFrame): Full returns history.
            symbols (List[str]): List of active symbols in the current matrix.
            prices_t (np.ndarray): Array of current prices for all symbols.
            idx_df (Optional[pd.DataFrame]): Index data for regime score calculation.
            sector_map (Optional[dict]): Sector assignments for diversification.
            open_px (Optional[pd.DataFrame]): Open prices for execution modeling.
            high_px/low_px: Daily candle ranges.
            member_universe (Optional[set]): Point-in-time members for this date.
            date_pos (Optional[int]): Pre-computed integer index of the current date.
            log_rets_arr (Optional[np.ndarray]): Log returns matrix for CVaR lookbacks.
        """
        cfg = self.engine.cfg

        sym_to_global_idx = {sym: i for i, sym in enumerate(symbols)}

        active_symbols = symbols
        active_prices = prices_t
        if member_universe is not None:
            member_set = {str(sym) for sym in member_universe}
            member_set.update(self.state.shares.keys())
            active_symbols = [sym for sym in symbols if sym in member_set]
            if not active_symbols:
                return
            active_positions = [sym_to_global_idx[sym] for sym in active_symbols]
            active_prices = prices_t[active_positions]

        if date_pos is None:
            _loc = close.index.get_loc(date)
            if not isinstance(_loc, (int, np.integer)):
                raise ValueError(
                    f"Duplicate timestamp {date} detected in close index. "
                    "Deduplicate the index in build_precomputed_matrices before running backtest."
                )
            date_pos = int(_loc)
        if log_rets_arr is None:
            log_rets_arr = np.log1p(returns).replace([np.inf, -np.inf], np.nan).values

        col_to_idx = {sym: i for i, sym in enumerate(close.columns)}
        prev_idx = date_pos - 1
        if prev_idx < 0:
            return
        signal_date = close.index[prev_idx]
        active_col_indices = np.array([col_to_idx[sym] for sym in active_symbols], dtype=int)
        hist_log_rets = pd.DataFrame(
            log_rets_arr[:prev_idx + 1, active_col_indices],
            index=returns.index[:prev_idx + 1],
            columns=active_symbols,
        )

        adv_vector, close_notional = _build_adv_vector(
            active_symbols, close, volume, date, cfg=cfg, return_notional=True
        )

        # FIX-ADV-GLITCH: A one-day data gap (missing volume for a liquid stock)
        # causes _build_adv_vector to return ADV=0, which the OSQP constraint
        # translates to adv_limit=1e-9 — forcing a complete liquidation of an
        # otherwise healthy position.  For symbols currently held in the
        # portfolio, substitute a trailing ADV from the available history rather
        # than defaulting to zero.  Truly illiquid / delisted stocks are handled
        # by the absent_periods mechanism, not the ADV gate.
        _adv_lookback = int(getattr(cfg, "ADV_LOOKBACK", 20)) if cfg is not None else 20
        for _adv_i, _adv_sym in enumerate(active_symbols):
            if np.isclose(float(adv_vector[_adv_i]), 0.0, rtol=1e-9, atol=1e-12) and self.state.shares.get(_adv_sym, 0) > 0:
                if _adv_sym in close.columns and _adv_sym in volume.columns:
                    try:
                        _trail = close_notional[_adv_sym].iloc[-_adv_lookback:].clip(lower=0).dropna()
                        if not _trail.empty:
                            _fallback_adv = float(_trail.mean())
                            if _fallback_adv > 0:
                                adv_vector[_adv_i] = _fallback_adv
                                logger.debug(
                                    "[Backtest] ADV glitch: using trailing fallback "
                                    "ADV=%.0f for held position %s on %s.",
                                    _fallback_adv, _adv_sym, date,
                                )
                    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
                        # FIX-BE-BROAD-EXCEPT: narrow this fallback to known
                        # data-shape/type errors, mirroring _build_adv_vector.
                        logger.debug(
                            "[Backtest] ADV fallback failed for %s on %s; keeping ADV=0.",
                            _adv_sym,
                            date,
                        )
                        pass  # leave at 0; absent_periods will handle it

        valuation_close = close.loc[signal_date]

        valuation_prices = np.array([
            float(valuation_close[sym]) if (sym in valuation_close.index and pd.notna(valuation_close[sym]))
            else _ffill_price(self.state, sym, cfg)
            for sym in active_symbols
        ])

        pv, gross_exposure = self._compute_portfolio_value_and_gross_exposure(
            valuation_close=valuation_close,
            cfg=cfg,
        )
        prev_w_dict = _build_prev_weights(self.state, active_symbols, pv)

        idx_slice    = idx_df.loc[:signal_date] if idx_df is not None and not getattr(idx_df, "empty", False) else None
        regime_score = compute_regime_score(idx_slice, cfg=cfg, universe_close_hist=close.loc[:signal_date])

        if len(self.state.equity_hist) >= cfg.CVAR_MIN_HISTORY:
            realised_cvar = self.state.realised_cvar(min_obs=cfg.CVAR_MIN_HISTORY)
        else:
            realised_cvar = 0.0

        self.state.update_exposure(regime_score, realised_cvar, cfg, gross_exposure=gross_exposure)

        target_weights         = np.zeros(len(active_symbols))
        apply_decay            = False
        optimization_succeeded = False
        sel_idx: List[int]     = []
        _force_full_cash       = False
        soft_cvar_breach       = False

        if self.state.shares:
            book_cvar = compute_book_cvar(self.state, valuation_prices, active_symbols, hist_log_rets, cfg)
            hard_multiplier = getattr(cfg, "CVAR_HARD_BREACH_MULTIPLIER", 1.5)
            hard_breach_threshold = cfg.CVAR_DAILY_LIMIT * hard_multiplier

            if book_cvar > hard_breach_threshold:
                logger.warning(
                    "[Backtest] Book CVaR %.4f%% exceeds HARD limit %.4f%% (%.1fx) on %s — "
                    "skipping optimization, forcing immediate liquidation.",
                    book_cvar * 100, hard_breach_threshold * 100, hard_multiplier, date,
                )
                self.state.consecutive_failures += 1
                apply_decay      = True
                _force_full_cash = True
                activate_override_on_stress(self.state, cfg)

            elif book_cvar > cfg.CVAR_DAILY_LIMIT + 1e-6:
                soft_cvar_breach = True
                logger.info(
                    "[Backtest] Book CVaR soft breach %.4f%% (limit %.4f%%, hard %.4f%%) on %s — "
                    "running optimizer with CVaR constraint active.",
                    book_cvar * 100, cfg.CVAR_DAILY_LIMIT * 100, hard_breach_threshold * 100, date,
                )

        if not _force_full_cash:
            try:
                raw_daily, adj_scores, sel_idx, _gate_counts = generate_signals(
                    hist_log_rets,
                    adv_vector,
                    cfg,
                    prev_weights=prev_w_dict,
                )
            except SignalGenerationError as ve:
                logger.debug(
                    "[Backtest] generate_signals raised ValueError on %s: %s — "
                    "treating as empty universe for this bar.",
                    date, ve,
                )
                self.state.decay_rounds         = 0
                self.state.consecutive_failures = 0
                return

            if sel_idx:
                sel_syms      = [active_symbols[i] for i in sel_idx]
                sector_labels = _build_sector_labels(sel_syms, sector_map)
                prev_weights  = np.array([prev_w_dict.get(sym, 0.0) for sym in active_symbols])

                try:
                    weights_sel = self.engine.optimize(
                        expected_returns    = raw_daily[sel_idx],
                        historical_returns  = hist_log_rets[[active_symbols[i] for i in sel_idx]],
                        execution_date      = date,
                        adv_shares          = adv_vector[sel_idx],
                        prices              = valuation_prices[sel_idx],
                        portfolio_value     = pv,
                        prev_w              = prev_weights[sel_idx],
                        exposure_multiplier = self.state.exposure_multiplier,
                        sector_labels       = sector_labels,
                    )
                    target_weights[sel_idx]  = weights_sel
                    self.state.consecutive_failures = 0
                    self.state.decay_rounds  = 0
                    optimization_succeeded   = True

                except OptimizationError as oe:
                    if oe.error_type != OptimizationErrorType.DATA:
                        self.state.consecutive_failures += 1
                        logger.debug(
                            "[Backtest] Solver failure #%d on %s: %s",
                            self.state.consecutive_failures, date, oe,
                        )
                        if soft_cvar_breach:
                            logger.warning(
                                "[Backtest] Solver failure during active soft CVaR "
                                "breach on %s — bypassing 3-failure wait, "
                                "triggering immediate decay.",
                                date,
                            )
                            apply_decay = True
                        elif self.state.consecutive_failures >= 3:
                            logger.debug(
                                "[Backtest] 3 consecutive solver failures on %s — "
                                "triggering gate-filtered pro-rata liquidation.",
                                date,
                            )
                            apply_decay = True
            elif self.state.shares:
                apply_decay = True
            else:
                self.state.decay_rounds         = 0
                self.state.consecutive_failures = 0

        _exhaust_decay = False
        if apply_decay and not optimization_succeeded:
            if _force_full_cash or self.state.decay_rounds >= cfg.MAX_DECAY_ROUNDS:
                target_weights = np.zeros(len(active_symbols), dtype=float)
                logger.warning(
                    "[Backtest] %s on %s — forcing full liquidation to cash.",
                    "Book CVaR breach" if _force_full_cash else
                    f"MAX_DECAY_ROUNDS={cfg.MAX_DECAY_ROUNDS} exhausted",
                    date,
                )
                _exhaust_decay = True
                activate_override_on_stress(self.state, cfg)
            else:
                target_weights = compute_decay_targets(self.state, sel_idx, active_symbols, cfg, current_prices=valuation_prices, pv=pv)
                sel_idx_set = set(sel_idx)
                sym_to_pos  = {s: i for i, s in enumerate(active_symbols)}
                n_gated = sum(
                    1 for s in self.state.shares
                    if s in sym_to_pos and sym_to_pos[s] not in sel_idx_set
                )
                logger.debug(
                    "[Backtest] Decay round %d/%d: scaling %d gate-passing, "
                    "force-closing %d gated positions.",
                    self.state.decay_rounds + 1, cfg.MAX_DECAY_ROUNDS,
                    len(sel_idx), n_gated,
                )

        if optimization_succeeded or apply_decay:
            _T = min(len(hist_log_rets), self.engine.cfg.CVAR_LOOKBACK)
            _L = -(hist_log_rets.iloc[-_T:].reindex(columns=active_symbols, fill_value=0.0).values)

            exec_prices, open_fallback_mask = _execution_prices(
                active_symbols, date, active_prices, open_px, high_px, low_px, return_open_fallback_mask=True
            )
            if open_fallback_mask.any():
                sig_px_arr = close.values[prev_idx, active_col_indices]
                cur_px_arr = close.values[date_pos, active_col_indices]
                first_day_mask = (
                    open_fallback_mask
                    & np.isfinite(sig_px_arr)
                    & np.isfinite(cur_px_arr)
                    & (sig_px_arr == cur_px_arr)
                )
                if first_day_mask.any():
                    skipped_syms = [sym for sym, bad in zip(active_symbols, first_day_mask, strict=True) if bad]
                    logger.warning(
                        "[Backtest] Skipping first-day symbols with NaN open fallback on %s: %s",
                        date,
                        skipped_syms,
                    )
                    target_weights[first_day_mask] = 0.0

            execute_rebalance(
                self.state, target_weights, exec_prices, active_symbols, cfg,
                adv_shares     = adv_vector,
                date_context   = date,
                trade_log      = self.trades,
                apply_decay    = apply_decay and not _exhaust_decay,
                scenario_losses = None if _exhaust_decay else _L,
                force_rebalance_trades = soft_cvar_breach,
            )
            if _exhaust_decay:
                self.state.decay_rounds = 0
                self.state.consecutive_failures = 0

            self._rebal_rows.append({
                "date":               date,
                "regime_score":       round(regime_score, 4),
                "realised_cvar":      round(realised_cvar, 6),
                "exposure_multiplier":round(self.state.exposure_multiplier, 4),
                "override_active":    self.state.override_active,
                "n_positions":        len(self.state.shares),
                "apply_decay":        apply_decay,
                "forced_to_cash":     bool(_force_full_cash or _exhaust_decay),
                "force_cash_reason":  (
                    "book_cvar_breach" if _force_full_cash else
                    "max_decay_rounds" if _exhaust_decay else
                    ""
                ),
            })


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _compute_warmup_start(start_date: str, cfg: UltimateConfig) -> str:
    """
    Compute the date to start fetching data from to satisfy warm-up requirements.
    Calculates the maximum lookback needed for CVaR, momentum, and regime scores.

    Args:
        start_date (str): The requested backtest start date.
        cfg (UltimateConfig): Configuration object containing lookback parameters.

    Returns:
        str: ISO date string for the warm-up start.
    """
    halflife_slow = int(getattr(cfg, "HALFLIFE_SLOW", 63))
    cvar_lookback = int(getattr(cfg, "CVAR_LOOKBACK", 90))
    history_gate  = int(getattr(cfg, "HISTORY_GATE",  90))

    required_trading_days = max(halflife_slow * 4, cvar_lookback, history_gate)
    required_calendar_days = int(required_trading_days * 1.4) + 30
    warmup_calendar_days   = max(400, required_calendar_days)

    warmup_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=warmup_calendar_days)
    ).strftime("%Y-%m-%d")

    logger.info(
        "[Backtest] Warm-up: %d trading days needed → fetching from %s "
        "(user start_date: %s, %d calendar days of pre-history).",
        required_trading_days, warmup_start, start_date, warmup_calendar_days,
    )
    return warmup_start


def _build_prev_weights(state: PortfolioState, symbols: List[str], pv: float) -> Dict[str, float]:
    """
    Compute previous portfolio weights from current share counts and prices.

    FIX-MB2-PREVWGT: Ghost positions use haircut-adjusted price for both
    numerator and denominator, preventing overstated ghost weights.
    """
    result: Dict[str, float] = {}
    if pv <= 0:
        return result
    for sym in symbols:
        n = state.shares.get(sym, 0)
        if n <= 0:
            continue
        raw_px = state.last_known_prices.get(sym)
        if raw_px is None or not np.isfinite(raw_px) or raw_px <= 0:
            continue
        absent_n = int(state.absent_periods.get(sym, 0))
        if absent_n > 0:
            state_max_absent_periods = (
                state.max_absent_periods
                if state.max_absent_periods is not None
                else DEFAULT_MAX_ABSENT_PERIODS
            )
            px = float(absent_symbol_effective_price(raw_px, absent_n, state_max_absent_periods))
        else:
            px = float(raw_px)
        if px <= 0:
            continue
        result[sym] = (n * px) / pv
    return result


def _build_adv_vector(
    symbols: List[str],
    close: pd.DataFrame,
    volume: pd.DataFrame,
    date: pd.Timestamp,
    cfg: Optional[UltimateConfig] = None,
    return_notional: bool = False,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    """
    Compute the Average Daily Value (ADV) for a list of symbols on a given date.
    Used for liquidity gating and sizing constraints.

    Args:
        symbols (List[str]): Symbols to process.
        close (pd.DataFrame): Daily close prices.
        volume (pd.DataFrame): Daily volumes.
        date (pd.Timestamp): Target rebalance date (ADV is computed up to T-1).
        cfg (Optional[UltimateConfig]): Config for ADV_LOOKBACK.
        return_notional (bool): If True, also returns the underlying daily notional matrix.

    Returns:
        np.ndarray | tuple: ADV vector, or (ADV vector, notional DataFrame).
    """
    adv_zero_reasons: dict[str, list[str]] = {
        "missing_column": [],
        "empty_lookback": [],
        "nonfinite_mean": [],
        "exception": [],
    }
    signal_date: Optional[pd.Timestamp] = None

    if not volume.empty:
        idx = volume.index
        if date in idx:
            pos = idx.get_loc(date)
            if isinstance(pos, slice):
                pos = pos.start
            elif isinstance(pos, np.ndarray):
                if np.issubdtype(pos.dtype, np.bool_):
                    matches = np.flatnonzero(pos)
                    pos = int(matches[0]) if len(matches) else -1
                else:
                    pos = int(pos[0]) if len(pos) else -1
            if pos > 0:
                signal_date = idx[pos - 1]
            else:
                # FIX-NEW-BE-05: date is the first entry in the volume index, so
                # there is no prior bar to use as the ADV signal date.  All symbols
                # will receive ADV=0 and fail the liquidity gate, producing a
                # zero-position portfolio on this bar.  Log a warning so operators
                # can distinguish this intentional guard from a data problem.
                signal_date = None
                logger.warning(
                    "[Backtest] ADV signal_date is None on %s (first bar in volume index). "
                    "All symbols receive ADV=0; liquidity gate will suppress all positions "
                    "on this rebalance. Consider advancing start_date by one bar.",
                    date,
                )

    close_notional = pd.DataFrame(index=close.index if signal_date is None else close.loc[:signal_date].index)
    adv = np.zeros(len(symbols), dtype=float)
    if signal_date is not None:
        close_sub = close.loc[:signal_date]
        volume_sub = volume.loc[:signal_date]
        common_cols = [s for s in symbols if s in close_sub.columns and s in volume_sub.columns]
        if common_cols:
            close_slice = close_sub[common_cols]
            volume_aligned = volume_sub[common_cols].reindex(close_slice.index)
            close_notional = (close_slice * volume_aligned).clip(lower=0)
            adv_lookback = int(getattr(cfg, "ADV_LOOKBACK", 20)) if cfg is not None else 20
            adv_vector = close_notional.iloc[-adv_lookback:].mean(axis=0, skipna=True)
            adv_fallback = float(getattr(cfg, "ADV_FALLBACK", 0.0)) if cfg is not None else 0.0
            for i, sym in enumerate(symbols):
                if sym in adv_vector.index:
                    val = float(adv_vector[sym])
                    all_volume_nan = bool(volume_aligned[sym].isna().all())
                    if all_volume_nan and adv_fallback > 0:
                        adv[i] = adv_fallback
                    elif np.isfinite(val):
                        adv[i] = val
                    else:
                        adv_zero_reasons["nonfinite_mean"].append(sym)
                else:
                    adv_zero_reasons["missing_column"].append(sym)
        else:
            adv_zero_reasons["missing_column"].extend(symbols)
    else:
        adv_zero_reasons["empty_lookback"].extend(symbols)
    if any(adv_zero_reasons.values()):
        reason_counts = {k: len(v) for k, v in adv_zero_reasons.items()}
        reason_samples = {k: v[:5] for k, v in adv_zero_reasons.items() if v}
        logger.info(
            "[Backtest] ADV=0 summary on %s: counts=%s samples=%s",
            date,
            reason_counts,
            reason_samples,
        )
    adv_out = adv
    if return_notional:
        return adv_out, close_notional
    return adv_out


def _build_sector_labels(sel_syms: List[str], sector_map: Optional[dict]) -> Optional[np.ndarray]:
    """
    Map symbols to integer indices representing their respective sectors.
    Used for sector concentration constraints in the optimizer.

    Args:
        sel_syms (List[str]): List of selected symbols for the optimizer.
        sector_map (Optional[dict]): Dictionary mapping symbols to sector strings.

    Returns:
        Optional[np.ndarray]: Integer array of sector labels, or None if no map provided.
    """
    if not sector_map:
        return None
    known_sectors = sorted(s for s in set(sector_map.get(sym, "Unknown") for sym in sel_syms)
                           if s != "Unknown")
    sec_idx = {s: i for i, s in enumerate(known_sectors)}
    return np.array(
        [sec_idx.get(sector_map.get(sym, "Unknown"), -1) for sym in sel_syms],
        dtype=int,
    )


def _ffill_price(state: PortfolioState, sym: str, cfg: UltimateConfig) -> float:
    """
    Perform a forward-fill of prices for symbols with data gaps.
    Applies a haircut to the last known price based on the duration of the absence.
    """
    px = state.last_known_prices.get(sym)
    if px is None or not np.isfinite(px):
        return 0.0
    absent_n = int(state.absent_periods.get(sym, 0))
    return float(absent_symbol_effective_price(float(px), absent_n, cfg.MAX_ABSENT_PERIODS))


def _execution_prices(
    symbols: List[str],
    date: pd.Timestamp,
    close_prices: np.ndarray,
    open_px: Optional[pd.DataFrame],
    high_px: Optional[pd.DataFrame],
    low_px: Optional[pd.DataFrame],
    return_open_fallback_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Determine the optimal execution price for a set of symbols on a given date.
    Prefers Open price if available, otherwise falls back to Close.

    Args:
        symbols (List[str]): Symbols to process.
        date (pd.Timestamp): Execution date.
        close_prices (np.ndarray): Pre-calculated close price array for this date.
        open_px (Optional[pd.DataFrame]): Daily open prices.
        high_px/low_px: Daily candle ranges.
        return_open_fallback_mask (bool): If True, track which symbols used the fallback.

    Returns:
        np.ndarray | tuple: Execution price vector, optionally with the fallback mask.
    """
    exec_px = close_prices.copy()
    open_fallback_mask = np.zeros(len(symbols), dtype=bool)

    if open_px is not None and date in open_px.index:
        opens = open_px.loc[date].reindex(symbols).values.astype(float)
        open_fallback_mask = ~np.isfinite(opens)
        if open_fallback_mask.any():
            bad_syms = [sym for sym, bad in zip(symbols, open_fallback_mask, strict=True) if bad]
            logger.warning(
                "[Backtest] Non-finite open prices on %s for %s; falling back to close prices.",
                date,
                bad_syms,
            )
        exec_px = np.where(np.isfinite(opens) & (opens > 0), opens, exec_px)

    if return_open_fallback_mask:
        return exec_px, open_fallback_mask
    return exec_px


def _detect_suspension_gaps(df: pd.DataFrame, threshold_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Identify prolonged gaps in time series data exceeding a threshold.
    Used to trigger synthetic holiday/halt simulations.

    Args:
        df (pd.DataFrame): Input time series (typically OHLCV).
        threshold_days (int): Gap size in calendar days to flag.

    Returns:
        List[Tuple]: List of (start, end) timestamps for detected gaps.
    """
    if len(df) < 2:
        return []
    gap_days = df.index.to_series().diff().dt.days
    gap_end_dates = list(gap_days[gap_days > threshold_days].index)
    gaps: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for gap_end in gap_end_dates:
        end_loc = df.index.get_loc(gap_end)
        if isinstance(end_loc, slice):
            end_loc = end_loc.start
        if isinstance(end_loc, np.ndarray):
            end_loc = int(np.flatnonzero(end_loc)[0]) if end_loc.any() else 0
        if int(end_loc) <= 0:
            continue
        gap_start = df.index[int(end_loc) - 1]
        gaps.append((pd.Timestamp(gap_start), pd.Timestamp(gap_end)))
    return gaps


def _generate_synthetic_fill(
    ticker: str,
    gap_start: pd.Timestamp,
    gap_end: pd.Timestamp,
    pre_gap_df: pd.DataFrame,
    has_adj_close: bool,
) -> pd.DataFrame:
    """
    Generate synthetic price history for a trading halt using Brownian motion.
    Maintains statistical continuity of the price path during delistings or suspensions.

    Args:
        ticker (str): Symbol being processed.
        gap_start (pd.Timestamp): Last known valid price date.
        gap_end (pd.Timestamp): Resumption date (or end of window).
        pre_gap_df (pd.DataFrame): History leading up to the halt.
        has_adj_close (bool): Whether to generate adjusted close values.

    Returns:
        pd.DataFrame: Synthetic OHLCV rows to fill the gap.
    """
    gap_idx = pd.bdate_range(gap_start, gap_end)
    synth_idx = gap_idx[(gap_idx > gap_start) & (gap_idx < gap_end)].difference(pre_gap_df.index)
    if len(synth_idx) == 0:
        return pd.DataFrame()

    pre_gap_close = pre_gap_df["Close"].loc[:gap_start]
    pre_gap_rets = pre_gap_close.pct_change().dropna()
    hist_vol = float(pre_gap_rets.std()) if len(pre_gap_rets) > 10 else 0.02

    seed_material = f"{ticker}_{pd.Timestamp(gap_start).strftime('%Y%m%d')}"
    seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16) % (2**32 - 1)
    rng = np.random.RandomState(seed)
    noise_rets = rng.normal(0, hist_vol, len(synth_idx))
    walk_returns = np.cumprod(1.0 + noise_rets)

    synth = pd.DataFrame(index=synth_idx)
    close_anchor = float(pre_gap_df.loc[gap_start, COLUMN_CLOSE])
    synth[COLUMN_CLOSE] = close_anchor * walk_returns

    if has_adj_close:
        adj_anchor = pre_gap_df.loc[gap_start, COLUMN_ADJ_CLOSE]
        if pd.isna(adj_anchor):
            adj_anchor = close_anchor
        adj_close_ratio = float(adj_anchor) / max(float(close_anchor), 1e-12)
        synth[COLUMN_ADJ_CLOSE] = synth[COLUMN_CLOSE] * adj_close_ratio

    if COLUMN_VOLUME in pre_gap_df.columns:
        synth[COLUMN_VOLUME] = 0.0
    return synth


def _repair_suspension_gaps(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """In-memory suspension simulation used only during backtest runtime.

    FIX-MB-H-02: All gap fills are computed against the original df (before any
    synthetic rows are added), then all synthetic DataFrames are concatenated at
    once.  The previous approach grew 'out' incrementally, which caused:
      (a) gap_idx.difference(out.index) for a second gap to exclude dates that
          were already filled synthetically by gap 1, silently truncating the
          second gap's noise array;
      (b) the random walk for gap 2 to be seeded from gap_start_2 but only
          generate len(synth_idx_2) draws, where synth_idx_2 < full gap_idx_2
          if any dates coincided with gap 1 synthetics.
    Processing against the original df guarantees each gap's synth_idx equals
    the full business-day range (gap_start, gap_end), making the price path
    deterministic and length-consistent with the seed.
    """
    gaps = _detect_suspension_gaps(df=df, threshold_days=_SUSPENSION_GAP_DAYS)
    max_gap = int(df.index.to_series().diff().dt.days.max()) if len(df) > 1 else 0
    if max_gap <= _SUSPENSION_GAP_DAYS:
        return df.copy()

    logger.warning(
        "[Backtest] %s: Prolonged trading gap of %d days detected. "
        "Applying in-memory synthetic halt simulation.",
        ticker, max_gap,
    )

    if not gaps:
        return df.copy()

    synth_frames = []
    for gap_start, gap_end in gaps:
        synth = _generate_synthetic_fill(
            ticker=ticker,
            gap_start=gap_start,
            gap_end=gap_end,
            pre_gap_df=df,
            has_adj_close=(COLUMN_ADJ_CLOSE in df.columns),
        )
        if not synth.empty:
            synth_frames.append(synth)

    if not synth_frames:
        return df.copy()

    # Concatenate original + all synthetic frames in one pass, then sort/ffill.
    out = pd.concat([df] + synth_frames)
    out = out.sort_index().ffill()
    return out


def apply_halt_simulation(market_data: dict) -> dict:
    """
    Apply in-memory suspension repairs to all symbols in a market data dictionary.
    Ensures backtests don't 'look ahead' through data gaps or benefit from delistings.
    """
    return {k: _repair_suspension_gaps(v, k) for k, v in market_data.items()}


def _deduplicate_index(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate index entries, keeping the last one."""
    if not df.index.is_unique:
        return df[~df.index.duplicated(keep="last")]
    return df


def _extract_series_for_symbol(market_data: dict, sym: str, column: str, cfg: UltimateConfig) -> pd.Series:
    """
    Extract a specific data column for a symbol, applying forward-fills and price adjustments.

    Args:
        market_data (dict): Source dictionary of DataFrames.
        sym (str): Target symbol.
        column (str): Column name (close, open, volume, etc.).
        cfg (UltimateConfig): Config for price adjustment settings.

    Returns:
        pd.Series: Cleaned and aligned time series for the requested attribute.
    """
    key = sym if sym.endswith(".NS") else f"{sym}.NS"
    row = market_data.get(key)
    if row is None:
        row = market_data.get(sym)
    if row is None or row.empty:
        return pd.Series(dtype=float)

    max_absent_periods = max(0, int(getattr(cfg, "MAX_ABSENT_PERIODS", 10)))
    if column == "close":
        valuation_series = row.get(COLUMN_ADJ_CLOSE, row["Close"]) if cfg.AUTO_ADJUST_PRICES else row["Close"]
        return valuation_series.ffill(limit=max_absent_periods)
    if column == "close_adj":
        return row.get(COLUMN_ADJ_CLOSE, row["Close"]).ffill(limit=max_absent_periods)
    if column == "open":
        return row.get("Open", row["Close"]).ffill(limit=max_absent_periods)
    if column == "high":
        return row.get("High", row["Close"]).ffill(limit=max_absent_periods)
    if column == "low":
        return row.get("Low", row["Close"]).ffill(limit=max_absent_periods)
    if column == "dividends":
        return row.get("Dividends", pd.Series(0.0, index=row.index)).fillna(0.0)
    if column == "splits":
        return row.get("Stock Splits", pd.Series(0.0, index=row.index)).fillna(0.0)
    if column == "volume":
        return row["Volume"]
    raise ValueError(f"Unsupported column kind: {column}")


def build_precomputed_matrices(
    market_data: dict,
    cfg: Optional[UltimateConfig] = None,
    symbols: Optional[set[str]] = None,
) -> dict:
    """
    Transform raw per-symbol market data into cross-sectional matrices.
    Used by the backtest engine to access data via efficient vectorized slices.

    Args:
        market_data (dict): Dictionary mapping symbols to DataFrames.
        cfg (Optional[UltimateConfig]): Config for price adjustment logic.
        symbols (Optional[set]): Subset of symbols to include.

    Returns:
        dict: Mapping of matrix names (close, volume, etc.) to aligned DataFrames.
    """
    if cfg is None:
        cfg = UltimateConfig()

    # FIX-MB-BE-05: strip .NS suffix so caller-supplied symbols with the suffix
    # match the bare-symbol keys used internally throughout the engine.
    target_symbols = {s[:-3] if s.endswith(".NS") else s for s in (symbols or set())}
    if not target_symbols:
        for key in market_data.keys():
            if isinstance(key, str) and key.endswith(".NS"):
                target_symbols.add(key[:-3])
    close_d, close_adj_d, open_d, high_d, low_d, div_d, split_d, volume_d = {}, {}, {}, {}, {}, {}, {}, {}

    for sym in target_symbols:
        if not sym:
            continue
        close_s = _extract_series_for_symbol(market_data, sym, "close", cfg)
        if close_s.empty:
            continue
        close_d[sym] = close_s
        close_adj_d[sym] = _extract_series_for_symbol(market_data, sym, "close_adj", cfg)
        open_d[sym] = _extract_series_for_symbol(market_data, sym, "open", cfg)
        high_d[sym] = _extract_series_for_symbol(market_data, sym, "high", cfg)
        low_d[sym] = _extract_series_for_symbol(market_data, sym, "low", cfg)
        div_d[sym] = _extract_series_for_symbol(market_data, sym, "dividends", cfg)
        split_d[sym] = _extract_series_for_symbol(market_data, sym, "splits", cfg)
        volume_d[sym] = _extract_series_for_symbol(market_data, sym, "volume", cfg)

    if not close_d:
        return {}

    close = _deduplicate_index(pd.DataFrame(close_d).sort_index())
    close = close.dropna(how="all")
    shared_index = close.index
    close_adj = _deduplicate_index(pd.DataFrame(close_adj_d).sort_index())
    open_df = _deduplicate_index(pd.DataFrame(open_d).sort_index())
    high_df = _deduplicate_index(pd.DataFrame(high_d).sort_index())
    low_df = _deduplicate_index(pd.DataFrame(low_d).sort_index())
    dividends_df = _deduplicate_index(pd.DataFrame(div_d).sort_index())
    splits_df = _deduplicate_index(pd.DataFrame(split_d).sort_index())
    volume_df = _deduplicate_index(pd.DataFrame(volume_d).sort_index())
    close_adj = close_adj.reindex(shared_index)
    open_df = open_df.reindex(shared_index)
    high_df = high_df.reindex(shared_index)
    low_df = low_df.reindex(shared_index)
    dividends_df = dividends_df.reindex(shared_index).fillna(0.0)
    splits_df = splits_df.reindex(shared_index).fillna(0.0)
    volume_df = volume_df.reindex(shared_index)

    # FIX-MB-BE-02: returns derived from close (valuation_series) not always close_adj.
    returns_base = close if not cfg.AUTO_ADJUST_PRICES else close_adj

    returns = returns_base.pct_change(fill_method=None).clip(lower=-0.99)
    common_idx = returns.index.intersection(close.index)
    close = close.reindex(common_idx)
    close_adj = close_adj.reindex(common_idx)
    open_df = open_df.reindex(common_idx)
    high_df = high_df.reindex(common_idx)
    low_df = low_df.reindex(common_idx)
    dividends_df = dividends_df.reindex(common_idx).fillna(0.0)
    splits_df = splits_df.reindex(common_idx).fillna(0.0)
    volume_df = volume_df.reindex(common_idx)
    returns = returns.reindex(common_idx)
    if not close.index.equals(volume_df.index):
        raise ValueError("close and volume index mismatch")
    if not close.index.equals(returns.index):
        raise ValueError("close and returns index mismatch after alignment")

    return {
        "close": close,
        "close_adj": close_adj,
        "open": open_df,
        "high": high_df,
        "low": low_df,
        "dividends": dividends_df,
        "splits": splits_df,
        "volume": volume_df,
        "returns": returns,
    }

def _resolve_universe_by_date(
    market_data: dict,
    universe_type: Optional[str],
    universe: Optional[List[str]],
    all_target_dates: pd.DatetimeIndex,
    cfg: UltimateConfig,
) -> Tuple[set[str], Dict[pd.Timestamp, set[str]], str]:
    """
    Resolve the constituent symbols for each rebalance date in the backtest.
    Supports either a static universe or dynamic historical snapshots.

    Args:
        market_data (dict): Source market data to check for history availability.
        universe_type (Optional[str]): Name of the universe (e.g., 'nse500').
        universe (Optional[List[str]]): Explicit list of symbols (overrides universe_type).
        all_target_dates (pd.DatetimeIndex): Rebalance dates to resolve symbols for.
        cfg (UltimateConfig): Config for HISTORY_GATE gating.

    Returns:
        Tuple: (union of all symbols, constituents per date, selected universe type string).
    """
    def _normalize_symbol(sym: str) -> str:
        """Strip suffixes and ensure uppercase formatting for consistency."""
        s = str(sym).strip().upper()
        return s[:-3] if s.endswith(".NS") else s

    union_universe = {_normalize_symbol(sym) for sym in (universe or []) if str(sym).strip()}
    universe_by_rebalance_date: Dict[pd.Timestamp, set[str]] = {}
    selected_universe_type = universe_type or "nse_total"

    if union_universe:
        history_gate = int(getattr(cfg, "HISTORY_GATE", 20))
        vol_dict: Dict[str, pd.Series] = {}
        for sym in sorted(union_universe):
            key = sym if sym.endswith(".NS") else f"{sym}.NS"
            row = market_data.get(key)
            if row is None:
                row = market_data.get(sym)
            if row is not None and not row.empty and "Volume" in row.columns:
                valid_volume = row["Volume"].replace(0, np.nan).notna().astype(float)
                vol_dict[sym] = valid_volume.rolling(history_gate, min_periods=history_gate).sum()

        if vol_dict:
            cum_vol_df = pd.DataFrame(vol_dict).sort_index()
            pit_eligible_map = (
                cum_vol_df
                .reindex(cum_vol_df.index.union(all_target_dates))
                .sort_index()
                .ffill()
                .reindex(all_target_dates)
                >= history_gate
            )
            for d in all_target_dates:
                ts = pd.Timestamp(d)
                if ts not in pit_eligible_map.index:
                    universe_by_rebalance_date[ts] = set()
                    continue
                eligible_row = pit_eligible_map.loc[ts]
                pit_syms = set(eligible_row[eligible_row].index.tolist())
                universe_by_rebalance_date[ts] = pit_syms & union_universe
        else:
            for d in all_target_dates:
                universe_by_rebalance_date[pd.Timestamp(d)] = set(union_universe)
    else:
        for d in all_target_dates:
            historical_members = get_historical_universe(selected_universe_type, d)
            member_set = {_normalize_symbol(sym) for sym in (historical_members or []) if str(sym).strip()}
            universe_by_rebalance_date[pd.Timestamp(d)] = member_set
            union_universe.update(member_set)

    return union_universe, universe_by_rebalance_date, selected_universe_type


def _prepare_backtest_matrices(
    market_data: dict,
    union_universe: set[str],
    warmup_start: str,
    end_date: str,
    cfg: UltimateConfig,
    precomputed_matrices: Optional[dict],
) -> Dict[str, pd.DataFrame]:
    """
    Prepare and clip OHLCV matrices for the backtest date range.
    Ensures all matrices are aligned on the same date index and symbol columns.

    Args:
        market_data (dict): Raw source data.
        union_universe (set): Set of all symbols ever active in the test.
        warmup_start (str): Start date including warm-up period.
        end_date (str): Test end date.
        cfg (UltimateConfig): Config for building matrices from scratch if needed.
        precomputed_matrices (Optional[dict]): Pre-built matrices if available.

    Returns:
        Dict: Clipped and aligned DataFrames (close, open, high, low, etc.).
    """
    matrices = precomputed_matrices
    if matrices:
        selected = sorted(union_universe)
        _warmup_ts = pd.Timestamp(warmup_start)
        _end_ts = pd.Timestamp(end_date) if end_date else None

        def _clip(df: pd.DataFrame) -> pd.DataFrame:
            out = df.loc[_warmup_ts:] if _warmup_ts is not None else df
            if _end_ts is not None:
                out = out.loc[:_end_ts]
            return out

        def _resolve_column(df: pd.DataFrame, sym: str) -> pd.Series:
            if sym in df.columns:
                return df[sym]
            if isinstance(sym, str) and sym.endswith(".NS") and sym[:-3] in df.columns:
                return df[sym[:-3]]
            ns_sym = f"{sym}.NS" if isinstance(sym, str) and not sym.endswith(".NS") else sym
            if ns_sym in df.columns:
                return df[ns_sym]
            raise KeyError(sym)

        unresolved = [sym for sym in selected if sym not in matrices["close"].columns and
                      (not (isinstance(sym, str) and sym.endswith(".NS") and sym[:-3] in matrices["close"].columns)) and
                      (f"{sym}.NS" if isinstance(sym, str) and not sym.endswith(".NS") else sym) not in matrices["close"].columns]
        if unresolved:
            raise ValueError(
                "Precomputed matrices missing symbols for requested universe: "
                + ", ".join(sorted(map(str, unresolved)))
            )

        def _select(df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({sym: _resolve_column(df, sym) for sym in selected}, index=df.index)

        close = _select(_clip(matrices["close"]))
        open_px = _select(_clip(matrices["open"]))
        high_px = _select(_clip(matrices["high"]))
        low_px = _select(_clip(matrices["low"]))
        dividends = _select(_clip(matrices["dividends"]))
        splits = _select(_clip(matrices["splits"]))
        volume = _select(_clip(matrices["volume"]))
        returns = _select(_clip(matrices["returns"]))
    else:
        matrices = build_precomputed_matrices(
            {k: v for k, v in market_data.items() if not isinstance(v, pd.DataFrame) or v.empty or v.index[-1] >= pd.Timestamp(warmup_start)},
            cfg=cfg,
            symbols=union_universe,
        )
        if not matrices:
            raise ValueError("No valid symbols found in market_data for the dynamic historical universe.")
        close = matrices["close"]
        open_px = matrices["open"]
        high_px = matrices["high"]
        low_px = matrices["low"]
        dividends = matrices["dividends"]
        splits = matrices["splits"]
        volume = matrices["volume"]
        returns = matrices["returns"]

    common_idx = close.index.intersection(returns.index)
    return {
        "close": close.reindex(common_idx),
        "open": open_px.reindex(common_idx),
        "high": high_px.reindex(common_idx),
        "low": low_px.reindex(common_idx),
        "dividends": dividends.reindex(common_idx).fillna(0.0),
        "splits": splits.reindex(common_idx).fillna(0.0),
        "volume": volume.reindex(common_idx),
        "returns": returns.reindex(common_idx),
    }


def _snap_rebalance_dates_to_holidays(
    close_index: pd.DatetimeIndex,
    all_target_dates: pd.DatetimeIndex,
    universe_by_rebalance_date: Dict[pd.Timestamp, set[str]],
) -> Tuple[pd.DatetimeIndex, Dict[pd.Timestamp, set[str]]]:
    """
    Align theoretical rebalance dates with the nearest preceding valid trading day.
    Handles holiday collisions by merging universe members if multiple targets snap to the same day.

    Args:
        close_index (pd.DatetimeIndex): Master trading calendar from price data.
        all_target_dates (pd.DatetimeIndex): Theoretical (e.g. weekly) rebalance dates.
        universe_by_rebalance_date (Dict): Point-in-time members for theoretical dates.

    Returns:
        Tuple: (Snapped DatetimeIndex, Updated universe mapping with snapped keys).
    """
    trading_index = pd.to_datetime(close_index)
    valid = []
    for target in all_target_dates:
        lower_bound = target - pd.Timedelta(days=_REBALANCE_SNAP_WINDOW_DAYS)
        eligible = trading_index[(trading_index <= target) & (trading_index >= lower_bound)]
        if len(eligible) == 0:
            continue
        valid.append(eligible[-1])
    rebal_dates = pd.DatetimeIndex(pd.DatetimeIndex(valid).unique())

    snapped_universe: Dict[pd.Timestamp, set[str]] = {}
    for target_d, members in sorted(universe_by_rebalance_date.items()):
        lower = target_d - pd.Timedelta(days=_REBALANCE_SNAP_WINDOW_DAYS)
        eligible = trading_index[(trading_index <= target_d) & (trading_index >= lower)]
        if len(eligible) > 0:
            snapped_key = eligible[-1]
            if snapped_key in snapped_universe:
                logger.warning(
                    "[Backtest] Holiday snap collision: target %s also maps to %s; merging member sets.",
                    target_d.date(),
                    snapped_key.date(),
                )
                snapped_universe[snapped_key].update(members)
            else:
                snapped_universe[snapped_key] = set(members)
    return rebal_dates, snapped_universe if snapped_universe else universe_by_rebalance_date


def run_backtest(
    market_data: dict,
    universe_type: Optional[str] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2020-12-31",
    cfg: Optional[UltimateConfig] = None,
    sector_map: Optional[dict] = None,
    universe: Optional[List[str]] = None,
    precomputed_matrices: Optional[dict] = None,
) -> BacktestResults:
    """
    The primary entry point for executing a standalone backtest.
    Orchestrates data preparation, universe resolution, and engine execution.

    Args:
        market_data (dict): Dictionary of per-symbol DataFrames.
        universe_type (Optional[str]): Built-in universe name (e.g. 'nse500').
        start_date/end_date (str): Evaluation window.
        cfg (Optional[UltimateConfig]): Config overrides.
        sector_map (Optional[dict]): Symbol-to-sector mapping.
        universe (Optional[List]): Explicit symbol list.
        precomputed_matrices (Optional[dict]): Cached matrices to skip preparation.

    Returns:
        BacktestResults: Results container including equity curve, trades, and metrics.
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if start_ts > end_ts:
        raise ValueError(
            f"Invalid backtest date range: start_date ({start_ts.date()}) is after end_date ({end_ts.date()})."
        )

    if cfg is None:
        cfg = UltimateConfig()

    warmup_start = _compute_warmup_start(start_date, cfg)
    all_target_dates = pd.date_range(start_date, end_date, freq=cfg.REBALANCE_FREQ)

    union_universe, universe_by_rebalance_date, _selected_universe_type = _resolve_universe_by_date(
        market_data=market_data,
        universe_type=universe_type,
        universe=universe,
        all_target_dates=all_target_dates,
        cfg=cfg,
    )
    if not union_universe:
        raise RuntimeError(
            "No historical constituents resolved across requested backtest dates; "
            "verify universe snapshots or date range."
        )

    prepared = _prepare_backtest_matrices(
        market_data=market_data,
        union_universe=union_universe,
        warmup_start=warmup_start,
        end_date=end_date,
        cfg=cfg,
        precomputed_matrices=precomputed_matrices,
    )
    close = prepared["close"]
    open_px = prepared["open"]
    high_px = prepared["high"]
    low_px = prepared["low"]
    dividends = prepared["dividends"]
    splits = prepared["splits"]
    volume = prepared["volume"]
    returns = prepared["returns"]

    if returns.empty:
        raise RuntimeError("Backtest aborted: returns matrix is empty after clipping.")
    log_rets = np.log1p(returns).replace([np.inf, -np.inf], np.nan)
    log_rets_arr = log_rets.values

    if close.empty:
        raise RuntimeError(
            "Backtest aborted: no tradable symbol price history was loaded. "
            "Please verify data providers/cache connectivity and try again."
        )

    symbols_with_data = int(close.notna().any(axis=0).sum())
    if symbols_with_data == 0:
        raise RuntimeError(
            "Backtest aborted: downloaded price frames contain no usable close "
            "data for the requested universe/date range."
        )

    rebal_dates, universe_by_rebalance_date = _snap_rebalance_dates_to_holidays(
        close_index=close.index,
        all_target_dates=all_target_dates,
        universe_by_rebalance_date=universe_by_rebalance_date,
    )
    if rebal_dates.empty:
        raise RuntimeError(
            "Backtest aborted: no valid rebalance dates intersect available "
            "trading history in the requested window."
        )

    idx_df = market_data.get("^CRSLDX")
    if idx_df is None or idx_df.empty:
        idx_df = market_data.get("^NSEI")

    engine = InstitutionalRiskEngine(cfg)
    bt = BacktestEngine(engine, initial_cash=cfg.INITIAL_CAPITAL)

    bt.run(
        close, volume, returns, rebal_dates,
        start_date, end_date=end_date,
        idx_df=idx_df, sector_map=sector_map,
        open_px=open_px, high_px=high_px, low_px=low_px,
        dividends=dividends, splits=splits,
        universe_by_rebalance_date=universe_by_rebalance_date,
        log_rets_arr=log_rets_arr,
    )

    eq_daily = pd.Series(bt._eq_vals, index=bt._eq_dates)
    eq_weekly = eq_daily[eq_daily.index.isin(rebal_dates)]

    if eq_weekly.empty and not eq_daily.empty:
        logger.warning(
            "[Backtest] No rebalance dates align with equity curve index. "
            "Defaulting to daily series for metrics."
        )
        eq_weekly = eq_daily

    rebal_log = pd.DataFrame(bt._rebal_rows).set_index("date") if bt._rebal_rows else pd.DataFrame()

    return BacktestResults(
        equity_curve=eq_weekly,
        trades=bt.trades,
        metrics=_compute_metrics(eq_daily, cfg.INITIAL_CAPITAL, cfg.SIGNAL_ANNUAL_FACTOR, trades=bt.trades),
        rebal_log=rebal_log,
    )
def print_backtest_results(results: BacktestResults) -> None:
    """
    Print a formatted summary of backtest performance metrics to the console.
    Includes CAGR, Sharpe, Sortino, Max Drawdown, Calmar, and Hit Rate.
    """
    m = results.metrics
    if not m:
        print("\n  \033[31m[!] Backtest returned no metrics. Check date range.\033[0m")
        return

    print("\n  \033[1;36mBACKTEST RESULTS\033[0m")
    print(f"  \033[90m{chr(9472)*65}\033[0m")
    sortino = m.get('sortino', 0)
    sortino_display = f"{sortino:.2f}" if np.isfinite(sortino) else "N/A"
    print(
        f"  \033[1mFinal:\033[0m \033[32m₹{m.get('final', 0):,.0f}\033[0m  "
        f"\033[1mCAGR:\033[0m {m.get('cagr', 0):.2f}%  "
        f"\033[1mSharpe:\033[0m {m.get('sharpe', 0):.2f}  "
        f"\033[1mSortino:\033[0m {sortino_display}  "
        f"\033[1mMaxDD:\033[0m {m.get('max_dd', 0):.2f}%  "
        f"\033[1mCalmar:\033[0m {m.get('calmar', 0):.2f}"
    )
    print(
        f"  \033[1mHitRate:\033[0m {m.get('hit_rate', 0):.2f}%  "
        f"\033[1mTurnover:\033[0m {m.get('turnover', 0):.4f}x"
    )
    print(f"  \033[90m{chr(9472)*65}\033[0m\n")


def _calc_cagr(eq: pd.Series, initial: float) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR).
    Uses actual calendar days elapsed for higher accuracy on irregular intervals.
    """
    if eq.empty or initial <= 0 or len(eq) < 2:
        return 0.0
    first_date = pd.Timestamp(eq.index[0])
    last_date = pd.Timestamp(eq.index[-1])
    days_elapsed = (last_date - first_date).days
    years_elapsed = max(days_elapsed / 365.25, 0.001)
    if years_elapsed <= 0.001:
        return 0.0
    final = float(eq.iloc[-1])
    return ((final / initial) ** (1.0 / years_elapsed) - 1.0) * 100.0


def _calc_drawdown(eq: pd.Series) -> Tuple[pd.Series, float]:
    """
    Compute rolling drawdown percentage and the maximum drawdown over the peak.
    """
    if eq.empty:
        return pd.Series(dtype=float), 0.0
    dd = (eq / eq.cummax() - 1.0) * 100.0
    return dd, float(dd.min())


def _calc_sharpe_sortino(daily_returns: pd.Series, periods_per_year: int) -> Tuple[float, float]:
    """
    Calculate annualized Sharpe and Sortino ratios.
    Sortino uses downside deviation (negative returns only) as the risk proxy.
    """
    if len(daily_returns) <= 1 or daily_returns.std() <= 0:
        return 0.0, 0.0
    ppy = float(periods_per_year)
    sharpe = (daily_returns.mean() * ppy) / (daily_returns.std() * np.sqrt(ppy))
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (daily_returns.mean() * ppy) / (downside.std() * np.sqrt(ppy))
    else:
        sortino = np.nan
    return float(sharpe), float(sortino)


def _calc_calmar(cagr: float, max_dd: float) -> float:
    """
    Calculate the Calmar Ratio (CAGR / |Max Drawdown|).
    """
    if max_dd >= 0.0:
        return cagr
    return cagr / max(abs(max_dd), 1.0)


def _calc_hit_rate(trades: List[Trade]) -> float:
    """
    Calculate the percentage of profitable trades (round-trips).
    Uses a FIFO (First-In-First-Out) queue to match sells against buys.
    """
    round_trip_pnls: List[float] = []
    buy_queue: Dict[str, List[tuple[int, float]]] = {}

    for trade in trades:
        if trade.delta_shares == 0:
            continue
        qty = abs(int(trade.delta_shares))
        price = float(trade.exec_price)
        if trade.direction == "BUY" and trade.delta_shares > 0:
            buy_queue.setdefault(trade.symbol, []).append((qty, price))
        elif trade.direction == "SELL" and trade.delta_shares < 0:
            lots = buy_queue.setdefault(trade.symbol, [])
            remaining = qty
            while remaining > 0 and lots:
                lot_qty, lot_px = lots[0]
                matched = min(remaining, lot_qty)
                round_trip_pnls.append((price - lot_px) * matched)
                remaining -= matched
                lot_qty -= matched
                if lot_qty == 0:
                    lots.pop(0)
                else:
                    lots[0] = (lot_qty, lot_px)

    if not round_trip_pnls:
        return 0.0
    return (sum(1 for pnl in round_trip_pnls if pnl > 0) / len(round_trip_pnls)) * 100.0


def _calc_turnover(trades: List[Trade], eq: pd.Series, years: float) -> float:
    """
    Calculate annualized portfolio turnover.
    Turnover = (Avg (Buys, Sells) / Avg Equity) / Years.
    """
    if not trades or eq.empty:
        return 0.0
    buy_trades = [t for t in trades if t.direction == "BUY" and t.delta_shares > 0]
    sell_trades = [t for t in trades if t.direction == "SELL" and t.delta_shares < 0]
    total_buy_notional = sum(t.delta_shares * t.exec_price for t in buy_trades)
    total_sell_notional = sum(abs(t.delta_shares) * t.exec_price for t in sell_trades)
    avg_equity = float(eq.mean()) if len(eq) > 0 else 0.0
    if avg_equity <= 0:
        return 0.0
    turnover = ((total_buy_notional + total_sell_notional) / 2.0) / avg_equity
    clamped_years = max(years, 1.0 / 252)
    return turnover / clamped_years if clamped_years > 0 else 0.0


def _compute_metrics(
    eq: pd.Series,
    initial: float,
    periods_per_year: int = 252,
    trades: Optional[List[Trade]] = None,
) -> Dict:
    """
    Aggregate all performance metrics for a backtest run.

    Args:
        eq (pd.Series): Daily equity curve.
        initial (float): Starting capital.
        periods_per_year (int): Annualization factor (e.g. 252 for daily).
        trades (Optional[List]): Trade history for turnover and hit rate.

    Returns:
        Dict: Final performance dashboard.
    """
    if initial <= 0:
        logger.warning(
            "[Backtest] Non-positive initial capital (%.4f) supplied; returning neutral metrics.",
            initial,
        )
        return {
            "cagr": 0.0, "max_dd": 0.0,
            "final": float(eq.iloc[-1]) if not eq.empty else float(initial),
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
            "hit_rate": 0.0, "turnover": 0.0,
        }

    if eq.empty:
        return {
            "cagr": 0.0, "max_dd": 0.0, "final": initial,
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
            "hit_rate": 0.0, "turnover": 0.0,
        }

    final = float(eq.iloc[-1])
    cagr = _calc_cagr(eq, initial)
    _, max_dd = _calc_drawdown(eq)
    dr = eq.pct_change().dropna()
    sharpe, sortino = _calc_sharpe_sortino(dr, periods_per_year)
    calmar = _calc_calmar(cagr, max_dd)

    hit_rate = 0.0
    turnover = 0.0
    if trades:
        hit_rate = _calc_hit_rate(trades)
        if hasattr(eq.index, 'dtype') and np.issubdtype(eq.index.dtype, np.datetime64) and len(eq) >= 2:
            years = (eq.index[-1] - eq.index[0]).days / 365.25
        else:
            years = (len(eq) - 1) / float(periods_per_year)
        turnover = _calc_turnover(trades, eq, years)

    return {
        "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2),
        "final": round(final, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2) if np.isfinite(sortino) else sortino,
        "calmar": round(calmar, 2),
        "hit_rate": round(hit_rate, 2),
        "turnover": round(turnover, 4),
    }