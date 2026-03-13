"""
backtest_engine.py — Deterministic Walk-Forward Engine v11.48
=============================================================
Weekly rebalance cadence with full equity ledger, CVaR risk management,
and sector-diversified portfolio construction.

Now properly integrated with Impact-Aligned Execution and Historical Constituents
to eliminate Survivorship Bias. Fixes Look-Ahead PV/ADV computation biases.

FIX #5 (snapped_universe survivorship bias on holiday clusters):
When two consecutive calendar target dates (e.g. Fri Dec 31 + Mon Jan 1) both
snap to the same last trading day (e.g. Thu Dec 30), the previous code took the
UNION of their member sets. Union is wrong: it can include stocks that were added
to the index on Jan 1 and back-fill them into the Dec 30 snapshot, introducing
forward-looking survivorship bias on holiday weekends.

Fix: use the EARLIEST member set for the snapped date. When multiple target dates
collapse to the same trading day, the constituent list from the earliest target
date is the most conservative (no future additions).
"""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from momentum_engine import (
    InstitutionalRiskEngine,
    UltimateConfig,
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

logger = logging.getLogger(__name__)
_REBALANCE_SNAP_WINDOW_DAYS = 5
_SUSPENSION_GAP_DAYS = 30

# ─── Results container ────────────────────────────────────────────────────────

@dataclass
class BacktestResults:
    equity_curve: pd.Series
    trades:       List[Trade]
    metrics:      Dict
    rebal_log:    pd.DataFrame   # one row per rebalance date

# ─── Engine ───────────────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(self, engine: InstitutionalRiskEngine, initial_cash: float = 1_000_000):
        self.engine              = engine
        self.state               = PortfolioState(cash=initial_cash)
        self.state.equity_hist_cap = engine.cfg.EQUITY_HIST_CAP
        self.state.max_absent_periods = engine.cfg.MAX_ABSENT_PERIODS
        self.trades:  List[Trade]  = []
        self._eq_dates: list       = []
        self._eq_vals:  list       = []
        self._rebal_rows: list     = []

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
    ) -> pd.DataFrame:
        if close.empty:
            logger.warning("[Backtest] Received empty close dataframe; skipping run.")
            return pd.DataFrame({"equity": pd.Series(dtype=float)})

        start_dt = pd.Timestamp(start_date)
        end_dt   = pd.Timestamp(end_date) if end_date else close.index[-1]
        symbols  = list(close.columns)

        rebal_set  = set(rebalance_dates)
        active_idx = {sym: i for i, sym in enumerate(symbols)}

        for date in close.index:
            if date < start_dt or date > end_dt:
                continue

            close_t  = close.loc[date]
            prices_t = close_t.values.astype(float)

            if splits is not None and date in splits.index and not self.engine.cfg.AUTO_ADJUST_PRICES:
                split_row = splits.loc[date]
                for sym, old_shares in list(self.state.shares.items()):
                    if old_shares <= 0 or sym not in split_row.index:
                        continue
                    split_ratio = float(split_row[sym]) if pd.notna(split_row[sym]) else 0.0
                    if split_ratio <= 0:
                        continue

                    theoretical_new = old_shares * split_ratio
                    new_shares = int(np.floor(theoretical_new + 1e-12))
                    fractional_shares = max(0.0, theoretical_new - new_shares)
                    price_now = float(close_t[sym]) if sym in close_t.index and pd.notna(close_t[sym]) else 0.0
                    if price_now > 0 and fractional_shares > 0:
                        self.state.cash = round(self.state.cash + fractional_shares * price_now, 10)

                    self.state.shares[sym] = new_shares
                    old_entry = float(self.state.entry_prices.get(sym, price_now * max(split_ratio, 1e-12)))
                    self.state.entry_prices[sym] = round(old_entry / max(split_ratio, 1e-12), 4)

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

            if date in rebal_set:
                self._run_rebalance(
                    date, close, volume, returns, symbols, prices_t,
                    idx_df, sector_map, open_px=open_px, high_px=high_px, low_px=low_px,
                    member_universe=(universe_by_rebalance_date or {}).get(pd.Timestamp(date)),
                )

            price_dict = {
                sym: prices_t[active_idx[sym]]
                for sym in symbols
                if pd.notna(close_t[sym])
            }
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
    ) -> None:
        cfg = self.engine.cfg

        sym_to_global_idx = {sym: i for i, sym in enumerate(symbols)}

        active_symbols = symbols
        active_prices = prices_t
        if member_universe is not None:
            member_set = {str(sym) for sym in member_universe}
            active_symbols = [sym for sym in symbols if sym in member_set]
            if not active_symbols:
                return
            active_positions = [sym_to_global_idx[sym] for sym in active_symbols]
            active_prices = prices_t[active_positions]

        prev_idx = close.index.get_loc(date) - 1
        if prev_idx < 0:
            return
        signal_date = close.index[prev_idx]

        hist_log_rets = (
            np.log1p(returns.loc[:signal_date, active_symbols])
            .replace([np.inf, -np.inf], np.nan)
        )

        adv_vector = _build_adv_vector(active_symbols, close, volume, date, cfg=cfg)

        # Value the pre-trade portfolio using last fully-observed prices (T-1 close).
        # This avoids lookahead when we size orders that execute on the current bar.
        valuation_close = close.loc[signal_date]
        
        # FIX (Phase 1 & 2): Create an aligned vector of strictly T-1 prices.
        # This isolates the risk checks and optimizer from looking at T+0 data.
        valuation_prices = np.array([
            float(valuation_close[sym]) if (sym in valuation_close.index and pd.notna(valuation_close[sym]))
            else _ffill_price(self.state, sym, cfg)
            for sym in active_symbols
        ])
        
        pv = self.state.cash + sum(
            self.state.shares.get(sym, 0) * (
                float(valuation_close[sym])
                if (sym in close.columns and pd.notna(valuation_close[sym]))
                else _ffill_price(self.state, sym, cfg)
            )
            for sym in self.state.shares
        )
        prev_w_dict = _build_prev_weights(self.state, active_symbols, pv)

        _idx_ok      = idx_df is not None and not (hasattr(idx_df, "empty") and idx_df.empty)
        idx_slice    = idx_df.loc[:signal_date] if _idx_ok else None
        regime_score = compute_regime_score(idx_slice, cfg=cfg, universe_close_hist=close.loc[:signal_date])

        if len(self.state.equity_hist) >= cfg.CVAR_MIN_HISTORY:
            realised_cvar = self.state.realised_cvar(min_obs=cfg.CVAR_MIN_HISTORY)
        else:
            realised_cvar = 0.0

        gross_exposure = sum(
            self.state.shares.get(sym, 0) * (
                float(valuation_close[sym])
                if (sym in valuation_close.index and pd.notna(valuation_close[sym]))
                else _ffill_price(self.state, sym, cfg)
            )
            for sym in self.state.shares
        ) / max(pv, 1e-6)

        self.state.update_exposure(regime_score, realised_cvar, cfg, gross_exposure=gross_exposure)

        target_weights         = np.zeros(len(active_symbols))
        apply_decay            = False
        optimization_succeeded = False
        sel_idx: List[int]     = []
        _force_full_cash       = False
        soft_cvar_breach       = False

        # ── Book CVaR screen ──────────────────────────────────────────────────
        if self.state.shares:
            # FIX (Phase 1 Look-ahead): Use valuation_prices (T-1), NOT active_prices (T+0).
            # Passing active_prices here allowed the risk engine to magically foresee
            # intraday crashes before optimization execution, inflating results.
            book_cvar = compute_book_cvar(self.state, valuation_prices, active_symbols, hist_log_rets, cfg)
            hard_multiplier = getattr(cfg, "CVAR_HARD_BREACH_MULTIPLIER", 1.5)
            hard_breach_threshold = cfg.CVAR_DAILY_LIMIT * hard_multiplier

            if book_cvar > hard_breach_threshold:
                # HARD breach: liquidate immediately.
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
                # SOFT breach: elevated but manageable. Let the QP handle it.
                soft_cvar_breach = True
                logger.info(
                    "[Backtest] Book CVaR soft breach %.4f%% (limit %.4f%%, hard %.4f%%) on %s — "
                    "running optimizer with CVaR constraint active.",
                    book_cvar * 100, cfg.CVAR_DAILY_LIMIT * 100, hard_breach_threshold * 100, date,
                )

        # ── Signal generation + optimization ─────────────────────────────────
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
                        # FIX (Phase 7 related): Supply T-1 valuation prices for constraint sizing.
                        # Do not leak execution logic into the constraint solver.
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
                        # FIX (Risk-Breach Paralysis): if a soft CVaR breach is
                        # already active and the optimizer fails (KKT infeasibility
                        # or post-check rejection), the portfolio is in a
                        # mathematically unsafe state RIGHT NOW.  Waiting for 3
                        # consecutive failures before triggering decay means the
                        # position could remain over-risk for up to 3 rebalance
                        # periods.  Bypass the counter and force decay immediately.
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
            else:
                if self.state.shares:
                    apply_decay = True
                else:
                    self.state.decay_rounds         = 0
                    self.state.consecutive_failures = 0

        # ── Gate-filtered decay target computation ────────────────────────────
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
                target_weights = compute_decay_targets(self.state, sel_idx, active_symbols, cfg)
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
            
            # Executions correctly remain grounded in the reality of T+0 data (e.g. Open/VWAP/Close).
            exec_prices = _execution_prices(active_symbols, date, active_prices, open_px, high_px, low_px)
            
            execute_rebalance(
                self.state, target_weights, exec_prices, active_symbols, cfg,
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
            })


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_prev_weights(state: PortfolioState, symbols: List[str], pv: float) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if pv <= 0:
        return result
    for sym in symbols:
        n = state.shares.get(sym, 0)
        if n <= 0:
            continue
        px = state.last_known_prices.get(sym)
        if px is None or not np.isfinite(px) or px <= 0:
            continue
        result[sym] = (n * px) / pv
    return result


def _build_adv_vector(
    symbols: List[str],
    close: pd.DataFrame,
    volume: pd.DataFrame,
    date: pd.Timestamp,
    cfg: Optional[UltimateConfig] = None,
) -> np.ndarray:
    adv = []
    signal_date: Optional[pd.Timestamp] = None

    if not volume.empty:
        idx = volume.index
        if date in idx:
            pos = idx.get_loc(date)
            if isinstance(pos, slice):
                pos = pos.start
            elif isinstance(pos, np.ndarray):
                if pos.dtype == bool:
                    matches = np.flatnonzero(pos)
                    pos = int(matches[0]) if len(matches) else -1
                else:
                    pos = int(pos[0]) if len(pos) else -1
            if pos > 0:
                signal_date = idx[pos - 1]
            else:
                signal_date = None

    for sym in symbols:
        if sym in volume.columns and sym in close.columns and signal_date is not None:
            try:
                c_series = close.loc[:signal_date, sym]
                v_series = volume.loc[:signal_date, sym]
                notional = (c_series * v_series).clip(lower=0).dropna()
                adv_lookback = int(getattr(cfg, "ADV_LOOKBACK", 20)) if cfg is not None else 20
                lookback = notional.tail(adv_lookback)
                if lookback.empty:
                    adv.append(0.0)
                else:
                    # FIX #4 (backtest_engine mirror): Use mean not median, consistent
                    # with the fix applied to signals.compute_adv.
                    val = float(lookback.mean())
                    adv.append(val if np.isfinite(val) else 0.0)
            except Exception:
                adv.append(0.0)
        else:
            adv.append(0.0)
    return np.array(adv, dtype=float)


def _build_sector_labels(sel_syms: List[str], sector_map: Optional[dict]) -> Optional[np.ndarray]:
    if not sector_map:
        return None
    # FIX (Sector-Cap Strangulation): symbols whose sector could not be fetched
    # from universe_manager default to the string "Unknown".  If we assign them a
    # regular integer label, the OSQP constraint builder groups ALL unknown-sector
    # assets into one synthetic super-sector and applies MAX_SECTOR_WEIGHT to their
    # combined allocation.  When many mid-cap names have missing sector data this
    # creates an invisible allocation cap that strangulates the optimizer.
    #
    # Sentinel -1 is used for "Unknown" so that the constraint builder loop can
    # detect and skip it, allowing the global budget constraint alone to govern
    # those assets.  Known sectors receive non-negative sequential IDs as before.
    known_sectors = sorted(s for s in set(sector_map.get(sym, "Unknown") for sym in sel_syms)
                           if s != "Unknown")
    sec_idx = {s: i for i, s in enumerate(known_sectors)}
    return np.array(
        [sec_idx.get(sector_map.get(sym, "Unknown"), -1) for sym in sel_syms],
        dtype=int,
    )

def _ffill_price(state: PortfolioState, sym: str, cfg: UltimateConfig) -> float:
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
) -> np.ndarray:
    if open_px is not None and date in open_px.index:
        opens = open_px.loc[date].reindex(symbols).values.astype(float)
        if np.isfinite(opens).any():
            return np.where(np.isfinite(opens) & (opens > 0), opens, close_prices)
    if high_px is not None and low_px is not None and date in high_px.index and date in low_px.index:
        highs = high_px.loc[date].reindex(symbols).values.astype(float)
        lows = low_px.loc[date].reindex(symbols).values.astype(float)
        vwap = (highs + lows + close_prices) / 3.0
        return np.where(np.isfinite(vwap) & (vwap > 0), vwap, close_prices)
    return close_prices


def _repair_suspension_gaps(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """In-memory suspension simulation used only during backtest runtime."""
    if len(df) < 2:
        # Always return a copy so callers never mutate shared market_data frames.
        return df.copy()

    gap_days = df.index.to_series().diff().dt.days
    max_gap = int(gap_days.max()) if not gap_days.empty else 0
    if max_gap <= _SUSPENSION_GAP_DAYS:
        return df.copy()

    out = df.copy()
    logger.warning(
        "[Backtest] %s: Prolonged trading gap of %d days detected. Applying in-memory synthetic halt simulation.",
        ticker,
        max_gap,
    )

    gap_end_dates = list(gap_days[gap_days > _SUSPENSION_GAP_DAYS].index)
    if not gap_end_dates:
        return out

    for gap_end in gap_end_dates:
        end_loc = out.index.get_loc(gap_end)
        if isinstance(end_loc, slice):
            end_loc = end_loc.start
        if isinstance(end_loc, np.ndarray):
            end_loc = int(np.flatnonzero(end_loc)[0]) if end_loc.any() else 0
        if end_loc <= 0:
            continue

        gap_start = out.index[int(end_loc) - 1]

        # Synthesize only inside this specific prolonged gap.
        gap_idx = pd.bdate_range(gap_start, gap_end)
        gap_idx = gap_idx[(gap_idx > gap_start) & (gap_idx < gap_end)]
        if len(gap_idx) == 0:
            continue

        # Avoid overwriting real bars if any timestamp already exists.
        synth_idx = gap_idx.difference(out.index)
        if len(synth_idx) == 0:
            continue

        pre_gap_close = out["Close"].loc[:gap_start]
        pre_gap_rets = pre_gap_close.pct_change().dropna()
        hist_vol = float(pre_gap_rets.std()) if len(pre_gap_rets) > 10 else 0.02

        seed_material = f"{ticker}_{pd.Timestamp(gap_start).strftime('%Y%m%d')}"
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.RandomState(seed)
        noise_rets = rng.normal(0, hist_vol, len(synth_idx))
        walk_returns = np.cumprod(1.0 + noise_rets)

        synth = pd.DataFrame(index=synth_idx)

        close_anchor = float(out.loc[gap_start, "Close"])
        synth["Close"] = close_anchor * walk_returns

        if "Adj Close" in out.columns:
            adj_anchor = out.loc[gap_start, "Adj Close"]
            if pd.isna(adj_anchor):
                adj_anchor = close_anchor
            synth["Adj Close"] = float(adj_anchor) * walk_returns

        if "Volume" in out.columns:
            synth["Volume"] = 0.0

        out = pd.concat([out, synth])

    out = out.sort_index().ffill()
    return out

def apply_halt_simulation(market_data: dict) -> dict:
    """
    Pre-apply _repair_suspension_gaps to every ticker once, returning a new dict
    of repaired frames.  Called by pre_load_data so the per-trial objective can
    share a single repaired copy without any per-trial allocation.

    MB-03 FIX (memory): The v11.48 fix passed {k: v.copy()} on every run_backtest
    call — 500 tickers × 300 trials × 4 WFO slices = 600,000 DataFrame allocations
    per optimizer run.  Pre-computing the repair once and passing a read-only
    reference eliminates this O(N×trials) allocation cascade.  _repair_suspension_gaps
    is deterministic (seeded from ticker hash), so computing it once is equivalent
    to computing it on every trial.
    """
    return {k: _repair_suspension_gaps(v, k) for k, v in market_data.items()}

def run_backtest(
    market_data:   dict,
    universe_type: Optional[str] = None,
    start_date:    str = "2020-01-01",
    end_date:      str = "2020-12-31",
    cfg:           Optional[UltimateConfig] = None,
    sector_map:    Optional[dict]           = None,
    universe:      Optional[List[str]]      = None,
) -> BacktestResults:
    if cfg is None:
        cfg = UltimateConfig()

    all_target_dates = pd.date_range(start_date, end_date, freq=cfg.REBALANCE_FREQ)

    union_universe = set(universe or [])
    universe_by_rebalance_date: Dict[pd.Timestamp, set[str]] = {}
    selected_universe_type = universe_type or "nse_total"

    # ── Custom screener path ──────────────────────────────────────────────────
    # get_historical_universe("custom", ...) now logs a survivorship warning and
    # returns [] instead of raising ValueError (Fixed Issue #1). For custom
    # backtests, `universe` is pre-populated by the caller (daily_workflow.py)
    # so the union_universe branch below handles it correctly without needing
    # special-casing here.
    if union_universe:
        # MB-04 FIX: Custom universes previously received NO historical narrowing —
        # every target date was assigned the full current-day screener list, silently
        # trading 2022-IPO stocks in 2018.  Bayesian optimization on this survivor-biased
        # universe learned parameters that exploit look-ahead; the resulting
        # optimal_cfg.json was lethal in live trading.
        #
        # Fix: use cumulative-volume existence gates from market_data (already loaded)
        # to narrow each rebalance date's universe to assets that had verifiable trading
        # activity prior to that date (>= HISTORY_GATE cumulative trading days).
        # This mirrors the PIT logic in build_historical_fallback.build_parquet.
        history_gate = int(getattr(cfg, "HISTORY_GATE", 20))
        vol_dict: Dict[str, pd.Series] = {}
        for sym in list(union_universe):
            key = sym if sym.endswith(".NS") else sym + ".NS"
            row = market_data.get(key)
            if row is None:
                row = market_data.get(sym)
            if row is not None and not row.empty and "Volume" in row.columns:
                import numpy as _np
                vol_dict[sym] = row["Volume"].replace(0, _np.nan).notna().cumsum()

        if vol_dict:
            cum_vol_df = pd.DataFrame(vol_dict).sort_index()
            for d in all_target_dates:
                ts = pd.Timestamp(d)
                past = cum_vol_df[cum_vol_df.index <= ts]
                if past.empty:
                    universe_by_rebalance_date[ts] = set()
                    continue
                eligible = past.iloc[-1]
                pit_syms = set(eligible[eligible >= history_gate].index.tolist())
                universe_by_rebalance_date[ts] = pit_syms & union_universe
                logger.debug(
                    "[Backtest][MB-04] %s: %d/%d custom universe symbols pass PIT volume gate.",
                    ts.date(), len(universe_by_rebalance_date[ts]), len(union_universe),
                )
        else:
            # No volume data available — fall back to full union with warning
            logger.warning(
                "[Backtest][MB-04] No volume data found for custom universe; "
                "survivorship bias is NOT corrected for this backtest."
            )
            for d in all_target_dates:
                universe_by_rebalance_date[pd.Timestamp(d)] = set(union_universe)
    else:
        for d in all_target_dates:
            historical_members = get_historical_universe(selected_universe_type, d)
            member_set = set(historical_members or [])
            universe_by_rebalance_date[pd.Timestamp(d)] = member_set
            union_universe.update(member_set)

        if not union_universe:
            raise RuntimeError(
                "No historical constituents resolved across requested backtest dates; "
                "verify universe snapshots or date range."
            )

    close_d, close_adj_d, open_d, high_d, low_d, div_d, split_d, volume_d = {}, {}, {}, {}, {}, {}, {}, {}
    for sym in union_universe:
        if not sym:
            continue
        key = sym if sym.endswith(".NS") else sym + ".NS"
        if key not in market_data:
            continue
        row = market_data[key]
        # MB-03 FIX: Gap repair is now pre-computed once in pre_load_data via
        # apply_halt_simulation().  The SIMULATE_HALTS branch here is intentionally
        # removed: market_data already contains repaired frames when called from
        # the optimizer.  For direct run_backtest calls without pre-loading (e.g.
        # the final OOS validation), repair is applied at call site if needed.
        valuation_series = row.get("Adj Close", row["Close"]) if cfg.AUTO_ADJUST_PRICES else row["Close"]
        close_d[sym]  = valuation_series.ffill()
        close_adj_d[sym] = row.get("Adj Close", row["Close"]).ffill()
        open_d[sym] = row.get("Open", row["Close"]).ffill()
        high_d[sym] = row.get("High", row["Close"]).ffill()
        low_d[sym] = row.get("Low", row["Close"]).ffill()
        div_d[sym] = row.get("Dividends", pd.Series(0.0, index=row.index)).fillna(0.0)
        split_d[sym] = row.get("Stock Splits", pd.Series(0.0, index=row.index)).fillna(0.0)
        volume_d[sym] = row["Volume"]

    if not close_d:
        raise ValueError("No valid symbols found in market_data for the dynamic historical universe.")

    close   = pd.DataFrame(close_d).sort_index()
    close_adj = pd.DataFrame(close_adj_d).sort_index()
    open_px = pd.DataFrame(open_d).sort_index()
    high_px = pd.DataFrame(high_d).sort_index()
    low_px = pd.DataFrame(low_d).sort_index()
    dividends = pd.DataFrame(div_d).sort_index().fillna(0.0)
    splits = pd.DataFrame(split_d).sort_index().fillna(0.0)
    volume  = pd.DataFrame(volume_d).sort_index()
    returns = close_adj.pct_change(fill_method=None).clip(lower=-0.99)

    trading_index = pd.DatetimeIndex(close.index).sort_values()
    valid = []
    for target in all_target_dates:
        lower_bound = target - pd.Timedelta(days=_REBALANCE_SNAP_WINDOW_DAYS)
        eligible = trading_index[(trading_index <= target) & (trading_index >= lower_bound)]
        if len(eligible) == 0:
            logger.debug(
                "Calendar guard: no prior trading day within %d days of %s; deferring rebalance.",
                _REBALANCE_SNAP_WINDOW_DAYS,
                target.date(),
            )
            continue
        valid.append(eligible[-1])

    rebal_dates = pd.DatetimeIndex(pd.DatetimeIndex(valid).unique())

    # FIX B3 + FIX #5: universe_by_rebalance_date was keyed by *target* calendar dates
    # (e.g. 2021-01-01 — New Year) but BacktestEngine.run() looks up by the *snapped*
    # trading date (e.g. 2020-12-31). The key mismatch caused the constituent filter
    # to silently return None on every market holiday, falling back to the full union
    # universe and injecting survivorship bias.
    #
    # FIX #5 (survivorship bias on holiday clusters):
    # The previous code used UNION when two target dates snapped to the same trading day.
    # Union includes stocks added in the LATER target date's snapshot, which is future
    # information relative to the EARLIER target date's trading bar. This is subtle
    # survivorship bias that manifests around bank holidays and year-end clusters.
    #
    # Fix: use the EARLIEST member set. When two targets collapse to the same bar,
    # the constituent list from the chronologically-first target date is always the
    # most conservative — it never includes stocks that were added after that date.
    if universe_by_rebalance_date:
        snapped_universe: Dict[pd.Timestamp, set] = {}
        # Track which calendar target date first claimed each snapped trading date.
        snapped_universe_target_date: Dict[pd.Timestamp, pd.Timestamp] = {}

        for target_d, members in sorted(universe_by_rebalance_date.items()):
            lower = target_d - pd.Timedelta(days=_REBALANCE_SNAP_WINDOW_DAYS)
            eligible = trading_index[(trading_index <= target_d) & (trading_index >= lower)]
            if len(eligible) > 0:
                snapped_key = eligible[-1]
                if snapped_key in snapped_universe:
                    # FIX #5: A second target date snapped to the same trading day.
                    # Keep the EARLIEST target's member set (already stored); discard
                    # the later one. The dict is iterated in sorted order above so
                    # snapped_universe[snapped_key] always holds the earliest target's
                    # constituents at this point — no update needed.
                    existing_target = snapped_universe_target_date[snapped_key]
                    logger.debug(
                        "[Backtest] Holiday snap collision: targets %s and %s both snap to %s. "
                        "Keeping earliest target's member set (%s) to avoid look-ahead bias.",
                        existing_target.date(), target_d.date(), snapped_key.date(),
                        existing_target.date(),
                    )
                    # Do NOT update snapped_universe[snapped_key] — keep the earliest.
                else:
                    snapped_universe[snapped_key] = set(members)
                    snapped_universe_target_date[snapped_key] = target_d
            # If no eligible trading day exists for this target, the rebalance was already
            # skipped in the snapping loop above — no entry needed.
        universe_by_rebalance_date = snapped_universe

    idx_df = market_data.get("^CRSLDX")
    if idx_df is None or idx_df.empty:
        idx_df = market_data.get("^NSEI")

    engine = InstitutionalRiskEngine(cfg)
    bt     = BacktestEngine(engine, initial_cash=cfg.INITIAL_CAPITAL)
    bt.run(close, volume, returns, rebal_dates, start_date, end_date=end_date, idx_df=idx_df, sector_map=sector_map, open_px=open_px, high_px=high_px, low_px=low_px, dividends=dividends, splits=splits, universe_by_rebalance_date=universe_by_rebalance_date)

    eq_daily  = pd.Series(bt._eq_vals, index=bt._eq_dates)
    eq_weekly = eq_daily[eq_daily.index.isin(rebal_dates)]

    if eq_weekly.empty and not eq_daily.empty:
        logger.warning(
            "[Backtest] No rebalance dates align with equity curve index. "
            "Defaulting to daily series for metrics."
        )
        eq_weekly = eq_daily

    rebal_log = (
        pd.DataFrame(bt._rebal_rows).set_index("date")
        if bt._rebal_rows
        else pd.DataFrame()
    )

    return BacktestResults(
        equity_curve = eq_weekly,
        trades       = bt.trades,
        metrics      = _compute_metrics(eq_daily, cfg.INITIAL_CAPITAL, cfg.SIGNAL_ANNUAL_FACTOR, trades=bt.trades),
        rebal_log    = rebal_log,
    )

def print_backtest_results(results: BacktestResults) -> None:
    m = results.metrics
    if not m:
        print("\n  \033[31m[!] Backtest returned no metrics. Check date range.\033[0m")
        return

    print(f"\n  \033[1;36mBACKTEST RESULTS\033[0m")
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


def _compute_metrics(
    eq: pd.Series,
    initial: float,
    periods_per_year: int = 252,
    trades: Optional[List[Trade]] = None,
) -> Dict:
    if initial <= 0:
        logger.warning(
            "[Backtest] Non-positive initial capital (%.4f) supplied; returning neutral metrics.",
            initial,
        )
        return {
            "cagr": 0.0,
            "max_dd": 0.0,
            "final": float(eq.iloc[-1]) if not eq.empty else float(initial),
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
        }

    if eq.empty:
        return {
            "cagr": 0.0,
            "max_dd": 0.0,
            "final": initial,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
        }

    final  = float(eq.iloc[-1])
    n_periods = max(len(eq) - 1, 1)
    cagr   = ((final / initial) ** (periods_per_year / n_periods) - 1.0) * 100.0
    dd     = (eq / eq.cummax() - 1.0) * 100.0
    max_dd = float(dd.min())

    dr = eq.pct_change(fill_method=None).dropna()
    if len(dr) > 1 and dr.std() > 0:
        ppy = float(periods_per_year)
        sharpe = (dr.mean() * ppy) / (dr.std() * np.sqrt(ppy))

        downside = dr[dr < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = (dr.mean() * ppy) / (downside.std() * np.sqrt(ppy))
        else:
            sortino = np.nan
    else:
        sharpe  = 0.0
        sortino = 0.0

    # FIX: Prevent perfectly good 0% drawdown strategies from scoring a 0.0 Calmar
    # Floor the denominator at 1.0% to yield a highly positive score for near-zero drawdowns.
    if max_dd >= 0.0:
        calmar = cagr  # Treat 0% drawdown as a 1% denominator so Calmar = CAGR
    else:
        calmar = cagr / max(abs(max_dd), 1.0)

    hit_rate = 0.0
    turnover = 0.0
    if trades:
        buy_trades = [t for t in trades if t.direction == "BUY" and t.delta_shares > 0]
        sell_trades = [t for t in trades if t.direction == "SELL" and t.delta_shares < 0]

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

        if round_trip_pnls:
            hit_rate = (sum(1 for pnl in round_trip_pnls if pnl > 0) / len(round_trip_pnls)) * 100.0

        total_buy_notional = sum(t.delta_shares * t.exec_price for t in buy_trades)
        total_sell_notional = sum(abs(t.delta_shares) * t.exec_price for t in sell_trades)
        avg_equity = float(eq.mean()) if len(eq) > 0 else float(initial)
        if avg_equity > 0:
            turnover = ((total_buy_notional + total_sell_notional) / 2.0) / avg_equity
            # MB-16 FIX: Use the actual date span to annualize turnover instead of
            # n_periods/252.  When eq is a weekly-frequency rebalance-date slice,
            # n_periods/252 gives ~0.21 years for a year of weekly bars, overstating
            # annual turnover by ~5x and causing Optuna to over-penalize low-turnover
            # parameters.  Use calendar days when a DatetimeIndex is available.
            if hasattr(eq.index, 'dtype') and np.issubdtype(eq.index.dtype, np.datetime64) and len(eq) >= 2:
                years = (eq.index[-1] - eq.index[0]).days / 365.25
            else:
                years = n_periods / float(periods_per_year)
            if years > 0:
                turnover = turnover / years

    return {
        "cagr":    round(cagr,    2),
        "max_dd":  round(max_dd,  2),
        "final":   round(final,   2),
        "sharpe":  round(sharpe,  2),
        "sortino": round(sortino, 2) if np.isfinite(sortino) else sortino,
        "calmar":  round(calmar,  2),
        "hit_rate": round(hit_rate, 2),
        "turnover": round(turnover, 4),
    }