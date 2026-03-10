"""
backtest_engine.py — Deterministic Walk-Forward Engine v11.46
=============================================================
Weekly rebalance cadence with full equity ledger, CVaR risk management,
and sector-diversified portfolio construction.

Now properly integrated with Impact-Aligned Execution and Historical Constituents
to eliminate Survivorship Bias. Fixes Look-Ahead PV/ADV computation biases.
"""

from __future__ import annotations

import logging
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
)
from signals import (
    generate_signals,
    compute_regime_score,
    SignalGenerationError,
)
from universe_manager import get_historical_universe

logger = logging.getLogger(__name__)
_REBALANCE_SNAP_WINDOW_DAYS = 5

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
    ) -> pd.DataFrame:
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

            if dividends is not None and date in dividends.index and self.engine.cfg.DIVIDEND_SWEEP:
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
    ) -> None:
        cfg = self.engine.cfg

        prev_idx = close.index.get_loc(date) - 1
        if prev_idx < 0:
            return
        signal_date = close.index[prev_idx]

        hist_log_rets = (
            np.log1p(returns.loc[:signal_date])
            .replace([np.inf, -np.inf], np.nan)
        )

        adv_vector = _build_adv_vector(symbols, close, volume, date)

        # Use execution-date prices for portfolio valuation inputs so manual/live
        # walk-forward replication (which rebalances on this same bar) remains
        # byte-identical to the backtest engine state transitions.
        valuation_close = close.loc[date]
        
        pv = self.state.cash + sum(
            self.state.shares.get(sym, 0) * (
                float(valuation_close[sym])
                if (sym in close.columns and pd.notna(valuation_close[sym]))
                else _ffill_price(self.state, sym)
            )
            for sym in self.state.shares
        )
        prev_w_dict = _build_prev_weights(self.state, symbols, pv)

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
                if pd.notna(valuation_close[sym])
                else _ffill_price(self.state, sym)
            )
            for sym in self.state.shares
            if sym in symbols
        ) / max(pv, 1e-6)

        self.state.update_exposure(regime_score, realised_cvar, cfg, gross_exposure=gross_exposure)

        target_weights         = np.zeros(len(symbols))
        apply_decay            = False
        optimization_succeeded = False
        sel_idx: List[int]     = []
        _force_full_cash       = False

        # ── Book CVaR screen ──────────────────────────────────────────────────
        # Two-tier breach architecture (see CVAR_HARD_BREACH_MULTIPLIER in UltimateConfig):
        #
        #  HARD breach  (CVaR > limit × CVAR_HARD_BREACH_MULTIPLIER, default 1.5×):
        #    CVaR is so elevated the QP solver is unlikely to find a feasible
        #    solution. Skip the optimizer, force full liquidation immediately.
        #
        #  SOFT breach  (limit < CVaR ≤ hard threshold):
        #    Let the optimizer run — its explicit QP CVaR constraint will build
        #    a de-risked portfolio naturally. The previous code triggered a full
        #    liquidation for any breach, including marginal ones like 6.507% vs
        #    6.500%, causing 33 unnecessary liquidations per 6-year backtest.
        if self.state.shares:
            # Use prices_t (T+0): screens current stress before order submission.
            book_cvar = compute_book_cvar(self.state, prices_t, symbols, hist_log_rets, cfg)
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
                logger.info(
                    "[Backtest] Book CVaR soft breach %.4f%% (limit %.4f%%, hard %.4f%%) on %s — "
                    "running optimizer with CVaR constraint active.",
                    book_cvar * 100, cfg.CVAR_DAILY_LIMIT * 100, hard_breach_threshold * 100, date,
                )
                # Do NOT set _force_full_cash — fall through to signal generation.

        # ── Signal generation + optimization ─────────────────────────────────
        if not _force_full_cash:
            try:
                raw_daily, adj_scores, sel_idx = generate_signals(
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
                sel_syms      = [symbols[i] for i in sel_idx]
                sector_labels = _build_sector_labels(sel_syms, sector_map)
                prev_weights  = np.array([prev_w_dict.get(sym, 0.0) for sym in symbols])

                try:
                    weights_sel = self.engine.optimize(
                        expected_returns    = raw_daily[sel_idx],
                        historical_returns  = hist_log_rets[[symbols[i] for i in sel_idx]],
                        execution_date      = date,
                        adv_shares          = adv_vector[sel_idx],
                        prices              = _execution_prices(symbols, date, prices_t, open_px, high_px, low_px)[sel_idx],
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
                        if self.state.consecutive_failures >= 3:
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
                target_weights = np.zeros(len(symbols), dtype=float)
                logger.warning(
                    "[Backtest] %s on %s — forcing full liquidation to cash.",
                    "Book CVaR breach" if _force_full_cash else
                    f"MAX_DECAY_ROUNDS={cfg.MAX_DECAY_ROUNDS} exhausted",
                    date,
                )
                _exhaust_decay = True
                activate_override_on_stress(self.state, cfg)
            else:
                target_weights = compute_decay_targets(self.state, sel_idx, symbols, cfg)
                sel_idx_set = set(sel_idx)
                sym_to_pos  = {s: i for i, s in enumerate(symbols)}
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
            _L = -(hist_log_rets.iloc[-_T:].reindex(columns=symbols, fill_value=0.0).values)
            
            exec_prices = _execution_prices(symbols, date, prices_t, open_px, high_px, low_px)
            execute_rebalance(
                self.state, target_weights, exec_prices, symbols, cfg,
                adv_shares     = adv_vector,
                date_context   = date, 
                trade_log      = self.trades,
                apply_decay    = apply_decay and not _exhaust_decay,
                scenario_losses = None if _exhaust_decay else _L,
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


def _build_adv_vector(symbols: List[str], close: pd.DataFrame, volume: pd.DataFrame, date: pd.Timestamp) -> np.ndarray:
    adv = []
    signal_date: Optional[pd.Timestamp] = None

    if not volume.empty:
        idx = volume.index
        if date in idx:
            # pandas 2.x always returns an int from get_loc on a unique DatetimeIndex.
            # The isinstance(pos, slice/ndarray) branches below guard against pandas < 2.0
            # behaviour (where get_loc could return a slice or boolean array on duplicates).
            # They are dead code in the current environment (pandas 2.3+) but kept for
            # defensive compatibility should the code be run on an older environment.
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
                signal_date = None  # first trading day in dataset — no prior bar, keep lookahead-free

    for sym in symbols:
        if sym in volume.columns and sym in close.columns and signal_date is not None:
            try:
                c_series = close.loc[:signal_date, sym]
                v_series = volume.loc[:signal_date, sym]
                notional = (c_series * v_series).replace(0, np.nan).ffill().dropna()
                lookback = notional.tail(20)
                if lookback.empty:
                    adv.append(0.0)
                else:
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
    unique_sectors = sorted(set(sector_map.get(s, "Unknown") for s in sel_syms))
    sec_idx        = {s: i for i, s in enumerate(unique_sectors)}
    return np.array([sec_idx[sector_map.get(sym, "Unknown")] for sym in sel_syms], dtype=int)



def _ffill_price(state: PortfolioState, sym: str) -> float:
    px = state.last_known_prices.get(sym)
    return float(px) if px is not None and np.isfinite(px) else 0.0


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

# ─── Public API ───────────────────────────────────────────────────────────────

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
    if not union_universe:
        selected_universe_type = universe_type or "nse_total"

        for d in all_target_dates:
            historical_members = get_historical_universe(selected_universe_type, d)
            if historical_members:
                union_universe.update(historical_members)

        if not union_universe:
            raise RuntimeError(
                "No historical constituents resolved across requested backtest dates; "
                "verify universe snapshots or date range."
            )

    close_d, close_adj_d, open_d, high_d, low_d, div_d, volume_d = {}, {}, {}, {}, {}, {}, {}
    for sym in union_universe:
        if not sym:
            continue
        key = sym if sym.endswith(".NS") else sym + ".NS"
        if key not in market_data:
            continue
        row = market_data[key]
        close_d[sym]  = row["Close"].ffill()
        close_adj_d[sym] = row.get("Adj Close", row["Close"]).ffill()
        open_d[sym] = row.get("Open", row["Close"]).ffill()
        high_d[sym] = row.get("High", row["Close"]).ffill()
        low_d[sym] = row.get("Low", row["Close"]).ffill()
        div_d[sym] = row.get("Dividends", pd.Series(0.0, index=row.index)).fillna(0.0)
        volume_d[sym] = row["Volume"]

    if not close_d:
        raise ValueError("No valid symbols found in market_data for the dynamic historical universe.")

    close   = pd.DataFrame(close_d).sort_index()
    close_adj = pd.DataFrame(close_adj_d).sort_index()
    open_px = pd.DataFrame(open_d).sort_index()
    high_px = pd.DataFrame(high_d).sort_index()
    low_px = pd.DataFrame(low_d).sort_index()
    dividends = pd.DataFrame(div_d).sort_index().fillna(0.0)
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

    idx_df = market_data.get("^CRSLDX")
    if idx_df is None or idx_df.empty:
        idx_df = market_data.get("^NSEI")

    engine = InstitutionalRiskEngine(cfg)
    bt     = BacktestEngine(engine, initial_cash=cfg.INITIAL_CAPITAL)
    bt.run(close, volume, returns, rebal_dates, start_date, end_date=end_date, idx_df=idx_df, sector_map=sector_map, open_px=open_px, high_px=high_px, low_px=low_px, dividends=dividends)

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

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

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
