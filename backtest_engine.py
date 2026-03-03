"""
backtest_engine.py — Deterministic Walk-Forward Engine
=======================================================
Weekly rebalance cadence with full equity ledger, CVaR risk management,
and sector-diversified portfolio construction.
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
    Trade,
    to_ns,
)
from signals import generate_signals, compute_regime_score, compute_single_adv

logger = logging.getLogger(__name__)


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
        self._rebal_rows: list     = []   # accumulated rebalance log rows

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
    ) -> pd.DataFrame:
        start_dt = pd.Timestamp(start_date)
        end_dt   = pd.Timestamp(end_date) if end_date else close.index[-1]
        symbols  = list(close.columns)

        # O(1) membership tests for rapid daily simulation
        rebal_set = set(rebalance_dates)

        for date in close.index:
            if date < start_dt or date > end_dt:
                continue

            close_t  = close.loc[date]
            prices_t = close_t.values.astype(float)

            active_idx = {sym: i for i, sym in enumerate(symbols)}
            pv = self.state.cash + sum(
                self.state.shares.get(sym, 0) * (
                    float(close_t[sym])
                    if (sym in active_idx and pd.notna(close_t[sym]))
                    else self.state.last_known_prices.get(sym, 0.0)
                )
                for sym in self.state.shares
            )

            if date in rebal_set:
                self._run_rebalance(
                    date, close, volume, returns, symbols, prices_t,
                    pv, idx_df, sector_map,
                )

            price_dict = {
                sym: prices_t[active_idx[sym]]
                for sym in symbols
                if pd.notna(close_t[sym])
            }
            self.state.record_eod(price_dict)
            post_pv = self.state.equity_hist[-1] if self.state.equity_hist else pv

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
        pv:         float,
        idx_df,
        sector_map,
    ) -> None:
        cfg = self.engine.cfg

        prev_idx = close.index.get_loc(date) - 1
        if prev_idx < 0:
            return
        signal_date = close.index[prev_idx]

        # Explicit T-1 history (exclude execution date to prevent look-ahead).
        hist_log_rets = (
            np.log1p(returns.loc[:signal_date])
            .replace([np.inf, -np.inf], np.nan)
        )

        adv_vector = _build_adv_vector(symbols, volume, date)

        # FIX #7: Replaced opaque double-for-in-comprehension with a readable
        # helper so the weight calculation is easy to audit and test.
        prev_w_dict = _build_prev_weights(self.state, symbols, pv)

        raw_daily, adj_scores, sel_idx = generate_signals(
            hist_log_rets,
            adv_vector,
            cfg,
            prev_weights=prev_w_dict,
        )

        _idx_ok      = idx_df is not None and not (hasattr(idx_df, "empty") and idx_df.empty)
        idx_slice    = idx_df.loc[:signal_date] if _idx_ok else None
        # Pass cfg so compute_regime_score uses the dynamic vol threshold (FIX G2).
        regime_score = compute_regime_score(idx_slice, cfg=cfg)

        # Guard: only use realised CVaR once enough history has accumulated.
        if len(self.state.equity_hist) >= cfg.CVAR_MIN_HISTORY:
            realised_cvar = self.state.realised_cvar(min_obs=cfg.CVAR_MIN_HISTORY)
        else:
            realised_cvar = 0.0

        gross_exposure = sum(
            self.state.shares.get(sym, 0) * (
                float(close.loc[date, sym])
                if pd.notna(close.loc[date, sym])
                else self.state.last_known_prices.get(sym, 0.0)
            )
            for sym in self.state.shares
            if sym in symbols
        ) / max(pv, 1e-6)

        self.state.update_exposure(regime_score, realised_cvar, cfg, gross_exposure=gross_exposure)

        target_weights         = np.zeros(len(symbols))
        apply_decay            = False
        optimization_succeeded = False

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
                    prices              = prices_t[sel_idx],
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
                    if self.state.consecutive_failures >= 2:
                        logger.debug(
                            "[Backtest] Applying %.0f%% deleverage on %s.",
                            (1 - cfg.DECAY_FACTOR) * 100, date,
                        )
                        apply_decay = True
        else:
            # FIX #6: No optimization attempt this bar (e.g. empty candidate set)
            # is NOT a solver failure. Reset decay_rounds so stale decay state
            # from a previous failure sequence does not persist into a regime
            # where the universe is simply temporarily empty.
            self.state.decay_rounds = 0

        if optimization_succeeded or apply_decay:
            # FIX C4: pass scenario_losses for post-decay CVaR check.
            _T = min(len(hist_log_rets), self.engine.cfg.CVAR_LOOKBACK)
            _L = -(hist_log_rets.iloc[-_T:].reindex(columns=symbols, fill_value=0.0).values)
            execute_rebalance(
                self.state, target_weights, prices_t, symbols, cfg,
                date_context=date, trade_log=self.trades, apply_decay=apply_decay,
                scenario_losses=_L,
            )
            self._rebal_rows.append({
                "date":              date,
                "regime_score":       round(regime_score, 4),
                "realised_cvar":      round(realised_cvar, 6),
                "exposure_multiplier":round(self.state.exposure_multiplier, 4),
                "override_active":    self.state.override_active,
                "n_positions":        len(self.state.shares),
                "apply_decay":        apply_decay,
            })


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_prev_weights(
    state:   PortfolioState,
    symbols: List[str],
    pv:      float,
) -> Dict[str, float]:
    """
    Compute current mark-to-market weights using T-1 (last known) prices.

    Using last_known_prices instead of today's execution price prevents a
    subtle look-ahead bias where today's close would be used to determine
    the starting weight before today's rebalance decision is made.
    """
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


def _build_adv_vector(symbols: List[str], volume: pd.DataFrame, date: pd.Timestamp) -> np.ndarray:
    """
    Build the ADV (average daily volume) vector for sizing constraints.
    Applies the T-1 slice (iloc[:-1]) before delegating to compute_single_adv.
    """
    adv = []
    for sym in symbols:
        if sym in volume.columns:
            try:
                series = volume.loc[:date, sym].iloc[:-1]
                adv.append(compute_single_adv(series))
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


# ─── Public API ───────────────────────────────────────────────────────────────

def run_backtest(
    market_data: dict,
    universe:    List[str],
    start_date:  str,
    end_date:    str,
    cfg:         Optional[UltimateConfig] = None,
    sector_map:  Optional[dict]           = None,
) -> BacktestResults:
    if cfg is None:
        cfg = UltimateConfig()

    close_d  = {}
    volume_d = {}
    for sym in universe:
        if not sym:
            continue
        key = sym if sym.endswith(".NS") else sym + ".NS"
        if key not in market_data:
            continue
        close_d[sym]  = market_data[key]["Close"].ffill()
        volume_d[sym] = market_data[key]["Volume"]

    if not close_d:
        raise ValueError("No valid symbols found in market_data for the given universe.")

    close   = pd.DataFrame(close_d).sort_index()
    volume  = pd.DataFrame(volume_d).sort_index()
    returns = close.pct_change(fill_method=None).clip(lower=-0.99)

    # FIX C2: Calendar alignment - forward-only resolution with same-week guard.
    # The original method=pad (look-behind) maps a holiday Friday to the PRECEDING
    # Thursday. In live execution the rebalance occurs on the FOLLOWING Monday,
    # creating a 1-3 day temporal asymmetry. Corrected: use backfill (look-forward)
    # plus a same-ISO-week guard that skips the rebalance entirely if the resolved
    # session falls outside the target week, matching live no-rebal-week behaviour.
    all_target_dates = pd.date_range(start_date, end_date, freq=cfg.REBALANCE_FREQ)
    trading_index    = pd.DatetimeIndex(close.index).sort_values()

    idx   = trading_index.get_indexer(all_target_dates, method="backfill")
    valid = []
    for target, resolved_pos in zip(all_target_dates, idx):
        if resolved_pos < 0 or resolved_pos >= len(trading_index):
            continue
        resolved = trading_index[resolved_pos]
        t_iso = target.isocalendar()
        r_iso = resolved.isocalendar()
        if r_iso[0] != t_iso[0] or r_iso[1] != t_iso[1]:
            logger.debug("Calendar guard: %s -> %s crosses ISO week boundary, skipping.",
                         target.date(), resolved.date())
            continue
        valid.append(resolved)

    rebal_dates = pd.DatetimeIndex(pd.DatetimeIndex(valid).unique())

    idx_df = market_data.get("^CRSLDX")
    if idx_df is None or idx_df.empty:
        idx_df = market_data.get("^NSEI")

    engine = InstitutionalRiskEngine(cfg)
    bt     = BacktestEngine(engine, initial_cash=cfg.INITIAL_CAPITAL)
    bt.run(close, volume, returns, rebal_dates, start_date, end_date=end_date, idx_df=idx_df, sector_map=sector_map)

    # FIX #3: equity_curve is weekly-sampled (rebalance dates only) for display.
    # metrics are intentionally computed on the full daily series for statistical
    # accuracy (Sharpe, Sortino, max-drawdown all need daily granularity).
    # The discrepancy is documented here to prevent misinterpretation.
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
        equity_curve = eq_weekly,          # weekly-sampled for charting
        trades       = bt.trades,
        metrics      = _compute_metrics(eq_daily, cfg.INITIAL_CAPITAL),  # daily for accuracy
        rebal_log    = rebal_log,
    )


def print_backtest_results(results: BacktestResults) -> None:
    m = results.metrics
    print(f"\n  \033[1;36mBACKTEST RESULTS\033[0m")
    print(f"  \033[90m{chr(9472)*65}\033[0m")
    print(
        f"  \033[1mFinal:\033[0m \033[32m₹{m['final']:,.0f}\033[0m  "
        f"\033[1mCAGR:\033[0m {m['cagr']:.2f}%  "
        f"\033[1mSharpe:\033[0m {m['sharpe']:.2f}  "
        f"\033[1mSortino:\033[0m {m['sortino']:.2f}  "
        f"\033[1mMaxDD:\033[0m {m['max_dd']:.2f}%  "
        f"\033[1mCalmar:\033[0m {m['calmar']:.2f}"
    )
    print(f"  \033[90m{chr(9472)*65}\033[0m\n")


def _compute_metrics(eq: pd.Series, initial: float) -> Dict:
    if eq.empty:
        return {"cagr": 0.0, "max_dd": 0.0, "final": initial, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0}

    final  = float(eq.iloc[-1])
    span   = (eq.index[-1] - eq.index[0]).days
    years  = max(span / 365.25, 0.1)
    cagr   = ((final / initial) ** (1.0 / years) - 1.0) * 100.0
    dd     = (eq / eq.cummax() - 1.0) * 100.0
    max_dd = float(dd.min())

    dr = eq.pct_change(fill_method=None).dropna()
    if len(dr) > 1 and dr.std() > 0:
        avg_days_between = span / len(dr)
        periods_per_year = 365.25 / avg_days_between
        sharpe = (dr.mean() * periods_per_year) / (dr.std() * np.sqrt(periods_per_year))

        downside = dr[dr < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = (dr.mean() * periods_per_year) / (downside.std() * np.sqrt(periods_per_year))
        else:
            sortino = sharpe  # Fallback if no downside volatility exists
    else:
        sharpe  = 0.0
        sortino = 0.0

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    return {
        "cagr":    round(cagr,    2),
        "max_dd":  round(max_dd,  2),
        "final":   round(final,   2),
        "sharpe":  round(sharpe,  2),
        "sortino": round(sortino, 2),
        "calmar":  round(calmar,  2),
    }
