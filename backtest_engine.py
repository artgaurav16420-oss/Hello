"""
backtest_engine.py — Deterministic Walk-Forward Engine v11.44
=============================================================
Weekly rebalance cadence with full equity ledger, CVaR risk management,
and sector-diversified portfolio construction.

Now properly integrated with Impact-Aligned Execution and Historical Constituents
to eliminate Survivorship Bias.
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
)
from signals import (
    generate_signals,
    compute_regime_score,
    compute_single_adv,
    SignalGenerationError,
)
from universe_manager import get_historical_universe

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

            if date in rebal_set:
                self._run_rebalance(
                    date, close, volume, returns, symbols, prices_t,
                    idx_df, sector_map,
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

        adv_vector = _build_adv_vector(symbols, volume, date)

        close_t = close.loc[date]
        pv = self.state.cash + sum(
            self.state.shares.get(sym, 0) * (
                float(close_t[sym])
                if (sym in close.columns and pd.notna(close_t[sym]))
                else self.state.last_known_prices.get(sym, 0.0)
            )
            for sym in self.state.shares
        )
        prev_w_dict = _build_prev_weights(self.state, symbols, pv)

        _idx_ok      = idx_df is not None and not (hasattr(idx_df, "empty") and idx_df.empty)
        idx_slice    = idx_df.loc[:signal_date] if _idx_ok else None
        regime_score = compute_regime_score(idx_slice, cfg=cfg)

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
        sel_idx: List[int]     = []
        _force_full_cash       = False

        # ── Book CVaR screen ──────────────────────────────────────────────────
        if self.state.shares:
            book_cvar = compute_book_cvar(self.state, prices_t, symbols, hist_log_rets, cfg)
            if book_cvar > cfg.CVAR_DAILY_LIMIT + 1e-6:
                logger.warning(
                    "[Backtest] Book CVaR %.4f%% exceeds limit %.4f%% on %s — "
                    "skipping optimization, forcing immediate liquidation.",
                    book_cvar * 100, cfg.CVAR_DAILY_LIMIT * 100, date,
                )
                self.state.consecutive_failures += 1
                apply_decay      = True
                _force_full_cash = True
                _activate_override_on_stress(self.state, cfg)

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
                _activate_override_on_stress(self.state, cfg)
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
            
            # ── FIX: Institutional Impact Alignment passed to executor ──
            execute_rebalance(
                self.state, target_weights, prices_t, symbols, cfg,
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


def _activate_override_on_stress(state: PortfolioState, cfg: UltimateConfig) -> None:
    """Activate exposure override after hard risk events (breach/exhaustion)."""
    state.override_active = True
    state.override_cooldown = max(state.override_cooldown, 4)
    state.exposure_multiplier = float(max(cfg.MIN_EXPOSURE_FLOOR, state.exposure_multiplier * 0.5))


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


def _build_adv_vector(symbols: List[str], volume: pd.DataFrame, date: pd.Timestamp) -> np.ndarray:
    adv = []
    signal_date: Optional[pd.Timestamp] = None

    if not volume.empty:
        idx = volume.index
        if date in idx:
            pos = idx.get_loc(date)
            if isinstance(pos, slice):
                pos = pos.start
            elif isinstance(pos, np.ndarray):
                pos = int(pos[0]) if len(pos) else -1
            if pos > 0:
                signal_date = idx[pos - 1]

    for sym in symbols:
        if sym in volume.columns and signal_date is not None:
            try:
                series = volume.loc[:signal_date, sym]
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

        # ── FIX: Historical Constituents ──
        # Rebuilds the universe tracking point-in-time list to avoid Survivorship Bias
        for d in all_target_dates:
            historical_members = get_historical_universe(selected_universe_type, d)
            if historical_members:
                union_universe.update(historical_members)

        if not union_universe:
            logger.warning("Historical integration failed to yield symbols. Using current universe.")
            from universe_manager import get_nifty500, fetch_nse_equity_universe
            if selected_universe_type == "nifty500":
                union_universe.update(get_nifty500())
            else:
                union_universe.update(fetch_nse_equity_universe())

    close_d, volume_d = {}, {}
    for sym in union_universe:
        if not sym:
            continue
        key = sym if sym.endswith(".NS") else sym + ".NS"
        if key not in market_data:
            continue
        close_d[sym]  = market_data[key]["Close"].ffill()
        volume_d[sym] = market_data[key]["Volume"]

    if not close_d:
        raise ValueError("No valid symbols found in market_data for the dynamic historical universe.")

    close   = pd.DataFrame(close_d).sort_index()
    volume  = pd.DataFrame(volume_d).sort_index()
    returns = close.pct_change(fill_method=None).clip(lower=-0.99)

    trading_index = pd.DatetimeIndex(close.index).sort_values()
    idx           = trading_index.get_indexer(all_target_dates, method="pad")
    
    valid = []
    for target, resolved_pos in zip(all_target_dates, idx):
        if resolved_pos < 0 or resolved_pos >= len(trading_index):
            continue
        resolved = trading_index[resolved_pos]
        t_iso = target.isocalendar()
        r_iso = resolved.isocalendar()
        if r_iso[0] != t_iso[0] or r_iso[1] != t_iso[1]:
            logger.debug("Calendar guard: %s -> %s crosses ISO week boundary, skipping.", target.date(), resolved.date())
            continue
        valid.append(resolved)

    rebal_dates = pd.DatetimeIndex(pd.DatetimeIndex(valid).unique())

    idx_df = market_data.get("^CRSLDX")
    if idx_df is None or idx_df.empty:
        idx_df = market_data.get("^NSEI")

    engine = InstitutionalRiskEngine(cfg)
    bt     = BacktestEngine(engine, initial_cash=cfg.INITIAL_CAPITAL)
    bt.run(close, volume, returns, rebal_dates, start_date, end_date=end_date, idx_df=idx_df, sector_map=sector_map)

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
        metrics      = _compute_metrics(eq_daily, cfg.INITIAL_CAPITAL),
        rebal_log    = rebal_log,
    )

def print_backtest_results(results: BacktestResults) -> None:
    m = results.metrics
    if not m:
        print("\n  \033[31m[!] Backtest returned no metrics. Check date range.\033[0m")
        return

    print(f"\n  \033[1;36mBACKTEST RESULTS\033[0m")
    print(f"  \033[90m{chr(9472)*65}\033[0m")
    print(
        f"  \033[1mFinal:\033[0m \033[32m₹{m.get('final', 0):,.0f}\033[0m  "
        f"\033[1mCAGR:\033[0m {m.get('cagr', 0):.2f}%  "
        f"\033[1mSharpe:\033[0m {m.get('sharpe', 0):.2f}  "
        f"\033[1mSortino:\033[0m {m.get('sortino', 0):.2f}  "
        f"\033[1mMaxDD:\033[0m {m.get('max_dd', 0):.2f}%  "
        f"\033[1mCalmar:\033[0m {m.get('calmar', 0):.2f}"
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
            sortino = float("nan")
    else:
        sharpe  = 0.0
        sortino = 0.0

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    return {
        "cagr":    round(cagr,    2),
        "max_dd":  round(max_dd,  2),
        "final":   round(final,   2),
        "sharpe":  round(sharpe,  2),
        "sortino": sortino if not np.isfinite(sortino) else round(sortino, 2),
        "calmar":  round(calmar,  2),
    }