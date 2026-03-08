"""
daily_workflow.py — Ultimate Momentum v11.46
============================================
Interactive CLI for live scanning, status display, and backtesting.
Features robust capital management, direct Screener.in web scraping,
Dividend Sweeping, and Impact-Aligned Rebalancing.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

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
    to_ns,
    to_bare,
    Trade,
)
from universe_manager import (
    fetch_nse_equity_universe,
    get_nifty500,
    get_sector_map,
    get_historical_universe,
    UniverseFetchError,
)
from data_cache import get_cache_summary, invalidate_cache, load_or_fetch
from backtest_engine import run_backtest, print_backtest_results
from signals import generate_signals, compute_adv, compute_regime_score

__version__ = "11.46"

BACKUP_GENERATIONS = 3
PAPER_MODE = False

# ─── ANSI colour palette ─────────────────────────────────────────────────────

class C:
    if sys.stdout.isatty():
        BLU   = "\033[34m"
        CYN   = "\033[36m"
        GRN   = "\033[32m"
        YLW   = "\033[33m"
        RED   = "\033[31m"
        GRY   = "\033[90m"
        RST   = "\033[0m"
        BLD   = "\033[1m"
        B_CYN = "\033[1;36m"
        B_GRN = "\033[1;32m"
        B_RED = "\033[1;31m"
    else:
        BLU = CYN = GRN = YLW = RED = GRY = RST = BLD = B_CYN = B_GRN = B_RED = ""

logger = logging.getLogger(__name__)

_DEFAULT_SCREENER_URL = os.environ.get(
    "SCREENER_URL",
    "https://www.screener.in/screens/3506127/hello/",
)

def load_optimized_config() -> UltimateConfig:
    cfg = UltimateConfig()
    if os.path.exists("data/optimal_cfg.json"):
        with open("data/optimal_cfg.json", "r") as f:
            best_params = json.load(f)
            valid_fields = UltimateConfig.__dataclass_fields__
            for k, v in best_params.items():
                if k not in valid_fields:
                    logger.warning("[Config] Ignoring unknown/stale optimized parameter: %s", k)
                    continue
                setattr(cfg, k, v)
    return cfg

def _render_meter(label: str, progress: float, width: int = 30) -> str:
    """Build a professional text meter to show long-running stage progress."""
    clipped = max(0.0, min(1.0, progress))
    filled = int(round(width * clipped))
    bar = f"{'█' * filled}{'░' * (width - filled)}"
    pct = f"{clipped * 100:5.1f}%"
    return f"  {C.CYN}{label:<18}{C.RST} [{bar}] {C.BLD}{pct}{C.RST}"

def _print_stage_status(label: str, progress: float, detail: str) -> None:
    """Print stage meter and contextual status text for user visibility."""
    print(_render_meter(label, progress))
    print(f"  {C.GRY}{detail}{C.RST}")

# ─── Screener.in Scraper & Prompters ─────────────────────────────────────────

def _scrape_screener(base_url: str) -> List[str]:
    try:
        import requests
        from bs4 import BeautifulSoup
        import re
    except ImportError:
        print(f"\n  {C.RED}[!] Missing dependencies for web scraping.{C.RST}")
        print(f"  {C.GRY}Please run: pip install requests beautifulsoup4{C.RST}\n")
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    symbols = set()
    page = 1
    max_pages = 50

    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query)
    qs.pop('page', None)
    clean_url = urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))

    while page <= max_pages:
        sep = "&" if "?" in clean_url else "?"
        url = f"{clean_url}{sep}page={page}"
        try:
            resp = requests.get(url, headers=headers, timeout=15)
        except requests.RequestException as e:
            logger.error("[Screener] Network error while reaching Screener.in: %s", e)
            break

        if resp.status_code in (401, 403):
            print(f"\n  {C.RED}[!] Screener.in denied access (HTTP {resp.status_code}).{C.RST}")
            print(f"  {C.GRY}Verify the screen is marked Public at:{C.RST}")
            print(f"  {C.GRY}screener.in → Your Screen → Edit → Visibility: Public{C.RST}\n")
            break
        elif resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, 'html.parser')
        links = soup.find_all('a', href=re.compile(r'^/company/[^/]+/(?:consolidated/)?$'))

        page_symbols = 0
        before_count = len(symbols)
        for link in links:
            match = re.search(r'/company/([^/]+)/', link['href'])
            if match:
                sym = match.group(1).upper()
                symbols.add(sym)
                page_symbols += 1

        if page_symbols == 0 or len(symbols) == before_count:
            break

        page += 1
        time.sleep(1)

    if page > max_pages:
        logger.warning("[Screener] Reached pagination safety limit (%d pages).", max_pages)

    return list(symbols)

def _filter_valid_custom_tickers(tickers: List[str]) -> List[str]:
    filtered: List[str] = []
    invalid_count = 0

    for raw in tickers:
        sym = raw.strip().upper()
        if not sym:
            continue
        if sym.isdigit():
            invalid_count += 1
            continue
        filtered.append(sym)

    if invalid_count:
        logger.warning(
            "[Universe] Ignored %d non-NSE numeric ticker code(s) from custom screener.",
            invalid_count,
        )

    return list(dict.fromkeys(filtered))

def _get_custom_universe() -> List[str]:
    saved_url = _DEFAULT_SCREENER_URL

    print(f"\n  {C.B_CYN}── Custom Screener Integration ──{C.RST}")
    logger.info("[Screener] Fetching universe from: %s", saved_url)

    tickers = _filter_valid_custom_tickers(_scrape_screener(saved_url))
    if tickers:
        return tickers

    logger.warning("[Screener] Scraping failed or returned 0 tickers. Attempting local file fallback...")

    files = ["custom_screener.csv", "custom_screener.txt"]
    for f in files:
        if os.path.exists(f):
            try:
                with open(f, "r") as file:
                    content = file.read().replace(",", "\n")
                    tickers = [line.strip().upper() for line in content.split("\n") if line.strip()]
                    tickers = [t for t in tickers if t not in ("SYMBOL", "TICKER", "")]
                    tickers = _filter_valid_custom_tickers(tickers)
                    if tickers:
                        print(f"\n  {C.YLW}[!] Web scrape failed. Found local fallback: {f}{C.RST}")
                        print(f"  {C.YLW}    This file may be stale. Universe: {len(tickers)} tickers.{C.RST}")
                        confirm = input(f"  {C.CYN}Proceed with local data? (y/n): {C.RST}").strip().lower()
                        if confirm == "y":
                            return tickers
                        else:
                            print(f"  {C.GRY}Cancelled. Returning empty universe.{C.RST}")
                            return []
            except Exception as e:
                logger.error("[Screener] Failed to read %s: %s", f, e)
    return []

def _check_and_prompt_initial_capital(state: PortfolioState, label: str, name: str) -> None:
    if not state.shares and not state.equity_hist and abs(state.cash - 1_000_000.0) < 1.0:
        print(f"\n  {C.YLW}⚡ New portfolio detected for {label}{C.RST}")
        try:
            raw_cap = input(f"  {C.CYN}Enter your starting capital (₹) [Default 10,00,000]: {C.RST}").replace(",", "").strip()
            if raw_cap:
                cap = float(raw_cap)
                if cap > 0:
                    state.cash = cap
                    save_portfolio_state(state, name)
                    print(f"  {C.GRN}[+] Initial capital set to ₹{cap:,.2f}{C.RST}\n")
        except ValueError:
            print(f"  {C.RED}Invalid input. Using default ₹10,00,000.{C.RST}\n")

# ─── Corporate action / split detection ──────────────────────────────────────

def detect_and_apply_splits(state: PortfolioState, market_data: dict, cfg: UltimateConfig) -> List[str]:
    """
    Detects splits and sweeps dividends to ensure cash ledger accuracy.
    """
    adjusted: List[str] = []
    
    for sym in list(state.shares.keys()):
        ns = to_ns(sym)
        row = market_data.get(ns)
        if row is None or row.empty:
            row = market_data.get(sym)
        if row is None or row.empty:
            continue
            
        if getattr(cfg, "DIVIDEND_SWEEP", True) and "Dividends" in row.columns:
            dividends = row["Dividends"][row["Dividends"] > 0]
            if not dividends.empty:
                shares_held = state.shares.get(sym, 0)
                if shares_held > 0:
                    last_event_id = state.dividend_ledger.get(sym, "")
                    last_event_date = last_event_id.split(':')[0] if last_event_id else "1900-01-01"
                    
                    for div_date, div_val in dividends.items():
                        div_date_str = pd.Timestamp(div_date).strftime("%Y-%m-%d")
                        if div_date_str > last_event_date:
                            div_val_float = float(div_val)
                            state.cash = round(state.cash + (div_val_float * shares_held), 10)
                            state.dividend_ledger[sym] = f"{div_date_str}:{div_val_float:.8f}"
                            logger.info(
                                "DIVIDEND SWEEP: %s distributed ₹%.2f per share (x %d shares) on %s. Added to cash.",
                                sym, div_val_float, shares_held, div_date_str
                            )

        current_price = float(row["Close"].iloc[-1])
        if not np.isfinite(current_price) or current_price <= 0:
            continue
            
        last_price = state.last_known_prices.get(sym)
        if last_price is None or last_price <= 0:
            continue

        ratio = last_price / current_price

        # Broader institutional ratio list with a tighter tolerance
        split_tolerance = getattr(cfg, "SPLIT_TOLERANCE", 0.005)
        for r in [2, 5, 10, 3, 4, 20, 1.5, 1.25, 0.666, 0.5, 0.2]:
            if abs(ratio - r) / r <= split_tolerance:
                old_shares     = state.shares[sym]
                theoretical_new_shares = old_shares * r
                new_shares     = int(np.floor(theoretical_new_shares + 1e-12))
                old_entry      = state.entry_prices.get(sym, current_price * r)
                new_entry      = old_entry / r

                # Safely sweep fractional shares
                fractional_new_shares = max(0.0, theoretical_new_shares - new_shares)
                fractional_value = fractional_new_shares * current_price
                state.cash = round(state.cash + fractional_value, 10)

                logger.warning(
                    "SPLIT DETECTED: %s ratio=%.3f (≈%gx) "
                    "shares %d→%d entry_price ₹%.2f→₹%.2f",
                    sym, ratio, r, old_shares, new_shares, old_entry, new_entry,
                )
                state.shares[sym]       = new_shares
                state.entry_prices[sym] = round(new_entry, 4)
                state.last_known_prices[sym] = current_price
                adjusted.append(sym)
                break

    return adjusted

# ─── State persistence ────────────────────────────────────────────────────────

def save_portfolio_state(state: PortfolioState, name: str) -> None:
    if PAPER_MODE:
        print(f"  {C.YLW}[!] Paper mode active. State will not be saved.{C.RST}")
        return

    os.makedirs("data", exist_ok=True)
    state_file = f"data/portfolio_state_{name}.json"
    tmp_file   = f"{state_file}.tmp"
    try:
        for i in range(BACKUP_GENERATIONS - 1, -1, -1):
            src, dst = f"{state_file}.bak.{i}", f"{state_file}.bak.{i+1}"
            if os.path.exists(src):
                shutil.copy2(src, dst)
        if os.path.exists(state_file):
            shutil.copy2(state_file, f"{state_file}.bak.0")

        with open(tmp_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_file, state_file)

        if os.name == "posix":
            dir_fd = os.open("data", os.O_DIRECTORY)
            os.fsync(dir_fd)
            os.close(dir_fd)
    except Exception as exc:
        logger.error("Durable save failed for '%s': %s", name, exc)
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass

def load_portfolio_state(name: str) -> PortfolioState:
    state_file = f"data/portfolio_state_{name}.json"
    backups    = [state_file] + [f"{state_file}.bak.{i}" for i in range(BACKUP_GENERATIONS)]
    for path in backups:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return PortfolioState.from_dict(json.load(f))
            except Exception as exc:
                logger.warning("Corrupted state at %s: %s", path, exc)
    return PortfolioState()

# ─── Core scan logic ──────────────────────────────────────────────────────────

def _run_scan(
    universe: List[str],
    state:    PortfolioState,
    label:    str,
    cfg_override: Optional[UltimateConfig] = None,
) -> tuple:
    scan_started_at = time.perf_counter()
    _print_stage_status("Download", 0.05, f"Preparing {len(universe):,} symbols for {label}...")

    cfg    = cfg_override if cfg_override else load_optimized_config()
    engine = InstitutionalRiskEngine(cfg)
    state.equity_hist_cap = cfg.EQUITY_HIST_CAP

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")

    # FIX (Bug-D — Delisting Crash / PV Gap): Always include currently-held symbols
    # in the fetch batch, even if they have been dropped from the scan universe
    # (e.g. delisted, merged, or fallen below the ADV floor).  Without this,
    # a held-but-absent stock never receives a price update, its market value is
    # silently omitted from the PV calculation, and the position-sizer deploys
    # too much of the available cash budget on the next rebalance.
    held_syms  = {to_ns(s) for s in state.shares.keys()}
    all_syms   = list({to_ns(t) for t in universe} | held_syms | {"^NSEI", "^CRSLDX"})
    _print_stage_status(
        "Download",
        0.35,
        f"Fetching/caching OHLCV data ({start_date} → {end_date}) for {len(all_syms):,} instruments...",
    )
    market_data = load_or_fetch(all_syms, start_date, end_date, cfg=cfg)
    _print_stage_status("Download", 1.0, f"Data ready. Starting iteration and signal analysis for {label}.")
    _print_stage_status("Analysis", 0.10, "Normalizing market snapshots and benchmark regime inputs...")

    idx_df = market_data.get("^CRSLDX")
    if idx_df is None or idx_df.empty:
        idx_df = market_data.get("^NSEI")

    idx_slice    = idx_df.iloc[:-1] if idx_df is not None and not idx_df.empty else None
    regime_score = compute_regime_score(idx_slice, cfg=cfg)

    # Detect splits and sweep dividends BEFORE marking MTM values
    split_syms = detect_and_apply_splits(state, market_data, cfg)
    if split_syms:
        logger.warning("[Scan] Applied split adjustments for: %s", split_syms)

    close_d: Dict[str, pd.Series] = {}
    for sym in universe:
        ns = to_ns(sym)
        if ns in market_data:
            close_d[to_bare(ns)] = market_data[ns]["Close"].ffill()

    if not close_d:
        logger.warning("[Scan] No data available for any universe symbol.")
        return state, market_data

    _print_stage_status("Analysis", 0.35, f"Built close-price matrix for {len(close_d):,} active symbols.")

    close    = pd.DataFrame(close_d).sort_index()
    active   = list(close.columns)
    prices   = close.iloc[-1].values.astype(float)
    active_idx = {sym: i for i, sym in enumerate(active)}

    mtm_notional = sum(
        state.shares.get(sym, 0) * prices[active_idx[sym]]
        for sym in state.shares
        if sym in active_idx
    )
    # FIX (Bug-D — Delisting PV Gap): Include held stocks that are absent from the
    # current scan universe (possibly delisted / suspended) at their last-known
    # price.  Omitting them under-counts total equity, causing the optimizer to
    # over-deploy cash and breach position-size limits on the next trade.
    for _absent_sym in state.shares:
        if _absent_sym not in active_idx:
            _fallback_px = state.last_known_prices.get(_absent_sym, 0.0)
            if _fallback_px > 0:
                mtm_notional += state.shares[_absent_sym] * _fallback_px
                logger.warning(
                    "[Scan] Held symbol '%s' absent from current market data "
                    "(possibly delisted/suspended). Using last-known price ₹%.2f "
                    "for PV calculation. Position will be force-closed after "
                    "%d consecutive absent periods.",
                    _absent_sym, _fallback_px, cfg.MAX_ABSENT_PERIODS,
                )
    pv = mtm_notional + state.cash
    initial_cash = state.cash
    initial_gross_exposure = mtm_notional / pv if pv > 0 else 1.0

    close_hist    = close.iloc[:-1]
    log_rets      = np.log1p(close_hist.pct_change(fill_method=None).clip(lower=-0.99)).replace([np.inf, -np.inf], np.nan)
    adv_arr       = compute_adv(market_data, active)
    prev_w_arr    = np.array([state.weights.get(sym, 0.0) for sym in active])
    _print_stage_status("Analysis", 0.55, "Running momentum iterations, liquidity filters, and risk gates...")

    state.update_exposure(
        regime_score,
        state.realised_cvar(min_obs=cfg.CVAR_MIN_HISTORY),
        cfg,
        gross_exposure=initial_gross_exposure,
    )

    weights              = np.zeros(len(active))
    apply_decay          = False
    optimization_succeeded = False
    total_slippage       = 0.0
    trade_log: List[Trade] = []
    sel_idx: List[int]   = []
    _force_full_cash     = False

    # ── Book CVaR screen ──────────────────────────────────────────────────────
    if state.shares:
        book_cvar = compute_book_cvar(state, prices, active, log_rets, cfg)
        if book_cvar > cfg.CVAR_DAILY_LIMIT + 1e-6:
            logger.warning(
                "[Scan] Book CVaR %.4f%% exceeds limit %.4f%% — "
                "skipping optimization, forcing immediate liquidation.",
                book_cvar * 100, cfg.CVAR_DAILY_LIMIT * 100,
            )
            state.consecutive_failures += 1
            apply_decay      = True
            _force_full_cash = True
            _activate_override_on_stress(state, cfg)

    if not _force_full_cash:
        try:
            raw_daily, adj_scores, sel_idx = generate_signals(
                log_rets, adv_arr, cfg, prev_weights=state.weights
            )
            if not sel_idx:
                raise OptimizationError("No valid universe candidates.", OptimizationErrorType.DATA)
            sel_syms      = [active[i] for i in sel_idx]
            sector_map    = get_sector_map(sel_syms, cfg=cfg)
            unique_sectors = sorted(set(sector_map.values()))
            sec_idx        = {s: i for i, s in enumerate(unique_sectors)}
            sector_labels  = np.array([sec_idx[sector_map[sym]] for sym in sel_syms], dtype=int)

            weights_sel = engine.optimize(
                expected_returns    = raw_daily[sel_idx],
                historical_returns  = log_rets[[active[i] for i in sel_idx]],
                execution_date      = pd.Timestamp(end_date),
                adv_shares          = adv_arr[sel_idx],
                prices              = prices[sel_idx],
                portfolio_value     = pv,
                prev_w              = prev_w_arr[sel_idx],
                exposure_multiplier = state.exposure_multiplier,
                sector_labels       = sector_labels,
            )
            weights[sel_idx]               = weights_sel
            state.consecutive_failures     = 0
            state.decay_rounds             = 0
            optimization_succeeded         = True

        except (OptimizationError, ValueError) as exc:
            is_data_error = isinstance(exc, ValueError) or (
                isinstance(exc, OptimizationError) and exc.error_type == OptimizationErrorType.DATA
            )
            if not is_data_error:
                state.consecutive_failures += 1
                logger.error("Solver failure #%d: %s. Freezing state.", state.consecutive_failures, exc)
                if state.consecutive_failures >= 3:
                    logger.warning(
                        "3 consecutive solver failures — triggering gate-filtered "
                        "pro-rata liquidation (%.0f%% of gate-passing positions).",
                        cfg.DECAY_FACTOR * 100,
                    )
                    apply_decay = True
            else:
                logger.warning("Data error (empty universe / thin history): %s. Resetting counters and freezing state.", exc)
                state.consecutive_failures = 0
                state.decay_rounds = 0

    # ── Gate-filtered decay target computation ────────────────────────────────
    _exhaust_decay = False
    if apply_decay and not optimization_succeeded:
        if _force_full_cash or state.decay_rounds >= cfg.MAX_DECAY_ROUNDS:
            weights = np.zeros(len(active), dtype=float)
            logger.warning(
                "[Scan] %s — forcing full liquidation to cash.",
                "Book CVaR breach" if _force_full_cash else
                f"MAX_DECAY_ROUNDS={cfg.MAX_DECAY_ROUNDS} exhausted",
            )
            _exhaust_decay = True
        else:
            weights = compute_decay_targets(state, sel_idx, active, cfg)

    if optimization_succeeded or apply_decay:
        _T_cvar = min(len(log_rets), cfg.CVAR_LOOKBACK)
        _scenario_losses = -(
            log_rets.iloc[-_T_cvar:]
            .reindex(columns=active, fill_value=0.0)
            .values
        )
        total_slippage = execute_rebalance(
            state, weights, prices, active, cfg,
            adv_shares=adv_arr,
            date_context=pd.Timestamp(end_date), trade_log=trade_log,
            apply_decay    = apply_decay and not _exhaust_decay,
            scenario_losses = None if _exhaust_decay else _scenario_losses,
        )
        if _exhaust_decay:
            state.decay_rounds = 0
            state.consecutive_failures = 0

    _print_stage_status("Analysis", 0.85, "Applying rebalance decisions and updating portfolio marks...")

    price_dict = {sym: prices[active_idx[sym]] for sym in active}
    state.record_eod(price_dict)
    final_pv = state.equity_hist[-1] if state.equity_hist else pv

    logger.info(
        "%s%s%s | Regime: %.2f | CVaR: %.2f%% | Failures: %d | "
        "Equity: %s₹%s%s | Slippage: %s₹%s%s",
        C.BLU, label, C.RST,
        regime_score,
        state.realised_cvar(min_obs=cfg.CVAR_MIN_HISTORY) * 100,
        state.consecutive_failures,
        C.GRN, f"{final_pv:,.0f}", C.RST,
        C.RED, f"{total_slippage:,.0f}", C.RST,
    )

    elapsed = time.perf_counter() - scan_started_at
    _print_stage_status("Analysis", 1.0, f"{label} completed in {elapsed:.1f}s.")

    if trade_log:
        final_mtm = sum(state.shares.get(sym, 0) * prices[active_idx[sym]] for sym in state.shares if sym in active_idx)
        final_gross_exposure = final_mtm / final_pv if final_pv > 0 else 0.0
        net_cash_delta = state.cash - initial_cash

        print(f"\n  {C.B_CYN}=== PHASE A: TRADE INTENT SUMMARY ==={C.RST}")
        print(f"  {C.GRY}{'─' * 66}{C.RST}")
        print(f"  {C.BLD}Gross Exp Before:{C.RST} {initial_gross_exposure:>7.1%}")
        print(f"  {C.BLD}Gross Exp After: {C.RST} {final_gross_exposure:>7.1%}")
        cash_color = C.B_GRN if net_cash_delta >= 0 else C.B_RED
        print(f"  {C.BLD}Net Cash Delta:  {C.RST} {cash_color}₹{net_cash_delta:+,.0f}{C.RST}")
        print(f"  {C.BLD}Total Slippage:  {C.RST} {C.RED}₹{total_slippage:,.0f}{C.RST}")

        largest_trade = max(trade_log, key=lambda t: abs(t.delta_shares) * t.exec_price, default=None)
        if largest_trade:
            notional = abs(largest_trade.delta_shares) * largest_trade.exec_price
            print(f"  {C.BLD}Largest Change:  {C.RST} {largest_trade.direction} {largest_trade.symbol} (₹{notional:,.0f})")

        print(f"\n  {C.B_CYN}=== PHASE B: EXECUTION ACTION SHEET ==={C.RST}")
        print(f"  {C.GRY}{'─' * 66}{C.RST}")

        sorted_trades = sorted(trade_log, key=lambda t: state.weights.get(t.symbol, 0.0), reverse=True)

        for t in sorted_trades:
            action_color = C.B_GRN if t.direction == "BUY" else C.B_RED
            target_weight = state.weights.get(t.symbol, 0.0)
            print(
                f"  {action_color}{t.direction:<4}{C.RST} | {C.BLD}{t.symbol:<12}{C.RST} | "
                f"{abs(t.delta_shares):>6,d} shares @ ≈ ₹{t.exec_price:>9,.2f} | Tgt: {C.CYN}{target_weight:>5.1%}{C.RST}"
            )
        print(f"  {C.GRY}{'─' * 66}{C.RST}\n")

    return state, market_data


def _activate_override_on_stress(state: PortfolioState, cfg: UltimateConfig) -> None:
    """Activate exposure override immediately after hard risk events.

    BUG-6 NOTE: An identical copy of this function exists in backtest_engine.py.
    If the override logic changes here it must be mirrored there manually.
    Consolidation into momentum_engine.py is the correct long-term fix.
    """
    state.override_active = True
    state.override_cooldown = max(state.override_cooldown, 4)
    state.exposure_multiplier = float(max(cfg.MIN_EXPOSURE_FLOOR, state.exposure_multiplier * 0.5))

# ─── Status display ───────────────────────────────────────────────────────────

def _print_status(state: PortfolioState, label: str, market_data: dict, cfg: Optional[UltimateConfig] = None) -> None:
    if cfg is None:
        cfg = UltimateConfig()

    print(f"\n  {C.GRY}╭{'─' * 88}╮{C.RST}")
    print(f"  {C.GRY}│{C.BLD}  STATUS — {label}  {C.RST}{C.GRY}{' ' * (75 - len(label))}│{C.RST}")
    print(f"  {C.GRY}╰{'─' * 88}╯{C.RST}")

    if not state.shares:
        print(f"  {C.GRY}No open positions.{C.RST}\n")
        return

    active     = list(state.shares.keys())
    prices_now = {}
    for sym in active:
        ns = to_ns(sym)
        if ns in market_data and not market_data[ns].empty:
            prices_now[sym] = float(market_data[ns]["Close"].iloc[-1])

    mtm = sum(
        state.shares[s] * (prices_now.get(s) or state.last_known_prices.get(s, 0.0))
        for s in active
    )
    pv  = mtm + state.cash

    rows      = []
    total_pnl = 0.0
    for sym in active:
        shares   = state.shares[sym]
        price    = prices_now.get(sym) or state.last_known_prices.get(sym, float("nan"))
        entry    = state.entry_prices.get(sym, float("nan"))
        notional = shares * price if np.isfinite(price) else 0.0
        weight   = notional / pv if pv > 0 else 0.0
        pnl      = (price - entry) * shares if (np.isfinite(price) and np.isfinite(entry)) else float("nan")
        if np.isfinite(pnl):
            total_pnl += pnl
        rows.append({
            "sym": sym, "shares": shares, "price": price,
            "entry": entry, "weight": weight, "notional": notional, "pnl": pnl,
        })

    rows.sort(key=lambda x: x["weight"], reverse=True)
    c_pipe = f"{C.GRY}│{C.RST}"

    print(f"  {C.GRY}┌──────────────┬─────────┬───────────┬───────────┬────────┬─────────────┬─────────────┐{C.RST}")
    print(
        f"  {c_pipe} {C.B_CYN}{'Symbol':<12}{C.RST} {c_pipe} {C.B_CYN}{'Shares':>7}{C.RST} "
        f"{c_pipe} {C.B_CYN}{'Price':>9}{C.RST} {c_pipe} {C.B_CYN}{'Entry':>9}{C.RST} "
        f"{c_pipe} {C.B_CYN}{'Weight':>6}{C.RST} {c_pipe} {C.B_CYN}{'Notional':>11}{C.RST} "
        f"{c_pipe} {C.B_CYN}{'Unreal P&L':>11}{C.RST} {c_pipe}"
    )
    print(f"  {C.GRY}├──────────────┼─────────┼───────────┼───────────┼────────┼─────────────┼─────────────┤{C.RST}")

    for r in rows:
        pnl_raw   = f"₹{r['pnl']:+,.0f}" if np.isfinite(r["pnl"]) else "n/a"
        pnl_color = C.B_GRN if r["pnl"] > 0 else (C.B_RED if r["pnl"] < 0 else C.RST)
        print(
            f"  {c_pipe} {C.BLD}{r['sym']:<12}{C.RST} {c_pipe} {r['shares']:>7,d} "
            f"{c_pipe} {r['price']:>9,.2f} {c_pipe} {r['entry']:>9,.2f} "
            f"{c_pipe} {C.CYN}{r['weight']:>6.1%}{C.RST} {c_pipe} {r['notional']:>11,.0f} "
            f"{c_pipe} {pnl_color}{pnl_raw:>11}{C.RST} {c_pipe}"
        )

    print(f"  {C.GRY}├──────────────┼─────────┼───────────┼───────────┼────────┼─────────────┼─────────────┤{C.RST}")
    print(
        f"  {c_pipe} {C.BLD}{'Cash':<12}{C.RST} {c_pipe} {'':>7} {c_pipe} {'':>9} {c_pipe} {'':>9} "
        f"{c_pipe} {C.CYN}{state.cash/pv:>6.1%}{C.RST} {c_pipe} {state.cash:>11,.0f} {c_pipe} {'':>11} {c_pipe}"
    )
    print(f"  {C.GRY}├──────────────┼─────────┼───────────┼───────────┼────────┼─────────────┼─────────────┤{C.RST}")
    tot_color = C.B_GRN if total_pnl > 0 else (C.B_RED if total_pnl < 0 else C.RST)
    print(
        f"  {c_pipe} {C.BLD}{'TOTAL':<12}{C.RST} {c_pipe} {'':>7} {c_pipe} {'':>9} {c_pipe} {'':>9} "
        f"{c_pipe} {C.BLD}{1.0:>6.1%}{C.RST} {c_pipe} {C.BLD}{pv:>11,.0f}{C.RST} "
        f"{c_pipe} {tot_color}{'₹'+f'{total_pnl:+,.0f}':>11}{C.RST} {c_pipe}"
    )
    print(f"  {C.GRY}└──────────────┴─────────┴───────────┴───────────┴────────┴─────────────┴─────────────┘{C.RST}")

    cvar        = state.realised_cvar(min_obs=cfg.CVAR_MIN_HISTORY)
    cvar_color  = C.RED if cvar > 0.12 else C.GRN
    print(f"\n  {C.BLD}Portfolio Diagnostics:{C.RST}")
    print(f"  {C.YLW}⚡{C.RST} Exposure Multiplier : {C.BLD}{state.exposure_multiplier:.3f}{C.RST}")
    print(f"  {C.RED}🛡️ {C.RST} Override Active     : {C.BLD}{state.override_active}{C.RST}  {C.GRY}(Cooldown: {state.override_cooldown}){C.RST}")
    print(f"  {C.CYN}📉{C.RST} CVaR (realised)     : {cvar_color}{cvar:.2%}{C.RST}")
    print(f"  {C.RED}⚠️ {C.RST} Consec. Failures    : {C.BLD}{state.consecutive_failures}{C.RST}")
    print(f"  {C.BLU}📊{C.RST} Equity History Pts  : {C.BLD}{len(state.equity_hist)}{C.RST}\n")

def _portfolio_activity_badge(state: PortfolioState) -> str:
    has_activity = bool(state.shares or state.equity_hist or abs(state.cash - 1_000_000.0) >= 1.0)
    if not has_activity:
        return f"{C.GRY}Idle{C.RST}"
    positions = len(state.shares)
    return f"{C.B_GRN}Active{C.RST} {C.GRY}({positions} pos | Cash ₹{state.cash:,.0f}){C.RST}"

def _render_main_menu(states: Dict[str, PortfolioState]) -> None:
    box_width = 78

    def _menu_box_line(text: str = "") -> str:
        trimmed = text[:box_width]
        return f"{C.BLU}  │{C.RST}{trimmed:<{box_width}}{C.BLU}│{C.RST}"

    now = datetime.now().strftime("%d %b %Y, %I:%M %p")
    title = f"ULTIMATE MOMENTUM V{__version__} — DAILY WORKFLOW"
    snapshot = f"Snapshot: {now}    Tip: Run status after each scan."

    print(f"\n{C.BLU}  ╭{'─' * box_width}╮{C.RST}")
    print(_menu_box_line(f"{title:^{box_width}}"))
    print(_menu_box_line(f"  {snapshot}"))
    print(f"{C.BLU}  ╰{'─' * box_width}╯{C.RST}")

    print(f"  {C.B_CYN}Scans & Research{C.RST}")
    print(f"    {C.BLD}[1]{C.RST} NSE Total Scan      {C.GRY}Run full-market rebalance preview.{C.RST}")
    print(f"    {C.BLD}[2]{C.RST} Nifty 500 Scan      {C.GRY}Focused large-cap and liquid basket.{C.RST}")
    print(f"    {C.BLD}[3]{C.RST} Custom Screener     {C.GRY}Use Screener.in or local custom list.{C.RST}")
    print(f"    {C.BLD}[4]{C.RST} Backtest            {C.GRY}Replay strategy performance by date.{C.RST}")

    print(f"  {C.B_CYN}Portfolio Operations{C.RST}")
    print(f"    {C.BLD}[5]{C.RST} Status              {C.GRY}Holdings table + risk diagnostics.{C.RST}")
    print(f"    {C.BLD}[6]{C.RST} Manage Cash         {C.GRY}Deposit/withdraw portfolio cash.{C.RST}")
    print(f"    {C.BLD}[7]{C.RST} Clear States        {C.GRY}Reset local state and cache files.{C.RST}")
    print(f"    {C.BLD}[q]{C.RST} Quit\n")

    print(f"  {C.BLD}Portfolio Health:{C.RST}")
    print(f"    NSE Total       → {_portfolio_activity_badge(states['nse_total'])}")
    print(f"    Nifty 500       → {_portfolio_activity_badge(states['nifty'])}")
    print(f"    Custom Screener → {_portfolio_activity_badge(states['custom'])}")

def _prompt_menu_choice(prompt: str, valid: List[str], default: Optional[str] = None) -> str:
    while True:
        raw = input(prompt).strip().lower()
        if not raw and default is not None:
            return default
        if raw not in valid:
            print(f"  {C.RED}Invalid choice. Valid options: {', '.join(valid)}{C.RST}")
            continue
        return raw

def _normalise_start_date(raw: str, default: str = "2020-01-01") -> str:
    candidate = raw.strip() or default
    try:
        datetime.strptime(candidate, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date '{candidate}'. Expected format YYYY-MM-DD.") from exc
    return candidate

def _prompt_survival_mode(err: UniverseFetchError, universe_name: str) -> Optional[List[str]]:
    print(f"\n  {C.B_RED}[!] UNIVERSE FETCH FAILURE — {universe_name}{C.RST}")
    print(f"  {C.RED}{err}{C.RST}")
    n = len(err.fallback_universe)
    print(f"\n  {C.YLW}Hard-floor fallback available: {n} Nifty 50 stocks.{C.RST}")
    print(f"  {C.YLW}WARNING: This is a material regime shift.{C.RST}")
    print(f"  {C.YLW}         Sector exposure, turnover and liquidity will differ significantly.{C.RST}")
    confirm = input(
        f"\n  {C.CYN}Proceed with the {n}-stock hard-floor fallback? (y/n): {C.RST}"
    ).strip().lower()
    if confirm == "y":
        logger.warning(
            "[Universe] Operator confirmed survival-mode fallback (%d stocks) for %s.",
            n, universe_name,
        )
        return err.fallback_universe
    print(f"  {C.GRY}Cancelled. Returning to main menu.{C.RST}")
    return None

# ─── Main menu ────────────────────────────────────────────────────────────────

def main_menu() -> None:
    states    = {
        "nse_total": load_portfolio_state("nse_total"),
        "nifty":     load_portfolio_state("nifty"),
        "custom":    load_portfolio_state("custom"),
    }
    mkt_cache: dict = {"nse_total": {}, "nifty": {}, "custom": {}}

    while True:
        _render_main_menu(states)

        c = _prompt_menu_choice(f"\n  {C.CYN}Choice: {C.RST}", ["1", "2", "3", "4", "5", "6", "7", "q"])
        if not c:
            continue

        if c == "1":
            _check_and_prompt_initial_capital(states["nse_total"], "NSE TOTAL", "nse_total")
            cfg = load_optimized_config()
            try:
                _universe = fetch_nse_equity_universe(cfg=cfg)
            except UniverseFetchError as e:
                _universe = _prompt_survival_mode(e, "NSE Total Equity")
                if _universe is None:
                    continue
            preview      = copy.deepcopy(states["nse_total"])
            preview, mkt = _run_scan(_universe, preview, "NSE TOTAL MKT SCAN", cfg)
            mkt_cache["nse_total"] = mkt
            _print_status(preview, "PREVIEW — NSE TOTAL", mkt, cfg=cfg)
            if input(f"  {C.YLW}Save these changes? (y/n): {C.RST}").strip().lower() == "y":
                states["nse_total"] = preview
                save_portfolio_state(preview, "nse_total")
                print(f"  {C.GRN}[+] Saved permanently.{C.RST}")
            else:
                print(f"  {C.GRY}[-] Discarded.{C.RST}")

        elif c == "2":
            _check_and_prompt_initial_capital(states["nifty"], "NIFTY 500", "nifty")
            cfg = load_optimized_config()
            try:
                _universe = get_nifty500()
            except UniverseFetchError as e:
                _universe = _prompt_survival_mode(e, "Nifty 500")
                if _universe is None:
                    continue
            preview      = copy.deepcopy(states["nifty"])
            preview, mkt = _run_scan(_universe, preview, "NIFTY 500 SCAN", cfg)
            mkt_cache["nifty"] = mkt
            _print_status(preview, "PREVIEW — NIFTY 500", mkt, cfg=cfg)
            if input(f"  {C.YLW}Save these changes? (y/n): {C.RST}").strip().lower() == "y":
                states["nifty"] = preview
                save_portfolio_state(preview, "nifty")
                print(f"  {C.GRN}[+] Saved permanently.{C.RST}")
            else:
                print(f"  {C.GRY}[-] Discarded.{C.RST}")

        elif c == "3":
            universe = _get_custom_universe()
            if not universe:
                print(f"  {C.RED}[!] No custom universe found.{C.RST}")
                print(f"  {C.GRY}Please verify the Screener.in URL or provide a local file and try again.{C.RST}")
                continue

            logger.info("[Universe] Loaded %d symbols from custom screener.", len(universe))
            _check_and_prompt_initial_capital(states["custom"], "CUSTOM SCREENER", "custom")

            custom_cfg = load_optimized_config()
            if len(universe) < 100:
                custom_cfg.MAX_POSITIONS = 8

            preview      = copy.deepcopy(states["custom"])
            preview, mkt = _run_scan(universe, preview, "CUSTOM SCREENER", custom_cfg)
            mkt_cache["custom"] = mkt
            _print_status(preview, "PREVIEW — CUSTOM SCREENER", mkt, cfg=custom_cfg)
            if input(f"  {C.YLW}Save these changes? (y/n): {C.RST}").strip().lower() == "y":
                states["custom"] = preview
                save_portfolio_state(preview, "custom")
                print(f"  {C.GRN}[+] Saved permanently.{C.RST}")
            else:
                print(f"  {C.GRY}[-] Discarded.{C.RST}")

        elif c == "4":
            print(f"\n  {C.CYN}Backtest — Select Universe:{C.RST}")
            print(f"  [1] NSE Total  [2] Nifty 500  [3] Custom Screener")
            bt_c = _prompt_menu_choice(f"  {C.CYN}Choice [Default 2]: {C.RST}", ["1", "2", "3"], default="2")
            if not bt_c:
                continue

            raw_start = input(f"  {C.CYN}Start (YYYY-MM-DD) [Default 2020-01-01]: {C.RST}")
            try:
                start = _normalise_start_date(raw_start)
            except ValueError as exc:
                print(f"  {C.RED}{exc}{C.RST}")
                continue

            if bt_c == "1":
                universe_identifier = "nse_total"
            elif bt_c == "3":
                universe_identifier = "custom"
            else:
                universe_identifier = "nifty500"

            end        = datetime.today().strftime("%Y-%m-%d")
            
            bt_cfg = load_optimized_config()
            all_target_dates = pd.date_range(start, end, freq=bt_cfg.REBALANCE_FREQ)
            historical_union = set()
            for target_date in all_target_dates:
                historical_union.update(get_historical_universe(universe_identifier, target_date))

            if not historical_union and bt_c == "3":
                historical_union.update(_get_custom_universe())
            elif not historical_union:
                historical_union.update(get_nifty500())

            # FIX (Bug-8): Pass cfg=bt_cfg so load_or_fetch uses the optimized
            # CVAR_LOOKBACK for padding instead of the hardcoded 200-day default.
            # Without this, strategies with CVAR_LOOKBACK > 200 (e.g., 500) would
            # receive insufficient historical data, causing early rebalance dates to
            # fail the dimensionality check inside InstitutionalRiskEngine.optimize().
            data = load_or_fetch(list(historical_union) + ["^NSEI", "^CRSLDX"], start, end, cfg=bt_cfg)
            # FIX (Bug-A — Survivorship Bias Guard): run_backtest raises RuntimeError
            # when no point-in-time historical universe files are found, intentionally
            # refusing to fall back to the current Nifty 500 constituents (which
            # would silently inflate CAGR by 3-5% p.a. via survivorship bias).
            # Catch that error here so the CLI loop stays alive and guide the user.
            try:
                print_backtest_results(run_backtest(data, universe_identifier, start, end, cfg=bt_cfg))
            except RuntimeError as exc:
                print(f"\n  {C.B_RED}[!] BACKTEST FAILED — Historical Universe Data Missing{C.RST}")
                print(f"  {C.RED}{exc}{C.RST}")
                print(f"\n  {C.YLW}This safeguard prevents silently survivorship-biased backtests.{C.RST}")
                print(f"  {C.YLW}Backtesting with today's index members over-states CAGR by ~3-5%{C.RST}")
                print(f"  {C.YLW}p.a. because it excludes companies that were delisted or demoted.{C.RST}")
                print(f"\n  {C.CYN}Fix: run the following command to generate required snapshots:{C.RST}")
                print(f"  {C.BLD}    python historical_builder.py{C.RST}")
                print(f"  {C.GRY}Required files:{C.RST}")
                print(f"  {C.GRY}    data/historical_nifty500.parquet  (or data/historical_nse_total.parquet){C.RST}\n")

        elif c == "5":
            for name, label in [("nse_total", "NSE TOTAL"), ("nifty", "NIFTY 500"), ("custom", "CUSTOM SCREENER")]:
                has_activity = states[name].shares or states[name].equity_hist or abs(states[name].cash - 1_000_000.0) >= 1.0
                if has_activity:
                    mkt = mkt_cache.get(name) or {}
                    if not mkt and states[name].shares:
                        syms = list({to_ns(s) for s in states[name].shares})
                        end  = datetime.today().strftime("%Y-%m-%d")
                        mkt  = load_or_fetch(
                            syms,
                            (datetime.today() - timedelta(days=22)).strftime("%Y-%m-%d"),
                            end,
                        )
                        mkt_cache[name] = mkt
                    _print_status(states[name], label, mkt)
            if not any((states[n].shares or states[n].equity_hist or abs(states[n].cash - 1_000_000.0) >= 1.0) for n in states):
                print(f"  {C.GRY}All portfolios are empty.{C.RST}")

        elif c == "6":
            print(f"\n  {C.CYN}Manage Cash — Select Portfolio:{C.RST}")
            print(f"  [1] NSE Total  [2] Nifty 500  [3] Custom Screener")
            p_c = _prompt_menu_choice(f"  {C.CYN}Choice: {C.RST}", ["1", "2", "3"])
            if not p_c:
                continue
            p_map = {"1": "nse_total", "2": "nifty", "3": "custom"}
            if p_c in p_map:
                name = p_map[p_c]
                state = states[name]
                print(f"\n  {C.BLD}Current Cash: {C.GRN}₹{state.cash:,.2f}{C.RST}")
                print(f"  {C.GRY}Use positive number to deposit, negative to withdraw.{C.RST}")
                try:
                    amt_str = input(f"  {C.CYN}Amount (₹): {C.RST}").replace(",", "").strip()
                    amt = float(amt_str)
                    state.cash = max(0.0, state.cash + amt)
                    save_portfolio_state(state, name)
                    action = "Deposited" if amt >= 0 else "Withdrew"
                    print(f"  {C.GRN}[+] {action} ₹{abs(amt):,.2f}. New Cash: ₹{state.cash:,.2f}{C.RST}")
                except ValueError:
                    print(f"  {C.RED}Invalid amount.{C.RST}")
            else:
                print(f"  {C.RED}Invalid choice.{C.RST}")

        elif c == "7":
            print(f"\n  {C.B_RED}WARNING: This will permanently erase ALL portfolio states and caches.{C.RST}")
            confirm = input(f"  {C.CYN}Type 'YES' to confirm: {C.RST}").strip()
            if confirm.upper() == "YES":
                invalidate_cache()
                from universe_manager import invalidate_universe_cache
                invalidate_universe_cache()
                for n in ["nse_total", "nifty", "custom"]:
                    p = f"data/portfolio_state_{n}.json"
                    for suffix in ["", ".bak.0", ".bak.1", ".bak.2"]:
                        target = p + suffix
                        if os.path.exists(target):
                            os.remove(target)
                states    = {"nse_total": PortfolioState(), "nifty": PortfolioState(), "custom": PortfolioState()}
                mkt_cache = {"nse_total": {}, "nifty": {}, "custom": {}}
                print(f"  {C.GRN}[+] All states and caches cleared.{C.RST}")
            else:
                print(f"  {C.GRY}Cancelled.{C.RST}")

        elif c == "q":
            print(f"  {C.GRY}Goodbye!{C.RST}\n")
            break


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultimate Momentum daily workflow")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Enable paper trading mode (disables portfolio state file writes).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    PAPER_MODE = bool(args.paper)

    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=f"{C.GRY}[%(asctime)s]{C.RST} %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/ultimate.log", encoding="utf-8", mode="a")
        ]
    )

    logger.info("Ultimate Momentum v%s started", __version__)
    if PAPER_MODE:
        logger.warning("[!] Paper mode active. State will not be saved.")
    main_menu()
