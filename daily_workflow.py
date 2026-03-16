"""
daily_workflow.py — Ultimate Momentum v11.48
============================================
Interactive CLI for live scanning, status display, and backtesting.
Features robust capital management, direct Screener.in web scraping,
Dividend Sweeping, and Impact-Aligned Rebalancing.
"""
from __future__ import annotations
import importlib.util

if importlib.util.find_spec("dotenv") is not None:
    from dotenv import load_dotenv

    load_dotenv()

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
    absent_symbol_effective_price,
    to_ns,
    to_bare,
    Trade,
    activate_override_on_stress,
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

__version__ = "11.48"

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


def _next_rebalance_due(last_rebalance_date: str, rebalance_freq: str) -> Optional[pd.Timestamp]:
    if not last_rebalance_date:
        return None
    try:
        anchor = pd.Timestamp(last_rebalance_date).normalize()
        return anchor + pd.tseries.frequencies.to_offset(rebalance_freq)
    except Exception:
        logger.warning(
            "[Scan] Invalid last_rebalance_date='%s' or REBALANCE_FREQ='%s'; disabling cadence gate this run.",
            last_rebalance_date,
            rebalance_freq,
        )
        return None

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

    # CONNECT_TIMEOUT: 5s — fail fast if Screener.in is unreachable.
    # READ_TIMEOUT:   30s — allow enough time for a slow first-byte response
    #                       without hanging the CLI indefinitely (requests has
    #                       no default timeout; omitting it risks an infinite
    #                       stall at the socket level on a stalled connection).
    _TIMEOUT = (5, 30)

    symbols = set()
    page = 1
    max_pages = 50

    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query)
    qs.pop('page', None)
    clean_url = urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))

    # Session reuses the underlying TCP/TLS connection across paginated requests,
    # eliminating per-page handshake overhead and automatically propagating any
    # session cookies that Screener.in sets on the first request.
    with requests.Session() as session:
        session.headers.update(headers)
        while page <= max_pages:
            sep = "&" if "?" in clean_url else "?"
            url = f"{clean_url}{sep}page={page}"
            try:
                resp = session.get(url, timeout=_TIMEOUT)
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

            before_count = len(symbols)
            for link in links:
                match = re.search(r'/company/([^/]+)/', link['href'])
                if match:
                    symbols.add(match.group(1).upper())

            if len(symbols) == before_count:
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
    PHASE 9 FIX: Strictly prevents double-counting splits when AUTO_ADJUST_PRICES=True.
    """
    adjusted: List[str] = []

    for sym in list(state.shares.keys()):
        ns = to_ns(sym)
        row = market_data.get(ns)
        if row is None or row.empty:
            row = market_data.get(sym)
        if row is None or row.empty:
            continue
            
        # MB-09 FIX: Skip dividend sweep when AUTO_ADJUST_PRICES=True.
        # With auto-adjusted prices, Adj Close already embeds dividend distributions
        # (the price falls on ex-date to reflect the payout).  Crediting cash on top
        # of an already-adjusted price double-counts the dividend, overstating NAV by
        # 2× the dividend amount per share.  The backtest engine applies the same
        # guard in BacktestEngine.run(); this aligns the live-scan path with it.
        if getattr(cfg, "DIVIDEND_SWEEP", True) and "Dividends" in row.columns and not getattr(cfg, "AUTO_ADJUST_PRICES", True):
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

        # FIX B2: The previous PHASE 9 implementation used an unconditional `continue`
        # here when AUTO_ADJUST_PRICES=True, skipping the entire split-detection block.
        # This is incorrect. AUTO_ADJUST_PRICES controls whether *prices* in the
        # valuation matrix use Adj Close (split-adjusted). It does NOT mean the share
        # ledger is automatically updated — that is this function's responsibility.
        # If we hold 10 shares and a 2:1 split occurs, we must update to 20 shares
        # regardless of price-adjustment mode, otherwise portfolio NAV halves on the
        # ex-date (10 shares × new price of ₹50 = ₹500 vs correct 20 × ₹50 = ₹1000).
        # Two committed tests (test_detect_and_apply_splits_applies_when_stock_splits_column_marks_event
        # and test_detect_and_apply_splits_runs_even_when_auto_adjust_enabled) confirm
        # this requirement. We still update last_known_prices here before falling through.
        if getattr(cfg, "AUTO_ADJUST_PRICES", True):
            state.last_known_prices[sym] = current_price
            # ↓ Fall through to split detection below — do NOT continue.

        split_ratio = 0.0
        if "Stock Splits" in row.columns and not row["Stock Splits"].empty:
            split_series = row["Stock Splits"].fillna(0.0)
            last_scan_date = None
            if state.last_rebalance_date:
                try:
                    last_scan_date = pd.Timestamp(state.last_rebalance_date)
                except Exception:
                    last_scan_date = None

            if last_scan_date is not None:
                split_index_tz = getattr(split_series.index, "tz", None)
                if split_index_tz is not None:
                    if last_scan_date.tzinfo is None:
                        last_scan_date = last_scan_date.tz_localize(split_index_tz)
                    else:
                        last_scan_date = last_scan_date.tz_convert(split_index_tz)
                elif last_scan_date.tzinfo is not None:
                    last_scan_date = last_scan_date.tz_localize(None)

                window = split_series.loc[(split_series.index > last_scan_date) & (split_series.index <= split_series.index.max())]
            else:
                window = split_series.tail(1)

            if not window.empty:
                positive = window[window > 0]
                if not positive.empty:
                    split_ratio = float(np.prod(positive.values))
        if not np.isfinite(split_ratio) or split_ratio <= 0:
            state.last_known_prices[sym] = current_price
            continue

        old_shares = state.shares[sym]
        theoretical_new_shares = old_shares * split_ratio
        new_shares = int(np.floor(theoretical_new_shares + 1e-12))
        old_entry = state.entry_prices.get(sym, current_price * split_ratio)
        new_entry = old_entry / split_ratio

        # Safely sweep fractional shares
        fractional_new_shares = max(0.0, theoretical_new_shares - new_shares)
        fractional_value = fractional_new_shares * current_price
        state.cash = round(state.cash + fractional_value, 10)

        logger.warning(
            "SPLIT DETECTED: %s stock_splits=%.6f shares %d→%d entry_price ₹%.2f→₹%.2f",
            sym, split_ratio, old_shares, new_shares, old_entry, new_entry,
        )
        state.shares[sym] = new_shares
        state.entry_prices[sym] = round(new_entry, 4)
        state.last_known_prices[sym] = current_price
        adjusted.append(sym)

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
    corrupted_paths = []

    for path in backups:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return PortfolioState.from_dict(json.load(f))
            except Exception as exc:
                logger.warning("Corrupted state at %s: %s", path, exc)
                corrupted_paths.append(path)

    if corrupted_paths:
        raise RuntimeError(
            "Portfolio state recovery failed: all discovered state files are corrupted "
            f"for '{name}' ({', '.join(corrupted_paths)})."
        )

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
    state.max_absent_periods = cfg.MAX_ABSENT_PERIODS

    today = pd.Timestamp(datetime.today().date())
    next_due = _next_rebalance_due(state.last_rebalance_date, cfg.REBALANCE_FREQ)
    rebalance_allowed = next_due is None or today >= next_due
    if not rebalance_allowed:
        logger.info(
            "[Scan] Cadence gate active: last rebalance %s, next due %s (%s). Scan will mark-to-market only.",
            state.last_rebalance_date,
            next_due.strftime("%Y-%m-%d"),
            cfg.REBALANCE_FREQ,
        )

    # Pass the actual desired terminal date; load_or_fetch applies the
    # yfinance exclusive-end +1 day correction internally.
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

    # Detect splits and sweep dividends BEFORE marking MTM values
    split_syms = detect_and_apply_splits(state, market_data, cfg)
    if split_syms:
        logger.warning("[Scan] Applied split adjustments for: %s", split_syms)

    use_adj = getattr(cfg, "AUTO_ADJUST_PRICES", True)

    close_d: Dict[str, pd.Series] = {}
    for sym in universe:
        ns = to_ns(sym)
        if ns in market_data:
            col = "Adj Close" if use_adj and "Adj Close" in market_data[ns].columns else "Close"
            close_d[to_bare(ns)] = market_data[ns][col].ffill()

    if not close_d:
        logger.warning("[Scan] No data available for any universe symbol.")
        for held_sym in list(state.shares.keys()):
            if held_sym not in close_d:
                state.absent_periods[held_sym] = int(state.absent_periods.get(held_sym, 0)) + 1
        return state, market_data

    _print_stage_status("Analysis", 0.35, f"Built close-price matrix for {len(close_d):,} active symbols.")

    close    = pd.DataFrame(close_d).sort_index()
    active   = list(close.columns)
    prices   = close.iloc[-1].values.astype(float)
    active_idx = {sym: i for i, sym in enumerate(active)}

    # Guard live PV mark-to-market against feed corruption (e.g., all-NaN close
    # for a held symbol). NaN here poisons pv and causes optimizer hard-fail.
    mtm_notional = 0.0
    for sym in state.shares:
        if sym in active_idx:
            px = prices[active_idx[sym]]
            if np.isnan(px) or px <= 0:
                px = float(state.last_known_prices.get(sym, 0.0))
            mtm_notional += state.shares.get(sym, 0) * px
    # FIX (Bug-D — Delisting PV Gap): Include held stocks that are absent from the
    # current scan universe (possibly delisted / suspended) at their last-known
    # price.  Omitting them under-counts total equity, causing the optimizer to
    # over-deploy cash and breach position-size limits on the next trade.
    for _absent_sym in state.shares:
        if _absent_sym not in active_idx:
            _fallback_px = state.last_known_prices.get(_absent_sym, 0.0)
            if _fallback_px > 0:
                _absent_n = int(state.absent_periods.get(_absent_sym, 0))
                _mtm_px = absent_symbol_effective_price(_fallback_px, _absent_n, cfg.MAX_ABSENT_PERIODS)
                mtm_notional += state.shares[_absent_sym] * _mtm_px
                logger.warning(
                    "[Scan] Held symbol '%s' absent from current market data "
                    "(possibly delisted/suspended). Applying absence haircut %.1f%% "
                    "to last-known price ₹%.2f for PV calculation (mark ₹%.2f). "
                    "Position will be force-closed after %d consecutive absent periods.",
                    _absent_sym,
                    (1.0 - (_mtm_px / _fallback_px if _fallback_px > 0 else 0.0)) * 100.0,
                    _fallback_px,
                    _mtm_px,
                    cfg.MAX_ABSENT_PERIODS,
                )
    pv = mtm_notional + state.cash
    initial_cash = state.cash
    initial_gross_exposure = mtm_notional / pv if pv > 0 else 1.0

    close_hist    = close.iloc[:-1]
    regime_score = compute_regime_score(idx_slice, cfg=cfg, universe_close_hist=close_hist)
    log_rets      = np.log1p(close_hist.pct_change(fill_method=None).clip(lower=-0.99)).replace([np.inf, -np.inf], np.nan)
    adv_arr       = compute_adv(market_data, active, cfg=cfg)
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
    _soft_cvar_breach    = False
    rebalanced_this_scan = False

    # ── Book CVaR screen ──────────────────────────────────────────────────────
    if state.shares:
        book_cvar = compute_book_cvar(state, prices, active, log_rets, cfg)
        hard_multiplier = getattr(cfg, "CVAR_HARD_BREACH_MULTIPLIER", 1.5)
        hard_breach_threshold = cfg.CVAR_DAILY_LIMIT * hard_multiplier

        if book_cvar > hard_breach_threshold:
            logger.warning(
                "[Scan] Book CVaR %.4f%% exceeds HARD limit %.4f%% (%.1fx) — "
                "skipping optimization, forcing immediate liquidation.",
                book_cvar * 100, hard_breach_threshold * 100, hard_multiplier,
            )
            state.consecutive_failures += 1
            apply_decay      = True
            _force_full_cash = True
            activate_override_on_stress(state, cfg)
        elif book_cvar > cfg.CVAR_DAILY_LIMIT + 1e-6:
            _soft_cvar_breach = True
            logger.info(
                "[Scan] Book CVaR soft breach %.4f%% (limit %.4f%%, hard %.4f%%) — "
                "running optimizer with CVaR constraint active.",
                book_cvar * 100, cfg.CVAR_DAILY_LIMIT * 100, hard_breach_threshold * 100,
            )

    if rebalance_allowed and not _force_full_cash:
        try:
            raw_daily, adj_scores, sel_idx, gate_counts = generate_signals(
                log_rets, adv_arr, cfg, prev_weights=state.weights
            )
            # MB-18 FIX: Log the per-gate rejection funnel at INFO level so the
            # portfolio manager can distinguish a volatility spike (knife gate) from
            # a data-provider outage (history gate) when the eligible universe shrinks.
            logger.info(
                "[Scan] Universe funnel: %d total → %d history-gated → %d ADV-gated "
                "→ %d knife-gated → %d selected.",
                gate_counts["total"],
                gate_counts["history_gated"],
                gate_counts["adv_gated"],
                gate_counts["knife_gated"],
                gate_counts["selected"],
            )
            if not sel_idx:
                raise OptimizationError("No valid universe candidates.", OptimizationErrorType.DATA)
            sel_syms      = [active[i] for i in sel_idx]
            sector_map    = get_sector_map(sel_syms, cfg=cfg)
            # FIX (Sector-Cap Strangulation): exclude "Unknown" from the integer
            # label space by assigning it the reserved sentinel -1.  The optimizer's
            # constraint builder skips sec_id == -1, so unknown-sector assets are
            # governed only by the global budget constraint instead of being grouped
            # into a synthetic super-sector that strangulates their combined weight.
            known_sectors  = sorted(s for s in set(sector_map.values()) if s != "Unknown")
            sec_idx        = {s: i for i, s in enumerate(known_sectors)}
            sector_labels  = np.array(
                [sec_idx.get(sector_map[sym], -1) for sym in sel_syms], dtype=int
            )

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
                # FIX (Risk-Breach Paralysis): if a soft CVaR breach is already
                # active and the optimizer fails (KKT infeasibility or post-check
                # rejection), the portfolio is in a mathematically unsafe state.
                # Waiting for 3 consecutive failures before triggering decay leaves
                # the live portfolio over-risk for potentially weeks.  Bypass the
                # counter and force decay on this exact bar.
                if _soft_cvar_breach:
                    logger.warning(
                        "Solver failure during active soft CVaR breach — "
                        "bypassing 3-failure wait, triggering immediate decay."
                    )
                    apply_decay = True
                elif state.consecutive_failures >= 3:
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

    if (rebalance_allowed or _force_full_cash) and (optimization_succeeded or apply_decay):
        _T_cvar = min(len(log_rets), cfg.CVAR_LOOKBACK)
        _scenario_losses = -(
            log_rets.iloc[-_T_cvar:]
            .reindex(columns=active, fill_value=0.0)
            .values
        )
        total_slippage = execute_rebalance(
            state, weights, prices, active, cfg,
            adv_shares=adv_arr,
            date_context=today, trade_log=trade_log,
            apply_decay    = apply_decay and not _exhaust_decay,
            scenario_losses = None if _exhaust_decay else _scenario_losses,
            force_rebalance_trades = _soft_cvar_breach,
        )
        state.last_rebalance_date = today.strftime("%Y-%m-%d")
        rebalanced_this_scan = True
        if _exhaust_decay:
            state.decay_rounds = 0
            state.consecutive_failures = 0

    _print_stage_status("Analysis", 0.85, "Applying rebalance decisions and updating portfolio marks...")

    price_dict = {sym: prices[active_idx[sym]] for sym in active}
    state.record_eod(price_dict)

    for held_sym in list(state.shares.keys()):
        if held_sym not in active_idx:
            state.absent_periods[held_sym] = int(state.absent_periods.get(held_sym, 0)) + 1
        else:
            state.absent_periods.pop(held_sym, None)

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

def _preserve_risk_metadata(source: PortfolioState, target: PortfolioState) -> None:
    """Copy stress-tracking risk fields from *source* into *target* in-place.

    Called when the user discards trade-level changes after a scan.  The
    holdings (shares, cash, weights, entry_prices) revert to their pre-scan
    values, but the risk engine must retain any stress signals that were
    detected during the scan (consecutive_failures, override state, decay
    progress).  Without this, a portfolio that detected three solver failures
    or an override trigger during a scan silently forgets the stress event
    the moment the user presses 'n'.
    """
    target.consecutive_failures = source.consecutive_failures
    target.override_cooldown    = source.override_cooldown
    target.override_active      = source.override_active
    target.decay_rounds         = source.decay_rounds
    target.absent_periods       = copy.deepcopy(source.absent_periods)


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
                _universe = fetch_nse_equity_universe()
            except UniverseFetchError as e:
                _universe = _prompt_survival_mode(e, "NSE Total")
                if _universe is None:
                    continue
            preview      = copy.deepcopy(states["nse_total"])
            preview, mkt = _run_scan(_universe, preview, "NSE TOTAL SCAN", cfg)
            mkt_cache["nse_total"] = mkt
            _print_status(preview, "PREVIEW — NSE TOTAL", mkt, cfg=cfg)
            if input(f"  {C.YLW}Save these changes? (y/n): {C.RST}").strip().lower() == "y":
                states["nse_total"] = preview
                save_portfolio_state(preview, "nse_total")
                print(f"  {C.GRN}[+] Saved permanently.{C.RST}")
            else:
                # FIX (State Desync Amnesia): discard trade-level changes but
                # preserve any risk metadata updated during the scan.  Stress
                # signals (consecutive_failures, override state, decay_rounds)
                # must survive a user rejection so the risk engine remembers
                # that the portfolio was under stress during this bar.
                _preserve_risk_metadata(source=preview, target=states["nse_total"])
                save_portfolio_state(states["nse_total"], "nse_total")
                print(f"  {C.GRY}[-] Trade changes discarded; risk metadata saved.{C.RST}")

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
                _preserve_risk_metadata(source=preview, target=states["nifty"])
                save_portfolio_state(states["nifty"], "nifty")
                print(f"  {C.GRY}[-] Trade changes discarded; risk metadata saved.{C.RST}")

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
                _preserve_risk_metadata(source=preview, target=states["custom"])
                save_portfolio_state(states["custom"], "custom")
                print(f"  {C.GRY}[-] Trade changes discarded; risk metadata saved.{C.RST}")

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

            end    = datetime.today().strftime("%Y-%m-%d")
            bt_cfg = load_optimized_config()

            # ── Custom universe path ───────────────────────────────────────────
            # get_historical_universe() returns [] for universe_type="custom"
            # (Fix #1 — no longer raises ValueError). We build the union from
            # the *current* live screener list instead, and warn the user that
            # this introduces survivorship bias before proceeding.
            if universe_identifier == "custom":
                custom_syms = _get_custom_universe()
                if not custom_syms:
                    print(f"  {C.RED}[!] No custom universe found. Cannot run backtest.{C.RST}")
                    print(f"  {C.GRY}Please verify the Screener.in URL or provide a local file and try again.{C.RST}")
                    continue

                print(f"\n  {C.B_RED}⚠  SURVIVORSHIP BIAS WARNING{C.RST}")
                print(f"  {C.YLW}Custom screener backtests use today's live stock list for ALL historical{C.RST}")
                print(f"  {C.YLW}rebalance dates. Stocks that were delisted, merged, or removed from your{C.RST}")
                print(f"  {C.YLW}screener between {start} and {end} are silently excluded.{C.RST}")
                print(f"  {C.YLW}This will inflate historical returns. Use Nifty 500 or NSE Total for{C.RST}")
                print(f"  {C.YLW}unbiased backtesting.{C.RST}")
                confirm = input(f"\n  {C.CYN}Proceed anyway? (y/n): {C.RST}").strip().lower()
                if confirm != "y":
                    print(f"  {C.GRY}Cancelled. Returning to main menu.{C.RST}")
                    continue

                historical_union = set(custom_syms)
                data = load_or_fetch(list(historical_union) + ["^NSEI", "^CRSLDX"], start, end, cfg=bt_cfg)

                try:
                    print_backtest_results(
                        run_backtest(data, universe_identifier, start, end, cfg=bt_cfg, universe=custom_syms)
                    )
                except RuntimeError as exc:
                    print(f"\n  {C.B_RED}[!] BACKTEST FAILED{C.RST}")
                    print(f"  {C.RED}{exc}{C.RST}\n")

            # ── Standard path (nifty500 / nse_total) ──────────────────────────
            else:
                all_target_dates = pd.date_range(start, end, freq=bt_cfg.REBALANCE_FREQ)
                historical_union = set()
                for target_date in all_target_dates:
                    members = get_historical_universe(universe_identifier, target_date)
                    historical_union.update(members)

                if not historical_union:
                    print(f"\n  {C.B_RED}[!] BACKTEST BLOCKED — No historical universe data found{C.RST}")
                    print(f"  {C.YLW}Run the following command to generate required snapshots:{C.RST}")
                    print(f"  {C.BLD}    python historical_builder.py{C.RST}")
                    print(f"  {C.GRY}Required files:{C.RST}")
                    print(f"  {C.GRY}    data/historical_nifty500.parquet  (or data/historical_nse_total.parquet){C.RST}\n")
                    continue

                data = load_or_fetch(list(historical_union) + ["^NSEI", "^CRSLDX"], start, end, cfg=bt_cfg)

                try:
                    print_backtest_results(run_backtest(data, universe_identifier, start, end, cfg=bt_cfg))
                except RuntimeError as exc:
                    print(f"\n  {C.B_RED}[!] BACKTEST FAILED — Historical Universe Data Missing{C.RST}")
                    print(f"  {C.RED}{exc}{C.RST}")
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
            print(f"\n  {C.B_RED}WARNING: This will reset your holdings to zero (cash only).{C.RST}")
            print(f"  {C.GRY}Market data cache and optimal_cfg.json are NOT affected.{C.RST}")
            print(f"  {C.GRY}Use this when you want to start a fresh portfolio with new capital.{C.RST}")
            confirm = input(f"  {C.CYN}Type 'YES' to confirm: {C.RST}").strip()
            if confirm.upper() == "YES":
                # Only delete portfolio state files — never touch the data cache
                # or optimal_cfg.json. Wiping those forces a full re-download and
                # loses optimized parameters, neither of which is wanted here.
                for n in ["nse_total", "nifty", "custom"]:
                    p = f"data/portfolio_state_{n}.json"
                    for suffix in ["", ".bak.0", ".bak.1", ".bak.2"]:
                        target = p + suffix
                        if not os.path.exists(target):
                            continue
                        # FIX (Transactional File Wipe): on Windows, antivirus and
                        # filesystem indexers briefly lock state files, causing a raw
                        # os.remove() to raise PermissionError and crash the loop
                        # mid-wipe.  The remaining files are left on disk in an
                        # inconsistent state.  We retry up to 3 times with a short
                        # back-off; if all retries fail we log a warning and continue
                        # to the next file rather than aborting the entire wipe.
                        _removed = False
                        for _attempt in range(3):
                            try:
                                os.remove(target)
                                _removed = True
                                break
                            except OSError:
                                if _attempt < 2:
                                    time.sleep(0.1)
                        if not _removed:
                            logger.warning(
                                "[Clear] Could not delete '%s' after 3 attempts "
                                "(file may be locked). Skipping.",
                                target,
                            )
                states    = {"nse_total": PortfolioState(), "nifty": PortfolioState(), "custom": PortfolioState()}
                mkt_cache = {"nse_total": {}, "nifty": {}, "custom": {}}
                print(f"  {C.GRN}[+] Portfolio holdings cleared. Cache and config untouched.{C.RST}")
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
