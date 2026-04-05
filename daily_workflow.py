"""
daily_workflow.py — Ultimate Momentum v11.48
============================================
Interactive CLI for live scanning, status display, and backtesting.
Features robust capital management, direct Screener.in web scraping,
Dividend Sweeping, and Impact-Aligned Rebalancing.

BUG FIXES (murder board):
- FIX-MB-GATENAMES: Updated gate_counts log line in _run_scan to use renamed
  keys "history_failed" / "adv_failed" / "knife_failed" (were "history_gated" /
  "adv_gated" / "knife_gated"). The old names implied counts of symbols that
  passed — the values are actually counts of symbols removed by each gate.
  Log message updated to say "removed" instead of "selected" for clarity.
- FIX-MB-FDLEAK: save_portfolio_state directory fsync on POSIX now uses
  try/finally to guarantee the directory file descriptor is closed even when
  os.fsync() raises (e.g. on a remounted read-only filesystem). Without the
  finally block a raised exception left the fd open indefinitely.
- FIX-MB-RESIDUALCASH: _run_scan was sizing residual-cash buys against pv_exec
  computed BEFORE execute_rebalance ran sells. After sells cash is lower, so
  the residual allocation could overshoot and drive state.cash negative after
  slippage deduction. The residual allocation is now handled entirely inside
  execute_rebalance (which already does multi-pass residual allocation on the
  final post-sell cash balance) — the separate pre-call residual sizing block
  that existed only in the scan path has been removed, making the live scan
  path consistent with the backtest path.
- FIX-MB2-EQUITYCAP: _run_scan unconditionally overwrote state.equity_hist_cap
  and state.max_absent_periods with cfg defaults on every call, discarding any
  user-persisted custom values. Now only applies the cfg default when the field
  still holds its dataclass default value.
- FIX-MB2-PAPERRISK: save_portfolio_state in PAPER_MODE now writes a thin
  risk-metadata-only overlay file (portfolio_risk_{name}.json) so that
  consecutive_failures, override_cooldown, decay_rounds and absent_periods
  survive process restarts. load_portfolio_state merges this overlay back.
  Without this fix a crash during a CVaR breach in paper mode would cause an
  immediate unrestricted rebalance on the next run.
- FIX-MB-H-04: _run_scan post-scan absent_periods loop now only runs when
  execute_rebalance did NOT fire (rebalanced_this_scan=False). On rebalance
  days execute_rebalance owns absent_periods accounting; the post-scan loop
  was incrementing it a second time, halving the effective MAX_ABSENT_PERIODS
  grace period and causing premature force-close of suspended stocks.
- PROD-FIX-1: load_portfolio_state now logs CRITICAL and returns a safe
  zero-position PortfolioState() when all state files are corrupted, rather
  than raising RuntimeError and blocking restart.
- PROD-FIX-3: _run_scan circuit breaker tracks consecutive empty-universe
  scans and raises RuntimeError after EMPTY_UNIVERSE_HALT_AFTER (default 3)
  consecutive failures, preventing silent full-cash drift during outages.
- PROD-FIX-5: ScanContext injects a per-scan correlation_id into every log
  record. DeadLetterTracker accumulates stale-price symbols and emits a
  single structured report at flush() time. configure_logging() replaces
  basicConfig with JsonFormatter for machine-parseable structured output.
- FIX-WRITE-VERIFY: save_portfolio_state reads back and JSON-parses the
  written file immediately after os.replace(), catching filesystem-level
  corruption while backups and in-memory state are still intact.
- FIX-STALE-PRICE: _run_scan suppresses the rebalance (not force-liquidations)
  when any held symbol has price data older than 2 trading days, preventing
  incorrect position sizing against stale prices during provider outages.
- FIX-CB-PERSIST: _CONSECUTIVE_EMPTY_SCANS is now persisted to
  data/circuit_breaker.json on every change and loaded at startup, so the
  circuit breaker threshold survives process restarts during outages.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import pathlib
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

# OSQP must be imported BEFORE numpy/pandas on Python 3.13/Windows to avoid
# a silent ABI crash (exit code 0xC0000005). momentum_engine imports osqp,
# but by the time Python resolves that import numpy is already loaded — too late.
import osqp  # noqa: F401

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
from data_cache import load_or_fetch
from backtest_engine import run_backtest, print_backtest_results
from signals import generate_signals, compute_adv, compute_regime_score
from log_config import ScanContext, DeadLetterTracker
from shared_utils import atomic_write_file
from shared_constants import (
    COLUMN_STOCK_SPLITS,
    TIMEZONE_IST,
    MARKET_INDEX_NSEI,
    MARKET_INDEX_CRSLDX,
    LABEL_CUSTOM_SCREENER,
    LABEL_NSE_TOTAL,
    LABEL_NIFTY_500,
)

__version__ = "11.48"

BACKUP_GENERATIONS = 3
PAPER_MODE = False
DEFAULT_INITIAL_CAPITAL = float(PortfolioState().cash)



# PROD-FIX-3: Circuit-breaker counter for consecutive scans that return an
# empty universe (provider outage / all symbols filtered).  Reset to zero on
# any scan that yields at least one symbol.  When it hits
# _EMPTY_UNIVERSE_HALT_AFTER the scan raises RuntimeError so the caller
# (cron job / supervisor) knows to page an operator rather than silently
# running indefinitely with no positions.
_EMPTY_UNIVERSE_HALT_AFTER: int = int(os.environ.get("EMPTY_UNIVERSE_HALT_AFTER", "3"))
_CIRCUIT_BREAKER_FILE = "data/circuit_breaker.json"
_CIRCUIT_BREAKER_LOCK_FILE = "data/circuit_breaker.lock"


@dataclass
class CircuitBreaker:
    # ARCH-FIX-7
    """CircuitBreaker type used by the backtesting system."""
    count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def increment(self) -> int:
        with self._lock:
            self.count += 1
            return self.count

    def reset(self) -> None:
        with self._lock:
            self.count = 0

    def load(self, path: str) -> None:
        """Load persisted count from disk; silently ignore missing file."""
        json_path = pathlib.Path(path)
        lock_path = pathlib.Path(_CIRCUIT_BREAKER_LOCK_FILE)
        with self._lock:
            if not json_path.exists() and lock_path.exists():
                logger.warning("circuit_breaker.lock found without JSON; treating count as 1")  # ARCH-FIX-10
                self.count = 1
                return
            try:
                if json_path.exists():
                    self.count = int(json.loads(json_path.read_text(encoding="utf-8")).get("consecutive_empty", 0))
            except (json.JSONDecodeError, ValueError) as exc:
                logger.error("Failed to parse circuit breaker state %s: %s", json_path, exc)  # ARCH-FIX-7
                self.count = 1 if lock_path.exists() else 0
            except OSError as exc:
                logger.warning("Failed to read circuit breaker state %s: %s", json_path, exc)  # ARCH-FIX-7
                self.count = 1 if lock_path.exists() else 0

    def save(self, path: str) -> None:
        """Persist count to disk; log error on failure, write lock sentinel."""
        with self._lock:  # ARCH-FIX-7
            try:
                os.makedirs("data", exist_ok=True)
                snapshot = int(self.count)
                atomic_write_file(
                    pathlib.Path(path),
                    lambda tmp: tmp.write_text(
                        json.dumps({"consecutive_empty": snapshot}),
                        encoding="utf-8",
                    ),
                    suffix=".tmp",
                    fsync_file=True,
                    fsync_dir=True,
                )
                if pathlib.Path(_CIRCUIT_BREAKER_LOCK_FILE).exists():
                    pathlib.Path(_CIRCUIT_BREAKER_LOCK_FILE).unlink()
            except Exception as exc:
                logger.error("Failed to persist circuit breaker count: %s", exc)  # ARCH-FIX-10
                try:
                    pathlib.Path(_CIRCUIT_BREAKER_LOCK_FILE).touch()
                except Exception:
                    pass


def _pending_sentinel_path(name: str) -> pathlib.Path:
    return pathlib.Path(f"data/pending_rebalance_{name}.json")


def _pending_claim_path(name: str) -> pathlib.Path:
    return pathlib.Path(f"data/pending_rebalance_{name}.claim")


def _write_pending_sentinel(name: str, token: str, date_str: str) -> pathlib.Path:
    # ARCH-FIX-3
    """_write_pending_sentinel operation.
    
    Args:
        name (str): Input parameter.
        token (str): Input parameter.
        date_str (str): Input parameter.
    
    Returns:
        pathlib.Path: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    os.makedirs("data", exist_ok=True)
    path = _pending_sentinel_path(name)
    payload = {
        "date": date_str,
        "token_hash": hashlib.sha256(token.encode("utf-8")).hexdigest(),
    }
    atomic_write_file(
        path,
        lambda tmp: tmp.write_text(json.dumps(payload), encoding="utf-8"),
        suffix=".tmp",
        fsync_file=True,
        fsync_dir=True,
    )
    return path


def _is_claim_stale(
    claim_path: pathlib.Path,
    current_date_str: str,
    max_age_hours: int,
    stat_result: os.stat_result,
    pre_read_claim_date_str: Optional[str] = None,
) -> bool:
    """_is_claim_stale operation.

    Args:
        claim_path (pathlib.Path): Input parameter.
        current_date_str (str): Input parameter.
        max_age_hours (int): Input parameter.
        stat_result (os.stat_result): Input parameter.
        pre_read_claim_date_str (Optional[str]): Input parameter.

    Returns:
        bool: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    mtime = datetime.fromtimestamp(stat_result.st_mtime)
    if pre_read_claim_date_str is not None:
        claim_date_str = pre_read_claim_date_str
    else:
        claim_date_str = ""
        try:
            existing_claim = json.loads(claim_path.read_text(encoding="utf-8"))
            if isinstance(existing_claim, dict):
                claim_date_str = str(existing_claim.get("date", ""))
        except (json.JSONDecodeError, OSError):
            claim_date_str = ""

    return (
        claim_date_str != current_date_str
        or (datetime.now() - mtime).total_seconds() > (max_age_hours * 3600)
    )


def _try_claim_pending_sentinel(name: str, token: str, date_str: str) -> bool:
    """_try_claim_pending_sentinel operation.
    
    Args:
        name (str): Input parameter.
        token (str): Input parameter.
        date_str (str): Input parameter.
    
    Returns:
        bool: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    os.makedirs("data", exist_ok=True)
    claim_path = _pending_claim_path(name)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    # Try creating the claim file with exclusive flags
    try:
        fd = os.open(str(claim_path), flags)
    except FileExistsError:
        # Claim file exists — check if it's stale
        try:
            stat_result = claim_path.stat()
            mtime = datetime.fromtimestamp(stat_result.st_mtime)

            # Try to read the existing claim to check its date
            try:
                existing_claim = json.loads(claim_path.read_text(encoding="utf-8"))
                if isinstance(existing_claim, dict):
                    claim_date_str = existing_claim.get("date", "")
                else:
                    # Malformed claim (not a dict) — treat as stale
                    claim_date_str = ""
            except (json.JSONDecodeError, OSError):
                # Corrupted or unreadable claim — treat as stale
                claim_date_str = ""

            is_stale = _is_claim_stale(
                claim_path=claim_path,
                current_date_str=date_str,
                max_age_hours=24,
                stat_result=stat_result,
                pre_read_claim_date_str=str(claim_date_str),
            )

            if is_stale:
                logger.warning(
                    "Stale claim file detected for %s (claim_date=%s, requested=%s, age=%.1fh). Removing and retrying.",
                    name, claim_date_str, date_str, (datetime.now() - mtime).total_seconds() / 3600
                )
                try:
                    claim_path.unlink()
                except OSError as e:
                    logger.error("Failed to remove stale claim file %s: %s", claim_path, e)
                    return False

                # Retry the claim once after removing stale file
                try:
                    fd = os.open(str(claim_path), flags)
                except FileExistsError:
                    # Another process claimed it in the race window
                    return False
            else:
                # Not stale — another process holds a valid claim
                return False
        except OSError as e:
            logger.error("Failed to check claim file staleness for %s: %s", claim_path, e)
            return False

    # Successfully created the claim file (either first try or after removing stale)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            payload = {
                "date": date_str,
                "token_hash": hashlib.sha256(token.encode("utf-8")).hexdigest(),
            }
            fh.write(json.dumps(payload))
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        try:
            claim_path.unlink()
        except OSError:
            pass
        raise
    if os.name == "posix":
        dir_fd = os.open(str(claim_path.parent), getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    return True


def _clear_pending_sentinel(name: str) -> None:
    # ARCH-FIX-3
    """_clear_pending_sentinel operation.
    
    Args:
        name (str): Input parameter.
    
    Returns:
        None: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    try:
        _pending_sentinel_path(name).unlink()
    except FileNotFoundError:
        pass
    try:
        _pending_claim_path(name).unlink()
    except FileNotFoundError:
        pass


def _load_pending_sentinel(name: str) -> dict | None:
    # ARCH-FIX-3
    """_load_pending_sentinel operation.
    
    Args:
        name (str): Input parameter.
    
    Returns:
        dict | None: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    path = _pending_sentinel_path(name)
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
        else:
            # Malformed sentinel (not a dict) — treat as missing
            logger.debug("_load_pending_sentinel: malformed sentinel for %s (not a dict), treating as missing", name)
            return None
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("_load_pending_sentinel: failed to load sentinel for %s: %s", name, e)
        return None

# ─── ANSI colour palette ─────────────────────────────────────────────────────

class C:
    """C type used by the backtesting system."""
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
_circuit_breaker = CircuitBreaker()
_circuit_breaker.load(_CIRCUIT_BREAKER_FILE)  # ARCH-FIX-7

_DEFAULT_SCREENER_URL = os.environ.get(
    "SCREENER_URL",
    "https://www.screener.in/screens/3506127/hello/",
)

def _validate_config_cross_fields(cfg: UltimateConfig) -> UltimateConfig:
    """_validate_config_cross_fields operation.

    Args:
        cfg (UltimateConfig): Input parameter.

    Returns:
        UltimateConfig: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    defaults = UltimateConfig()
    if cfg.HALFLIFE_FAST > cfg.HALFLIFE_SLOW:
        logger.error(
            "[Config] optimal_cfg.json has HALFLIFE_FAST (%d) > HALFLIFE_SLOW (%d) - invalid. Resetting both.",
            cfg.HALFLIFE_FAST, cfg.HALFLIFE_SLOW,
        )
        cfg.HALFLIFE_FAST = defaults.HALFLIFE_FAST
        cfg.HALFLIFE_SLOW = defaults.HALFLIFE_SLOW
    if cfg.MIN_EXPOSURE_FLOOR > 1.0 or cfg.MIN_EXPOSURE_FLOOR < 0.0:
        logger.error(
            "[Config] optimal_cfg.json has MIN_EXPOSURE_FLOOR=%.4f outside [0,1]. Resetting.",
            cfg.MIN_EXPOSURE_FLOOR,
        )
        cfg.MIN_EXPOSURE_FLOOR = defaults.MIN_EXPOSURE_FLOOR
    if cfg.CVAR_DAILY_LIMIT <= 0.0 or cfg.CVAR_DAILY_LIMIT > 0.5:
        logger.error(
            "[Config] optimal_cfg.json has CVAR_DAILY_LIMIT=%.4f outside (0, 0.5]. Resetting.",
            cfg.CVAR_DAILY_LIMIT,
        )
        cfg.CVAR_DAILY_LIMIT = defaults.CVAR_DAILY_LIMIT
    if not isinstance(cfg.MAX_POSITIONS, int) or cfg.MAX_POSITIONS < 2:
        logger.error(
            "[Config] optimal_cfg.json has MAX_POSITIONS=%r outside [2, infinity). Resetting.",
            cfg.MAX_POSITIONS,
        )
        cfg.MAX_POSITIONS = defaults.MAX_POSITIONS
    if not isinstance(cfg.SIGNAL_LAG_DAYS, int) or cfg.SIGNAL_LAG_DAYS < 0:
        logger.error(
            "[Config] optimal_cfg.json has SIGNAL_LAG_DAYS=%r < 0. Resetting.",
            cfg.SIGNAL_LAG_DAYS,
        )
        cfg.SIGNAL_LAG_DAYS = defaults.SIGNAL_LAG_DAYS
    return cfg


def load_optimized_config() -> UltimateConfig:
    """load_optimized_config operation.

    Returns:
        UltimateConfig: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    cfg = UltimateConfig()
    if os.path.exists("data/optimal_cfg.json"):
        with open("data/optimal_cfg.json", "r", encoding="utf-8") as f:
            try:
                best_params = json.load(f)
            except json.JSONDecodeError as e:
                logger.error("[Config] optimal_cfg.json is corrupted: %s. Using defaults.", e)
                return cfg
            valid_fields = UltimateConfig.__dataclass_fields__
            for k, v in best_params.items():
                if k not in valid_fields:
                    logger.warning("[Config] Ignoring unknown/stale optimized parameter: %s", k)
                    continue
                setattr(cfg, k, v)
    return _validate_config_cross_fields(cfg)
def _render_meter(label: str, progress: float, width: int = 30) -> str:
    clipped = max(0.0, min(1.0, progress))
    filled = int(round(width * clipped))
    bar = f"{'█' * filled}{'░' * (width - filled)}"
    pct = f"{clipped * 100:5.1f}%"
    return f"  {C.CYN}{label:<18}{C.RST} [{bar}] {C.BLD}{pct}{C.RST}"

def _print_stage_status(label: str, progress: float, detail: str) -> None:
    print(_render_meter(label, progress))
    print(f"  {C.GRY}{detail}{C.RST}")


def _next_rebalance_due(last_rebalance_date: str, rebalance_freq: str) -> Optional[pd.Timestamp]:
    """_next_rebalance_due operation.
    
    Args:
        last_rebalance_date (str): Input parameter.
        rebalance_freq (str): Input parameter.
    
    Returns:
        Optional[pd.Timestamp]: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
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
    """_scrape_screener operation.
    
    Args:
        base_url (str): Input parameter.
    
    Returns:
        List[str]: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
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

    _TIMEOUT = (5, 30)

    symbols: set[str] = set()
    page = 1
    max_pages = 50

    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query)
    qs.pop('page', None)
    clean_url = urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))

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
            if resp.status_code != 200:
                break

            soup = BeautifulSoup(resp.text, 'html.parser')
            links = soup.find_all('a', href=re.compile(r'^/company/[^/]+/(?:consolidated/)?$'))
            if links is None:
                links = []

            before_count = len(symbols)
            for link in links:
                match = re.search(r'/company/([^/]+)/', str(link['href']))
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
    """_filter_valid_custom_tickers operation.
    
    Args:
        tickers (List[str]): Input parameter.
    
    Returns:
        List[str]: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    normalized = [raw.strip().upper() for raw in tickers]
    filtered = [sym for sym in normalized if sym and not sym.isdigit()]
    invalid_count = sum(1 for sym in normalized if sym.isdigit())

    if invalid_count:
        logger.warning(
            "[Universe] Ignored %d non-NSE numeric ticker code(s) from custom screener.",
            invalid_count,
        )

    return list(dict.fromkeys(filtered))

def _get_custom_universe() -> List[str]:
    """_get_custom_universe operation.
    
    Returns:
        List[str]: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
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
                with open(f, "r", encoding="utf-8") as file:
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
                        print(f"  {C.GRY}Cancelled. Returning empty universe.{C.RST}")
                        return []
            except Exception as e:
                logger.error("[Screener] Failed to read %s: %s", f, e)
    return []

def _check_and_prompt_initial_capital(state: PortfolioState, label: str, name: str) -> None:
    """_check_and_prompt_initial_capital operation.
    
    Args:
        state (PortfolioState): Input parameter.
        label (str): Input parameter.
        name (str): Input parameter.
    
    Returns:
        None: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    if not state.shares and not state.equity_hist and abs(state.cash - DEFAULT_INITIAL_CAPITAL) < 1.0:
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



def _sweep_dividends(state: PortfolioState, sym: str, row: pd.DataFrame, cfg: UltimateConfig) -> None:
    """_sweep_dividends operation.

    Args:
        state (PortfolioState): Input parameter.
        sym (str): Input parameter.
        row (pd.DataFrame): Input parameter.
        cfg (UltimateConfig): Input parameter.

    Returns:
        None: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    if not getattr(cfg, "DIVIDEND_SWEEP", True) or "Dividends" not in row.columns:
        return
    if getattr(cfg, "AUTO_ADJUST_PRICES", True):
        return
    dividends = row["Dividends"][row["Dividends"] > 0]
    if dividends.empty:
        return
    shares_held = state.shares.get(sym, 0)
    if shares_held <= 0:
        return
    last_event_id = state.dividend_ledger.get(sym, "")
    last_event_date = last_event_id.split(':')[0] if last_event_id else "1900-01-01"
    for div_date, div_val in dividends.items():
        div_date_str = pd.Timestamp(div_date).strftime("%Y-%m-%d")
        if div_date_str <= last_event_date:
            continue
        div_val_float = float(div_val)
        state.cash = round(state.cash + (div_val_float * shares_held), 10)
        state.dividend_ledger[sym] = f"{div_date_str}:{div_val_float:.8f}"
        logger.info(
            "DIVIDEND SWEEP: %s distributed ₹%.2f per share (x %d shares) on %s. Added to cash.",
            sym, div_val_float, shares_held, div_date_str
        )
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

        _sweep_dividends(state, sym, row, cfg)

        current_price = float(row["Close"].iloc[-1])
        if not np.isfinite(current_price) or current_price <= 0:
            continue

        if getattr(cfg, "AUTO_ADJUST_PRICES", True):
            # [PHASE 2 FIX] C-03: When AUTO_ADJUST_PRICES=True, yfinance
            # already adjusts historical Close/Adj Close prices for splits.
            # However, yfinance does NOT adjust our state.shares — that is
            # the engine's responsibility.  The original `continue` here
            # skipped ALL split logic, so any stock split would leave share
            # counts un-adjusted while prices were halved, silently halving
            # the portfolio value and potentially triggering CVaR cascades.
            #
            # Fix: use the explicit `Stock Splits` column (always present in
            # yfinance data) to detect split events and adjust shares/entry
            # accordingly.  Guard against double-application via the
            # dividend_ledger marker keyed by date:ratio.
            if COLUMN_STOCK_SPLITS in row.columns:
                split_series = row[COLUMN_STOCK_SPLITS].fillna(0.0)
                positive_splits = split_series[split_series > 0].sort_index()
                if not positive_splits.empty and sym in state.shares:
                    marker_key = f"split:{sym}"
                    raw_markers = state.dividend_ledger.get(marker_key, "")
                    if isinstance(raw_markers, str):
                        applied_event_ids = {m for m in raw_markers.split("|") if m}
                    elif isinstance(raw_markers, (list, tuple, set)):
                        applied_event_ids = {str(v) for v in raw_markers if str(v)}
                    else:
                        raw = str(raw_markers or "")
                        applied_event_ids = {m for m in raw.split("|") if m}
                    for split_date, split_val in positive_splits.items():
                        split_date_str = pd.Timestamp(split_date).strftime("%Y-%m-%d")
                        event_id = f"{split_date_str}:{split_val:.8f}"
                        if event_id in applied_event_ids:
                            continue  # already applied — skip

                        old_shares = state.shares[sym]
                        theoretical = old_shares * split_val
                        new_shares = int(np.floor(theoretical + 1e-12))
                        old_entry = state.entry_prices.get(sym, current_price * split_val)
                        new_entry = old_entry / split_val

                        fractional = max(0.0, theoretical - new_shares)
                        one_way_slip = (cfg.ROUND_TRIP_SLIPPAGE_BPS / 2) / 10_000.0
                        fractional_value = fractional * current_price * max(0.0, 1.0 - one_way_slip)
                        state.cash = round(state.cash + fractional_value, 10)

                        logger.warning(
                            "SPLIT DETECTED (adj-price mode): %s ratio=%.6f "
                            "shares %d→%d entry ₹%.2f→₹%.2f",
                            sym, split_val, old_shares, new_shares, old_entry, new_entry,
                        )
                        state.shares[sym] = new_shares
                        state.entry_prices[sym] = round(new_entry, 4)
                        applied_event_ids.add(event_id)
                        state.dividend_ledger[marker_key] = "|".join(sorted(applied_event_ids))
                        adjusted.append(sym)

            state.last_known_prices[sym] = current_price
            continue

        split_ratio = 0.0
        if COLUMN_STOCK_SPLITS in row.columns and not row[COLUMN_STOCK_SPLITS].empty:
            split_series = row[COLUMN_STOCK_SPLITS].fillna(0.0)
            split_start_date = None
            if state.last_rebalance_date:
                try:
                    split_start_date = pd.Timestamp(state.last_rebalance_date)
                except (ValueError, TypeError):
                    split_start_date = None

            # FIX-SPLIT-FIRST-RUN: on the first live scan we must not compound
            # every historical split in the provider payload onto today's share
            # count.  Prefer the position's entry marker (stored in
            # dividend_ledger when the position is opened) as the lower bound.
            # If no trustworthy anchor exists, conservatively consider only the
            # latest explicit split signal rather than replaying the full series.
            if split_start_date is None:
                marker = state.dividend_ledger.get(sym, "")
                marker_date = marker.split(":", 1)[0] if marker else ""
                if marker_date:
                    try:
                        split_start_date = pd.Timestamp(marker_date)
                    except (ValueError, TypeError):
                        split_start_date = None

            if split_start_date is not None:
                split_index_tz = getattr(split_series.index, "tz", None)
                if split_index_tz is not None:
                    if split_start_date.tzinfo is None:
                        split_start_date = split_start_date.tz_localize(split_index_tz)
                    else:
                        split_start_date = split_start_date.tz_convert(split_index_tz)
                elif split_start_date.tzinfo is not None:
                    split_start_date = split_start_date.tz_localize(None)

                window = split_series.loc[split_series.index > split_start_date]
            else:
                # When no trustworthy anchor exists at all, prefer compounding the
                # explicit split signals present in the payload rather than silently
                # dropping earlier split events inside the visible window. This keeps
                # manual/fixture states without last_rebalance_date or a position
                # marker internally consistent when multiple splits appear.
                #
                # Residual risk: a truly anchor-free restored/manual state can still
                # over-apply older historical splits because there is no reliable way
                # to distinguish "pre-position" events from "post-position" events.
                # Normal live positions should carry either last_rebalance_date or a
                # dividend_ledger marker and therefore avoid this fallback path.
                window = split_series[split_series > 0]

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

        fractional_new_shares = max(0.0, theoretical_new_shares - new_shares)
        one_way_slip = (cfg.ROUND_TRIP_SLIPPAGE_BPS / 2) / 10_000.0
        fractional_value = fractional_new_shares * current_price * max(0.0, 1.0 - one_way_slip)
        state.cash = round(state.cash + fractional_value, 10)

        logger.warning(
            "SPLIT DETECTED: %s stock_splits=%.6f shares %d→%d entry_price ₹%.2f→₹%.2f",
            sym, split_ratio, old_shares, new_shares, old_entry, new_entry,
        )
        state.shares[sym] = new_shares
        state.entry_prices[sym] = round(new_entry, 4)
        state.last_known_prices[sym] = current_price
        adjusted.append(sym)

    return list(dict.fromkeys(adjusted))

# ─── State persistence ────────────────────────────────────────────────────────
def _rotate_backup_chain(state_file: pathlib.Path, generations: int) -> None:
    """_rotate_backup_chain operation.

    Args:
        state_file (pathlib.Path): Input parameter.
        generations (int): Input parameter.

    Returns:
        None: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    for i in range(generations - 1, -1, -1):
        src = state_file.with_suffix(f"{state_file.suffix}.bak.{i}")
        dst = state_file.with_suffix(f"{state_file.suffix}.bak.{i + 1}")
        if src.exists():
            shutil.copy2(src, dst)
    if state_file.exists():
        shutil.copy2(state_file, state_file.with_suffix(f"{state_file.suffix}.bak.0"))


def _verify_state_readback(state_file: pathlib.Path, expected_state: PortfolioState) -> bool:
    """_verify_state_readback operation.

    Args:
        state_file (pathlib.Path): Input parameter.
        expected_state (PortfolioState): Input parameter.

    Returns:
        bool: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    with state_file.open("r", encoding="utf-8") as vf:
        parsed = json.loads(vf.read())
    expected = expected_state.to_dict()
    return parsed == expected

def save_portfolio_state(state: PortfolioState, name: str) -> None:
    """save_portfolio_state operation.

    Args:
        state (PortfolioState): Input parameter.
        name (str): Input parameter.

    Returns:
        None: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    if PAPER_MODE:
        print(f"  {C.YLW}[!] Paper mode active. Trades not saved; risk metadata persisted.{C.RST}")
        os.makedirs("data", exist_ok=True)
        risk_file = f"data/portfolio_risk_{name}.json"
        risk_payload = {
            "consecutive_failures": state.consecutive_failures,
            "override_active": state.override_active,
            "override_cooldown": state.override_cooldown,
            "decay_rounds": state.decay_rounds,
            "absent_periods": dict(sorted(state.absent_periods.items())),
            "last_rebalance_date": state.last_rebalance_date,
        }
        try:
            tmp = f"{risk_file}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(risk_payload, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, risk_file)
            if os.name == "posix":
                dir_path = os.path.dirname(risk_file) or "."
                dir_fd = os.open(dir_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except (OSError, IOError, TypeError, ValueError) as exc:
            logger.warning("Paper-mode risk metadata save failed for '%s': %s", name, exc)
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        _clear_pending_sentinel(name)
        return

    os.makedirs("data", exist_ok=True)
    state_file = pathlib.Path(f"data/portfolio_state_{name}.json")
    risk_file = pathlib.Path(f"data/portfolio_risk_{name}.json")
    tmp_file = pathlib.Path(f"{state_file}.tmp")
    try:
        _rotate_backup_chain(state_file=state_file, generations=BACKUP_GENERATIONS)

        with tmp_file.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_file, state_file)

        try:
            if not _verify_state_readback(state_file=state_file, expected_state=state):
                raise RuntimeError("State file semantic corruption: read-back values do not match in-memory state.")
        except (json.JSONDecodeError, RuntimeError) as jexc:
            logger.critical(
                "State file write-back verification FAILED for '%s': %s. "
                "The file on disk may be corrupted. Backups are intact. "
                "Do not restart the process until the issue is resolved.",
                name, jexc,
            )
            raise

        if os.name == "posix":
            dir_fd = os.open("data", getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

        if risk_file.exists():
            try:
                os.remove(risk_file)
            except OSError as exc:
                logger.warning("Could not remove stale paper-mode risk overlay for '%s': %s", name, exc)
        _clear_pending_sentinel(name)

    except (OSError, IOError, TypeError, ValueError, RuntimeError) as exc:
        logger.error("Durable save failed for '%s': %s", name, exc)
        if tmp_file.exists():
            try:
                os.remove(tmp_file)
            except OSError as cleanup_exc:
                logger.debug("Cleanup failed for tmp state file '%s': %s", tmp_file, cleanup_exc)
        try:
            _clear_pending_sentinel(name)
        except Exception as sentinel_exc:
            logger.debug("Cleanup failed for pending sentinel '%s': %s", name, sentinel_exc)
        raise
def _apply_risk_overlay(ps: PortfolioState, name: str) -> PortfolioState:
    """_apply_risk_overlay operation.

    Args:
        ps (PortfolioState): Input parameter.
        name (str): Input parameter.

    Returns:
        PortfolioState: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    risk_file = f"data/portfolio_risk_{name}.json"
    if not os.path.exists(risk_file):
        return ps
    try:
        with open(risk_file, "r", encoding="utf-8") as rf:
            risk = json.load(rf)

        def _ri(key, fallback):
            v = risk.get(key)
            return int(v) if v is not None else fallback

        def _rb(key, fallback):
            v = risk.get(key)
            if v is None:
                return fallback
            if isinstance(v, bool):
                return v
            return bool(int(v))

        try:
            ps.consecutive_failures = _ri("consecutive_failures", ps.consecutive_failures)
        except (TypeError, ValueError) as _e:
            logger.warning("Risk overlay: could not merge consecutive_failures: %s", _e)
        try:
            ps.override_active = _rb("override_active", ps.override_active)
        except (TypeError, ValueError) as _e:
            logger.warning("Risk overlay: could not merge override_active: %s", _e)
        try:
            ps.override_cooldown = _ri("override_cooldown", ps.override_cooldown)
        except (TypeError, ValueError) as _e:
            logger.warning("Risk overlay: could not merge override_cooldown: %s", _e)
        try:
            ps.decay_rounds = _ri("decay_rounds", ps.decay_rounds)
        except (TypeError, ValueError) as _e:
            logger.warning("Risk overlay: could not merge decay_rounds: %s", _e)
        absent_raw = risk.get("absent_periods")
        if isinstance(absent_raw, dict):
            ps.absent_periods.clear()
            for k, v in absent_raw.items():
                try:
                    ps.absent_periods[str(k)] = int(v)
                except (TypeError, ValueError):
                    pass
        try:
            lrd = risk.get("last_rebalance_date")
            if lrd and isinstance(lrd, str):
                ps.last_rebalance_date = lrd
        except (TypeError, ValueError) as _e:
            logger.warning("Risk overlay: could not merge last_rebalance_date: %s", _e)
    except (json.JSONDecodeError, ValueError, TypeError, KeyError, OSError, FileNotFoundError) as exc:
        logger.warning("Could not read paper-mode risk overlay for '%s': %s", name, exc)
    return ps

def load_portfolio_state(name: str) -> PortfolioState:
    """load_portfolio_state operation.
    
    Args:
        name (str): Input parameter.
    
    Returns:
        PortfolioState: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    state_file = f"data/portfolio_state_{name}.json"
    risk_file = f"data/portfolio_risk_{name}.json"
    # FIX-NEW-DW-03: backups list covers the primary file plus all BACKUP_GENERATIONS
    # rotation slots (bak.0 … bak.{BACKUP_GENERATIONS-1}).  All slots are tried in
    # order; only files that exist AND parse successfully are returned.  Files that
    # exist but fail JSON parsing are recorded in corrupted_paths — a RuntimeError
    # is raised only when every *existing* file is corrupted, not when some slots
    # are simply absent (normal on a fresh installation).
    backups    = [state_file] + [f"{state_file}.bak.{i}" for i in range(BACKUP_GENERATIONS)]
    corrupted_paths = []
    found_any_state_file = False
    found_risk_overlay = os.path.exists(risk_file)

    for path in backups:
        if os.path.exists(path):
            found_any_state_file = True
            try:
                with open(path, "r", encoding="utf-8") as f:
                    ps = PortfolioState.from_dict(json.load(f))
                if PAPER_MODE:
                    return _apply_risk_overlay(ps, name)
                return ps
            except Exception as exc:
                logger.warning("Corrupted state at %s: %s", path, exc)
                corrupted_paths.append(path)

    if corrupted_paths:
        # PROD-FIX-1: Do NOT raise here. Refusing to start when all state files
        # are corrupted (e.g. crash mid-write during a CVaR breach) forces manual
        # intervention before the system can run again. Instead, log at CRITICAL
        # and return a safe zero-position, full-cash state so the operator can
        # inspect while the system stays live. The next save_portfolio_state call
        # will overwrite the corrupted files with the fresh default state.
        logger.critical(
            "PORTFOLIO STATE RECOVERY FAILED for '%s': all %d discovered state "
            "file(s) are corrupted. Starting from a ZERO-POSITION, FULL-CASH "
            "state. Verify the live position manually before any rebalance "
            "executes. Corrupted files: %s",
            name, len(corrupted_paths), corrupted_paths,
        )
        return PortfolioState()

    if not found_any_state_file:
        if PAPER_MODE and found_risk_overlay:
            logger.info(
                "Primary state files missing for '%s', but risk overlay exists. "
                "Recovering risk metadata into default PortfolioState().",
                name,
            )
            return _apply_risk_overlay(PortfolioState(), name)
        logger.info(
            "No portfolio state files found for '%s'. Starting clean first-run state.",
            name,
        )

    return PortfolioState()

# ─── Core scan logic ──────────────────────────────────────────────────────────
def _scan_phase_download_data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """_scan_phase_download_data operation.

    Args:
        ctx (Dict[str, Any]): Input parameter.

    Returns:
        Dict[str, Any]: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    # TODO: Phase 1 should resolve session date range, assemble symbol universe,
    # and populate ctx["market_data"] via load_or_fetch.
    return ctx


def _scan_phase_regime_prep(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """_scan_phase_regime_prep operation.

    Args:
        ctx (Dict[str, Any]): Input parameter.

    Returns:
        Dict[str, Any]: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    # TODO: Phase 2 should detect/apply splits, build close matrix, forward-fill,
    # and compute regime inputs (idx slice, close_hist, log returns).
    return ctx


def _scan_phase_exposure_cvar(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """_scan_phase_exposure_cvar operation.

    Args:
        ctx (Dict[str, Any]): Input parameter.

    Returns:
        Dict[str, Any]: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    # TODO: Phase 3 should update exposure multiplier and evaluate book CVaR
    # hard/soft breach flags used by optimization and decay phases.
    return ctx


def _scan_phase_optimization(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """_scan_phase_optimization operation.

    Args:
        ctx (Dict[str, Any]): Input parameter.

    Returns:
        Dict[str, Any]: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    # TODO: Phase 4 should generate signals, sector labels, and optimizer
    # targets; record solver/data failure outcomes in ctx.
    return ctx


def _scan_phase_decay_targeting(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """_scan_phase_decay_targeting operation.

    Args:
        ctx (Dict[str, Any]): Input parameter.

    Returns:
        Dict[str, Any]: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    # TODO: Phase 5 should compute decay targets when optimization fails and
    # enforce full-liquidation behavior when limits are exhausted.
    return ctx


def _scan_phase_stale_price_gate(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """_scan_phase_stale_price_gate operation.

    Args:
        ctx (Dict[str, Any]): Input parameter.

    Returns:
        Dict[str, Any]: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    # TODO: Phase 6 should run trading-calendar stale-bar checks and lock/adjust
    # weights for stale held symbols before execution.
    return ctx


def _scan_phase_execution(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """_scan_phase_execution operation.

    Args:
        ctx (Dict[str, Any]): Input parameter.

    Returns:
        Dict[str, Any]: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    # TODO: Phase 7 should perform pending-sentinel claim/write and call
    # execute_rebalance with finalized targets and scenario losses.
    return ctx


def _scan_phase_eod_accounting(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """_scan_phase_eod_accounting operation.

    Args:
        ctx (Dict[str, Any]): Input parameter.

    Returns:
        Dict[str, Any]: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    # TODO: Phase 8 should record EOD prices/equity, update absent tracking,
    # emit summary logs, and render trade action sheets.
    return ctx

def _run_scan(
    universe: List[str],
    state:    PortfolioState,
    label:    str,
    cfg_override: Optional[UltimateConfig] = None,
    circuit_breaker: CircuitBreaker = _circuit_breaker,
    name: str = "scan",
) -> tuple:
    """_run_scan operation.
    
    Args:
        universe (List[str]): Input parameter.
        state (PortfolioState): Input parameter.
        label (str): Input parameter.
        cfg_override (Optional[UltimateConfig]): Input parameter.
        circuit_breaker (CircuitBreaker): Input parameter.
        name (str): Input parameter.
    
    Returns:
        tuple: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    scan_started_at = time.perf_counter()
    _print_stage_status("Download", 0.05, f"Preparing {len(universe):,} symbols for {label}...")

    # PROD-FIX-5: wrap the scan in a ScanContext so every log record emitted
    # during this run carries a correlation_id field for log aggregator queries.
    #
    # FIX-SCAN-CTX-SAFETY: use the context manager protocol directly (with statement)
    # instead of manual __enter__/__exit__ calls.  The previous manual approach had
    # no try/finally guard, so any exception that escaped the scan body left the
    # thread-local correlation_id set for the next scan and silently dropped all
    # accumulated dead-letter symbols.  The `with` statement guarantees __exit__
    # is called on every exit path, including exceptions.
    _dead_letter = DeadLetterTracker(threshold=10)

    def _scan_body(cfg: UltimateConfig) -> tuple:
        """_scan_body operation.
        
        Args:
            cfg (UltimateConfig): Input parameter.
        
        Returns:
            tuple: Result of this operation.
        
        Raises:
            Exception: Propagates runtime, validation, I/O, or provider errors.
        """
        phase_ctx: Dict[str, Any] = {}
        _scan_phase_download_data(phase_ctx)
        _scan_phase_regime_prep(phase_ctx)
        _scan_phase_exposure_cvar(phase_ctx)
        _scan_phase_optimization(phase_ctx)
        _scan_phase_decay_targeting(phase_ctx)
        _scan_phase_stale_price_gate(phase_ctx)
        _scan_phase_execution(phase_ctx)
        _scan_phase_eod_accounting(phase_ctx)

        def _build_close_series(universe_symbols: list[str], mkt_data: dict, use_adjusted: bool) -> Dict[str, pd.Series]:
            """Build a map of bare symbols to close-price series.

            Args:
                universe_symbols (list[str]): Symbols to include in the output map.
                mkt_data (dict): Downloaded market data keyed by NSE ticker.
                use_adjusted (bool): Whether to prefer ``Adj Close`` when available.

            Returns:
                Dict[str, pd.Series]: Per-symbol close series keyed by bare symbol.

            Raises:
                Exception: Propagates unexpected runtime or data access failures.
            """
            close_map: Dict[str, pd.Series] = {}
            for sym in universe_symbols:
                ns = to_ns(sym)
                if ns not in mkt_data:
                    continue
                col = "Adj Close" if use_adjusted and "Adj Close" in mkt_data[ns].columns else "Close"
                close_map[to_bare(ns)] = mkt_data[ns][col]
            return close_map

        engine = InstitutionalRiskEngine(cfg)
        # FIX-MB2-EQUITYCAP: apply cfg defaults only when these fields were absent
        # in persisted state (represented as None by PortfolioState.from_dict).
        # Explicit persisted values (including 500/12) must be preserved.
        if state.equity_hist_cap is None:
            state.equity_hist_cap = cfg.EQUITY_HIST_CAP
        if state.max_absent_periods is None:
            state.max_absent_periods = cfg.MAX_ABSENT_PERIODS

        # Derive session_date from market timezone to avoid date flip mid-session
        session_date = pd.Timestamp.now(tz=TIMEZONE_IST).normalize().tz_localize(None)
        today = session_date  # Keep today for backward compatibility with other uses
        next_due = _next_rebalance_due(state.last_rebalance_date, cfg.REBALANCE_FREQ)
        rebalance_allowed = next_due is None or today >= next_due
        if not rebalance_allowed:
            logger.info(
                "[Scan] Cadence gate active: last rebalance %s, next due %s (%s). Scan will mark-to-market only.",
                state.last_rebalance_date,
                next_due.strftime("%Y-%m-%d") if next_due else "None",
                cfg.REBALANCE_FREQ,
            )

        end_date   = session_date.strftime("%Y-%m-%d")
        start_date = (session_date - pd.Timedelta(days=400)).strftime("%Y-%m-%d")

        held_syms  = {to_ns(s) for s in state.shares.keys()}
        all_syms   = list({to_ns(t) for t in universe} | held_syms | {MARKET_INDEX_NSEI, MARKET_INDEX_CRSLDX})
        _print_stage_status(
            "Download",
            0.35,
            f"Fetching/caching OHLCV data ({start_date} → {end_date}) for {len(all_syms):,} instruments...",
        )
        market_data = load_or_fetch(all_syms, start_date, end_date, cfg=cfg)
        _print_stage_status("Download", 1.0, f"Data ready. Starting iteration and signal analysis for {label}.")
        _print_stage_status("Analysis", 0.10, "Normalizing market snapshots and benchmark regime inputs...")

        idx_df = market_data.get(MARKET_INDEX_CRSLDX)
        if idx_df is None or idx_df.empty:
            idx_df = market_data.get(MARKET_INDEX_NSEI)

        idx_slice    = idx_df.iloc[:-1] if idx_df is not None and not idx_df.empty else None

        split_syms = detect_and_apply_splits(state, market_data, cfg)
        if split_syms:
            logger.warning("[Scan] Applied split adjustments for: %s", split_syms)

        use_adj = getattr(cfg, "AUTO_ADJUST_PRICES", True)

        close_d = _build_close_series(universe, market_data, use_adj)

        if not close_d:
            # PROD-FIX-3: Circuit breaker — count consecutive empty-universe scans.
            cb_count = circuit_breaker.increment()  # ARCH-FIX-7
            circuit_breaker.save(_CIRCUIT_BREAKER_FILE)  # ARCH-FIX-7
            logger.warning(
                "[Scan] No data available for any universe symbol "
                "(consecutive empty scans: %d / halt threshold: %d).",
                cb_count, _EMPTY_UNIVERSE_HALT_AFTER,
            )
            if cb_count >= _EMPTY_UNIVERSE_HALT_AFTER:
                raise RuntimeError(
                    f"[Scan] CIRCUIT BREAKER TRIPPED: {cb_count} consecutive "
                    f"scans returned an empty universe (threshold={_EMPTY_UNIVERSE_HALT_AFTER}). "
                    "Possible data provider outage or misconfigured universe.  "
                    "Halting to prevent unintended full-cash drift.  "
                    "Set EMPTY_UNIVERSE_HALT_AFTER env var to adjust threshold."
                )
            for held_sym in list(state.shares.keys()):
                if held_sym not in close_d:
                    state.absent_periods[held_sym] = int(state.absent_periods.get(held_sym, 0)) + 1
            return state, market_data
    
        # PROD-FIX-3: Reset empty-universe circuit breaker — we have data.
        circuit_breaker.reset()  # ARCH-FIX-7
        circuit_breaker.save(_CIRCUIT_BREAKER_FILE)  # ARCH-FIX-7
    
        _print_stage_status("Analysis", 0.35, f"Built close-price matrix for {len(close_d):,} active symbols.")
    
        close    = pd.DataFrame(close_d).sort_index()
        # BUG-FIX-WEEKEND-ALIGN: fill *after* frame assembly so NaNs introduced by
        # union-index alignment (e.g. mixed Friday/Saturday last bars across symbols)
        # are forward-filled consistently for every column.
        # Bound forward fill to avoid carrying very old quotes too far forward.
        # Convert stale-day allowance to row-count (daily OHLCV cadence => 1 row/day).
        # CC-01: Now that MAX_PRICE_STALE_DAYS is structurally defined in
        # UltimateConfig, we can simplify this without fearing AttributeError.
        max_stale_days = int(getattr(cfg, "MAX_PRICE_STALE_DAYS", 2))
        limit_rows = max(1, max_stale_days)
        close    = close.ffill(axis=0, limit=limit_rows)
        close    = close.loc[close.index <= pd.Timestamp(end_date)]
        latest_row = close.iloc[-1]
        active = [
            sym for sym, px in latest_row.items()
            if np.isfinite(px) and float(px) > 0.0
        ]
        if not active:
            logger.warning(
                "[Scan] No symbols have a sufficiently recent valid close after bounded "
                "forward-fill (MAX_PRICE_STALE_DAYS=%d).",
                max_stale_days,
            )
            for held_sym in list(state.shares.keys()):
                state.absent_periods[held_sym] = int(state.absent_periods.get(held_sym, 0)) + 1
            return state, market_data
        close = close.loc[:, active]
        prices   = close.iloc[-1].values.astype(float)
        active_idx = {sym: i for i, sym in enumerate(active)}
    
        # FIX-NEW-DW-02: replace NaN or non-positive prices with last_known_prices
        # before any downstream computation (compute_book_cvar, execute_rebalance,
        # record_eod).  compute_book_cvar does not guard against NaN in its prices
        # argument — a NaN propagates into notional → pv → weights → CVaR, producing
        # an incorrect (often zero) CVaR that suppresses the risk gate.
        for _sym, _i in active_idx.items():
            if not np.isfinite(prices[_i]) or prices[_i] <= 0:
                _lkp = state.last_known_prices.get(_sym, 0.0)
                prices[_i] = float(_lkp) if np.isfinite(_lkp) and _lkp > 0 else 0.0
                # PROD-FIX-5: accumulate stale-price symbols for dead-letter report
                _dead_letter.add(
                    _sym,
                    reason="stale_price" if (np.isfinite(_lkp) and _lkp > 0) else "no_price",
                    detail=f"last_known={_lkp:.4f}" if np.isfinite(_lkp) else "no_last_known",
                )
    
        mtm_notional = 0.0
        for sym in state.shares:
            if sym in active_idx:
                px = prices[active_idx[sym]]
                if np.isnan(px) or px <= 0:
                    px = float(state.last_known_prices.get(sym, 0.0))
                mtm_notional += state.shares.get(sym, 0) * px
        for _absent_sym in state.shares:
            if _absent_sym not in active_idx:
                _fallback_px = state.last_known_prices.get(_absent_sym, 0.0)
                if _fallback_px > 0:
                    _absent_n = int(state.absent_periods.get(_absent_sym, 0))
                    _mtm_px = absent_symbol_effective_price(_fallback_px, _absent_n, cfg.MAX_ABSENT_PERIODS)
                    mtm_notional += state.shares[_absent_sym] * _mtm_px
                    logger.warning(
                        "[Scan] Held symbol '%s' absent from current market data "
                        "(possibly delisted/suspended). Absent %d/%d periods; "
                        "last-known ₹%.2f haircut to ₹%.2f for PV. "
                        "Force-close triggers at %d consecutive absent periods.",
                        _absent_sym,
                        _absent_n,
                        cfg.MAX_ABSENT_PERIODS,
                        _fallback_px,
                        _mtm_px,
                        cfg.MAX_ABSENT_PERIODS,
                    )
        pv = mtm_notional + state.cash
        initial_cash = state.cash
        initial_gross_exposure = mtm_notional / pv if pv > 0 else 1.0
    
        today_ts = pd.Timestamp(end_date)
        # DW-03: Date-based T-1 exclusion handles pre-market/holiday boundaries better
        # than positional .iloc[:-1], which could accidentally exclude T-2 if today is missing.
        close_hist = close.loc[close.index < today_ts]
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
                # FIX-MB-GATENAMES: Keys renamed to "history_failed" / "adv_failed" /
                # "knife_failed". Log message now clearly states "removed by" each gate
                # so operators can distinguish a data-provider outage (many history
                # failures) from a volatility spike (many knife-gate removals).
                logger.info(
                    "[Scan] Universe funnel: %d total → %d removed by history gate → "
                    "%d removed by ADV gate → %d removed by knife gate → %d selected.",
                    gate_counts.get("total", 0),
                    gate_counts.get("history_failed", 0),
                    gate_counts.get("adv_failed", 0),
                    gate_counts.get("knife_failed", 0),
                    gate_counts.get("selected", 0),
                )
                if not sel_idx:
                    raise OptimizationError("No valid universe candidates.", OptimizationErrorType.DATA)
                sel_syms      = [active[i] for i in sel_idx]
                sector_map    = get_sector_map(sel_syms, cfg=cfg)
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
                weights = compute_decay_targets(state, sel_idx, active, cfg, current_prices=prices, pv=pv)
    
        # FIX-MB-RESIDUALCASH: The residual-cash allocation is handled entirely inside
        # execute_rebalance via its multi-pass proportional allocation loop, which
        # operates on the actual post-sell cash balance. We must NOT attempt to size
        # additional buys here against pv_exec (computed before sells execute) because
        # after sells complete the available cash is lower than pv_exec implies, and
        # oversizing here causes state.cash to go negative after slippage deduction.
        # The execute_rebalance call below receives the target weights and handles
        # residual distribution correctly with no pre-call adjustment needed.
    
        # FIX-STALE-PRICE: Single source-of-truth stale-price gate. We validate
        # held symbols against market_data using the close-index trading calendar
        # and suppress normal rebalances when stale, while still allowing
        # _force_full_cash liquidations in stress events.
        # Use an authoritative expected session date derived from market calendar
        # (fallback: latest business day) so uniformly stale payloads cannot pass
        # just because every series shares the same stale close.index endpoint.
        # Before executing a rebalance, check whether any held
        # position has a price that is older than _STALE_PRICE_DAYS trading days.
        # A rebalance using a 3-day-old price is likely worse than no rebalance,
        # because the optimizer will size incorrectly relative to the current
        # portfolio value. Force-liquidations (CVaR breach) are exempt — it is
        # always safer to exit a position than to hold it with a stale price.
        _STALE_PRICE_DAYS = 2  # trading days
        _rebalance_stale_held: list = []
        if (rebalance_allowed or _force_full_cash) and (optimization_succeeded or apply_decay):
            valid_days = None
            expected_session_date = pd.Timestamp.now(tz=TIMEZONE_IST).normalize()
            calendar_window_start = (expected_session_date - pd.Timedelta(days=366)).replace(tzinfo=None)
            try:
                import pandas_market_calendars as mcal

                nse_calendar = mcal.get_calendar("NSE")
                valid_days = nse_calendar.valid_days(
                    start_date=calendar_window_start,
                    end_date=expected_session_date,
                )
                if len(valid_days) > 0:
                    expected_session_date = pd.Timestamp(valid_days[-1]).tz_convert(TIMEZONE_IST).normalize().replace(tzinfo=None)
                else:
                    expected_session_date = (expected_session_date - pd.offsets.BDay(1)).replace(tzinfo=None)
            except Exception:
                expected_session_date = (expected_session_date - pd.offsets.BDay(1)).tz_localize(None)

            trusted_close_index = pd.DatetimeIndex([])
            if valid_days is not None and len(valid_days) > 0:
                trusted_close_index = pd.DatetimeIndex(valid_days)
                if trusted_close_index.tz is not None:
                    trusted_close_index = trusted_close_index.tz_localize(None)
                trusted_close_index = trusted_close_index.normalize()
                trusted_close_index = trusted_close_index[trusted_close_index <= expected_session_date]

            if trusted_close_index.empty or trusted_close_index.max() < expected_session_date:
                trusted_close_index = pd.date_range(
                    start=calendar_window_start,
                    end=expected_session_date,
                    freq="B",
                )

            for _chk_sym in state.shares:
                if _chk_sym not in active_idx:
                    continue  # absent symbols handled by execute_rebalance itself
                _chk_col = to_ns(_chk_sym)
                _chk_df  = market_data.get(_chk_col) if market_data else None
                if _chk_df is not None and not _chk_df.empty:
                    _last_ts = _chk_df.index[-1]
                    # Normalise timezone before comparison
                    if hasattr(_last_ts, "tzinfo") and _last_ts.tzinfo is not None:
                        _last_ts = _last_ts.tz_convert(TIMEZONE_IST).replace(tzinfo=None)
                    _last_ts = pd.Timestamp(_last_ts)
                    # BUG-FIX-MONDAY: use the close index (actual NSE trading
                    # calendar) as the business-day ruler rather than raw calendar
                    # days.  A Friday close evaluated on Monday gives 0 bars elapsed
                    # (the weekend produced no trading bars), correctly passing the
                    # gate.  Calendar-day arithmetic (today - last_ts).days would
                    # yield 3, suppressing every Monday rebalance as a false alarm.
                    _trading_bars_elapsed = int((trusted_close_index > _last_ts).sum())
                    if _trading_bars_elapsed > _STALE_PRICE_DAYS:
                        _rebalance_stale_held.append((_chk_sym, _trading_bars_elapsed))
            if _rebalance_stale_held:
                logger.warning(
                    "[Scan] STALENESS GATE: %d held symbol(s) have prices "
                    "older than %d trading bar(s) in the NSE close index: %s. "
                    "Skipping rebalance to avoid sizing on stale data. "
                    "Will retry on next scan once provider returns fresh data.",
                    len(_rebalance_stale_held),
                    _STALE_PRICE_DAYS,
                    [(s, f"{d}d") for s, d in _rebalance_stale_held],
                )
                optimization_succeeded = False
                # BUG-DW-02: Forced liquidations (CVaR hard breach) must bypass
                # stale-price suppression to ensure the execution condition
                # (rebalance_allowed or _force_full_cash) and (optimization_succeeded or
                # apply_decay) evaluates to True. Only reset apply_decay when not in a
                # forced liquidation scenario.
                if not _force_full_cash:
                    apply_decay = False

            # [PHASE 2 FIX] Stale Price Gate Bypass: Ensure forced liquidation handling
            # holds stale/halted symbols at their current weight, bypassing sell-orders
            # so CVaR execution avoids fake realizations at stale prices.
            if _rebalance_stale_held and (_force_full_cash or apply_decay):
                for _s_bare, _ in _rebalance_stale_held:
                    if _s_bare in active_idx:
                        _s_idx = active_idx[_s_bare]
                        _s_shares = state.shares.get(_s_bare, 0)
                        _s_px = max(float(prices[_s_idx]), 1e-6)
                        _mtm_w = (_s_shares * _s_px) / max(pv, 1.0)
                        weights[_s_idx] = _mtm_w
    
        if (rebalance_allowed or _force_full_cash) and (optimization_succeeded or apply_decay):
            pending = _load_pending_sentinel(name=name)  # ARCH-FIX-3
            if pending and pending.get("date") == session_date.strftime("%Y-%m-%d"):
                logger.warning("Rebalance already committed for today, skipping")
                return state, market_data
            token = str(uuid.uuid4())
            if not _try_claim_pending_sentinel(name=name, token=token, date_str=session_date.strftime("%Y-%m-%d")):
                logger.warning("Rebalance claim already exists for today, skipping")
                return state, market_data
            _write_pending_sentinel(
                name=name,
                token=token,
                date_str=today.strftime("%Y-%m-%d"),
            )
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
    
        # FIX-MB-H-04: execute_rebalance already increments absent_periods for
        # every held symbol not found in active_idx (and clears it for symbols
        # that ARE present).  Running the same loop here after a rebalance causes
        # a double-increment, halving the effective MAX_ABSENT_PERIODS grace period.
        #
        # Rule: absent_periods accounting belongs exclusively to execute_rebalance
        # on rebalance days.  This post-scan loop only runs on mark-to-market-only
        # days (cadence gate blocked the rebalance) so that absent positions are
        # still tracked even when no rebalance fires.
        if not rebalanced_this_scan:
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
    
    with ScanContext(label=label):
        cfg = cfg_override if cfg_override else load_optimized_config()
        try:
            return _scan_body(cfg)
        finally:
            # PROD-FIX-5: flush the tracker while the scan correlation_id is still active.
            _dead_letter.flush()


# ─── Status display ───────────────────────────────────────────────────────────

def _render_holdings_table(rows: List[dict], cash: float, pv: float) -> str:
    """_render_holdings_table operation.

    Args:
        rows (List[dict]): Input parameter.
        cash (float): Input parameter.
        pv (float): Input parameter.

    Returns:
        str: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    c_pipe = f"{C.GRY}│{C.RST}"
    lines: List[str] = []
    lines.append(f"  {C.GRY}┌──────────────┬─────────┬───────────┬───────────┬────────┬─────────────┬─────────────┐{C.RST}")
    lines.append(
        f"  {c_pipe} {C.B_CYN}{'Symbol':<12}{C.RST} {c_pipe} {C.B_CYN}{'Shares':>7}{C.RST} "
        f"{c_pipe} {C.B_CYN}{'Price':>9}{C.RST} {c_pipe} {C.B_CYN}{'Entry':>9}{C.RST} "
        f"{c_pipe} {C.B_CYN}{'Weight':>6}{C.RST} {c_pipe} {C.B_CYN}{'Notional':>11}{C.RST} "
        f"{c_pipe} {C.B_CYN}{'Unreal P&L':>11}{C.RST} {c_pipe}"
    )
    lines.append(f"  {C.GRY}├──────────────┼─────────┼───────────┼───────────┼────────┼─────────────┼─────────────┤{C.RST}")

    total_pnl = 0.0
    for r in rows:
        pnl_val = float(r.get('pnl', float('nan')))
        pnl_raw = f"₹{pnl_val:+,.0f}" if np.isfinite(pnl_val) else "n/a"
        pnl_color = C.B_GRN if pnl_val > 0 else (C.B_RED if pnl_val < 0 else C.RST)
        if np.isfinite(pnl_val):
            total_pnl += pnl_val
        lines.append(
            f"  {c_pipe} {C.BLD}{r['sym']:<12}{C.RST} {c_pipe} {int(r['shares']):>7,d} "
            f"{c_pipe} {float(r['price']):>9,.2f} {c_pipe} {float(r['entry']):>9,.2f} "
            f"{c_pipe} {C.CYN}{float(r['weight']):>6.1%}{C.RST} {c_pipe} {float(r['notional']):>11,.0f} "
            f"{c_pipe} {pnl_color}{pnl_raw:>11}{C.RST} {c_pipe}"
        )

    lines.append(f"  {C.GRY}├──────────────┼─────────┼───────────┼───────────┼────────┼─────────────┼─────────────┤{C.RST}")
    cash_weight = (cash / pv) if pv > 0 else 0.0
    lines.append(
        f"  {c_pipe} {C.BLD}{'Cash':<12}{C.RST} {c_pipe} {'':>7} {c_pipe} {'':>9} {c_pipe} {'':>9} "
        f"{c_pipe} {C.CYN}{cash_weight:>6.1%}{C.RST} {c_pipe} {cash:>11,.0f} {c_pipe} {'':>11} {c_pipe}"
    )
    lines.append(f"  {C.GRY}├──────────────┼─────────┼───────────┼───────────┼────────┼─────────────┼─────────────┤{C.RST}")
    tot_color = C.B_GRN if total_pnl > 0 else (C.B_RED if total_pnl < 0 else C.RST)
    lines.append(
        f"  {c_pipe} {C.BLD}{'TOTAL':<12}{C.RST} {c_pipe} {'':>7} {c_pipe} {'':>9} {c_pipe} {'':>9} "
        f"{c_pipe} {C.BLD}{1.0:>6.1%}{C.RST} {c_pipe} {C.BLD}{pv:>11,.0f}{C.RST} "
        f"{c_pipe} {tot_color}{'₹'+f'{total_pnl:+,.0f}':>11}{C.RST} {c_pipe}"
    )
    lines.append(f"  {C.GRY}└──────────────┴─────────┴───────────┴───────────┴────────┴─────────────┴─────────────┘{C.RST}")
    return "\n".join(lines)


def _render_portfolio_diagnostics(state: PortfolioState, cfg: UltimateConfig) -> str:
    """_render_portfolio_diagnostics operation.

    Args:
        state (PortfolioState): Input parameter.
        cfg (UltimateConfig): Input parameter.

    Returns:
        str: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    cvar = state.realised_cvar(min_obs=cfg.CVAR_MIN_HISTORY)
    cvar_color = C.RED if cvar > cfg.CVAR_DAILY_LIMIT else C.GRN
    return "\n".join([
        f"\n  {C.BLD}Portfolio Diagnostics:{C.RST}",
        f"  {C.YLW}⚡{C.RST} Exposure Multiplier : {C.BLD}{state.exposure_multiplier:.3f}{C.RST}",
        f"  {C.RED}🛡️ {C.RST} Override Active     : {C.BLD}{state.override_active}{C.RST}  {C.GRY}(Cooldown: {state.override_cooldown}){C.RST}",
        f"  {C.CYN}📉{C.RST} CVaR (realised)     : {cvar_color}{cvar:.2%}{C.RST}",
        f"  {C.RED}⚠️ {C.RST} Consec. Failures    : {C.BLD}{state.consecutive_failures}{C.RST}",
        f"  {C.BLU}📊{C.RST} Equity History Pts  : {C.BLD}{len(state.equity_hist)}{C.RST}\n",
    ])


def _print_status(state: PortfolioState, label: str, market_data: dict, cfg: Optional[UltimateConfig] = None) -> None:
    """_print_status operation.

    Args:
        state (PortfolioState): Input parameter.
        label (str): Input parameter.
        market_data (dict): Input parameter.
        cfg (Optional[UltimateConfig]): Input parameter.

    Returns:
        None: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    if cfg is None:
        cfg = UltimateConfig()

    print(f"\n  {C.GRY}╭{'─' * 88}╮{C.RST}")
    print(f"  {C.GRY}│{C.BLD}  STATUS — {label}  {C.RST}{C.GRY}{' ' * (75 - len(label))}│{C.RST}")
    print(f"  {C.GRY}╰{'─' * 88}╯{C.RST}")

    if not state.shares:
        print(f"  {C.GRY}No open positions.{C.RST}\n")
        return

    active = list(state.shares.keys())
    prices_now: Dict[str, float] = {}
    for sym in active:
        ns = to_ns(sym)
        if ns in market_data and not market_data[ns].empty:
            close_series = market_data[ns]["Close"].dropna()
            if not close_series.empty:
                prices_now[sym] = float(close_series.iloc[-1])

    mtm = sum(
        state.shares[s] * (prices_now.get(s) or state.last_known_prices.get(s, 0.0))
        for s in active
    )
    pv = mtm + state.cash

    rows: List[dict] = []
    for sym in active:
        shares = state.shares[sym]
        price = prices_now.get(sym) or state.last_known_prices.get(sym, float("nan"))
        entry = state.entry_prices.get(sym, float("nan"))
        notional = shares * price if np.isfinite(price) else 0.0
        weight = notional / pv if pv > 0 else 0.0
        pnl = (price - entry) * shares if (np.isfinite(price) and np.isfinite(entry)) else float("nan")
        rows.append({
            "sym": sym,
            "shares": shares,
            "price": price,
            "entry": entry,
            "weight": weight,
            "notional": notional,
            "pnl": pnl,
        })

    rows.sort(key=lambda x: float(x.get("weight", 0.0) or 0.0), reverse=True)
    print(_render_holdings_table(rows=rows, cash=state.cash, pv=pv))
    print(_render_portfolio_diagnostics(state=state, cfg=cfg))
def _portfolio_activity_badge(state: PortfolioState) -> str:
    has_activity = bool(state.shares or state.equity_hist or abs(state.cash - DEFAULT_INITIAL_CAPITAL) >= 1.0)
    if not has_activity:
        return f"{C.GRY}Idle{C.RST}"
    positions = len(state.shares)
    return f"{C.B_GRN}Active{C.RST} {C.GRY}({positions} pos | Cash ₹{state.cash:,.0f}){C.RST}"

def _render_main_menu(states: Dict[str, PortfolioState]) -> None:
    """_render_main_menu operation.
    
    Args:
        states (Dict[str, PortfolioState]): Input parameter.
    
    Returns:
        None: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
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
    """_prompt_menu_choice operation.
    
    Args:
        prompt (str): Input parameter.
        valid (List[str]): Input parameter.
        default (Optional[str]): Input parameter.
    
    Returns:
        str: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
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
    """_prompt_survival_mode operation.
    
    Args:
        err (UniverseFetchError): Input parameter.
        universe_name (str): Input parameter.
    
    Returns:
        Optional[List[str]]: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
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
    """Copy latest scan risk-state fields from *source* into *target* in-place."""
    target.consecutive_failures = source.consecutive_failures
    target.override_cooldown    = source.override_cooldown
    target.override_active      = source.override_active
    target.decay_rounds         = source.decay_rounds
    target.absent_periods       = copy.deepcopy(source.absent_periods)


def _preview_scan_and_maybe_save(
    states: dict,
    mkt_cache: dict,
    portfolio_key: str,
    label: str,
    universe: List[str],
    cfg: UltimateConfig,
) -> None:
    """Run a preview scan, render output, and optionally persist changes.

    Args:
        states (dict): Portfolio state objects keyed by portfolio identifier.
        mkt_cache (dict): Mutable market-data cache keyed by portfolio identifier.
        portfolio_key (str): Portfolio key to preview (e.g., ``"nifty"``).
        label (str): Human-readable label for UI/log output.
        universe (List[str]): Universe symbols to scan.
        cfg (UltimateConfig): Strategy configuration for this scan.

    Returns:
        None: Mutates ``states``/``mkt_cache`` and writes portfolio files.

    Raises:
        Exception: Propagates scan/runtime failures from downstream helpers.
    """
    preview = copy.deepcopy(states[portfolio_key])
    preview, mkt = _run_scan(universe, preview, f"{label} SCAN", cfg, name=portfolio_key)
    mkt_cache[portfolio_key] = mkt
    _print_status(preview, f"PREVIEW — {label}", mkt, cfg=cfg)
    if input(f"  {C.YLW}Save these changes? (y/n): {C.RST}").strip().lower() == "y":
        states[portfolio_key] = preview
        save_portfolio_state(preview, portfolio_key)
        print(f"  {C.GRN}[+] Saved permanently.{C.RST}")
        return

    # Intentional behavior: even when trade edits are discarded, the latest
    # risk-state remains authoritative.
    _preserve_risk_metadata(source=preview, target=states[portfolio_key])
    save_portfolio_state(states[portfolio_key], portfolio_key)
    print(f"  {C.GRY}[-] Trade changes discarded; risk metadata saved.{C.RST}")


# ─── Main menu ────────────────────────────────────────────────────────────────

def _handle_nse_total_scan(states: Dict[str, PortfolioState], mkt_cache: dict) -> None:
    """_handle_nse_total_scan operation.

    Args:
        states (Dict[str, PortfolioState]): Input parameter.
        mkt_cache (dict): Input parameter.

    Returns:
        None: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    _check_and_prompt_initial_capital(states["nse_total"], LABEL_NSE_TOTAL, "nse_total")
    cfg = load_optimized_config()
    universe: Optional[List[str]]
    try:
        universe = fetch_nse_equity_universe()
    except UniverseFetchError as e:
        universe = _prompt_survival_mode(e, "NSE Total")
        if universe is None:
            return
    _preview_scan_and_maybe_save(states, mkt_cache, "nse_total", LABEL_NSE_TOTAL, universe, cfg)


def _handle_nifty500_scan(states: Dict[str, PortfolioState], mkt_cache: dict) -> None:
    """_handle_nifty500_scan operation.

    Args:
        states (Dict[str, PortfolioState]): Input parameter.
        mkt_cache (dict): Input parameter.

    Returns:
        None: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    _check_and_prompt_initial_capital(states["nifty"], LABEL_NIFTY_500, "nifty")
    cfg = load_optimized_config()
    try:
        universe = get_nifty500()
    except UniverseFetchError as e:
        universe = _prompt_survival_mode(e, "Nifty 500")
        if universe is None:
            return
    _preview_scan_and_maybe_save(states, mkt_cache, "nifty", LABEL_NIFTY_500, universe, cfg)


def _handle_custom_scan(states: Dict[str, PortfolioState], mkt_cache: dict) -> None:
    """_handle_custom_scan operation.

    Args:
        states (Dict[str, PortfolioState]): Input parameter.
        mkt_cache (dict): Input parameter.

    Returns:
        None: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    universe = _get_custom_universe()
    if not universe:
        print(f"  {C.RED}[!] No custom universe found.{C.RST}")
        print(f"  {C.GRY}Please verify the Screener.in URL or provide a local file and try again.{C.RST}")
        return

    logger.info("[Universe] Loaded %d symbols from custom screener.", len(universe))
    _check_and_prompt_initial_capital(states["custom"], LABEL_CUSTOM_SCREENER, "custom")

    custom_cfg = load_optimized_config()
    if len(universe) < 100:
        custom_cfg.MAX_POSITIONS = 8

    _preview_scan_and_maybe_save(
        states, mkt_cache, "custom", LABEL_CUSTOM_SCREENER, universe, custom_cfg
    )


def _handle_backtest(states: Dict[str, PortfolioState]) -> None:
    """_handle_backtest operation.

    Args:
        states (Dict[str, PortfolioState]): Input parameter.

    Returns:
        None: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    # states parameter reserved for future use (e.g., backtest-from-live-state).
    print(f"\n  {C.CYN}Backtest — Select Universe:{C.RST}")
    print("  [1] NSE Total  [2] Nifty 500  [3] Custom Screener")
    bt_c = _prompt_menu_choice(f"  {C.CYN}Choice [Default 2]: {C.RST}", ["1", "2", "3"], default="2")
    if not bt_c:
        return

    raw_start = input(f"  {C.CYN}Start (YYYY-MM-DD) [Default 2020-01-01]: {C.RST}")
    try:
        start = _normalise_start_date(raw_start)
    except ValueError as exc:
        print(f"  {C.RED}{exc}{C.RST}")
        return

    if bt_c == "1":
        universe_identifier = "nse_total"
    elif bt_c == "3":
        universe_identifier = "custom"
    else:
        universe_identifier = "nifty500"

    end = datetime.today().strftime("%Y-%m-%d")
    if pd.Timestamp(start) > pd.Timestamp(end):
        print(f"  {C.RED}Invalid date range: start date {start} is after end date {end}.{C.RST}")
        return
    bt_cfg = load_optimized_config()

    if universe_identifier == "custom":
        custom_syms = _get_custom_universe()
        if not custom_syms:
            print(f"  {C.RED}[!] No custom universe found. Cannot run backtest.{C.RST}")
            print(f"  {C.GRY}Please verify the Screener.in URL or provide a local file and try again.{C.RST}")
            return

        print(f"\n  {C.B_RED}⚠  SURVIVORSHIP BIAS WARNING{C.RST}")
        print(f"  {C.YLW}Custom screener backtests use today's live stock list for ALL historical{C.RST}")
        print(f"  {C.YLW}rebalance dates. Stocks that were delisted, merged, or removed from your{C.RST}")
        print(f"  {C.YLW}screener between {start} and {end} are silently excluded.{C.RST}")
        print(f"  {C.YLW}This will inflate historical returns. Use Nifty 500 or NSE Total for{C.RST}")
        print(f"  {C.YLW}unbiased backtesting.{C.RST}")
        confirm = input(f"\n  {C.CYN}Proceed anyway? (y/n): {C.RST}").strip().lower()
        if confirm != "y":
            print(f"  {C.GRY}Cancelled. Returning to main menu.{C.RST}")
            return

        historical_union = set(custom_syms)
        data = load_or_fetch([*historical_union, MARKET_INDEX_NSEI, MARKET_INDEX_CRSLDX], start, end, cfg=bt_cfg)

        try:
            print_backtest_results(
                run_backtest(data, universe_identifier, start, end, cfg=bt_cfg, universe=custom_syms)
            )
        except RuntimeError as exc:
            print(f"\n  {C.B_RED}[!] BACKTEST FAILED{C.RST}")
            print(f"  {C.RED}{exc}{C.RST}\n")
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
            return

        data = load_or_fetch([*historical_union, MARKET_INDEX_NSEI, MARKET_INDEX_CRSLDX], start, end, cfg=bt_cfg)

        try:
            print_backtest_results(run_backtest(data, universe_identifier, start, end, cfg=bt_cfg))
        except RuntimeError as exc:
            print(f"\n  {C.B_RED}[!] BACKTEST FAILED — Historical Universe Data Missing{C.RST}")
            print(f"  {C.RED}{exc}{C.RST}")
            print(f"\n  {C.CYN}Fix: run the following command to generate required snapshots:{C.RST}")
            print(f"  {C.BLD}    python historical_builder.py{C.RST}")
            print(f"  {C.GRY}Required files:{C.RST}")
            print(f"  {C.GRY}    data/historical_nifty500.parquet  (or data/historical_nse_total.parquet){C.RST}\n")


def _handle_status(states: Dict[str, PortfolioState], mkt_cache: dict, cfg: UltimateConfig) -> None:
    """_handle_status operation.

    Args:
        states (Dict[str, PortfolioState]): Input parameter.
        mkt_cache (dict): Input parameter.
        cfg (UltimateConfig): Input parameter.

    Returns:
        None: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    for name, label in [
        ("nse_total", LABEL_NSE_TOTAL),
        ("nifty", LABEL_NIFTY_500),
        ("custom", LABEL_CUSTOM_SCREENER),
    ]:
        has_activity = states[name].shares or states[name].equity_hist or abs(states[name].cash - DEFAULT_INITIAL_CAPITAL) >= 1.0
        if has_activity:
            mkt = mkt_cache.get(name) or {}
            if not mkt and states[name].shares:
                syms = list({to_ns(s) for s in states[name].shares})
                end = datetime.today().strftime("%Y-%m-%d")
                mkt = load_or_fetch(
                    syms,
                    (datetime.today() - timedelta(days=22)).strftime("%Y-%m-%d"),
                    end,
                    cfg=cfg,
                )
                mkt_cache[name] = mkt
            _print_status(states[name], label, mkt, cfg=cfg)
    if not any((states[n].shares or states[n].equity_hist or abs(states[n].cash - DEFAULT_INITIAL_CAPITAL) >= 1.0) for n in states):
        print(f"  {C.GRY}All portfolios are empty.{C.RST}")


def _handle_manage_cash(states: Dict[str, PortfolioState]) -> None:
    """_handle_manage_cash operation.

    Args:
        states (Dict[str, PortfolioState]): Input parameter.

    Returns:
        None: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    print(f"\n  {C.CYN}Manage Cash — Select Portfolio:{C.RST}")
    print("  [1] NSE Total  [2] Nifty 500  [3] Custom Screener")
    p_c = _prompt_menu_choice(f"  {C.CYN}Choice: {C.RST}", ["1", "2", "3"])
    if not p_c:
        return
    p_map = {"1": "nse_total", "2": "nifty", "3": "custom"}
    if p_c in p_map:
        name = p_map[p_c]
        state = states[name]
        print(f"\n  {C.BLD}Current Cash: {C.GRN}₹{state.cash:,.2f}{C.RST}")
        print(f"  {C.GRY}Use positive number to deposit, negative to withdraw.{C.RST}")
        try:
            amt_str = input(f"  {C.CYN}Amount (₹): {C.RST}").replace(",", "").strip()
            amt = float(amt_str)
            old_cash = state.cash
            if amt >= 0:
                actual_change = amt
            else:
                actual_change = max(amt, -old_cash)
            state.cash = old_cash + actual_change
            save_portfolio_state(state, name)
            action = "Deposited" if actual_change >= 0 else "Withdrew"
            print(f"  {C.GRN}[+] {action} \u20b9{abs(actual_change):,.2f}. New Cash: \u20b9{state.cash:,.2f}{C.RST}")
        except ValueError:
            print(f"  {C.RED}Invalid amount.{C.RST}")
    else:
        print(f"  {C.RED}Invalid choice.{C.RST}")


def _handle_clear_states(states: Dict[str, PortfolioState], mkt_cache: dict) -> Tuple[Dict[str, PortfolioState], dict]:
    """_handle_clear_states operation.

    Args:
        states (Dict[str, PortfolioState]): Input parameter.
        mkt_cache (dict): Input parameter.

    Returns:
        Tuple[Dict[str, PortfolioState], dict]: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    print(f"\n  {C.B_RED}WARNING: This will reset your holdings to zero (cash only).{C.RST}")
    print(f"  {C.GRY}Market data cache and optimal_cfg.json are NOT affected.{C.RST}")
    print(f"  {C.GRY}Use this when you want to start a fresh portfolio with new capital.{C.RST}")
    confirm = input(f"  {C.CYN}Type 'YES' to confirm: {C.RST}").strip()
    if confirm.upper() == "YES":
        for n in ["nse_total", "nifty", "custom"]:
            targets: List[str] = []
            p = f"data/portfolio_state_{n}.json"
            targets.append(p)
            for i in range(BACKUP_GENERATIONS):
                targets.append(f"{p}.bak.{i}")
            targets.append(f"data/portfolio_risk_{n}.json")
            targets.extend(str(path) for path in pathlib.Path("data").glob(f"pending_rebalance_{n}*"))
            for target in targets:
                if not os.path.exists(target):
                    continue
                removed = False
                for attempt in range(3):
                    try:
                        os.remove(target)
                        removed = True
                        break
                    except OSError:
                        if attempt < 2:
                            time.sleep(0.1)
                if not removed:
                    logger.warning(
                        "[Clear] Could not delete '%s' after 3 attempts (file may be locked). Skipping.",
                        target,
                    )
        states = {"nse_total": PortfolioState(), "nifty": PortfolioState(), "custom": PortfolioState()}
        mkt_cache = {"nse_total": {}, "nifty": {}, "custom": {}}
        print(f"  {C.GRN}[+] Portfolio holdings cleared. Cache and config untouched.{C.RST}")
    else:
        print(f"  {C.GRY}Cancelled.{C.RST}")
    return states, mkt_cache


def main_menu() -> None:
    """main_menu operation.

    Returns:
        None: Result of this operation.

    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    states = {
        "nse_total": load_portfolio_state("nse_total"),
        "nifty": load_portfolio_state("nifty"),
        "custom": load_portfolio_state("custom"),
    }
    mkt_cache: dict = {"nse_total": {}, "nifty": {}, "custom": {}}

    while True:
        _render_main_menu(states)
        choice = _prompt_menu_choice(f"\n  {C.CYN}Choice: {C.RST}", ["1", "2", "3", "4", "5", "6", "7", "q"])
        if not choice:
            continue

        if choice == "q":
            print(f"  {C.GRY}Goodbye!{C.RST}\n")
            break

        if choice == "1":
            _handle_nse_total_scan(states, mkt_cache)
        elif choice == "2":
            _handle_nifty500_scan(states, mkt_cache)
        elif choice == "3":
            _handle_custom_scan(states, mkt_cache)
        elif choice == "4":
            _handle_backtest(states)
        elif choice == "5":
            _handle_status(states, mkt_cache, load_optimized_config())
        elif choice == "6":
            _handle_manage_cash(states)
        elif choice == "7":
            states, mkt_cache = _handle_clear_states(states, mkt_cache)
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """_parse_args operation.
    
    Args:
        argv (Optional[List[str]]): Input parameter.
    
    Returns:
        argparse.Namespace: Result of this operation.
    
    Raises:
        Exception: Propagates runtime, validation, I/O, or provider errors.
    """
    parser = argparse.ArgumentParser(description="Ultimate Momentum daily workflow")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Enable paper trading mode (disables portfolio state file writes).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    from log_config import load_dotenv_safe
    from log_config import configure_logging
    from data_cache import configure_data_cache

    # Load .env first so startup-time environment-derived settings resolve from
    # the same source as other entry points.
    load_dotenv_safe()
    _EMPTY_UNIVERSE_HALT_AFTER = int(os.environ.get("EMPTY_UNIVERSE_HALT_AFTER", "3"))
    _DEFAULT_SCREENER_URL = os.environ.get(
        "SCREENER_URL",
        "https://www.screener.in/screens/3506127/hello/",
    )

    args = _parse_args()
    PAPER_MODE = bool(args.paper)

    os.makedirs("logs", exist_ok=True)

    # PROD-FIX-5: use structured JSON logging in production.
    # Pass json_stdout=False to keep human-readable output in dev.
    _use_json = os.environ.get("LOG_JSON", "1").strip().lower() not in ("0", "false", "no")
    configure_logging(
        level=logging.INFO,
        json_stdout=_use_json,
        log_file="logs/ultimate.log",
    )
    configure_data_cache()

    logger.info("Ultimate Momentum v%s started", __version__)
    if PAPER_MODE:
        logger.warning("[!] Paper mode active. State will not be saved.")
    main_menu()




