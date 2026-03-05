# Murder Board — Hello Codebase

## Executive summary

The current codebase has solid ambition (risk controls, CVaR-aware optimization, survivorship-bias handling), but test evidence shows multiple reliability regressions in core pathways (signal validation, split handling, decay liquidation logic, and cache integration).

## High-severity findings

1. **Sector map contract was broken at import time**
   - `test_momentum.py` imports `STATIC_NSE_SECTORS` from `universe_manager`, but it did not exist there.
   - This was causing test collection to fail before runtime checks.
   - **Fix applied:** Introduced `STATIC_NSE_SECTORS` in `universe_manager.py` and wired `get_sector_map` to consume the local constant.

2. **Signal validation message mismatch indicates contract drift**
   - `generate_signals` rejects empty input, but exception text differs from test contract (`"no valid data"`).
   - Suggestion: stabilize exception API (type + message fragment) for defensive boundaries.

3. **Corporate action split path appears non-functional for integer share flooring case**
   - Fractional-cash split test expects 101 -> 50 shares plus cash remainder; current behavior leaves shares unchanged.
   - Suggestion: review split detection thresholding and symbol normalization between `A` vs `A.NS` keys.

4. **Data cache module lost a monkeypatch seam (`_download_with_timeout`)**
   - Tests expect to patch this internal function, but symbol is absent.
   - Suggestion: reintroduce stable downloader wrapper or update tests and call graph together.

5. **Decay and forced liquidation controls not executing under failure exhaustion paths**
   - Tests indicate expected forced sells/liquidation are not occurring when decay rounds/failure thresholds are reached.
   - This is a critical risk-control bug: strategy may remain exposed when it should be de-risked.

## Medium-severity findings

1. **Cross-module ownership ambiguity for static metadata**
   - Sector fallback data was previously referenced through `daily_workflow` from inside `universe_manager`, creating brittle coupling.
   - Consolidating metadata in `universe_manager` improves ownership clarity.

2. **Backtest resilience pathways need deterministic test harnesses**
   - Multiple failure-path tests fail together, suggesting behavior drift in shared control-flow utilities.
   - Recommendation: isolate and unit-test decay state machine independently from full backtest runs.

## Suggested next actions (priority order)

1. Fix decay exhaustion + forced liquidation behavior and add focused unit tests.
2. Fix split detection/execution math with symbol canonicalization tests (`bare` vs `.NS`).
3. Restore cache downloader seam or redesign test seam explicitly.
4. Normalize/lock external-facing exception messages for validation boundaries.
5. Add a short architecture note documenting ownership of static universe/sector metadata.
