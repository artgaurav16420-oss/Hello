import re

with open("momentum_engine.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Add the global function before execute_rebalance
new_func = """
def _collect_force_close_symbols(
    state: PortfolioState,
    prices: Dict[str, float],
    cfg: UltimateConfig,
    absence_threshold: int,
) -> set[str]:
    \"\"\"Collect symbols to force-close after prolonged universe absence.\"\"\"
    to_close: set[str] = set()
    for sym in list(state.shares.keys()):
        if sym in prices:
            continue
        count = state.absent_periods.get(sym, 0) + 1
        state.absent_periods[sym] = count
        if count >= absence_threshold:
            import logging
            logging.getLogger(__name__).warning(
                "execute_rebalance: %s absent for %d consecutive periods "
                "(≥ MAX_ABSENT_PERIODS=%d) — treating as delisted; closing position.",
                sym, count, absence_threshold,
            )
            to_close.add(sym)
    return to_close

def execute_rebalance("""

text = text.replace("def execute_rebalance(", new_func)

# 2. Remove the nested _collect_force_close_symbols
start_nested = text.find("    def _collect_force_close_symbols()")
if start_nested != -1:
    end_nested = text.find("    local_prices = _refresh_prices_and_absence_marks()", start_nested)
    if end_nested != -1:
        text = text[:start_nested] + text[end_nested:]
    else:
        print("Could not find end of nested _collect_force_close_symbols")
        exit(1)
else:
    print("Could not find nested _collect_force_close_symbols")
    exit(1)

# 3. Modify the call inside execute_rebalance
old_call = """
    local_prices = _refresh_prices_and_absence_marks()
    symbols_to_force_close = _collect_force_close_symbols()

    # Phase 1: build pv_exec excluding force-close candidates (see docstring)
    force_close_set = set(symbols_to_force_close)
"""
new_call = """
    local_prices = _refresh_prices_and_absence_marks()
    price_dict = {sym: local_prices[i] for sym, i in active_idx.items()}
    symbols_to_force_close = _collect_force_close_symbols(state, price_dict, cfg, cfg.MAX_ABSENT_PERIODS)

    # Phase 1: build pv_exec excluding force-close candidates (see docstring)
    force_close_set = symbols_to_force_close
"""
if old_call[1:] not in text and old_call not in text:
    old_call = """    local_prices = _refresh_prices_and_absence_marks()
    symbols_to_force_close = _collect_force_close_symbols()

    # Phase 1: build pv_exec excluding force-close candidates (see docstring)
    force_close_set = set(symbols_to_force_close)"""
    new_call = """    local_prices = _refresh_prices_and_absence_marks()
    price_dict = {sym: local_prices[i] for sym, i in active_idx.items()}
    symbols_to_force_close = _collect_force_close_symbols(state, price_dict, cfg, cfg.MAX_ABSENT_PERIODS)

    # Phase 1: build pv_exec excluding force-close candidates (see docstring)
    force_close_set = symbols_to_force_close"""

text = text.replace(old_call, new_call)

with open("momentum_engine.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Extracted _collect_force_close_symbols")
