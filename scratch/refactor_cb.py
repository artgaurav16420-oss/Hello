import re

with open("optimizer.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Define BestTrialCallbackHandler
new_class = """class BestTrialCallbackHandler:
    \"\"\"Callable handler to log formatted summaries when new historically 'best' trials emerge.\"\"\"
    
    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        ranked_best = _deterministic_best_trials(study)
        if not ranked_best:
            return
        if ranked_best[0].number != trial.number:
            return
        diags = trial.user_attrs.get("slice_diags", [])
        hdr = (
            f"\\n\\033[1;33m{'─'*72}\\033[0m"
            f"\\n\\033[1;33m  NEW BEST  Trial #{trial.number}  "
            f"Aggregate={trial.values[0]:.4f} Calmar={trial.values[1]:.4f}\\033[0m"
            f"\\n\\033[1;33m{'─'*72}\\033[0m"
        )
        logger.info(hdr)
        logger.info("  Parameters:")
        for k, v in trial.params.items():
            logger.info("    %-28s %s", k, v)
        logger.info("")
        logger.info(
            "  %-6s  %-8s  %-8s  %-8s  %-8s  %-8s  %-10s  %-10s  %s",
            "Year","CAGR%","DD%","Turn","AvgPos","AvgExp","ForcedCash","Score","Flags",
        )
        logger.info(f"  {'-' * 82}")
        for d in diags:
            flags = []
            if d.get("ceiling_hit"):
                flags.append("CEIL")
            if d.get("dd_gate_hit"):
                flags.append("DD-GATE")
            if d.get("anomaly_hit"):
                flags.append("ANOM")
            logger.info(
                "  %-6s  %+7.1f%%  %6.1f%%  %6.2fx  %6.1f  %7.3f  %9.4f  %9.4f  %s",
                d["year"],
                d["cagr"], abs(d["max_dd"]), d["turnover"],
                d["avg_positions"], d["avg_exposure"],
                d.get("forced_cash_penalty", 0.0),
                d["score"], " ".join(flags) if flags else "—",
            )
        logger.info(f"  {'-' * 82}")
        if diags:
            avg_cagr = sum(d["cagr"]       for d in diags) / len(diags)
            avg_dd   = sum(abs(d["max_dd"]) for d in diags) / len(diags)
            ceil_n   = sum(1 for d in diags if d.get("ceiling_hit"))
            logger.info(
                "  %-6s  %+7.1f%%  %6.1f%%  %54s  ceiling_hits=%d/%d",
                "AVG", avg_cagr, avg_dd, "", ceil_n, len(diags),
            )
        logger.info("")

"""

target = "def _error_class_from_trial"
idx = text.find(target)
if idx != -1:
    text = text[:idx] + new_class + text[idx:]

# 2. Remove the old _best_trial_callback inside run_optimization
start_cb = text.find("    def _best_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:")
if start_cb != -1:
    end_cb = text.find("    try:", start_cb)
    if end_cb != -1:
        text = text[:start_cb] + text[end_cb:]

# 3. Update the callbacks usage
text = text.replace("callbacks         = [_error_triage_callback_factory(), _best_trial_callback],", "callbacks         = [_error_triage_callback_factory(), BestTrialCallbackHandler()],")

with open("optimizer.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Extracted _best_trial_callback")
