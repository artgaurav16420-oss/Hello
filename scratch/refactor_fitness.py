import re

with open("optimizer.py", "r", encoding="utf-8") as f:
    text = f.read()

helpers = """def _calculate_penalty_multipliers(
    avg_positions: float,
    avg_exposure: float,
    avg_cvar: float,
    max_dd: float,
) -> tuple[float, float, float]:
    \"\"\"Calculate score multipliers for concentration, risk, and structural under-exposure.\"\"\"
    _pos_deficit       = max(0.0, 6.0 - avg_positions)
    concentration_mult = 1.0 + _pos_deficit * 0.30

    risk_penalty = (max_dd + (avg_cvar * 100.0 * 2.0) + 1.0) * concentration_mult

    exposure_penalty = 0.0 if avg_exposure >= 0.25 else (0.25 - avg_exposure) * 2.0
    if avg_positions < 1.0:
        exposure_penalty += 0.5

    return concentration_mult, risk_penalty, exposure_penalty


def _build_diagnostics_dict(
    cagr: float, max_dd: float, turnover: float, final_multiple: float,
    cagr_net: float, avg_cvar: float, avg_exposure: float, avg_positions: float,
    n_rebalances: int, concentration_mult: float, sortino_quality: float,
    risk_penalty: float, exposure_penalty: float, dd_penalty: float,
    forced_cash_penalty: float, raw: float, score: float,
    ceiling_hit: bool, dd_gate_hit: bool, anomaly_hit: bool,
    cagr_is_near_zero: bool, max_dd_is_near_zero: bool,
) -> dict:
    \"\"\"Construct the standardized dictionary of diagnostic metrics.\"\"\"
    return {
        "cagr":                round(cagr,    2),
        "max_dd":              round(-max_dd,  2),
        "turnover":            round(turnover, 4),
        "final_multiple":      round(final_multiple,  4),
        "cagr_net":            round(cagr_net,        2),
        "avg_cvar_pct":        round(avg_cvar * 100.0, 4),
        "avg_exposure":        round(avg_exposure,    4),
        "avg_positions":       round(avg_positions,   2),
        "n_rebalances":        n_rebalances,
        "concentration_mult":  round(concentration_mult, 4),
        "sortino_quality":     round(sortino_quality, 4),
        "risk_penalty":        round(risk_penalty,    4),
        "exposure_penalty":    round(exposure_penalty, 4),
        "dd_penalty":          round(dd_penalty,      4),
        "forced_cash_penalty": round(forced_cash_penalty, 4),
        "raw_score":           round(raw, 6) if not (cagr_is_near_zero and max_dd_is_near_zero) else 0.0,
        "score":               round(score, 6),
        "ceiling_hit":         ceiling_hit,
        "dd_gate_hit":         dd_gate_hit,
        "anomaly_hit":         anomaly_hit,
    }


def _fitness_from_metrics(
"""

text = text.replace("def _fitness_from_metrics(\n", helpers)

new_fitness = """def _fitness_from_metrics(
    metrics: dict,
    rebal_log: pd.DataFrame,
) -> tuple[float, float, dict]:
    \"\"\"
    Compute a scalar fitness score plus a diagnostics dict for logging.

    IS_DD_GATE = 40%: kept at original value. Lowering to 35% caused the 2020
    COVID fold to always return -2.0 (COVID drove 35-45% DD for any momentum
    strategy), making positive aggregate scores mathematically impossible after
    181 trials. IS_DD_GATE is not the same concept as OOS_MAX_DD_CAP:
      IS_DD_GATE      = "this fold is catastrophic noise, score it harshly"
      OOS_MAX_DD_CAP  = "acceptable drawdown in live trading"
    These thresholds serve different purposes and need not be equal.

    IS_DD_PENALTY_PCT = 12%: lowered from 15%. Quadratic penalty fires earlier,
    adding more continuous pressure on moderately-high-drawdown parameter sets
    without making the gate threshold itself too aggressive.

    forced_cash_penalty is logged only for observability compatibility and no
    longer affects the fitness score.

    Positive raw scores use log1p(raw), preserving ranking without a hard
    asymptotic score ceiling.
    Score floor  : hard -2.0
    \"\"\"
    cagr         = float(metrics.get("cagr",    0.0))
    max_dd       = abs(float(metrics.get("max_dd", 100.0)))
    turnover     = float(metrics.get("turnover", 0.0))
    sortino      = float(metrics.get("sortino",  0.0) or 0.0)
    final_equity = float(metrics.get("final", BASE_INITIAL_CAPITAL) or BASE_INITIAL_CAPITAL)
    final_multiple = final_equity / max(BASE_INITIAL_CAPITAL, 1e-9)

    _, _, turnover_drag = _compute_turnover_drag(turnover)
    cagr_net = cagr - turnover_drag

    avg_cvar, avg_exposure, avg_positions, n_rebalances = _extract_rebalance_summary(rebal_log)
    forced_cash_penalty = 0.0

    concentration_mult, risk_penalty, exposure_penalty = _calculate_penalty_multipliers(
        avg_positions, avg_exposure, avg_cvar, max_dd
    )

    import math as _math
    if sortino is None or not _math.isfinite(sortino):
        sortino_quality = 1.0
    else:
        sortino_quality = min(max(sortino / 2.5, 0.50), 1.15)

    IS_DD_GATE        = 40.0
    IS_DD_PENALTY_PCT = 12.0

    dd_penalty = 0.0

    if max_dd > IS_DD_GATE:
        raw   = -(max_dd / 5.0)
        score = max(raw, -2.0)
        ceiling_hit = False
        dd_gate_hit = True
        anomaly_hit = False
    else:
        dd_excess  = max(0.0, max_dd - IS_DD_PENALTY_PCT)
        dd_penalty = (dd_excess ** 2) / 100.0

        anomaly_hit = (
            cagr > MAX_REASONABLE_CAGR_PCT
            or final_multiple > MAX_REASONABLE_FINAL_MULTIPLE
        )

        cagr_is_near_zero = _math.isfinite(cagr) and _math.isclose(cagr, 0.0, rel_tol=1e-9, abs_tol=1e-12)
        max_dd_is_near_zero = _math.isfinite(max_dd) and _math.isclose(max_dd, 0.0, rel_tol=1e-9, abs_tol=1e-12)

        if anomaly_hit:
            raw         = -(
                max(cagr - MAX_REASONABLE_CAGR_PCT, 0.0) / 50.0
                + max(final_multiple - MAX_REASONABLE_FINAL_MULTIPLE, 0.0)
            )
            score       = max(raw, -2.0)
            ceiling_hit = False
            dd_gate_hit = False
        elif cagr_is_near_zero and max_dd_is_near_zero:
            raw         = 0.0
            score       = 0.0
            ceiling_hit = False
            dd_gate_hit = False
        else:
            raw = (
                (cagr_net / risk_penalty) * sortino_quality
                - exposure_penalty
                - dd_penalty
            )
            score = _math.copysign(_math.log1p(abs(raw)), raw)
            ceiling_hit = False
            score = max(score, -2.0)
            dd_gate_hit = False

    cagr_is_near_zero = _math.isfinite(cagr) and _math.isclose(cagr, 0.0, rel_tol=1e-9, abs_tol=1e-12)
    max_dd_is_near_zero = _math.isfinite(max_dd) and _math.isclose(max_dd, 0.0, rel_tol=1e-9, abs_tol=1e-12)

    diag = _build_diagnostics_dict(
        cagr, max_dd, turnover, final_multiple, cagr_net, avg_cvar,
        avg_exposure, avg_positions, n_rebalances, concentration_mult,
        sortino_quality, risk_penalty, exposure_penalty, dd_penalty,
        forced_cash_penalty, raw, score, ceiling_hit, dd_gate_hit, anomaly_hit,
        cagr_is_near_zero, max_dd_is_near_zero
    )
    
    calmar_score = cagr_net / max(abs(max_dd), DRAWDOWN_FLOOR)
    return score, calmar_score, diag
"""

start_idx = text.find("def _fitness_from_metrics(\n")
end_idx = text.find("def _int_bounds_with_step", start_idx)
if start_idx != -1 and end_idx != -1:
    text = text[:start_idx] + new_fitness + "\n\n" + text[end_idx:]
else:
    print("Could not find _fitness_from_metrics block")

with open("optimizer.py", "w", encoding="utf-8") as f:
    f.write(text)

print("success")
