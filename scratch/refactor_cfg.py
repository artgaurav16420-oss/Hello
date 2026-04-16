import re

with open("momentum_engine.py", "r", encoding="utf-8") as f:
    text = f.read()

new_config = """@dataclass
class UltimateConfig:
    \"\"\"
    Global configuration schema for the Institutional Risk Engine.
    Controls sizing constraints, risk thresholds, and execution modeling.
    Kept as a flat dataclass to ensure native compatibility with YAML/JSON 
    serialization and Optuna hyperparameter tracking.
    \"\"\"

    # --- Core & Sizing Constraints ---
    INITIAL_CAPITAL:          float = 1_000_000.0  # Starting cash balance for the portfolio.
    MAX_POSITIONS:            int   = 10           # Maximum active line items allowed.
    MAX_PORTFOLIO_RISK_PCT:   float = 0.20         # Maximum aggregate CVaR allowed.
    MAX_SINGLE_NAME_WEIGHT:   float = 0.25         # Hard cap on individual stock weights.
    MAX_SECTOR_WEIGHT:        float = 1.0          # Cap on total sector exposure.

    # --- Liquidity & Execution ---
    MAX_ADV_PCT:              float = 0.05         # Maximum participation rate of Average Daily Volume.
    MIN_ADV_CRORES:           float = 100.0        # Minimum ADV limit for universe inclusion.
    IMPACT_COEFF:             float = 5e-4         # Coefficient scaling quadratic market impact.
    ROUND_TRIP_SLIPPAGE_BPS:  float = 20.0         # Estimated round-trip friction in basis points.
    REBALANCE_FREQ:           str   = "W-FRI"      # Calendar frequency for scheduled rebalances.

    # --- Risk & Drawdown Management (CVaR) ---
    CVAR_DAILY_LIMIT:            float = 0.055     # Daily expected shortfall ceiling.
    CVAR_ALPHA:                  float = 0.95      # Confidence level for CVaR computation (e.g. 95%).
    CVAR_LOOKBACK:               int   = 90        # Lookback window for covariance and risk models.
    ADV_LOOKBACK:                int   = 90        # Lookback window for average daily volume.
    CVAR_SENTINEL_MULTIPLIER:    float = 2.5       # Multiplier identifying anomalous volatility spikes.
    CVAR_MIN_HISTORY:            int   = 20        # Minimum trading days required for risk assessment.
    CVAR_HARD_BREACH_MULTIPLIER: float = 1.5       # Triggers strict deleveraging when current CVaR exceeds limit by this multiplier.

    # --- Exposure Management ---
    DELEVERAGING_LIMIT:          float = 0.20      # Max forced cash raise fraction per interval.
    MIN_EXPOSURE_FLOOR:          float = 0.05      # Minimum cash usage unless fully deleveraged.
    CAPITAL_ELASTICITY:          float = 0.15      # Sensitivity of target weight to current cash.
    DRIFT_TOLERANCE:             float = 0.02      # Tolerable weight drift before forcing trades.

    # --- Signal Generation & Optimization ---
    SIGNAL_ANNUAL_FACTOR:     int   = 252          # Trading days in a year for signal scaling.
    HISTORY_GATE:             int   = 90           # Minimum history to permit strategy inclusion.
    HALFLIFE_FAST:            int   = 21           # Fast EWMA momentum halflife.
    HALFLIFE_SLOW:            int   = 63           # Slow EWMA momentum halflife.
    SIGNAL_LAG_DAYS:          int   = 21           # Lag applied between fast/slow signals.
    RISK_AVERSION:            float = 5.0          # Target lambda parameter in objective function.
    SLACK_PENALTY:            float = 1000.0       # Penalty parameter for OSQP soft constraints.
    DIMENSIONALITY_MULTIPLIER:int   = 3            # Factor determining history requirements vs positions.

    # --- Continuity & Gate Logic ---
    Z_SCORE_CLIP:             float = 3.0          # Caps normalized signals to prevent extreme outliers.
    CONTINUITY_BONUS:         float = 0.15         # Reward factor for maintaining existing positions.
    CONTINUITY_DISPERSION_FLOOR: float = 0.1       # Minimum dispersion threshold to apply continuity.
    CONTINUITY_MAX_SCALAR:    float = 0.20         # Maximum applicable continuity bonus score.
    CONTINUITY_MAX_HOLD_WEIGHT: float = 0.10       # Maximum weight limit for applying continuity.
    CONTINUITY_ACTIVITY_WINDOW: int = 5            # Days checked for recent trading activity.
    CONTINUITY_STALE_SESSIONS: int = 10            # Inactivity periods triggering stale status.
    CONTINUITY_FLAT_RET_EPS: float = 1e-12         # Epsilon rounding to identify dead zeroes.
    CONTINUITY_MIN_ADV_NOTIONAL: float = 0.0       # Minimal required ADV for continuity to apply.
    KNIFE_WINDOW:             int   = 20           # Window for detecting falling knife anomalies.
    KNIFE_THRESHOLD:          float = -0.15        # Negative return threshold for falling knife.

    # --- Data Edge Cases & Decay ---
    MAX_ABSENT_PERIODS:       int   = 12           # Consecutive missing periods before force-closing a position.
    MAX_DECAY_ROUNDS:         int   = 3            # Allowable decay attempts before forced liquidation.
    GHOST_VOL_LOOKBACK:       int   = 20           # Lookback for ghost asset pseudo-volatility.
    GHOST_RET_DRIFT:          float = -0.02        # Artificial drift added to untradable ghost stock.
    GHOST_VOL_FALLBACK:       float = 0.04         # Baseline volatility for missing data assets.

    # --- Network & Data Fetching ---
    YF_BATCH_TIMEOUT:         float = 120.0        # Global timeout for YF ticker batch requests.
    YF_CHUNK_TIMEOUT:         float = 90.0         # Block timeout for specific YF thread chunks.
    YF_ADV_TIMEOUT:           float = 60.0         # Timeout for concurrent ADV calculations.
    SECTOR_FETCH_TIMEOUT:     float = 8.0          # Max waiting time for sector definitions.

    # --- Regime Modeling (Market Environment) ---
    REGIME_VOL_FLOOR:         float = 0.18         # Default floor assumed for market volatility.
    REGIME_VOL_MULTIPLIER:    float = 1.5          # Shock multiplier identifying high-stress regimes.
    REGIME_SIGMOID_STEEPNESS: float = 10.0         # Steepness of regime transfer function.
    REGIME_SMA_FAST_WINDOW:   int   = 50           # Short term moving average.
    REGIME_SMA_WINDOW:        int   = 200          # Long term moving average for market trend.
    REGIME_VOL_EWMA_SPAN:     int   = 20           # Time span for fast volatility decay.
    REGIME_LT_VOL_EWMA_SPAN:  int   = 1260         # Rolling baseline parameter for deep historical volatility.

    # --- Institutional Safety & Corporate Actions ---
    SIMULATE_HALTS:           bool  = False        # If True, randomly simulate trading halts.
    DIVIDEND_SWEEP:           bool  = True         # Auto-reinvest detected dividends into cash ledger.
    SPLIT_TOLERANCE:          float = 0.005        # Acceptable error margin for auto-reconstructing splits.
    AUTO_ADJUST_PRICES:       bool  = True         # Enables implicit backward price adjustment.
    EQUITY_HIST_CAP:          int   = 500          # Hard cap on number of periods maintained in historical equity curves to prevent O(N) CVaR scale-outs.
"""

start_str = "@dataclass\nclass UltimateConfig:"
start_idx = text.find(start_str)
if start_idx != -1:
    end_idx = text.find("\n\n@dataclass", start_idx + len(start_str))
    if end_idx != -1:
        text = text[:start_idx] + new_config + text[end_idx:]
    else:
        print("End pattern not found.")
else:
    print("Start pattern not found.")

with open("momentum_engine.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated UltimateConfig")
