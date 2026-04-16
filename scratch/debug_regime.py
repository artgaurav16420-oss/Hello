import pandas as pd
import numpy as np
import osqp_preimport
from signals import compute_regime_score, UltimateConfig

dates = pd.date_range("2024-01-01", periods=80)
idx = pd.DataFrame({"Close": np.linspace(100, 105, 80)}, index=dates)
weak = pd.DataFrame({f"S{i}": np.concatenate([np.ones(65) * 100, np.ones(15) * (80 if i < 7 else 120)]) for i in range(10)}, index=dates)

score_weak = compute_regime_score(idx, UltimateConfig(), universe_close_hist=weak)
print(f"SCORE_WEAK: {score_weak}")

# Deep dive into breadth
from signals import _check_market_crash
crash_val = _check_market_crash(weak, UltimateConfig())
print(f"CRASH_VAL: {crash_val}")
