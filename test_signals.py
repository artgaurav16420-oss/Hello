"""
test_signals.py — Tests for signal generation logic v11.48
=========================================================
"""

from __future__ import annotations
import osqp_preimport

import numpy as np
import pandas as pd
import pytest
import logging

from signals import generate_signals, compute_adv, compute_regime_score
from momentum_engine import UltimateConfig

# Helper functions from test_momentum.py
def _make_log_rets(n_days: int, n_syms: int, seed: int = 42) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, size=(n_days, n_syms))
    cols = [f"SYM{i:02d}" for i in range(n_syms)]
    idx  = pd.date_range("2020-01-02", periods=n_days, freq="B")
    return pd.DataFrame(data, index=idx, columns=cols)


class TestSignals:
    @staticmethod
    def test_generate_signals_deterministic():
        """Same inputs must produce identical outputs on repeated calls."""
        log_rets = _make_log_rets(120, 6)
        adv      = np.ones(6) * 1e6
        cfg      = UltimateConfig(HISTORY_GATE=90, MAX_POSITIONS=5)
        slice_t1 = log_rets.iloc[:100]
        raw1, scores1, sel1, _ = generate_signals(slice_t1, adv, cfg)
        raw2, scores2, sel2, _ = generate_signals(slice_t1, adv, cfg)
        np.testing.assert_array_equal(raw1, raw2)
        assert sel1 == sel2

    @staticmethod
    def test_generate_signals_history_gate():
        """Assets with insufficient history must be excluded from selection."""
        log_rets = _make_log_rets(100, 5)
        log_rets.iloc[:90, 0] = np.nan   # SYM00 has only 10 valid rows
        adv      = np.ones(5) * 1e6
        cfg      = UltimateConfig(HISTORY_GATE=95, MAX_POSITIONS=5)
        _, _, sel_idx, _ = generate_signals(log_rets, adv, cfg)
        assert 0 not in sel_idx, "SYM00 should be excluded by history gate."

    @staticmethod
    def test_generate_signals_continuity_bonus():
        """An asset with a non-zero previous weight must score higher than an identical twin."""
        rng      = np.random.default_rng(0)
        base_col = rng.normal(0, 0.01, 100)
        log_rets = pd.DataFrame(
            np.column_stack([base_col, base_col]), columns=["SYM0", "SYM1"]
        )
        adv          = np.ones(2) * 1e6
        cfg          = UltimateConfig(HISTORY_GATE=10, MAX_POSITIONS=2)
        prev_weights = {"SYM0": 0.10, "SYM1": 0.0}
        _, scores, _, _ = generate_signals(
            log_rets, adv, cfg, prev_weights=prev_weights,
        )
        assert scores[0] > scores[1], "Held asset must receive continuity bonus."

    @staticmethod
    def test_generate_signals_continuity_decay_scales_with_prev_weight():
        """Continuity bonus should scale from 25% to 100% with previous holding size."""
        base_col = np.linspace(-0.01, 0.01, 120)
        log_rets = pd.DataFrame(
            np.column_stack([base_col, base_col, base_col]), columns=["SMALL", "LARGE", "NONE"]
        )
        adv = np.ones(3) * 1e6
        cfg = UltimateConfig(HISTORY_GATE=10, MAX_POSITIONS=3)
    
        _, scores, _, _ = generate_signals(
            log_rets,
            adv,
            cfg,
            prev_weights={"SMALL": 0.01, "LARGE": 0.10, "NONE": 0.0},
        )
    
        small_bonus = float(scores[0] - scores[2])
        large_bonus = float(scores[1] - scores[2])
    
        raw_scores, _, _, _ = generate_signals(log_rets, adv, cfg)
        finite = raw_scores[np.isfinite(raw_scores)]
        std_cross = max(float(np.nanstd(finite)) if finite.size else 0.0, 1e-8)
        dispersion_scale = min(1.0, std_cross / max(cfg.CONTINUITY_DISPERSION_FLOOR, 1e-12))
        base_bonus = min(cfg.CONTINUITY_BONUS, cfg.CONTINUITY_MAX_SCALAR) * dispersion_scale
    
        assert small_bonus == pytest.approx(base_bonus * 0.25, abs=1e-9)
        assert large_bonus == pytest.approx(base_bonus * 1.0, abs=1e-9)

    @staticmethod
    def test_generate_signals_continuity_bonus_blocked_for_flatlined_symbol():
        """Flatlined names must not receive continuity bonus even with prior weight."""
        n_days = 120
        live = np.linspace(-0.01, 0.01, n_days)
        flat = np.zeros(n_days)
        log_rets = pd.DataFrame({"LIVE": live, "FLAT": flat})
        adv = np.array([1e6, 1e6], dtype=float)
        cfg = UltimateConfig(HISTORY_GATE=10, MAX_POSITIONS=2, CONTINUITY_STALE_SESSIONS=10)
    
        _, scores_with_hold, _, _ = generate_signals(
            log_rets,
            adv,
            cfg,
            prev_weights={"FLAT": 0.10, "LIVE": 0.0},
        )
        _, scores_no_hold, _, _ = generate_signals(log_rets, adv, cfg)
    
        assert scores_with_hold[1] == pytest.approx(scores_no_hold[1], abs=1e-12)
        assert scores_with_hold[0] > scores_with_hold[1], "Flatlined held name must not outrank active name via continuity."

    @staticmethod
    def test_generate_signals_continuity_denial_logging_counts(caplog):
        """Continuity denial logging should report stale and liquidity counter totals."""
        n_days = 60
        live = np.linspace(-0.01, 0.01, n_days)
        stale = np.zeros(n_days)
        illiquid = np.concatenate([np.zeros(n_days - 5), np.array([0.01, -0.01, 0.01, -0.01, 0.01])])
        log_rets = pd.DataFrame({"LIVE": live, "STALE": stale, "ILLIQ": illiquid})
        adv = np.array([1e6, 1e6, 1e3], dtype=float)
        cfg = UltimateConfig(
            HISTORY_GATE=10,
            MAX_POSITIONS=3,
            CONTINUITY_STALE_SESSIONS=10,
            CONTINUITY_MIN_ADV_NOTIONAL=1e5,
            SIGNAL_LAG_DAYS=0,
        )
    
        with caplog.at_level("DEBUG"):
            generate_signals(
                log_rets,
                adv,
                cfg,
                prev_weights={"STALE": 0.10, "ILLIQ": 0.10, "LIVE": 0.0},
            )
    
        assert "Continuity denied for 1 stale and 1 illiquid symbols." in caplog.text

    @staticmethod
    def test_generate_signals_blocks_empty_input():
        """A completely empty array should trip the defensive barrier before math crash."""
        cfg = UltimateConfig()
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="no valid data"):
            generate_signals(empty_df, np.array([]), cfg)

    @staticmethod
    def test_regime_score_neutral_on_thin_history():
        idx = pd.DataFrame({"Close": [100.0] * 100}, index=pd.date_range("2020-01-01", periods=100))
        assert compute_regime_score(idx) == 0.6

    @staticmethod
    def test_regime_score_neutral_below_vol_lookback_requirement():
        idx = pd.DataFrame(
            {"Close": np.linspace(100.0, 120.0, 251)},
            index=pd.date_range("2020-01-01", periods=251),
        )
        assert compute_regime_score(idx) > 0.5

    @staticmethod
    def test_compute_regime_score_short_history_expanding_mean_path():
        idx = pd.DataFrame({"Close": np.linspace(100.0, 103.0, 30)}, index=pd.date_range("2024-01-01", periods=30))
        universe = pd.DataFrame({"A": np.linspace(10.0, 11.0, 30)}, index=idx.index)
        score = compute_regime_score(idx, UltimateConfig(REGIME_SMA_WINDOW=50), universe_close_hist=universe)
        assert 0.0 <= score <= 1.0

    @staticmethod
    def test_compute_regime_score_vol_spike_penalty_path():
        closes = np.concatenate([np.linspace(100, 110, 260), np.array([90, 120, 85, 125, 80])])
        idx = pd.DataFrame({"Close": closes}, index=pd.date_range("2023-01-01", periods=len(closes)))
        score = compute_regime_score(idx, UltimateConfig())
        assert score < 0.8

    @staticmethod
    def test_compute_regime_score_crash_and_early_warning_paths():
        dates = pd.date_range("2024-01-01", periods=80)
        idx = pd.DataFrame({"Close": np.linspace(100, 105, 80)}, index=dates)
        weak = pd.DataFrame({f"S{i}": np.concatenate([np.ones(65) * 100, np.ones(15) * (80 if i < 7 else 120)]) for i in range(10)}, index=dates)
        crash = pd.DataFrame({f"S{i}": np.concatenate([np.ones(65) * 100, np.ones(15) * (70 if i < 8 else 120)]) for i in range(10)}, index=dates)
        score_weak = compute_regime_score(idx, UltimateConfig(), universe_close_hist=weak)
        score_crash = compute_regime_score(idx, UltimateConfig(), universe_close_hist=crash)
        assert score_weak <= 0.5
        assert score_crash == 0.0

    @staticmethod
    def test_regime_score_uses_configurable_sigmoid_steepness():
        closes = np.concatenate([np.linspace(100, 102, 200), np.linspace(102, 103, 60)])
        idx = pd.DataFrame({"Close": closes}, index=pd.date_range("2021-01-01", periods=len(closes), freq="B"))
        low = compute_regime_score(idx, cfg=UltimateConfig(REGIME_SIGMOID_STEEPNESS=5.0))
        high = compute_regime_score(idx, cfg=UltimateConfig(REGIME_SIGMOID_STEEPNESS=40.0))
        assert high > low

    @staticmethod
    def test_regime_score_bull_market():
        """Price well above SMA200 should give score > 0.5."""
        closes = np.linspace(80, 120, 400)      # steady uptrend
        idx = pd.DataFrame({"Close": closes}, index=pd.date_range("2020-01-01", periods=400))
        assert compute_regime_score(idx) > 0.5

    @staticmethod
    def test_regime_score_bear_market():
        """Price well below SMA200 should give score < 0.5."""
        closes = np.linspace(120, 60, 400)      # steady downtrend
        idx = pd.DataFrame({"Close": closes}, index=pd.date_range("2020-01-01", periods=400))
        assert compute_regime_score(idx) < 0.5

    @staticmethod
    def test_regime_breadth_requires_sufficient_history_per_symbol():
        idx = pd.DataFrame(
            {"Close": np.linspace(100.0, 120.0, 300)},
            index=pd.date_range("2020-01-01", periods=300),
        )
    
        universe_close = pd.DataFrame(
            {
                "OLD": np.linspace(100.0, 120.0, 300),
                # Recent IPO-like series: only 20 observations inside a 200-day window.
                "IPO": [np.nan] * 280 + list(np.linspace(100.0, 150.0, 20)),
            },
            index=idx.index,
        )
    
        score_with_ipo = compute_regime_score(idx, universe_close_hist=universe_close)
        score_without_ipo = compute_regime_score(idx, universe_close_hist=universe_close[["OLD"]])
    
        assert score_with_ipo == pytest.approx(score_without_ipo, abs=1e-12)

    @staticmethod
    def test_compute_adv_respects_configurable_lookback():
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        market_data = {
            "ABC.NS": pd.DataFrame(
                {"Close": [100.0] * 5, "Volume": [1, 2, 3, 4, 5]},
                index=idx,
            )
        }
    
        adv_short = compute_adv(market_data, ["ABC"], cfg=UltimateConfig(ADV_LOOKBACK=2))
        adv_long = compute_adv(market_data, ["ABC"], cfg=UltimateConfig(ADV_LOOKBACK=5))
    
        assert adv_short[0] == pytest.approx(450.0)
        assert adv_long[0] == pytest.approx(300.0)

    @staticmethod
    def test_compute_adv_does_not_penalize_pre_ipo_nan_history():
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        market_data = {
            "IPO.NS": pd.DataFrame(
                {
                    "Close": [np.nan, np.nan, np.nan, 100.0, 100.0],
                    "Volume": [np.nan, np.nan, np.nan, 1_000_000.0, 1_000_000.0],
                },
                index=idx,
            )
        }
    
        adv = compute_adv(market_data, ["IPO"], cfg=UltimateConfig(ADV_LOOKBACK=5))
        assert adv[0] == pytest.approx(100_000_000.0)

    @staticmethod
    def test_continuity_bonus_respects_max_scalar_cap():
        """When CONTINUITY_BONUS exceeds CONTINUITY_MAX_SCALAR the cap clamps the bonus."""
        base_col = np.linspace(-0.01, 0.01, 120)
        log_rets = pd.DataFrame(
            np.column_stack([base_col, base_col, base_col]), columns=["A", "B", "C"]
        )
        adv = np.ones(3) * 1e6
    
        cfg_capped   = UltimateConfig(HISTORY_GATE=10, MAX_POSITIONS=3, CONTINUITY_BONUS=0.30, CONTINUITY_MAX_SCALAR=0.20)
        cfg_uncapped = UltimateConfig(HISTORY_GATE=10, MAX_POSITIONS=3, CONTINUITY_BONUS=0.20, CONTINUITY_MAX_SCALAR=0.20)
    
        _, scores_capped,   _, _ = generate_signals(log_rets, adv, cfg_capped,   prev_weights={"A": 0.10})
        _, scores_uncapped, _, _ = generate_signals(log_rets, adv, cfg_uncapped, prev_weights={"A": 0.10})
    
        assert scores_capped[0] == pytest.approx(scores_uncapped[0], abs=1e-9)
        assert scores_capped[0] > scores_capped[2]

    @staticmethod
    def test_nan_sorting_trap_no_truncation():
        rng      = np.random.default_rng(7)
        n_syms   = 20
        log_rets = pd.DataFrame(
            rng.normal(0.0, 0.01, (200, n_syms)),
            columns=[f"SYM{i:02d}" for i in range(n_syms)],
            index=pd.date_range("2020-01-02", periods=200, freq="B"),
        )
        for i in range(10):
            log_rets.iloc[:, i] = np.nan
    
        adv = np.ones(n_syms) * 1e6
        cfg = UltimateConfig(HISTORY_GATE=5, MAX_POSITIONS=10)
        _, adj_scores, sel_idx, _ = generate_signals(log_rets, adv, cfg)
    
        assert all(np.isfinite(adj_scores[i]) for i in sel_idx), \
            "Selected indices must not contain NaN-scored assets."
        assert len(sel_idx) == 10, f"Expected 10 valid selections, got {len(sel_idx)} — NaN trap still active."

    @staticmethod
    def test_compute_adv_valid_data():
        """Test the compute_adv function with valid data."""
        data = pd.DataFrame({
            'High': [10, 12, 15, 13, 16],
            'Low': [8, 9, 11, 10, 12],
            'Close': [9, 11, 14, 12, 15],
            'Volume': [100, 150, 200, 120, 180]
        })
        market_data_dict = {"TEST_SYM.NS": data}
        active_symbols = ["TEST_SYM"]
        cfg = UltimateConfig(ADV_LOOKBACK=3)
        adv = compute_adv(market_data_dict, active_symbols, cfg=cfg)
        assert isinstance(adv, np.ndarray)
        assert adv.shape == (1,)
        assert np.isfinite(adv[0])
        assert adv[0] > 0


    @staticmethod
    def test_compute_adv_empty_input():
        """Test compute_adv with an empty DataFrame."""
        empty_data = pd.DataFrame(columns=['Close', 'Volume'])
        market_data_dict = {"TEST_SYM.NS": empty_data}
        active_symbols = ["TEST_SYM"]
        cfg = UltimateConfig(ADV_LOOKBACK=3)
        adv = compute_adv(market_data_dict, active_symbols, cfg=cfg)
        assert adv.shape == (1,)
        assert adv[0] == 0.0

    @staticmethod
    def test_compute_adv_missing_columns_returns_zero():
        """Test compute_adv returns 0.0 for symbols with missing required columns."""
        invalid_data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        market_data_dict = {"TEST_SYM.NS": invalid_data}
        active_symbols = ["TEST_SYM"]
        cfg = UltimateConfig(ADV_LOOKBACK=3)
        adv = compute_adv(market_data_dict, active_symbols, cfg=cfg)
        assert adv[0] == 0.0

    @staticmethod
    def test_compute_adv_incorrect_data_type_raises_attribute_error():
        """Test compute_adv raises AttributeError for non-DataFrame input."""
        market_data_dict = {"TEST_SYM.NS": [1, 2, 3]} # This is not a DataFrame
        active_symbols = ["TEST_SYM"]
        cfg = UltimateConfig(ADV_LOOKBACK=3)
        with pytest.raises(AttributeError): 
            compute_adv(market_data_dict, active_symbols, cfg=cfg)

    @staticmethod
    def test_compute_adv_with_nan_in_data():
        """Test compute_adv with NaN values in data."""
        data_with_nan = pd.DataFrame({
            'Close': [9, 11, 14, 12, 15],
            'Volume': [100, np.nan, 200, 120, 180]
        })
        market_data_dict = {"TEST_SYM.NS": data_with_nan}
        active_symbols = ["TEST_SYM"]
        cfg = UltimateConfig(ADV_LOOKBACK=3)
        adv = compute_adv(market_data_dict, active_symbols, cfg=cfg)
        assert adv.shape == (1,)
        assert np.isfinite(adv[0])
        assert adv[0] == pytest.approx((14*200 + 12*120 + 15*180) / 3)

    @staticmethod
    def test_compute_adv_zero_volume():
        """Test compute_adv with zero volume."""
        data_zero_vol = pd.DataFrame({
            'Close': [9, 11, 14, 12, 15],
            'Volume': [0, 0, 0, 0, 0]
        })
        market_data_dict = {"TEST_SYM": data_zero_vol}
        active_symbols = ["TEST_SYM"]
        cfg = UltimateConfig(ADV_LOOKBACK=3)
        adv = compute_adv(market_data_dict, active_symbols, cfg=cfg)
        assert adv.shape == (1,)
        assert adv[0] == 0.0

class TestComputeADV:
    @staticmethod
    def test_compute_adv_basic():
        """Basic ADV calculation should be correct."""
        idx = pd.date_range("2024-01-01", periods=20, freq="B")
        market_data = {
            "ABC.NS": pd.DataFrame(
                {"Close": [100.0] * 20, "Volume": [1e6] * 20},
                index=idx,
            )
        }
        adv = compute_adv(market_data, ["ABC"], cfg=UltimateConfig(ADV_LOOKBACK=20))
        assert adv[0] == pytest.approx(100.0 * 1e6)

    @staticmethod
    def test_compute_adv_missing_symbol():
        """A symbol not in market_data should have an ADV of 0."""
        idx = pd.date_range("2024-01-01", periods=20, freq="B")
        market_data = {
            "ABC.NS": pd.DataFrame(
                {"Close": [100.0] * 20, "Volume": [1e6] * 20},
                index=idx,
            )
        }
        adv = compute_adv(market_data, ["ABC", "XYZ"], cfg=UltimateConfig(ADV_LOOKBACK=20))
        assert adv[0] == pytest.approx(100.0 * 1e6)
        assert adv[1] == 0.0

    @staticmethod
    def test_compute_adv_with_nans():
        """NaNs in data should be handled gracefully."""
        idx = pd.date_range("2024-01-01", periods=20, freq="B")
        volume = [1e6] * 20
        volume[5] = np.nan
        market_data = {
            "ABC.NS": pd.DataFrame(
                {"Close": [100.0] * 20, "Volume": volume},
                index=idx,
            )
        }
        adv = compute_adv(market_data, ["ABC"], cfg=UltimateConfig(ADV_LOOKBACK=20))
        # The mean should be calculated over the 19 valid data points
        expected_adv = 100.0 * 1e6
        assert adv[0] == pytest.approx(expected_adv)

    @staticmethod
    def test_compute_adv_target_date_slicing():
        """`target_date` should correctly slice the history."""
        idx = pd.date_range("2024-01-01", periods=20, freq="B")
        market_data = {
            "ABC.NS": pd.DataFrame(
                {"Close": [100.0] * 20, "Volume": list(range(1, 21))},
                index=idx,
            )
        }
        target_date = idx[9].strftime("%Y-%m-%d") # 10th day
        adv = compute_adv(market_data, ["ABC"], cfg=UltimateConfig(ADV_LOOKBACK=10), target_date=target_date)
        # Mean of volumes from 1 to 10
        expected_volume_mean = np.mean(list(range(1, 11)))
        assert adv[0] == pytest.approx(100.0 * expected_volume_mean)

    @staticmethod
    def test_compute_adv_empty_active_symbols():
        """Should return an empty array for no symbols."""
        market_data = {"ABC.NS": pd.DataFrame()}
        adv = compute_adv(market_data, [], cfg=UltimateConfig())
        assert adv.shape == (0,)

    @staticmethod
    def test_compute_adv_empty_market_data():
        """Should return zeros for all symbols if market_data is empty."""
        adv = compute_adv({}, ["ABC", "XYZ"], cfg=UltimateConfig())
        assert np.all(adv == 0.0)

    @staticmethod
    def test_compute_adv_insufficient_periods():
        """Should return 0 if valid periods are less than min_periods."""
        idx = pd.date_range("2024-01-01", periods=20, freq="B")
        market_data = {
            "ABC.NS": pd.DataFrame(
                {"Close": [100.0] * 5 + [np.nan] * 15, "Volume": [1e6] * 5 + [np.nan] * 15},
                index=idx,
            )
        }
        # ADV_LOOKBACK is 20, min_periods will be 10
        adv = compute_adv(market_data, ["ABC"], cfg=UltimateConfig(ADV_LOOKBACK=20))
        assert adv[0] == 0.0


class TestSignalsCoverage:
    """Extra tests to fill coverage gaps in signals.py."""

    @staticmethod
    def test_compute_regime_score_empty_inputs():
        """Handles None or empty index history."""
        assert compute_regime_score(None) == 0.5
        assert compute_regime_score(pd.DataFrame()) == 0.5
        assert compute_regime_score(pd.DataFrame({"Close": []})) == 0.5

    @staticmethod
    def test_compute_regime_score_last_is_today():
        """Verifies that the last bar is excluded if it matches today's date."""
        today = pd.Timestamp("2024-01-11")
        dates = pd.date_range(today - pd.Timedelta(days=10), periods=11, freq="D")
        idx = pd.DataFrame({"Close": np.linspace(100, 110, 11)}, index=dates)
        
        # With today included
        score_today = compute_regime_score(idx, as_of_date=today)
        
        # Without today included (manually sliced)
        idx_no_today = idx.iloc[:-1]
        score_no_today = compute_regime_score(idx_no_today, as_of_date=today)
        
        assert score_today == score_no_today

    @staticmethod
    def test_compute_regime_score_last_is_today_trims_universe_history_with_tz():
        """Verifies universe_close_hist is date-trimmed safely for tz-aware indices."""
        today = pd.Timestamp("2024-01-11")
        idx_dates = pd.date_range(today - pd.Timedelta(days=10), periods=11, freq="D")
        idx = pd.DataFrame({"Close": np.linspace(100, 110, 11)}, index=idx_dates)

        tz_dates = pd.date_range("2024-01-01 00:00:00+05:30", periods=11, freq="D")
        universe_with_today = pd.DataFrame({"A": np.linspace(50, 60, 11)}, index=tz_dates)
        universe_without_today = universe_with_today.iloc[:-1]

        score_with_today = compute_regime_score(
            idx,
            universe_close_hist=universe_with_today,
            as_of_date=today,
        )
        score_without_today = compute_regime_score(
            idx,
            universe_close_hist=universe_without_today,
            as_of_date=today,
        )

        assert score_with_today == score_without_today

    @staticmethod
    def test_compute_regime_score_benchmark_only_universe(caplog):
        """Breadth component should default to 0.5 if only benchmarks are provided."""
        idx = pd.DataFrame({"Close": [100.0] * 250}, index=pd.date_range("2020-01-01", periods=250))
        benchmarks = pd.DataFrame({"^NSEI": [18000.0] * 250}, index=idx.index)
        
        with caplog.at_level("DEBUG"):
            score = compute_regime_score(idx, universe_close_hist=benchmarks)
        
        assert score > 0
        assert "universe_close_hist contains only benchmark columns" in caplog.text

    @staticmethod
    def test_compute_regime_score_breadth_no_valid_symbols(caplog):
        """Breadth defaults to 0.5 if no symbols pass the validity filter."""
        idx = pd.DataFrame({"Close": [100.0] * 250}, index=pd.date_range("2020-01-01", periods=250))
        # All NaNs in universe history
        universe = pd.DataFrame({"A": [np.nan] * 250}, index=idx.index)
        
        with caplog.at_level("DEBUG"):
            score = compute_regime_score(idx, universe_close_hist=universe)
        
        assert "no symbols passed validity filter" in caplog.text

    @staticmethod
    def test_compute_single_adv_error_handling():
        """Verifies compute_single_adv catches specific errors."""
        from signals import compute_single_adv
        # Missing columns
        assert compute_single_adv(pd.DataFrame({"A": [1]})) == 0.0
        # Empty
        assert compute_single_adv(pd.DataFrame()) == 0.0
        # AttributeError (passing a list instead of DataFrame)
        assert compute_single_adv([1, 2, 3]) == 0.0

    @staticmethod
    def test_generate_signals_lag_truncation_errors():
        """Verifies lag truncation error handling."""
        log_rets = pd.DataFrame({"A": [0.01] * 5}, index=pd.date_range("2020-01-01", periods=5))
        adv = np.array([1e6])
        
        # Lag equals available rows
        cfg_equal = UltimateConfig(SIGNAL_LAG_DAYS=5)
        from signals import SignalGenerationError
        with pytest.raises(SignalGenerationError, match="no valid data after lag truncation"):
            generate_signals(log_rets, adv, cfg_equal)
            
        # Lag greater than available rows
        cfg_greater = UltimateConfig(SIGNAL_LAG_DAYS=10)
        # Should log a warning and use raw log_rets if lag > len
        raw, _, _, _ = generate_signals(log_rets, adv, cfg_greater)
        assert len(raw) == 1

    @staticmethod
    def test_generate_signals_all_nan_momentum():
        """Handles cases where all momentum scores are NaN."""
        log_rets = pd.DataFrame({"A": [np.nan] * 10}, index=pd.date_range("2020-01-01", periods=10))
        adv = np.array([1e6])
        cfg = UltimateConfig(HISTORY_GATE=5)
        
        raw, adj, sel, counts = generate_signals(log_rets, adv, cfg)
        assert np.all(np.isnan(raw))
        assert adj[0] == -np.inf
        assert sel == []

    @staticmethod
    def test_generate_signals_nan_sorting():
        """Verifies that NaN adj_scores are treated as -inf."""
        # SYM0 will have NaN momentum if its tail is NaN
        log_rets = pd.DataFrame({
            "SYM0": [0.01] * 90 + [np.nan] * 10,
            "SYM1": [0.01] * 100
        }, index=pd.date_range("2020-01-01", periods=100))
        adv = np.array([1e6, 1e6])
        # Set SIGNAL_LAG_DAYS=0 so tail(5) actually sees the NaNs
        cfg = UltimateConfig(HISTORY_GATE=5, MAX_POSITIONS=2, SIGNAL_LAG_DAYS=0)
        
        _, adj, sel, _ = generate_signals(log_rets, adv, cfg)
        # SYM0 fails history gate (tail is NaN), so it gets -inf anyway.
        assert adj[0] == -np.inf
        assert 0 not in sel


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
