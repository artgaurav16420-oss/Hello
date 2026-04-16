import osqp_preimport
import pytest
import numpy as np
import pandas as pd

import backtest_engine as be
from momentum_engine import InstitutionalRiskEngine, UltimateConfig, RebalancePipelineResult


def _run_single_rebalance(bt, close, volume, returns):
    bt._run_rebalance(
        pd.Timestamp("2020-01-02"),
        close,
        volume,
        returns,
        ["AAA"],
        close.loc[pd.Timestamp("2020-01-02")].values.astype(float),
        idx_df=None,
        sector_map=None,
        open_px=close,
        high_px=close,
        low_px=close,
    )


def test_rebalance_values_portfolio_from_previous_close(monkeypatch):
    cfg = UltimateConfig(CVAR_MIN_HISTORY=9999, SIGNAL_LAG_DAYS=0)
    engine = InstitutionalRiskEngine(cfg)
    bt = be.BacktestEngine(engine, initial_cash=100.0)
    bt.state.shares["AAA"] = 10
    bt.state.last_known_prices["AAA"] = 10.0

    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")])
    close = pd.DataFrame({"AAA": [10.0, 20.0]}, index=dates)
    volume = pd.DataFrame({"AAA": [1_000_000, 1_000_000]}, index=dates)
    returns = close.pct_change(fill_method=None).fillna(0.0)

    captured = {}

    def _fake_run_rebalance_pipeline(ctx):
        captured["pv"] = ctx.pv
        return RebalancePipelineResult(
            total_slippage=0.0,
            applied_decay=False,
            optimization_succeeded=True,
            soft_cvar_breach=False,
            target_weights=np.array([0.0]),
        )

    monkeypatch.setattr(be, "run_rebalance_pipeline", _fake_run_rebalance_pipeline)

    _run_single_rebalance(bt, close, volume, returns)

    # cash (100) + shares (10) * previous close (10)
    assert captured["pv"] == 200.0


def test_run_rebalance_passes_adv_vector_to_execute_rebalance(monkeypatch):
    cfg = UltimateConfig(CVAR_MIN_HISTORY=9999, SIGNAL_LAG_DAYS=0)
    engine = InstitutionalRiskEngine(cfg)
    bt = be.BacktestEngine(engine, initial_cash=100.0)

    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")])
    close = pd.DataFrame({"AAA": [10.0, 11.0]}, index=dates)
    volume = pd.DataFrame({"AAA": [1_000_000, 1_000_000]}, index=dates)
    returns = close.pct_change(fill_method=None).fillna(0.0)
    captured = {}

    def _fake_run_rebalance_pipeline(ctx):
        captured["adv_shares"] = ctx.adv_vector
        return RebalancePipelineResult(
            total_slippage=0.0,
            applied_decay=False,
            optimization_succeeded=True,
            soft_cvar_breach=False,
            target_weights=np.array([1.0]),
        )

    monkeypatch.setattr(be, "run_rebalance_pipeline", _fake_run_rebalance_pipeline)
    monkeypatch.setattr(be, "_build_adv_vector", lambda *_args, **_kwargs: (np.array([12345.0]), pd.DataFrame()))

    _run_single_rebalance(bt, close, volume, returns)

    assert np.array_equal(captured["adv_shares"], np.array([12345.0]))


def test_run_applies_splits_and_fractional_slippage(monkeypatch):
    """Verify that the engine's split logic correctly adjusts shares and entry price,
    while deducting one-way slippage from fractional-share cash credits."""
    # cfg.AUTO_ADJUST_PRICES=False is required to trigger the engine's split logic.
    cfg = UltimateConfig(AUTO_ADJUST_PRICES=False, ROUND_TRIP_SLIPPAGE_BPS=100.0, CVAR_MIN_HISTORY=9999)
    engine = InstitutionalRiskEngine(cfg)
    bt = be.BacktestEngine(engine, initial_cash=0.0)
    # Start with 10 shares of AAA at 100.0 (total 1000.0).
    bt.state.shares["AAA"] = 10
    bt.state.entry_prices["AAA"] = 100.0

    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    # Day 1: 100, Day 2: 44.44 (approx 100/2.25)
    close = pd.DataFrame({"AAA": [100.0, 44.44]}, index=dates)
    volume = pd.DataFrame({"AAA": [1e6, 1e6]}, index=dates)
    returns = close.pct_change(fill_method=None).fillna(0.0)
    # 2.25:1 split on Day 2.
    splits = pd.DataFrame({"AAA": [1.0, 2.25]}, index=dates)

    # Run backtest for Day 1 and Day 2.
    bt.run(close, volume, returns, pd.DatetimeIndex([]), "2020-01-01", "2020-01-02", splits=splits)

    # 10 shares * 2.25 = 22.5 shares.
    # Floor(22.5) = 22 shares.
    # Fractional = 0.5 shares.
    # Price on Day 2 = 44.44.
    # Fractional Proceeds = 0.5 * 44.44 * (1 - 0.5 * 100/10000) = 22.22 * (1 - 0.005) = 22.22 * 0.995 = 22.1089.
    assert bt.state.shares["AAA"] == 22
    assert bt.state.cash == pytest.approx(22.1089)
    # Entry price should be adjusted: 100.0 / 2.25 = 44.4444.
    assert bt.state.entry_prices["AAA"] == pytest.approx(44.4444, abs=0.0001)


def test_run_skips_corporate_actions_when_auto_adjust_prices_enabled(monkeypatch):
    cfg = UltimateConfig(AUTO_ADJUST_PRICES=True, DIVIDEND_SWEEP=True, CVAR_MIN_HISTORY=9999)
    engine = InstitutionalRiskEngine(cfg)
    bt = be.BacktestEngine(engine, initial_cash=0.0)
    bt.state.shares["AAA"] = 10
    bt.state.entry_prices["AAA"] = 100.0

    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")])
    close = pd.DataFrame({"AAA": [100.0, 50.0]}, index=dates)
    volume = pd.DataFrame({"AAA": [1_000_000, 1_000_000]}, index=dates)
    returns = close.pct_change(fill_method=None).fillna(0.0)

    dividends = pd.DataFrame({"AAA": [0.0, 5.0]}, index=dates)
    splits = pd.DataFrame({"AAA": [0.0, 2.0]}, index=dates)

    monkeypatch.setattr(bt, "_run_rebalance", lambda *args, **kwargs: None)

    bt.run(
        close=close,
        volume=volume,
        returns=returns,
        rebalance_dates=pd.DatetimeIndex([]),
        start_date="2020-01-01",
        end_date="2020-01-02",
        dividends=dividends,
        splits=splits,
    )

    assert bt.state.shares["AAA"] == 10
    assert bt.state.cash == 0.0


def test_run_applies_dividends_on_rebalance_day(monkeypatch):
    """Verify that dividends are credited correctly on rebalance days,
    ensuring T+0 cash availability for record_eod but not for that day's rebalance."""
    cfg = UltimateConfig(AUTO_ADJUST_PRICES=False, DIVIDEND_SWEEP=True, CVAR_MIN_HISTORY=9999)
    engine = InstitutionalRiskEngine(cfg)
    bt = be.BacktestEngine(engine, initial_cash=0.0)
    bt.state.shares["AAA"] = 100

    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    close = pd.DataFrame({"AAA": [10.0, 10.0]}, index=dates)
    volume = pd.DataFrame({"AAA": [1e6, 1e6]}, index=dates)
    returns = close.pct_change(fill_method=None).fillna(0.0)
    # 1.0 dividend on Day 2.
    dividends = pd.DataFrame({"AAA": [0.0, 1.0]}, index=dates)

    rebalance_dates = pd.DatetimeIndex([dates[1]])
    
    # Capture cash during rebalance to verify it doesn't include the dividend yet.
    captured_cash = []
    original_rebalance = bt._run_rebalance
    def mock_rebalance(*args, **kwargs):
        captured_cash.append(bt.state.cash)
        return original_rebalance(*args, **kwargs)
    
    monkeypatch.setattr(bt, "_run_rebalance", mock_rebalance)
    # Mock pipeline to do nothing.
    monkeypatch.setattr(be, "run_rebalance_pipeline", lambda ctx: RebalancePipelineResult(
        total_slippage=0.0, applied_decay=False, optimization_succeeded=True, 
        soft_cvar_breach=False, target_weights=np.array([0.0])
    ))

    bt.run(close, volume, returns, rebalance_dates, "2020-01-01", "2020-01-02", dividends=dividends)

    # During rebalance, cash should still be 0.0.
    assert captured_cash[0] == 0.0
    # After Day 2, cash should include the dividend (100 shares * 1.0).
    assert bt.state.cash == 100.0
    # Final equity should include the dividend: cash(100) + shares(100)*price(10) = 1100.
    assert bt._eq_vals[-1] == 1100.0


    assert bt._eq_vals[-1] == 1100.0


def test_snap_rebalance_dates_to_holidays_merges_universe():
    """Verify that multiple theoretical rebalance dates mapping to the same
    valid trading day (e.g. holiday collision) results in a merged universe."""
    trading_index = pd.to_datetime(["2020-01-01", "2020-01-03"]) # Jan 2 is a holiday
    theoretical_dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    universe_map = {
        pd.Timestamp("2020-01-02"): {"AAA"},
        pd.Timestamp("2020-01-03"): {"BBB"},
    }
    
    # Both Jan 2 and Jan 3 should snap to Jan 3 (if Jan 3 is a trading day and target).
    # Actually Jan 2 snaps to Jan 1. Jan 3 snaps to Jan 3.
    # To test merging, we need them to snap to the SAME day.
    # Let's make Jan 3 a holiday too.
    trading_index = pd.to_datetime(["2020-01-01", "2020-01-05"])
    # Jan 2, 3, 4 are holidays.
    theoretical_dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    universe_map = {
        pd.Timestamp("2020-01-02"): {"AAA"},
        pd.Timestamp("2020-01-03"): {"BBB"},
    }
    
    snapped_dates, snapped_univ = be._snap_rebalance_dates_to_holidays(
        trading_index, theoretical_dates, universe_map
    )
    
    # Both should snap to Jan 1.
    assert len(snapped_dates) == 1
    assert snapped_dates[0] == pd.Timestamp("2020-01-01")
    assert snapped_univ[pd.Timestamp("2020-01-01")] == {"AAA", "BBB"}


def test_run_backtest_uses_adjusted_close_for_valuation_when_auto_adjust_enabled(monkeypatch):
    cfg = UltimateConfig(AUTO_ADJUST_PRICES=True, REBALANCE_FREQ="W-FRI", SIGNAL_LAG_DAYS=0)

    dates = pd.DatetimeIndex([
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ])
    market_data = {
        "AAA.NS": pd.DataFrame(
            {
                "Close": [100.0, 50.0, 50.0],
                "Adj Close": [100.0, 100.0, 100.0],
                "Open": [100.0, 50.0, 50.0],
                "High": [100.0, 50.0, 50.0],
                "Low": [100.0, 50.0, 50.0],
                "Volume": [1_000_000, 1_000_000, 1_000_000],
                "Dividends": [0.0, 0.0, 0.0],
                "Stock Splits": [0.0, 2.0, 0.0],
            },
            index=dates,
        ),
        "^NSEI": pd.DataFrame({"Close": [1.0, 1.0, 1.0]}, index=dates),
    }

    captured = {}

    def _fake_run(self, close, *_args, **_kwargs):
        captured["close"] = close.copy()
        # Return a non-empty dataframe to avoid issues in metric calculation
        return pd.DataFrame({"equity": pd.Series([1.0], index=[dates[0]])})

    import unittest.mock as mock
    with mock.patch.object(be.BacktestEngine, "run", _fake_run):
        results = be.run_backtest(
            market_data=market_data,
            start_date="2020-01-01",
            end_date="2020-01-03",
            cfg=cfg,
            universe=["AAA"],
        )

    assert list(captured["close"]["AAA"]) == [100.0, 100.0, 100.0]


def test_execution_prices_prefers_open_when_intraday_range_available():
    date = pd.Timestamp("2020-01-02")
    symbols = ["AAA"]
    close_prices = np.array([12.0])
    open_px = pd.DataFrame({"AAA": [10.0]}, index=[date])
    high_px = pd.DataFrame({"AAA": [15.0]}, index=[date])
    low_px = pd.DataFrame({"AAA": [9.0]}, index=[date])

    exec_px = be._execution_prices(symbols, date, close_prices, open_px, high_px, low_px)

    assert exec_px[0] == 10.0


def test_run_backtest_custom_universe_requires_recent_contiguous_volume():
    cfg = UltimateConfig(HISTORY_GATE=3, REBALANCE_FREQ="D")
    idx = pd.DatetimeIndex([
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-10"),
        pd.Timestamp("2020-01-13"),
    ])
    market_data = {
        "AAA.NS": pd.DataFrame(
            {
                "Close": [10.0, 10.0, 10.0, 10.0, 10.0],
                "Adj Close": [10.0, 10.0, 10.0, 10.0, 10.0],
                "Open": [10.0, 10.0, 10.0, 10.0, 10.0],
                "High": [10.0, 10.0, 10.0, 10.0, 10.0],
                "Low": [10.0, 10.0, 10.0, 10.0, 10.0],
                "Volume": [100.0, 100.0, 100.0, 0.0, 100.0],
                "Dividends": [0.0] * 5,
                "Stock Splits": [0.0] * 5,
            },
            index=idx,
        ),
        "^NSEI": pd.DataFrame({"Close": [1.0] * 5}, index=idx),
    }
    captured = {}

    def _fake_run(self, close, volume, returns, rebalance_dates, start_date, **kwargs):
        captured["universe_by_rebalance_date"] = kwargs["universe_by_rebalance_date"]
        return pd.DataFrame({"equity": pd.Series(dtype=float)})

    import unittest.mock as mock
    with mock.patch.object(be.BacktestEngine, "run", _fake_run):
        be.run_backtest(
            market_data=market_data,
            start_date="2020-01-10",
            end_date="2020-01-13",
            cfg=cfg,
            universe=["AAA"],
        )

    assert captured["universe_by_rebalance_date"][pd.Timestamp("2020-01-13")] == set()


def test_run_backtest_accepts_precomputed_bare_columns_for_ns_universe(monkeypatch):
    cfg = UltimateConfig(REBALANCE_FREQ="D")
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")])

    market_data = {
        "AAA.NS": pd.DataFrame(
            {
                "Close": [10.0, 11.0],
                "Adj Close": [10.0, 11.0],
                "Open": [10.0, 11.0],
                "High": [10.0, 11.0],
                "Low": [10.0, 11.0],
                "Volume": [1000.0, 1200.0],
                "Dividends": [0.0, 0.0],
                "Stock Splits": [0.0, 0.0],
            },
            index=idx,
        ),
        "^NSEI": pd.DataFrame({"Close": [1.0, 1.0]}, index=idx),
    }

    precomputed = {
        "close": pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx),
        "close_adj": pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx),
        "open": pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx),
        "high": pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx),
        "low": pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx),
        "dividends": pd.DataFrame({"AAA": [0.0, 0.0]}, index=idx),
        "splits": pd.DataFrame({"AAA": [0.0, 0.0]}, index=idx),
        "volume": pd.DataFrame({"AAA": [1000.0, 1200.0]}, index=idx),
        "returns": pd.DataFrame({"AAA": [0.0, 0.1]}, index=idx),
    }

    captured = {}

    def _fake_run(self, close, volume, returns, rebalance_dates, start_date, **kwargs):
        captured["columns"] = list(close.columns)
        return pd.DataFrame({"equity": pd.Series(dtype=float)})

    import unittest.mock as mock
    with mock.patch.object(be.BacktestEngine, "run", _fake_run):
        be.run_backtest(
            market_data=market_data,
            precomputed_matrices=precomputed,
            start_date="2020-01-01",
            end_date="2020-01-02",
            cfg=cfg,
            universe=["AAA.NS"],
        )

    assert captured["columns"] == ["AAA"]


def test_backtest_run_handles_empty_close_dataframe():
    cfg = UltimateConfig()
    engine = InstitutionalRiskEngine(cfg)
    bt = be.BacktestEngine(engine)

    empty = pd.DataFrame()
    out = bt.run(
        close=empty,
        volume=empty,
        returns=empty,
        rebalance_dates=pd.DatetimeIndex([]),
        start_date="2020-01-01",
    )

    assert out.empty
    assert list(out.columns) == ["equity"]


def test_compute_metrics_handles_non_positive_initial_capital():
    eq = pd.Series([100.0, 110.0], index=pd.date_range("2020-01-01", periods=2, freq="B"))

    metrics = be._compute_metrics(eq, initial=0.0)

    assert metrics["cagr"] == 0.0
    assert metrics["calmar"] == 0.0
    assert metrics["final"] == 110.0


def test_compute_metrics_turnover_uses_round_trip_units():
    eq = pd.Series(
        [100.0, 100.0],
        index=pd.to_datetime(["2020-01-01", "2020-12-31"]),
    )
    trades = [
        be.Trade("AAA", pd.Timestamp("2020-01-02"), 1, 100.0, 0.0, "BUY"),
        be.Trade("AAA", pd.Timestamp("2020-12-30"), -1, 100.0, 0.0, "SELL"),
    ]

    metrics = be._compute_metrics(eq, trades=trades, initial=100.0)

    assert metrics["turnover"] == pytest.approx(1.0, abs=0.01)


def test_repair_suspension_gaps_only_fills_detected_gap_not_entire_history():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-03-10"),
        pd.Timestamp("2020-03-11"),
    ])
    df = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0, 105.0, 106.0],
            "Adj Close": [50.0, 50.5, 51.0, 52.5, 53.0],
            "Volume": [1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000],
        },
        index=idx,
    )

    repaired = be._repair_suspension_gaps(df, "AAA.NS")

    # Original bars remain present (function must not replace the entire history index).
    for ts in idx:
        assert ts in repaired.index

    # Synthetic rows are created within the prolonged gap.
    assert len(repaired.index) > len(df.index)


def test_repair_suspension_gaps_preserves_adj_close_scale():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-03-10"),
    ])
    # Deliberately keep Adj Close at 1/10th Close to emulate split-adjusted scaling.
    df = pd.DataFrame(
        {
            "Close": [1000.0, 1020.0, 1010.0, 1030.0],
            "Adj Close": [100.0, 102.0, 101.0, 103.0],
            "Volume": [1_000_000, 1_000_000, 1_000_000, 1_000_000],
        },
        index=idx,
    )

    repaired = be._repair_suspension_gaps(df, "BBB.NS")
    gap_rows = repaired.index.difference(df.index)

    assert len(gap_rows) > 0

    ratio = repaired.loc[gap_rows, "Adj Close"] / repaired.loc[gap_rows, "Close"]
    # Synthetic Adj Close should preserve pre-gap adjusted scale, not be copied from raw Close.
    assert np.allclose(ratio.values, 0.1, atol=1e-6)

class TestBacktestEngine:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # Using a pytest fixture to ensure clean setup for each test
        self.dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="B")
        self.market_data = {
            "STOCK1.NS": pd.DataFrame({
                "Open": np.linspace(100, 105, len(self.dates)),
                "High": np.linspace(101, 106, len(self.dates)),
                "Low": np.linspace(99, 104, len(self.dates)),
                "Close": np.linspace(100, 105, len(self.dates)),
                "Adj Close": np.linspace(100, 105, len(self.dates)),
                "Volume": 1_000_000,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            }, index=self.dates),
            "STOCK2.NS": pd.DataFrame({
                "Open": np.linspace(200, 210, len(self.dates)),
                "High": np.linspace(202, 212, len(self.dates)),
                "Low": np.linspace(198, 208, len(self.dates)),
                "Close": np.linspace(200, 210, len(self.dates)),
                "Adj Close": np.linspace(200, 210, len(self.dates)),
                "Volume": 500_000,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            }, index=self.dates),
            "^NSEI": pd.DataFrame({
                "Close": np.linspace(18000, 18200, len(self.dates))
            }, index=self.dates)
        }
        self.cfg = UltimateConfig(
            INITIAL_CAPITAL=1_000_000,
            REBALANCE_FREQ="W-MON",
            # Use short lookbacks to fit into the small dataset
            HISTORY_GATE=5,
            CVAR_LOOKBACK=5,
            HALFLIFE_FAST=2,
            HALFLIFE_SLOW=4,
            ADV_LOOKBACK=5,
            # Adjusted these for smaller dataset and to satisfy min_periods <= window
            REGIME_SMA_FAST_WINDOW=10, # min_fast_periods becomes max(10, 10*0.8=8) = 10
            REGIME_SMA_WINDOW=25,     # min_sma_periods becomes max(20, 25*0.8=20) = 20
            EQUITY_HIST_CAP=10,
            SIGNAL_LAG_DAYS = 0,
        )

        self.universe = ["STOCK1.NS", "STOCK2.NS"]

    def test_run_backtest_main_scenario(self, monkeypatch):
        """
        Tests the main run_backtest function with a small, fixed dataset.
        Verifies that the backtest produces a valid result with trades.
        """
        # We have limited data, so we must override the warmup calculation
        def mock_compute_warmup_start(start_date, cfg):
            # Just need a few days for our short lookbacks
            return (pd.Timestamp(start_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

        monkeypatch.setattr(be, "_compute_warmup_start", mock_compute_warmup_start)

        results = be.run_backtest(
            market_data=self.market_data,
            universe=self.universe,
            start_date="2023-01-10", # Needs enough history for lookbacks
            end_date="2023-01-31",
            cfg=self.cfg,
        )

        assert isinstance(results, be.BacktestResults)
        assert not results.equity_curve.empty
        assert "cagr" in results.metrics
        # In a rising market with a momentum strategy, expect some profit.
        assert results.metrics["final"] > self.cfg.INITIAL_CAPITAL
        assert len(results.trades) > 0
        assert results.rebal_log is not None
        assert not results.rebal_log.empty

    def test_run_backtest_no_trading_days_in_range(self):
        """
        Tests run_backtest with a date range that contains no trading days
        from the market data, which should result in no rebalance dates.
        """
        with pytest.raises(RuntimeError, match="no valid rebalance dates"):
            be.run_backtest(
                market_data=self.market_data,
                universe=self.universe,
                start_date="2023-02-01",  # A date range with no data
                end_date="2023-02-10",
                cfg=self.cfg,
            )

    def test_run_backtest_invalid_date_range(self):
        """
        Tests that run_backtest raises a ValueError for an invalid date range
        where the start date is after the end date.
        """
        with pytest.raises(ValueError, match="Invalid backtest date range"):
            be.run_backtest(
                market_data=self.market_data,
                universe=self.universe,
                start_date="2023-01-31",
                end_date="2023-01-01",
                cfg=self.cfg,
            )
