import numpy as np
import pandas as pd

import backtest_engine as be
from momentum_engine import InstitutionalRiskEngine, UltimateConfig


def test_rebalance_values_portfolio_from_previous_close(monkeypatch):
    cfg = UltimateConfig(CVAR_MIN_HISTORY=9999)
    engine = InstitutionalRiskEngine(cfg)
    bt = be.BacktestEngine(engine, initial_cash=100.0)
    bt.state.shares["AAA"] = 10
    bt.state.last_known_prices["AAA"] = 10.0

    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")])
    close = pd.DataFrame({"AAA": [10.0, 20.0]}, index=dates)
    volume = pd.DataFrame({"AAA": [1_000_000, 1_000_000]}, index=dates)
    returns = close.pct_change(fill_method=None).fillna(0.0)

    captured = {}

    def _fake_generate_signals(*_args, **_kwargs):
        return np.array([0.01]), np.array([0.01]), [0]

    def _fake_optimize(**kwargs):
        captured["pv"] = kwargs["portfolio_value"]
        return np.array([0.0])

    monkeypatch.setattr(be, "generate_signals", _fake_generate_signals)
    monkeypatch.setattr(be, "compute_regime_score", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(be, "compute_book_cvar", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(be, "execute_rebalance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "optimize", _fake_optimize)

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

    # cash (100) + shares (10) * previous close (10)
    assert captured["pv"] == 200.0


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
        dividends=dividends,
        splits=splits,
    )

    assert bt.state.shares["AAA"] == 10
    assert bt.state.cash == 0.0
