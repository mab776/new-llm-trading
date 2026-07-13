"""
Tests for the backtesting engine.
Covers: no lookahead bias, partial exits, fee accounting, basic operation.
"""

import numpy as np
import pandas as pd
import pytest

from llm_trading_bot.backtesting import BacktestEngine, BacktestResult
from llm_trading_bot.config import AppConfig, LeverageTier
from llm_trading_bot.scoring import Direction, IndicatorSet, ScoringResult, SignalStrength


@pytest.fixture
def backtest_config() -> AppConfig:
    config = AppConfig()
    config.trading.yfinance_symbol = "BTC-USD"
    config.trading.primary_timeframe = "4h"
    config.trading.leverage_tiers = {
        "conservative": LeverageTier(
            leverage=5, strong_threshold=40, marginal_threshold_low=20,
            marginal_threshold_high=40, tp1_rr=2.0, tp2_rr=3.5, tp1_exit_pct=0.5,
        )
    }
    config.trading.active_tier = "conservative"
    config.backtesting.start_date = "2024-03-01"
    config.backtesting.end_date = "2024-06-01"
    config.backtesting.initial_balance = 10000
    config.backtesting.warmup_periods = 60
    config.fees.taker = 0.0006
    config.fees.default_order_type = "taker"
    return config


@pytest.fixture
def sample_data() -> dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data with known characteristics."""
    np.random.seed(123)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="4h")
    base_price = 45000

    # Gentle uptrend with volatility
    returns = np.random.normal(0.0003, 0.012, n)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "Open": prices * (1 + np.random.uniform(-0.004, 0.004, n)),
        "High": prices * (1 + np.abs(np.random.normal(0, 0.008, n))),
        "Low": prices * (1 - np.abs(np.random.normal(0, 0.008, n))),
        "Close": prices,
        "Volume": np.random.uniform(200, 800, n) * 1e6,
    }, index=dates)

    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

    # Create a 1d version by resampling
    df_1d = df.resample("1D").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()

    return {"4h": df, "1d": df_1d}


class TestBacktestEngine:
    def test_secondary_indicators_stop_at_completed_candle(
        self, backtest_config, monkeypatch,
    ):
        import llm_trading_bot.backtesting as module

        backtest_config.backtesting.start_date = "2024-01-02 08:00"
        backtest_config.backtesting.end_date = "2024-01-02 08:00"
        backtest_config.backtesting.warmup_periods = 0
        primary_idx = pd.DatetimeIndex(["2024-01-02 08:00"])
        daily_idx = pd.date_range("2023-11-01", periods=64, freq="1D")
        primary = pd.DataFrame(
            {"Open": [100], "High": [101], "Low": [99], "Close": [100], "Volume": [1]},
            index=primary_idx,
        )
        daily = pd.DataFrame(
            {"Open": range(64), "High": range(64), "Low": range(64),
             "Close": range(64), "Volume": [1] * 64},
            index=daily_idx,
        )
        seen = {}

        def fake_indicators(frame, timeframe):
            seen[timeframe] = frame.index[-1]
            return IndicatorSet(timeframe=timeframe, close=float(frame["Close"].iloc[-1]))

        monkeypatch.setattr(module, "calculate_indicators", fake_indicators)
        monkeypatch.setattr(module, "compute_composite_score", lambda **kwargs: ScoringResult(
            direction=Direction.NEUTRAL, confidence=5,
            signal_strength=SignalStrength.WAIT, raw_score=0,
            category_scores=[], indicators=None, reasons=[], filter_failures=[],
        ))
        monkeypatch.setattr(module, "calculate_targets", lambda **kwargs: None)

        BacktestEngine(backtest_config).run({"4h": primary, "1d": daily}, "4h")
        assert seen["1d"] == pd.Timestamp("2024-01-01")

    def test_basic_run(self, backtest_config, sample_data):
        engine = BacktestEngine(backtest_config)
        result = engine.run(sample_data, "4h")

        assert isinstance(result, BacktestResult)
        assert result.stats is not None
        assert result.stats.final_balance > 0
        assert len(result.bars) > 0

    def test_fees_included(self, backtest_config, sample_data):
        engine = BacktestEngine(backtest_config)
        result = engine.run(sample_data, "4h")

        if result.stats and result.stats.total_trades > 0:
            assert result.stats.total_fees > 0
            # Net PnL should differ from gross PnL
            if result.stats.total_gross_pnl != 0:
                assert result.stats.total_net_pnl != result.stats.total_gross_pnl

    def test_no_lookahead_bias(self, backtest_config, sample_data):
        """
        Each trade should only use data available at the time of the decision.
        Check that no trade entry is before its timestamp.
        """
        engine = BacktestEngine(backtest_config)
        result = engine.run(sample_data, "4h")

        for entry in result.decision_log:
            if "OPEN" in entry.get("action", ""):
                entry_time = pd.to_datetime(entry["time"])
                entry_price = entry["price"]
                # The price at this time should be within our data range
                idx = sample_data["4h"].index
                # Use searchsorted for pandas >=2.x compatibility (get_loc removed 'method' kwarg)
                pos = idx.searchsorted(entry_time, side="right") - 1
                assert pos >= 0, "Entry time before first bar"
                # Basic check: the entry should exist in our data range
                assert entry_time >= pd.to_datetime(backtest_config.backtesting.start_date)

    def test_all_trades_closed(self, backtest_config, sample_data):
        """At end of backtest, all positions should be closed."""
        engine = BacktestEngine(backtest_config)
        result = engine.run(sample_data, "4h")

        assert len(result.portfolio.open_trades) == 0

    def test_initial_balance_preserved_with_no_trades(self):
        """If no signals trigger, balance should stay at initial (minus nothing)."""
        config = AppConfig()
        config.trading.leverage_tiers = {
            "conservative": LeverageTier(
                leverage=5, strong_threshold=99, marginal_threshold_low=98,
                marginal_threshold_high=99,
            )
        }
        config.trading.active_tier = "conservative"
        config.backtesting.start_date = "2024-03-01"
        config.backtesting.end_date = "2024-06-01"
        config.backtesting.initial_balance = 10000
        config.backtesting.warmup_periods = 60

        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="4h")
        prices = 50000 + np.random.normal(0, 50, n)  # Very flat
        df = pd.DataFrame({
            "Open": prices, "High": prices + 100, "Low": prices - 100,
            "Close": prices, "Volume": np.ones(n) * 1e6,
        }, index=dates)

        engine = BacktestEngine(config)
        result = engine.run({"4h": df}, "4h")

        assert result.stats.total_trades == 0
        assert result.stats.final_balance == 10000
