"""Completed-candle availability shared by backtest, analysis, and live."""

import pandas as pd

from llm_trading_bot.timeframes import (
    completed_market_snapshot,
    decision_close,
    slice_completed_at,
    timeframe_delta,
)


def _frame(index):
    n = len(index)
    return pd.DataFrame(
        {
            "Open": range(n), "High": range(n), "Low": range(n),
            "Close": range(n), "Volume": [1.0] * n,
        },
        index=index,
    )


def test_timeframe_durations_are_generic():
    assert timeframe_delta("5m") == pd.Timedelta(minutes=5)
    assert timeframe_delta("1h") == pd.Timedelta(hours=1)
    assert timeframe_delta("4h") == pd.Timedelta(hours=4)


def test_higher_timeframe_open_is_hidden_until_close():
    daily = _frame(pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC"))
    # Primary 4h bar opened Jan 2 08:00 and closes at 12:00. The Jan 2 daily
    # candle is still open then, so Jan 1 must be the final available daily row.
    cutoff = decision_close(pd.Timestamp("2024-01-02 08:00", tz="UTC"), "4h")
    visible = slice_completed_at(daily, "1d", cutoff)
    assert visible.index[-1] == pd.Timestamp("2024-01-01", tz="UTC")


def test_live_snapshot_ignores_forming_primary_and_freezes_secondaries():
    primary = _frame(pd.date_range("2024-01-01", periods=9, freq="4h", tz="UTC"))
    hourly = _frame(pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC"))
    daily = _frame(pd.date_range("2023-12-30", periods=5, freq="1D", tz="UTC"))
    snapshot, primary_open = completed_market_snapshot(
        {"1h": hourly, "4h": primary, "1d": daily}, "4h",
        now=pd.Timestamp("2024-01-02 08:30", tz="UTC"),
    )
    # 08:00 primary is forming; 04:00 -> 08:00 is the decision bar.
    assert primary_open == pd.Timestamp("2024-01-02 04:00", tz="UTC")
    assert snapshot["4h"].index[-1] == primary_open
    assert snapshot["1h"].index[-1] == pd.Timestamp("2024-01-02 07:00", tz="UTC")
    assert snapshot["1d"].index[-1] == pd.Timestamp("2024-01-01", tz="UTC")
