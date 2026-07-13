"""Focused tests for the research-only lower-timeframe runner."""

import pandas as pd

from opt.lower_timeframes import (
    causal_precompute,
    normalize_to_bar_open,
    timeframe_delta,
)


def _frame(index, closes):
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [value + 1 for value in closes],
            "Low": [value - 1 for value in closes],
            "Close": closes,
            "Volume": [10.0] * len(closes),
        },
        index=index,
    )


def test_timeframe_delta_supports_intraday_intervals():
    assert timeframe_delta("5m") == pd.Timedelta(minutes=5)
    assert timeframe_delta("1h") == pd.Timedelta(hours=1)
    assert timeframe_delta("4h") == pd.Timedelta(hours=4)


def test_normalize_close_stamps_to_bar_open():
    index = pd.DatetimeIndex(
        ["2024-01-01 00:04:59.999000+00:00", "2024-01-01 00:09:59.999000+00:00"]
    )
    normalized = normalize_to_bar_open(_frame(index, [100.0, 101.0]), "5m")
    assert list(normalized.index) == list(pd.date_range(
        "2024-01-01", periods=2, freq="5min", tz="UTC"
    ))


def test_secondary_bar_is_invisible_until_its_close():
    # 60 primary 1h rows are enough for indicators.  The 4h row opened at 08:00
    # contains an unmistakable close=999, but it must not be visible to the 10:00
    # primary decision (which closes at 11:00).  It becomes visible to the 11:00
    # primary decision, which closes at 12:00.
    primary_index = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
    secondary_index = pd.date_range("2023-12-20", periods=100, freq="4h", tz="UTC")
    closes = [100.0] * len(secondary_index)
    special = secondary_index.get_loc(pd.Timestamp("2024-01-01 08:00", tz="UTC"))
    closes[special] = 999.0
    pre = causal_precompute(
        {
            "1h": _frame(primary_index, [100.0] * len(primary_index)),
            "4h": _frame(secondary_index, closes),
        },
        "1h",
        warmup=0,
    )

    at_10 = primary_index.get_loc(pd.Timestamp("2024-01-01 10:00", tz="UTC"))
    at_11 = primary_index.get_loc(pd.Timestamp("2024-01-01 11:00", tz="UTC"))
    assert pre.sec_by_bar[at_10]["4h"].close == 100.0
    assert pre.sec_by_bar[at_11]["4h"].close == 999.0


def test_one_hour_precompute_can_replay_twelve_five_minute_exit_bars():
    primary_index = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
    sub_index = pd.date_range("2024-01-01", periods=60 * 12, freq="5min", tz="UTC")
    primary = _frame(primary_index, [100.0] * len(primary_index))
    sub = _frame(sub_index, [100.0] * len(sub_index))
    pre = causal_precompute(
        {"1h": primary}, "1h", warmup=0, exit_subframe=sub
    )
    assert len(pre.subbars[0]) == 12
    assert pre.subbars[0][0] == (101.0, 99.0, 100.0)
