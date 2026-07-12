"""Funding-rate modeling — sign conventions, per-bar aggregation, cache round-trip."""
from __future__ import annotations

import pandas as pd

from llm_trading_bot.funding import (
    aggregate_funding_to_bars,
    funding_cost,
    _load_cache,
    _save_cache,
)
from llm_trading_bot.portfolio import Portfolio


def test_funding_cost_sign_conventions():
    # Positive rate: LONG pays, SHORT receives
    assert funding_cost("LONG", 0.0001, size=1.0, mark_price=50_000) == 5.0
    assert funding_cost("SHORT", 0.0001, size=1.0, mark_price=50_000) == -5.0
    # Negative rate: LONG receives, SHORT pays
    assert funding_cost("LONG", -0.0001, size=1.0, mark_price=50_000) == -5.0
    assert funding_cost("SHORT", -0.0001, size=1.0, mark_price=50_000) == 5.0
    # Charged on notional (size × price), not margin
    assert funding_cost("LONG", 0.0001, size=2.0, mark_price=50_000) == 10.0


def test_aggregate_funding_to_bars_buckets_by_bar_open():
    bars = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 04:00", "2024-01-01 08:00"], tz="UTC"
    )
    funding = pd.Series(
        [0.0001, 0.0002, 0.0003],
        index=pd.DatetimeIndex(
            # 00:00 -> bar 0; 08:00 -> bar 2; 12:00 falls past the last bar's window end?
            # last bar covers 08:00-12:00 (exclusive), so 08:00 event lands in bar 2.
            ["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 11:59"], tz="UTC"
        ),
    )
    sums = aggregate_funding_to_bars(funding, bars, bar_hours=4)
    assert sums == [0.0001, 0.0, 0.0005]


def test_aggregate_ignores_events_outside_bars():
    bars = pd.DatetimeIndex(["2024-01-01 08:00"], tz="UTC")
    funding = pd.Series(
        [0.001, 0.002, 0.003],
        index=pd.DatetimeIndex(
            ["2024-01-01 00:00",   # before the first bar
             "2024-01-01 08:00",   # inside
             "2024-01-01 12:00"],  # at/after window end (exclusive)
            tz="UTC",
        ),
    )
    sums = aggregate_funding_to_bars(funding, bars, bar_hours=4)
    assert sums == [0.002]


def test_portfolio_apply_funding_moves_balance_and_trade_pnl():
    p = Portfolio(initial_balance=1000)
    t = p.open_trade("LONG", 50_000, "t0", 49_000, 60_000, 70_000,
                     leverage=25, risk_pct=0.02, tp1_exit_pct=0.5)
    bal_before = p.balance
    pnl_before = t.net_pnl
    p.apply_funding(t, 2.5)     # pays 2.5
    assert p.balance == bal_before - 2.5
    assert t.net_pnl == pnl_before - 2.5
    assert t.funding_paid == 2.5
    p.apply_funding(t, -1.0)    # receives 1.0
    assert p.balance == bal_before - 1.5
    assert t.funding_paid == 1.5


def test_cache_round_trip(tmp_path):
    path = tmp_path / "fund.csv"
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 08:00"], tz="UTC")
    s = pd.Series([0.0001, -0.0002], index=idx, name="rate")
    _save_cache(path, s)
    loaded = _load_cache(path)
    assert list(loaded.values) == [0.0001, -0.0002]
    assert (loaded.index == idx).all()
