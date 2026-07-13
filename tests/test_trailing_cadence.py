"""
Live trailing-stop cadence — the ratchet must fire once per COMPLETED primary bar,
using that bar's favorable extreme, never on every position-check tick.

Why this matters: an honest 1h sub-bar backtest showed that ratcheting the trailing
stop hourly (let alone every 15 min on spot price) collapses the strategy's edge
(84x -> 5x over 2021-2025) — winners get choked on intrabar noise. The backtested
edge assumes bar-close ratcheting with the stop fixed intrabar. These tests pin the
scheduler to that behavior.
"""
from __future__ import annotations

import types

import pandas as pd
import pytest

import llm_trading_bot.scheduler as sched_mod
from llm_trading_bot.config import AppConfig, load_config
from llm_trading_bot.scheduler import TradingScheduler
from llm_trading_bot.portfolio import Portfolio
from opt.fastbt import _check_exits, _ratchet_trailing_stop


class FakeExchange:
    def __init__(self):
        self.modify_calls = []

    def modify_stop_loss(self, symbol, hold_side, size, new_sl):
        self.modify_calls.append(new_sl)


class FakePos:
    symbol = "BTC-USDT"
    side = "long"
    size = 1.0
    entry_price = 100.0
    unrealized_pnl = 5.0


def _mk_scheduler(monkeypatch, df):
    cfg = load_config("config.json")
    cfg.trading.trailing_stop.enabled = True
    cfg.trading.trailing_stop.activation_pct = 1.0
    cfg.trading.trailing_stop.callback_pct = 1.0
    sch = TradingScheduler.__new__(TradingScheduler)  # skip __init__ (no exchange creds)
    sch.config = cfg
    sch.exchange = FakeExchange()
    sch._tracked_trades = {
        "BTC-USDT": {"direction": "LONG", "entry": 100.0, "current_sl": 95.0}
    }
    sch._log = lambda msg: None
    monkeypatch.setattr(sched_mod, "fetch_multi_timeframe", lambda **kw: {"4h": df})
    return sch


def test_subbar_exit_replay_keeps_trailing_fixed_until_primary_close():
    """A favorable first hour cannot raise the stop for a later hour in the same 4h bar."""
    cfg = AppConfig()
    cfg.trading.trailing_stop.enabled = True
    cfg.trading.trailing_stop.activation_pct = 1.0
    cfg.trading.trailing_stop.callback_pct = 0.5
    port = Portfolio(initial_balance=1000)
    trade = port.open_trade(
        "LONG", 100, "t0", 90, 200, 300, leverage=1, risk_pct=.1,
    )
    strat = {"trail_mode": "pct"}

    _check_exits(
        port, trade, 110, 100, 105, "t1", cfg.risk_management, 4,
        False, True, cfg.trading.trailing_stop, st=strat,
        ratchet_trailing=False,
    )
    _check_exits(
        port, trade, 105, 95, 100, "t1", cfg.risk_management, 4,
        False, True, cfg.trading.trailing_stop, st=strat, count_bar=False,
        ratchet_trailing=False,
    )
    assert trade.is_open
    assert trade.stop_loss == 90

    _ratchet_trailing_stop(trade, 110, cfg.trading.trailing_stop, strat)
    assert trade.stop_loss == pytest.approx(109.5)


def _bars(rows):
    """rows: list of (ts, high, low). Timestamps are bar OPEN times."""
    idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC")
    return pd.DataFrame(
        {"Open": [100.0] * len(rows), "High": [r[1] for r in rows],
         "Low": [r[2] for r in rows], "Close": [100.0] * len(rows),
         "Volume": [1.0] * len(rows)},
        index=idx,
    )


def test_ratchet_uses_last_completed_bar_high(monkeypatch):
    """The still-forming bar must be ignored; the completed bar's high drives the stop."""
    now = pd.Timestamp.now(tz="UTC").floor("4h")
    df = _bars([
        (now - pd.Timedelta(hours=8), 110.0, 101.0),   # completed bar: high 110
        (now, 200.0, 100.0),                            # still-forming bar (ignore!)
    ])
    sch = _mk_scheduler(monkeypatch, df)
    sch._maybe_trail_stop(FakePos())
    # callback 1% of entry(100) = 1.0 -> new stop = 110 - 1 = 109, NOT 199
    assert sch.exchange.modify_calls == [109.0]


def test_ratchet_fires_once_per_completed_bar(monkeypatch):
    """Repeated 15-min position checks within the same bar must not re-ratchet."""
    now = pd.Timestamp.now(tz="UTC").floor("4h")
    df = _bars([
        (now - pd.Timedelta(hours=8), 110.0, 101.0),
        (now, 120.0, 100.0),
    ])
    sch = _mk_scheduler(monkeypatch, df)
    for _ in range(5):  # five ticks inside the same 4h bar
        sch._maybe_trail_stop(FakePos())
    assert len(sch.exchange.modify_calls) == 1


def test_new_completed_bar_ratchets_again(monkeypatch):
    now = pd.Timestamp.now(tz="UTC").floor("4h")
    df1 = _bars([
        (now - pd.Timedelta(hours=8), 110.0, 101.0),
        (now, 120.0, 100.0),
    ])
    sch = _mk_scheduler(monkeypatch, df1)
    sch._maybe_trail_stop(FakePos())
    assert sch.exchange.modify_calls == [109.0]
    sch._tracked_trades["BTC-USDT"]["current_sl"] = 109.0

    # next 4h bar completes with a higher high
    df2 = _bars([
        (now - pd.Timedelta(hours=4), 115.0, 105.0),
        (now, 116.0, 100.0),
    ])
    monkeypatch.setattr(sched_mod, "fetch_multi_timeframe", lambda **kw: {"4h": df2})
    sch._maybe_trail_stop(FakePos())
    assert sch.exchange.modify_calls == [109.0, 114.0]
