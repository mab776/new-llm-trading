"""
Intrabar path conservatism — guards against optimistic fills.

Since a backtest only has OHLC per bar (not the intrabar path), the engine MUST assume
the ADVERSE extreme is reached before the favorable one. These tests prove:

1. SL/TP variant: a bar that spans BOTH the stop loss AND a take profit exits at the SL
   (the loss), never the TP — i.e. "straight to the low then higher" cannot be claimed
   as a win.
2. Trailing variant: the trailing stop is ratcheted using a bar's high only AFTER that
   bar's low has been checked against the pre-existing stop, and it only affects
   SUBSEQUENT bars — so a reversal bar exits at the PRIOR trailed level, not at a level
   derived from the same bar's high.
"""
from __future__ import annotations

import copy

from llm_trading_bot.config import load_config
from llm_trading_bot.backtesting import BacktestEngine


def _engine(trailing=False, activation=1.0, callback=1.0):
    cfg = load_config("config.json")
    cfg.risk_management.max_holding_hours = 0
    cfg.backtesting.enable_partial_exits = True
    cfg.backtesting.enable_trailing_stops = trailing
    cfg.trading.trailing_stop.enabled = trailing
    cfg.trading.trailing_stop.activation_pct = activation
    cfg.trading.trailing_stop.callback_pct = callback
    return BacktestEngine(cfg)


def _open(engine, direction, entry, sl, tp1, tp2):
    return engine.portfolio.open_trade(
        direction=direction, entry_price=entry, entry_time="t0",
        stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
        leverage=1, risk_pct=0.02, tp1_exit_pct=0.5,
    )


def test_long_bar_spanning_sl_and_tp_exits_at_sl():
    """LONG: one bar whose low pierces the SL and whose high pierces TP2 -> must take SL."""
    eng = _engine()
    t = _open(eng, "LONG", entry=100, sl=95, tp1=110, tp2=120)
    # Bar goes to the low first (95 hit) then rallies to 125 — worst case must win.
    eng._check_exits(t, bar_high=125.0, bar_low=94.0, bar_close=120.0, bar_time="t1")
    assert not t.is_open
    assert t.exit_reason == "sl"
    assert t.exit_price == 95.0
    assert t.net_pnl < 0


def test_short_bar_spanning_sl_and_tp_exits_at_sl():
    """SHORT: one bar whose high pierces the SL and whose low pierces TP2 -> must take SL."""
    eng = _engine()
    t = _open(eng, "SHORT", entry=100, sl=105, tp1=90, tp2=80)
    eng._check_exits(t, bar_high=106.0, bar_low=79.0, bar_close=80.0, bar_time="t1")
    assert not t.is_open
    assert t.exit_reason == "sl"
    assert t.exit_price == 105.0
    assert t.net_pnl < 0


def test_trailing_stop_only_applies_next_bar_and_uses_prior_level():
    """
    LONG with trailing (activation 1%, callback 1% of entry=100 -> callback distance 1.0).

    Bar A: high 110 -> after the bar the stop trails to 110-1 = 109 (low 103 didn't breach 95).
    Bar B (reversal): high 112, low 104. The stop coming INTO bar B is 109. The bar's low
      (104) breaches 109, so we exit at 109 — the PRIOR trailed level.

    The optimistic bug would first trail to 112-1 = 111 using bar B's own high, then exit at
    111. So the exit price proves which behavior is active: must be 109, never 111.
    """
    eng = _engine(trailing=True, activation=1.0, callback=1.0)
    t = _open(eng, "LONG", entry=100, sl=95, tp1=1000, tp2=2000)  # TPs far away

    # Bar A — order mirrors run(): check exits first, then ratchet the trail.
    eng._check_exits(t, bar_high=110.0, bar_low=103.0, bar_close=108.0, bar_time="A")
    eng._update_trailing_stop(t, bar_high=110.0, bar_low=103.0)
    assert t.is_open
    assert t.stop_loss == 109.0  # trailed using bar A's high, for use on later bars

    # Bar B — reversal.
    eng._check_exits(t, bar_high=112.0, bar_low=104.0, bar_close=105.0, bar_time="B")
    assert not t.is_open
    assert t.exit_price == 109.0   # prior trailed level — NOT 111 (bar B's high-derived level)


def test_trailing_does_not_skip_a_stop_hit_on_the_ratchet_bar():
    """
    If, on the very bar that would ratchet the stop up, the low ALSO breaches the current
    (pre-ratchet) stop, the exit must happen at the pre-ratchet stop — the ratchet must not
    rescue the trade by moving the stop out of the way first.
    """
    eng = _engine(trailing=True, activation=1.0, callback=1.0)
    t = _open(eng, "LONG", entry=100, sl=99, tp1=1000, tp2=2000)
    # Bar makes a high of 110 (would trail to 109) but its low is 98 (< current stop 99).
    eng._check_exits(t, bar_high=110.0, bar_low=98.0, bar_close=100.0, bar_time="A")
    eng._update_trailing_stop(t, bar_high=110.0, bar_low=98.0)
    assert not t.is_open
    assert t.exit_reason in ("sl", "trailing_stop")
    assert t.exit_price == 99.0    # exited at the original stop, not rescued by the ratchet
