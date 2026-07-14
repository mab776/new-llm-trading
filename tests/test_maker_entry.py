"""Honest maker-entry fill, expiry, fee, and live lifecycle tests."""

from __future__ import annotations

import pandas as pd
import pytest

import llm_trading_bot.backtesting as bt_mod
from llm_trading_bot.backtesting import BacktestEngine
from llm_trading_bot.config import AppConfig, LeverageTier
from llm_trading_bot.entry import maker_limit_touched
from llm_trading_bot.exchange import BitgetClient
from llm_trading_bot.portfolio import Portfolio
from llm_trading_bot.scoring import (
    Direction, IndicatorSet, ScoringResult, SignalStrength, TradeTargets,
)
from llm_trading_bot.scheduler import TradingScheduler


def test_shared_maker_touch_rule():
    assert maker_limit_touched("LONG", 100, bar_high=110, bar_low=100)
    assert not maker_limit_touched("LONG", 100, bar_high=110, bar_low=100.01)
    assert maker_limit_touched("SHORT", 100, bar_high=100, bar_low=90)
    assert not maker_limit_touched("SHORT", 100, bar_high=99.99, bar_low=90)


def _engine_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.trading.entry_mode = "maker"
    cfg.trading.primary_timeframe = "4h"
    cfg.trading.leverage_tiers = {
        "x": LeverageTier(leverage=1, strong_threshold=20,
                           marginal_threshold_low=10, tp1_rr=2, tp2_rr=3)
    }
    cfg.trading.active_tier = "x"
    cfg.backtesting.start_date = "2024-01-01"
    cfg.backtesting.end_date = "2024-01-02"
    cfg.backtesting.warmup_periods = 0
    cfg.backtesting.enable_trailing_stops = False
    cfg.risk_management.cooldown_candles_after_sl = 0
    cfg.filters.min_profit_after_fees = False
    cfg.filters.min_category_agreement = 0
    cfg.filters.skip_choppy_regime = False
    return cfg


def _patch_always_long(monkeypatch):
    def indicators(df, tf):
        row = df.iloc[-1]
        return IndicatorSet(
            timeframe=tf, open=float(row.Open), high=float(row.High),
            low=float(row.Low), close=float(row.Close), atr_14=2.5,
            atr_pct=2.5, adx=30,
        )

    monkeypatch.setattr(bt_mod, "calculate_indicators", indicators)
    monkeypatch.setattr(bt_mod, "compute_composite_score", lambda **kw: ScoringResult(
        direction=Direction.BULLISH, confidence=80,
        signal_strength=SignalStrength.STRONG, raw_score=50,
        category_scores=[], indicators=kw["indicators_by_tf"], reasons=[],
    ))
    monkeypatch.setattr(bt_mod, "calculate_targets", lambda indicators, **kw: TradeTargets(
        entry=indicators.close, stop_loss=95, take_profit_1=110,
        take_profit_2=120, risk_amount=5, reward_1=10, reward_2=20,
        direction=Direction.BULLISH,
    ))
    monkeypatch.setattr(bt_mod, "apply_pre_trade_filters", lambda **kw: [])


def test_engine_maker_fill_can_stop_out_on_fill_bar(monkeypatch):
    _patch_always_long(monkeypatch)
    idx = pd.date_range("2024-01-01", periods=2, freq="4h")
    df = pd.DataFrame({
        "Open": [100, 102], "High": [101, 115], "Low": [99, 90],
        "Close": [100, 105], "Volume": [1, 1],
    }, index=idx)
    result = BacktestEngine(_engine_cfg()).run({"4h": df})
    trade = result.portfolio.trades[0]
    assert trade.entry_price == 100
    assert trade.exit_reason == "sl"  # not TP despite spanning both
    assert trade.exit_time == str(idx[1])
    assert trade.net_pnl < 0


def test_engine_cancels_unfilled_maker_after_one_bar(monkeypatch):
    _patch_always_long(monkeypatch)
    idx = pd.date_range("2024-01-01", periods=2, freq="4h")
    df = pd.DataFrame({
        "Open": [100, 103], "High": [101, 106], "Low": [99, 101],
        "Close": [100, 105], "Volume": [1, 1],
    }, index=idx)
    result = BacktestEngine(_engine_cfg()).run({"4h": df})
    assert result.portfolio.trades == []
    assert any(x["action"] == "CANCEL_MAKER_UNFILLED" for x in result.decision_log)


def test_entry_fee_can_be_selected_per_trade():
    p = Portfolio(initial_balance=1000, maker_fee=0.0002, taker_fee=0.0006)
    maker = p.open_trade("LONG", 100, "a", 90, 110, 120,
                         leverage=1, risk_pct=.1, order_type="maker")
    p.close_trade(maker, 100, "b", "manual")
    taker = p.open_trade("LONG", 100, "c", 90, 110, 120,
                         leverage=1, risk_pct=.1, order_type="taker")
    assert maker.entry_fee < taker.entry_fee


def test_limit_order_is_post_only_and_keeps_safety_targets(monkeypatch):
    client = BitgetClient(_engine_cfg().bitget)
    calls = []
    monkeypatch.setattr(client, "_request", lambda method, path, **kw:
                        calls.append((method, path, kw)) or
                        {"data": {"orderId": "x"}, "code": "00000"})
    targets = TradeTargets(100, 95, 110, 120, 5, 10, 20, Direction.BULLISH)
    client.place_order("BTCUSDT", "buy", 1, targets, 5,
                       order_type="limit", price=100)
    body = calls[-1][2]["body"]
    assert body["force"] == "post_only"
    assert body["presetStopLossPrice"] == "95"
    assert body["presetStopSurplusPrice"] == "120"  # TP2 until per-lot plans reconcile


def test_live_pending_fill_is_promoted_to_trailing_context(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = _engine_cfg()
    cfg.trading.symbol = "BTCUSDT"
    scheduler = TradingScheduler(cfg)
    scheduler._pending_orders = {
        "o1": {"symbol": "BTCUSDT", "direction": "LONG", "entry": 100,
               "stop_loss": 95, "take_profit_1": 110, "take_profit_2": 120,
               "size": 1, "placed_at": 0, "expires_at": 99999999999}
    }
    monkeypatch.setattr(scheduler.exchange, "get_order_detail",
                        lambda *a: {"state": "filled", "priceAvg": "99.5"})
    scheduler._reconcile_pending_orders()
    assert scheduler._pending_orders == {}
    tracked = next(iter(scheduler._tracked_trades.values()))
    assert tracked["direction"] == "LONG"
    assert tracked["entry"] == 99.5
    assert tracked["current_sl"] == 95
    assert tracked["last_trail_bar"]


def test_live_expired_pending_is_cancelled(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    scheduler = TradingScheduler(_engine_cfg())
    scheduler._pending_orders = {
        "o1": {"symbol": "BTCUSDT", "direction": "LONG", "entry": 100,
               "stop_loss": 95, "take_profit_1": 110, "take_profit_2": 120,
               "size": 1, "placed_at": 0, "expires_at": 0}
    }
    details = iter([{"state": "live"}, {"state": "canceled", "baseVolume": "0"}])
    monkeypatch.setattr(scheduler.exchange, "get_order_detail", lambda *a: next(details))
    cancelled = []
    monkeypatch.setattr(scheduler.exchange, "cancel_order",
                        lambda symbol, oid: cancelled.append((symbol, oid)) or {})
    scheduler._reconcile_pending_orders()
    assert cancelled == [("BTCUSDT", "o1")]
    assert scheduler._pending_orders == {}
