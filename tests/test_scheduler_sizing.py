"""
Tests for risk-based live position sizing (replaces the old hardcoded size=0.001)
and the live trailing-stop trigger in the scheduler.
"""

import pytest

from llm_trading_bot.config import AppConfig, LeverageTier
from llm_trading_bot.exchange import Position
from llm_trading_bot.routing import RoutingDecision
from llm_trading_bot.scheduler import TradingScheduler
from llm_trading_bot.scoring import (
    Direction,
    ScoringResult,
    SignalStrength,
    TradeTargets,
)


def _config() -> AppConfig:
    cfg = AppConfig()
    cfg.trading.symbol = "BTC/USDT:USDT"
    cfg.trading.leverage_tiers = {"aggressive": LeverageTier(leverage=10)}
    cfg.trading.active_tier = "aggressive"
    cfg.position_sizing.risk_pct_per_trade = 0.02
    cfg.position_sizing.max_position_usd = 100
    return cfg


def _decision(direction=Direction.BULLISH) -> RoutingDecision:
    targets = TradeTargets(
        entry=50000, stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
        risk_amount=1000, reward_1=2000, reward_2=4000, direction=direction,
    )
    sr = ScoringResult(
        direction=direction, confidence=80, signal_strength=SignalStrength.STRONG,
        raw_score=40, category_scores=[], indicators=None, reasons=[], filter_failures=[],
    )
    return RoutingDecision(
        signal_strength=SignalStrength.STRONG, scoring_result=sr, targets=targets,
    )


class TestLiveSizing:
    def test_size_from_risk_not_hardcoded(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)  # keep logs/ out of the repo
        sched = TradingScheduler(_config())
        monkeypatch.setattr(sched.exchange, "get_available_balance", lambda *a, **k: 5000.0)
        monkeypatch.setattr(sched.exchange, "get_positions", lambda *a, **k: [])

        captured = {}

        def fake_place_order(symbol, side, size, targets, leverage):
            captured["size"] = size
            from llm_trading_bot.exchange import OrderResult
            return OrderResult(
                order_id="x", symbol=symbol, side=side, size=size, price=None,
                stop_loss=targets.stop_loss, take_profit_1=targets.take_profit_1,
                take_profit_2=targets.take_profit_2, status="submitted",
                timestamp="t", raw_response={},
            )

        monkeypatch.setattr(sched.exchange, "place_order", fake_place_order)

        sched._execute_trade(_decision())

        # margin = min(5000 * 0.02, 100) = 100; notional = 100 * 10 = 1000; size = 1000/50000
        assert captured["size"] == pytest.approx(1000 / 50000)
        assert captured["size"] != 0.001

    def test_zero_balance_skips_trade(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        sched = TradingScheduler(_config())
        monkeypatch.setattr(sched.exchange, "get_available_balance", lambda *a, **k: 0.0)
        monkeypatch.setattr(sched.exchange, "get_positions", lambda *a, **k: [])

        called = {"placed": False}
        monkeypatch.setattr(
            sched.exchange, "place_order",
            lambda **k: called.__setitem__("placed", True),
        )
        sched._execute_trade(_decision())
        assert called["placed"] is False


class TestLiveTrailing:
    def test_trailing_update_moves_stop_up_on_completed_bar(self, monkeypatch, tmp_path):
        """The ratchet fires on the last COMPLETED primary bar's high — never on the
        current price (see tests/test_trailing_cadence.py for why cadence matters)."""
        import pandas as pd
        import llm_trading_bot.scheduler as sched_mod

        monkeypatch.chdir(tmp_path)
        cfg = _config()
        cfg.trading.trailing_stop.enabled = True
        cfg.trading.trailing_stop.activation_pct = 1.0
        cfg.trading.trailing_stop.callback_pct = 0.5
        sched = TradingScheduler(cfg)
        sched._tracked_trades["BTC/USDT:USDT"] = {
            "direction": "LONG", "entry": 50000.0, "current_sl": 49000.0,
        }

        moved = {}
        monkeypatch.setattr(
            sched.exchange, "modify_stop_loss",
            lambda symbol, hold_side, size, new_sl: moved.update(new_sl=new_sl) or {"code": "00000"},
        )

        # Completed 4h bar with high 51000 (2% above entry, past 1% activation);
        # the still-forming bar spikes to 60000 and must be ignored.
        now = pd.Timestamp.now(tz="UTC").floor("4h")
        idx = pd.DatetimeIndex([now - pd.Timedelta(hours=8), now], tz="UTC")
        df = pd.DataFrame(
            {"Open": [50000.0, 51000.0], "High": [51000.0, 60000.0],
             "Low": [49500.0, 50500.0], "Close": [50900.0, 59000.0],
             "Volume": [1.0, 1.0]},
            index=idx,
        )
        monkeypatch.setattr(sched_mod, "fetch_multi_timeframe", lambda **kw: {"4h": df})

        pos = Position(
            symbol="BTC/USDT:USDT", side="long", size=0.04, entry_price=50000.0,
            unrealized_pnl=40.0, leverage=10, margin_mode="crossed", timestamp="t",
        )
        sched._maybe_trail_stop(pos)
        # new SL = completed-bar high 51000 - 0.5% of entry (250) = 50750 (NOT 59750)
        assert moved["new_sl"] == pytest.approx(50750.0)
        assert sched._tracked_trades["BTC/USDT:USDT"]["current_sl"] == pytest.approx(50750.0)

        # a second tick inside the same bar must not ratchet again
        moved.clear()
        sched._maybe_trail_stop(pos)
        assert moved == {}
