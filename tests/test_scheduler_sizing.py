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
    def test_run_cycle_uses_completed_snapshot_and_persists_once_gate(
        self, monkeypatch, tmp_path,
    ):
        import pandas as pd
        import llm_trading_bot.scheduler as module
        from llm_trading_bot.live_state import SharedLiveState
        from llm_trading_bot.scoring import IndicatorSet

        cfg = _config()
        cfg.trading.primary_timeframe = "4h"
        cfg.trading.timeframes = ["1h", "4h", "1d"]
        state_path = tmp_path / "state.json"
        state = SharedLiveState(state_path)
        scheduler = TradingScheduler(cfg, shared_state=state, log_dir=tmp_path)
        now = pd.Timestamp.now(tz="UTC").floor("4h")

        def frame(index):
            n = len(index)
            return pd.DataFrame(
                {"Open": range(n), "High": range(n), "Low": range(n),
                 "Close": range(n), "Volume": [1.0] * n}, index=index,
            )

        data = {
            "1h": frame(pd.date_range(end=now, periods=240, freq="1h", tz="UTC")),
            "4h": frame(pd.date_range(end=now, periods=60, freq="4h", tz="UTC")),
            "1d": frame(pd.date_range(end=now.floor("D"), periods=60, freq="1D", tz="UTC")),
        }
        seen = {}

        def fake_indicators(df, timeframe):
            seen[timeframe] = df.index[-1]
            return IndicatorSet(timeframe=timeframe, close=float(df["Close"].iloc[-1]))

        fetches = []
        monkeypatch.setattr(
            module, "fetch_multi_timeframe",
            lambda **kwargs: fetches.append(kwargs) or data,
        )
        monkeypatch.setattr(module, "calculate_indicators", fake_indicators)
        monkeypatch.setattr(module, "route_signal", lambda indicators, config: _decision())
        monkeypatch.setattr(scheduler, "_reconcile_pending_orders", lambda: None)
        executed = []
        monkeypatch.setattr(scheduler, "execute_decision", lambda decision: executed.append(decision))

        scheduler.run_cycle()
        scheduler.run_cycle()
        assert len(executed) == 1
        assert len(fetches) == 1
        completed_primary = now - pd.Timedelta(hours=4)
        assert seen["4h"] == completed_primary
        assert seen["1h"] == now - pd.Timedelta(hours=1)
        expected_daily = now - pd.Timedelta(days=1)
        assert seen["1d"] == expected_daily.floor("D")

        restored = SharedLiveState(state_path)
        assert restored.last_analysis_bars[cfg.trading.symbol] == str(completed_primary)

    def test_marginal_signal_executes_deterministically(self, monkeypatch, tmp_path):
        # MARGINAL entries are traded directly (backtest parity) — this is a pure
        # technical-signal bot with no LLM gate on marginal setups.
        monkeypatch.chdir(tmp_path)
        cfg = _config()
        sched = TradingScheduler(cfg)
        decision = _decision()
        decision.signal_strength = SignalStrength.MARGINAL
        decision.scoring_result.signal_strength = SignalStrength.MARGINAL
        called = []
        monkeypatch.setattr(sched, "_execute_trade", lambda value: called.append(value))

        sched.execute_decision(decision)
        assert called == [decision]

    def test_size_from_risk_not_hardcoded(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)  # keep logs/ out of the repo
        sched = TradingScheduler(_config())
        monkeypatch.setattr(sched.exchange, "get_available_balance", lambda *a, **k: 5000.0)
        monkeypatch.setattr(sched.exchange, "get_positions", lambda *a, **k: [])

        captured = {}

        def fake_place_order(symbol, side, size, targets, leverage, client_oid=None, **kwargs):
            captured["size"] = size
            captured["client_oid"] = client_oid
            from llm_trading_bot.exchange import OrderResult
            return OrderResult(
                order_id="x", symbol=symbol, side=side, size=size, price=None,
                stop_loss=targets.stop_loss, take_profit_1=targets.take_profit_1,
                take_profit_2=targets.take_profit_2, status="submitted",
                timestamp="t", raw_response={},
            )

        monkeypatch.setattr(sched.exchange, "place_order", fake_place_order)

        sched._execute_trade(_decision())

        # margin = 5000 * min(0.02, 0.66) = 100; notional = 100 * 10 = 1000; size = 1000/50000
        assert captured["size"] == pytest.approx(1000 / 50000)
        assert captured["size"] != 0.001
        assert captured["client_oid"].startswith("llt-")

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

    def test_exchange_wide_exposure_cap_scales_new_order(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        cfg = _config()
        cfg.position_sizing.global_max_margin_pct = .01
        sched = TradingScheduler(cfg)
        monkeypatch.setattr(sched.exchange, "get_available_balance", lambda *a, **k: 5000.0)
        monkeypatch.setattr(sched.exchange, "get_account_equity", lambda *a, **k: 5000.0)

        existing = Position(
            symbol="ETH-USDT", side="long", size=.1, entry_price=4000,
            unrealized_pnl=0, leverage=10, margin_mode="crossed",
            timestamp="t", margin_size=40,
        )
        monkeypatch.setattr(
            sched.exchange, "get_positions",
            lambda symbol=None: [] if symbol else [existing],
        )
        monkeypatch.setattr(sched.exchange, "get_pending_orders", lambda: [])
        captured = {}
        monkeypatch.setattr(
            sched.exchange, "place_order",
            lambda symbol, side, size, targets, leverage, client_oid=None, **kwargs:
            captured.update(size=size, client_oid=client_oid)
            or type("R", (), {
                "order_id": "x", "price": None, "size": size,
                "stop_loss": targets.stop_loss,
                "take_profit_1": targets.take_profit_1,
                "take_profit_2": targets.take_profit_2,
            })(),
        )

        sched._execute_trade(_decision())

        # Equity cap is $50 margin; $40 is already committed, so only $10 remains.
        assert captured["size"] == pytest.approx((10 * 10) / 50000)


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
            "stop_plan_id": "plan-1",
            "last_trail_bar": str(
                pd.Timestamp.now(tz="UTC").floor("4h") - pd.Timedelta(hours=12)
            ),
        }

        moved = {}
        monkeypatch.setattr(
            sched.exchange, "modify_stop_loss",
            lambda symbol, hold_side, size, new_sl, plan_order_id=None:
            moved.update(new_sl=new_sl, plan_order_id=plan_order_id) or {"code": "00000"},
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
        assert moved["plan_order_id"] == "plan-1"
        assert sched._tracked_trades["BTC/USDT:USDT"]["current_sl"] == pytest.approx(50750.0)

        # a second tick inside the same bar must not ratchet again
        moved.clear()
        sched._maybe_trail_stop(pos)
        assert moved == {}


def test_stale_analysis_bar_skips_entry(monkeypatch, tmp_path):
    """An entry whose analysis bar closed far in the past (e.g. a cold start on a
    stale bar) is skipped before any sizing or exchange call."""
    sched = TradingScheduler(_config(), log_dir=tmp_path)
    sched._candidate_analysis_bar = "2020-01-01 00:00:00+00:00"
    logged = []
    monkeypatch.setattr(sched, "_log_decision", lambda rec: logged.append(rec))
    reached_sizing = []
    monkeypatch.setattr(
        sched.exchange, "get_available_balance",
        lambda *a, **k: reached_sizing.append(1) or 100.0,
    )
    sched._execute_trade(_decision())
    assert any(r["action"] == "SKIP_STALE_BAR" for r in logged)
    assert reached_sizing == []


def test_recent_analysis_bar_does_not_skip(monkeypatch, tmp_path):
    """A just-closed analysis bar passes the staleness guard and proceeds to sizing."""
    import llm_trading_bot.scheduler as module
    sched = TradingScheduler(_config(), log_dir=tmp_path)
    sched._candidate_analysis_bar = str(
        module.latest_completed_bar_open(sched.config.trading.primary_timeframe)
    )

    class _Sentinel(Exception):
        pass

    def _boom(*a, **k):
        raise _Sentinel()

    monkeypatch.setattr(sched.exchange, "get_available_balance", _boom)
    with pytest.raises(_Sentinel):
        sched._execute_trade(_decision())
