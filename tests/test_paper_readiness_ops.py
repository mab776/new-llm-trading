"""Paper-readiness operations: daily log rotation/retention, structured records,
live cooldown/loss-penalty parity, max_position_usd in simulators, engine
slippage/liquidation, the account-scoped process lock, and max-holding expiry."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest

from llm_trading_bot.backtesting import BacktestEngine
from llm_trading_bot.config import AppConfig, LeverageTier
from llm_trading_bot.live_state import SharedLiveState
from llm_trading_bot.portfolio import Portfolio
from llm_trading_bot.process_lock import (
    AccountLockError, acquire_account_lock, release_account_lock,
)
from llm_trading_bot.routing import RoutingDecision
from llm_trading_bot.scheduler import TradingScheduler
from llm_trading_bot.scoring import (
    Direction, ScoringResult, SignalStrength, TradeTargets,
)


def _config() -> AppConfig:
    cfg = AppConfig()
    cfg.trading.symbol = "BTC-USDT"
    cfg.trading.leverage_tiers = {
        "x": LeverageTier(
            leverage=10, strong_threshold=20, marginal_threshold_low=10,
            tp1_rr=2, tp2_rr=3, tp1_exit_pct=0.7,
        )
    }
    cfg.trading.active_tier = "x"
    return cfg


def _lot(**overrides) -> dict:
    lot = {
        "symbol": "BTC-USDT", "direction": "LONG", "entry_order_id": "entry-1",
        "client_oid": "llt-entry", "lifecycle": "protected",
        "original_size": 1.0, "remaining_size": 1.0, "entry": 100.0,
        "entry_fee": 0.01, "filled_at_ms": 1, "stop_loss": 95.0,
        "take_profit_1": 110.0, "take_profit_2": 120.0,
        "tp1_exit_pct": 0.7, "tp1_size": 0.7, "tp2_size": 0.3,
        "current_sl": 95.0, "protection_verified": True,
        "plan_ids": {"sl": "sl-1", "tp1": "tp1-1", "tp2": "tp2-1"},
        "plan_client_oids": {
            "sl": "client-sl", "tp1": "client-tp1", "tp2": "client-tp2",
        },
        "last_trail_bar": "2026-07-13 08:00:00+00:00",
    }
    lot.update(overrides)
    return lot


def _decision(score: float = 40.0) -> RoutingDecision:
    targets = TradeTargets(
        entry=50000, stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
        risk_amount=1000, reward_1=2000, reward_2=4000, direction=Direction.BULLISH,
    )
    sr = ScoringResult(
        direction=Direction.BULLISH, confidence=80,
        signal_strength=SignalStrength.STRONG, raw_score=score,
        category_scores=[], indicators=None, reasons=[], filter_failures=[],
    )
    return RoutingDecision(
        signal_strength=SignalStrength.STRONG, scoring_result=sr, targets=targets,
    )


# ---------------------------------------------------------------------------
# Daily log files + retention
# ---------------------------------------------------------------------------

class TestDailyLogs:
    def test_log_writes_daily_dated_files(self, tmp_path) -> None:
        scheduler = TradingScheduler(_config(), log_dir=tmp_path)
        scheduler._log("hello")
        scheduler._log_decision({"action": "WAIT", "reason": "test"})

        day = datetime.now().astimezone().strftime("%Y-%m-%d")  # LOCAL day
        log_file = tmp_path / f"trading-{day}.log"
        decisions_file = tmp_path / f"decisions-{day}.jsonl"
        assert log_file.exists() and "hello" in log_file.read_text()
        record = json.loads(decisions_file.read_text().splitlines()[-1])
        assert record["action"] == "WAIT"
        assert record["symbol"] == "BTC-USDT"
        assert record["timestamp"].startswith(day)
        # Local timestamp keeps its UTC offset so records stay unambiguous.
        assert datetime.fromisoformat(record["timestamp"]).tzinfo is not None

    def test_retention_prunes_only_old_dated_files(self, tmp_path) -> None:
        old_log = tmp_path / "trading-2020-01-01.log"
        old_decisions = tmp_path / "decisions-2020-01-01.jsonl"
        unrelated = tmp_path / "trading-notes.log"
        for path in (old_log, old_decisions, unrelated):
            path.write_text("x")

        scheduler = TradingScheduler(_config(), log_dir=tmp_path)
        scheduler._log("tick")

        assert not old_log.exists()
        assert not old_decisions.exists()
        assert unrelated.exists()  # unparsable date suffix is never deleted
        day = datetime.now().astimezone().strftime("%Y-%m-%d")
        assert (tmp_path / f"trading-{day}.log").exists()

    def test_retention_window_is_configurable(self, tmp_path) -> None:
        cfg = _config()
        assert cfg.scheduling.log_retention_days == 90  # shipped default
        cfg.scheduling.log_retention_days = 3650  # ~10 years
        old_log = tmp_path / "trading-2020-01-01.log"
        old_log.write_text("x")
        TradingScheduler(cfg, log_dir=tmp_path)
        assert old_log.exists()


# ---------------------------------------------------------------------------
# Live cooldown + consecutive-loss penalty (backtest parity)
# ---------------------------------------------------------------------------

class TestLiveRiskCounters:
    def test_sl_loss_arms_cooldown_and_blocks_entry(self, tmp_path) -> None:
        cfg = _config()
        cfg.risk_management.cooldown_candles_after_sl = 2
        scheduler = TradingScheduler(cfg, log_dir=tmp_path)

        net = scheduler._apply_close_outcome(_lot(), 95.0, "sl")
        assert net < 0
        counters = scheduler._risk_state()
        assert counters["cooldown_remaining"] == 2
        assert counters["consecutive_losses"] == 1

        executed = []
        scheduler._execute_trade = lambda decision: executed.append(decision)
        scheduler.execute_decision(_decision())
        assert executed == []
        assert any(d["action"] == "COOLDOWN_SKIP" for d in scheduler.decision_log)

    def test_win_resets_counters_and_no_cooldown_for_tp2(self, tmp_path) -> None:
        scheduler = TradingScheduler(_config(), log_dir=tmp_path)
        counters = scheduler._risk_state()
        counters["consecutive_losses"] = 3
        counters["candles_since_last_loss"] = 0

        net = scheduler._apply_close_outcome(_lot(), 120.0, "tp2")
        assert net > 0
        counters = scheduler._risk_state()
        assert counters["consecutive_losses"] == 0
        assert counters["cooldown_remaining"] == 0
        assert counters["candles_since_last_loss"] == 999

    def test_tick_decrements_once_per_completed_bar_and_catches_up(
        self, tmp_path,
    ) -> None:
        scheduler = TradingScheduler(_config(), log_dir=tmp_path)
        counters = scheduler._risk_state()
        counters["cooldown_remaining"] = 3

        bar0 = pd.Timestamp("2026-07-14 00:00:00+00:00")
        scheduler._tick_risk_counters(str(bar0))
        assert scheduler._risk_state()["cooldown_remaining"] == 2
        # Same bar twice must not double-tick.
        scheduler._tick_risk_counters(str(bar0))
        assert scheduler._risk_state()["cooldown_remaining"] == 2
        # Restart after downtime: two 4h bars elapsed -> two ticks at once.
        scheduler._tick_risk_counters(str(bar0 + pd.Timedelta(hours=8)))
        assert scheduler._risk_state()["cooldown_remaining"] == 0

    def test_loss_penalty_matches_engine_semantics(self, tmp_path) -> None:
        cfg = _config()
        cfg.risk_management.consecutive_loss_penalty = 5.0
        cfg.risk_management.max_consecutive_loss_penalty = 20.0
        cfg.risk_management.loss_penalty_decay_candles = 10
        scheduler = TradingScheduler(cfg, log_dir=tmp_path)
        counters = scheduler._risk_state()

        counters.update(consecutive_losses=2, candles_since_last_loss=0)
        assert scheduler._loss_penalty() == pytest.approx(10.0)
        counters.update(consecutive_losses=10)  # capped
        assert scheduler._loss_penalty() == pytest.approx(20.0)
        counters.update(consecutive_losses=2, candles_since_last_loss=15)
        assert scheduler._loss_penalty() == pytest.approx(10.0 * 0.5)  # decayed
        counters.update(candles_since_last_loss=999)
        assert scheduler._loss_penalty() == pytest.approx(0.0)

    def test_break_even_lot_after_tp1_counts_as_win(self, tmp_path) -> None:
        scheduler = TradingScheduler(_config(), log_dir=tmp_path)
        lot = _lot(
            lifecycle="remainder", remaining_size=0.3, current_sl=100.0,
            tp1_fill_size=0.7,
        )
        # 70% closed at TP1 (110) + remainder at break-even (100): a clear win.
        net = scheduler._apply_close_outcome(lot, 100.0, "sl")
        assert net > 0
        assert scheduler._risk_state()["cooldown_remaining"] == 0

    def test_counters_survive_restart(self, tmp_path) -> None:
        state_path = tmp_path / "state.json"
        state = SharedLiveState(state_path)
        scheduler = TradingScheduler(_config(), shared_state=state, log_dir=tmp_path)
        scheduler._apply_close_outcome(_lot(), 95.0, "sl")

        restored = SharedLiveState(state_path)
        assert restored.risk_counters["BTC-USDT"]["consecutive_losses"] == 1
        assert restored.risk_counters["BTC-USDT"]["cooldown_remaining"] > 0

    def test_v3_state_loads_with_empty_counters(self, tmp_path) -> None:
        path = tmp_path / "state.json"
        path.write_text(json.dumps({
            "version": 3, "peak_balance": 10, "pending_orders": {},
            "lots": {}, "last_analysis_bars": {},
        }))
        state = SharedLiveState(path)
        assert state.risk_counters == {}
        assert state.peak_balance == 10


# ---------------------------------------------------------------------------
# max_position_usd parity in the simulators
# ---------------------------------------------------------------------------

class TestMaxPositionUsd:
    def test_portfolio_caps_margin_per_trade(self) -> None:
        port = Portfolio(initial_balance=10000)
        capped = port.open_trade(
            direction="LONG", entry_price=100.0, entry_time="t",
            stop_loss=95.0, take_profit_1=110.0, take_profit_2=120.0,
            leverage=10, risk_pct=0.02, max_margin_usd=100.0,
        )
        # margin = min(10000 * 0.02, 100) = 100 -> notional 1000 -> size 10
        assert capped.size == pytest.approx(10.0)

        uncapped = port.open_trade(
            direction="LONG", entry_price=100.0, entry_time="t",
            stop_loss=95.0, take_profit_1=110.0, take_profit_2=120.0,
            leverage=10, risk_pct=0.02,
        )
        assert uncapped.size > capped.size  # None preserves legacy sizing

    def test_engine_passes_config_cap_to_fills(self) -> None:
        cfg = _config()
        cfg.position_sizing.max_position_usd = 123.0
        engine = BacktestEngine(cfg)
        assert engine.max_margin_usd == 123.0


# ---------------------------------------------------------------------------
# Engine slippage + isolated-margin liquidation (fastbt parity)
# ---------------------------------------------------------------------------

class TestEngineExecutionRealism:
    def _engine(self, slippage: float, liquidation: bool,
                leverage: int = 10) -> BacktestEngine:
        cfg = _config()
        cfg.trading.leverage_tiers["x"].leverage = leverage
        cfg.backtesting.slippage_pct = slippage
        cfg.backtesting.model_liquidation = liquidation
        cfg.backtesting.enable_partial_exits = True
        return BacktestEngine(cfg)

    def test_sl_fill_pays_slippage(self) -> None:
        engine = self._engine(slippage=0.001, liquidation=False)
        trade = engine.portfolio.open_trade(
            direction="LONG", entry_price=100.0, entry_time="t",
            stop_loss=95.0, take_profit_1=110.0, take_profit_2=120.0,
            leverage=10, risk_pct=0.02,
        )
        engine._check_exits(trade, bar_high=101.0, bar_low=94.0,
                            bar_close=96.0, bar_time="t1")
        assert trade.exit_reason == "sl"
        assert trade.exit_price == pytest.approx(95.0 * 0.999)

    def test_stop_beyond_liquidation_exits_at_liquidation(self) -> None:
        engine = self._engine(slippage=0.0, liquidation=True, leverage=25)
        # liq distance = 1/25 - 0.005 = 3.5% -> LONG liq at 96.5
        trade = engine.portfolio.open_trade(
            direction="LONG", entry_price=100.0, entry_time="t",
            stop_loss=90.0, take_profit_1=110.0, take_profit_2=120.0,
            leverage=25, risk_pct=0.02,
        )
        engine._check_exits(trade, bar_high=100.0, bar_low=96.0,
                            bar_close=97.0, bar_time="t1")
        assert trade.exit_reason == "sl"
        assert trade.exit_price == pytest.approx(96.5)

    def test_tp_fills_stay_exact_with_slippage_enabled(self) -> None:
        engine = self._engine(slippage=0.001, liquidation=True)
        trade = engine.portfolio.open_trade(
            direction="LONG", entry_price=100.0, entry_time="t",
            stop_loss=95.0, take_profit_1=110.0, take_profit_2=120.0,
            leverage=10, risk_pct=0.02, tp1_exit_pct=0.5,
        )
        engine._check_exits(trade, bar_high=125.0, bar_low=99.0,
                            bar_close=124.0, bar_time="t1")
        assert trade.exit_reason == "tp2"
        assert trade.exit_price == pytest.approx(120.0)  # limit exit, no slip
        assert trade.partial_exits[0]["price"] == pytest.approx(110.0)

    def test_defaults_reproduce_legacy_engine(self) -> None:
        engine = self._engine(slippage=0.0, liquidation=False)
        trade = engine.portfolio.open_trade(
            direction="SHORT", entry_price=100.0, entry_time="t",
            stop_loss=105.0, take_profit_1=90.0, take_profit_2=80.0,
            leverage=10, risk_pct=0.02,
        )
        engine._check_exits(trade, bar_high=106.0, bar_low=99.0,
                            bar_close=104.0, bar_time="t1")
        assert trade.exit_price == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# Account-scoped process lock
# ---------------------------------------------------------------------------

class TestAccountLock:
    def test_second_acquire_on_same_account_is_rejected(self) -> None:
        cfg = _config()
        handle = acquire_account_lock(cfg.bitget)
        try:
            with pytest.raises(AccountLockError, match="already owns"):
                acquire_account_lock(cfg.bitget)
        finally:
            release_account_lock(handle)
        # After release the account is takeable again.
        handle = acquire_account_lock(cfg.bitget)
        release_account_lock(handle)

    def test_different_accounts_do_not_conflict(self) -> None:
        first = _config()
        second = _config()
        second.bitget.api_key = "another-account"
        h1 = acquire_account_lock(first.bitget)
        try:
            h2 = acquire_account_lock(second.bitget)
            release_account_lock(h2)
        finally:
            release_account_lock(h1)


# ---------------------------------------------------------------------------
# Max-holding expiry + heartbeat
# ---------------------------------------------------------------------------

class TestMaxHoldingAndHeartbeat:
    def test_max_holding_closes_expired_lot(self, monkeypatch, tmp_path) -> None:
        cfg = _config()
        cfg.risk_management.max_holding_hours = 4  # one 4h bar
        scheduler = TradingScheduler(cfg, log_dir=tmp_path)
        scheduler._tracked_trades["llt-old"] = _lot(filled_at_ms=1)  # 1970 — long expired

        closed = {}
        monkeypatch.setattr(
            scheduler.exchange, "close_position",
            lambda symbol, side, size, client_oid=None:
            closed.update(symbol=symbol, side=side, size=size)
            or {"data": {"orderId": "close-1"}},
        )
        scheduler._maybe_expire_lots()

        assert closed["side"] == "long" and closed["size"] == 1.0
        lot = scheduler._tracked_trades["llt-old"]
        assert lot["lifecycle"] == "closing"
        assert lot["close_reason"] == "time_expired"
        assert any(d["action"] == "TIME_EXPIRED_CLOSE" for d in scheduler.decision_log)

    def test_disabled_max_holding_is_inert(self, monkeypatch, tmp_path) -> None:
        scheduler = TradingScheduler(_config(), log_dir=tmp_path)  # default 0
        scheduler._tracked_trades["llt-old"] = _lot(filled_at_ms=1)
        monkeypatch.setattr(
            scheduler.exchange, "close_position",
            lambda *a, **k: pytest.fail("must not close when disabled"),
        )
        scheduler._maybe_expire_lots()
        assert scheduler._tracked_trades["llt-old"]["lifecycle"] == "protected"

    def test_check_positions_emits_heartbeat_record(self, monkeypatch, tmp_path) -> None:
        scheduler = TradingScheduler(_config(), log_dir=tmp_path)
        monkeypatch.setattr(
            scheduler.exchange, "get_positions", lambda symbol=None: [],
        )
        monkeypatch.setattr(
            scheduler.exchange, "get_account_equity", lambda *a, **k: 1234.5,
        )
        scheduler.check_positions()
        beats = [d for d in scheduler.decision_log if d["action"] == "HEARTBEAT"]
        assert len(beats) == 1
        assert beats[0]["equity"] == 1234.5
        assert beats[0]["symbol"] == "BTC-USDT"
        assert "disk_free_mb" in beats[0]
