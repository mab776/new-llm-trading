"""Shared live orchestration, locking, and restart-state tests."""

from __future__ import annotations

import fcntl
import json

import pytest

from llm_trading_bot.config import AppConfig
from llm_trading_bot.exchange import Position
from llm_trading_bot.live_state import SharedLiveState
from llm_trading_bot.orchestrator import SharedTradingOrchestrator
from llm_trading_bot.scheduler import TradingScheduler


def _config(symbol: str) -> AppConfig:
    config = AppConfig()
    config.trading.symbol = symbol
    return config


def test_shared_state_survives_restart(tmp_path) -> None:
    path = tmp_path / "shared.json"
    state = SharedLiveState(path)
    state.update_peak(1234.5)
    state.pending_orders["o1"] = {"symbol": "BTC-USDT"}
    state.tracked_trades["BTC-USDT"] = {
        "direction": "LONG", "entry": 100, "current_sl": 95,
    }
    state.last_analysis_bars["BTC-USDT"] = "2026-07-13 08:00:00+00:00"
    state.save()

    restored = SharedLiveState(path)
    assert restored.peak_balance == 1234.5
    assert restored.pending_orders == state.pending_orders
    assert restored.tracked_trades == state.tracked_trades
    assert restored.last_analysis_bars == state.last_analysis_bars


def test_orchestrator_gives_every_symbol_one_state_and_lock(tmp_path) -> None:
    orchestrator = SharedTradingOrchestrator(
        [_config("BTC-USDT"), _config("ETH-USDT")], log_dir=tmp_path,
    )
    btc, eth = orchestrator.schedulers
    assert btc._live_state is eth._live_state is orchestrator.state
    assert btc._execution_lock is eth._execution_lock is orchestrator.state.lock
    btc._pending_orders["btc"] = {"symbol": "BTC-USDT"}
    assert "btc" in eth._pending_orders


def test_orchestrator_rejects_duplicate_symbol(tmp_path) -> None:
    with pytest.raises(ValueError, match="unique"):
        SharedTradingOrchestrator(
            [_config("BTC-USDT"), _config("BTC-USDT")], log_dir=tmp_path,
        )


def test_process_lock_rejects_second_orchestrator(tmp_path) -> None:
    first = SharedTradingOrchestrator([_config("BTC-USDT")], log_dir=tmp_path)
    second = SharedTradingOrchestrator([_config("ETH-USDT")], log_dir=tmp_path)
    first._acquire_process_lock()
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            second._acquire_process_lock()
    finally:
        fcntl.flock(first._lock_handle.fileno(), fcntl.LOCK_UN)
        first._lock_handle.close()
        first._lock_handle = None


def test_reconciliation_is_symbol_local_with_shared_pending(tmp_path, monkeypatch) -> None:
    state = SharedLiveState(tmp_path / "state.json")
    state.pending_orders.update({
        "btc": {"symbol": "BTC-USDT", "direction": "LONG", "entry": 100,
                "stop_loss": 95, "expires_at": 99999999999},
        "eth": {"symbol": "ETH-USDT", "direction": "LONG", "entry": 50,
                "stop_loss": 45, "expires_at": 99999999999},
    })
    scheduler = TradingScheduler(
        _config("BTC-USDT"), shared_state=state, log_dir=tmp_path,
    )
    monkeypatch.setattr(
        scheduler.exchange, "get_order_detail",
        lambda symbol, order_id: {"state": "filled", "priceAvg": "99"},
    )
    scheduler._reconcile_pending_orders()
    assert "btc" not in state.pending_orders
    assert "eth" in state.pending_orders
    assert "BTC-USDT" in state.tracked_trades


def test_position_check_persists_realized_account_peak(tmp_path, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    state = SharedLiveState(state_path)
    scheduler = TradingScheduler(
        _config("BTC-USDT"), shared_state=state, log_dir=tmp_path,
    )
    position = Position(
        symbol="ETH-USDT", side="long", size=1, entry_price=100,
        unrealized_pnl=50, leverage=10, margin_mode="crossed", timestamp="t",
    )
    monkeypatch.setattr(
        scheduler.exchange, "get_positions",
        lambda symbol=None: [] if symbol else [position],
    )
    monkeypatch.setattr(scheduler.exchange, "get_account_equity", lambda: 1050)
    scheduler.check_positions()

    assert SharedLiveState(state_path).peak_balance == 1000


def test_legacy_pending_file_is_migrated_only_once(tmp_path) -> None:
    legacy = tmp_path / "pending_orders.json"
    legacy.write_text(json.dumps({"old": {"symbol": "BTC-USDT"}}))
    first = TradingScheduler(_config("BTC-USDT"), log_dir=tmp_path)
    assert "old" in first._pending_orders
    first._pending_orders.clear()
    first._save_live_state()

    restarted = TradingScheduler(_config("BTC-USDT"), log_dir=tmp_path)
    assert restarted._pending_orders == {}
