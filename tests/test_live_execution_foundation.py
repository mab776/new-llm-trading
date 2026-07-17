"""Fail-closed live state, preflight, reconciliation, and per-lot exit lifecycle."""

from __future__ import annotations

import json
import time
from decimal import Decimal

import pytest

from llm_trading_bot.config import AppConfig, BitgetConfig, LeverageTier
from llm_trading_bot.exchange import (
    BitgetClient, PendingOrder, PlanOrder, Position, SafetyViolation,
)
from llm_trading_bot.live_state import LiveStateError, SharedLiveState
from llm_trading_bot.scheduler import TradingScheduler


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


def _plan(order_id: str, action: str, *, status: str = "live",
          filled_size: float = 0) -> PlanOrder:
    plan_type = "loss_plan" if action == "sl" else "profit_plan"
    trigger = {"sl": 95, "tp1": 110, "tp2": 120}[action]
    return PlanOrder(
        order_id=order_id, client_oid=f"client-{action}", symbol="BTCUSDT",
        plan_type=plan_type, side="long", size=1, trigger_price=trigger,
        status=status, updated_at_ms=123, filled_size=filled_size,
    )


def _lot() -> dict:
    return {
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


def test_corrupt_live_state_blocks_startup(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not-json")
    with pytest.raises(LiveStateError, match="Cannot trust"):
        SharedLiveState(path)


def test_v2_symbol_state_migrates_to_unreconciled_lot(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "version": 2,
        "tracked_trades": {
            "BTC-USDT": {"direction": "LONG", "entry": 100, "current_sl": 95}
        },
    }))
    state = SharedLiveState(path)
    assert "BTC-USDT" not in state.lots
    lot = next(iter(state.lots.values()))
    assert lot["symbol"] == "BTC-USDT"
    assert lot["lifecycle"] == "unreconciled"
    assert lot["protection_verified"] is False


def test_preflight_rejects_missing_credentials() -> None:
    with pytest.raises(SafetyViolation, match="requires explicit Bitget credentials"):
        BitgetClient(BitgetConfig()).preflight("BTCUSDT")


def test_preflight_verifies_clock_and_account_modes(monkeypatch) -> None:
    client = BitgetClient(BitgetConfig(
        api_key="k", api_secret="s", passphrase="p",
        position_mode="one_way", margin_mode="crossed",
    ))
    now_ms = int(time.time() * 1000)
    monkeypatch.setattr(client, "get_single_account", lambda symbol: {
        "requestTime": now_ms,
        "data": {"posMode": "one_way_mode", "marginMode": "crossed"},
    })
    monkeypatch.setattr(
        client, "get_contract_spec",
        lambda symbol: type("Spec", (), {"symbol": "BTCUSDT"})(),
    )
    assert client.preflight("BTC-USDT")["position_mode"] == "one_way_mode"

    monkeypatch.setattr(client, "get_single_account", lambda symbol: {
        "requestTime": now_ms,
        "data": {"posMode": "hedge_mode", "marginMode": "crossed"},
    })
    with pytest.raises(SafetyViolation, match="position mode"):
        client.preflight("BTC-USDT")


def test_fill_protection_creates_exact_per_lot_plans(monkeypatch, tmp_path) -> None:
    scheduler = TradingScheduler(_config(), log_dir=tmp_path)
    scheduler.exchange._dry_run = False
    scheduler._tracked_trades["llt-entry"] = _lot() | {
        "lifecycle": "protecting", "protection_verified": False,
        "plan_ids": {}, "plan_client_oids": {},
    }
    monkeypatch.setattr(
        scheduler.exchange, "split_size",
        lambda symbol, size, pct: (Decimal("0.7"), Decimal("0.3")),
    )
    placed = []

    def place(symbol, side, size, trigger, plan_type, client_oid):
        action = "sl" if plan_type == "loss_plan" else (
            "tp1" if trigger == 110 else "tp2"
        )
        plan = _plan(f"{action}-new", action)
        plan = PlanOrder(**{**plan.__dict__, "client_oid": client_oid, "size": size})
        placed.append(plan)
        return plan

    reads = iter([placed])
    monkeypatch.setattr(scheduler.exchange, "place_tpsl_order", place)
    monkeypatch.setattr(scheduler.exchange, "get_tpsl_orders", lambda *a, **k: next(reads))
    monkeypatch.setattr(scheduler.exchange, "cancel_tpsl_order", lambda *a, **k: {})

    assert scheduler._ensure_lot_protection("llt-entry", active_plans=[])
    assert [(p.plan_type, p.size) for p in placed] == [
        ("loss_plan", 1.0), ("profit_plan", 0.7), ("profit_plan", 0.3),
    ]
    assert scheduler._tracked_trades["llt-entry"]["protection_verified"]


def test_preset_cleanup_never_cancels_unattributed_manual_plan(monkeypatch, tmp_path) -> None:
    scheduler = TradingScheduler(_config(), log_dir=tmp_path)
    scheduler.exchange._dry_run = False
    lot = _lot() | {"filled_at_ms": 1_000_000}
    scheduler._tracked_trades["llt-entry"] = lot
    manual = PlanOrder(
        "manual", "operator-plan", "BTCUSDT", "loss_plan", "long",
        1, 95, "live", created_at_ms=1_000_000,
    )
    preset = PlanOrder(
        "preset", "BITGET#123", "BTCUSDT", "loss_plan", "long",
        1, 95, "live", created_at_ms=1_000_000,
    )
    monkeypatch.setattr(
        scheduler.exchange, "get_tpsl_orders", lambda *a, **k: [manual, preset],
    )
    cancelled = []
    monkeypatch.setattr(
        scheduler.exchange, "cancel_tpsl_order",
        lambda symbol, order_id, plan_type: cancelled.append(order_id) or {},
    )
    scheduler._cancel_replaced_presets("BTC-USDT")
    assert cancelled == ["preset"]


def test_tp1_execution_moves_lot_stop_to_break_even(monkeypatch, tmp_path) -> None:
    scheduler = TradingScheduler(_config(), log_dir=tmp_path)
    scheduler.exchange._dry_run = False
    scheduler._tracked_trades["llt-entry"] = _lot()
    active = [_plan("sl-1", "sl"), _plan("tp2-1", "tp2")]
    history_rows = [_plan("tp1-1", "tp1", status="executed", filled_size=0.7)]
    # Keep the current/history distinction explicit and readable.
    monkeypatch.setattr(
        scheduler.exchange, "get_tpsl_orders",
        lambda symbol, history=False, **kwargs: history_rows if history else active,
    )
    modified = []
    monkeypatch.setattr(
        scheduler.exchange, "modify_stop_loss",
        lambda symbol, side, size, price, **kwargs:
        modified.append(("sl", size, price, kwargs)) or {},
    )
    monkeypatch.setattr(
        scheduler.exchange, "modify_tpsl_order",
        lambda symbol, order_id, side, size, price, **kwargs:
        modified.append(("tp2", size, price, kwargs)) or {},
    )

    scheduler._reconcile_lot_lifecycle()
    lot = scheduler._tracked_trades["llt-entry"]
    assert lot["lifecycle"] == "remainder"
    assert lot["remaining_size"] == pytest.approx(0.3)
    assert lot["current_sl"] == 100
    assert modified[0][:3] == ("sl", pytest.approx(0.3), 100.0)
    assert modified[0][3]["position_level"] is False
    assert modified[1][:3] == ("tp2", pytest.approx(0.3), 120.0)


def test_startup_rejects_unknown_exchange_order(monkeypatch, tmp_path) -> None:
    scheduler = TradingScheduler(_config(), log_dir=tmp_path)
    scheduler.exchange._dry_run = False
    monkeypatch.setattr(scheduler.exchange, "preflight", lambda symbol: {})
    monkeypatch.setattr(scheduler.exchange, "get_pending_order_rows", lambda symbol=None: [])
    monkeypatch.setattr(scheduler.exchange, "get_pending_orders", lambda symbol=None: [
        PendingOrder("foreign", "BTCUSDT", "long", 1, 0, 100, 10, "manual-order")
    ])
    monkeypatch.setattr(scheduler.exchange, "get_positions", lambda symbol=None: [])
    monkeypatch.setattr(scheduler.exchange, "get_tpsl_orders", lambda *a, **k: [])
    with pytest.raises(SafetyViolation, match="Unknown exchange order"):
        scheduler.reconcile_startup()


def test_startup_rejects_unexplained_position(monkeypatch, tmp_path) -> None:
    scheduler = TradingScheduler(_config(), log_dir=tmp_path)
    scheduler.exchange._dry_run = False
    monkeypatch.setattr(scheduler.exchange, "preflight", lambda symbol: {})
    monkeypatch.setattr(scheduler.exchange, "get_pending_order_rows", lambda symbol=None: [])
    monkeypatch.setattr(scheduler.exchange, "get_pending_orders", lambda symbol=None: [])
    monkeypatch.setattr(scheduler.exchange, "get_positions", lambda symbol=None: [
        Position("BTCUSDT", "long", 1, 100, 0, 10, "crossed", "t")
    ])
    monkeypatch.setattr(scheduler.exchange, "get_order_history", lambda symbol=None: [])
    monkeypatch.setattr(scheduler.exchange, "get_tpsl_orders", lambda *a, **k: [])
    with pytest.raises(SafetyViolation, match="Unexplained"):
        scheduler.reconcile_startup()


def test_empty_account_startup_reconciliation_passes(monkeypatch, tmp_path) -> None:
    scheduler = TradingScheduler(_config(), log_dir=tmp_path)
    scheduler.exchange._dry_run = False
    monkeypatch.setattr(scheduler.exchange, "preflight", lambda symbol: {
        "symbol": "BTCUSDT", "position_mode": "one_way_mode",
        "margin_mode": "crossed", "clock_drift_ms": 2,
    })
    monkeypatch.setattr(scheduler.exchange, "get_pending_orders", lambda symbol=None: [])
    monkeypatch.setattr(scheduler.exchange, "get_pending_order_rows", lambda symbol=None: [])
    monkeypatch.setattr(scheduler.exchange, "get_positions", lambda symbol=None: [])
    monkeypatch.setattr(scheduler.exchange, "get_order_history", lambda symbol=None: [])
    monkeypatch.setattr(scheduler.exchange, "get_tpsl_orders", lambda *a, **k: [])
    scheduler.reconcile_startup()
    assert scheduler._startup_reconciled


def test_startup_adopts_bot_order_accepted_before_local_save(monkeypatch, tmp_path) -> None:
    scheduler = TradingScheduler(_config(), log_dir=tmp_path)
    scheduler.exchange._dry_run = False
    pending = PendingOrder(
        "entry-1", "BTCUSDT", "long", 1, 0, 100, 10, "llt-entry",
    )
    detail = {
        "orderId": "entry-1", "clientOid": "llt-entry", "symbol": "BTCUSDT",
        "state": "live", "side": "buy", "size": "1", "price": "100",
        "presetStopLossPrice": "95", "presetStopSurplusPrice": "115",
        "orderType": "limit", "leverage": "10", "cTime": "1000",
    }
    monkeypatch.setattr(scheduler.exchange, "preflight", lambda symbol: {
        "symbol": "BTCUSDT", "position_mode": "one_way_mode",
        "margin_mode": "crossed", "clock_drift_ms": 1,
    })
    monkeypatch.setattr(
        scheduler.exchange, "get_pending_order_rows",
        lambda symbol=None: [{
            "orderId": "entry-1", "clientOid": "llt-entry",
            "symbol": "BTCUSDT", "side": "buy", "tradeSide": "open",
        }],
    )
    monkeypatch.setattr(
        scheduler.exchange, "get_pending_orders", lambda symbol=None: [pending],
    )
    monkeypatch.setattr(scheduler.exchange, "get_positions", lambda symbol=None: [])
    monkeypatch.setattr(scheduler.exchange, "get_order_detail", lambda *a, **k: detail)
    monkeypatch.setattr(scheduler.exchange, "get_order_history", lambda *a, **k: [])
    monkeypatch.setattr(scheduler.exchange, "get_tpsl_orders", lambda *a, **k: [])

    scheduler.reconcile_startup()
    assert scheduler._pending_orders["entry-1"]["client_oid"] == "llt-entry"
    assert scheduler._startup_reconciled


def test_preset_cleanup_cancels_numeric_coid_preset(monkeypatch, tmp_path) -> None:
    """Bitget attaches entry presets with an all-numeric auto clientOid, not a
    BITGET# one. They must still be cancelled once the per-lot bracket exists, while
    a named operator plan with the same trigger/size is left alone."""
    scheduler = TradingScheduler(_config(), log_dir=tmp_path)
    scheduler.exchange._dry_run = False
    scheduler._tracked_trades["llt-entry"] = _lot() | {"filled_at_ms": 1_000_000}
    manual = PlanOrder(
        "manual", "operator-plan", "BTCUSDT", "loss_plan", "long",
        1, 95, "live", created_at_ms=1_000_000,
    )
    numeric_preset = PlanOrder(
        "preset", "1461740694906347522", "BTCUSDT", "loss_plan", "long",
        1, 95, "live", created_at_ms=1_000_000,
    )
    monkeypatch.setattr(
        scheduler.exchange, "get_tpsl_orders", lambda *a, **k: [manual, numeric_preset],
    )
    cancelled = []
    monkeypatch.setattr(
        scheduler.exchange, "cancel_tpsl_order",
        lambda symbol, order_id, plan_type: cancelled.append(order_id) or {},
    )
    scheduler._cancel_replaced_presets("BTC-USDT")
    assert cancelled == ["preset"]


def test_post_only_instant_cancel_logged_distinctly(monkeypatch, tmp_path) -> None:
    """An exchange-side cancel with zero fill (post-only would-cross rejection) must
    log MAKER_POST_ONLY_CANCELLED, not be silently dropped like a bar expiry."""
    scheduler = TradingScheduler(_config(), log_dir=tmp_path)
    scheduler.exchange._dry_run = False
    scheduler._pending_orders["ord-1"] = {
        "symbol": "BTC-USDT", "direction": "LONG", "entry_mode": "maker",
        "expires_at": time.time() + 3600,
    }
    monkeypatch.setattr(
        scheduler.exchange, "get_order_detail",
        lambda symbol, oid: {"state": "canceled", "baseVolume": "0"},
    )
    logged = []
    monkeypatch.setattr(scheduler, "_log_decision", lambda rec: logged.append(rec))
    scheduler._reconcile_pending_orders()
    assert scheduler._pending_orders == {}
    assert [r["action"] for r in logged] == ["MAKER_POST_ONLY_CANCELLED"]


def test_startup_rejects_leverage_mismatch(monkeypatch, tmp_path) -> None:
    """Account leverage that drifted from the active tier fails startup closed."""
    scheduler = TradingScheduler(_config(), log_dir=tmp_path)  # active tier leverage=10
    scheduler.exchange._dry_run = False
    monkeypatch.setattr(scheduler.exchange, "preflight", lambda symbol: {
        "symbol": "BTCUSDT", "position_mode": "one_way_mode",
        "margin_mode": "isolated", "clock_drift_ms": 1,
        "leverage_long": 25, "leverage_short": 25,
    })
    with pytest.raises(SafetyViolation, match="leverage"):
        scheduler.reconcile_startup()


def test_preflight_surfaces_isolated_leverage(monkeypatch) -> None:
    client = BitgetClient(BitgetConfig(
        api_key="k", api_secret="s", passphrase="p",
        position_mode="one_way", margin_mode="isolated",
    ))
    now_ms = int(time.time() * 1000)
    monkeypatch.setattr(client, "get_single_account", lambda symbol: {
        "requestTime": now_ms,
        "data": {
            "posMode": "one_way_mode", "marginMode": "isolated",
            "isolatedLongLever": "25", "isolatedShortLever": "25",
        },
    })
    monkeypatch.setattr(
        client, "get_contract_spec",
        lambda symbol: type("Spec", (), {"symbol": "BTCUSDT"})(),
    )
    out = client.preflight("BTC-USDT")
    assert out["leverage_long"] == 25 and out["leverage_short"] == 25
