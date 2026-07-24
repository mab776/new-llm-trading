"""
Scheduler — runs the trading bot on a schedule and manages positions.

Handles:
- Periodic market analysis
- Position monitoring and trailing stop updates
- Signal routing and trade execution
- Decision logging
"""

from __future__ import annotations

import json
import hashlib
import re
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import schedule

from llm_trading_bot.config import AppConfig
from llm_trading_bot.data import clear_cache, configure_cache, fetch_multi_timeframe
from llm_trading_bot.exchange import BitgetClient, ExchangeError, PlanOrder, SafetyViolation
from llm_trading_bot.exposure import (
    anti_martingale_multiplier, cap_risk_pct, outcome_streak,
)
from llm_trading_bot.live_state import SharedLiveState
from llm_trading_bot.process_lock import acquire_account_lock, release_account_lock
from llm_trading_bot.routing import RoutingDecision, route_signal
from llm_trading_bot.scoring import (
    Direction, SignalStrength, TradeTargets, calculate_indicators,
)
from llm_trading_bot.trailing import compute_trailing_stop
from llm_trading_bot.timeframes import (
    completed_market_snapshot, latest_completed_bar_open, timeframe_delta,
    timeframe_hours,
)


class TradingScheduler:
    """
    Main automation controller.

    Runs on a schedule:
    1. Fetch market data
    2. Score and route signal
    3. Execute trade (if applicable)
    4. Monitor existing positions
    """

    # A new entry is only valid shortly after its analysis bar closes. Beyond this
    # (e.g. a cold start that analyzed a bar which closed hours ago) a maker limit
    # priced off the stale bar crosses the book and is post-only-rejected, and a
    # market entry chases a moved price. Trailing/exits on open lots are unaffected.
    STALE_ENTRY_MAX_SECONDS = 1800

    def __init__(self, config: AppConfig, *,
                 shared_state: SharedLiveState | None = None,
                 log_dir: str | Path = "logs"):
        self.config = config
        self.exchange = BitgetClient(config.bitget)
        self.decision_log: list[dict] = []
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        safe_symbol = re.sub(r"[^A-Za-z0-9]+", "-", config.trading.symbol).strip("-")
        standalone_state_path = self._log_dir / f"live_state-{safe_symbol}.json"
        migrate_legacy = shared_state is None and not standalone_state_path.exists()
        self._live_state = shared_state or SharedLiveState(standalone_state_path)
        self._execution_lock = self._live_state.lock if shared_state else threading.RLock()

        # Per-symbol trade context for live trailing stops:
        # {symbol: {"direction": "LONG"|"SHORT", "entry": float, "current_sl": float}}
        self._tracked_trades = self._live_state.tracked_trades
        self._pending_orders = self._live_state.pending_orders
        self._candidate_analysis_bar: str | None = None
        self._startup_reconciled = False
        # Backtest-parity risk state refreshed at each completed-bar analysis.
        self._current_loss_penalty = 0.0
        # Daily log-file rotation: prune once per local day (and once at startup).
        self._pruned_log_day: str | None = None
        self._prune_old_logs(datetime.now().astimezone().strftime("%Y-%m-%d"))
        self._lock_handle = None
        # One-time compatibility migration from the pre-shared-state maker file.
        legacy = self._log_dir / "pending_orders.json"
        if migrate_legacy and not self._pending_orders and legacy.exists():
            migrated = self._load_pending_orders(legacy)
            self._pending_orders.update(migrated)
            if migrated:
                self._live_state.save()

        configure_cache(config.data_cache.ttl_seconds)

    @staticmethod
    def _load_pending_orders(path: Path) -> dict[str, dict]:
        try:
            if path.exists():
                data = json.loads(path.read_text())
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            pass
        return {}

    def _save_pending_orders(self) -> None:
        try:
            self._live_state.save()
        except OSError as e:
            self._log(f"Warning: could not persist pending orders: {e}")
            self._startup_reconciled = False
            if not self.exchange._dry_run:
                raise

    def _save_live_state(self) -> None:
        state = getattr(self, "_live_state", None)
        if state is not None:
            try:
                state.save()
            except OSError as e:
                self._log(f"Warning: could not persist live state: {e}")
                self._startup_reconciled = False
                if not self.exchange._dry_run:
                    raise

    @staticmethod
    def _timeframe_seconds(timeframe: str) -> int:
        return int(timeframe_delta(timeframe).total_seconds())

    @staticmethod
    def _same_symbol(left: str, right: str) -> bool:
        normalize = lambda value: re.sub(r"[^A-Za-z0-9]", "", value).upper()
        return normalize(left) == normalize(right)

    def _entry_client_oid(self, side: str) -> str:
        """Stable Bitget idempotency key for one symbol/bar/action/account."""
        material = "|".join((
            self.config.bitget.api_key,
            self.config.trading.symbol,
            self._candidate_analysis_bar or "manual",
            side,
            "entry",
        ))
        return "llt-" + hashlib.sha256(material.encode()).hexdigest()[:28]

    def _plan_client_oid(self, lot_id: str, action: str) -> str:
        material = "|".join((
            self.config.bitget.api_key, lot_id, action, "tpsl",
        ))
        return "llt-" + hashlib.sha256(material.encode()).hexdigest()[:28]

    def _lots_for_symbol(self, symbol: str, side: str | None = None) -> list[tuple[str, dict]]:
        lots = []
        for lot_id, lot in self._tracked_trades.items():
            lot_symbol = str(lot.get("symbol") or lot_id)
            if not self._same_symbol(lot_symbol, symbol):
                continue
            lot_side = "long" if lot.get("direction") == "LONG" else "short"
            if side is None or lot_side == side:
                lots.append((lot_id, lot))
        return lots

    @staticmethod
    def _detail_filled_size(detail: dict, fallback: float) -> float:
        return float(
            detail.get("baseVolume") or detail.get("filledQty")
            or detail.get("filledSize") or fallback
        )

    def _build_lot(self, order_id: str, pending: dict, detail: dict) -> tuple[str, dict]:
        client_oid = str(detail.get("clientOid") or pending.get("client_oid") or order_id)
        direction = str(pending["direction"])
        filled_size = self._detail_filled_size(detail, float(pending["size"]))
        entry = float(detail.get("priceAvg") or pending["entry"])
        filled_at_ms = int(detail.get("uTime") or detail.get("cTime") or time.time() * 1000)
        tp1_exit_pct = pending.get("tp1_exit_pct")
        if tp1_exit_pct is None:
            tp1_exit_pct = (
                self.config.trading.active_leverage_tier.tp1_exit_pct
                if self.config.trading.leverage_tiers else 0.7
            )
        lot = {
            "symbol": pending["symbol"],
            "direction": direction,
            "entry_order_id": order_id,
            "client_oid": client_oid,
            "lifecycle": "protecting",
            "original_size": filled_size,
            "remaining_size": filled_size,
            "entry": entry,
            "entry_fee": abs(float(detail.get("fee") or 0)),
            "filled_at_ms": filled_at_ms,
            "stop_loss": float(pending["stop_loss"]),
            "take_profit_1": float(pending["take_profit_1"]),
            "take_profit_2": float(pending["take_profit_2"]),
            "tp1_exit_pct": float(tp1_exit_pct),
            "current_sl": float(pending["stop_loss"]),
            "plan_ids": {},
            "plan_client_oids": {},
            "protection_verified": False,
            # The fill bar's extreme may precede the maker fill. Market entries also
            # occur after their decision bar, so both wait for the next completed bar.
            "last_trail_bar": str(latest_completed_bar_open(
                self.config.trading.primary_timeframe
            )),
        }
        return client_oid, lot

    def _split_lot_size(self, lot: dict) -> tuple[float, float]:
        total = float(lot["original_size"])
        pct = float(lot["tp1_exit_pct"])
        if self.exchange._dry_run:
            first = total * pct
            return first, total - first
        first, remainder = self.exchange.split_size(lot["symbol"], total, pct)
        return float(first), float(remainder)

    def _ensure_lot_protection(
        self, lot_id: str, active_plans: list[PlanOrder] | None = None,
    ) -> bool:
        """Idempotently establish independently sized SL, TP1 and TP2 plans."""
        lot = self._tracked_trades[lot_id]
        side = "long" if lot["direction"] == "LONG" else "short"
        tp1_size, tp2_size = self._split_lot_size(lot)
        lot["tp1_size"] = tp1_size
        lot["tp2_size"] = tp2_size
        lot.setdefault("plan_ids", {})
        lot.setdefault("plan_client_oids", {})
        specs = {
            "sl": ("loss_plan", float(lot["remaining_size"]), float(lot["current_sl"])),
            "tp1": ("profit_plan", tp1_size, float(lot["take_profit_1"])),
            "tp2": ("profit_plan", tp2_size, float(lot["take_profit_2"])),
        }
        if active_plans is None:
            active_plans = ([] if self.exchange._dry_run
                            else self.exchange.get_tpsl_orders(lot["symbol"]))
        by_client = {plan.client_oid: plan for plan in active_plans if plan.client_oid}
        for action, (plan_type, size, trigger) in specs.items():
            client_oid = lot["plan_client_oids"].setdefault(
                action, self._plan_client_oid(lot_id, action),
            )
            existing = by_client.get(client_oid)
            if existing:
                lot["plan_ids"][action] = existing.order_id
                continue
            placed = self.exchange.place_tpsl_order(
                lot["symbol"], side, size, trigger, plan_type, client_oid,
            )
            lot["plan_ids"][action] = placed.order_id

        if self.exchange._dry_run:
            lot["protection_verified"] = True
            lot["lifecycle"] = "protected"
            self._save_live_state()
            return True

        verified = self.exchange.get_tpsl_orders(lot["symbol"])
        verified_ids = {plan.order_id for plan in verified}
        required = set(lot["plan_ids"].values())
        lot["protection_verified"] = bool(required) and required <= verified_ids
        if not lot["protection_verified"]:
            lot["lifecycle"] = "protecting"
            self._save_live_state()
            return False

        lot["lifecycle"] = "protected"
        self._save_live_state()
        return True

    def _cancel_replaced_presets(self, symbol: str) -> None:
        """Remove anonymous entry presets only after every lot has explicit plans."""
        lots = self._lots_for_symbol(symbol)
        if not lots or any(not lot.get("protection_verified") for _, lot in lots):
            return
        known_ids = {
            str(plan_id)
            for _, lot in lots for plan_id in lot.get("plan_ids", {}).values()
        }
        expected_presets = [
            (
                int(lot.get("filled_at_ms", 0) or 0),
                float(price),
                float(lot.get("original_size", 0) or 0),
            )
            for _, lot in lots
            for price in (lot.get("stop_loss"), lot.get("take_profit_2"))
            if price is not None
        ]
        for plan in self.exchange.get_tpsl_orders(symbol):
            if plan.order_id in known_ids:
                continue
            # Positively identify a Bitget-attached entry preset: it carries no
            # clientOid, a BITGET#-prefixed one, or a Bitget-generated all-numeric
            # id. A human/operator plan uses a custom (non-numeric) clientOid and is
            # never cancelled on a mere trigger/size match.
            coid = plan.client_oid or ""
            if coid and not (coid.upper().startswith("BITGET#") or coid.isdigit()):
                continue
            if plan.plan_type not in (
                "profit_plan", "loss_plan", "pos_profit", "pos_loss",
            ):
                continue
            matches_known_preset = any(
                abs(plan.trigger_price - trigger) < 1e-12
                and (plan.size == 0 or abs(plan.size - size) < 1e-12)
                and created_at > 0
                and abs(plan.created_at_ms - created_at) <= 300_000
                for created_at, trigger, size in expected_presets
            )
            if matches_known_preset:
                self.exchange.cancel_tpsl_order(symbol, plan.order_id, plan.plan_type)

    def _activate_filled_pending(self, order_id: str, detail: dict) -> None:
        pending = self._pending_orders[order_id]
        detail_oid = str(detail.get("clientOid") or "")
        expected_oid = str(pending.get("client_oid") or "")
        if detail_oid and expected_oid and detail_oid != expected_oid:
            # The exchange resolved this order id to an order that is NOT the
            # one this pending entry placed (stale clientOid adoption). Never
            # build or overwrite a lot from someone else's fill.
            self._pending_orders.pop(order_id, None)
            self._save_pending_orders()
            self._log(
                f"CRITICAL: pending {order_id} resolved to a foreign order "
                f"(clientOid {detail_oid} != expected {expected_oid}) — "
                "dropped without activation"
            )
            self._log_decision({
                "action": "MAKER_ACTIVATE_MISMATCH", "order_id": order_id,
                "detail_client_oid": detail_oid,
                "expected_client_oid": expected_oid,
            })
            return
        pending = self._pending_orders.pop(order_id)
        fills = self.exchange.get_order_fills(pending["symbol"], order_id, detail)
        if fills:
            total_size = sum(fill.size for fill in fills)
            detail = dict(detail)
            detail["baseVolume"] = total_size
            detail["priceAvg"] = sum(fill.price * fill.size for fill in fills) / total_size
            detail["fee"] = sum(fill.fee for fill in fills)
            detail["uTime"] = max(fill.timestamp_ms for fill in fills)
        lot_id, lot = self._build_lot(order_id, pending, detail)
        lot["fills"] = [fill.__dict__ for fill in fills]
        self._tracked_trades[lot_id] = lot
        self._save_live_state()
        try:
            protected = self._ensure_lot_protection(lot_id)
            if protected and not self.exchange._dry_run:
                self._cancel_replaced_presets(pending["symbol"])
        except Exception as exc:
            protected = False
            lot["protection_verified"] = False
            lot["lifecycle"] = "protecting"
            self._save_live_state()
            self._log(f"CRITICAL: fill {order_id} protection reconciliation failed: {exc}")
        self._log(
            f"Maker order {order_id} filled @ ${lot['entry']:,.2f}; "
            f"per-lot protection {'verified' if protected else 'pending'}"
        )
        self._log_decision({
            "action": "MAKER_FILL", "order_id": order_id,
            "lot_id": lot_id, "entry": lot["entry"], "side": pending["direction"],
            "size": lot["original_size"], "entry_fee": lot["entry_fee"],
            "filled_at_ms": lot["filled_at_ms"],
            "protection_verified": protected,
        })

    def _retry_rejected_maker(self, order_id: str, pending: dict,
                              now: float) -> bool:
        """Maker v2: re-place a post-only would-cross rejected entry.

        Re-pegs at min(intended, best bid) for LONG / max(intended, best ask)
        for SHORT — crossing-proof by construction, so the replacement rests
        and fills at or better than the intended limit (the price the backtest
        fill model assumes). Inherits the same-bar expiry and per-lot targets;
        gives up after ``trading.maker_retry_max`` attempts (0 = v1 behavior).
        """
        retries = int(pending.get("retries", 0))
        if (pending.get("entry_mode") != "maker"
                or retries >= self.config.trading.maker_retry_max
                or now >= float(pending["expires_at"])):
            return False
        symbol = pending["symbol"]
        side = "buy" if pending["direction"] == "LONG" else "sell"
        quote = self.exchange.get_ticker(symbol)
        intended = float(pending.get("intended_entry", pending["entry"]))
        reprice = (min(intended, quote["bid"]) if side == "buy"
                   else max(intended, quote["ask"]))
        stop_loss = float(pending["stop_loss"])
        if ((side == "buy" and reprice <= stop_loss)
                or (side == "sell" and reprice >= stop_loss)):
            return False  # market ran through the stop — entry no longer sane
        tp1 = float(pending["take_profit_1"])
        tp2 = float(pending["take_profit_2"])
        targets = TradeTargets(
            entry=reprice, stop_loss=stop_loss,
            take_profit_1=tp1, take_profit_2=tp2,
            risk_amount=abs(reprice - stop_loss),
            reward_1=abs(tp1 - reprice), reward_2=abs(tp2 - reprice),
            direction=(Direction.BULLISH if side == "buy"
                       else Direction.BEARISH),
        )
        # Derive the retry oid from the ORIGINAL entry oid, NOT from
        # _entry_client_oid(): the candidate bar is unset during later
        # reconcile cycles, which collapsed every retry onto one eternal
        # H("manual") oid — Bitget's clientOid idempotency then resurrected
        # a long-dead order and reconcile "filled" it again (phantom fills,
        # 2026-07-23). Root+attempt is unique per attempt yet deterministic,
        # so lost-response recovery still finds THIS attempt's order.
        root = str(pending.get("entry_oid_root")
                   or pending.get("client_oid") or order_id)
        client_oid = f"{root}-r{retries + 1}"
        result = self.exchange.place_order(
            symbol=symbol, side=side, size=float(pending["size"]),
            targets=targets, leverage=int(pending.get("leverage") or 1),
            order_type="limit", price=reprice, client_oid=client_oid,
        )
        new_pending = dict(pending)
        new_pending.update({
            "entry": result.price if result.price is not None else reprice,
            "intended_entry": intended,
            "entry_oid_root": root,
            "client_oid": client_oid,
            "placed_at": now,
            "retries": retries + 1,
        })
        self._pending_orders[result.order_id] = new_pending
        self._save_pending_orders()
        self._log(
            f"Maker retry {retries + 1}/{self.config.trading.maker_retry_max}: "
            f"re-placed rejected order {order_id} as {result.order_id} "
            f"@ {result.price if result.price is not None else reprice} "
            f"(intended {intended})"
        )
        self._log_decision({
            "action": "MAKER_RETRY", "order_id": result.order_id,
            "replaces": order_id, "attempt": retries + 1,
            "price": result.price if result.price is not None else reprice,
            "intended": intended,
        })
        return True

    def _reconcile_pending_orders(self) -> None:
        """Promote filled maker orders and cancel orders after one primary bar.

        Cancellation is followed by a detail query to handle the fill/cancel race.
        A partially-filled order has its remainder cancelled at expiry and the filled
        position remains protected by the preset TP/SL attached at placement.
        """
        now = time.time()
        for order_id, pending in list(self._pending_orders.items()):
            if not self._same_symbol(pending.get("symbol", ""),
                                     self.config.trading.symbol):
                continue
            try:
                detail = self.exchange.get_order_detail(pending["symbol"], order_id)
                state = str(detail.get("state") or detail.get("status") or "").lower()
                if state == "filled":
                    self._activate_filled_pending(order_id, detail)
                    continue
                if state == "canceled":
                    filled_size = float(detail.get("baseVolume") or 0)
                    if filled_size > 0:
                        self._activate_filled_pending(order_id, detail)
                    else:
                        # Exchange cancelled it before our expiry — for a post-only
                        # maker entry this is a would-cross rejection. Log distinctly
                        # from an untouched bar-expiry so fill-rate bucketing can tell
                        # "post-only rejected" from "expired unfilled".
                        self._pending_orders.pop(order_id, None)
                        self._save_pending_orders()
                        retried = False
                        try:
                            retried = self._retry_rejected_maker(
                                order_id, pending, now)
                        except Exception as exc:  # never break the reconcile loop
                            self._log(f"Maker retry failed for {order_id}: {exc}")
                        if not retried:
                            self._log(
                                f"Maker order {order_id} cancelled by exchange "
                                f"before expiry (post-only would-cross rejection)"
                            )
                            self._log_decision({
                                "action": "MAKER_POST_ONLY_CANCELLED",
                                "order_id": order_id,
                                "retries": int(pending.get("retries", 0)),
                            })
                    continue
                if pending.get("entry_mode", "maker") != "maker":
                    # A market order is expected to fill promptly, but an ambiguous
                    # state is never "fixed" by sending a second order or maker cancel.
                    continue
                if now < float(pending["expires_at"]):
                    continue

                self.exchange.cancel_order(pending["symbol"], order_id)
                # Query after cancellation: the order may have filled as cancel arrived.
                final = self.exchange.get_order_detail(pending["symbol"], order_id)
                final_state = str(final.get("state") or final.get("status") or "").lower()
                filled_size = float(final.get("baseVolume") or detail.get("baseVolume") or 0)
                if final_state == "filled" or filled_size > 0:
                    self._activate_filled_pending(order_id, final or detail)
                else:
                    self._pending_orders.pop(order_id, None)
                    self._save_pending_orders()
                    self._log(f"Maker order {order_id} expired unfilled and was cancelled")
                    self._log_decision({
                        "action": "MAKER_CANCEL_UNFILLED", "order_id": order_id,
                    })
            except Exception as e:
                # Keep the order tracked: losing lifecycle state is less safe than
                # retrying reconciliation on the next scheduler tick.
                self._log(f"Warning: maker order {order_id} reconciliation failed: {e}")

    def _pending_from_exchange_detail(self, detail: dict) -> dict:
        """Reconstruct strategy targets for a bot-owned order after lost persistence."""
        entry = float(detail.get("priceAvg") or detail.get("price") or 0)
        stop = float(detail.get("presetStopLossPrice") or 0)
        tp2 = float(detail.get("presetStopSurplusPrice") or 0)
        size = self._detail_filled_size(detail, float(detail.get("size") or 0))
        side = str(detail.get("side") or "").lower()
        if entry <= 0 or stop <= 0 or tp2 <= 0 or size <= 0 or side not in ("buy", "sell"):
            raise SafetyViolation(
                "REFUSED: Cannot reconstruct a bot order without entry, size, SL and TP2"
            )
        tier = self.config.trading.active_leverage_tier
        # Recover the original deterministic targets from the two exchange presets.
        # This remains exact even when a market fill slipped away from decision entry.
        risk = abs(tp2 - stop) / (tier.tp2_rr + 1)
        target_entry = stop + risk if side == "buy" else stop - risk
        tp1 = (
            target_entry + risk * tier.tp1_rr
            if side == "buy" else target_entry - risk * tier.tp1_rr
        )
        return {
            "symbol": str(detail.get("symbol") or self.config.trading.symbol),
            "direction": "LONG" if side == "buy" else "SHORT",
            "entry": entry,
            "stop_loss": stop,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "tp1_exit_pct": tier.tp1_exit_pct,
            "size": size,
            "leverage": int(float(detail.get("leverage") or tier.leverage)),
            "client_oid": str(detail.get("clientOid") or ""),
            "entry_mode": "maker" if str(detail.get("orderType", "")).lower() == "limit" else "taker",
            "placed_at": int(detail.get("cTime") or time.time() * 1000) / 1000,
            "expires_at": 0,
        }

    def _validate_pending_against_exchange(self, pending: dict, detail: dict) -> None:
        """Reject persisted entry state that disagrees with Bitget's safety preset."""
        expected_client = str(pending.get("client_oid") or "")
        actual_client = str(detail.get("clientOid") or "")
        if expected_client and actual_client and expected_client != actual_client:
            raise SafetyViolation("REFUSED: Pending entry clientOid mismatch")
        actual_side = str(detail.get("side") or "").lower()
        expected_side = "buy" if pending.get("direction") == "LONG" else "sell"
        if actual_side and actual_side != expected_side:
            raise SafetyViolation("REFUSED: Pending entry side mismatch")
        checks = {
            "size": (detail.get("size"), pending.get("size")),
            "stop loss": (detail.get("presetStopLossPrice"), pending.get("stop_loss")),
            "TP2": (detail.get("presetStopSurplusPrice"), pending.get("take_profit_2")),
        }
        for label, (actual, expected) in checks.items():
            if actual in (None, "") or expected in (None, ""):
                raise SafetyViolation(f"REFUSED: Pending entry {label} is unavailable")
            if abs(float(actual) - float(expected)) > 1e-12:
                raise SafetyViolation(f"REFUSED: Pending entry {label} mismatch")

    def _reconcile_lot_lifecycle(self) -> None:
        """Apply exchange-observed TP1/SL/TP2 transitions to every local lot."""
        symbol = self.config.trading.symbol
        lots = self._lots_for_symbol(symbol)
        if not lots:
            return
        active = self.exchange.get_tpsl_orders(symbol)
        history = self.exchange.get_tpsl_orders(symbol, history=True)
        active_ids = {plan.order_id for plan in active}
        history_by_id = {plan.order_id: plan for plan in history}

        for lot_id, lot in list(lots):
            if lot.get("lifecycle") == "closing":
                close_order_id = str(lot.get("close_order_id") or "")
                close_client_oid = str(lot.get("close_client_oid") or "")
                detail = self.exchange.get_order_detail(
                    lot["symbol"], close_order_id or None,
                    client_oid=None if close_order_id else close_client_oid,
                )
                state = str(detail.get("state") or detail.get("status") or "").lower()
                if state != "filled":
                    raise SafetyViolation(
                        f"Lot {lot_id} close order is unresolved ({state or 'unknown'})"
                    )
                for action, plan_id in lot.get("plan_ids", {}).items():
                    if plan_id in active_ids:
                        plan_type = "loss_plan" if action == "sl" else "profit_plan"
                        self.exchange.cancel_tpsl_order(lot["symbol"], plan_id, plan_type)
                reason = str(lot.get("close_reason") or "signal_flip")
                exit_price = float(detail.get("priceAvg") or 0) or float(lot["entry"])
                net_est = self._apply_close_outcome(lot, exit_price, reason)
                self._tracked_trades.pop(lot_id, None)
                self._save_live_state()
                self._log_decision({
                    "action": "LOT_CLOSED", "lot_id": lot_id,
                    "reason": reason, "exit_price": exit_price,
                    "filled_size": float(lot.get("remaining_size") or 0),
                    "entry": lot.get("entry"), "direction": lot.get("direction"),
                    "net_pnl_est": net_est,
                })
                continue
            if not lot.get("protection_verified"):
                self._ensure_lot_protection(lot_id, active)
                continue
            ids = lot.get("plan_ids", {})
            sl_event = history_by_id.get(ids.get("sl", ""))
            tp2_event = history_by_id.get(ids.get("tp2", ""))
            tp1_event = history_by_id.get(ids.get("tp1", ""))
            terminal = next(
                (event for event in (sl_event, tp2_event)
                 if event and event.status == "executed"),
                None,
            )
            if terminal:
                for action, plan_id in ids.items():
                    if plan_id in active_ids:
                        plan_type = "loss_plan" if action == "sl" else "profit_plan"
                        self.exchange.cancel_tpsl_order(lot["symbol"], plan_id, plan_type)
                reason = "sl" if terminal.order_id == ids.get("sl") else "tp2"
                exit_price = (float(lot.get("current_sl") or 0) if reason == "sl"
                              else float(lot.get("take_profit_2") or 0))
                net_est = self._apply_close_outcome(lot, exit_price, reason)
                self._tracked_trades.pop(lot_id, None)
                self._save_live_state()
                self._log_decision({
                    "action": "LOT_CLOSED", "lot_id": lot_id,
                    "reason": reason,
                    "exit_price": exit_price,
                    "filled_size": terminal.filled_size,
                    "entry": lot.get("entry"), "direction": lot.get("direction"),
                    "net_pnl_est": net_est,
                    "execute_order_id": terminal.execute_order_id,
                })
                continue

            if (tp1_event and tp1_event.status == "executed"
                    and lot.get("lifecycle") != "remainder"):
                actual_tp1 = tp1_event.filled_size or float(lot["tp1_size"])
                remaining = max(0.0, float(lot["original_size"]) - actual_tp1)
                if remaining <= 0:
                    raise SafetyViolation(f"TP1 unexpectedly consumed all of lot {lot_id}")
                side = "long" if lot["direction"] == "LONG" else "short"
                self.exchange.modify_stop_loss(
                    lot["symbol"], side, remaining, float(lot["entry"]),
                    plan_order_id=ids.get("sl"), position_level=False,
                )
                self.exchange.modify_tpsl_order(
                    lot["symbol"], ids.get("tp2", ""), side, remaining,
                    float(lot["take_profit_2"]), protective=False,
                )
                lot["remaining_size"] = remaining
                lot["current_sl"] = float(lot["entry"])
                lot["lifecycle"] = "remainder"
                lot["tp1_fill_size"] = actual_tp1
                lot["tp1_filled_at_ms"] = tp1_event.updated_at_ms
                lot["protection_verified"] = (
                    ids.get("sl") in active_ids and ids.get("tp2") in active_ids
                )
                self._save_live_state()
                self._log_decision({
                    "action": "TP1_PARTIAL", "lot_id": lot_id,
                    "price": lot["take_profit_1"],
                    "filled_size": actual_tp1, "remaining_size": remaining,
                    "break_even_sl": lot["entry"],
                    "entry": lot.get("entry"), "direction": lot.get("direction"),
                })
                continue

            required_actions = ("sl", "tp2") if lot.get("lifecycle") == "remainder" else (
                "sl", "tp1", "tp2",
            )
            missing = [action for action in required_actions if ids.get(action) not in active_ids]
            if missing:
                lot["protection_verified"] = False
                lot["lifecycle"] = "protecting"
                self._save_live_state()
                raise SafetyViolation(
                    f"Lot {lot_id} is missing active protection plans: {', '.join(missing)}"
                )

    def reconcile_startup(self, allowed_symbols: set[str] | None = None) -> None:
        """Make exchange state authoritative before any analysis or new order."""
        with self._execution_lock:
            result = self.exchange.preflight(self.config.trading.symbol)
            # Fail closed if the account leverage drifted from the active tier. The
            # scheduler does not set leverage globally, so a mismatch (e.g. changed
            # in the Bitget UI) would silently size positions wrong. Keys are absent
            # on mocked preflight results, which skips the check.
            expected_leverage = self.config.trading.active_leverage_tier.leverage
            for key in ("leverage_long", "leverage_short"):
                lev = result.get(key)
                if lev is not None and int(lev) != int(expected_leverage):
                    raise SafetyViolation(
                        f"REFUSED: Bitget {key} is {lev}, expected {expected_leverage}"
                    )
            allowed_symbols = allowed_symbols or {self.config.trading.symbol}
            raw_pending = self.exchange.get_pending_order_rows()
            account_pending = self.exchange.get_pending_orders()
            account_positions = self.exchange.get_positions()
            account_plans = self.exchange.get_tpsl_orders()
            unexpected_symbols = {
                symbol for symbol in [
                    *(item.symbol for item in account_pending),
                    *(item.symbol for item in account_positions),
                    *(item.symbol for item in account_plans),
                    *(str(row.get("symbol", "")) for row in raw_pending),
                ]
                if symbol and not any(
                    self._same_symbol(symbol, allowed) for allowed in allowed_symbols
                )
            }
            if unexpected_symbols:
                raise SafetyViolation(
                    "REFUSED: Exchange exposure exists outside configured symbols: "
                    + ", ".join(sorted(unexpected_symbols))
                )
            known_close_clients = {
                str(lot.get("close_client_oid", ""))
                for lot in self._tracked_trades.values()
                if lot.get("lifecycle") == "closing"
            }
            for row in raw_pending:
                trade_side = str(row.get("tradeSide") or "open").lower()
                reduce_only = str(row.get("reduceOnly", "NO")).upper() == "YES"
                if trade_side == "close" or reduce_only:
                    client_oid = str(row.get("clientOid", ""))
                    if client_oid not in known_close_clients:
                        raise SafetyViolation(
                            f"REFUSED: Unknown closing/reduce-only order {row.get('orderId', '')}"
                        )
            exchange_pending = self.exchange.get_pending_orders(self.config.trading.symbol)
            pending_by_id = {order.order_id: order for order in exchange_pending}

            # Adopt accepted bot entries whose response/local save was lost. Never
            # cancel or guess at orders created outside this bot namespace.
            for order in exchange_pending:
                if order.order_id in self._pending_orders:
                    continue
                if not order.client_oid.startswith("llt-"):
                    raise SafetyViolation(
                        f"REFUSED: Unknown exchange order {order.order_id} ({order.client_oid or 'no clientOid'})"
                    )
                detail = self.exchange.get_order_detail(order.symbol, order.order_id)
                self._pending_orders[order.order_id] = self._pending_from_exchange_detail(detail)

            for order_id, pending in list(self._pending_orders.items()):
                if not self._same_symbol(pending.get("symbol", ""), self.config.trading.symbol):
                    continue
                if order_id in pending_by_id:
                    detail = self.exchange.get_order_detail(pending["symbol"], order_id)
                    self._validate_pending_against_exchange(pending, detail)
                else:
                    detail = self.exchange.get_order_detail(pending["symbol"], order_id)
                    state = str(detail.get("state") or detail.get("status") or "").lower()
                    if state == "filled" or self._detail_filled_size(detail, 0) > 0:
                        self._activate_filled_pending(order_id, detail)
                    elif state in ("canceled", "cancelled"):
                        self._pending_orders.pop(order_id, None)
                    else:
                        raise SafetyViolation(
                            f"REFUSED: Local order {order_id} has unresolved exchange state {state!r}"
                        )

            # V2 stored one ambiguous symbol-level trailing context. It cannot prove
            # lot quantities or TPSL ownership, so rebuild it from exchange history.
            for lot_id, lot in list(self._lots_for_symbol(self.config.trading.symbol)):
                if lot.get("lifecycle") == "unreconciled" and not lot.get("original_size"):
                    self._tracked_trades.pop(lot_id, None)

            positions = self.exchange.get_positions(self.config.trading.symbol)
            history = self.exchange.get_order_history(self.config.trading.symbol)
            known_clients = {str(lot.get("client_oid", "")) for _, lot in self._lots_for_symbol(
                self.config.trading.symbol
            )}
            for pos in positions:
                known_size = sum(
                    float(lot.get("remaining_size", 0))
                    for _, lot in self._lots_for_symbol(pos.symbol, pos.side.lower())
                )
                if known_size + 1e-12 < pos.size:
                    candidates = sorted(
                        (
                            row for row in history
                            if str(row.get("clientOid", "")).startswith("llt-")
                            and str(row.get("clientOid")) not in known_clients
                            and str(row.get("status", "")).lower() == "filled"
                            and str(row.get("reduceOnly", "NO")).upper() != "YES"
                            and str(row.get("tradeSide") or "open").lower()
                            not in ("close", "reduce_close_long", "reduce_close_short")
                            and self._same_symbol(
                                str(row.get("symbol", "")), self.config.trading.symbol
                            )
                            and ((str(row.get("side", "")).lower() == "buy") == (pos.side.lower() == "long"))
                        ),
                        key=lambda row: int(row.get("cTime", 0) or 0), reverse=True,
                    )
                    for row in candidates:
                        pending = self._pending_from_exchange_detail(row)
                        lot_id, lot = self._build_lot(str(row.get("orderId", "")), pending, row)
                        self._tracked_trades[lot_id] = lot
                        known_clients.add(lot_id)
                        known_size += float(lot["remaining_size"])
                        if known_size + 1e-12 >= pos.size:
                            break
                if known_size + 1e-12 < pos.size:
                    raise SafetyViolation(
                        f"REFUSED: Unexplained {pos.side} {pos.symbol} position size "
                        f"{pos.size}; only {known_size} is attributable to bot lots"
                    )

            self._save_live_state()
            active = self.exchange.get_tpsl_orders(self.config.trading.symbol)
            for lot_id, _lot in self._lots_for_symbol(self.config.trading.symbol):
                self._ensure_lot_protection(lot_id, active)
            self._cancel_replaced_presets(self.config.trading.symbol)
            self._reconcile_lot_lifecycle()
            if any(not lot.get("protection_verified")
                   for _, lot in self._lots_for_symbol(self.config.trading.symbol)):
                raise SafetyViolation("REFUSED: One or more open lots lack verified SL/TP protection")
            positions = self.exchange.get_positions(self.config.trading.symbol)
            for side in ("long", "short"):
                exchange_size = sum(pos.size for pos in positions if pos.side.lower() == side)
                local_size = sum(
                    float(lot.get("remaining_size", 0))
                    for _, lot in self._lots_for_symbol(self.config.trading.symbol, side)
                )
                if abs(exchange_size - local_size) > 1e-12:
                    raise SafetyViolation(
                        f"REFUSED: {side} position/lot mismatch: exchange={exchange_size}, "
                        f"local={local_size}"
                    )
            known_plan_ids = {
                str(plan_id)
                for _, lot in self._lots_for_symbol(self.config.trading.symbol)
                for plan_id in lot.get("plan_ids", {}).values()
            }
            unexplained_plans = [
                plan for plan in self.exchange.get_tpsl_orders(self.config.trading.symbol)
                if plan.order_id not in known_plan_ids
            ]
            if unexplained_plans:
                raise SafetyViolation(
                    "REFUSED: Unexplained active TPSL plans: "
                    + ", ".join(plan.order_id for plan in unexplained_plans)
                )
            self._startup_reconciled = True
            self._log(
                f"Startup reconciliation passed for {result['symbol']} "
                f"({result['position_mode']}, {result['margin_mode']}, "
                f"clock drift {result['clock_drift_ms']}ms)"
            )

    # ------------------------------------------------------------------
    # Logging: one file per LOCAL day, deleted after scheduling.log_retention_days
    # (default 90). decisions-YYYY-MM-DD.jsonl is the structured stream used to
    # evaluate the paper/live run offline (Grafana + live-vs-backtest drift);
    # trading-*.log is the human-readable mirror. Timestamps are local time with
    # UTC offset in the structured stream (decision BARS remain UTC-aligned —
    # that is exchange reality, not a logging choice).
    # ------------------------------------------------------------------

    def _prune_old_logs(self, today: str) -> None:
        """Delete dated log files older than the retention window (once a day)."""
        if today == self._pruned_log_day:
            return
        self._pruned_log_day = today
        retention = timedelta(days=self.config.scheduling.log_retention_days)
        cutoff_date = (datetime.now().astimezone() - retention).date()
        for pattern, prefix in (("trading-*.log", "trading-"),
                                ("decisions-*.jsonl", "decisions-")):
            for path in self._log_dir.glob(pattern):
                stamp = path.name[len(prefix):].split(".")[0]
                try:
                    file_date = datetime.strptime(stamp, "%Y-%m-%d").date()
                except ValueError:
                    continue  # not one of our dated files — never delete it
                if file_date < cutoff_date:
                    path.unlink(missing_ok=True)

    def _daily_log_path(self, prefix: str, suffix: str) -> Path:
        day = datetime.now().astimezone().strftime("%Y-%m-%d")
        self._prune_old_logs(day)
        return self._log_dir / f"{prefix}-{day}{suffix}"

    def _log(self, msg: str) -> None:
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{self.config.trading.symbol}] {msg}"
        print(line)
        with open(self._daily_log_path("trading", ".log"), "a") as f:
            f.write(line + "\n")

    def _log_decision(self, decision: dict) -> None:
        decision.setdefault("symbol", self.config.trading.symbol)
        # Local time with explicit UTC offset so records stay unambiguous.
        decision["timestamp"] = datetime.now().astimezone().isoformat()
        self.decision_log.append(decision)
        # Append to the persistent structured log
        with open(self._daily_log_path("decisions", ".jsonl"), "a") as f:
            f.write(json.dumps(decision, default=str) + "\n")

    # ------------------------------------------------------------------
    # Backtest-parity risk state: post-SL cooldown, consecutive-loss entry
    # penalty (persisted per symbol so restarts cannot soften risk behavior).
    # Semantics mirror BacktestEngine/_on_trade_closed + fastbt exactly:
    # counters tick once per completed primary bar, a losing close bumps the
    # penalty, an SL-family loss also arms the cooldown, and a win resets both.
    # ------------------------------------------------------------------

    def _risk_state(self) -> dict:
        counters = self._live_state.risk_counters.setdefault(
            self.config.trading.symbol, {}
        )
        counters.setdefault("consecutive_losses", 0)
        counters.setdefault("candles_since_last_loss", 999)
        counters.setdefault("cooldown_remaining", 0)
        counters.setdefault("last_counter_bar", "")
        return counters

    def _tick_risk_counters(self, analysis_bar: str) -> None:
        """Advance per-bar counters exactly once per completed primary bar.

        Downtime is handled by counting the number of primary bars elapsed since
        the last tick, so a restarted bot does not carry a stale cooldown.
        """
        counters = self._risk_state()
        if counters["last_counter_bar"] == analysis_bar:
            return
        elapsed = 1
        if counters["last_counter_bar"]:
            try:
                delta = pd.Timestamp(analysis_bar) - pd.Timestamp(
                    counters["last_counter_bar"]
                )
                bar_seconds = self._timeframe_seconds(
                    self.config.trading.primary_timeframe
                )
                elapsed = max(1, int(delta.total_seconds() // bar_seconds))
            except (ValueError, TypeError):
                elapsed = 1
        for _ in range(min(elapsed, 1000)):
            counters["candles_since_last_loss"] = min(
                999, counters["candles_since_last_loss"] + 1
            )
            if counters["cooldown_remaining"] > 0:
                counters["cooldown_remaining"] -= 1
        counters["last_counter_bar"] = analysis_bar
        self._save_live_state()

    def _loss_penalty(self) -> float:
        """Entry-threshold penalty from consecutive losses (engine parity)."""
        counters = self._risk_state()
        if counters["consecutive_losses"] == 0:
            return 0.0
        rm = self.config.risk_management
        base = min(
            counters["consecutive_losses"] * rm.consecutive_loss_penalty,
            rm.max_consecutive_loss_penalty,
        )
        decay = rm.loss_penalty_decay_candles
        since = counters["candles_since_last_loss"]
        if decay > 0 and since > decay:
            return base * max(0.0, 1.0 - (since - decay) / decay)
        return base

    def _lot_realized_pnl(self, lot: dict, exit_price: float,
                          exit_reason: str) -> float:
        """Estimate a closed lot's realized net PnL from its recorded lifecycle.

        Uses the same convention as the backtest Trade: recorded target prices,
        actual entry fee, maker fee for TP exits when configured, taker fee for
        market exits. Funding is settled by the exchange and not attributed here.
        """
        sign = 1.0 if lot.get("direction") == "LONG" else -1.0
        entry = float(lot.get("entry") or 0.0)
        fees = self.config.fees

        def exit_fee_rate(reason: str) -> float:
            if self.config.risk_management.use_maker_fee_for_tp and reason in (
                "tp1", "tp2",
            ):
                return fees.maker
            return fees.taker

        net = -abs(float(lot.get("entry_fee") or 0.0))
        tp1_size = float(lot.get("tp1_fill_size") or 0.0)
        if tp1_size > 0:
            tp1 = float(lot.get("take_profit_1") or 0.0)
            net += (tp1 - entry) * sign * tp1_size
            net -= tp1 * tp1_size * exit_fee_rate("tp1")
        size = float(lot.get("remaining_size") or 0.0)
        net += (exit_price - entry) * sign * size
        net -= exit_price * size * exit_fee_rate(exit_reason)
        return net

    def _apply_close_outcome(self, lot: dict, exit_price: float,
                             exit_reason: str) -> float:
        """Update cooldown/penalty counters for one fully closed lot."""
        net = self._lot_realized_pnl(lot, exit_price, exit_reason)
        counters = self._risk_state()
        if net <= 0:
            counters["consecutive_losses"] += 1
            counters["candles_since_last_loss"] = 0
            if exit_reason == "sl":
                counters["cooldown_remaining"] = (
                    self.config.risk_management.cooldown_candles_after_sl
                )
        else:
            counters["consecutive_losses"] = 0
            counters["candles_since_last_loss"] = 999
        self._save_live_state()
        return net

    def analyze_market(self) -> Optional[RoutingDecision]:
        """Fetch data, score, and route the signal."""
        self._candidate_analysis_bar = None

        expected_bar = str(latest_completed_bar_open(
            self.config.trading.primary_timeframe
        ))
        symbol_key = self.config.trading.symbol
        if self._live_state.last_analysis_bars.get(symbol_key) == expected_bar:
            self._log(f"Primary candle {expected_bar} already analyzed")
            return None
        self._log("Fetching market data...")

        try:
            ds = self.config.data_source
            symbol = ds.exchange_symbol if ds.source != "yfinance" else self.config.trading.yfinance_symbol
            clear_cache()  # never promote a cached forming row after it closes
            data_by_tf = fetch_multi_timeframe(
                symbol=symbol,
                timeframes=self.config.trading.timeframes,
                warmup_periods=self.config.scoring.atr_period * 15,
                source=ds.source,
                market=ds.market,
            )
        except Exception as e:
            self._log(f"ERROR fetching data: {e}")
            return None

        if not data_by_tf:
            self._log("No data available")
            return None

        data_by_tf, primary_bar = completed_market_snapshot(
            data_by_tf, self.config.trading.primary_timeframe
        )
        if primary_bar is None:
            self._log("No completed primary candle available")
            return None
        missing_frames = set(self.config.trading.timeframes) - set(data_by_tf)
        if missing_frames:
            self._log(
                "ERROR: completed snapshot missing required timeframes: "
                + ", ".join(sorted(missing_frames))
            )
            return None
        analysis_bar = str(primary_bar)
        if self._live_state.last_analysis_bars.get(symbol_key) == analysis_bar:
            self._log(f"Primary candle {analysis_bar} already analyzed")
            return None

        # Per-bar risk-counter tick (cooldown decay, penalty decay) — mirrors the
        # backtest, which ticks once per completed primary bar before scoring.
        self._tick_risk_counters(analysis_bar)

        # Calculate indicators for each timeframe
        indicators_by_tf = {}
        for tf, df in data_by_tf.items():
            try:
                indicators_by_tf[tf] = calculate_indicators(df, tf)
            except Exception as e:
                self._log(f"ERROR: Failed to calculate required {tf} indicators: {e}")
                return None

        if not indicators_by_tf:
            self._log("No indicators calculated")
            return None

        # Route signal
        decision = route_signal(indicators_by_tf, self.config)
        self._candidate_analysis_bar = analysis_bar

        # Consecutive-loss penalty raises the effective entry thresholds exactly
        # like the backtest engine (classification with penalty-shifted bounds).
        penalty = self._loss_penalty()
        self._current_loss_penalty = penalty
        if penalty > 0 and decision.signal_strength != SignalStrength.WAIT:
            tier = self.config.trading.active_leverage_tier
            abs_score = abs(decision.scoring_result.raw_score)
            if abs_score < tier.marginal_threshold_low + penalty:
                decision.signal_strength = SignalStrength.WAIT
                decision.scoring_result.signal_strength = SignalStrength.WAIT
                decision.skip_reason = (
                    f"Consecutive-loss penalty +{penalty:.1f} raised the entry "
                    f"threshold to {tier.marginal_threshold_low + penalty:.1f}"
                )
            elif abs_score < tier.strong_threshold + penalty:
                decision.signal_strength = SignalStrength.MARGINAL
                decision.scoring_result.signal_strength = SignalStrength.MARGINAL

        self._log(
            f"Signal: {decision.signal_strength.value} | "
            f"Direction: {decision.scoring_result.direction.value} | "
            f"Score: {decision.scoring_result.raw_score:+.1f} | "
            f"Confidence: {decision.scoring_result.confidence:.0f}%"
            + (f" | Loss penalty: +{penalty:.1f}" if penalty > 0 else "")
        )

        return decision

    def _claim_analysis_bar(self, analysis_bar: str) -> bool:
        """Persist an at-most-once live decision gate before order execution."""
        symbol = self.config.trading.symbol
        with self._live_state.lock:
            if self._live_state.last_analysis_bars.get(symbol) == analysis_bar:
                return False
            self._live_state.last_analysis_bars[symbol] = analysis_bar
            self._live_state.save()
            return True

    def _maybe_opposite_exit(self, decision: RoutingDecision) -> None:
        """Close open positions when the composite score flips hard against them
        (risk_management.opposite_exit_threshold; 0 = disabled). Mirrors the
        backtest engine's signal_flip exit."""
        threshold = self.config.risk_management.opposite_exit_threshold
        if threshold <= 0:
            return
        direction = decision.scoring_result.direction
        if direction == Direction.NEUTRAL:
            return
        if abs(decision.scoring_result.raw_score) < threshold:
            return
        want_side = "long" if direction == Direction.BULLISH else "short"
        # A resting opposite entry has no position to close yet; cancel it instead.
        for order_id, pending in list(self._pending_orders.items()):
            if not self._same_symbol(pending.get("symbol", ""),
                                     self.config.trading.symbol):
                continue
            pending_side = "long" if pending["direction"] == "LONG" else "short"
            if pending_side != want_side:
                try:
                    self.exchange.cancel_order(pending["symbol"], order_id)
                    self._pending_orders.pop(order_id, None)
                    self._save_pending_orders()
                    self._log(f"Cancelled opposite resting maker order {order_id}")
                except Exception as e:
                    self._log(f"Warning: could not cancel opposite maker order: {e}")
        try:
            positions = self.exchange.get_positions(self.config.trading.symbol)
        except Exception as e:
            self._log(f"Warning: opposite-exit position check failed: {e}")
            return
        for pos in positions:
            if pos.side.lower() != want_side:
                self._log(
                    f"SIGNAL FLIP ({decision.scoring_result.raw_score:+.1f}) against "
                    f"{pos.side} position — closing {pos.size}"
                )
                try:
                    close_client_oid = self._plan_client_oid(
                        f"{self.config.trading.symbol}:{self._candidate_analysis_bar or 'manual'}",
                        "signal-flip-close",
                    )
                    response = self.exchange.close_position(
                        self.config.trading.symbol, pos.side, pos.size,
                        client_oid=close_client_oid,
                    )
                    close_order_id = str((response.get("data", {}) or {}).get("orderId", ""))
                    for _lot_id, lot in self._lots_for_symbol(pos.symbol, pos.side.lower()):
                        lot["lifecycle"] = "closing"
                        lot["close_reason"] = "signal_flip"
                        lot["close_client_oid"] = close_client_oid
                        lot["close_order_id"] = close_order_id
                    self._save_live_state()
                    self._log_decision({
                        "action": "SIGNAL_FLIP_CLOSE",
                        "side": pos.side, "size": pos.size,
                        "score": decision.scoring_result.raw_score,
                        "close_order_id": close_order_id,
                        "bar": self._candidate_analysis_bar,
                    })
                except Exception as e:
                    self._log(f"ERROR closing position on signal flip: {e}")

    def execute_decision(self, decision: RoutingDecision) -> None:
        """Act on a routing decision."""
        # Opposite-signal exit runs regardless of entry signal strength
        self._maybe_opposite_exit(decision)

        if decision.signal_strength == SignalStrength.WAIT:
            self._log(f"WAIT — {decision.skip_reason or 'Score too low'}")
            self._log_decision({
                "action": "WAIT",
                "reason": decision.skip_reason,
                "score": decision.scoring_result.raw_score,
                "bar": self._candidate_analysis_bar,
            })
            return

        # Post-SL cooldown blocks new entries for N completed primary bars,
        # mirroring the backtest (opposite-signal exits above still run).
        cooldown = self._risk_state()["cooldown_remaining"]
        if cooldown > 0:
            self._log(f"COOLDOWN — skipping entry ({cooldown} bar(s) remaining)")
            self._log_decision({
                "action": "COOLDOWN_SKIP",
                "cooldown_remaining": cooldown,
                "score": decision.scoring_result.raw_score,
                "bar": self._candidate_analysis_bar,
            })
            return

        if decision.signal_strength == SignalStrength.STRONG:
            self._log("STRONG signal — using deterministic template")
            self._log(decision.template_response or "")
            self._execute_trade(decision)
            return

        if decision.signal_strength == SignalStrength.MARGINAL:
            # MARGINAL entries are traded deterministically, exactly as the backtest
            # counts them — they are part of the validated edge. This is a pure
            # technical-signal bot; there is no LLM gate on marginal setups.
            self._log("MARGINAL signal — deterministic execution (backtest parity)")
            self._execute_trade(decision)
            return

    def _execute_trade(self, decision: RoutingDecision) -> None:
        """Atomically check account exposure, size, and place one new order."""
        with self._execution_lock:
            self._execute_trade_locked(decision)

    def _execute_trade_locked(self, decision: RoutingDecision) -> None:
        """Execute a trade via the exchange."""
        if not decision.targets:
            self._log("No targets calculated — cannot trade")
            return
        bar = self._candidate_analysis_bar
        if bar:
            try:
                bar_close = pd.Timestamp(bar) + timeframe_delta(
                    self.config.trading.primary_timeframe
                )
                if bar_close.tzinfo is None:
                    bar_close = bar_close.tz_localize("UTC")
                age = (pd.Timestamp.now(tz="UTC") - bar_close).total_seconds()
            except (ValueError, TypeError):
                age = 0.0
            if age > self.STALE_ENTRY_MAX_SECONDS:
                self._log(
                    f"SKIP: analysis bar {bar} closed {age / 60:.1f} min ago "
                    f"(> {self.STALE_ENTRY_MAX_SECONDS / 60:.0f} min) — stale, not entering"
                )
                self._log_decision({
                    "action": "SKIP_STALE_BAR", "bar": bar,
                    "bar_age_seconds": round(age, 1),
                })
                return
        unprotected = [
            lot_id for lot_id, lot in self._tracked_trades.items()
            if not lot.get("protection_verified", False)
        ]
        if unprotected:
            self._log(
                "SAFETY: new entries blocked while lots lack verified protection: "
                + ", ".join(unprotected)
            )
            return

        targets = decision.targets
        tier = self.config.trading.active_leverage_tier
        ps = self.config.position_sizing
        side = "buy" if targets.direction == Direction.BULLISH else "sell"
        want_side = "long" if targets.direction == Direction.BULLISH else "short"

        balance = self.exchange.get_available_balance()
        equity = self.exchange.get_account_equity(dry_run_default=balance)

        rm = self.config.risk_management
        slots = ps.max_positions
        throttle_risk_mult = 1.0

        # Entry slots: up to max_positions concurrent SAME-direction positions
        # (pyramiding); never stack against an opposite-direction position.
        try:
            positions = self.exchange.get_positions(self.config.trading.symbol)
            global_positions = self.exchange.get_positions()
            exchange_pending = self.exchange.get_pending_orders()
            exchange_pending_ids = {p.order_id for p in exchange_pending}
            local_pending = [
                p for oid, p in self._pending_orders.items()
                if oid not in exchange_pending_ids
            ]
            pending_for_symbol = [
                pending for pending in self._pending_orders.values()
                if self._same_symbol(
                    pending.get("symbol", ""), self.config.trading.symbol
                )
            ]
            pending_for_symbol += [
                {"direction": "LONG" if p.side in ("long", "buy") else "SHORT"}
                for p in exchange_pending
                if self._same_symbol(p.symbol, self.config.trading.symbol)
                and p.order_id not in self._pending_orders
            ]
            # Persist one account-wide realized-balance peak. Exchange equity minus
            # unrealized PnL mirrors the backtest's realized Portfolio.balance more
            # closely than available margin, which falls when orders reserve funds.
            realized_balance = equity - sum(p.unrealized_pnl for p in global_positions)
            peak_balance = self._live_state.update_peak(realized_balance)
            self._peak_balance = peak_balance
            if rm.dd_throttle_threshold > 0 and peak_balance > 0:
                dd = (peak_balance - realized_balance) / peak_balance
                if dd >= rm.dd_throttle_threshold:
                    slots = min(slots, rm.dd_throttle_slots)
                    throttle_risk_mult = rm.dd_throttle_risk
                    self._log(
                        f"DD THROTTLE active ({dd:.1%} >= {rm.dd_throttle_threshold:.1%}) — "
                        f"slots capped at {slots}, risk x{throttle_risk_mult}"
                    )
            # Bitget nets same-side futures into one exchange position, so exchange
            # row count cannot represent pyramided strategy slots. Durable lots can.
            open_lots = len(self._lots_for_symbol(self.config.trading.symbol))
            committed = open_lots + len(pending_for_symbol)
            if committed >= slots:
                self._log(f"Already have {committed}/{slots} committed slot(s) — skipping")
                return
            if any(p.side.lower() != want_side for p in positions):
                self._log("Open position in the opposite direction — not stacking, skipping")
                return
            if any(("long" if p["direction"] == "LONG" else "short") != want_side
                   for p in pending_for_symbol):
                self._log("Resting order in the opposite direction — not stacking, skipping")
                return
            global_committed = (
                len(self._tracked_trades) + len(exchange_pending) + len(local_pending)
            )
            if ps.global_max_positions > 0 and global_committed >= ps.global_max_positions:
                self._log(
                    f"GLOBAL EXPOSURE: {global_committed}/{ps.global_max_positions} "
                    "committed slots — skipping"
                )
                return
        except Exception as e:
            self._log(f"Exposure check failed — skipping new order: {e}")
            return

        # Risk-based position sizing: commit min(balance * risk_pct, max_usd) as margin,
        # leverage it up to the notional, then convert to base-currency size at entry.
        risk_pct = ps.risk_pct_per_trade * throttle_risk_mult
        # Conviction sizing: scale risk with signal strength (mirrors backtest
        # engine, which normalizes by the penalty-raised STRONG threshold).
        eff_strong = tier.strong_threshold + self._current_loss_penalty
        if ps.conviction_exponent > 0 and eff_strong > 0:
            m = (abs(decision.scoring_result.raw_score) / eff_strong) ** ps.conviction_exponent
            risk_pct *= max(0.5, min(1.5, m))
        if ps.anti_martingale_step > 0:
            try:
                history = self.exchange.get_position_history(
                    self.config.trading.symbol, limit=100
                )
                history = sorted(history, key=lambda row: int(row.get("utime", 0) or 0))
                streak = outcome_streak([
                    float(row.get("netProfit", 0) or 0) for row in history
                ])
            except Exception as e:
                streak = 0
                self._log(f"Outcome-streak history unavailable; using neutral size: {e}")
            streak_mult = anti_martingale_multiplier(
                streak, ps.anti_martingale_step,
                ps.anti_martingale_min, ps.anti_martingale_max,
            )
            risk_pct *= streak_mult
        committed_margin = sum(
            p.margin_size if p.margin_size > 0
            else (p.size * p.entry_price / p.leverage if p.leverage > 0 else 0)
            for p in global_positions
        )
        committed_notional = sum(
            p.size * p.entry_price for p in global_positions
        )
        for pending in exchange_pending:
            remaining = max(0.0, pending.size - pending.filled_size)
            pending_notional = remaining * pending.price
            committed_notional += pending_notional
            if pending.leverage > 0:
                committed_margin += pending_notional / pending.leverage
        for pending in local_pending:
            pending_notional = float(pending["size"]) * float(pending["entry"])
            leverage = int(pending.get("leverage", tier.leverage))
            committed_notional += pending_notional
            if leverage > 0:
                committed_margin += pending_notional / leverage
        # Size and cap on the REALIZED balance (equity minus open PnL), matching
        # the backtests, which size on the portfolio's realized balance. The
        # available-balance bound remains as an exchange reality: reserved maker
        # margin cannot be committed twice.
        risk_pct = cap_risk_pct(
            risk_pct, tier.leverage, realized_balance,
            committed_margin, committed_notional,
            risk_multiplier=ps.portfolio_risk_multiplier,
            max_margin_pct=ps.global_max_margin_pct,
            max_notional_pct=ps.global_max_notional_pct,
        )
        margin = min(
            realized_balance * min(risk_pct, ps.max_position_pct), balance,
        )
        size = (margin * tier.leverage) / targets.entry if targets.entry > 0 else 0.0

        def _rescue_min_size():
            """Min-size rescue (opt/probe_overshoot, gates passed): return the
            smallest TP1-splittable lot when the rescue conditions hold, else
            None. Fires only for a positive-but-sub-minimum computed size —
            zero cap headroom never reaches this path (risk_pct was 0)."""
            over = ps.min_size_overshoot
            score_gate = ps.min_size_overshoot_score
            if over is None or score_gate is None:
                return None
            if abs(decision.scoring_result.raw_score) < score_gate:
                return None
            symbol = self.config.trading.symbol
            spec = self.exchange.get_contract_spec(symbol)
            step = float(spec.size_step)
            cand = float(spec.min_size)
            rescued = None
            for _ in range(10):  # smallest size whose TP1 split is exchange-valid
                try:
                    self.exchange.split_size(symbol, cand, tier.tp1_exit_pct)
                    rescued = cand
                    break
                except SafetyViolation:
                    cand = round(cand + step, 12)
            if rescued is None:
                return None
            new_margin = rescued * targets.entry / tier.leverage
            new_notional = rescued * targets.entry
            if new_margin > balance:
                return None  # reserved maker margin cannot be committed twice
            if ps.global_max_margin_pct and (
                    committed_margin + new_margin
                    > (1 + over) * ps.global_max_margin_pct * realized_balance):
                return None
            if ps.global_max_notional_pct and (
                    committed_notional + new_notional
                    > (1 + over) * ps.global_max_notional_pct * realized_balance):
                return None
            return rescued
        if size <= 0:
            self._log(
                f"Computed non-positive size (balance=${balance:,.2f}, margin=${margin:,.2f}) — skipping trade"
            )
            return
        if not self.exchange._dry_run:
            # Reject an entry before submission when either 70% TP1 or the remainder
            # would fall below the contract's executable size step/minimum. This is
            # the deliberate fail-closed "min-size skip" (opt/sizing_scenarios) — a
            # normal small-balance outcome, so log it as a decision (the drift
            # dataset counts skips to decide if the floor policy is ever needed)
            # instead of letting it crash the cycle as an anonymous ERROR.
            try:
                self.exchange.split_size(
                    self.config.trading.symbol, size, tier.tp1_exit_pct,
                )
            except SafetyViolation as exc:
                rescued = _rescue_min_size()
                if rescued is None:
                    self._log(
                        f"MIN-SIZE SKIP: {exc} (margin ${margin:,.2f} @ "
                        f"{tier.leverage}x -> size {size:.6f})"
                    )
                    self._log_decision({
                        "action": "MIN_SIZE_SKIP",
                        "bar": self._candidate_analysis_bar,
                        "score": decision.scoring_result.raw_score,
                        "margin": margin, "size": size,
                        "reason": str(exc),
                    })
                    return
                new_margin = rescued * targets.entry / tier.leverage
                self._log(
                    f"MIN-SIZE RESCUE: computed {size:.6f} -> smallest splittable "
                    f"lot {rescued} (margin ${new_margin:,.2f}, caps stretched "
                    f"x{1 + ps.min_size_overshoot:.2f})"
                )
                self._log_decision({
                    "action": "MIN_SIZE_RESCUE",
                    "bar": self._candidate_analysis_bar,
                    "score": decision.scoring_result.raw_score,
                    "computed_size": size, "size": rescued,
                    "margin": new_margin,
                })
                size, margin = rescued, new_margin

        self._log(
            f"Executing {side.upper()} @ ${targets.entry:,.2f} | "
            f"Size: {size:.6f} (margin ${margin:,.2f} @ {tier.leverage}x) | "
            f"SL: ${targets.stop_loss:,.2f} | "
            f"TP1: ${targets.take_profit_1:,.2f} | "
            f"TP2: ${targets.take_profit_2:,.2f}"
        )

        try:
            client_oid = self._entry_client_oid(side)
            if self.config.trading.entry_mode == "maker":
                result = self.exchange.place_order(
                    symbol=self.config.trading.symbol, side=side, size=size,
                    targets=targets, leverage=tier.leverage,
                    order_type="limit", price=targets.entry, client_oid=client_oid,
                )
            else:
                result = self.exchange.place_order(
                    symbol=self.config.trading.symbol, side=side, size=size,
                    targets=targets, leverage=tier.leverage, client_oid=client_oid,
                )
            self._log(f"Order placed: {result.order_id}")

            direction = "LONG" if targets.direction == Direction.BULLISH else "SHORT"
            placed_at = time.time()
            if self.config.trading.entry_mode == "maker":
                tf_seconds = self._timeframe_seconds(
                    self.config.trading.primary_timeframe
                )
                # Expire on the next UTC-aligned primary-bar close, not an
                # arbitrary wall-clock duration after a delayed scheduler tick.
                expires_at = (int(placed_at // tf_seconds) + 1) * tf_seconds
            else:
                # Market orders are reconciled from exchange-confirmed fills instead
                # of assuming response-time price/quantity. They are never cancelled
                # by the maker expiry path.
                expires_at = 0
            self._pending_orders[result.order_id] = {
                "symbol": self.config.trading.symbol,
                "direction": direction,
                "entry": result.price if result.price is not None else targets.entry,
                "stop_loss": result.stop_loss,
                "take_profit_1": result.take_profit_1,
                "take_profit_2": result.take_profit_2,
                "tp1_exit_pct": tier.tp1_exit_pct,
                "size": result.size,
                "leverage": tier.leverage,
                "client_oid": client_oid,
                "entry_mode": self.config.trading.entry_mode,
                "placed_at": placed_at,
                "expires_at": expires_at,
            }
            self._save_pending_orders()

            self._log_decision({
                "action": (f"PLACE_{side.upper()}_MAKER"
                           if self.config.trading.entry_mode == "maker"
                           else f"TRADE_{side.upper()}"),
                "order_id": result.order_id,
                "client_oid": client_oid,
                "bar": self._candidate_analysis_bar,
                "entry": targets.entry,
                "sl": targets.stop_loss,
                "tp1": targets.take_profit_1,
                "tp2": targets.take_profit_2,
                "size": result.size,
                "margin": margin,
                "risk_pct": risk_pct,
                "leverage": tier.leverage,
                "score": decision.scoring_result.raw_score,
                "confidence": decision.scoring_result.confidence,
                "loss_penalty": self._current_loss_penalty,
                "equity": equity,
                "available_balance": balance,
                "realized_balance": realized_balance,
                "peak_balance": peak_balance,
            })

        except SafetyViolation as e:
            self._log(f"SAFETY VIOLATION: {e}")
        except Exception as e:
            self._log(f"Trade execution error: {e}")

    def check_positions(self) -> None:
        """Check and manage existing positions."""
        self._reconcile_pending_orders()
        try:
            if not self.exchange._dry_run and not self._startup_reconciled:
                self.reconcile_startup()
            if not self.exchange._dry_run:
                self._reconcile_lot_lifecycle()
            positions = self.exchange.get_positions(self.config.trading.symbol)
            global_positions = self.exchange.get_positions()
            equity = self.exchange.get_account_equity()
            realized_balance = equity - sum(
                position.unrealized_pnl for position in global_positions
            )
            self._peak_balance = self._live_state.update_peak(realized_balance)
            self._heartbeat(equity, realized_balance, len(positions))
            self._maybe_expire_lots()
            if not positions:
                return

            for pos in positions:
                self._log(
                    f"Position: {pos.side} {pos.size} @ ${pos.entry_price:,.2f} | "
                    f"Unrealized PnL: ${pos.unrealized_pnl:,.2f}"
                )
                self._maybe_trail_stop(pos)

        except Exception as e:
            self._log(f"Position check error: {e}")

    def _heartbeat(self, equity: float, realized_balance: float,
                   position_count: int) -> None:
        """Structured liveness/equity record on every position check.

        This is the evaluation stream's account snapshot AND the operational
        heartbeat: its absence from decisions-*.jsonl signals a dead process,
        and it carries a disk-space check so logging/state can't silently fill
        the volume.
        """
        try:
            disk_free_mb = shutil.disk_usage(self._log_dir).free // (1024 * 1024)
        except OSError:
            disk_free_mb = -1
        if 0 <= disk_free_mb < 200:
            self._log(f"ERROR: low disk space — {disk_free_mb} MB free in {self._log_dir}")
        counters = self._risk_state()
        self._log_decision({
            "action": "HEARTBEAT",
            "equity": equity,
            "realized_balance": realized_balance,
            "peak_balance": self._live_state.peak_balance,
            "open_lots": len(self._lots_for_symbol(self.config.trading.symbol)),
            "pending_orders": len(self._pending_orders),
            "positions": position_count,
            "cooldown_remaining": counters["cooldown_remaining"],
            "consecutive_losses": counters["consecutive_losses"],
            "disk_free_mb": disk_free_mb,
        })

    def _maybe_expire_lots(self) -> None:
        """Force-close lots older than risk_management.max_holding_hours.

        Mirrors the backtest's time_expired exit (bar-counted, so the deadline is
        max_holding_hours floored to whole primary bars). Disabled when 0.
        """
        max_hours = self.config.risk_management.max_holding_hours
        if max_hours <= 0:
            return
        tf_hours = timeframe_hours(self.config.trading.primary_timeframe)
        max_bars = int(max_hours // tf_hours)
        if max_bars <= 0:
            return
        deadline_ms = max_bars * tf_hours * 3_600_000
        now_ms = time.time() * 1000
        for lot_id, lot in self._lots_for_symbol(self.config.trading.symbol):
            if lot.get("lifecycle") == "closing":
                continue
            filled_ms = float(lot.get("filled_at_ms") or 0)
            if filled_ms <= 0 or now_ms - filled_ms < deadline_ms:
                continue
            side = "long" if lot.get("direction") == "LONG" else "short"
            size = float(lot.get("remaining_size") or 0)
            if size <= 0:
                continue
            try:
                close_client_oid = self._plan_client_oid(lot_id, "time-expired-close")
                response = self.exchange.close_position(
                    lot["symbol"], side, size, client_oid=close_client_oid,
                )
                close_order_id = str((response.get("data", {}) or {}).get("orderId", ""))
                lot["lifecycle"] = "closing"
                lot["close_reason"] = "time_expired"
                lot["close_client_oid"] = close_client_oid
                lot["close_order_id"] = close_order_id
                self._save_live_state()
                self._log(f"TIME EXPIRED — closing lot {lot_id} ({size} after {max_bars} bars)")
                self._log_decision({
                    "action": "TIME_EXPIRED_CLOSE", "lot_id": lot_id,
                    "size": size, "close_order_id": close_order_id,
                })
            except Exception as e:
                self._log(f"ERROR closing time-expired lot {lot_id}: {e}")

    def _maybe_trail_stop(self, pos) -> None:
        """Ratchet the position's stop — ONLY on completed primary-timeframe bars.

        ⚠️ CADENCE IS THE STRATEGY. The backtested edge ratchets the trailing stop
        once per COMPLETED primary bar (4h), using that bar's favorable extreme; the
        stop stays fixed intrabar (the exchange triggers it if touched). Ratcheting
        on every 15-min position check using the current price tightens the stop ~16×
        more often and chokes winners on noise — an honest sub-bar backtest showed it
        destroys the edge (84× → 5× over 2021-2025, and NO wider callback recovers
        it). Do not "improve" this back to continuous trailing.
        """
        trailing = self.config.trading.trailing_stop
        if not trailing.enabled:
            return
        if pos.size <= 0:
            return

        tracked_lots = self._lots_for_symbol(pos.symbol, pos.side.lower())
        if not tracked_lots:
            return  # We didn't open this position (or startup reconciliation failed).

        expected_bar = str(latest_completed_bar_open(
            self.config.trading.primary_timeframe
        ))
        eligible = []
        for lot_id, tracked in tracked_lots:
            if not tracked.get("last_trail_bar"):
                tracked["last_trail_bar"] = expected_bar
                self._save_live_state()
                self._log(
                    f"Trailing initialized for {lot_id}; waiting for the next completed primary bar"
                )
                continue
            if tracked.get("last_trail_bar") != expected_bar:
                eligible.append((lot_id, tracked))
        if not eligible:
            return

        # Fetch the last COMPLETED primary-timeframe candle; only ratchet when a new
        # one has closed since the last update (bar-close cadence, like the backtest).
        tf = self.config.trading.primary_timeframe
        try:
            ds = self.config.data_source
            clear_cache()  # obtain the final OHLC of the completed trail bar
            data_by_tf = fetch_multi_timeframe(
                symbol=ds.exchange_symbol, timeframes=[tf],
                source=ds.source, market=ds.market,
            )
            frozen, _primary_bar = completed_market_snapshot(data_by_tf, tf)
            df = frozen.get(tf)
        except Exception as e:
            self._log(f"Trailing: could not fetch {tf} candles: {e}")
            return
        if df is None or df.empty:
            return
        last = df.iloc[-1]
        last_ts = df.index[-1]

        for lot_id, tracked in eligible:
            if tracked.get("last_trail_bar") == str(last_ts):
                continue
            favorable = (
                float(last["High"]) if tracked["direction"] == "LONG"
                else float(last["Low"])
            )
            new_sl = compute_trailing_stop(
                direction=tracked["direction"],
                entry_price=tracked["entry"],
                favorable_extreme=favorable,
                current_sl=tracked["current_sl"],
                activation_pct=trailing.activation_pct,
                callback_pct=trailing.callback_pct,
            )
            if new_sl is None:
                tracked["last_trail_bar"] = str(last_ts)
                self._save_live_state()
                continue
            plan_ids = tracked.get("plan_ids", {})
            plan_id = plan_ids.get("sl") or tracked.get("stop_plan_id")
            try:
                kwargs = {
                    "symbol": pos.symbol,
                    "hold_side": pos.side,
                    "size": float(tracked.get("remaining_size") or pos.size),
                    "new_sl": new_sl,
                    "plan_order_id": plan_id,
                }
                if plan_ids:
                    kwargs["position_level"] = False
                self.exchange.modify_stop_loss(**kwargs)
                tracked["current_sl"] = new_sl
                tracked["last_trail_bar"] = str(last_ts)
                self._save_live_state()
                self._log(
                    f"Trailing stop moved to ${new_sl:,.2f} ({tracked['direction']}, lot {lot_id})"
                )
                self._log_decision({
                    "action": "TRAIL_RATCHET", "lot_id": lot_id,
                    "new_sl": new_sl, "bar": str(last_ts),
                    "favorable_extreme": favorable,
                    "direction": tracked["direction"],
                    "entry": tracked.get("entry"),
                })
            except Exception as e:
                self._log(f"Failed to update trailing stop for lot {lot_id}: {e}")

    def run_cycle(self) -> None:
        """Run one full analysis + execution cycle."""
        self._log("=" * 50)
        self._log("Starting analysis cycle")

        if not self.exchange._dry_run and not self._startup_reconciled:
            self.reconcile_startup()

        self._reconcile_pending_orders()
        if not self.exchange._dry_run:
            self._reconcile_lot_lifecycle()

        decision = self.analyze_market()
        if decision:
            analysis_bar = self._candidate_analysis_bar
            if analysis_bar is None or self._claim_analysis_bar(analysis_bar):
                self.execute_decision(decision)
            else:
                self._log(f"Primary candle {analysis_bar} was already claimed")

        self._log("Cycle complete")
        self._log("")

    def start(self) -> None:
        """Start the scheduled trading loop."""
        interval = self.config.scheduling.interval_minutes
        pos_interval = self.config.scheduling.check_positions_interval_minutes

        # One live process per exchange account — the shared caps, DD throttle,
        # and lot lifecycle all assume a single writer.
        self._lock_handle = acquire_account_lock(self.config.bitget)
        try:
            # Paper/live is never allowed to degrade silently into credential-free
            # dry run; a startup reconciliation failure must abort loudly here.
            self.reconcile_startup()
            poll_interval = 1
            self._log(
                f"Starting scheduler — completed-bar analysis poll every {poll_interval}min "
                f"(configured cadence {interval}min), position checks every {pos_interval}min"
            )

            # Run immediately
            self.run_cycle()

            # Schedule recurring
            schedule.every(poll_interval).minutes.do(self.run_cycle)
            schedule.every(pos_interval).minutes.do(self.check_positions)

            while True:
                try:
                    schedule.run_pending()
                except Exception as e:  # a job crash must not kill the loop
                    self._log(f"ERROR: scheduled job crashed: {e}")
                time.sleep(1)
        except KeyboardInterrupt:
            self._log("Scheduler stopped by user")
        finally:
            release_account_lock(self._lock_handle)
            self._lock_handle = None
