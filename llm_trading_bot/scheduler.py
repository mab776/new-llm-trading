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
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import schedule

from llm_trading_bot.config import AppConfig
from llm_trading_bot.data import clear_cache, configure_cache, fetch_multi_timeframe
from llm_trading_bot.exchange import BitgetClient, SafetyViolation
from llm_trading_bot.exposure import (
    anti_martingale_multiplier, cap_risk_pct, outcome_streak,
)
from llm_trading_bot.live_state import SharedLiveState
from llm_trading_bot.openwebui_client import run_consensus
from llm_trading_bot.routing import RoutingDecision, route_signal
from llm_trading_bot.scoring import Direction, SignalStrength, calculate_indicators
from llm_trading_bot.trailing import compute_trailing_stop
from llm_trading_bot.timeframes import (
    completed_market_snapshot, latest_completed_bar_open, timeframe_delta,
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

    def _save_live_state(self) -> None:
        state = getattr(self, "_live_state", None)
        if state is not None:
            try:
                state.save()
            except OSError as e:
                self._log(f"Warning: could not persist live state: {e}")

    @staticmethod
    def _timeframe_seconds(timeframe: str) -> int:
        return int(timeframe_delta(timeframe).total_seconds())

    @staticmethod
    def _same_symbol(left: str, right: str) -> bool:
        normalize = lambda value: re.sub(r"[^A-Za-z0-9]", "", value).upper()
        return normalize(left) == normalize(right)

    def _activate_filled_pending(self, order_id: str, detail: dict) -> None:
        pending = self._pending_orders.pop(order_id)
        avg = float(detail.get("priceAvg") or pending["entry"])
        self._tracked_trades[pending["symbol"]] = {
            "direction": pending["direction"],
            "entry": avg,
            "current_sl": pending["stop_loss"],
        }
        self._save_live_state()
        self._log(f"Maker order {order_id} filled @ ${avg:,.2f}")
        self._log_decision({
            "action": "MAKER_FILL", "order_id": order_id,
            "entry": avg, "side": pending["direction"],
        })

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
                        self._pending_orders.pop(order_id, None)
                        self._save_pending_orders()
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

    def _log(self, msg: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(self._log_dir / "trading.log", "a") as f:
            f.write(line + "\n")

    def _log_decision(self, decision: dict) -> None:
        decision["timestamp"] = datetime.now().isoformat()
        self.decision_log.append(decision)
        # Append to persistent log
        log_file = self._log_dir / "decisions.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(decision, default=str) + "\n")

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
        analysis_bar = str(primary_bar)
        if self._live_state.last_analysis_bars.get(symbol_key) == analysis_bar:
            self._log(f"Primary candle {analysis_bar} already analyzed")
            return None

        # Calculate indicators for each timeframe
        indicators_by_tf = {}
        for tf, df in data_by_tf.items():
            try:
                indicators_by_tf[tf] = calculate_indicators(df, tf)
            except Exception as e:
                self._log(f"Warning: Failed to calculate {tf} indicators: {e}")

        if not indicators_by_tf:
            self._log("No indicators calculated")
            return None

        # Route signal
        decision = route_signal(indicators_by_tf, self.config)
        self._candidate_analysis_bar = analysis_bar

        self._log(
            f"Signal: {decision.signal_strength.value} | "
            f"Direction: {decision.scoring_result.direction.value} | "
            f"Score: {decision.scoring_result.raw_score:+.1f} | "
            f"Confidence: {decision.scoring_result.confidence:.0f}%"
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
                    self.exchange.close_position(self.config.trading.symbol, pos.side, pos.size)
                    self._tracked_trades.pop(self.config.trading.symbol, None)
                    self._save_live_state()
                    self._log_decision({
                        "action": "SIGNAL_FLIP_CLOSE",
                        "side": pos.side, "size": pos.size,
                        "score": decision.scoring_result.raw_score,
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
            })
            return

        if decision.signal_strength == SignalStrength.STRONG:
            self._log("STRONG signal — using deterministic template")
            self._log(decision.template_response or "")
            self._execute_trade(decision)
            return

        if decision.signal_strength == SignalStrength.MARGINAL:
            if self.config.openwebui.marginal_execution == "deterministic":
                self._log("MARGINAL signal — deterministic execution (backtest parity)")
                self._execute_trade(decision)
                return
            self._log("MARGINAL signal — querying LLM consensus...")
            consensus = run_consensus(
                config=self.config.openwebui,
                scoring_result=decision.scoring_result,
                targets=decision.targets,
            )

            self._log(f"Consensus: {consensus.decision} ({consensus.agreement_pct:.0f}% agreement)")
            self._log(consensus.reasoning_summary)

            if consensus.decision in ("LONG", "SHORT"):
                # Update direction based on consensus
                if consensus.decision == "LONG":
                    decision.scoring_result.direction = Direction.BULLISH
                else:
                    decision.scoring_result.direction = Direction.BEARISH
                self._execute_trade(decision)
            else:
                self._log("Consensus: WAIT — not trading")
                self._log_decision({
                    "action": "LLM_WAIT",
                    "consensus": consensus.decision,
                    "agreement": consensus.agreement_pct,
                })

    def _execute_trade(self, decision: RoutingDecision) -> None:
        """Atomically check account exposure, size, and place one new order."""
        with self._execution_lock:
            self._execute_trade_locked(decision)

    def _execute_trade_locked(self, decision: RoutingDecision) -> None:
        """Execute a trade via the exchange."""
        if not decision.targets:
            self._log("No targets calculated — cannot trade")
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
            committed = len(positions) + len(pending_for_symbol)
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
            global_committed = len(global_positions) + len(exchange_pending) + len(local_pending)
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
        # Conviction sizing: scale risk with signal strength (mirrors backtest engine)
        if ps.conviction_exponent > 0 and tier.strong_threshold > 0:
            m = (abs(decision.scoring_result.raw_score) / tier.strong_threshold) ** ps.conviction_exponent
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
        risk_pct = cap_risk_pct(
            risk_pct, tier.leverage, equity,
            committed_margin, committed_notional,
            risk_multiplier=ps.portfolio_risk_multiplier,
            max_margin_pct=ps.global_max_margin_pct,
            max_notional_pct=ps.global_max_notional_pct,
        )
        margin = min(equity * risk_pct, balance, ps.max_position_usd)
        size = (margin * tier.leverage) / targets.entry if targets.entry > 0 else 0.0
        if size <= 0:
            self._log(
                f"Computed non-positive size (balance=${balance:,.2f}, margin=${margin:,.2f}) — skipping trade"
            )
            return

        self._log(
            f"Executing {side.upper()} @ ${targets.entry:,.2f} | "
            f"Size: {size:.6f} (margin ${margin:,.2f} @ {tier.leverage}x) | "
            f"SL: ${targets.stop_loss:,.2f} | "
            f"TP1: ${targets.take_profit_1:,.2f} | "
            f"TP2: ${targets.take_profit_2:,.2f}"
        )

        try:
            if self.config.trading.entry_mode == "maker":
                result = self.exchange.place_order(
                    symbol=self.config.trading.symbol, side=side, size=size,
                    targets=targets, leverage=tier.leverage,
                    order_type="limit", price=targets.entry,
                )
            else:
                result = self.exchange.place_order(
                    symbol=self.config.trading.symbol, side=side, size=size,
                    targets=targets, leverage=tier.leverage,
                )
            self._log(f"Order placed: {result.order_id}")

            direction = "LONG" if targets.direction == Direction.BULLISH else "SHORT"
            if self.config.trading.entry_mode == "maker":
                placed_at = time.time()
                tf_seconds = self._timeframe_seconds(
                    self.config.trading.primary_timeframe
                )
                # Expire on the next UTC-aligned primary-bar close, not an
                # arbitrary wall-clock duration after a delayed scheduler tick.
                expires_at = (int(placed_at // tf_seconds) + 1) * tf_seconds
                self._pending_orders[result.order_id] = {
                    "symbol": self.config.trading.symbol,
                    "direction": direction,
                    "entry": targets.entry,
                    "stop_loss": targets.stop_loss,
                    "take_profit_1": targets.take_profit_1,
                    "take_profit_2": targets.take_profit_2,
                    "size": size,
                    "leverage": tier.leverage,
                    "placed_at": placed_at,
                    "expires_at": expires_at,
                }
                self._save_pending_orders()
            else:
                # Market orders are immediately eligible for trailing management.
                self._tracked_trades[self.config.trading.symbol] = {
                    "direction": direction,
                    "entry": targets.entry,
                    "current_sl": targets.stop_loss,
                }
                self._save_live_state()

            self._log_decision({
                "action": (f"PLACE_{side.upper()}_MAKER"
                           if self.config.trading.entry_mode == "maker"
                           else f"TRADE_{side.upper()}"),
                "order_id": result.order_id,
                "entry": targets.entry,
                "sl": targets.stop_loss,
                "tp1": targets.take_profit_1,
                "tp2": targets.take_profit_2,
                "leverage": tier.leverage,
                "score": decision.scoring_result.raw_score,
                "confidence": decision.scoring_result.confidence,
            })

        except SafetyViolation as e:
            self._log(f"SAFETY VIOLATION: {e}")
        except Exception as e:
            self._log(f"Trade execution error: {e}")

    def check_positions(self) -> None:
        """Check and manage existing positions."""
        self._reconcile_pending_orders()
        try:
            positions = self.exchange.get_positions(self.config.trading.symbol)
            global_positions = self.exchange.get_positions()
            equity = self.exchange.get_account_equity()
            realized_balance = equity - sum(
                position.unrealized_pnl for position in global_positions
            )
            self._peak_balance = self._live_state.update_peak(realized_balance)
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

        tracked = self._tracked_trades.get(pos.symbol)
        if not tracked:
            return  # We didn't open this position (or lost context after a restart).
        if pos.size <= 0:
            return

        expected_bar = str(latest_completed_bar_open(
            self.config.trading.primary_timeframe
        ))
        if tracked.get("last_trail_bar") == expected_bar:
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

        if tracked.get("last_trail_bar") == str(last_ts):
            return  # already ratcheted on this bar
        tracked["last_trail_bar"] = str(last_ts)

        favorable = float(last["High"]) if tracked["direction"] == "LONG" else float(last["Low"])
        new_sl = compute_trailing_stop(
            direction=tracked["direction"],
            entry_price=tracked["entry"],
            favorable_extreme=favorable,
            current_sl=tracked["current_sl"],
            activation_pct=trailing.activation_pct,
            callback_pct=trailing.callback_pct,
        )
        if new_sl is None:
            return

        try:
            self.exchange.modify_stop_loss(
                symbol=pos.symbol, hold_side=pos.side, size=pos.size, new_sl=new_sl,
            )
            tracked["current_sl"] = new_sl
            self._save_live_state()
            self._log(f"Trailing stop moved to ${new_sl:,.2f} ({tracked['direction']})")
        except Exception as e:
            self._log(f"Failed to update trailing stop: {e}")

    def run_cycle(self) -> None:
        """Run one full analysis + execution cycle."""
        self._log("=" * 50)
        self._log("Starting analysis cycle")

        self._reconcile_pending_orders()

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

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self._log("Scheduler stopped by user")
