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
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import schedule

from llm_trading_bot.config import AppConfig
from llm_trading_bot.data import configure_cache, fetch_multi_timeframe
from llm_trading_bot.exchange import BitgetClient, SafetyViolation
from llm_trading_bot.openwebui_client import run_consensus
from llm_trading_bot.routing import RoutingDecision, route_signal
from llm_trading_bot.scoring import Direction, SignalStrength, calculate_indicators
from llm_trading_bot.trailing import compute_trailing_stop


class TradingScheduler:
    """
    Main automation controller.

    Runs on a schedule:
    1. Fetch market data
    2. Score and route signal
    3. Execute trade (if applicable)
    4. Monitor existing positions
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.exchange = BitgetClient(config.bitget)
        self.decision_log: list[dict] = []
        self._log_dir = Path("logs")
        self._log_dir.mkdir(exist_ok=True)

        # Per-symbol trade context for live trailing stops:
        # {symbol: {"direction": "LONG"|"SHORT", "entry": float, "current_sl": float}}
        self._tracked_trades: dict[str, dict] = {}

        configure_cache(config.data_cache.ttl_seconds)

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
        self._log("Fetching market data...")

        try:
            ds = self.config.data_source
            symbol = ds.exchange_symbol if ds.source != "yfinance" else self.config.trading.yfinance_symbol
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

        self._log(
            f"Signal: {decision.signal_strength.value} | "
            f"Direction: {decision.scoring_result.direction.value} | "
            f"Score: {decision.scoring_result.raw_score:+.1f} | "
            f"Confidence: {decision.scoring_result.confidence:.0f}%"
        )

        return decision

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
        """Execute a trade via the exchange."""
        if not decision.targets:
            self._log("No targets calculated — cannot trade")
            return

        targets = decision.targets
        tier = self.config.trading.active_leverage_tier
        ps = self.config.position_sizing
        side = "buy" if targets.direction == Direction.BULLISH else "sell"
        want_side = "long" if targets.direction == Direction.BULLISH else "short"

        # Entry slots: up to max_positions concurrent SAME-direction positions
        # (pyramiding); never stack against an opposite-direction position.
        try:
            positions = self.exchange.get_positions(self.config.trading.symbol)
            if len(positions) >= ps.max_positions:
                self._log(f"Already have {len(positions)}/{ps.max_positions} position(s) — skipping")
                return
            if any(p.side.lower() != want_side for p in positions):
                self._log("Open position in the opposite direction — not stacking, skipping")
                return
        except Exception as e:
            self._log(f"Warning: Could not check positions: {e}")

        # Risk-based position sizing: commit min(balance * risk_pct, max_usd) as margin,
        # leverage it up to the notional, then convert to base-currency size at entry.
        balance = self.exchange.get_available_balance()
        risk_pct = ps.risk_pct_per_trade
        # Conviction sizing: scale risk with signal strength (mirrors backtest engine)
        if ps.conviction_exponent > 0 and tier.strong_threshold > 0:
            m = (abs(decision.scoring_result.raw_score) / tier.strong_threshold) ** ps.conviction_exponent
            risk_pct *= max(0.5, min(1.5, m))
        margin = min(balance * risk_pct, ps.max_position_usd)
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
            result = self.exchange.place_order(
                symbol=self.config.trading.symbol,
                side=side,
                size=size,
                targets=targets,
                leverage=tier.leverage,
            )
            self._log(f"Order placed: {result.order_id}")

            # Track the trade so check_positions can trail its stop.
            self._tracked_trades[self.config.trading.symbol] = {
                "direction": "LONG" if targets.direction == Direction.BULLISH else "SHORT",
                "entry": targets.entry,
                "current_sl": targets.stop_loss,
            }

            self._log_decision({
                "action": f"TRADE_{side.upper()}",
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
        try:
            positions = self.exchange.get_positions(self.config.trading.symbol)
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
        """Move the position's stop loss up (long) / down (short) if trailing is enabled."""
        trailing = self.config.trading.trailing_stop
        if not trailing.enabled:
            return

        tracked = self._tracked_trades.get(pos.symbol)
        if not tracked:
            return  # We didn't open this position (or lost context after a restart).

        # Derive the current price from the position's unrealized PnL:
        #   long:  unrealized = (price - entry) * size  ->  price = entry + unrealized/size
        #   short: unrealized = (entry - price) * size  ->  price = entry - unrealized/size
        if pos.size <= 0:
            return
        if tracked["direction"] == "LONG":
            current_price = pos.entry_price + pos.unrealized_pnl / pos.size
        else:
            current_price = pos.entry_price - pos.unrealized_pnl / pos.size

        new_sl = compute_trailing_stop(
            direction=tracked["direction"],
            entry_price=tracked["entry"],
            favorable_extreme=current_price,
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
            self._log(f"Trailing stop moved to ${new_sl:,.2f} ({tracked['direction']})")
        except Exception as e:
            self._log(f"Failed to update trailing stop: {e}")

    def run_cycle(self) -> None:
        """Run one full analysis + execution cycle."""
        self._log("=" * 50)
        self._log("Starting analysis cycle")

        decision = self.analyze_market()
        if decision:
            self.execute_decision(decision)

        self._log("Cycle complete")
        self._log("")

    def start(self) -> None:
        """Start the scheduled trading loop."""
        interval = self.config.scheduling.interval_minutes
        pos_interval = self.config.scheduling.check_positions_interval_minutes

        self._log(f"Starting scheduler — analysis every {interval}min, position checks every {pos_interval}min")

        # Run immediately
        self.run_cycle()

        # Schedule recurring
        schedule.every(interval).minutes.do(self.run_cycle)
        schedule.every(pos_interval).minutes.do(self.check_positions)

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self._log("Scheduler stopped by user")
