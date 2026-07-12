"""
Backtesting engine — replay historical candles without lookahead bias.

Features:
- Multi-timeframe analysis per bar
- Proper warmup period (indicators calculated before test period)
- Partial exits at TP1, full exit at TP2 or SL
- Trailing stop support
- Fee-aware portfolio simulation
- No future data leakage: each bar only sees past data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from llm_trading_bot.config import AppConfig
from llm_trading_bot.portfolio import Portfolio, PortfolioStats, Trade
from llm_trading_bot.trailing import compute_trailing_stop
from llm_trading_bot.scoring import (
    Direction,
    IndicatorSet,
    MarketRegime,
    SignalStrength,
    TradeTargets,
    apply_pre_trade_filters,
    calculate_indicators,
    calculate_targets,
    compute_composite_score,
    detect_market_regime,
)


@dataclass
class BacktestBar:
    """One bar of the backtest with its state."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    direction: Optional[str] = None
    signal: Optional[str] = None
    score: Optional[float] = None
    trade_action: Optional[str] = None


@dataclass
class BacktestResult:
    """Complete backtest output."""
    bars: list[BacktestBar] = field(default_factory=list)
    stats: Optional[PortfolioStats] = None
    portfolio: Optional[Portfolio] = None
    config_summary: dict = field(default_factory=dict)
    decision_log: list[dict] = field(default_factory=list)


class BacktestEngine:
    """
    Bar-by-bar backtesting engine.

    IMPORTANT: No lookahead bias.
    - At each bar, only data UP TO AND INCLUDING that bar is visible.
    - Warmup data is loaded BEFORE the test period starts.
    - Indicators are recalculated each bar using only past data.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        bt_cfg = config.backtesting
        risk_cfg = config.risk_management
        self.portfolio = Portfolio(
            initial_balance=bt_cfg.initial_balance,
            maker_fee=config.fees.maker,
            taker_fee=config.fees.taker,
            default_order_type=config.fees.default_order_type,
            use_maker_fee_for_tp=risk_cfg.use_maker_fee_for_tp,
        )
        self.tier = config.trading.active_leverage_tier
        self.enable_partial = bt_cfg.enable_partial_exits
        self.enable_trailing = bt_cfg.enable_trailing_stops
        self.trailing_config = config.trading.trailing_stop
        self.warmup = bt_cfg.warmup_periods
        self.decision_log: list[dict] = []

        # Risk management state (imported from predecessor project)
        self.risk_cfg = risk_cfg
        self.consecutive_losses = 0
        self.candles_since_last_loss = 999  # large initial = no penalty
        self.cooldown_remaining = 0  # candles to skip after SL

    def _check_exits(self, trade: Trade, bar_high: float, bar_low: float,
                     bar_close: float, bar_time: str) -> None:
        """
        Check if any open trade should be exited based on the current bar.
        Uses high/low to check SL/TP hits — assumes worst-case execution order:
        SL checked first (conservative assumption).
        Also checks max holding time (force close after N hours).
        """
        if not trade.is_open:
            return

        trade.bars_held += 1
        is_long = trade.direction == "LONG"

        # Check max holding time first (force close on current bar close)
        max_hours = self.risk_cfg.max_holding_hours
        if max_hours > 0:
            # Convert hours to bars based on primary timeframe
            tf = self.config.trading.primary_timeframe
            hours_per_bar = {"1h": 1, "4h": 4, "1d": 24}.get(tf, 4)
            max_bars = max_hours // hours_per_bar
            if trade.bars_held >= max_bars:
                self.portfolio.close_trade(trade, bar_close, bar_time, "time_expired")
                print(f"    << TIME_EXPIRED {trade.direction} after {trade.bars_held} bars @ ${bar_close:,.2f} | PnL: ${trade.net_pnl:+,.2f}")
                self.decision_log.append({
                    "time": bar_time, "action": "TIME_EXPIRED",
                    "price": bar_close, "trade_id": trade.trade_id,
                    "pnl": trade.net_pnl, "bars_held": trade.bars_held,
                })
                self._on_trade_closed(trade)
                return

        # Check Stop Loss first (conservative)
        sl_hit = (bar_low <= trade.stop_loss) if is_long else (bar_high >= trade.stop_loss)
        if sl_hit:
            self.portfolio.close_trade(trade, trade.stop_loss, bar_time, "sl")
            print(f"    << SL_HIT {trade.direction} @ ${trade.stop_loss:,.2f} | PnL: ${trade.net_pnl:+,.2f}")
            self.decision_log.append({
                "time": bar_time, "action": "SL_HIT",
                "price": trade.stop_loss, "trade_id": trade.trade_id,
                "pnl": trade.net_pnl,
            })
            self._on_trade_closed(trade)
            return

        # Check TP1 (partial exit)
        if self.enable_partial and not trade.partial_exits:
            tp1_hit = (bar_high >= trade.take_profit_1) if is_long else (bar_low <= trade.take_profit_1)
            if tp1_hit:
                pnl = self.portfolio.partial_exit(
                    trade, trade.take_profit_1, bar_time,
                    trade.tp1_exit_pct, "tp1"
                )
                print(f"    << TP1_PARTIAL {trade.direction} @ ${trade.take_profit_1:,.2f} | Partial PnL: ${pnl:+,.2f}")
                self.decision_log.append({
                    "time": bar_time, "action": "TP1_PARTIAL",
                    "price": trade.take_profit_1, "trade_id": trade.trade_id,
                    "pnl": pnl, "remaining_pct": 1 - trade.tp1_exit_pct,
                })

                # If trailing stops enabled, move SL to entry (breakeven)
                if self.enable_trailing:
                    trade.stop_loss = trade.entry_price

        # Check TP2 (full exit of remaining)
        if trade.is_open:
            tp2_hit = (bar_high >= trade.take_profit_2) if is_long else (bar_low <= trade.take_profit_2)
            if tp2_hit:
                self.portfolio.close_trade(trade, trade.take_profit_2, bar_time, "tp2")
                print(f"    << TP2_HIT {trade.direction} @ ${trade.take_profit_2:,.2f} | PnL: ${trade.net_pnl:+,.2f}")
                self.decision_log.append({
                    "time": bar_time, "action": "TP2_HIT",
                    "price": trade.take_profit_2, "trade_id": trade.trade_id,
                    "pnl": trade.net_pnl,
                })
                self._on_trade_closed(trade)

    def _on_trade_closed(self, trade: Trade) -> None:
        """
        Update consecutive loss tracking and cooldown after a trade closes.
        Imported from predecessor project's risk management system.
        """
        if trade.net_pnl <= 0:
            self.consecutive_losses += 1
            self.candles_since_last_loss = 0
            # Activate cooldown after SL hits (not after time_expired)
            if trade.exit_reason in ("sl", "trailing_stop"):
                self.cooldown_remaining = self.risk_cfg.cooldown_candles_after_sl
        else:
            # Win resets consecutive losses
            self.consecutive_losses = 0
            self.candles_since_last_loss = 999

    def _get_loss_penalty(self) -> float:
        """
        Calculate how much to raise the entry threshold based on consecutive losses.
        Penalty decays after N candles without a new loss.
        """
        if self.consecutive_losses == 0:
            return 0.0

        base_penalty = min(
            self.consecutive_losses * self.risk_cfg.consecutive_loss_penalty,
            self.risk_cfg.max_consecutive_loss_penalty,
        )

        # Apply time decay: reduce penalty linearly after decay_candles
        decay = self.risk_cfg.loss_penalty_decay_candles
        if decay > 0 and self.candles_since_last_loss > decay:
            decay_factor = max(0.0, 1.0 - (self.candles_since_last_loss - decay) / decay)
            return base_penalty * decay_factor

        return base_penalty

    def _update_trailing_stop(self, trade: Trade, bar_high: float, bar_low: float) -> None:
        """Update trailing stop if enabled and activated (shared math with live trading)."""
        if not self.enable_trailing or not self.trailing_config.enabled:
            return
        if not trade.is_open:
            return

        # LONG trails off the bar high, SHORT off the bar low.
        favorable = bar_high if trade.direction == "LONG" else bar_low
        new_sl = compute_trailing_stop(
            direction=trade.direction,
            entry_price=trade.entry_price,
            favorable_extreme=favorable,
            current_sl=trade.stop_loss,
            activation_pct=self.trailing_config.activation_pct,
            callback_pct=self.trailing_config.callback_pct,
        )
        if new_sl is not None:
            trade.stop_loss = new_sl

    def run(
        self,
        data_by_tf: dict[str, pd.DataFrame],
        primary_timeframe: str = "4h",
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            data_by_tf: Dict of timeframe → full OHLCV DataFrame (including warmup)
            primary_timeframe: The timeframe to iterate on

        Returns:
            BacktestResult with all trades, stats, and decision log
        """
        primary_df = data_by_tf[primary_timeframe]
        start_date = pd.to_datetime(self.config.backtesting.start_date)
        end_date = pd.to_datetime(self.config.backtesting.end_date)

        # Align tz-awareness before any comparison
        if hasattr(primary_df.index, 'tz') and primary_df.index.tz is not None:
            start_date = start_date.tz_localize(primary_df.index.tz)
            end_date = end_date.tz_localize(primary_df.index.tz)

        # The data BEFORE start_date is warmup
        test_mask = primary_df.index >= start_date

        if end_date is not None:
            test_mask = test_mask & (primary_df.index <= end_date)

        test_indices = primary_df.index[test_mask]

        if len(test_indices) == 0:
            print("Warning: No bars found in test period")
            return BacktestResult(stats=self.portfolio.compute_stats(), portfolio=self.portfolio)

        bars: list[BacktestBar] = []
        scoring_cfg = self.config.scoring

        print(f"Backtesting {len(test_indices)} bars from {test_indices[0]} to {test_indices[-1]}")
        print(f"Starting balance: ${self.portfolio.initial_balance:,.2f}")
        print(f"Leverage: {self.tier.leverage}x | Fees: {self.config.fees.active_fee_rate*100:.2f}%")
        print()

        total_bars = len(test_indices)
        progress_step = max(1, total_bars // 10)  # Print every ~10%

        for i, idx in enumerate(test_indices):
            # Progress update
            if i % progress_step == 0 or i == total_bars - 1:
                pct = (i + 1) / total_bars * 100
                bal = self.portfolio.balance
                open_ct = len(self.portfolio.open_trades)
                closed_ct = len(self.portfolio.trades) - open_ct
                bar_date = str(idx)[:10]
                print(f"  [{pct:5.1f}%] Bar {i+1}/{total_bars} | {bar_date} | Balance: ${bal:,.2f} | Open: {open_ct} | Closed: {closed_ct}")

            bar_idx = primary_df.index.get_loc(idx)
            if bar_idx < self.warmup:
                continue  # Still in warmup

            # Slice data up to current bar (NO LOOKAHEAD)
            current_primary = primary_df.iloc[:bar_idx + 1]
            bar = primary_df.iloc[bar_idx]

            bar_time = str(idx)
            bar_high = float(bar["High"])
            bar_low = float(bar["Low"])
            bar_close = float(bar["Close"])

            # 1. Check exits on existing positions FIRST (on this bar's high/low).
            #    Intrabar path is unknown, so we assume the ADVERSE extreme is hit before the
            #    favorable one (worst case): exits are checked against the stop as it stands at
            #    the START of the bar, and the trailing stop is only ratcheted AFTERWARDS (using
            #    this bar's favorable extreme) so it can only affect SUBSEQUENT bars. Trailing
            #    first and then checking would optimistically credit an exit near the bar's top.
            for trade in list(self.portfolio.open_trades):
                self._check_exits(trade, bar_high, bar_low, bar_close, bar_time)
                self._update_trailing_stop(trade, bar_high, bar_low)

            # Tick risk-management counters
            self.candles_since_last_loss += 1
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1

            # 2. Calculate indicators on data UP TO this bar
            indicators_by_tf: dict[str, IndicatorSet] = {}

            try:
                primary_ind = calculate_indicators(current_primary, primary_timeframe)
                indicators_by_tf[primary_timeframe] = primary_ind
            except ValueError:
                continue  # Not enough data yet

            # Secondary timeframes: find the latest data up to this timestamp
            for tf, tf_df in data_by_tf.items():
                if tf == primary_timeframe:
                    continue
                try:
                    tf_mask = tf_df.index <= idx
                    tf_slice = tf_df[tf_mask]
                    if len(tf_slice) >= 50:
                        indicators_by_tf[tf] = calculate_indicators(tf_slice, tf)
                except (ValueError, KeyError):
                    pass

            # 3. Score
            result = compute_composite_score(
                indicators_by_tf=indicators_by_tf,
                weights=scoring_cfg.weights,
                primary_timeframe=primary_timeframe,
                confidence_min=scoring_cfg.confidence_min,
                confidence_max=scoring_cfg.confidence_max,
            )

            # 4. Calculate targets (use tier R:R ratios)
            targets = calculate_targets(
                indicators=primary_ind,
                direction=result.direction,
                sl_strategy=self.config.trading.stop_loss_strategy,
                atr_sl_mult=scoring_cfg.atr_sl_multiplier,
                tp1_rr=self.tier.tp1_rr,
                tp2_rr=self.tier.tp2_rr,
            )

            # 5. Signal classification (with consecutive loss penalty)
            abs_score = abs(result.raw_score)
            loss_penalty = self._get_loss_penalty()
            effective_marginal_low = self.tier.marginal_threshold_low + loss_penalty
            effective_strong = self.tier.strong_threshold + loss_penalty

            if abs_score >= effective_strong:
                signal = SignalStrength.STRONG
            elif abs_score >= effective_marginal_low:
                signal = SignalStrength.MARGINAL
            else:
                signal = SignalStrength.WAIT

            # 5.5 Opposite-signal exit: the composite flipped hard against open positions
            opp_threshold = self.risk_cfg.opposite_exit_threshold
            if (opp_threshold > 0 and self.portfolio.open_trades
                    and result.direction != Direction.NEUTRAL and abs_score >= opp_threshold):
                want = "LONG" if result.direction == Direction.BULLISH else "SHORT"
                for trade in list(self.portfolio.open_trades):
                    if trade.direction != want:
                        self.portfolio.close_trade(trade, bar_close, bar_time, "signal_flip")
                        print(f"    << SIGNAL_FLIP {trade.direction} @ ${bar_close:,.2f} | PnL: ${trade.net_pnl:+,.2f}")
                        self.decision_log.append({
                            "time": bar_time, "action": "SIGNAL_FLIP",
                            "price": bar_close, "trade_id": trade.trade_id,
                            "pnl": trade.net_pnl, "flip_score": result.raw_score,
                        })
                        self._on_trade_closed(trade)

            # 6. Pre-trade filters (enhanced with category agreement + regime)
            trade_action = None
            if signal in (SignalStrength.STRONG, SignalStrength.MARGINAL) and targets:
                # Cooldown check — skip entries for N candles after SL
                if self.cooldown_remaining > 0:
                    pass  # blocked by cooldown
                else:
                    filter_failures = apply_pre_trade_filters(
                        indicators=primary_ind,
                        targets=targets,
                        min_adx=self.config.filters.min_adx,
                        min_volatility_pct=self.config.filters.min_volatility_pct,
                        fee_rate=self.config.fees.active_fee_rate,
                        leverage=self.tier.leverage,
                        check_profit_after_fees=self.config.filters.min_profit_after_fees,
                        category_scores=result.category_scores,
                        direction=result.direction,
                        min_category_agreement=self.config.filters.min_category_agreement,
                        require_trend_momentum_agree=self.config.filters.require_trend_momentum_agree,
                        skip_choppy_regime=self.config.filters.skip_choppy_regime,
                        skip_volatile_regime=self.config.filters.skip_volatile_regime,
                    )

                    # Entry slot: up to max_positions concurrent SAME-direction trades
                    # (pyramiding); opposite-direction entries are never stacked.
                    ps_cfg = self.config.position_sizing
                    direction_str = "LONG" if result.direction == Direction.BULLISH else "SHORT"
                    open_now = self.portfolio.open_trades
                    can_enter = (len(open_now) < ps_cfg.max_positions
                                 and all(t.direction == direction_str for t in open_now))

                    if not filter_failures and can_enter:
                        # For backtesting, MARGINAL signals are treated as trades too
                        # (we can't call the LLM in a backtest)
                        # Conviction sizing: scale risk with signal strength (off when exponent=0)
                        risk_eff = ps_cfg.risk_pct_per_trade
                        if ps_cfg.conviction_exponent > 0 and effective_strong > 0:
                            m = (abs_score / effective_strong) ** ps_cfg.conviction_exponent
                            risk_eff *= max(0.5, min(1.5, m))
                        trade = self.portfolio.open_trade(
                            direction=direction_str,
                            entry_price=bar_close,
                            entry_time=bar_time,
                            stop_loss=targets.stop_loss,
                            take_profit_1=targets.take_profit_1,
                            take_profit_2=targets.take_profit_2,
                            leverage=self.tier.leverage,
                            risk_pct=risk_eff,
                            tp1_exit_pct=self.tier.tp1_exit_pct,
                        )
                        trade_action = f"OPEN_{direction_str}"
                        print(f"    >> {trade_action} @ ${bar_close:,.2f} | Score: {result.raw_score:+.1f} | SL: ${targets.stop_loss:,.2f} | TP1: ${targets.take_profit_1:,.2f} | TP2: ${targets.take_profit_2:,.2f}")

                        self.decision_log.append({
                            "time": bar_time, "action": trade_action,
                            "price": bar_close, "score": result.raw_score,
                            "confidence": result.confidence,
                            "signal": signal.value,
                            "sl": targets.stop_loss, "tp1": targets.take_profit_1,
                            "tp2": targets.take_profit_2,
                        })

            # Record bar
            backtest_bar = BacktestBar(
                timestamp=bar_time,
                open=float(bar["Open"]),
                high=bar_high,
                low=bar_low,
                close=bar_close,
                volume=float(bar["Volume"]),
                direction=result.direction.value if result.direction != Direction.NEUTRAL else None,
                signal=signal.value,
                score=result.raw_score,
                trade_action=trade_action,
            )
            bars.append(backtest_bar)

            # Periodic snapshot
            if i % 10 == 0:
                self.portfolio.record_snapshot(bar_time, bar_close)

        # Close any remaining open positions at the last bar's close
        if bars:
            last_close = bars[-1].close
            last_time = bars[-1].timestamp
            for trade in list(self.portfolio.open_trades):
                self.portfolio.close_trade(trade, last_close, last_time, "end_of_backtest")

        # Final snapshot
        if bars:
            self.portfolio.record_snapshot(bars[-1].timestamp, bars[-1].close)

        stats = self.portfolio.compute_stats()

        print(f"\n{'='*50}")
        print(f"Backtest Complete")
        print(f"{'='*50}")
        print(f"Total trades: {stats.total_trades}")
        print(f"Win rate: {stats.win_rate:.1f}%")
        print(f"Total net PnL: ${stats.total_net_pnl:,.2f}")
        print(f"Total fees paid: ${stats.total_fees:,.2f}")
        print(f"Max drawdown: {stats.max_drawdown_pct:.1f}%")
        print(f"Profit factor: {stats.profit_factor:.2f}")
        print(f"Final balance: ${stats.final_balance:,.2f} ({stats.total_return_pct:+.1f}%)")

        return BacktestResult(
            bars=bars,
            stats=stats,
            portfolio=self.portfolio,
            config_summary={
                "symbol": self.config.trading.yfinance_symbol,
                "primary_timeframe": primary_timeframe,
                "leverage": self.tier.leverage,
                "active_tier": self.config.trading.active_tier,
                "start_date": self.config.backtesting.start_date,
                "end_date": self.config.backtesting.end_date,
                "initial_balance": self.config.backtesting.initial_balance,
                "fee_rate": self.config.fees.active_fee_rate,
                "sl_strategy": self.config.trading.stop_loss_strategy,
            },
            decision_log=self.decision_log,
        )
