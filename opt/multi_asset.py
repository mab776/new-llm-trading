"""Shared-balance multi-asset fast backtest.

Streams are interleaved by primary-bar timestamp into one fee-aware Portfolio.
Each symbol retains its own entry slots, cooldown, loss penalty, pending maker
order, targets, and trailing state; balance, peak equity, and DD throttle are
portfolio-wide.  Equal timestamps use sorted symbol order for deterministic
capital allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from llm_trading_bot.entry import PendingEntry, maker_limit_touched
from llm_trading_bot.funding import funding_cost
from llm_trading_bot.portfolio import Portfolio
from llm_trading_bot.scoring import (
    Direction, SignalStrength, apply_pre_trade_filters, calculate_targets,
    compute_composite_score,
)
from opt.fastbt import Precomputed, _check_exits


@dataclass
class AssetInput:
    pre: Precomputed
    config: object
    funding_by_pos: Optional[list[float]] = None


@dataclass
class _AssetState:
    item: AssetInput
    index_by_ts: dict[pd.Timestamp, int]
    pending: PendingEntry | None = None
    consecutive_losses: int = 0
    candles_since_loss: int = 999
    cooldown: int = 0
    last_close: float | None = None
    last_time: str | None = None


@dataclass
class MultiAssetResult:
    return_pct: float
    final_balance: float
    trades: int
    win_rate: float
    profit_factor: float
    max_dd_pct: float
    sharpe: float
    per_symbol: dict[str, dict] = field(default_factory=dict)
    portfolio: Portfolio | None = None


def _loss_penalty(state: _AssetState) -> float:
    risk = state.item.config.risk_management
    if state.consecutive_losses == 0:
        return 0.0
    base = min(state.consecutive_losses * risk.consecutive_loss_penalty,
               risk.max_consecutive_loss_penalty)
    decay = risk.loss_penalty_decay_candles
    if decay > 0 and state.candles_since_loss > decay:
        factor = max(0.0, 1.0 - (state.candles_since_loss - decay) / decay)
        return base * factor
    return base


def simulate_multi(
    assets: dict[str, AssetInput], start_date: str, end_date: str,
    *, slip: float = 0.0, model_liquidation: bool = True,
    maintenance_margin: float = 0.005, exit_granularity: str = "primary",
) -> MultiAssetResult:
    """Replay multiple symbols against one compounding balance."""
    if not assets:
        raise ValueError("At least one asset is required")
    first = next(iter(assets.values())).config
    for symbol, item in assets.items():
        cfg = item.config
        if cfg.backtesting.initial_balance != first.backtesting.initial_balance:
            raise ValueError(f"{symbol}: all assets must share initial_balance")
        if (cfg.fees.maker, cfg.fees.taker) != (first.fees.maker, first.fees.taker):
            raise ValueError(f"{symbol}: all assets must share fee rates")

    port = Portfolio(
        initial_balance=first.backtesting.initial_balance,
        maker_fee=first.fees.maker, taker_fee=first.fees.taker,
        default_order_type=first.fees.default_order_type,
        use_maker_fee_for_tp=first.risk_management.use_maker_fee_for_tp,
    )
    sd, ed = pd.Timestamp(start_date), pd.Timestamp(end_date)
    states: dict[str, _AssetState] = {}
    events: set[pd.Timestamp] = set()
    for symbol, item in assets.items():
        idx = pd.DatetimeIndex(item.pre.timestamps)
        if idx.tz is not None and sd.tzinfo is None:
            local_sd, local_ed = sd.tz_localize(idx.tz), ed.tz_localize(idx.tz)
        else:
            local_sd, local_ed = sd, ed
        mapping = {pd.Timestamp(ts): i for i, ts in enumerate(idx)
                   if local_sd <= ts <= local_ed and i >= item.pre.warmup}
        states[symbol] = _AssetState(item=item, index_by_ts=mapping)
        events.update(mapping)

    latest_prices: dict[str, float] = {}
    event_count = 0
    for ts in sorted(events):
        for symbol in sorted(states):
            state = states[symbol]
            if ts not in state.index_by_ts:
                continue
            i = state.index_by_ts[ts]
            pre, cfg = state.item.pre, state.item.config
            prim = pre.primary[i]
            if prim is None:
                continue
            tr, tier = cfg.trading, cfg.trading.active_leverage_tier
            risk, bt, ps, ft, sc = (cfg.risk_management, cfg.backtesting,
                                     cfg.position_sizing, cfg.filters, cfg.scoring)
            bar_high, bar_low, bar_close = prim.high, prim.low, prim.close
            bar_time = str(ts)
            state.last_close, state.last_time = bar_close, bar_time
            latest_prices[symbol] = bar_close
            tf_hours = {"1h": 1, "4h": 4, "1d": 24}.get(tr.primary_timeframe, 4)
            symbol_trades = [t for t in port.open_trades if t.symbol == symbol]

            # Resolve the prior good-for-one-bar maker order before exit checks.
            fresh = None
            fresh_sub_start = 0
            subs = (pre.subbars[i] if exit_granularity == "sub" and pre.subbars else None)
            if state.pending is not None:
                p = state.pending
                touched = False
                if subs:
                    for sub_i, (hi, lo, _cl) in enumerate(subs):
                        if maker_limit_touched(p.direction, p.limit_price, hi, lo):
                            touched, fresh_sub_start = True, sub_i
                            break
                else:
                    touched = maker_limit_touched(
                        p.direction, p.limit_price, bar_high, bar_low
                    )
                if touched:
                    fresh = port.open_trade(
                        p.direction, p.limit_price, bar_time, p.stop_loss,
                        p.take_profit_1, p.take_profit_2, leverage=p.leverage,
                        risk_pct=p.risk_pct, tp1_exit_pct=p.tp1_exit_pct,
                        order_type="maker", symbol=symbol,
                    )
                    fresh._atr_entry = p.atr_at_entry
                    symbol_trades.append(fresh)
                state.pending = None

            strat = {
                "trail_mode": "pct", "trail_act_atr": .5, "trail_cb_atr": .6,
            }
            for trade in list(symbol_trades):
                if subs:
                    start = fresh_sub_start if trade is fresh else 0
                    first_sub = True
                    for hi, lo, close in subs[start:]:
                        if not trade.is_open:
                            break
                        _check_exits(
                            port, trade, hi, lo, close, bar_time, risk, tf_hours,
                            bt.enable_partial_exits, bt.enable_trailing_stops,
                            tr.trailing_stop, slip, model_liquidation,
                            maintenance_margin, strat, count_bar=first_sub,
                        )
                        first_sub = False
                else:
                    _check_exits(
                        port, trade, bar_high, bar_low, bar_close, bar_time,
                        risk, tf_hours, bt.enable_partial_exits,
                        bt.enable_trailing_stops, tr.trailing_stop, slip,
                        model_liquidation, maintenance_margin, strat,
                    )

            # Funding belongs only to this symbol's surviving positions.
            if state.item.funding_by_pos is not None:
                rate_sum = state.item.funding_by_pos[i]
                if rate_sum:
                    for trade in [t for t in port.open_trades if t.symbol == symbol]:
                        port.apply_funding(
                            trade, funding_cost(trade.direction, rate_sum,
                                                trade.remaining_size, bar_close)
                        )

            # Consume only this symbol's close events; leave the rest queued.
            queued = getattr(port, "_risk_events", [])
            mine = [ev for ev in queued if ev.get("symbol", "") == symbol]
            port._risk_events = [ev for ev in queued if ev not in mine]
            for ev in mine:
                if ev["loss"]:
                    state.consecutive_losses += 1
                    state.candles_since_loss = 0
                    if ev["sl"]:
                        state.cooldown = risk.cooldown_candles_after_sl
                else:
                    state.consecutive_losses = 0
                    state.candles_since_loss = 999
            state.candles_since_loss += 1
            if state.cooldown > 0:
                state.cooldown -= 1

            inds = {tr.primary_timeframe: prim}
            inds.update(pre.sec_by_bar[i])
            result = compute_composite_score(
                indicators_by_tf=inds, weights=sc.weights,
                primary_timeframe=tr.primary_timeframe,
                confidence_min=sc.confidence_min, confidence_max=sc.confidence_max,
                scoring_points=getattr(sc, "points", None),
            )
            targets = calculate_targets(
                prim, result.direction, tr.stop_loss_strategy,
                sc.atr_sl_multiplier, tier.tp1_rr, tier.tp2_rr,
            )
            abs_score = abs(result.raw_score)
            penalty = _loss_penalty(state)
            strong, marginal = (tier.strong_threshold + penalty,
                                tier.marginal_threshold_low + penalty)
            signal = (SignalStrength.STRONG if abs_score >= strong else
                      SignalStrength.MARGINAL if abs_score >= marginal else
                      SignalStrength.WAIT)

            open_symbol = [t for t in port.open_trades if t.symbol == symbol]
            if (risk.opposite_exit_threshold > 0 and open_symbol
                    and result.direction != Direction.NEUTRAL
                    and abs_score >= risk.opposite_exit_threshold):
                want = "LONG" if result.direction == Direction.BULLISH else "SHORT"
                for trade in list(open_symbol):
                    if trade.direction != want:
                        fill = (bar_close * (1 - slip) if trade.direction == "LONG"
                                else bar_close * (1 + slip))
                        port.close_trade(trade, fill, bar_time, "signal_flip")
                        if trade.net_pnl <= 0:
                            state.consecutive_losses += 1
                            state.candles_since_loss = 0
                        else:
                            state.consecutive_losses = 0
                            state.candles_since_loss = 999

            if (signal in (SignalStrength.STRONG, SignalStrength.MARGINAL)
                    and targets and state.cooldown <= 0):
                failures = apply_pre_trade_filters(
                    indicators=prim, targets=targets, min_adx=ft.min_adx,
                    min_volatility_pct=ft.min_volatility_pct,
                    fee_rate=cfg.fees.active_fee_rate, leverage=tier.leverage,
                    check_profit_after_fees=ft.min_profit_after_fees,
                    category_scores=result.category_scores, direction=result.direction,
                    min_category_agreement=ft.min_category_agreement,
                    require_trend_momentum_agree=ft.require_trend_momentum_agree,
                    skip_choppy_regime=ft.skip_choppy_regime,
                    skip_volatile_regime=ft.skip_volatile_regime,
                )
                direction = "LONG" if result.direction == Direction.BULLISH else "SHORT"
                open_symbol = [t for t in port.open_trades if t.symbol == symbol]
                slots, throttled = ps.max_positions, False
                if risk.dd_throttle_threshold > 0 and port.peak_balance > 0:
                    dd = (port.peak_balance - port.balance) / port.peak_balance
                    if dd >= risk.dd_throttle_threshold:
                        slots = min(slots, risk.dd_throttle_slots)
                        throttled = True
                committed = len(open_symbol) + (state.pending is not None)
                same_side = all(t.direction == direction for t in open_symbol)
                if not failures and committed < slots and same_side:
                    risk_eff = ps.risk_pct_per_trade
                    if ps.conviction_exponent > 0 and strong > 0:
                        mult = (abs_score / strong) ** ps.conviction_exponent
                        risk_eff *= max(.5, min(1.5, mult))
                    if throttled:
                        risk_eff *= risk.dd_throttle_risk
                    if tr.entry_mode == "maker":
                        state.pending = PendingEntry(
                            direction, bar_close, targets.stop_loss,
                            targets.take_profit_1, targets.take_profit_2,
                            tier.leverage, risk_eff, tier.tp1_exit_pct,
                            prim.atr_14, bar_time,
                        )
                    else:
                        entry = (bar_close * (1 + slip) if direction == "LONG"
                                 else bar_close * (1 - slip))
                        trade = port.open_trade(
                            direction, entry, bar_time, targets.stop_loss,
                            targets.take_profit_1, targets.take_profit_2,
                            leverage=tier.leverage, risk_pct=risk_eff,
                            tp1_exit_pct=tier.tp1_exit_pct, order_type="taker",
                            symbol=symbol,
                        )
                        trade._atr_entry = prim.atr_14

        event_count += 1
        if event_count % 10 == 0:
            port.record_snapshot(str(ts), latest_prices)

    for trade in list(port.open_trades):
        state = states[trade.symbol]
        port.close_trade(trade, state.last_close, state.last_time, "end_of_backtest")
    if events:
        port.record_snapshot(str(max(events)), latest_prices)
    stats = port.compute_stats()
    per_symbol = {}
    for symbol in states:
        trades = [t for t in port.trades if t.symbol == symbol and not t.is_open]
        wins = [t for t in trades if t.net_pnl > 0]
        per_symbol[symbol] = {
            "trades": len(trades),
            "win_rate": (100 * len(wins) / len(trades)) if trades else 0.0,
            "net_pnl": sum(t.net_pnl for t in trades),
            "fees": sum(t.total_fees for t in trades),
            "funding": sum(t.funding_paid for t in trades),
        }
    return MultiAssetResult(
        return_pct=stats.total_return_pct, final_balance=stats.final_balance,
        trades=stats.total_trades, win_rate=stats.win_rate,
        profit_factor=stats.profit_factor, max_dd_pct=stats.max_drawdown_pct,
        sharpe=stats.sharpe_ratio, per_symbol=per_symbol, portfolio=port,
    )
