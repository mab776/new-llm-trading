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
from llm_trading_bot.exposure import (
    anti_martingale_multiplier, cap_risk_pct, update_outcome_streak,
)
from llm_trading_bot.timeframes import timeframe_hours
from llm_trading_bot.funding import funding_cost
from llm_trading_bot.portfolio import Portfolio
from llm_trading_bot.scoring import (
    Direction, SignalStrength, apply_pre_trade_filters, calculate_targets,
    compute_composite_score, detect_market_regime,
)
from opt.fastbt import (
    DEFAULT_STRAT, Precomputed, _check_exits, _ratchet_trailing_stop,
    deterministic_maker_fill, maker_queue_eligible, apply_daily_overlay,
)
from opt.drawdown import EquityPoint


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
    outcome_streak: int = 0
    candles_since_loss: int = 999
    cooldown: int = 0
    last_close: float | None = None
    last_time: str | None = None
    last_raw_score: float | None = None  # newest composite score (rotation support)


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
    equity_curve: list[EquityPoint] = field(default_factory=list)
    portfolio: Portfolio | None = None
    maker_orders: int = 0
    maker_touches: int = 0
    maker_queue_eligible: int = 0
    maker_fills: int = 0
    rotations: int = 0


def _mark_to_market_equity(port: Portfolio,
                           latest_prices: dict[str, float]) -> float:
    """Return current equity without mutating portfolio risk state."""
    unrealized = 0.0
    for trade in port.open_trades:
        price = latest_prices.get(trade.symbol, trade.entry_price)
        if trade.direction == "LONG":
            unrealized += (price - trade.entry_price) * trade.remaining_size
        else:
            unrealized += (trade.entry_price - price) * trade.remaining_size
    return port.balance + unrealized


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


def committed_exposure(port: Portfolio, pending: list[PendingEntry],
                       balance: float) -> tuple[int, float, float]:
    """Return global committed slots, isolated margin, and entry notional.

    Resting maker entries count because they can fill without another strategy
    decision. Pending size is estimated from the risk percentage locked in when
    the order was placed, matching the fast backtest's fill sizing convention.
    """
    slots = len(port.open_trades) + len(pending)
    margin = sum(
        t.remaining_size * t.entry_price / t.leverage
        for t in port.open_trades if t.leverage > 0
    )
    notional = sum(t.remaining_size * t.entry_price for t in port.open_trades)
    for order in pending:
        order_risk = order.risk_pct
        if order.max_margin_pct is not None:
            order_risk = min(order_risk, order.max_margin_pct)
        order_margin = balance * order_risk
        margin += order_margin
        notional += order_margin * order.leverage
    return slots, margin, notional


def apply_exposure_caps(risk_pct: float, leverage: int, balance: float,
                        committed_margin: float, committed_notional: float,
                        strategy: dict) -> float:
    """Scale a proposed order to remaining ex-ante portfolio capacity."""
    return cap_risk_pct(
        risk_pct, leverage, balance, committed_margin, committed_notional,
        risk_multiplier=strategy["portfolio_risk_multiplier"],
        max_margin_pct=strategy["global_max_margin_pct"] or 0.0,
        max_notional_pct=strategy["global_max_notional_pct"] or 0.0,
    )


def _min_size_risk_adjust(port: Portfolio, risk_pct: float,
                          max_margin_pct: float | None, leverage: int,
                          price: float, strategy: dict, symbol: str):
    """Research-only exchange-minimum modeling. Returns adjusted risk or None (skip).

    Mirrors ``Portfolio._calculate_position_size`` (size = balance × min(risk, cap)
    × leverage / price), rounds the size DOWN to the contract step, and applies the
    ``min_size_policy`` when it lands below the exchange minimum quantity: "skip"
    fails closed like live, "floor" bumps to the minimum (which inflates effective
    risk exactly when the balance is smallest — measured, not recommended).
    ``strategy["min_qty"] is None`` disables everything (exact legacy behavior).
    """
    minq_map = strategy.get("min_qty")
    if not minq_map or symbol not in minq_map:
        return risk_pct
    if risk_pct <= 0 or leverage <= 0 or price <= 0 or port.balance <= 0:
        return risk_pct
    eff = risk_pct if max_margin_pct is None else min(risk_pct, max_margin_pct)
    size = port.balance * eff * leverage / price
    step = (strategy.get("size_step") or {}).get(symbol)
    if step:
        size = int(size / step) * step
    minq = minq_map[symbol]
    if size >= minq:
        if step:  # reflect the step round-down in the sized risk
            return size * price / leverage / port.balance
        return risk_pct
    # Counters live in a caller-provided nested dict so they survive the
    # simulate_multi() shallow copy of the strat mapping.
    counters = strategy.get("_min_counters")
    if strategy.get("min_size_policy", "skip") == "floor":
        if counters is not None:
            counters["floors"] += 1
        return minq * price / leverage / port.balance
    if counters is not None:
        counters["skips"] += 1
    return None


def simulate_multi(
    assets: dict[str, AssetInput], start_date: str, end_date: str,
    *, slip: float = 0.0, model_liquidation: bool = True,
    maintenance_margin: float = 0.005, exit_granularity: str = "primary",
    strat: dict | None = None,
) -> MultiAssetResult:
    """Replay multiple symbols against one compounding balance."""
    if not assets:
        raise ValueError("At least one asset is required")
    first = next(iter(assets.values())).config
    ps_first = first.position_sizing
    strategy = dict(DEFAULT_STRAT)
    strategy.update({
        "anti_martingale_step": getattr(ps_first, "anti_martingale_step", 0.0),
        "anti_martingale_min": getattr(ps_first, "anti_martingale_min", 0.7),
        "anti_martingale_max": getattr(ps_first, "anti_martingale_max", 1.1),
        "portfolio_risk_multiplier": getattr(ps_first, "portfolio_risk_multiplier", 1.0),
        "global_max_positions": getattr(ps_first, "global_max_positions", 0) or None,
        "global_max_margin_pct": getattr(ps_first, "global_max_margin_pct", 0.0) or None,
        "global_max_notional_pct": getattr(ps_first, "global_max_notional_pct", 0.0) or None,
    })
    if strat:
        strategy.update(strat)
    if strategy["maker_queue_penetration_bps"] < 0:
        raise ValueError("maker_queue_penetration_bps must be non-negative")
    if not 0 <= strategy["maker_fill_probability"] <= 1:
        raise ValueError("maker_fill_probability must be between 0 and 1")
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
    equity_curve: list[EquityPoint] = []
    study_peak = port.initial_balance
    event_count = 0
    maker_orders = 0
    maker_touches = 0
    maker_queue_eligible_count = 0
    maker_fills = 0
    rotations = 0
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
            regime = detect_market_regime(prim).value
            bar_time = str(ts)
            state.last_close, state.last_time = bar_close, bar_time
            latest_prices[symbol] = bar_close
            tf_hours = timeframe_hours(tr.primary_timeframe)
            symbol_trades = [t for t in port.open_trades if t.symbol == symbol]

            # Resolve the prior good-for-one-bar maker order before exit checks.
            fresh = None
            fresh_sub_start = 0
            subs = (pre.subbars[i] if exit_granularity == "sub" and pre.subbars else None)
            if state.pending is not None:
                p = state.pending
                touched = False
                queue_eligible = False
                if subs:
                    for sub_i, (hi, lo, _cl) in enumerate(subs):
                        if maker_limit_touched(p.direction, p.limit_price, hi, lo):
                            touched = True
                        if maker_queue_eligible(
                            p.direction, p.limit_price, hi, lo,
                            strategy["maker_queue_penetration_bps"],
                        ):
                            queue_eligible, fresh_sub_start = True, sub_i
                            break
                else:
                    touched = maker_limit_touched(
                        p.direction, p.limit_price, bar_high, bar_low
                    )
                    queue_eligible = maker_queue_eligible(
                        p.direction, p.limit_price, bar_high, bar_low,
                        strategy["maker_queue_penetration_bps"],
                    )
                maker_touches += int(touched)
                maker_queue_eligible_count += int(queue_eligible)
                fill_key = (f"{symbol}|{p.decision_time}|{p.direction}|"
                            f"{p.limit_price:.12g}")
                fills_queue = queue_eligible and deterministic_maker_fill(
                    strategy["maker_fill_probability"],
                    strategy["maker_fill_seed"], fill_key,
                )
                if fills_queue:
                    fresh = port.open_trade(
                        p.direction, p.limit_price, bar_time, p.stop_loss,
                        p.take_profit_1, p.take_profit_2, leverage=p.leverage,
                        risk_pct=p.risk_pct, tp1_exit_pct=p.tp1_exit_pct,
                        order_type="maker", symbol=symbol,
                        max_margin_pct=p.max_margin_pct,
                    )
                    fresh._atr_entry = p.atr_at_entry
                    symbol_trades.append(fresh)
                    maker_fills += 1
                state.pending = None

            bar_strat = {
                "trail_mode": "pct", "trail_act_atr": .5, "trail_cb_atr": .6,
                "_trail_activation_multiplier": strategy[
                    "regime_trailing_activation_mults"
                ].get(regime, 1.0),
                "_trail_callback_multiplier": strategy[
                    "regime_trailing_callback_mults"
                ].get(regime, 1.0),
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
                            maintenance_margin, bar_strat, count_bar=first_sub,
                            ratchet_trailing=False,
                        )
                        first_sub = False
                    if (trade.is_open and bt.enable_trailing_stops
                            and tr.trailing_stop.enabled):
                        active_subs = subs[start:]
                        favorable = (max(row[0] for row in active_subs)
                                     if trade.direction == "LONG"
                                     else min(row[1] for row in active_subs))
                        _ratchet_trailing_stop(
                            trade, favorable, tr.trailing_stop, bar_strat
                        )
                else:
                    _check_exits(
                        port, trade, bar_high, bar_low, bar_close, bar_time,
                        risk, tf_hours, bt.enable_partial_exits,
                        bt.enable_trailing_stops, tr.trailing_stop, slip,
                        model_liquidation, maintenance_margin, bar_strat,
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
                if not ev.get("streak_applied", False):
                    state.outcome_streak = update_outcome_streak(
                        state.outcome_streak, not ev["loss"]
                    )
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
                alignment_mode=strategy["alignment_mode"],
                alignment_scale=strategy["alignment_scale"],
                alignment_k=strategy["alignment_k"],
                alignment_scale_by_tf=(strategy["alignment_scale_by_tf"]
                                       if strategy["alignment_scale_by_tf"] is not None
                                       else getattr(sc, "alignment_scale_by_tf", None)),
                exclude_alignment_tfs=({"1d"} if strategy.get("daily_trend_replace_align") else None),
            )
            apply_daily_overlay(result, inds.get("1d"), strategy, getattr(sc, "points", None))
            targets = calculate_targets(
                prim, result.direction, tr.stop_loss_strategy,
                sc.atr_sl_multiplier, tier.tp1_rr, tier.tp2_rr,
            )
            abs_score = abs(result.raw_score)
            state.last_raw_score = result.raw_score
            penalty = _loss_penalty(state)
            threshold_mult = strategy["regime_threshold_mults"].get(regime, 1.0)
            strong, marginal = (
                tier.strong_threshold * threshold_mult + penalty,
                tier.marginal_threshold_low * threshold_mult + penalty,
            )
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
                        state.outcome_streak = update_outcome_streak(
                            state.outcome_streak, trade.net_pnl > 0
                        )
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
                pending_all = [s.pending for s in states.values() if s.pending is not None]
                global_slots, committed_margin, committed_notional = committed_exposure(
                    port, pending_all, port.balance
                )
                global_limit = strategy["global_max_positions"]
                global_slot_ok = global_limit is None or global_slots < global_limit
                if (not failures and committed < slots and same_side
                        and global_slot_ok):
                    risk_eff = ps.risk_pct_per_trade
                    if ps.conviction_exponent > 0 and strong > 0:
                        mult = (abs_score / strong) ** ps.conviction_exponent
                        risk_eff *= max(.5, min(1.5, mult))
                    if throttled:
                        risk_eff *= risk.dd_throttle_risk
                    risk_eff *= anti_martingale_multiplier(
                        state.outcome_streak,
                        strategy["anti_martingale_step"],
                        strategy["anti_martingale_min"],
                        strategy["anti_martingale_max"],
                    )
                    leverage = max(1, int(round(
                        tier.leverage
                        * strategy["regime_leverage_mults"].get(regime, 1.0)
                    )))
                    risk_pre_cap = risk_eff
                    risk_eff = apply_exposure_caps(
                        risk_eff, leverage, port.balance,
                        committed_margin, committed_notional, strategy,
                    )
                    # Cross-asset rotation (research, opt-in): a cap-squeezed
                    # STRONG entry may evict the weakest OTHER symbol's position.
                    rot_w = strategy["rotate_weak_support"]
                    rot_g = strategy["rotate_min_gap"]
                    if (rot_w is not None and rot_g is not None
                            and signal == SignalStrength.STRONG
                            and risk_eff
                            < strategy["rotate_squeeze_frac"] * risk_pre_cap):
                        victims = []
                        for t in port.open_trades:
                            if t.symbol == symbol:
                                continue
                            vscore = states[t.symbol].last_raw_score
                            if vscore is None:
                                continue
                            support = vscore if t.direction == "LONG" else -vscore
                            if support <= rot_w and abs_score - support >= rot_g:
                                victims.append((support, t.symbol))
                        if victims:
                            _, vsym = min(victims)
                            vstate = states[vsym]
                            for t in [t for t in port.open_trades
                                      if t.symbol == vsym]:
                                fill = (vstate.last_close * (1 - slip)
                                        if t.direction == "LONG"
                                        else vstate.last_close * (1 + slip))
                                port.close_trade(t, fill, bar_time, "rotation")
                                vstate.outcome_streak = update_outcome_streak(
                                    vstate.outcome_streak, t.net_pnl > 0)
                                if t.net_pnl <= 0:
                                    vstate.consecutive_losses += 1
                                    vstate.candles_since_loss = 0
                                else:
                                    vstate.consecutive_losses = 0
                                    vstate.candles_since_loss = 999
                            rotations += 1
                            _, committed_margin, committed_notional = (
                                committed_exposure(port, pending_all,
                                                   port.balance))
                            risk_eff = apply_exposure_caps(
                                risk_pre_cap, leverage, port.balance,
                                committed_margin, committed_notional, strategy,
                            )
                    if risk_eff <= 0:
                        continue
                    # Exchange-minimum modeling (research-only; None = disabled).
                    risk_eff = _min_size_risk_adjust(
                        port, risk_eff, ps.max_position_pct, leverage,
                        bar_close, strategy, symbol,
                    )
                    if risk_eff is None:
                        continue
                    if tr.entry_mode == "maker":
                        state.pending = PendingEntry(
                            direction, bar_close, targets.stop_loss,
                            targets.take_profit_1, targets.take_profit_2,
                            leverage, risk_eff, tier.tp1_exit_pct,
                            prim.atr_14, bar_time,
                            max_margin_pct=ps.max_position_pct,
                        )
                        maker_orders += 1
                    else:
                        entry = (bar_close * (1 + slip) if direction == "LONG"
                                 else bar_close * (1 - slip))
                        trade = port.open_trade(
                            direction, entry, bar_time, targets.stop_loss,
                            targets.take_profit_1, targets.take_profit_2,
                            leverage=leverage, risk_pct=risk_eff,
                            tp1_exit_pct=tier.tp1_exit_pct, order_type="taker",
                            symbol=symbol,
                            max_margin_pct=ps.max_position_pct,
                        )
                        trade._atr_entry = prim.atr_14

        event_count += 1
        equity = _mark_to_market_equity(port, latest_prices)
        study_peak = max(study_peak, equity)
        equity_curve.append(EquityPoint(
            timestamp=pd.Timestamp(ts), equity=equity,
            drawdown_pct=(100 * (study_peak - equity) / study_peak
                          if study_peak > 0 else 0.0),
        ))
        if event_count % 10 == 0:
            port.record_snapshot(str(ts), latest_prices)

    for trade in list(port.open_trades):
        state = states[trade.symbol]
        port.close_trade(trade, state.last_close, state.last_time, "end_of_backtest")
    if events:
        port.record_snapshot(str(max(events)), latest_prices)
        # Reflect forced end-of-test liquidation and its exit fees at the final
        # timestamp. The independent study peak deliberately remains read-only.
        final_equity = port.balance
        study_peak = max(study_peak, final_equity)
        equity_curve[-1] = EquityPoint(
            timestamp=pd.Timestamp(max(events)), equity=final_equity,
            drawdown_pct=(100 * (study_peak - final_equity) / study_peak
                          if study_peak > 0 else 0.0),
        )
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
        sharpe=stats.sharpe_ratio, per_symbol=per_symbol,
        equity_curve=equity_curve, portfolio=port, maker_orders=maker_orders,
        maker_touches=maker_touches,
        maker_queue_eligible=maker_queue_eligible_count,
        maker_fills=maker_fills,
        rotations=rotations,
    )
