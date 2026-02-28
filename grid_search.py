"""
Grid search — pre-computes indicators once, then sweeps parameters in seconds.

Key optimization: indicators, scores, and category breakdowns are INDEPENDENT of
leverage, thresholds, R:R ratios, and filter toggles. By pre-computing them once
for every bar, we avoid the ~225s/run bottleneck and can test hundreds of combos.

Usage:
    python grid_search.py
    python grid_search.py --config config.json --top 20
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from llm_trading_bot.config import load_config, AppConfig
from llm_trading_bot.data import configure_cache, fetch_multi_timeframe
from llm_trading_bot.scoring import (
    CategoryScore,
    Direction,
    IndicatorSet,
    MarketRegime,
    SignalStrength,
    calculate_indicators,
    compute_composite_score,
    detect_market_regime,
)

# ---------------------------------------------------------------------------
# Pre-computed bar data (indicator-heavy, computed once)
# ---------------------------------------------------------------------------

@dataclass
class PrecomputedBar:
    """All the heavy indicator/scoring data for one bar, computed once."""
    idx: int
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Scoring
    raw_score: float
    direction: Direction
    category_scores: list[CategoryScore]
    # Key indicator values needed for target calculation & filters
    atr_14: float
    adx: float
    atr_pct: float
    bb_width: float
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    regime: MarketRegime
    alignment_bonus: float = 0.0
    # Individual indicator values for strategy-level scoring
    rsi_14: float = 50.0
    stoch_k: float = 50.0
    macd_histogram: float = 0.0
    volume_ratio: float = 1.0
    bb_position: float = 0.5
    ema_aligned: int = 0       # -1 bearish stack, 0 mixed, +1 bullish stack
    above_ema200: bool = True
    obv_bullish: bool = True
    cci_20: float = 0.0
    roc_10: float = 0.0
    change_pct: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0


# ---------------------------------------------------------------------------
# Lightweight trade simulation (no Portfolio class overhead)
# ---------------------------------------------------------------------------

@dataclass
class SimTrade:
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    size: float
    remaining_size: float
    leverage: int
    stop_loss: float
    tp1: float
    tp2: float
    tp1_exit_pct: float
    entry_fee: float
    net_pnl: float = 0.0
    partial_done: bool = False
    is_open: bool = True
    # Trailing stop fields
    peak_price: float = 0.0
    trail_distance: float = 0.0
    trail_activation_price: float = 0.0
    trail_active: bool = False
    bars_held: int = 0


def _sim_fee(size: float, price: float, fee_rate: float) -> float:
    return size * price * fee_rate


def _sim_position_size(balance: float, price: float, leverage: int,
                       risk_pct: float, fee_rate: float) -> float:
    margin = balance * risk_pct
    notional = margin * leverage
    return notional / price


# ---------------------------------------------------------------------------
# Fast backtest runner using pre-computed bars
# ---------------------------------------------------------------------------

def fast_backtest(
    bars: list[PrecomputedBar],
    *,
    leverage: int,
    atr_sl_mult: float,
    tp1_rr: float,
    tp2_rr: float,
    tp1_exit_pct: float,
    marginal_low: float,
    strong_thresh: float,
    min_adx: float,
    min_volatility_pct: float,
    min_category_agreement: int,
    require_trend_momentum_agree: bool,
    skip_choppy: bool,
    skip_volatile: bool,
    sl_strategy: str,
    initial_balance: float,
    fee_rate: float,
    risk_pct: float = 0.02,
    exit_strategy: str = "tp1_tp2",
    trail_atr_mult: float = 2.0,
    trail_activation_atr: float = 1.0,
    scoring_weights: Optional[dict[str, float]] = None,
    score_override_fn: Optional[Callable[["PrecomputedBar"], tuple[float, Direction]]] = None,
) -> dict:
    """
    Run a complete backtest on pre-computed bars. Returns stats dict.
    ~1000x faster than the full engine because no indicator recalculation.

    score_override_fn: if provided, called with each bar to get (score, direction)
    instead of using the pre-computed raw_score/direction or scoring_weights.
    """
    balance = initial_balance
    peak_balance = initial_balance
    max_dd_pct = 0.0
    trade: Optional[SimTrade] = None
    closed_pnls: list[float] = []
    total_fees = 0.0

    for bar in bars:
        # --- 1. Check exits on open trade ---
        if trade and trade.is_open:
            is_long = trade.direction == "LONG"
            h, l = bar.high, bar.low
            trade.bars_held += 1
            exited = False

            # Update peak price for trailing strategies
            if exit_strategy in ("trailing", "tp1_trail"):
                if is_long:
                    trade.peak_price = max(trade.peak_price, h)
                else:
                    trade.peak_price = min(trade.peak_price, l)

            if exit_strategy == "tp1_tp2":
                # === Fixed TP1 (partial) + TP2 (full) ===
                sl_hit = (l <= trade.stop_loss) if is_long else (h >= trade.stop_loss)
                if sl_hit:
                    gross = ((trade.stop_loss - trade.entry_price) if is_long
                             else (trade.entry_price - trade.stop_loss)) * trade.remaining_size
                    ex_fee = _sim_fee(trade.remaining_size, trade.stop_loss, fee_rate)
                    net = gross - ex_fee
                    trade.net_pnl += net
                    balance += net
                    total_fees += ex_fee
                    trade.is_open = False
                    exited = True
                else:
                    # TP1 partial
                    if not trade.partial_done:
                        tp1_hit = (h >= trade.tp1) if is_long else (l <= trade.tp1)
                        if tp1_hit:
                            exit_size = trade.remaining_size * trade.tp1_exit_pct
                            gross = ((trade.tp1 - trade.entry_price) if is_long
                                     else (trade.entry_price - trade.tp1)) * exit_size
                            ex_fee = _sim_fee(exit_size, trade.tp1, fee_rate)
                            net = gross - ex_fee
                            trade.net_pnl += net
                            trade.remaining_size -= exit_size
                            balance += net
                            total_fees += ex_fee
                            trade.partial_done = True
                    # TP2 full exit
                    if trade.is_open:
                        tp2_hit = (h >= trade.tp2) if is_long else (l <= trade.tp2)
                        if tp2_hit:
                            gross = ((trade.tp2 - trade.entry_price) if is_long
                                     else (trade.entry_price - trade.tp2)) * trade.remaining_size
                            ex_fee = _sim_fee(trade.remaining_size, trade.tp2, fee_rate)
                            net = gross - ex_fee
                            trade.net_pnl += net
                            balance += net
                            total_fees += ex_fee
                            trade.is_open = False
                            exited = True

            elif exit_strategy == "trailing":
                # === Pure trailing stop — no fixed TPs ===
                # Check activation
                if not trade.trail_active:
                    if is_long:
                        trade.trail_active = trade.peak_price >= trade.trail_activation_price
                    else:
                        trade.trail_active = trade.peak_price <= trade.trail_activation_price
                # Compute effective SL (ratchet: only tighten, never loosen)
                if trade.trail_active:
                    if is_long:
                        trail_sl = trade.peak_price - trade.trail_distance
                        effective_sl = max(trade.stop_loss, trail_sl)
                    else:
                        trail_sl = trade.peak_price + trade.trail_distance
                        effective_sl = min(trade.stop_loss, trail_sl)
                else:
                    effective_sl = trade.stop_loss

                sl_hit = (l <= effective_sl) if is_long else (h >= effective_sl)
                if sl_hit:
                    gross = ((effective_sl - trade.entry_price) if is_long
                             else (trade.entry_price - effective_sl)) * trade.remaining_size
                    ex_fee = _sim_fee(trade.remaining_size, effective_sl, fee_rate)
                    net = gross - ex_fee
                    trade.net_pnl += net
                    balance += net
                    total_fees += ex_fee
                    trade.is_open = False
                    exited = True

            elif exit_strategy == "tp1_trail":
                # === TP1 partial exit, then trail remainder ===
                if not trade.partial_done:
                    # Phase 1: SL or TP1
                    sl_hit = (l <= trade.stop_loss) if is_long else (h >= trade.stop_loss)
                    if sl_hit:
                        gross = ((trade.stop_loss - trade.entry_price) if is_long
                                 else (trade.entry_price - trade.stop_loss)) * trade.remaining_size
                        ex_fee = _sim_fee(trade.remaining_size, trade.stop_loss, fee_rate)
                        net = gross - ex_fee
                        trade.net_pnl += net
                        balance += net
                        total_fees += ex_fee
                        trade.is_open = False
                        exited = True
                    else:
                        tp1_hit = (h >= trade.tp1) if is_long else (l <= trade.tp1)
                        if tp1_hit:
                            exit_size = trade.remaining_size * trade.tp1_exit_pct
                            gross = ((trade.tp1 - trade.entry_price) if is_long
                                     else (trade.entry_price - trade.tp1)) * exit_size
                            ex_fee = _sim_fee(exit_size, trade.tp1, fee_rate)
                            net = gross - ex_fee
                            trade.net_pnl += net
                            trade.remaining_size -= exit_size
                            balance += net
                            total_fees += ex_fee
                            trade.partial_done = True
                            trade.trail_active = True
                else:
                    # Phase 2: trailing stop on remainder
                    if is_long:
                        trail_sl = trade.peak_price - trade.trail_distance
                        effective_sl = max(trade.stop_loss, trail_sl)
                    else:
                        trail_sl = trade.peak_price + trade.trail_distance
                        effective_sl = min(trade.stop_loss, trail_sl)

                    sl_hit = (l <= effective_sl) if is_long else (h >= effective_sl)
                    if sl_hit:
                        gross = ((effective_sl - trade.entry_price) if is_long
                                 else (trade.entry_price - effective_sl)) * trade.remaining_size
                        ex_fee = _sim_fee(trade.remaining_size, effective_sl, fee_rate)
                        net = gross - ex_fee
                        trade.net_pnl += net
                        balance += net
                        total_fees += ex_fee
                        trade.is_open = False
                        exited = True

            # Common post-exit bookkeeping
            if exited:
                closed_pnls.append(trade.net_pnl)
                if balance > peak_balance:
                    peak_balance = balance
                dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
                if dd > max_dd_pct:
                    max_dd_pct = dd
                trade = None

        # --- 2. Try opening a new trade if no position ---
        if trade is not None:
            continue

        # Re-derive score with custom weights or override function
        if score_override_fn:
            eff_raw, eff_dir = score_override_fn(bar)
        elif scoring_weights:
            wt = sum(cat.raw_score * scoring_weights.get(cat.name, 0)
                     for cat in bar.category_scores)
            eff_raw = max(-100.0, min(100.0, wt + bar.alignment_bonus))
            eff_dir = (Direction.BULLISH if eff_raw > 10
                       else Direction.BEARISH if eff_raw < -10
                       else Direction.NEUTRAL)
        else:
            eff_raw = bar.raw_score
            eff_dir = bar.direction

        abs_score = abs(eff_raw)
        if abs_score < marginal_low:
            continue  # WAIT signal

        if eff_dir == Direction.NEUTRAL:
            continue

        # Calculate targets
        atr = bar.atr_14
        if not atr or atr == 0:
            continue

        is_long = eff_dir == Direction.BULLISH
        entry = bar.close

        # SL distance (hybrid strategy)
        if sl_strategy == "hybrid":
            atr_sl = atr * atr_sl_mult
            if is_long and bar.nearest_support:
                structure_sl = entry - bar.nearest_support
                sl_distance = max(min(atr_sl, structure_sl * 1.1), atr * 0.5)
            elif not is_long and bar.nearest_resistance:
                structure_sl = bar.nearest_resistance - entry
                sl_distance = max(min(atr_sl, structure_sl * 1.1), atr * 0.5)
            else:
                sl_distance = atr_sl
        else:
            sl_distance = atr * atr_sl_mult

        if sl_distance <= 0:
            continue

        if is_long:
            stop_loss = entry - sl_distance
            tp1 = entry + sl_distance * tp1_rr
            tp2 = entry + sl_distance * tp2_rr
        else:
            stop_loss = entry + sl_distance
            tp1 = entry - sl_distance * tp1_rr
            tp2 = entry - sl_distance * tp2_rr

        # --- 3. Apply filters ---
        # ADX filter
        if bar.adx is not None and bar.adx < min_adx:
            continue

        # Volatility filter
        if bar.atr_pct is not None and bar.atr_pct < min_volatility_pct:
            continue

        # Profit-after-fees filter
        total_fee_pct = 2 * fee_rate * leverage * 100
        if exit_strategy == "trailing":
            reward_1 = trail_activation_atr * atr if trail_activation_atr > 0 else atr
        else:
            reward_1 = abs(tp1 - entry)
        profit_at_tp1_pct = (reward_1 / entry * 100 * leverage) if entry else 0
        if profit_at_tp1_pct <= total_fee_pct:
            continue

        # Category agreement filter
        if min_category_agreement > 0 and bar.category_scores:
            agreeing = sum(
                1 for cat in bar.category_scores
                if (cat.raw_score > 0 and is_long) or (cat.raw_score < 0 and not is_long)
            )
            if agreeing < min_category_agreement:
                continue

        # Trend + momentum agreement
        if require_trend_momentum_agree and bar.category_scores:
            trend_cat = next((c for c in bar.category_scores if c.name == "trend"), None)
            mom_cat = next((c for c in bar.category_scores if c.name == "momentum"), None)
            if trend_cat and mom_cat:
                trend_ok = (trend_cat.raw_score > 0) == is_long
                mom_ok = (mom_cat.raw_score > 0) == is_long
                if not (trend_ok and mom_ok):
                    continue

        # Regime filter
        if skip_choppy and bar.regime == MarketRegime.CHOPPY:
            continue
        if skip_volatile and bar.regime == MarketRegime.VOLATILE:
            continue

        # --- 4. Open trade ---
        size = _sim_position_size(balance, entry, leverage, risk_pct, fee_rate)
        entry_fee = _sim_fee(size, entry, fee_rate)
        balance -= entry_fee
        total_fees += entry_fee

        # Trailing parameters
        trail_dist = atr * trail_atr_mult if exit_strategy in ("trailing", "tp1_trail") else 0.0
        if exit_strategy == "trailing":
            trail_act = entry + trail_activation_atr * atr if is_long else entry - trail_activation_atr * atr
        else:
            trail_act = 0.0

        trade = SimTrade(
            direction="LONG" if is_long else "SHORT",
            entry_price=entry,
            size=size,
            remaining_size=size,
            leverage=leverage,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            tp1_exit_pct=tp1_exit_pct,
            entry_fee=entry_fee,
            peak_price=entry,
            trail_distance=trail_dist,
            trail_activation_price=trail_act,
        )

    # Force-close at end
    if trade and trade.is_open:
        last = bars[-1]
        is_long = trade.direction == "LONG"
        gross = ((last.close - trade.entry_price) if is_long
                 else (trade.entry_price - last.close)) * trade.remaining_size
        ex_fee = _sim_fee(trade.remaining_size, last.close, fee_rate)
        net = gross - ex_fee
        trade.net_pnl += net
        balance += net
        total_fees += ex_fee
        closed_pnls.append(trade.net_pnl)

    # Compute stats
    n_trades = len(closed_pnls)
    if n_trades == 0:
        return {
            "trades": 0, "win_rate": 0, "return_pct": 0, "net_pnl": 0,
            "max_dd_pct": 0, "profit_factor": 0, "sharpe": 0,
            "final_balance": round(balance, 2), "fees": round(total_fees, 2),
            "avg_win": 0, "avg_loss": 0,
        }

    winners = [p for p in closed_pnls if p > 0]
    losers = [abs(p) for p in closed_pnls if p <= 0]
    gross_wins = sum(winners)
    gross_losses = sum(losers)
    pf = gross_wins / gross_losses if gross_losses > 0 else 999.0

    # Sharpe
    if n_trades > 1:
        rets = [p / initial_balance for p in closed_pnls]
        sharpe = (np.mean(rets) / np.std(rets) * (252 ** 0.5)) if np.std(rets) > 0 else 0
    else:
        sharpe = 0

    return {
        "trades": n_trades,
        "win_rate": round(len(winners) / n_trades * 100, 1),
        "return_pct": round((balance - initial_balance) / initial_balance * 100, 2),
        "net_pnl": round(balance - initial_balance, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "profit_factor": round(pf, 2),
        "sharpe": round(float(sharpe), 2),
        "final_balance": round(balance, 2),
        "fees": round(total_fees, 2),
        "avg_win": round(np.mean(winners), 2) if winners else 0,
        "avg_loss": round(np.mean(losers), 2) if losers else 0,
    }


# ---------------------------------------------------------------------------
# Pre-computation phase
# ---------------------------------------------------------------------------

def precompute_bars(
    data_by_tf: dict[str, pd.DataFrame],
    config: AppConfig,
) -> list[PrecomputedBar]:
    """
    Calculate indicators and scores for every bar in the test period.
    This is the slow part — run once, then reuse for all grid combos.
    """
    primary_tf = config.trading.primary_timeframe
    primary_df = data_by_tf[primary_tf]
    scoring_cfg = config.scoring
    warmup = config.backtesting.warmup_periods

    start_date = pd.to_datetime(config.backtesting.start_date)
    end_date = pd.to_datetime(config.backtesting.end_date)
    if hasattr(primary_df.index, 'tz') and primary_df.index.tz is not None:
        start_date = start_date.tz_localize(primary_df.index.tz)
        end_date = end_date.tz_localize(primary_df.index.tz)

    test_mask = (primary_df.index >= start_date) & (primary_df.index <= end_date)
    test_indices = primary_df.index[test_mask]

    total = len(test_indices)
    print(f"Pre-computing indicators for {total} bars...")
    bars: list[PrecomputedBar] = []
    t0 = time.time()
    progress_step = max(1, total // 20)

    for i, idx in enumerate(test_indices):
        if i % progress_step == 0:
            elapsed = time.time() - t0
            pct = (i + 1) / total * 100
            eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            print(f"  [{pct:5.1f}%] Bar {i+1}/{total} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

        bar_idx = primary_df.index.get_loc(idx)
        if bar_idx < warmup:
            continue

        current_primary = primary_df.iloc[:bar_idx + 1]
        bar = primary_df.iloc[bar_idx]

        # Calculate indicators on primary TF
        try:
            primary_ind = calculate_indicators(current_primary, primary_tf)
        except (ValueError, Exception):
            continue

        # Build multi-TF indicators dict
        indicators_by_tf: dict[str, IndicatorSet] = {primary_tf: primary_ind}
        for tf, tf_df in data_by_tf.items():
            if tf == primary_tf:
                continue
            try:
                tf_mask = tf_df.index <= idx
                tf_slice = tf_df[tf_mask]
                if len(tf_slice) >= 50:
                    indicators_by_tf[tf] = calculate_indicators(tf_slice, tf)
            except (ValueError, KeyError):
                pass

        # Score
        result = compute_composite_score(
            indicators_by_tf=indicators_by_tf,
            weights=scoring_cfg.weights,
            primary_timeframe=primary_tf,
            confidence_min=scoring_cfg.confidence_min,
            confidence_max=scoring_cfg.confidence_max,
        )

        # Extract alignment bonus (raw_score minus weighted category total)
        weighted_cat_sum = sum(cs.weighted_score for cs in result.category_scores)
        alignment = result.raw_score - weighted_cat_sum

        # Regime
        regime = detect_market_regime(primary_ind)

        # Extract individual indicator values for strategy-level scoring
        ema_aligned = 0
        if primary_ind.ema_9 and primary_ind.ema_21 and primary_ind.ema_50:
            if primary_ind.ema_9 > primary_ind.ema_21 > primary_ind.ema_50:
                ema_aligned = 1
            elif primary_ind.ema_9 < primary_ind.ema_21 < primary_ind.ema_50:
                ema_aligned = -1

        bars.append(PrecomputedBar(
            idx=i,
            timestamp=str(idx),
            open=float(bar["Open"]),
            high=float(bar["High"]),
            low=float(bar["Low"]),
            close=float(bar["Close"]),
            volume=float(bar["Volume"]),
            raw_score=result.raw_score,
            direction=result.direction,
            category_scores=result.category_scores,
            atr_14=primary_ind.atr_14 or 0.0,
            adx=primary_ind.adx or 0.0,
            atr_pct=primary_ind.atr_pct or 0.0,
            bb_width=primary_ind.bb_width or 0.0,
            nearest_support=primary_ind.nearest_support,
            nearest_resistance=primary_ind.nearest_resistance,
            regime=regime,
            alignment_bonus=alignment,
            rsi_14=primary_ind.rsi_14 if primary_ind.rsi_14 is not None else 50.0,
            stoch_k=primary_ind.stoch_k if primary_ind.stoch_k is not None else 50.0,
            macd_histogram=primary_ind.macd_histogram if primary_ind.macd_histogram is not None else 0.0,
            volume_ratio=primary_ind.volume_ratio if primary_ind.volume_ratio is not None else 1.0,
            bb_position=primary_ind.bb_position if primary_ind.bb_position is not None else 0.5,
            ema_aligned=ema_aligned,
            above_ema200=(primary_ind.close > primary_ind.ema_200
                          if primary_ind.close and primary_ind.ema_200 else True),
            obv_bullish=(primary_ind.obv > primary_ind.obv_sma_20
                         if primary_ind.obv is not None and primary_ind.obv_sma_20 is not None else True),
            cci_20=primary_ind.cci_20 if primary_ind.cci_20 is not None else 0.0,
            roc_10=primary_ind.roc_10 if primary_ind.roc_10 is not None else 0.0,
            change_pct=primary_ind.change_pct if primary_ind.change_pct is not None else 0.0,
            plus_di=primary_ind.plus_di if primary_ind.plus_di is not None else 0.0,
            minus_di=primary_ind.minus_di if primary_ind.minus_di is not None else 0.0,
        ))

    elapsed = time.time() - t0
    print(f"  Done: {len(bars)} bars pre-computed in {elapsed:.1f}s")
    return bars


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

def build_grid(strategy: str = "tp1_tp2") -> list[dict]:
    """
    Build the parameter grid for a given exit strategy.
    Filters out invalid combinations (e.g. tp2 <= tp1, strong <= marginal).
    """
    common = {
        "leverage":        [3, 5, 7, 10, 15, 20],
        "atr_sl_mult":     [0.8, 1.0, 1.2, 1.5, 2.0, 2.5],
        "marginal_low":    [15, 20, 25, 30],
        "strong_thresh":   [25, 30, 35, 40, 45],
        "min_cat_agree":   [2, 3],
        "trend_mom_agree": [True, False],
        "skip_choppy":     [True, False],
        "skip_volatile":   [False],
    }

    if strategy == "tp1_tp2":
        specific = {
            "tp1_rr":               [1.0, 1.5, 2.0, 2.5, 3.0],
            "tp2_rr":               [2.0, 3.0, 4.0, 5.0, 6.0],
            "tp1_exit_pct":         [0.3, 0.5, 0.7],
            "trail_atr_mult":       [0.0],
            "trail_activation_atr": [0.0],
        }
    elif strategy == "trailing":
        specific = {
            "tp1_rr":               [0.0],
            "tp2_rr":               [0.0],
            "tp1_exit_pct":         [0.0],
            "trail_atr_mult":       [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            "trail_activation_atr": [0.0, 0.5, 1.0, 1.5, 2.0],
        }
    elif strategy == "tp1_trail":
        specific = {
            "tp1_rr":               [1.0, 1.5, 2.0, 2.5, 3.0],
            "tp2_rr":               [0.0],
            "tp1_exit_pct":         [0.3, 0.5, 0.7],
            "trail_atr_mult":       [1.0, 1.5, 2.0, 3.0, 4.0],
            "trail_activation_atr": [0.0],
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    grid = {**common, **specific}
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    valid = []
    for combo in combos:
        params = dict(zip(keys, combo))
        # Common constraint: strong threshold must exceed marginal
        if params["strong_thresh"] <= params["marginal_low"]:
            continue
        # Strategy-specific constraints
        if strategy == "tp1_tp2":
            if params["tp2_rr"] <= params["tp1_rr"]:
                continue
        valid.append(params)

    return valid


# ---------------------------------------------------------------------------
# Risk Profiles & Scoring Weight Grid
# ---------------------------------------------------------------------------

RISK_PROFILES = {
    "aggressive": {
        "leverage": 20, "atr_sl_mult": 1.5,
        "tp1_rr": 3.0, "tp2_rr": 6.0, "tp1_exit_pct": 0.3,
        "marginal_low": 25, "strong_thresh": 30,
        "min_cat_agree": 2, "trend_mom_agree": True,
        "skip_choppy": True, "skip_volatile": False,
    },
    "medium": {
        "leverage": 10, "atr_sl_mult": 1.2,
        "tp1_rr": 2.0, "tp2_rr": 4.0, "tp1_exit_pct": 0.5,
        "marginal_low": 25, "strong_thresh": 35,
        "min_cat_agree": 2, "trend_mom_agree": True,
        "skip_choppy": True, "skip_volatile": False,
    },
    "safe": {
        "leverage": 5, "atr_sl_mult": 1.0,
        "tp1_rr": 1.5, "tp2_rr": 3.0, "tp1_exit_pct": 0.5,
        "marginal_low": 30, "strong_thresh": 40,
        "min_cat_agree": 3, "trend_mom_agree": True,
        "skip_choppy": True, "skip_volatile": False,
    },
}


def build_scoring_grid(resolution: float = 0.05, min_weight: float = 0.05) -> list[dict[str, float]]:
    """
    Generate all valid category weight combinations summing to 1.0.
    With resolution=0.05, min_weight=0.05 → 3876 combos.
    """
    cats = ["trend", "momentum", "volume", "support_resistance", "risk"]
    n_cats = len(cats)
    steps = round(1.0 / resolution)
    min_steps = round(min_weight / resolution)
    remaining = steps - min_steps * n_cats
    if remaining < 0:
        return []

    combos: list[dict[str, float]] = []
    for a in range(remaining + 1):
        for b in range(remaining - a + 1):
            for c in range(remaining - a - b + 1):
                for d in range(remaining - a - b - c + 1):
                    e = remaining - a - b - c - d
                    extras = [a, b, c, d, e]
                    weights = {cat: round((min_steps + extra) * resolution, 2)
                               for cat, extra in zip(cats, extras)}
                    combos.append(weights)
    return combos


def run_scoring_mode(args, config, bars):
    """Grid search scoring weights with fixed risk profiles."""
    weight_combos = build_scoring_grid(resolution=0.05, min_weight=0.05)
    profiles_to_run = (
        list(RISK_PROFILES.keys()) if args.risk_profile == "all"
        else [args.risk_profile]
    )
    print(f"Phase 3: Testing {len(weight_combos)} weight combos × "
          f"{len(profiles_to_run)} risk profile(s)...")
    print()

    initial_bal = config.backtesting.initial_balance
    fee_rate = config.fees.active_fee_rate
    sl_strategy = config.trading.stop_loss_strategy
    min_vol = config.filters.min_volatility_pct
    min_adx = config.filters.min_adx

    all_profile_results: dict[str, list[dict]] = {}

    for profile_name in profiles_to_run:
        p = RISK_PROFILES[profile_name]
        print(f"--- Risk Profile: {profile_name.upper()} "
              f"(Lev={p['leverage']} SL={p['atr_sl_mult']} "
              f"TP1={p['tp1_rr']} TP2={p['tp2_rr']} Exit={p['tp1_exit_pct']}) ---")

        results: list[dict] = []
        t_start = time.time()
        total = len(weight_combos)

        for i, weights in enumerate(weight_combos):
            stats = fast_backtest(
                bars,
                leverage=p["leverage"],
                atr_sl_mult=p["atr_sl_mult"],
                tp1_rr=p["tp1_rr"],
                tp2_rr=p["tp2_rr"],
                tp1_exit_pct=p["tp1_exit_pct"],
                marginal_low=p["marginal_low"],
                strong_thresh=p["strong_thresh"],
                min_adx=min_adx,
                min_volatility_pct=min_vol,
                min_category_agreement=p["min_cat_agree"],
                require_trend_momentum_agree=p["trend_mom_agree"],
                skip_choppy=p["skip_choppy"],
                skip_volatile=p["skip_volatile"],
                sl_strategy=sl_strategy,
                initial_balance=initial_bal,
                fee_rate=fee_rate,
                scoring_weights=weights,
            )

            calmar = (stats["return_pct"] / stats["max_dd_pct"]
                      if stats["max_dd_pct"] > 0 else 0)
            stats["calmar"] = round(calmar, 2)

            row = {
                "risk_profile": profile_name,
                "w_trend": weights["trend"],
                "w_momentum": weights["momentum"],
                "w_volume": weights["volume"],
                "w_sr": weights["support_resistance"],
                "w_risk": weights["risk"],
                **stats,
            }
            results.append(row)

            if (i + 1) % 500 == 0 or i + 1 == total:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                best = max(results, key=lambda x: x.get(args.sort_by, 0))
                print(f"  [{i+1:>6}/{total}] {elapsed:.0f}s | {rate:.0f}/s | "
                      f"ETA: {eta:.0f}s | Best {args.sort_by}: {best[args.sort_by]}")

        total_time = time.time() - t_start

        # Filter and sort
        meaningful = [r for r in results if r["trades"] >= 10]
        print(f"  {len(meaningful)}/{len(results)} combos had >= 10 trades "
              f"({total_time:.1f}s)")

        if args.sort_by == "calmar":
            meaningful.sort(key=lambda x: x["calmar"], reverse=True)
        elif args.sort_by == "sharpe":
            meaningful.sort(key=lambda x: x["sharpe"], reverse=True)
        elif args.sort_by == "profit_factor":
            meaningful.sort(key=lambda x: x["profit_factor"], reverse=True)
        else:
            meaningful.sort(key=lambda x: x["return_pct"], reverse=True)

        for i, r in enumerate(meaningful):
            r["rank"] = i + 1
        all_profile_results[profile_name] = meaningful

        # Display top N
        print()
        print(f"  TOP {args.top} — {profile_name.upper()}")
        print(
            f"  {'#':>3} | {'Trend':>5} | {'Mom':>5} | {'Vol':>5} | "
            f"{'S/R':>5} | {'Risk':>5} | {'Trades':>6} | {'WR%':>5} | "
            f"{'Return':>8} | {'MaxDD':>6} | {'PF':>5} | "
            f"{'Sharpe':>6} | {'Calmar':>6} | {'Balance':>10}"
        )
        print("  " + "-" * 115)

        for r in meaningful[:args.top]:
            print(
                f"  {r['rank']:>3} | {r['w_trend']:>5.2f} | "
                f"{r['w_momentum']:>5.2f} | {r['w_volume']:>5.2f} | "
                f"{r['w_sr']:>5.2f} | {r['w_risk']:>5.2f} | "
                f"{r['trades']:>6} | {r['win_rate']:>5.1f} | "
                f"{r['return_pct']:>+7.1f}% | {r['max_dd_pct']:>5.1f}% | "
                f"{r['profit_factor']:>5.2f} | {r['sharpe']:>6.2f} | "
                f"{r['calmar']:>6.2f} | ${r['final_balance']:>9.2f}"
            )
        print()

    # Write combined CSV
    csv_path = Path(args.output)
    csv_fields = [
        "rank", "risk_profile",
        "w_trend", "w_momentum", "w_volume", "w_sr", "w_risk",
        "trades", "win_rate", "return_pct", "net_pnl", "max_dd_pct",
        "profit_factor", "sharpe", "calmar", "final_balance", "fees",
        "avg_win", "avg_loss",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for pname in profiles_to_run:
            writer.writerows(all_profile_results[pname])
    print(f"Results saved to {csv_path}")

    # Cross-profile comparison
    if len(profiles_to_run) > 1:
        print()
        print("=" * 100)
        print("  CROSS-PROFILE COMPARISON (best per profile)")
        print("=" * 100)
        for pname in profiles_to_run:
            res = all_profile_results[pname]
            if res:
                b = res[0]
                print(
                    f"  {pname:>12}: "
                    f"T={b['w_trend']:.2f} M={b['w_momentum']:.2f} "
                    f"V={b['w_volume']:.2f} SR={b['w_sr']:.2f} R={b['w_risk']:.2f} | "
                    f"{b['return_pct']:>+6.1f}% | DD={b['max_dd_pct']:.1f}% | "
                    f"PF={b['profit_factor']:.2f} | Sh={b['sharpe']:.2f} | "
                    f"Cal={b['calmar']:.2f} | {b['trades']} trades"
                )
        print()
        print("  Current default weights: T=0.30 M=0.25 V=0.15 SR=0.20 R=0.10")


# ---------------------------------------------------------------------------
# Scoring Strategy Factory Functions
# ---------------------------------------------------------------------------
# Each returns a Callable[[PrecomputedBar], tuple[float, Direction]].
# These are fundamentally different scoring PHILOSOPHIES, not just weight tweaks.

def make_confirmation_fn(min_agree: int = 5, adx_thresh: float = 20) -> Callable:
    """
    CONFIRMATION COUNTING (Voting) — Count individual indicator signals that agree.
    Trade when >= min_agree out of 10 indicators point in the same direction.
    """
    def score_fn(bar: PrecomputedBar) -> tuple[float, Direction]:
        signals: list[int] = []

        # 1. EMA alignment
        signals.append(bar.ema_aligned)
        # 2. Price vs EMA-200
        signals.append(1 if bar.above_ema200 else -1)
        # 3. MACD histogram
        signals.append(1 if bar.macd_histogram > 0 else (-1 if bar.macd_histogram < 0 else 0))
        # 4. ADX direction (only count if ADX strong enough)
        if bar.adx >= adx_thresh and bar.plus_di != bar.minus_di:
            signals.append(1 if bar.plus_di > bar.minus_di else -1)
        else:
            signals.append(0)
        # 5. RSI direction
        signals.append(1 if bar.rsi_14 > 55 else (-1 if bar.rsi_14 < 45 else 0))
        # 6. Stochastic direction
        signals.append(1 if bar.stoch_k > 55 else (-1 if bar.stoch_k < 45 else 0))
        # 7. ROC direction
        signals.append(1 if bar.roc_10 > 0 else (-1 if bar.roc_10 < 0 else 0))
        # 8. CCI direction
        signals.append(1 if bar.cci_20 > 0 else (-1 if bar.cci_20 < 0 else 0))
        # 9. Volume + price direction confirmation
        if bar.volume_ratio > 1.0 and bar.change_pct > 0:
            signals.append(1)
        elif bar.volume_ratio > 1.0 and bar.change_pct < 0:
            signals.append(-1)
        else:
            signals.append(0)
        # 10. OBV trend
        signals.append(1 if bar.obv_bullish else -1)

        n = len(signals)
        bullish = sum(1 for s in signals if s > 0)
        bearish = sum(1 for s in signals if s < 0)

        if bullish >= min_agree:
            return bullish / n * 100, Direction.BULLISH
        elif bearish >= min_agree:
            return -(bearish / n * 100), Direction.BEARISH
        return 0.0, Direction.NEUTRAL

    return score_fn


def make_momentum_breakout_fn(
    rsi_weight: float = 0.5, volume_mult: float = 1.0,
    roc_weight: float = 0.3, min_rsi_dev: float = 10,
) -> Callable:
    """
    MOMENTUM BREAKOUT — Pure momentum signal with volume amplification.
    Trades when RSI deviates strongly from 50 and ROC confirms direction.
    High volume amplifies the signal.
    """
    def score_fn(bar: PrecomputedBar) -> tuple[float, Direction]:
        rsi_dev = bar.rsi_14 - 50  # -50 to +50
        if abs(rsi_dev) < min_rsi_dev:
            return 0.0, Direction.NEUTRAL

        # Volume factor: amplify when volume is high, dampen when low
        vol_factor = (1.0 + (bar.volume_ratio - 1.0) * volume_mult
                      if bar.volume_ratio > 1.0 else 0.5)

        # ROC scaled to -50..+50 range
        roc_signal = max(-50, min(50, bar.roc_10 * 10))

        score = (rsi_dev * rsi_weight + roc_signal * roc_weight) * vol_factor
        score = max(-100, min(100, score))

        if score > 10:
            return score, Direction.BULLISH
        elif score < -10:
            return score, Direction.BEARISH
        return 0.0, Direction.NEUTRAL

    return score_fn


def make_trend_gated_fn(
    require_full_stack: bool = True, min_adx: float = 25,
    mom_w: float = 0.6, vol_w: float = 0.2,
) -> Callable:
    """
    TREND GATED — Require strong established trend (EMA stack + ADX), then
    use momentum for timing and volume for confirmation.
    High-conviction trend-following strategy.
    """
    def score_fn(bar: PrecomputedBar) -> tuple[float, Direction]:
        # Gate 1: EMA alignment
        if require_full_stack and bar.ema_aligned == 0:
            return 0.0, Direction.NEUTRAL

        # Gate 2: ADX strength
        if bar.adx < min_adx:
            return 0.0, Direction.NEUTRAL

        # Determine trend direction
        trend_dir = bar.ema_aligned
        if trend_dir == 0:
            # Fall back to MACD + EMA200 if partial alignment
            if bar.macd_histogram > 0 and bar.above_ema200:
                trend_dir = 1
            elif bar.macd_histogram < 0 and not bar.above_ema200:
                trend_dir = -1
            else:
                return 0.0, Direction.NEUTRAL

        # Get category scores for magnitude
        mom_score = next((c.raw_score for c in bar.category_scores
                          if c.name == "momentum"), 0)
        vol_score = next((c.raw_score for c in bar.category_scores
                          if c.name == "volume"), 0)

        # Only enter if momentum agrees with trend
        if (trend_dir > 0 and mom_score < -10) or (trend_dir < 0 and mom_score > 10):
            return 0.0, Direction.NEUTRAL

        score = (abs(mom_score) * mom_w + abs(vol_score) * vol_w) * trend_dir
        score = max(-100, min(100, score))

        direction = Direction.BULLISH if score > 0 else Direction.BEARISH
        return score, direction

    return score_fn


def make_mean_reversion_fn(
    rsi_low: float = 30, rsi_high: float = 70,
    require_sr: bool = True, sr_bonus: float = 1.5,
) -> Callable:
    """
    MEAN REVERSION — Buy oversold conditions, sell overbought.
    Opposite of trend-following. Best in ranging markets.
    Uses RSI, Stochastic, and Bollinger Band extremes.
    S/R proximity provides confirmation.
    """
    def score_fn(bar: PrecomputedBar) -> tuple[float, Direction]:
        extremes = 0
        direction = 0  # +1 = expect bounce up, -1 = expect drop

        # RSI extreme
        if bar.rsi_14 < rsi_low:
            extremes += 1
            direction += 1  # Oversold → buy
        elif bar.rsi_14 > rsi_high:
            extremes += 1
            direction -= 1  # Overbought → sell

        # Stochastic extreme
        if bar.stoch_k < 20:
            extremes += 1
            direction += 1
        elif bar.stoch_k > 80:
            extremes += 1
            direction -= 1

        # Bollinger Band extreme
        if bar.bb_position < 0.05:
            extremes += 1
            direction += 1
        elif bar.bb_position > 0.95:
            extremes += 1
            direction -= 1

        if extremes == 0 or direction == 0:
            return 0.0, Direction.NEUTRAL

        is_long = direction > 0

        # S/R confirmation
        sr_mult = 1.0
        if require_sr:
            sr_score = next((c.raw_score for c in bar.category_scores
                             if c.name == "support_resistance"), 0)
            # For long: want near support (sr_score > 0)
            # For short: want near resistance (sr_score < 0)
            if (is_long and sr_score > 0) or (not is_long and sr_score < 0):
                sr_mult = sr_bonus
            elif (is_long and sr_score < -15) or (not is_long and sr_score > 15):
                return 0.0, Direction.NEUTRAL  # S/R strongly disagrees

        score = min(100, extremes * 33 * sr_mult)

        if is_long:
            return score, Direction.BULLISH
        return -score, Direction.BEARISH

    return score_fn


def make_volume_confirmed_fn(
    vol_thresh: float = 1.2, trend_w: float = 0.5,
    mom_w: float = 0.4, vol_scale: float = 1.5,
) -> Callable:
    """
    VOLUME CONFIRMED — Only trade when above-average volume backs the signal.
    Base signal from trend + momentum, amplified by volume factor.
    No volume = no trade. Avoids low-conviction fakeouts.
    """
    def score_fn(bar: PrecomputedBar) -> tuple[float, Direction]:
        # Hard gate: volume must be above threshold
        if bar.volume_ratio < vol_thresh:
            return 0.0, Direction.NEUTRAL

        trend = next((c.raw_score for c in bar.category_scores
                       if c.name == "trend"), 0)
        mom = next((c.raw_score for c in bar.category_scores
                     if c.name == "momentum"), 0)

        base = trend * trend_w + mom * mom_w

        # Amplify by volume
        vol_factor = 1.0 + (bar.volume_ratio - 1.0) * vol_scale
        score = base * vol_factor
        score = max(-100, min(100, score))

        if score > 10:
            return score, Direction.BULLISH
        elif score < -10:
            return score, Direction.BEARISH
        return 0.0, Direction.NEUTRAL

    return score_fn


def make_regime_adaptive_fn(
    trending_trend_w: float = 0.5, trending_mom_w: float = 0.3,
    ranging_sr_w: float = 0.4, ranging_mom_flip: float = 0.3,
    volatile_risk_floor: float = 0,
) -> Callable:
    """
    REGIME ADAPTIVE — Changes scoring approach based on detected market regime.
    - TRENDING: trend-following with heavy trend + momentum weights
    - RANGING/CHOPPY: mean-reversion (flip momentum) + S/R emphasis
    - VOLATILE: risk-gated; only trade if risk score is favorable
    """
    def score_fn(bar: PrecomputedBar) -> tuple[float, Direction]:
        cats = {c.name: c.raw_score for c in bar.category_scores}

        if bar.regime in (MarketRegime.TRENDING, MarketRegime.WEAK_TREND):
            remain = max(0, 1.0 - trending_trend_w - trending_mom_w)
            score = (cats.get("trend", 0) * trending_trend_w +
                     cats.get("momentum", 0) * trending_mom_w +
                     cats.get("volume", 0) * remain * 0.5 +
                     cats.get("support_resistance", 0) * remain * 0.5)

        elif bar.regime in (MarketRegime.RANGING, MarketRegime.CHOPPY):
            remain = max(0, 1.0 - ranging_sr_w - ranging_mom_flip)
            # FLIP momentum for mean reversion in ranging markets
            score = (cats.get("support_resistance", 0) * ranging_sr_w +
                     cats.get("momentum", 0) * (-ranging_mom_flip) +
                     cats.get("trend", 0) * remain * 0.3 +
                     cats.get("risk", 0) * remain * 0.7)

        elif bar.regime == MarketRegime.VOLATILE:
            risk_score = cats.get("risk", 0)
            if risk_score < volatile_risk_floor:
                return 0.0, Direction.NEUTRAL
            score = (cats.get("trend", 0) * 0.4 +
                     cats.get("momentum", 0) * 0.3 +
                     cats.get("risk", 0) * 0.3)

        else:
            score = sum(c.raw_score * 0.2 for c in bar.category_scores)

        score = max(-100, min(100, score))
        if score > 10:
            return score, Direction.BULLISH
        elif score < -10:
            return score, Direction.BEARISH
        return 0.0, Direction.NEUTRAL

    return score_fn


# ---------------------------------------------------------------------------
# Strategy Grid Definitions
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, dict] = {
    "confirmation_count": {
        "factory": make_confirmation_fn,
        "grid": [
            {"min_agree": m, "adx_thresh": a}
            for m in [3, 4, 5, 6, 7, 8]
            for a in [15, 20, 25, 30]
        ],
        "display_cols": ["min_agree", "adx_thresh"],
        "description": "Vote: trade when N/10 indicators agree",
    },
    "momentum_breakout": {
        "factory": make_momentum_breakout_fn,
        "grid": [
            {"rsi_weight": r, "volume_mult": v, "roc_weight": rc, "min_rsi_dev": d}
            for r in [0.3, 0.5, 0.7, 0.9]
            for v in [0.5, 1.0, 1.5, 2.0]
            for rc in [0.1, 0.3, 0.5]
            for d in [5, 10, 15, 20]
        ],
        "display_cols": ["rsi_weight", "volume_mult", "roc_weight", "min_rsi_dev"],
        "description": "Pure momentum: RSI deviation + ROC + volume amp",
    },
    "trend_gated": {
        "factory": make_trend_gated_fn,
        "grid": [
            {"require_full_stack": fs, "min_adx": a, "mom_w": m, "vol_w": v}
            for fs in [True, False]
            for a in [20, 25, 30, 35]
            for m in [0.4, 0.6, 0.8]
            for v in [0.1, 0.2, 0.3]
        ],
        "display_cols": ["require_full_stack", "min_adx", "mom_w", "vol_w"],
        "description": "Require strong trend, time with momentum",
    },
    "mean_reversion": {
        "factory": make_mean_reversion_fn,
        "grid": [
            {"rsi_low": rl, "rsi_high": rh, "require_sr": sr, "sr_bonus": sb}
            for rl in [25, 30, 35, 40]
            for rh in [60, 65, 70, 75]
            for sr in [True, False]
            for sb in [1.0, 1.5, 2.0]
        ],
        "display_cols": ["rsi_low", "rsi_high", "require_sr", "sr_bonus"],
        "description": "Buy oversold, sell overbought + S/R confirm",
    },
    "volume_confirmed": {
        "factory": make_volume_confirmed_fn,
        "grid": [
            {"vol_thresh": vt, "trend_w": tw, "mom_w": mw, "vol_scale": vs}
            for vt in [1.0, 1.2, 1.5, 2.0]
            for tw in [0.3, 0.5, 0.7]
            for mw in [0.2, 0.4, 0.6]
            for vs in [1.0, 1.5, 2.0]
        ],
        "display_cols": ["vol_thresh", "trend_w", "mom_w", "vol_scale"],
        "description": "Only trade when volume backs the signal",
    },
    "regime_adaptive": {
        "factory": make_regime_adaptive_fn,
        "grid": [
            {"trending_trend_w": tt, "trending_mom_w": tm,
             "ranging_sr_w": rs, "ranging_mom_flip": rm,
             "volatile_risk_floor": vf}
            for tt in [0.4, 0.5, 0.6]
            for tm in [0.2, 0.3, 0.4]
            for rs in [0.3, 0.4, 0.5]
            for rm in [0.2, 0.3, 0.4]
            for vf in [-10, 0, 10]
        ],
        "display_cols": ["trending_trend_w", "trending_mom_w", "ranging_sr_w",
                         "ranging_mom_flip", "volatile_risk_floor"],
        "description": "Adapt scoring to market regime (trend/range/volatile)",
    },
}


def run_strategy_mode(args, config, bars):
    """Grid search fundamentally different scoring strategies."""
    strategies_to_run = (
        list(STRATEGY_REGISTRY.keys()) if args.strategy_name == "all"
        else [args.strategy_name]
    )
    profiles_to_run = (
        list(RISK_PROFILES.keys()) if args.risk_profile == "all"
        else [args.risk_profile]
    )

    total_combos = sum(len(STRATEGY_REGISTRY[s]["grid"]) for s in strategies_to_run)
    print(f"Phase 3: Testing {len(strategies_to_run)} strategies × "
          f"{total_combos} total combos × {len(profiles_to_run)} risk profile(s)...")
    print()

    initial_bal = config.backtesting.initial_balance
    fee_rate = config.fees.active_fee_rate
    sl_strategy = config.trading.stop_loss_strategy
    min_vol = config.filters.min_volatility_pct
    min_adx_cfg = config.filters.min_adx

    all_results: list[dict] = []
    best_per_strategy: dict[str, dict] = {}  # strategy×profile → best result

    for strat_name in strategies_to_run:
        strat_info = STRATEGY_REGISTRY[strat_name]
        factory = strat_info["factory"]
        grid = strat_info["grid"]
        display_cols = strat_info["display_cols"]

        print(f"{'='*80}")
        print(f"  Strategy: {strat_name} ({strat_info['description']})")
        print(f"  Combos: {len(grid)}")
        print(f"{'='*80}")

        for profile_name in profiles_to_run:
            p = RISK_PROFILES[profile_name]
            print(f"\n  --- {profile_name.upper()} (Lev={p['leverage']} "
                  f"SL={p['atr_sl_mult']} TP1={p['tp1_rr']} TP2={p['tp2_rr']}) ---")

            results: list[dict] = []
            t_start = time.time()
            total = len(grid)

            for i, params in enumerate(grid):
                score_fn = factory(**params)

                stats = fast_backtest(
                    bars,
                    leverage=p["leverage"],
                    atr_sl_mult=p["atr_sl_mult"],
                    tp1_rr=p["tp1_rr"],
                    tp2_rr=p["tp2_rr"],
                    tp1_exit_pct=p["tp1_exit_pct"],
                    marginal_low=p["marginal_low"],
                    strong_thresh=p["strong_thresh"],
                    min_adx=min_adx_cfg,
                    min_volatility_pct=min_vol,
                    min_category_agreement=0,  # Strategy handles its own filtering
                    require_trend_momentum_agree=False,
                    skip_choppy=p["skip_choppy"],
                    skip_volatile=p["skip_volatile"],
                    sl_strategy=sl_strategy,
                    initial_balance=initial_bal,
                    fee_rate=fee_rate,
                    score_override_fn=score_fn,
                )

                calmar = (stats["return_pct"] / stats["max_dd_pct"]
                          if stats["max_dd_pct"] > 0 else 0)
                stats["calmar"] = round(calmar, 2)

                row = {
                    "strategy": strat_name,
                    "risk_profile": profile_name,
                    **params,
                    **stats,
                }
                results.append(row)

                if (i + 1) % 100 == 0 or i + 1 == total:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (total - i - 1) / rate if rate > 0 else 0
                    best = max(results, key=lambda x: x.get(args.sort_by, 0))
                    print(f"    [{i+1:>5}/{total}] {elapsed:.1f}s | {rate:.0f}/s | "
                          f"Best {args.sort_by}: {best.get(args.sort_by, 0):.2f}")

            # Filter and sort
            meaningful = [r for r in results if r["trades"] >= 5]
            if args.sort_by == "calmar":
                meaningful.sort(key=lambda x: x["calmar"], reverse=True)
            elif args.sort_by == "sharpe":
                meaningful.sort(key=lambda x: x["sharpe"], reverse=True)
            elif args.sort_by == "profit_factor":
                meaningful.sort(key=lambda x: x["profit_factor"], reverse=True)
            else:
                meaningful.sort(key=lambda x: x["return_pct"], reverse=True)

            valid_count = len(meaningful)
            print(f"    {valid_count}/{len(results)} combos had >= 5 trades")
            all_results.extend(meaningful)

            # Track best
            if meaningful:
                key = f"{strat_name}|{profile_name}"
                best_per_strategy[key] = meaningful[0]

            # Display top N
            if meaningful:
                top_n = min(args.top, len(meaningful))
                header_parts = [f"{'#':>3}"]
                for col in display_cols:
                    header_parts.append(f"{col[:8]:>8}")
                header_parts.extend([
                    f"{'Trades':>6}", f"{'WR%':>5}", f"{'Return':>8}",
                    f"{'MaxDD':>6}", f"{'PF':>5}", f"{'Sharpe':>6}",
                    f"{'Calmar':>6}", f"{'Balance':>10}",
                ])
                print(f"\n    TOP {top_n} — {strat_name} × {profile_name}")
                print("    " + " | ".join(header_parts))
                print("    " + "-" * (sum(len(p) + 3 for p in header_parts)))

                for rank, r in enumerate(meaningful[:top_n], 1):
                    parts = [f"{rank:>3}"]
                    for col in display_cols:
                        val = r.get(col, "")
                        if isinstance(val, bool):
                            parts.append(f"{'Y' if val else 'N':>8}")
                        elif isinstance(val, float):
                            parts.append(f"{val:>8.2f}")
                        else:
                            parts.append(f"{val:>8}")
                        parts.append(f"{r['trades']:>6}")
                    parts.append(f"{r['win_rate']:>5.1f}")
                    parts.append(f"{r['return_pct']:>+7.1f}%")
                    parts.append(f"{r['max_dd_pct']:>5.1f}%")
                    parts.append(f"{r['profit_factor']:>5.2f}")
                    parts.append(f"{r['sharpe']:>6.2f}")
                    parts.append(f"{r['calmar']:>6.2f}")
                    parts.append(f"${r['final_balance']:>9.2f}")
                    print("    " + " | ".join(parts))

    # Write CSV
    csv_path = Path(args.output)
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {csv_path}")

    # Cross comparison: best of each strategy × profile
    print()
    print("=" * 110)
    print("  STRATEGY COMPARISON — Best result per strategy × risk profile")
    print("=" * 110)
    print(f"  {'Strategy':<22} | {'Profile':<10} | {'Return':>8} | {'MaxDD':>6} | "
          f"{'PF':>5} | {'Sharpe':>6} | {'Calmar':>6} | {'WR%':>5} | {'Trades':>6} | Key Params")
    print("  " + "-" * 108)

    for key in sorted(best_per_strategy.keys()):
        r = best_per_strategy[key]
        sname, pname = key.split("|")
        display_cols = STRATEGY_REGISTRY[sname]["display_cols"]
        param_str = " ".join(
            f"{c[:6]}={r.get(c, '?')}" for c in display_cols[:4]
        )
        print(
            f"  {sname:<22} | {pname:<10} | {r['return_pct']:>+7.1f}% | "
            f"{r['max_dd_pct']:>5.1f}% | {r['profit_factor']:>5.2f} | "
            f"{r['sharpe']:>6.2f} | {r['calmar']:>6.2f} | "
            f"{r['win_rate']:>5.1f} | {r['trades']:>6} | {param_str}"
        )

    # Overall winner
    if best_per_strategy:
        overall = max(best_per_strategy.values(), key=lambda x: x.get(args.sort_by, 0))
        print(
            f"\n  >>> OVERALL BEST ({args.sort_by}): {overall['strategy']} × "
            f"{overall['risk_profile']} = {overall['return_pct']:+.1f}% return, "
            f"{overall['max_dd_pct']:.1f}% DD, Sharpe={overall['sharpe']:.2f}, "
            f"Calmar={overall['calmar']:.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grid Search for Optimal Parameters")
    parser.add_argument("--config", "-c", default="config.json")
    parser.add_argument("--top", "-t", type=int, default=20,
                        help="Number of top results to display")
    parser.add_argument("--output", "-o", default="grid_results.csv",
                        help="CSV output file (written progressively)")
    parser.add_argument("--sort-by", "-s", default="return_pct",
                        choices=["return_pct", "sharpe", "profit_factor", "calmar"],
                        help="Metric to optimize for")
    parser.add_argument("--strategy", default="tp1_tp2",
                        choices=["tp1_tp2", "trailing", "tp1_trail"],
                        help="Exit strategy to test (params mode only)")
    parser.add_argument("--mode", default="params",
                        choices=["params", "scoring", "strategy"],
                        help="params: vary trade params; scoring: vary scoring weights; "
                             "strategy: test different scoring strategies")
    parser.add_argument("--risk-profile", default="all",
                        choices=["all", "aggressive", "medium", "safe"],
                        help="Risk profile for scoring/strategy mode")
    parser.add_argument("--strategy-name", default="all",
                        choices=["all"] + list(STRATEGY_REGISTRY.keys()),
                        help="Which scoring strategy to test (strategy mode only)")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_cache(config.data_cache.ttl_seconds)

    ds = config.data_source
    symbol = ds.exchange_symbol if ds.source != "yfinance" else config.trading.yfinance_symbol

    print("=" * 70)
    print("            GRID SEARCH - PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Symbol: {symbol} | Source: {ds.source}")
    print(f"Period: {config.backtesting.start_date} to {config.backtesting.end_date}")
    print(f"Initial balance: ${config.backtesting.initial_balance:,.2f}")
    print(f"Optimizing for: {args.sort_by}")
    if args.mode == "params":
        print(f"Exit strategy: {args.strategy}")
    elif args.mode == "scoring":
        print(f"Mode: scoring weights | Profile: {args.risk_profile}")
    else:
        strats = args.strategy_name if args.strategy_name != "all" else "all 6"
        print(f"Mode: strategy search | Strategies: {strats} | Profile: {args.risk_profile}")
    print()

    # --- 1. Fetch data (cached) ---
    print("Phase 1: Fetching data...")
    data_by_tf = fetch_multi_timeframe(
        symbol=symbol,
        timeframes=config.trading.timeframes,
        start_date=config.backtesting.start_date,
        end_date=config.backtesting.end_date,
        warmup_periods=config.backtesting.warmup_periods,
        source=ds.source,
    )
    for tf, df in data_by_tf.items():
        print(f"  {tf}: {len(df)} candles")
    print()

    # --- 2. Pre-compute indicators (the slow part, done once) ---
    print("Phase 2: Pre-computing indicators and scores (one-time cost)...")
    bars = precompute_bars(data_by_tf, config)
    print()

    # Scoring mode: fixed trade params, vary category weights
    if args.mode == "scoring":
        run_scoring_mode(args, config, bars)
        return

    # Strategy mode: test fundamentally different scoring approaches
    if args.mode == "strategy":
        run_strategy_mode(args, config, bars)
        return

    # --- 3. Build grid ---
    grid = build_grid(args.strategy)
    print(f"Phase 3: Running {len(grid)} parameter combinations...")
    print()

    # --- 4. Run all combos (fast!) ---
    initial_bal = config.backtesting.initial_balance
    fee_rate = config.fees.active_fee_rate
    sl_strategy = config.trading.stop_loss_strategy
    min_vol = config.filters.min_volatility_pct
    min_adx_default = config.filters.min_adx

    # Progressive CSV output
    csv_path = Path(args.output)
    csv_fields = [
        "rank", "exit_strategy", "leverage", "atr_sl_mult",
        "tp1_rr", "tp2_rr", "tp1_exit_pct",
        "trail_atr_mult", "trail_activation_atr",
        "marginal_low", "strong_thresh", "min_cat_agree", "trend_mom_agree",
        "skip_choppy", "skip_volatile",
        "trades", "win_rate", "return_pct", "net_pnl", "max_dd_pct",
        "profit_factor", "sharpe", "calmar", "final_balance", "fees",
        "avg_win", "avg_loss",
    ]

    all_results: list[dict] = []
    t_start = time.time()
    combos_done = 0
    total_combos = len(grid)

    for params in grid:
        stats = fast_backtest(
            bars,
            leverage=params["leverage"],
            atr_sl_mult=params["atr_sl_mult"],
            tp1_rr=params["tp1_rr"],
            tp2_rr=params["tp2_rr"],
            tp1_exit_pct=params["tp1_exit_pct"],
            marginal_low=params["marginal_low"],
            strong_thresh=params["strong_thresh"],
            min_adx=min_adx_default,
            min_volatility_pct=min_vol,
            min_category_agreement=params["min_cat_agree"],
            require_trend_momentum_agree=params["trend_mom_agree"],
            skip_choppy=params["skip_choppy"],
            skip_volatile=params.get("skip_volatile", False),
            sl_strategy=sl_strategy,
            initial_balance=initial_bal,
            fee_rate=fee_rate,
            exit_strategy=args.strategy,
            trail_atr_mult=params.get("trail_atr_mult", 0.0),
            trail_activation_atr=params.get("trail_activation_atr", 0.0),
        )

        # Calmar ratio
        calmar = stats["return_pct"] / stats["max_dd_pct"] if stats["max_dd_pct"] > 0 else 0
        stats["calmar"] = round(calmar, 2)

        row = {"exit_strategy": args.strategy, **params, **stats}
        all_results.append(row)
        combos_done += 1

        # Progress
        if combos_done % 500 == 0 or combos_done == total_combos:
            elapsed = time.time() - t_start
            rate = combos_done / elapsed if elapsed > 0 else 0
            eta = (total_combos - combos_done) / rate if rate > 0 else 0
            best_so_far = max(all_results, key=lambda x: x.get(args.sort_by, 0))
            print(
                f"  [{combos_done:>6}/{total_combos}] "
                f"{elapsed:.0f}s elapsed | {rate:.0f} combos/s | ETA: {eta:.0f}s | "
                f"Best {args.sort_by}: {best_so_far[args.sort_by]}"
            )

    total_time = time.time() - t_start
    print(f"\n  All {total_combos} combos done in {total_time:.1f}s "
          f"({total_combos/total_time:.0f} combos/sec)")

    # --- 5. Sort and rank ---
    # Filter out combos with < 10 trades (not meaningful)
    meaningful = [r for r in all_results if r["trades"] >= 10]
    print(f"  {len(meaningful)}/{len(all_results)} combos had >= 10 trades")

    if args.sort_by == "calmar":
        meaningful.sort(key=lambda x: x["calmar"], reverse=True)
    elif args.sort_by == "sharpe":
        meaningful.sort(key=lambda x: x["sharpe"], reverse=True)
    elif args.sort_by == "profit_factor":
        meaningful.sort(key=lambda x: x["profit_factor"], reverse=True)
    else:
        meaningful.sort(key=lambda x: x["return_pct"], reverse=True)

    for i, r in enumerate(meaningful):
        r["rank"] = i + 1

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(meaningful)
    print(f"  Results saved to {csv_path}")

    # --- 6. Print top results ---
    print()
    print("=" * 120)
    print(f"                              TOP {args.top} RESULTS (sorted by {args.sort_by})")
    print("=" * 120)
    print(
        f"{'#':>3} | {'Lev':>3} | {'SL':>4} | {'TP1':>4} | {'TP2':>4} | {'Exit%':>5} | "
        f"{'Trail':>5} | {'TrAct':>5} | "
        f"{'Marg':>4} | {'Str':>3} | {'Cat':>3} | {'T+M':>3} | {'Chop':>4} |"
        f" {'Trades':>6} | {'WR%':>5} | {'Return':>8} | {'MaxDD':>6} | {'PF':>5} | "
        f"{'Sharpe':>6} | {'Calmar':>6} | {'Balance':>10}"
    )
    print("-" * 140)

    for r in meaningful[:args.top]:
        print(
            f"{r['rank']:>3} | {r['leverage']:>3} | {r['atr_sl_mult']:>4.1f} | "
            f"{r['tp1_rr']:>4.1f} | {r['tp2_rr']:>4.1f} | {r['tp1_exit_pct']:>5.1f} | "
            f"{r.get('trail_atr_mult', 0):>5.1f} | {r.get('trail_activation_atr', 0):>5.1f} | "
            f"{r['marginal_low']:>4} | {r['strong_thresh']:>3} | {r['min_cat_agree']:>3} | "
            f"{'Y' if r['trend_mom_agree'] else 'N':>3} | "
            f"{'Y' if r['skip_choppy'] else 'N':>4} | "
            f"{r['trades']:>6} | {r['win_rate']:>5.1f} | {r['return_pct']:>+7.1f}% | "
            f"{r['max_dd_pct']:>5.1f}% | {r['profit_factor']:>5.2f} | "
            f"{r['sharpe']:>6.2f} | {r['calmar']:>6.2f} | ${r['final_balance']:>9.2f}"
        )

    print("=" * 120)

    # --- 7. Best by different criteria ---
    print("\n--- Best by Category ---")

    if meaningful:
        best_return = max(meaningful, key=lambda x: x["return_pct"])
        best_sharpe = max(meaningful, key=lambda x: x["sharpe"])
        best_calmar = max(meaningful, key=lambda x: x["calmar"])
        best_pf = max(meaningful, key=lambda x: x["profit_factor"])
        safe = [r for r in meaningful if r["max_dd_pct"] <= 15]
        best_safe = max(safe, key=lambda x: x["return_pct"]) if safe else None

        def _fmt(r, label):
            trail_info = ""
            if r.get("trail_atr_mult", 0) > 0:
                trail_info = f" Trail={r['trail_atr_mult']}"
                if r.get("trail_activation_atr", 0) > 0:
                    trail_info += f" TrAct={r['trail_activation_atr']}"
            return (
                f"  {label}: Lev={r['leverage']} SL={r['atr_sl_mult']} "
                f"TP1={r['tp1_rr']} TP2={r['tp2_rr']} Exit={r['tp1_exit_pct']}"
                f"{trail_info} "
                f"Marg={r['marginal_low']} Strong={r['strong_thresh']} "
                f"Cat={r['min_cat_agree']} T+M={'Y' if r['trend_mom_agree'] else 'N'} "
                f"Choppy={'Y' if r['skip_choppy'] else 'N'}"
                f"\n    -> {r['return_pct']:+.1f}% return | {r['max_dd_pct']:.1f}% DD | "
                f"PF={r['profit_factor']:.2f} | Sharpe={r['sharpe']:.2f} | "
                f"Calmar={r['calmar']:.2f} | {r['trades']} trades"
            )

        print(_fmt(best_return, "Best Return      "))
        print(_fmt(best_sharpe, "Best Sharpe      "))
        print(_fmt(best_calmar, "Best Calmar      "))
        print(_fmt(best_pf,     "Best Profit Fac  "))
        if best_safe:
            print(_fmt(best_safe, "Best (DD<=15%)   "))

    print(f"\nFull results: {csv_path}")


if __name__ == "__main__":
    main()
