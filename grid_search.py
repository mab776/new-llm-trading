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
from typing import Optional

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
) -> dict:
    """
    Run a complete backtest on pre-computed bars. Returns stats dict.
    ~1000x faster than the full engine because no indicator recalculation.
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

            # SL check first (conservative)
            sl_hit = (l <= trade.stop_loss) if is_long else (h >= trade.stop_loss)
            if sl_hit:
                gross = ((trade.stop_loss - trade.entry_price) if is_long
                         else (trade.entry_price - trade.stop_loss)) * trade.remaining_size * trade.leverage
                ex_fee = _sim_fee(trade.remaining_size, trade.stop_loss, fee_rate)
                net = gross - ex_fee
                trade.net_pnl += net
                balance += net
                total_fees += ex_fee
                trade.is_open = False
                closed_pnls.append(trade.net_pnl)
                if balance > peak_balance:
                    peak_balance = balance
                dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
                if dd > max_dd_pct:
                    max_dd_pct = dd
                trade = None
                # Continue to potentially open a new trade this bar
            else:
                # TP1 partial
                if not trade.partial_done:
                    tp1_hit = (h >= trade.tp1) if is_long else (l <= trade.tp1)
                    if tp1_hit:
                        exit_size = trade.remaining_size * trade.tp1_exit_pct
                        gross = ((trade.tp1 - trade.entry_price) if is_long
                                 else (trade.entry_price - trade.tp1)) * exit_size * trade.leverage
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
                                 else (trade.entry_price - trade.tp2)) * trade.remaining_size * trade.leverage
                        ex_fee = _sim_fee(trade.remaining_size, trade.tp2, fee_rate)
                        net = gross - ex_fee
                        trade.net_pnl += net
                        balance += net
                        total_fees += ex_fee
                        trade.is_open = False
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

        abs_score = abs(bar.raw_score)
        if abs_score < marginal_low:
            continue  # WAIT signal

        if bar.direction == Direction.NEUTRAL:
            continue

        # Calculate targets
        atr = bar.atr_14
        if not atr or atr == 0:
            continue

        is_long = bar.direction == Direction.BULLISH
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
        )

    # Force-close at end
    if trade and trade.is_open:
        last = bars[-1]
        is_long = trade.direction == "LONG"
        gross = ((last.close - trade.entry_price) if is_long
                 else (trade.entry_price - last.close)) * trade.remaining_size * trade.leverage
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

        # Regime
        regime = detect_market_regime(primary_ind)

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
        ))

    elapsed = time.time() - t0
    print(f"  Done: {len(bars)} bars pre-computed in {elapsed:.1f}s")
    return bars


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

def build_grid() -> list[dict]:
    """
    Build the full parameter grid. Filters out invalid combinations
    (e.g. tp2 <= tp1, strong <= marginal).
    """
    grid = {
        "leverage":        [3, 5, 7, 10, 15, 20],
        "atr_sl_mult":     [0.8, 1.0, 1.2, 1.5, 2.0, 2.5],
        "tp1_rr":          [1.0, 1.5, 2.0, 2.5, 3.0],
        "tp2_rr":          [2.0, 3.0, 4.0, 5.0, 6.0],
        "tp1_exit_pct":    [0.3, 0.5, 0.7],
        "marginal_low":    [15, 20, 25, 30],
        "strong_thresh":   [25, 30, 35, 40, 45],
        "min_cat_agree":   [2, 3, 4],
        "trend_mom_agree": [True, False],
        "skip_choppy":     [True, False],
        "skip_volatile":   [False],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    valid = []
    for combo in combos:
        params = dict(zip(keys, combo))
        # Constraints
        if params["tp2_rr"] <= params["tp1_rr"]:
            continue
        if params["strong_thresh"] <= params["marginal_low"]:
            continue
        valid.append(params)

    return valid


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

    # --- 3. Build grid ---
    grid = build_grid()
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
        "rank", "leverage", "atr_sl_mult", "tp1_rr", "tp2_rr", "tp1_exit_pct",
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
        )

        # Calmar ratio
        calmar = stats["return_pct"] / stats["max_dd_pct"] if stats["max_dd_pct"] > 0 else 0
        stats["calmar"] = round(calmar, 2)

        row = {**params, **stats}
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
        f"{'Marg':>4} | {'Str':>3} | {'Cat':>3} | {'T+M':>3} | {'Chop':>4} |"
        f" {'Trades':>6} | {'WR%':>5} | {'Return':>8} | {'MaxDD':>6} | {'PF':>5} | "
        f"{'Sharpe':>6} | {'Calmar':>6} | {'Balance':>10}"
    )
    print("-" * 120)

    for r in meaningful[:args.top]:
        print(
            f"{r['rank']:>3} | {r['leverage']:>3} | {r['atr_sl_mult']:>4.1f} | "
            f"{r['tp1_rr']:>4.1f} | {r['tp2_rr']:>4.1f} | {r['tp1_exit_pct']:>5.1f} | "
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
            return (
                f"  {label}: Lev={r['leverage']} SL={r['atr_sl_mult']} "
                f"TP1={r['tp1_rr']} TP2={r['tp2_rr']} Exit={r['tp1_exit_pct']} "
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
