#!/usr/bin/env python3
"""
Debug script to dump composite scores across the backtest period.
Useful for diagnosing directional bias, score ranges, and signal gaps.
Also validates signals by checking what price actually did afterwards.

Usage:
    python debug_scores.py                      # Uses config.json defaults
    python debug_scores.py --start 2026-01-01 --end 2026-02-27
    python debug_scores.py --step 1             # Every bar (default: every 6th)
    python debug_scores.py --detail             # Show category breakdown
    python debug_scores.py --lookahead 12       # Check 12 bars ahead (default: 6)
"""

import argparse
import pandas as pd

from llm_trading_bot.config import load_config
from llm_trading_bot.data import fetch_multi_timeframe
from llm_trading_bot.scoring import calculate_indicators, compute_composite_score


def validate_signal(df: pd.DataFrame, bar_loc: int, direction: str, lookahead: int) -> dict:
    """
    Look ahead N bars from the signal to see if the direction was correct.

    Returns dict with:
      - move_pct: actual % price move over the lookahead window
      - max_favorable: best % move in the predicted direction
      - max_adverse: worst % move against predicted direction
      - correct: whether the net move agreed with the signal direction
      - grade: ✓✓ (strong correct), ✓ (correct), ✗ (wrong), ✗✗ (badly wrong), - (neutral)
    """
    close_at_signal = df.iloc[bar_loc]["Close"]
    end_loc = min(bar_loc + lookahead, len(df) - 1)

    if end_loc <= bar_loc:
        return {"move_pct": 0, "max_favorable": 0, "max_adverse": 0, "correct": None, "grade": "?"}

    future_slice = df.iloc[bar_loc + 1 : end_loc + 1]
    if len(future_slice) == 0:
        return {"move_pct": 0, "max_favorable": 0, "max_adverse": 0, "correct": None, "grade": "?"}

    future_highs = future_slice["High"]
    future_lows = future_slice["Low"]
    close_at_end = float(future_slice.iloc[-1]["Close"])

    move_pct = (close_at_end - close_at_signal) / close_at_signal * 100

    if direction == "BULLISH":
        max_favorable = (float(future_highs.max()) - close_at_signal) / close_at_signal * 100
        max_adverse = (float(future_lows.min()) - close_at_signal) / close_at_signal * 100
        correct = bool(move_pct > 0.1)
    elif direction == "BEARISH":
        max_favorable = (close_at_signal - float(future_lows.min())) / close_at_signal * 100
        max_adverse = (float(future_highs.max()) - close_at_signal) / close_at_signal * 100
        correct = bool(move_pct < -0.1)
    else:  # NEUTRAL
        max_favorable = max(
            (float(future_highs.max()) - close_at_signal) / close_at_signal * 100,
            (close_at_signal - float(future_lows.min())) / close_at_signal * 100,
        )
        max_adverse = 0
        correct = bool(abs(move_pct) < 1.0)  # Neutral is "correct" if price didn't move much

    # Grade: how good was the call?
    if direction == "NEUTRAL":
        grade = "—" if abs(move_pct) < 1.0 else "✗" if abs(move_pct) > 2.0 else "~"
    else:
        if correct and abs(move_pct) > 1.5:
            grade = "✓✓"
        elif correct:
            grade = "✓"
        elif abs(move_pct) < 0.2:
            grade = "~"
        elif abs(move_pct) > 1.5:
            grade = "✗✗"
        else:
            grade = "✗"

    return {
        "move_pct": move_pct,
        "max_favorable": max_favorable,
        "max_adverse": max_adverse,
        "correct": correct,
        "grade": grade,
    }


def main():
    parser = argparse.ArgumentParser(description="Dump composite scores for diagnosis")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD), default from config")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), default from config")
    parser.add_argument("--step", type=int, default=6, help="Sample every Nth bar (default: 6 ≈ daily)")
    parser.add_argument("--detail", action="store_true", help="Show per-category breakdown")
    parser.add_argument("--timeframe", default=None, help="Primary timeframe (default from config)")
    parser.add_argument("--lookahead", type=int, default=6, help="Bars to look ahead for validation (default: 6)")
    args = parser.parse_args()

    config = load_config(args.config)
    start_date = args.start or config.backtesting.start_date
    end_date = args.end or config.backtesting.end_date
    tf = args.timeframe or config.trading.primary_timeframe

    print(f"Fetching {tf} data for {config.trading.yfinance_symbol}...")
    data = fetch_multi_timeframe(
        config.trading.yfinance_symbol,
        [tf],
        start_date,
        end_date,
        config.backtesting.warmup_periods,
    )
    df = data[tf]

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    test_idx = df.index[(df.index >= start_ts) & (df.index <= end_ts)]

    if len(test_idx) == 0:
        print("No bars found in the specified date range.")
        return

    print(f"Scoring {len(test_idx)} bars (sampling every {args.step}, validating {args.lookahead} bars ahead)...\n")

    # Counters
    bullish_ct = bearish_ct = neutral_ct = 0
    correct_ct = wrong_ct = unclear_ct = 0
    max_score = (-999, "")
    min_score = (999, "")
    rows = []  # Collect for summary stats

    header = f"{'Date':>10s} | {'Score':>6s} | {'Dir':>7s} | {'Move%':>6s} | {'Best%':>6s} | {'Wrst%':>6s} | Grade"
    if args.detail:
        header += " | Trend  | Momntm | Volume | Sup/Res | Risk"
    print(header)
    print("-" * len(header))

    for idx in test_idx[:: args.step]:
        bar_loc = df.index.get_loc(idx)
        if bar_loc < config.backtesting.warmup_periods:
            continue

        sliced = df.iloc[: bar_loc + 1]
        ind = calculate_indicators(sliced, tf)
        result = compute_composite_score(
            {tf: ind},
            config.scoring.weights,
            tf,
            config.scoring.confidence_min,
            config.scoring.confidence_max,
        )

        date_str = str(idx)[:10]
        direction = result.direction.value
        score = result.raw_score

        # Track stats
        if direction == "BULLISH":
            bullish_ct += 1
        elif direction == "BEARISH":
            bearish_ct += 1
        else:
            neutral_ct += 1

        if score > max_score[0]:
            max_score = (score, date_str)
        if score < min_score[0]:
            min_score = (score, date_str)

        # Validate: what did price actually do?
        val = validate_signal(df, bar_loc, direction, args.lookahead)

        if val["correct"] is True:
            correct_ct += 1
        elif val["correct"] is False:
            wrong_ct += 1
        else:
            unclear_ct += 1

        rows.append({"direction": direction, "score": score, "correct": val["correct"],
                      "move_pct": val["move_pct"], "grade": val["grade"]})

        line = (
            f"{date_str} | {score:+6.1f} | {direction:>7s}"
            f" | {val['move_pct']:+5.1f}% | {val['max_favorable']:+5.1f}% | {val['max_adverse']:+5.1f}% | {val['grade']:>4s}"
        )

        if args.detail and result.category_scores:
            cats = {c.name: c.raw_score for c in result.category_scores}
            line += (
                f" | {cats.get('trend', 0):+6.1f}"
                f" | {cats.get('momentum', 0):+6.1f}"
                f" | {cats.get('volume', 0):+6.1f}"
                f" | {cats.get('support_resistance', 0):+7.1f}"
                f" | {cats.get('risk', 0):+5.1f}"
            )

        print(line)

    # Summary
    total = bullish_ct + bearish_ct + neutral_ct
    tier = config.trading.active_leverage_tier
    print(f"\n{'='*60}")
    print(f"Summary ({total} sampled bars, {args.lookahead}-bar lookahead)")
    print(f"{'='*60}")
    print(f"  Bullish:  {bullish_ct:3d} ({bullish_ct/total*100:.0f}%)")
    print(f"  Bearish:  {bearish_ct:3d} ({bearish_ct/total*100:.0f}%)")
    print(f"  Neutral:  {neutral_ct:3d} ({neutral_ct/total*100:.0f}%)")
    print(f"  Max score: {max_score[0]:+.1f} on {max_score[1]}")
    print(f"  Min score: {min_score[0]:+.1f} on {min_score[1]}")

    # Validation accuracy
    directional = correct_ct + wrong_ct
    print(f"\n--- Signal Validation ---")
    print(f"  Correct calls:  {correct_ct:3d}" + (f" ({correct_ct/directional*100:.0f}%)" if directional else ""))
    print(f"  Wrong calls:    {wrong_ct:3d}" + (f" ({wrong_ct/directional*100:.0f}%)" if directional else ""))
    print(f"  Unclear/Neutral:{unclear_ct:3d}")
    if directional:
        print(f"  Directional accuracy: {correct_ct/directional*100:.1f}%")

    # Accuracy by direction
    bullish_rows = [r for r in rows if r["direction"] == "BULLISH" and r["correct"] is not None]
    bearish_rows = [r for r in rows if r["direction"] == "BEARISH" and r["correct"] is not None]
    if bullish_rows:
        bull_correct = sum(1 for r in bullish_rows if r["correct"])
        avg_move = sum(r["move_pct"] for r in bullish_rows) / len(bullish_rows)
        print(f"\n  BULLISH signals: {bull_correct}/{len(bullish_rows)} correct ({bull_correct/len(bullish_rows)*100:.0f}%), avg move: {avg_move:+.2f}%")
    if bearish_rows:
        bear_correct = sum(1 for r in bearish_rows if r["correct"])
        avg_move = sum(r["move_pct"] for r in bearish_rows) / len(bearish_rows)
        print(f"  BEARISH signals: {bear_correct}/{len(bearish_rows)} correct ({bear_correct/len(bearish_rows)*100:.0f}%), avg move: {avg_move:+.2f}%")

    # Grade distribution
    grades = {}
    for r in rows:
        g = r["grade"]
        grades[g] = grades.get(g, 0) + 1
    print(f"\n--- Grade Distribution ---")
    for g in ["✓✓", "✓", "~", "—", "✗", "✗✗", "?"]:
        if g in grades:
            bar = "█" * grades[g]
            print(f"  {g:>2s}: {grades[g]:3d} {bar}")

    print(f"\n  Marginal threshold: {tier.marginal_threshold_low}")
    print(f"  Strong threshold:   {tier.strong_threshold}")

    if abs(min_score[0]) < tier.marginal_threshold_low:
        print(f"\n  ⚠ Strongest bearish score ({min_score[0]:+.1f}) is below marginal threshold ({tier.marginal_threshold_low})")
        print(f"    → No SHORT trades will trigger. Consider lowering thresholds.")
    if abs(max_score[0]) < tier.marginal_threshold_low:
        print(f"\n  ⚠ Strongest bullish score ({max_score[0]:+.1f}) is below marginal threshold ({tier.marginal_threshold_low})")
        print(f"    → No LONG trades will trigger. Consider lowering thresholds.")


if __name__ == "__main__":
    main()
