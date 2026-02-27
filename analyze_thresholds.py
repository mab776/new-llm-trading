#!/usr/bin/env python3
"""Analyze the relationship between score strength and signal quality."""

import pandas as pd
import numpy as np
from llm_trading_bot.config import load_config
from llm_trading_bot.data import fetch_multi_timeframe
from llm_trading_bot.scoring import calculate_indicators, compute_composite_score


def main():
    config = load_config("config.json")
    data = fetch_multi_timeframe(
        config.trading.yfinance_symbol, ["4h"],
        config.backtesting.start_date, config.backtesting.end_date,
        config.backtesting.warmup_periods,
    )
    df = data["4h"]

    start = pd.Timestamp(config.backtesting.start_date, tz="UTC")
    end = pd.Timestamp(config.backtesting.end_date, tz="UTC")
    test_idx = df.index[(df.index >= start) & (df.index <= end)]

    # Score EVERY bar for proper analysis
    print(f"Scoring {len(test_idx)} bars...")
    results = []
    for idx in test_idx:
        bar_loc = df.index.get_loc(idx)
        if bar_loc < 200:
            continue
        sliced = df.iloc[:bar_loc + 1]
        ind = calculate_indicators(sliced, "4h")
        r = compute_composite_score(
            {"4h": ind}, config.scoring.weights, "4h", 5, 95
        )

        # Lookahead: 6 bars
        end_loc = min(bar_loc + 6, len(df) - 1)
        if end_loc <= bar_loc:
            continue
        future = df.iloc[bar_loc + 1 : end_loc + 1]
        if len(future) == 0:
            continue
        close_now = float(df.iloc[bar_loc]["Close"])
        close_then = float(future.iloc[-1]["Close"])
        move = (close_then - close_now) / close_now * 100

        direction = r.direction.value
        abs_score = abs(r.raw_score)

        if direction == "BULLISH":
            correct = move > 0.1
        elif direction == "BEARISH":
            correct = move < -0.1
        else:
            correct = abs(move) < 1.0

        # Category agreement: how many categories agree with composite direction?
        n_agree = 0
        n_cats = 0
        for cat in r.category_scores:
            n_cats += 1
            if (r.raw_score > 0 and cat.raw_score > 0) or (r.raw_score < 0 and cat.raw_score < 0):
                n_agree += 1
        agreement = n_agree / n_cats if n_cats else 0

        results.append({
            "score": r.raw_score,
            "abs_score": abs_score,
            "dir": direction,
            "correct": bool(correct),
            "move": move,
            "agreement": agreement,
        })

    months = 2.0
    print(f"Analyzed {len(results)} bars\n")

    # 1. Accuracy by threshold
    print("=" * 70)
    print("Accuracy by |Score| Threshold")
    print("=" * 70)
    print(f"{'Thresh':>6s} | {'Signals':>7s} | {'Correct':>7s} | {'Accuracy':>8s} | {'AvgWinMove':>10s} | {'Sig/month':>9s}")
    print("-" * 70)

    for threshold in [10, 15, 20, 25, 30, 35, 40, 45]:
        directional = [r for r in results if r["abs_score"] >= threshold and r["dir"] != "NEUTRAL"]
        if not directional:
            print(f"{threshold:>6d} | {0:>7d} |       - |        - |          - |       0.0")
            continue
        correct = sum(1 for r in directional if r["correct"])
        avg_move_win = np.mean([abs(r["move"]) for r in directional if r["correct"]]) if correct else 0
        signals_per_mo = len(directional) / months
        acc = correct / len(directional) * 100
        print(f"{threshold:>6d} | {len(directional):>7d} | {correct:>7d} | {acc:>7.1f}% | {avg_move_win:>+9.2f}% | {signals_per_mo:>9.1f}")

    # 2. Accuracy by direction
    print()
    print("=" * 70)
    print("Accuracy by Direction at Different Thresholds")
    print("=" * 70)
    for threshold in [10, 15, 20, 25, 30]:
        longs = [r for r in results if r["score"] >= threshold]
        shorts = [r for r in results if r["score"] <= -threshold]
        l_acc = sum(1 for r in longs if r["correct"]) / len(longs) * 100 if longs else 0
        s_acc = sum(1 for r in shorts if r["correct"]) / len(shorts) * 100 if shorts else 0
        l_avg = np.mean([r["move"] for r in longs]) if longs else 0
        s_avg = np.mean([r["move"] for r in shorts]) if shorts else 0
        print(
            f"  |Score| >= {threshold:2d}: "
            f"LONG {len(longs):3d} sig ({l_acc:.0f}% acc, avg {l_avg:+.2f}%) | "
            f"SHORT {len(shorts):3d} sig ({s_acc:.0f}% acc, avg {s_avg:+.2f}%)"
        )

    # 3. Score distribution
    print()
    print("=" * 70)
    print("Score Distribution (percentiles)")
    print("=" * 70)
    all_scores = [r["score"] for r in results]
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  P{p:2d}: {np.percentile(all_scores, p):+.1f}")

    # 4. Category agreement filter
    print()
    print("=" * 70)
    print("Category Agreement Filter (does agreement improve accuracy?)")
    print("=" * 70)
    for min_agree in [0.4, 0.6, 0.8]:
        directional = [r for r in results if r["dir"] != "NEUTRAL" and r["abs_score"] >= 15]
        filtered = [r for r in directional if r["agreement"] >= min_agree]
        if not filtered:
            continue
        corr = sum(1 for r in filtered if r["correct"])
        acc = corr / len(filtered) * 100
        avg_move = np.mean([abs(r["move"]) for r in filtered if r["correct"]]) if corr else 0
        print(
            f"  Agreement >= {min_agree:.0%}: "
            f"{len(filtered):3d} signals, {acc:.1f}% accurate, "
            f"avg win move: {avg_move:+.2f}%"
        )

    # 5. Combined: score + agreement
    print()
    print("=" * 70)
    print("Best Combos: Threshold + Agreement")
    print("=" * 70)
    print(f"{'Thresh':>6s} | {'Agree':>5s} | {'Signals':>7s} | {'Acc':>5s} | {'AvgWin':>7s} | {'Sig/mo':>6s} | {'Score'}")
    print("-" * 70)

    best_score = 0
    best_combo = None
    for threshold in [10, 15, 20, 25, 30]:
        for min_agree in [0.4, 0.6, 0.8]:
            filtered = [
                r for r in results
                if r["dir"] != "NEUTRAL"
                and r["abs_score"] >= threshold
                and r["agreement"] >= min_agree
            ]
            if len(filtered) < 3:
                continue
            corr = sum(1 for r in filtered if r["correct"])
            acc = corr / len(filtered) * 100
            avg_win = np.mean([abs(r["move"]) for r in filtered if r["correct"]]) if corr else 0
            sig_mo = len(filtered) / months
            # Score: balance accuracy and trade frequency
            # We want high accuracy with reasonable frequency
            combo_score = acc * min(sig_mo, 20) / 20 * (1 + avg_win / 3)
            marker = ""
            if combo_score > best_score:
                best_score = combo_score
                best_combo = (threshold, min_agree)
                marker = " <-- BEST"
            print(
                f"{threshold:>6d} | {min_agree:>4.0%} | {len(filtered):>7d} | {acc:>4.1f}% | {avg_win:>+6.2f}% | {sig_mo:>6.1f} | {combo_score:>5.1f}{marker}"
            )

    if best_combo:
        print(f"\n>>> Recommended: threshold={best_combo[0]}, min_agreement={best_combo[1]:.0%}")
        print(f"    (best balance of accuracy × frequency × avg win size)")


if __name__ == "__main__":
    main()
