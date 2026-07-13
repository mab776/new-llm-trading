"""Multi-seed/search-size robustness study for annual walk-forward retuning."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import statistics
from pathlib import Path

from opt.walk_forward_retune import run


DEFAULT_SEEDS = [17, 73, 211, 419, 887]


def experiment_matrix(seeds: list[int]) -> list[tuple[int, int, float]]:
    matrix = [(60, seed, 0.0) for seed in seeds]
    matrix += [(300, seed, 0.0) for seed in seeds]
    matrix += [(1000, seed, 0.0) for seed in seeds[:3]]
    matrix += [(300, seed, 15.0) for seed in seeds]
    return matrix


def summarize(runs: list[dict]) -> list[dict]:
    groups: dict[tuple[int, float], list[dict]] = {}
    for row in runs:
        groups.setdefault(
            (row["trials_per_window"], row["turnover_penalty"]), []
        ).append(row)
    summaries = []
    for (trials, penalty), rows in sorted(groups.items()):
        ratios = [row["growth_ratio"] for row in rows]
        tuned = [row["walk_forward_compound_x"] for row in rows]
        turnover = [row["total_parameter_turnover"] for row in rows]
        window_wins = [
            sum(window["unseen"]["mean_ret"] > window["static"]["mean_ret"]
                for window in row["windows"])
            for row in rows
        ]
        recent = [
            next(window for window in row["windows"]
                 if window["target"] == "2025H1")
            for row in rows
        ]
        winner_stability = {}
        for target in ("2023", "2024", "2025H1"):
            winners = [
                next(window for window in row["windows"]
                     if window["target"] == target)["winner"]
                for row in rows
            ]
            counts = Counter(json.dumps(winner, sort_keys=True) for winner in winners)
            winner_stability[target] = {
                "unique_winners": len(counts),
                "most_common_fraction": max(counts.values()) / len(winners),
                "static_fraction": sum(not winner for winner in winners) / len(winners),
            }
        summaries.append({
            "trials_per_window": trials,
            "turnover_penalty": penalty,
            "seeds": [row["seed"] for row in rows],
            "growth_ratio_median": statistics.median(ratios),
            "growth_ratio_min": min(ratios),
            "growth_ratio_max": max(ratios),
            "fraction_beating_static": sum(value > 1 for value in ratios) / len(ratios),
            "walk_forward_x_median": statistics.median(tuned),
            "parameter_turnover_median": statistics.median(turnover),
            "unseen_window_wins_median": statistics.median(window_wins),
            "recent_2025h1_win_fraction": sum(
                row["unseen"]["mean_ret"] > row["static"]["mean_ret"]
                for row in recent
            ) / len(recent),
            "winner_stability": winner_stability,
        })
    return summaries


def run_study(seeds: list[int]) -> dict:
    runs = []
    evaluation_cache = {}
    for trials, seed, penalty in experiment_matrix(seeds):
        result = run(
            trials, seed, entry_mode="maker", exit_granularity="sub",
            turnover_penalty=penalty,
            _evaluation_cache=evaluation_cache,
        )
        runs.append(result)
        print(
            f"trials={trials:4d} seed={seed:3d} penalty={penalty:4.1f} "
            f"ratio={result['growth_ratio']:.3f} "
            f"turnover={result['total_parameter_turnover']:.3f}"
        )
    return {
        "method": ("annual trailing-two-year selection, next-year unseen trade; "
                   "maker entry; honest sub-bar exits with 4h-close trailing; "
                   "funding; liquidation; 2bps market slippage"),
        "summaries": summarize(runs),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--output", default="opt/walk_forward_robustness_results.json")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    result = run_study(seeds)
    path = Path(args.output)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summaries"], indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
