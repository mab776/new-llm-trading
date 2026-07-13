"""TRAIN-only search for causal market-regime strategy overlays."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from opt.driver import FOLDS, TEST_FOLDS, TRAIN_FOLDS
from opt.multi_asset import simulate_multi
from opt.multi_portfolio import evaluate_shared, load_assets


CHRONO_FOLDS = [
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-01"),
]


def sample_candidate(rng: random.Random) -> dict:
    """A bounded overlay around the static, already validated strategy."""
    return {
        "regime_threshold_mults": {
            "trending": round(rng.uniform(.88, 1.08), 2),
            "weak_trend": round(rng.uniform(.95, 1.18), 2),
            "ranging": round(rng.uniform(1.0, 1.30), 2),
            "volatile": round(rng.uniform(.95, 1.25), 2),
        },
        "regime_leverage_mults": {
            "volatile": round(rng.uniform(.50, 1.0), 2),
            "ranging": round(rng.uniform(.70, 1.0), 2),
        },
        "regime_trailing_activation_mults": {
            "volatile": round(rng.uniform(1.0, 1.50), 2),
        },
        "regime_trailing_callback_mults": {
            "volatile": round(rng.uniform(1.0, 2.0), 2),
        },
    }


def objective(result: dict) -> float:
    if result["worst_fold"] <= 0 or result["max_dd"] > 45:
        return -1e12
    return result["geo_pct"] - .35 * result["max_dd"]


def evaluate(assets, folds, candidate) -> dict:
    return evaluate_shared(
        assets, folds=folds, slip=.0002,
        exit_granularity="sub", strat=candidate,
    )


def run(trials_per_seed: int, seeds: list[int]) -> dict:
    assets = load_assets("maker", "aggressive")
    baseline = {
        "train": evaluate(assets, TRAIN_FOLDS, {}),
        "held_out_test": evaluate(assets, TEST_FOLDS, {}),
        "chronological": evaluate(assets, CHRONO_FOLDS, {}),
        "annual": evaluate(assets, FOLDS, {}),
    }
    ranked_all = [(objective(baseline["train"]), {}, baseline["train"])]
    seed_winners = []
    for seed in seeds:
        rng = random.Random(seed)
        ranked = [(objective(baseline["train"]), {}, baseline["train"])]
        for _ in range(trials_per_seed):
            candidate = sample_candidate(rng)
            trained = evaluate(assets, TRAIN_FOLDS, candidate)
            ranked.append((objective(trained), candidate, trained))
        ranked.sort(key=lambda row: row[0], reverse=True)
        ranked_all.extend(ranked[1:])
        score, winner, trained = ranked[0]
        seed_winners.append({
            "seed": seed, "selection_score": score, "candidate": winner,
            "train": trained,
            "held_out_test": evaluate(assets, TEST_FOLDS, winner),
            "chronological": evaluate(assets, CHRONO_FOLDS, winner),
        })
        print(
            f"seed={seed} train={trained['compound_x']:.2f}x "
            f"test={seed_winners[-1]['held_out_test']['compound_x']:.2f}x "
            f"chrono={seed_winners[-1]['chronological']['compound_x']:.2f}x"
        )

    ranked_all.sort(key=lambda row: row[0], reverse=True)
    score, selected, selected_train = ranked_all[0]
    selected_results = {
        "selection_score": score,
        "candidate": selected,
        "train": selected_train,
        "held_out_test": evaluate(assets, TEST_FOLDS, selected),
        "chronological": evaluate(assets, CHRONO_FOLDS, selected),
        "annual": evaluate(assets, FOLDS, selected),
    }
    continuous = simulate_multi(
        assets, "2021-01-01", "2025-06-01", slip=.0002,
        exit_granularity="sub", strat=selected,
    )
    selected_results["continuous"] = {
        "compound_x": max(.01, 1 + continuous.return_pct / 100),
        "reported_max_dd": continuous.max_dd_pct,
        "mark_to_market_max_dd": max(
            (point.drawdown_pct for point in continuous.equity_curve), default=0.0
        ),
        "trades": continuous.trades,
    }
    baseline_continuous = simulate_multi(
        assets, "2021-01-01", "2025-06-01", slip=.0002,
        exit_granularity="sub",
    )
    baseline["continuous"] = {
        "compound_x": max(.01, 1 + baseline_continuous.return_pct / 100),
        "reported_max_dd": baseline_continuous.max_dd_pct,
        "mark_to_market_max_dd": max(
            (point.drawdown_pct for point in baseline_continuous.equity_curve),
            default=0.0,
        ),
        "trades": baseline_continuous.trades,
    }
    stable_test_wins = sum(
        row["held_out_test"]["compound_x"]
        > baseline["held_out_test"]["compound_x"]
        for row in seed_winners
    )
    checks = {
        "held_out_test_improves": (
            selected_results["held_out_test"]["compound_x"]
            > baseline["held_out_test"]["compound_x"]
        ),
        "chronological_improves": (
            selected_results["chronological"]["compound_x"]
            > baseline["chronological"]["compound_x"]
        ),
        "continuous_improves": (
            selected_results["continuous"]["compound_x"]
            > baseline["continuous"]["compound_x"]
        ),
        "annual_every_fold_green": selected_results["annual"]["worst_fold"] > 0,
        "drawdown_within_two_points": (
            max(selected_results["continuous"]["reported_max_dd"],
                selected_results["continuous"]["mark_to_market_max_dd"])
            <= max(baseline["continuous"]["reported_max_dd"],
                   baseline["continuous"]["mark_to_market_max_dd"]) + 2
        ),
        "majority_seed_winners_improve_test": stable_test_wins > len(seed_winners) / 2,
    }
    return {
        "method": ("select only on interleaved TRAIN half-years; validate held-out TEST, "
                   "chronological, annual, and continuous; maker entry; honest 1h exits "
                   "with 4h-close trailing; funding; liquidation; 2bps market slippage"),
        "trials_per_seed": trials_per_seed,
        "seeds": seeds,
        "baseline": baseline,
        "seed_winners": seed_winners,
        "selected": selected_results,
        "acceptance_checks": checks,
        "adopt_recommended": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-per-seed", type=int, default=60)
    parser.add_argument("--seeds", default="17,73,211,419,887")
    parser.add_argument("--output", default="opt/regime_search_results.json")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    result = run(args.trials_per_seed, seeds)
    path = Path(args.output)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["acceptance_checks"], indent=2))
    print(f"adopt_recommended={result['adopt_recommended']}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
