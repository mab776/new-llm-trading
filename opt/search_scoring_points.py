"""Constrained search over canonical scoring-internal point values.

Candidates are ranked on interleaved TRAIN halves only.  The single TRAIN winner
is then reported on held-out TEST halves and a chronological 2024-2025 split;
neither validation set influences selection.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import opt.driver as drv


CHRONO_TRAIN = [fold for fold in drv.FOLDS if fold[0] in ("2021", "2022", "2023")]
CHRONO_TEST = [fold for fold in drv.FOLDS if fold[0] in ("2024", "2025")]


def sample_points(rng: random.Random) -> dict[str, float]:
    """Search a deliberately small, interpretable subset of the point surface."""
    return {
        "trend.ema_stack": round(rng.uniform(20, 40), 1),
        "trend.ema200": round(rng.uniform(8, 22), 1),
        "trend.di": round(rng.uniform(12, 28), 1),
        "trend.macd": round(rng.uniform(8, 22), 1),
        "momentum.rsi_trend": round(rng.uniform(8, 22), 1),
        "momentum.roc_strong": round(rng.uniform(8, 22), 1),
        "volume.confirmation": round(rng.uniform(10, 30), 1),
        "volume.obv": round(rng.uniform(8, 22), 1),
        "sr.proximity": round(rng.uniform(15, 35), 1),
    }


def objective(result: dict) -> float:
    if result["worst_fold"] <= 0 or result["max_dd"] > 45:
        return -1e12
    if result["total_trades"] < 250:
        return -1e12
    return result["geo_pct"] - .35 * result["max_dd"]


def _eval(points: dict, folds) -> dict:
    return drv.evaluate(
        {"scoring.points": points}, folds=folds, slip=.0002, funding=True,
        strat={"entry_mode": "maker"}, exit_granularity="sub",
    )


def run(trials: int = 500, seed: int = 113) -> dict:
    drv.setup()
    rng = random.Random(seed)
    baseline_train = _eval({}, drv.TRAIN_FOLDS)
    ranked = [(objective(baseline_train), {}, baseline_train)]
    for _ in range(trials):
        points = sample_points(rng)
        train = _eval(points, drv.TRAIN_FOLDS)
        ranked.append((objective(train), points, train))
    ranked.sort(key=lambda row: row[0], reverse=True)

    # Selection ends here. Validation numbers below are reporting-only.
    _score, winner, train = ranked[0]
    baseline = {
        "train": baseline_train,
        "test": _eval({}, drv.TEST_FOLDS),
        "chrono_train": _eval({}, CHRONO_TRAIN),
        "chrono_test": _eval({}, CHRONO_TEST),
    }
    selected = {
        "points": winner, "train": train,
        "test": _eval(winner, drv.TEST_FOLDS),
        "chrono_train": _eval(winner, CHRONO_TRAIN),
        "chrono_test": _eval(winner, CHRONO_TEST),
    }
    cohort = []
    for _obj, points, training in ranked[:20]:
        cohort.append({
            "points": points, "train": training,
            "test": _eval(points, drv.TEST_FOLDS),
        })
    return {
        "selection_rule": "rank TRAIN only; TEST and chronological split report-only",
        "trials": trials, "seed": seed, "slip": .0002,
        "funding": True, "entry_mode": "maker", "exit_granularity": "sub",
        "baseline": baseline, "selected": selected, "train_top20": cohort,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--output", default="reports/scoring_points_search.json")
    args = parser.parse_args()
    result = run(args.trials, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    base, pick = result["baseline"], result["selected"]
    print(f"TRAIN  static {base['train']['geo_pct']:+.1f}%/f -> "
          f"selected {pick['train']['geo_pct']:+.1f}%/f")
    print(f"TEST   static {base['test']['geo_pct']:+.1f}%/f -> "
          f"selected {pick['test']['geo_pct']:+.1f}%/f")
    print(f"CHRONO 24-25 static {base['chrono_test']['compound_x']:.2f}x -> "
          f"selected {pick['chrono_test']['compound_x']:.2f}x")
    print(f"points={json.dumps(pick['points'], sort_keys=True)}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
