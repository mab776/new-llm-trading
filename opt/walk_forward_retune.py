"""Leakage-free annual walk-forward retuning experiment.

For target year N, candidates are selected using N-2 and N-1 only, then the
single winner is evaluated on N.  The target fold never participates in search
or ranking.  Compare the chained unseen returns with the unchanged static config.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import opt.driver as drv


TRAIN_WINDOWS = [
    ("2023", [("2021", "2021-01-01", "2021-12-31"),
              ("2022", "2022-01-01", "2022-12-31")],
     [("2023", "2023-01-01", "2023-12-31")]),
    ("2024", [("2022", "2022-01-01", "2022-12-31"),
              ("2023", "2023-01-01", "2023-12-31")],
     [("2024", "2024-01-01", "2024-12-31")]),
    ("2025H1", [("2023", "2023-01-01", "2023-12-31"),
                ("2024", "2024-01-01", "2024-12-31")],
     [("2025H1", "2025-01-01", "2025-06-01")]),
]

PARAMETER_RANGES = {
    "tier.strong_threshold": 9.0,
    "tier.marginal_threshold_low": 7.0,
    "tier.tp1_rr": .8,
    "tier.tp2_rr": 1.4,
    "tier.tp1_exit_pct": .25,
    "scoring.atr_sl_multiplier": 1.0,
    "filters.min_adx": 7.0,
    "trailing.activation_pct": .55,
    "trailing.callback_pct": .25,
    "weights.trend": .14,
    "weights.momentum": .15,
    "weights.volume": .13,
    "weights.support_resistance": .12,
    "weights.risk": .09,
}


def sample_candidate(rng: random.Random) -> dict:
    """Constrained neighborhood around the validated static strategy."""
    strong = rng.uniform(18.0, 27.0)
    marginal = rng.uniform(10.0, min(17.0, strong - 2.0))
    weights = {
        "trend": rng.uniform(.15, .29),
        "momentum": rng.uniform(.24, .39),
        "volume": rng.uniform(.14, .27),
        "support_resistance": rng.uniform(.11, .23),
        "risk": rng.uniform(.06, .15),
    }
    return {
        "tier.strong_threshold": round(strong, 2),
        "tier.marginal_threshold_low": round(marginal, 2),
        "tier.tp1_rr": round(rng.uniform(1.65, 2.45), 2),
        "tier.tp2_rr": round(rng.uniform(2.7, 4.1), 2),
        "tier.tp1_exit_pct": round(rng.uniform(.55, .80), 2),
        "scoring.atr_sl_multiplier": round(rng.uniform(1.8, 2.8), 2),
        "filters.min_adx": round(rng.uniform(17.0, 24.0), 1),
        "trailing.activation_pct": round(rng.uniform(.70, 1.25), 2),
        "trailing.callback_pct": round(rng.uniform(.25, .50), 2),
        "weights": weights,
    }


def objective(result: dict) -> float:
    if result["worst_fold"] <= 0 or result["max_dd"] > 45:
        return -1e12
    if result["total_trades"] < 150:
        return -1e12
    # Penalize DD gently after enforcing green training years.
    return result["geo_pct"] - .35 * result["max_dd"]


def _parameter_vector(overrides: dict) -> dict[str, float]:
    """Materialize a candidate, including static values for omitted parameters."""
    cfg = drv.build_config(overrides)
    tier = cfg.trading.active_leverage_tier
    vector = {
        "tier.strong_threshold": tier.strong_threshold,
        "tier.marginal_threshold_low": tier.marginal_threshold_low,
        "tier.tp1_rr": tier.tp1_rr,
        "tier.tp2_rr": tier.tp2_rr,
        "tier.tp1_exit_pct": tier.tp1_exit_pct,
        "scoring.atr_sl_multiplier": cfg.scoring.atr_sl_multiplier,
        "filters.min_adx": cfg.filters.min_adx,
        "trailing.activation_pct": cfg.trading.trailing_stop.activation_pct,
        "trailing.callback_pct": cfg.trading.trailing_stop.callback_pct,
    }
    vector.update({f"weights.{key}": value
                   for key, value in cfg.scoring.weights.items()})
    return vector


def parameter_turnover(left: dict, right: dict) -> float:
    """Mean normalized parameter change between two deployable candidates."""
    left_vector = _parameter_vector(left)
    right_vector = _parameter_vector(right)
    changes = [
        abs(left_vector[key] - right_vector[key]) / width
        for key, width in PARAMETER_RANGES.items()
    ]
    return sum(changes) / len(changes)


def run(trials: int = 300, seed: int = 73, *, entry_mode: str = "maker",
        exit_granularity: str = "sub",
        turnover_penalty: float = 0.0,
        _evaluation_cache: dict | None = None) -> dict:
    drv.setup()
    strat = {"entry_mode": entry_mode}
    rows = []
    previous_winner: dict = {}
    for window_i, (target_name, train_folds, target_fold) in enumerate(TRAIN_WINDOWS):
        rng = random.Random(seed + window_i)
        candidates = [{}] + [sample_candidate(rng) for _ in range(trials)]
        ranked = []
        for overrides in candidates:
            cache_key = (
                tuple(tuple(fold) for fold in train_folds),
                json.dumps(overrides, sort_keys=True),
                entry_mode, exit_granularity,
            )
            trained = (_evaluation_cache.get(cache_key)
                       if _evaluation_cache is not None else None)
            if trained is None:
                trained = drv.evaluate(
                    overrides, folds=train_folds, slip=.0002, funding=True,
                    strat=strat, exit_granularity=exit_granularity,
                )
                if _evaluation_cache is not None:
                    _evaluation_cache[cache_key] = trained
            turnover = parameter_turnover(previous_winner, overrides)
            selection_score = objective(trained) - turnover_penalty * turnover
            ranked.append((selection_score, overrides, trained, turnover))
        ranked.sort(key=lambda row: row[0], reverse=True)
        selection_score, winner, train_result, turnover = ranked[0]
        unseen = drv.evaluate(
            winner, folds=target_fold, slip=.0002, funding=True,
            strat=strat, exit_granularity=exit_granularity,
        )
        static = drv.evaluate(
            {}, folds=target_fold, slip=.0002, funding=True,
            strat=strat, exit_granularity=exit_granularity,
        )
        rows.append({
            "target": target_name, "winner": winner,
            "train": train_result, "unseen": unseen, "static": static,
            "selection_score": selection_score,
            "parameter_turnover": turnover,
        })
        previous_winner = winner

    tuned_factor = math.prod(1 + row["unseen"]["mean_ret"] / 100 for row in rows)
    static_factor = math.prod(1 + row["static"]["mean_ret"] / 100 for row in rows)
    return {
        "method": "annual trailing-two-year tune, next-year trade",
        "trials_per_window": trials,
        "seed": seed, "entry_mode": entry_mode,
        "exit_granularity": exit_granularity,
        "turnover_penalty": turnover_penalty,
        "total_parameter_turnover": sum(row["parameter_turnover"] for row in rows),
        "walk_forward_compound_x": tuned_factor,
        "static_compound_x": static_factor,
        "growth_ratio": tuned_factor / static_factor if static_factor else 0,
        "windows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--entry-mode", choices=("taker", "maker"), default="maker")
    parser.add_argument("--exit-granularity", choices=("primary", "sub"), default="sub")
    parser.add_argument("--turnover-penalty", type=float, default=0.0)
    parser.add_argument("--output", default="reports/walk_forward_retune.json")
    args = parser.parse_args()
    result = run(args.trials, args.seed, entry_mode=args.entry_mode,
                 exit_granularity=args.exit_granularity,
                 turnover_penalty=args.turnover_penalty)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"walk-forward {result['walk_forward_compound_x']:.3f}x vs "
          f"static {result['static_compound_x']:.3f}x "
          f"(ratio {result['growth_ratio']:.3f})")
    for row in result["windows"]:
        print(f"  {row['target']}: tuned {row['unseen']['mean_ret']:+.1f}% vs "
              f"static {row['static']['mean_ret']:+.1f}%")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
