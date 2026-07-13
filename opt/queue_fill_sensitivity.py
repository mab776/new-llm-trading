"""Stress the aggressive portfolio's touched-maker fill assumption.

This is a robustness study, not a parameter optimizer. The canonical strategy
remains penetration=0/fill_probability=1; progressively harsher scenarios show
how much historical compounding survives queue priority and non-fill risk.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from opt.driver import FOLDS, TEST_FOLDS, TRAIN_FOLDS
from opt.multi_asset import simulate_multi
from opt.multi_portfolio import evaluate_shared, load_assets


CHRONO_FOLDS = [
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-01"),
]
CONTINUOUS = ("2021-01-01", "2025-06-01")


def _summary(assets, folds, strat) -> dict:
    return evaluate_shared(
        assets, folds=folds, slip=.0002,
        exit_granularity="sub", strat=strat,
    )


def validate(assets, penetration_bps: float, fill_probability: float,
             seed: int) -> dict:
    strat = {
        "maker_queue_penetration_bps": penetration_bps,
        "maker_fill_probability": fill_probability,
        "maker_fill_seed": seed,
    }
    annual = _summary(assets, FOLDS, strat)
    train = _summary(assets, TRAIN_FOLDS, strat)
    test = _summary(assets, TEST_FOLDS, strat)
    chrono = _summary(assets, CHRONO_FOLDS, strat)
    continuous_result = simulate_multi(
        assets, *CONTINUOUS, slip=.0002, exit_granularity="sub", strat=strat,
    )
    continuous_x = max(.01, 1 + continuous_result.return_pct / 100)
    return {
        "penetration_bps": penetration_bps,
        "fill_probability": fill_probability,
        "seed": seed,
        "annual": annual,
        "train": train,
        "held_out_test": test,
        "chronological": chrono,
        "continuous": {
            "compound_x": continuous_x,
            "reported_max_dd": continuous_result.max_dd_pct,
            "mark_to_market_max_dd": max(
                (point.drawdown_pct for point in continuous_result.equity_curve),
                default=0.0,
            ),
            "trades": continuous_result.trades,
            "maker_orders": continuous_result.maker_orders,
            "maker_touches": continuous_result.maker_touches,
            "maker_queue_eligible": continuous_result.maker_queue_eligible,
            "maker_fills": continuous_result.maker_fills,
            "touch_rate": (continuous_result.maker_touches /
                           continuous_result.maker_orders
                           if continuous_result.maker_orders else 0),
            "queue_eligibility_rate": (continuous_result.maker_queue_eligible /
                                       continuous_result.maker_orders
                                       if continuous_result.maker_orders else 0),
            "realized_fill_rate": (continuous_result.maker_fills /
                                   continuous_result.maker_orders
                                   if continuous_result.maker_orders else 0),
        },
    }


def scenario_grid(seeds: list[int]) -> list[tuple[float, float, int]]:
    scenarios = [(0.0, 1.0, seeds[0])]
    scenarios.extend((0.0, probability, seed)
                     for probability in (.95, .90, .80, .70)
                     for seed in seeds)
    scenarios.extend((penetration, 1.0, seeds[0])
                     for penetration in (.5, 1.0, 2.0, 5.0, 10.0))
    scenarios.extend((penetration, probability, seed)
                     for penetration, probability in ((1.0, .90), (2.0, .80), (5.0, .70))
                     for seed in seeds)
    return scenarios


def aggregate(rows: list[dict], baseline_x: float) -> list[dict]:
    grouped: dict[tuple[float, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (row["penetration_bps"], row["fill_probability"]), []
        ).append(row)
    output = []
    baseline_log = math.log(baseline_x)
    for (penetration, probability), samples in sorted(grouped.items()):
        continuous = [row["continuous"] for row in samples]
        multiples = [row["compound_x"] for row in continuous]
        output.append({
            "penetration_bps": penetration,
            "fill_probability": probability,
            "seeds": [row["seed"] for row in samples],
            "continuous_x_median": statistics.median(multiples),
            "continuous_x_min": min(multiples),
            "continuous_x_max": max(multiples),
            "log_growth_retention_median": (
                statistics.median(math.log(max(.01, value)) for value in multiples)
                / baseline_log if baseline_log else 0
            ),
            "reported_max_dd_worst": max(
                row["reported_max_dd"] for row in continuous
            ),
            "mark_to_market_max_dd_worst": max(
                row["mark_to_market_max_dd"] for row in continuous
            ),
            "realized_fill_rate_median": statistics.median(
                row["realized_fill_rate"] for row in continuous
            ),
            "held_out_test_x_median": statistics.median(
                row["held_out_test"]["compound_x"] for row in samples
            ),
            "held_out_test_worst_fold": min(
                row["held_out_test"]["worst_fold"] for row in samples
            ),
            "chronological_x_median": statistics.median(
                row["chronological"]["compound_x"] for row in samples
            ),
            "annual_worst_fold": min(
                row["annual"]["worst_fold"] for row in samples
            ),
        })
    return output


def run(seeds: list[int]) -> dict:
    assets = load_assets("maker", "aggressive")
    rows = [validate(assets, penetration, probability, seed)
            for penetration, probability, seed in scenario_grid(seeds)]
    baseline = next(row for row in rows
                    if row["penetration_bps"] == 0
                    and row["fill_probability"] == 1)
    baseline_x = baseline["continuous"]["compound_x"]
    return {
        "method": ("aggressive BTC+ETH+SOL shared portfolio; completed-candle "
                   "alignment; maker entry; honest 1h sub-bar exits; funding; "
                   "liquidation; 2bps market-exit slippage"),
        "canonical_baseline": baseline,
        "aggregates": aggregate(rows, baseline_x),
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="17,73,211,419,887")
    parser.add_argument("--output", default="opt/queue_fill_sensitivity_results.json")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("at least one seed is required")
    result = run(seeds)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    for row in result["aggregates"]:
        print(
            f"penetration={row['penetration_bps']:>4.1f}bps "
            f"prob={row['fill_probability']:.2f} "
            f"continuous={row['continuous_x_median']:,.2f}x "
            f"log-retained={row['log_growth_retention_median']:.1%} "
            f"fill={row['realized_fill_rate_median']:.1%} "
            f"test={row['held_out_test_x_median']:.2f}x "
            f"worst={row['annual_worst_fold']:+.1f}%"
        )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
