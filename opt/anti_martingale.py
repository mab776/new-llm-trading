"""TRAIN-selected anti-martingale sizing screen for the shared portfolio.

The search uses only interleaved TRAIN half-years.  Held-out TEST,
chronological 2024-2025H1, annual folds, and the continuous full period are
reported only after one candidate has been selected.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from opt.driver import FOLDS, TEST_FOLDS, TRAIN_FOLDS
from opt.multi_portfolio import _print, evaluate_shared, load_assets


STEPS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
MINIMUMS = (0.25, 0.40, 0.55, 0.70)
MAXIMUMS = (1.00, 1.10, 1.25, 1.50)
DD_LIMIT = 25.0


def strategy(step: float, minimum: float, maximum: float) -> dict:
    return {
        "anti_martingale_step": step,
        "anti_martingale_min": minimum,
        "anti_martingale_max": maximum,
    }


def candidate_grid() -> list[dict]:
    return [strategy(*values) for values in itertools.product(
        STEPS, MINIMUMS, MAXIMUMS
    )]


def select_on_train(rows: list[dict]) -> tuple[dict, bool]:
    """Maximize TRAIN return subject to DD; otherwise select minimum TRAIN DD."""
    feasible = [row for row in rows if row["train"]["max_dd"] <= DD_LIMIT]
    if feasible:
        return max(feasible, key=lambda row: (
            row["train"]["geo_pct"], row["train"]["worst_fold"]
        )), True
    return min(rows, key=lambda row: (
        row["train"]["max_dd"], -row["train"]["geo_pct"]
    )), False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-granularity", choices=("primary", "sub"), default="sub")
    parser.add_argument("--output", default="opt/anti_martingale_results.json")
    args = parser.parse_args()

    assets = load_assets("maker")
    common = {"exit_granularity": args.exit_granularity}
    baseline_train = evaluate_shared(assets, folds=TRAIN_FOLDS, **common)
    rows = []
    grid = candidate_grid()
    for number, strat in enumerate(grid, 1):
        train = evaluate_shared(assets, folds=TRAIN_FOLDS, strat=strat, **common)
        rows.append({"strat": strat, "train": train})
        if number % 12 == 0:
            print(f"searched {number}/{len(grid)}", flush=True)

    selected, feasible = select_on_train(rows)
    chosen = selected["strat"]
    validation = {
        "test": evaluate_shared(assets, folds=TEST_FOLDS, strat=chosen, **common),
        "chronological_oos": evaluate_shared(
            assets, folds=(("24-25H1", "2024-01-01", "2025-06-01"),),
            strat=chosen, **common,
        ),
        "annual": evaluate_shared(assets, folds=FOLDS, strat=chosen, **common),
        "full_continuous": evaluate_shared(
            assets, folds=(("full", "2021-01-01", "2025-06-01"),),
            strat=chosen, **common,
        ),
    }
    baseline_validation = {
        "test": evaluate_shared(assets, folds=TEST_FOLDS, **common),
        "chronological_oos": evaluate_shared(
            assets, folds=(("24-25H1", "2024-01-01", "2025-06-01"),), **common,
        ),
        "annual": evaluate_shared(assets, folds=FOLDS, **common),
        "full_continuous": evaluate_shared(
            assets, folds=(("full", "2021-01-01", "2025-06-01"),), **common,
        ),
    }

    print("\nTRAIN selection (baseline):")
    _print("baseline", baseline_train)
    print(f"\nSelected {'feasible' if feasible else 'minimum-DD fallback'}: {chosen}")
    _print("selected TRAIN", selected["train"])
    for label in ("test", "chronological_oos", "annual", "full_continuous"):
        print(f"\n{label}:")
        _print("baseline", baseline_validation[label])
        _print("selected", validation[label])

    payload = {
        "method": "maker, 2bps market slip, funding, liquidation, honest 1h sub exits",
        "dd_limit": DD_LIMIT,
        "baseline_train": baseline_train,
        "selected_feasible": feasible,
        "selected": selected,
        "baseline_validation": baseline_validation,
        "validation": validation,
        "train_candidates": rows,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
