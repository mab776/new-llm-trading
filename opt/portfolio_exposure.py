"""TRAIN-only search for shared-portfolio ex-ante exposure controls.

Round 15's anti-martingale candidate is kept enabled.  Selection uses a 22%
TRAIN maxDD buffer so held-out validation has room around the 25% target.
TEST, chronological OOS, annual, and continuous results are evaluated only
after the single TRAIN winner is fixed.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from opt.driver import FOLDS, TEST_FOLDS, TRAIN_FOLDS
from opt.multi_portfolio import _print, evaluate_shared, load_assets


ANTI_MARTINGALE = {
    "anti_martingale_step": 0.05,
    "anti_martingale_min": 0.70,
    "anti_martingale_max": 1.10,
    "global_max_positions": None,
    "portfolio_risk_multiplier": 1.0,
    "global_max_margin_pct": None,
    "global_max_notional_pct": None,
}
GLOBAL_SLOTS = (None, 2, 3, 4, 5)
RISK_MULTIPLIERS = (0.75, 0.85, 0.95, 1.00, 1.05, 1.15)
MARGIN_CAPS = (0.05, 0.055, 0.06, 0.065, 0.07, 0.075,
               0.08, 0.085, 0.09, 0.10, 0.12)
TRAIN_DD_BUFFER = 22.0
ACCEPTANCE_DD = 25.1  # approximately 25%; permits normal reporting-rounding noise
LEVERAGE = 25
SHIPPED_STRATEGY = ANTI_MARTINGALE | {
    "global_max_margin_pct": 0.044,
    "global_max_notional_pct": 1.10,
}


def candidate_grid() -> list[dict]:
    rows = []
    for slots, risk, margin in itertools.product(
        GLOBAL_SLOTS, RISK_MULTIPLIERS, MARGIN_CAPS
    ):
        rows.append(ANTI_MARTINGALE | {
            "global_max_positions": slots,
            "portfolio_risk_multiplier": risk,
            "global_max_margin_pct": margin,
            # All searched assets currently use 25x, so this is equivalent to
            # the margin cap today and remains an explicit leverage-safe guard.
            "global_max_notional_pct": margin * LEVERAGE,
        })
    return rows


def select_on_train(rows: list[dict]) -> dict:
    feasible = [r for r in rows if r["train"]["max_dd"] <= TRAIN_DD_BUFFER]
    if not feasible:
        raise RuntimeError("No candidate met the predeclared TRAIN DD buffer")
    return max(feasible, key=lambda r: (
        r["train"]["geo_pct"], r["train"]["worst_fold"],
        -r["train"]["max_dd"],
    ))


def validation_sets(assets, strategy, common) -> dict:
    return {
        "test": evaluate_shared(assets, folds=TEST_FOLDS, strat=strategy, **common),
        "chronological_oos": evaluate_shared(
            assets, folds=(("24-25H1", "2024-01-01", "2025-06-01"),),
            strat=strategy, **common,
        ),
        "annual": evaluate_shared(assets, folds=FOLDS, strat=strategy, **common),
        "full_continuous": evaluate_shared(
            assets, folds=(("full", "2021-01-01", "2025-06-01"),),
            strat=strategy, **common,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-granularity", choices=("primary", "sub"), default="sub")
    parser.add_argument("--output", default="opt/portfolio_exposure_results.json")
    args = parser.parse_args()

    assets = load_assets("maker")
    common = {"exit_granularity": args.exit_granularity}
    baseline_train = evaluate_shared(
        assets, folds=TRAIN_FOLDS, strat=ANTI_MARTINGALE, **common
    )
    rows = []
    grid = candidate_grid()
    for number, strategy in enumerate(grid, 1):
        train = evaluate_shared(
            assets, folds=TRAIN_FOLDS, strat=strategy, **common
        )
        rows.append({"strat": strategy, "train": train})
        if number % 30 == 0:
            print(f"searched {number}/{len(grid)}", flush=True)

    selected = select_on_train(rows)
    initial_validation = validation_sets(assets, selected["strat"], common)
    selected_accepted = (
        selected["train"]["max_dd"] <= ACCEPTANCE_DD
        and all(result["max_dd"] <= ACCEPTANCE_DD
                for result in initial_validation.values())
    )
    recommended_strategy = selected["strat"] if selected_accepted else SHIPPED_STRATEGY
    final_train = evaluate_shared(
        assets, folds=TRAIN_FOLDS, strat=recommended_strategy, **common
    )
    validation = validation_sets(assets, recommended_strategy, common)
    baseline_validation = validation_sets(assets, ANTI_MARTINGALE, common)
    accepted = (
        final_train["max_dd"] <= ACCEPTANCE_DD
        and all(result["max_dd"] <= ACCEPTANCE_DD for result in validation.values())
    )

    print("\nTRAIN selection:")
    _print("uncapped anti", baseline_train)
    print(f"\nInitial return-max winner with {TRAIN_DD_BUFFER:.0f}% TRAIN buffer: "
          f"{selected['strat']}")
    _print("selected", selected["train"])
    print(f"\nRecommended policy: {recommended_strategy}")
    print(f"TRAIN winner passed held-out DD acceptance: {selected_accepted}")
    _print("final TRAIN", final_train)
    for label in ("test", "chronological_oos", "annual", "full_continuous"):
        print(f"\n{label}:")
        _print("uncapped anti", baseline_validation[label])
        _print("initial", initial_validation[label])
        _print("final", validation[label])
    print(f"\nACCEPTED AT APPROXIMATELY 25% ON EVERY SPLIT: {accepted} "
          f"(tolerance {ACCEPTANCE_DD:.1f}%)")

    payload = {
        "method": "maker, 2bps market slip, funding, liquidation, honest 1h sub exits",
        "train_dd_buffer": TRAIN_DD_BUFFER,
        "acceptance_dd": ACCEPTANCE_DD,
        "baseline_strategy": ANTI_MARTINGALE,
        "baseline_train": baseline_train,
        "selected": selected,
        "initial_validation": initial_validation,
        "selected_accepted": selected_accepted,
        "shipped_strategy_before_study": SHIPPED_STRATEGY,
        "final_strategy": recommended_strategy,
        "final_train": final_train,
        "accepted": accepted,
        "baseline_validation": baseline_validation,
        "validation": validation,
        "train_candidates": rows,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
