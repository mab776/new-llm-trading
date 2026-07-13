"""Revalidate both profiles after fixing sub-bar trailing cadence."""

from __future__ import annotations

import json
from pathlib import Path

from opt.drawdown import analyze_drawdowns
from opt.driver import FOLDS, TEST_FOLDS, TRAIN_FOLDS
from opt.multi_asset import simulate_multi
from opt.multi_portfolio import evaluate_shared, load_assets
from opt.fastbt import simulate


CHRONO_FOLDS = [
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-01"),
]


def _evaluate(assets, folds) -> dict:
    return evaluate_shared(
        assets, folds=folds, slip=.0002, exit_granularity="sub",
    )


def validate_profile(profile: str) -> dict:
    assets = load_assets("maker", profile)
    continuous = simulate_multi(
        assets, "2021-01-01", "2025-06-01", slip=.0002,
        exit_granularity="sub",
    )
    study = analyze_drawdowns(continuous.equity_curve, thresholds=(10, 20, 25, 30, 33))
    standalone = {}
    for symbol, item in assets.items():
        result = simulate(
            item.pre, item.config, "2021-01-01", "2025-06-01", slip=.0002,
            funding_by_pos=item.funding_by_pos, exit_granularity="sub",
        )
        standalone[symbol] = {
            "compound_x": max(.01, 1 + result.return_pct / 100),
            "max_dd": result.max_dd_pct,
            "trades": result.trades,
        }
    return {
        "train": _evaluate(assets, TRAIN_FOLDS),
        "held_out_test": _evaluate(assets, TEST_FOLDS),
        "chronological": _evaluate(assets, CHRONO_FOLDS),
        "annual": _evaluate(assets, FOLDS),
        "continuous": {
            "compound_x": max(.01, 1 + continuous.return_pct / 100),
            "final_balance": continuous.final_balance,
            "reported_max_dd": continuous.max_dd_pct,
            "trades": continuous.trades,
            "mark_to_market_max_dd": study.max_drawdown_pct,
            "top_episode_depths": [episode.depth_pct for episode in study.episodes[:3]],
            "thresholds": {
                str(int(threshold)): vars(stats)
                for threshold, stats in study.thresholds.items()
            },
        },
        "standalone_continuous": standalone,
    }


def main() -> None:
    result = {
        "method": ("corrected honest 1h sub-bar exits with trailing stop fixed intrabar "
                   "and ratcheted once after each completed 4h bar; maker entry; funding; "
                   "liquidation; 2bps market-exit slippage"),
        "supersedes_round_16_17_subbar_results": True,
        "standard": validate_profile("standard"),
        "aggressive": validate_profile("aggressive"),
    }
    path = Path("opt/cadence_correction_results.json")
    path.write_text(json.dumps(result, indent=2, default=str))
    for profile in ("standard", "aggressive"):
        row = result[profile]
        print(
            f"{profile}: continuous={row['continuous']['compound_x']:,.2f}x "
            f"DD={row['continuous']['reported_max_dd']:.2f}% "
            f"MTM-DD={row['continuous']['mark_to_market_max_dd']:.2f}% "
            f"test={row['held_out_test']['compound_x']:,.2f}x"
        )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
