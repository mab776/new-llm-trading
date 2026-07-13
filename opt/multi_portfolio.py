"""Evaluate BTC+ETH(+SOL) on one shared compounding portfolio."""

from __future__ import annotations

import math

from opt.driver import FOLDS, load_context
from opt.multi_asset import AssetInput, simulate_multi


ASSETS = {
    "BTC": ("BTC/USDT:USDT", "config.json"),
    "ETH": ("ETH/USDT:USDT", "config-eth.json"),
    "SOL": ("SOL/USDT:USDT", "config-sol.json"),
}
AGGRESSIVE_ASSETS = {
    "BTC": ("BTC/USDT:USDT", "config-aggressive.json"),
    "ETH": ("ETH/USDT:USDT", "config-eth-aggressive.json"),
    "SOL": ("SOL/USDT:USDT", "config-sol-aggressive.json"),
}


def load_assets(entry_mode: str = "maker", profile: str = "standard") -> dict[str, AssetInput]:
    configs = AGGRESSIVE_ASSETS if profile == "aggressive" else ASSETS
    out = {}
    for label, (symbol, config_path) in configs.items():
        ctx = load_context(symbol, config_path)
        ctx.config.trading.entry_mode = entry_mode
        out[label] = AssetInput(ctx.pre, ctx.config, ctx.funding)
    return out


def evaluate_shared(assets: dict[str, AssetInput], *, slip: float = .0002,
                    exit_granularity: str = "primary", folds=FOLDS,
                    strat: dict | None = None) -> dict:
    per, factors, max_dd, total_trades = {}, [], 0.0, 0
    for name, start, end in folds:
        result = simulate_multi(
            assets, start, end, slip=slip,
            exit_granularity=exit_granularity,
            strat=strat,
        )
        factor = max(.01, 1 + result.return_pct / 100)
        factors.append(factor)
        max_dd = max(max_dd, result.max_dd_pct)
        total_trades += result.trades
        per[name] = {
            "return_pct": result.return_pct, "max_dd": result.max_dd_pct,
            "trades": result.trades, "symbols": result.per_symbol,
            "maker_orders": result.maker_orders,
            "maker_touches": result.maker_touches,
            "maker_queue_eligible": result.maker_queue_eligible,
            "maker_fills": result.maker_fills,
        }
    compound = math.prod(factors)
    geo = math.exp(sum(math.log(x) for x in factors) / len(factors))
    return {
        "geo_pct": (geo - 1) * 100, "compound_x": compound,
        "worst_fold": min(x["return_pct"] for x in per.values()),
        "max_dd": max_dd, "trades": total_trades,
        "maker_orders": sum(x["maker_orders"] for x in per.values()),
        "maker_touches": sum(x["maker_touches"] for x in per.values()),
        "maker_queue_eligible": sum(x["maker_queue_eligible"] for x in per.values()),
        "maker_fills": sum(x["maker_fills"] for x in per.values()),
        "per": per,
    }


def _print(label: str, result: dict) -> None:
    print(f"{label:18s} geo {result['geo_pct']:+.1f}%/fold  "
          f"compound {result['compound_x']:.2f}x  "
          f"worst {result['worst_fold']:+.1f}%  "
          f"maxDD {result['max_dd']:.1f}%  trades {result['trades']}")
    print("  " + "  ".join(
        f"{year}:{row['return_pct']:+.0f}%(dd{row['max_dd']:.0f})"
        for year, row in result["per"].items()
    ))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-granularity", choices=("primary", "sub"), default="primary")
    parser.add_argument("--entry-mode", choices=("both", "taker", "maker"), default="both")
    parser.add_argument("--profile", choices=("standard", "aggressive"), default="standard")
    args = parser.parse_args()
    modes = ("taker", "maker") if args.entry_mode == "both" else (args.entry_mode,)
    for mode in modes:
        assets = load_assets(mode, args.profile)
        _print(f"shared {mode}", evaluate_shared(
            assets, exit_granularity=args.exit_granularity
        ))


if __name__ == "__main__":
    main()
