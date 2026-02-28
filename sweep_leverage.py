"""
Leverage sweep — runs backtests across multiple leverage levels to find the sweet spot.

Usage:
    python sweep_leverage.py
    python sweep_leverage.py --leverages 3,5,7,10,15
    python sweep_leverage.py --start 2024-06-01 --end 2025-06-01
"""

import argparse
import copy
import json
import sys
from pathlib import Path

from llm_trading_bot.config import load_config
from llm_trading_bot.backtesting import BacktestEngine
from llm_trading_bot.data import configure_cache, fetch_multi_timeframe


def run_sweep(config_path: str, leverages: list[int], quiet: bool = True) -> list[dict]:
    """Run backtests for each leverage level and collect results."""
    base_config = load_config(config_path)
    configure_cache(base_config.data_cache.ttl_seconds)

    ds = base_config.data_source
    symbol = ds.exchange_symbol if ds.source != "yfinance" else base_config.trading.yfinance_symbol

    print(f"Fetching data for {symbol} (source: {ds.source})...")
    print(f"Period: {base_config.backtesting.start_date} to {base_config.backtesting.end_date}")
    print()

    data_by_tf = fetch_multi_timeframe(
        symbol=symbol,
        timeframes=base_config.trading.timeframes,
        start_date=base_config.backtesting.start_date,
        end_date=base_config.backtesting.end_date,
        warmup_periods=base_config.backtesting.warmup_periods,
        source=ds.source,
    )

    for tf, df in data_by_tf.items():
        print(f"  {tf}: {len(df)} candles")
    print()

    results = []

    for lev in leverages:
        print(f"{'='*60}")
        print(f"  Testing {lev}x leverage")
        print(f"{'='*60}")

        # Deep copy config and override leverage
        raw_path = Path(config_path)
        with open(raw_path) as f:
            raw = json.load(f)

        # Override leverage in the active tier
        active_tier = raw["trading"]["active_tier"]
        raw["trading"]["leverage_tiers"][active_tier]["leverage"] = lev
        from llm_trading_bot.config import AppConfig
        config = AppConfig(**raw)

        engine = BacktestEngine(config)

        # Suppress individual trade prints if quiet
        if quiet:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = engine.run(data_by_tf, config.trading.primary_timeframe)
        else:
            result = engine.run(data_by_tf, config.trading.primary_timeframe)

        stats = result.stats
        if stats:
            row = {
                "leverage": lev,
                "trades": stats.total_trades,
                "win_rate": round(stats.win_rate, 1),
                "return_pct": round(stats.total_return_pct, 1),
                "net_pnl": round(stats.total_net_pnl, 2),
                "max_dd_pct": round(stats.max_drawdown_pct, 1),
                "profit_factor": round(stats.profit_factor, 2),
                "sharpe": round(stats.sharpe_ratio, 2) if stats.sharpe_ratio else 0,
                "final_balance": round(stats.final_balance, 2),
                "fees": round(stats.total_fees, 2),
                "avg_win": round(stats.avg_win, 2) if stats.avg_win else 0,
                "avg_loss": round(stats.avg_loss, 2) if stats.avg_loss else 0,
            }
            results.append(row)
            print(f"  -> {lev}x: {stats.total_return_pct:+.1f}% return | "
                  f"{stats.win_rate:.0f}% WR | {stats.max_drawdown_pct:.1f}% DD | "
                  f"PF={stats.profit_factor:.2f}")
        print()

    return results


def print_comparison(results: list[dict]) -> None:
    """Print a nice comparison table."""
    if not results:
        print("No results to compare.")
        return

    print()
    print("=" * 82)
    print("                         LEVERAGE COMPARISON RESULTS")
    print("=" * 82)
    print(f"{'Lev':>5} | {'Trades':>6} | {'WinRate':>7} | {'Return':>8} | {'MaxDD':>7} | {'PF':>6} | {'Sharpe':>6} | {'Balance':>14}")
    print("-" * 82)

    best_return = max(r["return_pct"] for r in results)
    best_sharpe = max(r["sharpe"] for r in results)

    for r in results:
        ret_marker = " *" if r["return_pct"] == best_return else "  "
        sharpe_marker = " *" if r["sharpe"] == best_sharpe else "  "
        print(
            f"{r['leverage']:>4}x | {r['trades']:>6} | {r['win_rate']:>5.1f}% |"
            f" {r['return_pct']:>+6.1f}%{ret_marker}| {r['max_dd_pct']:>6.1f}% |"
            f" {r['profit_factor']:>5.2f} | {r['sharpe']:>5.2f}{sharpe_marker}|"
            f"  ${r['final_balance']:>12.2f}"
        )

    print("=" * 82)

    # Risk-adjusted recommendation
    print("\n--- Analysis ---")
    for r in results:
        if r["max_dd_pct"] > 0:
            r["return_per_dd"] = r["return_pct"] / r["max_dd_pct"]
            r["calmar"] = r["return_pct"] / r["max_dd_pct"]
        else:
            r["return_per_dd"] = 0
            r["calmar"] = 0

    best_risk_adj = max(results, key=lambda x: x["return_per_dd"])
    best_absolute = max(results, key=lambda x: x["return_pct"])

    print(f"  Best absolute return: {best_absolute['leverage']}x ({best_absolute['return_pct']:+.1f}%)")
    print(f"  Best risk-adjusted (return/DD): {best_risk_adj['leverage']}x "
          f"({best_risk_adj['return_pct']:+.1f}% return / {best_risk_adj['max_dd_pct']:.1f}% DD = "
          f"{best_risk_adj['return_per_dd']:.2f} ratio)")

    # Recommend based on DD tolerance
    safe = [r for r in results if r["max_dd_pct"] <= 15]
    if safe:
        best_safe = max(safe, key=lambda x: x["return_pct"])
        print(f"  Best under 15% DD: {best_safe['leverage']}x ({best_safe['return_pct']:+.1f}% / {best_safe['max_dd_pct']:.1f}% DD)")


def main():
    parser = argparse.ArgumentParser(description="Leverage Sweep Backtest")
    parser.add_argument("--config", "-c", default="config.json")
    parser.add_argument("--leverages", "-l", default="3,5,7,10,15",
                        help="Comma-separated leverage values to test")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show individual trade logs")
    args = parser.parse_args()

    leverages = [int(x.strip()) for x in args.leverages.split(",")]
    print(f"Sweeping leverage: {leverages}")
    print()

    results = run_sweep(args.config, leverages, quiet=not args.verbose)
    print_comparison(results)


if __name__ == "__main__":
    main()
