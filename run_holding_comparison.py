"""
Comparison script: aggressive vs conservative × max_holding_hours 168 vs 0 (infinite)
Runs 4 backtests and prints a summary table.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm_trading_bot.backtesting import BacktestEngine
from llm_trading_bot.config import load_config
from llm_trading_bot.data import configure_cache, fetch_multi_timeframe


def run_combo(base_config, tier: str, max_holding_hours: int, data_by_tf: dict) -> dict:
    """Run a single backtest combo and return a compact stats dict."""
    config = base_config.model_copy(deep=True)
    config.trading.active_tier = tier
    config.risk_management.max_holding_hours = max_holding_hours

    label = f"{tier:>12s} | max_hold={'7d (168h)' if max_holding_hours else 'infinite (0)'}"
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")

    engine = BacktestEngine(config)
    result = engine.run(data_by_tf, config.trading.primary_timeframe)
    stats = result.stats

    if stats is None:
        print("  No stats returned (no trades?)")
        return {
            "tier": tier,
            "max_holding_hours": max_holding_hours,
            "trades": 0,
            "win_rate": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "final_balance": base_config.backtesting.initial_balance,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
        }

    print(f"  Trades: {stats.total_trades}  |  Win rate: {stats.win_rate:.1f}%  |  "
          f"Return: {stats.total_return_pct:+.1f}%  |  MaxDD: {stats.max_drawdown_pct:.1f}%  |  "
          f"PF: {stats.profit_factor:.2f}")

    return {
        "tier": tier,
        "max_holding_hours": max_holding_hours,
        "trades": stats.total_trades,
        "win_rate": stats.win_rate,
        "total_return_pct": stats.total_return_pct,
        "max_drawdown_pct": stats.max_drawdown_pct,
        "profit_factor": stats.profit_factor,
        "sharpe_ratio": stats.sharpe_ratio,
        "final_balance": stats.final_balance,
        "avg_win_pct": getattr(stats, "avg_win_pct", 0.0),
        "avg_loss_pct": getattr(stats, "avg_loss_pct", 0.0),
    }


def print_summary(results: list[dict], initial_balance: float) -> None:
    sep = "+" + "-"*18 + "+" + "-"*12 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*10 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*14 + "+"
    header = (
        f"| {'Tier':<16} | {'Max Hold':<10} | {'Trades':>6} | {'Win%':>6} | {'Return%':>6} | "
        f"{'MaxDD%':>8} | {'P/F':>6} | {'Sharpe':>6} | {'Final $':>12} |"
    )

    print("\n" + "="*110)
    print("BACKTEST COMPARISON SUMMARY")
    print(f"Initial balance: ${initial_balance:,.2f}")
    print("="*110)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        hold_label = "7d (168h)" if r["max_holding_hours"] else "infinite"
        pf = r["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "  inf"
        print(
            f"| {r['tier']:<16} | {hold_label:<10} | {r['trades']:>6} | {r['win_rate']:>5.1f}% | "
            f"{r['total_return_pct']:>+6.1f}% | {r['max_drawdown_pct']:>7.1f}% | "
            f"{pf_str:>6} | {r['sharpe_ratio']:>6.2f} | ${r['final_balance']:>11,.2f} |"
        )

    print(sep)
    print()

    # Delta analysis
    print("DELTA ANALYSIS (7d vs infinite max hold)")
    print("-"*60)
    for tier in ["conservative", "aggressive"]:
        r7  = next(r for r in results if r["tier"] == tier and r["max_holding_hours"] != 0)
        ri  = next(r for r in results if r["tier"] == tier and r["max_holding_hours"] == 0)
        dr  = r7["total_return_pct"] - ri["total_return_pct"]
        ddd = r7["max_drawdown_pct"]  - ri["max_drawdown_pct"]
        print(f"  [{tier:>12}]  ΔReturn: {dr:+.1f}%   ΔMaxDD: {ddd:+.1f}%   "
              f"ΔTrades: {r7['trades']-ri['trades']:+d}")
    print()


def main() -> None:
    config = load_config("config.json")
    configure_cache(config.data_cache.ttl_seconds)

    ds = config.data_source
    symbol = ds.exchange_symbol if ds.source != "yfinance" else config.trading.yfinance_symbol

    print(f"Fetching shared data for {symbol} (source: {ds.source})")
    print(f"Period: {config.backtesting.start_date} → {config.backtesting.end_date}")
    print(f"Warmup: {config.backtesting.warmup_periods} periods")

    data_by_tf = fetch_multi_timeframe(
        symbol=symbol,
        timeframes=config.trading.timeframes,
        start_date=config.backtesting.start_date,
        end_date=config.backtesting.end_date,
        warmup_periods=config.backtesting.warmup_periods,
        source=ds.source,
    )
    for tf, df in data_by_tf.items():
        print(f"  {tf}: {len(df)} candles (incl. warmup)")

    combos = [
        ("conservative", 168),
        ("conservative", 0),
        ("aggressive",   168),
        ("aggressive",   0),
    ]

    results = []
    for tier, hours in combos:
        r = run_combo(config, tier, hours, data_by_tf)
        results.append(r)

    print_summary(results, config.backtesting.initial_balance)


if __name__ == "__main__":
    main()
