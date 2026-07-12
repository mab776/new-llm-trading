"""
LLM Trading Bot — main entry point.

Usage:
    python -m llm_trading_bot.main --config config.json
    python -m llm_trading_bot.main --config config.json --mode backtest
    python -m llm_trading_bot.main --config config.json --mode analyze
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_trading_bot.config import load_config


def run_live(config_path: str) -> None:
    """Run in live/scheduled trading mode."""
    from llm_trading_bot.scheduler import TradingScheduler

    config = load_config(config_path)
    scheduler = TradingScheduler(config)
    scheduler.start()


def run_analyze(config_path: str) -> None:
    """Run a single analysis cycle (no trading)."""
    from llm_trading_bot.data import configure_cache, fetch_multi_timeframe
    from llm_trading_bot.routing import route_signal
    from llm_trading_bot.scoring import calculate_indicators, format_scoring_report

    config = load_config(config_path)
    configure_cache(config.data_cache.ttl_seconds)

    ds = config.data_source
    symbol = ds.exchange_symbol if ds.source != "yfinance" else config.trading.yfinance_symbol
    print(f"Fetching data for {symbol} (source: {ds.source})...")
    data_by_tf = fetch_multi_timeframe(
        symbol=symbol,
        timeframes=config.trading.timeframes,
        source=ds.source,
        market=ds.market,
    )

    indicators_by_tf = {}
    for tf, df in data_by_tf.items():
        indicators_by_tf[tf] = calculate_indicators(df, tf)
        print(f"  {tf}: {len(df)} candles loaded")

    decision = route_signal(indicators_by_tf, config)
    report = format_scoring_report(decision.scoring_result, decision.targets)
    print(report)
    print()
    print(f"Signal: {decision.signal_strength.value}")
    if decision.template_response:
        print(decision.template_response)
    elif decision.skip_reason:
        print(f"Skip: {decision.skip_reason}")
    elif decision.needs_llm:
        print("→ Marginal signal — would send to LLM for consensus")


def run_backtest(config_path: str) -> None:
    """Run backtesting engine."""
    from llm_trading_bot.backtesting import BacktestEngine
    from llm_trading_bot.data import configure_cache, fetch_multi_timeframe
    from llm_trading_bot.reporting import (
        export_decision_log,
        export_trades_csv,
        format_stats_report,
        generate_backtest_charts,
    )

    config = load_config(config_path)
    configure_cache(config.data_cache.ttl_seconds)

    ds = config.data_source
    symbol = ds.exchange_symbol if ds.source != "yfinance" else config.trading.yfinance_symbol
    print(f"Fetching data for backtest: {symbol} (source: {ds.source})")
    print(f"Period: {config.backtesting.start_date} to {config.backtesting.end_date}")
    print(f"Warmup: {config.backtesting.warmup_periods} periods")
    print()

    data_by_tf = fetch_multi_timeframe(
        symbol=symbol,
        timeframes=config.trading.timeframes,
        start_date=config.backtesting.start_date,
        end_date=config.backtesting.end_date,
        warmup_periods=config.backtesting.warmup_periods,
        source=ds.source,
        market=ds.market,
    )

    for tf, df in data_by_tf.items():
        print(f"  {tf}: {len(df)} candles (incl. warmup)")

    engine = BacktestEngine(config)
    result = engine.run(data_by_tf, config.trading.primary_timeframe)

    if result.stats:
        report = format_stats_report(result.stats, result.config_summary)
        print("\n" + report)

    # Generate outputs
    print("\nGenerating reports...")
    charts = generate_backtest_charts(result)
    for c in charts:
        print(f"  Chart: {c}")

    if result.decision_log:
        log_path = export_decision_log(result.decision_log)
        print(f"  Decision log: {log_path}")

    if result.portfolio:
        csv_path = export_trades_csv(result.portfolio)
        print(f"  Trades CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Hybrid Trading Bot")
    parser.add_argument(
        "--config", "-c", default="config.json",
        help="Path to config file (default: config.json)"
    )
    parser.add_argument(
        "--mode", "-m", choices=["live", "analyze", "backtest"],
        default="analyze",
        help="Run mode: live (scheduled trading), analyze (single analysis), backtest"
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    print(f"LLM Trading Bot — Mode: {args.mode}")
    print(f"Config: {args.config}")
    print()

    if args.mode == "live":
        run_live(args.config)
    elif args.mode == "analyze":
        run_analyze(args.config)
    elif args.mode == "backtest":
        run_backtest(args.config)


if __name__ == "__main__":
    main()
