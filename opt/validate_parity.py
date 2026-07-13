"""Exact fastbt ↔ full-engine parity check for a bounded period."""

from __future__ import annotations

import argparse
import contextlib
import io
import json

from llm_trading_bot.backtesting import BacktestEngine
from llm_trading_bot.config import load_config
from llm_trading_bot.data import fetch_multi_timeframe
from llm_trading_bot.funding import fetch_funding_history
from opt.fastbt import precompute, simulate


def run(config_path: str, start: str, end: str, entry_mode: str) -> dict:
    cfg = load_config(config_path)
    cfg.backtesting.start_date = start
    cfg.backtesting.end_date = end
    cfg.trading.entry_mode = entry_mode
    ds = cfg.data_source
    data = fetch_multi_timeframe(
        ds.exchange_symbol, cfg.trading.timeframes, start_date=start,
        end_date=end, warmup_periods=cfg.backtesting.warmup_periods,
        source=ds.source, market=ds.market,
    )
    funding = (fetch_funding_history(ds.exchange_symbol, start, end)
               if cfg.backtesting.include_funding else None)
    engine = BacktestEngine(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        full = engine.run(data, cfg.trading.primary_timeframe, funding=funding)

    fast_pre = precompute(data, cfg.trading.primary_timeframe,
                          cfg.backtesting.warmup_periods)
    funding_by_pos = None
    if funding is not None:
        from llm_trading_bot.funding import aggregate_funding_to_bars
        hours = {"1h": 1, "4h": 4, "1d": 24}.get(cfg.trading.primary_timeframe, 4)
        funding_by_pos = aggregate_funding_to_bars(
            funding, data[cfg.trading.primary_timeframe].index, hours
        )
    fast = simulate(
        fast_pre, cfg, start, end, model_liquidation=False,
        funding_by_pos=funding_by_pos,
    )
    engine_values = {
        "return_pct": full.stats.total_return_pct,
        "final_balance": full.stats.final_balance,
        "trades": full.stats.total_trades,
        "win_rate": full.stats.win_rate,
        "profit_factor": full.stats.profit_factor,
        "max_dd_pct": full.stats.max_drawdown_pct,
        "sharpe": full.stats.sharpe_ratio,
    }
    fast_values = {
        "return_pct": fast.return_pct, "final_balance": fast.final_balance,
        "trades": fast.trades, "win_rate": fast.win_rate,
        "profit_factor": fast.profit_factor, "max_dd_pct": fast.max_dd_pct,
        "sharpe": fast.sharpe,
    }
    mismatches = {key: (engine_values[key], fast_values[key])
                  for key in engine_values if engine_values[key] != fast_values[key]}
    return {"engine": engine_values, "fastbt": fast_values,
            "exact": not mismatches, "mismatches": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--entry-mode", choices=("taker", "maker"), default="maker")
    args = parser.parse_args()
    result = run(args.config, args.start, args.end, args.entry_mode)
    print(json.dumps(result, indent=2))
    if not result["exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
