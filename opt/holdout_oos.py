"""Clean out-of-sample holdout: replay the FROZEN configs on 2025-06 -> 2026-06,
data the strategy was never tuned on. Loads the full history so indicators are warm
at the holdout start, then runs the same live-execution model as the headline
(maker entry, 1h sub-bar exits, funding, isolated liquidation, 2bps slippage).

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.holdout_oos
"""
from __future__ import annotations

import sys

import pandas as pd

import opt.fastbt as fb
from llm_trading_bot.config import load_config
from llm_trading_bot.data import configure_cache, fetch_multi_timeframe
from llm_trading_bot.funding import fetch_funding_history, aggregate_funding_to_bars
from llm_trading_bot.timeframes import timeframe_hours
from opt.multi_asset import AssetInput, simulate_multi
from opt.drawdown import analyze_drawdowns

HOLD_START = "2025-06-01"
HOLD_END = "2026-04-30"  # 1h Bitget cache has a hole in May-2026; stop before it
LOAD_START = "2020-10-01"

PROFILES = {
    "standard":   {"BTC": "config.json",  "ETH": "config-eth.json",  "SOL": "config-sol.json"},
    "aggressive": {"BTC": "config-aggressive.json", "ETH": "config-eth-aggressive.json",
                   "SOL": "config-sol-aggressive.json"},
}
SYMBOLS = {"BTC": "BTC/USDT:USDT", "ETH": "ETH/USDT:USDT", "SOL": "SOL/USDT:USDT"}


def _load(symbol: str, config_path: str) -> AssetInput:
    cfg = load_config(config_path)
    configure_cache(cfg.data_cache.ttl_seconds)
    ds = cfg.data_source
    ds.exchange_symbol = symbol
    cfg.trading.entry_mode = "maker"
    data = fetch_multi_timeframe(
        symbol, cfg.trading.timeframes,
        start_date=LOAD_START, end_date=HOLD_END,
        warmup_periods=0, source=ds.source, market=ds.market,
    )
    print(f"  {symbol} 4h: {len(data['4h'])} rows "
          f"{data['4h'].index[0].date()} -> {data['4h'].index[-1].date()}", file=sys.stderr)
    pre = fb.precompute(data, cfg.trading.primary_timeframe, 200)
    fund = fetch_funding_history(symbol, start_date="2020-08-01", end_date=HOLD_END)
    funding = aggregate_funding_to_bars(
        fund, pd.DatetimeIndex(pre.timestamps),
        timeframe_hours(cfg.trading.primary_timeframe),
    )
    return AssetInput(pre, cfg, funding)


def run_profile(profile: str) -> dict:
    assets = {label: _load(SYMBOLS[label], cfgpath)
              for label, cfgpath in PROFILES[profile].items()}
    res = simulate_multi(assets, HOLD_START, HOLD_END, slip=.0002,
                         exit_granularity="sub")
    study = analyze_drawdowns(res.equity_curve, thresholds=(10, 20, 25, 30))
    standalone = {}
    for label, item in assets.items():
        one = simulate_multi({label: item}, HOLD_START, HOLD_END, slip=.0002,
                             exit_granularity="sub")
        standalone[label] = round(max(.01, 1 + one.return_pct / 100), 2)
    return {
        "compound_x": round(max(.01, 1 + res.return_pct / 100), 3),
        "reported_dd": round(res.max_dd_pct, 2),
        "mtm_dd": round(study.max_drawdown_pct, 2),
        "trades": res.trades,
        "win_rate": round(res.win_rate, 1),
        "standalone": standalone,
    }


def main() -> None:
    print(f"Clean OOS holdout {HOLD_START} -> {HOLD_END} (rolling-VWAP, frozen configs)\n")
    for profile in ("standard", "aggressive"):
        r = run_profile(profile)
        print(f"\n=== {profile.upper()} ===")
        print(f"  continuous : {r['compound_x']:,}x")
        print(f"  reported DD: {r['reported_dd']}%   MTM DD: {r['mtm_dd']}%")
        print(f"  trades     : {r['trades']}   win_rate: {r['win_rate']}%")
        print(f"  standalone : {r['standalone']}")


if __name__ == "__main__":
    main()
