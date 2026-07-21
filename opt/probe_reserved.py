"""Probe: shared portfolio vs RESERVED per-asset capital slices.

Question (Marc, 2026-07-20): what happens if each asset owns a fixed reserved
portion of the portfolio instead of competing for one shared balance + global
margin cap? Motivated by live: the BTC position eats the 4.4% portfolio cap and
ETH/SOL adds get MIN_SIZE_SKIPped at small balance.

Design (pre-committed; equal thirds ONLY — no allocation tuning, the holdout is
worn and this is a mechanism comparison, not selection):
  * Window: the canonical clean-OOS holdout (2025-06 -> 2026-04-30), same
    execution model as opt.holdout_oos (maker entry, 1h sub-bar exits, funding,
    2bps slip, isolated liquidation).
  * SHARED   = simulate_multi over {BTC,ETH,SOL} with one compounding balance
    (reproduces the canonical anchor as a sanity check).
  * RESERVED = three standalone single-asset sims, each with 1/3 of the initial
    balance; portfolio equity = sum of the three curves (forward-filled), so
    max-DD is measured on the combined curve. No rebalancing between slices.
  * Axis 1 — structural (no minimums): pure architecture effect
    (cross-subsidy/compounding vs isolation). Scale-invariant.
  * Axis 2 — $193 with real Bitget minimums: today's practical effect
    (a 1/3 slice is ~$64; ETH's min lot is ~$19 notional).

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.probe_reserved
"""
from __future__ import annotations

import sys

import pandas as pd

import opt.fastbt as fb
from llm_trading_bot.config import load_config
from llm_trading_bot.data import configure_cache, fetch_multi_timeframe
from llm_trading_bot.funding import fetch_funding_history, aggregate_funding_to_bars
from llm_trading_bot.timeframes import timeframe_hours
from opt.drawdown import EquityPoint, analyze_drawdowns
from opt.multi_asset import AssetInput, simulate_multi

HOLD_START = "2025-06-01"
HOLD_END = "2026-04-30"   # stop before the May-2026 Bitget 1h hole
LOAD_START = "2020-10-01"

CONFIGS = {"BTC": "config.json", "ETH": "config-eth.json", "SOL": "config-sol.json"}
SYMBOLS = {"BTC": "BTC/USDT:USDT", "ETH": "ETH/USDT:USDT", "SOL": "SOL/USDT:USDT"}
MIN_QTY = {"BTC": 0.0001, "ETH": 0.01, "SOL": 0.1}
SIZE_STEP = dict(MIN_QTY)

LIVE_BALANCE = 193.0


def _load(label: str) -> AssetInput:
    symbol = SYMBOLS[label]
    cfg = load_config(CONFIGS[label])
    configure_cache(cfg.data_cache.ttl_seconds)
    ds = cfg.data_source
    ds.exchange_symbol = symbol
    cfg.trading.entry_mode = "maker"
    data = fetch_multi_timeframe(
        symbol, cfg.trading.timeframes,
        start_date=LOAD_START, end_date=HOLD_END,
        warmup_periods=0, source=ds.source, market=ds.market,
    )
    print(f"  {label} 4h: {len(data['4h'])} rows "
          f"{data['4h'].index[0].date()} -> {data['4h'].index[-1].date()}", file=sys.stderr)
    pre = fb.precompute(data, cfg.trading.primary_timeframe, 200)
    fund = fetch_funding_history(symbol, start_date="2020-08-01", end_date=HOLD_END)
    funding = aggregate_funding_to_bars(
        fund, pd.DatetimeIndex(pre.timestamps),
        timeframe_hours(cfg.trading.primary_timeframe),
    )
    return AssetInput(pre, cfg, funding)


def _with_balance(assets: dict[str, AssetInput], balance: float):
    for item in assets.values():
        item.config.backtesting.initial_balance = balance


def _run(assets: dict[str, AssetInput], balance: float, mins: bool):
    _with_balance(assets, balance)
    strat = {"min_qty": MIN_QTY, "size_step": SIZE_STEP} if mins else None
    return simulate_multi(assets, HOLD_START, HOLD_END, slip=.0002,
                          exit_granularity="sub", strat=strat)


def _merged_curve(results: dict[str, object]) -> list[EquityPoint]:
    """Sum standalone equity curves on the union timeline (forward-fill)."""
    series = {}
    for label, res in results.items():
        pts = res.equity_curve
        series[label] = pd.Series([p.equity for p in pts],
                                  index=pd.DatetimeIndex([p.timestamp for p in pts]))
    frame = pd.DataFrame(series).sort_index().ffill().dropna()
    total = frame.sum(axis=1)
    peak = total.cummax()
    dd = (peak - total) / peak * 100.0   # positive-DD convention (opt/drawdown.py)
    return [EquityPoint(ts, eq, d) for ts, eq, d in zip(total.index, total, dd)]


def _summary(tag: str, mult: float, mtm_dd: float, trades: int, extra: str = ""):
    print(f"  {tag:34} {mult:8.2f}x   MTM DD {mtm_dd:5.1f}%   trades {trades:4d}  {extra}")


def run_axis(assets: dict[str, AssetInput], balance: float, mins: bool, title: str):
    print(f"\n=== {title} (initial ${balance:g}, mins={'REAL' if mins else 'off'}) ===")

    shared = _run(assets, balance, mins)
    shared_study = analyze_drawdowns(shared.equity_curve, thresholds=(10, 20, 25, 30))
    _summary("SHARED (one pot, global cap)",
             max(.01, 1 + shared.return_pct / 100), shared_study.max_drawdown_pct,
             shared.trades)

    slice_bal = balance / len(assets)
    singles = {}
    for label in assets:
        singles[label] = _run({label: assets[label]}, slice_bal, mins)
    merged = _merged_curve(singles)
    merged_study = analyze_drawdowns(merged, thresholds=(10, 20, 25, 30))
    final = sum(r.final_balance for r in singles.values())
    mult = final / balance
    trades = sum(r.trades for r in singles.values())
    _summary(f"RESERVED (thirds of ${balance:g})", mult,
             merged_study.max_drawdown_pct, trades)
    for label, r in singles.items():
        print(f"      {label}: {max(.01, 1 + r.return_pct / 100):6.2f}x on "
              f"${slice_bal:.2f}  (trades {r.trades}, win {r.win_rate:.0f}%, "
              f"own-DD {r.max_dd_pct:.1f}%)")
    return shared, singles


def main() -> None:
    print(f"Reserved-allocation probe | clean OOS {HOLD_START} -> {HOLD_END} | "
          f"maker + sub-bar exits + funding + 2bps slip")
    assets = {label: _load(label) for label in CONFIGS}

    # Axis 1: structure only (no minimums; scale-invariant)
    run_axis(assets, 3000.0, mins=False, title="STRUCTURAL (no minimums)")

    # Axis 2: today's balance with real Bitget minimums
    run_axis(assets, LIVE_BALANCE, mins=True, title="LIVE-SCALE granularity")

    print("\nNotes: RESERVED = no cross-subsidy (a slice can't borrow another's "
          "margin room) and no shared compounding (winners compound only their "
          "own slice). Equal thirds by design — no allocation tuning.")


if __name__ == "__main__":
    main()
