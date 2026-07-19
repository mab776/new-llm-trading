"""Scalper grid runner — pre-committed TRAIN/TEST/HOLDOUT protocol.

Protocol (fixed BEFORE any results were looked at):
- TRAIN  : 2021-01-01 .. 2023-12-31, six half-year folds. ALL selection here.
- TEST   : 2024-01-01 .. 2025-05-31, three folds. Report-only, opened for the
           shortlisted finalists.
- HOLDOUT: 2025-06-01 .. 2026-07-17. Touched ONCE at the very end for the
           final candidate(s); invariance check only, never selection.

Selection gates on TRAIN (pre-committed):
  trades >= 200 over TRAIN, >= 5/6 folds positive, worst-fold maxDD <= 25%.
Rank survivors by geometric mean fold growth.

Usage:
  PYTHONPATH=. python opt/scalp/grid.py --tf 5m  --symbol BTCUSDT --out opt/scalp/results/grid_5m_btc.csv
  PYTHONPATH=. python opt/scalp/grid.py --tf 15m --symbol BTCUSDT --out opt/scalp/results/grid_15m_btc.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opt.scalp.engine import (
    ExecParams, aggregate, atr, build_subbars, load_futures, simulate,
)
from opt.scalp.strategies import STRATEGIES, build_context

DATA_START = "2020-10-01"
DATA_END = "2026-07-19"

TRAIN_FOLDS = [
    ("21H1", "2021-01-01", "2021-07-01"),
    ("21H2", "2021-07-01", "2022-01-01"),
    ("22H1", "2022-01-01", "2022-07-01"),
    ("22H2", "2022-07-01", "2023-01-01"),
    ("23H1", "2023-01-01", "2023-07-01"),
    ("23H2", "2023-07-01", "2024-01-01"),
]
TEST_FOLDS = [
    ("24H1", "2024-01-01", "2024-07-01"),
    ("24H2", "2024-07-01", "2025-01-01"),
    ("25P1", "2025-01-01", "2025-06-01"),
]
HOLDOUT = ("HOLD", "2025-06-01", "2026-07-18")

TF_MINUTES = {"5m": 5, "15m": 15}

# module-level data (loaded once; shared with fork()ed workers)
_G: dict = {}


def load_all(symbol: str, tf: str) -> None:
    df5 = load_futures(symbol, "5m", DATA_START, DATA_END)
    df = df5 if tf == "5m" else aggregate(df5, "15min")
    df_1h = load_futures(symbol, "1h", DATA_START, DATA_END)

    from llm_trading_bot.funding import aggregate_funding_to_bars, fetch_funding_history
    ccxt_sym = f"{symbol[:-4]}/USDT:USDT"
    fund = fetch_funding_history(ccxt_sym, start_date=DATA_START, end_date="2026-07-18")
    fund_arr = np.asarray(
        aggregate_funding_to_bars(fund, df.index, TF_MINUTES[tf] / 60.0)
    )

    a = atr(df["High"], df["Low"], df["Close"], 14).to_numpy()
    ohlc = {
        "open": df["Open"].to_numpy(), "high": df["High"].to_numpy(),
        "low": df["Low"].to_numpy(), "close": df["Close"].to_numpy(),
    }
    ctx = build_context(df_1h, df.index, TF_MINUTES[tf])
    # 15m primary gets data-resolved exit ordering from its 5m sub-bars (the
    # same pattern as the 4h product's 1h sub-bar replay). 5m primary has no
    # finer history (Bitget serves no reliable 1m) -> pure adverse-first.
    subs = build_subbars(df.index, 15, df5, 5) if tf == "15m" else None
    _G.update(df=df, ohlc=ohlc, atr=a, ctx=ctx, funding=fund_arr, tf=tf,
              index=df.index, sig_cache={}, subbars=subs)


def fold_bounds(label_start_end) -> tuple[int, int]:
    _, s, e = label_start_end
    idx = _G["index"]
    i0 = int(idx.searchsorted(pd.Timestamp(s, tz="UTC")))
    i1 = int(idx.searchsorted(pd.Timestamp(e, tz="UTC")))
    return i0, i1


def get_signals(strat: str, sig_params_key: tuple, sig_params: dict):
    cache = _G["sig_cache"]
    key = (strat, sig_params_key)
    if key not in cache:
        cache[key] = STRATEGIES[strat](_G["df"], sig_params, _G["ctx"])
    return cache[key]


def run_folds(strat: str, sig_params: dict, ep: ExecParams, folds) -> dict:
    sig_key = tuple(sorted(sig_params.items(), key=lambda kv: kv[0]))
    long_sig, short_sig, mel, mes = get_signals(strat, sig_key, sig_params)
    rows = {}
    growths = []
    for fold in folds:
        i0, i1 = fold_bounds(fold)
        r = simulate(
            _G["ohlc"], _G["atr"], long_sig, short_sig, ep,
            mean_exit_long=mel, mean_exit_short=mes,
            funding=_G["funding"], start_i=i0, end_i=i1,
            subbars=_G.get("subbars"),
        )
        rows[fold[0]] = r
        growths.append(max(r.growth_x, 0.01))
    geo = math.exp(sum(math.log(g) for g in growths) / len(growths))
    n_pos = sum(1 for g in growths if g > 1.0)
    return {
        "geo_growth": geo,
        "folds_positive": n_pos,
        "worst_growth": min(growths),
        "worst_dd": max(r.max_dd_pct for r in rows.values()),
        "trades": sum(r.trades for r in rows.values()),
        "fees": sum(r.fees_paid for r in rows.values()),
        "gross": sum(r.gross_pnl for r in rows.values()),
        "win_rate": (sum(r.win_rate * r.trades for r in rows.values())
                     / max(1, sum(r.trades for r in rows.values()))),
        "avg_hold": (sum(r.avg_hold_bars * r.trades for r in rows.values())
                     / max(1, sum(r.trades for r in rows.values()))),
        "maker_orders": sum(r.maker_orders for r in rows.values()),
        "maker_fills": sum(r.maker_fills for r in rows.values()),
        "per_fold": {k: (round(v.growth_x, 4), round(v.max_dd_pct, 1), v.trades)
                     for k, v in rows.items()},
    }


# ----------------------------------------------------------------------
# Grid definition
# ----------------------------------------------------------------------

def sig_grids(tf: str) -> list[tuple[str, dict]]:
    """Round 2: passive-fade era. rsi_reversion dropped (worst family round 1,
    bb covers the MR space); thresholds pushed deeper; donchian re-run after
    the read-only-array fix."""
    combos: list[tuple[str, dict]] = []
    # windows expressed in bars; scale for cadence (5m: 12 bars/h, 15m: 4 bars/h)
    bh = 12 if tf == "5m" else 4

    for n in (2 * bh, 4 * bh, 8 * bh):
        for z_in in (2.0, 2.5, 3.0):
            for er in (None, 0.3):
                for tg in (None, "with"):
                    combos.append(("bb_reversion",
                                   {"n": n, "z_in": z_in, "er_max": er,
                                    "er_n": 4 * bh, "trend_gate": tg}))
    for n in (2 * bh, 6 * bh, 12 * bh):
        for d_in in (2.5, 3.5, 4.5):
            for tg in (None, "with"):
                combos.append(("vwap_reversion",
                               {"n": n, "d_in": d_in, "trend_gate": tg}))
    for n in (4 * bh, 12 * bh, 24 * bh):
        for vx in (None, 1.2, 1.5):
            for tg in (None, "with"):
                combos.append(("donchian_breakout",
                               {"n": n, "vol_expand": vx, "vol_n": 8 * bh,
                                "trend_gate": tg}))
    return combos


def exec_grids(strat: str, tf: str) -> list[ExecParams]:
    bh = 12 if tf == "5m" else 4
    out = []
    mr = strat != "donchian_breakout"
    if mr:
        # passive fade: maker-only, limit rested INTO the extreme
        for sl_atr, tp_atr in ((1.0, 1.0), (1.5, 1.5), (1.0, 2.0),
                               (2.0, 1.0), (1.5, 2.5)):
            for off in (0.0, 0.25, 0.5, 1.0):
                for ttl in (1, 4):
                    for eom in (False, True):
                        out.append(ExecParams(
                            entry_mode="maker", limit_offset_atr=off,
                            maker_ttl=ttl, sl_atr=sl_atr, tp_atr=tp_atr,
                            time_stop_bars=8 * bh, exit_on_mean=eom,
                            cooldown_bars=bh // 2,
                        ))
    else:
        for sl_atr, tp_atr in ((1.5, 3.0), (2.0, 4.0), (2.5, 5.0), (2.0, 3.0)):
            for entry_mode in ("maker", "taker"):
                for trail in (0.0, 2.0):
                    out.append(ExecParams(
                        entry_mode=entry_mode, sl_atr=sl_atr, tp_atr=tp_atr,
                        time_stop_bars=0, trail_atr=trail, trail_arm_atr=1.0,
                        cooldown_bars=bh // 2,
                    ))
    return out


def ep_desc(ep: ExecParams) -> str:
    return (f"{ep.entry_mode}|off{ep.limit_offset_atr}|ttl{ep.maker_ttl}"
            f"|sl{ep.sl_atr}|tp{ep.tp_atr}|ts{ep.time_stop_bars}"
            f"|mean{int(ep.exit_on_mean)}|be{ep.breakeven_atr}|tr{ep.trail_atr}")


def worker(job):
    strat, sig_params, ep = job
    try:
        agg = run_folds(strat, sig_params, ep, TRAIN_FOLDS)
    except Exception as exc:  # keep the grid alive; report the failure
        return {"strategy": strat, "sig": str(sig_params), "exec": ep_desc(ep),
                "error": str(exc)}
    row = {"strategy": strat, "sig": str(sig_params), "exec": ep_desc(ep)}
    row.update({k: v for k, v in agg.items() if k != "per_fold"})
    row["per_fold"] = str(agg["per_fold"])
    return row


def donchian_refine_grids(tf: str) -> list[tuple[str, dict, ExecParams]]:
    """Round 3: zoom on the surviving donchian/vol-expansion plateau."""
    bh = 12 if tf == "5m" else 4
    jobs = []
    for n in (8, 16, 32, 48, 96):
        for vx in (1.3, 1.5, 1.8, 2.2):
            sp = {"n": n, "vol_expand": vx, "vol_n": 8 * bh, "trend_gate": None}
            for sl_atr in (1.5, 2.0, 2.5):
                for tp_atr in (3.0, 4.0, 5.0, 6.0):
                    for entry_mode in ("maker", "taker"):
                        jobs.append(("donchian_breakout", sp, ExecParams(
                            entry_mode=entry_mode, sl_atr=sl_atr, tp_atr=tp_atr,
                            cooldown_bars=bh // 2,
                        )))
                # trailing instead of fixed TP (runner-capture)
                for trail in (1.5, 2.5):
                    jobs.append(("donchian_breakout", sp, ExecParams(
                        entry_mode="maker", sl_atr=sl_atr, tp_atr=99.0,
                        trail_atr=trail, trail_arm_atr=1.0,
                        cooldown_bars=bh // 2,
                    )))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=("5m", "15m"), required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", choices=("full", "donchian"), default="full")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 8) - 8))
    args = ap.parse_args()

    t0 = time.monotonic()
    load_all(args.symbol, args.tf)
    print(f"data loaded: {len(_G['index'])} {args.tf} bars "
          f"({_G['index'][0]} .. {_G['index'][-1]}) in {time.monotonic()-t0:.1f}s",
          flush=True)

    if args.grid == "donchian":
        jobs = donchian_refine_grids(args.tf)
    else:
        jobs = []
        for strat, sp in sig_grids(args.tf):
            for ep in exec_grids(strat, args.tf):
                jobs.append((strat, sp, ep))
    print(f"{len(jobs)} jobs", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import concurrent.futures as cf
    rows = []
    t0 = time.monotonic()
    with cf.ProcessPoolExecutor(max_workers=args.procs) as pool:
        for k, row in enumerate(pool.map(worker, jobs, chunksize=8)):
            rows.append(row)
            if (k + 1) % 200 == 0:
                el = time.monotonic() - t0
                print(f"  {k+1}/{len(jobs)} ({el:.0f}s, {el/(k+1)*1000:.0f}ms/job)",
                      flush=True)

    fields = sorted({k for r in rows for k in r})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if "error" not in r or not r.get("error")]
    print(f"wrote {len(rows)} rows -> {out_path} ({time.monotonic()-t0:.0f}s)",
          flush=True)

    # pre-committed gates + ranking preview
    surv = [r for r in ok
            if r.get("trades", 0) >= 200 and r.get("folds_positive", 0) >= 5
            and r.get("worst_dd", 100) <= 25.0]
    surv.sort(key=lambda r: -r["geo_growth"])
    print(f"\n{len(surv)} TRAIN survivors (>=200 trades, >=5/6 folds, worstDD<=25%)")
    for r in surv[:15]:
        print(f"  {r['geo_growth']:.3f}x/fold  dd{r['worst_dd']:.0f}%  "
              f"tr{r['trades']}  win{r['win_rate']:.0f}%  {r['strategy']}  "
              f"{r['sig']}  {r['exec']}")


if __name__ == "__main__":
    main()
