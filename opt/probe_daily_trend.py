"""Probe a dedicated 1d-TREND OVERLAY added on top of the composite score (Marc,
2026-07-19). Today the daily timeframe only casts the weak sign-only ±5 alignment
vote (same as 1h); this gives the daily regime a real, magnitude-aware term:

    raw_score += daily_trend_beta * shape(daily_metric)          (clamped ±100)

Knobs (all opt-in, DEFAULT_STRAT beta=0 ⇒ engine-identical, 397 tests green; live
scheduler untouched):
  daily_trend_source : "score"|"ema200"|"ema_stack"|"adx_di"  — the daily metric
  daily_trend_shape  : "sign"|"linear"|"tanh"                  — magnitude → weight
  daily_trend_beta   : points added at full daily trend
  daily_trend_k      : saturation scale for linear/tanh
  daily_trend_deadband, daily_trend_replace_align (drop the 1d ±5 to avoid double-count)

Alignment stays DISCRETE here (Marc: default discrete for this test), so we isolate
the overlay. Discipline: SELECT on BTC TRAIN half-years, REPORT held-out TEST + full
folds; the 3-asset OOS holdout on the TRAIN-top picks is an INVARIANCE check, not
selection. Pre-commitment: adopt only if a variant clearly beats baseline on TRAIN
*and* holds up on TEST (no TEST cherry-picking).

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python opt/probe_daily_trend.py
"""
from __future__ import annotations

import opt.driver as drv
from opt.driver import evaluate, TRAIN_FOLDS, TEST_FOLDS, FOLDS

SLIP = 0.0002
_rows: list[tuple] = []


def line(tag: str, strat: dict | None) -> None:
    tr = evaluate({}, folds=TRAIN_FOLDS, slip=SLIP, funding=True, strat=strat)
    te = evaluate({}, folds=TEST_FOLDS, slip=SLIP, funding=True, strat=strat)
    fl = evaluate({}, folds=FOLDS, slip=SLIP, funding=True, strat=strat)
    print(f"{tag:<40} TRAIN cx{tr['compound_x']:7.1f} geo{tr['geo_pct']:+6.1f} "
          f"| TEST cx{te['compound_x']:6.1f} geo{te['geo_pct']:+6.1f} wf{te['worst_fold']:+5.0f} "
          f"| FULL cx{fl['compound_x']:8.1f} dd{fl['max_dd']:.0f} t{fl['total_trades']}",
          flush=True)
    if strat is not None:
        _rows.append((tag, strat, tr['compound_x'], te['compound_x']))


def ov(source, shape, beta, replace=False, k=40.0, deadband=0.0) -> dict:
    return {"daily_trend_source": source, "daily_trend_shape": shape,
            "daily_trend_beta": beta, "daily_trend_k": k,
            "daily_trend_deadband": deadband, "daily_trend_replace_align": replace}


def part1_grid() -> None:
    drv.setup()  # BTC (config.json), discrete alignment
    print("\n== baseline + regression (overlay off must equal baseline) ==")
    line("baseline (no overlay)", None)
    line("overlay beta=0 (==base)", ov("score", "tanh", 0.0))

    for replace in (False, True):
        mode = "REPLACE 1d ±5" if replace else "ADD on top of 1d ±5"
        print(f"\n== overlay: {mode} ==")
        for source in ("score", "ema200", "ema_stack", "adx_di"):
            for shape in ("sign", "tanh"):
                for beta in (10.0, 20.0):
                    line(f"{source:9} {shape:6} beta={beta:.0f} {'repl' if replace else 'add '}",
                         ov(source, shape, beta, replace))


def part2_holdout(topn=5) -> None:
    from opt.holdout_oos import HOLD_START, HOLD_END, PROFILES, SYMBOLS, _load
    from opt.multi_asset import simulate_multi
    MIN_QTY = {"BTC": 0.0001, "ETH": 0.01, "SOL": 0.1}
    SIZE_STEP = {"BTC": 0.0001, "ETH": 0.01, "SOL": 0.1}
    picks = sorted(_rows, key=lambda r: r[2], reverse=True)[:topn]  # TRAIN-top
    print(f"\n== Part 2: 3-asset clean OOS holdout — TRAIN-top {topn} picks (invariance) ==")
    print(f"   (baseline discrete = 4.00x / 16.3%DD / 1057tr from the alignment probe)")
    assets = {lab: _load(SYMBOLS[lab], cfg) for lab, cfg in PROFILES["standard"].items()}
    for item in assets.values():
        item.config.backtesting.initial_balance = 100.0

    def go(tag, extra):
        strat = {"min_qty": MIN_QTY, "size_step": SIZE_STEP}
        strat.update(extra)
        r = simulate_multi(assets, HOLD_START, HOLD_END, slip=.0002,
                           exit_granularity="sub", strat=strat)
        print(f"  {tag:<36} {max(.01,1+r.return_pct/100):6.2f}x  maxDD {r.max_dd_pct:4.1f}%  "
              f"trades {r.trades:4d}  win {r.win_rate:.0f}%")
    go("discrete baseline (no overlay)", {})
    for tag, strat, trcx, tecx in picks:
        go(f"{tag} [TRAIN {trcx:.0f} TEST {tecx:.1f}]", strat)


if __name__ == "__main__":
    part1_grid()
    part2_holdout()
