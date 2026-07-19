"""Probe a CONTINUOUS multi-timeframe alignment score (Marc, 2026-07-19, after
the live BTC whipsaw: at the 04:00 UTC Jul-17 bar a near-zero secondary timeframe
flipped its flat ±5 sign-vote between live and backtest data, pushing the score
across the hard −20 opposite-exit cliff → dumped both longs at the low + flipped
short = the whole −$3 live loss).

The legacy alignment casts a binary ±5 vote on the SIGN of each secondary TF's
trend (magnitude discarded), so a barely-positive and a screaming-bullish TF both
add +5, and a tiny data wobble near zero teleports the score ±5 (up to ±10 with
two secondary TFs) — 25–50% of the ±20 thresholds. "continuous" scales the vote by
conviction: alignment_scale * tanh(tf_trend / alignment_k). A near-zero TF adds ~0.

Knob is opt-in (DEFAULT_STRAT alignment_mode="discrete" ⇒ engine-identical, 397
tests green); the live scheduler + engine call compute_composite_score without it.

Discipline: SELECT on BTC TRAIN half-years, REPORT held-out TEST + full folds; the
3-asset OOS holdout is an INVARIANCE check, not selection. Pre-commitment: the
RETURN case for continuous is "at least neutral on TRAIN *and* TEST" — we do NOT
expect more compound (the jitter test showed the discreteness isn't costing PnL);
the real payoff is the reproducibility metric (Part 3). Verdict "adopt" only if
returns are ≥ neutral AND reproducibility clearly improves.

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python opt/probe_alignment.py
"""
from __future__ import annotations

import numpy as np

import opt.driver as drv
from opt.driver import evaluate, TRAIN_FOLDS, TEST_FOLDS, FOLDS

SLIP = 0.0002


def line(tag: str, strat: dict | None) -> None:
    tr = evaluate({}, folds=TRAIN_FOLDS, slip=SLIP, funding=True, strat=strat)
    te = evaluate({}, folds=TEST_FOLDS, slip=SLIP, funding=True, strat=strat)
    fl = evaluate({}, folds=FOLDS, slip=SLIP, funding=True, strat=strat)
    print(f"{tag:<34} TRAIN geo{tr['geo_pct']:+6.1f}% cx{tr['compound_x']:7.1f} "
          f"| TEST geo{te['geo_pct']:+6.1f}% cx{te['compound_x']:6.1f} "
          f"wf{te['worst_fold']:+6.0f} "
          f"| FULL cx{fl['compound_x']:8.1f} wf{fl['worst_fold']:+6.0f} "
          f"dd{fl['max_dd']:.0f} t{fl['total_trades']}", flush=True)


def cont(k: float, scale: float = 5.0) -> dict:
    return {"alignment_mode": "continuous", "alignment_k": k, "alignment_scale": scale}


def part1_grid() -> None:
    drv.setup()  # BTC (config.json)
    print("\n== Part 1: BTC TRAIN/TEST grid (funding ON, 2bps slip) ==")
    print("== baseline + regression (explicit discrete must equal baseline) ==")
    line("baseline (discrete)", None)
    line("discrete knobs present (==base)",
         {"alignment_mode": "discrete", "alignment_scale": 5.0, "alignment_k": 30.0})
    print("\n== continuous, scale=5 (pure smoothing of the same ±5 cap), grid k ==")
    for k in (10, 15, 20, 30, 50):
        line(f"continuous scale=5 k={k}", cont(k))
    print("\n== continuous, stronger cap scale=8, grid k ==")
    for k in (20, 30):
        line(f"continuous scale=8 k={k}", cont(k, 8.0))


def part2_holdout(ks=(20, 30)) -> None:
    """3-asset clean OOS holdout, discrete vs continuous (INVARIANCE, not selection)."""
    from opt.holdout_oos import HOLD_START, HOLD_END, PROFILES, SYMBOLS, _load
    from opt.multi_asset import simulate_multi
    MIN_QTY = {"BTC": 0.0001, "ETH": 0.01, "SOL": 0.1}
    SIZE_STEP = {"BTC": 0.0001, "ETH": 0.01, "SOL": 0.1}
    print("\n== Part 2: 3-asset clean OOS holdout (2025-06→2026-04, $100 + real mins) ==")
    assets = {lab: _load(SYMBOLS[lab], cfg) for lab, cfg in PROFILES["standard"].items()}
    for item in assets.values():
        item.config.backtesting.initial_balance = 100.0

    def go(tag, extra):
        strat = {"min_qty": MIN_QTY, "size_step": SIZE_STEP}
        strat.update(extra)
        r = simulate_multi(assets, HOLD_START, HOLD_END, slip=.0002,
                           exit_granularity="sub", strat=strat)
        print(f"  {tag:<28} {max(.01,1+r.return_pct/100):6.2f}x  maxDD {r.max_dd_pct:4.1f}%  "
              f"trades {r.trades:4d}  win {r.win_rate:.0f}%")
    go("discrete (baseline)", {})
    for k in ks:
        go(f"continuous scale=5 k={k}", cont(k))


def part3_reproducibility() -> None:
    """The actual goal: how big a SCORE jump does a small secondary-TF data wobble
    cause, discrete vs continuous? The live/backtest divergence is exactly this —
    slightly different 1h/1d data at the same bar. Discrete is bimodal (0 on most
    bars, then a full ±5/±10 teleport when a near-zero TF flips its sign vote —
    the cliff-crosser that cost the −$3); continuous replaces that fat tail with a
    proportional sub-point nudge. We report the |Δscore| distribution under a ±3-pt
    wobble of every secondary-TF trend, and the count of catastrophic (≥5) jumps."""
    import pandas as pd
    from llm_trading_bot.scoring import score_trend, compute_composite_score
    from opt.holdout_oos import SYMBOLS, _load
    print("\n== Part 3: reproducibility — |Δscore| under a ±3-pt secondary-TF wobble ==")
    item = _load(SYMBOLS["BTC"], "config.json")
    pre, cfg = item.pre, item.config
    sc, tr = cfg.scoring, cfg.trading
    ts = pd.DatetimeIndex(pre.timestamps)
    lo, hi = pd.Timestamp("2025-06-01", tz="UTC"), pd.Timestamp("2026-04-30", tz="UTC")

    def raw_with_wobble(i, mode, k, dtrend):
        prim = pre.primary[i]
        # no secondary → raw == clamped primary weighted_total (alignment = 0)
        wt = compute_composite_score(
            indicators_by_tf={tr.primary_timeframe: prim}, weights=sc.weights,
            primary_timeframe=tr.primary_timeframe, confidence_min=sc.confidence_min,
            confidence_max=sc.confidence_max, scoring_points=getattr(sc, "points", None),
            alignment_mode=mode, alignment_k=k, alignment_scale=5.0).raw_score
        psign = 1.0 if wt > 0 else -1.0 if wt < 0 else 0.0
        align = 0.0
        for tf, ind in pre.sec_by_bar[i].items():
            t = score_trend(ind, getattr(sc, "points", None)).raw_score + dtrend
            if psign == 0.0 or t == 0.0:
                continue
            if mode == "continuous":
                align += 5.0 * float(np.tanh(t / k)) * psign
            else:
                align += 5.0 if (t > 0) == (wt > 0) else -5.0
        return max(-100.0, min(100.0, wt + align))

    print(f"  {'mode':<18}{'mean':>7}{'95pct':>7}{'max':>7}{'|Δ|≥5 bars':>12}")
    for mode, k in (("discrete", 30.0), ("continuous", 20.0),
                    ("continuous", 30.0), ("continuous", 50.0)):
        deltas = []
        for i, t in enumerate(ts):
            if not (lo <= t <= hi) or i < pre.warmup:
                continue
            base = raw_with_wobble(i, mode, k, 0.0)
            deltas.append(max(abs(raw_with_wobble(i, mode, k, +3.0) - base),
                              abs(raw_with_wobble(i, mode, k, -3.0) - base)))
        d = np.array(deltas)
        tag = f"{mode}" + (f" k={k:.0f}" if mode == "continuous" else "")
        print(f"  {tag:<18}{d.mean():>7.2f}{np.percentile(d,95):>7.2f}{d.max():>7.2f}"
              f"{int((d>=5).sum()):>12}")


if __name__ == "__main__":
    part1_grid()
    part2_holdout()
    part3_reproducibility()
