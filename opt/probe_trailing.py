"""Probe: static trailing-stop retune (activation/callback) — house gates.

Origin (2026-07-24 overnight): the multi-asset walk-forward study's winners
cluster HARD on trailing — 32/32 activation below static (med 0.76 vs 0.94),
28/32 callback inside the newly-opened 0.20-0.25 zone (med 0.22) — and a
post-hoc decomposition cell (static + activation 0.76 / callback 0.22 only)
chains 8.80x vs static on the 2023-01..2026-04 OOS span, MORE than the
walk-forward median (~8.1x), with lower DD in every window. Fourth and fifth
independent sightings of the tighter-trailing family (#4 rotation, #10
trendline stops, BTC-rerun cluster, multi-study cluster). If this passes the
house gates, most of the walk-forward pipeline's value may be bankable as a
one-line static config change.

⚠️ Provenance: the candidate values came from walk-forward TRAIN-window
winners (not from unseen outcomes), but the *decision to test trailing* is
informed by the whole study — hence this probe re-derives the choice through
the standard discipline on the house folds, which include 2021-2022 regimes
the walk-forward span never saw.

PRE-COMMITTED PROTOCOL (written before results):
  * Multi-asset portfolio sims (BTC+ETH+SOL, maker, sub-bar exits, funding,
    2bps slip, frictionless): TRAIN = 21H1,22H1,23H1,24H1,25H1;
    TEST = 21H2,22H2,23H2,24H2 (probe_btc_delay conventions).
  * Grid: activation {0.70, 0.76, 0.82, 0.88} x callback {0.20, 0.22, 0.26}
    + baseline (0.94/0.33). Values pre-set, no refinement pass.
  * GATE 1: best cell beats baseline TRAIN geo. GATE 2: survivors beat
    baseline TEST geo. Worst-fold + maxDD reported alongside (the DD claim
    is half the pitch). Holdout @$193 + real mins: invariance only.
  * No adoption from this script — Marc's go + supervised deploy required;
    trailing params live in all three configs + both profiles.

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.probe_trailing
"""
from __future__ import annotations

import math
import sys

from opt.multi_asset import simulate_multi
from opt.probe_reserved import (CONFIGS, MIN_QTY, SIZE_STEP, _load,
                                _with_balance)
from opt.driver import HALF_FOLDS

SLIP = 0.0002
TRAIN = [HALF_FOLDS[i] for i in (0, 2, 4, 6, 8)]
TEST = [HALF_FOLDS[i] for i in (1, 3, 5, 7)]
HOLD_START, HOLD_END = "2025-06-01", "2026-04-30"

ACTS = (0.70, 0.76, 0.82, 0.88)
CBS = (0.20, 0.22, 0.26)


def set_trailing(assets, act: float | None, cb: float | None) -> None:
    for item in assets.values():
        ts = item.config.trading.trailing_stop
        if act is None:
            ts.activation_pct, ts.callback_pct = item._orig_trailing
        else:
            ts.activation_pct, ts.callback_pct = act, cb


def eval_folds(assets, folds) -> dict:
    rets, trades, per = [], 0, []
    dd = 0.0
    for name, start, end in folds:
        _with_balance(assets, 3000.0)
        r = simulate_multi(assets, start, end, slip=SLIP,
                           exit_granularity="sub")
        rets.append(r.return_pct / 100.0)
        trades += r.trades
        dd = max(dd, r.max_dd_pct)
        per.append((name, r.return_pct))
    geo = (math.prod(1 + x for x in rets) ** (1 / len(rets)) - 1) * 100
    return {"geo": geo, "worst": min(v for _, v in per), "dd": dd,
            "trades": trades, "folds": per}


def holdout(assets) -> tuple[float, float]:
    _with_balance(assets, 193.0)
    r = simulate_multi(assets, HOLD_START, HOLD_END, slip=SLIP,
                       exit_granularity="sub",
                       strat={"min_qty": MIN_QTY, "size_step": SIZE_STEP})
    return max(.01, 1 + r.return_pct / 100), r.max_dd_pct


def row(tag, r) -> str:
    folds = "  ".join(f"{n}:{v:+.0f}" for n, v in r["folds"])
    return (f"{tag:>12}{r['geo']:>+9.1f}{r['worst']:>+8.1f}{r['dd']:>6.1f}"
            f"{r['trades']:>6d}   {folds}")


def main() -> None:
    print("Trailing retune probe | multi-asset half-year folds | maker + sub-bar "
          "+ funding + 2bps | frictionless folds, mins on holdout only")
    assets = {label: _load(label) for label in CONFIGS}
    for item in assets.values():
        ts = item.config.trading.trailing_stop
        item._orig_trailing = (ts.activation_pct, ts.callback_pct)

    print(f"\n{'cell':>12}{'TRAINgeo':>9}{'worstF':>8}{'maxDD':>6}{'tr':>6}   folds")
    set_trailing(assets, None, None)
    base = eval_folds(assets, TRAIN)
    print(row("baseline", base), flush=True)
    train = {}
    for act in ACTS:
        for cb in CBS:
            set_trailing(assets, act, cb)
            train[(act, cb)] = eval_folds(assets, TRAIN)
            print(row(f"a{act}/c{cb}", train[(act, cb)]), flush=True)

    best = max(train, key=lambda k: train[k]["geo"])
    winners = [k for k in train if train[k]["geo"] > base["geo"]]
    print(f"\nBaseline TRAIN {base['geo']:+.1f} | best a{best[0]}/c{best[1]} "
          f"{train[best]['geo']:+.1f} | gate-1 winners: {len(winners)}/12")
    if train[best]["geo"] <= base["geo"]:
        print("VERDICT (gate 1): no cell beats baseline TRAIN. STOP.")
        return

    print("\nGate 1 PASSED. TEST (baseline + best cell only — one shot, "
          "no cell-shopping on TEST):")
    print(f"{'cell':>12}{'TESTgeo':>9}{'worstF':>8}{'maxDD':>6}{'tr':>6}   folds")
    set_trailing(assets, None, None)
    tb = eval_folds(assets, TEST)
    print(row("baseline", tb))
    set_trailing(assets, *best)
    tt = eval_folds(assets, TEST)
    print(row(f"a{best[0]}/c{best[1]}", tt))
    if tt["geo"] <= tb["geo"]:
        print(f"\nVERDICT (gate 2): best TRAIN cell fails TEST "
              f"({tt['geo']:+.1f} vs {tb['geo']:+.1f}) — split noise. STOP.")
        return
    print(f"\nGate 2 PASSED ({tt['geo']:+.1f} vs {tb['geo']:+.1f}). "
          "Holdout @$193 + mins (invariance only):")
    set_trailing(assets, None, None)
    hb, hbd = holdout(assets)
    set_trailing(assets, *best)
    ht, htd = holdout(assets)
    print(f"  baseline: {hb:.2f}x dd {hbd:.1f}% | a{best[0]}/c{best[1]}: "
          f"{ht:.2f}x dd {htd:.1f}% | ratio {ht/hb:.3f}")
    print("\nVERDICT: gates passed — candidate for Marc's deploy decision "
          "(supervised, all configs + both profiles).")


if __name__ == "__main__":
    main()
