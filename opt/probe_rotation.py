"""Probe: cross-asset rotation — evict a weak position for a cap-squeezed STRONG entry.

Marc's idea (2026-07-20): signal_flip already gives up a position when its OWN
symbol turns hard against it; what about giving one up when ANOTHER asset's
signal is screaming and can't get margin? Live motivation: ETH STRONG +31.2 got
MIN_SIZE_SKIPped while BTC (weaker score) held the whole 4.4% portfolio cap.
Related evidence, both directions: the reserved-allocation probe (37b898a)
showed cross-subsidy concentration IS the edge (rotation doubles down on it);
the decay-exit probe (f6b31eb) showed exiting on score-weakness alone HURTS
(weak-score holds are net-positive). Rotation differs from decay-exit in that
the exit is paid for by a better entry, and only fires when the cap binds.

Mechanism (opt-in knobs in simulate_multi, default-off = engine-identical):
  When a STRONG entry's post-cap risk < 0.5x its pre-cap risk (squeeze fraction
  is a FIXED design constant, not searched), the weakest OTHER symbol's open
  position — signed support = raw_score for LONG / -raw_score for SHORT — is
  closed at taker (reason "rotation") iff:
      support <= W   (victim is weak)     AND
      |new score| - support >= G          (newcomer clearly stronger)
  then exposure caps recompute on the freed margin.

PRE-COMMITTED PROTOCOL (written before results):
  * Folds: the house half-year interleave on MULTI-ASSET portfolio sims
    (BTC+ETH+SOL, maker entry, sub-bar exits, funding, 2bps slip, no mins):
    TRAIN = 21H1,22H1,23H1,24H1,25H1  /  TEST = 21H2,22H2,23H2,24H2.
  * Grid: W in {0,5,10,15} x G in {10,20,30}, selected on TRAIN geo-mean only.
  * Gates: (1) best TRAIN must beat baseline TRAIN, else verdict = mechanism
    does not help, stop. (2) TEST (top-3 by TRAIN + baseline only) must be
    >= baseline TEST - 2 pts. (3) OOS holdout invariance at $193 + real mins:
    candidate/baseline compound ratio >= 0.92 (report-only sanity, NOT selection).
  * No adoption from this script — live changes remain Marc's call with the
    full deployment protocol.

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.probe_rotation
"""
from __future__ import annotations

import math
import sys

from opt.driver import HALF_FOLDS
from opt.multi_asset import simulate_multi
from opt.probe_reserved import (CONFIGS, MIN_QTY, SIZE_STEP, _load,
                                _with_balance)

SLIP = 0.0002
TRAIN = [HALF_FOLDS[i] for i in (0, 2, 4, 6, 8)]
TEST = [HALF_FOLDS[i] for i in (1, 3, 5, 7)]
HOLD_START, HOLD_END = "2025-06-01", "2026-04-30"
GRID_W = (0.0, 5.0, 10.0, 15.0)
GRID_G = (10.0, 20.0, 30.0)


def _rot(w: float | None, g: float | None) -> dict | None:
    if w is None:
        return None
    return {"rotate_weak_support": w, "rotate_min_gap": g}


def eval_folds(assets: dict, folds, strat: dict | None) -> dict:
    rets, rota, trades = [], 0, 0
    per_fold = []
    for name, start, end in folds:
        _with_balance(assets, 3000.0)
        res = simulate_multi(assets, start, end, slip=SLIP,
                             exit_granularity="sub", strat=strat)
        rets.append(res.return_pct / 100.0)
        rota += res.rotations
        trades += res.trades
        per_fold.append((name, res.return_pct))
    geo = (math.prod(1 + r for r in rets) ** (1 / len(rets)) - 1) * 100
    return {"geo": geo, "rotations": rota, "trades": trades,
            "folds": per_fold, "worst": min(r for _, r in per_fold)}


def holdout(assets: dict, strat_extra: dict | None) -> float:
    _with_balance(assets, 193.0)
    strat = {"min_qty": MIN_QTY, "size_step": SIZE_STEP}
    if strat_extra:
        strat.update(strat_extra)
    res = simulate_multi(assets, HOLD_START, HOLD_END, slip=SLIP,
                         exit_granularity="sub", strat=strat)
    return max(.01, 1 + res.return_pct / 100)


def main() -> None:
    print("Cross-asset rotation probe | multi-asset half-year folds | "
          "maker + sub-bar + funding + 2bps | frictionless folds, mins on holdout only")
    assets = {label: _load(label) for label in CONFIGS}

    print(f"\n{'W':>4}{'G':>4}{'TRAINgeo':>10}{'worstF':>8}{'rot':>5}{'trades':>7}")
    base = eval_folds(assets, TRAIN, None)
    print(f"{'—':>4}{'—':>4}{base['geo']:>+10.1f}{base['worst']:>+8.1f}"
          f"{base['rotations']:>5d}{base['trades']:>7d}   <- baseline", flush=True)
    rows = []
    for w in GRID_W:
        for g in GRID_G:
            r = eval_folds(assets, TRAIN, _rot(w, g))
            rows.append((w, g, r))
            print(f"{w:>4.0f}{g:>4.0f}{r['geo']:>+10.1f}{r['worst']:>+8.1f}"
                  f"{r['rotations']:>5d}{r['trades']:>7d}", flush=True)

    rows.sort(key=lambda t: -t[2]["geo"])
    best = rows[0]
    print(f"\nBaseline TRAIN geo {base['geo']:+.1f} | best candidate "
          f"W={best[0]:.0f} G={best[1]:.0f} -> {best[2]['geo']:+.1f} "
          f"({best[2]['rotations']} rotations)")
    if best[2]["geo"] <= base["geo"]:
        print("VERDICT (gate 1): no candidate beats baseline on TRAIN — "
              "rotation does not help. Stop.")
        return

    print("\nTEST (baseline + top-3 by TRAIN):")
    bt = eval_folds(assets, TEST, None)
    print(f"  baseline      TEST geo {bt['geo']:+.1f}  worst {bt['worst']:+.1f}")
    for w, g, tr in rows[:3]:
        te = eval_folds(assets, TEST, _rot(w, g))
        gate = "PASS" if te["geo"] >= bt["geo"] - 2.0 else "FAIL"
        print(f"  W={w:<3.0f} G={g:<3.0f}  TEST geo {te['geo']:+.1f}  "
              f"worst {te['worst']:+.1f}  rot {te['rotations']}  [gate2 {gate}]",
              flush=True)

    w, g, _ = rows[0]
    hb = holdout(assets, None)
    hc = holdout(assets, _rot(w, g))
    ratio = hc / hb
    print(f"\nHoldout invariance ($193 + real mins): baseline {hb:.2f}x vs "
          f"W={w:.0f}/G={g:.0f} {hc:.2f}x  ratio {ratio:.3f} "
          f"[gate3 {'PASS' if ratio >= 0.92 else 'FAIL'}]")
    print("\nReminder: no adoption from this script; full deployment protocol "
          "applies if the gates pass and Marc wants it.")


if __name__ == "__main__":
    main()
