"""Probe: conditional cap-overshoot for min-size-squeezed STRONG entries.

Marc's idea (2026-07-20 night, after rotation was rejected): don't evict anyone —
just let the global margin/notional caps STRETCH when they are squeezing a
strong signal below the exchange minimum (the live MIN_SIZE_SKIP: ETH STRONG
+31.2 refused at $0.10 free margin while BTC held the 4.4% cap). Two searched
knobs: the overshoot allowance O (caps become (1+O)x) and the |score| S that
unlocks the provision.

Control arm: unconditional "floor" policy (Marc's earlier idea; sizing_scenarios
measured it slightly better than skip at $100 but it was never adopted). If the
conditional provision cannot beat plain floor on TRAIN, the score gate and
bounded overshoot add nothing.

PROTOCOL NOTE — folds run WITH real Bitget minimums at $193 initial (per fold):
min-size cancels do not exist in frictionless sims, so the house frictionless
folds cannot see this mechanism. This makes the probe a SMALL-ACCOUNT question
by construction: the provision's benefit decays as balance grows (skips fade
~20.5% tax @$100 -> 0 @$2500) and whatever wins here is a graduation aid, not a
structural edge.

PRE-COMMITTED PROTOCOL (written before results):
  * Folds: house half-year interleave, multi-asset portfolio sims (BTC+ETH+SOL,
    maker, sub-bar exits, funding, 2bps slip), real mins, $193/fold.
    TRAIN = 21H1,22H1,23H1,24H1,25H1  /  TEST = 21H2,22H2,23H2,24H2.
  * Arms: baseline(skip), floor(control), grid O in {0.25,0.5,1.0} x
    S in {22,25,30} — selected on TRAIN geo only.
  * Gates: (1) best conditional TRAIN > baseline TRAIN, else verdict "does not
    help", stop; also report whether it beats the floor control (if not:
    "just floor" is the whole story). (2) TEST >= baseline TEST - 2 pts for the
    TRAIN-top-3. (3) OOS holdout invariance @ $193 + mins: ratio >= 0.92
    (report-only sanity, holdout is worn).
  * No adoption from this script; live would additionally need scheduler-side
    code (the sim knob only exists in simulate_multi).

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.probe_overshoot
"""
from __future__ import annotations

import math

from opt.driver import HALF_FOLDS
from opt.multi_asset import simulate_multi
from opt.probe_reserved import (CONFIGS, MIN_QTY, SIZE_STEP, _load,
                                _with_balance)

SLIP = 0.0002
TRAIN = [HALF_FOLDS[i] for i in (0, 2, 4, 6, 8)]
TEST = [HALF_FOLDS[i] for i in (1, 3, 5, 7)]
HOLD_START, HOLD_END = "2025-06-01", "2026-04-30"
BALANCE = 193.0
GRID_O = (0.25, 0.5, 1.0)
GRID_S = (22.0, 25.0, 30.0)


def _strat(extra: dict | None) -> dict:
    counters = {"skips": 0, "floors": 0, "overshoots": 0}
    strat = {"min_qty": MIN_QTY, "size_step": SIZE_STEP,
             "_min_counters": counters}
    if extra:
        strat.update(extra)
    return strat


def eval_folds(assets: dict, folds, extra: dict | None) -> dict:
    rets, trades = [], 0
    skips = floors = overs = 0
    per_fold = []
    for name, start, end in folds:
        _with_balance(assets, BALANCE)
        strat = _strat(extra)
        res = simulate_multi(assets, start, end, slip=SLIP,
                             exit_granularity="sub", strat=strat)
        rets.append(res.return_pct / 100.0)
        trades += res.trades
        c = strat["_min_counters"]
        skips += c["skips"]; floors += c["floors"]; overs += c["overshoots"]
        per_fold.append((name, res.return_pct))
    geo = (math.prod(1 + r for r in rets) ** (1 / len(rets)) - 1) * 100
    return {"geo": geo, "trades": trades, "skips": skips, "floors": floors,
            "overshoots": overs, "folds": per_fold,
            "worst": min(r for _, r in per_fold)}


def holdout(assets: dict, extra: dict | None) -> float:
    _with_balance(assets, BALANCE)
    res = simulate_multi(assets, HOLD_START, HOLD_END, slip=SLIP,
                         exit_granularity="sub", strat=_strat(extra))
    return max(.01, 1 + res.return_pct / 100)


def main() -> None:
    print("Conditional cap-overshoot probe | multi-asset half-year folds "
          f"WITH real mins @ ${BALANCE:g}/fold | maker + sub-bar + funding + 2bps")
    assets = {label: _load(label) for label in CONFIGS}

    print(f"\n{'arm':>16}{'TRAINgeo':>10}{'worstF':>8}{'skip':>6}{'flr':>5}"
          f"{'over':>6}{'trades':>7}")
    base = eval_folds(assets, TRAIN, None)
    print(f"{'baseline(skip)':>16}{base['geo']:>+10.1f}{base['worst']:>+8.1f}"
          f"{base['skips']:>6d}{base['floors']:>5d}{base['overshoots']:>6d}"
          f"{base['trades']:>7d}", flush=True)
    floor = eval_folds(assets, TRAIN, {"min_size_policy": "floor"})
    print(f"{'floor(control)':>16}{floor['geo']:>+10.1f}{floor['worst']:>+8.1f}"
          f"{floor['skips']:>6d}{floor['floors']:>5d}{floor['overshoots']:>6d}"
          f"{floor['trades']:>7d}", flush=True)

    rows = []
    for o in GRID_O:
        for s in GRID_S:
            extra = {"min_size_overshoot": o, "min_size_overshoot_score": s}
            r = eval_folds(assets, TRAIN, extra)
            rows.append((o, s, r))
            print(f"{f'O={o:g} S={s:g}':>16}{r['geo']:>+10.1f}{r['worst']:>+8.1f}"
                  f"{r['skips']:>6d}{r['floors']:>5d}{r['overshoots']:>6d}"
                  f"{r['trades']:>7d}", flush=True)

    rows.sort(key=lambda t: -t[2]["geo"])
    best = rows[0]
    print(f"\nTRAIN: baseline {base['geo']:+.1f} | floor {floor['geo']:+.1f} | "
          f"best conditional O={best[0]:g} S={best[1]:g} -> {best[2]['geo']:+.1f} "
          f"({best[2]['overshoots']} overshoots)")
    if best[2]["geo"] <= base["geo"]:
        print("VERDICT (gate 1): no conditional arm beats baseline on TRAIN — "
              "the provision does not help. Stop.")
        return
    if best[2]["geo"] <= floor["geo"]:
        print("NOTE: best conditional does NOT beat the plain-floor control — "
              "the score gate + bounded overshoot add nothing over 'floor'.")

    print("\nTEST (baseline + floor + top-3 by TRAIN):")
    bt = eval_folds(assets, TEST, None)
    ft = eval_folds(assets, TEST, {"min_size_policy": "floor"})
    print(f"  baseline      TEST geo {bt['geo']:+.1f}  worst {bt['worst']:+.1f}")
    print(f"  floor         TEST geo {ft['geo']:+.1f}  worst {ft['worst']:+.1f}")
    for o, s, tr in rows[:3]:
        te = eval_folds(assets, TEST,
                        {"min_size_overshoot": o, "min_size_overshoot_score": s})
        gate = "PASS" if te["geo"] >= bt["geo"] - 2.0 else "FAIL"
        print(f"  O={o:<5g}S={s:<5g} TEST geo {te['geo']:+.1f}  "
              f"worst {te['worst']:+.1f}  over {te['overshoots']}  "
              f"[gate2 {gate}]", flush=True)

    o, s, _ = rows[0]
    hb = holdout(assets, None)
    hc = holdout(assets, {"min_size_overshoot": o, "min_size_overshoot_score": s})
    ratio = hc / hb
    print(f"\nHoldout invariance (${BALANCE:g} + mins): baseline {hb:.2f}x vs "
          f"O={o:g}/S={s:g} {hc:.2f}x  ratio {ratio:.3f} "
          f"[gate3 {'PASS' if ratio >= 0.92 else 'FAIL'}]")
    print("\nReminder: small-account provision by construction (benefit decays "
          "with balance); adoption needs scheduler-side code + the full protocol.")


if __name__ == "__main__":
    main()
