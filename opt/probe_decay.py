"""Probe signal-decay features (Marc, 2026-07-17, after watching the live
+29 -> +19 -> +13 -> -21.6 whipsaw): (1) decay EXIT — leave when the signed
score fell N straight bars below a floor, without waiting for the +-20 flip
cliff; (2) entry FRESHNESS — refuse entries whose score arrived on the way
down from a higher peak.

Discipline: select on TRAIN half-years, report held-out TEST half-years and
the full folds. Funding settlement ON + 2bps slip (baseline reality). Knobs
are fastbt-only research code (DEFAULT_STRAT None => engine-identical); the
live scheduler has no decay code at all. Pre-commitment: unless TRAIN and
TEST both clearly improve, verdict is "no change".

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python opt/probe_decay.py
"""
from __future__ import annotations

import opt.driver as drv
from opt.driver import evaluate, TRAIN_FOLDS, TEST_FOLDS, FOLDS

drv.setup()
SLIP = 0.0002


def line(tag: str, strat: dict | None) -> None:
    tr = evaluate({}, folds=TRAIN_FOLDS, slip=SLIP, funding=True, strat=strat)
    te = evaluate({}, folds=TEST_FOLDS, slip=SLIP, funding=True, strat=strat)
    fl = evaluate({}, folds=FOLDS, slip=SLIP, funding=True, strat=strat)
    print(f"{tag:<30} TRAIN geo{tr['geo_pct']:+6.1f}% cx{tr['compound_x']:7.1f} "
          f"| TEST geo{te['geo_pct']:+6.1f}% cx{te['compound_x']:6.1f} "
          f"wf{te['worst_fold']:+6.0f} "
          f"| FULL cx{fl['compound_x']:7.1f} wf{fl['worst_fold']:+6.0f} "
          f"dd{fl['max_dd']:.0f} t{fl['total_trades']}", flush=True)


print("== baseline + regression (explicit Nones must equal baseline) ==")
line("baseline", None)
line("knobs present but None (==base)", {
    "entry_require_rising": None, "decay_exit_bars": None,
    "decay_exit_floor": None})

print("\n== entry freshness: require signed score rising over last K bars ==")
for k in (1, 2):
    line(f"require_rising K={k}", {"entry_require_rising": k})

print("\n== decay exit: N straight falling bars AND signed score < floor ==")
for n in (2, 3):
    for floor in (5.0, 10.0, 15.0):
        line(f"decay_exit N={n} floor={floor:.0f}",
             {"decay_exit_bars": n, "decay_exit_floor": floor})

print("\n== combined (best-guess pairs) ==")
for k, n, floor in ((1, 2, 10.0), (1, 3, 10.0), (2, 2, 5.0)):
    line(f"rising K={k} + exit N={n} f={floor:.0f}",
         {"entry_require_rising": k, "decay_exit_bars": n,
          "decay_exit_floor": floor})
