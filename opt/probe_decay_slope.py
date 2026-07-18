"""Continuous/contrarian extension of the entry-freshness gate ("K fractional
or negative" — Marc, 2026-07-18). Slope = 1-bar change of the
direction-signed composite score at entry time.

- entry_slope_min: block entries with slope < X. X=0 must equal yesterday's
  require_rising K=1 (sanity anchor); negative X = tolerant "fractional"
  strictness interpolating toward baseline.
- entry_slope_max: block entries with slope > X — the contrarian gate (only
  decaying signals enter).

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python opt/probe_decay_slope.py
"""
from __future__ import annotations

import opt.driver as drv
from opt.driver import evaluate, TRAIN_FOLDS, TEST_FOLDS, FOLDS

drv.setup()
SLIP = 0.0002


def line(tag, strat):
    tr = evaluate({}, folds=TRAIN_FOLDS, slip=SLIP, funding=True, strat=strat)
    te = evaluate({}, folds=TEST_FOLDS, slip=SLIP, funding=True, strat=strat)
    fl = evaluate({}, folds=FOLDS, slip=SLIP, funding=True, strat=strat)
    print(f"{tag:<30} TRAIN geo{tr['geo_pct']:+6.1f}% cx{tr['compound_x']:7.1f} "
          f"| TEST geo{te['geo_pct']:+6.1f}% cx{te['compound_x']:6.1f} "
          f"wf{te['worst_fold']:+6.0f} "
          f"| FULL cx{fl['compound_x']:7.1f} wf{fl['worst_fold']:+6.0f} "
          f"dd{fl['max_dd']:.0f} t{fl['total_trades']}", flush=True)


print("== anchors ==")
line("baseline", None)
line("slope_min=0 (==K=1 yesterday)", {"entry_slope_min": 0.0})

print("\n== fractional strictness: tolerate mild decay, block steep ==")
for x in (-5.0, -10.0, -15.0, -20.0):
    line(f"slope_min={x:+.0f}", {"entry_slope_min": x})

print("\n== negative K (contrarian): only decaying entries ==")
for x in (0.0, -5.0):
    line(f"slope_max={x:+.0f}", {"entry_slope_max": x})

print("\n== require STEEP rise (extra-strict, for the dose-response curve) ==")
line("slope_min=+5", {"entry_slope_min": 5.0})
