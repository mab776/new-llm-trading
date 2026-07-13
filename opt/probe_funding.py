"""
Probe the funding-as-signal gate. Discipline: select on TRAIN half-years, report
held-out TEST half-years AND the full yearly folds (chronological regimes). Always
funding settlement ON + slippage. Defaults (no gate) must reproduce the baseline exactly.

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python opt/probe_funding.py
"""
from __future__ import annotations
import sys
import opt.driver as drv
from opt.driver import evaluate, TRAIN_FOLDS, TEST_FOLDS, FOLDS

drv.setup()
SLIP = 0.0002  # 2 bps/side

def ev(strat, folds, gate):
    return evaluate({}, folds=folds, slip=SLIP, funding=True,
                    strat=strat, fund_signal=gate)

def line(tag, strat, gate):
    tr = ev(strat, TRAIN_FOLDS, gate)
    te = ev(strat, TEST_FOLDS, gate)
    fl = ev(strat, FOLDS, gate)
    print(f"{tag:<34} TRAIN geo{tr['geo_pct']:+6.1f}% cx{tr['compound_x']:7.1f} "
          f"| TEST geo{te['geo_pct']:+6.1f}% cx{te['compound_x']:6.1f} wf{te['worst_fold']:+6.0f} "
          f"| FULL cx{fl['compound_x']:7.1f} wf{fl['worst_fold']:+6.0f} dd{fl['max_dd']:.0f} t{fl['total_trades']}")

# 0. Baseline — gate OFF. Also assert gate ON with no thresholds == gate OFF (regression).
print("== baseline (no funding gate) ==")
line("baseline funding-settle only", None, False)
line("gate-on but no thresholds (==base)", {"funding_block_long": None, "funding_block_short": None}, True)

print("\n== block LONG when funding high AND downtrend (trend-gated) ==")
for thr in (1.0e-4, 1.5e-4, 2.0e-4, 3.0e-4):
    line(f"blockLong>={thr:.1e} trendgate", {"funding_block_long": thr, "funding_trend_gate": True}, True)

print("\n== block LONG when funding high, NO trend gate (naive fade) ==")
for thr in (1.5e-4, 2.5e-4, 4.0e-4):
    line(f"blockLong>={thr:.1e} notrend", {"funding_block_long": thr, "funding_trend_gate": False}, True)

print("\n== block SHORT when funding low AND uptrend (don't short capitulation) ==")
for thr in (0.0, 1.0e-5, 2.0e-5):
    line(f"blockShort<={thr:.1e} trendgate", {"funding_block_short": thr, "funding_trend_gate": True}, True)

print("\n== combined: blockLong high+down AND blockShort low+up ==")
for lt, stv in ((1.5e-4, 0.0), (2.0e-4, 1.0e-5), (1.5e-4, 2.0e-5)):
    line(f"L>={lt:.1e} S<={stv:.1e}",
         {"funding_block_long": lt, "funding_block_short": stv, "funding_trend_gate": True}, True)

print("\n== SHORT-boost: ease short thresholds when funding high + downtrend ==")
for boost in (0.85, 0.7, 0.5):
    for bthr in (1.0e-4, 1.5e-4, 2.0e-4):
        line(f"shortBoost x{boost} @>= {bthr:.1e}",
             {"funding_short_boost": boost, "funding_boost_thr": bthr}, True)
