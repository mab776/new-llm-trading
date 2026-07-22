"""Sim gate for the aggressive live deploy: reproduce canonical anchors on the
exact aggressive config chain + $100-with-mins expectation + per-asset split."""
import sys
import opt.probe_reserved as pr
from opt.multi_asset import simulate_multi
from opt.probe_reserved import MIN_QTY, SIZE_STEP, _load, _with_balance

SLIP = 0.0002
HOLD_START, HOLD_END = "2025-06-01", "2026-04-30"

# Point the house loader at the aggressive chain (same symbols).
pr.CONFIGS = {"BTC": "config-aggressive.json", "ETH": "config-eth-aggressive.json",
              "SOL": "config-sol-aggressive.json"}
assets = {l: _load(l) for l in pr.CONFIGS}

def run(assets, bal, mins, label):
    _with_balance(assets, bal)
    strat = {"min_qty": MIN_QTY, "size_step": SIZE_STEP} if mins else None
    r = simulate_multi(assets, HOLD_START, HOLD_END, slip=SLIP,
                       exit_granularity="sub", strat=strat)
    mult = 1 + r.return_pct / 100
    print(f"{label:>34}: {mult:8.2f}x  maxDD {r.max_dd_pct:5.1f}%  trades {r.trades}",
          flush=True)
    return mult

print("== aggressive holdout gate (2025-06 -> 2026-04) ==")
run(assets, 3000.0, False, "portfolio frictionless (anchor)")
run(assets, 100.0, True,  "portfolio $100 + real mins")
print("-- per-asset standalone (frictionless) --")
for sym in ("BTC", "ETH", "SOL"):
    solo = {sym: assets[sym]}
    run(solo, 3000.0, False, f"{sym} standalone")
