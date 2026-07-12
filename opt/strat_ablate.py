"""Strategy-variant ablation on top of the optimized config.json.
Each row = new config + ONE structural strategy change, train/test half-year validated."""
from __future__ import annotations
import json
import opt.driver as drv

SLIP = 0.0002

def line(tag, strat=None, ov=None):
    ov = ov or {}
    tr = drv.evaluate(ov, folds=drv.TRAIN_FOLDS, slip=SLIP, strat=strat)
    te = drv.evaluate(ov, folds=drv.TEST_FOLDS, slip=SLIP, strat=strat)
    full = drv.evaluate(ov, folds=drv.FOLDS, slip=SLIP, strat=strat)
    print(f"{tag:36s} | tr {tr['geo_pct']:+6.1f} te {te['geo_pct']:+6.1f} "
          f"| FULL {full['compound_x']:8.2f}x worst {full['worst_fold']:+6.1f}% "
          f"DD {full['max_dd']:4.1f}% n={full['total_trades']}")
    return full

def main():
    drv.setup()
    print("Each row = optimized config + ONE strategy change (2bps slip):\n")
    line("BASE (new config.json)")
    print()
    line("vol_target_lev=30", {"vol_target_lev": 30.0})
    line("vol_target_lev=40", {"vol_target_lev": 40.0})
    line("vol_target_lev=50", {"vol_target_lev": 50.0})
    print()
    line("trail=atr 0.4/0.5", {"trail_mode": "atr", "trail_act_atr": 0.4, "trail_cb_atr": 0.5})
    line("trail=atr 0.5/0.35", {"trail_mode": "atr", "trail_act_atr": 0.5, "trail_cb_atr": 0.35})
    line("trail=atr 0.8/0.5", {"trail_mode": "atr", "trail_act_atr": 0.8, "trail_cb_atr": 0.5})
    line("trail=atr 0.3/0.3", {"trail_mode": "atr", "trail_act_atr": 0.3, "trail_cb_atr": 0.3})
    print()
    line("conviction_sizing k=1", {"conviction_sizing": 1.0})
    line("conviction_sizing k=2", {"conviction_sizing": 2.0})
    line("marginal_size=0.5", {"marginal_size_frac": 0.5})
    print()
    line("opposite_exit=20", {"opposite_exit": 20.0})
    line("opposite_exit=30", {"opposite_exit": 30.0})
    line("opposite_exit=40", {"opposite_exit": 40.0})
    print()
    line("shorts stricter x1.3", {"short_threshold_mult": 1.3})
    line("shorts stricter x1.6", {"short_threshold_mult": 1.6})
    line("longs easier x0.8", {"long_threshold_mult": 0.8})
    print()
    line("max_positions=2", {"max_positions": 2})
    line("max_positions=3", {"max_positions": 3})

if __name__ == "__main__":
    main()
