"""Ablation + candidate evaluation across train/test/reverse splits and full yearly folds.
Isolates which levers carry the out-of-sample edge."""
from __future__ import annotations
import json
import numpy as np
import opt.driver as drv

SLIP = 0.0002
REV_TRAIN = drv.TEST_FOLDS      # reverse split: train on even halves
REV_TEST = drv.TRAIN_FOLDS

def g(ov, folds):
    r = drv.evaluate(ov, folds=folds, slip=SLIP)
    return r

def line(tag, ov):
    tr = g(ov, drv.TRAIN_FOLDS); te = g(ov, drv.TEST_FOLDS)
    rtr = g(ov, REV_TRAIN); rte = g(ov, REV_TEST)
    full = g(ov, drv.FOLDS)
    print(f"{tag:28s} | trainGeo {tr['geo_pct']:+6.1f} testGeo {te['geo_pct']:+6.1f} "
          f"| revTr {rtr['geo_pct']:+6.1f} revTe {rte['geo_pct']:+6.1f} "
          f"| FULL comp {full['compound_x']:8.2f}x worst {full['worst_fold']:+6.1f}% "
          f"maxDD {full['max_dd']:4.1f}% tr {full['total_trades']}")
    return full

def main():
    drv.setup()
    print("=== ABLATION (each row = baseline + ONE change) ===")
    line("baseline", {})
    line("+trailing(1.0/0.5)", {"bt.enable_trailing_stops": True, "trailing.enabled": True,
                                 "trailing.activation_pct": 1.0, "trailing.callback_pct": 0.5})
    line("+trailing(1.0/0.4)", {"bt.enable_trailing_stops": True, "trailing.enabled": True,
                                 "trailing.activation_pct": 1.0, "trailing.callback_pct": 0.4})
    line("leverage=25", {"tier.leverage": 25})
    line("leverage=30", {"tier.leverage": 30})
    line("strong=20,marg=13", {"tier.strong_threshold": 20, "tier.marginal_threshold_low": 13})
    line("tp1exit=0.6", {"tier.tp1_exit_pct": 0.6})
    line("no_tm_agree", {"filters.require_trend_momentum_agree": False})
    line("min_agree=1", {"filters.min_category_agreement": 1})
    line("cooldown=0", {"risk.cooldown_candles_after_sl": 0})
    line("loss_pen=0", {"risk.consecutive_loss_penalty": 0.0})

    print("\n=== TOP GENERALIZING CANDIDATES from wf_results ===")
    wf = json.load(open("opt/wf_results.json"))
    for i, e in enumerate(wf[:6]):
        line(f"wf#{i}", e["ov"])

if __name__ == "__main__":
    main()
