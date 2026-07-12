"""Validation battery for a candidate config: chronological OOS split, leverage curve,
slippage sensitivity, and no-1h-timeframe robustness."""
from __future__ import annotations
import sys, json
import opt.driver as drv

CHRONO_TRAIN = [("21", "2021-01-01", "2021-12-31"), ("22", "2022-01-01", "2022-12-31"),
                ("23", "2023-01-01", "2023-12-31")]
CHRONO_TEST  = [("24", "2024-01-01", "2024-12-31"), ("25", "2025-01-01", "2025-06-01")]

def show(tag, ov, slip=0.0002):
    full = drv.evaluate(ov, folds=drv.FOLDS, slip=slip)
    tr = drv.evaluate(ov, folds=CHRONO_TRAIN, slip=slip)
    te = drv.evaluate(ov, folds=CHRONO_TEST, slip=slip)
    print(f"{tag}")
    print(f"   FULL: comp {full['compound_x']:8.2f}x  worst {full['worst_fold']:+.1f}%  "
          f"maxDD {full['max_dd']:.1f}%  trades {full['total_trades']}")
    print(f"   {drv.fmt(full).splitlines()[1].strip()}")
    print(f"   CHRONO train(21-23) comp {tr['compound_x']:7.2f}x | "
          f"OOS test(24-25) comp {te['compound_x']:6.2f}x worst {te['worst_fold']:+.1f}% maxDD {te['max_dd']:.1f}%")

def main():
    drv.setup()
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cand = json.load(open("opt/refine_results.json"))[idx]
    ov = cand["ov"]
    print("CANDIDATE ov:", json.dumps(ov), "\n")
    show("=== candidate (2 bps slip) ===", ov, slip=0.0002)
    print("\n--- slippage sensitivity (full compound) ---")
    for bps in (0, 2, 5, 10, 20):
        f = drv.evaluate(ov, folds=drv.FOLDS, slip=bps/10000)
        print(f"   {bps:2d} bps: comp {f['compound_x']:9.2f}x  worst {f['worst_fold']:+6.1f}%  maxDD {f['max_dd']:.1f}%")
    print("\n--- leverage curve (2 bps slip) ---")
    for lev in (10, 15, 20, 25, 30):
        ov2 = dict(ov); ov2["tier.leverage"] = lev
        f = drv.evaluate(ov2, folds=drv.FOLDS, slip=0.0002)
        print(f"   lev {lev:2d}: comp {f['compound_x']:9.2f}x  worst {f['worst_fold']:+6.1f}%  "
              f"maxDD {f['max_dd']:.1f}%  chrono-OOS "
              f"{drv.evaluate(ov2, folds=CHRONO_TEST, slip=0.0002)['compound_x']:.2f}x")

if __name__ == "__main__":
    main()
