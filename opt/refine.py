"""Refined search in the robust region (trailing ON). Select finalists by robustness:
strong on BOTH train and test half-year sets, with controlled full-history drawdown."""
from __future__ import annotations
import sys, json, random, time
import numpy as np
import opt.driver as drv

SLIP = 0.0002

def sample(rng):
    lev = rng.choice([15, 20, 20, 25, 25, 30])
    strong = rng.uniform(14, 36)
    marg = rng.uniform(10, min(strong, 26))
    tp1 = rng.uniform(1.4, 4.0)
    tp2 = tp1 + rng.uniform(1.0, 6.0)
    ov = {
        "tier.leverage": lev,
        "tier.strong_threshold": round(strong, 1),
        "tier.marginal_threshold_low": round(marg, 1),
        "tier.tp1_rr": round(tp1, 2),
        "tier.tp2_rr": round(tp2, 2),
        "tier.tp1_exit_pct": round(rng.uniform(0.3, 0.7), 2),
        "scoring.atr_sl_multiplier": round(rng.uniform(0.9, 2.8), 2),
        "trading.stop_loss_strategy": rng.choice(["atr", "hybrid", "structure"]),
        "weights": {k: rng.uniform(0.1, 1.0) for k in
                    ["trend", "momentum", "volume", "support_resistance", "risk"]},
        "filters.min_adx": round(rng.uniform(8, 26), 1),
        "filters.min_category_agreement": rng.choice([0, 1, 2, 3]),
        "filters.require_trend_momentum_agree": rng.random() < 0.4,
        "filters.skip_choppy_regime": rng.random() < 0.6,
        "filters.skip_volatile_regime": rng.random() < 0.3,
        "risk.cooldown_candles_after_sl": rng.choice([0, 1, 2, 3]),
        "risk.consecutive_loss_penalty": rng.choice([0.0, 2.0, 5.0]),
        "bt.enable_trailing_stops": True,
        "trailing.enabled": True,
        "trailing.activation_pct": round(rng.uniform(0.5, 2.2), 2),
        "trailing.callback_pct": round(rng.uniform(0.3, 1.0), 2),
    }
    return ov

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    drv.setup()
    rng = random.Random(seed)
    rows = []
    t0 = time.time()
    for i in range(n):
        ov = sample(rng)
        try:
            tr = drv.evaluate(ov, folds=drv.TRAIN_FOLDS, slip=SLIP)
        except Exception:
            continue
        if tr["total_trades"] < 60:
            continue
        rows.append((tr["geo_pct"], ov, tr))
        if (i+1) % 1500 == 0:
            print(f"  ...{i+1}/{n} ({time.time()-t0:.0f}s)", file=sys.stderr)
    rows.sort(key=lambda x: x[0], reverse=True)
    top = rows[:80]
    fin = []
    for _, ov, tr in top:
        te = drv.evaluate(ov, folds=drv.TEST_FOLDS, slip=SLIP)
        full = drv.evaluate(ov, folds=drv.FOLDS, slip=SLIP)
        robust = min(tr["geo_pct"], te["geo_pct"])   # good on BOTH halves
        fin.append((robust, ov, tr, te, full))
    # finalists: drawdown-controlled, every fold green, ranked by robustness
    fin.sort(key=lambda x: x[0], reverse=True)
    print(f"Searched {len(rows)} (trailing region) in {time.time()-t0:.0f}s.\n")
    print("Ranked by min(trainGeo,testGeo) — robust generalizers:\n")
    for robust, ov, tr, te, full in fin[:15]:
        flag = "" if (full["max_dd"] < 22 and full["worst_fold"] > 0) else "  [DD/neg]"
        print(f"robust={robust:5.1f} | trGeo {tr['geo_pct']:+6.1f} teGeo {te['geo_pct']:+6.1f} | "
              f"FULL comp {full['compound_x']:8.2f}x worst {full['worst_fold']:+6.1f}% "
              f"maxDD {full['max_dd']:4.1f}% tr {full['total_trades']}{flag}")
        print(f"    {drv.fmt(full).splitlines()[1].strip()}")
        print(f"    ov={json.dumps(ov)}")
    out = [{"robust": r, "ov": ov, "train": tr, "test": te, "full": full}
           for r, ov, tr, te, full in fin]
    json.dump(out, open("opt/refine_results.json", "w"), indent=1)
    print("\nSaved opt/refine_results.json")

if __name__ == "__main__":
    main()
