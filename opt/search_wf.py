"""Walk-forward random search: optimize on TRAIN half-years, validate on held-out TEST
half-years. Slippage baked in so pure-churn edges are penalized. The number that matters
is TEST performance of configs selected purely by TRAIN rank."""
from __future__ import annotations

import sys, json, random, time
import numpy as np

import opt.driver as drv
from opt.search import sample

SLIP = 0.0002  # 2 bps per market side (conservative for a small account on liquid BTC-perp)


def obj(res, min_trades):
    if res["total_trades"] < min_trades:
        return -1e9
    if res["worst_fold"] < -40:
        return -1e9
    if res["max_dd"] > 60:
        return -1e9
    return res["geo_pct"]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    drv.setup()
    rng = random.Random(seed)

    base_tr = drv.evaluate({}, folds=drv.TRAIN_FOLDS, slip=SLIP)
    base_te = drv.evaluate({}, folds=drv.TEST_FOLDS, slip=SLIP)
    print(f"BASELINE  train geo={base_tr['geo_pct']:+.2f}% comp={base_tr['compound_x']:.2f}x"
          f" | test geo={base_te['geo_pct']:+.2f}% comp={base_te['compound_x']:.2f}x\n")

    rows = []
    t0 = time.time()
    for i in range(n):
        ov = sample(rng)
        try:
            tr = drv.evaluate(ov, folds=drv.TRAIN_FOLDS, slip=SLIP)
        except Exception:
            continue
        rows.append((obj(tr, 70), ov, tr))
        if (i + 1) % 1000 == 0:
            print(f"  ...{i+1}/{n} ({time.time()-t0:.0f}s)", file=sys.stderr)

    rows.sort(key=lambda x: x[0], reverse=True)
    top = rows[:40]
    # Evaluate the train-top-40 on the held-out test folds
    enriched = []
    for tro, ov, tr in top:
        te = drv.evaluate(ov, folds=drv.TEST_FOLDS, slip=SLIP)
        enriched.append((tro, te, ov, tr))

    # Rank the train-selected cohort by TEST objective (unbiased)
    enriched.sort(key=lambda x: obj(x[1], 50), reverse=True)
    print(f"Searched {len(rows)} in {time.time()-t0:.0f}s. Train-top-40 ranked by TEST geo:\n")
    test_geos = [e[1]["geo_pct"] for e in enriched]
    print(f"Cohort TEST geo: median={np.median(test_geos):.1f}%  best={max(test_geos):.1f}%  "
          f"worst={min(test_geos):.1f}%  (baseline test {base_te['geo_pct']:.1f}%)\n")
    for tro, te, ov, tr in enriched[:15]:
        print(f"TRAIN geo={tr['geo_pct']:+7.1f}% comp={tr['compound_x']:7.2f}x | "
              f"TEST geo={te['geo_pct']:+7.1f}% comp={te['compound_x']:6.2f}x worst={te['worst_fold']:+.0f}% "
              f"maxDD={te['max_dd']:.0f}% tr={te['total_trades']}")
        print(f"    ov={json.dumps(ov)}")

    out = [{"train": tr, "test": te, "ov": ov} for tro, te, ov, tr in enriched]
    with open("opt/wf_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nSaved to opt/wf_results.json")


if __name__ == "__main__":
    main()
