"""Walk-forward retuning deployed on the CLEAN HOLDOUT window (BTC, single-asset).

Answers "what does the walk-forward process do on the TEST sample?" — the
Round-20 study's deployment windows stop at 2025-06-01; this adds the one
window it never touched: train on the trailing two years (2023-06..2025-06,
mid-year offset = the trailing-two-year convention applied to the holdout
start), deploy the single winner on 2025-06-01..2026-04-30.

Protocol-clean: the holdout NEVER participates in search or ranking — one
deployment evaluation per seed, same discipline as every other unseen window.
Pre-declared: rng = Random(seed + 3) (the study's window_i convention, next
window index); seeds/tiers mirror the study (5x300, 3x1000); no adoption.

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.wf_holdout_deploy
"""
from __future__ import annotations

import json
import random

import opt.driver as drv
from opt.walk_forward_retune import sample_candidate, objective

TRAIN = [("2023/24", "2023-06-01", "2024-06-01"),
         ("2024/25", "2024-06-01", "2025-06-01")]
TARGET = [("HOLDOUT", "2025-06-01", "2026-04-30")]
STRAT = {"entry_mode": "maker"}
_cache: dict = {}


def ev(overrides: dict, folds) -> dict:
    key = (json.dumps(overrides, sort_keys=True), tuple(f[0] for f in folds))
    if key not in _cache:
        _cache[key] = drv.evaluate(overrides, folds=folds, slip=2e-4,
                                   funding=True, strat=STRAT,
                                   exit_granularity="sub")
    return _cache[key]


def main() -> None:
    ctx = drv.load_context(None, data_start="2020-10-01",
                           data_end="2026-04-30", funding_end="2026-05-01")
    drv._PRE, drv._BASE, drv._FUND, drv._FMETRIC = (
        ctx.pre, ctx.config, ctx.funding, ctx.funding_metric)

    static = ev({}, TARGET)
    print(f"static  HOLDOUT: {static['mean_ret']:+7.1f}%  "
          f"dd {static['max_dd']:4.1f}%  tr {static['total_trades']}", flush=True)
    sf = 1 + static["mean_ret"] / 100

    for trials, seeds in ((300, (17, 73, 211, 419, 887)), (1000, (17, 73, 211))):
        ratios = []
        for seed in seeds:
            rng = random.Random(seed + 3)
            cands = [{}] + [sample_candidate(rng) for _ in range(trials)]
            best, winner = None, {}
            for ov in cands:
                score = objective(ev(ov, TRAIN))
                if best is None or score > best:
                    best, winner = score, ov
            u = ev(winner, TARGET)
            r = (1 + u["mean_ret"] / 100) / sf
            ratios.append(r)
            print(f"t={trials:4d} seed={seed:3d}: tuned {u['mean_ret']:+7.1f}%  "
                  f"dd {u['max_dd']:4.1f}%  tr {u['total_trades']}  ratio {r:.3f}",
                  flush=True)
        mid = sorted(ratios)[len(ratios) // 2]
        print(f"  -> t={trials}: median ratio {mid:.3f}, "
              f"beat static {sum(r > 1 for r in ratios)}/{len(ratios)}", flush=True)


if __name__ == "__main__":
    main()
