"""Retune-cadence sweep for multi-asset walk-forward (Marc, 2026-07-24 night).

"Maybe it should be run pretty often? weekly or monthly — annually is
extremely large." Cost is a non-issue (~0.6s per 2-yr 3-asset eval); the open
question is whether faster retuning HELPS. This sweep answers it on the same
chained OOS span as the annual study (2023-01-01..2026-04-30):

  cadence arms: 12M (annual, comparability arm), 3M (quarterly), 1M (monthly)
  For each deployment slice: train = trailing 2 years split into two 1-year
  folds (same green-folds gate + geo - 0.35*worstDD objective as
  walk_forward_multi), select on train only, deploy the winner on the slice.
  Static = untouched deployed configs on the same slices.

PRE-DECLARED:
  * rng per slice = Random(seed * 1000 + slice_index); seeds 17, 73, 211;
    300 trials/slice (the annual study's medians moved little 300->1000).
  * Same extended space as walk_forward_multi (callback >= 0.20, alignment
    weights searched).
  * Churn metric: fraction of consecutive slices whose winner changed, plus
    mean L1 distance over the numeric knobs (normalized by range widths).
  * ESCALATION RULE (pre-committed): weekly (1W) runs ONLY if monthly beats
    quarterly on median chained ratio — otherwise it is dominated and we
    don't burn folds on it.
  * No adoption from this script.

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.walk_forward_cadence [--procs 12]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from multiprocessing import Pool

from dateutil.relativedelta import relativedelta

import opt.walk_forward_multi as wfm

SPAN_START = datetime(2023, 1, 1)
SPAN_END = datetime(2026, 4, 30)
CADENCES = {"12M": {"months": 12}, "3M": {"months": 3}, "1M": {"months": 1},
            "1W": {"weeks": 1}}
SEEDS = (17, 73, 211)
TRIALS = 300

RANGES = {  # normalization widths for the churn metric
    "tier.strong_threshold": 9.0, "tier.marginal_threshold_low": 7.0,
    "tier.tp1_rr": .8, "tier.tp2_rr": 1.4, "tier.tp1_exit_pct": .25,
    "scoring.atr_sl_multiplier": 1.0, "filters.min_adx": 7.0,
    "trailing.activation_pct": .55, "trailing.callback_pct": .30,
    "align.1h": 2.0, "align.1d": 5.0, "align.1w": 4.0,
}


def slices(step: dict) -> list[tuple[str, str]]:
    out, cur = [], SPAN_START
    while cur < SPAN_END:
        nxt = min(cur + relativedelta(**step), SPAN_END)
        out.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return out


def train_folds(slice_start: str) -> list[tuple[str, str]]:
    s = datetime.fromisoformat(slice_start)
    return [((s - relativedelta(years=2)).strftime("%Y-%m-%d"),
             (s - relativedelta(years=1)).strftime("%Y-%m-%d")),
            ((s - relativedelta(years=1)).strftime("%Y-%m-%d"), slice_start)]


def churn(a: dict, b: dict) -> float:
    if not a or not b:
        return 1.0
    return sum(abs(float(a[k]) - float(b[k])) / w
               for k, w in RANGES.items()) / len(RANGES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--cadences", nargs="+", default=list(CADENCES))
    ap.add_argument("--output", default="opt/walk_forward_cadence_results.json")
    args = ap.parse_args()

    print("Loading assets...", file=sys.stderr, flush=True)
    wfm.ASSETS = {label: wfm._load(label) for label in wfm.CONFIGS}

    cache: dict[tuple, list] = {}
    out = {"span": [SPAN_START.isoformat(), SPAN_END.isoformat()],
           "trials": TRIALS, "runs": []}

    with Pool(processes=args.procs) as pool:
        for cad in args.cadences:
            sl = slices(CADENCES[cad])
            # static per slice (shared by all seeds)
            for s in sl:
                key = ("dep", s, "{}")
                if key not in cache:
                    cache[key] = wfm.eval_candidate((0, "{}", [s]))[2]
            static_x = math.prod(1 + cache[("dep", s, "{}")][0][0] / 100
                                 for s in sl)
            for seed in SEEDS:
                prev, churns, changed = {}, [], 0
                tuned_rets = []
                for i, s in enumerate(sl):
                    tf = train_folds(s[0])
                    rng = random.Random(seed * 1000 + i)
                    cands = ["{}"] + [
                        json.dumps(wfm.sample_candidate(rng), sort_keys=True)
                        for _ in range(TRIALS)]
                    todo = [(0, cj, tf) for cj in dict.fromkeys(cands)
                            if ("tr", tf[0][0], cj) not in cache]
                    for _, cj, rows in pool.imap_unordered(
                            wfm.eval_candidate, todo, chunksize=4):
                        cache[("tr", tf[0][0], cj)] = rows
                    best = max(cands,
                               key=lambda cj: wfm.objective(cache[("tr", tf[0][0], cj)]))
                    dkey = ("dep", s, best)
                    if dkey not in cache:
                        cache[dkey] = wfm.eval_candidate((0, best, [s]))[2]
                    tuned_rets.append(cache[dkey][0])
                    w = json.loads(best)
                    if i:
                        churns.append(churn(prev, w))
                        changed += bool(best != prev_j)
                    prev, prev_j = w, best
                tx = math.prod(1 + r[0] / 100 for r in tuned_rets)
                dd = max(r[1] for r in tuned_rets)
                sdd = max(cache[("dep", s, "{}")][0][1] for s in sl)
                run = {"cadence": cad, "seed": seed, "slices": len(sl),
                       "tuned_x": tx, "static_x": static_x,
                       "ratio": tx / static_x, "worst_slice_dd": dd,
                       "static_worst_slice_dd": sdd,
                       "winner_changed_frac": changed / max(1, len(sl) - 1),
                       "mean_churn": (sum(churns) / len(churns)) if churns else 0}
                out["runs"].append(run)
                print(f"{cad} seed={seed}: tuned {tx:9.1f}x vs static "
                      f"{static_x:8.1f}x ratio {tx/static_x:6.3f} | "
                      f"worstDD {dd:.1f} vs {sdd:.1f} | winner-change "
                      f"{run['winner_changed_frac']:.0%} churn {run['mean_churn']:.2f}",
                      flush=True)

    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"saved {args.output}")
    for cad in args.cadences:
        rr = sorted(r["ratio"] for r in out["runs"] if r["cadence"] == cad)
        print(f"  {cad}: median ratio {rr[len(rr)//2]:.3f}  ({', '.join(f'{r:.2f}' for r in rr)})")


if __name__ == "__main__":
    main()
