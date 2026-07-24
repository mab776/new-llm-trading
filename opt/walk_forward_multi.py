"""MULTI-ASSET walk-forward retuning study — the pre-deploy evidence run.

Marc (2026-07-24): "I want the multiasset walk test in full — I'm really
interested to deploy." Extends the Round-20 BTC-only study (and its 1w-base
rerun, f0dda72) to the real deliverable: the BTC+ETH+SOL shared-portfolio
sim, the same thing the live bots run.

Changes vs the BTC study, all pre-declared:
  * EXTENDED SEARCH SPACE (Marc's asks):
      - trailing callback_pct now sampled 0.20-0.50 (was 0.25-0.50 — all nine
        1000-trial winners sat ON the old 0.25 floor);
      - NEW recently-deployed variables join the search: the alignment vote
        weights {1h: 0-2, 1d: 0-5, 1w: 0-4} (integers; deployed static values
        are 0/3/2). Candidates pass them via strat alignment_scale_by_tf;
        the static arm keeps the shipped config values untouched.
    (Loss-penalty params, min-size rescue and maker knobs stay OUT — execution
    machinery, not strategy search.)
  * FOUR deployment windows: 2023, 2024, 2025H1 (annual convention) + the
    HOLDOUT window 2025-06-01..2026-04-30 (train = trailing two years,
    2023-06..2025-06). Chained unseen span = contiguous 2023-01..2026-04.
  * Portfolio sims: probe_btc_delay conventions — maker entry, honest sub-bar
    exits, funding, 2bps slip, frictionless (mins are an execution question).
  * Train eval = the window's two one-year folds simulated separately
    (keeps the green-folds gate); objective = fold-geo - 0.35 * worst-fold-DD;
    gates: every train fold > 0, maxDD <= 45, >= 300 trades (2x the BTC gate —
    3 assets trade more).
  * Selection on train only; the unseen window never participates. Static
    config ({}) is always candidate #0. No adoption from this script — deploy
    remains Marc's call with the full protocol.
  * Parallel: fork pool over candidates (assets loaded once, COW-shared).

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.walk_forward_multi \
        [--tiers 300:17,73,211,419,887 1000:17,73,211] [--procs 12]
Calibrate: --calibrate  (times one static eval per window, prints ETA, exits)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from multiprocessing import Pool

from opt.multi_asset import AssetInput, simulate_multi
from opt.probe_reserved import CONFIGS, _load

SLIP = 0.0002

WINDOWS = [
    ("2023", [("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31")],
     ("2023-01-01", "2023-12-31")),
    ("2024", [("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31")],
     ("2024-01-01", "2024-12-31")),
    ("2025H1", [("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")],
     ("2025-01-01", "2025-06-01")),
    ("HOLDOUT", [("2023-06-01", "2024-06-01"), ("2024-06-01", "2025-06-01")],
     ("2025-06-01", "2026-04-30")),
]

ASSETS: dict[str, AssetInput] = {}


def sample_candidate(rng: random.Random) -> dict:
    """Round-20 neighborhood + extended callback floor + alignment weights."""
    strong = rng.uniform(18.0, 27.0)
    marginal = rng.uniform(10.0, min(17.0, strong - 2.0))
    weights = {
        "trend": rng.uniform(.15, .29),
        "momentum": rng.uniform(.24, .39),
        "volume": rng.uniform(.14, .27),
        "support_resistance": rng.uniform(.11, .23),
        "risk": rng.uniform(.06, .15),
    }
    return {
        "tier.strong_threshold": round(strong, 2),
        "tier.marginal_threshold_low": round(marginal, 2),
        "tier.tp1_rr": round(rng.uniform(1.65, 2.45), 2),
        "tier.tp2_rr": round(rng.uniform(2.7, 4.1), 2),
        "tier.tp1_exit_pct": round(rng.uniform(.55, .80), 2),
        "scoring.atr_sl_multiplier": round(rng.uniform(1.8, 2.8), 2),
        "filters.min_adx": round(rng.uniform(17.0, 24.0), 1),
        "trailing.activation_pct": round(rng.uniform(.70, 1.25), 2),
        "trailing.callback_pct": round(rng.uniform(.20, .50), 2),  # floor 0.20
        "weights": weights,
        "align.1h": rng.randint(0, 2),
        "align.1d": rng.randint(0, 5),
        "align.1w": rng.randint(0, 4),
    }


def _apply(cand: dict) -> tuple[dict, dict | None]:
    """Candidate -> (per-label override configs, alignment strat dict|None)."""
    out = {}
    align = None
    if cand:
        align = {"1h": cand["align.1h"], "1d": cand["align.1d"],
                 "1w": cand["align.1w"]}
    for label, item in ASSETS.items():
        cfg = item.config.model_copy(deep=True)
        if cand:
            tier = cfg.trading.leverage_tiers[cfg.trading.active_tier]
            tier.strong_threshold = cand["tier.strong_threshold"]
            tier.marginal_threshold_low = cand["tier.marginal_threshold_low"]
            tier.tp1_rr = cand["tier.tp1_rr"]
            tier.tp2_rr = cand["tier.tp2_rr"]
            tier.tp1_exit_pct = cand["tier.tp1_exit_pct"]
            cfg.scoring.atr_sl_multiplier = cand["scoring.atr_sl_multiplier"]
            cfg.filters.min_adx = cand["filters.min_adx"]
            cfg.trading.trailing_stop.activation_pct = cand["trailing.activation_pct"]
            cfg.trading.trailing_stop.callback_pct = cand["trailing.callback_pct"]
            tot = sum(cand["weights"].values())
            cfg.scoring.weights = {k: v / tot for k, v in cand["weights"].items()}
        cfg.backtesting.initial_balance = 3000.0
        out[label] = cfg
    return out, align


def eval_candidate(args) -> tuple:
    """(window_i, cand_json, folds) -> per-fold [ret, dd, trades]."""
    window_i, cand_json, folds = args
    cand = json.loads(cand_json)
    cfgs, align = _apply(cand)
    assets = {lab: AssetInput(ASSETS[lab].pre, cfgs[lab],
                              ASSETS[lab].funding_by_pos) for lab in ASSETS}
    strat = {"alignment_scale_by_tf": align} if align else None
    rows = []
    for start, end in folds:
        res = simulate_multi(assets, start, end, slip=SLIP,
                             exit_granularity="sub", strat=strat)
        rows.append((res.return_pct, res.max_dd_pct, res.trades))
    return window_i, cand_json, rows


def objective(rows: list) -> float:
    rets = [r for r, _, _ in rows]
    if min(rets) <= 0 or max(d for _, d, _ in rows) > 45:
        return -1e12
    if sum(t for _, _, t in rows) < 300:
        return -1e12
    geo = (math.prod(1 + r / 100 for r in rets) ** (1 / len(rets)) - 1) * 100
    return geo - .35 * max(d for _, d, _ in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+",
                    default=["300:17,73,211,419,887", "1000:17,73,211"])
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--output", default="opt/walk_forward_multi_results.json")
    args = ap.parse_args()
    tiers = [(int(t.split(":")[0]),
              [int(s) for s in t.split(":")[1].split(",")])
             for t in args.tiers]

    print("Loading assets...", file=sys.stderr, flush=True)
    global ASSETS
    ASSETS = {label: _load(label) for label in CONFIGS}

    if args.calibrate:
        for wi, (name, train, _) in enumerate(WINDOWS):
            t0 = time.time()
            _, _, rows = eval_candidate((wi, "{}", train))
            print(f"{name}: train eval {time.time()-t0:.1f}s rows={rows}")
        return

    # Distinct (window, candidate) evaluation cache; 300-seq is a prefix of
    # the 1000-seq for the same seed (same rng stream), so tiers share work.
    cache: dict[tuple[int, str], list] = {}
    results = {"windows": [w[0] for w in WINDOWS], "runs": []}
    with Pool(processes=args.procs) as pool:
        for trials, seeds in tiers:
            for seed in seeds:
                run_rows = []
                for wi, (name, train, target) in enumerate(WINDOWS):
                    rng = random.Random(seed + wi)
                    cands = ["{}"] + [
                        json.dumps(sample_candidate(rng), sort_keys=True)
                        for _ in range(trials)]
                    todo = [(wi, cj, train) for cj in dict.fromkeys(cands)
                            if (wi, cj) not in cache]
                    for w_i, cj, rows in pool.imap_unordered(
                            eval_candidate, todo, chunksize=4):
                        cache[(w_i, cj)] = rows
                    best_cj = max(cands, key=lambda cj: objective(cache[(wi, cj)]))
                    # unseen deployment (winner + static, cached under target key)
                    dep = {}
                    for tag, cj in (("tuned", best_cj), ("static", "{}")):
                        key = (wi + 100, cj)
                        if key not in cache:
                            cache[key] = eval_candidate((wi, cj, [target]))[2]
                        dep[tag] = cache[key][0]
                    run_rows.append({
                        "window": name, "winner": json.loads(best_cj),
                        "tuned": dep["tuned"], "static": dep["static"]})
                    print(f"t={trials} seed={seed} {name}: "
                          f"tuned {dep['tuned'][0]:+.1f}% dd {dep['tuned'][1]:.1f} "
                          f"vs static {dep['static'][0]:+.1f}% dd {dep['static'][1]:.1f}",
                          flush=True)
                tx = math.prod(1 + r["tuned"][0] / 100 for r in run_rows)
                sx = math.prod(1 + r["static"][0] / 100 for r in run_rows)
                results["runs"].append({
                    "trials": trials, "seed": seed, "windows": run_rows,
                    "tuned_x": tx, "static_x": sx,
                    "ratio": tx / sx if sx else 0})
                print(f"== t={trials} seed={seed}: chained tuned {tx:.2f}x "
                      f"vs static {sx:.2f}x ratio {tx/sx:.3f}", flush=True)

    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    ratios = sorted(r["ratio"] for r in results["runs"])
    print(f"\nsaved {args.output} | ratios: "
          + " ".join(f"{r:.3f}" for r in ratios))


if __name__ == "__main__":
    main()
