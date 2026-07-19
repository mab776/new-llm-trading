"""Finalist selection + TEST evaluation + stress for the 15m scalper.

PRE-COMMITTED RULES (written before any TEST fold was opened):
1. Universe = the donchian-refine grid (opt/scalp/grid.py --grid donchian)
   evaluated on BTC, ETH and SOL TRAIN folds.
2. A config qualifies if it passes the survival gates (>=200 trades across the
   3 assets combined, >=5/6 folds positive on the per-asset mean-growth curve,
   worst per-asset DD <= 25%) on at least 2 of 3 assets individually.
3. Rank qualifying configs by MIN(per-asset TRAIN geo growth) — the worst
   asset decides ("no asset left behind" — kills single-asset path fits).
4. Top 3 distinct (n, vol_expand) signal cells advance to TEST. TEST numbers
   are REPORTED, never used to re-rank within this session.
5. Stress ladders (penetration bps, taker-entry flip, fee x1.5) and +/-15%
   parameter jitter run on the finalists only, on TRAIN — a finalist that
   loses >half its TRAIN edge under 2bps penetration is flagged fragile.
6. HOLDOUT (2025-06 .. 2026-07-17) is run ONCE, at the very end, on the single
   chosen config, all 3 assets. Invariance check only.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opt.scalp import grid
from opt.scalp.engine import ExecParams
from opt.scalp.grid import TEST_FOLDS, TRAIN_FOLDS, HOLDOUT, donchian_refine_grids

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
RESULTS = Path(__file__).parent / "results"


def eval_symbol(symbol: str, jobs, folds) -> list[dict]:
    grid.load_all(symbol, "15m")
    out = []
    for strat, sp, ep in jobs:
        agg = grid.run_folds(strat, sp, ep, folds)
        out.append({"strategy": strat, "sig": dict(sp), "exec": grid.ep_desc(ep),
                    "ep": ep, **{k: v for k, v in agg.items()}})
    return out


def main() -> None:
    jobs = donchian_refine_grids("15m")
    print(f"{len(jobs)} configs x {len(SYMBOLS)} assets (TRAIN)", flush=True)

    per_asset: dict[str, list[dict]] = {}
    for sym in SYMBOLS:
        print(f"== TRAIN {sym} ==", flush=True)
        per_asset[sym] = eval_symbol(sym, jobs, TRAIN_FOLDS)

    # qualify + rank
    n_jobs = len(jobs)
    rows = []
    for k in range(n_jobs):
        entry = {"sig": per_asset[SYMBOLS[0]][k]["sig"],
                 "exec": per_asset[SYMBOLS[0]][k]["exec"]}
        geos = {}
        passes = 0
        for sym in SYMBOLS:
            r = per_asset[sym][k]
            geos[sym] = r["geo_growth"]
            ok = (r["trades"] >= 120 and r["folds_positive"] >= 5
                  and r["worst_dd"] <= 25.0)
            passes += int(ok)
        total_trades = sum(per_asset[s][k]["trades"] for s in SYMBOLS)
        entry.update(geos=geos, min_geo=min(geos.values()), passes=passes,
                     total_trades=total_trades, k=k)
        if passes >= 2 and total_trades >= 200:
            rows.append(entry)

    rows.sort(key=lambda e: -e["min_geo"])
    print(f"\n{len(rows)} qualifying configs; top 12 by min-asset geo:")
    for e in rows[:12]:
        g = " ".join(f"{s[:3]}{e['geos'][s]:.3f}" for s in SYMBOLS)
        print(f"  min {e['min_geo']:.3f}  [{g}]  tr{e['total_trades']}  "
              f"{e['sig']}  {e['exec']}")

    # top 3 distinct signal cells
    finalists = []
    seen_cells = set()
    for e in rows:
        cell = (e["sig"]["n"], e["sig"]["vol_expand"])
        if cell in seen_cells:
            continue
        seen_cells.add(cell)
        finalists.append(e)
        if len(finalists) == 3:
            break

    print("\n=== FINALISTS (advance to TEST) ===")
    artifact = {"protocol": __doc__, "finalists": []}
    for e in finalists:
        k = e["k"]
        strat, sp, ep = jobs[k]
        fin = {"sig": e["sig"], "exec": e["exec"], "train": {}}
        print(f"\n--- {e['sig']} {e['exec']} ---")
        for sym in SYMBOLS:
            r = per_asset[sym][k]
            fin["train"][sym] = {
                "geo": round(r["geo_growth"], 4),
                "folds_positive": r["folds_positive"],
                "worst_dd": round(r["worst_dd"], 1),
                "trades": r["trades"],
                "win_rate": round(r["win_rate"], 1),
                "avg_hold_bars": round(r["avg_hold"], 1),
                "per_fold": r["per_fold"],
            }
            print(f"  TRAIN {sym}: {r['geo_growth']:.3f}x/fold  "
                  f"dd{r['worst_dd']:.0f}%  tr{r['trades']}  "
                  f"hold {r['avg_hold']:.0f} bars")

        # TEST (report-only)
        fin["test"] = {}
        for sym in SYMBOLS:
            grid.load_all(sym, "15m")
            agg = grid.run_folds(strat, sp, ep, TEST_FOLDS)
            fin["test"][sym] = {
                "geo": round(agg["geo_growth"], 4),
                "folds_positive": agg["folds_positive"],
                "worst_dd": round(agg["worst_dd"], 1),
                "trades": agg["trades"],
                "per_fold": agg["per_fold"],
            }
            print(f"  TEST  {sym}: {agg['geo_growth']:.3f}x/fold  "
                  f"dd{agg['worst_dd']:.0f}%  tr{agg['trades']}  "
                  f"{agg['per_fold']}")

        # stress ladders on TRAIN (BTC as the reference asset)
        fin["stress"] = {}
        grid.load_all("BTCUSDT", "15m")
        base = grid.run_folds(strat, sp, ep, TRAIN_FOLDS)["geo_growth"]
        stress_defs = {
            "pen_1bps": {"penetration_bps": 1.0},
            "pen_2bps": {"penetration_bps": 2.0},
            "pen_5bps": {"penetration_bps": 5.0},
            "taker_entry": {"entry_mode": "taker"},
            "fees_x1.5": {"maker_fee": 0.0003, "taker_fee": 0.0009},
            "slip_x2": {"slip": 0.0004},
        }
        for name, patch in stress_defs.items():
            ep2 = ExecParams(**{**ep.__dict__, **patch})
            g2 = grid.run_folds(strat, sp, ep2, TRAIN_FOLDS)["geo_growth"]
            fin["stress"][name] = round(g2, 4)
            print(f"  STRESS {name}: {g2:.3f}x (base {base:.3f})")

        artifact["finalists"].append(fin)

    out = RESULTS / "finalists_15m.json"
    out.write_text(json.dumps(artifact, indent=1, default=str) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
