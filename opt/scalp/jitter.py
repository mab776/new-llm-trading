"""Value-shaking robustness: random +/-15% perturbation of every finalist knob.

Pre-committed rule: the finalist passes if >=80% of jittered variants keep a
positive TRAIN edge (geo > 1.0) on the worst asset, and the median jittered
min-asset geo stays >= half the finalist's log-edge. Run on TRAIN only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opt.scalp import grid
from opt.scalp.engine import ExecParams
from opt.scalp.grid import TRAIN_FOLDS

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

FINALIST_SIG = {"n": 96, "vol_expand": 1.3, "vol_n": 32, "trend_gate": None}
FINALIST_EP = dict(entry_mode="maker", sl_atr=2.0, tp_atr=6.0, cooldown_bars=2)
N_DRAWS = 40


def main() -> None:
    rng = np.random.default_rng(2026)
    draws = []
    for _ in range(N_DRAWS):
        f = lambda: 1.0 + rng.uniform(-0.15, 0.15)
        sig = dict(FINALIST_SIG)
        sig["n"] = max(4, int(round(FINALIST_SIG["n"] * f())))
        sig["vol_expand"] = round(FINALIST_SIG["vol_expand"] * f(), 3)
        sig["vol_n"] = max(8, int(round(FINALIST_SIG["vol_n"] * f())))
        ep = ExecParams(**{**FINALIST_EP,
                           "sl_atr": round(FINALIST_EP["sl_atr"] * f(), 3),
                           "tp_atr": round(FINALIST_EP["tp_atr"] * f(), 3)})
        draws.append((sig, ep))

    # evaluate per asset (one data load per asset, all draws)
    min_geos = np.full(N_DRAWS, np.inf)
    per_asset = {}
    for sym in SYMBOLS:
        grid.load_all(sym, "15m")
        geos = []
        for sig, ep in draws:
            agg = grid.run_folds("donchian_breakout", sig, ep, TRAIN_FOLDS)
            geos.append(agg["geo_growth"])
        per_asset[sym] = geos
        min_geos = np.minimum(min_geos, np.asarray(geos))
        print(f"{sym}: median {np.median(geos):.4f}  min {np.min(geos):.4f}  "
              f"neg {(np.asarray(geos) <= 1.0).sum()}/{N_DRAWS}", flush=True)

    frac_pos = float((min_geos > 1.0).mean())
    med = float(np.median(min_geos))
    base_min_geo = 1.053  # finalist's min-asset TRAIN geo (from finalists run)
    print(f"\nworst-asset jitter: {frac_pos*100:.0f}% positive "
          f"(gate >=80%), median {med:.4f} "
          f"(gate >= {math.exp(math.log(base_min_geo)/2):.4f})")
    verdict = frac_pos >= 0.80 and med >= math.exp(math.log(base_min_geo) / 2)
    print("JITTER VERDICT:", "PASS" if verdict else "FAIL")

    out = Path(__file__).parent / "results" / "jitter_15m.json"
    out.write_text(json.dumps({
        "finalist_sig": FINALIST_SIG, "finalist_ep": FINALIST_EP,
        "n_draws": N_DRAWS, "per_asset": per_asset,
        "min_geo_positive_frac": frac_pos, "min_geo_median": med,
        "verdict": "PASS" if verdict else "FAIL",
    }, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
