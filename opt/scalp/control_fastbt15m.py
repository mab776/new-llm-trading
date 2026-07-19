"""Control experiment: the shipped 4h composite-score strategy at 15m, retuned.

The 2026-07-13 static transplant already showed the untuned strategy dies at
5m (0.24x) and degrades at 1h. The fair question for the scalper project is:
does an honest RETUNE (entry thresholds, ATR stop scale, alignment weights
adapted to the 15m/1h/4h ladder) make the house scorer competitive with the
dedicated scalp strategies at 15m? Same Binance futures data, same fees, same
TRAIN folds as the scalp grids. Funding omitted (small at these horizons,
identical across variants).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_trading_bot.config import load_config
from opt.fastbt import precompute, simulate
from opt.scalp.engine import aggregate, load_futures
from opt.scalp.grid import TRAIN_FOLDS

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    df5 = load_futures("BTCUSDT", "5m", "2020-10-01", "2026-07-19")
    data = {
        "15m": aggregate(df5, "15min"),
        "1h": aggregate(df5, "1h"),
        "4h": aggregate(df5, "4h"),
    }
    t0 = time.monotonic()
    pre = precompute(data, "15m", warmup=200)
    print(f"precompute: {len(pre.timestamps)} bars in {time.monotonic()-t0:.0f}s",
          flush=True)

    base = load_config(ROOT / "config.json")
    base.trading.primary_timeframe = "15m"
    base.trading.timeframes = ["15m", "1h", "4h"]
    # analogous alignment to the shipped {"1h":0,"1d":3}: mute the next TF up,
    # weight the top of the ladder
    base.scoring.alignment_scale_by_tf = {"1h": 0.0, "4h": 3.0}

    tier = base.trading.active_leverage_tier
    variants = []
    for thr_mult in (1.0, 1.5, 2.0, 3.0):
        for atr_mult in (1.0, 0.75):
            variants.append({"thr_mult": thr_mult, "atr_mult": atr_mult})

    results = []
    for v in variants:
        cfg = base.model_copy(deep=True)
        t = cfg.trading.active_leverage_tier
        t.strong_threshold = tier.strong_threshold * v["thr_mult"]
        t.marginal_threshold_low = tier.marginal_threshold_low * v["thr_mult"]
        t.marginal_threshold_high = tier.marginal_threshold_high * v["thr_mult"]
        cfg.scoring.atr_sl_multiplier = base.scoring.atr_sl_multiplier * v["atr_mult"]
        growths = []
        rows = {}
        for label, s, e in TRAIN_FOLDS:
            r = simulate(pre, cfg, s, e, slip=0.0002, model_liquidation=True,
                         funding_by_pos=None, exit_granularity="primary")
            growths.append(max(0.01, 1 + r.return_pct / 100))
            rows[label] = (round(1 + r.return_pct / 100, 4),
                           round(r.max_dd_pct, 1), r.trades)
        geo = math.exp(sum(math.log(g) for g in growths) / len(growths))
        entry = {**v, "geo_growth": round(geo, 4),
                 "folds_positive": sum(1 for g in growths if g > 1),
                 "worst_dd": max(x[1] for x in rows.values()),
                 "trades": sum(x[2] for x in rows.values()),
                 "per_fold": rows}
        results.append(entry)
        print(f"thr x{v['thr_mult']:.1f} atr x{v['atr_mult']:.2f}: "
              f"{geo:.4f}x/fold  fp{entry['folds_positive']}  "
              f"dd{entry['worst_dd']:.0f}%  tr{entry['trades']}", flush=True)

    out = Path(__file__).parent / "results" / "control_fastbt_15m.json"
    out.write_text(json.dumps({"note": __doc__, "results": results},
                              indent=1, default=str) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
