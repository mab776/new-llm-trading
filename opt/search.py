"""Random search over config-space parameters, ranked by robust geo-mean objective."""
from __future__ import annotations

import sys, json, random, time
import numpy as np

import opt.driver as drv


def sample(rng: random.Random) -> dict:
    lev = rng.choice([10, 12, 15, 18, 20, 25, 30])
    strong = rng.uniform(12, 45)
    marg = rng.uniform(10, strong)  # marginal_low <= strong
    tp1 = rng.uniform(1.2, 4.5)
    tp2 = tp1 + rng.uniform(0.5, 6.0)
    tp1exit = rng.uniform(0.2, 0.7)
    atr_sl = rng.uniform(0.8, 3.0)
    sl_strat = rng.choice(["atr", "hybrid", "structure"])
    # weights via random positive numbers
    w = {k: rng.uniform(0.05, 1.0) for k in
         ["trend", "momentum", "volume", "support_resistance", "risk"]}
    min_adx = rng.uniform(8, 28)
    min_agree = rng.choice([0, 1, 2, 3])
    tm_agree = rng.random() < 0.5
    skip_choppy = rng.random() < 0.7
    skip_vol = rng.random() < 0.3
    cooldown = rng.choice([0, 1, 2, 3, 5])
    loss_pen = rng.choice([0.0, 2.0, 5.0, 8.0])
    use_trail = rng.random() < 0.4

    ov = {
        "tier.leverage": lev,
        "tier.strong_threshold": round(strong, 1),
        "tier.marginal_threshold_low": round(marg, 1),
        "tier.tp1_rr": round(tp1, 2),
        "tier.tp2_rr": round(tp2, 2),
        "tier.tp1_exit_pct": round(tp1exit, 2),
        "scoring.atr_sl_multiplier": round(atr_sl, 2),
        "trading.stop_loss_strategy": sl_strat,
        "weights": w,
        "filters.min_adx": round(min_adx, 1),
        "filters.min_category_agreement": min_agree,
        "filters.require_trend_momentum_agree": tm_agree,
        "filters.skip_choppy_regime": skip_choppy,
        "filters.skip_volatile_regime": skip_vol,
        "risk.cooldown_candles_after_sl": cooldown,
        "risk.consecutive_loss_penalty": loss_pen,
    }
    if use_trail:
        ov["bt.enable_trailing_stops"] = True
        ov["trailing.enabled"] = True
        ov["trailing.activation_pct"] = round(rng.uniform(0.5, 3.0), 2)
        ov["trailing.callback_pct"] = round(rng.uniform(0.3, 2.0), 2)
    return ov


def objective(res: dict) -> float:
    """Robust profit objective. Reward compounding across regimes; hard-penalize blowups
    and too-few-trades (statistical noise)."""
    if res["total_trades"] < 120:       # ~27/yr minimum for significance
        return -1e9
    if res["worst_fold"] < -35:          # any regime blowing up is disqualifying
        return -1e9
    if res["max_dd"] > 55:
        return -1e9
    return res["geo_pct"]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    drv.setup()
    rng = random.Random(seed)
    base = drv.evaluate({})
    print("BASELINE:", drv.fmt(base), f"[obj={objective(base):.2f}]\n")

    results = []
    t0 = time.time()
    for i in range(n):
        ov = sample(rng)
        try:
            res = drv.evaluate(ov)
        except Exception:
            continue
        obj = objective(res)
        results.append((obj, ov, res))
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{n} ({time.time()-t0:.0f}s)", file=sys.stderr)

    results.sort(key=lambda x: x[0], reverse=True)
    print(f"Searched {len(results)} configs in {time.time()-t0:.0f}s. Top 12:\n")
    for obj, ov, res in results[:12]:
        print(f"[obj={obj:6.2f}] {drv.fmt(res)}")
        print(f"    ov={json.dumps(ov)}")
    # save top 40
    out = [{"obj": o, "ov": ov, "res": res} for o, ov, res in results[:40]]
    with open("opt/top_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved top 40 to opt/top_results.json")


if __name__ == "__main__":
    main()
