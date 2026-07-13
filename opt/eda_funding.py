"""
EDA: does extreme perp funding predict forward BTC returns? (causal, no leakage)

Thesis (backlog #1): extreme positive funding = crowded longs → mean-reversion fuel
(and you're paid to short). Before wiring funding into the strategy we MEASURE the raw
conditional edge: bucket bars by a causal funding metric, look at forward N-bar returns.

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python opt/eda_funding.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

import opt.driver as drv

drv.setup()
pre = drv._PRE
fund = drv._FUND  # per-4h-bar sum of 8h funding rates, aligned to pre.timestamps

ts = pd.DatetimeIndex(pre.timestamps)
close = np.array([p.close if p is not None else np.nan for p in pre.primary], dtype=float)
f = np.array(fund, dtype=float)
n = len(close)

print(f"bars={n}  span {ts[0].date()}..{ts[-1].date()}", file=sys.stderr)
print(f"per-bar funding: nonzero={np.count_nonzero(f)}  "
      f"mean={f.mean():.6f} std={f.std():.6f} min={f.min():.5f} max={f.max():.5f}")

# Causal funding metrics (only bars <= i):
#  - fema: EWM mean of the per-bar funding sum (smooths the 0/nonzero settlement pattern)
#  - fz:   z-score of fema vs its own trailing window (regime-adaptive; funding magnitudes
#          differ a lot 2021 vs 2023)
fser = pd.Series(f, index=ts)
for span in (12, 30, 60):  # 2d, 5d, 10d at 4h
    fema = fser.ewm(span=span, adjust=False).mean()
    globals()[f"fema{span}"] = fema.to_numpy()

fema = fema30 = globals()["fema30"]
roll_mean = pd.Series(fema, index=ts).rolling(180, min_periods=30).mean()
roll_std = pd.Series(fema, index=ts).rolling(180, min_periods=30).std()
fz = ((pd.Series(fema, index=ts) - roll_mean) / roll_std).to_numpy()

# Forward returns over H bars (H*4h). Long-side simple return.
def fwd_ret(H):
    r = np.full(n, np.nan)
    r[:n-H] = close[H:] / close[:n-H] - 1.0
    return r

def bucket_report(metric, name, horizons=(1, 3, 6, 12, 30)):
    valid = np.isfinite(metric)
    q = np.full(n, np.nan)
    m = metric[valid]
    # quintiles of the causal metric over the whole sample (descriptive, not a trading rule)
    edges = np.nanquantile(m, [0.2, 0.4, 0.6, 0.8])
    qb = np.digitize(metric, edges)  # 0..4
    print(f"\n=== metric={name}  quintile edges={np.round(edges,5)} ===")
    print(f"{'H(bars)':>8} " + " ".join(f"Q{k}:{'':>7}" for k in range(5)) + "   Q4-Q0(long) fadeSpread")
    for H in horizons:
        fr = fwd_ret(H)
        cells = []
        means = []
        for k in range(5):
            sel = (qb == k) & np.isfinite(fr) & valid
            mu = np.nanmean(fr[sel]) * 100 if sel.sum() else np.nan
            means.append(mu)
            cells.append(f"{mu:+6.2f}%({sel.sum():>4})")
        # If thesis holds: high-funding (Q4) forward returns < low-funding (Q0)
        spread = means[4] - means[0]
        print(f"{H:>8} " + " ".join(cells) + f"   {spread:+6.2f}%")

bucket_report(fema30, "fema30 (5d EWM funding)")
bucket_report(fz, "fz (z-score of fema30, 30d window)")
bucket_report(fema12, "fema12 (2d EWM funding)")
