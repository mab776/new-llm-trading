"""
EDA pass 2: extreme tails + trend control. Is the funding->forward-return effect
real, or just a proxy for trend (high funding in bull markets where longs win anyway)?

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python opt/eda_funding2.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

import opt.driver as drv

drv.setup()
pre = drv._PRE
ts = pd.DatetimeIndex(pre.timestamps)
close = np.array([p.close if p is not None else np.nan for p in pre.primary], dtype=float)
ema200 = np.array([p.ema_200 if (p is not None and p.ema_200) else np.nan for p in pre.primary], dtype=float)
f = np.array(drv._FUND, dtype=float)
n = len(close)

fser = pd.Series(f, index=ts)
fema = fser.ewm(span=30, adjust=False).mean().to_numpy()

def fwd_ret(H):
    r = np.full(n, np.nan); r[:n-H] = close[H:] / close[:n-H] - 1.0; return r

print("=== EXTREME TAILS of fema30 (5d EWM funding), forward returns ===")
lo10, lo05 = np.nanquantile(fema, 0.10), np.nanquantile(fema, 0.05)
hi90, hi95 = np.nanquantile(fema, 0.90), np.nanquantile(fema, 0.95)
print(f"edges: p5={lo05:.5f} p10={lo10:.5f} p90={hi90:.5f} p95={hi95:.5f}")
for H in (3, 6, 12, 30, 60):
    fr = fwd_ret(H); ok = np.isfinite(fr) & np.isfinite(fema)
    def mu(sel):
        s = sel & ok
        return (np.nanmean(fr[s])*100, s.sum(), (np.nanmean(fr[s] > 0)*100 if s.sum() else np.nan))
    b5 = mu(fema <= lo05); b10 = mu(fema <= lo10)
    t90 = mu(fema >= hi90); t95 = mu(fema >= hi95); allm = mu(np.ones(n, bool))
    print(f"H={H:>3}  bot5%:{b5[0]:+6.2f}%(n{b5[1]},{b5[2]:.0f}%+)  bot10%:{b10[0]:+6.2f}%  "
          f"ALL:{allm[0]:+6.2f}%({allm[2]:.0f}%+)  top10%:{t90[0]:+6.2f}%  "
          f"top5%:{t95[0]:+6.2f}%(n{t95[1]},{t95[2]:.0f}%+)")

print("\n=== TREND CONTROL: forward return by funding tercile, split by trend (close vs ema200) ===")
lo, hi = np.nanquantile(fema, [1/3, 2/3])
fb = np.digitize(fema, [lo, hi])  # 0 low,1 mid,2 high funding
up = close > ema200
for H in (6, 12, 30):
    fr = fwd_ret(H); ok = np.isfinite(fr) & np.isfinite(fema) & np.isfinite(ema200)
    print(f" H={H}:")
    for tr_name, trmask in (("UPtrend ", up), ("DOWNtrnd", ~up)):
        cells = []
        for k, kn in ((0, "loF"), (1, "midF"), (2, "hiF")):
            s = (fb == k) & trmask & ok
            m = np.nanmean(fr[s])*100 if s.sum() else np.nan
            cells.append(f"{kn}:{m:+6.2f}%(n{s.sum():>4})")
        print(f"    {tr_name}  " + "  ".join(cells))
