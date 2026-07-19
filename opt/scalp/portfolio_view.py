"""Equal-weight 3-asset portfolio view of the chosen scalp config.

Each asset runs independently on 1/3 of capital (no cross-margining, matching
the live bot's shared-orchestrator one-symbol-per-scheduler structure).
Reports continuous TRAIN (2021-2024) and TEST (2024-2025-06) portfolio curves:
growth, max drawdown on the combined curve, and correlation of per-asset
daily returns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opt.scalp import grid
from opt.scalp.engine import ExecParams, simulate
from opt.scalp.strategies import STRATEGIES

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

SIG = {"n": 96, "vol_expand": 1.3, "vol_n": 32, "trend_gate": None}
EP = ExecParams(entry_mode="maker", sl_atr=2.0, tp_atr=6.0, cooldown_bars=2)

PERIODS = {
    "TRAIN": ("2021-01-01", "2024-01-01"),
    "TEST": ("2024-01-01", "2025-06-01"),
}


def main() -> None:
    curves: dict[str, dict[str, pd.Series]] = {p: {} for p in PERIODS}
    for sym in SYMBOLS:
        grid.load_all(sym, "15m")
        ls, ss, mel, mes = STRATEGIES["donchian_breakout"](_g("df"), SIG, _g("ctx"))
        for period, (s, e) in PERIODS.items():
            i0, i1 = grid.fold_bounds(("x", s, e))
            r = simulate(_g("ohlc"), _g("atr"), ls, ss, EP,
                         funding=_g("funding"), start_i=i0, end_i=i1,
                         subbars=_g("subbars"), return_equity=True)
            idx = _g("index")[i0:i0 + len(r.equity)]
            curves[period][sym] = pd.Series(r.equity, index=idx)
            print(f"{period} {sym}: {r.growth_x:.3f}x dd{r.max_dd_pct:.1f}% "
                  f"tr{r.trades} hold{r.avg_hold_bars:.0f}bars "
                  f"fees{r.fees_paid:.0f}", flush=True)

    summary = {}
    for period in PERIODS:
        aligned = pd.DataFrame(curves[period]).ffill().dropna()
        norm = aligned / aligned.iloc[0]
        port = norm.mean(axis=1)
        peak = port.cummax()
        dd = float(((peak - port) / peak).max() * 100)
        daily = norm.resample("1D").last().pct_change().dropna()
        corr = daily.corr()
        growth = float(port.iloc[-1])
        yrs = (port.index[-1] - port.index[0]).days / 365.25
        cagr = growth ** (1 / yrs) - 1
        print(f"\n== {period} equal-weight portfolio ==")
        print(f"growth {growth:.3f}x  CAGR {cagr*100:.1f}%  maxDD {dd:.1f}%")
        print("daily-return correlations:")
        print(corr.round(2).to_string())
        summary[period] = {
            "growth_x": round(growth, 4), "cagr_pct": round(cagr * 100, 2),
            "max_dd_pct": round(dd, 2),
            "corr": {f"{a}-{b}": round(float(corr.loc[a, b]), 3)
                     for ai, a in enumerate(SYMBOLS) for b in SYMBOLS[ai + 1:]},
        }

    out = Path(__file__).parent / "results" / "portfolio_15m.json"
    out.write_text(json.dumps({"sig": SIG, "exec": grid.ep_desc(EP),
                               "periods": summary}, indent=1) + "\n")
    print(f"\nwrote {out}")


def _g(key: str):
    return grid._G[key]


if __name__ == "__main__":
    main()
