"""Fee-reality framing for the scalper: how big is a scalp move vs the fee load?

For each cadence, compare the ATR (the natural TP/SL scale) against round-trip
fee+slippage costs. This is the structural reason a naive 5m transplant loses:
if fees eat >30-50% of the median favorable excursion, no entry signal can save
the strategy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opt.scalp.engine import aggregate, atr, load_futures

MAKER = 0.0002
TAKER = 0.0006
SLIP = 0.0002

ROUND_TRIPS = {
    "maker+maker": MAKER + MAKER,
    "maker+taker+slip": MAKER + TAKER + SLIP,
    "taker+taker+2slip": TAKER + SLIP + TAKER + SLIP,
}


def main() -> None:
    df5 = load_futures("BTCUSDT", "5m", "2021-01-01", "2026-07-19")
    for label, df in (("5m", df5), ("15m", aggregate(df5, "15min")),
                      ("1h", aggregate(df5, "1h")), ("4h", aggregate(df5, "4h"))):
        a = atr(df["High"], df["Low"], df["Close"], 14)
        atr_pct = (a / df["Close"] * 100).dropna()
        rng_pct = ((df["High"] - df["Low"]) / df["Close"] * 100).dropna()
        med = atr_pct.median()
        print(f"\n== {label} (BTC, 2021->2026) ==")
        print(f"ATR%   p25 {atr_pct.quantile(.25):.4f}  med {med:.4f}  "
              f"p75 {atr_pct.quantile(.75):.4f}")
        print(f"range% med {rng_pct.median():.4f}")
        for name, cost in ROUND_TRIPS.items():
            cost_pct = cost * 100
            for mult in (1.0, 1.5, 2.5):
                tp = mult * med
                print(f"  {name:>18} vs {mult:>3}xATR TP: fees = "
                      f"{cost_pct / tp * 100:5.1f}% of the move")


if __name__ == "__main__":
    main()
