"""Backlog #4 — maker-entry modeling.

Current entries are market/taker (0.06% + slip). This screens the alternative: rest a
limit at the decision bar's close, fill it only if the next bar trades back to it
(maker 0.02%, no entry slip), and eat the missed fills when price runs away. At this
trade count the per-entry saving (0.04% fee + 2bps slip) is real money IF the missed
fills don't cost more edge than they save.

Apples-to-apples: BOTH runs use slip=2bps + funding + liquidation; the ONLY difference
is entry_mode. Exit fees/slip are identical (SL=taker+slip, TP=maker, unchanged).
Select on TRAIN halves, confirm on held-out TEST + full yearly (chrono) folds.
"""
from __future__ import annotations
import opt.driver as drv

SLIP = 0.0002  # 2bps — applies to market fills (taker entries + all SL/EOB exits)


def row(tag, symbol_label, strat):
    tr = drv.evaluate({}, folds=drv.TRAIN_FOLDS, slip=SLIP, strat=strat, funding=True)
    te = drv.evaluate({}, folds=drv.TEST_FOLDS, slip=SLIP, strat=strat, funding=True)
    fu = drv.evaluate({}, folds=drv.FOLDS, slip=SLIP, strat=strat, funding=True)
    print(f"  {tag:14s} | TRAIN {tr['geo_pct']:+7.1f}%/f  TEST {te['geo_pct']:+7.1f}%/f  "
          f"| FULL {fu['compound_x']:9.2f}x  worst {fu['worst_fold']:+6.1f}%  "
          f"DD {fu['max_dd']:4.1f}%  fills {fu['total_trades']}")
    return fu


def bench(symbol_label, symbol):
    drv._PRE = None  # force reload for the new symbol
    drv.setup(symbol=symbol)
    print(f"\n=== {symbol_label} ({symbol}) — 2bps slip + funding + liquidation ===")
    taker = row("taker (base)", symbol_label, {"entry_mode": "taker"})
    maker = row("maker limit", symbol_label, {"entry_mode": "maker"})
    miss = taker["total_trades"] - maker["total_trades"]
    pct = 100.0 * miss / taker["total_trades"] if taker["total_trades"] else 0.0
    print(f"  -> maker missed {miss}/{taker['total_trades']} taker fills ({pct:.1f}%); "
          f"FULL {taker['compound_x']:.2f}x -> {maker['compound_x']:.2f}x")


def main():
    bench("BTC", "BTC/USDT:USDT")
    bench("ETH", "ETH/USDT:USDT")


if __name__ == "__main__":
    main()
