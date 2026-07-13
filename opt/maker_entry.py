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


def row(tag, symbol_label, strat, exit_granularity="primary"):
    kwargs = dict(slip=SLIP, strat=strat, funding=True,
                  exit_granularity=exit_granularity)
    tr = drv.evaluate({}, folds=drv.TRAIN_FOLDS, **kwargs)
    te = drv.evaluate({}, folds=drv.TEST_FOLDS, **kwargs)
    fu = drv.evaluate({}, folds=drv.FOLDS, **kwargs)
    print(f"  {tag:14s} | TRAIN {tr['geo_pct']:+7.1f}%/f  TEST {te['geo_pct']:+7.1f}%/f  "
          f"| FULL {fu['compound_x']:9.2f}x  worst {fu['worst_fold']:+6.1f}%  "
          f"DD {fu['max_dd']:4.1f}%  fills {fu['total_trades']}")
    return fu


def bench(symbol_label, symbol, exit_granularity="primary"):
    drv._PRE = None  # force reload for the new symbol
    drv.setup(symbol=symbol)
    print(f"\n=== {symbol_label} ({symbol}) — {exit_granularity} exits, "
          "2bps slip + funding + liquidation ===")
    taker = row("taker (base)", symbol_label, {"entry_mode": "taker"}, exit_granularity)
    maker = row("maker limit", symbol_label, {"entry_mode": "maker"}, exit_granularity)
    miss = taker["total_trades"] - maker["total_trades"]
    pct = 100.0 * miss / taker["total_trades"] if taker["total_trades"] else 0.0
    print(f"  -> maker missed {miss}/{taker['total_trades']} taker fills ({pct:.1f}%); "
          f"FULL {taker['compound_x']:.2f}x -> {maker['compound_x']:.2f}x")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-granularity", choices=("primary", "sub"), default="primary")
    parser.add_argument("--include-sol", action="store_true")
    args = parser.parse_args()
    bench("BTC", "BTC/USDT:USDT", args.exit_granularity)
    bench("ETH", "ETH/USDT:USDT", args.exit_granularity)
    if args.include_sol:
        bench("SOL", "SOL/USDT:USDT", args.exit_granularity)


if __name__ == "__main__":
    main()
