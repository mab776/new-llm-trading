"""Geometric-structure probe — swing/trendline awareness on top of the strategy.

Motivation (Marc, 2026-07-23): the live whipsaw day put every ATR stop just
ABOVE the rising trendline BTC then tagged and bounced from. The strategy has
no geometric concept — stops are entry - k*ATR, S/R scoring is horizontal
only. This probe searches basic structure trading on top of the frozen
strategy: (A) structural STOP placement, (B) entry proximity to structure,
(C) structure-break exits.

PRE-COMMITTED PROTOCOL (select-TRAIN / report-TEST; holdout untouched here):
  Stage A  structural stops:  sl_structure {swing,trendline} x mode
           {widen,tighten,replace} x wing {2,3,5} x buffer {0.25,0.5} x
           size_comp {off,on} on BTC TRAIN folds.
  Stage B  entry gate:        wing {2,3,5} x entry_struct_max_atr {1,2,3}.
  Stage C  break exit:        source {swing,trendline} x wing {2,3,5}.
  Stage D  the best TRAIN cell of each stage (only if it beats baseline
           TRAIN geo) + their combination -> TEST folds. GATE: TEST geo must
           also beat baseline TEST geo. Worst fold reported (must not
           materially degrade). Survivors -> ETH/SOL fold check, then
           multi-asset clean-holdout INVARIANCE (report-only, no selection).
  Design notes: thresholds/filters/TP rails always see the baseline ATR
  geometry — only the executed SL moves (isolates stop placement); TPs stay
  ATR rails; liquidation modeling bounds over-wide stops at 25x. Engine
  identity with every knob off verified digit-equal vs HEAD before running.

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.probe_geometry [stageA|stageB|stageC|stageD]
"""
from __future__ import annotations

import sys

import opt.driver as drv

SLIP = 2e-4
KW = dict(slip=SLIP, funding=True, exit_granularity="primary")


def row(tag: str, strat: dict, folds) -> dict:
    r = drv.evaluate({}, folds=folds, strat=strat or None, **KW)
    print(f"  {tag:44s} geo {r['geo_pct']:+7.2f}%/f  cx {r['compound_x']:9.2f}x  "
          f"worst {r['worst_fold']:+7.1f}%  DD {r['max_dd']:4.1f}%  tr {r['total_trades']:4d}",
          flush=True)
    return r


def stage_a() -> None:
    print("\n== Stage A — structural STOP placement (BTC TRAIN) ==")
    base = row("baseline", {}, drv.TRAIN_FOLDS)
    results = []
    for src in ("swing", "trendline"):
        for mode in ("widen", "tighten", "replace"):
            for wing in (2, 3, 5):
                for buf in (0.25, 0.5):
                    for comp in (False, True):
                        strat = {"struct_pivot_n": wing, "sl_structure": src,
                                 "sl_structure_mode": mode,
                                 "sl_structure_buffer_atr": buf,
                                 "sl_structure_size_comp": comp}
                        tag = f"{src}/{mode} w{wing} b{buf} comp{int(comp)}"
                        r = row(tag, strat, drv.TRAIN_FOLDS)
                        results.append((r["geo_pct"], tag, strat))
    top = sorted(results, reverse=True)[:5]
    print(f"  baseline TRAIN geo {base['geo_pct']:+.2f} | top5:")
    for g, t, _ in top:
        print(f"    {t}: {g:+.2f}")


def stage_b() -> None:
    print("\n== Stage B — entry proximity gate (BTC TRAIN) ==")
    base = row("baseline", {}, drv.TRAIN_FOLDS)
    results = []
    for wing in (2, 3, 5):
        for x in (1.0, 2.0, 3.0):
            strat = {"struct_pivot_n": wing, "entry_struct_max_atr": x}
            tag = f"gate w{wing} x{x}"
            r = row(tag, strat, drv.TRAIN_FOLDS)
            results.append((r["geo_pct"], tag, strat))
    top = sorted(results, reverse=True)[:3]
    print(f"  baseline TRAIN geo {base['geo_pct']:+.2f} | top3:")
    for g, t, _ in top:
        print(f"    {t}: {g:+.2f}")


def stage_c() -> None:
    print("\n== Stage C — structure-break exit (BTC TRAIN) ==")
    base = row("baseline", {}, drv.TRAIN_FOLDS)
    results = []
    for src in ("swing", "trendline"):
        for wing in (2, 3, 5):
            strat = {"struct_pivot_n": wing, "struct_break_exit": src}
            tag = f"break {src} w{wing}"
            r = row(tag, strat, drv.TRAIN_FOLDS)
            results.append((r["geo_pct"], tag, strat))
    top = sorted(results, reverse=True)[:3]
    print(f"  baseline TRAIN geo {base['geo_pct']:+.2f} | top3:")
    for g, t, _ in top:
        print(f"    {t}: {g:+.2f}")


def stage_d(candidates: dict[str, dict]) -> None:
    """TEST gate for named TRAIN winners (edit-in from stage outputs)."""
    print("\n== Stage D — TEST gate ==")
    bt = row("baseline TRAIN", {}, drv.TRAIN_FOLDS)
    be = row("baseline TEST ", {}, drv.TEST_FOLDS)
    for tag, strat in candidates.items():
        row(f"{tag} TRAIN", strat, drv.TRAIN_FOLDS)
        row(f"{tag} TEST ", strat, drv.TEST_FOLDS)
    print(f"  gates: TRAIN > {bt['geo_pct']:+.2f} AND TEST > {be['geo_pct']:+.2f}")


if __name__ == "__main__":
    drv.setup()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "stageA"):
        stage_a()
    if which in ("all", "stageB"):
        stage_b()
    if which in ("all", "stageC"):
        stage_c()
