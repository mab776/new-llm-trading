"""Weekly (1w) alignment vote — long-range trend in the score.

Marc's ask (2026-07-23): "long range trends are super important — should a
1-week candle score join the equation?" The alignment already votes 1h (weight
0 — searched, pure noise) and 1d (weight 3 — searched, real alpha). This probe
adds a 1w vote through the SAME machinery and the SAME protocol as the
2026-07-19 alignment sweep that shipped {"1h":0,"1d":3}.

PRE-COMMITTED PROTOCOL (select-TRAIN / report-TEST; holdout untouched here):
  cells: 1w weight w in {1,2,3,5} holding {"1h":0,"1d":3}, plus two fixed
  trade-off cells {"1d":0,"1w":3} (weekly REPLACES daily) and {"1d":2,"1w":2}.
  GATE 1: TRAIN geo must beat the baseline {"1h":0,"1d":3,"1w":0}.
  GATE 2: survivors must also beat baseline TEST geo.
  Survivors then: ETH/SOL fold check + multi-asset clean-holdout INVARIANCE
  (report-only). Anchor sanity: the w=0 baseline must reproduce the known
  no-1w TRAIN geo (+166.40) — the 1w data may not change anything at weight 0.

  ⚠️ Coverage caveat (pre-declared): Bitget history starts 2020-08; weekly
  indicators need 50 candles, so the 1w vote only exists from ~mid-2021 —
  21H1 (TRAIN) and part of 21H2 (TEST) run without it in EVERY cell alike.
  ⚠️ A TF absent from alignment_scale_by_tf votes at the legacy FLAT scale —
  every cell passes an explicit full dict.

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.probe_weekly_alignment
"""
from __future__ import annotations

import opt.driver as drv

SLIP = 2e-4
KW = dict(slip=SLIP, funding=True, exit_granularity="primary")

CELLS = {
    "baseline 1w=0":      {"1h": 0, "1d": 3, "1w": 0},
    "1w=1":               {"1h": 0, "1d": 3, "1w": 1},
    "1w=2":               {"1h": 0, "1d": 3, "1w": 2},
    "1w=3":               {"1h": 0, "1d": 3, "1w": 3},
    "1w=5":               {"1h": 0, "1d": 3, "1w": 5},
    "1d->1w swap (0/3)":  {"1h": 0, "1d": 0, "1w": 3},
    "split 1d=2 1w=2":    {"1h": 0, "1d": 2, "1w": 2},
}


def row(tag: str, weights: dict, folds) -> dict:
    r = drv.evaluate({}, folds=folds,
                     strat={"alignment_scale_by_tf": weights}, **KW)
    print(f"  {tag:22s} geo {r['geo_pct']:+8.2f}%/f  cx {r['compound_x']:9.2f}x  "
          f"worst {r['worst_fold']:+7.1f}%  DD {r['max_dd']:4.1f}%  tr {r['total_trades']:4d}",
          flush=True)
    return r


if __name__ == "__main__":
    drv.setup(extra_timeframes=["1w"])
    print("== 1w alignment vote — TRAIN (select) ==")
    train = {tag: row(tag, w, drv.TRAIN_FOLDS) for tag, w in CELLS.items()}
    base_t = train["baseline 1w=0"]["geo_pct"]
    winners = [t for t, r in train.items()
               if t != "baseline 1w=0" and r["geo_pct"] > base_t]
    print(f"\n  TRAIN baseline {base_t:+.2f} (anchor: must equal the no-1w +166.40)")
    print(f"  TRAIN winners: {winners or 'NONE'}")
    print("\n== TEST (gate) ==")
    test = {tag: row(tag, CELLS[tag], drv.TEST_FOLDS)
            for tag in ["baseline 1w=0"] + winners}
    base_e = test["baseline 1w=0"]["geo_pct"]
    for t in winners:
        verdict = "PASS" if test[t]["geo_pct"] > base_e else "FAIL"
        print(f"  {t}: TEST {test[t]['geo_pct']:+.2f} vs baseline {base_e:+.2f} -> {verdict}")
