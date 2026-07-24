"""Release gate: min-size rescue (O=0.25, S=30) re-validated on the CURRENT base.

The probe (opt/probe_overshoot, gates passed 2026-07-20) ran on the pre-1w
strategy with the raw exchange minimums. Two things changed for adoption
(2026-07-23), so per the standard deployment process the arms re-run before
the supervised restart:

1. CURRENT BASE — the 1w alignment vote {"1h":0,"1d":3,"1w":2} is live; the
   1d-overlay lesson says advantages can invert on a new base.
2. TRUE LIVE FLOORS — live lots must TP1-SPLIT (70/30), so the smallest
   executable lot is the smallest splittable size, not the raw exchange min:
   BTC 0.0002 / ETH 0.02 / SOL 0.2 (2 steps each). Both the skip threshold
   and the rescue floor use these here, mirroring scheduler._rescue_min_size.

Arms (pre-committed): baseline(skip) vs rescue O=0.25/S=30 — the adopted
config values. Gates: (1) TRAIN geo better; (2) TEST geo >= baseline - 2 pts
(probe tolerance); (3) holdout invariance ratio >= 0.92 (report-only).
Balance $189/fold = the standard bot's current realized balance.

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.release_min_size_rescue
"""
from __future__ import annotations

import math

from opt.driver import HALF_FOLDS
from opt.multi_asset import simulate_multi
from opt.probe_reserved import CONFIGS, _load, _with_balance

SLIP = 0.0002
TRAIN = [HALF_FOLDS[i] for i in (0, 2, 4, 6, 8)]
TEST = [HALF_FOLDS[i] for i in (1, 3, 5, 7)]
HOLD_START, HOLD_END = "2025-06-01", "2026-04-30"
BALANCE = 189.0
# Smallest TP1-splittable lots (live floors), not the raw exchange minimums.
MIN_QTY = {"BTC": 0.0002, "ETH": 0.02, "SOL": 0.2}
SIZE_STEP = {"BTC": 0.0001, "ETH": 0.01, "SOL": 0.1}
RESCUE = {"min_size_overshoot": 0.25, "min_size_overshoot_score": 30.0}


def _strat(extra: dict | None) -> dict:
    counters = {"skips": 0, "floors": 0, "overshoots": 0}
    strat = {"min_qty": MIN_QTY, "size_step": SIZE_STEP,
             "_min_counters": counters}
    if extra:
        strat.update(extra)
    return strat


def eval_folds(assets: dict, folds, extra: dict | None) -> dict:
    rets, trades = [], 0
    skips = overs = 0
    per_fold = []
    for name, start, end in folds:
        _with_balance(assets, BALANCE)
        strat = _strat(extra)
        res = simulate_multi(assets, start, end, slip=SLIP,
                             exit_granularity="sub", strat=strat)
        rets.append(res.return_pct / 100.0)
        trades += res.trades
        c = strat["_min_counters"]
        skips += c["skips"]; overs += c["overshoots"]
        per_fold.append((name, res.return_pct))
    geo = (math.prod(1 + r for r in rets) ** (1 / len(rets)) - 1) * 100
    return {"geo": geo, "trades": trades, "skips": skips, "overshoots": overs,
            "worst": min(r for _, r in per_fold), "folds": per_fold}


def holdout(assets: dict, extra: dict | None):
    _with_balance(assets, BALANCE)
    strat = _strat(extra)
    res = simulate_multi(assets, HOLD_START, HOLD_END, slip=SLIP,
                         exit_granularity="sub", strat=strat)
    return (max(.01, 1 + res.return_pct / 100), res.max_dd_pct,
            strat["_min_counters"])


def _row(tag: str, r: dict) -> None:
    print(f"  {tag:18s} geo {r['geo']:+8.1f}  worst {r['worst']:+7.1f}  "
          f"skips {r['skips']:5d}  rescues {r['overshoots']:4d}  tr {r['trades']}",
          flush=True)


def main() -> None:
    print("Min-size rescue RELEASE GATE | current base (1w vote) | true live "
          f"floors {MIN_QTY} | ${BALANCE:g}/fold | maker+sub+funding+2bps")
    assets = {label: _load(label) for label in CONFIGS}

    print("\n== TRAIN ==")
    bt = eval_folds(assets, TRAIN, None); _row("baseline(skip)", bt)
    rt = eval_folds(assets, TRAIN, RESCUE); _row("rescue O.25/S30", rt)
    print("\n== TEST ==")
    be = eval_folds(assets, TEST, None); _row("baseline(skip)", be)
    re_ = eval_folds(assets, TEST, RESCUE); _row("rescue O.25/S30", re_)
    print("\n== Holdout (invariance) ==")
    bx, bdd, _ = holdout(assets, None)
    rx, rdd, rc = holdout(assets, RESCUE)
    print(f"  baseline {bx:5.2f}x DD {bdd:4.1f}%   rescue {rx:5.2f}x DD {rdd:4.1f}%"
          f"   ratio {rx/bx:.3f}  rescues {rc['overshoots']}")

    g1 = rt["geo"] > bt["geo"]
    g2 = re_["geo"] >= be["geo"] - 2.0
    g3 = rx / bx >= 0.92
    print(f"\nGATE 1 TRAIN better: {'PASS' if g1 else 'FAIL'} "
          f"({rt['geo']:+.1f} vs {bt['geo']:+.1f})")
    print(f"GATE 2 TEST within tolerance: {'PASS' if g2 else 'FAIL'} "
          f"({re_['geo']:+.1f} vs {be['geo']:+.1f})")
    print(f"GATE 3 holdout ratio >= 0.92: {'PASS' if g3 else 'FAIL'} ({rx/bx:.3f})")
    print("VERDICT:", "RELEASE OK" if (g1 and g2 and g3) else "DO NOT RELEASE")


if __name__ == "__main__":
    main()
