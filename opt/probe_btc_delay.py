"""Probe: BTC-only entry delay — let ETH/SOL claim margin before BTC piles in.

Marc's idea (2026-07-21): BTC is often leading and taking the whole portfolio —
delay BTC's entry by 1-2 candles after its trigger while ETH/SOL stay
unfiltered. Live motivation: BTC (weakest asset OOS, standalone 1.65x) claims
the 4.4% portfolio margin cap first every episode (symbols are processed in
sorted order = BTC first, live and sim) and squeezes ETH (6.52x) / SOL (6.21x)
into MIN_SIZE_SKIPs at $193.

Related evidence, both directions: the decay probe (f6b31eb) showed per-asset
freshness gating HURTS (fresh and decaying entries both carry edge) — but that
tested ALL symbols; this is a PORTFOLIO-allocation hypothesis (shift capital
from the weakest asset to the strongest ones), not a signal-quality one. The
reserved-allocation probe (37b898a) showed cross-subsidy concentration is the
edge — a BTC delay keeps the shared pot, only re-orders who drinks first.

Mechanism (opt-in knob in simulate_multi, default-off = engine-identical,
verified bit-exact on 21H1: 4200.23/825 trades/9.54 dd):
  strat["entry_confirm_bars"] = {symbol: N} blocks the named symbol's entries
  during the first N bars of each fresh same-direction signal episode
  (consecutive entry-eligible streak). Later episode bars — pyramid adds
  included — pass untouched. Exits / signal-flips are never delayed.

PRE-COMMITTED PROTOCOL (written before results):
  * Folds: house half-year interleave on MULTI-ASSET portfolio sims
    (BTC+ETH+SOL, maker entry, sub-bar exits, funding, 2bps slip, frictionless):
    TRAIN = 21H1,22H1,23H1,24H1,25H1  /  TEST = 21H2,22H2,23H2,24H2.
  * Grid: BTC-delay N in {1, 2}; control arm ALL-delay 1 (delays every symbol)
    to separate "crowding relief" from "generic delay" — if BTC-only wins while
    ALL loses, the crowding story holds; if ALL wins too, it's a timing effect
    the decay probe already rejected (treat as suspect).
  * Selection on TRAIN geo-mean only. Gates: (1) best BTC-delay TRAIN must beat
    baseline TRAIN, else verdict = mechanism does not help, stop. (2) TEST
    (baseline + candidates) must be >= baseline TEST - 2 pts. (3) OOS holdout
    invariance at $193 + real mins: candidate/baseline compound ratio >= 0.92
    (report-only sanity, NOT selection).
  * No adoption from this script — live changes remain Marc's call with the
    full deployment protocol (scheduler has no such knob; live code would be new).

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.probe_btc_delay
"""
from __future__ import annotations

import math

from opt.driver import HALF_FOLDS
from opt.multi_asset import simulate_multi
from opt.probe_reserved import (CONFIGS, MIN_QTY, SIZE_STEP, _load,
                                _with_balance)

SLIP = 0.0002
TRAIN = [HALF_FOLDS[i] for i in (0, 2, 4, 6, 8)]
TEST = [HALF_FOLDS[i] for i in (1, 3, 5, 7)]
HOLD_START, HOLD_END = "2025-06-01", "2026-04-30"

# (label, strat-extra) — order matters for the report only.
VARIANTS = [
    ("baseline", None),
    ("BTC-1", {"entry_confirm_bars": {"BTC": 1}}),
    ("BTC-2", {"entry_confirm_bars": {"BTC": 2}}),
    ("ALL-1", {"entry_confirm_bars": {"BTC": 1, "ETH": 1, "SOL": 1}}),
]


def eval_folds(assets: dict, folds, strat: dict | None) -> dict:
    rets, trades = [], 0
    per_fold = []
    for name, start, end in folds:
        _with_balance(assets, 3000.0)
        res = simulate_multi(assets, start, end, slip=SLIP,
                             exit_granularity="sub", strat=strat)
        rets.append(res.return_pct / 100.0)
        trades += res.trades
        per_fold.append((name, res.return_pct))
    geo = (math.prod(1 + r for r in rets) ** (1 / len(rets)) - 1) * 100
    return {"geo": geo, "trades": trades, "folds": per_fold,
            "worst": min(r for _, r in per_fold)}


def holdout(assets: dict, strat_extra: dict | None) -> tuple[float, float]:
    _with_balance(assets, 193.0)
    strat = {"min_qty": MIN_QTY, "size_step": SIZE_STEP}
    if strat_extra:
        strat.update(strat_extra)
    res = simulate_multi(assets, HOLD_START, HOLD_END, slip=SLIP,
                         exit_granularity="sub", strat=strat)
    return max(.01, 1 + res.return_pct / 100), res.max_dd_pct


def main() -> None:
    print("BTC entry-delay probe | multi-asset half-year folds | "
          "maker + sub-bar + funding + 2bps | frictionless folds, mins on holdout only")
    assets = {label: _load(label) for label in CONFIGS}

    print(f"\n{'variant':>9}{'TRAINgeo':>10}{'worstF':>8}{'trades':>7}   folds")
    results = {}
    for label, extra in VARIANTS:
        r = eval_folds(assets, TRAIN, extra)
        results[label] = r
        folds = "  ".join(f"{n}:{v:+.0f}" for n, v in r["folds"])
        print(f"{label:>9}{r['geo']:>+10.1f}{r['worst']:>+8.1f}"
              f"{r['trades']:>7d}   {folds}", flush=True)

    base = results["baseline"]
    btc_best = max(("BTC-1", "BTC-2"), key=lambda k: results[k]["geo"])
    print(f"\nBaseline TRAIN geo {base['geo']:+.1f} | best BTC-delay {btc_best} "
          f"{results[btc_best]['geo']:+.1f} | ALL-1 {results['ALL-1']['geo']:+.1f}")

    if results[btc_best]["geo"] <= base["geo"]:
        print("VERDICT (gate 1): no BTC-delay variant beats baseline on TRAIN — "
              "mechanism does not help. STOP (TEST/holdout not consulted).")
        return

    print("\nGate 1 PASSED. TEST (baseline + candidates):")
    print(f"{'variant':>9}{'TESTgeo':>10}{'worstF':>8}{'trades':>7}   folds")
    test_rows = {}
    for label, extra in VARIANTS:
        r = eval_folds(assets, TEST, extra)
        test_rows[label] = r
        folds = "  ".join(f"{n}:{v:+.0f}" for n, v in r["folds"])
        print(f"{label:>9}{r['geo']:>+10.1f}{r['worst']:>+8.1f}"
              f"{r['trades']:>7d}   {folds}", flush=True)

    tb = test_rows["baseline"]["geo"]
    ok2 = test_rows[btc_best]["geo"] >= tb - 2.0
    print(f"\nGate 2 ({btc_best} TEST {test_rows[btc_best]['geo']:+.1f} vs "
          f"baseline {tb:+.1f} - 2): {'PASSED' if ok2 else 'FAILED'}")
    if not ok2:
        print("VERDICT: TRAIN winner fails the TEST gate — split-disagreement "
              "noise (same signature as decay-exit / rotation). STOP.")
        return

    print("\nGate 3: OOS holdout invariance @ $193 + real mins (report-only):")
    hb, db = holdout(assets, None)
    print(f"  baseline  {hb:6.2f}x  maxDD {db:.1f}%")
    variant_map = dict(VARIANTS)
    for label in (btc_best, "ALL-1"):
        hx, dx = holdout(assets, variant_map[label])
        ratio = hx / hb
        print(f"  {label:>9} {hx:6.2f}x  maxDD {dx:.1f}%  ratio {ratio:.3f} "
              f"({'>=0.92 OK' if ratio >= 0.92 else 'BELOW 0.92'})")
    print("\nAdoption remains Marc's call — live scheduler has no delay knob; "
          "shipping this would be new live code + supervised deploy.")


if __name__ == "__main__":
    main()
