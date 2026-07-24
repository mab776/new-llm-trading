"""Probe: cross-market context votes — external daily series in the score.

Marc's ask (2026-07-23): "external symbols integration in the score, and BTC
in the alternate money score." The strategy is 100% self-referential; the 1w
alignment vote (deployed 2026-07-23, 216f565) proved the delivery mechanism —
slow external context at low bounded weight through the alignment machinery.
This probe extends the SAME discrete ±weight vote to three cross-market
sources (GOOD_IDEAS "cross-market context votes", top research candidate):

  btc  — BTC's 1d trend votes in the ETH and SOL scores only (BTC never votes
         on itself). Data in-house (Bitget 1d candles, history/ cache).
  dxy  — US dollar index (DX-Y.NYB) daily trend, sign INVERTED (pre-declared:
         dollar strength = crypto-bearish), votes on all three assets.
  spx  — S&P 500 (^GSPC) daily trend, positive sign (risk-on = crypto-bullish),
         votes on all three assets.

Mechanism: opt-in ``context_votes`` strat knob (fastbt.apply_context_votes,
called in BOTH engines right after apply_daily_overlay; default None ⇒
engine-identical, 411 tests green). Trend metric = the SAME score_trend the
alignment vote uses, computed on the source's daily bars. Votes stay DISCRETE
(isolate the mechanism; tanh is a separately parked knob).

PRE-COMMITTED PROTOCOL (written before any results):
  * Folds: multi-asset portfolio sims (BTC+ETH+SOL, maker entry, sub-bar
    exits, funding, 2bps slip, frictionless): TRAIN = 21H1,22H1,23H1,24H1,25H1
    / TEST = 21H2,22H2,23H2,24H2 (probe_btc_delay precedent — a BTC→alts vote
    is invisible in single-asset BTC sims, so selection is portfolio-level).
  * Grid (9 cells + baseline): btc w∈{2,3,5} on (ETH,SOL); dxy(inv) w∈{2,3};
    spx w∈{2,3}; plus ONE combo cell built at runtime = each source's
    best-TRAIN cell, for the sources whose best cell beats baseline TRAIN
    (combo runs only if >=2 sources qualify).
  * Selection on TRAIN geo only. GATE 1: a cell must beat baseline TRAIN geo.
    GATE 2: survivors (per-source best + combo) must beat baseline TEST geo
    (probe_weekly_alignment convention — strict, no -2pt allowance).
  * OOS holdout 2025-06-01..2026-04-30 @$193 + real mins: INVARIANCE check
    only (candidate/baseline compound ratio, report-only, NOT selection).
  * ANCHOR: baseline (no knob) must equal context_votes=[] bit-exact on 21H1
    (knob inertness), and the full baseline TRAIN geo is printed against the
    pre-1w-era probe_btc_delay baseline (+611.9) for drift awareness only
    (the 1w vote + config evolution legitimately moved it).
  * Pre-declared causality: external daily bar dated D usable at decision
    closes >= D+1 00:00 UTC (cash/futures close 20:00-22:00 UTC), stale
    (forward-filled) over weekends/holidays exactly as live would see it;
    in-house BTC 1d candle open D usable >= D+1 00:00 UTC (candle complete).
    Data: pinned CSVs history/external/{dxy,spx}_1d.csv (opt/
    fetch_external_daily.py); 50-candle indicator warmup per source (votes
    exist from ~2020-02 for dxy/spx, ~2020-12 for btc — coverage printed).
  * DXY sign (-1) and SPX sign (+1) are pre-declared conventions, NOT
    searched. BTC vote applies to ETH/SOL only — also not searched.
  * No adoption from this script — live changes remain Marc's call with the
    full deployment protocol (scheduler/live scoring have no such knob yet).

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.probe_context_votes
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

import opt.fastbt as fb
from llm_trading_bot.data import fetch_multi_timeframe
from llm_trading_bot.scoring import score_trend
from opt.driver import HALF_FOLDS
from opt.multi_asset import simulate_multi
from opt.probe_reserved import (CONFIGS, HOLD_END, LOAD_START, MIN_QTY,
                                SIZE_STEP, SYMBOLS, _load, _with_balance)

SLIP = 0.0002
TRAIN = [HALF_FOLDS[i] for i in (0, 2, 4, 6, 8)]
TEST = [HALF_FOLDS[i] for i in (1, 3, 5, 7)]
HOLD_START = "2025-06-01"
EXTERNAL = Path(__file__).parent.parent / "history" / "external"


# ── context trend series ──────────────────────────────────────────────

def _series_from_daily(df: pd.DataFrame, label: str) -> tuple[pd.DatetimeIndex, list]:
    """(avail, trend) from a daily OHLCV frame: trend = score_trend on each
    completed bar's IndicatorSet; bar dated/opened D usable from D+1 00:00 UTC.
    build_indicatorsets returns None for the first <50 rows (warmup) — dropped."""
    inds = fb.build_indicatorsets(df, "1d")
    avail, trend = [], []
    for open_ts, ind in zip(df.index, inds):
        if ind is None:
            continue
        ts = pd.Timestamp(open_ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        avail.append(ts.normalize() + pd.Timedelta(days=1))
        trend.append(score_trend(ind).raw_score)
    idx = pd.DatetimeIndex(avail)
    print(f"  {label}: {len(idx)} usable days, votes from {idx[0].date()}",
          file=sys.stderr)
    return idx, trend


def _external(name: str) -> tuple[pd.DatetimeIndex, list]:
    df = pd.read_csv(EXTERNAL / f"{name}_1d.csv", index_col=0, parse_dates=True)
    df["Volume"] = df["Volume"].clip(lower=1.0)  # DXY has none; unused by trend
    return _series_from_daily(df, name)


def _btc_1d() -> tuple[pd.DatetimeIndex, list]:
    data = fetch_multi_timeframe(SYMBOLS["BTC"], ["1d"], start_date=LOAD_START,
                                 end_date=HOLD_END, warmup_periods=0,
                                 source="bitget", market="swap")
    return _series_from_daily(data["1d"], "btc")


def _vote(name, weight, sign, symbols, series) -> dict:
    avail, trend = series
    return {"name": name, "weight": float(weight), "sign": float(sign),
            "symbols": symbols, "avail": avail, "trend": trend}


# ── harness (probe_btc_delay pattern) ─────────────────────────────────

def eval_folds(assets: dict, folds, votes: list | None) -> dict:
    strat = {"context_votes": votes} if votes is not None else None
    rets, trades, per_fold = [], 0, []
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


def holdout(assets: dict, votes: list | None) -> tuple[float, float]:
    _with_balance(assets, 193.0)
    strat = {"min_qty": MIN_QTY, "size_step": SIZE_STEP}
    if votes is not None:
        strat["context_votes"] = votes
    res = simulate_multi(assets, HOLD_START, HOLD_END, slip=SLIP,
                         exit_granularity="sub", strat=strat)
    return max(.01, 1 + res.return_pct / 100), res.max_dd_pct


def row(tag: str, r: dict) -> str:
    folds = "  ".join(f"{n}:{v:+.0f}" for n, v in r["folds"])
    return f"{tag:>10}{r['geo']:>+10.1f}{r['worst']:>+8.1f}{r['trades']:>7d}   {folds}"


def main() -> None:
    print("Cross-market context-vote probe | multi-asset half-year folds | "
          "maker + sub-bar + funding + 2bps | frictionless folds, mins on holdout only")
    print("sources: btc->(ETH,SOL) | dxy inverted | spx  — discrete votes, "
          "select-TRAIN / report-TEST, holdout invariance-only\n")

    print("Loading context series...", file=sys.stderr)
    S = {"btc": _btc_1d(), "dxy": _external("dxy"), "spx": _external("spx")}
    assets = {label: _load(label) for label in CONFIGS}

    def cells() -> list[tuple[str, str, list | None]]:
        out = [("baseline", "", None)]
        for w in (2, 3, 5):
            out.append((f"btc-{w}", "btc",
                        [_vote("btc", w, 1.0, ("ETH", "SOL"), S["btc"])]))
        for w in (2, 3):
            out.append((f"dxy-{w}", "dxy", [_vote("dxy", w, -1.0, None, S["dxy"])]))
        for w in (2, 3):
            out.append((f"spx-{w}", "spx", [_vote("spx", w, 1.0, None, S["spx"])]))
        return out

    grid = cells()

    # ANCHOR: knob inertness — explicit empty vote list must be bit-exact.
    a_base = eval_folds(assets, [TRAIN[0]], None)
    a_empty = eval_folds(assets, [TRAIN[0]], [])
    if a_base["folds"][0][1] != a_empty["folds"][0][1]:
        print(f"ANCHOR FAILED: 21H1 baseline {a_base['folds'][0][1]} != "
              f"context_votes=[] {a_empty['folds'][0][1]} — knob not inert. STOP.")
        return
    print(f"Anchor OK: 21H1 baseline {a_base['folds'][0][1]:+.2f} == context_votes=[] "
          f"(bit-exact). Pre-1w-era btc_delay baseline was +611.9 TRAIN geo "
          f"(drift awareness only).\n")

    print(f"{'cell':>10}{'TRAINgeo':>10}{'worstF':>8}{'trades':>7}   folds")
    train = {}
    for tag, src, votes in grid:
        train[tag] = (src, eval_folds(assets, TRAIN, votes), votes)
        print(row(tag, train[tag][1]), flush=True)

    base_geo = train["baseline"][1]["geo"]
    # Per-source best-TRAIN cell; GATE 1 per cell = beats baseline TRAIN.
    best = {}
    for tag, (src, r, votes) in train.items():
        if src and (src not in best or r["geo"] > train[best[src]][1]["geo"]):
            best[src] = tag
    winners = [t for t in best.values() if train[t][1]["geo"] > base_geo]
    print(f"\nBaseline TRAIN geo {base_geo:+.1f} | per-source best: "
          + ", ".join(f"{s}:{t} {train[t][1]['geo']:+.1f}" for s, t in sorted(best.items()))
          + f" | GATE-1 winners: {winners or 'NONE'}")

    combo_tag = None
    if len(winners) >= 2:
        combo_votes = [v for t in winners for v in train[t][2]]
        combo_tag = "combo(" + "+".join(sorted(winners)) + ")"
        r = eval_folds(assets, TRAIN, combo_votes)
        train[combo_tag] = ("combo", r, combo_votes)
        print(row(combo_tag, r))
        if r["geo"] > base_geo:
            winners.append(combo_tag)

    if not winners:
        print("\nVERDICT (gate 1): no cell beats baseline on TRAIN — mechanism "
              "does not help. STOP (TEST/holdout not consulted).")
        return

    print("\nGate 1 PASSED. TEST (baseline + survivors):")
    print(f"{'cell':>10}{'TESTgeo':>10}{'worstF':>8}{'trades':>7}   folds")
    tb = eval_folds(assets, TEST, None)
    print(row("baseline", tb))
    survivors = []
    for t in winners:
        r = eval_folds(assets, TEST, train[t][2])
        print(row(t, r))
        if r["geo"] > tb["geo"]:
            survivors.append((t, r))
    for t, r in survivors:
        print(f"  {t}: TEST {r['geo']:+.1f} vs baseline {tb['geo']:+.1f} -> PASS")
    if not survivors:
        print(f"\nVERDICT (gate 2): no TRAIN winner beats baseline TEST geo "
              f"({tb['geo']:+.1f}) — split-disagreement noise. STOP.")
        return

    print("\nGate 2 PASSED. OOS holdout @$193 + real mins (INVARIANCE only, "
          "not selection):")
    hb, hb_dd = holdout(assets, None)
    print(f"  baseline: {hb:.2f}x  maxDD {hb_dd:.1f}%")
    for t, _ in survivors:
        hx, hdd = holdout(assets, train[t][2])
        print(f"  {t}: {hx:.2f}x  maxDD {hdd:.1f}%  ratio {hx / hb:.3f}")
    print("\nVERDICT: gate-passing survivors above. No adoption from this "
          "script — Marc's call with the full deployment protocol.")


if __name__ == "__main__":
    main()
