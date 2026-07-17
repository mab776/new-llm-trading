"""Maker-fill bracket: bound the maker-vs-taker question on the clean OOS window.

The canonical maker model fills a touched limit at exact price (fantasy best
case). This ladder re-runs the standard profile with the pessimistic
queue-penetration rule (fill only if the bar trades THROUGH the limit by X bps)
plus a 70%-fill stress, and regenerates the taker reference, all with real
Bitget minimums so every number is comparable with opt/sizing_scenarios.

Decision rule (Marc, 2026-07-16): if the harshest maker bound still beats the
taker reference, maker wins outright and the 2-week live fill-rate measurement
is a formality. Otherwise live fill data decides (~2026-07-30, 86% rule).

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.maker_fill_bracket
"""
from __future__ import annotations

from opt.holdout_oos import HOLD_START, HOLD_END, PROFILES, SYMBOLS, _load
from opt.multi_asset import simulate_multi
from opt.sizing_scenarios import MIN_QTY, SIZE_STEP

MINS = {"min_qty": MIN_QTY, "size_step": SIZE_STEP}

LADDER = [
    # (label, strat overrides on top of mins)
    ("maker touch (canonical)", {}),
    ("maker 1bps penetration",  {"maker_queue_penetration_bps": 1.0}),
    ("maker 2bps penetration",  {"maker_queue_penetration_bps": 2.0}),
    ("maker 5bps penetration",  {"maker_queue_penetration_bps": 5.0}),
    ("maker 5bps + 70% fill",   {"maker_queue_penetration_bps": 5.0,
                                 "maker_fill_probability": 0.7}),
]


def _row(label: str, res) -> str:
    compound = max(0.01, 1 + res.return_pct / 100)
    funnel = (f"orders {res.maker_orders:4d} touch {res.maker_touches:4d} "
              f"elig {res.maker_queue_eligible:4d} fill {res.maker_fills:4d}"
              if res.maker_orders else "taker: all entries fill")
    return (f"{label:26s} {compound:8.2f}x  DD {res.max_dd_pct:5.1f}%  "
            f"tr {res.trades:4d}  win {res.win_rate:4.1f}%  | {funnel}")


def main() -> None:
    assets = {label: _load(SYMBOLS[label], cfgpath)
              for label, cfgpath in PROFILES["standard"].items()}
    print(f"\nMaker-fill bracket — clean OOS {HOLD_START} -> {HOLD_END}, "
          f"standard profile, real Bitget minimums, 2bps slip\n")
    results = {}
    for label, overrides in LADDER:
        res = simulate_multi(assets, HOLD_START, HOLD_END, slip=.0002,
                             exit_granularity="sub", strat=MINS | overrides)
        results[label] = res
        print(_row(label, res), flush=True)

    # Taker reference last: entry_mode lives on the shared configs, so this
    # mutation must not precede any maker run.
    for item in assets.values():
        item.config.trading.entry_mode = "taker"
    taker = simulate_multi(assets, HOLD_START, HOLD_END, slip=.0002,
                           exit_granularity="sub", strat=dict(MINS))
    print(_row("taker (reference)", taker))

    taker_x = max(0.01, 1 + taker.return_pct / 100)
    harshest = min(max(0.01, 1 + r.return_pct / 100) for r in results.values())
    print(f"\nharshest maker bound {harshest:.2f}x vs taker {taker_x:.2f}x -> "
          + ("MAKER WINS OUTRIGHT (pessimistic bound >= taker)"
             if harshest >= taker_x else
             "INCONCLUSIVE: live fill rate decides (86% rule, eval ~2026-07-30)"))


if __name__ == "__main__":
    main()
