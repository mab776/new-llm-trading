"""Validate the 1x contingency configs (config-*-1x.json).

Canada context (2026-07-17): serving leverage is banned, so if 25x access ever
disappears the bot must run at 1x. These configs keep NOTIONAL identical
(risk 2%->50%, cap 4.4%->100%, rail kept at 0.66 — the leverage_scenarios sim
showed rail=1.0 is strictly worse) and leave every strategy parameter alone.

Two checks:
1. OOS replay from the config FILES must reproduce the in-memory scenario
   (~3.69x vs live 4.00x) — proves the overlays encode exactly that variant.
2. TRAIN-window (2021-01 -> 2025-05) comparison 25x-vs-1x — config blessing
   belongs on the selection window, not the holdout.

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.validate_1x
"""
from __future__ import annotations

from opt.holdout_oos import HOLD_START, HOLD_END, SYMBOLS, _load
from opt.multi_asset import simulate_multi
from opt.sizing_scenarios import MIN_QTY, SIZE_STEP

MINS = {"min_qty": MIN_QTY, "size_step": SIZE_STEP}
TRAIN_START, TRAIN_END = "2021-01-01", "2025-05-31"

CONFIG_SETS = {
    "25x (live)": {"BTC": "config.json", "ETH": "config-eth.json",
                   "SOL": "config-sol.json"},
    "1x  (contingency)": {"BTC": "config-1x.json", "ETH": "config-eth-1x.json",
                          "SOL": "config-sol-1x.json"},
}


def main() -> None:
    print(f"\n1x contingency validation — real minimums, maker touch, 2bps slip")
    print(f"{'config set':20s} {'window':22s} {'compound':>9s} {'maxDD':>7s} "
          f"{'trades':>7s} {'win%':>6s}")
    for label, paths in CONFIG_SETS.items():
        assets = {sym: _load(SYMBOLS[sym], cfg) for sym, cfg in paths.items()}
        for window, (start, end) in {
            "TRAIN 2021-01..25-05": (TRAIN_START, TRAIN_END),
            "OOS   2025-06..26-04": (HOLD_START, HOLD_END),
        }.items():
            res = simulate_multi(assets, start, end, slip=.0002,
                                 exit_granularity="sub", strat=dict(MINS))
            compound = max(.01, 1 + res.return_pct / 100)
            print(f"{label:20s} {window:22s} {compound:8.2f}x "
                  f"{res.max_dd_pct:6.1f}% {res.trades:7d} {res.win_rate:5.1f}%")


if __name__ == "__main__":
    main()
