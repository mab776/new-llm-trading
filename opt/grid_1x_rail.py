"""Grid the max_position_pct rail for the 1x contingency configs ("66% is out
of my hat" — Marc, 2026-07-18). At 25x the rail never binds; at 1x a lot wants
50% x conviction(0.5..1.5) = up to 75% of balance, so the rail decides how
cash is split between the first lot and later pyramid lots.

Discipline: holdout untouched. Select on TRAIN-A, confirm on TRAIN-B.
Values > 0.75 cannot bind (1.00 = unbound anchor).

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.grid_1x_rail
"""
from __future__ import annotations

from opt.holdout_oos import SYMBOLS, _load
from opt.multi_asset import simulate_multi
from opt.sizing_scenarios import MIN_QTY, SIZE_STEP

MINS = {"min_qty": MIN_QTY, "size_step": SIZE_STEP}
WINDOWS = {"TRAIN-A 21-01..23-06": ("2021-01-01", "2023-06-30"),
           "TRAIN-B 23-07..25-05": ("2023-07-01", "2025-05-31")}
CONFIGS = {"BTC": "config-1x.json", "ETH": "config-eth-1x.json",
           "SOL": "config-sol-1x.json"}
RAILS = (0.40, 0.50, 0.60, 0.66, 0.70, 0.75, 1.00)


def main() -> None:
    assets = {sym: _load(SYMBOLS[sym], cfg) for sym, cfg in CONFIGS.items()}
    print("\n1x rail grid — real minimums, maker touch, 2bps slip "
          "(holdout NOT used)\n")
    header = f"{'rail':>6s}"
    for w in WINDOWS:
        header += f" | {w}: {'cx':>8s} {'DD':>6s} {'tr':>5s}"
    print(header)
    for rail in RAILS:
        for item in assets.values():
            item.config.position_sizing.max_position_pct = rail
        row = f"{rail:6.2f}"
        for start, end in WINDOWS.values():
            res = simulate_multi(assets, start, end, slip=.0002,
                                 exit_granularity="sub", strat=dict(MINS))
            cx = max(.01, 1 + res.return_pct / 100)
            row += f" | {'':21s}{cx:8.2f}x {res.max_dd_pct:5.1f}% {res.trades:5d}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
