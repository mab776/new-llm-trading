"""Leverage-vs-margin-cap equivalence on the clean OOS window.

Marc's question (2026-07-17): is 25x + 4.4% margin cap riskier than 5x + 22%
or 1x + 100%? Fees/funding are on notional, so scenarios keep NOTIONAL
constant by scaling risk_pct inversely with leverage. Differences left are:
isolated-liquidation distance (modeled), the 66% per-trade rail (binds at 1x),
and the physical margin<=balance bound (binds at 1x).

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.leverage_scenarios
"""
from __future__ import annotations

from opt.holdout_oos import HOLD_START, HOLD_END, PROFILES, SYMBOLS, _load
from opt.multi_asset import simulate_multi
from opt.sizing_scenarios import MIN_QTY, SIZE_STEP

MINS = {"min_qty": MIN_QTY, "size_step": SIZE_STEP}

SCENARIOS = [
    # label, leverage, risk_pct_per_trade, global_max_margin_pct, max_position_pct
    ("25x / risk 2% / cap 4.4% (live)", 25, 0.02, 0.044, 0.66),
    ("5x  / risk 10% / cap 22%",         5, 0.10, 0.22, 0.66),
    ("1x  / risk 50% / cap 100%",        1, 0.50, 1.00, 0.66),
    ("1x  / risk 50% / cap+rail 100%",   1, 0.50, 1.00, 1.00),
]


def main() -> None:
    assets = {label: _load(SYMBOLS[label], cfgpath)
              for label, cfgpath in PROFILES["standard"].items()}
    print(f"\nLeverage scenarios — clean OOS {HOLD_START} -> {HOLD_END}, "
          f"standard profile, real minimums, maker touch, 2bps slip\n")
    print(f"{'scenario':34s} {'compound':>9s} {'maxDD':>7s} {'trades':>7s} "
          f"{'win%':>6s}")
    for label, lev, risk, mcap, rail in SCENARIOS:
        for item in assets.values():
            cfg = item.config
            cfg.trading.leverage_tiers[cfg.trading.active_tier].leverage = lev
            cfg.position_sizing.risk_pct_per_trade = risk
            cfg.position_sizing.global_max_margin_pct = mcap
            cfg.position_sizing.max_position_pct = rail
        res = simulate_multi(assets, HOLD_START, HOLD_END, slip=.0002,
                             exit_granularity="sub", strat=dict(MINS))
        compound = max(.01, 1 + res.return_pct / 100)
        print(f"{label:34s} {compound:8.2f}x {res.max_dd_pct:6.1f}% "
              f"{res.trades:7d} {res.win_rate:5.1f}%")


if __name__ == "__main__":
    main()
