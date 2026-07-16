"""Risk-% and exchange-minimum sizing scenarios at $100 starting balance.

Question (Marc, 2026-07-16): with only ~$100 in the account, should
risk_pct_per_trade rise from 2% to 4%/8%, or should sub-minimum orders be
floored up to the exchange minimum? Run every option on the clean OOS window
(2025-06 -> 2026-04) with the real Bitget contract minimums and report.

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.sizing_scenarios
"""
from __future__ import annotations

from opt.holdout_oos import HOLD_START, HOLD_END, PROFILES, SYMBOLS, _load
from opt.multi_asset import simulate_multi

# Real Bitget USDT-FUTURES contract minimums (fetched 2026-07-16):
# BTCUSDT minTradeNum 0.0001 (~$6.5), ETHUSDT 0.01 (~$19), SOLUSDT 0.1 (~$7.7).
MIN_QTY = {"BTC": 0.0001, "ETH": 0.01, "SOL": 0.1}
SIZE_STEP = {"BTC": 0.0001, "ETH": 0.01, "SOL": 0.1}

SCENARIOS = [
    # (label, risk_pct, strat overrides)
    ("2% no-min (reference)", 0.02, {}),
    ("2% + mins, skip",       0.02, {"min_qty": MIN_QTY, "size_step": SIZE_STEP}),
    ("2% + mins, floor",      0.02, {"min_qty": MIN_QTY, "size_step": SIZE_STEP,
                                     "min_size_policy": "floor"}),
    ("4% + mins, skip",       0.04, {"min_qty": MIN_QTY, "size_step": SIZE_STEP}),
    ("8% + mins, skip",       0.08, {"min_qty": MIN_QTY, "size_step": SIZE_STEP}),
    # honest scale-up: 4% per-trade with the portfolio caps doubled to match
    ("4% + caps x2 + mins",   0.04, {"min_qty": MIN_QTY, "size_step": SIZE_STEP,
                                     "global_max_margin_pct": 0.088,
                                     "global_max_notional_pct": 2.2}),
]


def main() -> None:
    assets = {label: _load(SYMBOLS[label], cfgpath)
              for label, cfgpath in PROFILES["standard"].items()}
    base_risk = {label: item.config.position_sizing.risk_pct_per_trade
                 for label, item in assets.items()}
    print(f"\nClean OOS {HOLD_START} -> {HOLD_END}, standard profile, "
          f"initial balance $100, real Bitget minimums\n")
    print(f"{'scenario':26s} {'compound':>9s} {'maxDD':>7s} {'trades':>7s} "
          f"{'skips':>6s} {'floors':>7s}")
    for label, risk, overrides in SCENARIOS:
        for item in assets.values():
            item.config.position_sizing.risk_pct_per_trade = risk
        counters = {"skips": 0, "floors": 0}
        strat = dict(overrides)
        strat["_min_counters"] = counters  # shared nested dict survives the copy
        res = simulate_multi(assets, HOLD_START, HOLD_END, slip=.0002,
                             exit_granularity="sub", strat=strat)
        for sym, item in assets.items():  # restore
            item.config.position_sizing.risk_pct_per_trade = base_risk[sym]
        print(f"{label:26s} {max(.01, 1 + res.return_pct / 100):>8.2f}x "
              f"{res.max_dd_pct:>6.1f}% {res.trades:>7d} "
              f"{counters['skips']:>6d} {counters['floors']:>7d}")


if __name__ == "__main__":
    main()
