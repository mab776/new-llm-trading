"""
Trailing-stop math — the single source of truth shared by the backtest engine and live
trading, so the two paths can never drift apart.

A trailing stop only ever moves in the trade's favour:
  - LONG:  activates once price has risen `activation_pct` above entry, then trails
           `callback_pct` below the highest price seen; the stop only ratchets UP.
  - SHORT: activates once price has fallen `activation_pct` below entry, then trails
           `callback_pct` above the lowest price seen; the stop only ratchets DOWN.

`activation_pct` and `callback_pct` are percentages of the entry price (e.g. 1.0 = 1%).
"""

from __future__ import annotations

from typing import Optional


def compute_trailing_stop(
    direction: str,          # "LONG" or "SHORT"
    entry_price: float,
    favorable_extreme: float,  # bar high for LONG, bar low for SHORT (or current price live)
    current_sl: float,
    activation_pct: float,
    callback_pct: float,
) -> Optional[float]:
    """
    Return a new stop-loss price if the trailing stop should move, else None.

    Never returns a stop that moves against the position (long: only up; short: only down).
    """
    is_long = direction == "LONG"
    activation_distance = entry_price * activation_pct / 100.0
    callback_distance = entry_price * callback_pct / 100.0

    if is_long:
        # Only trail once price has moved far enough in our favour.
        if favorable_extreme >= entry_price + activation_distance:
            new_sl = favorable_extreme - callback_distance
            if new_sl > current_sl:
                return new_sl
    else:
        if favorable_extreme <= entry_price - activation_distance:
            new_sl = favorable_extreme + callback_distance
            if new_sl < current_sl:
                return new_sl

    return None
