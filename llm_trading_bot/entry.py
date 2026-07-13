"""Shared entry-order lifecycle primitives.

Maker entries are good for one primary bar: a limit is placed at the completed
decision bar's close, may fill during the following bar, and is otherwise
cancelled.  Backtest and fastbt both use :func:`maker_limit_touched` so the
fill rule cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PendingEntry:
    direction: str
    limit_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    leverage: int
    risk_pct: float
    tp1_exit_pct: float
    atr_at_entry: float | None = None
    decision_time: str = ""


def maker_limit_touched(
    direction: str, limit_price: float, bar_high: float, bar_low: float
) -> bool:
    """Return whether a resting entry limit trades during an OHLC bar."""
    if direction == "LONG":
        return bar_low <= limit_price
    if direction == "SHORT":
        return bar_high >= limit_price
    raise ValueError(f"Unknown entry direction: {direction}")
