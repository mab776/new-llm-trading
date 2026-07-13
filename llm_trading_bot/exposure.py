"""Shared causal position-sizing and portfolio-exposure math.

Used by the full backtest, fast/shared harness, and live scheduler so the
validated ex-ante caps cannot drift between execution paths.
"""

from __future__ import annotations


def update_outcome_streak(streak: int, won: bool) -> int:
    """Return the signed consecutive closed-trade streak (wins +, losses -)."""
    if won:
        return streak + 1 if streak > 0 else 1
    return streak - 1 if streak < 0 else -1


def outcome_streak(net_profits: list[float]) -> int:
    """Calculate the current streak from outcomes ordered oldest to newest."""
    streak = 0
    for profit in net_profits:
        streak = update_outcome_streak(streak, profit > 0)
    return streak


def anti_martingale_multiplier(streak: int, step: float,
                               minimum: float = 0.5,
                               maximum: float = 1.5) -> float:
    """Causal risk multiplier for the streak known before an entry is placed."""
    if step <= 0:
        return 1.0
    return max(minimum, min(maximum, 1.0 + streak * step))


def cap_risk_pct(risk_pct: float, leverage: int, equity: float,
                 committed_margin: float, committed_notional: float,
                 *, risk_multiplier: float = 1.0,
                 max_margin_pct: float = 0.0,
                 max_notional_pct: float = 0.0) -> float:
    """Scale proposed margin risk to remaining ex-ante portfolio capacity.

    Zero caps disable that dimension. The function only reduces a new order; it
    never changes or closes an existing position.
    """
    if equity <= 0 or risk_pct <= 0 or leverage <= 0 or risk_multiplier <= 0:
        return 0.0
    risk_pct *= risk_multiplier
    if max_margin_pct > 0:
        remaining_margin = equity * max_margin_pct - committed_margin
        risk_pct = min(risk_pct, max(0.0, remaining_margin / equity))
    if max_notional_pct > 0:
        remaining_notional = equity * max_notional_pct - committed_notional
        risk_pct = min(
            risk_pct, max(0.0, remaining_notional / (equity * leverage))
        )
    return max(0.0, risk_pct)
