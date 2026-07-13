"""Shared exposure and outcome-streak math tests."""

import pytest

from llm_trading_bot.exposure import cap_risk_pct, outcome_streak


def test_outcome_streak_uses_latest_consecutive_side():
    assert outcome_streak([1, 2, -1, -2, -3]) == -3


def test_cap_risk_pct_uses_tighter_margin_or_notional_capacity():
    got = cap_risk_pct(
        .02, 25, 1000, committed_margin=40, committed_notional=1000,
        max_margin_pct=.05, max_notional_pct=1.1,
    )
    assert got == pytest.approx(.004)  # $100 remaining notional / (1000 * 25)
