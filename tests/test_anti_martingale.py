"""Focused tests for experimental anti-martingale sizing."""

import pytest

from llm_trading_bot.exposure import (
    anti_martingale_multiplier, update_outcome_streak,
)


def test_signed_outcome_streak_resets_when_outcome_flips():
    streak = 0
    for won in (True, True, True, False, False, True):
        streak = update_outcome_streak(streak, won)
    assert streak == 1


@pytest.mark.parametrize(
    ("streak", "expected"),
    [(-4, 0.5), (-2, 0.6), (-1, 0.8), (0, 1.0),
     (1, 1.2), (2, 1.4), (4, 1.5)],
)
def test_anti_martingale_multiplier_is_bounded(streak, expected):
    assert anti_martingale_multiplier(streak, 0.2, 0.5, 1.5) == pytest.approx(expected)


def test_zero_step_preserves_baseline_for_any_streak():
    assert anti_martingale_multiplier(-20, 0.0) == 1.0
    assert anti_martingale_multiplier(20, 0.0) == 1.0
