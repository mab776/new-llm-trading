"""
Tests for the shared trailing-stop math (used by both backtest and live trading).
"""

from llm_trading_bot.trailing import compute_trailing_stop


class TestTrailingLong:
    def test_not_activated_below_activation(self):
        # Price only 0.5% up, activation is 1% -> no move.
        assert compute_trailing_stop("LONG", 100.0, 100.5, 98.0, 1.0, 0.5) is None

    def test_activates_and_trails_up(self):
        # Price 2% up; callback 0.5% -> new SL = 102 - 0.5 = 101.5 (> current 98).
        new_sl = compute_trailing_stop("LONG", 100.0, 102.0, 98.0, 1.0, 0.5)
        assert new_sl == 101.5

    def test_never_moves_down(self):
        # New computed SL (101.5) is below the current SL (101.8) -> keep current.
        assert compute_trailing_stop("LONG", 100.0, 102.0, 101.8, 1.0, 0.5) is None


class TestTrailingShort:
    def test_activates_and_trails_down(self):
        # Price 2% down; callback 0.5% -> new SL = 98 + 0.5 = 98.5 (< current 102).
        new_sl = compute_trailing_stop("SHORT", 100.0, 98.0, 102.0, 1.0, 0.5)
        assert new_sl == 98.5

    def test_never_moves_up(self):
        assert compute_trailing_stop("SHORT", 100.0, 98.0, 98.2, 1.0, 0.5) is None
