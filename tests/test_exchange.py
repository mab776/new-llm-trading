"""
Tests for the exchange module (safety checks).
"""

import pytest

from llm_trading_bot.config import BitgetConfig
from llm_trading_bot.exchange import BitgetClient, SafetyViolation
from llm_trading_bot.scoring import Direction, TradeTargets


@pytest.fixture
def dry_run_client() -> BitgetClient:
    """Client with no credentials — runs in dry-run mode."""
    return BitgetClient(BitgetConfig(testnet=True))


class TestSafetyChecks:
    def test_no_sl_raises(self, dry_run_client):
        targets = TradeTargets(
            entry=50000, stop_loss=0, take_profit_1=52000,
            take_profit_2=54000, risk_amount=1000, reward_1=2000,
            reward_2=4000, direction=Direction.BULLISH,
        )
        with pytest.raises(SafetyViolation, match="stop loss"):
            dry_run_client.place_order("BTC-USDT", "buy", 0.001, targets, 5)

    def test_no_tp_raises(self, dry_run_client):
        targets = TradeTargets(
            entry=50000, stop_loss=49000, take_profit_1=0,
            take_profit_2=54000, risk_amount=1000, reward_1=0,
            reward_2=4000, direction=Direction.BULLISH,
        )
        with pytest.raises(SafetyViolation, match="take profit"):
            dry_run_client.place_order("BTC-USDT", "buy", 0.001, targets, 5)

    def test_none_sl_raises(self, dry_run_client):
        targets = TradeTargets(
            entry=50000, stop_loss=None, take_profit_1=52000,
            take_profit_2=54000, risk_amount=1000, reward_1=2000,
            reward_2=4000, direction=Direction.BULLISH,
        )
        with pytest.raises(SafetyViolation):
            dry_run_client.place_order("BTC-USDT", "buy", 0.001, targets, 5)

    def test_valid_order_succeeds(self, dry_run_client):
        targets = TradeTargets(
            entry=50000, stop_loss=49000, take_profit_1=52000,
            take_profit_2=54000, risk_amount=1000, reward_1=2000,
            reward_2=4000, direction=Direction.BULLISH,
        )
        result = dry_run_client.place_order("BTC-USDT", "buy", 0.001, targets, 5)
        assert result.stop_loss == 49000
        assert result.take_profit_1 == 52000
        assert result.status == "submitted"
