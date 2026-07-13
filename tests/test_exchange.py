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


class TestAvailableBalance:
    def test_dry_run_returns_default(self, dry_run_client):
        assert dry_run_client.get_available_balance(dry_run_default=250.0) == 250.0

    def test_parses_usdt_account(self, monkeypatch):
        client = BitgetClient(BitgetConfig(api_key="k", api_secret="s", passphrase="p"))
        monkeypatch.setattr(client, "_dry_run", False)
        monkeypatch.setattr(
            client, "get_account_info",
            lambda: {"data": [{"marginCoin": "USDT", "available": "1234.5"}]},
        )
        assert client.get_available_balance() == 1234.5

    def test_unparseable_returns_zero(self, monkeypatch):
        client = BitgetClient(BitgetConfig(api_key="k", api_secret="s", passphrase="p"))
        monkeypatch.setattr(client, "_dry_run", False)
        monkeypatch.setattr(client, "get_account_info", lambda: {"data": []})
        assert client.get_available_balance() == 0.0

    def test_parses_account_equity(self, monkeypatch):
        client = BitgetClient(BitgetConfig(api_key="k", api_secret="s", passphrase="p"))
        monkeypatch.setattr(
            client, "get_account_info",
            lambda: {"data": [{"marginCoin": "USDT", "accountEquity": "4321.5"}]},
        )
        assert client.get_account_equity() == 4321.5


class TestExposureQueries:
    def test_pending_orders_keep_only_exposure_adding_orders(self, monkeypatch):
        client = BitgetClient(BitgetConfig(api_key="k", api_secret="s", passphrase="p"))
        monkeypatch.setattr(client, "_request", lambda *a, **k: {"data": {
            "entrustedList": [
                {"orderId": "open", "symbol": "BTCUSDT", "size": "2",
                 "baseVolume": ".5", "price": "100", "leverage": "10",
                 "tradeSide": "open", "reduceOnly": "NO", "posSide": "long"},
                {"orderId": "close", "symbol": "BTCUSDT", "size": "1",
                 "price": "110", "leverage": "10", "tradeSide": "close"},
            ]
        }})
        orders = client.get_pending_orders()
        assert [o.order_id for o in orders] == ["open"]
        assert orders[0].filled_size == .5

    def test_position_history_returns_list(self, monkeypatch):
        client = BitgetClient(BitgetConfig(api_key="k", api_secret="s", passphrase="p"))
        monkeypatch.setattr(client, "_request", lambda *a, **k: {
            "data": {"list": [{"positionId": "x", "netProfit": "2"}]}
        })
        assert client.get_position_history("BTCUSDT")[0]["positionId"] == "x"


class TestModifyStopLoss:
    def test_rejects_invalid_stop(self, dry_run_client):
        with pytest.raises(SafetyViolation):
            dry_run_client.modify_stop_loss("BTC-USDT", "long", 0.01, 0)

    def test_dry_run_is_noop_ok(self, dry_run_client):
        # Dry-run returns the canned response without raising.
        res = dry_run_client.modify_stop_loss("BTC-USDT", "long", 0.01, 49500)
        assert res.get("code") == "00000"

    def test_testnet_adds_paptrading_header(self, dry_run_client):
        headers = dry_run_client._headers("POST", "/x", "")
        assert headers.get("paptrading") == "1"
