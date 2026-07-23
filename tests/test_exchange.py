"""
Tests for the exchange module (safety checks).
"""

import hashlib
import hmac
import base64
import math
import requests

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

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_values_are_rejected(self, dry_run_client, bad):
        targets = TradeTargets(
            entry=50000, stop_loss=bad, take_profit_1=52000,
            take_profit_2=54000, risk_amount=1000, reward_1=2000,
            reward_2=4000, direction=Direction.BULLISH,
        )
        with pytest.raises(SafetyViolation):
            dry_run_client.place_order("BTC-USDT", "buy", 0.001, targets, 5)

    def test_wrong_side_targets_are_rejected(self, dry_run_client):
        targets = TradeTargets(
            entry=50000, stop_loss=51000, take_profit_1=49000,
            take_profit_2=48000, risk_amount=1000, reward_1=2000,
            reward_2=4000, direction=Direction.BULLISH,
        )
        with pytest.raises(SafetyViolation, match="wrong side"):
            dry_run_client.place_order("BTC-USDT", "buy", 0.001, targets, 5)


class TestRequestSigning:
    def test_get_signs_exact_query_string_sent(self, monkeypatch):
        client = BitgetClient(BitgetConfig(
            api_key="key", api_secret="secret", passphrase="pass", testnet=False,
        ))
        monkeypatch.setattr("llm_trading_bot.exchange.time.time", lambda: 1.234)
        captured = {}

        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                return {"code": "00000", "data": {}}

        def fake_get(url, **kwargs):
            captured.update(url=url, **kwargs)
            return Response()

        monkeypatch.setattr("llm_trading_bot.exchange.requests.get", fake_get)
        client._request("GET", "/api/test", params={"symbol": "BTCUSDT", "limit": "20"})

        request_path = "/api/test?symbol=BTCUSDT&limit=20"
        expected = base64.b64encode(hmac.new(
            b"secret", f"1234GET{request_path}".encode(), hashlib.sha256,
        ).digest()).decode()
        assert captured["url"].endswith(request_path)
        assert captured["headers"]["ACCESS-SIGN"] == expected
        assert "params" not in captured

    def test_timeout_recovers_accepted_order_by_client_oid(self, monkeypatch):
        client = BitgetClient(BitgetConfig())
        targets = TradeTargets(100, 90, 110, 120, 10, 10, 20, Direction.BULLISH)
        monkeypatch.setattr(client, "set_leverage", lambda *a, **k: {})
        calls = []

        def fake_request(method, path, **kwargs):
            calls.append((method, path, kwargs))
            if method == "POST":
                raise requests.Timeout("response lost")
            return {"data": {"orderId": "accepted", "clientOid": "stable"}}

        monkeypatch.setattr(client, "_request", fake_request)
        result = client.place_order(
            "BTCUSDT", "buy", 1, targets, 5, client_oid="stable",
        )
        assert result.order_id == "accepted"
        assert calls[-1][2]["params"]["clientOid"] == "stable"

    def test_definite_rejection_never_recovers_stale_order(self, monkeypatch):
        """HTTP 4xx means Bitget REJECTED the request (e.g. duplicate
        clientOid). Recovery must not adopt whatever order currently holds
        that oid — that resurrected a filled order from an earlier bar and
        double-counted its fill (phantom fills, 2026-07-23)."""
        from types import SimpleNamespace
        client = BitgetClient(BitgetConfig())
        targets = TradeTargets(100, 90, 110, 120, 10, 10, 20, Direction.BULLISH)
        monkeypatch.setattr(client, "set_leverage", lambda *a, **k: {})
        lookups = []

        def fake_request(method, path, **kwargs):
            if method == "POST":
                raise requests.HTTPError(
                    "400 duplicate clientOid",
                    response=SimpleNamespace(status_code=400),
                )
            lookups.append(path)
            return {"data": {"orderId": "stale-zombie", "clientOid": "stable"}}

        monkeypatch.setattr(client, "_request", fake_request)
        with pytest.raises(requests.HTTPError):
            client.place_order(
                "BTCUSDT", "buy", 1, targets, 5, client_oid="stable",
            )
        assert lookups == []  # definite rejection → no clientOid lookup

    def test_gateway_5xx_still_recovers_by_client_oid(self, monkeypatch):
        """A gateway 5xx leaves the outcome unknown — the order may have been
        accepted — so clientOid recovery must still apply there."""
        from types import SimpleNamespace
        client = BitgetClient(BitgetConfig())
        targets = TradeTargets(100, 90, 110, 120, 10, 10, 20, Direction.BULLISH)
        monkeypatch.setattr(client, "set_leverage", lambda *a, **k: {})

        def fake_request(method, path, **kwargs):
            if method == "POST":
                raise requests.HTTPError(
                    "502 bad gateway",
                    response=SimpleNamespace(status_code=502),
                )
            return {"data": {"orderId": "accepted", "clientOid": "stable"}}

        monkeypatch.setattr(client, "_request", fake_request)
        result = client.place_order(
            "BTCUSDT", "buy", 1, targets, 5, client_oid="stable",
        )
        assert result.order_id == "accepted"


class TestRestContractSemantics:
    def test_private_symbols_are_canonicalized(self, monkeypatch):
        client = BitgetClient(BitgetConfig())
        bodies = []
        monkeypatch.setattr(client, "_request", lambda *a, **k: bodies.append(k["body"]) or {})
        client.set_leverage("BTC/USDT:USDT", 5)
        assert bodies[0]["symbol"] == "BTCUSDT"

    def test_one_way_close_is_reduce_only(self, monkeypatch):
        client = BitgetClient(BitgetConfig(position_mode="one_way"))
        captured = {}
        monkeypatch.setattr(client, "_request", lambda *a, **k: captured.update(k["body"]) or {})
        client.close_position("BTC-USDT", "long", 0.01)
        assert captured["symbol"] == "BTCUSDT"
        assert captured["side"] == "sell"
        assert captured["reduceOnly"] == "YES"
        assert "tradeSide" not in captured

    def test_hedge_close_uses_bitget_close_pair(self, monkeypatch):
        client = BitgetClient(BitgetConfig(position_mode="hedge"))
        captured = {}
        monkeypatch.setattr(client, "_request", lambda *a, **k: captured.update(k["body"]) or {})
        client.close_position("BTC-USDT", "long", 0.01)
        assert captured["side"] == "buy"
        assert captured["tradeSide"] == "close"

    def test_contract_precision_and_minimums_are_enforced(self, monkeypatch):
        client = BitgetClient(BitgetConfig(
            api_key="k", api_secret="s", passphrase="p", position_mode="one_way",
        ))
        calls = []

        def fake_request(method, path, **kwargs):
            calls.append((method, path, kwargs))
            if path.endswith("/contracts"):
                return {"data": [{
                    "symbol": "BTCUSDT", "symbolStatus": "normal",
                    "pricePlace": "1", "priceEndStep": "1",
                    "volumePlace": "3", "sizeMultiplier": "0.001",
                    "minTradeNum": "0.001", "minTradeUSDT": "5",
                    "maxMarketOrderQty": "100", "maxOrderQty": "200",
                    "minLever": "1", "maxLever": "125",
                }]}
            return {"data": {"orderId": "x"}}

        monkeypatch.setattr(client, "_request", fake_request)
        targets = TradeTargets(
            50000.08, 49000.01, 52000.09, 54000.09,
            1000, 2000, 4000, Direction.BULLISH,
        )
        result = client.place_order(
            "BTC-USDT", "buy", 0.0019, targets, 5,
            order_type="limit", price=50000.08, client_oid="stable-id",
        )
        body = calls[-1][2]["body"]
        assert body["size"] == "0.001"
        assert body["price"] == "50000.0"
        assert body["presetStopLossPrice"] == "49000.1"  # tightened upward
        assert body["presetStopSurplusPrice"] == "54000.0"  # TP2 safety net
        assert body["clientOid"] == "stable-id"
        assert result.size == 0.001

    def test_contract_minimum_rejects_before_leverage_or_order(self, monkeypatch):
        client = BitgetClient(BitgetConfig(api_key="k", api_secret="s", passphrase="p"))
        calls = []

        def fake_request(method, path, **kwargs):
            calls.append(path)
            return {"data": [{
                "symbol": "BTCUSDT", "symbolStatus": "normal",
                "pricePlace": "1", "priceEndStep": "1", "volumePlace": "3",
                "sizeMultiplier": "0.001", "minTradeNum": "0.001",
                "minTradeUSDT": "50", "maxMarketOrderQty": "100",
                "maxOrderQty": "200", "minLever": "1", "maxLever": "125",
            }]}

        monkeypatch.setattr(client, "_request", fake_request)
        targets = TradeTargets(100, 90, 110, 120, 10, 10, 20, Direction.BULLISH)
        with pytest.raises(SafetyViolation, match="notional"):
            client.place_order("BTCUSDT", "buy", 0.01, targets, 5)
        assert calls == ["/api/v2/mix/market/contracts"]


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

    def test_one_way_pending_buy_is_long_not_net(self, monkeypatch):
        client = BitgetClient(BitgetConfig(api_key="k", api_secret="s", passphrase="p"))
        monkeypatch.setattr(client, "_request", lambda *a, **k: {"data": {
            "entrustedList": [{"orderId": "x", "symbol": "BTCUSDT", "size": "1",
                                "price": "100", "side": "buy", "posSide": "net"}]
        }})
        assert client.get_pending_orders()[0].side == "long"

    def test_position_history_returns_list(self, monkeypatch):
        client = BitgetClient(BitgetConfig(api_key="k", api_secret="s", passphrase="p"))
        monkeypatch.setattr(client, "_request", lambda *a, **k: {
            "data": {"list": [{"positionId": "x", "netProfit": "2"}]}
        })
        assert client.get_position_history("BTCUSDT")[0]["positionId"] == "x"

    def test_exact_order_fills_include_fees_and_timestamps(self, monkeypatch):
        client = BitgetClient(BitgetConfig(
            api_key="k", api_secret="s", passphrase="p", testnet=False,
        ))
        monkeypatch.setattr(client, "_request", lambda *a, **k: {"data": {
            "fillList": [{
                "tradeId": "t1", "orderId": "o1", "symbol": "BTCUSDT",
                "price": "100", "baseVolume": "0.7", "side": "buy",
                "cTime": "123", "feeDetail": [{"totalFee": "-0.014"}],
            }]
        }})
        fill = client.get_order_fills("BTC-USDT", "o1")[0]
        assert fill.trade_id == "t1"
        assert fill.size == 0.7
        assert fill.fee == -0.014
        assert fill.timestamp_ms == 123


class TestModifyStopLoss:
    def test_rejects_invalid_stop(self, dry_run_client):
        with pytest.raises(SafetyViolation):
            dry_run_client.modify_stop_loss("BTC-USDT", "long", 0.01, 0)

    def test_dry_run_is_noop_ok(self, dry_run_client):
        # Dry-run returns the canned response without raising.
        res = dry_run_client.modify_stop_loss(
            "BTC-USDT", "long", 0.01, 49500, plan_order_id="plan-1"
        )
        assert res.get("code") == "00000"

    def test_rejects_missing_existing_plan_id(self, dry_run_client):
        with pytest.raises(SafetyViolation, match="plan order ID"):
            dry_run_client.modify_stop_loss("BTC-USDT", "long", 0.01, 49500)

    def test_modifies_existing_plan_instead_of_placing_another(self, dry_run_client, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            dry_run_client, "_request",
            lambda method, path, **kwargs: captured.update(
                method=method, path=path, body=kwargs["body"]
            ) or {"code": "00000"},
        )
        dry_run_client.modify_stop_loss(
            "BTC-USDT", "long", 0.01, 49500, plan_order_id="plan-1"
        )
        assert captured["path"].endswith("modify-tpsl-order")
        assert captured["body"]["orderId"] == "plan-1"
        assert captured["body"]["size"] == ""

    def test_testnet_adds_paptrading_header(self, dry_run_client):
        headers = dry_run_client._headers("POST", "/x", "")
        assert headers.get("paptrading") == "1"

    def test_per_lot_tpsl_uses_size_and_one_way_hold_side(
        self, dry_run_client, monkeypatch,
    ):
        captured = []
        monkeypatch.setattr(
            dry_run_client, "_request",
            lambda method, path, **kwargs:
            captured.append((path, kwargs["body"]))
            or {"data": {"orderId": "plan-1", "clientOid": "lot-sl"}},
        )
        plan = dry_run_client.place_tpsl_order(
            "BTC-USDT", "long", 0.7, 49000, "loss_plan", "lot-sl",
        )
        assert plan.order_id == "plan-1"
        assert captured[0][1]["holdSide"] == "buy"
        assert captured[0][1]["size"] == "0.7"
        dry_run_client.modify_stop_loss(
            "BTC-USDT", "long", 0.3, 50000,
            plan_order_id="plan-1", position_level=False,
        )
        assert captured[1][1]["size"] == "0.3"


def test_get_positions_filters_by_symbol(monkeypatch):
    """Bitget's all-position endpoint ignores the symbol param; the client must
    filter client-side so a per-symbol caller never sees another symbol's position."""
    client = BitgetClient(BitgetConfig(
        api_key="k", api_secret="s", passphrase="p",
    ))
    payload = {"data": [
        {"symbol": "BTCUSDT", "holdSide": "long", "total": "0.0017",
         "openPriceAvg": "64000", "unrealizedPL": "0", "leverage": "25",
         "marginMode": "isolated", "marginSize": "4.35"},
        {"symbol": "ETHUSDT", "holdSide": "long", "total": "0.01",
         "openPriceAvg": "1900", "unrealizedPL": "0", "leverage": "25",
         "marginMode": "isolated", "marginSize": "0.76"},
    ]}
    monkeypatch.setattr(client, "_request", lambda *a, **k: payload)
    eth = client.get_positions("ETH-USDT")
    assert [p.symbol for p in eth] == ["ETHUSDT"]
    btc = client.get_positions("BTC/USDT:USDT")
    assert [p.symbol for p in btc] == ["BTCUSDT"]
    assert {p.symbol for p in client.get_positions()} == {"BTCUSDT", "ETHUSDT"}
