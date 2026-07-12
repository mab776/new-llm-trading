"""
Bitget exchange integration — futures trading with mandatory TP/SL.

SAFETY: This module REFUSES to place any order without a stop loss and take profit.
This is the most important safety feature of the entire system.

Supports:
- Testnet and mainnet
- USDT-M futures
- Market and limit orders with attached TP/SL
- Trailing stops
- Position management
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

from llm_trading_bot.config import BitgetConfig, TrailingStopConfig
from llm_trading_bot.scoring import Direction, TradeTargets


# ──────────────────────────────────────────────────────────────────────
# Safety Exception
# ──────────────────────────────────────────────────────────────────────

class SafetyViolation(Exception):
    """Raised when a trade would violate safety rules."""
    pass


class ExchangeError(Exception):
    """Raised for exchange API errors."""
    pass


# ──────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    size: float
    price: Optional[float]
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    status: str
    timestamp: str
    raw_response: dict


@dataclass
class Position:
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: int
    margin_mode: str
    timestamp: str


# ──────────────────────────────────────────────────────────────────────
# Bitget Client
# ──────────────────────────────────────────────────────────────────────

class BitgetClient:
    """
    Bitget Futures API client with mandatory safety checks.

    SAFETY INVARIANT: No order is EVER placed without TP and SL.
    Any attempt to bypass this raises SafetyViolation.
    """

    # Bitget demo/paper trading runs on the SAME REST host as production; it is selected
    # per-request with the `paptrading: 1` header (plus demo API keys), NOT a separate URL.
    MAINNET_URL = "https://api.bitget.com"
    TESTNET_URL = "https://api.bitget.com"

    def __init__(self, config: BitgetConfig):
        self.config = config
        self.base_url = self.TESTNET_URL if config.testnet else self.MAINNET_URL

        if not config.api_key or not config.api_secret:
            print("⚠ Bitget credentials not configured — dry-run mode only")
            self._dry_run = True
        else:
            self._dry_run = False

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate HMAC-SHA256 signature for Bitget API."""
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.config.api_secret.encode(), message.encode(), hashlib.sha256
        ).digest()
        import base64
        return base64.b64encode(signature).decode()

    def _headers(self, method: str, path: str, body: str = "") -> dict:
        """Build authenticated headers."""
        timestamp = str(int(time.time() * 1000))
        headers = {
            "ACCESS-KEY": self.config.api_key,
            "ACCESS-SIGN": self._sign(timestamp, method, path, body),
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.config.passphrase,
            "Content-Type": "application/json",
            "locale": "en-US",
        }
        # Route requests to Bitget's demo/paper environment when testnet is enabled.
        if self.config.testnet:
            headers["paptrading"] = "1"
        return headers

    def _request(self, method: str, path: str, params: dict = None, body: dict = None) -> dict:
        """Make an authenticated API request."""
        if self._dry_run:
            print(f"[DRY RUN] {method} {path}")
            if body:
                print(f"  Body: {json.dumps(body, indent=2)}")
            return {"code": "00000", "msg": "dry_run", "data": {}}

        url = self.base_url + path
        body_str = json.dumps(body) if body else ""
        headers = self._headers(method, path, body_str)

        if method == "GET":
            resp = requests.get(url, params=params, headers=headers, timeout=10)
        else:
            resp = requests.post(url, data=body_str, headers=headers, timeout=10)

        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "00000":
            raise ExchangeError(f"Bitget API error: {data.get('msg', 'Unknown')} (code: {data.get('code')})")

        return data

    # ── Safety Validation ──

    def _validate_order_safety(
        self,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> None:
        """
        MANDATORY safety check. Raises SafetyViolation if TP or SL is missing.
        This is THE most important function in the entire codebase.
        """
        if stop_loss is None or stop_loss <= 0:
            raise SafetyViolation(
                "REFUSED: Cannot place order without a stop loss. "
                "This is a non-negotiable safety rule."
            )
        if take_profit is None or take_profit <= 0:
            raise SafetyViolation(
                "REFUSED: Cannot place order without a take profit. "
                "This is a non-negotiable safety rule."
            )

    # ── Trading Operations ──

    def set_leverage(self, symbol: str, leverage: int, side: str = "long") -> dict:
        """Set leverage for a symbol."""
        path = "/api/v2/mix/account/set-leverage"
        body = {
            "symbol": symbol,
            "productType": self.config.product_type,
            "marginCoin": "USDT",
            "leverage": str(leverage),
            "holdSide": side,
        }
        return self._request("POST", path, body=body)

    def place_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        size: float,
        targets: TradeTargets,
        leverage: int,
        order_type: str = "market",
        price: Optional[float] = None,
    ) -> OrderResult:
        """
        Place a futures order with MANDATORY TP/SL.

        This method ALWAYS validates that TP and SL are present.
        There is NO parameter or flag to bypass this check.
        """
        # SAFETY CHECK — absolutely non-negotiable
        self._validate_order_safety(targets.stop_loss, targets.take_profit_1)

        # Set leverage
        hold_side = "long" if side == "buy" else "short"
        self.set_leverage(symbol, leverage, hold_side)

        # Determine trade side for Bitget API
        trade_side = "open"

        path = "/api/v2/mix/order/place-order"
        body = {
            "symbol": symbol,
            "productType": self.config.product_type,
            "marginMode": "crossed",
            "marginCoin": "USDT",
            "size": str(size),
            "side": side,
            "tradeSide": trade_side,
            "orderType": order_type,
            "presetStopSurplusPrice": str(targets.take_profit_1),
            "presetStopLossPrice": str(targets.stop_loss),
        }

        if order_type == "limit" and price is not None:
            body["price"] = str(price)

        result = self._request("POST", path, body=body)

        order_id = result.get("data", {}).get("orderId", "dry_run_id")

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            stop_loss=targets.stop_loss,
            take_profit_1=targets.take_profit_1,
            take_profit_2=targets.take_profit_2,
            status="submitted",
            timestamp=datetime.now().isoformat(),
            raw_response=result,
        )

    def place_trailing_stop(
        self,
        symbol: str,
        side: str,
        size: float,
        trailing_config: TrailingStopConfig,
    ) -> dict:
        """Place a trailing stop order."""
        path = "/api/v2/mix/order/place-tpsl-order"
        body = {
            "symbol": symbol,
            "productType": self.config.product_type,
            "marginCoin": "USDT",
            "planType": "moving_plan",
            "triggerType": "fill_price",
            "size": str(size),
            "side": side,
            "callbackRatio": str(trailing_config.callback_pct),
        }
        return self._request("POST", path, body=body)

    def modify_stop_loss(
        self,
        symbol: str,
        hold_side: str,   # "long" or "short"
        size: float,
        new_sl: float,
    ) -> dict:
        """
        Move the position's stop loss to ``new_sl`` (a position-level TPSL update).

        Used by live trailing stops. Goes through ``_request`` so it is a no-op in
        dry-run. The caller is responsible for only ever moving the stop in the trade's
        favour (see llm_trading_bot.trailing.compute_trailing_stop).
        """
        if new_sl is None or new_sl <= 0:
            raise SafetyViolation("REFUSED: modify_stop_loss called without a valid stop price.")

        path = "/api/v2/mix/order/place-tpsl-order"
        body = {
            "symbol": symbol,
            "productType": self.config.product_type,
            "marginCoin": "USDT",
            "planType": "pos_loss",       # position-level stop loss
            "triggerType": "fill_price",
            "triggerPrice": str(new_sl),
            "holdSide": hold_side,
            "size": str(size),
        }
        return self._request("POST", path, body=body)

    def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Get current open positions."""
        path = "/api/v2/mix/position/all-position"
        params = {"productType": self.config.product_type}
        if symbol:
            params["symbol"] = symbol

        result = self._request("GET", path, params=params)
        positions = []

        for pos_data in result.get("data", []):
            if float(pos_data.get("total", 0)) > 0:
                positions.append(Position(
                    symbol=pos_data.get("symbol", ""),
                    side=pos_data.get("holdSide", ""),
                    size=float(pos_data.get("total", 0)),
                    entry_price=float(pos_data.get("openPriceAvg", 0)),
                    unrealized_pnl=float(pos_data.get("unrealizedPL", 0)),
                    leverage=int(pos_data.get("leverage", 1)),
                    margin_mode=pos_data.get("marginMode", ""),
                    timestamp=datetime.now().isoformat(),
                ))

        return positions

    def close_position(self, symbol: str, side: str, size: float) -> dict:
        """Close a position."""
        close_side = "sell" if side == "long" else "buy"
        path = "/api/v2/mix/order/place-order"
        body = {
            "symbol": symbol,
            "productType": self.config.product_type,
            "marginMode": "crossed",
            "marginCoin": "USDT",
            "size": str(size),
            "side": close_side,
            "tradeSide": "close",
            "orderType": "market",
        }
        return self._request("POST", path, body=body)

    def get_account_info(self) -> dict:
        """Get account balance and info."""
        path = "/api/v2/mix/account/accounts"
        params = {"productType": self.config.product_type}
        return self._request("GET", path, params=params)

    def get_available_balance(self, dry_run_default: float = 100.0) -> float:
        """
        Return the available USDT balance for position sizing.

        In dry-run (no credentials) returns ``dry_run_default`` so live-dry-run still
        produces a realistic, non-hardcoded size. Falls back to 0.0 if the response
        can't be parsed (callers must guard against a zero/negative size).
        """
        if self._dry_run:
            return dry_run_default

        try:
            data = self.get_account_info().get("data", [])
            # Bitget returns a list of margin-coin accounts; find the USDT one.
            for acct in data:
                if acct.get("marginCoin") == "USDT":
                    return float(acct.get("available", acct.get("crossedMaxAvailable", 0)) or 0)
            if data:  # single-coin account shape
                acct = data[0]
                return float(acct.get("available", 0) or 0)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"⚠ Could not parse account balance: {e}")
        return 0.0
