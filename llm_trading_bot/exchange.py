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
import math
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from datetime import datetime
from typing import Optional
from urllib.parse import urlencode

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
    margin_size: float = 0.0


@dataclass
class PendingOrder:
    """One exchange-wide resting futures entry used by exposure controls."""
    order_id: str
    symbol: str
    side: str
    size: float
    filled_size: float
    price: float
    leverage: int
    client_oid: str = ""
    created_at_ms: int = 0


@dataclass(frozen=True)
class PlanOrder:
    """Active or historical Bitget TP/SL plan."""
    order_id: str
    client_oid: str
    symbol: str
    plan_type: str
    side: str
    size: float
    trigger_price: float
    status: str
    created_at_ms: int = 0
    updated_at_ms: int = 0
    execute_order_id: str = ""
    filled_size: float = 0.0


@dataclass(frozen=True)
class Fill:
    trade_id: str
    order_id: str
    symbol: str
    price: float
    size: float
    fee: float
    timestamp_ms: int
    side: str


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    price_step: Decimal
    size_step: Decimal
    min_size: Decimal
    min_notional: Decimal
    max_market_size: Decimal
    max_limit_size: Decimal
    min_leverage: int
    max_leverage: int


# ──────────────────────────────────────────────────────────────────────
# Bitget Client
# ──────────────────────────────────────────────────────────────────────

def _is_definite_rejection(exc: requests.RequestException) -> bool:
    """True when the exchange definitely REJECTED the request (HTTP 4xx).

    clientOid-recovery exists for AMBIGUOUS failures — a lost response
    (timeout/connection error) or a gateway 5xx where the order may have been
    accepted. A definite 4xx (e.g. duplicate clientOid) is not ambiguous;
    recovering there can adopt a stale order from an earlier attempt/bar and
    double-count its fill (phantom fills, 2026-07-23).
    """
    response = getattr(exc, "response", None)
    return response is not None and 400 <= response.status_code < 500


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
        self._contract_specs: dict[str, ContractSpec] = {}

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

        method = method.upper()
        query = urlencode(params or {}, doseq=True)
        request_path = f"{path}?{query}" if query else path
        url = self.base_url + request_path
        body_str = json.dumps(body) if body else ""
        # Bitget signs the exact request target.  GET query parameters must therefore
        # be included in both the signature and the URL in identical order/encoding.
        headers = self._headers(method, request_path, body_str)

        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=10)
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
        *,
        entry: Optional[float] = None,
        size: Optional[float] = None,
        side: Optional[str] = None,
        take_profit_2: Optional[float] = None,
    ) -> None:
        """
        MANDATORY safety check. Raises SafetyViolation if TP or SL is missing.
        This is THE most important function in the entire codebase.
        """
        if stop_loss is None or not math.isfinite(stop_loss) or stop_loss <= 0:
            raise SafetyViolation(
                "REFUSED: Cannot place order without a stop loss. "
                "This is a non-negotiable safety rule."
            )
        if take_profit is None or not math.isfinite(take_profit) or take_profit <= 0:
            raise SafetyViolation(
                "REFUSED: Cannot place order without a take profit. "
                "This is a non-negotiable safety rule."
            )
        if entry is not None and (not math.isfinite(entry) or entry <= 0):
            raise SafetyViolation("REFUSED: Entry price must be finite and positive.")
        if size is not None and (not math.isfinite(size) or size <= 0):
            raise SafetyViolation("REFUSED: Order size must be finite and positive.")
        if take_profit_2 is not None and (
            not math.isfinite(take_profit_2) or take_profit_2 <= 0
        ):
            raise SafetyViolation("REFUSED: TP2 must be finite and positive.")
        if entry is not None and side == "buy":
            if not stop_loss < entry < take_profit:
                raise SafetyViolation("REFUSED: Long SL/entry/TP prices are on the wrong side.")
            if take_profit_2 is not None and take_profit_2 < take_profit:
                raise SafetyViolation("REFUSED: Long TP2 must not be below TP1.")
        if entry is not None and side == "sell":
            if not take_profit < entry < stop_loss:
                raise SafetyViolation("REFUSED: Short TP/entry/SL prices are on the wrong side.")
            if take_profit_2 is not None and take_profit_2 > take_profit:
                raise SafetyViolation("REFUSED: Short TP2 must not be above TP1.")

    @staticmethod
    def _rest_symbol(symbol: str) -> str:
        """Canonical Bitget V2 private REST symbol (e.g. BTC/USDT:USDT -> BTCUSDT)."""
        return symbol.split(":", 1)[0].replace("/", "").replace("-", "").upper()

    @staticmethod
    def _step_round(value: Decimal, step: Decimal, rounding: str) -> Decimal:
        if step <= 0:
            raise ExchangeError(f"Invalid contract step {step}")
        return (value / step).to_integral_value(rounding=rounding) * step

    def get_contract_spec(self, symbol: str) -> ContractSpec:
        """Load and cache Bitget's symbol-specific precision and trading limits."""
        symbol = self._rest_symbol(symbol)
        cached = self._contract_specs.get(symbol)
        if cached is not None:
            return cached
        if self._dry_run:
            raise ExchangeError("Contract metadata is unavailable in credential-free dry-run")

        result = self._request("GET", "/api/v2/mix/market/contracts", params={
            "productType": self.config.product_type,
            "symbol": symbol,
        })
        rows = result.get("data", [])
        row = next((item for item in rows if self._rest_symbol(item.get("symbol", "")) == symbol), None)
        if not row:
            raise ExchangeError(f"No Bitget contract metadata returned for {symbol}")
        if str(row.get("symbolStatus", "")).lower() != "normal":
            raise ExchangeError(f"Bitget contract {symbol} is not tradable: {row.get('symbolStatus')}")
        try:
            price_place = int(row["pricePlace"])
            price_step = Decimal(str(row.get("priceEndStep", "1"))).scaleb(-price_place)
            size_multiplier = row.get("sizeMultiplier")
            size_step = (
                Decimal(str(size_multiplier)) if size_multiplier
                else Decimal(1).scaleb(-int(row["volumePlace"]))
            )
            spec = ContractSpec(
                symbol=symbol,
                price_step=price_step,
                size_step=size_step,
                min_size=Decimal(str(row["minTradeNum"])),
                min_notional=Decimal(str(row["minTradeUSDT"])),
                max_market_size=Decimal(str(row.get("maxMarketOrderQty") or "Infinity")),
                max_limit_size=Decimal(str(row.get("maxOrderQty") or "Infinity")),
                min_leverage=int(row.get("minLever", 1)),
                max_leverage=int(row.get("maxLever", 1)),
            )
        except (KeyError, InvalidOperation, TypeError, ValueError) as exc:
            raise ExchangeError(f"Malformed Bitget contract metadata for {symbol}") from exc
        if spec.price_step <= 0 or spec.size_step <= 0 or spec.min_size <= 0:
            raise ExchangeError(f"Invalid Bitget contract limits for {symbol}")
        self._contract_specs[symbol] = spec
        return spec

    def get_ticker(self, symbol: str) -> dict:
        """Best bid/ask/last for a futures symbol (maker re-peg pricing)."""
        result = self._request("GET", "/api/v2/mix/market/ticker", params={
            "productType": self.config.product_type,
            "symbol": self._rest_symbol(symbol),
        })
        rows = result.get("data") or []
        row = rows[0] if isinstance(rows, list) and rows else {}
        try:
            return {
                "bid": float(row["bidPr"]),
                "ask": float(row["askPr"]),
                "last": float(row["lastPr"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ExchangeError(f"Malformed Bitget ticker for {symbol}: {row}") from exc

    def _normalize_open_order(
        self,
        symbol: str,
        side: str,
        size: float,
        entry: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: Optional[float],
        leverage: int,
        order_type: str,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal, Optional[Decimal]]:
        """Quantize an opening order; stop rounding may only tighten protection."""
        spec = self.get_contract_spec(symbol)
        if not spec.min_leverage <= leverage <= spec.max_leverage:
            raise SafetyViolation(
                f"REFUSED: Leverage {leverage} outside {spec.min_leverage}..{spec.max_leverage}"
            )
        q_size = self._step_round(Decimal(str(size)), spec.size_step, ROUND_DOWN)
        entry_rounding = ROUND_DOWN if side == "buy" else ROUND_UP
        stop_rounding = ROUND_UP if side == "buy" else ROUND_DOWN
        tp_rounding = ROUND_DOWN if side == "buy" else ROUND_UP
        q_entry = self._step_round(Decimal(str(entry)), spec.price_step, entry_rounding)
        q_sl = self._step_round(Decimal(str(stop_loss)), spec.price_step, stop_rounding)
        q_tp1 = self._step_round(Decimal(str(take_profit_1)), spec.price_step, tp_rounding)
        q_tp2 = (
            self._step_round(Decimal(str(take_profit_2)), spec.price_step, tp_rounding)
            if take_profit_2 is not None else None
        )
        max_size = spec.max_limit_size if order_type == "limit" else spec.max_market_size
        if q_size < spec.min_size or q_size > max_size:
            raise SafetyViolation(
                f"REFUSED: Quantized size {q_size} outside {spec.min_size}..{max_size} for {symbol}"
            )
        if q_size * q_entry < spec.min_notional:
            raise SafetyViolation(
                f"REFUSED: Order notional {q_size * q_entry} below {spec.min_notional} USDT"
            )
        self._validate_order_safety(
            float(q_sl), float(q_tp1), entry=float(q_entry), size=float(q_size), side=side,
            take_profit_2=float(q_tp2) if q_tp2 is not None else None,
        )
        return q_size, q_entry, q_sl, q_tp1, q_tp2

    def quantize_size(self, symbol: str, size: float) -> Decimal:
        """Round a closing/plan quantity down to the contract size step."""
        if not math.isfinite(size) or size <= 0:
            raise SafetyViolation("REFUSED: Quantity must be finite and positive.")
        spec = self.get_contract_spec(symbol)
        value = self._step_round(Decimal(str(size)), spec.size_step, ROUND_DOWN)
        if value < spec.min_size:
            raise SafetyViolation(
                f"REFUSED: Quantized size {value} is below {spec.min_size} for {spec.symbol}"
            )
        return value

    def split_size(self, symbol: str, size: float, first_fraction: float) -> tuple[Decimal, Decimal]:
        """Return precision-safe TP1 and remainder quantities, both exchange-valid."""
        if not 0 < first_fraction < 1:
            raise SafetyViolation("REFUSED: TP1 exit fraction must be between zero and one")
        total = self.quantize_size(symbol, size)
        first = self.quantize_size(symbol, float(total) * first_fraction)
        remainder = self.quantize_size(symbol, float(total - first))
        if first + remainder != total:
            raise SafetyViolation("REFUSED: Partial-exit quantities do not sum to the entry size")
        return first, remainder

    def quantize_trigger_price(self, symbol: str, price: float, side: str,
                               *, protective: bool) -> Decimal:
        """Quantize TP/SL triggers without weakening protective stops."""
        if not math.isfinite(price) or price <= 0:
            raise SafetyViolation("REFUSED: Trigger price must be finite and positive.")
        spec = self.get_contract_spec(symbol)
        if protective:
            rounding = ROUND_UP if side == "long" else ROUND_DOWN
        else:
            rounding = ROUND_DOWN if side == "long" else ROUND_UP
        return self._step_round(Decimal(str(price)), spec.price_step, rounding)

    # ── Trading Operations ──

    def set_leverage(self, symbol: str, leverage: int, side: str = "long") -> dict:
        """Set leverage for a symbol."""
        path = "/api/v2/mix/account/set-leverage"
        body = {
            "symbol": self._rest_symbol(symbol),
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
        client_oid: Optional[str] = None,
    ) -> OrderResult:
        """
        Place a futures order with MANDATORY TP/SL.

        This method ALWAYS validates that TP and SL are present.
        There is NO parameter or flag to bypass this check.
        """
        # SAFETY CHECK — absolutely non-negotiable
        entry = price if price is not None else targets.entry
        self._validate_order_safety(
            targets.stop_loss,
            targets.take_profit_1,
            entry=entry,
            size=size,
            side=side,
            take_profit_2=targets.take_profit_2,
        )

        symbol = self._rest_symbol(symbol)

        if not self._dry_run:
            q_size, q_entry, q_sl, q_tp1, q_tp2 = self._normalize_open_order(
                symbol, side, size, entry, targets.stop_loss, targets.take_profit_1,
                targets.take_profit_2, leverage, order_type,
            )
            size = float(q_size)
            entry = float(q_entry)
            price = entry if price is not None else None
            stop_loss = float(q_sl)
            take_profit_1 = float(q_tp1)
            take_profit_2 = float(q_tp2) if q_tp2 is not None else None
        else:
            stop_loss = targets.stop_loss
            take_profit_1 = targets.take_profit_1
            take_profit_2 = targets.take_profit_2

        # Set leverage
        hold_side = "long" if side == "buy" else "short"
        self.set_leverage(symbol, leverage, hold_side)

        # Determine trade side for Bitget API
        trade_side = "open"

        path = "/api/v2/mix/order/place-order"
        body = {
            "symbol": symbol,
            "productType": self.config.product_type,
            "marginMode": self.config.margin_mode,
            "marginCoin": "USDT",
            "size": format(Decimal(str(size)), "f"),
            "side": side,
            "tradeSide": trade_side,
            "orderType": order_type,
            # The preset is an immediate post-fill safety net. The scheduler replaces
            # it with explicitly sized per-lot TP1/TP2 plans after fill reconciliation.
            "presetStopSurplusPrice": format(
                Decimal(str(take_profit_2 if take_profit_2 is not None else take_profit_1)), "f"
            ),
            "presetStopLossPrice": format(Decimal(str(stop_loss)), "f"),
        }
        if self.config.position_mode == "one_way":
            body.pop("tradeSide")
            body["reduceOnly"] = "NO"

        if order_type == "limit" and price is not None:
            body["price"] = format(Decimal(str(price)), "f")
            # A maker strategy must never silently cross the spread and pay taker.
            body["force"] = "post_only"
        if client_oid:
            body["clientOid"] = client_oid

        try:
            result = self._request("POST", path, body=body)
        except requests.RequestException as original_error:
            # Never resend an opening POST after an ambiguous transport failure.  A
            # deterministic clientOid lets us discover an order Bitget accepted even
            # when its response was lost.
            if not client_oid or _is_definite_rejection(original_error):
                raise
            try:
                detail = self.get_order_detail(symbol, client_oid=client_oid)
            except Exception:
                raise original_error
            if not detail or not (detail.get("orderId") or detail.get("clientOid")):
                raise original_error
            result = {"code": "00000", "msg": "recovered_by_clientOid", "data": detail}

        order_id = result.get("data", {}).get("orderId", "dry_run_id")

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            status="submitted",
            timestamp=datetime.now().isoformat(),
            raw_response=result,
        )

    def get_order_detail(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        *,
        client_oid: Optional[str] = None,
    ) -> dict:
        """Return one futures order's current exchange state."""
        if not order_id and not client_oid:
            raise ValueError("order_id or client_oid is required")
        path = "/api/v2/mix/order/detail"
        params = {
            "symbol": self._rest_symbol(symbol),
            "productType": self.config.product_type,
        }
        if order_id:
            params["orderId"] = order_id
        else:
            params["clientOid"] = client_oid
        result = self._request("GET", path, params=params)
        return result.get("data", {})

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel one still-resting futures order."""
        path = "/api/v2/mix/order/cancel-order"
        return self._request("POST", path, body={
            "symbol": self._rest_symbol(symbol),
            "productType": self.config.product_type,
            "marginCoin": "USDT",
            "orderId": order_id,
        })

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
            "symbol": self._rest_symbol(symbol),
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
        plan_order_id: Optional[str] = None,
        *,
        position_level: bool = True,
    ) -> dict:
        """
        Move the position's stop loss to ``new_sl`` (a position-level TPSL update).

        Used by live trailing stops. Goes through ``_request`` so it is a no-op in
        dry-run. The caller is responsible for only ever moving the stop in the trade's
        favour (see llm_trading_bot.trailing.compute_trailing_stop).
        """
        if new_sl is None or not math.isfinite(new_sl) or new_sl <= 0:
            raise SafetyViolation("REFUSED: modify_stop_loss called without a valid stop price.")
        if not plan_order_id:
            raise SafetyViolation(
                "REFUSED: Cannot modify a stop without its existing Bitget TPSL plan order ID."
            )

        return self.modify_tpsl_order(
            symbol, plan_order_id, hold_side, size, new_sl,
            protective=True, position_level=position_level,
        )

    def modify_tpsl_order(
        self,
        symbol: str,
        plan_order_id: str,
        hold_side: str,
        size: float,
        trigger_price: float,
        *,
        protective: bool,
        position_level: bool = False,
    ) -> dict:
        """Modify one existing TP/SL plan's trigger and quantity."""
        if not plan_order_id:
            raise SafetyViolation("REFUSED: TPSL modification requires an existing plan ID")
        symbol = self._rest_symbol(symbol)
        if not self._dry_run:
            trigger = self.quantize_trigger_price(
                symbol, trigger_price, hold_side, protective=protective,
            )
            q_size = "" if position_level else format(self.quantize_size(symbol, size), "f")
        else:
            trigger = Decimal(str(trigger_price))
            q_size = "" if position_level else format(Decimal(str(size)), "f")
        return self._request("POST", "/api/v2/mix/order/modify-tpsl-order", body={
            "orderId": str(plan_order_id),
            "symbol": symbol,
            "productType": self.config.product_type,
            "marginCoin": "USDT",
            "triggerType": "fill_price",
            "triggerPrice": format(trigger, "f"),
            "executePrice": "0",
            "size": q_size,
        })

    def place_tpsl_order(
        self,
        symbol: str,
        hold_side: str,
        size: float,
        trigger_price: float,
        plan_type: str,
        client_oid: str,
    ) -> PlanOrder:
        """Place one explicitly sized per-lot TP or SL plan."""
        if plan_type not in ("profit_plan", "loss_plan"):
            raise SafetyViolation(f"REFUSED: Unsupported per-lot TPSL type {plan_type!r}")
        if hold_side not in ("long", "short"):
            raise SafetyViolation(f"REFUSED: Invalid TPSL hold side {hold_side!r}")
        if not client_oid:
            raise SafetyViolation("REFUSED: TPSL plans require a deterministic clientOid")
        symbol = self._rest_symbol(symbol)
        if self._dry_run:
            q_size = Decimal(str(size))
            q_trigger = Decimal(str(trigger_price))
        else:
            q_size = self.quantize_size(symbol, size)
            q_trigger = self.quantize_trigger_price(
                symbol, trigger_price, hold_side, protective=plan_type == "loss_plan",
            )
        api_hold_side = hold_side
        if self.config.position_mode == "one_way":
            api_hold_side = "buy" if hold_side == "long" else "sell"
        body = {
            "symbol": symbol,
            "productType": self.config.product_type,
            "marginCoin": "USDT",
            "planType": plan_type,
            "triggerPrice": format(q_trigger, "f"),
            "triggerType": "fill_price",
            "executePrice": "0",
            "holdSide": api_hold_side,
            "size": format(q_size, "f"),
            "clientOid": client_oid,
        }
        try:
            result = self._request("POST", "/api/v2/mix/order/place-tpsl-order", body=body)
        except requests.RequestException as original_error:
            # As with entries, never repeat an ambiguous state-changing POST. Recover
            # the accepted plan through its deterministic clientOid.
            try:
                existing = next(
                    plan for plan in self.get_tpsl_orders(symbol)
                    if plan.client_oid == client_oid
                )
            except Exception:
                raise original_error
            return existing
        data = result.get("data", {}) or {}
        return PlanOrder(
            order_id=str(data.get("orderId", "dry_run_plan")),
            client_oid=str(data.get("clientOid") or client_oid),
            symbol=symbol,
            plan_type=plan_type,
            side=hold_side,
            size=float(q_size),
            trigger_price=float(q_trigger),
            status="live",
        )

    def cancel_tpsl_order(self, symbol: str, order_id: str, plan_type: str) -> dict:
        """Cancel one known TP/SL plan without broad symbol-level cancellation."""
        if not order_id:
            raise SafetyViolation("REFUSED: TPSL cancellation requires an order ID")
        return self._request("POST", "/api/v2/mix/order/cancel-plan-order", body={
            "orderIdList": [{"orderId": str(order_id), "clientOid": ""}],
            "symbol": self._rest_symbol(symbol),
            "productType": self.config.product_type,
            "marginCoin": "USDT",
            "planType": plan_type,
        })

    @staticmethod
    def _parse_plan(row: dict) -> PlanOrder:
        pos_side = str(row.get("posSide") or row.get("holdSide") or "").lower()
        raw_side = str(row.get("side") or "").lower()
        side = pos_side if pos_side in ("long", "short") else (
            "long" if raw_side in ("buy", "long") else "short"
        )
        return PlanOrder(
            order_id=str(row.get("orderId", "")),
            client_oid=str(row.get("clientOid", "")),
            symbol=str(row.get("symbol", "")),
            plan_type=str(row.get("planType", "")),
            side=side,
            size=float(row.get("size", 0) or 0),
            trigger_price=float(
                row.get("triggerPrice") or row.get("stopSurplusTriggerPrice")
                or row.get("stopLossTriggerPrice") or 0
            ),
            status=str(row.get("planStatus", "")),
            created_at_ms=int(row.get("cTime", 0) or 0),
            updated_at_ms=int(row.get("uTime", 0) or 0),
            execute_order_id=str(row.get("executeOrderId", "")),
            filled_size=float(row.get("baseVolume", 0) or 0),
        )

    def get_tpsl_orders(self, symbol: Optional[str] = None, *, history: bool = False,
                        plan_status: Optional[str] = None) -> list[PlanOrder]:
        """Return current or historical futures TP/SL plans."""
        path = ("/api/v2/mix/order/orders-plan-history" if history
                else "/api/v2/mix/order/orders-plan-pending")
        params = {
            "productType": self.config.product_type,
            "planType": "profit_loss",
            "limit": "100",
        }
        if symbol:
            params["symbol"] = self._rest_symbol(symbol)
        if history and plan_status:
            params["planStatus"] = plan_status
        result = self._request("GET", path, params=params)
        data = result.get("data", {}) or {}
        rows = data.get("entrustedList") or [] if isinstance(data, dict) else []
        return [self._parse_plan(row) for row in rows]

    def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Get current open positions."""
        path = "/api/v2/mix/position/all-position"
        params = {"productType": self.config.product_type}
        if symbol:
            params["symbol"] = self._rest_symbol(symbol)

        result = self._request("GET", path, params=params)
        positions = []

        # Bitget's all-position endpoint ignores the symbol param and returns every
        # open position, so filter client-side. Without this a symbol-scoped caller
        # (e.g. a per-symbol scheduler in the shared orchestrator) sees other
        # symbols' positions and fails its position/lot reconciliation.
        want = self._rest_symbol(symbol) if symbol else None
        for pos_data in result.get("data", []):
            if want and str(pos_data.get("symbol", "")) != want:
                continue
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
                    margin_size=float(pos_data.get("marginSize", 0) or 0),
                ))

        return positions

    def get_pending_order_rows(self, symbol: Optional[str] = None) -> list[dict]:
        """Return every current normal order, including reduce-only/closing orders."""
        path = "/api/v2/mix/order/orders-pending"
        params = {"productType": self.config.product_type, "status": "live"}
        if symbol:
            params["symbol"] = self._rest_symbol(symbol)
        result = self._request("GET", path, params=params)
        data = result.get("data", {}) or {}
        return data.get("entrustedList") or [] if isinstance(data, dict) else []

    def get_pending_orders(self, symbol: Optional[str] = None) -> list[PendingOrder]:
        """Return all normal opening orders still capable of adding exposure."""
        rows = self.get_pending_order_rows(symbol)
        orders = []
        for row in rows:
            trade_side = str(row.get("tradeSide") or "open").lower()
            reduce_only = str(row.get("reduceOnly", "NO")).upper()
            if trade_side not in ("open", "buy_single", "sell_single"):
                continue
            if reduce_only == "YES":
                continue
            raw_side = str(row.get("side") or "").lower()
            pos_side = str(row.get("posSide") or "").lower()
            exposure_side = pos_side if pos_side in ("long", "short") else (
                "long" if raw_side == "buy" else "short"
            )
            orders.append(PendingOrder(
                order_id=str(row.get("orderId", "")),
                symbol=str(row.get("symbol", "")),
                side=exposure_side,
                size=float(row.get("size", 0) or 0),
                filled_size=float(row.get("baseVolume", 0) or 0),
                price=float(row.get("price", 0) or 0),
                leverage=int(float(row.get("leverage", 1) or 1)),
                client_oid=str(row.get("clientOid", "")),
                created_at_ms=int(row.get("cTime", 0) or 0),
            ))
        return orders

    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> list[dict]:
        """Return recent normal and plan-generated futures orders."""
        params = {
            "productType": self.config.product_type,
            "limit": str(max(1, min(100, limit))),
        }
        if symbol:
            params["symbol"] = self._rest_symbol(symbol)
        result = self._request("GET", "/api/v2/mix/order/orders-history", params=params)
        data = result.get("data", {}) or {}
        return data.get("entrustedList") or [] if isinstance(data, dict) else []

    def get_order_fills(self, symbol: str, order_id: str,
                        detail: Optional[dict] = None) -> list[Fill]:
        """Return exact entry fills; demo falls back to its aggregate order detail."""
        if not order_id:
            raise ValueError("order_id is required")
        rows = []
        if not self.config.testnet and not self._dry_run:
            result = self._request("GET", "/api/v2/mix/order/fill-history", params={
                "orderId": order_id,
                "symbol": self._rest_symbol(symbol),
                "productType": self.config.product_type,
                "limit": "100",
            })
            data = result.get("data", {}) or {}
            rows = data.get("fillList") or [] if isinstance(data, dict) else []
        if not rows and detail:
            rows = [{
                "tradeId": detail.get("tradeId", ""),
                "orderId": order_id,
                "symbol": detail.get("symbol") or self._rest_symbol(symbol),
                "price": detail.get("priceAvg") or detail.get("price"),
                "baseVolume": detail.get("baseVolume") or detail.get("filledQty"),
                "fee": detail.get("fee", 0),
                "side": detail.get("side", ""),
                "cTime": detail.get("uTime") or detail.get("cTime"),
            }]
        fills = []
        for row in rows:
            fee = float(row.get("fee", 0) or 0)
            if row.get("feeDetail"):
                fee = sum(
                    float(item.get("totalFee", 0) or 0)
                    for item in row.get("feeDetail", [])
                )
            size = float(row.get("baseVolume", 0) or 0)
            price = float(row.get("price", 0) or 0)
            if size <= 0 or price <= 0:
                continue
            fills.append(Fill(
                trade_id=str(row.get("tradeId", "")),
                order_id=str(row.get("orderId") or order_id),
                symbol=str(row.get("symbol") or self._rest_symbol(symbol)),
                price=price,
                size=size,
                fee=fee,
                timestamp_ms=int(row.get("cTime", 0) or 0),
                side=str(row.get("side", "")),
            ))
        return fills

    def get_position_history(self, symbol: Optional[str] = None,
                             limit: int = 100) -> list[dict]:
        """Return recent closed positions for causal live streak sizing."""
        path = "/api/v2/mix/position/history-position"
        params = {
            "productType": self.config.product_type,
            "limit": str(max(1, min(100, limit))),
        }
        if symbol:
            params["symbol"] = self._rest_symbol(symbol)
        result = self._request("GET", path, params=params)
        data = result.get("data", {}) or {}
        return data.get("list") or [] if isinstance(data, dict) else []

    def close_position(self, symbol: str, side: str, size: float,
                       client_oid: Optional[str] = None) -> dict:
        """Close a position."""
        if side not in ("long", "short"):
            raise SafetyViolation(f"REFUSED: Unknown position side {side!r}.")
        if not math.isfinite(size) or size <= 0:
            raise SafetyViolation("REFUSED: Close size must be finite and positive.")

        # One-way uses the intuitive opposing order plus reduceOnly.  Bitget hedge
        # mode instead pairs buy/close with a long and sell/close with a short.
        if self.config.position_mode == "one_way":
            close_side = "sell" if side == "long" else "buy"
        else:
            close_side = "buy" if side == "long" else "sell"
        symbol = self._rest_symbol(symbol)
        if not self._dry_run:
            size = float(self.quantize_size(symbol, size))
        path = "/api/v2/mix/order/place-order"
        body = {
            "symbol": symbol,
            "productType": self.config.product_type,
            "marginMode": self.config.margin_mode,
            "marginCoin": "USDT",
            "size": str(size),
            "side": close_side,
            "tradeSide": "close",
            "orderType": "market",
        }
        if self.config.position_mode == "one_way":
            body.pop("tradeSide")
            body["reduceOnly"] = "YES"
        if client_oid:
            body["clientOid"] = client_oid
        try:
            return self._request("POST", path, body=body)
        except requests.RequestException as original_error:
            if not client_oid or _is_definite_rejection(original_error):
                raise
            try:
                detail = self.get_order_detail(symbol, client_oid=client_oid)
            except Exception:
                raise original_error
            if not detail:
                raise original_error
            return {"code": "00000", "msg": "recovered_by_clientOid", "data": detail}

    def get_account_info(self) -> dict:
        """Get account balance and info."""
        path = "/api/v2/mix/account/accounts"
        params = {"productType": self.config.product_type}
        return self._request("GET", path, params=params)

    def get_single_account(self, symbol: str) -> dict:
        """Return symbol account configuration including position and margin modes."""
        return self._request("GET", "/api/v2/mix/account/account", params={
            "symbol": self._rest_symbol(symbol),
            "productType": self.config.product_type,
            "marginCoin": "USDT",
        })

    def preflight(self, symbol: str) -> dict:
        """Fail closed unless credentials, clock, contract and account modes are valid."""
        if self._dry_run:
            raise SafetyViolation(
                "REFUSED: Paper/live startup requires explicit Bitget credentials; "
                "credential-free operation is analyze-only."
            )
        response = self.get_single_account(symbol)
        server_ms = int(response.get("requestTime", 0) or 0)
        if not server_ms:
            raise ExchangeError("Bitget preflight did not return server requestTime")
        drift_ms = abs(int(time.time() * 1000) - server_ms)
        if drift_ms > 30_000:
            raise ExchangeError(f"Local clock differs from Bitget by {drift_ms / 1000:.1f}s")
        account = response.get("data", {}) or {}
        expected_pos = "one_way_mode" if self.config.position_mode == "one_way" else "hedge_mode"
        actual_pos = str(account.get("posMode", ""))
        if actual_pos != expected_pos:
            raise SafetyViolation(
                f"REFUSED: Bitget position mode is {actual_pos or 'unknown'}, expected {expected_pos}"
            )
        actual_margin = str(account.get("marginMode", ""))
        if actual_margin != self.config.margin_mode:
            raise SafetyViolation(
                f"REFUSED: Bitget margin mode is {actual_margin or 'unknown'}, "
                f"expected {self.config.margin_mode}"
            )
        spec = self.get_contract_spec(symbol)
        # Surface the account's configured leverage so the scheduler can verify it
        # against the active tier (isolated leverage is per-side). Absent on mocks.
        if actual_margin == "isolated":
            lev_long = account.get("isolatedLongLever")
            lev_short = account.get("isolatedShortLever")
        else:
            lev_long = lev_short = account.get("crossedMarginLeverage")
        return {
            "symbol": spec.symbol,
            "position_mode": actual_pos,
            "margin_mode": actual_margin,
            "clock_drift_ms": drift_ms,
            "leverage_long": int(lev_long) if lev_long not in (None, "") else None,
            "leverage_short": int(lev_short) if lev_short not in (None, "") else None,
            "demo": self.config.testnet,
        }

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

    def get_account_equity(self, dry_run_default: float = 100.0) -> float:
        """Return USDT futures account equity, including unrealized PnL."""
        if self._dry_run:
            return dry_run_default
        try:
            data = self.get_account_info().get("data", [])
            for acct in data:
                if acct.get("marginCoin") == "USDT":
                    return float(acct.get("accountEquity", acct.get("usdtEquity", 0)) or 0)
            if data:
                return float(data[0].get("accountEquity", 0) or 0)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"⚠ Could not parse account equity: {e}")
        return 0.0
