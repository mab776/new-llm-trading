"""
Configuration models using Pydantic for validation.
Single source of truth for all configuration structures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class OpenWebUIConfig(BaseModel):
    base_url: str = "http://localhost:3000"
    api_key: str = ""
    model_ids: list[str] = ["llama3.1:8b"]
    timeout_seconds: int = 120


class LeverageTier(BaseModel):
    leverage: int = 20
    strong_threshold: float = 30
    marginal_threshold_low: float = 25
    marginal_threshold_high: float = 30
    tp1_rr: float = 3.0
    tp2_rr: float = 6.0
    tp1_exit_pct: float = 0.3


class TrailingStopConfig(BaseModel):
    enabled: bool = False
    activation_pct: float = 1.0
    callback_pct: float = 0.5


class TradingConfig(BaseModel):
    symbol: str = "BTC-USDT"
    yfinance_symbol: str = "BTC-USD"
    timeframes: list[str] = ["1h", "4h", "1d"]
    primary_timeframe: str = "4h"
    leverage_tiers: dict[str, LeverageTier] = Field(default_factory=dict)
    active_tier: str = "conservative"
    stop_loss_strategy: Literal["atr", "structure", "hybrid"] = "hybrid"
    trailing_stop: TrailingStopConfig = Field(default_factory=TrailingStopConfig)

    @property
    def active_leverage_tier(self) -> LeverageTier:
        return self.leverage_tiers[self.active_tier]


class ScoringConfig(BaseModel):
    weights: dict[str, float] = Field(default_factory=lambda: {
        "trend": 0.30, "momentum": 0.25, "volume": 0.15,
        "support_resistance": 0.20, "risk": 0.10
    })
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    atr_tp1_multiplier: float = 3.0
    atr_tp2_multiplier: float = 6.0
    adx_ranging_threshold: float = 20
    min_volatility_pct: float = 0.3
    confidence_min: float = 5
    confidence_max: float = 95

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")
        return v


class FiltersConfig(BaseModel):
    min_adx: float = 20
    min_volatility_pct: float = 0.3
    min_profit_after_fees: bool = True
    min_category_agreement: int = 2        # At least N/5 categories must agree
    require_trend_momentum_agree: bool = True  # Trend + momentum must agree
    skip_choppy_regime: bool = True        # Skip trades in choppy markets
    skip_volatile_regime: bool = False     # Skip trades in extremely volatile markets


class FeesConfig(BaseModel):
    maker: float = 0.0002
    taker: float = 0.0006
    default_order_type: Literal["maker", "taker"] = "taker"

    @property
    def active_fee_rate(self) -> float:
        return self.maker if self.default_order_type == "maker" else self.taker


class BitgetConfig(BaseModel):
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    testnet: bool = True
    product_type: str = "USDT-FUTURES"


class RiskManagementConfig(BaseModel):
    """Risk management rules imported from the predecessor project."""
    max_holding_hours: int = 0        # Force close after this many hours (0 = disabled)
    cooldown_candles_after_sl: int = 3  # Skip N candles after a SL hit
    consecutive_loss_penalty: float = 5.0   # Add this to entry threshold per consecutive loss
    max_consecutive_loss_penalty: float = 20.0  # Cap on total penalty
    loss_penalty_decay_candles: int = 10    # Candles after last loss before penalty starts decaying
    use_maker_fee_for_tp: bool = True       # TP exits (limit) use maker fee, SL (market) uses taker


class PositionSizingConfig(BaseModel):
    """How much capital to put behind each trade (used by live AND backtest)."""
    risk_pct_per_trade: float = 0.02   # Fraction of account balance risked as margin per trade
    max_position_usd: float = 100      # Hard cap on the margin committed to a single trade


class BacktestingConfig(BaseModel):
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"
    initial_balance: float = 10000
    warmup_periods: int = 200
    enable_partial_exits: bool = True
    enable_trailing_stops: bool = False


class SchedulingConfig(BaseModel):
    interval_minutes: int = 60
    check_positions_interval_minutes: int = 15


class DataCacheConfig(BaseModel):
    ttl_seconds: int = 300


class DataSourceConfig(BaseModel):
    """Data source configuration. Controls where OHLCV data comes from."""
    source: str = "yfinance"      # "yfinance", "binance", "bitget", or any ccxt exchange
    exchange_symbol: str = "BTC/USDT"  # Symbol format for ccxt exchanges
    market: str = "futures"       # "futures" (swap) or "spot" — Bitget market for fetching


class AppConfig(BaseModel):
    """Root configuration — single source of truth."""
    openwebui: OpenWebUIConfig = Field(default_factory=OpenWebUIConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    fees: FeesConfig = Field(default_factory=FeesConfig)
    bitget: BitgetConfig = Field(default_factory=BitgetConfig)
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    backtesting: BacktestingConfig = Field(default_factory=BacktestingConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    risk_management: RiskManagementConfig = Field(default_factory=RiskManagementConfig)
    data_cache: DataCacheConfig = Field(default_factory=DataCacheConfig)
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)


def load_config(path: str | Path = "config.json") -> AppConfig:
    """Load and validate configuration from JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = json.load(f)
    return AppConfig(**raw)
