"""
Configuration models using Pydantic for validation.
Single source of truth for all configuration structures.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class OpenWebUIConfig(BaseModel):
    base_url: str = "http://localhost:3000"
    api_key: str = ""
    model_ids: list[str] = ["llama3.1:8b"]
    timeout_seconds: int = 120
    # deterministic matches the validated backtest: passed MARGINAL signals trade
    # directly. consensus remains available for explicit experiments.
    marginal_execution: Literal["deterministic", "consensus"] = "consensus"


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
    # taker = market at decision close; maker = post-only limit at that close,
    # good for the following primary bar only.
    entry_mode: Literal["taker", "maker"] = "taker"
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
    # Partial overrides of openwebui_filter.DEFAULT_SCORING_POINTS.
    points: dict[str, float] = Field(default_factory=dict)

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
    opposite_exit_threshold: float = 0.0    # Close open positions when the composite score flips
    #                                         beyond this against them (0 = disabled)
    # Drawdown circuit-breaker: while balance drawdown from its peak >= threshold,
    # pyramiding is capped at dd_throttle_slots and per-trade risk is multiplied by
    # dd_throttle_risk, until equity recovers. Meant as tail insurance against a regime
    # break (set wide, e.g. 0.25); tight thresholds cost return — see opt/README.md.
    dd_throttle_threshold: float = 0.0      # 0 = disabled; fraction, e.g. 0.25 = 25% DD
    dd_throttle_slots: int = 1              # max concurrent positions while throttled
    dd_throttle_risk: float = 0.5           # risk multiplier while throttled


class PositionSizingConfig(BaseModel):
    """How much capital to put behind each trade (used by live AND backtest)."""
    risk_pct_per_trade: float = 0.02   # Fraction of account balance risked as margin per trade
    max_position_usd: float = 100      # Hard cap on the margin committed to a single trade
    max_positions: int = 1             # Concurrent SAME-direction positions (pyramiding); 1 = classic
    conviction_exponent: float = 0.0   # 0 = off; scale risk by (|score|/strong_threshold)^k,
    #                                    clamped to [0.5, 1.5] — bigger signals get bigger size
    anti_martingale_step: float = Field(0.0, ge=0)  # signed streak step; 0 = disabled
    anti_martingale_min: float = Field(0.7, gt=0)   # lower multiplier after losses
    anti_martingale_max: float = Field(1.1, gt=0)   # upper multiplier after wins
    portfolio_risk_multiplier: float = Field(1.0, gt=0)  # before exposure caps
    global_max_positions: int = Field(0, ge=0)      # all symbols + resting entries; 0 = off
    global_max_margin_pct: float = Field(0.0, ge=0) # committed margin / equity; 0 = off
    global_max_notional_pct: float = Field(0.0, ge=0)  # entry notional / equity; 0 = off

    @model_validator(mode="after")
    def validate_anti_martingale_bounds(self):
        if self.anti_martingale_min > self.anti_martingale_max:
            raise ValueError("anti_martingale_min must be <= anti_martingale_max")
        return self


class BacktestingConfig(BaseModel):
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"
    initial_balance: float = 10000
    warmup_periods: int = 200
    enable_partial_exits: bool = True
    enable_trailing_stops: bool = False
    include_funding: bool = True    # Model perp funding payments (every 8h on notional).
    #                                 Live trading ignores this — the exchange settles it.


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


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge a small profile override into a base config."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_config_dict(path: Path, seen: set[Path]) -> dict:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Circular config inheritance detected at: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = json.load(f)
    parent = raw.pop("_extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str) or not parent:
        raise ValueError("_extends must be a non-empty config path")
    parent_path = (path.parent / parent).resolve()
    base = _load_config_dict(parent_path, seen | {resolved})
    return _deep_merge(base, raw)


def load_config(path: str | Path = "config.json") -> AppConfig:
    """Load and validate a JSON config, including optional profile inheritance."""
    path = Path(path)
    raw = _load_config_dict(path, set())
    return AppConfig(**raw)
