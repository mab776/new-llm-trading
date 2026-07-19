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


class LeverageTier(BaseModel):
    leverage: int = Field(20, ge=1, le=125)
    strong_threshold: float = Field(30, ge=0, le=100)
    marginal_threshold_low: float = Field(25, ge=0, le=100)
    marginal_threshold_high: float = Field(30, ge=0, le=100)
    tp1_rr: float = Field(3.0, gt=0)
    tp2_rr: float = Field(6.0, gt=0)
    tp1_exit_pct: float = Field(0.3, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_thresholds_and_targets(self):
        if self.marginal_threshold_low > min(self.marginal_threshold_high, self.strong_threshold):
            raise ValueError("marginal_threshold_low must not exceed the other signal thresholds")
        if self.tp2_rr < self.tp1_rr:
            raise ValueError("tp2_rr must be >= tp1_rr")
        return self


class TrailingStopConfig(BaseModel):
    enabled: bool = False
    activation_pct: float = Field(1.0, gt=0, le=100)
    callback_pct: float = Field(0.5, gt=0, le=100)


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

    @model_validator(mode="after")
    def validate_references(self):
        if not self.timeframes or self.primary_timeframe not in self.timeframes:
            raise ValueError("primary_timeframe must be present in timeframes")
        if self.leverage_tiers and self.active_tier not in self.leverage_tiers:
            raise ValueError(f"active_tier {self.active_tier!r} is not configured")
        return self

    @property
    def active_leverage_tier(self) -> LeverageTier:
        return self.leverage_tiers[self.active_tier]


class ScoringConfig(BaseModel):
    weights: dict[str, float] = Field(default_factory=lambda: {
        "trend": 0.30, "momentum": 0.25, "volume": 0.15,
        "support_resistance": 0.20, "risk": 0.10
    })
    atr_period: int = Field(14, gt=0)
    atr_sl_multiplier: float = Field(1.5, gt=0)
    atr_tp1_multiplier: float = Field(3.0, gt=0)
    atr_tp2_multiplier: float = Field(6.0, gt=0)
    adx_ranging_threshold: float = 20
    min_volatility_pct: float = 0.3
    confidence_min: float = Field(5, ge=5, le=95)
    confidence_max: float = Field(95, ge=5, le=95)
    # Partial overrides of openwebui_filter.DEFAULT_SCORING_POINTS.
    points: dict[str, float] = Field(default_factory=dict)
    # Per-timeframe discrete alignment-vote weight (e.g. {"1h": 0, "1d": 3}).
    # None = legacy flat ±5 for every secondary timeframe. Gridded 2026-07-19
    # (independent 1h/1d sweeps, select-TRAIN report-TEST + OOS invariance):
    # the 1h vote is noise (TRAIN monotone-better toward 0), 1d wants ~3;
    # OOS holdout 4.00x/16.3%DD -> 5.15x/12.9%DD at {"1h": 0, "1d": 3}.
    alignment_scale_by_tf: dict[str, float] | None = None

    @field_validator("alignment_scale_by_tf")
    @classmethod
    def validate_alignment_scale_by_tf(
        cls, v: dict[str, float] | None
    ) -> dict[str, float] | None:
        if v is not None and any(w < 0 for w in v.values()):
            raise ValueError("alignment_scale_by_tf weights cannot be negative")
        return v

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: dict[str, float]) -> dict[str, float]:
        if any(weight < 0 for weight in v.values()):
            raise ValueError("Weights cannot be negative")
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")
        return v

    @model_validator(mode="after")
    def validate_confidence_bounds(self):
        if self.confidence_min > self.confidence_max:
            raise ValueError("confidence_min must be <= confidence_max")
        return self


class FiltersConfig(BaseModel):
    min_adx: float = 20
    min_volatility_pct: float = 0.3
    min_profit_after_fees: bool = True
    min_category_agreement: int = 2        # At least N/5 categories must agree
    require_trend_momentum_agree: bool = True  # Trend + momentum must agree
    skip_choppy_regime: bool = True        # Skip trades in choppy markets
    skip_volatile_regime: bool = False     # Skip trades in extremely volatile markets


class FeesConfig(BaseModel):
    maker: float = Field(0.0002, ge=0, lt=0.1)
    taker: float = Field(0.0006, ge=0, lt=0.1)
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
    position_mode: Literal["one_way", "hedge"] = "one_way"
    margin_mode: Literal["crossed", "isolated"] = "crossed"


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
    risk_pct_per_trade: float = Field(0.02, gt=0, le=1)
    # Per-trade margin ceiling as a FRACTION of the sizing balance (scale-
    # invariant sanity rail against sizing bugs — it never freezes compounding
    # the way the old absolute max_position_usd did). Normal sizing (~2-3%)
    # sits far below it; it only bites on a runaway size computation.
    max_position_pct: float = Field(0.66, gt=0, le=1)
    max_positions: int = Field(1, ge=1)
    conviction_exponent: float = Field(0.0, ge=0)
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
    initial_balance: float = Field(10000, gt=0)
    warmup_periods: int = Field(200, gt=0)
    enable_partial_exits: bool = True
    enable_trailing_stops: bool = False
    include_funding: bool = True    # Model perp funding payments (every 8h on notional).
    #                                 Live trading ignores this — the exchange settles it.
    # Execution realism (matches the research harness / opt/fastbt semantics).
    # Per-side price slippage applied to MARKET fills only (taker entry, SL,
    # signal-flip, time-expired); maker fills and TP limit exits are unslipped.
    slippage_pct: float = Field(0.0, ge=0, lt=0.05)
    # Cap stops at the isolated-margin liquidation price (a stop beyond it can
    # never be reached — the position is force-closed at liquidation first).
    model_liquidation: bool = False
    maintenance_margin: float = Field(0.005, ge=0, lt=0.5)


class SchedulingConfig(BaseModel):
    interval_minutes: int = Field(60, gt=0)
    check_positions_interval_minutes: int = Field(15, gt=0)
    # Live logging: one trading-YYYY-MM-DD.log / decisions-YYYY-MM-DD.jsonl file
    # per LOCAL day; files older than this many days are deleted automatically.
    log_retention_days: int = Field(90, ge=1)


class DataCacheConfig(BaseModel):
    ttl_seconds: int = Field(300, ge=0)


class DataSourceConfig(BaseModel):
    """Data source configuration. Controls where OHLCV data comes from."""
    source: str = "yfinance"      # "yfinance", "binance", "bitget", or any ccxt exchange
    exchange_symbol: str = "BTC/USDT"  # Symbol format for ccxt exchanges
    market: Literal["futures", "spot"] = "futures"


class AppConfig(BaseModel):
    """Root configuration — single source of truth."""
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
