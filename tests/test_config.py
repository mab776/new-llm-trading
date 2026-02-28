"""
Tests for configuration models and loading.
Covers: Pydantic validation, weight constraints, leverage tiers, load_config.
"""

import json
import tempfile
from pathlib import Path

import pytest

from llm_trading_bot.config import (
    AppConfig,
    BacktestingConfig,
    FeesConfig,
    FiltersConfig,
    LeverageTier,
    ScoringConfig,
    TradingConfig,
    load_config,
)


class TestDefaultConfig:
    def test_default_app_config(self):
        config = AppConfig()
        assert config.trading.primary_timeframe == "4h"
        assert config.fees.taker == 0.0006
        assert config.backtesting.initial_balance == 10000

    def test_default_leverage_tier(self):
        tier = LeverageTier()
        assert tier.leverage == 20
        assert tier.tp1_rr == 3.0
        assert tier.tp2_rr == 6.0
        assert tier.tp1_exit_pct == 0.3


class TestWeightValidation:
    def test_valid_weights_sum_to_one(self):
        cfg = ScoringConfig(weights={
            "trend": 0.30, "momentum": 0.25, "volume": 0.15,
            "support_resistance": 0.20, "risk": 0.10,
        })
        assert sum(cfg.weights.values()) == pytest.approx(1.0)

    def test_invalid_weights_rejected(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ScoringConfig(weights={
                "trend": 0.50, "momentum": 0.50, "volume": 0.50,
                "support_resistance": 0.50, "risk": 0.50,
            })

    def test_slightly_off_weights_rejected(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ScoringConfig(weights={
                "trend": 0.30, "momentum": 0.25, "volume": 0.15,
                "support_resistance": 0.20, "risk": 0.15,  # sums to 1.05
            })


class TestActiveLeverageTier:
    def test_active_tier_property(self):
        config = AppConfig()
        config.trading.leverage_tiers = {
            "conservative": LeverageTier(leverage=5),
            "aggressive": LeverageTier(leverage=15),
        }
        config.trading.active_tier = "aggressive"
        assert config.trading.active_leverage_tier.leverage == 15

    def test_missing_tier_raises(self):
        config = AppConfig()
        config.trading.leverage_tiers = {"conservative": LeverageTier()}
        config.trading.active_tier = "nonexistent"
        with pytest.raises(KeyError):
            _ = config.trading.active_leverage_tier


class TestActiveFeeRate:
    def test_taker_fee(self):
        fees = FeesConfig(maker=0.0002, taker=0.0006, default_order_type="taker")
        assert fees.active_fee_rate == 0.0006

    def test_maker_fee(self):
        fees = FeesConfig(maker=0.0002, taker=0.0006, default_order_type="maker")
        assert fees.active_fee_rate == 0.0002


class TestLoadConfig:
    def test_load_from_valid_json(self, tmp_path):
        config_data = {
            "trading": {
                "symbol": "ETH-USDT",
                "primary_timeframe": "1h",
                "leverage_tiers": {
                    "default": {"leverage": 7, "tp1_rr": 2.5, "tp2_rr": 4.0}
                },
                "active_tier": "default",
            },
            "scoring": {
                "weights": {
                    "trend": 0.30, "momentum": 0.25, "volume": 0.15,
                    "support_resistance": 0.20, "risk": 0.10,
                },
            },
            "backtesting": {"initial_balance": 5000},
        }
        cfg_file = tmp_path / "test_config.json"
        cfg_file.write_text(json.dumps(config_data))

        config = load_config(str(cfg_file))
        assert config.trading.symbol == "ETH-USDT"
        assert config.backtesting.initial_balance == 5000
        assert config.trading.active_leverage_tier.leverage == 7

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.json")

    def test_invalid_json_raises(self, tmp_path):
        cfg_file = tmp_path / "bad.json"
        cfg_file.write_text("not valid json {{{")
        with pytest.raises(Exception):
            load_config(str(cfg_file))


class TestFiltersConfig:
    def test_defaults(self):
        f = FiltersConfig()
        assert f.min_adx == 20
        assert f.min_category_agreement == 2
        assert f.require_trend_momentum_agree is True
        assert f.skip_choppy_regime is True
        assert f.skip_volatile_regime is False


class TestBacktestingConfig:
    def test_defaults(self):
        bc = BacktestingConfig()
        assert bc.warmup_periods == 200
        assert bc.enable_partial_exits is True
        assert bc.enable_trailing_stops is False
