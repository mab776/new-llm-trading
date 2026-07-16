"""
Tests for configuration models and loading.
Covers: Pydantic validation, weight constraints, leverage tiers, load_config.
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_trading_bot.config import (
    AppConfig,
    BacktestingConfig,
    FeesConfig,
    FiltersConfig,
    LeverageTier,
    PositionSizingConfig,
    RiskManagementConfig,
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

    def test_profile_inherits_and_deep_merges_base_config(self, tmp_path):
        base = tmp_path / "base.json"
        base.write_text(json.dumps({
            "trading": {"symbol": "BTC-USDT", "primary_timeframe": "4h"},
            "position_sizing": {
                "risk_pct_per_trade": .02, "global_max_margin_pct": .044,
            },
        }))
        profile = tmp_path / "profile.json"
        profile.write_text(json.dumps({
            "_extends": "base.json",
            "position_sizing": {"global_max_margin_pct": 0},
        }))

        config = load_config(profile)
        assert config.trading.symbol == "BTC-USDT"
        assert config.position_sizing.risk_pct_per_trade == .02
        assert config.position_sizing.global_max_margin_pct == 0

    def test_profile_rejects_circular_inheritance(self, tmp_path):
        one, two = tmp_path / "one.json", tmp_path / "two.json"
        one.write_text(json.dumps({"_extends": "two.json"}))
        two.write_text(json.dumps({"_extends": "one.json"}))
        with pytest.raises(ValueError, match="Circular config inheritance"):
            load_config(one)

    @pytest.mark.parametrize("name", (
        "config-aggressive.json", "config-eth-aggressive.json",
        "config-sol-aggressive.json",
    ))
    def test_aggressive_profiles_disable_shared_exposure_caps(self, name):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / name)
        assert config.bitget.testnet is True
        assert config.position_sizing.anti_martingale_step == .05
        assert config.position_sizing.global_max_margin_pct == 0
        assert config.position_sizing.global_max_notional_pct == 0
        # The per-trade rail is now a scale-invariant fraction inherited from
        # the base config; normal sizing (~2-3%) never reaches it.
        assert config.position_sizing.max_position_pct == 0.66

    @pytest.mark.parametrize("name", (
        "config.json", "config-eth.json", "config-sol.json",
        "config-aggressive.json", "config-eth-aggressive.json",
        "config-sol-aggressive.json",
    ))
    def test_shipped_profiles_have_no_llm_config(self, name):
        # Pure technical-signal bot: no OpenWebUI/LLM config anywhere.
        cfg = load_config(Path(__file__).resolve().parents[1] / name)
        assert not hasattr(cfg, "openwebui")

    @pytest.mark.parametrize("name", (
        "config.json", "config-eth.json", "config-sol.json",
        "config-aggressive.json", "config-eth-aggressive.json",
        "config-sol-aggressive.json",
    ))
    def test_shipped_profiles_pin_isolated_one_way(self, name):
        # All symbols run against ONE shared demo/live account set to isolated +
        # one-way; preflight fails closed on mismatch. A config that falls back to
        # the crossed default would refuse to start on paper day (ETH/SOL once did).
        cfg = load_config(Path(__file__).resolve().parents[1] / name)
        assert cfg.bitget.margin_mode == "isolated", name
        assert cfg.bitget.position_mode == "one_way", name
        assert cfg.bitget.testnet is True, name


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


class TestPositionSizingConfig:
    def test_rejects_inverted_anti_martingale_bounds(self):
        with pytest.raises(ValidationError, match="anti_martingale_min"):
            PositionSizingConfig(anti_martingale_min=1.2, anti_martingale_max=1.1)

    def test_rejects_negative_exposure_cap(self):
        with pytest.raises(ValidationError):
            PositionSizingConfig(global_max_margin_pct=-0.01)


class TestSafetyBounds:
    @pytest.mark.parametrize("kwargs", (
        {"confidence_min": 4}, {"confidence_max": 96}, {"atr_sl_multiplier": 0},
    ))
    def test_scoring_safety_bounds(self, kwargs):
        with pytest.raises(ValidationError):
            ScoringConfig(**kwargs)

    def test_leverage_and_tp_fraction_bounds(self):
        with pytest.raises(ValidationError):
            LeverageTier(leverage=126)
        with pytest.raises(ValidationError):
            LeverageTier(tp1_exit_pct=1)

    def test_active_tier_rejected_during_validation(self):
        with pytest.raises(ValidationError, match="active_tier"):
            TradingConfig(
                leverage_tiers={"conservative": LeverageTier()}, active_tier="missing"
            )


class TestRiskManagementConfig:
    def test_defaults(self):
        rm = RiskManagementConfig()
        assert rm.max_holding_hours == 0  # 0 = forced time-close disabled (default)
        assert rm.cooldown_candles_after_sl == 3
        assert rm.consecutive_loss_penalty == 5.0
        assert rm.max_consecutive_loss_penalty == 20.0
        assert rm.loss_penalty_decay_candles == 10
        assert rm.use_maker_fee_for_tp is True

    def test_custom_values(self):
        rm = RiskManagementConfig(
            max_holding_hours=48,
            cooldown_candles_after_sl=5,
            consecutive_loss_penalty=10.0,
        )
        assert rm.max_holding_hours == 48
        assert rm.cooldown_candles_after_sl == 5
        assert rm.consecutive_loss_penalty == 10.0

    def test_in_app_config(self):
        cfg = AppConfig()
        assert cfg.risk_management.max_holding_hours == 0
        assert cfg.risk_management.use_maker_fee_for_tp is True
