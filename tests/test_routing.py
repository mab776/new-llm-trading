"""
Tests for the signal routing module.
"""

import pytest

from llm_trading_bot.config import AppConfig, LeverageTier, load_config
from llm_trading_bot.routing import (
    RoutingDecision,
    build_llm_context,
    build_template_response,
    classify_signal,
    route_signal,
)
from llm_trading_bot.scoring import (
    Direction,
    IndicatorSet,
    ScoringResult,
    SignalStrength,
    TradeTargets,
)


@pytest.fixture
def conservative_tier() -> LeverageTier:
    return LeverageTier(
        leverage=5,
        strong_threshold=70,
        marginal_threshold_low=45,
        marginal_threshold_high=70,
        tp1_rr=2.0,
        tp2_rr=3.5,
        tp1_exit_pct=0.5,
    )


@pytest.fixture
def aggressive_tier() -> LeverageTier:
    return LeverageTier(
        leverage=15,
        strong_threshold=80,
        marginal_threshold_low=55,
        marginal_threshold_high=80,
        tp1_rr=1.5,
        tp2_rr=2.5,
        tp1_exit_pct=0.6,
    )


class TestClassifySignal:
    def test_strong_signal(self, conservative_tier):
        assert classify_signal(85, conservative_tier) == SignalStrength.STRONG
        assert classify_signal(-75, conservative_tier) == SignalStrength.STRONG

    def test_marginal_signal(self, conservative_tier):
        assert classify_signal(55, conservative_tier) == SignalStrength.MARGINAL
        assert classify_signal(-50, conservative_tier) == SignalStrength.MARGINAL

    def test_wait_signal(self, conservative_tier):
        assert classify_signal(20, conservative_tier) == SignalStrength.WAIT
        assert classify_signal(-10, conservative_tier) == SignalStrength.WAIT
        assert classify_signal(0, conservative_tier) == SignalStrength.WAIT

    def test_aggressive_higher_thresholds(self, aggressive_tier):
        # 75 is STRONG for conservative but MARGINAL for aggressive
        assert classify_signal(75, aggressive_tier) == SignalStrength.MARGINAL
        assert classify_signal(85, aggressive_tier) == SignalStrength.STRONG


class TestBuildTemplateResponse:
    def test_contains_essential_info(self, conservative_tier):
        result = ScoringResult(
            direction=Direction.BULLISH,
            confidence=82,
            signal_strength=SignalStrength.STRONG,
            raw_score=75,
            reasons=["trend: bullish (+60)", "momentum aligned"],
        )
        targets = TradeTargets(
            entry=50000, stop_loss=49000, take_profit_1=52000,
            take_profit_2=54000, risk_amount=1000, reward_1=2000,
            reward_2=4000, direction=Direction.BULLISH,
        )
        response = build_template_response(result, targets, conservative_tier)

        assert "LONG" in response
        assert "50,000" in response or "50000" in response
        assert "Stop Loss" in response
        assert "TP1" in response
        assert "TP2" in response
        assert "STRONG" in response


class TestBuildLLMContext:
    def test_contains_data_injection_marker(self):
        result = ScoringResult(
            direction=Direction.BULLISH,
            confidence=55,
            signal_strength=SignalStrength.MARGINAL,
            raw_score=50,
        )
        context = build_llm_context(result, None)

        assert "FINANCIAL DATA INJECTION" in context
        assert "Do NOT invent" in context
        assert "JSON" in context
        assert '"decision"' in context

    def test_includes_target_data(self):
        result = ScoringResult(
            direction=Direction.BULLISH,
            confidence=55,
            signal_strength=SignalStrength.MARGINAL,
            raw_score=50,
        )
        targets = TradeTargets(
            entry=50000, stop_loss=49000, take_profit_1=52000,
            take_profit_2=54000, risk_amount=1000, reward_1=2000,
            reward_2=4000, direction=Direction.BULLISH,
        )
        context = build_llm_context(result, targets)
        assert "Entry" in context
        assert "Stop Loss" in context


class TestRouteSignal:
    def test_route_with_bullish_indicators(self):
        """Integration test: route a signal from indicators to decision."""
        ind = IndicatorSet(
            timeframe="4h",
            close=50000, open=49500, high=50200, low=49300,
            change_pct=1.0,
            ema_9=49800, ema_21=49500, ema_50=49000, ema_200=47000,
            sma_50=49100, sma_200=47200,
            adx=35, plus_di=30, minus_di=15,
            macd_line=150, macd_signal=100, macd_histogram=50,
            rsi_14=62, stoch_k=65, stoch_d=60,
            cci_20=80, williams_r=-35, roc_10=3.5,
            volume=800e6, volume_sma_20=500e6, volume_ratio=1.6,
            obv=1e10, obv_sma_20=9.5e9, vwap=49700,
            atr_14=800, atr_pct=1.6,
            bb_upper=51000, bb_middle=49500, bb_lower=48000,
            bb_width=6.06, bb_position=0.67,
            pivot=49500, support_1=48800, support_2=48000,
            resistance_1=50200, resistance_2=51000,
            nearest_support=48800, nearest_resistance=50200,
        )
        config = AppConfig()
        config.trading.leverage_tiers = {
            "conservative": LeverageTier()
        }
        config.trading.active_tier = "conservative"

        decision = route_signal({"4h": ind}, config)

        assert isinstance(decision, RoutingDecision)
        assert decision.scoring_result.direction in (Direction.BULLISH, Direction.BEARISH, Direction.NEUTRAL)
        assert decision.signal_strength in (SignalStrength.STRONG, SignalStrength.MARGINAL, SignalStrength.WAIT)
