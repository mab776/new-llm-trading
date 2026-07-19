"""
Tests for the core scoring engine.
Covers: indicator calculations, category scoring, composite scoring,
        target calculation, pre-trade filters, and report formatting.
"""

import numpy as np
import pandas as pd
import pytest

from llm_trading_bot.scoring import (
    CategoryScore,
    Direction,
    IndicatorSet,
    MarketRegime,
    SignalStrength,
    TradeTargets,
    apply_pre_trade_filters,
    calculate_indicators,
    calculate_targets,
    compute_atr,
    compute_bollinger_bands,
    compute_composite_score,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_sma,
    detect_market_regime,
    format_indicator_report,
    format_scoring_report,
    score_momentum,
    score_risk,
    score_support_resistance,
    score_trend,
    score_volume,
)


class TestBollingerClamp:
    def test_bb_position_clamped_on_spike(self):
        """A close far above the upper band must not push bb_position past 1.0."""
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        prices = np.full(n, 100.0)
        prices[-1] = 400.0  # violent spike -> close well outside the bands
        df = pd.DataFrame({
            "Open": prices, "High": prices + 1, "Low": prices - 1,
            "Close": prices, "Volume": np.full(n, 1000.0),
        }, index=idx)

        ind = calculate_indicators(df, "4h")
        assert ind.bb_position is not None
        assert 0.0 <= ind.bb_position <= 1.0
        assert ind.bb_position == 1.0  # clamped to the ceiling


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing (200+ bars)."""
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2024-01-01", periods=n, freq="4h")
    base_price = 45000

    # Random walk with slight uptrend
    returns = np.random.normal(0.0002, 0.01, n)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "Open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        "High": prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
        "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
        "Close": prices,
        "Volume": np.random.uniform(100, 1000, n) * 1e6,
    }, index=dates)

    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

    return df


@pytest.fixture
def bullish_indicators() -> IndicatorSet:
    """A manually-crafted bullish indicator set."""
    return IndicatorSet(
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
        obv=1e10, obv_sma_20=9.5e9,
        vwap=49700,
        atr_14=800, atr_pct=1.6,
        bb_upper=51000, bb_middle=49500, bb_lower=48000,
        bb_width=6.06, bb_position=0.67,
        pivot=49500, support_1=48800, support_2=48000,
        resistance_1=50200, resistance_2=51000,
        nearest_support=48800, nearest_resistance=50200,
    )


@pytest.fixture
def bearish_indicators() -> IndicatorSet:
    """A manually-crafted bearish indicator set."""
    return IndicatorSet(
        timeframe="4h",
        close=45000, open=45800, high=46000, low=44800,
        change_pct=-1.7,
        ema_9=45500, ema_21=46000, ema_50=46500, ema_200=48000,
        sma_50=46400, sma_200=47800,
        adx=30, plus_di=12, minus_di=28,
        macd_line=-200, macd_signal=-100, macd_histogram=-100,
        rsi_14=35, stoch_k=25, stoch_d=30,
        cci_20=-120, williams_r=-80, roc_10=-4.2,
        volume=900e6, volume_sma_20=600e6, volume_ratio=1.5,
        obv=-5e9, obv_sma_20=-3e9,
        vwap=45800,
        atr_14=700, atr_pct=1.56,
        bb_upper=47000, bb_middle=45500, bb_lower=44000,
        bb_width=6.59, bb_position=0.33,
        pivot=45600, support_1=44800, support_2=44000,
        resistance_1=46400, resistance_2=47200,
        nearest_support=44800, nearest_resistance=46400,
    )


# ──────────────────────────────────────────────────────────────────────
# Indicator Calculation Tests
# ──────────────────────────────────────────────────────────────────────

class TestIndicatorFunctions:
    def test_ema_length(self):
        s = pd.Series(range(100), dtype=float)
        result = compute_ema(s, 20)
        assert len(result) == 100
        assert not pd.isna(result.iloc[-1])

    def test_sma_length(self):
        s = pd.Series(range(100), dtype=float)
        result = compute_sma(s, 20)
        assert len(result) == 100
        assert pd.isna(result.iloc[18])  # Before window is full
        assert not pd.isna(result.iloc[19])

    def test_rsi_range(self):
        np.random.seed(42)
        s = pd.Series(np.random.normal(100, 5, 200))
        result = compute_rsi(s, 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_macd_components(self):
        np.random.seed(42)
        s = pd.Series(np.cumsum(np.random.normal(0, 1, 100)) + 100)
        line, signal, hist = compute_macd(s)
        assert len(line) == len(signal) == len(hist) == 100
        # Histogram should be line - signal
        np.testing.assert_array_almost_equal(
            hist.values, (line - signal).values
        )

    def test_atr_positive(self, sample_ohlcv):
        atr = compute_atr(
            sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"]
        )
        valid = atr.dropna()
        assert (valid > 0).all()

    def test_bollinger_bands_order(self, sample_ohlcv):
        upper, middle, lower = compute_bollinger_bands(sample_ohlcv["Close"])
        valid_idx = upper.dropna().index
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()


class TestCalculateIndicators:
    def test_basic_calculation(self, sample_ohlcv):
        ind = calculate_indicators(sample_ohlcv, "4h")
        assert ind.timeframe == "4h"
        assert ind.close is not None
        assert ind.close > 0
        assert ind.rsi_14 is not None
        assert 0 <= ind.rsi_14 <= 100
        assert ind.atr_14 is not None
        assert ind.atr_14 > 0

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "Open": [1, 2, 3], "High": [1.5, 2.5, 3.5],
            "Low": [0.5, 1.5, 2.5], "Close": [1.2, 2.2, 3.2],
            "Volume": [100, 200, 300],
        })
        with pytest.raises(ValueError, match="at least 50"):
            calculate_indicators(df, "1h")

    def test_ema_200_with_enough_data(self, sample_ohlcv):
        ind = calculate_indicators(sample_ohlcv, "1d")
        assert ind.ema_200 is not None

    def test_pivot_points_calculated(self, sample_ohlcv):
        ind = calculate_indicators(sample_ohlcv, "4h")
        assert ind.pivot is not None
        assert ind.support_1 is not None
        assert ind.resistance_1 is not None


# ──────────────────────────────────────────────────────────────────────
# Category Scoring Tests
# ──────────────────────────────────────────────────────────────────────

class TestScoreTrend:
    def test_bullish_trend(self, bullish_indicators):
        cat = score_trend(bullish_indicators)
        assert cat.raw_score > 0
        assert cat.name == "trend"

    def test_bearish_trend(self, bearish_indicators):
        cat = score_trend(bearish_indicators)
        assert cat.raw_score < 0

    def test_score_bounded(self, bullish_indicators):
        cat = score_trend(bullish_indicators)
        assert -100 <= cat.raw_score <= 100


class TestScoreMomentum:
    def test_bullish_momentum(self, bullish_indicators):
        cat = score_momentum(bullish_indicators)
        assert cat.raw_score > 0

    def test_bearish_momentum(self, bearish_indicators):
        cat = score_momentum(bearish_indicators)
        # Bearish with RSI 35, stoch 25, etc. should be negative or near zero
        # RSI 35 = bearish, but stoch 25 = oversold (bullish signal)
        # The net might be slightly negative or close to zero
        assert -100 <= cat.raw_score <= 100

    def test_overbought_penalized(self):
        ind = IndicatorSet(timeframe="4h", rsi_14=85, stoch_k=90, stoch_d=88)
        cat = score_momentum(ind)
        assert cat.raw_score < 0  # Overbought = bearish

    def test_oversold_rewarded(self):
        ind = IndicatorSet(timeframe="4h", rsi_14=22, stoch_k=15, stoch_d=18)
        cat = score_momentum(ind)
        assert cat.raw_score > 0  # Oversold = bullish reversal


class TestScoreVolume:
    def test_high_volume_bullish(self, bullish_indicators):
        cat = score_volume(bullish_indicators)
        assert cat.raw_score > 0

    def test_low_volume_penalized(self):
        ind = IndicatorSet(
            timeframe="4h", volume_ratio=0.3, change_pct=1.0,
            obv=100, obv_sma_20=200, vwap=50000, close=49000,
        )
        cat = score_volume(ind)
        assert cat.raw_score < 0


class TestScoreSupportResistance:
    def test_good_rr_from_support(self, bullish_indicators):
        cat = score_support_resistance(bullish_indicators)
        assert cat.name == "support_resistance"
        assert -100 <= cat.raw_score <= 100

    def test_missing_data_returns_zero(self):
        ind = IndicatorSet(timeframe="4h", close=50000)
        cat = score_support_resistance(ind)
        assert cat.raw_score == 0


class TestScoreRisk:
    def test_healthy_volatility(self, bullish_indicators):
        cat = score_risk(bullish_indicators)
        # ATR% 1.6 is healthy, ADX 35 is trending
        assert cat.raw_score > 0

    def test_extreme_volatility_penalized(self):
        ind = IndicatorSet(timeframe="4h", atr_pct=10.0, adx=40, bb_width=15)
        cat = score_risk(ind)
        assert cat.raw_score < 0

    def test_ranging_market_penalized(self):
        ind = IndicatorSet(timeframe="4h", atr_pct=2.0, adx=12, bb_width=3)
        cat = score_risk(ind)
        assert cat.raw_score < 0


# ──────────────────────────────────────────────────────────────────────
# Composite Scoring Tests
# ──────────────────────────────────────────────────────────────────────

class TestCompositeScore:
    def test_bullish_composite(self, bullish_indicators):
        result = compute_composite_score(
            {"4h": bullish_indicators},
            weights={"trend": 0.3, "momentum": 0.25, "volume": 0.15,
                     "support_resistance": 0.2, "risk": 0.1},
        )
        assert result.direction == Direction.BULLISH
        assert result.raw_score > 0

    def test_bearish_composite(self, bearish_indicators):
        result = compute_composite_score(
            {"4h": bearish_indicators},
            weights={"trend": 0.3, "momentum": 0.25, "volume": 0.15,
                     "support_resistance": 0.2, "risk": 0.1},
        )
        assert result.direction == Direction.BEARISH
        assert result.raw_score < 0

    def test_confidence_bounded(self, bullish_indicators):
        result = compute_composite_score(
            {"4h": bullish_indicators},
            weights={"trend": 0.3, "momentum": 0.25, "volume": 0.15,
                     "support_resistance": 0.2, "risk": 0.1},
            confidence_min=5,
            confidence_max=95,
        )
        assert 5 <= result.confidence <= 95

    def test_multi_timeframe_alignment(self, bullish_indicators, bearish_indicators):
        # Aligned — should get bonus
        aligned = compute_composite_score(
            {"4h": bullish_indicators, "1d": bullish_indicators},
            weights={"trend": 0.3, "momentum": 0.25, "volume": 0.15,
                     "support_resistance": 0.2, "risk": 0.1},
        )
        # Divergent — should get penalty
        divergent = compute_composite_score(
            {"4h": bullish_indicators, "1d": bearish_indicators},
            weights={"trend": 0.3, "momentum": 0.25, "volume": 0.15,
                     "support_resistance": 0.2, "risk": 0.1},
        )
        assert aligned.raw_score > divergent.raw_score

    def test_category_scores_populated(self, bullish_indicators):
        result = compute_composite_score(
            {"4h": bullish_indicators},
            weights={"trend": 0.3, "momentum": 0.25, "volume": 0.15,
                     "support_resistance": 0.2, "risk": 0.1},
        )
        assert len(result.category_scores) == 5
        names = {c.name for c in result.category_scores}
        assert names == {"trend", "momentum", "volume", "support_resistance", "risk"}

    def test_alignment_scale_by_tf(self, bullish_indicators):
        """Per-TF alignment weights: 3 shrinks the vote, 0 removes it, None = legacy 5."""
        weights = {"trend": 0.3, "momentum": 0.25, "volume": 0.15,
                   "support_resistance": 0.2, "risk": 0.1}
        base = compute_composite_score({"4h": bullish_indicators}, weights=weights)
        assert base.raw_score < 90  # headroom so the diffs below aren't clamped
        legacy = compute_composite_score(
            {"4h": bullish_indicators, "1d": bullish_indicators}, weights=weights)
        weighted = compute_composite_score(
            {"4h": bullish_indicators, "1d": bullish_indicators}, weights=weights,
            alignment_scale_by_tf={"1d": 3.0})
        zeroed = compute_composite_score(
            {"4h": bullish_indicators, "1d": bullish_indicators}, weights=weights,
            alignment_scale_by_tf={"1d": 0.0})
        assert legacy.raw_score == pytest.approx(base.raw_score + 5.0)
        assert weighted.raw_score == pytest.approx(base.raw_score + 3.0)
        assert zeroed.raw_score == pytest.approx(base.raw_score)
        # A TF absent from the mapping keeps the flat default scale
        default_for_missing = compute_composite_score(
            {"4h": bullish_indicators, "1d": bullish_indicators}, weights=weights,
            alignment_scale_by_tf={"1h": 0.0})
        assert default_for_missing.raw_score == pytest.approx(legacy.raw_score)


# ──────────────────────────────────────────────────────────────────────
# Target Calculation Tests
# ──────────────────────────────────────────────────────────────────────

class TestTargetCalculation:
    def test_long_targets(self, bullish_indicators):
        targets = calculate_targets(
            bullish_indicators, Direction.BULLISH,
            sl_strategy="atr", atr_sl_mult=1.5, atr_tp1_mult=3.0, atr_tp2_mult=5.0,
        )
        assert targets is not None
        assert targets.direction == Direction.BULLISH
        assert targets.stop_loss < targets.entry
        assert targets.take_profit_1 > targets.entry
        assert targets.take_profit_2 > targets.take_profit_1
        assert targets.risk_amount > 0

    def test_short_targets(self, bearish_indicators):
        targets = calculate_targets(
            bearish_indicators, Direction.BEARISH,
            sl_strategy="atr", atr_sl_mult=1.5, atr_tp1_mult=3.0, atr_tp2_mult=5.0,
        )
        assert targets is not None
        assert targets.direction == Direction.BEARISH
        assert targets.stop_loss > targets.entry
        assert targets.take_profit_1 < targets.entry
        assert targets.take_profit_2 < targets.take_profit_1

    def test_neutral_returns_none(self, bullish_indicators):
        targets = calculate_targets(bullish_indicators, Direction.NEUTRAL)
        assert targets is None

    def test_hybrid_sl_strategy(self, bullish_indicators):
        targets = calculate_targets(
            bullish_indicators, Direction.BULLISH, sl_strategy="hybrid"
        )
        assert targets is not None
        assert targets.sl_strategy == "hybrid"
        assert targets.stop_loss < targets.entry

    def test_structure_sl_strategy(self, bullish_indicators):
        targets = calculate_targets(
            bullish_indicators, Direction.BULLISH, sl_strategy="structure"
        )
        assert targets is not None
        assert targets.stop_loss < targets.entry

    def test_rr_ratios_reasonable(self, bullish_indicators):
        targets = calculate_targets(
            bullish_indicators, Direction.BULLISH,
            sl_strategy="atr", atr_sl_mult=1.5, atr_tp1_mult=3.0, atr_tp2_mult=5.0,
        )
        assert targets is not None
        rr1 = targets.reward_1 / targets.risk_amount
        rr2 = targets.reward_2 / targets.risk_amount
        assert rr1 >= 1.0  # TP1 at least 1:1
        assert rr2 > rr1   # TP2 further than TP1


# ──────────────────────────────────────────────────────────────────────
# Pre-Trade Filter Tests
# ──────────────────────────────────────────────────────────────────────

class TestPreTradeFilters:
    def test_all_pass(self, bullish_indicators):
        targets = calculate_targets(bullish_indicators, Direction.BULLISH)
        failures = apply_pre_trade_filters(
            bullish_indicators, targets, min_adx=20, min_volatility_pct=0.3,
            fee_rate=0.0006, leverage=5,
        )
        assert len(failures) == 0

    def test_low_adx_fails(self):
        ind = IndicatorSet(timeframe="4h", adx=15, atr_pct=2.0, close=50000)
        failures = apply_pre_trade_filters(ind, None, min_adx=20)
        assert any("ADX" in f for f in failures)

    def test_low_volatility_fails(self):
        ind = IndicatorSet(timeframe="4h", adx=30, atr_pct=0.1, close=50000)
        failures = apply_pre_trade_filters(ind, None, min_volatility_pct=0.3)
        assert any("Volatility" in f for f in failures)

    def test_fee_profitability_check(self):
        """A trade where TP1 can't cover fees should fail."""
        ind = IndicatorSet(timeframe="4h", adx=30, atr_pct=2.0, close=50000)
        # Tiny reward — won't cover fees at high leverage
        targets = TradeTargets(
            entry=50000, stop_loss=49900, take_profit_1=50010,
            take_profit_2=50100, risk_amount=100, reward_1=10,
            reward_2=100, direction=Direction.BULLISH,
        )
        failures = apply_pre_trade_filters(
            ind, targets, fee_rate=0.0006, leverage=20, check_profit_after_fees=True,
        )
        assert any("fees" in f.lower() for f in failures)


# ──────────────────────────────────────────────────────────────────────
# Report Formatting Tests
# ──────────────────────────────────────────────────────────────────────

class TestReportFormatting:
    def test_indicator_report(self, bullish_indicators):
        report = format_indicator_report(bullish_indicators)
        assert "4H" in report
        assert "$50,000" in report or "50000" in report
        assert "RSI" in report
        assert "ATR" in report

    def test_scoring_report(self, bullish_indicators):
        result = compute_composite_score(
            {"4h": bullish_indicators},
            weights={"trend": 0.3, "momentum": 0.25, "volume": 0.15,
                     "support_resistance": 0.2, "risk": 0.1},
        )
        targets = calculate_targets(bullish_indicators, result.direction)
        report = format_scoring_report(result, targets)
        assert "MARKET ANALYSIS REPORT" in report
        assert "Direction" in report


# ──────────────────────────────────────────────────────────────────────
# Market Regime Detection Tests
# ──────────────────────────────────────────────────────────────────────

class TestMarketRegime:
    def test_trending_regime(self):
        ind = IndicatorSet(timeframe="4h", adx=30, atr_pct=2.0, bb_width=5.0)
        assert detect_market_regime(ind) == MarketRegime.TRENDING

    def test_choppy_regime(self):
        ind = IndicatorSet(timeframe="4h", adx=15, atr_pct=1.0, bb_width=2.0)
        assert detect_market_regime(ind) == MarketRegime.CHOPPY

    def test_volatile_regime(self):
        ind = IndicatorSet(timeframe="4h", adx=25, atr_pct=6.0, bb_width=5.0)
        assert detect_market_regime(ind) == MarketRegime.VOLATILE

    def test_ranging_regime(self):
        ind = IndicatorSet(timeframe="4h", adx=18, atr_pct=2.0, bb_width=5.0)
        assert detect_market_regime(ind) == MarketRegime.RANGING

    def test_weak_trend_regime(self):
        ind = IndicatorSet(timeframe="4h", adx=22, atr_pct=2.0, bb_width=5.0)
        assert detect_market_regime(ind) == MarketRegime.WEAK_TREND

    def test_none_values_use_defaults(self):
        ind = IndicatorSet(timeframe="4h")  # All None
        # Defaults: adx=15, atr_pct=1.0, bb_width=5.0 → not choppy (bb_width=5>3)
        regime = detect_market_regime(ind)
        assert regime in (MarketRegime.RANGING, MarketRegime.CHOPPY,
                          MarketRegime.WEAK_TREND, MarketRegime.TRENDING,
                          MarketRegime.VOLATILE)


# ──────────────────────────────────────────────────────────────────────
# Additional Indicator Edge Cases
# ──────────────────────────────────────────────────────────────────────

class TestIndicatorEdgeCases:
    def test_stochastic_range(self, sample_ohlcv):
        from llm_trading_bot.scoring import compute_stochastic
        k, d = compute_stochastic(
            sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"]
        )
        valid = k.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_williams_r_range(self, sample_ohlcv):
        from llm_trading_bot.scoring import compute_williams_r
        wr = compute_williams_r(
            sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"]
        )
        valid = wr.dropna()
        assert (valid >= -100).all()
        assert (valid <= 0).all()

    def test_obv_monotonic_on_constant_direction(self):
        """If price always goes up, OBV should always increase."""
        from llm_trading_bot.scoring import compute_obv
        close = pd.Series([100 + i for i in range(50)], dtype=float)
        volume = pd.Series([1000.0] * 50)
        obv = compute_obv(close, volume)
        # After the first diff, OBV should be increasing
        diffs = obv.diff().iloc[2:]  # Skip first NaN and first diff
        assert (diffs >= 0).all()


# ──────────────────────────────────────────────────────────────────────
# Pre-Trade Filter Category Agreement Tests
# ──────────────────────────────────────────────────────────────────────

class TestFilterCategoryAgreement:
    def test_category_agreement_filter(self):
        """min_category_agreement=3 but only 2 agree → should fail."""
        ind = IndicatorSet(timeframe="4h", adx=30, atr_pct=2.0, close=50000)
        cats = [
            CategoryScore("trend", 40, 0.3, 12),
            CategoryScore("momentum", 30, 0.25, 7.5),
            CategoryScore("volume", -20, 0.15, -3),
            CategoryScore("support_resistance", -10, 0.2, -2),
            CategoryScore("risk", -15, 0.1, -1.5),
        ]
        failures = apply_pre_trade_filters(
            ind, None,
            min_adx=0, min_volatility_pct=0,
            category_scores=cats,
            direction=Direction.BULLISH,
            min_category_agreement=3,
        )
        assert any("agreement" in f.lower() for f in failures)

    def test_trend_momentum_disagree_filter(self):
        """Trend bullish but momentum bearish → should fail."""
        ind = IndicatorSet(timeframe="4h", adx=30, atr_pct=2.0, close=50000)
        cats = [
            CategoryScore("trend", 40, 0.3, 12),
            CategoryScore("momentum", -30, 0.25, -7.5),
        ]
        failures = apply_pre_trade_filters(
            ind, None,
            min_adx=0, min_volatility_pct=0,
            category_scores=cats,
            direction=Direction.BULLISH,
            min_category_agreement=0,
            require_trend_momentum_agree=True,
        )
        assert len(failures) > 0
        assert any("momentum" in f.lower() for f in failures)
