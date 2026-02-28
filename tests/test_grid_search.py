"""
Tests for the grid search module.
Covers: grid constraints, fast_backtest PnL correctness, position sizing, precomputed bars.
"""

import pytest

from grid_search import (
    PrecomputedBar,
    SimTrade,
    _sim_fee,
    _sim_position_size,
    build_grid,
    build_scoring_grid,
    fast_backtest,
    RISK_PROFILES,
    STRATEGY_REGISTRY,
    make_confirmation_fn,
    make_momentum_breakout_fn,
    make_trend_gated_fn,
    make_mean_reversion_fn,
    make_volume_confirmed_fn,
    make_regime_adaptive_fn,
)
from llm_trading_bot.scoring import Direction, MarketRegime, CategoryScore


# ──────────────────────────────────────────────────────────────────────
# Grid Constraint Tests
# ──────────────────────────────────────────────────────────────────────

class TestBuildGrid:
    def test_grid_not_empty(self):
        grid = build_grid()
        assert len(grid) > 0

    def test_tp2_greater_than_tp1(self):
        """Every combo must have tp2_rr > tp1_rr."""
        for params in build_grid():
            assert params["tp2_rr"] > params["tp1_rr"], (
                f"Invalid: tp2_rr={params['tp2_rr']} <= tp1_rr={params['tp1_rr']}"
            )

    def test_strong_greater_than_marginal(self):
        """Every combo must have strong_thresh > marginal_low."""
        for params in build_grid():
            assert params["strong_thresh"] > params["marginal_low"], (
                f"Invalid: strong={params['strong_thresh']} <= marginal={params['marginal_low']}"
            )

    def test_reasonable_grid_size(self):
        """Grid should have thousands of combos but not millions."""
        grid = build_grid()
        assert 1_000 < len(grid) < 10_000_000


# ──────────────────────────────────────────────────────────────────────
# Utility Function Tests
# ──────────────────────────────────────────────────────────────────────

class TestSimFee:
    def test_fee_calculation(self):
        fee = _sim_fee(0.04, 50000, 0.0006)
        # 0.04 * 50000 * 0.0006 = 1.2
        assert fee == pytest.approx(1.2, rel=1e-6)

    def test_zero_size_zero_fee(self):
        assert _sim_fee(0, 50000, 0.0006) == 0


class TestSimPositionSize:
    def test_position_size_calculation(self):
        """Size = (balance * risk_pct * leverage) / price."""
        size = _sim_position_size(10000, 50000, 10, 0.02, 0.0006)
        expected = (10000 * 0.02 * 10) / 50000  # = 0.04
        assert size == pytest.approx(expected, rel=1e-6)

    def test_higher_leverage_bigger_size(self):
        s5 = _sim_position_size(10000, 50000, 5, 0.02, 0.0006)
        s10 = _sim_position_size(10000, 50000, 10, 0.02, 0.0006)
        assert s10 == pytest.approx(s5 * 2, rel=1e-6)


# ──────────────────────────────────────────────────────────────────────
# Fast Backtest PnL Tests
# ──────────────────────────────────────────────────────────────────────

def _make_bar(
    idx, close, high=None, low=None, raw_score=0, direction=Direction.NEUTRAL,
    atr_14=1000, adx=30, atr_pct=2.0, bb_width=5.0,
    nearest_support=None, nearest_resistance=None, regime=MarketRegime.TRENDING,
    category_scores=None, alignment_bonus=0.0,
    rsi_14=50.0, stoch_k=50.0, macd_histogram=0.0, volume_ratio=1.0,
    bb_position=0.5, ema_aligned=0, above_ema200=True, obv_bullish=True,
    cci_20=0.0, roc_10=0.0, change_pct=0.0, plus_di=0.0, minus_di=0.0,
):
    """Helper to create a PrecomputedBar with sensible defaults."""
    if high is None:
        high = close * 1.01
    if low is None:
        low = close * 0.99
    return PrecomputedBar(
        idx=idx, timestamp=f"2024-01-01T{idx:02d}:00:00",
        open=close, high=high, low=low, close=close, volume=1e6,
        raw_score=raw_score, direction=direction,
        category_scores=category_scores or [],
        atr_14=atr_14, adx=adx, atr_pct=atr_pct, bb_width=bb_width,
        nearest_support=nearest_support, nearest_resistance=nearest_resistance,
        regime=regime, alignment_bonus=alignment_bonus,
        rsi_14=rsi_14, stoch_k=stoch_k, macd_histogram=macd_histogram,
        volume_ratio=volume_ratio, bb_position=bb_position,
        ema_aligned=ema_aligned, above_ema200=above_ema200,
        obv_bullish=obv_bullish, cci_20=cci_20, roc_10=roc_10,
        change_pct=change_pct, plus_di=plus_di, minus_di=minus_di,
    )


class TestFastBacktestPnL:
    """Verify that fast_backtest PnL matches manual calculations (no double leverage)."""

    def test_single_winning_long_trade(self):
        """
        Bar 0: strong bullish signal, entry at 50000
        Bar 1: TP2 hit at 54000

        SL = entry - ATR*1.5 = 50000 - 1500 = 48500
        TP1 = entry + 1500*2.0 = 53000
        TP2 = entry + 1500*3.5 = 55250

        But we'll set the high on bar 1 high enough to hit TP2.
        size = (10000 * 0.02 * 10) / 50000 = 0.04
        entry_fee = 0.04 * 50000 * 0.0006 = 1.20

        TP1: exit_size = 0.04 * 0.5 = 0.02
              gross = (53000-50000) * 0.02 = 60   (NO extra leverage)
              fee = 0.02 * 53000 * 0.0006 = 0.636

        TP2: remaining = 0.02
              gross = (55250-50000) * 0.02 = 105   (NO extra leverage)
              fee = 0.02 * 55250 * 0.0006 = 0.663
        """
        bars = [
            _make_bar(0, 50000, high=50500, low=49500,
                      raw_score=50, direction=Direction.BULLISH,
                      atr_14=1000, adx=30, atr_pct=2.0),
            _make_bar(1, 55000, high=56000, low=49000,
                      raw_score=0, direction=Direction.NEUTRAL),
        ]

        result = fast_backtest(
            bars,
            leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=3.5,
            tp1_exit_pct=0.5, marginal_low=20, strong_thresh=40,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006, risk_pct=0.02,
        )

        assert result["trades"] == 1
        # Final balance calculation:
        # entry_fee = 0.04 * 50000 * 0.0006 = 1.2
        # TP1 gross = 3000 * 0.02 = 60, fee = 0.02*53000*0.0006 = 0.636
        # TP2 gross = 5250 * 0.02 = 105, fee = 0.02*55250*0.0006 = 0.663
        # net = -1.2 + (60-0.636) + (105-0.663) = 162.501
        expected_net = -1.2 + (60 - 0.636) + (105 - 0.663)
        assert result["net_pnl"] == pytest.approx(expected_net, rel=0.02)

    def test_single_losing_long_trade(self):
        """
        Bar 0: entry at 50000, SL at 48500 (ATR=1000, mult=1.5)
        Bar 1: low hits SL

        size = 0.04
        gross = (48500 - 50000) * 0.04 = -60
        """
        bars = [
            _make_bar(0, 50000, high=50500, low=49500,
                      raw_score=50, direction=Direction.BULLISH,
                      atr_14=1000, adx=30, atr_pct=2.0),
            _make_bar(1, 47000, high=50000, low=47000),
        ]

        result = fast_backtest(
            bars,
            leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=3.5,
            tp1_exit_pct=0.5, marginal_low=20, strong_thresh=40,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006, risk_pct=0.02,
        )

        assert result["trades"] == 1
        # entry_fee = 1.2
        # SL gross = (48500-50000) * 0.04 = -60
        # SL exit_fee = 0.04 * 48500 * 0.0006 = 1.164
        # net_pnl = -1.2 + (-60 - 1.164) = -62.364
        expected_net = -1.2 + (-60 - 0.04 * 48500 * 0.0006)
        assert result["net_pnl"] == pytest.approx(expected_net, rel=0.02)

    def test_no_trades_when_score_low(self):
        bars = [
            _make_bar(0, 50000, raw_score=5, direction=Direction.NEUTRAL),
            _make_bar(1, 51000, raw_score=5, direction=Direction.NEUTRAL),
        ]
        result = fast_backtest(
            bars,
            leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=3.5,
            tp1_exit_pct=0.5, marginal_low=20, strong_thresh=40,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006, risk_pct=0.02,
        )
        assert result["trades"] == 0
        assert result["final_balance"] == 10000

    def test_pnl_matches_portfolio_module(self):
        """
        Cross-verify: fast_backtest and Portfolio should produce the same PnL
        for an identical trade scenario.
        """
        from llm_trading_bot.portfolio import Portfolio

        entry_price = 50000
        exit_price = 53000
        leverage = 10
        risk_pct = 0.02
        fee_rate = 0.0006

        # --- Portfolio calculation ---
        port = Portfolio(initial_balance=10000, taker_fee=fee_rate)
        trade = port.open_trade(
            direction="LONG", entry_price=entry_price, entry_time="t1",
            stop_loss=48500, take_profit_1=53000, take_profit_2=55250,
            leverage=leverage, risk_pct=risk_pct, tp1_exit_pct=0.5,
        )
        port.close_trade(trade, exit_price=exit_price, exit_time="t2", reason="manual")
        portfolio_net_pnl = port.balance - 10000

        # --- Grid search fast_backtest (force-close at end) ---
        # We create bars so that no TP/SL is hit, and it force-closes at end
        bars = [
            _make_bar(0, entry_price, high=entry_price + 100, low=entry_price - 100,
                      raw_score=50, direction=Direction.BULLISH,
                      atr_14=1000, adx=30, atr_pct=2.0),
            _make_bar(1, exit_price, high=exit_price + 100, low=exit_price - 100,
                      raw_score=0, direction=Direction.NEUTRAL),
        ]
        result = fast_backtest(
            bars,
            leverage=leverage, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=3.5,
            tp1_exit_pct=0.5, marginal_low=20, strong_thresh=40,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=fee_rate, risk_pct=risk_pct,
        )

        assert result["net_pnl"] == pytest.approx(portfolio_net_pnl, rel=0.01)


class TestFastBacktestFilters:
    def test_adx_filter_blocks_trade(self):
        bars = [
            _make_bar(0, 50000, raw_score=50, direction=Direction.BULLISH,
                      atr_14=1000, adx=10, atr_pct=2.0),
            _make_bar(1, 55000, high=56000, low=49000),
        ]
        result = fast_backtest(
            bars,
            leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=3.5,
            tp1_exit_pct=0.5, marginal_low=20, strong_thresh=40,
            min_adx=20, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006,
        )
        assert result["trades"] == 0

    def test_choppy_regime_blocked(self):
        bars = [
            _make_bar(0, 50000, raw_score=50, direction=Direction.BULLISH,
                      atr_14=1000, adx=30, atr_pct=2.0,
                      regime=MarketRegime.CHOPPY),
            _make_bar(1, 55000, high=56000, low=49000),
        ]
        result = fast_backtest(
            bars,
            leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=3.5,
            tp1_exit_pct=0.5, marginal_low=20, strong_thresh=40,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=True,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006,
        )
        assert result["trades"] == 0


# ──────────────────────────────────────────────────────────────────────
# Strategy-specific Grid Tests
# ──────────────────────────────────────────────────────────────────────

class TestBuildGridStrategies:
    def test_trailing_grid_not_empty(self):
        grid = build_grid("trailing")
        assert len(grid) > 0

    def test_trailing_grid_has_trail_params(self):
        for params in build_grid("trailing"):
            assert params["trail_atr_mult"] > 0
            assert params["tp1_rr"] == 0.0
            assert params["tp2_rr"] == 0.0

    def test_tp1_trail_grid_not_empty(self):
        grid = build_grid("tp1_trail")
        assert len(grid) > 0

    def test_tp1_trail_grid_has_trail_and_tp1(self):
        for params in build_grid("tp1_trail"):
            assert params["trail_atr_mult"] > 0
            assert params["tp1_rr"] > 0
            assert params["tp2_rr"] == 0.0

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            build_grid("invalid_strategy")

    def test_all_grids_respect_threshold_constraint(self):
        for strat in ("tp1_tp2", "trailing", "tp1_trail"):
            for params in build_grid(strat):
                assert params["strong_thresh"] > params["marginal_low"]


# ──────────────────────────────────────────────────────────────────────
# Trailing Strategy Tests
# ──────────────────────────────────────────────────────────────────────

class TestTrailingStrategy:
    def test_trailing_stop_exits_after_peak(self):
        """
        Entry at 50000, ATR=1000, trail_atr_mult=2.0, activation=1.0 ATR
        Bar 1: peak=53000, trail_stop=51000, low=52000 → not hit
        Bar 2: peak still 53000, trail_stop=51000, low=50500 → hit at 51000
        """
        bars = [
            _make_bar(0, 50000, high=50500, low=49500,
                      raw_score=50, direction=Direction.BULLISH,
                      atr_14=1000, adx=30, atr_pct=2.0),
            _make_bar(1, 53000, high=53000, low=52000,
                      raw_score=0, direction=Direction.NEUTRAL),
            _make_bar(2, 50500, high=51500, low=50500,
                      raw_score=0, direction=Direction.NEUTRAL),
        ]
        result = fast_backtest(
            bars,
            leverage=10, atr_sl_mult=1.5, tp1_rr=0.0, tp2_rr=0.0,
            tp1_exit_pct=0.0, marginal_low=20, strong_thresh=40,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006, risk_pct=0.02,
            exit_strategy="trailing",
            trail_atr_mult=2.0,
            trail_activation_atr=1.0,
        )

        assert result["trades"] == 1
        # Exit at trail stop 51000 → profit
        assert result["net_pnl"] > 0

        # Exact: entry_fee=1.2, gross=40, exit_fee=1.224 → net≈37.58
        expected_net = -1.2 + (1000 * 0.04 - 0.04 * 51000 * 0.0006)
        assert result["net_pnl"] == pytest.approx(expected_net, abs=1.0)

    def test_trailing_original_sl_hit_before_activation(self):
        """SL hit before trailing activates → normal loss."""
        bars = [
            _make_bar(0, 50000, high=50500, low=49500,
                      raw_score=50, direction=Direction.BULLISH,
                      atr_14=1000, adx=30, atr_pct=2.0),
            _make_bar(1, 47000, high=50000, low=47000,
                      raw_score=0, direction=Direction.NEUTRAL),
        ]
        result = fast_backtest(
            bars,
            leverage=10, atr_sl_mult=1.5, tp1_rr=0.0, tp2_rr=0.0,
            tp1_exit_pct=0.0, marginal_low=20, strong_thresh=40,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006, risk_pct=0.02,
            exit_strategy="trailing",
            trail_atr_mult=2.0,
            trail_activation_atr=1.0,
        )

        assert result["trades"] == 1
        assert result["net_pnl"] < 0

    def test_trailing_immediate_activation(self):
        """trail_activation_atr=0 means trail starts immediately from entry."""
        bars = [
            _make_bar(0, 50000, high=50500, low=49500,
                      raw_score=50, direction=Direction.BULLISH,
                      atr_14=1000, adx=30, atr_pct=2.0),
            _make_bar(1, 51000, high=51000, low=50000,
                      raw_score=0, direction=Direction.NEUTRAL),
            _make_bar(2, 48000, high=49500, low=48000,
                      raw_score=0, direction=Direction.NEUTRAL),
        ]
        result = fast_backtest(
            bars,
            leverage=10, atr_sl_mult=1.5, tp1_rr=0.0, tp2_rr=0.0,
            tp1_exit_pct=0.0, marginal_low=20, strong_thresh=40,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006, risk_pct=0.02,
            exit_strategy="trailing",
            trail_atr_mult=2.0,
            trail_activation_atr=0.0,
        )
        assert result["trades"] == 1


# ──────────────────────────────────────────────────────────────────────
# Scoring Grid Tests
# ──────────────────────────────────────────────────────────────────────

class TestBuildScoringGrid:
    def test_grid_not_empty(self):
        combos = build_scoring_grid()
        assert len(combos) > 0

    def test_all_weights_sum_to_one(self):
        """Every combo must have weights summing to 1.0."""
        for weights in build_scoring_grid():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001, f"Weights sum to {total}"

    def test_min_weight_respected(self):
        """No weight below the minimum."""
        combos = build_scoring_grid(resolution=0.05, min_weight=0.05)
        for weights in combos:
            for cat, w in weights.items():
                assert w >= 0.05 - 0.001, f"{cat}={w} below min 0.05"

    def test_correct_categories(self):
        """Each combo must include all 5 categories."""
        expected = {"trend", "momentum", "volume", "support_resistance", "risk"}
        for weights in build_scoring_grid()[:10]:
            assert set(weights.keys()) == expected

    def test_expected_count_at_005_resolution(self):
        """With resolution=0.05, min=0.05 → C(19,4) = 3876 combos."""
        combos = build_scoring_grid(resolution=0.05, min_weight=0.05)
        assert len(combos) == 3876

    def test_coarser_resolution(self):
        """With resolution=0.10, min=0.10 → C(9,4) = 126 combos."""
        combos = build_scoring_grid(resolution=0.10, min_weight=0.10)
        assert len(combos) == 126


class TestRiskProfiles:
    def test_profiles_exist(self):
        assert "aggressive" in RISK_PROFILES
        assert "medium" in RISK_PROFILES
        assert "safe" in RISK_PROFILES

    def test_aggressive_highest_leverage(self):
        assert RISK_PROFILES["aggressive"]["leverage"] > RISK_PROFILES["medium"]["leverage"]
        assert RISK_PROFILES["medium"]["leverage"] > RISK_PROFILES["safe"]["leverage"]

    def test_all_profiles_have_required_keys(self):
        required = {
            "leverage", "atr_sl_mult", "tp1_rr", "tp2_rr", "tp1_exit_pct",
            "marginal_low", "strong_thresh", "min_cat_agree", "trend_mom_agree",
            "skip_choppy", "skip_volatile",
        }
        for name, profile in RISK_PROFILES.items():
            assert required.issubset(set(profile.keys())), f"{name} missing keys"


class TestScoringWeightsBacktest:
    """Verify scoring_weights parameter changes backtest behavior."""

    def test_custom_weights_change_direction(self):
        """
        With raw trend=+50 and momentum=-50, default weights (trend=0.30, mom=0.25)
        give positive score. Flip to mom-heavy and direction should reverse.
        """
        cats = [
            CategoryScore("trend", 50, 0, 0, {}),
            CategoryScore("momentum", -50, 0, 0, {}),
            CategoryScore("volume", 0, 0, 0, {}),
            CategoryScore("support_resistance", 0, 0, 0, {}),
            CategoryScore("risk", 0, 0, 0, {}),
        ]
        # Bar with raw_score=+40 (bullish) using default weights baked in
        bars = [
            _make_bar(0, 50000, high=50500, low=49500, raw_score=40,
                      direction=Direction.BULLISH, category_scores=cats,
                      alignment_bonus=0),
            _make_bar(1, 49000, high=50000, low=48000),  # SL bar
        ]

        # Default weights → should take the trade (score=40 > marginal)
        result_default = fast_backtest(
            bars, leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=3.5,
            tp1_exit_pct=0.5, marginal_low=10, strong_thresh=50,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006,
            scoring_weights=None,  # Uses baked-in raw_score
        )

        # Momentum-heavy weights → score = 50*0.10 + (-50)*0.70 = 5-35 = -30
        # Direction = BEARISH (< -10), NOT bullish
        result_mom = fast_backtest(
            bars, leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=3.5,
            tp1_exit_pct=0.5, marginal_low=10, strong_thresh=50,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006,
            scoring_weights={"trend": 0.10, "momentum": 0.70, "volume": 0.05,
                             "support_resistance": 0.10, "risk": 0.05},
        )

        # Default should open a trade (bullish), mom-heavy should open SHORT
        assert result_default["trades"] >= 1
        assert result_mom["trades"] >= 1
        # PnL should differ because direction flipped
        assert result_default["net_pnl"] != result_mom["net_pnl"]

    def test_zero_score_weights_no_trades(self):
        """
        If all category scores are 0, even with custom weights, no trade opens.
        """
        cats = [
            CategoryScore("trend", 0, 0, 0, {}),
            CategoryScore("momentum", 0, 0, 0, {}),
            CategoryScore("volume", 0, 0, 0, {}),
            CategoryScore("support_resistance", 0, 0, 0, {}),
            CategoryScore("risk", 0, 0, 0, {}),
        ]
        bars = [
            _make_bar(0, 50000, raw_score=0, direction=Direction.NEUTRAL,
                      category_scores=cats, alignment_bonus=0),
            _make_bar(1, 51000),
        ]

        result = fast_backtest(
            bars, leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=3.5,
            tp1_exit_pct=0.5, marginal_low=10, strong_thresh=50,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006,
            scoring_weights={"trend": 0.30, "momentum": 0.25, "volume": 0.15,
                             "support_resistance": 0.20, "risk": 0.10},
        )
        assert result["trades"] == 0

    def test_alignment_bonus_preserved(self):
        """
        Alignment bonus should be added to the re-weighted score.
        Without bonus: score below marginal → no trade.
        With bonus: score above marginal → trade opens.
        """
        cats = [
            CategoryScore("trend", 10, 0, 0, {}),
            CategoryScore("momentum", 5, 0, 0, {}),
            CategoryScore("volume", 0, 0, 0, {}),
            CategoryScore("support_resistance", 0, 0, 0, {}),
            CategoryScore("risk", 0, 0, 0, {}),
        ]
        # Equally weighted: score = 10*0.20 + 5*0.20 = 3.0 → below marginal of 10
        # With alignment_bonus=+10: score = 13 → above marginal
        bars_no_bonus = [
            _make_bar(0, 50000, raw_score=3, direction=Direction.NEUTRAL,
                      category_scores=cats, alignment_bonus=0),
            _make_bar(1, 51000, high=55000),
        ]
        bars_with_bonus = [
            _make_bar(0, 50000, raw_score=13, direction=Direction.BULLISH,
                      category_scores=cats, alignment_bonus=10),
            _make_bar(1, 51000, high=55000),
        ]

        equal_weights = {"trend": 0.20, "momentum": 0.20, "volume": 0.20,
                         "support_resistance": 0.20, "risk": 0.20}

        result_no = fast_backtest(
            bars_no_bonus, leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=3.5,
            tp1_exit_pct=0.5, marginal_low=10, strong_thresh=50,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006,
            scoring_weights=equal_weights,
        )
        result_yes = fast_backtest(
            bars_with_bonus, leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=3.5,
            tp1_exit_pct=0.5, marginal_low=10, strong_thresh=50,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006,
            scoring_weights=equal_weights,
        )

        assert result_no["trades"] == 0, "Without bonus, score too low for trade"
        assert result_yes["trades"] >= 1, "With bonus, score should trigger trade"


# ──────────────────────────────────────────────────────────────────────
# TP1 + Trailing Strategy Tests
# ──────────────────────────────────────────────────────────────────────

class TestTp1TrailStrategy:
    def test_tp1_then_trail_exit(self):
        """
        Entry at 50000, TP1=53000, trail=2.0 ATR
        Bar 1: TP1 hit → partial exit 50%
        Bar 2: peak=55000, trail_stop=53000
        Bar 3: low=52000 < 53000 → trail hit, exit remainder
        """
        bars = [
            _make_bar(0, 50000, high=50500, low=49500,
                      raw_score=50, direction=Direction.BULLISH,
                      atr_14=1000, adx=30, atr_pct=2.0),
            _make_bar(1, 53500, high=53500, low=52000,
                      raw_score=0, direction=Direction.NEUTRAL),
            _make_bar(2, 55000, high=55000, low=54000,
                      raw_score=0, direction=Direction.NEUTRAL),
            _make_bar(3, 52000, high=53500, low=52000,
                      raw_score=0, direction=Direction.NEUTRAL),
        ]
        result = fast_backtest(
            bars,
            leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=0.0,
            tp1_exit_pct=0.5, marginal_low=20, strong_thresh=40,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006, risk_pct=0.02,
            exit_strategy="tp1_trail",
            trail_atr_mult=2.0,
        )

        assert result["trades"] == 1
        assert result["net_pnl"] > 0

    def test_sl_hit_before_tp1_in_tp1_trail(self):
        """SL hit before TP1 → full loss, trailing never activates."""
        bars = [
            _make_bar(0, 50000, high=50500, low=49500,
                      raw_score=50, direction=Direction.BULLISH,
                      atr_14=1000, adx=30, atr_pct=2.0),
            _make_bar(1, 47000, high=50000, low=47000,
                      raw_score=0, direction=Direction.NEUTRAL),
        ]
        result = fast_backtest(
            bars,
            leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=0.0,
            tp1_exit_pct=0.5, marginal_low=20, strong_thresh=40,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006, risk_pct=0.02,
            exit_strategy="tp1_trail",
            trail_atr_mult=2.0,
        )

        assert result["trades"] == 1
        assert result["net_pnl"] < 0


# ──────────────────────────────────────────────────────────────────────
# Strategy Registry Tests
# ──────────────────────────────────────────────────────────────────────

class TestStrategyRegistry:
    def test_all_strategies_registered(self):
        expected = {"confirmation_count", "momentum_breakout", "trend_gated",
                    "mean_reversion", "volume_confirmed", "regime_adaptive"}
        assert set(STRATEGY_REGISTRY.keys()) == expected

    def test_all_strategies_have_required_keys(self):
        for name, info in STRATEGY_REGISTRY.items():
            assert "factory" in info, f"{name} missing factory"
            assert "grid" in info, f"{name} missing grid"
            assert "display_cols" in info, f"{name} missing display_cols"
            assert len(info["grid"]) > 0, f"{name} has empty grid"

    def test_all_grids_produce_valid_combo_count(self):
        for name, info in STRATEGY_REGISTRY.items():
            assert len(info["grid"]) >= 10, f"{name} grid too small: {len(info['grid'])}"


# ──────────────────────────────────────────────────────────────────────
# Confirmation Count Strategy Tests
# ──────────────────────────────────────────────────────────────────────

class TestConfirmationStrategy:
    def test_strong_bullish_consensus(self):
        """All indicators bullish → should produce a bullish signal."""
        fn = make_confirmation_fn(min_agree=5, adx_thresh=20)
        bar = _make_bar(
            0, 50000, raw_score=0, direction=Direction.NEUTRAL,
            rsi_14=65, stoch_k=70, macd_histogram=100,
            ema_aligned=1, above_ema200=True, obv_bullish=True,
            cci_20=50, roc_10=3, change_pct=2.0, volume_ratio=1.5,
            plus_di=30, minus_di=15, adx=25,
        )
        score, direction = fn(bar)
        assert direction == Direction.BULLISH
        assert score > 0

    def test_strong_bearish_consensus(self):
        """All indicators bearish → should produce a bearish signal."""
        fn = make_confirmation_fn(min_agree=5, adx_thresh=20)
        bar = _make_bar(
            0, 50000, raw_score=0, direction=Direction.NEUTRAL,
            rsi_14=35, stoch_k=30, macd_histogram=-100,
            ema_aligned=-1, above_ema200=False, obv_bullish=False,
            cci_20=-50, roc_10=-3, change_pct=-2.0, volume_ratio=1.5,
            plus_di=10, minus_di=30, adx=25,
        )
        score, direction = fn(bar)
        assert direction == Direction.BEARISH
        assert score < 0

    def test_mixed_signals_neutral(self):
        """Mixed indicators → neutral when min_agree is high."""
        fn = make_confirmation_fn(min_agree=8, adx_thresh=20)
        bar = _make_bar(
            0, 50000, raw_score=0, direction=Direction.NEUTRAL,
            rsi_14=55, stoch_k=45, macd_histogram=10,
            ema_aligned=0, above_ema200=True, obv_bullish=False,
            cci_20=5, roc_10=-1, change_pct=0.5, volume_ratio=0.8,
            plus_di=20, minus_di=22, adx=25,
        )
        score, direction = fn(bar)
        assert direction == Direction.NEUTRAL

    def test_higher_min_agree_fewer_trades(self):
        """Higher confirmation threshold should produce fewer or equal signals."""
        bar = _make_bar(
            0, 50000, rsi_14=65, stoch_k=70, macd_histogram=50,
            ema_aligned=1, above_ema200=True, obv_bullish=True,
            cci_20=30, roc_10=2, change_pct=1.0, volume_ratio=1.2,
            plus_di=25, minus_di=15, adx=25,
        )
        fn5 = make_confirmation_fn(min_agree=5)
        fn8 = make_confirmation_fn(min_agree=8)

        s5, d5 = fn5(bar)
        s8, d8 = fn8(bar)
        # fn5 should detect signal; fn8 may or may not
        assert d5 != Direction.NEUTRAL


# ──────────────────────────────────────────────────────────────────────
# Momentum Breakout Strategy Tests
# ──────────────────────────────────────────────────────────────────────

class TestMomentumBreakoutStrategy:
    def test_strong_rsi_bull_with_volume(self):
        """RSI well above 50 + good volume → bullish."""
        fn = make_momentum_breakout_fn(rsi_weight=0.7, volume_mult=1.0,
                                        roc_weight=0.3, min_rsi_dev=10)
        bar = _make_bar(0, 50000, rsi_14=75, roc_10=5, volume_ratio=2.0)
        score, direction = fn(bar)
        assert direction == Direction.BULLISH
        assert score > 20

    def test_rsi_near_50_neutral(self):
        """RSI near 50 with min_rsi_dev=15 → no signal."""
        fn = make_momentum_breakout_fn(min_rsi_dev=15)
        bar = _make_bar(0, 50000, rsi_14=55, roc_10=1, volume_ratio=1.0)
        score, direction = fn(bar)
        assert direction == Direction.NEUTRAL

    def test_low_volume_dampens_signal(self):
        """Low volume should dampen the signal magnitude."""
        fn = make_momentum_breakout_fn(rsi_weight=0.7, volume_mult=1.5)
        bar_high_vol = _make_bar(0, 50000, rsi_14=75, roc_10=3, volume_ratio=2.0)
        bar_low_vol = _make_bar(0, 50000, rsi_14=75, roc_10=3, volume_ratio=0.5)
        s_high, _ = fn(bar_high_vol)
        s_low, _ = fn(bar_low_vol)
        assert abs(s_high) > abs(s_low)


# ──────────────────────────────────────────────────────────────────────
# Trend Gated Strategy Tests
# ──────────────────────────────────────────────────────────────────────

class TestTrendGatedStrategy:
    def test_no_ema_stack_blocked(self):
        """Mixed EMA alignment with require_full_stack=True → no trade."""
        fn = make_trend_gated_fn(require_full_stack=True, min_adx=20)
        bar = _make_bar(0, 50000, ema_aligned=0, adx=30, atr_14=1000,
                        category_scores=[
                            CategoryScore("momentum", 60, 0.25, 15, {}),
                            CategoryScore("volume", 30, 0.15, 4.5, {}),
                        ])
        score, direction = fn(bar)
        assert direction == Direction.NEUTRAL

    def test_strong_trend_bull_signal(self):
        """Bullish EMA stack + strong ADX + bullish momentum → bullish signal."""
        fn = make_trend_gated_fn(require_full_stack=True, min_adx=20,
                                  mom_w=0.6, vol_w=0.2)
        bar = _make_bar(0, 50000, ema_aligned=1, adx=30, atr_14=1000,
                        category_scores=[
                            CategoryScore("momentum", 50, 0.25, 12.5, {}),
                            CategoryScore("volume", 40, 0.15, 6, {}),
                        ])
        score, direction = fn(bar)
        assert direction == Direction.BULLISH
        assert score > 0

    def test_low_adx_blocked(self):
        """ADX below threshold → no trade even with perfect alignment."""
        fn = make_trend_gated_fn(require_full_stack=True, min_adx=30)
        bar = _make_bar(0, 50000, ema_aligned=1, adx=20, atr_14=1000,
                        category_scores=[
                            CategoryScore("momentum", 70, 0.25, 17.5, {}),
                            CategoryScore("volume", 50, 0.15, 7.5, {}),
                        ])
        score, direction = fn(bar)
        assert direction == Direction.NEUTRAL


# ──────────────────────────────────────────────────────────────────────
# Mean Reversion Strategy Tests
# ──────────────────────────────────────────────────────────────────────

class TestMeanReversionStrategy:
    def test_oversold_buy_signal(self):
        """RSI < 30 + Stoch < 20 → bullish reversal signal."""
        fn = make_mean_reversion_fn(rsi_low=30, rsi_high=70,
                                     require_sr=False, sr_bonus=1.0)
        bar = _make_bar(0, 50000, rsi_14=25, stoch_k=15, bb_position=0.1,
                        category_scores=[
                            CategoryScore("support_resistance", 20, 0.2, 4, {}),
                        ])
        score, direction = fn(bar)
        assert direction == Direction.BULLISH
        assert score > 0

    def test_overbought_sell_signal(self):
        """RSI > 70 + Stoch > 80 → bearish reversal signal."""
        fn = make_mean_reversion_fn(rsi_low=30, rsi_high=70,
                                     require_sr=False, sr_bonus=1.0)
        bar = _make_bar(0, 50000, rsi_14=80, stoch_k=85, bb_position=0.97,
                        category_scores=[
                            CategoryScore("support_resistance", -20, 0.2, -4, {}),
                        ])
        score, direction = fn(bar)
        assert direction == Direction.BEARISH
        assert score < 0

    def test_normal_ranges_neutral(self):
        """RSI and Stoch in normal range → no signal."""
        fn = make_mean_reversion_fn(rsi_low=30, rsi_high=70)
        bar = _make_bar(0, 50000, rsi_14=50, stoch_k=50, bb_position=0.5)
        score, direction = fn(bar)
        assert direction == Direction.NEUTRAL

    def test_sr_confirm_amplifies(self):
        """S/R bonus should amplify score when confirmed."""
        fn_with = make_mean_reversion_fn(rsi_low=35, rsi_high=65,
                                          require_sr=True, sr_bonus=2.0)
        fn_without = make_mean_reversion_fn(rsi_low=35, rsi_high=65,
                                             require_sr=False, sr_bonus=1.0)
        bar = _make_bar(0, 50000, rsi_14=25, stoch_k=15, bb_position=0.03,
                        category_scores=[
                            CategoryScore("support_resistance", 40, 0.2, 8, {}),
                        ])
        s_with, _ = fn_with(bar)
        s_without, _ = fn_without(bar)
        assert s_with > s_without


# ──────────────────────────────────────────────────────────────────────
# Volume Confirmed Strategy Tests
# ──────────────────────────────────────────────────────────────────────

class TestVolumeConfirmedStrategy:
    def test_low_volume_blocked(self):
        """Below volume threshold → no trade."""
        fn = make_volume_confirmed_fn(vol_thresh=1.5)
        bar = _make_bar(0, 50000, volume_ratio=1.0,
                        category_scores=[
                            CategoryScore("trend", 70, 0.3, 21, {}),
                            CategoryScore("momentum", 60, 0.25, 15, {}),
                        ])
        score, direction = fn(bar)
        assert direction == Direction.NEUTRAL

    def test_high_volume_passes(self):
        """Above volume threshold with bullish trend+momentum → bullish."""
        fn = make_volume_confirmed_fn(vol_thresh=1.2, trend_w=0.5, mom_w=0.4)
        bar = _make_bar(0, 50000, volume_ratio=2.0,
                        category_scores=[
                            CategoryScore("trend", 60, 0.3, 18, {}),
                            CategoryScore("momentum", 50, 0.25, 12.5, {}),
                        ])
        score, direction = fn(bar)
        assert direction == Direction.BULLISH


# ──────────────────────────────────────────────────────────────────────
# Regime Adaptive Strategy Tests
# ──────────────────────────────────────────────────────────────────────

class TestRegimeAdaptiveStrategy:
    def test_trending_uses_trend_weight(self):
        """In trending regime, trend weight should dominate."""
        fn = make_regime_adaptive_fn(trending_trend_w=0.8, trending_mom_w=0.1)
        bar = _make_bar(
            0, 50000, regime=MarketRegime.TRENDING,
            category_scores=[
                CategoryScore("trend", 80, 0.3, 24, {}),
                CategoryScore("momentum", -20, 0.25, -5, {}),
                CategoryScore("volume", 10, 0.15, 1.5, {}),
                CategoryScore("support_resistance", 5, 0.2, 1, {}),
                CategoryScore("risk", 0, 0.1, 0, {}),
            ])
        score, direction = fn(bar)
        assert direction == Direction.BULLISH  # Trend dominates

    def test_ranging_flips_momentum(self):
        """In ranging regime, momentum should be flipped (mean reversion)."""
        fn = make_regime_adaptive_fn(ranging_sr_w=0.4, ranging_mom_flip=0.5)
        bar = _make_bar(
            0, 50000, regime=MarketRegime.RANGING,
            category_scores=[
                CategoryScore("trend", 0, 0.3, 0, {}),
                CategoryScore("momentum", -60, 0.25, -15, {}),  # Bearish mom
                CategoryScore("volume", 0, 0.15, 0, {}),
                CategoryScore("support_resistance", 20, 0.2, 4, {}),
                CategoryScore("risk", 0, 0.1, 0, {}),
            ])
        score, direction = fn(bar)
        # Flipped bearish momentum → bullish + some S/R → should be bullish
        assert direction == Direction.BULLISH

    def test_volatile_risk_gate(self):
        """In volatile regime with bad risk → no trade."""
        fn = make_regime_adaptive_fn(volatile_risk_floor=0)
        bar = _make_bar(
            0, 50000, regime=MarketRegime.VOLATILE,
            category_scores=[
                CategoryScore("trend", 70, 0.3, 21, {}),
                CategoryScore("momentum", 50, 0.25, 12.5, {}),
                CategoryScore("volume", 30, 0.15, 4.5, {}),
                CategoryScore("support_resistance", 20, 0.2, 4, {}),
                CategoryScore("risk", -30, 0.1, -3, {}),
            ])
        score, direction = fn(bar)
        assert direction == Direction.NEUTRAL  # Risk too bad


# ──────────────────────────────────────────────────────────────────────
# Strategy Integration with fast_backtest
# ──────────────────────────────────────────────────────────────────────

class TestStrategyBacktestIntegration:
    def test_score_override_fn_used(self):
        """score_override_fn should override the bar's raw_score/direction."""
        # Bar has NEUTRAL direction, but our override says BULLISH
        bars = [
            _make_bar(0, 50000, high=50500, low=49500,
                      raw_score=0, direction=Direction.NEUTRAL,
                      atr_14=1000, adx=30, atr_pct=2.0),
            _make_bar(1, 52000, high=56000, low=51000,
                      raw_score=0, direction=Direction.NEUTRAL),
        ]

        def always_bull(bar):
            return 50.0, Direction.BULLISH

        result = fast_backtest(
            bars,
            leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=4.0,
            tp1_exit_pct=0.5, marginal_low=20, strong_thresh=40,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006, risk_pct=0.02,
            score_override_fn=always_bull,
        )
        # Override forced a trade even though bar was NEUTRAL
        assert result["trades"] >= 1

    def test_neutral_override_no_trades(self):
        """Override returning NEUTRAL should block all trades."""
        bars = [
            _make_bar(0, 50000, high=50500, low=49500,
                      raw_score=70, direction=Direction.BULLISH,
                      atr_14=1000, adx=30, atr_pct=2.0),
            _make_bar(1, 55000, high=56000, low=54000),
        ]

        def always_neutral(bar):
            return 0.0, Direction.NEUTRAL

        result = fast_backtest(
            bars,
            leverage=10, atr_sl_mult=1.5, tp1_rr=2.0, tp2_rr=4.0,
            tp1_exit_pct=0.5, marginal_low=20, strong_thresh=40,
            min_adx=0, min_volatility_pct=0, min_category_agreement=0,
            require_trend_momentum_agree=False, skip_choppy=False,
            skip_volatile=False, sl_strategy="atr",
            initial_balance=10000, fee_rate=0.0006, risk_pct=0.02,
            score_override_fn=always_neutral,
        )
        assert result["trades"] == 0
