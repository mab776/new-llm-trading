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
    fast_backtest,
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
    category_scores=None,
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
        regime=regime,
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
