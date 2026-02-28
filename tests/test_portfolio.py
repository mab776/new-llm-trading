"""
Tests for portfolio simulation.
Covers: fee accounting, partial exits, PnL calculation, drawdown tracking.
"""

import pytest

from llm_trading_bot.portfolio import Portfolio, Trade


class TestFeeAccounting:
    def test_fees_on_leveraged_notional(self):
        """Fees must be calculated on leveraged notional, not margin."""
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="2024-01-01",
            stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
            leverage=10, risk_pct=0.02,
        )
        # Position size: margin=200, notional=2000, size=2000/50000=0.04 BTC
        # Entry fee: 0.04 * 50000 * 0.0006 = 1.2
        assert trade.entry_fee > 0
        expected_notional = trade.size * 50000
        expected_fee = expected_notional * 0.0006
        assert abs(trade.entry_fee - expected_fee) < 0.01

    def test_balance_decreases_by_entry_fee(self):
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="t",
            stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
            leverage=5,
        )
        assert port.balance < 10000
        assert abs(port.balance - (10000 - trade.entry_fee)) < 0.01


class TestTradeExecution:
    def test_profitable_long_trade(self):
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="t1",
            stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
            leverage=5, risk_pct=0.02,
        )
        port.close_trade(trade, exit_price=52000, exit_time="t2", reason="tp1")

        assert trade.net_pnl > 0
        assert trade.exit_price == 52000
        assert trade.exit_reason == "tp1"
        assert not trade.is_open
        assert trade.is_profitable

    def test_losing_long_trade(self):
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="t1",
            stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
            leverage=5, risk_pct=0.02,
        )
        port.close_trade(trade, exit_price=49000, exit_time="t2", reason="sl")

        assert trade.net_pnl < 0
        assert not trade.is_profitable

    def test_short_trade(self):
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        trade = port.open_trade(
            direction="SHORT", entry_price=50000, entry_time="t1",
            stop_loss=51000, take_profit_1=48000, take_profit_2=46000,
            leverage=5, risk_pct=0.02,
        )
        port.close_trade(trade, exit_price=48000, exit_time="t2", reason="tp1")

        assert trade.net_pnl > 0

    def test_fees_eat_into_profit(self):
        """Net PnL should always be less than gross PnL due to fees."""
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="t1",
            stop_loss=49000, take_profit_1=55000, take_profit_2=60000,
            leverage=10, risk_pct=0.02,
        )
        port.close_trade(trade, exit_price=55000, exit_time="t2", reason="tp1")

        assert trade.net_pnl < trade.gross_pnl
        assert trade.total_fees > 0


class TestPartialExits:
    def test_tp1_partial_exit(self):
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="t1",
            stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
            leverage=5, risk_pct=0.02, tp1_exit_pct=0.5,
        )
        initial_size = trade.remaining_size

        # Partial exit at TP1
        pnl = port.partial_exit(trade, 52000, "t2", 0.5, "tp1")
        assert pnl > 0
        assert trade.remaining_size == pytest.approx(initial_size * 0.5, rel=1e-6)
        assert len(trade.partial_exits) == 1
        assert trade.is_open  # Still has remaining size

    def test_full_close_after_partial(self):
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="t1",
            stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
            leverage=5, risk_pct=0.02,
        )

        # TP1 partial
        port.partial_exit(trade, 52000, "t2", 0.5, "tp1")
        # TP2 full close
        port.close_trade(trade, 54000, "t3", "tp2")

        assert not trade.is_open
        assert trade.remaining_size == 0
        assert trade.net_pnl > 0
        assert len(trade.partial_exits) == 1


class TestDrawdownTracking:
    def test_drawdown_increases_on_loss(self):
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)

        # Losing trade
        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="t1",
            stop_loss=48000, take_profit_1=54000, take_profit_2=58000,
            leverage=5, risk_pct=0.05,
        )
        port.close_trade(trade, 48000, "t2", "sl")

        assert port.max_drawdown_pct > 0

    def test_peak_updates_on_profit(self):
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)

        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="t1",
            stop_loss=49000, take_profit_1=55000, take_profit_2=60000,
            leverage=5, risk_pct=0.02,
        )
        port.close_trade(trade, 55000, "t2", "tp1")

        assert port.peak_balance > 10000


class TestPnLMagnitude:
    """
    CRITICAL regression tests: verify that PnL amounts match manual calculation.
    Catches the double-leverage bug where PnL was inflated by leverage factor.
    """

    def test_long_pnl_exact_magnitude(self):
        """
        Manual calculation:
          balance=10000, risk_pct=0.02, leverage=10, entry=50000
          margin = 10000 * 0.02 = 200
          notional = 200 * 10 = 2000
          size = 2000 / 50000 = 0.04 BTC

          Price goes 50000 → 52000, diff = +2000
          gross_pnl = 2000 * 0.04 = 80  (NO extra leverage — size IS leveraged)
          entry_fee = 0.04 * 50000 * 0.0006 = 1.2
          exit_fee  = 0.04 * 52000 * 0.0006 = 1.248
          net_pnl = 80 - 1.248 = 78.752
        """
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="t1",
            stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
            leverage=10, risk_pct=0.02,
        )
        assert trade.size == pytest.approx(0.04, rel=1e-6)
        assert trade.entry_fee == pytest.approx(1.2, rel=1e-3)

        port.close_trade(trade, exit_price=52000, exit_time="t2", reason="tp1")

        # PnL should be ~$80 gross, NOT $800 (double-leverage bug)
        assert trade.gross_pnl == pytest.approx(80.0, rel=1e-4)
        exit_fee = 0.04 * 52000 * 0.0006
        assert trade.exit_fee == pytest.approx(exit_fee, rel=1e-3)
        assert trade.net_pnl == pytest.approx(80.0 - exit_fee, rel=1e-3)

    def test_long_pnl_at_stop_loss(self):
        """
        SL hit: price 50000 → 49000, diff = -1000
        size = 0.04, gross_pnl = -1000 * 0.04 = -40
        """
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="t1",
            stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
            leverage=10, risk_pct=0.02,
        )
        port.close_trade(trade, exit_price=49000, exit_time="t2", reason="sl")

        assert trade.gross_pnl == pytest.approx(-40.0, rel=1e-4)

    def test_short_pnl_exact_magnitude(self):
        """
        SHORT at 50000, close at 48000, diff = +2000 (favorable)
        size = 0.04, gross_pnl = 2000 * 0.04 = 80
        """
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        trade = port.open_trade(
            direction="SHORT", entry_price=50000, entry_time="t1",
            stop_loss=51000, take_profit_1=48000, take_profit_2=46000,
            leverage=10, risk_pct=0.02,
        )
        port.close_trade(trade, exit_price=48000, exit_time="t2", reason="tp1")

        assert trade.gross_pnl == pytest.approx(80.0, rel=1e-4)

    def test_partial_exit_pnl_magnitude(self):
        """
        Partial exit at TP1 (50% of position):
          size=0.04, exit_size=0.02, price_diff=2000
          partial_gross = 2000 * 0.02 = 40
        Remainder at TP2:
          remaining=0.02, price_diff=4000
          rest_gross = 4000 * 0.02 = 80
        Total gross = 120
        """
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="t1",
            stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
            leverage=10, risk_pct=0.02, tp1_exit_pct=0.5,
        )
        port.partial_exit(trade, 52000, "t2", 0.5, "tp1")
        port.close_trade(trade, 54000, "t3", "tp2")

        assert trade.gross_pnl == pytest.approx(120.0, rel=1e-3)

    def test_pnl_scales_with_leverage(self):
        """
        Higher leverage -> bigger size -> bigger PnL (linearly).
        5x: size=0.02, gross_pnl=2000*0.02=40
        10x: size=0.04, gross_pnl=2000*0.04=80
        The ratio should be exactly 2.0.
        """
        results = {}
        for lev in (5, 10, 20):
            port = Portfolio(initial_balance=10000, taker_fee=0.0)
            trade = port.open_trade(
                direction="LONG", entry_price=50000, entry_time="t1",
                stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
                leverage=lev, risk_pct=0.02,
            )
            port.close_trade(trade, exit_price=52000, exit_time="t2", reason="tp1")
            results[lev] = trade.gross_pnl

        # PnL should scale linearly with leverage
        assert results[10] / results[5] == pytest.approx(2.0, rel=1e-6)
        assert results[20] / results[10] == pytest.approx(2.0, rel=1e-6)

    def test_balance_after_round_trip_trade(self):
        """
        After a winning trade, balance should increase by net PnL.
        balance_after = initial - entry_fee + net_pnl_from_close
        """
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="t1",
            stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
            leverage=10, risk_pct=0.02,
        )
        entry_fee = trade.entry_fee
        port.close_trade(trade, exit_price=52000, exit_time="t2", reason="tp1")

        expected_balance = 10000 - entry_fee + trade.gross_pnl - trade.exit_fee
        assert port.balance == pytest.approx(expected_balance, rel=1e-6)


class TestSnapshotUnrealized:
    def test_unrealized_pnl_no_extra_leverage(self):
        """
        Unrealized PnL should NOT multiply by leverage (size already leveraged).
        """
        port = Portfolio(initial_balance=10000, taker_fee=0.0)
        trade = port.open_trade(
            direction="LONG", entry_price=50000, entry_time="t1",
            stop_loss=49000, take_profit_1=52000, take_profit_2=54000,
            leverage=10, risk_pct=0.02,
        )
        # size = 0.04 BTC, current price = 51000, diff = +1000
        snap = port.record_snapshot("t1", current_price=51000)
        expected_unrealized = 1000 * 0.04  # = 40, NOT 400
        assert snap.unrealized_pnl == pytest.approx(expected_unrealized, rel=1e-4)

    def test_short_unrealized_pnl(self):
        port = Portfolio(initial_balance=10000, taker_fee=0.0)
        trade = port.open_trade(
            direction="SHORT", entry_price=50000, entry_time="t1",
            stop_loss=51000, take_profit_1=48000, take_profit_2=46000,
            leverage=10, risk_pct=0.02,
        )
        # Price went DOWN to 49000, unrealized = (50000-49000)*0.04 = 40
        snap = port.record_snapshot("t1", current_price=49000)
        assert snap.unrealized_pnl == pytest.approx(40.0, rel=1e-4)


class TestPortfolioStats:
    def test_stats_with_no_trades(self):
        port = Portfolio(initial_balance=10000)
        stats = port.compute_stats()
        assert stats.total_trades == 0
        assert stats.final_balance == 10000

    def test_stats_calculation(self):
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)

        # Winning trade
        t1 = port.open_trade("LONG", 50000, "t1", 49000, 52000, 54000, 5, 0.02)
        port.close_trade(t1, 52000, "t2", "tp1")

        # Losing trade
        t2 = port.open_trade("LONG", 51000, "t3", 50000, 53000, 55000, 5, 0.02)
        port.close_trade(t2, 50000, "t4", "sl")

        stats = port.compute_stats()
        assert stats.total_trades == 2
        assert stats.winning_trades == 1
        assert stats.losing_trades == 1
        assert stats.win_rate == 50.0
        assert stats.total_fees > 0

    def test_return_pct_consistent(self):
        """Return % should match (final - initial) / initial * 100."""
        port = Portfolio(initial_balance=10000, taker_fee=0.0006)
        t1 = port.open_trade("LONG", 50000, "t1", 49000, 52000, 54000, 10, 0.02)
        port.close_trade(t1, 52000, "t2", "tp1")
        stats = port.compute_stats()
        expected_pct = (stats.final_balance - 10000) / 10000 * 100
        # Both are rounded to 2 decimals, so abs tolerance is appropriate
        assert stats.total_return_pct == pytest.approx(expected_pct, abs=0.02)
