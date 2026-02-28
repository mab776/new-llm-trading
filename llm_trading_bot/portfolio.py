"""
Portfolio simulation — realistic fee-aware portfolio tracking.

Handles:
- Position sizing and leverage
- Fee accounting on leveraged notional (maker/taker)
- Partial exits (TP1 partial, TP2 remainder)
- Trailing stops
- Drawdown tracking and balance history
- Win/loss statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Trade:
    """A completed or open trade."""
    trade_id: int
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    entry_time: str
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    size: float = 0.0  # in base currency (e.g., BTC)
    leverage: int = 1
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    tp1_exit_pct: float = 0.5  # fraction of position to close at TP1

    # Execution details
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    total_fees: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    exit_reason: str = ""  # "tp1", "tp2", "sl", "trailing_stop", "manual"

    # Partial exit tracking
    partial_exits: list[dict] = field(default_factory=list)
    remaining_size: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_price is None and self.remaining_size > 0

    @property
    def is_profitable(self) -> bool:
        return self.net_pnl > 0


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""
    timestamp: str
    balance: float
    equity: float  # balance + unrealized PnL
    unrealized_pnl: float
    drawdown_pct: float
    peak_balance: float


@dataclass
class PortfolioStats:
    """Summary statistics for the portfolio."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_gross_pnl: float = 0.0
    total_fees: float = 0.0
    total_net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    avg_rr_ratio: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_trade_duration: str = ""
    final_balance: float = 0.0
    total_return_pct: float = 0.0


class Portfolio:
    """
    Simulated portfolio with realistic fee accounting.

    All fees are calculated on the LEVERAGED notional value,
    not on the margin amount. This is critical for realistic backtesting.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0006,
        default_order_type: str = "taker",
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.default_order_type = default_order_type

        self.trades: list[Trade] = []
        self.open_trades: list[Trade] = []
        self.history: list[PortfolioSnapshot] = []
        self.peak_balance = initial_balance
        self.max_drawdown_pct = 0.0
        self._trade_counter = 0

    @property
    def fee_rate(self) -> float:
        return self.maker_fee if self.default_order_type == "maker" else self.taker_fee

    def _calculate_fee(self, size: float, price: float, leverage: int) -> float:
        """
        Calculate fee on leveraged notional.
        Fee = size * price * fee_rate (the notional IS the leveraged amount).
        """
        notional = size * price
        return notional * self.fee_rate

    def _calculate_position_size(
        self, price: float, leverage: int, risk_pct: float = 0.02
    ) -> float:
        """
        Calculate position size based on risk percentage of balance.
        Returns size in base currency.
        """
        risk_amount = self.balance * risk_pct
        margin = risk_amount  # The margin allocated to this trade
        notional = margin * leverage
        size = notional / price
        return size

    def open_trade(
        self,
        direction: str,
        entry_price: float,
        entry_time: str,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        leverage: int = 5,
        risk_pct: float = 0.02,
        tp1_exit_pct: float = 0.5,
    ) -> Trade:
        """Open a new trade."""
        self._trade_counter += 1

        size = self._calculate_position_size(entry_price, leverage, risk_pct)
        entry_fee = self._calculate_fee(size, entry_price, leverage)

        trade = Trade(
            trade_id=self._trade_counter,
            direction=direction,
            entry_price=entry_price,
            entry_time=entry_time,
            size=size,
            remaining_size=size,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            tp1_exit_pct=tp1_exit_pct,
            entry_fee=entry_fee,
        )

        self.balance -= entry_fee
        self.open_trades.append(trade)
        self.trades.append(trade)

        return trade

    def _compute_pnl(self, trade: Trade, exit_price: float, exit_size: float) -> tuple[float, float, float]:
        """
        Compute PnL for a (partial) exit.
        Returns (gross_pnl, exit_fee, net_pnl).

        NOTE: position size already includes leverage (size = margin * leverage / price),
        so we do NOT multiply by leverage again here.
        """
        if trade.direction == "LONG":
            price_diff = exit_price - trade.entry_price
        else:
            price_diff = trade.entry_price - exit_price

        gross_pnl = price_diff * exit_size
        exit_fee = self._calculate_fee(exit_size, exit_price, trade.leverage)
        net_pnl = gross_pnl - exit_fee

        return gross_pnl, exit_fee, net_pnl

    def partial_exit(
        self, trade: Trade, exit_price: float, exit_time: str, fraction: float, reason: str
    ) -> float:
        """
        Exit a fraction of a position. Returns net PnL of the partial exit.
        """
        exit_size = trade.remaining_size * fraction
        if exit_size <= 0:
            return 0.0

        gross_pnl, exit_fee, net_pnl = self._compute_pnl(trade, exit_price, exit_size)

        trade.partial_exits.append({
            "price": exit_price,
            "time": exit_time,
            "size": exit_size,
            "fraction": fraction,
            "gross_pnl": gross_pnl,
            "exit_fee": exit_fee,
            "net_pnl": net_pnl,
            "reason": reason,
        })

        trade.remaining_size -= exit_size
        trade.exit_fee += exit_fee
        trade.total_fees = trade.entry_fee + trade.exit_fee
        trade.gross_pnl += gross_pnl
        trade.net_pnl += net_pnl

        self.balance += net_pnl

        return net_pnl

    def close_trade(
        self, trade: Trade, exit_price: float, exit_time: str, reason: str
    ) -> float:
        """
        Fully close remaining position. Returns net PnL of the closing.
        """
        if trade.remaining_size <= 0:
            return 0.0

        remaining = trade.remaining_size
        gross_pnl, exit_fee, net_pnl = self._compute_pnl(trade, exit_price, remaining)

        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.exit_reason = reason
        trade.exit_fee += exit_fee
        trade.total_fees = trade.entry_fee + trade.exit_fee
        trade.gross_pnl += gross_pnl
        trade.net_pnl += net_pnl
        trade.remaining_size = 0

        self.balance += net_pnl
        self.open_trades = [t for t in self.open_trades if t.is_open]

        # Track drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        dd_pct = (self.peak_balance - self.balance) / self.peak_balance * 100
        if dd_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = dd_pct

        return net_pnl

    def record_snapshot(self, timestamp: str, current_price: float = 0) -> PortfolioSnapshot:
        """Record a point-in-time snapshot."""
        unrealized = 0.0
        for trade in self.open_trades:
            # Size already includes leverage, so no extra leverage multiplier
            if trade.direction == "LONG":
                unrealized += (current_price - trade.entry_price) * trade.remaining_size
            else:
                unrealized += (trade.entry_price - current_price) * trade.remaining_size

        equity = self.balance + unrealized
        if equity > self.peak_balance:
            self.peak_balance = equity

        dd_pct = (self.peak_balance - equity) / self.peak_balance * 100 if self.peak_balance > 0 else 0

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            balance=round(self.balance, 2),
            equity=round(equity, 2),
            unrealized_pnl=round(unrealized, 2),
            drawdown_pct=round(dd_pct, 2),
            peak_balance=round(self.peak_balance, 2),
        )
        self.history.append(snapshot)
        return snapshot

    def compute_stats(self) -> PortfolioStats:
        """Compute comprehensive portfolio statistics."""
        closed = [t for t in self.trades if not t.is_open]

        if not closed:
            return PortfolioStats(final_balance=self.balance, total_return_pct=0)

        winners = [t for t in closed if t.is_profitable]
        losers = [t for t in closed if not t.is_profitable]

        total_gross = sum(t.gross_pnl for t in closed)
        total_fees = sum(t.total_fees for t in closed)
        total_net = sum(t.net_pnl for t in closed)

        win_amounts = [t.net_pnl for t in winners] if winners else [0]
        loss_amounts = [abs(t.net_pnl) for t in losers] if losers else [0]

        gross_wins = sum(win_amounts)
        gross_losses = sum(loss_amounts)
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Consecutive wins/losses
        max_consec_w = max_consec_l = consec_w = consec_l = 0
        for t in closed:
            if t.is_profitable:
                consec_w += 1
                consec_l = 0
                max_consec_w = max(max_consec_w, consec_w)
            else:
                consec_l += 1
                consec_w = 0
                max_consec_l = max(max_consec_l, consec_l)

        # Sharpe ratio approximation (from trade returns)
        if len(closed) > 1:
            returns = [t.net_pnl / self.initial_balance for t in closed]
            import numpy as np
            avg_ret = np.mean(returns)
            std_ret = np.std(returns)
            sharpe = (avg_ret / std_ret * (252 ** 0.5)) if std_ret > 0 else 0
        else:
            sharpe = 0

        return PortfolioStats(
            total_trades=len(closed),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners) / len(closed) * 100 if closed else 0,
            total_gross_pnl=round(total_gross, 2),
            total_fees=round(total_fees, 2),
            total_net_pnl=round(total_net, 2),
            max_drawdown_pct=round(self.max_drawdown_pct, 2),
            max_consecutive_wins=max_consec_w,
            max_consecutive_losses=max_consec_l,
            avg_win=round(sum(win_amounts) / len(winners), 2) if winners else 0,
            avg_loss=round(sum(loss_amounts) / len(losers), 2) if losers else 0,
            profit_factor=round(profit_factor, 2),
            sharpe_ratio=round(sharpe, 2),
            best_trade=round(max(t.net_pnl for t in closed), 2),
            worst_trade=round(min(t.net_pnl for t in closed), 2),
            final_balance=round(self.balance, 2),
            total_return_pct=round((self.balance - self.initial_balance) / self.initial_balance * 100, 2),
        )
