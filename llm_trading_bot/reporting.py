"""
Reporting module — generates charts, text reports, and decision logs.

Handles:
- Backtest equity curves and trade visualization
- Portfolio stats formatting
- Decision log export
- LLM-friendly text reports
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from llm_trading_bot.backtesting import BacktestResult
from llm_trading_bot.portfolio import Portfolio, PortfolioStats


def generate_backtest_charts(
    result: BacktestResult,
    output_dir: str = "reports",
) -> list[str]:
    """
    Generate backtest visualization charts.
    Returns list of saved file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files: list[str] = []

    portfolio = result.portfolio
    if not portfolio:
        return saved_files

    # ── 1. Equity Curve ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1, 1]})

    # Balance history
    if portfolio.history:
        times = [s.timestamp for s in portfolio.history]
        balances = [s.balance for s in portfolio.history]
        equities = [s.equity for s in portfolio.history]
        drawdowns = [s.drawdown_pct for s in portfolio.history]

        ax1 = axes[0]
        ax1.plot(range(len(balances)), balances, label="Balance", color="blue", linewidth=1.5)
        ax1.plot(range(len(equities)), equities, label="Equity", color="green", alpha=0.7, linewidth=1)
        ax1.axhline(y=portfolio.initial_balance, color="gray", linestyle="--", alpha=0.5, label="Initial")
        ax1.set_title("Equity Curve", fontsize=14)
        ax1.set_ylabel("USD")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = axes[1]
        ax2.fill_between(range(len(drawdowns)), drawdowns, color="red", alpha=0.3)
        ax2.plot(range(len(drawdowns)), drawdowns, color="red", linewidth=0.8)
        ax2.set_title("Drawdown (%)", fontsize=12)
        ax2.set_ylabel("%")
        ax2.grid(True, alpha=0.3)

    # Trade PnL distribution
    closed_trades = [t for t in portfolio.trades if not t.is_open]
    if closed_trades:
        pnls = [t.net_pnl for t in closed_trades]
        ax3 = axes[2]
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax3.set_title("Trade PnL", fontsize=12)
        ax3.set_ylabel("USD")
        ax3.axhline(y=0, color="black", linewidth=0.5)
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    equity_path = str(output_path / f"backtest_equity_{timestamp}.png")
    plt.savefig(equity_path, dpi=150)
    plt.close()
    saved_files.append(equity_path)

    # ── 2. Price Chart with Trades ──
    if result.bars:
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        prices = [b.close for b in result.bars]
        ax.plot(range(len(prices)), prices, color="black", linewidth=0.8, alpha=0.8)

        # Mark trades
        for entry in result.decision_log:
            action = entry.get("action", "")
            idx_time = entry.get("time", "")
            price = entry.get("price", 0)

            # Find bar index
            bar_idx = None
            for bi, b in enumerate(result.bars):
                if b.timestamp == idx_time:
                    bar_idx = bi
                    break

            if bar_idx is not None:
                if "OPEN_LONG" in action:
                    ax.scatter(bar_idx, price, marker="^", color="green", s=100, zorder=5)
                elif "OPEN_SHORT" in action:
                    ax.scatter(bar_idx, price, marker="v", color="red", s=100, zorder=5)
                elif action == "SL_HIT":
                    ax.scatter(bar_idx, price, marker="x", color="red", s=80, zorder=5)
                elif "TP" in action:
                    ax.scatter(bar_idx, price, marker="o", color="green", s=80, zorder=5)

        ax.set_title(f"Price & Trades — {result.config_summary.get('symbol', 'BTC')}", fontsize=14)
        ax.set_ylabel("Price (USD)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        trades_path = str(output_path / f"backtest_trades_{timestamp}.png")
        plt.savefig(trades_path, dpi=150)
        plt.close()
        saved_files.append(trades_path)

    return saved_files


def format_stats_report(stats: PortfolioStats, config_summary: dict = None) -> str:
    """Format portfolio stats into a readable text report."""
    lines = [
        "+=" * 23 + "+",
        "|          BACKTEST RESULTS REPORT             |",
        "+=" * 23 + "+",
    ]

    if config_summary:
        lines.append("\n--- Configuration ---")
        for k, v in config_summary.items():
            lines.append(f"  {k}: {v}")

    lines.extend([
        "\n--- Performance ---",
        f"  Final Balance:    ${stats.final_balance:>12,.2f}",
        f"  Total Return:     {stats.total_return_pct:>12.1f}%",
        f"  Net PnL:          ${stats.total_net_pnl:>12,.2f}",
        f"  Gross PnL:        ${stats.total_gross_pnl:>12,.2f}",
        f"  Total Fees:       ${stats.total_fees:>12,.2f}",
        "",
        "--- Trade Statistics ---",
        f"  Total Trades:     {stats.total_trades:>12d}",
        f"  Winners:          {stats.winning_trades:>12d}",
        f"  Losers:           {stats.losing_trades:>12d}",
        f"  Win Rate:         {stats.win_rate:>12.1f}%",
        f"  Avg Win:          ${stats.avg_win:>12,.2f}",
        f"  Avg Loss:         ${stats.avg_loss:>12,.2f}",
        f"  Best Trade:       ${stats.best_trade:>12,.2f}",
        f"  Worst Trade:      ${stats.worst_trade:>12,.2f}",
        "",
        "--- Risk Metrics ---",
        f"  Max Drawdown:     {stats.max_drawdown_pct:>12.1f}%",
        f"  Profit Factor:    {stats.profit_factor:>12.2f}",
        f"  Sharpe Ratio:     {stats.sharpe_ratio:>12.2f}",
        f"  Max Consec Wins:  {stats.max_consecutive_wins:>12d}",
        f"  Max Consec Losses:{stats.max_consecutive_losses:>12d}",
    ])

    return "\n".join(lines)


def export_decision_log(
    decision_log: list[dict],
    output_dir: str = "reports",
    filename: Optional[str] = None,
) -> str:
    """Export the decision log to JSON for audit."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"decision_log_{timestamp}.json"

    filepath = str(output_path / filename)
    with open(filepath, "w") as f:
        json.dump(decision_log, f, indent=2, default=str)

    return filepath


def export_trades_csv(
    portfolio: Portfolio,
    output_dir: str = "reports",
    filename: Optional[str] = None,
) -> str:
    """Export trades to CSV for analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trades_{timestamp}.csv"

    filepath = str(output_path / filename)

    rows = []
    for t in portfolio.trades:
        rows.append({
            "trade_id": t.trade_id,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "entry_time": t.entry_time,
            "exit_price": t.exit_price,
            "exit_time": t.exit_time,
            "size": t.size,
            "leverage": t.leverage,
            "stop_loss": t.stop_loss,
            "tp1": t.take_profit_1,
            "tp2": t.take_profit_2,
            "gross_pnl": t.gross_pnl,
            "total_fees": t.total_fees,
            "net_pnl": t.net_pnl,
            "exit_reason": t.exit_reason,
            "partial_exits": len(t.partial_exits),
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return filepath
