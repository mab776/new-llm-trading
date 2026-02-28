"""
Tests for the reporting module.
Covers: stats formatting, decision log export, trades CSV export.
"""

import json
import os
from pathlib import Path

import pytest

from llm_trading_bot.portfolio import Portfolio, PortfolioStats
from llm_trading_bot.reporting import (
    export_decision_log,
    export_trades_csv,
    format_stats_report,
)


@pytest.fixture
def sample_stats():
    return PortfolioStats(
        total_trades=50,
        winning_trades=30,
        losing_trades=20,
        win_rate=60.0,
        total_gross_pnl=1500.0,
        total_fees=120.0,
        total_net_pnl=1380.0,
        max_drawdown_pct=12.5,
        max_consecutive_wins=5,
        max_consecutive_losses=3,
        avg_win=75.0,
        avg_loss=46.0,
        profit_factor=2.45,
        sharpe_ratio=1.8,
        best_trade=200.0,
        worst_trade=-100.0,
        final_balance=11380.0,
        total_return_pct=13.8,
    )


@pytest.fixture
def sample_portfolio():
    port = Portfolio(initial_balance=10000, taker_fee=0.0006)
    t1 = port.open_trade("LONG", 50000, "2024-01-01", 49000, 52000, 54000, 5, 0.02)
    port.close_trade(t1, 52000, "2024-01-02", "tp1")
    t2 = port.open_trade("SHORT", 51000, "2024-01-03", 52000, 49000, 47000, 5, 0.02)
    port.close_trade(t2, 49000, "2024-01-04", "tp1")
    return port


class TestFormatStatsReport:
    def test_contains_key_sections(self, sample_stats):
        report = format_stats_report(sample_stats)
        assert "BACKTEST RESULTS REPORT" in report
        assert "Performance" in report
        assert "Trade Statistics" in report
        assert "Risk Metrics" in report

    def test_contains_values(self, sample_stats):
        report = format_stats_report(sample_stats)
        assert "11,380" in report or "11380" in report  # Final balance
        assert "60.0%" in report  # Win rate
        assert "12.5%" in report  # Max DD
        assert "2.45" in report  # Profit factor

    def test_with_config_summary(self, sample_stats):
        summary = {"symbol": "BTC-USD", "leverage": 10}
        report = format_stats_report(sample_stats, config_summary=summary)
        assert "BTC-USD" in report
        assert "Configuration" in report


class TestExportDecisionLog:
    def test_creates_file(self, tmp_path):
        log = [
            {"action": "OPEN_LONG", "price": 50000, "time": "2024-01-01"},
            {"action": "SL_HIT", "price": 49000, "time": "2024-01-02"},
        ]
        filepath = export_decision_log(log, output_dir=str(tmp_path), filename="test_log.json")
        assert Path(filepath).exists()

        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["action"] == "OPEN_LONG"

    def test_creates_output_dir(self, tmp_path):
        subdir = tmp_path / "nested" / "dir"
        log = [{"action": "test"}]
        filepath = export_decision_log(log, output_dir=str(subdir), filename="test.json")
        assert Path(filepath).exists()

    def test_auto_filename(self, tmp_path):
        log = [{"action": "test"}]
        filepath = export_decision_log(log, output_dir=str(tmp_path))
        assert "decision_log_" in filepath
        assert filepath.endswith(".json")


class TestExportTradesCSV:
    def test_creates_csv(self, tmp_path, sample_portfolio):
        filepath = export_trades_csv(sample_portfolio, output_dir=str(tmp_path), filename="trades.csv")
        assert Path(filepath).exists()

        import pandas as pd
        df = pd.read_csv(filepath)
        assert len(df) == 2  # Two closed trades
        assert "direction" in df.columns
        assert "net_pnl" in df.columns
        assert "gross_pnl" in df.columns

    def test_trade_data_correct(self, tmp_path, sample_portfolio):
        filepath = export_trades_csv(sample_portfolio, output_dir=str(tmp_path), filename="trades.csv")
        import pandas as pd
        df = pd.read_csv(filepath)
        assert df.iloc[0]["direction"] == "LONG"
        assert df.iloc[1]["direction"] == "SHORT"
        assert df.iloc[0]["entry_price"] == 50000
