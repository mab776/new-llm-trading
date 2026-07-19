"""Download Binance USDT-perp futures candles for the scalper research.

5m primary + 15m (aggregated from 5m at load time) + 1h/4h/1d context TFs,
BTC/ETH/SOL, 2020-10 -> present. Monthly archives; the loader falls back to
daily zips for the current month automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_trading_bot.binance_csv import download_binance_csv

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["5m", "1h", "4h", "1d"]
START = "2020-10-01"
END = "2026-07-19"


def main() -> None:
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"== {symbol} {tf} ==", flush=True)
            df = download_binance_csv(
                symbol=symbol, timeframe=tf,
                start_date=START, end_date=END,
                warmup_days=0, market="futures",
            )
            print(f"   rows={len(df)} span={df.index[0]} -> {df.index[-1]}", flush=True)


if __name__ == "__main__":
    main()
