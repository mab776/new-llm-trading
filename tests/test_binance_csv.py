"""Binance archive timestamps follow the project bar-open convention."""

import pandas as pd

from llm_trading_bot.binance_csv import _load_csv


def test_loader_indexes_candles_by_open_not_close(tmp_path):
    path = tmp_path / "BTCUSDT-5m.csv"
    path.write_text(
        "1704067200000,100,101,99,100.5,10,1704067499999,0,1,0,0,0\n"
    )
    frame = _load_csv(path)
    assert frame.index[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
