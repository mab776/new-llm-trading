"""
Tests for the data module.
Covers: cache, 4H aggregation, timeframe mapping, warmup calculation.
"""

import time

import numpy as np
import pandas as pd
import pytest

from llm_trading_bot import data as data_mod
from llm_trading_bot.data import (
    DataCache,
    aggregate_to_4h,
    fetch_ohlcv,
    _yf_interval,
    _period_for_warmup,
    configure_cache,
    fetch_multi_timeframe,
)


def _tiny_df():
    idx = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"Open": [1, 2, 3], "High": [1, 2, 3], "Low": [1, 2, 3],
         "Close": [1, 2, 3], "Volume": [1, 2, 3]},
        index=idx,
    )


class TestBitgetRouting:
    def test_bitget_source_uses_csv_getter(self, monkeypatch):
        configure_cache(0)  # disable cache so the branch runs
        called = {}

        def fake_csv(symbol, timeframe, start_date, end_date, warmup_periods, market):
            called["market"] = market
            return _tiny_df()

        monkeypatch.setattr(data_mod, "_fetch_bitget_csv", fake_csv)
        df = fetch_ohlcv("BTC/USDT:USDT", "4h", source="bitget", market="futures")
        assert called["market"] == "futures"
        assert len(df) == 3

    def test_multi_timeframe_fails_closed_if_one_fetch_fails(self, monkeypatch):
        def fake_fetch(symbol, timeframe, *args, **kwargs):
            if timeframe == "1d":
                raise RuntimeError("missing daily data")
            return _tiny_df()

        monkeypatch.setattr(data_mod, "fetch_ohlcv", fake_fetch)
        with pytest.raises(ValueError, match="1d: missing daily data"):
            fetch_multi_timeframe("BTC/USDT:USDT", ["4h", "1d"], source="bitget")

    def test_bitget_falls_back_to_windowed_ccxt_on_cache_error(self, monkeypatch):
        configure_cache(0)

        def boom(*a, **k):
            raise RuntimeError("disk cache unavailable")

        fell_back = {}

        def fake_ccxt(symbol, timeframe, exchange_id, start_date, end_date, warmup_periods, market):
            fell_back["exchange_id"] = exchange_id
            return _tiny_df()

        monkeypatch.setattr(data_mod, "_fetch_bitget_csv", boom)
        monkeypatch.setattr(data_mod, "_fetch_ccxt", fake_ccxt)
        df = fetch_ohlcv("BTC/USDT:USDT", "4h", source="bitget")
        assert fell_back["exchange_id"] == "bitget"
        assert len(df) == 3


# ──────────────────────────────────────────────────────────────────────
# Cache Tests
# ──────────────────────────────────────────────────────────────────────

class TestDataCache:
    def test_put_and_get(self):
        cache = DataCache(ttl_seconds=60)
        df = pd.DataFrame({"A": [1, 2, 3]})
        cache.put("key1", df, "4h")
        result = cache.get("key1")
        assert result is not None
        assert list(result["A"]) == [1, 2, 3]

    def test_get_returns_copy(self):
        """Mutating the returned df should not affect the cache."""
        cache = DataCache(ttl_seconds=60)
        df = pd.DataFrame({"A": [1, 2, 3]})
        cache.put("key1", df, "4h")
        result = cache.get("key1")
        result["A"] = [10, 20, 30]
        original = cache.get("key1")
        assert list(original["A"]) == [1, 2, 3]

    def test_expired_entry_returns_none(self):
        cache = DataCache(ttl_seconds=0)  # Expire immediately
        df = pd.DataFrame({"A": [1]})
        cache.put("key1", df, "4h")
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_missing_key_returns_none(self):
        cache = DataCache(ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_clear(self):
        cache = DataCache(ttl_seconds=60)
        cache.put("k1", pd.DataFrame({"A": [1]}), "4h")
        cache.put("k2", pd.DataFrame({"A": [2]}), "1h")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None


# ──────────────────────────────────────────────────────────────────────
# 4H Aggregation Tests
# ──────────────────────────────────────────────────────────────────────

class TestAggregateTo4H:
    @pytest.fixture
    def hourly_data(self):
        """Generate 48 hours (2 days) of 1H OHLCV data."""
        n = 48
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        np.random.seed(42)
        base = 50000
        prices = base + np.cumsum(np.random.normal(0, 50, n))

        return pd.DataFrame({
            "Open": prices + np.random.uniform(-20, 20, n),
            "High": prices + abs(np.random.normal(0, 30, n)),
            "Low": prices - abs(np.random.normal(0, 30, n)),
            "Close": prices,
            "Volume": np.random.uniform(100, 500, n) * 1e6,
        }, index=dates)

    def test_output_length(self, hourly_data):
        """48 hours / 4 = 12 four-hour candles."""
        result = aggregate_to_4h(hourly_data)
        assert len(result) == 12

    def test_ohlcv_aggregation_rules(self, hourly_data):
        """Open=first, High=max, Low=min, Close=last, Volume=sum."""
        result = aggregate_to_4h(hourly_data)

        # Check first 4H candle (hours 0-3)
        first_block = hourly_data.iloc[:4]
        first_candle = result.iloc[0]

        assert first_candle["Open"] == first_block["Open"].iloc[0]
        assert first_candle["High"] == first_block["High"].max()
        assert first_candle["Low"] == first_block["Low"].min()
        assert first_candle["Close"] == first_block["Close"].iloc[-1]
        assert first_candle["Volume"] == pytest.approx(first_block["Volume"].sum())

    def test_empty_input(self):
        empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        result = aggregate_to_4h(empty_df)
        assert result.empty

    def test_high_gte_low(self, hourly_data):
        result = aggregate_to_4h(hourly_data)
        assert (result["High"] >= result["Low"]).all()


# ──────────────────────────────────────────────────────────────────────
# Utility Function Tests
# ──────────────────────────────────────────────────────────────────────

class TestYfInterval:
    def test_direct_mapping(self):
        assert _yf_interval("1h") == "1h"
        assert _yf_interval("1d") == "1d"

    def test_4h_maps_to_1h(self):
        """yfinance doesn't support 4H, so we fetch 1H and aggregate."""
        assert _yf_interval("4h") == "1h"

    def test_unknown_passthrough(self):
        assert _yf_interval("15m") == "15m"


class TestPeriodForWarmup:
    def test_hourly_warmup(self):
        days, cpd = _period_for_warmup("1h", warmup_periods=200)
        assert days >= 30  # Should request enough days
        assert cpd == 24

    def test_4h_warmup(self):
        days, cpd = _period_for_warmup("4h", warmup_periods=200)
        assert days >= 30
        assert cpd == 24

    def test_daily_warmup(self):
        days, cpd = _period_for_warmup("1d", warmup_periods=200)
        assert days >= 200
        assert cpd == 1


class TestConfigureCache:
    def test_reconfigure(self):
        """configure_cache should update the global cache TTL."""
        configure_cache(120)
        # No assertion needed — just ensure it doesn't crash.
        # Real validation is that subsequent fetches use the new TTL.
        configure_cache(300)  # Reset
