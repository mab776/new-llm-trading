"""
Tests for the Bitget windowed history getter.

The whole point of this module is that Bitget's history endpoint is 200-cap and
END-anchored — so these tests pin down the windowing contract (limit=200, an explicit
`until` per window, windows advancing by 200*gap, correct range filtering, termination
on empty windows) and the monthly disk cache (complete months written, current month not).
"""

import pandas as pd
import pytest

from llm_trading_bot import bitget_csv
from llm_trading_bot.bitget_csv import (
    MAX_FETCH_LIMIT,
    _TF_MS,
    download_bitget_csv,
    fetch_ohlcv_range,
)


class FakeExchange:
    """
    Simulates Bitget's strict boundaries: rows are after `since` and before `until`.

    Records every fetch_ohlcv call. Returns up to `limit` candles at `timeframe` spacing
    starting at `since`, stopping before `until`. Timestamps earlier than `listing_ms`
    return nothing (models pre-listing history holes).
    """

    def __init__(self, timeframe: str, listing_ms: int = 0):
        self.gap = _TF_MS[timeframe]
        self.listing_ms = listing_ms
        self.calls: list[dict] = []

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None, params=None):
        params = params or {}
        self.calls.append({"since": since, "limit": limit, "params": dict(params)})
        until = params.get("until")
        rows = []
        ts = since + self.gap
        while ts < until and len(rows) < limit:
            if ts >= self.listing_ms:
                # [ts, open, high, low, close, volume] + an extra field to ensure we slice [:6]
                rows.append([ts, 100.0, 101.0, 99.0, 100.5, 10.0, "EXTRA"])
            ts += self.gap
        return rows


class TestFetchOhlcvRange:
    def test_passes_until_and_limit_every_call(self):
        tf = "1h"
        gap = _TF_MS[tf]
        start, end = 0, 500 * gap  # spans multiple 200-candle windows
        ex = FakeExchange(tf)
        fetch_ohlcv_range(ex, "BTC/USDT:USDT", tf, start, end)

        assert ex.calls, "expected at least one fetch call"
        for c in ex.calls:
            assert c["limit"] == MAX_FETCH_LIMIT == 200
            assert "until" in c["params"], "every window must be END-anchored with `until`"

    def test_windows_advance_by_200_gap(self):
        tf = "1h"
        gap = _TF_MS[tf]
        start, end = 0, 500 * gap
        ex = FakeExchange(tf)
        fetch_ohlcv_range(ex, "BTC/USDT:USDT", tf, start, end)

        sinces = [c["since"] for c in ex.calls]
        # Each request overlaps its desired window by one candle so strict `since`
        # semantics cannot omit the boundary candle.
        assert sinces == [-gap, 199 * gap, 399 * gap]
        # Each window's `until` = min(end, since + 200*gap)
        assert ex.calls[0]["params"]["until"] == 200 * gap
        assert ex.calls[-1]["params"]["until"] == end

    def test_daily_windows_respect_90_day_span(self):
        gap = _TF_MS["1d"]
        ex = FakeExchange("1d")
        fetch_ohlcv_range(ex, "BTC/USDT:USDT", "1d", 0, 250 * gap)

        assert len(ex.calls) == 3
        assert all(c["params"]["until"] - c["since"] <= 90 * gap for c in ex.calls)

    def test_strict_page_boundaries_have_no_missing_candles(self):
        gap = _TF_MS["4h"]
        ex = FakeExchange("4h")
        rows = fetch_ohlcv_range(ex, "BTC/USDT:USDT", "4h", 0, 401 * gap)
        assert [row[0] for row in rows] == list(range(0, 401 * gap, gap))

    def test_gap_inside_history_fails_closed(self):
        gap = _TF_MS["1h"]

        class GappedExchange(FakeExchange):
            def fetch_ohlcv(self, *args, **kwargs):
                return [row for row in super().fetch_ohlcv(*args, **kwargs) if row[0] != 50 * gap]

        with pytest.raises(ValueError, match="Incomplete 1h history"):
            fetch_ohlcv_range(GappedExchange("1h"), "BTC/USDT:USDT", "1h", 0, 100 * gap)

    def test_rows_are_within_range_and_complete(self):
        tf = "1h"
        gap = _TF_MS[tf]
        start, end = 10 * gap, 460 * gap
        ex = FakeExchange(tf)
        rows = fetch_ohlcv_range(ex, "BTC/USDT:USDT", tf, start, end)

        ts = [r[0] for r in rows]
        assert min(ts) == start
        assert max(ts) == end - gap                 # end is exclusive
        assert all(start <= t < end for t in ts)     # nothing outside [start, end)
        assert len(rows) == (end - start) // gap     # no gaps, no dups
        assert all(len(r) == 6 for r in rows)        # extra ccxt field dropped

    def test_empty_leading_window_skipped_and_terminates(self):
        tf = "1h"
        gap = _TF_MS[tf]
        # Listing starts at window 2 (400*gap). First two windows return nothing.
        listing = 400 * gap
        start, end = 0, 500 * gap
        ex = FakeExchange(tf, listing_ms=listing)
        rows = fetch_ohlcv_range(ex, "BTC/USDT:USDT", tf, start, end)

        assert rows, "should still collect rows after the empty leading windows"
        assert min(r[0] for r in rows) == listing
        # Terminated cleanly (didn't loop forever): exactly 3 windows attempted.
        assert len(ex.calls) == 3

    def test_unsupported_timeframe_raises(self):
        with pytest.raises(ValueError):
            fetch_ohlcv_range(FakeExchange("1h"), "BTC/USDT:USDT", "1s", 0, 10)


class TestDownloadBitgetCsvCache:
    @pytest.fixture(autouse=True)
    def _tmp_history(self, tmp_path, monkeypatch):
        """Redirect the on-disk cache to a temp dir and inject the fake exchange."""
        monkeypatch.setattr(bitget_csv, "HISTORY_ROOT", tmp_path / "history")
        self._fake = None

        def _fake_public_exchange(market="futures"):
            self._fake = FakeExchange("1d")
            return self._fake

        monkeypatch.setattr(bitget_csv, "public_exchange", _fake_public_exchange)
        self.tmp_path = tmp_path

    def test_returns_ohlcv_frame(self):
        df = download_bitget_csv(
            "BTC/USDT:USDT", "1d",
            start_date="2022-01-05", end_date="2022-03-20", warmup_days=0,
        )
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"
        assert df.index.is_monotonic_increasing
        assert not df.index.has_duplicates

    def test_complete_months_written_to_disk(self):
        download_bitget_csv(
            "BTC/USDT:USDT", "1d",
            start_date="2022-01-05", end_date="2022-03-20", warmup_days=0,
        )
        cache_dir = bitget_csv.HISTORY_ROOT / "bitget" / "BTCUSDT_USDT" / "1d"
        written = sorted(p.name for p in cache_dir.glob("*.csv"))
        # Jan, Feb, Mar 2022 are all complete (well in the past) -> all cached.
        assert any("2022-01" in n for n in written)
        assert any("2022-02" in n for n in written)
        assert any("2022-03" in n for n in written)

    def test_current_month_not_cached(self):
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        start = (now - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        download_bitget_csv("BTC/USDT:USDT", "1d", start_date=start, end_date=end, warmup_days=0)

        cache_dir = bitget_csv.HISTORY_ROOT / "bitget" / "BTCUSDT_USDT" / "1d"
        current_tag = now.strftime("%Y-%m")
        written = [p.name for p in cache_dir.glob("*.csv")]
        assert not any(current_tag in n for n in written), (
            "the current (partial) month must never be cached"
        )
