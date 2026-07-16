"""
Data fetching module — OHLCV data from multiple sources.

Supports:
- Yahoo Finance (yfinance) — easy, but hourly data capped at ~730 days
- Binance (ccxt) — years of historical data, no API key needed
- Bitget (ccxt) — years of historical data, no API key needed

Handles:
- Multi-timeframe fetching (1h, 4h, 1d)
- 4H candle aggregation from 1H data (for yfinance only; exchanges support 4H natively)
- In-memory caching with configurable TTL
- Extra warmup data for indicator calculation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class CacheEntry:
    data: pd.DataFrame
    timestamp: float
    timeframe: str


class DataCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[pd.DataFrame]:
        entry = self._cache.get(key)
        if entry and (time.time() - entry.timestamp) < self.ttl:
            return entry.data.copy()
        return None

    def put(self, key: str, data: pd.DataFrame, timeframe: str) -> None:
        self._cache[key] = CacheEntry(
            data=data.copy(), timestamp=time.time(), timeframe=timeframe
        )

    def clear(self) -> None:
        self._cache.clear()


# Module-level cache instance
_cache = DataCache()


def configure_cache(ttl_seconds: int) -> None:
    """Reconfigure the global cache TTL."""
    global _cache
    _cache = DataCache(ttl_seconds)


def clear_cache() -> None:
    """Discard cached market frames before a bar-close-sensitive live fetch."""
    _cache.clear()


def aggregate_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-hour OHLCV candles to 4-hour candles.
    Groups by 4-hour blocks aligned to midnight UTC.
    """
    if df_1h.empty:
        return df_1h

    df = df_1h.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Floor to 4H blocks
    df["group"] = df.index.floor("4h")

    agg = df.groupby("group").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })

    # Only keep complete candles (4 bars each, except possibly the last)
    counts = df.groupby("group").size()
    complete = counts[counts == 4].index
    # Always include the last group even if incomplete (it's the current candle)
    if len(agg) > 0:
        last_group = agg.index[-1]
        valid_groups = complete.union(pd.Index([last_group]))
        agg = agg.loc[agg.index.isin(valid_groups)]

    agg.index.name = "Datetime"
    return agg


def _yf_interval(timeframe: str) -> str:
    """Map our timeframe names to yfinance interval strings."""
    mapping = {"1h": "1h", "4h": "1h", "1d": "1d", "1w": "1wk"}
    return mapping.get(timeframe, timeframe)


def _period_for_warmup(timeframe: str, warmup_periods: int = 200) -> tuple[int, int]:
    """
    Calculate how many calendar days of data to fetch for warmup.
    Returns (extra_days, candles_expected_per_day).
    """
    if timeframe == "1h":
        # 1h is a secondary timeframe: indicators need ~warmup_periods hours
        # (≈9 days at 210) plus EMA-seed margin. Keep this window TIGHT — an
        # oversized window drags old exchange data holes into the strict gap
        # validator for zero indicator benefit (all indicators have bounded
        # memory; VWAP is a fixed 100-bar window). A Bitget-wide missing hour
        # (2026-05-19 03:00 UTC) inside the old ~125-day window blocked every
        # live analysis cycle on all symbols.
        days = max(warmup_periods // 24 + 14, 30)
        return days, 24
    if timeframe == "4h":
        days = max(warmup_periods // 6 + 30, 60)
        return days, 24
    elif timeframe == "1d":
        days = warmup_periods + 30
        return days, 1
    else:
        return warmup_periods + 30, 1


# ---------------------------------------------------------------------------
# Source: Yahoo Finance
# ---------------------------------------------------------------------------

def _fetch_yfinance(
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    warmup_periods: int = 200,
) -> pd.DataFrame:
    """Fetch from Yahoo Finance. Hourly data capped at ~730 days."""
    import yfinance as yf

    yf_interval = _yf_interval(timeframe)
    extra_days, _ = _period_for_warmup(timeframe, warmup_periods)

    if end_date:
        end_dt = pd.to_datetime(end_date) + timedelta(days=1)
    else:
        end_dt = pd.Timestamp.now()

    if start_date:
        start_dt = pd.to_datetime(start_date) - timedelta(days=extra_days)
    else:
        # Live mode: extra_days already covers warmup; keep the safety buffer
        # small so the window doesn't reach back into old exchange data holes.
        start_dt = end_dt - timedelta(days=extra_days + 7)

    # yfinance caps hourly data to ~730 days
    if yf_interval == "1h":
        min_start = end_dt - timedelta(days=729)
        if start_dt < min_start:
            start_dt = min_start

    ticker = yf.Ticker(symbol)
    df = ticker.history(
        interval=yf_interval,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError(f"No data returned for {symbol} ({timeframe}) from yfinance")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    # yfinance doesn't support 4H — aggregate from 1H
    if timeframe == "4h":
        df = aggregate_to_4h(df)

    return df


# ---------------------------------------------------------------------------
# Source: CCXT (Binance, Bitget, etc.) — live API fallback
# ---------------------------------------------------------------------------

def _ccxt_timeframe(timeframe: str) -> str:
    """Map our timeframe names to ccxt timeframe strings."""
    mapping = {"1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}
    return mapping.get(timeframe, timeframe)


def _fetch_ccxt(
    symbol: str,
    timeframe: str,
    exchange_id: str = "binance",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    warmup_periods: int = 200,
    market: str = "futures",
) -> pd.DataFrame:
    """
    Fetch OHLCV data from a CCXT-supported exchange (live API).
    Used as fallback when CSV archive is not available.
    No API key needed for public market data.

    Bitget is special: its history endpoint is 200-cap and END-anchored, so we use the
    windowed `until` pagination from bitget_csv (naive forward pagination silently drops
    everything but the tail of the range). Other exchanges (Binance, ...) use the plain
    forward pagination below.
    """
    import ccxt

    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unknown exchange: {exchange_id}")

    ccxt_tf = _ccxt_timeframe(timeframe)
    extra_days, _ = _period_for_warmup(timeframe, warmup_periods)

    if end_date:
        end_dt = pd.to_datetime(end_date) + timedelta(days=1)
    else:
        end_dt = pd.Timestamp.now()

    if start_date:
        start_dt = pd.to_datetime(start_date) - timedelta(days=extra_days)
    else:
        # Live mode: extra_days already covers warmup; keep the safety buffer
        # small so the window doesn't reach back into old exchange data holes.
        start_dt = end_dt - timedelta(days=extra_days + 7)

    since_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Bitget: windowed END-anchored pagination (shared helper — the correct way).
    if exchange_id == "bitget":
        from llm_trading_bot.bitget_csv import fetch_ohlcv_range, public_exchange

        exchange = public_exchange(market)
        print(f"    Fetching {timeframe} from bitget (windowed)...", end="", flush=True)
        all_candles = fetch_ohlcv_range(exchange, symbol, ccxt_tf, since_ms, end_ms)
        print(f" {len(all_candles)} candles")
    else:
        exchange = exchange_class({"enableRateLimit": True})

        # CCXT returns max 500-1500 candles per request, so paginate forward
        all_candles = []
        limit = 1000
        current_since = since_ms

        print(f"    Fetching {timeframe} from {exchange_id}...", end="", flush=True)
        while current_since < end_ms:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol, ccxt_tf, since=current_since, limit=limit
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to fetch {symbol} {timeframe} from {exchange_id}: {e}"
                ) from e

            if not candles:
                break

            all_candles.extend(candles)

            # Move to after the last candle timestamp
            last_ts = candles[-1][0]
            if last_ts <= current_since:
                break  # No progress — avoid infinite loop
            current_since = last_ts + 1

            # Stop if we got less than limit (no more data)
            if len(candles) < limit:
                break

        print(f" {len(all_candles)} candles")

    if not all_candles:
        raise ValueError(f"No data returned for {symbol} ({timeframe}) from {exchange_id}")

    df = pd.DataFrame(
        all_candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df.set_index("Datetime", inplace=True)
    df.drop(columns=["Timestamp"], inplace=True)

    # Filter to requested range (keep warmup + test period)
    df = df[df.index <= pd.Timestamp(end_dt, tz="UTC")]

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    # Remove duplicates (can happen at pagination boundaries)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    return df


# ---------------------------------------------------------------------------
# Source: Binance CSV archive (preferred — cached on disk)
# ---------------------------------------------------------------------------

def _fetch_binance_csv(
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    warmup_periods: int = 200,
    market: str = "spot",
) -> pd.DataFrame:
    """
    Fetch from Binance's public CSV archive (data.binance.vision).
    Downloads monthly/daily ZIPs, caches as CSVs on disk.
    Subsequent runs are instant — no network calls needed.
    """
    from llm_trading_bot.binance_csv import download_binance_csv

    extra_days, _ = _period_for_warmup(timeframe, warmup_periods)

    # Convert symbol format: "BTC/USDT" -> "BTCUSDT"
    csv_symbol = symbol.replace("/", "")

    return download_binance_csv(
        symbol=csv_symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        warmup_days=extra_days,
        market=market,
    )


# ---------------------------------------------------------------------------
# Source: Bitget CSV archive (windowed pagination + monthly disk cache)
# ---------------------------------------------------------------------------

def _fetch_bitget_csv(
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    warmup_periods: int = 200,
    market: str = "futures",
) -> pd.DataFrame:
    """
    Fetch from Bitget with the correct END-anchored windowed pagination, caching
    complete months to disk. Subsequent runs are instant.
    """
    from llm_trading_bot.bitget_csv import download_bitget_csv

    extra_days, _ = _period_for_warmup(timeframe, warmup_periods)

    return download_bitget_csv(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        warmup_days=extra_days,
        market=market,
    )


# ---------------------------------------------------------------------------
# Unified fetch interface
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    warmup_periods: int = 200,
    source: str = "yfinance",
    exchange: str = "binance",
    market: str = "futures",
) -> pd.DataFrame:
    """
    Fetch OHLCV data from the configured source.

    Args:
        symbol: Trading pair (e.g. "BTC-USD" for yfinance, "BTC/USDT" for ccxt/binance)
        timeframe: "1h", "4h", "1d"
        start_date: Start of test period (warmup is added automatically)
        end_date: End of test period
        warmup_periods: Extra bars before start_date for indicator warmup
        source: "yfinance", "binance", "bitget", or any ccxt exchange name
        exchange: Alias for source when using ccxt

    Source routing:
        "yfinance" -> Yahoo Finance (capped at 730 days hourly)
        "binance"  -> Binance CSV archive (disk-cached, years of data) with ccxt fallback
        "bitget", other -> CCXT live API
    """
    cache_key = (
        f"{source}_{exchange}_{market}_{symbol}_{timeframe}_"
        f"{start_date}_{end_date}_{warmup_periods}"
    )
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    if source == "yfinance":
        df = _fetch_yfinance(symbol, timeframe, start_date, end_date, warmup_periods)
    elif source == "binance":
        # Prefer CSV archive (fast, disk-cached), fall back to ccxt API
        try:
            df = _fetch_binance_csv(
                symbol, timeframe, start_date, end_date, warmup_periods, market
            )
        except Exception as e:
            print(f"    CSV archive failed ({e}), falling back to ccxt API...")
            df = _fetch_ccxt(
                symbol, timeframe, "binance", start_date, end_date, warmup_periods, market
            )
    elif source == "bitget":
        # Prefer the windowed CSV archive (disk-cached), fall back to the windowed live API
        try:
            df = _fetch_bitget_csv(symbol, timeframe, start_date, end_date, warmup_periods, market)
        except Exception as e:
            print(f"    Bitget CSV failed ({e}), falling back to windowed ccxt live fetch...")
            df = _fetch_ccxt(symbol, timeframe, "bitget", start_date, end_date, warmup_periods, market)
    else:
        # Treat source as a ccxt exchange ID
        exchange_id = source if source != "ccxt" else exchange
        df = _fetch_ccxt(symbol, timeframe, exchange_id, start_date, end_date, warmup_periods, market)

    _cache.put(cache_key, df, timeframe)
    return df


def fetch_multi_timeframe(
    symbol: str,
    timeframes: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    warmup_periods: int = 200,
    source: str = "yfinance",
    exchange: str = "binance",
    market: str = "futures",
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple timeframes."""
    result: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    for tf in timeframes:
        try:
            result[tf] = fetch_ohlcv(
                symbol, tf, start_date, end_date, warmup_periods,
                source=source, exchange=exchange, market=market,
            )
        except Exception as e:
            failures.append(f"{tf}: {e}")
    if failures:
        raise ValueError(
            f"Required market data unavailable for {symbol}: " + "; ".join(failures)
        )
    return result
