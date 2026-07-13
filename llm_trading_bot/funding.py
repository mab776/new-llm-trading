"""
Perpetual funding-rate history — fetch, disk cache, and per-bar aggregation.

Perps charge funding every 8h (00:00/08:00/16:00 UTC) on the position's NOTIONAL:
rate > 0 → longs pay shorts; rate < 0 → shorts pay longs. At high leverage this is a
material carry cost (0.01%/8h on notional = 0.75%/day of margin at 25x), so backtests
that ignore it overstate long profits in bull markets.

Source: Binance BTCUSDT-perp funding via ccxt (full history since 2019, forward-
paginated 1000/page). Bitget only serves ~3 months of funding history, but the rate is
arbitraged across venues — Binance's series is a good proxy for Bitget's (documented
approximation). Live trading needs none of this: the exchange settles funding itself.

Cache: single CSV per (exchange, symbol) under history/funding/, extended incrementally.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd

CACHE_DIR = Path("history/funding")


def _cache_path(symbol: str, exchange: str) -> Path:
    safe = symbol.replace("/", "").replace(":", "_")
    return CACHE_DIR / f"{safe}-{exchange}-funding.csv"


def _load_cache(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    idx = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return pd.Series(df["rate"].values, index=idx, name="rate").sort_index()


def _save_cache(path: Path, series: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit epoch-ms conversion — DatetimeIndex integer views are resolution-
    # dependent (ns vs ms) across pandas versions, so don't use .view("int64").
    ts_ms = [int(t.timestamp() * 1000) for t in series.index]
    out = pd.DataFrame({"timestamp_ms": ts_ms, "rate": series.values})
    out.to_csv(path, index=False)


def _fetch_ccxt(symbol: str, exchange: str, since_ms: int, until_ms: int) -> pd.Series:
    """Forward-paginated funding history fetch via ccxt."""
    import ccxt
    ex = getattr(ccxt, exchange)()
    rows: list[tuple[int, float]] = []
    cursor = since_ms
    while cursor < until_ms:
        batch = ex.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
        if not batch:
            break
        for x in batch:
            ts = int(x["timestamp"])
            if since_ms <= ts <= until_ms:
                rows.append((ts, float(x["fundingRate"])))
        last = int(batch[-1]["timestamp"])
        if last <= cursor:  # no progress — stop
            break
        cursor = last + 1
        time.sleep(0.2)  # be polite
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
    return pd.Series([r[1] for r in rows], index=idx, name="rate").sort_index()


def fetch_funding_history(
    symbol: str = "BTC/USDT:USDT",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
) -> pd.Series:
    """Return the funding-rate series (UTC DatetimeIndex → rate) covering
    [start_date, end_date], using the disk cache and fetching only what's missing."""
    start = pd.Timestamp(start_date or "2020-01-01", tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC") if end_date else pd.Timestamp.now(tz="UTC")

    path = _cache_path(symbol, exchange)
    cached = _load_cache(path)

    fetched = pd.Series(dtype=float)
    if cached.empty:
        fetched = _fetch_ccxt(symbol, exchange, int(start.value // 10**6), int(end.value // 10**6))
    else:
        parts = []
        if start < cached.index[0]:  # missing head
            head = _fetch_ccxt(symbol, exchange, int(start.value // 10**6),
                               int(cached.index[0].value // 10**6) - 1)
            if not head.empty:
                parts.append(head)
        if end > cached.index[-1]:   # missing tail
            tail = _fetch_ccxt(symbol, exchange, int(cached.index[-1].value // 10**6) + 1,
                               int(end.value // 10**6))
            if not tail.empty:
                parts.append(tail)
        if parts:
            fetched = pd.concat(parts)

    if not fetched.empty:
        merged = pd.concat([cached, fetched])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        _save_cache(path, merged)
        cached = merged

    return cached[(cached.index >= start) & (cached.index <= end)]


def aggregate_funding_to_bars(funding: pd.Series, bar_index: pd.DatetimeIndex,
                              bar_hours: float) -> list[float]:
    """Sum funding rates into per-bar buckets: bar i (stamped at its OPEN) collects
    events with bar_open <= t < bar_open + bar_hours. Pure and testable."""
    sums = [0.0] * len(bar_index)
    if funding.empty or len(bar_index) == 0:
        return sums
    ftimes = funding.index
    rates = funding.values
    starts = bar_index
    # position of each funding event among bar opens
    pos = starts.searchsorted(ftimes, side="right") - 1
    delta = pd.Timedelta(hours=bar_hours)
    for p, t, r in zip(pos, ftimes, rates):
        if 0 <= p < len(starts) and starts[p] <= t < starts[p] + delta:
            sums[p] += float(r)
    return sums


def funding_cost(direction: str, rate_sum: float, size: float, mark_price: float) -> float:
    """Signed funding cost for a position over a bar (positive = the position PAYS).
    LONG pays positive rates; SHORT receives them (and vice versa)."""
    notional = size * mark_price
    cost = rate_sum * notional
    return cost if direction == "LONG" else -cost
