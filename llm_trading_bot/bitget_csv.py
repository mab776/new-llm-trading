"""
Bitget OHLCV downloader — windowed pagination + monthly disk cache.

Why this module exists (the Bitget quirk):
Bitget's history-candles endpoint is 200-cap and END-anchored — it returns the LAST
`limit` candles *before* the `until` timestamp. A naive open-ended `since` request
therefore silently returns only the tail of the requested range. To fetch a long range
correctly we must page it in EXPLICIT windows of <= 200 candles, passing
`params={"until": window_end}` each time, then keep only the rows inside the window.

Approach (simplified port of tradingml/history_csv.py, futures/swap market):
- Fetch public market data via ccxt (no API keys needed for candles).
- Cache COMPLETE months to disk under history/bitget/{SYMBOL}/{tf}/ — the current
  (partial) month is always re-fetched fresh so live runs stay up to date.
- Return a DataFrame with columns Open, High, Low, Close, Volume and a UTC DatetimeIndex.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dateutil.relativedelta import relativedelta


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HISTORY_ROOT = Path(__file__).parent.parent / "history"

# Bitget caps the history-candles endpoint at 200 rows per request and rejects
# request spans greater than 90 days.
MAX_FETCH_LIMIT = 200
MAX_REQUEST_SPAN_MS = 90 * 86_400_000

# Milliseconds per candle for each supported timeframe.
_TF_MS = {
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}


# ---------------------------------------------------------------------------
# Exchange + windowed fetch (shared with data.py's live fallback)
# ---------------------------------------------------------------------------

def public_exchange(market: str = "futures"):
    """Build a keyless ccxt bitget client for PUBLIC market data (candles only)."""
    import ccxt

    default_type = "swap" if market == "futures" else "spot"
    return ccxt.bitget({
        "enableRateLimit": True,
        "options": {"defaultType": default_type},
    })


def fetch_ohlcv_range(
    exchange,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    """
    Fetch candles in [start_ms, end_ms) using overlapping, sub-90-day windows.

    Bitget treats both boundaries as strict: candles must be after ``since`` and before
    ``until``.  Each request therefore starts one candle before the desired window.  This
    deliberate boundary overlap prevents the first candle of every page/month from being
    lost.  Window size is constrained by both the 200-row limit and the 90-day API limit.

    Returns rows as [ts_ms, open, high, low, close, volume] (ccxt order, extras dropped).
    """
    if timeframe not in _TF_MS:
        raise ValueError(f"Unsupported timeframe for Bitget fetch: {timeframe}")
    if end_ms <= start_ms:
        return []

    gap = _TF_MS[timeframe]
    # The request begins one gap before the desired first candle.  Leave room for that
    # gap inside Bitget's maximum request span.
    max_span_candles = (MAX_REQUEST_SPAN_MS // gap) - 1
    window_candles = min(MAX_FETCH_LIMIT, max_span_candles)
    if window_candles < 1:
        raise ValueError(f"Timeframe {timeframe} cannot fit inside Bitget's 90-day limit")

    rows: list[list] = []
    window_start = start_ms

    while window_start < end_ms:
        window_end = min(end_ms, window_start + window_candles * gap)
        request_start = window_start - gap
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=request_start,
            limit=MAX_FETCH_LIMIT,
            params={"until": window_end},
        )
        if batch:
            rows.extend(r[:6] for r in batch if start_ms <= r[0] < end_ms)
        # Advance unconditionally: an empty leading window may precede listing and this
        # also guarantees termination.
        window_start = window_end

    # Boundary overlap can legitimately return duplicates on adapters whose boundary
    # semantics differ.  Normalize here so every caller gets a stable, validated series.
    by_timestamp = {int(row[0]): row for row in rows}
    normalized = [by_timestamp[ts] for ts in sorted(by_timestamp)]
    _validate_rows(normalized, timeframe)
    return normalized


def _validate_rows(rows: list[list], timeframe: str) -> None:
    """Fail closed on gaps, malformed OHLCV values, or timestamp misalignment."""
    if not rows:
        return

    gap = _TF_MS[timeframe]
    previous: int | None = None
    for row in rows:
        if len(row) < 6:
            raise ValueError(f"Malformed {timeframe} OHLCV row: {row!r}")
        ts = int(row[0])
        if ts % gap:
            raise ValueError(f"Misaligned {timeframe} candle timestamp: {ts}")
        if previous is not None and ts - previous != gap:
            raise ValueError(
                f"Incomplete {timeframe} history: expected {previous + gap}, got {ts}"
            )
        previous = ts

        try:
            open_, high, low, close, volume = (float(value) for value in row[1:6])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric {timeframe} OHLCV row at {ts}") from exc
        values = pd.Series([open_, high, low, close, volume])
        if not values.map(pd.notna).all() or not values.map(lambda value: value != float("inf") and value != float("-inf")).all():
            raise ValueError(f"Non-finite {timeframe} OHLCV row at {ts}")
        if min(open_, high, low, close) <= 0 or volume < 0:
            raise ValueError(f"Invalid non-positive {timeframe} OHLCV row at {ts}")
        if high < max(open_, low, close) or low > min(open_, high, close):
            raise ValueError(f"Invalid OHLC ordering for {timeframe} candle at {ts}")


# ---------------------------------------------------------------------------
# Path / dataframe helpers
# ---------------------------------------------------------------------------

def _sanitize_symbol(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTCUSDT_USDT' (safe for a filesystem path)."""
    return symbol.replace("/", "").replace(":", "_")


def _download_dir(symbol: str, timeframe: str) -> Path:
    d = HISTORY_ROOT / "bitget" / _sanitize_symbol(symbol) / timeframe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _month_csv_path(download_dir: Path, symbol: str, timeframe: str, month: datetime) -> Path:
    return download_dir / f"{_sanitize_symbol(symbol)}-{timeframe}-{month.strftime('%Y-%m')}.csv"


def _rows_to_df(rows: list[list]) -> pd.DataFrame:
    """Build an OHLCV DataFrame (UTC DatetimeIndex) from raw ccxt rows."""
    df = pd.DataFrame(rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df.set_index("Datetime", inplace=True)
    df.drop(columns=["Timestamp"], inplace=True)
    return df.astype(float)


def _load_month_csv(csv_path: Path) -> pd.DataFrame:
    """Load a cached month CSV (index = UTC timestamp, columns OHLCV)."""
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "Datetime"
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def _validate_month_frame(
    df: pd.DataFrame,
    timeframe: str,
    month_start: datetime,
    month_end: datetime,
    *,
    require_complete: bool,
) -> None:
    """Validate cached/fetched month cadence and, for complete months, boundaries."""
    rows = [
        [int(ts.timestamp() * 1000), *values]
        for ts, values in df[["Open", "High", "Low", "Close", "Volume"]].iterrows()
    ]
    _validate_rows(rows, timeframe)
    if not require_complete or df.empty:
        return
    gap = _TF_MS[timeframe]
    expected_first = _to_ms(month_start)
    expected_last = _to_ms(month_end) - gap
    actual_first = int(df.index[0].timestamp() * 1000)
    actual_last = int(df.index[-1].timestamp() * 1000)
    if actual_first != expected_first or actual_last != expected_last:
        raise ValueError(
            f"Incomplete cached {timeframe} month: expected {expected_first}..{expected_last}, "
            f"got {actual_first}..{actual_last}"
        )


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_bitget_csv(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "4h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    warmup_days: int = 60,
    market: str = "futures",
) -> pd.DataFrame:
    """
    Download OHLCV data from Bitget, month by month, caching complete months to disk.

    Args:
        symbol: ccxt unified symbol (futures: "BTC/USDT:USDT", spot: "BTC/USDT")
        timeframe: "1h", "4h", "1d", ... (spot lacks 2h/1s)
        start_date: Start of the test period (warmup is subtracted automatically)
        end_date: End of the test period
        warmup_days: Extra days before start_date for indicator warmup
        market: "futures" (swap) or "spot"

    Returns:
        DataFrame with columns Open, High, Low, Close, Volume and a UTC DatetimeIndex.
    """
    download_dir = _download_dir(symbol, timeframe)

    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
    end_dt = pd.to_datetime(end_date) if end_date else now_utc
    if start_date:
        start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)
    else:
        # Live mode: warmup_days already covers indicator needs; a small buffer
        # keeps the window from reaching back into old exchange data holes.
        start_dt = end_dt - pd.Timedelta(days=warmup_days + 7)

    current = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_month = end_dt.replace(day=1) + relativedelta(months=1)

    exchange = None  # lazy: not created when everything is already cached
    all_dfs: list[pd.DataFrame] = []
    started = False

    print(
        f"    Bitget {timeframe} {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}...",
        end="", flush=True,
    )

    while current < end_month:
        month_end = current + relativedelta(months=1)
        csv_path = _month_csv_path(download_dir, symbol, timeframe, current)

        if csv_path.exists():
            try:
                month_df = _load_month_csv(csv_path)
                _validate_month_frame(
                    month_df, timeframe, current.to_pydatetime(), month_end.to_pydatetime(),
                    require_complete=True,
                )
            except (OSError, ValueError, pd.errors.ParserError):
                # Old pagination omitted boundary candles.  Invalid caches are removed
                # and fetched again instead of silently contaminating a replay.
                csv_path.unlink(missing_ok=True)
            else:
                all_dfs.append(month_df)
                started = True
                current = month_end
                continue

        if exchange is None:
            exchange = public_exchange(market)

        rows = fetch_ohlcv_range(
            exchange, symbol, timeframe,
            _to_ms(current.to_pydatetime()), _to_ms(month_end.to_pydatetime()),
        )

        if not rows:
            if not started:
                # Bitget history doesn't reach this far back yet — skip leading empty months.
                current = month_end
                continue
            # Empty complete month after listing is a data hole, not end-of-history.
            if month_end <= now_utc:
                raise ValueError(f"Missing complete Bitget month: {symbol} {timeframe} {current:%Y-%m}")
            break

        month_df = _rows_to_df(rows)
        # Only archive COMPLETE months; the current/partial month is always re-fetched.
        if month_end <= now_utc:
            _validate_month_frame(
                month_df, timeframe, current.to_pydatetime(), month_end.to_pydatetime(),
                require_complete=started,
            )
            month_df.to_csv(csv_path)

        all_dfs.append(month_df)
        started = True
        current = month_end

    if not all_dfs:
        raise ValueError(f"No data found for {symbol} {timeframe} from Bitget")

    df = pd.concat(all_dfs)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    start_ts = pd.Timestamp(start_dt, tz="UTC")
    end_ts = pd.Timestamp(end_dt + pd.Timedelta(days=1), tz="UTC")
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]
    df.dropna(inplace=True)

    print(f" {len(df)} candles (cached on disk)")
    return df
