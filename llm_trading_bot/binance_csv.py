"""
Binance CSV data downloader — downloads and caches OHLCV data from data.binance.vision.

Approach inspired by tradingml project:
- Downloads monthly ZIP archives from Binance's public data repository
- Falls back to daily ZIPs for the most recent (incomplete) month
- Stores extracted CSVs on disk: history/{spot|futures}/{SYMBOL}/{timeframe}/
- Only downloads what's missing — subsequent runs are instant
- Validates SHA256 checksums before extracting

No API key or rate limiting needed — this is Binance's public data archive.
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HISTORY_ROOT = Path(__file__).parent.parent / "history"

BINANCE_OHLCV_COLUMNS = [
    "time_open", "open", "high", "low", "close", "volume",
    "time_close", "quote_volume", "nb_trades",
    "taker_buy_base", "taker_buy_quote", "_ignore",
]

KEEP_COLUMNS = ["open", "high", "low", "close", "volume"]

TIMEFRAME_MAP = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}


# ---------------------------------------------------------------------------
# Path / URL helpers
# ---------------------------------------------------------------------------

def _get_download_dir(symbol: str, timeframe: str, market: str = "spot") -> Path:
    """Get local directory for cached CSVs: history/{market}/{SYMBOL}/{timeframe}/"""
    d = HISTORY_ROOT / market / symbol / timeframe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _monthly_filename(symbol: str, tf: str, date: datetime) -> str:
    return f"{symbol}-{tf}-{date.strftime('%Y-%m')}"


def _daily_filename(symbol: str, tf: str, date: datetime) -> str:
    return f"{symbol}-{tf}-{date.strftime('%Y-%m-%d')}"


def _binance_url(market: str, delta: str, symbol: str, tf: str, filename: str) -> str:
    """Build the data.binance.vision URL."""
    market_path = "spot" if market == "spot" else "futures/um"
    return f"https://data.binance.vision/data/{market_path}/{delta}/klines/{symbol}/{tf}/{filename}"


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path) -> bool:
    """Download a file, return True if successful."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        return True
    except Exception:
        return False


def _validate_checksum(zip_path: Path, check_path: Path) -> bool:
    """Validate SHA256 checksum of downloaded ZIP."""
    sha256 = hashlib.sha256()
    with open(zip_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)

    with open(check_path, "r") as f:
        expected = f.readline().split(" ", 1)[0].strip()

    return sha256.hexdigest() == expected


def _download_and_extract(
    symbol: str, tf: str, date: datetime,
    delta: str, download_dir: Path, market: str = "spot",
) -> Optional[Path]:
    """
    Download a single ZIP from data.binance.vision, validate, extract, clean up.
    Returns the CSV path if successful, None otherwise.
    """
    if delta == "monthly":
        base_name = _monthly_filename(symbol, tf, date)
    else:
        base_name = _daily_filename(symbol, tf, date)

    csv_path = download_dir / f"{base_name}.csv"
    zip_path = download_dir / f"{base_name}.zip"
    check_path = download_dir / f"{base_name}.zip.CHECKSUM"

    # Already downloaded?
    if csv_path.exists():
        return csv_path

    # Build URLs
    zip_url = _binance_url(market, delta, symbol, tf, f"{base_name}.zip")
    check_url = f"{zip_url}.CHECKSUM"

    # Clean up any previous partial downloads
    for p in [zip_path, check_path]:
        if p.exists():
            os.remove(p)

    # Download ZIP + checksum
    if not _download_file(zip_url, zip_path):
        return None
    if not _download_file(check_url, check_path):
        # No checksum available, clean up and skip validation
        if zip_path.exists():
            os.remove(zip_path)
        return None

    # Validate
    if not _validate_checksum(zip_path, check_path):
        print(f"    WARN: Checksum mismatch for {base_name}, skipping")
        for p in [zip_path, check_path]:
            if p.exists():
                os.remove(p)
        return None

    # Extract
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract(f"{base_name}.csv", download_dir)
    except Exception as e:
        print(f"    WARN: Failed to extract {base_name}: {e}")
        return None

    # Clean up ZIP + checksum
    for p in [zip_path, check_path]:
        if p.exists():
            os.remove(p)

    return csv_path


def _load_csv(csv_path: Path) -> pd.DataFrame:
    """Load a Binance CSV into a DataFrame with proper column names and types."""
    # Detect if header exists (Binance CSVs sometimes have headers, sometimes not)
    sample = pd.read_csv(csv_path, nrows=1)
    has_header = not str(sample.columns[0]).replace(".", "").replace("-", "").isdigit()

    if has_header:
        df = pd.read_csv(csv_path)
        df.columns = BINANCE_OHLCV_COLUMNS[:len(df.columns)]
    else:
        cols = BINANCE_OHLCV_COLUMNS[:len(sample.columns)]
        df = pd.read_csv(csv_path, names=cols, header=None)

    # Canonical project indexes are candle OPEN timestamps. Availability is
    # determined separately from index + timeframe duration; indexing Binance by
    # time_close while Bitget uses time_open makes cross-source causality ambiguous.
    ts_col = "time_open"
    sample_ts = float(df[ts_col].iloc[0])
    if sample_ts > 1e16:
        ts_unit = "ns"
    elif sample_ts > 1e13:
        ts_unit = "us"
    else:
        ts_unit = "ms"

    df["Datetime"] = pd.to_datetime(df[ts_col], unit=ts_unit, utc=True)

    df.set_index("Datetime", inplace=True)

    # Keep only OHLCV
    available = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[available].copy()
    df.columns = [c.capitalize() for c in df.columns]  # open -> Open, etc.
    df = df.astype(float)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_binance_csv(
    symbol: str = "BTCUSDT",
    timeframe: str = "4h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    warmup_days: int = 60,
    market: str = "spot",
) -> pd.DataFrame:
    """
    Download OHLCV data from Binance's public CSV archive.

    Downloads monthly archives first, falls back to daily for the latest
    incomplete month. Caches CSVs on disk — subsequent calls are instant.

    Args:
        symbol: Binance symbol without slash (e.g. "BTCUSDT")
        timeframe: "1h", "4h", "1d"
        start_date: Start of test period (e.g. "2024-06-01")
        end_date: End of test period (e.g. "2025-06-01")
        warmup_days: Extra days before start_date for indicator warmup
        market: "spot" or "futures"

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        DatetimeIndex in UTC
    """
    tf = TIMEFRAME_MAP.get(timeframe, timeframe)
    download_dir = _get_download_dir(symbol, tf, market)

    # Calculate date range
    if end_date:
        end_dt = pd.to_datetime(end_date)
    else:
        end_dt = pd.Timestamp.now()

    if start_date:
        start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)
    else:
        start_dt = end_dt - pd.Timedelta(days=warmup_days + 60)

    # Round start to first of its month
    current = start_dt.replace(day=1)
    end_month = end_dt.replace(day=1) + relativedelta(months=1)

    all_dfs: list[pd.DataFrame] = []
    total_downloaded = 0

    print(f"    {tf} data from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}...", end="", flush=True)

    while current < end_month:
        # Try monthly first
        csv_path = _download_and_extract(symbol, tf, current, "monthly", download_dir, market)

        if csv_path is not None:
            was_cached = True  # We'll check by file existence before download
            df_chunk = _load_csv(csv_path)
            all_dfs.append(df_chunk)
            current += relativedelta(months=1)
            continue

        # Monthly not available — try daily for this month
        day = current
        month_end = current + relativedelta(months=1)
        while day < month_end and day <= end_dt + pd.Timedelta(days=1):
            csv_path = _download_and_extract(symbol, tf, day, "daily", download_dir, market)
            if csv_path is not None:
                df_chunk = _load_csv(csv_path)
                all_dfs.append(df_chunk)
                total_downloaded += 1
            else:
                # No data for this day — probably future date
                pass
            day += pd.Timedelta(days=1)

        current += relativedelta(months=1)

    if not all_dfs:
        raise ValueError(f"No data found for {symbol} {timeframe} from Binance CSV archive")

    # Concat and clean
    df = pd.concat(all_dfs)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    # Trim to requested range (keep warmup)
    start_ts = pd.Timestamp(start_dt, tz="UTC")
    end_ts = pd.Timestamp(end_dt + pd.Timedelta(days=1), tz="UTC")
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]

    df.dropna(inplace=True)
    print(f" {len(df)} candles (cached on disk)")

    return df
