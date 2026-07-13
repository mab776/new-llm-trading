"""Shared timeframe and completed-candle availability rules."""

from __future__ import annotations

import re

import pandas as pd


_TIMEFRAME_RE = re.compile(r"^(\d+)([mhdw])$")


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    """Convert an exchange timeframe such as ``5m`` or ``4h`` to a duration."""
    match = _TIMEFRAME_RE.fullmatch(timeframe)
    if not match:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    count, unit = int(match.group(1)), match.group(2)
    keyword = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}[unit]
    return pd.Timedelta(**{keyword: count})


def timeframe_hours(timeframe: str) -> float:
    """Return timeframe duration in hours (fractional for minute bars)."""
    return timeframe_delta(timeframe) / pd.Timedelta(hours=1)


def decision_close(primary_open, primary_timeframe: str) -> pd.Timestamp:
    """Close timestamp of a primary candle stamped at its open."""
    return pd.Timestamp(primary_open) + timeframe_delta(primary_timeframe)


def last_usable_open(decision_time, timeframe: str) -> pd.Timestamp:
    """Latest bar-open timestamp whose candle is completed by ``decision_time``."""
    return pd.Timestamp(decision_time) - timeframe_delta(timeframe)


def latest_completed_bar_open(timeframe: str, *, now=None) -> pd.Timestamp:
    """UTC-aligned open timestamp of the most recently completed candle."""
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    delta = timeframe_delta(timeframe)
    return current.floor(delta) - delta


def slice_completed_at(
    frame: pd.DataFrame, timeframe: str, decision_time,
) -> pd.DataFrame:
    """Return rows whose close is no later than the decision timestamp."""
    if frame.empty:
        return frame
    return frame[frame.index <= last_usable_open(decision_time, timeframe)]


def completed_market_snapshot(
    data_by_tf: dict[str, pd.DataFrame],
    primary_timeframe: str,
    *,
    now=None,
) -> tuple[dict[str, pd.DataFrame], pd.Timestamp | None]:
    """Freeze every timeframe at the latest completed primary-bar close.

    All input indexes must represent bar OPEN timestamps. The returned primary
    timestamp identifies the one decision bar represented by the snapshot.
    """
    primary = data_by_tf.get(primary_timeframe)
    if primary is None or primary.empty:
        return {}, None
    current = pd.Timestamp.now(tz=primary.index.tz) if now is None else pd.Timestamp(now)
    if primary.index.tz is not None and current.tzinfo is None:
        current = current.tz_localize(primary.index.tz)
    elif primary.index.tz is None and current.tzinfo is not None:
        current = current.tz_localize(None)

    completed_primary = slice_completed_at(primary, primary_timeframe, current)
    if completed_primary.empty:
        return {}, None
    primary_open = pd.Timestamp(completed_primary.index[-1])
    frozen_close = decision_close(primary_open, primary_timeframe)
    snapshot = {
        timeframe: slice_completed_at(frame, timeframe, frozen_close)
        for timeframe, frame in data_by_tf.items()
    }
    snapshot = {timeframe: frame for timeframe, frame in snapshot.items() if not frame.empty}
    return snapshot, primary_open
