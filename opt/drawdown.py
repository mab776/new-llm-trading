"""Read-only drawdown episode analysis for multi-asset equity curves."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class EquityPoint:
    timestamp: pd.Timestamp
    equity: float
    drawdown_pct: float


@dataclass(frozen=True)
class DrawdownEpisode:
    peak_time: pd.Timestamp
    trough_time: pd.Timestamp
    recovery_time: pd.Timestamp | None
    depth_pct: float
    days_to_trough: float
    days_to_recovery: float | None
    total_days: float


@dataclass(frozen=True)
class ThresholdSummary:
    threshold_pct: float
    episodes: int
    bars: int
    time_pct: float
    weeks: int
    weeks_pct: float
    longest_days: float


@dataclass(frozen=True)
class DrawdownStudy:
    episodes: list[DrawdownEpisode] = field(default_factory=list)
    thresholds: dict[float, ThresholdSummary] = field(default_factory=dict)
    max_drawdown_pct: float = 0.0
    median_drawdown_pct: float = 0.0
    p90_drawdown_pct: float = 0.0
    p95_drawdown_pct: float = 0.0
    p99_drawdown_pct: float = 0.0
    underwater_time_pct: float = 0.0


def _days(delta: pd.Timedelta) -> float:
    return delta.total_seconds() / 86_400


def _longest_run_days(points: list[EquityPoint], threshold: float,
                      cadence: pd.Timedelta) -> float:
    longest = current = 0
    for point in points:
        if point.drawdown_pct >= threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return _days(cadence * longest)


def analyze_drawdowns(points: list[EquityPoint],
                      thresholds: tuple[float, ...] = (10, 20, 30, 33)) -> DrawdownStudy:
    """Rank underwater episodes and summarize threshold frequency.

    Each episode starts at the preceding equity peak and ends at recovery to a
    new/equal peak. An unfinished final episode is measured through the final
    sample. Threshold episode counts are distinct peak-to-recovery episodes,
    not the number of underwater bars.
    """
    if not points:
        return DrawdownStudy()
    ordered = sorted(points, key=lambda point: point.timestamp)
    timestamps = [pd.Timestamp(point.timestamp) for point in ordered]
    if len(timestamps) > 1:
        diffs = pd.Series(timestamps).diff().dropna()
        cadence = diffs[diffs > pd.Timedelta(0)].median()
        if pd.isna(cadence):
            cadence = pd.Timedelta(0)
    else:
        cadence = pd.Timedelta(0)

    episodes: list[DrawdownEpisode] = []
    peak_time = timestamps[0]
    active_peak: pd.Timestamp | None = None
    trough_time = timestamps[0]
    depth = 0.0
    eps = 1e-12
    for point, timestamp in zip(ordered, timestamps):
        dd = point.drawdown_pct
        if active_peak is None:
            if dd <= eps:
                peak_time = timestamp
            else:
                active_peak = peak_time
                trough_time, depth = timestamp, dd
        elif dd > eps:
            if dd > depth:
                trough_time, depth = timestamp, dd
        else:
            episodes.append(DrawdownEpisode(
                peak_time=active_peak, trough_time=trough_time,
                recovery_time=timestamp, depth_pct=depth,
                days_to_trough=_days(trough_time - active_peak),
                days_to_recovery=_days(timestamp - trough_time),
                total_days=_days(timestamp - active_peak),
            ))
            active_peak = None
            peak_time = timestamp
            depth = 0.0
    if active_peak is not None:
        end = timestamps[-1]
        episodes.append(DrawdownEpisode(
            peak_time=active_peak, trough_time=trough_time,
            recovery_time=None, depth_pct=depth,
            days_to_trough=_days(trough_time - active_peak),
            days_to_recovery=None, total_days=_days(end - active_peak),
        ))
    episodes.sort(key=lambda episode: episode.depth_pct, reverse=True)

    dds = pd.Series([point.drawdown_pct for point in ordered], dtype=float)
    weeks_all = {(timestamp.isocalendar().year, timestamp.isocalendar().week)
                 for timestamp in timestamps}
    summaries = {}
    for threshold in thresholds:
        bars = sum(point.drawdown_pct >= threshold for point in ordered)
        weeks = {
            (timestamp.isocalendar().year, timestamp.isocalendar().week)
            for point, timestamp in zip(ordered, timestamps)
            if point.drawdown_pct >= threshold
        }
        summaries[threshold] = ThresholdSummary(
            threshold_pct=threshold,
            episodes=sum(episode.depth_pct >= threshold for episode in episodes),
            bars=bars, time_pct=100 * bars / len(ordered), weeks=len(weeks),
            weeks_pct=(100 * len(weeks) / len(weeks_all) if weeks_all else 0.0),
            longest_days=_longest_run_days(ordered, threshold, cadence),
        )
    return DrawdownStudy(
        episodes=episodes, thresholds=summaries,
        max_drawdown_pct=float(dds.max()),
        median_drawdown_pct=float(dds.quantile(.5)),
        p90_drawdown_pct=float(dds.quantile(.9)),
        p95_drawdown_pct=float(dds.quantile(.95)),
        p99_drawdown_pct=float(dds.quantile(.99)),
        underwater_time_pct=100 * float((dds > eps).mean()),
    )
