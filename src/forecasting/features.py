"""
Feature preparation utilities for OSHA capacity forecasting.

This module prepares historical telemetry before it is passed to a
forecasting model. It intentionally contains no ML-specific code so
different predictors can reuse the same feature pipeline.
"""

from __future__ import annotations

from datetime import datetime
from statistics import fmean
from typing import Iterable, Sequence

from .types import TelemetryPoint


def prepare_series(
    points: Iterable[TelemetryPoint],
) -> tuple[TelemetryPoint, ...]:
    """
    Prepare telemetry for forecasting.

    The series is:
    - deduplicated by timestamp
    - ordered from oldest to newest

    If multiple observations have the same timestamp, the last
    observation supplied is retained.
    """

    indexed: dict[datetime, TelemetryPoint] = {}

    for point in points:
        indexed[point.timestamp] = point

    return tuple(
        indexed[timestamp]
        for timestamp in sorted(indexed)
    )


def values(
    points: Sequence[TelemetryPoint],
) -> tuple[float, ...]:
    """Extract utilization values from telemetry points."""

    return tuple(
        point.value
        for point in points
    )


def elapsed_hours(
    points: Sequence[TelemetryPoint],
) -> tuple[float, ...]:
    """
    Convert telemetry timestamps into hours elapsed from the
    first observation.

    Example:

        12:00 -> 0.0
        13:00 -> 1.0
        14:30 -> 2.5
    """

    if not points:
        return ()

    start = points[0].timestamp

    return tuple(
        (point.timestamp - start).total_seconds() / 3600.0
        for point in points
    )


def recent_window(
    points: Sequence[TelemetryPoint],
    *,
    window_hours: float,
) -> tuple[TelemetryPoint, ...]:
    """
    Return only telemetry contained inside the requested historical
    window.

    The newest telemetry point determines the end of the window.
    """

    if window_hours <= 0:
        raise ValueError(
            "window_hours must be greater than zero"
        )

    if not points:
        return ()

    newest_timestamp = points[-1].timestamp

    cutoff_timestamp = (
        newest_timestamp.timestamp()
        - (window_hours * 3600.0)
    )

    return tuple(
        point
        for point in points
        if point.timestamp.timestamp() >= cutoff_timestamp
    )


def rolling_mean(
    points: Sequence[TelemetryPoint],
    *,
    size: int = 3,
) -> tuple[float, ...]:
    """
    Calculate a rolling utilization average.

    Early observations use however many historical points are
    available until the requested window size is reached.
    """

    if size <= 0:
        raise ValueError(
            "rolling mean size must be greater than zero"
        )

    if not points:
        return ()

    result: list[float] = []

    for index in range(len(points)):
        start = max(
            0,
            index - size + 1,
        )

        window = points[start:index + 1]

        result.append(
            fmean(
                point.value
                for point in window
            )
        )

    return tuple(result)


def utilization_change(
    points: Sequence[TelemetryPoint],
) -> tuple[float, ...]:
    """
    Calculate utilization change between consecutive observations.

    Example:

        60 -> 65 -> 63

    becomes:

        0 -> +5 -> -2
    """

    if not points:
        return ()

    changes: list[float] = [0.0]

    for previous, current in zip(
        points,
        points[1:],
    ):
        changes.append(
            current.value - previous.value
        )

    return tuple(changes)


def utilization_rate_per_hour(
    points: Sequence[TelemetryPoint],
) -> tuple[float, ...]:
    """
    Calculate the rate of utilization change per hour.

    This accounts for irregular telemetry collection intervals.

    A positive value means utilization is increasing.
    A negative value means utilization is decreasing.
    """

    if not points:
        return ()

    rates: list[float] = [0.0]

    for previous, current in zip(
        points,
        points[1:],
    ):
        elapsed_seconds = (
            current.timestamp - previous.timestamp
        ).total_seconds()

        if elapsed_seconds <= 0:
            rates.append(0.0)
            continue

        hours = elapsed_seconds / 3600.0

        rates.append(
            (current.value - previous.value)
            / hours
        )

    return tuple(rates)


def average_utilization(
    points: Sequence[TelemetryPoint],
) -> float:
    """Return the average utilization of the supplied series."""

    if not points:
        return 0.0

    return fmean(
        point.value
        for point in points
    )


def minimum_utilization(
    points: Sequence[TelemetryPoint],
) -> float:
    """Return the minimum utilization in the series."""

    if not points:
        return 0.0

    return min(
        point.value
        for point in points
    )


def maximum_utilization(
    points: Sequence[TelemetryPoint],
) -> float:
    """Return the maximum utilization in the series."""

    if not points:
        return 0.0

    return max(
        point.value
        for point in points
    )


def sample_count(
    points: Sequence[TelemetryPoint],
) -> int:
    """Return the number of available telemetry observations."""

    return len(points)