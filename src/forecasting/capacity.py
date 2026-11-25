"""
Capacity analysis utilities for OSHA forecasting.

This module contains the mathematical operations used to analyze
historical resource utilization and estimate future capacity
exhaustion.

It intentionally contains no infrastructure-specific or ML-framework
code. Predictors can reuse these calculations regardless of whether
the final forecasting model is linear, statistical, or neural.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from math import sqrt
from typing import Sequence

from .features import elapsed_hours
from .types import TelemetryPoint


def clamp_utilization(value: float) -> float:
    """
    Clamp a utilization percentage to the valid range 0..100.

    Forecasting algorithms can mathematically produce values below
    zero or above 100. Infrastructure utilization percentages should
    remain inside their valid range.
    """

    return min(
        100.0,
        max(0.0, value),
    )


def linear_trend(
    points: Sequence[TelemetryPoint],
) -> tuple[float, float, float]:
    """
    Calculate the linear utilization trend.

    Returns:

        slope_per_hour
        intercept
        r_squared

    slope_per_hour indicates how quickly utilization is changing.

    Example:

        +2.5

    means utilization is increasing by approximately 2.5 percentage
    points per hour.

    A negative slope means utilization is decreasing.
    """

    if len(points) < 2:
        raise ValueError(
            "at least two telemetry points are required"
        )

    x = elapsed_hours(points)

    y = tuple(
        point.value
        for point in points
    )

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    x_variance = sum(
        (value - x_mean) ** 2
        for value in x
    )

    if x_variance == 0:
        raise ValueError(
            "telemetry timestamps must span a non-zero duration"
        )

    covariance = sum(
        (x_value - x_mean)
        * (y_value - y_mean)
        for x_value, y_value in zip(x, y)
    )

    slope = covariance / x_variance

    intercept = (
        y_mean
        - (slope * x_mean)
    )

    residual_sum = sum(
        (
            y_value
            - (
                slope * x_value
                + intercept
            )
        ) ** 2
        for x_value, y_value in zip(x, y)
    )

    total_sum = sum(
        (y_value - y_mean) ** 2
        for y_value in y
    )

    if total_sum == 0:
        r_squared = (
            1.0
            if residual_sum == 0
            else 0.0
        )
    else:
        r_squared = (
            1.0
            - residual_sum / total_sum
        )

        r_squared = max(
            0.0,
            min(1.0, r_squared),
        )

    return (
        slope,
        intercept,
        r_squared,
    )


def estimate_threshold_crossing(
    *,
    current_time: datetime,
    current_value: float,
    slope_per_hour: float,
    threshold: float,
) -> datetime | None:
    """
    Estimate when utilization will reach the supplied threshold.

    Returns None when the current trend is flat or falling and the
    threshold has not already been reached.
    """

    if not 0.0 <= current_value <= 100.0:
        raise ValueError(
            "current_value must be between 0 and 100"
        )

    if not 0.0 <= threshold <= 100.0:
        raise ValueError(
            "threshold must be between 0 and 100"
        )

    if current_time.tzinfo is None:
        raise ValueError(
            "current_time must be timezone-aware"
        )

    # Capacity has already reached or exceeded the threshold.
    if current_value >= threshold:
        return current_time

    # A flat or declining trend will not reach a higher threshold
    # under this linear estimate.
    if slope_per_hour <= 0:
        return None

    remaining_capacity = (
        threshold - current_value
    )

    hours_until_threshold = (
        remaining_capacity / slope_per_hour
    )

    return (
        current_time
        + timedelta(
            hours=hours_until_threshold
        )
    )


def hours_until_threshold(
    *,
    current_value: float,
    slope_per_hour: float,
    threshold: float,
) -> float | None:
    """
    Estimate the number of hours before a threshold is reached.

    This is useful when OSHA needs to produce messages such as:

        disk likely to reach 90% in 12 hours
    """

    if current_value >= threshold:
        return 0.0

    if slope_per_hour <= 0:
        return None

    return (
        threshold - current_value
    ) / slope_per_hour


def projected_utilization(
    *,
    current_value: float,
    slope_per_hour: float,
    hours_ahead: float,
) -> float:
    """
    Estimate utilization a number of hours into the future.

    Example:

        current = 70%
        growth = 2% / hour
        horizon = 5 hours

        projected utilization = 80%
    """

    if hours_ahead < 0:
        raise ValueError(
            "hours_ahead cannot be negative"
        )

    projected = (
        current_value
        + slope_per_hour * hours_ahead
    )

    return clamp_utilization(projected)


def residual_standard_error(
    points: Sequence[TelemetryPoint],
    slope: float,
    intercept: float,
) -> float:
    """
    Calculate the residual standard error of a linear forecast.

    This measures how far historical observations deviate from the
    fitted trend and can later be used for prediction intervals.
    """

    if len(points) <= 2:
        return 0.0

    x = elapsed_hours(points)

    squared_errors = sum(
        (
            point.value
            - (
                slope * elapsed
                + intercept
            )
        ) ** 2
        for elapsed, point in zip(
            x,
            points,
        )
    )

    degrees_of_freedom = (
        len(points) - 2
    )

    return sqrt(
        squared_errors
        / degrees_of_freedom
    )


def is_threshold_within_horizon(
    *,
    current_value: float,
    slope_per_hour: float,
    threshold: float,
    horizon_hours: float,
) -> bool:
    """
    Determine whether capacity is expected to cross a threshold
    within the requested forecasting horizon.
    """

    if horizon_hours <= 0:
        raise ValueError(
            "horizon_hours must be greater than zero"
        )

    time_remaining = hours_until_threshold(
        current_value=current_value,
        slope_per_hour=slope_per_hour,
        threshold=threshold,
    )

    if time_remaining is None:
        return False

    return time_remaining <= horizon_hours