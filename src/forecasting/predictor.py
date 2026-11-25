"""
Capacity forecasting predictors for OSHA.

This module defines the predictor interface used by the forecasting
service and provides an initial linear forecasting implementation.

The interface is intentionally model-independent so OSHA can later
plug in trained forecasting models without changing the forecasting
service.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Sequence

from .capacity import (
    clamp_utilization,
    linear_trend,
    residual_standard_error,
)
from .types import CapacityMetric, ForecastPoint, TelemetryPoint


class CapacityPredictor(ABC):
    """
    Base interface for OSHA capacity forecasting models.

    Every future forecasting implementation should implement this
    interface.

    Possible implementations include:

        LinearCapacityPredictor
        LSTMCapacityPredictor
        TransformerCapacityPredictor
        TCNPredictor
    """

    name: str = "abstract"

    @abstractmethod
    def predict(
        self,
        history: Sequence[TelemetryPoint],
        *,
        horizon_hours: float,
        step_minutes: int = 60,
        metric: CapacityMetric | None = None,
    ) -> tuple[
        tuple[ForecastPoint, ...],
        float,
        float,
    ]:
        """
        Generate future capacity predictions.

        Returns:

            (
                forecast_points,
                slope_per_hour,
                confidence,
            )

        confidence must be between 0 and 1.
        """

        raise NotImplementedError

    def name_for(
        self,
        metric: CapacityMetric,
        history: Sequence[TelemetryPoint],
    ) -> str:
        return self.name

    def trained_for(
        self,
        metric: CapacityMetric,
        history: Sequence[TelemetryPoint],
    ) -> bool:
        return False


class LinearCapacityPredictor(CapacityPredictor):
    """
    Initial OSHA capacity forecasting baseline.

    Uses least-squares linear regression over historical utilization.

    This provides an operational forecasting baseline while allowing
    a trained ML model to replace it later without changing callers.
    """

    name = "linear-capacity-v1"

    def predict(
        self,
        history: Sequence[TelemetryPoint],
        *,
        horizon_hours: float,
        step_minutes: int = 60,
        metric: CapacityMetric | None = None,
    ) -> tuple[
        tuple[ForecastPoint, ...],
        float,
        float,
    ]:
        """
        Forecast utilization across the requested horizon.
        """

        if len(history) < 2:
            raise ValueError(
                "at least two telemetry points are required "
                "for forecasting"
            )

        if horizon_hours <= 0:
            raise ValueError(
                "horizon_hours must be greater than zero"
            )

        if step_minutes <= 0:
            raise ValueError(
                "step_minutes must be greater than zero"
            )

        slope, intercept, r_squared = linear_trend(
            history
        )

        standard_error = residual_standard_error(
            history,
            slope,
            intercept,
        )

        first_timestamp = history[0].timestamp
        last_timestamp = history[-1].timestamp

        elapsed_at_last = (
            last_timestamp - first_timestamp
        ).total_seconds() / 3600.0

        forecast_points: list[ForecastPoint] = []

        step_hours = step_minutes / 60.0

        hours_ahead = step_hours

        while hours_ahead <= horizon_hours:
            future_elapsed = (
                elapsed_at_last
                + hours_ahead
            )

            predicted_value = (
                slope * future_elapsed
                + intercept
            )

            predicted_value = clamp_utilization(
                predicted_value
            )

            # Approximate 95% prediction interval around
            # the fitted baseline.
            margin = (
                1.96
                * standard_error
            )

            lower_bound = clamp_utilization(
                predicted_value - margin
            )

            upper_bound = clamp_utilization(
                predicted_value + margin
            )

            forecast_timestamp = (
                last_timestamp
                + timedelta(
                    hours=hours_ahead
                )
            )

            forecast_points.append(
                ForecastPoint(
                    timestamp=forecast_timestamp,
                    value=predicted_value,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            )

            hours_ahead += step_hours

        # Ensure the exact horizon is represented when the
        # requested horizon isn't divisible by step_minutes.
        if (
            not forecast_points
            or forecast_points[-1].timestamp
            < last_timestamp
            + timedelta(hours=horizon_hours)
        ):
            future_elapsed = (
                elapsed_at_last
                + horizon_hours
            )

            predicted_value = clamp_utilization(
                slope * future_elapsed
                + intercept
            )

            margin = (
                1.96
                * standard_error
            )

            forecast_points.append(
                ForecastPoint(
                    timestamp=(
                        last_timestamp
                        + timedelta(
                            hours=horizon_hours
                        )
                    ),
                    value=predicted_value,
                    lower_bound=clamp_utilization(
                        predicted_value - margin
                    ),
                    upper_bound=clamp_utilization(
                        predicted_value + margin
                    ),
                )
            )

        confidence = max(
            0.0,
            min(
                1.0,
                r_squared,
            ),
        )

        return (
            tuple(forecast_points),
            slope,
            confidence,
        )