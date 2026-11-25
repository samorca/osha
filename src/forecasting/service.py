"""
OSHA capacity forecasting service.

This module provides the main application-facing interface for
capacity forecasting.

It coordinates:

    historical telemetry
        -> feature preparation
        -> capacity prediction
        -> threshold crossing analysis
        -> risk classification
        -> CapacityForecast

The service does not perform remediation. It only produces forecasting
evidence that other OSHA layers can consume.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

from .features import prepare_series, recent_window
from .predictor import CapacityPredictor, LinearCapacityPredictor
from .thresholds import ThresholdPolicy
from .types import (
    CapacityForecast,
    CapacityMetric,
    ForecastPoint,
    ForecastStatus,
    TelemetryPoint,
)


class CapacityForecastingService:
    """
    Main service for OSHA infrastructure capacity forecasting.

    The service accepts historical utilization observations and
    produces a standardized CapacityForecast.

    By default it uses LinearCapacityPredictor as the initial
    forecasting baseline. A trained model can later be injected by
    implementing CapacityPredictor.
    """

    def __init__(
        self,
        predictor: CapacityPredictor | None = None,
        threshold_policy: ThresholdPolicy | None = None,
        *,
        minimum_points: int = 4,
        training_window_hours: float = 24.0,
    ) -> None:
        if minimum_points < 2:
            raise ValueError(
                "minimum_points must be at least 2"
            )

        if training_window_hours <= 0:
            raise ValueError(
                "training_window_hours must be greater than zero"
            )

        self.predictor = (
            predictor
            if predictor is not None
            else LinearCapacityPredictor()
        )

        self.threshold_policy = (
            threshold_policy
            if threshold_policy is not None
            else ThresholdPolicy()
        )

        self.minimum_points = minimum_points
        self.training_window_hours = training_window_hours

    def forecast(
        self,
        *,
        resource_id: str,
        metric: CapacityMetric,
        history: Iterable[TelemetryPoint],
        horizon_hours: float = 12.0,
        step_minutes: int = 60,
    ) -> CapacityForecast:
        """
        Forecast future utilization for one infrastructure resource.

        Example:

            resource_id = "node-01:/var"
            metric = CapacityMetric.DISK
            horizon_hours = 12

        The result can indicate that a critical threshold is expected
        to be crossed during that horizon.
        """

        if not resource_id:
            raise ValueError(
                "resource_id cannot be empty"
            )

        if horizon_hours <= 0:
            raise ValueError(
                "horizon_hours must be greater than zero"
            )

        if step_minutes <= 0:
            raise ValueError(
                "step_minutes must be greater than zero"
            )

        #
        # 1. Prepare telemetry.
        #

        prepared_history = prepare_series(
            history
        )

        #
        # 2. Keep only the configured historical window.
        #

        series = recent_window(
            prepared_history,
            window_hours=self.training_window_hours,
        )

        generated_at = datetime.now(
            timezone.utc
        )

        #
        # 3. Load thresholds for this metric.
        #

        threshold_band = (
            self.threshold_policy.band_for(
                metric
            )
        )

        critical_threshold = (
            threshold_band.critical
        )
        model_name = self.predictor.name_for(metric, series)
        trained_model = self.predictor.trained_for(metric, series)

        #
        # 4. Handle insufficient telemetry.
        #
        # OSHA should explicitly report that forecasting cannot yet
        # be performed rather than manufacturing a prediction.
        #

        if len(series) < self.minimum_points:
            current_value = (
                series[-1].value
                if series
                else 0.0
            )

            return CapacityForecast(
                resource_id=resource_id,
                metric=metric,
                generated_at=generated_at,
                horizon_hours=horizon_hours,
                current_value=current_value,
                threshold=critical_threshold,
                status=(
                    ForecastStatus.INSUFFICIENT_DATA
                ),
                points=(),
                predicted_threshold_time=None,
                time_to_threshold_hours=None,
                confidence=0.0,
                model_name=model_name,
                metadata={
                    "sample_count": len(series),
                    "minimum_points": self.minimum_points,
                    "training_window_hours": (
                        self.training_window_hours
                    ),
                    "trained_model": trained_model,
                },
            )

        #
        # 5. Generate future utilization predictions.
        #

        (
            forecast_points,
            slope_per_hour,
            confidence,
        ) = self.predictor.predict(
            series,
            horizon_hours=horizon_hours,
            step_minutes=step_minutes,
            metric=metric,
        )

        current_value = (
            series[-1].value
        )

        current_timestamp = (
            series[-1].timestamp
        )

        #
        # 6. Estimate critical threshold crossing.
        #

        predicted_threshold_time = self._threshold_crossing_time(
            current_time=current_timestamp,
            current_value=current_value,
            points=forecast_points,
            threshold=critical_threshold,
        )

        #
        # 7. Ensure the predicted crossing happens inside the
        # requested forecasting horizon.
        #

        horizon_seconds = (
            horizon_hours * 3600.0
        )

        horizon_end_timestamp = (
            current_timestamp.timestamp()
            + horizon_seconds
        )

        if (
            predicted_threshold_time is not None
            and predicted_threshold_time.timestamp()
            > horizon_end_timestamp
        ):
            predicted_threshold_time = None

        #
        # 8. Calculate time remaining before the critical threshold.
        #

        time_to_threshold_hours: float | None = None

        if predicted_threshold_time is not None:
            time_to_threshold_hours = max(
                0.0,
                (
                    predicted_threshold_time
                    - current_timestamp
                ).total_seconds()
                / 3600.0,
            )

        #
        # 9. Determine the highest utilization expected within the
        # forecasting horizon.
        #

        maximum_forecast_value = max(
            (
                point.value
                for point in forecast_points
            ),
            default=current_value,
        )

        evaluation_value = max(
            current_value,
            maximum_forecast_value,
        )

        #
        # 10. Convert forecast utilization into OSHA capacity status.
        #

        status = (
            self.threshold_policy.status_for(
                metric,
                evaluation_value,
            )
        )

        #
        # 11. Produce standardized forecast result.
        #

        return CapacityForecast(
            resource_id=resource_id,
            metric=metric,
            generated_at=generated_at,
            horizon_hours=horizon_hours,
            current_value=current_value,
            threshold=critical_threshold,
            status=status,
            points=forecast_points,
            predicted_threshold_time=(
                predicted_threshold_time
            ),
            time_to_threshold_hours=(
                time_to_threshold_hours
            ),
            confidence=confidence,
            model_name=model_name,
            metadata={
                "sample_count": len(series),
                "minimum_points": self.minimum_points,
                "training_window_hours": (
                    self.training_window_hours
                ),
                "slope_per_hour": slope_per_hour,
                "maximum_forecast_value": (
                    maximum_forecast_value
                ),
                "trained_model": trained_model,
            },
        )

    @staticmethod
    def _threshold_crossing_time(
        *,
        current_time: datetime,
        current_value: float,
        points: Sequence[ForecastPoint],
        threshold: float,
    ) -> datetime | None:
        if current_value >= threshold:
            return current_time

        previous_time = current_time
        previous_value = current_value
        for point in points:
            if point.value >= threshold:
                change = point.value - previous_value
                if change <= 0:
                    return point.timestamp
                fraction = (threshold - previous_value) / change
                return previous_time + (
                    point.timestamp - previous_time
                ) * fraction
            previous_time = point.timestamp
            previous_value = point.value
        return None