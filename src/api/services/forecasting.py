from __future__ import annotations

import math
import re

from typing import Any

from src.api.errors import (
    FeatureDisabledError,
    InferenceInputAPIError,
)
from src.api.schema import (
    APIResponseMetadata,
    CapacityForecastPointResponse,
    CapacityForecastRequest,
    CapacityForecastResponse,
    CapacityForecastStatusAPI,
    CapacityMetricAPI,
)
from src.ai.schema import AIModelFamily
from src.forecasting import (
    CapacityPredictor,
    CapacityForecastingService,
    CapacityMetric,
    ForecastingConfig,
    TelemetryPoint,
    ThresholdPolicy,
)
from src.forecasting.config import (
    from_environment as forecasting_config_from_environment,
)


CAPACITY_FORECAST_SERVICE_NAME = "capacity_forecasting"
MAX_FORECAST_POINTS = 1000


class CapacityForecastAPIService:
    """Translate public API requests to the capacity forecasting domain."""

    def __init__(
        self,
        *,
        config: ForecastingConfig | None = None,
        forecasting: CapacityForecastingService | None = None,
        model_registry: Any = None,
    ) -> None:
        self.config = config or ForecastingConfig()
        self.config.validate()
        self.forecasting = forecasting or CapacityForecastingService(
            threshold_policy=ThresholdPolicy(),
            minimum_points=self.config.minimum_points,
            training_window_hours=self.config.training_window_hours,
        )
        self.model_registry = model_registry
        self._closed = False

    @staticmethod
    def _version_key(version: str) -> tuple[int, ...]:
        parts = re.findall(r"\d+", version)
        return tuple(int(part) for part in parts)

    def _forecasting_service(self) -> CapacityForecastingService:
        if self.model_registry is None:
            return self.forecasting

        descriptors = self.model_registry.descriptors(
            family=AIModelFamily.CAPACITY_FORECASTING.value,
        )
        if not descriptors:
            return self.forecasting

        descriptor = max(
            descriptors,
            key=lambda item: (
                self._version_key(item.identity.version),
                item.identity.model_name,
            ),
        )
        adapter = self.model_registry.load(descriptor.identity.model_id)
        predictor = getattr(adapter, "model", None)
        if not isinstance(predictor, CapacityPredictor):
            raise RuntimeError(
                "The registered capacity-forecast artifact did not load a predictor."
            )

        return CapacityForecastingService(
            predictor=predictor,
            threshold_policy=self.forecasting.threshold_policy,
            minimum_points=self.forecasting.minimum_points,
            training_window_hours=self.forecasting.training_window_hours,
        )

    def forecast(
        self,
        request: CapacityForecastRequest,
    ) -> CapacityForecastResponse:
        if self._closed:
            raise FeatureDisabledError(
                "The capacity forecasting service is closed."
            )
        if not self.config.enabled:
            raise FeatureDisabledError(
                "Capacity forecasting is disabled."
            )

        horizon_hours = (
            request.horizon_hours or self.config.default_horizon_hours
        )
        step_minutes = request.step_minutes or self.config.step_minutes
        forecast_points = math.ceil(horizon_hours * 60 / step_minutes)
        if forecast_points > MAX_FORECAST_POINTS:
            raise InferenceInputAPIError(
                "The requested forecast resolution produces too many points.",
                details={
                    "maximum_forecast_points": MAX_FORECAST_POINTS,
                    "requested_forecast_points": forecast_points,
                },
            )

        result = self._forecasting_service().forecast(
            resource_id=request.resource_id,
            metric=CapacityMetric(request.metric),
            history=(
                TelemetryPoint(
                    timestamp=point.timestamp,
                    value=point.value,
                )
                for point in request.history
            ),
            horizon_hours=horizon_hours,
            step_minutes=step_minutes,
        )
        return CapacityForecastResponse(
            resource_id=result.resource_id,
            metric=CapacityMetricAPI(result.metric.value),
            generated_at=result.generated_at,
            horizon_hours=result.horizon_hours,
            current_value=result.current_value,
            threshold=result.threshold,
            status=CapacityForecastStatusAPI(result.status.value),
            points=[
                CapacityForecastPointResponse(
                    timestamp=point.timestamp,
                    value=point.value,
                    lower_bound=point.lower_bound,
                    upper_bound=point.upper_bound,
                )
                for point in result.points
            ],
            predicted_threshold_time=result.predicted_threshold_time,
            time_to_threshold_hours=result.time_to_threshold_hours,
            confidence=result.confidence,
            model_name=result.model_name,
            diagnostics=dict(result.metadata),
            metadata=APIResponseMetadata(generated_at=result.generated_at),
        )

    def close(self) -> None:
        self._closed = True


def register_capacity_forecasting_service(
    dependencies: Any,
    *,
    replace: bool = False,
) -> CapacityForecastAPIService:
    config = forecasting_config_from_environment()
    service = CapacityForecastAPIService(
        config=config,
        model_registry=dependencies.model_registry,
    )
    dependencies.register_service(
        CAPACITY_FORECAST_SERVICE_NAME,
        service,
        replace=replace,
    )
    return service
