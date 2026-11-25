from __future__ import annotations

import json
import math

from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capacity import clamp_utilization
from .predictor import CapacityPredictor, LinearCapacityPredictor
from .types import CapacityMetric, ForecastPoint, TelemetryPoint


CAPACITY_FORECAST_MODEL_NAME = "osha-capacity-autoregressive"
CAPACITY_FORECAST_MODEL_TYPE = "osha.capacity.autoregressive-ridge"
CAPACITY_FORECAST_SCHEMA_VERSION = "1"


class AutoregressiveCapacityPredictor(CapacityPredictor):
    """CPU, memory, and disk model loaded from a promoted JSON artifact."""

    name = CAPACITY_FORECAST_MODEL_NAME

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("model_type") != CAPACITY_FORECAST_MODEL_TYPE:
            raise ValueError("Unsupported capacity-forecast model type.")
        if payload.get("schema_version") != CAPACITY_FORECAST_SCHEMA_VERSION:
            raise ValueError("Unsupported capacity-forecast model schema.")

        interval = payload.get("interval_seconds")
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ValueError("Model interval_seconds must be a positive integer.")

        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("Model must contain at least one trained metric.")

        self.interval_seconds = interval
        self.metric_models: dict[str, dict[str, Any]] = {}
        for raw_metric, raw_model in metrics.items():
            metric = str(raw_metric)
            if metric not in {
                CapacityMetric.CPU.value,
                CapacityMetric.MEMORY.value,
                CapacityMetric.DISK.value,
                CapacityMetric.NETWORK.value,
            }:
                raise ValueError(f"Unknown trained capacity metric: {metric}")
            if not isinstance(raw_model, Mapping):
                raise ValueError(f"Model parameters for {metric} must be an object.")

            coefficients = raw_model.get("coefficients")
            if (
                not isinstance(coefficients, list)
                or not coefficients
                or len(coefficients) > 96
            ):
                raise ValueError(f"Invalid autoregressive coefficients for {metric}.")
            numeric_coefficients = [
                self._finite_number(value, f"{metric} coefficient")
                for value in coefficients
            ]
            intercept = self._finite_number(
                raw_model.get("intercept"),
                f"{metric} intercept",
            )
            residual_std = self._finite_number(
                raw_model.get("residual_std"),
                f"{metric} residual_std",
            )
            validation_r_squared = self._finite_number(
                raw_model.get("validation_r_squared", 0.0),
                f"{metric} validation_r_squared",
            )
            if residual_std < 0:
                raise ValueError(f"Residual deviation for {metric} cannot be negative.")

            self.metric_models[metric] = {
                **dict(raw_model),
                "coefficients": numeric_coefficients,
                "intercept": intercept,
                "residual_std": residual_std,
                "validation_r_squared": validation_r_squared,
            }

        self.fallback = LinearCapacityPredictor()

    @staticmethod
    def _finite_number(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite.")
        return number

    @classmethod
    def load(cls, path: str | Path) -> AutoregressiveCapacityPredictor:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, Mapping):
            raise ValueError("Capacity-forecast model artifact must be an object.")
        return cls(payload)

    @property
    def trained_metrics(self) -> tuple[str, ...]:
        return tuple(sorted(self.metric_models))

    def _lag_values(
        self,
        history: Sequence[TelemetryPoint],
        *,
        lookback: int,
    ) -> list[float] | None:
        if not history:
            return None

        newest = history[-1].timestamp
        cursor = len(history) - 1
        lag_values: list[float] = []
        maximum_age = self.interval_seconds * 1.5

        for lag in range(lookback):
            target = newest - timedelta(seconds=self.interval_seconds * lag)
            while cursor > 0 and history[cursor].timestamp > target:
                cursor -= 1
            point = history[cursor]
            age = (target - point.timestamp).total_seconds()
            if age < 0 or age > maximum_age:
                return None
            lag_values.append(point.value)

        return lag_values

    def name_for(
        self,
        metric: CapacityMetric,
        history: Sequence[TelemetryPoint],
    ) -> str:
        model = self.metric_models.get(metric.value)
        if model is not None and self._lag_values(
            history,
            lookback=len(model["coefficients"]),
        ) is not None:
            return self.name
        return self.fallback.name

    def trained_for(
        self,
        metric: CapacityMetric,
        history: Sequence[TelemetryPoint],
    ) -> bool:
        return self.name_for(metric, history) == self.name

    def predict(
        self,
        history: Sequence[TelemetryPoint],
        *,
        horizon_hours: float,
        step_minutes: int = 60,
        metric: CapacityMetric | None = None,
    ) -> tuple[tuple[ForecastPoint, ...], float, float]:
        if metric is None:
            return self.fallback.predict(
                history,
                horizon_hours=horizon_hours,
                step_minutes=step_minutes,
            )

        model = self.metric_models.get(metric.value)
        lag_values = (
            self._lag_values(history, lookback=len(model["coefficients"]))
            if model is not None
            else None
        )
        if model is None or lag_values is None:
            return self.fallback.predict(
                history,
                horizon_hours=horizon_hours,
                step_minutes=step_minutes,
                metric=metric,
            )

        if horizon_hours <= 0 or step_minutes <= 0:
            raise ValueError("Forecast horizon and step must be positive.")

        horizon_seconds = horizon_hours * 3600.0
        step_seconds = step_minutes * 60
        interval = self.interval_seconds
        internal_steps = math.ceil(horizon_seconds / interval)
        values = [history[-1].value]
        current_lags = list(lag_values)

        coefficients = model["coefficients"]
        for _ in range(internal_steps):
            prediction = model["intercept"] + sum(
                coefficient * value
                for coefficient, value in zip(
                    coefficients,
                    current_lags,
                    strict=True,
                )
            )
            prediction = clamp_utilization(prediction)
            values.append(prediction)
            current_lags = [prediction, *current_lags[:-1]]

        points: list[ForecastPoint] = []
        point_count = math.ceil(horizon_seconds / step_seconds)
        residual_std = model["residual_std"]
        last_timestamp = history[-1].timestamp

        for point_index in range(1, point_count + 1):
            offset_seconds = min(point_index * step_seconds, horizon_seconds)
            left_index = int(offset_seconds // interval)
            fraction = (offset_seconds - left_index * interval) / interval
            left_value = values[left_index]
            right_value = values[min(left_index + 1, len(values) - 1)]
            value = clamp_utilization(
                left_value + (right_value - left_value) * fraction
            )
            margin = 1.96 * residual_std * math.sqrt(
                max(offset_seconds / interval, 1.0)
            )
            points.append(
                ForecastPoint(
                    timestamp=last_timestamp + timedelta(seconds=offset_seconds),
                    value=value,
                    lower_bound=clamp_utilization(value - margin),
                    upper_bound=clamp_utilization(value + margin),
                )
            )

        slope_per_hour = (
            (values[1] - values[0]) / (interval / 3600.0)
            if len(values) > 1
            else 0.0
        )
        confidence = max(
            0.0,
            min(1.0, model["validation_r_squared"]),
        )
        return tuple(points), slope_per_hour, confidence


__all__ = [
    "CAPACITY_FORECAST_MODEL_NAME",
    "CAPACITY_FORECAST_MODEL_TYPE",
    "CAPACITY_FORECAST_SCHEMA_VERSION",
    "AutoregressiveCapacityPredictor",
]
