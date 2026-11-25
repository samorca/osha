"""
Core domain types for OSHA capacity forecasting.

These types provide a common contract between telemetry preparation,
forecasting models, capacity analysis, and the forecasting service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


class CapacityMetric(str, Enum):
    """Infrastructure resources that OSHA can forecast."""

    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"


class ForecastStatus(str, Enum):
    """Capacity state determined from the forecast."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class TelemetryPoint:
    """
    One historical utilization observation.

    value represents utilization as a percentage from 0 to 100.
    """

    timestamp: datetime
    value: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "TelemetryPoint timestamp must be timezone-aware"
            )

        if not 0.0 <= self.value <= 100.0:
            raise ValueError(
                "TelemetryPoint value must be between 0 and 100"
            )


@dataclass(frozen=True, slots=True)
class ForecastPoint:
    """
    One predicted utilization point.

    lower_bound and upper_bound can later represent the model's
    prediction interval.
    """

    timestamp: datetime
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "ForecastPoint timestamp must be timezone-aware"
            )

        if not 0.0 <= self.value <= 100.0:
            raise ValueError(
                "ForecastPoint value must be between 0 and 100"
            )

        if self.lower_bound is not None:
            if not 0.0 <= self.lower_bound <= 100.0:
                raise ValueError(
                    "lower_bound must be between 0 and 100"
                )

        if self.upper_bound is not None:
            if not 0.0 <= self.upper_bound <= 100.0:
                raise ValueError(
                    "upper_bound must be between 0 and 100"
                )

        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError(
                "lower_bound cannot be greater than upper_bound"
            )


@dataclass(frozen=True, slots=True)
class CapacityForecast:
    """
    Final forecasting result produced for an infrastructure resource.

    Example:
        resource_id = "node-01:/var"
        metric = CapacityMetric.DISK
        current_value = 78.0
        threshold = 90.0
        time_to_threshold_hours = 11.2
    """

    resource_id: str
    metric: CapacityMetric
    generated_at: datetime

    horizon_hours: float

    current_value: float
    threshold: float

    status: ForecastStatus

    points: Tuple[ForecastPoint, ...] = ()

    predicted_threshold_time: Optional[datetime] = None
    time_to_threshold_hours: Optional[float] = None

    confidence: float = 0.0

    model_name: str = "unknown"

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("resource_id cannot be empty")

        if self.generated_at.tzinfo is None:
            raise ValueError(
                "generated_at must be timezone-aware"
            )

        if self.horizon_hours <= 0:
            raise ValueError(
                "horizon_hours must be greater than zero"
            )

        if not 0.0 <= self.current_value <= 100.0:
            raise ValueError(
                "current_value must be between 0 and 100"
            )

        if not 0.0 <= self.threshold <= 100.0:
            raise ValueError(
                "threshold must be between 0 and 100"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1"
            )

        if self.time_to_threshold_hours is not None:
            if self.time_to_threshold_hours < 0:
                raise ValueError(
                    "time_to_threshold_hours cannot be negative"
                )

    @property
    def exhaustion_predicted(self) -> bool:
        """
        True when OSHA predicts that the configured capacity
        threshold will be reached within the forecasting horizon.
        """

        return self.predicted_threshold_time is not None