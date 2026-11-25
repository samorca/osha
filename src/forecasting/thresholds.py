"""
Capacity threshold policies for OSHA forecasting.

This module defines warning and critical utilization thresholds for
CPU, memory, disk, and network capacity forecasts.

The forecasting model predicts future utilization. This module
interprets that prediction and assigns a capacity status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .types import CapacityMetric, ForecastStatus


@dataclass(frozen=True, slots=True)
class ThresholdBand:
    """
    Warning and critical thresholds for one capacity metric.

    Example:

        warning = 80.0
        critical = 90.0

    means:

        < 80%       -> healthy
        80% to <90% -> warning
        >= 90%      -> critical
    """

    warning: float
    critical: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.warning <= 100.0:
            raise ValueError(
                "warning threshold must be between 0 and 100"
            )

        if not 0.0 <= self.critical <= 100.0:
            raise ValueError(
                "critical threshold must be between 0 and 100"
            )

        if self.warning >= self.critical:
            raise ValueError(
                "warning threshold must be lower than "
                "critical threshold"
            )


DEFAULT_BANDS: Mapping[
    CapacityMetric,
    ThresholdBand,
] = {
    CapacityMetric.CPU: ThresholdBand(
        warning=80.0,
        critical=90.0,
    ),

    CapacityMetric.MEMORY: ThresholdBand(
        warning=80.0,
        critical=90.0,
    ),

    CapacityMetric.DISK: ThresholdBand(
        warning=80.0,
        critical=90.0,
    ),

    CapacityMetric.NETWORK: ThresholdBand(
        warning=80.0,
        critical=95.0,
    ),
}


@dataclass(frozen=True, slots=True)
class ThresholdPolicy:
    """
    Capacity threshold policy used by OSHA forecasting.

    The default policy can later be replaced by thresholds loaded
    from OSHA configuration.
    """

    bands: Mapping[
        CapacityMetric,
        ThresholdBand,
    ] = field(
        default_factory=lambda: dict(
            DEFAULT_BANDS
        )
    )

    def band_for(
        self,
        metric: CapacityMetric,
    ) -> ThresholdBand:
        """
        Return the configured threshold band for a metric.
        """

        try:
            return self.bands[metric]

        except KeyError as exc:
            raise KeyError(
                f"No threshold policy configured for "
                f"{metric.value}"
            ) from exc

    def warning_threshold(
        self,
        metric: CapacityMetric,
    ) -> float:
        """Return the warning threshold for a metric."""

        return self.band_for(
            metric
        ).warning

    def critical_threshold(
        self,
        metric: CapacityMetric,
    ) -> float:
        """Return the critical threshold for a metric."""

        return self.band_for(
            metric
        ).critical

    def status_for(
        self,
        metric: CapacityMetric,
        value: float,
    ) -> ForecastStatus:
        """
        Convert a utilization value into an OSHA forecast status.
        """

        if not 0.0 <= value <= 100.0:
            raise ValueError(
                "utilization value must be between 0 and 100"
            )

        band = self.band_for(
            metric
        )

        if value >= band.critical:
            return ForecastStatus.CRITICAL

        if value >= band.warning:
            return ForecastStatus.WARNING

        return ForecastStatus.HEALTHY

    def is_warning(
        self,
        metric: CapacityMetric,
        value: float,
    ) -> bool:
        """
        Return True when utilization is at least warning level.
        """

        status = self.status_for(
            metric,
            value,
        )

        return status in {
            ForecastStatus.WARNING,
            ForecastStatus.CRITICAL,
        }

    def is_critical(
        self,
        metric: CapacityMetric,
        value: float,
    ) -> bool:
        """
        Return True when utilization has reached critical capacity.
        """

        return (
            self.status_for(
                metric,
                value,
            )
            == ForecastStatus.CRITICAL
        )