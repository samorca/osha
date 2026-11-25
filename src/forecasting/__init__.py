"""
OSHA capacity forecasting subsystem.

This package provides infrastructure capacity forecasting for:

- CPU utilization
- Memory utilization
- Disk utilization
- Network utilization

The public interface exposed here allows other OSHA components to use
forecasting without depending directly on internal implementation
modules.
"""

from .predictor import (
    CapacityPredictor,
    LinearCapacityPredictor,
)

from .config import (
    ForecastingConfig,
)

from .service import (
    CapacityForecastingService,
)

from .trained import (
    AutoregressiveCapacityPredictor,
)

from .thresholds import (
    ThresholdBand,
    ThresholdPolicy,
)

from .types import (
    CapacityForecast,
    CapacityMetric,
    ForecastPoint,
    ForecastStatus,
    TelemetryPoint,
)


__all__ = [
    "CapacityForecast",
    "CapacityForecastingService",
    "CapacityMetric",
    "CapacityPredictor",
    "AutoregressiveCapacityPredictor",
    "ForecastPoint",
    "ForecastingConfig",
    "ForecastStatus",
    "LinearCapacityPredictor",
    "TelemetryPoint",
    "ThresholdBand",
    "ThresholdPolicy",
]