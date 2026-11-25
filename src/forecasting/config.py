from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Mapping, Optional

from src.core.validation import boolean, integer, number


@dataclass(frozen=True, slots=True)
class ForecastingConfig:
    enabled: bool = True
    minimum_points: int = 4
    training_window_hours: float = 24.0
    default_horizon_hours: float = 12.0
    step_minutes: int = 60

    def validate(self) -> None:
        integer(self.minimum_points, name="minimum_points", minimum=2)
        number(
            self.training_window_hours,
            name="training_window_hours",
            minimum=0.001,
        )
        number(
            self.default_horizon_hours,
            name="default_horizon_hours",
            minimum=0.001,
            maximum=24.0 * 30,
        )
        integer(self.step_minutes, name="step_minutes", minimum=1)


def from_environment(
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> ForecastingConfig:
    values = os.environ if environ is None else environ

    def value(name: str, default: str) -> str:
        configured = values.get("OSHA_FORECASTING_" + name)
        if configured is None or not configured.strip():
            return default
        return configured

    config = ForecastingConfig(
        enabled=boolean(
            value("ENABLED", "true"),
            name="OSHA_FORECASTING_ENABLED",
        ),
        minimum_points=integer(
            value("MINIMUM_POINTS", "4"),
            name="OSHA_FORECASTING_MINIMUM_POINTS",
            minimum=2,
        ),
        training_window_hours=number(
            value("TRAINING_WINDOW_HOURS", "24"),
            name="OSHA_FORECASTING_TRAINING_WINDOW_HOURS",
            minimum=0.001,
        ),
        default_horizon_hours=number(
            value("DEFAULT_HORIZON_HOURS", "12"),
            name="OSHA_FORECASTING_DEFAULT_HORIZON_HOURS",
            minimum=0.001,
            maximum=24.0 * 30,
        ),
        step_minutes=integer(
            value("STEP_MINUTES", "60"),
            name="OSHA_FORECASTING_STEP_MINUTES",
            minimum=1,
        ),
    )
    config.validate()
    return config
