from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.ai.config import AIConfig, AIPathConfig
from src.ai.loader import AIModelLoader
from src.ai.registry import AIModelRegistry
from src.ai.schema import AIModelFamily
from src.data.checksums import sha256_file
from src.api.errors import InferenceInputAPIError
from src.api.routes.forecasting import (
    get_capacity_forecasting_service,
    router,
)
from src.api.app import app as osha_app
from src.api.schema import (
    CapacityForecastRequest,
    CapacityMetricAPI,
    CapacityTelemetryPointRequest,
)
from src.api.services.forecasting import CapacityForecastAPIService
from src.core.exceptions import ValidationError
from src.forecasting import (
    CapacityForecastingService,
    CapacityMetric,
    ForecastStatus,
    ForecastingConfig,
    LinearCapacityPredictor,
    TelemetryPoint,
)
from src.forecasting.config import from_environment
from src.forecasting.trained import (
    CAPACITY_FORECAST_MODEL_NAME,
    CAPACITY_FORECAST_MODEL_TYPE,
)
from src.training.artifacts import ModelArtifactManager
from src.training.config import TrainingConfig, TrainingPathConfig
from src.training.assembly import default_example_builder
from src.training.errors import TrainingTaskMismatchError
from src.training.schema import (
    CheckpointState,
    EvaluationResult,
    TrainingCheckpoint,
)


def history(values: list[float]) -> list[TelemetryPoint]:
    end = datetime(2026, 11, 25, 21, tzinfo=timezone.utc)
    return [
        TelemetryPoint(
            timestamp=end - timedelta(hours=len(values) - index - 1),
            value=value,
        )
        for index, value in enumerate(values)
    ]


def test_linear_forecast_projects_threshold_crossing() -> None:
    service = CapacityForecastingService(
        predictor=LinearCapacityPredictor(),
        minimum_points=4,
    )

    forecast = service.forecast(
        resource_id="node-01",
        metric=CapacityMetric.MEMORY,
        history=history([60.0, 65.0, 70.0, 75.0]),
        horizon_hours=4,
        step_minutes=60,
    )

    assert forecast.model_name == "linear-capacity-v1"
    assert forecast.status is ForecastStatus.CRITICAL
    assert forecast.time_to_threshold_hours == pytest.approx(3.0)
    assert len(forecast.points) == 4
    assert forecast.points[-1].value == pytest.approx(95.0)


def test_capacity_forecast_reports_insufficient_history() -> None:
    result = CapacityForecastingService().forecast(
        resource_id="node-01",
        metric=CapacityMetric.CPU,
        history=history([24.0, 26.0, 28.0]),
    )

    assert result.status is ForecastStatus.INSUFFICIENT_DATA
    assert result.points == ()
    assert result.confidence == 0.0


def test_forecasting_config_loads_environment_values() -> None:
    config = from_environment(
        environ={
            "OSHA_FORECASTING_ENABLED": "false",
            "OSHA_FORECASTING_MINIMUM_POINTS": "6",
            "OSHA_FORECASTING_TRAINING_WINDOW_HOURS": "48",
            "OSHA_FORECASTING_DEFAULT_HORIZON_HOURS": "8",
            "OSHA_FORECASTING_STEP_MINUTES": "30",
        }
    )

    assert config == ForecastingConfig(
        enabled=False,
        minimum_points=6,
        training_window_hours=48.0,
        default_horizon_hours=8.0,
        step_minutes=30,
    )


def test_forecasting_config_rejects_invalid_minimum_points() -> None:
    with pytest.raises(ValidationError):
        from_environment(
            environ={"OSHA_FORECASTING_MINIMUM_POINTS": "1"}
        )


def test_api_accepts_capacity_forecast_request() -> None:
    service = CapacityForecastAPIService()
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_capacity_forecasting_service] = lambda: service

    payload = {
        "resource_id": "node-01",
        "metric": "memory",
        "history": [
            {
                "timestamp": point.timestamp.isoformat(),
                "value": point.value,
            }
            for point in history([60.0, 65.0, 70.0, 75.0])
        ],
        "horizon_hours": 4,
        "step_minutes": 60,
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/forecasting/capacity", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["metric"] == CapacityMetricAPI.MEMORY.value
    assert result["status"] == ForecastStatus.CRITICAL.value
    assert result["model_name"] == "linear-capacity-v1"
    assert len(result["points"]) == 4


def test_api_rejects_timezone_naive_telemetry() -> None:
    with pytest.raises(ValueError, match="timezone"):
        CapacityTelemetryPointRequest(
            timestamp=datetime(2026, 11, 25, 21),
            value=50,
        )


def test_forecast_api_rejects_unknown_request_fields() -> None:
    with pytest.raises(ValueError):
        CapacityForecastRequest(
            resource_id="node-01",
            metric=CapacityMetricAPI.CPU,
            history=[],
            unrecognized=True,
        )


def test_forecast_api_rejects_excessive_output_resolution() -> None:
    request = CapacityForecastRequest(
        resource_id="node-01",
        metric=CapacityMetricAPI.CPU,
        history=[],
        horizon_hours=24,
        step_minutes=1,
    )

    with pytest.raises(InferenceInputAPIError):
        CapacityForecastAPIService().forecast(request)


def test_capacity_forecast_endpoint_is_in_osha_openapi() -> None:
    assert "/api/v1/forecasting/capacity" in osha_app.openapi()["paths"]


def test_capacity_training_rejects_unordered_generic_examples() -> None:
    with pytest.raises(TrainingTaskMismatchError, match="ordered time-series"):
        default_example_builder(
            {"source_dataset": "alibaba"},
            AIModelFamily.CAPACITY_FORECASTING.value,
        )


def test_promoted_capacity_model_is_loaded_and_used(tmp_path: Path) -> None:
    payload = {
        "model_type": CAPACITY_FORECAST_MODEL_TYPE,
        "schema_version": "1",
        "model_name": CAPACITY_FORECAST_MODEL_NAME,
        "interval_seconds": 300,
        "trained_metrics": ["cpu"],
        "metrics": {
            "cpu": {
                "coefficients": [0.9, 0.05],
                "intercept": 1.0,
                "residual_std": 0.5,
                "validation_r_squared": 0.8,
                "lookback_points": 2,
            }
        },
    }
    paths = TrainingPathConfig(
        checkpoint_directory=str(tmp_path / "checkpoints"),
        artifact_directory=str(tmp_path / "artifacts"),
        model_directory=str(tmp_path / "models"),
    )
    training_config = TrainingConfig(paths=paths)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
    run_id = "forecast-test-run"
    checkpoint = TrainingCheckpoint(
        run_id=run_id,
        model_name=CAPACITY_FORECAST_MODEL_NAME,
        path=str(checkpoint_path),
        state=CheckpointState.BEST.value,
        checksum=sha256_file(checkpoint_path),
    )
    evaluation = EvaluationResult(
        run_id=run_id,
        model_name=CAPACITY_FORECAST_MODEL_NAME,
        split="test",
        passed=True,
    )
    manager = ModelArtifactManager(config=training_config)
    candidate = manager.create_candidate(
        checkpoint=checkpoint,
        task=AIModelFamily.CAPACITY_FORECASTING.value,
        version="1.0.0",
        evaluation=evaluation,
    )
    promoted = manager.promote(candidate, evaluation=evaluation)

    ai_config = AIConfig(
        allowed_families=(AIModelFamily.CAPACITY_FORECASTING.value,),
        paths=AIPathConfig(models_directory=str(tmp_path / "models")),
    )
    registry = AIModelRegistry(config=ai_config)
    loader = AIModelLoader(config=ai_config, registry=registry)
    entry = loader.register_artifact(Path(promoted.path).parent)

    now = datetime(2026, 11, 25, 21, tzinfo=timezone.utc)
    request = CapacityForecastRequest(
        resource_id="node-01",
        metric=CapacityMetricAPI.CPU,
        history=[
            CapacityTelemetryPointRequest(
                timestamp=now - timedelta(minutes=5 * (3 - index)),
                value=50 + index,
            )
            for index in range(4)
        ],
        horizon_hours=0.25,
        step_minutes=5,
    )
    result = CapacityForecastAPIService(model_registry=registry).forecast(request)

    assert registry.load(entry.model_id).ready
    assert result.model_name == CAPACITY_FORECAST_MODEL_NAME
    assert result.diagnostics["trained_model"] is True
    assert len(result.points) == 3

    network_request = request.model_copy(update={"metric": CapacityMetricAPI.NETWORK})
    fallback = CapacityForecastAPIService(model_registry=registry).forecast(
        network_request
    )
    assert fallback.model_name == "linear-capacity-v1"
    assert fallback.diagnostics["trained_model"] is False