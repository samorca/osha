from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.app import app
from src.api.routes.frameworks import FrameworkAnalysisRequest
from src.api.schema import MAX_CONTENT_LENGTH, MAX_BATCH_SIZE
from src.frameworks import FrameworkApplication
from src.memory.store import create_memory_store


def _record() -> dict:
    return {
        "record_type": "linux_telemetry",
        "provider": "linux",
        "signal_type": "metric",
        "timestamp": "2026-11-26T00:00:00Z",
        "environment_id": "test",
        "host_name": "node-1",
        "payload": {
            "name": "cpu_usage",
            "value": 92.0,
            "unit": "percent",
        },
    }


def test_framework_application_processes_a_record_through_all_stages(
    tmp_path,
) -> None:
    application = FrameworkApplication(
        process_role="test",
        memory_store=create_memory_store(
            backend="json",
            path=tmp_path / "memory.json",
        ),
    )
    try:
        result = application.process_records(
            [_record()],
            environment_id="test",
            correlation_id="test-run",
        )
    finally:
        application.close()

    assert result["status"] == "completed"
    assert result["summary"]["normalized_record_count"] == 1
    assert result["summary"]["orchestration_state"] == "completed"
    run = result["orchestration"]["coordination"]["run"]
    assert run["stop_reason"] == "completed"
    assert result["integrations"]["registry"]["adapter_count"] == 0
    assert run["context"]["dry_run"] is True


def test_framework_application_enforces_record_limit(tmp_path) -> None:
    application = FrameworkApplication(
        process_role="test",
        memory_store=create_memory_store(
            backend="json",
            path=tmp_path / "memory.json",
        ),
        maximum_records=1,
    )
    try:
        with pytest.raises(ValueError, match="1-record limit"):
            application.process_records([_record(), _record()])
    finally:
        application.close()


def test_framework_analysis_request_enforces_batch_and_payload_limits() -> None:
    with pytest.raises(ValidationError):
        FrameworkAnalysisRequest.model_validate(
            {"records": [{}] * (MAX_BATCH_SIZE + 1)}
        )

    with pytest.raises(ValidationError):
        FrameworkAnalysisRequest.model_validate(
            {"records": [{"payload": "x" * MAX_CONTENT_LENGTH}]}
        )


def test_framework_api_routes_are_registered() -> None:
    paths = app.openapi()["paths"]

    assert "get" in paths["/api/v1/frameworks/status"]
    assert "post" in paths["/api/v1/frameworks/analyze"]
