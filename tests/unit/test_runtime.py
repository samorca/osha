from __future__ import annotations

from typing import Any

import pytest

from src.runtime.config import (
    RuntimeConfig,
    ServiceConfig,
)

from src.runtime.runtime import (
    OshaRuntime,
    RuntimeLifecycleError,
    RuntimeSnapshot,
    runtime_from_environment,
)

from src.runtime.schema import (
    ReadinessStatus,
    RuntimeState,
)

from src.runtime.services import (
    RuntimeService,
    RuntimeServiceRegistry,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def make_config(
    *,
    services: dict[
        str,
        ServiceConfig,
    ]
    | None = None,
) -> RuntimeConfig:
    return RuntimeConfig(
        environment_id="test",
        instance_name="osha-test",
        dry_run=True,
        startup_timeout_seconds=5.0,
        shutdown_timeout_seconds=5.0,
        health_interval_seconds=1.0,
        fail_fast=True,
        services=services or {},
    )


def make_runtime(
    *,
    config: RuntimeConfig | None = None,
    registry: RuntimeServiceRegistry | None = None,
) -> OshaRuntime:
    return OshaRuntime(
        config=config or make_config(),
        registry=(
            registry
            or RuntimeServiceRegistry()
        ),
    )


def service_config(
    *,
    required: bool = True,
) -> ServiceConfig:
    return ServiceConfig(
        enabled=True,
        required=required,
        startup_timeout_seconds=5.0,
        shutdown_timeout_seconds=5.0,
        restart_limit=0,
    )


# -------------------------------------------------------------------
# Construction
# -------------------------------------------------------------------


def test_runtime_starts_uncreated() -> None:
    runtime = make_runtime()

    assert runtime.status is None
    assert runtime.started is False
    assert runtime.ready is False


def test_runtime_uses_supplied_config() -> None:
    config = make_config()

    runtime = make_runtime(
        config=config
    )

    assert runtime.config is config

    assert (
        runtime.config.environment_id
        == "test"
    )

    assert (
        runtime.config.instance_name
        == "osha-test"
    )

    assert runtime.config.dry_run is True


def test_runtime_uses_supplied_registry() -> None:
    registry = (
        RuntimeServiceRegistry()
    )

    runtime = make_runtime(
        registry=registry
    )

    assert runtime.registry is registry


# -------------------------------------------------------------------
# Summary before startup
# -------------------------------------------------------------------


def test_summary_before_start() -> None:
    runtime = make_runtime()

    summary = runtime.summary()

    assert (
        summary["instance_name"]
        == "osha-test"
    )

    assert (
        summary["environment_id"]
        == "test"
    )

    assert summary["dry_run"] is True

    assert (
        summary["runtime_created"]
        is False
    )

    assert (
        summary["state"]
        == RuntimeState.CREATED.value
    )

    assert (
        summary["readiness"]
        == ReadinessStatus.UNKNOWN.value
    )

    assert summary["runtime_id"] is None

    assert (
        summary["registered_services"]
        == []
    )

    assert summary["started"] is False
    assert summary["ready"] is False


# -------------------------------------------------------------------
# Snapshot before startup
# -------------------------------------------------------------------


def test_snapshot_before_start_fails() -> None:
    runtime = make_runtime()

    with pytest.raises(
        RuntimeLifecycleError,
        match="Runtime has not been",
    ):
        runtime.snapshot()


# -------------------------------------------------------------------
# Shutdown before startup
# -------------------------------------------------------------------


def test_shutdown_before_start_fails() -> None:
    runtime = make_runtime()

    with pytest.raises(
        RuntimeLifecycleError,
        match="Runtime has not been",
    ):
        runtime.shutdown()


# -------------------------------------------------------------------
# Registration
# -------------------------------------------------------------------


def test_register_service_before_start() -> None:
    config = make_config(
        services={
            "example": service_config(),
        }
    )

    runtime = make_runtime(
        config=config
    )

    service = RuntimeService(
        name="example",
        config=config.service(
            "example"
        ),
    )

    result = runtime.register(
        service
    )

    assert result is service

    assert (
        "example"
        in runtime.registry.names()
    )


def test_register_rebuilds_service_managers() -> None:
    config = make_config(
        services={
            "example": service_config(),
        }
    )

    runtime = make_runtime(
        config=config
    )

    original_services = runtime.services
    original_bootstrap = (
        runtime.bootstrapper
    )
    original_shutdown = (
        runtime.shutdown_manager
    )

    runtime.register(
        RuntimeService(
            name="example",
            config=config.service(
                "example"
            ),
        )
    )

    assert (
        runtime.services
        is not original_services
    )

    assert (
        runtime.bootstrapper
        is not original_bootstrap
    )

    assert (
        runtime.shutdown_manager
        is not original_shutdown
    )


def test_register_after_start_fails() -> None:
    runtime = make_runtime()

    result = runtime.start(
        host_id="test-host",
        started_by="pytest",
    )

    assert result.success is True

    with pytest.raises(
        RuntimeLifecycleError,
        match="cannot be registered",
    ):
        runtime.register(
            RuntimeService(
                name="late-service",
            )
        )

    runtime.shutdown()


# -------------------------------------------------------------------
# Empty runtime lifecycle
# -------------------------------------------------------------------


def test_start_empty_runtime() -> None:
    runtime = make_runtime()

    result = runtime.start(
        host_id="test-host",
        started_by="pytest",
    )

    assert result.success is True

    assert runtime.status is not None

    assert runtime.started is True

    assert (
        runtime.status.state
        in {
            RuntimeState.RUNNING.value,
            RuntimeState.DEGRADED.value,
        }
    )


def test_start_sets_runtime_identity() -> None:
    runtime = make_runtime()

    result = runtime.start(
        host_id="unit-test-host",
        started_by="pytest",
    )

    assert result.success is True

    assert runtime.status is not None

    assert (
        runtime.status.runtime_id
        is not None
    )

    assert (
        runtime.status.context.host_id
        == "unit-test-host"
    )

    assert (
        runtime.status.context.started_by
        == "pytest"
    )


def test_start_runtime_twice_fails() -> None:
    runtime = make_runtime()

    runtime.start(
        host_id="test-host",
        started_by="pytest",
    )

    with pytest.raises(
        RuntimeLifecycleError,
        match="already exists",
    ):
        runtime.start(
            host_id="test-host",
            started_by="pytest",
        )

    runtime.shutdown()


# -------------------------------------------------------------------
# Runtime service lifecycle
# -------------------------------------------------------------------


def test_required_service_starts() -> None:
    events: list[str] = []

    config = make_config(
        services={
            "example": service_config(
                required=True
            ),
        }
    )

    registry = (
        RuntimeServiceRegistry()
    )

    registry.register(
        RuntimeService(
            name="example",
            config=config.service(
                "example"
            ),
            start_hook=lambda: (
                events.append(
                    "start"
                )
            ),
            stop_hook=lambda: (
                events.append(
                    "stop"
                )
            ),
            readiness_probe=lambda: True,
        )
    )

    runtime = make_runtime(
        config=config,
        registry=registry,
    )

    result = runtime.start(
        host_id="test-host",
        started_by="pytest",
    )

    assert result.success is True
    assert events == ["start"]

    shutdown = runtime.shutdown()

    assert shutdown is not None

    assert events == [
        "start",
        "stop",
    ]


def test_service_readiness_updates_runtime() -> None:
    config = make_config(
        services={
            "example": service_config(
                required=True
            ),
        }
    )

    registry = (
        RuntimeServiceRegistry()
    )

    registry.register(
        RuntimeService(
            name="example",
            config=config.service(
                "example"
            ),
            start_hook=lambda: None,
            stop_hook=lambda: None,
            readiness_probe=lambda: True,
        )
    )

    runtime = make_runtime(
        config=config,
        registry=registry,
    )

    result = runtime.start(
        host_id="test-host",
        started_by="pytest",
    )

    assert result.success is True
    assert runtime.ready is True

    assert runtime.status is not None

    assert (
        runtime.status.readiness
        == ReadinessStatus.READY.value
    )

    runtime.shutdown()


# -------------------------------------------------------------------
# Required service failure
# -------------------------------------------------------------------


def test_required_service_start_failure_fails_bootstrap() -> None:
    def fail_start() -> None:
        raise RuntimeError(
            "required startup failure"
        )

    config = make_config(
        services={
            "required-service": (
                service_config(
                    required=True
                )
            ),
        }
    )

    registry = (
        RuntimeServiceRegistry()
    )

    registry.register(
        RuntimeService(
            name="required-service",
            config=config.service(
                "required-service"
            ),
            start_hook=fail_start,
        )
    )

    runtime = make_runtime(
        config=config,
        registry=registry,
    )

    result = runtime.start(
        host_id="test-host",
        started_by="pytest",
    )

    assert result.success is False

    assert runtime.status is not None

    assert (
        runtime.status.state
        == RuntimeState.FAILED.value
    )

    assert runtime.started is False
    assert runtime.ready is False

    assert (
        "required-service"
        in result.failed_service_names
    )


# -------------------------------------------------------------------
# Optional service failure
# -------------------------------------------------------------------


def test_optional_service_failure_degrades_runtime() -> None:
    def fail_start() -> None:
        raise RuntimeError(
            "optional startup failure"
        )

    config = make_config(
        services={
            "optional-service": (
                service_config(
                    required=False
                )
            ),
        }
    )

    registry = (
        RuntimeServiceRegistry()
    )

    registry.register(
        RuntimeService(
            name="optional-service",
            config=config.service(
                "optional-service"
            ),
            start_hook=fail_start,
        )
    )

    runtime = make_runtime(
        config=config,
        registry=registry,
    )

    result = runtime.start(
        host_id="test-host",
        started_by="pytest",
    )

    assert result.success is True
    assert result.degraded is True

    assert runtime.status is not None

    assert (
        runtime.status.state
        == RuntimeState.DEGRADED.value
    )

    assert (
        "optional-service"
        in result.failed_service_names
    )

    runtime.shutdown()


# -------------------------------------------------------------------
# Runtime snapshot
# -------------------------------------------------------------------


def test_snapshot_after_start() -> None:
    runtime = make_runtime()

    runtime.start(
        host_id="test-host",
        started_by="pytest",
    )

    snapshot = runtime.snapshot()

    assert isinstance(
        snapshot,
        RuntimeSnapshot,
    )

    data = snapshot.to_dict()

    assert "status" in data
    assert "config" in data

    assert (
        data["config"][
            "environment_id"
        ]
        == "test"
    )

    assert (
        data["config"][
            "instance_name"
        ]
        == "osha-test"
    )

    runtime.shutdown()


def test_snapshot_config_is_safe_mapping() -> None:
    runtime = make_runtime()

    runtime.start(
        host_id="test-host",
        started_by="pytest",
    )

    snapshot = runtime.snapshot()

    assert isinstance(
        snapshot.config,
        dict,
    )

    assert (
        snapshot.config["dry_run"]
        is True
    )

    runtime.shutdown()


# -------------------------------------------------------------------
# Shutdown
# -------------------------------------------------------------------


def test_shutdown_stops_runtime() -> None:
    runtime = make_runtime()

    runtime.start(
        host_id="test-host",
        started_by="pytest",
    )

    result = runtime.shutdown(
        reason="unit test complete"
    )

    assert result is not None

    assert runtime.status is not None

    assert (
        runtime.status.state
        == RuntimeState.STOPPED.value
    )

    assert runtime.started is False
    assert runtime.ready is False


def test_second_shutdown_fails() -> None:
    runtime = make_runtime()

    runtime.start(
        host_id="test-host",
        started_by="pytest",
    )

    runtime.shutdown()

    with pytest.raises(
        Exception
    ):
        runtime.shutdown()


# -------------------------------------------------------------------
# Summary after startup/shutdown
# -------------------------------------------------------------------


def test_summary_after_start() -> None:
    runtime = make_runtime()

    runtime.start(
        host_id="test-host",
        started_by="pytest",
    )

    summary = runtime.summary()

    assert (
        summary[
            "runtime_created"
        ]
        is True
    )

    assert (
        summary["runtime_id"]
        is not None
    )

    assert summary["started"] is True

    runtime.shutdown()


def test_summary_after_shutdown() -> None:
    runtime = make_runtime()

    runtime.start(
        host_id="test-host",
        started_by="pytest",
    )

    runtime.shutdown()

    summary = runtime.summary()

    assert (
        summary["runtime_created"]
        is True
    )

    assert (
        summary["state"]
        == RuntimeState.STOPPED.value
    )

    assert summary["started"] is False
    assert summary["ready"] is False


# -------------------------------------------------------------------
# Context manager
# -------------------------------------------------------------------


def test_context_manager_starts_and_stops_runtime() -> None:
    runtime = make_runtime()

    with runtime as active_runtime:
        assert (
            active_runtime
            is runtime
        )

        assert runtime.started is True

        assert runtime.status is not None

    assert runtime.status is not None

    assert (
        runtime.status.state
        == RuntimeState.STOPPED.value
    )


def test_context_manager_preserves_original_exception() -> None:
    runtime = make_runtime()

    with pytest.raises(
        ValueError,
        match="test failure",
    ):
        with runtime:
            raise ValueError(
                "test failure"
            )

    assert runtime.status is not None

    assert (
        runtime.status.state
        == RuntimeState.STOPPED.value
    )


# -------------------------------------------------------------------
# Environment-backed runtime
# -------------------------------------------------------------------


def test_runtime_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OSHA_ENVIRONMENT_ID",
        "pytest-env",
    )

    monkeypatch.setenv(
        "OSHA_INSTANCE_NAME",
        "pytest-runtime",
    )

    runtime = runtime_from_environment()

    assert isinstance(
        runtime,
        OshaRuntime,
    )

    assert (
        runtime.config.environment_id
        == "pytest-env"
    )

    assert (
        runtime.config.instance_name
        == "pytest-runtime"
    )


def test_runtime_from_environment_accepts_mapping() -> None:
    environment = {
        "OSHA_ENVIRONMENT_ID": (
            "mapping-env"
        ),

        "OSHA_INSTANCE_NAME": (
            "mapping-runtime"
        ),

        "OSHA_DRY_RUN": "true",
    }

    runtime = runtime_from_environment(
        environ=environment
    )

    assert (
        runtime.config.environment_id
        == "mapping-env"
    )

    assert (
        runtime.config.instance_name
        == "mapping-runtime"
    )

    assert runtime.config.dry_run is True


# -------------------------------------------------------------------
# Explicit registry through environment constructor
# -------------------------------------------------------------------


def test_runtime_from_environment_uses_registry() -> None:
    registry = (
        RuntimeServiceRegistry()
    )

    runtime = runtime_from_environment(
        environ={
            "OSHA_ENVIRONMENT_ID": "test",
            "OSHA_INSTANCE_NAME": "osha-test",
        },
        registry=registry,
    )

    assert runtime.registry is registry


# -------------------------------------------------------------------
# Runtime config validation
# -------------------------------------------------------------------


def test_invalid_runtime_config_is_rejected() -> None:
    config = make_config()

    config.environment_id = ""

    with pytest.raises(
        ValueError,
        match="environment_id",
    ):
        OshaRuntime(
            config=config
        )
