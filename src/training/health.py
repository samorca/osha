from __future__ import annotations

"""
Training subsystem health checks for OSHA.

Accelerator discovery, hardware health, device
selection, CUDA/ROCm detection, and CPU fallback
are delegated to:

    src.accelerators.AcceleratorManager

This module checks whether the training subsystem
is configured and operationally ready.

It does not:

    train models
    evaluate model quality
    promote models
    execute remediation
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
)

from src.accelerators import (
    AcceleratorConfig,
    AcceleratorError,
    AcceleratorHealthStatus,
    AcceleratorManager,
    AcceleratorSelectionConfig,
)
from src.training.config import (
    TrainingConfig,
)
from src.training.errors import (
    CheckpointError,
    ModelArtifactError,
    TrainingSourceError,
)

# ------------------------------------------------------------------
# Training health status
# ------------------------------------------------------------------


class TrainingHealthStatus(
    str,
    Enum,
):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


# ------------------------------------------------------------------
# Component health
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingComponentHealth:
    component: str
    status: str
    message: str

    attributes: Mapping[
        str,
        Any,
    ] = field(default_factory=dict)

    error_type: str | None = None

    def validate(
        self,
    ) -> None:
        if not str(self.component).strip():
            raise ValueError("Training health component cannot be empty.")

        if self.status not in {item.value for item in TrainingHealthStatus}:
            raise ValueError(f"Unknown training health status: {self.status}")

        if not str(self.message).strip():
            raise ValueError("Training health message cannot be empty.")

    def to_dict(
        self,
    ) -> dict[str, Any]:
        self.validate()

        return {
            "component": (self.component),
            "status": self.status,
            "message": (self.message),
            "attributes": dict(self.attributes),
            "error_type": (self.error_type),
        }


# ------------------------------------------------------------------
# Overall training health
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingHealth:
    status: str

    components: tuple[
        TrainingComponentHealth,
        ...,
    ]

    enabled: bool

    model_quality_claimed: bool = False

    promotion_safety_claimed: bool = False

    training_started: bool = False

    def validate(
        self,
    ) -> None:
        if self.status not in {item.value for item in TrainingHealthStatus}:
            raise ValueError(f"Unknown training health status: {self.status}")

        for component in self.components:
            component.validate()

    @property
    def ready(
        self,
    ) -> bool:
        return self.enabled and self.status in {
            TrainingHealthStatus.HEALTHY.value,
            TrainingHealthStatus.DEGRADED.value,
        }

    def to_dict(
        self,
    ) -> dict[str, Any]:
        self.validate()

        return {
            "status": self.status,
            "enabled": (self.enabled),
            "components": [component.to_dict() for component in self.components],
            "component_count": len(self.components),
            "model_quality_claimed": (self.model_quality_claimed),
            "promotion_safety_claimed": (self.promotion_safety_claimed),
            "ready": self.ready,
            "training_started": (self.training_started),
        }


# ------------------------------------------------------------------
# Directory health
# ------------------------------------------------------------------


def directory_health(
    *,
    component: str,
    path: str | Path,
    required: bool,
    writable_required: bool,
) -> TrainingComponentHealth:
    source = Path(path).expanduser().resolve()

    if not source.exists():
        return TrainingComponentHealth(
            component=component,
            status=(
                TrainingHealthStatus.UNHEALTHY.value
                if required
                else TrainingHealthStatus.DEGRADED.value
            ),
            message=("Configured training directory does not exist."),
            attributes={
                "path": str(source),
                "required": (required),
                "writable_required": (writable_required),
            },
        )

    if not source.is_dir():
        return TrainingComponentHealth(
            component=component,
            status=(TrainingHealthStatus.UNHEALTHY.value),
            message=("Configured training path is not a directory."),
            attributes={
                "path": str(source),
            },
        )

    readable = os.access(
        source,
        os.R_OK,
    )

    writable = os.access(
        source,
        os.W_OK,
    )

    if not readable:
        return TrainingComponentHealth(
            component=component,
            status=(TrainingHealthStatus.UNHEALTHY.value),
            message=("Training directory is not readable."),
            attributes={
                "path": str(source),
                "readable": False,
                "writable": (writable),
            },
        )

    if writable_required and not writable:
        return TrainingComponentHealth(
            component=component,
            status=(TrainingHealthStatus.UNHEALTHY.value),
            message=("Training directory is not writable."),
            attributes={
                "path": str(source),
                "readable": (readable),
                "writable": False,
            },
        )

    return TrainingComponentHealth(
        component=component,
        status=(TrainingHealthStatus.HEALTHY.value),
        message=("Training directory is available."),
        attributes={
            "path": str(source),
            "readable": readable,
            "writable": writable,
        },
    )


# ------------------------------------------------------------------
# Build accelerator policy from TrainingConfig
# ------------------------------------------------------------------


def accelerator_config_from_training(
    config: TrainingConfig,
) -> AcceleratorConfig:
    """
    Translate the training-local accelerator
    preference into the shared accelerator layer.

    TrainingConfig remains responsible for
    workload policy.

    src.accelerators remains responsible for
    hardware discovery and selection.
    """

    preferred = str(config.accelerator.preferred).strip().lower()

    allow_cpu_fallback = bool(config.accelerator.allow_fallback_to_cpu)

    require_accelerator = bool(
        not allow_cpu_fallback
        and preferred
        in {
            "cuda",
            "rocm",
        }
    )

    result = AcceleratorConfig(
        selection=(
            AcceleratorSelectionConfig(
                preferred_kind=preferred,
                device_index=(config.accelerator.device_index),
                allow_cpu_fallback=(allow_cpu_fallback),
                allow_degraded_devices=True,
                require_accelerator=(require_accelerator),
                prefer_highest_free_memory=(True),
            )
        ),
        attributes={
            "consumer": ("training.health"),
        },
    )

    result.validate()

    return result


# ------------------------------------------------------------------
# Accelerator health mapping
# ------------------------------------------------------------------


def _map_accelerator_status(
    status: str,
    *,
    preferred: str,
) -> str:
    """
    Translate accelerator-subsystem health into
    training health.

    Optional CUDA/ROCm absence should not make
    training unhealthy when CPU fallback is
    allowed. Selection readiness is checked
    separately.
    """

    if status == (AcceleratorHealthStatus.HEALTHY.value):
        return TrainingHealthStatus.HEALTHY.value

    if status == (AcceleratorHealthStatus.DEGRADED.value):
        return TrainingHealthStatus.DEGRADED.value

    if status == (AcceleratorHealthStatus.UNHEALTHY.value):
        return TrainingHealthStatus.UNHEALTHY.value

    if status == (AcceleratorHealthStatus.UNAVAILABLE.value):
        if preferred == "auto":
            return TrainingHealthStatus.DEGRADED.value

        return TrainingHealthStatus.DEGRADED.value

    return TrainingHealthStatus.UNKNOWN.value


# ------------------------------------------------------------------
# Accelerator subsystem health
# ------------------------------------------------------------------


def accelerator_subsystem_health(
    manager: AcceleratorManager,
    *,
    deep: bool = False,
) -> TrainingComponentHealth:
    preferred = manager.config.selection.preferred_kind

    try:
        report = manager.health(
            deep=deep,
            refresh_inventory=True,
        )

    except AcceleratorError as exc:
        return TrainingComponentHealth(
            component=("accelerator_subsystem"),
            status=(TrainingHealthStatus.UNHEALTHY.value),
            message=("Shared accelerator health check failed."),
            attributes={
                "provider": ("src.accelerators"),
                "preferred": (preferred),
            },
            error_type=(type(exc).__name__),
        )

    return TrainingComponentHealth(
        component=("accelerator_subsystem"),
        status=(
            _map_accelerator_status(
                report.status,
                preferred=preferred,
            )
        ),
        message=("Shared accelerator subsystem health was evaluated."),
        attributes={
            "provider": ("src.accelerators"),
            "preferred": (preferred),
            "accelerator_status": (report.status),
            "accelerator_ready": (report.ready),
            "deep_check": deep,
            "inventory": (
                report.inventory.to_dict() if report.inventory is not None else None
            ),
            "components": [component.to_dict() for component in report.components],
        },
    )


# ------------------------------------------------------------------
# Training device selection health
# ------------------------------------------------------------------


def training_accelerator_health(
    manager: AcceleratorManager,
    *,
    mixed_precision: bool,
) -> TrainingComponentHealth:
    """
    Verify that the same AcceleratorManager policy
    used by training can actually produce a
    training device.

    This does not execute a training workload.
    """

    try:
        selection = manager.select_for_training(
            preferred_kind=(manager.config.selection.preferred_kind),
            device_index=(manager.config.selection.device_index),
            require_accelerator=(manager.config.selection.require_accelerator),
            allow_cpu_fallback=(manager.config.selection.allow_cpu_fallback),
            mixed_precision=(mixed_precision),
            refresh=False,
        )

        runtime_device = manager.runtime_device(selection)

        # This health check should not retain an
        # active selection.
        manager.release(selection)

        return TrainingComponentHealth(
            component="accelerator",
            status=(
                TrainingHealthStatus.HEALTHY.value
                if not selection.fallback_used
                else TrainingHealthStatus.DEGRADED.value
            ),
            message=(
                "A training accelerator was selected successfully."
                if not selection.fallback_used
                else ("Training is available through accelerator fallback.")
            ),
            attributes={
                "provider": ("src.accelerators"),
                "preferred": (selection.requirements.preferred_kind),
                "selected": (selection.device_id),
                "selected_kind": (selection.kind),
                "runtime_device": str(runtime_device),
                "fallback_used": (selection.fallback_used),
                "reason": (selection.reason),
                "visible_devices": len(selection.candidates_considered),
                "mixed_precision": (mixed_precision),
            },
        )

    except AcceleratorError as exc:
        return TrainingComponentHealth(
            component="accelerator",
            status=(TrainingHealthStatus.UNHEALTHY.value),
            message=("No accelerator device satisfies the training requirements."),
            attributes={
                "provider": ("src.accelerators"),
                "preferred": (manager.config.selection.preferred_kind),
                "cpu_fallback": (manager.config.selection.allow_cpu_fallback),
                "mixed_precision": (mixed_precision),
            },
            error_type=(type(exc).__name__),
        )


# ------------------------------------------------------------------
# Aggregate overall status
# ------------------------------------------------------------------


def aggregate_status(
    components: list[TrainingComponentHealth],
) -> str:
    states = {component.status for component in components}

    if TrainingHealthStatus.UNHEALTHY.value in states:
        return TrainingHealthStatus.UNHEALTHY.value

    if TrainingHealthStatus.DEGRADED.value in states:
        return TrainingHealthStatus.DEGRADED.value

    if TrainingHealthStatus.HEALTHY.value in states:
        return TrainingHealthStatus.HEALTHY.value

    return TrainingHealthStatus.UNKNOWN.value


# ------------------------------------------------------------------
# Training health checker
# ------------------------------------------------------------------


class TrainingHealthChecker:
    """
    Aggregate health checker for the shared
    training subsystem.
    """

    def __init__(
        self,
        *,
        config: TrainingConfig | None = None,
        source_manager: Any | None = None,
        checkpoints: Any | None = None,
        artifacts: Any | None = None,
        accelerator_manager: AcceleratorManager | None = None,
    ) -> None:
        self.config = config or TrainingConfig()

        self.source_manager = source_manager

        self.checkpoints = checkpoints

        self.artifacts = artifacts

        accelerator_config = accelerator_config_from_training(self.config)

        self.accelerator_manager = accelerator_manager or AcceleratorManager(
            config=(accelerator_config)
        )

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    def _configuration_health(
        self,
    ) -> TrainingComponentHealth:
        try:
            self.config.validate()

        except ValueError as exc:
            return TrainingComponentHealth(
                component=("configuration"),
                status=(TrainingHealthStatus.UNHEALTHY.value),
                message=("Training configuration is invalid."),
                error_type=(type(exc).__name__),
            )

        return TrainingComponentHealth(
            component="configuration",
            status=(TrainingHealthStatus.HEALTHY.value),
            message=("Training configuration is valid."),
        )

    # ---------------------------------------------------------
    # Directories
    # ---------------------------------------------------------

    def _directory_health(
        self,
    ) -> list[TrainingComponentHealth]:
        paths = self.config.paths.resolved()

        return [
            directory_health(
                component=("prepared_data_directory"),
                path=(paths["prepared_data"]),
                required=False,
                writable_required=False,
            ),
            directory_health(
                component=("orchestration_data_directory"),
                path=(paths["orchestration_data"]),
                required=True,
                writable_required=False,
            ),
            directory_health(
                component=("checkpoint_directory"),
                path=(paths["checkpoints"]),
                required=True,
                writable_required=True,
            ),
            directory_health(
                component=("artifact_directory"),
                path=(paths["artifacts"]),
                required=True,
                writable_required=True,
            ),
            directory_health(
                component=("model_directory"),
                path=paths["models"],
                required=True,
                writable_required=True,
            ),
            directory_health(
                component=("training_log_directory"),
                path=(paths["logs"]),
                required=False,
                writable_required=True,
            ),
        ]

    # ---------------------------------------------------------
    # Training sources
    # ---------------------------------------------------------

    def _source_health(
        self,
    ) -> TrainingComponentHealth:
        if self.source_manager is None:
            return TrainingComponentHealth(
                component=("training_sources"),
                status=(TrainingHealthStatus.DEGRADED.value),
                message=("Training source manager is not attached."),
            )

        try:
            summary = self.source_manager.summary()

        except TrainingSourceError as exc:
            return TrainingComponentHealth(
                component=("training_sources"),
                status=(TrainingHealthStatus.UNHEALTHY.value),
                message=("Training source discovery failed."),
                error_type=(type(exc).__name__),
            )

        source_count = int(
            summary.get(
                "source_count",
                0,
            )
        )

        file_count = int(
            summary.get(
                "file_count",
                0,
            )
        )

        if source_count == 0 or file_count == 0:
            return TrainingComponentHealth(
                component=("training_sources"),
                status=(TrainingHealthStatus.DEGRADED.value),
                message=("No eligible training sources were discovered."),
                attributes=summary,
            )

        return TrainingComponentHealth(
            component=("training_sources"),
            status=(TrainingHealthStatus.HEALTHY.value),
            message=("Eligible training sources were discovered."),
            attributes={
                "source_count": (source_count),
                "file_count": (file_count),
            },
        )

    # ---------------------------------------------------------
    # Checkpoint manager
    # ---------------------------------------------------------

    def _checkpoint_health(
        self,
    ) -> TrainingComponentHealth:
        if self.checkpoints is None:
            return TrainingComponentHealth(
                component="checkpoints",
                status=(TrainingHealthStatus.DEGRADED.value),
                message=("Checkpoint manager is not attached."),
            )

        try:
            summary = self.checkpoints.summary()

        except CheckpointError as exc:
            return TrainingComponentHealth(
                component="checkpoints",
                status=(TrainingHealthStatus.UNHEALTHY.value),
                message=("Checkpoint manager health check failed."),
                error_type=(type(exc).__name__),
            )

        return TrainingComponentHealth(
            component="checkpoints",
            status=(TrainingHealthStatus.HEALTHY.value),
            message=("Checkpoint manager is configured."),
            attributes=summary,
        )

    # ---------------------------------------------------------
    # Artifact manager
    # ---------------------------------------------------------

    def _artifact_health(
        self,
    ) -> TrainingComponentHealth:
        if self.artifacts is None:
            return TrainingComponentHealth(
                component="artifacts",
                status=(TrainingHealthStatus.DEGRADED.value),
                message=("Artifact manager is not attached."),
            )

        try:
            summary = self.artifacts.summary()

        except ModelArtifactError as exc:
            return TrainingComponentHealth(
                component="artifacts",
                status=(TrainingHealthStatus.UNHEALTHY.value),
                message=("Artifact manager health check failed."),
                error_type=(type(exc).__name__),
            )

        return TrainingComponentHealth(
            component="artifacts",
            status=(TrainingHealthStatus.HEALTHY.value),
            message=("Artifact manager is configured."),
            attributes=summary,
        )

    # ---------------------------------------------------------
    # Check
    # ---------------------------------------------------------

    def check(
        self,
        *,
        deep_accelerator_check: bool = False,
    ) -> TrainingHealth:
        if not self.config.enabled:
            return TrainingHealth(
                status=(TrainingHealthStatus.DISABLED.value),
                components=(
                    TrainingComponentHealth(
                        component="training",
                        status=(TrainingHealthStatus.DISABLED.value),
                        message=("Training is disabled."),
                    ),
                ),
                enabled=False,
            )

        components: list[TrainingComponentHealth] = [
            self._configuration_health(),
            *self._directory_health(),
            self._source_health(),
            accelerator_subsystem_health(
                self.accelerator_manager,
                deep=(deep_accelerator_check),
            ),
            training_accelerator_health(
                self.accelerator_manager,
                mixed_precision=(self.config.accelerator.mixed_precision),
            ),
            self._checkpoint_health(),
            self._artifact_health(),
        ]

        result = TrainingHealth(
            status=aggregate_status(components),
            components=tuple(components),
            enabled=True,
            model_quality_claimed=False,
            promotion_safety_claimed=False,
            training_started=False,
        )

        result.validate()

        return result

    # ---------------------------------------------------------
    # Compatibility alias
    # ---------------------------------------------------------

    def health(
        self,
        *,
        deep_accelerator_check: bool = False,
    ) -> TrainingHealth:
        return self.check(deep_accelerator_check=(deep_accelerator_check))

    # ---------------------------------------------------------
    # Ready
    # ---------------------------------------------------------

    def ready(
        self,
        *,
        deep_accelerator_check: bool = False,
    ) -> bool:
        return self.check(deep_accelerator_check=(deep_accelerator_check)).ready

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
        *,
        deep_accelerator_check: bool = False,
    ) -> dict[str, Any]:
        return self.check(deep_accelerator_check=(deep_accelerator_check)).to_dict()


# ------------------------------------------------------------------
# Compatibility name
# ------------------------------------------------------------------

TrainingHealthService = TrainingHealthChecker


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "TrainingComponentHealth",
    "TrainingHealth",
    "TrainingHealthChecker",
    "TrainingHealthService",
    "TrainingHealthStatus",
    "accelerator_config_from_training",
    "accelerator_subsystem_health",
    "aggregate_status",
    "directory_health",
    "training_accelerator_health",
]
