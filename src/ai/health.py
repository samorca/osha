from __future__ import annotations

"""
Health and readiness reporting for OSHA AI serving.

Accelerator discovery, hardware health, device
selection, CUDA/ROCm detection, and CPU fallback
are delegated to:

    src.accelerators.AcceleratorManager

This module checks:

    AI configuration
    runtime directories
    accelerator readiness
    inference-device selection
    promoted-model discovery
    model registry
    enabled model families
    optional deep model loading

It does not:

    train models
    execute inference
    evaluate model quality
    promote artifacts
    perform remediation
"""

import os

from dataclasses import (
    dataclass,
    field,
)

from enum import Enum

from pathlib import Path

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

from src.accelerators import (
    AcceleratorConfig,
    AcceleratorHealthStatus,
    AcceleratorManager,
    AcceleratorPrecision,
    AcceleratorSelectionConfig,
)

from src.ai.config import (
    AIConfig,
)

from src.ai.loader import (
    AIModelLoader,
)

from src.ai.registry import (
    AIModelRegistry,
)

from src.ai.schema import (
    AIModelState,
)


# ------------------------------------------------------------------
# Health status
# ------------------------------------------------------------------


class AIHealthStatus(
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
class AIComponentHealth:
    component: str
    status: str
    message: str

    attributes: Mapping[
        str,
        Any,
    ] = field(default_factory=dict)

    error_type: Optional[str] = None

    def validate(
        self,
    ) -> None:
        if not str(self.component).strip():
            raise ValueError("AI health component cannot be empty.")

        if self.status not in {item.value for item in AIHealthStatus}:
            raise ValueError(f"Unknown AI health status: {self.status}")

        if not str(self.message).strip():
            raise ValueError("AI health message cannot be empty.")

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "component": (self.component),
            "status": self.status,
            "message": (self.message),
            "attributes": dict(self.attributes),
            "error_type": (self.error_type),
        }


# ------------------------------------------------------------------
# Aggregate AI health
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AIHealth:
    status: str

    components: Tuple[
        AIComponentHealth,
        ...,
    ]

    enabled: bool

    deep_check: bool = False

    def validate(
        self,
    ) -> None:
        if self.status not in {item.value for item in AIHealthStatus}:
            raise ValueError(f"Unknown AI health status: {self.status}")

        for component in self.components:
            component.validate()

    @property
    def ready(
        self,
    ) -> bool:
        return bool(
            self.enabled
            and self.status
            in {
                AIHealthStatus.HEALTHY.value,
                AIHealthStatus.DEGRADED.value,
            }
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "status": self.status,
            "enabled": (self.enabled),
            "ready": self.ready,
            "deep_check": (self.deep_check),
            "component_count": len(self.components),
            "components": [component.to_dict() for component in self.components],
            "model_quality_claimed": (False),
            "remediation_readiness_claimed": (False),
        }


# ------------------------------------------------------------------
# Directory health
# ------------------------------------------------------------------


def directory_health(
    *,
    component: str,
    path: str | Path,
    required: bool,
    writable: bool,
) -> AIComponentHealth:
    source = Path(path).expanduser().resolve()

    if not source.exists():
        return AIComponentHealth(
            component=component,
            status=(
                AIHealthStatus.UNHEALTHY.value
                if required
                else AIHealthStatus.DEGRADED.value
            ),
            message=("Configured AI directory does not exist."),
            attributes={
                "path": str(source),
                "required": (required),
                "writable_required": (writable),
            },
        )

    if not source.is_dir():
        return AIComponentHealth(
            component=component,
            status=(AIHealthStatus.UNHEALTHY.value),
            message=("Configured AI path is not a directory."),
            attributes={
                "path": str(source),
            },
        )

    readable = os.access(
        source,
        os.R_OK,
    )

    writable_now = os.access(
        source,
        os.W_OK,
    )

    if not readable:
        return AIComponentHealth(
            component=component,
            status=(AIHealthStatus.UNHEALTHY.value),
            message=("AI directory is not readable."),
            attributes={
                "path": str(source),
                "readable": False,
                "writable": (writable_now),
            },
        )

    if writable and not writable_now:
        return AIComponentHealth(
            component=component,
            status=(AIHealthStatus.UNHEALTHY.value),
            message=("AI directory is not writable."),
            attributes={
                "path": str(source),
                "readable": (readable),
                "writable": False,
            },
        )

    return AIComponentHealth(
        component=component,
        status=(AIHealthStatus.HEALTHY.value),
        message=("AI directory is available."),
        attributes={
            "path": str(source),
            "readable": readable,
            "writable": writable_now,
        },
    )


# ------------------------------------------------------------------
# Translate AI accelerator policy
# ------------------------------------------------------------------


def accelerator_config_from_ai(
    config: AIConfig,
) -> AcceleratorConfig:
    """
    Translate AIConfig device policy into the
    shared accelerator subsystem.

    AIConfig owns inference policy.

    src.accelerators owns hardware discovery and
    device selection.
    """

    config.validate()

    preferred = str(config.device.preferred).strip().lower()

    allow_cpu_fallback = bool(config.device.allow_cpu_fallback)

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
                device_index=(config.device.device_index),
                allow_cpu_fallback=(allow_cpu_fallback),
                allow_degraded_devices=True,
                require_accelerator=(require_accelerator),
                prefer_highest_free_memory=True,
            )
        ),
        attributes={
            "consumer": ("ai.health"),
        },
    )

    result.validate()

    return result


# ------------------------------------------------------------------
# Accelerator health mapping
# ------------------------------------------------------------------


def _map_accelerator_status(
    status: str,
) -> str:
    if status == (AcceleratorHealthStatus.HEALTHY.value):
        return AIHealthStatus.HEALTHY.value

    if status == (AcceleratorHealthStatus.DEGRADED.value):
        return AIHealthStatus.DEGRADED.value

    if status == (AcceleratorHealthStatus.UNHEALTHY.value):
        return AIHealthStatus.UNHEALTHY.value

    if status == (AcceleratorHealthStatus.UNAVAILABLE.value):
        return AIHealthStatus.DEGRADED.value

    return AIHealthStatus.UNKNOWN.value


# ------------------------------------------------------------------
# Accelerator subsystem component
# ------------------------------------------------------------------


def accelerator_subsystem_health(
    manager: AcceleratorManager,
    *,
    deep: bool,
) -> AIComponentHealth:
    """
    Report shared accelerator subsystem health.

    This reports hardware/backend state.

    Actual AI inference-device selection is checked
    separately.
    """

    try:
        report = manager.health(
            deep=deep,
            refresh_inventory=True,
        )

    except Exception as exc:
        return AIComponentHealth(
            component=("accelerator_subsystem"),
            status=(AIHealthStatus.UNHEALTHY.value),
            message=("Shared accelerator health check failed."),
            attributes={
                "provider": ("src.accelerators"),
                "preferred": (manager.config.selection.preferred_kind),
            },
            error_type=(type(exc).__name__),
        )

    return AIComponentHealth(
        component=("accelerator_subsystem"),
        status=(_map_accelerator_status(report.status)),
        message=("Shared accelerator subsystem health was evaluated."),
        attributes={
            "provider": ("src.accelerators"),
            "accelerator_status": (report.status),
            "accelerator_ready": (report.ready),
            "preferred": (manager.config.selection.preferred_kind),
            "deep_check": deep,
            "inventory": (
                report.inventory.to_dict() if report.inventory is not None else None
            ),
            "components": [component.to_dict() for component in report.components],
        },
    )


# ------------------------------------------------------------------
# AI inference-device readiness
# ------------------------------------------------------------------


def inference_accelerator_health(
    manager: AcceleratorManager,
    config: AIConfig,
) -> AIComponentHealth:
    """
    Verify that the same shared accelerator policy
    used by model adapters can select a device for
    AI inference.

    No model or inference workload is executed.
    """

    required_precisions = ()

    if config.device.mixed_precision:
        required_precisions = (AcceleratorPrecision.FLOAT16.value,)

    try:
        selection = manager.select_for_inference(
            preferred_kind=(config.device.preferred),
            device_index=(config.device.device_index),
            require_accelerator=(
                not config.device.allow_cpu_fallback
                and config.device.preferred
                in {
                    "cuda",
                    "rocm",
                }
            ),
            allow_cpu_fallback=(config.device.allow_cpu_fallback),
            mixed_precision=(config.device.mixed_precision),
            required_precisions=(required_precisions),
            refresh=False,
        )

        runtime_device = manager.runtime_device(selection)

        manager.release(selection)

        return AIComponentHealth(
            component="accelerator",
            status=(
                AIHealthStatus.DEGRADED.value
                if selection.fallback_used
                else AIHealthStatus.HEALTHY.value
            ),
            message=(
                "AI inference accelerator was selected successfully."
                if not selection.fallback_used
                else ("AI inference is available through accelerator fallback.")
            ),
            attributes={
                "provider": ("src.accelerators"),
                "preferred": (config.device.preferred),
                "selected": (selection.device_id),
                "selected_kind": (selection.kind),
                "runtime_device": str(runtime_device),
                "fallback_used": (selection.fallback_used),
                "reason": (selection.reason),
                "mixed_precision": (config.device.mixed_precision),
            },
        )

    except Exception as exc:
        return AIComponentHealth(
            component="accelerator",
            status=(AIHealthStatus.UNHEALTHY.value),
            message=("No accelerator device satisfies AI inference requirements."),
            attributes={
                "provider": ("src.accelerators"),
                "preferred": (config.device.preferred),
                "device_index": (config.device.device_index),
                "cpu_fallback": (config.device.allow_cpu_fallback),
                "mixed_precision": (config.device.mixed_precision),
            },
            error_type=(type(exc).__name__),
        )


# ------------------------------------------------------------------
# Aggregate status
# ------------------------------------------------------------------


def aggregate_status(
    components: List[AIComponentHealth],
) -> str:
    states = {component.status for component in components}

    if AIHealthStatus.UNHEALTHY.value in states:
        return AIHealthStatus.UNHEALTHY.value

    if AIHealthStatus.DEGRADED.value in states:
        return AIHealthStatus.DEGRADED.value

    if states == {AIHealthStatus.DISABLED.value}:
        return AIHealthStatus.DISABLED.value

    if AIHealthStatus.HEALTHY.value in states:
        return AIHealthStatus.HEALTHY.value

    return AIHealthStatus.UNKNOWN.value


# ------------------------------------------------------------------
# AI health checker
# ------------------------------------------------------------------


class AIHealthChecker:
    """
    Aggregate readiness checks for OSHA AI
    serving.

    Shallow health:

        configuration
        directories
        accelerator subsystem
        inference-device selection
        model discovery
        registry
        model-family availability

    Deep health additionally attempts to load one
    registered model from every enabled family.
    """

    def __init__(
        self,
        *,
        config: Optional[AIConfig] = None,
        registry: Optional[AIModelRegistry] = None,
        loader: Optional[AIModelLoader] = None,
        accelerator_manager: Optional[AcceleratorManager] = None,
    ) -> None:
        self.config = config or AIConfig()

        self.config.validate()

        self.registry = (
            registry
            if registry is not None
            else AIModelRegistry(config=self.config)
        )

        self.loader = loader or AIModelLoader(
            config=self.config,
            registry=self.registry,
        )

        accelerator_config = accelerator_config_from_ai(self.config)

        self.accelerator_manager = accelerator_manager or AcceleratorManager(
            config=(accelerator_config)
        )

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    def _configuration_health(
        self,
    ) -> AIComponentHealth:
        try:
            self.config.validate()

        except Exception as exc:
            return AIComponentHealth(
                component="configuration",
                status=(AIHealthStatus.UNHEALTHY.value),
                message=("AI serving configuration is invalid."),
                error_type=(type(exc).__name__),
            )

        return AIComponentHealth(
            component="configuration",
            status=(AIHealthStatus.HEALTHY.value),
            message=("AI serving configuration is valid."),
        )

    # ---------------------------------------------------------
    # Filesystem
    # ---------------------------------------------------------

    def _filesystem_health(
        self,
    ) -> List[AIComponentHealth]:
        paths = self.config.paths.resolved()

        return [
            directory_health(
                component=("models_directory"),
                path=paths["models"],
                required=True,
                writable=False,
            ),
            directory_health(
                component=("ai_cache_directory"),
                path=paths["cache"],
                required=False,
                writable=True,
            ),
            directory_health(
                component=("ai_log_directory"),
                path=paths["logs"],
                required=False,
                writable=True,
            ),
        ]

    # ---------------------------------------------------------
    # Model discovery
    # ---------------------------------------------------------

    def _discovery_health(
        self,
    ) -> AIComponentHealth:
        try:
            result = self.loader.discover(fail_fast=False)

        except Exception as exc:
            return AIComponentHealth(
                component=("model_discovery"),
                status=(AIHealthStatus.UNHEALTHY.value),
                message=("Promoted-model discovery failed."),
                error_type=(type(exc).__name__),
            )

        if result.model_count == 0 and result.failure_count == 0:
            return AIComponentHealth(
                component=("model_discovery"),
                status=(AIHealthStatus.DEGRADED.value),
                message=("No promoted AI models were discovered."),
                attributes={
                    "model_count": 0,
                    "failure_count": 0,
                },
            )

        if result.failure_count > 0:
            return AIComponentHealth(
                component=("model_discovery"),
                status=(
                    AIHealthStatus.DEGRADED.value
                    if result.model_count > 0
                    else AIHealthStatus.UNHEALTHY.value
                ),
                message=(
                    "One or more promoted model artifacts failed discovery validation."
                ),
                attributes={
                    "model_count": (result.model_count),
                    "failure_count": (result.failure_count),
                },
            )

        return AIComponentHealth(
            component=("model_discovery"),
            status=(AIHealthStatus.HEALTHY.value),
            message=("Promoted AI models were discovered successfully."),
            attributes={
                "model_count": (result.model_count),
                "failure_count": 0,
            },
        )

    # ---------------------------------------------------------
    # Registry
    # ---------------------------------------------------------

    def _registry_health(
        self,
    ) -> AIComponentHealth:
        try:
            summary = self.registry.summary()

        except Exception as exc:
            return AIComponentHealth(
                component=("model_registry"),
                status=(AIHealthStatus.UNHEALTHY.value),
                message=("AI model registry health check failed."),
                error_type=(type(exc).__name__),
            )

        registered = int(
            summary.get(
                "registered_models",
                0,
            )
        )

        if registered == 0:
            return AIComponentHealth(
                component=("model_registry"),
                status=(AIHealthStatus.DEGRADED.value),
                message=("No AI models are currently registered."),
                attributes=summary,
            )

        return AIComponentHealth(
            component=("model_registry"),
            status=(AIHealthStatus.HEALTHY.value),
            message=("AI model registry is available."),
            attributes=summary,
        )

    # ---------------------------------------------------------
    # Family readiness
    # ---------------------------------------------------------

    def _family_health(
        self,
    ) -> List[AIComponentHealth]:
        output: List[AIComponentHealth] = []

        for family in self.config.allowed_families:
            entries = self.registry.find(family=family)

            if not entries:
                output.append(
                    AIComponentHealth(
                        component=("model_family:" + family),
                        status=(AIHealthStatus.DEGRADED.value),
                        message=(
                            "No registered model is available for the enabled family."
                        ),
                        attributes={
                            "family": family,
                            "registered_models": 0,
                        },
                    )
                )

                continue

            ready_count = sum(1 for entry in entries if entry.ready)

            failed_count = sum(
                1
                for entry in entries
                if (entry.descriptor.state == AIModelState.FAILED.value)
            )

            routable_count = sum(
                1
                for entry in entries
                if (
                    entry.descriptor.state
                    in {
                        AIModelState.REGISTERED.value,
                        AIModelState.READY.value,
                        AIModelState.UNLOADED.value,
                        AIModelState.DEGRADED.value,
                    }
                )
            )

            if routable_count == 0:
                status = AIHealthStatus.UNHEALTHY.value

                message = "No routable model is available for the enabled family."

            elif failed_count > 0:
                status = AIHealthStatus.DEGRADED.value

                message = (
                    "The AI model family "
                    "is available but one "
                    "or more models are "
                    "failed."
                )

            else:
                status = AIHealthStatus.HEALTHY.value

                message = "The AI model family has routable models."

            output.append(
                AIComponentHealth(
                    component=("model_family:" + family),
                    status=status,
                    message=message,
                    attributes={
                        "family": family,
                        "registered_models": len(entries),
                        "routable_models": (routable_count),
                        "ready_models": (ready_count),
                        "failed_models": (failed_count),
                    },
                )
            )

        return output

    # ---------------------------------------------------------
    # Deep model-load readiness
    # ---------------------------------------------------------

    def _deep_model_health(
        self,
    ) -> List[AIComponentHealth]:
        """
        Attempt to load one model from each enabled
        family.

        If the selected model was not previously
        loaded, unload it after the check.

        Model adapters themselves use
        AcceleratorManager, so deep health follows
        the same serving hardware policy as real
        inference.
        """

        output: List[AIComponentHealth] = []

        for family in self.config.allowed_families:
            entries = self.registry.find(family=family)

            routable = [
                entry
                for entry in entries
                if (
                    entry.descriptor.state
                    in {
                        AIModelState.REGISTERED.value,
                        AIModelState.READY.value,
                        AIModelState.UNLOADED.value,
                        AIModelState.DEGRADED.value,
                    }
                )
            ]

            if not routable:
                output.append(
                    AIComponentHealth(
                        component=("runtime_model:" + family),
                        status=(AIHealthStatus.DEGRADED.value),
                        message=(
                            "No registered "
                            "routable model is "
                            "available for deep "
                            "runtime validation."
                        ),
                        attributes={
                            "family": family,
                        },
                    )
                )

                continue

            selected = next(
                (entry for entry in routable if entry.ready),
                routable[0],
            )

            was_loaded = selected.loaded

            try:
                adapter = self.registry.load(selected.model_id)

                output.append(
                    AIComponentHealth(
                        component=("runtime_model:" + family),
                        status=(AIHealthStatus.HEALTHY.value),
                        message=("A model for this family loaded successfully."),
                        attributes={
                            "family": family,
                            "model_id": (selected.model_id),
                            "device": (adapter.descriptor.device),
                            "ready": (adapter.ready),
                            "accelerator_provider": ("src.accelerators"),
                        },
                    )
                )

            except Exception as exc:
                output.append(
                    AIComponentHealth(
                        component=("runtime_model:" + family),
                        status=(AIHealthStatus.UNHEALTHY.value),
                        message=("A registered model failed deep runtime loading."),
                        attributes={
                            "family": family,
                            "model_id": (selected.model_id),
                            "accelerator_provider": ("src.accelerators"),
                        },
                        error_type=(type(exc).__name__),
                    )
                )

            finally:
                if not was_loaded and selected.loaded:
                    try:
                        self.registry.unload(selected.model_id)

                    except Exception:
                        pass

        return output

    # ---------------------------------------------------------
    # Check
    # ---------------------------------------------------------

    def check(
        self,
        *,
        deep: bool = False,
    ) -> AIHealth:
        if not self.config.enabled:
            component = AIComponentHealth(
                component="ai",
                status=(AIHealthStatus.DISABLED.value),
                message=("AI serving is disabled."),
            )

            return AIHealth(
                status=(AIHealthStatus.DISABLED.value),
                components=(component,),
                enabled=False,
                deep_check=deep,
            )

        components: List[AIComponentHealth] = [
            self._configuration_health(),
            *self._filesystem_health(),
            accelerator_subsystem_health(
                self.accelerator_manager,
                deep=deep,
            ),
            inference_accelerator_health(
                self.accelerator_manager,
                self.config,
            ),
            self._discovery_health(),
            self._registry_health(),
            *self._family_health(),
        ]

        if deep:
            components.extend(self._deep_model_health())

        health = AIHealth(
            status=aggregate_status(components),
            components=tuple(components),
            enabled=True,
            deep_check=deep,
        )

        health.validate()

        return health

    # ---------------------------------------------------------
    # Ready
    # ---------------------------------------------------------

    def ready(
        self,
        *,
        deep: bool = False,
    ) -> bool:
        return self.check(deep=deep).ready

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
        *,
        deep: bool = False,
    ) -> Dict[str, Any]:
        result = self.check(deep=deep)

        return {
            **result.to_dict(),
            "accelerator_provider": ("src.accelerators"),
            "inference_executed": (False),
            "training_executed": (False),
            "remediation_executed": (False),
        }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "AIHealthStatus",
    "AIComponentHealth",
    "AIHealth",
    "directory_health",
    "accelerator_config_from_ai",
    "accelerator_subsystem_health",
    "inference_accelerator_health",
    "aggregate_status",
    "AIHealthChecker",
]
