from __future__ import annotations

"""
OSHA accelerator subsystem.

Public architecture:

    AcceleratorConfig
          ↓
    AcceleratorDetector
          ↓
    AcceleratorInventory
          ↓
    AcceleratorManager
          ↓
    AcceleratorRequirements
          ↓
    AcceleratorSelection
          ↓
    CPU / CUDA / ROCm runtime device

Most callers should use:

    AcceleratorManager

rather than importing individual backend
implementations directly.
"""

# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

from src.accelerators.schema import (
    ACCELERATOR_SCHEMA_VERSION,

    AcceleratorCapability,
    AcceleratorDevice,
    AcceleratorHealth,
    AcceleratorHealthReport,
    AcceleratorHealthStatus,
    AcceleratorInventory,
    AcceleratorKind,
    AcceleratorMemory,
    AcceleratorPrecision,
    AcceleratorRequirements,
    AcceleratorSelection,
    AcceleratorState,
)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

from src.accelerators.config import (
    CONCRETE_BACKENDS,
    DEFAULT_BACKEND_PRIORITY,
    DEFAULT_ENV_PREFIX,

    AcceleratorConfig,
    AcceleratorDiscoveryConfig,
    AcceleratorHealthConfig,
    AcceleratorPrecisionConfig,
    AcceleratorRuntimeConfig,
    AcceleratorSelectionConfig,

    from_environment as accelerator_config_from_environment,
    from_mapping as accelerator_config_from_mapping,
    safe_config_summary,
)


# ------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------

from src.accelerators.errors import (
    AcceleratorError,

    AcceleratorConfigurationError,
    AcceleratorBackendConfigurationError,

    AcceleratorBackendError,
    AcceleratorBackendUnavailableError,
    AcceleratorBackendInitializationError,
    AcceleratorBackendUnsupportedError,
    AcceleratorBackendOperationError,

    AcceleratorDiscoveryError,
    AcceleratorDeviceDiscoveryError,
    AcceleratorInventoryError,
    AcceleratorDuplicateDeviceError,

    AcceleratorDeviceError,
    AcceleratorDeviceNotFoundError,
    AcceleratorDeviceUnavailableError,
    AcceleratorDeviceIndexError,
    AcceleratorDeviceStateError,

    AcceleratorSelectionError,
    AcceleratorNoMatchingDeviceError,
    AcceleratorAmbiguousSelectionError,
    AcceleratorPreferredDeviceUnavailableError,

    AcceleratorFallbackError,
    AcceleratorFallbackDisabledError,
    AcceleratorFallbackUnavailableError,

    AcceleratorCapabilityError,
    AcceleratorCapabilityMissingError,
    AcceleratorTrainingUnsupportedError,
    AcceleratorInferenceUnsupportedError,

    AcceleratorPrecisionError,
    AcceleratorPrecisionUnsupportedError,
    AcceleratorMixedPrecisionUnsupportedError,

    AcceleratorMemoryError,
    AcceleratorMemoryQueryError,
    AcceleratorMemoryUnavailableError,
    AcceleratorInsufficientMemoryError,
    AcceleratorAllocationError,
    AcceleratorOutOfMemoryError,

    AcceleratorSynchronizationError,

    AcceleratorCacheError,
    AcceleratorCacheClearError,

    AcceleratorRuntimeDeviceError,

    AcceleratorRuntimeError,
    AcceleratorRuntimeUnavailableError,
    AcceleratorRuntimeInitializationError,
    AcceleratorRuntimeOperationError,

    AcceleratorHealthError,
    AcceleratorHealthCheckError,

    safe_accelerator_error_details,
    selection_failure_details,
    wrap_backend_error,
)


# ------------------------------------------------------------------
# Backend contracts
# ------------------------------------------------------------------

from src.accelerators.base import (
    AcceleratorBackend,
    BackendDiscoveryResult,

    CPUAcceleratorBackend,
    CUDAAcceleratorBackend,
    ROCmAcceleratorBackend,
)


# ------------------------------------------------------------------
# Concrete backends
# ------------------------------------------------------------------

from src.accelerators.cpu import (
    CPUAccelerator,
    CPUBackend,
)

from src.accelerators.cuda import (
    CUDAAccelerator,
    CUDABackend,
)

from src.accelerators.rocm import (
    ROCmAccelerator,
    ROCmBackend,
)


# ------------------------------------------------------------------
# Detection
# ------------------------------------------------------------------

from src.accelerators.detector import (
    AcceleratorDetectionResult,
    AcceleratorDetector,
    create_accelerator_detector,
)


# ------------------------------------------------------------------
# Management / selection
# ------------------------------------------------------------------

from src.accelerators.manager import (
    AcceleratorCandidate,
    AcceleratorManager,
    AcceleratorSelectionAttempt,

    create_accelerator_manager,
)


# ------------------------------------------------------------------
# Convenience API
# ------------------------------------------------------------------

def detect_accelerators(
    *,
    config: AcceleratorConfig | None = None,
    fail_fast: bool | None = None,
) -> AcceleratorInventory:
    """
    Detect the accelerator inventory using the
    standard OSHA accelerator subsystem.

    This convenience function performs discovery
    only. It does not select a device.
    """

    detector = AcceleratorDetector(
        config=config
    )

    return detector.detect(
        fail_fast=fail_fast
    ).inventory


def select_accelerator(
    requirements: AcceleratorRequirements | None = None,
    *,
    config: AcceleratorConfig | None = None,
    refresh: bool = True,
    training: bool = False,
    inference: bool = False,
) -> AcceleratorSelection:
    """
    Select one accelerator using the standard
    manager.

    When requirements is omitted, configuration
    defaults plus the training/inference flags are
    used.
    """

    manager = AcceleratorManager(
        config=config
    )

    return manager.select(
        requirements,
        refresh=refresh,
        training=training,
        inference=inference,
    )


def select_training_accelerator(
    *,
    config: AcceleratorConfig | None = None,
    preferred_kind: str | None = None,
    device_index: int | None = None,
    device_id: str | None = None,
    minimum_total_memory_bytes: int | None = None,
    minimum_free_memory_bytes: int | None = None,
    require_accelerator: bool | None = None,
    allow_cpu_fallback: bool | None = None,
    mixed_precision: bool | None = None,
    required_precisions: tuple[str, ...] = (),
    refresh: bool = True,
) -> AcceleratorSelection:
    """
    Convenience selector for model-training
    workloads.
    """

    manager = AcceleratorManager(
        config=config
    )

    return manager.select_for_training(
        preferred_kind=preferred_kind,
        device_index=device_index,
        device_id=device_id,
        minimum_total_memory_bytes=(
            minimum_total_memory_bytes
        ),
        minimum_free_memory_bytes=(
            minimum_free_memory_bytes
        ),
        require_accelerator=(
            require_accelerator
        ),
        allow_cpu_fallback=(
            allow_cpu_fallback
        ),
        mixed_precision=(
            mixed_precision
        ),
        required_precisions=(
            required_precisions
        ),
        refresh=refresh,
    )


def select_inference_accelerator(
    *,
    config: AcceleratorConfig | None = None,
    preferred_kind: str | None = None,
    device_index: int | None = None,
    device_id: str | None = None,
    minimum_total_memory_bytes: int | None = None,
    minimum_free_memory_bytes: int | None = None,
    require_accelerator: bool | None = None,
    allow_cpu_fallback: bool | None = None,
    mixed_precision: bool | None = None,
    required_precisions: tuple[str, ...] = (),
    refresh: bool = True,
) -> AcceleratorSelection:
    """
    Convenience selector for model-inference
    workloads.
    """

    manager = AcceleratorManager(
        config=config
    )

    return manager.select_for_inference(
        preferred_kind=preferred_kind,
        device_index=device_index,
        device_id=device_id,
        minimum_total_memory_bytes=(
            minimum_total_memory_bytes
        ),
        minimum_free_memory_bytes=(
            minimum_free_memory_bytes
        ),
        require_accelerator=(
            require_accelerator
        ),
        allow_cpu_fallback=(
            allow_cpu_fallback
        ),
        mixed_precision=(
            mixed_precision
        ),
        required_precisions=(
            required_precisions
        ),
        refresh=refresh,
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    # Schema
    "ACCELERATOR_SCHEMA_VERSION",

    "AcceleratorKind",
    "AcceleratorState",
    "AcceleratorHealthStatus",
    "AcceleratorPrecision",
    "AcceleratorCapability",

    "AcceleratorMemory",
    "AcceleratorDevice",
    "AcceleratorRequirements",
    "AcceleratorInventory",
    "AcceleratorSelection",

    "AcceleratorHealth",
    "AcceleratorHealthReport",

    # Configuration
    "DEFAULT_ENV_PREFIX",
    "CONCRETE_BACKENDS",
    "DEFAULT_BACKEND_PRIORITY",

    "AcceleratorDiscoveryConfig",
    "AcceleratorSelectionConfig",
    "AcceleratorPrecisionConfig",
    "AcceleratorRuntimeConfig",
    "AcceleratorHealthConfig",
    "AcceleratorConfig",

    "accelerator_config_from_mapping",
    "accelerator_config_from_environment",
    "safe_config_summary",

    # Error hierarchy
    "AcceleratorError",

    "AcceleratorConfigurationError",
    "AcceleratorBackendConfigurationError",

    "AcceleratorBackendError",
    "AcceleratorBackendUnavailableError",
    "AcceleratorBackendInitializationError",
    "AcceleratorBackendUnsupportedError",
    "AcceleratorBackendOperationError",

    "AcceleratorDiscoveryError",
    "AcceleratorDeviceDiscoveryError",
    "AcceleratorInventoryError",
    "AcceleratorDuplicateDeviceError",

    "AcceleratorDeviceError",
    "AcceleratorDeviceNotFoundError",
    "AcceleratorDeviceUnavailableError",
    "AcceleratorDeviceIndexError",
    "AcceleratorDeviceStateError",

    "AcceleratorSelectionError",
    "AcceleratorNoMatchingDeviceError",
    "AcceleratorAmbiguousSelectionError",
    "AcceleratorPreferredDeviceUnavailableError",

    "AcceleratorFallbackError",
    "AcceleratorFallbackDisabledError",
    "AcceleratorFallbackUnavailableError",

    "AcceleratorCapabilityError",
    "AcceleratorCapabilityMissingError",
    "AcceleratorTrainingUnsupportedError",
    "AcceleratorInferenceUnsupportedError",

    "AcceleratorPrecisionError",
    "AcceleratorPrecisionUnsupportedError",
    "AcceleratorMixedPrecisionUnsupportedError",

    "AcceleratorMemoryError",
    "AcceleratorMemoryQueryError",
    "AcceleratorMemoryUnavailableError",
    "AcceleratorInsufficientMemoryError",
    "AcceleratorAllocationError",
    "AcceleratorOutOfMemoryError",

    "AcceleratorSynchronizationError",

    "AcceleratorCacheError",
    "AcceleratorCacheClearError",

    "AcceleratorRuntimeDeviceError",

    "AcceleratorRuntimeError",
    "AcceleratorRuntimeUnavailableError",
    "AcceleratorRuntimeInitializationError",
    "AcceleratorRuntimeOperationError",

    "AcceleratorHealthError",
    "AcceleratorHealthCheckError",

    "safe_accelerator_error_details",
    "selection_failure_details",
    "wrap_backend_error",

    # Backend contracts
    "BackendDiscoveryResult",
    "AcceleratorBackend",

    "CPUAcceleratorBackend",
    "CUDAAcceleratorBackend",
    "ROCmAcceleratorBackend",

    # Concrete backends
    "CPUBackend",
    "CPUAccelerator",

    "CUDABackend",
    "CUDAAccelerator",

    "ROCmBackend",
    "ROCmAccelerator",

    # Detection
    "AcceleratorDetectionResult",
    "AcceleratorDetector",
    "create_accelerator_detector",

    # Management
    "AcceleratorCandidate",
    "AcceleratorSelectionAttempt",
    "AcceleratorManager",
    "create_accelerator_manager",

    # Convenience
    "detect_accelerators",
    "select_accelerator",
    "select_training_accelerator",
    "select_inference_accelerator",
]
