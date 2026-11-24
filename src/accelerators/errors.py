"""
Typed errors for accelerator discovery, selection,
runtime operations, and health checks.
"""

from __future__ import annotations

from itertools import islice
import re

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
)


def _default_error_code(
    error_name: str,
) -> str:
    value = re.sub(
        r"(?<!^)(?=[A-Z])",
        "_",
        error_name,
    ).lower()

    if value.endswith("_error"):
        value = value[:-6]

    return value


class AcceleratorError(RuntimeError):
    """Base exception for expected accelerator failures."""

    default_code = "accelerator_error"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        retryable: bool = False,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.code = code or (
            self.default_code
            if type(self) is AcceleratorError
            else _default_error_code(type(self).__name__)
        )
        self.retryable = bool(retryable)
        self.details = dict(details or {})


class AcceleratorConfigurationError(AcceleratorError):
    pass


class AcceleratorBackendConfigurationError(AcceleratorConfigurationError):
    pass


class AcceleratorBackendError(AcceleratorError):
    pass


class AcceleratorBackendUnavailableError(AcceleratorBackendError):
    pass


class AcceleratorBackendInitializationError(AcceleratorBackendError):
    pass


class AcceleratorBackendUnsupportedError(AcceleratorBackendError):
    pass


class AcceleratorBackendOperationError(AcceleratorBackendError):
    pass


class AcceleratorDiscoveryError(AcceleratorError):
    pass


class AcceleratorDeviceDiscoveryError(AcceleratorDiscoveryError):
    pass


class AcceleratorInventoryError(AcceleratorDiscoveryError):
    pass


class AcceleratorDuplicateDeviceError(AcceleratorInventoryError):
    pass


class AcceleratorDeviceError(AcceleratorError):
    pass


class AcceleratorDeviceNotFoundError(AcceleratorDeviceError):
    pass


class AcceleratorDeviceUnavailableError(AcceleratorDeviceError):
    pass


class AcceleratorDeviceIndexError(AcceleratorDeviceError):
    pass


class AcceleratorDeviceStateError(AcceleratorDeviceError):
    pass


class AcceleratorSelectionError(AcceleratorError):
    pass


class AcceleratorNoMatchingDeviceError(AcceleratorSelectionError):
    pass


class AcceleratorAmbiguousSelectionError(AcceleratorSelectionError):
    pass


class AcceleratorPreferredDeviceUnavailableError(AcceleratorSelectionError):
    pass


class AcceleratorFallbackError(AcceleratorSelectionError):
    pass


class AcceleratorFallbackDisabledError(AcceleratorFallbackError):
    pass


class AcceleratorFallbackUnavailableError(AcceleratorFallbackError):
    pass


class AcceleratorCapabilityError(AcceleratorError):
    pass


class AcceleratorCapabilityMissingError(AcceleratorCapabilityError):
    pass


class AcceleratorTrainingUnsupportedError(AcceleratorCapabilityError):
    pass


class AcceleratorInferenceUnsupportedError(AcceleratorCapabilityError):
    pass


class AcceleratorPrecisionError(AcceleratorError):
    pass


class AcceleratorPrecisionUnsupportedError(AcceleratorPrecisionError):
    pass


class AcceleratorMixedPrecisionUnsupportedError(AcceleratorPrecisionError):
    pass


class AcceleratorMemoryError(AcceleratorError):
    pass


class AcceleratorMemoryQueryError(AcceleratorMemoryError):
    pass


class AcceleratorMemoryUnavailableError(AcceleratorMemoryError):
    pass


class AcceleratorInsufficientMemoryError(AcceleratorMemoryError):
    pass


class AcceleratorAllocationError(AcceleratorMemoryError):
    pass


class AcceleratorOutOfMemoryError(AcceleratorAllocationError):
    pass


class AcceleratorCacheError(AcceleratorError):
    pass


class AcceleratorCacheClearError(AcceleratorCacheError):
    pass


class AcceleratorRuntimeDeviceError(AcceleratorError):
    pass


class AcceleratorRuntimeError(AcceleratorError):
    pass


class AcceleratorSynchronizationError(AcceleratorRuntimeError):
    pass


class AcceleratorRuntimeUnavailableError(AcceleratorRuntimeError):
    pass


class AcceleratorRuntimeInitializationError(AcceleratorRuntimeError):
    pass


class AcceleratorRuntimeOperationError(AcceleratorRuntimeError):
    pass


class AcceleratorHealthError(AcceleratorError):
    pass


class AcceleratorHealthCheckError(AcceleratorHealthError):
    pass


def safe_accelerator_error_details(
    error: BaseException,
    *,
    operation: Optional[str] = None,
    backend: Optional[str] = None,
    device_id: Optional[str] = None,
    runtime_device: Optional[str] = None,
) -> Dict[str, Any]:
    """Return bounded error metadata without raw exception text."""

    details: Dict[str, Any] = {
        "error_type": type(error).__name__,
        "error_module": type(error).__module__,
    }

    for key, value in (
        ("operation", operation),
        ("backend", backend),
        ("device_id", device_id),
        ("runtime_device", runtime_device),
    ):
        if value is not None:
            details[key] = str(value)[:256]

    if isinstance(error, AcceleratorError):
        details["error_code"] = error.code
        details["retryable"] = error.retryable

    return details


def selection_failure_details(
    *,
    preferred_kind: Optional[str] = None,
    device_id: Optional[str] = None,
    device_index: Optional[int] = None,
    candidates: Optional[Mapping[str, Sequence[str]]] = None,
    fallback_allowed: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build bounded, structured diagnostics for a failed selection."""

    details: Dict[str, Any] = {}

    for key, value in (
        ("preferred_kind", preferred_kind),
        ("device_id", device_id),
        ("device_index", device_index),
        ("fallback_allowed", fallback_allowed),
    ):
        if value is not None:
            details[key] = value

    if candidates is not None:
        details["candidate_rejections"] = {
            str(candidate)[:256]: [str(reason)[:256] for reason in reasons[:32]]
            for candidate, reasons in islice(candidates.items(), 64)
        }

    return details


def wrap_backend_error(
    error: BaseException,
    *,
    message: str = "Accelerator backend operation failed.",
    operation: Optional[str] = None,
    backend: Optional[str] = None,
    device_id: Optional[str] = None,
    runtime_device: Optional[str] = None,
    retryable: bool = False,
) -> AcceleratorBackendOperationError:
    """Wrap a provider error without exposing its raw message."""

    return AcceleratorBackendOperationError(
        message,
        retryable=retryable,
        details=safe_accelerator_error_details(
            error,
            operation=operation,
            backend=backend,
            device_id=device_id,
            runtime_device=runtime_device,
        ),
    )


__all__ = [
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
]
