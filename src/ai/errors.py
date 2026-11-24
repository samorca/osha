from __future__ import annotations

"""
Exception hierarchy for the OSHA AI serving
layer.

These exceptions isolate model-runtime,
filesystem, device, routing, and inference
failures from downstream intelligence and API
layers.
"""

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)


# ---------------------------------------------------------------
# Base error
# ---------------------------------------------------------------


class AIError(RuntimeError):
    """
    Base exception for the OSHA AI subsystem.
    """

    default_code = "ai_error"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        retryable: bool = False,
        details: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        super().__init__(message)

        self.message = message

        self.code = code or self.default_code

        self.retryable = bool(retryable)

        self.details = dict(details or {})


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------


class AIConfigurationError(AIError):
    default_code = "ai_configuration_error"


# ---------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------


class AIModelRegistryError(AIError):
    default_code = "ai_model_registry_error"


class AIModelAlreadyRegisteredError(AIModelRegistryError):
    default_code = "ai_model_already_registered"


class AIModelNotRegisteredError(AIModelRegistryError):
    default_code = "ai_model_not_registered"


class AIModelReplacementError(AIModelRegistryError):
    default_code = "ai_model_replacement_error"


# ---------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------


class AIModelDiscoveryError(AIError):
    default_code = "ai_model_discovery_error"


class AIModelArtifactNotFoundError(AIModelDiscoveryError):
    default_code = "ai_model_artifact_not_found"


class AIModelArtifactInvalidError(AIModelDiscoveryError):
    default_code = "ai_model_artifact_invalid"


class AIModelNotPromotedError(AIModelDiscoveryError):
    default_code = "ai_model_not_promoted"


# ---------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------


class AIModelIntegrityError(AIError):
    default_code = "ai_model_integrity_error"


class AIModelChecksumMissingError(AIModelIntegrityError):
    default_code = "ai_model_checksum_missing"


class AIModelChecksumMismatchError(AIModelIntegrityError):
    default_code = "ai_model_checksum_mismatch"


# ---------------------------------------------------------------
# Loading / unloading
# ---------------------------------------------------------------


class AIModelLoadError(AIError):
    default_code = "ai_model_load_error"


class AIModelUnloadError(AIError):
    default_code = "ai_model_unload_error"


class AIModelUnsupportedError(AIModelLoadError):
    default_code = "ai_model_unsupported"


class AIModelConfigurationError(AIModelLoadError):
    default_code = "ai_model_configuration_error"


class AIModelStateError(AIModelLoadError):
    default_code = "ai_model_state_error"


class AIModelCacheError(AIModelLoadError):
    default_code = "ai_model_cache_error"


class AIModelCacheLimitError(AIModelCacheError):
    default_code = "ai_model_cache_limit"


# ---------------------------------------------------------------
# Device / accelerator
# ---------------------------------------------------------------


class AIDeviceError(AIError):
    default_code = "ai_device_error"


class AIDeviceUnavailableError(AIDeviceError):
    default_code = "ai_device_unavailable"


class AIDeviceSelectionError(AIDeviceError):
    default_code = "ai_device_selection_error"


class AIDeviceFallbackError(AIDeviceError):
    default_code = "ai_device_fallback_error"


# ---------------------------------------------------------------
# Routing
# ---------------------------------------------------------------


class AIRoutingError(AIError):
    default_code = "ai_routing_error"


class AIModelRouteNotFoundError(AIRoutingError):
    default_code = "ai_model_route_not_found"


class AIAmbiguousRouteError(AIRoutingError):
    default_code = "ai_ambiguous_route"


class AIFamilyNotAllowedError(AIRoutingError):
    default_code = "ai_family_not_allowed"


class AIVersionNotFoundError(AIRoutingError):
    default_code = "ai_model_version_not_found"


# ---------------------------------------------------------------
# Inference request/input
# ---------------------------------------------------------------


class AIInferenceError(AIError):
    default_code = "ai_inference_error"


class AIInferenceRequestError(AIInferenceError):
    default_code = "ai_inference_request_error"


class AIInferenceInputError(AIInferenceRequestError):
    default_code = "ai_inference_input_error"


class AIInferencePayloadTooLargeError(AIInferenceRequestError):
    default_code = "ai_inference_payload_too_large"


class AIInferenceNonFiniteInputError(AIInferenceInputError):
    default_code = "ai_inference_non_finite_input"


class AIInferenceShapeError(AIInferenceInputError):
    default_code = "ai_inference_shape_error"


# ---------------------------------------------------------------
# Event inference
# ---------------------------------------------------------------


class AIEventInferenceError(AIInferenceError):
    default_code = "ai_event_inference_error"


class AIEventTextMissingError(AIEventInferenceError):
    default_code = "ai_event_text_missing"


class AIVocabularyError(AIEventInferenceError):
    default_code = "ai_event_vocabulary_error"


class AILabelMapError(AIEventInferenceError):
    default_code = "ai_event_label_map_error"


# ---------------------------------------------------------------
# Anomaly inference
# ---------------------------------------------------------------


class AIAnomalyInferenceError(AIInferenceError):
    default_code = "ai_anomaly_inference_error"


class AIAnomalyFeatureError(AIAnomalyInferenceError):
    default_code = "ai_anomaly_feature_error"


class AIAnomalyThresholdError(AIAnomalyInferenceError):
    default_code = "ai_anomaly_threshold_error"


# ---------------------------------------------------------------
# Execution/runtime inference
# ---------------------------------------------------------------


class AIInferenceExecutionError(AIInferenceError):
    default_code = "ai_inference_execution_error"


class AIInferenceTimeoutError(AIInferenceExecutionError):
    default_code = "ai_inference_timeout"

    def __init__(
        self,
        message: str = ("AI inference timed out."),
        *,
        details: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        super().__init__(
            message,
            code=self.default_code,
            retryable=True,
            details=details,
        )


class AIInferenceNumericalError(AIInferenceExecutionError):
    default_code = "ai_inference_numerical_error"


class AIModelUnavailableError(AIInferenceExecutionError):
    default_code = "ai_model_unavailable"

    def __init__(
        self,
        message: str = ("AI model is unavailable."),
        *,
        retryable: bool = True,
        details: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        super().__init__(
            message,
            code=self.default_code,
            retryable=retryable,
            details=details,
        )


# ---------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------


class AIBatchInferenceError(AIInferenceError):
    default_code = "ai_batch_inference_error"


class AIBatchSizeError(AIBatchInferenceError):
    default_code = "ai_batch_size_error"


class AIBatchDisabledError(AIBatchInferenceError):
    default_code = "ai_batch_inference_disabled"


class AIBatchFamilyMismatchError(AIBatchInferenceError):
    default_code = "ai_batch_family_mismatch"


# ---------------------------------------------------------------
# Health
# ---------------------------------------------------------------


class AIHealthError(AIError):
    default_code = "ai_health_error"


# ---------------------------------------------------------------
# Safe underlying-error metadata
# ---------------------------------------------------------------


def safe_ai_error_details(
    error: BaseException,
    *,
    operation: Optional[str] = None,
    model_id: Optional[str] = None,
    family: Optional[str] = None,
    request_id: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Produce bounded metadata from a lower-level
    framework/filesystem/device exception.

    Raw exception messages are deliberately not
    copied because they may expose file paths,
    framework internals, or source data.
    """

    details: Dict[
        str,
        Any,
    ] = {
        "error_type": (type(error).__name__),
        "error_module": (type(error).__module__),
    }

    if operation is not None:
        details["operation"] = str(operation)

    if model_id is not None:
        details["model_id"] = str(model_id)

    if family is not None:
        details["family"] = str(family)

    if request_id is not None:
        details["request_id"] = str(request_id)

    if device is not None:
        details["device"] = str(device)

    return details


# ---------------------------------------------------------------
# Generic AI error wrapper
# ---------------------------------------------------------------


def wrap_ai_error(
    error: BaseException,
    *,
    message: str = ("AI operation failed."),
    code: str = ("ai_operation_failed"),
    retryable: bool = False,
    operation: Optional[str] = None,
    model_id: Optional[str] = None,
    family: Optional[str] = None,
    request_id: Optional[str] = None,
    device: Optional[str] = None,
) -> AIError:
    """
    Wrap a framework/runtime exception without
    leaking its arbitrary raw message.
    """

    return AIError(
        message,
        code=code,
        retryable=retryable,
        details=(
            safe_ai_error_details(
                error,
                operation=operation,
                model_id=model_id,
                family=family,
                request_id=request_id,
                device=device,
            )
        ),
    )


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "AIError",
    "AIConfigurationError",
    "AIModelRegistryError",
    "AIModelAlreadyRegisteredError",
    "AIModelNotRegisteredError",
    "AIModelReplacementError",
    "AIModelDiscoveryError",
    "AIModelArtifactNotFoundError",
    "AIModelArtifactInvalidError",
    "AIModelNotPromotedError",
    "AIModelIntegrityError",
    "AIModelChecksumMissingError",
    "AIModelChecksumMismatchError",
    "AIModelLoadError",
    "AIModelUnloadError",
    "AIModelUnsupportedError",
    "AIModelConfigurationError",
    "AIModelStateError",
    "AIModelCacheError",
    "AIModelCacheLimitError",
    "AIDeviceError",
    "AIDeviceUnavailableError",
    "AIDeviceSelectionError",
    "AIDeviceFallbackError",
    "AIRoutingError",
    "AIModelRouteNotFoundError",
    "AIAmbiguousRouteError",
    "AIFamilyNotAllowedError",
    "AIVersionNotFoundError",
    "AIInferenceError",
    "AIInferenceRequestError",
    "AIInferenceInputError",
    "AIInferencePayloadTooLargeError",
    "AIInferenceNonFiniteInputError",
    "AIInferenceShapeError",
    "AIEventInferenceError",
    "AIEventTextMissingError",
    "AIVocabularyError",
    "AILabelMapError",
    "AIAnomalyInferenceError",
    "AIAnomalyFeatureError",
    "AIAnomalyThresholdError",
    "AIInferenceExecutionError",
    "AIInferenceTimeoutError",
    "AIInferenceNumericalError",
    "AIModelUnavailableError",
    "AIBatchInferenceError",
    "AIBatchSizeError",
    "AIBatchDisabledError",
    "AIBatchFamilyMismatchError",
    "AIHealthError",
    "safe_ai_error_details",
    "wrap_ai_error",
]
