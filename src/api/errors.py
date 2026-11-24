from __future__ import annotations

"""
Production-safe API exception handling for OSHA.

This module provides:

    stable public API error codes
    HTTP status mapping
    retryability metadata
    bounded safe error details
    Pydantic validation-error translation
    internal exception translation
    APIErrorResponse construction

Important security boundary:

    Internal exceptions may contain filesystem
    paths, hostnames, driver messages, tokens,
    infrastructure details, or model internals.

    Those values must not automatically cross the
    public HTTP boundary.

FastAPI exception-handler registration belongs in:

    src/api/app.py

Request ID extraction belongs in:

    src/api/middleware.py
"""

import logging

from enum import Enum

from http import HTTPStatus

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from pydantic import (
    ValidationError,
)

from src.api.schema import (
    APIError,
    APIErrorField,
    APIErrorResponse,
    APIResponseMetadata,
    APIStatus,
)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Limits
# ------------------------------------------------------------------

MAX_ERROR_CODE_LENGTH = 128

MAX_PUBLIC_MESSAGE_LENGTH = 2048

MAX_ERROR_DETAIL_COUNT = 64

MAX_ERROR_FIELD_COUNT = 128

MAX_ERROR_LOCATION_PARTS = 32

MAX_ERROR_LOCATION_PART_LENGTH = 256

MAX_SAFE_DETAIL_TEXT_LENGTH = 1024


# ------------------------------------------------------------------
# Public API error codes
# ------------------------------------------------------------------


class APIErrorCode(
    str,
    Enum,
):
    # Generic
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    FEATURE_DISABLED = "feature_disabled"
    NOT_IMPLEMENTED = "not_implemented"

    # Request
    INVALID_REQUEST = "invalid_request"
    VALIDATION_ERROR = "validation_error"
    BAD_JSON = "bad_json"
    REQUEST_TOO_LARGE = "request_too_large"
    REQUEST_TIMEOUT = "request_timeout"
    RATE_LIMITED = "rate_limited"
    CONCURRENCY_LIMIT = "concurrency_limit"

    # Authentication / authorization
    AUTHENTICATION_REQUIRED = "authentication_required"
    INVALID_CREDENTIALS = "invalid_credentials"
    PERMISSION_DENIED = "permission_denied"
    FORBIDDEN = "forbidden"

    # Resource
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    ALREADY_EXISTS = "already_exists"
    PRECONDITION_FAILED = "precondition_failed"
    VERSION_CONFLICT = "version_conflict"

    # Accelerator
    ACCELERATOR_ERROR = "accelerator_error"
    ACCELERATOR_UNAVAILABLE = "accelerator_unavailable"
    ACCELERATOR_NOT_FOUND = "accelerator_not_found"
    ACCELERATOR_SELECTION_FAILED = "accelerator_selection_failed"
    ACCELERATOR_MEMORY_INSUFFICIENT = "accelerator_memory_insufficient"
    ACCELERATOR_PRECISION_UNSUPPORTED = "accelerator_precision_unsupported"
    ACCELERATOR_CAPABILITY_MISSING = "accelerator_capability_missing"
    ACCELERATOR_OPERATION_FAILED = "accelerator_operation_failed"
    ACCELERATOR_METRICS_UNAVAILABLE = "accelerator_metrics_unavailable"
    ACCELERATOR_OPTIMIZATION_FAILED = "accelerator_optimization_failed"

    # AI / inference
    AI_ERROR = "ai_error"
    INFERENCE_FAILED = "inference_failed"
    INFERENCE_INPUT_INVALID = "inference_input_invalid"
    INFERENCE_TIMEOUT = "inference_timeout"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_LOAD_FAILED = "model_load_failed"
    MODEL_OPERATION_FAILED = "model_operation_failed"
    MODEL_CONFIGURATION_INVALID = "model_configuration_invalid"

    # Incident
    INCIDENT_NOT_FOUND = "incident_not_found"
    INCIDENT_CONFLICT = "incident_conflict"
    INCIDENT_OPERATION_FAILED = "incident_operation_failed"

    # Decision
    DECISION_NOT_FOUND = "decision_not_found"
    DECISION_DENIED = "decision_denied"
    DECISION_EXPIRED = "decision_expired"
    DECISION_EVALUATION_FAILED = "decision_evaluation_failed"

    # Remediation
    REMEDIATION_NOT_FOUND = "remediation_not_found"
    REMEDIATION_NOT_APPROVED = "remediation_not_approved"
    REMEDIATION_CONFIRMATION_REQUIRED = "remediation_confirmation_required"
    REMEDIATION_VERSION_CONFLICT = "remediation_version_conflict"
    REMEDIATION_EXECUTION_DISABLED = "remediation_execution_disabled"
    REMEDIATION_EXECUTION_FAILED = "remediation_execution_failed"
    REMEDIATION_ROLLBACK_FAILED = "remediation_rollback_failed"


# ------------------------------------------------------------------
# Safe values
# ------------------------------------------------------------------

_SAFE_SCALAR_TYPES = (
    str,
    int,
    float,
    bool,
    type(None),
)


_SAFE_DETAIL_KEYS = {
    "operation",
    "resource",
    "resource_id",
    "model_id",
    "model_name",
    "model_version",
    "family",
    "device_id",
    "device_index",
    "kind",
    "backend",
    "preferred_kind",
    "runtime_device",
    "request_id",
    "incident_id",
    "decision_id",
    "plan_id",
    "step_id",
    "provider",
    "expected_version",
    "actual_version",
    "required_capability",
    "required_precision",
    "minimum_memory_bytes",
    "available_memory_bytes",
    "retry_after_seconds",
    "limit",
    "current",
    "state",
    "status",
    "fallback_allowed",
    "error_type",
}


_SENSITIVE_KEY_PARTS = {
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "authorization",
    "cookie",
    "key",
    "private",
    "certificate",
    "connection_string",
    "dsn",
    "path",
    "traceback",
    "stack",
    "environment",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _clean_code(
    value: Any,
) -> str:
    code = str(value).strip().lower()

    if not code:
        return APIErrorCode.INTERNAL_ERROR.value

    normalized = []

    for character in code:
        if character.isalnum() or character in {
            "_",
            "-",
            ".",
        }:
            normalized.append(character)

        else:
            normalized.append("_")

    result = "".join(normalized)[:MAX_ERROR_CODE_LENGTH]

    return result or APIErrorCode.INTERNAL_ERROR.value


def _clean_message(
    value: Any,
    *,
    fallback: str,
) -> str:
    message = str(value if value is not None else fallback).strip()

    if not message:
        message = fallback

    return message[:MAX_PUBLIC_MESSAGE_LENGTH]


def _key_is_sensitive(
    key: str,
) -> bool:
    normalized = str(key).strip().lower()

    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _safe_scalar(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if value != value:
            return None

        if value in {
            float("inf"),
            float("-inf"),
        }:
            return None

        return value

    if isinstance(
        value,
        str,
    ):
        return value[:MAX_SAFE_DETAIL_TEXT_LENGTH]

    return str(type(value).__name__)


def safe_error_details(
    details: Optional[
        Mapping[
            str,
            Any,
        ]
    ],
    *,
    allow_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Return bounded, public-safe error metadata.

    Unknown keys are excluded by default.

    Nested arbitrary objects are not serialized.
    """

    if not details:
        return {}

    allowed = set(allow_keys or _SAFE_DETAIL_KEYS)

    output: Dict[
        str,
        Any,
    ] = {}

    for raw_key, value in details.items():
        if len(output) >= MAX_ERROR_DETAIL_COUNT:
            break

        key = str(raw_key).strip()

        if not key or key not in allowed or _key_is_sensitive(key):
            continue

        if isinstance(
            value,
            _SAFE_SCALAR_TYPES,
        ):
            output[key] = _safe_scalar(value)

            continue

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            safe_items = []

            for item in list(value)[:64]:
                if isinstance(
                    item,
                    _SAFE_SCALAR_TYPES,
                ):
                    safe_items.append(_safe_scalar(item))

            output[key] = safe_items

            continue

        if isinstance(
            value,
            Mapping,
        ):
            nested: Dict[
                str,
                Any,
            ] = {}

            for nested_key, nested_value in list(value.items())[:32]:
                safe_key = str(nested_key).strip()

                if not safe_key or _key_is_sensitive(safe_key):
                    continue

                if isinstance(
                    nested_value,
                    _SAFE_SCALAR_TYPES,
                ):
                    nested[safe_key[:MAX_ERROR_LOCATION_PART_LENGTH]] = _safe_scalar(
                        nested_value
                    )

            output[key] = nested

    return output


def exception_type_name(
    error: BaseException,
) -> str:
    return type(error).__name__


def exception_module_name(
    error: BaseException,
) -> str:
    return type(error).__module__


# ------------------------------------------------------------------
# Base API exception
# ------------------------------------------------------------------


class APIException(Exception):
    """
    Base exception intentionally safe to expose
    through the public API.

    public_message must not contain secrets,
    internal paths, stack traces, or raw backend
    error text.
    """

    default_code = APIErrorCode.INTERNAL_ERROR.value

    default_status_code = int(HTTPStatus.INTERNAL_SERVER_ERROR)

    default_message = "An internal service error occurred."

    default_retryable = False

    def __init__(
        self,
        public_message: Optional[str] = None,
        *,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
        retryable: Optional[bool] = None,
        details: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        fields: Optional[Sequence[APIErrorField]] = None,
        headers: Optional[
            Mapping[
                str,
                str,
            ]
        ] = None,
        log_level: int = logging.WARNING,
    ) -> None:
        self.code = _clean_code(code or self.default_code)

        self.status_code = int(
            status_code if status_code is not None else self.default_status_code
        )

        if self.status_code < 400 or self.status_code > 599:
            raise ValueError("API exception status code must be between 400 and 599")

        self.public_message = _clean_message(
            public_message,
            fallback=(self.default_message),
        )

        self.retryable = bool(
            self.default_retryable if retryable is None else retryable
        )

        self.details = safe_error_details(details)

        self.fields = tuple(fields or ())[:MAX_ERROR_FIELD_COUNT]

        self.headers = {
            str(key): str(value)
            for key, value in dict(headers or {}).items()
            if str(key).strip()
        }

        self.log_level = int(log_level)

        super().__init__(self.public_message)

    def to_api_error(
        self,
    ) -> APIError:
        return APIError(
            code=self.code,
            message=(self.public_message),
            retryable=(self.retryable),
            details=dict(self.details),
            fields=list(self.fields),
        )

    def to_response(
        self,
        *,
        request_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> APIErrorResponse:
        return APIErrorResponse(
            status=(APIStatus.ERROR.value),
            error=(self.to_api_error()),
            metadata=(
                APIResponseMetadata(
                    request_id=request_id,
                    duration_ms=duration_ms,
                )
            ),
        )


# ------------------------------------------------------------------
# Generic API exceptions
# ------------------------------------------------------------------


class BadRequestError(APIException):
    default_code = APIErrorCode.INVALID_REQUEST.value

    default_status_code = int(HTTPStatus.BAD_REQUEST)

    default_message = "The request is invalid."


class RequestValidationAPIError(APIException):
    default_code = APIErrorCode.VALIDATION_ERROR.value

    default_status_code = int(HTTPStatus.UNPROCESSABLE_ENTITY)

    default_message = "Request validation failed."


class AuthenticationRequiredError(APIException):
    default_code = APIErrorCode.AUTHENTICATION_REQUIRED.value

    default_status_code = int(HTTPStatus.UNAUTHORIZED)

    default_message = "Authentication is required."


class PermissionDeniedError(APIException):
    default_code = APIErrorCode.PERMISSION_DENIED.value

    default_status_code = int(HTTPStatus.FORBIDDEN)

    default_message = "Permission is denied."


class ResourceNotFoundError(APIException):
    default_code = APIErrorCode.NOT_FOUND.value

    default_status_code = int(HTTPStatus.NOT_FOUND)

    default_message = "The requested resource was not found."


class ResourceConflictError(APIException):
    default_code = APIErrorCode.CONFLICT.value

    default_status_code = int(HTTPStatus.CONFLICT)

    default_message = "The request conflicts with the current resource state."


class PreconditionFailedError(APIException):
    default_code = APIErrorCode.PRECONDITION_FAILED.value

    default_status_code = int(HTTPStatus.PRECONDITION_FAILED)

    default_message = "A request precondition failed."


class FeatureDisabledError(APIException):
    default_code = APIErrorCode.FEATURE_DISABLED.value

    default_status_code = int(HTTPStatus.NOT_FOUND)

    default_message = "The requested API feature is disabled."


class RequestTooLargeError(APIException):
    default_code = APIErrorCode.REQUEST_TOO_LARGE.value

    default_status_code = int(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)

    default_message = "The request body is too large."


class RequestTimeoutAPIError(APIException):
    default_code = APIErrorCode.REQUEST_TIMEOUT.value

    default_status_code = int(HTTPStatus.GATEWAY_TIMEOUT)

    default_message = "The request timed out."

    default_retryable = True


class RateLimitExceededError(APIException):
    default_code = APIErrorCode.RATE_LIMITED.value

    default_status_code = int(HTTPStatus.TOO_MANY_REQUESTS)

    default_message = "The request rate limit was exceeded."

    default_retryable = True


class ConcurrencyLimitError(APIException):
    default_code = APIErrorCode.CONCURRENCY_LIMIT.value

    default_status_code = int(HTTPStatus.SERVICE_UNAVAILABLE)

    default_message = "The service is currently at its concurrency limit."

    default_retryable = True


class ServiceUnavailableAPIError(APIException):
    default_code = APIErrorCode.SERVICE_UNAVAILABLE.value

    default_status_code = int(HTTPStatus.SERVICE_UNAVAILABLE)

    default_message = "The service is temporarily unavailable."

    default_retryable = True


class InternalServerAPIError(APIException):
    default_code = APIErrorCode.INTERNAL_ERROR.value

    default_status_code = int(HTTPStatus.INTERNAL_SERVER_ERROR)

    default_message = "An internal service error occurred."

    default_retryable = False

    def __init__(
        self,
        public_message: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault(
            "log_level",
            logging.ERROR,
        )

        super().__init__(
            public_message,
            **kwargs,
        )


# ------------------------------------------------------------------
# Accelerator API exceptions
# ------------------------------------------------------------------


class AcceleratorAPIError(APIException):
    default_code = APIErrorCode.ACCELERATOR_ERROR.value

    default_status_code = int(HTTPStatus.INTERNAL_SERVER_ERROR)

    default_message = "The accelerator operation failed."


class AcceleratorUnavailableAPIError(AcceleratorAPIError):
    default_code = APIErrorCode.ACCELERATOR_UNAVAILABLE.value

    default_status_code = int(HTTPStatus.SERVICE_UNAVAILABLE)

    default_message = "No accelerator device satisfies the request."

    default_retryable = True


class AcceleratorNotFoundAPIError(AcceleratorAPIError):
    default_code = APIErrorCode.ACCELERATOR_NOT_FOUND.value

    default_status_code = int(HTTPStatus.NOT_FOUND)

    default_message = "The requested accelerator device was not found."


class AcceleratorMemoryAPIError(AcceleratorAPIError):
    default_code = APIErrorCode.ACCELERATOR_MEMORY_INSUFFICIENT.value

    default_status_code = int(HTTPStatus.CONFLICT)

    default_message = "The accelerator does not have sufficient available memory."

    default_retryable = True


class AcceleratorOptimizationAPIError(AcceleratorAPIError):
    default_code = APIErrorCode.ACCELERATOR_OPTIMIZATION_FAILED.value

    default_status_code = int(HTTPStatus.UNPROCESSABLE_ENTITY)

    default_message = "An accelerator optimization plan could not be generated."


# ------------------------------------------------------------------
# AI and model API exceptions
# ------------------------------------------------------------------


class AIAPIError(APIException):
    default_code = APIErrorCode.AI_ERROR.value

    default_status_code = int(HTTPStatus.INTERNAL_SERVER_ERROR)

    default_message = "The AI operation failed."


class InferenceAPIError(AIAPIError):
    default_code = APIErrorCode.INFERENCE_FAILED.value

    default_status_code = int(HTTPStatus.INTERNAL_SERVER_ERROR)

    default_message = "Inference execution failed."


class InferenceInputAPIError(InferenceAPIError):
    default_code = APIErrorCode.INFERENCE_INPUT_INVALID.value

    default_status_code = int(HTTPStatus.UNPROCESSABLE_ENTITY)

    default_message = "The inference input is invalid."


class InferenceTimeoutAPIError(InferenceAPIError):
    default_code = APIErrorCode.INFERENCE_TIMEOUT.value

    default_status_code = int(HTTPStatus.GATEWAY_TIMEOUT)

    default_message = "Inference execution timed out."

    default_retryable = True


class ModelNotFoundAPIError(AIAPIError):
    default_code = APIErrorCode.MODEL_NOT_FOUND.value

    default_status_code = int(HTTPStatus.NOT_FOUND)

    default_message = "The requested model was not found."


class ModelUnavailableAPIError(AIAPIError):
    default_code = APIErrorCode.MODEL_UNAVAILABLE.value

    default_status_code = int(HTTPStatus.SERVICE_UNAVAILABLE)

    default_message = "The requested model is unavailable."

    default_retryable = True


class ModelLoadAPIError(AIAPIError):
    default_code = APIErrorCode.MODEL_LOAD_FAILED.value

    default_status_code = int(HTTPStatus.INTERNAL_SERVER_ERROR)

    default_message = "The model could not be loaded."


# ------------------------------------------------------------------
# Incident, decision, remediation exceptions
# ------------------------------------------------------------------


class IncidentNotFoundAPIError(ResourceNotFoundError):
    default_code = APIErrorCode.INCIDENT_NOT_FOUND.value

    default_message = "The requested incident was not found."


class DecisionNotFoundAPIError(ResourceNotFoundError):
    default_code = APIErrorCode.DECISION_NOT_FOUND.value

    default_message = "The requested decision was not found."


class DecisionDeniedAPIError(APIException):
    default_code = APIErrorCode.DECISION_DENIED.value

    default_status_code = int(HTTPStatus.FORBIDDEN)

    default_message = "The requested action was denied by policy."


class RemediationNotFoundAPIError(ResourceNotFoundError):
    default_code = APIErrorCode.REMEDIATION_NOT_FOUND.value

    default_message = "The requested remediation plan was not found."


class RemediationNotApprovedAPIError(APIException):
    default_code = APIErrorCode.REMEDIATION_NOT_APPROVED.value

    default_status_code = int(HTTPStatus.CONFLICT)

    default_message = "The remediation plan is not approved."


class RemediationConfirmationRequiredAPIError(APIException):
    default_code = APIErrorCode.REMEDIATION_CONFIRMATION_REQUIRED.value

    default_status_code = int(HTTPStatus.BAD_REQUEST)

    default_message = "Explicit remediation confirmation is required."


class RemediationVersionConflictAPIError(ResourceConflictError):
    default_code = APIErrorCode.REMEDIATION_VERSION_CONFLICT.value

    default_message = (
        "The remediation plan version does not match the expected version."
    )


class RemediationExecutionDisabledAPIError(FeatureDisabledError):
    default_code = APIErrorCode.REMEDIATION_EXECUTION_DISABLED.value

    default_message = "Remediation execution is disabled."


class RemediationExecutionAPIError(APIException):
    default_code = APIErrorCode.REMEDIATION_EXECUTION_FAILED.value

    default_status_code = int(HTTPStatus.INTERNAL_SERVER_ERROR)

    default_message = "Remediation execution failed."


# ------------------------------------------------------------------
# Pydantic validation translation
# ------------------------------------------------------------------


def validation_error_fields(
    error: ValidationError,
) -> Tuple[
    APIErrorField,
    ...,
]:
    """
    Translate Pydantic validation errors without
    exposing raw input values.
    """

    output: List[APIErrorField] = []

    for item in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )[:MAX_ERROR_FIELD_COUNT]:
        location: list[str | int] = []

        for part in item.get(
            "loc",
            (),
        )[:MAX_ERROR_LOCATION_PARTS]:
            if isinstance(
                part,
                int,
            ):
                location.append(part)

            else:
                location.append(str(part)[:MAX_ERROR_LOCATION_PART_LENGTH])

        raw_type = item.get(
            "type",
            "validation_error",
        )

        message = item.get(
            "msg",
            "Invalid value.",
        )

        output.append(
            APIErrorField(
                location=location,
                code=_clean_code(raw_type),
                message=_clean_message(
                    message,
                    fallback=("Invalid value."),
                ),
            )
        )

    return tuple(output)


def from_validation_error(
    error: ValidationError,
) -> RequestValidationAPIError:
    return RequestValidationAPIError(
        fields=validation_error_fields(error),
        details={
            "error_type": (exception_type_name(error)),
        },
    )


# ------------------------------------------------------------------
# Internal exception translation
# ------------------------------------------------------------------

_ACCELERATOR_NOT_FOUND_TYPES = {
    "AcceleratorDeviceNotFoundError",
}


_ACCELERATOR_UNAVAILABLE_TYPES = {
    "AcceleratorBackendUnavailableError",
    "AcceleratorDeviceUnavailableError",
    "AcceleratorNoMatchingDeviceError",
    "AcceleratorPreferredDeviceUnavailableError",
    "AcceleratorFallbackDisabledError",
    "AcceleratorFallbackUnavailableError",
}


_ACCELERATOR_MEMORY_TYPES = {
    "AcceleratorInsufficientMemoryError",
    "AcceleratorMemoryUnavailableError",
    "AcceleratorOutOfMemoryError",
    "AcceleratorAllocationError",
}


_ACCELERATOR_PRECISION_TYPES = {
    "AcceleratorPrecisionUnsupportedError",
    "AcceleratorMixedPrecisionUnsupportedError",
}


_ACCELERATOR_CAPABILITY_TYPES = {
    "AcceleratorCapabilityMissingError",
    "AcceleratorTrainingUnsupportedError",
    "AcceleratorInferenceUnsupportedError",
}


_AI_INPUT_TYPES = {
    "AIEventTextMissingError",
    "AIAnomalyFeatureError",
    "AIAnomalyThresholdError",
    "AIInferenceShapeError",
    "AIInferenceNonFiniteInputError",
}


_MODEL_NOT_FOUND_TYPES = {
    "AIModelArtifactNotFoundError",
    "AIModelNotFoundError",
    "ModelNotFoundError",
}


_MODEL_LOAD_TYPES = {
    "AIModelLoadError",
    "AIVocabularyError",
    "AILabelMapError",
}


_MODEL_CONFIGURATION_TYPES = {
    "AIModelConfigurationError",
}


_INFERENCE_TYPES = {
    "AIInferenceExecutionError",
    "AIInferenceError",
}


def _internal_details(
    error: BaseException,
) -> Dict[str, Any]:
    details = getattr(
        error,
        "details",
        None,
    )

    safe = safe_error_details(
        details
        if isinstance(
            details,
            Mapping,
        )
        else None
    )

    safe.setdefault(
        "error_type",
        exception_type_name(error),
    )

    return safe


def translate_exception(
    error: BaseException,
) -> APIException:
    """
    Translate an internal exception into a stable
    public API exception.

    Translation primarily uses exception type
    names to avoid importing every domain module
    into the API boundary and introducing circular
    dependencies.
    """

    if isinstance(
        error,
        APIException,
    ):
        return error

    if isinstance(
        error,
        ValidationError,
    ):
        return from_validation_error(error)

    error_type = exception_type_name(error)

    module_name = exception_module_name(error)

    details = _internal_details(error)

    retryable = bool(
        getattr(
            error,
            "retryable",
            False,
        )
    )

    # ------------------------------------------------------
    # Accelerator errors
    # ------------------------------------------------------

    if error_type in _ACCELERATOR_NOT_FOUND_TYPES:
        return AcceleratorNotFoundAPIError(
            details=details,
        )

    if error_type in _ACCELERATOR_UNAVAILABLE_TYPES:
        return AcceleratorUnavailableAPIError(
            retryable=True,
            details=details,
        )

    if error_type in _ACCELERATOR_MEMORY_TYPES:
        return AcceleratorMemoryAPIError(
            retryable=retryable,
            details=details,
        )

    if error_type in _ACCELERATOR_PRECISION_TYPES:
        return AcceleratorAPIError(
            code=(APIErrorCode.ACCELERATOR_PRECISION_UNSUPPORTED.value),
            status_code=int(HTTPStatus.UNPROCESSABLE_ENTITY),
            public_message=("The requested accelerator precision is unsupported."),
            details=details,
        )

    if error_type in _ACCELERATOR_CAPABILITY_TYPES:
        return AcceleratorAPIError(
            code=(APIErrorCode.ACCELERATOR_CAPABILITY_MISSING.value),
            status_code=int(HTTPStatus.UNPROCESSABLE_ENTITY),
            public_message=("No accelerator device provides the required capability."),
            details=details,
        )

    if module_name.startswith("src.accelerators"):
        return AcceleratorAPIError(
            retryable=retryable,
            details=details,
        )

    # ------------------------------------------------------
    # AI/model errors
    # ------------------------------------------------------

    if error_type in _AI_INPUT_TYPES:
        return InferenceInputAPIError(
            details=details,
        )

    if error_type in _MODEL_NOT_FOUND_TYPES:
        return ModelNotFoundAPIError(
            details=details,
        )

    if error_type in _MODEL_LOAD_TYPES:
        return ModelLoadAPIError(
            retryable=retryable,
            details=details,
        )

    if error_type in _MODEL_CONFIGURATION_TYPES:
        return AIAPIError(
            code=(APIErrorCode.MODEL_CONFIGURATION_INVALID.value),
            status_code=int(HTTPStatus.UNPROCESSABLE_ENTITY),
            public_message=("The model configuration is invalid."),
            details=details,
        )

    if error_type in _INFERENCE_TYPES:
        return InferenceAPIError(
            retryable=retryable,
            details=details,
        )

    if module_name.startswith("src.ai"):
        return AIAPIError(
            retryable=retryable,
            details=details,
        )

    # ------------------------------------------------------
    # Common Python errors
    # ------------------------------------------------------

    if isinstance(
        error,
        (
            ValueError,
            TypeError,
        ),
    ):
        return BadRequestError(
            details={
                "error_type": error_type,
            },
        )

    if isinstance(
        error,
        TimeoutError,
    ):
        return RequestTimeoutAPIError(
            details={
                "error_type": error_type,
            },
        )

    if isinstance(
        error,
        FileNotFoundError,
    ):
        return ResourceNotFoundError(
            details={
                "error_type": error_type,
            },
        )

    if isinstance(
        error,
        PermissionError,
    ):
        return PermissionDeniedError(
            details={
                "error_type": error_type,
            },
        )

    # ------------------------------------------------------
    # Unknown internal failure
    # ------------------------------------------------------

    return InternalServerAPIError(
        details={
            "error_type": error_type,
        },
    )


# ------------------------------------------------------------------
# Public response construction
# ------------------------------------------------------------------


def error_response(
    error: BaseException,
    *,
    request_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
) -> Tuple[
    int,
    APIErrorResponse,
    Dict[str, str],
]:
    """
    Translate an exception and build the complete
    status/body/header result needed by a FastAPI
    exception handler.
    """

    api_error = translate_exception(error)

    response = api_error.to_response(
        request_id=request_id,
        duration_ms=duration_ms,
    )

    return (
        api_error.status_code,
        response,
        dict(api_error.headers),
    )


# ------------------------------------------------------------------
# Safe logging
# ------------------------------------------------------------------


def log_api_exception(
    error: BaseException,
    *,
    request_id: Optional[str] = None,
    method: Optional[str] = None,
    route: Optional[str] = None,
) -> APIException:
    """
    Log bounded exception metadata.

    The original exception message is not included
    in the structured log fields.
    """

    translated = translate_exception(error)

    log_fields = {
        "api_error_code": (translated.code),
        "status_code": (translated.status_code),
        "retryable": (translated.retryable),
        "internal_error_type": (exception_type_name(error)),
        "internal_error_module": (exception_module_name(error)),
        "request_id": (request_id),
        "method": method,
        "route": route,
    }

    logger.log(
        translated.log_level,
        "API request failed",
        extra={
            "api_error": (log_fields),
        },
        # Tracebacks should only be logged for
        # server-side failures. They are never
        # returned to the caller.
        exc_info=(translated.status_code >= 500),
    )

    return translated


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "MAX_ERROR_CODE_LENGTH",
    "MAX_PUBLIC_MESSAGE_LENGTH",
    "MAX_ERROR_DETAIL_COUNT",
    "MAX_ERROR_FIELD_COUNT",
    "APIErrorCode",
    "safe_error_details",
    "exception_type_name",
    "exception_module_name",
    "APIException",
    "BadRequestError",
    "RequestValidationAPIError",
    "AuthenticationRequiredError",
    "PermissionDeniedError",
    "ResourceNotFoundError",
    "ResourceConflictError",
    "PreconditionFailedError",
    "FeatureDisabledError",
    "RequestTooLargeError",
    "RequestTimeoutAPIError",
    "RateLimitExceededError",
    "ConcurrencyLimitError",
    "ServiceUnavailableAPIError",
    "InternalServerAPIError",
    "AcceleratorAPIError",
    "AcceleratorUnavailableAPIError",
    "AcceleratorNotFoundAPIError",
    "AcceleratorMemoryAPIError",
    "AcceleratorOptimizationAPIError",
    "AIAPIError",
    "InferenceAPIError",
    "InferenceInputAPIError",
    "InferenceTimeoutAPIError",
    "ModelNotFoundAPIError",
    "ModelUnavailableAPIError",
    "ModelLoadAPIError",
    "IncidentNotFoundAPIError",
    "DecisionNotFoundAPIError",
    "DecisionDeniedAPIError",
    "RemediationNotFoundAPIError",
    "RemediationNotApprovedAPIError",
    "RemediationConfirmationRequiredAPIError",
    "RemediationVersionConflictAPIError",
    "RemediationExecutionDisabledAPIError",
    "RemediationExecutionAPIError",
    "validation_error_fields",
    "from_validation_error",
    "translate_exception",
    "error_response",
    "log_api_exception",
]
