from __future__ import annotations

"""
Inference execution for OSHA AI serving.

Execution flow:

    InferenceRequest
        -> validate request/payload limits
        -> route to one registered model
        -> lazy-load model adapter
        -> execute family-specific inference
        -> InferenceResult

Batch inference uses the same route/load/execute
path for every request. It never bypasses registry
or routing controls.

This module does not:

    train models
    promote artifacts
    interpret incidents
    choose remediation
"""

import json
import math
import time

from collections.abc import (
    Mapping,
    Sequence,
)

from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)

from dataclasses import (
    dataclass,
)

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.ai.base import (
    AIModelAdapter,
)

from src.ai.config import (
    AIConfig,
)

from src.ai.errors import (
    AIBatchDisabledError,
    AIBatchFamilyMismatchError,
    AIBatchInferenceError,
    AIBatchSizeError,
    AIError,
    AIInferenceInputError,
    AIInferenceNonFiniteInputError,
    AIInferencePayloadTooLargeError,
    AIInferenceRequestError,
    AIInferenceTimeoutError,
    AIModelUnavailableError,
    safe_ai_error_details,
)

from src.ai.registry import (
    AIModelRegistry,
)

from src.ai.routing import (
    AIRoute,
    AIModelRouter,
)

from src.ai.schema import (
    AIModelIdentity,
    BatchInferenceRequest,
    BatchInferenceResult,
    InferenceProvenance,
    InferenceRequest,
    InferenceResult,
    InferenceState,
)


# ---------------------------------------------------------------
# Payload validation result
# ---------------------------------------------------------------

@dataclass(frozen=True)
class PayloadValidation:
    request_id: str

    payload_bytes: int

    maximum_payload_bytes: int

    finite_values_checked: int

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "request_id": (
                self.request_id
            ),

            "payload_bytes": (
                self.payload_bytes
            ),

            "maximum_payload_bytes": (
                self.maximum_payload_bytes
            ),

            "finite_values_checked": (
                self.finite_values_checked
            ),
        }


# ---------------------------------------------------------------
# Inference execution record
# ---------------------------------------------------------------

@dataclass(frozen=True)
class InferenceExecution:
    """
    Internal execution summary.

    The public model output remains
    InferenceResult.
    """

    route: AIRoute

    result: InferenceResult

    payload_validation: PayloadValidation

    total_latency_ms: float

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "route": (
                self.route.to_dict()
            ),

            "result": (
                self.result.to_dict()
            ),

            "payload_validation": (
                self.payload_validation.
                to_dict()
            ),

            "total_latency_ms": (
                self.total_latency_ms
            ),
        }


# ---------------------------------------------------------------
# JSON payload size
# ---------------------------------------------------------------

def serialized_payload_size(
    payload: Mapping[
        str,
        Any,
    ],
) -> int:
    """
    Return the UTF-8 byte size of a canonical JSON
    payload.

    Payloads must be serializable because they may
    cross API, worker, and audit boundaries.
    """

    try:
        encoded = json.dumps(
            dict(
                payload
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        ).encode(
            "utf-8"
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise AIInferenceInputError(
            "Inference payload must be "
            "JSON-serializable.",
            details={
                "error_type": (
                    type(exc).__name__
                ),
            },
        ) from exc

    return len(
        encoded
    )


# ---------------------------------------------------------------
# Recursive finite-value validation
# ---------------------------------------------------------------

def validate_finite_values(
    value: Any,
    *,
    path: str = "payload",
) -> int:
    """
    Recursively reject NaN and infinite numeric
    values.

    Returns the number of finite numeric values
    inspected.
    """

    if value is None:
        return 0

    if isinstance(
        value,
        bool,
    ):
        return 0

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        numeric = float(
            value
        )

        if not math.isfinite(
            numeric
        ):
            raise (
                AIInferenceNonFiniteInputError(
                    "Inference payload "
                    "contains a non-finite "
                    "numeric value.",
                    details={
                        "path": path,
                    },
                )
            )

        return 1

    if isinstance(
        value,
        Mapping,
    ):
        count = 0

        for key, item in value.items():
            child_path = (
                path
                + "."
                + str(
                    key
                )
            )

            count += validate_finite_values(
                item,
                path=child_path,
            )

        return count

    if isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        count = 0

        for index, item in enumerate(
            value
        ):
            child_path = (
                path
                + "["
                + str(
                    index
                )
                + "]"
            )

            count += validate_finite_values(
                item,
                path=child_path,
            )

        return count

    return 0


# ---------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------

def validate_inference_request(
    request: InferenceRequest,
    *,
    config: AIConfig,
) -> PayloadValidation:
    """
    Validate one request before routing or model
    loading.
    """

    try:
        request.finalize()

    except Exception as exc:
        raise AIInferenceRequestError(
            "Inference request is invalid.",
            details={
                "error_type": (
                    type(exc).__name__
                ),
            },
        ) from exc

    if not config.enabled:
        raise AIModelUnavailableError(
            "AI serving is disabled.",
            retryable=False,
            details={
                "request_id": (
                    request.request_id
                ),
            },
        )

    if not config.family_allowed(
        request.family
    ):
        raise AIInferenceRequestError(
            "Inference request family is "
            "not enabled.",
            details={
                "request_id": (
                    request.request_id
                ),
                "family": (
                    request.family
                ),
            },
        )

    payload_bytes = (
        serialized_payload_size(
            request.payload
        )
    )

    maximum = (
        config.inference.
        maximum_payload_bytes
    )

    if payload_bytes > maximum:
        raise (
            AIInferencePayloadTooLargeError(
                "Inference payload exceeds "
                "the configured byte "
                "limit.",
                details={
                    "request_id": (
                        request.request_id
                    ),
                    "payload_bytes": (
                        payload_bytes
                    ),
                    "maximum_payload_bytes": (
                        maximum
                    ),
                },
            )
        )

    finite_values_checked = 0

    if (
        config.inference.
        reject_non_finite_inputs
    ):
        finite_values_checked = (
            validate_finite_values(
                request.payload
            )
        )

        finite_values_checked += (
            validate_finite_values(
                request.options,
                path="options",
            )
        )

    return PayloadValidation(
        request_id=str(
            request.request_id
        ),
        payload_bytes=(
            payload_bytes
        ),
        maximum_payload_bytes=(
            maximum
        ),
        finite_values_checked=(
            finite_values_checked
        ),
    )


# ---------------------------------------------------------------
# Failure identity
# ---------------------------------------------------------------

def failure_model_identity(
    request: InferenceRequest,
    *,
    route: Optional[
        AIRoute
    ] = None,
) -> AIModelIdentity:
    """
    Construct the best available model identity
    for a failed inference result.

    If routing succeeded, preserve the exact
    selected model. Otherwise use requested
    identity with bounded placeholders.
    """

    if route is not None:
        return (
            route.descriptor.
            identity
        )

    return AIModelIdentity(
        model_name=(
            request.model_name
            or "unrouted"
        ),

        family=(
            request.family
        ),

        version=(
            request.model_version
            or "unresolved"
        ),
    )


# ---------------------------------------------------------------
# Structured failure result
# ---------------------------------------------------------------

def inference_failure_result(
    request: InferenceRequest,
    error: BaseException,
    *,
    route: Optional[
        AIRoute
    ] = None,
    latency_ms: Optional[
        float
    ] = None,
) -> InferenceResult:
    """
    Convert an inference failure into a bounded
    result suitable for batch responses.

    Raw exception messages are not copied.
    """

    request.finalize()

    error_code = getattr(
        error,
        "code",
        "ai_inference_failed",
    )

    retryable = bool(
        getattr(
            error,
            "retryable",
            False,
        )
    )

    state = (
        InferenceState.
        REJECTED.value
        if isinstance(
            error,
            (
                AIInferenceRequestError,
                AIInferenceInputError,
                AIInferencePayloadTooLargeError,
                AIInferenceNonFiniteInputError,
            ),
        )
        else InferenceState.FAILED.value
    )

    return InferenceResult(
        request_id=str(
            request.request_id
        ),

        model=failure_model_identity(
            request,
            route=route,
        ),

        state=state,

        provenance=(
            request.provenance
        ),

        latency_ms=(
            latency_ms
        ),

        attributes={
            "error_code": (
                error_code
            ),

            "retryable": (
                retryable
            ),

            "route_resolved": (
                route is not None
            ),
        },

        error_type=(
            type(error).__name__
        ),
    ).finalize()


# ---------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------

class AIInferenceEngine:
    """
    Route, load, and execute OSHA AI models.

    Single-request inference raises expected
    AIError exceptions.

    Batch inference can either fail fast or
    return one structured failure result for each
    failed request.
    """

    def __init__(
        self,
        *,
        config: Optional[
            AIConfig
        ] = None,
        registry: Optional[
            AIModelRegistry
        ] = None,
        router: Optional[
            AIModelRouter
        ] = None,
    ) -> None:
        self.config = (
            config
            or AIConfig()
        )

        self.config.validate()

        self.registry = (
            registry
            or AIModelRegistry(
                config=self.config
            )
        )

        self.router = (
            router
            or AIModelRouter(
                config=self.config,
                registry=self.registry,
            )
        )

    # ---------------------------------------------------------
    # Adapter execution
    # ---------------------------------------------------------

    @staticmethod
    def _execute_adapter(
        adapter: AIModelAdapter,
        request: InferenceRequest,
    ) -> InferenceResult:
        return adapter.infer(
            request
        )

    # ---------------------------------------------------------
    # Timeout execution
    # ---------------------------------------------------------

    def _execute_with_timeout(
        self,
        adapter: AIModelAdapter,
        request: InferenceRequest,
    ) -> InferenceResult:
        """
        Execute adapter inference with the
        configured wall-clock timeout.

        The worker future cannot interrupt a
        low-level device kernel already running.
        The timeout prevents the caller from
        waiting indefinitely, while runtime
        worker/process isolation should provide
        stronger cancellation when needed.
        """

        timeout_seconds = (
            self.config.
            inference.
            timeout_seconds
        )

        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=(
                "osha-ai-inference"
            ),
        )

        future = executor.submit(
            self._execute_adapter,
            adapter,
            request,
        )

        try:
            return future.result(
                timeout=timeout_seconds
            )

        except FutureTimeoutError as exc:
            future.cancel()

            raise AIInferenceTimeoutError(
                details={
                    "request_id": (
                        request.request_id
                    ),
                    "model_id": (
                        adapter.model_id
                    ),
                    "timeout_seconds": (
                        timeout_seconds
                    ),
                },
            ) from exc

        finally:
            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

    # ---------------------------------------------------------
    # Execute one with details
    # ---------------------------------------------------------

    def execute(
        self,
        request: InferenceRequest,
    ) -> InferenceExecution:
        started = time.perf_counter()

        route: Optional[
            AIRoute
        ] = None

        validation = (
            validate_inference_request(
                request,
                config=self.config,
            )
        )

        try:
            route = self.router.route(
                request
            )

            adapter = (
                self.registry.load(
                    route.model_id
                )
            )

            result = (
                self.
                _execute_with_timeout(
                    adapter,
                    request,
                )
            )

            total_latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            result.attributes.update(
                {
                    "route_reason": (
                        route.reason
                    ),

                    "fallback_route": (
                        route.
                        fallback_used
                    ),

                    "payload_bytes": (
                        validation.
                        payload_bytes
                    ),

                    "total_latency_ms": (
                        total_latency_ms
                    ),
                }
            )

            result.finalize()

            return InferenceExecution(
                route=route,
                result=result,
                payload_validation=(
                    validation
                ),
                total_latency_ms=(
                    total_latency_ms
                ),
            )

        except AIError:
            raise

        except Exception as exc:
            model_id = (
                route.model_id
                if route is not None
                else None
            )

            raise AIModelUnavailableError(
                "AI inference execution "
                "failed unexpectedly.",
                details=safe_ai_error_details(
                    exc,
                    operation=(
                        "execute_inference"
                    ),
                    model_id=model_id,
                    family=(
                        request.family
                    ),
                    request_id=(
                        request.request_id
                    ),
                ),
            ) from exc

    # ---------------------------------------------------------
    # Public single inference
    # ---------------------------------------------------------

    def infer(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        return self.execute(
            request
        ).result

    # ---------------------------------------------------------
    # Safe single inference
    # ---------------------------------------------------------

    def infer_safe(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        """
        Return a structured failure instead of
        raising an expected inference exception.
        """

        started = time.perf_counter()

        route: Optional[
            AIRoute
        ] = None

        try:
            request.finalize()

            validate_inference_request(
                request,
                config=self.config,
            )

            route = self.router.route(
                request
            )

            adapter = (
                self.registry.load(
                    route.model_id
                )
            )

            result = (
                self.
                _execute_with_timeout(
                    adapter,
                    request,
                )
            )

            total_latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            result.attributes.update(
                {
                    "route_reason": (
                        route.reason
                    ),

                    "fallback_route": (
                        route.
                        fallback_used
                    ),

                    "total_latency_ms": (
                        total_latency_ms
                    ),
                }
            )

            return result.finalize()

        except Exception as exc:
            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            return inference_failure_result(
                request,
                exc,
                route=route,
                latency_ms=(
                    latency_ms
                ),
            )

    # ---------------------------------------------------------
    # Validate batch
    # ---------------------------------------------------------

    def _validate_batch(
        self,
        batch: BatchInferenceRequest,
    ) -> None:
        try:
            batch.finalize()

        except Exception as exc:
            raise AIBatchInferenceError(
                "Inference batch is "
                "invalid.",
                details={
                    "error_type": (
                        type(exc).__name__
                    ),
                },
            ) from exc

        if not (
            self.config.
            inference.
            enable_batch_inference
        ):
            raise AIBatchDisabledError(
                "Batch inference is "
                "disabled."
            )

        maximum = (
            self.config.
            inference.
            maximum_batch_size
        )

        if (
            len(
                batch.requests
            )
            > maximum
        ):
            raise AIBatchSizeError(
                "Inference batch exceeds "
                "the configured size "
                "limit.",
                details={
                    "batch_id": (
                        batch.batch_id
                    ),
                    "request_count": len(
                        batch.requests
                    ),
                    "maximum_batch_size": (
                        maximum
                    ),
                },
            )

    # ---------------------------------------------------------
    # Optional homogeneous batch guard
    # ---------------------------------------------------------

    @staticmethod
    def validate_same_family(
        requests: Iterable[
            InferenceRequest
        ],
    ) -> str:
        families = {
            request.family
            for request
            in requests
        }

        if not families:
            raise AIBatchInferenceError(
                "Inference batch cannot "
                "be empty."
            )

        if len(
            families
        ) != 1:
            raise (
                AIBatchFamilyMismatchError(
                    "Inference batch "
                    "contains multiple "
                    "model families.",
                    details={
                        "families": sorted(
                            families
                        ),
                    },
                )
            )

        return next(
            iter(
                families
            )
        )

    # ---------------------------------------------------------
    # Batch inference
    # ---------------------------------------------------------

    def infer_batch(
        self,
        batch: BatchInferenceRequest,
        *,
        fail_fast: bool = False,
        require_same_family: bool = False,
    ) -> BatchInferenceResult:
        self._validate_batch(
            batch
        )

        if require_same_family:
            self.validate_same_family(
                batch.requests
            )

        results: List[
            InferenceResult
        ] = []

        for request in batch.requests:
            if fail_fast:
                results.append(
                    self.infer(
                        request
                    )
                )

            else:
                results.append(
                    self.infer_safe(
                        request
                    )
                )

        return BatchInferenceResult(
            batch_id=str(
                batch.batch_id
            ),
            results=results,
        ).finalize()

    # ---------------------------------------------------------
    # Convenience batch
    # ---------------------------------------------------------

    def infer_many(
        self,
        requests: Iterable[
            InferenceRequest
        ],
        *,
        fail_fast: bool = False,
        require_same_family: bool = False,
    ) -> BatchInferenceResult:
        batch = BatchInferenceRequest(
            requests=list(
                requests
            )
        ).finalize()

        return self.infer_batch(
            batch,
            fail_fast=fail_fast,
            require_same_family=(
                require_same_family
            ),
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "enabled": (
                self.config.enabled
            ),

            "maximum_batch_size": (
                self.config.
                inference.
                maximum_batch_size
            ),

            "maximum_payload_bytes": (
                self.config.
                inference.
                maximum_payload_bytes
            ),

            "timeout_seconds": (
                self.config.
                inference.
                timeout_seconds
            ),

            "batch_inference_enabled": (
                self.config.
                inference.
                enable_batch_inference
            ),

            "reject_non_finite_inputs": (
                self.config.
                inference.
                reject_non_finite_inputs
            ),

            "include_representations": (
                self.config.
                inference.
                include_representations
            ),

            "registry": (
                self.registry.summary()
            ),

            "router": (
                self.router.summary()
            ),

            "training_performed": (
                False
            ),

            "remediation_performed": (
                False
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "PayloadValidation",
    "InferenceExecution",

    "serialized_payload_size",
    "validate_finite_values",
    "validate_inference_request",

    "failure_model_identity",
    "inference_failure_result",

    "AIInferenceEngine",
]
