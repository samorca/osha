from __future__ import annotations

"""
Production inference service for OSHA API.

Responsibilities:

    validate inference feature availability
    translate public API requests to internal AI requests
    invoke the shared AIPipeline
    enforce API-side batch and concurrency limits
    apply service-level inference timeouts
    translate internal inference results to API responses
    isolate failures in batch inference
    preserve request and provenance identifiers
    emit safe service summaries

Architecture:

    routes/inference.py
            |
            v
    InferenceService
            |
            v
    AIPipeline
            |
            v
    AI registry / loader / inference engine
            |
            v
    AcceleratorManager
            |
            v
    event or anomaly model

This module does not:

    define FastAPI routes
    create the FastAPI application
    construct models per request
    select accelerators directly
    expose checkpoint paths
    execute remediation
"""

import asyncio
import inspect
import logging
import time

from dataclasses import dataclass

from threading import RLock

from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from uuid import uuid4

from src.ai.pipeline import AIPipeline

from src.ai.schema import (
    InferenceRequest,
)

from src.api.config import APIConfig

from src.api.errors import (
    APIException,
    FeatureDisabledError,
    InferenceAPIError,
    InferenceInputAPIError,
    InferenceTimeoutAPIError,
    InternalServerAPIError,
    ModelNotFoundAPIError,
    ModelUnavailableAPIError,
    ServiceUnavailableAPIError,
    translate_exception,
)

from src.api.schema import (
    APIError,
    APIResponseMetadata,
    AnomalyInferenceDetailResponse,
    AnomalyInferenceRequest,
    BatchInferenceItemRequest,
    BatchInferenceItemResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    EventInferenceDetailResponse,
    EventInferenceRequest,
    InferenceProvenanceResponse,
    InferenceResponse,
    InferenceStateAPI,
    ModelFamilyAPI,
    ModelIdentityResponse,
    PredictionResponse,
)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Service identity
# ------------------------------------------------------------------

INFERENCE_SERVICE_NAME = "inference"

INFERENCE_SERVICE_VERSION = "1"


# ------------------------------------------------------------------
# Operation context
# ------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceOperationContext:
    """
    Request-scoped metadata supplied by the route.

    The service remains independent of FastAPI's
    Request object.
    """

    request_id: Optional[str] = None

    actor_id: Optional[str] = None

    tenant_id: Optional[str] = None

    correlation_id: Optional[str] = None

    client_host: Optional[str] = None

    def safe_attributes(
        self,
    ) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "actor_id": self.actor_id,
            "tenant_id": self.tenant_id,
            "correlation_id": self.correlation_id,
            "client_host": self.client_host,
        }


# ------------------------------------------------------------------
# Generic object helpers
# ------------------------------------------------------------------


def _mapping(
    value: Any,
) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(
        value,
        Mapping,
    ):
        return dict(value)

    converter = getattr(
        value,
        "to_dict",
        None,
    )

    if callable(converter):
        converted = converter()

        if isinstance(
            converted,
            Mapping,
        ):
            return dict(converted)

    dumper = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(dumper):
        converted = dumper()

        if isinstance(
            converted,
            Mapping,
        ):
            return dict(converted)

    return {}


def _value(
    source: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(
        source,
        Mapping,
    ):
        return source.get(
            name,
            default,
        )

    return getattr(
        source,
        name,
        default,
    )


def _string(
    value: Any,
    *,
    default: str = "",
) -> str:
    if value is None:
        return default

    return str(value)


def _string_list(
    value: Any,
) -> List[str]:
    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        return [value]

    if isinstance(
        value,
        Iterable,
    ):
        return [str(item) for item in value]

    return [str(value)]


def _optional_float(
    value: Any,
) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def _optional_int(
    value: Any,
) -> Optional[int]:
    if value is None:
        return None

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def _request_identifier(
    context: Optional[InferenceOperationContext],
    *,
    suffix: Optional[str] = None,
) -> str:
    base = (
        context.request_id
        if (context is not None and context.request_id)
        else str(uuid4())
    )

    if suffix:
        return str(base) + ":" + str(suffix)

    return str(base)


# ------------------------------------------------------------------
# Constructor compatibility helper
# ------------------------------------------------------------------


def _construct_with_supported_arguments(
    target: Any,
    values: Mapping[
        str,
        Any,
    ],
) -> Any:
    """
    Construct one domain object using only keyword
    arguments supported by its current signature.

    This keeps the API boundary compatible with
    small internal schema changes without silently
    changing public API contracts.
    """

    try:
        signature = inspect.signature(target)

    except (
        TypeError,
        ValueError,
    ):
        return target(**dict(values))

    parameters = signature.parameters

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_kwargs:
        arguments = dict(values)

    else:
        arguments = {
            name: value for name, value in values.items() if name in parameters
        }

    return target(**arguments)


# ------------------------------------------------------------------
# Provenance translation
# ------------------------------------------------------------------


def provenance_payload(
    provenance: Any,
    *,
    context: Optional[InferenceOperationContext] = None,
) -> Dict[str, Any]:
    source = _mapping(provenance)

    if context is not None and context.tenant_id and not source.get("tenant_id"):
        source["tenant_id"] = context.tenant_id

    if (
        context is not None
        and context.correlation_id
        and not source.get("correlation_id")
    ):
        source["correlation_id"] = context.correlation_id

    return source


def provenance_response(
    provenance: Any,
) -> Optional[InferenceProvenanceResponse]:
    source = _mapping(provenance)

    if not source:
        return None

    return InferenceProvenanceResponse(
        source=source.get("source"),
        source_id=source.get("source_id"),
        tenant_id=source.get("tenant_id"),
        environment=source.get("environment"),
        cluster=source.get("cluster"),
        host=source.get("host"),
        correlation_id=source.get("correlation_id"),
        observed_at=source.get("observed_at"),
        attributes=_mapping(source.get("attributes")),
    )


# ------------------------------------------------------------------
# Internal request translation
# ------------------------------------------------------------------


def event_domain_request(
    request: EventInferenceRequest,
    *,
    request_id: str,
    context: Optional[InferenceOperationContext] = None,
) -> InferenceRequest:
    payload = dict(request.payload)

    payload.setdefault(
        "content",
        request.content,
    )

    options = dict(request.options)

    if request.max_length is not None:
        options.setdefault(
            "max_length",
            request.max_length,
        )

    if request.model_name:
        options.setdefault(
            "model_name",
            request.model_name,
        )

    if request.model_version:
        options.setdefault(
            "model_version",
            request.model_version,
        )

    values = {
        "request_id": request_id,
        "family": (ModelFamilyAPI.EVENT.value),
        "model_family": (ModelFamilyAPI.EVENT.value),
        "model_name": (request.model_name),
        "model_version": (request.model_version),
        "payload": payload,
        "options": options,
        "provenance": (
            provenance_payload(
                request.provenance,
                context=context,
            )
        ),
        "attributes": {
            "api_request": True,
            "api_request_id": (
                context.request_id if context is not None else request_id
            ),
        },
    }

    try:
        domain_request = _construct_with_supported_arguments(
            InferenceRequest,
            values,
        )

        validator = getattr(
            domain_request,
            "validate",
            None,
        )

        if callable(validator):
            validator()

        return domain_request

    except Exception as exc:
        raise InferenceInputAPIError(
            "The event inference request could not be translated.",
            details={
                "operation": ("event_request_translation"),
                "request_id": request_id,
                "error_type": (type(exc).__name__),
            },
        ) from exc


def anomaly_domain_request(
    request: AnomalyInferenceRequest,
    *,
    request_id: str,
    context: Optional[InferenceOperationContext] = None,
) -> InferenceRequest:
    payload = dict(request.payload)

    payload.setdefault(
        "features",
        list(request.features),
    )

    options = dict(request.options)

    if request.threshold is not None:
        options.setdefault(
            "threshold",
            request.threshold,
        )

    if request.model_name:
        options.setdefault(
            "model_name",
            request.model_name,
        )

    if request.model_version:
        options.setdefault(
            "model_version",
            request.model_version,
        )

    values = {
        "request_id": request_id,
        "family": (ModelFamilyAPI.ANOMALY.value),
        "model_family": (ModelFamilyAPI.ANOMALY.value),
        "model_name": (request.model_name),
        "model_version": (request.model_version),
        "payload": payload,
        "options": options,
        "provenance": (
            provenance_payload(
                request.provenance,
                context=context,
            )
        ),
        "attributes": {
            "api_request": True,
            "api_request_id": (
                context.request_id if context is not None else request_id
            ),
        },
    }

    try:
        domain_request = _construct_with_supported_arguments(
            InferenceRequest,
            values,
        )

        validator = getattr(
            domain_request,
            "validate",
            None,
        )

        if callable(validator):
            validator()

        return domain_request

    except Exception as exc:
        raise InferenceInputAPIError(
            "The anomaly inference request could not be translated.",
            details={
                "operation": ("anomaly_request_translation"),
                "request_id": request_id,
                "error_type": (type(exc).__name__),
            },
        ) from exc


def batch_item_domain_request(
    item: BatchInferenceItemRequest,
    *,
    request_id: str,
    context: Optional[InferenceOperationContext] = None,
) -> InferenceRequest:
    if item.family == ModelFamilyAPI.EVENT.value:
        if item.event is None:
            raise InferenceInputAPIError(
                "Event batch item has no event payload.",
                details={
                    "request_id": request_id,
                },
            )

        return event_domain_request(
            item.event,
            request_id=request_id,
            context=context,
        )

    if item.family == ModelFamilyAPI.ANOMALY.value:
        if item.anomaly is None:
            raise InferenceInputAPIError(
                "Anomaly batch item has no anomaly payload.",
                details={
                    "request_id": request_id,
                },
            )

        return anomaly_domain_request(
            item.anomaly,
            request_id=request_id,
            context=context,
        )

    raise InferenceInputAPIError(
        "Unsupported inference model family.",
        details={
            "request_id": request_id,
            "family": item.family,
        },
    )


# ------------------------------------------------------------------
# Result translation
# ------------------------------------------------------------------


def prediction_response(
    value: Any,
) -> PredictionResponse:
    source = _mapping(value)

    return PredictionResponse(
        label=_string(
            source.get(
                "label",
                _value(
                    value,
                    "label",
                    "",
                ),
            )
        ),
        score=_optional_float(
            source.get(
                "score",
                _value(
                    value,
                    "score",
                    None,
                ),
            )
        ),
        probability=_optional_float(
            source.get(
                "probability",
                _value(
                    value,
                    "probability",
                    None,
                ),
            )
        ),
        attributes=_mapping(
            source.get(
                "attributes",
                _value(
                    value,
                    "attributes",
                    {},
                ),
            )
        ),
    )


def event_detail_response(
    value: Any,
) -> Optional[EventInferenceDetailResponse]:
    source = _mapping(value)

    if not source:
        return None

    source_prediction = source.get("source")

    component_prediction = source.get("component")

    event_prediction = source.get("event")

    return EventInferenceDetailResponse(
        source=(
            prediction_response(source_prediction)
            if source_prediction is not None
            else None
        ),
        component=(
            prediction_response(component_prediction)
            if component_prediction is not None
            else None
        ),
        event=(
            prediction_response(event_prediction)
            if event_prediction is not None
            else None
        ),
        representation_dimension=(
            _optional_int(source.get("representation_dimension"))
        ),
        attributes=_mapping(source.get("attributes")),
    )


def anomaly_detail_response(
    value: Any,
) -> Optional[AnomalyInferenceDetailResponse]:
    source = _mapping(value)

    if not source:
        return None

    probability = _optional_float(source.get("probability"))

    threshold = _optional_float(source.get("threshold"))

    if probability is None:
        raise InternalServerAPIError(
            "Anomaly inference result has no probability.",
            details={
                "resource": ("anomaly_inference_result"),
            },
        )

    if threshold is None:
        raise InternalServerAPIError(
            "Anomaly inference result has no threshold.",
            details={
                "resource": ("anomaly_inference_result"),
            },
        )

    return AnomalyInferenceDetailResponse(
        probability=probability,
        threshold=threshold,
        anomalous=bool(
            source.get(
                "anomalous",
                probability >= threshold,
            )
        ),
        score=_optional_float(source.get("score")),
        representation_dimension=(
            _optional_int(source.get("representation_dimension"))
        ),
        attributes=_mapping(source.get("attributes")),
    )


def model_identity_response(
    value: Any,
) -> ModelIdentityResponse:
    source = _mapping(value)

    return ModelIdentityResponse(
        model_id=_string(
            source.get(
                "model_id",
                _value(
                    value,
                    "model_id",
                    "",
                ),
            )
        ),
        name=_string(
            source.get(
                "name",
                _value(
                    value,
                    "name",
                    "",
                ),
            )
        ),
        family=_string(
            source.get(
                "family",
                _value(
                    value,
                    "family",
                    "",
                ),
            )
        ),
        version=_string(
            source.get(
                "version",
                _value(
                    value,
                    "version",
                    "",
                ),
            )
        ),
    )


def inference_response(
    result: Any,
    *,
    request_id: Optional[str] = None,
) -> InferenceResponse:
    source = _mapping(result)

    result_request_id = _string(
        source.get(
            "request_id",
            _value(
                result,
                "request_id",
                request_id,
            ),
        ),
        default=(request_id or str(uuid4())),
    )

    model = source.get(
        "model",
        _value(
            result,
            "model",
            None,
        ),
    )

    if model is None:
        raise InternalServerAPIError(
            "Inference result has no model identity.",
            details={
                "request_id": (result_request_id),
                "resource": ("inference_result"),
            },
        )

    predictions = source.get(
        "predictions",
        _value(
            result,
            "predictions",
            (),
        ),
    )

    return InferenceResponse(
        request_id=result_request_id,
        state=_string(
            source.get(
                "state",
                _value(
                    result,
                    "state",
                    InferenceStateAPI.COMPLETED.value,
                ),
            )
        ),
        model=model_identity_response(model),
        predictions=[
            prediction_response(prediction) for prediction in (predictions or ())
        ],
        event=event_detail_response(
            source.get(
                "event",
                _value(
                    result,
                    "event",
                    None,
                ),
            )
        ),
        anomaly=anomaly_detail_response(
            source.get(
                "anomaly",
                _value(
                    result,
                    "anomaly",
                    None,
                ),
            )
        ),
        provenance=provenance_response(
            source.get(
                "provenance",
                _value(
                    result,
                    "provenance",
                    None,
                ),
            )
        ),
        latency_ms=_optional_float(
            source.get(
                "latency_ms",
                _value(
                    result,
                    "latency_ms",
                    None,
                ),
            )
        ),
        attributes=_mapping(
            source.get(
                "attributes",
                _value(
                    result,
                    "attributes",
                    {},
                ),
            )
        ),
        metadata=APIResponseMetadata(
            request_id=(request_id or result_request_id),
        ),
    )


# ------------------------------------------------------------------
# Pipeline invocation
# ------------------------------------------------------------------


async def _await_if_needed(
    value: Any,
) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


def _invoke_with_supported_arguments(
    method: Callable[..., Any],
    values: Mapping[
        str,
        Any,
    ],
) -> Any:
    try:
        signature = inspect.signature(method)

    except (
        TypeError,
        ValueError,
    ):
        return method(**dict(values))

    parameters = signature.parameters

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_kwargs:
        arguments = dict(values)

    else:
        arguments = {
            name: value for name, value in values.items() if name in parameters
        }

    return method(**arguments)


async def invoke_pipeline(
    pipeline: AIPipeline,
    request: InferenceRequest,
) -> Any:
    """
    Invoke the pipeline through its supported
    public inference method.

    Preferred method order:

        infer
        inference
        predict
        execute
        run
    """

    for method_name in (
        "infer",
        "inference",
        "predict",
        "execute",
        "run",
    ):
        method = getattr(
            pipeline,
            method_name,
            None,
        )

        if not callable(method):
            continue

        try:
            return await _await_if_needed(method(request))

        except TypeError:
            # Compatibility path for keyword-only
            # pipeline implementations.
            try:
                return await _await_if_needed(
                    _invoke_with_supported_arguments(
                        method,
                        {
                            "request": request,
                            "inference_request": (request),
                        },
                    )
                )

            except TypeError:
                continue

    raise ServiceUnavailableAPIError(
        "The AI pipeline exposes no supported inference operation.",
        details={
            "resource": ("ai_pipeline"),
        },
    )


# ------------------------------------------------------------------
# Inference service
# ------------------------------------------------------------------


class InferenceService:
    """
    Process-scoped production inference service.

    One instance should be registered in
    APIDependencies and shared across requests.
    """

    def __init__(
        self,
        *,
        config: APIConfig,
        pipeline: AIPipeline,
    ) -> None:
        config.validate()

        self.config = config
        self.pipeline = pipeline

        self._closed = False
        self._lock = RLock()

        self._inference_semaphore = asyncio.Semaphore(
            config.limits.maximum_concurrent_inference_requests
        )

    # ------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------

    def _ensure_open(
        self,
    ) -> None:
        with self._lock:
            if self._closed:
                raise ServiceUnavailableAPIError(
                    "The inference service is closed.",
                    details={
                        "resource": (INFERENCE_SERVICE_NAME),
                    },
                )

    def _require_enabled(
        self,
    ) -> None:
        if not (self.config.features.inference_enabled):
            raise FeatureDisabledError(
                details={
                    "resource": ("inference"),
                },
            )

    def close(
        self,
    ) -> None:
        with self._lock:
            self._closed = True

    # ------------------------------------------------------
    # Execution
    # ------------------------------------------------------

    async def _execute(
        self,
        domain_request: InferenceRequest,
        *,
        public_request_id: str,
    ) -> Any:
        self._ensure_open()
        self._require_enabled()

        timeout = self.config.timeouts.inference_seconds

        started = time.perf_counter()

        try:
            async with self._inference_semaphore:
                result = await asyncio.wait_for(
                    invoke_pipeline(
                        self.pipeline,
                        domain_request,
                    ),
                    timeout=timeout,
                )

        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutAPIError(
                details={
                    "operation": ("inference"),
                    "request_id": (public_request_id),
                },
            ) from exc

        except APIException:
            raise

        except Exception as exc:
            translated = translate_exception(exc)

            if (
                isinstance(
                    translated,
                    APIException,
                )
                and translated.status_code < 500
            ):
                raise translated from exc

            raise InferenceAPIError(
                details={
                    "operation": ("inference"),
                    "request_id": (public_request_id),
                    "error_type": (type(exc).__name__),
                },
            ) from exc

        if result is None:
            raise InferenceAPIError(
                "The inference pipeline returned no result.",
                details={
                    "operation": ("inference"),
                    "request_id": (public_request_id),
                },
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        result_attributes = _value(
            result,
            "attributes",
            None,
        )

        if isinstance(
            result_attributes,
            dict,
        ):
            result_attributes.setdefault(
                "service_duration_ms",
                elapsed_ms,
            )

        return result

    # ------------------------------------------------------
    # Event inference
    # ------------------------------------------------------

    async def infer_event(
        self,
        request: EventInferenceRequest,
        *,
        context: Optional[InferenceOperationContext] = None,
    ) -> InferenceResponse:
        request_id = _request_identifier(context)

        domain_request = event_domain_request(
            request,
            request_id=request_id,
            context=context,
        )

        result = await self._execute(
            domain_request,
            public_request_id=request_id,
        )

        return inference_response(
            result,
            request_id=request_id,
        )

    # ------------------------------------------------------
    # Anomaly inference
    # ------------------------------------------------------

    async def infer_anomaly(
        self,
        request: AnomalyInferenceRequest,
        *,
        context: Optional[InferenceOperationContext] = None,
    ) -> InferenceResponse:
        request_id = _request_identifier(context)

        domain_request = anomaly_domain_request(
            request,
            request_id=request_id,
            context=context,
        )

        result = await self._execute(
            domain_request,
            public_request_id=request_id,
        )

        return inference_response(
            result,
            request_id=request_id,
        )

    # ------------------------------------------------------
    # Batch item
    # ------------------------------------------------------

    async def _infer_batch_item(
        self,
        item: BatchInferenceItemRequest,
        *,
        root_request_id: str,
        context: Optional[InferenceOperationContext] = None,
    ) -> BatchInferenceItemResponse:
        item_request_id = root_request_id + ":" + item.item_id

        try:
            domain_request = batch_item_domain_request(
                item,
                request_id=(item_request_id),
                context=context,
            )

            result = await self._execute(
                domain_request,
                public_request_id=(item_request_id),
            )

            return BatchInferenceItemResponse(
                item_id=item.item_id,
                success=True,
                result=inference_response(
                    result,
                    request_id=(item_request_id),
                ),
                error=None,
            )

        except Exception as exc:
            translated = translate_exception(exc)

            logger.warning(
                "Batch inference item failed",
                extra={
                    "batch_inference": {
                        "request_id": (root_request_id),
                        "item_id": (item.item_id),
                        "family": (item.family),
                        "error_code": (translated.code),
                        "error_type": (type(exc).__name__),
                    },
                },
            )

            return BatchInferenceItemResponse(
                item_id=item.item_id,
                success=False,
                result=None,
                error=translated.to_api_error(),
            )

    # ------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------

    async def infer_batch(
        self,
        request: BatchInferenceRequest,
        *,
        context: Optional[InferenceOperationContext] = None,
    ) -> BatchInferenceResponse:
        self._ensure_open()
        self._require_enabled()

        if len(request.items) > self.config.limits.maximum_batch_inference_items:
            raise InferenceInputAPIError(
                "The inference batch exceeds the configured API limit.",
                details={
                    "limit": (self.config.limits.maximum_batch_inference_items),
                    "current": len(request.items),
                },
            )

        if request.require_same_family:
            families = {item.family for item in request.items}

            if len(families) > 1:
                raise InferenceInputAPIError(
                    "The batch requires a single model family.",
                    details={
                        "operation": ("batch_inference"),
                    },
                )

        root_request_id = _request_identifier(context)

        responses: List[BatchInferenceItemResponse] = []

        if request.fail_fast:
            for item in request.items:
                response = await self._infer_batch_item(
                    item,
                    root_request_id=(root_request_id),
                    context=context,
                )

                responses.append(response)

                if not response.success:
                    break

        else:
            responses = list(
                await asyncio.gather(
                    *(
                        self._infer_batch_item(
                            item,
                            root_request_id=(root_request_id),
                            context=context,
                        )
                        for item in request.items
                    )
                )
            )

        succeeded = sum(1 for response in responses if response.success)

        failed = len(responses) - succeeded

        return BatchInferenceResponse(
            items=responses,
            total=len(responses),
            succeeded=succeeded,
            failed=failed,
            metadata=APIResponseMetadata(
                request_id=(root_request_id),
            ),
        )

    # ------------------------------------------------------
    # Health
    # ------------------------------------------------------

    def health(
        self,
        *,
        deep: bool = False,
    ) -> Dict[str, Any]:
        self._ensure_open()

        checker = getattr(
            self.pipeline,
            "health",
            None,
        )

        if callable(checker):
            try:
                result = checker(deep=deep)

            except TypeError:
                result = checker()

            return _mapping(result)

        summary = getattr(
            self.pipeline,
            "summary",
            None,
        )

        if callable(summary):
            try:
                result = summary(deep=deep)

            except TypeError:
                result = summary()

            return _mapping(result)

        return {
            "status": ("unknown"),
            "ready": False,
            "message": ("The AI pipeline exposes no health operation."),
        }

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            closed = self._closed

        return {
            "service": (INFERENCE_SERVICE_NAME),
            "version": (INFERENCE_SERVICE_VERSION),
            "closed": closed,
            "enabled": (self.config.features.inference_enabled),
            "timeout_seconds": (self.config.timeouts.inference_seconds),
            "maximum_batch_items": (self.config.limits.maximum_batch_inference_items),
            "maximum_concurrent_inference_requests": (
                self.config.limits.maximum_concurrent_inference_requests
            ),
            "pipeline": type(self.pipeline).__name__,
        }


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


def create_inference_service(
    *,
    config: APIConfig,
    pipeline: AIPipeline,
) -> InferenceService:
    return InferenceService(
        config=config,
        pipeline=pipeline,
    )


# ------------------------------------------------------------------
# Dependency registration
# ------------------------------------------------------------------


def register_inference_service(
    dependencies: Any,
    *,
    replace: bool = False,
) -> InferenceService:
    """
    Construct and register InferenceService in
    APIDependencies.

    The dependency container is not imported here,
    preventing a circular import.
    """

    config = getattr(
        dependencies,
        "config",
        None,
    )

    pipeline = getattr(
        dependencies,
        "ai_pipeline",
        None,
    )

    register = getattr(
        dependencies,
        "register_service",
        None,
    )

    if config is None:
        raise InternalServerAPIError(
            "API configuration is unavailable.",
            details={
                "resource": ("api_config"),
            },
        )

    if pipeline is None:
        raise InternalServerAPIError(
            "AI pipeline is unavailable.",
            details={
                "resource": ("ai_pipeline"),
            },
        )

    if not callable(register):
        raise InternalServerAPIError(
            "API service registration is unavailable.",
            details={
                "resource": ("api_dependencies"),
            },
        )

    service = create_inference_service(
        config=config,
        pipeline=pipeline,
    )

    register(
        INFERENCE_SERVICE_NAME,
        service,
        replace=replace,
    )

    return service


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "INFERENCE_SERVICE_NAME",
    "INFERENCE_SERVICE_VERSION",
    "InferenceOperationContext",
    "provenance_payload",
    "provenance_response",
    "event_domain_request",
    "anomaly_domain_request",
    "batch_item_domain_request",
    "prediction_response",
    "event_detail_response",
    "anomaly_detail_response",
    "model_identity_response",
    "inference_response",
    "invoke_pipeline",
    "InferenceService",
    "create_inference_service",
    "register_inference_service",
]
