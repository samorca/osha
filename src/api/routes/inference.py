from __future__ import annotations

"""
AI inference HTTP routes.

Mounted under:

    /api/v1/inference
"""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Request,
)

from src.api.routes import (
    actor_id,
    correlation_id,
    request_id,
    tenant_id,
    typed_service,
)

from src.api.schema import (
    AnomalyInferenceRequest,
    BatchInferenceRequest,
    BatchInferenceResponse,
    EventInferenceRequest,
    InferenceResponse,
)

from src.api.services.inference import (
    INFERENCE_SERVICE_NAME,
    InferenceOperationContext,
    InferenceService,
)


router = APIRouter(
    prefix="/inference",
    tags=["Inference"],
)


def get_inference_service(
    request: Request,
) -> InferenceService:
    return typed_service(
        request,
        INFERENCE_SERVICE_NAME,
    )


InferenceServiceDependency = Annotated[
    InferenceService,
    Depends(get_inference_service),
]


def operation_context(
    request: Request,
) -> InferenceOperationContext:
    client = request.client

    return InferenceOperationContext(
        request_id=request_id(request),
        actor_id=actor_id(request),
        tenant_id=tenant_id(request),
        correlation_id=correlation_id(request),
        client_host=(
            client.host
            if client is not None
            else None
        ),
    )


@router.post(
    "/events",
    response_model=InferenceResponse,
    summary="Run event inference",
    operation_id="infer_event",
)
async def infer_event(
    payload: EventInferenceRequest,
    request: Request,
    service: InferenceServiceDependency,
) -> InferenceResponse:
    return await service.infer_event(
        payload,
        context=operation_context(request),
    )


@router.post(
    "/anomalies",
    response_model=InferenceResponse,
    summary="Run anomaly inference",
    operation_id="infer_anomaly",
)
async def infer_anomaly(
    payload: AnomalyInferenceRequest,
    request: Request,
    service: InferenceServiceDependency,
) -> InferenceResponse:
    return await service.infer_anomaly(
        payload,
        context=operation_context(request),
    )


@router.post(
    "/batch",
    response_model=BatchInferenceResponse,
    summary="Run batch inference",
    operation_id="infer_batch",
)
async def infer_batch(
    payload: BatchInferenceRequest,
    request: Request,
    service: InferenceServiceDependency,
) -> BatchInferenceResponse:
    return await service.infer_batch(
        payload,
        context=operation_context(request),
    )


__all__ = [
    "router",
    "get_inference_service",
    "operation_context",
]
