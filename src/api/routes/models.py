from __future__ import annotations

"""
Serving-model management routes.

Mounted under:

    /api/v1/models

These endpoints manage serving state. They do not
train or promote models.
"""

from typing import Annotated, Optional

from fastapi import (
    APIRouter,
    Depends,
    Query,
    Request,
)

from src.api.routes import (
    actor_id,
    correlation_id,
    idempotency_key,
    request_id,
    tenant_id,
    typed_service,
)

from src.api.schema import (
    ModelListResponse,
    ModelOperationRequest,
    ModelRefreshRequest,
    ModelRefreshResponse,
    ModelResponse,
    OperationResponse,
)

from src.api.services.models import (
    MODEL_SERVICE_NAME,
    ModelOperationContext,
    ModelService,
)


router = APIRouter(
    prefix="/models",
    tags=["Models"],
)


def get_model_service(
    request: Request,
) -> ModelService:
    return typed_service(
        request,
        MODEL_SERVICE_NAME,
    )


ModelServiceDependency = Annotated[
    ModelService,
    Depends(get_model_service),
]


def operation_context(
    request: Request,
) -> ModelOperationContext:
    return ModelOperationContext(
        request_id=request_id(request),
        actor_id=actor_id(request),
        tenant_id=tenant_id(request),
        correlation_id=correlation_id(request),
        idempotency_key=idempotency_key(request),
    )


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List serving models",
    operation_id="list_models",
)
def list_models(
    request: Request,
    service: ModelServiceDependency,
    family: Optional[str] = None,
    state: Optional[str] = None,
    loaded: Optional[bool] = None,
    ready: Optional[bool] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=1000),
    ] = 100,
    offset: Annotated[
        int,
        Query(ge=0),
    ] = 0,
) -> ModelListResponse:
    return service.list_models(
        family=family,
        state=state,
        loaded=loaded,
        ready=ready,
        limit=limit,
        offset=offset,
        context=operation_context(request),
    )


@router.get(
    "/{model_id}",
    response_model=ModelResponse,
    summary="Get serving model",
    operation_id="get_model",
)
def get_model(
    model_id: str,
    request: Request,
    service: ModelServiceDependency,
) -> ModelResponse:
    return service.get_model(
        model_id,
        context=operation_context(request),
    )


@router.post(
    "/refresh",
    response_model=ModelRefreshResponse,
    summary="Refresh promoted-model discovery",
    operation_id="refresh_models",
)
async def refresh_models(
    payload: ModelRefreshRequest,
    request: Request,
    service: ModelServiceDependency,
) -> ModelRefreshResponse:
    return await service.refresh_models(
        payload,
        context=operation_context(request),
    )


@router.post(
    "/{model_id}/load",
    response_model=OperationResponse,
    summary="Load serving model",
    operation_id="load_model",
)
async def load_model(
    model_id: str,
    payload: ModelOperationRequest,
    request: Request,
    service: ModelServiceDependency,
) -> OperationResponse:
    return await service.load_model(
        model_id,
        payload,
        context=operation_context(request),
    )


@router.post(
    "/{model_id}/unload",
    response_model=OperationResponse,
    summary="Unload serving model",
    operation_id="unload_model",
)
async def unload_model(
    model_id: str,
    payload: ModelOperationRequest,
    request: Request,
    service: ModelServiceDependency,
) -> OperationResponse:
    return await service.unload_model(
        model_id,
        payload,
        context=operation_context(request),
    )


__all__ = [
    "router",
    "get_model_service",
    "operation_context",
]
