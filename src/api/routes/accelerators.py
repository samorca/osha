from __future__ import annotations

"""
Accelerator HTTP routes.

Mounted under:

    /api/v1/accelerators
"""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Query,
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
    AcceleratorDeviceResponse,
    AcceleratorInventoryResponse,
    AcceleratorMetricsResponse,
    AcceleratorOptimizationRequest,
    AcceleratorOptimizationResponse,
    AcceleratorSelectionRequest,
    AcceleratorSelectionResponse,
)

from src.api.services.accelerators import (
    ACCELERATOR_SERVICE_NAME,
    AcceleratorOperationContext,
    AcceleratorService,
)


router = APIRouter(
    prefix="/accelerators",
    tags=["Accelerators"],
)


def get_accelerator_service(
    request: Request,
) -> AcceleratorService:
    return typed_service(
        request,
        ACCELERATOR_SERVICE_NAME,
    )


AcceleratorServiceDependency = Annotated[
    AcceleratorService,
    Depends(get_accelerator_service),
]


def operation_context(
    request: Request,
) -> AcceleratorOperationContext:
    return AcceleratorOperationContext(
        request_id=request_id(request),
        actor_id=actor_id(request),
        tenant_id=tenant_id(request),
        correlation_id=correlation_id(request),
    )


@router.get(
    "",
    response_model=AcceleratorInventoryResponse,
    summary="List accelerator devices",
    operation_id="list_accelerators",
)
def list_accelerators(
    request: Request,
    service: AcceleratorServiceDependency,
    refresh: Annotated[
        bool,
        Query(
            description=("Refresh backend discovery before returning the inventory.")
        ),
    ] = False,
) -> AcceleratorInventoryResponse:
    return service.list_devices(
        refresh=refresh,
        context=operation_context(request),
    )


@router.get(
    "/{device_id}",
    response_model=AcceleratorDeviceResponse,
    summary="Get accelerator device",
    operation_id="get_accelerator",
)
def get_accelerator(
    device_id: str,
    request: Request,
    service: AcceleratorServiceDependency,
    refresh: bool = False,
) -> AcceleratorDeviceResponse:
    return service.get_device(
        device_id,
        refresh=refresh,
        context=operation_context(request),
    )


@router.get(
    "/{device_id}/metrics",
    response_model=AcceleratorMetricsResponse,
    summary="Get accelerator metrics",
    operation_id="get_accelerator_metrics",
)
def get_accelerator_metrics(
    device_id: str,
    request: Request,
    service: AcceleratorServiceDependency,
    refresh_device: bool = False,
) -> AcceleratorMetricsResponse:
    return service.metrics(
        device_id,
        refresh_device=refresh_device,
        context=operation_context(request),
    )


@router.post(
    "/select",
    response_model=AcceleratorSelectionResponse,
    summary="Select an accelerator",
    operation_id="select_accelerator",
)
def select_accelerator(
    payload: AcceleratorSelectionRequest,
    request: Request,
    service: AcceleratorServiceDependency,
) -> AcceleratorSelectionResponse:
    return service.select(
        payload,
        context=operation_context(request),
    )


@router.post(
    "/optimize",
    response_model=AcceleratorOptimizationResponse,
    summary="Generate an accelerator optimization plan",
    operation_id="optimize_accelerator",
)
def optimize_accelerator(
    payload: AcceleratorOptimizationRequest,
    request: Request,
    service: AcceleratorServiceDependency,
) -> AcceleratorOptimizationResponse:
    """
    Return recommendations only.

    The optimizer does not automatically change
    the host, model, batch size, or runtime.
    """

    return service.optimize(
        payload,
        context=operation_context(request),
    )


__all__ = [
    "router",
    "get_accelerator_service",
    "operation_context",
]
