from __future__ import annotations

"""
Incident HTTP routes.

Mounted under:

    /api/v1/incidents
"""

from datetime import datetime
from typing import Annotated, Optional

from fastapi import (
    APIRouter,
    Depends,
    Query,
    Request,
)

from pydantic import (
    Field,
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
    APIInputModel,
    IncidentCreateRequest,
    IncidentListResponse,
    IncidentResponse,
    IncidentStatusAPI,
)

from src.api.services.incidents import (
    INCIDENT_SERVICE_NAME,
    IncidentOperationContext,
    IncidentService,
)


class IncidentStatusUpdateRequest(
    APIInputModel
):
    status: IncidentStatusAPI

    expected_version: int = Field(
        ge=1,
    )


router = APIRouter(
    prefix="/incidents",
    tags=["Incidents"],
)


def get_incident_service(
    request: Request,
) -> IncidentService:
    return typed_service(
        request,
        INCIDENT_SERVICE_NAME,
    )


IncidentServiceDependency = Annotated[
    IncidentService,
    Depends(get_incident_service),
]


def operation_context(
    request: Request,
) -> IncidentOperationContext:
    return IncidentOperationContext(
        request_id=request_id(request),
        actor_id=actor_id(request),
        tenant_id=tenant_id(request),
        correlation_id=correlation_id(request),
        idempotency_key=idempotency_key(request),
    )


@router.get(
    "",
    response_model=IncidentListResponse,
    summary="List incidents",
    operation_id="list_incidents",
)
async def list_incidents(
    request: Request,
    service_api: IncidentServiceDependency,
    incident_status: Annotated[
        Optional[IncidentStatusAPI],
        Query(alias="status"),
    ] = None,
    severity: Optional[str] = None,
    service: Optional[str] = None,
    environment: Optional[str] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=1000),
    ] = 100,
    offset: Annotated[
        int,
        Query(ge=0),
    ] = 0,
    descending: bool = True,
) -> IncidentListResponse:
    return await service_api.list_incidents(
        status=(
            incident_status.value
            if incident_status is not None
            else None
        ),
        severity=severity,
        service=service,
        environment=environment,
        created_after=created_after,
        created_before=created_before,
        limit=limit,
        offset=offset,
        descending=descending,
        context=operation_context(request),
    )


@router.post(
    "",
    response_model=IncidentResponse,
    status_code=201,
    summary="Create incident",
    operation_id="create_incident",
)
async def create_incident(
    payload: IncidentCreateRequest,
    request: Request,
    service: IncidentServiceDependency,
) -> IncidentResponse:
    return await service.create_incident(
        payload,
        context=operation_context(request),
    )


@router.get(
    "/{incident_id}",
    response_model=IncidentResponse,
    summary="Get incident",
    operation_id="get_incident",
)
async def get_incident(
    incident_id: str,
    request: Request,
    service: IncidentServiceDependency,
) -> IncidentResponse:
    return await service.get_incident(
        incident_id,
        context=operation_context(request),
    )


@router.patch(
    "/{incident_id}/status",
    response_model=IncidentResponse,
    summary="Update incident status",
    operation_id="update_incident_status",
)
async def update_incident_status(
    incident_id: str,
    payload: IncidentStatusUpdateRequest,
    request: Request,
    service: IncidentServiceDependency,
) -> IncidentResponse:
    status_value = (
        payload.status.value
        if hasattr(payload.status, "value")
        else str(payload.status)
    )

    return await service.update_status(
        incident_id,
        status_value,
        expected_version=(
            payload.expected_version
        ),
        context=operation_context(request),
    )


__all__ = [
    "router",
    "IncidentStatusUpdateRequest",
    "get_incident_service",
    "operation_context",
]
