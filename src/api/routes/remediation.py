from __future__ import annotations

"""
Guarded remediation HTTP routes.

Mounted under:

    /api/v1/remediation

No route accesses infrastructure providers
directly. All safeguards remain in
RemediationService.
"""

from typing import (
    Annotated,
    List,
    Optional,
)

from fastapi import (
    APIRouter,
    Depends,
    Query,
    Request,
)

from pydantic import Field

from src.api.routes import (
    actor_id,
    approval_identity,
    correlation_id,
    request_id,
    tenant_id,
    typed_service,
)

from src.api.schema import (
    APIOutputModel,
    APIResponseMetadata,
    PaginationMetadata,
    RemediationApprovalRequest,
    RemediationExecutionRequest,
    RemediationPlanCreateRequest,
    RemediationPlanResponse,
    RemediationPlanStatusAPI,
)

from src.api.services.remediation import (
    REMEDIATION_SERVICE_NAME,
    RemediationOperationContext,
    RemediationService,
)


class RemediationPlanListResponse(APIOutputModel):
    plans: List[RemediationPlanResponse] = Field(
        default_factory=list,
    )

    pagination: PaginationMetadata

    metadata: APIResponseMetadata = Field(
        default_factory=APIResponseMetadata,
    )


router = APIRouter(
    prefix="/remediation",
    tags=["Remediation"],
)


def get_remediation_service(
    request: Request,
) -> RemediationService:
    return typed_service(
        request,
        REMEDIATION_SERVICE_NAME,
    )


RemediationServiceDependency = Annotated[
    RemediationService,
    Depends(get_remediation_service),
]


def operation_context(
    request: Request,
) -> RemediationOperationContext:
    return RemediationOperationContext(
        request_id=request_id(request),
        actor_id=actor_id(request),
        tenant_id=tenant_id(request),
        correlation_id=correlation_id(request),
        approval_identity=(approval_identity(request)),
    )


@router.get(
    "/plans",
    response_model=RemediationPlanListResponse,
    summary="List remediation plans",
    operation_id="list_remediation_plans",
)
async def list_remediation_plans(
    request: Request,
    service: RemediationServiceDependency,
    incident_id: Optional[str] = None,
    decision_id: Optional[str] = None,
    plan_status: Annotated[
        Optional[RemediationPlanStatusAPI],
        Query(alias="status"),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=1000),
    ] = 100,
    offset: Annotated[
        int,
        Query(ge=0),
    ] = 0,
) -> RemediationPlanListResponse:
    plans, total = await service.list_plans(
        incident_id=incident_id,
        decision_id=decision_id,
        status=(plan_status.value if plan_status is not None else None),
        limit=limit,
        offset=offset,
        context=operation_context(request),
    )

    returned = len(plans)

    return RemediationPlanListResponse(
        plans=list(plans),
        pagination=PaginationMetadata(
            limit=limit,
            offset=offset,
            returned=returned,
            total=total,
            has_more=(offset + returned < total),
        ),
        metadata=APIResponseMetadata(
            request_id=request_id(request),
        ),
    )


@router.post(
    "/plans",
    response_model=RemediationPlanResponse,
    status_code=201,
    summary="Create remediation plan",
    operation_id="create_remediation_plan",
)
async def create_remediation_plan(
    payload: RemediationPlanCreateRequest,
    request: Request,
    service: RemediationServiceDependency,
) -> RemediationPlanResponse:
    return await service.create_plan(
        payload,
        context=operation_context(request),
    )


@router.get(
    "/plans/{plan_id}",
    response_model=RemediationPlanResponse,
    summary="Get remediation plan",
    operation_id="get_remediation_plan",
)
async def get_remediation_plan(
    plan_id: str,
    request: Request,
    service: RemediationServiceDependency,
) -> RemediationPlanResponse:
    return await service.get_plan(
        plan_id,
        context=operation_context(request),
    )


@router.post(
    "/plans/{plan_id}/approve",
    response_model=RemediationPlanResponse,
    summary="Approve or reject remediation plan",
    operation_id="approve_remediation_plan",
)
async def approve_remediation_plan(
    plan_id: str,
    payload: RemediationApprovalRequest,
    request: Request,
    service: RemediationServiceDependency,
) -> RemediationPlanResponse:
    return await service.approve_plan(
        plan_id,
        payload,
        context=operation_context(request),
    )


@router.post(
    "/plans/{plan_id}/execute",
    response_model=RemediationPlanResponse,
    summary="Execute remediation plan",
    operation_id="execute_remediation_plan",
)
async def execute_remediation_plan(
    plan_id: str,
    payload: RemediationExecutionRequest,
    request: Request,
    service: RemediationServiceDependency,
) -> RemediationPlanResponse:
    """
    Execution remains subject to:

        confirmation
        version matching
        decision validity
        plan approval
        feature gates
        idempotency
        dry-run/live-execution policy
    """

    return await service.execute_plan(
        plan_id,
        payload,
        context=operation_context(request),
    )


__all__ = [
    "router",
    "RemediationPlanListResponse",
    "get_remediation_service",
    "operation_context",
]
