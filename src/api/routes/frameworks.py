from __future__ import annotations

import json

from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    Request,
)
from pydantic import Field, field_validator

from src.api.routes import (
    actor_id,
    correlation_id,
    typed_service,
)
from src.api.schema import (
    APIInputModel,
    MAX_BATCH_SIZE,
    MAX_CONTENT_LENGTH,
)
from src.api.services.frameworks import (
    FRAMEWORK_SERVICE_NAME,
    FrameworkService,
)


router = APIRouter(
    prefix="/frameworks",
    tags=["Frameworks"],
)


class FrameworkAnalysisRequest(APIInputModel):
    records: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )
    environment_id: Optional[str] = Field(default=None, max_length=256)
    provider_hint: Optional[str] = Field(default=None, max_length=128)

    @field_validator("records")
    @classmethod
    def validate_payload_size(
        cls,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        encoded = json.dumps(
            records,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > MAX_CONTENT_LENGTH:
            raise ValueError(
                f"records payload cannot exceed {MAX_CONTENT_LENGTH} bytes"
            )
        return records


def get_framework_service(request: Request) -> FrameworkService:
    return typed_service(request, FRAMEWORK_SERVICE_NAME)


@router.get(
    "/status",
    summary="Get integrated framework status",
    operation_id="get_framework_status",
)
async def framework_status(request: Request) -> Dict[str, Any]:
    return get_framework_service(request).status()


@router.post(
    "/analyze",
    summary="Normalize and analyze telemetry",
    operation_id="analyze_telemetry_frameworks",
)
async def analyze_telemetry(
    payload: FrameworkAnalysisRequest,
    request: Request,
) -> Dict[str, Any]:
    return get_framework_service(request).process_records(
        payload.records,
        environment_id=payload.environment_id,
        provider_hint=payload.provider_hint,
        requested_by=actor_id(request),
        correlation_id=correlation_id(request),
    )


__all__ = [
    "FrameworkAnalysisRequest",
    "framework_status",
    "analyze_telemetry",
    "router",
]
