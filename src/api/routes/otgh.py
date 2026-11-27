from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from src.api.routes import correlation_id, typed_service
from src.api.schema import APIInputModel, MAX_BATCH_SIZE, MAX_CONTENT_LENGTH
from src.api.services.frameworks import FRAMEWORK_SERVICE_NAME, FrameworkService


router = APIRouter(prefix="/otgh", tags=["OTGH"])


class OTGHDestination(APIInputModel):
    id: str
    type: str
    endpoint: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class OTGHRecord(APIInputModel):
    id: str
    tenant_id: str | None = None
    correlation_id: str | None = None
    type: str
    priority: int
    source: dict[str, Any] = Field(default_factory=dict)
    resource: dict[str, Any] | None = None
    content_type: str
    compression: str
    created_at: datetime
    payload: str
    metadata: dict[str, str] = Field(default_factory=dict)


class OTGHBatch(APIInputModel):
    id: str
    destination: OTGHDestination
    records: list[OTGHRecord] = Field(min_length=1, max_length=MAX_BATCH_SIZE)
    created_at: datetime


def get_framework_service(request: Request) -> FrameworkService:
    return typed_service(request, FRAMEWORK_SERVICE_NAME)


def normalize_record(record: OTGHRecord) -> dict[str, Any]:
    try:
        payload_bytes = base64.b64decode(record.payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"record {record.id!r} has an invalid base64 payload",
        ) from exc

    if record.content_type == "application/json":
        try:
            payload = json.loads(payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"record {record.id!r} payload is not valid JSON",
            ) from exc
        normalized = payload if isinstance(payload, dict) else {"value": payload}
    elif record.content_type == "text/plain":
        normalized = {"message": payload_bytes.decode("utf-8", errors="replace")}
    else:
        normalized = {
            "payload_base64": record.payload,
            "content_type": record.content_type,
        }

    normalized.setdefault(
        "otgh",
        {
            "record_id": record.id,
            "record_type": record.type,
            "priority": record.priority,
            "tenant_id": record.tenant_id,
            "correlation_id": record.correlation_id,
            "source": record.source,
            "resource": record.resource,
            "created_at": record.created_at.isoformat(),
            "metadata": record.metadata,
        },
    )
    return normalized


@router.post(
    "/batches",
    status_code=202,
    summary="Receive and analyze an OTGH delivery batch",
    operation_id="receive_otgh_batch",
)
async def receive_batch(
    batch: OTGHBatch,
    request: Request,
) -> dict[str, Any]:
    expected_token = os.getenv("OSHA_OTGH_INGEST_TOKEN", "").strip()
    if expected_token:
        authorization = request.headers.get("authorization", "")
        provided_token = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(provided_token, expected_token):
            raise HTTPException(status_code=401, detail="invalid OTGH credentials")

    records = [normalize_record(record) for record in batch.records]
    encoded_records = json.dumps(
        records,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded_records) > MAX_CONTENT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"normalized batch exceeds {MAX_CONTENT_LENGTH} bytes",
        )

    environment_ids = {
        str(record.resource.get("environment"))
        for record in batch.records
        if record.resource and record.resource.get("environment")
    }
    if len(environment_ids) == 1:
        environment_id = next(iter(environment_ids))
    else:
        environment_id = None

    provider_hints = {
        str(record.source.get("system"))
        for record in batch.records
        if record.source.get("system")
    }
    provider_hint = next(iter(provider_hints)) if len(provider_hints) == 1 else "otgh"

    correlation_ids = {
        record.correlation_id
        for record in batch.records
        if record.correlation_id
    }
    request_correlation_id = (
        next(iter(correlation_ids))
        if len(correlation_ids) == 1
        else correlation_id(request)
    )

    get_framework_service(request).process_records(
        records,
        environment_id=environment_id,
        provider_hint=provider_hint,
        requested_by="otgh",
        correlation_id=request_correlation_id,
    )

    return {
        "accepted_records": len(records),
        "rejected_records": 0,
        "retryable": False,
        "message": "OTGH batch accepted for OSHA analysis",
        "batch_id": batch.id,
    }


__all__ = ["OTGHBatch", "OTGHRecord", "normalize_record", "receive_batch", "router"]