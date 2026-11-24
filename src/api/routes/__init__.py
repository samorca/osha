from __future__ import annotations

"""
Shared API route helpers.

Routes remain responsible only for:

    HTTP parameter binding
    dependency retrieval
    request-context construction
    service invocation
    response-model declaration
"""

from typing import Any, Optional

from fastapi import Request

from src.api.dependencies import (
    APIDependencies,
    get_api_dependencies,
)

from src.api.errors import (
    ServiceUnavailableAPIError,
)


def request_id(request: Request) -> Optional[str]:
    value = getattr(
        request.state,
        "request_id",
        None,
    )

    return str(value) if value is not None else None


def optional_header(
    request: Request,
    name: str,
    *,
    maximum_length: int = 256,
) -> Optional[str]:
    value = request.headers.get(name)

    if value is None:
        return None

    text = value.strip()

    if not text:
        return None

    return text[:maximum_length]


def actor_id(request: Request) -> Optional[str]:
    """
    Temporary identity adapter.

    Authentication middleware should eventually
    set request.state.actor_id. Header fallback
    should only be trusted behind an authenticated
    gateway.
    """

    value = getattr(
        request.state,
        "actor_id",
        None,
    )

    if value is not None:
        return str(value)

    return optional_header(
        request,
        "X-Actor-ID",
    )


def tenant_id(request: Request) -> Optional[str]:
    value = getattr(
        request.state,
        "tenant_id",
        None,
    )

    if value is not None:
        return str(value)

    return optional_header(
        request,
        "X-Tenant-ID",
    )


def correlation_id(
    request: Request,
) -> Optional[str]:
    value = getattr(
        request.state,
        "correlation_id",
        None,
    )

    if value is not None:
        return str(value)

    return optional_header(
        request,
        "X-Correlation-ID",
    )


def idempotency_key(
    request: Request,
) -> Optional[str]:
    return optional_header(
        request,
        "Idempotency-Key",
        maximum_length=256,
    )


def approval_identity(
    request: Request,
) -> Optional[str]:
    value = getattr(
        request.state,
        "approval_identity",
        None,
    )

    if value is not None:
        return str(value)

    return actor_id(request)


def service_from_request(
    request: Request,
    name: str,
) -> Any:
    dependencies = get_api_dependencies(request)

    return dependencies.service(name)


def typed_service(
    request: Request,
    name: str,
) -> Any:
    return service_from_request(
        request,
        name,
    )


__all__ = [
    "APIDependencies",
    "get_api_dependencies",
    "request_id",
    "optional_header",
    "actor_id",
    "tenant_id",
    "correlation_id",
    "idempotency_key",
    "approval_identity",
    "service_from_request",
    "typed_service",
]
