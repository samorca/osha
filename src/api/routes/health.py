from __future__ import annotations

"""
OSHA API health endpoints.

Endpoints:

    GET /health
        Aggregated application health.

    GET /health/live
        Lightweight process liveness probe.

    GET /health/ready
        Dependency/readiness probe.

Health routes intentionally live outside /api/v1
so infrastructure probes can use stable paths
independent of API version changes.

These routes do not:

    execute inference
    load models
    allocate accelerators
    perform remediation
    expose secrets or configuration contents
"""

import inspect
import logging
import time

from typing import (
    Annotated,
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
)

from fastapi import (
    APIRouter,
    Depends,
    Request,
    Response,
    status,
)

from src.api.config import (
    APIConfig,
)

from src.api.dependencies import (
    APIDependencies,
    get_api_dependencies,
)

from src.api.middleware import (
    current_request_id,
)

from src.api.schema import (
    APIResponseMetadata,
    HealthComponentResponse,
    HealthResponse,
    HealthStatus,
    LivenessResponse,
    ReadinessResponse,
)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Router
# ------------------------------------------------------------------

router = APIRouter(
    prefix="/health",
    tags=[
        "Health",
    ],
)


# ------------------------------------------------------------------
# Dependency aliases
# ------------------------------------------------------------------

Dependencies = Annotated[
    APIDependencies,
    Depends(get_api_dependencies),
]


# ------------------------------------------------------------------
# Status groups
# ------------------------------------------------------------------

_HEALTHY_STATES = {
    "healthy",
    "ok",
    "ready",
    "available",
    "up",
    "running",
}


_DEGRADED_STATES = {
    "degraded",
    "partial",
    "warning",
}


_UNHEALTHY_STATES = {
    "unhealthy",
    "failed",
    "error",
    "down",
}


_UNAVAILABLE_STATES = {
    "unavailable",
    "disabled",
    "not_ready",
    "not-ready",
}


# ------------------------------------------------------------------
# Helpers
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
        result = converter()

        if isinstance(
            result,
            Mapping,
        ):
            return dict(result)

    dumper = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(dumper):
        result = dumper()

        if isinstance(
            result,
            Mapping,
        ):
            return dict(result)

    return {}


async def _await_if_needed(
    value: Any,
) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


def _normalize_status(
    value: Any,
    *,
    ready: Optional[bool] = None,
) -> str:
    text = str(value or "").strip().lower()

    if text in _HEALTHY_STATES:
        return HealthStatus.HEALTHY.value

    if text in _DEGRADED_STATES:
        return HealthStatus.DEGRADED.value

    if text in _UNHEALTHY_STATES:
        return HealthStatus.UNHEALTHY.value

    if text in _UNAVAILABLE_STATES:
        return HealthStatus.UNAVAILABLE.value

    if ready is True:
        return HealthStatus.HEALTHY.value

    if ready is False:
        return HealthStatus.DEGRADED.value

    return HealthStatus.UNKNOWN.value


def _component_message(
    component: str,
    health_status: str,
    source: Mapping[
        str,
        Any,
    ],
) -> str:
    message = source.get("message")

    if message:
        return str(message)[:4096]

    if health_status == HealthStatus.HEALTHY.value:
        return component + " is healthy."

    if health_status == HealthStatus.DEGRADED.value:
        return component + " is degraded."

    if health_status == HealthStatus.UNHEALTHY.value:
        return component + " is unhealthy."

    if health_status == HealthStatus.UNAVAILABLE.value:
        return component + " is unavailable."

    return component + " health is unknown."


# ------------------------------------------------------------------
# Safe component attributes
# ------------------------------------------------------------------


def _safe_attributes(
    source: Mapping[
        str,
        Any,
    ],
) -> Dict[str, Any]:
    """
    Expose bounded operational metadata while
    excluding obvious sensitive values.
    """

    blocked = {
        "password",
        "secret",
        "token",
        "credential",
        "authorization",
        "cookie",
        "private",
        "certificate",
        "traceback",
        "stack",
        "path",
        "environment",
    }

    output: Dict[
        str,
        Any,
    ] = {}

    for raw_key, value in list(source.items())[:64]:
        key = str(raw_key)

        normalized = key.lower()

        if any(fragment in normalized for fragment in blocked):
            continue

        if key in {
            "status",
            "ready",
            "message",
            "error",
            "error_message",
        }:
            continue

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                type(None),
            ),
        ):
            output[key] = value

        elif isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            output[key] = [
                item
                for item in value[:32]
                if isinstance(
                    item,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    ),
                )
            ]

    return output


# ------------------------------------------------------------------
# Component check
# ------------------------------------------------------------------


async def _service_health(
    name: str,
    service: Any,
    *,
    critical: bool,
    deep: bool,
) -> HealthComponentResponse:
    checker = getattr(
        service,
        "health",
        None,
    )

    if not callable(checker):
        return HealthComponentResponse(
            component=name,
            status=(HealthStatus.UNKNOWN.value),
            message=(name + " does not expose a health check."),
            critical=critical,
            attributes={},
        )

    try:
        try:
            result = checker(deep=deep)

        except TypeError:
            result = checker()

        result = await _await_if_needed(result)

        source = _mapping(result)

        ready = source.get("ready")

        health_status = _normalize_status(
            source.get("status"),
            ready=(bool(ready) if ready is not None else None),
        )

        return HealthComponentResponse(
            component=name,
            status=health_status,
            message=(
                _component_message(
                    name,
                    health_status,
                    source,
                )
            ),
            critical=critical,
            error_type=None,
            attributes=(_safe_attributes(source)),
        )

    except Exception as exc:
        logger.warning(
            "Health component check failed",
            extra={
                "health_component": {
                    "component": name,
                    "error_type": (type(exc).__name__),
                },
            },
        )

        return HealthComponentResponse(
            component=name,
            status=(HealthStatus.UNHEALTHY.value),
            message=(name + " health check failed."),
            critical=critical,
            error_type=(type(exc).__name__),
            attributes={},
        )


# ------------------------------------------------------------------
# Known services
# ------------------------------------------------------------------

_SERVICE_HEALTH_POLICY: Tuple[
    Tuple[
        str,
        bool,
    ],
    ...,
] = (
    (
        "accelerators",
        True,
    ),
    (
        "models",
        True,
    ),
    (
        "inference",
        True,
    ),
    (
        "incidents",
        False,
    ),
    (
        "decisions",
        False,
    ),
    (
        "remediation",
        False,
    ),
)


# ------------------------------------------------------------------
# Component aggregation
# ------------------------------------------------------------------


async def collect_components(
    dependencies: APIDependencies,
    *,
    deep: bool,
) -> Tuple[
    HealthComponentResponse,
    ...,
]:
    components = []

    for (
        service_name,
        critical,
    ) in _SERVICE_HEALTH_POLICY:
        service = dependencies.services.get(service_name)

        if service is None:
            components.append(
                HealthComponentResponse(
                    component=service_name,
                    status=(HealthStatus.UNAVAILABLE.value),
                    message=(service_name + " service is not registered."),
                    critical=critical,
                    attributes={},
                )
            )

            continue

        components.append(
            await _service_health(
                service_name,
                service,
                critical=critical,
                deep=deep,
            )
        )

    return tuple(components)


def overall_status(
    components: Iterable[HealthComponentResponse],
) -> Tuple[
    str,
    bool,
]:
    """
    Calculate API health/readiness.

    Critical unhealthy or unavailable components
    make the application unready.

    Optional component failures degrade the API
    but do not automatically make core serving
    unready.
    """

    values = tuple(components)

    critical_bad = any(
        component.critical
        and component.status
        in {
            HealthStatus.UNHEALTHY.value,
            HealthStatus.UNAVAILABLE.value,
        }
        for component in values
    )

    critical_degraded = any(
        component.critical
        and component.status
        in {
            HealthStatus.DEGRADED.value,
            HealthStatus.UNKNOWN.value,
        }
        for component in values
    )

    optional_bad = any(
        not component.critical and component.status != HealthStatus.HEALTHY.value
        for component in values
    )

    if critical_bad:
        return (
            HealthStatus.UNHEALTHY.value,
            False,
        )

    if critical_degraded or optional_bad:
        return (
            HealthStatus.DEGRADED.value,
            not critical_degraded,
        )

    return (
        HealthStatus.HEALTHY.value,
        True,
    )


# ------------------------------------------------------------------
# HTTP response-status helper
# ------------------------------------------------------------------


def _apply_health_status_code(
    response: Response,
    *,
    ready: bool,
) -> None:
    if ready:
        response.status_code = status.HTTP_200_OK

    else:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------


@router.get(
    "",
    response_model=HealthResponse,
    summary="Get OSHA API health",
    operation_id="get_api_health",
    responses={
        200: {
            "description": (
                "API is healthy or degraded but core dependencies are ready."
            ),
        },
        503: {
            "description": ("One or more critical API dependencies are not ready."),
        },
    },
)
async def health(
    response: Response,
    dependencies: Dependencies,
) -> HealthResponse:
    started = time.perf_counter()

    components = await collect_components(
        dependencies,
        deep=True,
    )

    (
        health_status,
        ready,
    ) = overall_status(components)

    _apply_health_status_code(
        response,
        ready=ready,
    )

    duration_ms = (time.perf_counter() - started) * 1000.0

    return HealthResponse(
        service=(dependencies.config.service_name),
        status=health_status,
        ready=ready,
        live=(not dependencies.closed),
        version=(str(dependencies.config.api_version)),
        components=list(components),
        metadata=(
            APIResponseMetadata(
                request_id=(current_request_id()),
                duration_ms=(duration_ms),
            )
        ),
    )


# ------------------------------------------------------------------
# GET /health/live
# ------------------------------------------------------------------


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Check OSHA API process liveness",
    operation_id="get_api_liveness",
    responses={
        200: {
            "description": ("API worker process is live."),
        },
    },
)
async def liveness() -> LivenessResponse:
    """
    Liveness deliberately avoids deep dependency
    checks.

    If this handler executes, the ASGI worker is
    alive enough to serve the probe.
    """

    return LivenessResponse(
        status=(HealthStatus.HEALTHY.value),
        live=True,
        metadata=(
            APIResponseMetadata(
                request_id=(current_request_id()),
            )
        ),
    )


# ------------------------------------------------------------------
# GET /health/ready
# ------------------------------------------------------------------


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Check OSHA API readiness",
    operation_id="get_api_readiness",
    responses={
        200: {
            "description": ("Critical API dependencies are ready."),
        },
        503: {
            "description": ("One or more critical API dependencies are not ready."),
        },
    },
)
async def readiness(
    response: Response,
    dependencies: Dependencies,
) -> ReadinessResponse:
    components = await collect_components(
        dependencies,
        deep=False,
    )

    (
        health_status,
        ready,
    ) = overall_status(components)

    reasons = [
        (component.component + ": " + component.status)
        for component in components
        if (component.critical and component.status != HealthStatus.HEALTHY.value)
    ]

    if not dependencies.ready:
        ready = False

        health_status = HealthStatus.UNHEALTHY.value

        reasons.append("api_dependencies: not_ready")

    _apply_health_status_code(
        response,
        ready=ready,
    )

    return ReadinessResponse(
        status=health_status,
        ready=ready,
        reasons=reasons,
        metadata=(
            APIResponseMetadata(
                request_id=(current_request_id()),
            )
        ),
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "router",
    "Dependencies",
    "collect_components",
    "overall_status",
    "health",
    "liveness",
    "readiness",
]
