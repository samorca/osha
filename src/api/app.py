from __future__ import annotations

"""
Production FastAPI application composition for OSHA.

Responsibilities:

    load API configuration
    construct FastAPI
    manage application lifespan
    initialize process-scoped dependencies
    install middleware
    register exception handlers
    register API routers
    expose application metadata

This module is the HTTP composition root.

It does not:

    execute inference directly
    select accelerators directly
    load models per request
    correlate incidents
    make remediation decisions
    execute infrastructure remediation

Those responsibilities remain in their domain
services.
"""

import logging

from contextlib import (
    asynccontextmanager,
)

from importlib import (
    import_module,
)

from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    Optional,
    Tuple,
)

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from fastapi.exceptions import (
    RequestValidationError,
)

from fastapi.responses import (
    JSONResponse,
)

from pydantic import (
    ValidationError,
)

from starlette.exceptions import (
    HTTPException as StarletteHTTPException,
)

from src.api.config import (
    APIConfig,
    APIEnvironment,
    from_environment,
    safe_config_summary,
)

from src.api.dependencies import (
    APIDependencies,
    close_api_dependencies,
    create_api_dependencies,
    install_api_dependencies,
)

from src.api.errors import (
    APIException,
    BadRequestError,
    InternalServerAPIError,
    ResourceNotFoundError,
    RequestValidationAPIError,
    error_response,
    log_api_exception,
    validation_error_fields,
)

from src.api.middleware import (
    current_request_duration_ms,
    install_middleware,
)

from src.api.schema import (
    APIErrorResponse,
    APIResponseMetadata,
    APIStatus,
)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logger = logging.getLogger(
    __name__
)


# ------------------------------------------------------------------
# Application metadata
# ------------------------------------------------------------------

APPLICATION_TITLE = (
    "OSHA Orchestration API"
)

APPLICATION_DESCRIPTION = (
    "HTTP control plane for OSHA infrastructure "
    "intelligence, accelerator management, model "
    "serving, incident analysis, decisions, and "
    "guarded remediation."
)

APPLICATION_VERSION = "1.0.0"


# ------------------------------------------------------------------
# Router modules
# ------------------------------------------------------------------

ROUTER_MODULES: Tuple[
    str,
    ...,
] = (
    "src.api.routes.health",
    "src.api.routes.accelerators",
    "src.api.routes.inference",
    "src.api.routes.models",
    "src.api.routes.incidents",
    "src.api.routes.decisions",
    "src.api.routes.remediation",
)


# ------------------------------------------------------------------
# Request state helpers
# ------------------------------------------------------------------

def request_id_from_request(
    request: Request,
) -> Optional[str]:
    value = getattr(
        request.state,
        "request_id",
        None,
    )

    if value is None:
        return None

    return str(
        value
    )


# ------------------------------------------------------------------
# JSON error response
# ------------------------------------------------------------------

def api_error_json_response(
    error: BaseException,
    *,
    request: Request,
) -> JSONResponse:
    request_id = (
        request_id_from_request(
            request
        )
    )

    duration_ms = (
        current_request_duration_ms()
    )

    status_code, body, headers = (
        error_response(
            error,
            request_id=request_id,
            duration_ms=duration_ms,
        )
    )

    return JSONResponse(
        status_code=status_code,

        content=body.model_dump(
            mode="json",
            exclude_none=True,
        ),

        headers=headers,
    )


# ------------------------------------------------------------------
# Exception handlers
# ------------------------------------------------------------------

async def api_exception_handler(
    request: Request,
    error: APIException,
) -> JSONResponse:
    log_api_exception(
        error,
        request_id=(
            request_id_from_request(
                request
            )
        ),
        method=request.method,
        route=request.url.path,
    )

    return api_error_json_response(
        error,
        request=request,
    )


async def request_validation_exception_handler(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    """
    Translate FastAPI request-validation failures.

    Raw request body values are deliberately not
    copied into the response.
    """

    fields = []

    for item in error.errors()[
        :128
    ]:
        location = []

        for part in item.get(
            "loc",
            (),
        )[
            :32
        ]:
            if isinstance(
                part,
                int,
            ):
                location.append(
                    part
                )

            else:
                location.append(
                    str(
                        part
                    )[
                        :256
                    ]
                )

        fields.append(
            {
                "location": (
                    location
                ),

                "code": str(
                    item.get(
                        "type",
                        "validation_error",
                    )
                )[
                    :128
                ],

                "message": str(
                    item.get(
                        "msg",
                        "Invalid value.",
                    )
                )[
                    :2048
                ],
            }
        )

    translated = (
        RequestValidationAPIError(
            fields=[
                from_error_field(
                    value
                )
                for value
                in fields
            ],

            details={
                "error_type": (
                    type(
                        error
                    ).__name__
                ),
            },
        )
    )

    log_api_exception(
        translated,
        request_id=(
            request_id_from_request(
                request
            )
        ),
        method=request.method,
        route=request.url.path,
    )

    return api_error_json_response(
        translated,
        request=request,
    )


def from_error_field(
    value: Dict[
        str,
        Any,
    ],
):
    """
    Local helper so FastAPI's validation-error
    structure cannot leak its raw input field.
    """

    from src.api.schema import (
        APIErrorField,
    )

    return APIErrorField(
        location=list(
            value.get(
                "location",
                (),
            )
        ),

        code=str(
            value.get(
                "code",
                "validation_error",
            )
        ),

        message=str(
            value.get(
                "message",
                "Invalid value.",
            )
        ),
    )


async def pydantic_validation_exception_handler(
    request: Request,
    error: ValidationError,
) -> JSONResponse:
    translated = (
        RequestValidationAPIError(
            fields=list(
                validation_error_fields(
                    error
                )
            ),

            details={
                "error_type": (
                    type(
                        error
                    ).__name__
                ),
            },
        )
    )

    log_api_exception(
        translated,
        request_id=(
            request_id_from_request(
                request
            )
        ),
        method=request.method,
        route=request.url.path,
    )

    return api_error_json_response(
        translated,
        request=request,
    )


async def http_exception_handler(
    request: Request,
    error: StarletteHTTPException,
) -> JSONResponse:
    """
    Convert framework HTTP errors into the same
    public API error envelope.

    Internal exception details are not echoed.
    """

    status_code = int(
        error.status_code
    )

    if status_code == 404:
        translated: APIException = (
            ResourceNotFoundError()
        )

    elif status_code == 400:
        translated = BadRequestError()

    else:
        translated = APIException(
            public_message=(
                "The HTTP request could not "
                "be completed."
            ),

            code=(
                "http_"
                + str(
                    status_code
                )
            ),

            status_code=(
                status_code
                if 400
                <= status_code
                <= 599
                else 500
            ),
        )

    log_api_exception(
        translated,
        request_id=(
            request_id_from_request(
                request
            )
        ),
        method=request.method,
        route=request.url.path,
    )

    return api_error_json_response(
        translated,
        request=request,
    )


async def unhandled_exception_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """
    Last-resort boundary for route/service errors.

    The public response never receives the raw
    exception text.
    """

    translated = (
        InternalServerAPIError(
            details={
                "error_type": (
                    type(
                        error
                    ).__name__
                ),
            },
        )
    )

    log_api_exception(
        error,
        request_id=(
            request_id_from_request(
                request
            )
        ),
        method=request.method,
        route=request.url.path,
    )

    return api_error_json_response(
        translated,
        request=request,
    )


def install_exception_handlers(
    app: FastAPI,
) -> None:
    app.add_exception_handler(
        APIException,
        api_exception_handler,
    )

    app.add_exception_handler(
        RequestValidationError,
        request_validation_exception_handler,
    )

    app.add_exception_handler(
        ValidationError,
        pydantic_validation_exception_handler,
    )

    app.add_exception_handler(
        StarletteHTTPException,
        http_exception_handler,
    )

    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )


# ------------------------------------------------------------------
# Router feature map
# ------------------------------------------------------------------

def router_enabled(
    module_name: str,
    config: APIConfig,
) -> bool:
    """
    Decide whether a router family should be
    mounted at all.

    Individual routes still enforce their own
    finer-grained feature gates.
    """

    if module_name.endswith(
        ".health"
    ):
        return bool(
            config.features.
            health_enabled
        )

    if module_name.endswith(
        ".accelerators"
    ):
        return bool(
            config.features.
            accelerator_read_enabled
            or config.features.
            accelerator_optimization_enabled
        )

    if module_name.endswith(
        ".inference"
    ):
        return bool(
            config.features.
            inference_enabled
        )

    if module_name.endswith(
        ".models"
    ):
        return bool(
            config.features.
            model_read_enabled
            or config.features.
            model_load_operations_enabled
        )

    if module_name.endswith(
        ".incidents"
    ):
        return bool(
            config.features.
            incident_read_enabled
            or config.features.
            incident_create_enabled
        )

    if module_name.endswith(
        ".decisions"
    ):
        return bool(
            config.features.
            decision_read_enabled
            or config.features.
            decision_evaluation_enabled
        )

    if module_name.endswith(
        ".remediation"
    ):
        return bool(
            config.features.
            remediation_read_enabled
            or config.features.
            remediation_plan_creation_enabled
            or config.features.
            remediation_approval_enabled
            or config.features.
            remediation_execution_enabled
        )

    return False


# ------------------------------------------------------------------
# Router registration
# ------------------------------------------------------------------

def install_routers(
    app: FastAPI,
    config: APIConfig,
) -> Tuple[str, ...]:
    """
    Import and mount routers that expose:

        router = APIRouter(...)

    Empty route modules are tolerated during the
    staged implementation of src/api/routes.
    """

    installed = []

    for module_name in ROUTER_MODULES:
        if not router_enabled(
            module_name,
            config,
        ):
            continue

        try:
            module = import_module(
                module_name
            )

        except ModuleNotFoundError as exc:
            # Ignore only when the route module
            # itself is absent. Missing imports
            # inside an existing route module are
            # real application errors.
            if exc.name == module_name:
                logger.warning(
                    "API router module not found",
                    extra={
                        "api_router": {
                            "module": (
                                module_name
                            ),
                        },
                    },
                )

                continue

            raise

        router = getattr(
            module,
            "router",
            None,
        )

        if router is None:
            logger.debug(
                "API route module contains no "
                "router yet",
                extra={
                    "api_router": {
                        "module": (
                            module_name
                        ),
                    },
                },
            )

            continue

        app.include_router(
            router,
            prefix=(
                ""
                if module_name.endswith(
                    ".health"
                )
                else config.
                versioned_prefix
            ),
        )

        installed.append(
            module_name
        )

    return tuple(
        installed
    )


# ------------------------------------------------------------------
# OpenAPI configuration
# ------------------------------------------------------------------

def documentation_paths(
    config: APIConfig,
) -> Dict[
    str,
    Optional[str],
]:
    if not (
        config.documentation.enabled
    ):
        return {
            "docs_url": None,
            "redoc_url": None,
            "openapi_url": None,
        }

    return {
        "docs_url": (
            config.documentation.
            docs_path
        ),

        "redoc_url": (
            config.documentation.
            redoc_path
        ),

        "openapi_url": (
            config.documentation.
            openapi_path
        ),
    }


# ------------------------------------------------------------------
# Lifespan factory
# ------------------------------------------------------------------

def create_lifespan(
    config: APIConfig,
    *,
    dependencies: Optional[
        APIDependencies
    ] = None,
):
    """
    Build the application lifespan manager.

    Long-lived models and accelerator resources
    are initialized once per worker process.
    """

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        resolved_dependencies = (
            dependencies
            or create_api_dependencies(
                config=config
            )
        )

        try:
            await (
                resolved_dependencies.
                start()
            )

            install_api_dependencies(
                app,
                resolved_dependencies,
            )

            logger.info(
                "OSHA API started",
                extra={
                    "osha_api": {
                        "service": (
                            config.
                            service_name
                        ),

                        "environment": (
                            config.
                            environment
                        ),

                        "prefix": (
                            config.
                            versioned_prefix
                        ),
                    },
                },
            )

            yield

        finally:
            await close_api_dependencies(
                app
            )

            logger.info(
                "OSHA API stopped",
                extra={
                    "osha_api": {
                        "service": (
                            config.
                            service_name
                        ),
                    },
                },
            )

    return lifespan


# ------------------------------------------------------------------
# Application factory
# ------------------------------------------------------------------

def create_app(
    *,
    config: Optional[
        APIConfig
    ] = None,
    dependencies: Optional[
        APIDependencies
    ] = None,
) -> FastAPI:
    """
    Construct a fully configured OSHA FastAPI
    application.

    Application construction does not start
    heavyweight resources. Those are initialized
    by the lifespan manager when the ASGI server
    starts.
    """

    resolved_config = (
        config
        or from_environment()
    )

    resolved_config.validate()

    docs = documentation_paths(
        resolved_config
    )

    app = FastAPI(
        title=APPLICATION_TITLE,

        description=(
            APPLICATION_DESCRIPTION
        ),

        version=(
            APPLICATION_VERSION
        ),

        docs_url=(
            docs[
                "docs_url"
            ]
        ),

        redoc_url=(
            docs[
                "redoc_url"
            ]
        ),

        openapi_url=(
            docs[
                "openapi_url"
            ]
        ),

        root_path=(
            resolved_config.
            server.root_path
        ),

        lifespan=create_lifespan(
            resolved_config,
            dependencies=dependencies,
        ),
    )

    # Store immutable configuration immediately.
    # Runtime dependencies are stored separately
    # during lifespan startup.
    app.state.osha_api_config = (
        resolved_config
    )

    app.state.osha_api_config_summary = (
        safe_config_summary(
            resolved_config
        )
    )

    # ------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------

    install_exception_handlers(
        app
    )

    # ------------------------------------------------------
    # Middleware
    # ------------------------------------------------------

    install_middleware(
        app,
        resolved_config,
    )

    # ------------------------------------------------------
    # Routers
    # ------------------------------------------------------

    installed_routers = install_routers(
        app,
        resolved_config,
    )

    app.state.osha_installed_routers = (
        installed_routers
    )

    return app


# ------------------------------------------------------------------
# Default ASGI application
# ------------------------------------------------------------------

#
# Uvicorn entry point:
#
#     uvicorn src.api.app:app
#
# This constructs configuration and the FastAPI
# object at import time, but heavyweight OSHA
# subsystem resources are still deferred until
# lifespan startup.
#

app = create_app()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "APPLICATION_TITLE",
    "APPLICATION_DESCRIPTION",
    "APPLICATION_VERSION",

    "ROUTER_MODULES",

    "request_id_from_request",
    "api_error_json_response",

    "api_exception_handler",
    "request_validation_exception_handler",
    "pydantic_validation_exception_handler",
    "http_exception_handler",
    "unhandled_exception_handler",

    "install_exception_handlers",

    "router_enabled",
    "install_routers",

    "documentation_paths",

    "create_lifespan",
    "create_app",

    "app",
]
