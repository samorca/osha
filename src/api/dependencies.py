from __future__ import annotations

"""
Process-scoped dependency management for OSHA API.

Lifecycle:

    FastAPI lifespan startup
            ↓
    create_api_dependencies()
            ↓
    APIDependencies
            │
            ├── APIConfig
            ├── AcceleratorManager
            ├── AIConfig
            ├── AIModelRegistry
            ├── AIModelLoader
            └── AIPipeline
            ↓
    app.state.osha_dependencies
            ↓
    lightweight FastAPI dependency functions
            ↓
    route/service layer

Shutdown:

    close_api_dependencies()
            ↓
    unload AI models
    release accelerator selections
    clear process-scoped state

Important:

    This module does not create heavyweight
    application resources at import time.

    Resource initialization belongs to FastAPI's
    lifespan startup.

    API services are registered when the dependency
    graph is constructed.
"""

import asyncio
import inspect
import logging

from dataclasses import dataclass, field

from threading import (
    RLock,
)

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

from fastapi import (
    Request,
)

from src.accelerators import (
    AcceleratorConfig,
    AcceleratorManager,
)

from src.ai.config import (
    AIConfig,
    from_environment as ai_config_from_environment,
)

from src.ai.loader import (
    AIModelLoader,
)

from src.ai.pipeline import (
    AIPipeline,
)

from src.ai.registry import (
    AIModelRegistry,
)

from src.api.config import (
    APIConfig,
    from_environment,
    safe_config_summary,
)

from src.api.errors import (
    InternalServerAPIError,
    ServiceUnavailableAPIError,
)

from src.database.engine import DatabaseEngineManager


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

APP_DEPENDENCY_STATE_KEY = "osha_dependencies"

APP_CONFIG_STATE_KEY = "osha_api_config"


# ------------------------------------------------------------------
# Optional-call helpers
# ------------------------------------------------------------------


def _call_if_available(
    target: Any,
    name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Invoke target.name() only when available.

    This is used for compatibility between
    subsystem implementations whose explicit
    lifecycle API may differ.
    """

    if target is None:
        return None

    method = getattr(
        target,
        name,
        None,
    )

    if not callable(method):
        return None

    return method(
        *args,
        **kwargs,
    )


async def _call_maybe_async(
    target: Any,
    name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Invoke a lifecycle method and await its result
    when it returns an awaitable.
    """

    result = _call_if_available(
        target,
        name,
        *args,
        **kwargs,
    )

    if inspect.isawaitable(result):
        return await result

    return result


# ------------------------------------------------------------------
# Dependency container
# ------------------------------------------------------------------


@dataclass
class APIDependencies:
    """
    Process-local dependency container.

    One instance should exist per worker process.

    The container deliberately stores long-lived
    runtime objects instead of creating them for
    every HTTP request.
    """

    config: APIConfig

    accelerator_manager: AcceleratorManager

    ai_config: AIConfig

    model_registry: AIModelRegistry

    model_loader: AIModelLoader

    ai_pipeline: AIPipeline

    database_pipeline: Optional[Any] = None

    services: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    started: bool = False

    closed: bool = False

    _lock: RLock = field(
        default_factory=RLock,
        init=False,
        repr=False,
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    def validate(
        self,
    ) -> None:
        self.config.validate()

        self.ai_config.validate()

        if self.closed:
            raise RuntimeError("API dependency container is already closed.")

    # ------------------------------------------------------
    # Service registration
    # ------------------------------------------------------

    def register_service(
        self,
        name: str,
        service: Any,
        *,
        replace: bool = False,
    ) -> None:
        """
        Register a process-scoped API service.

        Used later by:

            services/accelerators.py
            services/inference.py
            services/models.py
            services/incidents.py
            services/decisions.py
            services/remediation.py
        """

        service_name = str(name).strip()

        if not service_name:
            raise ValueError("service name cannot be empty")

        if service is None:
            raise ValueError("service cannot be None")

        with self._lock:
            if service_name in self.services and not replace:
                raise ValueError(f"API service is already registered: {service_name}")

            self.services[service_name] = service

    # ------------------------------------------------------
    # Service lookup
    # ------------------------------------------------------

    def service(
        self,
        name: str,
    ) -> Any:
        service_name = str(name).strip()

        with self._lock:
            value = self.services.get(service_name)

        if value is None:
            raise ServiceUnavailableAPIError(
                "The requested API service is unavailable.",
                details={
                    "resource": ("api_service"),
                    "resource_id": (service_name),
                },
            )

        return value

    # ------------------------------------------------------
    # Start
    # ------------------------------------------------------

    async def start(
        self,
    ) -> None:
        """
        Initialize long-lived subsystem resources.

        This method is idempotent.
        """

        with self._lock:
            if self.started:
                return

            if self.closed:
                raise RuntimeError("Closed API dependency container cannot start.")

        self.validate()

        # --------------------------------------------------
        # AI model discovery / pipeline startup
        # --------------------------------------------------
        #
        # The concrete implementation decides what
        # start() means. We intentionally do not
        # force eager model loading here.
        #

        try:
            if self.database_pipeline is not None:
                self.database_pipeline.start()
                telemetry_factory = getattr(
                    self.database_pipeline,
                    "telemetry_repository",
                    None,
                )
                if callable(telemetry_factory):
                    telemetry_factory().ensure_schema()

            try:
                await _call_maybe_async(
                    self.ai_pipeline,
                    "start",
                )

            except TypeError:
                # Some pipeline versions may require
                # explicit discovery behavior.
                await _call_maybe_async(
                    self.ai_pipeline,
                    "start",
                    discover_existing=True,
                )

            framework_service = self.services.get("frameworks")
            await _call_maybe_async(framework_service, "start")

        except Exception:
            framework_service = self.services.get("frameworks")
            try:
                await _call_maybe_async(framework_service, "close")
            except Exception:
                logger.exception("OSHA framework service failed during startup cleanup")
            if self.database_pipeline is not None and self.database_pipeline.ready:
                self.database_pipeline.stop()
            raise

        with self._lock:
            self.started = True

        logger.info(
            "OSHA API dependencies started",
            extra={
                "api_dependencies": {
                    "service": (self.config.service_name),
                    "environment": (self.config.environment),
                },
            },
        )

    # ------------------------------------------------------
    # Close
    # ------------------------------------------------------

    async def close(
        self,
    ) -> None:
        """
        Release process-scoped resources.

        Shutdown is best-effort and idempotent.
        """

        with self._lock:
            if self.closed:
                return

        failures = []

        # --------------------------------------------------
        # API services
        # --------------------------------------------------

        with self._lock:
            services = list(self.services.items())

        for (
            service_name,
            service,
        ) in reversed(services):
            try:
                if callable(
                    getattr(
                        service,
                        "close",
                        None,
                    )
                ):
                    await _call_maybe_async(
                        service,
                        "close",
                    )

                elif callable(
                    getattr(
                        service,
                        "shutdown",
                        None,
                    )
                ):
                    await _call_maybe_async(
                        service,
                        "shutdown",
                    )

            except Exception as exc:
                failures.append(
                    {
                        "resource": ("api_service"),
                        "resource_id": (service_name),
                        "error_type": (type(exc).__name__),
                    }
                )

        if self.database_pipeline is not None:
            try:
                self.database_pipeline.stop()
            except Exception as exc:
                failures.append(
                    {
                        "resource": "database_pipeline",
                        "error_type": type(exc).__name__,
                    }
                )

        # --------------------------------------------------
        # AI pipeline
        # --------------------------------------------------

        try:
            if callable(
                getattr(
                    self.ai_pipeline,
                    "close",
                    None,
                )
            ):
                await _call_maybe_async(
                    self.ai_pipeline,
                    "close",
                )

            elif callable(
                getattr(
                    self.ai_pipeline,
                    "shutdown",
                    None,
                )
            ):
                await _call_maybe_async(
                    self.ai_pipeline,
                    "shutdown",
                )

            elif callable(
                getattr(
                    self.ai_pipeline,
                    "stop",
                    None,
                )
            ):
                await _call_maybe_async(
                    self.ai_pipeline,
                    "stop",
                )

        except Exception as exc:
            failures.append(
                {
                    "resource": ("ai_pipeline"),
                    "error_type": (type(exc).__name__),
                }
            )

        # --------------------------------------------------
        # Registry model unload
        # --------------------------------------------------

        try:
            unload_all = getattr(
                self.model_registry,
                "unload_all",
                None,
            )

            if callable(unload_all):
                result = unload_all()

                if inspect.isawaitable(result):
                    await result

            else:
                # Compatibility path when the
                # registry exposes entries plus
                # per-model unload.
                entries = []

                find = getattr(
                    self.model_registry,
                    "find",
                    None,
                )

                if callable(find):
                    try:
                        entries = list(find())

                    except TypeError:
                        entries = []

                unload = getattr(
                    self.model_registry,
                    "unload",
                    None,
                )

                if callable(unload):
                    for entry in entries:
                        model_id = getattr(
                            entry,
                            "model_id",
                            None,
                        )

                        loaded = bool(
                            getattr(
                                entry,
                                "loaded",
                                False,
                            )
                        )

                        if model_id and loaded:
                            result = unload(model_id)

                            if inspect.isawaitable(result):
                                await result

        except Exception as exc:
            failures.append(
                {
                    "resource": ("model_registry"),
                    "error_type": (type(exc).__name__),
                }
            )

        # --------------------------------------------------
        # Accelerator selections
        # --------------------------------------------------

        try:
            self.accelerator_manager.release_all()

        except Exception as exc:
            failures.append(
                {
                    "resource": ("accelerator_manager"),
                    "error_type": (type(exc).__name__),
                }
            )

        with self._lock:
            self.services.clear()

            self.closed = True
            self.started = False

        if failures:
            logger.warning(
                "OSHA API dependency shutdown completed with cleanup failures",
                extra={
                    "api_dependency_failures": (failures),
                },
            )

        else:
            logger.info("OSHA API dependencies closed")

    # ------------------------------------------------------
    # Readiness
    # ------------------------------------------------------

    @property
    def ready(
        self,
    ) -> bool:
        with self._lock:
            return bool(
                self.started
                and not self.closed
                and bool(
                    getattr(
                        self.services.get("frameworks"),
                        "ready",
                        True,
                    )
                )
                and (
                    self.database_pipeline is None
                    or self.database_pipeline.ready
                )
            )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            service_names = sorted(self.services)

            started = self.started
            closed = self.closed

        return {
            "ready": bool(started and not closed),
            "started": started,
            "closed": closed,
            "config": (safe_config_summary(self.config)),
            "services": (service_names),
            "resources": {
                "accelerator_manager": (self.accelerator_manager is not None),
                "model_registry": (self.model_registry is not None),
                "model_loader": (self.model_loader is not None),
                "ai_pipeline": (self.ai_pipeline is not None),
                "database_pipeline": (
                    self.database_pipeline is not None
                    and self.database_pipeline.ready
                ),
                "frameworks": bool(
                    getattr(
                        self.services.get("frameworks"),
                        "ready",
                        False,
                    )
                ),
            },
            "attributes": dict(self.attributes),
        }


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


def _register_default_api_services(
    dependencies: APIDependencies,
) -> None:
    from src.api.persistence import (
        SQLAlchemyDecisionRepository,
        SQLAlchemyIncidentRepository,
        SQLAlchemyRemediationRepository,
    )
    from src.api.services.accelerators import (
        register_accelerator_service,
    )
    from src.api.services.decisions import (
        register_decision_service,
    )
    from src.api.services.incidents import (
        register_incident_service,
    )
    from src.api.services.inference import (
        register_inference_service,
    )
    from src.api.services.forecasting import (
        register_capacity_forecasting_service,
    )
    from src.api.services.frameworks import (
        register_framework_service,
    )
    from src.api.services.models import (
        register_model_service,
    )
    from src.api.services.remediation import (
        register_remediation_service,
    )

    register_accelerator_service(dependencies)
    register_model_service(dependencies)
    register_inference_service(dependencies)
    register_capacity_forecasting_service(dependencies)
    register_framework_service(dependencies)
    register_incident_service(
        dependencies,
        repository=SQLAlchemyIncidentRepository(),
    )
    register_decision_service(
        dependencies,
        repository=SQLAlchemyDecisionRepository(),
    )
    register_remediation_service(
        dependencies,
        repository=SQLAlchemyRemediationRepository(),
    )


def create_api_dependencies(
    *,
    config: Optional[APIConfig] = None,
    accelerator_manager: Optional[AcceleratorManager] = None,
    accelerator_config: Optional[AcceleratorConfig] = None,
    ai_config: Optional[AIConfig] = None,
    model_registry: Optional[AIModelRegistry] = None,
    model_loader: Optional[AIModelLoader] = None,
    ai_pipeline: Optional[AIPipeline] = None,
    database_pipeline: Optional[Any] = None,
) -> APIDependencies:
    """
    Construct the process-scoped dependency graph.

    This function performs construction only.

    Call:

        await dependencies.start()

    from FastAPI lifespan startup.
    """

    resolved_config = config or from_environment()

    resolved_config.validate()

    resolved_accelerator_manager = accelerator_manager or AcceleratorManager(
        config=(accelerator_config or AcceleratorConfig())
    )

    resolved_ai_config = ai_config or ai_config_from_environment()

    resolved_ai_config.validate()

    resolved_registry = (
        model_registry
        if model_registry is not None
        else AIModelRegistry(config=(resolved_ai_config))
    )

    resolved_loader = model_loader or AIModelLoader(
        config=(resolved_ai_config),
        registry=(resolved_registry),
    )

    resolved_pipeline = ai_pipeline or AIPipeline(
        config=(resolved_ai_config),
        registry=(resolved_registry),
        loader=(resolved_loader),
    )

    dependencies = APIDependencies(
        config=(resolved_config),
        accelerator_manager=(resolved_accelerator_manager),
        ai_config=(resolved_ai_config),
        model_registry=(resolved_registry),
        model_loader=(resolved_loader),
        ai_pipeline=(resolved_pipeline),
        database_pipeline=(database_pipeline or DatabaseEngineManager()),
        attributes={
            "scope": ("worker_process"),
        },
    )

    dependencies.validate()

    _register_default_api_services(dependencies)

    return dependencies


# ------------------------------------------------------------------
# FastAPI app-state installation
# ------------------------------------------------------------------


def install_api_dependencies(
    app: Any,
    dependencies: APIDependencies,
) -> None:
    """
    Store the dependency graph on FastAPI /
    Starlette app.state.
    """

    if app is None:
        raise ValueError("app cannot be None")

    if not hasattr(
        app,
        "state",
    ):
        raise ValueError("app does not expose state")

    setattr(
        app.state,
        APP_DEPENDENCY_STATE_KEY,
        dependencies,
    )

    setattr(
        app.state,
        APP_CONFIG_STATE_KEY,
        dependencies.config,
    )


# ------------------------------------------------------------------
# FastAPI app-state removal
# ------------------------------------------------------------------


def remove_api_dependencies(
    app: Any,
) -> None:
    if app is None or not hasattr(
        app,
        "state",
    ):
        return

    for name in (
        APP_DEPENDENCY_STATE_KEY,
        APP_CONFIG_STATE_KEY,
    ):
        if hasattr(
            app.state,
            name,
        ):
            delattr(
                app.state,
                name,
            )


# ------------------------------------------------------------------
# Container retrieval
# ------------------------------------------------------------------


def get_api_dependencies(
    request: Request,
) -> APIDependencies:
    """
    Retrieve the process-scoped dependency graph
    from the FastAPI application.
    """

    value = getattr(
        request.app.state,
        APP_DEPENDENCY_STATE_KEY,
        None,
    )

    if value is None:
        raise ServiceUnavailableAPIError(
            "API runtime dependencies have not been initialized.",
            details={
                "resource": ("api_dependencies"),
            },
        )

    if not isinstance(
        value,
        APIDependencies,
    ):
        raise InternalServerAPIError(
            "API runtime dependency state is invalid.",
            details={
                "resource": ("api_dependencies"),
            },
        )

    if not value.ready:
        raise ServiceUnavailableAPIError(
            "API runtime dependencies are not ready.",
            details={
                "resource": ("api_dependencies"),
            },
        )

    return value


# ------------------------------------------------------------------
# Core request dependencies
# ------------------------------------------------------------------


def get_api_config(
    request: Request,
) -> APIConfig:
    return get_api_dependencies(request).config


def get_accelerator_manager(
    request: Request,
) -> AcceleratorManager:
    return get_api_dependencies(request).accelerator_manager


def get_ai_config(
    request: Request,
) -> AIConfig:
    return get_api_dependencies(request).ai_config


def get_model_registry(
    request: Request,
) -> AIModelRegistry:
    return get_api_dependencies(request).model_registry


def get_model_loader(
    request: Request,
) -> AIModelLoader:
    return get_api_dependencies(request).model_loader


def get_ai_pipeline(
    request: Request,
) -> AIPipeline:
    return get_api_dependencies(request).ai_pipeline


# ------------------------------------------------------------------
# Named API service retrieval
# ------------------------------------------------------------------


def get_api_service(
    request: Request,
    name: str,
) -> Any:
    return get_api_dependencies(request).service(name)


def typed_api_service(
    request: Request,
    name: str,
) -> Any:
    """
    Typed wrapper for service retrieval.

    Static typing is supplied by the caller.
    Runtime service validation remains the
    service's responsibility.
    """

    return get_api_service(
        request,
        name,
    )


# ------------------------------------------------------------------
# Feature gates
# ------------------------------------------------------------------


def require_feature(
    config: APIConfig,
    feature_name: str,
) -> None:
    """
    Require one APIFeatureConfig flag.

    Route dependencies/services can call this
    before executing an optional API capability.
    """

    name = str(feature_name).strip()

    if not name:
        raise ValueError("feature_name cannot be empty")

    if not hasattr(
        config.features,
        name,
    ):
        raise InternalServerAPIError(
            "Unknown API feature gate.",
            details={
                "resource": ("api_feature"),
                "resource_id": (name),
            },
        )

    enabled = bool(
        getattr(
            config.features,
            name,
        )
    )

    if not enabled:
        from src.api.errors import (
            FeatureDisabledError,
        )

        raise FeatureDisabledError(
            details={
                "resource": ("api_feature"),
                "resource_id": (name),
            },
        )


def require_inference_feature(
    request: Request,
) -> APIConfig:
    config = get_api_config(request)

    require_feature(
        config,
        "inference_enabled",
    )

    return config


def require_accelerator_read_feature(
    request: Request,
) -> APIConfig:
    config = get_api_config(request)

    require_feature(
        config,
        "accelerator_read_enabled",
    )

    return config


def require_accelerator_optimization_feature(
    request: Request,
) -> APIConfig:
    config = get_api_config(request)

    require_feature(
        config,
        "accelerator_optimization_enabled",
    )

    return config


def require_model_read_feature(
    request: Request,
) -> APIConfig:
    config = get_api_config(request)

    require_feature(
        config,
        "model_read_enabled",
    )

    return config


def require_model_operations_feature(
    request: Request,
) -> APIConfig:
    config = get_api_config(request)

    require_feature(
        config,
        "model_load_operations_enabled",
    )

    return config


def require_incident_create_feature(
    request: Request,
) -> APIConfig:
    config = get_api_config(request)

    require_feature(
        config,
        "incident_create_enabled",
    )

    return config


def require_decision_evaluation_feature(
    request: Request,
) -> APIConfig:
    config = get_api_config(request)

    require_feature(
        config,
        "decision_evaluation_enabled",
    )

    return config


def require_remediation_plan_feature(
    request: Request,
) -> APIConfig:
    config = get_api_config(request)

    require_feature(
        config,
        "remediation_plan_creation_enabled",
    )

    return config


def require_remediation_approval_feature(
    request: Request,
) -> APIConfig:
    config = get_api_config(request)

    require_feature(
        config,
        "remediation_approval_enabled",
    )

    return config


def require_remediation_execution_feature(
    request: Request,
) -> APIConfig:
    config = get_api_config(request)

    require_feature(
        config,
        "remediation_execution_enabled",
    )

    return config


def require_live_remediation_feature(
    request: Request,
) -> APIConfig:
    config = get_api_config(request)

    require_feature(
        config,
        "remediation_execution_enabled",
    )

    require_feature(
        config,
        "remediation_live_execution_enabled",
    )

    return config


# ------------------------------------------------------------------
# Startup helper
# ------------------------------------------------------------------


async def start_api_dependencies(
    app: Any,
    *,
    config: Optional[APIConfig] = None,
    dependencies: Optional[APIDependencies] = None,
) -> APIDependencies:
    """
    Create, start, and install dependencies.

    Designed for src/api/app.py lifespan.
    """

    resolved = dependencies or create_api_dependencies(config=config)

    try:
        await resolved.start()

        install_api_dependencies(
            app,
            resolved,
        )

        return resolved

    except Exception:
        try:
            await resolved.close()

        except Exception:
            logger.exception("API dependency cleanup failed after startup failure")

        raise


# ------------------------------------------------------------------
# Shutdown helper
# ------------------------------------------------------------------


async def close_api_dependencies(
    app: Any,
) -> None:
    """
    Close and remove process-scoped resources.

    Designed for src/api/app.py lifespan.
    """

    if app is None or not hasattr(
        app,
        "state",
    ):
        return

    dependencies = getattr(
        app.state,
        APP_DEPENDENCY_STATE_KEY,
        None,
    )

    try:
        if isinstance(
            dependencies,
            APIDependencies,
        ):
            await dependencies.close()

    finally:
        remove_api_dependencies(app)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "APP_DEPENDENCY_STATE_KEY",
    "APP_CONFIG_STATE_KEY",
    "APIDependencies",
    "create_api_dependencies",
    "install_api_dependencies",
    "remove_api_dependencies",
    "get_api_dependencies",
    "get_api_config",
    "get_accelerator_manager",
    "get_ai_config",
    "get_model_registry",
    "get_model_loader",
    "get_ai_pipeline",
    "get_api_service",
    "typed_api_service",
    "require_feature",
    "require_inference_feature",
    "require_accelerator_read_feature",
    "require_accelerator_optimization_feature",
    "require_model_read_feature",
    "require_model_operations_feature",
    "require_incident_create_feature",
    "require_decision_evaluation_feature",
    "require_remediation_plan_feature",
    "require_remediation_approval_feature",
    "require_remediation_execution_feature",
    "require_live_remediation_feature",
    "start_api_dependencies",
    "close_api_dependencies",
]
