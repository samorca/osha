from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
)

from src.runtime.config import (
    RuntimeConfig,
    ServiceConfig,
)

from src.runtime.schema import (
    ReadinessStatus,
    RuntimeStatus,
    ServiceState,
    ServiceStatus,
)

from src.runtime.state import (
    RuntimeStateManager,
)


# ---------------------------------------------------------------
# Service lifecycle hooks
# ---------------------------------------------------------------

StartHook = Callable[
    [],
    Any,
]

StopHook = Callable[
    [],
    Any,
]

ReadinessProbe = Callable[
    [],
    bool,
]


# ---------------------------------------------------------------
# Runtime service
# ---------------------------------------------------------------

@dataclass
class RuntimeService:
    """
    Lifecycle contract for one internal OSHA
    application service.

    Examples may later include:

      orchestrator
      worker manager
      persistence layer
      API runtime
      observability runtime

    This is not an external infrastructure
    service definition.
    """

    name: str

    dependencies: List[
        str
    ] = field(
        default_factory=list
    )

    start_hook: Optional[
        StartHook
    ] = None

    stop_hook: Optional[
        StopHook
    ] = None

    readiness_probe: Optional[
        ReadinessProbe
    ] = None

    config: ServiceConfig = field(
        default_factory=ServiceConfig
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "RuntimeService":
        self.name = (
            self.name.strip()
        )

        if not self.name:
            raise ValueError(
                "Runtime service name "
                "is required"
            )

        self.dependencies = sorted(
            set(
                self.dependencies
            )
        )

        if (
            self.name
            in self.dependencies
        ):
            raise ValueError(
                f"Service {self.name} "
                "cannot depend on itself"
            )

        self.config.validate()

        return self

    def status(
        self,
    ) -> ServiceStatus:
        self.finalize()

        return ServiceStatus(
            name=self.name,
            state=(
                ServiceState.
                REGISTERED.value
            ),
            readiness=(
                ReadinessStatus.
                UNKNOWN.value
            ),
            required=(
                self.config.required
            ),
            dependencies=list(
                self.dependencies
            ),
            attributes={
                **dict(
                    self.attributes
                ),
                "enabled": (
                    self.config.enabled
                ),
                "startup_timeout_seconds": (
                    self.config.
                    startup_timeout_seconds
                ),
                "shutdown_timeout_seconds": (
                    self.config.
                    shutdown_timeout_seconds
                ),
                "restart_limit": (
                    self.config.
                    restart_limit
                ),
            },
        ).finalize()


# ---------------------------------------------------------------
# Registry
# ---------------------------------------------------------------

class ServiceRegistryError(
    RuntimeError
):
    pass


class RuntimeServiceRegistry:
    """
    Registry of internal OSHA runtime services.

    The registry is also responsible for
    validating service dependency relationships.
    """

    def __init__(
        self,
    ) -> None:
        self._services: Dict[
            str,
            RuntimeService,
        ] = {}

    def register(
        self,
        service: RuntimeService,
    ) -> RuntimeService:
        service.finalize()

        if (
            service.name
            in self._services
        ):
            raise (
                ServiceRegistryError(
                    "Runtime service "
                    "already registered: "
                    f"{service.name}"
                )
            )

        self._services[
            service.name
        ] = service

        return service

    def get(
        self,
        name: str,
    ) -> Optional[
        RuntimeService
    ]:
        return self._services.get(
            name
        )

    def names(
        self,
    ) -> List[str]:
        return sorted(
            self._services
        )

    def services(
        self,
    ) -> List[
        RuntimeService
    ]:
        return [
            self._services[
                name
            ]
            for name
            in self.names()
        ]

    # ---------------------------------------------------------
    # Dependency validation
    # ---------------------------------------------------------

    def validate_dependencies(
        self,
    ) -> None:
        names = set(
            self._services
        )

        for service in (
            self._services.values()
        ):
            missing = sorted(
                set(
                    service.dependencies
                )
                - names
            )

            if missing:
                raise (
                    ServiceRegistryError(
                        f"Service "
                        f"{service.name} has "
                        "missing "
                        "dependencies: "
                        f"{missing}"
                    )
                )

        # startup_order performs cycle
        # detection.
        self.startup_order()

    # ---------------------------------------------------------
    # Startup order
    # ---------------------------------------------------------

    def startup_order(
        self,
    ) -> List[str]:
        """
        Return deterministic dependency order
        using topological sorting.

        Dependencies start before dependents.
        """

        names = set(
            self._services
        )

        indegree: Dict[
            str,
            int,
        ] = {
            name: 0
            for name in names
        }

        dependents: Dict[
            str,
            Set[str],
        ] = {
            name: set()
            for name in names
        }

        for service in (
            self._services.values()
        ):
            for dependency in (
                service.dependencies
            ):
                if (
                    dependency
                    not in names
                ):
                    raise (
                        ServiceRegistryError(
                            f"Service "
                            f"{service.name} "
                            "depends on "
                            "unknown service "
                            f"{dependency}"
                        )
                    )

                indegree[
                    service.name
                ] += 1

                dependents[
                    dependency
                ].add(
                    service.name
                )

        ready = sorted(
            name
            for name, count
            in indegree.items()
            if count == 0
        )

        ordered: List[
            str
        ] = []

        while ready:
            current = ready.pop(
                0
            )

            ordered.append(
                current
            )

            for dependent in sorted(
                dependents[
                    current
                ]
            ):
                indegree[
                    dependent
                ] -= 1

                if (
                    indegree[
                        dependent
                    ]
                    == 0
                ):
                    ready.append(
                        dependent
                    )

                    ready.sort()

        if (
            len(ordered)
            != len(names)
        ):
            unresolved = sorted(
                name
                for name, count
                in indegree.items()
                if count > 0
            )

            raise (
                ServiceRegistryError(
                    "Runtime service "
                    "dependency cycle "
                    "detected: "
                    f"{unresolved}"
                )
            )

        return ordered

    # ---------------------------------------------------------
    # Shutdown order
    # ---------------------------------------------------------

    def shutdown_order(
        self,
    ) -> List[str]:
        """
        Shutdown happens in reverse dependency
        order.

        Dependents stop before their
        dependencies.
        """

        return list(
            reversed(
                self.startup_order()
            )
        )

    # ---------------------------------------------------------
    # Attach runtime statuses
    # ---------------------------------------------------------

    def attach_statuses(
        self,
        runtime: RuntimeStatus,
    ) -> RuntimeStatus:
        runtime.services = [
            self._services[
                name
            ].status()
            for name
            in self.startup_order()
        ]

        return runtime.finalize()


# ---------------------------------------------------------------
# Service operation result
# ---------------------------------------------------------------

@dataclass(frozen=True)
class ServiceOperationResult:
    name: str

    success: bool

    ready: bool

    reason: str

    error_type: Optional[
        str
    ] = None


# ---------------------------------------------------------------
# Runtime service manager
# ---------------------------------------------------------------

class RuntimeServiceManager:
    """
    Starts and stops internal services using
    registered lifecycle hooks.

    The manager does not execute infrastructure
    remediation commands.
    """

    def __init__(
        self,
        registry: RuntimeServiceRegistry,
        *,
        state: Optional[
            RuntimeStateManager
        ] = None,
    ) -> None:
        self.registry = registry

        self.state = (
            state
            or RuntimeStateManager()
        )

        self.registry.validate_dependencies()

    # ---------------------------------------------------------
    # Dependency readiness
    # ---------------------------------------------------------

    def _dependencies_ready(
        self,
        runtime: RuntimeStatus,
        service: RuntimeService,
    ) -> bool:
        for dependency_name in (
            service.dependencies
        ):
            dependency = (
                runtime.service(
                    dependency_name
                )
            )

            if (
                dependency is None
                or dependency.readiness
                != ReadinessStatus.
                READY.value
            ):
                return False

        return True

    # ---------------------------------------------------------
    # Start one service
    # ---------------------------------------------------------

    def start_service(
        self,
        runtime: RuntimeStatus,
        name: str,
    ) -> ServiceOperationResult:
        service = (
            self.registry.get(
                name
            )
        )

        if service is None:
            raise (
                ServiceRegistryError(
                    "Unknown runtime "
                    f"service: {name}"
                )
            )

        if not service.config.enabled:
            status = runtime.service(
                name
            )

            if status is not None:
                status.state = (
                    ServiceState.
                    STOPPED.value
                )

                status.readiness = (
                    ReadinessStatus.
                    NOT_READY.value
                )

                status.attributes[
                    "disabled"
                ] = True

            return (
                ServiceOperationResult(
                    name=name,
                    success=True,
                    ready=False,
                    reason=(
                        "Service is disabled "
                        "by runtime "
                        "configuration."
                    ),
                )
            )

        if not (
            self._dependencies_ready(
                runtime,
                service,
            )
        ):
            return (
                ServiceOperationResult(
                    name=name,
                    success=False,
                    ready=False,
                    reason=(
                        "Required service "
                        "dependencies are "
                        "not ready."
                    ),
                )
            )

        self.state.service_starting(
            runtime,
            name,
        )

        try:
            if (
                service.start_hook
                is not None
            ):
                service.start_hook()

            ready = (
                True
                if (
                    service.
                    readiness_probe
                    is None
                )
                else bool(
                    service.
                    readiness_probe()
                )
            )

            if ready:
                self.state.service_running(
                    runtime,
                    name,
                )

                return (
                    ServiceOperationResult(
                        name=name,
                        success=True,
                        ready=True,
                        reason=(
                            "Service started "
                            "and is ready."
                        ),
                    )
                )

            self.state.service_failed(
                runtime,
                name,
                reason=(
                    "Service readiness "
                    "probe did not report "
                    "ready."
                ),
            )

            return (
                ServiceOperationResult(
                    name=name,
                    success=False,
                    ready=False,
                    reason=(
                        "Service started but "
                        "did not become "
                        "ready."
                    ),
                )
            )

        except Exception as exc:
            # Avoid storing arbitrary exception
            # content in runtime state.
            self.state.service_failed(
                runtime,
                name,
                reason=(
                    "Service startup failed."
                ),
                error=(
                    "Service startup hook "
                    "failed."
                ),
            )

            return (
                ServiceOperationResult(
                    name=name,
                    success=False,
                    ready=False,
                    reason=(
                        "Service startup "
                        "failed."
                    ),
                    error_type=(
                        type(exc).__name__
                    ),
                )
            )

    # ---------------------------------------------------------
    # Stop one service
    # ---------------------------------------------------------

    def stop_service(
        self,
        runtime: RuntimeStatus,
        name: str,
    ) -> ServiceOperationResult:
        service = (
            self.registry.get(
                name
            )
        )

        if service is None:
            raise (
                ServiceRegistryError(
                    "Unknown runtime "
                    f"service: {name}"
                )
            )

        status = runtime.service(
            name
        )

        if status is None:
            raise (
                ServiceRegistryError(
                    "Runtime has no status "
                    f"for service: {name}"
                )
            )

        if (
            status.state
            == ServiceState.
            STOPPED.value
        ):
            return (
                ServiceOperationResult(
                    name=name,
                    success=True,
                    ready=False,
                    reason=(
                        "Service is already "
                        "stopped."
                    ),
                )
            )

        # A registered service may never have
        # been started because bootstrap stopped
        # earlier. No stop hook is required.
        if (
            status.state
            == ServiceState.
            REGISTERED.value
        ):
            status.state = (
                ServiceState.
                STOPPED.value
            )

            status.readiness = (
                ReadinessStatus.
                NOT_READY.value
            )

            return (
                ServiceOperationResult(
                    name=name,
                    success=True,
                    ready=False,
                    reason=(
                        "Service was never "
                        "started."
                    ),
                )
            )

        self.state.service_stopping(
            runtime,
            name,
        )

        try:
            if (
                service.stop_hook
                is not None
            ):
                service.stop_hook()

            self.state.service_stopped(
                runtime,
                name,
            )

            return (
                ServiceOperationResult(
                    name=name,
                    success=True,
                    ready=False,
                    reason=(
                        "Service stopped."
                    ),
                )
            )

        except Exception as exc:
            self.state.service_failed(
                runtime,
                name,
                reason=(
                    "Service shutdown "
                    "failed."
                ),
                error=(
                    "Service shutdown "
                    "hook failed."
                ),
            )

            return (
                ServiceOperationResult(
                    name=name,
                    success=False,
                    ready=False,
                    reason=(
                        "Service shutdown "
                        "failed."
                    ),
                    error_type=(
                        type(exc).__name__
                    ),
                )
            )

    # ---------------------------------------------------------
    # Start all
    # ---------------------------------------------------------

    def start_all(
        self,
        runtime: RuntimeStatus,
        *,
        fail_fast: bool = True,
    ) -> List[
        ServiceOperationResult
    ]:
        results: List[
            ServiceOperationResult
        ] = []

        for name in (
            self.registry.
            startup_order()
        ):
            result = (
                self.start_service(
                    runtime,
                    name,
                )
            )

            results.append(
                result
            )

            service = (
                self.registry.get(
                    name
                )
            )

            if result.success:
                continue

            if (
                fail_fast
                or (
                    service is not None
                    and service.
                    config.required
                )
            ):
                break

        return results

    # ---------------------------------------------------------
    # Stop all
    # ---------------------------------------------------------

    def stop_all(
        self,
        runtime: RuntimeStatus,
    ) -> List[
        ServiceOperationResult
    ]:
        results: List[
            ServiceOperationResult
        ] = []

        for name in (
            self.registry.
            shutdown_order()
        ):
            results.append(
                self.stop_service(
                    runtime,
                    name,
                )
            )

        return results


# ---------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------

def registry_from_config(
    config: RuntimeConfig,
    *,
    hooks: Optional[
        Mapping[
            str,
            Mapping[
                str,
                Callable[
                    ...,
                    Any,
                ],
            ],
        ]
    ] = None,
    dependencies: Optional[
        Mapping[
            str,
            Iterable[str],
        ]
    ] = None,
) -> RuntimeServiceRegistry:
    """
    Build a runtime registry from RuntimeConfig.

    hooks example:

        {
            "orchestrator": {
                "start": ...,
                "stop": ...,
                "ready": ...,
            }
        }

    No hook names are invented from strings.
    Callables must be supplied explicitly.
    """

    registry = (
        RuntimeServiceRegistry()
    )

    hook_map = (
        hooks
        or {}
    )

    dependency_map = (
        dependencies
        or {}
    )

    for (
        name,
        service_config,
    ) in sorted(
        config.services.items()
    ):
        service_hooks = (
            hook_map.get(
                name,
                {},
            )
        )

        registry.register(
            RuntimeService(
                name=name,
                dependencies=list(
                    dependency_map.get(
                        name,
                        (),
                    )
                ),
                start_hook=(
                    service_hooks.get(
                        "start"
                    )
                ),
                stop_hook=(
                    service_hooks.get(
                        "stop"
                    )
                ),
                readiness_probe=(
                    service_hooks.get(
                        "ready"
                    )
                ),
                config=(
                    service_config
                ),
            )
        )

    registry.validate_dependencies()

    return registry
