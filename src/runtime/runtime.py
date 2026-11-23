from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

from src.runtime.bootstrap import (
    BootstrapResult,
    RuntimeBootstrap,
)

from src.runtime.config import (
    RuntimeConfig,
    from_environment,
    safe_config_summary,
)

from src.runtime.schema import (
    ReadinessStatus,
    RuntimeState,
    RuntimeStatus,
    ShutdownReason,
)

from src.runtime.services import (
    RuntimeService,
    RuntimeServiceManager,
    RuntimeServiceRegistry,
)

from src.runtime.shutdown import (
    RuntimeShutdown,
    ShutdownResult,
)

from src.runtime.state import (
    RuntimeStateManager,
    recompute_readiness,
)


# ---------------------------------------------------------------
# Runtime snapshot
# ---------------------------------------------------------------

@dataclass
class RuntimeSnapshot:
    """
    Read-only-style representation of the
    current application runtime status plus a
    safe configuration summary.

    The runtime status object itself remains the
    authoritative in-process state.
    """

    status: RuntimeStatus

    config: Dict[
        str,
        Any,
    ]

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "status": (
                self.status.to_dict()
            ),
            "config": dict(
                self.config
            ),
        }


# ---------------------------------------------------------------
# Runtime lifecycle error
# ---------------------------------------------------------------

class RuntimeLifecycleError(
    RuntimeError
):
    pass


# ---------------------------------------------------------------
# OSHA runtime facade
# ---------------------------------------------------------------

class OshaRuntime:
    """
    Public facade for one OSHA application
    runtime lifecycle.

    Responsibilities:

      configuration
      service registration
      bootstrap
      runtime readiness
      status snapshots
      graceful shutdown

    It does not perform infrastructure
    remediation.
    """

    def __init__(
        self,
        *,
        config: Optional[
            RuntimeConfig
        ] = None,
        registry: Optional[
            RuntimeServiceRegistry
        ] = None,
        state: Optional[
            RuntimeStateManager
        ] = None,
    ) -> None:
        self.config = (
            config
            or RuntimeConfig()
        )

        self.config.validate()

        self.registry = (
            registry
            or RuntimeServiceRegistry()
        )

        self.state = (
            state
            or RuntimeStateManager()
        )

        self.services = (
            RuntimeServiceManager(
                self.registry,
                state=self.state,
            )
        )

        self.bootstrapper = (
            RuntimeBootstrap(
                config=self.config,
                registry=self.registry,
                state=self.state,
                services=self.services,
            )
        )

        self.shutdown_manager = (
            RuntimeShutdown(
                services=self.services,
                state=self.state,
            )
        )

        self._status: Optional[
            RuntimeStatus
        ] = None

        self._last_bootstrap: Optional[
            BootstrapResult
        ] = None

        self._last_shutdown: Optional[
            ShutdownResult
        ] = None

    # ---------------------------------------------------------
    # Current status
    # ---------------------------------------------------------

    @property
    def status(
        self,
    ) -> Optional[
        RuntimeStatus
    ]:
        return self._status

    # ---------------------------------------------------------
    # Started state
    # ---------------------------------------------------------

    @property
    def started(
        self,
    ) -> bool:
        return bool(
            self._status is not None
            and self._status.state
            in {
                RuntimeState.
                STARTING.value,

                RuntimeState.
                RUNNING.value,

                RuntimeState.
                DEGRADED.value,
            }
        )

    # ---------------------------------------------------------
    # Readiness
    # ---------------------------------------------------------

    @property
    def ready(
        self,
    ) -> bool:
        if self._status is None:
            return False

        recompute_readiness(
            self._status
        )

        return (
            self._status.readiness
            == ReadinessStatus.
            READY.value
        )

    # ---------------------------------------------------------
    # Service registration
    # ---------------------------------------------------------

    def register(
        self,
        service: RuntimeService,
    ) -> RuntimeService:
        """
        Register an internal runtime service.

        Registration is only allowed before a
        runtime instance has been created.
        """

        if (
            self._status
            is not None
        ):
            raise RuntimeLifecycleError(
                "Runtime services cannot "
                "be registered after a "
                "runtime instance has been "
                "created."
            )

        registered = (
            self.registry.register(
                service
            )
        )

        # Registry contents changed before
        # startup, so recreate the managers that
        # depend on that registry.
        self.services = (
            RuntimeServiceManager(
                self.registry,
                state=self.state,
            )
        )

        self.bootstrapper = (
            RuntimeBootstrap(
                config=self.config,
                registry=self.registry,
                state=self.state,
                services=self.services,
            )
        )

        self.shutdown_manager = (
            RuntimeShutdown(
                services=self.services,
                state=self.state,
            )
        )

        return registered

    # ---------------------------------------------------------
    # Startup
    # ---------------------------------------------------------

    def start(
        self,
        *,
        host_id: Optional[
            str
        ] = None,
        started_by: Optional[
            str
        ] = None,
    ) -> BootstrapResult:
        """
        Bootstrap this runtime exactly once.

        Restarting a terminal runtime should use
        a new OshaRuntime instance so lifecycle
        history remains unambiguous.
        """

        if (
            self._status
            is not None
        ):
            raise RuntimeLifecycleError(
                "Runtime instance already "
                "exists in state "
                f"{self._status.state}."
            )

        result = (
            self.bootstrapper.bootstrap(
                host_id=host_id,
                started_by=started_by,
            )
        )

        self._status = (
            result.runtime
        )

        self._last_bootstrap = (
            result
        )

        return result

    # ---------------------------------------------------------
    # Shutdown
    # ---------------------------------------------------------

    def shutdown(
        self,
        *,
        reason: str = (
            "Runtime shutdown requested."
        ),
        shutdown_reason: str = (
            ShutdownReason.
            REQUESTED.value
        ),
    ) -> ShutdownResult:
        """
        Gracefully stop the current OSHA runtime
        and its registered internal services.
        """

        if self._status is None:
            raise RuntimeLifecycleError(
                "Runtime has not been "
                "created or started."
            )

        result = (
            self.shutdown_manager.
            shutdown(
                self._status,
                reason=reason,
                shutdown_reason=(
                    shutdown_reason
                ),
            )
        )

        self._status = (
            result.runtime
        )

        self._last_shutdown = (
            result
        )

        return result

    # ---------------------------------------------------------
    # Snapshot
    # ---------------------------------------------------------

    def snapshot(
        self,
    ) -> RuntimeSnapshot:
        """
        Return current runtime state with a safe
        runtime-configuration summary.
        """

        if self._status is None:
            raise RuntimeLifecycleError(
                "Runtime has not been "
                "created or started."
            )

        return RuntimeSnapshot(
            status=(
                self._status.finalize()
            ),
            config=(
                safe_config_summary(
                    self.config
                )
            ),
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        """
        Small operational summary suitable for
        local runtime diagnostics.

        This does not claim infrastructure
        health.
        """

        status = (
            self._status
        )

        return {
            "instance_name": (
                self.config.
                instance_name
            ),
            "environment_id": (
                self.config.
                environment_id
            ),
            "dry_run": (
                self.config.dry_run
            ),
            "runtime_created": (
                status is not None
            ),
            "state": (
                status.state
                if status is not None
                else (
                    RuntimeState.
                    CREATED.value
                )
            ),
            "readiness": (
                status.readiness
                if status is not None
                else (
                    ReadinessStatus.
                    UNKNOWN.value
                )
            ),
            "runtime_id": (
                status.runtime_id
                if status is not None
                else None
            ),
            "registered_services": (
                self.registry.names()
            ),
            "started": self.started,
            "ready": (
                self.ready
                if status is not None
                else False
            ),
        }

    # ---------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------

    def __enter__(
        self,
    ) -> "OshaRuntime":
        """
        Support:

            with OshaRuntime(...) as runtime:
                ...

        The runtime starts on entry and attempts
        controlled shutdown on exit.
        """

        self.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:
        if (
            self._status
            is not None
            and self._status.state
            != RuntimeState.
            STOPPED.value
        ):
            shutdown_reason = (
                ShutdownReason.
                FAILURE.value
                if exc_type is not None
                else (
                    ShutdownReason.
                    REQUESTED.value
                )
            )

            try:
                self.shutdown(
                    reason=(
                        "Runtime context "
                        "exited after an "
                        "exception."
                        if (
                            exc_type
                            is not None
                        )
                        else (
                            "Runtime context "
                            "exited normally."
                        )
                    ),
                    shutdown_reason=(
                        shutdown_reason
                    ),
                )

            except Exception:
                # Never hide the original
                # exception if cleanup also
                # fails.
                if exc_type is None:
                    raise

        # Returning False lets any original
        # exception propagate.
        return False


# ---------------------------------------------------------------
# Environment-backed runtime
# ---------------------------------------------------------------

def runtime_from_environment(
    *,
    environ: Optional[
        Mapping[
            str,
            str,
        ]
    ] = None,
    registry: Optional[
        RuntimeServiceRegistry
    ] = None,
) -> OshaRuntime:
    """
    Construct an OSHA runtime from supported
    OSHA_* environment settings.

    This constructs the runtime facade only.
    Call start() explicitly when appropriate.
    """

    config = from_environment(
        environ=environ
    )

    return OshaRuntime(
        config=config,
        registry=registry,
    )
