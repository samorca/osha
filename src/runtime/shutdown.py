from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from src.runtime.schema import (
    RuntimeEvidence,
    RuntimeState,
    RuntimeStatus,
    ServiceState,
    ShutdownReason,
)

from src.runtime.services import (
    RuntimeServiceManager,
    ServiceOperationResult,
)

from src.runtime.state import (
    RuntimeStateManager,
)


# ---------------------------------------------------------------
# Shutdown result
# ---------------------------------------------------------------


@dataclass
class ShutdownResult:
    """
    Result from shutting down one OSHA runtime.
    """

    runtime: RuntimeStatus

    service_results: List[ServiceOperationResult] = field(default_factory=list)

    success: bool = False

    failed_service_names: List[str] = field(default_factory=list)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "success": self.success,
            "failed_service_names": (sorted(set(self.failed_service_names))),
            "service_results": [
                {
                    "name": item.name,
                    "success": (item.success),
                    "ready": (item.ready),
                    "reason": (item.reason),
                    "error_type": (item.error_type),
                }
                for item in self.service_results
            ],
            "runtime": (self.runtime.to_dict()),
        }


# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------


def valid_shutdown_reason(
    value: str,
) -> str:
    if value not in {item.value for item in ShutdownReason}:
        raise ValueError(f"Unknown shutdown reason: {value}")

    return value


# ---------------------------------------------------------------
# Failure extraction
# ---------------------------------------------------------------


def shutdown_failures(
    results: List[ServiceOperationResult],
) -> List[str]:
    return sorted(set(item.name for item in results if not item.success))


# ---------------------------------------------------------------
# Runtime shutdown
# ---------------------------------------------------------------


class RuntimeShutdown:
    """
    Gracefully stop one OSHA application runtime
    instance.

    Services stop in the reverse of their startup
    dependency order.

    This module manages OSHA's own application
    runtime. It does not shut down monitored
    infrastructure.
    """

    def __init__(
        self,
        *,
        services: RuntimeServiceManager,
        state: Optional[RuntimeStateManager] = None,
    ) -> None:
        self.services = services

        self.state = state or services.state

    def shutdown(
        self,
        runtime: RuntimeStatus,
        *,
        reason: str = ("Runtime shutdown requested."),
        shutdown_reason: str = (ShutdownReason.REQUESTED.value),
    ) -> ShutdownResult:
        runtime.finalize()

        shutdown_reason = valid_shutdown_reason(shutdown_reason)

        # -----------------------------------------------------
        # Already stopped
        # -----------------------------------------------------

        if runtime.state == RuntimeState.STOPPED.value:
            return ShutdownResult(
                runtime=runtime,
                success=True,
            )

        # -----------------------------------------------------
        # Valid starting states
        # -----------------------------------------------------

        if runtime.state not in {
            RuntimeState.CREATED.value,
            RuntimeState.STARTING.value,
            RuntimeState.RUNNING.value,
            RuntimeState.DEGRADED.value,
            RuntimeState.FAILED.value,
        }:
            raise ValueError(
                f"Runtime cannot begin shutdown from state {runtime.state}"
            )

        # -----------------------------------------------------
        # Runtime was never started
        # -----------------------------------------------------

        if runtime.state == RuntimeState.CREATED.value:
            for service in runtime.services:
                if service.state == ServiceState.REGISTERED.value:
                    service.state = ServiceState.STOPPED.value

            runtime.shutdown_reason = shutdown_reason

            self.state.stopped(
                runtime,
                reason=("Runtime stopped before startup began."),
            )

            return ShutdownResult(
                runtime=(runtime.finalize()),
                success=True,
            )

        # -----------------------------------------------------
        # Enter STOPPING
        # -----------------------------------------------------

        if runtime.state != RuntimeState.FAILED.value:
            self.state.stopping(
                runtime,
                reason=reason,
                shutdown_reason=(shutdown_reason),
            )

        else:
            # FAILED is allowed to transition to
            # STOPPING so resources can still be
            # cleaned up.
            runtime.shutdown_reason = shutdown_reason

            self.state.stopping(
                runtime,
                reason=("Failed runtime is entering controlled shutdown."),
                shutdown_reason=(shutdown_reason),
            )

        runtime.evidence.append(
            RuntimeEvidence(
                evidence_type=("shutdown"),
                description=("Runtime shutdown started."),
                source_id=(runtime.runtime_id),
                attributes={
                    "shutdown_reason": (shutdown_reason),
                    "service_count": (len(runtime.services)),
                },
            )
        )

        # -----------------------------------------------------
        # Stop services
        # -----------------------------------------------------

        results = self.services.stop_all(runtime)

        failures = shutdown_failures(results)

        # -----------------------------------------------------
        # Shutdown failure
        # -----------------------------------------------------

        if failures:
            runtime.attributes["shutdown_failed_services"] = failures

            self.state.fail(
                runtime,
                reason=("One or more runtime services failed during shutdown."),
                error=("Runtime service shutdown failure."),
            )

            return ShutdownResult(
                runtime=(runtime.finalize()),
                service_results=results,
                success=False,
                failed_service_names=(failures),
            )

        # -----------------------------------------------------
        # Successful shutdown
        # -----------------------------------------------------

        self.state.stopped(
            runtime,
            reason=("Runtime shutdown completed."),
        )

        runtime.evidence.append(
            RuntimeEvidence(
                evidence_type=("shutdown_complete"),
                description=("Runtime shutdown completed."),
                source_id=(runtime.runtime_id),
                attributes={
                    "shutdown_reason": (runtime.shutdown_reason),
                    "stopped_services": [
                        service.name
                        for service in runtime.services
                        if (service.state == ServiceState.STOPPED.value)
                    ],
                },
            )
        )

        return ShutdownResult(
            runtime=(runtime.finalize()),
            service_results=results,
            success=True,
            failed_service_names=[],
        )
