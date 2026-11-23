from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Optional,
    Set,
)

from src.runtime.schema import (
    ReadinessStatus,
    RuntimeEvidence,
    RuntimeState,
    RuntimeStatus,
    ServiceState,
    ServiceStatus,
    ShutdownReason,
    utc_now,
)


# ---------------------------------------------------------------
# Runtime transitions
# ---------------------------------------------------------------

RUNTIME_TRANSITIONS: Dict[
    str,
    Set[str],
] = {
    RuntimeState.CREATED.value: {
        RuntimeState.STARTING.value,
        RuntimeState.STOPPED.value,
        RuntimeState.FAILED.value,
    },

    RuntimeState.STARTING.value: {
        RuntimeState.RUNNING.value,
        RuntimeState.DEGRADED.value,
        RuntimeState.STOPPING.value,
        RuntimeState.FAILED.value,
    },

    RuntimeState.RUNNING.value: {
        RuntimeState.DEGRADED.value,
        RuntimeState.STOPPING.value,
        RuntimeState.FAILED.value,
    },

    RuntimeState.DEGRADED.value: {
        RuntimeState.RUNNING.value,
        RuntimeState.STOPPING.value,
        RuntimeState.FAILED.value,
    },

    RuntimeState.STOPPING.value: {
        RuntimeState.STOPPED.value,
        RuntimeState.FAILED.value,
    },

    RuntimeState.STOPPED.value: (
        set()
    ),

    RuntimeState.FAILED.value: {
        RuntimeState.STOPPING.value,
        RuntimeState.STOPPED.value,
    },
}


# ---------------------------------------------------------------
# Service transitions
# ---------------------------------------------------------------

SERVICE_TRANSITIONS: Dict[
    str,
    Set[str],
] = {
    ServiceState.REGISTERED.value: {
        ServiceState.STARTING.value,
        ServiceState.STOPPED.value,
        ServiceState.FAILED.value,
    },

    ServiceState.STARTING.value: {
        ServiceState.RUNNING.value,
        ServiceState.DEGRADED.value,
        ServiceState.STOPPING.value,
        ServiceState.FAILED.value,
    },

    ServiceState.RUNNING.value: {
        ServiceState.DEGRADED.value,
        ServiceState.STOPPING.value,
        ServiceState.FAILED.value,
    },

    ServiceState.DEGRADED.value: {
        ServiceState.RUNNING.value,
        ServiceState.STOPPING.value,
        ServiceState.FAILED.value,
    },

    ServiceState.STOPPING.value: {
        ServiceState.STOPPED.value,
        ServiceState.FAILED.value,
    },

    ServiceState.STOPPED.value: (
        set()
    ),

    ServiceState.FAILED.value: {
        ServiceState.STOPPING.value,
        ServiceState.STOPPED.value,
    },
}


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------

class InvalidRuntimeTransition(
    ValueError
):
    pass


class InvalidServiceTransition(
    ValueError
):
    pass


# ---------------------------------------------------------------
# Transition record
# ---------------------------------------------------------------

@dataclass(frozen=True)
class StateTransition:
    """
    Immutable runtime/service transition record.
    """

    subject: str

    previous_state: str

    next_state: str

    reason: str

    occurred_at: str = field(
        default_factory=utc_now
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "previous_state": (
                self.previous_state
            ),
            "next_state": (
                self.next_state
            ),
            "reason": self.reason,
            "occurred_at": (
                self.occurred_at
            ),
            "attributes": dict(
                self.attributes
            ),
        }


# ---------------------------------------------------------------
# Runtime transition
# ---------------------------------------------------------------

def transition_runtime(
    runtime: RuntimeStatus,
    next_state: str,
    *,
    reason: str,
    shutdown_reason: Optional[
        str
    ] = None,
) -> StateTransition:
    runtime.finalize()

    previous = runtime.state

    if (
        next_state != previous
        and next_state
        not in RUNTIME_TRANSITIONS.get(
            previous,
            set(),
        )
    ):
        raise (
            InvalidRuntimeTransition(
                "Invalid runtime "
                "transition: "
                f"{previous} -> "
                f"{next_state}"
            )
        )

    if shutdown_reason is not None:
        if (
            shutdown_reason
            not in {
                item.value
                for item
                in ShutdownReason
            }
        ):
            raise (
                InvalidRuntimeTransition(
                    "Unknown shutdown "
                    f"reason: "
                    f"{shutdown_reason}"
                )
            )

        runtime.shutdown_reason = (
            shutdown_reason
        )

    event = StateTransition(
        subject="runtime",
        previous_state=previous,
        next_state=next_state,
        reason=reason,
    )

    runtime.state = (
        next_state
    )

    # ---------------------------------------------------------
    # Runtime timestamps/readiness
    # ---------------------------------------------------------

    if (
        next_state
        == RuntimeState.STARTING.value
        and runtime.started_at
        is None
    ):
        runtime.started_at = (
            utc_now()
        )

    if (
        next_state
        == RuntimeState.RUNNING.value
    ):
        runtime.readiness = (
            ReadinessStatus.READY.value
        )

    elif (
        next_state
        == RuntimeState.DEGRADED.value
    ):
        runtime.readiness = (
            ReadinessStatus.
            DEGRADED.value
        )

    elif (
        next_state
        in {
            RuntimeState.CREATED.value,
            RuntimeState.STARTING.value,
            RuntimeState.STOPPING.value,
        }
    ):
        runtime.readiness = (
            ReadinessStatus.
            NOT_READY.value
        )

    elif (
        next_state
        in {
            RuntimeState.STOPPED.value,
            RuntimeState.FAILED.value,
        }
    ):
        runtime.readiness = (
            ReadinessStatus.
            NOT_READY.value
        )

        runtime.stopped_at = (
            utc_now()
        )

    # ---------------------------------------------------------
    # Transition history
    # ---------------------------------------------------------

    runtime.attributes.setdefault(
        "state_transitions",
        [],
    ).append(
        event.to_dict()
    )

    runtime.evidence.append(
        RuntimeEvidence(
            evidence_type=(
                "runtime_transition"
            ),
            description=(
                f"{previous} -> "
                f"{next_state}: "
                f"{reason}"
            ),
            source_id=(
                runtime.runtime_id
            ),
            attributes={
                "previous_state": (
                    previous
                ),
                "next_state": (
                    next_state
                ),
            },
        )
    )

    runtime.finalize()

    return event


# ---------------------------------------------------------------
# Service transition
# ---------------------------------------------------------------

def transition_service(
    runtime: RuntimeStatus,
    service_name: str,
    next_state: str,
    *,
    reason: str,
    error: Optional[
        str
    ] = None,
) -> StateTransition:
    runtime.finalize()

    service = runtime.service(
        service_name
    )

    if service is None:
        raise InvalidServiceTransition(
            "Unknown runtime service: "
            f"{service_name}"
        )

    previous = service.state

    if (
        next_state != previous
        and next_state
        not in SERVICE_TRANSITIONS.get(
            previous,
            set(),
        )
    ):
        raise (
            InvalidServiceTransition(
                "Invalid service "
                "transition: "
                f"{service_name}: "
                f"{previous} -> "
                f"{next_state}"
            )
        )

    event = StateTransition(
        subject=(
            f"service:{service_name}"
        ),
        previous_state=previous,
        next_state=next_state,
        reason=reason,
    )

    service.state = (
        next_state
    )

    if error is not None:
        service.last_error = (
            error
        )

    # ---------------------------------------------------------
    # Service state/readiness
    # ---------------------------------------------------------

    if (
        next_state
        == ServiceState.
        STARTING.value
    ):
        if service.started_at is None:
            service.started_at = (
                utc_now()
            )

        service.readiness = (
            ReadinessStatus.
            NOT_READY.value
        )

    elif (
        next_state
        == ServiceState.
        RUNNING.value
    ):
        service.readiness = (
            ReadinessStatus.
            READY.value
        )

    elif (
        next_state
        == ServiceState.
        DEGRADED.value
    ):
        service.readiness = (
            ReadinessStatus.
            DEGRADED.value
        )

    elif (
        next_state
        == ServiceState.
        STOPPING.value
    ):
        service.readiness = (
            ReadinessStatus.
            NOT_READY.value
        )

    elif (
        next_state
        in {
            ServiceState.
            STOPPED.value,
            ServiceState.
            FAILED.value,
        }
    ):
        service.readiness = (
            ReadinessStatus.
            NOT_READY.value
        )

        service.stopped_at = (
            utc_now()
        )

    service.attributes.setdefault(
        "state_transitions",
        [],
    ).append(
        event.to_dict()
    )

    runtime.evidence.append(
        RuntimeEvidence(
            evidence_type=(
                "service_transition"
            ),
            description=(
                f"{service_name}: "
                f"{previous} -> "
                f"{next_state}: "
                f"{reason}"
            ),
            source_id=(
                runtime.runtime_id
            ),
            attributes={
                "service": (
                    service_name
                ),
                "previous_state": (
                    previous
                ),
                "next_state": (
                    next_state
                ),
            },
        )
    )

    runtime.finalize()

    return event


# ---------------------------------------------------------------
# Readiness calculation
# ---------------------------------------------------------------

def recompute_readiness(
    runtime: RuntimeStatus,
) -> str:
    """
    Calculate application-runtime readiness from
    required internal services.

    This is NOT an infrastructure health
    conclusion.
    """

    runtime.finalize()

    required = [
        service
        for service
        in runtime.services
        if service.required
    ]

    if (
        runtime.state
        in {
            RuntimeState.STOPPING.value,
            RuntimeState.STOPPED.value,
            RuntimeState.FAILED.value,
        }
    ):
        runtime.readiness = (
            ReadinessStatus.
            NOT_READY.value
        )

    elif any(
        service.state
        == ServiceState.FAILED.value
        for service
        in required
    ):
        runtime.readiness = (
            ReadinessStatus.
            NOT_READY.value
        )

    elif any(
        service.readiness
        == ReadinessStatus.
        DEGRADED.value
        for service
        in required
    ):
        runtime.readiness = (
            ReadinessStatus.
            DEGRADED.value
        )

    elif (
        required
        and all(
            service.readiness
            == ReadinessStatus.
            READY.value
            for service
            in required
        )
    ):
        runtime.readiness = (
            ReadinessStatus.
            READY.value
        )

    elif (
        not required
        and runtime.state
        == RuntimeState.
        RUNNING.value
    ):
        runtime.readiness = (
            ReadinessStatus.
            READY.value
        )

    else:
        runtime.readiness = (
            ReadinessStatus.
            NOT_READY.value
        )

    runtime.finalize()

    return runtime.readiness


# ---------------------------------------------------------------
# State manager
# ---------------------------------------------------------------

class RuntimeStateManager:
    """
    Controlled manager for runtime and internal
    service state.

    This module changes runtime metadata only.
    """

    def start(
        self,
        runtime: RuntimeStatus,
    ) -> RuntimeStatus:
        transition_runtime(
            runtime,
            RuntimeState.
            STARTING.value,
            reason=(
                "Runtime startup began."
            ),
        )

        return runtime

    def running(
        self,
        runtime: RuntimeStatus,
    ) -> RuntimeStatus:
        transition_runtime(
            runtime,
            RuntimeState.RUNNING.value,
            reason=(
                "Runtime startup "
                "completed."
            ),
        )

        recompute_readiness(
            runtime
        )

        return runtime

    def degrade(
        self,
        runtime: RuntimeStatus,
        *,
        reason: str,
    ) -> RuntimeStatus:
        transition_runtime(
            runtime,
            RuntimeState.
            DEGRADED.value,
            reason=reason,
        )

        recompute_readiness(
            runtime
        )

        return runtime

    def fail(
        self,
        runtime: RuntimeStatus,
        *,
        reason: str,
        error: Optional[
            str
        ] = None,
    ) -> RuntimeStatus:
        runtime.error = error

        transition_runtime(
            runtime,
            RuntimeState.FAILED.value,
            reason=reason,
            shutdown_reason=(
                ShutdownReason.
                FAILURE.value
            ),
        )

        return runtime

    def stopping(
        self,
        runtime: RuntimeStatus,
        *,
        reason: str,
        shutdown_reason: str = (
            ShutdownReason.
            REQUESTED.value
        ),
    ) -> RuntimeStatus:
        transition_runtime(
            runtime,
            RuntimeState.
            STOPPING.value,
            reason=reason,
            shutdown_reason=(
                shutdown_reason
            ),
        )

        return runtime

    def stopped(
        self,
        runtime: RuntimeStatus,
        *,
        reason: str,
    ) -> RuntimeStatus:
        transition_runtime(
            runtime,
            RuntimeState.STOPPED.value,
            reason=reason,
        )

        return runtime

    # ---------------------------------------------------------
    # Service helpers
    # ---------------------------------------------------------

    def service_starting(
        self,
        runtime: RuntimeStatus,
        name: str,
    ) -> ServiceStatus:
        transition_service(
            runtime,
            name,
            ServiceState.
            STARTING.value,
            reason=(
                "Service startup began."
            ),
        )

        service = runtime.service(
            name
        )

        if service is None:
            raise InvalidServiceTransition(
                f"Unknown service: {name}"
            )

        return service

    def service_running(
        self,
        runtime: RuntimeStatus,
        name: str,
    ) -> ServiceStatus:
        transition_service(
            runtime,
            name,
            ServiceState.RUNNING.value,
            reason=(
                "Service startup "
                "completed."
            ),
        )

        recompute_readiness(
            runtime
        )

        service = runtime.service(
            name
        )

        if service is None:
            raise InvalidServiceTransition(
                f"Unknown service: {name}"
            )

        return service

    def service_failed(
        self,
        runtime: RuntimeStatus,
        name: str,
        *,
        reason: str,
        error: Optional[
            str
        ] = None,
    ) -> ServiceStatus:
        transition_service(
            runtime,
            name,
            ServiceState.FAILED.value,
            reason=reason,
            error=error,
        )

        recompute_readiness(
            runtime
        )

        service = runtime.service(
            name
        )

        if service is None:
            raise InvalidServiceTransition(
                f"Unknown service: {name}"
            )

        return service

    def service_stopping(
        self,
        runtime: RuntimeStatus,
        name: str,
    ) -> ServiceStatus:
        transition_service(
            runtime,
            name,
            ServiceState.STOPPING.value,
            reason=(
                "Service shutdown began."
            ),
        )

        service = runtime.service(
            name
        )

        if service is None:
            raise InvalidServiceTransition(
                f"Unknown service: {name}"
            )

        return service

    def service_stopped(
        self,
        runtime: RuntimeStatus,
        name: str,
    ) -> ServiceStatus:
        transition_service(
            runtime,
            name,
            ServiceState.STOPPED.value,
            reason=(
                "Service shutdown "
                "completed."
            ),
        )

        recompute_readiness(
            runtime
        )

        service = runtime.service(
            name
        )

        if service is None:
            raise InvalidServiceTransition(
                f"Unknown service: {name}"
            )

        return service
