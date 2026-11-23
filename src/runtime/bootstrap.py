from __future__ import annotations

import os
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

from src.runtime.config import (
    RuntimeConfig,
    safe_config_summary,
)

from src.runtime.schema import (
    ReadinessStatus,
    RuntimeContext,
    RuntimeEvidence,
    RuntimeState,
    RuntimeStatus,
    ServiceState,
)

from src.runtime.services import (
    RuntimeServiceManager,
    RuntimeServiceRegistry,
    ServiceOperationResult,
)

from src.runtime.state import (
    RuntimeStateManager,
    recompute_readiness,
)


# ---------------------------------------------------------------
# Bootstrap result
# ---------------------------------------------------------------

@dataclass
class BootstrapResult:
    runtime: RuntimeStatus

    service_results: List[
        ServiceOperationResult
    ] = field(
        default_factory=list
    )

    success: bool = False

    degraded: bool = False

    failed_service_names: List[
        str
    ] = field(
        default_factory=list
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "success": self.success,
            "degraded": (
                self.degraded
            ),
            "failed_service_names": (
                sorted(
                    set(
                        self.
                        failed_service_names
                    )
                )
            ),
            "service_results": [
                {
                    "name": item.name,
                    "success": (
                        item.success
                    ),
                    "ready": (
                        item.ready
                    ),
                    "reason": (
                        item.reason
                    ),
                    "error_type": (
                        item.error_type
                    ),
                }
                for item
                in self.service_results
            ],
            "runtime": (
                self.runtime.to_dict()
            ),
        }


# ---------------------------------------------------------------
# Runtime construction
# ---------------------------------------------------------------

def build_runtime_status(
    config: RuntimeConfig,
    registry: RuntimeServiceRegistry,
    *,
    host_id: Optional[
        str
    ] = None,
    started_by: Optional[
        str
    ] = None,
) -> RuntimeStatus:
    """
    Construct a new runtime instance and attach
    registered service statuses.

    This does not start any services.
    """

    config.validate()

    context = RuntimeContext(
        environment_id=(
            config.environment_id
        ),
        instance_name=(
            config.instance_name
        ),
        dry_run=config.dry_run,
        process_id=os.getpid(),
        host_id=host_id,
        started_by=started_by,
        attributes={
            "runtime_config": (
                safe_config_summary(
                    config
                )
            ),
        },
    ).finalize()

    runtime = RuntimeStatus(
        context=context,
        state=(
            RuntimeState.CREATED.value
        ),
        readiness=(
            ReadinessStatus.
            UNKNOWN.value
        ),
        attributes={
            "fail_fast": (
                config.fail_fast
            ),
            "startup_timeout_seconds": (
                config.
                startup_timeout_seconds
            ),
        },
    ).finalize()

    registry.attach_statuses(
        runtime
    )

    return runtime.finalize()


# ---------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------

def required_failures(
    runtime: RuntimeStatus,
    results: List[
        ServiceOperationResult
    ],
) -> List[str]:
    failed: List[
        str
    ] = []

    for result in results:
        if result.success:
            continue

        status = runtime.service(
            result.name
        )

        if (
            status is not None
            and status.required
        ):
            failed.append(
                result.name
            )

    return sorted(
        set(
            failed
        )
    )


def optional_failures(
    runtime: RuntimeStatus,
    results: List[
        ServiceOperationResult
    ],
) -> List[str]:
    failed: List[
        str
    ] = []

    for result in results:
        if result.success:
            continue

        status = runtime.service(
            result.name
        )

        if (
            status is not None
            and not status.required
        ):
            failed.append(
                result.name
            )

    return sorted(
        set(
            failed
        )
    )


# ---------------------------------------------------------------
# Runtime bootstrap
# -----------------------------------------

class RuntimeBootstrap:
    """
    Construct and start one OSHA application
    runtime instance.

    Bootstrap only controls internal application
    services through their registered lifecycle
    hooks.
    """

    def __init__(
        self,
        *,
        config: RuntimeConfig,
        registry: RuntimeServiceRegistry,
        state: Optional[
            RuntimeStateManager
        ] = None,
        services: Optional[
            RuntimeServiceManager
        ] = None,
    ) -> None:
        self.config = config

        self.config.validate()

        self.registry = registry

        self.registry.validate_dependencies()

        self.state = (
            state
            or RuntimeStateManager()
        )

        self.services = (
            services
            or RuntimeServiceManager(
                registry,
                state=self.state,
            )
        )

    # ---------------------------------------------------------
    # Bootstrap
    # ---------------------------------------------------------

    def bootstrap(
        self,
        *,
        host_id: Optional[
            str
        ] = None,
        started_by: Optional[
            str
        ] = None,
    ) -> BootstrapResult:
        runtime = (
            build_runtime_status(
                self.config,
                self.registry,
                host_id=host_id,
                started_by=started_by,
            )
        )

        # -----------------------------------------------------
        # Begin runtime startup
        # -----------------------------------------------------

        self.state.start(
            runtime
        )

        runtime.evidence.append(
            RuntimeEvidence(
                evidence_type=(
                    "bootstrap"
                ),
                description=(
                    "Runtime bootstrap "
                    "started."
                ),
                source_id=(
                    runtime.runtime_id
                ),
                attributes={
                    "service_count": (
                        len(
                            runtime.services
                        )
                    ),
                    "dry_run": (
                        runtime.context.
                        dry_run
                    ),
                },
            )
        )

        # -----------------------------------------------------
        # Start internal services
        # -----------------------------------------------------

        results = (
            self.services.start_all(
                runtime,
                fail_fast=(
                    self.config.
                    fail_fast
                ),
            )
        )

        required = required_failures(
            runtime,
            results,
        )

        optional = optional_failures(
            runtime,
            results,
        )

        # -----------------------------------------------------
        # Required startup failure
        # -----------------------------------------------------

        if required:
            self.state.fail(
                runtime,
                reason=(
                    "One or more required "
                    "runtime services failed "
                    "to start."
                ),
                error=(
                    "Required runtime "
                    "service startup failure."
                ),
            )

            runtime.attributes[
                "failed_required_services"
            ] = required

            runtime.attributes[
                "failed_optional_services"
            ] = optional

            return BootstrapResult(
                runtime=(
                    runtime.finalize()
                ),
                service_results=results,
                success=False,
                degraded=False,
                failed_service_names=(
                    required
                    + optional
                ),
            )

        # -----------------------------------------------------
        # Runtime entered RUNNING
        # -----------------------------------------------------

        self.state.running(
            runtime
        )

        recompute_readiness(
            runtime
        )

        # -----------------------------------------------------
        # Optional service failure
        # -----------------------------------------------------

        if optional:
            self.state.degrade(
                runtime,
                reason=(
                    "One or more optional "
                    "runtime services failed "
                    "to start."
                ),
            )

            runtime.attributes[
                "failed_optional_services"
            ] = optional

            return BootstrapResult(
                runtime=(
                    runtime.finalize()
                ),
                service_results=results,
                success=True,
                degraded=True,
                failed_service_names=(
                    optional
                ),
            )

        # -----------------------------------------------------
        # Required service readiness check
        # -----------------------------------------------------

        not_ready_required = [
            service.name
            for service
            in runtime.services
            if (
                service.required
                and service.readiness
                != ReadinessStatus.
                READY.value
                and not bool(
                    service.attributes.get(
                        "disabled",
                        False,
                    )
                )
            )
        ]

        if not_ready_required:
            self.state.degrade(
                runtime,
                reason=(
                    "Runtime started but "
                    "required service "
                    "readiness is incomplete."
                ),
            )

            runtime.attributes[
                "not_ready_required_services"
            ] = (
                not_ready_required
            )

            return BootstrapResult(
                runtime=(
                    runtime.finalize()
                ),
                service_results=results,
                success=True,
                degraded=True,
                failed_service_names=[],
            )

        # -----------------------------------------------------
        # Successful startup
        # -----------------------------------------------------

        runtime.evidence.append(
            RuntimeEvidence(
                evidence_type=(
                    "bootstrap_complete"
                ),
                description=(
                    "Runtime bootstrap "
                    "completed."
                ),
                source_id=(
                    runtime.runtime_id
                ),
                attributes={
                    "readiness": (
                        runtime.readiness
                    ),
                    "started_services": [
                        service.name
                        for service
                        in runtime.services
                        if (
                            service.state
                            == ServiceState.
                            RUNNING.value
                        )
                    ],
                },
            )
        )

        return BootstrapResult(
            runtime=(
                runtime.finalize()
            ),
            service_results=results,
            success=True,
            degraded=False,
            failed_service_names=[],
        )
