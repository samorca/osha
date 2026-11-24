from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.orchestrator.schema import (
    OrchestrationRun,
    OrchestrationState,
    StageName,
    StageStatus,
    utc_now,
)


# ---------------------------------------------------------------
# Health status
# ---------------------------------------------------------------


class HealthStatus(
    str,
    Enum,
):
    READY = "ready"

    DEGRADED = "degraded"

    UNAVAILABLE = "unavailable"

    UNKNOWN = "unknown"


# ---------------------------------------------------------------
# Component health
# ---------------------------------------------------------------


@dataclass(frozen=True)
class ComponentHealth:
    """
    Readiness result for one orchestrator
    dependency.

    This describes orchestration capability,
    not infrastructure health.
    """

    component: str

    status: str

    required: bool = True

    message: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    checked_at: str = field(default_factory=utc_now)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "component": (self.component),
            "status": self.status,
            "required": (self.required),
            "message": (self.message),
            "attributes": dict(self.attributes),
            "checked_at": (self.checked_at),
        }


# ---------------------------------------------------------------
# Overall orchestrator health
# ---------------------------------------------------------------


@dataclass
class OrchestratorHealth:
    """
    Aggregate readiness of the orchestration
    layer.
    """

    status: str = HealthStatus.UNKNOWN.value

    components: List[ComponentHealth] = field(default_factory=list)

    registered_stages: List[str] = field(default_factory=list)

    missing_required_stages: List[str] = field(default_factory=list)

    active_run_count: int = 0

    failed_run_count: int = 0

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    checked_at: str = field(default_factory=utc_now)

    def finalize(
        self,
    ) -> "OrchestratorHealth":
        self.registered_stages = sorted(set(self.registered_stages))

        self.missing_required_stages = sorted(set(self.missing_required_stages))

        self.active_run_count = max(
            0,
            int(self.active_run_count),
        )

        self.failed_run_count = max(
            0,
            int(self.failed_run_count),
        )

        # Required unavailable components make
        # the orchestrator unavailable.
        if any(
            (item.required and item.status == HealthStatus.UNAVAILABLE.value)
            for item in self.components
        ):
            self.status = HealthStatus.UNAVAILABLE.value

        # Missing required capabilities or
        # degraded probes produce DEGRADED.
        elif self.missing_required_stages or any(
            item.status == HealthStatus.DEGRADED.value for item in self.components
        ):
            self.status = HealthStatus.DEGRADED.value

        elif self.components and all(
            item.status == HealthStatus.READY.value for item in self.components
        ):
            self.status = HealthStatus.READY.value

        elif not self.components and not self.missing_required_stages:
            self.status = HealthStatus.READY.value

        else:
            self.status = HealthStatus.UNKNOWN.value

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "status": self.status,
            "components": [item.to_dict() for item in self.components],
            "registered_stages": (self.registered_stages),
            "missing_required_stages": (self.missing_required_stages),
            "active_run_count": (self.active_run_count),
            "failed_run_count": (self.failed_run_count),
            "attributes": dict(self.attributes),
            "checked_at": (self.checked_at),
        }


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class HealthConfig:
    """
    Orchestrator readiness policy.

    Collection and telemetry are not mandatory
    here because some entry points may start
    from already-normalized telemetry or an
    existing incident.
    """

    required_stages: tuple[
        str,
        ...,
    ] = (
        StageName.CORRELATION.value,
        StageName.INTELLIGENCE.value,
        StageName.DECISION.value,
        StageName.REMEDIATION.value,
        StageName.SELF_HEALING.value,
    )

    fail_on_probe_exception: bool = False

    def validate(
        self,
    ) -> None:
        valid = {item.value for item in StageName}

        unknown = sorted(set(self.required_stages) - valid)

        if unknown:
            raise ValueError(f"Unknown required stages: {unknown}")


# ---------------------------------------------------------------
# Probe contract
# ---------------------------------------------------------------

HealthProbe = Callable[
    [],
    ComponentHealth,
]


# ---------------------------------------------------------------
# Handler readiness
# ---------------------------------------------------------------


def handler_health(
    handlers: Mapping[
        str,
        Any,
    ],
    *,
    required_stages: Iterable[str],
) -> List[ComponentHealth]:
    """
    Determine whether required stage handlers
    have been registered.

    Registration does not prove that the
    underlying infrastructure is healthy.
    """

    registered = set(handlers)

    required = set(required_stages)

    return [
        ComponentHealth(
            component=(f"stage:{stage}"),
            status=(
                HealthStatus.READY.value
                if stage in registered
                else HealthStatus.UNAVAILABLE.value
            ),
            required=(stage in required),
            message=(
                "Stage handler is registered."
                if stage in registered
                else ("Required stage handler is not registered.")
            ),
            attributes={
                "stage": stage,
            },
        )
        for stage in sorted(required)
    ]


# ---------------------------------------------------------------
# Run statistics
# ---------------------------------------------------------------


def run_health(
    runs: Iterable[OrchestrationRun],
) -> Dict[str, int]:
    """
    Summarize orchestration run states.

    These values describe orchestrator activity,
    not external infrastructure health.
    """

    values = list(runs)

    active_states = {
        OrchestrationState.CREATED.value,
        OrchestrationState.QUEUED.value,
        OrchestrationState.RUNNING.value,
        OrchestrationState.WAITING.value,
        OrchestrationState.PAUSED.value,
    }

    return {
        "run_count": len(values),
        "active_run_count": sum(1 for run in values if run.state in active_states),
        "failed_run_count": sum(
            1 for run in values if (run.state == OrchestrationState.FAILED.value)
        ),
        "escalated_run_count": (
            sum(
                1 for run in values if (run.state == OrchestrationState.ESCALATED.value)
            )
        ),
    }


# ---------------------------------------------------------------
# Health evaluation
# ---------------------------------------------------------------


def assess_health(
    *,
    handlers: Mapping[
        str,
        Any,
    ],
    runs: Iterable[OrchestrationRun] = (),
    probes: Iterable[HealthProbe] = (),
    config: Optional[HealthConfig] = None,
) -> OrchestratorHealth:
    """
    Assess orchestrator readiness.

    Optional health probes can represent internal
    subsystem readiness checks.

    Probe failure does not expose arbitrary
    exception text.
    """

    configuration = config or HealthConfig()

    configuration.validate()

    registered = sorted(set(handlers))

    missing = sorted(set(configuration.required_stages) - set(registered))

    components = handler_health(
        handlers,
        required_stages=(configuration.required_stages),
    )

    # ---------------------------------------------------------
    # Optional probes
    # ---------------------------------------------------------

    for probe in probes:
        try:
            result = probe()

            if not isinstance(
                result,
                ComponentHealth,
            ):
                raise TypeError("Health probe must return ComponentHealth")

            components.append(result)

        except Exception as exc:
            components.append(
                ComponentHealth(
                    component=(
                        getattr(
                            probe,
                            "__name__",
                            "health_probe",
                        )
                    ),
                    status=(
                        HealthStatus.UNAVAILABLE.value
                        if configuration.fail_on_probe_exception
                        else (HealthStatus.DEGRADED.value)
                    ),
                    required=(configuration.fail_on_probe_exception),
                    message=("Health probe failed."),
                    attributes={
                        "error_type": (type(exc).__name__),
                    },
                )
            )

    counts = run_health(runs)

    return OrchestratorHealth(
        components=components,
        registered_stages=(registered),
        missing_required_stages=(missing),
        active_run_count=(counts["active_run_count"]),
        failed_run_count=(counts["failed_run_count"]),
        attributes={
            "run_count": (counts["run_count"]),
            "escalated_run_count": (counts["escalated_run_count"]),
            "scope": ("orchestrator_readiness"),
            "infrastructure_health_claimed": (False),
        },
    ).finalize()


# ---------------------------------------------------------------
# Stage failure summary
# ---------------------------------------------------------------


def recent_stage_failures(
    runs: Iterable[OrchestrationRun],
) -> Dict[str, int]:
    """
    Count recorded failed stages across the
    supplied orchestration runs.

    The caller defines which runs are considered
    recent.
    """

    failures: Dict[
        str,
        int,
    ] = {}

    for run in runs:
        for stage in run.stages:
            if stage.status == StageStatus.FAILED.value:
                failures[stage.stage] = (
                    failures.get(
                        stage.stage,
                        0,
                    )
                    + 1
                )

    return failures


# ---------------------------------------------------------------
# Health checker facade
# ---------------------------------------------------------------


class OrchestratorHealthChecker:
    """
    Read-only readiness evaluator.

    It cannot remediate infrastructure or mutate
    orchestration runs.
    """

    def __init__(
        self,
        *,
        handlers: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        probes: Iterable[HealthProbe] = (),
        config: Optional[HealthConfig] = None,
    ) -> None:
        self.handlers = dict(handlers or {})

        self.probes = list(probes)

        self.config = config or HealthConfig()

        self.config.validate()

    def check(
        self,
        *,
        runs: Iterable[OrchestrationRun] = (),
    ) -> OrchestratorHealth:
        return assess_health(
            handlers=self.handlers,
            runs=runs,
            probes=self.probes,
            config=self.config,
        )

    def ready(
        self,
        *,
        runs: Iterable[OrchestrationRun] = (),
    ) -> bool:
        return self.check(runs=runs).status == HealthStatus.READY.value
