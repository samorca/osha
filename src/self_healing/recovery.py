from __future__ import annotations

import math
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.remediation.schema import (
    RemediationMode,
    RemediationResult,
    VerificationStatus,
)

from src.self_healing.schema import (
    HealingEvidence,
    RecoveryAssessment,
    RecoveryStatus,
    checked_score,
    utc_now,
)


# ---------------------------------------------------------------
# Recovery observations
# ---------------------------------------------------------------

@dataclass(frozen=True)
class RecoveryObservation:
    """
    One independent post-remediation health
    observation.

    These observations should normally originate
    from collectors, telemetry, health checks or
    monitoring systems, not solely from the
    execution adapter.
    """

    check_id: str

    name: str

    resource_id: str

    healthy: Optional[
        bool
    ]

    score: float = 0.0

    source_id: Optional[
        str
    ] = None

    description: Optional[
        str
    ] = None

    service_id: Optional[
        str
    ] = None

    attributes: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    observed_at: str = field(
        default_factory=utc_now
    )

    def normalized_score(
        self,
    ) -> float:
        return checked_score(
            self.score
        )

    def to_evidence(
        self,
    ) -> HealingEvidence:
        return HealingEvidence(
            evidence_type=(
                "recovery_observation"
            ),
            description=(
                self.description
                or self.name
            ),
            score=(
                self.normalized_score()
            ),
            source_id=(
                self.source_id
            ),
            resource_ids=(
                [
                    self.resource_id
                ]
                if self.resource_id
                else []
            ),
            attributes={
                "check_id": (
                    self.check_id
                ),
                "healthy": (
                    self.healthy
                ),
                "service_id": (
                    self.service_id
                ),
                **dict(
                    self.attributes
                ),
            },
            observed_at=(
                self.observed_at
            ),
        )


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class RecoveryConfig:
    """
    Recovery-assessment configuration.

    These values are OSHA development thresholds,
    not calibrated universal health standards.
    """

    minimum_observations: int = 1

    minimum_resolved_observations: int = 1

    recovered_score_threshold: float = 0.75

    partial_score_threshold: float = 0.40

    minimum_healthy_ratio: float = 1.0

    fail_on_any_critical_failure: bool = True

    require_all_target_resources_observed: bool = (
        True
    )

    allow_dry_run_recovery_claim: bool = False

    def validate(
        self,
    ) -> None:
        if self.minimum_observations < 1:
            raise ValueError(
                "minimum_observations "
                "must be >= 1"
            )

        if (
            self.
            minimum_resolved_observations
            < 1
        ):
            raise ValueError(
                "minimum_resolved_observations "
                "must be >= 1"
            )

        for name, value in (
            (
                "recovered_score_threshold",
                self.
                recovered_score_threshold,
            ),
            (
                "partial_score_threshold",
                self.
                partial_score_threshold,
            ),
            (
                "minimum_healthy_ratio",
                self.minimum_healthy_ratio,
            ),
        ):
            if not (
                0.0
                <= value
                <= 1.0
            ):
                raise ValueError(
                    f"{name} must be "
                    "between 0 and 1"
                )

        if (
            self.partial_score_threshold
            > self.
            recovered_score_threshold
        ):
            raise ValueError(
                "partial_score_threshold "
                "must not exceed "
                "recovered_score_threshold"
            )


# ---------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------

def finite_score(
    value: Any,
) -> Optional[float]:
    if isinstance(
        value,
        bool,
    ):
        return None

    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        number
    ):
        return None

    return checked_score(
        number
    )


def aggregate_health(
    observations: Iterable[
        RecoveryObservation
    ],
) -> Optional[float]:
    scores = []

    for observation in observations:
        value = finite_score(
            observation.score
        )

        if value is not None:
            scores.append(
                value
            )

    if not scores:
        return None

    return checked_score(
        sum(scores)
        / len(scores)
    )


def healthy_ratio(
    observations: Iterable[
        RecoveryObservation
    ],
) -> Optional[float]:
    resolved = [
        item
        for item in observations
        if item.healthy is not None
    ]

    if not resolved:
        return None

    return (
        sum(
            1
            for item in resolved
            if item.healthy
        )
        / len(resolved)
    )


def resource_health(
    observations: Iterable[
        RecoveryObservation
    ],
) -> Dict[
    str,
    Optional[bool],
]:
    """
    Aggregate health by resource.

    Any explicit failed observation makes that
    resource unhealthy. A resource is healthy
    only if its resolved observations all pass.
    """

    grouped: Dict[
        str,
        List[
            RecoveryObservation
        ],
    ] = {}

    for observation in observations:
        if observation.resource_id:
            grouped.setdefault(
                observation.resource_id,
                [],
            ).append(
                observation
            )

    result: Dict[
        str,
        Optional[bool],
    ] = {}

    for resource_id, values in (
        grouped.items()
    ):
        states = [
            item.healthy
            for item in values
            if item.healthy is not None
        ]

        if not states:
            result[
                resource_id
            ] = None

        elif any(
            state is False
            for state in states
        ):
            result[
                resource_id
            ] = False

        else:
            result[
                resource_id
            ] = True

    return result


# ---------------------------------------------------------------
# Main assessment
# ---------------------------------------------------------------

def assess_recovery(
    remediation: RemediationResult,
    *,
    observations: Iterable[
        RecoveryObservation
    ] = (),
    health_before: Optional[
        float
    ] = None,
    target_resource_ids: Iterable[
        str
    ] = (),
    config: Optional[
        RecoveryConfig
    ] = None,
) -> RecoveryAssessment:
    """
    Assess actual recovery from independent
    observations.

    Remediation execution success is a
    prerequisite, not proof of recovery.
    """

    configuration = (
        config
        or RecoveryConfig()
    )

    configuration.validate()

    remediation.finalize()

    values = list(
        observations
    )

    targets = sorted(
        set(
            target_resource_ids
            or remediation.request.
            target_resource_ids
        )
    )

    before = finite_score(
        health_before
    )

    # ---------------------------------------------------------
    # Dry-run boundary
    # ---------------------------------------------------------

    if (
        remediation.request.mode
        == RemediationMode.
        DRY_RUN.value
        and not configuration.
        allow_dry_run_recovery_claim
    ):
        return RecoveryAssessment(
            status=(
                RecoveryStatus.
                MONITORING.value
            ),
            score=0.0,
            health_before=before,
            health_current=None,
            checks=[
                (
                    "dry_run_"
                    "no_recovery_claim"
                )
            ],
            failed_checks=[],
            evidence=[
                HealingEvidence(
                    evidence_type=(
                        "recovery_boundary"
                    ),
                    description=(
                        "Dry-run remediation "
                        "did not change "
                        "infrastructure, so "
                        "recovery cannot be "
                        "claimed."
                    ),
                    score=0.0,
                    source_id=(
                        remediation.
                        remediation_id
                    ),
                    resource_ids=targets,
                )
            ],
            attributes={
                "dry_run": True,
                "observation_count": (
                    len(values)
                ),
                "recovery_claimed": (
                    False
                ),
            },
        ).finalize()

    # ---------------------------------------------------------
    # Execution boundary
    # ---------------------------------------------------------

    if (
        remediation.execution.status
        not in {
            "succeeded",
            "dry_run",
        }
    ):
        return RecoveryAssessment(
            status=(
                RecoveryStatus.
                FAILED.value
            ),
            score=0.0,
            health_before=before,
            health_current=None,
            unhealthy_resource_ids=(
                targets
            ),
            checks=[
                "remediation_execution"
            ],
            failed_checks=[
                "remediation_execution"
            ],
            evidence=[
                HealingEvidence(
                    evidence_type=(
                        "recovery_boundary"
                    ),
                    description=(
                        "Remediation execution "
                        "did not succeed, so "
                        "recovery is not "
                        "established."
                    ),
                    score=0.0,
                    source_id=(
                        remediation.
                        remediation_id
                    ),
                    resource_ids=targets,
                    attributes={
                        "execution_status": (
                            remediation.
                            execution.status
                        ),
                    },
                )
            ],
        ).finalize()

    # ---------------------------------------------------------
    # Observation availability
    # ---------------------------------------------------------

    if (
        len(values)
        < configuration.
        minimum_observations
    ):
        return RecoveryAssessment(
            status=(
                RecoveryStatus.
                MONITORING.value
            ),
            score=0.0,
            health_before=before,
            health_current=None,
            checks=[
                item.check_id
                for item in values
            ],
            failed_checks=[
                "insufficient_observations"
            ],
            evidence=[
                item.to_evidence()
                for item in values
            ],
            attributes={
                "observation_count": (
                    len(values)
                ),
                "minimum_observations": (
                    configuration.
                    minimum_observations
                ),
            },
        ).finalize()

    resolved = [
        item
        for item in values
        if item.healthy is not None
    ]

    current = aggregate_health(
        values
    )

    ratio = healthy_ratio(
        values
    )

    per_resource = resource_health(
        values
    )

    observed_resources = set(
        per_resource
    )

    missing_targets = sorted(
        set(targets)
        - observed_resources
    )

    recovered_resources = sorted(
        resource_id
        for resource_id, state
        in per_resource.items()
        if state is True
    )

    unhealthy_resources = sorted(
        resource_id
        for resource_id, state
        in per_resource.items()
        if state is False
    )

    failed_observations = [
        item
        for item in resolved
        if item.healthy is False
    ]

    critical_failures = [
        item
        for item
        in failed_observations
        if bool(
            item.attributes.get(
                "critical",
                False,
            )
        )
    ]

    failed_checks = [
        item.check_id
        for item
        in failed_observations
    ]

    reasons: List[str] = []

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    if (
        len(resolved)
        < configuration.
        minimum_resolved_observations
    ):
        status = (
            RecoveryStatus.
            MONITORING.value
        )

        reasons.append(
            "Too few observations have "
            "a resolved health state."
        )

    elif (
        configuration.
        require_all_target_resources_observed
        and missing_targets
    ):
        status = (
            RecoveryStatus.
            MONITORING.value
        )

        reasons.append(
            "Not all target resources have "
            "post-remediation observations."
        )

    elif (
        configuration.
        fail_on_any_critical_failure
        and critical_failures
    ):
        status = (
            RecoveryStatus.
            FAILED.value
        )

        reasons.append(
            "A critical recovery check "
            "failed."
        )

    elif (
        current is not None
        and current
        >= configuration.
        recovered_score_threshold
        and ratio is not None
        and ratio
        >= configuration.
        minimum_healthy_ratio
        and not unhealthy_resources
    ):
        status = (
            RecoveryStatus.
            RECOVERED.value
        )

        reasons.append(
            "Post-remediation observations "
            "meet the recovery thresholds."
        )

    elif (
        current is not None
        and current
        >= configuration.
        partial_score_threshold
        and recovered_resources
    ):
        status = (
            RecoveryStatus.
            PARTIAL.value
        )

        reasons.append(
            "Some recovery evidence is "
            "positive, but full recovery "
            "is not established."
        )

    elif failed_observations:
        status = (
            RecoveryStatus.
            FAILED.value
        )

        reasons.append(
            "Post-remediation observations "
            "contain failed health checks."
        )

    else:
        status = (
            RecoveryStatus.
            MONITORING.value
        )

        reasons.append(
            "Recovery evidence remains "
            "unresolved."
        )

    score = (
        current
        if current is not None
        else (
            ratio
            if ratio is not None
            else 0.0
        )
    )

    evidence = [
        item.to_evidence()
        for item in values
    ]

    evidence.append(
        HealingEvidence(
            evidence_type=(
                "recovery_summary"
            ),
            description=(
                reasons[0]
            ),
            score=score,
            source_id=(
                remediation.
                remediation_id
            ),
            resource_ids=targets,
            attributes={
                "status": status,
                "healthy_ratio": ratio,
                "health_before": (
                    before
                ),
                "health_current": (
                    current
                ),
                "missing_target_resources": (
                    missing_targets
                ),
                "reasons": reasons,
            },
        )
    )

    return RecoveryAssessment(
        status=status,
        score=score,
        health_before=before,
        health_current=current,
        recovered_resource_ids=(
            recovered_resources
        ),
        unhealthy_resource_ids=(
            unhealthy_resources
        ),
        checks=[
            item.check_id
            for item in values
        ],
        failed_checks=(
            failed_checks
        ),
        evidence=evidence,
        attributes={
            "observation_count": (
                len(values)
            ),
            "resolved_observation_count": (
                len(resolved)
            ),
            "healthy_ratio": ratio,
            "missing_target_resources": (
                missing_targets
            ),
            "critical_failure_count": (
                len(
                    critical_failures
                )
            ),
            "reasons": reasons,
            "remediation_verification_status": (
                remediation.
                verification.status
            ),
        },
    ).finalize()


# ---------------------------------------------------------------
# Remediation verification conversion
# ---------------------------------------------------------------

def assessment_from_verification(
    remediation: RemediationResult,
) -> RecoveryAssessment:
    """
    Initialize a RecoveryAssessment from the
    remediation verification result.

    This is useful as an initial bridge, but
    later observation-based assessment is
    stronger.

    Dry-run verification never becomes recovered.
    """

    remediation.finalize()

    verification = (
        remediation.verification
    )

    if (
        remediation.request.mode
        == RemediationMode.
        DRY_RUN.value
    ):
        status = (
            RecoveryStatus.
            MONITORING.value
        )

    elif (
        verification.status
        == VerificationStatus.
        PASSED.value
    ):
        status = (
            RecoveryStatus.
            RECOVERED.value
        )

    elif (
        verification.status
        == VerificationStatus.
        FAILED.value
    ):
        status = (
            RecoveryStatus.
            FAILED.value
        )

    else:
        status = (
            RecoveryStatus.
            MONITORING.value
        )

    return RecoveryAssessment(
        status=status,
        score=(
            verification.health_after
            or 0.0
        ),
        health_before=(
            verification.health_before
        ),
        health_current=(
            verification.health_after
        ),
        recovered_resource_ids=(
            list(
                remediation.request.
                target_resource_ids
            )
            if status
            == RecoveryStatus.
            RECOVERED.value
            else []
        ),
        unhealthy_resource_ids=(
            list(
                remediation.request.
                target_resource_ids
            )
            if status
            == RecoveryStatus.
            FAILED.value
            else []
        ),
        checks=list(
            verification.checks
        ),
        failed_checks=list(
            verification.failed_checks
        ),
        evidence=[
            HealingEvidence(
                evidence_type=(
                    "remediation_verification"
                ),
                description=(
                    "Recovery assessment was "
                    "initialized from the "
                    "remediation verification "
                    "result."
                ),
                score=(
                    verification.
                    health_after
                    or 0.0
                ),
                source_id=(
                    remediation.
                    remediation_id
                ),
                resource_ids=list(
                    remediation.request.
                    target_resource_ids
                ),
                attributes={
                    "verification_status": (
                        verification.status
                    ),
                    "dry_run": (
                        remediation.
                        request.mode
                        == RemediationMode.
                        DRY_RUN.value
                    ),
                },
            )
        ],
        attributes={
            "source": (
                "remediation_verification"
            ),
            "recovery_claimed": (
                status
                == RecoveryStatus.
                RECOVERED.value
            ),
        },
    ).finalize()


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------

def recovery_summary(
    assessments: Iterable[
        RecoveryAssessment
    ],
) -> Dict[str, Any]:
    values = list(
        assessments
    )

    statuses: Dict[
        str,
        int,
    ] = {}

    for assessment in values:
        statuses[
            assessment.status
        ] = (
            statuses.get(
                assessment.status,
                0,
            )
            + 1
        )

    return {
        "assessment_count": (
            len(values)
        ),
        "recovered_count": sum(
            1
            for item in values
            if (
                item.status
                == RecoveryStatus.
                RECOVERED.value
            )
        ),
        "partial_count": sum(
            1
            for item in values
            if (
                item.status
                == RecoveryStatus.
                PARTIAL.value
            )
        ),
        "monitoring_count": sum(
            1
            for item in values
            if (
                item.status
                == RecoveryStatus.
                MONITORING.value
            )
        ),
        "failed_count": sum(
            1
            for item in values
            if (
                item.status
                == RecoveryStatus.
                FAILED.value
            )
        ),
        "statuses": statuses,
    }


class RecoveryEvaluator:
    """
    Non-mutating recovery evaluator.

    It consumes observations and produces a
    RecoveryAssessment. It cannot execute
    remediation.
    """

    def __init__(
        self,
        *,
        config: Optional[
            RecoveryConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or RecoveryConfig()
        )

        self.config.validate()

    def assess(
        self,
        remediation: RemediationResult,
        *,
        observations: Iterable[
            RecoveryObservation
        ] = (),
        health_before: Optional[
            float
        ] = None,
        target_resource_ids: Iterable[
            str
        ] = (),
    ) -> RecoveryAssessment:
        return assess_recovery(
            remediation,
            observations=observations,
            health_before=(
                health_before
            ),
            target_resource_ids=(
                target_resource_ids
            ),
            config=self.config,
        )

    def from_verification(
        self,
        remediation: RemediationResult,
    ) -> RecoveryAssessment:
        return (
            assessment_from_verification(
                remediation
            )
        )
