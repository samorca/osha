from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from datetime import (
    datetime,
    timezone,
)
from statistics import mean
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.self_healing.schema import (
    HealingEvidence,
    RecoveryAssessment,
    RecoveryStatus,
    StabilityAssessment,
    StabilityStatus,
    checked_score,
    utc_now,
)


# ---------------------------------------------------------------
# Stability observations
# ---------------------------------------------------------------

@dataclass(frozen=True)
class StabilityObservation:
    """
    One health sample collected after recovery
    has initially been established.

    A sequence of these samples is used to detect
    persistent health or regression.
    """

    sample_id: str

    resource_id: str

    healthy: Optional[
        bool
    ]

    score: float = 0.0

    source_id: Optional[
        str
    ] = None

    service_id: Optional[
        str
    ] = None

    description: Optional[
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
                "stability_observation"
            ),
            description=(
                self.description
                or (
                    "Stability sample "
                    f"{self.sample_id}"
                )
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
                "sample_id": (
                    self.sample_id
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
class StabilizationConfig:
    """
    OSHA development stabilization thresholds.

    These values should eventually become
    service/action-specific profiles.
    """

    minimum_samples: int = 3

    minimum_resolved_samples: int = 3

    minimum_stable_score: float = 0.75

    minimum_healthy_ratio: float = 1.0

    minimum_observation_window_seconds: float = (
        60.0
    )

    maximum_score_drop: float = 0.20

    require_recovery_first: bool = True

    require_all_resources_observed: bool = True

    fail_on_critical_sample: bool = True

    def validate(
        self,
    ) -> None:
        if self.minimum_samples < 1:
            raise ValueError(
                "minimum_samples "
                "must be >= 1"
            )

        if (
            self.
            minimum_resolved_samples
            < 1
        ):
            raise ValueError(
                "minimum_resolved_samples "
                "must be >= 1"
            )

        if (
            self.
            minimum_observation_window_seconds
            < 0
        ):
            raise ValueError(
                "minimum_observation_window_"
                "seconds must be >= 0"
            )

        for name, value in (
            (
                "minimum_stable_score",
                self.
                minimum_stable_score,
            ),
            (
                "minimum_healthy_ratio",
                self.
                minimum_healthy_ratio,
            ),
            (
                "maximum_score_drop",
                self.maximum_score_drop,
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


# ---------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------

def parse_timestamp(
    value: str,
) -> Optional[
    datetime
]:
    text = str(
        value
    ).strip()

    if not text:
        return None

    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        result = (
            datetime.fromisoformat(
                text
            )
        )

    except ValueError:
        return None

    if result.tzinfo is None:
        result = result.replace(
            tzinfo=timezone.utc
        )

    return result.astimezone(
        timezone.utc
    )


def observation_window_seconds(
    observations: Iterable[
        StabilityObservation
    ],
) -> Optional[float]:
    timestamps = [
        parsed
        for parsed in (
            parse_timestamp(
                item.observed_at
            )
            for item
            in observations
        )
        if parsed is not None
    ]

    if not timestamps:
        return None

    return max(
        0.0,
        (
            max(timestamps)
            - min(timestamps)
        ).total_seconds(),
    )


# ---------------------------------------------------------------
# Health calculations
# ---------------------------------------------------------------

def resolved_ratio(
    observations: Iterable[
        StabilityObservation
    ],
) -> Optional[float]:
    values = [
        item
        for item in observations
        if item.healthy is not None
    ]

    if not values:
        return None

    return (
        sum(
            1
            for item in values
            if item.healthy
        )
        / len(values)
    )


def average_score(
    observations: Iterable[
        StabilityObservation
    ],
) -> Optional[float]:
    values = [
        item.normalized_score()
        for item in observations
    ]

    if not values:
        return None

    return checked_score(
        mean(values)
    )


# ---------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------

def score_regressed(
    observations: Iterable[
        StabilityObservation
    ],
    *,
    maximum_drop: float,
) -> bool:
    """
    Determine whether later health falls
    materially below the first stabilization
    sample.

    This is intentionally simple for the initial
    OSHA implementation.
    """

    values = list(
        observations
    )

    if len(values) < 2:
        return False

    ordered = sorted(
        values,
        key=lambda item: (
            parse_timestamp(
                item.observed_at
            )
            or datetime.min.replace(
                tzinfo=timezone.utc
            )
        ),
    )

    baseline = (
        ordered[
            0
        ].normalized_score()
    )

    later_minimum = min(
        item.normalized_score()
        for item
        in ordered[1:]
    )

    return (
        baseline
        - later_minimum
        > maximum_drop
    )


# ---------------------------------------------------------------
# Resource view
# ---------------------------------------------------------------

def resource_states(
    observations: Iterable[
        StabilityObservation
    ],
) -> Dict[
    str,
    Optional[bool],
]:
    grouped: Dict[
        str,
        List[
            StabilityObservation
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

    output: Dict[
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
            output[
                resource_id
            ] = None

        elif any(
            state is False
            for state in states
        ):
            output[
                resource_id
            ] = False

        else:
            output[
                resource_id
            ] = True

    return output


# ---------------------------------------------------------------
# Stability assessment
# ---------------------------------------------------------------

def assess_stability(
    recovery: RecoveryAssessment,
    *,
    observations: Iterable[
        StabilityObservation
    ] = (),
    target_resource_ids: Iterable[
        str
    ] = (),
    config: Optional[
        StabilizationConfig
    ] = None,
) -> StabilityAssessment:
    """
    Assess whether an initially recovered system
    remains healthy over multiple observations.

    This function performs no remediation.
    """

    configuration = (
        config
        or StabilizationConfig()
    )

    configuration.validate()

    recovery.finalize()

    values = list(
        observations
    )

    targets = sorted(
        set(
            target_resource_ids
        )
    )

    # ---------------------------------------------------------
    # Recovery prerequisite
    # ---------------------------------------------------------

    if (
        configuration.
        require_recovery_first
        and recovery.status
        != RecoveryStatus.
        RECOVERED.value
    ):
        return StabilityAssessment(
            status=(
                StabilityStatus.
                OBSERVING.value
            ),
            score=0.0,
            sample_count=(
                len(values)
            ),
            observation_window_seconds=(
                observation_window_seconds(
                    values
                )
            ),
            regression_detected=False,
            reasons=[
                (
                    "Recovery has not yet "
                    "been established, so "
                    "stability cannot be "
                    "concluded."
                )
            ],
            evidence=[
                item.to_evidence()
                for item in values
            ],
            attributes={
                "recovery_status": (
                    recovery.status
                ),
                "stability_claimed": (
                    False
                ),
            },
        ).finalize()

    # ---------------------------------------------------------
    # Sample count
    # ---------------------------------------------------------

    if (
        len(values)
        < configuration.
        minimum_samples
    ):
        return StabilityAssessment(
            status=(
                StabilityStatus.
                OBSERVING.value
            ),
            score=0.0,
            sample_count=(
                len(values)
            ),
            observation_window_seconds=(
                observation_window_seconds(
                    values
                )
            ),
            reasons=[
                (
                    "Not enough "
                    "post-recovery samples "
                    "are available."
                )
            ],
            evidence=[
                item.to_evidence()
                for item in values
            ],
        ).finalize()

    resolved = [
        item
        for item in values
        if item.healthy is not None
    ]

    window = (
        observation_window_seconds(
            values
        )
    )

    ratio = resolved_ratio(
        values
    )

    score = average_score(
        values
    )

    regression = score_regressed(
        values,
        maximum_drop=(
            configuration.
            maximum_score_drop
        ),
    )

    per_resource = (
        resource_states(
            values
        )
    )

    missing_resources = sorted(
        set(targets)
        - set(per_resource)
    )

    unhealthy_resources = sorted(
        resource_id
        for resource_id, state
        in per_resource.items()
        if state is False
    )

    critical_failures = [
        item
        for item in values
        if (
            item.healthy is False
            and bool(
                item.attributes.get(
                    "critical",
                    False,
                )
            )
        )
    ]

    reasons: List[
        str
    ] = []

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    if (
        len(resolved)
        < configuration.
        minimum_resolved_samples
    ):
        status = (
            StabilityStatus.
            OBSERVING.value
        )

        reasons.append(
            "Not enough samples have "
            "a resolved health state."
        )

    elif (
        window is None
        or window
        < configuration.
        minimum_observation_window_seconds
    ):
        status = (
            StabilityStatus.
            OBSERVING.value
        )

        reasons.append(
            "The stability observation "
            "window is not yet long enough."
        )

    elif (
        configuration.
        require_all_resources_observed
        and missing_resources
    ):
        status = (
            StabilityStatus.
            OBSERVING.value
        )

        reasons.append(
            "Not all target resources have "
            "stability observations."
        )

    elif (
        configuration.
        fail_on_critical_sample
        and critical_failures
    ):
        status = (
            StabilityStatus.
            UNSTABLE.value
        )

        reasons.append(
            "A critical stability sample "
            "failed."
        )

    elif regression:
        status = (
            StabilityStatus.
            REGRESSED.value
        )

        reasons.append(
            "Health regressed beyond the "
            "configured score-drop "
            "tolerance."
        )

    elif unhealthy_resources:
        status = (
            StabilityStatus.
            UNSTABLE.value
        )

        reasons.append(
            "At least one target resource "
            "became unhealthy during "
            "stabilization."
        )

    elif (
        ratio is not None
        and ratio
        < configuration.
        minimum_healthy_ratio
    ):
        status = (
            StabilityStatus.
            UNSTABLE.value
        )

        reasons.append(
            "Healthy sample ratio is "
            "below the configured minimum."
        )

    elif (
        score is not None
        and score
        < configuration.
        minimum_stable_score
    ):
        status = (
            StabilityStatus.
            UNSTABLE.value
        )

        reasons.append(
            "Average stability score is "
            "below the configured minimum."
        )

    else:
        status = (
            StabilityStatus.
            STABLE.value
        )

        reasons.append(
            "Recovery remained healthy "
            "across the configured "
            "stabilization window."
        )

    # ---------------------------------------------------------
    # Evidence
    # ---------------------------------------------------------

    evidence = [
        item.to_evidence()
        for item in values
    ]

    evidence.append(
        HealingEvidence(
            evidence_type=(
                "stability_summary"
            ),
            description=(
                reasons[0]
            ),
            score=(
                score
                if score is not None
                else (
                    ratio
                    if ratio is not None
                    else 0.0
                )
            ),
            resource_ids=targets,
            attributes={
                "status": status,
                "sample_count": (
                    len(values)
                ),
                "resolved_sample_count": (
                    len(resolved)
                ),
                "healthy_ratio": ratio,
                "observation_window_seconds": (
                    window
                ),
                "regression_detected": (
                    regression
                ),
                "missing_target_resources": (
                    missing_resources
                ),
                "unhealthy_resources": (
                    unhealthy_resources
                ),
                "critical_failure_count": (
                    len(
                        critical_failures
                    )
                ),
            },
        )
    )

    return StabilityAssessment(
        status=status,
        score=(
            score
            if score is not None
            else 0.0
        ),
        sample_count=len(
            values
        ),
        observation_window_seconds=(
            window
        ),
        regression_detected=(
            regression
        ),
        reasons=reasons,
        evidence=evidence,
        attributes={
            "resolved_sample_count": (
                len(resolved)
            ),
            "healthy_ratio": ratio,
            "missing_target_resources": (
                missing_resources
            ),
            "unhealthy_resources": (
                unhealthy_resources
            ),
            "critical_failure_count": (
                len(
                    critical_failures
                )
            ),
            "recovery_status": (
                recovery.status
            ),
        },
    ).finalize()


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------

def stabilization_summary(
    assessments: Iterable[
        StabilityAssessment
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
        "stable_count": sum(
            1
            for item in values
            if (
                item.status
                == StabilityStatus.
                STABLE.value
            )
        ),
        "observing_count": sum(
            1
            for item in values
            if (
                item.status
                == StabilityStatus.
                OBSERVING.value
            )
        ),
        "unstable_count": sum(
            1
            for item in values
            if (
                item.status
                == StabilityStatus.
                UNSTABLE.value
            )
        ),
        "regressed_count": sum(
            1
            for item in values
            if (
                item.status
                == StabilityStatus.
                REGRESSED.value
            )
        ),
        "statuses": statuses,
    }


class StabilizationEvaluator:
    """
    Non-mutating recovery-stability evaluator.
    """

    def __init__(
        self,
        *,
        config: Optional[
            StabilizationConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or StabilizationConfig()
        )

        self.config.validate()

    def assess(
        self,
        recovery: RecoveryAssessment,
        *,
        observations: Iterable[
            StabilityObservation
        ] = (),
        target_resource_ids: Iterable[
            str
        ] = (),
    ) -> StabilityAssessment:
        return assess_stability(
            recovery,
            observations=(
                observations
            ),
            target_resource_ids=(
                target_resource_ids
            ),
            config=self.config,
        )
