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
    ExecutionResult,
    ExecutionStatus,
    RemediationEvidence,
    RemediationMode,
    RemediationRequest,
    VerificationResult,
    VerificationStatus,
    checked_score,
    utc_now,
)


# ---------------------------------------------------------------
# Verification observations
# ---------------------------------------------------------------

@dataclass(frozen=True)
class VerificationObservation:
    """
    One post-remediation health observation.

    Observations should eventually come from the
    telemetry/collector path, not directly from
    the execution adapter.

    That separation helps prevent the component
    performing an action from also being the only
    component declaring the action successful.
    """

    check_id: str

    name: str

    passed: Optional[
        bool
    ]

    score: float = 0.0

    description: Optional[
        str
    ] = None

    source_id: Optional[
        str
    ] = None

    resource_ids: tuple[
        str,
        ...,
    ] = ()

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

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "check_id": (
                self.check_id
            ),
            "name": self.name,
            "passed": self.passed,
            "score": (
                self.normalized_score()
            ),
            "description": (
                self.description
            ),
            "source_id": (
                self.source_id
            ),
            "resource_ids": sorted(
                set(
                    self.resource_ids
                )
            ),
            "attributes": dict(
                self.attributes
            ),
            "observed_at": (
                self.observed_at
            ),
        }


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class VerificationConfig:
    """
    OSHA remediation-verification configuration.

    These are development policy defaults, not
    calibrated universal thresholds.
    """

    minimum_observations: int = 1

    minimum_pass_ratio: float = 1.0

    minimum_health_score: float = 0.70

    minimum_health_improvement: float = 0.0

    require_execution_success: bool = True

    require_changed_state_for_live_verification: bool = (
        True
    )

    allow_dry_run_verification: bool = True

    fail_on_explicit_failed_check: bool = True

    inconclusive_on_missing_observations: bool = (
        True
    )

    def validate(
        self,
    ) -> None:
        if self.minimum_observations < 1:
            raise ValueError(
                "minimum_observations "
                "must be >= 1"
            )

        if not (
            0.0
            <= self.minimum_pass_ratio
            <= 1.0
        ):
            raise ValueError(
                "minimum_pass_ratio must "
                "be between 0 and 1"
            )

        if not (
            0.0
            <= self.minimum_health_score
            <= 1.0
        ):
            raise ValueError(
                "minimum_health_score must "
                "be between 0 and 1"
            )

        if not (
            -1.0
            <= self.
            minimum_health_improvement
            <= 1.0
        ):
            raise ValueError(
                "minimum_health_improvement "
                "must be between -1 and 1"
            )


# ---------------------------------------------------------------
# Health aggregation
# ---------------------------------------------------------------

def finite_number(
    value: Any,
) -> Optional[float]:
    if isinstance(
        value,
        bool,
    ):
        return None

    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        result
    ):
        return None

    return result


def observation_health(
    observations: Iterable[
        VerificationObservation
    ],
) -> Optional[float]:
    """
    Aggregate scored observations.

    Only observations carrying finite scores are
    included.
    """

    scores = []

    for observation in observations:
        score = finite_number(
            observation.score
        )

        if score is None:
            continue

        scores.append(
            checked_score(
                score
            )
        )

    if not scores:
        return None

    return checked_score(
        sum(scores)
        / len(scores)
    )


def pass_ratio(
    observations: Iterable[
        VerificationObservation
    ],
) -> Optional[float]:
    """
    Ratio across observations carrying explicit
    pass/fail states.

    passed=None means unknown and is excluded.
    """

    resolved = [
        observation
        for observation
        in observations
        if observation.passed
        is not None
    ]

    if not resolved:
        return None

    passed_count = sum(
        1
        for observation
        in resolved
        if observation.passed
    )

    return (
        passed_count
        / len(resolved)
    )


# ---------------------------------------------------------------
# Execution eligibility
# ---------------------------------------------------------------

def execution_allows_verification(
    request: RemediationRequest,
    execution: ExecutionResult,
    *,
    config: VerificationConfig,
) -> tuple[
    bool,
    Optional[str],
]:
    """
    Validate whether post-action verification
    makes sense for the execution result.
    """

    request.finalize()

    execution.finalize()

    if (
        request.mode
        == RemediationMode.
        DRY_RUN.value
    ):
        if not (
            config.
            allow_dry_run_verification
        ):
            return (
                False,
                (
                    "Dry-run verification "
                    "is disabled."
                ),
            )

        if (
            execution.status
            != ExecutionStatus.
            DRY_RUN.value
        ):
            return (
                False,
                (
                    "Dry-run request does "
                    "not have a dry-run "
                    "execution result."
                ),
            )

        return (
            True,
            None,
        )

    if (
        config.
        require_execution_success
        and execution.status
        != ExecutionStatus.
        SUCCEEDED.value
    ):
        return (
            False,
            (
                "Live verification requires "
                "successful execution."
            ),
        )

    if (
        config.
        require_changed_state_for_live_verification
        and not execution.changed
    ):
        return (
            False,
            (
                "Live execution reported no "
                "state change to verify."
            ),
        )

    return (
        True,
        None,
    )


# ---------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------

def verify_execution(
    request: RemediationRequest,
    execution: ExecutionResult,
    *,
    observations: Iterable[
        VerificationObservation
    ] = (),
    health_before: Optional[
        float
    ] = None,
    config: Optional[
        VerificationConfig
    ] = None,
) -> VerificationResult:
    """
    Verify the operational outcome independently
    from adapter execution.

    This function performs no infrastructure
    action and does not query providers itself.
    """

    configuration = (
        config
        or VerificationConfig()
    )

    configuration.validate()

    request.finalize()

    execution.finalize()

    started_at = utc_now()

    observation_list = list(
        observations
    )

    allowed, reason = (
        execution_allows_verification(
            request,
            execution,
            config=configuration,
        )
    )

    if not allowed:
        return VerificationResult(
            status=(
                VerificationStatus.
                INCONCLUSIVE.value
            ),
            passed=False,
            health_before=(
                health_before
            ),
            health_after=None,
            checks=[],
            failed_checks=[],
            evidence=[
                RemediationEvidence(
                    evidence_type=(
                        "verification"
                    ),
                    description=(
                        reason
                        or (
                            "Execution cannot "
                            "be independently "
                            "verified."
                        )
                    ),
                    score=0.0,
                    source_id=(
                        request.request_id
                    ),
                    resource_ids=list(
                        request.
                        target_resource_ids
                    ),
                )
            ],
            started_at=started_at,
            finished_at=utc_now(),
            attributes={
                "verification_allowed": (
                    False
                ),
                "mode": request.mode,
                "execution_status": (
                    execution.status
                ),
            },
        ).finalize()

    # ---------------------------------------------------------
    # Dry-run
    # ---------------------------------------------------------

    if (
        request.mode
        == RemediationMode  ):
        return VerificationResult(
            status=(
                VerificationStatus.
                INCONCLUSIVE.value
            ),
            passed=False,
            health_before=(
                health_before
            ),
            health_after=None,
            checks=[
                "dry_run_execution"
            ],
            failed_checks=[],
            evidence=[
                RemediationEvidence(
                    evidence_type=(
                        "dry_run_verification"
                    ),
                    description=(
                        "Dry-run execution was "
                        "successful, but no "
                        "infrastructure recovery "
                        "claim can be made "
                        "because no state was "
                        "changed."
                    ),
                    score=0.0,
                    source_id=(
                        request.request_id
                    ),
                    resource_ids=list(
                        request.
                        target_resource_ids
                    ),
                )
            ],
            started_at=started_at,
            finished_at=utc_now(),
            attributes={
                "verification_allowed": (
                    True
                ),
                "dry_run": True,
                "recovery_claimed": (
                    False
                ),
            },
        ).finalize()

    # ---------------------------------------------------------
    # Real execution verification
    # ---------------------------------------------------------

    if (
        len(
            observation_list
        )
        < configuration.
        minimum_observations
    ):
        status = (
            VerificationStatus.
            INCONCLUSIVE.value
            if configuration.
            inconclusive_on_missing_observations
            else VerificationStatus.
            FAILED.value
        )

        return VerificationResult(
            status=status,
            passed=False,
            health_before=(
                health_before
            ),
            health_after=None,
            checks=[
                observation.check_id
                for observation
                in observation_list
            ],
            failed_checks=[
                "insufficient_observations"
            ],
            evidence=[
                RemediationEvidence(
                    evidence_type=(
                        "verification"
                    ),
                    description=(
                        "Not enough independent "
                        "post-action observations "
                        "were supplied."
                    ),
                    score=0.0,
                    source_id=(
                        request.request_id
                    ),
                    resource_ids=list(
                        request.
                        target_resource_ids
                    ),
                    attributes={
                        "observation_count": (
                            len(
                                observation_list
                            )
                        ),
                        "minimum_observations": (
                            configuration.
                            minimum_observations
                        ),
                    },
                )
            ],
            started_at=started_at,
            finished_at=utc_now(),
        ).finalize()

    explicit_failures = [
        observation
        for observation
        in observation_list
        if observation.passed
        is False
    ]

    ratio = pass_ratio(
        observation_list
    )

    health_after = observation_health(
        observation_list
    )

    before = (
        checked_score(
            health_before
        )
        if health_before
        is not None
        else None
    )

    improvement = None

    if (
        before is not None
        and health_after
        is not None
    ):
        improvement = (
            health_after
            - before
        )

    failed_checks = [
        observation.check_id
        for observation
        in explicit_failures
    ]

    reasons = []

    failed = False

    if (
        configuration.
        fail_on_explicit_failed_check
        and explicit_failures
    ):
        failed = True

        reasons.append(
            "One or more verification "
            "checks explicitly failed."
        )

    if (
        ratio is not None
        and ratio
        < configuration.
        minimum_pass_ratio
    ):
        failed = True

        reasons.append(
            "Verification pass ratio is "
            "below the configured minimum."
        )

    if (
        health_after is not None
        and health_after
        < configuration.
        minimum_health_score
    ):
        failed = True

        reasons.append(
            "Post-action health score is "
            "below the configured minimum."
        )

    if (
        improvement is not None
        and improvement
        < configuration.
        minimum_health_improvement
    ):
        failed = True

        reasons.append(
            "Post-action health improvement "
            "is below the configured minimum."
        )

    unresolved = [
        observation
        for observation
        in observation_list
        if observation.passed
        is None
    ]

    # If every check is unknown and no useful
    # health measurement exists, there isn't
    # sufficient evidence for PASS or FAIL.
    if (
        len(unresolved)
        == len(
            observation_list
        )
        and health_after is None
    ):
        status = (
            VerificationStatus.
            INCONCLUSIVE.value
        )

    elif failed:
        status = (
            VerificationStatus.
            FAILED.value
        )

    else:
        status = (
            VerificationStatus.
            PASSED.value
        )

    evidence = [
        RemediationEvidence(
            evidence_type=(
                "post_action_observation"
            ),
            description=(
                observation.description
                or observation.name
            ),
            score=(
                observation.
                normalized_score()
            ),
            source_id=(
                observation.source_id
            ),
            resource_ids=list(
                observation.resource_ids
            ),
            attributes={
                "check_id": (
                    observation.check_id
                ),
                "passed": (
                    observation.passed
                ),
                **dict(
                    observation.attributes
                ),
            },
            observed_at=(
                observation.observed_at
            ),
        )
        for observation
        in observation_list
    ]

    evidence.append(
        RemediationEvidence(
            evidence_type=(
                "verification_summary"
            ),
            description=(
                (
                    "Post-action verification "
                    "passed."
                )
                if status
                == VerificationStatus.
                PASSED.value
                else (
                    "Post-action verification "
                    "did not establish "
                    "successful recovery."
                )
            ),
            score=(
                health_after
                if health_after
                is not None
                else (
                    ratio
                    if ratio
                    is not None
                    else 0.0
                )
            ),
            source_id=(
                request.request_id
            ),
            resource_ids=list(
                request.
                target_resource_ids
            ),
            attributes={
                "pass_ratio": ratio,
                "health_before": (
                    before
                ),
                "health_after": (
                    health_after
                ),
                "health_improvement": (
                    improvement
                ),
                "failure_reasons": (
                    reasons
                ),
            },
        )
    )

    return VerificationResult(
        status=status,
        passed=(
            status
            == VerificationStatus.
            PASSED.value
        ),
        health_before=before,
        health_after=(
            health_after
        ),
        checks=[
            observation.check_id
            for observation
            in observation_list
        ],
        failed_checks=(
            failed_checks
        ),
        evidence=evidence,
        started_at=started_at,
        finished_at=utc_now(),
        attributes={
            "observation_count": (
                len(
                    observation_list
                )
            ),
            "resolved_check_count": (
                sum(
                    1
                    for observation
                    in observation_list
                    if observation.passed
                    is not None
                )
            ),
            "unresolved_check_count": (
                len(unresolved)
            ),
            "pass_ratio": ratio,
            "health_improvement": (
                improvement
            ),
            "failure_reasons": (
                reasons
            ),
        },
    ).finalize()


# ---------------------------------------------------------------
# Batch
# ---------------------------------------------------------------

def verify_executions(
    items: Iterable[
        tuple[
            RemediationRequest,
            ExecutionResult,
            Iterable[
                VerificationObservation
            ],
            Optional[float],
        ]
    ],
    *,
    config: Optional[
        VerificationConfig
    ] = None,
) -> List[
    VerificationResult
]:
    configuration = (
        config
        or VerificationConfig()
    )

    configuration.validate()

    return [
        verify_execution(
            request,
            execution,
            observations=observations,
            health_before=(
                health_before
            ),
            config=configuration,
        )
        for (
            request,
            execution,
            observations,
            health_before,
        )
        in items
    ]


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------

def verification_summary(
    results: Iterable[
        VerificationResult
    ],
) -> Dict[str, Any]:
    values = list(
        results
    )

    statuses: Dict[
        str,
        int,
    ] = {}

    for result in values:
        statuses[
            result.status
        ] = (
            statuses.get(
                result.status,
                0,
            )
            + 1
        )

    return {
        "verification_count": (
            len(values)
        ),
        "passed_count": sum(
            1
            for result in values
            if (
                result.status
                == VerificationStatus.
                PASSED.value
            )
        ),
        "failed_count": sum(
            1
            for result in values
            if (
                result.status
                == VerificationStatus.
                FAILED.value
            )
        ),
        "inconclusive_count": sum(
            1
            for result in values
            if (
                result.status
                == VerificationStatus.
                INCONCLUSIVE.value
            )
        ),
        "statuses": statuses,
    }


class RemediationVerifier:
    """
    Non-mutating post-action verifier.

    External telemetry should be converted into
    VerificationObservation objects before being
    passed here.
    """

    def __init__(
        self,
        *,
        config: Optional[
            VerificationConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or VerificationConfig()
        )

        self.config.validate()

    def verify(
        self,
        request: RemediationRequest,
        execution: ExecutionResult,
        *,
        observations: Iterable[
            VerificationObservation
        ] = (),
        health_before: Optional[
            float
        ] = None,
    ) -> VerificationResult:
        return verify_execution(
            request,
            execution,
            observations=observations,
            health_before=(
                health_before
            ),
            config=self.config,
        )
