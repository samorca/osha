from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.self_healing.schema import (
    EscalationLevel,
    EscalationRequest,
    HealingEvidence,
    HealingSession,
    RecoveryStatus,
    StabilityStatus,
)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class EscalationConfig:
    """
    OSHA escalation policy.

    These values are project policy defaults,
    not universal incident-response rules.
    """

    maximum_healing_attempts: int = 2

    maximum_failed_attempts: int = 1

    maximum_unresolved_cycles: int = 3

    minimum_recovery_score: float = 0.40

    minimum_stability_score: float = 0.50

    escalate_partial_recovery: bool = True

    escalate_failed_recovery: bool = True

    escalate_regression: bool = True

    escalate_unstable_recovery: bool = True

    escalate_repeated_attempts: bool = True

    escalate_unresolved_cycles: bool = True

    escalate_critical_incident: bool = True

    operator_role: str = "infrastructure_operator"

    senior_operator_role: str = "senior_infrastructure_operator"

    incident_command_role: str = "incident_commander"

    def validate(
        self,
    ) -> None:
        if self.maximum_healing_attempts < 1:
            raise ValueError("maximum_healing_attempts must be >= 1")

        if self.maximum_failed_attempts < 1:
            raise ValueError("maximum_failed_attempts must be >= 1")

        if self.maximum_unresolved_cycles < 1:
            raise ValueError("maximum_unresolved_cycles must be >= 1")

        for name, value in (
            (
                "minimum_recovery_score",
                self.minimum_recovery_score,
            ),
            (
                "minimum_stability_score",
                self.minimum_stability_score,
            ),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1")


# ---------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------


@dataclass(frozen=True)
class EscalationEvaluation:
    """
    Internal escalation-policy result.

    EscalationRequest is the persistent schema
    object. This structure preserves policy
    reasoning before building that request.
    """

    required: bool

    level: str

    reasons: tuple[
        str,
        ...,
    ]

    reason_codes: tuple[
        str,
        ...,
    ]

    required_roles: tuple[
        str,
        ...,
    ]

    attributes: Dict[
        str,
        Any,
    ]

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "required": (self.required),
            "level": self.level,
            "reasons": list(self.reasons),
            "reason_codes": list(self.reason_codes),
            "required_roles": list(self.required_roles),
            "attributes": dict(self.attributes),
        }


# ---------------------------------------------------------------
# Counters
# ---------------------------------------------------------------


def counter(
    session: HealingSession,
    name: str,
) -> int:
    value = session.attributes.get(
        name,
        0,
    )

    try:
        number = int(value)

    except (
        TypeError,
        ValueError,
    ):
        return 0

    return max(
        0,
        number,
    )


def healing_attempts(
    session: HealingSession,
) -> int:
    return counter(
        session,
        "healing_attempts",
    )


def failed_attempts(
    session: HealingSession,
) -> int:
    return counter(
        session,
        "failed_healing_attempts",
    )


def unresolved_cycles(
    session: HealingSession,
) -> int:
    return counter(
        session,
        "unresolved_cycles",
    )


def increment_counter(
    session: HealingSession,
    name: str,
    *,
    amount: int = 1,
) -> int:
    """
    Increment lifecycle counters.

    These counters should eventually be persisted
    with the HealingSession.
    """

    current = counter(
        session,
        name,
    )

    updated = max(
        0,
        current + amount,
    )

    session.attributes[name] = updated

    session.finalize()

    return updated


def record_healing_attempt(
    session: HealingSession,
) -> int:
    return increment_counter(
        session,
        "healing_attempts",
    )


def record_failed_attempt(
    session: HealingSession,
) -> int:
    return increment_counter(
        session,
        "failed_healing_attempts",
    )


def record_unresolved_cycle(
    session: HealingSession,
) -> int:
    return increment_counter(
        session,
        "unresolved_cycles",
    )


# ---------------------------------------------------------------
# Severity
# ---------------------------------------------------------------


def incident_severity(
    session: HealingSession,
) -> str:
    """
    Retrieve severity when the upstream pipeline
    has placed it in session attributes.

    Missing severity remains unknown.
    """

    value = session.attributes.get(
        "incident_severity",
        "unknown",
    )

    return str(value).strip().lower()


# ---------------------------------------------------------------
# Escalation level ordering
# ---------------------------------------------------------------

LEVEL_RANK = {
    EscalationLevel.NONE.value: 0,
    EscalationLevel.OPERATOR.value: 10,
    EscalationLevel.SENIOR_OPERATOR.value: 20,
    EscalationLevel.INCIDENT_COMMAND.value: 30,
}


def stricter_level(
    current: str,
    proposed: str,
) -> str:
    current_rank = LEVEL_RANK.get(
        current,
        0,
    )

    proposed_rank = LEVEL_RANK.get(
        proposed,
        0,
    )

    if proposed_rank > current_rank:
        return proposed

    return current


# ---------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------


def evaluate_escalation(
    session: HealingSession,
    *,
    config: Optional[EscalationConfig] = None,
) -> EscalationEvaluation:
    """
    Determine whether the self-healing loop
    should stop and require human intervention.

    This function does not notify anyone.
    """

    configuration = config or EscalationConfig()

    configuration.validate()

    session.finalize()

    reasons: List[str] = []

    reason_codes: List[str] = []

    level = EscalationLevel.NONE.value

    attempts = healing_attempts(session)

    failures = failed_attempts(session)

    unresolved = unresolved_cycles(session)

    severity = incident_severity(session)

    # ---------------------------------------------------------
    # Failed recovery
    # ---------------------------------------------------------

    if (
        configuration.escalate_failed_recovery
        and session.recovery.status == RecoveryStatus.FAILED.value
    ):
        level = stricter_level(
            level,
            EscalationLevel.SENIOR_OPERATOR.value,
        )

        reason_codes.append("recovery_failed")

        reasons.append("Post-remediation recovery checks indicate failure.")

    # ---------------------------------------------------------
    # Partial recovery
    # ---------------------------------------------------------

    if (
        configuration.escalate_partial_recovery
        and session.recovery.status == RecoveryStatus.PARTIAL.value
    ):
        level = stricter_level(
            level,
            EscalationLevel.OPERATOR.value,
        )

        reason_codes.append("partial_recovery")

        reasons.append("Infrastructure only partially recovered.")

    # ---------------------------------------------------------
    # Poor recovery score
    # ---------------------------------------------------------

    if (
        session.recovery.status
        not in {
            RecoveryStatus.UNKNOWN.value,
            RecoveryStatus.NOT_STARTED.value,
            RecoveryStatus.MONITORING.value,
        }
        and session.recovery.score < configuration.minimum_recovery_score
    ):
        level = stricter_level(
            level,
            EscalationLevel.OPERATOR.value,
        )

        reason_codes.append("recovery_score_below_limit")

        reasons.append("Recovery score remains below the escalation threshold.")

    # ---------------------------------------------------------
    # Regression
    # ---------------------------------------------------------

    if (
        configuration.escalate_regression
        and session.stability.status == StabilityStatus.REGRESSED.value
    ):
        level = stricter_level(
            level,
            EscalationLevel.SENIOR_OPERATOR.value,
        )

        reason_codes.append("recovery_regressed")

        reasons.append("Infrastructure regressed after apparent recovery.")

    # ---------------------------------------------------------
    # Unstable recovery
    # ---------------------------------------------------------

    if (
        configuration.escalate_unstable_recovery
        and session.stability.status == StabilityStatus.UNSTABLE.value
    ):
        level = stricter_level(
            level,
            EscalationLevel.SENIOR_OPERATOR.value,
        )

        reason_codes.append("recovery_unstable")

        reasons.append("Post-recovery infrastructure remains unstable.")

    # ---------------------------------------------------------
    # Low stability score
    # ---------------------------------------------------------

    if (
        session.stability.status
        not in {
            StabilityStatus.UNKNOWN.value,
            StabilityStatus.OBSERVING.value,
        }
        and session.stability.score < configuration.minimum_stability_score
    ):
        level = stricter_level(
            level,
            EscalationLevel.OPERATOR.value,
        )

        reason_codes.append("stability_score_below_limit")

        reasons.append("Stability score remains below the configured threshold.")

    # ---------------------------------------------------------
    # Repeated healing attempts
    # ---------------------------------------------------------

    if (
        configuration.escalate_repeated_attempts
        and attempts >= configuration.maximum_healing_attempts
    ):
        level = stricter_level(
            level,
            EscalationLevel.SENIOR_OPERATOR.value,
        )

        reason_codes.append("healing_attempt_limit")

        reasons.append("Automated healing attempt limit has been reached.")

    # ---------------------------------------------------------
    # Repeated failures
    # ---------------------------------------------------------

    if failures >= configuration.maximum_failed_attempts:
        level = stricter_level(
            level,
            EscalationLevel.SENIOR_OPERATOR.value,
        )

        reason_codes.append("failed_attempt_limit")

        reasons.append(
            "The configured number of failed healing attempts has been reached."
        )

    # ---------------------------------------------------------
    # Unresolved cycles
    # ---------------------------------------------------------

    if (
        configuration.escalate_unresolved_cycles
        and unresolved >= configuration.maximum_unresolved_cycles
    ):
        level = stricter_level(
            level,
            EscalationLevel.OPERATOR.value,
        )

        reason_codes.append("unresolved_cycle_limit")

        reasons.append(
            "The incident remained unresolved across too many healing cycles."
        )

    # ---------------------------------------------------------
    # Critical incident
    # ---------------------------------------------------------

    if configuration.escalate_critical_incident and severity in {
        "critical",
        "emergency",
    }:
        level = stricter_level(
            level,
            EscalationLevel.INCIDENT_COMMAND.value,
        )

        reason_codes.append("critical_incident")

        reasons.append("Incident severity requires incident-command escalation.")

    required = level != EscalationLevel.NONE.value

    if level == EscalationLevel.INCIDENT_COMMAND.value:
        roles = (
            configuration.incident_command_role,
            configuration.senior_operator_role,
        )

    elif level == EscalationLevel.SENIOR_OPERATOR.value:
        roles = (configuration.senior_operator_role,)

    elif level == EscalationLevel.OPERATOR.value:
        roles = (configuration.operator_role,)

    else:
        roles = ()

    return EscalationEvaluation(
        required=required,
        level=level,
        reasons=tuple(dict.fromkeys(reasons)),
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        required_roles=tuple(dict.fromkeys(roles)),
        attributes={
            "healing_attempts": (attempts),
            "failed_healing_attempts": (failures),
            "unresolved_cycles": (unresolved),
            "incident_severity": (severity),
            "recovery_status": (session.recovery.status),
            "recovery_score": (session.recovery.score),
            "stability_status": (session.stability.status),
            "stability_score": (session.stability.score),
        },
    )


# ---------------------------------------------------------------
# Persistent escalation request
# ---------------------------------------------------------------


def build_escalation_request(
    session: HealingSession,
    evaluation: EscalationEvaluation,
) -> EscalationRequest:
    """
    Convert policy evaluation into the persistent
    EscalationRequest schema.
    """

    reason = (
        " ".join(evaluation.reasons)
        if evaluation.reasons
        else ("No escalation is currently required.")
    )

    evidence = []

    if evaluation.required:
        evidence.append(
            HealingEvidence(
                evidence_type=("escalation_decision"),
                description=reason,
                score=1.0,
                source_id=(session.session_id),
                resource_ids=list(session.resource_ids),
                attributes={
                    "level": (evaluation.level),
                    "reason_codes": list(evaluation.reason_codes),
                },
            )
        )

    return EscalationRequest(
        level=evaluation.level,
        required=(evaluation.required),
        reason=reason,
        required_roles=list(evaluation.required_roles),
        incident_id=(session.incident_id),
        action_id=(session.action_id),
        remediation_id=(session.remediation_id),
        evidence=evidence,
        attributes={
            "reason_codes": list(evaluation.reason_codes),
            **evaluation.attributes,
        },
    ).finalize()


def assess_escalation(
    session: HealingSession,
    *,
    config: Optional[EscalationConfig] = None,
) -> EscalationRequest:
    """
    Evaluate escalation and attach the resulting
    request to the HealingSession.

    This does not contact operators or external
    services.
    """

    evaluation = evaluate_escalation(
        session,
        config=config,
    )

    request = build_escalation_request(
        session,
        evaluation,
    )

    session.escalation = request

    session.evidence.extend(request.evidence)

    session.attributes["escalation_reason_codes"] = list(evaluation.reason_codes)

    session.finalize()

    return request


# ---------------------------------------------------------------
# Automation stop rule
# ---------------------------------------------------------------


def automation_should_stop(
    session: HealingSession,
    *,
    config: Optional[EscalationConfig] = None,
) -> bool:
    """
    Return True when the policy requires human
    escalation.

    The pipeline can use this as a hard boundary
    before proposing another automated healing
    attempt.
    """

    return evaluate_escalation(
        session,
        config=config,
    ).required


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------


def escalation_summary(
    requests: Iterable[EscalationRequest],
) -> Dict[str, Any]:
    values = list(requests)

    levels: Dict[
        str,
        int,
    ] = {}

    for request in values:
        levels[request.level] = (
            levels.get(
                request.level,
                0,
            )
            + 1
        )

    return {
        "request_count": len(values),
        "required_count": sum(1 for request in values if request.required),
        "operator_count": sum(
            1 for request in values if (request.level == EscalationLevel.OPERATOR.value)
        ),
        "senior_operator_count": sum(
            1
            for request in values
            if (request.level == EscalationLevel.SENIOR_OPERATOR.value)
        ),
        "incident_command_count": sum(
            1
            for request in values
            if (request.level == EscalationLevel.INCIDENT_COMMAND.value)
        ),
        "levels": levels,
    }


class EscalationEvaluator:
    """
    Non-mutating escalation-policy evaluator.

    It determines when automated healing should
    stop. It does not send notifications.
    """

    def __init__(
        self,
        *,
        config: Optional[EscalationConfig] = None,
    ) -> None:
        self.config = config or EscalationConfig()

        self.config.validate()

    def evaluate(
        self,
        session: HealingSession,
    ) -> EscalationEvaluation:
        return evaluate_escalation(
            session,
            config=self.config,
        )

    def assess(
        self,
        session: HealingSession,
    ) -> EscalationRequest:
        return assess_escalation(
            session,
            config=self.config,
        )

    def should_stop(
        self,
        session: HealingSession,
    ) -> bool:
        return automation_should_stop(
            session,
            config=self.config,
        )
