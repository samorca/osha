from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
)

from src.remediation.schema import (
    RemediationResult,
    RemediationStatus,
    RollbackStatus,
    VerificationStatus,
)

from src.self_healing.schema import (
    EscalationLevel,
    HealingEvidence,
    HealingSession,
    HealingState,
    OutcomeStatus,
    RecoveryAssessment,
    RecoveryStatus,
    StabilityAssessment,
    StabilityStatus,
    utc_now,
)


# ---------------------------------------------------------------
# Allowed state transitions
# ---------------------------------------------------------------

ALLOWED_TRANSITIONS: Dict[
    str,
    Set[str],
] = {
    HealingState.CREATED.value: {
        HealingState.OBSERVING.value,
        HealingState.DETECTED.value,
        HealingState.FAILED.value,
    },

    HealingState.OBSERVING.value: {
        HealingState.DETECTED.value,
        HealingState.CLOSED.value,
        HealingState.FAILED.value,
    },

    HealingState.DETECTED.value: {
        HealingState.ANALYZING.value,
        HealingState.FAILED.value,
        HealingState.CLOSED.value,
    },

    HealingState.ANALYZING.value: {
        HealingState.DECIDING.value,
        HealingState.DEGRADED.value,
        HealingState.ESCALATED.value,
        HealingState.FAILED.value,
    },

    HealingState.DECIDING.value: {
        HealingState.AWAITING_APPROVAL.value,
        HealingState.REMEDIATING.value,
        HealingState.OBSERVING.value,
        HealingState.ESCALATED.value,
        HealingState.FAILED.value,
    },

    HealingState.AWAITING_APPROVAL.value: {
        HealingState.REMEDIATING.value,
        HealingState.OBSERVING.value,
        HealingState.ESCALATED.value,
        HealingState.FAILED.value,
    },

    HealingState.REMEDIATING.value: {
        HealingState.VERIFYING.value,
        HealingState.DEGRADED.value,
        HealingState.ESCALATED.value,
        HealingState.FAILED.value,
    },

    HealingState.VERIFYING.value: {
        HealingState.STABILIZING.value,
        HealingState.RECOVERED.value,
        HealingState.DEGRADED.value,
        HealingState.ESCALATED.value,
        HealingState.FAILED.value,
    },

    HealingState.STABILIZING.value: {
        HealingState.RECOVERED.value,
        HealingState.DEGRADED.value,
        HealingState.ESCALATED.value,
        HealingState.FAILED.value,
    },

    HealingState.RECOVERED.value: {
        HealingState.CLOSED.value,
        HealingState.DEGRADED.value,
        HealingState.OBSERVING.value,
    },

    HealingState.DEGRADED.value: {
        HealingState.OBSERVING.value,
        HealingState.ANALYZING.value,
        HealingState.DECIDING.value,
        HealingState.ESCALATED.value,
        HealingState.FAILED.value,
    },

    HealingState.ESCALATED.value: {
        HealingState.OBSERVING.value,
        HealingState.ANALYZING.value,
        HealingState.CLOSED.value,
        HealingState.FAILED.value,
    },

    HealingState.FAILED.value: {
        HealingState.ESCALATED.value,
        HealingState.CLOSED.value,
    },

    HealingState.CLOSED.value: set(),
}


# ---------------------------------------------------------------
# Transition record
# ---------------------------------------------------------------

@dataclass(frozen=True)
class StateTransition:
    """
    Immutable description of one state change.
    """

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


class InvalidTransitionError(
    ValueError
):
    """
    Raised when a state transition violates the
    self-healing state machine.
    """


# ---------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------

def valid_state(
    value: str,
) -> bool:
    return value in {
        state.value
        for state in HealingState
    }


def can_transition(
    current_state: str,
    next_state: str,
) -> bool:
    if current_state == next_state:
        return True

    return (
        next_state
        in ALLOWED_TRANSITIONS.get(
            current_state,
            set(),
        )
    )


def transition(
    session: HealingSession,
    next_state: str,
    *,
    reason: str,
    attributes: Optional[
        Dict[str, Any]
    ] = None,
) -> StateTransition:
    """
    Move one HealingSession through the lifecycle.

    Invalid state jumps are rejected.
    """

    session.finalize()

    if not valid_state(
        next_state
    ):
        raise InvalidTransitionError(
            "Unknown healing state: "
            f"{next_state}"
        )

    previous_state = (
        session.state
    )

    if not can_transition(
        previous_state,
        next_state,
    ):
        raise InvalidTransitionError(
            "Invalid self-healing "
            "transition: "
            f"{previous_state} -> "
            f"{next_state}"
        )

    event = StateTransition(
        previous_state=(
            previous_state
        ),
        next_state=(
            next_state
        ),
        reason=reason,
        attributes=dict(
            attributes
            or {}
        ),
    )

    session.state = (
        next_state
    )

    session.updated_at = (
        utc_now()
    )

    history = session.attributes.setdefault(
        "state_transitions",
        [],
    )

    history.append(
        event.to_dict()
    )

    session.evidence.append(
        HealingEvidence(
            evidence_type=(
                "state_transition"
            ),
            description=(
                f"{previous_state} -> "
                f"{next_state}: "
                f"{reason}"
            ),
            score=1.0,
            source_id=(
                session.session_id
            ),
            resource_ids=list(
                session.resource_ids
            ),
            attributes={
                "previous_state": (
                    previous_state
                ),
                "next_state": (
                    next_state
                ),
            },
        )
    )

    session.finalize()

    return event


# ---------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------

def create_session(
    *,
    environment_id: str,
    incident_id: str,
    decision_id: Optional[
        str
    ] = None,
    plan_id: Optional[
        str
    ] = None,
    action_id: Optional[
        str
    ] = None,
    resource_ids: Iterable[
        str
    ] = (),
    service_ids: Iterable[
        str
    ] = (),
    attributes: Optional[
        Dict[str, Any]
    ] = None,
) -> HealingSession:
    """
    Create a self-healing session.

    A session establishes correlation only. It
    performs no infrastructure operation.
    """

    environment = (
        environment_id.strip()
    )

    incident = (
        incident_id.strip()
    )

    if not environment:
        raise ValueError(
            "environment_id is required"
        )

    if not incident:
        raise ValueError(
            "incident_id is required"
        )

    return HealingSession(
        environment_id=environment,
        incident_id=incident,
        decision_id=decision_id,
        plan_id=plan_id,
        action_id=action_id,
        resource_ids=list(
            resource_ids
        ),
        service_ids=list(
            service_ids
        ),
        attributes=dict(
            attributes
            or {}
        ),
    ).finalize()


# ---------------------------------------------------------------
# Remediation synchronization
# ---------------------------------------------------------------

def attach_remediation(
    session: HealingSession,
    remediation: RemediationResult,
) -> HealingSession:
    """
    Associate a completed/current remediation
    lifecycle with the healing session.

    This function does not execute remediation.
    """

    session.finalize()

    remediation.finalize()

    remediation_id = (
        remediation.remediation_id
    )

    session.remediation_id = (
        remediation_id
    )

    if not session.action_id:
        session.action_id = (
            remediation.request.
            action_id
        )

    session.evidence.append(
        HealingEvidence(
            evidence_type=(
                "remediation"
            ),
            description=(
                "Remediation lifecycle "
                f"reported status "
                f"{remediation.status}."
            ),
            score=(
                1.0
                if remediation.execution.
                success
                else 0.0
            ),
            source_id=(
                remediation_id
            ),
            resource_ids=list(
                remediation.request.
                target_resource_ids
            ),
            attributes={
                "remediation_status": (
                    remediation.status
                ),
                "execution_status": (
                    remediation.execution.
                    status
                ),
                "verification_status": (
                    remediation.verification.
                    status
                ),
                "rollback_status": (
                    remediation.rollback.
                    status
                ),
                "changed": (
                    remediation.execution.
                    changed
                ),
                "dry_run": (
                    remediation.request.mode
                    == "dry_run"
                ),
            },
        )
    )

    session.attributes[
        "remediation_status"
    ] = remediation.status

    session.attributes[
        "remediation_execution_status"
    ] = (
        remediation.execution.status
    )

    session.attributes[
        "remediation_verification_status"
    ] = (
        remediation.verification.status
    )

    session.attributes[
        "remediation_rollback_status"
    ] = (
        remediation.rollback.status
    )

    return session.finalize()


# ---------------------------------------------------------------
# Remediation -> healing state mapping
# ---------------------------------------------------------------

def state_from_remediation(
    remediation: RemediationResult,
) -> tuple[
    str,
    str,
]:
    """
    Translate remediation lifecycle output into a
    self-healing state.

    Dry-run completion does not imply recovery.
    """

    remediation.finalize()

    if (
        remediation.status
        == RemediationStatus.
        BLOCKED.value
    ):
        return (
            HealingState.
            ESCALATED.value,
            (
                "Remediation was blocked "
                "before execution."
            ),
        )

    if (
        remediation.status
        == RemediationStatus.
        FAILED.value
    ):
        return (
            HealingState.
            FAILED.value,
            (
                "Remediation failed."
            ),
        )

    if (
        remediation.request.mode
        == "dry_run"
    ):
        return (
            HealingState.
            OBSERVING.value,
            (
                "Dry-run remediation "
                "completed without changing "
                "infrastructure."
            ),
        )

    if (
        remediation.verification.status
        == VerificationStatus.
        PASSED.value
    ):
        return (
            HealingState.
            STABILIZING.value,
            (
                "Initial remediation "
                "verification passed; "
                "stability monitoring is "
                "required."
            ),
        )

    if (
        remediation.verification.status
        == VerificationStatus.
        FAILED.value
    ):
        if (
            remediation.rollback.status
            == RollbackStatus.
            SUCCEEDED.value
        ):
            return (
                HealingState.
                DEGRADED.value,
                (
                    "Recovery verification "
                    "failed and remediation "
                    "was rolled back."
                ),
            )

        return (
            HealingState.
            ESCALATED.value,
            (
                "Recovery verification "
                "failed."
            ),
        )

    return (
        HealingState.
        VERIFYING.value,
        (
            "Recovery has not yet been "
            "conclusively established."
        ),
    )


def synchronize_remediation(
    session: HealingSession,
    remediation: RemediationResult,
) -> HealingSession:
    """
    Attach remediation and move the healing
    session toward an appropriate lifecycle
    state.

    Where a direct transition is not valid, the
    caller should move through the required prior
    states first rather than bypassing them.
    """

    attach_remediation(
        session,
        remediation,
    )

    target, reason = (
        state_from_remediation(
            remediation
        )
    )

    if session.state == target:
        return session.finalize()

    if can_transition(
        session.state,
        target,
    ):
        transition(
            session,
            target,
            reason=reason,
            attributes={
                "remediation_id": (
                    remediation.
                      remediation_id
                ),
            },
        )

    else:
        session.limitations.append(
            (
                "Remediation suggests "
                f"state {target}, but "
                f"{session.state} -> "
                f"{target} is not a valid "
                "direct transition."
            )
        )

    return session.finalize()


# ---------------------------------------------------------------
# Recovery synchronization
# ---------------------------------------------------------------

def update_recovery(
    session: HealingSession,
    assessment: RecoveryAssessment,
) -> HealingSession:
    """
    Attach independent recovery evidence.

    Recovery state should come from monitoring,
    not merely from remediation execution.
    """

    assessment.finalize()

    session.recovery = (
        assessment
    )

    session.evidence.extend(
      assessment.evidence
    )

    if (
        assessment.status
        == RecoveryStatus.
        RECOVERED.value
    ):
        target = (
            HealingState.
            STABILIZING.value
        )

        reason = (
            "Recovery evidence indicates "
            "service health has returned; "
            "stability monitoring begins."
        )

    elif (
        assessment.status
        == RecoveryStatus.
        PARTIAL.value
    ):
        target = (
            HealingState.
            DEGRADED.value
        )

        reason = (
            "Only partial recovery was "
            "observed."
        )

    elif (
        assessment.status
        in {
            RecoveryStatus.
            FAILED.value,
            RecoveryStatus.
            UNSTABLE.value,
        }
    ):
        target = (
            HealingState.
            ESCALATED.value
        )

        reason = (
            "Recovery evidence indicates "
            "failure or instability."
        )

    else:
        target = (
            HealingState.
            VERIFYING.value
        )

        reason = (
            "Recovery remains under "
            "observation."
        )

    if can_transition(
        session.state,
        target,
    ):
        transition(
            session,
            target,
            reason=reason,
        )

    elif session.state != target:
        session.limitations.append(
            (
                "Recovery assessment "
                f"suggests {target}, but "
                f"{session.state} -> "
                f"{target} is not a valid "
                "direct transition."
            )
        )

    return session.finalize()


# ---------------------------------------------------------------
# Stability synchronization
# ---------------------------------------------------------------

def update_stability(
    session: HealingSession,
    assessment: StabilityAssessment,
) -> HealingSession:
    """
    Update stability after apparent recovery.
    """

    assessment.finalize()

    session.stability = (
        assessment
    )

    session.evidence.extend(
        assessment.evidence
    )

    if (
        assessment.status
        == StabilityStatus.
        STABLE.value
    ):
        target = (
            HealingState.
            RECOVERED.value
        )

        reason = (
            "Recovery remained stable "
            "through the observation window."
        )

    elif (
        assessment.status
        == StabilityStatus.
        REGRESSED.value
    ):
        target = (
            HealingState.
            DEGRADED.value
        )

        reason = (
            "Infrastructure regressed "
            "after apparent recovery."
        )

    elif (
        assessment.status
        == StabilityStatus.
        UNSTABLE.value
    ):
        target = (
            HealingState.
            ESCALATED.value
        )

        reason = (
            "Infrastructure remains "
            "unstable."
        )

    else:
        target = (
            HealingState.
            STABILIZING.value
        )

        reason = (
            "Stability observation is "
            "still in progress."
        )

    if can_transition(
        session.state,
        target,
    ):
        transition(
            session,
            target,
            reason=reason,
        )

    elif session.state != target:
        session.limitations.append(
            (
                "Stability assessment "
                f"suggests {target}, but "
                f"{session.state} -> "
                f"{target} is not a valid "
                "direct transition."
            )
        )

    return session.finalize()


# ---------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------

def escalate(
    session: HealingSession,
    *,
    level: str,
    reason: str,
    required_roles: Iterable[
        str
    ] = (),
) -> HealingSession:
    """
    Mark escalation requirements.

    This function does not send notifications or
    contact an operator.
    """

    if level not in {
        item.value
        for item
        in EscalationLevel
    }:
        raise ValueError(
            "Unknown escalation level: "
            f"{level}"
        )

    session.escalation.level = (
        level
    )

    session.escalation.required = (
        level
        != EscalationLevel.
        NONE.value
    )

    session.escalation.reason = (
        reason
    )

    session.escalation.required_roles = (
        list(
            required_roles
        )
    )

    session.escalation.incident_id = (
        session.incident_id
    )

    session.escalation.action_id = (
        session.action_id
    )

    session.escalation.remediation_id = (
        session.remediation_id
    )

    session.escalation.finalize()

    session.evidence.append(
        HealingEvidence(
            evidence_type="escalation",
            description=reason,
            score=(
                1.0
                if session.escalation.
                required
                else 0.0
            ),
            source_id=(
                session.escalation.
                escalation_id
            ),
            resource_ids=list(
                session.resource_ids
            ),
            attributes={
                "level": level,
                "required_roles": (
                    list(
                        required_roles
                    )
                ),
            },
        )
    )

    if (
        session.escalation.required
        and can_transition(
            session.state,
            HealingState.
            ESCALATED.value,
        )
    ):
        transition(
            session,
            HealingState.
            ESCALATED.value,
            reason=reason,
        )

    return session.finalize()


# ---------------------------------------------------------------
# Closure
# ---------------------------------------------------------------

def can_close(
    session: HealingSession,
) -> tuple[
    bool,
    str,
]:
    """
    Determine whether the self-healing session
    can be closed.

    A recovered session should have both recovery
    and stability evidence.
    """

    session.finalize()

    if (
        session.state
        == HealingState.
        RECOVERED.value
    ):
        if (
            session.recovery.status
            != RecoveryStatus.
            RECOVERED.value
        ):
            return (
                False,
                (
                    "Recovery has not been "
                    "recorded as recovered."
                ),
            )

        if (
            session.stability.status
            != StabilityStatus.
            STABLE.value
        ):
            return (
                False,
                (
                    "Recovery has not been "
                    "shown to remain stable."
                ),
            )

        return (
            True,
            (
                "Recovery and stability "
                "requirements are satisfied."
            ),
        )

    if (
        session.state
        in {
            HealingState.
            ESCALATED.value,
            HealingState.
            FAILED.value,
        }
    ):
        return (
            True,
            (
                "Session can be closed after "
                "handoff/escalation."
            ),
        )

    return (
        False,
        (
            "Session lifecycle is not ready "
            "for closure."
        ),
    )


def close_session(
    session: HealingSession,
    *,
    reason: Optional[
        str
    ] = None,
) -> HealingSession:
    allowed, explanation = (
        can_close(
            session
        )
    )

    if not allowed:
        raise InvalidTransitionError(
            explanation
        )

    transition(
        session,
        HealingState.CLOSED.value,
        reason=(
            reason
            or explanation
        ),
    )

    return session.finalize()


# ---------------------------------------------------------------
# Controller
# ---------------------------------------------------------------

class SelfHealingController:
    """
    State-machine coordinator for the OSHA
    self-healing lifecycle.

    It coordinates state only.

    It does not execute remediation, invoke
    infrastructure commands, or bypass approval.
    """

    def create(
        self,
        *,
        environment_id: str,
        incident_id: str,
        decision_id: Optional[
            str
        ] = None,
        plan_id: Optional[
            str
        ] = None,
        action_id: Optional[
            str
        ] = None,
        resource_ids: Iterable[
            str
        ] = (),
        service_ids: Iterable[
            str
        ] = (),
    ) -> HealingSession:
        return create_session(
            environment_id=(
                environment_id
            ),
            incident_id=incident_id,
            decision_id=decision_id,
            plan_id=plan_id,
            action_id=action_id,
            resource_ids=(
                resource_ids
            ),
            service_ids=(
                service_ids
            ),
        )

    def transition(
        self,
        session: HealingSession,
        next_state: str,
        *,
        reason: str,
    ) -> StateTransition:
        return transition(
            session,
            next_state,
            reason=reason,
        )

    def remediation(
        self,
        session: HealingSession,
        result: RemediationResult,
    ) -> HealingSession:
        return synchronize_remediation(
            session,
            result,
        )

    def recovery(
        self,
        session: HealingSession,
        assessment: RecoveryAssessment,
    ) -> HealingSession:
        return update_recovery(
            session,
            assessment,
        )

    def stability(
        self,
        session: HealingSession,
        assessment: StabilityAssessment,
    ) -> HealingSession:
        return update_stability(
            session,
            assessment,
        )

    def escalate(
        self,
        session: HealingSession,
        *,
        level: str,
        reason: str,
        required_roles: Iterable[
            str
        ] = (),
    ) -> HealingSession:
        return escalate(
            session,
            level=level,
            reason=reason,
            required_roles=(
                required_roles
            ),
        )

    def close(
        self,
        session: HealingSession,
        *,
        reason: Optional[
            str
        ] = None,
    ) -> HealingSession:
        return close_session(
            session,
            reason=reason,
        )