from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.remediation.schema import (
    RemediationMode,
    RemediationResult,
    RollbackStatus,
)

from src.self_healing.schema import (
    EscalationLevel,
    HealingEvidence,
    HealingOutcome,
    HealingSession,
    OutcomeStatus,
    RecoveryStatus,
    StabilityStatus,
)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class OutcomeConfig:
    """
    OSHA healing-outcome policy.

    These rules determine how recovery,
    stability, rollback and escalation are
    translated into an operational outcome.
    """

    require_verified_recovery: bool = True

    require_stability_for_recovered: bool = True

    mark_partial_recovery: bool = True

    rollback_overrides_recovery: bool = True

    escalation_overrides_partial: bool = True

    dry_run_is_never_recovered: bool = True


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------


def remediation_is_dry_run(
    remediation: Optional[RemediationResult],
) -> bool:
    if remediation is None:
        return False

    return remediation.request.mode == RemediationMode.DRY_RUN.value


def remediation_rolled_back(
    remediation: Optional[RemediationResult],
) -> bool:
    if remediation is None:
        return False

    return remediation.rollback.status == RollbackStatus.SUCCEEDED.value


def recovery_verified(
    session: HealingSession,
) -> bool:
    return session.recovery.status == RecoveryStatus.RECOVERED.value


def recovery_stable(
    session: HealingSession,
) -> bool:
    return session.stability.status == StabilityStatus.STABLE.value


def escalation_required(
    session: HealingSession,
) -> bool:
    return bool(
        session.escalation.required
        and session.escalation.level != EscalationLevel.NONE.value
    )


# ---------------------------------------------------------------
# Classification
# ---------------------------------------------------------------


def classify_outcome(
    session: HealingSession,
    *,
    remediation: Optional[RemediationResult] = None,
    config: Optional[OutcomeConfig] = None,
) -> tuple[
    str,
    str,
]:
    """
    Determine the final healing outcome.

    Ordering is intentional:

      dry-run boundary
      rollback
      stable verified recovery
      escalation
      partial recovery
      failed recovery
      unknown

    The exact ordering is OSHA policy.
    """

    configuration = config or OutcomeConfig()

    session.finalize()

    if remediation is not None:
        remediation.finalize()

    # ---------------------------------------------------------
    # Dry-run boundary
    # ---------------------------------------------------------

    if configuration.dry_run_is_never_recovered and remediation_is_dry_run(remediation):
        return (
            OutcomeStatus.UNKNOWN.value,
            (
                "Dry-run completed without "
                "changing infrastructure; "
                "no real recovery outcome "
                "can be claimed."
            ),
        )

    # ---------------------------------------------------------
    # Rollback
    # ---------------------------------------------------------

    if configuration.rollback_overrides_recovery and remediation_rolled_back(
        remediation
    ):
        return (
            OutcomeStatus.ROLLED_BACK.value,
            ("The remediation was rolled back."),
        )

    # ---------------------------------------------------------
    # Full recovery
    # ---------------------------------------------------------

    recovered = recovery_verified(session)

    stable = recovery_stable(session)

    verified_ok = recovered if configuration.require_verified_recovery else True

    stability_ok = stable if configuration.require_stability_for_recovered else True

    if verified_ok and stability_ok and recovered:
        return (
            OutcomeStatus.RECOVERED.value,
            ("Recovery was established and remained stable."),
        )

    # ---------------------------------------------------------
    # Escalation
    # ---------------------------------------------------------

    if escalation_required(session):
        return (
            OutcomeStatus.ESCALATED.value,
            (
                "Automated healing did not "
                "reach a safely closable "
                "state and requires human "
                "escalation."
            ),
        )

    # ---------------------------------------------------------
    # Partial recovery
    # ---------------------------------------------------------

    if (
        configuration.mark_partial_recovery
        and session.recovery.status == RecoveryStatus.PARTIAL.value
    ):
        return (
            OutcomeStatus.PARTIALLY_RECOVERED.value,
            ("Some recovery was observed but full recovery was not established."),
        )

    # ---------------------------------------------------------
    # Recovery followed by instability
    # ---------------------------------------------------------

    if recovered and session.stability.status in {
        StabilityStatus.UNSTABLE.value,
        StabilityStatus.REGRESSED.value,
    }:
        return (
            OutcomeStatus.NOT_RECOVERED.value,
            ("Initial recovery did not remain stable."),
        )

    # ---------------------------------------------------------
    # Explicit recovery failure
    # ---------------------------------------------------------

    if session.recovery.status in {
        RecoveryStatus.FAILED.value,
        RecoveryStatus.UNSTABLE.value,
    }:
        return (
            OutcomeStatus.NOT_RECOVERED.value,
            (
                "Recovery evidence indicates "
                "the infrastructure did not "
                "recover successfully."
            ),
        )

    # ---------------------------------------------------------
    # Still monitoring
    # ---------------------------------------------------------

    return (
        OutcomeStatus.UNKNOWN.value,
        ("There is not yet enough evidence for a final recovery outcome."),
    )


# ---------------------------------------------------------------
# Evidence construction
# ---------------------------------------------------------------


def outcome_evidence(
    session: HealingSession,
    *,
    remediation: Optional[RemediationResult] = None,
) -> List[HealingEvidence]:
    """
    Collect bounded evidence references for the
    final outcome.

    This reuses normalized evidence rather than
    copying raw adapter output.
    """

    evidence: List[HealingEvidence] = []

    evidence.extend(session.recovery.evidence)

    evidence.extend(session.stability.evidence)

    evidence.extend(session.escalation.evidence)

    if remediation is not None:
        evidence.append(
            HealingEvidence(
                evidence_type=("remediation_outcome"),
                description=(
                    "Healing outcome includes the final remediation lifecycle state."
                ),
                score=(1.0 if (remediation.execution.success) else 0.0),
                source_id=(remediation.remediation_id),
                resource_ids=list(remediation.request.target_resource_ids),
                attributes={
                    "remediation_status": (remediation.status),
                    "execution_status": (remediation.execution.status),
                    "execution_changed": (remediation.execution.changed),
                    "verification_status": (remediation.verification.status),
                    "rollback_status": (remediation.rollback.status),
                    "dry_run": (
                        remediation.request.mode == RemediationMode.DRY_RUN.value
                    ),
                },
            )
        )

    return evidence


# ---------------------------------------------------------------
# Build outcome
# ---------------------------------------------------------------


def build_outcome(
    session: HealingSession,
    *,
    remediation: Optional[RemediationResult] = None,
    config: Optional[OutcomeConfig] = None,
) -> HealingOutcome:
    """
    Build and attach a final HealingOutcome.

    This method does not persist memory itself.
    outcomes.py establishes the outcome contract;
    the memory bridge belongs in the pipeline or
    memory integration.
    """

    configuration = config or OutcomeConfig()

    session.finalize()

    status, summary = classify_outcome(
        session,
        remediation=remediation,
        config=configuration,
    )

    dry_run = remediation_is_dry_run(remediation)

    rolled_back = remediation_rolled_back(remediation)

    escalated = escalation_required(session)

    recovered = status == OutcomeStatus.RECOVERED.value

    stable = recovered and recovery_stable(session)

    outcome = HealingOutcome(
        status=status,
        incident_id=(session.incident_id),
        decision_id=(session.decision_id),
        action_id=(session.action_id),
        remediation_id=(
            (remediation.remediation_id)
            if remediation is not None
            else session.remediation_id
        ),
        recovery_score=(session.recovery.score),
        stability_score=(session.stability.score),
        recovery_verified=(recovered and not dry_run),
        stable=stable,
        rolled_back=(rolled_back),
        escalated=(escalated),
        summary=summary,
        evidence=outcome_evidence(
            session,
            remediation=remediation,
        ),
        attributes={
            "environment_id": (session.environment_id),
            "recovery_status": (session.recovery.status),
            "stability_status": (session.stability.status),
            "escalation_level": (session.escalation.level),
            "dry_run": dry_run,
            "remediation_status": (
                remediation.status if remediation is not None else None
            ),
            "healing_attempts": (
                session.attributes.get(
                    "healing_attempts",
                    0,
                )
            ),
            "failed_healing_attempts": (
                session.attributes.get(
                    "failed_healing_attempts",
                    0,
                )
            ),
            "unresolved_cycles": (
                session.attributes.get(
                    "unresolved_cycles",
                    0,
                )
            ),
        },
    ).finalize()

    session.outcome = outcome

    session.attributes["outcome_status"] = outcome.status

    session.attributes["outcome_id"] = outcome.outcome_id

    session.finalize()

    return outcome


# ---------------------------------------------------------------
# Memory-safe outcome representation
# ---------------------------------------------------------------


def outcome_memory_payload(
    outcome: HealingOutcome,
) -> Dict[str, Any]:
    """
    Produce a provider-independent payload that
    the memory layer can translate into its own
    outcome schema.

    This function does not write memory.
    """

    outcome.finalize()

    return {
        "outcome_id": (outcome.outcome_id),
        "status": outcome.status,
        "incident_id": (outcome.incident_id),
        "decision_id": (outcome.decision_id),
        "action_id": (outcome.action_id),
        "remediation_id": (outcome.remediation_id),
        "recovery_score": (outcome.recovery_score),
        "stability_score": (outcome.stability_score),
        "recovery_verified": (outcome.recovery_verified),
        "stable": outcome.stable,
        "rolled_back": (outcome.rolled_back),
        "escalated": (outcome.escalated),
        "summary": (outcome.summary),
        "completed_at": (outcome.completed_at),
        "attributes": dict(outcome.attributes),
    }


def outcome_is_verified_success(
    outcome: HealingOutcome,
) -> bool:
    """
    Return True only for a real, verified and
    stable recovery.

    Dry-run outcomes therefore cannot qualify.
    """

    outcome.finalize()

    return bool(
        outcome.status == OutcomeStatus.RECOVERED.value
        and outcome.recovery_verified
        and outcome.stable
        and not outcome.rolled_back
        and not outcome.escalated
        and not bool(
            outcome.attributes.get(
                "dry_run",
                False,
            )
        )
    )


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------


def outcomes_summary(
    outcomes: Iterable[HealingOutcome],
) -> Dict[str, Any]:
    values = list(outcomes)

    statuses: Dict[
        str,
        int,
    ] = {}

    for outcome in values:
        statuses[outcome.status] = (
            statuses.get(
                outcome.status,
                0,
            )
            + 1
        )

    return {
        "outcome_count": (len(values)),
        "recovered_count": sum(
            1 for outcome in values if (outcome.status == OutcomeStatus.RECOVERED.value)
        ),
        "verified_success_count": sum(
            1 for outcome in values if (outcome_is_verified_success(outcome))
        ),
        "partial_count": sum(
            1
            for outcome in values
            if (outcome.status == OutcomeStatus.PARTIALLY_RECOVERED.value)
        ),
        "not_recovered_count": sum(
            1
            for outcome in values
            if (outcome.status == OutcomeStatus.NOT_RECOVERED.value)
        ),
        "rolled_back_count": sum(
            1
            for outcome in values
            if (outcome.status == OutcomeStatus.ROLLED_BACK.value)
        ),
        "escalated_count": sum(
            1 for outcome in values if (outcome.status == OutcomeStatus.ESCALATED.value)
        ),
        "unknown_count": sum(
            1 for outcome in values if (outcome.status == OutcomeStatus.UNKNOWN.value)
        ),
        "statuses": statuses,
    }


class OutcomeEvaluator:
    """
    Final healing-outcome evaluator.

    This component classifies and packages
    outcomes. It does not execute remediation or
    persist memory.
    """

    def __init__(
        self,
        *,
        config: Optional[OutcomeConfig] = None,
    ) -> None:
        self.config = config or OutcomeConfig()

    def classify(
        self,
        session: HealingSession,
        *,
        remediation: Optional[RemediationResult] = None,
    ) -> tuple[
        str,
        str,
    ]:
        return classify_outcome(
            session,
            remediation=remediation,
            config=self.config,
        )

    def build(
        self,
        session: HealingSession,
        *,
        remediation: Optional[RemediationResult] = None,
    ) -> HealingOutcome:
        return build_outcome(
            session,
            remediation=remediation,
            config=self.config,
        )

    def verified_success(
        self,
        outcome: HealingOutcome,
    ) -> bool:
        return outcome_is_verified_success(outcome)
