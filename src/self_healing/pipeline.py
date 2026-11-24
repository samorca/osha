from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.decision.schema import (
    ActionCandidate,
    Decision,
    DecisionPlan,
)

from src.remediation.pipeline import (
    RemediationPipeline,
    request_from_action,
)

from src.remediation.schema import (
    RemediationMode,
    RemediationResult,
)

from src.self_healing.controller import (
    SelfHealingController,
    attach_remediation,
    can_transition,
    transition,
)

from src.self_healing.escalation import (
    EscalationEvaluator,
)

from src.self_healing.outcomes import (
    OutcomeEvaluator,
    outcome_memory_payload,
)

from src.self_healing.recovery import (
    RecoveryEvaluator,
    RecoveryObservation,
)

from src.self_healing.schema import (
    HealingSession,
    HealingState,
    SelfHealingResult,
)

from src.self_healing.stabilization import (
    StabilizationEvaluator,
    StabilityObservation,
)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class SelfHealingConfig:
    """
    Top-level self-healing configuration.

    Default remediation mode is dry_run.

    Changing this value to execute is not enough
    to enable execution. The remediation
    guardrails, executor and adapter registry
    must independently permit execution.
    """

    remediation_mode: str = RemediationMode.DRY_RUN.value

    remediation_request_ttl_minutes: int = 30

    require_eligible_action: bool = True

    stop_on_escalation: bool = True

    close_verified_recovery: bool = True

    def validate(
        self,
    ) -> None:
        if self.remediation_mode not in {item.value for item in RemediationMode}:
            raise ValueError(f"Unsupported remediation mode: {self.remediation_mode}")

        if self.remediation_request_ttl_minutes < 1:
            raise ValueError("remediation_request_ttl_minutes must be >= 1")


# ---------------------------------------------------------------
# Output
# ---------------------------------------------------------------


@dataclass
class HealingCycleResult:
    """
    Complete result for one healing cycle.

    memory_payload is an integration-neutral
    representation. This module does not
    automatically write memory.
    """

    session: HealingSession

    remediation: RemediationResult

    memory_payload: Dict[
        str,
        Any,
    ]

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "session": (self.session.to_dict()),
            "remediation": (self.remediation.to_dict()),
            "memory_payload": dict(self.memory_payload),
        }


# ---------------------------------------------------------------
# State progression
# ---------------------------------------------------------------


def advance_to_remediation(
    session: HealingSession,
) -> HealingSession:
    """
    Advance a newly created healing session
    through its pre-remediation states.

    The path remains explicit so the session
    history shows each lifecycle stage.
    """

    path = (
        (
            HealingState.DETECTED.value,
            ("Incident entered the self-healing controller."),
        ),
        (
            HealingState.ANALYZING.value,
            ("Incident evidence is being analyzed."),
        ),
        (
            HealingState.DECIDING.value,
            ("A decision action has been selected for evaluation."),
        ),
        (
            HealingState.REMEDIATING.value,
            ("The controlled remediation lifecycle is starting."),
        ),
    )

    for state, reason in path:
        if session.state == state:
            continue

        if not can_transition(
            session.state,
            state,
        ):
            raise ValueError(
                f"Cannot advance self-healing session from {session.state} to {state}"
            )

        transition(
            session,
            state,
            reason=reason,
        )

    return session.finalize()


# ---------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------


def validate_action_for_healing(
    action: ActionCandidate,
    *,
    config: SelfHealingConfig,
) -> None:
    """
    Reject ineligible or structurally incomplete
    decision actions before remediation request
    construction.
    """

    action.finalize()

    if config.require_eligible_action and not action.eligibility.eligible:
        raise ValueError("Self-healing pipeline requires an eligible decision action")

    if not action.action_id:
        raise ValueError("Decision action must have an action_id")

    if not action.environment_id:
        raise ValueError("Decision action must have an environment_id")


def incident_id_for(
    action: ActionCandidate,
    decision: Optional[Decision],
    plan: Optional[DecisionPlan],
) -> str:
    """
    Resolve incident identity without inventing
    one.
    """

    if decision is not None and decision.incident_id:
        return decision.incident_id

    if plan is not None and plan.incident_id:
        return plan.incident_id

    incident_id = action.attributes.get("incident_id")

    if incident_id:
        return str(incident_id)

    raise ValueError("Self-healing cycle requires an incident_id")


# ---------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------


class SelfHealingPipeline:
    """
    Controlled self-healing feedback loop.

    Default construction uses a dry-run-only
    remediation pipeline.
    """

    def __init__(
        self,
        *,
        remediation_pipeline: Optional[RemediationPipeline] = None,
        controller: Optional[SelfHealingController] = None,
        recovery_evaluator: Optional[RecoveryEvaluator] = None,
        stabilization_evaluator: Optional[StabilizationEvaluator] = None,
        escalation_evaluator: Optional[EscalationEvaluator] = None,
        outcome_evaluator: Optional[OutcomeEvaluator] = None,
        config: Optional[SelfHealingConfig] = None,
    ) -> None:
        self.config = config or SelfHealingConfig()

        self.config.validate()

        self.remediation_pipeline = remediation_pipeline or RemediationPipeline()

        self.controller = controller or SelfHealingController()

        self.recovery_evaluator = recovery_evaluator or RecoveryEvaluator()

        self.stabilization_evaluator = (
            stabilization_evaluator or StabilizationEvaluator()
        )

        self.escalation_evaluator = escalation_evaluator or EscalationEvaluator()

        self.outcome_evaluator = outcome_evaluator or OutcomeEvaluator()

    def run(
        self,
        action: ActionCandidate,
        *,
        decision: Optional[Decision] = None,
        plan: Optional[DecisionPlan] = None,
        recovery_observations: Iterable[RecoveryObservation] = (),
        stability_observations: Iterable[StabilityObservation] = (),
        health_before: Optional[float] = None,
        requested_by: Optional[str] = None,
    ) -> HealingCycleResult:
        """
        Run one controlled healing cycle.

        The default run performs dry-run
        remediation and therefore cannot claim
        real recovery.
        """

        validate_action_for_healing(
            action,
            config=self.config,
        )

        incident_id = incident_id_for(
            action,
            decision,
            plan,
        )

        # -----------------------------------------------------
        # Create session
        # -----------------------------------------------------

        session = self.controller.create(
            environment_id=(action.environment_id),
            incident_id=incident_id,
            decision_id=(decision.decision_id if decision else None),
            plan_id=(plan.plan_id if plan else None),
            action_id=(action.action_id),
            resource_ids=(action.target_resource_ids),
            service_ids=(action.target_service_ids),
        )

        session.attributes.update(
            {
                "incident_severity": (
                    decision.attributes.get(
                        "incident_severity",
                        "unknown",
                    )
                    if decision is not None
                    else str(
                        action.attributes.get(
                            "incident_severity",
                            "unknown",
                        )
                    )
                ),
                "remediation_mode": (self.config.remediation_mode),
                "healing_attempts": 1,
            }
        )

        advance_to_remediation(session)

        # -----------------------------------------------------
        # Remediation
        # -----------------------------------------------------

        request = request_from_action(
            action,
            decision=decision,
            plan=plan,
            mode=(self.config.remediation_mode),
            requested_by=(requested_by),
            expires_in_minutes=(self.config.remediation_request_ttl_minutes),
        )

        remediation = self.remediation_pipeline.run(
            request,
            actor_id=requested_by,
        )

        attach_remediation(
            session,
            remediation,
        )

        # -----------------------------------------------------
        # Recovery
        # -----------------------------------------------------

        recovery = self.recovery_evaluator.assess(
            remediation,
            observations=(recovery_observations),
            health_before=(health_before),
            target_resource_ids=(action.target_resource_ids),
        )

        self.controller.recovery(
            session,
            recovery,
        )

        # -----------------------------------------------------
        # Stability
        # -----------------------------------------------------

        stability = self.stabilization_evaluator.assess(
            recovery,
            observations=(stability_observations),
            target_resource_ids=(action.target_resource_ids),
        )

        self.controller.stability(
            session,
            stability,
        )

        # -----------------------------------------------------
        # Escalation
        # -----------------------------------------------------

        escalation = self.escalation_evaluator.assess(session)

        if escalation.required:
            self.controller.escalate(
                session,
                level=(escalation.level),
                reason=(escalation.reason or ("Human escalation is required.")),
                required_roles=(escalation.required_roles),
            )

        # -----------------------------------------------------
        # Outcome
        # -----------------------------------------------------

        outcome = self.outcome_evaluator.build(
            session,
            remediation=remediation,
        )

        # Only real, verified and stable recovery
        # can close automatically.
        if (
            self.config.close_verified_recovery
            and self.outcome_evaluator.verified_success(outcome)
            and can_transition(
                session.state,
                HealingState.CLOSED.value,
            )
        ):
            self.controller.close(
                session,
                reason=(
                    "Verified recovery "
                    "remained stable and "
                    "the healing cycle is "
                    "complete."
                ),
            )

        if escalation.required and self.config.stop_on_escalation:
            session.attributes["automation_stopped"] = True

        else:
            session.attributes["automation_stopped"] = False

        session.finalize()

        return HealingCycleResult(
            session=session,
            remediation=remediation,
            memory_payload=(outcome_memory_payload(outcome)),
        )

    def run_many(
        self,
        items: Iterable[
            tuple[
                ActionCandidate,
                Optional[Decision],
                Optional[DecisionPlan],
            ]
        ],
    ) -> SelfHealingResult:
        """
        Run multiple independent healing cycles.

        Each tuple contains:

          action
          optional decision
          optional plan
        """

        sessions: List[HealingSession] = []

        for (
            action,
            decision,
            plan,
        ) in items:
            result = self.run(
                action,
                decision=decision,
                plan=plan,
            )

            sessions.append(result.session)

        return SelfHealingResult(
            sessions=sessions,
            metadata={
                "remediation_mode": (self.config.remediation_mode),
                "execution_expected": (
                    self.config.remediation_mode == RemediationMode.EXECUTE.value
                ),
            },
        ).finalize()


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------


def self_healing_summary(
    result: SelfHealingResult,
) -> Dict[str, Any]:
    result.finalize()

    return {
        "session_count": len(result.sessions),
        "recovered_count": sum(
            1
            for session in result.sessions
            if (session.outcome and session.outcome.status == "recovered")
        ),
        "escalated_count": sum(
            1 for session in result.sessions if session.escalation.required
        ),
        "dry_run_count": sum(
            1
            for session in result.sessions
            if (
                session.attributes.get("remediation_mode")
                == RemediationMode.DRY_RUN.value
            )
        ),
        "automation_stopped_count": (
            sum(
                1
                for session in result.sessions
                if (session.attributes.get("automation_stopped") is True)
            )
        ),
    }
