from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.decision.policy import (
    PolicyEffect,
    PolicyEngine,
    PolicyEvaluation,
)

from src.decision.schema import (
    ActionCandidate,
    ActionCategory,
    ActionState,
    ApprovalMode,
    DecisionContext,
    DecisionEvidence,
    EligibilityAssessment,
    EligibilityStatus,
    Reversibility,
    RiskLevel,
    checked_score,
)


MUTATING_CATEGORIES = {
    ActionCategory.CONFIGURATION.value,
    ActionCategory.SERVICE_CONTROL.value,
    ActionCategory.RESOURCE_CONTROL.value,
    ActionCategory.NETWORK_CONTROL.value,
    ActionCategory.SECURITY_CONTROL.value,
    ActionCategory.CAPACITY.value,
    ActionCategory.RECOVERY.value,
}


READ_ONLY_CATEGORIES = {
    ActionCategory.OBSERVATION.value,
    ActionCategory.INVESTIGATION.value,
    ActionCategory.NOTIFICATION.value,
}


@dataclass(frozen=True)
class EligibilityConfig:
    """
    OSHA action-eligibility configuration.

    These are development policy defaults.

    They determine whether an ActionCandidate can
    continue through planning. They do not grant
    execution authority.
    """

    minimum_investigation_confidence: float = 0.20

    minimum_recommendation_confidence: float = 0.40

    minimum_mutation_confidence: float = 0.70

    maximum_automatic_risk: float = 0.20

    maximum_eligible_risk: float = 0.85

    require_target_for_mutation: bool = True

    require_root_cause_for_mutation: bool = True

    require_rollback_for_mutation: bool = True

    block_unverified_critical_risk: bool = True

    allow_read_only_without_root_cause: bool = True

    def validate(
        self,
    ) -> None:
        scores = (
            self.minimum_investigation_confidence,
            self.minimum_recommendation_confidence,
            self.minimum_mutation_confidence,
            self.maximum_automatic_risk,
            self.maximum_eligible_risk,
        )

        if any(value < 0.0 or value > 1.0 for value in scores):
            raise ValueError("Eligibility scores must be between 0 and 1")

        if self.maximum_automatic_risk > self.maximum_eligible_risk:
            raise ValueError(
                "maximum_automatic_risk must not exceed maximum_eligible_risk"
            )


# ---------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------


def action_is_mutating(
    action: ActionCandidate,
) -> bool:
    return action.category in MUTATING_CATEGORIES


def action_is_read_only(
    action: ActionCandidate,
) -> bool:
    return action.category in READ_ONLY_CATEGORIES


# ---------------------------------------------------------------
# Hard blockers
# ---------------------------------------------------------------


def policy_blockers(
    evaluation: PolicyEvaluation,
) -> List[str]:
    if evaluation.allowed:
        return []

    return list(evaluation.blocking_reasons) or [
        "Configured policy does not permit this action."
    ]


def target_blockers(
    action: ActionCandidate,
    *,
    config: EligibilityConfig,
) -> List[str]:
    blockers = []

    if (
        action_is_mutating(action)
        and config.require_target_for_mutation
        and not (action.target_resource_ids or action.target_service_ids)
    ):
        blockers.append("Mutating action has no explicit target resource or service.")

    return blockers


def confidence_blockers(
    action: ActionCandidate,
    context: DecisionContext,
    *,
    config: EligibilityConfig,
) -> List[str]:
    blockers = []

    confidence = checked_score(context.incident_confidence)

    if action_is_mutating(action):
        if confidence < config.minimum_mutation_confidence:
            blockers.append(
                "Incident confidence "
                f"{confidence:.2f} is below "
                "the mutation eligibility "
                f"minimum of "
                f"{config.minimum_mutation_confidence:.2f}."
            )

    elif action.category == ActionCategory.NOTIFICATION.value:
        if confidence < config.minimum_recommendation_confidence:
            blockers.append(
                "Incident confidence is too "
                "low for the configured "
                "recommendation threshold."
            )

    elif action_is_read_only(action):
        if confidence < config.minimum_investigation_confidence:
            blockers.append(
                "Incident confidence is below "
                "the minimum configured for "
                "investigation planning."
            )

    return blockers


def root_cause_blockers(
    action: ActionCandidate,
    context: DecisionContext,
    *,
    config: EligibilityConfig,
) -> List[str]:
    blockers = []

    if not action_is_mutating(action):
        return blockers

    if not (config.require_root_cause_for_mutation):
        return blockers

    if not (context.root_cause_resource_id):
        blockers.append("Mutating action requires a root-cause hypothesis.")

        return blockers

    if context.root_cause_score <= 0:
        blockers.append("Root-cause hypothesis has no usable evidence score.")

    return blockers


def reversibility_blockers(
    action: ActionCandidate,
    *,
    config: EligibilityConfig,
) -> List[str]:
    blockers = []

    if not action_is_mutating(action):
        return blockers

    if not (config.require_rollback_for_mutation):
        return blockers

    if action.reversibility == Reversibility.NONE.value:
        blockers.append("Irreversible infrastructure action is not eligible.")

        return blockers

    if action.reversibility in {
        Reversibility.PARTIAL.value,
        Reversibility.UNKNOWN.value,
    }:
        blockers.append(
            "Mutating action must have full reversibility before it becomes eligible."
        )

    if (
        action.reversibility == Reversibility.FULL.value
        and not action.rollback_action_type
    ):
        blockers.append(
            "Action claims full reversibility but has no rollback action type."
        )

    return blockers


def risk_blockers(
    action: ActionCandidate,
    context: DecisionContext,
    *,
    config: EligibilityConfig,
) -> List[str]:
    blockers = []

    risk = checked_score(action.risk.score)

    if risk > config.maximum_eligible_risk:
        blockers.append("Action risk exceeds the maximum eligibility threshold.")

    if (
        config.block_unverified_critical_risk
        and action.risk.level == RiskLevel.CRITICAL.value
        and not context.root_cause_verified
    ):
        blockers.append("Critical-risk action requires verified root-cause evidence.")

    return blockers


def approval_blockers(
    action: ActionCandidate,
) -> List[str]:
    if action.approval.mode == ApprovalMode.FORBIDDEN.value:
        return [("Action is forbidden by its approval/policy state.")]

    return []


# ---------------------------------------------------------------
# Conditional requirements
# ---------------------------------------------------------------


def eligibility_conditions(
    action: ActionCandidate,
    context: DecisionContext,
    policy: PolicyEvaluation,
    *,
    config: EligibilityConfig,
) -> List[str]:
    """
    Conditions that don't block planning but must
    be satisfied before future execution.
    """

    conditions = []

    if policy.effect == PolicyEffect.ALLOW_WITH_APPROVAL:
        conditions.append("Explicit approval is required before execution.")

    if action.approval.required:
        conditions.append(
            "Approval requirement must be resolved before remediation handoff."
        )

    if action_is_mutating(action) and not context.root_cause_verified:
        conditions.append("Root-cause evidence remains unverified.")

    if action.risk.score > config.maximum_automatic_risk:
        conditions.append("Action risk exceeds the automatic-action threshold.")

    if not (context.user_impact_known):
        conditions.append("User/business impact remains unverified.")

    return list(dict.fromkeys(conditions))


def satisfied_requirements(
    action: ActionCandidate,
    context: DecisionContext,
    policy: PolicyEvaluation,
    *,
    config: EligibilityConfig,
) -> List[str]:
    output = []

    if policy.allowed:
        output.append("policy_allows_action")

    if action.target_resource_ids or action.target_service_ids:
        output.append("explicit_target_present")

    if context.incident_confidence >= config.minimum_mutation_confidence:
        output.append("strong_incident_confidence")

    if context.root_cause_resource_id:
        output.append("root_cause_hypothesis_present")

    if context.root_cause_verified:
        output.append("root_cause_verified")

    if action.reversibility == Reversibility.FULL.value and action.rollback_action_type:
        output.append("rollback_defined")

    if action.risk.score <= config.maximum_eligible_risk:
        output.append("risk_within_eligibility_limit")

    return output


# ---------------------------------------------------------------
# Eligibility score
# ---------------------------------------------------------------


def eligibility_score(
    action: ActionCandidate,
    context: DecisionContext,
    policy: PolicyEvaluation,
    *,
    config: EligibilityConfig,
) -> float:
    """
    Produce a transparent evidence-strength score
    for eligibility.

    Hard blockers still override this score.
    """

    policy_score = 1.0 if policy.allowed else 0.0

    confidence_score = checked_score(context.incident_confidence)

    root_cause_score = checked_score(context.root_cause_score)

    verification_score = (
        1.0
        if context.root_cause_verified
        else (0.50 if context.root_cause_resource_id else 0.0)
    )

    reversibility_score = {
        Reversibility.FULL.value: (1.0 if action.rollback_action_type else 0.70),
        Reversibility.PARTIAL.value: (0.40),
        Reversibility.NONE.value: (0.0),
        Reversibility.UNKNOWN.value: (0.20),
    }.get(
        action.reversibility,
        0.20,
    )

    risk_score = 1.0 - checked_score(action.risk.score)

    if action_is_read_only(action):
        return checked_score(
            (policy_score * 0.35) + (confidence_score * 0.35) + (risk_score * 0.30)
        )

    return checked_score(
        (policy_score * 0.25)
        + (confidence_score * 0.20)
        + (root_cause_score * 0.15)
        + (verification_score * 0.15)
        + (reversibility_score * 0.10)
        + (risk_score * 0.15)
    )


# ---------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------


def assess_eligibility(
    action: ActionCandidate,
    context: DecisionContext,
    *,
    policy_engine: Optional[PolicyEngine] = None,
    config: Optional[EligibilityConfig] = None,
) -> EligibilityAssessment:
    """
    Evaluate one action's planning eligibility.

    The result does not approve or execute the
    action.
    """

    configuration = config or EligibilityConfig()

    configuration.validate()

    action.finalize()

    context.finalize()

    policies = policy_engine or PolicyEngine()

    policy = policies.evaluate(
        action,
        context,
    )

    blockers = []

    blockers.extend(policy_blockers(policy))

    blockers.extend(
        target_blockers(
            action,
            config=configuration,
        )
    )

    blockers.extend(
        confidence_blockers(
            action,
            context,
            config=configuration,
        )
    )

    blockers.extend(
        root_cause_blockers(
            action,
            context,
            config=configuration,
        )
    )

    blockers.extend(
        reversibility_blockers(
            action,
            config=configuration,
        )
    )

    blockers.extend(
        risk_blockers(
            action,
            context,
            config=configuration,
        )
    )

    blockers.extend(approval_blockers(action))

    blockers = list(dict.fromkeys(blockers))

    conditions = eligibility_conditions(
        action,
        context,
        policy,
        config=configuration,
    )

    satisfied = satisfied_requirements(
        action,
        context,
        policy,
        config=configuration,
    )

    score = eligibility_score(
        action,
        context,
        policy,
        config=configuration,
    )

    if blockers:
        status = EligibilityStatus.INELIGIBLE.value

        eligible = False

    elif conditions:
        status = EligibilityStatus.CONDITIONAL.value

        # Conditional means eligible for further
        # decision/approval planning, not
        # execution.
        eligible = True

    else:
        status = EligibilityStatus.ELIGIBLE.value

        eligible = True

    evidence = [
        DecisionEvidence(
            evidence_type=("eligibility"),
            description=(
                "Action eligibility was "
                "evaluated against policy, "
                "risk, evidence confidence, "
                "root-cause context, targets "
                "and reversibility."
            ),
            score=score,
            incident_id=(context.incident_id),
            resource_ids=list(action.target_resource_ids),
            attributes={
                "policy": (policy.to_dict()),
                "conditions": (conditions),
                "blockers": blockers,
                "satisfied": (satisfied),
            },
        )
    ]

    return EligibilityAssessment(
        status=status,
        score=score,
        eligible=eligible,
        conditions=conditions,
        blocking_reasons=(blockers),
        satisfied_requirements=(satisfied),
        evidence=evidence,
        attributes={
            "policy_evaluation": (policy.to_dict()),
            "mutating_action": (action_is_mutating(action)),
            "read_only_action": (action_is_read_only(action)),
        },
    ).finalize()


# ---------------------------------------------------------------
# Attach eligibility to ActionCandidate
# ---------------------------------------------------------------


def apply_eligibility(
    action: ActionCandidate,
    context: DecisionContext,
    *,
    policy_engine: Optional[PolicyEngine] = None,
    config: Optional[EligibilityConfig] = None,
) -> ActionCandidate:
    """
    Evaluate and attach eligibility metadata.
    """

    policy_engine = policy_engine or PolicyEngine()

    # Apply policy first so the ActionCandidate's
    # approval mode/policy IDs are synchronized.
    action = policy_engine.apply(
        action,
        context,
    )

    assessment = assess_eligibility(
        action,
        context,
        policy_engine=(policy_engine),
        config=config,
    )

    action.eligibility = assessment

    action.evidence.extend(assessment.evidence)

    if not assessment.eligible:
        action.state = ActionState.INELIGIBLE.value

    elif action.approval.required:
        action.state = ActionState.REQUIRES_APPROVAL.value

    else:
        action.state = ActionState.ELIGIBLE.value

    return action.finalize()


def evaluate_actions(
    actions: Iterable[ActionCandidate],
    context: DecisionContext,
    *,
    policy_engine: Optional[PolicyEngine] = None,
    config: Optional[EligibilityConfig] = None,
) -> List[ActionCandidate]:
    engine = policy_engine or PolicyEngine()

    return [
        apply_eligibility(
            action,
            context,
            policy_engine=engine,
            config=config,
        )
        for action in actions
    ]


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------


def eligibility_summary(
    actions: Iterable[ActionCandidate],
) -> Dict[str, Any]:
    action_list = list(actions)

    statuses: Dict[
        str,
        int,
    ] = {}

    blockers: Dict[
        str,
        int,
    ] = {}

    for action in action_list:
        statuses[action.eligibility.status] = (
            statuses.get(
                action.eligibility.status,
                0,
            )
            + 1
        )

        for blocker in action.eligibility.blocking_reasons:
            blockers[blocker] = (
                blockers.get(
                    blocker,
                    0,
                )
                + 1
            )

    return {
        "action_count": (len(action_list)),
        "eligible_count": (
            sum(1 for action in action_list if action.eligibility.eligible)
        ),
        "ineligible_count": (
            sum(1 for action in action_list if not action.eligibility.eligible)
        ),
        "statuses": statuses,
        "blocking_reasons": (blockers),
    }


class EligibilityEngine:
    """
    Reusable eligibility evaluator.

    This component cannot approve or execute
    infrastructure mutations.
    """

    def __init__(
        self,
        *,
        policy_engine: Optional[PolicyEngine] = None,
        config: Optional[EligibilityConfig] = None,
    ) -> None:
        self.policy_engine = policy_engine or PolicyEngine()

        self.config = config or EligibilityConfig()

        self.config.validate()

    def assess(
        self,
        action: ActionCandidate,
        context: DecisionContext,
    ) -> EligibilityAssessment:
        return assess_eligibility(
            action,
            context,
            policy_engine=(self.policy_engine),
            config=self.config,
        )

    def apply(
        self,
        action: ActionCandidate,
        context: DecisionContext,
    ) -> ActionCandidate:
        return apply_eligibility(
            action,
            context,
            policy_engine=(self.policy_engine),
            config=self.config,
        )

    def apply_many(
        self,
        actions: Iterable[ActionCandidate],
        context: DecisionContext,
    ) -> List[ActionCandidate]:
        return evaluate_actions(
            actions,
            context,
            policy_engine=(self.policy_engine),
            config=self.config,
        )
