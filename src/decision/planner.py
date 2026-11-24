from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.decision.actions import (
    ActionCatalog,
    deduplicate_actions,
)

from src.decision.approval import (
    ApprovalEngine,
)

from src.decision.eligibility import (
    EligibilityEngine,
)

from src.decision.policy import (
    PolicyEngine,
)

from src.decision.risk import (
    RiskEngine,
)

from src.decision.schema import (
    ActionCandidate,
    ActionCategory,
    ApprovalMode,
    ApprovalStatus,
    Decision,
    DecisionContext,
    DecisionPlan,
    DecisionStatus,
    DecisionType,
    RiskLevel,
    highest_risk,
)


CATEGORY_PRIORITY: Dict[
    str,
    int,
] = {
    ActionCategory.OBSERVATION.value: 10,
    ActionCategory.INVESTIGATION.value: 20,
    ActionCategory.NOTIFICATION.value: 30,
    ActionCategory.CONFIGURATION.value: 50,
    ActionCategory.SERVICE_CONTROL.value: 60,
    ActionCategory.CAPACITY.value: 70,
    ActionCategory.RECOVERY.value: 80,
    ActionCategory.SECURITY_CONTROL.value: 90,
    ActionCategory.NETWORK_CONTROL.value: 100,
    ActionCategory.RESOURCE_CONTROL.value: 110,
    ActionCategory.OTHER.value: 999,
}


RISK_PRIORITY: Dict[
    str,
    int,
] = {
    RiskLevel.NONE.value: 0,
    RiskLevel.LOW.value: 10,
    RiskLevel.MEDIUM.value: 20,
    RiskLevel.HIGH.value: 30,
    RiskLevel.CRITICAL.value: 40,
    RiskLevel.UNKNOWN.value: 50,
}


@dataclass(frozen=True)
class PlannerConfig:
    """
    OSHA decision-planning configuration.

    Plans remain proposals. No setting in this
    class grants infrastructure execution.
    """

    include_mutating_candidates: bool = True

    include_ineligible_actions: bool = True

    maximum_actions_per_plan: int = 10

    prioritize_read_only_actions: bool = True

    require_at_least_one_safe_action: bool = True

    def validate(
        self,
    ) -> None:
        if self.maximum_actions_per_plan < 1:
            raise ValueError("maximum_actions_per_plan must be >= 1")


# ---------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------


def action_sort_key(
    action: ActionCandidate,
) -> tuple[
    int,
    int,
    int,
    str,
]:
    """
    Prioritize:

      immediately eligible actions
      read-only/investigation actions
      lower-risk actions
      deterministic action ID

    Ineligible and forbidden actions remain
    visible but sort later.
    """

    approval_rank = 0

    if action.approval.mode == ApprovalMode.HUMAN.value:
        approval_rank = 10

    elif action.approval.mode == ApprovalMode.MULTI_PARTY.value:
        approval_rank = 20

    elif action.approval.mode == ApprovalMode.FORBIDDEN.value:
        approval_rank = 100

    if not (action.eligibility.eligible):
        approval_rank += 100

    return (
        approval_rank,
        CATEGORY_PRIORITY.get(
            action.category,
            999,
        ),
        RISK_PRIORITY.get(
            action.risk.level,
            50,
        ),
        (action.action_id or action.action_type),
    )


# ---------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------


def evaluate_candidates(
    actions: Iterable[ActionCandidate],
    context: DecisionContext,
    *,
    policy_engine: Optional[PolicyEngine] = None,
    risk_engine: Optional[RiskEngine] = None,
    eligibility_engine: Optional[EligibilityEngine] = None,
    approval_engine: Optional[ApprovalEngine] = None,
) -> List[ActionCandidate]:
    """
    Pass each action through all decision gates:

      risk
      policy/eligibility
      approval requirement

    No action is executed.
    """

    policy = policy_engine or PolicyEngine()

    risk = risk_engine or RiskEngine()

    eligibility = eligibility_engine or EligibilityEngine(policy_engine=policy)

    approval = approval_engine or ApprovalEngine()

    evaluated = []

    for candidate in deduplicate_actions(actions):
        candidate = risk.apply(
            candidate,
            context,
        )

        candidate = eligibility.apply(
            candidate,
            context,
        )

        candidate = approval.assign(
            candidate,
            context,
        )

        evaluated.append(candidate.finalize())

    evaluated.sort(key=action_sort_key)

    return evaluated


def select_plan_actions(
    actions: Iterable[ActionCandidate],
    *,
    config: PlannerConfig,
) -> List[ActionCandidate]:
    selected = []

    for action in actions:
        if not config.include_ineligible_actions and not action.eligibility.eligible:
            continue

        selected.append(action)

        if len(selected) >= config.maximum_actions_per_plan:
            break

    return selected


# ---------------------------------------------------------------
# Plan summary
# ---------------------------------------------------------------


def plan_summary(
    actions: Iterable[ActionCandidate],
) -> str:
    action_list = list(actions)

    eligible = sum(1 for item in action_list if item.eligibility.eligible)

    pending = sum(
        1
        for item in action_list
        if (item.approval.status == ApprovalStatus.PENDING.value)
    )

    blocked = sum(1 for item in action_list if not item.eligibility.eligible)

    return (
        f"Plan contains "
        f"{len(action_list)} "
        "action candidate(s): "
        f"{eligible} eligible, "
        f"{pending} awaiting approval, "
        f"and {blocked} blocked."
    )


# ---------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------


def build_plan(
    context: DecisionContext,
    *,
    actions: Optional[Iterable[ActionCandidate]] = None,
    action_catalog: Optional[ActionCatalog] = None,
    policy_engine: Optional[PolicyEngine] = None,
    risk_engine: Optional[RiskEngine] = None,
    eligibility_engine: Optional[EligibilityEngine] = None,
    approval_engine: Optional[ApprovalEngine] = None,
    config: Optional[PlannerConfig] = None,
) -> DecisionPlan:
    """
    Build one declarative decision plan.

    If actions are not provided, the planner asks
    the ActionCatalog for conservative defaults.
    """

    configuration = config or PlannerConfig()

    configuration.validate()

    context.finalize()

    catalog = action_catalog or ActionCatalog()

    if actions is not None:
        candidates = list(actions)

    else:
        candidates = catalog.defaults(
            context,
            include_mutating_candidates=(configuration.include_mutating_candidates),
        )

    evaluated = evaluate_candidates(
        candidates,
        context,
        policy_engine=(policy_engine),
        risk_engine=(risk_engine),
        eligibility_engine=(eligibility_engine),
        approval_engine=(approval_engine),
    )

    selected = select_plan_actions(
        evaluated,
        config=configuration,
    )

    safe_actions = [
        item
        for item in selected
        if (
            item.eligibility.eligible
            and not item.approval.required
            and item.approval.mode != ApprovalMode.FORBIDDEN.value
        )
    ]

    # If all generated actions are blocked or
    # approval-bound, retain a safe observation
    # candidate when policy permits it.
    if configuration.require_at_least_one_safe_action and not safe_actions:
        observation = catalog.create(
            "observe_resource",
            context,
            target_resource_ids=(context.resource_ids),
            target_service_ids=(context.service_ids),
            rationale=(
                "Fallback safe action "
                "because no other "
                "immediately eligible "
                "action was available."
            ),
        )

        observation = evaluate_candidates(
            [observation],
            context,
            policy_engine=(policy_engine),
            risk_engine=(risk_engine),
            eligibility_engine=(eligibility_engine),
            approval_engine=(approval_engine),
        )[0]

        selected = (
            deduplicate_actions(
                [
                    observation,
                    *selected,
                ]
            )
        )[: configuration.maximum_actions_per_plan]

        selected.sort(key=action_sort_key)

    plan = DecisionPlan(
        environment_id=(context.environment_id),
        incident_id=(context.incident_id),
        actions=selected,
        summary=plan_summary(selected),
        highest_risk_level=(highest_risk([item.risk.level for item in selected])),
        validation_required=True,
        attributes={
            "context_id": (context.context_id),
            "candidate_count": (len(candidates)),
            "evaluated_count": (len(evaluated)),
            "selected_count": (len(selected)),
            "eligible_action_ids": [
                item.action_id
                for item in selected
                if (item.eligibility.eligible and item.action_id)
            ],
            "blocked_action_ids": [
                item.action_id
                for item in selected
                if (not item.eligibility.eligible and item.action_id)
            ],
            "approval_pending_action_ids": [
                item.action_id
                for item in selected
                if (
                    item.approval.status == ApprovalStatus.PENDING.value
                    and item.action_id
                )
            ],
        },
    )

    return plan.finalize()


# ---------------------------------------------------------------
# Decision generation
# ---------------------------------------------------------------


def decision_type_for_plan(
    plan: DecisionPlan,
) -> str:
    eligible = [item for item in plan.actions if item.eligibility.eligible]

    if not eligible:
        return DecisionType.NO_ACTION.value

    if any(item.approval.required for item in eligible):
        return DecisionType.REQUEST_APPROVAL.value

    if any(item.category == ActionCategory.INVESTIGATION.value for item in eligible):
        return DecisionType.INVESTIGATE.value

    if any(item.category == ActionCategory.NOTIFICATION.value for item in eligible):
        return DecisionType.RECOMMEND.value

    return DecisionType.OBSERVE.value


def decision_status_for_plan(
    plan: DecisionPlan,
) -> str:
    eligible = [item for item in plan.actions if item.eligibility.eligible]

    if not eligible:
        return DecisionStatus.DEFERRED.value

    if any(item.approval.status == ApprovalStatus.PENDING.value for item in eligible):
        return DecisionStatus.PENDING_APPROVAL.value

    if all(
        item.approval.status
        in {
            ApprovalStatus.NOT_REQUIRED.value,
            ApprovalStatus.APPROVED.value,
        }
        for item in eligible
    ):
        return DecisionStatus.PROPOSED.value

    return DecisionStatus.DRAFT.value


def decision_from_plan(
    context: DecisionContext,
    plan: DecisionPlan,
) -> Decision:
    """
    Build one top-level Decision from a plan.
    """

    plan.finalize()

    eligible = [item for item in plan.actions if item.eligibility.eligible]

    blocked = [item for item in plan.actions if not item.eligibility.eligible]

    decision_type = decision_type_for_plan(plan)

    status = decision_status_for_plan(plan)

    reason_codes = []

    if eligible:
        reason_codes.append("eligible_actions_available")

    if any(item.approval.required for item in eligible):
        reason_codes.append("approval_required")

    if blocked:
        reason_codes.append("blocked_actions_present")

    if not eligible:
        reason_codes.append("no_eligible_actions")

    evidence = []

    for action in plan.actions:
        evidence.extend(action.evidence)

    return Decision(
        decision_type=decision_type,
        environment_id=(context.environment_id),
        status=status,
        incident_id=(context.incident_id),
        context_id=(context.context_id),
        plan_id=plan.plan_id,
        title=(
            "Approval required for proposed incident actions"
            if decision_type == DecisionType.REQUEST_APPROVAL.value
            else ("Incident decision plan prepared")
        ),
        summary=plan.summary,
        confidence=(context.incident_confidence),
        risk_level=(plan.highest_risk_level),
        reason_codes=(reason_codes),
        evidence=evidence,
        limitations=list(context.limitations),
        attributes={
            "eligible_action_count": (len(eligible)),
            "blocked_action_count": (len(blocked)),
            "requires_approval": (plan.requires_approval),
            "rollback_required": (plan.rollback_required),
        },
    ).finalize()


def plan_decision(
    context: DecisionContext,
    *,
    actions: Optional[Iterable[ActionCandidate]] = None,
    action_catalog: Optional[ActionCatalog] = None,
    policy_engine: Optional[PolicyEngine] = None,
    risk_engine: Optional[RiskEngine] = None,
    eligibility_engine: Optional[EligibilityEngine] = None,
    approval_engine: Optional[ApprovalEngine] = None,
    config: Optional[PlannerConfig] = None,
) -> tuple[
    DecisionPlan,
    Decision,
]:
    plan = build_plan(
        context,
        actions=actions,
        action_catalog=(action_catalog),
        policy_engine=(policy_engine),
        risk_engine=risk_engine,
        eligibility_engine=(eligibility_engine),
        approval_engine=(approval_engine),
        config=config,
    )

    return (
        plan,
        decision_from_plan(
            context,
            plan,
        ),
    )


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------


def planner_summary(
    plans: Iterable[DecisionPlan],
) -> Dict[str, Any]:
    plan_list = list(plans)

    actions = [action for plan in plan_list for action in plan.actions]

    return {
        "plan_count": (len(plan_list)),
        "action_count": (len(actions)),
        "eligible_action_count": (
            sum(1 for item in actions if item.eligibility.eligible)
        ),
        "blocked_action_count": (
            sum(1 for item in actions if not item.eligibility.eligible)
        ),
        "approval_required_count": (
            sum(1 for item in actions if item.approval.required)
        ),
        "plans_requiring_approval": (
            sum(1 for plan in plan_list if plan.requires_approval)
        ),
    }


class DecisionPlanner:
    """
    Reusable declarative decision planner.

    This class has no execution capability.
    """

    def __init__(
        self,
        *,
        action_catalog: Optional[ActionCatalog] = None,
        policy_engine: Optional[PolicyEngine] = None,
        risk_engine: Optional[RiskEngine] = None,
        eligibility_engine: Optional[EligibilityEngine] = None,
        approval_engine: Optional[ApprovalEngine] = None,
        config: Optional[PlannerConfig] = None,
    ) -> None:
        self.action_catalog = action_catalog or ActionCatalog()

        self.policy_engine = policy_engine or PolicyEngine()

        self.risk_engine = risk_engine or RiskEngine()

        self.eligibility_engine = eligibility_engine or EligibilityEngine(
            policy_engine=(self.policy_engine)
        )

        self.approval_engine = approval_engine or ApprovalEngine()

        self.config = config or PlannerConfig()

        self.config.validate()

    def plan(
        self,
        context: DecisionContext,
        *,
        actions: Optional[Iterable[ActionCandidate]] = None,
    ) -> tuple[
        DecisionPlan,
        Decision,
    ]:
        return plan_decision(
            context,
            actions=actions,
            action_catalog=(self.action_catalog),
            policy_engine=(self.policy_engine),
            risk_engine=(self.risk_engine),
            eligibility_engine=(self.eligibility_engine),
            approval_engine=(self.approval_engine),
            config=self.config,
        )
