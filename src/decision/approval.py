from __future__ import annotations

from dataclasses import (
    dataclass,
)
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.decision.schema import (
    ActionCandidate,
    ActionCategory,
    ActionState,
    ApprovalMode,
    ApprovalRequirement,
    ApprovalStatus,
    DecisionContext,
    DecisionEvidence,
    EligibilityStatus,
    RiskLevel,
    utc_now,
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


@dataclass(frozen=True)
class ApprovalConfig:
    """
    OSHA approval configuration.

    These are project-policy defaults rather
    than externally mandated thresholds.

    At this stage, mutating actions remain
    approval-bound by default.
    """

    low_risk_mode: str = ApprovalMode.HUMAN.value

    medium_risk_mode: str = ApprovalMode.HUMAN.value

    high_risk_mode: str = ApprovalMode.HUMAN.value

    critical_risk_mode: str = ApprovalMode.MULTI_PARTY.value

    default_human_role: str = "infrastructure_operator"

    high_risk_role: str = "senior_infrastructure_operator"

    security_role: str = "security_operator"

    production_role: str = "production_approver"

    critical_minimum_approvals: int = 2

    approval_ttl_minutes: int = 30

    require_human_for_mutations: bool = True

    require_multi_party_for_critical: bool = True

    escalate_unverified_high_risk: bool = True

    def validate(
        self,
    ) -> None:
        valid_modes = {item.value for item in ApprovalMode}

        modes = (
            self.low_risk_mode,
            self.medium_risk_mode,
            self.high_risk_mode,
            self.critical_risk_mode,
        )

        if any(mode not in valid_modes for mode in modes):
            raise ValueError("Unsupported approval mode")

        if self.critical_minimum_approvals < 1:
            raise ValueError("critical_minimum_approvals must be >= 1")

        if self.approval_ttl_minutes < 1:
            raise ValueError("approval_ttl_minutes must be >= 1")


# ---------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------


def action_is_mutating(
    action: ActionCandidate,
) -> bool:
    return (
        action.category in MUTATING_CATEGORIES
        or action.attributes.get("mutating") is True
    )


def utc_datetime() -> datetime:
    return datetime.now(timezone.utc)


def approval_expiry(
    *,
    minutes: int,
) -> str:
    return (utc_datetime() + timedelta(minutes=minutes)).isoformat()


def risk_mode(
    action: ActionCandidate,
    *,
    config: ApprovalConfig,
) -> str:
    """
    Map an already-assessed action risk to the
    configured approval mode.
    """

    level = action.risk.level

    if level == RiskLevel.CRITICAL.value:
        return config.critical_risk_mode

    if level == RiskLevel.HIGH.value:
        return config.high_risk_mode

    if level == RiskLevel.MEDIUM.value:
        return config.medium_risk_mode

    return config.low_risk_mode


def stricter_mode(
    first: str,
    second: str,
) -> str:
    """
    Return the stricter of two approval modes.
    """

    ranks = {
        ApprovalMode.NONE.value: 0,
        ApprovalMode.AUTOMATIC.value: 1,
        ApprovalMode.HUMAN.value: 2,
        ApprovalMode.MULTI_PARTY.value: 3,
        ApprovalMode.FORBIDDEN.value: 4,
    }

    return max(
        (
            first,
            second,
        ),
        key=lambda mode: ranks.get(
            mode,
            4,
        ),
    )


# ---------------------------------------------------------------
# Roles
# ---------------------------------------------------------------


def approval_roles(
    action: ActionCandidate,
    context: DecisionContext,
    mode: str,
    *,
    config: ApprovalConfig,
) -> List[str]:
    """
    Determine abstract approver roles.

    These are role identifiers only. Identity and
    authorization resolution belong to a later
    integration boundary.
    """

    if mode in {
        ApprovalMode.NONE.value,
        ApprovalMode.AUTOMATIC.value,
        ApprovalMode.FORBIDDEN.value,
    }:
        return []

    roles = {config.default_human_role}

    if action.risk.level in {
        RiskLevel.HIGH.value,
        RiskLevel.CRITICAL.value,
    }:
        roles.add(config.high_risk_role)

    if action.category == ActionCategory.SECURITY_CONTROL.value:
        roles.add(config.security_role)

    environment = context.environment_id.strip().lower()

    if environment in {
        "prod",
        "production",
    }:
        roles.add(config.production_role)

    return sorted(roles)


# ---------------------------------------------------------------
# Approval requirement
# ---------------------------------------------------------------


def derive_approval_requirement(
    action: ActionCandidate,
    context: DecisionContext,
    *,
    config: Optional[ApprovalConfig] = None,
) -> ApprovalRequirement:
    """
    Derive approval requirements after policy,
    risk and eligibility have been evaluated.

    This function never approves an action.
    """

    configuration = config or ApprovalConfig()

    configuration.validate()

    action.finalize()

    context.finalize()

    # ---------------------------------------------------------
    # Hard stop: action is ineligible.
    # ---------------------------------------------------------

    if not (action.eligibility.eligible):
        return ApprovalRequirement(
            mode=(ApprovalMode.FORBIDDEN.value),
            status=(ApprovalStatus.REJECTED.value),
            required=True,
            minimum_approvals=0,
            reason=("Action is ineligible and cannot proceed to approval."),
            resolved_at=utc_now(),
            attributes={
                "eligibility_status": (action.eligibility.status),
                "blocking_reasons": (list(action.eligibility.blocking_reasons)),
            },
        ).finalize()

    # ---------------------------------------------------------
    # Policy may already forbid the action.
    # ---------------------------------------------------------

    if action.approval.mode == ApprovalMode.FORBIDDEN.value:
        return ApprovalRequirement(
            mode=(ApprovalMode.FORBIDDEN.value),
            status=(ApprovalStatus.REJECTED.value),
            required=True,
            minimum_approvals=0,
            reason=("Decision policy forbids this action."),
            resolved_at=utc_now(),
        ).finalize()

    mutating = action_is_mutating(action)

    existing_mode = action.approval.mode or ApprovalMode.HUMAN.value

    if mutating:
        calculated_mode = risk_mode(
            action,
            config=configuration,
        )

        mode = stricter_mode(
            existing_mode,
            calculated_mode,
        )

        if configuration.require_human_for_mutations:
            mode = stricter_mode(
                mode,
                ApprovalMode.HUMAN.value,
            )

    else:
        # Preserve policy requirements for
        # read-only actions. Otherwise no
        # approval is required.
        mode = (
            existing_mode
            if existing_mode
            not in {
                ApprovalMode.HUMAN.value,
                ApprovalMode.MULTI_PARTY.value,
            }
            else existing_mode
        )

        if not action.approval.required and existing_mode not in {
            ApprovalMode.HUMAN.value,
            ApprovalMode.MULTI_PARTY.value,
        }:
            mode = ApprovalMode.NONE.value

    # Critical mutating actions get the strongest
    # configured review.
    if (
        mutating
        and configuration.require_multi_party_for_critical
        and action.risk.level == RiskLevel.CRITICAL.value
    ):
        mode = stricter_mode(
            mode,
            ApprovalMode.MULTI_PARTY.value,
        )

    # Unverified high-risk root-cause context
    # escalates to multi-party review.
    if (
        mutating
        and configuration.escalate_unverified_high_risk
        and action.risk.level
        in {
            RiskLevel.HIGH.value,
            RiskLevel.CRITICAL.value,
        }
        and not context.root_cause_verified
    ):
        mode = stricter_mode(
            mode,
            ApprovalMode.MULTI_PARTY.value,
        )

    if mode == ApprovalMode.FORBIDDEN.value:
        return ApprovalRequirement(
            mode=mode,
            status=(ApprovalStatus.REJECTED.value),
            required=True,
            minimum_approvals=0,
            reason=("Approvalids this action."),
            resolved_at=utc_now(),
        ).finalize()

    if mode in {
        ApprovalMode.NONE.value,
        ApprovalMode.AUTOMATIC.value,
    }:
        return ApprovalRequirement(
            mode=mode,
            status=(ApprovalStatus.NOT_REQUIRED.value),
            required=False,
            minimum_approvals=0,
            reason=(
                "No per-action human "
                "approval is required by "
                "the current decision "
                "policy."
            ),
        ).finalize()

    roles = approval_roles(
        action,
        context,
        mode,
        config=configuration,
    )

    minimum_approvals = (
        configuration.critical_minimum_approvals
        if mode == ApprovalMode.MULTI_PARTY.value
        else 1
    )

    return ApprovalRequirement(
        mode=mode,
        status=(ApprovalStatus.PENDING.value),
        required=True,
        required_roles=roles,
        minimum_approvals=(minimum_approvals),
        requested_at=(utc_now()),
        expires_at=(approval_expiry(minutes=(configuration.approval_ttl_minutes))),
        reason=("Action requires explicit approval before remediation handoff."),
        attributes={
            "risk_level": (action.risk.level),
            "risk_score": (action.risk.score),
            "mutating": mutating,
            "root_cause_verified": (context.root_cause_verified),
        },
    ).finalize()


# ---------------------------------------------------------------
# Apply requirement
# ---------------------------------------------------------------


def apply_approval_requirement(
    action: ActionCandidate,
    context: DecisionContext,
    *,
    config: Optional[ApprovalConfig] = None,
) -> ActionCandidate:
    """
    Attach an approval requirement to the action.

    This does not grant approval.
    """

    action.approval = derive_approval_requirement(
        action,
        context,
        config=config,
    )

    action.evidence.append(
        DecisionEvidence(
            evidence_type=("approval_requirement"),
            description=(
                "Approval requirements were "
                "derived from action policy, "
                "eligibility, risk and "
                "operational context."
            ),
            score=(action.risk.score),
            incident_id=(context.incident_id),
            resource_ids=list(action.target_resource_ids),
            attributes={
                "mode": (action.approval.mode),
                "status": (action.approval.status),
                "required_roles": (action.approval.required_roles),
                "minimum_approvals": (action.approval.minimum_approvals),
            },
        )
    )

    if action.approval.mode == ApprovalMode.FORBIDDEN.value:
        action.state = ActionState.REJECTED.value

    elif action.approval.required:
        action.state = ActionState.REQUIRES_APPROVAL.value

    else:
        action.state = ActionState.ELIGIBLE.value

    return action.finalize()


# ---------------------------------------------------------------
# Explicit approval decisions
# ---------------------------------------------------------------


def approve_action(
    action: ActionCandidate,
    *,
    approver_id: str,
    approver_roles: Iterable[str] = (),
) -> ActionCandidate:
    """
    Record an explicit approval decision.

    This changes decision metadata only.

    It does NOT execute the action.
    """

    action.finalize()

    requirement = action.approval

    if requirement.mode == ApprovalMode.FORBIDDEN.value:
        raise ValueError("Forbidden action cannot be approved")

    if not requirement.required:
        raise ValueError("Action does not require explicit approval")

    if requirement.status == ApprovalStatus.EXPIRED.value:
        raise ValueError("Approval request has expired")

    if requirement.status == ApprovalStatus.REJECTED.value:
        raise ValueError("Rejected approval request cannot be approved")

    approver = approver_id.strip()

    if not approver:
        raise ValueError("approver_id is required")

    supplied_roles = {role.strip() for role in approver_roles if role.strip()}

    required_roles = set(requirement.required_roles)

    if required_roles and not required_roles.intersection(supplied_roles):
        raise ValueError("Approver does not hold an eligible approval role")

    requirement.approved_by = sorted(set(requirement.approved_by) | {approver})

    if len(requirement.approved_by) >= requirement.minimum_approvals:
        requirement.status = ApprovalStatus.APPROVED.value

        requirement.resolved_at = utc_now()

        action.state = ActionState.APPROVED.value

    else:
        requirement.status = ApprovalStatus.PENDING.value

        action.state = ActionState.REQUIRES_APPROVAL.value

    return action.finalize()


def reject_action(
    action: ActionCandidate,
    *,
    approver_id: str,
    reason: Optional[str] = None,
) -> ActionCandidate:
    """
    Explicitly reject a proposed action.

    No infrastructure mutation occurs.
    """

    action.finalize()

    approver = approver_id.strip()

    if not approver:
        raise ValueError("approver_id is required")

    action.approval.rejected_by = sorted(set(action.approval.rejected_by) | {approver})

    action.approval.status = ApprovalStatus.REJECTED.value

    action.approval.resolved_at = utc_now()

    if reason:
        action.approval.reason = reason

    action.state = ActionState.REJECTED.value

    return action.finalize()


# ---------------------------------------------------------------
# Expiration
# ---------------------------------------------------------------


def parse_timestamp(
    value: Optional[str],
) -> Optional[datetime]:
    if not value:
        return None

    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)

    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def approval_is_expired(
    requirement: ApprovalRequirement,
    *,
    now: Optional[datetime] = None,
) -> bool:
    expiry = parse_timestamp(requirement.expires_at)

    if expiry is None:
        return False

    current = now or utc_datetime()

    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)

    return current.astimezone(timezone.utc) >= expiry


def refresh_approval_state(
    action: ActionCandidate,
    *,
    now: Optional[datetime] = None,
) -> ActionCandidate:
    """
    Expire unresolved approval requests.

    Approval expiry requires re-evaluation rather
    than silently permitting stale authorization.
    """

    action.finalize()

    if action.approval.status != ApprovalStatus.PENDING.value:
        return action

    if approval_is_expired(
        action.approval,
        now=now,
    ):
        action.approval.status = ApprovalStatus.EXPIRED.value

        action.approval.resolved_at = utc_now()

        action.state = ActionState.DEFERRED.value

    return action.finalize()


# ---------------------------------------------------------------
# Batch handling
# ---------------------------------------------------------------


def assign_approvals(
    actions: Iterable[ActionCandidate],
    context: DecisionContext,
    *,
    config: Optional[ApprovalConfig] = None,
) -> List[ActionCandidate]:
    return [
        apply_approval_requirement(
            action,
            context,
            config=config,
        )
        for action in actions
    ]


def approval_summary(
    actions: Iterable[ActionCandidate],
) -> Dict[str, Any]:
    action_list = list(actions)

    modes: Dict[
        str,
        int,
    ] = {}

    statuses: Dict[
        str,
        int,
    ] = {}

    for action in action_list:
        modes[action.approval.mode] = (
            modes.get(
                action.approval.mode,
                0,
            )
            + 1
        )

        statuses[action.approval.status] = (
            statuses.get(
                action.approval.status,
                0,
            )
            + 1
        )

    return {
        "action_count": (len(action_list)),
        "approval_required_count": (
            sum(1 for action in action_list if action.approval.required)
        ),
        "approved_count": (
            sum(
                1
                for action in action_list
                if (action.approval.status == ApprovalStatus.APPROVED.value)
            )
        ),
        "pending_count": (
            sum(
                1
                for action in action_list
                if (action.approval.status == ApprovalStatus.PENDING.value)
            )
        ),
        "modes": modes,
        "statuses": statuses,
    }


class ApprovalEngine:
    """
    Reusable approval-state manager.

    This object manages authorization metadata.
    It cannot perform remediation.
    """

    def __init__(
        self,
        *,
        config: Optional[ApprovalConfig] = None,
    ) -> None:
        self.config = config or ApprovalConfig()

        self.config.validate()

    def assign(
        self,
        action: ActionCandidate,
        context: DecisionContext,
    ) -> ActionCandidate:
        return apply_approval_requirement(
            action,
            context,
            config=self.config,
        )

    def assign_many(
        self,
        actions: Iterable[ActionCandidate],
        context: DecisionContext,
    ) -> List[ActionCandidate]:
        return assign_approvals(
            actions,
            context,
            config=self.config,
        )

    def approve(
        self,
        action: ActionCandidate,
        *,
        approver_id: str,
        approver_roles: Iterable[str] = (),
    ) -> ActionCandidate:
        return approve_action(
            action,
            approver_id=(approver_id),
            approver_roles=(approver_roles),
        )

    def reject(
        self,
        action: ActionCandidate,
        *,
        approver_id: str,
        reason: Optional[str] = None,
    ) -> ActionCandidate:
        return reject_action(
            action,
            approver_id=(approver_id),
            reason=reason,
        )

    def refresh(
        self,
        action: ActionCandidate,
    ) -> ActionCandidate:
        return refresh_approval_state(action)
