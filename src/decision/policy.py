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
    Mapping,
    Optional,
    Set,
)

from src.decision.schema import (
    ActionCandidate,
    ActionCategory,
    ApprovalMode,
    DecisionContext,
    DecisionEvidence,
    Reversibility,
)


# ---------------------------------------------------------------
# Policy actions
# ---------------------------------------------------------------


class PolicyEffect:
    """
    Policy evaluation effects.

    Kept as strings rather than a separate Enum
    so policy configuration can later be loaded
    more easily from YAML/JSON.
    """

    ALLOW = "allow"

    ALLOW_WITH_APPROVAL = "allow_with_approval"

    DENY = "deny"


# ---------------------------------------------------------------
# Policy rule
# ---------------------------------------------------------------


@dataclass(frozen=True)
class PolicyRule:
    """
    One declarative decision-policy rule.

    A rule defines what OSHA may *consider*.

    It never executes the action.
    """

    policy_id: str

    name: str

    description: str

    effect: str

    categories: tuple[
        str,
        ...,
    ] = ()

    action_types: tuple[
        str,
        ...,
    ] = ()

    environments: tuple[
        str,
        ...,
    ] = ()

    minimum_confidence: float = 0.0

    require_verified_root_cause: bool = False

    require_known_user_impact: bool = False

    require_reversible: bool = False

    maximum_target_resources: Optional[int] = None

    required_approval_mode: Optional[str] = None

    priority: int = 100

    enabled: bool = True

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        if self.effect not in {
            PolicyEffect.ALLOW,
            PolicyEffect.ALLOW_WITH_APPROVAL,
            PolicyEffect.DENY,
        }:
            raise ValueError(f"Unsupported policy effect: {self.effect}")

        if not (0.0 <= self.minimum_confidence <= 1.0):
            raise ValueError("minimum_confidence must be between 0 and 1")

        if (
            self.maximum_target_resources is not None
            and self.maximum_target_resources < 1
        ):
            raise ValueError("maximum_target_resources must be >= 1")


# ---------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------


@dataclass
class PolicyEvaluation:
    """
    Result of evaluating one action against the
    configured policy rules.
    """

    allowed: bool

    effect: str

    matched_policy_ids: List[str] = field(default_factory=list)

    approval_mode: str = ApprovalMode.HUMAN.value

    reasons: List[str] = field(default_factory=list)

    blocking_reasons: List[str] = field(default_factory=list)

    evidence: List[DecisionEvidence] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "PolicyEvaluation":
        self.matched_policy_ids = sorted(set(self.matched_policy_ids))

        self.reasons = list(dict.fromkeys(self.reasons))

        self.blocking_reasons = list(dict.fromkeys(self.blocking_reasons))

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "allowed": self.allowed,
            "effect": self.effect,
            "matched_policy_ids": (self.matched_policy_ids),
            "approval_mode": (self.approval_mode),
            "reasons": (self.reasons),
            "blocking_reasons": (self.blocking_reasons),
            "evidence": [item.to_dict() for item in self.evidence],
            "attributes": dict(self.attributes),
        }


# ---------------------------------------------------------------
# Default policy
# ---------------------------------------------------------------


def default_policy_rules() -> List[PolicyRule]:
    """
    Conservative OSHA development defaults.

    These are project policy choices, not
    external regulatory requirements.

    Unknown or unmatched actions are denied by
    PolicyEngine's default-deny behavior.
    """

    return [
        PolicyRule(
            policy_id=("osha.observe.default"),
            name=("Allow observation"),
            description=("Permit read-only observation and evidence collection."),
            effect=(PolicyEffect.ALLOW),
            categories=(ActionCategory.OBSERVATION.value,),
            required_approval_mode=(ApprovalMode.NONE.value),
            priority=10,
        ),
        PolicyRule(
            policy_id=("osha.investigate.default"),
            name=("Allow investigation"),
            description=("Permit non-mutating investigation steps."),
            effect=(PolicyEffect.ALLOW),
            categories=(ActionCategory.INVESTIGATION.value,),
            required_approval_mode=(ApprovalMode.NONE.value),
            priority=10,
        ),
        PolicyRule(
            policy_id=("osha.notification.default"),
            name=("Allow notification"),
            description=(
                "Permit creation of operator-facing recommendations or notifications."
            ),
            effect=(PolicyEffect.ALLOW),
            categories=(ActionCategory.NOTIFICATION.value,),
            required_approval_mode=(ApprovalMode.NONE.value),
            priority=10,
        ),
        PolicyRule(
            policy_id=("osha.configuration.human-approval"),
            name=("Configuration changes require human approval"),
            description=(
                "Configuration mutations may "
                "be considered but require "
                "explicit human approval."
            ),
            effect=(PolicyEffect.ALLOW_WITH_APPROVAL),
            categories=(ActionCategory.CONFIGURATION.value,),
            minimum_confidence=0.70,
            require_reversible=True,
            required_approval_mode=(ApprovalMode.HUMAN.value),
            priority=20,
        ),
        PolicyRule(
            policy_id=("osha.service-control.human-approval"),
            name=("Service control requires human approval"),
            description=("Service mutations require explicit human approval."),
            effect=(PolicyEffect.ALLOW_WITH_APPROVAL),
            categories=(ActionCategory.SERVICE_CONTROL.value,),
            minimum_confidence=0.75,
            require_reversible=True,
            required_approval_mode=(ApprovalMode.HUMAN.value),
            priority=20,
        ),
        PolicyRule(
            policy_id=("osha.resource-control.human-approval"),
            name=("Resource control requires human approval"),
            description=(
                "Host, VM, container or other "
                "resource mutations require "
                "human approval."
            ),
            effect=(PolicyEffect.ALLOW_WITH_APPROVAL),
            categories=(ActionCategory.RESOURCE_CONTROL.value,),
            minimum_confidence=0.80,
            require_verified_root_cause=True,
            require_reversible=True,
            required_approval_mode=(ApprovalMode.HUMAN.value),
            priority=20,
        ),
        PolicyRule(
            policy_id=("osha.network-control.human-approval"),
            name=("Network control requires human approval"),
            description=(
                "Network mutations require a "
                "verified root-cause context "
                "and explicit human approval."
            ),
            effect=(PolicyEffect.ALLOW_WITH_APPROVAL),
            categories=(ActionCategory.NETWORK_CONTROL.value,),
            minimum_confidence=0.80,
            require_verified_root_cause=True,
            require_reversible=True,
            required_approval_mode=(ApprovalMode.HUMAN.value),
            priority=20,
        ),
        PolicyRule(
            policy_id=("osha.security-control.human-approval"),
            name=("Security control requires human approval"),
            description=(
                "Security-affecting changes require explicit operator approval."
            ),
            effect=(PolicyEffect.ALLOW_WITH_APPROVAL),
            categories=(ActionCategory.SECURITY_CONTROL.value,),
            minimum_confidence=0.80,
            required_approval_mode=(ApprovalMode.HUMAN.value),
            priority=20,
        ),
        PolicyRule(
            policy_id=("osha.capacity.human-approval"),
            name=("Capacity changes require human approval"),
            description=("Capacity-affecting actions require operator approval."),
            effect=(PolicyEffect.ALLOW_WITH_APPROVAL),
            categories=(ActionCategory.CAPACITY.value,),
            minimum_confidence=0.75,
            require_reversible=True,
            required_approval_mode=(ApprovalMode.HUMAN.value),
            priority=20,
        ),
        PolicyRule(
            policy_id=("osha.recovery.human-approval"),
            name=("Recovery operations require human approval"),
            description=(
                "Recovery actions may be planned but require explicit human approval."
            ),
            effect=(PolicyEffect.ALLOW_WITH_APPROVAL),
            categories=(ActionCategory.RECOVERY.value,),
            minimum_confidence=0.75,
            require_reversible=True,
            required_approval_mode=(ApprovalMode.HUMAN.value),
            priority=20,
        ),
        PolicyRule(
            policy_id=("osha.other.deny"),
            name=("Deny unknown actions"),
            description=("Unknown or uncategorized actions are denied."),
            effect=(PolicyEffect.DENY),
            categories=(ActionCategory.OTHER.value,),
            required_approval_mode=(ApprovalMode.FORBIDDEN.value),
            priority=1,
        ),
    ]


# ---------------------------------------------------------------
# Matching
# ---------------------------------------------------------------


def rule_matches_action(
    rule: PolicyRule,
    action: ActionCandidate,
    context: DecisionContext,
) -> bool:
    """
    Determine whether a rule applies to an
    action/context pair.

    Requirements such as minimum confidence are
    evaluated separately after matching.
    """

    if not rule.enabled:
        return False

    if rule.categories and action.category not in rule.categories:
        return False

    if rule.action_types and action.action_type not in rule.action_types:
        return False

    if rule.environments and context.environment_id not in rule.environments:
        return False

    return True


def requirement_failures(
    rule: PolicyRule,
    action: ActionCandidate,
    context: DecisionContext,
) -> List[str]:
    """
    Evaluate policy preconditions.

    These failures do not execute or mutate
    anything.
    """

    failures = []

    if context.incident_confidence < rule.minimum_confidence:
        failures.append(
            "Incident confidence "
            f"{context.incident_confidence:.2f} "
            "is below policy minimum "
            f"{rule.minimum_confidence:.2f}."
        )

    if rule.require_verified_root_cause and not context.root_cause_verified:
        failures.append("Policy requires a verified root-cause conclusion.")

    if rule.require_known_user_impact and not context.user_impact_known:
        failures.append("Policy requires independently known user impact.")

    if rule.require_reversible:
        if action.reversibility not in {
            Reversibility.FULL.value,
        }:
            failures.append("Policy requires the action to be fully reversible.")

    if (
        rule.maximum_target_resources is not None
        and len(action.target_resource_ids) > rule.maximum_target_resources
    ):
        failures.append("Action affects more resources than this policy allows.")

    return failures


# ---------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------


def evaluate_action_policy(
    action: ActionCandidate,
    context: DecisionContext,
    *,
    rules: Optional[Iterable[PolicyRule]] = None,
) -> PolicyEvaluation:
    """
    Evaluate one proposed action.

    Policy behavior is deny-by-default:

        no matching allow rule
            -> denied

        any matching deny rule
            -> denied

        matching allow rule with failed
        requirements
            -> denied

        matching allow_with_approval
            -> allowed only as an approval-bound
               candidate
    """

    action.finalize()

    context.finalize()

    configured_rules = list(rules if rules is not None else default_policy_rules())

    for rule in configured_rules:
        rule.validate()

    matching = [
        rule
        for rule in configured_rules
        if rule_matches_action(
            rule,
            action,
            context,
        )
    ]

    matching.sort(
        key=lambda rule: (
            rule.priority,
            rule.policy_id,
        )
    )

    if not matching:
        return PolicyEvaluation(
            allowed=False,
            effect=PolicyEffect.DENY,
            approval_mode=(ApprovalMode.FORBIDDEN.value),
            blocking_reasons=[("No policy rule explicitly allows this action.")],
            evidence=[
                DecisionEvidence(
                    evidence_type=("policy"),
                    description=(
                        "Default-deny policy applied because no allowing rule matched."
                    ),
                    score=1.0,
                    incident_id=(context.incident_id),
                    resource_ids=list(action.target_resource_ids),
                )
            ],
        ).finalize()

    deny_rules = [rule for rule in matching if (rule.effect == PolicyEffect.DENY)]

    if deny_rules:
        return PolicyEvaluation(
            allowed=False,
            effect=PolicyEffect.DENY,
            matched_policy_ids=[rule.policy_id for rule in deny_rules],
            approval_mode=(ApprovalMode.FORBIDDEN.value),
            blocking_reasons=[
                (f"Action is explicitly denied by policy {rule.policy_id}.")
                for rule in deny_rules
            ],
            evidence=[
                DecisionEvidence(
                    evidence_type=("policy"),
                    description=("An explicit deny policy matched the action."),
                    score=1.0,
                    incident_id=(context.incident_id),
                    resource_ids=list(action.target_resource_ids),
                    attributes={
                        "policy_ids": [rule.policy_id for rule in deny_rules],
                    },
                )
            ],
        ).finalize()

    allow_rules = [
        rule
        for rule in matching
        if rule.effect
        in {
            PolicyEffect.ALLOW,
            PolicyEffect.ALLOW_WITH_APPROVAL,
        }
    ]

    if not allow_rules:
        return PolicyEvaluation(
            allowed=False,
            effect=PolicyEffect.DENY,
            approval_mode=(ApprovalMode.FORBIDDEN.value),
            blocking_reasons=["No allow rule matched."],
        ).finalize()

    failures_by_rule: Dict[
        str,
        List[str],
    ] = {}

    passing_rules = []

    for rule in allow_rules:
        failures = requirement_failures(
            rule,
            action,
            context,
        )

        if failures:
            failures_by_rule[rule.policy_id] = failures

        else:
            passing_rules.append(rule)

    if not passing_rules:
        blocking = []

        for (
            policy_id,
            failures,
        ) in failures_by_rule.items():
            blocking.extend((f"{policy_id}: {failure}") for failure in failures)

        return PolicyEvaluation(
            allowed=False,
            effect=PolicyEffect.DENY,
            matched_policy_ids=[rule.policy_id for rule in allow_rules],
            approval_mode=(ApprovalMode.FORBIDDEN.value),
            blocking_reasons=(blocking),
            evidence=[
                DecisionEvidence(
                    evidence_type=("policy_requirements"),
                    description=(
                        "Policy matched but "
                        "required evidence or "
                        "safety conditions were "
                        "not satisfied."
                    ),
                    score=1.0,
                    incident_id=(context.incident_id),
                    resource_ids=list(action.target_resource_ids),
                    attributes={
                        "failures": (failures_by_rule),
                    },
                )
            ],
        ).finalize()

    # Prefer the strictest passing approval
    # requirement.
    #
    # No approval < automatic < human <
    # multi-party < forbidden.
    approval_rank = {
        ApprovalMode.NONE.value: 0,
        ApprovalMode.AUTOMATIC.value: 1,
        ApprovalMode.HUMAN.value: 2,
        ApprovalMode.MULTI_PARTY.value: 3,
        ApprovalMode.FORBIDDEN.value: 4,
    }

    selected_approval = ApprovalMode.NONE.value

    effect = PolicyEffect.ALLOW

    for rule in passing_rules:
        mode = rule.required_approval_mode or (
            ApprovalMode.HUMAN.value
            if rule.effect == PolicyEffect.ALLOW_WITH_APPROVAL
            else ApprovalMode.NONE.value
        )

        if approval_rank.get(
            mode,
            4,
        ) > approval_rank.get(
            selected_approval,
            0,
        ):
            selected_approval = mode

        if rule.effect == PolicyEffect.ALLOW_WITH_APPROVAL:
            effect = PolicyEffect.ALLOW_WITH_APPROVAL

    evaluation = PolicyEvaluation(
        allowed=True,
        effect=effect,
        matched_policy_ids=[rule.policy_id for rule in passing_rules],
        approval_mode=(selected_approval),
        reasons=[("Action is explicitly allowed by configured policy.")],
        evidence=[
            DecisionEvidence(
                evidence_type=("policy"),
                description=("Action matched one or more allowing policy rules."),
                score=1.0,
                incident_id=(context.incident_id),
                resource_ids=list(action.target_resource_ids),
                attributes={
                    "policy_ids": [rule.policy_id for rule in passing_rules],
                    "approval_mode": (selected_approval),
                },
            )
        ],
        attributes={
            "failed_matching_rules": (failures_by_rule),
        },
    )

    return evaluation.finalize()


# ---------------------------------------------------------------
# Apply policy metadata to ActionCandidate
# ---------------------------------------------------------------


def apply_policy(
    action: ActionCandidate,
    context: DecisionContext,
    *,
    rules: Optional[Iterable[PolicyRule]] = None,
) -> ActionCandidate:
    """
    Attach policy evaluation to an
    ActionCandidate.

    This remains declarative and does not perform
    the action.
    """

    evaluation = evaluate_action_policy(
        action,
        context,
        rules=rules,
    )

    action.policy_ids = evaluation.matched_policy_ids

    action.evidence.extend(evaluation.evidence)

    action.attributes["policy_evaluation"] = evaluation.to_dict()

    if not evaluation.allowed:
        action.approval.required = True

        action.approval.mode = ApprovalMode.FORBIDDEN.value

        action.approval.reason = "; ".join(evaluation.blocking_reasons)

        return action.finalize()

    action.approval.mode = evaluation.approval_mode

    if evaluation.approval_mode == ApprovalMode.NONE.value:
        action.approval.required = False

    elif evaluation.approval_mode == ApprovalMode.AUTOMATIC.value:
        action.approval.required = False

    else:
        action.approval.required = True

    action.approval.reason = "; ".join(evaluation.reasons)

    return action.finalize()


# ---------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------


def evaluate_actions(
    actions: Iterable[ActionCandidate],
    context: DecisionContext,
    *,
    rules: Optional[Iterable[PolicyRule]] = None,
) -> List[ActionCandidate]:
    configured_rules = list(rules) if rules is not None else default_policy_rules()

    return [
        apply_policy(
            action,
            context,
            rules=(configured_rules),
        )
        for action in actions
    ]


# ---------------------------------------------------------------
# Policy configuration from mappings
# ---------------------------------------------------------------


def rule_from_mapping(
    value: Mapping[
        str,
        Any,
    ],
) -> PolicyRule:
    """
    Build one PolicyRule from configuration.

    This prepares OSHA for future YAML/JSON
    policy files.
    """

    rule = PolicyRule(
        policy_id=str(value["policy_id"]),
        name=str(
            value.get(
                "name",
                value["policy_id"],
            )
        ),
        description=str(
            value.get(
                "description",
                "",
            )
        ),
        effect=str(
            value.get(
                "effect",
                PolicyEffect.DENY,
            )
        ),
        categories=tuple(
            str(item)
            for item in value.get(
                "categories",
                (),
            )
        ),
        action_types=tuple(
            str(item)
            for item in value.get(
                "action_types",
                (),
            )
        ),
        environments=tuple(
            str(item)
            for item in value.get(
                "environments",
                (),
            )
        ),
        minimum_confidence=float(
            value.get(
                "minimum_confidence",
                0.0,
            )
        ),
        require_verified_root_cause=bool(
            value.get(
                "require_verified_root_cause",
                False,
            )
        ),
        require_known_user_impact=bool(
            value.get(
                "require_known_user_impact",
                False,
            )
        ),
        require_reversible=bool(
            value.get(
                "require_reversible",
                False,
            )
        ),
        maximum_target_resources=(
            int(value["maximum_target_resources"])
            if value.get("maximum_target_resources") is not None
            else None
        ),
        required_approval_mode=(
            str(value["required_approval_mode"])
            if value.get("required_approval_mode") is not None
            else None
        ),
        priority=int(
            value.get(
                "priority",
                100,
            )
        ),
        enabled=bool(
            value.get(
                "enabled",
                True,
            )
        ),
        attributes=dict(
            value.get(
                "attributes",
                {},
            )
        ),
    )

    rule.validate()

    return rule


def rules_from_mapping(
    values: Iterable[
        Mapping[
            str,
            Any,
        ]
    ],
) -> List[PolicyRule]:
    return [rule_from_mapping(value) for value in values]


# ---------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------


class PolicyEngine:
    """
    Reusable policy evaluator.

    This class has no execution capability.
    """

    def __init__(
        self,
        *,
        rules: Optional[Iterable[PolicyRule]] = None,
    ) -> None:
        self.rules = list(rules if rules is not None else default_policy_rules())

        for rule in self.rules:
            rule.validate()

    def evaluate(
        self,
        action: ActionCandidate,
        context: DecisionContext,
    ) -> PolicyEvaluation:
        return evaluate_action_policy(
            action,
            context,
            rules=self.rules,
        )

    def apply(
        self,
        action: ActionCandidate,
        context: DecisionContext,
    ) -> ActionCandidate:
        return apply_policy(
            action,
            context,
            rules=self.rules,
        )

    def apply_many(
        self,
        actions: Iterable[ActionCandidate],
        context: DecisionContext,
    ) -> List[ActionCandidate]:
        return evaluate_actions(
            actions,
            context,
            rules=self.rules,
        )

    def enabled_rules(
        self,
    ) -> List[PolicyRule]:
        return [rule for rule in self.rules if rule.enabled]
