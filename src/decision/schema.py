from __future__ import annotations

import hashlib
import json
import math
from dataclasses import (
    asdict,
    dataclass,
    field,
)
from datetime import (
    datetime,
    timezone,
)
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)


DECISION_SCHEMA_VERSION = "1"


# -----------------------------------------------------------------
# Enums
# -----------------------------------------------------------------


class DecisionType(
    str,
    Enum,
):
    """
    High-level decision produced by OSHA.

    Decision types describe what should happen
    next. They do not execute infrastructure
    changes.
    """

    OBSERVE = "observe"

    INVESTIGATE = "investigate"

    RECOMMEND = "recommend"

    REQUEST_APPROVAL = "request_approval"

    APPROVE_ACTION = "approve_action"

    REJECT_ACTION = "reject_action"

    DEFER = "defer"

    NO_ACTION = "no_action"


class DecisionStatus(
    str,
    Enum,
):
    """
    Lifecycle of a decision.
    """

    DRAFT = "draft"

    PROPOSED = "proposed"

    PENDING_APPROVAL = "pending_approval"

    APPROVED = "approved"

    REJECTED = "rejected"

    DEFERRED = "deferred"

    EXPIRED = "expired"

    COMPLETED = "completed"


class ActionCategory(
    str,
    Enum,
):
    """
    Broad category of proposed action.

    Actual implementation belongs to the future
    remediation layer.
    """

    OBSERVATION = "observation"

    INVESTIGATION = "investigation"

    NOTIFICATION = "notification"

    CONFIGURATION = "configuration"

    SERVICE_CONTROL = "service_control"

    RESOURCE_CONTROL = "resource_control"

    NETWORK_CONTROL = "network_control"

    SECURITY_CONTROL = "security_control"

    CAPACITY = "capacity"

    RECOVERY = "recovery"

    OTHER = "other"


class ActionState(
    str,
    Enum,
):
    """
    Planning state of an ActionCandidate.

    This is not remediation execution state.
    """

    PROPOSED = "proposed"

    ELIGIBLE = "eligible"

    INELIGIBLE = "ineligible"

    REQUIRES_APPROVAL = "requires_approval"

    APPROVED = "approved"

    REJECTED = "rejected"

    DEFERRED = "deferred"


class RiskLevel(
    str,
    Enum,
):
    """
    Operational risk classification.
    """

    NONE = "none"

    LOW = "low"

    MEDIUM = "medium"

    HIGH = "high"

    CRITICAL = "critical"

    UNKNOWN = "unknown"


class EligibilityStatus(
    str,
    Enum,
):
    """
    Whether OSHA may consider an action.
    """

    ELIGIBLE = "eligible"

    INELIGIBLE = "ineligible"

    CONDITIONAL = "conditional"

    UNKNOWN = "unknown"


class ApprovalMode(
    str,
    Enum,
):
    """
    Approval level required before execution.

    AUTOMATIC means policy permits future
    execution without per-action human approval.
    It does NOT itself execute anything.
    """

    NONE = "none"

    AUTOMATIC = "automatic"

    HUMAN = "human"

    MULTI_PARTY = "multi_party"

    FORBIDDEN = "forbidden"


class ApprovalStatus(
    str,
    Enum,
):
    """
    Current approval state.
    """

    NOT_REQUIRED = "not_required"

    PENDING = "pending"

    APPROVED = "approved"

    REJECTED = "rejected"

    EXPIRED = "expired"


class Reversibility(
    str,
    Enum,
):
    """
    How safely an action could be undone.
    """

    FULL = "full"

    PARTIAL = "partial"

    NONE = "none"

    UNKNOWN = "unknown"


# -----------------------------------------------------------------
# Common helpers
# -----------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def checked_score(
    value: Any,
) -> float:
    """
    Normalize heuristic evidence values into
    0.0..1.0.

    Decision scores remain policy/evidence
    indicators unless explicitly calibrated by a
    later model.
    """

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if not math.isfinite(number):
        return 0.0

    return min(
        1.0,
        max(
            0.0,
            number,
        ),
    )


def json_safe(
    value: Any,
    *,
    depth: int = 0,
    maximum_depth: int = 20,
) -> Any:
    """
    Convert decision data into JSON-safe
    structures.
    """

    if depth > maximum_depth:
        return "[MAX_DEPTH_REACHED]"

    if value is None:
        return None

    if isinstance(
        value,
        (
            bool,
            int,
            str,
        ),
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if math.isfinite(value):
            return value

        return None

    if isinstance(
        value,
        datetime,
    ):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.isoformat()

    if isinstance(
        value,
        bytes,
    ):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): json_safe(
                item,
                depth=depth + 1,
                maximum_depth=(maximum_depth),
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        return [
            json_safe(
                item,
                depth=depth + 1,
                maximum_depth=(maximum_depth),
            )
            for item in value
        ]

    if hasattr(
        value,
        "__dataclass_fields__",
    ):
        return json_safe(
            asdict(value),
            depth=depth + 1,
            maximum_depth=(maximum_depth),
        )

    return str(value)


def stable_hash(
    value: Mapping[
        str,
        Any,
    ],
) -> str:
    encoded = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


# -----------------------------------------------------------------
# Decision evidence
# -----------------------------------------------------------------


@dataclass(frozen=True)
class DecisionEvidence:
    """
    Evidence used by policy, risk, eligibility,
    approval or planning.

    Keeping evidence explicit lets operators
    understand why a decision was made.
    """

    evidence_type: str

    description: str

    score: float = 0.0

    source_id: Optional[str] = None

    incident_id: Optional[str] = None

    memory_id: Optional[str] = None

    resource_ids: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "evidence_type": (self.evidence_type),
            "description": (self.description),
            "score": checked_score(self.score),
            "source_id": (self.source_id),
            "incident_id": (self.incident_id),
            "memory_id": (self.memory_id),
            "resource_ids": (sorted(set(self.resource_ids))),
            "attributes": json_safe(self.attributes),
        }


# -----------------------------------------------------------------
# Decision context
# -----------------------------------------------------------------


@dataclass
class DecisionContext:
    """
    Bounded evidence supplied to the decision
    layer.

    Decision code should consume structured
    intelligence and memory rather than reaching
    directly into raw collector payloads.
    """

    environment_id: str

    incident_id: Optional[str] = None

    incident_severity: str = "unknown"

    incident_confidence: float = 0.0

    incident_confidence_level: str = "none"

    impact_level: str = "unknown"

    impact_score: float = 0.0

    user_impact_known: bool = False

    resource_ids: List[str] = field(default_factory=list)

    service_ids: List[str] = field(default_factory=list)

    root_cause_resource_id: Optional[str] = None

    root_cause_score: float = 0.0

    root_cause_status: Optional[str] = None

    root_cause_verified: bool = False

    similar_incident_ids: List[str] = field(default_factory=list)

    verified_similar_incident_ids: List[str] = field(default_factory=list)

    historical_outcome_ids: List[str] = field(default_factory=list)

    evidence: List[DecisionEvidence] = field(default_factory=list)

    limitations: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    created_at: str = field(default_factory=utc_now)

    context_id: Optional[str] = None

    def build_context_id(
        self,
    ) -> str:
        return "decision:context:" + stable_hash(
            {
                "environment_id": (self.environment_id),
                "incident_id": (self.incident_id),
                "resource_ids": (sorted(set(self.resource_ids))),
            }
        )

    def finalize(
        self,
    ) -> "DecisionContext":
        self.incident_confidence = checked_score(self.incident_confidence)

        self.impact_score = checked_score(self.impact_score)

        self.root_cause_score = checked_score(self.root_cause_score)

        self.resource_ids = sorted(set(self.resource_ids))

        self.service_ids = sorted(set(self.service_ids))

        self.similar_incident_ids = sorted(set(self.similar_incident_ids))

        self.verified_similar_incident_ids = sorted(
            set(self.verified_similar_incident_ids)
        )

        self.historical_outcome_ids = sorted(set(self.historical_outcome_ids))

        self.limitations = sorted(set(self.limitations))

        if not self.context_id:
            self.context_id = self.build_context_id()

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "context_id": (self.context_id),
            "environment_id": (self.environment_id),
            "incident_id": (self.incident_id),
            "incident_severity": (self.incident_severity),
            "incident_confidence": (self.incident_confidence),
            "incident_confidence_level": (self.incident_confidence_level),
            "impact_level": (self.impact_level),
            "impact_score": (self.impact_score),
            "user_impact_known": (self.user_impact_known),
            "resource_ids": (self.resource_ids),
            "service_ids": (self.service_ids),
            "root_cause_resource_id": (self.root_cause_resource_id),
            "root_cause_score": (self.root_cause_score),
            "root_cause_status": (self.root_cause_status),
            "root_cause_verified": (self.root_cause_verified),
            "similar_incident_ids": (self.similar_incident_ids),
            "verified_similar_incident_ids": (self.verified_similar_incident_ids),
            "historical_outcome_ids": (self.historical_outcome_ids),
            "evidence": [item.to_dict() for item in self.evidence],
            "limitations": (self.limitations),
            "attributes": json_safe(self.attributes),
            "created_at": (self.created_at),
        }


# -----------------------------------------------------------------
# Risk
# -----------------------------------------------------------------


@dataclass
class RiskAssessment:
    """
    Risk associated with performing one action.

    This represents action risk, not incident
    severity.
    """

    level: str = RiskLevel.UNKNOWN.value

    score: float = 0.0

    blast_radius_score: float = 0.0

    irreversibility_score: float = 0.0

    service_impact_score: float = 0.0

    uncertainty_score: float = 0.0

    privilege_score: float = 0.0

    reasons: List[str] = field(default_factory=list)

    evidence: List[DecisionEvidence] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "RiskAssessment":
        self.score = checked_score(self.score)

        self.blast_radius_score = checked_score(self.blast_radius_score)

        self.irreversibility_score = checked_score(self.irreversibility_score)

        self.service_impact_score = checked_score(self.service_impact_score)

        self.uncertainty_score = checked_score(self.uncertainty_score)

        self.privilege_score = checked_score(self.privilege_score)

        self.reasons = list(dict.fromkeys(self.reasons))

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "level": self.level,
            "score": self.score,
            "blast_radius_score": (self.blast_radius_score),
            "irreversibility_score": (self.irreversibility_score),
            "service_impact_score": (self.service_impact_score),
            "uncertainty_score": (self.uncertainty_score),
            "privilege_score": (self.privilege_score),
            "reasons": (self.reasons),
            "evidence": [item.to_dict() for item in self.evidence],
            "attributes": json_safe(self.attributes),
        }


# -----------------------------------------------------------------
# Eligibility
# -----------------------------------------------------------------


@dataclass
class EligibilityAssessment:
    """
    Whether policy permits OSHA to continue
    considering one proposed action.
    """

    status: str = EligibilityStatus.UNKNOWN.value

    score: float = 0.0

    eligible: bool = False

    conditions: List[str] = field(default_factory=list)

    blocking_reasons: List[str] = field(default_factory=list)

    satisfied_requirements: List[str] = field(default_factory=list)

    evidence: List[DecisionEvidence] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "EligibilityAssessment":
        self.score = checked_score(self.score)

        self.conditions = list(dict.fromkeys(self.conditions))

        self.blocking_reasons = list(dict.fromkeys(self.blocking_reasons))

        self.satisfied_requirements = list(dict.fromkeys(self.satisfied_requirements))

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "status": self.status,
            "score": self.score,
            "eligible": (self.eligible),
            "conditions": (self.conditions),
            "blocking_reasons": (self.blocking_reasons),
            "satisfied_requirements": (self.satisfied_requirements),
            "evidence": [item.to_dict() for item in self.evidence],
            "attributes": json_safe(self.attributes),
        }


# -----------------------------------------------------------------
# Approval
# -----------------------------------------------------------------


@dataclass
class ApprovalRequirement:
    """
    Approval controls attached to a proposed
    action.

    Approval is represented explicitly rather
    than inferred from risk alone.
    """

    mode: str = ApprovalMode.HUMAN.value

    status: str = ApprovalStatus.PENDING.value

    required: bool = True

    required_roles: List[str] = field(default_factory=list)

    minimum_approvals: int = 1

    approved_by: List[str] = field(default_factory=list)

    rejected_by: List[str] = field(default_factory=list)

    reason: Optional[str] = None

    requested_at: Optional[str] = None

    resolved_at: Optional[str] = None

    expires_at: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "ApprovalRequirement":
        self.minimum_approvals = max(
            0,
            int(self.minimum_approvals),
        )

        self.required_roles = sorted(set(self.required_roles))

        self.approved_by = sorted(set(self.approved_by))

        self.rejected_by = sorted(set(self.rejected_by))

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return json_safe(asdict(self))


# -----------------------------------------------------------------
# Action candidates
# -----------------------------------------------------------------


@dataclass
class ActionCandidate:
    """
    Structured proposal for a future remediation
    or investigation step.

    Creating an ActionCandidate DOES NOT execute
    the action.
    """

    action_type: str

    category: str

    title: str

    environment_id: str

    target_resource_ids: List[str] = field(default_factory=list)

    target_service_ids: List[str] = field(default_factory=list)

    description: Optional[str] = None

    rationale: Optional[str] = None

    parameters: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    reversibility: str = Reversibility.UNKNOWN.value

    rollback_action_type: Optional[str] = None

    expected_effect: Optional[str] = None

    state: str = ActionState.PROPOSED.value

    risk: RiskAssessment = field(default_factory=(RiskAssessment))

    eligibility: EligibilityAssessment = field(default_factory=(EligibilityAssessment))

    approval: ApprovalRequirement = field(default_factory=(ApprovalRequirement))

    evidence: List[DecisionEvidence] = field(default_factory=list)

    policy_ids: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    created_at: str = field(default_factory=utc_now)

    action_id: Optional[str] = None

    def build_action_id(
        self,
    ) -> str:
        return "decision:action:" + stable_hash(
            {
                "action_type": (self.action_type),
                "environment_id": (self.environment_id),
                "resources": (sorted(set(self.target_resource_ids))),
                "services": (sorted(set(self.target_service_ids))),
                "parameters": (self.parameters),
            }
        )

    def finalize(
        self,
    ) -> "ActionCandidate":
        self.target_resource_ids = sorted(set(self.target_resource_ids))

        self.target_service_ids = sorted(set(self.target_service_ids))

        self.policy_ids = sorted(set(self.policy_ids))

        self.risk.finalize()

        self.eligibility.finalize()

        self.approval.finalize()

        if not self.action_id:
            self.action_id = self.build_action_id()

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "action_id": (self.action_id),
            "action_type": (self.action_type),
            "category": (self.category),
            "title": self.title,
            "environment_id": (self.environment_id),
            "target_resource_ids": (self.target_resource_ids),
            "target_service_ids": (self.target_service_ids),
            "description": (self.description),
            "rationale": (self.rationale),
            "parameters": json_safe(self.parameters),
            "reversibility": (self.reversibility),
            "rollback_action_type": (self.rollback_action_type),
            "expected_effect": (self.expected_effect),
            "state": self.state,
            "risk": (self.risk.to_dict()),
            "eligibility": (self.eligibility.to_dict()),
            "approval": (self.approval.to_dict()),
            "evidence": [item.to_dict() for item in self.evidence],
            "policy_ids": (self.policy_ids),
            "attributes": json_safe(self.attributes),
            "created_at": (self.created_at),
        }


# -----------------------------------------------------------------
# Decision plan
# -----------------------------------------------------------------


@dataclass
class DecisionPlan:
    """
    Ordered collection of proposed actions.

    A plan is still declarative. The remediation
    layer will later determine how approved
    actions are executed.
    """

    environment_id: str

    incident_id: Optional[str] = None

    actions: List[ActionCandidate] = field(default_factory=list)

    summary: Optional[str] = None

    requires_approval: bool = False

    highest_risk_level: str = RiskLevel.UNKNOWN.value

    rollback_required: bool = False

    validation_required: bool = True

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    created_at: str = field(default_factory=utc_now)

    plan_id: Optional[str] = None

    def build_plan_id(
        self,
    ) -> str:
        return "decision:plan:" + stable_hash(
            {
                "environment_id": (self.environment_id),
                "incident_id": (self.incident_id),
                "action_ids": [
                    action.action_id or action.build_action_id()
                    for action in self.actions
                ],
            }
        )

    def finalize(
        self,
    ) -> "DecisionPlan":
        self.actions = [action.finalize() for action in self.actions]

        self.requires_approval = any(
            action.approval.required for action in self.actions
        )

        self.rollback_required = any(
            action.reversibility
            in {
                Reversibility.PARTIAL.value,
                Reversibility.NONE.value,
                Reversibility.UNKNOWN.value,
            }
            for action in self.actions
        )

        if not self.plan_id:
            self.plan_id = self.build_plan_id()

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "plan_id": (self.plan_id),
            "environment_id": (self.environment_id),
            "incident_id": (self.incident_id),
            "summary": self.summary,
            "requires_approval": (self.requires_approval),
            "highest_risk_level": (self.highest_risk_level),
            "rollback_required": (self.rollback_required),
            "validation_required": (self.validation_required),
            "actions": [action.to_dict() for action in self.actions],
            "attributes": json_safe(self.attributes),
            "created_at": (self.created_at),
        }


# -----------------------------------------------------------------
# Decision
# -----------------------------------------------------------------


@dataclass
class Decision:
    """
    Decision concerning one incident/context.

    This object says what OSHA concluded should
    happen next.

    It still does not execute anything.
    """

    decision_type: str

    environment_id: str

    status: str = DecisionStatus.DRAFT.value

    incident_id: Optional[str] = None

    context_id: Optional[str] = None

    plan_id: Optional[str] = None

    title: Optional[str] = None

    summary: Optional[str] = None

    confidence: float = 0.0

    risk_level: str = RiskLevel.UNKNOWN.value

    reason_codes: List[str] = field(default_factory=list)

    evidence: List[DecisionEvidence] = field(default_factory=list)

    limitations: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    created_at: str = field(default_factory=utc_now)

    decision_id: Optional[str] = None

    def build_decision_id(
        self,
    ) -> str:
        return "decision:" + stable_hash(
            {
                "decision_type": (self.decision_type),
                "environment_id": (self.environment_id),
                "incident_id": (self.incident_id),
                "context_id": (self.context_id),
                "plan_id": (self.plan_id),
            }
        )

    def finalize(
        self,
    ) -> "Decision":
        self.confidence = checked_score(self.confidence)

        self.reason_codes = sorted(set(self.reason_codes))

        self.limitations = sorted(set(self.limitations))

        if not self.decision_id:
            self.decision_id = self.build_decision_id()

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "decision_id": (self.decision_id),
            "decision_type": (self.decision_type),
            "environment_id": (self.environment_id),
            "status": self.status,
            "incident_id": (self.incident_id),
            "context_id": (self.context_id),
            "plan_id": (self.plan_id),
            "title": self.title,
            "summary": self.summary,
            "confidence": (self.confidence),
            "risk_level": (self.risk_level),
            "reason_codes": (self.reason_codes),
            "evidence": [item.to_dict() for item in self.evidence],
            "limitations": (self.limitations),
            "attributes": json_safe(self.attributes),
            "created_at": (self.created_at),
        }


# -----------------------------------------------------------------
# Complete decision result
# -----------------------------------------------------------------


@dataclass
class DecisionResult:
    """
    Complete output from one OSHA decision run.

    The result can safely be handed to a later
    remediation layer because it remains
    declarative.
    """

    decisions: List[Decision] = field(default_factory=list)

    plans: List[DecisionPlan] = field(default_factory=list)

    contexts: List[DecisionContext] = field(default_factory=list)

    unplanned_incident_ids: List[str] = field(default_factory=list)

    schema_version: str = DECISION_SCHEMA_VERSION

    generated_at: str = field(default_factory=utc_now)

    metadata: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "DecisionResult":
        self.contexts = [context.finalize() for context in self.contexts]

        self.plans = [plan.finalize() for plan in self.plans]

        self.decisions = [decision.finalize() for decision in self.decisions]

        self.unplanned_incident_ids = sorted(set(self.unplanned_incident_ids))

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (self.schema_version),
            "generated_at": (self.generated_at),
            "contexts": [context.to_dict() for context in self.contexts],
            "plans": [plan.to_dict() for plan in self.plans],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "unplanned_incident_ids": (self.unplanned_incident_ids),
            "metadata": json_safe(self.metadata),
        }

    def to_json(
        self,
    ) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
            default=str,
        )


# -----------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------


def risk_rank(
    level: str,
) -> int:
    ranks = {
        RiskLevel.NONE.value: 0,
        RiskLevel.LOW.value: 1,
        RiskLevel.MEDIUM.value: 2,
        RiskLevel.HIGH.value: 3,
        RiskLevel.CRITICAL.value: 4,
        RiskLevel.UNKNOWN.value: 5,
    }

    return ranks.get(
        level,
        5,
    )


def highest_risk(
    levels: List[str],
) -> str:
    if not levels:
        return RiskLevel.UNKNOWN.value

    return max(
        levels,
        key=risk_rank,
    )


def action_requires_approval(
    action: ActionCandidate,
) -> bool:
    action.finalize()

    return action.approval.required or action.approval.mode in {
        ApprovalMode.HUMAN.value,
        ApprovalMode.MULTI_PARTY.value,
    }


def action_is_executable_candidate(
    action: ActionCandidate,
) -> bool:
    """
    Return whether an action is structurally
    eligible for handoff to remediation.

    This does NOT execute the action.
    """

    action.finalize()

    if not action.eligibility.eligible:
        return False

    if action.approval.mode == ApprovalMode.FORBIDDEN.value:
        return False

    if (
        action.approval.required
        and action.approval.status != ApprovalStatus.APPROVED.value
    ):
        return False

    return True
