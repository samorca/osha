"""
Context-aware authorization policy engine for OSHA.

This module sits above workload identity and coarse permission evaluation.

Security flow
-------------

    AuthenticatedWorkload
            |
            v
      PermissionSet
            |
            v
    AuthorizationContext
            |
            v
      AuthorizationPolicy
            |
            v
    PolicyDecisionResult
            |
       +----+----+
       |         |
      DENY      ALLOW
       |
       +---- REQUIRE_APPROVAL

Responsibilities
----------------
This module:

- requires authenticated workload identity
- requires coarse-grained permission
- evaluates contextual authorization rules
- supports explicit deny precedence
- supports human-approval requirements
- supports environment and resource constraints
- carries evidence/correlation identifiers
- records which policies produced a decision
- fails closed when no policy allows an operation

This module does NOT:

- execute remediation
- modify infrastructure
- authenticate workload certificates
- perform cryptographic signing
- create permissions
- determine AI model correctness
- bypass approval requirements

Higher-risk remediation-specific controls belong in remediation_policy.py.
"""

from __future__ import annotations

import re
import time
import uuid

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Iterable, Mapping

from ..identity.permissions import (
    AuthorizationResult,
    Permission,
    PermissionDeniedError,
    PermissionSet,
)
from ..identity.workload_identity import (
    AuthenticatedWorkload,
    DeploymentEnvironment,
    WorkloadIdentity,
    WorkloadType,
)


AUTHORIZATION_FORMAT_VERSION: Final[int] = 1

MAX_POLICY_ID_LENGTH: Final[int] = 128
MAX_REQUEST_ID_LENGTH: Final[int] = 128
MAX_REASON_LENGTH: Final[int] = 1024
MAX_EVIDENCE_ITEMS: Final[int] = 128
MAX_ATTRIBUTE_ITEMS: Final[int] = 128
MAX_ATTRIBUTE_LENGTH: Final[int] = 512

_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"
)

_RESOURCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9][a-z0-9._:/@*-]*$"
)


class AuthorizationPolicyError(Exception):
    """Base exception for OSHA authorization-policy failures."""


class InvalidAuthorizationContextError(
    AuthorizationPolicyError,
    ValueError,
):
    """Authorization request context is malformed."""


class InvalidAuthorizationPolicyError(
    AuthorizationPolicyError,
    ValueError,
):
    """Authorization policy is malformed."""


class AuthorizationDeniedError(
    AuthorizationPolicyError,
):
    """Authorization was denied."""


class ApprovalRequiredError(
    AuthorizationPolicyError,
):
    """Authorization requires approval before execution."""


class PolicyEffect(str, Enum):
    """
    Effect produced by a contextual authorization policy.
    """

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class PolicyDecision(str, Enum):
    """
    Final authorization decision.
    """

    ALLOWED = "allowed"
    DENIED = "denied"
    REQUIRE_APPROVAL = "require_approval"


class RiskLevel(str, Enum):
    """
    Coarse contextual risk classification.

    This is policy metadata, not an AI confidence score.
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


_RISK_ORDER: Final[dict[RiskLevel, int]] = {
    RiskLevel.LOW: 10,
    RiskLevel.MODERATE: 20,
    RiskLevel.HIGH: 30,
    RiskLevel.CRITICAL: 40,
}


@dataclass(frozen=True, slots=True)
class AuthorizationContext:
    """
    Context accompanying an authorization request.

    permission:
        Operation the workload wants to perform.

    risk_level:
        Security/operational risk classification supplied by an upstream
        trusted risk-classification step.

    correlation_id:
        Connects the decision to an incident, inference, workflow, or
        envelope.

    evidence_ids:
        References to evidence already retained elsewhere.

    attributes:
        Additional non-secret policy attributes.

    human_approved:
        Records whether a trusted upstream approval mechanism has supplied
        approval. This flag does NOT itself authenticate the approving human.
        Production approval evidence must later be cryptographically bound
        or otherwise verified by the approval subsystem.
    """

    permission: Permission
    risk_level: RiskLevel
    correlation_id: str

    evidence_ids: tuple[str, ...] = ()
    attributes: Mapping[str, str] | None = None

    human_approved: bool = False

    def __post_init__(self) -> None:
        if not isinstance(
            self.permission,
            Permission,
        ):
            raise InvalidAuthorizationContextError(
                "permission must be Permission."
            )

        if not isinstance(
            self.risk_level,
            RiskLevel,
        ):
            raise InvalidAuthorizationContextError(
                "risk_level must be RiskLevel."
            )

        _validate_identifier(
            self.correlation_id,
            field="correlation_id",
            max_length=MAX_REQUEST_ID_LENGTH,
        )

        if not isinstance(
            self.evidence_ids,
            tuple,
        ):
            raise InvalidAuthorizationContextError(
                "evidence_ids must be a tuple."
            )

        if len(self.evidence_ids) > MAX_EVIDENCE_ITEMS:
            raise InvalidAuthorizationContextError(
                f"evidence_ids cannot contain more than "
                f"{MAX_EVIDENCE_ITEMS} entries."
            )

        for evidence_id in self.evidence_ids:
            _validate_identifier(
                evidence_id,
                field="evidence_id",
                max_length=MAX_REQUEST_ID_LENGTH,
            )

        normalized_attributes = _normalize_attributes(
            self.attributes
        )

        object.__setattr__(
            self,
            "attributes",
            normalized_attributes,
        )

        if not isinstance(
            self.human_approved,
            bool,
        ):
            raise InvalidAuthorizationContextError(
                "human_approved must be boolean."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "attributes": dict(
                sorted(
                    (self.attributes or {}).items()
                )
            ),
            "correlation_id": self.correlation_id,
            "evidence_ids": list(
                self.evidence_ids
            ),
            "human_approved": self.human_approved,
            "permission": self.permission.as_dict(),
            "risk_level": self.risk_level.value,
        }


@dataclass(frozen=True, slots=True)
class AuthorizationPolicy:
    """
    One contextual OSHA authorization policy.

    Matching selectors are optional. None means any value.

    resource_pattern permits an exact resource or a terminal wildcard.

    Example:

        remediation/kubernetes/*

    max_risk_level limits the highest risk the rule applies to.
    """

    policy_id: str
    effect: PolicyEffect

    resource_pattern: str

    actions: frozenset

    trust_domain: str | None = None
    environment: DeploymentEnvironment | None = None
    workload_type: WorkloadType | None = None
    namespace: str | None = None
    workload_name: str | None = None

    max_risk_level: RiskLevel | None = None
    require_evidence: bool = False

    description: str = ""

    def __post_init__(self) -> None:
        _validate_identifier(
            self.policy_id,
            field="policy_id",
            max_length=MAX_POLICY_ID_LENGTH,
        )

        if not isinstance(
            self.effect,
            PolicyEffect,
        ):
            raise InvalidAuthorizationPolicyError(
                "effect must be PolicyEffect."
            )

        pattern = _normalize_resource_pattern(
            self.resource_pattern
        )

        object.__setattr__(
            self,
            "resource_pattern",
            pattern,
        )

        if not isinstance(
            self.actions,
            frozenset,
        ):
            raise InvalidAuthorizationPolicyError(
                "actions must be frozenset."
            )

        if not self.actions:
            raise InvalidAuthorizationPolicyError(
                "Authorization policy requires at least "
                "one action."
            )

        if (
            self.environment is not None
            and not isinstance(
                self.environment,
                DeploymentEnvironment,
            )
        ):
            raise InvalidAuthorizationPolicyError(
                "environment must be DeploymentEnvironment "
                "or None."
            )

        if (
            self.workload_type is not None
            and not isinstance(
                self.workload_type,
                WorkloadType,
            )
        ):
            raise InvalidAuthorizationPolicyError(
                "workload_type must be WorkloadType or None."
            )

        if (
            self.max_risk_level is not None
            and not isinstance(
                self.max_risk_level,
                RiskLevel,
            )
        ):
            raise InvalidAuthorizationPolicyError(
                "max_risk_level must be RiskLevel or None."
            )

        if not isinstance(
            self.require_evidence,
            bool,
        ):
            raise InvalidAuthorizationPolicyError(
                "require_evidence must be boolean."
            )

        for field, value in (
            ("trust_domain", self.trust_domain),
            ("namespace", self.namespace),
            ("workload_name", self.workload_name),
        ):
            if value is not None:
                _validate_selector(
                    value,
                    field=field,
                )

        if not isinstance(
            self.description,
            str,
        ):
            raise InvalidAuthorizationPolicyError(
                "description must be a string."
            )

        if len(self.description) > MAX_REASON_LENGTH:
            raise InvalidAuthorizationPolicyError(
                "description exceeds maximum length."
            )

    def matches(
        self,
        identity: WorkloadIdentity,
        context: AuthorizationContext,
    ) -> bool:
        """
        Determine whether this policy applies.

        Matching does not grant access. It only determines whether the
        policy participates in the final decision.
        """

        if not isinstance(
            identity,
            WorkloadIdentity,
        ):
            return False

        if not isinstance(
            context,
            AuthorizationContext,
        ):
            return False

        if (
            self.trust_domain is not None
            and identity.trust_domain
            != self.trust_domain
        ):
            return False

        if (
            self.environment is not None
            and identity.environment
            is not self.environment
        ):
            return False

        if (
            self.workload_type is not None
            and identity.workload_type
            is not self.workload_type
        ):
            return False

        if (
            self.namespace is not None
            and identity.namespace
            != self.namespace
        ):
            return False

        if (
            self.workload_name is not None
            and identity.name
            != self.workload_name
        ):
            return False

        if context.permission.action not in (
            self.actions
        ):
            return False

        if not _resource_matches(
            context.permission.resource,
            self.resource_pattern,
        ):
            return False

        if (
            self.max_risk_level is not None
            and _RISK_ORDER[
                context.risk_level
            ]
            > _RISK_ORDER[
                self.max_risk_level
            ]
        ):
            return False

        if (
            self.require_evidence
            and not context.evidence_ids
        ):
            return False

        return True


@dataclass(frozen=True, slots=True)
class PolicyDecisionResult:
    """
    Immutable authorization decision.

    This object is suitable for later signing and inclusion in an audit chain.
    """

    decision_id: str
    decision: PolicyDecision

    workload_id: str
    permission: Permission

    correlation_id: str
    risk_level: RiskLevel

    reason: str
    matched_policy_ids: tuple[str, ...]

    permission_result: AuthorizationResult

    evaluated_at_ns: int

    def __post_init__(self) -> None:
        _validate_identifier(
            self.decision_id,
            field="decision_id",
            max_length=MAX_REQUEST_ID_LENGTH,
        )

        if not isinstance(
            self.decision,
            PolicyDecision,
        ):
            raise InvalidAuthorizationPolicyError(
                "decision must be PolicyDecision."
            )

        if not isinstance(
            self.permission,
            Permission,
        ):
            raise InvalidAuthorizationPolicyError(
                "permission must be Permission."
            )

        if not isinstance(
            self.risk_level,
            RiskLevel,
        ):
            raise InvalidAuthorizationPolicyError(
                "risk_level must be RiskLevel."
            )

        if not isinstance(
            self.permission_result,
            AuthorizationResult,
        ):
            raise InvalidAuthorizationPolicyError(
                "permission_result must be AuthorizationResult."
            )

        if (
            not isinstance(self.reason, str)
            or not self.reason
        ):
            raise InvalidAuthorizationPolicyError(
                "reason must be non-empty."
            )

        if len(self.reason) > MAX_REASON_LENGTH:
            raise InvalidAuthorizationPolicyError(
                "reason exceeds maximum length."
            )

        if (
            isinstance(self.evaluated_at_ns, bool)
            or not isinstance(
                self.evaluated_at_ns,
                int,
            )
            or self.evaluated_at_ns <= 0
        ):
            raise InvalidAuthorizationPolicyError(
                "evaluated_at_ns must be positive."
            )

    @property
    def allowed(self) -> bool:
        return (
            self.decision
            is PolicyDecision.ALLOWED
        )

    @property
    def approval_required(self) -> bool:
        return (
            self.decision
            is PolicyDecision.REQUIRE_APPROVAL
        )

    def require_allowed(self) -> None:
        """
        Fail closed unless the final decision is ALLOWED.
        """

        if self.decision is PolicyDecision.REQUIRE_APPROVAL:
            raise ApprovalRequiredError(
                f"Authorization requires approval: "
                f"{self.reason}"
            )

        if not self.allowed:
            raise AuthorizationDeniedError(
                f"Authorization denied: "
                f"{self.reason}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "decision": self.decision.value,
            "decision_id": self.decision_id,
            "evaluated_at_ns": self.evaluated_at_ns,
            "format_version": (
                AUTHORIZATION_FORMAT_VERSION
            ),
            "matched_policy_ids": list(
                self.matched_policy_ids
            ),
            "permission": self.permission.as_dict(),
            "permission_result": (
                self.permission_result.as_dict()
            ),
            "reason": self.reason,
            "risk_level": self.risk_level.value,
            "workload_id": self.workload_id,
        }


class AuthorizationEngine:
    """
    OSHA contextual authorization policy engine.

    Evaluation order
    ----------------
    1. Require an authenticated workload.
    2. Evaluate coarse PermissionSet.
    3. Find contextual policies that match.
    4. Explicit DENY overrides everything.
    5. REQUIRE_APPROVAL is evaluated next.
    6. If approval has already been supplied, matching approval policies
       may become ALLOW.
    7. Explicit ALLOW permits the request.
    8. No applicable allow -> DENY.

    This is deliberately fail-closed.
    """

    def __init__(
        self,
        *,
        permissions: PermissionSet,
        policies: Iterable[
            AuthorizationPolicy
        ] = (),
    ) -> None:

        if not isinstance(
            permissions,
            PermissionSet,
        ):
            raise TypeError(
                "permissions must be PermissionSet."
            )

        resolved_policies = tuple(
            policies
        )

        seen_ids: set[str] = set()

        for policy in resolved_policies:
            if not isinstance(
                policy,
                AuthorizationPolicy,
            ):
                raise InvalidAuthorizationPolicyError(
                    "Every policy must be "
                    "AuthorizationPolicy."
                )

            if policy.policy_id in seen_ids:
                raise InvalidAuthorizationPolicyError(
                    f"Duplicate policy ID: "
                    f"{policy.policy_id!r}."
                )

            seen_ids.add(
                policy.policy_id
            )

        self._permissions = permissions
        self._policies = resolved_policies

    @property
    def policies(
        self,
    ) -> tuple[AuthorizationPolicy, ...]:
        return self._policies

    def evaluate(
        self,
        workload: AuthenticatedWorkload,
        context: AuthorizationContext,
    ) -> PolicyDecisionResult:
        """
        Evaluate a complete authorization request.
        """

        if not isinstance(
            workload,
            AuthenticatedWorkload,
        ):
            raise InvalidAuthorizationContextError(
                "workload must be AuthenticatedWorkload."
            )

        if not isinstance(
            context,
            AuthorizationContext,
        ):
            raise InvalidAuthorizationContextError(
                "context must be AuthorizationContext."
            )

        identity = workload.identity

        permission_result = (
            self._permissions.evaluate(
                workload,
                context.permission,
            )
        )

        if not workload.verified:
            return _decision(
                decision=PolicyDecision.DENIED,
                workload=workload,
                context=context,
                permission_result=permission_result,
                reason=(
                    "Workload identity has not been authenticated."
                ),
                matched_policy_ids=(),
            )

        if not permission_result.allowed:
            return _decision(
                decision=PolicyDecision.DENIED,
                workload=workload,
                context=context,
                permission_result=permission_result,
                reason=(
                    "Coarse workload permission denied."
                ),
                matched_policy_ids=(),
            )

        matching = tuple(
            policy
            for policy in self._policies
            if policy.matches(
                identity,
                context,
            )
        )

        deny_policies = tuple(
            policy
            for policy in matching
            if policy.effect
            is PolicyEffect.DENY
        )

        if deny_policies:
            return _decision(
                decision=PolicyDecision.DENIED,
                workload=workload,
                context=context,
                permission_result=permission_result,
                reason=(
                    "An explicit contextual deny policy "
                    "matched the request."
                ),
                matched_policy_ids=tuple(
                    policy.policy_id
                    for policy in deny_policies
                ),
            )

        approval_policies = tuple(
            policy
            for policy in matching
            if policy.effect
            is PolicyEffect.REQUIRE_APPROVAL
        )

        if (
            approval_policies
            and not context.human_approved
        ):
            return _decision(
                decision=(
                    PolicyDecision.REQUIRE_APPROVAL
                ),
                workload=workload,
                context=context,
                permission_result=permission_result,
                reason=(
                    "Contextual policy requires approval "
                    "before this operation may continue."
                ),
                matched_policy_ids=tuple(
                    policy.policy_id
                    for policy in approval_policies
                ),
            )

        allow_policies = tuple(
            policy
            for policy in matching
            if policy.effect
            is PolicyEffect.ALLOW
        )

        effective_allow_ids = list(
            policy.policy_id
            for policy in allow_policies
        )

        if (
            context.human_approved
            and approval_policies
        ):
            effective_allow_ids.extend(
                policy.policy_id
                for policy in approval_policies
            )

        if effective_allow_ids:
            return _decision(
                decision=PolicyDecision.ALLOWED,
                workload=workload,
                context=context,
                permission_result=permission_result,
                reason=(
                    "Request was explicitly allowed by "
                    "contextual authorization policy."
                ),
                matched_policy_ids=tuple(
                    effective_allow_ids
                ),
            )

        return _decision(
            decision=PolicyDecision.DENIED,
            workload=workload,
            context=context,
            permission_result=permission_result,
            reason=(
                "No contextual authorization policy "
                "explicitly allowed the request."
            ),
            matched_policy_ids=(),
        )

    def require(
        self,
        workload: AuthenticatedWorkload,
        context: AuthorizationContext,
    ) -> PolicyDecisionResult:
        """
        Evaluate and require final ALLOW.
        """

        result = self.evaluate(
            workload,
            context,
        )

        result.require_allowed()

        return result


def _decision(
    *,
    decision: PolicyDecision,
    workload: AuthenticatedWorkload,
    context: AuthorizationContext,
    permission_result: AuthorizationResult,
    reason: str,
    matched_policy_ids: tuple[str, ...],
) -> PolicyDecisionResult:

    return PolicyDecisionResult(
        decision_id=str(
            uuid.uuid4()
        ),
        decision=decision,
        workload_id=(
            workload.identity.workload_id
        ),
        permission=context.permission,
        correlation_id=context.correlation_id,
        risk_level=context.risk_level,
        reason=reason,
        matched_policy_ids=matched_policy_ids,
        permission_result=permission_result,
        evaluated_at_ns=time.time_ns(),
    )


def _resource_matches(
    resource: str,
    pattern: str,
) -> bool:

    if pattern == "*":
        return True

    if pattern.endswith("*"):
        return resource.startswith(
            pattern[:-1]
        )

    return resource == pattern


def _normalize_resource_pattern(
    value: str,
) -> str:

    if not isinstance(value, str):
        raise InvalidAuthorizationPolicyError(
            "resource_pattern must be a string."
        )

    normalized = value.strip().lower()

    if not normalized:
        raise InvalidAuthorizationPolicyError(
            "resource_pattern cannot be empty."
        )

    if (
        _RESOURCE_PATTERN.fullmatch(
            normalized
        )
        is None
    ):
        raise InvalidAuthorizationPolicyError(
            "resource_pattern contains "
            "invalid characters."
        )

    wildcard_count = normalized.count("*")

    if wildcard_count > 1:
        raise InvalidAuthorizationPolicyError(
            "resource_pattern supports at most "
            "one wildcard."
        )

    if (
        wildcard_count == 1
        and not normalized.endswith("*")
    ):
        raise InvalidAuthorizationPolicyError(
            "Wildcard is allowed only at the "
            "end of resource_pattern."
        )

    return normalized


def _normalize_attributes(
    attributes: Mapping[str, str] | None,
) -> Mapping[str, str]:

    if attributes is None:
        return {}

    if not isinstance(
        attributes,
        Mapping,
    ):
        raise InvalidAuthorizationContextError(
            "attributes must be a mapping or None."
        )

    if len(attributes) > MAX_ATTRIBUTE_ITEMS:
        raise InvalidAuthorizationContextError(
            f"attributes cannot exceed "
            f"{MAX_ATTRIBUTE_ITEMS} entries."
        )

    normalized: dict[str, str] = {}

    for key, value in attributes.items():
        if (
            not isinstance(key, str)
            or not key
        ):
            raise InvalidAuthorizationContextError(
                "Attribute names must be "
                "non-empty strings."
            )

        if not isinstance(value, str):
            raise InvalidAuthorizationContextError(
                "Attribute values must be strings."
            )

        normalized_key = key.strip().lower()
        normalized_value = value.strip()

        if not normalized_key:
            raise InvalidAuthorizationContextError(
                "Attribute name cannot be empty."
            )

        if (
            len(normalized_key)
            > MAX_ATTRIBUTE_LENGTH
            or len(normalized_value)
            > MAX_ATTRIBUTE_LENGTH
        ):
            raise InvalidAuthorizationContextError(
                "Authorization attribute exceeds "
                "maximum length."
            )

        normalized[
            normalized_key
        ] = normalized_value

    return normalized


def _validate_identifier(
    value: str,
    *,
    field: str,
    max_length: int,
) -> None:

    if (
        not isinstance(value, str)
        or not value
    ):
        raise ValueError(
            f"{field} must be a non-empty string."
        )

    if len(value) > max_length:
        raise ValueError(
            f"{field} exceeds maximum length "
            f"of {max_length}."
        )

    if (
        _ID_PATTERN.fullmatch(value)
        is None
    ):
        raise ValueError(
            f"{field} contains invalid characters."
        )


def _validate_selector(
    value: str,
    *,
    field: str,
) -> None:

    if (
        not isinstance(value, str)
        or not value
    ):
        raise InvalidAuthorizationPolicyError(
            f"{field} must be a non-empty string."
        )

    if value != value.strip().lower():
        raise InvalidAuthorizationPolicyError(
            f"{field} must be normalized lowercase."
        )


__all__ = [
    "AUTHORIZATION_FORMAT_VERSION",
    "ApprovalRequiredError",
    "AuthorizationContext",
    "AuthorizationDeniedError",
    "AuthorizationEngine",
    "AuthorizationPolicy",
    "AuthorizationPolicyError",
    "InvalidAuthorizationContextError",
    "InvalidAuthorizationPolicyError",
    "PolicyDecision",
    "PolicyDecisionResult",
    "PolicyEffect",
    "RiskLevel",
]
