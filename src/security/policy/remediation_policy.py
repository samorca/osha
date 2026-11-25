"""
Safety policy engine for OSHA infrastructure remediation.

This module is the security/policy boundary immediately before OSHA's
remediation or self-healing execution layer.

It DOES NOT execute infrastructure actions.

It determines whether a proposed remediation is:

    ALLOWED
    REQUIRE_APPROVAL
    DENIED

Evaluation considers:

- authenticated workload authorization
- contextual authorization result
- target platform
- action category
- operational risk
- AI confidence
- supporting evidence
- human approval state
- rollback availability
- change scope
- protected-resource status
- dry-run state
- policy-defined automation limits

Security principles
-------------------
1. Default deny.
2. AI confidence never grants authority by itself.
3. Contextual authorization must succeed first.
4. Explicit deny overrides all other outcomes.
5. Critical-risk autonomous remediation is denied by default.
6. High-risk remediation requires approval by default.
7. Low/moderate-risk automation requires policy, evidence, rollback,
   sufficient confidence, and bounded blast radius.
8. Protected resources require approval or explicit denial.
9. Execution is outside this module.
10. Decisions are structured for later signing and audit.

The values and thresholds in this module are OSHA policy defaults, not
universal security standards.
"""

from __future__ import annotations

import math
import re
import time
import uuid

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Iterable

from ..identity.permissions import (
    PermissionAction,
)
from ..identity.workload_identity import (
    AuthenticatedWorkload,
    DeploymentEnvironment,
)
from .authorisation import (
    AuthorizationContext,
    AuthorizationEngine,
    PolicyDecision,
    PolicyDecisionResult,
    RiskLevel,
)


REMEDIATION_POLICY_FORMAT_VERSION: Final[int] = 1

DEFAULT_MIN_AUTOMATION_CONFIDENCE: Final[float] = 0.90
DEFAULT_MAX_AUTOMATED_TARGETS: Final[int] = 1
MAX_TARGETS: Final[int] = 10_000

MAX_ID_LENGTH: Final[int] = 128
MAX_TARGET_LENGTH: Final[int] = 512
MAX_REASON_LENGTH: Final[int] = 2048
MAX_EVIDENCE_ITEMS: Final[int] = 128

_IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"
)


class RemediationPolicyError(Exception):
    """Base exception for OSHA remediation-policy failures."""


class InvalidRemediationRequestError(
    RemediationPolicyError,
    ValueError,
):
    """Proposed remediation is malformed."""


class RemediationDeniedError(
    RemediationPolicyError,
):
    """Remediation policy denied execution."""


class RemediationApprovalRequiredError(
    RemediationPolicyError,
):
    """Remediation requires approval before execution."""


class InfrastructurePlatform(str, Enum):
    """
    Infrastructure platforms understood by OSHA remediation policy.
    """

    LINUX = "linux"
    KUBERNETES = "kubernetes"
    OPENSTACK = "openstack"
    CEPH = "ceph"
    MINIO = "minio"
    KAFKA = "kafka"
    REDIS = "redis"
    TERRAFORM = "terraform"
    IBM_CLOUD = "ibm_cloud"
    GENERIC = "generic"


class RemediationAction(str, Enum):
    """
    Logical remediation actions.

    These are authorization categories, not shell commands or provider API
    operations.
    """

    RESTART_WORKLOAD = "restart_workload"
    RESTART_SERVICE = "restart_service"

    SCALE_WORKLOAD = "scale_workload"

    QUARANTINE_WORKLOAD = "quarantine_workload"
    QUARANTINE_NODE = "quarantine_node"

    ROLLBACK_DEPLOYMENT = "rollback_deployment"

    RELOAD_CONFIGURATION = "reload_configuration"

    ROTATE_CREDENTIAL = "rotate_credential"

    FAILOVER = "failover"

    DRAIN_NODE = "drain_node"

    REPAIR_STORAGE = "repair_storage"

    APPLY_NETWORK_POLICY = "apply_network_policy"

    CUSTOM = "custom"


class RemediationDecision(str, Enum):
    """Final remediation-policy decision."""

    ALLOWED = "allowed"
    REQUIRE_APPROVAL = "require_approval"
    DENIED = "denied"


@dataclass(frozen=True, slots=True)
class RemediationRequest:
    """
    Proposed infrastructure remediation.

    target is a logical resource identifier, not a shell command.

    confidence is the upstream model/pipeline confidence value represented
    from 0.0 to 1.0.

    Confidence is evidence, not authorization.

    protected_resource should be set by trusted inventory/policy integration,
    not by an untrusted requestor.
    """

    request_id: str
    correlation_id: str

    platform: InfrastructurePlatform
    action: RemediationAction

    target: str
    target_count: int

    risk_level: RiskLevel
    confidence: float

    evidence_ids: tuple[str, ...]

    rollback_available: bool

    protected_resource: bool = False
    dry_run: bool = False
    human_approved: bool = False

    def __post_init__(self) -> None:
        _validate_identifier(
            self.request_id,
            field="request_id",
        )

        _validate_identifier(
            self.correlation_id,
            field="correlation_id",
        )

        if not isinstance(
            self.platform,
            InfrastructurePlatform,
        ):
            raise InvalidRemediationRequestError(
                "platform must be InfrastructurePlatform."
            )

        if not isinstance(
            self.action,
            RemediationAction,
        ):
            raise InvalidRemediationRequestError(
                "action must be RemediationAction."
            )

        _validate_target(
            self.target
        )

        if (
            isinstance(self.target_count, bool)
            or not isinstance(
                self.target_count,
                int,
            )
            or not 1 <= self.target_count <= MAX_TARGETS
        ):
            raise InvalidRemediationRequestError(
                f"target_count must be between 1 "
                f"and {MAX_TARGETS}."
            )

        if not isinstance(
            self.risk_level,
            RiskLevel,
        ):
            raise InvalidRemediationRequestError(
                "risk_level must be RiskLevel."
            )

        if (
            isinstance(self.confidence, bool)
            or not isinstance(
                self.confidence,
                (int, float),
            )
            or not math.isfinite(
                float(self.confidence)
            )
            or not 0.0 <= float(
                self.confidence
            ) <= 1.0
        ):
            raise InvalidRemediationRequestError(
                "confidence must be a finite number "
                "between 0.0 and 1.0."
            )

        object.__setattr__(
            self,
            "confidence",
            float(self.confidence),
        )

        if not isinstance(
            self.evidence_ids,
            tuple,
        ):
            raise InvalidRemediationRequestError(
                "evidence_ids must be a tuple."
            )

        if len(self.evidence_ids) > MAX_EVIDENCE_ITEMS:
            raise InvalidRemediationRequestError(
                f"evidence_ids cannot exceed "
                f"{MAX_EVIDENCE_ITEMS} entries."
            )

        for evidence_id in self.evidence_ids:
            _validate_identifier(
                evidence_id,
                field="evidence_id",
            )

        for field, value in (
            (
                "rollback_available",
                self.rollback_available,
            ),
            (
                "protected_resource",
                self.protected_resource,
            ),
            (
                "dry_run",
                self.dry_run,
            ),
            (
                "human_approved",
                self.human_approved,
            ),
        ):
            if not isinstance(value, bool):
                raise InvalidRemediationRequestError(
                    f"{field} must be boolean."
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "correlation_id": self.correlation_id,
            "dry_run": self.dry_run,
            "evidence_ids": list(
                self.evidence_ids
            ),
            "human_approved": self.human_approved,
            "platform": self.platform.value,
            "protected_resource": (
                self.protected_resource
            ),
            "request_id": self.request_id,
            "risk_level": self.risk_level.value,
            "rollback_available": (
                self.rollback_available
            ),
            "target": self.target,
            "target_count": self.target_count,
        }


@dataclass(frozen=True, slots=True)
class RemediationPolicy:
    """
    Configuration controlling automated remediation.

    automated_actions is an explicit allow-list.

    Anything not listed cannot run automatically.

    denied_actions is an explicit deny-list and takes precedence over
    automated_actions.

    production_requires_approval can force approval for all non-dry-run
    production remediation.
    """

    automated_actions: frozenset[RemediationAction]

    denied_actions: frozenset[RemediationAction] = frozenset()

    minimum_confidence: float = (
        DEFAULT_MIN_AUTOMATION_CONFIDENCE
    )

    max_automated_targets: int = (
        DEFAULT_MAX_AUTOMATED_TARGETS
    )

    require_evidence: bool = True
    require_rollback: bool = True

    allow_moderate_risk_automation: bool = False

    production_requires_approval: bool = True

    protected_resources_require_approval: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.automated_actions,
            frozenset,
        ):
            raise InvalidRemediationRequestError(
                "automated_actions must be frozenset."
            )

        if not isinstance(
            self.denied_actions,
            frozenset,
        ):
            raise InvalidRemediationRequestError(
                "denied_actions must be frozenset."
            )

        for action in (
            self.automated_actions
            | self.denied_actions
        ):
            if not isinstance(
                action,
                RemediationAction,
            ):
                raise InvalidRemediationRequestError(
                    "Policy actions must be RemediationAction."
                )

        overlap = (
            self.automated_actions
            & self.denied_actions
        )

        if overlap:
            names = ", ".join(
                sorted(
                    action.value
                    for action in overlap
                )
            )

            raise InvalidRemediationRequestError(
                "Actions cannot be both automated and denied: "
                + names
            )

        if (
            isinstance(self.minimum_confidence, bool)
            or not isinstance(
                self.minimum_confidence,
                (int, float),
            )
            or not math.isfinite(
                float(self.minimum_confidence)
            )
            or not 0.0 <= float(
                self.minimum_confidence
            ) <= 1.0
        ):
            raise InvalidRemediationRequestError(
                "minimum_confidence must be between "
                "0.0 and 1.0."
            )

        object.__setattr__(
            self,
            "minimum_confidence",
            float(self.minimum_confidence),
        )

        if (
            isinstance(self.max_automated_targets, bool)
            or not isinstance(
                self.max_automated_targets,
                int,
            )
            or not 1 <= (
                self.max_automated_targets
            ) <= MAX_TARGETS
        ):
            raise InvalidRemediationRequestError(
                "max_automated_targets is invalid."
            )

        for field, value in (
            (
                "require_evidence",
                self.require_evidence,
            ),
            (
                "require_rollback",
                self.require_rollback,
            ),
            (
                "allow_moderate_risk_automation",
                self.allow_moderate_risk_automation,
            ),
            (
                "production_requires_approval",
                self.production_requires_approval,
            ),
            (
                "protected_resources_require_approval",
                self.protected_resources_require_approval,
            ),
        ):
            if not isinstance(value, bool):
                raise InvalidRemediationRequestError(
                    f"{field} must be boolean."
                )


@dataclass(frozen=True, slots=True)
class RemediationDecisionResult:
    """
    Structured remediation-safety decision.

    This object should later be signed and written to the audit chain before
    execution.
    """

    decision_id: str
    decision: RemediationDecision

    request: RemediationRequest

    authorization: PolicyDecisionResult

    workload_id: str

    reason: str

    evaluated_at_ns: int

    checks: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_identifier(
            self.decision_id,
            field="decision_id",
        )

        if not isinstance(
            self.decision,
            RemediationDecision,
        ):
            raise InvalidRemediationRequestError(
                "decision must be RemediationDecision."
            )

        if not isinstance(
            self.request,
            RemediationRequest,
        ):
            raise InvalidRemediationRequestError(
                "request must be RemediationRequest."
            )

        if not isinstance(
            self.authorization,
            PolicyDecisionResult,
        ):
            raise InvalidRemediationRequestError(
                "authorization must be PolicyDecisionResult."
            )

        if (
            not isinstance(self.reason, str)
            or not self.reason
        ):
            raise InvalidRemediationRequestError(
                "reason must be non-empty."
            )

        if len(self.reason) > MAX_REASON_LENGTH:
            raise InvalidRemediationRequestError(
                "reason exceeds maximum length."
            )

    @property
    def allowed(self) -> bool:
        return (
            self.decision
            is RemediationDecision.ALLOWED
        )

    @property
    def approval_required(self) -> bool:
        return (
            self.decision
            is RemediationDecision.REQUIRE_APPROVAL
        )

    def require_allowed(self) -> None:
        """
        Fail closed unless remediation is allowed.
        """

        if self.approval_required:
            raise RemediationApprovalRequiredError(
                self.reason
            )

        if not self.allowed:
            raise RemediationDeniedError(
                self.reason
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "authorization": (
                self.authorization.as_dict()
            ),
            "checks": list(self.checks),
            "decision": self.decision.value,
            "decision_id": self.decision_id,
            "evaluated_at_ns": self.evaluated_at_ns,
            "format_version": (
                REMEDIATION_POLICY_FORMAT_VERSION
            ),
            "reason": self.reason,
            "request": self.request.as_dict(),
            "workload_id": self.workload_id,
        }


class RemediationPolicyEngine:
    """
    OSHA remediation safety-policy engine.

    AuthorizationEngine is evaluated first.

    Only after authorization succeeds does this layer evaluate the additional
    safety restrictions required for infrastructure modification.

    Evaluation is deliberately fail-closed.
    """

    def __init__(
        self,
        *,
        authorization_engine: AuthorizationEngine,
        policy: RemediationPolicy,
    ) -> None:
        if not isinstance(
            authorization_engine,
            AuthorizationEngine,
        ):
            raise TypeError(
                "authorization_engine must be "
                "AuthorizationEngine."
            )

        if not isinstance(
            policy,
            RemediationPolicy,
        ):
            raise TypeError(
                "policy must be RemediationPolicy."
            )

        self._authorization_engine = (
            authorization_engine
        )
        self._policy = policy

    def evaluate(
        self,
        workload: AuthenticatedWorkload,
        request: RemediationRequest,
    ) -> RemediationDecisionResult:
        """
        Evaluate whether proposed remediation may proceed.

        This method does not execute the action.
        """

        if not isinstance(
            workload,
            AuthenticatedWorkload,
        ):
            raise InvalidRemediationRequestError(
                "workload must be AuthenticatedWorkload."
            )

        if not isinstance(
            request,
            RemediationRequest,
        ):
            raise InvalidRemediationRequestError(
                "request must be RemediationRequest."
            )

        checks: list[str] = []

        authorization_context = (
            AuthorizationContext(
                permission=_remediation_permission(
                    request
                ),
                risk_level=request.risk_level,
                correlation_id=request.correlation_id,
                evidence_ids=request.evidence_ids,
                attributes={
                    "action": request.action.value,
                    "platform": request.platform.value,
                    "target": request.target,
                },
                human_approved=request.human_approved,
            )
        )

        authorization = (
            self._authorization_engine.evaluate(
                workload,
                authorization_context,
            )
        )

        checks.append(
            "contextual_authorization_evaluated"
        )

        if authorization.decision is (
            PolicyDecision.DENIED
        ):
            return self._result(
                workload=workload,
                request=request,
                authorization=authorization,
                decision=RemediationDecision.DENIED,
                reason=(
                    "Contextual authorization denied "
                    "the remediation request."
                ),
                checks=checks,
            )

        if authorization.decision is (
            PolicyDecision.REQUIRE_APPROVAL
        ):
            return self._result(
                workload=workload,
                request=request,
                authorization=authorization,
                decision=(
                    RemediationDecision.REQUIRE_APPROVAL
                ),
                reason=(
                    "Contextual authorization requires "
                    "approval before remediation."
                ),
                checks=checks,
            )

        checks.append(
            "contextual_authorization_allowed"
        )

        if request.action in (
            self._policy.denied_actions
        ):
            checks.append(
                "explicit_action_deny"
            )

            return self._result(
                workload=workload,
                request=request,
                authorization=authorization,
                decision=RemediationDecision.DENIED,
                reason=(
                    "The remediation action is explicitly "
                    "denied by policy."
                ),
                checks=checks,
            )

        checks.append(
            "action_not_explicitly_denied"
        )

        if (
            request.risk_level
            is RiskLevel.CRITICAL
        ):
            checks.append(
                "critical_risk"
            )

            return self._result(
                workload=workload,
                request=request,
                authorization=authorization,
                decision=RemediationDecision.DENIED,
                reason=(
                    "Critical-risk remediation cannot be "
                    "automatically authorized by this policy."
                ),
                checks=checks,
            )

        if (
            request.protected_resource
            and self._policy
            .protected_resources_require_approval
            and not request.human_approved
        ):
            checks.append(
                "protected_resource_requires_approval"
            )

            return self._result(
                workload=workload,
                request=request,
                authorization=authorization,
                decision=(
                    RemediationDecision.REQUIRE_APPROVAL
                ),
                reason=(
                    "The target is a protected resource and "
                    "requires approval."
                ),
                checks=checks,
            )

        if (
            workload.identity.environment
            is DeploymentEnvironment.PRODUCTION
            and self._policy.production_requires_approval
            and not request.dry_run
            and not request.human_approved
        ):
            checks.append(
                "production_requires_approval"
            )

            return self._result(
                workload=workload,
                request=request,
                authorization=authorization,
                decision=(
                    RemediationDecision.REQUIRE_APPROVAL
                ),
                reason=(
                    "Production remediation requires "
                    "approval under current policy."
                ),
                checks=checks,
            )

        if (
            request.risk_level
            is RiskLevel.HIGH
            and not request.human_approved
        ):
            checks.append(
                "high_risk_requires_approval"
            )

            return self._result(
                workload=workload,
                request=request,
                authorization=authorization,
                decision=(
                    RemediationDecision.REQUIRE_APPROVAL
                ),
                reason=(
                    "High-risk remediation requires "
                    "approval."
                ),
                checks=checks,
            )

        if (
            request.risk_level
            is RiskLevel.MODERATE
            and not self._policy
            .allow_moderate_risk_automation
            and not request.human_approved
        ):
            checks.append(
                "moderate_risk_requires_approval"
            )

            return self._result(
                workload=workload,
                request=request,
                authorization=authorization,
                decision=(
                    RemediationDecision.REQUIRE_APPROVAL
                ),
                reason=(
                    "Moderate-risk autonomous remediation "
                    "is disabled by policy."
                ),
                checks=checks,
            )

        if request.human_approved:
            checks.append(
                "trusted_upstream_approval_present"
            )

        if (
            self._policy.require_evidence
            and not request.evidence_ids
        ):
            checks.append(
                "evidence_missing"
            )

            return self._result(
                workload=workload,
                request=request,
                authorization=authorization,
                decision=RemediationDecision.DENIED,
                reason=(
                    "Required remediation evidence is missing."
                ),
                checks=checks,
            )

        checks.append(
            "evidence_requirement_satisfied"
        )

        if (
            self._policy.require_rollback
            and not request.rollback_available
            and not request.dry_run
        ):
            checks.append(
                "rollback_missing"
            )

            return self._result(
                workload=workload,
                request=request,
                authorization=authorization,
                decision=RemediationDecision.DENIED,
                reason=(
                    "Rollback is required for this policy "
                    "but is unavailable."
                ),
                checks=checks,
            )

        checks.append(
            "rollback_requirement_satisfied"
        )

        if (
            request.action
            not in self._policy.automated_actions
            and not request.human_approved
        ):
            checks.append(
                "action_not_automation_allowlisted"
            )

            return self._result(
                workload=workload,
                request=request,
                authorization=authorization,
                decision=(
                    RemediationDecision.REQUIRE_APPROVAL
                ),
                reason=(
                    "The action is not approved for "
                    "autonomous remediation."
                ),
                checks=checks,
            )

        if (
            request.target_count
            > self._policy.max_automated_targets
            and not request.human_approved
        ):
            checks.append(
                "automation_blast_radius_exceeded"
            )

            return self._result(
                workload=workload,
                request=request,
                authorization=authorization,
                decision=(
                    RemediationDecision.REQUIRE_APPROVAL
                ),
                reason=(
                    "The requested remediation exceeds the "
                    "configured autonomous target limit."
                ),
                checks=checks,
            )

        checks.append(
            "target_scope_accepted"
        )

        if (
            request.confidence
            < self._policy.minimum_confidence
            and not request.human_approved
        ):
            checks.append(
                "confidence_below_automation_threshold"
            )

            return self._result(
                workload=workload,
                request=request,
                authorization=authorization,
                decision=(
                    RemediationDecision.REQUIRE_APPROVAL
                ),
                reason=(
                    "AI confidence is below the configured "
                    "autonomous-remediation threshold."
                ),
                checks=checks,
            )

        checks.append(
            "confidence_requirement_satisfied"
        )

        if request.dry_run:
            checks.append(
                "dry_run"
            )

        checks.append(
            "remediation_policy_allowed"
        )

        return self._result(
            workload=workload,
            request=request,
            authorization=authorization,
            decision=RemediationDecision.ALLOWED,
            reason=(
                "Remediation request satisfied all configured "
                "authorization and safety-policy checks."
            ),
            checks=checks,
        )

    def require(
        self,
        workload: AuthenticatedWorkload,
        request: RemediationRequest,
    ) -> RemediationDecisionResult:
        """
        Evaluate remediation and require final ALLOW.
        """

        result = self.evaluate(
            workload,
            request,
        )

        result.require_allowed()

        return result

    @staticmethod
    def _result(
        *,
        workload: AuthenticatedWorkload,
        request: RemediationRequest,
        authorization: PolicyDecisionResult,
        decision: RemediationDecision,
        reason: str,
        checks: Iterable[str],
    ) -> RemediationDecisionResult:

        return RemediationDecisionResult(
            decision_id=str(
                uuid.uuid4()
            ),
            decision=decision,
            request=request,
            authorization=authorization,
            workload_id=(
                workload.identity.workload_id
            ),
            reason=reason,
            evaluated_at_ns=time.time_ns(),
            checks=tuple(checks),
        )


def new_remediation_request(
    *,
    correlation_id: str,
    platform: InfrastructurePlatform,
    action: RemediationAction,
    target: str,
    risk_level: RiskLevel,
    confidence: float,
    evidence_ids: tuple[str, ...],
    rollback_available: bool,
    target_count: int = 1,
    protected_resource: bool = False,
    dry_run: bool = False,
    human_approved: bool = False,
) -> RemediationRequest:
    """
    Construct a remediation request with a random request ID.
    """

    return RemediationRequest(
        request_id=str(
            uuid.uuid4()
        ),
        correlation_id=correlation_id,
        platform=platform,
        action=action,
        target=target,
        target_count=target_count,
        risk_level=risk_level,
        confidence=confidence,
        evidence_ids=evidence_ids,
        rollback_available=rollback_available,
        protected_resource=protected_resource,
        dry_run=dry_run,
        human_approved=human_approved,
    )


def _remediation_permission(
    request: RemediationRequest,
):
    """
    Convert a remediation proposal into the permission resource expected by
    permissions.py and authorisation.py.
    """

    from ..identity.permissions import Permission

    resource = (
        f"remediation/"
        f"{request.platform.value}/"
        f"{request.target}"
    )

    return Permission(
        action=PermissionAction.REMEDIATE,
        resource=resource,
    )


def _validate_identifier(
    value: str,
    *,
    field: str,
) -> None:

    if (
        not isinstance(value, str)
        or not value
    ):
        raise InvalidRemediationRequestError(
            f"{field} must be a non-empty string."
        )

    if len(value) > MAX_ID_LENGTH:
        raise InvalidRemediationRequestError(
            f"{field} exceeds maximum length "
            f"of {MAX_ID_LENGTH}."
        )

    if (
        _IDENTIFIER_PATTERN.fullmatch(
            value
        )
        is None
    ):
        raise InvalidRemediationRequestError(
            f"{field} contains invalid characters."
        )


def _validate_target(
    target: str,
) -> None:

    if (
        not isinstance(target, str)
        or not target
    ):
        raise InvalidRemediationRequestError(
            "target must be a non-empty string."
        )

    if target != target.strip().lower():
        raise InvalidRemediationRequestError(
            "target must be normalized lowercase."
        )

    if len(target) > MAX_TARGET_LENGTH:
        raise InvalidRemediationRequestError(
            "target exceeds maximum length."
        )

    if not re.fullmatch(
        r"[a-z0-9][a-z0-9._:/@-]*",
        target,
    ):
        raise InvalidRemediationRequestError(
            "target contains invalid characters."
        )


__all__ = [
    "DEFAULT_MAX_AUTOMATED_TARGETS",
    "DEFAULT_MIN_AUTOMATION_CONFIDENCE",
    "InfrastructurePlatform",
    "InvalidRemediationRequestError",
    "REMEDIATION_POLICY_FORMAT_VERSION",
    "RemediationAction",
    "RemediationApprovalRequiredError",
    "RemediationDecision",
    "RemediationDecisionResult",
    "RemediationDeniedError",
    "RemediationPolicy",
    "RemediationPolicyEngine",
    "RemediationPolicyError",
    "RemediationRequest",
    "new_remediation_request",
]
