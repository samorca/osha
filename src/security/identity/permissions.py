"""
Permission and authorization primitives for OSHA workloads.

This module answers:

    "What is this authenticated OSHA workload allowed to request?"

It builds on:

    workload_identity.py
        -> describes workload identity

    certificates.py
        -> provides cryptographic authentication evidence

    permissions.py
        -> evaluates coarse-grained workload permissions

    policy/authorisation.py
        -> will make higher-level contextual authorization decisions

Security design
---------------
1. Permissions are denied by default.
2. Only authenticated workloads may receive authorization.
3. Permissions are explicit action/resource pairs.
4. Rules can constrain trust domain, environment, workload type,
   namespace, workload name, resource, and action.
5. Wildcards are supported only where explicitly represented by rules.
6. Explicit DENY rules override ALLOW rules.
7. No network-location-based implicit trust is granted.
8. Permission evaluation contains no cryptographic secrets.
9. This module does not execute actions.
10. Higher-risk remediation decisions require additional policy evaluation.
"""

from __future__ import annotations

import re
import time

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Iterable

from .workload_identity import (
    AuthenticatedWorkload,
    DeploymentEnvironment,
    WorkloadIdentity,
    WorkloadType,
)


PERMISSION_FORMAT_VERSION: Final[int] = 1

MAX_RESOURCE_LENGTH: Final[int] = 512
MAX_RULE_ID_LENGTH: Final[int] = 128
MAX_REASON_LENGTH: Final[int] = 512

_RESOURCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9][a-z0-9._:/@*-]*$"
)


class PermissionError(Exception):
    """Base exception for OSHA permission failures."""


class InvalidPermissionError(
    PermissionError,
    ValueError,
):
    """Permission or permission rule is malformed."""


class PermissionDeniedError(PermissionError):
    """Required authorization was denied."""


class PermissionAction(str, Enum):
    """
    Coarse-grained actions understood by OSHA authorization.

    These actions describe authorization intent. They do not perform
    the corresponding infrastructure operation.
    """

    READ = "read"
    WRITE = "write"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"

    COLLECT = "collect"
    INGEST = "ingest"

    TRAIN = "train"
    INFER = "infer"

    CORRELATE = "correlate"
    ANALYZE = "analyze"
    DECIDE = "decide"

    RECOMMEND = "recommend"
    REMEDIATE = "remediate"

    SIGN = "sign"
    VERIFY = "verify"

    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"

    AUDIT = "audit"

    HEALTH_CHECK = "health_check"
    ADMINISTER = "administer"


class PermissionEffect(str, Enum):
    """Effect of a permission rule."""

    ALLOW = "allow"
    DENY = "deny"


class PermissionDecision(str, Enum):
    """Result of permission evaluation."""

    ALLOWED = "allowed"
    DENIED = "denied"


@dataclass(frozen=True, slots=True)
class Permission:
    """
    Requested authorization.

    resource examples:

        telemetry/linux/node-01
        models/event-classifier
        datasets/loghub
        remediation/kubernetes/deployment
        audit/events
        inference/event-classifier

    The resource is a logical OSHA resource identifier, not necessarily
    a filesystem path or URL.
    """

    action: PermissionAction
    resource: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.action,
            PermissionAction,
        ):
            raise InvalidPermissionError(
                "action must be PermissionAction."
            )

        resource = _normalize_resource(
            self.resource
        )

        object.__setattr__(
            self,
            "resource",
            resource,
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "action": self.action.value,
            "resource": self.resource,
        }

    def __str__(self) -> str:
        return (
            f"{self.action.value}:{self.resource}"
        )


@dataclass(frozen=True, slots=True)
class PermissionRule:
    """
    One immutable authorization rule.

    None in an identity selector means "any value".

    resource_pattern supports:

        exact resource:
            models/event-classifier

        subtree wildcard:
            telemetry/*

        all resources:
            *

    Wildcards are intentionally limited to a terminal '*'.
    """

    rule_id: str
    effect: PermissionEffect

    actions: frozenset[PermissionAction]

    resource_pattern: str

    trust_domain: str | None = None
    environment: DeploymentEnvironment | None = None
    workload_type: WorkloadType | None = None
    namespace: str | None = None
    workload_name: str | None = None

    description: str = ""

    def __post_init__(self) -> None:
        _validate_rule_id(
            self.rule_id
        )

        if not isinstance(
            self.effect,
            PermissionEffect,
        ):
            raise InvalidPermissionError(
                "effect must be PermissionEffect."
            )

        if not isinstance(
            self.actions,
            frozenset,
        ):
            raise InvalidPermissionError(
                "actions must be a frozenset."
            )

        if not self.actions:
            raise InvalidPermissionError(
                "Permission rule must contain at least "
                "one action."
            )

        for action in self.actions:
            if not isinstance(
                action,
                PermissionAction,
            ):
                raise InvalidPermissionError(
                    "Every rule action must be "
                    "PermissionAction."
                )

        pattern = _normalize_resource_pattern(
            self.resource_pattern
        )

        object.__setattr__(
            self,
            "resource_pattern",
            pattern,
        )

        if (
            self.environment is not None
            and not isinstance(
                self.environment,
                DeploymentEnvironment,
            )
        ):
            raise InvalidPermissionError(
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
            raise InvalidPermissionError(
                "workload_type must be WorkloadType or None."
            )

        if self.trust_domain is not None:
            _validate_selector(
                self.trust_domain,
                field="trust_domain",
            )

        if self.namespace is not None:
            _validate_selector(
                self.namespace,
                field="namespace",
            )

        if self.workload_name is not None:
            _validate_selector(
                self.workload_name,
                field="workload_name",
            )

        if not isinstance(
            self.description,
            str,
        ):
            raise InvalidPermissionError(
                "description must be a string."
            )

        if len(self.description) > (
            MAX_REASON_LENGTH
        ):
            raise InvalidPermissionError(
                "description exceeds maximum length."
            )

    def matches_identity(
        self,
        identity: WorkloadIdentity,
    ) -> bool:
        """Return whether this rule applies to the workload."""

        if not isinstance(
            identity,
            WorkloadIdentity,
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

        return True

    def matches_permission(
        self,
        permission: Permission,
    ) -> bool:
        """Return whether this rule applies to the permission."""

        if not isinstance(
            permission,
            Permission,
        ):
            return False

        if permission.action not in self.actions:
            return False

        return _resource_matches(
            permission.resource,
            self.resource_pattern,
        )

    def matches(
        self,
        identity: WorkloadIdentity,
        permission: Permission,
    ) -> bool:
        return (
            self.matches_identity(identity)
            and self.matches_permission(permission)
        )


@dataclass(frozen=True, slots=True)
class AuthorizationResult:
    """
    Result of one permission evaluation.

    matched_rule_ids contains the relevant rule IDs but no secrets.
    """

    decision: PermissionDecision
    permission: Permission
    workload_id: str
    reason: str
    evaluated_at_ns: int
    matched_rule_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.decision,
            PermissionDecision,
        ):
            raise InvalidPermissionError(
                "decision must be PermissionDecision."
            )

        if not isinstance(
            self.permission,
            Permission,
        ):
            raise InvalidPermissionError(
                "permission must be Permission."
            )

        if (
            not isinstance(self.workload_id, str)
            or not self.workload_id
        ):
            raise InvalidPermissionError(
                "workload_id must be non-empty."
            )

        if (
            not isinstance(self.reason, str)
            or not self.reason
        ):
            raise InvalidPermissionError(
                "reason must be non-empty."
            )

        if len(self.reason) > MAX_REASON_LENGTH:
            raise InvalidPermissionError(
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
            raise InvalidPermissionError(
                "evaluated_at_ns must be positive."
            )

    @property
    def allowed(self) -> bool:
        return (
            self.decision
            is PermissionDecision.ALLOWED
        )

    def require_allowed(self) -> None:
        """Raise unless authorization succeeded."""

        if not self.allowed:
            raise PermissionDeniedError(
                f"Permission denied for "
                f"{self.workload_id}: "
                f"{self.permission}. "
                f"{self.reason}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "evaluated_at_ns": self.evaluated_at_ns,
            "format_version": (
                PERMISSION_FORMAT_VERSION
            ),
            "matched_rule_ids": list(
                self.matched_rule_ids
            ),
            "permission": (
                self.permission.as_dict()
            ),
            "reason": self.reason,
            "workload_id": self.workload_id,
        }


class PermissionSet:
    """
    Immutable-style collection of OSHA permission rules.

    Evaluation rules
    ----------------
    1. Caller must be authenticated.
    2. Find rules matching identity + requested permission.
    3. If any matching rule DENIES -> DENY.
    4. Else if at least one matching rule ALLOWS -> ALLOW.
    5. Otherwise -> DENY.

    This provides default-deny behavior and explicit-deny precedence.
    """

    def __init__(
        self,
        rules: Iterable[PermissionRule] = (),
    ) -> None:

        resolved = tuple(rules)

        seen_ids: set[str] = set()

        for rule in resolved:
            if not isinstance(
                rule,
                PermissionRule,
            ):
                raise InvalidPermissionError(
                    "Every rule must be PermissionRule."
                )

            if rule.rule_id in seen_ids:
                raise InvalidPermissionError(
                    f"Duplicate permission rule ID: "
                    f"{rule.rule_id!r}."
                )

            seen_ids.add(
                rule.rule_id
            )

        self._rules = resolved

    @property
    def rules(
        self,
    ) -> tuple[PermissionRule, ...]:
        return self._rules

    def evaluate(
        self,
        workload: AuthenticatedWorkload,
        permission: Permission,
    ) -> AuthorizationResult:
        """
        Evaluate an authenticated workload request.

        Unverified workloads are always denied.
        """

        if not isinstance(
            workload,
            AuthenticatedWorkload,
        ):
            raise InvalidPermissionError(
                "workload must be AuthenticatedWorkload."
            )

        if not isinstance(
            permission,
            Permission,
        ):
            raise InvalidPermissionError(
                "permission must be Permission."
            )

        identity = workload.identity

        if not workload.verified:
            return _authorization_result(
                decision=PermissionDecision.DENIED,
                permission=permission,
                workload_id=identity.workload_id,
                reason=(
                    "Workload identity is not authenticated."
                ),
                matched_rule_ids=(),
            )

        matches = tuple(
            rule
            for rule in self._rules
            if rule.matches(
                identity,
                permission,
            )
        )

        deny_rules = tuple(
            rule
            for rule in matches
            if rule.effect
            is PermissionEffect.DENY
        )

        if deny_rules:
            return _authorization_result(
                decision=PermissionDecision.DENIED,
                permission=permission,
                workload_id=identity.workload_id,
                reason=(
                    "An explicit deny rule matched "
                    "the requested permission."
                ),
                matched_rule_ids=tuple(
                    rule.rule_id
                    for rule in deny_rules
                ),
            )

        allow_rules = tuple(
            rule
            for rule in matches
            if rule.effect
            is PermissionEffect.ALLOW
        )

        if allow_rules:
            return _authorization_result(
                decision=PermissionDecision.ALLOWED,
                permission=permission,
                workload_id=identity.workload_id,
                reason=(
                    "Permission granted by an explicit "
                    "allow rule."
                ),
                matched_rule_ids=tuple(
                    rule.rule_id
                    for rule in allow_rules
                ),
            )

        return _authorization_result(
            decision=PermissionDecision.DENIED,
            permission=permission,
            workload_id=identity.workload_id,
            reason=(
                "No permission rule explicitly allows "
                "the requested operation."
            ),
            matched_rule_ids=(),
        )

    def require(
        self,
        workload: AuthenticatedWorkload,
        permission: Permission,
    ) -> AuthorizationResult:
        """
        Require permission and fail closed when denied.
        """

        result = self.evaluate(
            workload,
            permission,
        )

        result.require_allowed()

        return result


def collector_default_rules(
    *,
    trust_domain: str,
    environment: DeploymentEnvironment,
) -> tuple[PermissionRule, ...]:
    """
    Suggested minimal collector rules.

    These are *suggested OSHA defaults*, not external standards.

    Collectors may collect and submit telemetry, but these defaults do not
    grant remediation, administration, training, or deletion permissions.
    """

    return (
        PermissionRule(
            rule_id="collector-collect-telemetry",
            effect=PermissionEffect.ALLOW,
            actions=frozenset(
                {
                    PermissionAction.COLLECT,
                }
            ),
            resource_pattern="telemetry/*",
            trust_domain=trust_domain,
            environment=environment,
            workload_type=WorkloadType.COLLECTOR,
            description=(
                "Collectors may collect telemetry."
            ),
        ),
        PermissionRule(
            rule_id="collector-ingest-telemetry",
            effect=PermissionEffect.ALLOW,
            actions=frozenset(
                {
                    PermissionAction.INGEST,
                }
            ),
            resource_pattern="telemetry/*",
            trust_domain=trust_domain,
            environment=environment,
            workload_type=WorkloadType.COLLECTOR,
            description=(
                "Collectors may submit telemetry."
            ),
        ),
    )


def remediation_default_rules(
    *,
    trust_domain: str,
    environment: DeploymentEnvironment,
    namespace: str = "system",
) -> tuple[PermissionRule, ...]:
    """
    Suggested coarse remediation permissions.

    IMPORTANT:
    Permission to request REMEDIATE does NOT mean an action should execute.

    The later remediation policy layer must evaluate risk, target, evidence,
    approvals, safety constraints, and execution policy.
    """

    return (
        PermissionRule(
            rule_id="remediation-request",
            effect=PermissionEffect.ALLOW,
            actions=frozenset(
                {
                    PermissionAction.REMEDIATE,
                }
            ),
            resource_pattern="remediation/*",
            trust_domain=trust_domain,
            environment=environment,
            workload_type=WorkloadType.REMEDIATION,
            namespace=namespace,
            description=(
                "Authenticated remediation workloads may "
                "request remediation policy evaluation."
            ),
        ),
    )


def _authorization_result(
    *,
    decision: PermissionDecision,
    permission: Permission,
    workload_id: str,
    reason: str,
    matched_rule_ids: tuple[str, ...],
) -> AuthorizationResult:

    return AuthorizationResult(
        decision=decision,
        permission=permission,
        workload_id=workload_id,
        reason=reason,
        evaluated_at_ns=time.time_ns(),
        matched_rule_ids=matched_rule_ids,
    )


def _resource_matches(
    resource: str,
    pattern: str,
) -> bool:

    if pattern == "*":
        return True

    if pattern.endswith("*"):
        prefix = pattern[:-1]

        return resource.startswith(
            prefix
        )

    return resource == pattern


def _normalize_resource(
    value: str,
) -> str:

    if not isinstance(value, str):
        raise InvalidPermissionError(
            "resource must be a string."
        )

    normalized = value.strip().lower()

    if not normalized:
        raise InvalidPermissionError(
            "resource cannot be empty."
        )

    if len(normalized) > MAX_RESOURCE_LENGTH:
        raise InvalidPermissionError(
            "resource exceeds maximum length."
        )

    if "*" in normalized:
        raise InvalidPermissionError(
            "Requested permissions cannot contain "
            "resource wildcards."
        )

    if (
        _RESOURCE_PATTERN.fullmatch(
            normalized
        )
        is None
    ):
        raise InvalidPermissionError(
            "resource contains invalid characters."
        )

    return normalized


def _normalize_resource_pattern(
    value: str,
) -> str:

    if not isinstance(value, str):
        raise InvalidPermissionError(
            "resource_pattern must be a string."
        )

    normalized = value.strip().lower()

    if not normalized:
        raise InvalidPermissionError(
            "resource_pattern cannot be empty."
        )

    if len(normalized) > MAX_RESOURCE_LENGTH:
        raise InvalidPermissionError(
            "resource_pattern exceeds maximum length."
        )

    if (
        _RESOURCE_PATTERN.fullmatch(
            normalized
        )
        is None
    ):
        raise InvalidPermissionError(
            "resource_pattern contains invalid characters."
        )

    wildcard_count = normalized.count("*")

    if wildcard_count > 1:
        raise InvalidPermissionError(
            "resource_pattern can contain at most "
            "one wildcard."
        )

    if (
        wildcard_count == 1
        and not normalized.endswith("*")
    ):
        raise InvalidPermissionError(
            "Resource wildcard is allowed only "
            "at the end of a pattern."
        )

    return normalized


def _validate_rule_id(
    value: str,
) -> None:

    if (
        not isinstance(value, str)
        or not value
    ):
        raise InvalidPermissionError(
            "rule_id must be a non-empty string."
        )

    if len(value) > MAX_RULE_ID_LENGTH:
        raise InvalidPermissionError(
            "rule_id exceeds maximum length."
        )

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        value,
    ):
        raise InvalidPermissionError(
            "rule_id contains invalid characters."
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
        raise InvalidPermissionError(
            f"{field} must be a non-empty string."
        )

    if value != value.strip().lower():
        raise InvalidPermissionError(
            f"{field} must already be normalized "
            "lowercase text."
        )


__all__ = [
    "AuthorizationResult",
    "InvalidPermissionError",
    "PERMISSION_FORMAT_VERSION",
    "Permission",
    "PermissionAction",
    "PermissionDecision",
    "PermissionDeniedError",
    "PermissionEffect",
    "PermissionError",
    "PermissionRule",
    "PermissionSet",
    "collector_default_rules",
    "remediation_default_rules",
]
