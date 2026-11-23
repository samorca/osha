from __future__ import annotations

"""
Shared security contracts for OSHA authentication,
authorization, policy, secrets and audit.
"""

from dataclasses import (
    dataclass,
    field,
)

from enum import Enum

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from src.core.identifiers import (
    random_id,
)

from src.core.serialization import (
    to_json_safe,
)

from src.core.time import (
    utc_now,
)

from src.core.validation import (
    environment_id,
    required_string,
    unique_strings,
)


SECURITY_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------
# Principal types
# ---------------------------------------------------------------

class PrincipalType(
    str,
    Enum,
):
    USER = "user"

    SERVICE = "service"

    WORKER = "worker"

    SYSTEM = "system"


# ---------------------------------------------------------------
# Authentication state
# ---------------------------------------------------------------

class AuthenticationStatus(
    str,
    Enum,
):
    UNKNOWN = "unknown"

    AUTHENTICATED = "authenticated"

    UNAUTHENTICATED = "unauthenticated"

    EXPIRED = "expired"

    REVOKED = "revoked"


# ---------------------------------------------------------------
# Authorization decisions
# ---------------------------------------------------------------

class AuthorizationDecision(
    str,
    Enum,
):
    ALLOW = "allow"

    DENY = "deny"

    REQUIRE_APPROVAL = (
        "require_approval"
    )

    ABSTAIN = "abstain"


# ---------------------------------------------------------------
# Policy effect
# ---------------------------------------------------------------

class PolicyEffect(
    str,
    Enum,
):
    ALLOW = "allow"

    DENY = "deny"

    REQUIRE_APPROVAL = (
        "require_approval"
    )


# ---------------------------------------------------------------
# Audit outcome
# ---------------------------------------------------------------

class AuditOutcome(
    str,
    Enum,
):
    SUCCESS = "success"

    FAILURE = "failure"

    BLOCKED = "blocked"

    UNKNOWN = "unknown"


# ---------------------------------------------------------------
# Security principal
# ---------------------------------------------------------------

@dataclass(frozen=True)
class SecurityPrincipal:
    """
    An authenticated or otherwise identified
    OSHA actor.

    Examples:

      human user
      internal service
      worker identity
      trusted system process
    """

    principal_id: str

    principal_type: str

    display_name: Optional[
        str
    ] = None

    roles: List[
        str
    ] = field(
        default_factory=list
    )

    groups: List[
        str
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "SecurityPrincipal":
        required_string(
            self.principal_id,
            name="principal_id",
            maximum_length=512,
        )

        if (
            self.principal_type
            not in {
                item.value
                for item
                in PrincipalType
            }
        ):
            raise ValueError(
                "Unknown principal type: "
                f"{self.principal_type}"
            )

        object.__setattr__(
            self,
            "roles",
            unique_strings(
                self.roles
            ),
        )

        object.__setattr__(
            self,
            "groups",
            unique_strings(
                self.groups
            ),
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "principal_id": (
                self.principal_id
            ),
            "principal_type": (
                self.principal_type
            ),
            "display_name": (
                self.display_name
            ),
            "roles": list(
                self.roles
            ),
            "groups": list(
                self.groups
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Authentication context
# ---------------------------------------------------------------

@dataclass
class AuthenticationContext:
    """
    Authentication state associated with an OSHA
    request/session.

    No raw password, token or private key belongs
    in this object.
    """

    status: str = (
        AuthenticationStatus.
        UNKNOWN.value
    )

    principal: Optional[
        SecurityPrincipal
    ] = None

    method: Optional[
        str
    ] = None

    authenticated_at: Optional[
        str
    ] = None

    expires_at: Optional[
        str
    ] = None

    session_id: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "AuthenticationContext":
        if (
            self.status
            not in {
                item.value
                for item
                in AuthenticationStatus
            }
        ):
            raise ValueError(
                "Unknown authentication "
                f"status: {self.status}"
            )

        if self.principal is not None:
            self.principal.finalize()

        if (
            self.status
            == AuthenticationStatus.
            AUTHENTICATED.value
            and self.principal is None
        ):
            raise ValueError(
                "Authenticated context "
                "requires a principal"
            )

        if not self.session_id:
            self.session_id = (
                random_id(
                    "security:session"
                )
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "status": self.status,

            "principal": (
                self.principal.to_dict()
                if self.principal
                else None
            ),

            "method": self.method,

            "authenticated_at": (
                self.authenticated_at
            ),

            "expires_at": (
                self.expires_at
            ),

            "session_id": (
                self.session_id
            ),

            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Permission
# ---------------------------------------------------------------

@dataclass(frozen=True)
class Permission:
    """
    Permission being requested.

    Example conceptual values:

      resource = remediation
      action   = execute
      scope    = cluster-A
    """

    resource: str

    action: str

    scope: Optional[
        str
    ] = None

    environment_id: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "Permission":
        required_string(
            self.resource,
            name=(
                "permission resource"
            ),
            maximum_length=256,
        )

        required_string(
            self.action,
            name=(
                "permission action"
            ),
            maximum_length=128,
        )

        if (
            self.environment_id
            is not None
        ):
            environment_id(
                self.environment_id
            )

        return self

    def key(
        self,
    ) -> str:
        self.finalize()

        return ":".join(
            part
            for part
            in (
                self.environment_id
                or "*",

                self.resource,

                self.action,

                self.scope
                or "*",
            )
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "resource": (
                self.resource
            ),
            "action": self.action,
            "scope": self.scope,
            "environment_id": (
                self.environment_id
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Authorization request
# ---------------------------------------------------------------

@dataclass
class AuthorizationRequest:
    principal: SecurityPrincipal

    permission: Permission

    resource_id: Optional[
        str
    ] = None

    requested_at: str = field(
        default_factory=utc_now
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    request_id: Optional[
        str
    ] = None

    def finalize(
        self,
    ) -> "AuthorizationRequest":
        self.principal.finalize()
        self.permission.finalize()

        if not self.request_id:
            self.request_id = (
                random_id(
                    "security:"
                    "authorization-request"
                )
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "request_id": (
                self.request_id
            ),
            "principal": (
                self.principal.
                to_dict()
            ),
            "permission": (
                self.permission.
                to_dict()
            ),
            "resource_id": (
                self.resource_id
            ),
            "requested_at": (
                self.requested_at
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Authorization result
# ---------------------------------------------------------------

@dataclass
class AuthorizationResult:
    decision: str

    request_id: Optional[
        str
    ] = None

    reason: Optional[
        str
    ] = None

    policy_ids: List[
        str
    ] = field(
        default_factory=list
    )

    required_approval_roles: List[
        str
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    decided_at: str = field(
        default_factory=utc_now
    )

    decision_id: Optional[
        str
    ] = None

    def finalize(
        self,
    ) -> "AuthorizationResult":
        if (
            self.decision
            not in {
                item.value
                for item
                in AuthorizationDecision
            }
        ):
            raise ValueError(
                "Unknown authorization "
                f"decision: {self.decision}"
            )

        self.policy_ids = (
            unique_strings(
                self.policy_ids
            )
        )

        self.required_approval_roles = (
            unique_strings(
                self.
                required_approval_roles
            )
        )

        if not self.decision_id:
            self.decision_id = (
                random_id(
                    "security:"
                    "authorization-decision"
                )
            )

        return self

    @property
    def allowed(
        self,
    ) -> bool:
        return (
            self.decision
            == AuthorizationDecision.
            ALLOW.value
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "decision_id": (
                self.decision_id
            ),
            "request_id": (
                self.request_id
            ),
            "decision": (
                self.decision
            ),
            "allowed": (
                self.allowed
            ),
            "reason": self.reason,
            "policy_ids": list(
                self.policy_ids
            ),
            "required_approval_roles": (
                list(
                    self.
                    required_approval_roles
                )
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
            "decided_at": (
                self.decided_at
            ),
        }


# ---------------------------------------------------------------
# Security policy
# ---------------------------------------------------------------

@dataclass(frozen=True)
class SecurityPolicy:
    """
    Declarative authorization policy contract.

    Policy evaluation itself belongs in
    policies.py.
    """

    policy_id: str

    name: str

    effect: str

    resource: str

    actions: List[
        str
    ] = field(
        default_factory=list
    )

    roles: List[
        str
    ] = field(
        default_factory=list
    )

    environments: List[
        str
    ] = field(
        default_factory=list
    )

    enabled: bool = True

    priority: int = 100

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "SecurityPolicy":
        required_string(
            self.policy_id,
            name="policy_id",
            maximum_length=256,
        )

        required_string(
            self.name,
            name="policy name",
            maximum_length=256,
        )

        required_string(
            self.resource,
            name=(
                "policy resource"
            ),
            maximum_length=256,
        )

        if (
            self.effect
            not in {
                item.value
                for item
                in PolicyEffect
            }
        ):
            raise ValueError(
                "Unknown policy effect: "
                f"{self.effect}"
            )

        object.__setattr__(
            self,
            "actions",
            unique_strings(
                self.actions
            ),
        )

        object.__setattr__(
            self,
            "roles",
            unique_strings(
                self.roles
            ),
        )

        object.__setattr__(
            self,
            "environments",
            unique_strings(
                self.environments
            ),
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "policy_id": (
                self.policy_id
            ),
            "name": self.name,
            "effect": (
                self.effect
            ),
            "resource": (
                self.resource
            ),
            "actions": list(
                self.actions
            ),
            "roles": list(
                self.roles
            ),
            "environments": list(
                self.environments
            ),
            "enabled": self.enabled,
            "priority": self.priority,
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Secret reference
# ---------------------------------------------------------------

@dataclass
class SecretReference:
    """
    Reference to a secret.

    The actual secret VALUE must never be stored
    in this structure.
    """

    name: str

    provider: str

    key: Optional[
        str
    ] = None

    version: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    reference_id: Optional[
        str
    ] = None

    def finalize(
        self,
    ) -> "SecretReference":
        required_string(
            self.name,
            name="secret name",
            maximum_length=256,
        )

        required_string(
            self.provider,
            name="secret provider",
            maximum_length=128,
        )

        if not self.reference_id:
            self.reference_id = (
                random_id(
                    "security:"
                    "secret-reference"
                )
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "reference_id": (
                self.reference_id
            ),
            "name": self.name,
            "provider": (
                self.provider
            ),
            "key": self.key,
            "version": self.version,
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Security audit record
# ---------------------------------------------------------------

@dataclass
class SecurityAuditRecord:
    event: str

    outcome: str

    principal_id: Optional[
        str
    ] = None

    environment_id: Optional[
        str
    ] = None

    resource: Optional[
        str
    ] = None

    action: Optional[
        str
    ] = None

    reference_ids: List[
        str
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    occurred_at: str = field(
        default_factory=utc_now
    )

    audit_id: Optional[
        str
    ] = None

    def finalize(
        self,
    ) -> "SecurityAuditRecord":
        required_string(
            self.event,
            name=(
                "security audit event"
            ),
            maximum_length=256,
        )

        if (
            self.outcome
            not in {
                item.value
                for item
                in AuditOutcome
            }
        ):
            raise ValueError(
                "Unknown audit outcome: "
                f"{self.outcome}"
            )

        if (
            self.environment_id
            is not None
        ):
            environment_id(
                self.environment_id
            )

        self.reference_ids = (
            unique_strings(
                self.reference_ids
            )
        )

        if not self.audit_id:
            self.audit_id = (
                random_id(
                    "security:audit"
                )
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (
                SECURITY_SCHEMA_VERSION
            ),
            "audit_id": (
                self.audit_id
            ),
            "event": self.event,
            "outcome": self.outcome,
            "principal_id": (
                self.principal_id
            ),
            "environment_id": (
                self.environment_id
            ),
            "resource": (
                self.resource
            ),
            "action": self.action,
            "reference_ids": list(
                self.reference_ids
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
            "occurred_at": (
                self.occurred_at
            ),
        }


# ---------------------------------------------------------------
# Security context
# ---------------------------------------------------------------

@dataclass
class SecurityContext:
    """
    Security state associated with one OSHA
    operation/request.

    This gives future API/orchestrator paths a
    common object for passing authenticated
    identity and authorization results.
    """

    environment_id: str

    authentication: AuthenticationContext = (
        field(
            default_factory=(
                AuthenticationContext
            )
        )
    )

    authorization: Optional[
        AuthorizationResult
    ] = None

    correlation_id: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    context_id: Optional[
        str
    ] = None

    def finalize(
        self,
    ) -> "SecurityContext":
        environment_id(
            self.environment_id
        )

        self.authentication.finalize()

        if (
            self.authorization
            is not None
        ):
            self.authorization.finalize()

        if not self.context_id:
            self.context_id = (
                random_id(
                    "security:context"
                )
            )

        return self

    @property
    def principal(
        self,
    ) -> Optional[
        SecurityPrincipal
    ]:
        return (
            self.authentication.
            principal
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (
                SECURITY_SCHEMA_VERSION
            ),
            "context_id": (
                self.context_id
            ),
            "environment_id": (
                self.environment_id
            ),
            "authentication": (
                self.authentication.
                to_dict()
            ),
            "authorization": (
                self.authorization.
                to_dict()
                if self.authorization
                else None
            ),
            "correlation_id": (
                self.correlation_id
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "SECURITY_SCHEMA_VERSION",

    "PrincipalType",
    "AuthenticationStatus",
    "AuthorizationDecision",
    "PolicyEffect",
    "AuditOutcome",

    "SecurityPrincipal",
    "AuthenticationContext",

    "Permission",

    "AuthorizationRequest",
    "AuthorizationResult",

    "SecurityPolicy",

    "SecretReference",

    "SecurityAuditRecord",

    "SecurityContext",
]
