"""
OSHA Audit Event Model.

Provides durable audit records for security-sensitive, operational and
administrative activity within OSHA.

Audit events may represent:

    - remediation proposal / approval / rejection / execution
    - infrastructure configuration changes
    - incident lifecycle operations
    - authentication and authorisation activity
    - resource discovery changes
    - policy decisions
    - administrative operations
    - model or automation decisions
    - security-sensitive actions

The audit log is intended to be append-oriented. Existing audit records
should not normally be modified or deleted through ordinary application
workflows.

Sensitive values such as passwords, tokens, private keys and raw secrets
must never be written to audit payloads.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base, TenantMixin


# ---------------------------------------------------------------------------
# Audit classification
# ---------------------------------------------------------------------------

class AuditCategory(str, enum.Enum):
    """High-level category of an OSHA audit event."""

    AUTHENTICATION = "authentication"
    AUTHORISATION = "authorisation"

    ADMINISTRATION = "administration"
    CONFIGURATION = "configuration"

    RESOURCE = "resource"
    DISCOVERY = "discovery"

    ANOMALY = "anomaly"
    PREDICTION = "prediction"
    INCIDENT = "incident"

    REMEDIATION = "remediation"

    POLICY = "policy"
    SECURITY = "security"

    MODEL = "model"
    AUTOMATION = "automation"

    DATABASE = "database"

    SYSTEM = "system"

    OTHER = "other"


class AuditAction(str, enum.Enum):
    """Canonical action represented by an audit event."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    LOGIN = "login"
    LOGOUT = "logout"

    ENABLE = "enable"
    DISABLE = "disable"

    APPROVE = "approve"
    REJECT = "reject"

    EXECUTE = "execute"
    CANCEL = "cancel"

    START = "start"
    STOP = "stop"

    ACKNOWLEDGE = "acknowledge"
    RESOLVE = "resolve"
    CLOSE = "close"

    ASSIGN = "assign"

    DISCOVER = "discover"

    PREDICT = "predict"
    DETECT = "detect"
    CORRELATE = "correlate"

    REMEDIATE = "remediate"
    ROLLBACK = "rollback"

    ACCESS = "access"
    DENY = "deny"

    CUSTOM = "custom"


class AuditOutcome(str, enum.Enum):
    """Result of the audited operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"

    PARTIAL = "partial"
    CANCELLED = "cancelled"

    UNKNOWN = "unknown"


class AuditSeverity(str, enum.Enum):
    """Importance of an audit event."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActorType(str, enum.Enum):
    """Identity type responsible for an audited operation."""

    USER = "user"
    SERVICE = "service"
    COLLECTOR = "collector"
    WORKER = "worker"
    MODEL = "model"
    POLICY = "policy"
    SYSTEM = "system"
    API = "api"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Audit event model
# ---------------------------------------------------------------------------

class AuditEvent(
    TenantMixin,
    Base,
):
    """
    Immutable-by-policy record describing an OSHA operation.

    This model deliberately does not inherit VersionMixin because ordinary
    application code should treat audit records as append-only rather than
    repeatedly updating the same row.

    Database permissions and migrations should eventually reinforce that
    application-level policy.
    """

    __tablename__ = "audit_events"

    __table_args__ = (
        Index(
            "ix_audit_tenant_occurred",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_audit_category_occurred",
            "category",
            "occurred_at",
        ),
        Index(
            "ix_audit_actor",
            "actor_type",
            "actor_id",
        ),
        Index(
            "ix_audit_resource",
            "resource_id",
            "occurred_at",
        ),
        Index(
            "ix_audit_incident",
            "incident_id",
            "occurred_at",
        ),
        Index(
            "ix_audit_remediation",
            "remediation_id",
            "occurred_at",
        ),
        Index(
            "ix_audit_request",
            "request_id",
        ),
        Index(
            "ix_audit_correlation",
            "correlation_id",
        ),
        Index(
            "ix_audit_outcome",
            "outcome",
            "occurred_at",
        ),
    )

    # ------------------------------------------------------------------
    # Event identity
    # ------------------------------------------------------------------

    event_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    category: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=AuditCategory.SYSTEM.value,
        index=True,
    )

    action: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    outcome: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AuditOutcome.UNKNOWN.value,
        index=True,
    )

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AuditSeverity.INFO.value,
        index=True,
    )

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    # ------------------------------------------------------------------
    # Actor
    # ------------------------------------------------------------------

    actor_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ActorType.UNKNOWN.value,
        index=True,
    )

    actor_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    actor_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    actor_service: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------

    target_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    target_id: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        index=True,
    )

    target_name: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # OSHA entity references
    # ------------------------------------------------------------------

    resource_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    incident_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    anomaly_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    prediction_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    remediation_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------
    # Request / distributed tracing correlation
    # ------------------------------------------------------------------

    request_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    correlation_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    trace_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    span_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Network / client context
    # ------------------------------------------------------------------

    source_ip: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    user_agent: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Operation details
    # ------------------------------------------------------------------

    message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Change tracking
    # ------------------------------------------------------------------

    changed_fields: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    previous_state: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    new_state: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    context: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    policy_context: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_payload: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    # ------------------------------------------------------------------
    # Security indicators
    # ------------------------------------------------------------------

    security_relevant: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    privileged_action: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    # ------------------------------------------------------------------
    # Failure details
    # ------------------------------------------------------------------

    error_code: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Typed helpers
    # ------------------------------------------------------------------

    @property
    def category_kind(self) -> AuditCategory | None:
        """Return the typed event category when recognised."""
        try:
            return AuditCategory(self.category)
        except ValueError:
            return None

    @property
    def action_kind(self) -> AuditAction | None:
        """Return the typed action when recognised."""
        try:
            return AuditAction(self.action)
        except ValueError:
            return None

    @property
    def outcome_kind(self) -> AuditOutcome | None:
        """Return the typed outcome when recognised."""
        try:
            return AuditOutcome(self.outcome)
        except ValueError:
            return None

    @property
    def severity_kind(self) -> AuditSeverity | None:
        """Return the typed severity when recognised."""
        try:
            return AuditSeverity(self.severity)
        except ValueError:
            return None

    @property
    def actor_kind(self) -> ActorType | None:
        """Return the typed actor type when recognised."""
        try:
            return ActorType(self.actor_type)
        except ValueError:
            return None

    @property
    def succeeded(self) -> bool:
        """Return whether the audited operation succeeded."""
        return self.outcome == AuditOutcome.SUCCESS.value

    # ------------------------------------------------------------------
    # Safe serialisation
    # ------------------------------------------------------------------

    def audit_summary(self) -> dict[str, Any]:
        """
        Return a safe serialisable audit-event summary.

        Raw state payloads are intentionally omitted because they may contain
        infrastructure information inappropriate for every consumer.
        """
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "event_name": self.event_name,
            "category": self.category,
            "action": self.action,
            "outcome": self.outcome,
            "severity": self.severity,
            "occurred_at": self.occurred_at.isoformat(),
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "resource_id": (
                str(self.resource_id)
                if self.resource_id is not None
                else None
            ),
            "incident_id": (
                str(self.incident_id)
                if self.incident_id is not None
                else None
            ),
            "remediation_id": (
                str(self.remediation_id)
                if self.remediation_id is not None
                else None
            ),
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "source_ip": self.source_ip,
            "security_relevant": self.security_relevant,
            "privileged_action": self.privileged_action,
            "error_code": self.error_code,
        }


__all__ = [
    "ActorType",
    "AuditAction",
    "AuditCategory",
    "AuditEvent",
    "AuditOutcome",
    "AuditSeverity",
]
