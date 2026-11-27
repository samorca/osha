"""
OSHA Remediation Persistence Model.

Stores remediation decisions and execution state for infrastructure managed
by OSHA.

A remediation is not simply a command. It represents the complete lifecycle:

    proposed
        -> awaiting approval
        -> approved
        -> executing
        -> succeeded / failed
        -> optionally rolled back

The persistence model intentionally separates remediation intent from the
underlying executor implementation.

Examples:

    - restart a failed service
    - restart or replace a workload
    - scale a Kubernetes workload
    - cordon/drain a node through an authorised executor
    - recover a failed component
    - apply a safe configuration adjustment
    - execute a pre-approved operational runbook

The model preserves safety, approval, execution and rollback information so
OSHA does not treat autonomous infrastructure changes as ordinary database
updates.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    false,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base, TenantMixin, VersionMixin


# ---------------------------------------------------------------------------
# Remediation classification
# ---------------------------------------------------------------------------

class RemediationType(str, enum.Enum):
    """Canonical remediation categories."""

    RESTART = "restart"
    REDEPLOY = "redeploy"

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"

    RECONFIGURE = "reconfigure"

    FAILOVER = "failover"
    RECOVER = "recover"

    ISOLATE = "isolate"

    CORDON = "cordon"
    DRAIN = "drain"

    CLEANUP = "cleanup"
    REBALANCE = "rebalance"

    RUNBOOK = "runbook"

    CUSTOM = "custom"


class RemediationStatus(str, enum.Enum):
    """Lifecycle of a remediation."""

    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"

    APPROVED = "approved"
    REJECTED = "rejected"

    SCHEDULED = "scheduled"

    EXECUTING = "executing"

    SUCCEEDED = "succeeded"
    FAILED = "failed"

    CANCELLED = "cancelled"

    ROLLBACK_PENDING = "rollback_pending"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"


class ApprovalMode(str, enum.Enum):
    """Approval requirements for remediation execution."""

    MANUAL = "manual"
    POLICY = "policy"
    AUTOMATIC = "automatic"


class RemediationRisk(str, enum.Enum):
    """Estimated operational risk of a remediation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionMode(str, enum.Enum):
    """How the remediation is intended to execute."""

    MANUAL = "manual"
    AUTOMATIC = "automatic"
    SCHEDULED = "scheduled"


class RemediationOutcome(str, enum.Enum):
    """Observed outcome after remediation execution."""

    UNKNOWN = "unknown"

    RESOLVED = "resolved"
    IMPROVED = "improved"
    NO_CHANGE = "no_change"
    DEGRADED = "degraded"

    FAILED = "failed"

    ROLLED_BACK = "rolled_back"


# ---------------------------------------------------------------------------
# Remediation model
# ---------------------------------------------------------------------------

class Remediation(
    TenantMixin,
    VersionMixin,
    Base,
):
    """
    Persisted remediation plan and execution state.

    Every remediation should be traceable to:

        why OSHA proposed it
        what resource it targets
        what action is intended
        who/what approved it
        when execution occurred
        what executor performed it
        whether it succeeded
        whether rollback was required
        what the resulting infrastructure state was

    Execution code itself belongs in OSHA's remediation/executor subsystem,
    not inside this ORM model.
    """

    __tablename__ = "remediations"

    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR "
            "(confidence >= 0.0 AND confidence <= 1.0)",
            name="remediation_confidence_range",
        ),
        CheckConstraint(
            "max_attempts >= 1",
            name="remediation_max_attempts_positive",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="remediation_attempt_count_non_negative",
        ),
        Index(
            "ix_remediations_resource_created",
            "resource_id",
            "created_at",
        ),
        Index(
            "ix_remediations_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_remediations_tenant_risk",
            "tenant_id",
            "risk",
        ),
        Index(
            "ix_remediations_incident",
            "incident_id",
        ),
        Index(
            "ix_remediations_anomaly",
            "anomaly_id",
        ),
        Index(
            "ix_remediations_prediction",
            "prediction_id",
        ),
        Index(
            "ix_remediations_scheduled",
            "scheduled_at",
        ),
    )

    # ------------------------------------------------------------------
    # Target resource
    # ------------------------------------------------------------------

    resource_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            "resources.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    resource: Mapped["Resource"] = relationship(
        "Resource",
        foreign_keys=[resource_id],
        lazy="selectin",
    )

    # ------------------------------------------------------------------
    # Operational references
    # ------------------------------------------------------------------

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

    correlation_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------
    # Remediation identity
    # ------------------------------------------------------------------

    remediation_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RemediationStatus.PROPOSED.value,
        index=True,
    )

    execution_mode: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ExecutionMode.MANUAL.value,
        index=True,
    )

    outcome: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RemediationOutcome.UNKNOWN.value,
        index=True,
    )

    # ------------------------------------------------------------------
    # Risk and confidence
    # ------------------------------------------------------------------

    risk: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RemediationRisk.MEDIUM.value,
        index=True,
    )

    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Approval controls
    # ------------------------------------------------------------------

    approval_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    approval_mode: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ApprovalMode.MANUAL.value,
    )

    approved: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    approved_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    approval_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    rejected_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    rejected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    rejection_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Action specification
    # ------------------------------------------------------------------

    action_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    executor_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    action_parameters: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    preconditions: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    safety_constraints: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    expected_result: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    # ------------------------------------------------------------------
    # Execution timing
    # ------------------------------------------------------------------

    proposed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Retry / execution state
    # ------------------------------------------------------------------

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    max_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    execution_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    executor_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Execution result
    # ------------------------------------------------------------------

    result: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    error_code: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    rollback_supported: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    rollback_action: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    rollback_parameters: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    rollback_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    rollback_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    rollback_result: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    # ------------------------------------------------------------------
    # Evidence / provenance
    # ------------------------------------------------------------------

    evidence: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    decision_context: Mapped[dict[str, Any]] = mapped_column(
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

    api_managed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
        index=True,
    )

    api_idempotency_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
    )

    api_decision_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    api_execution_idempotency_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
    )

    # ------------------------------------------------------------------
    # Typed helpers
    # ------------------------------------------------------------------

    @property
    def kind(self) -> RemediationType | None:
        """Return the typed remediation type."""
        try:
            return RemediationType(self.remediation_type)
        except ValueError:
            return None

    @property
    def lifecycle(self) -> RemediationStatus | None:
        """Return the typed remediation lifecycle state."""
        try:
            return RemediationStatus(self.status)
        except ValueError:
            return None

    @property
    def risk_kind(self) -> RemediationRisk | None:
        """Return the typed remediation risk."""
        try:
            return RemediationRisk(self.risk)
        except ValueError:
            return None

    @property
    def outcome_kind(self) -> RemediationOutcome | None:
        """Return the typed remediation outcome."""
        try:
            return RemediationOutcome(self.outcome)
        except ValueError:
            return None

    @property
    def requires_human_approval(self) -> bool:
        """Return whether manual approval is required."""
        return (
            self.approval_required
            and self.approval_mode == ApprovalMode.MANUAL.value
        )

    @property
    def can_execute(self) -> bool:
        """
        Return whether the persisted remediation state permits execution.

        The executor must still evaluate preconditions, permissions and safety
        policy immediately before making an infrastructure change.
        """
        if self.status not in {
            RemediationStatus.APPROVED.value,
            RemediationStatus.SCHEDULED.value,
        }:
            return False

        if self.approval_required and not self.approved:
            return False

        return True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def set_confidence(
        self,
        confidence: float | None,
    ) -> None:
        """Set recommendation confidence after range validation."""
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "remediation confidence must be between 0.0 and 1.0"
            )

        self.confidence = confidence

    # ------------------------------------------------------------------
    # Approval state transitions
    # ------------------------------------------------------------------

    def request_approval(self) -> None:
        """Move the remediation into the approval queue."""
        self.status = RemediationStatus.PENDING_APPROVAL.value

    def approve(
        self,
        *,
        approved_by: str,
        approved_at: datetime,
        reason: str | None = None,
    ) -> None:
        """Record approval for remediation execution."""
        self.approved = True
        self.approved_by = approved_by
        self.approved_at = approved_at
        self.approval_reason = reason

        self.rejected_by = None
        self.rejected_at = None
        self.rejection_reason = None

        self.status = RemediationStatus.APPROVED.value

    def reject(
        self,
        *,
        rejected_by: str,
        rejected_at: datetime,
        reason: str | None = None,
    ) -> None:
        """Reject the remediation."""
        self.approved = False
        self.rejected_by = rejected_by
        self.rejected_at = rejected_at
        self.rejection_reason = reason

        self.status = RemediationStatus.REJECTED.value

    # ------------------------------------------------------------------
    # Execution state transitions
    # ------------------------------------------------------------------

    def mark_executing(
        self,
        *,
        started_at: datetime,
        execution_id: uuid.UUID,
        executor_id: str | None = None,
    ) -> None:
        """
        Record the start of execution.

        This does not execute the remediation itself.
        """
        if not self.can_execute:
            raise ValueError(
                "remediation is not in an executable state"
            )

        if self.attempt_count >= self.max_attempts:
            raise ValueError(
                "remediation maximum execution attempts reached"
            )

        self.status = RemediationStatus.EXECUTING.value
        self.started_at = started_at
        self.execution_id = execution_id
        self.executor_id = executor_id
        self.attempt_count += 1

    def mark_succeeded(
        self,
        *,
        completed_at: datetime,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Record successful remediation execution."""
        self.status = RemediationStatus.SUCCEEDED.value
        self.outcome = RemediationOutcome.UNKNOWN.value
        self.completed_at = completed_at

        if result is not None:
            self.result = result

        self.error_code = None
        self.error_message = None

    def mark_failed(
        self,
        *,
        completed_at: datetime,
        error_message: str,
        error_code: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Record failed remediation execution."""
        self.status = RemediationStatus.FAILED.value
        self.outcome = RemediationOutcome.FAILED.value
        self.completed_at = completed_at
        self.error_code = error_code
        self.error_message = error_message

        if result is not None:
            self.result = result

    # ------------------------------------------------------------------
    # Outcome evaluation
    # ------------------------------------------------------------------

    def set_outcome(
        self,
        outcome: RemediationOutcome,
    ) -> None:
        """Record the observed infrastructure outcome."""
        self.outcome = outcome.value

    # ------------------------------------------------------------------
    # Rollback lifecycle
    # ------------------------------------------------------------------

    def request_rollback(self) -> None:
        """Request rollback of a completed or failed remediation."""
        if not self.rollback_supported:
            raise ValueError(
                "remediation does not support rollback"
            )

        self.status = RemediationStatus.ROLLBACK_PENDING.value

    def mark_rollback_started(
        self,
        *,
        started_at: datetime,
    ) -> None:
        """Record rollback execution start."""
        if not self.rollback_supported:
            raise ValueError(
                "remediation does not support rollback"
            )

        self.status = RemediationStatus.ROLLING_BACK.value
        self.rollback_started_at = started_at

    def mark_rolled_back(
        self,
        *,
        completed_at: datetime,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Record successful rollback."""
        self.status = RemediationStatus.ROLLED_BACK.value
        self.outcome = RemediationOutcome.ROLLED_BACK.value
        self.rollback_completed_at = completed_at

        if result is not None:
            self.rollback_result = result

    def mark_rollback_failed(
        self,
        *,
        completed_at: datetime,
        error_message: str,
    ) -> None:
        """Record a failed rollback attempt."""
        self.status = RemediationStatus.ROLLBACK_FAILED.value
        self.rollback_completed_at = completed_at
        self.error_message = error_message

    # ------------------------------------------------------------------
    # Safe representation
    # ------------------------------------------------------------------

    def remediation_summary(self) -> dict[str, Any]:
        """Return safe operational remediation information."""
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "resource_id": str(self.resource_id),
            "incident_id": (
                str(self.incident_id)
                if self.incident_id is not None
                else None
            ),
            "remediation_type": self.remediation_type,
            "title": self.title,
            "action_name": self.action_name,
            "status": self.status,
            "execution_mode": self.execution_mode,
            "risk": self.risk,
            "confidence": self.confidence,
            "approval_required": self.approval_required,
            "approved": self.approved,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "outcome": self.outcome,
            "proposed_at": self.proposed_at.isoformat(),
            "started_at": (
                self.started_at.isoformat()
                if self.started_at is not None
                else None
            ),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at is not None
                else None
            ),
        }


__all__ = [
    "ApprovalMode",
    "ExecutionMode",
    "Remediation",
    "RemediationOutcome",
    "RemediationRisk",
    "RemediationStatus",
    "RemediationType",
]
