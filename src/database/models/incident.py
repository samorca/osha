"""
OSHA Incident Persistence Model.

An Incident represents an operational situation that OSHA has determined
requires investigation, tracking, remediation or human attention.

An incident can correlate:

    resources
        -> anomalies
        -> predictions
        -> events
        -> topology evidence
        -> root-cause hypotheses
        -> remediation actions
        -> final outcome

Anomaly != Incident.

An anomaly is an abnormal observation. Multiple anomalies may contribute to
a single incident.

An incident therefore acts as the durable operational record of a problem
from detection through investigation, remediation and resolution.
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
# Incident classification
# ---------------------------------------------------------------------------

class IncidentType(str, enum.Enum):
    """Canonical categories of OSHA incidents."""

    AVAILABILITY = "availability"
    PERFORMANCE = "performance"
    CAPACITY = "capacity"

    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"

    SERVICE_FAILURE = "service_failure"
    HOST_FAILURE = "host_failure"
    CONTAINER_FAILURE = "container_failure"
    KUBERNETES = "kubernetes"

    STORAGE = "storage"
    DATABASE = "database"

    SECURITY = "security"

    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"

    ANOMALY = "anomaly"
    PREDICTION = "prediction"

    UNKNOWN = "unknown"
    CUSTOM = "custom"


class IncidentSeverity(str, enum.Enum):
    """Operational severity of an incident."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentPriority(str, enum.Enum):
    """Operational priority for response."""

    P1 = "p1"
    P2 = "p2"
    P3 = "p3"
    P4 = "p4"
    P5 = "p5"


class IncidentStatus(str, enum.Enum):
    """Lifecycle state of an incident."""

    DETECTED = "detected"
    OPEN = "open"

    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"

    MITIGATING = "mitigating"
    MONITORING = "monitoring"

    RESOLVED = "resolved"
    CLOSED = "closed"

    SUPPRESSED = "suppressed"


class IncidentSource(str, enum.Enum):
    """How OSHA created the incident."""

    ANOMALY = "anomaly"
    PREDICTION = "prediction"
    CORRELATION = "correlation"
    ALERT = "alert"
    EVENT = "event"
    USER = "user"
    EXTERNAL = "external"
    AUTOMATION = "automation"


class RootCauseStatus(str, enum.Enum):
    """Confidence lifecycle of incident root-cause analysis."""

    UNKNOWN = "unknown"
    INVESTIGATING = "investigating"
    SUSPECTED = "suspected"
    PROBABLE = "probable"
    CONFIRMED = "confirmed"


# ---------------------------------------------------------------------------
# Incident model
# ---------------------------------------------------------------------------

class Incident(
    TenantMixin,
    VersionMixin,
    Base,
):
    """
    Durable operational incident managed by OSHA.

    The primary resource represents the infrastructure entity most directly
    associated with the incident.

    Other affected resources can later be represented through dedicated
    incident/resource association records or topology queries rather than
    duplicating infrastructure data in this row.
    """

    __tablename__ = "incidents"

    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR "
            "(confidence >= 0.0 AND confidence <= 1.0)",
            name="incident_confidence_range",
        ),
        CheckConstraint(
            "root_cause_confidence IS NULL OR "
            "(root_cause_confidence >= 0.0 "
            "AND root_cause_confidence <= 1.0)",
            name="incident_root_cause_confidence_range",
        ),
        CheckConstraint(
            "occurrence_count >= 1",
            name="incident_occurrence_count_positive",
        ),
        Index(
            "ix_incidents_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_incidents_tenant_severity",
            "tenant_id",
            "severity",
        ),
        Index(
            "ix_incidents_tenant_priority",
            "tenant_id",
            "priority",
        ),
        Index(
            "ix_incidents_resource_status",
            "primary_resource_id",
            "status",
        ),
        Index(
            "ix_incidents_detected",
            "detected_at",
        ),
        Index(
            "ix_incidents_correlation",
            "correlation_id",
        ),
    )

    # ------------------------------------------------------------------
    # Primary resource
    # ------------------------------------------------------------------

    primary_resource_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            "resources.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    primary_resource: Mapped["Resource | None"] = relationship(
        "Resource",
        foreign_keys=[primary_resource_id],
        lazy="selectin",
    )

    # ------------------------------------------------------------------
    # Incident identity
    # ------------------------------------------------------------------

    incident_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default=IncidentType.UNKNOWN.value,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Detection / correlation
    # ------------------------------------------------------------------

    source: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=IncidentSource.CORRELATION.value,
        index=True,
    )

    correlation_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    trigger_anomaly_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    trigger_prediction_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Operational classification
    # ------------------------------------------------------------------

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=IncidentSeverity.MEDIUM.value,
        index=True,
    )

    priority: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=IncidentPriority.P3.value,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=IncidentStatus.DETECTED.value,
        index=True,
    )

    actionable: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    automated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ------------------------------------------------------------------
    # Incident timing
    # ------------------------------------------------------------------

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    investigation_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    mitigation_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_activity_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------
    # Recurrence
    # ------------------------------------------------------------------

    recurring: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    occurrence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    # ------------------------------------------------------------------
    # Ownership / acknowledgement
    # ------------------------------------------------------------------

    acknowledged_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    assigned_to: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    owner_team: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------
    # Impact
    # ------------------------------------------------------------------

    impact_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    affected_resource_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    affected_services: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    impact_data: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    # ------------------------------------------------------------------
    # Root-cause analysis
    # ------------------------------------------------------------------

    root_cause_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RootCauseStatus.UNKNOWN.value,
        index=True,
    )

    root_cause_resource_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            "resources.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    root_cause_resource: Mapped["Resource | None"] = relationship(
        "Resource",
        foreign_keys=[root_cause_resource_id],
        lazy="selectin",
    )

    suspected_root_cause: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    confirmed_root_cause: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    root_cause_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    root_cause_evidence: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    # ------------------------------------------------------------------
    # Correlation / evidence
    # ------------------------------------------------------------------

    evidence: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    correlated_events: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    telemetry_refs: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    topology_context: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    resolution_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    resolution_code: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    resolved_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Learning / post-incident information
    # ------------------------------------------------------------------

    lessons_learned: Mapped[dict[str, Any]] = mapped_column(
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

    # ------------------------------------------------------------------
    # Typed helpers
    # ------------------------------------------------------------------

    @property
    def kind(self) -> IncidentType | None:
        """Return typed incident classification when recognised."""
        try:
            return IncidentType(self.incident_type)
        except ValueError:
            return None

    @property
    def severity_kind(self) -> IncidentSeverity | None:
        """Return typed incident severity."""
        try:
            return IncidentSeverity(self.severity)
        except ValueError:
            return None

    @property
    def priority_kind(self) -> IncidentPriority | None:
        """Return typed incident priority."""
        try:
            return IncidentPriority(self.priority)
        except ValueError:
            return None

    @property
    def lifecycle(self) -> IncidentStatus | None:
        """Return typed incident lifecycle state."""
        try:
            return IncidentStatus(self.status)
        except ValueError:
            return None

    @property
    def root_cause_state(self) -> RootCauseStatus | None:
        """Return typed root-cause state."""
        try:
            return RootCauseStatus(self.root_cause_status)
        except ValueError:
            return None

    @property
    def is_open(self) -> bool:
        """Return whether the incident remains operationally active."""
        return self.status in {
            IncidentStatus.DETECTED.value,
            IncidentStatus.OPEN.value,
            IncidentStatus.ACKNOWLEDGED.value,
            IncidentStatus.INVESTIGATING.value,
            IncidentStatus.MITIGATING.value,
            IncidentStatus.MONITORING.value,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def set_confidence(
        self,
        confidence: float | None,
    ) -> None:
        """Set incident confidence with range validation."""
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "incident confidence must be between 0.0 and 1.0"
            )

        self.confidence = confidence

    def set_root_cause_confidence(
        self,
        confidence: float | None,
    ) -> None:
        """Set root-cause confidence with range validation."""
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "root-cause confidence must be between 0.0 and 1.0"
            )

        self.root_cause_confidence = confidence

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open a detected incident."""
        self.status = IncidentStatus.OPEN.value

    def acknowledge(
        self,
        *,
        acknowledged_by: str,
        acknowledged_at: datetime,
    ) -> None:
        """Acknowledge the incident."""
        self.status = IncidentStatus.ACKNOWLEDGED.value
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = acknowledged_at
        self.last_activity_at = acknowledged_at

    def start_investigation(
        self,
        *,
        started_at: datetime,
    ) -> None:
        """Move the incident into investigation."""
        self.status = IncidentStatus.INVESTIGATING.value
        self.investigation_started_at = started_at
        self.last_activity_at = started_at

    def start_mitigation(
        self,
        *,
        started_at: datetime,
    ) -> None:
        """Move the incident into mitigation."""
        self.status = IncidentStatus.MITIGATING.value
        self.mitigation_started_at = started_at
        self.last_activity_at = started_at

    def start_monitoring(self) -> None:
        """Move the incident into post-mitigation monitoring."""
        self.status = IncidentStatus.MONITORING.value

    def resolve(
        self,
        *,
        resolved_at: datetime,
        summary: str,
        resolved_by: str | None = None,
        resolution_code: str | None = None,
    ) -> None:
        """Resolve the incident while preserving its history."""
        self.status = IncidentStatus.RESOLVED.value
        self.resolved_at = resolved_at
        self.last_activity_at = resolved_at

        self.resolution_summary = summary
        self.resolved_by = resolved_by
        self.resolution_code = resolution_code

        self.actionable = False

    def close(
        self,
        *,
        closed_at: datetime,
    ) -> None:
        """Close a resolved incident."""
        if self.status != IncidentStatus.RESOLVED.value:
            raise ValueError(
                "incident must be resolved before it can be closed"
            )

        self.status = IncidentStatus.CLOSED.value
        self.closed_at = closed_at
        self.last_activity_at = closed_at
        self.actionable = False

    # ------------------------------------------------------------------
    # Root-cause operations
    # ------------------------------------------------------------------

    def set_suspected_root_cause(
        self,
        *,
        cause: str,
        confidence: float | None = None,
        resource_id: uuid.UUID | None = None,
    ) -> None:
        """Record a suspected root cause."""
        self.root_cause_status = RootCauseStatus.SUSPECTED.value
        self.suspected_root_cause = cause

        if resource_id is not None:
            self.root_cause_resource_id = resource_id

        self.set_root_cause_confidence(confidence)

    def confirm_root_cause(
        self,
        *,
        cause: str,
        confidence: float | None = None,
        resource_id: uuid.UUID | None = None,
    ) -> None:
        """Record the confirmed root cause."""
        self.root_cause_status = RootCauseStatus.CONFIRMED.value
        self.confirmed_root_cause = cause

        if resource_id is not None:
            self.root_cause_resource_id = resource_id

        self.set_root_cause_confidence(confidence)

    # ------------------------------------------------------------------
    # Recurrence
    # ------------------------------------------------------------------

    def mark_recurring(
        self,
        *,
        observed_at: datetime,
    ) -> None:
        """Record another occurrence of the incident."""
        self.recurring = True
        self.occurrence_count += 1
        self.last_activity_at = observed_at

    # ------------------------------------------------------------------
    # Safe representation
    # ------------------------------------------------------------------

    def incident_summary(self) -> dict[str, Any]:
        """Return a serialisable operational incident summary."""
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "primary_resource_id": (
                str(self.primary_resource_id)
                if self.primary_resource_id is not None
                else None
            ),
            "incident_type": self.incident_type,
            "title": self.title,
            "severity": self.severity,
            "priority": self.priority,
            "status": self.status,
            "source": self.source,
            "confidence": self.confidence,
            "root_cause_status": self.root_cause_status,
            "root_cause_resource_id": (
                str(self.root_cause_resource_id)
                if self.root_cause_resource_id is not None
                else None
            ),
            "actionable": self.actionable,
            "recurring": self.recurring,
            "occurrence_count": self.occurrence_count,
            "detected_at": self.detected_at.isoformat(),
            "resolved_at": (
                self.resolved_at.isoformat()
                if self.resolved_at is not None
                else None
            ),
        }


__all__ = [
    "Incident",
    "IncidentPriority",
    "IncidentSeverity",
    "IncidentSource",
    "IncidentStatus",
    "IncidentType",
    "RootCauseStatus",
]
