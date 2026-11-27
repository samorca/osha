"""
OSHA Anomaly Persistence Model.

An Anomaly represents abnormal behaviour detected in infrastructure
monitored by OSHA.

Examples:

    - unexpected CPU spike
    - abnormal memory growth
    - unusual disk latency
    - network traffic deviation
    - abnormal Kubernetes pod restart behaviour
    - service latency deviation
    - resource behaviour inconsistent with learned baseline

An anomaly is an observation/detection, not automatically an incident.

Multiple anomalies and events may later be correlated into a single
Incident by OSHA's correlation and intelligence layers.
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
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base, TenantMixin, VersionMixin


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class AnomalyType(str, enum.Enum):
    """Canonical categories of abnormal infrastructure behaviour."""

    METRIC = "metric"
    LOG = "log"
    EVENT = "event"
    TRACE = "trace"

    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"

    AVAILABILITY = "availability"
    RESOURCE_USAGE = "resource_usage"
    CAPACITY = "capacity"

    PROCESS = "process"
    SERVICE = "service"
    CONTAINER = "container"
    KUBERNETES = "kubernetes"

    BEHAVIOURAL = "behavioural"
    STATISTICAL = "statistical"
    MODEL = "model"

    SECURITY = "security"

    CUSTOM = "custom"


class AnomalySeverity(str, enum.Enum):
    """Operational severity assigned to an anomaly."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyStatus(str, enum.Enum):
    """Lifecycle state of an anomaly."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"


class DetectionMethod(str, enum.Enum):
    """How OSHA detected the anomaly."""

    THRESHOLD = "threshold"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    RULE = "rule"
    CORRELATION = "correlation"
    EXTERNAL = "external"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Anomaly model
# ---------------------------------------------------------------------------

class Anomaly(
    TenantMixin,
    VersionMixin,
    Base,
):
    """
    Abnormal condition detected on an OSHA Resource.

    Resource identity is kept separate from the anomaly itself.

    Example:

        Resource
            kubernetes pod/payment-api-7f58

        Anomaly
            memory usage deviated significantly from baseline

    Evidence, model provenance and observed values are retained so anomaly
    decisions remain explainable and auditable.
    """

    __tablename__ = "anomalies"

    __table_args__ = (
        CheckConstraint(
            "score IS NULL OR (score >= 0.0 AND score <= 1.0)",
            name="anomaly_score_range",
        ),
        CheckConstraint(
            "confidence IS NULL OR "
            "(confidence >= 0.0 AND confidence <= 1.0)",
            name="anomaly_confidence_range",
        ),
        Index(
            "ix_anomalies_resource_detected",
            "resource_id",
            "detected_at",
        ),
        Index(
            "ix_anomalies_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_anomalies_tenant_severity",
            "tenant_id",
            "severity",
        ),
        Index(
            "ix_anomalies_type_status",
            "anomaly_type",
            "status",
        ),
        Index(
            "ix_anomalies_model",
            "model_name",
            "model_version",
        ),
        Index(
            "ix_anomalies_incident",
            "incident_id",
        ),
        Index(
            "ix_anomalies_correlation",
            "correlation_id",
        ),
    )

    # ------------------------------------------------------------------
    # Target resource
    # ------------------------------------------------------------------

    resource_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "resources.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    resource: Mapped["Resource"] = relationship(
        "Resource",
        foreign_keys=[resource_id],
        lazy="selectin",
    )

    # ------------------------------------------------------------------
    # Optional operational references
    # ------------------------------------------------------------------

    incident_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    correlation_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    prediction_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    anomaly_type: Mapped[str] = mapped_column(
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

    metric_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    detection_method: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=DetectionMethod.MACHINE_LEARNING.value,
        index=True,
    )

    detector_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    detector_version: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Model provenance
    # ------------------------------------------------------------------

    model_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    model_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    model_version: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    training_run_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------
    # Detection values
    # ------------------------------------------------------------------

    observed_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    expected_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    baseline_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    threshold_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    deviation: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    unit: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Time window
    # ------------------------------------------------------------------

    window_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    window_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    first_observed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_observed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Operational classification
    # ------------------------------------------------------------------

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AnomalySeverity.MEDIUM.value,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AnomalyStatus.OPEN.value,
        index=True,
    )

    actionable: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    recurring: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    occurrence_count: Mapped[int] = mapped_column(
        nullable=False,
        default=1,
    )

    # ------------------------------------------------------------------
    # Explainability and evidence
    # ------------------------------------------------------------------

    evidence: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    features: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    explanation: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    telemetry_refs: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    labels: Mapped[dict[str, Any]] = mapped_column(
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
    # Human acknowledgement
    # ------------------------------------------------------------------

    acknowledged_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    resolution_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Typed helpers
    # ------------------------------------------------------------------

    @property
    def kind(self) -> AnomalyType | None:
        """Return typed anomaly classification when recognised."""
        try:
            return AnomalyType(self.anomaly_type)
        except ValueError:
            return None

    @property
    def severity_kind(self) -> AnomalySeverity | None:
        """Return typed anomaly severity."""
        try:
            return AnomalySeverity(self.severity)
        except ValueError:
            return None

    @property
    def lifecycle(self) -> AnomalyStatus | None:
        """Return typed anomaly lifecycle state."""
        try:
            return AnomalyStatus(self.status)
        except ValueError:
            return None

    @property
    def detector_kind(self) -> DetectionMethod | None:
        """Return typed detection method."""
        try:
            return DetectionMethod(self.detection_method)
        except ValueError:
            return None

    @property
    def is_open(self) -> bool:
        """Return whether the anomaly remains operationally open."""
        return self.status in {
            AnomalyStatus.OPEN.value,
            AnomalyStatus.ACKNOWLEDGED.value,
            AnomalyStatus.INVESTIGATING.value,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def set_score(
        self,
        score: float | None,
    ) -> None:
        """Set anomaly score after validating its range."""
        if score is not None and not 0.0 <= score <= 1.0:
            raise ValueError(
                "anomaly score must be between 0.0 and 1.0"
            )

        self.score = score

    def set_confidence(
        self,
        confidence: float | None,
    ) -> None:
        """Set detection confidence after validating its range."""
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "anomaly confidence must be between 0.0 and 1.0"
            )

        self.confidence = confidence

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def acknowledge(
        self,
        *,
        acknowledged_by: str,
        acknowledged_at: datetime,
    ) -> None:
        """Acknowledge the anomaly."""
        self.status = AnomalyStatus.ACKNOWLEDGED.value
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = acknowledged_at

    def start_investigation(self) -> None:
        """Mark the anomaly as under investigation."""
        self.status = AnomalyStatus.INVESTIGATING.value

    def resolve(
        self,
        *,
        resolved_at: datetime,
        notes: str | None = None,
    ) -> None:
        """Resolve the anomaly while preserving its history."""
        self.status = AnomalyStatus.RESOLVED.value
        self.resolved_at = resolved_at
        self.resolution_notes = notes
        self.actionable = False

    def suppress(
        self,
        *,
        notes: str | None = None,
    ) -> None:
        """Suppress further operational action for the anomaly."""
        self.status = AnomalyStatus.SUPPRESSED.value
        self.actionable = False

        if notes is not None:
            self.resolution_notes = notes

    def mark_recurring(
        self,
        *,
        observed_at: datetime,
    ) -> None:
        """Record another occurrence of the anomaly."""
        self.recurring = True
        self.occurrence_count += 1
        self.last_observed_at = observed_at

    # ------------------------------------------------------------------
    # Safe representation
    # ------------------------------------------------------------------

    def anomaly_summary(self) -> dict[str, Any]:
        """Return a serialisable operational anomaly summary."""
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "resource_id": str(self.resource_id),
            "incident_id": (
                str(self.incident_id)
                if self.incident_id is not None
                else None
            ),
            "anomaly_type": self.anomaly_type,
            "title": self.title,
            "metric_name": self.metric_name,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "threshold_value": self.threshold_value,
            "unit": self.unit,
            "score": self.score,
            "confidence": self.confidence,
            "severity": self.severity,
            "status": self.status,
            "detection_method": self.detection_method,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "detected_at": self.detected_at.isoformat(),
            "actionable": self.actionable,
        }


__all__ = [
    "Anomaly",
    "AnomalySeverity",
    "AnomalyStatus",
    "AnomalyType",
    "DetectionMethod",
]
