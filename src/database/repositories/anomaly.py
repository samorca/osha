"""
OSHA Anomaly Repository.

Persistence interface for anomalies detected by OSHA.

The repository supports:

    - storing anomaly detections
    - retrieving anomalies by resource
    - querying open anomalies
    - severity/status filtering
    - recurring anomaly detection
    - acknowledgement
    - investigation
    - suppression
    - resolution

Transaction ownership remains with OSHA's session/service layer.
Repositories flush when required but do not commit.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..models.anomaly import (
    Anomaly,
    AnomalySeverity,
    AnomalyStatus,
)
from ..models.resource import Resource
from .base import BaseRepository


class AnomalyRepository(BaseRepository[Anomaly]):
    """Repository for OSHA anomaly records."""

    model = Anomaly

    def __init__(
        self,
        session: Session,
    ) -> None:
        super().__init__(session)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_with_resource(
        self,
        *,
        anomaly_id: uuid.UUID,
    ) -> Anomaly | None:
        """Return an anomaly with its Resource eagerly loaded."""
        statement = (
            select(Anomaly)
            .options(
                selectinload(
                    Anomaly.resource
                )
            )
            .where(
                Anomaly.id == anomaly_id
            )
            .limit(1)
        )

        return self.session.scalar(statement)

    # ------------------------------------------------------------------
    # Resource-specific anomalies
    # ------------------------------------------------------------------

    def list_for_resource(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
        open_only: bool = False,
    ) -> Sequence:
        """Return anomalies associated with an infrastructure resource."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(Anomaly).where(
            Anomaly.tenant_id == tenant_id,
            Anomaly.resource_id == resource_id,
        )

        if open_only:
            statement = statement.where(
                Anomaly.status.in_(
                    (
                        AnomalyStatus.OPEN.value,
                        AnomalyStatus.ACKNOWLEDGED.value,
                        AnomalyStatus.INVESTIGATING.value,
                    )
                )
            )

        statement = (
            statement
            .order_by(
                Anomaly.detected_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Open anomalies
    # ------------------------------------------------------------------

    def list_open(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID | None = None,
        severity: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return operationally open anomalies."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(Anomaly).where(
            Anomaly.tenant_id == tenant_id,
            Anomaly.status.in_(
                (
                    AnomalyStatus.OPEN.value,
                    AnomalyStatus.ACKNOWLEDGED.value,
                    AnomalyStatus.INVESTIGATING.value,
                )
            ),
        )

        if resource_id is not None:
            statement = statement.where(
                Anomaly.resource_id == resource_id
            )

        if severity is not None:
            statement = statement.where(
                Anomaly.severity == severity
            )

        statement = (
            statement
            .options(
                selectinload(
                    Anomaly.resource
                )
            )
            .order_by(
                Anomaly.detected_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Severity queries
    # ------------------------------------------------------------------

    def list_by_severity(
        self,
        *,
        tenant_id: uuid.UUID,
        severity: str,
        offset: int = 0,
        limit: int = 100,
        open_only: bool = False,
    ) -> Sequence:
        """Return anomalies at a particular severity."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(Anomaly).where(
            Anomaly.tenant_id == tenant_id,
            Anomaly.severity == severity,
        )

        if open_only:
            statement = statement.where(
                Anomaly.status.in_(
                    (
                        AnomalyStatus.OPEN.value,
                        AnomalyStatus.ACKNOWLEDGED.value,
                        AnomalyStatus.INVESTIGATING.value,
                    )
                )
            )

        statement = (
            statement
            .order_by(
                Anomaly.detected_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    def list_critical(
        self,
        *,
        tenant_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
        open_only: bool = True,
    ) -> Sequence:
        """Return critical anomalies."""
        return self.list_by_severity(
            tenant_id=tenant_id,
            severity=AnomalySeverity.CRITICAL.value,
            offset=offset,
            limit=limit,
            open_only=open_only,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def list_by_status(
        self,
        *,
        tenant_id: uuid.UUID,
        status: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return anomalies in a particular lifecycle state."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Anomaly)
            .where(
                Anomaly.tenant_id == tenant_id,
                Anomaly.status == status,
            )
            .order_by(
                Anomaly.detected_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Incident / correlation lookups
    # ------------------------------------------------------------------

    def list_for_incident(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return anomalies associated with an incident."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Anomaly)
            .where(
                Anomaly.tenant_id == tenant_id,
                Anomaly.incident_id == incident_id,
            )
            .order_by(
                Anomaly.detected_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    def list_for_correlation(
        self,
        *,
        tenant_id: uuid.UUID,
        correlation_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return anomalies belonging to a correlation group."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Anomaly)
            .where(
                Anomaly.tenant_id == tenant_id,
                Anomaly.correlation_id == correlation_id,
            )
            .order_by(
                Anomaly.detected_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Resource validation
    # ------------------------------------------------------------------

    def _require_resource(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
    ) -> Resource:
        """Validate that the anomaly target belongs to the tenant."""
        statement = (
            select(Resource)
            .where(
                Resource.id == resource_id,
                Resource.tenant_id == tenant_id,
            )
            .limit(1)
        )

        resource = self.session.scalar(statement)

        if resource is None:
            raise ValueError(
                f"resource {resource_id} does not exist "
                "for the specified tenant"
            )

        return resource

    # ------------------------------------------------------------------
    # Detection persistence
    # ------------------------------------------------------------------

    def create_detection(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        anomaly_type: str,
        title: str,
        detected_at: datetime,
        detection_method: str,
        severity: str = AnomalySeverity.MEDIUM.value,
        description: str | None = None,
        metric_name: str | None = None,
        detector_name: str | None = None,
        detector_version: str | None = None,
        model_id: uuid.UUID | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        training_run_id: uuid.UUID | None = None,
        observed_value: float | None = None,
        expected_value: float | None = None,
        baseline_value: float | None = None,
        threshold_value: float | None = None,
        deviation: float | None = None,
        unit: str | None = None,
        score: float | None = None,
        confidence: float | None = None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        correlation_id: uuid.UUID | None = None,
        prediction_id: uuid.UUID | None = None,
        evidence: dict[str, Any] | None = None,
        features: dict[str, Any] | None = None,
        explanation: dict[str, Any] | None = None,
        telemetry_refs: dict[str, Any] | None = None,
        labels: dict[str, Any] | None = None,
        metadata_payload: dict[str, Any] | None = None,
        actionable: bool = True,
    ) -> Anomaly:
        """
        Persist a new anomaly detection.

        The target Resource is verified against the tenant before the
        anomaly is added.

        The Session is flushed but not committed.
        """
        self._require_resource(
            tenant_id=tenant_id,
            resource_id=resource_id,
        )

        if score is not None and not 0.0 <= score <= 1.0:
            raise ValueError(
                "anomaly score must be between 0.0 and 1.0"
            )

        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "anomaly confidence must be between 0.0 and 1.0"
            )

        anomaly = Anomaly(
            tenant_id=tenant_id,
            resource_id=resource_id,
            anomaly_type=anomaly_type,
            title=title,
            description=description,
            metric_name=metric_name,
            detection_method=detection_method,
            detector_name=detector_name,
            detector_version=detector_version,
            model_id=model_id,
            model_name=model_name,
            model_version=model_version,
            training_run_id=training_run_id,
            observed_value=observed_value,
            expected_value=expected_value,
            baseline_value=baseline_value,
            threshold_value=threshold_value,
            deviation=deviation,
            unit=unit,
            score=score,
            confidence=confidence,
            window_start=window_start,
            window_end=window_end,
            detected_at=detected_at,
            first_observed_at=detected_at,
            last_observed_at=detected_at,
            severity=severity,
            status=AnomalyStatus.OPEN.value,
            actionable=actionable,
            correlation_id=correlation_id,
            prediction_id=prediction_id,
            evidence=evidence or {},
            features=features or {},
            explanation=explanation or {},
            telemetry_refs=telemetry_refs or {},
            labels=labels or {},
            metadata_payload=metadata_payload or {},
        )

        self.add(
            anomaly,
            flush=True,
        )

        return anomaly

    # ------------------------------------------------------------------
    # Incident association
    # ------------------------------------------------------------------

    def attach_to_incident(
        self,
        *,
        anomaly_id: uuid.UUID,
        incident_id: uuid.UUID,
    ) -> Anomaly:
        """
        Associate an anomaly with an incident.

        The database model currently stores incident_id as a UUID. Once the
        cross-model foreign-key wiring is completed, PostgreSQL will also
        enforce the incident reference.
        """
        anomaly = self.get_required(
            anomaly_id
        )

        anomaly.incident_id = incident_id

        self.flush()

        return anomaly

    # ------------------------------------------------------------------
    # Acknowledgement
    # ------------------------------------------------------------------

    def acknowledge(
        self,
        *,
        anomaly_id: uuid.UUID,
        acknowledged_by: str,
        acknowledged_at: datetime,
    ) -> Anomaly:
        """Acknowledge an anomaly."""
        anomaly = self.get_required(
            anomaly_id
        )

        anomaly.acknowledge(
            acknowledged_by=acknowledged_by,
            acknowledged_at=acknowledged_at,
        )

        self.flush()

        return anomaly

    # ------------------------------------------------------------------
    # Investigation
    # ------------------------------------------------------------------

    def start_investigation(
        self,
        *,
        anomaly_id: uuid.UUID,
    ) -> Anomaly:
        """Move an anomaly into investigation."""
        anomaly = self.get_required(
            anomaly_id
        )

        anomaly.start_investigation()

        self.flush()

        return anomaly

    # ------------------------------------------------------------------
    # Recurrence
    # ------------------------------------------------------------------

    def mark_recurring(
        self,
        *,
        anomaly_id: uuid.UUID,
        observed_at: datetime,
        observed_value: float | None = None,
        score: float | None = None,
        confidence: float | None = None,
    ) -> Anomaly:
        """Record another occurrence of an existing anomaly."""
        anomaly = self.get_required(
            anomaly_id
        )

        anomaly.mark_recurring(
            observed_at=observed_at
        )

        if observed_value is not None:
            anomaly.observed_value = observed_value

        if score is not None:
            anomaly.set_score(score)

        if confidence is not None:
            anomaly.set_confidence(confidence)

        self.flush()

        return anomaly

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        *,
        anomaly_id: uuid.UUID,
        resolved_at: datetime,
        notes: str | None = None,
    ) -> Anomaly:
        """Resolve an anomaly."""
        anomaly = self.get_required(
            anomaly_id
        )

        anomaly.resolve(
            resolved_at=resolved_at,
            notes=notes,
        )

        self.flush()

        return anomaly

    def suppress(
        self,
        *,
        anomaly_id: uuid.UUID,
        notes: str | None = None,
    ) -> Anomaly:
        """Suppress an anomaly."""
        anomaly = self.get_required(
            anomaly_id
        )

        anomaly.suppress(
            notes=notes
        )

        self.flush()

        return anomaly


__all__ = [
    "AnomalyRepository",
]
