"""
OSHA Incident Repository.

Persistence interface for operational incidents.

An incident represents a correlated operational problem rather than a single
raw observation.

Typical lifecycle:

    detection
        -> open
        -> acknowledgement
        -> investigation
        -> root-cause analysis
        -> mitigation
        -> monitoring
        -> resolution
        -> closure

The repository does not commit transactions. Transaction ownership belongs
to OSHA's service/session layer.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..models.incident import (
    Incident,
    IncidentPriority,
    IncidentSeverity,
    IncidentSource,
    IncidentStatus,
    IncidentType,
)
from ..models.resource import Resource
from .base import BaseRepository


class IncidentRepository(BaseRepository[Incident]):
    """Repository for OSHA operational incidents."""

    model = Incident

    def __init__(
        self,
        session: Session,
    ) -> None:
        super().__init__(session)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_with_resources(
        self,
        *,
        incident_id: uuid.UUID,
    ) -> Incident | None:
        """
        Return an incident with primary and root-cause resources loaded.
        """
        statement = (
            select(Incident)
            .options(
                selectinload(
                    Incident.primary_resource
                ),
                selectinload(
                    Incident.root_cause_resource
                ),
            )
            .where(
                Incident.id == incident_id
            )
            .limit(1)
        )

        return self.session.scalar(statement)

    # ------------------------------------------------------------------
    # Tenant incident listing
    # ------------------------------------------------------------------

    def list_for_tenant(
        self,
        *,
        tenant_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Incident]:
        """
        Return incidents belonging to a tenant.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Incident)
            .where(
                Incident.tenant_id == tenant_id
            )
            .order_by(
                Incident.detected_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Open incidents
    # ------------------------------------------------------------------

    def list_open(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID | None = None,
        severity: str | None = None,
        priority: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Incident]:
        """
        Return operationally active incidents.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        open_statuses = (
            IncidentStatus.DETECTED.value,
            IncidentStatus.OPEN.value,
            IncidentStatus.ACKNOWLEDGED.value,
            IncidentStatus.INVESTIGATING.value,
            IncidentStatus.MITIGATING.value,
            IncidentStatus.MONITORING.value,
        )

        statement = select(Incident).where(
            Incident.tenant_id == tenant_id,
            Incident.status.in_(open_statuses),
        )

        if resource_id is not None:
            statement = statement.where(
                Incident.primary_resource_id == resource_id
            )

        if severity is not None:
            statement = statement.where(
                Incident.severity == severity
            )

        if priority is not None:
            statement = statement.where(
                Incident.priority == priority
            )

        statement = (
            statement
            .options(
                selectinload(
                    Incident.primary_resource
                )
            )
            .order_by(
                Incident.detected_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Resource incidents
    # ------------------------------------------------------------------

    def list_for_resource(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
        open_only: bool = False,
    ) -> Sequence[Incident]:
        """
        Return incidents whose primary resource matches resource_id.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(Incident).where(
            Incident.tenant_id == tenant_id,
            Incident.primary_resource_id == resource_id,
        )

        if open_only:
            statement = statement.where(
                Incident.status.in_(
                    (
                        IncidentStatus.DETECTED.value,
                        IncidentStatus.OPEN.value,
                        IncidentStatus.ACKNOWLEDGED.value,
                        IncidentStatus.INVESTIGATING.value,
                        IncidentStatus.MITIGATING.value,
                        IncidentStatus.MONITORING.value,
                    )
                )
            )

        statement = (
            statement
            .order_by(
                Incident.detected_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
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
    ) -> Sequence[Incident]:
        """Return incidents in a specific lifecycle state."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Incident)
            .where(
                Incident.tenant_id == tenant_id,
                Incident.status == status,
            )
            .order_by(
                Incident.detected_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Severity
    # ------------------------------------------------------------------

    def list_by_severity(
        self,
        *,
        tenant_id: uuid.UUID,
        severity: str,
        offset: int = 0,
        limit: int = 100,
        open_only: bool = False,
    ) -> Sequence[Incident]:
        """Return incidents at a particular severity."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(Incident).where(
            Incident.tenant_id == tenant_id,
            Incident.severity == severity,
        )

        if open_only:
            statement = statement.where(
                Incident.status.in_(
                    (
                        IncidentStatus.DETECTED.value,
                        IncidentStatus.OPEN.value,
                        IncidentStatus.ACKNOWLEDGED.value,
                        IncidentStatus.INVESTIGATING.value,
                        IncidentStatus.MITIGATING.value,
                        IncidentStatus.MONITORING.value,
                    )
                )
            )

        statement = (
            statement
            .order_by(
                Incident.detected_at.desc()
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
    ) -> Sequence[Incident]:
        """Return critical incidents."""
        return self.list_by_severity(
            tenant_id=tenant_id,
            severity=IncidentSeverity.CRITICAL.value,
            offset=offset,
            limit=limit,
            open_only=open_only,
        )

    # ------------------------------------------------------------------
    # Priority
    # ------------------------------------------------------------------

    def list_by_priority(
        self,
        *,
        tenant_id: uuid.UUID,
        priority: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Incident]:
        """Return incidents at a particular priority."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Incident)
            .where(
                Incident.tenant_id == tenant_id,
                Incident.priority == priority,
            )
            .order_by(
                Incident.detected_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def list_for_correlation(
        self,
        *,
        tenant_id: uuid.UUID,
        correlation_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Incident]:
        """Return incidents belonging to a correlation group."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Incident)
            .where(
                Incident.tenant_id == tenant_id,
                Incident.correlation_id == correlation_id,
            )
            .order_by(
                Incident.detected_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Ownership
    # ------------------------------------------------------------------

    def list_assigned_to(
        self,
        *,
        tenant_id: uuid.UUID,
        assigned_to: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Incident]:
        """Return incidents assigned to an operator."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Incident)
            .where(
                Incident.tenant_id == tenant_id,
                Incident.assigned_to == assigned_to,
            )
            .order_by(
                Incident.detected_at.desc()
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
        """
        Validate that a referenced resource belongs to the tenant.
        """
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
    # Incident creation
    # ------------------------------------------------------------------

    def create_incident(
        self,
        *,
        tenant_id: uuid.UUID,
        title: str,
        detected_at: datetime,
        incident_type: str = IncidentType.UNKNOWN.value,
        severity: str = IncidentSeverity.MEDIUM.value,
        priority: str = IncidentPriority.P3.value,
        source: str = IncidentSource.CORRELATION.value,
        primary_resource_id: uuid.UUID | None = None,
        summary: str | None = None,
        description: str | None = None,
        correlation_id: uuid.UUID | None = None,
        trigger_anomaly_id: uuid.UUID | None = None,
        trigger_prediction_id: uuid.UUID | None = None,
        confidence: float | None = None,
        impact_summary: str | None = None,
        affected_resource_count: int = 1,
        affected_services: dict[str, Any] | None = None,
        impact_data: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
        correlated_events: dict[str, Any] | None = None,
        telemetry_refs: dict[str, Any] | None = None,
        topology_context: dict[str, Any] | None = None,
        metadata_payload: dict[str, Any] | None = None,
        automated: bool = False,
        actionable: bool = True,
    ) -> Incident:
        """
        Create an operational incident.

        The primary resource, when supplied, is validated against tenant
        ownership before persistence.

        The Session is flushed but not committed.
        """
        if primary_resource_id is not None:
            self._require_resource(
                tenant_id=tenant_id,
                resource_id=primary_resource_id,
            )

        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "incident confidence must be between 0.0 and 1.0"
            )

        if affected_resource_count < 0:
            raise ValueError(
                "affected_resource_count cannot be negative"
            )

        incident = Incident(
            tenant_id=tenant_id,
            primary_resource_id=primary_resource_id,
            incident_type=incident_type,
            title=title,
            summary=summary,
            description=description,
            source=source,
            correlation_id=correlation_id,
            trigger_anomaly_id=trigger_anomaly_id,
            trigger_prediction_id=trigger_prediction_id,
            confidence=confidence,
            severity=severity,
            priority=priority,
            status=IncidentStatus.DETECTED.value,
            actionable=actionable,
            automated=automated,
            detected_at=detected_at,
            started_at=detected_at,
            last_activity_at=detected_at,
            impact_summary=impact_summary,
            affected_resource_count=affected_resource_count,
            affected_services=affected_services or {},
            impact_data=impact_data or {},
            evidence=evidence or {},
            correlated_events=correlated_events or {},
            telemetry_refs=telemetry_refs or {},
            topology_context=topology_context or {},
            metadata_payload=metadata_payload or {},
        )

        self.add(
            incident,
            flush=True,
        )

        return incident

    # ------------------------------------------------------------------
    # Incident ownership
    # ------------------------------------------------------------------

    def assign(
        self,
        *,
        incident_id: uuid.UUID,
        assigned_to: str,
        owner_team: str | None = None,
        assigned_at: datetime | None = None,
    ) -> Incident:
        """Assign an incident to an operator or owner."""
        incident = self.get_required(
            incident_id
        )

        incident.assigned_to = assigned_to

        if owner_team is not None:
            incident.owner_team = owner_team

        if assigned_at is not None:
            incident.last_activity_at = assigned_at

        self.flush()

        return incident

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(
        self,
        *,
        incident_id: uuid.UUID,
        opened_at: datetime | None = None,
    ) -> Incident:
        """Open a detected incident."""
        incident = self.get_required(
            incident_id
        )

        incident.open()

        if opened_at is not None:
            incident.last_activity_at = opened_at

        self.flush()

        return incident

    def acknowledge(
        self,
        *,
        incident_id: uuid.UUID,
        acknowledged_by: str,
        acknowledged_at: datetime,
    ) -> Incident:
        """Acknowledge an incident."""
        incident = self.get_required(
            incident_id
        )

        incident.acknowledge(
            acknowledged_by=acknowledged_by,
            acknowledged_at=acknowledged_at,
        )

        self.flush()

        return incident

    def start_investigation(
        self,
        *,
        incident_id: uuid.UUID,
        started_at: datetime,
    ) -> Incident:
        """Move an incident into investigation."""
        incident = self.get_required(
            incident_id
        )

        incident.start_investigation(
            started_at=started_at
        )

        self.flush()

        return incident

    def start_mitigation(
        self,
        *,
        incident_id: uuid.UUID,
        started_at: datetime,
    ) -> Incident:
        """Move an incident into mitigation."""
        incident = self.get_required(
            incident_id
        )

        incident.start_mitigation(
            started_at=started_at
        )

        self.flush()

        return incident

    def start_monitoring(
        self,
        *,
        incident_id: uuid.UUID,
        started_at: datetime | None = None,
    ) -> Incident:
        """Move an incident into post-mitigation monitoring."""
        incident = self.get_required(
            incident_id
        )

        incident.start_monitoring()

        if started_at is not None:
            incident.last_activity_at = started_at

        self.flush()

        return incident

    # ------------------------------------------------------------------
    # Root-cause analysis
    # ------------------------------------------------------------------

    def set_suspected_root_cause(
        self,
        *,
        incident_id: uuid.UUID,
        cause: str,
        confidence: float | None = None,
        resource_id: uuid.UUID | None = None,
    ) -> Incident:
        """Record a suspected incident root cause."""
        incident = self.get_required(
            incident_id
        )

        if resource_id is not None:
            self._require_resource(
                tenant_id=incident.tenant_id,
                resource_id=resource_id,
            )

        incident.set_suspected_root_cause(
            cause=cause,
            confidence=confidence,
            resource_id=resource_id,
        )

        self.flush()

        return incident

    def confirm_root_cause(
        self,
        *,
        incident_id: uuid.UUID,
        cause: str,
        confidence: float | None = None,
        resource_id: uuid.UUID | None = None,
    ) -> Incident:
        """Record the confirmed root cause."""
        incident = self.get_required(
            incident_id
        )

        if resource_id is not None:
            self._require_resource(
                tenant_id=incident.tenant_id,
                resource_id=resource_id,
            )

        incident.confirm_root_cause(
            cause=cause,
            confidence=confidence,
            resource_id=resource_id,
        )

        self.flush()

        return incident

    # ------------------------------------------------------------------
    # Impact
    # ------------------------------------------------------------------

    def update_impact(
        self,
        *,
        incident_id: uuid.UUID,
        impact_summary: str | None = None,
        affected_resource_count: int | None = None,
        affected_services: dict[str, Any] | None = None,
        impact_data: dict[str, Any] | None = None,
        updated_at: datetime | None = None,
    ) -> Incident:
        """Update the current incident impact assessment."""
        incident = self.get_required(
            incident_id
        )

        if (
            affected_resource_count is not None
            and affected_resource_count < 0
        ):
            raise ValueError(
                "affected_resource_count cannot be negative"
            )

        if impact_summary is not None:
            incident.impact_summary = impact_summary

        if affected_resource_count is not None:
            incident.affected_resource_count = affected_resource_count

        if affected_services is not None:
            incident.affected_services = affected_services

        if impact_data is not None:
            incident.impact_data = impact_data

        if updated_at is not None:
            incident.last_activity_at = updated_at

        self.flush()

        return incident

    # ------------------------------------------------------------------
    # Recurrence
    # ------------------------------------------------------------------

    def mark_recurring(
        self,
        *,
        incident_id: uuid.UUID,
        observed_at: datetime,
    ) -> Incident:
        """Record another occurrence of an incident."""
        incident = self.get_required(
            incident_id
        )

        incident.mark_recurring(
            observed_at=observed_at
        )

        self.flush()

        return incident

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        *,
        incident_id: uuid.UUID,
        resolved_at: datetime,
        summary: str,
        resolved_by: str | None = None,
        resolution_code: str | None = None,
    ) -> Incident:
        """Resolve an incident."""
        incident = self.get_required(
            incident_id
        )

        incident.resolve(
            resolved_at=resolved_at,
            summary=summary,
            resolved_by=resolved_by,
            resolution_code=resolution_code,
        )

        self.flush()

        return incident

    def close(
        self,
        *,
        incident_id: uuid.UUID,
        closed_at: datetime,
    ) -> Incident:
        """Close a previously resolved incident."""
        incident = self.get_required(
            incident_id
        )

        incident.close(
            closed_at=closed_at
        )

        self.flush()

        return incident

    # ------------------------------------------------------------------
    # Severity / priority
    # ------------------------------------------------------------------

    def set_severity(
        self,
        *,
        incident_id: uuid.UUID,
        severity: str,
        updated_at: datetime | None = None,
    ) -> Incident:
        """Update incident severity."""
        incident = self.get_required(
            incident_id
        )

        incident.severity = severity

        if updated_at is not None:
            incident.last_activity_at = updated_at

        self.flush()

        return incident

    def set_priority(
        self,
        *,
        incident_id: uuid.UUID,
        priority: str,
        updated_at: datetime | None = None,
    ) -> Incident:
        """Update incident priority."""
        incident = self.get_required(
            incident_id
        )

        incident.priority = priority

        if updated_at is not None:
            incident.last_activity_at = updated_at

        self.flush()

        return incident


__all__ = [
    "IncidentRepository",
]
