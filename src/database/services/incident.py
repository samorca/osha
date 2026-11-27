"""
OSHA Incident Coordination Service.

Coordinates incident workflows across:

    - resources
    - anomalies
    - predictions
    - topology
    - incidents
    - remediations
    - audit events

The service does not perform anomaly detection, model inference, correlation
algorithms or remediation execution. Those responsibilities belong to their
respective OSHA subsystems.

This service coordinates persistence and lifecycle transitions after those
subsystems have made a decision.

Typical flow:

    anomaly / prediction / external signal
                    |
                    v
            IncidentService
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
    Incident     Topology     Audit
        |
        v
    Remediation proposal

Transaction ownership remains outside this service. Callers should use
OSHA's session_scope() when an incident operation must be atomic.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from ..models.anomaly import Anomaly
from ..models.audit import (
    ActorType,
    AuditAction,
    AuditCategory,
    AuditOutcome,
    AuditSeverity,
)
from ..models.incident import (
    Incident,
    IncidentPriority,
    IncidentSeverity,
    IncidentSource,
    IncidentStatus,
    IncidentType,
)
from ..models.prediction import Prediction
from ..models.remediation import Remediation
from ..repositories.anomaly import AnomalyRepository
from ..repositories.audit import AuditRepository
from ..repositories.incident import IncidentRepository
from ..repositories.prediction import PredictionRepository
from ..repositories.remediation import RemediationRepository
from ..repositories.resource import ResourceRepository
from .topology import TopologyService, TopologyTraversal


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class IncidentServiceError(RuntimeError):
    """Base error raised by the OSHA incident service."""


class IncidentTenantMismatchError(IncidentServiceError):
    """Raised when linked records belong to different tenants."""


class IncidentStateError(IncidentServiceError):
    """Raised when an invalid incident lifecycle operation is requested."""


class IncidentEvidenceNotFoundError(IncidentServiceError):
    """Raised when incident evidence cannot be found."""


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class IncidentCreationResult:
    """Result returned when an incident is created."""

    incident: Incident
    anomaly: Anomaly | None = None
    prediction: Prediction | None = None
    topology: TopologyTraversal | None = None


@dataclass(frozen=True, slots=True)
class IncidentContext:
    """Aggregated operational context for an existing incident."""

    incident: Incident
    anomalies: tuple[Anomaly, ...]
    predictions: tuple[Prediction, ...]
    remediations: tuple[Remediation, ...]
    topology: TopologyTraversal | None


# ---------------------------------------------------------------------------
# Incident service
# ---------------------------------------------------------------------------

class IncidentService:
    """
    Coordinate OSHA incident persistence and lifecycle workflows.

    Every repository shares the same Session so one service operation can be
    committed or rolled back atomically by the caller.
    """

    def __init__(
        self,
        session: Session,
    ) -> None:
        if session is None:
            raise ValueError(
                "IncidentService requires an active Session"
            )

        self.session = session

        self.resources = ResourceRepository(session)
        self.anomalies = AnomalyRepository(session)
        self.predictions = PredictionRepository(session)
        self.incidents = IncidentRepository(session)
        self.remediations = RemediationRepository(session)
        self.audit = AuditRepository(session)

        self.topology = TopologyService(session)

    # ------------------------------------------------------------------
    # Manual/general incident creation
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
        source: str = IncidentSource.USER.value,
        primary_resource_id: uuid.UUID | None = None,
        summary: str | None = None,
        description: str | None = None,
        correlation_id: uuid.UUID | None = None,
        confidence: float | None = None,
        impact_summary: str | None = None,
        affected_resource_count: int = 1,
        evidence: dict[str, Any] | None = None,
        telemetry_refs: dict[str, Any] | None = None,
        topology_context: dict[str, Any] | None = None,
        automated: bool = False,
        actor_type: str = ActorType.SYSTEM.value,
        actor_id: str | None = None,
        actor_name: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> IncidentCreationResult:
        """
        Create a standalone incident and record its audit event.
        """
        if not title.strip():
            raise ValueError(
                "incident title cannot be empty"
            )

        if affected_resource_count < 1:
            raise ValueError(
                "affected_resource_count must be at least 1"
            )

        incident = self.incidents.create_incident(
            tenant_id=tenant_id,
            title=title,
            detected_at=detected_at,
            incident_type=incident_type,
            severity=severity,
            priority=priority,
            source=source,
            primary_resource_id=primary_resource_id,
            summary=summary,
            description=description,
            correlation_id=correlation_id,
            confidence=confidence,
            impact_summary=impact_summary,
            affected_resource_count=affected_resource_count,
            evidence=evidence,
            telemetry_refs=telemetry_refs,
            topology_context=topology_context,
            automated=automated,
            actionable=True,
        )

        topology = self._load_topology_context(
            tenant_id=tenant_id,
            resource_id=primary_resource_id,
        )

        self._record_audit(
            tenant_id=tenant_id,
            event_name="incident.created",
            action=AuditAction.CREATE.value,
            occurred_at=detected_at,
            incident=incident,
            resource_id=primary_resource_id,
            actor_type=actor_type,
            actor_id=actor_id,
            actor_name=actor_name,
            request_id=request_id,
            trace_id=trace_id,
            message="OSHA incident created",
            context={
                "incident_type": incident_type,
                "severity": severity,
                "priority": priority,
                "source": source,
            },
        )

        return IncidentCreationResult(
            incident=incident,
            topology=topology,
        )

    # ------------------------------------------------------------------
    # Create from anomaly
    # ------------------------------------------------------------------

    def create_from_anomaly(
        self,
        *,
        tenant_id: uuid.UUID,
        anomaly_id: uuid.UUID,
        title: str,
        detected_at: datetime,
        incident_type: str = IncidentType.ANOMALY.value,
        severity: str | None = None,
        priority: str = IncidentPriority.P3.value,
        summary: str | None = None,
        correlation_id: uuid.UUID | None = None,
        confidence: float | None = None,
        actor_type: str = ActorType.SYSTEM.value,
        actor_id: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> IncidentCreationResult:
        """
        Create an incident from a persisted anomaly.

        The anomaly is linked to the newly created incident in the same
        transaction.
        """
        anomaly = self.anomalies.get_with_resource(
            anomaly_id=anomaly_id
        )

        if anomaly is None:
            raise IncidentEvidenceNotFoundError(
                f"anomaly {anomaly_id} was not found"
            )

        self._require_same_tenant(
            expected_tenant_id=tenant_id,
            actual_tenant_id=anomaly.tenant_id,
            entity_name="anomaly",
        )

        incident_severity = severity or anomaly.severity

        incident = self.incidents.create_incident(
            tenant_id=tenant_id,
            title=title,
            detected_at=detected_at,
            incident_type=incident_type,
            severity=incident_severity,
            priority=priority,
            source=IncidentSource.ANOMALY.value,
            primary_resource_id=anomaly.resource_id,
            summary=summary or anomaly.description,
            correlation_id=(
                correlation_id
                if correlation_id is not None
                else anomaly.correlation_id
            ),
            trigger_anomaly_id=anomaly.id,
            confidence=(
                confidence
                if confidence is not None
                else anomaly.confidence
            ),
            affected_resource_count=1,
            evidence={
                "anomaly_id": str(anomaly.id),
                "anomaly_type": anomaly.anomaly_type,
                "score": anomaly.score,
                "confidence": anomaly.confidence,
                "evidence": anomaly.evidence,
            },
            telemetry_refs=anomaly.telemetry_refs,
            automated=True,
            actionable=True,
        )

        self.anomalies.attach_to_incident(
            anomaly_id=anomaly.id,
            incident_id=incident.id,
        )

        topology = self._load_topology_context(
            tenant_id=tenant_id,
            resource_id=anomaly.resource_id,
        )

        if topology is not None:
            incident.affected_resource_count = len(
                topology.resources
            )

            incident.topology_context = (
                self._serialise_topology(topology)
            )

        self._record_audit(
            tenant_id=tenant_id,
            event_name="incident.created_from_anomaly",
            action=AuditAction.CREATE.value,
            occurred_at=detected_at,
            incident=incident,
            resource_id=anomaly.resource_id,
            anomaly_id=anomaly.id,
            actor_type=actor_type,
            actor_id=actor_id,
            request_id=request_id,
            trace_id=trace_id,
            message="Incident created from anomaly",
            context={
                "anomaly_type": anomaly.anomaly_type,
                "anomaly_severity": anomaly.severity,
            },
        )

        return IncidentCreationResult(
            incident=incident,
            anomaly=anomaly,
            topology=topology,
        )

    # ------------------------------------------------------------------
    # Create from prediction
    # ------------------------------------------------------------------

    def create_from_prediction(
        self,
        *,
        tenant_id: uuid.UUID,
        prediction_id: uuid.UUID,
        title: str,
        detected_at: datetime,
        incident_type: str = IncidentType.PREDICTION.value,
        severity: str | None = None,
        priority: str = IncidentPriority.P3.value,
        summary: str | None = None,
        correlation_id: uuid.UUID | None = None,
        actor_type: str = ActorType.SYSTEM.value,
        actor_id: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> IncidentCreationResult:
        """
        Create an incident from a persisted prediction.
        """
        prediction = self.predictions.get_with_resource(
            prediction_id=prediction_id
        )

        if prediction is None:
            raise IncidentEvidenceNotFoundError(
                f"prediction {prediction_id} was not found"
            )

        self._require_same_tenant(
            expected_tenant_id=tenant_id,
            actual_tenant_id=prediction.tenant_id,
            entity_name="prediction",
        )

        incident = self.incidents.create_incident(
            tenant_id=tenant_id,
            title=title,
            detected_at=detected_at,
            incident_type=incident_type,
            severity=severity or prediction.severity,
            priority=priority,
            source=IncidentSource.PREDICTION.value,
            primary_resource_id=prediction.resource_id,
            summary=summary or prediction.description,
            correlation_id=(
                correlation_id
                if correlation_id is not None
                else prediction.correlation_id
            ),
            trigger_prediction_id=prediction.id,
            confidence=prediction.confidence,
            affected_resource_count=1,
            evidence={
                "prediction_id": str(prediction.id),
                "prediction_type": prediction.prediction_type,
                "model_name": prediction.model_name,
                "model_version": prediction.model_version,
                "probability": prediction.probability,
                "confidence": prediction.confidence,
                "predicted_for": (
                    prediction.predicted_for.isoformat()
                    if prediction.predicted_for is not None
                    else None
                ),
                "evidence": prediction.evidence,
            },
            automated=True,
            actionable=prediction.actionable,
        )

        self.predictions.attach_to_incident(
            prediction_id=prediction.id,
            incident_id=incident.id,
        )

        topology = self._load_topology_context(
            tenant_id=tenant_id,
            resource_id=prediction.resource_id,
        )

        if topology is not None:
            incident.affected_resource_count = len(
                topology.resources
            )

            incident.topology_context = (
                self._serialise_topology(topology)
            )

        self._record_audit(
            tenant_id=tenant_id,
            event_name="incident.created_from_prediction",
            action=AuditAction.CREATE.value,
            occurred_at=detected_at,
            incident=incident,
            resource_id=prediction.resource_id,
            prediction_id=prediction.id,
            actor_type=actor_type,
            actor_id=actor_id,
            request_id=request_id,
            trace_id=trace_id,
            message="Incident created from prediction",
            context={
                "prediction_type": prediction.prediction_type,
                "model_name": prediction.model_name,
                "model_version": prediction.model_version,
            },
        )

        return IncidentCreationResult(
            incident=incident,
            prediction=prediction,
            topology=topology,
        )

    # ------------------------------------------------------------------
    # Link existing evidence
    # ------------------------------------------------------------------

    def attach_anomaly(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        anomaly_id: uuid.UUID,
        attached_at: datetime,
        actor_type: str = ActorType.SYSTEM.value,
        actor_id: str | None = None,
    ) -> Anomaly:
        """Attach an existing anomaly to an incident."""
        incident = self._require_incident(
            tenant_id=tenant_id,
            incident_id=incident_id,
        )

        anomaly = self.anomalies.get_required(
            anomaly_id
        )

        self._require_same_tenant(
            expected_tenant_id=tenant_id,
            actual_tenant_id=anomaly.tenant_id,
            entity_name="anomaly",
        )

        self.anomalies.attach_to_incident(
            anomaly_id=anomaly.id,
            incident_id=incident.id,
        )

        incident.last_activity_at = attached_at

        self._record_audit(
            tenant_id=tenant_id,
            event_name="incident.anomaly_attached",
            action=AuditAction.UPDATE.value,
            occurred_at=attached_at,
            incident=incident,
            resource_id=anomaly.resource_id,
            anomaly_id=anomaly.id,
            actor_type=actor_type,
            actor_id=actor_id,
            message="Anomaly attached to incident",
        )

        return anomaly

    def attach_prediction(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        prediction_id: uuid.UUID,
        attached_at: datetime,
        actor_type: str = ActorType.SYSTEM.value,
        actor_id: str | None = None,
    ) -> Prediction:
        """Attach an existing prediction to an incident."""
        incident = self._require_incident(
            tenant_id=tenant_id,
            incident_id=incident_id,
        )

        prediction = self.predictions.get_required(
            prediction_id
        )

        self._require_same_tenant(
            expected_tenant_id=tenant_id,
            actual_tenant_id=prediction.tenant_id,
            entity_name="prediction",
        )

        self.predictions.attach_to_incident(
            prediction_id=prediction.id,
            incident_id=incident.id,
        )

        incident.last_activity_at = attached_at

        self._record_audit(
            tenant_id=tenant_id,
            event_name="incident.prediction_attached",
            action=AuditAction.UPDATE.value,
            occurred_at=attached_at,
            incident=incident,
            resource_id=prediction.resource_id,
            prediction_id=prediction.id,
            actor_type=actor_type,
            actor_id=actor_id,
            message="Prediction attached to incident",
        )

        return prediction

    # ------------------------------------------------------------------
    # Incident context
    # ------------------------------------------------------------------

    def get_context(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        topology_depth: int = 2,
        topology_max_nodes: int = 250,
    ) -> IncidentContext:
        """Return persisted incident evidence and topology context."""
        incident = self._require_incident(
            tenant_id=tenant_id,
            incident_id=incident_id,
        )

        anomalies = self.anomalies.list_for_incident(
            tenant_id=tenant_id,
            incident_id=incident.id,
            offset=0,
            limit=1000,
        )

        predictions = self.predictions.list_for_incident(
            tenant_id=tenant_id,
            incident_id=incident.id,
            offset=0,
            limit=1000,
        )

        remediations = self.remediations.list_for_incident(
            tenant_id=tenant_id,
            incident_id=incident.id,
            offset=0,
            limit=1000,
        )

        topology: TopologyTraversal | None = None

        if incident.primary_resource_id is not None:
            topology = self.topology.impact_neighbourhood(
                tenant_id=tenant_id,
                resource_id=incident.primary_resource_id,
                max_depth=topology_depth,
                max_nodes=topology_max_nodes,
            )

        return IncidentContext(
            incident=incident,
            anomalies=tuple(anomalies),
            predictions=tuple(predictions),
            remediations=tuple(remediations),
            topology=topology,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def acknowledge(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        acknowledged_by: str,
        acknowledged_at: datetime,
        actor_type: str = ActorType.USER.value,
        actor_id: str | None = None,
    ) -> Incident:
        """Acknowledge an incident and record an audit event."""
        incident = self._require_incident(
            tenant_id=tenant_id,
            incident_id=incident_id,
        )

        self._require_status(
            incident,
            {
                IncidentStatus.DETECTED.value,
                IncidentStatus.OPEN.value,
            },
            "acknowledge",
        )

        incident = self.incidents.acknowledge(
            incident_id=incident.id,
            acknowledged_by=acknowledged_by,
            acknowledged_at=acknowledged_at,
        )

        self._record_audit(
            tenant_id=tenant_id,
            event_name="incident.acknowledged",
            action=AuditAction.ACKNOWLEDGE.value,
            occurred_at=acknowledged_at,
            incident=incident,
            resource_id=incident.primary_resource_id,
            actor_type=actor_type,
            actor_id=actor_id or acknowledged_by,
            actor_name=acknowledged_by,
            message="Incident acknowledged",
        )

        return incident

    def start_investigation(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        started_at: datetime,
        actor_type: str = ActorType.USER.value,
        actor_id: str | None = None,
    ) -> Incident:
        """Start incident investigation."""
        incident = self._require_incident(
            tenant_id=tenant_id,
            incident_id=incident_id,
        )

        self._require_status(
            incident,
            {
                IncidentStatus.DETECTED.value,
                IncidentStatus.OPEN.value,
                IncidentStatus.ACKNOWLEDGED.value,
            },
            "start investigation",
        )

        incident = self.incidents.start_investigation(
            incident_id=incident.id,
            started_at=started_at,
        )

        self._record_audit(
            tenant_id=tenant_id,
            event_name="incident.investigation_started",
            action=AuditAction.START.value,
            occurred_at=started_at,
            incident=incident,
            resource_id=incident.primary_resource_id,
            actor_type=actor_type,
            actor_id=actor_id,
            message="Incident investigation started",
        )

        return incident

    def set_suspected_root_cause(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        cause: str,
        observed_at: datetime,
        confidence: float | None = None,
        resource_id: uuid.UUID | None = None,
        evidence: dict[str, Any] | None = None,
        actor_type: str = ActorType.SYSTEM.value,
        actor_id: str | None = None,
    ) -> Incident:
        """Record a suspected root cause."""
        incident = self._require_incident(
            tenant_id=tenant_id,
            incident_id=incident_id,
        )

        incident = self.incidents.set_suspected_root_cause(
            incident_id=incident.id,
            cause=cause,
            confidence=confidence,
            resource_id=resource_id,
        )

        if evidence is not None:
            incident.root_cause_evidence = evidence

        incident.last_activity_at = observed_at

        self._record_audit(
            tenant_id=tenant_id,
            event_name="incident.root_cause_suspected",
            action=AuditAction.UPDATE.value,
            occurred_at=observed_at,
            incident=incident,
            resource_id=resource_id or incident.primary_resource_id,
            actor_type=actor_type,
            actor_id=actor_id,
            message="Suspected incident root cause recorded",
            context={
                "confidence": confidence,
            },
        )

        return incident

    def confirm_root_cause(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        cause: str,
        confirmed_at: datetime,
        confidence: float | None = None,
        resource_id: uuid.UUID | None = None,
        evidence: dict[str, Any] | None = None,
        actor_type: str = ActorType.USER.value,
        actor_id: str | None = None,
    ) -> Incident:
        """Record the confirmed incident root cause."""
        incident = self._require_incident(
            tenant_id=tenant_id,
            incident_id=incident_id,
        )

        incident = self.incidents.confirm_root_cause(
            incident_id=incident.id,
            cause=cause,
            confidence=confidence,
            resource_id=resource_id,
        )

        if evidence is not None:
            incident.root_cause_evidence = evidence

        incident.last_activity_at = confirmed_at

        self._record_audit(
            tenant_id=tenant_id,
            event_name="incident.root_cause_confirmed",
            action=AuditAction.UPDATE.value,
            occurred_at=confirmed_at,
            incident=incident,
            resource_id=resource_id or incident.primary_resource_id,
            actor_type=actor_type,
            actor_id=actor_id,
            message="Incident root cause confirmed",
            context={
                "confidence": confidence,
            },
        )

        return incident

    def start_mitigation(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        started_at: datetime,
        actor_type: str = ActorType.SYSTEM.value,
        actor_id: str | None = None,
    ) -> Incident:
        """Move an incident into mitigation."""
        incident = self._require_incident(
            tenant_id=tenant_id,
            incident_id=incident_id,
        )

        self._require_status(
            incident,
            {
                IncidentStatus.ACKNOWLEDGED.value,
                IncidentStatus.INVESTIGATING.value,
            },
            "start mitigation",
        )

        incident = self.incidents.start_mitigation(
            incident_id=incident.id,
            started_at=started_at,
        )

        self._record_audit(
            tenant_id=tenant_id,
            event_name="incident.mitigation_started",
            action=AuditAction.START.value,
            occurred_at=started_at,
            incident=incident,
            resource_id=incident.primary_resource_id,
            actor_type=actor_type,
            actor_id=actor_id,
            message="Incident mitigation started",
        )

        return incident

    def start_monitoring(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        started_at: datetime,
        actor_type: str = ActorType.SYSTEM.value,
        actor_id: str | None = None,
    ) -> Incident:
        """Move a mitigating incident into monitoring."""
        incident = self._require_incident(
            tenant_id=tenant_id,
            incident_id=incident_id,
        )

        self._require_status(
            incident,
            {
                IncidentStatus.MITIGATING.value,
            },
            "start monitoring",
        )

        incident = self.incidents.start_monitoring(
            incident_id=incident.id,
            started_at=started_at,
        )

        self._record_audit(
            tenant_id=tenant_id,
            event_name="incident.monitoring_started",
            action=AuditAction.START.value,
            occurred_at=started_at,
            incident=incident,
            resource_id=incident.primary_resource_id,
            actor_type=actor_type,
            actor_id=actor_id,
            message="Post-mitigation monitoring started",
        )

        return incident

    def resolve(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        resolved_at: datetime,
        summary: str,
        resolved_by: str | None = None,
        resolution_code: str | None = None,
        actor_type: str = ActorType.USER.value,
        actor_id: str | None = None,
    ) -> Incident:
        """Resolve an active incident."""
        incident = self._require_incident(
            tenant_id=tenant_id,
            incident_id=incident_id,
        )

        self._require_status(
            incident,
            {
                IncidentStatus.OPEN.value,
                IncidentStatus.ACKNOWLEDGED.value,
                IncidentStatus.INVESTIGATING.value,
                IncidentStatus.MITIGATING.value,
                IncidentStatus.MONITORING.value,
            },
            "resolve",
        )

        incident = self.incidents.resolve(
            incident_id=incident.id,
            resolved_at=resolved_at,
            summary=summary,
            resolved_by=resolved_by,
            resolution_code=resolution_code,
        )

        self._record_audit(
            tenant_id=tenant_id,
            event_name="incident.resolved",
            action=AuditAction.RESOLVE.value,
            occurred_at=resolved_at,
            incident=incident,
            resource_id=incident.primary_resource_id,
            actor_type=actor_type,
            actor_id=actor_id or resolved_by,
            actor_name=resolved_by,
            message="Incident resolved",
            context={
                "resolution_code": resolution_code,
            },
        )

        return incident

    def close(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        closed_at: datetime,
        actor_type: str = ActorType.USER.value,
        actor_id: str | None = None,
    ) -> Incident:
        """Close a resolved incident."""
        incident = self._require_incident(
            tenant_id=tenant_id,
            incident_id=incident_id,
        )

        self._require_status(
            incident,
            {
                IncidentStatus.RESOLVED.value,
            },
            "close",
        )

        incident = self.incidents.close(
            incident_id=incident.id,
            closed_at=closed_at,
        )

        self._record_audit(
            tenant_id=tenant_id,
            event_name="incident.closed",
            action=AuditAction.CLOSE.value,
            occurred_at=closed_at,
            incident=incident,
            resource_id=incident.primary_resource_id,
            actor_type=actor_type,
            actor_id=actor_id,
            message="Incident closed",
        )

        return incident

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_incident(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
    ) -> Incident:
        """Return a tenant-owned incident."""
        incident = self.incidents.get(
            incident_id
        )

        if incident is None:
            raise IncidentEvidenceNotFoundError(
                f"incident {incident_id} was not found"
            )

        self._require_same_tenant(
            expected_tenant_id=tenant_id,
            actual_tenant_id=incident.tenant_id,
            entity_name="incident",
        )

        return incident

    @staticmethod
    def _require_same_tenant(
        *,
        expected_tenant_id: uuid.UUID,
        actual_tenant_id: uuid.UUID,
        entity_name: str,
    ) -> None:
        """Prevent cross-tenant incident associations."""
        if actual_tenant_id != expected_tenant_id:
            raise IncidentTenantMismatchError(
                f"{entity_name} does not belong to "
                "the specified tenant"
            )

    @staticmethod
    def _require_status(
        incident: Incident,
        allowed: set[str],
        operation: str,
    ) -> None:
        """Validate an incident lifecycle transition."""
        if incident.status not in allowed:
            raise IncidentStateError(
                f"cannot {operation} incident while status is "
                f"{incident.status!r}"
            )

    def _load_topology_context(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID | None,
    ) -> TopologyTraversal | None:
        """Load bounded topology context when a resource is available."""
        if resource_id is None:
            return None

        return self.topology.impact_neighbourhood(
            tenant_id=tenant_id,
            resource_id=resource_id,
            max_depth=2,
            max_nodes=250,
        )

    @staticmethod
    def _serialise_topology(
        topology: TopologyTraversal,
    ) -> dict[str, Any]:
        """Create a compact topology snapshot for incident evidence."""
        return {
            "root_resource_id": str(
                topology.root.id
            ),
            "resource_ids": [
                str(resource.id)
                for resource in topology.resources
            ],
            "relationships": [
                {
                    "id": str(relationship.id),
                    "source_resource_id": str(
                        relationship.source_resource_id
                    ),
                    "target_resource_id": str(
                        relationship.target_resource_id
                    ),
                    "relationship_type": (
                        relationship.relationship_type
                    ),
                }
                for relationship in topology.relationships
            ],
            "max_depth": topology.max_depth,
            "truncated": topology.truncated,
        }

    def _record_audit(
        self,
        *,
        tenant_id: uuid.UUID,
        event_name: str,
        action: str,
        occurred_at: datetime,
        incident: Incident,
        resource_id: uuid.UUID | None,
        actor_type: str,
        actor_id: str | None,
        actor_name: str | None = None,
        anomaly_id: uuid.UUID | None = None,
        prediction_id: uuid.UUID | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
        message: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Append a standard incident audit event."""
        self.audit.record(
            tenant_id=tenant_id,
            event_name=event_name,
            category=AuditCategory.INCIDENT.value,
            action=action,
            outcome=AuditOutcome.SUCCESS.value,
            severity=self._audit_severity(
                incident.severity
            ),
            occurred_at=occurred_at,
            actor_type=actor_type,
            actor_id=actor_id,
            actor_name=actor_name,
            target_type="incident",
            target_id=str(incident.id),
            target_name=incident.title,
            resource_id=resource_id,
            incident_id=incident.id,
            anomaly_id=anomaly_id,
            prediction_id=prediction_id,
            request_id=request_id,
            correlation_id=(
                str(incident.correlation_id)
                if incident.correlation_id is not None
                else None
            ),
            trace_id=trace_id,
            message=message,
            context=context or {},
            security_relevant=False,
            privileged_action=False,
        )

    @staticmethod
    def _audit_severity(
        incident_severity: str,
    ) -> str:
        """Map incident severity into audit severity."""
        mapping = {
            IncidentSeverity.INFO.value: (
                AuditSeverity.INFO.value
            ),
            IncidentSeverity.LOW.value: (
                AuditSeverity.LOW.value
            ),
            IncidentSeverity.MEDIUM.value: (
                AuditSeverity.MEDIUM.value
            ),
            IncidentSeverity.HIGH.value: (
                AuditSeverity.HIGH.value
            ),
            IncidentSeverity.CRITICAL.value: (
                AuditSeverity.CRITICAL.value
            ),
        }

        return mapping.get(
            incident_severity,
            AuditSeverity.INFO.value,
        )


__all__ = [
    "IncidentContext",
    "IncidentCreationResult",
    "IncidentEvidenceNotFoundError",
    "IncidentService",
    "IncidentServiceError",
    "IncidentStateError",
    "IncidentTenantMismatchError",
]
