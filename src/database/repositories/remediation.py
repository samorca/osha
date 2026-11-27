"""
OSHA Remediation Repository.

Persistence interface for remediation plans and execution lifecycle.

The repository manages database state for:

    proposal
        -> approval
        -> scheduling
        -> execution
        -> result
        -> outcome
        -> rollback

IMPORTANT:

This repository does NOT execute infrastructure actions.

Actual remediation execution belongs to OSHA's guarded remediation/executor
subsystem, where permissions, policies, safety constraints and runtime
preconditions must be evaluated before changing infrastructure.

Transaction ownership remains with OSHA's service/session layer.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..models.remediation import (
    ApprovalMode,
    ExecutionMode,
    Remediation,
    RemediationOutcome,
    RemediationRisk,
    RemediationStatus,
)
from ..models.resource import Resource
from .base import BaseRepository


class RemediationRepository(BaseRepository[Remediation]):
    """Repository for OSHA remediation lifecycle records."""

    model = Remediation

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
        remediation_id: uuid.UUID,
    ) -> Remediation | None:
        """Return remediation with its target resource loaded."""
        statement = (
            select(Remediation)
            .options(
                selectinload(
                    Remediation.resource
                )
            )
            .where(
                Remediation.id == remediation_id
            )
            .limit(1)
        )

        return self.session.scalar(statement)

    # ------------------------------------------------------------------
    # Tenant listing
    # ------------------------------------------------------------------

    def list_for_tenant(
        self,
        *,
        tenant_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Remediation]:
        """
        Return remediation records belonging to a tenant.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Remediation)
            .where(
                Remediation.tenant_id == tenant_id
            )
            .order_by(
                Remediation.proposed_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Resource listing
    # ------------------------------------------------------------------

    def list_for_resource(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Remediation]:
        """
        Return remediation history for a resource.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Remediation)
            .where(
                Remediation.tenant_id == tenant_id,
                Remediation.resource_id == resource_id,
            )
            .order_by(
                Remediation.proposed_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Incident / anomaly / prediction associations
    # ------------------------------------------------------------------

    def list_for_incident(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Remediation]:
        """
        Return remediations associated with an incident.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Remediation)
            .where(
                Remediation.tenant_id == tenant_id,
                Remediation.incident_id == incident_id,
            )
            .order_by(
                Remediation.proposed_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    def list_for_anomaly(
        self,
        *,
        tenant_id: uuid.UUID,
        anomaly_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Remediation]:
        """
        Return remediations associated with an anomaly.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Remediation)
            .where(
                Remediation.tenant_id == tenant_id,
                Remediation.anomaly_id == anomaly_id,
            )
            .order_by(
                Remediation.proposed_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    def list_for_prediction(
        self,
        *,
        tenant_id: uuid.UUID,
        prediction_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Remediation]:
        """
        Return remediations associated with a prediction.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Remediation)
            .where(
                Remediation.tenant_id == tenant_id,
                Remediation.prediction_id == prediction_id,
            )
            .order_by(
                Remediation.proposed_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def list_by_status(
        self,
        *,
        tenant_id: uuid.UUID,
        status: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Remediation]:
        """Return remediations in a specific lifecycle state."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Remediation)
            .where(
                Remediation.tenant_id == tenant_id,
                Remediation.status == status,
            )
            .order_by(
                Remediation.proposed_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Approval queue
    # ------------------------------------------------------------------

    def list_pending_approval(
        self,
        *,
        tenant_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Remediation]:
        """
        Return remediations waiting for approval.
        """
        return self.list_by_status(
            tenant_id=tenant_id,
            status=RemediationStatus.PENDING_APPROVAL.value,
            offset=offset,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Executable queue
    # ------------------------------------------------------------------

    def list_executable(
        self,
        *,
        tenant_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Remediation]:
        """
        Return persisted remediation records whose state allows execution.

        This query does NOT mean execution is automatically safe.

        The execution subsystem must still check runtime policy,
        authorisation, preconditions and infrastructure state immediately
        before performing an action.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Remediation)
            .where(
                Remediation.tenant_id == tenant_id,
                Remediation.status.in_(
                    (
                        RemediationStatus.APPROVED.value,
                        RemediationStatus.SCHEDULED.value,
                    )
                ),
                (
                    (Remediation.approval_required.is_(False))
                    | (Remediation.approved.is_(True))
                ),
            )
            .order_by(
                Remediation.scheduled_at.asc().nullsfirst(),
                Remediation.proposed_at.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Risk queries
    # ------------------------------------------------------------------

    def list_by_risk(
        self,
        *,
        tenant_id: uuid.UUID,
        risk: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Remediation]:
        """
        Return remediation records at a given risk level.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Remediation)
            .where(
                Remediation.tenant_id == tenant_id,
                Remediation.risk == risk,
            )
            .order_by(
                Remediation.proposed_at.desc()
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
        Validate that a remediation target belongs to the tenant.
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
    # Proposal creation
    # ------------------------------------------------------------------

    def create_proposal(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        remediation_type: str,
        title: str,
        action_name: str,
        proposed_at: datetime,
        description: str | None = None,
        reason: str | None = None,
        execution_mode: str = ExecutionMode.MANUAL.value,
        risk: str = RemediationRisk.MEDIUM.value,
        confidence: float | None = None,
        approval_required: bool = True,
        approval_mode: str = ApprovalMode.MANUAL.value,
        executor_type: str | None = None,
        action_parameters: dict[str, Any] | None = None,
        preconditions: dict[str, Any] | None = None,
        safety_constraints: dict[str, Any] | None = None,
        expected_result: dict[str, Any] | None = None,
        max_attempts: int = 1,
        rollback_supported: bool = False,
        rollback_action: str | None = None,
        rollback_parameters: dict[str, Any] | None = None,
        incident_id: uuid.UUID | None = None,
        anomaly_id: uuid.UUID | None = None,
        prediction_id: uuid.UUID | None = None,
        correlation_id: uuid.UUID | None = None,
        evidence: dict[str, Any] | None = None,
        decision_context: dict[str, Any] | None = None,
        policy_context: dict[str, Any] | None = None,
        metadata_payload: dict[str, Any] | None = None,
    ) -> Remediation:
        """
        Create a remediation proposal.

        This persists remediation intent only. It does not perform the
        infrastructure action.

        Resource ownership is validated before persistence.
        """
        self._require_resource(
            tenant_id=tenant_id,
            resource_id=resource_id,
        )

        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "remediation confidence must be between 0.0 and 1.0"
            )

        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least 1"
            )

        status = RemediationStatus.PROPOSED.value

        if approval_required:
            status = RemediationStatus.PENDING_APPROVAL.value

        remediation = Remediation(
            tenant_id=tenant_id,
            resource_id=resource_id,
            incident_id=incident_id,
            anomaly_id=anomaly_id,
            prediction_id=prediction_id,
            correlation_id=correlation_id,
            remediation_type=remediation_type,
            title=title,
            description=description,
            reason=reason,
            status=status,
            execution_mode=execution_mode,
            outcome=RemediationOutcome.UNKNOWN.value,
            risk=risk,
            confidence=confidence,
            approval_required=approval_required,
            approval_mode=approval_mode,
            approved=False,
            action_name=action_name,
            executor_type=executor_type,
            action_parameters=action_parameters or {},
            preconditions=preconditions or {},
            safety_constraints=safety_constraints or {},
            expected_result=expected_result or {},
            proposed_at=proposed_at,
            attempt_count=0,
            max_attempts=max_attempts,
            rollback_supported=rollback_supported,
            rollback_action=rollback_action,
            rollback_parameters=rollback_parameters or {},
            evidence=evidence or {},
            decision_context=decision_context or {},
            policy_context=policy_context or {},
            metadata_payload=metadata_payload or {},
        )

        self.add(
            remediation,
            flush=True,
        )

        return remediation

    # ------------------------------------------------------------------
    # Association helpers
    # ------------------------------------------------------------------

    def attach_to_incident(
        self,
        *,
        remediation_id: uuid.UUID,
        incident_id: uuid.UUID,
    ) -> Remediation:
        """Associate remediation with an incident."""
        remediation = self.get_required(
            remediation_id
        )

        remediation.incident_id = incident_id

        self.flush()

        return remediation

    def attach_to_anomaly(
        self,
        *,
        remediation_id: uuid.UUID,
        anomaly_id: uuid.UUID,
    ) -> Remediation:
        """Associate remediation with an anomaly."""
        remediation = self.get_required(
            remediation_id
        )

        remediation.anomaly_id = anomaly_id

        self.flush()

        return remediation

    def attach_to_prediction(
        self,
        *,
        remediation_id: uuid.UUID,
        prediction_id: uuid.UUID,
    ) -> Remediation:
        """Associate remediation with a prediction."""
        remediation = self.get_required(
            remediation_id
        )

        remediation.prediction_id = prediction_id

        self.flush()

        return remediation

    # ------------------------------------------------------------------
    # Approval
    # ------------------------------------------------------------------

    def request_approval(
        self,
        *,
        remediation_id: uuid.UUID,
    ) -> Remediation:
        """Move remediation into the approval queue."""
        remediation = self.get_required(
            remediation_id
        )

        remediation.request_approval()

        self.flush()

        return remediation

    def approve(
        self,
        *,
        remediation_id: uuid.UUID,
        approved_by: str,
        approved_at: datetime,
        reason: str | None = None,
    ) -> Remediation:
        """
        Record approval for remediation.

        This still does not execute the infrastructure action.
        """
        remediation = self.get_required(
            remediation_id
        )

        remediation.approve(
            approved_by=approved_by,
            approved_at=approved_at,
            reason=reason,
        )

        self.flush()

        return remediation

    def reject(
        self,
        *,
        remediation_id: uuid.UUID,
        rejected_by: str,
        rejected_at: datetime,
        reason: str | None = None,
    ) -> Remediation:
        """Reject a remediation proposal."""
        remediation = self.get_required(
            remediation_id
        )

        remediation.reject(
            rejected_by=rejected_by,
            rejected_at=rejected_at,
            reason=reason,
        )

        self.flush()

        return remediation

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(
        self,
        *,
        remediation_id: uuid.UUID,
        scheduled_at: datetime,
    ) -> Remediation:
        """
        Schedule an approved remediation.

        Approval requirements remain enforced by persisted state.
        """
        remediation = self.get_required(
            remediation_id
        )

        if remediation.approval_required and not remediation.approved:
            raise ValueError(
                "remediation cannot be scheduled without approval"
            )

        if remediation.status not in {
            RemediationStatus.APPROVED.value,
            RemediationStatus.SCHEDULED.value,
        }:
            raise ValueError(
                "remediation is not in a schedulable state"
            )

        remediation.scheduled_at = scheduled_at
        remediation.status = RemediationStatus.SCHEDULED.value

        self.flush()

        return remediation

    # ------------------------------------------------------------------
    # Execution state
    # ------------------------------------------------------------------

    def mark_executing(
        self,
        *,
        remediation_id: uuid.UUID,
        started_at: datetime,
        execution_id: uuid.UUID,
        executor_id: str | None = None,
    ) -> Remediation:
        """
        Record that an authorised executor has started execution.

        This method does not execute the action itself.
        """
        remediation = self.get_required(
            remediation_id
        )

        remediation.mark_executing(
            started_at=started_at,
            execution_id=execution_id,
            executor_id=executor_id,
        )

        self.flush()

        return remediation

    def mark_succeeded(
        self,
        *,
        remediation_id: uuid.UUID,
        completed_at: datetime,
        result: dict[str, Any] | None = None,
    ) -> Remediation:
        """Record successful remediation execution."""
        remediation = self.get_required(
            remediation_id
        )

        remediation.mark_succeeded(
            completed_at=completed_at,
            result=result,
        )

        self.flush()

        return remediation

    def mark_failed(
        self,
        *,
        remediation_id: uuid.UUID,
        completed_at: datetime,
        error_message: str,
        error_code: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> Remediation:
        """Record failed remediation execution."""
        remediation = self.get_required(
            remediation_id
        )

        remediation.mark_failed(
            completed_at=completed_at,
            error_message=error_message,
            error_code=error_code,
            result=result,
        )

        self.flush()

        return remediation

    # ------------------------------------------------------------------
    # Outcome
    # ------------------------------------------------------------------

    def set_outcome(
        self,
        *,
        remediation_id: uuid.UUID,
        outcome: RemediationOutcome,
    ) -> Remediation:
        """
        Record the observed infrastructure outcome after remediation.
        """
        remediation = self.get_required(
            remediation_id
        )

        remediation.set_outcome(
            outcome
        )

        self.flush()

        return remediation

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def request_rollback(
        self,
        *,
        remediation_id: uuid.UUID,
    ) -> Remediation:
        """Request rollback of a remediation."""
        remediation = self.get_required(
            remediation_id
        )

        remediation.request_rollback()

        self.flush()

        return remediation

    def mark_rollback_started(
        self,
        *,
        remediation_id: uuid.UUID,
        started_at: datetime,
    ) -> Remediation:
        """Record rollback execution start."""
        remediation = self.get_required(
            remediation_id
        )

        remediation.mark_rollback_started(
            started_at=started_at
        )

        self.flush()

        return remediation

    def mark_rolled_back(
        self,
        *,
        remediation_id: uuid.UUID,
        completed_at: datetime,
        result: dict[str, Any] | None = None,
    ) -> Remediation:
        """Record successful rollback."""
        remediation = self.get_required(
            remediation_id
        )

        remediation.mark_rolled_back(
            completed_at=completed_at,
            result=result,
        )

        self.flush()

        return remediation

    def mark_rollback_failed(
        self,
        *,
        remediation_id: uuid.UUID,
        completed_at: datetime,
        error_message: str,
    ) -> Remediation:
        """Record rollback failure."""
        remediation = self.get_required(
            remediation_id
        )

        remediation.mark_rollback_failed(
            completed_at=completed_at,
            error_message=error_message,
        )

        self.flush()

        return remediation


__all__ = [
    "RemediationRepository",
]
