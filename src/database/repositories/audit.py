"""
OSHA Audit Repository.

Append-oriented persistence interface for OSHA audit events.

Audit events record security-sensitive, administrative and operational
activity such as:

    - authentication and authorisation decisions
    - infrastructure discovery
    - incident operations
    - anomaly handling
    - prediction/model activity
    - remediation proposals
    - remediation approvals/rejections
    - remediation execution and rollback
    - configuration changes
    - policy decisions
    - privileged administrative operations

Audit records are intentionally treated as append-oriented history.

This repository therefore exposes event creation and retrieval operations but
does not expose ordinary update/delete helpers for audit records.

Sensitive values such as passwords, tokens, private keys and raw credentials
must never be passed into audit payloads.

Transaction ownership remains with OSHA's session/service layer.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.audit import (
    ActorType,
    AuditAction,
    AuditCategory,
    AuditEvent,
    AuditOutcome,
    AuditSeverity,
)
from .base import BaseRepository


class AuditRepository(BaseRepository[AuditEvent]):
    """
    Append-oriented repository for OSHA audit events.

    Unlike normal repositories, this repository should not be used for
    modifying or deleting existing audit records through normal application
    workflows.
    """

    model = AuditEvent

    def __init__(
        self,
        session: Session,
    ) -> None:
        super().__init__(session)

    # ------------------------------------------------------------------
    # Append audit event
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        tenant_id: uuid.UUID,
        event_name: str,
        action: str,
        occurred_at: datetime,
        category: str = AuditCategory.SYSTEM.value,
        outcome: str = AuditOutcome.UNKNOWN.value,
        severity: str = AuditSeverity.INFO.value,
        actor_type: str = ActorType.UNKNOWN.value,
        actor_id: str | None = None,
        actor_name: str | None = None,
        actor_service: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        target_name: str | None = None,
        resource_id: uuid.UUID | None = None,
        incident_id: uuid.UUID | None = None,
        anomaly_id: uuid.UUID | None = None,
        prediction_id: uuid.UUID | None = None,
        remediation_id: uuid.UUID | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        source_ip: str | None = None,
        user_agent: str | None = None,
        message: str | None = None,
        reason: str | None = None,
        changed_fields: dict[str, Any] | None = None,
        previous_state: dict[str, Any] | None = None,
        new_state: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        policy_context: dict[str, Any] | None = None,
        metadata_payload: dict[str, Any] | None = None,
        security_relevant: bool = False,
        privileged_action: bool = False,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> AuditEvent:
        """
        Append an audit event.

        The event is flushed so its generated identifier is available to the
        caller, but the transaction is not committed here.
        """
        if not event_name.strip():
            raise ValueError(
                "audit event_name cannot be empty"
            )

        if not action.strip():
            raise ValueError(
                "audit action cannot be empty"
            )

        event = AuditEvent(
            tenant_id=tenant_id,
            event_name=event_name,
            category=category,
            action=action,
            outcome=outcome,
            severity=severity,
            occurred_at=occurred_at,
            actor_type=actor_type,
            actor_id=actor_id,
            actor_name=actor_name,
            actor_service=actor_service,
            target_type=target_type,
            target_id=target_id,
            target_name=target_name,
            resource_id=resource_id,
            incident_id=incident_id,
            anomaly_id=anomaly_id,
            prediction_id=prediction_id,
            remediation_id=remediation_id,
            request_id=request_id,
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=span_id,
            source_ip=source_ip,
            user_agent=user_agent,
            message=message,
            reason=reason,
            changed_fields=changed_fields or {},
            previous_state=previous_state or {},
            new_state=new_state or {},
            context=context or {},
            policy_context=policy_context or {},
            metadata_payload=metadata_payload or {},
            security_relevant=security_relevant,
            privileged_action=privileged_action,
            error_code=error_code,
            error_message=error_message,
        )

        self.add(
            event,
            flush=True,
        )

        return event

    # ------------------------------------------------------------------
    # Tenant history
    # ------------------------------------------------------------------

    def list_for_tenant(
        self,
        *,
        tenant_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return audit history for a tenant."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(AuditEvent)
            .where(
                AuditEvent.tenant_id == tenant_id
            )
            .order_by(
                AuditEvent.occurred_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Actor history
    # ------------------------------------------------------------------

    def list_for_actor(
        self,
        *,
        tenant_id: uuid.UUID,
        actor_id: str,
        actor_type: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return audit activity performed by an actor."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(AuditEvent).where(
            AuditEvent.tenant_id == tenant_id,
            AuditEvent.actor_id == actor_id,
        )

        if actor_type is not None:
            statement = statement.where(
                AuditEvent.actor_type == actor_type
            )

        statement = (
            statement
            .order_by(
                AuditEvent.occurred_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Target history
    # ------------------------------------------------------------------

    def list_for_target(
        self,
        *,
        tenant_id: uuid.UUID,
        target_type: str,
        target_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return audit history associated with a target entity."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(AuditEvent)
            .where(
                AuditEvent.tenant_id == tenant_id,
                AuditEvent.target_type == target_type,
                AuditEvent.target_id == target_id,
            )
            .order_by(
                AuditEvent.occurred_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Resource history
    # ------------------------------------------------------------------

    def list_for_resource(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return audit activity associated with a resource."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(AuditEvent)
            .where(
                AuditEvent.tenant_id == tenant_id,
                AuditEvent.resource_id == resource_id,
            )
            .order_by(
                AuditEvent.occurred_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Incident history
    # ------------------------------------------------------------------

    def list_for_incident(
        self,
        *,
        tenant_id: uuid.UUID,
        incident_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return the audit timeline associated with an incident."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(AuditEvent)
            .where(
                AuditEvent.tenant_id == tenant_id,
                AuditEvent.incident_id == incident_id,
            )
            .order_by(
                AuditEvent.occurred_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Remediation history
    # ------------------------------------------------------------------

    def list_for_remediation(
        self,
        *,
        tenant_id: uuid.UUID,
        remediation_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return audit history associated with a remediation."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(AuditEvent)
            .where(
                AuditEvent.tenant_id == tenant_id,
                AuditEvent.remediation_id == remediation_id,
            )
            .order_by(
                AuditEvent.occurred_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Request / correlation tracing
    # ------------------------------------------------------------------

    def list_for_request(
        self,
        *,
        tenant_id: uuid.UUID,
        request_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return events associated with an application request."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(AuditEvent)
            .where(
                AuditEvent.tenant_id == tenant_id,
                AuditEvent.request_id == request_id,
            )
            .order_by(
                AuditEvent.occurred_at.asc()
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
        correlation_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return audit events sharing a correlation ID."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(AuditEvent)
            .where(
                AuditEvent.tenant_id == tenant_id,
                AuditEvent.correlation_id == correlation_id,
            )
            .order_by(
                AuditEvent.occurred_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Security events
    # ------------------------------------------------------------------

    def list_security_events(
        self,
        *,
        tenant_id: uuid.UUID,
        privileged_only: bool = False,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return security-relevant audit records."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(AuditEvent).where(
            AuditEvent.tenant_id == tenant_id,
            AuditEvent.security_relevant.is_(True),
        )

        if privileged_only:
            statement = statement.where(
                AuditEvent.privileged_action.is_(True)
            )

        statement = (
            statement
            .order_by(
                AuditEvent.occurred_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Failed operations
    # ------------------------------------------------------------------

    def list_failures(
        self,
        *,
        tenant_id: uuid.UUID,
        category: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return failed audited operations."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(AuditEvent).where(
            AuditEvent.tenant_id == tenant_id,
            AuditEvent.outcome == AuditOutcome.FAILURE.value,
        )

        if category is not None:
            statement = statement.where(
                AuditEvent.category == category
            )

        statement = (
            statement
            .order_by(
                AuditEvent.occurred_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Time window
    # ------------------------------------------------------------------

    def list_between(
        self,
        *,
        tenant_id: uuid.UUID,
        start: datetime,
        end: datetime,
        category: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return audit events occurring within a time window."""
        if end < start:
            raise ValueError(
                "audit end time cannot be earlier than start time"
            )

        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(AuditEvent).where(
            AuditEvent.tenant_id == tenant_id,
            AuditEvent.occurred_at >= start,
            AuditEvent.occurred_at <= end,
        )

        if category is not None:
            statement = statement.where(
                AuditEvent.category == category
            )

        statement = (
            statement
            .order_by(
                AuditEvent.occurred_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )


__all__ = [
    "AuditRepository",
]
