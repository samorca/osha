"""SQLAlchemy-backed repositories for API incident workflows."""

from __future__ import annotations

import asyncio
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.api.errors import (
    DecisionNotFoundAPIError,
    IncidentNotFoundAPIError,
    RemediationNotFoundAPIError,
    RemediationVersionConflictAPIError,
    ResourceConflictError,
)
from src.api.services.decisions import DecisionRecord, PolicyResult
from src.api.services.incidents import (
    IncidentQuery,
    IncidentRecord,
    IncidentSignalRecord,
)
from src.api.services.remediation import (
    RemediationPlanRecord,
    RemediationStepRecord,
)
from src.database.models import Decision, Incident, Remediation
from src.database.repositories import (
    DecisionRepository,
    IncidentRepository,
    RemediationRepository,
)
from src.database.session import get_session_factory


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    return value


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _stable_uuid(kind: str, value: str | None) -> UUID:
    return uuid5(NAMESPACE_URL, f"osha:{kind}:{value or '<default>'}")


def _optional_uuid(value: str | None) -> UUID | None:
    if not value:
        return None
    try:
        return UUID(value)
    except (TypeError, ValueError):
        return None


def _incident_payload(row: Incident) -> IncidentRecord:
    payload = row.metadata_payload.get("api_record")
    if not isinstance(payload, dict):
        raise RuntimeError("API-managed incident is missing its record payload")

    signals = tuple(
        IncidentSignalRecord(
            signal_id=signal["signal_id"],
            source=signal["source"],
            signal_type=signal["signal_type"],
            observed_at=_parse_datetime(signal["observed_at"]),
            severity=signal["severity"],
            payload=signal.get("payload", {}),
            attributes=signal.get("attributes", {}),
        )
        for signal in payload["signals"]
    )
    return IncidentRecord(
        incident_id=payload["incident_id"],
        title=payload["title"],
        summary=payload.get("summary"),
        severity=payload["severity"],
        status=payload["status"],
        service=payload.get("service"),
        environment=payload.get("environment"),
        signals=signals,
        attributes=payload.get("attributes", {}),
        created_at=_parse_datetime(payload["created_at"]),
        updated_at=_parse_datetime(payload["updated_at"]),
        version=row.version,
        tenant_id=payload.get("tenant_id"),
        created_by=payload.get("created_by"),
        correlation_id=payload.get("correlation_id"),
        idempotency_key=payload.get("idempotency_key"),
    )


def _incident_values(record: IncidentRecord) -> dict[str, Any]:
    payload = _json_value(record)
    return {
        "id": _stable_uuid("incident", record.incident_id),
        "tenant_id": _stable_uuid("tenant", record.tenant_id),
        "version": record.version,
        "incident_type": "unknown",
        "title": record.title,
        "summary": record.summary,
        "source": "event",
        "severity": record.severity,
        "status": record.status,
        "detected_at": record.created_at,
        "started_at": record.created_at,
        "last_activity_at": record.updated_at,
        "correlation_id": _optional_uuid(record.correlation_id),
        "affected_services": {
            "service": record.service,
            "environment": record.environment,
        },
        "evidence": {"signals": payload["signals"]},
        "metadata_payload": {"api_record": payload},
        "api_managed": True,
        "api_idempotency_key": record.idempotency_key or None,
    }


class SQLAlchemyIncidentRepository:
    """Persist the API incident contract through the SQL incident repository."""

    def __init__(self, sessions: sessionmaker | None = None) -> None:
        self._sessions = sessions or get_session_factory()

    async def create(self, incident: IncidentRecord) -> IncidentRecord:
        incident.validate()
        try:
            return await asyncio.to_thread(self._create, incident)
        except IntegrityError:
            if incident.idempotency_key:
                existing = await self.find_by_idempotency_key(incident.idempotency_key)
                if existing is not None:
                    return existing
            raise ResourceConflictError(
                "An incident with this ID or idempotency key already exists.",
                details={"incident_id": incident.incident_id},
            )

    def _create(self, incident: IncidentRecord) -> IncidentRecord:
        with self._sessions.begin() as session:
            repository = IncidentRepository(session)
            if incident.idempotency_key:
                duplicate = session.scalar(
                    select(Incident).where(
                        Incident.api_idempotency_key == incident.idempotency_key
                    )
                )
                if duplicate is not None:
                    return _incident_payload(duplicate)

            entity_id = _stable_uuid("incident", incident.incident_id)
            if repository.get(entity_id) is not None:
                raise ResourceConflictError(
                    "An incident with this ID already exists.",
                    details={"incident_id": incident.incident_id},
                )

            entity = Incident(**_incident_values(incident))
            repository.add(entity, flush=True)
        return incident

    async def get(self, incident_id: str) -> IncidentRecord | None:
        return await asyncio.to_thread(self._get, incident_id)

    def _get(self, incident_id: str) -> IncidentRecord | None:
        with self._sessions.begin() as session:
            entity = IncidentRepository(session).get(
                _stable_uuid("incident", incident_id)
            )
            if entity is None or not entity.api_managed:
                return None
            return _incident_payload(entity)

    async def list(
        self,
        query: IncidentQuery,
    ) -> tuple[tuple[IncidentRecord, ...], int]:
        query.validate()
        return await asyncio.to_thread(self._list, query)

    def _list(
        self,
        query: IncidentQuery,
    ) -> tuple[tuple[IncidentRecord, ...], int]:
        statement = select(Incident).where(Incident.api_managed.is_(True))
        if query.tenant_id is not None:
            statement = statement.where(
                Incident.tenant_id == _stable_uuid("tenant", query.tenant_id)
            )
        if query.status is not None:
            statement = statement.where(Incident.status == query.status)
        if query.severity is not None:
            statement = statement.where(Incident.severity == query.severity)
        if query.created_after is not None:
            statement = statement.where(
                Incident.detected_at >= query.created_after
            )
        if query.created_before is not None:
            statement = statement.where(
                Incident.detected_at <= query.created_before
            )

        with self._sessions.begin() as session:
            entities = session.scalars(statement).all()
            records = [_incident_payload(entity) for entity in entities]

        if query.service is not None:
            records = [item for item in records if item.service == query.service]
        if query.environment is not None:
            records = [
                item for item in records if item.environment == query.environment
            ]
        records.sort(
            key=lambda item: (item.created_at, item.incident_id),
            reverse=query.descending,
        )
        total = len(records)
        page = records[query.offset : query.offset + query.limit]
        return tuple(page), total

    async def update(
        self,
        incident: IncidentRecord,
        *,
        expected_version: int,
    ) -> IncidentRecord:
        incident.validate()
        return await asyncio.to_thread(self._update, incident, expected_version)

    def _update(
        self,
        incident: IncidentRecord,
        expected_version: int,
    ) -> IncidentRecord:
        with self._sessions.begin() as session:
            repository = IncidentRepository(session)
            entity = repository.get(_stable_uuid("incident", incident.incident_id))
            if entity is None or not entity.api_managed:
                raise IncidentNotFoundAPIError(
                    details={"incident_id": incident.incident_id}
                )
            if entity.version != expected_version:
                raise ResourceConflictError(
                    "The incident version changed.",
                    code="version_conflict",
                    details={
                        "incident_id": incident.incident_id,
                        "expected_version": expected_version,
                        "actual_version": entity.version,
                    },
                )
            for name, value in _incident_values(incident).items():
                setattr(entity, name, value)
            session.flush()
        return incident

    async def find_by_idempotency_key(self, key: str) -> IncidentRecord | None:
        normalized = str(key).strip()
        if not normalized:
            return None
        return await asyncio.to_thread(self._find_by_idempotency_key, normalized)

    def _find_by_idempotency_key(self, key: str) -> IncidentRecord | None:
        with self._sessions.begin() as session:
            entity = session.scalar(
                select(Incident).where(
                    Incident.api_idempotency_key == key,
                    Incident.api_managed.is_(True),
                )
            )
            return _incident_payload(entity) if entity is not None else None

    async def close(self) -> None:
        return None


def _remediation_payload(row: Remediation) -> RemediationPlanRecord:
    payload = row.metadata_payload.get("api_record")
    if not isinstance(payload, dict):
        raise RuntimeError("API-managed remediation is missing its record payload")

    steps = tuple(
        RemediationStepRecord(
            step_id=step["step_id"],
            action=step["action"],
            provider=step["provider"],
            target=step["target"],
            parameters=step["parameters"],
            timeout_seconds=step["timeout_seconds"],
            reversible=step["reversible"],
            rollback_action=step.get("rollback_action"),
            state=step.get("state", "pending"),
            started_at=_parse_datetime(step.get("started_at")),
            completed_at=_parse_datetime(step.get("completed_at")),
            error_code=step.get("error_code"),
            error_message=step.get("error_message"),
            attributes=step.get("attributes", {}),
        )
        for step in payload["steps"]
    )
    return RemediationPlanRecord(
        plan_id=payload["plan_id"],
        incident_id=payload["incident_id"],
        decision_id=payload["decision_id"],
        title=payload["title"],
        description=payload.get("description"),
        risk=payload["risk"],
        status=payload["status"],
        dry_run=payload["dry_run"],
        approval_required=payload["approval_required"],
        steps=steps,
        created_at=_parse_datetime(payload["created_at"]),
        updated_at=_parse_datetime(payload["updated_at"]),
        version=row.version,
        tenant_id=payload.get("tenant_id"),
        created_by=payload.get("created_by"),
        correlation_id=payload.get("correlation_id"),
        idempotency_key=payload.get("idempotency_key"),
        approved_by=payload.get("approved_by"),
        approved_at=_parse_datetime(payload.get("approved_at")),
        rejection_reason=payload.get("rejection_reason"),
        execution_id=payload.get("execution_id"),
        execution_idempotency_key=payload.get("execution_idempotency_key"),
        attributes=payload.get("attributes", {}),
    )


def _remediation_values(record: RemediationPlanRecord) -> dict[str, Any]:
    payload = _json_value(record)
    return {
        "id": _stable_uuid("remediation", record.plan_id),
        "tenant_id": _stable_uuid("tenant", record.tenant_id),
        "version": record.version,
        "resource_id": None,
        "incident_id": _stable_uuid("incident", record.incident_id),
        "remediation_type": "api_plan",
        "title": record.title,
        "description": record.description,
        "reason": record.description,
        "status": record.status,
        "execution_mode": "manual",
        "risk": record.risk,
        "approval_required": record.approval_required,
        "approved": record.status == "approved",
        "approved_by": record.approved_by,
        "approved_at": record.approved_at,
        "rejection_reason": record.rejection_reason,
        "action_name": record.steps[0].action,
        "action_parameters": {"steps": payload["steps"]},
        "proposed_at": record.created_at,
        "execution_id": _optional_uuid(record.execution_id),
        "rollback_supported": any(step.reversible for step in record.steps),
        "rollback_action": next(
            (step.rollback_action for step in record.steps if step.rollback_action),
            None,
        ),
        "metadata_payload": {"api_record": payload},
        "api_managed": True,
        "api_idempotency_key": record.idempotency_key or None,
        "api_decision_id": record.decision_id,
        "api_execution_idempotency_key": record.execution_idempotency_key or None,
    }


class SQLAlchemyRemediationRepository:
    """Persist API remediation plans through the SQL remediation repository."""

    def __init__(self, sessions: sessionmaker | None = None) -> None:
        self._sessions = sessions or get_session_factory()

    async def create(self, plan: RemediationPlanRecord) -> RemediationPlanRecord:
        plan.validate()
        try:
            return await asyncio.to_thread(self._create, plan)
        except IntegrityError:
            if plan.idempotency_key:
                existing = await self.find_by_idempotency_key(plan.idempotency_key)
                if existing is not None:
                    return existing
            raise ResourceConflictError(
                "A remediation plan with this ID or idempotency key already exists.",
                details={"plan_id": plan.plan_id},
            )

    def _create(self, plan: RemediationPlanRecord) -> RemediationPlanRecord:
        with self._sessions.begin() as session:
            repository = RemediationRepository(session)
            if plan.idempotency_key:
                duplicate = session.scalar(
                    select(Remediation).where(
                        Remediation.api_idempotency_key == plan.idempotency_key
                    )
                )
                if duplicate is not None:
                    return _remediation_payload(duplicate)

            entity_id = _stable_uuid("remediation", plan.plan_id)
            if repository.get(entity_id) is not None:
                raise ResourceConflictError(
                    "A remediation plan with this ID already exists.",
                    details={"plan_id": plan.plan_id},
                )

            repository.add(Remediation(**_remediation_values(plan)), flush=True)
        return plan

    async def get(self, plan_id: str) -> RemediationPlanRecord | None:
        return await asyncio.to_thread(self._get, plan_id)

    def _get(self, plan_id: str) -> RemediationPlanRecord | None:
        with self._sessions.begin() as session:
            entity = RemediationRepository(session).get(
                _stable_uuid("remediation", plan_id)
            )
            if entity is None or not entity.api_managed:
                return None
            return _remediation_payload(entity)

    async def list(
        self,
        *,
        incident_id: str | None = None,
        decision_id: str | None = None,
        status: str | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[tuple[RemediationPlanRecord, ...], int]:
        return await asyncio.to_thread(
            self._list,
            incident_id,
            decision_id,
            status,
            tenant_id,
            limit,
            offset,
        )

    def _list(
        self,
        incident_id: str | None,
        decision_id: str | None,
        status: str | None,
        tenant_id: str | None,
        limit: int,
        offset: int,
    ) -> tuple[tuple[RemediationPlanRecord, ...], int]:
        if limit < 1 or limit > 1000 or offset < 0:
            raise ValueError("invalid remediation pagination")
        statement = select(Remediation).where(Remediation.api_managed.is_(True))
        if incident_id is not None:
            statement = statement.where(
                Remediation.incident_id == _stable_uuid("incident", incident_id)
            )
        if decision_id is not None:
            statement = statement.where(Remediation.api_decision_id == decision_id)
        if status is not None:
            statement = statement.where(Remediation.status == status)
        if tenant_id is not None:
            statement = statement.where(
                Remediation.tenant_id == _stable_uuid("tenant", tenant_id)
            )

        with self._sessions.begin() as session:
            entities = session.scalars(statement).all()
            records = [_remediation_payload(entity) for entity in entities]
        records.sort(
            key=lambda item: (item.created_at, item.plan_id),
            reverse=True,
        )
        total = len(records)
        return tuple(records[offset : offset + limit]), total

    async def update(
        self,
        plan: RemediationPlanRecord,
        *,
        expected_version: int,
    ) -> RemediationPlanRecord:
        plan.validate()
        return await asyncio.to_thread(self._update, plan, expected_version)

    def _update(
        self,
        plan: RemediationPlanRecord,
        expected_version: int,
    ) -> RemediationPlanRecord:
        with self._sessions.begin() as session:
            repository = RemediationRepository(session)
            entity = repository.get(_stable_uuid("remediation", plan.plan_id))
            if entity is None or not entity.api_managed:
                raise RemediationNotFoundAPIError(details={"plan_id": plan.plan_id})
            if entity.version != expected_version:
                raise RemediationVersionConflictAPIError(
                    details={
                        "plan_id": plan.plan_id,
                        "expected_version": expected_version,
                        "actual_version": entity.version,
                    }
                )
            for name, value in _remediation_values(plan).items():
                setattr(entity, name, value)
            session.flush()
        return plan

    async def find_by_idempotency_key(
        self,
        key: str,
    ) -> RemediationPlanRecord | None:
        normalized = str(key).strip()
        if not normalized:
            return None
        return await asyncio.to_thread(self._find_by_key, normalized, False)

    async def find_by_execution_key(
        self,
        key: str,
    ) -> RemediationPlanRecord | None:
        normalized = str(key).strip()
        if not normalized:
            return None
        return await asyncio.to_thread(self._find_by_key, normalized, True)

    def _find_by_key(self, key: str, execution: bool) -> RemediationPlanRecord | None:
        column = (
            Remediation.api_execution_idempotency_key
            if execution
            else Remediation.api_idempotency_key
        )
        with self._sessions.begin() as session:
            entity = session.scalar(
                select(Remediation).where(
                    column == key,
                    Remediation.api_managed.is_(True),
                )
            )
            return _remediation_payload(entity) if entity is not None else None

    async def close(self) -> None:
        return None


def _decision_payload(row: Decision) -> DecisionRecord:
    return DecisionRecord(
        decision_id=row.decision_id,
        incident_id=row.incident_id,
        candidate_action=row.candidate_action,
        target=row.target,
        status=row.status,
        reason=row.reason,
        risk=row.risk,
        approval_required=row.approval_required,
        policy_results=tuple(PolicyResult(**value) for value in row.policy_results),
        created_at=row.created_at,
        expires_at=row.expires_at,
        version=row.version,
        dry_run=row.dry_run,
        tenant_id=row.tenant_id,
        actor_id=row.actor_id,
        correlation_id=row.correlation_id,
        idempotency_key=row.idempotency_key,
        attributes=row.attributes,
    )


class SQLAlchemyDecisionRepository:
    """Persist policy decisions through the SQL decision repository."""

    def __init__(self, sessions: sessionmaker | None = None) -> None:
        self._sessions = sessions or get_session_factory()

    async def create(self, decision: DecisionRecord) -> DecisionRecord:
        decision.validate()
        try:
            return await asyncio.to_thread(self._create, decision)
        except IntegrityError:
            if decision.idempotency_key:
                existing = await self.find_by_idempotency_key(
                    decision.idempotency_key
                )
                if existing is not None:
                    return existing
            raise ResourceConflictError(
                "A decision with this ID or idempotency key already exists.",
                details={"decision_id": decision.decision_id},
            )

    def _create(self, decision: DecisionRecord) -> DecisionRecord:
        with self._sessions.begin() as session:
            repository = DecisionRepository(session)
            if decision.idempotency_key:
                duplicate = repository.find_by_idempotency_key(
                    decision.idempotency_key
                )
                if duplicate is not None:
                    return _decision_payload(duplicate)
            if repository.get_by_decision_id(decision.decision_id) is not None:
                raise ResourceConflictError(
                    "A decision with this ID already exists.",
                    details={"decision_id": decision.decision_id},
                )

            data = _json_value(decision)
            entity = Decision(
                decision_id=decision.decision_id,
                incident_id=decision.incident_id,
                tenant_id=decision.tenant_id,
                candidate_action=decision.candidate_action,
                target=data["target"],
                status=decision.status,
                reason=decision.reason,
                risk=decision.risk,
                approval_required=decision.approval_required,
                policy_results=data["policy_results"],
                created_at=decision.created_at,
                expires_at=decision.expires_at,
                version=decision.version,
                dry_run=decision.dry_run,
                actor_id=decision.actor_id,
                correlation_id=decision.correlation_id,
                idempotency_key=decision.idempotency_key or None,
                attributes=data["attributes"],
            )
            repository.add(entity, flush=True)
        return decision

    async def get(self, decision_id: str) -> DecisionRecord | None:
        return await asyncio.to_thread(self._get, decision_id)

    def _get(self, decision_id: str) -> DecisionRecord | None:
        with self._sessions.begin() as session:
            entity = DecisionRepository(session).get_by_decision_id(decision_id)
            return _decision_payload(entity) if entity is not None else None

    async def list(
        self,
        *,
        incident_id: str | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[tuple[DecisionRecord, ...], int]:
        return await asyncio.to_thread(
            self._list,
            incident_id,
            tenant_id,
            limit,
            offset,
        )

    def _list(
        self,
        incident_id: str | None,
        tenant_id: str | None,
        limit: int,
        offset: int,
    ) -> tuple[tuple[DecisionRecord, ...], int]:
        with self._sessions.begin() as session:
            entities, total = DecisionRepository(session).list_for_api(
                incident_id=incident_id,
                tenant_id=tenant_id,
                limit=limit,
                offset=offset,
            )
            return tuple(_decision_payload(row) for row in entities), total

    async def find_by_idempotency_key(self, key: str) -> DecisionRecord | None:
        normalized = str(key).strip()
        if not normalized:
            return None
        return await asyncio.to_thread(self._find_by_key, normalized)

    def _find_by_key(self, key: str) -> DecisionRecord | None:
        with self._sessions.begin() as session:
            entity = DecisionRepository(session).find_by_idempotency_key(key)
            return _decision_payload(entity) if entity is not None else None

    async def close(self) -> None:
        return None


__all__ = [
    "SQLAlchemyDecisionRepository",
    "SQLAlchemyIncidentRepository",
    "SQLAlchemyRemediationRepository",
]