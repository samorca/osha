from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.errors import ResourceConflictError
from src.api.persistence import (
    SQLAlchemyDecisionRepository,
    SQLAlchemyIncidentRepository,
    SQLAlchemyRemediationRepository,
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
from src.database.base import Base


@pytest.fixture
def sessions() -> sessionmaker:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    Base.metadata.drop_all(engine)
    engine.dispose()


def _incident(now: datetime, *, incident_id: str = "inc-test") -> IncidentRecord:
    return IncidentRecord(
        incident_id=incident_id,
        title="API latency elevated",
        summary="Latency exceeded the service threshold.",
        severity="high",
        status="open",
        service="api",
        environment="test",
        signals=(
            IncidentSignalRecord(
                signal_id="signal-1",
                source="collector",
                signal_type="latency",
                observed_at=now,
                severity="high",
                payload={"value_ms": 450},
                attributes={"region": "west"},
            ),
        ),
        attributes={"owner": "platform"},
        created_at=now,
        updated_at=now,
        tenant_id="tenant-test",
        idempotency_key="incident-key-1",
    )


def test_incident_repository_round_trips_records_and_filters(sessions) -> None:
    now = datetime.now(timezone.utc)
    repository = SQLAlchemyIncidentRepository(sessions)
    incident = _incident(now)

    stored = asyncio.run(repository.create(incident))
    loaded = asyncio.run(repository.get(incident.incident_id))
    records, total = asyncio.run(
        repository.list(
            IncidentQuery(
                tenant_id="tenant-test",
                service="api",
                environment="test",
            )
        )
    )

    assert stored == incident
    assert loaded == incident
    assert records == (incident,)
    assert total == 1


def test_incident_repository_enforces_idempotency_and_versions(sessions) -> None:
    now = datetime.now(timezone.utc)
    repository = SQLAlchemyIncidentRepository(sessions)
    incident = _incident(now)
    asyncio_run = asyncio.run

    asyncio_run(repository.create(incident))
    duplicate = _incident(now, incident_id="inc-retry")
    assert asyncio_run(repository.create(duplicate)) == incident

    updated = IncidentRecord(
        **{
            **incident.__dict__,
            "status": "investigating",
            "updated_at": now + timedelta(seconds=1),
            "version": 2,
        }
    )
    assert asyncio_run(repository.update(updated, expected_version=1)) == updated

    with pytest.raises(ResourceConflictError):
        asyncio_run(repository.update(updated, expected_version=1))


def test_remediation_repository_round_trips_plan_steps(sessions) -> None:
    now = datetime.now(timezone.utc)
    plan = RemediationPlanRecord(
        plan_id="rem-test",
        incident_id="inc-test",
        decision_id="dec-test",
        title="Restart the affected service",
        description="Dry-run recovery plan.",
        risk="medium",
        status="pending_approval",
        dry_run=True,
        approval_required=True,
        steps=(
            RemediationStepRecord(
                step_id="step-1",
                action="restart",
                provider="kubernetes",
                target={"namespace": "test", "name": "api"},
                parameters={"grace_period": 30},
                timeout_seconds=60,
                reversible=False,
                rollback_action=None,
            ),
        ),
        created_at=now,
        updated_at=now,
        tenant_id="tenant-test",
        idempotency_key="remediation-key-1",
    )
    repository = SQLAlchemyRemediationRepository(sessions)
    asyncio_run = asyncio.run

    assert asyncio_run(repository.create(plan)) == plan
    assert asyncio_run(repository.get(plan.plan_id)) == plan
    records, total = asyncio_run(
        repository.list(
            incident_id=plan.incident_id,
            decision_id=plan.decision_id,
            tenant_id=plan.tenant_id,
        )
    )

    assert records == (plan,)
    assert total == 1
    assert asyncio_run(
        repository.find_by_idempotency_key("remediation-key-1")
    ) == plan


def test_decision_repository_round_trips_policy_results_and_idempotency(sessions) -> None:
    now = datetime.now(timezone.utc)
    decision = DecisionRecord(
        decision_id="dec-test",
        incident_id="inc-test",
        candidate_action="restart",
        target={"service": "api"},
        status="allowed",
        reason="Policy checks passed.",
        risk="low",
        approval_required=False,
        policy_results=(
            PolicyResult(
                policy="maintenance-window",
                allowed=True,
                reason="Within the approved window.",
            ),
        ),
        created_at=now,
        expires_at=now + timedelta(minutes=5),
        tenant_id="tenant-test",
        actor_id="operator-1",
        idempotency_key="decision-key-1",
        attributes={"source": "test"},
    )
    repository = SQLAlchemyDecisionRepository(sessions)
    asyncio_run = asyncio.run

    assert asyncio_run(repository.create(decision)) == decision
    loaded = asyncio_run(repository.get(decision.decision_id))
    assert loaded is not None
    assert loaded.decision_id == decision.decision_id
    assert loaded.policy_results == decision.policy_results
    assert loaded.target == decision.target
    assert loaded.idempotency_key == decision.idempotency_key

    records, total = asyncio_run(
        repository.list(incident_id="inc-test", tenant_id="tenant-test")
    )
    assert len(records) == 1
    assert total == 1
    assert asyncio_run(
        repository.find_by_idempotency_key("decision-key-1")
    ) == loaded