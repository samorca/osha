"""Repository for persisted policy decisions."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models.decision import Decision
from .base import BaseRepository


class DecisionRepository(BaseRepository[Decision]):
    """Query and persist durable decision records."""

    model = Decision

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def get_by_decision_id(self, decision_id: str) -> Decision | None:
        statement = select(Decision).where(
            Decision.decision_id == decision_id
        )
        return self.session.scalar(statement)

    def find_by_idempotency_key(self, key: str) -> Decision | None:
        normalized = str(key).strip()
        if not normalized:
            return None

        statement = select(Decision).where(
            Decision.idempotency_key == normalized
        )
        return self.session.scalar(statement)

    def list_for_api(
        self,
        *,
        incident_id: str | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[tuple[Decision, ...], int]:
        self._validate_pagination(offset=offset, limit=limit)

        filters = []
        if incident_id is not None:
            filters.append(Decision.incident_id == incident_id)
        if tenant_id is not None:
            filters.append(Decision.tenant_id == tenant_id)

        total = self.session.scalar(
            select(func.count(Decision.id)).where(*filters)
        ) or 0
        statement = (
            select(Decision)
            .where(*filters)
            .order_by(Decision.created_at.desc(), Decision.decision_id.desc())
            .offset(offset)
            .limit(limit)
        )
        records = tuple(self.session.scalars(statement).all())
        return records, int(total)


__all__ = ["DecisionRepository"]