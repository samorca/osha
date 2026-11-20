from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class AuditEntry:
    actor: str
    action: str
    resource: str
    outcome: str = "success"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    audit_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        self._lock = RLock()

    def record(
        self,
        actor: str,
        action: str,
        resource: str,
        outcome: str = "success",
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            metadata=metadata or {},
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def list(self, limit: int | None = None) -> list[AuditEntry]:
        with self._lock:
            entries = list(self._entries)
        entries.reverse()
        return entries[:limit] if limit is not None else entries