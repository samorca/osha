from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import List


@dataclass(frozen=True)
class AuditEntry:
    actor: str
    action: str
    resource_id: str
    timestamp: datetime


class AuditLogger:
    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []
        self._lock = RLock()

    def record(
        self,
        actor: str,
        action: str,
        resource_id: str,
    ) -> AuditEntry:
        entry = AuditEntry(
            actor=actor,
            action=action,
            resource_id=resource_id,
            timestamp=datetime.now(timezone.utc),
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def list(self) -> List[AuditEntry]:
        with self._lock:
            return list(self._entries)
