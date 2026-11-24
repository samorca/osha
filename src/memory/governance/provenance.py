from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Dict, List


@dataclass(frozen=True)
class ProvenanceRecord:
    entity_id: str
    source: str
    action: str
    timestamp: datetime


class ProvenanceTracker:
    def __init__(self) -> None:
        self._history: Dict[str, List[ProvenanceRecord]] = {}
        self._lock = RLock()

    def record(
        self,
        entity_id: str,
        source: str,
        action: str,
    ) -> ProvenanceRecord:
        record = ProvenanceRecord(
            entity_id=entity_id,
            source=source,
            action=action,
            timestamp=datetime.now(timezone.utc),
        )
        with self._lock:
            self._history.setdefault(entity_id, []).append(record)
        return record

    def history(self, entity_id: str) -> List[ProvenanceRecord]:
        with self._lock:
            return list(self._history.get(entity_id, ()))
