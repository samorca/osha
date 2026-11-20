from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ProvenanceRecord:
    artifact_id: str
    source: str
    operation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


class ProvenanceTracker:
    def __init__(self) -> None:
        self._records: dict[str, list[ProvenanceRecord]] = {}
        self._lock = RLock()

    def record(
        self,
        artifact_id: str,
        source: str,
        operation: str,
        metadata: dict[str, Any] | None = None,
    ) -> ProvenanceRecord:
        if not all(isinstance(value, str) and value for value in (artifact_id, source, operation)):
            raise ValueError("artifact_id, source, and operation must be non-empty strings")
        record = ProvenanceRecord(
            artifact_id=artifact_id,
            source=source,
            operation=operation,
            metadata=metadata or {},
        )
        with self._lock:
            self._records.setdefault(artifact_id, []).append(record)
        return record

    def history(self, artifact_id: str) -> list[ProvenanceRecord]:
        with self._lock:
            return list(self._records.get(artifact_id, []))