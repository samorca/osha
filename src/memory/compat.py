from __future__ import annotations

from threading import RLock
from typing import Any, Dict, List


class InMemoryStore:
    """Thread-safe in-memory CRUD store for resource-style records."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: Dict[str, Any] = {}

    @staticmethod
    def _record_id(record: Any) -> str:
        for attribute in (
            "resource_id",
            "event_id",
            "incident_id",
            "experience_id",
            "id",
        ):
            value = getattr(record, attribute, None)
            if value is not None:
                return str(value)
        raise TypeError("Record must expose a stable identifier.")

    def put(self, record: Any) -> Any:
        record_id = self._record_id(record)
        with self._lock:
            self._records[record_id] = record
        return record

    def get(self, record_id: str) -> Any:
        with self._lock:
            return self._records.get(record_id)

    def list(self) -> List[Any]:
        with self._lock:
            return list(self._records.values())

    def delete(self, record_id: str) -> bool:
        with self._lock:
            return self._records.pop(record_id, None) is not None
