from threading import RLock
from typing import Any, Iterable

from .base import MemoryStore


class InMemoryStore(MemoryStore):
    """Thread-safe, process-local memory store."""

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}
        self._lock = RLock()

    @staticmethod
    def _item_id(item: Any) -> str:
        if isinstance(item, dict):
            for key in ("telemetry_id", "event_id", "incident_id", "action_id",
                        "experience_id", "resource_id", "relation_id", "id"):
                if key in item:
                    return str(item[key])
            raise ValueError("Dictionary item has no supported identifier")

        for key in ("telemetry_id", "event_id", "incident_id", "action_id",
                    "experience_id", "resource_id", "relation_id", "id"):
            if hasattr(item, key):
                return str(getattr(item, key))
        raise ValueError(f"{type(item).__name__} has no supported identifier")

    def put(self, item: Any) -> Any:
        item_id = self._item_id(item)
        with self._lock:
            self._items[item_id] = item
        return item

    def get(self, item_id: str) -> Any | None:
        if not isinstance(item_id, str):
            raise TypeError("item_id must be a string")
        with self._lock:
            return self._items.get(item_id)

    def list(self, limit: int | None = None) -> Iterable[Any]:
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool):
                raise TypeError("limit must be an integer")
            if limit < 0:
                raise ValueError("limit must be non-negative")
        with self._lock:
            items = list(self._items.values())
        return items[:limit] if limit is not None else items

    def delete(self, item_id: str) -> bool:
        if not isinstance(item_id, str):
            raise TypeError("item_id must be a string")
        with self._lock:
            return self._items.pop(item_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
