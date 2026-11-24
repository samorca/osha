from __future__ import annotations

from collections import deque
from threading import RLock
from typing import Any, Deque, Dict, List, Mapping


class MetricsReceiver:
    def __init__(self) -> None:
        self._queue: Deque[Dict[str, Any]] = deque()
        self._lock = RLock()

    def submit(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("Metrics payload must be a mapping.")
        with self._lock:
            self._queue.append(dict(payload))

    def receive(self) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._queue)
            self._queue.clear()
        return records
