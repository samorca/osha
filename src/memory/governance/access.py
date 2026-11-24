from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Iterable


class AccessController:
    def __init__(self) -> None:
        self._grants: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._lock = RLock()

    def grant(
        self,
        principal: str,
        resource_id: str,
        actions: Iterable[str],
    ) -> None:
        with self._lock:
            self._grants[(principal, resource_id)].update(actions)

    def is_allowed(
        self,
        principal: str,
        resource_id: str,
        action: str,
    ) -> bool:
        with self._lock:
            return action in self._grants.get(
                (principal, resource_id),
                set(),
            )
