from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True)
class AccessPolicy:
    principal: str
    resource: str
    actions: frozenset[str]


class AccessController:
    def __init__(self) -> None:
        self._policies: dict[tuple[str, str], frozenset[str]] = {}
        self._lock = RLock()

    def grant(self, principal: str, resource: str, actions: frozenset[str] | set[str]) -> None:
        if not principal or not resource:
            raise ValueError("principal and resource must be non-empty")
        if not isinstance(actions, (frozenset, set)):
            raise TypeError("actions must be a set or frozenset")
        if not all(isinstance(action, str) and action for action in actions):
            raise ValueError("actions must contain non-empty strings")
        with self._lock:
            key = (principal, resource)
            existing = self._policies.get(key, frozenset())
            self._policies[key] = existing.union(actions)

    def revoke(self, principal: str, resource: str, actions: frozenset[str] | set[str] | None = None) -> None:
        with self._lock:
            key = (principal, resource)
            if actions is None:
                self._policies.pop(key, None)
                return
            existing = self._policies.get(key, frozenset())
            self._policies[key] = existing.difference(actions)

    def is_allowed(self, principal: str, resource: str, action: str) -> bool:
        with self._lock:
            allowed = self._policies.get((principal, resource), frozenset())
            return action in allowed or "*" in allowed