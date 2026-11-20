from abc import ABC, abstractmethod
from typing import Any, Iterable


class MemoryStore(ABC):
    @abstractmethod
    def put(self, item: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get(self, item_id: str) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def list(self, limit: int | None = None) -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, item_id: str) -> bool:
        raise NotImplementedError
