from abc import ABC, abstractmethod
from typing import Any


class Receiver(ABC):
    """Base interface for OSHA telemetry receivers."""

    @abstractmethod
    def receive(self) -> list[dict[str, Any]]:
        """Return raw records collected from a source."""
        raise NotImplementedError