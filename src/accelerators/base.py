from abc import ABC, abstractmethod


class Accelerator(ABC):
    """Base interface for all compute accelerator backends."""

    name: str = "unknown"

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def device_count(self) -> int:
        pass

    @abstractmethod
    def get_device(self, index: int = 0):
        pass

    @abstractmethod
    def device_name(self, index: int = 0) -> str:
        pass