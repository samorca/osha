from typing import Any

from .base import Receiver


class MetricsReceiver(Receiver):
    """
    Simple receiver for metrics supplied by another OSHA component.

    Later, source-specific receivers can collect from nodes,
    accelerators, Kubernetes, OpenStack, and other systems.
    """

    def __init__(self) -> None:
        self._buffer: list[dict[str, Any]] = []

    def submit(self, record: dict[str, Any]) -> None:
        if not isinstance(record, dict):
            raise TypeError("record must be a dictionary")
        self._buffer.append(record)

    def receive(self) -> list[dict[str, Any]]:
        records = self._buffer.copy()
        self._buffer.clear()

        return records