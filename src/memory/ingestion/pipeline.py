from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from src.memory.ingestion.validators.telemetry import (
    TelemetryValidationError,
)


_SENSITIVE_KEY = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|credential)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TelemetryRecord:
    metric: str
    value: Any
    source_id: str | None
    source_type: str | None
    labels: dict[str, Any]
    metadata: dict[str, Any]
    timestamp: datetime


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item) for item in value)
    return value


class TelemetryIngestionPipeline:
    def process(self, payload: Mapping[str, Any]) -> TelemetryRecord:
        if not isinstance(payload, Mapping):
            raise TypeError("Telemetry payload must be a mapping.")

        metric = payload.get("metric")
        if not isinstance(metric, str) or not metric.strip():
            raise TelemetryValidationError(
                "Telemetry metric must be a non-empty string."
            )
        if "value" not in payload:
            raise TelemetryValidationError("Telemetry payload is missing value.")

        raw_timestamp = payload.get("timestamp")
        if raw_timestamp is None:
            timestamp = datetime.now(timezone.utc)
        elif isinstance(raw_timestamp, datetime):
            timestamp = raw_timestamp
        elif isinstance(raw_timestamp, str):
            try:
                timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
            except ValueError as error:
                raise TelemetryValidationError(
                    "Telemetry timestamp is invalid."
                ) from error
        else:
            raise TelemetryValidationError(
                "Telemetry timestamp must be a datetime or ISO string."
            )

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp = timestamp.astimezone(timezone.utc)

        return TelemetryRecord(
            metric=metric.strip(),
            value=payload["value"],
            source_id=payload.get("source_id"),
            source_type=payload.get("source_type"),
            labels=dict(_redact(payload.get("labels", {}))),
            metadata=dict(_redact(payload.get("metadata", {}))),
            timestamp=timestamp,
        )
