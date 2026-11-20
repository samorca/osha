from typing import Any
from datetime import datetime

from ...schemas.telemetry import Telemetry


def normalize_telemetry(
    record: dict[str, Any],
) -> Telemetry:
    timestamp = record.get("timestamp")
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be an ISO-8601 datetime") from exc
    elif timestamp is not None and not isinstance(timestamp, datetime):
        raise TypeError("timestamp must be a datetime or ISO-8601 string")

    values = dict(
        metric=record["metric"],
        value=record["value"],
        source_id=record["source_id"],
        source_type=record["source_type"],
        unit=record.get("unit"),
        labels=record.get("labels", {}),
        metadata=record.get("metadata", {}),
    )
    if timestamp is not None:
        values["timestamp"] = timestamp
    if "telemetry_id" in record:
        values["telemetry_id"] = record["telemetry_id"]
    return Telemetry(**values)