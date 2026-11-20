import math
from typing import Any


class TelemetryValidationError(ValueError):
    pass


REQUIRED_FIELDS = {
    "metric",
    "value",
    "source_id",
    "source_type",
}


def validate_telemetry(
    record: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TelemetryValidationError("Telemetry record must be a dictionary")

    missing = REQUIRED_FIELDS - record.keys()

    if missing:
        raise TelemetryValidationError(
            f"Missing telemetry fields: {sorted(missing)}"
        )

    if not isinstance(record["metric"], str):
        raise TelemetryValidationError(
            "metric must be a string"
        )

    if not isinstance(record["source_id"], str):
        raise TelemetryValidationError(
            "source_id must be a string"
        )

    if not isinstance(record["source_type"], str):
        raise TelemetryValidationError(
            "source_type must be a string"
        )

    if not isinstance(record["value"], (int, float)):
        raise TelemetryValidationError(
            "value must be numeric"
        )

    if isinstance(record["value"], bool) or not math.isfinite(record["value"]):
        raise TelemetryValidationError(
            "value must be a finite number"
        )

    if not record["metric"].strip():
        raise TelemetryValidationError("metric must not be empty")
    if not record["source_id"].strip():
        raise TelemetryValidationError("source_id must not be empty")
    if not record["source_type"].strip():
        raise TelemetryValidationError("source_type must not be empty")

    labels = record.get("labels", {})
    metadata = record.get("metadata", {})
    if not isinstance(labels, dict):
        raise TelemetryValidationError("labels must be a dictionary")
    if not isinstance(metadata, dict):
        raise TelemetryValidationError("metadata must be a dictionary")

    return dict(record)