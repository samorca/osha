from __future__ import annotations

import socket
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hostname() -> str:
    return socket.gethostname()


def build_record(
    *,
    signal_type: str,
    payload: dict[str, Any],
    timestamp: str | None = None,
    environment_id: str = "default",
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "record_type": "linux_telemetry",
        "provider": "linux",
        "signal_type": signal_type,
        "timestamp": (timestamp if timestamp else utc_now()),
        "environment_id": (environment_id),
        "host_name": hostname(),
        "payload": payload,
        "collected_at": utc_now(),
    }
