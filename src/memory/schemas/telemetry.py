from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class Telemetry:
    metric: str
    value: float | int

    source_id: str
    source_type: str

    unit: str | None = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    telemetry_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)