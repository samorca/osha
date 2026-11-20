from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class EventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Event:
    event_type: str
    source_id: str
    source_type: str
    message: str

    severity: EventSeverity = EventSeverity.INFO

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    event_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    correlation_id: str | None = None

    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)