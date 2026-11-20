from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class Incident:
    title: str
    description: str

    severity: IncidentSeverity

    incident_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    status: IncidentStatus = IncidentStatus.OPEN

    detected_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    resolved_at: datetime | None = None

    affected_resources: list[str] = field(default_factory=list)
    related_events: list[str] = field(default_factory=list)

    symptoms: list[str] = field(default_factory=list)

    suspected_cause: str | None = None
    confidence: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)