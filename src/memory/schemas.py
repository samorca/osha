from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


class EventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class Event:
    event_type: str
    source_id: str
    source_type: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str | None = None


@dataclass
class Incident:
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus = IncidentStatus.OPEN
    incident_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class Experience:
    incident_id: str
    action_id: str
    outcome: str
    incident_signature: str
    successful: bool
    reward: float
    experience_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class ResourceNode:
    resource_id: str
    resource_type: str
    name: str


@dataclass
class ResourceRelation:
    source_id: str
    target_id: str
    relation_type: str
