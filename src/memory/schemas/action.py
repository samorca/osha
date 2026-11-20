from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class ActionStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class ActionRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Action:
    action_type: str
    target_id: str

    incident_id: str | None = None

    action_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    status: ActionStatus = ActionStatus.PROPOSED
    risk: ActionRisk = ActionRisk.LOW

    requested_by: str = "osha"

    reason: str | None = None
    policy_id: str | None = None

    confidence: float | None = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    started_at: datetime | None = None
    completed_at: datetime | None = None

    parameters: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)