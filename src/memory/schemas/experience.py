from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class Experience:
    incident_id: str
    action_id: str

    outcome: str

    experience_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    incident_signature: str | None = None

    successful: bool = False

    recovery_time_seconds: float | None = None
    reward: float | None = None

    context: dict[str, Any] = field(default_factory=dict)

    side_effects: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )