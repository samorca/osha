from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class ResourceNode:
    resource_id: str
    resource_type: str
    name: str

    status: str = "unknown"

    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    discovered_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class ResourceRelation:
    source_id: str
    target_id: str
    relation_type: str

    relation_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    metadata: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
