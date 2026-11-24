from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List


@dataclass(frozen=True)
class RetentionPolicy:
    max_age_days: int

    def __post_init__(self) -> None:
        if self.max_age_days < 0:
            raise ValueError("max_age_days must be non-negative.")


def apply_retention(
    records: Iterable[Any],
    policy: RetentionPolicy,
    now: datetime | None = None,
) -> List[Any]:
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    cutoff = current_time - timedelta(days=policy.max_age_days)

    kept: List[Any] = []
    for record in records:
        timestamp = getattr(record, "timestamp", None)
        if not isinstance(timestamp, datetime):
            kept.append(record)
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if timestamp >= cutoff:
            kept.append(record)
    return kept
