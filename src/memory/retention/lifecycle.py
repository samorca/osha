from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable


@dataclass(frozen=True)
class RetentionPolicy:
    max_age_days: int = 90
    max_items: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_age_days, int) or isinstance(self.max_age_days, bool):
            raise TypeError("max_age_days must be an integer")
        if self.max_age_days < 0:
            raise ValueError("max_age_days must be non-negative")
        if self.max_items is not None:
            if not isinstance(self.max_items, int) or isinstance(self.max_items, bool):
                raise TypeError("max_items must be an integer")
            if self.max_items < 0:
                raise ValueError("max_items must be non-negative")


def get_record_timestamp(record: Any) -> datetime | None:
    for name in ("timestamp", "created_at", "detected_at", "discovered_at"):
        value = getattr(record, name, None)
        if isinstance(value, datetime):
            return value
    if isinstance(record, dict):
        for name in ("timestamp", "created_at", "detected_at", "discovered_at"):
            value = record.get(name)
            if isinstance(value, datetime):
                return value
    return None


def apply_retention(
    records: Iterable[Any],
    policy: RetentionPolicy,
    now: datetime | None = None,
) -> list[Any]:
    current = now or datetime.now(timezone.utc)
    cutoff = current - timedelta(days=policy.max_age_days)
    kept = []

    for record in records:
        timestamp = get_record_timestamp(record)
        if timestamp is None:
            kept.append(record)
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if timestamp >= cutoff:
            kept.append(record)

    if policy.max_items is not None:
        kept.sort(
            key=lambda record: get_record_timestamp(record)
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        kept = kept[:policy.max_items]
    return kept