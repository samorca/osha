from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any


def _timestamp(item: Any) -> datetime:
    value = getattr(item, "timestamp", None)
    if value is None:
        value = getattr(item, "created_at", None)
    if value is None:
        value = getattr(item, "detected_at", None)
    return value or datetime.min.replace(tzinfo=timezone.utc)


def recent_context(items: Iterable[Any], limit: int = 10) -> list[Any]:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise TypeError("limit must be an integer")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    records = list(items)
    records.sort(key=_timestamp, reverse=True)
    return records[:limit]