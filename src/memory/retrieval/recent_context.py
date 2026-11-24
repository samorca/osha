from __future__ import annotations

from collections.abc import Iterable
from typing import Any, List


def recent_context(records: Iterable[Any], limit: int = 10) -> List[Any]:
    if limit < 0:
        raise ValueError("limit must be non-negative.")
    ordered = sorted(
        records,
        key=lambda record: getattr(record, "timestamp", None),
        reverse=True,
    )
    return ordered[:limit]
