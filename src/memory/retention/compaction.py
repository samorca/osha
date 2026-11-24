from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, List


def compact_records(
    records: Iterable[Any],
    key_fields: Sequence[str],
) -> List[Any]:
    compacted: List[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        key = tuple(getattr(record, field, None) for field in key_fields)
        try:
            duplicate = key in seen
        except TypeError:
            key = tuple(repr(value) for value in key)
            duplicate = key in seen
        if not duplicate:
            seen.add(key)
            compacted.append(record)
    return compacted
