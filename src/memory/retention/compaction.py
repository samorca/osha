from collections.abc import Iterable
from typing import Any


def compact_records(records: Iterable[Any], key_fields: tuple[str, ...]) -> list[Any]:
    if not isinstance(key_fields, tuple):
        raise TypeError("key_fields must be a tuple")
    if not key_fields:
        raise ValueError("key_fields must not be empty")

    compacted = []
    seen = set()
    for record in records:
        key = tuple(
            record.get(field) if isinstance(record, dict) else getattr(record, field, None)
            for field in key_fields
        )
        if key in seen:
            continue
        seen.add(key)
        compacted.append(record)
    return compacted