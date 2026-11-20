from collections.abc import Iterable
from typing import Any


def delete_where(
    records: Iterable[Any],
    predicate,
) -> tuple[list[Any], list[Any]]:
    if not callable(predicate):
        raise TypeError("predicate must be callable")
    kept = []
    deleted = []
    for record in records:
        if predicate(record):
            deleted.append(record)
        else:
            kept.append(record)
    return kept, deleted