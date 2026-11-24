from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, List, Tuple


def delete_where(
    records: Iterable[Any],
    predicate: Callable[[Any], bool],
) -> Tuple[List[Any], List[Any]]:
    kept: List[Any] = []
    deleted: List[Any] = []
    for record in records:
        (deleted if predicate(record) else kept).append(record)
    return kept, deleted
