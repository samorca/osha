from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from src.memory.schemas import Event


def correlate_events(events: Iterable[Event]) -> Dict[str, List[Event]]:
    groups: Dict[str, List[Event]] = defaultdict(list)
    for event in events:
        if event.correlation_id:
            groups[event.correlation_id].append(event)
    return dict(groups)
