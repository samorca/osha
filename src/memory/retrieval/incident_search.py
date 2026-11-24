from __future__ import annotations

from collections.abc import Iterable
from typing import List

from src.memory.schemas import Incident


def incident_search(
    incidents: Iterable[Incident],
    *,
    query: str = "",
    severity: str | None = None,
    status: str | None = None,
) -> List[Incident]:
    needle = query.casefold().strip()
    severity_value = getattr(severity, "value", severity)
    status_value = getattr(status, "value", status)
    matches = []
    for incident in incidents:
        if (
            needle
            and needle not in (incident.title + " " + incident.description).casefold()
        ):
            continue
        if severity_value is not None and (
            incident.severity.value.casefold() != str(severity_value).casefold()
        ):
            continue
        if status_value is not None and (
            incident.status.value.casefold() != str(status_value).casefold()
        ):
            continue
        matches.append(incident)
    return matches
