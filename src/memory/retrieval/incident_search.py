from collections.abc import Iterable
from typing import Any


def incident_search(
    incidents: Iterable[Any],
    query: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[Any]:
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("limit must be an integer")
        if limit < 0:
            raise ValueError("limit must be non-negative")

    needle = query.lower().strip() if query else None
    results = []
    for incident in incidents:
        if severity is not None:
            actual = getattr(incident.severity, "value", incident.severity)
            if str(actual).lower() != str(severity).lower():
                continue
        if status is not None:
            actual = getattr(incident.status, "value", incident.status)
            if str(actual).lower() != str(status).lower():
                continue
        if needle:
            searchable = " ".join(
                str(value)
                for value in (
                    getattr(incident, "title", ""),
                    getattr(incident, "description", ""),
                    getattr(incident, "suspected_cause", ""),
                    getattr(incident, "metadata", {}),
                )
            ).lower()
            if needle not in searchable:
                continue
        results.append(incident)
    return results[:limit]