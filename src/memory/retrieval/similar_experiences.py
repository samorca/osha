from collections.abc import Iterable
from typing import Any


def similar_experiences(
    experiences: Iterable[Any],
    incident_signature: str | None = None,
    successful: bool | None = None,
    limit: int = 10,
) -> list[Any]:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise TypeError("limit must be an integer")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if successful is not None and not isinstance(successful, bool):
        raise TypeError("successful must be a boolean")

    needle = incident_signature.lower().strip() if incident_signature else None
    results = []
    for experience in experiences:
        if successful is not None and experience.successful != successful:
            continue
        signature = getattr(experience, "incident_signature", None)
        if needle and not (
            signature and needle in str(signature).lower()
        ):
            continue
        results.append(experience)

    results.sort(
        key=lambda item: (
            item.reward if item.reward is not None else float("-inf"),
            item.created_at,
        ),
        reverse=True,
    )
    return results[:limit]