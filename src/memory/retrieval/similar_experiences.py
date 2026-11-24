from __future__ import annotations

from collections.abc import Iterable
from typing import List

from src.memory.schemas import Experience


def similar_experiences(
    experiences: Iterable[Experience],
    *,
    incident_signature: str,
    successful: bool | None = None,
) -> List[Experience]:
    query = incident_signature.casefold()
    matches = []
    for experience in experiences:
        if query not in experience.incident_signature.casefold():
            continue
        if successful is not None and experience.successful != successful:
            continue
        matches.append(experience)
    return matches
