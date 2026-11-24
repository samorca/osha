from __future__ import annotations

from typing import (
    Dict,
    Iterable,
    List,
    Optional,
)

from src.correlation.schema import (
    CorrelationGroup,
)

from src.memory.schema import (
    EpisodeMemory,
    VerificationStatus,
)

from src.memory.store import (
    MemoryQuery,
    MemoryStore,
)


def episode_from_group(
    group: CorrelationGroup,
    *,
    incident_id: Optional[str] = None,
    verification_status: str = (VerificationStatus.UNVERIFIED.value),
) -> EpisodeMemory:
    """
    Convert one correlation group into
    operational episode memory.

    A correlation episode is historical evidence.
    It does not automatically represent an
    incident.
    """

    group.finalize()

    episode = EpisodeMemory(
        environment_id=(group.environment_id),
        correlation_group_id=(group.group_id),
        record_ids=list(group.record_ids),
        resource_ids=list(group.resource_ids),
        providers=list(group.providers),
        signal_types=list(group.signal_types),
        started_at=(group.start_timestamp),
        ended_at=(group.end_timestamp),
        strongest_correlation_score=(group.strongest_score),
        average_correlation_score=(group.average_score),
        incident_id=incident_id,
        verification_status=(verification_status),
        source_ids=([group.group_id] if group.group_id else []),
        tags=["correlation_episode"],
        attributes={
            "candidate_ids": list(group.candidate_ids),
            "group_status": (group.status),
            "group_attributes": dict(group.attributes),
        },
    )

    return episode.finalize()


def find_episode_for_group(
    store: MemoryStore,
    group_id: str,
    *,
    environment_id: Optional[str] = None,
) -> Optional[EpisodeMemory]:
    """
    Retrieve previously stored memory for one
    correlation group.
    """

    query = MemoryQuery(
        memory_types=("episode",),
        environment_ids=((environment_id,) if environment_id else ()),
        source_ids=(group_id,),
        newest_first=True,
        limit=10,
    )

    for memory in store.query(query):
        if (
            isinstance(
                memory,
                EpisodeMemory,
            )
            and memory.correlation_group_id == group_id
        ):
            return memory

    return None


def merge_episode(
    existing: EpisodeMemory,
    incoming: EpisodeMemory,
) -> EpisodeMemory:
    """
    Merge repeated observations of the same
    correlation episode.

    This makes ingestion idempotent when a group
    appears in more than one processing run.
    """

    existing.record_ids = sorted(set(existing.record_ids) | set(incoming.record_ids))

    existing.resource_ids = sorted(
        set(existing.resource_ids) | set(incoming.resource_ids)
    )

    existing.providers = sorted(set(existing.providers) | set(incoming.providers))

    existing.signal_types = sorted(
        set(existing.signal_types) | set(incoming.signal_types)
    )

    existing.source_ids = sorted(set(existing.source_ids) | set(incoming.source_ids))

    existing.tags = sorted(set(existing.tags) | set(incoming.tags))

    existing.strongest_correlation_score = max(
        existing.strongest_correlation_score,
        incoming.strongest_correlation_score,
    )

    scores = [
        existing.average_correlation_score,
        incoming.average_correlation_score,
    ]

    existing.average_correlation_score = sum(scores) / len(scores)

    if incoming.started_at and (
        not existing.started_at or incoming.started_at < existing.started_at
    ):
        existing.started_at = incoming.started_at

    if incoming.ended_at and (
        not existing.ended_at or incoming.ended_at > existing.ended_at
    ):
        existing.ended_at = incoming.ended_at

    if incoming.incident_id:
        existing.incident_id = incoming.incident_id

    # Never automatically promote verification
    # merely because an episode was seen again.
    if (
        existing.verification_status == VerificationStatus.UNVERIFIED.value
        and incoming.verification_status != VerificationStatus.UNVERIFIED.value
    ):
        existing.verification_status = incoming.verification_status

    existing.attributes.update(incoming.attributes)

    existing.touch()

    return existing.finalize()


def remember_episode(
    store: MemoryStore,
    group: CorrelationGroup,
    *,
    incident_id: Optional[str] = None,
    verification_status: str = (VerificationStatus.UNVERIFIED.value),
) -> EpisodeMemory:
    """
    Store or update one correlation episode.
    """

    incoming = episode_from_group(
        group,
        incident_id=incident_id,
        verification_status=(verification_status),
    )

    existing = None

    if group.group_id:
        existing = find_episode_for_group(
            store,
            group.group_id,
            environment_id=(group.environment_id),
        )

    if existing is not None:
        incoming = merge_episode(
            existing,
            incoming,
        )

    stored = store.put(incoming)

    if not isinstance(
        stored,
        EpisodeMemory,
    ):
        raise TypeError("Memory store returned a non-EpisodeMemory object")

    return stored


def remember_episodes(
    store: MemoryStore,
    groups: Iterable[CorrelationGroup],
    *,
    incident_by_group: Optional[Dict[str, str]] = None,
) -> List[EpisodeMemory]:
    """
    Store a complete correlation batch.

    incident_by_group allows intelligence to link
    an episode to an incident without making
    every episode an incident.
    """

    incident_map = incident_by_group or {}

    output = []

    for group in groups:
        incident_id = incident_map.get(group.group_id or "")

        output.append(
            remember_episode(
                store,
                group,
                incident_id=(incident_id),
            )
        )

    return output


def list_episodes(
    store: MemoryStore,
    *,
    environment_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    active_only: bool = False,
    limit: Optional[int] = None,
) -> List[EpisodeMemory]:
    """
    Query episode memory.
    """

    query = MemoryQuery(
        memory_types=("episode",),
        environment_ids=((environment_id,) if environment_id else ()),
        resource_ids=((resource_id,) if resource_id else ()),
        active_only=active_only,
        limit=limit,
    )

    return [
        memory
        for memory in store.query(query)
        if isinstance(
            memory,
            EpisodeMemory,
        )
    ]


def episode_history_for_resource(
    store: MemoryStore,
    resource_id: str,
    *,
    environment_id: Optional[str] = None,
    limit: Optional[int] = 50,
) -> List[EpisodeMemory]:
    """
    Return operational episodes involving one
    resource.
    """

    return list_episodes(
        store,
        environment_id=(environment_id),
        resource_id=(resource_id),
        limit=limit,
    )


def episode_history_for_incident(
    store: MemoryStore,
    incident_id: str,
    *,
    environment_id: Optional[str] = None,
) -> List[EpisodeMemory]:
    """
    Return episodes linked to one incident.
    """

    candidates = list_episodes(
        store,
        environment_id=(environment_id),
    )

    return [episode for episode in candidates if (episode.incident_id == incident_id)]


def episode_summary(
    episodes: Iterable[EpisodeMemory],
) -> Dict[str, Any]:
    """
    Small operational summary of episode memory.
    """

    episode_list = list(episodes)

    environments = {episode.environment_id for episode in episode_list}

    providers = {provider for episode in episode_list for provider in episode.providers}

    resources = {
        resource_id for episode in episode_list for resource_id in episode.resource_ids
    }

    linked_incidents = {
        episode.incident_id for episode in episode_list if episode.incident_id
    }

    return {
        "episode_count": len(episode_list),
        "environment_count": len(environments),
        "provider_count": len(providers),
        "resource_count": len(resources),
        "incident_linked_episode_count": (
            sum(1 for episode in episode_list if episode.incident_id)
        ),
        "linked_incident_count": len(linked_incidents),
    }


class EpisodeMemoryManager:
    """
    Convenience interface used by
    memory.pipeline.
    """

    def __init__(
        self,
        store: MemoryStore,
    ) -> None:
        self.store = store

    def remember(
        self,
        group: CorrelationGroup,
        *,
        incident_id: Optional[str] = None,
    ) -> EpisodeMemory:
        return remember_episode(
            self.store,
            group,
            incident_id=(incident_id),
        )

    def remember_many(
        self,
        groups: Iterable[CorrelationGroup],
        *,
        incident_by_group: Optional[Dict[str, str]] = None,
    ) -> List[EpisodeMemory]:
        return remember_episodes(
            self.store,
            groups,
            incident_by_group=(incident_by_group),
        )

    def for_resource(
        self,
        resource_id: str,
        *,
        environment_id: Optional[str] = None,
        limit: Optional[int] = 50,
    ) -> List[EpisodeMemory]:
        return episode_history_for_resource(
            self.store,
            resource_id,
            environment_id=(environment_id),
            limit=limit,
        )

    def for_incident(
        self,
        incident_id: str,
        *,
        environment_id: Optional[str] = None,
    ) -> List[EpisodeMemory]:
        return episode_history_for_incident(
            self.store,
            incident_id,
            environment_id=(environment_id),
        )
