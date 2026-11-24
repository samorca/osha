from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from difflib import SequenceMatcher
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
)

from src.intelligence.schema import (
    IncidentCandidate,
    Pattern,
)

from src.memory.schema import (
    BaselineMemory,
    EpisodeMemory,
    IncidentMemory,
    MemoryObject,
    OutcomeMemory,
    PatternMemory,
    RootCauseMemory,
    VerificationStatus,
    checked_score,
)

from src.memory.store import (
    MemoryQuery,
    MemoryStore,
)


@dataclass(frozen=True)
class RetrievalConfig:
    """
    Configuration for deterministic historical
    retrieval.

    Similarity scores are ranking heuristics,
    not calibrated probabilities.
    """

    minimum_similarity: float = 0.30

    maximum_results: int = 10

    verified_boost: float = 0.10

    resource_weight: float = 0.30

    pattern_weight: float = 0.25

    service_weight: float = 0.15

    severity_weight: float = 0.10

    title_weight: float = 0.10

    environment_weight: float = 0.10

    def validate(
        self,
    ) -> None:
        if not (0.0 <= self.minimum_similarity <= 1.0):
            raise ValueError("minimum_similarity must be between 0 and 1")

        if self.maximum_results < 1:
            raise ValueError("maximum_results must be >= 1")

        weights = (
            self.resource_weight,
            self.pattern_weight,
            self.service_weight,
            self.severity_weight,
            self.title_weight,
            self.environment_weight,
        )

        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("retrieval weights must be non-negative and total > 0")


@dataclass(frozen=True)
class MemoryMatch:
    """
    Ranked historical-memory match.

    `reasons` and dimension scores explain why
    the memory was retrieved.
    """

    memory_id: str

    memory_type: str

    score: float

    verification_status: str

    reasons: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "memory_id": (self.memory_id),
            "memory_type": (self.memory_type),
            "score": checked_score(self.score),
            "verification_status": (self.verification_status),
            "reasons": list(self.reasons),
            "attributes": dict(self.attributes),
        }


# ---------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------


def normalized_text(
    value: Optional[str],
) -> str:
    return " ".join((value or "").strip().lower().split())


def jaccard(
    left: Iterable[str],
    right: Iterable[str],
) -> float:
    """
    Measure set overlap:

        intersection / union
    """

    left_set = {item for item in left if item}

    right_set = {item for item in right if item}

    if not left_set and not right_set:
        return 0.0

    union = left_set | right_set

    if not union:
        return 0.0

    return len(left_set & right_set) / len(union)


def text_similarity(
    left: Optional[str],
    right: Optional[str],
) -> float:
    """
    Return deterministic title-text similarity.

    This is only one low-weight retrieval
    dimension.
    """

    first = normalized_text(left)

    second = normalized_text(right)

    if not first or not second:
        return 0.0

    return SequenceMatcher(
        None,
        first,
        second,
        autojunk=False,
    ).ratio()


def incident_services(
    memory: IncidentMemory,
) -> Set[str]:
    return {value.lower() for value in memory.affected_services if value}


def candidate_services(
    incident: IncidentCandidate,
) -> Set[str]:
    services = {value.lower() for value in incident.impact.affected_services if value}

    raw_services = incident.attributes.get(
        "services",
        [],
    )

    if isinstance(
        raw_services,
        Sequence,
    ) and not isinstance(
        raw_services,
        (
            str,
            bytes,
        ),
    ):
        services.update(str(value).lower() for value in raw_services if value)

    return services


# ---------------------------------------------------------------
# Similar incident retrieval
# ---------------------------------------------------------------


def incident_similarity(
    current: IncidentCandidate,
    historical: IncidentMemory,
    *,
    config: Optional[RetrievalConfig] = None,
) -> MemoryMatch:
    """
    Compare a current incident candidate with one
    historical incident.

    Verified historical incidents receive a
    small configurable ranking boost, but
    unverified history remains retrievable and
    clearly labelled.
    """

    configuration = config or RetrievalConfig()

    configuration.validate()

    dimensions = {
        "resource": jaccard(
            current.resource_ids,
            (historical.affected_resource_ids or historical.resource_ids),
        ),
        "pattern": jaccard(
            current.pattern_ids,
            historical.pattern_ids,
        ),
        "service": jaccard(
            candidate_services(current),
            incident_services(historical),
        ),
        "severity": (1.0 if current.severity == historical.severity else 0.0),
        "title": text_similarity(
            current.title,
            historical.title,
        ),
        "environment": (
            1.0 if current.environment_id == historical.environment_id else 0.0
        ),
    }

    weights = {
        "resource": (configuration.resource_weight),
        "pattern": (configuration.pattern_weight),
        "service": (configuration.service_weight),
        "severity": (configuration.severity_weight),
        "title": (configuration.title_weight),
        "environment": (configuration.environment_weight),
    }

    total_weight = sum(weights.values())

    base_score = (
        sum(dimensions[name] * weights[name] for name in weights) / total_weight
    )

    verified = historical.verification_status == VerificationStatus.VERIFIED.value

    score = checked_score(
        base_score + (configuration.verified_boost if verified else 0.0)
    )

    reasons = [
        (f"{name} similarity={value:.2f}")
        for name, value in dimensions.items()
        if value > 0
    ]

    if verified:
        reasons.append("historical incident is verified")

    return MemoryMatch(
        memory_id=(historical.memory_id or ""),
        memory_type=(historical.memory_type),
        score=score,
        verification_status=(historical.verification_status),
        reasons=reasons,
        attributes={
            "incident_id": (historical.incident_id),
            "title": (historical.title),
            "outcome_memory_id": (historical.outcome_memory_id),
            "dimension_scores": (dimensions),
        },
    )


def similar_incidents(
    store: MemoryStore,
    incident: IncidentCandidate,
    *,
    config: Optional[RetrievalConfig] = None,
    verified_only: bool = False,
) -> List[MemoryMatch]:
    """
    Retrieve and rank historical incidents from
    the same environment.
    """

    configuration = config or RetrievalConfig()

    configuration.validate()

    query = MemoryQuery(
        memory_types=("incident",),
        environment_ids=(incident.environment_id,),
        verification_statuses=(
            (VerificationStatus.VERIFIED.value,) if verified_only else ()
        ),
    )

    matches = []

    for memory in store.query(query):
        if not isinstance(
            memory,
            IncidentMemory,
        ):
            continue

        if memory.incident_id == incident.incident_id:
            continue

        match = incident_similarity(
            incident,
            memory,
            config=configuration,
        )

        if match.score >= configuration.minimum_similarity:
            matches.append(match)

    matches.sort(
        key=lambda item: (
            -item.score,
            item.memory_id,
        )
    )

    return matches[: configuration.maximum_results]


# ---------------------------------------------------------------
# Pattern retrieval
# ---------------------------------------------------------------


def find_pattern_history(
    store: MemoryStore,
    pattern: Pattern,
) -> Optional[PatternMemory]:
    query = MemoryQuery(
        memory_types=("pattern",),
        environment_ids=(pattern.environment_id,),
    )

    for memory in store.query(query):
        if (
            isinstance(
                memory,
                PatternMemory,
            )
            and memory.fingerprint == pattern.fingerprint
        ):
            return memory

    return None


def pattern_context(
    store: MemoryStore,
    patterns: Iterable[Pattern],
) -> Dict[
    str,
    Dict[str, Any],
]:
    """
    Return historical context for current
    intelligence patterns.

    Verified rates exclude unverified
    observations.
    """

    output: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for pattern in patterns:
        pattern.finalize()

        history = find_pattern_history(
            store,
            pattern,
        )

        key = pattern.pattern_id or pattern.fingerprint

        if history is None:
            output[key] = {
                "known": False,
                "occurrence_count": 0,
                "verified_incident_rate": (None),
                "verified_normal_rate": (None),
            }

            continue

        verified_total = history.verified_incident_count + history.verified_normal_count

        output[key] = {
            "known": True,
            "memory_id": (history.memory_id),
            "occurrence_count": (history.occurrence_count),
            "verified_incident_count": (history.verified_incident_count),
            "verified_normal_count": (history.verified_normal_count),
            "verified_incident_rate": (
                history.verified_incident_count / verified_total
                if verified_total > 0
                else None
            ),
            "verified_normal_rate": (
                history.verified_normal_count / verified_total
                if verified_total > 0
                else None
            ),
            "last_seen": (history.last_seen),
        }

    return output


# ---------------------------------------------------------------
# Resource history
# ---------------------------------------------------------------


def resource_history(
    store: MemoryStore,
    resource_id: str,
    *,
    environment_id: Optional[str] = None,
    memory_types: Sequence[str] = (),
    verified_only: bool = False,
    limit: int = 50,
) -> List[MemoryObject]:
    """
    Retrieve memory involving one resource.
    """

    return store.query(
        MemoryQuery(
            memory_types=tuple(memory_types),
            environment_ids=((environment_id,) if environment_id else ()),
            resource_ids=(resource_id,),
            verification_statuses=(
                (VerificationStatus.VERIFIED.value,) if verified_only else ()
            ),
            limit=limit,
        )
    )


def verified_root_causes_for_resource(
    store: MemoryStore,
    resource_id: str,
    *,
    environment_id: Optional[str] = None,
    limit: int = 20,
) -> List[RootCauseMemory]:
    results = resource_history(
        store,
        resource_id,
        environment_id=(environment_id),
        memory_types=("root_cause",),
        verified_only=True,
        limit=limit,
    )

    return [
        item
        for item in results
        if isinstance(
            item,
            RootCauseMemory,
        )
    ]


# ---------------------------------------------------------------
# Outcome retrieval
# ---------------------------------------------------------------


def outcome_for_incident(
    store: MemoryStore,
    incident: IncidentMemory | str,
) -> Optional[OutcomeMemory]:
    """
    Retrieve the latest known outcome for an
    incident.
    """

    historical = (
        incident
        if isinstance(
            incident,
            IncidentMemory,
        )
        else None
    )

    incident_id = historical.incident_id if historical is not None else incident

    outcome_id = historical.outcome_memory_id if historical is not None else None

    if outcome_id:
        result = store.get(outcome_id)

        if isinstance(
            result,
            OutcomeMemory,
        ):
            return result

    query = MemoryQuery(
        memory_types=("outcome",),
        source_ids=(incident_id,),
        newest_first=True,
    )

    for item in store.query(query):
        if (
            isinstance(
                item,
                OutcomeMemory,
            )
            and item.incident_id == incident_id
        ):
            return item

    return None


# ---------------------------------------------------------------
# Baselines and episodes
# ---------------------------------------------------------------


def baseline_for_metric(
    store: MemoryStore,
    *,
    environment_id: str,
    resource_id: str,
    signal_name: str,
    unit: Optional[str] = None,
) -> Optional[BaselineMemory]:
    query = MemoryQuery(
        memory_types=("baseline",),
        environment_ids=(environment_id,),
        resource_ids=(resource_id,),
        newest_first=True,
    )

    for memory in store.query(query):
        if not isinstance(
            memory,
            BaselineMemory,
        ):
            continue

        if memory.resource_id != resource_id or memory.signal_name != signal_name:
            continue

        if unit is not None and memory.metric_unit != unit:
            continue

        return memory

    return None


def recent_episodes(
    store: MemoryStore,
    *,
    environment_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    limit: int = 20,
) -> List[EpisodeMemory]:
    query = MemoryQuery(
        memory_types=("episode",),
        environment_ids=((environment_id,) if environment_id else ()),
        resource_ids=((resource_id,) if resource_id else ()),
        active_only=True,
        limit=limit,
        newest_first=True,
    )

    return [
        item
        for item in store.query(query)
        if isinstance(
            item,
            EpisodeMemory,
        )
    ]


# ---------------------------------------------------------------
# Combined intelligence context
# ---------------------------------------------------------------


def intelligence_context(
    store: MemoryStore,
    *,
    incident: Optional[IncidentCandidate] = None,
    patterns: Iterable[Pattern] = (),
    resource_ids: Iterable[str] = (),
    config: Optional[RetrievalConfig] = None,
) -> Dict[str, Any]:
    """
    Build bounded historical context for a future
    intelligence run.

    This returns structured memory context. It
    does not mutate the current intelligence
    conclusion.
    """

    pattern_list = list(patterns)

    resources = set(resource_ids)

    if incident is not None:
        resources.update(incident.resource_ids)

    context: Dict[
        str,
        Any,
    ] = {
        "patterns": pattern_context(
            store,
            pattern_list,
        ),
        "resources": {},
        "similar_incidents": [],
    }

    for resource_id in sorted(resources):
        history = resource_history(
            store,
            resource_id,
            limit=20,
        )

        verified_causes = verified_root_causes_for_resource(
            store,
            resource_id,
        )

        context["resources"][resource_id] = {
            "memory_count": len(history),
            "memory_ids": [item.memory_id for item in history if item.memory_id],
            "verified_root_cause_memory_ids": [
                item.memory_id for item in verified_causes if item.memory_id
            ],
        }

    if incident is not None:
        context["similar_incidents"] = [
            match.to_dict()
            for match in similar_incidents(
                store,
                incident,
                config=config,
            )
        ]

    return context


class MemoryRetriever:
    """
    Reusable retrieval interface for
    intelligence and decision layers.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self.store = store

        self.config = config or RetrievalConfig()

        self.config.validate()

    def similar_incidents(
        self,
        incident: IncidentCandidate,
        *,
        verified_only: bool = False,
    ) -> List[MemoryMatch]:
        return similar_incidents(
            self.store,
            incident,
            config=self.config,
            verified_only=(verified_only),
        )

    def pattern_context(
        self,
        patterns: Iterable[Pattern],
    ) -> Dict[
        str,
        Dict[str, Any],
    ]:
        return pattern_context(
            self.store,
            patterns,
        )

    def resource_history(
        self,
        resource_id: str,
        *,
        environment_id: Optional[str] = None,
        verified_only: bool = False,
        limit: int = 50,
    ) -> List[MemoryObject]:
        return resource_history(
            self.store,
            resource_id,
            environment_id=(environment_id),
            verified_only=(verified_only),
            limit=limit,
        )

    def context(
        self,
        *,
        incident: Optional[IncidentCandidate] = None,
        patterns: Iterable[Pattern] = (),
        resource_ids: Iterable[str] = (),
    ) -> Dict[str, Any]:
        return intelligence_context(
            self.store,
            incident=incident,
            patterns=patterns,
            resource_ids=(resource_ids),
            config=self.config,
        )
