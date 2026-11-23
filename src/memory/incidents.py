from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

from src.intelligence.schema import (
    Explanation,
    IncidentCandidate,
    RootCauseCandidate,
    RootCauseStatus,
)

from src.memory.schema import (
    IncidentMemory,
    OutcomeMemory,
    OutcomeType,
    RootCauseMemory,
    VerificationStatus,
)

from src.memory.store import (
    MemoryQuery,
    MemoryStore,
)


# ---------------------------------------------------------------
# Root-cause memory conversion
# ---------------------------------------------------------------

def root_cause_verification_status(
    candidate: RootCauseCandidate,
) -> str:
    """
    Translate intelligence root-cause status into
    persistent verification state.

    Crucial distinction:

        likely     != verified
        suspected  != verified
        confirmed  = verified

    Intelligence ranking alone must never create
    verified historical truth.
    """

    if (
        candidate.status
        == RootCauseStatus.
        CONFIRMED.value
    ):
        return (
            VerificationStatus.
            VERIFIED.value
        )

    if (
        candidate.status
        == RootCauseStatus.
        REJECTED.value
    ):
        return (
            VerificationStatus.
            REJECTED.value
        )

    return (
        VerificationStatus.
        UNVERIFIED.value
    )


def root_cause_memory_from_candidate(
    incident: IncidentCandidate,
    candidate: RootCauseCandidate,
) -> RootCauseMemory:
    """
    Convert an intelligence root-cause hypothesis
    into historical memory.
    """

    incident.finalize()

    candidate.finalize()

    evidence_types = sorted(
        {
            evidence.evidence_type
            for evidence
            in candidate.evidence
        }
    )

    source_ids = []

    if incident.incident_id:
        source_ids.append(
            incident.incident_id
        )

    if candidate.candidate_id:
        source_ids.append(
            candidate.candidate_id
        )

    memory = RootCauseMemory(
        environment_id=(
            incident.environment_id
        ),
        incident_id=(
            incident.incident_id
        ),
        candidate_id=(
            candidate.candidate_id
        ),
        root_resource_id=(
            candidate.resource_id
        ),
        title=(
            candidate.title
        ),
        hypothesis_score=(
            candidate.score
        ),
        evidence_types=(
            evidence_types
        ),
        causal_path=list(
            candidate.causal_path
        ),
        verification_status=(
            root_cause_verification_status(
                candidate
            )
        ),
        source_ids=source_ids,
        resource_ids=[
            candidate.resource_id
        ],
        tags=[
            "root_cause_hypothesis"
        ],
        attributes={
            "intelligence_status": (
                candidate.status
            ),
            "description": (
                candidate.description
            ),
            "contributing_record_ids": (
                list(
                    candidate.
                    contributing_record_ids
                )
            ),
            "contributing_anomaly_ids": (
                list(
                    candidate.
                    contributing_anomaly_ids
                )
            ),
            "candidate_attributes": (
                dict(
                    candidate.attributes
                )
            ),
            "evidence": [
                evidence.to_dict()
                for evidence
                in candidate.evidence
            ],
        },
    )

    return memory.finalize()


# ---------------------------------------------------------------
# Root-cause lookup / persistence
# ---------------------------------------------------------------

def find_root_cause_memory(
    store: MemoryStore,
    *,
    incident_id: str,
    candidate_id: Optional[
        str
    ] = None,
    resource_id: Optional[
        str
    ] = None,
) -> Optional[
    RootCauseMemory
]:
    query = MemoryQuery(
        memory_types=(
            "root_cause",
        ),
        resource_ids=(
            (
                resource_id,
            )
            if resource_id
            else ()
        ),
        newest_first=True,
    )

    for memory in store.query(
        query
    ):
        if not isinstance(
            memory,
            RootCauseMemory,
        ):
            continue

        if (
            memory.incident_id
            != incident_id
        ):
            continue

        if (
            candidate_id is not None
            and memory.candidate_id
            != candidate_id
        ):
            continue

        return memory

    return None


def merge_root_cause_memory(
    existing: RootCauseMemory,
    incoming: RootCauseMemory,
) -> RootCauseMemory:
    """
    Refresh evidence on an existing hypothesis.

    Existing verified/rejected status is not
    silently overwritten by a later unverified
    intelligence run.
    """

    existing.hypothesis_score = (
        incoming.hypothesis_score
    )

    existing.title = (
        incoming.title
        or existing.title
    )

    existing.evidence_types = sorted(
        set(
            existing.evidence_types
        )
        | set(
            incoming.evidence_types
        )
    )

    existing.causal_path = (
        list(
            incoming.causal_path
        )
        if incoming.causal_path
        else list(
            existing.causal_path
        )
    )

    existing.source_ids = sorted(
        set(
            existing.source_ids
        )
        | set(
            incoming.source_ids
        )
    )

    existing.resource_ids = sorted(
        set(
            existing.resource_ids
        )
        | set(
            incoming.resource_ids
        )
    )

    existing.tags = sorted(
        set(
            existing.tags
        )
        | set(
            incoming.tags
        )
    )

    existing.attributes.update(
        incoming.attributes
    )

    # Only authoritative incoming states may alter
    # persistent verification.
    if (
        incoming.verification_status
        in {
            VerificationStatus.
            VERIFIED.value,
            VerificationStatus.
            REJECTED.value,
        }
    ):
        existing.verification_status = (
            incoming.
            verification_status
        )

    existing.touch()

    return existing.finalize()


def remember_root_cause(
    store: MemoryStore,
    incident: IncidentCandidate,
    candidate: RootCauseCandidate,
) -> RootCauseMemory:
    incoming = (
        root_cause_memory_from_candidate(
            incident,
            candidate,
        )
    )

    existing = find_root_cause_memory(
        store,
        incident_id=(
            incident.incident_id
            or ""
        ),
        candidate_id=(
            candidate.candidate_id
        ),
        resource_id=(
            candidate.resource_id
        ),
    )

    if existing is not None:
        incoming = (
            merge_root_cause_memory(
                existing,
                incoming,
            )
        )

    stored = store.put(
        incoming
    )

    if not isinstance(
        stored,
        RootCauseMemory,
    ):
        raise TypeError(
            "Memory store returned a "
            "non-RootCauseMemory object"
        )

    return stored


def remember_root_causes(
    store: MemoryStore,
    incident: IncidentCandidate,
) -> List[
    RootCauseMemory
]:
    return [
        remember_root_cause(
            store,
            incident,
            candidate,
        )
        for candidate
        in incident.root_causes
    ]


# ---------------------------------------------------------------
# Incident conversion
# ---------------------------------------------------------------

def explanation_for_incident(
    incident: IncidentCandidate,
    explanations: Iterable[
        Explanation
    ],
) -> Optional[
    Explanation
]:
    for explanation in explanations:
        if (
            explanation.incident_id
            == incident.incident_id
        ):
            return explanation

    return None


def incident_memory_from_incident(
    incident: IncidentCandidate,
    *,
    root_cause_memory_ids: Iterable[
        str
    ] = (),
    explanation: Optional[
        Explanation
    ] = None,
) -> IncidentMemory:
    """
    Convert the intelligence incident candidate
    into persistent operational memory.

    Candidate incidents remain unverified unless
    an authoritative workflow later verifies
    them.
    """

    incident.finalize()

    source_ids = []

    if incident.incident_id:
        source_ids.append(
            incident.incident_id
        )

    source_ids.extend(
        incident.correlation_group_ids
    )

    attributes: Dict[
        str,
        Any,
    ] = {
        "intelligence_status": (
            incident.status
        ),
        "incident_attributes": (
            dict(
                incident.attributes
            )
        ),
        "confidence_limitations": (
            list(
                incident.confidence.
                limitations
            )
        ),
        "impact_score": (
            incident.impact.score
        ),
        "impact_user_known": (
            incident.impact.
            user_impact_known
        ),
        "potentially_exposed_resources": (
            incident.impact.attributes.get(
                "potentially_exposed_resources",
                {},
            )
        ),
    }

    if explanation is not None:
        attributes[
            "explanation_headline"
        ] = explanation.headline

        attributes[
            "recommended_investigations"
        ] = list(
            explanation.
            recommended_investigations
        )

    verification_status = (
        VerificationStatus.
        UNVERIFIED.value
    )

    memory = IncidentMemory(
        environment_id=(
            incident.environment_id
        ),
        incident_id=(
            incident.incident_id
            or ""
        ),
        title=(
            incident.title
        ),
        severity=(
            incident.severity
        ),
        incident_status=(
            incident.status
        ),
        started_at=(
            incident.started_at
        ),
        ended_at=(
            incident.
            last_observed_at
        ),
        anomaly_ids=list(
            incident.anomaly_ids
        ),
        pattern_ids=list(
            incident.pattern_ids
        ),
        correlation_group_ids=list(
            incident.
            correlation_group_ids
        ),
        root_cause_memory_ids=sorted(
            set(
                root_cause_memory_ids
            )
        ),
        affected_resource_ids=list(
            incident.impact.
            affected_resource_ids
            or incident.resource_ids
        ),
        affected_services=list(
            incident.impact.
            affected_services
        ),
        impact_level=(
            incident.impact.level
        ),
        confidence_score=(
            incident.confidence.overall
        ),
        confidence_level=(
            incident.confidence.level
        ),
        explanation_summary=(
            explanation.summary
            if explanation
            is not None
            else incident.summary
        ),
        verification_status=(
            verification_status
        ),
        source_ids=source_ids,
        resource_ids=list(
            incident.resource_ids
        ),
        tags=[
            "incident_history"
        ],
        attributes=attributes,
    )

    return memory.finalize()


# ---------------------------------------------------------------
# Incident lookup / persistence
# ---------------------------------------------------------------

def find_incident_memory(
    store: MemoryStore,
    incident_id: str,
) -> Optional[
    IncidentMemory
]:
    query = MemoryQuery(
        memory_types=(
            "incident",
        ),
        source_ids=(
            incident_id,
        ),
        newest_first=True,
    )

    for memory in store.query(
        query
    ):
        if (
            isinstance(
                memory,
                IncidentMemory,
            )
            and memory.incident_id
            == incident_id
        ):
            return memory

    return None


def merge_incident_memory(
    existing: IncidentMemory,
    incoming: IncidentMemory,
) -> IncidentMemory:
    """
    Refresh an existing incident memory.

    Explicit verification is preserved.
    """

    existing.title = (
        incoming.title
        or existing.title
    )

    existing.severity = (
        incoming.severity
    )

    existing.incident_status = (
        incoming.incident_status
    )

    if (
        incoming.started_at
        and (
            not existing.started_at
            or incoming.started_at
            < existing.started_at
        )
    ):
        existing.started_at = (
            incoming.started_at
        )

    if (
        incoming.ended_at
        and (
            not existing.ended_at
            or incoming.ended_at
            > existing.ended_at
        )
    ):
        existing.ended_at = (
            incoming.ended_at
        )

    existing.anomaly_ids = sorted(
        set(
            existing.anomaly_ids
        )
        | set(
            incoming.anomaly_ids
        )
    )

    existing.pattern_ids = sorted(
        set(
            existing.pattern_ids
        )
        | set(
            incoming.pattern_ids
        )
    )

    existing.correlation_group_ids = (
        sorted(
            set(
                existing.
                correlation_group_ids
            )
            | set(
                incoming.
                correlation_group_ids
            )
        )
    )

    existing.root_cause_memory_ids = (
        sorted(
            set(
                existing.
                root_cause_memory_ids
            )
            | set(
                incoming.
                root_cause_memory_ids
            )
        )
    )

    existing.affected_resource_ids = (
        sorted(
            set(
                existing.
                affected_resource_ids
            )
            | set(
                incoming.
                affected_resource_ids
            )
        )
    )

    existing.affected_services = sorted(
        set(
            existing.
            affected_services
        )
        | set(
            incoming.
            affected_services
        )
    )

    existing.resource_ids = sorted(
        set(
            existing.resource_ids
        )
        | set(
            incoming.resource_ids
        )
    )

    existing.source_ids = sorted(
        set(
            existing.source_ids
        )
        | set(
            incoming.source_ids
        )
    )

    existing.tags = sorted(
        set(
            existing.tags
        )
        | set(
            incoming.tags
        )
    )

    existing.impact_level = (
        incoming.impact_level
    )

    existing.confidence_score = (
        incoming.confidence_score
    )

    existing.confidence_level = (
        incoming.confidence_level
    )

    existing.explanation_summary = (
        incoming.explanation_summary
        or existing.
        explanation_summary
    )

    existing.attributes.update(
        incoming.attributes
    )

    # Never downgrade authoritative historical
    # verification because a new intelligence run
    # produced an unverified candidate.
    if (
        incoming.verification_status
        != VerificationStatus.
        UNVERIFIED.value
    ):
        existing.verification_status = (
            incoming.
            verification_status
        )

    existing.touch()

    return existing.finalize()


def remember_incident(
    store: MemoryStore,
    incident: IncidentCandidate,
    *,
    explanations: Iterable[
        Explanation
    ] = (),
) -> IncidentMemory:
    """
    Persist one incident plus its root-cause
    hypotheses.
    """

    incident.finalize()

    root_cause_memories = (
        remember_root_causes(
            store,
            incident,
        )
    )

    root_cause_memory_ids = [
        memory.memory_id
        for memory
        in root_cause_memories
        if memory.memory_id
    ]

    explanation = (
        explanation_for_incident(
            incident,
            explanations,
        )
    )

    incoming = (
        incident_memory_from_incident(
            incident,
            root_cause_memory_ids=(
                root_cause_memory_ids
            ),
            explanation=(
                explanation
            ),
        )
    )

    existing = None

    if incident.incident_id:
        existing = (
            find_incident_memory(
                store,
                incident.
                incident_id
            )
        )

    if existing is not None:
        incoming = (
            merge_incident_memory(
                existing,
                incoming,
            )
        )

    stored = store.put(
        incoming
    )

    if not isinstance(
        stored,
        IncidentMemory,
    ):
        raise TypeError(
            "Memory store returned a "
            "non-IncidentMemory object"
        )

    return stored


def remember_incidents(
    store: MemoryStore,
    incidents: Iterable[
        IncidentCandidate
    ],
    *,
    explanations: Iterable[
        Explanation
    ] = (),
) -> List[
    IncidentMemory
]:
    explanation_list = list(
        explanations
    )

    return [
        remember_incident(
            store,
            incident,
            explanations=(
                explanation_list
            ),
        )
        for incident
        in incidents
    ]


# ---------------------------------------------------------------
# Explicit verification
# ---------------------------------------------------------------

def verify_incident(
    store: MemoryStore,
    incident_id: str,
    *,
    verification_source: str,
) -> IncidentMemory:
    """
    Explicitly mark historical incident memory as
    verified.

    Intelligence output alone does not call this.
    """

    memory = find_incident_memory(
        store,
        incident_id,
    )

    if memory is None:
        raise KeyError(
            "Incident memory not found: "
            f"{incident_id}"
        )

    memory.verification_status = (
        VerificationStatus.
        VERIFIED.value
    )

    memory.attributes[
        "verification_source"
    ] = verification_source

    memory.touch()

    stored = store.put(
        memory
    )

    if not isinstance(
        stored,
        IncidentMemory,
    ):
        raise TypeError(
            "Memory store returned a "
            "non-IncidentMemory object"
        )

    return stored


def verify_root_cause(
    store: MemoryStore,
    memory_id: str,
    *,
    verification_source: str,
) -> RootCauseMemory:
    """
    Explicitly verify one root-cause hypothesis.
    """

    memory = store.require(
        memory_id
    )

    if not isinstance(
        memory,
        RootCauseMemory,
    ):
        raise TypeError(
            "memory_id does not reference "
            "RootCauseMemory"
        )

    memory.verification_status = (
        VerificationStatus.
        VERIFIED.value
    )

    memory.verification_source = (
        verification_source
    )

    from src.memory.schema import (
        utc_now,
    )

    memory.verified_at = (
        utc_now()
    )

    memory.touch()

    stored = store.put(
        memory
    )

    if not isinstance(
        stored,
        RootCauseMemory,
    ):
        raise TypeError(
            "Memory store returned a "
            "non-RootCauseMemory object"
        )

    return stored


def reject_root_cause(
    store: MemoryStore,
    memory_id: str,
    *,
    verification_source: str,
) -> RootCauseMemory:
    """
    Explicitly reject one historical hypothesis.
    """

    memory = store.require(
        memory_id
    )

    if not isinstance(
        memory,
        RootCauseMemory,
    ):
        raise TypeError(
            "memory_id does not reference "
            "RootCauseMemory"
        )

    memory.verification_status = (
        VerificationStatus.
        REJECTED.value
    )

    memory.verification_source = (
        verification_source
    )

    memory.touch()

    stored = store.put(
        memory
    )

    if not isinstance(
        stored,
        RootCauseMemory,
    ):
        raise TypeError(
            "Memory store returned a "
            "non-RootCauseMemory object"
        )

    return stored


# ---------------------------------------------------------------
# Outcome memory
# ---------------------------------------------------------------

def record_outcome(
    store: MemoryStore,
    *,
    environment_id: str,
    incident_id: str,
    outcome_type: str,
    summary: Optional[str] = None,
    remediation_action_ids: Iterable[
        str
    ] = (),
    before_health_score: Optional[
        float
    ] = None,
    after_health_score: Optional[
        float
    ] = None,
    recovery_verified: bool = False,
    verification_source: Optional[
        str
    ] = None,
) -> OutcomeMemory:
    """
    Store an observed incident outcome.

    This records what happened after investigation
    or remediation. It does not execute any
    remediation action.
    """

    allowed_outcomes = {
        item.value
        for item
        in OutcomeType
    }

    if outcome_type not in allowed_outcomes:
        raise ValueError(
            "Unsupported outcome type: "
            f"{outcome_type}"
        )

    verification = (
        VerificationStatus.
        VERIFIED.value
        if recovery_verified
        else VerificationStatus.
        UNVERIFIED.value
    )

    memory = OutcomeMemory(
        environment_id=(
            environment_id
        ),
        incident_id=incident_id,
        outcome_type=outcome_type,
        summary=summary,
        remediation_action_ids=(
            sorted(
                set(
                    remediation_action_ids
                )
            )
        ),
        before_health_score=(
            before_health_score
        ),
        after_health_score=(
            after_health_score
        ),
        recovery_verified=(
            recovery_verified
        ),
        verification_source=(
            verification_source
        ),
        verification_status=(
            verification
        ),
        source_ids=[
            incident_id
        ],
        tags=[
            "incident_outcome"
        ],
    ).finalize()

    stored = store.put(
        memory
    )

    if not isinstance(
        stored,
        OutcomeMemory,
    ):
        raise TypeError(
            "Memory store returned a "
            "non-OutcomeMemory object"
        )

    incident_memory = (
        find_incident_memory(
            store,
            incident_id,
        )
    )

    if (
        incident_memory is not None
        and stored.memory_id
    ):
        incident_memory.outcome_memory_id = (
            stored.memory_id
        )

        incident_memory.touch()

        store.put(
            incident_memory
        )

    return stored


# ---------------------------------------------------------------
# Queries
# ---------------------------------------------------------------

def list_incident_memory(
    store: MemoryStore,
    *,
    environment_id: Optional[
        str
    ] = None,
    resource_id: Optional[
        str
    ] = None,
    verified_only: bool = False,
    limit: Optional[int] = None,
) -> List[
    IncidentMemory
]:
    query = MemoryQuery(
        memory_types=(
            "incident",
        ),
        environment_ids=(
            (
                environment_id,
            )
            if environment_id
            else ()
        ),
        resource_ids=(
            (
                resource_id,
            )
            if resource_id
            else ()
        ),
        verification_statuses=(
            (
                VerificationStatus.
                VERIFIED.value,
            )
            if verified_only
            else ()
        ),
        limit=limit,
    )

    return [
        memory
        for memory
        in store.query(query)
        if isinstance(
            memory,
            IncidentMemory,
        )
    ]


def root_causes_for_incident(
    store: MemoryStore,
    incident_id: str,
    *,
    verified_only: bool = False,
) -> List[
    RootCauseMemory
]:
    query = MemoryQuery(
        memory_types=(
            "root_cause",
        ),
        verification_statuses=(
            (
                VerificationStatus.
                VERIFIED.value,
            )
            if verified_only
            else ()
        ),
    )

    return [
        memory
        for memory
        in store.query(query)
        if (
            isinstance(
                memory,
                RootCauseMemory,
            )
            and memory.incident_id
            == incident_id
        )
    ]


def incident_memory_summary(
    incidents: Iterable[
        IncidentMemory
    ],
) -> Dict[str, Any]:
    incident_list = list(
        incidents
    )

    return {
        "incident_memory_count": (
            len(
                incident_list
            )
        ),
        "verified_incident_count": (
            sum(
                1
                for incident
                in incident_list
                if (
                    incident.
                    verification_status
                    == VerificationStatus.
                    VERIFIED.value
                )
            )
        ),
        "environment_count": len(
            {
                incident.environment_id
                for incident
                in incident_list
            }
        ),
        "affected_resource_count": len(
            {
                resource_id
                for incident
                in incident_list
                for resource_id
                in incident.
                affected_resource_ids
            }
        ),
        "root_cause_reference_count": (
            sum(
                len(
                    incident.
                    root_cause_memory_ids
                )
                for incident
                in incident_list
            )
        ),
        "recorded_outcome_count": (
            sum(
                1
                for incident
                in incident_list
                if (
                    incident.
                    outcome_memory_id
                )
            )
        ),
    }


class IncidentMemoryManager:
    """
    Convenience interface for memory.pipeline and
    later operator feedback workflows.
    """

    def __init__(
        self,
        store: MemoryStore,
    ) -> None:
        self.store = store

    def remember(
        self,
        incident: IncidentCandidate,
        *,
        explanations: Iterable[
            Explanation
        ] = (),
    ) -> IncidentMemory:
        return remember_incident(
            self.store,
            incident,
            explanations=(
                explanations
            ),
        )

    def remember_many(
        self,
        incidents: Iterable[
            IncidentCandidate
        ],
        *,
        explanations: Iterable[
            Explanation
        ] = (),
    ) -> List[
        IncidentMemory
    ]:
        return remember_incidents(
            self.store,
            incidents,
            explanations=(
                explanations
            ),
        )

    def verify_incident(
        self,
        incident_id: str,
        *,
        verification_source: str,
    ) -> IncidentMemory:
        return verify_incident(
            self.store,
            incident_id,
            verification_source=(
                verification_source
            ),
        )

    def verify_root_cause(
        self,
        memory_id: str,
        *,
        verification_source: str,
    ) -> RootCauseMemory:
        return verify_root_cause(
            self.store,
            memory_id,
            verification_source=(
                verification_source
            ),
        )

    def reject_root_cause(
        self,
        memory_id: str,
        *,
        verification_source: str,
    ) -> RootCauseMemory:
        return reject_root_cause(
            self.store,
            memory_id,
            verification_source=(
                verification_source
            ),
        )

    def incidents(
        self,
        *,
        environment_id: Optional[
            str
        ] = None,
        resource_id: Optional[
            str
        ] = None,
        verified_only: bool = False,
        limit: Optional[int] = None,
    ) -> List[
        IncidentMemory
    ]:
        return list_incident_memory(
            self.store,
            environment_id=(
                environment_id
            ),
            resource_id=(
                resource_id
            ),
            verified_only=(
                verified_only
            ),
            limit=limit,
        )
