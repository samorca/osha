from __future__ import annotations

from collections import (
    Counter,
    defaultdict,
)
from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.correlation.registry import (
    ResourceRegistry,
)

from src.correlation.topology import (
    TopologyGraph,
)

from src.intelligence.schema import (
    Anomaly,
    EvidenceReference,
    IncidentCandidate,
    RootCauseCandidate,
    RootCauseStatus,
    checked_score,
)

from src.telemetry.schema import (
    Severity,
    TelemetryRecord,
)


SEVERITY_SCORES = {
    Severity.TRACE.value: 0.00,
    Severity.DEBUG.value: 0.05,
    Severity.INFO.value: 0.10,
    Severity.NOTICE.value: 0.20,
    Severity.WARNING.value: 0.50,
    Severity.ERROR.value: 0.75,
    Severity.CRITICAL.value: 0.90,
    Severity.ALERT.value: 0.95,
    Severity.EMERGENCY.value: 1.00,
    Severity.UNKNOWN.value: 0.10,
}


@dataclass(frozen=True)
class RootCauseConfig:
    """
    Configuration for deterministic root-cause
    hypothesis ranking.

    Scores are evidence strengths, not calibrated
    probabilities.
    """

    temporal_weight: float = 0.25

    anomaly_weight: float = 0.30

    severity_weight: float = 0.20

    topology_weight: float = 0.15

    evidence_weight: float = 0.10

    minimum_score: float = 0.30

    likely_threshold: float = 0.70

    maximum_candidates: int = 5

    topology_depth: int = 6

    def validate(
        self,
    ) -> None:
        weights = (
            self.temporal_weight,
            self.anomaly_weight,
            self.severity_weight,
            self.topology_weight,
            self.evidence_weight,
        )

        if any(
            value < 0
            for value in weights
        ):
            raise ValueError(
                "root-cause weights "
                "must be >= 0"
            )

        if sum(weights) <= 0:
            raise ValueError(
                "at least one root-cause "
                "weight must be > 0"
            )

        if not (
            0.0
            <= self.minimum_score
            <= 1.0
        ):
            raise ValueError(
                "minimum_score must be "
                "between 0 and 1"
            )

        if not (
            0.0
            <= self.likely_threshold
            <= 1.0
        ):
            raise ValueError(
                "likely_threshold must be "
                "between 0 and 1"
            )

        if self.maximum_candidates < 1:
            raise ValueError(
                "maximum_candidates must "
                "be >= 1"
            )

        if self.topology_depth < 1:
            raise ValueError(
                "topology_depth must "
                "be >= 1"
            )


def parse_timestamp(
    value: Any,
) -> Optional[datetime]:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            text
        )

    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )


def severity_score(
    value: str,
) -> float:
    return checked_score(
        SEVERITY_SCORES.get(
            value,
            SEVERITY_SCORES[
                Severity.UNKNOWN.value
            ],
        )
    )


def records_for_incident(
    incident: IncidentCandidate,
    registry: ResourceRegistry,
) -> List[
    TelemetryRecord
]:
    output = []

    for record_id in (
        incident.record_ids
    ):
        record = registry.get_record(
            record_id
        )

        if record is not None:
            output.append(record)

    return output


def anomalies_for_incident(
    incident: IncidentCandidate,
    anomalies: Iterable[
        Anomaly
    ],
) -> List[
    Anomaly
]:
    wanted_ids = set(
        incident.anomaly_ids
    )

    incident_records = set(
        incident.record_ids
    )

    incident_resources = set(
        incident.resource_ids
    )

    output = []

    for anomaly in anomalies:
        anomaly.finalize()

        if (
            anomaly.anomaly_id
            and anomaly.anomaly_id
            in wanted_ids
        ):
            output.append(anomaly)
            continue

        if incident_records.intersection(
            anomaly.record_ids
        ):
            output.append(anomaly)
            continue

        if incident_resources.intersection(
            anomaly.resource_ids
        ):
            output.append(anomaly)

    return output


# ---------------------------------------------------------------
# Evidence dimensions
# ---------------------------------------------------------------

def earliest_time_by_resource(
    records: Iterable[
        TelemetryRecord
    ],
) -> Dict[
    str,
    datetime,
]:
    output: Dict[
        str,
        datetime,
    ] = {}

    for record in records:
        timestamp = parse_timestamp(
            record.signal.timestamp
        )

        if timestamp is None:
            continue

        resource_id = (
            record.resource.effective_id()
        )

        current = output.get(
            resource_id
        )

        if (
            current is None
            or timestamp < current
        ):
            output[
                resource_id
            ] = timestamp

    return output


def temporal_scores(
    records: Iterable[
        TelemetryRecord
    ],
) -> Dict[str, float]:
    """
    Earlier resources receive stronger temporal
    hypothesis evidence.

    Early occurrence is not proof of causation.
    """

    earliest = (
        earliest_time_by_resource(
            records
        )
    )

    if not earliest:
        return {}

    unique_times = sorted(
        set(
            earliest.values()
        )
    )

    if len(unique_times) == 1:
        return {
            resource_id: 1.0
            for resource_id
            in earliest
        }

    start = unique_times[0]

    finish = unique_times[-1]

    duration = max(
        (
            finish
            - start
        ).total_seconds(),
        0.000001,
    )

    return {
        resource_id: checked_score(
            1.0
            - (
                (
                    timestamp
                    - start
                ).total_seconds()
                / duration
            )
        )
        for resource_id, timestamp
        in earliest.items()
    }


def anomaly_scores(
    anomalies: Iterable[
        Anomaly
    ],
) -> Dict[str, float]:
    scores: Dict[
        str,
        List[float],
    ] = defaultdict(list)

    for anomaly in anomalies:
        for resource_id in (
            anomaly.resource_ids
        ):
            scores[
                resource_id
            ].append(
                anomaly.score
            )

    output = {}

    for (
        resource_id,
        values,
    ) in scores.items():
        strongest = max(
            values,
            default=0.0,
        )

        density = min(
            1.0,
            len(values) / 3.0,
        )

        output[
            resource_id
        ] = checked_score(
            (
                strongest
                * 0.80
            )
            + (
                density
                * 0.20
            )
        )

    return output


def severity_scores(
    records: Iterable[
        TelemetryRecord
    ],
) -> Dict[str, float]:
    output: Dict[
        str,
        float,
    ] = defaultdict(float)

    for record in records:
        resource_id = (
            record.resource.effective_id()
        )

        output[
            resource_id
        ] = max(
            output[
                resource_id
            ],
            severity_score(
                record.signal.severity
            ),
        )

    return dict(output)


def topology_scores(
    resource_ids: Iterable[str],
    topology: Optional[
        TopologyGraph
    ],
    *,
    max_depth: int,
) -> Dict[str, float]:
    """
    Rank resources by whether they can reach
    other affected resources through outgoing
    topology paths.

    This supports an upstream-origin hypothesis,
    but topology direction still doesn't prove
    event causation.
    """

    identifiers = set(
        resource_ids
    )

    if topology is None:
        return {
            resource_id: 0.0
            for resource_id
            in identifiers
        }

    if not identifiers:
        return {}

    output = {}

    for resource_id in identifiers:
        reachable = 0

        distance_total = 0

        direct_outgoing = 0

        for target_id in identifiers:
            if target_id == resource_id:
                continue

            path = topology.shortest_path(
                resource_id,
                target_id,
                max_depth=max_depth,
                direction="out",
            )

            if path is not None:
                reachable += 1

                distance_total += path.distance

        for relationship in (
            topology.outgoing(
                resource_id
            )
        ):
            if (
                relationship.
                target_resource_id
                in identifiers
            ):
                direct_outgoing += 1

        denominator = max(
            1,
            len(identifiers) - 1,
        )

        reach_score = (
            reachable
            / denominator
        )

        proximity = (
            reachable
            / distance_total
            if distance_total > 0
            else 0.0
        )

        direct_score = min(
            1.0,
            direct_outgoing
            / denominator,
        )

        output[
            resource_id
        ] = checked_score(
            (
                reach_score
                * 0.50
            )
            + (
                proximity
                * 0.30
            )
            + (
                direct_score
                * 0.20
            )
        )

    return output


def evidence_scores(
    records: Iterable[
        TelemetryRecord
    ],
    anomalies: Iterable[
        Anomaly
    ],
) -> Dict[str, float]:
    """
    Score how much direct incident evidence
    references each resource.
    """

    record_counts = Counter(
        record.resource.effective_id()
        for record in records
    )

    anomaly_counts = Counter(
        resource_id
        for anomaly in anomalies
        for resource_id
        in anomaly.resource_ids
    )

    max_records = max(
        record_counts.values(),
        default=1,
    )

    max_anomalies = max(
        anomaly_counts.values(),
        default=1,
    )

    resource_ids = (
        set(record_counts)
        | set(anomaly_counts)
    )

    return {
        resource_id: checked_score(
            (
                record_counts.get(
                    resource_id,
                    0,
                )
                / max_records
            )
            * 0.50
            + (
                anomaly_counts.get(
                    resource_id,
                    0,
                )
                / max_anomalies
            )
            * 0.50
        )
        for resource_id
        in resource_ids
    }


# ---------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------

def candidate_title(
    resource_id: str,
    registry: ResourceRegistry,
) -> str:
    resource = registry.get_resource(
        resource_id
    )

    if resource is None:
        return (
            "Possible originating resource "
            f"{resource_id}"
        )

    label = (
        resource.name
        or resource.host
        or resource.provider_id
    )

    return (
        f"Possible {resource.resource_type} "
        f"origin: {label}"
    )


def build_evidence(
    resource_id: str,
    records: List[
        TelemetryRecord
    ],
    anomalies: List[
        Anomaly
    ],
    dimensions: Mapping[
        str,
        float,
    ],
) -> List[
    EvidenceReference
]:
    resource_records = [
        record
        for record in records
        if (
            record.resource.effective_id()
            == resource_id
        )
    ]

    resource_anomalies = [
        anomaly
        for anomaly in anomalies
        if resource_id
        in anomaly.resource_ids
    ]

    descriptions = {
        "temporal": (
            "This resource appears early in "
            "the incident timeline"
        ),
        "anomaly": (
            "Anomaly findings directly "
            "reference this resource"
        ),
        "severity": (
            "Elevated-severity signals "
            "reference this resource"
        ),
        "topology": (
            "Topology position can reach or "
            "connect affected resources"
        ),
        "evidence": (
            "Multiple incident observations "
            "reference this resource"
        ),
    }

    output = []

    for (
        dimension,
        score,
    ) in dimensions.items():
        if score <= 0:
            continue

        output.append(
            EvidenceReference(
                evidence_type=dimension,
                description=(
                    descriptions[
                        dimension
                    ]
                ),
                score=score,
                record_ids=[
                    record.record_id
                    for record
                    in resource_records
                    if record.record_id
                ],
                resource_ids=[
                    resource_id
                ],
                attributes={
                    "anomaly_ids": [
                        anomaly.anomaly_id
                        for anomaly
                        in resource_anomalies
                        if anomaly.anomaly_id
                    ],
                },
            )
        )

    return output


def causal_path(
    resource_id: str,
    incident: IncidentCandidate,
    topology: Optional[
        TopologyGraph
    ],
    *,
    max_depth: int,
) -> List[str]:
    """
    Return the longest discovered outgoing path
    from this hypothesis to another affected
    resource.

    It is a topology path, not a confirmed causal
    chain.
    """

    if topology is None:
        return [
            resource_id
        ]

    best_path = None

    for target_id in (
        incident.resource_ids
    ):
        if target_id == resource_id:
            continue

        path = topology.shortest_path(
            resource_id,
            target_id,
            max_depth=max_depth,
            direction="out",
        )

        if (
            path is not None
            and (
                best_path is None
                or path.distance
                > best_path.distance
            )
        ):
            best_path = path

    if best_path is None:
        return [
            resource_id
        ]

    return best_path.resource_ids


def rank_root_causes(
    incident: IncidentCandidate,
    registry: ResourceRegistry,
    anomalies: Iterable[
        Anomaly
    ],
    *,
    topology: Optional[
        TopologyGraph
    ] = None,
    config: Optional[
        RootCauseConfig
    ] = None,
) -> List[
    RootCauseCandidate
]:
    """
    Rank possible root-cause resources for one
    incident candidate.
    """

    configuration = (
        config
        or RootCauseConfig()
    )

    configuration.validate()

    records = records_for_incident(
        incident,
        registry,
    )

    anomaly_list = (
        anomalies_for_incident(
            incident,
            anomalies,
        )
    )

    resource_ids = set(
        incident.resource_ids
    )

    resource_ids.update(
        record.resource.effective_id()
        for record in records
    )

    resource_ids.update(
        resource_id
        for anomaly in anomaly_list
        for resource_id
        in anomaly.resource_ids
    )

    temporal = temporal_scores(
        records
    )

    anomaly = anomaly_scores(
        anomaly_list
    )

    severity = severity_scores(
        records
    )

    topology_dimension = (
        topology_scores(
            resource_ids,
            topology,
            max_depth=(
                configuration.
                topology_depth
            ),
        )
    )

    evidence = evidence_scores(
        records,
        anomaly_list,
    )

    weights = {
        "temporal": (
            configuration.
            temporal_weight
        ),
        "anomaly": (
            configuration.
            anomaly_weight
        ),
        "severity": (
            configuration.
            severity_weight
        ),
        "topology": (
            configuration.
            topology_weight
        ),
        "evidence": (
            configuration.
            evidence_weight
        ),
    }

    weight_total = sum(
        weights.values()
    )

    candidates = []

    for resource_id in resource_ids:
        dimensions = {
            "temporal": (
                temporal.get(
                    resource_id,
                    0.0,
                )
            ),
            "anomaly": (
                anomaly.get(
                    resource_id,
                    0.0,
                )
            ),
            "severity": (
                severity.get(
                    resource_id,
                    0.0,
                )
            ),
            "topology": (
                topology_dimension.get(
                    resource_id,
                    0.0,
                )
            ),
            "evidence": (
                evidence.get(
                    resource_id,
                    0.0,
                )
            ),
        }

        score = checked_score(
            sum(
                dimensions[
                    dimension
                ]
                * weights[
                    dimension
                ]
                for dimension
                in weights
            )
            / weight_total
        )

        if (
            score
            < configuration.
            minimum_score
        ):
            continue

        if (
            score
            >= configuration.
            likely_threshold
        ):
            status = (
                RootCauseStatus.
                LIKELY.value
            )

        else:
            status = (
                RootCauseStatus.
                SUSPECTED.value
            )

        resource_records = [
            record
            for record in records
            if (
                record.resource.
                effective_id()
                == resource_id
            )
        ]

        resource_anomalies = [
            item
            for item
            in anomaly_list
            if resource_id
            in item.resource_ids
        ]

        candidate = (
            RootCauseCandidate(
                title=candidate_title(
                    resource_id,
                    registry,
                ),
                resource_id=(
                    resource_id
                ),
                score=score,
                status=status,
                description=(
                    "This resource is ranked "
                    "as a root-cause hypothesis "
                    "based on timeline, anomaly, "
                    "severity, topology, and "
                    "evidence coverage."
                ),
                contributing_record_ids=[
                    record.record_id
                    for record
                    in resource_records
                    if record.record_id
                ],
                contributing_anomaly_ids=[
                    item.anomaly_id
                    for item
                    in resource_anomalies
                    if item.anomaly_id
                ],
                evidence=build_evidence(
                    resource_id,
                    records,
                    anomaly_list,
                    dimensions,
                ),
                causal_path=causal_path(
                    resource_id,
                    incident,
                    topology,
                    max_depth=(
                        configuration.
                        topology_depth
                    ),
                ),
                attributes={
                    "dimension_scores": (
                        dimensions
                    ),
                    "incident_id": (
                        incident.
                        incident_id
                    ),
                },
            )
        )

        candidates.append(
            candidate.finalize()
        )

    candidates.sort(
        key=lambda item: (
            -item.score,
            item.resource_id,
        )
    )

    return candidates[
        :configuration.
        maximum_candidates
    ]


def analyze_incident_root_causes(
    incident: IncidentCandidate,
    registry: ResourceRegistry,
    anomalies: Iterable[
        Anomaly
    ],
    *,
    topology: Optional[
        TopologyGraph
    ] = None,
    config: Optional[
        RootCauseConfig
    ] = None,
) -> IncidentCandidate:
    incident.root_causes = (
        rank_root_causes(
            incident,
            registry,
            anomalies,
            topology=topology,
            config=config,
        )
    )

    return incident.finalize()


def analyze_root_causes(
    incidents: Iterable[
        IncidentCandidate
    ],
    registry: ResourceRegistry,
    anomalies: Iterable[
        Anomaly
    ],
    *,
    topology: Optional[
        TopologyGraph
    ] = None,
    config: Optional[
        RootCauseConfig
    ] = None,
) -> List[
    IncidentCandidate
]:
    anomaly_list = list(
        anomalies
    )

    return [
        analyze_incident_root_causes(
            incident,
            registry,
            anomaly_list,
            topology=topology,
            config=config,
        )
        for incident
        in incidents
    ]


def root_cause_summary(
    incidents: Iterable[
        IncidentCandidate
    ],
) -> Dict[str, Any]:
    incident_list = list(
        incidents
    )

    candidates = [
        candidate
        for incident in incident_list
        for candidate
        in incident.root_causes
    ]

    return {
        "incident_count": len(
            incident_list
        ),
        "candidate_count": len(
            candidates
        ),
        "statuses": dict(
            Counter(
                candidate.status
                for candidate
                in candidates
            )
        ),
        "resources": dict(
            Counter(
                candidate.resource_id
                for candidate
                in candidates
            )
        ),
        "incidents_without_candidates": (
            sum(
                1
                for incident
                in incident_list
                if not incident.
                root_causes
            )
        ),
    }


class RootCauseAnalyzer:
    """
    Reusable root-cause hypothesis analyzer.
    """

    def __init__(
        self,
        registry: ResourceRegistry,
        *,
        topology: Optional[
            TopologyGraph
        ] = None,
        config: Optional[
            RootCauseConfig
        ] = None,
    ) -> None:
        self.registry = registry

        self.topology = topology

        self.config = (
            config
            or RootCauseConfig()
        )

        self.config.validate()

    def analyze(
        self,
        incidents: Iterable[
            IncidentCandidate
        ],
        anomalies: Iterable[
            Anomaly
        ],
    ) -> List[
        IncidentCandidate
    ]:
        return analyze_root_causes(
            incidents,
            self.registry,
            anomalies,
            topology=self.topology,
            config=self.config,
        )
