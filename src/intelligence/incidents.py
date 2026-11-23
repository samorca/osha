from __future__ import annotations

from collections import (
    Counter,
    defaultdict,
)
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
)

from src.correlation.schema import (
    CorrelationGroup,
)

from src.intelligence.schema import (
    Anomaly,
    IncidentCandidate,
    IncidentSeverity,
    IncidentStatus,
    Pattern,
    checked_score,
    highest_severity,
    severity_rank,
)


DEFAULT_INCIDENT_THRESHOLD = 0.60


@dataclass(frozen=True)
class IncidentConfig:
    """
    Configuration governing when correlated
    anomalies become incident candidates.

    These thresholds are OSHA policy defaults,
    not calibrated probabilities.
    """

    incident_threshold: float = (
        DEFAULT_INCIDENT_THRESHOLD
    )

    minimum_anomalies: int = 1

    minimum_records: int = 2

    require_correlation_group: bool = True

    allow_critical_single_record: bool = True

    def validate(
        self,
    ) -> None:
        if not (
            0.0
            <= self.incident_threshold
            <= 1.0
        ):
            raise ValueError(
                "incident_threshold must "
                "be between 0 and 1"
            )

        if self.minimum_anomalies < 1:
            raise ValueError(
                "minimum_anomalies must "
                "be >= 1"
            )

        if self.minimum_records < 1:
            raise ValueError(
                "minimum_records must "
                "be >= 1"
            )


def anomalies_for_group(
    group: CorrelationGroup,
    anomalies: Iterable[
        Anomaly
    ],
) -> List[Anomaly]:
    """
    Find anomalies connected to a correlation
    group through records or resources.
    """

    record_ids = set(
        group.record_ids
    )

    resource_ids = set(
        group.resource_ids
    )

    output = []

    for anomaly in anomalies:
        if record_ids.intersection(
            anomaly.record_ids
        ):
            output.append(
                anomaly
            )

            continue

        if resource_ids.intersection(
            anomaly.resource_ids
        ):
            output.append(
                anomaly
            )

    return output


def patterns_for_group(
    group: CorrelationGroup,
    patterns: Iterable[
        Pattern
    ],
) -> List[Pattern]:
    """
    Find patterns associated with a correlation
    group through records, resources or explicit
    correlation-group references.
    """

    record_ids = set(
        group.record_ids
    )

    resource_ids = set(
        group.resource_ids
    )

    output = []

    for pattern in patterns:
        if record_ids.intersection(
            pattern.record_ids
        ):
            output.append(
                pattern
            )

            continue

        if resource_ids.intersection(
            pattern.resource_ids
        ):
            output.append(
                pattern
            )

            continue

        group_ids = pattern.attributes.get(
            "correlation_group_ids",
            [],
        )

        if (
            group.group_id
            and isinstance(
                group_ids,
                list,
            )
            and group.group_id
            in group_ids
        ):
            output.append(
                pattern
            )

    return output


def incident_score(
    group: CorrelationGroup,
    anomalies: Iterable[
        Anomaly
    ],
) -> float:
    """
    Calculate incident-candidate evidence from:

      correlation strength
      strongest anomaly
      anomaly density

    This is not a probability.
    """

    anomaly_list = list(
        anomalies
    )

    anomaly_strength = max(
        (
            anomaly.score
            for anomaly
            in anomaly_list
        ),
        default=0.0,
    )

    anomaly_density = min(
        1.0,
        len(anomaly_list)
        / max(
            1,
            len(group.record_ids),
        ),
    )

    correlation_strength = max(
        group.strongest_score,
        group.average_score,
    )

    return checked_score(
        (
            correlation_strength
            * 0.45
        )
        + (
            anomaly_strength
            * 0.40
        )
        + (
            anomaly_density
            * 0.15
        )
    )


def incident_severity(
    anomalies: Iterable[
        Anomaly
    ],
    score: float,
) -> str:
    """
    Choose the stronger severity from anomaly
    severity and total incident evidence.
    """

    anomaly_severities = [
        anomaly.severity
        for anomaly
        in anomalies
    ]

    anomaly_severity = (
        highest_severity(
            anomaly_severities
        )
    )

    if score >= 0.90:
        score_severity = (
            IncidentSeverity.
            CRITICAL.value
        )

    elif score >= 0.75:
        score_severity = (
            IncidentSeverity.HIGH.value
        )

    elif score >= 0.55:
        score_severity = (
            IncidentSeverity.MEDIUM.value
        )

    elif score >= 0.30:
        score_severity = (
            IncidentSeverity.LOW.value
        )

    else:
        score_severity = (
            IncidentSeverity.INFO.value
        )

    return max(
        (
            anomaly_severity,
            score_severity,
        ),
        key=severity_rank,
    )


def incident_title(
    group: CorrelationGroup,
    anomalies: Iterable[
        Anomaly
    ],
) -> str:
    anomaly_list = list(
        anomalies
    )

    if anomaly_list:
        strongest = max(
            anomaly_list,
            key=lambda anomaly: (
                anomaly.score
            ),
        )

        return strongest.title

    providers = (
        ", ".join(
            group.providers
        )
        if group.providers
        else "infrastructure"
    )

    return (
        "Correlated abnormal activity "
        f"in {providers}"
    )


def group_is_incident(
    group: CorrelationGroup,
    anomalies: Iterable[
        Anomaly
    ],
    *,
    config: IncidentConfig,
) -> bool:
    """
    Decide whether one correlation group contains
    enough abnormal evidence to become an
    incident candidate.
    """

    anomaly_list = list(
        anomalies
    )

    if (
        len(group.record_ids)
        < config.minimum_records
    ):
        return False

    if (
        len(anomaly_list)
        < config.minimum_anomalies
    ):
        return False

    score = incident_score(
        group,
        anomaly_list,
    )

    return (
        score
        >= config.incident_threshold
    )


def incident_from_group(
    group: CorrelationGroup,
    anomalies: Iterable[
        Anomaly
    ],
    patterns: Iterable[
        Pattern
    ] = (),
    *,
    config: Optional[
        IncidentConfig
    ] = None,
) -> Optional[
    IncidentCandidate
]:
    """
    Convert a sufficiently supported correlation
    group into an IncidentCandidate.
    """

    configuration = (
        config
        or IncidentConfig()
    )

    configuration.validate()

    anomaly_list = list(
        anomalies
    )

    pattern_list = list(
        patterns
    )

    if not group_is_incident(
        group,
        anomaly_list,
        config=configuration,
    ):
        return None

    score = incident_score(
        group,
        anomaly_list,
    )

    severity = incident_severity(
        anomaly_list,
        score,
    )

    anomaly_ids = [
        anomaly.anomaly_id
        for anomaly
        in anomaly_list
        if anomaly.anomaly_id
    ]

    pattern_ids = [
        pattern.pattern_id
        for pattern
        in pattern_list
        if pattern.pattern_id
    ]

    incident = IncidentCandidate(
        title=incident_title(
            group,
            anomaly_list,
        ),
        environment_id=(
            group.environment_id
        ),
        severity=severity,
        status=(
            IncidentStatus.
            CANDIDATE.value
        ),
        summary=(
            f"{len(anomaly_list)} anomaly "
            "finding(s) across "
            f"{len(group.resource_ids)} "
            "resource(s) in one correlated "
            "group."
        ),
        correlation_group_ids=(
            [group.group_id]
            if group.group_id
            else []
        ),
        record_ids=list(
            group.record_ids
        ),
        resource_ids=list(
            group.resource_ids
        ),
        anomaly_ids=anomaly_ids,
        pattern_ids=pattern_ids,
        started_at=(
            group.start_timestamp
        ),
        last_observed_at=(
            group.end_timestamp
        ),
        attributes={
            "incident_score": score,
            "correlation_strongest_score": (
                group.strongest_score
            ),
            "correlation_average_score": (
                group.average_score
            ),
            "anomaly_count": (
                len(anomaly_list)
            ),
            "pattern_count": (
                len(pattern_list)
            ),
            "providers": list(
                group.providers
            ),
            "signal_types": list(
                group.signal_types
            ),
        },
    )

    return incident.finalize()


def standalone_critical_incidents(
    anomalies: Iterable[
        Anomaly
    ],
    *,
    grouped_anomaly_ids: Set[str],
    config: IncidentConfig,
) -> List[
    IncidentCandidate
]:
    """
    Optionally promote exceptionally strong,
    critical standalone anomalies.

    This is deliberately restricted to critical
    anomalies with score >= 0.90.
    """

    if not (
        config.
        allow_critical_single_record
    ):
        return []

    output = []

    for anomaly in anomalies:
        anomaly.finalize()

        if not anomaly.anomaly_id:
            continue

        if (
            anomaly.anomaly_id
            in grouped_anomaly_ids
        ):
            continue

        if (
            anomaly.severity
            != IncidentSeverity.
            CRITICAL.value
        ):
            continue

        if anomaly.score < 0.90:
            continue

        incident = IncidentCandidate(
            title=anomaly.title,
            environment_id=(
                anomaly.environment_id
            ),
            severity=(
                IncidentSeverity.
                CRITICAL.value
            ),
            status=(
                IncidentStatus.
                CANDIDATE.value
            ),
            summary=(
                "A critical anomaly exceeded "
                "the standalone incident "
                "threshold."
            ),
            record_ids=list(
                anomaly.record_ids
            ),
            resource_ids=list(
                anomaly.resource_ids
            ),
            anomaly_ids=[
                anomaly.anomaly_id
            ],
            started_at=(
                anomaly.detected_at
            ),
            last_observed_at=(
                anomaly.detected_at
            ),
            attributes={
                "incident_score": (
                    anomaly.score
                ),
                "standalone_critical_anomaly": (
                    True
                ),
            },
        )

        output.append(
            incident.finalize()
        )

    return output


def deduplicate_incidents(
    incidents: Iterable[
        IncidentCandidate
    ],
) -> List[
    IncidentCandidate
]:
    """
    Deduplicate using deterministic incident IDs.

    If duplicate representations exist, keep the
    version with the stronger incident score.
    """

    output: Dict[
        str,
        IncidentCandidate,
    ] = {}

    for incident in incidents:
        incident.finalize()

        if not incident.incident_id:
            continue

        existing = output.get(
            incident.incident_id
        )

        current_score = float(
            incident.attributes.get(
                "incident_score",
                0.0,
            )
        )

        existing_score = (
            float(
                existing.attributes.get(
                    "incident_score",
                    0.0,
                )
            )
            if existing is not None
            else -1.0
        )

        if (
            existing is None
            or current_score
            > existing_score
        ):
            output[
                incident.incident_id
            ] = incident

    return sorted(
        output.values(),
        key=lambda incident: (
            -severity_rank(
                incident.severity
            ),
            -float(
                incident.attributes.get(
                    "incident_score",
                    0.0,
                )
            ),
            incident.incident_id
            or "",
        ),
    )


def detect_incidents(
    groups: Iterable[
        CorrelationGroup
    ],
    anomalies: Iterable[
        Anomaly
    ],
    *,
    patterns: Iterable[
        Pattern
    ] = (),
    config: Optional[
        IncidentConfig
    ] = None,
) -> List[
    IncidentCandidate
]:
    """
    Detect incident candidates from correlation
    groups and anomaly findings.
    """

    configuration = (
        config
        or IncidentConfig()
    )

    configuration.validate()

    group_list = list(
        groups
    )

    anomaly_list = list(
        anomalies
    )

    pattern_list = list(
        patterns
    )

    incidents: List[
        IncidentCandidate
    ] = []

    grouped_anomaly_ids: Set[
        str
    ] = set()

    for group in group_list:
        group_anomalies = (
            anomalies_for_group(
                group,
                anomaly_list,
            )
        )

        group_patterns = (
            patterns_for_group(
                group,
                pattern_list,
            )
        )

        incident = incident_from_group(
            group,
            group_anomalies,
            group_patterns,
            config=configuration,
        )

        if incident is None:
            continue

        incidents.append(
            incident
        )

        grouped_anomaly_ids.update(
            anomaly.anomaly_id
            for anomaly
            in group_anomalies
            if anomaly.anomaly_id
        )

    incidents.extend(
        standalone_critical_incidents(
            anomaly_list,
            grouped_anomaly_ids=(
                grouped_anomaly_ids
            ),
            config=configuration,
        )
    )

    return deduplicate_incidents(
        incidents
    )


def incidents_by_resource(
    incidents: Iterable[
        IncidentCandidate
    ],
) -> Dict[
    str,
    List[IncidentCandidate],
]:
    output: Dict[
        str,
        List[IncidentCandidate],
    ] = defaultdict(list)

    for incident in incidents:
        for resource_id in (
            incident.resource_ids
        ):
            output[
                resource_id
            ].append(
                incident
            )

    return dict(output)


def incident_summary(
    incidents: Iterable[
        IncidentCandidate
    ],
) -> Dict[str, Any]:
    incident_list = list(
        incidents
    )

    return {
        "incident_count": (
            len(incident_list)
        ),
        "severities": dict(
            Counter(
                incident.severity
                for incident
                in incident_list
            )
        ),
        "statuses": dict(
            Counter(
                incident.status
                for incident
                in incident_list
            )
        ),
        "affected_resource_count": len(
            {
                resource_id
                for incident
                in incident_list
                for resource_id
                in incident.resource_ids
            }
        ),
        "environment_count": len(
            {
                incident.environment_id
                for incident
                in incident_list
            }
        ),
    }


class IncidentEngine:
    """
    Stateful incident-candidate detector for the
    intelligence pipeline.
    """

    def __init__(
        self,
        *,
        config: Optional[
            IncidentConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or IncidentConfig()
        )

        self.config.validate()

        self._incidents: Dict[
            str,
            IncidentCandidate,
        ] = {}

    def analyze(
        self,
        groups: Iterable[
            CorrelationGroup
        ],
        anomalies: Iterable[
            Anomaly
        ],
        *,
        patterns: Iterable[
            Pattern
        ] = (),
    ) -> List[
        IncidentCandidate
    ]:
        discovered = detect_incidents(
            groups,
            anomalies,
            patterns=patterns,
            config=self.config,
        )

        for incident in discovered:
            incident.finalize()

            if incident.incident_id:
                self._incidents[
                    incident.incident_id
                ] = incident

        return discovered

    def incidents(
        self,
    ) -> List[
        IncidentCandidate
    ]:
        return deduplicate_incidents(
            self._incidents.values()
        )

    def get(
        self,
        incident_id: str,
    ) -> Optional[
        IncidentCandidate
    ]:
        return self._incidents.get(
            incident_id
        )

    def clear(
        self,
    ) -> None:
        self._incidents.clear()
