from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
)

from src.correlation.registry import (
    ResourceRegistry,
)

from src.correlation.schema import (
    CorrelationGroup,
)

from src.correlation.topology import (
    TopologyGraph,
)

from src.intelligence.schema import (
    Anomaly,
    ConfidenceAssessment,
    ConfidenceLevel,
    IncidentCandidate,
    RootCauseStatus,
    checked_score,
    confidence_level,
)

from src.telemetry.schema import (
    TelemetryRecord,
)


@dataclass(frozen=True)
class ConfidenceConfig:
    """
    OSHA intelligence confidence configuration.

    Scores are evidence-strength indicators, not
    calibrated statistical probabilities.
    """

    data_quality_weight: float = 0.20

    correlation_weight: float = 0.25

    anomaly_weight: float = 0.20

    root_cause_weight: float = 0.25

    impact_weight: float = 0.10

    missing_topology_penalty: float = 0.08

    missing_history_penalty: float = 0.08

    unknown_user_impact_penalty: float = 0.04

    single_provider_penalty: float = 0.03

    missing_root_cause_penalty: float = 0.10

    minimum_data_quality: float = 0.10

    def validate(
        self,
    ) -> None:
        weights = (
            self.data_quality_weight,
            self.correlation_weight,
            self.anomaly_weight,
            self.root_cause_weight,
            self.impact_weight,
        )

        if any(value < 0 for value in weights):
            raise ValueError("confidence weights must be >= 0")

        if sum(weights) <= 0:
            raise ValueError("at least one confidence weight must be > 0")

        penalties = (
            self.missing_topology_penalty,
            self.missing_history_penalty,
            self.unknown_user_impact_penalty,
            self.single_provider_penalty,
            self.missing_root_cause_penalty,
        )

        if any(value < 0 for value in penalties):
            raise ValueError("confidence penalties must be >= 0")

        if not (0.0 <= self.minimum_data_quality <= 1.0):
            raise ValueError("minimum_data_quality must be between 0 and 1")


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def records_for_incident(
    incident: IncidentCandidate,
    registry: ResourceRegistry,
) -> List[TelemetryRecord]:
    output = []

    for record_id in incident.record_ids:
        telemetry = registry.get_record(record_id)

        if telemetry is not None:
            output.append(telemetry)

    return output


def anomalies_for_incident(
    incident: IncidentCandidate,
    anomalies: Iterable[Anomaly],
) -> List[Anomaly]:
    wanted_ids = set(incident.anomaly_ids)

    wanted_records = set(incident.record_ids)

    wanted_resources = set(incident.resource_ids)

    output = []

    for anomaly in anomalies:
        anomaly.finalize()

        if anomaly.anomaly_id and anomaly.anomaly_id in wanted_ids:
            output.append(anomaly)

            continue

        if wanted_records.intersection(anomaly.record_ids):
            output.append(anomaly)

            continue

        if wanted_resources.intersection(anomaly.resource_ids):
            output.append(anomaly)

    return output


def groups_for_incident(
    incident: IncidentCandidate,
    groups: Iterable[CorrelationGroup],
) -> List[CorrelationGroup]:
    wanted_ids = set(incident.correlation_group_ids)

    wanted_records = set(incident.record_ids)

    output = []

    for group in groups:
        if group.group_id and group.group_id in wanted_ids:
            output.append(group)

            continue

        if wanted_records.intersection(group.record_ids):
            output.append(group)

    return output


# ---------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------


def record_quality_score(
    telemetry: TelemetryRecord,
) -> float:
    """
    Estimate how much useful normalized context
    exists on one canonical telemetry record.

    This is intentionally structural. It doesn't
    determine whether the observation is true.
    """

    checks = []

    checks.append(bool(telemetry.record_id))

    checks.append(bool(telemetry.signal.timestamp))

    checks.append(bool(telemetry.resource.effective_id()))

    checks.append(bool(telemetry.resource.provider))

    checks.append(bool(telemetry.resource.resource_type))

    checks.append(bool(telemetry.signal.signal_type))

    checks.append(bool(telemetry.resource.environment_id))

    # Signal-specific context.
    signal_context = (
        telemetry.signal.signal_name
        or telemetry.signal.message
        or telemetry.signal.value is not None
    )

    checks.append(bool(signal_context))

    if not checks:
        return 0.0

    return checked_score(sum(1 for value in checks if value) / len(checks))


def data_quality_score(
    records: Iterable[TelemetryRecord],
) -> float:
    values = [record_quality_score(telemetry) for telemetry in records]

    if not values:
        return 0.0

    return checked_score(sum(values) / len(values))


# ---------------------------------------------------------------
# Correlation confidence
# ---------------------------------------------------------------


def correlation_strength(
    groups: Iterable[CorrelationGroup],
) -> float:
    groups_list = list(groups)

    if not groups_list:
        return 0.0

    strongest = max(
        (group.strongest_score for group in groups_list),
        default=0.0,
    )

    averages = [group.average_score for group in groups_list]

    average = sum(averages) / len(averages) if averages else 0.0

    return checked_score((strongest * 0.60) + (average * 0.40))


# ---------------------------------------------------------------
# Anomaly confidence
# ---------------------------------------------------------------


def anomaly_strength(
    anomalies: Iterable[Anomaly],
) -> float:
    anomaly_list = list(anomalies)

    if not anomaly_list:
        return 0.0

    scores = [anomaly.score for anomaly in anomaly_list]

    strongest = max(
        scores,
        default=0.0,
    )

    average = sum(scores) / len(scores) if scores else 0.0

    density = min(
        1.0,
        len(anomaly_list) / 5.0,
    )

    return checked_score((strongest * 0.55) + (average * 0.30) + (density * 0.15))


# ---------------------------------------------------------------
# Root-cause confidence
# ---------------------------------------------------------------


def root_cause_strength(
    incident: IncidentCandidate,
) -> float:
    candidates = list(incident.root_causes)

    if not candidates:
        return 0.0

    candidates.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    strongest = candidates[0]

    score = checked_score(strongest.score)

    # A likely hypothesis has passed OSHA's
    # stronger evidence threshold, but it is
    # still not automatically confirmed.
    if strongest.status == RootCauseStatus.LIKELY.value:
        status_component = 0.80

    elif strongest.status == RootCauseStatus.CONFIRMED.value:
        status_component = 1.00

    elif strongest.status == RootCauseStatus.SUSPECTED.value:
        status_component = 0.50

    else:
        status_component = 0.0

    # Separation from competing hypotheses adds
    # evidence that one explanation is stronger.
    if len(candidates) >= 2:
        margin = max(
            0.0,
            candidates[0].score - candidates[1].score,
        )

    else:
        margin = candidates[0].score

    margin_component = min(
        1.0,
        margin * 2.0,
    )

    return checked_score(
        (score * 0.60) + (status_component * 0.25) + (margin_component * 0.15)
    )


# ---------------------------------------------------------------
# Impact confidence
# ---------------------------------------------------------------


def impact_strength(
    incident: IncidentCandidate,
) -> float:
    impact = incident.impact

    base = checked_score(impact.score)

    evidence_count = len(impact.evidence)

    evidence_component = min(
        1.0,
        evidence_count / 4.0,
    )

    directly_affected = len(impact.affected_resource_ids)

    resource_component = min(
        1.0,
        directly_affected / 4.0,
    )

    return checked_score(
        (base * 0.60) + (evidence_component * 0.25) + (resource_component * 0.15)
    )


# ---------------------------------------------------------------
# Coverage and limitations
# ---------------------------------------------------------------


def providers_for_records(
    records: Iterable[TelemetryRecord],
) -> Set[str]:
    return {
        telemetry.resource.provider
        for telemetry in records
        if telemetry.resource.provider
    }


def topology_available(
    topology: Optional[TopologyGraph],
    incident: IncidentCandidate,
) -> bool:
    if topology is None:
        return False

    if len(topology) <= 0:
        return False

    for resource_id in incident.resource_ids:
        if topology.degree(resource_id)["total"] > 0:
            return True

    return False


def historical_context_available(
    incident: IncidentCandidate,
) -> bool:
    """
    The memory/history layer can later explicitly
    record that historical evidence was used.

    Until then, we don't infer historical context
    from pattern occurrence counts alone.
    """

    value = incident.attributes.get("historical_context_used")

    return value is True


def collect_limitations(
    incident: IncidentCandidate,
    records: Iterable[TelemetryRecord],
    *,
    topology: Optional[TopologyGraph] = None,
) -> List[str]:
    records_list = list(records)

    limitations = []

    providers = providers_for_records(records_list)

    if len(providers) == 1:
        provider = next(iter(providers))

        limitations.append(
            "Analysis currently contains "
            f"telemetry from only one provider "
            f"({provider})."
        )

    if not topology_available(
        topology,
        incident,
    ):
        limitations.append(
            "No meaningful topology context was available for the affected resources."
        )

    if not (historical_context_available(incident)):
        limitations.append(
            "No explicit historical incident or baseline context was used."
        )

    if not (incident.impact.user_impact_known):
        limitations.append(
            "User or business impact has not been independently verified."
        )

    if not incident.root_causes:
        limitations.append(
            "No root-cause hypothesis met the configured minimum evidence score."
        )

    elif incident.root_causes[0].status != RootCauseStatus.CONFIRMED.value:
        limitations.append(
            "The highest-ranked root cause "
            "remains a hypothesis and has not "
            "been independently confirmed."
        )

    if not records_list:
        limitations.append(
            "Incident telemetry records were not available in the registry."
        )

    return sorted(set(limitations))


def confidence_penalty(
    incident: IncidentCandidate,
    records: Iterable[TelemetryRecord],
    *,
    topology: Optional[TopologyGraph],
    config: ConfidenceConfig,
) -> float:
    records_list = list(records)

    penalty = 0.0

    providers = providers_for_records(records_list)

    if len(providers) == 1:
        penalty += config.single_provider_penalty

    if not topology_available(
        topology,
        incident,
    ):
        penalty += config.missing_topology_penalty

    if not (historical_context_available(incident)):
        penalty += config.missing_history_penalty

    if not (incident.impact.user_impact_known):
        penalty += config.unknown_user_impact_penalty

    if not incident.root_causes:
        penalty += config.missing_root_cause_penalty

    return max(
        0.0,
        penalty,
    )


# ---------------------------------------------------------------
# Overall confidence calculation
# ---------------------------------------------------------------


def calculate_confidence(
    incident: IncidentCandidate,
    registry: ResourceRegistry,
    anomalies: Iterable[Anomaly],
    groups: Iterable[CorrelationGroup],
    *,
    topology: Optional[TopologyGraph] = None,
    config: Optional[ConfidenceConfig] = None,
) -> ConfidenceAssessment:
    """
    Calculate transparent intelligence confidence.

    Confidence reflects evidence coverage and
    consistency. It does not mean an incident
    conclusion has been scientifically proven.
    """

    configuration = config or ConfidenceConfig()

    configuration.validate()

    records = records_for_incident(
        incident,
        registry,
    )

    incident_anomalies = anomalies_for_incident(
        incident,
        anomalies,
    )

    incident_groups = groups_for_incident(
        incident,
        groups,
    )

    data_quality = max(
        configuration.minimum_data_quality,
        data_quality_score(records),
    )

    correlation = correlation_strength(incident_groups)

    anomaly = anomaly_strength(incident_anomalies)

    root_cause = root_cause_strength(incident)

    impact = impact_strength(incident)

    dimensions = {
        "data_quality": data_quality,
        "correlation": correlation,
        "anomaly": anomaly,
        "root_cause": root_cause,
        "impact": impact,
    }

    weights = {
        "data_quality": (configuration.data_quality_weight),
        "correlation": (configuration.correlation_weight),
        "anomaly": (configuration.anomaly_weight),
        "root_cause": (configuration.root_cause_weight),
        "impact": (configuration.impact_weight),
    }

    total_weight = sum(weights.values())

    base_score = checked_score(
        sum(dimensions[dimension] * weights[dimension] for dimension in weights)
        / total_weight
    )

    penalty = confidence_penalty(
        incident,
        records,
        topology=topology,
        config=configuration,
    )

    overall = checked_score(base_score - penalty)

    limitations = collect_limitations(
        incident,
        records,
        topology=topology,
    )

    level = confidence_level(overall)

    explanation = (
        "Confidence is based on normalized "
        "telemetry quality, correlation "
        "strength, anomaly evidence, root-cause "
        "evidence, and impact evidence."
    )

    if limitations:
        explanation += (
            " The score is reduced where "
            "important investigation context "
            "is unavailable or unverified."
        )

    assessment = ConfidenceAssessment(
        overall=overall,
        level=level,
        data_quality=data_quality,
        correlation_strength=(correlation),
        anomaly_strength=anomaly,
        root_cause_strength=(root_cause),
        impact_strength=impact,
        explanation=explanation,
        limitations=limitations,
        attributes={
            "base_score": base_score,
            "penalty": penalty,
            "dimension_scores": (dimensions),
            "weights": weights,
            "provider_count": len(providers_for_records(records)),
            "record_count": len(records),
            "anomaly_count": len(incident_anomalies),
            "correlation_group_count": len(incident_groups),
            "root_cause_candidate_count": (len(incident.root_causes)),
            "topology_available": (
                topology_available(
                    topology,
                    incident,
                )
            ),
            "historical_context_available": (historical_context_available(incident)),
        },
    )

    return assessment.finalize()


def analyze_incident_confidence(
    incident: IncidentCandidate,
    registry: ResourceRegistry,
    anomalies: Iterable[Anomaly],
    groups: Iterable[CorrelationGroup],
    *,
    topology: Optional[TopologyGraph] = None,
    config: Optional[ConfidenceConfig] = None,
) -> IncidentCandidate:
    incident.confidence = calculate_confidence(
        incident,
        registry,
        anomalies,
        groups,
        topology=topology,
        config=config,
    )

    return incident.finalize()


def analyze_confidence(
    incidents: Iterable[IncidentCandidate],
    registry: ResourceRegistry,
    anomalies: Iterable[Anomaly],
    groups: Iterable[CorrelationGroup],
    *,
    topology: Optional[TopologyGraph] = None,
    config: Optional[ConfidenceConfig] = None,
) -> List[IncidentCandidate]:
    anomaly_list = list(anomalies)

    group_list = list(groups)

    return [
        analyze_incident_confidence(
            incident,
            registry,
            anomaly_list,
            group_list,
            topology=topology,
            config=config,
        )
        for incident in incidents
    ]


# ---------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------


def confidence_summary(
    incidents: Iterable[IncidentCandidate],
) -> Dict[str, Any]:
    incident_list = list(incidents)

    levels = Counter(incident.confidence.level for incident in incident_list)

    values = [incident.confidence.overall for incident in incident_list]

    average = sum(values) / len(values) if values else 0.0

    limitations = Counter(
        limitation
        for incident in incident_list
        for limitation in incident.confidence.limitations
    )

    return {
        "incident_count": (len(incident_list)),
        "confidence_levels": (dict(levels)),
        "average_confidence": (checked_score(average)),
        "limitations": dict(limitations),
    }


class ConfidenceAnalyzer:
    """
    Reusable confidence evaluation component for
    intelligence.pipeline.
    """

    def __init__(
        self,
        registry: ResourceRegistry,
        *,
        topology: Optional[TopologyGraph] = None,
        config: Optional[ConfidenceConfig] = None,
    ) -> None:
        self.registry = registry

        self.topology = topology

        self.config = config or ConfidenceConfig()

        self.config.validate()

    def analyze(
        self,
        incidents: Iterable[IncidentCandidate],
        anomalies: Iterable[Anomaly],
        groups: Iterable[CorrelationGroup],
    ) -> List[IncidentCandidate]:
        return analyze_confidence(
            incidents,
            self.registry,
            anomalies,
            groups,
            topology=self.topology,
            config=self.config,
        )
