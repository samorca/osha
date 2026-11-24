from __future__ import annotations

from collections import (
    Counter,
    defaultdict,
    deque,
)
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
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
    ImpactAssessment,
    ImpactLevel,
    IncidentCandidate,
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
class ImpactConfig:
    """
    Configuration for conservative operational
    impact estimation.

    These values are OSHA evidence-weighting
    defaults, not calibrated probabilities.
    """

    direct_anomaly_weight: float = 0.35

    severity_weight: float = 0.20

    resource_spread_weight: float = 0.20

    service_spread_weight: float = 0.10

    topology_weight: float = 0.15

    topology_depth: int = 3

    localized_threshold: float = 0.20

    limited_threshold: float = 0.40

    significant_threshold: float = 0.65

    widespread_threshold: float = 0.85

    def validate(
        self,
    ) -> None:
        weights = (
            self.direct_anomaly_weight,
            self.severity_weight,
            self.resource_spread_weight,
            self.service_spread_weight,
            self.topology_weight,
        )

        if any(value < 0 for value in weights):
            raise ValueError("impact weights must be >= 0")

        if sum(weights) <= 0:
            raise ValueError("at least one impact weight must be > 0")

        if self.topology_depth < 1:
            raise ValueError("topology_depth must be >= 1")

        thresholds = (
            self.localized_threshold,
            self.limited_threshold,
            self.significant_threshold,
            self.widespread_threshold,
        )

        if any(value < 0.0 or value > 1.0 for value in thresholds):
            raise ValueError("impact thresholds must be between 0 and 1")

        if not (
            self.localized_threshold
            <= self.limited_threshold
            <= self.significant_threshold
            <= self.widespread_threshold
        ):
            raise ValueError(
                "impact thresholds must be "
                "ordered localized <= limited "
                "<= significant <= widespread"
            )


def severity_score(
    severity: str,
) -> float:
    return checked_score(
        SEVERITY_SCORES.get(
            severity,
            SEVERITY_SCORES[Severity.UNKNOWN.value],
        )
    )


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

    incident_records = set(incident.record_ids)

    incident_resources = set(incident.resource_ids)

    output = []

    for anomaly in anomalies:
        anomaly.finalize()

        if anomaly.anomaly_id and anomaly.anomaly_id in wanted_ids:
            output.append(anomaly)

            continue

        if incident_records.intersection(anomaly.record_ids):
            output.append(anomaly)

            continue

        if incident_resources.intersection(anomaly.resource_ids):
            output.append(anomaly)

    return output


# ---------------------------------------------------------------
# Directly affected resources
# ---------------------------------------------------------------


def directly_affected_resources(
    incident: IncidentCandidate,
    anomalies: Iterable[Anomaly],
) -> Set[str]:
    """
    Resource IDs supported by direct anomaly or
    incident evidence.

    These are stronger than resources discovered
    only through topology proximity.
    """

    resource_ids = set(incident.resource_ids)

    for anomaly in anomalies:
        resource_ids.update(anomaly.resource_ids)

    return resource_ids


def affected_services(
    resource_ids: Iterable[str],
    registry: ResourceRegistry,
) -> Set[str]:
    output = set()

    for resource_id in resource_ids:
        resource = registry.get_resource(resource_id)

        if resource is None:
            continue

        if resource.service:
            output.add(resource.service)

    return output


def affected_providers(
    resource_ids: Iterable[str],
    registry: ResourceRegistry,
) -> Set[str]:
    output = set()

    for resource_id in resource_ids:
        resource = registry.get_resource(resource_id)

        if resource is None:
            continue

        output.add(resource.provider)

    return output


# ---------------------------------------------------------------
# Topology exposure / blast radius
# ---------------------------------------------------------------


def topology_exposure(
    origins: Iterable[str],
    topology: Optional[TopologyGraph],
    *,
    depth: int,
) -> Dict[
    str,
    int,
]:
    """
    Find topology-near resources and their
    minimum hop distance from directly affected
    resources.

    This represents possible exposure only.
    """

    if topology is None:
        return {}

    origin_set = set(origins)

    distances: Dict[
        str,
        int,
    ] = {}

    queue = deque()

    for resource_id in origin_set:
        distances[resource_id] = 0

        queue.append(
            (
                resource_id,
                0,
            )
        )

    while queue:
        (
            current,
            current_depth,
        ) = queue.popleft()

        if current_depth >= depth:
            continue

        neighbors = topology.neighbors(
            current,
            direction="both",
        )

        for neighbor in neighbors:
            candidate_distance = current_depth + 1

            existing = distances.get(neighbor)

            if existing is not None and existing <= candidate_distance:
                continue

            distances[neighbor] = candidate_distance

            queue.append(
                (
                    neighbor,
                    candidate_distance,
                )
            )

    return distances


def maximum_blast_radius_hops(
    exposure: Mapping[
        str,
        int,
    ],
) -> Optional[int]:
    if not exposure:
        return None

    return max(exposure.values())


def topology_exposure_score(
    direct_resource_ids: Set[str],
    exposure: Mapping[
        str,
        int,
    ],
) -> float:
    """
    Estimate how broadly the directly affected
    set connects into nearby topology.

    This is an exposure score and does not claim
    that all nearby resources failed.
    """

    if not direct_resource_ids:
        return 0.0

    nearby = {
        resource_id
        for resource_id, distance in exposure.items()
        if (resource_id not in direct_resource_ids and distance > 0)
    }

    if not nearby:
        return 0.0

    direct_count = len(direct_resource_ids)

    nearby_count = len(nearby)

    return checked_score(nearby_count / float(nearby_count + direct_count))


# ---------------------------------------------------------------
# Score dimensions
# ---------------------------------------------------------------


def direct_anomaly_score(
    anomalies: Iterable[Anomaly],
) -> float:
    values = [anomaly.score for anomaly in anomalies]

    if not values:
        return 0.0

    strongest = max(values)

    density = min(
        1.0,
        len(values) / 5.0,
    )

    return checked_score((strongest * 0.80) + (density * 0.20))


def incident_severity_score(
    records: Iterable[TelemetryRecord],
) -> float:
    return max(
        (severity_score(telemetry.signal.severity) for telemetry in records),
        default=0.0,
    )


def resource_spread_score(
    direct_resources: Set[str],
) -> float:
    """
    Initial resource spread heuristic.

    A single directly affected resource is local.
    Increasing independently evidenced resources
    increases impact evidence.

    This score does not use total fleet size yet.
    memory/database layers can later provide that
    denominator.
    """

    count = len(direct_resources)

    if count <= 0:
        return 0.0

    if count == 1:
        return 0.20

    if count == 2:
        return 0.40

    if count <= 4:
        return 0.60

    if count <= 8:
        return 0.80

    return 1.0


def service_spread_score(
    services: Set[str],
) -> float:
    count = len(services)

    if count <= 0:
        return 0.0

    if count == 1:
        return 0.25

    if count == 2:
        return 0.50

    if count <= 4:
        return 0.75

    return 1.0


def calculate_impact_score(
    *,
    anomaly_score: float,
    severity: float,
    resource_spread: float,
    service_spread: float,
    topology: float,
    config: ImpactConfig,
) -> float:
    dimensions = {
        "anomaly": (anomaly_score),
        "severity": severity,
        "resource_spread": (resource_spread),
        "service_spread": (service_spread),
        "topology": topology,
    }

    weights = {
        "anomaly": (config.direct_anomaly_weight),
        "severity": (config.severity_weight),
        "resource_spread": (config.resource_spread_weight),
        "service_spread": (config.service_spread_weight),
        "topology": (config.topology_weight),
    }

    total_weight = sum(weights.values())

    return checked_score(
        sum(dimensions[dimension] * weights[dimension] for dimension in weights)
        / total_weight
    )


def impact_level(
    score: float,
    *,
    config: ImpactConfig,
) -> str:
    value = checked_score(score)

    if value >= config.widespread_threshold:
        return ImpactLevel.WIDESPREAD.value

    if value >= config.significant_threshold:
        return ImpactLevel.SIGNIFICANT.value

    if value >= config.limited_threshold:
        return ImpactLevel.LIMITED.value

    if value >= config.localized_threshold:
        return ImpactLevel.LOCALIZED.value

    if value > 0:
        return ImpactLevel.NONE.value

    return ImpactLevel.UNKNOWN.value


# ---------------------------------------------------------------
# Impact evidence
# ---------------------------------------------------------------


def build_impact_evidence(
    *,
    incident: IncidentCandidate,
    direct_resources: Set[str],
    services: Set[str],
    providers: Set[str],
    exposure: Mapping[
        str,
        int,
    ],
    anomaly_score: float,
    severity: float,
    resource_spread: float,
    service_spread: float,
    topology_score: float,
) -> List[EvidenceReference]:
    evidence = []

    if anomaly_score > 0:
        evidence.append(
            EvidenceReference(
                evidence_type=("direct_anomaly"),
                description=("Direct anomaly findings support operational impact"),
                score=anomaly_score,
                record_ids=list(incident.record_ids),
                resource_ids=sorted(direct_resources),
            )
        )

    if severity > 0:
        evidence.append(
            EvidenceReference(
                evidence_type=("signal_severity"),
                description=("Elevated signal severity supports greater impact"),
                score=severity,
                record_ids=list(incident.record_ids),
                resource_ids=sorted(direct_resources),
            )
        )

    if resource_spread > 0:
        evidence.append(
            EvidenceReference(
                evidence_type=("resource_spread"),
                description=(
                    "Multiple independently observed resources appear affected"
                ),
                score=resource_spread,
                resource_ids=sorted(direct_resources),
                attributes={
                    "resource_count": (len(direct_resources)),
                },
            )
        )

    if service_spread > 0:
        evidence.append(
            EvidenceReference(
                evidence_type=("service_spread"),
                description=("Affected resources span one or more known services"),
                score=service_spread,
                resource_ids=sorted(direct_resources),
                attributes={
                    "services": sorted(services),
                },
            )
        )

    if topology_score > 0:
        exposed_resources = {
            resource_id: distance
            for resource_id, distance in exposure.items()
            if (resource_id not in direct_resources)
        }

        evidence.append(
            EvidenceReference(
                evidence_type=("topology_exposure"),
                description=(
                    "Topology contains nearby "
                    "resources that could be "
                    "relevant to the incident "
                    "blast radius"
                ),
                score=topology_score,
                resource_ids=sorted(direct_resources),
                attributes={
                    "potentially_exposed": (exposed_resources),
                    "providers": sorted(providers),
                },
            )
        )

    return evidence


# ---------------------------------------------------------------
# Main impact analysis
# ---------------------------------------------------------------


def assess_impact(
    incident: IncidentCandidate,
    registry: ResourceRegistry,
    anomalies: Iterable[Anomaly],
    *,
    topology: Optional[TopologyGraph] = None,
    config: Optional[ImpactConfig] = None,
) -> ImpactAssessment:
    """
    Estimate impact for one incident candidate.

    `affected_resource_ids` contains resources
    supported by direct incident/anomaly evidence.

    Topology-only neighbors are stored separately
    as potentially exposed resources.
    """

    configuration = config or ImpactConfig()

    configuration.validate()

    incident.finalize()

    records = records_for_incident(
        incident,
        registry,
    )

    anomaly_list = anomalies_for_incident(
        incident,
        anomalies,
    )

    direct_resources = directly_affected_resources(
        incident,
        anomaly_list,
    )

    services = affected_services(
        direct_resources,
        registry,
    )

    providers = affected_providers(
        direct_resources,
        registry,
    )

    exposure = topology_exposure(
        direct_resources,
        topology,
        depth=(configuration.topology_depth),
    )

    anomaly_component = direct_anomaly_score(anomaly_list)

    severity_component = incident_severity_score(records)

    resource_component = resource_spread_score(direct_resources)

    service_component = service_spread_score(services)

    topology_component = topology_exposure_score(
        direct_resources,
        exposure,
    )

    score = calculate_impact_score(
        anomaly_score=(anomaly_component),
        severity=(severity_component),
        resource_spread=(resource_component),
        service_spread=(service_component),
        topology=(topology_component),
        config=configuration,
    )

    level = impact_level(
        score,
        config=configuration,
    )

    exposed_only = {
        resource_id: distance
        for resource_id, distance in exposure.items()
        if resource_id not in direct_resources
    }

    user_impact_known = False

    # User/business impact should be explicitly
    # established by later service/SLO/business
    # context. Infrastructure topology by itself
    # is not enough to assert it.
    summary = f"{len(direct_resources)} resource(s) have direct incident evidence"

    if services:
        summary += f" across {len(services)} known service(s)"

    summary += "."

    if exposed_only:
        summary += (
            f" {len(exposed_only)} additional "
            "resource(s) are topology-near and "
            "potentially exposed, but are not "
            "counted as confirmed impacted."
        )

    assessment = ImpactAssessment(
        level=level,
        score=score,
        affected_resource_ids=sorted(direct_resources),
        affected_services=sorted(services),
        affected_providers=sorted(providers),
        blast_radius_hops=(maximum_blast_radius_hops(exposure)),
        user_impact_known=(user_impact_known),
        summary=summary,
        evidence=build_impact_evidence(
            incident=incident,
            direct_resources=(direct_resources),
            services=services,
            providers=providers,
            exposure=exposure,
            anomaly_score=(anomaly_component),
            severity=(severity_component),
            resource_spread=(resource_component),
            service_spread=(service_component),
            topology_score=(topology_component),
        ),
        attributes={
            "dimension_scores": {
                "anomaly": (anomaly_component),
                "severity": (severity_component),
                "resource_spread": (resource_component),
                "service_spread": (service_component),
                "topology_exposure": (topology_component),
            },
            "direct_resource_count": (len(direct_resources)),
            "known_service_count": (len(services)),
            "potentially_exposed_resources": (exposed_only),
            "topology_depth": (configuration.topology_depth),
            "user_impact_source": (None),
        },
    )

    return assessment.finalize()


def analyze_incident_impact(
    incident: IncidentCandidate,
    registry: ResourceRegistry,
    anomalies: Iterable[Anomaly],
    *,
    topology: Optional[TopologyGraph] = None,
    config: Optional[ImpactConfig] = None,
) -> IncidentCandidate:
    incident.impact = assess_impact(
        incident,
        registry,
        anomalies,
        topology=topology,
        config=config,
    )

    return incident.finalize()


def analyze_impacts(
    incidents: Iterable[IncidentCandidate],
    registry: ResourceRegistry,
    anomalies: Iterable[Anomaly],
    *,
    topology: Optional[TopologyGraph] = None,
    config: Optional[ImpactConfig] = None,
) -> List[IncidentCandidate]:
    anomaly_list = list(anomalies)

    return [
        analyze_incident_impact(
            incident,
            registry,
            anomaly_list,
            topology=topology,
            config=config,
        )
        for incident in incidents
    ]


# ---------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------


def impact_summary(
    incidents: Iterable[IncidentCandidate],
) -> Dict[str, Any]:
    incident_list = list(incidents)

    level_counts = Counter(incident.impact.level for incident in incident_list)

    affected_resources = {
        resource_id
        for incident in incident_list
        for resource_id in incident.impact.affected_resource_ids
    }

    affected_services = {
        service
        for incident in incident_list
        for service in incident.impact.affected_services
    }

    user_impact_known = sum(
        1 for incident in incident_list if incident.impact.user_impact_known
    )

    return {
        "incident_count": (len(incident_list)),
        "impact_levels": dict(level_counts),
        "affected_resource_count": (len(affected_resources)),
        "affected_service_count": (len(affected_services)),
        "incidents_with_known_user_impact": (user_impact_known),
    }


class ImpactAnalyzer:
    """
    Stateful impact-analysis interface for the
    intelligence pipeline.
    """

    def __init__(
        self,
        registry: ResourceRegistry,
        *,
        topology: Optional[TopologyGraph] = None,
        config: Optional[ImpactConfig] = None,
    ) -> None:
        self.registry = registry

        self.topology = topology

        self.config = config or ImpactConfig()

        self.config.validate()

    def analyze(
        self,
        incidents: Iterable[IncidentCandidate],
        anomalies: Iterable[Anomaly],
    ) -> List[IncidentCandidate]:
        return analyze_impacts(
            incidents,
            self.registry,
            anomalies,
            topology=self.topology,
            config=self.config,
        )
