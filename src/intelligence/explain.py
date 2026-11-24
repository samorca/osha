from __future__ import annotations

from collections import Counter
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
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
    EvidenceReference,
    Explanation,
    IncidentCandidate,
    Pattern,
    RootCauseStatus,
)

from src.telemetry.schema import (
    TelemetryRecord,
)


# ---------------------------------------------------------------
# Lookup helpers
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

    output.sort(key=lambda item: -item.score)

    return output


def patterns_for_incident(
    incident: IncidentCandidate,
    patterns: Iterable[Pattern],
) -> List[Pattern]:
    wanted_ids = set(incident.pattern_ids)

    wanted_records = set(incident.record_ids)

    wanted_resources = set(incident.resource_ids)

    output = []

    for pattern in patterns:
        pattern.finalize()

        if pattern.pattern_id and pattern.pattern_id in wanted_ids:
            output.append(pattern)

            continue

        if wanted_records.intersection(pattern.record_ids):
            output.append(pattern)

            continue

        if wanted_resources.intersection(pattern.resource_ids):
            output.append(pattern)

    output.sort(
        key=lambda item: (
            -item.occurrence_count,
            item.pattern_type,
        )
    )

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
# Resource descriptions
# ---------------------------------------------------------------


def resource_label(
    resource_id: str,
    registry: ResourceRegistry,
) -> str:
    resource = registry.get_resource(resource_id)

    if resource is None:
        return resource_id

    label = (
        resource.name
        or resource.service
        or resource.host
        or resource.provider_id
        or resource_id
    )

    return f"{label} ({resource.resource_type}, {resource.provider})"


def resource_labels(
    resource_ids: Iterable[str],
    registry: ResourceRegistry,
) -> List[str]:
    return [
        resource_label(
            resource_id,
            registry,
        )
        for resource_id in sorted(set(resource_ids))
    ]


# ---------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------


def incident_timeline(
    records: Iterable[TelemetryRecord],
    registry: ResourceRegistry,
    *,
    maximum_items: int = 10,
) -> List[str]:
    """
    Produce concise timeline statements.

    This reports sequence only. It does not
    convert temporal order into causation.
    """

    records_list = list(records)

    def sort_key(
        telemetry: TelemetryRecord,
    ) -> str:
        return telemetry.signal.timestamp or ""

    records_list.sort(key=sort_key)

    findings = []

    for telemetry in records_list[:maximum_items]:
        resource_id = telemetry.resource.effective_id()

        label = resource_label(
            resource_id,
            registry,
        )

        signal_name = telemetry.signal.signal_name or telemetry.signal.signal_type

        statement = f"{telemetry.signal.timestamp}: {label} reported {signal_name}"

        if telemetry.signal.severity:
            statement += f" with severity {telemetry.signal.severity}"

        findings.append(statement + ".")

    if len(records_list) > maximum_items:
        findings.append(
            f"{len(records_list) - maximum_items} "
            "additional telemetry observation(s) "
            "are part of this incident candidate."
        )

    return findings


# ---------------------------------------------------------------
# Root cause explanation
# ---------------------------------------------------------------


def root_cause_findings(
    incident: IncidentCandidate,
    registry: ResourceRegistry,
) -> List[str]:
    findings = []

    if not incident.root_causes:
        findings.append(
            "No root-cause hypothesis met the configured minimum evidence score."
        )

        return findings

    strongest = incident.root_causes[0]

    label = resource_label(
        strongest.resource_id,
        registry,
    )

    if strongest.status == RootCauseStatus.CONFIRMED.value:
        prefix = "Confirmed root cause"

    elif strongest.status == RootCauseStatus.LIKELY.value:
        prefix = "Strongest root-cause hypothesis"

    else:
        prefix = "Leading root-cause hypothesis"

    findings.append(f"{prefix}: {label}. Evidence strength {strongest.score:.2f}.")

    dimensions = strongest.attributes.get(
        "dimension_scores",
        {},
    )

    if isinstance(
        dimensions,
        dict,
    ):
        strongest_dimensions = sorted(
            (
                (
                    name,
                    float(value),
                )
                for name, value in dimensions.items()
                if isinstance(
                    value,
                    (
                        int,
                        float,
                    ),
                )
                and value > 0
            ),
            key=lambda item: -item[1],
        )

        if strongest_dimensions:
            dimension_text = ", ".join(
                (f"{name}={value:.2f}") for name, value in strongest_dimensions
            )

            findings.append(f"Root-cause ranking was supported by: {dimension_text}.")

    if len(strongest.causal_path) > 1:
        path_labels = resource_labels(
            strongest.causal_path,
            registry,
        )

        findings.append(
            "A relevant topology path was observed: " + " -> ".join(path_labels) + "."
        )

    if strongest.status != RootCauseStatus.CONFIRMED.value:
        findings.append(
            "This root-cause result remains "
            "a hypothesis rather than a "
            "confirmed causal conclusion."
        )

    if len(incident.root_causes) > 1:
        alternatives = []

        for candidate in incident.root_causes[1:4]:
            alternatives.append(
                (
                    resource_label(
                        candidate.resource_id,
                        registry,
                    )
                    + (f" ({candidate.score:.2f})")
                )
            )

        if alternatives:
            findings.append("Other ranked hypotheses: " + ", ".join(alternatives) + ".")

    return findings


# ---------------------------------------------------------------
# Impact explanation
# ---------------------------------------------------------------


def impact_findings(
    incident: IncidentCandidate,
    registry: ResourceRegistry,
) -> List[str]:
    impact = incident.impact

    findings = []

    directly_affected = resource_labels(
        impact.affected_resource_ids,
        registry,
    )

    findings.append(
        (
            f"Estimated impact level: "
            f"{impact.level} "
            f"with evidence strength "
            f"{impact.score:.2f}."
        )
    )

    if directly_affected:
        findings.append(
            "Resources with direct impact "
            "evidence: " + ", ".join(directly_affected) + "."
        )

    if impact.affected_services:
        findings.append(
            "Known affected service "
            "context: " + ", ".join(impact.affected_services) + "."
        )

    if impact.affected_providers:
        findings.append(
            "Telemetry providers represented "
            "in direct impact evidence: " + ", ".join(impact.affected_providers) + "."
        )

    exposed = impact.attributes.get(
        "potentially_exposed_resources",
        {},
    )

    if (
        isinstance(
            exposed,
            dict,
        )
        and exposed
    ):
        exposed_labels = resource_labels(
            exposed.keys(),
            registry,
        )

        findings.append(
            "Topology-near resources that may "
            "require checking, but are not "
            "confirmed impacted: " + ", ".join(exposed_labels) + "."
        )

    if not impact.user_impact_known:
        findings.append(
            "User or business impact has not been independently established."
        )

    return findings


# ---------------------------------------------------------------
# Anomaly and pattern explanation
# ---------------------------------------------------------------


def anomaly_findings(
    anomalies: Iterable[Anomaly],
) -> List[str]:
    anomaly_list = list(anomalies)

    if not anomaly_list:
        return [("No anomaly finding was associated with this incident candidate.")]

    type_counts = Counter(anomaly.anomaly_type for anomaly in anomaly_list)

    severity_counts = Counter(anomaly.severity for anomaly in anomaly_list)

    strongest = max(
        anomaly_list,
        key=lambda item: item.score,
    )

    findings = [
        (f"{len(anomaly_list)} anomaly finding(s) support this incident candidate."),
        (
            "Anomaly types: "
            + ", ".join(
                (f"{name}={count}") for name, count in sorted(type_counts.items())
            )
            + "."
        ),
        (
            "Anomaly severities: "
            + ", ".join(
                (f"{name}={count}") for name, count in sorted(severity_counts.items())
            )
            + "."
        ),
        (f"Strongest anomaly: {strongest.title} ({strongest.score:.2f})."),
    ]

    return findings


def pattern_findings(
    patterns: Iterable[Pattern],
) -> List[str]:
    patterns_list = list(patterns)

    if not patterns_list:
        return []

    repeated = [pattern for pattern in patterns_list if pattern.occurrence_count > 1]

    findings = [
        (f"{len(patterns_list)} pattern(s) were associated with the incident evidence.")
    ]

    if repeated:
        strongest_repeat = max(
            repeated,
            key=lambda item: item.occurrence_count,
        )

        findings.append(
            "Most repeated pattern: "
            f"{strongest_repeat.name} "
            f"({strongest_repeat.occurrence_count} "
            "occurrences)."
        )

    return findings


# ---------------------------------------------------------------
# Evidence references
# ---------------------------------------------------------------


def collect_evidence(
    incident: IncidentCandidate,
    anomalies: Iterable[Anomaly],
) -> List[EvidenceReference]:
    """
    Collect structured evidence for operator
    inspection.

    Evidence is deduplicated using type,
    description, record IDs and resource IDs.
    """

    output = []

    output.extend(incident.impact.evidence)

    output.extend(
        evidence
        for candidate in incident.root_causes
        for evidence in candidate.evidence
    )

    output.extend(evidence for anomaly in anomalies for evidence in anomaly.evidence)

    seen = set()

    deduplicated = []

    for evidence in output:
        key = (
            evidence.evidence_type,
            evidence.description,
            tuple(sorted(evidence.record_ids)),
            tuple(sorted(evidence.resource_ids)),
        )

        if key in seen:
            continue

        seen.add(key)

        deduplicated.append(evidence)

    deduplicated.sort(
        key=lambda item: (
            -item.score,
            item.evidence_type,
        )
    )

    return deduplicated


# ---------------------------------------------------------------
# Investigation suggestions
# ---------------------------------------------------------------


def investigation_suggestions(
    incident: IncidentCandidate,
    registry: ResourceRegistry,
) -> List[str]:
    """
    Suggest evidence-gathering steps only.

    This module intentionally does not produce
    automated remediation instructions.
    """

    suggestions = []

    if incident.root_causes:
        strongest = incident.root_causes[0]

        label = resource_label(
            strongest.resource_id,
            registry,
        )

        suggestions.append(
            "Verify the leading root-cause "
            f"hypothesis against {label}'s "
            "native logs, state, and recent "
            "configuration/change history."
        )

        if strongest.status != RootCauseStatus.CONFIRMED.value:
            suggestions.append(
                "Look for independent evidence "
                "that could confirm or reject "
                "the leading root-cause "
                "hypothesis."
            )

    else:
        suggestions.append(
            "Gather additional resource, state, "
            "and timeline evidence because no "
            "root-cause hypothesis currently "
            "meets the configured threshold."
        )

    if not (incident.impact.user_impact_known):
        suggestions.append(
            "Check service health, availability, "
            "request/error telemetry, or other "
            "authoritative service indicators "
            "before asserting user impact."
        )

    if any(
        "only one provider" in limitation
        for limitation in incident.confidence.limitations
    ):
        suggestions.append(
            "Corroborate the incident with an "
            "independent telemetry source when "
            "one becomes available."
        )

    if any(
        "historical" in limitation.lower()
        for limitation in incident.confidence.limitations
    ):
        suggestions.append(
            "Compare this episode with verified "
            "historical incidents or normal "
            "baseline behavior when historical "
            "memory becomes available."
        )

    if any(
        "topology" in limitation.lower()
        for limitation in incident.confidence.limitations
    ):
        suggestions.append(
            "Collect or verify infrastructure "
            "relationships for the affected "
            "resources before drawing broader "
            "blast-radius conclusions."
        )

    return list(dict.fromkeys(suggestions))


# ---------------------------------------------------------------
# Main explanation
# ---------------------------------------------------------------


def explanation_headline(
    incident: IncidentCandidate,
) -> str:
    return f"{incident.severity.upper()} incident candidate: {incident.title}"


def explanation_summary(
    incident: IncidentCandidate,
) -> str:
    """
    Produce a conservative one-paragraph machine
    summary from structured results.
    """

    resource_count = len(incident.resource_ids)

    anomaly_count = len(incident.anomaly_ids)

    text = (
        f"OSHA identified a correlated episode "
        f"in environment "
        f"{incident.environment_id} involving "
        f"{resource_count} resource(s) and "
        f"{anomaly_count} anomaly finding(s). "
        f"The estimated impact is "
        f"{incident.impact.level}. "
        f"Overall evidence confidence is "
        f"{incident.confidence.level} "
        f"({incident.confidence.overall:.2f})."
    )

    if incident.root_causes:
        strongest = incident.root_causes[0]

        if strongest.status == RootCauseStatus.CONFIRMED.value:
            text += " A root cause is marked confirmed by the structured evidence."

        else:
            text += (
                " The leading root-cause result "
                "remains a hypothesis requiring "
                "further validation."
            )

    else:
        text += (
            " No root-cause hypothesis currently "
            "meets the configured evidence "
            "threshold."
        )

    return text


def explain_incident(
    incident: IncidentCandidate,
    registry: ResourceRegistry,
    anomalies: Iterable[Anomaly],
    patterns: Iterable[Pattern],
    groups: Iterable[CorrelationGroup],
    *,
    topology: Optional[TopologyGraph] = None,
) -> Explanation:
    """
    Generate a human-readable evidence-grounded
    explanation.

    No new anomaly, incident, impact or root-cause
    conclusion is created here.
    """

    incident.finalize()

    records = records_for_incident(
        incident,
        registry,
    )

    incident_anomalies = anomalies_for_incident(
        incident,
        anomalies,
    )

    incident_patterns = patterns_for_incident(
        incident,
        patterns,
    )

    incident_groups = groups_for_incident(
        incident,
        groups,
    )

    findings = []

    findings.extend(
        incident_timeline(
            records,
            registry,
        )
    )

    findings.extend(anomaly_findings(incident_anomalies))

    findings.extend(pattern_findings(incident_patterns))

    findings.extend(
        root_cause_findings(
            incident,
            registry,
        )
    )

    findings.extend(
        impact_findings(
            incident,
            registry,
        )
    )

    if incident_groups:
        strongest_group = max(
            incident_groups,
            key=lambda group: group.strongest_score,
        )

        findings.append(
            "Strongest correlation-group "
            "evidence strength: "
            f"{strongest_group.strongest_score:.2f}; "
            "average group evidence strength: "
            f"{strongest_group.average_score:.2f}."
        )

    limitations = list(incident.confidence.limitations)

    evidence = collect_evidence(
        incident,
        incident_anomalies,
    )

    recommendations = investigation_suggestions(
        incident,
        registry,
    )

    return Explanation(
        headline=(explanation_headline(incident)),
        summary=(explanation_summary(incident)),
        incident_id=(incident.incident_id),
        findings=findings,
        evidence=evidence,
        limitations=limitations,
        recommended_investigations=(recommendations),
    )


def explain_incidents(
    incidents: Iterable[IncidentCandidate],
    registry: ResourceRegistry,
    anomalies: Iterable[Anomaly],
    patterns: Iterable[Pattern],
    groups: Iterable[CorrelationGroup],
    *,
    topology: Optional[TopologyGraph] = None,
) -> List[Explanation]:
    anomaly_list = list(anomalies)

    pattern_list = list(patterns)

    group_list = list(groups)

    return [
        explain_incident(
            incident,
            registry,
            anomaly_list,
            pattern_list,
            group_list,
            topology=topology,
        )
        for incident in incidents
    ]


# ---------------------------------------------------------------
# Operator-friendly text rendering
# ---------------------------------------------------------------


def render_explanation(
    explanation: Explanation,
) -> str:
    """
    Render an Explanation for CLI/logging.

    Structured JSON remains the primary machine
    representation.
    """

    lines = [
        explanation.headline,
        "",
        explanation.summary,
    ]

    if explanation.findings:
        lines.extend(
            [
                "",
                "Findings:",
            ]
        )

        lines.extend((f"- {finding}") for finding in explanation.findings)

    if explanation.limitations:
        lines.extend(
            [
                "",
                "Limitations:",
            ]
        )

        lines.extend((f"- {limitation}") for limitation in explanation.limitations)

    if explanation.recommended_investigations:
        lines.extend(
            [
                "",
                "Recommended investigations:",
            ]
        )

        lines.extend(
            (f"- {recommendation}")
            for recommendation in explanation.recommended_investigations
        )

    return "\n".join(lines)


def explanation_summary_stats(
    explanations: Iterable[Explanation],
) -> Dict[str, Any]:
    explanation_list = list(explanations)

    evidence_count = sum(len(explanation.evidence) for explanation in explanation_list)

    limitation_count = sum(
        len(explanation.limitations) for explanation in explanation_list
    )

    investigation_count = sum(
        len(explanation.recommended_investigations) for explanation in explanation_list
    )

    return {
        "explanation_count": (len(explanation_list)),
        "evidence_reference_count": (evidence_count),
        "limitation_count": (limitation_count),
        "recommended_investigation_count": (investigation_count),
    }


class ExplanationEngine:
    """
    Operator-facing explanation component.
    """

    def __init__(
        self,
        registry: ResourceRegistry,
        *,
        topology: Optional[TopologyGraph] = None,
    ) -> None:
        self.registry = registry

        self.topology = topology

    def explain(
        self,
        incidents: Iterable[IncidentCandidate],
        anomalies: Iterable[Anomaly],
        patterns: Iterable[Pattern],
        groups: Iterable[CorrelationGroup],
    ) -> List[Explanation]:
        return explain_incidents(
            incidents,
            self.registry,
            anomalies,
            patterns,
            groups,
            topology=self.topology,
        )

    def render(
        self,
        explanation: Explanation,
    ) -> str:
        return render_explanation(explanation)
