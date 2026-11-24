from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.correlation.schema import (
    CorrelationCandidate,
    CorrelationDecision,
    CorrelationEvidence,
    CorrelationScore,
    EvidenceType,
    checked_score,
    combine_independent_scores,
    evidence_by_type,
)

from src.correlation.temporal import (
    TemporalAnalyzer,
)

from src.correlation.topology import (
    TopologyGraph,
)

from src.telemetry.identity import (
    compare_resources,
)

from src.telemetry.schema import (
    Severity,
    SignalType,
    TelemetryRecord,
)


# Correlation thresholds are OSHA design choices.
# They are configurable and are not calibrated
# probabilities.
DEFAULT_CORRELATED_THRESHOLD = 0.75

DEFAULT_CANDIDATE_THRESHOLD = 0.50

DEFAULT_WEAK_THRESHOLD = 0.25


DEFAULT_WEIGHTS = {
    "identity": 0.24,
    "temporal": 0.18,
    "topology": 0.22,
    "resource": 0.10,
    "service": 0.08,
    "signal": 0.08,
    "severity": 0.05,
    "historical": 0.05,
}


SEVERITY_SCORES = {
    Severity.TRACE.value: 0.05,
    Severity.DEBUG.value: 0.10,
    Severity.INFO.value: 0.20,
    Severity.NOTICE.value: 0.30,
    Severity.WARNING.value: 0.55,
    Severity.ERROR.value: 0.75,
    Severity.CRITICAL.value: 0.90,
    Severity.ALERT.value: 0.95,
    Severity.EMERGENCY.value: 1.00,
    Severity.UNKNOWN.value: 0.15,
}


SIGNAL_COMPATIBILITY = {
    # Event followed by log/metric/state change is a useful
    # candidate pattern, but not proof of causation.
    (
        SignalType.EVENT.value,
        SignalType.LOG.value,
    ): 0.75,
    (
        SignalType.EVENT.value,
        SignalType.METRIC.value,
    ): 0.70,
    (
        SignalType.EVENT.value,
        SignalType.STATE_CHANGE.value,
    ): 0.85,
    (
        SignalType.EVENT.value,
        SignalType.RESOURCE_STATE.value,
    ): 0.75,
    (
        SignalType.ALERT.value,
        SignalType.METRIC.value,
    ): 0.85,
    (
        SignalType.ALERT.value,
        SignalType.LOG.value,
    ): 0.75,
    (
        SignalType.LOG.value,
        SignalType.METRIC.value,
    ): 0.65,
    (
        SignalType.LOG.value,
        SignalType.STATE_CHANGE.value,
    ): 0.70,
    (
        SignalType.METRIC.value,
        SignalType.STATE_CHANGE.value,
    ): 0.70,
    (
        SignalType.RESOURCE_STATE.value,
        SignalType.STATE_CHANGE.value,
    ): 0.80,
    (
        SignalType.HEALTH.value,
        SignalType.ALERT.value,
    ): 0.85,
    (
        SignalType.SECURITY.value,
        SignalType.ALERT.value,
    ): 0.85,
}


@dataclass(frozen=True)
class ScoringConfig:
    """
    Configuration for OSHA's heuristic
    correlation scorer.
    """

    weights: Dict[
        str,
        float,
    ] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    correlated_threshold: float = DEFAULT_CORRELATED_THRESHOLD

    candidate_threshold: float = DEFAULT_CANDIDATE_THRESHOLD

    weak_threshold: float = DEFAULT_WEAK_THRESHOLD

    topology_max_depth: int = 6

    temporal_max_seconds: float = 300.0

    temporal_half_life_seconds: float = 30.0

    require_same_environment: bool = True

    reject_incompatible_environments: bool = True

    def normalized_weights(
        self,
    ) -> Dict[str, float]:
        """
        Normalize non-negative weights so their
        total is 1.0.
        """

        cleaned = {
            key: max(
                0.0,
                float(value),
            )
            for key, value in self.weights.items()
        }

        total = sum(cleaned.values())

        if total <= 0:
            return dict(DEFAULT_WEIGHTS)

        return {key: (value / total) for key, value in cleaned.items()}


def optional_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def normalized_text(
    value: Any,
) -> Optional[str]:
    text = optional_string(value)

    if text is None:
        return None

    return text.lower()


def records_compatible(
    source: TelemetryRecord,
    target: TelemetryRecord,
    config: ScoringConfig,
) -> bool:
    """
    Apply hard compatibility rules before
    combining heuristic evidence.
    """

    if (
        config.require_same_environment
        and source.resource.environment_id != target.resource.environment_id
    ):
        return False

    source.finalize()
    target.finalize()

    if source.record_id and target.record_id and source.record_id == target.record_id:
        return False

    return True


def identity_evidence(
    source: TelemetryRecord,
    target: TelemetryRecord,
) -> Optional[CorrelationEvidence]:
    """
    Compare normalized resource identities.

    Candidate identity matches remain weaker than
    automatic merges.
    """

    match = compare_resources(
        source.resource,
        target.resource,
    )

    if match.decision == "distinct":
        return None

    if match.decision == "merge":
        score = max(
            match.score,
            0.90,
        )
    else:
        score = match.score

    return CorrelationEvidence(
        evidence_type=(EvidenceType.IDENTITY.value),
        score=checked_score(score),
        source_record_id=(source.record_id),
        target_record_id=(target.record_id),
        source_resource_id=(source.resource.effective_id()),
        target_resource_id=(target.resource.effective_id()),
        description=(f"Normalized resources have {match.decision} identity evidence"),
        attributes={
            "identity_decision": (match.decision),
            "identity_score": (match.score),
            "identity_evidence": [item.to_dict() for item in match.evidence],
            "conflicts": (match.conflicts),
        },
    )


def same_resource_evidence(
    source: TelemetryRecord,
    target: TelemetryRecord,
) -> Optional[CorrelationEvidence]:
    """
    Create strong resource evidence when two
    observations explicitly reference the same
    effective resource.
    """

    source_id = source.resource.effective_id()

    target_id = target.resource.effective_id()

    if source_id != target_id:
        return None

    return CorrelationEvidence(
        evidence_type=(EvidenceType.RESOURCE.value),
        score=1.0,
        source_record_id=(source.record_id),
        target_record_id=(target.record_id),
        source_resource_id=(source_id),
        target_resource_id=(target_id),
        description=("Observations reference the same normalized resource"),
        attributes={
            "same_resource": True,
        },
    )


def topology_evidence(
    source: TelemetryRecord,
    target: TelemetryRecord,
    topology: Optional[TopologyGraph],
    *,
    max_depth: int,
) -> Optional[CorrelationEvidence]:
    """
    Score infrastructure proximity.

    Distance 0:
        same resource

    Distance 1:
        directly connected

    Larger distances decay by hop count.
    """

    if topology is None:
        return None

    source_id = source.resource.effective_id()

    target_id = target.resource.effective_id()

    path = topology.shortest_path(
        source_id,
        target_id,
        max_depth=max_depth,
        direction="both",
    )

    if path is None:
        return None

    if path.distance == 0:
        score = 1.0

    else:
        # One hop = 0.85
        # Two hops = 0.7225
        # Three hops = 0.6141
        score = 0.85**path.distance

    return CorrelationEvidence(
        evidence_type=(EvidenceType.TOPOLOGY.value),
        score=checked_score(score),
        source_record_id=(source.record_id),
        target_record_id=(target.record_id),
        source_resource_id=(source_id),
        target_resource_id=(target_id),
        description=(f"Resources are {path.distance} topology hops apart"),
        attributes={
            "distance": (path.distance),
            "resource_path": (path.resource_ids),
            "relationship_path": (path.relationships),
        },
    )


def service_evidence(
    source: TelemetryRecord,
    target: TelemetryRecord,
) -> Optional[CorrelationEvidence]:
    source_service = normalized_text(
        source.resource.service
        or source.signal.attributes.get("service")
        or source.signal.attributes.get("service_name")
    )

    target_service = normalized_text(
        target.resource.service
        or target.signal.attributes.get("service")
        or target.signal.attributes.get("service_name")
    )

    if source_service is None or source_service != target_service:
        return None

    return CorrelationEvidence(
        evidence_type=(EvidenceType.SERVICE.value),
        score=0.80,
        source_record_id=(source.record_id),
        target_record_id=(target.record_id),
        source_resource_id=(source.resource.effective_id()),
        target_resource_id=(target.resource.effective_id()),
        description=("Observations reference the same logical service"),
        attributes={
            "service": (source_service),
        },
    )


def signal_compatibility_score(
    source_signal_type: str,
    target_signal_type: str,
) -> float:
    """
    Return compatibility strength between two
    common signal types.
    """

    if source_signal_type == target_signal_type:
        if source_signal_type in {
            SignalType.EVENT.value,
            SignalType.ALERT.value,
            SignalType.STATE_CHANGE.value,
        }:
            return 0.70

        return 0.50

    direct = SIGNAL_COMPATIBILITY.get(
        (
            source_signal_type,
            target_signal_type,
        )
    )

    if direct is not None:
        return direct

    reverse = SIGNAL_COMPATIBILITY.get(
        (
            target_signal_type,
            source_signal_type,
        )
    )

    if reverse is not None:
        return reverse

    return 0.20


def signal_evidence(
    source: TelemetryRecord,
    target: TelemetryRecord,
) -> CorrelationEvidence:
    source_type = source.signal.signal_type

    target_type = target.signal.signal_type

    score = signal_compatibility_score(
        source_type,
        target_type,
    )

    source_name = normalized_text(source.signal.signal_name)

    target_name = normalized_text(target.signal.signal_name)

    same_name = source_name is not None and source_name == target_name

    if same_name:
        score = combine_independent_scores(
            [
                score,
                0.70,
            ]
        )

    return CorrelationEvidence(
        evidence_type=(EvidenceType.SIGNAL.value),
        score=checked_score(score),
        source_record_id=(source.record_id),
        target_record_id=(target.record_id),
        source_resource_id=(source.resource.effective_id()),
        target_resource_id=(target.resource.effective_id()),
        description=("Signal types are evaluated for semantic compatibility"),
        attributes={
            "source_signal_type": (source_type),
            "target_signal_type": (target_type),
            "source_signal_name": (source.signal.signal_name),
            "target_signal_name": (target.signal.signal_name),
            "same_signal_name": (same_name),
        },
    )


def severity_value(
    severity: str,
) -> float:
    return SEVERITY_SCORES.get(
        severity,
        SEVERITY_SCORES[Severity.UNKNOWN.value],
    )


def severity_evidence(
    source: TelemetryRecord,
    target: TelemetryRecord,
) -> CorrelationEvidence:
    source_score = severity_value(source.signal.severity)

    target_score = severity_value(target.signal.severity)

    # The stronger severity is relevant, while
    # the weaker signal still contributes.
    score = (
        max(
            source_score,
            target_score,
        )
        * 0.75
        + min(
            source_score,
            target_score,
        )
        * 0.25
    )

    return CorrelationEvidence(
        evidence_type=(EvidenceType.SEVERITY.value),
        score=checked_score(score),
        source_record_id=(source.record_id),
        target_record_id=(target.record_id),
        source_resource_id=(source.resource.effective_id()),
        target_resource_id=(target.resource.effective_id()),
        description=("Signal severities contribute correlation priority"),
        attributes={
            "source_severity": (source.signal.severity),
            "target_severity": (target.signal.severity),
        },
    )


def historical_evidence(
    source: TelemetryRecord,
    target: TelemetryRecord,
    historical_score: Optional[float],
) -> Optional[CorrelationEvidence]:
    """
    Convert an externally calculated history
    score into evidence.

    memory/ will eventually provide this value.
    """

    if historical_score is None:
        return None

    score = checked_score(historical_score)

    if score <= 0:
        return None

    return CorrelationEvidence(
        evidence_type=(EvidenceType.HISTORICAL.value),
        score=score,
        source_record_id=(source.record_id),
        target_record_id=(target.record_id),
        source_resource_id=(source.resource.effective_id()),
        target_resource_id=(target.resource.effective_id()),
        description=("Historical observations support this correlation candidate"),
    )


def collect_evidence(
    source: TelemetryRecord,
    target: TelemetryRecord,
    *,
    topology: Optional[TopologyGraph] = None,
    temporal: Optional[TemporalAnalyzer] = None,
    config: Optional[ScoringConfig] = None,
    historical_score: Optional[float] = None,
) -> List[CorrelationEvidence]:
    """
    Generate correlation evidence for one pair.

    This function does not assign the final
    candidate decision.
    """

    configuration = config or ScoringConfig()

    if not records_compatible(
        source,
        target,
        configuration,
    ):
        return []

    source.finalize()
    target.finalize()

    temporal_analyzer = temporal or TemporalAnalyzer(
        maximum_distance_seconds=(configuration.temporal_max_seconds),
        half_life_seconds=(configuration.temporal_half_life_seconds),
    )

    evidence: List[CorrelationEvidence] = []

    temporal_item = temporal_analyzer.evidence(
        source,
        target,
    )

    if temporal_item is not None:
        evidence.append(temporal_item)

    resource_item = same_resource_evidence(
        source,
        target,
    )

    if resource_item is not None:
        evidence.append(resource_item)

    identity_item = identity_evidence(
        source,
        target,
    )

    if identity_item is not None:
        evidence.append(identity_item)

    topology_item = topology_evidence(
        source,
        target,
        topology,
        max_depth=(configuration.topology_max_depth),
    )

    if topology_item is not None:
        evidence.append(topology_item)

    service_item = service_evidence(
        source,
        target,
    )

    if service_item is not None:
        evidence.append(service_item)

    evidence.append(
        signal_evidence(
            source,
            target,
        )
    )

    evidence.append(
        severity_evidence(
            source,
            target,
        )
    )

    history_item = historical_evidence(
        source,
        target,
        historical_score,
    )

    if history_item is not None:
        evidence.append(history_item)

    return evidence


def dimension_score(
    evidence: Iterable[CorrelationEvidence],
    evidence_type: str,
) -> float:
    """
    Combine evidence inside one scoring
    dimension.
    """

    scores = [
        item.normalized_score()
        for item in evidence
        if item.evidence_type == evidence_type
    ]

    return combine_independent_scores(scores)


def calculate_score(
    evidence: Iterable[CorrelationEvidence],
    *,
    config: Optional[ScoringConfig] = None,
) -> CorrelationScore:
    """
    Calculate a transparent weighted score.

    Missing dimensions contribute zero rather
    than being silently removed from the
    weighting model.
    """

    configuration = config or ScoringConfig()

    evidence_list = list(evidence)

    identity = dimension_score(
        evidence_list,
        EvidenceType.IDENTITY.value,
    )

    temporal = dimension_score(
        evidence_list,
        EvidenceType.TEMPORAL.value,
    )

    topology = dimension_score(
        evidence_list,
        EvidenceType.TOPOLOGY.value,
    )

    resource = dimension_score(
        evidence_list,
        EvidenceType.RESOURCE.value,
    )

    service = dimension_score(
        evidence_list,
        EvidenceType.SERVICE.value,
    )

    signal = dimension_score(
        evidence_list,
        EvidenceType.SIGNAL.value,
    )

    severity = dimension_score(
        evidence_list,
        EvidenceType.SEVERITY.value,
    )

    historical = dimension_score(
        evidence_list,
        EvidenceType.HISTORICAL.value,
    )

    dimensions = {
        "identity": identity,
        "temporal": temporal,
        "topology": topology,
        "resource": resource,
        "service": service,
        "signal": signal,
        "severity": severity,
        "historical": historical,
    }

    weights = configuration.normalized_weights()

    overall = sum(
        dimensions.get(
            dimension,
            0.0,
        )
        * weight
        for dimension, weight in weights.items()
    )

    return CorrelationScore(
        identity=identity,
        temporal=temporal,
        topology=topology,
        resource=resource,
        service=service,
        signal=signal,
        severity=severity,
        historical=historical,
        overall=checked_score(overall),
    )


def decide_correlation(
    scores: CorrelationScore,
    *,
    config: Optional[ScoringConfig] = None,
    evidence: Optional[Iterable[CorrelationEvidence]] = None,
) -> str:
    """
    Convert scores into a correlation decision.

    A medium weighted score without any temporal,
    identity, resource, or topology evidence is
    prevented from becoming correlated based on
    severity/signal compatibility alone.
    """

    configuration = config or ScoringConfig()

    normalized = scores.normalized()

    structural_support = max(
        normalized.identity,
        normalized.resource,
        normalized.topology,
    )

    temporal_support = normalized.temporal

    has_context = (
        structural_support > 0 or temporal_support > 0 or normalized.historical > 0
    )

    if normalized.overall >= configuration.correlated_threshold and has_context:
        return CorrelationDecision.CORRELATED.value

    if normalized.overall >= configuration.candidate_threshold and has_context:
        return CorrelationDecision.CANDIDATE.value

    if normalized.overall >= configuration.weak_threshold:
        return CorrelationDecision.WEAK.value

    return CorrelationDecision.REJECTED.value


def score_candidate(
    candidate: CorrelationCandidate,
    *,
    config: Optional[ScoringConfig] = None,
) -> CorrelationCandidate:
    """
    Score and classify an existing candidate.
    """

    configuration = config or ScoringConfig()

    candidate.scores = calculate_score(
        candidate.evidence,
        config=configuration,
    )

    candidate.decision = decide_correlation(
        candidate.scores,
        config=configuration,
        evidence=(candidate.evidence),
    )

    return candidate.finalize()


def score_records(
    source: TelemetryRecord,
    target: TelemetryRecord,
    *,
    topology: Optional[TopologyGraph] = None,
    temporal: Optional[TemporalAnalyzer] = None,
    config: Optional[ScoringConfig] = None,
    historical_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Diagnostic helper for inspecting why two
    records would or would not correlate.
    """

    configuration = config or ScoringConfig()

    evidence = collect_evidence(
        source,
        target,
        topology=topology,
        temporal=temporal,
        config=configuration,
        historical_score=(historical_score),
    )

    scores = calculate_score(
        evidence,
        config=configuration,
    )

    decision = decide_correlation(
        scores,
        config=configuration,
        evidence=evidence,
    )

    return {
        "source_record_id": (source.record_id),
        "target_record_id": (target.record_id),
        "decision": decision,
        "scores": (scores.to_dict()),
        "evidence": [item.to_dict() for item in evidence],
    }


class CorrelationScorer:
    """
    Reusable scorer for correlate.py and the
    runtime pipeline.
    """

    def __init__(
        self,
        *,
        config: Optional[ScoringConfig] = None,
        topology: Optional[TopologyGraph] = None,
        temporal: Optional[TemporalAnalyzer] = None,
    ) -> None:
        self.config = config or ScoringConfig()

        self.topology = topology

        self.temporal = temporal or TemporalAnalyzer(
            maximum_distance_seconds=(self.config.temporal_max_seconds),
            half_life_seconds=(self.config.temporal_half_life_seconds),
        )

    def evidence(
        self,
        source: TelemetryRecord,
        target: TelemetryRecord,
        *,
        historical_score: Optional[float] = None,
    ) -> List[CorrelationEvidence]:
        return collect_evidence(
            source,
            target,
            topology=self.topology,
            temporal=self.temporal,
            config=self.config,
            historical_score=(historical_score),
        )

    def scores(
        self,
        evidence: Iterable[CorrelationEvidence],
    ) -> CorrelationScore:
        return calculate_score(
            evidence,
            config=self.config,
        )

    def candidate(
        self,
        candidate: CorrelationCandidate,
    ) -> CorrelationCandidate:
        return score_candidate(
            candidate,
            config=self.config,
        )

    def compare(
        self,
        source: TelemetryRecord,
        target: TelemetryRecord,
        *,
        historical_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        return score_records(
            source,
            target,
            topology=self.topology,
            temporal=self.temporal,
            config=self.config,
            historical_score=(historical_score),
        )
