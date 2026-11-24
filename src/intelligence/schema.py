from __future__ import annotations

import hashlib
import json
import math
from dataclasses import (
    asdict,
    dataclass,
    field,
)
from datetime import (
    datetime,
    timezone,
)
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)


INTELLIGENCE_SCHEMA_VERSION = "1"


class AnomalyType(str, Enum):
    LOG_PATTERN = "log_pattern"

    METRIC_DEVIATION = "metric_deviation"

    STATE_CHANGE = "state_change"

    EVENT_BURST = "event_burst"

    RESOURCE_HEALTH = "resource_health"

    SECURITY = "security"

    TOPOLOGY_CHANGE = "topology_change"

    UNKNOWN = "unknown"


class IncidentStatus(str, Enum):
    CANDIDATE = "candidate"

    OPEN = "open"

    INVESTIGATING = "investigating"

    MITIGATING = "mitigating"

    MONITORING = "monitoring"

    RESOLVED = "resolved"

    CLOSED = "closed"


class IncidentSeverity(str, Enum):
    INFO = "info"

    LOW = "low"

    MEDIUM = "medium"

    HIGH = "high"

    CRITICAL = "critical"

    UNKNOWN = "unknown"


class RootCauseStatus(str, Enum):
    SUSPECTED = "suspected"

    LIKELY = "likely"

    CONFIRMED = "confirmed"

    REJECTED = "rejected"


class ImpactLevel(str, Enum):
    NONE = "none"

    LOCALIZED = "localized"

    LIMITED = "limited"

    SIGNIFICANT = "significant"

    WIDESPREAD = "widespread"

    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    NONE = "none"

    LOW = "low"

    MEDIUM = "medium"

    HIGH = "high"

    VERY_HIGH = "very_high"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checked_score(
    value: Any,
) -> float:
    """
    Normalize a score into the range 0.0 to 1.0.

    These scores represent OSHA evidence
    strengths unless explicitly calibrated by a
    later model.
    """

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if not math.isfinite(number):
        return 0.0

    return min(
        1.0,
        max(
            0.0,
            number,
        ),
    )


def normalize_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def json_safe(
    value: Any,
    *,
    depth: int = 0,
    maximum_depth: int = 20,
) -> Any:
    """
    Convert intelligence output into JSON-safe
    structures.
    """

    if depth > maximum_depth:
        return "[MAX_DEPTH_REACHED]"

    if value is None or isinstance(
        value,
        (
            bool,
            int,
            str,
        ),
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if math.isfinite(value):
            return value

        return None

    if isinstance(
        value,
        datetime,
    ):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.isoformat()

    if isinstance(
        value,
        bytes,
    ):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): json_safe(
                item,
                depth=depth + 1,
                maximum_depth=(maximum_depth),
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        return [
            json_safe(
                item,
                depth=depth + 1,
                maximum_depth=(maximum_depth),
            )
            for item in value
        ]

    if hasattr(
        value,
        "__dataclass_fields__",
    ):
        return json_safe(
            asdict(value),
            depth=depth + 1,
            maximum_depth=(maximum_depth),
        )

    return str(value)


def stable_hash(
    value: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        json_safe(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EvidenceReference:
    """
    Evidence supporting an anomaly, root-cause
    candidate, impact assessment or explanation.
    """

    evidence_type: str

    description: str

    score: float = 0.0

    record_ids: List[str] = field(default_factory=list)

    resource_ids: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "evidence_type": (self.evidence_type),
            "description": (self.description),
            "score": checked_score(self.score),
            "record_ids": sorted(set(self.record_ids)),
            "resource_ids": sorted(set(self.resource_ids)),
            "attributes": json_safe(self.attributes),
        }


@dataclass
class Pattern:
    """
    Repeated or newly discovered operational
    pattern.

    Pattern does not automatically mean anomaly.
    """

    pattern_type: str

    name: str

    fingerprint: str

    environment_id: str

    occurrence_count: int = 1

    record_ids: List[str] = field(default_factory=list)

    resource_ids: List[str] = field(default_factory=list)

    providers: List[str] = field(default_factory=list)

    signal_types: List[str] = field(default_factory=list)

    first_seen: Optional[str] = None

    last_seen: Optional[str] = None

    novelty_score: float = 0.0

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    pattern_id: Optional[str] = None

    def finalize(
        self,
    ) -> "Pattern":
        self.record_ids = sorted(set(self.record_ids))

        self.resource_ids = sorted(set(self.resource_ids))

        self.providers = sorted(set(self.providers))

        self.signal_types = sorted(set(self.signal_types))

        self.novelty_score = checked_score(self.novelty_score)

        self.occurrence_count = max(
            0,
            int(self.occurrence_count),
        )

        if not self.pattern_id:
            self.pattern_id = "intel:pattern:" + stable_hash(
                {
                    "pattern_type": (self.pattern_type),
                    "fingerprint": (self.fingerprint),
                    "environment_id": (self.environment_id),
                }
            )

        return self

    def to_dict(self) -> dict:
        self.finalize()

        return json_safe(asdict(self))


@dataclass
class Anomaly:
    """
    One abnormal observation or correlated set of
    abnormal observations.

    An anomaly is not automatically an incident.
    """

    anomaly_type: str

    title: str

    environment_id: str

    score: float

    severity: str = IncidentSeverity.UNKNOWN.value

    description: Optional[str] = None

    record_ids: List[str] = field(default_factory=list)

    resource_ids: List[str] = field(default_factory=list)

    pattern_ids: List[str] = field(default_factory=list)

    evidence: List[EvidenceReference] = field(default_factory=list)

    detected_at: str = field(default_factory=utc_now)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    anomaly_id: Optional[str] = None

    def finalize(
        self,
    ) -> "Anomaly":
        self.score = checked_score(self.score)

        self.record_ids = sorted(set(self.record_ids))

        self.resource_ids = sorted(set(self.resource_ids))

        self.pattern_ids = sorted(set(self.pattern_ids))

        if not self.anomaly_id:
            self.anomaly_id = "intel:anomaly:" + stable_hash(
                {
                    "anomaly_type": (self.anomaly_type),
                    "environment_id": (self.environment_id),
                    "record_ids": (self.record_ids),
                    "resource_ids": (self.resource_ids),
                    "detected_at": (self.detected_at),
                }
            )

        return self

    def to_dict(self) -> dict:
        self.finalize()

        data = asdict(self)

        data["evidence"] = [item.to_dict() for item in self.evidence]

        return json_safe(data)


@dataclass
class RootCauseCandidate:
    """
    Ranked possible origin of an incident.

    Suspected and likely candidates are not
    confirmed causes.
    """

    title: str

    resource_id: str

    score: float

    status: str = RootCauseStatus.SUSPECTED.value

    description: Optional[str] = None

    contributing_record_ids: List[str] = field(default_factory=list)

    contributing_anomaly_ids: List[str] = field(default_factory=list)

    evidence: List[EvidenceReference] = field(default_factory=list)

    causal_path: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    candidate_id: Optional[str] = None

    def finalize(
        self,
    ) -> "RootCauseCandidate":
        self.score = checked_score(self.score)

        self.contributing_record_ids = sorted(set(self.contributing_record_ids))

        self.contributing_anomaly_ids = sorted(set(self.contributing_anomaly_ids))

        if not self.candidate_id:
            self.candidate_id = "intel:root-cause:" + stable_hash(
                {
                    "resource_id": (self.resource_id),
                    "title": (self.title),
                    "records": (self.contributing_record_ids),
                }
            )

        return self

    def to_dict(self) -> dict:
        self.finalize()

        data = asdict(self)

        data["evidence"] = [item.to_dict() for item in self.evidence]

        return json_safe(data)


@dataclass
class ImpactAssessment:
    """
    Estimated incident scope.

    Topology proximity alone should not be
    interpreted as confirmed user impact.
    """

    level: str = ImpactLevel.UNKNOWN.value

    score: float = 0.0

    affected_resource_ids: List[str] = field(default_factory=list)

    affected_services: List[str] = field(default_factory=list)

    affected_providers: List[str] = field(default_factory=list)

    blast_radius_hops: Optional[int] = None

    user_impact_known: bool = False

    summary: Optional[str] = None

    evidence: List[EvidenceReference] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "ImpactAssessment":
        self.score = checked_score(self.score)

        self.affected_resource_ids = sorted(set(self.affected_resource_ids))

        self.affected_services = sorted(set(self.affected_services))

        self.affected_providers = sorted(set(self.affected_providers))

        return self

    def to_dict(self) -> dict:
        self.finalize()

        data = asdict(self)

        data["evidence"] = [item.to_dict() for item in self.evidence]

        return json_safe(data)


@dataclass
class ConfidenceAssessment:
    """
    Separate evidence dimensions contributing to
    an intelligence conclusion.

    These values are not calibrated probabilities
    unless a later calibrated model explicitly
    produces them.
    """

    overall: float = 0.0

    level: str = ConfidenceLevel.NONE.value

    data_quality: float = 0.0

    correlation_strength: float = 0.0

    anomaly_strength: float = 0.0

    root_cause_strength: float = 0.0

    impact_strength: float = 0.0

    explanation: Optional[str] = None

    limitations: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "ConfidenceAssessment":
        self.overall = checked_score(self.overall)

        self.data_quality = checked_score(self.data_quality)

        self.correlation_strength = checked_score(self.correlation_strength)

        self.anomaly_strength = checked_score(self.anomaly_strength)

        self.root_cause_strength = checked_score(self.root_cause_strength)

        self.impact_strength = checked_score(self.impact_strength)

        self.limitations = sorted(set(self.limitations))

        return self

    def to_dict(self) -> dict:
        self.finalize()

        return json_safe(asdict(self))


@dataclass
class IncidentCandidate:
    """
    Intelligence-layer representation of a
    possible operational incident.

    An IncidentCandidate does not become a
    confirmed incident merely by construction.
    """

    title: str

    environment_id: str

    severity: str

    status: str = IncidentStatus.CANDIDATE.value

    summary: Optional[str] = None

    correlation_group_ids: List[str] = field(default_factory=list)

    record_ids: List[str] = field(default_factory=list)

    resource_ids: List[str] = field(default_factory=list)

    anomaly_ids: List[str] = field(default_factory=list)

    pattern_ids: List[str] = field(default_factory=list)

    root_causes: List[RootCauseCandidate] = field(default_factory=list)

    impact: ImpactAssessment = field(default_factory=(ImpactAssessment))

    confidence: ConfidenceAssessment = field(default_factory=(ConfidenceAssessment))

    started_at: Optional[str] = None

    last_observed_at: Optional[str] = None

    detected_at: str = field(default_factory=utc_now)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    incident_id: Optional[str] = None

    def finalize(
        self,
    ) -> "IncidentCandidate":
        self.correlation_group_ids = sorted(set(self.correlation_group_ids))

        self.record_ids = sorted(set(self.record_ids))

        self.resource_ids = sorted(set(self.resource_ids))

        self.anomaly_ids = sorted(set(self.anomaly_ids))

        self.pattern_ids = sorted(set(self.pattern_ids))

        self.root_causes = [item.finalize() for item in self.root_causes]

        self.root_causes.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        self.impact.finalize()

        self.confidence.finalize()

        if not self.incident_id:
            self.incident_id = "intel:incident:" + stable_hash(
                {
                    "environment_id": (self.environment_id),
                    "groups": (self.correlation_group_ids),
                    "records": (self.record_ids),
                    "started_at": (self.started_at),
                }
            )

        return self

    def to_dict(self) -> dict:
        self.finalize()

        return {
            "incident_id": (self.incident_id),
            "title": self.title,
            "environment_id": (self.environment_id),
            "severity": (self.severity),
            "status": self.status,
            "summary": self.summary,
            "correlation_group_ids": (self.correlation_group_ids),
            "record_ids": (self.record_ids),
            "resource_ids": (self.resource_ids),
            "anomaly_ids": (self.anomaly_ids),
            "pattern_ids": (self.pattern_ids),
            "root_causes": [item.to_dict() for item in self.root_causes],
            "impact": (self.impact.to_dict()),
            "confidence": (self.confidence.to_dict()),
            "started_at": (self.started_at),
            "last_observed_at": (self.last_observed_at),
            "detected_at": (self.detected_at),
            "attributes": json_safe(self.attributes),
        }


@dataclass
class Explanation:
    """
    Human-readable explanation generated from
    structured evidence.

    Recommended investigations are not automated
    remediation actions.
    """

    headline: str

    summary: str

    incident_id: Optional[str] = None

    findings: List[str] = field(default_factory=list)

    evidence: List[EvidenceReference] = field(default_factory=list)

    limitations: List[str] = field(default_factory=list)

    recommended_investigations: List[str] = field(default_factory=list)

    generated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "summary": self.summary,
            "incident_id": (self.incident_id),
            "findings": list(self.findings),
            "evidence": [item.to_dict() for item in self.evidence],
            "limitations": list(self.limitations),
            "recommended_investigations": (list(self.recommended_investigations)),
            "generated_at": (self.generated_at),
        }


@dataclass
class IntelligenceResult:
    """
    Final output of the OSHA intelligence layer.
    """

    incidents: List[IncidentCandidate] = field(default_factory=list)

    anomalies: List[Anomaly] = field(default_factory=list)

    patterns: List[Pattern] = field(default_factory=list)

    explanations: List[Explanation] = field(default_factory=list)

    unclassified_group_ids: List[str] = field(default_factory=list)

    schema_version: str = INTELLIGENCE_SCHEMA_VERSION

    generated_at: str = field(default_factory=utc_now)

    metadata: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "IntelligenceResult":
        self.incidents = [item.finalize() for item in self.incidents]

        self.anomalies = [item.finalize() for item in self.anomalies]

        self.patterns = [item.finalize() for item in self.patterns]

        self.unclassified_group_ids = sorted(set(self.unclassified_group_ids))

        return self

    def to_dict(self) -> dict:
        self.finalize()

        return {
            "schema_version": (self.schema_version),
            "generated_at": (self.generated_at),
            "incidents": [item.to_dict() for item in self.incidents],
            "anomalies": [item.to_dict() for item in self.anomalies],
            "patterns": [item.to_dict() for item in self.patterns],
            "explanations": [item.to_dict() for item in self.explanations],
            "unclassified_group_ids": (self.unclassified_group_ids),
            "metadata": json_safe(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
            default=str,
        )


def confidence_level(
    score: float,
) -> str:
    """
    Map a normalized OSHA evidence score to a
    descriptive confidence band.

    These thresholds are design defaults, not
    calibrated probabilities.
    """

    value = checked_score(score)

    if value >= 0.90:
        return ConfidenceLevel.VERY_HIGH.value

    if value >= 0.75:
        return ConfidenceLevel.HIGH.value

    if value >= 0.50:
        return ConfidenceLevel.MEDIUM.value

    if value > 0.0:
        return ConfidenceLevel.LOW.value

    return ConfidenceLevel.NONE.value


def severity_rank(
    severity: str,
) -> int:
    ranks = {
        IncidentSeverity.UNKNOWN.value: 0,
        IncidentSeverity.INFO.value: 1,
        IncidentSeverity.LOW.value: 2,
        IncidentSeverity.MEDIUM.value: 3,
        IncidentSeverity.HIGH.value: 4,
        IncidentSeverity.CRITICAL.value: 5,
    }

    return ranks.get(
        severity,
        0,
    )


def highest_severity(
    values: Iterable[str],
) -> str:
    values_list = list(values)

    if not values_list:
        return IncidentSeverity.UNKNOWN.value

    return max(
        values_list,
        key=severity_rank,
    )
