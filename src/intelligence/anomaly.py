from __future__ import annotations

import math
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
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from src.correlation.registry import (
    ResourceRegistry,
)

from src.correlation.schema import (
    CorrelationGroup,
)

from src.intelligence.schema import (
    Anomaly,
    AnomalyType,
    EvidenceReference,
    IncidentSeverity,
    Pattern,
    checked_score,
)

from src.telemetry.schema import (
    Severity,
    SignalType,
    TelemetryRecord,
)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

DEFAULT_ANOMALY_THRESHOLD = 0.60

DEFAULT_EVENT_BURST_COUNT = 5

DEFAULT_EVENT_BURST_SCORE = 0.70

DEFAULT_NOVELTY_WEIGHT = 0.35

DEFAULT_SEVERITY_WEIGHT = 0.30

DEFAULT_STATE_WEIGHT = 0.20

DEFAULT_CORRELATION_WEIGHT = 0.15


SEVERITY_SCORES = {
    Severity.TRACE.value: 0.00,
    Severity.DEBUG.value: 0.05,
    Severity.INFO.value: 0.10,
    Severity.NOTICE.value: 0.20,
    Severity.WARNING.value: 0.55,
    Severity.ERROR.value: 0.75,
    Severity.CRITICAL.value: 0.90,
    Severity.ALERT.value: 0.95,
    Severity.EMERGENCY.value: 1.00,
    Severity.UNKNOWN.value: 0.10,
}


UNHEALTHY_STATES = {
    "failed",
    "failure",
    "error",
    "critical",
    "down",
    "dead",
    "inactive",
    "unhealthy",
    "degraded",
    "crashed",
    "crash",
    "terminated",
    "oom",
    "oomkilled",
    "notready",
    "not_ready",
}


WARNING_STATES = {
    "warning",
    "warn",
    "pending",
    "unknown",
    "recovering",
    "restarting",
    "activating",
    "deactivating",
}


ModelScoreProvider = Callable[
    [TelemetryRecord],
    Optional[float],
]


@dataclass(frozen=True)
class AnomalyConfig:
    """
    Deterministic OSHA anomaly configuration.

    The values are heuristic defaults and must
    not be presented as calibrated probabilities.
    """

    anomaly_threshold: float = DEFAULT_ANOMALY_THRESHOLD

    event_burst_count: int = DEFAULT_EVENT_BURST_COUNT

    event_burst_score: float = DEFAULT_EVENT_BURST_SCORE

    novelty_weight: float = DEFAULT_NOVELTY_WEIGHT

    severity_weight: float = DEFAULT_SEVERITY_WEIGHT

    state_weight: float = DEFAULT_STATE_WEIGHT

    correlation_weight: float = DEFAULT_CORRELATION_WEIGHT

    include_info_anomalies: bool = False

    def validate(
        self,
    ) -> None:
        if not (0.0 <= self.anomaly_threshold <= 1.0):
            raise ValueError("anomaly_threshold must be between 0 and 1")

        if self.event_burst_count < 2:
            raise ValueError("event_burst_count must be >= 2")

        weights = (
            self.novelty_weight,
            self.severity_weight,
            self.state_weight,
            self.correlation_weight,
        )

        if any(value < 0 for value in weights):
            raise ValueError("anomaly weights must be >= 0")


# ---------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------


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

    return (
        text.lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "-",
            "_",
        )
    )


def parse_timestamp(
    value: Any,
) -> Optional[datetime]:
    text = optional_string(value)

    if text is None:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)

    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def severity_score(
    severity: str,
) -> float:
    return checked_score(
        SEVERITY_SCORES.get(
            severity,
            SEVERITY_SCORES[Severity.UNKNOWN.value],
        )
    )


def incident_severity_for_score(
    score: float,
) -> str:
    """
    Translate anomaly evidence strength into a
    human-readable severity band.

    This does not prove operational impact.
    """

    value = checked_score(score)

    if value >= 0.90:
        return IncidentSeverity.CRITICAL.value

    if value >= 0.75:
        return IncidentSeverity.HIGH.value

    if value >= 0.55:
        return IncidentSeverity.MEDIUM.value

    if value >= 0.30:
        return IncidentSeverity.LOW.value

    return IncidentSeverity.INFO.value


def selected_state(
    telemetry: TelemetryRecord,
) -> Optional[str]:
    """
    Extract a normalized state/status from the
    common signal.
    """

    if isinstance(
        telemetry.signal.value,
        str,
    ):
        state = normalized_text(telemetry.signal.value)

        if state:
            return state

    attributes = telemetry.signal.attributes or {}

    for key in (
        "state",
        "status",
        "active_state",
        "sub_state",
        "phase",
        "health",
    ):
        if key not in attributes:
            continue

        state = normalized_text(attributes.get(key))

        if state:
            return state

    healthy = attributes.get("healthy")

    if healthy is False:
        return "unhealthy"

    if healthy is True:
        return "healthy"

    available = attributes.get("available")

    if available is False:
        return "unavailable"

    return None


def state_anomaly_score(
    telemetry: TelemetryRecord,
) -> float:
    state = selected_state(telemetry)

    if state is None:
        return 0.0

    compact = state.replace(
        "_",
        "",
    )

    if state in UNHEALTHY_STATES or compact in UNHEALTHY_STATES:
        return 1.0

    if state in WARNING_STATES or compact in WARNING_STATES:
        return 0.55

    return 0.0


def weighted_score(
    values: Mapping[
        str,
        float,
    ],
    weights: Mapping[
        str,
        float,
    ],
) -> float:
    """
    Transparent weighted evidence score.

    Missing values contribute zero.
    """

    applicable_weights = {
        key: max(
            0.0,
            float(weight),
        )
        for key, weight in weights.items()
    }

    total_weight = sum(applicable_weights.values())

    if total_weight <= 0:
        return 0.0

    score = sum(
        checked_score(
            values.get(
                key,
                0.0,
            )
        )
        * weight
        for key, weight in applicable_weights.items()
    )

    return checked_score(score / total_weight)


# ---------------------------------------------------------------
# Pattern matching / novelty
# ---------------------------------------------------------------


def record_pattern_index(
    patterns: Iterable[Pattern],
) -> Dict[
    str,
    List[Pattern],
]:
    index: Dict[
        str,
        List[Pattern],
    ] = defaultdict(list)

    for pattern in patterns:
        pattern.finalize()

        for record_id in pattern.record_ids:
            index[record_id].append(pattern)

    return dict(index)


def pattern_novelty_for_record(
    telemetry: TelemetryRecord,
    pattern_index: Mapping[
        str,
        List[Pattern],
    ],
) -> Tuple[
    float,
    List[Pattern],
]:
    telemetry.finalize()

    if not telemetry.record_id:
        return (
            0.0,
            [],
        )

    matches = list(
        pattern_index.get(
            telemetry.record_id,
            [],
        )
    )

    if not matches:
        # Absence of a pattern is not enough to
        # call something anomalous.
        return (
            0.0,
            [],
        )

    novelty = max(
        (pattern.novelty_score for pattern in matches),
        default=0.0,
    )

    return (
        checked_score(novelty),
        matches,
    )


# ---------------------------------------------------------------
# Correlation support
# ---------------------------------------------------------------


def record_group_index(
    groups: Iterable[CorrelationGroup],
) -> Dict[
    str,
    List[CorrelationGroup],
]:
    index: Dict[
        str,
        List[CorrelationGroup],
    ] = defaultdict(list)

    for group in groups:
        for record_id in group.record_ids:
            index[record_id].append(group)

    return dict(index)


def correlation_strength_for_record(
    telemetry: TelemetryRecord,
    group_index: Mapping[
        str,
        List[CorrelationGroup],
    ],
) -> Tuple[
    float,
    List[CorrelationGroup],
]:
    telemetry.finalize()

    if not telemetry.record_id:
        return (
            0.0,
            [],
        )

    groups = list(
        group_index.get(
            telemetry.record_id,
            [],
        )
    )

    if not groups:
        return (
            0.0,
            [],
        )

    score = max(
        (group.strongest_score for group in groups),
        default=0.0,
    )

    return (
        checked_score(score),
        groups,
    )


# ---------------------------------------------------------------
# Single-record anomalies
# ---------------------------------------------------------------


def record_anomaly(
    telemetry: TelemetryRecord,
    *,
    pattern_index: Mapping[
        str,
        List[Pattern],
    ],
    group_index: Mapping[
        str,
        List[CorrelationGroup],
    ],
    config: AnomalyConfig,
    model_score_provider: Optional[ModelScoreProvider] = None,
) -> Optional[Anomaly]:
    """
    Assess one telemetry observation.

    This combines deterministic evidence with an
    optional external model anomaly score.

    Model score integration is optional and has
    no assumed implementation here.
    """

    telemetry.finalize()

    if not telemetry.record_id:
        return None

    novelty, patterns = pattern_novelty_for_record(
        telemetry,
        pattern_index,
    )

    correlation_strength, groups = correlation_strength_for_record(
        telemetry,
        group_index,
    )

    severity = severity_score(telemetry.signal.severity)

    state = state_anomaly_score(telemetry)

    model_score = None

    if model_score_provider is not None:
        try:
            model_score = model_score_provider(telemetry)

        except Exception:
            # Model integration failure should
            # not break deterministic analysis.
            model_score = None

    evidence_values = {
        "novelty": novelty,
        "severity": severity,
        "state": state,
        "correlation": (correlation_strength),
    }

    evidence_weights = {
        "novelty": (config.novelty_weight),
        "severity": (config.severity_weight),
        "state": (config.state_weight),
        "correlation": (config.correlation_weight),
    }

    deterministic_score = weighted_score(
        evidence_values,
        evidence_weights,
    )

    if model_score is not None:
        # Keep deterministic and model evidence
        # visible separately.
        #
        # The simple average is an initial OSHA
        # policy, not statistical calibration.
        final_score = checked_score(
            (deterministic_score + checked_score(model_score)) / 2.0
        )

    else:
        final_score = deterministic_score

    if final_score < config.anomaly_threshold:
        return None

    if (
        telemetry.signal.severity
        in {
            Severity.TRACE.value,
            Severity.DEBUG.value,
            Severity.INFO.value,
        }
        and state <= 0
        and novelty <= 0
        and not config.include_info_anomalies
    ):
        return None

    state_value = selected_state(telemetry)

    if state > 0:
        anomaly_type = AnomalyType.STATE_CHANGE.value

        title = "Abnormal resource state"

    elif telemetry.signal.signal_type == SignalType.SECURITY.value:
        anomaly_type = AnomalyType.SECURITY.value

        title = "Security-related anomaly"

    elif telemetry.signal.signal_type == SignalType.METRIC.value:
        anomaly_type = AnomalyType.METRIC_DEVIATION.value

        title = "Metric anomaly candidate"

    elif telemetry.signal.signal_type == SignalType.LOG.value:
        anomaly_type = AnomalyType.LOG_PATTERN.value

        title = "Log pattern anomaly"

    else:
        anomaly_type = AnomalyType.UNKNOWN.value

        title = "Telemetry anomaly"

    evidence = [
        EvidenceReference(
            evidence_type=("severity"),
            description=("Signal severity contributes to anomaly evidence"),
            score=severity,
            record_ids=[telemetry.record_id],
            resource_ids=[telemetry.resource.effective_id()],
            attributes={
                "severity": (telemetry.signal.severity),
            },
        )
    ]

    if novelty > 0:
        evidence.append(
            EvidenceReference(
                evidence_type=("pattern_novelty"),
                description=("Pattern novelty contributes to anomaly evidence"),
                score=novelty,
                record_ids=[telemetry.record_id],
                resource_ids=[telemetry.resource.effective_id()],
                attributes={
                    "pattern_ids": [
                        pattern.pattern_id
                        for pattern in patterns
                        if (pattern.pattern_id)
                    ],
                },
            )
        )

    if state > 0:
        evidence.append(
            EvidenceReference(
                evidence_type=("resource_state"),
                description=("Observed resource state appears unhealthy or degraded"),
                score=state,
                record_ids=[telemetry.record_id],
                resource_ids=[telemetry.resource.effective_id()],
                attributes={
                    "state": (state_value),
                },
            )
        )

    if correlation_strength > 0:
        evidence.append(
            EvidenceReference(
                evidence_type=("correlation"),
                description=("Observation participates in correlated activity"),
                score=(correlation_strength),
                record_ids=[telemetry.record_id],
                resource_ids=[telemetry.resource.effective_id()],
                attributes={
                    "group_ids": [group.group_id for group in groups if group.group_id],
                },
            )
        )

    if model_score is not None:
        evidence.append(
            EvidenceReference(
                evidence_type=("model"),
                description=("External anomaly model contributed a score"),
                score=checked_score(model_score),
                record_ids=[telemetry.record_id],
                resource_ids=[telemetry.resource.effective_id()],
            )
        )

    pattern_ids = [pattern.pattern_id for pattern in patterns if pattern.pattern_id]

    anomaly = Anomaly(
        anomaly_type=anomaly_type,
        title=title,
        environment_id=(telemetry.resource.environment_id),
        score=final_score,
        severity=(incident_severity_for_score(final_score)),
        description=(
            "Telemetry observation exceeded the configured anomaly evidence threshold."
        ),
        record_ids=[telemetry.record_id],
        resource_ids=[telemetry.resource.effective_id()],
        pattern_ids=pattern_ids,
        evidence=evidence,
        attributes={
            "provider": (telemetry.resource.provider),
            "resource_type": (telemetry.resource.resource_type),
            "signal_type": (telemetry.signal.signal_type),
            "signal_name": (telemetry.signal.signal_name),
            "signal_severity": (telemetry.signal.severity),
            "state": state_value,
            "deterministic_score": (deterministic_score),
            "model_score": (
                checked_score(model_score) if model_score is not None else None
            ),
        },
    )

    return anomaly.finalize()


# ---------------------------------------------------------------
# Event-burst anomalies
# ---------------------------------------------------------------


def burst_key(
    telemetry: TelemetryRecord,
) -> Tuple[
    str,
    str,
    str,
]:
    """
    Event bursts are grouped by environment,
    resource and stable signal name/type.
    """

    return (
        telemetry.resource.environment_id,
        telemetry.resource.effective_id(),
        (telemetry.signal.signal_name or telemetry.signal.signal_type),
    )


def event_burst_anomalies(
    records: Iterable[TelemetryRecord],
    *,
    config: AnomalyConfig,
) -> List[Anomaly]:
    """
    Detect simple concentrated repeated event or
    alert activity.

    Correlation windows already bound the wider
    analysis in the intelligence pipeline; this
    function only evaluates repeated occurrences.
    """

    buckets: Dict[
        Tuple[str, str, str],
        List[TelemetryRecord],
    ] = defaultdict(list)

    for telemetry in records:
        if telemetry.signal.signal_type not in {
            SignalType.EVENT.value,
            SignalType.ALERT.value,
        }:
            continue

        buckets[burst_key(telemetry)].append(telemetry)

    output = []

    for (
        (
            environment_id,
            resource_id,
            signal_name,
        ),
        members,
    ) in buckets.items():
        if len(members) < config.event_burst_count:
            continue

        record_ids = []

        timestamps = []

        severities = []

        for telemetry in members:
            telemetry.finalize()

            if telemetry.record_id:
                record_ids.append(telemetry.record_id)

            parsed = parse_timestamp(telemetry.signal.timestamp)

            if parsed is not None:
                timestamps.append(
                    (
                        parsed,
                        telemetry.signal.timestamp,
                    )
                )

            severities.append(severity_score(telemetry.signal.severity))

        timestamps.sort(key=lambda item: item[0])

        severity_component = max(
            severities,
            default=0.0,
        )

        count_component = min(
            1.0,
            len(members) / float(config.event_burst_count * 2),
        )

        score = checked_score(
            (config.event_burst_score * 0.60)
            + (severity_component * 0.20)
            + (count_component * 0.20)
        )

        output.append(
            Anomaly(
                anomaly_type=(AnomalyType.EVENT_BURST.value),
                title=("Repeated event activity"),
                environment_id=(environment_id),
                score=score,
                severity=(incident_severity_for_score(score)),
                description=(
                    "Multiple occurrences of the same event or alert were observed."
                ),
                record_ids=(record_ids),
                resource_ids=[resource_id],
                evidence=[
                    EvidenceReference(
                        evidence_type=("event_frequency"),
                        description=(
                            "Repeated event frequency exceeded the configured threshold"
                        ),
                        score=score,
                        record_ids=(record_ids),
                        resource_ids=[resource_id],
                        attributes={
                            "occurrences": (len(members)),
                            "threshold": (config.event_burst_count),
                            "signal_name": (signal_name),
                        },
                    )
                ],
                attributes={
                    "signal_name": (signal_name),
                    "occurrence_count": (len(members)),
                    "first_seen": (timestamps[0][1] if timestamps else None),
                    "last_seen": (timestamps[-1][1] if timestamps else None),
                },
            ).finalize()
        )

    return output


# ---------------------------------------------------------------
# Correlation-group anomalies
# ---------------------------------------------------------------


def group_anomaly(
    group: CorrelationGroup,
) -> Optional[Anomaly]:
    """
    Convert a strongly correlated episode into an
    anomaly candidate when its signal composition
    contains meaningful warning/error evidence.

    A correlated episode by itself is not
    automatically anomalous.
    """

    severity_counts = {}

    if isinstance(
        group.attributes,
        Mapping,
    ):
        candidate = group.attributes.get("severity_counts")

        if isinstance(
            candidate,
            Mapping,
        ):
            severity_counts = dict(candidate)

    weighted_severity = 0.0

    total = 0

    for severity, raw_count in severity_counts.items():
        try:
            count = int(raw_count)
        except (
            TypeError,
            ValueError,
        ):
            continue

        total += count

        weighted_severity += severity_score(str(severity)) * count

    if total > 0:
        severity_component = weighted_severity / total

    else:
        severity_component = 0.0

    correlation_component = max(
        group.strongest_score,
        group.average_score,
    )

    score = checked_score((correlation_component * 0.60) + (severity_component * 0.40))

    if score < 0.60:
        return None

    error_like = sum(
        int(
            severity_counts.get(
                severity,
                0,
            )
            or 0
        )
        for severity in (
            Severity.WARNING.value,
            Severity.ERROR.value,
            Severity.CRITICAL.value,
            Severity.ALERT.value,
            Severity.EMERGENCY.value,
        )
    )

    if error_like <= 0 and severity_component < 0.50:
        return None

    return Anomaly(
        anomaly_type=(AnomalyType.RESOURCE_HEALTH.value),
        title=("Correlated abnormal activity"),
        environment_id=(group.environment_id),
        score=score,
        severity=(incident_severity_for_score(score)),
        description=(
            "A correlated episode contains "
            "warning/error evidence and strong "
            "correlation support."
        ),
        record_ids=list(group.record_ids),
        resource_ids=list(group.resource_ids),
        evidence=[
            EvidenceReference(
                evidence_type=("correlation_group"),
                description=("Correlation-group strength supports anomaly evidence"),
                score=(correlation_component),
                record_ids=list(group.record_ids),
                resource_ids=list(group.resource_ids),
                attributes={
                    "group_id": (group.group_id),
                    "strongest_score": (group.strongest_score),
                    "average_score": (group.average_score),
                },
            ),
            EvidenceReference(
                evidence_type=("severity_distribution"),
                description=(
                    "The correlated episode contains elevated severity signals"
                ),
                score=(severity_component),
                record_ids=list(group.record_ids),
                resource_ids=list(group.resource_ids),
                attributes={
                    "severity_counts": (severity_counts),
                },
            ),
        ],
        attributes={
            "correlation_group_id": (group.group_id),
        },
    ).finalize()


# ---------------------------------------------------------------
# Main anomaly extraction
# ---------------------------------------------------------------


def deduplicate_anomalies(
    anomalies: Iterable[Anomaly],
) -> List[Anomaly]:
    """
    Anomaly IDs are deterministic after finalize,
    allowing duplicate findings from multiple
    passes to collapse safely.
    """

    output: Dict[
        str,
        Anomaly,
    ] = {}

    for anomaly in anomalies:
        anomaly.finalize()

        if not anomaly.anomaly_id:
            continue

        existing = output.get(anomaly.anomaly_id)

        if existing is None or anomaly.score > existing.score:
            output[anomaly.anomaly_id] = anomaly

    return sorted(
        output.values(),
        key=lambda item: (
            -item.score,
            item.anomaly_type,
            item.anomaly_id or "",
        ),
    )


def detect_anomalies(
    records: Iterable[TelemetryRecord],
    *,
    patterns: Iterable[Pattern] = (),
    groups: Iterable[CorrelationGroup] = (),
    config: Optional[AnomalyConfig] = None,
    model_score_provider: Optional[ModelScoreProvider] = None,
) -> List[Anomaly]:
    """
    Run deterministic anomaly detection over
    canonical telemetry.

    External ML scores may optionally contribute,
    but no model is required.
    """

    configuration = config or AnomalyConfig()

    configuration.validate()

    records_list = list(records)

    patterns_list = list(patterns)

    groups_list = list(groups)

    pattern_index = record_pattern_index(patterns_list)

    group_index = record_group_index(groups_list)

    anomalies: List[Anomaly] = []

    for telemetry in records_list:
        anomaly = record_anomaly(
            telemetry,
            pattern_index=(pattern_index),
            group_index=(group_index),
            config=configuration,
            model_score_provider=(model_score_provider),
        )

        if anomaly is not None:
            anomalies.append(anomaly)

    anomalies.extend(
        event_burst_anomalies(
            records_list,
            config=configuration,
        )
    )

    for group in groups_list:
        anomaly = group_anomaly(group)

        if anomaly is not None:
            anomalies.append(anomaly)

    return deduplicate_anomalies(anomalies)


# ---------------------------------------------------------------
# Lookup / summaries
# ---------------------------------------------------------------


def anomalies_by_resource(
    anomalies: Iterable[Anomaly],
) -> Dict[
    str,
    List[Anomaly],
]:
    output: Dict[
        str,
        List[Anomaly],
    ] = defaultdict(list)

    for anomaly in anomalies:
        for resource_id in anomaly.resource_ids:
            output[resource_id].append(anomaly)

    return dict(output)


def anomalies_by_type(
    anomalies: Iterable[Anomaly],
) -> Dict[
    str,
    List[Anomaly],
]:
    output: Dict[
        str,
        List[Anomaly],
    ] = defaultdict(list)

    for anomaly in anomalies:
        output[anomaly.anomaly_type].append(anomaly)

    return dict(output)


def anomaly_summary(
    anomalies: Iterable[Anomaly],
) -> Dict[str, Any]:
    anomalies_list = list(anomalies)

    type_counts = Counter(anomaly.anomaly_type for anomaly in anomalies_list)

    severity_counts = Counter(anomaly.severity for anomaly in anomalies_list)

    resources = {
        resource_id
        for anomaly in anomalies_list
        for resource_id in anomaly.resource_ids
    }

    environments = {anomaly.environment_id for anomaly in anomalies_list}

    return {
        "anomaly_count": (len(anomalies_list)),
        "types": dict(type_counts),
        "severities": dict(severity_counts),
        "affected_resource_count": (len(resources)),
        "environment_count": (len(environments)),
    }


class AnomalyEngine:
    """
    Stateful anomaly detector for the future
    intelligence pipeline.
    """

    def __init__(
        self,
        *,
        config: Optional[AnomalyConfig] = None,
        model_score_provider: Optional[ModelScoreProvider] = None,
    ) -> None:
        self.config = config or AnomalyConfig()

        self.config.validate()

        self.model_score_provider = model_score_provider

        self._anomalies: Dict[
            str,
            Anomaly,
        ] = {}

    def analyze(
        self,
        records: Iterable[TelemetryRecord],
        *,
        patterns: Iterable[Pattern] = (),
        groups: Iterable[CorrelationGroup] = (),
    ) -> List[Anomaly]:
        discovered = detect_anomalies(
            records,
            patterns=patterns,
            groups=groups,
            config=self.config,
            model_score_provider=(self.model_score_provider),
        )

        for anomaly in discovered:
            anomaly.finalize()

            if anomaly.anomaly_id:
                self._anomalies[anomaly.anomaly_id] = anomaly

        return discovered

    def anomalies(
        self,
    ) -> List[Anomaly]:
        return sorted(
            self._anomalies.values(),
            key=lambda anomaly: (
                -anomaly.score,
                anomaly.anomaly_type,
            ),
        )

    def get(
        self,
        anomaly_id: str,
    ) -> Optional[Anomaly]:
        return self._anomalies.get(anomaly_id)

    def clear(
        self,
    ) -> None:
        self._anomalies.clear()
