from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from collections import (
    Counter,
    defaultdict,
)
from dataclasses import (
    dataclass,
    field,
)
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
    Pattern,
    checked_score,
)

from src.telemetry.schema import (
    SignalType,
    TelemetryRecord,
)


# ---------------------------------------------------------------
# Pattern configuration
# ---------------------------------------------------------------

DEFAULT_MINIMUM_OCCURRENCES = 2

DEFAULT_MAX_TEMPLATE_LENGTH = 4096


UUID_PATTERN = re.compile(
    r"\b"
    r"[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
    r"\b"
)


HEX_PATTERN = re.compile(r"\b0x[0-9a-fA-F]+\b")


LONG_HEX_PATTERN = re.compile(r"\b[0-9a-fA-F]{16,}\b")


INTEGER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"-?\d+"
    r"(?![A-Za-z0-9_.-])"
)


FLOAT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"-?\d+\.\d+"
    r"(?![A-Za-z0-9_.-])"
)


ISO_TIMESTAMP_PATTERN = re.compile(
    r"\b"
    r"\d{4}-\d{2}-\d{2}"
    r"[T ]"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:\d{2})?"
    r"\b"
)


MAC_PATTERN = re.compile(
    r"\b"
    r"(?:[0-9A-Fa-f]{2}:){5}"
    r"[0-9A-Fa-f]{2}"
    r"\b"
)


PATH_NUMBER_PATTERN = re.compile(r"(?<=/)\d+(?=/|$)")


WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class PatternConfig:
    """
    Configuration for deterministic OSHA pattern
    extraction.

    This is deliberately independent of any ML
    model so pattern discovery remains usable
    before model integration.
    """

    minimum_occurrences: int = DEFAULT_MINIMUM_OCCURRENCES

    maximum_template_length: int = DEFAULT_MAX_TEMPLATE_LENGTH

    include_singletons: bool = False

    include_metric_values: bool = False

    normalize_case: bool = True

    mask_ip_addresses: bool = True

    mask_uuids: bool = True

    mask_numbers: bool = True

    mask_hex: bool = True

    mask_mac_addresses: bool = True

    def validate(
        self,
    ) -> None:
        if self.minimum_occurrences < 1:
            raise ValueError("minimum_occurrences must be >= 1")

        if self.maximum_template_length < 1:
            raise ValueError("maximum_template_length must be >= 1")


@dataclass
class PatternObservation:
    """
    Internal representation of one telemetry
    observation before observations are grouped
    into Pattern objects.
    """

    fingerprint: str

    pattern_type: str

    name: str

    template: str

    record_id: str

    resource_id: str

    environment_id: str

    provider: str

    signal_type: str

    timestamp: str

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)


@dataclass
class PatternStatistics:
    observations: int = 0

    patterns: int = 0

    repeated_patterns: int = 0

    singleton_patterns: int = 0

    environments: int = 0

    providers: int = 0

    signal_types: int = 0

    def to_dict(self) -> dict:
        return {
            "observations": (self.observations),
            "patterns": (self.patterns),
            "repeated_patterns": (self.repeated_patterns),
            "singleton_patterns": (self.singleton_patterns),
            "environments": (self.environments),
            "providers": (self.providers),
            "signal_types": (self.signal_types),
        }


# ---------------------------------------------------------------
# Basic normalization
# ---------------------------------------------------------------


def optional_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


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


def stable_fingerprint(
    value: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def normalize_ip_tokens(
    text: str,
) -> str:
    """
    Replace valid IPv4/IPv6 tokens with <IP>.

    This operates token-by-token to avoid treating
    arbitrary dotted text as an address.
    """

    tokens = text.split()

    output = []

    for token in tokens:
        prefix = ""

        suffix = ""

        candidate = token

        while candidate and candidate[0] in "([{<\"'":
            prefix += candidate[0]

            candidate = candidate[1:]

        while candidate and candidate[-1] in ")]}>\"',;":
            suffix = candidate[-1] + suffix

            candidate = candidate[:-1]

        # Handle common host:port forms.
        address_candidate = candidate

        if candidate.count(":") == 1 and "." in candidate:
            possible_ip, possible_port = candidate.rsplit(
                ":",
                1,
            )

            if possible_port.isdigit():
                address_candidate = possible_ip

        try:
            ipaddress.ip_address(address_candidate)

            if address_candidate != candidate:
                port = candidate[len(address_candidate) :]

                candidate = "<IP>" + port

            else:
                candidate = "<IP>"

        except ValueError:
            pass

        output.append(prefix + candidate + suffix)

    return " ".join(output)


def normalize_log_template(
    message: str,
    *,
    config: PatternConfig,
) -> str:
    """
    Produce a deterministic log template.

    It intentionally masks common dynamic values,
    but it is not intended to replace a full log
    parsing/mining algorithm such as Drain.
    """

    text = message.strip()

    if config.normalize_case:
        text = text.lower()

    if config.mask_ip_addresses:
        text = normalize_ip_tokens(text)

    if config.mask_uuids:
        text = UUID_PATTERN.sub(
            "<UUID>",
            text,
        )

    if config.mask_mac_addresses:
        text = MAC_PATTERN.sub(
            "<MAC>",
            text,
        )

    if config.mask_hex:
        text = HEX_PATTERN.sub(
            "<HEX>",
            text,
        )

        text = LONG_HEX_PATTERN.sub(
            "<HEX>",
            text,
        )

    text = ISO_TIMESTAMP_PATTERN.sub(
        "<TIMESTAMP>",
        text,
    )

    if config.mask_numbers:
        # Floats must be replaced before integers.
        text = FLOAT_PATTERN.sub(
            "<FLOAT>",
            text,
        )

        text = PATH_NUMBER_PATTERN.sub(
            "<NUM>",
            text,
        )

        text = INTEGER_PATTERN.sub(
            "<NUM>",
            text,
        )

    text = WHITESPACE_PATTERN.sub(
        " ",
        text,
    ).strip()

    if len(text) > config.maximum_template_length:
        text = text[: config.maximum_template_length] + "[TRUNCATED]"

    return text


def normalized_signal_name(
    telemetry: TelemetryRecord,
) -> str:
    name = optional_string(telemetry.signal.signal_name)

    if name:
        return name.lower()

    return telemetry.signal.signal_type


def selected_state(
    telemetry: TelemetryRecord,
) -> Any:
    signal = telemetry.signal

    if signal.value is not None:
        return signal.value

    attributes = signal.attributes or {}

    for key in (
        "state",
        "status",
        "active_state",
        "sub_state",
        "phase",
        "healthy",
        "available",
        "enabled",
    ):
        if key in attributes:
            return attributes.get(key)

    return None


# ---------------------------------------------------------------
# Observation builders
# ---------------------------------------------------------------


def log_observation(
    telemetry: TelemetryRecord,
    *,
    config: PatternConfig,
) -> Optional[PatternObservation]:
    message = optional_string(telemetry.signal.message)

    if message is None:
        return None

    template = normalize_log_template(
        message,
        config=config,
    )

    if not template:
        return None

    fingerprint = stable_fingerprint(
        {
            "pattern_type": "log",
            "provider": (telemetry.resource.provider),
            "service": (telemetry.resource.service),
            "resource_type": (telemetry.resource.resource_type),
            "template": template,
        }
    )

    telemetry.finalize()

    if not telemetry.record_id:
        return None

    return PatternObservation(
        fingerprint=fingerprint,
        pattern_type="log",
        name=template[:160],
        template=template,
        record_id=(telemetry.record_id),
        resource_id=(telemetry.resource.effective_id()),
        environment_id=(telemetry.resource.environment_id),
        provider=(telemetry.resource.provider),
        signal_type=(telemetry.signal.signal_type),
        timestamp=(telemetry.signal.timestamp),
        attributes={
            "severity": (telemetry.signal.severity),
            "service": (telemetry.resource.service),
        },
    )


def metric_observation(
    telemetry: TelemetryRecord,
    *,
    config: PatternConfig,
) -> Optional[PatternObservation]:
    signal_name = optional_string(telemetry.signal.signal_name)

    if signal_name is None:
        return None

    body: Dict[str, Any] = {
        "pattern_type": "metric",
        "provider": (telemetry.resource.provider),
        "resource_type": (telemetry.resource.resource_type),
        "metric_name": (signal_name.lower()),
        "unit": (telemetry.signal.unit),
    }

    if config.include_metric_values:
        body["value"] = telemetry.signal.value

    fingerprint = stable_fingerprint(body)

    telemetry.finalize()

    if not telemetry.record_id:
        return None

    template = f"{signal_name}" + (
        f" [{telemetry.signal.unit}]" if telemetry.signal.unit else ""
    )

    return PatternObservation(
        fingerprint=fingerprint,
        pattern_type="metric",
        name=signal_name,
        template=template,
        record_id=(telemetry.record_id),
        resource_id=(telemetry.resource.effective_id()),
        environment_id=(telemetry.resource.environment_id),
        provider=(telemetry.resource.provider),
        signal_type=(telemetry.signal.signal_type),
        timestamp=(telemetry.signal.timestamp),
        attributes={
            "unit": (telemetry.signal.unit),
        },
    )


def state_observation(
    telemetry: TelemetryRecord,
) -> Optional[PatternObservation]:
    state = selected_state(telemetry)

    signal_name = normalized_signal_name(telemetry)

    if state is None:
        state_text = "<UNKNOWN>"
    else:
        state_text = str(state).strip().lower()

    fingerprint = stable_fingerprint(
        {
            "pattern_type": ("state"),
            "provider": (telemetry.resource.provider),
            "resource_type": (telemetry.resource.resource_type),
            "signal_name": (signal_name),
            "state": state_text,
        }
    )

    telemetry.finalize()

    if not telemetry.record_id:
        return None

    template = f"{signal_name}:{state_text}"

    return PatternObservation(
        fingerprint=fingerprint,
        pattern_type="state",
        name=template,
        template=template,
        record_id=(telemetry.record_id),
        resource_id=(telemetry.resource.effective_id()),
        environment_id=(telemetry.resource.environment_id),
        provider=(telemetry.resource.provider),
        signal_type=(telemetry.signal.signal_type),
        timestamp=(telemetry.signal.timestamp),
        attributes={
            "state": state,
            "severity": (telemetry.signal.severity),
        },
    )


def named_signal_observation(
    telemetry: TelemetryRecord,
) -> Optional[PatternObservation]:
    signal_name = normalized_signal_name(telemetry)

    if not signal_name:
        return None

    fingerprint = stable_fingerprint(
        {
            "pattern_type": (telemetry.signal.signal_type),
            "provider": (telemetry.resource.provider),
            "resource_type": (telemetry.resource.resource_type),
            "signal_name": (signal_name),
        }
    )

    telemetry.finalize()

    if not telemetry.record_id:
        return None

    return PatternObservation(
        fingerprint=fingerprint,
        pattern_type=(telemetry.signal.signal_type),
        name=signal_name,
        template=signal_name,
        record_id=(telemetry.record_id),
        resource_id=(telemetry.resource.effective_id()),
        environment_id=(telemetry.resource.environment_id),
        provider=(telemetry.resource.provider),
        signal_type=(telemetry.signal.signal_type),
        timestamp=(telemetry.signal.timestamp),
        attributes={
            "severity": (telemetry.signal.severity),
        },
    )


def observation_from_record(
    telemetry: TelemetryRecord,
    *,
    config: Optional[PatternConfig] = None,
) -> Optional[PatternObservation]:
    configuration = config or PatternConfig()

    signal_type = telemetry.signal.signal_type

    if signal_type == SignalType.LOG.value:
        return log_observation(
            telemetry,
            config=configuration,
        )

    if signal_type == SignalType.METRIC.value:
        return metric_observation(
            telemetry,
            config=configuration,
        )

    if signal_type in {
        SignalType.RESOURCE_STATE.value,
        SignalType.STATE_CHANGE.value,
        SignalType.HEALTH.value,
    }:
        return state_observation(telemetry)

    if signal_type in {
        SignalType.EVENT.value,
        SignalType.ALERT.value,
        SignalType.SECURITY.value,
        SignalType.CONFIGURATION.value,
    }:
        return named_signal_observation(telemetry)

    return named_signal_observation(telemetry)


# ---------------------------------------------------------------
# Pattern construction
# ---------------------------------------------------------------


def observation_time_range(
    observations: Sequence[PatternObservation],
) -> Tuple[
    Optional[str],
    Optional[str],
]:
    ordered = []

    for observation in observations:
        timestamp = parse_timestamp(observation.timestamp)

        if timestamp is not None:
            ordered.append(
                (
                    timestamp,
                    observation.timestamp,
                )
            )

    if not ordered:
        return (
            None,
            None,
        )

    ordered.sort(key=lambda item: item[0])

    return (
        ordered[0][1],
        ordered[-1][1],
    )


def novelty_from_count(
    count: int,
) -> float:
    """
    Initial statistical novelty heuristic.

    First observation:
        high novelty

    Repeated observations:
        decreasing novelty

    This is not ML novelty probability.
    """

    if count <= 0:
        return 1.0

    return checked_score(1.0 / float(count))


def pattern_from_observations(
    observations: Sequence[PatternObservation],
) -> Pattern:
    if not observations:
        raise ValueError("Cannot create Pattern from an empty observation set")

    first = observations[0]

    first_seen, last_seen = observation_time_range(observations)

    providers = sorted({item.provider for item in observations})

    signal_types = sorted({item.signal_type for item in observations})

    record_ids = [item.record_id for item in observations]

    resource_ids = [item.resource_id for item in observations]

    severity_counts = Counter(
        str(
            item.attributes.get(
                "severity",
                "unknown",
            )
        )
        for item in observations
    )

    pattern = Pattern(
        pattern_type=(first.pattern_type),
        name=first.name,
        fingerprint=(first.fingerprint),
        environment_id=(first.environment_id),
        occurrence_count=(len(observations)),
        record_ids=record_ids,
        resource_ids=resource_ids,
        providers=providers,
        signal_types=(signal_types),
        first_seen=first_seen,
        last_seen=last_seen,
        novelty_score=(novelty_from_count(len(observations))),
        attributes={
            "template": (first.template),
            "severity_counts": (dict(severity_counts)),
        },
    )

    return pattern.finalize()


def extract_patterns(
    records: Iterable[TelemetryRecord],
    *,
    config: Optional[PatternConfig] = None,
) -> List[Pattern]:
    """
    Extract deterministic patterns from
    canonical telemetry.

    Patterns are grouped inside each environment.
    """

    configuration = config or PatternConfig()

    configuration.validate()

    buckets: Dict[
        Tuple[str, str],
        List[PatternObservation],
    ] = defaultdict(list)

    for telemetry in records:
        observation = observation_from_record(
            telemetry,
            config=configuration,
        )

        if observation is None:
            continue

        bucket_key = (
            observation.environment_id,
            observation.fingerprint,
        )

        buckets[bucket_key].append(observation)

    output = []

    for observations in buckets.values():
        count = len(observations)

        if (
            not configuration.include_singletons
            and count < configuration.minimum_occurrences
        ):
            continue

        output.append(pattern_from_observations(observations))

    output.sort(
        key=lambda pattern: (
            -pattern.occurrence_count,
            pattern.pattern_type,
            pattern.fingerprint,
        )
    )

    return output


# ---------------------------------------------------------------
# Correlation-group pattern extraction
# ---------------------------------------------------------------


def records_for_group(
    group: CorrelationGroup,
    registry: ResourceRegistry,
) -> List[TelemetryRecord]:
    output = []

    for record_identifier in group.record_ids:
        telemetry = registry.get_record(record_identifier)

        if telemetry is not None:
            output.append(telemetry)

    return output


def patterns_for_group(
    group: CorrelationGroup,
    registry: ResourceRegistry,
    *,
    config: Optional[PatternConfig] = None,
) -> List[Pattern]:
    """
    Discover patterns specifically within one
    correlation group.
    """

    records = records_for_group(
        group,
        registry,
    )

    configuration = config or PatternConfig(
        include_singletons=True,
    )

    return extract_patterns(
        records,
        config=configuration,
    )


# ---------------------------------------------------------------
# Cross-group sequence patterns
# ---------------------------------------------------------------


def group_signature(
    group: CorrelationGroup,
) -> str:
    """
    Stable structural signature for a correlated
    episode.

    Dynamic record/resource IDs are deliberately
    excluded.
    """

    body = {
        "providers": sorted(group.providers),
        "signal_types": sorted(group.signal_types),
        "record_count": len(group.record_ids),
        "resource_count": len(group.resource_ids),
        "decision_counts": (
            group.attributes.get(
                "decision_counts",
                {},
            )
            if isinstance(
                group.attributes,
                Mapping,
            )
            else {}
        ),
        "severity_counts": (
            group.attributes.get(
                "severity_counts",
                {},
            )
            if isinstance(
                group.attributes,
                Mapping,
            )
            else {}
        ),
        "resource_type_counts": (
            group.attributes.get(
                "resource_type_counts",
                {},
            )
            if isinstance(
                group.attributes,
                Mapping,
            )
            else {}
        ),
    }

    return stable_fingerprint(body)


def extract_group_patterns(
    groups: Iterable[CorrelationGroup],
) -> List[Pattern]:
    """
    Find repeated correlated-episode structures.

    This does not determine whether the repeated
    episode is healthy or anomalous.
    """

    buckets: Dict[
        Tuple[str, str],
        List[CorrelationGroup],
    ] = defaultdict(list)

    for group in groups:
        signature = group_signature(group)

        buckets[
            (
                group.environment_id,
                signature,
            )
        ].append(group)

    patterns = []

    for (
        (
            environment_id,
            signature,
        ),
        matching_groups,
    ) in buckets.items():
        group_ids = [group.group_id for group in matching_groups if group.group_id]

        resource_ids = []

        providers = []

        signal_types = []

        start_values = []

        end_values = []

        for group in matching_groups:
            resource_ids.extend(group.resource_ids)

            providers.extend(group.providers)

            signal_types.extend(group.signal_types)

            start_values.append(group.start_timestamp)

            end_values.append(group.end_timestamp)

        starts = [
            (
                parse_timestamp(value),
                value,
            )
            for value in start_values
            if parse_timestamp(value) is not None
        ]

        ends = [
            (
                parse_timestamp(value),
                value,
            )
            for value in end_values
            if parse_timestamp(value) is not None
        ]

        starts.sort(key=lambda item: item[0])

        ends.sort(key=lambda item: item[0])

        pattern = Pattern(
            pattern_type=("correlation_group"),
            name=("Repeated correlation group structure"),
            fingerprint=signature,
            environment_id=(environment_id),
            occurrence_count=(len(matching_groups)),
            record_ids=[],
            resource_ids=(resource_ids),
            providers=providers,
            signal_types=(signal_types),
            first_seen=(starts[0][1] if starts else None),
            last_seen=(ends[-1][1] if ends else None),
            novelty_score=(novelty_from_count(len(matching_groups))),
            attributes={
                "correlation_group_ids": (group_ids),
            },
        )

        patterns.append(pattern.finalize())

    patterns.sort(
        key=lambda pattern: (
            -pattern.occurrence_count,
            pattern.fingerprint,
        )
    )

    return patterns


# ---------------------------------------------------------------
# Pattern lookup and statistics
# ---------------------------------------------------------------


def patterns_by_resource(
    patterns: Iterable[Pattern],
) -> Dict[
    str,
    List[Pattern],
]:
    output: Dict[
        str,
        List[Pattern],
    ] = defaultdict(list)

    for pattern in patterns:
        for resource_id in pattern.resource_ids:
            output[resource_id].append(pattern)

    return dict(output)


def patterns_by_type(
    patterns: Iterable[Pattern],
) -> Dict[
    str,
    List[Pattern],
]:
    output: Dict[
        str,
        List[Pattern],
    ] = defaultdict(list)

    for pattern in patterns:
        output[pattern.pattern_type].append(pattern)

    return dict(output)


def pattern_statistics(
    records: Iterable[TelemetryRecord],
    patterns: Iterable[Pattern],
) -> PatternStatistics:
    records_list = list(records)

    patterns_list = list(patterns)

    providers = {telemetry.resource.provider for telemetry in records_list}

    environments = {telemetry.resource.environment_id for telemetry in records_list}

    signal_types = {telemetry.signal.signal_type for telemetry in records_list}

    repeated = sum(1 for pattern in patterns_list if pattern.occurrence_count > 1)

    singletons = sum(1 for pattern in patterns_list if pattern.occurrence_count == 1)

    return PatternStatistics(
        observations=len(records_list),
        patterns=len(patterns_list),
        repeated_patterns=(repeated),
        singleton_patterns=(singletons),
        environments=len(environments),
        providers=len(providers),
        signal_types=len(signal_types),
    )


class PatternEngine:
    """
    Stateful interface for intelligence pipeline
    pattern extraction.
    """

    def __init__(
        self,
        *,
        config: Optional[PatternConfig] = None,
    ) -> None:
        self.config = config or PatternConfig()

        self.config.validate()

        self._patterns: Dict[
            str,
            Pattern,
        ] = {}

    def analyze(
        self,
        records: Iterable[TelemetryRecord],
    ) -> List[Pattern]:
        discovered = extract_patterns(
            records,
            config=self.config,
        )

        for pattern in discovered:
            pattern.finalize()

            if pattern.pattern_id:
                self._patterns[pattern.pattern_id] = pattern

        return discovered

    def analyze_groups(
        self,
        groups: Iterable[CorrelationGroup],
    ) -> List[Pattern]:
        discovered = extract_group_patterns(groups)

        for pattern in discovered:
            pattern.finalize()

            if pattern.pattern_id:
                self._patterns[pattern.pattern_id] = pattern

        return discovered

    def patterns(
        self,
    ) -> List[Pattern]:
        return sorted(
            self._patterns.values(),
            key=lambda pattern: (
                -pattern.occurrence_count,
                pattern.pattern_type,
            ),
        )

    def get(
        self,
        pattern_id: str,
    ) -> Optional[Pattern]:
        return self._patterns.get(pattern_id)

    def clear(
        self,
    ) -> None:
        self._patterns.clear()
