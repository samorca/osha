from __future__ import annotations

import math
from dataclasses import (
    dataclass,
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
    Optional,
)

from src.correlation.schema import (
    CorrelationDirection,
    CorrelationEvidence,
    EvidenceType,
    checked_score,
)

from src.telemetry.schema import (
    TelemetryRecord,
)


DEFAULT_MAX_DISTANCE_SECONDS = 300.0

DEFAULT_HALF_LIFE_SECONDS = 30.0

DEFAULT_SIMULTANEOUS_TOLERANCE = 0.001


class TemporalOrder(
    str,
    Enum,
):
    """
    Ordering relationship between two signals.

    This describes time only, not causation.
    """

    BEFORE = "before"

    AFTER = "after"

    SIMULTANEOUS = (
        "simultaneous"
    )

    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TemporalRelation:
    """
    Exact temporal relationship between two
    telemetry observations.

    signed_delta_seconds:

        target - source

    Positive:
        target occurred after source.

    Negative:
        target occurred before source.
    """

    source_record_id: Optional[
        str
    ]

    target_record_id: Optional[
        str
    ]

    source_timestamp: Optional[
        str
    ]

    target_timestamp: Optional[
        str
    ]

    delta_seconds: Optional[
        float
    ]

    signed_delta_seconds: Optional[
        float
    ]

    order: str

    direction: str

    within_range: bool

    score: float

    def to_dict(self) -> dict:
        return {
            "source_record_id": (
                self.source_record_id
            ),
            "target_record_id": (
                self.target_record_id
            ),
            "source_timestamp": (
                self.source_timestamp
            ),
            "target_timestamp": (
                self.target_timestamp
            ),
            "delta_seconds": (
                self.delta_seconds
            ),
            "signed_delta_seconds": (
                self.
                signed_delta_seconds
            ),
            "order": self.order,
            "direction": (
                self.direction
            ),
            "within_range": (
                self.within_range
            ),
            "score": (
                checked_score(
                    self.score
                )
            ),
        }


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
    """
    Parse ISO-8601 timestamp into UTC.

    Invalid or missing timestamps return None.
    """

    text = optional_string(
        value
    )

    if text is None:
        return None

    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        timestamp = (
            datetime.fromisoformat(
                text
            )
        )

    except ValueError:
        return None

    if timestamp.tzinfo is None:
        timestamp = (
            timestamp.replace(
                tzinfo=timezone.utc
            )
        )

    return timestamp.astimezone(
        timezone.utc
    )


def telemetry_timestamp(
    telemetry: TelemetryRecord,
) -> Optional[datetime]:
    return parse_timestamp(
        telemetry.signal.timestamp
    )


def normalized_record_id(
    telemetry: TelemetryRecord,
) -> Optional[str]:
    telemetry.finalize()

    return telemetry.record_id


def validate_seconds(
    value: Any,
    *,
    name: str,
    allow_zero: bool = False,
) -> float:
    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            f"{name} must be numeric"
        )

    if not math.isfinite(
        number
    ):
        raise ValueError(
            f"{name} must be finite"
        )

    if allow_zero:
        if number < 0:
            raise ValueError(
                f"{name} must be >= 0"
            )
    else:
        if number <= 0:
            raise ValueError(
                f"{name} must be > 0"
            )

    return number


def temporal_delta_seconds(
    source: TelemetryRecord,
    target: TelemetryRecord,
) -> Optional[float]:
    """
    Return:

        target_time - source_time

    in seconds.
    """

    source_time = (
        telemetry_timestamp(
            source
        )
    )

    target_time = (
        telemetry_timestamp(
            target
        )
    )

    if (
        source_time is None
        or target_time is None
    ):
        return None

    return (
        target_time
        - source_time
    ).total_seconds()


def absolute_delta_seconds(
    source: TelemetryRecord,
    target: TelemetryRecord,
) -> Optional[float]:
    delta = temporal_delta_seconds(
        source,
        target,
    )

    if delta is None:
        return None

    return abs(delta)


def temporal_order(
    source: TelemetryRecord,
    target: TelemetryRecord,
    *,
    simultaneous_tolerance: float = (
        DEFAULT_SIMULTANEOUS_TOLERANCE
    ),
) -> str:
    """
    Determine record ordering.

    A small tolerance handles timestamps that
    differ only because of precision/rounding.
    """

    tolerance = validate_seconds(
        simultaneous_tolerance,
        name=(
            "simultaneous_tolerance"
        ),
        allow_zero=True,
    )

    delta = temporal_delta_seconds(
        source,
        target,
    )

    if delta is None:
        return (
            TemporalOrder.UNKNOWN.value
        )

    if abs(delta) <= tolerance:
        return (
            TemporalOrder.
            SIMULTANEOUS.value
        )

    if delta > 0:
        return (
            TemporalOrder.BEFORE.value
        )

    return TemporalOrder.AFTER.value


def correlation_direction(
    source: TelemetryRecord,
    target: TelemetryRecord,
    *,
    simultaneous_tolerance: float = (
        DEFAULT_SIMULTANEOUS_TOLERANCE
    ),
) -> str:
    """
    Translate time order into candidate
    direction.

    This direction describes ordering only.
    """

    order = temporal_order(
        source,
        target,
        simultaneous_tolerance=(
            simultaneous_tolerance
        ),
    )

    if (
        order
        == TemporalOrder.BEFORE.value
    ):
        return (
            CorrelationDirection.
            FORWARD.value
        )

    if (
        order
        == TemporalOrder.AFTER.value
    ):
        return (
            CorrelationDirection.
            REVERSE.value
        )

    if (
        order
        == TemporalOrder.
        SIMULTANEOUS.value
    ):
        return (
            CorrelationDirection.
            BIDIRECTIONAL.value
        )

    return (
        CorrelationDirection.NONE.value
    )


def exponential_decay_score(
    delta_seconds: float,
    *,
    half_life_seconds: float = (
        DEFAULT_HALF_LIFE_SECONDS
    ),
) -> float:
    """
    Exponential temporal proximity score.

    score = 0.5 ** (
        delta / half_life
    )

    Therefore:

        delta = 0
            score = 1

        delta = half_life
            score = 0.5

    This is an OSHA heuristic correlation score,
    not a probability.
    """

    half_life = validate_seconds(
        half_life_seconds,
        name="half_life_seconds",
    )

    delta = max(
        0.0,
        float(
            delta_seconds
        ),
    )

    score = (
        0.5
        ** (
            delta
            / half_life
        )
    )

    return checked_score(
        score
    )


def linear_decay_score(
    delta_seconds: float,
    *,
    maximum_distance_seconds: float = (
        DEFAULT_MAX_DISTANCE_SECONDS
    ),
) -> float:
    """
    Simple linear proximity score.

    0 seconds:
        1.0

    maximum distance:
        0.0

    Beyond maximum distance:
        0.0
    """

    limit = validate_seconds(
        maximum_distance_seconds,
        name=(
            "maximum_distance_seconds"
        ),
    )

    delta = max(
        0.0,
        float(
            delta_seconds
        ),
    )

    if delta >= limit:
        return 0.0

    return checked_score(
        1.0
        - (
            delta
            / limit
        )
    )


def temporal_score(
    source: TelemetryRecord,
    target: TelemetryRecord,
    *,
    maximum_distance_seconds: float = (
        DEFAULT_MAX_DISTANCE_SECONDS
    ),
    half_life_seconds: float = (
        DEFAULT_HALF_LIFE_SECONDS
    ),
    decay: str = "exponential",
) -> float:
    """
    Return temporal proximity evidence.

    `maximum_distance_seconds` is a hard cutoff.

    `decay` controls scoring inside that range:

        exponential
        linear
    """

    limit = validate_seconds(
        maximum_distance_seconds,
        name=(
            "maximum_distance_seconds"
        ),
    )

    delta = absolute_delta_seconds(
        source,
        target,
    )

    if delta is None:
        return 0.0

    if delta > limit:
        return 0.0

    if decay == "exponential":
        return (
            exponential_decay_score(
                delta,
                half_life_seconds=(
                    half_life_seconds
                ),
            )
        )

    if decay == "linear":
        return linear_decay_score(
            delta,
            maximum_distance_seconds=(
                limit
            ),
        )

    raise ValueError(
        "decay must be "
        "'exponential' or 'linear'"
    )


def temporal_relation(
    source: TelemetryRecord,
    target: TelemetryRecord,
    *,
    maximum_distance_seconds: float = (
        DEFAULT_MAX_DISTANCE_SECONDS
    ),
    half_life_seconds: float = (
        DEFAULT_HALF_LIFE_SECONDS
    ),
    simultaneous_tolerance: float = (
        DEFAULT_SIMULTANEOUS_TOLERANCE
    ),
    decay: str = "exponential",
) -> TemporalRelation:
    """
    Produce the complete temporal comparison.
    """

    limit = validate_seconds(
        maximum_distance_seconds,
        name=(
            "maximum_distance_seconds"
        ),
    )

    source_time = (
        telemetry_timestamp(
            source
        )
    )

    target_time = (
        telemetry_timestamp(
            target
        )
    )

    source_id = normalized_record_id(
        source
    )

    target_id = normalized_record_id(
        target
    )

    if (
        source_time is None
        or target_time is None
    ):
        return TemporalRelation(
            source_record_id=(
                source_id
            ),
            target_record_id=(
                target_id
            ),
            source_timestamp=(
                source.signal.timestamp
            ),
            target_timestamp=(
                target.signal.timestamp
            ),
            delta_seconds=None,
            signed_delta_seconds=None,
            order=(
                TemporalOrder.UNKNOWN.value
            ),
            direction=(
                CorrelationDirection.
                NONE.value
            ),
            within_range=False,
            score=0.0,
        )

    signed_delta = (
        target_time
        - source_time
    ).total_seconds()

    delta = abs(
        signed_delta
    )

    within_range = (
        delta <= limit
    )

    order = temporal_order(
        source,
        target,
        simultaneous_tolerance=(
            simultaneous_tolerance
        ),
    )

    direction = (
        correlation_direction(
            source,
            target,
            simultaneous_tolerance=(
                simultaneous_tolerance
            ),
        )
    )

    score = temporal_score(
        source,
        target,
        maximum_distance_seconds=(
            limit
        ),
        half_life_seconds=(
            half_life_seconds
        ),
        decay=decay,
    )

    return TemporalRelation(
        source_record_id=(
            source_id
        ),
        target_record_id=(
            target_id
        ),
        source_timestamp=(
            source.signal.timestamp
        ),
        target_timestamp=(
            target.signal.timestamp
        ),
        delta_seconds=delta,
        signed_delta_seconds=(
            signed_delta
        ),
        order=order,
        direction=direction,
        within_range=(
            within_range
        ),
        score=score,
    )


def temporal_evidence(
    source: TelemetryRecord,
    target: TelemetryRecord,
    *,
    maximum_distance_seconds: float = (
        DEFAULT_MAX_DISTANCE_SECONDS
    ),
    half_life_seconds: float = (
        DEFAULT_HALF_LIFE_SECONDS
    ),
    simultaneous_tolerance: float = (
        DEFAULT_SIMULTANEOUS_TOLERANCE
    ),
    decay: str = "exponential",
) -> Optional[
    CorrelationEvidence
]:
    """
    Convert temporal comparison into correlation
    evidence.

    No evidence is returned when timestamps are
    invalid or outside the hard time range.
    """

    relation = temporal_relation(
        source,
        target,
        maximum_distance_seconds=(
            maximum_distance_seconds
        ),
        half_life_seconds=(
            half_life_seconds
        ),
        simultaneous_tolerance=(
            simultaneous_tolerance
        ),
        decay=decay,
    )

    if (
        relation.delta_seconds
        is None
        or not relation.within_range
    ):
        return None

    return CorrelationEvidence(
        evidence_type=(
            EvidenceType.TEMPORAL.value
        ),
        score=relation.score,
        source_record_id=(
            relation.source_record_id
        ),
        target_record_id=(
            relation.target_record_id
        ),
        source_resource_id=(
            source.resource.effective_id()
        ),
        target_resource_id=(
            target.resource.effective_id()
        ),
        description=(
            "Telemetry observations occurred "
            f"{relation.delta_seconds:.6f} "
            "seconds apart"
        ),
        attributes={
            "delta_seconds": (
                relation.delta_seconds
            ),
            "signed_delta_seconds": (
                relation.
                signed_delta_seconds
            ),
            "order": relation.order,
            "direction": (
                relation.direction
            ),
            "maximum_distance_seconds": (
                maximum_distance_seconds
            ),
            "decay": decay,
        },
    )


def compare_many(
    anchor: TelemetryRecord,
    candidates: Iterable[
        TelemetryRecord
    ],
    *,
    maximum_distance_seconds: float = (
        DEFAULT_MAX_DISTANCE_SECONDS
    ),
    half_life_seconds: float = (
        DEFAULT_HALF_LIFE_SECONDS
    ),
    decay: str = "exponential",
) -> List[
    TemporalRelation
]:
    """
    Compare one observation with many candidates.
    """

    output = []

    anchor.finalize()

    for candidate in candidates:
        candidate.finalize()

        if (
            anchor.record_id
            == candidate.record_id
        ):
            continue

        output.append(
            temporal_relation(
                anchor,
                candidate,
                maximum_distance_seconds=(
                    maximum_distance_seconds
                ),
                half_life_seconds=(
                    half_life_seconds
                ),
                decay=decay,
            )
        )

    output.sort(
        key=lambda item: (
            item.delta_seconds
            if item.delta_seconds
            is not None
            else math.inf
        )
    )

    return output


def nearest_records(
    anchor: TelemetryRecord,
    candidates: Iterable[
        TelemetryRecord
    ],
    *,
    limit: int = 10,
    maximum_distance_seconds: float = (
        DEFAULT_MAX_DISTANCE_SECONDS
    ),
) -> List[
    TemporalRelation
]:
    """
    Return temporally nearest valid candidates.

    This is useful for diagnostics and candidate
    inspection, not causal inference.
    """

    if limit < 1:
        raise ValueError(
            "limit must be >= 1"
        )

    relations = compare_many(
        anchor,
        candidates,
        maximum_distance_seconds=(
            maximum_distance_seconds
        ),
    )

    return [
        relation
        for relation in relations
        if relation.within_range
    ][
        :limit
    ]


def ordered_records(
    records: Iterable[
        TelemetryRecord
    ],
) -> List[
    TelemetryRecord
]:
    """
    Return telemetry ordered by valid timestamp.

    Records with invalid timestamps are placed at
    the end.
    """

    return sorted(
        records,
        key=lambda telemetry: (
            telemetry_timestamp(
                telemetry
            )
            or datetime.max.replace(
                tzinfo=timezone.utc
            )
        ),
    )


def event_sequence(
    records: Iterable[
        TelemetryRecord
    ],
) -> List[
    Dict[str, Any]
]:
    """
    Produce a compact ordered timeline suitable
    for correlation diagnostics and eventual
    incident timelines.
    """

    output = []

    for telemetry in ordered_records(
        records
    ):
        telemetry.finalize()

        output.append(
            {
                "record_id": (
                    telemetry.record_id
                ),
                "timestamp": (
                    telemetry.signal.
                    timestamp
                ),
                "provider": (
                    telemetry.resource.
                    provider
                ),
                "resource_id": (
                    telemetry.resource.
                    effective_id()
                ),
                "signal_type": (
                    telemetry.signal.
                    signal_type
                ),
                "signal_name": (
                    telemetry.signal.
                    signal_name
                ),
                "severity": (
                    telemetry.signal.
                    severity
                ),
            }
        )

    return output


class TemporalAnalyzer:
    """
    Reusable temporal-correlation configuration.
    """

    def __init__(
        self,
        *,
        maximum_distance_seconds: float = (
            DEFAULT_MAX_DISTANCE_SECONDS
        ),
        half_life_seconds: float = (
            DEFAULT_HALF_LIFE_SECONDS
        ),
        simultaneous_tolerance: float = (
            DEFAULT_SIMULTANEOUS_TOLERANCE
        ),
        decay: str = "exponential",
    ) -> None:
        self.maximum_distance_seconds = (
            validate_seconds(
                maximum_distance_seconds,
                name=(
                    "maximum_distance_seconds"
                ),
            )
        )

        self.half_life_seconds = (
            validate_seconds(
                half_life_seconds,
                name=(
                    "half_life_seconds"
                ),
            )
        )

        self.simultaneous_tolerance = (
            validate_seconds(
                simultaneous_tolerance,
                name=(
                    "simultaneous_tolerance"
                ),
                allow_zero=True,
            )
        )

        if decay not in {
            "exponential",
            "linear",
        }:
            raise ValueError(
                "decay must be "
                "'exponential' or 'linear'"
            )

        self.decay = decay

    def compare(
        self,
        source: TelemetryRecord,
        target: TelemetryRecord,
    ) -> TemporalRelation:
        return temporal_relation(
            source,
            target,
            maximum_distance_seconds=(
                self.
                maximum_distance_seconds
            ),
            half_life_seconds=(
                self.half_life_seconds
            ),
            simultaneous_tolerance=(
                self.
                simultaneous_tolerance
            ),
            decay=self.decay,
        )

    def evidence(
        self,
        source: TelemetryRecord,
        target: TelemetryRecord,
    ) -> Optional[
        CorrelationEvidence
    ]:
        return temporal_evidence(
            source,
            target,
            maximum_distance_seconds=(
                self.
                maximum_distance_seconds
            ),
            half_life_seconds=(
                self.half_life_seconds
            ),
            simultaneous_tolerance=(
                self.
                simultaneous_tolerance
            ),
            decay=self.decay,
        )
