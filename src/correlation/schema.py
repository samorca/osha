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
    Sequence,
)


CORRELATION_SCHEMA_VERSION = "1"


class EvidenceType(str, Enum):
    """
    Evidence families that can contribute to a
    correlation.

    A single candidate may contain multiple
    independent types of evidence.
    """

    IDENTITY = "identity"

    TEMPORAL = "temporal"

    TOPOLOGY = "topology"

    RESOURCE = "resource"

    SERVICE = "service"

    SIGNAL = "signal"

    SEVERITY = "severity"

    EVENT = "event"

    METRIC = "metric"

    LOG = "log"

    HISTORICAL = "historical"

    BEHAVIORAL = "behavioral"

    MANUAL = "manual"

    OTHER = "other"


class CorrelationDecision(
    str,
    Enum,
):
    """
    High-level outcome of evaluating a
    correlation candidate.
    """

    CORRELATED = "correlated"

    CANDIDATE = "candidate"

    WEAK = "weak"

    REJECTED = "rejected"


class CorrelationDirection(
    str,
    Enum,
):
    """
    Correlation direction.

    NONE:
        Association only.

    FORWARD:
        Source happened before/points toward
        target.

    REVERSE:
        Target happened before/points toward
        source.

    BIDIRECTIONAL:
        Relationship is symmetric.

    Direction does NOT imply causation.
    """

    NONE = "none"

    FORWARD = "forward"

    REVERSE = "reverse"

    BIDIRECTIONAL = (
        "bidirectional"
    )


class GroupStatus(str, Enum):
    """
    Lifecycle of a correlation group.
    """

    OPEN = "open"

    STABLE = "stable"

    CLOSED = "closed"


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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
    Convert correlation objects into JSON-safe
    structures.

    Sensitive-data redaction is expected to have
    happened in src/telemetry before correlation.
    """

    if depth > maximum_depth:
        return (
            "[MAX_DEPTH_REACHED]"
        )

    if value is None:
        return None

    if isinstance(
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
            value = value.replace(
                tzinfo=timezone.utc
            )

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
                maximum_depth=(
                    maximum_depth
                ),
            )
            for key, item
            in value.items()
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
                maximum_depth=(
                    maximum_depth
                ),
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
            maximum_depth=(
                maximum_depth
            ),
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

    return hashlib.sha256(
        encoded
    ).hexdigest()


def checked_score(
    value: float,
) -> float:
    """
    Clamp a heuristic correlation score into the
    range 0.0 to 1.0.

    A correlation score is evidence strength. It
    is not automatically a calibrated
    probability.
    """

    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if not math.isfinite(
        number
    ):
        return 0.0

    return min(
        1.0,
        max(
            0.0,
            number,
        ),
    )


@dataclass(frozen=True)
class CorrelationEvidence:
    """
    One piece of evidence connecting two signals
    or resources.

    Examples:

        timestamps 5 seconds apart
        resources share identity
        pod runs on node
        volume attached to VM
        service names match
    """

    evidence_type: str

    score: float

    source_record_id: Optional[
        str
    ] = None

    target_record_id: Optional[
        str
    ] = None

    source_resource_id: Optional[
        str
    ] = None

    target_resource_id: Optional[
        str
    ] = None

    description: Optional[str] = (
        None
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def normalized_score(
        self,
    ) -> float:
        return checked_score(
            self.score
        )

    def to_dict(self) -> dict:
        return {
            "evidence_type": (
                self.evidence_type
            ),
            "score": (
                self.normalized_score()
            ),
            "source_record_id": (
                self.source_record_id
            ),
            "target_record_id": (
                self.target_record_id
            ),
            "source_resource_id": (
                self.source_resource_id
            ),
            "target_resource_id": (
                self.target_resource_id
            ),
            "description": (
                self.description
            ),
            "attributes": (
                json_safe(
                    self.attributes
                )
            ),
        }


@dataclass(frozen=True)
class CorrelationScore:
    """
    Breakdown of correlation evidence.

    Individual dimensions stay separate so the
    future intelligence layer can understand why
    two observations were correlated.

    These fields are heuristic scores, not
    calibrated probabilities.
    """

    identity: float = 0.0

    temporal: float = 0.0

    topology: float = 0.0

    resource: float = 0.0

    service: float = 0.0

    signal: float = 0.0

    severity: float = 0.0

    historical: float = 0.0

    overall: float = 0.0

    def normalized(
        self,
    ) -> "CorrelationScore":
        return CorrelationScore(
            identity=checked_score(
                self.identity
            ),
            temporal=checked_score(
                self.temporal
            ),
            topology=checked_score(
                self.topology
            ),
            resource=checked_score(
                self.resource
            ),
            service=checked_score(
                self.service
            ),
            signal=checked_score(
                self.signal
            ),
            severity=checked_score(
                self.severity
            ),
            historical=checked_score(
                self.historical
            ),
            overall=checked_score(
                self.overall
            ),
        )

    def to_dict(self) -> dict:
        return json_safe(
            asdict(
                self.normalized()
            )
        )


@dataclass
class CorrelationCandidate:
    """
    Pair of observations/resources that may be
    related.

    Candidate generation belongs in correlate.py.
    Scoring belongs in scoring.py.
    """

    source_record_id: str

    target_record_id: str

    source_resource_id: str

    target_resource_id: str

    source_timestamp: str

    target_timestamp: str

    environment_id: str

    source_provider: Optional[
        str
    ] = None

    target_provider: Optional[
        str
    ] = None

    source_signal_type: Optional[
        str
    ] = None

    target_signal_type: Optional[
        str
    ] = None

    direction: str = (
        CorrelationDirection.NONE.value
    )

    evidence: List[
        CorrelationEvidence
    ] = field(
        default_factory=list
    )

    scores: CorrelationScore = field(
        default_factory=(
            CorrelationScore
        )
    )

    decision: str = (
        CorrelationDecision.WEAK.value
    )

    created_at: str = field(
        default_factory=utc_now
    )

    candidate_id: Optional[str] = (
        None
    )

    def build_candidate_id(
        self,
    ) -> str:
        body = {
            "source_record_id": (
                self.source_record_id
            ),
            "target_record_id": (
                self.target_record_id
            ),
            "source_resource_id": (
                self.source_resource_id
            ),
            "target_resource_id": (
                self.target_resource_id
            ),
            "environment_id": (
                self.environment_id
            ),
        }

        return (
            "corr:candidate:"
            + stable_hash(body)
        )

    def finalize(
        self,
    ) -> "CorrelationCandidate":
        if not self.candidate_id:
            self.candidate_id = (
                self.build_candidate_id()
            )

        return self

    def add_evidence(
        self,
        evidence: (
            CorrelationEvidence
        ),
    ) -> None:
        self.evidence.append(
            evidence
        )

    def to_dict(self) -> dict:
        self.finalize()

        return {
            "candidate_id": (
                self.candidate_id
            ),
            "source_record_id": (
                self.source_record_id
            ),
            "target_record_id": (
                self.target_record_id
            ),
            "source_resource_id": (
                self.source_resource_id
            ),
            "target_resource_id": (
                self.target_resource_id
            ),
            "source_timestamp": (
                self.source_timestamp
            ),
            "target_timestamp": (
                self.target_timestamp
            ),
            "environment_id": (
                self.environment_id
            ),
            "source_provider": (
                self.source_provider
            ),
            "target_provider": (
                self.target_provider
            ),
            "source_signal_type": (
                self.source_signal_type
            ),
            "target_signal_type": (
                self.target_signal_type
            ),
            "direction": (
                self.direction
            ),
            "evidence": [
                item.to_dict()
                for item
                in self.evidence
            ],
            "scores": (
                self.scores.to_dict()
            ),
            "decision": (
                self.decision
            ),
            "created_at": (
                self.created_at
            ),
        }


@dataclass(frozen=True)
class TimeWindow:
    """
    Correlation time window.

    The window specifies scope only. Being inside
    one window does not itself prove correlation.
    """

    start: str

    end: str

    window_seconds: float

    environment_id: str

    anchor_record_id: Optional[
        str
    ] = None

    anchor_resource_id: Optional[
        str
    ] = None

    def to_dict(self) -> dict:
        return json_safe(
            asdict(self)
        )


@dataclass
class CorrelationGroup:
    """
    Group of telemetry records considered part
    of one correlated episode.

    This is not yet an incident. The
    intelligence layer may later promote a group
    into an incident.
    """

    environment_id: str

    record_ids: List[str]

    resource_ids: List[str]

    start_timestamp: str

    end_timestamp: str

    group_id: Optional[str] = None

    candidate_ids: List[
        str
    ] = field(
        default_factory=list
    )

    providers: List[str] = field(
        default_factory=list
    )

    signal_types: List[str] = field(
        default_factory=list
    )

    strongest_score: float = 0.0

    average_score: float = 0.0

    status: str = (
        GroupStatus.OPEN.value
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=utc_now
    )

    def build_group_id(
        self,
    ) -> str:
        body = {
            "environment_id": (
                self.environment_id
            ),
            "record_ids": sorted(
                set(
                    self.record_ids
                )
            ),
            "resource_ids": sorted(
                set(
                    self.resource_ids
                )
            ),
            "start_timestamp": (
                self.start_timestamp
            ),
            "end_timestamp": (
                self.end_timestamp
            ),
        }

        return (
            "corr:group:"
            + stable_hash(body)
        )

    def finalize(
        self,
    ) -> "CorrelationGroup":
        self.record_ids = sorted(
            set(
                self.record_ids
            )
        )

        self.resource_ids = sorted(
            set(
                self.resource_ids
            )
        )

        self.candidate_ids = sorted(
            set(
                self.candidate_ids
            )
        )

        self.providers = sorted(
            set(
                self.providers
            )
        )

        self.signal_types = sorted(
            set(
                self.signal_types
            )
        )

        self.strongest_score = (
            checked_score(
                self.strongest_score
            )
        )

        self.average_score = (
            checked_score(
                self.average_score
            )
        )

        if not self.group_id:
            self.group_id = (
                self.build_group_id()
            )

        return self

    def to_dict(self) -> dict:
        self.finalize()

        return {
            "group_id": (
                self.group_id
            ),
            "environment_id": (
                self.environment_id
            ),
            "record_ids": (
                self.record_ids
            ),
            "resource_ids": (
                self.resource_ids
            ),
            "candidate_ids": (
                self.candidate_ids
            ),
            "providers": (
                self.providers
            ),
            "signal_types": (
                self.signal_types
            ),
            "start_timestamp": (
                self.start_timestamp
            ),
            "end_timestamp": (
                self.end_timestamp
            ),
            "strongest_score": (
                self.strongest_score
            ),
            "average_score": (
                self.average_score
            ),
            "status": self.status,
            "attributes": (
                json_safe(
                    self.attributes
                )
            ),
            "created_at": (
                self.created_at
            ),
        }


@dataclass
class CorrelationResult:
    """
    Final output from one execution of the
    correlation pipeline.
    """

    groups: List[
        CorrelationGroup
    ] = field(
        default_factory=list
    )

    candidates: List[
        CorrelationCandidate
    ] = field(
        default_factory=list
    )

    uncorrelated_record_ids: List[
        str
    ] = field(
        default_factory=list
    )

    schema_version: str = (
        CORRELATION_SCHEMA_VERSION
    )

    generated_at: str = field(
        default_factory=utc_now
    )

    metadata: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "CorrelationResult":
        for candidate in (
            self.candidates
        ):
            candidate.finalize()

        for group in self.groups:
            group.finalize()

        self.uncorrelated_record_ids = (
            sorted(
                set(
                    self.
                    uncorrelated_record_ids
                )
            )
        )

        return self

    def to_dict(self) -> dict:
        self.finalize()

        return {
            "schema_version": (
                self.schema_version
            ),
            "generated_at": (
                self.generated_at
            ),
            "groups": [
                group.to_dict()
                for group
                in self.groups
            ],
            "candidates": [
                candidate.to_dict()
                for candidate
                in self.candidates
            ],
            "uncorrelated_record_ids": (
                self.
                uncorrelated_record_ids
            ),
            "metadata": (
                json_safe(
                    self.metadata
                )
            ),
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


def combine_independent_scores(
    scores: Iterable[float],
) -> float:
    """
    Combine independent heuristic evidence using:

        1 - product(1 - score)

    scoring.py may use a more explicit weighted
    model later. This helper is useful when
    combining evidence inside one dimension.

    Result is evidence strength, not a calibrated
    probability.
    """

    remaining = 1.0

    found = False

    for raw_score in scores:
        score = checked_score(
            raw_score
        )

        found = True

        remaining *= (
            1.0 - score
        )

    if not found:
        return 0.0

    return checked_score(
        1.0 - remaining
    )


def evidence_by_type(
    evidence: Sequence[
        CorrelationEvidence
    ],
) -> Dict[
    str,
    List[CorrelationEvidence],
]:
    output: Dict[
        str,
        List[CorrelationEvidence],
    ] = {}

    for item in evidence:
        output.setdefault(
            item.evidence_type,
            [],
        ).append(item)

    return output


def strongest_evidence(
    evidence: Iterable[
        CorrelationEvidence
    ],
) -> Optional[
    CorrelationEvidence
]:
    strongest = None

    for item in evidence:
        if (
            strongest is None
            or item.normalized_score()
            > strongest.normalized_score()
        ):
            strongest = item

    return strongest
