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
    List,
    Mapping,
    Optional,
)


MEMORY_SCHEMA_VERSION = "1"


class MemoryType(str, Enum):
    """
    High-level categories of operational memory.
    """

    EPISODE = "episode"

    PATTERN = "pattern"

    BASELINE = "baseline"

    INCIDENT = "incident"

    ROOT_CAUSE = "root_cause"

    OUTCOME = "outcome"


class MemoryStatus(str, Enum):
    """
    Memory lifecycle state.
    """

    ACTIVE = "active"

    ARCHIVED = "archived"

    EXPIRED = "expired"


class VerificationStatus(str, Enum):
    """
    Whether an operational conclusion has been
    independently verified.

    This distinction is important because OSHA
    must not treat a hypothesis as historical
    truth merely because it was previously
    generated.
    """

    UNVERIFIED = "unverified"

    PARTIALLY_VERIFIED = (
        "partially_verified"
    )

    VERIFIED = "verified"

    REJECTED = "rejected"


class OutcomeType(str, Enum):
    """
    Recorded outcome after investigation or
    remediation.
    """

    RECOVERED = "recovered"

    IMPROVED = "improved"

    UNCHANGED = "unchanged"

    DEGRADED = "degraded"

    FAILED = "failed"

    UNKNOWN = "unknown"


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


def checked_score(
    value: Any,
) -> float:
    """
    Normalize evidence strength to 0..1.

    Memory scores are evidence metadata unless
    explicitly produced by a calibrated model.
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


def json_safe(
    value: Any,
    *,
    depth: int = 0,
    maximum_depth: int = 20,
) -> Any:
    """
    Convert memory data to JSON-safe structures.

    Sensitive information is expected to have
    already passed the telemetry redaction
    boundary before entering memory.
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
        if math.isfinite(
            value
        ):
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
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


@dataclass
class MemoryRecord:
    """
    Base operational-memory contract.

    Every persisted memory object should have:

      stable identity
      environment scope
      creation/update timestamps
      verification state
      provenance
    """

    memory_type: str

    environment_id: str

    memory_id: Optional[str] = None

    status: str = (
        MemoryStatus.ACTIVE.value
    )

    verification_status: str = (
        VerificationStatus.
        UNVERIFIED.value
    )

    created_at: str = field(
        default_factory=utc_now
    )

    updated_at: str = field(
        default_factory=utc_now
    )

    last_accessed_at: Optional[
        str
    ] = None

    source_ids: List[str] = field(
        default_factory=list
    )

    resource_ids: List[str] = field(
        default_factory=list
    )

    tags: List[str] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    schema_version: str = (
        MEMORY_SCHEMA_VERSION
    )

    def identity_payload(
        self,
    ) -> Dict[str, Any]:
        """
        Fields used for deterministic memory ID
        generation.

        Subclasses may override this.
        """

        return {
            "memory_type": (
                self.memory_type
            ),
            "environment_id": (
                self.environment_id
            ),
            "source_ids": sorted(
                set(
                    self.source_ids
                )
            ),
            "resource_ids": sorted(
                set(
                    self.resource_ids
                )
            ),
        }

    def build_memory_id(
        self,
    ) -> str:
        return (
            "memory:"
            f"{self.memory_type}:"
            + stable_hash(
                self.identity_payload()
            )
        )

    def finalize(
        self,
    ) -> "MemoryRecord":
        self.source_ids = sorted(
            set(
                self.source_ids
            )
        )

        self.resource_ids = sorted(
            set(
                self.resource_ids
            )
        )

        self.tags = sorted(
            set(
                self.tags
            )
        )

        if not self.memory_id:
            self.memory_id = (
                self.build_memory_id()
            )

        return self

    def touch(
        self,
    ) -> "MemoryRecord":
        self.updated_at = utc_now()

        return self

    def mark_accessed(
        self,
    ) -> "MemoryRecord":
        self.last_accessed_at = (
            utc_now()
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return json_safe(
            asdict(self)
        )

    def to_json(
        self,
    ) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
            default=str,
        )


@dataclass
class EpisodeMemory(
    MemoryRecord
):
    """
    Historical memory of one correlated
    operational episode.

    An episode is not automatically an incident.
    """

    memory_type: str = field(
        default=(
            MemoryType.EPISODE.value
        ),
        init=False,
    )

    correlation_group_id: Optional[
        str
    ] = None

    record_ids: List[str] = field(
        default_factory=list
    )

    providers: List[str] = field(
        default_factory=list
    )

    signal_types: List[str] = field(
        default_factory=list
    )

    started_at: Optional[str] = None

    ended_at: Optional[str] = None

    strongest_correlation_score: float = (
        0.0
    )

    average_correlation_score: float = (
        0.0
    )

    incident_id: Optional[str] = None

    summary: Optional[str] = None

    def identity_payload(
        self,
    ) -> Dict[str, Any]:
        return {
            "memory_type": (
                self.memory_type
            ),
            "environment_id": (
                self.environment_id
            ),
            "correlation_group_id": (
                self.correlation_group_id
            ),
            "record_ids": sorted(
                set(
                    self.record_ids
                )
            ),
        }

    def finalize(
        self,
    ) -> "EpisodeMemory":
        self.record_ids = sorted(
            set(
                self.record_ids
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

        self.strongest_correlation_score = (
            checked_score(
                self.
                strongest_correlation_score
            )
        )

        self.average_correlation_score = (
            checked_score(
                self.
                average_correlation_score
            )
        )

        super().finalize()

        return self


@dataclass
class PatternMemory(
    MemoryRecord
):
    """
    Long-term history for a stable operational
    pattern fingerprint.
    """

    memory_type: str = field(
        default=(
            MemoryType.PATTERN.value
        ),
        init=False,
    )

    pattern_id: Optional[str] = None

    fingerprint: str = ""

    pattern_type: str = "unknown"

    name: Optional[str] = None

    occurrence_count: int = 0

    first_seen: Optional[str] = None

    last_seen: Optional[str] = None

    providers: List[str] = field(
        default_factory=list
    )

    signal_types: List[str] = field(
        default_factory=list
    )

    incident_ids: List[str] = field(
        default_factory=list
    )

    verified_incident_count: int = 0

    verified_normal_count: int = 0

    latest_novelty_score: float = (
        0.0
    )

    def identity_payload(
        self,
    ) -> Dict[str, Any]:
        return {
            "memory_type": (
                self.memory_type
            ),
            "environment_id": (
                self.environment_id
            ),
            "fingerprint": (
                self.fingerprint
            ),
            "pattern_type": (
                self.pattern_type
            ),
        }

    def finalize(
        self,
    ) -> "PatternMemory":
        self.occurrence_count = max(
            0,
            int(
                self.occurrence_count
            ),
        )

        self.verified_incident_count = (
            max(
                0,
                int(
                    self.
                    verified_incident_count
                ),
            )
        )

        self.verified_normal_count = (
            max(
                0,
                int(
                    self.
                    verified_normal_count
                ),
            )
        )

        self.latest_novelty_score = (
            checked_score(
                self.latest_novelty_score
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

        self.incident_ids = sorted(
            set(
                self.incident_ids
            )
        )

        super().finalize()

        return self


@dataclass
class BaselineMemory(
    MemoryRecord
):
    """
    Historical baseline for a resource/signal.

    Examples:

      normal CPU range
      service restart frequency
      log-pattern frequency
      memory usage baseline
      event frequency

    Statistical calculation belongs in
    baselines.py rather than this schema.
    """

    memory_type: str = field(
        default=(
            MemoryType.BASELINE.value
        ),
        init=False,
    )

    baseline_key: str = ""

    resource_id: Optional[str] = None

    signal_name: Optional[str] = None

    metric_unit: Optional[str] = None

    sample_count: int = 0

    minimum: Optional[float] = None

    maximum: Optional[float] = None

    mean: Optional[float] = None

    variance: Optional[float] = None

    standard_deviation: Optional[
        float
    ] = None

    first_sample_at: Optional[str] = None

    last_sample_at: Optional[str] = None

    model_type: str = "descriptive"

    def identity_payload(
        self,
    ) -> Dict[str, Any]:
        return {
            "memory_type": (
                self.memory_type
            ),
            "environment_id": (
                self.environment_id
            ),
            "baseline_key": (
                self.baseline_key
            ),
            "resource_id": (
                self.resource_id
            ),
            "signal_name": (
                self.signal_name
            ),
        }

    def finalize(
        self,
    ) -> "BaselineMemory":
        self.sample_count = max(
            0,
            int(
                self.sample_count
            ),
        )

        super().finalize()

        return self


@dataclass
class RootCauseMemory(
    MemoryRecord
):
    """
    Historical root-cause hypothesis or verified
    root-cause outcome.

    verification_status is critical here.
    Unverified hypotheses must never become
    historical ground truth automatically.
    """

    memory_type: str = field(
        default=(
            MemoryType.ROOT_CAUSE.value
        ),
        init=False,
    )

    incident_id: Optional[str] = None

    candidate_id: Optional[str] = None

    root_resource_id: Optional[
        str
    ] = None

    title: Optional[str] = None

    hypothesis_score: float = 0.0

    evidence_types: List[str] = field(
        default_factory=list
    )

    causal_path: List[str] = field(
        default_factory=list
    )

    verification_source: Optional[
        str
    ] = None

    verified_at: Optional[str] = None

    def identity_payload(
        self,
    ) -> Dict[str, Any]:
        return {
            "memory_type": (
                self.memory_type
            ),
            "environment_id": (
                self.environment_id
            ),
            "incident_id": (
                self.incident_id
            ),
            "candidate_id": (
                self.candidate_id
            ),
            "root_resource_id": (
                self.root_resource_id
            ),
        }

    def finalize(
        self,
    ) -> "RootCauseMemory":
        self.hypothesis_score = (
            checked_score(
                self.hypothesis_score
            )
        )

        self.evidence_types = sorted(
            set(
                self.evidence_types
            )
        )

        super().finalize()

        return self


@dataclass
class IncidentMemory(
    MemoryRecord
):
    """
    Historical incident candidate or verified
    incident.

    This stores investigation state, not raw
    unbounded telemetry.
    """

    memory_type: str = field(
        default=(
            MemoryType.INCIDENT.value
        ),
        init=False,
    )

    incident_id: str = ""

    title: Optional[str] = None

    severity: str = "unknown"

    incident_status: str = "candidate"

    started_at: Optional[str] = None

    ended_at: Optional[str] = None

    anomaly_ids: List[str] = field(
        default_factory=list
    )

    pattern_ids: List[str] = field(
        default_factory=list
    )

    correlation_group_ids: List[
        str
    ] = field(
        default_factory=list
    )

    root_cause_memory_ids: List[
        str
    ] = field(
        default_factory=list
    )

    affected_resource_ids: List[
        str
    ] = field(
        default_factory=list
    )

    affected_services: List[str] = field(
        default_factory=list
    )

    impact_level: str = "unknown"

    confidence_score: float = 0.0

    confidence_level: str = "none"

    explanation_summary: Optional[
        str
    ] = None

    outcome_memory_id: Optional[
        str
    ] = None

    def identity_payload(
        self,
    ) -> Dict[str, Any]:
        return {
            "memory_type": (
                self.memory_type
            ),
            "environment_id": (
                self.environment_id
            ),
            "incident_id": (
                self.incident_id
            ),
        }

    def finalize(
        self,
    ) -> "IncidentMemory":
        self.anomaly_ids = sorted(
            set(
                self.anomaly_ids
            )
        )

        self.pattern_ids = sorted(
            set(
                self.pattern_ids
            )
        )

        self.correlation_group_ids = (
            sorted(
                set(
                    self.
                    correlation_group_ids
                )
            )
        )

        self.root_cause_memory_ids = (
            sorted(
                set(
                    self.
                    root_cause_memory_ids
                )
            )
        )

        self.affected_resource_ids = (
            sorted(
                set(
                    self.
                    affected_resource_ids
                )
            )
        )

        self.affected_services = sorted(
            set(
                self.affected_services
            )
        )

        self.confidence_score = (
            checked_score(
                self.confidence_score
            )
        )

        super().finalize()

        return self


@dataclass
class OutcomeMemory(
    MemoryRecord
):
    """
    Verified or observed outcome after an
    incident investigation or remediation.

    This is deliberately separate from the
    remediation implementation.
    """

    memory_type: str = field(
        default=(
            MemoryType.OUTCOME.value
        ),
        init=False,
    )

    incident_id: str = ""

    outcome_type: str = (
        OutcomeType.UNKNOWN.value
    )

    summary: Optional[str] = None

    remediation_action_ids: List[
        str
    ] = field(
        default_factory=list
    )

    before_health_score: Optional[
        float
    ] = None

    after_health_score: Optional[
        float
    ] = None

    recovery_verified: bool = False

    verification_source: Optional[
        str
    ] = None

    observed_at: str = field(
        default_factory=utc_now
    )

    def identity_payload(
        self,
    ) -> Dict[str, Any]:
        return {
            "memory_type": (
                self.memory_type
            ),
            "environment_id": (
                self.environment_id
            ),
            "incident_id": (
                self.incident_id
            ),
            "outcome_type": (
                self.outcome_type
            ),
            "observed_at": (
                self.observed_at
            ),
        }

    def finalize(
        self,
    ) -> "OutcomeMemory":
        self.remediation_action_ids = (
            sorted(
                set(
                    self.
                    remediation_action_ids
                )
            )
        )

        if (
            self.before_health_score
            is not None
        ):
            self.before_health_score = (
                checked_score(
                    self.
                    before_health_score
                )
            )

        if (
            self.after_health_score
            is not None
        ):
            self.after_health_score = (
                checked_score(
                    self.
                    after_health_score
                )
            )

        super().finalize()

        return self


MemoryObject = (
    EpisodeMemory
    | PatternMemory
    | BaselineMemory
    | IncidentMemory
    | RootCauseMemory
    | OutcomeMemory
)


def memory_type(
    memory: MemoryObject,
) -> str:
    return memory.memory_type


def memory_is_verified(
    memory: MemoryObject,
) -> bool:
    return (
        memory.verification_status
        == VerificationStatus.
        VERIFIED.value
    )


def memory_is_active(
    memory: MemoryObject,
) -> bool:
    return (
        memory.status
        == MemoryStatus.ACTIVE.value
    )


def memory_summary(
    memories: List[
        MemoryObject
    ],
) -> Dict[str, Any]:
    by_type: Dict[
        str,
        int,
    ] = {}

    by_verification: Dict[
        str,
        int,
    ] = {}

    by_status: Dict[
        str,
        int,
    ] = {}

    environments = set()

    resources = set()

    for memory in memories:
        memory.finalize()

        by_type[
            memory.memory_type
        ] = (
            by_type.get(
                memory.memory_type,
                0,
            )
            + 1
        )

        by_verification[
            memory.verification_status
        ] = (
            by_verification.get(
                memory.
                verification_status,
                0,
            )
            + 1
        )

        by_status[
            memory.status
        ] = (
            by_status.get(
                memory.status,
                0,
            )
            + 1
        )

        environments.add(
            memory.environment_id
        )

        resources.update(
            memory.resource_ids
        )

    return {
        "memory_count": len(
            memories
        ),
        "memory_types": by_type,
        "verification_statuses": (
            by_verification
        ),
        "statuses": by_status,
        "environment_count": len(
            environments
        ),
        "resource_count": len(
            resources
        ),
    }
