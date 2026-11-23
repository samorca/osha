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


SELF_HEALING_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------
# Lifecycle enums
# ---------------------------------------------------------------

class HealingState(
    str,
    Enum,
):
    """
    Overall self-healing lifecycle.
    """

    CREATED = "created"

    OBSERVING = "observing"

    DETECTED = "detected"

    ANALYZING = "analyzing"

    DECIDING = "deciding"

    AWAITING_APPROVAL = (
        "awaiting_approval"
    )

    REMEDIATING = "remediating"

    VERIFYING = "verifying"

    STABILIZING = "stabilizing"

    RECOVERED = "recovered"

    DEGRADED = "degraded"

    ESCALATED = "escalated"

    FAILED = "failed"

    CLOSED = "closed"


class RecoveryStatus(
    str,
    Enum,
):
    """
    Current recovery state.
    """

    UNKNOWN = "unknown"

    NOT_STARTED = "not_started"

    MONITORING = "monitoring"

    RECOVERING = "recovering"

    RECOVERED = "recovered"

    PARTIAL = "partial"

    UNSTABLE = "unstable"

    FAILED = "failed"


class StabilityStatus(
    str,
    Enum,
):
    """
    Stability after apparent recovery.
    """

    UNKNOWN = "unknown"

    OBSERVING = "observing"

    STABLE = "stable"

    UNSTABLE = "unstable"

    REGRESSED = "regressed"


class EscalationLevel(
    str,
    Enum,
):
    """
    Abstract escalation level.

    Identity and notification integrations belong
    outside this schema.
    """

    NONE = "none"

    OPERATOR = "operator"

    SENIOR_OPERATOR = (
        "senior_operator"
    )

    INCIDENT_COMMAND = (
        "incident_command"
    )


class OutcomeStatus(
    str,
    Enum,
):
    """
    Final observed incident outcome.
    """

    UNKNOWN = "unknown"

    RECOVERED = "recovered"

    PARTIALLY_RECOVERED = (
        "partially_recovered"
    )

    NOT_RECOVERED = (
        "not_recovered"
    )

    ROLLED_BACK = "rolled_back"

    ESCALATED = "escalated"


# ---------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def checked_score(
    value: Any,
) -> float:
    try:
        number = float(
            value
        )

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
    if depth > maximum_depth:
        return (
            "[MAX_DEPTH_REACHED]"
        )

    if (
        value is None
        or isinstance(
            value,
            (
                bool,
                int,
                str,
            ),
        )
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        return (
            value
            if math.isfinite(
                value
            )
            else None
        )

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
    value: Mapping[
        str,
        Any,
    ],
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


# ---------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------

@dataclass(frozen=True)
class HealingEvidence:
    """
    Evidence contributing to recovery, stability
    or escalation conclusions.

    The self-healing layer should consume
    normalized evidence rather than unrestricted
    raw command output.
    """

    evidence_type: str

    description: str

    score: float = 0.0

    source_id: Optional[
        str
    ] = None

    resource_ids: List[
        str
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    observed_at: str = field(
        default_factory=utc_now
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "evidence_type": (
                self.evidence_type
            ),
            "description": (
                self.description
            ),
            "score": (
                checked_score(
                    self.score
                )
            ),
            "source_id": (
                self.source_id
            ),
            "resource_ids": (
                sorted(
                    set(
                        self.resource_ids
                    )
                )
            ),
            "attributes": (
                json_safe(
                    self.attributes
                )
            ),
            "observed_at": (
                self.observed_at
            ),
        }


# ---------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------

@dataclass
class RecoveryAssessment:
    """
    Current evidence about actual infrastructure
    recovery.

    Recovery should ultimately be derived from
    post-remediation telemetry rather than simply
    from successful command execution.
    """

    status: str = (
        RecoveryStatus.UNKNOWN.value
    )

    score: float = 0.0

    health_before: Optional[
        float
    ] = None

    health_current: Optional[
        float
    ] = None

    recovered_resource_ids: List[
        str
    ] = field(
        default_factory=list
    )

    unhealthy_resource_ids: List[
        str
    ] = field(
        default_factory=list
    )

    checks: List[
        str
    ] = field(
        default_factory=list
    )

    failed_checks: List[
        str
    ] = field(
        default_factory=list
    )

    evidence: List[
        HealingEvidence
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    assessed_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "RecoveryAssessment":
        self.score = checked_score(
            self.score
        )

        if (
            self.health_before
            is not None
        ):
            self.health_before = (
                checked_score(
                    self.health_before
                )
            )

        if (
            self.health_current
            is not None
        ):
            self.health_current = (
                checked_score(
                    self.health_current
                )
            )

        self.recovered_resource_ids = (
            sorted(
                set(
                    self.
                    recovered_resource_ids
                )
            )
        )

        self.unhealthy_resource_ids = (
            sorted(
                set(
                    self.
                    unhealthy_resource_ids
                )
            )
        )

        self.checks = list(
            dict.fromkeys(
                self.checks
            )
        )

        self.failed_checks = list(
            dict.fromkeys(
                self.failed_checks
            )
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return json_safe(
            asdict(self)
        )


# ---------------------------------------------------------------
# Stability
# ---------------------------------------------------------------

@dataclass
class StabilityAssessment:
    """
    Determines whether apparent recovery remains
    stable over later observations.
    """

    status: str = (
        StabilityStatus.
        UNKNOWN.value
    )

    score: float = 0.0

    sample_count: int = 0

    observation_window_seconds: Optional[
        float
    ] = None

    regression_detected: bool = False

    reasons: List[
        str
    ] = field(
        default_factory=list
    )

    evidence: List[
        HealingEvidence
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    assessed_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "StabilityAssessment":
        self.score = checked_score(
            self.score
        )

        self.sample_count = max(
            0,
            int(
                self.sample_count
            ),
        )

        if (
            self.
            observation_window_seconds
            is not None
        ):
            self.observation_window_seconds = (
                max(
                    0.0,
                    float(
                        self.
                        observation_window_seconds
                    ),
                )
            )

        self.reasons = list(
            dict.fromkeys(
                self.reasons
            )
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return json_safe(
            asdict(self)
        )


# ---------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------

@dataclass
class EscalationRequest:
    """
    Escalation recommendation when automated
    recovery cannot establish a safe stable
    result.

    This object does not contact anyone.
    """

    level: str = (
        EscalationLevel.NONE.value
    )

    required: bool = False

    reason: Optional[
        str
    ] = None

    required_roles: List[
        str
    ] = field(
        default_factory=list
    )

    incident_id: Optional[
        str
    ] = None

    action_id: Optional[
        str
    ] = None

    remediation_id: Optional[
        str
    ] = None

    evidence: List[
        HealingEvidence
    ] = field(
        default_factory=list
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

    escalation_id: Optional[
        str
    ] = None

    def build_escalation_id(
        self,
    ) -> str:
        return (
            "self-healing:escalation:"
            + stable_hash(
                {
                    "level": self.level,
                    "incident_id": (
                        self.incident_id
                    ),
                    "action_id": (
                        self.action_id
                    ),
                    "remediation_id": (
                        self.remediation_id
                    ),
                }
            )
        )

    def finalize(
        self,
    ) -> "EscalationRequest":
        self.required_roles = sorted(
            set(
                self.required_roles
            )
        )

        if not self.escalation_id:
            self.escalation_id = (
                self.build_escalation_id()
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return json_safe(
            asdict(self)
        )


# ---------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------

@dataclass
class HealingOutcome:
    """
    Final observed outcome.

    This is the object that can eventually be
    translated into memory/outcome records.
    """

    status: str = (
        OutcomeStatus.UNKNOWN.value
    )

    incident_id: Optional[
        str
    ] = None

    decision_id: Optional[
        str
    ] = None

    action_id: Optional[
        str
    ] = None

    remediation_id: Optional[
        str
    ] = None

    recovery_score: float = 0.0

    stability_score: float = 0.0

    recovery_verified: bool = False

    stable: bool = False

    rolled_back: bool = False

    escalated: bool = False

    summary: Optional[
        str
    ] = None

    evidence: List[
        HealingEvidence
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    completed_at: str = field(
        default_factory=utc_now
    )

    outcome_id: Optional[
        str
    ] = None

    def build_outcome_id(
        self,
    ) -> str:
        return (
            "self-healing:outcome:"
            + stable_hash(
                {
                    "incident_id": (
                        self.incident_id
                    ),
                    "decision_id": (
                        self.decision_id
                    ),
                    "action_id": (
                        self.action_id
                    ),
                    "remediation_id": (
                        self.remediation_id
                    ),
                    "status": (
                        self.status
                    ),
                }
            )
        )

    def finalize(
        self,
    ) -> "HealingOutcome":
        self.recovery_score = (
            checked_score(
                self.recovery_score
            )
        )

        self.stability_score = (
            checked_score(
                self.stability_score
            )
        )

        if not self.outcome_id:
            self.outcome_id = (
                self.build_outcome_id()
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return json_safe(
            asdict(self)
        )


# ---------------------------------------------------------------
# Healing session
# ---------------------------------------------------------------

@dataclass
class HealingSession:
    """
    State for one self-healing incident lifecycle.

    This connects upstream incident/decision
    identifiers to remediation and observed
    recovery.
    """

    environment_id: str

    incident_id: str

    state: str = (
        HealingState.CREATED.value
    )

    decision_id: Optional[
        str
    ] = None

    plan_id: Optional[
        str
    ] = None

    action_id: Optional[
        str
    ] = None

    remediation_id: Optional[
        str
    ] = None

    resource_ids: List[
        str
    ] = field(
        default_factory=list
    )

    service_ids: List[
        str
    ] = field(
        default_factory=list
    )

    recovery: RecoveryAssessment = field(
        default_factory=RecoveryAssessment
    )

    stability: StabilityAssessment = field(
        default_factory=StabilityAssessment
    )

    escalation: EscalationRequest = field(
        default_factory=EscalationRequest
    )

    outcome: Optional[
        HealingOutcome
    ] = None

    evidence: List[
        HealingEvidence
    ] = field(
        default_factory=list
    )

    limitations: List[
        str
    ] = field(
        default_factory=list
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

    updated_at: str = field(
        default_factory=utc_now
    )

    session_id: Optional[
        str
    ] = None

    def build_session_id(
        self,
    ) -> str:
        return (
            "self-healing:session:"
            + stable_hash(
                {
                    "environment_id": (
                        self.environment_id
                    ),
                    "incident_id": (
                        self.incident_id
                    ),
                    "decision_id": (
                        self.decision_id
                    ),
                    "action_id": (
                        self.action_id
                    ),
                }
            )
        )

    def finalize(
        self,
    ) -> "HealingSession":
        self.resource_ids = sorted(
            set(
                self.resource_ids
            )
        )

        self.service_ids = sorted(
            set(
                self.service_ids
            )
        )

        self.limitations = list(
            dict.fromkeys(
                self.limitations
            )
        )

        self.recovery.finalize()

        self.stability.finalize()

        self.escalation.finalize()

        if self.outcome is not None:
            self.outcome.finalize()

        if not self.session_id:
            self.session_id = (
                self.build_session_id()
            )

        self.updated_at = (
            utc_now()
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (
                SELF_HEALING_SCHEMA_VERSION
            ),
            "session_id": (
                self.session_id
            ),
            "environment_id": (
                self.environment_id
            ),
            "incident_id": (
                self.incident_id
            ),
            "state": self.state,
            "decision_id": (
                self.decision_id
            ),
            "plan_id": (
                self.plan_id
            ),
            "action_id": (
                self.action_id
            ),
            "remediation_id": (
                self.remediation_id
            ),
            "resource_ids": (
                self.resource_ids
            ),
            "service_ids": (
                self.service_ids
            ),
            "recovery": (
                self.recovery.to_dict()
            ),
            "stability": (
                self.stability.to_dict()
            ),
            "escalation": (
                self.escalation.to_dict()
            ),
            "outcome": (
                self.outcome.to_dict()
                if self.outcome
                else None
            ),
            "evidence": [
                item.to_dict()
                for item
                in self.evidence
            ],
            "limitations": (
                self.limitations
            ),
            "attributes": (
                json_safe(
                    self.attributes
                )
            ),
            "created_at": (
                self.created_at
            ),
            "updated_at": (
                self.updated_at
            ),
        }


# ---------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------

@dataclass
class SelfHealingResult:
    """
    Complete output from one self-healing
    orchestration cycle.
    """

    sessions: List[
        HealingSession
    ] = field(
        default_factory=list
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

    schema_version: str = (
        SELF_HEALING_SCHEMA_VERSION
    )

    def finalize(
        self,
    ) -> "SelfHealingResult":
        self.sessions = [
            session.finalize()
            for session
            in self.sessions
        ]

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        states: Dict[
            str,
            int,
        ] = {}

        for session in self.sessions:
            states[
                session.state
            ] = (
                states.get(
                    session.state,
                    0,
                )
                + 1
            )

        return {
            "schema_version": (
                self.schema_version
            ),
            "generated_at": (
                self.generated_at
            ),
            "session_count": len(
                self.sessions
            ),
            "states": states,
            "sessions": [
                session.to_dict()
                for session
                in self.sessions
            ],
            "metadata": (
                json_safe(
                    self.metadata
                )
            ),
        }
