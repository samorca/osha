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


ORCHESTRATOR_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------
# Orchestration lifecycle
# ---------------------------------------------------------------

class OrchestrationState(
    str,
    Enum,
):
    CREATED = "created"

    QUEUED = "queued"

    RUNNING = "running"

    WAITING = "waiting"

    PAUSED = "paused"

    COMPLETED = "completed"

    PARTIAL = "partial"

    FAILED = "failed"

    CANCELLED = "cancelled"

    ESCALATED = "escalated"


class StageName(
    str,
    Enum,
):
    """
    High-level OSHA processing stages.

    The orchestrator coordinates these layers.
    It does not reimplement them.
    """

    COLLECTION = "collection"

    TELEMETRY = "telemetry"

    CORRELATION = "correlation"

    INTELLIGENCE = "intelligence"

    MEMORY = "memory"

    DECISION = "decision"

    REMEDIATION = "remediation"

    SELF_HEALING = "self_healing"

    OUTCOME = "outcome"


class StageStatus(
    str,
    Enum,
):
    PENDING = "pending"

    RUNNING = "running"

    SUCCEEDED = "succeeded"

    SKIPPED = "skipped"

    BLOCKED = "blocked"

    FAILED = "failed"

    CANCELLED = "cancelled"


class TriggerType(
    str,
    Enum,
):
    MANUAL = "manual"

    TELEMETRY = "telemetry"

    INCIDENT = "incident"

    SCHEDULE = "schedule"

    API = "api"

    WORKER = "worker"

    INTERNAL = "internal"


class StopReason(
    str,
    Enum,
):
    NONE = "none"

    COMPLETED = "completed"

    BLOCKED = "blocked"

    APPROVAL_REQUIRED = (
        "approval_required"
    )

    ESCALATION_REQUIRED = (
        "escalation_required"
    )

    SAFETY_GUARDRAIL = (
        "safety_guardrail"
    )

    FAILURE = "failure"

    CANCELLED = "cancelled"


# ---------------------------------------------------------------
# Helpers
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
class OrchestrationEvidence:
    """
    Evidence attached to an orchestration run.

    The orchestrator should reference normalized
    upstream evidence rather than copying
    unrestricted logs or command output.
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
            "resource_ids": sorted(
                set(
                    self.resource_ids
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
# Stage result
# ---------------------------------------------------------------

@dataclass
class StageResult:
    """
    Result from one orchestration stage.

    StageResult holds coordination metadata only.
    """

    stage: str

    status: str = (
        StageStatus.PENDING.value
    )

    started_at: Optional[
        str
    ] = None

    completed_at: Optional[
        str
    ] = None

    input_count: int = 0

    output_count: int = 0

    output_ids: List[
        str
    ] = field(
        default_factory=list
    )

    error: Optional[
        str
    ] = None

    evidence: List[
        OrchestrationEvidence
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "StageResult":
        valid_stages = {
            item.value
            for item in StageName
        }

        valid_statuses = {
            item.value
            for item in StageStatus
        }

        if self.stage not in valid_stages:
            raise ValueError(
                "Unknown orchestration "
                f"stage: {self.stage}"
            )

        if (
            self.status
            not in valid_statuses
        ):
            raise ValueError(
                "Unknown stage status: "
                f"{self.status}"
            )

        self.input_count = max(
            0,
            int(
                self.input_count
            ),
        )

        self.output_count = max(
            0,
            int(
                self.output_count
            ),
        )

        self.output_ids = sorted(
            set(
                self.output_ids
            )
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "stage": self.stage,
            "status": self.status,
            "started_at": (
                self.started_at
            ),
            "completed_at": (
                self.completed_at
            ),
            "input_count": (
                self.input_count
            ),
            "output_count": (
                self.output_count
            ),
            "output_ids": (
                self.output_ids
            ),
            "error": self.error,
            "evidence": [
                item.to_dict()
                for item
                in self.evidence
            ],
            "attributes": (
                json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Orchestration context
# ---------------------------------------------------------------

@dataclass
class OrchestrationContext:
    """
    Scope and provenance for one orchestration
    run.
    """

    environment_id: str

    trigger_type: str = (
        TriggerType.INTERNAL.value
    )

    incident_id: Optional[
        str
    ] = None

    correlation_id: Optional[
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

    requested_by: Optional[
        str
    ] = None

    dry_run: bool = True

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=utc_now
    )

    context_id: Optional[
        str
    ] = None

    def build_context_id(
        self,
    ) -> str:
        return (
            "orchestrator:context:"
            + stable_hash(
                {
                    "environment_id": (
                        self.environment_id
                    ),
                    "trigger_type": (
                        self.trigger_type
                    ),
                    "incident_id": (
                        self.incident_id
                    ),
                    "correlation_id": (
                        self.correlation_id
                    ),
                    "resource_ids": (
                        sorted(
                            set(
                                self.
                                resource_ids
                            )
                        )
                    ),
                    "service_ids": (
                        sorted(
                            set(
                                self.
                                service_ids
                            )
                        )
                    ),
                    "dry_run": (
                        self.dry_run
                    ),
                }
            )
        )

    def finalize(
        self,
    ) -> "OrchestrationContext":
        self.environment_id = (
            self.environment_id.strip()
        )

        if not self.environment_id:
            raise ValueError(
                "environment_id is required"
            )

        if (
            self.trigger_type
            not in {
                item.value
                for item
                in TriggerType
            }
        ):
            raise ValueError(
                "Unknown trigger type: "
                f"{self.trigger_type}"
            )

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

        if not self.context_id:
            self.context_id = (
                self.build_context_id()
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
# Orchestration run
# ---------------------------------------------------------------

@dataclass
class OrchestrationRun:
    """
    Complete coordination state for one OSHA
    orchestration run.

    It points to outputs from the real modules
    rather than embedding their complete internal
    data.
    """

    context: OrchestrationContext

    state: str = (
        OrchestrationState.
        CREATED.value
    )

    current_stage: Optional[
        str
    ] = None

    stages: List[
        StageResult
    ] = field(
        default_factory=list
    )

    decision_id: Optional[
        str
    ] = None

    plan_id: Optional[
        str
    ] = None

    remediation_ids: List[
        str
    ] = field(
        default_factory=list
    )

    healing_session_ids: List[
        str
    ] = field(
        default_factory=list
    )

    outcome_ids: List[
        str
    ] = field(
        default_factory=list
    )

    stop_reason: str = (
        StopReason.NONE.value
    )

    error: Optional[
        str
    ] = None

    evidence: List[
        OrchestrationEvidence
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

    started_at: Optional[
        str
    ] = None

    completed_at: Optional[
        str
    ] = None

    updated_at: str = field(
        default_factory=utc_now
    )

    run_id: Optional[
        str
    ] = None

    schema_version: str = (
        ORCHESTRATOR_SCHEMA_VERSION
    )

    def build_run_id(
        self,
    ) -> str:
        self.context.finalize()

        return (
            "orchestrator:run:"
            + stable_hash(
                {
                    "context_id": (
                        self.context.
                        context_id
                    ),
                    "created_at": (
                        self.created_at
                    ),
                }
            )
        )

    def stage(
        self,
        stage_name: str,
    ) -> Optional[
        StageResult
    ]:
        for result in self.stages:
            if (
                result.stage
                == stage_name
            ):
                return result

        return None

    def finalize(
        self,
    ) -> "OrchestrationRun":
        self.context.finalize()

        if (
            self.state
            not in {
                item.value
                for item
                in OrchestrationState
            }
        ):
            raise ValueError(
                "Unknown orchestration "
                f"state: {self.state}"
            )

        if (
            self.current_stage
            is not None
            and self.current_stage
            not in {
                item.value
                for item
                in StageName
            }
        ):
            raise ValueError(
                "Unknown current stage: "
                f"{self.current_stage}"
            )

        if (
            self.stop_reason
            not in {
                item.value
                for item
                in StopReason
            }
        ):
            raise ValueError(
                "Unknown stop reason: "
                f"{self.stop_reason}"
            )

        seen = set()

        normalized = []

        for stage_result in self.stages:
            stage_result.finalize()

            if (
                stage_result.stage
                in seen
            ):
                raise ValueError(
                    "Duplicate stage result: "
                    f"{stage_result.stage}"
                )

            seen.add(
                stage_result.stage
            )

            normalized.append(
                stage_result
            )

        self.stages = normalized

        self.remediation_ids = sorted(
            set(
                self.remediation_ids
            )
        )

        self.healing_session_ids = sorted(
            set(
                self.healing_session_ids
            )
        )

        self.outcome_ids = sorted(
            set(
                self.outcome_ids
            )
        )

        if not self.run_id:
            self.run_id = (
                self.build_run_id()
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
                self.schema_version
            ),
            "run_id": (
                self.run_id
            ),
            "state": self.state,
            "current_stage": (
                self.current_stage
            ),
            "context": (
                self.context.to_dict()
            ),
            "stages": [
                item.to_dict()
                for item in self.stages
            ],
            "decision_id": (
                self.decision_id
            ),
            "plan_id": (
                self.plan_id
            ),
            "remediation_ids": (
                self.remediation_ids
            ),
            "healing_session_ids": (
                self.
                healing_session_ids
            ),
            "outcome_ids": (
                self.outcome_ids
            ),
            "stop_reason": (
                self.stop_reason
            ),
            "error": self.error,
            "evidence": [
                item.to_dict()
                for item
                in self.evidence
            ],
            "attributes": (
                json_safe(
                    self.attributes
                )
            ),
            "created_at": (
                self.created_at
            ),
            "started_at": (
                self.started_at
            ),
            "completed_at": (
                self.completed_at
            ),
            "updated_at": (
                self.updated_at
            ),
        }


# ---------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------

@dataclass
class OrchestrationResult:
    """
    Result across one or more independent OSHA
    orchestration runs.
    """

    runs: List[
        OrchestrationRun
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
        ORCHESTRATOR_SCHEMA_VERSION
    )

    def finalize(
        self,
    ) -> "OrchestrationResult":
        self.runs = [
            run.finalize()
            for run in self.runs
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

        stops: Dict[
            str,
            int,
        ] = {}

        for run in self.runs:
            states[
                run.state
            ] = (
                states.get(
                    run.state,
                    0,
                )
                + 1
            )

            stops[
                run.stop_reason
            ] = (
                stops.get(
                    run.stop_reason,
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
            "run_count": len(
                self.runs
            ),
            "states": states,
            "stop_reasons": stops,
            "runs": [
                run.to_dict()
                for run in self.runs
            ],
            "metadata": (
                json_safe(
                    self.metadata
                )
            ),
        }
