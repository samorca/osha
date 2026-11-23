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


REMEDIATION_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------
# Lifecycle enums
# ---------------------------------------------------------------

class RemediationMode(
    str,
    Enum,
):
    """
    Execution mode.

    DRY_RUN is intentionally the safest default.
    """

    DRY_RUN = "dry_run"

    EXECUTE = "execute"


class RemediationStatus(
    str,
    Enum,
):
    """
    Overall remediation lifecycle.
    """

    CREATED = "created"

    VALIDATING = "validating"

    BLOCKED = "blocked"

    READY = "ready"

    EXECUTING = "executing"

    EXECUTED = "executed"

    VERIFYING = "verifying"

    VERIFIED = "verified"

    VERIFICATION_FAILED = (
        "verification_failed"
    )

    ROLLING_BACK = (
        "rolling_back"
    )

    ROLLED_BACK = "rolled_back"

    FAILED = "failed"

    CANCELLED = "cancelled"

    COMPLETED = "completed"


class ExecutionStatus(
    str,
    Enum,
):
    """
    Result of an individual adapter execution.
    """

    NOT_STARTED = "not_started"

    DRY_RUN = "dry_run"

    RUNNING = "running"

    SUCCEEDED = "succeeded"

    FAILED = "failed"

    TIMED_OUT = "timed_out"

    CANCELLED = "cancelled"

    BLOCKED = "blocked"


class VerificationStatus(
    str,
    Enum,
):
    """
    Post-action recovery verification status.
    """

    NOT_RUN = "not_run"

    RUNNING = "running"

    PASSED = "passed"

    FAILED = "failed"

    INCONCLUSIVE = "inconclusive"


class RollbackStatus(
    str,
    Enum,
):
    """
    Rollback lifecycle.
    """

    NOT_REQUIRED = (
        "not_required"
    )

    AVAILABLE = "available"

    REQUIRED = "required"

    RUNNING = "running"

    SUCCEEDED = "succeeded"

    FAILED = "failed"

    UNAVAILABLE = "unavailable"


class GuardrailStatus(
    str,
    Enum,
):
    """
    Guardrail evaluation outcome.
    """

    PASSED = "passed"

    FAILED = "failed"

    WARNING = "warning"


class ValidationStatus(
    str,
    Enum,
):
    """
    Validation state before execution.
    """

    VALID = "valid"

    INVALID = "invalid"

    CONDITIONAL = "conditional"


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
# Audit evidence
# ---------------------------------------------------------------

@dataclass(frozen=True)
class RemediationEvidence:
    """
    Evidence attached to validation, execution,
    verification or rollback.

    Never place secrets or unrestricted raw
    command output here.
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
# Guardrails
# ---------------------------------------------------------------

@dataclass
class GuardrailCheck:
    """
    Result of one safety guardrail.

    Examples later may include:

      approved action
      target count
      protected environment
      execution mode
      rollback availability
      stale decision
    """

    guardrail_id: str

    name: str

    status: str

    description: str

    blocking: bool = True

    evidence: List[
        RemediationEvidence
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def passed(
        self,
    ) -> bool:
        return (
            self.status
            == GuardrailStatus.
            PASSED.value
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "guardrail_id": (
                self.guardrail_id
            ),
            "name": self.name,
            "status": self.status,
            "description": (
                self.description
            ),
            "blocking": (
                self.blocking
            ),
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
            "attributes": (
                json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Pre-execution validation result.

    The executor must not treat an INVALID result
    as executable.
    """

    status: str = (
        ValidationStatus.
        INVALID.value
    )

    valid: bool = False

    checks: List[
        GuardrailCheck
    ] = field(
        default_factory=list
    )

    blocking_reasons: List[
        str
    ] = field(
        default_factory=list
    )

    warnings: List[
        str
    ] = field(
        default_factory=list
    )

    validated_at: str = field(
        default_factory=utc_now
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "ValidationResult":
        self.blocking_reasons = list(
            dict.fromkeys(
                self.blocking_reasons
            )
        )

        self.warnings = list(
            dict.fromkeys(
                self.warnings
            )
        )

        blocking_failures = [
            check
            for check in self.checks
            if (
                check.blocking
                and not check.passed()
            )
        ]

        if blocking_failures:
            self.valid = False

            self.status = (
                ValidationStatus.
                INVALID.value
            )

        elif self.warnings:
            self.valid = True

            self.status = (
                ValidationStatus.
                CONDITIONAL.value
            )

        else:
            self.valid = True

            self.status = (
                ValidationStatus.
                VALID.value
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "status": self.status,
            "valid": self.valid,
            "checks": [
                check.to_dict()
                for check in self.checks
            ],
            "blocking_reasons": (
                self.blocking_reasons
            ),
            "warnings": (
                self.warnings
            ),
            "validated_at": (
                self.validated_at
            ),
            "attributes": (
                json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Remediation request
# ---------------------------------------------------------------

@dataclass
class RemediationRequest:
    """
    Controlled request created from an approved
    Decision ActionCandidate.

    Default mode is DRY_RUN.

    Creating this object does not execute
    anything.
    """

    action_id: str

    action_type: str

    environment_id: str

    mode: str = (
        RemediationMode.
        DRY_RUN.value
    )

    decision_id: Optional[
        str
    ] = None

    plan_id: Optional[
        str
    ] = None

    incident_id: Optional[
        str
    ] = None

    target_resource_ids: List[
        str
    ] = field(
        default_factory=list
    )

    target_service_ids: List[
        str
    ] = field(
        default_factory=list
    )

    parameters: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    requested_by: Optional[
        str
    ] = None

    approval_status: Optional[
        str
    ] = None

    approval_ids: List[
        str
    ] = field(
        default_factory=list
    )

    risk_level: Optional[
        str
    ] = None

    risk_score: float = 0.0

    reversibility: Optional[
        str
    ] = None

    rollback_action_type: Optional[
        str
    ] = None

    requested_at: str = field(
        default_factory=utc_now
    )

    expires_at: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    request_id: Optional[
        str
    ] = None

    def build_request_id(
        self,
    ) -> str:
        return (
            "remediation:request:"
            + stable_hash(
                {
                    "action_id": (
                        self.action_id
                    ),
                    "environment_id": (
                        self.environment_id
                    ),
                    "targets": (
                        sorted(
                            set(
                                self.
                                target_resource_ids
                            )
                        )
                    ),
                    "services": (
                        sorted(
                            set(
                                self.
                                target_service_ids
                            )
                        )
                    ),
                    "parameters": (
                        self.parameters
                    ),
                    "mode": self.mode,
                }
            )
        )

    def finalize(
        self,
    ) -> "RemediationRequest":
        self.target_resource_ids = (
            sorted(
                set(
                    self.
                    target_resource_ids
                )
            )
        )

        self.target_service_ids = (
            sorted(
                set(
                    self.
                    target_service_ids
                )
            )
        )

        self.approval_ids = sorted(
            set(
                self.approval_ids
            )
        )

        self.risk_score = (
            checked_score(
                self.risk_score
            )
        )

        if self.mode not in {
            item.value
            for item
            in RemediationMode
        }:
            raise ValueError(
                "Unsupported remediation "
                f"mode: {self.mode}"
            )

        if not self.request_id:
            self.request_id = (
                self.build_request_id()
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return json_safe(
            asdict(self)
        )


# -
# ---------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------

@dataclass
class ExecutionResult:
    """
    Result returned by a remediation adapter.

    stdout/stderr are deliberately represented as
    bounded optional strings. Future executors
    must redact and truncate them before storing.
    """

    status: str = (
        ExecutionStatus.
        NOT_STARTED.value
    )

    success: bool = False

    adapter_name: Optional[
        str
    ] = None

    operation: Optional[
        str
    ] = None

    started_at: onal[
        str
    ] = None

    finished_at: Optional[
        str
    ] = None

    duration_seconds: Optional[
        float
    ] = None

    exit_code: Optional[
        int
    ] = None

    stdout: Optional[
        str
    ] = None

    stderr: Optional[
        str
    ] = None

    changed: bool = False

    evidence: List[
        RemediationEvidence
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
    ) -> "ExecutionResult":
        if (
            self.duration_seconds
            is not None
        ):
            self.duration_seconds = max(
                0.0,
                float(
                    self.duration_seconds
                ),
            )

        self.success = (
            self.status
            in {
                ExecutionStatus.
                SUCCEEDED.value,
                ExecutionStatus.
                DRY_RUN.value,
            }
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
# Verification result
# ---------------------------------------------------------------

@dataclass
class VerificationResult:
    """
    Evidence that the action achieved its intended
    operational effect.

    Successful execution and successful recovery
    are deliberately separate concepts.
    """

    status: str = (
        VerificationStatus.
        NOT_RUN.value
    )

    passed: bool = False

    health_before: Optional[
        float
    ] = None

    health_after: Optional[
        float
    ] = None

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
        RemediationEvidence
    ] = field(
        default_factory=list
    )

    started_at: Optional[
        str
    ] = None

    finished_at: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "VerificationResult":
        if self.health_before is not None:
            self.health_before = (
                checked_score(
                    self.health_before
                )
            )

        if self.health_after is not None:
            self.health_after = (
                checked_score(
                    self.health_after
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

        self.passed = (
            self.status
            == VerificationStatus.
            PASSED.value
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
# Rollback result
# ---------------------------------------------------------------

@dataclass
class RollbackResult:
    """
    Result of a rollback attempt.

    Rollback is modeled separately from the
    original action so both outcomes are retained
    for audit and memory.
    """

    status: str = (
        RollbackStatus.
        NOT_REQUIRED.value
    )

    success: bool = False

    action_type: Optional[
        str
    ] = None

    adapter_name: Optional[
        str
    ] = None

    started_at: Optional[
        str
    ] = None

    finished_at: Optional[
        str
    ] = None

    reason: Optional[
        str
    ] = None

    evidence: List[
        RemediationEvidence
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
    ) -> "RollbackResult":
        self.success = (
            self.status
            == RollbackStatus.
            SUCCEEDED.value
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
# Audit event
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AuditEvent:
    """
    Immutable logical audit event.

    Persistent append-only semantics will be
    implemented later in audit.py.
    """

    event_type: str

    description: str

    remediation_id: Optional[
        str
    ] = None

    request_id: Optional[
        str
    ] = None

    action_id: Optional[
        str
    ] = None

    actor_id: Optional[
        str
    ] = None

    occurred_at: str = field(
        default_factory=utc_now
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return json_safe(
            asdict(self)
        )


# ---------------------------------------------------------------
# Complete remediation result
# ---------------------------------------------------------------

@dataclass
class RemediationResult:
    """
    Complete lifecycle record for one remediation
    request.
    """

    request: RemediationRequest

    status: str = (
        RemediationStatus.
        CREATED.value
    )

    validation: ValidationResult = field(
        default_factory=(
            ValidationResult
        )
    )

    execution: ExecutionResult = field(
        default_factory=(
            ExecutionResult
        )
    )

    verification: VerificationResult = field(
        default_factory=(
            VerificationResult
        )
    )

    rollback: RollbackResult = field(
        default_factory=(
            RollbackResult
        )
    )

    audit_events: List[
        AuditEvent
    ] = field(
        default_factory=list
    )

    evidence: List[
        RemediationEvidence
    ] = field(
        default_factory=list
    )

    started_at: Optional[
        str
    ] = None

    completed_at: Optional[
        str
    ] = None

    error: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    remediation_id: Optional[
        str
    ] = None

    schema_version: str = (
        REMEDIATION_SCHEMA_VERSION
    )

    def build_remediation_id(
        self,
    ) -> str:
        self.request.finalize()

        return (
            "remediation:"
            + stable_hash(
                {
                    "request_id": (
                        self.request.
                        request_id
                    ),
                    "action_id": (
                        self.request.
                        action_id
                    ),
                    "environment_id": (
                        self.request.
                        environment_id
                    ),
                }
            )
        )

    def finalize(
        self,
    ) -> "RemediationResult":
        self.request.finalize()

        self.validation.finalize()

        self.execution.finalize()

        self.verification.finalize()

        self.rollback.finalize()

        if not self.remediation_id:
            self.remediation_id = (
                self.build_remediation_id()
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
            "remediation_id": (
                self.remediation_id
            ),
            "status": self.status,
            "request": (
                self.request.to_dict()
            ),
            "validation": (
                self.validation.to_dict()
            ),
            "execution": (
                self.execution.to_dict()
            ),
            "verification": (
                self.verification.to_dict()
            ),
            "rollback": (
                self.rollback.to_dict()
            ),
            "audit_events": [
                event.to_dict()
                for event
                in self.audit_events
            ],
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
            "started_at": (
                self.started_at
            ),
            "completed_at": (
                self.completed_at
            ),
            "error": self.error,
            "attributes": (
                json_safe(
                    self.attributes
                )
            ),
        }

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


# ---------------------------------------------------------------
# Batch result
# ---------------------------------------------------------------

@dataclass
class RemediationBatchResult:
    """
    Output from processing multiple requests.
    """

    results: List[
        RemediationResult
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

    def finalize(
        self,
    ) -> "RemediationBatchResult":
        self.results = [
            result.finalize()
            for result
            in self.results
        ]

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        statuses: Dict[
            str,
            int,
        ] = {}

        for result in self.results:
            statuses[
                result.status
            ] = (
                statuses.get(
                    result.status,
                    0,
                )
                + 1
            )

        return {
            "schema_version": (
                REMEDIATION_SCHEMA_VERSION
            ),
            "generated_at": (
                self.generated_at
            ),
            "result_count": len(
                self.results
            ),
            "statuses": statuses,
            "results": [
                result.to_dict()
                for result
                in self.results
            ],
            "metadata": (
                json_safe(
                    self.metadata
                )
            ),
        }


# ---------------------------------------------------------------
# Utility checks
# ---------------------------------------------------------------

def is_dry_run(
    request: RemediationRequest,
) -> bool:
    return (
        request.mode
        == RemediationMode.
        DRY_RUN.value
    )


def execution_succeeded(
    result: RemediationResult,
) -> bool:
    return (
        result.execution.status
        == ExecutionStatus.
        SUCCEEDED.value
    )


def recovery_verified(
    result: RemediationResult,
) -> bool:
    return (
        result.verification.status
        == VerificationStatus.
        PASSED.value
    )


def rollback_succeeded(
    result: RemediationResult,
) -> bool:
    return (
        result.rollback.status
        == RollbackStatus.
        SUCCEEDED.value
    )
