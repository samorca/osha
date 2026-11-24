from __future__ import annotations

"""
Production remediation service boundary for OSHA.

Architecture:

    Incident
       ↓
    DecisionService
       ↓
    DecisionRecord
       ↓
    RemediationService
       ├── validate decision
       ├── create versioned plan
       ├── require approval
       ├── enforce idempotency
       ├── execute dry run
       ├── authorize live execution
       ├── record step results
       └── attempt rollback
               ↓
        RemediationExecutor
               ↓
        provider-specific implementation

Safety boundary:

    Plan creation does not execute infrastructure.

    Approval does not execute infrastructure.

    Execution requires:
        remediation execution feature enabled
        confirm=true
        expected plan version match
        valid non-expired decision
        allowed or explicitly approved decision
        approved remediation plan
        idempotency key
        dry-run unless live execution is separately enabled

The default executor performs dry-run validation only.
It never mutates infrastructure.
"""

import asyncio
import inspect
import logging

from copy import deepcopy

from dataclasses import (
    dataclass,
    field,
    replace,
)

from datetime import (
    datetime,
    timezone,
)

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from uuid import uuid4

from src.api.config import (
    APIConfig,
)

from src.api.errors import (
    APIException,
    DecisionDeniedAPIError,
    DecisionNotFoundAPIError,
    FeatureDisabledError,
    InternalServerAPIError,
    RemediationConfirmationRequiredAPIError,
    RemediationExecutionAPIError,
    RemediationExecutionDisabledAPIError,
    RemediationNotApprovedAPIError,
    RemediationNotFoundAPIError,
    RemediationVersionConflictAPIError,
    ResourceConflictError,
)

from src.api.schema import (
    APIError,
    DecisionStatusAPI,
    RemediationApprovalRequest,
    RemediationExecutionRequest,
    RemediationPlanCreateRequest,
    RemediationPlanResponse,
    RemediationPlanStatusAPI,
    RemediationRiskAPI,
    RemediationStepRequest,
    RemediationStepResponse,
)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Service identity
# ------------------------------------------------------------------

REMEDIATION_SERVICE_NAME = "remediation"

REMEDIATION_SERVICE_VERSION = "1"


# ------------------------------------------------------------------
# Limits
# ------------------------------------------------------------------

MAX_REMEDIATION_ATTRIBUTES = 256

MAX_EXECUTION_RESULTS = 10_000

MAX_IDEMPOTENCY_RECORDS = 10_000


# ------------------------------------------------------------------
# Time
# ------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_datetime(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


# ------------------------------------------------------------------
# Safe mappings
# ------------------------------------------------------------------

_SENSITIVE_KEY_FRAGMENTS = {
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "authorization",
    "cookie",
    "private",
    "certificate",
    "connection_string",
    "traceback",
    "stack",
}


def safe_mapping(
    value: Optional[
        Mapping[
            str,
            Any,
        ]
    ],
    *,
    limit: int = MAX_REMEDIATION_ATTRIBUTES,
) -> Dict[str, Any]:
    """
    Return bounded JSON-like metadata without
    exposing obvious credentials or internal
    traceback information.
    """

    if not value:
        return {}

    output: Dict[str, Any] = {}

    for raw_key, item in list(value.items())[:limit]:
        key = str(raw_key).strip()

        if not key:
            continue

        normalized = key.lower()

        if any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS):
            continue

        if isinstance(
            item,
            (
                str,
                int,
                float,
                bool,
                type(None),
            ),
        ):
            output[key] = item
            continue

        if isinstance(
            item,
            (
                list,
                tuple,
            ),
        ):
            output[key] = [
                child
                for child in item[:128]
                if isinstance(
                    child,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    ),
                )
            ]

            continue

        if isinstance(
            item,
            Mapping,
        ):
            output[key] = safe_mapping(
                item,
                limit=64,
            )

            continue

        output[key] = {
            "type": type(item).__name__,
        }

    return output


# ------------------------------------------------------------------
# Async compatibility
# ------------------------------------------------------------------


async def await_if_needed(
    value: Any,
) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


# ------------------------------------------------------------------
# Operation context
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RemediationOperationContext:
    request_id: Optional[str] = None

    actor_id: Optional[str] = None

    tenant_id: Optional[str] = None

    correlation_id: Optional[str] = None

    approval_identity: Optional[str] = None

    def safe_attributes(
        self,
    ) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "actor_id": self.actor_id,
            "tenant_id": self.tenant_id,
            "correlation_id": self.correlation_id,
            "approval_identity": self.approval_identity,
        }


# ------------------------------------------------------------------
# Internal step record
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RemediationStepRecord:
    step_id: str

    action: str

    provider: str

    target: Mapping[
        str,
        Any,
    ]

    parameters: Mapping[
        str,
        Any,
    ]

    timeout_seconds: int

    reversible: bool

    rollback_action: Optional[str]

    state: str = "pending"

    started_at: Optional[datetime] = None

    completed_at: Optional[datetime] = None

    error_code: Optional[str] = None

    error_message: Optional[str] = None

    attributes: Mapping[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        if not self.step_id.strip():
            raise ValueError("step_id cannot be empty")

        if not self.action.strip():
            raise ValueError("step action cannot be empty")

        if not self.provider.strip():
            raise ValueError("step provider cannot be empty")

        if self.timeout_seconds < 1:
            raise ValueError("step timeout must be >= 1")

        if self.reversible and not self.rollback_action:
            raise ValueError("reversible step requires rollback_action")


# ------------------------------------------------------------------
# Internal remediation plan
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RemediationPlanRecord:
    plan_id: str

    incident_id: str

    decision_id: str

    title: str

    description: Optional[str]

    risk: str

    status: str

    dry_run: bool

    approval_required: bool

    steps: Tuple[
        RemediationStepRecord,
        ...,
    ]

    created_at: datetime

    updated_at: datetime

    version: int = 1

    tenant_id: Optional[str] = None

    created_by: Optional[str] = None

    correlation_id: Optional[str] = None

    idempotency_key: Optional[str] = None

    approved_by: Optional[str] = None

    approved_at: Optional[datetime] = None

    rejection_reason: Optional[str] = None

    execution_id: Optional[str] = None

    execution_idempotency_key: Optional[str] = None

    attributes: Mapping[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        if not self.plan_id.strip():
            raise ValueError("plan_id cannot be empty")

        if not self.incident_id.strip():
            raise ValueError("incident_id cannot be empty")

        if not self.decision_id.strip():
            raise ValueError("decision_id cannot be empty")

        if not self.title.strip():
            raise ValueError("plan title cannot be empty")

        if self.risk not in {value.value for value in RemediationRiskAPI}:
            raise ValueError(f"Unsupported remediation risk: {self.risk}")

        if self.status not in {value.value for value in RemediationPlanStatusAPI}:
            raise ValueError(f"Unsupported remediation status: {self.status}")

        if self.version < 1:
            raise ValueError("plan version must be >= 1")

        if not self.steps:
            raise ValueError("remediation plan requires at least one step")

        step_ids = [step.step_id for step in self.steps]

        if len(step_ids) != len(set(step_ids)):
            raise ValueError("remediation step IDs must be unique")

        for step in self.steps:
            step.validate()

        if (
            self.status == RemediationPlanStatusAPI.APPROVED.value
            and self.approved_at is None
        ):
            raise ValueError("approved plan requires approved_at")


# ------------------------------------------------------------------
# Repository protocol
# ------------------------------------------------------------------


@runtime_checkable
class RemediationRepository(
    Protocol,
):
    async def create(
        self,
        plan: RemediationPlanRecord,
    ) -> RemediationPlanRecord: ...

    async def get(
        self,
        plan_id: str,
    ) -> Optional[RemediationPlanRecord]: ...

    async def list(
        self,
        *,
        incident_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        status: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[
        Tuple[
            RemediationPlanRecord,
            ...,
        ],
        int,
    ]: ...

    async def update(
        self,
        plan: RemediationPlanRecord,
        *,
        expected_version: int,
    ) -> RemediationPlanRecord: ...

    async def find_by_idempotency_key(
        self,
        key: str,
    ) -> Optional[RemediationPlanRecord]: ...

    async def find_by_execution_key(
        self,
        key: str,
    ) -> Optional[RemediationPlanRecord]: ...

    async def close(
        self,
    ) -> None: ...


# ------------------------------------------------------------------
# Process-local repository
# ------------------------------------------------------------------


class ProcessLocalRemediationRepository:
    """
    Process-local development repository.

    It is not durable and is not shared between
    API workers.
    """

    def __init__(
        self,
    ) -> None:
        self._records: Dict[
            str,
            RemediationPlanRecord,
        ] = {}

        self._creation_idempotency: Dict[
            str,
            str,
        ] = {}

        self._execution_idempotency: Dict[
            str,
            str,
        ] = {}

        self._lock = asyncio.Lock()

        self._closed = False

    def _ensure_open(
        self,
    ) -> None:
        if self._closed:
            raise RuntimeError("Remediation repository is closed.")

    async def create(
        self,
        plan: RemediationPlanRecord,
    ) -> RemediationPlanRecord:
        plan.validate()

        async with self._lock:
            self._ensure_open()

            if plan.plan_id in self._records:
                raise ResourceConflictError(
                    "A remediation plan with this ID already exists.",
                    details={
                        "plan_id": plan.plan_id,
                    },
                )

            if plan.idempotency_key:
                existing_id = self._creation_idempotency.get(plan.idempotency_key)

                if existing_id:
                    return deepcopy(self._records[existing_id])

            self._records[plan.plan_id] = deepcopy(plan)

            if plan.idempotency_key:
                self._bounded_index_insert(
                    self._creation_idempotency,
                    plan.idempotency_key,
                    plan.plan_id,
                )

            return deepcopy(plan)

    async def get(
        self,
        plan_id: str,
    ) -> Optional[RemediationPlanRecord]:
        normalized = str(plan_id).strip()

        async with self._lock:
            self._ensure_open()

            plan = self._records.get(normalized)

            return deepcopy(plan) if plan is not None else None

    async def list(
        self,
        *,
        incident_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        status: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[
        Tuple[
            RemediationPlanRecord,
            ...,
        ],
        int,
    ]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")

        if offset < 0:
            raise ValueError("offset must be >= 0")

        async with self._lock:
            self._ensure_open()

            output = []

            for plan in self._records.values():
                if incident_id is not None and plan.incident_id != incident_id:
                    continue

                if decision_id is not None and plan.decision_id != decision_id:
                    continue

                if status is not None and plan.status != status:
                    continue

                if tenant_id is not None and plan.tenant_id != tenant_id:
                    continue

                output.append(plan)

            output.sort(
                key=lambda value: (
                    value.created_at,
                    value.plan_id,
                ),
                reverse=True,
            )

            total = len(output)

            selected = output[offset : offset + limit]

            return (
                tuple(deepcopy(selected)),
                total,
            )

    async def update(
        self,
        plan: RemediationPlanRecord,
        *,
        expected_version: int,
    ) -> RemediationPlanRecord:
        plan.validate()

        async with self._lock:
            self._ensure_open()

            current = self._records.get(plan.plan_id)

            if current is None:
                raise RemediationNotFoundAPIError(
                    details={
                        "plan_id": plan.plan_id,
                    },
                )

            if current.version != expected_version:
                raise RemediationVersionConflictAPIError(
                    details={
                        "plan_id": plan.plan_id,
                        "expected_version": (expected_version),
                        "actual_version": (current.version),
                    },
                )

            self._records[plan.plan_id] = deepcopy(plan)

            if plan.execution_idempotency_key:
                self._bounded_index_insert(
                    self._execution_idempotency,
                    plan.execution_idempotency_key,
                    plan.plan_id,
                )

            return deepcopy(plan)

    async def find_by_idempotency_key(
        self,
        key: str,
    ) -> Optional[RemediationPlanRecord]:
        normalized = str(key).strip()

        if not normalized:
            return None

        async with self._lock:
            self._ensure_open()

            plan_id = self._creation_idempotency.get(normalized)

            if plan_id is None:
                return None

            return deepcopy(self._records.get(plan_id))

    async def find_by_execution_key(
        self,
        key: str,
    ) -> Optional[RemediationPlanRecord]:
        normalized = str(key).strip()

        if not normalized:
            return None

        async with self._lock:
            self._ensure_open()

            plan_id = self._execution_idempotency.get(normalized)

            if plan_id is None:
                return None

            return deepcopy(self._records.get(plan_id))

    async def close(
        self,
    ) -> None:
        async with self._lock:
            self._closed = True

            self._records.clear()
            self._creation_idempotency.clear()
            self._execution_idempotency.clear()

    @staticmethod
    def _bounded_index_insert(
        index: Dict[str, str],
        key: str,
        value: str,
    ) -> None:
        if len(index) >= MAX_IDEMPOTENCY_RECORDS:
            oldest = next(iter(index))

            index.pop(
                oldest,
                None,
            )

        index[key] = value


# ------------------------------------------------------------------
# Execution result
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RemediationExecutionResult:
    success: bool

    changed: bool = False

    message: Optional[str] = None

    attributes: Mapping[
        str,
        Any,
    ] = field(default_factory=dict)


# ------------------------------------------------------------------
# Executor protocol
# ------------------------------------------------------------------


@runtime_checkable
class RemediationExecutor(
    Protocol,
):
    async def validate_step(
        self,
        step: RemediationStepRecord,
        *,
        context: Mapping[
            str,
            Any,
        ],
    ) -> RemediationExecutionResult: ...

    async def execute_step(
        self,
        step: RemediationStepRecord,
        *,
        context: Mapping[
            str,
            Any,
        ],
    ) -> RemediationExecutionResult: ...

    async def rollback_step(
        self,
        step: RemediationStepRecord,
        *,
        context: Mapping[
            str,
            Any,
        ],
    ) -> RemediationExecutionResult: ...

    async def close(
        self,
    ) -> None: ...


# ------------------------------------------------------------------
# Safe default executor
# ------------------------------------------------------------------


class DryRunRemediationExecutor:
    """
    Validation-only executor.

    It never mutates external infrastructure.

    A real live executor must be injected
    explicitly and protected by provider policy,
    authentication, authorization, audit, and
    runtime feature gates.
    """

    async def validate_step(
        self,
        step: RemediationStepRecord,
        *,
        context: Mapping[
            str,
            Any,
        ],
    ) -> RemediationExecutionResult:
        step.validate()

        return RemediationExecutionResult(
            success=True,
            changed=False,
            message=("Remediation step passed dry-run validation."),
            attributes={
                "dry_run": True,
                "provider": step.provider,
                "action": step.action,
            },
        )

    async def execute_step(
        self,
        step: RemediationStepRecord,
        *,
        context: Mapping[
            str,
            Any,
        ],
    ) -> RemediationExecutionResult:
        raise RemediationExecutionDisabledAPIError(
            "The default remediation executor does not support live execution.",
            details={
                "step_id": step.step_id,
                "provider": step.provider,
            },
        )

    async def rollback_step(
        self,
        step: RemediationStepRecord,
        *,
        context: Mapping[
            str,
            Any,
        ],
    ) -> RemediationExecutionResult:
        return RemediationExecutionResult(
            success=True,
            changed=False,
            message=("No live change was made; rollback was not required."),
            attributes={
                "dry_run": True,
            },
        )

    async def close(
        self,
    ) -> None:
        return None


# ------------------------------------------------------------------
# Conversion helpers
# ------------------------------------------------------------------


def step_record_from_request(
    request: RemediationStepRequest,
) -> RemediationStepRecord:
    record = RemediationStepRecord(
        step_id=request.step_id,
        action=request.action,
        provider=request.provider,
        target=safe_mapping(request.target),
        parameters=safe_mapping(request.parameters),
        timeout_seconds=(request.timeout_seconds),
        reversible=request.reversible,
        rollback_action=(request.rollback_action),
        state="pending",
        attributes={},
    )

    record.validate()

    return record


def api_error_from_step(
    step: RemediationStepRecord,
) -> Optional[APIError]:
    if not step.error_code:
        return None

    return APIError(
        code=step.error_code,
        message=(step.error_message or "Remediation step failed."),
        retryable=False,
        details={
            "step_id": step.step_id,
            "provider": step.provider,
        },
    )


def step_response(
    step: RemediationStepRecord,
) -> RemediationStepResponse:
    step.validate()

    return RemediationStepResponse(
        step_id=step.step_id,
        action=step.action,
        provider=step.provider,
        state=step.state,
        reversible=step.reversible,
        started_at=step.started_at,
        completed_at=step.completed_at,
        error=api_error_from_step(step),
        attributes=safe_mapping(step.attributes),
    )


def plan_response(
    plan: RemediationPlanRecord,
) -> RemediationPlanResponse:
    plan.validate()

    attributes = safe_mapping(plan.attributes)

    attributes.update(
        {
            "execution_id": (plan.execution_id),
            "correlation_id": (plan.correlation_id),
        }
    )

    return RemediationPlanResponse(
        plan_id=plan.plan_id,
        version=plan.version,
        incident_id=(plan.incident_id),
        decision_id=(plan.decision_id),
        title=plan.title,
        description=(plan.description),
        risk=plan.risk,
        status=plan.status,
        dry_run=plan.dry_run,
        approval_required=(plan.approval_required),
        approved_by=(plan.approved_by),
        approved_at=(plan.approved_at),
        steps=[step_response(step) for step in plan.steps],
        created_at=(plan.created_at),
        updated_at=(plan.updated_at),
        attributes=attributes,
    )


# ------------------------------------------------------------------
# Decision helpers
# ------------------------------------------------------------------


def _decision_value(
    decision: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(
        decision,
        Mapping,
    ):
        return decision.get(
            name,
            default,
        )

    return getattr(
        decision,
        name,
        default,
    )


async def resolve_decision_record(
    decision_service: Any,
    decision_id: str,
    *,
    tenant_id: Optional[str] = None,
) -> Any:
    getter = getattr(
        decision_service,
        "get_record",
        None,
    )

    if not callable(getter):
        raise InternalServerAPIError(
            "Decision service does not support internal record lookup.",
            details={
                "resource": ("decision_service"),
            },
        )

    try:
        return await await_if_needed(
            getter(
                decision_id,
                tenant_id=tenant_id,
            )
        )

    except DecisionNotFoundAPIError:
        raise


def validate_decision_for_plan(
    decision: Any,
    request: RemediationPlanCreateRequest,
) -> Tuple[str, bool]:
    """
    Validate that the referenced decision permits
    remediation planning.

    Returns:
        risk
        approval_required
    """

    if bool(
        _decision_value(
            decision,
            "expired",
            False,
        )
    ):
        raise DecisionDeniedAPIError(
            "The decision has expired.",
            details={
                "decision_id": (request.decision_id),
            },
        )

    decision_incident_id = str(
        _decision_value(
            decision,
            "incident_id",
            "",
        )
    )

    if decision_incident_id != request.incident_id:
        raise ResourceConflictError(
            "The decision does not belong to the requested incident.",
            details={
                "incident_id": (request.incident_id),
                "decision_id": (request.decision_id),
            },
        )

    status = str(
        _decision_value(
            decision,
            "status",
            "",
        )
    )

    if status == DecisionStatusAPI.DENIED.value:
        raise DecisionDeniedAPIError(
            details={
                "decision_id": (request.decision_id),
            },
        )

    if status not in {
        DecisionStatusAPI.ALLOWED.value,
        DecisionStatusAPI.REQUIRES_APPROVAL.value,
    }:
        raise ResourceConflictError(
            "The decision is not in a state that permits remediation planning.",
            details={
                "decision_id": (request.decision_id),
                "state": status,
            },
        )

    decision_risk = str(
        _decision_value(
            decision,
            "risk",
            RemediationRiskAPI.HIGH.value,
        )
    )

    approval_required = bool(
        _decision_value(
            decision,
            "approval_required",
            True,
        )
    )

    if status == DecisionStatusAPI.REQUIRES_APPROVAL.value:
        approval_required = True

    return (
        decision_risk,
        approval_required,
    )


# ------------------------------------------------------------------
# Status transition validation
# ------------------------------------------------------------------

_ALLOWED_PLAN_TRANSITIONS: Dict[
    str,
    Tuple[str, ...],
] = {
    RemediationPlanStatusAPI.DRAFT.value: (
        RemediationPlanStatusAPI.PROPOSED.value,
        RemediationPlanStatusAPI.PENDING_APPROVAL.value,
        RemediationPlanStatusAPI.CANCELLED.value,
    ),
    RemediationPlanStatusAPI.PROPOSED.value: (
        RemediationPlanStatusAPI.PENDING_APPROVAL.value,
        RemediationPlanStatusAPI.APPROVED.value,
        RemediationPlanStatusAPI.REJECTED.value,
        RemediationPlanStatusAPI.CANCELLED.value,
    ),
    RemediationPlanStatusAPI.PENDING_APPROVAL.value: (
        RemediationPlanStatusAPI.APPROVED.value,
        RemediationPlanStatusAPI.REJECTED.value,
        RemediationPlanStatusAPI.CANCELLED.value,
    ),
    RemediationPlanStatusAPI.APPROVED.value: (
        RemediationPlanStatusAPI.EXECUTING.value,
        RemediationPlanStatusAPI.CANCELLED.value,
    ),
    RemediationPlanStatusAPI.EXECUTING.value: (
        RemediationPlanStatusAPI.SUCCEEDED.value,
        RemediationPlanStatusAPI.FAILED.value,
        RemediationPlanStatusAPI.ROLLING_BACK.value,
    ),
    RemediationPlanStatusAPI.ROLLING_BACK.value: (
        RemediationPlanStatusAPI.ROLLED_BACK.value,
        RemediationPlanStatusAPI.FAILED.value,
    ),
    RemediationPlanStatusAPI.SUCCEEDED.value: (),
    RemediationPlanStatusAPI.FAILED.value: (
        RemediationPlanStatusAPI.ROLLING_BACK.value,
    ),
    RemediationPlanStatusAPI.ROLLED_BACK.value: (),
    RemediationPlanStatusAPI.REJECTED.value: (),
    RemediationPlanStatusAPI.CANCELLED.value: (),
}


def validate_plan_transition(
    current: str,
    target: str,
) -> None:
    if current == target:
        return

    allowed = _ALLOWED_PLAN_TRANSITIONS.get(
        current,
        (),
    )

    if target not in allowed:
        raise ResourceConflictError(
            "The remediation plan status transition is not allowed.",
            details={
                "resource": ("remediation_plan"),
                "state": current,
                "status": target,
            },
        )


# ------------------------------------------------------------------
# Remediation service
# ------------------------------------------------------------------


class RemediationService:
    """
    Process-scoped guarded remediation service.
    """

    def __init__(
        self,
        *,
        config: APIConfig,
        repository: RemediationRepository,
        executor: RemediationExecutor,
        decision_service: Any,
    ) -> None:
        config.validate()

        self.config = config
        self.repository = repository
        self.executor = executor

        self.decision_service = decision_service

        self._closed = False

        self._lifecycle_lock = asyncio.Lock()

        self._plan_locks: Dict[
            str,
            asyncio.Lock,
        ] = {}

        self._plan_locks_guard = asyncio.Lock()

    # ------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------

    def _ensure_open(
        self,
    ) -> None:
        if self._closed:
            raise InternalServerAPIError(
                "The remediation service is closed.",
                details={
                    "resource": (REMEDIATION_SERVICE_NAME),
                },
            )

    def _require_read_enabled(
        self,
    ) -> None:
        if not (self.config.features.remediation_read_enabled):
            raise FeatureDisabledError(
                details={
                    "resource": ("remediation_read"),
                },
            )

    def _require_plan_enabled(
        self,
    ) -> None:
        if not (self.config.features.remediation_plan_creation_enabled):
            raise FeatureDisabledError(
                details={
                    "resource": ("remediation_plan_creation"),
                },
            )

    def _require_approval_enabled(
        self,
    ) -> None:
        if not (self.config.features.remediation_approval_enabled):
            raise FeatureDisabledError(
                details={
                    "resource": ("remediation_approval"),
                },
            )

    def _require_execution_enabled(
        self,
    ) -> None:
        if not (self.config.features.remediation_execution_enabled):
            raise RemediationExecutionDisabledAPIError()

    async def close(
        self,
    ) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return

            try:
                await await_if_needed(self.executor.close())

            finally:
                await await_if_needed(self.repository.close())

                self._closed = True

                async with self._plan_locks_guard:
                    self._plan_locks.clear()

    # ------------------------------------------------------
    # Locks
    # ------------------------------------------------------

    async def _plan_lock(
        self,
        plan_id: str,
    ) -> asyncio.Lock:
        async with self._plan_locks_guard:
            lock = self._plan_locks.get(plan_id)

            if lock is None:
                lock = asyncio.Lock()

                self._plan_locks[plan_id] = lock

            return lock

    # ------------------------------------------------------
    # Internal lookup
    # ------------------------------------------------------

    async def _get_record(
        self,
        plan_id: str,
        *,
        tenant_id: Optional[str] = None,
    ) -> RemediationPlanRecord:
        normalized = str(plan_id).strip()

        if not normalized:
            raise RemediationNotFoundAPIError(
                details={
                    "resource": ("remediation_plan"),
                },
            )

        plan = await await_if_needed(self.repository.get(normalized))

        if plan is None:
            raise RemediationNotFoundAPIError(
                details={
                    "plan_id": normalized,
                },
            )

        if tenant_id is not None and plan.tenant_id != tenant_id:
            raise RemediationNotFoundAPIError(
                details={
                    "plan_id": normalized,
                },
            )

        return plan

    # ------------------------------------------------------
    # Create plan
    # ------------------------------------------------------

    async def create_plan(
        self,
        request: RemediationPlanCreateRequest,
        *,
        context: Optional[RemediationOperationContext] = None,
    ) -> RemediationPlanResponse:
        self._ensure_open()
        self._require_plan_enabled()

        existing = await await_if_needed(
            self.repository.find_by_idempotency_key(request.idempotency_key)
        )

        if existing is not None:
            if (
                context is not None
                and context.tenant_id is not None
                and existing.tenant_id != context.tenant_id
            ):
                raise RemediationNotFoundAPIError()

            return plan_response(existing)

        decision = await resolve_decision_record(
            self.decision_service,
            request.decision_id,
            tenant_id=(context.tenant_id if context is not None else None),
        )

        (
            decision_risk,
            approval_required,
        ) = validate_decision_for_plan(
            decision,
            request,
        )

        risk = request.risk

        risk_order = {
            RemediationRiskAPI.LOW.value: 0,
            RemediationRiskAPI.MEDIUM.value: 1,
            RemediationRiskAPI.HIGH.value: 2,
            RemediationRiskAPI.CRITICAL.value: 3,
        }

        # Never lower the risk supplied by the
        # decision evaluation.
        if risk_order.get(
            decision_risk,
            2,
        ) > risk_order.get(
            risk,
            2,
        ):
            risk = decision_risk

        if risk in {
            RemediationRiskAPI.HIGH.value,
            RemediationRiskAPI.CRITICAL.value,
        }:
            approval_required = True

        now = utc_now()

        status = (
            RemediationPlanStatusAPI.PENDING_APPROVAL.value
            if approval_required
            else RemediationPlanStatusAPI.APPROVED.value
        )

        approved_at = None if approval_required else now

        approved_by = None if approval_required else "policy"

        plan = RemediationPlanRecord(
            plan_id=("rem-" + uuid4().hex),
            incident_id=(request.incident_id),
            decision_id=(request.decision_id),
            title=request.title,
            description=(request.description),
            risk=risk,
            status=status,
            dry_run=(request.dry_run),
            approval_required=(approval_required),
            steps=tuple(step_record_from_request(step) for step in request.steps),
            created_at=now,
            updated_at=now,
            version=1,
            tenant_id=(context.tenant_id if context is not None else None),
            created_by=(context.actor_id if context is not None else None),
            correlation_id=(context.correlation_id if context is not None else None),
            idempotency_key=(request.idempotency_key),
            approved_by=approved_by,
            approved_at=approved_at,
            attributes={
                "decision_status": (
                    _decision_value(
                        decision,
                        "status",
                    )
                ),
                "decision_risk": (decision_risk),
                "execution_policy": (
                    "dry_run" if request.dry_run else "live_requested"
                ),
            },
        )

        plan.validate()

        stored = await await_if_needed(self.repository.create(plan))

        logger.info(
            "Remediation plan created",
            extra={
                "remediation": {
                    "plan_id": (stored.plan_id),
                    "incident_id": (stored.incident_id),
                    "decision_id": (stored.decision_id),
                    "risk": stored.risk,
                    "status": stored.status,
                    "dry_run": (stored.dry_run),
                    "request_id": (context.request_id if context is not None else None),
                },
            },
        )

        return plan_response(stored)

    # ------------------------------------------------------
    # Read
    # ------------------------------------------------------

    async def get_plan(
        self,
        plan_id: str,
        *,
        context: Optional[RemediationOperationContext] = None,
    ) -> RemediationPlanResponse:
        self._ensure_open()
        self._require_read_enabled()

        plan = await self._get_record(
            plan_id,
            tenant_id=(context.tenant_id if context is not None else None),
        )

        return plan_response(plan)

    async def list_plans(
        self,
        *,
        incident_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        context: Optional[RemediationOperationContext] = None,
    ) -> Tuple[
        Tuple[
            RemediationPlanResponse,
            ...,
        ],
        int,
    ]:
        self._ensure_open()
        self._require_read_enabled()

        if status is not None and status not in {
            value.value for value in RemediationPlanStatusAPI
        }:
            raise ValueError(f"Unsupported remediation status: {status}")

        records, total = await await_if_needed(
            self.repository.list(
                incident_id=incident_id,
                decision_id=decision_id,
                status=status,
                tenant_id=(context.tenant_id if context is not None else None),
                limit=limit,
                offset=offset,
            )
        )

        return (
            tuple(plan_response(record) for record in records),
            total,
        )

    # ------------------------------------------------------
    # Approval
    # ------------------------------------------------------

    async def approve_plan(
        self,
        plan_id: str,
        request: RemediationApprovalRequest,
        *,
        context: Optional[RemediationOperationContext] = None,
    ) -> RemediationPlanResponse:
        self._ensure_open()
        self._require_approval_enabled()

        lock = await self._plan_lock(plan_id)

        async with lock:
            current = await self._get_record(
                plan_id,
                tenant_id=(context.tenant_id if context is not None else None),
            )

            if current.version != request.expected_plan_version:
                raise RemediationVersionConflictAPIError(
                    details={
                        "plan_id": (current.plan_id),
                        "expected_version": (request.expected_plan_version),
                        "actual_version": (current.version),
                    },
                )

            if current.status in {
                RemediationPlanStatusAPI.APPROVED.value,
                RemediationPlanStatusAPI.REJECTED.value,
            }:
                if (
                    current.status == RemediationPlanStatusAPI.APPROVED.value
                    and request.approved
                ):
                    return plan_response(current)

                if (
                    current.status == RemediationPlanStatusAPI.REJECTED.value
                    and not request.approved
                ):
                    return plan_response(current)

                raise ResourceConflictError(
                    "The remediation plan already has a final approval result.",
                    details={
                        "plan_id": (current.plan_id),
                        "state": (current.status),
                    },
                )

            identity = (
                context.approval_identity
                if (context is not None and context.approval_identity)
                else (context.actor_id if context is not None else None)
            )

            if not identity:
                raise RemediationNotApprovedAPIError(
                    "An approval identity is required.",
                    details={
                        "plan_id": (current.plan_id),
                    },
                )

            if request.approved:
                target_status = RemediationPlanStatusAPI.APPROVED.value

                validate_plan_transition(
                    current.status,
                    target_status,
                )

                updated = replace(
                    current,
                    status=target_status,
                    approved_by=identity,
                    approved_at=utc_now(),
                    rejection_reason=None,
                    updated_at=utc_now(),
                    version=(current.version + 1),
                    attributes={
                        **dict(current.attributes),
                        "approval_reason": (request.reason),
                    },
                )

            else:
                target_status = RemediationPlanStatusAPI.REJECTED.value

                validate_plan_transition(
                    current.status,
                    target_status,
                )

                updated = replace(
                    current,
                    status=target_status,
                    approved_by=identity,
                    approved_at=utc_now(),
                    rejection_reason=(request.reason),
                    updated_at=utc_now(),
                    version=(current.version + 1),
                )

            stored = await await_if_needed(
                self.repository.update(
                    updated,
                    expected_version=(current.version),
                )
            )

            logger.info(
                "Remediation approval recorded",
                extra={
                    "remediation": {
                        "plan_id": (stored.plan_id),
                        "approved": (request.approved),
                        "status": (stored.status),
                        "request_id": (
                            context.request_id if context is not None else None
                        ),
                    },
                },
            )

            return plan_response(stored)

    # ------------------------------------------------------
    # Execute step
    # ------------------------------------------------------

    async def _run_step(
        self,
        step: RemediationStepRecord,
        *,
        dry_run: bool,
        execution_context: Mapping[
            str,
            Any,
        ],
    ) -> RemediationStepRecord:
        started = utc_now()

        running = replace(
            step,
            state="running",
            started_at=started,
            completed_at=None,
            error_code=None,
            error_message=None,
        )

        try:
            if dry_run:
                result = await asyncio.wait_for(
                    await_if_needed(
                        self.executor.validate_step(
                            running,
                            context=(execution_context),
                        )
                    ),
                    timeout=(running.timeout_seconds),
                )

            else:
                result = await asyncio.wait_for(
                    await_if_needed(
                        self.executor.execute_step(
                            running,
                            context=(execution_context),
                        )
                    ),
                    timeout=(running.timeout_seconds),
                )

            if not isinstance(
                result,
                RemediationExecutionResult,
            ):
                raise TypeError("Remediation executor returned an invalid step result.")

            if not result.success:
                raise RemediationExecutionAPIError(
                    result.message or "Remediation step failed.",
                    details={
                        "step_id": (running.step_id),
                        "provider": (running.provider),
                    },
                )

            return replace(
                running,
                state="succeeded",
                completed_at=utc_now(),
                attributes={
                    **dict(running.attributes),
                    **safe_mapping(result.attributes),
                    "changed": (result.changed),
                    "message": (result.message),
                    "dry_run": (dry_run),
                },
            )

        except asyncio.TimeoutError:
            return replace(
                running,
                state="failed",
                completed_at=utc_now(),
                error_code=("remediation_step_timeout"),
                error_message=("Remediation step timed out."),
            )

        except APIException as exc:
            return replace(
                running,
                state="failed",
                completed_at=utc_now(),
                error_code=exc.code,
                error_message=(exc.public_message),
            )

        except Exception as exc:
            return replace(
                running,
                state="failed",
                completed_at=utc_now(),
                error_code=("remediation_step_failed"),
                error_message=("Remediation step failed."),
                attributes={
                    **dict(running.attributes),
                    "error_type": (type(exc).__name__),
                },
            )

    # ------------------------------------------------------
    # Rollback
    # ------------------------------------------------------

    async def _rollback_steps(
        self,
        steps: Sequence[RemediationStepRecord],
        *,
        execution_context: Mapping[
            str,
            Any,
        ],
    ) -> Tuple[
        Tuple[
            RemediationStepRecord,
            ...,
        ],
        bool,
    ]:
        output = list(steps)

        rollback_success = True

        for index in range(
            len(output) - 1,
            -1,
            -1,
        ):
            step = output[index]

            if (
                step.state != "succeeded"
                or not step.reversible
                or not step.rollback_action
            ):
                continue

            try:
                result = await asyncio.wait_for(
                    await_if_needed(
                        self.executor.rollback_step(
                            step,
                            context=(execution_context),
                        )
                    ),
                    timeout=(step.timeout_seconds),
                )

                if (
                    not isinstance(
                        result,
                        RemediationExecutionResult,
                    )
                    or not result.success
                ):
                    rollback_success = False

                    output[index] = replace(
                        step,
                        state=("rollback_failed"),
                        attributes={
                            **dict(step.attributes),
                            "rollback_message": (
                                result.message
                                if isinstance(
                                    result,
                                    RemediationExecutionResult,
                                )
                                else ("Invalid rollback result.")
                            ),
                        },
                    )

                else:
                    output[index] = replace(
                        step,
                        state="rolled_back",
                        attributes={
                            **dict(step.attributes),
                            **safe_mapping(result.attributes),
                            "rollback_message": (result.message),
                        },
                    )

            except Exception as exc:
                rollback_success = False

                output[index] = replace(
                    step,
                    state="rollback_failed",
                    attributes={
                        **dict(step.attributes),
                        "rollback_error_type": (type(exc).__name__),
                    },
                )

        return (
            tuple(output),
            rollback_success,
        )

    # ------------------------------------------------------
    # Execute plan
    # ------------------------------------------------------

    async def execute_plan(
        self,
        plan_id: str,
        request: RemediationExecutionRequest,
        *,
        context: Optional[RemediationOperationContext] = None,
    ) -> RemediationPlanResponse:
        self._ensure_open()
        self._require_execution_enabled()

        if not request.confirm:
            raise RemediationConfirmationRequiredAPIError()

        cached = await await_if_needed(
            self.repository.find_by_execution_key(request.idempotency_key)
        )

        if cached is not None:
            if (
                context is not None
                and context.tenant_id is not None
                and cached.tenant_id != context.tenant_id
            ):
                raise RemediationNotFoundAPIError()

            return plan_response(cached)

        lock = await self._plan_lock(plan_id)

        async with lock:
            current = await self._get_record(
                plan_id,
                tenant_id=(context.tenant_id if context is not None else None),
            )

            if current.version != request.expected_plan_version:
                raise RemediationVersionConflictAPIError(
                    details={
                        "plan_id": (current.plan_id),
                        "expected_version": (request.expected_plan_version),
                        "actual_version": (current.version),
                    },
                )

            if current.status != RemediationPlanStatusAPI.APPROVED.value:
                raise RemediationNotApprovedAPIError(
                    details={
                        "plan_id": (current.plan_id),
                        "state": (current.status),
                    },
                )

            decision = await resolve_decision_record(
                self.decision_service,
                current.decision_id,
                tenant_id=(current.tenant_id),
            )

            if bool(
                _decision_value(
                    decision,
                    "expired",
                    False,
                )
            ):
                raise DecisionDeniedAPIError(
                    "The authorization decision has expired.",
                    details={
                        "decision_id": (current.decision_id),
                    },
                )

            decision_status = str(
                _decision_value(
                    decision,
                    "status",
                    "",
                )
            )

            if decision_status == DecisionStatusAPI.DENIED.value:
                raise DecisionDeniedAPIError(
                    details={
                        "decision_id": (current.decision_id),
                    },
                )

            requested_dry_run = bool(request.dry_run)

            effective_dry_run = bool(requested_dry_run or current.dry_run)

            if (
                not effective_dry_run
                and not self.config.features.remediation_live_execution_enabled
            ):
                raise RemediationExecutionDisabledAPIError(
                    "Live remediation execution is disabled.",
                    details={
                        "plan_id": (current.plan_id),
                    },
                )

            execution_id = "exec-" + uuid4().hex

            executing = replace(
                current,
                status=(RemediationPlanStatusAPI.EXECUTING.value),
                updated_at=utc_now(),
                version=(current.version + 1),
                execution_id=(execution_id),
                execution_idempotency_key=(request.idempotency_key),
                attributes={
                    **dict(current.attributes),
                    "requested_dry_run": (requested_dry_run),
                    "effective_dry_run": (effective_dry_run),
                    "execution_started_at": (utc_now().isoformat()),
                },
            )

            stored = await await_if_needed(
                self.repository.update(
                    executing,
                    expected_version=(current.version),
                )
            )

            execution_context = {
                "plan_id": stored.plan_id,
                "execution_id": execution_id,
                "incident_id": (stored.incident_id),
                "decision_id": (stored.decision_id),
                "tenant_id": (stored.tenant_id),
                "actor_id": (context.actor_id if context is not None else None),
                "request_id": (context.request_id if context is not None else None),
                "dry_run": (effective_dry_run),
            }

            step_results = []

            failed = False

            for step in stored.steps:
                if failed:
                    step_results.append(step)

                    continue

                result = await self._run_step(
                    step,
                    dry_run=(effective_dry_run),
                    execution_context=(execution_context),
                )

                step_results.append(result)

                if result.state != "succeeded":
                    failed = True

            final_steps = tuple(step_results)

            if not failed:
                completed = replace(
                    stored,
                    status=(RemediationPlanStatusAPI.SUCCEEDED.value),
                    steps=final_steps,
                    updated_at=utc_now(),
                    version=(stored.version + 1),
                    attributes={
                        **dict(stored.attributes),
                        "execution_completed_at": (utc_now().isoformat()),
                        "execution_changed_infrastructure": (
                            any(
                                bool(
                                    step.attributes.get(
                                        "changed",
                                        False,
                                    )
                                )
                                for step in final_steps
                            )
                        ),
                    },
                )

                saved = await await_if_needed(
                    self.repository.update(
                        completed,
                        expected_version=(stored.version),
                    )
                )

                logger.info(
                    "Remediation execution completed",
                    extra={
                        "remediation": {
                            "plan_id": (saved.plan_id),
                            "execution_id": (execution_id),
                            "status": (saved.status),
                            "dry_run": (effective_dry_run),
                        },
                    },
                )

                return plan_response(saved)

            # Dry runs do not require rollback
            # because no infrastructure mutation
            # was allowed.
            if effective_dry_run:
                failed_plan = replace(
                    stored,
                    status=(RemediationPlanStatusAPI.FAILED.value),
                    steps=final_steps,
                    updated_at=utc_now(),
                    version=(stored.version + 1),
                    attributes={
                        **dict(stored.attributes),
                        "execution_completed_at": (utc_now().isoformat()),
                        "rollback_required": False,
                    },
                )

                saved = await await_if_needed(
                    self.repository.update(
                        failed_plan,
                        expected_version=(stored.version),
                    )
                )

                return plan_response(saved)

            rolling_back = replace(
                stored,
                status=(RemediationPlanStatusAPI.ROLLING_BACK.value),
                steps=final_steps,
                updated_at=utc_now(),
                version=(stored.version + 1),
            )

            rolling_back = await await_if_needed(
                self.repository.update(
                    rolling_back,
                    expected_version=(stored.version),
                )
            )

            (
                rolled_back_steps,
                rollback_success,
            ) = await self._rollback_steps(
                rolling_back.steps,
                execution_context=(execution_context),
            )

            final_status = (
                RemediationPlanStatusAPI.ROLLED_BACK.value
                if rollback_success
                else RemediationPlanStatusAPI.FAILED.value
            )

            final_plan = replace(
                rolling_back,
                status=final_status,
                steps=rolled_back_steps,
                updated_at=utc_now(),
                version=(rolling_back.version + 1),
                attributes={
                    **dict(rolling_back.attributes),
                    "rollback_completed_at": (utc_now().isoformat()),
                    "rollback_success": (rollback_success),
                },
            )

            saved = await await_if_needed(
                self.repository.update(
                    final_plan,
                    expected_version=(rolling_back.version),
                )
            )

            logger.error(
                "Remediation execution failed",
                extra={
                    "remediation": {
                        "plan_id": (saved.plan_id),
                        "execution_id": (execution_id),
                        "status": (saved.status),
                        "rollback_success": (rollback_success),
                    },
                },
            )

            return plan_response(saved)

    # ------------------------------------------------------
    # Health
    # ------------------------------------------------------

    def health(
        self,
    ) -> Dict[str, Any]:
        return {
            "status": ("unhealthy" if self._closed else "healthy"),
            "ready": (not self._closed),
            "repository": type(self.repository).__name__,
            "executor": type(self.executor).__name__,
            "persistent": (
                not isinstance(
                    self.repository,
                    ProcessLocalRemediationRepository,
                )
            ),
            "live_execution_enabled": (
                self.config.features.remediation_live_execution_enabled
            ),
        }

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "service": (REMEDIATION_SERVICE_NAME),
            "version": (REMEDIATION_SERVICE_VERSION),
            "closed": self._closed,
            "read_enabled": (self.config.features.remediation_read_enabled),
            "plan_creation_enabled": (
                self.config.features.remediation_plan_creation_enabled
            ),
            "approval_enabled": (self.config.features.remediation_approval_enabled),
            "execution_enabled": (self.config.features.remediation_execution_enabled),
            "live_execution_enabled": (
                self.config.features.remediation_live_execution_enabled
            ),
            "repository": type(self.repository).__name__,
            "executor": type(self.executor).__name__,
            "persistent_repository": (
                not isinstance(
                    self.repository,
                    ProcessLocalRemediationRepository,
                )
            ),
            "default_executor_is_dry_run_only": (
                isinstance(
                    self.executor,
                    DryRunRemediationExecutor,
                )
            ),
        }


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


def create_remediation_service(
    *,
    config: APIConfig,
    decision_service: Any,
    repository: Optional[RemediationRepository] = None,
    executor: Optional[RemediationExecutor] = None,
) -> RemediationService:
    return RemediationService(
        config=config,
        decision_service=(decision_service),
        repository=(repository or ProcessLocalRemediationRepository()),
        executor=(executor or DryRunRemediationExecutor()),
    )


# ------------------------------------------------------------------
# Dependency registration
# ------------------------------------------------------------------


def register_remediation_service(
    dependencies: Any,
    *,
    repository: Optional[RemediationRepository] = None,
    executor: Optional[RemediationExecutor] = None,
    replace: bool = False,
) -> RemediationService:
    config = getattr(
        dependencies,
        "config",
        None,
    )

    register = getattr(
        dependencies,
        "register_service",
        None,
    )

    service_lookup = getattr(
        dependencies,
        "service",
        None,
    )

    if config is None:
        raise InternalServerAPIError(
            "API configuration is unavailable.",
            details={
                "resource": ("api_config"),
            },
        )

    if not callable(register):
        raise InternalServerAPIError(
            "API service registration is unavailable.",
            details={
                "resource": ("api_dependencies"),
            },
        )

    if not callable(service_lookup):
        raise InternalServerAPIError(
            "API service lookup is unavailable.",
            details={
                "resource": ("api_dependencies"),
            },
        )

    decision_service = service_lookup("decisions")

    service = create_remediation_service(
        config=config,
        decision_service=(decision_service),
        repository=repository,
        executor=executor,
    )

    register(
        REMEDIATION_SERVICE_NAME,
        service,
        replace=replace,
    )

    return service


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "REMEDIATION_SERVICE_NAME",
    "REMEDIATION_SERVICE_VERSION",
    "RemediationOperationContext",
    "RemediationStepRecord",
    "RemediationPlanRecord",
    "RemediationRepository",
    "ProcessLocalRemediationRepository",
    "RemediationExecutionResult",
    "RemediationExecutor",
    "DryRunRemediationExecutor",
    "safe_mapping",
    "step_record_from_request",
    "step_response",
    "plan_response",
    "resolve_decision_record",
    "validate_decision_for_plan",
    "validate_plan_transition",
    "RemediationService",
    "create_remediation_service",
    "register_remediation_service",
]
