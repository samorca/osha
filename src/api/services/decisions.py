from __future__ import annotations

"""
Production decision service boundary for OSHA API.

Architecture:

    Incident
       |
       v
    DecisionService
       |
       +--> incident existence validation
       |
       +--> DecisionEvaluator
       |       |
       |       +--> policy / intelligence
       |       +--> risk evaluation
       |
       +--> DecisionRepository
       |
       v
    DecisionRecord
       |
       v
    RemediationService

Important safety boundary:

    A decision is authorization information.

    A decision is NOT infrastructure execution.

This service does not:

    restart services
    modify Kubernetes resources
    modify OpenStack resources
    execute shell commands
    perform remediation
    bypass approval requirements
"""

import asyncio
import inspect
import logging

from copy import deepcopy

from dataclasses import (
    dataclass,
    field,
)

from datetime import (
    datetime,
    timedelta,
    timezone,
)

from typing import (
    Any,
    Dict,
    Iterable,
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
    DecisionNotFoundAPIError,
    FeatureDisabledError,
    IncidentNotFoundAPIError,
    InternalServerAPIError,
    ResourceConflictError,
)

from src.api.schema import (
    APIResponseMetadata,
    DecisionEvaluateRequest,
    DecisionResponse,
    DecisionStatusAPI,
    RemediationRiskAPI,
)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logger = logging.getLogger(
    __name__
)


# ------------------------------------------------------------------
# Identity
# ------------------------------------------------------------------

DECISION_SERVICE_NAME = "decisions"

DECISION_SERVICE_VERSION = "1"


# ------------------------------------------------------------------
# Decision defaults
# ------------------------------------------------------------------

DEFAULT_DECISION_TTL_SECONDS = 900

MAX_DECISION_ATTRIBUTES = 256

MAX_POLICY_RESULTS = 128

MAX_IDEMPOTENCY_RECORDS = 10_000


# ------------------------------------------------------------------
# Time
# ------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def normalize_datetime(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    )


# ------------------------------------------------------------------
# Safe dictionaries
# ------------------------------------------------------------------

_SENSITIVE_KEY_FRAGMENTS = {
    "password",
    "secret",
    "token",
    "credential",
    "authorization",
    "cookie",
    "private",
    "certificate",
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
    limit: int = MAX_DECISION_ATTRIBUTES,
) -> Dict[str, Any]:
    if not value:
        return {}

    output: Dict[
        str,
        Any,
    ] = {}

    for raw_key, item in list(
        value.items()
    )[
        :limit
    ]:
        key = str(
            raw_key
        ).strip()

        if not key:
            continue

        normalized = key.lower()

        if any(
            fragment in normalized
            for fragment
            in _SENSITIVE_KEY_FRAGMENTS
        ):
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
            output[
                key
            ] = item

        elif isinstance(
            item,
            (
                list,
                tuple,
            ),
        ):
            output[
                key
            ] = [
                child
                for child in item[
                    :128
                ]
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

        elif isinstance(
            item,
            Mapping,
        ):
            output[
                key
            ] = safe_mapping(
                item,
                limit=64,
            )

        else:
            output[
                key
            ] = {
                "type": type(
                    item
                ).__name__,
            }

    return output


# ------------------------------------------------------------------
# Operation context
# ------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionOperationContext:
    request_id: Optional[str] = None

    actor_id: Optional[str] = None

    tenant_id: Optional[str] = None

    correlation_id: Optional[str] = None

    idempotency_key: Optional[str] = None

    def safe_attributes(
        self,
    ) -> Dict[str, Any]:
        return {
            "request_id": (
                self.request_id
            ),

            "actor_id": (
                self.actor_id
            ),

            "tenant_id": (
                self.tenant_id
            ),

            "correlation_id": (
                self.correlation_id
            ),

            "idempotency_key_present": bool(
                self.idempotency_key
            ),
        }


# ------------------------------------------------------------------
# Policy evaluation result
# ------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyResult:
    policy: str

    allowed: bool

    reason: str

    severity: str = "medium"

    attributes: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "allowed": self.allowed,
            "reason": self.reason,
            "severity": self.severity,
            "attributes": safe_mapping(
                self.attributes
            ),
        }


# ------------------------------------------------------------------
# Evaluation outcome
# ------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionEvaluation:
    """
    Backend-neutral evaluation result.

    status must be one of:

        allowed
        denied
        requires_approval
    """

    status: str

    reason: str

    risk: str

    approval_required: bool

    policy_results: Tuple[
        PolicyResult,
        ...,
    ] = ()

    ttl_seconds: int = (
        DEFAULT_DECISION_TTL_SECONDS
    )

    attributes: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        if (
            self.status
            not in {
                DecisionStatusAPI.
                ALLOWED.value,

                DecisionStatusAPI.
                DENIED.value,

                DecisionStatusAPI.
                REQUIRES_APPROVAL.value,
            }
        ):
            raise ValueError(
                "Unsupported evaluated "
                "decision status: "
                f"{self.status}"
            )

        if (
            self.risk
            not in {
                value.value
                for value
                in RemediationRiskAPI
            }
        ):
            raise ValueError(
                "Unsupported decision risk: "
                f"{self.risk}"
            )

        if self.ttl_seconds < 1:
            raise ValueError(
                "Decision TTL must be >= 1"
            )

        if (
            self.status
            == DecisionStatusAPI.
            REQUIRES_APPROVAL.value
            and not self.approval_required
        ):
            raise ValueError(
                "requires_approval decision "
                "must require approval"
            )

        if (
            self.status
            == DecisionStatusAPI.
            DENIED.value
            and self.approval_required
        ):
            raise ValueError(
                "Denied decision cannot "
                "require approval"
            )


# ------------------------------------------------------------------
# Persistent decision record
# ------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str

    incident_id: str

    candidate_action: str

    target: Mapping[
        str,
        Any,
    ]

    status: str

    reason: str

    risk: str

    approval_required: bool

    policy_results: Tuple[
        PolicyResult,
        ...,
    ]

    created_at: datetime

    expires_at: Optional[
        datetime
    ]

    version: int = 1

    dry_run: bool = True

    tenant_id: Optional[str] = None

    actor_id: Optional[str] = None

    correlation_id: Optional[str] = None

    idempotency_key: Optional[str] = None

    attributes: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        if not self.decision_id.strip():
            raise ValueError(
                "decision_id cannot be empty"
            )

        if not self.incident_id.strip():
            raise ValueError(
                "incident_id cannot be empty"
            )

        if not self.candidate_action.strip():
            raise ValueError(
                "candidate_action cannot be empty"
            )

        if (
            self.status
            not in {
                value.value
                for value
                in DecisionStatusAPI
            }
        ):
            raise ValueError(
                "Unsupported decision status: "
                f"{self.status}"
            )

        if (
            self.risk
            not in {
                value.value
                for value
                in RemediationRiskAPI
            }
        ):
            raise ValueError(
                "Unsupported decision risk: "
                f"{self.risk}"
            )

        if self.version < 1:
            raise ValueError(
                "decision version must be >= 1"
            )

        if (
            len(
                self.policy_results
            )
            > MAX_POLICY_RESULTS
        ):
            raise ValueError(
                "Too many decision policy results"
            )

    @property
    def expired(
        self,
    ) -> bool:
        if self.expires_at is None:
            return False

        return utc_now() >= normalize_datetime(
            self.expires_at
        )


# ------------------------------------------------------------------
# Repository
# ------------------------------------------------------------------

@runtime_checkable
class DecisionRepository(
    Protocol,
):
    async def create(
        self,
        decision: DecisionRecord,
    ) -> DecisionRecord:
        ...

    async def get(
        self,
        decision_id: str,
    ) -> Optional[DecisionRecord]:
        ...

    async def list(
        self,
        *,
        incident_id: Optional[
            str
        ] = None,
        tenant_id: Optional[
            str
        ] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[
        Tuple[
            DecisionRecord,
            ...,
        ],
        int,
    ]:
        ...

    async def find_by_idempotency_key(
        self,
        key: str,
    ) -> Optional[DecisionRecord]:
        ...

    async def close(
        self,
    ) -> None:
        ...


# ------------------------------------------------------------------
# Process-local decision repository
# ------------------------------------------------------------------

class ProcessLocalDecisionRepository:
    def __init__(
        self,
    ) -> None:
        self._records: Dict[
            str,
            DecisionRecord,
        ] = {}

        self._idempotency: Dict[
            str,
            str,
        ] = {}

        self._lock = asyncio.Lock()

        self._closed = False

    def _ensure_open(
        self,
    ) -> None:
        if self._closed:
            raise RuntimeError(
                "Decision repository is closed."
            )

    async def create(
        self,
        decision: DecisionRecord,
    ) -> DecisionRecord:
        decision.validate()

        async with self._lock:
            self._ensure_open()

            if (
                decision.decision_id
                in self._records
            ):
                raise ResourceConflictError(
                    "A decision with this ID "
                    "already exists.",
                    details={
                        "decision_id": (
                            decision.decision_id
                        ),
                    },
                )

            if decision.idempotency_key:
                existing_id = (
                    self._idempotency.get(
                        decision.idempotency_key
                    )
                )

                if existing_id:
                    return deepcopy(
                        self._records[
                            existing_id
                        ]
                    )

            self._records[
                decision.decision_id
            ] = deepcopy(
                decision
            )

            if decision.idempotency_key:
                if (
                    len(
                        self._idempotency
                    )
                    >= MAX_IDEMPOTENCY_RECORDS
                ):
                    oldest = next(
                        iter(
                            self._idempotency
                        )
                    )

                    self._idempotency.pop(
                        oldest,
                        None,
                    )

                self._idempotency[
                    decision.idempotency_key
                ] = decision.decision_id

            return deepcopy(
                decision
            )

    async def get(
        self,
        decision_id: str,
    ) -> Optional[DecisionRecord]:
        normalized = str(
            decision_id
        ).strip()

        async with self._lock:
            self._ensure_open()

            value = self._records.get(
                normalized
            )

            return (
                deepcopy(
                    value
                )
                if value is not None
                else None
            )

    async def list(
        self,
        *,
        incident_id: Optional[
            str
        ] = None,
        tenant_id: Optional[
            str
        ] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[
        Tuple[
            DecisionRecord,
            ...,
        ],
        int,
    ]:
        if (
            limit < 1
            or limit > 1000
        ):
            raise ValueError(
                "limit must be between "
                "1 and 1000"
            )

        if offset < 0:
            raise ValueError(
                "offset must be >= 0"
            )

        async with self._lock:
            self._ensure_open()

            values = []

            for decision in (
                self._records.values()
            ):
                if (
                    incident_id is not None
                    and decision.incident_id
                    != incident_id
                ):
                    continue

                if (
                    tenant_id is not None
                    and decision.tenant_id
                    != tenant_id
                ):
                    continue

                values.append(
                    decision
                )

            values.sort(
                key=lambda decision: (
                    decision.created_at,
                    decision.decision_id,
                ),
                reverse=True,
            )

            total = len(
                values
            )

            page = values[
                offset:
                offset + limit
            ]

            return (
                tuple(
                    deepcopy(
                        page
                    )
                ),
                total,
            )

    async def find_by_idempotency_key(
        self,
        key: str,
    ) -> Optional[DecisionRecord]:
        normalized = str(
            key
        ).strip()

        if not normalized:
            return None

        async with self._lock:
            self._ensure_open()

            decision_id = (
                self._idempotency.get(
                    normalized
                )
            )

            if decision_id is None:
                return None

            return deepcopy(
                self._records.get(
                    decision_id
                )
            )

    async def close(
        self,
    ) -> None:
        async with self._lock:
            self._closed = True

            self._records.clear()
            self._idempotency.clear()


# ------------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------------

@runtime_checkable
class DecisionEvaluator(
    Protocol,
):
    async def evaluate(
        self,
        *,
        incident: Any,
        candidate_action: str,
        target: Mapping[
            str,
            Any,
        ],
        context: Mapping[
            str,
            Any,
        ],
        dry_run: bool,
    ) -> DecisionEvaluation:
        ...


# ------------------------------------------------------------------
# Conservative default evaluator
# ------------------------------------------------------------------

class ConservativeDecisionEvaluator:
    """
    Safe default until OSHA's domain decision engine
    is wired into the API.

    It never automatically authorizes an
    infrastructure-changing action.

    Valid candidate requests are classified as
    requiring approval.

    This is deliberately conservative rather than
    inventing operational policy.
    """

    async def evaluate(
        self,
        *,
        incident: Any,
        candidate_action: str,
        target: Mapping[
            str,
            Any,
        ],
        context: Mapping[
            str,
            Any,
        ],
        dry_run: bool,
    ) -> DecisionEvaluation:
        action = str(
            candidate_action
        ).strip()

        if not action:
            return DecisionEvaluation(
                status=(
                    DecisionStatusAPI.
                    DENIED.value
                ),

                reason=(
                    "No candidate action "
                    "was supplied."
                ),

                risk=(
                    RemediationRiskAPI.
                    HIGH.value
                ),

                approval_required=False,

                policy_results=(
                    PolicyResult(
                        policy=(
                            "candidate_action_present"
                        ),

                        allowed=False,

                        reason=(
                            "Candidate action "
                            "is required."
                        ),

                        severity="high",
                    ),
                ),
            )

        return DecisionEvaluation(
            status=(
                DecisionStatusAPI.
                REQUIRES_APPROVAL.value
            ),

            reason=(
                "The default API decision evaluator "
                "does not automatically authorize "
                "infrastructure-changing actions."
            ),

            risk=(
                RemediationRiskAPI.
                HIGH.value
            ),

            approval_required=True,

            policy_results=(
                PolicyResult(
                    policy=(
                        "explicit_approval"
                    ),

                    allowed=False,

                    reason=(
                        "Explicit authorization "
                        "is required before "
                        "remediation planning."
                    ),

                    severity="high",
                ),
            ),

            attributes={
                "default_evaluator": True,
                "dry_run": bool(
                    dry_run
                ),
            },
        )


# ------------------------------------------------------------------
# Async helper
# ------------------------------------------------------------------

async def await_if_needed(
    value: Any,
) -> Any:
    if inspect.isawaitable(
        value
    ):
        return await value

    return value


# ------------------------------------------------------------------
# Response
# ------------------------------------------------------------------

def decision_response(
    decision: DecisionRecord,
) -> DecisionResponse:
    decision.validate()

    status = decision.status

    if decision.expired:
        status = (
            DecisionStatusAPI.
            EXPIRED.value
        )

    attributes = safe_mapping(
        decision.attributes
    )

    attributes.update(
        {
            "version": (
                decision.version
            ),

            "dry_run": (
                decision.dry_run
            ),

            "correlation_id": (
                decision.correlation_id
            ),
        }
    )

    return DecisionResponse(
        decision_id=(
            decision.decision_id
        ),

        incident_id=(
            decision.incident_id
        ),

        status=status,

        candidate_action=(
            decision.candidate_action
        ),

        reason=(
            decision.reason
        ),

        risk=decision.risk,

        approval_required=(
            decision.approval_required
        ),

        policy_results=[
            result.to_dict()
            for result
            in decision.policy_results
        ],

        created_at=(
            decision.created_at
        ),

        expires_at=(
            decision.expires_at
        ),

        attributes=attributes,
    )


# ------------------------------------------------------------------
# Incident service lookup helper
# ------------------------------------------------------------------

async def resolve_incident(
    incident_service: Any,
    incident_id: str,
    *,
    context: Optional[
        DecisionOperationContext
    ] = None,
) -> Any:
    if incident_service is None:
        raise InternalServerAPIError(
            "Incident service is unavailable.",
            details={
                "resource": (
                    "incident_service"
                ),
            },
        )

    getter = getattr(
        incident_service,
        "get_incident",
        None,
    )

    if not callable(
        getter
    ):
        raise InternalServerAPIError(
            "Incident service does not "
            "support incident retrieval.",
            details={
                "resource": (
                    "incident_service"
                ),
            },
        )

    try:
        incident_context = None

        try:
            from src.api.services.incidents import (
                IncidentOperationContext,
            )

            incident_context = (
                IncidentOperationContext(
                    request_id=(
                        context.request_id
                        if context
                        is not None
                        else None
                    ),

                    actor_id=(
                        context.actor_id
                        if context
                        is not None
                        else None
                    ),

                    tenant_id=(
                        context.tenant_id
                        if context
                        is not None
                        else None
                    ),

                    correlation_id=(
                        context.correlation_id
                        if context
                        is not None
                        else None
                    ),
                )
            )

        except Exception:
            incident_context = None

        return await await_if_needed(
            getter(
                incident_id,
                context=incident_context,
            )
        )

    except IncidentNotFoundAPIError:
        raise


# ------------------------------------------------------------------
# Decision service
# ------------------------------------------------------------------

class DecisionService:
    """
    Process-scoped decision application service.

    Evaluation and persistence are separated so
    OSHA's domain policy engine and persistent
    repository can be injected later.
    """

    def __init__(
        self,
        *,
        config: APIConfig,
        repository: DecisionRepository,
        evaluator: DecisionEvaluator,
        incident_service: Any,
    ) -> None:
        config.validate()

        self.config = config
        self.repository = repository
        self.evaluator = evaluator

        self.incident_service = (
            incident_service
        )

        self._closed = False

        self._lock = asyncio.Lock()

    # ------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------

    def _ensure_open(
        self,
    ) -> None:
        if self._closed:
            raise InternalServerAPIError(
                "The decision service is closed.",
                details={
                    "resource": (
                        DECISION_SERVICE_NAME
                    ),
                },
            )

    def _require_read_enabled(
        self,
    ) -> None:
        if not (
            self.config.features.
            decision_read_enabled
        ):
            raise FeatureDisabledError(
                details={
                    "resource": (
                        "decision_read"
                    ),
                },
            )

    def _require_evaluation_enabled(
        self,
    ) -> None:
        if not (
            self.config.features.
            decision_evaluation_enabled
        ):
            raise FeatureDisabledError(
                details={
                    "resource": (
                        "decision_evaluation"
                    ),
                },
            )

    async def close(
        self,
    ) -> None:
        async with self._lock:
            if self._closed:
                return

            await await_if_needed(
                self.repository.close()
            )

            self._closed = True

    # ------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------

    async def evaluate(
        self,
        request: DecisionEvaluateRequest,
        *,
        context: Optional[
            DecisionOperationContext
        ] = None,
    ) -> DecisionResponse:
        self._ensure_open()
        self._require_evaluation_enabled()

        idempotency_key = (
            context.idempotency_key
            if context is not None
            else None
        )

        if idempotency_key:
            existing = await await_if_needed(
                self.repository.
                find_by_idempotency_key(
                    idempotency_key
                )
            )

            if existing is not None:
                return decision_response(
                    existing
                )

        incident = await resolve_incident(
            self.incident_service,
            request.incident_id,
            context=context,
        )

        target = safe_mapping(
            request.target
        )

        evaluation_context = safe_mapping(
            request.context
        )

        if context is not None:
            evaluation_context.setdefault(
                "actor_id",
                context.actor_id,
            )

            evaluation_context.setdefault(
                "tenant_id",
                context.tenant_id,
            )

            evaluation_context.setdefault(
                "correlation_id",
                context.correlation_id,
            )

        try:
            evaluation = (
                await await_if_needed(
                    self.evaluator.evaluate(
                        incident=incident,

                        candidate_action=(
                            request.
                            candidate_action
                        ),

                        target=target,

                        context=(
                            evaluation_context
                        ),

                        dry_run=(
                            request.dry_run
                        ),
                    )
                )
            )

            if not isinstance(
                evaluation,
                DecisionEvaluation,
            ):
                raise TypeError(
                    "Decision evaluator returned "
                    "an invalid result."
                )

            evaluation.validate()

        except APIException:
            raise

        except Exception as exc:
            raise InternalServerAPIError(
                "Decision evaluation failed.",
                details={
                    "operation": (
                        "decision_evaluation"
                    ),

                    "incident_id": (
                        request.incident_id
                    ),

                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                },
            ) from exc

        created_at = utc_now()

        expires_at = (
            created_at
            + timedelta(
                seconds=(
                    evaluation.
                    ttl_seconds
                )
            )
        )

        record = DecisionRecord(
            decision_id=(
                "dec-"
                + uuid4().hex
            ),

            incident_id=(
                request.incident_id
            ),

            candidate_action=(
                request.candidate_action
            ),

            target=target,

            status=(
                evaluation.status
            ),

            reason=(
                evaluation.reason
            ),

            risk=(
                evaluation.risk
            ),

            approval_required=(
                evaluation.
                approval_required
            ),

            policy_results=(
                evaluation.
                policy_results
            ),

            created_at=(
                created_at
            ),

            expires_at=(
                expires_at
            ),

            version=1,

            dry_run=(
                request.dry_run
            ),

            tenant_id=(
                context.tenant_id
                if context is not None
                else None
            ),

            actor_id=(
                context.actor_id
                if context is not None
                else None
            ),

            correlation_id=(
                context.correlation_id
                if context is not None
                else None
            ),

            idempotency_key=(
                idempotency_key
            ),

            attributes={
                **safe_mapping(
                    evaluation.attributes
                ),

                "target": target,
            },
        )

        record.validate()

        stored = await await_if_needed(
            self.repository.create(
                record
            )
        )

        logger.info(
            "Decision evaluated",
            extra={
                "decision": {
                    "decision_id": (
                        stored.decision_id
                    ),

                    "incident_id": (
                        stored.incident_id
                    ),

                    "status": (
                        stored.status
                    ),

                    "risk": (
                        stored.risk
                    ),

                    "approval_required": (
                        stored.
                        approval_required
                    ),

                    "request_id": (
                        context.request_id
                        if context
                        is not None
                        else None
                    ),
                },
            },
        )

        return decision_response(
            stored
        )

    # ------------------------------------------------------
    # Get
    # ------------------------------------------------------

    async def get_decision(
        self,
        decision_id: str,
        *,
        context: Optional[
            DecisionOperationContext
        ] = None,
    ) -> DecisionResponse:
        self._ensure_open()
        self._require_read_enabled()

        normalized = str(
            decision_id
        ).strip()

        if not normalized:
            raise DecisionNotFoundAPIError(
                details={
                    "resource": (
                        "decision"
                    ),
                },
            )

        decision = await await_if_needed(
            self.repository.get(
                normalized
            )
        )

        if decision is None:
            raise DecisionNotFoundAPIError(
                details={
                    "decision_id": (
                        normalized
                    ),
                },
            )

        if (
            context is not None
            and context.tenant_id is not None
            and decision.tenant_id
            != context.tenant_id
        ):
            raise DecisionNotFoundAPIError(
                details={
                    "decision_id": (
                        normalized
                    ),
                },
            )

        return decision_response(
            decision
        )

    # ------------------------------------------------------
    # Internal record lookup for remediation
    # ------------------------------------------------------

    async def get_record(
        self,
        decision_id: str,
        *,
        tenant_id: Optional[
            str
        ] = None,
    ) -> DecisionRecord:
        self._ensure_open()

        decision = await await_if_needed(
            self.repository.get(
                decision_id
            )
        )

        if decision is None:
            raise DecisionNotFoundAPIError(
                details={
                    "decision_id": (
                        decision_id
                    ),
                },
            )

        if (
            tenant_id is not None
            and decision.tenant_id
            != tenant_id
        ):
            raise DecisionNotFoundAPIError(
                details={
                    "decision_id": (
                        decision_id
                    ),
                },
            )

        return decision

    # ------------------------------------------------------
    # List
    # ------------------------------------------------------

    async def list_decisions(
        self,
        *,
        incident_id: Optional[
            str
        ] = None,
        limit: int = 100,
        offset: int = 0,
        context: Optional[
            DecisionOperationContext
        ] = None,
    ) -> Tuple[
        Tuple[
            DecisionResponse,
            ...,
        ],
        int,
    ]:
        self._ensure_open()
        self._require_read_enabled()

        records, total = (
            await await_if_needed(
                self.repository.list(
                    incident_id=(
                        incident_id
                    ),

                    tenant_id=(
                        context.tenant_id
                        if context
                        is not None
                        else None
                    ),

                    limit=limit,

                    offset=offset,
                )
            )
        )

        return (
            tuple(
                decision_response(
                    record
                )
                for record
                in records
            ),
            total,
        )

    # ------------------------------------------------------
    # Health
    # ------------------------------------------------------

    def health(
        self,
    ) -> Dict[str, Any]:
        return {
            "status": (
                "unhealthy"
                if self._closed
                else "healthy"
            ),

            "ready": (
                not self._closed
            ),

            "repository": type(
                self.repository
            ).__name__,

            "evaluator": type(
                self.evaluator
            ).__name__,

            "persistent": (
                not isinstance(
                    self.repository,
                    ProcessLocalDecisionRepository,
                )
            ),
        }

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "service": (
                DECISION_SERVICE_NAME
            ),

            "version": (
                DECISION_SERVICE_VERSION
            ),

            "closed": (
                self._closed
            ),

            "read_enabled": (
                self.config.features.
                decision_read_enabled
            ),

            "evaluation_enabled": (
                self.config.features.
                decision_evaluation_enabled
            ),

            "repository": type(
                self.repository
            ).__name__,

            "evaluator": type(
                self.evaluator
            ).__name__,

            "persistent_repository": (
                not isinstance(
                    self.repository,
                    ProcessLocalDecisionRepository,
                )
            ),

            "infrastructure_execution": (
                False
            ),
        }


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def create_decision_service(
    *,
    config: APIConfig,
    incident_service: Any,
    repository: Optional[
        DecisionRepository
    ] = None,
    evaluator: Optional[
        DecisionEvaluator
    ] = None,
) -> DecisionService:
    return DecisionService(
        config=config,

        incident_service=(
            incident_service
        ),

        repository=(
            repository
            or ProcessLocalDecisionRepository()
        ),

        evaluator=(
            evaluator
            or ConservativeDecisionEvaluator()
        ),
    )


# ------------------------------------------------------------------
# Dependency registration
# ------------------------------------------------------------------

def register_decision_service(
    dependencies: Any,
    *,
    repository: Optional[
        DecisionRepository
    ] = None,
    evaluator: Optional[
        DecisionEvaluator
    ] = None,
    replace: bool = False,
) -> DecisionService:
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
                "resource": (
                    "api_config"
                ),
            },
        )

    if not callable(
        register
    ):
        raise InternalServerAPIError(
            "API service registration is "
            "unavailable.",
            details={
                "resource": (
                    "api_dependencies"
                ),
            },
        )

    if not callable(
        service_lookup
    ):
        raise InternalServerAPIError(
            "API service lookup is unavailable.",
            details={
                "resource": (
                    "api_dependencies"
                ),
            },
        )

    incident_service = (
        service_lookup(
            "incidents"
        )
    )

    service = create_decision_service(
        config=config,

        incident_service=(
            incident_service
        ),

        repository=repository,

        evaluator=evaluator,
    )

    register(
        DECISION_SERVICE_NAME,
        service,
        replace=replace,
    )

    return service


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "DECISION_SERVICE_NAME",
    "DECISION_SERVICE_VERSION",

    "DEFAULT_DECISION_TTL_SECONDS",

    "DecisionOperationContext",

    "PolicyResult",
    "DecisionEvaluation",
    "DecisionRecord",

    "DecisionRepository",
    "ProcessLocalDecisionRepository",

    "DecisionEvaluator",
    "ConservativeDecisionEvaluator",

    "safe_mapping",
    "decision_response",
    "resolve_incident",

    "DecisionService",

    "create_decision_service",
    "register_decision_service",
]
