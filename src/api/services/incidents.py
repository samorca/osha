from __future__ import annotations

"""
Production incident service boundary for OSHA API.

Responsibilities:

    create incidents
    enforce incident feature gates
    implement idempotent incident creation
    list and filter incidents
    retrieve a specific incident
    update incident lifecycle state
    preserve associated signals
    translate internal records to API responses
    isolate storage behind a repository protocol

Architecture:

    routes/incidents.py
            |
            v
    IncidentService
            |
            v
    IncidentRepository
            |
            +--> ProcessLocalIncidentRepository
            |
            +--> future persistent repository
                 PostgreSQL / OpenSearch /
                 OSHA incident memory

This module does not:

    perform signal correlation
    perform root-cause analysis
    make remediation decisions
    execute infrastructure mutations
    infer incidents from raw telemetry
    expose arbitrary exception details

Important:

    ProcessLocalIncidentRepository is suitable for
    development and single-worker validation.

    Production multi-worker deployments should
    inject a shared persistent repository.
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
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from uuid import uuid4

from src.api.config import APIConfig

from src.api.errors import (
    APIException,
    FeatureDisabledError,
    IncidentNotFoundAPIError,
    InternalServerAPIError,
    ResourceConflictError,
)

from src.api.schema import (
    APIResponseMetadata,
    IncidentCreateRequest,
    IncidentListResponse,
    IncidentResponse,
    IncidentSignalRequest,
    IncidentStatusAPI,
    PaginationMetadata,
    SeverityAPI,
)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logger = logging.getLogger(
    __name__
)


# ------------------------------------------------------------------
# Service identity
# ------------------------------------------------------------------

INCIDENT_SERVICE_NAME = "incidents"

INCIDENT_SERVICE_VERSION = "1"


# ------------------------------------------------------------------
# Limits
# ------------------------------------------------------------------

MAX_INCIDENT_ATTRIBUTES = 256

MAX_SIGNAL_ATTRIBUTES = 256

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
# Safe attributes
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


def safe_attributes(
    value: Optional[
        Mapping[
            str,
            Any,
        ]
    ],
    *,
    maximum_entries: int,
) -> Dict[str, Any]:
    """
    Preserve bounded JSON-like incident metadata
    while dropping keys that appear sensitive.

    Complex application objects are converted to
    their type name rather than implicitly
    serialized.
    """

    if not value:
        return {}

    output: Dict[str, Any] = {}

    for raw_key, item in list(
        value.items()
    )[
        :maximum_entries
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

            continue

        if isinstance(
            item,
            (
                list,
                tuple,
            ),
        ):
            values = []

            for child in item[
                :128
            ]:
                if isinstance(
                    child,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    ),
                ):
                    values.append(
                        child
                    )

            output[
                key
            ] = values

            continue

        if isinstance(
            item,
            Mapping,
        ):
            output[
                key
            ] = safe_attributes(
                item,
                maximum_entries=64,
            )

            continue

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
class IncidentOperationContext:
    """
    Request-specific information supplied by the
    route layer.
    """

    request_id: Optional[str] = None

    actor_id: Optional[str] = None

    tenant_id: Optional[str] = None

    correlation_id: Optional[str] = None

    idempotency_key: Optional[str] = None

    def safe_attributes(
        self,
    ) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "actor_id": self.actor_id,
            "tenant_id": self.tenant_id,
            "correlation_id": self.correlation_id,
            "idempotency_key_present": bool(
                self.idempotency_key
            ),
        }


# ------------------------------------------------------------------
# Internal incident records
# ------------------------------------------------------------------

@dataclass(frozen=True)
class IncidentSignalRecord:
    signal_id: str

    source: str

    signal_type: str

    observed_at: datetime

    severity: str

    payload: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
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
        if not self.signal_id.strip():
            raise ValueError(
                "signal_id cannot be empty"
            )

        if not self.source.strip():
            raise ValueError(
                "signal source cannot be empty"
            )

        if not self.signal_type.strip():
            raise ValueError(
                "signal_type cannot be empty"
            )

        if (
            self.severity
            not in {
                value.value
                for value
                in SeverityAPI
            }
        ):
            raise ValueError(
                "Unsupported signal severity: "
                f"{self.severity}"
            )


@dataclass(frozen=True)
class IncidentRecord:
    incident_id: str

    title: str

    summary: Optional[str]

    severity: str

    status: str

    service: Optional[str]

    environment: Optional[str]

    signals: Tuple[
        IncidentSignalRecord,
        ...,
    ]

    attributes: Mapping[
        str,
        Any,
    ]

    created_at: datetime

    updated_at: datetime

    version: int = 1

    tenant_id: Optional[str] = None

    created_by: Optional[str] = None

    correlation_id: Optional[str] = None

    idempotency_key: Optional[str] = None

    def validate(
        self,
    ) -> None:
        if not self.incident_id.strip():
            raise ValueError(
                "incident_id cannot be empty"
            )

        if not self.title.strip():
            raise ValueError(
                "incident title cannot be empty"
            )

        if (
            self.severity
            not in {
                value.value
                for value
                in SeverityAPI
            }
        ):
            raise ValueError(
                "Unsupported incident severity: "
                f"{self.severity}"
            )

        if (
            self.status
            not in {
                value.value
                for value
                in IncidentStatusAPI
            }
        ):
            raise ValueError(
                "Unsupported incident status: "
                f"{self.status}"
            )

        if self.version < 1:
            raise ValueError(
                "incident version must be >= 1"
            )

        signal_ids = [
            signal.signal_id
            for signal
            in self.signals
        ]

        if (
            len(
                signal_ids
            )
            != len(
                set(
                    signal_ids
                )
            )
        ):
            raise ValueError(
                "incident signal IDs must "
                "be unique"
            )

        for signal in self.signals:
            signal.validate()


# ------------------------------------------------------------------
# Query
# ------------------------------------------------------------------

@dataclass(frozen=True)
class IncidentQuery:
    status: Optional[str] = None

    severity: Optional[str] = None

    service: Optional[str] = None

    environment: Optional[str] = None

    tenant_id: Optional[str] = None

    created_after: Optional[datetime] = None

    created_before: Optional[datetime] = None

    limit: int = 100

    offset: int = 0

    descending: bool = True

    def validate(
        self,
    ) -> None:
        if (
            self.status is not None
            and self.status
            not in {
                value.value
                for value
                in IncidentStatusAPI
            }
        ):
            raise ValueError(
                "Unsupported incident status: "
                f"{self.status}"
            )

        if (
            self.severity is not None
            and self.severity
            not in {
                value.value
                for value
                in SeverityAPI
            }
        ):
            raise ValueError(
                "Unsupported incident severity: "
                f"{self.severity}"
            )

        if (
            self.limit < 1
            or self.limit > 1000
        ):
            raise ValueError(
                "limit must be between "
                "1 and 1000"
            )

        if self.offset < 0:
            raise ValueError(
                "offset must be >= 0"
            )

        if (
            self.created_after is not None
            and self.created_before is not None
            and normalize_datetime(
                self.created_after
            )
            > normalize_datetime(
                self.created_before
            )
        ):
            raise ValueError(
                "created_after cannot be "
                "later than created_before"
            )


# ------------------------------------------------------------------
# Repository protocol
# ------------------------------------------------------------------

@runtime_checkable
class IncidentRepository(
    Protocol,
):
    async def create(
        self,
        incident: IncidentRecord,
    ) -> IncidentRecord:
        ...

    async def get(
        self,
        incident_id: str,
    ) -> Optional[IncidentRecord]:
        ...

    async def list(
        self,
        query: IncidentQuery,
    ) -> Tuple[
        Tuple[
            IncidentRecord,
            ...,
        ],
        int,
    ]:
        ...

    async def update(
        self,
        incident: IncidentRecord,
        *,
        expected_version: int,
    ) -> IncidentRecord:
        ...

    async def find_by_idempotency_key(
        self,
        key: str,
    ) -> Optional[IncidentRecord]:
        ...

    async def close(
        self,
    ) -> None:
        ...


# ------------------------------------------------------------------
# Process-local repository
# ------------------------------------------------------------------

class ProcessLocalIncidentRepository:
    """
    Thread-safe and coroutine-safe process-local
    incident repository.

    This repository is not distributed and does
    not provide durability across process restarts.
    """

    def __init__(
        self,
    ) -> None:
        self._records: Dict[
            str,
            IncidentRecord,
        ] = {}

        self._idempotency_index: Dict[
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
                "Incident repository is closed."
            )

    async def create(
        self,
        incident: IncidentRecord,
    ) -> IncidentRecord:
        incident.validate()

        async with self._lock:
            self._ensure_open()

            if (
                incident.incident_id
                in self._records
            ):
                raise ResourceConflictError(
                    "An incident with this ID "
                    "already exists.",
                    details={
                        "incident_id": (
                            incident.incident_id
                        ),
                    },
                )

            if incident.idempotency_key:
                existing_id = (
                    self._idempotency_index.get(
                        incident.idempotency_key
                    )
                )

                if existing_id:
                    return deepcopy(
                        self._records[
                            existing_id
                        ]
                    )

            self._records[
                incident.incident_id
            ] = deepcopy(
                incident
            )

            if incident.idempotency_key:
                if (
                    len(
                        self._idempotency_index
                    )
                    >= MAX_IDEMPOTENCY_RECORDS
                ):
                    oldest = next(
                        iter(
                            self.
                            _idempotency_index
                        )
                    )

                    self._idempotency_index.pop(
                        oldest,
                        None,
                    )

                self._idempotency_index[
                    incident.idempotency_key
                ] = incident.incident_id

            return deepcopy(
                incident
            )

    async def get(
        self,
        incident_id: str,
    ) -> Optional[IncidentRecord]:
        normalized = str(
            incident_id
        ).strip()

        async with self._lock:
            self._ensure_open()

            incident = self._records.get(
                normalized
            )

            return (
                deepcopy(
                    incident
                )
                if incident is not None
                else None
            )

    async def list(
        self,
        query: IncidentQuery,
    ) -> Tuple[
        Tuple[
            IncidentRecord,
            ...,
        ],
        int,
    ]:
        query.validate()

        async with self._lock:
            self._ensure_open()

            records = list(
                self._records.values()
            )

            output = []

            for incident in records:
                if (
                    query.status is not None
                    and incident.status
                    != query.status
                ):
                    continue

                if (
                    query.severity is not None
                    and incident.severity
                    != query.severity
                ):
                    continue

                if (
                    query.service is not None
                    and incident.service
                    != query.service
                ):
                    continue

                if (
                    query.environment is not None
                    and incident.environment
                    != query.environment
                ):
                    continue

                if (
                    query.tenant_id is not None
                    and incident.tenant_id
                    != query.tenant_id
                ):
                    continue

                if (
                    query.created_after
                    is not None
                    and incident.created_at
                    < normalize_datetime(
                        query.created_after
                    )
                ):
                    continue

                if (
                    query.created_before
                    is not None
                    and incident.created_at
                    > normalize_datetime(
                        query.created_before
                    )
                ):
                    continue

                output.append(
                    incident
                )

            output.sort(
                key=lambda incident: (
                    incident.created_at,
                    incident.incident_id,
                ),
                reverse=query.descending,
            )

            total = len(
                output
            )

            selected = output[
                query.offset:
                query.offset
                + query.limit
            ]

            return (
                tuple(
                    deepcopy(
                        selected
                    )
                ),
                total,
            )

    async def update(
        self,
        incident: IncidentRecord,
        *,
        expected_version: int,
    ) -> IncidentRecord:
        incident.validate()

        async with self._lock:
            self._ensure_open()

            current = self._records.get(
                incident.incident_id
            )

            if current is None:
                raise IncidentNotFoundAPIError(
                    details={
                        "incident_id": (
                            incident.incident_id
                        ),
                    },
                )

            if (
                current.version
                != expected_version
            ):
                raise ResourceConflictError(
                    "The incident version changed.",
                    code="version_conflict",
                    details={
                        "incident_id": (
                            incident.incident_id
                        ),

                        "expected_version": (
                            expected_version
                        ),

                        "actual_version": (
                            current.version
                        ),
                    },
                )

            self._records[
                incident.incident_id
            ] = deepcopy(
                incident
            )

            return deepcopy(
                incident
            )

    async def find_by_idempotency_key(
        self,
        key: str,
    ) -> Optional[IncidentRecord]:
        normalized = str(
            key
        ).strip()

        if not normalized:
            return None

        async with self._lock:
            self._ensure_open()

            incident_id = (
                self._idempotency_index.get(
                    normalized
                )
            )

            if incident_id is None:
                return None

            return deepcopy(
                self._records.get(
                    incident_id
                )
            )

    async def close(
        self,
    ) -> None:
        async with self._lock:
            self._closed = True

            self._records.clear()
            self._idempotency_index.clear()


# ------------------------------------------------------------------
# Repository compatibility
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
# Request translation
# ------------------------------------------------------------------

def signal_record(
    request: IncidentSignalRequest,
) -> IncidentSignalRecord:
    record = IncidentSignalRecord(
        signal_id=request.signal_id,

        source=request.source,

        signal_type=request.signal_type,

        observed_at=normalize_datetime(
            request.observed_at
        ),

        severity=request.severity,

        payload=safe_attributes(
            request.payload,
            maximum_entries=(
                MAX_SIGNAL_ATTRIBUTES
            ),
        ),

        attributes=safe_attributes(
            request.attributes,
            maximum_entries=(
                MAX_SIGNAL_ATTRIBUTES
            ),
        ),
    )

    record.validate()

    return record


def incident_record_from_request(
    request: IncidentCreateRequest,
    *,
    context: Optional[
        IncidentOperationContext
    ] = None,
) -> IncidentRecord:
    now = utc_now()

    context_key = (
        context.idempotency_key
        if context is not None
        else None
    )

    idempotency_key = (
        request.idempotency_key
        or context_key
    )

    signals = tuple(
        signal_record(
            signal
        )
        for signal
        in request.signals
    )

    signal_ids = [
        signal.signal_id
        for signal
        in signals
    ]

    if (
        len(
            signal_ids
        )
        != len(
            set(
                signal_ids
            )
        )
    ):
        raise ValueError(
            "incident signal IDs must be unique"
        )

    attributes = safe_attributes(
        request.attributes,
        maximum_entries=(
            MAX_INCIDENT_ATTRIBUTES
        ),
    )

    attributes.setdefault(
        "source",
        "api",
    )

    record = IncidentRecord(
        incident_id=(
            "inc-"
            + uuid4().hex
        ),

        title=request.title,

        summary=request.summary,

        severity=request.severity,

        status=(
            IncidentStatusAPI.
            OPEN.value
        ),

        service=request.service,

        environment=(
            request.environment
        ),

        signals=signals,

        attributes=attributes,

        created_at=now,

        updated_at=now,

        version=1,

        tenant_id=(
            context.tenant_id
            if context is not None
            else None
        ),

        created_by=(
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
    )

    record.validate()

    return record


# ------------------------------------------------------------------
# Public response conversion
# ------------------------------------------------------------------

def incident_response(
    incident: IncidentRecord,
) -> IncidentResponse:
    incident.validate()

    public_attributes = safe_attributes(
        incident.attributes,
        maximum_entries=(
            MAX_INCIDENT_ATTRIBUTES
        ),
    )

    public_attributes.update(
        {
            "version": (
                incident.version
            ),

            "correlation_id": (
                incident.correlation_id
            ),
        }
    )

    return IncidentResponse(
        incident_id=(
            incident.incident_id
        ),

        title=incident.title,

        summary=incident.summary,

        severity=incident.severity,

        status=incident.status,

        service=incident.service,

        environment=(
            incident.environment
        ),

        signal_count=len(
            incident.signals
        ),

        created_at=(
            incident.created_at
        ),

        updated_at=(
            incident.updated_at
        ),

        attributes=(
            public_attributes
        ),
    )


# ------------------------------------------------------------------
# Status transitions
# ------------------------------------------------------------------

_ALLOWED_STATUS_TRANSITIONS: Dict[
    str,
    Tuple[str, ...],
] = {
    IncidentStatusAPI.OPEN.value: (
        IncidentStatusAPI.
        INVESTIGATING.value,

        IncidentStatusAPI.
        MITIGATED.value,

        IncidentStatusAPI.
        RESOLVED.value,

        IncidentStatusAPI.
        CLOSED.value,
    ),

    IncidentStatusAPI.
    INVESTIGATING.value: (
        IncidentStatusAPI.
        MITIGATED.value,

        IncidentStatusAPI.
        RESOLVED.value,

        IncidentStatusAPI.
        CLOSED.value,
    ),

    IncidentStatusAPI.
    MITIGATED.value: (
        IncidentStatusAPI.
        INVESTIGATING.value,

        IncidentStatusAPI.
        RESOLVED.value,

        IncidentStatusAPI.
        CLOSED.value,
    ),

    IncidentStatusAPI.
    RESOLVED.value: (
        IncidentStatusAPI.
        INVESTIGATING.value,

        IncidentStatusAPI.
        CLOSED.value,
    ),

    IncidentStatusAPI.
    CLOSED.value: (),
}


def validate_status_transition(
    current: str,
    target: str,
) -> None:
    if current == target:
        return

    allowed = (
        _ALLOWED_STATUS_TRANSITIONS.get(
            current,
            (),
        )
    )

    if target not in allowed:
        raise ResourceConflictError(
            "The incident status transition "
            "is not allowed.",
            details={
                "resource": (
                    "incident"
                ),

                "state": current,
                "status": target,
            },
        )


# ------------------------------------------------------------------
# Incident service
# ------------------------------------------------------------------

class IncidentService:
    """
    Process-scoped incident application service.
    """

    def __init__(
        self,
        *,
        config: APIConfig,
        repository: IncidentRepository,
    ) -> None:
        config.validate()

        self.config = config
        self.repository = repository

        self._closed = False
        self._lifecycle_lock = (
            asyncio.Lock()
        )

    # ------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------

    def _ensure_open(
        self,
    ) -> None:
        if self._closed:
            raise InternalServerAPIError(
                "The incident service is closed.",
                details={
                    "resource": (
                        INCIDENT_SERVICE_NAME
                    ),
                },
            )

    def _require_read_enabled(
        self,
    ) -> None:
        if not (
            self.config.features.
            incident_read_enabled
        ):
            raise FeatureDisabledError(
                details={
                    "resource": (
                        "incident_read"
                    ),
                },
            )

    def _require_create_enabled(
        self,
    ) -> None:
        if not (
            self.config.features.
            incident_create_enabled
        ):
            raise FeatureDisabledError(
                details={
                    "resource": (
                        "incident_create"
                    ),
                },
            )

    async def close(
        self,
    ) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return

            await await_if_needed(
                self.repository.close()
            )

            self._closed = True

    # ------------------------------------------------------
    # Create
    # ------------------------------------------------------

    async def create_incident(
        self,
        request: IncidentCreateRequest,
        *,
        context: Optional[
            IncidentOperationContext
        ] = None,
    ) -> IncidentResponse:
        self._ensure_open()
        self._require_create_enabled()

        idempotency_key = (
            request.idempotency_key
            or (
                context.idempotency_key
                if context is not None
                else None
            )
        )

        if idempotency_key:
            existing = await await_if_needed(
                self.repository.
                find_by_idempotency_key(
                    idempotency_key
                )
            )

            if existing is not None:
                return incident_response(
                    existing
                )

        record = (
            incident_record_from_request(
                request,
                context=context,
            )
        )

        try:
            created = await await_if_needed(
                self.repository.create(
                    record
                )
            )

            logger.info(
                "Incident created",
                extra={
                    "incident": {
                        "incident_id": (
                            created.incident_id
                        ),

                        "severity": (
                            created.severity
                        ),

                        "service": (
                            created.service
                        ),

                        "request_id": (
                            context.request_id
                            if context is not None
                            else None
                        ),
                    },
                },
            )

            return incident_response(
                created
            )

        except APIException:
            raise

        except Exception as exc:
            raise InternalServerAPIError(
                "The incident could not be created.",
                details={
                    "operation": (
                        "incident_create"
                    ),

                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                },
            ) from exc

    # ------------------------------------------------------
    # Get
    # ------------------------------------------------------

    async def get_incident(
        self,
        incident_id: str,
        *,
        context: Optional[
            IncidentOperationContext
        ] = None,
    ) -> IncidentResponse:
        self._ensure_open()
        self._require_read_enabled()

        normalized = str(
            incident_id
        ).strip()

        if not normalized:
            raise IncidentNotFoundAPIError(
                details={
                    "resource": (
                        "incident"
                    ),
                },
            )

        incident = await await_if_needed(
            self.repository.get(
                normalized
            )
        )

        if incident is None:
            raise IncidentNotFoundAPIError(
                details={
                    "resource": (
                        "incident"
                    ),

                    "incident_id": (
                        normalized
                    ),
                },
            )

        if (
            context is not None
            and context.tenant_id is not None
            and incident.tenant_id
            != context.tenant_id
        ):
            # Return not found to avoid revealing
            # cross-tenant resource existence.
            raise IncidentNotFoundAPIError(
                details={
                    "resource": (
                        "incident"
                    ),

                    "incident_id": (
                        normalized
                    ),
                },
            )

        return incident_response(
            incident
        )

    # ------------------------------------------------------
    # List
    # ------------------------------------------------------

    async def list_incidents(
        self,
        *,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        service: Optional[str] = None,
        environment: Optional[str] = None,
        created_after: Optional[
            datetime
        ] = None,
        created_before: Optional[
            datetime
        ] = None,
        limit: int = 100,
        offset: int = 0,
        descending: bool = True,
        context: Optional[
            IncidentOperationContext
        ] = None,
    ) -> IncidentListResponse:
        self._ensure_open()
        self._require_read_enabled()

        query = IncidentQuery(
            status=status,
            severity=severity,
            service=service,
            environment=environment,

            tenant_id=(
                context.tenant_id
                if context is not None
                else None
            ),

            created_after=(
                created_after
            ),

            created_before=(
                created_before
            ),

            limit=limit,
            offset=offset,
            descending=descending,
        )

        query.validate()

        try:
            incidents, total = (
                await await_if_needed(
                    self.repository.list(
                        query
                    )
                )
            )

        except APIException:
            raise

        except Exception as exc:
            raise InternalServerAPIError(
                "Incidents could not be listed.",
                details={
                    "operation": (
                        "incident_list"
                    ),

                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                },
            ) from exc

        returned = len(
            incidents
        )

        return IncidentListResponse(
            incidents=[
                incident_response(
                    incident
                )
                for incident
                in incidents
            ],

            pagination=(
                PaginationMetadata(
                    limit=limit,
                    offset=offset,
                    returned=returned,
                    total=total,
                    has_more=(
                        offset
                        + returned
                        < total
                    ),
                )
            ),

            metadata=(
                APIResponseMetadata(
                    request_id=(
                        context.request_id
                        if context is not None
                        else None
                    ),
                )
            ),
        )

    # ------------------------------------------------------
    # Status update
    # ------------------------------------------------------

    async def update_status(
        self,
        incident_id: str,
        status: str,
        *,
        expected_version: int,
        context: Optional[
            IncidentOperationContext
        ] = None,
    ) -> IncidentResponse:
        self._ensure_open()
        self._require_create_enabled()

        if expected_version < 1:
            raise ValueError(
                "expected_version must be >= 1"
            )

        if (
            status
            not in {
                value.value
                for value
                in IncidentStatusAPI
            }
        ):
            raise ValueError(
                "Unsupported incident status: "
                f"{status}"
            )

        normalized = str(
            incident_id
        ).strip()

        current = await await_if_needed(
            self.repository.get(
                normalized
            )
        )

        if current is None:
            raise IncidentNotFoundAPIError(
                details={
                    "incident_id": normalized,
                },
            )

        if (
            context is not None
            and context.tenant_id is not None
            and current.tenant_id
            != context.tenant_id
        ):
            raise IncidentNotFoundAPIError(
                details={
                    "incident_id": normalized,
                },
            )

        validate_status_transition(
            current.status,
            status,
        )

        if current.status == status:
            return incident_response(
                current
            )

        attributes = dict(
            current.attributes
        )

        attributes[
            "last_updated_by"
        ] = (
            context.actor_id
            if context is not None
            else None
        )

        updated = replace(
            current,

            status=status,

            updated_at=utc_now(),

            version=(
                current.version
                + 1
            ),

            attributes=attributes,
        )

        stored = await await_if_needed(
            self.repository.update(
                updated,
                expected_version=(
                    expected_version
                ),
            )
        )

        logger.info(
            "Incident status changed",
            extra={
                "incident": {
                    "incident_id": (
                        stored.incident_id
                    ),

                    "previous_status": (
                        current.status
                    ),

                    "status": (
                        stored.status
                    ),

                    "version": (
                        stored.version
                    ),

                    "request_id": (
                        context.request_id
                        if context is not None
                        else None
                    ),
                },
            },
        )

        return incident_response(
            stored
        )

    # ------------------------------------------------------
    # Signals
    # ------------------------------------------------------

    async def signals(
        self,
        incident_id: str,
        *,
        context: Optional[
            IncidentOperationContext
        ] = None,
    ) -> Tuple[
        IncidentSignalRecord,
        ...,
    ]:
        self._ensure_open()
        self._require_read_enabled()

        normalized = str(
            incident_id
        ).strip()

        incident = await await_if_needed(
            self.repository.get(
                normalized
            )
        )

        if incident is None:
            raise IncidentNotFoundAPIError(
                details={
                    "incident_id": normalized,
                },
            )

        if (
            context is not None
            and context.tenant_id is not None
            and incident.tenant_id
            != context.tenant_id
        ):
            raise IncidentNotFoundAPIError(
                details={
                    "incident_id": normalized,
                },
            )

        return tuple(
            deepcopy(
                incident.signals
            )
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

            "persistent": not isinstance(
                self.repository,
                ProcessLocalIncidentRepository,
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
                INCIDENT_SERVICE_NAME
            ),

            "version": (
                INCIDENT_SERVICE_VERSION
            ),

            "closed": (
                self._closed
            ),

            "read_enabled": (
                self.config.
                features.
                incident_read_enabled
            ),

            "create_enabled": (
                self.config.
                features.
                incident_create_enabled
            ),

            "repository": type(
                self.repository
            ).__name__,

            "persistent_repository": (
                not isinstance(
                    self.repository,
                    ProcessLocalIncidentRepository,
                )
            ),

            "correlation_supported": False,

            "root_cause_analysis_supported": (
                False
            ),
        }


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def create_incident_service(
    *,
    config: APIConfig,
    repository: Optional[
        IncidentRepository
    ] = None,
) -> IncidentService:
    resolved_repository = (
        repository
        or ProcessLocalIncidentRepository()
    )

    return IncidentService(
        config=config,
        repository=(
            resolved_repository
        ),
    )


# ------------------------------------------------------------------
# Dependency registration
# ------------------------------------------------------------------

def register_incident_service(
    dependencies: Any,
    *,
    repository: Optional[
        IncidentRepository
    ] = None,
    replace: bool = False,
) -> IncidentService:
    """
    Construct and register IncidentService without
    importing APIDependencies here.
    """

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

    service = create_incident_service(
        config=config,
        repository=repository,
    )

    register(
        INCIDENT_SERVICE_NAME,
        service,
        replace=replace,
    )

    return service


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "INCIDENT_SERVICE_NAME",
    "INCIDENT_SERVICE_VERSION",

    "IncidentOperationContext",

    "IncidentSignalRecord",
    "IncidentRecord",
    "IncidentQuery",

    "IncidentRepository",
    "ProcessLocalIncidentRepository",

    "safe_attributes",
    "signal_record",
    "incident_record_from_request",
    "incident_response",

    "validate_status_transition",

    "IncidentService",

    "create_incident_service",
    "register_incident_service",
]
