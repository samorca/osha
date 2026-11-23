from __future__ import annotations

"""
Production model-management service for OSHA API.

Responsibilities:

    list registered serving models
    retrieve one model
    refresh promoted-model discovery
    load a registered model
    unload a registered model
    translate internal model descriptors into
    stable public API responses
    enforce model API feature gates
    apply model-operation timeouts
    serialize conflicting model mutations

Architecture:

    routes/models.py
            |
            v
    ModelService
            |
            +--> AIModelRegistry
            |
            +--> AIModelLoader
            |
            +--> model adapters
                    |
                    v
             AcceleratorManager

This service manages serving state only.

It does not:

    train models
    evaluate model quality
    promote candidate models
    expose checkpoint paths
    expose artifact filesystem paths
    select accelerators directly
    execute inference
    execute remediation
"""

import asyncio
import inspect
import logging
import time

from dataclasses import (
    dataclass,
)

from datetime import (
    datetime,
    timezone,
)

from threading import (
    RLock,
)

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from uuid import (
    uuid4,
)

from src.ai.loader import (
    AIModelLoader,
)

from src.ai.registry import (
    AIModelRegistry,
)

from src.api.config import (
    APIConfig,
)

from src.api.errors import (
    APIException,
    FeatureDisabledError,
    InternalServerAPIError,
    ModelLoadAPIError,
    ModelNotFoundAPIError,
    ModelUnavailableAPIError,
    RequestTimeoutAPIError,
    ResourceConflictError,
    translate_exception,
)

from src.api.schema import (
    APIResponseMetadata,
    ModelListResponse,
    ModelOperationRequest,
    ModelRefreshRequest,
    ModelRefreshResponse,
    ModelResponse,
    OperationResponse,
    OperationState,
    PaginationMetadata,
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

MODEL_SERVICE_NAME = "models"

MODEL_SERVICE_VERSION = "1"


# ------------------------------------------------------------------
# Operation context
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ModelOperationContext:
    """
    Request-scoped information supplied by a route.

    The service does not depend on FastAPI Request.
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
# Generic object helpers
# ------------------------------------------------------------------

def _mapping(
    value: Any,
) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(
        value,
        Mapping,
    ):
        return dict(
            value
        )

    converter = getattr(
        value,
        "to_dict",
        None,
    )

    if callable(
        converter
    ):
        converted = converter()

        if isinstance(
            converted,
            Mapping,
        ):
            return dict(
                converted
            )

    dumper = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(
        dumper
    ):
        converted = dumper()

        if isinstance(
            converted,
            Mapping,
        ):
            return dict(
                converted
            )

    return {}


def _value(
    source: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(
        source,
        Mapping,
    ):
        return source.get(
            name,
            default,
        )

    return getattr(
        source,
        name,
        default,
    )


def _nested_value(
    source: Any,
    path: Sequence[str],
    default: Any = None,
) -> Any:
    current = source

    for name in path:
        if current is None:
            return default

        current = _value(
            current,
            name,
            None,
        )

    return (
        default
        if current is None
        else current
    )


def _text(
    value: Any,
    *,
    default: str = "",
) -> str:
    if value is None:
        return default

    return str(
        value
    ).strip()


def _optional_datetime(
    value: Any,
) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(
        value,
        datetime,
    ):
        return value

    text = str(
        value
    ).strip()

    if not text:
        return None

    try:
        return datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        )

    except ValueError:
        return None


def _utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def _operation_id() -> str:
    return str(
        uuid4()
    )


def _safe_public_attributes(
    value: Any,
) -> Dict[str, Any]:
    """
    Return bounded serving metadata while
    excluding path-like or credential-like keys.
    """

    source = _mapping(
        value
    )

    blocked_fragments = {
        "path",
        "token",
        "secret",
        "password",
        "credential",
        "authorization",
        "cookie",
        "private",
        "certificate",
        "environment",
        "traceback",
        "stack",
    }

    output: Dict[str, Any] = {}

    for raw_key, item in list(
        source.items()
    )[
        :128
    ]:
        key = str(
            raw_key
        ).strip()

        normalized = key.lower()

        if (
            not key
            or any(
                fragment in normalized
                for fragment
                in blocked_fragments
            )
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
                    :64
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
            nested: Dict[str, Any] = {}

            for nested_key, nested_value in list(
                item.items()
            )[
                :64
            ]:
                nested_name = str(
                    nested_key
                ).strip()

                nested_normalized = (
                    nested_name.lower()
                )

                if (
                    not nested_name
                    or any(
                        fragment
                        in nested_normalized
                        for fragment
                        in blocked_fragments
                    )
                ):
                    continue

                if isinstance(
                    nested_value,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    ),
                ):
                    nested[
                        nested_name
                    ] = nested_value

            output[
                key
            ] = nested

    return output


# ------------------------------------------------------------------
# Async compatibility
# ------------------------------------------------------------------

async def _await_if_needed(
    value: Any,
) -> Any:
    if inspect.isawaitable(
        value
    ):
        return await value

    return value


def _invoke_supported(
    method: Callable[..., Any],
    arguments: Mapping[str, Any],
) -> Any:
    """
    Invoke a method using only supported keyword
    arguments.
    """

    try:
        signature = inspect.signature(
            method
        )

    except (
        TypeError,
        ValueError,
    ):
        return method(
            **dict(
                arguments
            )
        )

    parameters = signature.parameters

    accepts_kwargs = any(
        parameter.kind
        == inspect.Parameter.VAR_KEYWORD
        for parameter
        in parameters.values()
    )

    if accepts_kwargs:
        values = dict(
            arguments
        )

    else:
        values = {
            name: value
            for name, value
            in arguments.items()
            if name in parameters
        }

    return method(
        **values
    )


# ------------------------------------------------------------------
# Registry entry helpers
# ------------------------------------------------------------------

def _descriptor(
    value: Any,
) -> Any:
    return _value(
        value,
        "descriptor",
        value,
    )


def model_id_of(
    value: Any,
) -> str:
    descriptor = _descriptor(
        value
    )

    return _text(
        _value(
            value,
            "model_id",
            _nested_value(
                descriptor,
                (
                    "identity",
                    "model_id",
                ),
                _value(
                    descriptor,
                    "model_id",
                    "",
                ),
            ),
        )
    )


def model_name_of(
    value: Any,
) -> str:
    descriptor = _descriptor(
        value
    )

    return _text(
        _nested_value(
            descriptor,
            (
                "identity",
                "name",
            ),
            _value(
                descriptor,
                "name",
                "",
            ),
        )
    )


def model_family_of(
    value: Any,
) -> str:
    descriptor = _descriptor(
        value
    )

    return _text(
        _nested_value(
            descriptor,
            (
                "identity",
                "family",
            ),
            _value(
                descriptor,
                "family",
                "",
            ),
        )
    )


def model_version_of(
    value: Any,
) -> str:
    descriptor = _descriptor(
        value
    )

    return _text(
        _nested_value(
            descriptor,
            (
                "identity",
                "version",
            ),
            _value(
                descriptor,
                "version",
                "",
            ),
        )
    )


def model_state_of(
    value: Any,
) -> str:
    descriptor = _descriptor(
        value
    )

    return _text(
        _value(
            descriptor,
            "state",
            _value(
                value,
                "state",
                "unknown",
            ),
        ),
        default="unknown",
    )


def model_ready_of(
    value: Any,
) -> bool:
    ready = _value(
        value,
        "ready",
        None,
    )

    if ready is not None:
        return bool(
            ready
        )

    descriptor = _descriptor(
        value
    )

    ready = _value(
        descriptor,
        "ready",
        None,
    )

    if ready is not None:
        return bool(
            ready
        )

    return model_state_of(
        value
    ) == "ready"


def model_loaded_of(
    value: Any,
) -> bool:
    loaded = _value(
        value,
        "loaded",
        None,
    )

    if loaded is not None:
        return bool(
            loaded
        )

    adapter = _value(
        value,
        "adapter",
        None,
    )

    if adapter is not None:
        loaded = _value(
            adapter,
            "loaded",
            None,
        )

        if loaded is not None:
            return bool(
                loaded
            )

    return model_state_of(
        value
    ) in {
        "loading",
        "ready",
        "degraded",
    }


def model_device_of(
    value: Any,
) -> Optional[str]:
    descriptor = _descriptor(
        value
    )

    device = _value(
        descriptor,
        "device",
        _value(
            value,
            "device",
            None,
        ),
    )

    if device is None:
        return None

    text = _text(
        device
    )

    return (
        text
        if text
        else None
    )


def model_response(
    value: Any,
) -> ModelResponse:
    descriptor = _descriptor(
        value
    )

    model_id = model_id_of(
        value
    )

    name = model_name_of(
        value
    )

    family = model_family_of(
        value
    )

    version = model_version_of(
        value
    )

    if not model_id:
        raise InternalServerAPIError(
            "Registered model has no "
            "public model identifier.",
            details={
                "resource": (
                    "model_registry_entry"
                ),
            },
        )

    if not name:
        name = model_id

    if not family:
        family = "unknown"

    if not version:
        version = "unknown"

    return ModelResponse(
        model_id=model_id,

        name=name,

        family=family,

        version=version,

        state=model_state_of(
            value
        ),

        ready=model_ready_of(
            value
        ),

        loaded=model_loaded_of(
            value
        ),

        device=model_device_of(
            value
        ),

        artifact_path_exposed=False,

        model_configuration=_safe_public_attributes(
            _value(
                descriptor,
                "model_config",
                {},
            )
        ),

        attributes=_safe_public_attributes(
            _value(
                descriptor,
                "attributes",
                _value(
                    value,
                    "attributes",
                    {},
                ),
            )
        ),

        registered_at=_optional_datetime(
            _value(
                descriptor,
                "registered_at",
                _value(
                    value,
                    "registered_at",
                    None,
                ),
            )
        ),

        loaded_at=_optional_datetime(
            _value(
                descriptor,
                "loaded_at",
                _value(
                    value,
                    "loaded_at",
                    None,
                ),
            )
        ),
    )


# ------------------------------------------------------------------
# Model service
# ------------------------------------------------------------------

class ModelService:
    """
    Process-scoped serving-model management.

    Mutation operations are serialized per model
    to prevent concurrent load/unload conflicts.
    """

    def __init__(
        self,
        *,
        config: APIConfig,
        registry: AIModelRegistry,
        loader: AIModelLoader,
    ) -> None:
        config.validate()

        self.config = config
        self.registry = registry
        self.loader = loader

        self._closed = False
        self._lock = RLock()

        self._operation_semaphore = (
            asyncio.Semaphore(
     config.
                limits.
                maximum_concurrent_model_operations
            )
        )

        self._model_locks: Dict[
            str,
            asyncio.Lock,
        ] = {}

        self._model_locks_guard = (
            asyncio.Lock()
        )

        self._idempotency_results: Dict[
            str,
            OperationResponse,
        ] = {}

    # ------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------

    def _ensure_open(
        self,
    ) -> None:
        with self._lock:
            if self._closed:
                raise ModelUnavailableAPIError(
                    "The model service is closed.",
                    details={
                        "resource": (
                            MODEL_SERVICE_NAME
                        ),
                    },
                )

    def _require_read_enabled(
        self,
    ) -> None:
        if not (
            self.config.
            features.
            model_read_enabled
        ):
            raise FeatureDisabledError(
                details={
                    "resource": (
                        "model_read"
                    ),
                },
            )

    def _require_operations_enabled(
        self,
    ) -> None:
        if not (
            self.config.
            features.
            model_load_operations_enabled
        ):
            raise FeatureDisabledError(
                details={
                    "resource": (
                        "model_operations"
                    ),
                },
            )

    async def close(
        self,
    ) -> None:
        with self._lock:
            self._closed = True
            self._idempotency_results.clear()

        async with self._model_locks_guard:
            self._model_locks.clear()

    # ------------------------------------------------------
    # Registry enumeration
    # ------------------------------------------------------

    def _all_entries(
        self,
    ) -> Tuple[Any, ...]:
        finder = getattr(
            self.registry,
            "find",
            None,
        )

        if callable(
            finder
        ):
            try:
                result = finder()

                if result is not None:
                    return tuple(
                        result
                    )

            except TypeError:
                pass

        entries = getattr(
            self.registry,
            "entries",
            None,
        )

        if isinstance(
            entries,
            Mapping,
        ):
            return tuple(
                entries.values()
            )

        if isinstance(
            entries,
            Iterable,
        ):
            return tuple(
                entries
            )

        lister = getattr(
            self.registry,
            "list",
            None,
        )

        if callable(
            lister
        ):
            result = lister()

            if result is not None:
                return tuple(
                    result
                )

        models = getattr(
            self.registry,
            "models",
            None,
        )

        if isinstance(
            models,
            Mapping,
        ):
            return tuple(
                models.values()
            )

        if isinstance(
            models,
            Iterable,
        ):
            return tuple(
                models
            )

        return ()

    def _filtered_entries(
        self,
        *,
        family: Optional[str] = None,
        state: Optional[str] = None,
        loaded: Optional[bool] = None,
        ready: Optional[bool] = None,
    ) -> Tuple[Any, ...]:
        entries = self._all_entries()

        family_filter = (
            _text(
                family
            ).lower()
            if family
            else None
        )

        state_filter = (
            _text(
                state
            ).lower()
            if state
            else None
        )

        output = []

        for entry in entries:
            if (
                family_filter
                and model_family_of(
                    entry
                ).lower()
                != family_filter
            ):
                continue

            if (
                state_filter
                and model_state_of(
                    entry
                ).lower()
                != state_filter
            ):
                continue

            if (
                loaded is not None
                and model_loaded_of(
                    entry
                )
                != loaded
            ):
                continue

            if (
                ready is not None
                and model_ready_of(
                    entry
                )
                != ready
            ):
                continue

            output.append(
                entry
            )

        return tuple(
            sorted(
                output,
                key=lambda entry: (
                    model_family_of(
                        entry
                    ),
                    model_name_of(
                        entry
                    ),
                    model_version_of(
                        entry
                    ),
                    model_id_of(
                        entry
                    ),
                ),
            )
        )

    # ------------------------------------------------------
    # Lookup
    # ------------------------------------------------------

    def _find_entry(
        self,
        model_id: str,
    ) -> Any:
        normalized = _text(
            model_id
        )

        if not normalized:
            raise ModelNotFoundAPIError(
                details={
                    "resource": (
                        "model"
                    ),
                },
            )

        getter = getattr(
            self.registry,
            "get",
            None,
        )

        if callable(
            getter
        ):
            try:
                entry = getter(
                    normalized
                )

                if entry is not None:
                    return entry

            except (
                KeyError,
                LookupError,
            ):
                pass

        for entry in self._all_entries():
            if (
                model_id_of(
                    entry
                )
                == normalized
            ):
                return entry

        raise ModelNotFoundAPIError(
            details={
                "resource": (
                    "model"
                ),

                "model_id": (
                    normalized
                ),
            },
        )

    # ------------------------------------------------------
    # Read operations
    # ------------------------------------------------------

    def list_models(
        self,
        *,
        family: Optional[str] = None,
        state: Optional[str] = None,
        loaded: Optional[bool] = None,
        ready: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
        context: Optional[
            ModelOperationContext
        ] = None,
    ) -> ModelListResponse:
        self._ensure_open()
        self._require_read_enabled()

        if limit < 1 or limit > 1000:
            raise ValueError(
                "limit must be between "
                "1 and 1000"
            )

        if offset < 0:
            raise ValueError(
                "offset must be >= 0"
            )

        entries = self._filtered_entries(
            family=family,
            state=state,
            loaded=loaded,
            ready=ready,
        )

        total = len(
            entries
        )

        selected = entries[
            offset:
            offset + limit
        ]

        returned = len(
            selected
        )

        return ModelListResponse(
            models=[
                model_response(
                    entry
                )
                for entry
                in selected
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
                        if context
                        is not None
                        else None
                    ),
                )
            ),
        )

    def get_model(
        self,
        model_id: str,
        *,
        context: Optional[
            ModelOperationContext
        ] = None,
    ) -> ModelResponse:
        self._ensure_open()
        self._require_read_enabled()

        return model_response(
            self._find_entry(
                model_id
            )
        )

    # ------------------------------------------------------
    # Per-model lock
    # ------------------------------------------------------

    async def _model_lock(
        self,
        model_id: str,
    ) -> asyncio.Lock:
        normalized = _text(
            model_id
        )

        async with self._model_locks_guard:
            lock = self._model_locks.get(
                normalized
            )

            if lock is None:
                lock = asyncio.Lock()

                self._model_locks[
                    normalized
                ] = lock

            return lock

    # ------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------

    @staticmethod
    def _idempotency_key(
        request: ModelOperationRequest,
        context: Optional[
            ModelOperationContext
        ],
    ) -> Optional[str]:
        key = (
            request.idempotency_key
            or (
                context.idempotency_key
                if context is not None
                else None
            )
        )

        if not key:
            return None

        return _text(
            key
        )

    def _cached_operation(
        self,
        key: Optional[str],
    ) -> Optional[OperationResponse]:
        if not key:
            return None

        with self._lock:
            return self._idempotency_results.get(
                key
            )

    def _store_operation(
        self,
        key: Optional[str],
        response: OperationResponse,
    ) -> None:
        if not key:
            return

        with self._lock:
            if (
                len(
                    self._idempotency_results
                )
                >= 10_000
            ):
                oldest = next(
                    iter(
                        self.
                        _idempotency_results
                    )
                )

                self._idempotency_results.pop(
                    oldest,
                    None,
                )

            self._idempotency_results[
                key
            ] = response

    # ------------------------------------------------------
    # Timed mutation
    # ------------------------------------------------------

    async def _run_operation(
        self,
        operation: Callable[
            [],
            Any,
        ],
        *,
        model_id: str,
        operation_name: str,
    ) -> Any:
        timeout = (
            self.config.
            timeouts.
            model_operation_seconds
        )

        lock = await self._model_lock(
            model_id
        )

        try:
            async with self._operation_semaphore:
                async with lock:
                    return await asyncio.wait_for(
                        _await_if_needed(
                            operation()
                        ),
                        timeout=timeout,
                    )

        except asyncio.TimeoutError as exc:
            raise RequestTimeoutAPIError(
                "The model operation timed out.",
                details={
                    "operation": (
                        operation_name
                    ),

                    "model_id": (
                        model_id
                    ),
                },
            ) from exc

    # ------------------------------------------------------
    # Load
    # ------------------------------------------------------

    async def load_model(
        self,
        model_id: str,
        request: Optional[
            ModelOperationRequest
        ] = None,
        *,
        context: Optional[
            ModelOperationContext
        ] = None,
    ) -> OperationResponse:
        self._ensure_open()
        self._require_operations_enabled()

        normalized = _text(
            model_id
        )

        operation_request = (
            request
            or ModelOperationRequest()
        )

        key = self._idempotency_key(
            operation_request,
            context,
        )

        cached = self._cached_operation(
            key
        )

        if cached is not None:
            return cached

        entry = self._find_entry(
            normalized
        )

        if (
            model_loaded_of(
                entry
            )
            and not operation_request.force
        ):
            response = OperationResponse(
                operation_id=uuid4(),

                state=(
                    OperationState.
                    COMPLETED.value
                ),

                message=(
                    "The model is already loaded."
                ),

                result={
                    "model": (
                        model_response(
                            entry
                        ).model_dump(
                            mode="json",
                            exclude_none=True,
                        )
                    ),

                    "changed": False,
                },

                metadata=APIResponseMetadata(
                    request_id=(
                        context.request_id
                        if context is not None
                        else None
                    ),
                ),
            )

            self._store_operation(
                key,
                response,
            )

            return response

        def operation() -> Any:
            loader = getattr(
                self.registry,
                "load",
                None,
            )

            if callable(
                loader
            ):
                try:
                    return loader(
                        normalized,
                        force=(
                            operation_request.force
                        ),
                    )

                except TypeError:
                    return loader(
                        normalized
                    )

            loader = getattr(
                self.loader,
                "load",
                None,
            )

            if callable(
                loader
            ):
                try:
                    return loader(
                        normalized,
                        force=(
                            operation_request.force
                        ),
                    )

                except TypeError:
                    return loader(
                        normalized
                    )

            raise ModelUnavailableAPIError(
                "No model load operation "
                "is available.",
                details={
                    "model_id": (
                        normalized
                    ),
                },
            )

        try:
            await self._run_operation(
                operation,
                model_id=normalized,
                operation_name=(
                    "model_load"
                ),
            )

            refreshed = self._find_entry(
                normalized
            )

            response = OperationResponse(
                operation_id=uuid4(),

                state=(
                    OperationState.
                    COMPLETED.value
                ),

                message=(
                    "The model was loaded."
                ),

                result={
                    "model": (
                        model_response(
                            refreshed
                        ).model_dump(
                            mode="json",
                            exclude_none=True,
                        )
                    ),

                    "changed": True,
                },

                metadata=APIResponseMetadata(
                    request_id=(
                        context.request_id
                        if context is not None
                        else None
                    ),
                ),
            )

            self._store_operation(
                key,
                response,
            )

            return response

        except APIException:
            raise

        except Exception as exc:
            translated = translate_exception(
                exc
            )

            if (
                translated.status_code
                < 500
            ):
                raise translated from exc

            raise ModelLoadAPIError(
                details={
                    "operation": (
                        "model_load"
                    ),

                    "model_id": (
                        normalized
                    ),

                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                },
            ) from exc

    # ------------------------------------------------------
    # Unload
    # ------------------------------------------------------

    async def unload_model(
        self,
        model_id: str,
        request: Optional[
            ModelOperationRequest
        ] = None,
        *,
        context: Optional[
            ModelOperationContext
        ] = None,
    ) -> OperationResponse:
        self._ensure_open()
        self._require_operations_enabled()

        normalized = _text(
            model_id
        )

        operation_request = (
            request
            or ModelOperationRequest()
        )

        key = self._idempotency_key(
            operation_request,
            context,
        )

        cached = self._cached_operation(
            key
        )

        if cached is not None:
            return cached

        entry = self._find_entry(
            normalized
        )

        if (
            not model_loaded_of(
                entry
            )
            and not operation_request.force
        ):
            response = OperationResponse(
                operation_id=uuid4(),

                state=(
                    OperationState.
                    COMPLETED.value
                ),

                message=(
                    "The model is already unloaded."
                ),

                result={
                    "model": (
                        model_response(
                            entry
                        ).model_dump(
                            mode="json",
                            exclude_none=True,
                        )
                    ),

                    "changed": False,
                },

                metadata=APIResponseMetadata(
                    request_id=(
                        context.request_id
                        if context is not None
                        else None
                    ),
                ),
            )

            self._store_operation(
                key,
                response,
            )

            return response

        def operation() -> Any:
            unloader = getattr(
                self.registry,
                "unload",
                None,
            )

            if not callable(
                unloader
            ):
                raise ModelUnavailableAPIError(
                    "No model unload operation "
                    "is available.",
                    details={
                        "model_id": (
                            normalized
                        ),
                    },
                )

            try:
                return unloader(
                    normalized,
                    force=(
                        operation_request.force
                    ),
                )

            except TypeError:
                return unloader(
                    normalized
                )

        try:
            await self._run_operation(
                operation,
                model_id=normalized,
                operation_name=(
                    "model_unload"
                ),
            )

            refreshed = self._find_entry(
                normalized
            )

            response = OperationResponse(
                operation_id=uuid4(),

                state=(
                    OperationState.
                    COMPLETED.value
                ),

                message=(
                    "The model was unloaded."
                ),

                result={
                    "model": (
                        model_response(
                            refreshed
                        ).model_dump(
                            mode="json",
                            exclude_none=True,
                        )
                    ),

                    "changed": True,
                },

                metadata=APIResponseMetadata(
                    request_id=(
                        context.request_id
                        if context is not None
                        else None
                    ),
                ),
            )

            self._store_operation(
                key,
                response,
            )

            return response

        except APIException:
            raise

        except Exception as exc:
            raise ModelUnavailableAPIError(
                "The model could not be unloaded.",
                retryable=False,
                details={
                    "operation": (
                        "model_unload"
                    ),

                    "model_id": (
                        normalized
                    ),

                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                },
            ) from exc

    # ------------------------------------------------------
    # Refresh discovery
    # ------------------------------------------------------

    async def refresh_models(
        self,
        request: Optional[
            ModelRefreshRequest
        ] = None,
        *,
        context: Optional[
            ModelOperationContext
        ] = None,
    ) -> ModelRefreshResponse:
        self._ensure_open()
        self._require_operations_enabled()

        refresh_request = (
            request
            or ModelRefreshRequest()
        )

        discover = getattr(
            self.loader,
            "discover",
            None,
        )

        if not callable(
            discover
        ):
            raise ModelUnavailableAPIError(
                "Model discovery is unavailable.",
                details={
                    "resource": (
                        "model_loader"
                    ),
                },
            )

        def operation() -> Any:
            return _invoke_supported(
                discover,
                {
                    "fail_fast": (
                        refresh_request.fail_fast
                    ),

                    "replace": (
                        refresh_request.replace
                    ),

                    "eager_load": (
                        refresh_request.eager_load
                    ),
                },
            )

        try:
            async with self._operation_semaphore:
                result = await asyncio.wait_for(
                    _await_if_needed(
                        operation()
                    ),
                    timeout=(
                        self.config.
                        timeouts.
                        model_operation_seconds
                    ),
                )

        except asyncio.TimeoutError as exc:
            raise RequestTimeoutAPIError(
                "Model discovery timed out.",
                details={
                    "operation": (
                        "model_discovery"
                    ),
                },
            ) from exc

        except APIException:
            raise

        except Exception as exc:
            raise ModelLoadAPIError(
                "Model discovery failed.",
                details={
                    "operation": (
                        "model_discovery"
                    ),

                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                },
            ) from exc

        source = _mapping(
            result
        )

        discovered = int(
            source.get(
                "model_count",
                source.get(
                    "discovered",
                    0,
                ),
            )
            or 0
        )

        registered = int(
            source.get(
                "registered",
                source.get(
                    "registered_count",
                    discovered,
                ),
            )
            or 0
        )

        failed = int(
            source.get(
                "failure_count",
                source.get(
                    "failed",
                    0,
                ),
            )
            or 0
        )

        raw_failures = source.get(
            "failures",
            (),
        )

        failures = []

        for failure in raw_failures or ():
            failure_source = _mapping(
                failure
            )

            failures.append(
                {
                    "model_id": (
                        failure_source.get(
                            "model_id"
                        )
                    ),

                    "error_type": (
                        failure_source.get(
                            "error_type",
                            "ModelDiscoveryError",
                        )
                    ),

                    "message": (
                        "Model discovery failed."
                    ),
                }
            )

        return ModelRefreshResponse(
            discovered=discovered,

            registered=registered,

            failed=failed,

            failures=failures,

            metadata=APIResponseMetadata(
                request_id=(
                    context.request_id
                    if context is not None
                    else None
                ),
            ),
        )

    # ------------------------------------------------------
    # Health
    # ------------------------------------------------------

    def health(
        self,
    ) -> Dict[str, Any]:
        self._ensure_open()

        try:
            summary_method = getattr(
                self.registry,
                "summary",
                None,
            )

            if callable(
                summary_method
            ):
                summary = _mapping(
                    summary_method()
                )

            else:
                summary = {}

            entries = self._all_entries()

            return {
                "status": (
                    "healthy"
                    if entries
                    else "degraded"
                ),

                "ready": bool(
                    entries
                ),

                "registered_models": len(
                    entries
                ),

                "loaded_models": sum(
                    1
                    for entry in entries
                    if model_loaded_of(
                        entry
                    )
                ),

                "ready_models": sum(
                    1
                    for entry in entries
                    if model_ready_of(
                        entry
                    )
                ),

                "registry": summary,
            }

        except Exception as exc:
            return {
                "status": "unhealthy",
                "ready": False,
                "registered_models": 0,
                "loaded_models": 0,
                "ready_models": 0,
                "error_type": (
                    type(
                        exc
                    ).__name__
                ),
            }

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            closed = self._closed

        return {
            "service": (
                MODEL_SERVICE_NAME
            ),

            "version": (
                MODEL_SERVICE_VERSION
            ),

            "closed": closed,

            "read_enabled": (
                self.config.
                features.
                model_read_enabled
            ),

            "operations_enabled": (
                self.config.
                features.
                model_load_operations_enabled
            ),

            "operation_timeout_seconds": (
                self.config.
                timeouts.
                model_operation_seconds
            ),

            "maximum_concurrent_operations": (
                self.config.
                limits.
                maximum_concurrent_model_operations
            ),

            "registry": type(
                self.registry
            ).__name__,

            "loader": type(
                self.loader
            ).__name__,

            "training_supported": False,

            "promotion_supported": False,
        }


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def create_model_service(
    *,
    config: APIConfig,
    registry: AIModelRegistry,
    loader: AIModelLoader,
) -> ModelService:
    return ModelService(
        config=config,
        registry=registry,
        loader=loader,
    )


# ------------------------------------------------------------------
# Dependency registration
# ------------------------------------------------------------------

def register_model_service(
    dependencies: Any,
    *,
    replace: bool = False,
) -> ModelService:
    """
    Construct and register ModelService without
    importing APIDependencies here.
    """

    config = getattr(
        dependencies,
        "config",
        None,
    )

    registry = getattr(
        dependencies,
        "model_registry",
        None,
    )

    loader = getattr(
        dependencies,
        "model_loader",
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

    if registry is None:
        raise InternalServerAPIError(
            "Model registry is unavailable.",
            details={
                "resource": (
                    "model_registry"
                ),
            },
        )

    if loader is None:
        raise InternalServerAPIError(
            "Model loader is unavailable.",
            details={
                "resource": (
                    "model_loader"
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

    service = create_model_service(
        config=config,
        registry=registry,
        loader=loader,
    )

    register(
        MODEL_SERVICE_NAME,
        service,
        replace=replace,
    )

    return service


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "MODEL_SERVICE_NAME",
    "MODEL_SERVICE_VERSION",

    "ModelOperationContext",

    "model_id_of",
    "model_name_of",
    "model_family_of",
    "model_version_of",
    "model_state_of",
    "model_ready_of",
    "model_loaded_of",
    "model_device_of",

    "model_response",

    "ModelService",

    "create_model_service",
    "register_model_service",
]
