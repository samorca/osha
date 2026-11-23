from __future__ import annotations

"""
Accelerator API service for OSHA.

This service translates public HTTP contracts into
the internal accelerator subsystem.

Responsibilities:

    list accelerator inventory
    retrieve one accelerator device
    select an accelerator
    collect backend-specific metrics
    generate backend-specific optimization plans
    translate internal objects into API responses
    enforce accelerator API feature gates
    release temporary accelerator selections

Architecture:

    routes/accelerators.py
            ↓
    AcceleratorService
            ↓
    AcceleratorManager
            ↓
    CPU / CUDA / ROCm backends
            ↓
    devices.py
    metrics.py
    optimizer.py

This module does not:

    define HTTP routes
    construct FastAPI
    mutate optimization settings automatically
    retain temporary selections after requests
    execute training
    execute inference
    clear accelerator caches
"""

import logging

from dataclasses import dataclass

from datetime import datetime

from threading import RLock

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from src.accelerators import (
    AcceleratorManager,
)

from src.accelerators.cpu.metrics import (
    CPUMetricsCollector,
)

from src.accelerators.cpu.optimizer import (
    CPUOptimizer,
)

from src.accelerators.cuda.metrics import (
    CUDAMetricsCollector,
)

from src.accelerators.cuda.optimizer import (
    CUDAOptimizer,
)

from src.accelerators.optimization import (
    AcceleratorWorkload,
    OptimizationConstraints,
    OptimizationPlan,
)

from src.accelerators.rocm.metrics import (
    ROCmMetricsCollector,
)

from src.accelerators.rocm.optimizer import (
    ROCmOptimizer,
)

from src.accelerators.schema import (
    AcceleratorDevice,
    AcceleratorKind,
    AcceleratorSelection,
)

from src.api.config import (
    APIConfig,
)

from src.api.errors import (
    AcceleratorAPIError,
    AcceleratorNotFoundAPIError,
    AcceleratorOptimizationAPIError,
    AcceleratorUnavailableAPIError,
    FeatureDisabledError,
    InternalServerAPIError,
)

from src.api.schema import (
    APIResponseMetadata,
    AcceleratorDeviceResponse,
    AcceleratorInventoryResponse,
    AcceleratorMemoryResponse,
    AcceleratorMetricsResponse,
    AcceleratorOptimizationRequest,
    AcceleratorOptimizationResponse,
    AcceleratorSelectionRequest,
    AcceleratorSelectionResponse,
    AcceleratorWorkloadRequest,
    OptimizationConstraintsRequest,
    OptimizationRecommendationResponse,
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

ACCELERATOR_SERVICE_NAME = (
    "accelerators"
)

ACCELERATOR_SERVICE_VERSION = "1"


# ------------------------------------------------------------------
# Operation context
# ------------------------------------------------------------------

@dataclass(frozen=True)
class AcceleratorOperationContext:
    """
    Request-specific metadata supplied by a route.

    The service does not depend directly on
    FastAPI Request objects.
    """

    request_id: Optional[str] = None

    actor_id: Optional[str] = None

    tenant_id: Optional[str] = None

    correlation_id: Optional[str] = None

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
        }


# ------------------------------------------------------------------
# Mapping helpers
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
        result = converter()

        if isinstance(
            result,
            Mapping,
        ):
            return dict(
                result
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


def _string_list(
    value: Any,
) -> List[str]:
    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        return [
            value
        ]

    if isinstance(
        value,
        Iterable,
    ):
        return [
            str(
                item
            )
            for item in value
        ]

    return [
        str(
            value
        )
    ]


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

    if isinstance(
        value,
        str,
    ):
        text = value.strip()

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

    return None


def _device_available(
    device: Any,
) -> bool:
    value = _value(
        device,
        "available",
        None,
    )

    if value is not None:
        return bool(
            value
        )

    state = str(
        _value(
            device,
            "state",
            "",
        )
    ).lower()

    return state in {
        "available",
        "healthy",
        "ready",
    }


def _device_degraded(
    device: Any,
) -> bool:
    value = _value(
        device,
        "degraded",
        None,
    )

    if value is not None:
        return bool(
            value
        )

    return (
        str(
            _value(
                device,
                "state",
                "",
            )
        ).lower()
        == "degraded"
    )


def _device_accelerator(
    device: Any,
) -> bool:
    value = _value(
        device,
        "accelerator",
        None,
    )

    if value is not None:
        return bool(
            value
        )

    return (
        str(
            _value(
                device,
                "kind",
                "",
            )
        ).lower()
        in {
            AcceleratorKind.CUDA.value,
            AcceleratorKind.ROCM.value,
        }
    )


# ------------------------------------------------------------------
# Public response conversion
# ------------------------------------------------------------------

def memory_response(
    value: Any,
) -> AcceleratorMemoryResponse:
    source = _mapping(
        value
    )

    total = source.get(
        "total_bytes"
    )

    free = source.get(
        "free_bytes"
    )

    used = source.get(
        "used_bytes"
    )

    if (
        used is None
        and total is not None
        and free is not None
    ):
        used = max(
            0,
            int(
                total
            )
            - int(
                free
            ),
        )

    utilization_ratio = source.get(
        "utilization_ratio"
    )

    if (
        utilization_ratio is None
        and total
        and used is not None
    ):
        utilization_ratio = (
            float(
                used
            )
            / float(
                total
            )
        )

    return AcceleratorMemoryResponse(
        total_bytes=total,

        free_bytes=free,

        used_bytes=used,

        allocated_bytes=source.get(
            "allocated_bytes"
        ),

        reserved_bytes=source.get(
            "reserved_bytes"
        ),

        utilization_ratio=(
            utilization_ratio
        ),
    )


def device_response(
    device: Any,
) -> AcceleratorDeviceResponse:
    source = _mapping(
        device
    )

    attributes = _mapping(
        source.get(
            "attributes",
            _value(
                device,
                "attributes",
                {},
            ),
        )
    )

    return AcceleratorDeviceResponse(
        device_id=str(
            source.get(
                "device_id",
                _value(
                    device,
                    "device_id",
                    "",
                ),
            )
        ),

        kind=str(
            source.get(
                "kind",
                _value(
                    device,
                    "kind",
                    "",
                ),
            )
        ),

        runtime_device=str(
            source.get(
                "runtime_device",
                _value(
                    device,
                    "runtime_device",
                    "",
                ),
            )
        ),

        index=source.get(
            "index",
            _value(
                device,
                "index",
                None,
            ),
        ),

        name=source.get(
            "name",
            _value(
                device,
                "name",
                None,
            ),
        ),

        state=str(
            source.get(
                "state",
                _value(
                    device,
                    "state",
                    "unknown",
                ),
            )
        ),

        available=_device_available(
            device
        ),

        degraded=_device_degraded(
            device
        ),

        accelerator=_device_accelerator(
            device
        ),

        memory=memory_response(
            source.get(
                "memory",
                _value(
                    device,
                    "memory",
                    None,
                ),
            )
        ),

        capabilities=_string_list(
            source.get(
                "capabilities",
                _value(
                    device,
                    "capabilities",
                    (),
                ),
            )
        ),

        supported_precisions=_string_list(
            source.get(
                "supported_precisions",
                _value(
                    device,
                    "supported_precisions",
                    (),
                ),
            )
        ),

        architecture=source.get(
            "architecture",
            _value(
                device,
                "architecture",
                None,
            ),
        ),

        compute_capability=source.get(
            "compute_capability",
            _value(
                device,
                "compute_capability",
                None,
            ),
        ),

        vendor=source.get(
            "vendor",
            _value(
                device,
                "vendor",
                None,
            ),
        ),

        runtime_version=source.get(
            "runtime_version",
            _value(
                device,
                "runtime_version",
                None,
            ),
        ),

        attributes=attributes,
    )


def inventory_devices(
    inventory: Any,
) -> Tuple[Any, ...]:
    devices = _value(
        inventory,
        "devices",
        (),
    )

    if devices is None:
        return ()

    return tuple(
        devices
    )


def inventory_response(
    inventory: Any,
    *,
    request_id: Optional[str] = None,
) -> AcceleratorInventoryResponse:
    devices = inventory_devices(
        inventory
    )

    failures = _value(
        inventory,
        "failures",
        (),
    )

    safe_failures = []

    for failure in failures or ():
        value = _mapping(
            failure
        )

        safe_failures.append(
            {
                "backend": value.get(
                    "backend"
                ),

                "error_type": value.get(
                    "error_type"
                ),

                "message": value.get(
                    "message",
                    (
                        "Accelerator backend "
                        "discovery failed."
                    ),
                ),
            }
        )

    return AcceleratorInventoryResponse(
        devices=[
            device_response(
                device
            )
            for device in devices
        ],

        device_count=len(
            devices
        ),

        available_device_count=sum(
            1
            for device in devices
            if _device_available(
                device
            )
        ),

        accelerator_device_count=sum(
            1
            for device in devices
            if (
                _device_available(
                    device
                )
                and _device_accelerator(
                    device
                )
            )
        ),

        detected_at=_optional_datetime(
            _value(
                inventory,
                "detected_at",
                None,
            )
        ),

        failures=safe_failures,

        metadata=APIResponseMetadata(
            request_id=request_id,
        ),
    )


def selection_response(
    selection: AcceleratorSelection,
    *,
    request_id: Optional[str] = None,
) -> AcceleratorSelectionResponse:
    selection_map = _mapping(
        selection
    )

    rejected = selection_map.get(
        "rejected_candidates",
        {},
    )

    safe_rejected = {
        str(
            device_id
        ): _string_list(
            reasons
        )
        for device_id, reasons
        in dict(
            rejected
            or {}
        ).items()
    }

    return AcceleratorSelectionResponse(
        device=device_response(
            selection.device
        ),

        reason=str(
            _value(
                selection,
                "reason",
                (
                    "Accelerator selected "
                    "successfully."
                ),
            )
        ),

        fallback_used=bool(
            _value(
                selection,
                "fallback_used",
                False,
            )
        ),

        candidates_considered=_string_list(
            _value(
                selection,
                "candidates_considered",
                (),
            )
        ),

        rejected_candidates=(
            safe_rejected
        ),

        selected_at=_optional_datetime(
            _value(
                selection,
                "selected_at",
                None,
            )
        ),

        metadata=APIResponseMetadata(
            request_id=request_id,
        ),
    )


def recommendation_response(
    recommendation: Any,
) -> OptimizationRecommendationResponse:
    source = _mapping(
        recommendation
    )

    return OptimizationRecommendationResponse(
        kind=str(
            source.get(
                "kind",
                _value(
                    recommendation,
                    "kind",
                    "",
                ),
            )
        ),

        title=str(
            source.get(
                "title",
                _value(
                    recommendation,
                    "title",
                    "",
                ),
            )
        ),

        reason=str(
            source.get(
                "reason",
                _value(
                    recommendation,
                    "reason",
                    "",
                ),
            )
        ),

        current_value=source.get(
            "current_value",
            _value(
                recommendation,
                "current_value",
                None,
            ),
        ),

        recommended_value=source.get(
            "recommended_value",
            _value(
                recommendation,
                "recommended_value",
                None,
            ),
        ),

        priority=str(
            source.get(
                "priority",
                _value(
                    recommendation,
                    "priority",
                    "medium",
                ),
            )
        ),

        confidence=str(
            source.get(
                "confidence",
                _value(
                    recommendation,
                    "confidence",
                    "medium",
                ),
            )
        ),

        disposition=str(
            source.get(
                "disposition",
                _value(
                    recommendation,
                    "disposition",
                    "recommended",
                ),
            )
        ),

        expected_effect=source.get(
            "expected_effect",
            _value(
                recommendation,
                "expected_effect",
                None,
            ),
        ),

        risks=_string_list(
            source.get(
                "risks",
                _value(
                    recommendation,
                    "risks",
                    (),
                ),
            )
        ),

        prerequisites=_string_list(
            source.get(
                "prerequisites",
                _value(
                    recommendation,
                    "prerequisites",
                    (),
                ),
            )
        ),

        evidence=_string_list(
            source.get(
                "evidence",
                _value(
                    recommendation,
                    "evidence",
                    (),
                ),
            )
        ),

        automatically_applicable=bool(
            source.get(
                "automatically_applicable",
                _value(
                    recommendation,
                    "automatically_applicable",
                    False,
                ),
            )
        ),

        attributes=_mapping(
            source.get(
                "attributes",
                _value(
                    recommendation,
                    "attributes",
                    {},
                ),
            )
        ),
    )


def optimization_response(
    plan: OptimizationPlan,
    *,
    request_id: Optional[str] = None,
) -> AcceleratorOptimizationResponse:
    source = plan.to_dict()

    return AcceleratorOptimizationResponse(
        device_id=str(
            source.get(
                "device_id",
                plan.device_id,
            )
        ),

        backend=str(
            source.get(
                "backend",
                plan.backend,
            )
        ),

        optimizer_name=source.get(
            "optimizer_name"
        ),

        optimizer_version=source.get(
            "optimizer_version"
        ),

        recommendation_count=int(
            source.get(
                "recommendation_count",
                len(
                    plan.recommendations
                ),
            )
        ),

        high_priority_count=int(
            source.get(
                "high_priority_count",
                plan.high_priority_count,
            )
        ),

        recommendations=[
            recommendation_response(
                item
            )
            for item
            in plan.recommendations
        ],

        profile=source.get(
            "profile"
        ),

        automatically_applied=bool(
            source.get(
                "automatically_applied",
                False,
            )
        ),

        generated_at=_optional_datetime(
            source.get(
                "generated_at"
            )
        ),

        metadata=APIResponseMetadata(
            request_id=request_id,
        ),
    )


# ------------------------------------------------------------------
# Request conversion
# ------------------------------------------------------------------

def workload_from_request(
    request: AcceleratorWorkloadRequest,
) -> AcceleratorWorkload:
    workload = AcceleratorWorkload(
        kind=request.kind,

        batch_size=(
            request.batch_size
        ),

        model_parameter_count=(
            request.
            model_parameter_count
        ),

        estimated_model_bytes=(
            request.
            estimated_model_bytes
        ),

        estimated_input_bytes=(
            request.
            estimated_input_bytes
        ),

        estimated_peak_memory_bytes=(
            request.
            estimated_peak_memory_bytes
        ),

        sequence_length=(
            request.sequence_length
        ),

        precision=request.precision,

        mixed_precision_enabled=(
            request.
            mixed_precision_enabled
        ),

        gradient_accumulation_steps=(
            request.
            gradient_accumulation_steps
        ),

        data_loader_workers=(
            request.
            data_loader_workers
        ),

        pin_memory_enabled=(
            request.
            pin_memory_enabled
        ),

        non_blocking_transfers=(
            request.
            non_blocking_transfers
        ),

        compiled=(
            request.compiled
        ),

        latency_sensitive=(
            request.latency_sensitive
        ),

        throughput_sensitive=(
            request.
            throughput_sensitive
        ),

        deterministic=(
            request.deterministic
        ),

        attributes=dict(
            request.attributes
        ),
    )

    workload.validate()

    return workload


def constraints_from_request(
    request: OptimizationConstraintsRequest,
) -> OptimizationConstraints:
    constraints = OptimizationConstraints(
        level=request.level,

        allow_precision_change=(
            request.
            allow_precision_change
        ),

        allow_mixed_precision=(
            request.
            allow_mixed_precision
        ),

        allow_batch_size_change=(
            request.
            allow_batch_size_change
        ),

        allow_compilation=(
            request.
            allow_compilation
        ),

        allow_thread_change=(
            request.
            allow_thread_change
        ),

        allow_data_loader_change=(
            request.
            allow_data_loader_change
        ),

        allow_memory_tuning=(
            request.
            allow_memory_tuning
        ),

        minimum_batch_size=(
            request.
            minimum_batch_size
        ),

        maximum_batch_size=(
            request.
            maximum_batch_size
        ),

        maximum_memory_fraction=(
            request.
            maximum_memory_fraction
        ),

        required_precision=(
            request.
            required_precision
        ),
    )

    constraints.validate()

    return constraints


# ------------------------------------------------------------------
# Service
# ------------------------------------------------------------------

class AcceleratorService:
    """
    Production-facing accelerator application
    service.

    Instances are process-scoped and shared across
    requests through APIDependencies.
    """

    def __init__(
        self,
        *,
        config: APIConfig,
        manager: AcceleratorManager,
        cpu_optimizer: Optional[
            CPUOptimizer
        ] = None,
        cuda_optimizer: Optional[
            CUDAOptimizer
        ] = None,
        rocm_optimizer: Optional[
            ROCmOptimizer
        ] = None,
    ) -> None:
        config.validate()

        self.config = config
        self.manager = manager

        self.cpu_optimizer = (
            cpu_optimizer
            or CPUOptimizer()
        )

        self.cuda_optimizer = (
            cuda_optimizer
            or CUDAOptimizer()
        )

        self.rocm_optimizer = (
            rocm_optimizer
            or ROCmOptimizer()
        )

        self._lock = RLock()

        self._closed = False

    # ------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------

    def _ensure_open(
        self,
    ) -> None:
        with self._lock:
            if self._closed:
                raise AcceleratorUnavailableAPIError(
                    "The accelerator service "
                    "is closed.",
                    details={
                        "resource": (
                            ACCELERATOR_SERVICE_NAME
                        ),
                    },
                )

    def close(
        self,
    ) -> None:
        with self._lock:
            self._closed = True

    # ------------------------------------------------------
    # Feature gates
    # ------------------------------------------------------

    def _require_read_enabled(
        self,
    ) -> None:
        if not (
            self.config.features.
            accelerator_read_enabled
        ):
            raise FeatureDisabledError(
                details={
                    "resource": (
                        "accelerator_read"
                    ),
                },
            )

    def _require_optimization_enabled(
        self,
    ) -> None:
        if not (
            self.config.features.
            accelerator_optimization_enabled
        ):
            raise FeatureDisabledError(
                details={
                    "resource": (
                        "accelerator_optimization"
                    ),
                },
            )

    # ------------------------------------------------------
    # Inventory
    # ------------------------------------------------------

    def list_devices(
        self,
        *,
        refresh: bool = False,
        context: Optional[
            AcceleratorOperationContext
        ] = None,
    ) -> AcceleratorInventoryResponse:
        self._ensure_open()
        self._require_read_enabled()

        request_id = (
            context.request_id
            if context is not None
            else None
        )

        try:
            inventory = (
                self.manager.inventory(
                    refresh=refresh
                )
            )

            return inventory_response(
                inventory,
                request_id=request_id,
            )

        except APIException:
            raise

        except Exception as exc:
            raise AcceleratorAPIError(
                "Accelerator inventory "
                "could not be retrieved.",
                details={
                    "operation": (
                        "accelerator_inventory"
                    ),

                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                },
            ) from exc

    # ------------------------------------------------------
    # Device lookup
    # ------------------------------------------------------

    def _find_device(
        self,
        device_id: str,
        *,
        refresh: bool = False,
    ) -> AcceleratorDevice:
        normalized = str(
            device_id
        ).strip().lower()

        if not normalized:
            raise AcceleratorNotFoundAPIError(
                details={
                    "resource": (
                        "accelerator_device"
                    ),
                },
            )

        inventory = self.manager.inventory(
            refresh=refresh
        )

        for device in inventory_devices(
            inventory
        ):
            if (
                str(
                    _value(
                        device,
                        "device_id",
                        "",
                    )
                ).strip().lower()
                == normalized
            ):
                return device

        raise AcceleratorNotFoundAPIError(
            details={
                "resource": (
                    "accelerator_device"
                ),

                "resource_id": (
                    normalized
                ),
            },
        )

    def get_device(
        self,
        device_id: str,
        *,
        refresh: bool = False,
        context: Optional[
            AcceleratorOperationContext
        ] = None,
    ) -> AcceleratorDeviceResponse:
        self._ensure_open()
        self._require_read_enabled()

        try:
            return device_response(
                self._find_device(
                    device_id,
                    refresh=refresh,
                )
            )

        except APIException:
            raise

        except Exception as exc:
            raise AcceleratorAPIError(
                "The accelerator device "
                "could not be retrieved.",
                details={
                    "operation": (
                        "accelerator_device"
                    ),

                    "device_id": (
                        device_id
                    ),

                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                },
            ) from exc

    # ------------------------------------------------------
    # Selection
    # ------------------------------------------------------

    def _selection_arguments(
        self,
        request: AcceleratorSelectionRequest,
    ) -> Dict[str, Any]:
        return {
            "preferred_kind": (
                request.preferred_kind
            ),

            "device_id": (
                request.device_id
            ),

            "device_index": (
                request.device_index
            ),

            "minimum_total_memory_bytes": (
                request.
                minimum_total_memory_bytes
            ),

            "minimum_free_memory_bytes": (
                request.
                minimum_free_memory_bytes
            ),

            "required_capabilities": tuple(
                request.
                required_capabilities
            ),

            "required_precisions": tuple(
                request.
                required_precisions
            ),

            "allow_degraded": (
                request.allow_degraded
            ),

            "require_accelerator": (
                request.
                require_accelerator
            ),

            "allow_cpu_fallback": (
                request.
                allow_cpu_fallback
            ),

            "mixed_precision": (
                request.mixed_precision
            ),

            "refresh": True,
        }

    @staticmethod
    def _remove_none_values(
        values: Mapping[
            str,
            Any,
        ],
    ) -> Dict[str, Any]:
        return {
            key: value
            for key, value
            in values.items()
            if value is not None
        }

    def select_internal(
        self,
        request: AcceleratorSelectionRequest,
    ) -> AcceleratorSelection:
        """
        Create a temporary accelerator selection.

        The caller owns release responsibility.
        """

        arguments = (
            self._remove_none_values(
                self._selection_arguments(
                    request
                )
            )
        )

        try:
            if request.training:
                return (
                    self.manager.
                    select_for_training(
                        **arguments
                    )
                )

            return (
                self.manager.
                select_for_inference(
                    **arguments
                )
            )

        except TypeError:
            # Compatibility fallback for manager
            # implementations with a narrower
            # method signature.
            compact = {
                key: value
                for key, value
                in arguments.items()
                if key
                in {
                    "preferred_kind",
                    "device_index",
                    "minimum_free_memory_bytes",
                    "required_precisions",
                    "require_accelerator",
                    "allow_cpu_fallback",
                    "mixed_precision",
                    "refresh",
                }
            }

            if request.training:
                return (
                    self.manager.
                    select_for_training(
                        **compact
                    )
                )

            return (
                self.manager.
                select_for_inference(
                    **compact
                )
            )

    def select(
        self,
        request: AcceleratorSelectionRequest,
        *,
        context: Optional[
            AcceleratorOperationContext
        ] = None,
    ) -> AcceleratorSelectionResponse:
        self._ensure_open()
        self._require_read_enabled()

        selection = None

        request_id = (
            context.request_id
            if context is not None
            else None
        )

        try:
            selection = (
                self.select_internal(
                    request
                )
            )

            return selection_response(
                selection,
                request_id=request_id,
            )

        except APIException:
            raise

        except Exception as exc:
            raise AcceleratorUnavailableAPIError(
                details={
                    "operation": (
                        "accelerator_selection"
                    ),

                    "preferred_kind": (
                        request.
                        preferred_kind
                    ),

                    "fallback_allowed": (
                        request.
                        allow_cpu_fallback
                    ),

                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                },
            ) from exc

        finally:
            if selection is not None:
                try:
                    self.manager.release(
                        selection
                    )

                except Exception:
                    logger.exception(
                        "Temporary accelerator "
                        "selection release failed",
                        extra={
                            "accelerator_service": {
                                "device_id": (
                                    selection.
                                    device_id
                                ),

                                "request_id": (
                                    request_id
                                ),
                            },
                        },
                    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    def metrics(
        self,
        device_id: str,
        *,
        refresh_device: bool = False,
        context: Optional[
            AcceleratorOperationContext
        ] = None,
    ) -> AcceleratorMetricsResponse:
        self._ensure_open()
        self._require_read_enabled()

        request_id = (
            context.request_id
            if context is not None
            else None
        )

        device = self._find_device(
            device_id,
            refresh=refresh_device,
        )

        if not _device_available(
            device
        ):
            raise AcceleratorUnavailableAPIError(
                details={
                    "device_id": (
                        device_id
                    ),

                    "operation": (
                        "accelerator_metrics"
                    ),
                },
            )

        kind = str(
            device.kind
        )

        try:
            if (
                kind
                == AcceleratorKind.CPU.value
            ):
                snapshot = (
                    CPUMetricsCollector().
                    collect(
                        interval=None,
                        include_process=True,
                    )
                )

            elif (
                kind
                == AcceleratorKind.CUDA.value
            ):
                snapshot = (
                    CUDAMetricsCollector(
                        device=device
                    ).collect(
                        refresh_profile=(
                            refresh_device
                        )
                    )
                )

            elif (
                kind
                == AcceleratorKind.ROCM.value
            ):
                snapshot = (
                    ROCmMetricsCollector(
                        device=device
                    ).collect(
                        refresh_profile=(
                            refresh_device
                        )
                    )
                )

            else:
                raise AcceleratorAPIError(
                    "Metrics are not supported "
                    "for this accelerator kind.",
                    details={
                        "device_id": (
                            device_id
                        ),

                        "kind": kind,
                    },
                )

            data = _mapping(
                snapshot
            )

            return AcceleratorMetricsResponse(
                device_id=device.device_id,

                kind=kind,

                runtime_device=(
                    device.runtime_device
                ),

                metrics=data,

                sampled_at=_optional_datetime(
                    data.get(
                        "sampled_at"
                    )
                ),

                metadata=(
                    APIResponseMetadata(
                        request_id=(
                            request_id
                        ),
                    )
                ),
            )

        except APIException:
            raise

        except Exception as exc:
            raise AcceleratorAPIError(
                "Accelerator metrics could "
                "not be collected.",
                code=(
                    "accelerator_metrics_unavailable"
                ),
                details={
                    "operation": (
                        "accelerator_metrics"
                    ),

                    "device_id": (
                        device_id
                    ),

                    "kind": kind,

                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                },
            ) from exc

    # ------------------------------------------------------
    # Optimization
    # ------------------------------------------------------

    def _optimizer_for_kind(
        self,
        kind: str,
    ) -> Any:
        if (
            kind
            == AcceleratorKind.CPU.value
        ):
            return self.cpu_optimizer

        if (
            kind
            == AcceleratorKind.CUDA.value
        ):
            return self.cuda_optimizer

        if (
            kind
            == AcceleratorKind.ROCM.value
        ):
            return self.rocm_optimizer

        raise AcceleratorOptimizationAPIError(
            details={
                "kind": kind,

                "operation": (
                    "accelerator_optimization"
                ),
            },
        )

    def optimize(
        self,
        request: AcceleratorOptimizationRequest,
        *,
        context: Optional[
            AcceleratorOperationContext
        ] = None,
    ) -> AcceleratorOptimizationResponse:
        """
        Generate an advisory optimization plan.

        No recommendation is applied.
        """

        self._ensure_open()
        self._require_optimization_enabled()

        selection = None

        request_id = (
            context.request_id
            if context is not None
            else None
        )

        try:
            selection = (
                self.select_internal(
                    request.selection
                )
            )

            workload = (
                workload_from_request(
                    request.workload
                )
            )

            constraints = (
                constraints_from_request(
                    request.constraints
                )
            )

            optimizer = (
                self._optimizer_for_kind(
                    selection.kind
                )
            )

            plan = optimizer.recommend(
                selection=selection,
                workload=workload,
                constraints=constraints,
            )

            if not isinstance(
                plan,
                OptimizationPlan,
            ):
                raise InternalServerAPIError(
                    "Accelerator optimizer "
                    "returned an invalid plan.",
                    details={
                        "operation": (
                            "accelerator_optimization"
                        ),

                        "device_id": (
                            selection.device_id
                        ),

                        "kind": (
                            selection.kind
                        ),
                    },
                )

            response = (
                optimization_response(
                    plan,
                    request_id=request_id,
                )
            )

            if (
                response.
                automatically_applied
            ):
                raise InternalServerAPIError(
                    "Accelerator optimizer "
                    "violated advisory-only "
                    "execution policy.",
                    details={
                        "operation": (
                            "accelerator_optimization"
                        ),

                        "device_id": (
                            selection.device_id
                        ),
                    },
                )

            return response

        except APIException:
            raise

        except Exception as exc:
            raise AcceleratorOptimizationAPIError(
                details={
                    "operation": (
                        "accelerator_optimization"
                    ),

                    "preferred_kind": (
                        request.selection.
                        preferred_kind
                    ),

                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                },
            ) from exc

        finally:
            if selection is not None:
                try:
                    self.manager.release(
                        selection
                    )

                except Exception:
                    logger.exception(
                        "Optimization accelerator "
                        "selection release failed",
                        extra={
                            "accelerator_service": {
                                "device_id": (
                                    selection.
                                    device_id
                                ),

                                "request_id": (
                                    request_id
                                ),
                            },
                        },
                    )

    # ------------------------------------------------------
    # Health summary
    # ------------------------------------------------------

    def health(
        self,
        *,
        deep: bool = False,
    ) -> Dict[str, Any]:
        self._ensure_open()

        try:
            report = self.manager.health(
                deep=deep,
                refresh_inventory=True,
            )

            return _mapping(
                report
            )

        except TypeError:
            try:
                report = (
                    self.manager.health()
                )

                return _mapping(
                    report
                )

            except Exception as exc:
                raise AcceleratorAPIError(
                    "Accelerator health could "
                    "not be evaluated.",
                    details={
                        "operation": (
                            "accelerator_health"
                        ),

                        "error_type": (
                            type(
                                exc
                            ).__name__
                        ),
                    },
                ) from exc

        except Exception as exc:
            raise AcceleratorAPIError(
                "Accelerator health could "
                "not be evaluated.",
                details={
                    "operation": (
                        "accelerator_health"
                    ),

                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                },
            ) from exc

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
                ACCELERATOR_SERVICE_NAME
            ),

            "version": (
                ACCELERATOR_SERVICE_VERSION
            ),

            "closed": closed,

            "read_enabled": (
                self.config.features.
                accelerator_read_enabled
            ),

            "optimization_enabled": (
                self.config.features.
                accelerator_optimization_enabled
            ),

            "optimizers": {
                "cpu": type(
                    self.cpu_optimizer
                ).__name__,

                "cuda": type(
                    self.cuda_optimizer
                ).__name__,

                "rocm": type(
                    self.rocm_optimizer
                ).__name__,
            },

            "optimization_policy": (
                "advisory_only"
            ),
        }


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def create_accelerator_service(
    *,
    config: APIConfig,
    manager: AcceleratorManager,
) -> AcceleratorService:
    return AcceleratorService(
        config=config,
        manager=manager,
    )


# ------------------------------------------------------------------
# Dependency registration
# ------------------------------------------------------------------

def register_accelerator_service(
    dependencies: Any,
    *,
    replace: bool = False,
) -> AcceleratorService:
    """
    Construct and register AcceleratorService in
    APIDependencies without importing the
    dependency container here.

    This avoids circular imports.
    """

    config = getattr(
        dependencies,
        "config",
        None,
    )

    manager = getattr(
        dependencies,
        "accelerator_manager",
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

    if manager is None:
        raise InternalServerAPIError(
            "Accelerator manager is unavailable.",
            details={
                "resource": (
                    "accelerator_manager"
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

    service = create_accelerator_service(
        config=config,
        manager=manager,
    )

    register(
        ACCELERATOR_SERVICE_NAME,
        service,
        replace=replace,
    )

    return service


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "ACCELERATOR_SERVICE_NAME",
    "ACCELERATOR_SERVICE_VERSION",

    "AcceleratorOperationContext",

    "memory_response",
    "device_response",
    "inventory_devices",
    "inventory_response",
    "selection_response",

    "recommendation_response",
    "optimization_response",

    "workload_from_request",
    "constraints_from_request",

    "AcceleratorService",

    "create_accelerator_service",
    "register_accelerator_service",
]
