from __future__ import annotations

"""
Framework-neutral accelerator contracts for OSHA.

This module defines the data exchanged between:

    CPU, CUDA, and ROCm backends
        -> AcceleratorDetector
        -> AcceleratorInventory
        -> AcceleratorManager
        -> training and AI serving

This module does not:

    detect hardware
    import PyTorch
    allocate memory
    move tensors
    select a device
    train or execute models
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from src.core.serialization import to_json_safe
from src.core.time import utc_now


ACCELERATOR_SCHEMA_VERSION = "1"


# ------------------------------------------------------------------
# Enumerations
# ------------------------------------------------------------------


class AcceleratorKind(str, Enum):
    """
    Logical accelerator backend family.

    AUTO is valid for requirements and configuration, but a discovered
    AcceleratorDevice must always have a concrete kind.
    """

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"


class AcceleratorState(str, Enum):
    """
    Operational state of a backend or device.
    """

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class AcceleratorHealthStatus(str, Enum):
    """
    Health status reported by an accelerator backend.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class AcceleratorPrecision(str, Enum):
    """
    Precision capabilities that callers may request.
    """

    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"


class AcceleratorCapability(str, Enum):
    """
    Standard accelerator capability names.

    Backends may also advertise namespaced custom capabilities through
    AcceleratorDevice.capabilities.
    """

    COMPUTE = "compute"
    INFERENCE = "inference"
    TRAINING = "training"
    MIXED_PRECISION = "mixed_precision"
    TENSOR_CORES = "tensor_cores"
    ASYNC_EXECUTION = "async_execution"
    MEMORY_STATS = "memory_stats"
    SYNCHRONIZATION = "synchronization"
    CACHE_CONTROL = "cache_control"
    PEER_TO_PEER = "peer_to_peer"
    UNIFIED_MEMORY = "unified_memory"


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------


def _required_text(
    value: Any,
    *,
    name: str,
    maximum_length: int = 1024,
) -> str:
    if value is None:
        raise ValueError(f"{name} is required")

    text = str(value).strip()

    if not text:
        raise ValueError(f"{name} cannot be empty")

    if len(text) > maximum_length:
        raise ValueError(f"{name} cannot exceed {maximum_length} characters")

    return text


def _optional_text(
    value: Any,
    *,
    name: str,
    maximum_length: int = 4096,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if len(text) > maximum_length:
        raise ValueError(f"{name} cannot exceed {maximum_length} characters")

    return text


def _non_negative_integer(
    value: Any,
    *,
    name: str,
    allow_none: bool = False,
) -> Optional[int]:
    if value is None:
        if allow_none:
            return None

        raise ValueError(f"{name} is required")

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")

    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if result < 0:
        raise ValueError(f"{name} must be >= 0")

    return result


def _finite_number(
    value: Any,
    *,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    allow_none: bool = False,
) -> Optional[float]:
    if value is None:
        if allow_none:
            return None

        raise ValueError(f"{name} is required")

    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")

    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")

    return result


def _unique_strings(
    values: Iterable[Any],
    *,
    name: str,
) -> Tuple[str, ...]:
    output: List[str] = []
    seen = set()

    for value in values:
        text = str(value).strip()

        if not text:
            raise ValueError(f"{name} cannot contain empty values")

        if text in seen:
            continue

        seen.add(text)
        output.append(text)

    return tuple(output)


# ------------------------------------------------------------------
# Memory information
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AcceleratorMemory:
    """
    Memory information for one accelerator device.

    Unknown values remain None. Backends must not invent unavailable
    memory measurements.
    """

    total_bytes: Optional[int] = None
    free_bytes: Optional[int] = None
    used_bytes: Optional[int] = None
    reserved_bytes: Optional[int] = None
    allocated_bytes: Optional[int] = None

    def validate(self) -> None:
        values = {
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "used_bytes": self.used_bytes,
            "reserved_bytes": self.reserved_bytes,
            "allocated_bytes": self.allocated_bytes,
        }

        for name, value in values.items():
            _non_negative_integer(
                value,
                name=name,
                allow_none=True,
            )

        if self.total_bytes is not None:
            for name in (
                "free_bytes",
                "used_bytes",
                "reserved_bytes",
                "allocated_bytes",
            ):
                value = values[name]

                if value is not None and value > self.total_bytes:
                    raise ValueError(f"{name} cannot exceed total_bytes")

        if (
            self.total_bytes is not None
            and self.free_bytes is not None
            and self.used_bytes is not None
            and self.free_bytes + self.used_bytes > self.total_bytes
        ):
            raise ValueError("free_bytes + used_bytes cannot exceed total_bytes")

    @property
    def utilization_ratio(self) -> Optional[float]:
        self.validate()

        if not self.total_bytes:
            return None

        if self.used_bytes is not None:
            return self.used_bytes / self.total_bytes

        if self.free_bytes is not None:
            return (self.total_bytes - self.free_bytes) / self.total_bytes

        return None

    @property
    def available_bytes(self) -> Optional[int]:
        """
        Best available memory estimate.

        free_bytes is preferred because allocated and reserved memory do
        not necessarily represent complete device usage.
        """

        self.validate()

        return self.free_bytes

    def satisfies(
        self,
        minimum_free_bytes: Optional[int],
    ) -> bool:
        self.validate()

        if minimum_free_bytes is None:
            return True

        required = _non_negative_integer(
            minimum_free_bytes,
            name="minimum_free_bytes",
        )

        if self.free_bytes is None:
            return False

        return self.free_bytes >= required

    def to_dict(self) -> Dict[str, Any]:
        self.validate()

        return {
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "used_bytes": self.used_bytes,
            "reserved_bytes": self.reserved_bytes,
            "allocated_bytes": self.allocated_bytes,
            "utilization_ratio": self.utilization_ratio,
        }


# ------------------------------------------------------------------
# Device descriptor
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AcceleratorDevice:
    """
    Framework-neutral description of one execution device.

    device_id:
        Stable accelerator-layer identity, such as "cpu" or "cuda:0".

    runtime_device:
        Device identifier used by the runtime adapter. It remains a
        string here so this schema does not depend on PyTorch.

    kind:
        Logical backend family, such as CPU, CUDA, or ROCm.
    """

    device_id: str
    kind: str
    runtime_device: str

    index: Optional[int] = None
    name: Optional[str] = None
    state: str = AcceleratorState.UNKNOWN.value

    memory: AcceleratorMemory = field(default_factory=AcceleratorMemory)

    capabilities: Tuple[str, ...] = ()
    supported_precisions: Tuple[str, ...] = (AcceleratorPrecision.FLOAT32.value,)

    compute_capability: Optional[str] = None
    vendor: Optional[str] = None
    architecture: Optional[str] = None
    driver_version: Optional[str] = None
    runtime_version: Optional[str] = None

    logical_processor_count: Optional[int] = None
    multiprocessor_count: Optional[int] = None

    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _required_text(
            self.device_id,
            name="device_id",
            maximum_length=256,
        )

        _required_text(
            self.runtime_device,
            name="runtime_device",
            maximum_length=256,
        )

        if self.kind not in {
            AcceleratorKind.CPU.value,
            AcceleratorKind.CUDA.value,
            AcceleratorKind.ROCM.value,
        }:
            raise ValueError("AcceleratorDevice.kind must be cpu, cuda, or rocm")

        if self.state not in {item.value for item in AcceleratorState}:
            raise ValueError(f"Unknown accelerator device state: {self.state}")

        _non_negative_integer(
            self.index,
            name="device index",
            allow_none=True,
        )

        _optional_text(
            self.name,
            name="device name",
            maximum_length=1024,
        )

        _optional_text(
            self.compute_capability,
            name="compute_capability",
            maximum_length=128,
        )

        _optional_text(
            self.vendor,
            name="vendor",
            maximum_length=256,
        )

        _optional_text(
            self.architecture,
            name="architecture",
            maximum_length=256,
        )

        _optional_text(
            self.driver_version,
            name="driver_version",
            maximum_length=256,
        )

        _optional_text(
            self.runtime_version,
            name="runtime_version",
            maximum_length=256,
        )

        _non_negative_integer(
            self.logical_processor_count,
            name="logical_processor_count",
            allow_none=True,
        )

        _non_negative_integer(
            self.multiprocessor_count,
            name="multiprocessor_count",
            allow_none=True,
        )

        self.memory.validate()

        capabilities = _unique_strings(
            self.capabilities,
            name="capabilities",
        )

        precisions = _unique_strings(
            self.supported_precisions,
            name="supported_precisions",
        )

        valid_precisions = {item.value for item in AcceleratorPrecision}

        unsupported_precisions = [
            precision for precision in precisions if precision not in valid_precisions
        ]

        if unsupported_precisions:
            raise ValueError(
                "Unsupported accelerator precisions: "
                + ", ".join(unsupported_precisions)
            )

        if len(capabilities) != len(self.capabilities):
            raise ValueError(
                "AcceleratorDevice.capabilities must not contain duplicates"
            )

        if len(precisions) != len(self.supported_precisions):
            raise ValueError(
                "AcceleratorDevice.supported_precisions must not contain duplicates"
            )

    @property
    def available(self) -> bool:
        return self.state in {
            AcceleratorState.AVAILABLE.value,
            AcceleratorState.DEGRADED.value,
        }

    @property
    def degraded(self) -> bool:
        return self.state == AcceleratorState.DEGRADED.value

    @property
    def accelerator(self) -> bool:
        return self.kind in {
            AcceleratorKind.CUDA.value,
            AcceleratorKind.ROCM.value,
        }

    def supports_capability(
        self,
        capability: str,
    ) -> bool:
        return str(capability).strip() in set(self.capabilities)

    def supports_precision(
        self,
        precision: str,
    ) -> bool:
        return str(precision).strip().lower() in set(self.supported_precisions)

    def to_dict(self) -> Dict[str, Any]:
        self.validate()

        return {
            "device_id": self.device_id,
            "kind": self.kind,
            "runtime_device": self.runtime_device,
            "index": self.index,
            "name": self.name,
            "state": self.state,
            "available": self.available,
            "degraded": self.degraded,
            "accelerator": self.accelerator,
            "memory": self.memory.to_dict(),
            "capabilities": list(self.capabilities),
            "supported_precisions": list(self.supported_precisions),
            "compute_capability": self.compute_capability,
            "vendor": self.vendor,
            "architecture": self.architecture,
            "driver_version": self.driver_version,
            "runtime_version": self.runtime_version,
            "logical_processor_count": (self.logical_processor_count),
            "multiprocessor_count": self.multiprocessor_count,
            "attributes": to_json_safe(dict(self.attributes)),
        }


# ------------------------------------------------------------------
# Accelerator requirements
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AcceleratorRequirements:
    """
    Compute requirements supplied by a caller.

    The manager uses this contract to select an eligible device.
    """

    preferred_kind: str = AcceleratorKind.AUTO.value
    device_index: Optional[int] = None
    device_id: Optional[str] = None

    allow_cpu_fallback: bool = True
    require_accelerator: bool = False

    minimum_total_memory_bytes: Optional[int] = None
    minimum_free_memory_bytes: Optional[int] = None

    required_capabilities: Tuple[str, ...] = ()
    required_precisions: Tuple[str, ...] = ()

    mixed_precision: bool = False
    training: bool = False
    inference: bool = False

    allow_degraded: bool = True

    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.preferred_kind not in {item.value for item in AcceleratorKind}:
            raise ValueError(
                f"Unsupported preferred accelerator kind: {self.preferred_kind}"
            )

        _non_negative_integer(
            self.device_index,
            name="device_index",
            allow_none=True,
        )

        _optional_text(
            self.device_id,
            name="device_id",
            maximum_length=256,
        )

        _non_negative_integer(
            self.minimum_total_memory_bytes,
            name="minimum_total_memory_bytes",
            allow_none=True,
        )

        _non_negative_integer(
            self.minimum_free_memory_bytes,
            name="minimum_free_memory_bytes",
            allow_none=True,
        )

        capabilities = _unique_strings(
            self.required_capabilities,
            name="required_capabilities",
        )

        precisions = _unique_strings(
            self.required_precisions,
            name="required_precisions",
        )

        if len(capabilities) != len(self.required_capabilities):
            raise ValueError("required_capabilities must not contain duplicates")

        if len(precisions) != len(self.required_precisions):
            raise ValueError("required_precisions must not contain duplicates")

        valid_precisions = {item.value for item in AcceleratorPrecision}

        unsupported = [
            precision for precision in precisions if precision not in valid_precisions
        ]

        if unsupported:
            raise ValueError(
                "Unsupported required precisions: " + ", ".join(unsupported)
            )

        if (
            self.require_accelerator
            and self.preferred_kind == AcceleratorKind.CPU.value
        ):
            raise ValueError(
                "require_accelerator cannot be used with preferred_kind=cpu"
            )

        if self.require_accelerator and self.allow_cpu_fallback:
            raise ValueError(
                "require_accelerator and allow_cpu_fallback cannot both be enabled"
            )

        if self.mixed_precision:
            requested_precisions = set(self.required_precisions)

            if not requested_precisions.intersection(
                {
                    AcceleratorPrecision.FLOAT16.value,
                    AcceleratorPrecision.BFLOAT16.value,
                }
            ):
                raise ValueError(
                    "mixed_precision requires float16 or bfloat16 in "
                    "required_precisions"
                )

    def matches(
        self,
        device: AcceleratorDevice,
    ) -> Tuple[bool, Tuple[str, ...]]:
        """
        Determine whether a device satisfies these requirements.

        Returns:
            (matches, rejection_reasons)
        """

        self.validate()
        device.validate()

        reasons: List[str] = []

        if not device.available:
            reasons.append("device_unavailable")

        if device.degraded and not self.allow_degraded:
            reasons.append("degraded_device_not_allowed")

        if (
            self.preferred_kind != AcceleratorKind.AUTO.value
            and device.kind != self.preferred_kind
        ):
            reasons.append("backend_kind_mismatch")

        if self.device_id is not None and device.device_id != self.device_id:
            reasons.append("device_id_mismatch")

        if self.device_index is not None and device.index != self.device_index:
            reasons.append("device_index_mismatch")

        if self.require_accelerator and not device.accelerator:
            reasons.append("hardware_accelerator_required")

        if self.minimum_total_memory_bytes is not None and (
            device.memory.total_bytes is None
            or device.memory.total_bytes < self.minimum_total_memory_bytes
        ):
            reasons.append("insufficient_total_memory")

        if self.minimum_free_memory_bytes is not None and (
            device.memory.free_bytes is None
            or device.memory.free_bytes < self.minimum_free_memory_bytes
        ):
            reasons.append("insufficient_free_memory")

        for capability in self.required_capabilities:
            if not device.supports_capability(capability):
                reasons.append(f"missing_capability:{capability}")

        for precision in self.required_precisions:
            if not device.supports_precision(precision):
                reasons.append(f"missing_precision:{precision}")

        if self.training and not device.supports_capability(
            AcceleratorCapability.TRAINING.value
        ):
            reasons.append("training_capability_missing")

        if self.inference and not device.supports_capability(
            AcceleratorCapability.INFERENCE.value
        ):
            reasons.append("inference_capability_missing")

        if self.mixed_precision and not device.supports_capability(
            AcceleratorCapability.MIXED_PRECISION.value
        ):
            reasons.append("mixed_precision_capability_missing")

        return (
            not reasons,
            tuple(reasons),
        )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()

        return {
            "preferred_kind": self.preferred_kind,
            "device_index": self.device_index,
            "device_id": self.device_id,
            "allow_cpu_fallback": self.allow_cpu_fallback,
            "require_accelerator": self.require_accelerator,
            "minimum_total_memory_bytes": (self.minimum_total_memory_bytes),
            "minimum_free_memory_bytes": (self.minimum_free_memory_bytes),
            "required_capabilities": list(self.required_capabilities),
            "required_precisions": list(self.required_precisions),
            "mixed_precision": self.mixed_precision,
            "training": self.training,
            "inference": self.inference,
            "allow_degraded": self.allow_degraded,
            "attributes": to_json_safe(dict(self.attributes)),
        }


# ------------------------------------------------------------------
# Accelerator inventory
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AcceleratorInventory:
    """
    Snapshot of all devices discovered by accelerator backends.
    """

    devices: Tuple[AcceleratorDevice, ...]
    detected_at: str = field(default_factory=utc_now)
    detector_version: str = ACCELERATOR_SCHEMA_VERSION
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _required_text(
            self.detected_at,
            name="detected_at",
            maximum_length=128,
        )

        _required_text(
            self.detector_version,
            name="detector_version",
            maximum_length=128,
        )

        seen = set()

        for device in self.devices:
            device.validate()

            if device.device_id in seen:
                raise ValueError(
                    "Accelerator inventory contains duplicate device_id: "
                    f"{device.device_id}"
                )

            seen.add(device.device_id)

    @property
    def available_devices(self) -> Tuple[AcceleratorDevice, ...]:
        self.validate()

        return tuple(device for device in self.devices if device.available)

    @property
    def accelerator_devices(self) -> Tuple[AcceleratorDevice, ...]:
        self.validate()

        return tuple(device for device in self.available_devices if device.accelerator)

    @property
    def cpu_devices(self) -> Tuple[AcceleratorDevice, ...]:
        self.validate()

        return tuple(
            device
            for device in self.available_devices
            if device.kind == AcceleratorKind.CPU.value
        )

    def by_kind(
        self,
        kind: str,
        *,
        available_only: bool = False,
    ) -> Tuple[AcceleratorDevice, ...]:
        if kind not in {
            AcceleratorKind.CPU.value,
            AcceleratorKind.CUDA.value,
            AcceleratorKind.ROCM.value,
        }:
            raise ValueError(f"Unsupported concrete accelerator kind: {kind}")

        source = self.available_devices if available_only else self.devices

        return tuple(device for device in source if device.kind == kind)

    def get(
        self,
        device_id: str,
    ) -> Optional[AcceleratorDevice]:
        expected = str(device_id).strip()

        for device in self.devices:
            if device.device_id == expected:
                return device

        return None

    def matching(
        self,
        requirements: AcceleratorRequirements,
    ) -> Tuple[AcceleratorDevice, ...]:
        requirements.validate()

        output: List[AcceleratorDevice] = []

        for device in self.devices:
            matches, _ = requirements.matches(device)

            if matches:
                output.append(device)

        return tuple(output)

    def to_dict(self) -> Dict[str, Any]:
        self.validate()

        by_kind: Dict[str, int] = {}

        for device in self.devices:
            by_kind[device.kind] = by_kind.get(device.kind, 0) + 1

        return {
            "schema_version": ACCELERATOR_SCHEMA_VERSION,
            "detected_at": self.detected_at,
            "detector_version": self.detector_version,
            "device_count": len(self.devices),
            "available_device_count": len(self.available_devices),
            "accelerator_device_count": len(self.accelerator_devices),
            "by_kind": by_kind,
            "devices": [device.to_dict() for device in self.devices],
            "attributes": to_json_safe(dict(self.attributes)),
        }


# ------------------------------------------------------------------
# Selection result
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AcceleratorSelection:
    """
    Completed accelerator-manager selection.

    This object records both the selected device and the policy reason
    so callers can observe fallback behavior.
    """

    device: AcceleratorDevice
    requirements: AcceleratorRequirements
    reason: str

    fallback_used: bool = False
    candidates_considered: Tuple[str, ...] = ()
    rejected_candidates: Mapping[
        str,
        Tuple[str, ...],
    ] = field(default_factory=dict)

    selected_at: str = field(default_factory=utc_now)

    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.device.validate()
        self.requirements.validate()

        _required_text(
            self.reason,
            name="selection reason",
            maximum_length=2048,
        )

        _required_text(
            self.selected_at,
            name="selected_at",
            maximum_length=128,
        )

        if not self.device.available:
            raise ValueError("Selected accelerator device must be available")

        _unique_strings(
            self.candidates_considered,
            name="candidates_considered",
        )

        for device_id, reasons in self.rejected_candidates.items():
            _required_text(
                device_id,
                name="rejected candidate device_id",
                maximum_length=256,
            )

            _unique_strings(
                reasons,
                name=f"rejection reasons for {device_id}",
            )

        if (
            self.fallback_used
            and self.device.kind != AcceleratorKind.CPU.value
            and self.requirements.preferred_kind != AcceleratorKind.AUTO.value
        ):
            # A future policy could fall back between GPU backends,
            # so this remains informational rather than an error.
            pass

    @property
    def runtime_device(self) -> str:
        return self.device.runtime_device

    @property
    def kind(self) -> str:
        return self.device.kind

    @property
    def device_id(self) -> str:
        return self.device.device_id

    def to_dict(self) -> Dict[str, Any]:
        self.validate()

        return {
            "schema_version": ACCELERATOR_SCHEMA_VERSION,
            "device": self.device.to_dict(),
            "requirements": self.requirements.to_dict(),
            "reason": self.reason,
            "fallback_used": self.fallback_used,
            "candidates_considered": list(self.candidates_considered),
            "rejected_candidates": {
                device_id: list(reasons)
                for device_id, reasons in self.rejected_candidates.items()
            },
            "selected_at": self.selected_at,
            "attributes": to_json_safe(dict(self.attributes)),
        }


# ------------------------------------------------------------------
# Backend health
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AcceleratorHealth:
    """
    Health result for one accelerator backend or device.
    """

    component: str
    status: str
    message: str

    kind: Optional[str] = None
    device_id: Optional[str] = None

    retryable: bool = False
    error_type: Optional[str] = None

    attributes: Mapping[str, Any] = field(default_factory=dict)

    checked_at: str = field(default_factory=utc_now)

    def validate(self) -> None:
        _required_text(
            self.component,
            name="health component",
            maximum_length=512,
        )

        _required_text(
            self.message,
            name="health message",
            maximum_length=4096,
        )

        _required_text(
            self.checked_at,
            name="checked_at",
            maximum_length=128,
        )

        if self.status not in {item.value for item in AcceleratorHealthStatus}:
            raise ValueError(f"Unknown accelerator health status: {self.status}")

        if self.kind is not None and self.kind not in {
            AcceleratorKind.CPU.value,
            AcceleratorKind.CUDA.value,
            AcceleratorKind.ROCM.value,
        }:
            raise ValueError(f"Unknown accelerator health kind: {self.kind}")

        _optional_text(
            self.device_id,
            name="health device_id",
            maximum_length=256,
        )

        _optional_text(
            self.error_type,
            name="health error_type",
            maximum_length=512,
        )

    @property
    def healthy(self) -> bool:
        return self.status == (AcceleratorHealthStatus.HEALTHY.value)

    @property
    def usable(self) -> bool:
        return self.status in {
            AcceleratorHealthStatus.HEALTHY.value,
            AcceleratorHealthStatus.DEGRADED.value,
        }

    def to_dict(self) -> Dict[str, Any]:
        self.validate()

        return {
            "schema_version": ACCELERATOR_SCHEMA_VERSION,
            "component": self.component,
            "status": self.status,
            "message": self.message,
            "kind": self.kind,
            "device_id": self.device_id,
            "healthy": self.healthy,
            "usable": self.usable,
            "retryable": self.retryable,
            "error_type": self.error_type,
            "attributes": to_json_safe(dict(self.attributes)),
            "checked_at": self.checked_at,
        }


# ------------------------------------------------------------------
# Aggregate health report
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AcceleratorHealthReport:
    """
    Aggregate health report covering registered accelerator backends.
    """

    components: Tuple[AcceleratorHealth, ...]
    inventory: Optional[AcceleratorInventory] = None
    checked_at: str = field(default_factory=utc_now)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _required_text(
            self.checked_at,
            name="checked_at",
            maximum_length=128,
        )

        for component in self.components:
            component.validate()

        if self.inventory is not None:
            self.inventory.validate()

    @property
    def status(self) -> str:
        self.validate()

        statuses = {component.status for component in self.components}

        if AcceleratorHealthStatus.UNHEALTHY.value in statuses:
            return AcceleratorHealthStatus.UNHEALTHY.value

        if AcceleratorHealthStatus.DEGRADED.value in statuses:
            return AcceleratorHealthStatus.DEGRADED.value

        if AcceleratorHealthStatus.HEALTHY.value in statuses:
            return AcceleratorHealthStatus.HEALTHY.value

        if AcceleratorHealthStatus.UNAVAILABLE.value in statuses:
            return AcceleratorHealthStatus.UNAVAILABLE.value

        return AcceleratorHealthStatus.UNKNOWN.value

    @property
    def ready(self) -> bool:
        return self.status in {
            AcceleratorHealthStatus.HEALTHY.value,
            AcceleratorHealthStatus.DEGRADED.value,
        }

    def to_dict(self) -> Dict[str, Any]:
        self.validate()

        return {
            "schema_version": ACCELERATOR_SCHEMA_VERSION,
            "status": self.status,
            "ready": self.ready,
            "component_count": len(self.components),
            "components": [component.to_dict() for component in self.components],
            "inventory": (
                self.inventory.to_dict() if self.inventory is not None else None
            ),
            "checked_at": self.checked_at,
            "attributes": to_json_safe(dict(self.attributes)),
        }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "ACCELERATOR_SCHEMA_VERSION",
    "AcceleratorKind",
    "AcceleratorState",
    "AcceleratorHealthStatus",
    "AcceleratorPrecision",
    "AcceleratorCapability",
    "AcceleratorMemory",
    "AcceleratorDevice",
    "AcceleratorRequirements",
    "AcceleratorInventory",
    "AcceleratorSelection",
    "AcceleratorHealth",
    "AcceleratorHealthReport",
]
