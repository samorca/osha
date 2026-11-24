from __future__ import annotations

"""
AMD ROCm/HIP runtime telemetry for OSHA.

OSHA logical identity:

    kind:
        rocm

    device_id:
        rocm:N

PyTorch HIP runtime identity:

    runtime_device:
        cuda:N

PyTorch intentionally exposes ROCm/HIP devices
through torch.cuda APIs. This module preserves the
logical ROCm identity while using the runtime APIs
required by PyTorch.

Responsibilities:

    global ROCm VRAM metrics
    PyTorch allocator metrics
    peak allocator metrics
    device utilization when supported
    memory-controller utilization when supported
    temperature when supported
    structured snapshots
    snapshot deltas
    optimization telemetry

This module is read-only except for the explicit
reset_peak_memory_stats() operation.

It does not:

    select devices
    clear caches
    synchronize workloads automatically
    change allocator configuration
    enable mixed precision
    compile models
    execute training or inference
"""

import math
import time

from dataclasses import (
    dataclass,
    field,
)

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

from src.accelerators.rocm.backend import (
    ROCmBackend,
    require_rocm_index,
    rocm_device_id,
    rocm_runtime_available,
    rocm_runtime_device_name,
    rocm_runtime_version,
    torch_available,
    torch_runtime_version,
)

from src.accelerators.rocm.devices import (
    ROCmDeviceInspector,
    ROCmDeviceProfile,
    resolve_rocm_device,
)

from src.accelerators.schema import (
    AcceleratorDevice,
    AcceleratorKind,
)

from src.core.serialization import (
    to_json_safe,
)

from src.core.time import (
    utc_now,
)


# ------------------------------------------------------------------
# Optional PyTorch runtime
# ------------------------------------------------------------------

try:
    import torch

except ImportError:
    torch = None


# ------------------------------------------------------------------
# Schema version
# ------------------------------------------------------------------

ROCM_METRICS_SCHEMA_VERSION = "1"


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------


def _non_negative_integer(
    value: Optional[int],
    *,
    name: str,
) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be >= 0")


def _finite_non_negative(
    value: Optional[float],
    *,
    name: str,
) -> None:
    if value is None:
        return

    numeric = float(value)

    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")


def _percentage(
    value: Optional[float],
    *,
    name: str,
) -> None:
    if value is None:
        return

    numeric = float(value)

    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 100.0:
        raise ValueError(f"{name} must be between 0 and 100")


# ------------------------------------------------------------------
# Global VRAM metrics
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ROCmGlobalMemoryMetrics:
    total_bytes: Optional[int] = None

    free_bytes: Optional[int] = None

    used_bytes: Optional[int] = None

    utilization_percent: Optional[float] = None

    def validate(
        self,
    ) -> None:
        for name, value in (
            (
                "total_bytes",
                self.total_bytes,
            ),
            (
                "free_bytes",
                self.free_bytes,
            ),
            (
                "used_bytes",
                self.used_bytes,
            ),
        ):
            _non_negative_integer(
                value,
                name=name,
            )

        _percentage(
            self.utilization_percent,
            name=("ROCm global memory utilization"),
        )

        if (
            self.total_bytes is not None
            and self.free_bytes is not None
            and self.free_bytes > self.total_bytes
        ):
            raise ValueError("ROCm free memory cannot exceed total memory.")

    @property
    def free_ratio(
        self,
    ) -> Optional[float]:
        if self.total_bytes is None or self.total_bytes <= 0 or self.free_bytes is None:
            return None

        return self.free_bytes / self.total_bytes

    @property
    def used_ratio(
        self,
    ) -> Optional[float]:
        if self.total_bytes is None or self.total_bytes <= 0 or self.used_bytes is None:
            return None

        return self.used_bytes / self.total_bytes

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "total_bytes": (self.total_bytes),
            "free_bytes": (self.free_bytes),
            "used_bytes": (self.used_bytes),
            "utilization_percent": (self.utilization_percent),
            "free_ratio": (self.free_ratio),
            "used_ratio": (self.used_ratio),
        }


# ------------------------------------------------------------------
# PyTorch allocator metrics
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ROCmAllocatorMetrics:
    """
    PyTorch HIP allocator telemetry.

    These values cover memory managed through the
    PyTorch allocator. They do not represent every
    allocation that may exist on the AMD device.
    """

    allocated_bytes: Optional[int] = None

    reserved_bytes: Optional[int] = None

    peak_allocated_bytes: Optional[int] = None

    peak_reserved_bytes: Optional[int] = None

    inactive_reserved_bytes: Optional[int] = None

    allocator_reserved_ratio: Optional[float] = None

    def validate(
        self,
    ) -> None:
        for name, value in (
            (
                "allocated_bytes",
                self.allocated_bytes,
            ),
            (
                "reserved_bytes",
                self.reserved_bytes,
            ),
            (
                "peak_allocated_bytes",
                self.peak_allocated_bytes,
            ),
            (
                "peak_reserved_bytes",
                self.peak_reserved_bytes,
            ),
            (
                "inactive_reserved_bytes",
                self.inactive_reserved_bytes,
            ),
        ):
            _non_negative_integer(
                value,
                name=name,
            )

        if (
            self.allocated_bytes is not None
            and self.reserved_bytes is not None
            and self.allocated_bytes > self.reserved_bytes
        ):
            raise ValueError(
                "ROCm allocated memory cannot exceed PyTorch reserved memory."
            )

        if self.allocator_reserved_ratio is not None:
            ratio = float(self.allocator_reserved_ratio)

            if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
                raise ValueError("allocator_reserved_ratio must be between 0 and 1")

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "allocated_bytes": (self.allocated_bytes),
            "reserved_bytes": (self.reserved_bytes),
            "peak_allocated_bytes": (self.peak_allocated_bytes),
            "peak_reserved_bytes": (self.peak_reserved_bytes),
            "inactive_reserved_bytes": (self.inactive_reserved_bytes),
            "allocator_reserved_ratio": (self.allocator_reserved_ratio),
        }


# ------------------------------------------------------------------
# Device activity metrics
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ROCmActivityMetrics:
    device_utilization_percent: Optional[float] = None

    memory_utilization_percent: Optional[float] = None

    temperature_celsius: Optional[float] = None

    def validate(
        self,
    ) -> None:
        _percentage(
            self.device_utilization_percent,
            name=("ROCm device utilization"),
        )

        _percentage(
            self.memory_utilization_percent,
            name=("ROCm memory-controller utilization"),
        )

        _finite_non_negative(
            self.temperature_celsius,
            name="temperature_celsius",
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "device_utilization_percent": (self.device_utilization_percent),
            "memory_utilization_percent": (self.memory_utilization_percent),
            "temperature_celsius": (self.temperature_celsius),
        }


# ------------------------------------------------------------------
# ROCm metrics snapshot
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ROCmMetricsSnapshot:
    """
    One read-only ROCm/HIP telemetry snapshot.
    """

    device_id: str

    device_index: int

    runtime_device: str

    memory: ROCmGlobalMemoryMetrics = field(default_factory=(ROCmGlobalMemoryMetrics))

    allocator: ROCmAllocatorMetrics = field(default_factory=(ROCmAllocatorMetrics))

    activity: ROCmActivityMetrics = field(default_factory=(ROCmActivityMetrics))

    architecture: Optional[str] = None

    rocm_runtime_version: Optional[str] = None

    torch_runtime_version: Optional[str] = None

    sampled_at: str = field(default_factory=utc_now)

    monotonic_time: float = field(default_factory=(time.monotonic))

    attributes: Mapping[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        expected_id = rocm_device_id(self.device_index)

        expected_runtime = rocm_runtime_device_name(self.device_index)

        if self.device_id != expected_id:
            raise ValueError("ROCm logical device ID does not match its index.")

        if self.runtime_device != expected_runtime:
            raise ValueError("ROCm PyTorch runtime device must be cuda:N.")

        if self.device_index < 0:
            raise ValueError("device_index must be >= 0")

        self.memory.validate()
        self.allocator.validate()
        self.activity.validate()

        _finite_non_negative(
            self.monotonic_time,
            name="monotonic_time",
        )

    @property
    def global_memory_pressure(
        self,
    ) -> Optional[float]:
        return self.memory.used_ratio

    @property
    def allocator_headroom_bytes(
        self,
    ) -> Optional[int]:
        return self.memory.free_bytes

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "schema_version": (ROCM_METRICS_SCHEMA_VERSION),
            "device_id": (self.device_id),
            "device_index": (self.device_index),
            "runtime_device": (self.runtime_device),
            "architecture": (self.architecture),
            "memory": (self.memory.to_dict()),
            "allocator": (self.allocator.to_dict()),
            "activity": (self.activity.to_dict()),
            "global_memory_pressure": (self.global_memory_pressure),
            "allocator_headroom_bytes": (self.allocator_headroom_bytes),
            "rocm_runtime_version": (self.rocm_runtime_version),
            "torch_runtime_version": (self.torch_runtime_version),
            "sampled_at": (self.sampled_at),
            "monotonic_time": (self.monotonic_time),
            "attributes": (to_json_safe(dict(self.attributes))),
        }


# ------------------------------------------------------------------
# Metrics delta
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ROCmMetricsDelta:
    elapsed_seconds: float

    free_memory_change_bytes: Optional[int] = None

    allocated_memory_change_bytes: Optional[int] = None

    reserved_memory_change_bytes: Optional[int] = None

    peak_allocated_change_bytes: Optional[int] = None

    peak_reserved_change_bytes: Optional[int] = None

    device_utilization_change_percent: Optional[float] = None

    def validate(
        self,
    ) -> None:
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and >= 0")

        if self.device_utilization_change_percent is not None and not math.isfinite(
            float(self.device_utilization_change_percent)
        ):
            raise ValueError("device_utilization_change_percent must be finite")

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "elapsed_seconds": (self.elapsed_seconds),
            "free_memory_change_bytes": (self.free_memory_change_bytes),
            "allocated_memory_change_bytes": (self.allocated_memory_change_bytes),
            "reserved_memory_change_bytes": (self.reserved_memory_change_bytes),
            "peak_allocated_change_bytes": (self.peak_allocated_change_bytes),
            "peak_reserved_change_bytes": (self.peak_reserved_change_bytes),
            "device_utilization_change_percent": (
                self.device_utilization_change_percent
            ),
        }


# ------------------------------------------------------------------
# Runtime guard
# ------------------------------------------------------------------


def rocm_metrics_available() -> bool:
    return bool(
        torch is not None
        and torch_available()
        and rocm_runtime_available()
        and rocm_runtime_version()
    )


def require_rocm_metrics_runtime() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is unavailable for ROCm metrics.")

    if not rocm_runtime_version():
        raise RuntimeError("The installed PyTorch runtime is not a ROCm/HIP build.")

    if not torch.cuda.is_available():
        raise RuntimeError("The ROCm/HIP runtime has no available accelerator device.")


# ------------------------------------------------------------------
# Global memory
# ------------------------------------------------------------------


def read_global_memory(
    index: int,
) -> ROCmGlobalMemoryMetrics:
    require_rocm_metrics_runtime()

    resolved = require_rocm_index(index)

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(resolved)

        total = int(total_bytes)

        free = int(free_bytes)

        used = max(
            0,
            total - free,
        )

        utilization = (used / total) * 100.0 if total > 0 else None

        result = ROCmGlobalMemoryMetrics(
            total_bytes=total,
            free_bytes=free,
            used_bytes=used,
            utilization_percent=(utilization),
        )

        result.validate()

        return result

    except Exception:
        return ROCmGlobalMemoryMetrics()


# ------------------------------------------------------------------
# Allocator metrics
# ------------------------------------------------------------------


def read_allocator_metrics(
    index: int,
) -> ROCmAllocatorMetrics:
    require_rocm_metrics_runtime()

    resolved = require_rocm_index(index)

    try:
        allocated = int(torch.cuda.memory_allocated(resolved))

    except Exception:
        allocated = None

    try:
        reserved = int(torch.cuda.memory_reserved(resolved))

    except Exception:
        reserved = None

    try:
        peak_allocated = int(torch.cuda.max_memory_allocated(resolved))

    except Exception:
        peak_allocated = None

    try:
        peak_reserved = int(torch.cuda.max_memory_reserved(resolved))

    except Exception:
        peak_reserved = None

    inactive = None
    reserved_ratio = None

    if allocated is not None and reserved is not None:
        inactive = max(
            0,
            reserved - allocated,
        )

        if reserved > 0:
            reserved_ratio = allocated / reserved

    result = ROCmAllocatorMetrics(
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        peak_allocated_bytes=(peak_allocated),
        peak_reserved_bytes=(peak_reserved),
        inactive_reserved_bytes=(inactive),
        allocator_reserved_ratio=(reserved_ratio),
    )

    result.validate()

    return result


# ------------------------------------------------------------------
# Device-reported memory use
# ------------------------------------------------------------------


def read_device_memory_used(
    index: int,
) -> Optional[int]:
    """
    Read device-wide used memory when the
    installed PyTorch runtime exposes the API.

    This measurement may include allocations that
    are outside PyTorch's allocator.
    """

    require_rocm_metrics_runtime()

    resolved = require_rocm_index(index)

    reader = getattr(
        torch.cuda,
        "device_memory_used",
        None,
    )

    if not callable(reader):
        return None

    try:
        value = int(reader(resolved))

        if value < 0:
            return None

        return value

    except Exception:
        return None


# ------------------------------------------------------------------
# Device utilization
# ------------------------------------------------------------------


def read_device_utilization(
    index: int,
) -> Optional[float]:
    """
    Read accelerator utilization when supported by
    the PyTorch ROCm runtime and system tooling.

    Unsupported readings return None.
    """

    require_rocm_metrics_runtime()

    resolved = require_rocm_index(index)

    reader = getattr(
        torch.cuda,
        "utilization",
        None,
    )

    if not callable(reader):
        return None

    try:
        value = float(reader(resolved))

        _percentage(
            value,
            name=("ROCm device utilization"),
        )

        return value

    except Exception:
        return None


# ------------------------------------------------------------------
# Memory-controller utilization
# ------------------------------------------------------------------


def read_memory_utilization(
    index: int,
) -> Optional[float]:
    require_rocm_metrics_runtime()

    resolved = require_rocm_index(index)

    reader = getattr(
        torch.cuda,
        "memory_usage",
        None,
    )

    if not callable(reader):
        return None

    try:
        value = float(reader(resolved))

        _percentage(
            value,
            name=("ROCm memory utilization"),
        )

        return value

    except Exception:
        return None


# ------------------------------------------------------------------
# Temperature
# ------------------------------------------------------------------


def read_temperature(
    index: int,
) -> Optional[float]:
    require_rocm_metrics_runtime()

    resolved = require_rocm_index(index)

    reader = getattr(
        torch.cuda,
        "temperature",
        None,
    )

    if not callable(reader):
        return None

    try:
        value = float(reader(resolved))

        _finite_non_negative(
            value,
            name=("ROCm temperature"),
        )

        return value

    except Exception:
        return None


# ------------------------------------------------------------------
# Activity
# ------------------------------------------------------------------


def read_activity(
    index: int,
) -> ROCmActivityMetrics:
    result = ROCmActivityMetrics(
        device_utilization_percent=(read_device_utilization(index)),
        memory_utilization_percent=(read_memory_utilization(index)),
        temperature_celsius=(read_temperature(index)),
    )

    result.validate()

    return result


# ------------------------------------------------------------------
# Peak-memory reset
# ------------------------------------------------------------------


def reset_peak_memory_stats(
    index: int,
) -> bool:
    """
    Explicitly reset PyTorch ROCm/HIP allocator
    peak-memory statistics.

    This changes telemetry counters and is
    intentionally opt-in.
    """

    require_rocm_metrics_runtime()

    resolved = require_rocm_index(index)

    resetter = getattr(
        torch.cuda,
        "reset_peak_memory_stats",
        None,
    )

    if not callable(resetter):
        return False

    try:
        resetter(resolved)

        return True

    except Exception:
        return False


# ------------------------------------------------------------------
# Metrics collector
# ------------------------------------------------------------------


class ROCmMetricsCollector:
    """
    Read-only ROCm metrics collector bound to one
    logical ROCm device.
    """

    def __init__(
        self,
        *,
        device: Optional[AcceleratorDevice] = None,
        index: int = 0,
        backend: Optional[ROCmBackend] = None,
        inspector: Optional[ROCmDeviceInspector] = None,
    ) -> None:
        self.backend = backend or ROCmBackend()

        self.inspector = inspector or ROCmDeviceInspector(backend=self.backend)

        self.device = device or resolve_rocm_device(
            index,
            backend=self.backend,
        )

        self.device.validate()

        if self.device.kind != AcceleratorKind.ROCM.value:
            raise ValueError("ROCmMetricsCollector requires a ROCm device.")

        self.index = require_rocm_index(self.device.index)

        expected_id = rocm_device_id(self.index)

        expected_runtime = rocm_runtime_device_name(self.index)

        if self.device.device_id != expected_id:
            raise ValueError("ROCm logical device ID does not match its index.")

        if self.device.runtime_device != expected_runtime:
            raise ValueError("ROCm runtime device must use cuda:N.")

    # ---------------------------------------------------------
    # Collect
    # ---------------------------------------------------------

    def collect(
        self,
        *,
        refresh_profile: bool = False,
    ) -> ROCmMetricsSnapshot:
        require_rocm_metrics_runtime()

        profile: Optional[ROCmDeviceProfile] = None

        if refresh_profile:
            profile = self.inspector.inspect(
                self.device,
                refresh_memory=True,
            )

        memory = read_global_memory(self.index)

        allocator = read_allocator_metrics(self.index)

        activity = read_activity(self.index)

        device_memory_used = read_device_memory_used(self.index)

        architecture = (
            profile.architecture if profile is not None else self.device.architecture
        )

        snapshot = ROCmMetricsSnapshot(
            device_id=(self.device.device_id),
            device_index=(self.index),
            runtime_device=(self.device.runtime_device),
            memory=memory,
            allocator=allocator,
            activity=activity,
            architecture=(architecture),
            rocm_runtime_version=(rocm_runtime_version()),
            torch_runtime_version=(torch_runtime_version()),
            attributes={
                "torch_available": (torch_available()),
                "rocm_runtime_available": (rocm_runtime_available()),
                "pytorch_uses_cuda_interface": (True),
                "device_memory_used_bytes": (device_memory_used),
                "profile": (profile.to_dict() if profile is not None else None),
            },
        )

        snapshot.validate()

        return snapshot


# ------------------------------------------------------------------
# Metrics delta
# ------------------------------------------------------------------


def metrics_delta(
    before: ROCmMetricsSnapshot,
    after: ROCmMetricsSnapshot,
) -> ROCmMetricsDelta:
    before.validate()
    after.validate()

    if before.device_id != after.device_id:
        raise ValueError("ROCm metrics snapshots belong to different devices.")

    elapsed = max(
        0.0,
        after.monotonic_time - before.monotonic_time,
    )

    def integer_delta(
        old: Optional[int],
        new: Optional[int],
    ) -> Optional[int]:
        if old is None or new is None:
            return None

        return new - old

    utilization_delta = None

    if (
        before.activity.device_utilization_percent is not None
        and after.activity.device_utilization_percent is not None
    ):
        utilization_delta = (
            after.activity.device_utilization_percent
            - before.activity.device_utilization_percent
        )

    result = ROCmMetricsDelta(
        elapsed_seconds=elapsed,
        free_memory_change_bytes=(
            integer_delta(
                before.memory.free_bytes,
                after.memory.free_bytes,
            )
        ),
        allocated_memory_change_bytes=(
            integer_delta(
                before.allocator.allocated_bytes,
                after.allocator.allocated_bytes,
            )
        ),
        reserved_memory_change_bytes=(
            integer_delta(
                before.allocator.reserved_bytes,
                after.allocator.reserved_bytes,
            )
        ),
        peak_allocated_change_bytes=(
            integer_delta(
                before.allocator.peak_allocated_bytes,
                after.allocator.peak_allocated_bytes,
            )
        ),
        peak_reserved_change_bytes=(
            integer_delta(
                before.allocator.peak_reserved_bytes,
                after.allocator.peak_reserved_bytes,
            )
        ),
        device_utilization_change_percent=(utilization_delta),
    )

    result.validate()

    return result


# ------------------------------------------------------------------
# Optimization attributes
# ------------------------------------------------------------------


def optimization_metrics_attributes(
    snapshot: ROCmMetricsSnapshot,
) -> Dict[str, Any]:
    """
    Translate ROCm telemetry into bounded
    optimizer input.

    No optimization recommendation is generated
    here.
    """

    snapshot.validate()

    return {
        "device_id": (snapshot.device_id),
        "device_index": (snapshot.device_index),
        "runtime_device": (snapshot.runtime_device),
        "architecture": (snapshot.architecture),
        "global_memory": (snapshot.memory.to_dict()),
        "allocator": (snapshot.allocator.to_dict()),
        "device_utilization_percent": (snapshot.activity.device_utilization_percent),
        "memory_utilization_percent": (snapshot.activity.memory_utilization_percent),
        "temperature_celsius": (snapshot.activity.temperature_celsius),
        "global_memory_pressure": (snapshot.global_memory_pressure),
        "device_memory_used_bytes": (
            snapshot.attributes.get("device_memory_used_bytes")
        ),
        "rocm_runtime_version": (snapshot.rocm_runtime_version),
        "torch_runtime_version": (snapshot.torch_runtime_version),
        "pytorch_uses_cuda_interface": (True),
    }


# ------------------------------------------------------------------
# Convenience collection
# ------------------------------------------------------------------


def collect_rocm_metrics(
    index: int = 0,
    *,
    device: Optional[AcceleratorDevice] = None,
    backend: Optional[ROCmBackend] = None,
    refresh_profile: bool = False,
) -> ROCmMetricsSnapshot:
    return ROCmMetricsCollector(
        device=device,
        index=index,
        backend=backend,
    ).collect(refresh_profile=(refresh_profile))


def get_memory_metrics(index: int = 0) -> Dict[str, Any]:
    try:
        import torch as runtime
    except ImportError:
        raise RuntimeError("ROCm runtime is unavailable.") from None

    cuda = getattr(runtime, "cuda", None)
    version = getattr(runtime, "version", None)
    available = getattr(cuda, "is_available", None)
    if not callable(available) or not available() or not getattr(version, "hip", None):
        raise RuntimeError("ROCm runtime is unavailable.")
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("Device index must be an integer.")
    count = int(cuda.device_count())
    if index < 0 or index >= count:
        raise ValueError(f"Invalid ROCm device index {index}.")

    device = runtime.device(f"cuda:{index}")
    free, total = cuda.mem_get_info(device)
    return {
        "backend": "rocm",
        "index": index,
        "total_memory_bytes": int(total),
        "free_memory_bytes": int(free),
        "used_memory_bytes": max(0, int(total) - int(free)),
        "allocated_memory_bytes": int(cuda.memory_allocated(device)),
        "reserved_memory_bytes": int(cuda.memory_reserved(device)),
    }


def get_all_memory_metrics() -> list[Dict[str, Any]]:
    try:
        import torch as runtime
    except ImportError:
        return []
    cuda = getattr(runtime, "cuda", None)
    version = getattr(runtime, "version", None)
    available = getattr(cuda, "is_available", None)
    if not callable(available) or not available() or not getattr(version, "hip", None):
        return []
    return [get_memory_metrics(index) for index in range(int(cuda.device_count()))]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "ROCM_METRICS_SCHEMA_VERSION",
    "ROCmGlobalMemoryMetrics",
    "ROCmAllocatorMetrics",
    "ROCmActivityMetrics",
    "ROCmMetricsSnapshot",
    "ROCmMetricsDelta",
    "rocm_metrics_available",
    "require_rocm_metrics_runtime",
    "read_global_memory",
    "read_allocator_metrics",
    "read_device_memory_used",
    "read_device_utilization",
    "read_memory_utilization",
    "read_temperature",
    "read_activity",
    "reset_peak_memory_stats",
    "ROCmMetricsCollector",
    "metrics_delta",
    "optimization_metrics_attributes",
    "collect_rocm_metrics",
    "get_memory_metrics",
    "get_all_memory_metrics",
]
