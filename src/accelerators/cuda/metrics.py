from __future__ import annotations

"""
NVIDIA CUDA runtime telemetry for OSHA.

Responsibilities:

    global CUDA VRAM metrics
    PyTorch allocator metrics
    peak allocator metrics
    CUDA device utilization
    memory-controller utilization
    temperature when available
    device-count/runtime metadata
    structured snapshots
    snapshot deltas
    optimization telemetry

This module is read-only except for the optional
explicit reset_peak_memory_stats() operation.

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

from src.accelerators.cuda.backend import (
    CUDABackend,
    cuda_runtime_version,
    hip_runtime_available,
    require_cuda_index,
    torch_available,
    torch_runtime_version,
)

from src.accelerators.cuda.devices import (
    CUDADeviceInspector,
    CUDADeviceProfile,
    resolve_cuda_device,
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

CUDA_METRICS_SCHEMA_VERSION = "1"


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------

def _non_negative_integer(
    value: Optional[int],
    *,
    name: str,
) -> None:
    if (
        value is not None
        and value < 0
    ):
        raise ValueError(
            f"{name} must be >= 0"
        )


def _finite_non_negative(
    value: Optional[float],
    *,
    name: str,
) -> None:
    if value is None:
        return

    numeric = float(
        value
    )

    if (
        not math.isfinite(
            numeric
        )
        or numeric < 0.0
    ):
        raise ValueError(
            f"{name} must be finite "
            "and >= 0"
        )


def _percentage(
    value: Optional[float],
    *,
    name: str,
) -> None:
    if value is None:
        return

    numeric = float(
        value
    )

    if (
        not math.isfinite(
            numeric
        )
        or numeric < 0.0
        or numeric > 100.0
    ):
        raise ValueError(
            f"{name} must be between "
            "0 and 100"
        )


# ------------------------------------------------------------------
# Global VRAM metrics
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CUDAGlobalMemoryMetrics:
    total_bytes: Optional[int] = None

    free_bytes: Optional[int] = None

    used_bytes: Optional[int] = None

    utilization_percent: Optional[
        float
    ] = None

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
            name=(
                "CUDA global memory "
                "utilization"
            ),
        )

        if (
            self.total_bytes is not None
            and self.free_bytes is not None
            and self.free_bytes
            > self.total_bytes
        ):
            raise ValueError(
                "CUDA free memory cannot "
                "exceed total memory."
            )

    @property
    def free_ratio(
        self,
    ) -> Optional[float]:
        if (
            self.total_bytes is None
            or self.total_bytes <= 0
            or self.free_bytes is None
        ):
            return None

        return (
            self.free_bytes
            / self.total_bytes
        )

    @property
    def used_ratio(
        self,
    ) -> Optional[float]:
        if (
            self.total_bytes is None
            or self.total_bytes <= 0
            or self.used_bytes is None
        ):
            return None

        return (
            self.used_bytes
            / self.total_bytes
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "total_bytes": (
                self.total_bytes
            ),

            "free_bytes": (
                self.free_bytes
            ),

            "used_bytes": (
                self.used_bytes
            ),

            "utilization_percent": (
                self.utilization_percent
            ),

            "free_ratio": (
                self.free_ratio
            ),

            "used_ratio": (
                self.used_ratio
            ),
        }


# ------------------------------------------------------------------
# PyTorch allocator metrics
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CUDAAllocatorMetrics:
    """
    PyTorch allocator telemetry.

    These values describe memory managed by the
    PyTorch CUDA caching allocator, not every
    allocation that may exist on the GPU.
    """

    allocated_bytes: Optional[int] = None

    reserved_bytes: Optional[int] = None

    peak_allocated_bytes: Optional[
        int
    ] = None

    peak_reserved_bytes: Optional[
        int
    ] = None

    inactive_reserved_bytes: Optional[
        int
    ] = None

    allocator_reserved_ratio: Optional[
        float
    ] = None

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
            and self.allocated_bytes
            > self.reserved_bytes
        ):
            raise ValueError(
                "CUDA allocated memory "
                "cannot exceed PyTorch "
                "reserved memory."
            )

        if (
            self.allocator_reserved_ratio
            is not None
        ):
            ratio = float(
                self.
                allocator_reserved_ratio
            )

            if (
                not math.isfinite(
                    ratio
                )
                or ratio < 0.0
                or ratio > 1.0
            ):
                raise ValueError(
                    "allocator_reserved_ratio "
                    "must be between 0 and 1"
                )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "allocated_bytes": (
                self.allocated_bytes
            ),

            "reserved_bytes": (
                self.reserved_bytes
            ),

            "peak_allocated_bytes": (
                self.peak_allocated_bytes
            ),

            "peak_reserved_bytes": (
                self.peak_reserved_bytes
            ),

            "inactive_reserved_bytes": (
                self.
                inactive_reserved_bytes
            ),

            "allocator_reserved_ratio": (
                self.
                allocator_reserved_ratio
            ),
        }


# ------------------------------------------------------------------
# Device activity metrics
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CUDAActivityMetrics:
    gpu_utilization_percent: Optional[
        float
    ] = None

    memory_utilization_percent: Optional[
        float
    ] = None

    temperature_celsius: Optional[
        float
    ] = None

    def validate(
        self,
    ) -> None:
        _percentage(
            self.gpu_utilization_percent,
            name=(
                "GPU utilization"
            ),
        )

        _percentage(
            self.
            memory_utilization_percent,
            name=(
                "CUDA memory-controller "
                "utilization"
            ),
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
            "gpu_utilization_percent": (
                self.
                gpu_utilization_percent
            ),

            "memory_utilization_percent": (
                self.
                memory_utilization_percent
            ),

            "temperature_celsius": (
                self.temperature_celsius
            ),
        }


# ------------------------------------------------------------------
# CUDA metrics snapshot
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CUDAMetricsSnapshot:
    """
    One read-only CUDA telemetry snapshot.
    """

    device_id: str

    device_index: int

    runtime_device: str

    memory: CUDAGlobalMemoryMetrics = field(
        default_factory=(
            CUDAGlobalMemoryMetrics
        )
    )

    allocator: CUDAAllocatorMetrics = field(
        default_factory=(
            CUDAAllocatorMetrics
        )
    )

    activity: CUDAActivityMetrics = field(
        default_factory=(
            CUDAActivityMetrics
        )
    )

    cuda_runtime_version: Optional[
        str
    ] = None

    torch_runtime_version: Optional[
        str
    ] = None

    sampled_at: str = field(
        default_factory=utc_now
    )

    monotonic_time: float = field(
        default_factory=(
            time.monotonic
        )
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
        if not str(
            self.device_id
        ).strip():
            raise ValueError(
                "device_id cannot be empty"
            )

        if self.device_index < 0:
            raise ValueError(
                "device_index must be >= 0"
            )

        if not str(
            self.runtime_device
        ).strip():
            raise ValueError(
                "runtime_device cannot "
                "be empty"
            )

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
        """
        Fraction of total VRAM currently not
        reported as free by the runtime.
        """

        return self.memory.used_ratio

    @property
    def allocator_headroom_bytes(
        self,
    ) -> Optional[int]:
        if (
            self.memory.free_bytes
            is None
        ):
            return None

        return self.memory.free_bytes

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "schema_version": (
                CUDA_METRICS_SCHEMA_VERSION
            ),

            "device_id": (
                self.device_id
            ),

            "device_index": (
                self.device_index
            ),

            "runtime_device": (
                self.runtime_device
            ),

            "memory": (
                self.memory.to_dict()
            ),

            "allocator": (
                self.allocator.to_dict()
            ),

            "activity": (
                self.activity.to_dict()
            ),

            "global_memory_pressure": (
                self.global_memory_pressure
            ),

            "allocator_headroom_bytes": (
                self.
                allocator_headroom_bytes
            ),

            "cuda_runtime_version": (
                self.cuda_runtime_version
            ),

            "torch_runtime_version": (
                self.torch_runtime_version
            ),

            "sampled_at": (
                self.sampled_at
            ),

            "monotonic_time": (
                self.monotonic_time
            ),

            "attributes": (
                to_json_safe(
                    dict(
                        self.attributes
                    )
                )
            ),
        }


# ------------------------------------------------------------------
# Metrics delta
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CUDAMetricsDelta:
    elapsed_seconds: float

    free_memory_change_bytes: Optional[
        int
    ] = None

    allocated_memory_change_bytes: Optional[
        int
    ] = None

    reserved_memory_change_bytes: Optional[
        int
    ] = None

    peak_allocated_change_bytes: Optional[
        int
    ] = None

    peak_reserved_change_bytes: Optional[
        int
    ] = None

    gpu_utilization_change_percent: Optional[
        float
    ] = None

    def validate(
        self,
    ) -> None:
        if (
            not math.isfinite(
                self.elapsed_seconds
            )
            or self.elapsed_seconds < 0
        ):
            raise ValueError(
                "elapsed_seconds must be "
                "finite and >= 0"
            )

        if (
            self.gpu_utilization_change_percent
            is not None
            and not math.isfinite(
                float(
                    self.
                    gpu_utilization_change_percent
                )
            )
        ):
            raise ValueError(
                "gpu_utilization_change_percent "
                "must be finite"
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "elapsed_seconds": (
                self.elapsed_seconds
            ),

            "free_memory_change_bytes": (
                self.
                free_memory_change_bytes
            ),

            "allocated_memory_change_bytes": (
                self.
                allocated_memory_change_bytes
            ),

            "reserved_memory_change_bytes": (
                self.
                reserved_memory_change_bytes
            ),

            "peak_allocated_change_bytes": (
                self.
                peak_allocated_change_bytes
            ),

            "peak_reserved_change_bytes": (
                self.
                peak_reserved_change_bytes
            ),

            "gpu_utilization_change_percent": (
                self.
                gpu_utilization_change_percent
            ),
        }


# ------------------------------------------------------------------
# Runtime guard
# ------------------------------------------------------------------

def cuda_metrics_available() -> bool:
    return bool(
        torch is not None
        and torch_available()
        and not hip_runtime_available()
        and torch.cuda.is_available()
    )


def require_cuda_metrics_runtime() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is unavailable for "
            "CUDA metrics."
        )

    if hip_runtime_available():
        raise RuntimeError(
            "The installed PyTorch runtime "
            "uses ROCm/HIP rather than "
            "NVIDIA CUDA."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable."
        )


# ------------------------------------------------------------------
# Global memory
# ------------------------------------------------------------------

def read_global_memory(
    index: int,
) -> CUDAGlobalMemoryMetrics:
    require_cuda_metrics_runtime()

    resolved = require_cuda_index(
        index
    )

    try:
        free_bytes, total_bytes = (
            torch.cuda.mem_get_info(
                resolved
            )
        )

        total = int(
            total_bytes
        )

        free = int(
            free_bytes
        )

        used = max(
            0,
            total - free,
        )

        utilization = (
            (
                used
                / total
            )
            * 100.0
            if total > 0
            else None
        )

        result = (
            CUDAGlobalMemoryMetrics(
                total_bytes=total,
                free_bytes=free,
                used_bytes=used,
                utilization_percent=(
                    utilization
                ),
            )
        )

        result.validate()

        return result

    except Exception:
        return CUDAGlobalMemoryMetrics()


# ------------------------------------------------------------------
# Allocator metrics
# ------------------------------------------------------------------

def read_allocator_metrics(
    index: int,
) -> CUDAAllocatorMetrics:
    require_cuda_metrics_runtime()

    resolved = require_cuda_index(
        index
    )

    try:
        allocated = int(
            torch.cuda.memory_allocated(
                resolved
            )
        )

    except Exception:
        allocated = None

    try:
        reserved = int(
            torch.cuda.memory_reserved(
                resolved
            )
        )

    except Exception:
        reserved = None

    try:
        peak_allocated = int(
            torch.cuda.
            max_memory_allocated(
                resolved
            )
        )

    except Exception:
        peak_allocated = None

    try:
        peak_reserved = int(
            torch.cuda.
            max_memory_reserved(
                resolved
            )
        )

    except Exception:
        peak_reserved = None

    inactive = None
    reserved_ratio = None

    if (
        allocated is not None
        and reserved is not None
    ):
        inactive = max(
            0,
            reserved - allocated,
        )

        if reserved > 0:
            reserved_ratio = (
                allocated
                / reserved
            )

    result = CUDAAllocatorMetrics(
        allocated_bytes=allocated,

        reserved_bytes=reserved,

        peak_allocated_bytes=(
            peak_allocated
        ),

        peak_reserved_bytes=(
            peak_reserved
        ),

        inactive_reserved_bytes=(
            inactive
        ),

        allocator_reserved_ratio=(
            reserved_ratio
        ),
    )

    result.validate()

    return result


# ------------------------------------------------------------------
# GPU utilization
# ------------------------------------------------------------------

def read_gpu_utilization(
    index: int,
) -> Optional[float]:
    require_cuda_metrics_runtime()

    resolved = require_cuda_index(
        index
    )

    reader = getattr(
        torch.cuda,
        "utilization",
        None,
    )

    if not callable(
        reader
    ):
        return None

    try:
        value = float(
            reader(
                resolved
            )
        )

        _percentage(
            value,
            name="gpu_utilization",
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
    require_cuda_metrics_runtime()

    resolved = require_cuda_index(
        index
    )

    reader = getattr(
        torch.cuda,
        "memory_usage",
        None,
    )

    if not callable(
        reader
    ):
        return None

    try:
        value = float(
            reader(
                resolved
            )
        )

        _percentage(
            value,
            name=(
                "memory_utilization"
            ),
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
    require_cuda_metrics_runtime()

    resolved = require_cuda_index(
        index
    )

    reader = getattr(
        torch.cuda,
        "temperature",
        None,
    )

    if not callable(
        reader
    ):
        return None

    try:
        value = float(
            reader(
                resolved
            )
        )

        _finite_non_negative(
            value,
            name="temperature",
        )

        return value

    except Exception:
        return None


# ------------------------------------------------------------------
# Device activity
# ------------------------------------------------------------------

def read_activity(
    index: int,
) -> CUDAActivityMetrics:
    result = CUDAActivityMetrics(
        gpu_utilization_percent=(
            read_gpu_utilization(
                index
            )
        ),

        memory_utilization_percent=(
            read_memory_utilization(
                index
            )
        ),

        temperature_celsius=(
            read_temperature(
                index
            )
        ),
    )

    result.validate()

    return result


# ------------------------------------------------------------------
# Peak reset
# ------------------------------------------------------------------

def reset_peak_memory_stats(
    index: int,
) -> bool:
    """
    Explicitly reset PyTorch CUDA peak allocator
    statistics.

    Unlike normal metric collection, this changes
    the allocator's telemetry counters, so it is
    intentionally opt-in.
    """

    require_cuda_metrics_runtime()

    resolved = require_cuda_index(
        index
    )

    resetter = getattr(
        torch.cuda,
        "reset_peak_memory_stats",
        None,
    )

    if not callable(
        resetter
    ):
        return False

    try:
        resetter(
            resolved
        )

        return True

    except Exception:
        return False


# ------------------------------------------------------------------
# Metrics collector
# ------------------------------------------------------------------

class CUDAMetricsCollector:
    """
    Read-only CUDA metrics collector.

    A collector is bound to one CUDA device.
    """

    def __init__(
        self,
        *,
        device: Optional[
     : int = 0,
        backend: Optional[
            CUDABackend
        ] = None,
        inspector: Optional[
            CUDADeviceInspector
        ] = None,
    ) -> None:
        self.backend = (
            backend
            or CUDABackend()
        )

        self.inspector = (
            inspector
            or CUDADeviceInspector(
                backend=self.backend
            )
        )

        self.device = (
            device
            or resolve_cuda_device(
                index,
                backend=self.backend,
            )
        )

        self.device.validate()

        if (
            self.device.kind
            != AcceleratorKind.CUDA.value
        ):
            raise ValueError(
                "CUDAMetricsCollector "
                "requires a CUDA device."
            )

        self.index = require_cuda_index(
            self.device.index
        )

    # ---------------------------------------------------------
    # Collect
    # ---------------------------------------------------------

    def collect(
        self,
        *,
        refresh_profile: bool = False,
    ) -> CUDAMetricsSnapshot:
        require_cuda_metrics_runtime()

        profile: Optional[
            CUDADeviceProfile
        ] = None

        if refresh_profile:
            profile = (
                self.inspector.inspect(
                    self.device,
                    refresh_memory=True,
                )
            )

        memory = read_global_memory(
            self.index
        )

        allocator = (
            read_allocator_metrics(
                self.index
            )
        )

        activity = read_activity(
            self.index
        )

        snapshot = CUDAMetricsSnapshot(
            device_id=(
                self.device.device_id
            ),

            device_index=(
                self.index
            ),

            runtime_device=(
                self.device.
                runtime_device
            ),

            memory=memory,

            allocator=allocator,

            activity=activity,

            cuda_runtime_version=(
                cuda_runtime_version()
            ),

            torch_runtime_version=(
                torch_runtime_version()
            ),

            attributes={
                "torch_available": (
                    torch_available()
                ),

                "hip_runtime": (
                    hip_runtime_available()
                ),

                "profile": (
                    profile.to_dict()
                    if profile
                    is not None
                    else None
                ),
            },
        )

        snapshot.validate()

        return snapshot


# ------------------------------------------------------------------
# Metrics delta
# ------------------------------------------------------------------

def metrics_delta(
    before: CUDAMetricsSnapshot,
    after: CUDAMetricsSnapshot,
) -> CUDAMetricsDelta:
    before.validate()
    after.validate()

    if (
        before.device_id
        != after.device_id
    ):
        raise ValueError(
            "CUDA metrics snapshots "
            "belong to different devices."
        )

    elapsed = max(
        0.0,
        after.monotonic_time
        - before.monotonic_time,
    )

    def integer_delta(
        old: Optional[int],
        new: Optional[int],
    ) -> Optional[int]:
        if (
            old is None
            or new is None
        ):
            return None

        return new - old

    utilization_delta = None

    if (
        before.activity.
        gpu_utilization_percent
        is not None
        and after.activity.
        gpu_utilization_percent
        is not None
    ):
        utilization_delta = (
            after.activity.
            gpu_utilization_percent
            - before.activity.
            gpu_utilization_percent
        )

    result = CUDAMetricsDelta(
        elapsed_seconds=elapsed,

        free_memory_change_bytes=(
            integer_delta(
                before.memory.
                free_bytes,
                after.memory.
                free_bytes,
            )
        ),

        allocated_memory_change_bytes=(
            integer_delta(
                before.allocator.
                allocated_bytes,
                after.allocator.
                allocated_bytes,
            )
        ),

        reserved_memory_change_bytes=(
            integer_delta(
                before.allocator.
                reserved_bytes,
                after.allocator.
                reserved_bytes,
            )
        ),

        peak_allocated_change_bytes=(
            integer_delta(
                before.allocator.
                peak_allocated_bytes,
                after.allocator.
                peak_allocated_bytes,
            )
        ),

        peak_reserved_change_bytes=(
            integer_delta(
                before.allocator.
                peak_reserved_bytes,
                after.allocator.
                peak_reserved_bytes,
            )
        ),

        gpu_utilization_change_percent=(
            utilization_delta
        ),
    )

    result.validate()

    return result


# ------------------------------------------------------------------
# Optimization attributes
# ------------------------------------------------------------------

def optimization_metrics_attributes(
    snapshot: CUDAMetricsSnapshot,
) -> Dict[str, Any]:
    """
    Translate CUDA telemetry into bounded
    optimizer input.

    No recommendation is made here.
    """

    snapshot.validate()

    return {
        "device_id": (
            snapshot.device_id
        ),

        "device_index": (
            snapshot.device_index
        ),

        "global_memory": (
            snapshot.memory.to_dict()
        ),

        "allocator": (
            snapshot.
            allocator.to_dict()
        ),

        "gpu_utilization_percent": (
            snapshot.activity.
            gpu_utilization_percent
        ),

        "memory_utilization_percent": (
            snapshot.activity.
            memory_utilization_percent
        ),

        "temperature_celsius": (
            snapshot.activity.
            temperature_celsius
        ),

        "global_memory_pressure": (
            snapshot.
            global_memory_pressure
        ),

        "cuda_runtime_version": (
            snapshot.
            cuda_runtime_version
        ),

        "torch_runtime_version": (
            snapshot.
            torch_runtime_version
        ),
    }


# ------------------------------------------------------------------
# Convenience collection
# ------------------------------------------------------------------

def collect_cuda_metrics(
    index: int = 0,
    *,
    device: Optional[
        AcceleratorDevice
    ] = None,
    backend: Optional[
        CUDABackend
    ] = None,
    refresh_profile: bool = False,
) -> CUDAMetricsSnapshot:
    return CUDAMetricsCollector(
        device=device,
        index=index,
        backend=backend,
    ).collect(
        refresh_profile=(
            refresh_profile
        )
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "CUDA_METRICS_SCHEMA_VERSION",

    "CUDAGlobalMemoryMetrics",
    "CUDAAllocatorMetrics",
    "CUDAActivityMetrics",

    "CUDAMetricsSnapshot",
    "CUDAMetricsDelta",

    "cuda_metrics_available",
    "require_cuda_metrics_runtime",

    "read_global_memory",
    "read_allocator_metrics",

    "read_gpu_utilization",
    "read_memory_utilization",
    "read_temperature",
    "read_activity",

    "reset_peak_memory_stats",

    "CUDAMetricsCollector",

    "metrics_delta",
    "optimization_metrics_attributes",

    "collect_cuda_metrics",
]
