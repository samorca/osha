from __future__ import annotations

"""
CPU runtime metrics for OSHA.

This module provides read-only CPU and process
telemetry for accelerator optimization and
observability.

Responsibilities:

    system CPU utilization
    per-CPU utilization
    CPU frequency
    load averages
    system memory utilization
    swap utilization
    current-process CPU utilization
    current-process memory utilization
    process thread count
    CPU time counters
    structured metric snapshots

This module does not:

    change CPU affinity
    change process priority
    change thread counts
    change DataLoader workers
    allocate model memory
    optimize workloads automatically
"""

import math
import os
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
    Tuple,
)

from src.accelerators.cpu.devices import (
    CPUDeviceInspector,
    CPUDeviceProfile,
    affinity_processor_count,
    logical_processor_count,
)

from src.core.serialization import (
    to_json_safe,
)

from src.core.time import (
    utc_now,
)


# ------------------------------------------------------------------
# Optional dependency
# ------------------------------------------------------------------

try:
    import psutil

except ImportError:
    psutil = None


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

CPU_METRICS_SCHEMA_VERSION = "1"


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------


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


def _non_negative(
    value: Optional[float],
    *,
    name: str,
) -> None:
    if value is None:
        return

    numeric = float(value)

    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")


def _non_negative_integer(
    value: Optional[int],
    *,
    name: str,
) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be >= 0")


# ------------------------------------------------------------------
# CPU frequency metrics
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CPUFrequencyMetrics:
    current_mhz: Optional[float] = None

    minimum_mhz: Optional[float] = None

    maximum_mhz: Optional[float] = None

    def validate(
        self,
    ) -> None:
        _non_negative(
            self.current_mhz,
            name="current_mhz",
        )

        _non_negative(
            self.minimum_mhz,
            name="minimum_mhz",
        )

        _non_negative(
            self.maximum_mhz,
            name="maximum_mhz",
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "current_mhz": (self.current_mhz),
            "minimum_mhz": (self.minimum_mhz),
            "maximum_mhz": (self.maximum_mhz),
        }


# ------------------------------------------------------------------
# CPU time metrics
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CPUTimesMetrics:
    user_seconds: Optional[float] = None

    system_seconds: Optional[float] = None

    idle_seconds: Optional[float] = None

    nice_seconds: Optional[float] = None

    io_wait_seconds: Optional[float] = None

    irq_seconds: Optional[float] = None

    soft_irq_seconds: Optional[float] = None

    steal_seconds: Optional[float] = None

    def validate(
        self,
    ) -> None:
        for name, value in (
            (
                "user_seconds",
                self.user_seconds,
            ),
            (
                "system_seconds",
                self.system_seconds,
            ),
            (
                "idle_seconds",
                self.idle_seconds,
            ),
            (
                "nice_seconds",
                self.nice_seconds,
            ),
            (
                "io_wait_seconds",
                self.io_wait_seconds,
            ),
            (
                "irq_seconds",
                self.irq_seconds,
            ),
            (
                "soft_irq_seconds",
                self.soft_irq_seconds,
            ),
            (
                "steal_seconds",
                self.steal_seconds,
            ),
        ):
            _non_negative(
                value,
                name=name,
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "user_seconds": (self.user_seconds),
            "system_seconds": (self.system_seconds),
            "idle_seconds": (self.idle_seconds),
            "nice_seconds": (self.nice_seconds),
            "io_wait_seconds": (self.io_wait_seconds),
            "irq_seconds": (self.irq_seconds),
            "soft_irq_seconds": (self.soft_irq_seconds),
            "steal_seconds": (self.steal_seconds),
        }


# ------------------------------------------------------------------
# System memory metrics
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CPUMemoryMetrics:
    total_bytes: Optional[int] = None

    available_bytes: Optional[int] = None

    used_bytes: Optional[int] = None

    free_bytes: Optional[int] = None

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
                "available_bytes",
                self.available_bytes,
            ),
            (
                "used_bytes",
                self.used_bytes,
            ),
            (
                "free_bytes",
                self.free_bytes,
            ),
        ):
            _non_negative_integer(
                value,
                name=name,
            )

        _percentage(
            self.utilization_percent,
            name=("memory utilization"),
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "total_bytes": (self.total_bytes),
            "available_bytes": (self.available_bytes),
            "used_bytes": (self.used_bytes),
            "free_bytes": (self.free_bytes),
            "utilization_percent": (self.utilization_percent),
        }


# ------------------------------------------------------------------
# Swap metrics
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CPUSwapMetrics:
    total_bytes: Optional[int] = None

    used_bytes: Optional[int] = None

    free_bytes: Optional[int] = None

    utilization_percent: Optional[float] = None

    swap_in_bytes: Optional[int] = None

    swap_out_bytes: Optional[int] = None

    def validate(
        self,
    ) -> None:
        for name, value in (
            (
                "total_bytes",
                self.total_bytes,
            ),
            (
                "used_bytes",
                self.used_bytes,
            ),
            (
                "free_bytes",
                self.free_bytes,
            ),
            (
                "swap_in_bytes",
                self.swap_in_bytes,
            ),
            (
                "swap_out_bytes",
                self.swap_out_bytes,
            ),
        ):
            _non_negative_integer(
                value,
                name=name,
            )

        _percentage(
            self.utilization_percent,
            name=("swap utilization"),
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "total_bytes": (self.total_bytes),
            "used_bytes": (self.used_bytes),
            "free_bytes": (self.free_bytes),
            "utilization_percent": (self.utilization_percent),
            "swap_in_bytes": (self.swap_in_bytes),
            "swap_out_bytes": (self.swap_out_bytes),
        }


# ------------------------------------------------------------------
# Process metrics
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CPUProcessMetrics:
    pid: int

    cpu_percent: Optional[float] = None

    resident_memory_bytes: Optional[int] = None

    virtual_memory_bytes: Optional[int] = None

    memory_percent: Optional[float] = None

    thread_count: Optional[int] = None

    user_cpu_seconds: Optional[float] = None

    system_cpu_seconds: Optional[float] = None

    def validate(
        self,
    ) -> None:
        if self.pid < 1:
            raise ValueError("pid must be >= 1")

        _non_negative(
            self.cpu_percent,
            name=("process cpu_percent"),
        )

        _non_negative_integer(
            self.resident_memory_bytes,
            name=("resident_memory_bytes"),
        )

        _non_negative_integer(
            self.virtual_memory_bytes,
            name=("virtual_memory_bytes"),
        )

        _non_negative(
            self.memory_percent,
            name=("process memory_percent"),
        )

        _non_negative_integer(
            self.thread_count,
            name="thread_count",
        )

        _non_negative(
            self.user_cpu_seconds,
            name="user_cpu_seconds",
        )

        _non_negative(
            self.system_cpu_seconds,
            name="system_cpu_seconds",
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "pid": self.pid,
            "cpu_percent": (self.cpu_percent),
            "resident_memory_bytes": (self.resident_memory_bytes),
            "virtual_memory_bytes": (self.virtual_memory_bytes),
            "memory_percent": (self.memory_percent),
            "thread_count": (self.thread_count),
            "user_cpu_seconds": (self.user_cpu_seconds),
            "system_cpu_seconds": (self.system_cpu_seconds),
        }


# ------------------------------------------------------------------
# CPU metrics snapshot
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CPUMetricsSnapshot:
    """
    One read-only CPU telemetry snapshot.
    """

    device_id: str

    cpu_percent: Optional[float] = None

    per_cpu_percent: Tuple[
        float,
        ...,
    ] = ()

    logical_processors: Optional[int] = None

    affinity_processors: Optional[int] = None

    load_average_1m: Optional[float] = None

    load_average_5m: Optional[float] = None

    load_average_15m: Optional[float] = None

    frequency: CPUFrequencyMetrics = field(default_factory=(CPUFrequencyMetrics))

    cpu_times: CPUTimesMetrics = field(default_factory=(CPUTimesMetrics))

    memory: CPUMemoryMetrics = field(default_factory=(CPUMemoryMetrics))

    swap: CPUSwapMetrics = field(default_factory=(CPUSwapMetrics))

    process: Optional[CPUProcessMetrics] = None

    sampled_at: str = field(default_factory=utc_now)

    monotonic_time: float = field(default_factory=(time.monotonic))

    attributes: Mapping[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        if not str(self.device_id).strip():
            raise ValueError("device_id cannot be empty")

        _percentage(
            self.cpu_percent,
            name="cpu_percent",
        )

        for index, value in enumerate(self.per_cpu_percent):
            _percentage(
                value,
                name=("per_cpu_percent[" + str(index) + "]"),
            )

        _non_negative_integer(
            self.logical_processors,
            name=("logical_processors"),
        )

        _non_negative_integer(
            self.affinity_processors,
            name=("affinity_processors"),
        )

        for name, value in (
            (
                "load_average_1m",
                self.load_average_1m,
            ),
            (
                "load_average_5m",
                self.load_average_5m,
            ),
            (
                "load_average_15m",
                self.load_average_15m,
            ),
        ):
            _non_negative(
                value,
                name=name,
            )

        self.frequency.validate()
        self.cpu_times.validate()
        self.memory.validate()
        self.swap.validate()

        if self.process is not None:
            self.process.validate()

        _non_negative(
            self.monotonic_time,
            name="monotonic_time",
        )

    @property
    def effective_cpu_count(
        self,
    ) -> Optional[int]:
        if self.affinity_processors:
            return self.affinity_processors

        return self.logical_processors

    @property
    def normalized_load_1m(
        self,
    ) -> Optional[float]:
        """
        Load average normalized by the number of
        CPUs visible to the process.

        This is descriptive telemetry, not a
        performance threshold.
        """

        count = self.effective_cpu_count

        if count is None or count < 1 or self.load_average_1m is None:
            return None

        return self.load_average_1m / count

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "schema_version": (CPU_METRICS_SCHEMA_VERSION),
            "device_id": (self.device_id),
            "cpu_percent": (self.cpu_percent),
            "per_cpu_percent": list(self.per_cpu_percent),
            "logical_processors": (self.logical_processors),
            "affinity_processors": (self.affinity_processors),
            "effective_cpu_count": (self.effective_cpu_count),
            "load_average_1m": (self.load_average_1m),
            "load_average_5m": (self.load_average_5m),
            "load_average_15m": (self.load_average_15m),
            "normalized_load_1m": (self.normalized_load_1m),
            "frequency": (self.frequency.to_dict()),
            "cpu_times": (self.cpu_times.to_dict()),
            "memory": (self.memory.to_dict()),
            "swap": (self.swap.to_dict()),
            "process": (self.process.to_dict() if self.process is not None else None),
            "sampled_at": (self.sampled_at),
            "monotonic_time": (self.monotonic_time),
            "attributes": (to_json_safe(dict(self.attributes))),
        }


# ------------------------------------------------------------------
# CPU delta metrics
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CPUMetricsDelta:
    """
    Difference between two compatible CPU
    snapshots.

    Memory deltas are signed because consumption
    may increase or decrease.
    """

    elapsed_seconds: float

    cpu_percent_change: Optional[float] = None

    available_memory_change_bytes: Optional[int] = None

    process_resident_memory_change_bytes: Optional[int] = None

    process_user_cpu_seconds: Optional[float] = None

    process_system_cpu_seconds: Optional[float] = None

    def validate(
        self,
    ) -> None:
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and >= 0")

        for name, value in (
            (
                "cpu_percent_change",
                self.cpu_percent_change,
            ),
            (
                "process_user_cpu_seconds",
                self.process_user_cpu_seconds,
            ),
            (
                "process_system_cpu_seconds",
                self.process_system_cpu_seconds,
            ),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "elapsed_seconds": (self.elapsed_seconds),
            "cpu_percent_change": (self.cpu_percent_change),
            "available_memory_change_bytes": (self.available_memory_change_bytes),
            "process_resident_memory_change_bytes": (
                self.process_resident_memory_change_bytes
            ),
            "process_user_cpu_seconds": (self.process_user_cpu_seconds),
            "process_system_cpu_seconds": (self.process_system_cpu_seconds),
        }


# ------------------------------------------------------------------
# psutil guard
# ------------------------------------------------------------------


def psutil_available() -> bool:
    return psutil is not None


# ------------------------------------------------------------------
# CPU utilization
# ------------------------------------------------------------------


def read_cpu_percent(
    *,
    interval: Optional[float] = None,
) -> Optional[float]:
    if psutil is None:
        return None

    try:
        return float(
            psutil.cpu_percent(
                interval=interval,
                percpu=False,
            )
        )

    except Exception:
        return None


def read_per_cpu_percent(
    *,
    interval: Optional[float] = None,
) -> Tuple[float, ...]:
    if psutil is None:
        return ()

    try:
        values = psutil.cpu_percent(
            interval=interval,
            percpu=True,
        )

        return tuple(float(value) for value in values)

    except Exception:
        return ()


# ------------------------------------------------------------------
# Frequency
# ------------------------------------------------------------------


def read_cpu_frequency() -> CPUFrequencyMetrics:
    if psutil is None:
        return CPUFrequencyMetrics()

    try:
        value = psutil.cpu_freq()

        if value is None:
            return CPUFrequencyMetrics()

        return CPUFrequencyMetrics(
            current_mhz=(float(value.current) if value.current is not None else None),
            minimum_mhz=(float(value.min) if value.min is not None else None),
            maximum_mhz=(float(value.max) if value.max is not None else None),
        )

    except Exception:
        return CPUFrequencyMetrics()


# ------------------------------------------------------------------
# CPU times
# ------------------------------------------------------------------


def read_cpu_times() -> CPUTimesMetrics:
    if psutil is None:
        return CPUTimesMetrics()

    try:
        value = psutil.cpu_times()

        return CPUTimesMetrics(
            user_seconds=(
                getattr(
                    value,
                    "user",
                    None,
                )
            ),
            system_seconds=(
                getattr(
                    value,
                    "system",
                    None,
                )
            ),
            idle_seconds=(
                getattr(
                    value,
                    "idle",
                    None,
                )
            ),
            nice_seconds=(
                getattr(
                    value,
                    "nice",
                    None,
                )
            ),
            io_wait_seconds=(
                getattr(
                    value,
                    "iowait",
                    None,
                )
            ),
            irq_seconds=(
                getattr(
                    value,
                    "irq",
                    getattr(
                        value,
                        "interrupt",
                        None,
                    ),
                )
            ),
            soft_irq_seconds=(
                getattr(
                    value,
                    "softirq",
                    None,
                )
            ),
            steal_seconds=(
                getattr(
                    value,
                    "steal",
                    None,
                )
            ),
        )

    except Exception:
        return CPUTimesMetrics()


# ------------------------------------------------------------------
# Memory
# ------------------------------------------------------------------


def read_memory() -> CPUMemoryMetrics:
    if psutil is None:
        return CPUMemoryMetrics()

    try:
        value = psutil.virtual_memory()

        return CPUMemoryMetrics(
            total_bytes=int(value.total),
            available_bytes=int(value.available),
            used_bytes=int(value.used),
            free_bytes=int(value.free),
            utilization_percent=float(value.percent),
        )

    except Exception:
        return CPUMemoryMetrics()


# ------------------------------------------------------------------
# Swap
# ------------------------------------------------------------------


def read_swap() -> CPUSwapMetrics:
    if psutil is None:
        return CPUSwapMetrics()

    try:
        value = psutil.swap_memory()

        return CPUSwapMetrics(
            total_bytes=int(value.total),
            used_bytes=int(value.used),
            free_bytes=int(value.free),
            utilization_percent=float(value.percent),
            swap_in_bytes=(
                int(
                    getattr(
                        value,
                        "sin",
                        0,
                    )
                )
            ),
            swap_out_bytes=(
                int(
                    getattr(
                        value,
                        "sout",
                        0,
                    )
                )
            ),
        )

    except Exception:
        return CPUSwapMetrics()


# ------------------------------------------------------------------
# Load average
# ------------------------------------------------------------------


def read_load_average() -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
]:
    getter = getattr(
        os,
        "getloadavg",
        None,
    )

    if not callable(getter):
        return (
            None,
            None,
            None,
        )

    try:
        one, five, fifteen = getter()

        return (
            float(one),
            float(five),
            float(fifteen),
        )

    except (
        OSError,
        ValueError,
    ):
        return (
            None,
            None,
            None,
        )


# ------------------------------------------------------------------
# Process metrics
# ------------------------------------------------------------------


def read_process_metrics(
    *,
    pid: Optional[int] = None,
) -> Optional[CPUProcessMetrics]:
    if psutil is None:
        return None

    process_id = os.getpid() if pid is None else int(pid)

    if process_id < 1:
        raise ValueError("pid must be >= 1")

    try:
        process = psutil.Process(process_id)

        memory_info = process.memory_info()

        cpu_times = process.cpu_times()

        return CPUProcessMetrics(
            pid=process_id,
            cpu_percent=float(process.cpu_percent(interval=None)),
            resident_memory_bytes=int(memory_info.rss),
            virtual_memory_bytes=int(memory_info.vms),
            memory_percent=float(process.memory_percent()),
            thread_count=int(process.num_threads()),
            user_cpu_seconds=float(cpu_times.user),
            system_cpu_seconds=float(cpu_times.system),
        )

    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
    ):
        return None

    except Exception:
        return None


# ------------------------------------------------------------------
# Metrics collector
# ------------------------------------------------------------------


class CPUMetricsCollector:
    """
    Read-only collector for CPU telemetry.

    interval=None performs a non-blocking CPU
    utilization query.

    A positive interval intentionally blocks for
    that sampling period.
    """

    def __init__(
        self,
        *,
        inspector: Optional[CPUDeviceInspector] = None,
        process_id: Optional[int] = None,
    ) -> None:
        self.inspector = inspector or CPUDeviceInspector()

        self.process_id = process_id

    # ---------------------------------------------------------
    # Collect
    # ---------------------------------------------------------

    def collect(
        self,
        *,
        interval: Optional[float] = None,
        include_process: bool = True,
    ) -> CPUMetricsSnapshot:
        if interval is not None and (
            not math.isfinite(float(interval)) or interval < 0
        ):
            raise ValueError("interval must be finite and >= 0")

        profile = self.inspector.inspect(refresh_memory=False)

        (
            load_1,
            load_5,
            load_15,
        ) = read_load_average()

        # Sampling system and per-CPU percentages
        # independently with the same configured
        # blocking behavior.
        cpu_percent = read_cpu_percent(interval=interval)

        per_cpu_percent = read_per_cpu_percent(interval=None)

        snapshot = CPUMetricsSnapshot(
            device_id=(profile.device.device_id),
            cpu_percent=cpu_percent,
            per_cpu_percent=(per_cpu_percent),
            logical_processors=(logical_processor_count()),
            affinity_processors=(affinity_processor_count()),
            load_average_1m=(load_1),
            load_average_5m=(load_5),
            load_average_15m=(load_15),
            frequency=(read_cpu_frequency()),
            cpu_times=(read_cpu_times()),
            memory=read_memory(),
            swap=read_swap(),
            process=(
                read_process_metrics(pid=(self.process_id)) if include_process else None
            ),
            attributes={
                "psutil_available": (psutil is not None),
                "sampling_interval_seconds": (interval),
                "process_metrics_included": (include_process),
            },
        )

        snapshot.validate()

        return snapshot


# ------------------------------------------------------------------
# Snapshot delta
# ------------------------------------------------------------------


def metrics_delta(
    before: CPUMetricsSnapshot,
    after: CPUMetricsSnapshot,
) -> CPUMetricsDelta:
    before.validate()
    after.validate()

    if before.device_id != after.device_id:
        raise ValueError("CPU metrics snapshots belong to different devices.")

    elapsed = max(
        0.0,
        after.monotonic_time - before.monotonic_time,
    )

    cpu_percent_change = None

    if before.cpu_percent is not None and after.cpu_percent is not None:
        cpu_percent_change = after.cpu_percent - before.cpu_percent

    memory_change = None

    if (
        before.memory.available_bytes is not None
        and after.memory.available_bytes is not None
    ):
        memory_change = after.memory.available_bytes - before.memory.available_bytes

    resident_change = None
    user_cpu_delta = None
    system_cpu_delta = None

    if (
        before.process is not None
        and after.process is not None
        and before.process.pid == after.process.pid
    ):
        if (
            before.process.resident_memory_bytes is not None
            and after.process.resident_memory_bytes is not None
        ):
            resident_change = (
                after.process.resident_memory_bytes
                - before.process.resident_memory_bytes
            )

        if (
            before.process.user_cpu_seconds is not None
            and after.process.user_cpu_seconds is not None
        ):
            user_cpu_delta = max(
                0.0,
                after.process.user_cpu_seconds - before.process.user_cpu_seconds,
            )

        if (
            before.process.system_cpu_seconds is not None
            and after.process.system_cpu_seconds is not None
        ):
            system_cpu_delta = max(
                0.0,
                after.process.system_cpu_seconds - before.process.system_cpu_seconds,
            )

    result = CPUMetricsDelta(
        elapsed_seconds=elapsed,
        cpu_percent_change=(cpu_percent_change),
        available_memory_change_bytes=(memory_change),
        process_resident_memory_change_bytes=(resident_change),
        process_user_cpu_seconds=(user_cpu_delta),
        process_system_cpu_seconds=(system_cpu_delta),
    )

    result.validate()

    return result


# ------------------------------------------------------------------
# Optimization profile with live metrics
# ------------------------------------------------------------------


def optimization_metrics_attributes(
    snapshot: CPUMetricsSnapshot,
) -> Dict[str, Any]:
    """
    Convert runtime metrics into bounded
    optimizer input attributes.

    The CPU optimizer may inspect these values but
    this function makes no recommendations.
    """

    snapshot.validate()

    return {
        "cpu_percent": (snapshot.cpu_percent),
        "per_cpu_percent": list(snapshot.per_cpu_percent),
        "load_average_1m": (snapshot.load_average_1m),
        "normalized_load_1m": (snapshot.normalized_load_1m),
        "logical_processors": (snapshot.logical_processors),
        "affinity_processors": (snapshot.affinity_processors),
        "effective_cpu_count": (snapshot.effective_cpu_count),
        "memory_available_bytes": (snapshot.memory.available_bytes),
        "memory_utilization_percent": (snapshot.memory.utilization_percent),
        "swap_utilization_percent": (snapshot.swap.utilization_percent),
        "frequency_current_mhz": (snapshot.frequency.current_mhz),
        "process": (
            snapshot.process.to_dict() if snapshot.process is not None else None
        ),
    }


# ------------------------------------------------------------------
# Convenience collection
# ------------------------------------------------------------------


def collect_cpu_metrics(
    *,
    interval: Optional[float] = None,
    include_process: bool = True,
    process_id: Optional[int] = None,
) -> CPUMetricsSnapshot:
    return CPUMetricsCollector(process_id=process_id).collect(
        interval=interval,
        include_process=(include_process),
    )


def get_cpu_metrics() -> Dict[str, Any]:
    if psutil is None:
        return {
            "utilization_percent": 0.0,
            "logical_cores": os.cpu_count() or 1,
        }
    return {
        "utilization_percent": float(psutil.cpu_percent(interval=0.1)),
        "logical_cores": int(psutil.cpu_count(logical=True) or 1),
    }


def get_memory_metrics() -> Dict[str, Any]:
    if psutil is None:
        return {
            "total_memory_bytes": 0,
            "available_memory_bytes": 0,
            "used_memory_bytes": 0,
            "memory_utilization_percent": 0.0,
        }
    memory = psutil.virtual_memory()
    total = int(memory.total)
    available = int(memory.available)
    return {
        "total_memory_bytes": total,
        "available_memory_bytes": available,
        "used_memory_bytes": int(memory.used),
        "memory_utilization_percent": float(memory.percent),
    }


def get_swap_metrics() -> Dict[str, Any]:
    if psutil is None:
        return {
            "total_swap_bytes": 0,
            "used_swap_bytes": 0,
            "free_swap_bytes": 0,
            "swap_utilization_percent": 0.0,
        }
    swap = psutil.swap_memory()
    return {
        "total_swap_bytes": int(swap.total),
        "used_swap_bytes": int(swap.used),
        "free_swap_bytes": int(swap.free),
        "swap_utilization_percent": float(swap.percent),
    }


def get_all_metrics() -> Dict[str, Any]:
    return {
        "cpu": get_cpu_metrics(),
        "memory": get_memory_metrics(),
        "swap": get_swap_metrics(),
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "CPU_METRICS_SCHEMA_VERSION",
    "CPUFrequencyMetrics",
    "CPUTimesMetrics",
    "CPUMemoryMetrics",
    "CPUSwapMetrics",
    "CPUProcessMetrics",
    "CPUMetricsSnapshot",
    "CPUMetricsDelta",
    "psutil_available",
    "read_cpu_percent",
    "read_per_cpu_percent",
    "read_cpu_frequency",
    "read_cpu_times",
    "read_memory",
    "read_swap",
    "read_load_average",
    "read_process_metrics",
    "CPUMetricsCollector",
    "metrics_delta",
    "optimization_metrics_attributes",
    "collect_cpu_metrics",
    "get_cpu_metrics",
    "get_memory_metrics",
    "get_swap_metrics",
    "get_all_metrics",
]
