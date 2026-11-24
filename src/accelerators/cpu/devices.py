from __future__ import annotations

"""
CPU device inspection and profiling for OSHA.

This module provides richer CPU-specific information
on top of the framework-neutral AcceleratorDevice
contract.

Responsibilities:

    CPU topology inspection
    host architecture inspection
    processor-count inspection
    memory profile construction
    precision/capability inspection
    workload suitability analysis
    optimization-profile construction

This module does not:

    change CPU affinity
    change PyTorch thread counts
    change DataLoader workers
    alter process scheduling
    enable mixed precision
    execute training or inference
"""

import os
import platform

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

from src.accelerators.cpu.backend import (
    CPUBackend,
    CPU_DEVICE_ID,
    CPU_RUNTIME_DEVICE,
    cpu_capabilities,
    cpu_name,
    cpu_supported_precisions,
    logical_processor_count,
    physical_processor_count,
    system_memory,
)

from src.accelerators.optimization import (
    AcceleratorOptimizationProfile,
    AcceleratorWorkload,
)

from src.accelerators.schema import (
    AcceleratorDevice,
    AcceleratorKind,
    AcceleratorMemory,
    AcceleratorPrecision,
    AcceleratorState,
)

from src.core.serialization import (
    to_json_safe,
)


# ------------------------------------------------------------------
# Optional dependencies
# ------------------------------------------------------------------

try:
    import psutil

except ImportError:
    psutil = None


try:
    import torch

except ImportError:
    torch = None


# ------------------------------------------------------------------
# CPU device schema
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CPUDeviceProfile:
    """
    Rich CPU-specific device profile.

    AcceleratorDevice remains the canonical common
    hardware identity used by AcceleratorManager.

    This profile adds CPU-oriented information that
    does not belong in the common accelerator schema.
    """

    device: AcceleratorDevice

    logical_processors: Optional[int] = None

    physical_processors: Optional[int] = None

    affinity_processors: Optional[int] = None

    machine: Optional[str] = None

    architecture: Optional[str] = None

    processor: Optional[str] = None

    system: Optional[str] = None

    memory: AcceleratorMemory = field(default_factory=AcceleratorMemory)

    torch_available: bool = False

    torch_num_threads: Optional[int] = None

    torch_num_interop_threads: Optional[int] = None

    supported_precisions: Tuple[
        str,
        ...,
    ] = ()

    capabilities: Tuple[
        str,
        ...,
    ] = ()

    attributes: Mapping[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        self.device.validate()

        if self.device.kind != AcceleratorKind.CPU.value:
            raise ValueError("CPUDeviceProfile requires a CPU AcceleratorDevice.")

        if self.device.device_id != CPU_DEVICE_ID:
            raise ValueError("CPUDeviceProfile requires the canonical CPU device.")

        for name, value in (
            (
                "logical_processors",
                self.logical_processors,
            ),
            (
                "physical_processors",
                self.physical_processors,
            ),
            (
                "affinity_processors",
                self.affinity_processors,
            ),
            (
                "torch_num_threads",
                self.torch_num_threads,
            ),
            (
                "torch_num_interop_threads",
                self.torch_num_interop_threads,
            ),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1")

        self.memory.validate()

        valid_precisions = {item.value for item in AcceleratorPrecision}

        unknown = [
            precision
            for precision in self.supported_precisions
            if precision not in valid_precisions
        ]

        if unknown:
            raise ValueError(
                "CPU profile contains "
                "unsupported precision "
                "identifiers: " + ", ".join(unknown)
            )

    @property
    def effective_processors(
        self,
    ) -> Optional[int]:
        """
        Best estimate of CPU concurrency visible to
        the current process.

        CPU affinity is preferred because a
        container/process may see fewer CPUs than
        the host reports.
        """

        if self.affinity_processors:
            return self.affinity_processors

        if self.logical_processors:
            return self.logical_processors

        return None

    @property
    def total_memory_bytes(
        self,
    ) -> Optional[int]:
        return self.memory.total_bytes

    @property
    def free_memory_bytes(
        self,
    ) -> Optional[int]:
        return self.memory.free_bytes

    def supports_precision(
        self,
        precision: str,
    ) -> bool:
        return str(precision).strip().lower() in set(self.supported_precisions)

    def supports_capability(
        self,
        capability: str,
    ) -> bool:
        return str(capability).strip() in set(self.capabilities)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "device": (self.device.to_dict()),
            "logical_processors": (self.logical_processors),
            "physical_processors": (self.physical_processors),
            "affinity_processors": (self.affinity_processors),
            "effective_processors": (self.effective_processors),
            "machine": self.machine,
            "architecture": (self.architecture),
            "processor": (self.processor),
            "system": self.system,
            "memory": (self.memory.to_dict()),
            "torch_available": (self.torch_available),
            "torch_num_threads": (self.torch_num_threads),
            "torch_num_interop_threads": (self.torch_num_interop_threads),
            "supported_precisions": list(self.supported_precisions),
            "capabilities": list(self.capabilities),
            "attributes": (to_json_safe(dict(self.attributes))),
        }


# ------------------------------------------------------------------
# CPU affinity
# ------------------------------------------------------------------


def affinity_processor_count() -> Optional[int]:
    """
    Return the number of CPUs available to the
    current process when the operating system
    exposes CPU affinity.

    This is inspection only. Affinity is never
    modified here.
    """

    affinity = getattr(
        os,
        "sched_getaffinity",
        None,
    )

    if callable(affinity):
        try:
            processors = affinity(0)

            count = len(processors)

            if count > 0:
                return count

        except (
            OSError,
            ValueError,
        ):
            pass

    if psutil is not None:
        try:
            process = psutil.Process()

            cpu_affinity = getattr(
                process,
                "cpu_affinity",
                None,
            )

            if callable(cpu_affinity):
                processors = cpu_affinity()

                if processors:
                    return len(processors)

        except Exception:
            pass

    return None


# ------------------------------------------------------------------
# Runtime thread inspection
# ------------------------------------------------------------------


def torch_thread_counts() -> Tuple[
    Optional[int],
    Optional[int],
]:
    """
    Inspect current PyTorch CPU thread settings.

    No thread configuration is changed.
    """

    if torch is None:
        return (
            None,
            None,
        )

    intra_op = None
    inter_op = None

    try:
        value = int(torch.get_num_threads())

        if value > 0:
            intra_op = value

    except Exception:
        pass

    try:
        value = int(torch.get_num_interop_threads())

        if value > 0:
            inter_op = value

    except Exception:
        pass

    return (
        intra_op,
        inter_op,
    )


# ------------------------------------------------------------------
# Environment inspection
# ------------------------------------------------------------------


def cpu_environment() -> Dict[str, Any]:
    """
    Inspect CPU/runtime-related environment
    configuration.

    Values are reported only when they already
    exist. Nothing is modified.
    """

    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )

    return {
        name: os.environ.get(name) for name in names if os.environ.get(name) is not None
    }


# ------------------------------------------------------------------
# CPU architecture information
# ------------------------------------------------------------------


def cpu_architecture_info() -> Dict[str, Any]:
    return {
        "machine": (platform.machine() or None),
        "architecture": (platform.architecture()[0] or None),
        "processor": (platform.processor() or None),
        "system": (platform.system() or None),
        "release": (platform.release() or None),
    }


# ------------------------------------------------------------------
# Resolve canonical CPU device
# ------------------------------------------------------------------


def resolve_cpu_device(
    backend: Optional[CPUBackend] = None,
) -> AcceleratorDevice:
    """
    Discover and return the canonical CPU device.
    """

    runtime_backend = backend or CPUBackend()

    result = runtime_backend.discover()

    for device in result.devices:
        if (
            device.kind == AcceleratorKind.CPU.value
            and device.device_id == CPU_DEVICE_ID
        ):
            return device

    # This should only occur when CPU discovery is
    # unavailable. Preserve a structured CPU device
    # for inspection callers rather than inventing
    # available hardware.
    return AcceleratorDevice(
        device_id=(CPU_DEVICE_ID),
        kind=(AcceleratorKind.CPU.value),
        runtime_device=(CPU_RUNTIME_DEVICE),
        index=None,
        name=cpu_name(),
        state=(AcceleratorState.UNAVAILABLE.value),
        memory=(AcceleratorMemory()),
        capabilities=(),
        supported_precisions=(AcceleratorPrecision.FLOAT32.value,),
        architecture=(platform.machine() or None),
        logical_processor_count=(logical_processor_count()),
        attributes={
            "resolved_without_available_backend": (True),
        },
    )


# ------------------------------------------------------------------
# CPU device inspector
# ------------------------------------------------------------------


class CPUDeviceInspector:
    """
    Build a rich CPUDeviceProfile.

    The inspector is intentionally read-only.
    """

    def __init__(
        self,
        *,
        backend: Optional[CPUBackend] = None,
    ) -> None:
        self.backend = backend or CPUBackend()

    # ---------------------------------------------------------
    # Inspect
    # ---------------------------------------------------------

    def inspect(
        self,
        device: Optional[AcceleratorDevice] = None,
        *,
        refresh_memory: bool = True,
    ) -> CPUDeviceProfile:
        resolved = device or resolve_cpu_device(self.backend)

        resolved.validate()

        if resolved.kind != AcceleratorKind.CPU.value:
            raise ValueError("CPUDeviceInspector cannot inspect a non-CPU device.")

        memory = resolved.memory

        memory_error_type = None

        if refresh_memory and resolved.available:
            try:
                memory = self.backend.memory(resolved)

            except Exception as exc:
                memory_error_type = type(exc).__name__

        (
            torch_threads,
            torch_interop_threads,
        ) = torch_thread_counts()

        architecture = cpu_architecture_info()

        profile = CPUDeviceProfile(
            device=resolved,
            logical_processors=(logical_processor_count()),
            physical_processors=(physical_processor_count()),
            affinity_processors=(affinity_processor_count()),
            machine=(architecture["machine"]),
            architecture=(architecture["architecture"]),
            processor=(architecture["processor"]),
            system=(architecture["system"]),
            memory=memory,
            torch_available=(torch is not None),
            torch_num_threads=(torch_threads),
            torch_num_interop_threads=(torch_interop_threads),
            supported_precisions=(
                resolved.supported_precisions or cpu_supported_precisions()
            ),
            capabilities=(resolved.capabilities or cpu_capabilities()),
            attributes={
                "cpu_name": (resolved.name or cpu_name()),
                "environment": (cpu_environment()),
                "memory_error_type": (memory_error_type),
                "psutil_available": (psutil is not None),
                "platform_release": (architecture["release"]),
            },
        )

        profile.validate()

        return profile

    # ---------------------------------------------------------
    # Optimization profile
    # ---------------------------------------------------------

    def optimization_profile(
        self,
        workload: AcceleratorWorkload,
        *,
        device: Optional[AcceleratorDevice] = None,
        refresh_memory: bool = True,
    ) -> AcceleratorOptimizationProfile:
        """
        Convert CPU inspection into the common
        optimization-profile contract.
        """

        workload.validate()

        profile = self.inspect(
            device,
            refresh_memory=refresh_memory,
        )

        result = AcceleratorOptimizationProfile(
            device=profile.device,
            workload=workload,
            free_memory_bytes=(profile.free_memory_bytes),
            total_memory_bytes=(profile.total_memory_bytes),
            utilization_percent=None,
            temperature_celsius=None,
            capabilities=(profile.capabilities),
            attributes={
                "logical_processors": (profile.logical_processors),
                "physical_processors": (profile.physical_processors),
                "affinity_processors": (profile.affinity_processors),
                "effective_processors": (profile.effective_processors),
                "torch_num_threads": (profile.torch_num_threads),
                "torch_num_interop_threads": (profile.torch_num_interop_threads),
                "supported_precisions": list(profile.supported_precisions),
                "environment": (
                    profile.attributes.get(
                        "environment",
                        {},
                    )
                ),
            },
        )

        result.validate()

        return result


# ------------------------------------------------------------------
# Workload suitability
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CPUWorkloadSuitability:
    """
    Read-only CPU/workload suitability result.

    This is descriptive information for the CPU
    optimizer. It is not an execution decision.
    """

    suitable: bool

    reasons: Tuple[
        str,
        ...,
    ] = ()

    warnings: Tuple[
        str,
        ...,
    ] = ()

    attributes: Mapping[
        str,
        Any,
    ] = field(default_factory=dict)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "suitable": (self.suitable),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "attributes": (to_json_safe(dict(self.attributes))),
        }


def evaluate_cpu_workload(
    profile: CPUDeviceProfile,
    workload: AcceleratorWorkload,
) -> CPUWorkloadSuitability:
    """
    Evaluate whether the CPU device can reasonably
    host the described workload based only on
    explicitly known requirements.

    No synthetic performance thresholds are used.
    """

    profile.validate()
    workload.validate()

    reasons = []
    warnings = []

    if not profile.device.available:
        reasons.append("cpu_device_unavailable")

    if not profile.supports_precision(workload.precision):
        reasons.append("workload_precision_unsupported")

    if workload.mixed_precision_enabled:
        # The generic CPU backend does not advertise
        # MIXED_PRECISION because support depends on
        # runtime/operator coverage.
        warnings.append("mixed_precision_requires_runtime_validation")

    if workload.estimated_peak_memory_bytes is not None:
        free = profile.free_memory_bytes

        if free is None:
            warnings.append("free_memory_unknown")

        elif workload.estimated_peak_memory_bytes > free:
            reasons.append("estimated_peak_memory_exceeds_available_memory")

    if profile.effective_processors is None:
        warnings.append("effective_processor_count_unknown")

    return CPUWorkloadSuitability(
        suitable=(not reasons),
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        attributes={
            "effective_processors": (profile.effective_processors),
            "free_memory_bytes": (profile.free_memory_bytes),
            "estimated_peak_memory_bytes": (workload.estimated_peak_memory_bytes),
        },
    )


# ------------------------------------------------------------------
# Convenience inspection
# ------------------------------------------------------------------


def inspect_cpu_device(
    *,
    backend: Optional[CPUBackend] = None,
    refresh_memory: bool = True,
) -> CPUDeviceProfile:
    return CPUDeviceInspector(backend=backend).inspect(refresh_memory=(refresh_memory))


# ------------------------------------------------------------------
# Compatibility helpers
# ------------------------------------------------------------------


def _compat_torch():
    try:
        import torch as runtime
    except ImportError:
        return None
    return runtime


def is_cpu_available() -> bool:
    return True


def get_device_count() -> int:
    return logical_processor_count() or 1


def get_device(index: int = 0) -> str:
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("Device index must be an integer.")
    if index != 0:
        raise ValueError("CPU exposes only device index 0.")
    runtime = _compat_torch()
    if runtime is None:
        return "cpu"
    return str(runtime.device("cpu"))


def get_device_info(index: int = 0) -> Dict[str, Any]:
    get_device(index)
    runtime = _compat_torch()
    return {
        "backend": "cpu",
        "name": "CPU",
        "pytorch_threads": (
            runtime.get_num_threads()
            if runtime is not None
            and callable(getattr(runtime, "get_num_threads", None))
            else None
        ),
        "pytorch_interop_threads": (
            runtime.get_num_interop_threads()
            if runtime is not None
            and callable(getattr(runtime, "get_num_interop_threads", None))
            else None
        ),
        "logical_processors": logical_processor_count(),
    }


def list_devices() -> list[Dict[str, Any]]:
    return [get_device_info()]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "CPUDeviceProfile",
    "CPUWorkloadSuitability",
    "affinity_processor_count",
    "torch_thread_counts",
    "cpu_environment",
    "cpu_architecture_info",
    "resolve_cpu_device",
    "CPUDeviceInspector",
    "evaluate_cpu_workload",
    "inspect_cpu_device",
    "is_cpu_available",
    "get_device",
    "get_device_count",
    "get_device_info",
    "list_devices",
]
