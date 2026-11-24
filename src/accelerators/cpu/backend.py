from __future__ import annotations

"""
CPU accelerator backend for OSHA.

The CPU backend provides:

    CPU availability detection
    logical and physical processor discovery
    optional system-memory reporting
    supported precision reporting
    framework runtime-device construction
    structured health reporting

CPU synchronization and accelerator-cache
clearing are intentional no-ops.

This backend does not configure global thread
counts automatically. Thread policy remains a
workload/runtime concern.
"""

import os
import platform

from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

from src.accelerators.base import (
    CPUAcceleratorBackend,
)

from src.accelerators.config import (
    AcceleratorConfig,
)

from src.accelerators.errors import (
    AcceleratorBackendInitializationError,
    AcceleratorMemoryQueryError,
    AcceleratorRuntimeDeviceError,
    safe_accelerator_error_details,
)

from src.accelerators.schema import (
    AcceleratorCapability,
    AcceleratorDevice,
    AcceleratorHealth,
    AcceleratorHealthStatus,
    AcceleratorKind,
    AcceleratorMemory,
    AcceleratorPrecision,
    AcceleratorState,
)


# ---------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------

try:
    import psutil

except ImportError:
    psutil = None


try:
    import torch

except ImportError:
    torch = None


# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

CPU_DEVICE_ID = "cpu"

CPU_RUNTIME_DEVICE = "cpu"


# ---------------------------------------------------------------
# Safe host information
# ---------------------------------------------------------------


def logical_processor_count() -> Optional[int]:
    """
    Return the number of logical processors
    visible to the current process.

    None is returned when the operating system
    cannot provide a value.
    """

    value = os.cpu_count()

    if value is None:
        return None

    count = int(value)

    return count if count > 0 else None


def physical_processor_count() -> Optional[int]:
    """
    Return the physical-core count when psutil is
    available.

    Unknown values remain None.
    """

    if psutil is None:
        return None

    try:
        value = psutil.cpu_count(logical=False)

    except Exception:
        return None

    if value is None:
        return None

    count = int(value)

    return count if count > 0 else None


def cpu_name() -> str:
    """
    Build a bounded operator-friendly CPU name.

    No shell commands or platform-specific
    subprocesses are executed.
    """

    processor = str(platform.processor() or "").strip()

    machine = str(platform.machine() or "").strip()

    if processor:
        return processor

    if machine:
        return "System CPU (" + machine + ")"

    return "System CPU"


def system_memory() -> AcceleratorMemory:
    """
    Query host virtual-memory information.

    psutil's available value is used as the
    accelerator schema's free/available estimate.

    When psutil is unavailable, memory values
    remain unknown instead of being invented.
    """

    if psutil is None:
        return AcceleratorMemory()

    try:
        memory = psutil.virtual_memory()

        total = int(memory.total)

        available = int(memory.available)

        used = max(
            0,
            total - available,
        )

        result = AcceleratorMemory(
            total_bytes=total,
            free_bytes=available,
            used_bytes=used,
            reserved_bytes=None,
            allocated_bytes=None,
        )

        result.validate()

        return result

    except Exception as exc:
        raise AcceleratorMemoryQueryError(
            "CPU system-memory query failed.",
            details=(
                safe_accelerator_error_details(
                    exc,
                    operation=("cpu_system_memory"),
                    backend=(AcceleratorKind.CPU.value),
                    device_id=(CPU_DEVICE_ID),
                    runtime_device=(CPU_RUNTIME_DEVICE),
                )
            ),
        ) from exc


# ---------------------------------------------------------------
# Precision discovery
# ---------------------------------------------------------------


def cpu_supported_precisions() -> Tuple[str, ...]:
    """
    Return conservative CPU precision support.

    FLOAT32 and FLOAT64 are always advertised for
    the supported Python/PyTorch numeric path.

    BFLOAT16 is advertised only when the installed
    PyTorch runtime exposes the dtype. This means
    the runtime recognizes the dtype, not that all
    CPU operations are hardware accelerated.
    """

    values = [
        AcceleratorPrecision.FLOAT64.value,
        AcceleratorPrecision.FLOAT32.value,
    ]

    if torch is not None and hasattr(
        torch,
        "bfloat16",
    ):
        values.append(AcceleratorPrecision.BFLOAT16.value)

    return tuple(values)


# ---------------------------------------------------------------
# Capability discovery
# ---------------------------------------------------------------


def cpu_capabilities() -> Tuple[str, ...]:
    """
    Capabilities consistently exposed by the CPU
    backend.
    """

    values = [
        AcceleratorCapability.COMPUTE.value,
        AcceleratorCapability.TRAINING.value,
        AcceleratorCapability.INFERENCE.value,
    ]

    if psutil is not None:
        values.append(AcceleratorCapability.MEMORY_STATS.value)

    return tuple(values)


# ---------------------------------------------------------------
# CPU backend
# ---------------------------------------------------------------


class CPUBackend(CPUAcceleratorBackend):
    """
    Concrete CPU implementation of the common
    AcceleratorBackend contract.

    One logical CPU execution device is exposed:

        device_id       = cpu
        runtime_device  = cpu
        kind            = cpu
    """

    def __init__(
        self,
        *,
        config: Optional[AcceleratorConfig] = None,
    ) -> None:
        super().__init__(config=config)

    # ---------------------------------------------------------
    # Availability
    # ---------------------------------------------------------

    def available(
        self,
    ) -> bool:
        """
        CPU execution is considered available when
        the process can observe at least one
        logical processor.

        If the operating system does not report a
        processor count, CPU availability remains
        true because the running Python process is
        already executing on a CPU-compatible host.
        """

        count = logical_processor_count()

        return bool(count is None or count > 0)

    # ---------------------------------------------------------
    # Device attributes
    # ---------------------------------------------------------

    def _device_attributes(
        self,
    ) -> Dict[str, Any]:
        return {
            "platform": (platform.system()),
            "platform_release": (platform.release()),
            "machine": (platform.machine()),
            "python_implementation": (platform.python_implementation()),
            "physical_processor_count": (physical_processor_count()),
            "psutil_available": (psutil is not None),
            "torch_available": (torch is not None),
        }

    # ---------------------------------------------------------
    # Discovery
    # ---------------------------------------------------------

    def _discover(
        self,
    ) -> Tuple[
        AcceleratorDevice,
        ...,
    ]:
        if not self.available():
            if not self.config.discovery.include_unavailable_devices:
                return ()

            return (
                AcceleratorDevice(
                    device_id=(CPU_DEVICE_ID),
                    kind=(AcceleratorKind.CPU.value),
                    runtime_device=(CPU_RUNTIME_DEVICE),
                    index=None,
                    name=cpu_name(),
                    state=(AcceleratorState.UNAVAILABLE.value),
                    memory=(AcceleratorMemory()),
                    capabilities=(),
                    supported_precisions=(AcceleratorPrecision.FLOAT32.value,),
                    vendor=None,
                    architecture=(platform.machine() or None),
                    logical_processor_count=(logical_processor_count()),
                    attributes=(self._device_attributes()),
                ),
            )

        memory = AcceleratorMemory()

        if self.config.discovery.refresh_memory_on_detection:
            try:
                memory = system_memory()

            except AcceleratorMemoryQueryError:
                # CPU execution is still usable
                # when optional host-memory
                # telemetry is unavailable.
                memory = AcceleratorMemory()

        device = AcceleratorDevice(
            device_id=(CPU_DEVICE_ID),
            kind=(AcceleratorKind.CPU.value),
            runtime_device=(CPU_RUNTIME_DEVICE),
            index=None,
            name=cpu_name(),
            state=(AcceleratorState.AVAILABLE.value),
            memory=memory,
            capabilities=(cpu_capabilities()),
            supported_precisions=(cpu_supported_precisions()),
            compute_capability=None,
            vendor=None,
            architecture=(platform.machine() or None),
            driver_version=None,
            runtime_version=(
                str(
                    getattr(
                        torch,
                        "__version__",
                        "",
                    )
                )
                if torch is not None
                else None
            ),
            logical_processor_count=(logical_processor_count()),
            multiprocessor_count=None,
            attributes=(self._device_attributes()),
        )

        device.validate()

        return (device,)

    # ---------------------------------------------------------
    # Memory
    # ---------------------------------------------------------

    def _memory(
        self,
        device: AcceleratorDevice,
    ) -> AcceleratorMemory:
        return system_memory()

    # ---------------------------------------------------------
    # Runtime device
    # ---------------------------------------------------------

    def _runtime_device(
        self,
        device: AcceleratorDevice,
    ) -> Any:
        """
        Return a native PyTorch CPU device when
        PyTorch is available.

        If PyTorch is absent, return the framework-
        neutral runtime-device string. This keeps
        CPU discovery usable by non-PyTorch
        callers.
        """

        if torch is None:
            return device.runtime_device

        try:
            return torch.device(device.runtime_device)

        except Exception as exc:
            raise AcceleratorRuntimeDeviceError(
                "CPU runtime device could not be constructed.",
                details=(
                    safe_accelerator_error_details(
                        exc,
                        operation=("cpu_runtime_device"),
                        backend=(AcceleratorKind.CPU.value),
                        device_id=(device.device_id),
                        runtime_device=(device.runtime_device),
                    )
                ),
            ) from exc

    # ---------------------------------------------------------
    # Synchronization
    # ---------------------------------------------------------

    def _synchronize(
        self,
        device: AcceleratorDevice,
    ) -> None:
        """
        CPU operations used by this backend are
        synchronous from the accelerator manager's
        perspective.

        No explicit backend synchronization is
        required.
        """

        return None

    # ---------------------------------------------------------
    # Cache clearing
    # ---------------------------------------------------------

    def _clear_cache(
        self,
        device: AcceleratorDevice,
    ) -> None:
        """
        CPU allocator/cache behavior is owned by
        Python, the operating system, and the
        selected framework.

        The accelerator layer does not force
        process-wide garbage collection or modify
        runtime allocator state.
        """

        return None

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------

    def _health(
        self,
        *,
        deep: bool,
    ) -> AcceleratorHealth:
        if not self.available():
            return AcceleratorHealth(
                component=("accelerator_backend:cpu"),
                status=(AcceleratorHealthStatus.UNAVAILABLE.value),
                message=("CPU execution backend is unavailable."),
                kind=(AcceleratorKind.CPU.value),
                device_id=(CPU_DEVICE_ID),
                retryable=False,
                attributes={
                    "deep_check": deep,
                },
            )

        memory = AcceleratorMemory()
        memory_error_type = None

        if self.config.health.check_memory:
            try:
                memory = system_memory()

            except Exception as exc:
                memory_error_type = type(exc).__name__

        minimum_free = self.config.health.minimum_free_memory_bytes

        insufficient_memory = bool(
            minimum_free is not None
            and (memory.free_bytes is None or memory.free_bytes < minimum_free)
        )

        if insufficient_memory:
            status = AcceleratorHealthStatus.DEGRADED.value

            message = (
                "CPU backend is available "
                "but does not satisfy the "
                "configured free-memory "
                "health requirement."
            )

        elif memory_error_type is not None:
            status = AcceleratorHealthStatus.DEGRADED.value

            message = (
                "CPU backend is available but memory telemetry could not be collected."
            )

        else:
            status = AcceleratorHealthStatus.HEALTHY.value

            message = "CPU accelerator backend is healthy."

        attributes: Dict[
            str,
            Any,
        ] = {
            "deep_check": deep,
            "logical_processor_count": (logical_processor_count()),
            "physical_processor_count": (physical_processor_count()),
            "memory": (memory.to_dict()),
            "memory_error_type": (memory_error_type),
            "minimum_free_memory_bytes": (minimum_free),
            "torch_available": (torch is not None),
            "psutil_available": (psutil is not None),
        }

        if deep and torch is not None:
            try:
                runtime_device = torch.device(CPU_RUNTIME_DEVICE)

                tensor = torch.tensor(
                    [
                        1.0,
                        2.0,
                    ],
                    dtype=torch.float32,
                    device=runtime_device,
                )

                result = float(tensor.sum().item())

                attributes["deep_tensor_check"] = bool(result == 3.0)

                if result != 3.0:
                    status = AcceleratorHealthStatus.UNHEALTHY.value

                    message = "CPU backend failed its deep runtime calculation."

            except Exception as exc:
                status = AcceleratorHealthStatus.UNHEALTHY.value

                message = "CPU backend failed its deep runtime check."

                attributes["deep_check_error_type"] = type(exc).__name__

        return AcceleratorHealth(
            component=("accelerator_backend:cpu"),
            status=status,
            message=message,
            kind=(AcceleratorKind.CPU.value),
            device_id=(CPU_DEVICE_ID),
            retryable=(status == AcceleratorHealthStatus.DEGRADED.value),
            error_type=(memory_error_type),
            attributes=attributes,
        )


# ---------------------------------------------------------------
# Compatibility API
# ---------------------------------------------------------------


class CPUAccelerator(CPUBackend):
    def is_available(self) -> bool:
        return self.available()

    def device_count(self) -> int:
        return self.discover().device_count

    def get_device(self, index: int = 0) -> str:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("Device index must be an integer.")
        if index != 0:
            raise ValueError("CPU exposes only device index 0.")
        return CPU_RUNTIME_DEVICE

    def device_name(self, index: int = 0) -> str:
        self.get_device(index)
        return "CPU"


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "CPU_DEVICE_ID",
    "CPU_RUNTIME_DEVICE",
    "logical_processor_count",
    "physical_processor_count",
    "cpu_name",
    "system_memory",
    "cpu_supported_precisions",
    "cpu_capabilities",
    "CPUBackend",
    "CPUAccelerator",
]
