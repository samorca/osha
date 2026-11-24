from __future__ import annotations

"""
NVIDIA CUDA accelerator backend for OSHA.

Responsibilities:

    detect a usable PyTorch CUDA runtime
    distinguish CUDA from ROCm/HIP
    enumerate visible NVIDIA devices
    report device properties and memory
    construct torch.device("cuda:N")
    synchronize a selected device
    clear unused cached allocator memory
    provide shallow and deep health checks

This backend does not:

    select a device across backends
    configure workload policy
    move models or tensors
    run training or inference
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from src.accelerators.base import (
    CUDAAcceleratorBackend,
)

from src.accelerators.config import (
    AcceleratorConfig,
)

from src.accelerators.errors import (
    AcceleratorBackendUnavailableError,
    AcceleratorDeviceIndexError,
    AcceleratorMemoryQueryError,
    AcceleratorRuntimeDeviceError,
    AcceleratorSynchronizationError,
    AcceleratorCacheClearError,
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
# Optional PyTorch runtime
# ---------------------------------------------------------------

try:
    import torch

except ImportError:
    torch = None


# ---------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------


def torch_available() -> bool:
    return torch is not None


def hip_runtime_available() -> bool:
    """
    Return True when the installed PyTorch runtime
    is backed by AMD HIP/ROCm.

    PyTorch exposes ROCm devices through much of
    the torch.cuda API, so the CUDA backend must
    explicitly exclude HIP-backed runtimes.
    """

    if torch is None:
        return False

    version = getattr(
        torch,
        "version",
        None,
    )

    return bool(
        version is not None
        and getattr(
            version,
            "hip",
            None,
        )
    )


def cuda_runtime_version() -> Optional[str]:
    if torch is None:
        return None

    version = getattr(
        torch,
        "version",
        None,
    )

    if version is None:
        return None

    value = getattr(
        version,
        "cuda",
        None,
    )

    if value is None:
        return None

    text = str(value).strip()

    return text or None


def torch_runtime_version() -> Optional[str]:
    if torch is None:
        return None

    value = getattr(
        torch,
        "__version__",
        None,
    )

    if value is None:
        return None

    text = str(value).strip()

    return text or None


# ---------------------------------------------------------------
# Device index validation
# ---------------------------------------------------------------


def require_cuda_index(
    index: Optional[int],
) -> int:
    if index is None:
        raise AcceleratorDeviceIndexError("CUDA device requires an integer index.")

    if isinstance(
        index,
        bool,
    ):
        raise AcceleratorDeviceIndexError("CUDA device index must be an integer.")

    try:
        resolved = int(index)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise AcceleratorDeviceIndexError(
            "CUDA device index must be an integer."
        ) from exc

    if resolved < 0:
        raise AcceleratorDeviceIndexError(
            "CUDA device index must be greater than or equal to zero."
        )

    if torch is not None:
        try:
            count = int(torch.cuda.device_count())

        except Exception as exc:
            raise AcceleratorDeviceIndexError(
                "CUDA device count could not be determined.",
                details=(
                    safe_accelerator_error_details(
                        exc,
                        operation=("cuda_device_count"),
                        backend=(AcceleratorKind.CUDA.value),
                    )
                ),
            ) from exc

        if resolved >= count:
            raise AcceleratorDeviceIndexError(
                "CUDA device index is outside the visible device range.",
                details={
                    "device_index": (resolved),
                    "visible_devices": (count),
                },
            )

    return resolved


# ---------------------------------------------------------------
# Compute capability
# ---------------------------------------------------------------


def compute_capability(
    index: int,
) -> Optional[str]:
    if torch is None:
        return None

    try:
        major, minor = torch.cuda.get_device_capability(index)

        return str(int(major)) + "." + str(int(minor))

    except Exception:
        return None


# ---------------------------------------------------------------
# BF16 support
# ---------------------------------------------------------------


def cuda_bfloat16_supported(
    index: int,
) -> bool:
    if torch is None:
        return False

    checker = getattr(
        torch.cuda,
        "is_bf16_supported",
        None,
    )

    if not callable(checker):
        return False

    try:
        with torch.cuda.device(index):
            return bool(checker())

    except (
        TypeError,
        RuntimeError,
    ):
        try:
            return bool(checker(including_emulation=False))

        except Exception:
            return False

    except Exception:
        return False


# ---------------------------------------------------------------
# Precision discovery
# ---------------------------------------------------------------


def cuda_supported_precisions(
    index: int,
) -> Tuple[str, ...]:
    """
    Return conservative precision support for a
    CUDA device.

    FLOAT32, FLOAT64 and FLOAT16 are exposed for
    normal CUDA execution. BFLOAT16 is included
    only when the runtime reports support.
    """

    values: List[str] = [
        AcceleratorPrecision.FLOAT64.value,
        AcceleratorPrecision.FLOAT32.value,
        AcceleratorPrecision.FLOAT16.value,
        AcceleratorPrecision.INT8.value,
    ]

    if cuda_bfloat16_supported(index):
        values.append(AcceleratorPrecision.BFLOAT16.value)

    return tuple(values)


# ---------------------------------------------------------------
# Capability discovery
# ---------------------------------------------------------------


def cuda_capabilities(
    index: int,
) -> Tuple[str, ...]:
    values: List[str] = [
        AcceleratorCapability.COMPUTE.value,
        AcceleratorCapability.TRAINING.value,
        AcceleratorCapability.INFERENCE.value,
        AcceleratorCapability.MIXED_PRECISION.value,
        AcceleratorCapability.ASYNC_EXECUTION.value,
        AcceleratorCapability.MEMORY_STATS.value,
        AcceleratorCapability.SYNCHRONIZATION.value,
        AcceleratorCapability.CACHE_CONTROL.value,
    ]

    capability = compute_capability(index)

    if capability is not None:
        try:
            major = int(
                capability.split(
                    ".",
                    1,
                )[0]
            )

        except (
            TypeError,
            ValueError,
        ):
            major = 0

        if major >= 7:
            values.append(AcceleratorCapability.TENSOR_CORES.value)

    if torch is not None:
        peer_checker = getattr(
            torch.cuda,
            "can_device_access_peer",
            None,
        )

        if callable(peer_checker):
            try:
                count = int(torch.cuda.device_count())

                if any(
                    other != index
                    and bool(
                        peer_checker(
                            index,
                            other,
                        )
                    )
                    for other in range(count)
                ):
                    values.append(AcceleratorCapability.PEER_TO_PEER.value)

            except Exception:
                pass

    return tuple(dict.fromkeys(values))


# ---------------------------------------------------------------
# Memory query
# ---------------------------------------------------------------


def cuda_memory(
    index: int,
) -> AcceleratorMemory:
    if torch is None:
        raise AcceleratorBackendUnavailableError(
            "PyTorch is unavailable for CUDA memory inspection."
        )

    resolved = require_cuda_index(index)

    runtime_device = "cuda:" + str(resolved)

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(resolved)

        total = int(total_bytes)

        free = int(free_bytes)

        used = max(
            0,
            total - free,
        )

        reserved = None
        allocated = None

        memory_reserved = getattr(
            torch.cuda,
            "memory_reserved",
            None,
        )

        if callable(memory_reserved):
            reserved = int(memory_reserved(resolved))

        memory_allocated = getattr(
            torch.cuda,
            "memory_allocated",
            None,
        )

        if callable(memory_allocated):
            allocated = int(memory_allocated(resolved))

        result = AcceleratorMemory(
            total_bytes=total,
            free_bytes=free,
            used_bytes=used,
            reserved_bytes=(reserved),
            allocated_bytes=(allocated),
        )

        result.validate()

        return result

    except AcceleratorMemoryQueryError:
        raise

    except Exception as exc:
        raise AcceleratorMemoryQueryError(
            "CUDA memory query failed.",
            details=(
                safe_accelerator_error_details(
                    exc,
                    operation=("cuda_memory"),
                    backend=(AcceleratorKind.CUDA.value),
                    device_id=(runtime_device),
                    runtime_device=(runtime_device),
                )
            ),
        ) from exc


# ---------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------


class CUDABackend(CUDAAcceleratorBackend):
    """
    Concrete NVIDIA CUDA backend.

    Each visible CUDA device is represented as a
    separate AcceleratorDevice:

        cuda:0
        cuda:1
        ...
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
        if torch is None:
            return False

        if hip_runtime_available():
            return False

        try:
            return bool(
                torch.cuda.is_available() and int(torch.cuda.device_count()) > 0
            )

        except Exception:
            return False

    # ---------------------------------------------------------
    # Device properties
    # ---------------------------------------------------------

    def _device_attributes(
        self,
        index: int,
        properties: Any,
    ) -> Dict[str, Any]:
        values: Dict[
            str,
            Any,
        ] = {
            "torch_version": (torch_runtime_version()),
            "cuda_runtime_version": (cuda_runtime_version()),
            "hip_runtime": False,
        }

        property_names = (
            "major",
            "minor",
            "total_memory",
            "multi_processor_count",
            "is_integrated",
            "is_multi_gpu_board",
        )

        for name in property_names:
            value = getattr(
                properties,
                name,
                None,
            )

            if value is not None:
                values[name] = value

        uuid_value = getattr(
            properties,
            "uuid",
            None,
        )

        if uuid_value is not None:
            values["uuid"] = str(uuid_value)

        pci_bus_id = getattr(
            properties,
            "pci_bus_id",
            None,
        )

        if pci_bus_id is not None:
            values["pci_bus_id"] = str(pci_bus_id)

        values["bfloat16_supported"] = cuda_bfloat16_supported(index)

        return values

    # ---------------------------------------------------------
    # Discover one device
    # ---------------------------------------------------------

    def _discover_device(
        self,
        index: int,
    ) -> AcceleratorDevice:
        if torch is None:
            raise AcceleratorBackendUnavailableError(
                "PyTorch is unavailable for CUDA discovery."
            )

        resolved = require_cuda_index(index)

        runtime_device = "cuda:" + str(resolved)

        try:
            properties = torch.cuda.get_device_properties(resolved)

            name = str(torch.cuda.get_device_name(resolved)).strip()

            if not name:
                name = "CUDA device " + str(resolved)

            memory = AcceleratorMemory()

            if self.config.discovery.refresh_memory_on_detection:
                try:
                    memory = cuda_memory(resolved)

                except AcceleratorMemoryQueryError:
                    total_memory = getattr(
                        properties,
                        "total_memory",
                        None,
                    )

                    memory = AcceleratorMemory(
                        total_bytes=(
                            int(total_memory) if total_memory is not None else None
                        )
                    )

            capability = compute_capability(resolved)

            device = AcceleratorDevice(
                device_id=(runtime_device),
                kind=(AcceleratorKind.CUDA.value),
                runtime_device=(runtime_device),
                index=resolved,
                name=name,
                state=(AcceleratorState.AVAILABLE.value),
                memory=memory,
                capabilities=(cuda_capabilities(resolved)),
                supported_precisions=(cuda_supported_precisions(resolved)),
                compute_capability=(capability),
                vendor="NVIDIA",
                architecture=(
                    "sm_"
                    + capability.replace(
                        ".",
                        "",
                    )
                    if capability
                    else None
                ),
                driver_version=None,
                runtime_version=(cuda_runtime_version()),
                logical_processor_count=None,
                multiprocessor_count=(
                    int(
                        getattr(
                            properties,
                            "multi_processor_count",
                            0,
                        )
                    )
                    or None
                ),
                attributes=(
                    self._device_attributes(
                        resolved,
                        properties,
                    )
                ),
            )

            device.validate()

            return device

        except Exception as exc:
            if isinstance(
                exc,
                (
                    AcceleratorBackendUnavailableError,
                    AcceleratorDeviceIndexError,
                ),
            ):
                raise

            raise AcceleratorBackendUnavailableError(
                "CUDA device could not be inspected.",
                details=(
                    safe_accelerator_error_details(
                        exc,
                        operation=("cuda_discover_device"),
                        backend=(AcceleratorKind.CUDA.value),
                        device_id=(runtime_device),
                        runtime_device=(runtime_device),
                    )
                ),
            ) from exc

    # ---------------------------------------------------------
    # Discovery
    # ---------------------------------------------------------

    def _discover(
        self,
    ) -> Tuple[
        AcceleratorDevice,
        ...,
    ]:
        if torch is None:
            return ()

        if hip_runtime_available():
            return ()

        try:
            count = int(torch.cuda.device_count())

        except Exception as exc:
            raise AcceleratorBackendUnavailableError(
                "CUDA device count could not be determined.",
                details=(
                    safe_accelerator_error_details(
                        exc,
                        operation=("cuda_device_count"),
                        backend=(AcceleratorKind.CUDA.value),
                    )
                ),
            ) from exc

        devices: List[AcceleratorDevice] = []

        for index in range(count):
            try:
                devices.append(self._discover_device(index))

            except Exception:
                if self.config.discovery.fail_on_backend_error:
                    raise

                if self.config.discovery.include_unavailable_devices:
                    runtime_device = "cuda:" + str(index)

                    devices.append(
                        AcceleratorDevice(
                            device_id=(runtime_device),
                            kind=(AcceleratorKind.CUDA.value),
                            runtime_device=(runtime_device),
                            index=index,
                            name=("CUDA device " + str(index)),
                            state=(AcceleratorState.UNAVAILABLE.value),
                            memory=(AcceleratorMemory()),
                            capabilities=(),
                            supported_precisions=(AcceleratorPrecision.FLOAT32.value,),
                            vendor="NVIDIA",
                            runtime_version=(cuda_runtime_version()),
                            attributes={
                                "inspection_failed": (True),
                            },
                        )
                    )

        return tuple(devices)

    # ---------------------------------------------------------
    # Memory
    # ---------------------------------------------------------

    def _memory(
        self,
        device: AcceleratorDevice,
    ) -> AcceleratorMemory:
        index = require_cuda_index(device.index)

        return cuda_memory(index)

    # ---------------------------------------------------------
    # Runtime device
    # ---------------------------------------------------------

    def _runtime_device(
        self,
        device: AcceleratorDevice,
    ) -> Any:
        if torch is None:
            raise AcceleratorRuntimeDeviceError(
                "PyTorch is unavailable for CUDA runtime-device construction."
            )

        index = require_cuda_index(device.index)

        try:
            runtime_device = torch.device(
                "cuda",
                index,
            )

            if runtime_device.type != "cuda":
                raise RuntimeError("Runtime device is not CUDA.")

            return runtime_device

        except Exception as exc:
            raise AcceleratorRuntimeDeviceError(
                "CUDA runtime device could not be constructed.",
                details=(
                    safe_accelerator_error_details(
                        exc,
                        operation=("cuda_runtime_device"),
                        backend=(AcceleratorKind.CUDA.value),
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
        if torch is None:
            raise AcceleratorSynchronizationError(
                "PyTorch is unavailable for CUDA synchronization."
            )

        index = require_cuda_index(device.index)

        try:
            torch.cuda.synchronize(index)

        except Exception as exc:
            raise AcceleratorSynchronizationError(
                "CUDA synchronization failed.",
                details=(
                    safe_accelerator_error_details(
                        exc,
                        operation=("cuda_synchronize"),
                        backend=(AcceleratorKind.CUDA.value),
                        device_id=(device.device_id),
                        runtime_device=(device.runtime_device),
                    )
                ),
            ) from exc

    # ---------------------------------------------------------
    # Cache clearing
    # ---------------------------------------------------------

    def _clear_cache(
        self,
        device: AcceleratorDevice,
    ) -> None:
        if torch is None:
            raise AcceleratorCacheClearError(
                "PyTorch is unavailable for CUDA cache clearing."
            )

        index = require_cuda_index(device.index)

        try:
            with torch.cuda.device(index):
                torch.cuda.empty_cache()

                ipc_collect = getattr(
                    torch.cuda,
                    "ipc_collect",
                    None,
                )

                if callable(ipc_collect):
                    ipc_collect()

        except Exception as exc:
            raise AcceleratorCacheClearError(
                "CUDA cache could not be cleared.",
                details=(
                    safe_accelerator_error_details(
                        exc,
                        operation=("cuda_clear_cache"),
                        backend=(AcceleratorKind.CUDA.value),
                        device_id=(device.device_id),
                        runtime_device=(device.runtime_device),
                    )
                ),
            ) from exc

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------

    def _health(
        self,
        *,
        deep: bool,
    ) -> AcceleratorHealth:
        attributes: Dict[
            str,
            Any,
        ] = {
            "deep_check": deep,
            "torch_available": (torch is not None),
            "hip_runtime": (hip_runtime_available()),
            "cuda_runtime_version": (cuda_runtime_version()),
            "torch_version": (torch_runtime_version()),
        }

        if torch is None:
            return AcceleratorHealth(
                component=("accelerator_backend:cuda"),
                status=(AcceleratorHealthStatus.UNAVAILABLE.value),
                message=("PyTorch CUDA runtime is unavailable."),
                kind=(AcceleratorKind.CUDA.value),
                retryable=False,
                attributes=attributes,
            )

        if hip_runtime_available():
            return AcceleratorHealth(
                component=("accelerator_backend:cuda"),
                status=(AcceleratorHealthStatus.UNAVAILABLE.value),
                message=("The installed PyTorch runtime is ROCm/HIP, not NVIDIA CUDA."),
                kind=(AcceleratorKind.CUDA.value),
                retryable=False,
                attributes=attributes,
            )

        if not self.available():
            return AcceleratorHealth(
                component=("accelerator_backend:cuda"),
                status=(AcceleratorHealthStatus.UNAVAILABLE.value),
                message=("No usable NVIDIA CUDA device is visible."),
                kind=(AcceleratorKind.CUDA.value),
                retryable=True,
                attributes=attributes,
            )

        discovery = self.discover()

        devices = discovery.available_devices

        attributes["visible_devices"] = len(devices)

        attributes["devices"] = [
            {
                "device_id": (device.device_id),
                "name": (device.name),
                "memory": (device.memory.to_dict()),
                "compute_capability": (device.compute_capability),
            }
            for device in devices
        ]

        minimum_free = self.config.health.minimum_free_memory_bytes

        insufficient_devices: List[str] = []

        memory_unknown: List[str] = []

        if self.config.health.check_memory:
            for device in devices:
                free_bytes = device.memory.free_bytes

                if free_bytes is None:
                    memory_unknown.append(device.device_id)

                elif minimum_free is not None and free_bytes < minimum_free:
                    insufficient_devices.append(device.device_id)

        status = AcceleratorHealthStatus.HEALTHY.value

        message = "CUDA accelerator backend is healthy."

        if devices and len(insufficient_devices) == len(devices):
            status = AcceleratorHealthStatus.DEGRADED.value

            message = (
                "CUDA devices are "
                "available but none "
                "satisfies the configured "
                "free-memory health "
                "requirement."
            )

        elif (
            devices
            and len(memory_unknown) == len(devices)
            and self.config.health.check_memory
        ):
            status = AcceleratorHealthStatus.DEGRADED.value

            message = "CUDA devices are available but memory telemetry is unavailable."

        attributes["minimum_free_memory_bytes"] = minimum_free

        attributes["insufficient_memory_devices"] = insufficient_devices

        attributes["memory_unknown_devices"] = memory_unknown

        if deep:
            deep_results: Dict[
                str,
                Any,
            ] = {}

            for device in devices:
                try:
                    runtime_device = self.runtime_device(device)

                    with torch.inference_mode():
                        tensor = torch.tensor(
                            [
                                1.0,
                                2.0,
                            ],
                            dtype=(torch.float32),
                            device=(runtime_device),
                        )

                        value = float(tensor.sum().item())

                    self.synchronize(device)

                    deep_results[device.device_id] = {
                        "passed": bool(value == 3.0),
                    }

                    if value != 3.0:
                        status = AcceleratorHealthStatus.UNHEALTHY.value

                        message = "A CUDA device failed its deep runtime check."

                except Exception as exc:
                    deep_results[device.device_id] = {
                        "passed": False,
                        "error_type": (type(exc).__name__),
                    }

                    status = AcceleratorHealthStatus.UNHEALTHY.value

                    message = "A CUDA device failed its deep runtime check."

            attributes["deep_results"] = deep_results

        return AcceleratorHealth(
            component=("accelerator_backend:cuda"),
            status=status,
            message=message,
            kind=(AcceleratorKind.CUDA.value),
            retryable=(
                status
                in {
                    AcceleratorHealthStatus.DEGRADED.value,
                    AcceleratorHealthStatus.UNHEALTHY.value,
                }
            ),
            attributes=attributes,
        )


# ---------------------------------------------------------------
# Compatibility API
# ---------------------------------------------------------------


class CUDAAccelerator(CUDABackend):
    def is_available(self) -> bool:
        runtime = _compat_torch()
        cuda = getattr(runtime, "cuda", None) if runtime else None
        version = getattr(runtime, "version", None) if runtime else None
        available = getattr(cuda, "is_available", None)
        return bool(
            runtime is not None
            and not getattr(version, "hip", None)
            and callable(available)
            and available()
        )

    def device_count(self) -> int:
        runtime = _compat_torch()
        if not self.is_available() or runtime is None:
            return 0
        return int(runtime.cuda.device_count())

    def get_device(self, index: int = 0):
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("Device index must be an integer.")
        if not self.is_available():
            raise RuntimeError("CUDA accelerator is not available.")
        count = self.device_count()
        if index < 0 or index >= count:
            raise ValueError(f"Invalid CUDA device index {index}.")
        runtime = _compat_torch()
        return runtime.device(f"cuda:{index}")

    def device_name(self, index: int = 0) -> str:
        self.get_device(index)
        runtime = _compat_torch()
        return str(runtime.cuda.get_device_name(index))


def _compat_torch():
    try:
        import torch as runtime
    except ImportError:
        return None
    return runtime


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "torch_available",
    "hip_runtime_available",
    "cuda_runtime_version",
    "torch_runtime_version",
    "require_cuda_index",
    "compute_capability",
    "cuda_bfloat16_supported",
    "cuda_supported_precisions",
    "cuda_capabilities",
    "cuda_memory",
    "CUDABackend",
    "CUDAAccelerator",
]
