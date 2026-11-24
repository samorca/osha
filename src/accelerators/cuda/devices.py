from __future__ import annotations

"""
CUDA device inspection and workload suitability.

This module enriches the framework-neutral
AcceleratorDevice contract with CUDA-specific
information.

Responsibilities:

    CUDA device resolution
    device-property inspection
    compute-capability inspection
    architecture inspection
    precision and Tensor Core inspection
    peer-access inspection
    memory-capacity inspection
    workload suitability analysis
    optimization-profile construction

This module does not:

    select devices across backends
    allocate tensors
    clear CUDA caches
    synchronize CUDA streams
    enable mixed precision
    compile models
    execute training or inference
"""

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

from src.accelerators.cuda.backend import (
    CUDABackend,
    compute_capability,
    cuda_bfloat16_supported,
    cuda_capabilities,
    cuda_memory,
    cuda_runtime_version,
    cuda_supported_precisions,
    hip_runtime_available,
    require_cuda_index,
    torch_available,
    torch_runtime_version,
)

from src.accelerators.optimization import (
    AcceleratorOptimizationProfile,
    AcceleratorWorkload,
)

from src.accelerators.schema import (
    AcceleratorCapability,
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
# Optional PyTorch runtime
# ------------------------------------------------------------------

try:
    import torch

except ImportError:
    torch = None


# ------------------------------------------------------------------
# CUDA device profile
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CUDADeviceProfile:
    """
    Rich NVIDIA CUDA device profile.

    AcceleratorDevice remains the common hardware
    identity used by AcceleratorManager.

    This profile adds CUDA-specific properties for
    metrics collection and optimization advice.
    """

    device: AcceleratorDevice

    index: int

    name: str

    compute_capability: Optional[str] = None

    compute_capability_major: Optional[int] = None

    compute_capability_minor: Optional[int] = None

    architecture: Optional[str] = None

    multiprocessor_count: Optional[int] = None

    total_memory_bytes: Optional[int] = None

    free_memory_bytes: Optional[int] = None

    allocated_memory_bytes: Optional[int] = None

    reserved_memory_bytes: Optional[int] = None

    tensor_core_candidate: bool = False

    bfloat16_supported: bool = False

    tf32_supported: bool = False

    peer_access_devices: Tuple[
        str,
        ...,
    ] = ()

    supported_precisions: Tuple[
        str,
        ...,
    ] = ()

    capabilities: Tuple[
        str,
        ...,
    ] = ()

    cuda_runtime_version: Optional[str] = None

    torch_runtime_version: Optional[str] = None

    attributes: Mapping[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        self.device.validate()

        if self.device.kind != AcceleratorKind.CUDA.value:
            raise ValueError("CUDADeviceProfile requires a CUDA AcceleratorDevice.")

        if self.index < 0:
            raise ValueError("CUDA device index must be >= 0.")

        if self.device.index is None or self.device.index != self.index:
            raise ValueError("CUDA profile index does not match the device index.")

        if not str(self.name).strip():
            raise ValueError("CUDA device name cannot be empty.")

        for field_name, value in (
            (
                "compute_capability_major",
                self.compute_capability_major,
            ),
            (
                "compute_capability_minor",
                self.compute_capability_minor,
            ),
            (
                "multiprocessor_count",
                self.multiprocessor_count,
            ),
            (
                "total_memory_bytes",
                self.total_memory_bytes,
            ),
            (
                "free_memory_bytes",
                self.free_memory_bytes,
            ),
            (
                "allocated_memory_bytes",
                self.allocated_memory_bytes,
            ),
            (
                "reserved_memory_bytes",
                self.reserved_memory_bytes,
            ),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be >= 0.")

        if (
            self.total_memory_bytes is not None
            and self.free_memory_bytes is not None
            and self.free_memory_bytes > self.total_memory_bytes
        ):
            raise ValueError("CUDA free memory cannot exceed total memory.")

        valid_precisions = {item.value for item in AcceleratorPrecision}

        unsupported = [
            precision
            for precision in self.supported_precisions
            if precision not in valid_precisions
        ]

        if unsupported:
            raise ValueError(
                "CUDA profile contains unsupported "
                "precision identifiers: " + ", ".join(unsupported)
            )

    @property
    def used_memory_bytes(
        self,
    ) -> Optional[int]:
        if self.total_memory_bytes is None or self.free_memory_bytes is None:
            return None

        return max(
            0,
            self.total_memory_bytes - self.free_memory_bytes,
        )

    @property
    def memory_utilization_ratio(
        self,
    ) -> Optional[float]:
        if (
            self.total_memory_bytes is None
            or self.total_memory_bytes <= 0
            or self.used_memory_bytes is None
        ):
            return None

        return self.used_memory_bytes / self.total_memory_bytes

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
            "index": self.index,
            "name": self.name,
            "compute_capability": (self.compute_capability),
            "compute_capability_major": (self.compute_capability_major),
            "compute_capability_minor": (self.compute_capability_minor),
            "architecture": (self.architecture),
            "multiprocessor_count": (self.multiprocessor_count),
            "total_memory_bytes": (self.total_memory_bytes),
            "free_memory_bytes": (self.free_memory_bytes),
            "used_memory_bytes": (self.used_memory_bytes),
            "allocated_memory_bytes": (self.allocated_memory_bytes),
            "reserved_memory_bytes": (self.reserved_memory_bytes),
            "memory_utilization_ratio": (self.memory_utilization_ratio),
            "tensor_core_candidate": (self.tensor_core_candidate),
            "bfloat16_supported": (self.bfloat16_supported),
            "tf32_supported": (self.tf32_supported),
            "peer_access_devices": list(self.peer_access_devices),
            "supported_precisions": list(self.supported_precisions),
            "capabilities": list(self.capabilities),
            "cuda_runtime_version": (self.cuda_runtime_version),
            "torch_runtime_version": (self.torch_runtime_version),
            "attributes": (to_json_safe(dict(self.attributes))),
        }


# ------------------------------------------------------------------
# CUDA runtime guards
# ------------------------------------------------------------------


def require_cuda_runtime() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is unavailable for CUDA device inspection.")

    if hip_runtime_available():
        raise RuntimeError(
            "The installed PyTorch runtime uses ROCm/HIP rather than NVIDIA CUDA."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")


# ------------------------------------------------------------------
# Compute-capability parsing
# ------------------------------------------------------------------


def parse_compute_capability(
    value: Optional[str],
) -> Tuple[
    Optional[int],
    Optional[int],
]:
    if value is None:
        return (
            None,
            None,
        )

    text = str(value).strip()

    if not text:
        return (
            None,
            None,
        )

    parts = text.split(
        ".",
        1,
    )

    try:
        major = int(parts[0])

        minor = int(parts[1]) if len(parts) > 1 else 0

    except (
        TypeError,
        ValueError,
    ):
        return (
            None,
            None,
        )

    return (
        major,
        minor,
    )


# ------------------------------------------------------------------
# Architecture
# ------------------------------------------------------------------


def cuda_architecture(
    index: int,
) -> Optional[str]:
    capability = compute_capability(require_cuda_index(index))

    if capability is None:
        return None

    return "sm_" + capability.replace(
        ".",
        "",
    )


# ------------------------------------------------------------------
# TF32 support
# ------------------------------------------------------------------


def cuda_tf32_supported(
    index: int,
) -> bool:
    """
    Query runtime TF32 support when the installed
    PyTorch version exposes the capability check.

    False means unsupported or unknown.
    """

    if torch is None:
        return False

    resolved = require_cuda_index(index)

    checker = getattr(
        torch.cuda,
        "is_tf32_supported",
        None,
    )

    if callable(checker):
        try:
            with torch.cuda.device(resolved):
                return bool(checker())

        except Exception:
            return False

    capability = compute_capability(resolved)

    major, _ = parse_compute_capability(capability)

    return bool(major is not None and major >= 8)


# ------------------------------------------------------------------
# Tensor Core candidate
# ------------------------------------------------------------------


def cuda_tensor_core_candidate(
    index: int,
) -> bool:
    """
    Return whether the generic CUDA capabilityility
    profile marks the device nsor Core
    capable.

    This does not guarantee that a particular
    workload will use Tensor Cores.
    """

    return AcceleratorCapability.TENSOR_CORES.value in set(
        cuda_capabilities(require_cuda_index(index))
    )


# ------------------------------------------------------------------
# Peer-access inspection
# ------------------------------------------------------------------


def cuda_peer_access_devices(
    index: int,
) -> Tuple[str, ...]:
    if torch is None:
        return ()

    resolved = require_cuda_index(index)

    checker = getattr(
        torch.cuda,
        "can_device_access_peer",
        None,
    )

    if not callable(checker):
        return ()

    try:
        count = int(torch.cuda.device_count())

    except Exception:
        return ()

    output = []

    for other in range(count):
        if other == resolved:
            continue

        try:
            if bool(
                checker(
                    resolved,
                    other,
                )
            ):
                output.append("cuda:" + str(other))

        except Exception:
            continue

    return tuple(output)


# ------------------------------------------------------------------
# Compiled architecture list
# ------------------------------------------------------------------


def compiled_cuda_architectures() -> Tuple[str, ...]:
    if torch is None:
        return ()

    getter = getattr(
        torch.cuda,
        "get_arch_list",
        None,
    )

    if not callable(getter):
        return ()

    try:
        values = getter()

        return tuple(str(value).strip() for value in values if str(value).strip())

    except Exception:
        return ()


# ------------------------------------------------------------------
# Device memory
# ------------------------------------------------------------------


def resolve_cuda_memory(
    index: int,
) -> AcceleratorMemory:
    return cuda_memory(require_cuda_index(index))


# ------------------------------------------------------------------
# Resolve one CUDA device
# ------------------------------------------------------------------


def resolve_cuda_device(
    index: int = 0,
    *,
    backend: Optional[CUDABackend] = None,
) -> AcceleratorDevice:
    runtime_backend = backend or CUDABackend()

    resolved_index = require_cuda_index(index)

    expected_id = "cuda:" + str(resolved_index)

    result = runtime_backend.discover()

    for device in result.devices:
        if (
            device.kind == AcceleratorKind.CUDA.value
            and device.device_id == expected_id
        ):
            return device

    return AcceleratorDevice(
        device_id=expected_id,
        kind=(AcceleratorKind.CUDA.value),
        runtime_device=expected_id,
        index=resolved_index,
        name=("CUDA device " + str(resolved_index)),
        state=(AcceleratorState.UNAVAILABLE.value),
        memory=(AcceleratorMemory()),
        capabilities=(),
        supported_precisions=(AcceleratorPrecision.FLOAT32.value,),
        vendor="NVIDIA",
        runtime_version=(cuda_runtime_version()),
        attributes={
            "resolved_without_available_backend": (True),
        },
    )


# ------------------------------------------------------------------
# CUDA device inspector
# ------------------------------------------------------------------


class CUDADeviceInspector:
    """
    Build a rich CUDADeviceProfile.

    Device inspection is read-only.
    """

    def __init__(
        self,
        *,
        backend: Optional[CUDABackend] = None,
    ) -> None:
        self.backend = backend or CUDABackend()

    # ---------------------------------------------------------
    # Inspect
    # ---------------------------------------------------------

    def inspect(
        self,
        device: Optional[AcceleratorDevice] = None,
        *,
        index: Optional[int] = None,
        refresh_memory: bool = True,
    ) -> CUDADeviceProfile:
        if device is None:
            resolved = resolve_cuda_device(
                0 if index is None else index,
                backend=self.backend,
            )

        else:
            resolved = device

        resolved.validate()

        if resolved.kind != AcceleratorKind.CUDA.value:
            raise ValueError("CUDADeviceInspector cannot inspect a non-CUDA device.")

        resolved_index = require_cuda_index(resolved.index)

        memory = resolved.memory
        memory_error_type = None

        if refresh_memory and resolved.available:
            try:
                memory = self.backend.memory(resolved)

            except Exception as exc:
                memory_error_type = type(exc).__name__

        capability = resolved.compute_capability or compute_capability(resolved_index)

        major, minor = parse_compute_capability(capability)

        properties = None
        property_error_type = None

        if torch is not None and resolved.available:
            try:
                properties = torch.cuda.get_device_properties(resolved_index)

            except Exception as exc:
                property_error_type = type(exc).__name__

        name = resolved.name or ("CUDA device " + str(resolved_index))

        multiprocessor_count = resolved.multiprocessor_count

        if multiprocessor_count is None and properties is not None:
            value = getattr(
                properties,
                "multi_processor_count",
                None,
            )

            if value is not None:
                multiprocessor_count = int(value)

        profile = CUDADeviceProfile(
            device=resolved,
            index=resolved_index,
            name=name,
            compute_capability=(capability),
            compute_capability_major=(major),
            compute_capability_minor=(minor),
            architecture=(resolved.architecture or cuda_architecture(resolved_index)),
            multiprocessor_count=(multiprocessor_count),
            total_memory_bytes=(memory.total_bytes),
            free_memory_bytes=(memory.free_bytes),
            allocated_memory_bytes=(memory.allocated_bytes),
            reserved_memory_bytes=(memory.reserved_bytes),
            tensor_core_candidate=(cuda_tensor_core_candidate(resolved_index)),
            bfloat16_supported=(cuda_bfloat16_supported(resolved_index)),
            tf32_supported=(cuda_tf32_supported(resolved_index)),
            peer_access_devices=(cuda_peer_access_devices(resolved_index)),
            supported_precisions=(
                resolved.supported_precisions
                or cuda_supported_precisions(resolved_index)
            ),
            capabilities=(resolved.capabilities or cuda_capabilities(resolved_index)),
            cuda_runtime_version=(cuda_runtime_version()),
            torch_runtime_version=(torch_runtime_version()),
            attributes={
                "memory_error_type": (memory_error_type),
                "property_error_type": (property_error_type),
                "torch_available": (torch_available()),
                "hip_runtime": (hip_runtime_available()),
                "compiled_architectures": list(compiled_cuda_architectures()),
                "vendor": (resolved.vendor or "NVIDIA"),
                "device_attributes": dict(resolved.attributes),
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
        index: Optional[int] = None,
        refresh_memory: bool = True,
    ) -> AcceleratorOptimizationProfile:
        workload.validate()

        profile = self.inspect(
            device,
            index=index,
            refresh_memory=(refresh_memory),
        )

        utilization_percent = None

        if profile.memory_utilization_ratio is not None:
            utilization_percent = profile.memory_utilization_ratio * 100.0

        result = AcceleratorOptimizationProfile(
            device=profile.device,
            workload=workload,
            free_memory_bytes=(profile.free_memory_bytes),
            total_memory_bytes=(profile.total_memory_bytes),
            utilization_percent=(utilization_percent),
            temperature_celsius=None,
            capabilities=(profile.capabilities),
            attributes={
                "compute_capability": (profile.compute_capability),
                "architecture": (profile.architecture),
                "multiprocessor_count": (profile.multiprocessor_count),
                "tensor_core_candidate": (profile.tensor_core_candidate),
                "bfloat16_supported": (profile.bfloat16_supported),
                "tf32_supported": (profile.tf32_supported),
                "peer_access_devices": list(profile.peer_access_devices),
                "supported_precisions": list(profile.supported_precisions),
                "allocated_memory_bytes": (profile.allocated_memory_bytes),
                "reserved_memory_bytes": (profile.reserved_memory_bytes),
                "cuda_runtime_version": (profile.cuda_runtime_version),
                "torch_runtime_version": (profile.torch_runtime_version),
            },
        )

        result.validate()

        return result


# ------------------------------------------------------------------
# Workload suitability
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CUDAWorkloadSuitability:
    """
    Read-only CUDA workload suitability result.

    This is descriptive input for the CUDA
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
            "suitable": self.suitable,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "attributes": (to_json_safe(dict(self.attributes))),
        }


def evaluate_cuda_workload(
    profile: CUDADeviceProfile,
    workload: AcceleratorWorkload,
) -> CUDAWorkloadSuitability:
    """
    Evaluate explicitly known CUDA workload
    requirements.

    This function does not predict throughput or
    invent hardware performance estimates.
    """

    profile.validate()
    workload.validate()

    reasons = []
    warnings = []

    if not profile.device.available:
        reasons.append("cuda_device_unavailable")

    if not profile.supports_precision(workload.precision):
        reasons.append("workload_precision_unsupported")

    if workload.mixed_precision_enabled and not profile.supports_capability(
        AcceleratorCapability.MIXED_PRECISION.value
    ):
        reasons.append("mixed_precision_capability_missing")

    if (
        workload.mixed_precision_enabled
        and workload.precision == AcceleratorPrecision.BFLOAT16.value
        and not profile.bfloat16_supported
    ):
        reasons.append("bfloat16_unsupported")

    if workload.estimated_peak_memory_bytes is not None:
        if profile.free_memory_bytes is None:
            warnings.append("free_device_memory_unknown")

        elif workload.estimated_peak_memory_bytes > profile.free_memory_bytes:
            reasons.append("estimated_peak_memory_exceeds_free_vram")

    if workload.mixed_precision_enabled and not profile.tensor_core_candidate:
        warnings.append("mixed_precision_without_tensor_core_capability")

    if profile.compute_capability is None:
        warnings.append("compute_capability_unknown")

    return CUDAWorkloadSuitability(
        suitable=(not reasons),
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        attributes={
            "device_id": (profile.device.device_id),
            "compute_capability": (profile.compute_capability),
            "architecture": (profile.architecture),
            "free_memory_bytes": (profile.free_memory_bytes),
            "estimated_peak_memory_bytes": (workload.estimated_peak_memory_bytes),
            "tensor_core_candidate": (profile.tensor_core_candidate),
            "bfloat16_supported": (profile.bfloat16_supported),
            "tf32_supported": (profile.tf32_supported),
        },
    )


# ------------------------------------------------------------------
# Convenience inspection
# ------------------------------------------------------------------


def inspect_cuda_device(
    index: int = 0,
    *,
    backend: Optional[CUDABackend] = None,
    refresh_memory: bool = True,
) -> CUDADeviceProfile:
    return CUDADeviceInspector(backend=backend).inspect(
        index=index,
        refresh_memory=(refresh_memory),
    )


def _compat_torch():
    try:
        import torch as runtime
    except ImportError:
        return None
    return runtime


def is_cuda_available() -> bool:
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


def get_device_count() -> int:
    runtime = _compat_torch()
    if not is_cuda_available() or runtime is None:
        return 0
    return int(runtime.cuda.device_count())


def get_device(index: int = 0):
    runtime = _compat_torch()
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("Device index must be an integer.")
    if not is_cuda_available() or runtime is None:
        raise RuntimeError("CUDA accelerator is not available.")
    count = get_device_count()
    if index < 0 or index >= count:
        raise ValueError(f"Invalid CUDA device index {index}.")
    return runtime.device(f"cuda:{index}")


def get_device_info(index: int = 0) -> Dict[str, Any]:
    runtime = _compat_torch()
    get_device(index)
    properties = runtime.cuda.get_device_properties(index)
    major = getattr(properties, "major", None)
    minor = getattr(properties, "minor", None)
    return {
        "backend": "cuda",
        "index": index,
        "name": str(runtime.cuda.get_device_name(index)),
        "total_memory_bytes": int(getattr(properties, "total_memory", 0)),
        "compute_capability": (
            (int(major), int(minor))
            if major is not None and minor is not None
            else None
        ),
    }


def list_devices() -> list[Dict[str, Any]]:
    return [get_device_info(index) for index in range(get_device_count())]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "CUDADeviceProfile",
    "CUDAWorkloadSuitability",
    "require_cuda_runtime",
    "parse_compute_capability",
    "cuda_architecture",
    "cuda_tf32_supported",
    "cuda_tensor_core_candidate",
    "cuda_peer_access_devices",
    "compiled_cuda_architectures",
    "resolve_cuda_memory",
    "resolve_cuda_device",
    "CUDADeviceInspector",
    "evaluate_cuda_workload",
    "inspect_cuda_device",
    "is_cuda_available",
    "get_device_count",
    "get_device",
    "get_device_info",
    "list_devices",
]
