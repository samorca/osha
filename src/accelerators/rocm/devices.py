from __future__ import annotations

"""
AMD ROCm/HIP device inspection for OSHA.

OSHA logical identity:

    kind:
        rocm

    device_id:
        rocm:N

PyTorch HIP runtime identity:

    runtime_device:
        cuda:N

PyTorch intentionally exposes HIP devices through
torch.cuda. This module maintains the logical ROCm
identity while using the runtime representation
required by PyTorch.

Responsibilities:

    ROCm device resolution
    AMD architecture inspection
    VRAM inspection
    precision-capability inspection
    BF16 and TF32 inspection
    peer-access inspection
    workload suitability analysis
    optimization-profile construction

This module does not:

    select devices across backends
    allocate workload tensors
    clear caches
    synchronize streams
    change precision
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

from src.accelerators.optimization import (
    AcceleratorOptimizationProfile,
    AcceleratorWorkload,
)

from src.accelerators.rocm.backend import (
    ROCmBackend,
    require_rocm_index,
    rocm_architecture,
    rocm_bfloat16_supported,
    rocm_capabilities,
    rocm_device_id,
    rocm_memory,
    rocm_runtime_available,
    rocm_runtime_device_name,
    rocm_runtime_version,
    rocm_supported_precisions,
    torch_available,
    torch_runtime_version,
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
# ROCm device profile
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ROCmDeviceProfile:
    """
    Rich AMD ROCm/HIP device profile.

    AcceleratorDevice remains the canonical common
    hardware identity used by AcceleratorManager.
    """

    device: AcceleratorDevice

    index: int

    name: str

    runtime_device: str

    architecture: Optional[
        str
    ] = None

    architecture_family: Optional[
        str
    ] = None

    multiprocessor_count: Optional[
        int
    ] = None

    total_memory_bytes: Optional[
        int
    ] = None

    free_memory_bytes: Optional[
        int
    ] = None

    allocated_memory_bytes: Optional[
        int
    ] = None

    reserved_memory_bytes: Optional[
        int
    ] = None

    bfloat16_supported: bool = False

    tf32_supported: bool = False

    matrix_acceleration_candidate: bool = False

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

    rocm_runtime_version: Optional[
        str
    ] = None

    torch_runtime_version: Optional[
        str
    ] = None

    attributes: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        self.device.validate()

        if (
            self.device.kind
            != AcceleratorKind.ROCM.value
        ):
            raise ValueError(
                "ROCmDeviceProfile requires "
                "a ROCm AcceleratorDevice."
            )

        if self.index < 0:
            raise ValueError(
                "ROCm device index must be >= 0."
            )

        if (
            self.device.index is None
            or self.device.index != self.index
        ):
            raise ValueError(
                "ROCm profile index does not "
                "match the device index."
            )

        if not str(
            self.name
        ).strip():
            raise ValueError(
                "ROCm device name cannot be empty."
            )

        expected_device_id = rocm_device_id(
            self.index
        )

        if (
            self.device.device_id
            != expected_device_id
        ):
            raise ValueError(
                "ROCm logical device identity "
                "does not match the device index."
            )

        expected_runtime = (
            rocm_runtime_device_name(
                self.index
            )
        )

        if (
            self.runtime_device
            != expected_runtime
            or self.device.runtime_device
            != expected_runtime
        ):
            raise ValueError(
                "ROCm PyTorch runtime device "
                "must use cuda:N."
            )

        for field_name, value in (
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
            if (
                value is not None
                and value < 0
            ):
                raise ValueError(
                    f"{field_name} must be >= 0."
                )

        if (
            self.total_memory_bytes is not None
            and self.free_memory_bytes is not None
            and self.free_memory_bytes
            > self.total_memory_bytes
        ):
            raise ValueError(
                "ROCm free memory cannot exceed "
                "total memory."
            )

        valid_precisions = {
            item.value
            for item
            in AcceleratorPrecision
        }

        unsupported = [
            precision
            for precision
            in self.supported_precisions
            if precision
            not in valid_precisions
        ]

        if unsupported:
            raise ValueError(
                "ROCm profile contains unsupported "
                "precision identifiers: "
                + ", ".join(
                    unsupported
                )
            )

    @property
    def used_memory_bytes(
        self,
    ) -> Optional[int]:
        if (
            self.total_memory_bytes is None
            or self.free_memory_bytes is None
        ):
            return None

        return max(
            0,
            self.total_memory_bytes
            - self.free_memory_bytes,
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

        return (
            self.used_memory_bytes
            / self.total_memory_bytes
        )

    def supports_precision(
        self,
        precision: str,
    ) -> bool:
        return (
            str(
                precision
            ).strip().lower()
            in set(
                self.supported_precisions
            )
        )

    def supports_capability(
        self,
        capability: str,
    ) -> bool:
        return (
            str(
                capability
            ).strip()
            in set(
                self.capabilities
            )
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "device": (
                self.device.to_dict()
            ),

            "index": self.index,

            "name": self.name,

            "logical_device_id": (
                self.device.device_id
            ),

            "runtime_device": (
                self.runtime_device
            ),

            "architecture": (
                self.architecture
            ),

            "architecture_family": (
                self.architecture_family
            ),

            "multiprocessor_count": (
                self.multiprocessor_count
            ),

            "total_memory_bytes": (
                self.total_memory_bytes
            ),

            "free_memory_bytes": (
                self.free_memory_bytes
            ),

            "used_memory_bytes": (
                self.used_memory_bytes
            ),

            "allocated_memory_bytes": (
                self.allocated_memory_bytes
            ),

            "reserved_memory_bytes": (
                self.reserved_memory_bytes
            ),

            "memory_utilization_ratio": (
                self.memory_utilization_ratio
            ),

            "bfloat16_supported": (
                self.bfloat16_supported
            ),

            "tf32_supported": (
                self.tf32_supported
            ),

            "matrix_acceleration_candidate": (
                self.
                matrix_acceleration_candidate
            ),

            "peer_access_devices": list(
                self.peer_access_devices
            ),

            "supported_precisions": list(
                self.supported_precisions
            ),

            "capabilities": list(
                self.capabilities
            ),

            "rocm_runtime_version": (
                self.rocm_runtime_version
            ),

            "torch_runtime_version": (
                self.torch_runtime_version
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
# Runtime guard
# ------------------------------------------------------------------

def require_rocm_runtime() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is unavailable for "
            "ROCm device inspection."
        )

    if not rocm_runtime_version():
        raise RuntimeError(
            "The installed PyTorch runtime "
            "is not a ROCm/HIP build."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The ROCm/HIP runtime has no "
            "available accelerator device."
        )


# ------------------------------------------------------------------
# Architecture normalization
# ------------------------------------------------------------------

def normalize_rocm_architecture(
    value: Optional[str],
) -> Optional[str]:
    """
    Normalize a ROCm architecture such as:

        gfx942:sramecc+:xnack-

    to its core architecture identifier:

        gfx942
    """

    if value is None:
        return None

    text = str(
        value
    ).strip().lower()

    if not text:
        return None

    return text.split(
        ":",
        1,
    )[0]


def rocm_architecture_family(
    architecture: Optional[str],
) -> Optional[str]:
    """
    Return a broad AMD architecture family when
    the GFX identifier is known.

    Unknown architectures remain None instead of
    being guessed.
    """

    normalized = (
        normalize_rocm_architecture(
            architecture
        )
    )

    if normalized is None:
        return None

    cdna = {
        "gfx908",
        "gfx90a",
        "gfx940",
        "gfx941",
        "gfx942",
    }

    rdna = {
        "gfx1010",
        "gfx1011",
        "gfx1012",
        "gfx1030",
        "gfx1031",
        "gfx1032",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1150",
        "gfx1151",
        "gfx1200",
        "gfx1201",
    }

    if normalized in cdna:
        return "cdna"

    if normalized in rdna:
        return "rdna"

    return None


# ------------------------------------------------------------------
# TF32 support
# ------------------------------------------------------------------

def rocm_tf32_supported(
    index: int,
) -> bool:
    """
    Query runtime TF32 support when available.

    False means unsupported or unknown. This
    function does not enable TF32.
    """

    if torch is None:
        return False

    resolved = require_rocm_index(
        index
    )

    checker = getattr(
        torch.cuda,
        "is_tf32_supported",
        None,
    )

    if callable(
        checker
    ):
        try:
            with torch.cuda.device(
                resolved
            ):
                return bool(
                    checker()
                )

        except Exception:
            return False

    return False


# ------------------------------------------------------------------
# Matrix-acceleration candidate
# ------------------------------------------------------------------

def rocm_matrix_acceleration_candidate(
    index: int,
) -> bool:
    """
    Return whether the device is a candidate for
    accelerated matrix execution.

    This is deliberately conservative. It does not
    claim that a particular workload will use AMD
    matrix acceleration.
    """

    resolved = require_rocm_index(
        index
    )

    architecture = rocm_architecture(
        resolved
    )

    normalized = (
        normalize_rocm_architecture(
            architecture
        )
    )

    known_matrix_architectures = {
        "gfx908",
        "gfx90a",
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1150",
        "gfx1151",
        "gfx1200",
        "gfx1201",
    }

    return bool(
        normalized
        in known_matrix_architectures
    )


# ------------------------------------------------------------------
# Peer-access inspection
# ------------------------------------------------------------------

def rocm_peer_access_devices(
    index: int,
) -> Tuple[str, ...]:
    if torch is None:
        return ()

    resolved = require_rocm_index(
        index
    )

    checker = getattr(
        torch.cuda,
        "can_device_access_peer",
        None,
    )

    if not callable(
        checker
    ):
        return ()

    try:
        count = int(
            torch.cuda.device_count()
        )

    except Exception:
        return ()

    output = []

    for other in range(
        count
    ):
        if other == resolved:
            continue

        try:
            if bool(
                checker(
                    resolved,
                    other,
                )
            ):
                output.append(
                    rocm_device_id(
                        other
                    )
                )

        except Exception:
            continue

    return tuple(
        output
    )


# ------------------------------------------------------------------
# Runtime architecture list
# ------------------------------------------------------------------

def compiled_rocm_architectures() -> Tuple[str, ...]:
    """
    Return the architecture list exposed by the
    installed PyTorch HIP runtime when available.
    """

    if torch is None:
        return ()

    getter = getattr(
        torch.cuda,
        "get_arch_list",
        None,
    )

    if not callable(
        getter
    ):
        return ()

    try:
        values = getter()

        return tuple(
            str(
                value
            ).strip()
            for value in values
            if str(
                value
            ).strip()
        )

    except Exception:
        return ()


# ------------------------------------------------------------------
# Device memory
# ------------------------------------------------------------------

def resolve_rocm_memory(
    index: int,
) -> AcceleratorMemory:
    return rocm_memory(
        require_rocm_index(
            index
        )
    )


# ------------------------------------------------------------------
# Resolve one ROCm device
# ------------------------------------------------------------------

def resolve_rocm_device(
    index: int = 0,
    *,
    backend: Optional[
        ROCmBackend
    ] = None,
) -> AcceleratorDevice:
    runtime_backend = (
        backend
        or ROCmBackend()
    )

    resolved_index = require_rocm_index(
        index
    )

    expected_id = rocm_device_id(
        resolved_index
    )

    result = runtime_backend.discover()

    for device in result.devices:
        if (
            device.kind
            == AcceleratorKind.ROCM.value
            and device.device_id
            == expected_id
        ):
            return device

    return AcceleratorDevice(
        device_id=expected_id,

        kind=(
            AcceleratorKind.ROCM.value
        ),

        runtime_device=(
            rocm_runtime_device_name(
                resolved_index
            )
        ),

        index=resolved_index,

        name=(
            "ROCm device "
            + str(
                resolved_index
            )
        ),

        state=(
            AcceleratorState.
            UNAVAILABLE.value
        ),

        memory=AcceleratorMemory(),

        capabilities=(),

        supported_precisions=(
            AcceleratorPrecision.
            FLOAT32.value,
        ),

        vendor="AMD",

        architecture=None,

        runtime_version=(
            rocm_runtime_version()
        ),

        attributes={
            "resolved_without_available_backend": (
                True
            ),

            "pytorch_uses_cuda_interface": (
                True
            ),
        },
    )


# ------------------------------------------------------------------
# ROCm device inspector
# ------------------------------------------------------------------

class ROCmDeviceInspector:
    """
    Build a rich ROCmDeviceProfile.

    Inspection is read-only.
    """

    def __init__(
        self,
        *,
        backend: Optional[
            ROCmBackend
        ] = None,
    ) -> None:
        self.backend = (
            backend
            or ROCmBackend()
        )

    # ---------------------------------------------------------
    # Inspect
    # ---------------------------------------------------------

    def inspect(
        self,
        device: Optional[
            AcceleratorDevice
        ] = None,
        *,
        index: Optional[int] = None,
        refresh_memory: bool = True,
    ) -> ROCmDeviceProfile:
        if device is None:
            resolved = resolve_rocm_device(
                0
                if index is None
                else index,
                backend=self.backend,
            )

        else:
            resolved = device

        resolved.validate()

        if (
            resolved.kind
            != AcceleratorKind.ROCM.value
        ):
            raise ValueError(
                "ROCmDeviceInspector cannot "
                "inspect a non-ROCm device."
            )

        resolved_index = require_rocm_index(
            resolved.index
        )

        memory = resolved.memory
        memory_error_type = None

        if (
            refresh_memory
            and resolved.available
        ):
            try:
                memory = self.backend.memory(
                    resolved
                )

            except Exception as exc:
                memory_error_type = (
                    type(exc).__name__
                )

        architecture = (
            resolved.architecture
            or rocm_architecture(
                resolved_index
            )
        )

        properties = None
        property_error_type = None

        if (
            torch is not None
            and resolved.available
        ):
            try:
                properties = (
                    torch.cuda.
                    get_device_properties(
                        resolved_index
                    )
                )

            except Exception as exc:
                property_error_type = (
                    type(exc).__name__
                )

        multiprocessor_count = (
            resolved.
            multiprocessor_count
        )

        if (
            multiprocessor_count is None
            and properties is not None
        ):
            value = getattr(
                properties,
                "multi_processor_count",
                None,
            )

            if value is not None:
                multiprocessor_count = int(
                    value
                )

        name = (
            resolved.name
            or (
                "ROCm device "
                + str(
                    resolved_index
                )
            )
        )

        profile = ROCmDeviceProfile(
            device=resolved,

            index=resolved_index,

            name=name,

            runtime_device=(
                rocm_runtime_device_name(
                    resolved_index
                )
            ),

            architecture=(
                architecture
            ),

            architecture_family=(
                rocm_architecture_family(
                    architecture
                )
            ),

            multiprocessor_count=(
                multiprocessor_count
            ),

            total_memory_bytes=(
                memory.total_bytes
            ),

            free_memory_bytes=(
                memory.free_bytes
            ),

            allocated_memory_bytes=(
                memory.allocated_bytes
            ),

            reserved_memory_bytes=(
                memory.reserved_bytes
            ),

            bfloat16_supported=(
                rocm_bfloat16_supported(
                    resolved_index
                )
            ),

            tf32_supported=(
                rocm_tf32_supported(
                    resolved_index
                )
            ),

            matrix_acceleration_candidate=(
                rocm_matrix_acceleration_candidate(
                    resolved_index
                )
            ),

            peer_access_devices=(
                rocm_peer_access_devices(
                    resolved_index
                )
            ),

            supported_precisions=(
                resolved.
                supported_precisions
                or rocm_supported_precisions(
                    resolved_index
                )
            ),

            capabilities=(
                resolved.capabilities
                or rocm_capabilities(
                    resolved_index
                )
            ),

            rocm_runtime_version=(
                rocm_runtime_version()
            ),

            torch_runtime_version=(
                torch_runtime_version()
            ),

            attributes={
                "memory_error_type": (
                    memory_error_type
                ),

                "property_error_type": (
                    property_error_type
                ),

                "torch_available": (
                    torch_available()
                ),

                "rocm_runtime_available": (
                    rocm_runtime_available()
                ),

                "compiled_architectures": list(
                    compiled_rocm_architectures()
                ),

                "logical_device_id": (
                    resolved.device_id
                ),

                "pytorch_runtime_device": (
                    resolved.runtime_device
                ),

                "pytorch_uses_cuda_interface": (
                    True
                ),

                "vendor": (
                    resolved.vendor
                    or "AMD"
                ),

                "device_attributes": dict(
                    resolved.attributes
                ),
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
        device: Optional[
            AcceleratorDevice
        ] = None,
        index: Optional[int] = None,
        refresh_memory: bool = True,
    ) -> AcceleratorOptimizationProfile:
        workload.validate()

        profile = self.inspect(
            device,
            index=index,
            refresh_memory=(
                refresh_memory
            ),
        )

        utilization_percent = None

        if (
            profile.
            memory_utilization_ratio
            is not None
        ):
            utilization_percent = (
                profile.
                memory_utilization_ratio
                * 100.0
            )

        result = (
            AcceleratorOptimizationProfile(
                device=profile.device,

                workload=workload,

                free_memory_bytes=(
                    profile.
                    free_memory_bytes
                ),

                total_memory_bytes=(
                    profile.
                    total_memory_bytes
                ),

                utilization_percent=(
                    utilization_percent
                ),

                temperature_celsius=None,

                capabilities=(
                    profile.capabilities
                ),

                attributes={
                    "logical_device_id": (
                        profile.
                        device.device_id
                    ),

                    "runtime_device": (
                        profile.
                        runtime_device
                    ),

                    "architecture": (
                        profile.architecture
                    ),

                    "architecture_family": (
                        profile.
                        architecture_family
                    ),

                    "multiprocessor_count": (
                        profile.
                        multiprocessor_count
                    ),

                    "matrix_acceleration_candidate": (
                        profile.
                        matrix_acceleration_candidate
                    ),

                    "bfloat16_supported": (
                        profile.
                        bfloat16_supported
                    ),

                    "tf32_supported": (
                        profile.
                        tf32_supported
                    ),

                    "peer_access_devices": list(
                        profile.
                        peer_access_devices
                    ),

                    "supported_precisions": list(
                        profile.
                        supported_precisions
                    ),

                    "allocated_memory_bytes": (
                        profile.
                        allocated_memory_bytes
                    ),

                    "reserved_memory_bytes": (
                        profile.
                        reserved_memory_bytes
                    ),

                    "rocm_runtime_version": (
                        profile.
                        rocm_runtime_version
                    ),

                    "torch_runtime_version": (
                        profile.
                        torch_runtime_version
                    ),
                },
            )
        )

        result.validate()

        return result


# ------------------------------------------------------------------
# Workload suitability
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ROCmWorkloadSuitability:
    """
    Read-only ROCm workload suitability result.

    This is descriptive input for the ROCm
    optimizer. It does not execute the workload.
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
    ] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "suitable": (
                self.suitable
            ),

            "reasons": list(
                self.reasons
            ),

            "warnings": list(
                self.warnings
            ),

            "attributes": (
                to_json_safe(
                    dict(
                        self.attributes
                    )
                )
            ),
        }


def evaluate_rocm_workload(
    profile: ROCmDeviceProfile,
    workload: AcceleratorWorkload,
) -> ROCmWorkloadSuitability:
    """
    Evaluate explicitly known ROCm workload
    requirements.

    No synthetic throughput or performance claims
    are produced.
    """

    profile.validate()
    workload.validate()

    reasons = []
    warnings = []

    if not profile.device.available:
        reasons.append(
            "rocm_device_unavailable"
        )

    if not profile.supports_precision(
        workload.precision
    ):
        reasons.append(
            "workload_precision_unsupported"
        )

    if (
        workload.mixed_precision_enabled
        and not profile.supports_capability(
            AcceleratorCapability.
            MIXED_PRECISION.value
        )
    ):
        reasons.append(
            "mixed_precision_capability_missing"
        )

    if (
        workload.mixed_precision_enabled
        and workload.precision
        == AcceleratorPrecision.
        BFLOAT16.value
        and not profile.
        bfloat16_supported
    ):
        reasons.append(
            "bfloat16_unsupported"
        )

    if (
        workload.
        estimated_peak_memory_bytes
        is not None
    ):
        if (
            profile.free_memory_bytes
            is None
        ):
            warnings.append(
                "free_device_memory_unknown"
            )

        elif (
            workload.
            estimated_peak_memory_bytes
            > profile.
            free_memory_bytes
        ):
            reasons.append(
                "estimated_peak_memory_exceeds_free_vram"
            )

    if (
        workload.mixed_precision_enabled
        and not profile.
        matrix_acceleration_candidate
    ):
        warnings.append(
            "mixed_precision_matrix_acceleration_unknown"
        )

    if profile.architecture is None:
        warnings.append(
            "rocm_architecture_unknown"
        )

    return ROCmWorkloadSuitability(
        suitable=(
            not reasons
        ),

        reasons=tuple(
            reasons
        ),

        warnings=tuple(
            warnings
        ),

        attributes={
            "device_id": (
                profile.device.device_id
            ),

            "runtime_device": (
                profile.runtime_device
            ),

            "architecture": (
                profile.architecture
            ),

            "architecture_family": (
                profile.
                architecture_family
            ),

            "free_memory_bytes": (
                profile.free_memory_bytes
            ),

            "estimated_peak_memory_bytes": (
                workload.
                estimated_peak_memory_bytes
            ),

            "matrix_acceleration_candidate": (
                profile.
                matrix_acceleration_candidate
            ),

            "bfloat16_supported": (
                profile.
                bfloat16_supported
            ),

            "tf32_supported": (
                profile.
                tf32_supported
            ),
        },
    )


# ------------------------------------------------------------------
# Convenience inspection
# ------------------------------------------------------------------

def inspect_rocm_device(
    index: int = 0,
    *,
    backend: Optional[
        ROCmBackend
    ] = None,
    refresh_memory: bool = True,
) -> ROCmDeviceProfile:
    return ROCmDeviceInspector(
        backend=backend
    ).inspect(
        index=index,
        refresh_memory=(
            refresh_memory
        ),
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "ROCmDeviceProfile",
    "ROCmWorkloadSuitability",

    "require_rocm_runtime",

    "normalize_rocm_architecture",
    "rocm_architecture_family",

    "rocm_tf32_supported",
    "rocm_matrix_acceleration_candidate",
    "rocm_peer_access_devices",
    "compiled_rocm_architectures",

    "resolve_rocm_memory",
    "resolve_rocm_device",

    "ROCmDeviceInspector",

    "evaluate_rocm_workload",
    "inspect_rocm_device",
]
