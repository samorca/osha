from __future__ import annotations

from typing import Any, Optional

from src.accelerators.base import (
    ROCmAcceleratorBackend,
)
from src.accelerators.config import (
    AcceleratorConfig,
)
from src.accelerators.schema import (
    AcceleratorDevice,
    AcceleratorKind,
    AcceleratorMemory,
    AcceleratorPrecision,
    AcceleratorState,
)


def _torch():
    try:
        import torch as runtime
    except ImportError:
        return None
    return runtime


def torch_available() -> bool:
    return _torch() is not None


def torch_runtime_version() -> Optional[str]:
    runtime = _torch()
    value = getattr(runtime, "__version__", None) if runtime else None
    return str(value) if value is not None else None


def rocm_runtime_available() -> bool:
    runtime = _torch()
    version = getattr(runtime, "version", None) if runtime else None
    cuda = getattr(runtime, "cuda", None) if runtime else None
    available = getattr(cuda, "is_available", None)
    return bool(
        runtime is not None
        and getattr(version, "hip", None)
        and callable(available)
        and available()
    )


def rocm_runtime_version() -> Optional[str]:
    runtime = _torch()
    version = getattr(runtime, "version", None) if runtime else None
    value = getattr(version, "hip", None)
    return str(value) if value is not None else None


def require_rocm_index(index: int) -> int:
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("Device index must be an integer.")
    runtime = _torch()
    count = (
        int(runtime.cuda.device_count())
        if runtime is not None and rocm_runtime_available()
        else 0
    )
    if index < 0 or index >= count:
        raise ValueError(f"Invalid ROCm device index {index}.")
    return index


def rocm_device_id(index: int) -> str:
    return f"rocm:{index}"


def rocm_runtime_device_name(index: int) -> str:
    runtime = _torch()
    if runtime is None:
        raise RuntimeError("ROCm runtime is unavailable.")
    require_rocm_index(index)
    return str(runtime.cuda.get_device_name(index))


def rocm_architecture(index: int) -> Optional[str]:
    runtime = _torch()
    if runtime is None:
        return None
    require_rocm_index(index)
    properties = runtime.cuda.get_device_properties(index)
    value = getattr(
        properties,
        "gcnArchName",
        getattr(properties, "gcn_arch_name", None),
    )
    return str(value) if value else None


def rocm_bfloat16_supported(index: int = 0) -> bool:
    runtime = _torch()
    checker = getattr(getattr(runtime, "cuda", None), "is_bf16_supported", None)
    return bool(checker(index)) if callable(checker) else False


def rocm_capabilities(index: int = 0) -> tuple[str, ...]:
    return ("gpu", "rocm") if rocm_runtime_available() else ()


def rocm_supported_precisions(index: int = 0) -> tuple[str, ...]:
    values = ["float32"]
    if rocm_bfloat16_supported(index):
        values.append("bfloat16")
    return tuple(values)


def rocm_memory(index: int = 0) -> AcceleratorMemory:
    runtime = _torch()
    if runtime is None or not rocm_runtime_available():
        return AcceleratorMemory()
    require_rocm_index(index)
    free, total = runtime.cuda.mem_get_info(index)
    return AcceleratorMemory(
        total_bytes=int(total),
        free_bytes=int(free),
        used_bytes=max(0, int(total) - int(free)),
    )


class ROCmAccelerator:
    name = "rocm"

    def is_available(self) -> bool:
        return rocm_runtime_available()

    def device_count(self) -> int:
        runtime = _torch()
        if not self.is_available() or runtime is None:
            return 0
        return int(runtime.cuda.device_count())

    def get_device(self, index: int = 0):
        if not self.is_available():
            raise RuntimeError("ROCm accelerator is not available.")
        require_rocm_index(index)
        return _torch().device(f"cuda:{index}")

    def device_name(self, index: int = 0) -> str:
        return rocm_runtime_device_name(index)


class ROCmBackend(ROCmAcceleratorBackend):
    def __init__(
        self,
        *,
        config: Optional[AcceleratorConfig] = None,
    ) -> None:
        super().__init__(config=config)

    def available(self) -> bool:
        try:
            runtime = _torch()
            return bool(
                rocm_runtime_available()
                and runtime is not None
                and int(runtime.cuda.device_count()) > 0
            )
        except Exception:
            return False

    def _discover(self) -> tuple[AcceleratorDevice, ...]:
        runtime = _torch()
        if not self.available() or runtime is None:
            return ()

        devices = []
        for index in range(int(runtime.cuda.device_count())):
            properties = runtime.cuda.get_device_properties(index)
            device = AcceleratorDevice(
                device_id=rocm_device_id(index),
                kind=AcceleratorKind.ROCM.value,
                runtime_device=f"cuda:{index}",
                index=index,
                name=rocm_runtime_device_name(index),
                state=AcceleratorState.AVAILABLE.value,
                memory=rocm_memory(index),
                capabilities=rocm_capabilities(index),
                supported_precisions=rocm_supported_precisions(index),
                vendor="AMD",
                architecture=rocm_architecture(index),
                runtime_version=rocm_runtime_version(),
                multiprocessor_count=getattr(
                    properties,
                    "multi_processor_count",
                    None,
                ),
            )
            device.validate()
            devices.append(device)
        return tuple(devices)

    def _memory(
        self,
        device: AcceleratorDevice,
    ) -> AcceleratorMemory:
        return rocm_memory(device.index or 0)

    def _runtime_device(self, device: AcceleratorDevice) -> Any:
        runtime = _torch()
        if runtime is None:
            raise RuntimeError("ROCm runtime is unavailable.")
        return runtime.device(device.runtime_device)
