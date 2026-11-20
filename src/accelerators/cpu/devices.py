import os
import platform
from typing import Any

def _torch():
    try:
        import torch
        return torch
    except (ImportError, ModuleNotFoundError):
        return None


def is_cpu_available() -> bool:
    """CPU is always available as the fallback accelerator."""
    return True


def get_device_count() -> int:
    """
    Return the number of logical CPU processors visible
    to the current process.
    """
    return os.cpu_count() or 1


def get_device(index: int = 0):
    """
    Return the PyTorch CPU device.

    CPUs are not addressed like cuda:0, cuda:1, etc.
    PyTorch uses a single logical 'cpu' device.
    """
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("Device index must be an integer.")
    if index != 0:
        raise ValueError(
            f"CPU device index {index} is invalid. Available devices: 0"
        )
    torch = _torch()
    return torch.device("cpu") if torch is not None else "cpu"


def get_device_info() -> dict[str, Any]:
    """Return information about the host CPU."""

    logical_cores = os.cpu_count() or 1
    torch = _torch()

    return {
        "backend": "cpu",
        "device": "cpu",
        "processor": platform.processor() or "unknown",
        "architecture": platform.machine(),
        "logical_cores": logical_cores,
        "pytorch_threads": torch.get_num_threads() if torch is not None else None,
        "pytorch_interop_threads": (
            torch.get_num_interop_threads() if torch is not None else None
        ),
    }


def list_devices() -> list[dict[str, Any]]:
    """
    CPU is represented as one PyTorch compute device,
    even when it contains multiple cores.
    """
    return [get_device_info()]