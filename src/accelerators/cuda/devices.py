from typing import Any


def _torch():
    try:
        import torch
        return torch
    except (ImportError, ModuleNotFoundError):
        return None


def is_cuda_available() -> bool:
    """Return True only when NVIDIA CUDA is available."""

    torch = _torch()
    if torch is None:
        return False
    cuda = getattr(torch, "cuda", None)
    version = getattr(torch, "version", None)
    return bool(
        cuda
        and callable(getattr(cuda, "is_available", None))
        and getattr(version, "cuda", None) is not None
        and getattr(version, "hip", None) is None
        and cuda.is_available()
    )


def get_device_count() -> int:
    if not is_cuda_available():
        return 0

    torch = _torch()
    return torch.cuda.device_count() if torch is not None else 0


def get_device(index: int = 0):
    count = get_device_count()

    if count == 0:
        raise RuntimeError("No CUDA devices are available.")

    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("Device index must be an integer.")
    if index < 0 or index >= count:
        raise ValueError(
            f"Invalid CUDA device index {index}. "
            f"Available devices: 0-{count - 1}"
        )

    return _torch().device(f"cuda:{index}")


def get_device_info(index: int = 0) -> dict[str, Any]:
    device = get_device(index)
    properties = _torch().cuda.get_device_properties(device)

    return {
        "index": index,
        "backend": "cuda",
        "device": str(device),
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "total_memory_gb": round(
            properties.total_memory / (1024 ** 3), 2
        ),
        "compute_capability": (
            properties.major,
            properties.minor,
        ),
    }


def list_devices() -> list[dict[str, Any]]:
    return [
        get_device_info(index)
        for index in range(get_device_count())
    ]