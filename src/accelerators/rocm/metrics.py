from typing import Any


def _torch():
    try:
        import torch
        return torch
    except (ImportError, ModuleNotFoundError):
        return None

from .devices import get_device_count, get_device


def get_memory_metrics(index: int = 0) -> dict[str, Any]:
    device = get_device(index)
    torch = _torch()
    if torch is None:
        raise RuntimeError("PyTorch is required to collect ROCm metrics.")

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)

    used_bytes = total_bytes - free_bytes

    utilization = (
        (used_bytes / total_bytes) * 100
        if total_bytes > 0
        else 0.0
    )

    return {
        "index": index,
        "backend": "rocm",

        "total_memory_bytes": total_bytes,
        "free_memory_bytes": free_bytes,
        "used_memory_bytes": used_bytes,

        "allocated_by_pytorch_bytes": allocated_bytes,
        "reserved_by_pytorch_bytes": reserved_bytes,

        "memory_utilization_percent": round(utilization, 2),
    }


def get_all_memory_metrics() -> list[dict[str, Any]]:
    return [
        get_memory_metrics(index)
        for index in range(get_device_count())
    ]