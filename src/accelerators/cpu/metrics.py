import time
import os
from typing import Any

try:
    import psutil
except (ImportError, ModuleNotFoundError):
    psutil = None


def get_cpu_metrics(interval: float = 0.1) -> dict[str, Any]:
    """Collect current CPU utilization information."""

    if psutil is None:
        return {
            "backend": "cpu",
            "utilization_percent": 0.0,
            "per_cpu_utilization_percent": [],
            "physical_cores": None,
            "logical_cores": os.cpu_count() or 1,
            "frequency_mhz": None,
            "load_average": (
                os.getloadavg()
                if hasattr(os, "getloadavg")
                else None
            ),
            "timestamp": time.time(),
        }

    overall_usage = psutil.cpu_percent(interval=interval)

    per_cpu_usage = psutil.cpu_percent(
        interval=None,
        percpu=True,
    )

    frequency = psutil.cpu_freq()
    load_average = (
        psutil.getloadavg()
        if hasattr(psutil, "getloadavg")
        else None
    )

    return {
        "backend": "cpu",
        "utilization_percent": overall_usage,
        "per_cpu_utilization_percent": per_cpu_usage,

        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),

        "frequency_mhz": (
            frequency.current
            if frequency is not None
            else None
        ),

        "load_average": load_average,

        "timestamp": time.time(),
    }


def get_memory_metrics() -> dict[str, Any]:
    """Collect host RAM usage."""

    if psutil is None:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf("SC_PHYS_PAGES")
            total = int(page_size * page_count)
        except (AttributeError, ValueError, OSError):
            total = 0
        return {
            "total_memory_bytes": total,
            "available_memory_bytes": total,
            "used_memory_bytes": 0,
            "memory_utilization_percent": 0.0,
            "total_memory_gb": round(total / (1024 ** 3), 2),
            "available_memory_gb": round(total / (1024 ** 3), 2),
            "used_memory_gb": 0.0,
        }

    memory = psutil.virtual_memory()

    return {
        "total_memory_bytes": memory.total,
        "available_memory_bytes": memory.available,
        "used_memory_bytes": memory.used,
        "memory_utilization_percent": memory.percent,

        "total_memory_gb": round(
            memory.total / (1024 ** 3), 2
        ),

        "available_memory_gb": round(
            memory.available / (1024 ** 3), 2
        ),

        "used_memory_gb": round(
            memory.used / (1024 ** 3), 2
        ),
    }


def get_swap_metrics() -> dict[str, Any]:
    """Collect swap-memory information."""

    if psutil is None:
        return {
            "total_swap_bytes": 0,
            "used_swap_bytes": 0,
            "free_swap_bytes": 0,
            "swap_utilization_percent": 0.0,
        }

    swap = psutil.swap_memory()

    return {
        "total_swap_bytes": swap.total,
        "used_swap_bytes": swap.used,
        "free_swap_bytes": swap.free,
        "swap_utilization_percent": swap.percent,
    }


def get_all_metrics() -> dict[str, Any]:
    """Return a complete CPU/host telemetry snapshot."""

    return {
        "cpu": get_cpu_metrics(),
        "memory": get_memory_metrics(),
        "swap": get_swap_metrics(),
    }