from __future__ import annotations

import os
from pathlib import Path

from src.collectors.linux.schema import (
    build_record,
)


def read_meminfo() -> dict[str, int]:
    path = Path(
        "/proc/meminfo"
    )

    result: dict[str, int] = {}

    if not path.is_file():
        return result

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            if ":" not in line:
                continue

            key, value = line.split(
                ":",
                1,
            )

            parts = (
                value.strip().split()
            )

            if not parts:
                continue

            try:
                number = int(parts[0])
            except ValueError:
                continue

            # /proc/meminfo values are
            # normally reported in kB.
            if (
                len(parts) > 1
                and parts[1].lower()
                == "kb"
            ):
                number *= 1024

            result[key] = number

    return result


def read_load_average() -> dict:
    try:
        one, five, fifteen = (
            os.getloadavg()
        )
    except OSError:
        return {}

    return {
        "load_1m": one,
        "load_5m": five,
        "load_15m": fifteen,
    }


def read_uptime() -> float | None:
    path = Path(
        "/proc/uptime"
    )

    try:
        first = (
            path.read_text(
                encoding="utf-8"
            )
            .split()[0]
        )

        return float(first)
    except (
        OSError,
        ValueError,
        IndexError,
    ):
        return None


def filesystem_usage(
    path: str,
) -> dict | None:
    try:
        stats = os.statvfs(path)
    except OSError:
        return None

    block_size = (
        stats.f_frsize
        or stats.f_bsize
    )

    total = (
        stats.f_blocks
        * block_size
    )

    available = (
        stats.f_bavail
        * block_size
    )

    used = total - (
        stats.f_bfree
        * block_size
    )

    usage_percent = (
        (used / total) * 100.0
        if total > 0
        else 0.0
    )

    return {
        "mount": path,
        "total_bytes": total,
        "used_bytes": used,
        "available_bytes": (
            available
        ),
        "usage_percent": (
            usage_percent
        ),
    }


def collect_metrics() -> dict:
    memory = read_meminfo()

    total_memory = memory.get(
        "MemTotal"
    )

    available_memory = (
        memory.get(
            "MemAvailable"
        )
    )

    used_memory = None
    memory_usage_percent = None

    if (
        total_memory is not None
        and available_memory
        is not None
    ):
        used_memory = (
            total_memory
            - available_memory
        )

        if total_memory > 0:
            memory_usage_percent = (
                used_memory
                / total_memory
                * 100.0
            )

    root_filesystem = (
        filesystem_usage("/")
    )

    payload = {
        **read_load_average(),
        "uptime_seconds": (
            read_uptime()
        ),
        "memory_total_bytes": (
            total_memory
        ),
        "memory_available_bytes": (
            available_memory
        ),
        "memory_used_bytes": (
            used_memory
        ),
        "memory_usage_percent": (
            memory_usage_percent
        ),
        "swap_total_bytes": (
            memory.get(
                "SwapTotal"
            )
         ),
        "swap_free_bytes": (
            memory.get(
                "SwapFree"
            )
        ),
        "root_filesystem": (
            root_filesystem
        ),
    }

    return build_record(
        signal_type="metric",
        payload=payload,
    )
