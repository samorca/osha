from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List

from src.collectors.linux.schema import (
    build_record,
)


PSEUDO_FILESYSTEMS = {
    "proc",
    "sysfs",
    "devtmpfs",
    "devpts",
    "tmpfs",
    "cgroup",
    "cgroup2",
    "pstore",
    "securityfs",
    "debugfs",
    "tracefs",
    "configfs",
    "fusectl",
    "mqueue",
    "hugetlbfs",
}


def run_command(
    command: List[str],
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30,
    )


def filesystem_usage(
    mountpoint: str,
) -> dict:
    try:
        stats = os.statvfs(
            mountpoint
        )
    except OSError:
        return {}

    block_size = (
        stats.f_frsize
        or stats.f_bsize
    )

    total = (
        stats.f_blocks
        * block_size
    )

    free = (
        stats.f_bfree
        * block_size
    )

    available = (
        stats.f_bavail
        * block_size
    )

    used = total - free

    return {
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
        "available_bytes": (
            available
        ),
        "usage_percent": (
            used / total * 100.0
            if total > 0
            else 0.0
        ),
        "inode_total": (
            stats.f_files
        ),
        "inode_free": (
            stats.f_ffree
        ),
    }


def collect_block_devices() -> List[dict]:
    if shutil.which("lsblk") is None:
        return []

    columns = (
        "NAME,KNAME,PATH,TYPE,SIZE,"
        "FSTYPE,FSVER,LABEL,UUID,"
        "MOUNTPOINTS,RO,RM,MODEL,"
        "SERIAL,TRAN"
    )

    result = run_command(
        [
            "lsblk",
            "--json",
            "--bytes",
            "--output",
            columns,
        ]
    )

    if result.returncode != 0:
        return []

    try:
        data = json.loads(
            result.stdout
        )
    except json.JSONDecodeError:
        return []

    records: List[dict] = []

    def visit(
        device: dict,
        parent: str | None = None,
    ) -> None:
        payload = {
            "name": device.get(
                "name"
            ),
            "kernel_name": (
                device.get("kname")
            ),
            "path": device.get(
                "path"
            ),
            "device_type": (
                device.get("type")
            ),
            "size_bytes": (
                device.get("size")
            ),
            "filesystem_type": (
                device.get("fstype")
            ),
            "filesystem_version": (
                device.get("fsver")
            ),
            "label": device.get(
                "label"
            ),
            "uuid": device.get(
                "uuid"
            ),
            "mountpoints": (
                device.get(
                    "mountpoints"
                )
                or []
            ),
            "read_only": (
                device.get("ro")
            ),
            "removable": (
                device.get("rm")
            ),
            "model": device.get(
                "model"
            ),
            "serial": device.get(
                "serial"
            ),
            "transport": (
                device.get("tran")
            ),
            "parent": parent,
        }

        records.append(
            build_record(
                signal_type=(
                    "block_device"
                ),
                payload=payload,
            )
        )

        for child in (
            device.get("children")
            or []
        ):
            visit(
                child,
                device.get("name"),
            )

    for device in (
        data.get("blockdevices")
        or []
    ):
        visit(device)

    return records


def collect_mounts() -> List[dict]:
    records: List[dict] = []

    path = Path(
        "/proc/self/mounts"
    )

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return records

    seen = set()

    for line in lines:
        parts = line.split()

        if len(parts) < 4:
            continue

        source = parts[0]
        mountpoint = (
            parts[1]
            .replace("\\040", " ")
        )
        filesystem_type = (
            parts[2]
        )
        options = (
            parts[3].split(",")
        )

        identity = (
            source,
            mountpoint,
            filesystem_type,
        )

        if identity in seen:
            continue

        seen.add(identity)

        usage = filesystem_usage(
            mountpoint
        )

        payload = {
            "source": source,
            "mountpoint": (
                mountpoint
            ),
            "filesystem_type": (
                filesystem_type
            ),
            "options": options,
            "read_only": (
                "ro" in options
            ),
            "pseudo_filesystem": (
                filesystem_type
                in PSEUDO_FILESYSTEMS
            ),
            **usage,
        }

        records.append(
            build_record(
                signal_type=(
                    "filesystem_state"
                ),
                payload=payload,
            )
        )

    return records


def collect_swap() -> List[dict]:
    path = Path(
        "/proc/swaps"
    )

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return []

    records: List[dict] = []

    for line in lines[1:]:
        parts = line.split()

        if len(parts) < 5:
            continue

        try:
            size = int(parts[2]) * 1024
            used = int(parts[3]) * 1024
            priority = int(parts[4])
        except ValueError:
            continue

        records.append(
            build_record(
                signal_type="swap_state",
                payload={
                    "name": parts[0],
                    "type": parts[1],
                    "size_bytes": size,
                    "used_bytes": used,
                    "priority": priority,
                },
            )
        )

    return records


def collect_storage() -> List[dict]:
    return (
        collect_block_devices()
        + collect_mounts()
        + collect_swap()
    )


if __name__ == "__main__":
    import json

    for record in collect_storage():
        print(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )
