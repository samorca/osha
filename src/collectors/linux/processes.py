from __future__ import annotations

import os
import pwd
from pathlib import Path
from typing import Dict, List, Optional

from src.collectors.linux.schema import (
    build_record,
)


PROCESS_STATES = {
    "R": "running",
    "S": "sleeping",
    "D": "disk_sleep",
    "Z": "zombie",
    "T": "stopped",
    "t": "tracing_stop",
    "X": "dead",
    "x": "dead",
    "I": "idle",
    "P": "parked",
}


def safe_read_text(
    path: Path,
    *,
    limit: int = 65536,
) -> Optional[str]:
    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as handle:
            return handle.read(
                limit
            )
    except (
        OSError,
        PermissionError,
    ):
        return None


def safe_read_link(
    path: Path,
) -> Optional[str]:
    try:
        return os.readlink(path)
    except (
        OSError,
        PermissionError,
    ):
        return None


def parse_status(
    text: str,
) -> Dict[str, str]:
    output: Dict[str, str] = {}

    for line in text.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(
            ":",
            1,
        )

        output[key.strip()] = (
            value.strip()
        )

    return output


def first_integer(
    value: Optional[str],
) -> Optional[int]:
    if not value:
        return None

    for part in value.split():
        try:
            return int(part)
        except ValueError:
            continue

    return None


def memory_bytes(
    value: Optional[str],
) -> Optional[int]:
    number = first_integer(value)

    if number is None:
        return None

    if value and "kb" in value.lower():
        return number * 1024

    return number


def user_name(
    uid: Optional[int],
) -> Optional[str]:
    if uid is None:
        return None

    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return None


def sanitize_command_line(
    raw: Optional[bytes],
    executable: Optional[str],
    name: Optional[str],
) -> List[str]:
    """
    Keep only the executable/program name.

    Full process arguments can contain passwords,
    tokens and connection strings, so they are not
    collected by default.
    """

    if executable:
        return [executable]

    if raw:
        parts = [
            part.decode(
                "utf-8",
                errors="replace",
            )
            for part in raw.split(b"\0")
            if part
        ]

        if parts:
            return [
                Path(parts[0]).name
            ]

    if name:
        return [name]

    return []


def read_cmdline(
    path: Path,
) -> Optional[bytes]:
    try:
        return path.read_bytes()[
            :65536
        ]
    except (
        OSError,
        PermissionError,
    ):
        return None


def parse_cgroup(
    text: Optional[str],
) -> List[str]:
    if not text:
        return []

    output: List[str] = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split(
            ":",
            2,
        )

        if len(parts) == 3:
            output.append(parts[2])

    return output


def collect_process(
    pid: int,
) -> Optional[dict]:
    base = Path("/proc") / str(pid)

    status_text = safe_read_text(
        base / "status"
    )

    stat_text = safe_read_text(
        base / "stat"
    )

    if not status_text:
        return None

    status = parse_status(
        status_text
    )

    name = status.get("Name")

    state_code = None
    parent_pid = first_integer(
        status.get("PPid")
    )

    if stat_text:
        closing = stat_text.rfind(")")

        if closing >= 0:
            remainder = (
                stat_text[
                    closing + 2:
                ].split()
            )

            if remainder:
                state_code = remainder[0]

            if (
                len(remainder) > 1
                and parent_pid is None
            ):
                try:
                    parent_pid = int(
                        remainder[1]
                    )
                except ValueError:
                    pass

    uid = first_integer(
        status.get("Uid")
    )

    gid = first_integer(
        status.get("Gid")
    )

    executable = safe_read_link(
        base / "exe"
    )

    cmdline = sanitize_command_line(
        read_cmdline(
            base / "cmdline"
        ),
        executable,
        name,
    )

    cgroups = parse_cgroup(
        safe_read_text(
            base / "cgroup"
        )
    )

    container_hint = None

    for cgroup in cgroups:
        lowered = cgroup.lower()

        if any(
            marker in lowered
            for marker in (
                "docker",
                "containerd",
                "kubepods",
                "libpod",
            )
        ):
            container_hint = cgroup
            break

    payload = {
        "pid": pid,
        "parent_pid": parent_pid,
        "name": name,
        "state_code": state_code,
        "state": PROCESS_STATES.get(
            state_code,
            "unknown",
        ),
        "uid": uid,
        "gid": gid,
        "user": user_name(uid),
        "executable": executable,
        "command_identity": cmdline,
        "threads": first_integer(
            status.get("Threads")
        ),
        "resident_memory_bytes": memory_bytes(
            status.get("VmRSS")
        ),
        "virtual_memory_bytes": memory_bytes(
            status.get("VmSize")
        ),
        "swap_bytes": memory_bytes(
            status.get("VmSwap")
        ),
        "cgroups": cgroups,
        "container_hint": container_hint,
    }

    return build_record(
        signal_type="process_state",
        payload=payload,
    )


def collect_processes() -> List[dict]:
    records: List[dict] = []

    proc = Path("/proc")

    for item in proc.iterdir():
        if not (
            item.is_dir()
            and item.name.isdigit()
        ):
            continue

        try:
            pid = int(item.name)
        except ValueError:
            continue

        record = collect_process(pid)

        if record is not None:
            records.append(record)

    records.sort(
        key=lambda record: (
            record["payload"].get(
                "pid",
                0,
            )
        )
    )

    return records


if __name__ == "__main__":
    import json

    for record in collect_processes():
        print(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )
