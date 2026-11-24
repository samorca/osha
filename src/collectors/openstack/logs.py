from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.collectors.openstack.discovery import (
    OpenStackContext,
)
from src.collectors.openstack.schema import (
    error,
    record,
    unavailable,
)


OPENSTACK_UNIT_TOKENS = (
    "openstack",
    "nova",
    "neutron",
    "cinder",
    "glance",
    "keystone",
    "placement",
    "heat",
    "octavia",
    "swift",
    "ceilometer",
    "gnocchi",
    "aodh",
    "barbican",
    "ironic",
    "manila",
)

OPENSTACK_LOG_ROOTS = (
    Path("/var/log/openstack"),
    Path("/var/log/nova"),
    Path("/var/log/neutron"),
    Path("/var/log/cinder"),
    Path("/var/log/glance"),
    Path("/var/log/keystone"),
    Path("/var/log/placement"),
    Path("/var/log/heat"),
    Path("/var/log/octavia"),
    Path("/var/log/swift"),
    Path("/var/log/ceilometer"),
    Path("/var/log/gnocchi"),
    Path("/var/log/aodh"),
    Path("/var/log/barbican"),
    Path("/var/log/ironic"),
    Path("/var/log/manila"),
)

LOG_FILE_PATTERN = re.compile(
    r"\.log(?:\.\d+)?$|\.json$",
    re.IGNORECASE,
)

SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"""(?i)\b(password|passwd|secret|token|"""
    r"""api[_-]?key|client[_-]?secret|"""
    r"""application[_-]?credential)\b"""
    r"""(\s*[:=]\s*)"""
    r"""(?:"[^"]*"|'[^']*'|[^\s,;]+)"""
)

BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[^\s,;]+")

PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    re.IGNORECASE,
)

PRIORITY_NAMES = {
    "0": "emergency",
    "1": "alert",
    "2": "critical",
    "3": "error",
    "4": "warning",
    "5": "notice",
    "6": "info",
    "7": "debug",
}


def redact_log_message(message: str) -> str:
    if PRIVATE_KEY_PATTERN.search(message):
        return "[REDACTED_PRIVATE_KEY]"

    message = SECRET_ASSIGNMENT_PATTERN.sub(
        r"\1\2[REDACTED]",
        message,
    )

    return BEARER_PATTERN.sub(
        "Bearer [REDACTED]",
        message,
    )


def journal_timestamp(
    entry: dict[str, Any],
) -> str | None:
    raw = entry.get("__REALTIME_TIMESTAMP")

    if raw is None:
        return None

    try:
        microseconds = int(raw)
    except (TypeError, ValueError):
        return None

    return datetime.fromtimestamp(
        microseconds / 1_000_000,
        tz=timezone.utc,
    ).isoformat()


def run_command(
    command: list[str],
    *,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout="",
            stderr=(f"Command timed out after {timeout} seconds"),
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            args=command,
            returncode=127,
            stdout="",
            stderr=str(exc),
        )


def _failure_record(
    ctx: OpenStackContext,
    operation: str,
    message: str,
) -> dict:
    safe_message = redact_log_message(message)

    return error(
        "logs",
        RuntimeError(safe_message),
        cloud_id=ctx.cloud_id,
        region=ctx.region,
        environment_id=ctx.environment_id,
        operation=operation,
    )


def _openstack_units(
    ctx: OpenStackContext,
) -> tuple[list[str], list[dict]]:
    if shutil.which("systemctl") is None:
        return [], [
            unavailable(
                "logs",
                "systemctl is not installed",
                cloud_id=ctx.cloud_id,
                region=ctx.region,
                environment_id=ctx.environment_id,
            )
        ]

    result = run_command(
        [
            "systemctl",
            "list-units",
            "--type=service",
            "--all",
            "--no-legend",
            "--no-pager",
            "--plain",
        ]
    )

    if result.returncode != 0:
        return [], [
            _failure_record(
                ctx,
                "list_openstack_units",
                result.stderr.strip() or "Unable to list systemd services",
            )
        ]

    units = set()

    for line in result.stdout.splitlines():
        parts = line.split()

        if not parts:
            continue

        unit = parts[0]

        if unit in {"●", "○", "×", "↻"}:
            if len(parts) < 2:
                continue
            unit = parts[1]

        lowered = unit.lower()

        if unit.endswith(".service") and any(
            token in lowered for token in OPENSTACK_UNIT_TOKENS
        ):
            units.add(unit)

    return sorted(units), []


def collect_journal_unit(
    ctx: OpenStackContext,
    unit: str,
    *,
    lines: int,
    since: str | None,
) -> list[dict]:
    command = [
        "journalctl",
        "--no-pager",
        "--output=json",
        "--unit",
        unit,
        "--lines",
        str(lines),
    ]

    if since:
        command.extend(
            [
                "--since",
                since,
            ]
        )

    result = run_command(command)

    if result.returncode != 0:
        return [
            _failure_record(
                ctx,
                f"read_journal:{unit}",
                result.stderr.strip() or f"Unable to read journal for {unit}",
            )
        ]

    records = []

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            records.append(
                _failure_record(
                    ctx,
                    f"parse_journal:{unit}",
                    str(exc),
                )
            )
            continue

        if not isinstance(entry, dict):
            continue

        raw_message = entry.get("MESSAGE", "")
        message = raw_message if isinstance(raw_message, str) else str(raw_message)

        priority = str(entry.get("PRIORITY", ""))

        payload = {
            "systemd_unit": unit,
            "message": redact_log_message(message),
            "priority": priority,
            "severity": PRIORITY_NAMES.get(priority),
            "hostname": entry.get("_HOSTNAME"),
            "process_ess_name": entry.get("_COMM"),
            "executable": entry.get("_EXE"),
            "pid": entry.get("_PID"),
            "syslog_identifier": entry.get("SYSLOG_IDENTIFIER"),
            "transport": entry.get("_TRANSPORT"),
            "boot_id": entry.get("_BOOT_ID"),
        }

        records.append(
            record(
                "logs",
                "log",
                payload,
                cloud_id=ctx.cloud_id,
                region=ctx.region,
                environment_id=ctx.environment_id,
                resource_id=unit,
                timestamp=journal_timestamp(entry),
                source="journald",
            )
        )

    return records


def _log_files(
    maximum_files: int,
) -> list[Path]:
    files = set()

    for root in OPENSTACK_LOG_ROOTS:
        if not root.is_dir():
            continue

        try:
            for path in root.rglob("*"):
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or not LOG_FILE_PATTERN.search(path.name)
                ):
                    continue

                files.add(path)
        except OSError:
            continue

    return sorted(files)[:maximum_files]


def _read_tail(
    path: Path,
    line_count: int,
) -> list[str]:
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        return list(
            deque(
                handle,
                maxlen=line_count,
            )
        )


def collect_log_files(
    ctx: OpenStackContext,
    *,
    lines: int,
    maximum_files: int,
) -> list[dict]:
    records = []

    for path in _log_files(maximum_files):
        try:
            tail = _read_tail(
                path,
                lines,
            )
        except OSError as exc:
            records.append(
                _failure_record(
                    ctx,
                    f"read_log_file:{path}",
                    str(exc),
                )
            )
            continue

        for line in tail:
            message = line.rstrip("\r\n")

            if not message:
                continue

            records.append(
                record(
                    "logs",
                    "log",
                    {
                        "file_path": str(path),
                        "message": redact_log_message(message),
                    },
                    cloud_id=ctx.cloud_id,
                    region=ctx.region,
                    environment_id=ctx.environment_id,
                    resource_id=str(path),
                    source="log_file",
                )
            )

    return records


def collect(
    ctx: OpenStackContext,
    *,
    lines_per_unit: int = 200,
    lines_per_file: int = 200,
    maximum_files: int = 500,
    since: str | None = None,
) -> list[dict]:
    if lines_per_unit < 1:
        raise ValueError("lines_per_unit must be positive")

    if lines_per_file < 1:
        raise ValueError("lines_per_file must be positive")

    if maximum_files < 0:
        raise ValueError("maximum_files cannot be negative")

    records = []

    if shutil.which("journalctl") is None:
        records.append(
            unavailable(
                "logs",
                "journalctl is not installed",
                cloud_id=ctx.cloud_id,
                region=ctx.region,
                environment_id=ctx.environment_id,
            )
        )
    else:
        units, status_records = _openstack_units(ctx)
        records.extend(status_records)

        for unit in units:
            records.extend(
                collect_journal_unit(
                    ctx,
                    unit,
                    lines=lines_per_unit,
                    since=since,
                )
            )

    records.extend(
        collect_log_files(
            ctx,
            lines=lines_per_file,
            maximum_files=maximum_files,
        )
    )

    return records
