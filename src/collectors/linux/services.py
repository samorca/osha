from __future__ import annotations

import shutil
import subprocess
from typing import Dict, List, Optional

from src.collectors.linux.schema import (
    build_record,
)


SERVICE_PROPERTIES = (
    "Id",
    "Names",
    "Description",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "UnitFilePreset",
    "MainPID",
    "ControlPID",
    "ExecMainPID",
    "ExecMainCode",
    "ExecMainStatus",
    "Restart",
    "NRestarts",
    "Type",
    "FragmentPath",
    "SourcePath",
    "DropInPaths",
    "Requires",
    "Wants",
    "After",
    "Before",
    "TriggeredBy",
    "Triggers",
    "MemoryCurrent",
    "MemoryPeak",
    "TasksCurrent",
    "CPUUsageNSec",
)


def command_available(
    command: str,
) -> bool:
    return shutil.which(command) is not None


def run_command(
    command: List[str],
    *,
    timeout: int = 20,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout="",
            stderr=(f"Command timed out after {timeout} seconds"),
        )


def parse_key_value_output(
    text: str,
) -> Dict[str, str]:
    output: Dict[str, str] = {}

    for line in text.splitlines():
        if "=" not in line:
            continue

        key, value = line.split(
            "=",
            1,
        )

        output[key.strip()] = value.strip()

    return output


def optional_int(
    value: Optional[str],
) -> Optional[int]:
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def split_systemd_list(
    value: Optional[str],
) -> List[str]:
    if not value:
        return []

    return [item for item in value.split() if item]


def list_loaded_services() -> List[str]:
    """
    Return service units currently loaded into systemd.

    --all includes inactive and failed loaded units as well
    as active services.
    """

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
        return []

    services: List[str] = []

    for line in result.stdout.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if not parts:
            continue

        unit = parts[0]

        # Some systemctl versions can prefix problematic
        # units with a marker.
        if unit in {
            "●",
            "○",
            "×",
            "↻",
        }:
            if len(parts) < 2:
                continue

            unit = parts[1]

        if unit.endswith(".service"):
            services.append(unit)

    return sorted(set(services))


def list_installed_services() -> Dict[str, str]:
    """
    Return installed service unit files and their
    enablement state.

    Examples of states include enabled, disabled,
    static and masked.
    """

    result = run_command(
        [
            "systemctl",
            "list-unit-files",
            "--type=service",
            "--no-legend",
            "--no-pager",
            "--plain",
        ]
    )

    if result.returncode != 0:
        return {}

    services: Dict[str, str] = {}

    for line in result.stdout.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        unit = parts[0]
        state = parts[1]

        if not unit.endswith(".service"):
            continue

        services[unit] = state

    return services


def get_service_properties(
    service_name: str,
) -> Dict[str, str]:
    command = [
        "systemctl",
        "show",
        service_name,
        "--no-pager",
    ]

    for property_name in SERVICE_PROPERTIES:
        command.extend(
            [
                "--property",
                property_name,
            ]
        )

    result = run_command(command)

    if result.returncode != 0:
        return {}

    return parse_key_value_output(result.stdout)


def normalize_service(
    service_name: str,
    properties: Dict[str, str],
    installed_state: Optional[str],
) -> dict:
    unit_file_state = properties.get("UnitFileState") or installed_state

    active_state = properties.get("ActiveState")

    sub_state = properties.get("SubState")

    load_state = properties.get("LoadState")

    service_status = active_state or "unknown"

    healthy = None

    if active_state == "failed":
        healthy = False
    elif active_state == "active":
        healthy = True

    payload = {
        "service_name": (properties.get("Id") or service_name),
        "names": split_systemd_list(properties.get("Names")),
        "description": (properties.get("Description")),
        "load_state": load_state,
        "active_state": (active_state),
        "sub_state": sub_state,
        "service_status": (service_status),
        "healthy": healthy,
        "unit_file_state": (unit_file_state),
        "unit_file_preset": (properties.get("UnitFilePreset")),
        "service_type": (properties.get("Type")),
        "main_pid": optional_int(properties.get("MainPID")),
        "control_pid": optional_int(properties.get("ControlPID")),
        "exec_main_pid": optional_int(properties.get("ExecMainPID")),
        "exec_main_code": optional_int(properties.get("ExecMainCode")),
        "exec_main_status": (optional_int(properties.get("ExecMainStatus"))),
        "restart_policy": (properties.get("Restart")),
        "restart_count": (optional_int(properties.get("NRestarts"))),
        "unit_file": (properties.get("FragmentPath") or None),
        "source_path": (properties.get("SourcePath") or None),
        "drop_in_paths": (split_systemd_list(properties.get("DropInPaths"))),
        "requires": (split_systemd_list(properties.get("Requires"))),
        "wants": (split_systemd_list(properties.get("Wants"))),
        "after": (split_systemd_list(properties.get("After"))),
        "before": (split_systemd_list(properties.get("Before"))),
        "triggered_by": (split_systemd_list(properties.get("TriggeredBy"))),
        "triggers": (split_systemd_list(properties.get("Triggers"))),
        "memory_current_bytes": (optional_int(properties.get("MemoryCurrent"))),
        "memory_peak_bytes": (optional_int(properties.get("MemoryPeak"))),
        "tasks_current": (optional_int(properties.get("TasksCurrent"))),
        "cpu_usage_nsec": (optional_int(properties.get("CPUUsageNSec"))),
    }

    return build_record(
        signal_type="service_state",
        payload=payload,
    )


def collector_error(
    *,
    error: str,
    message: str,
) -> dict:
    return build_record(
        signal_type="collector_error",
        payload={
            "collector": ("linux_services"),
            "error": error,
            "message": message,
        },
    )


def collect_services() -> List[dict]:
    """
    Discover Linux systemd services.

    The collector is read-only. It does not start,
    stop, enable, disable or restart services.
    """

    if not command_available("systemctl"):
        return [
            collector_error(
                error=("systemctl_not_found"),
                message=("systemctl is not available on this host"),
            )
        ]

    installed = list_installed_services()

    loaded = set(list_loaded_services())

    all_services = sorted(loaded | set(installed.keys()))

    if not all_services:
        return [
            collector_error(
                error=("services_not_discovered"),
                message=("No systemd services were discovered"),
            )
        ]

    records: List[dict] = []

    for service_name in all_services:
        properties = get_service_properties(service_name)

        if not properties:
            records.append(
                collector_error(
                    error=("service_inspection_failed"),
                    message=(f"Unable to inspect {service_name}"),
                )
            )

            continue

        records.append(
            normalize_service(
                service_name,
                properties,
                installed.get(service_name),
            )
        )

    return records


if __name__ == "__main__":
    import json

    records = collect_services()

    for record in records:
        print(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )
