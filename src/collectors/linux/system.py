from __future__ import annotations

import os
import platform
import socket
from pathlib import Path

from src.collectors.linux.schema import (
    build_record,
)


def read_os_release() -> dict[str, str]:
    path = Path(
        "/etc/os-release"
    )

    result: dict[str, str] = {}

    if not path.is_file():
        return result

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            line = line.strip()

            if (
                not line
                or line.startswith("#")
                or "=" not in line
            ):
                continue

            key, value = line.split(
                "=",
                1,
            )

            result[key] = (
                value.strip().strip(
                    '"'
                )
            )

    return result


def machine_id() -> str | None:
    paths = (
        Path("/etc/machine-id"),
        Path(
            "/var/lib/dbus/machine-id"
        ),
    )

    for path in paths:
        try:
            value = (
                path.read_text(
                    encoding="utf-8"
                )
                .strip()
            )

            if value:
                return value
        except OSError:
            continue

    return None


def collect_system_identity() -> dict:
    os_release = read_os_release()

    payload = {
        "machine_id": machine_id(),
        "hostname": (
            socket.gethostname()
        ),
        "fqdn": socket.getfqdn(),
        "os_name": os_release.get(
            "NAME"
        ),
        "os_id": os_release.get(
            "ID"
        ),
        "os_version": os_release.get(
            "VERSION_ID"
        ),
        "os_pretty_name": (
            os_release.get(
                "PRETTY_NAME"
            )
        ),
        "kernel_system": (
            platform.system()
        ),
        "kernel_release": (
            platform.release()
        ),
        "kernel_version": (
            platform.version()
        ),
        "architecture": (
            platform.machine()
        ),
        "processor": (
            platform.processor()
        ),
        "cpu_count": (
            os.cpu_count()
        ),
    }

    return build_record(
        signal_type="resource_state",
        payload=payload,
    )


def collect_interfaces() -> dict:
    addresses = []

    try:
        information = socket.getaddrinfo(
            socket.gethostname(),
            None,
        )

        seen = set()

        for item in information:
            address = item[4][0]

            if address in seen:
                continue

            seen.add(address)

            addresses.append(
                address
            )
    except OSError:
        pass

    return build_record(
        signal_type="network_state",
        payload={
            "addresses": addresses,
        },
    )
