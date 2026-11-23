from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import List

from src.collectors.linux.schema import (
    build_record,
)


def run_json_command(
    command: List[str],
) -> list:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return []

    if result.returncode != 0:
        return []

    try:
        value = json.loads(
            result.stdout
        )
    except json.JSONDecodeError:
        return []

    return (
        value
        if isinstance(value, list)
        else []
    )


def collect_interfaces() -> List[dict]:
    if shutil.which("ip") is None:
        return [
            build_record(
                signal_type=(
                    "collector_error"
                ),
                payload={
                    "collector": (
                        "linux_network"
                    ),
                    "error": "ip_not_found",
                },
            )
        ]

    interfaces = run_json_command(
        [
            "ip",
            "-j",
            "-s",
            "address",
            "show",
        ]
    )

    records: List[dict] = []

    for interface in interfaces:
        addresses = []

        for address in (
            interface.get(
                "addr_info"
            )
            or []
        ):
            addresses.append(
                {
                    "family": (
                        address.get(
                            "family"
                        )
                    ),
                    "local": (
                        address.get(
                            "local"
                        )
                    ),
                    "prefix_length": (
                        address.get(
                            "prefixlen"
                        )
                    ),
                    "scope": (
                        address.get(
                            "scope"
                        )
                    ),
                    "label": (
                        address.get(
                            "label"
                        )
                    ),
                    "valid_lifetime": (
                        address.get(
                            "valid_life_time"
                        )
                    ),
                    "preferred_lifetime": (
                        address.get(
                            "preferred_life_time"
                        )
                    ),
                }
            )

        payload = {
            "interface_index": (
                interface.get(
                    "ifindex"
                )
            ),
            "interface_name": (
                interface.get(
                    "ifname"
                )
            ),
            "flags": interface.get(
                "flags",
                [],
            ),
            "mtu": interface.get(
                "mtu"
            ),
            "operational_state": (
                interface.get(
                    "operstate"
                )
            ),
            "link_type": (
                interface.get(
                    "link_type"
                )
            ),
            "mac_address": (
                interface.get(
                    "address"
                )
            ),
            "broadcast_address": (
                interface.get(
                    "broadcast"
                )
            ),
            "addresses": addresses,
            "statistics": (
                interface.get(
                    "stats64"
                )
                or interface.get(
                    "stats"
                )
            ),
        }

        records.append(
            build_record(
                signal_type=(
                    "network_interface"
                ),
                payload=payload,
            )
        )

    return records


def collect_routes() -> List[dict]:
    if shutil.which("ip") is None:
        return []

    routes = run_json_command(
        [
            "ip",
            "-j",
            "route",
            "show",
            "table",
            "all",
        ]
    )

    records: List[dict] = []

    for route in routes:
        records.append(
            build_record(
                signal_type=(
                    "route_state"
                ),
                payload={
                    "destination": (
                        route.get("dst")
                        or "default"
                    ),
                    "gateway": (
                        route.get(
                            "gateway"
                        )
                    ),
                    "device": (
                        route.get("dev")
                    ),
                    "source": (
                        route.get(
                            "prefsrc"
                        )
                    ),
                    "protocol": (
                        route.get(
                            "protocol"
                        )
                    ),
                    "scope": (
                        route.get(
                            "scope"
                        )
                    ),
                    "metric": (
                        route.get(
                            "metric"
                        )
                    ),
                    "table": (
                        route.get(
                            "table"
                        )
                    ),
                    "route_type": (
                        route.get(
                            "type"
                        )
                    ),
                },
            )
        )

    return records


def collect_dns() -> List[dict]:
    path = Path(
        "/etc/resolv.conf"
    )

    nameservers = []
    search_domains = []

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        lines = []

    for line in lines:
        stripped = line.strip()

        if (
            not stripped
            or stripped.startswith("#")
        ):
            continue

        parts = stripped.split()

        if not parts:
            continue

        if (
            parts[0] == "nameserver"
            and len(parts) > 1
        ):
            nameservers.append(
                parts[1]
            )

        if (
            parts[0] in (
                "search",
                "domain",
            )
            and len(parts) > 1
        ):
            search_domains.extend(
                parts[1:]
            )

    return [
        build_record(
            signal_type="dns_state",
            payload={
                "nameservers": (
                    nameservers
                ),
                "search_domains": (
                    search_domains
                ),
                "source": str(path),
            },
        )
    ]


def collect_network() -> List[dict]:
    return (
        collect_interfaces()
        + collect_routes()
        + collect_dns()
    )


if __name__ == "__main__":
    for record in collect_network():
        print(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )
