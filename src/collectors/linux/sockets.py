from __future__ import annotations

import re
import shutil
import subprocess
from typing import List, Optional, Tuple

from src.collectors.linux.schema import (
    build_record,
)


PROCESS_PATTERN = re.compile(r'\("(?P<name>[^"]+)",pid=(?P<pid>\d+)')


def run_command(
    command: List[str],
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=20,
    )


def split_endpoint(
    endpoint: str,
) -> Tuple[str, Optional[int]]:
    endpoint = endpoint.strip()

    if endpoint.startswith("["):
        closing = endpoint.rfind("]")

        if closing >= 0:
            address = endpoint[1:closing]

            port_text = endpoint[closing + 1 :].lstrip(":")

            try:
                return (
                    address,
                    int(port_text),
                )
            except ValueError:
                return (
                    address,
                    None,
                )

    if ":" in endpoint:
        address, port_text = endpoint.rsplit(
            ":",
            1,
        )

        try:
            port = int(port_text)
        except ValueError:
            port = None

        return (
            address or "*",
            port,
        )

    return (
        endpoint,
        None,
    )


def exposure_type(
    address: str,
) -> str:
    if address in (
        "0.0.0.0",
        "::",
        "*",
    ):
        return "all_interfaces"

    if address in (
        "127.0.0.1",
        "::1",
    ):
        return "loopback"

    return "specific_interface"


def protocol_family(
    address: str,
) -> str:
    if ":" in address:
        return "ipv6"

    return "ipv4"


def process_details(
    text: str,
) -> Tuple[
    Optional[str],
    Optional[int],
]:
    match = PROCESS_PATTERN.search(text)

    if not match:
        return (
            None,
            None,
        )

    return (
        match.group("name"),
        int(match.group("pid")),
    )


def collect_sockets() -> List[dict]:
    if shutil.which("ss") is None:
        return [
            build_record(
                signal_type=("collector_error"),
                payload={
                    "collector": ("linux_sockets"),
                    "error": ("ss_not_found"),
                    "message": ("The ss command is not available"),
                },
            )
        ]

    result = run_command(
        [
            "ss",
            "-H",
            "-l",
            "-n",
            "-t",
            "-u",
            "-p",
        ]
    )

    if result.returncode != 0:
        return [
            build_record(
                signal_type=("collector_error"),
                payload={
                    "collector": ("linux_sockets"),
                    "error": ("ss_failed"),
                    "message": (result.stderr.strip()),
                },
            )
        ]

    records: List[dict] = []

    for line in result.stdout.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        protocol = parts[0].lower()
        state = parts[1]

        local_endpoint = parts[4]

        address, port = split_endpoint(local_endpoint)

        process_name, pid = process_details(line)

        payload = {
            "protocol": protocol,
            "state": state,
            "address_family": (protocol_family(address)),
            "local_address": address,
            "local_port": port,
            "exposure": (exposure_type(address)),
            "process_name": (process_name),
            "pid": pid,
            "process_visible": (process_name is not None),
        }

        records.append(
            build_record(
                signal_type=("listening_socket"),
                payload=payload,
            )
        )

    return records


if __name__ == "__main__":
    import json

    for record in collect_sockets():
        print(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )
