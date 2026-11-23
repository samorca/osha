from __future__ import annotations

import shutil
import subprocess
from typing import List

from src.collectors.linux.schema import (
    build_record,
)


def collect_debian_packages() -> List[dict]:
    result = subprocess.run(
        [
            "dpkg-query",
            "-W",
            "-f",
            (
                "${binary:Package}\t"
                "${Version}\t"
                "${Architecture}\t"
                "${db:Status-Abbrev}\n"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=60,
    )

    if result.returncode != 0:
        return [
            build_record(
                signal_type=(
                    "collector_error"
                ),
                payload={
                    "collector": (
                        "linux_packages"
                    ),
                    "error": (
                        "dpkg_query_failed"
                    ),
                    "message": (
                        result.stderr.strip()
                    ),
                },
            )
        ]

    records: List[dict] = []

    for line in result.stdout.splitlines():
        parts = line.split(
            "\t"
        )

        if len(parts) < 4:
            continue

        package, version, architecture, status = (
            parts[:4]
        )

        installed = (
            len(status) >= 2
            and status[1] == "i"
        )

        records.append(
            build_record(
                signal_type=(
                    "package_state"
                ),
                payload={
                    "package_manager": (
                        "dpkg"
                    ),
                    "name": package,
                    "version": version,
                    "architecture": (
                        architecture
                    ),
                    "status_abbreviation": (
                        status
                    ),
                    "installed": installed,
                },
            )
        )

    return records


def collect_rpm_packages() -> List[dict]:
    result = subprocess.run(
        [
            "rpm",
            "-qa",
            "--qf",
            (
                "%{NAME}\t"
                "%{VERSION}-%{RELEASE}\t"
                "%{ARCH}\n"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=60,
    )

    if result.returncode != 0:
        return []

    records: List[dict] = []

    for line in result.stdout.splitlines():
        parts = line.split(
            "\t"
        )

        if len(parts) < 3:
            continue

        records.append(
            build_record(
                signal_type=(
                    "package_state"
                ),
                payload={
                    "package_manager": (
                        "rpm"
                    ),
                    "name": parts[0],
                    "version": parts[1],
                    "architecture": (
                        parts[2]
                    ),
                    "installed": True,
                },
            )
        )

    return records


def collect_packages() -> List[dict]:
    if shutil.which(
        "dpkg-query"
    ):
        return collect_debian_packages()

    if shutil.which("rpm"):
        return collect_rpm_packages()

    return [
        build_record(
            signal_type=(
                "collector_error"
            ),
            payload={
                "collector": (
                    "linux_packages"
                ),
                "error": (
                    "unsupported_package_manager"
                ),
            },
        )
    ]


if __name__ == "__main__":
    import json

    for record in collect_packages():
        print(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )
