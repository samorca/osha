from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone

from src.collectors.linux.schema import (
    build_record,
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


def journal_timestamp(
    entry: dict,
) -> str:
    raw = entry.get(
        "__REALTIME_TIMESTAMP"
    )

    if raw is not None:
        try:
            microseconds = int(
                raw
            )

            return datetime.fromtimestamp(
                microseconds / 1_000_000,
                tz=timezone.utc,
            ).isoformat()
        except (
            TypeError,
            ValueError,
        ):
            pass

    return datetime.now(
        timezone.utc
    ).isoformat()


def journal_available() -> bool:
    return (
        shutil.which(
            "journalctl"
        )
        is not None
    )


def collect_journal(
    *,
    lines: int = 500,
    since: str | None = None,
) -> list[dict]:
    if not journal_available():
        return [
            build_record(
                signal_type=(
                    "collector_error"
                ),
                payload={
                    "collector": (
                        "linux_journal"
                    ),
                    "error": (
                        "journalctl_not_found"
                    ),
                },
            )
        ]

    command = [
        "journalctl",
        "--no-pager",
        "-o",
        "json",
        "-a",
    ]

    if since:
        command.extend(
            [
                "--since",
                since,
            ]
        )
    else:
        command.extend(
            [
                "-n",
                str(lines),
            ]
        )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return [
            build_record(
                signal_type=(
                    "collector_error"
                ),
                payload={
                    "collector": (
                        "linux_journal"
                    ),
                    "error": (
                        "journalctl_failed"
                    ),
                    "message": (
                        result.stderr.strip()
                    ),
                },
            )
        ]

    records: list[dict] = []

    for line in (
        result.stdout.splitlines()
    ):
        if not line.strip():
            continue

        try:
            entry = json.loads(
                line
            )
        except json.JSONDecodeError:
            continue

        message = entry.get(
            "MESSAGE"
        )

        if message is None:
            message = ""

        if not isinstance(
            message,
            str,
        ):
            message = str(message)

        priority = str(
            entry.get(
                "PRIORITY",
                "",
            )
        )

        payload = {
            "message": message,
            "priority": priority,
            "severity": (
                PRIORITY_NAMES.get(
                    priority
                )
            ),
            "systemd_unit": (
                entry.get(
                    "_SYSTEMD_UNIT"
                )
            ),
            "user_unit": (
                entry.get(
                    "_SYSTEMD_USER_UNIT"
                )
            ),
            "syslog_identifier": (
                entry.get(
                    "SYSLOG_IDENTIFIER"
                )
            ),
            "comm": entry.get(
                "_COMM"
            ),
            "executable": (
                entry.get("_EXE")
            ),
            "pid": entry.get(
                "_PID"
            ),
            "uid": entry.get(
                "_UID"
            ),
            "gid": entry.get(
                "_GID"
            ),
            "boot_id": (
                entry.get(
                    "_BOOT_ID"
                )
            ),
            "transport": (
                entry.get(
                    "_TRANSPORT"
                )
            ),
        }

        records.append(
            build_record(
                signal_type="log",
                timestamp=(
                    journal_timestamp(
                        entry
                    )
                ),
                payload=payload,
            )
        )

    return records
