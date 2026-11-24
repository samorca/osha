from __future__ import annotations

import pwd
import shutil
import subprocess
from pathlib import Path
from typing import List

from src.collectors.linux.schema import (
    build_record,
)


LOGIN_SHELLS = {
    "/bin/bash",
    "/bin/sh",
    "/bin/zsh",
    "/bin/fish",
    "/usr/bin/bash",
    "/usr/bin/zsh",
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
        timeout=20,
    )


def collect_login_users() -> List[dict]:
    records: List[dict] = []

    for user in pwd.getpwall():
        login_enabled = user.pw_shell in LOGIN_SHELLS

        if not login_enabled:
            continue

        records.append(
            build_record(
                signal_type=("security_identity"),
                payload={
                    "user_name": (user.pw_name),
                    "uid": user.pw_uid,
                    "gid": user.pw_gid,
                    "home_directory": (user.pw_dir),
                    "login_shell": (user.pw_shell),
                    "login_enabled": True,
                    "password_data_collected": (False),
                },
            )
        )

    return records


def collect_firewall() -> List[dict]:
    checks = []

    if shutil.which("ufw"):
        result = run_command(
            [
                "ufw",
                "status",
            ]
        )

        checks.append(
            {
                "backend": "ufw",
                "available": True,
                "query_succeeded": (result.returncode == 0),
                "summary": (result.stdout.splitlines()[:10]),
            }
        )

    if shutil.which("nft"):
        result = run_command(
            [
                "nft",
                "list",
                "ruleset",
            ]
        )

        checks.append(
            {
                "backend": "nftables",
                "available": True,
                "query_succeeded": (result.returncode == 0),
                "ruleset_present": (bool(result.stdout.strip())),
            }
        )

    if shutil.which("iptables"):
        result = run_command(
            [
                "iptables",
                "-S",
            ]
        )

        checks.append(
            {
                "backend": "iptables",
                "available": True,
                "query_succeeded": (result.returncode == 0),
                "rule_count": (
                    len(result.stdout.splitlines()) if result.returncode == 0 else None
                ),
            }
        )

    return [
        build_record(
            signal_type=("security_posture"),
            payload={
                "category": "firewall",
                "backends": checks,
            },
        )
    ]


def collect_kernel_security() -> List[dict]:
    allowed = {
        "kernel.randomize_va_space": (Path("/proc/sys/kernel/randomize_va_space")),
        "kernel.kptr_restrict": (Path("/proc/sys/kernel/kptr_restrict")),
        "kernel.dmesg_restrict": (Path("/proc/sys/kernel/dmesg_restrict")),
        "fs.protected_hardlinks": (Path("/proc/sys/fs/protected_hardlinks")),
        "fs.protected_symlinks": (Path("/proc/sys/fs/protected_symlinks")),
    }

    values = {}

    for name, path in allowed.items():
        try:
            values[name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            values[name] = None

    return [
        build_record(
            signal_type=("security_posture"),
            payload={
                "category": ("kernel_security"),
                "values": values,
            },
        )
    ]


def collect_security() -> List[dict]:
    return collect_login_users() + collect_firewall() + collect_kernel_security()


if __name__ == "__main__":
    import json

    for record in collect_security():
        print(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )
