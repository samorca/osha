from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Iterable, List

from src.collectors.linux.schema import (
    build_record,
)


ALLOWED_ROOTS = (
    Path("/etc/systemd/system"),
    Path("/etc/docker"),
    Path("/etc/containerd"),
    Path("/etc/ssh"),
    Path("/etc/network"),
)

SAFE_FILES = (
    Path("/etc/os-release"),
    Path("/etc/hostname"),
    Path("/etc/hosts"),
    Path("/etc/fstab"),
    Path("/etc/resolv.conf"),
    Path("/etc/docker/daemon.json"),
    Path("/etc/containerd/config.toml"),
)

SENSITIVE_NAMES = {
    "shadow",
    "gshadow",
    "sudoers",
    "id_rsa",
    "id_ed25519",
    "credentials",
    "token",
    "secret",
    "secrets",
}

SENSITIVE_SUFFIXES = {
    ".key",
    ".pem",
    ".p12",
    ".pfx",
    ".kubeconfig",
}


def sensitive_path(
    path: Path,
) -> bool:
    lowered = path.name.lower()

    if lowered in SENSITIVE_NAMES:
        return True

    if any(lowered.endswith(suffix) for suffix in SENSITIVE_SUFFIXES):
        return True

    parts = {part.lower() for part in path.parts}

    if {
        "private",
        "credentials",
        "secrets",
    } & parts:
        return True

    return False


def safe_hash(
    path: Path,
    *,
    maximum_bytes: int = 4_000_000,
) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > maximum_bytes:
            return None

        digest = hashlib.sha256()

        with path.open("rb") as handle:
            while True:
                chunk = handle.read(65536)

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()
    except (
        OSError,
        PermissionError,
    ):
        return None


def iter_configuration_paths() -> Iterable[Path]:
    seen: set[str] = set()

    for path in SAFE_FILES:
        if path.exists():
            resolved = str(path)

            if resolved not in seen:
                seen.add(resolved)
                yield path

    for root in ALLOWED_ROOTS:
        if not root.is_dir():
            continue

        try:
            iterator = root.rglob("*")
        except OSError:
            continue

        for path in iterator:
            if sensitive_path(path):
                continue

            try:
                if not path.is_file():
                    continue
            except OSError:
                continue

            resolved = str(path)

            if resolved in seen:
                continue

            seen.add(resolved)
            yield path


def configuration_category(
    path: Path,
) -> str:
    text = str(path)

    if "systemd" in text:
        return "systemd"

    if "docker" in text:
        return "docker"

    if "containerd" in text:
        return "containerd"

    if "/ssh/" in text:
        return "ssh"

    if "/network/" in text:
        return "network"

    if path == Path("/etc/fstab"):
        return "storage"

    return "system"


def collect_file_metadata(
    path: Path,
) -> dict | None:
    try:
        metadata = path.stat()
    except (
        OSError,
        PermissionError,
    ):
        return None

    mode = stat.filemode(metadata.st_mode)

    payload = {
        "path": str(path),
        "category": (configuration_category(path)),
        "size_bytes": (metadata.st_size),
        "owner_uid": (metadata.st_uid),
        "group_gid": (metadata.st_gid),
        "permissions": mode,
        "mode_octal": oct(stat.S_IMODE(metadata.st_mode)),
        "modified_timestamp": (metadata.st_mtime),
        "sha256": safe_hash(path),
        "contents_collected": False,
        "sensitive": False,
    }

    return build_record(
        signal_type="configuration_state",
        payload=payload,
    )


def collect_configuration() -> List[dict]:
    records: List[dict] = []

    for path in iter_configuration_paths():
        record = collect_file_metadata(path)

        if record is not None:
            records.append(record)

    return records


if __name__ == "__main__":
    import json

    for record in collect_configuration():
        print(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )
