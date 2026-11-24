from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.collectors.linux.schema import (
    build_record,
)


RUNTIME_COMMANDS = {
    "docker": "docker",
    "nerdctl": "nerdctl",
    "podman": "podman",
    "containerd": "containerd",
    "ctr": "ctr",
    "crictl": "crictl",
    "buildctl": "buildctl",
    "buildkitd": "buildkitd",
}


def run_command(
    command: List[str],
    *,
    timeout: int = 30,
    environment: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired:
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


def command_version(
    command: str,
) -> Optional[str]:
    executable = shutil.which(command)

    if executable is None:
        return None

    candidates = (
        [command, "--version"],
        [command, "version"],
    )

    for candidate in candidates:
        result = run_command(
            candidate,
            timeout=10,
        )

        output = result.stdout.strip() or result.stderr.strip()

        if output:
            return output.splitlines()[0]

    return None


def service_active(
    service_name: str,
) -> Optional[bool]:
    if shutil.which("systemctl") is None:
        return None

    result = run_command(
        [
            "systemctl",
            "is-active",
            service_name,
        ],
        timeout=10,
    )

    if result.returncode == 0:
        return True

    state = result.stdout.strip()

    if state in {
        "inactive",
        "failed",
        "unknown",
    }:
        return False

    return None


def socket_exists(
    path: str,
) -> Optional[bool]:
    try:
        return Path(path).exists()
    except OSError:
        return None


def runtime_records() -> List[dict]:
    records: List[dict] = []

    for runtime, command in RUNTIME_COMMANDS.items():
        executable = shutil.which(command)

        sockets: List[str] = []

        if runtime == "docker":
            sockets = [
                "/var/run/docker.sock",
                "/run/docker.sock",
            ]

        elif runtime in {
            "containerd",
            "ctr",
            "nerdctl",
        }:
            sockets = [
                ("/run/containerd/containerd.sock"),
                ("/var/run/containerd/containerd.sock"),
                (f"/run/user/{os.getuid()}/containerd-rootless/containerd.sock"),
            ]

        elif runtime in {
            "buildctl",
            "buildkitd",
        }:
            sockets = [
                ("/run/buildkit/buildkitd.sock"),
                (f"/run/user/{os.getuid()}/buildkit/buildkitd.sock"),
            ]

        service_name = None

        if runtime in {
            "containerd",
            "ctr",
            "nerdctl",
        }:
            service_name = "containerd.service"

        elif runtime == "docker":
            service_name = "docker.service"

        elif runtime in {
            "buildctl",
            "buildkitd",
        }:
            service_name = "buildkit.service"

        records.append(
            build_record(
                signal_type=("container_runtime"),
                payload={
                    "runtime": runtime,
                    "command": command,
                    "installed": (executable is not None),
                    "executable": executable,
                    "version": (command_version(command) if executable else None),
                    "service": (service_name),
                    "service_active": (
                        service_active(service_name) if service_name else None
                    ),
                    "known_sockets": [
                        {
                            "path": path,
                            "exists": (socket_exists(path)),
                        }
                        for path in sockets
                    ],
                },
            )
        )

    return records


def parse_json_lines(
    text: str,
) -> Iterable[dict]:
    for line in text.splitlines():
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid container JSON output: {exc}") from exc

        if not isinstance(record, dict):
            raise ValueError("Container JSON output must contain objects")

        yield record


def _crictl_containers(text: str) -> List[dict]:
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid crictl JSON output: {exc}") from exc

    if not isinstance(result, dict):
        raise ValueError("crictl JSON output must be an object")

    containers = result.get("containers", [])

    if not isinstance(containers, list):
        raise ValueError("crictl JSON output has an invalid containers field")

    return [item for item in containers if isinstance(item, dict)]


def _container_payload(
    runtime: str,
    item: dict,
) -> dict:
    if runtime == "crictl":
        metadata = item.get("metadata")
        image = item.get("image")

        if not isinstance(metadata, dict):
            metadata = {}

        if not isinstance(image, dict):
            image = {}

        return {
            "runtime": runtime,
            "container_id": item.get("id"),
            "name": metadata.get("name"),
            "image": image.get("image"),
            "state": item.get("state"),
            "created_at": item.get("createdAt"),
        }

    return {
        "runtime": runtime,
        "container_id": (item.get("ID") or item.get("Id")),
        "name": (item.get("Names") or item.get("Name")),
        "image": item.get("Image"),
        "state": item.get("State"),
        "status": item.get("Status"),
        "created_at": item.get("CreatedAt"),
    }


def collect_containers() -> List[dict]:
    records = runtime_records()

    for runtime in (
        "docker",
        "podman",
        "nerdctl",
        "crictl",
    ):
        command = RUNTIME_COMMANDS[runtime]

        if shutil.which(command) is None:
            continue

        if runtime == "crictl":
            list_command = [
                command,
                "ps",
                "--all",
                "--output",
                "json",
            ]
        else:
            list_command = [
                command,
                "ps",
                "--all",
                "--no-trunc",
                "--format",
                "{{json .}}",
            ]

        result = run_command(list_command)

        if result.returncode != 0:
            records.append(
                build_record(
                    signal_type="collector_error",
                    payload={
                        "collector": ("linux_containers"),
                        "runtime": runtime,
                        "error": ("container_list_failed"),
                        "message": (result.stderr.strip()),
                    },
                )
            )
            continue

        try:
            containers = (
                _crictl_containers(result.stdout)
                if runtime == "crictl"
                else list(parse_json_lines(result.stdout))
            )
        except ValueError as exc:
            records.append(
                build_record(
                    signal_type="collector_error",
                    payload={
                        "collector": ("linux_containers"),
                        "runtime": runtime,
                        "error": ("invalid_container_output"),
                        "message": str(exc),
                    },
                )
            )
            continue

        for item in containers:
            records.append(
                build_record(
                    signal_type=("container_instance"),
                    payload=_container_payload(
                        runtime,
                        item,
                    ),
                )
            )

    return records


if __name__ == "__main__":
    for record in collect_containers():
        print(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )
