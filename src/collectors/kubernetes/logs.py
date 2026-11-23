from __future__ import annotations

from kubernetes import client

from src.collectors.kubernetes.schema import (
    make_record,
    pod_identity,
    utc_now,
)


def collect_pod_logs(
    core_api: client.CoreV1Api,
    pod,
    cluster_id: str,
    environment_id: str,
    tail_lines: int,
) -> list[dict]:
    results: list[dict] = []

    identity = pod_identity(
        pod,
        cluster_id,
        environment_id,
    )

    containers = (
        pod.spec.containers
        or []
    )

    for container in containers:
        try:
            text = core_api.read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=pod.metadata.namespace,
                container=container.name,
                timestamps=True,
                tail_lines=tail_lines,
            )
        except Exception as exc:
            results.append(
                make_record(
                    signal_type=(
                        "collector_error"
                    ),
                    timestamp=utc_now(),
                    identity=identity,
                    payload={
                        "collector": (
                            "pod_log"
                        ),
                        "container_name": (
                            container.name
                        ),
                        "error": (
                            type(exc).__name__
                        ),
                        "message": str(exc),
                    },
                )
            )
            continue

        if not text:
            continue

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            timestamp = utc_now()
            message = line

            parts = line.split(
                " ",
                1,
            )

            if len(parts) == 2:
                candidate_time = (
                    parts[0]
                )

                if "T" in candidate_time:
                    timestamp = (
                        candidate_time
                    )
                    message = parts[1]

            results.append(
                make_record(
                    signal_type="log",
                    timestamp=timestamp,
                    identity=identity,
                    payload={
                        "container_name": (
                            container.name
                        ),
                        "message": message,
                    },
                )
            )

    return results
