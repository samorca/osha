from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.collectors.kubernetes.client import (
    load_kubernetes_client,
)
from src.collectors.kubernetes.config import (
    KubernetesCollectorConfig,
)
from src.collectors.kubernetes.events import (
    collect_events,
)
from src.collectors.kubernetes.logs import (
    collect_pod_logs,
)
from src.collectors.kubernetes.schema import (
    make_record,
    pod_identity,
    utc_now,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect correlated Kubernetes "
            "telemetry for OSHA."
        )
    )

    parser.add_argument(
        "--namespace",
        default=None,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--tail-lines",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--skip-logs",
        action="store_true",
    )

    parser.add_argument(
        "--skip-events",
        action="store_true",
    )

    return parser.parse_args()


def write_records(
    path: Path,
    records,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    written = 0

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    return written


def main():
    args = parse_args()

    settings = (
        KubernetesCollectorConfig
        .auto()
    )

    namespace = (
        args.namespace
        if args.namespace is not None
        else settings.namespace
    )

    output_path = Path(
        args.output
        if args.output
        else settings.output_path
    )

    tail_lines = (
        args.tail_lines
        if args.tail_lines is not None
        else settings.tail_lines
    )

    print(
        "Connecting to Kubernetes..."
    )

    core_api = load_kubernetes_client(
        settings
    )

    print("Connected.")

    if namespace:
        pods = (
            core_api
            .list_namespaced_pod(
                namespace=namespace,
            )
            .items
        )
    else:
        pods = (
            core_api
            .list_pod_for_all_namespaces(
                watch=False
            )
            .items
        )

    print(
        f"Pods discovered: {len(pods)}"
    )

    records = []

    for pod in pods:
        identity = pod_identity(
            pod,
            settings.cluster_id,
            settings.environment_id,
        )

        records.append(
            make_record(
                signal_type=(
                    "resource_state"
                ),
                timestamp=utc_now(),
                identity=identity,
                payload={
                    "containers": [
                        container.name
                        for container
                        in (
                            pod.spec.containers
                            or []
                        )
                    ],
                },
            )
        )

        if not args.skip_logs:
            records.extend(
                collect_pod_logs(
                    core_api,
                    pod,
                    settings.cluster_id,
                    settings.environment_id,
                    tail_lines,
                )
            )

    if not args.skip_events:
        records.extend(
            collect_events(
                core_api,
                cluster_id=(
                    settings.cluster_id
                ),
                environment_id=(
                    settings.environment_id
                ),
                namespace=namespace,
            )
        )

    written = write_records(
        output_path,
        records,
    )

    print(
        f"Records written: {written}"
    )

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()
