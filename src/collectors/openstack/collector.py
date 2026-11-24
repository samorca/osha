from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    List,
)

from src.collectors.openstack import (
    compute,
    discovery,
    events,
    identity,
    images,
    logs,
    network,
    services,
    storage,
    telemetry,
)

from src.collectors.openstack.discovery import (
    OpenStackContext,
)

from src.collectors.openstack.schema import (
    error,
    record,
)


Collector = Callable[
    [],
    Iterable[dict],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Read-only OSHA OpenStack collector"))

    parser.add_argument(
        "--cloud",
        default=None,
        help=(
            "Named cloud from clouds.yaml. "
            "Omit to use OS_* environment "
            "variables or default SDK "
            "configuration discovery."
        ),
    )

    parser.add_argument(
        "--region",
        default=None,
    )

    parser.add_argument(
        "--environment",
        default="default",
        help=("OSHA environment identifier."),
    )

    parser.add_argument(
        "--output",
        default=("orchestration-ai-data/live/openstack/telemetry.jsonl"),
    )

    parser.add_argument(
        "--insecure",
        action="store_true",
        help=("Disable TLS verification. Development use only."),
    )

    # Identity privacy controls
    parser.add_argument(
        "--include-users",
        action="store_true",
        help=("Include selected non-secret Keystone user metadata."),
    )

    parser.add_argument(
        "--include-groups",
        action="store_true",
        help=("Include selected Keystone group metadata."),
    )

    # Collector selection controls
    parser.add_argument(
        "--skip-services",
        action="store_true",
    )

    parser.add_argument(
        "--skip-compute",
        action="store_true",
    )

    parser.add_argument(
        "--skip-network",
        action="store_true",
    )

    parser.add_argument(
        "--skip-storage",
        action="store_true",
    )

    parser.add_argument(
        "--skip-images",
        action="store_true",
    )

    parser.add_argument(
        "--skip-identity",
        action="store_true",
    )

    parser.add_argument(
        "--skip-telemetry",
        action="store_true",
    )

    parser.add_argument(
        "--skip-events",
        action="store_true",
    )

    parser.add_argument(
        "--skip-logs",
        action="store_true",
    )

    # Telemetry controls
    parser.add_argument(
        "--skip-measurements",
        action="store_true",
        help=(
            "Collect Gnocchi resource and "
            "metric definitions, but not "
            "measurement values."
        ),
    )

    parser.add_argument(
        "--maximum-metrics",
        type=int,
        default=100,
        help=("Maximum number of metrics for which measurements are queried."),
    )

    parser.add_argument(
        "--metric-granularity",
        type=int,
        default=None,
        help=("Optional Gnocchi measurement granularity in seconds."),
    )

    # Event controls
    parser.add_argument(
        "--maximum-event-details",
        type=int,
        default=500,
        help=("Maximum number of Ceilometer events expanded with event show."),
    )

    # Log controls
    parser.add_argument(
        "--journal-lines",
        type=int,
        default=200,
        help=("Maximum journald entries per OpenStack systemd unit."),
    )

    parser.add_argument(
        "--file-lines",
        type=int,
        default=200,
        help=("Maximum lines read from each OpenStack log file."),
    )

    parser.add_argument(
        "--maximum-log-files",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--since",
        default=None,
        help=("Optional journalctl --since expression."),
    )

    # Output/execution controls
    parser.add_argument(
        "--truncate-output",
        action="store_true",
        help=("Replace the output file instead of appending."),
    )

    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help=("Return exit code 3 when any collector_error record exists."),
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    positive_values = {
        "maximum_metrics": (args.maximum_metrics),
        "maximum_event_details": (args.maximum_event_details),
        "journal_lines": (args.journal_lines),
        "file_lines": (args.file_lines),
        "maximum_log_files": (args.maximum_log_files),
    }

    for name, value in positive_values.items():
        if value < 1:
            option = name.replace(
                "_",
                "-",
            )

            raise ValueError(f"--{option} must be at least 1")

    if args.metric_granularity is not None and args.metric_granularity < 1:
        raise ValueError("--metric-granularity must be at least 1")


def run_collector(
    ctx: OpenStackContext,
    name: str,
    collector: Collector,
) -> List[dict]:
    """
    Execute one collector with failure
    isolation.

    One failing OpenStack subsystem must not
    stop the other collectors.
    """

    print(f"Collecting {name}...")

    try:
        records = list(collector())

    except Exception as exc:
        records = [
            error(
                name,
                exc,
                cloud_id=(ctx.cloud_id),
                region=ctx.region,
                environment_id=(ctx.environment_id),
                operation=(f"collect_{name}"),
            )
        ]

    print(f"  {name}: {len(records)} records")

    return records


def build_jobs(
    ctx: OpenStackContext,
    args: argparse.Namespace,
) -> List[tuple[str, Collector]]:
    """
    Build the enabled collector pipeline.

    Discovery always runs because it describes
    the cloud and service capabilities used by
    all other modules.
    """

    jobs: List[tuple[str, Collector]] = [
        (
            "discovery",
            lambda: discovery.collect(ctx),
        )
    ]

    if not args.skip_services:
        jobs.append(
            (
                "services",
                lambda: services.collect(ctx),
            )
        )

    if not args.skip_compute:
        jobs.append(
            (
                "compute",
                lambda: compute.collect(ctx),
            )
        )

    if not args.skip_network:
        jobs.append(
            (
                "network",
                lambda: network.collect(ctx),
            )
        )

    if not args.skip_storage:
        jobs.append(
            (
                "storage",
                lambda: storage.collect(ctx),
            )
        )

    if not args.skip_images:
        jobs.append(
            (
                "images",
                lambda: images.collect(ctx),
            )
        )

    if not args.skip_identity:
        jobs.append(
            (
                "identity",
                lambda: identity.collect(
                    ctx,
                    include_users=(args.include_users),
                    include_groups=(args.include_groups),
                ),
            )
        )

    if not args.skip_telemetry:
        jobs.append(
            (
                "telemetry",
                lambda: telemetry.collect(
                    ctx,
                    include_measurements=(not args.skip_measurements),
                    maximum_metrics=(args.maximum_metrics),
                    granularity=(args.metric_granularity),
                ),
            )
        )

    if not args.skip_events:
        jobs.append(
            (
                "events",
                lambda: events.collect(
                    ctx,
                    maximum_event_details=(args.maximum_event_details),
                ),
            )
        )

    if not args.skip_logs:
        jobs.append(
            (
                "logs",
                lambda: logs.collect(
                    ctx,
                    lines_per_unit=(args.journal_lines),
                    lines_per_file=(args.file_lines),
                    maximum_files=(args.maximum_log_files),
                    since=args.since,
                ),
            )
        )

    return jobs


def collection_summary(
    ctx: OpenStackContext,
    records: List[dict],
    elapsed_seconds: float,
) -> dict:
    signals = Counter(
        str(
            item.get(
                "signal_type",
                "unknown",
            )
        )
        for item in records
    )

    service_counts = Counter(
        str(
            item.get(
                "service",
                "unknown",
            )
        )
        for item in records
    )

    error_count = signals.get(
        "collector_error",
        0,
    )

    return record(
        "collector",
        "collection_summary",
        {
            "record_count_before_summary": (len(records)),
            "collector_error_count": (error_count),
            "signal_counts": (dict(signals)),
            "service_counts": (dict(service_counts)),
            "elapsed_seconds": (
                round(
                    elapsed_seconds,
                    6,
                )
            ),
            "successful": (error_count == 0),
        },
        cloud_id=(ctx.cloud_id),
        region=ctx.region,
        environment_id=(ctx.environment_id),
        resource_id=(ctx.cloud_id),
        source="osha",
    )


def write_jsonl(
    path: Path,
    records: Iterable[dict],
    *,
    truncate: bool,
) -> int:
    """
    Write normalized records as JSON Lines.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mode = "w" if truncate else "a"

    written = 0

    with path.open(
        mode,
        encoding="utf-8",
    ) as handle:
        for item in records:
            handle.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                    default=str,
                )
                + "\n"
            )

            written += 1

        handle.flush()

    return written


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)

    except ValueError as exc:
        print(
            f"Argument error: {exc}",
            file=sys.stderr,
        )

        return 2

    verify: bool | str = not args.insecure

    try:
        ctx = discovery.connect(
            cloud=args.cloud,
            region=args.region,
            verify=verify,
            environment_id=(args.environment),
        )

    except Exception as exc:
        print(
            (f"OpenStack connection failed: {type(exc).__name__}: {exc}"),
            file=sys.stderr,
        )

        return 1

    started = time.monotonic()

    records: List[dict] = []

    for name, collector_fn in build_jobs(
        ctx,
        args,
    ):
        records.extend(
            run_collector(
                ctx,
                name,
                collector_fn,
            )
        )

    elapsed = time.monotonic() - started

    records.append(
        collection_summary(
            ctx,
            records,
            elapsed,
        )
    )

    output_path = Path(args.output)

    try:
        written = write_jsonl(
            output_path,
            records,
            truncate=(args.truncate_output),
        )

    except OSError as exc:
        print(
            (f"Unable to write output: {exc}"),
            file=sys.stderr,
        )

        return 1

    counts = Counter(
        str(
            item.get(
                "signal_type",
                "unknown",
            )
        )
        for item in records
    )

    print("\nOpenStack discovery complete.")

    print(f"Records written: {written}")

    print(f"Output: {output_path}")

    print(f"Elapsed seconds: {elapsed:.3f}")

    print("\nSignal counts:")

    for signal, count in counts.most_common():
        print(f"  {signal}: {count}")

    errors = counts.get(
        "collector_error",
        0,
    )

    if errors:
        print(f"\nCollector errors: {errors}")

        if args.fail_on_error:
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
