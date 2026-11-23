from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator


def parse_timestamp(
    value: str,
) -> datetime:
    if not isinstance(value, str):
        raise ValueError(
            "timestamp must be a string"
        )

    text = value

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(
        text
    )

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )


def floor_window(
    timestamp: datetime,
    window_seconds: int,
) -> datetime:
    epoch_seconds = int(
        timestamp.timestamp()
    )

    start_seconds = (
        epoch_seconds
        - epoch_seconds % window_seconds
    )

    return datetime.fromtimestamp(
        start_seconds,
        tz=timezone.utc,
    )


def read_jsonl(
    path: Path,
) -> Iterator[dict]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}: {exc}"
                ) from exc

            if (
                record.get("record_type")
                != "normalized_telemetry"
            ):
                continue

            yield record


def make_window_id(
    resource_key: str,
    window_start: datetime,
    window_seconds: int,
) -> str:
    identity = (
        f"{resource_key}|"
        f"{window_start.isoformat()}|"
        f"{window_seconds}"
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def window_record(
    records: list[dict],
    *,
    window_start: datetime,
    window_seconds: int,
) -> dict:
    records.sort(
        key=lambda item: (
            item["timestamp"],
            item["record_id"],
        )
    )

    first = records[0]

    end = window_start + timedelta(
        seconds=window_seconds
    )

    signal_counts = Counter(
        item["signal_type"]
        for item in records
    )

    logs = []
    events = []
    metrics = []
    resource_states = []
    collector_errors = []

    record_ids = []

    for item in records:
        record_ids.append(
            item["record_id"]
        )

        signal_type = item[
            "signal_type"
        ]

        payload = item.get(
            "payload",
            {},
        )

        common = {
            "record_id": item[
                "record_id"
            ],
            "timestamp": item[
                "timestamp"
            ],
            "payload": payload,
        }

        if signal_type == "log":
            logs.append(common)
        elif signal_type == "kubernetes_event":
            events.append(common)
        elif signal_type == "metric":
            metrics.append(common)
        elif signal_type == "resource_state":
            resource_states.append(common)
        elif signal_type == "collector_error":
            collector_errors.append(common)

    return {
        "schema_version": "1",
        "record_type": "correlated_window",
        "window_id": make_window_id(
            first["resource_key"],
            window_start,
            window_seconds,
        ),
        "window_start": (
            window_start.isoformat()
        ),
        "window_end": end.isoformat(),
        "window_seconds": window_seconds,
        "environment_id": first[
            "environment_id"
        ],
        "cluster_id": first[
            "cluster_id"
        ],
        "namespace": first.get(
            "namespace"
        ),
        "service_name": first.get(
            "service_name"
        ),
        "resource_key": first[
            "resource_key"
        ],
        "pod_name": first.get(
            "pod_name"
        ),
        "pod_uid": first.get(
            "pod_uid"
        ),
        "node_name": first.get(
            "node_name"
        ),
        "container_name": first.get(
            "container_name"
        ),
        "signal_counts": dict(
            signal_counts
        ),
        "record_count": len(records),
        "record_ids": record_ids,
        "logs": logs,
        "events": events,
        "metrics": metrics,
        "resource_states": resource_states,
        "collector_errors": collector_errors,
    }


def build_windows(
    input_path: Path,
    output_path: Path,
    *,
    window_seconds: int,
    minimum_records: int,
) -> dict:
    if window_seconds < 1:
        raise ValueError(
            "window_seconds must be positive"
        )

    if minimum_records < 1:
        raise ValueError(
            "minimum_records must be positive"
        )

    groups = defaultdict(list)

    source_records = 0

    for record in read_jsonl(
        input_path
    ):
        source_records += 1

        timestamp = parse_timestamp(
            record["timestamp"]
        )

        start = floor_window(
            timestamp,
            window_seconds,
        )

        key = (
            record["resource_key"],
            start.isoformat(),
        )

        groups[key].append(record)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows_written = 0
    records_in_windows = 0

    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (
            item[0][1],
            item[0][0],
        ),
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for (
            _resource_key,
            start_text,
        ), records in sorted_groups:
            if len(records) < minimum_records:
                continue

            start = parse_timestamp(
                start_text
            )

            window = window_record(
                records,
                window_start=start,
                window_seconds=window_seconds,
            )

            handle.write(
                json.dumps(
                    window,
                    ensure_ascii=False,
                )
                + "\n"
            )

            windows_written += 1
            records_in_windows += len(
                records
            )

    return {
        "source_records": source_records,
        "candidate_windows": len(groups),
        "windows_written": windows_written,
        "records_in_windows": (
            records_in_windows
        ),
        "window_seconds": window_seconds,
        "minimum_records": minimum_records,
        "output": str(output_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build correlated OSHA telemetry windows."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "orchestration-ai-data/processed/"
            "live/kubernetes/normalized.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "orchestration-ai-data/processed/"
            "live/kubernetes/windows.jsonl"
        ),
    )

    parser.add_argument(
        "--window-seconds",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--minimum-records",
        type=int,
        default=1,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    report = build_windows(
        Path(args.input),
        Path(args.output),
        window_seconds=args.window_seconds,
        minimum_records=args.minimum_records,
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
