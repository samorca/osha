from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SUPPORTED_SIGNAL_TYPES = {
    "log",
    "kubernetes_event",
    "resource_state",
    "metric",
    "collector_error",
}

SENSITIVE_PATTERNS = (
    re.compile(
        r"(?i)(password|passwd|secret|token|api[_-]?key)"
        r"\s*[=:]\s*[^\s,;]+"
    ),
    re.compile(r"(?i)authorization:\s*bearer\s+[^\s]+"),
    re.compile(r"(?i)-----BEGIN [A-Z ]+PRIVATE KEY-----"),
)


def parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp is missing")

    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    # Python datetime supports microseconds, while Kubernetes
    # log timestamps can contain nanoseconds. Reduce only the
    # fractional portion to six digits.
    match = re.match(
        r"^(.*?\.)(\d+)([+-]\d{2}:\d{2})$",
        text,
    )

    if match and len(match.group(2)) > 6:
        text = match.group(1) + match.group(2)[:6] + match.group(3)

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported timestamp: {value}") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def timestamp_string(value: Any) -> str:
    return parse_timestamp(value).isoformat()


def redact_text(value: str) -> str:
    redacted = value

    for pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub(
            "[REDACTED]",
            redacted,
        )

    return redacted


def clean_string(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def clean_labels(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    output = {}

    for key, item in value.items():
        if key is None or item is None:
            continue

        output[str(key)] = str(item)

    return output


def discover_service_name(
    record: dict,
    labels: dict[str, str],
    container_name: str | None,
) -> str:
    candidates = (
        labels.get("app.kubernetes.io/instance"),
        labels.get("app.kubernetes.io/name"),
        labels.get("app"),
        container_name,
        clean_string(record.get("pod_name")),
        "unknown-service",
    )

    for candidate in candidates:
        if candidate:
            return candidate

    return "unknown-service"


def discover_cluster_id(
    record: dict,
) -> str:
    explicit = clean_string(record.get("cluster_id"))

    if explicit:
        return explicit

    labels = clean_labels(record.get("labels"))

    label_cluster = (
        labels.get("k8s.cluster.name")
        or labels.get("cluster")
        or labels.get("cluster_id")
    )

    if label_cluster:
        return label_cluster

    # Current local collector has node identity but cluster_id
    # is null. This fallback keeps records correlatable.
    node_name = clean_string(record.get("node_name"))

    if node_name:
        return f"kubernetes-{node_name}"

    return "kubernetes-local"


def build_resource_key(
    *,
    environment_id: str,
    cluster_id: str,
    namespace: str | None,
    pod_uid: str | None,
    pod_name: str | None,
    container_name: str | None,
    node_name: str | None,
) -> str:
    pod_identity = pod_uid or pod_name or node_name or "cluster"

    parts = (
        environment_id,
        cluster_id,
        namespace or "_cluster",
        pod_identity,
        container_name or "_resource",
    )

    return "/".join(parts)


def build_record_id(
    normalized: dict,
) -> str:
    identity = "|".join(
        [
            str(normalized.get("source_record_type")),
            str(normalized.get("signal_type")),
            str(normalized.get("timestamp")),
            str(normalized.get("resource_key")),
            json.dumps(
                normalized.get(
                    "payload",
                    {},
                ),
                sort_keys=True,
                ensure_ascii=False,
            ),
        ]
    )

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def normalize_payload(
    signal_type: str,
    payload: Any,
) -> dict:
    if not isinstance(payload, dict):
        payload = {}

    output = dict(payload)

    if signal_type == "log":
        message = output.get("message")

        if not isinstance(message, str):
            message = str(message or "")

        output["message"] = redact_text(message)

    if signal_type == "kubernetes_event":
        message = output.get("message")

        if isinstance(message, str):
            output["message"] = redact_text(message)

    return output


def normalize_record(
    record: dict,
) -> dict:
    if not isinstance(record, dict):
        raise ValueError("Telemetry record must be an object")

    if record.get("record_type") != "kubernetes_telemetry":
        raise ValueError("Unsupported record_type")

    signal_type = clean_string(record.get("signal_type"))

    if signal_type not in SUPPORTED_SIGNAL_TYPES:
        raise ValueError(f"Unsupported signal_type: {signal_type}")

    timestamp = timestamp_string(record.get("timestamp"))

    collected_at_value = record.get("collected_at") or record.get("timestamp")

    collected_at = timestamp_string(collected_at_value)

    environment_id = clean_string(record.get("environment_id")) or "default"

    cluster_id = discover_cluster_id(record)

    namespace = clean_string(record.get("namespace"))

    pod_name = clean_string(record.get("pod_name"))

    pod_uid = clean_string(record.get("pod_uid"))

    node_name = clean_string(record.get("node_name"))

    pod_ip = clean_string(record.get("pod_ip"))

    phase = clean_string(record.get("phase"))

    labels = clean_labels(record.get("labels"))

    payload = normalize_payload(
        signal_type,
        record.get("payload"),
    )

    container_name = clean_string(payload.get("container_name"))

    service_name = discover_service_name(
        record,
        labels,
        container_name,
    )

    resource_key = build_resource_key(
        environment_id=environment_id,
        cluster_id=cluster_id,
        namespace=namespace,
        pod_uid=pod_uid,
        pod_name=pod_name,
        container_name=container_name,
        node_name=node_name,
    )

    normalized = {
        "schema_version": "1",
        "record_type": "normalized_telemetry",
        "source_record_type": (record.get("record_type")),
        "signal_type": signal_type,
        "timestamp": timestamp,
        "collected_at": collected_at,
        "environment_id": environment_id,
        "cluster_id": cluster_id,
        "namespace": namespace,
        "service_name": service_name,
        "resource_key": resource_key,
        "pod_name": pod_name,
        "pod_uid": pod_uid,
        "node_name": node_name,
        "pod_ip": pod_ip,
        "container_name": container_name,
        "phase": phase,
        "labels": labels,
        "payload": payload,
    }

    normalized["record_id"] = build_record_id(normalized)

    return normalized


def read_jsonl(
    path: Path,
) -> Iterator[tuple[int, dict]]:
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
                yield (
                    line_number,
                    json.loads(line),
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc


def normalize_file(
    input_path: Path,
    output_path: Path,
    *,
    error_path: Path | None = None,
) -> dict:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if error_path is not None:
        error_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    processed = 0
    written = 0
    duplicates = 0
    failed = 0

    seen = set()

    error_handle = (
        error_path.open(
            "w",
            encoding="utf-8",
        )
        if error_path is not None
        else None
    )

    try:
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as output_handle:
            for line_number, record in read_jsonl(input_path):
                processed += 1

                try:
                    normalized = normalize_record(record)
                except Exception as exc:
                    failed += 1

                    if error_handle is not None:
                        error_handle.write(
                            json.dumps(
                                {
                                    "line_number": (line_number),
                                    "error": (type(exc).__name__),
                                    "message": str(exc),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    continue

                record_id = normalized["record_id"]

                if record_id in seen:
                    duplicates += 1
                    continue

                seen.add(record_id)

                output_handle.write(
                    json.dumps(
                        normalized,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                written += 1
    finally:
        if error_handle is not None:
            error_handle.close()

    return {
        "processed": processed,
        "written": written,
        "duplicates_removed": duplicates,
        "failed": failed,
        "output": str(output_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Normalize OSHA Kubernetes telemetry.")
    )

    parser.add_argument(
        "--input",
        default=("dataset/live/kubernetes/telemetry.jsonl"),
    )

    parser.add_argument(
        "--output",
        default=("dataset/processed/live/kubernetes/normalized.jsonl"),
    )

    parser.add_argument(
        "--errors",
        default=(
            "dataset/processed/live/kubernetes/normalization_errors.jsonl"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    report = normalize_file(
        Path(args.input),
        Path(args.output),
        error_path=Path(args.errors),
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
