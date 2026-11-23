from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def pod_identity(
    pod: Any,
    cluster_id: str,
    environment_id: str,
) -> dict:
    metadata = pod.metadata
    spec = pod.spec
    status = pod.status

    labels = (
        dict(metadata.labels)
        if metadata.labels
        else {}
    )

    return {
        "environment_id": environment_id,
        "cluster_id": cluster_id,
        "namespace": metadata.namespace,
        "pod_name": metadata.name,
        "pod_uid": metadata.uid,
        "node_name": spec.node_name,
        "pod_ip": status.pod_ip,
        "phase": status.phase,
        "labels": labels,
    }


def make_record(
    *,
    signal_type: str,
    timestamp: str,
    identity: dict,
    payload: dict,
) -> dict:
    return {
        "schema_version": "1",
        "record_type": (
            "kubernetes_telemetry"
        ),
        "signal_type": signal_type,
        "timestamp": timestamp,
        **identity,
        "payload": payload,
        "collected_at": utc_now(),
    }
