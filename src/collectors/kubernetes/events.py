from __future__ import annotations

from kubernetes import client

from src.collectors.kubernetes.schema import (
    make_record,
    utc_now,
)


def _datetime_string(
    value,
) -> str:
    if value is None:
        return utc_now()

    if hasattr(
        value,
        "isoformat",
    ):
        return value.isoformat()

    return str(value)


def collect_events(
    core_api: client.CoreV1Api,
    *,
    cluster_id: str,
    environment_id: str,
    namespace: str | None,
) -> list[dict]:
    if namespace:
        event_list = core_api.list_namespaced_event(namespace=namespace)
    else:
        event_list = core_api.list_event_for_all_namespaces()

    output: list[dict] = []

    for event in event_list.items:
        involved = event.involved_object

        timestamp = event.event_time or event.last_timestamp or event.first_timestamp

        identity = {
            "environment_id": (environment_id),
            "cluster_id": cluster_id,
            "namespace": (involved.namespace if involved else event.metadata.namespace),
            "pod_name": (
                involved.name if (involved and involved.kind == "Pod") else None
            ),
            "pod_uid": (involved.uid if involved else None),
            "node_name": None,
            "pod_ip": None,
            "phase": None,
            "labels": {},
        }

        output.append(
            make_record(
                signal_type=("kubernetes_event"),
                timestamp=(_datetime_string(timestamp)),
                identity=identity,
                payload={
                    "type": event.type,
                    "reason": event.reason,
                    "message": event.message,
                    "count": event.count,
                    "object_kind": (involved.kind if involved else None),
                    "object_name": (involved.name if involved else None),
                },
            )
        )

    return output
