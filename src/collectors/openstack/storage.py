from __future__ import annotations

from collections import Counter
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.collectors.openstack.discovery import (
    OpenStackContext,
    service_available,
)

from src.collectors.openstack.schema import (
    as_dict,
    error,
    record,
    sanitize,
    unavailable,
)


def optional_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def resource_identifier(
    payload: Mapping[str, Any],
    *,
    fallback: str,
) -> str:
    """
    Determine a stable resource identifier.
    """

    keys = (
        "volume_id",
        "snapshot_id",
        "backup_id",
        "volume_type_id",
        "service_id",
        "id",
        "name",
        "host",
    )

    for key in keys:
        value = optional_string(payload.get(key))

        if value:
            return value

    return fallback


def normalize_volume(
    volume: Any,
) -> dict:
    """
    Convert a Cinder Volume object into OSHA's
    storage representation.
    """

    raw = as_dict(volume)

    volume_id = raw.get("id")

    project_id = raw.get("project_id") or raw.get("os-vol-tenant-attr:tenant_id")

    status = optional_string(raw.get("status"))

    status_normalized = status.lower() if status else None

    healthy: Optional[bool] = None

    if status_normalized in {
        "available",
        "in-use",
        "reserved",
    }:
        healthy = True

    elif status_normalized in {
        "error",
        "error_deleting",
        "error_backing-up",
        "error_restoring",
        "error_extending",
        "error_managing",
    }:
        healthy = False

    attachments = raw.get("attachments") or raw.get("volume_attachment") or []

    metadata = raw.get("metadata") or {}

    return sanitize(
        {
            "volume_id": (volume_id),
            "name": raw.get("name"),
            "description": (raw.get("description")),
            "status": status,
            "healthy": healthy,
            "size_gb": (raw.get("size")),
            "project_id": (project_id),
            "user_id": (raw.get("user_id")),
            "volume_type": (raw.get("volume_type")),
            "volume_type_id": (raw.get("volume_type_id")),
            "availability_zone": (raw.get("availability_zone")),
            "bootable": (
                raw.get("is_bootable")
                if ("is_bootable" in raw)
                else raw.get("bootable")
            ),
            "encrypted": (
                raw.get("is_encrypted")
                if ("is_encrypted" in raw)
                else raw.get("encrypted")
            ),
            "multiattach": (
                raw.get("is_multiattach")
                if ("is_multiattach" in raw)
                else raw.get("multiattach")
            ),
            "snapshot_id": (raw.get("snapshot_id")),
            "source_volume_id": (raw.get("source_volume_id")),
            "consistency_group_id": (raw.get("consistency_group_id")),
            "group_id": (raw.get("group_id")),
            "host": (raw.get("host") or raw.get("os-vol-host-attr:host")),
            "migration_status": (raw.get("migration_status")),
            "replication_status": (raw.get("replication_status")),
            "attachments": (attachments),
            "attachment_count": (
                len(attachments)
                if isinstance(
                    attachments,
                    list,
                )
                else None
            ),
            "metadata": metadata,
            "created_at": (raw.get("created_at")),
            "updated_at": (raw.get("updated_at")),
        }
    )


def normalize_snapshot(
    snapshot: Any,
) -> dict:
    """
    Convert a Cinder snapshot into a stable OSHA
    representation.
    """

    raw = as_dict(snapshot)

    status = optional_string(raw.get("status"))

    normalized_status = status.lower() if status else None

    healthy: Optional[bool] = None

    if normalized_status in {
        "available",
        "creating",
    }:
        healthy = True

    elif normalized_status in {
        "error",
        "error_deleting",
    }:
        healthy = False

    return sanitize(
        {
            "snapshot_id": (raw.get("id")),
            "name": raw.get("name"),
            "description": (raw.get("description")),
            "status": status,
            "healthy": healthy,
            "volume_id": (raw.get("volume_id")),
            "size_gb": (raw.get("size")),
            "project_id": (
                raw.get("project_id")
                or raw.get("os-extended-snapshot-attributes:project_id")
            ),
            "progress": (raw.get("progress")),
            "metadata": (raw.get("metadata") or {}),
            "created_at": (raw.get("created_at")),
            "updated_at": (raw.get("updated_at")),
        }
    )


def normalize_backup(
    backup: Any,
) -> dict:
    raw = as_dict(backup)

    status = optional_string(raw.get("status"))

    normalized_status = status.lower() if status else None

    healthy: Optional[bool] = None

    if normalized_status in {
        "available",
        "creating",
        "restoring",
    }:
        healthy = True

    elif normalized_status in {
        "error",
        "error_deleting",
    }:
        healthy = False

    return sanitize(
        {
            "backup_id": (raw.get("id")),
            "name": raw.get("name"),
            "description": (raw.get("description")),
            "status": status,
            "healthy": healthy,
            "volume_id": (raw.get("volume_id")),
            "snapshot_id": (raw.get("snapshot_id")),
            "parent_id": (raw.get("parent_id")),
            "project_id": (raw.get("project_id")),
            "size_gb": (raw.get("size")),
            "object_count": (raw.get("object_count")),
            "container": (raw.get("container")),
            "availability_zone": (raw.get("availability_zone")),
            "is_incremental": (
                raw.get("is_incremental")
                if ("is_incremental" in raw)
                else raw.get("incremental")
            ),
            "has_dependent_backups": (raw.get("has_dependent_backups")),
            "created_at": (raw.get("created_at")),
            "updated_at": (raw.get("updated_at")),
        }
    )


def normalize_volume_type(
    volume_type: Any,
) -> dict:
    raw = as_dict(volume_type)

    extra_specs = raw.get("extra_specs") or raw.get("extra_specs_properties") or {}

    return sanitizee(
        {
            "volume_typeid": (raw.get("id")),
            "name": (raw.get("name")),
            "description": (raw.get("description")),
            "is_public": (raw.get("is_public")),
            "extra_specs": (extra_specs),
            "qos_specs_id": (raw.get("qos_specs_id")),
        }
    )


def normalize_service(
    service: Any,
) -> dict:
    """
    Normalize Cinder service state.
    """

    raw = as_dict(service)

    state = optional_string(raw.get("state"))

    status = optional_string(raw.get("status"))

    state_normalized = state.lower() if state else None

    status_normalized = status.lower() if status else None

    healthy: Optional[bool] = None

    if state_normalized == "up" and status_normalized in {
        None,
        "enabled",
    }:
        healthy = True

    elif state_normalized == "down" or status_normalized == "disabled":
        healthy = False

    return sanitize(
        {
            "service_id": (raw.get("id")),
            "binary": (raw.get("binary")),
            "host": (raw.get("host")),
            "zone": (raw.get("zone") or raw.get("availability_zone")),
            "state": state,
            "status": status,
            "healthy": healthy,
            "disabled_reason": (raw.get("disabled_reason")),
            "replication_status": (raw.get("replication_status")),
            "active_backend_id": (raw.get("active_backend_id")),
            "cluster_name": (raw.get("cluster") or raw.get("cluster_name")),
            "updated_at": (raw.get("updated_at")),
        }
    )


def normalize_availability_zone(
    zone: Any,
) -> dict:
    raw = as_dict(zone)

    return sanitize(
        {
            "name": (raw.get("name") or raw.get("zone_name")),
            "state": (raw.get("state")),
            "available": (
                raw.get("available") if ("available" in raw) else raw.get("zone_state")
            ),
        }
    )


def collect_resource_list(
    ctx: OpenStackContext,
    *,
    signal_type: str,
    operation: str,
    loader: Callable[
        [],
        Iterable[Any],
    ],
    normalizer: Callable[
        [Any],
        dict,
    ],
) -> List[dict]:
    """
    Run one isolated read-only Cinder listing.

    A failure in backups, availability zones,
    service state, or another optional extension
    must not prevent volumes from being
    collected.
    """

    records: List[dict] = []

    try:
        for resource in loader():
            payload = normalizer(resource)

            identifier = resource_identifier(
                payload,
                fallback=(f"unknown-{signal_type}"),
            )

            timestamp = payload.get("updated_at") or payload.get("created_at")

            records.append(
                record(
                    "block_storage",
                    signal_type,
                    payload,
                    cloud_id=(ctx.cloud_id),
                    region=ctx.region,
                    environment_id=(ctx.environment_id),
                    resource_id=(identifier),
                    timestamp=timestamp,
                    source="cinder",
                )
            )

    except Exception as exc:
        records.append(
            error(
                "block_storage",
                exc,
                cloud_id=(ctx.cloud_id),
                region=ctx.region,
                environment_id=(ctx.environment_id),
                operation=operation,
            )
        )

    return records


def collect_volumes(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Attempt cloud-wide volume discovery.

    all_projects=True may require administrative
    privileges, so project-scoped collection is
    used automatically if the broader query
    fails.
    """

    try:
        volumes = list(
            ctx.conn.block_storage.volumes(
                details=True,
                all_projects=True,
            )
        )

        records = []

        for volume in volumes:
            payload = normalize_volume(volume)

            records.append(
                record(
                    "block_storage",
                    "volume",
                    payload,
                    cloud_id=(ctx.cloud_id),
                    region=ctx.region,
                    environment_id=(ctx.environment_id),
                    resource_id=(
                        resource_identifier(
                            payload,
                            fallback=("unknown-volume"),
                        )
                    ),
                    timestamp=(payload.get("updated_at") or payload.get("created_at")),
                    source="cinder",
                )
            )

        return records

    except Exception:
        return collect_resource_list(
            ctx,
            signal_type="volume",
            operation=("list_project_volumes"),
            loader=lambda: ctx.conn.block_storage.volumes(
                details=True,
                all_projects=False,
            ),
            normalizer=(normalize_volume),
        )


def storage_summary(
    records: List[dict],
) -> dict:
    counts = Counter()

    volume_statuses = Counter()
    snapshot_statuses = Counter()
    backup_statuses = Counter()

    volume_total_gb = 0
    volume_attached_count = 0
    unhealthy_volumes = 0

    healthy_services = 0
    unhealthy_services = 0
    unknown_services = 0

    for item in records:
        signal = str(
            item.get(
                "signal_type",
                "unknown",
            )
        )

        counts[signal] += 1

        payload = item.get("payload") or {}

        if signal == "volume":
            status = payload.get("status") or "unknown"

            volume_statuses[str(status)] += 1

            try:
                volume_total_gb += int(payload.get("size_gb") or 0)
            except (
                TypeError,
                ValueError,
            ):
                pass

            if (payload.get("attachment_count") or 0) > 0:
                volume_attached_count += 1

            if payload.get("healthy") is False:
                unhealthy_volumes += 1

        elif signal == "snapshot":
            status = payload.get("status") or "unknown"

            snapshot_statuses[str(status)] += 1

        elif signal == "backup":
            status = payload.get("status") or "unknown"

            backup_statuses[str(status)] += 1

        elif signal == ("block_storage_service"):
            healthy = payload.get("healthy")

            if healthy is True:
                healthy_services += 1
            elif healthy is False:
                unhealthy_services += 1
            else:
                unknown_services += 1

    return {
        "record_counts": (dict(counts)),
        "volume_statuses": (dict(volume_statuses)),
        "snapshot_statuses": (dict(snapshot_statuses)),
        "backup_statuses": (dict(backup_statuses)),
        "volume_capacity": {
            "provisioned_gb": (volume_total_gb),
            "attached_volume_count": (volume_attached_count),
            "unhealthy_volume_count": (unhealthy_volumes),
        },
        "cinder_services": {
            "healthy": (healthy_services),
            "unhealthy": (unhealthy_services),
            "unknown": (unknown_services),
        },
    }


def collect(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Collect read-only Cinder storage inventory.

    Resources:
      volumes
      snapshots
      backups
      volume types
      Cinder services
      availability zones
    """

    if not service_available(
        ctx,
        "block-storage",
    ):
        return [
            unavailable(
                "block_storage",
                ("Block Storage service endpoint is unavailable"),
                cloud_id=(ctx.cloud_id),
                region=ctx.region,
                environment_id=(ctx.environment_id),
            )
        ]

    records: List[dict] = []

    records.extend(collect_volumes(ctx))

    records.extend(
        collect_resource_list(
            ctx,
            signal_type="snapshot",
            operation=("list_snapshots"),
            loader=lambda: ctx.conn.block_storage.snapshots(
                details=True,
            ),
            normalizer=(normalize_snapshot),
        )
    )

    records.extend(
        collect_resource_list(
            ctx,
            signal_type="backup",
            operation=("list_backups"),
            loader=lambda: ctx.conn.block_storage.backups(
                details=True,
            ),
            normalizer=(normalize_backup),
        )
    )

    records.extend(
        collect_resource_list(
            ctx,
            signal_type=("volume_type"),
            operation=("list_volume_types"),
            loader=lambda: ctx.conn.block_storage.types(),
            normalizer=(normalize_volume_type),
        )
    )

    records.extend(
        collect_resource_list(
            ctx,
            signal_type=("block_storage_service"),
            operation=("list_cinder_services"),
            loader=lambda: ctx.conn.block_storage.services(),
            normalizer=(normalize_service),
        )
    )

    records.extend(
        collect_resource_list(
            ctx,
            signal_type=("storage_availability_zone"),
            operation=("list_storage_availability_zones"),
            loader=lambda: ctx.conn.block_storage.availability_zones(),
            normalizer=(normalize_availability_zone),
        )
    )

    records.append(
        record(
            "block_storage",
            "storage_summary",
            storage_summary(records),
            cloud_id=(ctx.cloud_id),
            region=ctx.region,
            environment_id=(ctx.environment_id),
            resource_id=(ctx.cloud_id),
            source="osha",
        )
    )

    return records
