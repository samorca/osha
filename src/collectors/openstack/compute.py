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
    Select a stable identifier from a Nova
    resource payload.
    """

    for key in (
        "id",
        "uuid",
        "name",
        "host",
        "hypervisor_hostname",
    ):
        value = optional_string(payload.get(key))

        if value:
            return value

    return fallback


def normalize_addresses(
    value: Any,
) -> dict:
    """
    Normalize Nova's network-address mapping.

    Nova commonly returns a mapping where each
    network name points to a list of addresses.
    """

    if not isinstance(
        value,
        Mapping,
    ):
        return {}

    output = {}

    for network_name, addresses in value.items():
        safe_addresses = []

        if isinstance(
            addresses,
            list,
        ):
            for address in addresses:
                if not isinstance(
                    address,
                    Mapping,
                ):
                    continue

                safe_addresses.append(
                    sanitize(
                        {
                            "address": (address.get("addr") or address.get("address")),
                            "version": (address.get("version")),
                            "type": (
                                address.get("OS-EXT-IPS:type") or address.get("type")
                            ),
                            "mac_address": (
                                address.get("OS-EXT-IPS-MAC:mac_addr")
                                or address.get("mac_addr")
                            ),
                        }
                    )
                )

        output[str(network_name)] = safe_addresses

    return output


def normalize_server(
    server: Any,
) -> dict:
    """
    Convert a Nova server resource into a stable
    OSHA instance representation.
    """

    raw = as_dict(server)

    server_id = raw.get("id") or getattr(
        server,
        "id",
        None,
    )

    name = raw.get("name") or getattr(
        server,
        "name",
        None,
    )

    status = raw.get("status") or getattr(
        server,
        "status",
        None,
    )

    flavor = raw.get("flavor")

    flavor_id = raw.get("flavor_id")

    if isinstance(
        flavor,
        Mapping,
    ):
        flavor_id = flavor_id or flavor.get("id") or flavor.get("original_name")

    image = raw.get("image")

    image_id = raw.get("image_id")

    if isinstance(
        image,
        Mapping,
    ):
        image_id = image_id or image.get("id")

    project_id = raw.get("project_id") or raw.get("tenant_id")

    user_id = raw.get("user_id")

    host = (
        raw.get("compute_host")
        or raw.get("hypervisor_hostname")
        or raw.get("OS-EXT-SRV-ATTR:host")
    )

    instance_name = raw.get("instance_name") or raw.get("OS-EXT-SRV-ATTR:instance_name")

    availability_zone = raw.get("availability_zone") or raw.get(
        "OS-EXT-AZ:availability_zone"
    )

    power_state = raw.get("power_state") or raw.get("OS-EXT-STS:power_state")

    task_state = raw.get("task_state") or raw.get("OS-EXT-STS:task_state")

    vm_state = raw.get("vm_state") or raw.get("OS-EXT-STS:vm_state")

    status_text = optional_string(status)

    healthy = None

    if status_text:
        normalized_status = status_text.upper()

        if normalized_status in {
            "ACTIVE",
            "SHUTOFF",
            "PAUSED",
            "SUSPENDED",
        }:
            healthy = True

        elif normalized_status in {
            "ERROR",
            "UNKNOWN",
        }:
            healthy = False

    payload = {
        "server_id": (optional_string(server_id)),
        "name": (optional_string(name)),
        "status": status_text,
        "healthy": healthy,
        "project_id": (optional_string(project_id)),
        "user_id": (optional_string(user_id)),
        "host": (optional_string(host)),
        "instance_name": (optional_string(instance_name)),
        "availability_zone": (optional_string(availability_zone)),
        "flavor_id": (optional_string(flavor_id)),
        "image_id": (optional_string(image_id)),
        "key_name": (optional_string(raw.get("key_name"))),
        "power_state": (power_state),
        "task_state": (optional_string(task_state)),
        "vm_state": (optional_string(vm_state)),
        "created_at": (raw.get("created_at") or raw.get("created")),
        "updated_at": (raw.get("updated_at") or raw.get("updated")),
        "launched_at": (raw.get("launched_at")),
        "terminated_at": (raw.get("terminated_at")),
        "addresses": (normalize_addresses(raw.get("addresses"))),
        "attached_volumes": (
            sanitize(raw.get("attached_volumes") or raw.get("volumes_attached") or [])
        ),
        "security_groups": (sanitize(raw.get("security_groups") or [])),
        "metadata": (sanitize(raw.get("metadata") or {})),
        "tags": (sanitize(raw.get("tags") or [])),
    }

    return sanitize(payload)


def normalize_flavor(
    flavor: Any,
) -> dict:
    raw = as_dict(flavor)

    return sanitize(
        {
            "flavor_id": (raw.get("id")),
            "name": raw.get("name"),
            "vcpus": (raw.get("vcpus")),
            "ram_mb": (raw.get("ram")),
            "disk_gb": (raw.get("disk")),
            "ephemeral_gb": (raw.get("ephemeral")),
            "swap_mb": (raw.get("swap")),
            "is_public": (raw.get("is_public")),
            "disabled": (raw.get("disabled")),
            "description": (raw.get("description")),
            "extra_specs": (raw.get("extra_specs") or {}),
        }
    )


def normalize_hypervisor(
    hypervisor: Any,
) -> dict:
    raw = as_dict(hypervisor)

    return sanitize(
        {
            "hypervisor_id": (raw.get("id")),
            "hostname": (raw.get("hypervisor_hostname") or raw.get("name")),
            "host_ip": (raw.get("host_ip")),
            "state": raw.get("state"),
            "status": raw.get("status"),
            "type": raw.get("hypervisor_type"),
            "version": (raw.get("hypervisor_version")),
            "running_vms": (raw.get("running_vms")),
            "vcpus": (raw.get("vcpus")),
            "vcpus_used": (raw.get("vcpus_used")),
            "memory_mb": (raw.get("memory_size") or raw.get("memory_mb")),
            "memory_used_mb": (raw.get("memory_used") or raw.get("memory_mb_used")),
            "local_disk_gb": (raw.get("local_disk_size") or raw.get("local_gb")),
            "local_disk_used_gb": (
                raw.get("local_disk_used") or raw.get("local_gb_used")
            ),
            "current_workload": (raw.get("current_workload")),
            "disk_available_least": (raw.get("disk_available_least")),
        }
    )


def collect_resource_list(
    ctx: OpenStackContext,
    *,
    signal_type: str,
    operation: str,
    source: str,
    loader: Callable[
        [],
        Iterable[Any],
    ],
    normalizer: Callable[
        [Any],
        dict,
    ] = as_dict,
) -> List[dict]:
    """
    Run one read-only Nova list operation.

    Each operation is isolated, so insufficient
    privileges for hypervisors or aggregates do
    not stop instance/flavor collection.
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
                    "compute",
                    signal_type,
                    payload,
                    cloud_id=(ctx.cloud_id),
                    region=ctx.region,
                    environment_id=(ctx.environment_id),
                    resource_id=(identifier),
                    timestamp=timestamp,
                    source=source,
                )
            )

    except Exception as exc:
        records.append(
            error(
                "compute",
                exc,
                cloud_id=(ctx.cloud_id),
                region=ctx.region,
                environment_id=(ctx.environment_id),
                operation=operation,
            )
        )

    return records


def collect_servers(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Attempt cloud-wide server discovery first.

    Some accounts cannot use all_projects=True.
    If permission is denied, retry using only
    the authenticated project.
    """

    try:
        servers = list(
            ctx.conn.compute.servers(
                details=True,
                all_projects=True,
            )
        )

        return [
            record(
                "compute",
                "compute_instance",
                normalize_server(server),
                cloud_id=(ctx.cloud_id),
                region=ctx.region,
                environment_id=(ctx.environment_id),
                resource_id=(
                    optional_string(
                        getattr(
                            server,
                            "id",
                            None,
                        )
                    )
                    or "unknown-server"
                ),
                timestamp=(
                    getattr(
                        server,
                        "updated_at",
                        None,
                    )
                ),
                source="nova",
            )
            for server in servers
        ]

    except Exception:
        # Retry project-scoped discovery. The
        # original admin-level error is not
        # stored when the fallback succeeds.
        return collect_resource_list(
            ctx,
            signal_type=("compute_instance"),
            operation=("list_project_servers"),
            source="nova",
            loader=lambda: ctx.conn.compute.servers(
                details=True,
                all_projects=False,
            ),
            normalizer=(normalize_server),
        )


def compute_summary(
    records: List[dict],
) -> dict:
    counts = Counter()
    instance_statuses = Counter()
    unhealthy_instances = 0

    total_vcpus = 0
    total_memory_mb = 0
    total_disk_gb = 0

    for item in records:
        signal = item.get("signal_type")

        counts[str(signal)] += 1

        payload = item.get("payload") or {}

        if signal == "compute_instance":
            status = payload.get("status") or "unknown"

            instance_statuses[str(status)] += 1

            if payload.get("healthy") is False:
                unhealthy_instances += 1

        elif signal == "compute_hypervisor":
            total_vcpus += int(payload.get("vcpus") or 0)

            total_memory_mb += int(payload.get("memory_mb") or 0)

            total_disk_gb += int(payload.get("local_disk_gb") or 0)

    return {
        "record_counts": (dict(counts)),
        "instance_statuses": (dict(instance_statuses)),
        "unhealthy_instance_count": (unhealthy_instances),
        "hypervisor_capacity": {
            "vcpus": total_vcpus,
            "memory_mb": (total_memory_mb),
            "local_disk_gb": (total_disk_gb),
        },
    }


def collect(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Collect read-only Nova compute inventory.

    Resources:
      servers
      flavors
      hypervisors
      host aggregates
      availability zones
      server groups
      keypair metadata
    """

    if not service_available(
        ctx,
        "compute",
    ):
        return [
            unavailable(
                "compute",
                ("Compute service endpoint is unavailable"),
                cloud_id=(ctx.cloud_id),
                region=ctx.region,
                environment_id=(ctx.environment_id),
            )
        ]

    records: List[dict] = []

    records.extend(collect_servers(ctx))

    records.extend(
        collect_resource_list(
            ctx,
            signal_type=("compute_flavor"),
            operation=("list_flavors"),
            source="nova",
            loader=lambda: ctx.conn.compute.flavors(details=True),
            normalizer=(normalize_flavor),
        )
    )

    records.extend(
        collect_resource_list(
            ctx,
            signal_type=("compute_hypervisor"),
            operation=("list_hypervisors"),
            source="nova",
            loader=lambda: ctx.conn.compute.hypervisors(details=True),
            normalizer=(normalize_hypervisor),
        )
    )

    records.extend(
        collect_resource_list(
            ctx,
            signal_type=("compute_aggregate"),
            operation=("list_aggregates"),
            source="nova",
            loader=lambda: ctx.conn.compute.aggregates(),
        )
    )

    records.extend(
        collect_resource_list(
            ctx,
            signal_type=("availability_zone"),
            operation=("list_availability_zones"),
            source="nova",
            loader=lambda: ctx.conn.compute.availability_zones(),
        )
    )

    records.extend(
        collect_resource_list(
            ctx,
            signal_type=("server_group"),
            operation=("list_server_groups"),
            source="nova",
            loader=lambda: ctx.conn.compute.server_groups(all_projects=True),
        )
    )

    # The schema sanitizer removes fields whose
    # keys appear to contain private-key or
    # credential material. This collector stores
    # only keypair metadata exposed by the list
    # operation.
    records.extend(
        collect_resource_list(
            ctx,
            signal_type=("keypair_metadata"),
            operation=("list_keypairs"),
            source="nova",
            loader=lambda: ctx.conn.compute.keypairs(),
        )
    )

    records.append(
        record(
            "compute",
            "compute_summary",
            compute_summary(records),
            cloud_id=(ctx.cloud_id),
            region=ctx.region,
            environment_id=(ctx.environment_id),
            resource_id=(ctx.cloud_id),
            source="osha",
        )
    )

    return records
