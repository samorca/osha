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
    fallback: str,
) -> str:
    for key in (
        "id",
        "network_id",
        "subnet_id",
        "router_id",
        "port_id",
        "floating_ip_id",
        "security_group_id",
        "name",
    ):
        value = optional_string(payload.get(key))

        if value:
            return value

    return fallback


def normalize_network(
    network: Any,
) -> dict:
    raw = as_dict(network)

    return sanitize(
        {
            "network_id": (raw.get("id")),
            "name": raw.get("name"),
            "status": raw.get("status"),
            "project_id": (raw.get("project_id") or raw.get("tenant_id")),
            "admin_state_up": (
                raw.get("is_admin_state_up")
                if ("is_admin_state_up" in raw)
                else raw.get("admin_state_up")
            ),
            "shared": (
                raw.get("is_shared") if "is_shared" in raw else raw.get("shared")
            ),
            "external": (
                raw.get("is_router_external")
                if ("is_router_external" in raw)
                else raw.get("router:external")
            ),
            "provider_network_type": (
                raw.get("provider_network_type") or raw.get("provider:network_type")
            ),
            "provider_physical_network": (
                raw.get("provider_physical_network")
                or raw.get("provider:physical_network")
            ),
            "provider_segmentation_id": (
                raw.get("provider_segmentation_id")
                or raw.get("provider:segmentation_id")
            ),
            "mtu": raw.get("mtu"),
            "availability_zones": (raw.get("availability_zones") or []),
            "availability_zone_hints": (raw.get("availability_zone_hints") or []),
            "dns_domain": (raw.get("dns_domain")),
            "subnet_ids": (raw.get("subnet_ids") or raw.get("subnets") or []),
            "created_at": (raw.get("created_at")),
            "updated_at": (raw.get("updated_at")),
        }
    )


def normalize_subnet(
    subnet: Any,
) -> dict:
    raw = as_dict(subnet)

    allocation_pools = raw.get("allocation_pools") or []

    host_routes = raw.get("host_routes") or []

    dns_nameservers = raw.get("dns_nameservers") or raw.get("dns_nameservers_v2") or []

    return sanitize(
        {
            "subnet_id": (raw.get("id")),
            "name": raw.get("name"),
            "network_id": (raw.get("network_id")),
            "project_id": (raw.get("project_id") or raw.get("tenant_id")),
            "cidr": raw.get("cidr"),
            "ip_version": (raw.get("ip_version")),
            "gateway_ip": (raw.get("gateway_ip")),
            "dhcp_enabled": (
                raw.get("is_dhcp_enabled")
                if ("is_dhcp_enabled" in raw)
                else raw.get("enable_dhcp")
            ),
            "ipv6_address_mode": (raw.get("ipv6_address_mode")),
            "ipv6_ra_mode": (raw.get("ipv6_ra_mode")),
            "allocation_pools": (allocation_pools),
            "dns_nameservers": (dns_nameservers),
            "host_routes": (host_routes),
            "segment_id": (raw.get("segment_id")),
            "created_at": (raw.get("created_at")),
            "updated_at": (raw.get("updated_at")),
        }
    )


def normalize_port(
    port: Any,
) -> dict:
    raw = as_dict(port)

    fixed_ips = raw.get("fixed_ips") or []

    security_group_ids = (
        raw.get("security_group_ids") or raw.get("security_groups") or []
    )

    allowed_pairs = raw.get("allowed_address_pairs") or []

    return sanitize(
        {
            "port_id": (raw.get("id")),
            "name": raw.get("name"),
            "status": raw.get("status"),
            "network_id": (raw.get("network_id")),
            "project_id": (raw.get("project_id") or raw.get("tenant_id")),
            "device_id": (raw.get("device_id")),
            "device_owner": (raw.get("device_owner")),
            "mac_address": (raw.get("mac_address")),
            "fixed_ips": (fixed_ips),
            "security_group_ids": (security_group_ids),
            "allowed_address_pairs": (allowed_pairs),
            "admin_state_up": (
                raw.get("is_admin_state_up")
                if ("is_admin_state_up" in raw)
                else raw.get("admin_state_up")
            ),
            "port_security_enabled": (
                raw.get("is_port_security_enabled")
                if ("is_port_security_enabled" in raw)
                else raw.get("port_security_enabled")
            ),
            "binding_host_id": (
                raw.get("binding_host_id") or raw.get("binding:host_id")
            ),
            "binding_vnic_type": (
                raw.get("binding_vnic_type") or raw.get("binding:vnic_type")
            ),
            "binding_vif_type": (
                raw.get("binding_vif_type") or raw.get("binding:vif_type")
            ),
            "dns_name": (raw.get("dns_name")),
            "created_at": (raw.get("created_at")),
            "updated_at": (raw.get("updated_at")),
        }
    )


def normalize_router(
    router: Any,
) -> dict:
    raw = as_dict(router)

    gateway = raw.get("external_gateway_info") or {}

    gateway_network_id = None

    if isinstance(
        gateway,
        Mapping,
    ):
        gateway_network_id = gateway.get("network_id")

    return sanitize(
        {
            "router_id": (raw.get("id")),
            "name": raw.get("name"),
            "status": raw.get("status"),
            "project_id": (raw.get("project_id") or raw.get("tenant_id")),
            "admin_state_up": (
                raw.get("is_admin_state_up")
                if ("s_admin_state_up" in raw)
                else raw.get("admin_state_up")
            ),
            "distributed": (
                raw.get("is_distributed")
                if ("is_distributed" in raw)
                else raw.get("distributed")
            ),
            "ha": (raw.get("is_ha") if "is_ha" in raw else raw.get("ha")),
            "external_gateway": (gateway),
            "external_network_id": (gateway_network_id),
            "availability_zones": (raw.get("availability_zones") or []),
            "availability_zone_hints": (raw.get("availability_zone_hints") or []),
            "routes": (raw.get("routes") or []),
            "created_at": (raw.get("created_at")),
            "updated_at": (raw.get("updated_at")),
        }
    )


def normalize_security_group(
    group: Any,
) -> dict:
    raw = as_dict(group)

    rules = raw.get("security_group_rules") or raw.get("rules") or []

    return sanitize(
        {
            "security_group_id": (raw.get("id")),
            "name": raw.get("name"),
            "description": (raw.get("description")),
            "project_id": (raw.get("project_id") or raw.get("tenant_id")),
            "stateful": (
                raw.get("is_stateful")
                if ("is_stateful" in raw)
                else raw.get("stateful")
            ),
            "rule_count": len(rules),
            "rules": rules,
            "created_at": (raw.get("created_at")),
            "updated_at": (raw.get("updated_at")),
        }
    )


def normalize_floating_ip(
    floating_ip: Any,
) -> dict:
    raw = as_dict(floating_ip)

    return sanitize(
        {
            "floating_ip_id": (raw.get("id")),
            "floating_ip_address": (raw.get("floating_ip_address")),
            "fixed_ip_address": (raw.get("fixed_ip_address")),
            "floating_network_id": (raw.get("floating_network_id")),
            "port_id": (raw.get("port_id")),
            "router_id": (raw.get("router_id")),
            "project_id": (raw.get("project_id") or raw.get("tenant_id")),
            "status": (raw.get("status")),
            "description": (raw.get("description")),
            "created_at": (raw.get("created_at")),
            "updated_at": (raw.get("updated_at")),
        }
    )


def normalize_agent(
    agent: Any,
) -> dict:
    raw = as_dict(agent)

    alive = raw.get("is_alive")

    if alive is None:
        alive = raw.get("alive")

    admin_state_up = raw.get("is_admin_state_up")

    if admin_state_up is None:
        admin_state_up = raw.get("admin_state_up")

    healthy = None

    if alive is True and admin_state_up is not False:
        healthy = True

    elif alive is False or admin_state_up is False:
        healthy = False

    return sanitize(
        {
            "agent_id": (raw.get("id")),
            "agent_type": (raw.get("agent_type")),
            "binary": (raw.get("binary")),
            "host": (raw.get("host")),
            "topic": (raw.get("topic")),
            "alive": alive,
            "admin_state_up": (admin_state_up),
            "healthy": healthy,
            "availability_zone": (raw.get("availability_zone")),
            "heartbeat_timestamp": (raw.get("heartbeat_timestamp")),
            "started_at": (raw.get("started_at")),
            "created_at": (raw.get("created_at")),
            "configurations": (raw.get("configurations") or {}),
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
    Execute an isolated read-only Neutron list
    operation.
    """

    records: List[dict] = []

    try:
        for resource in loader():
            payload = normalizer(resource)

            identifier = resource_identifier(
                payload,
                fallback=(f"unknown-{signal_type}"),
            )

            timestamp = (
                payload.get("updated_at")
                or payload.get("created_at")
                or payload.get("heartbeat_timestamp")
            )

            records.append(
                record(
                    "network",
                    signal_type,
                    payload,
                    cloud_id=(ctx.cloud_id),
                    region=ctx.region,
                    environment_id=(ctx.environment_id),
                    resource_id=(identifier),
                    timestamp=timestamp,
                    source="neutron",
                )
            )

    except Exception as exc:
        records.append(
            error(
                "network",
                exc,
                cloud_id=(ctx.cloud_id),
                region=ctx.region,
                environment_id=(ctx.environment_id),
                operation=operation,
            )
        )

    return records


def network_summary(
    records: List[dict],
) -> dict:
    counts = Counter()

    network_statuses = Counter()
    port_statuses = Counter()
    router_statuses = Counter()

    external_networks = 0
    floating_ips = 0

    healthy_agents = 0
    unhealthy_agents = 0
    unknown_agents = 0

    for item in records:
        signal = str(
            item.get(
                "signal_type",
                "unknown",
            )
        )

        counts[signal] += 1

        payload = item.get("payload") or {}

        if signal == "network":
            status = payload.get("status") or "unknown"

            network_statuses[str(status)] += 1

            if payload.get("external") is True:
                external_networks += 1

        elif signal == "port":
            status = payload.get("status") or "unknown"

            port_statuses[str(status)] += 1

        elif signal == "router":
            status = payload.get("status") or "unknown"

            router_statuses[str(status)] += 1

        elif signal == "floating_ip":
            floating_ips += 1

        elif signal == "network_agent":
            healthy = payload.get("healthy")

            if healthy is True:
                healthy_agents += 1

            elif healthy is False:
                unhealthy_agents += 1

            else:
                unknown_agents += 1

    return {
        "record_counts": (dict(counts)),
        "network_statuses": (dict(network_statuses)),
        "port_statuses": (dict(port_statuses)),
        "router_statuses": (dict(router_statuses)),
        "external_network_count": (external_networks),
        "floating_ip_count": (floating_ips),
        "network_agents": {
            "healthy": (healthy_agents),
            "unhealthy": (unhealthy_agents),
            "unknown": (unknown_agents),
        },
    }


def collect(
    ctx: OpenStackContext,
) -> List[dict]:
    """
    Collect read-only Neutron networking
    inventory.

    Resources:
      networks
      subnets
      ports
      routers
      security groups
      floating IP addresses
      network agents
    """

    if not service_available(
        ctx,
        "network",
    ):
        return [
            unavailable(
                "network",
                ("Network service endpoint is unavailable"),
                cloud_id=(ctx.cloud_id),
                region=ctx.region,
                environment_id=(ctx.environment_id),
            )
        ]

    records: List[dict] = []

    operations = [
        (
            "network",
            "list_networks",
            lambda: ctx.conn.network.networks(),
            normalize_network,
        ),
        (
            "subnet",
            "list_subnets",
            lambda: ctx.conn.network.subnets(),
            normalize_subnet,
        ),
        (
            "port",
            "list_ports",
            lambda: ctx.conn.network.ports(),
            normalize_port,
        ),
        (
            "router",
            "list_routers",
            lambda: ctx.conn.network.routers(),
            normalize_router,
        ),
        (
            "security_group",
            "list_security_groups",
            lambda: ctx.conn.network.security_groups(),
            normalize_security_group,
        ),
        (
            "floating_ip",
            "list_floating_ips",
            lambda: ctx.conn.network.ips(),
            normalize_floating_ip,
        ),
        (
            "network_agent",
            "list_network_agents",
            lambda: ctx.conn.network.agents(),
            normalize_agent,
        ),
    ]

    for (
        signal_type,
        operation,
        loader,
        normalizer,
    ) in operations:
        records.extend(
            collect_resource_list(
                ctx,
                signal_type=(signal_type),
                operation=(operation),
                loader=loader,
                normalizer=(normalizer),
            )
        )

    records.append(
        record(
            "network",
            "network_summary",
            network_summary(records),
            cloud_id=(ctx.cloud_id),
            region=ctx.region,
            environment_id=(ctx.environment_id),
            resource_id=(ctx.cloud_id),
            source="osha",
        )
    )

    return records
