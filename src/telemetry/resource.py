from __future__ import annotations

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

from src.telemetry.redaction import (
    RedactionPolicy,
    redact_attributes,
)

from src.telemetry.schema import (
    Provider,
    ResourceIdentity,
    ResourceType,
    build_resource,
    normalize_string,
)


# -----------------------------------------------------------------
# Provider-independent resource type aliases
# -----------------------------------------------------------------

RESOURCE_TYPE_ALIASES = {
    # Linux
    "linux_host": (ResourceType.HOST.value),
    "host": (ResourceType.HOST.value),
    "machine": (ResourceType.HOST.value),
    "process": (ResourceType.PROCESS.value),
    "service": (ResourceType.SERVICE.value),
    "systemd_service": (ResourceType.SERVICE.value),
    "filesystem": (ResourceType.FILESYSTEM.value),
    "block_device": (ResourceType.BLOCK_DEVICE.value),
    # Kubernetes
    "kubernetes_cluster": (ResourceType.CLUSTER.value),
    "cluster": (ResourceType.CLUSTER.value),
    "kubernetes_node": (ResourceType.NODE.value),
    "node": (ResourceType.NODE.value),
    "namespace": (ResourceType.NAMESPACE.value),
    "pod": (ResourceType.POD.value),
    "container": (ResourceType.CONTAINER.value),
    "deployment": (ResourceType.WORKLOAD.value),
    "statefulset": (ResourceType.WORKLOAD.value),
    "daemonset": (ResourceType.WORKLOAD.value),
    "replicaset": (ResourceType.WORKLOAD.value),
    "job": (ResourceType.WORKLOAD.value),
    "cronjob": (ResourceType.WORKLOAD.value),
    # OpenStack
    "cloud": (ResourceType.CLOUD.value),
    "region": (ResourceType.REGION.value),
    "availability_zone": (ResourceType.ZONE.value),
    "compute_instance": (ResourceType.VM.value),
    "server": (ResourceType.VM.value),
    "virtual_machine": (ResourceType.VM.value),
    "network": (ResourceType.NETWORK.value),
    "subnet": (ResourceType.SUBNET.value),
    "router": (ResourceType.ROUTER.value),
    "port": (ResourceType.PORT.value),
    "floating_ip": (ResourceType.IP_ADDRESS.value),
    "volume": (ResourceType.VOLUME.value),
    "snapshot": (ResourceType.SNAPSHOT.value),
    "image": (ResourceType.IMAGE.value),
    "project": (ResourceType.PROJECT.value),
    "user": (ResourceType.USER.value),
    "group": (ResourceType.GROUP.value),
    # Application/database systems
    "database": (ResourceType.DATABASE.value),
    "queue": (ResourceType.QUEUE.value),
    "application": (ResourceType.APPLICATION.value),
}


# -----------------------------------------------------------------
# Signal-to-resource mappings
#
# These describe what type of resource a provider-specific
# observation usually represents.
#
# They are intentionally conservative.
# -----------------------------------------------------------------

LINUX_SIGNAL_RESOURCES = {
    "resource_state": (ResourceType.HOST.value),
    "metric": (ResourceType.HOST.value),
    "network_state": (ResourceType.HOST.value),
    "network_interface": (ResourceType.HOST.value),
    "route_state": (ResourceType.HOST.value),
    "dns_state": (ResourceType.HOST.value),
    "process_state": (ResourceType.PROCESS.value),
    "service_state": (ResourceType.SERVICE.value),
    "listening_socket": (ResourceType.PROCESS.value),
    "container_runtime": (ResourceType.SERVICE.value),
    "container_state": (ResourceType.CONTAINER.value),
    "block_device": (ResourceType.BLOCK_DEVICE.value),
    "filesystem_state": (ResourceType.FILESYSTEM.value),
    "swap_state": (ResourceType.STORAGE.value),
    "package_state": (ResourceType.HOST.value),
    "configuration_state": (ResourceType.HOST.value),
    "security_posture": (ResourceType.HOST.value),
    "security_identity": (ResourceType.USER.value),
    "log": (ResourceType.HOST.value),
}


OPENSTACK_SIGNAL_RESOURCES = {
    "cloud_state": (ResourceType.CLOUD.value),
    "openstack_service": (ResourceType.SERVICE.value),
    "compute_service": (ResourceType.SERVICE.value),
    "compute_instance": (ResourceType.VM.value),
    "compute_hypervisor": (ResourceType.HOST.value),
    "compute_flavor": (ResourceType.UNKNOWN.value),
    "compute_aggregate": (ResourceType.UNKNOWN.value),
    "availability_zone": (ResourceType.ZONE.value),
    "server_group": (ResourceType.UNKNOWN.value),
    "network": (ResourceType.NETWORK.value),
    "subnet": (ResourceType.SUBNET.value),
    "port": (ResourceType.PORT.value),
    "router": (ResourceType.ROUTER.value),
    "security_group": (ResourceType.UNKNOWN.value),
    "floating_ip": (ResourceType.IP_ADDRESS.value),
    "network_agent": (ResourceType.SERVICE.value),
    "volume": (ResourceType.VOLUME.value),
    "snapshot": (ResourceType.SNAPSHOT.value),
    "backup": (ResourceType.STORAGE.value),
    "volume_type": (ResourceType.STORAGE.value),
    "block_storage_service": (ResourceType.SERVICE.value),
    "storage_availability_zone": (ResourceType.ZONE.value),
    "image": (ResourceType.IMAGE.value),
    "project": (ResourceType.PROJECT.value),
    "domain": (ResourceType.UNKNOWN.value),
    "region": (ResourceType.REGION.value),
    "role": (ResourceType.UNKNOWN.value),
    "user": (ResourceType.USER.value),
    "group": (ResourceType.GROUP.value),
    "authenticated_identity": (ResourceType.USER.value),
    "metric_resource": (ResourceType.UNKNOWN.value),
    "metric_definition": (ResourceType.UNKNOWN.value),
    "metric": (ResourceType.UNKNOWN.value),
    "alarm": (ResourceType.UNKNOWN.value),
    "openstack_event": (ResourceType.UNKNOWN.value),
    "log": (ResourceType.SERVICE.value),
}


# Kubernetes collectors can emit different signal names depending
# on the implementation/version, so aliases cover common cases.
KUBERNETES_SIGNAL_RESOURCES = {
    "cluster": (ResourceType.CLUSTER.value),
    "cluster_state": (ResourceType.CLUSTER.value),
    "node": (ResourceType.NODE.value),
    "node_state": (ResourceType.NODE.value),
    "namespace": (ResourceType.NAMESPACE.value),
    "namespace_state": (ResourceType.NAMESPACE.value),
    "pod": (ResourceType.POD.value),
    "pod_state": (ResourceType.POD.value),
    "container": (ResourceType.CONTAINER.value),
    "container_state": (ResourceType.CONTAINER.value),
    "deployment": (ResourceType.WORKLOAD.value),
    "deployment_state": (ResourceType.WORKLOAD.value),
    "statefulset": (ResourceType.WORKLOAD.value),
    "daemonset": (ResourceType.WORKLOAD.value),
    "replicaset": (ResourceType.WORKLOAD.value),
    "job": (ResourceType.WORKLOAD.value),
    "cronjob": (ResourceType.WORKLOAD.value),
    "service": (ResourceType.SERVICE.value),
    "service_state": (ResourceType.SERVICE.value),
    "event": (ResourceType.UNKNOWN.value),
    "log": (ResourceType.UNKNOWN.value),
    "metric": (ResourceType.UNKNOWN.value),
}


PROVIDER_SIGNAL_RESOURCES = {
    Provider.LINUX.value: (LINUX_SIGNAL_RESOURCES),
    Provider.KUBERNETES.value: (KUBERNETES_SIGNAL_RESOURCES),
    Provider.OPENSTACK.value: (OPENSTACK_SIGNAL_RESOURCES),
}


# -----------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------


def optional_string(
    value: Any,
) -> Optional[str]:
    return normalize_string(value)


def first_value(
    payload: Mapping[str, Any],
    *keys: str,
) -> Any:
    """
    Return the first present, non-empty value.
    """

    for key in keys:
        if key not in payload:
            continue

        value = payload.get(key)

        if value is None:
            continue

        if isinstance(
            value,
            str,
        ):
            if not value.strip():
                continue

        return value

    return None


def normalized_native_type(
    value: Any,
) -> Optional[str]:
    text = optional_string(value)

    if text is None:
        return None

    return (
        text.lower()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )


def map_resource_type(
    *,
    provider: str,
    signal_type: Optional[str],
    native_type: Optional[str] = None,
) -> str:
    """
    Convert provider-native resource/signal
    names to the common ResourceType vocabulary.

    This does NOT perform cross-provider identity
    resolution.
    """

    normalized_native = normalized_native_type(native_type)

    if normalized_native:
        mapped = RESOURCE_TYPE_ALIASES.get(normalized_native)

        if mapped:
            return mapped

    normalized_signal = normalized_native_type(signal_type)

    if normalized_signal:
        provider_map = PROVIDER_SIGNAL_RESOURCES.get(
            str(provider),
            {},
        )

        mapped = provider_map.get(normalized_signal)

        if mapped:
            return mapped

        mapped = RESOURCE_TYPE_ALIASES.get(normalized_signal)

        if mapped:
            return mapped

    return ResourceType.UNKNOWN.value


def provider_resource_id(
    provider: str,
    record: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    """
    Find the strongest provider-native resource
    identifier available in a collector record.

    This ID is local to the provider/environment.
    It is not the final cross-source canonical ID.
    """

    candidates = (
        record.get("resource_id"),
        payload.get("resource_id"),
        payload.get("server_id"),
        payload.get("volume_id"),
        payload.get("snapshot_id"),
        payload.get("image_id"),
        payload.get("network_id"),
        payload.get("subnet_id"),
        payload.get("router_id"),
        payload.get("port_id"),
        payload.get("floating_ip_id"),
        payload.get("service_id"),
        payload.get("container_id"),
        payload.get("pod_uid"),
        payload.get("uid"),
        payload.get("pid"),
        payload.get("machine_id"),
        payload.get("hostname"),
        payload.get("name"),
    )

    for candidate in candidates:
        text = optional_string(candidate)

        if text:
            return text

    # Some records represent a provider or
    # environment rather than an object.
    cloud_id = optional_string(record.get("cloud_id"))

    if cloud_id:
        return cloud_id

    host_name = optional_string(record.get("host_name"))

    if host_name:
        return host_name

    return f"{provider}:unknown-resource"


# -----------------------------------------------------------------
# Linux resource mapping
# -----------------------------------------------------------------


def linux_resource(
    record_data: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    policy: RedactionPolicy,
) -> ResourceIdentity:
    signal_type = optional_string(record_data.get("signal_type"))

    native_type = first_value(
        payload,
        "resource_type",
        "type",
    )

    resource_type = map_resource_type(
        provider=(Provider.LINUX.value),
        signal_type=(signal_type),
        native_type=(optional_string(native_type)),
    )

    provider_id = provider_resource_id(
        Provider.LINUX.value,
        record_data,
        payload,
    )

    name = first_value(
        payload,
        "name",
        "service_name",
        "hostname",
        "process_name",
        "mountpoint",
        "path",
    )

    host = first_value(
        payload,
        "hostname",
        "host_name",
        "host",
    ) or record_data.get("host_name")

    attributes = {
        "native_signal_type": (signal_type),
    }

    if resource_type == (ResourceType.PROCESS.value):
        attributes.update(
            {
                "pid": (payload.get("pid")),
                "parent_pid": (payload.get("parent_pid")),
                "user": (payload.get("user")),
                "executable": (payload.get("executable")),
            }
        )

    elif resource_type == (ResourceType.SERVICE.value):
        attributes.update(
            {
                "state": (payload.get("active_state") or payload.get("state")),
                "sub_state": (payload.get("sub_state")),
                "main_pid": (payload.get("main_pid")),
            }
        )

    elif resource_type == (ResourceType.CONTAINER.value):
        attributes.update(
            {
                "runtime": (payload.get("runtime")),
                "namespace": (payload.get("namespace")),
                "image": (payload.get("image")),
            }
        )

    return build_resource(
        provider=(Provider.LINUX.value),
        resource_type=(resource_type),
        provider_id=(provider_id),
        environment_id=str(record_data.get("environment_id") or "default"),
        native_type=(optional_string(native_type) or signal_type),
        name=(optional_string(name)),
        host=(optional_string(host)),
        service=(optional_string(payload.get("service_name"))),
        attributes=(
            redact_attributes(
                attributes,
                policy=policy,
            )
        ),
    )


# -----------------------------------------------------------------
# Kubernetes resource mapping
# -----------------------------------------------------------------


def kubernetes_resource(
    record_data: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    policy: RedactionPolicy,
) -> ResourceIdentity:
    signal_type = optional_string(record_data.get("signal_type"))

    native_type = first_value(
        payload,
        "kind",
        "resource_type",
        "type",
    )

    resource_type = map_resource_type(
        provider=(Provider.KUBERNETES.value),
        signal_type=(signal_type),
        native_type=(optional_string(native_type)),
    )

    provider_id = provider_resource_id(
        Provider.KUBERNETES.value,
        record_data,
        payload,
    )

    name = first_value(
        payload,
        "name",
        "pod_name",
        "container_name",
        "node_name",
        "namespace_name",
    )

    namespace = first_value(
        payload,
        "namespace",
        "namespace_name",
    )

    cluster_id = first_value(
        payload,
        "cluster_id",
        "cluster_name",
    ) or record_data.get("cluster_id")

    host = first_value(
        payload,
        "node_name",
        "node",
        "host",
    )

    attributes = {
        "native_signal_type": (signal_type),
        "kind": (native_type),
        "labels": (payload.get("labels") or {}),
        "status": (payload.get("status") or payload.get("phase")),
    }

    return build_resource(
        provider=(Provider.KUBERNETES.value),
        resource_type=(resource_type),
        provider_id=(provider_id),
        environment_id=str(record_data.get("environment_id") or "default"),
        native_type=(optional_string(native_type) or signal_type),
        name=(optional_string(name)),
        cluster_id=(optional_string(cluster_id)),
        namespace=(optional_string(namespace)),
        host=(optional_string(host)),
        service=(
            optional_string(payload.get("service") or payload.get("service_name"))
        ),
        attributes=(
            redact_attributes(
                attributes,
                policy=policy,
            )
        ),
    )


# -----------------------------------------------------------------
# OpenStack resource mapping
# -----------------------------------------------------------------


def openstack_resource(
    record_data: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    policy: RedactionPolicy,
) -> ResourceIdentity:
    signal_type = optional_string(record_data.get("signal_type"))

    native_type = first_value(
        payload,
        "resource_type",
        "service_type",
        "type",
    )

    resource_type = map_resource_type(
        provider=(Provider.OPENSTACK.value),
        signal_type=(signal_type),
        native_type=(optional_string(native_type)),
    )

    provider_id = provider_resource_id(
        Provider.OPENSTACK.value,
        record_data,
        payload,
    )

    name = first_value(
        payload,
        "name",
        "service_name",
        "hostname",
        "host",
        "binary",
    )

    region = record_data.get("region") or payload.get("region")

    zone = first_value(
        payload,
        "availability_zone",
        "zone",
    )

    host = first_value(
        payload,
        "host",
        "hostname",
        "hypervisor_hostname",
    )

    service = (
        record_data.get("service")
        or payload.get("service_type")
        or payload.get("service_name")
    )

    attributes = {
        "native_signal_type": (signal_type),
        "project_id": (payload.get("project_id")),
        "status": (payload.get("status")),
        "state": (payload.get("state")),
        "healthy": (payload.get("healthy")),
    }

    return build_resource(
        provider=(Provider.OPENSTACK.value),
        resource_type=(resource_type),
        provider_id=(provider_id),
        environment_id=str(record_data.get("environment_id") or "default"),
        native_type=(optional_string(native_type) or signal_type),
        name=(optional_string(name)),
        region=(optional_string(region)),
        zone=(optional_string(zone)),
        host=(optional_string(host)),
        service=(optional_string(service)),
        attributes=(
            redact_attributes(
                attributes,
                policy=policy,
            )
        ),
    )


# -----------------------------------------------------------------
# Generic resource mapping
# -----------------------------------------------------------------


def generic_resource(
    provider: str,
    record_data: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    policy: RedactionPolicy,
) -> ResourceIdentity:
    signal_type = optional_string(record_data.get("signal_type"))

    native_type = first_value(
        payload,
        "resource_type",
        "kind",
        "type",
    )

    resource_type = map_resource_type(
        provider=provider,
        signal_type=(signal_type),
        native_type=(optional_string(native_type)),
    )

    provider_id = provider_resource_id(
        provider,
        record_data,
        payload,
    )

    name = first_value(
        payload,
        "name",
        "hostname",
        "service_name",
    )

    attributes = {
        "native_signal_type": (signal_type),
    }

    return build_resource(
        provider=provider,
        resource_type=(resource_type),
        provider_id=(provider_id),
        environment_id=str(record_data.get("environment_id") or "default"),
        native_type=(optional_string(native_type) or signal_type),
        name=(optional_string(name)),
        region=(optional_string(record_data.get("region"))),
        host=(
            optional_string(
                first_value(
                    payload,
                    "host",
                    "hostname",
                )
            )
        ),
        service=(optional_string(record_data.get("service"))),
        attributes=(
            redact_attributes(
                attributes,
                policy=policy,
            )
        ),
    )


# -----------------------------------------------------------------
# Public resource-normalization API
# -----------------------------------------------------------------


def resource_from_record(
    record_data: Mapping[str, Any],
    *,
    policy: Optional[RedactionPolicy] = None,
) -> ResourceIdentity:
    """
    Convert one provider-native collector record
    into OSHA's ResourceIdentity model.

    This function does NOT merge resources from
    different providers.

    Cross-source identity resolution belongs in
    telemetry/identity.py.
    """

    if policy is None:
        from src.telemetry.redaction import (
            DEFAULT_POLICY,
        )

        policy = DEFAULT_POLICY

    provider = optional_string(record_data.get("provider")) or Provider.OTHER.value

    raw_payload = record_data.get("payload") or {}

    if isinstance(
        raw_payload,
        Mapping,
    ):
        payload: Mapping[
            str,
            Any,
        ] = raw_payload
    else:
        payload = {}

    if provider == (Provider.LINUX.value):
        return linux_resource(
            record_data,
            payload,
            policy=policy,
        )

    if provider == (Provider.KUBERNETES.value):
        return kubernetes_resource(
            record_data,
            payload,
            policy=policy,
        )

    if provider == (Provider.OPENSTACK.value):
        return openstack_resource(
            record_data,
            payload,
            policy=policy,
        )

    return generic_resource(
        provider,
        record_data,
        payload,
        policy=policy,
    )


def resource_key(
    resource: ResourceIdentity,
) -> str:
    """
    Return the provider-scoped resource identity.

    Use this before cross-provider canonical
    identity resolution.
    """

    return resource.effective_id()


def resource_matches_provider(
    resource: ResourceIdentity,
    provider: str,
) -> bool:
    return resource.provider == str(provider)


def resource_matches_type(
    resource: ResourceIdentity,
    resource_type: str,
) -> bool:
    return resource.resource_type == str(resource_type)


def resource_context(
    resource: ResourceIdentity,
) -> Dict[str, Optional[str]]:
    """
    Small context object useful for correlation
    indexes.
    """

    return {
        "resource_id": (resource.effective_id()),
        "provider": (resource.provider),
        "resource_type": (resource.resource_type),
        "environment_id": (resource.environment_id),
        "region": (resource.region),
        "zone": (resource.zone),
        "cluster_id": (resource.cluster_id),
        "namespace": (resource.namespace),
        "host": (resource.host),
        "service": (resource.service),
    }
