from __future__ import annotations

from collections import Counter
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from src.correlation.registry import (
    ResourceRegistry,
)

from src.telemetry.schema import (
    Provider,
    Relationship,
    RelationshipType,
    ResourceIdentity,
    ResourceType,
    TelemetryRecord,
)


# Confidence values are OSHA heuristic evidence
# strengths. They are not calibrated
# probabilities.
EXPLICIT_RELATIONSHIP_CONFIDENCE = 0.98

PROVIDER_REFERENCE_CONFIDENCE = 0.95

HOST_CONTEXT_CONFIDENCE = 0.82

NAME_REFERENCE_CONFIDENCE = 0.70


KUBERNETES_KIND_RESOURCE_TYPES = {
    "cluster": ResourceType.CLUSTER.value,
    "node": ResourceType.NODE.value,
    "namespace": ResourceType.NAMESPACE.value,
    "pod": ResourceType.POD.value,
    "container": ResourceType.CONTAINER.value,
    "deployment": ResourceType.WORKLOAD.value,
    "statefulset": ResourceType.WORKLOAD.value,
    "daemonset": ResourceType.WORKLOAD.value,
    "replicaset": ResourceType.WORKLOAD.value,
    "job": ResourceType.WORKLOAD.value,
    "cronjob": ResourceType.WORKLOAD.value,
    "service": ResourceType.SERVICE.value,
}


def optional_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def normalized_name(
    value: Any,
) -> Optional[str]:
    text = optional_string(value)

    if text is None:
        return None

    return (
        text.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def native_payload(
    telemetry: TelemetryRecord,
) -> Mapping[str, Any]:
    """
    Return the provider payload retained under
    TelemetryRecord.native.
    """

    native = telemetry.native

    if not isinstance(
        native,
        Mapping,
    ):
        return {}

    payload = native.get(
        "payload"
    )

    if not isinstance(
        payload,
        Mapping,
    ):
        return {}

    return payload


def first_value(
    payload: Mapping[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        if key not in payload:
            continue

        value = payload.get(key)

        if value is None:
            continue

        if isinstance(value, str):
            if not value.strip():
                continue

        return value

    return None


def provider_resource_key(
    *,
    provider: str,
    environment_id: str,
    resource_type: str,
    provider_id: str,
) -> str:
    """
    Construct the same provider-scoped key used
    by ResourceIdentity.effective_id() before a
    canonical identity exists.
    """

    return "::".join(
        (
            provider,
            environment_id,
            resource_type,
            provider_id,
        )
    )


def resolve_resource_id(
    registry: ResourceRegistry,
    *,
    provider: str,
    environment_id: str,
    resource_type: str,
    provider_id: Optional[str] = None,
    name: Optional[str] = None,
    host: Optional[str] = None,
    namespace: Optional[str] = None,
    cluster_id: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve a relationship target against the
    resource registry.

    Strong provider IDs are preferred. Name and
    host matching are used only when exactly one
    compatible registered resource exists.
    """

    normalized_provider_id = (
        optional_string(provider_id)
    )

    if normalized_provider_id:
        direct_key = provider_resource_key(
            provider=provider,
            environment_id=(
                environment_id
            ),
            resource_type=(
                resource_type
            ),
            provider_id=(
                normalized_provider_id
            ),
        )

        direct = registry.get_resource(
            direct_key
        )

        if direct is not None:
            return direct.effective_id()

        matching_provider_ids = [
            resource
            for resource
            in registry.find_resources(
                provider=provider,
                resource_type=(
                    resource_type
                ),
                environment_id=(
                    environment_id
                ),
            )
            if resource.provider_id
            == normalized_provider_id
        ]

        if (
            len(matching_provider_ids)
            == 1
        ):
            return (
                matching_provider_ids[
                    0
                ].effective_id()
            )

    candidates = registry.find_resources(
        provider=provider,
        resource_type=resource_type,
        environment_id=environment_id,
        host=host,
        cluster_id=cluster_id,
        namespace=namespace,
    )

    normalized_display_name = (
        optional_string(name)
    )

    if normalized_display_name:
        candidates = [
            resource
            for resource in candidates
            if resource.name
            == normalized_display_name
        ]

    if len(candidates) == 1:
        return candidates[
            0
        ].effective_id()

    return None


def fallback_target_id(
    *,
    provider: str,
    environment_id: str,
    resource_type: str,
    provider_id: Optional[str],
) -> Optional[str]:
    """
    Build an unresolved provider-scoped target
    only when a strong provider ID is available.
    """

    identifier = optional_string(
        provider_id
    )

    if identifier is None:
        return None

    return provider_resource_key(
        provider=provider,
        environment_id=environment_id,
        resource_type=resource_type,
        provider_id=identifier,
    )


def relationship_key(
    relationship: Relationship,
) -> Tuple[str, str, str]:
    return (
        relationship.source_resource_id,
        relationship.target_resource_id,
        relationship.relationship_type,
    )


def make_relationship(
    telemetry: TelemetryRecord,
    *,
    target_resource_id: Optional[str],
    relationship_type: str,
    confidence: float,
    evidence_type: str,
    evidence: Optional[
        Mapping[str, Any]
    ] = None,
) -> Optional[Relationship]:
    """
    Construct one relationship while rejecting
    incomplete or self-referential edges.
    """

    if not target_resource_id:
        return None

    source_resource_id = (
        telemetry.resource.effective_id()
    )

    if (
        source_resource_id
        == target_resource_id
    ):
        return None

    evidence_payload = {
        "evidence_type": (
            evidence_type
        ),
        "source_record_id": (
            telemetry.record_id
        ),
    }

    if evidence:
        evidence_payload.update(
            dict(evidence)
        )

    return Relationship(
        source_resource_id=(
            source_resource_id
        ),
        target_resource_id=(
            target_resource_id
        ),
        relationship_type=(
            relationship_type
        ),
        provider=(
            telemetry.resource.provider
        ),
        environment_id=(
            telemetry.resource.
            environment_id
        ),
        timestamp=(
            telemetry.signal.timestamp
        ),
        confidence=confidence,
        evidence=evidence_payload,
    )


def append_relationship(
    output: List[Relationship],
    seen: Set[
        Tuple[str, str, str]
    ],
    relationship: Optional[
        Relationship
    ],
) -> None:
    if relationship is None:
        return

    key = relationship_key(
        relationship
    )

    if key in seen:
        return

    seen.add(key)

    output.append(
        relationship
    )


# ---------------------------------------------------------------
# Linux relationships
# ---------------------------------------------------------------

def linux_relationships(
    telemetry: TelemetryRecord,
    registry: ResourceRegistry,
) -> List[Relationship]:
    """
    Extract host relationships from normalized
    Linux telemetry.

    Examples:

        process runs_on host
        service runs_on host
        container runs_on host
        filesystem part_of host
        block device part_of host
    """

    resource = telemetry.resource

    if (
        resource.provider
        != Provider.LINUX.value
    ):
        return []

    if resource.resource_type == (
        ResourceType.HOST.value
    ):
        return []

    host = optional_string(
        resource.host
    )

    if host is None:
        return []

    target_id = resolve_resource_id(
        registry,
        provider=Provider.LINUX.value,
        environment_id=(
            resource.environment_id
        ),
        resource_type=(
            ResourceType.HOST.value
        ),
        name=host,
        host=host,
    )

    if target_id is None:
        return []

    if resource.resource_type in {
        ResourceType.PROCESS.value,
        ResourceType.SERVICE.value,
        ResourceType.CONTAINER.value,
    }:
        relationship_type = (
            RelationshipType.
            RUNS_ON.value
        )
    else:
        relationship_type = (
            RelationshipType.
            PART_OF.value
        )

    relationship = make_relationship(
        telemetry,
        target_resource_id=(
            target_id
        ),
        relationship_type=(
            relationship_type
        ),
        confidence=(
            HOST_CONTEXT_CONFIDENCE
        ),
        evidence_type=(
            "linux_host_context"
        ),
        evidence={
            "host": host,
        },
    )

    return (
        [relationship]
        if relationship
        else []
    )


# ---------------------------------------------------------------
# Kubernetes relationships
# ---------------------------------------------------------------

def owner_references(
    payload: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    references = first_value(
        payload,
        "owner_references",
        "ownerReferences",
        "owners",
    )

    if not isinstance(
        references,
        Sequence,
    ) or isinstance(
        references,
        (
            str,
            bytes,
        ),
    ):
        return []

    return [
        item
        for item in references
        if isinstance(
            item,
            Mapping,
        )
    ]


def kubernetes_owner_relationships(
    telemetry: TelemetryRecord,
    registry: ResourceRegistry,
    payload: Mapping[str, Any],
) -> List[Relationship]:
    """
    Extract Kubernetes ownership edges using
    owner UID/name references.
    """

    output: List[Relationship] = []

    namespace = optional_string(
        first_value(
            payload,
            "namespace",
            "namespace_name",
        )
        or telemetry.resource.namespace
    )

    cluster_id = optional_string(
        first_value(
            payload,
            "cluster_id",
            "cluster_name",
        )
        or telemetry.resource.cluster_id
    )

    for owner in owner_references(
        payload
    ):
        owner_uid = optional_string(
            first_value(
                owner,
                "uid",
                "UID",
            )
        )

        owner_name = optional_string(
            first_value(
                owner,
                "name",
            )
        )

        owner_kind = normalized_name(
            first_value(
                owner,
                "kind",
            )
        )

        owner_type = (
            KUBERNETES_KIND_RESOURCE_TYPES.get(
                owner_kind or "",
                ResourceType.
                WORKLOAD.value,
            )
        )

        target_id = resolve_resource_id(
            registry,
            provider=(
                Provider.
                KUBERNETES.value
            ),
            environment_id=(
                telemetry.resource.
                environment_id
            ),
            resource_type=(
                owner_type
            ),
            provider_id=(
                owner_uid
            ),
            name=owner_name,
            namespace=namespace,
            cluster_id=cluster_id,
        )

        if target_id is None:
            target_id = (
                fallback_target_id(
                    provider=(
                        Provider.
                        KUBERNETES.value
                    ),
                    environment_id=(
                        telemetry.resource.
                        environment_id
                    ),
                    resource_type=(
                        owner_type
                    ),
                    provider_id=(
                        owner_uid
                    ),
                )
            )

        relationship = make_relationship(
            telemetry,
            target_resource_id=(
                target_id
            ),
            relationship_type=(
                RelationshipType.
                MANAGED_BY.value
            ),
            confidence=(
                EXPLICIT_RELATIONSHIP_CONFIDENCE
                if owner_uid
                else NAME_REFERENCE_CONFIDENCE
            ),
            evidence_type=(
                "kubernetes_owner_reference"
            ),
            evidence={
                "owner_uid": owner_uid,
                "owner_name": (
                    owner_name
                ),
                "owner_kind": (
                    owner_kind
                ),
                "namespace": namespace,
                "cluster_id": cluster_id,
            },
        )

        if relationship:
            output.append(
                relationship
            )

    return output


def kubernetes_relationships(
    telemetry: TelemetryRecord,
    registry: ResourceRegistry,
) -> List[Relationship]:
    payload = native_payload(
        telemetry
    )

    resource = telemetry.resource

    if (
        resource.provider
        != Provider.KUBERNETES.value
    ):
        return []

    output: List[Relationship] = []

    seen: Set[
        Tuple[str, str, str]
    ] = set()

    environment_id = (
        resource.environment_id
    )

    cluster_id = optional_string(
        resource.cluster_id
        or first_value(
            payload,
            "cluster_id",
            "cluster_name",
        )
    )

    namespace = optional_string(
        resource.namespace
        or first_value(
            payload,
            "namespace",
            "namespace_name",
        )
    )

    # Pod -> Node
    if resource.resource_type == (
        ResourceType.POD.value
    ):
        node_uid = optional_string(
            first_value(
                payload,
                "node_uid",
            )
        )

        node_name = optional_string(
            first_value(
                payload,
                "node_name",
                "node",
                "spec_node_name",
            )
            or resource.host
        )

        target_id = resolve_resource_id(
            registry,
            provider=(
                Provider.
                KUBERNETES.value
            ),
            environment_id=(
                environment_id
            ),
            resource_type=(
                ResourceType.NODE.value
            ),
            provider_id=node_uid,
            name=node_name,
            host=node_name,
            cluster_id=cluster_id,
        )

        if target_id is None:
            target_id = (
                fallback_target_id(
                    provider=(
                        Provider.
                        KUBERNETES.value
                    ),
                    environment_id=(
                        environment_id
                    ),
                    resource_type=(
                        ResourceType.
                        NODE.value
                    ),
                    provider_id=node_uid,
                )
            )

        append_relationship(
            output,
            seen,
            make_relationship(
                telemetry,
                target_resource_id=(
                    target_id
                ),
                              confidence=(
                    EXPLICIT_RELATIONSHIP_CONFIDENCE
                    if node_uid
                    else HOST_CONTEXT_CONFIDENCE
                ),
                evidence_type=(
                    "kubernetes_node_assignment"
                ),
                evidence={
                    "node_uid": node_uid,
                    "node_name": node_name,
                    "cluster_id": (
                        cluster_id
                    ),
                },
            ),
        )

    # Container -> Pod
    if resource.resource_type == (
        ResourceType.CONTAINER.value
    ):
        pod_uid = optional_string(
            first_value(
                payload,
                "pod_uid",
            )
        )

        pod_name = optional_string(
            first_value(
                payload,
                "pod_name",
            )
        )

        target_id = resolve_resource_id(
            registry,
            provider=(
                Provider.
                KUBERNETES.value
            ),
            environment_id=(
                environment_id
            ),
            resource_type=(
                ResourceType.POD.value
            ),
            provider_id=pod_uid,
            name=pod_name,
            namespace=namespace,
            cluster_id=cluster_id,
        )

        if target_id is None:
            target_id = (
                fallback_target_id(
                    provider=(
                        Provider.
                        KUBERNETES.value
                    ),
                    environment_id=(
                        environment_id
                    ),
                    resource_type=(
                        ResourceType.
                        POD.value
                    ),
                    provider_id=pod_uid,
                )
            )

        append_relationship(
            output,
            seen,
            make_relationship(
                telemetry,
                target_resource_id=(
                    target_id
                ),
                relationship_type=(
                    RelationshipType.
                    PART_OF.value
                ),
                confidence=(
                    EXPLICIT_RELATIONSHIP_CONFIDENCE
                    if pod_uid
                    else NAME_REFERENCE_CONFIDENCE
                ),
                evidence_type=(
                    "kubernetes_pod_reference"
                ),
                evidence={
                    "pod_uid": pod_uid,
                    "pod_name": pod_name,
                    "namespace": namespace,
                    "cluster_id": (
                        cluster_id
                    ),
                },
            ),
        )

    # Namespaced resources -> Namespace
    if (
        namespace
        and resource.resource_type
        not in {
            ResourceType.
            NAMESPACE.value,
            ResourceType.
            CLUSTER.value,
            ResourceType.NODE.value,
        }
    ):
        namespace_uid = optional_string(
            first_value(
                payload,
                "namespace_uid",
            )
        )

        target_id = resolve_resource_id(
            registry,
            provider=(
                Provider.
                KUBERNETES.value
            ),
            environment_id=(
                environment_id
            ),
            resource_type=(
                ResourceType.
                NAMESPACE.value
            ),
            provider_id=(
                namespace_uid
            ),
            name=namespace,
            cluster_id=cluster_id,
        )

        if target_id is None:
            target_id = (
                fallback_target_id(
                    provider=(
                        Provider.
                        KUBERNETES.value
                    ),
                    environment_id=(
                        environment_id
                    ),
                    resource_type=(
                        ResourceType.
                        NAMESPACE.value
                    ),
                    provider_id=(
                        namespace_uid
                    ),
                )
            )

        append_relationship(
            output,
            seen,
            make_relationship(
                telemetry,
                target_resource_id=(
                    target_id
                ),
                relationship_type=(
                    RelationshipType.
                    PART_OF.value
                ),
                confidence=(
                    EXPLICIT_RELATIONSHIP_CONFIDENCE
                    if namespace_uid
                    else NAME_REFERENCE_CONFIDENCE
                ),
                evidence_type=(
                    "kubernetes_namespace"
                ),
                evidence={
                    "namespace": namespace,
                    "namespace_uid": (
                        namespace_uid
                    ),
                    "cluster_id": (
                        cluster_id
                    ),
                },
            ),
        )

    for relationship in (
        kubernetes_owner_relationships(
            telemetry,
            registry,
            payload,
        )
    ):
        append_relationship(
            output,
            seen,
            relationship,
        )

    return output


# ---------------------------------------------------------------
# OpenStack relationships
# ---------------------------------------------------------------

def openstack_target(
    telemetry: TelemetryRecord,
    registry: ResourceRegistry,
    *,
    resource_type: str,
    provider_id: Optional[str],
    name: Optional[str] = None,
    host: Optional[str] = None,
) -> Optional[str]:
    target = resolve_resource_id(
        registry,
        provider=(
            Provider.OPENSTACK.value
        ),
        environment_id=(
            telemetry.resource.
            environment_id
        ),
        resource_type=resource_type,
        provider_id=provider_id,
        name=name,
        host=host,
    )

    if target:
        return target

    return fallback_target_id(
        provider=(
            Provider.OPENSTACK.value
        ),
        environment_id=(
            telemetry.resource.
            environment_id
        ),
        resource_type=resource_type,
        provider_id=provider_id,
    )


def openstack_relationships(
    telemetry: TelemetryRecord,
    registry: ResourceRegistry,
) -> List[Relationship]:
    resource = telemetry.resource

    if (
        resource.provider
        != Provider.OPENSTACK.value
    ):
        return []

    payload = native_payload(
        telemetry
    )

    signal_type = normalized_name(
        telemetry.signal.signal_name
        or first_value(
            telemetry.native,
            "signal_type",
        )
    )

    output: List[Relationship] = []

    seen: Set[
        Tuple[str, str, str]
    ] = set()

    # VM -> compute host
    if resource.resource_type == (
        ResourceType.VM.value
    ):
        host = optional_string(
            first_value(
                payload,
                "host",
                "hypervisor_hostname",
                "compute_host",
            )
            or resource.host
        )

        target_id = (
            openstack_target(
                telemetry,
                registry,
                resource_type=(
                    ResourceType.
                    HOST.value
                ),
                provider_id=None,
                name=host,
                host=host,
            )
        )

        append_relationship(
            output,
            seen,
            make_relationship(
                telemetry,
                target_resource_id=(
                    target_id
                ),
                relationship_type=(
                    RelationshipType.
                    RUNS_ON.value
                ),
                confidence=(
                    HOST_CONTEXT_CONFIDENCE
                ),
                evidence_type=(
                    "openstack_compute_host"
                ),
                evidence={
                    "host": host,
                },
            ),
        )

        attachments = first_value(
            payload,
            "attached_volumes",
            "attachments",
        )

        if isinstance(
            attachments,
            Sequence,
        ) and not isinstance(
            attachments,
            (
                str,
                bytes,
            ),
        ):
            for attachment in attachments:
                if isinstance(
                    attachment,
                    Mapping,
                ):
                    volume_id = (
                        optional_string(
                            first_value(
                                attachment,
                                "id",
                                "volume_id",
                                "volumeId",
                            )
                        )
                    )
                else:
                    volume_id = (
                        optional_string(
                            attachment
                        )
                    )

                target_id = (
                    openstack_target(
                        telemetry,
                        registry,
                        resource_type=(
                            ResourceType.
                            VOLUME.value
                        ),
                        provider_id=(
                            volume_id
                        ),
                    )
                )

                append_relationship(
                    output,
                    seen,
                    make_relationship(
                        telemetry,
                        target_resource_id=(
                            target_id
                        ),
                        relationship_type=(
                            RelationshipType.
                            USES.value
                        ),
                        confidence=(
                            EXPLICIT_RELATIONSHIP_CONFIDENCE
                        ),
                        evidence_type=(
                            "openstack_volume_attachment"
                        ),
                        evidence={
                            "volume_id": (
                                volume_id
                            ),
                        },
                    ),
                )

    # Port -> Network and device
    if resource.resource_type == (
        ResourceType.PORT.value
    ):
        network_id = optional_string(
            payload.get(
                "network_id"
            )
        )

        append_relationship(
            output,
            seen,
            make_relationship(
                telemetry,
                target_resource_id=(
                    openstack_target(
                        telemetry,
                        registry,
                        resource_type=(
                            ResourceType.
                            NETWORK.value
                        ),
                        provider_id=(
                            network_id
                        ),
                    )
                ),
                relationship_type=(
                    RelationshipType.
                    CONNECTED_TO.value
                ),
                confidence=(
                    EXPLICIT_RELATIONSHIP_CONFIDENCE
                ),
                evidence_type=(
                    "openstack_port_network"
                ),
                evidence={
                    "network_id": (
                        network_id
                    ),
                },
            ),
        )

        device_id = optional_string(
            payload.get(
                "device_id"
            )
        )

        device_owner = (
            normalized_name(
                payload.get(
                    "device_owner"
                )
            )
            or ""
        )

        if device_id:
            if (
                "router"
                in device_owner
            ):
                device_type = (
                    ResourceType.
                    ROUTER.value
                )
            else:
                device_type = (
                    ResourceType.VM.value
                )

            append_relationship(
                output,
                seen,
                make_relationship(
                    telemetry,
                    target_resource_id=(
                        openstack_target(
                            telemetry,
                            registry,
                            resource_type=(
                                device_type
                            ),
                            provider_id=(
                                device_id
                            ),
                        )
                    ),
                    relationship_type=(
                        RelationshipType.
                        ATTACHED_TO.value
                    ),
                    confidence=(
                        PROVIDER_REFERENCE_CONFIDENCE
                    ),
                    evidence_type=(
                        "openstack_port_device"
                    ),
                    evidence={
                        "device_id": (
                            device_id
                        ),
                        "device_owner": (
                            device_owner
                        ),
                    },
                ),
            )

    # Subnet -> Network
    if resource.resource_type == (
        ResourceType.SUBNET.value
    ):
        network_id = optional_string(
            payload.get(
                "network_id"
            )
        )

        append_relationship(
            output,
            seen,
            make_relationship(
                telemetry,
                target_resource_id=(
                    openstack_target(
                        telemetry,
                        registry,
                        resource_type=(
                            ResourceType.
                            NETWORK.value
                        ),
                        provider_id=(
                            network_id
                        ),
                    )
                ),
                relationship_type=(
                    RelationshipType.
                    PART_OF.value
                ),
                confidence=(
                    EXPLICIT_RELATIONSHIP_CONFIDENCE
                ),
                evidence_type=(
                    "openstack_subnet_network"
                ),
                evidence={
                    "network_id": (
                        network_id
                    ),
                },
            ),
        )

    # Router -> external network
    if resource.resource_type == (
        ResourceType.ROUTER.value
    ):
        external_network_id = (
            optional_string(
                first_value(
                    payload,
                    "external_network_id",
                )
            )
        )

        append_relationship(
            output,
            seen,
            make_relationship(
                telemetry,
                target_resource_id=(
                    openstack_target(
                        telemetry,
                        registry,
                        resource_type=(
                            ResourceType.
                            NETWORK.value
                        ),
                        provider_id=(
                            external_network_id
                        ),
                    )
                ),
                relationship_type=(
                    RelationshipType.
                    ROUTES_TO.value
                ),
                confidence=(
                    EXPLICIT_RELATIONSHIP_CONFIDENCE
                ),
                evidence_type=(
                    "openstack_router_gateway"
                ),
                evidence={
                    "external_network_id": (
                        external_network_id
                    ),
                },
            ),
        )

    # Floating IP -> Port and Router
    if resource.resource_type == (
        ResourceType.IP_ADDRESS.value
    ):
        port_id = optional_string(
            payload.get("port_id")
        )

        router_id = optional_string(
            payload.get(
                "router_id"
            )
        )

        append_relationship(
            output,
            seen,
            make_relationship(
                telemetry,
                target_resource_id=(
                    openstack_target(
                        telemetry,
                        registry,
                        resource_type=(
                            ResourceType.
                            PORT.value
                        ),
                        provider_id=(
                            port_id
                        ),
                    )
                ),
                relationship_type=(
                    RelationshipType.
                    ATTACHED_TO.value
                ),
                confidence=(
                    EXPLICIT_RELATIONSHIP_CONFIDENCE
                ),
                evidence_type=(
                    "openstack_floating_ip_port"
                ),
                evidence={
                    "port_id": port_id,
                },
            ),
        )

        append_relationship(
            output,
            seen,
            make_relationship(
                telemetry,
                target_resource_id=(
                    openstack_target(
                        telemetry,
                        registry,
                        resource_type=(
                            ResourceType.
                            ROUTER.value
                        ),
                        provider_id=(
                            router_id
                        ),
                    )
                ),
                relationship_type=(
                    RelationshipType.
                    ROUTES_TO.value
                ),
                confidence=(
                    PROVIDER_REFERENCE_CONFIDENCE
                ),
                evidence_type=(
                    "openstack_floating_ip_router"
                ),
                evidence={
                    "router_id": (
                        router_id
                    ),
                },
            ),
        )

    # Volume -> VM attachments
    if resource.resource_type == (
        ResourceType.VOLUME.value
    ):
        attachments = payload.get(
            "attachments"
        )

        if isinstance(
            attachments,
            Sequence,
        ) and not isinstance(
            attachments,
            (
                str,
                bytes,
            ),
        ):
            for attachment in attachments:
                if not isinstance(
                    attachment,
                    Mapping,
                ):
                    continue

                server_id = optional_string(
                    first_value(
                        attachment,
                        "server_id",
                        "serverId",
                        "instance_id",
                    )
                )

                append_relationship(
                    output,
                    seen,
                    make_relationship(
                        telemetry,
                        target_resource_id=(
                            openstack_target(
                                telemetry,
                                registry,
                                resource_type=(
                                    ResourceType.
                                    VM.value
                                ),
                                provider_id=(
                                    server_id
                                ),
                            )
                        ),
                        relationship_type=(
                            RelationshipType.
                            ATTACHED_TO.value
                        ),
                        confidence=(
                            EXPLICIT_RELATIONSHIP_CONFIDENCE
                        ),
                        evidence_type=(
                            "openstack_volume_server_attachment"
                        ),
                        evidence={
                            "server_id": (
                                server_id
                            ),
                            "device": (
                                attachment.get(
                                    "device"
                                )
                            ),
                        },
                    ),
                )

    # Snapshot -> Volume
    if resource.resource_type == (
        ResourceType.SNAPSHOT.value
    ):
        volume_id = optional_string(
            payload.get(
                "volume_id"
            )
        )

        append_relationship(
            output,
            seen,
            make_relationship(
                telemetry,
                target_resource_id=(
                    openstack_target(
                        telemetry,
                        registry,
                        resource_type=(
                            ResourceType.
                            VOLUME.value
                        ),
                        provider_id=(
                            volume_id
                        ),
                    )
                ),
                relationship_type=(
                    RelationshipType.
                    PART_OF.value
                ),
                confidence=(
                    EXPLICIT_RELATIONSHIP_CONFIDENCE
                ),
                evidence_type=(
                    "openstack_snapshot_volume"
                ),
                evidence={
                    "volume_id": (
                        volume_id
                    ),
                },
            ),
        )

    # Project-owned resources -> Project
    project_id = optional_string(
        payload.get(
            "project_id"
        )
    )

    if (
        project_id
        and resource.resource_type
        != ResourceType.PROJECT.value
    ):
        append_relationship(
            output,
            seen,
            make_relationship(
                telemetry,
                target_resource_id=(
                    openstack_target(
                        telemetry,
                        registry,
                        resource_type=(
                            ResourceType.
                            PROJECT.value
                        ),
                        provider_id=(
                            project_id
                        ),
                    )
                ),
                relationship_type=(
                    RelationshipType.
                    PART_OF.value
                ),
                confidence=(
                    PROVIDER_REFERENCE_CONFIDENCE
                ),
                evidence_type=(
                    "openstack_project_scope"
                ),
                evidence={
                    "project_id": (
                        project_id
                    ),
                },
            ),
        )

    return output


# ---------------------------------------------------------------
# Generic extraction and batch API
# ---------------------------------------------------------------

def relationships_from_record(
    telemetry: TelemetryRecord,
    registry: ResourceRegistry,
) -> List[Relationship]:
    """
    Extract provider-backed relationships from
    one canonical telemetry record.
    """

    provider = (
        telemetry.resource.provider
    )

    if provider == Provider.LINUX.value:
        return linux_relationships(
            telemetry,
            registry,
        )

    if (
        provider
        == Provider.KUBERNETES.value
    ):
        return kubernetes_relationships(
            telemetry,
            registry,
        )

    if (
        provider
        == Provider.OPENSTACK.value
    ):
        return openstack_relationships(
            telemetry,
            registry,
        )

    return []


def extract_relationships(
    records: Iterable[
        TelemetryRecord
    ],
    registry: ResourceRegistry,
) -> List[Relationship]:
    """
    Extract and deduplicate relationships from a
    telemetry batch.

    Resources should be registered before this
    function runs so relationship targets can be
    resolved to canonical/effective IDs.
    """

    output: List[Relationship] = []

    seen: Set[
        Tuple[str, str, str]
    ] = set()

    for telemetry in records:
        for relationship in (
            relationships_from_record(
                telemetry,
                registry,
            )
        ):
            append_relationship(
                output,
                seen,
                relationship,
            )

    return output


def relationship_summary(
    relationships: Iterable[
        Relationship
    ],
) -> Dict[str, Any]:
    by_type = Counter()

    by_provider = Counter()

    source_resources = set()

    target_resources = set()

    count = 0

    for relationship in relationships:
        count += 1

        by_type[
            relationship.
            relationship_type
        ] += 1

        by_provider[
            relationship.provider
        ] += 1

        source_resources.add(
            relationship.
            source_resource_id
        )

        target_resources.add(
            relationship.
            target_resource_id
        )

    return {
        "relationship_count": count,
        "relationship_types": (
            dict(by_type)
        ),
        "providers": (
            dict(by_provider)
        ),
        "source_resource_count": (
            len(source_resources)
        ),
        "target_resource_count": (
            len(target_resources)
        ),
    }


class RelationshipExtractor:
    """
    Stateful convenience wrapper for runtime and
    correlation pipelines.
    """

    def __init__(
        self,
        registry: ResourceRegistry,
    ) -> None:
        self.registry = registry

        self._relationships: Dict[
            Tuple[str, str, str],
            Relationship,
        ] = {}

    def extract(
        self,
        telemetry: TelemetryRecord,
    ) -> List[Relationship]:
        relationships = (
            relationships_from_record(
                telemetry,
                self.registry,
            )
        )

        output = []

        for relationship in relationships:
            key = relationship_key(
                relationship
            )

            existing = (
                self._relationships.get(
                    key
                )
            )

            if (
                existing is None
                or (
                    relationship.confidence
                    or 0.0
                )
                > (
                    existing.confidence
                    or 0.0
                )
            ):
                self._relationships[
                    key
                ] = relationship

                output.append(
                    relationship
                )

        return output

    def extract_many(
        self,
        records: Iterable[
            TelemetryRecord
        ],
    ) -> List[Relationship]:
        output = []

        for telemetry in records:
            output.extend(
                self.extract(
                    telemetry
                )
            )

        return output

    def relationships(
        self,
    ) -> List[Relationship]:
        return list(
            self._relationships.values()
        )

    def clear(self) -> None:
        self._relationships.clear()
