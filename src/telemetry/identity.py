from __future__ import annotations

import hashlib
import json
import re
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from src.telemetry.schema import (
    Provider,
    ResourceIdentity,
    ResourceType,
    json_safe,
    normalize_string,
)


UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)


CONTAINER_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{32,64}$"
)


# Scores are OSHA design choices.
#
# They represent confidence in identity evidence,
# not AI probability.
EVIDENCE_WEIGHTS = {
    "same_canonical_id": 1.00,
    "same_provider_id": 1.00,

    "same_machine_id": 0.98,
    "same_system_uuid": 0.98,
    "same_kubernetes_uid": 0.98,
    "same_cloud_instance_id": 0.98,
    "same_container_id": 0.98,

    "same_cluster_and_uid": 0.98,
    "same_namespace_and_uid": 0.98,

    "same_environment_host": 0.78,
    "same_cluster_node": 0.82,
    "same_region_host": 0.76,

    "same_namespace_name_type": 0.68,
    "same_service_instance": 0.74,

    "same_name": 0.30,
    "same_host": 0.35,
    "same_ip_address": 0.55,
}


# Evidence at or above this score may assign a
# shared canonical identity automatically.
DEFAULT_AUTO_MERGE_THRESHOLD = 0.90


# Evidence at or above this score is considered a
# candidate match for correlation/manual review.
DEFAULT_CANDIDATE_THRESHOLD = 0.65


@dataclass(frozen=True)
class IdentityEvidence:
    """
    One piece of evidence relating two resources.
    """

    evidence_type: str

    score: float

    left_value: Optional[str] = None

    right_value: Optional[str] = None

    description: Optional[str] = None

    metadata: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "evidence_type": (
                self.evidence_type
            ),
            "score": self.score,
            "left_value": (
                self.left_value
            ),
            "right_value": (
                self.right_value
            ),
            "description": (
                self.description
            ),
            "metadata": (
                json_safe(
                    self.metadata
                )
            ),
        }


@dataclass
class IdentityMatch:
    """
    Result of comparing two normalized resources.
    """

    left_resource_id: str

    right_resource_id: str

    score: float

    decision: str

    evidence: List[
        IdentityEvidence
    ] = field(
        default_factory=list
    )

    conflicts: List[str] = field(
        default_factory=list
    )

    canonical_id: Optional[str] = (
        None
    )

    def to_dict(self) -> dict:
        return {
            "left_resource_id": (
                self.left_resource_id
            ),
            "right_resource_id": (
                self.right_resource_id
            ),
            "score": self.score,
            "decision": (
                self.decision
            ),
            "canonical_id": (
                self.canonical_id
            ),
            "evidence": [
                item.to_dict()
                for item
                in self.evidence
            ],
            "conflicts": list(
                self.conflicts
            ),
        }


@dataclass
class IdentityCluster:
    """
    Set of resources resolved to one canonical
    identity.
    """

    canonical_id: str

    resource_ids: List[str]

    resource_type: str

    confidence: float

    evidence: List[
        IdentityEvidence
    ] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        return {
            "canonical_id": (
                self.canonical_id
            ),
            "resource_ids": list(
                self.resource_ids
            ),
            "resource_type": (
                self.resource_type
            ),
            "confidence": (
                self.confidence
            ),
            "evidence": [
                item.to_dict()
                for item
                in self.evidence
            ],
        }


def optional_string(
    value: Any,
) -> Optional[str]:
    return normalize_string(value)


def normalized_text(
    value: Any,
) -> Optional[str]:
    text = optional_string(value)

    if text is None:
        return None

    return text.lower()


def normalized_hostname(
    value: Any,
) -> Optional[str]:
    """
    Normalize a hostname without assuming that a
    short hostname and FQDN are always identical.

    The full normalized hostname is preserved.
    """

    text = normalized_text(value)

    if text is None:
        return None

    return text.rstrip(".")


def short_hostname(
    value: Any,
) -> Optional[str]:
    hostname = normalized_hostname(
        value
    )

    if hostname is None:
        return None

    return hostname.split(
        ".",
        1,
    )[0]


def normalized_identifier(
    value: Any,
) -> Optional[str]:
    text = optional_string(value)

    if text is None:
        return None

    return text.strip().lower()


def normalized_ip(
    value: Any,
) -> Optional[str]:
    """
    Validate and normalize a single IP address.
    """

    text = optional_string(value)

    if text is None:
        return None

    try:
        import ipaddress

        return str(
            ipaddress.ip_address(
                text
            )
        )
    except ValueError:
        return None


def attribute_value(
    resource: ResourceIdentity,
    *keys: str,
) -> Any:
    attributes = (
        resource.attributes
        or {}
    )

    for key in keys:
        if key in attributes:
            value = attributes.get(
                key
            )

            if value is not None:
                return value

    return None


def resource_uid(
    resource: ResourceIdentity,
) -> Optional[str]:
    """
    Return the strongest native object UID found
    in the normalized resource.
    """

    value = attribute_value(
        resource,
        "uid",
        "resource_uid",
        "pod_uid",
        "node_uid",
        "namespace_uid",
        "kubernetes_uid",
    )

    if value is not None:
        return normalized_identifier(
            value
        )

    provider_id = (
        normalized_identifier(
            resource.provider_id
        )
    )

    if (
        resource.provider
        == Provider.KUBERNETES.value
        and provider_id
        and UUID_PATTERN.match(
            provider_id
        )
    ):
        return provider_id

    return None


def machine_id(
    resource: ResourceIdentity,
) -> Optional[str]:
    return normalized_identifier(
        attribute_value(
            resource,
            "machine_id",
            "machine-id",
            "host_id",
        )
    )


def system_uuid(
    resource: ResourceIdentity,
) -> Optional[str]:
    return normalized_identifier(
        attribute_value(
            resource,
            "system_uuid",
            "product_uuid",
            "dmi_product_uuid",
        )
    )


def cloud_instance_id(
    resource: ResourceIdentity,
) -> Optional[str]:
    value = attribute_value(
        resource,
        "instance_id",
        "server_id",
        "cloud_instance_id",
    )

    if value is not None:
        return normalized_identifier(
            value
        )

    if (
        resource.provider
        == Provider.OPENSTACK.value
        and resource.resource_type
        == ResourceType.VM.value
    ):
        return normalized_identifier(
            resource.provider_id
        )

    return None


def container_id(
    resource: ResourceIdentity,
) -> Optional[str]:
    value = attribute_value(
        resource,
        "container_id",
        "runtime_id",
    )

    candidate = (
        normalized_identifier(value)
        if value is not None
        else None
    )

    if candidate:
        return candidate

    if (
        resource.resource_type
        == ResourceType.CONTAINER.value
    ):
        provider_id = (
            normalized_identifier(
                resource.provider_id
            )
        )

        if (
            provider_id
            and CONTAINER_ID_PATTERN.match(
                provider_id
            )
        ):
            return provider_id

    return None


def resource_ips(
    resource: ResourceIdentity,
) -> set[str]:
    """
    Collect normalized IP addresses from selected
    resource attributes.
    """

    candidates: List[Any] = []

    for key in (
        "ip",
        "ip_address",
        "host_ip",
        "internal_ip",
        "external_ip",
        "fixed_ip",
        "floating_ip",
    ):
        value = attribute_value(
            resource,
            key,
        )

        if value is not None:
            candidates.append(value)

    addresses = attribute_value(
        resource,
        "addresses",
        "fixed_ips",
    )

    if isinstance(
        addresses,
        Mapping,
    ):
        candidates.extend(
            addresses.values()
        )

    elif isinstance(
        addresses,
        Sequence,
    ) and not isinstance(
        addresses,
        (
            str,
            bytes,
        ),
    ):
        candidates.extend(addresses)

    output: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(
            value,
            Mapping,
        ):
            for item in value.values():
                visit(item)

            return

        if isinstance(
            value,
            Sequence,
        ) and not isinstance(
            value,
            (
                str,
                bytes,
            ),
        ):
            for item in value:
                visit(item)

            return

        address = normalized_ip(value)

        if address:
            output.add(address)

    for candidate in candidates:
        visit(candidate)

    return output


def resource_types_compatible(
    left: ResourceIdentity,
    right: ResourceIdentity,
) -> bool:
    """
    Return whether two resource types could refer
    to the same underlying object.

    A Kubernetes node and Linux host may represent
    the same machine. An OpenStack VM and Linux
    host may also represent the same machine after
    the guest has been observed.

    A pod and a VM are not the same resource.
    """

    if (
        left.resource_type
        == right.resource_type
    ):
        return True

    compatible_groups = (
        {
            ResourceType.HOST.value,
            ResourceType.NODE.value,
            ResourceType.VM.value,
        },
        {
            ResourceType.CONTAINER.value,
        },
        {
            ResourceType.SERVICE.value,
            ResourceType.APPLICATION.value,
        },
    )

    pair = {
        left.resource_type,
        right.resource_type,
    }

    return any(
        pair.issubset(group)
        for group
        in compatible_groups
    )


def add_matching_evidence(
    evidence: List[IdentityEvidence],
    *,
    evidence_type: str,
    left_value: Any,
    right_value: Any,
    description: str,
) -> None:
    left_text = normalized_identifier(
        left_value
    )

    right_text = normalized_identifier(
        right_value
    )

    if (
        left_text is None
        or right_text is None
        or left_text != right_text
    ):
        return

    evidence.append(
        IdentityEvidence(
            evidence_type=(
                evidence_type
            ),
            score=(
                EVIDENCE_WEIGHTS[
                    evidence_type
                ]
            ),
            left_value=left_text,
            right_value=right_text,
            description=description,
        )
    )


def identity_evidence(
    left: ResourceIdentity,
    right: ResourceIdentity,
) -> tuple[
    List[IdentityEvidence],
    List[str],
]:
    """
    Build positive evidence and conflicts for two
    normalized resources.
    """

    evidence: List[
        IdentityEvidence
    ] = []

    conflicts: List[str] = []

    if not resource_types_compatible(
        left,
        right,
    ):
        conflicts.append(
            "incompatible_resource_types"
        )

        return (
            evidence,
            conflicts,
        )

    if (
        left.canonical_id
        and right.canonical_id
    ):
        if (
            left.canonical_id
            == right.canonical_id
        ):
            add_matching_evidence(
                evidence,
                evidence_type=(
                    "same_canonical_id"
                ),
                left_value=(
                    left.canonical_id
                ),
                right_value=(
                    right.canonical_id
                ),
                description=(
                    "Resources already share "
                    "a canonical identity"
                ),
            )
        else:
            conflicts.append(
                "different_canonical_ids"
            )

    # A provider-native ID match is strong only
    # within the same provider and environment.
    if (
        left.provider
        == right.provider
        and left.environment_id
        == right.environment_id
    ):
        add_matching_evidence(
            evidence,
            evidence_type=(
                "same_provider_id"
            ),
            left_value=(
                left.provider_id
            ),
            right_value=(
                right.provider_id
            ),
            description=(
                "Provider IDs match within "
                "the same environment"
            ),
        )

    add_matching_evidence(
        evidence,
        evidence_type=(
            "same_machine_id"
        ),
        left_value=machine_id(left),
        right_value=machine_id(right),
        description=(
            "Machine IDs match"
        ),
    )

    add_matching_evidence(
        evidence,
        evidence_type=(
            "same_system_uuid"
        ),
        left_value=system_uuid(left),
        right_value=system_uuid(
            right
        ),
        description=(
            "System UUIDs match"
        ),
    )

    left_uid = resource_uid(left)
    right_uid = resource_uid(right)

    if (
        left_uid
        and right_uid
        and left_uid == right_uid
    ):
        evidence_type = (
            "same_kubernetes_uid"
        )

        if (
            left.cluster_id
            and right.cluster_id
            and left.cluster_id
            == right.cluster_id
        ):
            evidence_type = (
                "same_cluster_and_uid"
            )

        add_matching_evidence(
            evidence,
            evidence_type=evidence_type,
            left_value=left_uid,
            right_value=right_uid,
            description=(
                "Kubernetes object UIDs "
                "match"
            ),
        )

    add_matching_evidence(
        evidence,
        evidence_type=(
            "same_cloud_instance_id"
        ),
        left_value=(
            cloud_instance_id(left)
        ),
        right_value=(
            cloud_instance_id(
                right
            )
        ),
        description=(
            "Cloud instance IDs match"
        ),
    )

    add_matching_evidence(
        evidence,
        evidence_type=(
            "same_container_id"
        ),
        left_value=(
            container_id(left)
        ),
        right_value=(
            container_id(right)
        ),
        description=(
            "Container runtime IDs match"
        ),
    )

    left_host = normalized_hostname(
        left.host or left.name
    )

    right_host = normalized_hostname(
        right.host or right.name
    )

    if (
        left_host
        and right_host
        and left_host == right_host
    ):
        if (
            left.environment_id
            == right.environment_id
        ):
            evidence.append(
                IdentityEvidence(
                    evidence_type=(
                        "same_environment_host"
                    ),
                    score=(
                        EVIDENCE_WEIGHTS[
                            "same_environment_host"
                        ]
                    ),
                    left_value=left_host,
                    right_value=(
                        right_host
                    ),
                    description=(
                        "Hostnames match in "
                        "the same environment"
                    ),
                )
            )

        if (
            left.cluster_id
            and right.cluster_id
            and left.cluster_id
            == right.cluster_id
        ):
            evidence.append(
                IdentityEvidence(
                    evidence_type=(
                        "same_cluster_node"
                    ),
                    score=(
                        EVIDENCE_WEIGHTS[
                            "same_cluster_node"
                        ]
                    ),
                    left_value=left_host,
                    right_value=(
                        right_host
                    ),
                    description=(
                        "Node/host names match "
                        "inside the same cluster"
                    ),
                )
            )

        if (
            left.region
            and right.region
            and left.region
            == right.region
        ):
            evidence.append(
                IdentityEvidence(
                    evidence_type=(
                        "same_region_host"
                    ),
                    score=(
                        EVIDENCE_WEIGHTS[
                            "same_region_host"
                        ]
                    ),
                    left_value=left_host,
                    right_value=(
                        right_host
                    ),
                    description=(
                        "Hostnames match in "
                        "the same region"
                    ),
                )
            )

    left_name = normalized_text(
        left.name
    )

    right_name = normalized_text(
        right.name
    )

    if (
        left_name
        and right_name
        and left_name == right_name
    ):
        if (
            left.namespace
            and right.namespace
            and left.namespace
            == right.namespace
            and left.resource_type
            == right.resource_type
        ):
            evidence.append(
                IdentityEvidence(
                    evidence_type=(
                        "same_namespace_name_type"
                    ),
                    score=(
                        EVIDENCE_WEIGHTS[
                            "same_namespace_name_type"
                        ]
                    ),
                    left_value=left_name,
                    right_value=(
                        right_name
                    ),
                    description=(
                        "Names and types match "
                        "inside the same "
                        "namespace"
                    ),
                )
            )
        else:
            evidence.append(
                IdentityEvidence(
                    evidence_type=(
                        "same_name"
                    ),
                    score=(
                        EVIDENCE_WEIGHTS[
                            "same_name"
                        ]
                    ),
                    left_value=left_name,
                    right_value=(
                        right_name
                    ),
                    description=(
                        "Display names match"
                    ),
                )
            )

    shared_ips = (
        resource_ips(left)
        & resource_ips(right)
    )

    if shared_ips:
        evidence.append(
            IdentityEvidence(
                evidence_type=(
                    "same_ip_address"
                ),
                score=(
                    EVIDENCE_WEIGHTS[
                        "same_ip_address"
                    ]
                ),
                left_value=(
                    ",".join(
                        sorted(
                            shared_ips
                        )
                    )
                ),
                right_value=(
                    ",".join(
                        sorted(
                            shared_ips
                        )
                    )
                ),
                description=(
                    "Resources share one or "
                    "more IP addresses"
                ),
                metadata={
                    "shared_ips": sorted(
                        shared_ips
                    )
                },
            )
        )

    return (
        evidence,
        conflicts,
    )


def combine_evidence_score(
    evidence: Iterable[
        IdentityEvidence
    ],
) -> float:
    """
    Combine independent evidence scores without
    simply adding them beyond 1.0.

    Formula:

        1 - product(1 - score)

    This produces stronger confidence when
    multiple pieces of evidence agree.
    """

    remaining = 1.0

    for item in evidence:
        score = min(
            1.0,
            max(
                0.0,
                float(item.score),
            ),
        )

        remaining *= (
            1.0 - score
        )

    return round(
        1.0 - remaining,
        6,
    )


def canonical_identity(
    resources: Iterable[
        ResourceIdentity
    ],
) -> str:
    """
    Produce a deterministic canonical identity
    from provider-scoped resource IDs.
    """

    identities = sorted(
        resource.effective_id()
        for resource in resources
    )

    encoded = json.dumps(
        identities,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256(
        encoded
    ).hexdigest()

    return (
        "osha:resource:"
        + digest
    )


def compare_resources(
    left: ResourceIdentity,
    right: ResourceIdentity,
    *,
    auto_merge_threshold: float = (
        DEFAULT_AUTO_MERGE_THRESHOLD
    ),
    candidate_threshold: float = (
        DEFAULT_CANDIDATE_THRESHOLD
    ),
) -> IdentityMatch:
    """
    Compare two resources and assign one of:

        merge
        candidate
        distinct

    `merge` still means the resolver found enough
    configured evidence. It does not alter either
    resource automatically.
    """

    evidence, conflicts = (
        identity_evidence(
            left,
            right,
        )
    )

    score = combine_evidence_score(
        evidence
    )

    if conflicts:
        decision = "distinct"

    elif score >= auto_merge_threshold:
        decision = "merge"

    elif score >= candidate_threshold:
        decision = "candidate"

    else:
        decision = "distinct"

    canonical_id = None

    if decision == "merge":
        canonical_id = (
            canonical_identity(
                [
                    left,
                    right,
                ]
            )
        )

    return IdentityMatch(
        left_resource_id=(
            left.effective_id()
        ),
        right_resource_id=(
            right.effective_id()
        ),
        score=score,
        decision=decision,
        evidence=evidence,
        conflicts=conflicts,
        canonical_id=(
            canonical_id
        ),
    )


def apply_canonical_id(
    resource: ResourceIdentity,
    canonical_id: str,
) -> ResourceIdentity:
    """
    Assign a canonical ID to a normalized
    resource.

    ResourceIdentity is mutable, but returning it
    makes pipeline use explicit.
    """

    resource.canonical_id = (
        canonical_id
    )

    return resource


class IdentityResolver:
    """
    In-memory conservative resource identity
    resolver.

    This class is suitable for normalization
    batches and development.

    A persistent production registry can later
    wrap the same comparison functions.
    """

    def __init__(
        self,
        *,
        auto_merge_threshold: float = (
            DEFAULT_AUTO_MERGE_THRESHOLD
        ),
        candidate_threshold: float = (
            DEFAULT_CANDIDATE_THRESHOLD
        ),
    ) -> None:
        self.auto_merge_threshold = (
            auto_merge_threshold
        )

        self.candidate_threshold = (
            candidate_threshold
        )

        self._resources: Dict[
            str,
            ResourceIdentity,
        ] = {}

        self._canonical_members: Dict[
            str,
            set[str],
        ] = {}

        self._matches: List[
            IdentityMatch
        ] = []

    def resources(
        self,
    ) -> List[ResourceIdentity]:
        return list(
            self._resources.values()
        )

    def matches(
        self,
    ) -> List[IdentityMatch]:
        return list(
            self._matches
        )

    def register(
        self,
        resource: ResourceIdentity,
    ) -> tuple[
        ResourceIdentity,
        List[IdentityMatch],
    ]:
        """
        Register one resource, compare it with
        existing resources, and apply a canonical
        ID only when merge evidence reaches the
        configured threshold.
        """

        resource_id = (
            resource.effective_id()
        )

        previous = list(
            self._resources.values()
        )

        new_matches: List[
            IdentityMatch
        ] = []

        merge_resources = [
            resource
        ]

        for existing in previous:
            match = compare_resources(
                resource,
                existing,
                auto_merge_threshold=(
                    self.
                    auto_merge_threshold
                ),
                candidate_threshold=(
                    self.
                    candidate_threshold
                ),
            )

            if (
                match.decision
                != "distinct"
            ):
                new_matches.append(
                    match
                )

                self._matches.append(
                    match
                )

            if match.decision == "merge":
                merge_resources.append(
                    existing
                )

        if len(merge_resources) > 1:
            existing_canonical_ids = {
                item.canonical_id
                for item
                in merge_resources
                if item.canonical_id
            }

            if (
                len(
                    existing_canonical_ids
                )
                == 1
            ):
                canonical_id = next(
                    iter(
                        existing_canonical_ids
                    )
                )
            else:
                canonical_id = (
                    canonical_identity(
                        merge_resources
                    )
                )

            for member in (
                merge_resources
            ):
                before = (
                    member.effective_id()
                )

                apply_canonical_id(
                    member,
                    canonical_id,
                )

                self._canonical_members.setdefault(
                    canonical_id,
                    set(),
                ).add(before)

            resource.canonical_id = (
                canonical_id
            )

        self._resources[
            resource_id
        ] = resource

        return (
            resource,
            new_matches,
        )

    def resolve_batch(
        self,
        resources: Iterable[
            ResourceIdentity
        ],
    ) -> List[
        ResourceIdentity
    ]:
        resolved = []

        for resource in resources:
            resolved_resource, _ = (
                self.register(
                    resource
                )
            )

            resolved.append(
                resolved_resource
            )

        return resolved

    def candidates(
        self,
    ) -> List[IdentityMatch]:
        return [
            match
            for match in self._matches
            if (
                match.decision
                == "candidate"
            )
        ]

    def canonical_groups(
        self,
    ) -> List[IdentityCluster]:
        groups: List[
            IdentityCluster
        ] = []

        by_canonical: Dict[
            str,
            List[ResourceIdentity],
        ] = {}

        for resource in (
            self._resources.values()
        ):
            if not resource.canonical_id:
                continue

            by_canonical.setdefault(
                resource.canonical_id,
                [],
            ).append(resource)

        for (
            canonical_id,
            resources,
        ) in by_canonical.items():
            evidence = []

            relevant_scores = []

            resource_ids = {
                resource.effective_id()
                for resource
                in resources
            }

            for match in self._matches:
                if (
                    match.decision
                    != "merge"
                ):
                    continue

                if (
                    match.left_resource_id
                    in resource_ids
                    or match.right_resource_id
                    in resource_ids
                ):
                    evidence.extend(
                        match.evidence
                    )

                    relevant_scores.append(
                        match.score
                    )

            resource_types = {
                resource.resource_type
                for resource
                in resources
            }

            resource_type = (
                next(
                    iter(
                        resource_types
                    )
                )
                if len(resource_types) == 1
                else "compute_resource"
            )

            confidence = (
                min(relevant_scores)
                if relevant_scores
                else 1.0
            )

            groups.append(
                IdentityCluster(
                    canonical_id=(
                        canonical_id
                    ),
                    resource_ids=sorted(
                        resource_ids
                    ),
                    resource_type=(
                        resource_type
                    ),
                    confidence=(
                        confidence
                    ),
                    evidence=evidence,
                )
            )

        return groups

    def get(
        self,
        resource_id: str,
    ) -> Optional[
        ResourceIdentity
    ]:
        return self._resources.get(
            resource_id
        )

    def clear(self) -> None:
        self._resources.clear()

        self._canonical_members.clear()

        self._matches.clear()
