"""
OSHA Infrastructure Discovery Service.

Coordinates persistence of infrastructure discovered by OSHA collectors.

The service sits between collector output and the repository layer:

    Linux / Kubernetes / OpenStack / Ceph Collector
                        |
                        v
                 DiscoveryService
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
    SourceRepository ResourceRepository RelationshipRepository
          |             |             |
          +-------------+-------------+
                        |
                        v
                    PostgreSQL

Collectors should describe what they discovered.

They should not need to understand SQLAlchemy, database sessions, table
layouts or repository implementation details.

The DiscoveryService coordinates:

    - source registration / refresh
    - resource discovery / refresh
    - parent-child hierarchy
    - topology relationships
    - discovery-cycle tracking
    - stale-resource detection
    - stale-relationship detection
    - source health state

Transaction ownership remains outside this service. The caller should run
discovery operations inside OSHA's session_scope() when atomic persistence
is required.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from ..models.relationship import (
    RelationshipDiscoveryMethod,
    ResourceRelationship,
)
from ..models.resource import Resource
from ..models.source import Source
from ..repositories.relationship import (
    ResourceRelationshipRepository,
)
from ..repositories.resource import ResourceRepository
from ..repositories.source import SourceRepository


# ---------------------------------------------------------------------------
# Discovery input structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SourceDiscovery:
    """
    Description of an infrastructure source discovered by a collector.

    Example:

        Linux host
        Kubernetes cluster
        OpenStack deployment
        Ceph cluster
    """

    tenant_id: uuid.UUID
    source_type: str
    external_id: str
    name: str
    seen_at: datetime

    display_name: str | None = None

    hostname: str | None = None
    ip_address: str | None = None
    address: str | None = None
    port: int | None = None

    environment: str | None = None
    region: str | None = None
    zone: str | None = None

    collector_id: uuid.UUID | None = None
    collector_name: str | None = None
    collector_version: str | None = None

    labels: dict[str, Any] | None = None
    attributes: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ResourceDiscovery:
    """
    Description of a resource discovered beneath an OSHA Source.

    Examples:

        Linux host
        process
        disk
        network interface

        Kubernetes node
        namespace
        pod
        container

        OpenStack instance
        network
        volume
    """

    resource_type: str
    external_id: str
    name: str

    parent_external_id: str | None = None
    parent_resource_type: str | None = None

    display_name: str | None = None

    hostname: str | None = None
    ip_address: str | None = None

    namespace: str | None = None
    cluster_name: str | None = None

    region: str | None = None
    zone: str | None = None
    environment: str | None = None

    collector_id: uuid.UUID | None = None
    collector_name: str | None = None
    discovery_method: str | None = None

    labels: dict[str, Any] | None = None
    annotations: dict[str, Any] | None = None
    attributes: dict[str, Any] | None = None
    capabilities: dict[str, Any] | None = None

    description: str | None = None


@dataclass(frozen=True, slots=True)
class RelationshipDiscovery:
    """
    Description of a topology edge discovered by a collector.

    Resource references use external identities within the same Source.

    Example:

        pod/payment-api
            RUNS_ON
        node/worker-01
    """

    source_resource_type: str
    source_external_id: str

    target_resource_type: str
    target_external_id: str

    relationship_type: str

    discovery_method: str = (
        RelationshipDiscoveryMethod.COLLECTOR.value
    )

    confidence: float = 1.0

    collector_id: uuid.UUID | None = None
    correlation_id: uuid.UUID | None = None

    custom_type: str | None = None

    attributes: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    labels: dict[str, Any] | None = None

    description: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Summary of a completed discovery persistence operation."""

    source: Source

    source_created: bool

    resources_created: int
    resources_updated: int

    relationships_created: int
    relationships_updated: int

    resources_stale: int = 0
    relationships_stale: int = 0


# ---------------------------------------------------------------------------
# Discovery errors
# ---------------------------------------------------------------------------

class DiscoveryServiceError(RuntimeError):
    """Base error raised by the discovery service."""


class DiscoveryParentNotFoundError(DiscoveryServiceError):
    """Raised when a resource references an unknown parent resource."""


class DiscoveryResourceNotFoundError(DiscoveryServiceError):
    """Raised when a relationship references an unknown resource."""


class DiscoveryValidationError(DiscoveryServiceError):
    """Raised when a discovery payload is structurally invalid."""


# ---------------------------------------------------------------------------
# Discovery service
# ---------------------------------------------------------------------------

class DiscoveryService:
    """
    Coordinate collector discovery persistence.

    A service instance is bound to one SQLAlchemy Session.

    Repositories share that same Session so an entire discovery operation
    can participate in the same transaction.
    """

    def __init__(
        self,
        session: Session,
    ) -> None:
        if session is None:
            raise ValueError(
                "DiscoveryService requires an active Session"
            )

        self.session = session

        self.sources = SourceRepository(
            session
        )

        self.resources = ResourceRepository(
            session
        )

        self.relationships = ResourceRelationshipRepository(
            session
        )

    # ------------------------------------------------------------------
    # Source discovery
    # ------------------------------------------------------------------

    def register_source(
        self,
        discovery: SourceDiscovery,
    ) -> tuple[Source, bool]:
        """
        Create or refresh an infrastructure source.

        Returns:
            (source, created)
        """
        self._validate_source(
            discovery
        )

        return self.sources.upsert_discovered(
            tenant_id=discovery.tenant_id,
            source_type=discovery.source_type,
            external_id=discovery.external_id,
            name=discovery.name,
            seen_at=discovery.seen_at,
            display_name=discovery.display_name,
            hostname=discovery.hostname,
            ip_address=discovery.ip_address,
            address=discovery.address,
            port=discovery.port,
            environment=discovery.environment,
            region=discovery.region,
            zone=discovery.zone,
            collector_id=discovery.collector_id,
            collector_name=discovery.collector_name,
            collector_version=discovery.collector_version,
            labels=discovery.labels,
            attributes=discovery.attributes,
        )

    # ------------------------------------------------------------------
    # Resource lookup
    # ------------------------------------------------------------------

    def resolve_resource(
        self,
        *,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        resource_type: str,
        external_id: str,
    ) -> Resource | None:
        """Resolve a discovered infrastructure resource."""
        return self.resources.get_by_external_id(
            tenant_id=tenant_id,
            source_id=source_id,
            resource_type=resource_type,
            external_id=external_id,
        )

    def require_resource(
        self,
        *,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        resource_type: str,
        external_id: str,
    ) -> Resource:
        """Resolve a resource or raise a discovery error."""
        resource = self.resolve_resource(
            tenant_id=tenant_id,
            source_id=source_id,
            resource_type=resource_type,
            external_id=external_id,
        )

        if resource is None:
            raise DiscoveryResourceNotFoundError(
                "discovery resource was not found: "
                f"type={resource_type!r}, "
                f"external_id={external_id!r}"
            )

        return resource

    # ------------------------------------------------------------------
    # Resource discovery
    # ------------------------------------------------------------------

    def persist_resource(
        self,
        *,
        source: Source,
        discovery: ResourceDiscovery,
        seen_at: datetime,
    ) -> tuple[Resource, bool]:
        """
        Create or refresh one discovered resource.

        When a parent identity is supplied, the parent must already exist
        within the same source.
        """
        self._validate_resource(
            discovery
        )

        parent_id: uuid.UUID | None = None

        if discovery.parent_external_id is not None:
            parent_type = discovery.parent_resource_type

            if not parent_type:
                raise DiscoveryValidationError(
                    "parent_resource_type is required when "
                    "parent_external_id is supplied"
                )

            parent = self.resources.get_by_external_id(
                tenant_id=source.tenant_id,
                source_id=source.id,
                resource_type=parent_type,
                external_id=discovery.parent_external_id,
            )

            if parent is None:
                raise DiscoveryParentNotFoundError(
                    "resource parent was not found: "
                    f"type={parent_type!r}, "
                    f"external_id="
                    f"{discovery.parent_external_id!r}"
                )

            parent_id = parent.id

        return self.resources.upsert_discovered(
            tenant_id=source.tenant_id,
            source_id=source.id,
            resource_type=discovery.resource_type,
            external_id=discovery.external_id,
            name=discovery.name,
            seen_at=seen_at,
            parent_id=parent_id,
            display_name=discovery.display_name,
            hostname=discovery.hostname,
            ip_address=discovery.ip_address,
            namespace=discovery.namespace,
            cluster_name=discovery.cluster_name,
            region=discovery.region,
            zone=discovery.zone,
            environment=discovery.environment,
            collector_id=discovery.collector_id,
            collector_name=discovery.collector_name,
            discovery_method=discovery.discovery_method,
            labels=discovery.labels,
            annotations=discovery.annotations,
            attributes=discovery.attributes,
            capabilities=discovery.capabilities,
            description=discovery.description,
        )

    # ------------------------------------------------------------------
    # Relationship discovery
    # ------------------------------------------------------------------

    def persist_relationship(
        self,
        *,
        source: Source,
        discovery: RelationshipDiscovery,
        seen_at: datetime,
    ) -> tuple[ResourceRelationship, bool]:
        """Create or refresh one topology relationship."""
        self._validate_relationship(
            discovery
        )

        source_resource = self.require_resource(
            tenant_id=source.tenant_id,
            source_id=source.id,
            resource_type=discovery.source_resource_type,
            external_id=discovery.source_external_id,
        )

        target_resource = self.require_resource(
            tenant_id=source.tenant_id,
            source_id=source.id,
            resource_type=discovery.target_resource_type,
            external_id=discovery.target_external_id,
        )

        return self.relationships.upsert_relationship(
            tenant_id=source.tenant_id,
            source_resource_id=source_resource.id,
            target_resource_id=target_resource.id,
            relationship_type=discovery.relationship_type,
            seen_at=seen_at,
            discovery_method=discovery.discovery_method,
            collector_id=discovery.collector_id,
            correlation_id=discovery.correlation_id,
            confidence=discovery.confidence,
            custom_type=discovery.custom_type,
            attributes=discovery.attributes,
            evidence=discovery.evidence,
            labels=discovery.labels,
            description=discovery.description,
        )

    # ------------------------------------------------------------------
    # Complete discovery cycle
    # ------------------------------------------------------------------

    def persist_discovery(
        self,
        *,
        source_discovery: SourceDiscovery,
        resources: list[ResourceDiscovery],
        relationships: list[RelationshipDiscovery] | None = None,
        reconcile_missing: bool = False,
    ) -> DiscoveryResult:
        """
        Persist a complete discovery snapshot.

        Processing order:

            1. create/refresh Source
            2. persist resources
            3. persist topology relationships
            4. optionally mark previously known but missing entities stale
            5. mark Source healthy

        Parent resources must appear before their children in ``resources``.

        Example Kubernetes ordering:

            cluster
            node
            namespace
            deployment
            pod
            container

        The caller controls commit/rollback through the surrounding Session.
        """
        topology = relationships or []

        source, source_created = self.register_source(
            source_discovery
        )

        resources_created = 0
        resources_updated = 0

        relationships_created = 0
        relationships_updated = 0

        observed_resource_ids: set[uuid.UUID] = set()
        observed_relationship_ids: set[uuid.UUID] = set()

        for discovered_resource in resources:
            resource, created = self.persist_resource(
                source=source,
                discovery=discovered_resource,
                seen_at=source_discovery.seen_at,
            )

            observed_resource_ids.add(
                resource.id
            )

            if created:
                resources_created += 1
            else:
                resources_updated += 1

        for discovered_relationship in topology:
            relationship, created = self.persist_relationship(
                source=source,
                discovery=discovered_relationship,
                seen_at=source_discovery.seen_at,
            )

            observed_relationship_ids.add(
                relationship.id
            )

            if created:
                relationships_created += 1
            else:
                relationships_updated += 1

        resources_stale = 0
        relationships_stale = 0

        if reconcile_missing:
            resources_stale = self._mark_missing_resources_stale(
                source=source,
                observed_resource_ids=observed_resource_ids,
            )

            relationships_stale = (
                self._mark_missing_relationships_stale(
                    source=source,
                    observed_relationship_ids=observed_relationship_ids,
                )
            )

        self.sources.mark_healthy(
            source_id=source.id,
            checked_at=source_discovery.seen_at,
        )

        return DiscoveryResult(
            source=source,
            source_created=source_created,
            resources_created=resources_created,
            resources_updated=resources_updated,
            relationships_created=relationships_created,
            relationships_updated=relationships_updated,
            resources_stale=resources_stale,
            relationships_stale=relationships_stale,
        )

    # ------------------------------------------------------------------
    # Snapshot reconciliation
    # ------------------------------------------------------------------

    def _mark_missing_resources_stale(
        self,
        *,
        source: Source,
        observed_resource_ids: set[uuid.UUID],
    ) -> int:
        """
        Mark previously known resources missing from a complete snapshot
        as stale.

        This must only be used when the collector guarantees that the current
        discovery input is a complete source snapshot.
        """
        stale_count = 0
        offset = 0
        page_size = 500

        while True:
            known_resources = self.resources.list_for_source(
                tenant_id=source.tenant_id,
                source_id=source.id,
                offset=offset,
                limit=page_size,
                include_deleted=False,
            )

            if not known_resources:
                break

            for resource in known_resources:
                if resource.id in observed_resource_ids:
                    continue

                if resource.lifecycle_state == "stale":
                    continue

                resource.mark_stale()
                stale_count += 1

            if len(known_resources) < page_size:
                break

            offset += page_size

        if stale_count:
            self.session.flush()

        return stale_count

    def _mark_missing_relationships_stale(
        self,
        *,
        source: Source,
        observed_relationship_ids: set[uuid.UUID],
    ) -> int:
        """
        Mark topology edges associated with this source's resources stale
        when absent from a complete snapshot.

        Relationships are inspected through source resources to prevent
        unrelated tenant topology from being modified.
        """
        stale_count = 0

        offset = 0
        page_size = 500

        while True:
            source_resources = self.resources.list_for_source(
                tenant_id=source.tenant_id,
                source_id=source.id,
                offset=offset,
                limit=page_size,
                include_deleted=False,
            )

            if not source_resources:
                break

            for resource in source_resources:
                edges = self.relationships.list_outgoing(
                    tenant_id=source.tenant_id,
                    resource_id=resource.id,
                    include_inactive=False,
                    offset=0,
                    limit=1000,
                )

                for edge in edges:
                    if edge.id in observed_relationship_ids:
                        continue

                    if edge.status == "stale":
                        continue

                    edge.mark_stale()
                    stale_count += 1

            if len(source_resources) < page_size:
                break

            offset += page_size

        if stale_count:
            self.session.flush()

        return stale_count

    # ------------------------------------------------------------------
    # Source health
    # ------------------------------------------------------------------

    def mark_source_unreachable(
        self,
        *,
        source_id: uuid.UUID,
        message: str | None = None,
        error_code: str | None = None,
    ) -> Source:
        """
        Mark a discovery source unreachable.

        Intended for collectors that fail before a valid discovery snapshot
        can be obtained.
        """
        return self.sources.mark_unreachable(
            source_id=source_id,
            message=message,
            error_code=error_code,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_source(
        discovery: SourceDiscovery,
    ) -> None:
        """Validate source discovery input."""
        if not discovery.source_type.strip():
            raise DiscoveryValidationError(
                "source_type cannot be empty"
            )

        if not discovery.external_id.strip():
            raise DiscoveryValidationError(
                "source external_id cannot be empty"
            )

        if not discovery.name.strip():
            raise DiscoveryValidationError(
                "source name cannot be empty"
            )

        if (
            discovery.port is not None
            and not 1 <= discovery.port <= 65535
        ):
            raise DiscoveryValidationError(
                "source port must be between 1 and 65535"
            )

    @staticmethod
    def _validate_resource(
        discovery: ResourceDiscovery,
    ) -> None:
        """Validate resource discovery input."""
        if not discovery.resource_type.strip():
            raise DiscoveryValidationError(
                "resource_type cannot be empty"
            )

        if not discovery.external_id.strip():
            raise DiscoveryValidationError(
                "resource external_id cannot be empty"
            )

        if not discovery.name.strip():
            raise DiscoveryValidationError(
                "resource name cannot be empty"
            )

        if (
            discovery.parent_resource_type is not None
            and discovery.parent_external_id is None
        ):
            raise DiscoveryValidationError(
                "parent_external_id is required when "
                "parent_resource_type is specified"
            )

    @staticmethod
    def _validate_relationship(
        discovery: RelationshipDiscovery,
    ) -> None:
        """Validate topology relationship discovery input."""
        if not discovery.source_resource_type.strip():
            raise DiscoveryValidationError(
                "relationship source_resource_type cannot be empty"
            )

        if not discovery.source_external_id.strip():
            raise DiscoveryValidationError(
                "relationship source_external_id cannot be empty"
            )

        if not discovery.target_resource_type.strip():
            raise DiscoveryValidationError(
                "relationship target_resource_type cannot be empty"
            )

        if not discovery.target_external_id.strip():
            raise DiscoveryValidationError(
                "relationship target_external_id cannot be empty"
            )

        if not discovery.relationship_type.strip():
            raise DiscoveryValidationError(
                "relationship_type cannot be empty"
            )

        if not 0.0 <= discovery.confidence <= 1.0:
            raise DiscoveryValidationError(
                "relationship confidence must be between 0.0 and 1.0"
            )


__all__ = [
    "DiscoveryParentNotFoundError",
    "DiscoveryResourceNotFoundError",
    "DiscoveryResult",
    "DiscoveryService",
    "DiscoveryServiceError",
    "DiscoveryValidationError",
    "RelationshipDiscovery",
    "ResourceDiscovery",
    "SourceDiscovery",
]
