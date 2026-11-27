"""
OSHA Resource Repository.

Persistence interface for infrastructure Resources discovered by OSHA.

Resources represent infrastructure objects such as:

    - Linux hosts
    - CPUs
    - disks
    - network interfaces
    - processes
    - system services
    - Kubernetes clusters
    - Kubernetes nodes
    - namespaces
    - pods
    - containers
    - OpenStack instances
    - storage volumes
    - Ceph components

The repository provides tenant-aware and source-aware resource lookup,
discovery upsert operations, hierarchy queries and lifecycle management.

Transactions remain owned by OSHA's session/service layer.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from ..models.resource import (
    Resource,
    ResourceLifecycleState,
    ResourceStatus,
)
from .base import BaseRepository


class ResourceRepository(BaseRepository[Resource]):
    """Repository for OSHA infrastructure resources."""

    model = Resource

    def __init__(
        self,
        session: Session,
    ) -> None:
        super().__init__(session)

    # ------------------------------------------------------------------
    # Identity lookup
    # ------------------------------------------------------------------

    def get_by_external_id(
        self,
        *,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        resource_type: str,
        external_id: str,
    ) -> Resource | None:
        """
        Return a resource by its source-scoped external identity.
        """
        statement = (
            select(Resource)
            .where(
                Resource.tenant_id == tenant_id,
                Resource.source_id == source_id,
                Resource.resource_type == resource_type,
                Resource.external_id == external_id,
            )
            .limit(1)
        )

        return self.session.scalar(statement)

    def get_by_name(
        self,
        *,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        name: str,
        resource_type: str | None = None,
    ) -> Resource | None:
        """Return a resource by name within a source."""
        statement = select(Resource).where(
            Resource.tenant_id == tenant_id,
            Resource.source_id == source_id,
            Resource.name == name,
        )

        if resource_type is not None:
            statement = statement.where(
                Resource.resource_type == resource_type
            )

        return self.session.scalar(
            statement.limit(1)
        )

    # ------------------------------------------------------------------
    # Source queries
    # ------------------------------------------------------------------

    def list_for_source(
        self,
        *,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
        include_deleted: bool = False,
    ) -> Sequence:
        """Return resources discovered from a source."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(Resource).where(
            Resource.tenant_id == tenant_id,
            Resource.source_id == source_id,
        )

        if not include_deleted:
            statement = statement.where(
                Resource.lifecycle_state
                != ResourceLifecycleState.DELETED.value
            )

        statement = (
            statement
            .order_by(
                Resource.resource_type.asc(),
                Resource.name.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    def list_by_type(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_type: str,
        source_id: uuid.UUID | None = None,
        offset: int = 0,
        limit: int = 100,
        include_deleted: bool = False,
    ) -> Sequence:
        """
        Return resources of a particular infrastructure type.

        A source can optionally be supplied to restrict the query.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(Resource).where(
            Resource.tenant_id == tenant_id,
            Resource.resource_type == resource_type,
        )

        if source_id is not None:
            statement = statement.where(
                Resource.source_id == source_id
            )

        if not include_deleted:
            statement = statement.where(
                Resource.lifecycle_state
                != ResourceLifecycleState.DELETED.value
            )

        statement = (
            statement
            .order_by(
                Resource.name.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    def list_roots(
        self,
        *,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """
        Return resources with no hierarchical parent.

        Examples might include a Linux host or Kubernetes cluster depending
        on the collector's resource model.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Resource)
            .where(
                Resource.tenant_id == tenant_id,
                Resource.source_id == source_id,
                Resource.parent_id.is_(None),
                Resource.lifecycle_state
                != ResourceLifecycleState.DELETED.value,
            )
            .order_by(
                Resource.name.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    def list_children(
        self,
        *,
        tenant_id: uuid.UUID,
        parent_id: uuid.UUID,
        resource_type: str | None = None,
        offset: int = 0,
        limit: int = 100,
        include_deleted: bool = False,
    ) -> Sequence:
        """Return direct children of an infrastructure resource."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(Resource).where(
            Resource.tenant_id == tenant_id,
            Resource.parent_id == parent_id,
        )

        if resource_type is not None:
            statement = statement.where(
                Resource.resource_type == resource_type
            )

        if not include_deleted:
            statement = statement.where(
                Resource.lifecycle_state
                != ResourceLifecycleState.DELETED.value
            )

        statement = (
            statement
            .order_by(
                Resource.resource_type.asc(),
                Resource.name.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    def get_with_children(
        self,
        *,
        resource_id: uuid.UUID,
    ) -> Resource | None:
        """
        Return a resource with its direct children eagerly loaded.

        This intentionally loads only one hierarchy level. Deeper topology
        traversal belongs in the topology service rather than recursively
        loading an unbounded resource tree.
        """
        statement = (
            select(Resource)
            .options(
                selectinload(Resource.children)
            )
            .where(
                Resource.id == resource_id
            )
            .limit(1)
        )

        return self.session.scalar(statement)

    # ------------------------------------------------------------------
    # Infrastructure coordinates
    # ------------------------------------------------------------------

    def find_by_hostname(
        self,
        *,
        tenant_id: uuid.UUID,
        hostname: str,
        source_id: uuid.UUID | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Find resources associated with a hostname."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(Resource).where(
            Resource.tenant_id == tenant_id,
            Resource.hostname == hostname,
            Resource.lifecycle_state
            != ResourceLifecycleState.DELETED.value,
        )

        if source_id is not None:
            statement = statement.where(
                Resource.source_id == source_id
            )

        statement = (
            statement
            .order_by(
                Resource.name.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    def find_by_ip(
        self,
        *,
        tenant_id: uuid.UUID,
        ip_address: str,
        source_id: uuid.UUID | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Find resources associated with an IP address."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(Resource).where(
            Resource.tenant_id == tenant_id,
            Resource.ip_address == ip_address,
            Resource.lifecycle_state
            != ResourceLifecycleState.DELETED.value,
        )

        if source_id is not None:
            statement = statement.where(
                Resource.source_id == source_id
            )

        statement = (
            statement
            .order_by(
                Resource.resource_type.asc(),
                Resource.name.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    def find_by_namespace(
        self,
        *,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        namespace: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return resources belonging to a namespace."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Resource)
            .where(
                Resource.tenant_id == tenant_id,
                Resource.source_id == source_id,
                Resource.namespace == namespace,
                Resource.lifecycle_state
                != ResourceLifecycleState.DELETED.value,
            )
            .order_by(
                Resource.resource_type.asc(),
                Resource.name.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Health queries
    # ------------------------------------------------------------------

    def list_by_status(
        self,
        *,
        tenant_id: uuid.UUID,
        status: str,
        source_id: uuid.UUID | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return resources with a particular operational status."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(Resource).where(
            Resource.tenant_id == tenant_id,
            Resource.status == status,
        )

        if source_id is not None:
            statement = statement.where(
                Resource.source_id == source_id
            )

        statement = (
            statement
            .order_by(
                Resource.last_seen_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Resource counts
    # ------------------------------------------------------------------

    def count_for_source(
        self,
        *,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        resource_type: str | None = None,
        include_deleted: bool = False,
    ) -> int:
        """Count resources discovered from a source."""
        statement = (
            select(func.count(Resource.id))
            .where(
                Resource.tenant_id == tenant_id,
                Resource.source_id == source_id,
            )
        )

        if resource_type is not None:
            statement = statement.where(
                Resource.resource_type == resource_type
            )

        if not include_deleted:
            statement = statement.where(
                Resource.lifecycle_state
                != ResourceLifecycleState.DELETED.value
            )

        result = self.session.scalar(statement)

        return int(result or 0)

    # ------------------------------------------------------------------
    # Discovery upsert
    # ------------------------------------------------------------------

    def upsert_discovered(
        self,
        *,
        tenant_id: uuid.UUID,
        source_id: uuid.UUID,
        resource_type: str,
        external_id: str,
        name: str,
        seen_at: datetime,
        parent_id: uuid.UUID | None = None,
        display_name: str | None = None,
        hostname: str | None = None,
        ip_address: str | None = None,
        namespace: str | None = None,
        cluster_name: str | None = None,
        region: str | None = None,
        zone: str | None = None,
        environment: str | None = None,
        collector_id: uuid.UUID | None = None,
        collector_name: str | None = None,
        discovery_method: str | None = None,
        labels: dict[str, Any] | None = None,
        annotations: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        capabilities: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> tuple[Resource, bool]:
        """
        Create or refresh a discovered resource.

        Identity is determined by:

            tenant_id
            source_id
            resource_type
            external_id

        Returns:
            (resource, created)

        The Session is flushed but not committed.
        """
        resource = self.get_by_external_id(
            tenant_id=tenant_id,
            source_id=source_id,
            resource_type=resource_type,
            external_id=external_id,
        )

        if resource is None:
            resource = Resource(
                tenant_id=tenant_id,
                source_id=source_id,
                resource_type=resource_type,
                external_id=external_id,
                name=name,
                parent_id=parent_id,
                display_name=display_name,
                hostname=hostname,
                ip_address=ip_address,
                namespace=namespace,
                cluster_name=cluster_name,
                region=region,
                zone=zone,
                environment=environment,
                collector_id=collector_id,
                collector_name=collector_name,
                discovery_method=discovery_method,
                labels=labels or {},
                annotations=annotations or {},
                attributes=attributes or {},
                capabilities=capabilities or {},
                description=description,
                status=ResourceStatus.DISCOVERED.value,
                lifecycle_state=ResourceLifecycleState.ACTIVE.value,
                monitored=True,
            )

            resource.mark_seen(
                seen_at=seen_at
            )

            self.add(
                resource,
                flush=True,
            )

            return resource, True

        # Refresh mutable discovery information.
        resource.name = name

        if parent_id is not None:
            resource.parent_id = parent_id

        if display_name is not None:
            resource.display_name = display_name

        if hostname is not None:
            resource.hostname = hostname

        if ip_address is not None:
            resource.ip_address = ip_address

        if namespace is not None:
            resource.namespace = namespace

        if cluster_name is not None:
            resource.cluster_name = cluster_name

        if region is not None:
            resource.region = region

        if zone is not None:
            resource.zone = zone

        if environment is not None:
            resource.environment = environment

        if collector_id is not None:
            resource.collector_id = collector_id

        if collector_name is not None:
            resource.collector_name = collector_name

        if discovery_method is not None:
            resource.discovery_method = discovery_method

        if labels is not None:
            resource.labels = labels

        if annotations is not None:
            resource.annotations = annotations

        if attributes is not None:
            resource.attributes = attributes

        if capabilities is not None:
            resource.capabilities = capabilities

        if description is not None:
            resource.description = description

        resource.monitored = True

        resource.mark_seen(
            seen_at=seen_at
        )

        self.flush()

        return resource, False

    # ------------------------------------------------------------------
    # Resource lifecycle
    # ------------------------------------------------------------------

    def mark_healthy(
        self,
        *,
        resource_id: uuid.UUID,
        seen_at: datetime | None = None,
    ) -> Resource:
        """Mark a resource healthy."""
        resource = self.get_required(
            resource_id
        )

        resource.mark_healthy(
            seen_at=seen_at
        )

        self.flush()

        return resource

    def mark_degraded(
        self,
        *,
        resource_id: uuid.UUID,
        message: str | None = None,
    ) -> Resource:
        """Mark a resource degraded."""
        resource = self.get_required(
            resource_id
        )

        resource.mark_degraded(
            message=message
        )

        self.flush()

        return resource

    def mark_failed(
        self,
        *,
        resource_id: uuid.UUID,
        message: str | None = None,
    ) -> Resource:
        """Mark a resource failed."""
        resource = self.get_required(
            resource_id
        )

        resource.mark_failed(
            message=message
        )

        self.flush()

        return resource

    def mark_stale(
        self,
        *,
        resource_id: uuid.UUID,
    ) -> Resource:
        """Mark a resource stale."""
        resource = self.get_required(
            resource_id
        )

        resource.mark_stale()

        self.flush()

        return resource

    def mark_deleted(
        self,
        *,
        resource_id: uuid.UUID,
        deleted_at: datetime,
    ) -> Resource:
        """
        Soft-delete an infrastructure resource.

        Physical deletion is deliberately avoided so historical anomalies,
        incidents, predictions and telemetry can continue referring to the
        infrastructure identity.
        """
        resource = self.get_required(
            resource_id
        )

        resource.mark_deleted(
            deleted_at=deleted_at
        )

        self.flush()

        return resource

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def set_monitored(
        self,
        *,
        resource_id: uuid.UUID,
        monitored: bool,
    ) -> Resource:
        """Enable or disable monitoring for a resource."""
        resource = self.get_required(
            resource_id
        )

        resource.monitored = monitored

        self.flush()

        return resource


__all__ = [
    "ResourceRepository",
]
