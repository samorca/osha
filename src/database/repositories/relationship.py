"""
OSHA Resource Relationship Repository.

Persistence interface for infrastructure topology relationships.

Examples:

    Pod
        RUNS_ON
    Kubernetes Node

    Kubernetes Node
        RUNS_ON
    Linux Host

    Application
        DEPENDS_ON
    Database Service

    Container
        MOUNTS
    Volume

    Service
        COMMUNICATES_WITH
    Service

The repository manages topology edges without owning transaction commits.
Transaction boundaries remain with OSHA's session/service layer.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, selectinload

from ..models.relationship import (
    RelationshipDiscoveryMethod,
    RelationshipStatus,
    ResourceRelationship,
)
from ..models.resource import Resource
from .base import BaseRepository


class ResourceRelationshipRepository(
    BaseRepository[ResourceRelationship]
):
    """Repository for OSHA infrastructure topology edges."""

    model = ResourceRelationship

    def __init__(
        self,
        session: Session,
    ) -> None:
        super().__init__(session)

    # ------------------------------------------------------------------
    # Edge lookup
    # ------------------------------------------------------------------

    def get_edge(
        self,
        *,
        tenant_id: uuid.UUID,
        source_resource_id: uuid.UUID,
        target_resource_id: uuid.UUID,
        relationship_type: str,
    ) -> ResourceRelationship | None:
        """
        Return a specific topology edge.

        Edge identity is:

            tenant
            source resource
            target resource
            relationship type
        """
        statement = (
            select(ResourceRelationship)
            .where(
                ResourceRelationship.tenant_id == tenant_id,
                ResourceRelationship.source_resource_id
                == source_resource_id,
                ResourceRelationship.target_resource_id
                == target_resource_id,
                ResourceRelationship.relationship_type
                == relationship_type,
            )
            .limit(1)
        )

        return self.session.scalar(statement)

    def get_with_resources(
        self,
        *,
        relationship_id: uuid.UUID,
    ) -> ResourceRelationship | None:
        """
        Return an edge with both endpoint resources eagerly loaded.
        """
        statement = (
            select(ResourceRelationship)
            .options(
                selectinload(
                    ResourceRelationship.source_resource
                ),
                selectinload(
                    ResourceRelationship.target_resource
                ),
            )
            .where(
                ResourceRelationship.id == relationship_id
            )
            .limit(1)
        )

        return self.session.scalar(statement)

    # ------------------------------------------------------------------
    # Outgoing topology
    # ------------------------------------------------------------------

    def list_outgoing(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        relationship_type: str | None = None,
        offset: int = 0,
        limit: int = 100,
        include_inactive: bool = False,
    ) -> Sequence:
        """
        Return edges originating from a resource.

        Example:

            pod
                -> RUNS_ON -> node
                -> USES -> volume
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(
            ResourceRelationship
        ).where(
            ResourceRelationship.tenant_id == tenant_id,
            ResourceRelationship.source_resource_id == resource_id,
        )

        if relationship_type is not None:
            statement = statement.where(
                ResourceRelationship.relationship_type
                == relationship_type
            )

        if not include_inactive:
            statement = statement.where(
                ResourceRelationship.active.is_(True),
                ResourceRelationship.status
                != RelationshipStatus.DELETED.value,
            )

        statement = (
            statement
            .options(
                selectinload(
                    ResourceRelationship.target_resource
                )
            )
            .order_by(
                ResourceRelationship.relationship_type.asc(),
                ResourceRelationship.created_at.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Incoming topology
    # ------------------------------------------------------------------

    def list_incoming(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        relationship_type: str | None = None,
        offset: int = 0,
        limit: int = 100,
        include_inactive: bool = False,
    ) -> Sequence:
        """
        Return edges targeting a resource.

        Example:

            node
                <- RUNS_ON <- pod-a
                <- RUNS_ON <- pod-b
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(
            ResourceRelationship
        ).where(
            ResourceRelationship.tenant_id == tenant_id,
            ResourceRelationship.target_resource_id == resource_id,
        )

        if relationship_type is not None:
            statement = statement.where(
                ResourceRelationship.relationship_type
                == relationship_type
            )

        if not include_inactive:
            statement = statement.where(
                ResourceRelationship.active.is_(True),
                ResourceRelationship.status
                != RelationshipStatus.DELETED.value,
            )

        statement = (
            statement
            .options(
                selectinload(
                    ResourceRelationship.source_resource
                )
            )
            .order_by(
                ResourceRelationship.relationship_type.asc(),
                ResourceRelationship.created_at.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Neighbours
    # ------------------------------------------------------------------

    def list_connected(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
        relationship_type: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """
        Return active edges connected to a resource in either direction.

        This provides one-hop topology access. Recursive or multi-hop graph
        traversal belongs in OSHA's topology service.
        """
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(
            ResourceRelationship
        ).where(
            ResourceRelationship.tenant_id == tenant_id,
            ResourceRelationship.active.is_(True),
            ResourceRelationship.status
            != RelationshipStatus.DELETED.value,
            or_(
                ResourceRelationship.source_resource_id == resource_id,
                ResourceRelationship.target_resource_id == resource_id,
            ),
        )

        if relationship_type is not None:
            statement = statement.where(
                ResourceRelationship.relationship_type
                == relationship_type
            )

        statement = (
            statement
            .options(
                selectinload(
                    ResourceRelationship.source_resource
                ),
                selectinload(
                    ResourceRelationship.target_resource
                ),
            )
            .order_by(
                ResourceRelationship.relationship_type.asc(),
                ResourceRelationship.created_at.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Relationship type
    # ------------------------------------------------------------------

    def list_by_type(
        self,
        *,
        tenant_id: uuid.UUID,
        relationship_type: str,
        offset: int = 0,
        limit: int = 100,
        include_inactive: bool = False,
    ) -> Sequence:
        """Return topology edges of a specific type."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = select(
            ResourceRelationship
        ).where(
            ResourceRelationship.tenant_id == tenant_id,
            ResourceRelationship.relationship_type
            == relationship_type,
        )

        if not include_inactive:
            statement = statement.where(
                ResourceRelationship.active.is_(True),
                ResourceRelationship.status
                != RelationshipStatus.DELETED.value,
            )

        statement = (
            statement
            .order_by(
                ResourceRelationship.last_seen_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Resource endpoint validation
    # ------------------------------------------------------------------

    def _require_resource(
        self,
        *,
        tenant_id: uuid.UUID,
        resource_id: uuid.UUID,
    ) -> Resource:
        """
        Return a tenant-owned endpoint resource.

        Relationship creation must not connect a tenant's topology to a
        resource owned by another tenant.
        """
        statement = (
            select(Resource)
            .where(
                Resource.id == resource_id,
                Resource.tenant_id == tenant_id,
            )
            .limit(1)
        )

        resource = self.session.scalar(statement)

        if resource is None:
            raise ValueError(
                f"resource {resource_id} does not exist "
                "for the specified tenant"
            )

        return resource

    # ------------------------------------------------------------------
    # Discovery upsert
    # ------------------------------------------------------------------

    def upsert_relationship(
        self,
        *,
        tenant_id: uuid.UUID,
        source_resource_id: uuid.UUID,
        target_resource_id: uuid.UUID,
        relationship_type: str,
        seen_at: datetime,
        discovery_method: str = (
            RelationshipDiscoveryMethod.COLLECTOR.value
        ),
        collector_id: uuid.UUID | None = None,
        correlation_id: uuid.UUID | None = None,
        confidence: float = 1.0,
        custom_type: str | None = None,
        attributes: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
        labels: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> tuple[ResourceRelationship, bool]:
        """
        Create or refresh a topology relationship.

        Returns:
            (relationship, created)

        The Session is flushed but not committed.
        """
        if source_resource_id == target_resource_id:
            raise ValueError(
                "topology relationship cannot reference "
                "the same resource as source and target"
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "relationship confidence must be between 0.0 and 1.0"
            )

        # Validate endpoint ownership before creating a topology edge.
        self._require_resource(
            tenant_id=tenant_id,
            resource_id=source_resource_id,
        )

        self._require_resource(
            tenant_id=tenant_id,
            resource_id=target_resource_id,
        )

        relationship = self.get_edge(
            tenant_id=tenant_id,
            source_resource_id=source_resource_id,
            target_resource_id=target_resource_id,
            relationship_type=relationship_type,
        )

        if relationship is None:
            relationship = ResourceRelationship(
                tenant_id=tenant_id,
                source_resource_id=source_resource_id,
                target_resource_id=target_resource_id,
                relationship_type=relationship_type,
                custom_type=custom_type,
                discovery_method=discovery_method,
                collector_id=collector_id,
                correlation_id=correlation_id,
                confidence=confidence,
                attributes=attributes or {},
                evidence=evidence or {},
                labels=labels or {},
                description=description,
                status=RelationshipStatus.ACTIVE.value,
                active=True,
            )

            relationship.mark_seen(
                seen_at=seen_at
            )

            self.add(
                relationship,
                flush=True,
            )

            return relationship, True

        relationship.discovery_method = discovery_method
        relationship.confidence = confidence

        if collector_id is not None:
            relationship.collector_id = collector_id

        if correlation_id is not None:
            relationship.correlation_id = correlation_id

        if custom_type is not None:
            relationship.custom_type = custom_type

        if attributes is not None:
            relationship.attributes = attributes

        if evidence is not None:
            relationship.evidence = evidence

        if labels is not None:
            relationship.labels = labels

        if description is not None:
            relationship.description = description

        relationship.mark_seen(
            seen_at=seen_at
        )

        self.flush()

        return relationship, False

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    def mark_seen(
        self,
        *,
        relationship_id: uuid.UUID,
        seen_at: datetime,
    ) -> ResourceRelationship:
        """Mark a topology edge as observed."""
        relationship = self.get_required(
            relationship_id
        )

        relationship.mark_seen(
            seen_at=seen_at
        )

        self.flush()

        return relationship

    def mark_stale(
        self,
        *,
        relationship_id: uuid.UUID,
    ) -> ResourceRelationship:
        """Mark a topology relationship stale."""
        relationship = self.get_required(
            relationship_id
        )

        relationship.mark_stale()

        self.flush()

        return relationship

    def mark_deleted(
        self,
        *,
        relationship_id: uuid.UUID,
        deleted_at: datetime,
    ) -> ResourceRelationship:
        """
        Soft-delete a topology relationship while retaining history.
        """
        relationship = self.get_required(
            relationship_id
        )

        relationship.mark_deleted(
            deleted_at=deleted_at
        )

        self.flush()

        return relationship

    # ------------------------------------------------------------------
    # Inference / confidence
    # ------------------------------------------------------------------

    def update_confidence(
        self,
        *,
        relationship_id: uuid.UUID,
        confidence: float,
    ) -> ResourceRelationship:
        """Update topology inference confidence."""
        relationship = self.get_required(
            relationship_id
        )

        relationship.set_confidence(
            confidence
        )

        self.flush()

        return relationship


__all__ = [
    "ResourceRelationshipRepository",
]
