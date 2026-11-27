"""
OSHA Infrastructure Resource Relationship Model.

ResourceRelationship represents topology and dependency connections between
infrastructure resources discovered by OSHA.

Examples:

    Kubernetes Node
        RUNS_ON
    Linux Host

    Pod
        RUNS_ON
    Kubernetes Node

    Container
        MEMBER_OF
    Pod

    Application
        DEPENDS_ON
    PostgreSQL Service

    Container
        MOUNTS
    Volume

    Service A
        COMMUNICATES_WITH
    Service B

The Resource.parent_id field is intended for simple containment hierarchy.
This model represents richer infrastructure topology that cannot be expressed
correctly as a parent-child tree.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base, TenantMixin, VersionMixin


# ---------------------------------------------------------------------------
# Relationship classification
# ---------------------------------------------------------------------------

class RelationshipType(str, enum.Enum):
    """Canonical relationship types understood by OSHA."""

    # Placement / execution
    RUNS_ON = "runs_on"
    HOSTED_ON = "hosted_on"
    SCHEDULED_ON = "scheduled_on"
    DEPLOYED_ON = "deployed_on"

    # Containment
    CONTAINS = "contains"
    MEMBER_OF = "member_of"
    PART_OF = "part_of"
    BELONGS_TO = "belongs_to"

    # Application / infrastructure dependency
    DEPENDS_ON = "depends_on"
    REQUIRED_BY = "required_by"
    USES = "uses"
    PROVIDES = "provides"

    # Management
    MANAGES = "manages"
    MANAGED_BY = "managed_by"
    CONTROLS = "controls"
    CONTROLLED_BY = "controlled_by"

    # Networking
    COMMUNICATES_WITH = "communicates_with"
    CONNECTED_TO = "connected_to"
    ROUTES_TO = "routes_to"
    EXPOSES = "exposes"

    # Storage
    MOUNTS = "mounts"
    ATTACHED_TO = "attached_to"
    STORED_ON = "stored_on"

    # Ownership
    OWNS = "owns"
    OWNED_BY = "owned_by"

    # Runtime / orchestration
    SPAWNS = "spawns"
    CREATED_BY = "created_by"
    SELECTED_BY = "selected_by"

    # Logical topology
    ASSOCIATED_WITH = "associated_with"
    RELATED_TO = "related_to"

    # Extensibility
    CUSTOM = "custom"


class RelationshipStatus(str, enum.Enum):
    """Lifecycle state of a topology relationship."""

    ACTIVE = "active"
    STALE = "stale"
    DELETED = "deleted"


class RelationshipDiscoveryMethod(str, enum.Enum):
    """How OSHA established the relationship."""

    COLLECTOR = "collector"
    API = "api"
    TELEMETRY = "telemetry"
    CORRELATION = "correlation"
    INFERENCE = "inference"
    USER = "user"
    IMPORT = "import"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Resource relationship
# ---------------------------------------------------------------------------

class ResourceRelationship(
    TenantMixin,
    VersionMixin,
    Base,
):
    """
    Directed relationship between two OSHA resources.

    The relationship is stored as:

        source_resource
                |
                | relationship_type
                v
        target_resource

    Example:

        nginx-container
                |
                | runs_on
                v
        kubernetes-node-01

    Relationships are directional unless the relationship type is logically
    symmetric, such as COMMUNICATES_WITH.

    Confidence is particularly useful for relationships inferred by OSHA's
    correlation or intelligence components. Collector-confirmed relationships
    can normally use confidence=1.0.
    """

    __tablename__ = "resource_relationships"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "source_resource_id",
            "target_resource_id",
            "relationship_type",
            name="uq_resource_relationship_edge",
        ),
        CheckConstraint(
            "source_resource_id <> target_resource_id",
            name="resource_relationship_no_self_reference",
        ),
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="resource_relationship_confidence_range",
        ),
        Index(
            "ix_relationship_source_type",
            "source_resource_id",
            "relationship_type",
        ),
        Index(
            "ix_relationship_target_type",
            "target_resource_id",
            "relationship_type",
        ),
        Index(
            "ix_relationship_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_relationship_source_target",
            "source_resource_id",
            "target_resource_id",
        ),
        Index(
            "ix_relationship_last_seen",
            "last_seen_at",
        ),
    )

    # ------------------------------------------------------------------
    # Graph endpoints
    # ------------------------------------------------------------------

    source_resource_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "resources.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    target_resource_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "resources.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    source_resource: Mapped["Resource"] = relationship(
        "Resource",
        foreign_keys=[source_resource_id],
        lazy="selectin",
    )

    target_resource: Mapped["Resource"] = relationship(
        "Resource",
        foreign_keys=[target_resource_id],
        lazy="selectin",
    )

    # ------------------------------------------------------------------
    # Relationship identity
    # ------------------------------------------------------------------

    relationship_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    custom_type: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Relationship lifecycle
    # ------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RelationshipStatus.ACTIVE.value,
        index=True,
    )

    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    first_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    discovery_method: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=RelationshipDiscoveryMethod.COLLECTOR.value,
        index=True,
    )

    collector_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    correlation_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
    )

    # ------------------------------------------------------------------
    # Relationship metadata
    # ------------------------------------------------------------------

    attributes: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    evidence: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    labels: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Typed helpers
    # ------------------------------------------------------------------

    @property
    def kind(self) -> RelationshipType | None:
        """Return the typed relationship when recognised."""
        try:
            return RelationshipType(self.relationship_type)
        except ValueError:
            return None

    @property
    def lifecycle(self) -> RelationshipStatus | None:
        """Return the typed relationship lifecycle state."""
        try:
            return RelationshipStatus(self.status)
        except ValueError:
            return None

    @property
    def discovery_kind(self) -> RelationshipDiscoveryMethod | None:
        """Return the typed discovery method when recognised."""
        try:
            return RelationshipDiscoveryMethod(self.discovery_method)
        except ValueError:
            return None

    @property
    def is_inferred(self) -> bool:
        """
        Return whether OSHA derived the relationship rather than receiving
        it directly from normal collector discovery.
        """
        return self.discovery_method in {
            RelationshipDiscoveryMethod.CORRELATION.value,
            RelationshipDiscoveryMethod.INFERENCE.value,
            RelationshipDiscoveryMethod.TELEMETRY.value,
        }

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    def mark_seen(
        self,
        *,
        seen_at: datetime,
    ) -> None:
        """Record successful rediscovery of this relationship."""
        if self.first_seen_at is None:
            self.first_seen_at = seen_at

        self.last_seen_at = seen_at
        self.status = RelationshipStatus.ACTIVE.value
        self.active = True
        self.deleted_at = None

    def mark_stale(self) -> None:
        """
        Mark the relationship as stale without deleting historical topology.
        """
        self.status = RelationshipStatus.STALE.value

    def mark_deleted(
        self,
        *,
        deleted_at: datetime,
    ) -> None:
        """
        Soft-delete the relationship while retaining historical topology.
        """
        self.status = RelationshipStatus.DELETED.value
        self.active = False
        self.deleted_at = deleted_at

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def set_confidence(
        self,
        confidence: float,
    ) -> None:
        """Set relationship confidence after validating its range."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "relationship confidence must be between 0.0 and 1.0"
            )

        self.confidence = confidence

    # ------------------------------------------------------------------
    # Topology representation
    # ------------------------------------------------------------------

    def topology_edge(self) -> dict[str, Any]:
        """
        Return a serialisable non-secret representation of the topology edge.
        """
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "source_resource_id": str(self.source_resource_id),
            "target_resource_id": str(self.target_resource_id),
            "relationship_type": self.relationship_type,
            "custom_type": self.custom_type,
            "status": self.status,
            "active": self.active,
            "confidence": self.confidence,
            "discovery_method": self.discovery_method,
        }


__all__ = [
    "RelationshipDiscoveryMethod",
    "RelationshipStatus",
    "RelationshipType",
    "ResourceRelationship",
]
