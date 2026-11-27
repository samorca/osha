"""
OSHA Infrastructure Resource Model.

A Resource represents an infrastructure object discovered by OSHA.

Examples:

    Linux:
        host
        CPU
        memory
        disk
        network interface
        process
        systemd service
        container

    Kubernetes:
        cluster
        node
        namespace
        deployment
        daemonset
        statefulset
        pod
        container
        service
        persistent volume

    OpenStack:
        project
        instance
        network
        subnet
        port
        router
        volume
        image

Resources belong to a Source and may form hierarchical parent-child
relationships.

More complex topology relationships such as RUNS_ON, DEPENDS_ON,
COMMUNICATES_WITH and MOUNTS are represented separately by the
ResourceRelationship model.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
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
# Resource classification
# ---------------------------------------------------------------------------

class ResourceType(str, enum.Enum):
    """Infrastructure resource types understood by OSHA."""

    # Generic
    UNKNOWN = "unknown"
    ENVIRONMENT = "environment"
    APPLICATION = "application"
    SERVICE = "service"

    # Linux / host infrastructure
    HOST = "host"
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    PARTITION = "partition"
    FILESYSTEM = "filesystem"
    MOUNT = "mount"
    NETWORK_INTERFACE = "network_interface"
    PROCESS = "process"
    SYSTEMD_SERVICE = "systemd_service"

    # Containers
    CONTAINER = "container"
    CONTAINER_IMAGE = "container_image"

    # Kubernetes
    KUBERNETES_CLUSTER = "kubernetes_cluster"
    KUBERNETES_NODE = "kubernetes_node"
    KUBERNETES_NAMESPACE = "kubernetes_namespace"
    KUBERNETES_POD = "kubernetes_pod"
    KUBERNETES_DEPLOYMENT = "kubernetes_deployment"
    KUBERNETES_STATEFULSET = "kubernetes_statefulset"
    KUBERNETES_DAEMONSET = "kubernetes_daemonset"
    KUBERNETES_REPLICASET = "kubernetes_replicaset"
    KUBERNETES_JOB = "kubernetes_job"
    KUBERNETES_CRONJOB = "kubernetes_cronjob"
    KUBERNETES_SERVICE = "kubernetes_service"
    KUBERNETES_INGRESS = "kubernetes_ingress"
    KUBERNETES_CONFIGMAP = "kubernetes_configmap"
    KUBERNETES_SECRET = "kubernetes_secret"
    KUBERNETES_PV = "kubernetes_persistent_volume"
    KUBERNETES_PVC = "kubernetes_persistent_volume_claim"

    # OpenStack
    OPENSTACK_PROJECT = "openstack_project"
    OPENSTACK_INSTANCE = "openstack_instance"
    OPENSTACK_NETWORK = "openstack_network"
    OPENSTACK_SUBNET = "openstack_subnet"
    OPENSTACK_PORT = "openstack_port"
    OPENSTACK_ROUTER = "openstack_router"
    OPENSTACK_VOLUME = "openstack_volume"
    OPENSTACK_IMAGE = "openstack_image"
    OPENSTACK_FLAVOR = "openstack_flavor"

    # Ceph
    CEPH_CLUSTER = "ceph_cluster"
    CEPH_MONITOR = "ceph_monitor"
    CEPH_MANAGER = "ceph_manager"
    CEPH_OSD = "ceph_osd"
    CEPH_POOL = "ceph_pool"

    # Network
    NETWORK = "network"
    SUBNET = "subnet"
    IP_ADDRESS = "ip_address"
    ROUTER = "router"
    SWITCH = "switch"
    LOAD_BALANCER = "load_balancer"

    # Storage
    STORAGE = "storage"
    VOLUME = "volume"


class ResourceStatus(str, enum.Enum):
    """Lifecycle/health state of a discovered resource."""

    UNKNOWN = "unknown"
    DISCOVERED = "discovered"
    ACTIVE = "active"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    FAILED = "failed"
    UNREACHABLE = "unreachable"
    TERMINATED = "terminated"
    DELETED = "deleted"


class ResourceLifecycleState(str, enum.Enum):
    """Persistence lifecycle of a discovered infrastructure object."""

    ACTIVE = "active"
    STALE = "stale"
    DELETED = "deleted"


# ---------------------------------------------------------------------------
# Resource model
# ---------------------------------------------------------------------------

class Resource(
    TenantMixin,
    VersionMixin,
    Base,
):
    """
    Infrastructure object discovered and monitored by OSHA.

    Resource provides OSHA with a canonical representation of infrastructure
    regardless of which collector discovered the original object.

    Resource identity is separate from telemetry.

    For example, Resource represents:

        node-01

    while telemetry represents observations about node-01:

        CPU usage
        memory usage
        logs
        events
        anomalies
        forecasts

    This distinction allows OSHA to correlate multiple observations to the
    same infrastructure entity.
    """

    __tablename__ = "resources"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "source_id",
            "resource_type",
            "external_id",
            name="uq_resources_source_type_external",
        ),
        Index(
            "ix_resources_tenant_source",
            "tenant_id",
            "source_id",
        ),
        Index(
            "ix_resources_source_type",
            "source_id",
            "resource_type",
        ),
        Index(
            "ix_resources_parent",
            "parent_id",
        ),
        Index(
            "ix_resources_status_type",
            "status",
            "resource_type",
        ),
        Index(
            "ix_resources_last_seen",
            "last_seen_at",
        ),
        Index(
            "ix_resources_lookup",
            "tenant_id",
            "source_id",
            "resource_type",
            "name",
        ),
    )

    # ------------------------------------------------------------------
    # Source ownership
    # ------------------------------------------------------------------

    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "sources.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    # ------------------------------------------------------------------
    # Infrastructure hierarchy
    # ------------------------------------------------------------------

    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            "resources.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    parent: Mapped["Resource | None"] = relationship(
        "Resource",
        remote_side="Resource.id",
        back_populates="children",
        foreign_keys=[parent_id],
    )

    children: Mapped[list["Resource"]] = relationship(
        "Resource",
        back_populates="parent",
        foreign_keys=[parent_id],
        passive_deletes=True,
    )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    resource_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    external_id: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        index=True,
    )

    display_name: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Infrastructure coordinates
    # ------------------------------------------------------------------

    hostname: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    ip_address: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    namespace: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    cluster_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    region: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    zone: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    environment: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------
    # Health and lifecycle
    # ------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ResourceStatus.DISCOVERED.value,
        index=True,
    )

    lifecycle_state: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ResourceLifecycleState.ACTIVE.value,
        index=True,
    )

    monitored: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    # ------------------------------------------------------------------
    # Discovery timestamps
    # ------------------------------------------------------------------

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
    # Collector provenance
    # ------------------------------------------------------------------

    collector_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    collector_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    discovery_method: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Infrastructure metadata
    # ------------------------------------------------------------------

    labels: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    annotations: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    attributes: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    capabilities: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Status details
    # ------------------------------------------------------------------

    status_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Resource helpers
    # ------------------------------------------------------------------

    @property
    def kind(self) -> ResourceType | None:
        """
        Return the typed resource type when recognised.

        Unknown resource types remain readable so newer collectors can
        introduce additional infrastructure types without immediately
        breaking older OSHA components.
        """
        try:
            return ResourceType(self.resource_type)
        except ValueError:
            return None

    @property
    def health_status(self) -> ResourceStatus | None:
        """Return the typed resource status when recognised."""
        try:
            return ResourceStatus(self.status)
        except ValueError:
            return None

    @property
    def lifecycle(self) -> ResourceLifecycleState | None:
        """Return the typed lifecycle state when recognised."""
        try:
            return ResourceLifecycleState(self.lifecycle_state)
        except ValueError:
            return None

    @property
    def is_active(self) -> bool:
        """Return whether OSHA currently treats the resource as active."""
        return (
            self.lifecycle_state == ResourceLifecycleState.ACTIVE.value
            and self.status not in {
                ResourceStatus.DELETED.value,
                ResourceStatus.TERMINATED.value,
            }
        )

    @property
    def is_root(self) -> bool:
        """Return whether this resource has no hierarchical parent."""
        return self.parent_id is None

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def mark_seen(
        self,
        *,
        seen_at: datetime,
    ) -> None:
        """Record successful discovery of this resource."""
        if self.first_seen_at is None:
            self.first_seen_at = seen_at

        self.last_seen_at = seen_at
        self.lifecycle_state = ResourceLifecycleState.ACTIVE.value

    def mark_healthy(
        self,
        *,
        seen_at: datetime | None = None,
    ) -> None:
        """Mark the discovered resource as healthy."""
        self.status = ResourceStatus.HEALTHY.value
        self.lifecycle_state = ResourceLifecycleState.ACTIVE.value
        self.status_message = None

        if seen_at is not None:
            self.mark_seen(seen_at=seen_at)

    def mark_degraded(
        self,
        *,
        message: str | None = None,
    ) -> None:
        """Mark the resource as degraded."""
        self.status = ResourceStatus.DEGRADED.value
        self.status_message = message

    def mark_failed(
        self,
        *,
        message: str | None = None,
    ) -> None:
        """Mark the resource as failed."""
        self.status = ResourceStatus.FAILED.value
        self.status_message = message

    def mark_stale(self) -> None:
        """
        Mark a resource as stale.

        Stale means OSHA has not recently rediscovered the resource but has
        not yet concluded that the infrastructure object was deleted.
        """
        self.lifecycle_state = ResourceLifecycleState.STALE.value

    def mark_deleted(
        self,
        *,
        deleted_at: datetime,
    ) -> None:
        """
        Soft-delete a resource.

        Historical identity is retained so telemetry, incidents and
        predictions can continue to reference the infrastructure object.
        """
        self.lifecycle_state = ResourceLifecycleState.DELETED.value
        self.status = ResourceStatus.DELETED.value
        self.monitored = False
        self.deleted_at = deleted_at

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def topology_identity(self) -> dict[str, Any]:
        """
        Return the non-secret identity used by OSHA topology services.
        """
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "source_id": str(self.source_id),
            "parent_id": (
                str(self.parent_id)
                if self.parent_id is not None
                else None
            ),
            "resource_type": self.resource_type,
            "external_id": self.external_id,
            "name": self.name,
            "display_name": self.display_name,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "namespace": self.namespace,
            "cluster_name": self.cluster_name,
            "environment": self.environment,
            "region": self.region,
            "zone": self.zone,
            "status": self.status,
            "lifecycle_state": self.lifecycle_state,
            "monitored": self.monitored,
        }


__all__ = [
    "Resource",
    "ResourceLifecycleState",
    "ResourceStatus",
    "ResourceType",
]
