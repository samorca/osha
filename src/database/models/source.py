"""
OSHA Infrastructure Source Model.

A Source represents an infrastructure environment or endpoint from which
OSHA discovers resources and collects telemetry.

Examples:

    - Linux host
    - Kubernetes cluster
    - OpenStack deployment
    - Ceph cluster
    - Docker host
    - cloud environment
    - network environment

Sources are discovery boundaries. Resources discovered from a source are
modelled separately and linked to their originating source.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base, TenantMixin, VersionMixin


class SourceType(str, enum.Enum):
    """Infrastructure source families supported by OSHA."""

    LINUX = "linux"
    KUBERNETES = "kubernetes"
    OPENSTACK = "openstack"
    CEPH = "ceph"

    DOCKER = "docker"
    CONTAINERD = "containerd"

    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"

    VMWARE = "vmware"
    NETWORK = "network"

    APPLICATION = "application"

    OTHER = "other"


class SourceStatus(str, enum.Enum):
    """Operational state of an OSHA infrastructure source."""

    PENDING = "pending"
    DISCOVERING = "discovering"
    ACTIVE = "active"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    DISABLED = "disabled"
    ERROR = "error"


class Source(
    TenantMixin,
    VersionMixin,
    Base,
):
    """
    Infrastructure monitored by OSHA.

    A Source identifies an environment OSHA connects to or discovers.

    It does not represent every individual infrastructure object.

    For example:

        Source
          Kubernetes production cluster

        Resources
          node-01
          namespace/default
          pod/payment-api
          container/payment-api

    or:

        Source
          Linux host atonixcorp

        Resources
          network interfaces
          disks
          services
          processes
          containers

    Sensitive credentials must not be stored in ``config``.
    """

    __tablename__ = "sources"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "source_type",
            "external_id",
            name="uq_sources_tenant_type_external",
        ),
        Index(
            "ix_sources_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_sources_tenant_type",
            "tenant_id",
            "source_type",
        ),
        Index(
            "ix_sources_last_seen",
            "last_seen_at",
        ),
    )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    display_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    source_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    external_id: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Network / endpoint identity
    # ------------------------------------------------------------------

    hostname: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    address: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )

    ip_address: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    port: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Environment classification
    # ------------------------------------------------------------------

    environment: Mapped[str | None] = mapped_column(
        String(64),
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

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=SourceStatus.PENDING.value,
        index=True,
    )

    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    managed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    # ------------------------------------------------------------------
    # Discovery information
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

    last_discovered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_healthy_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Collector ownership
    # ------------------------------------------------------------------

    collector_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        index=True,
    )

    collector_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    collector_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    labels: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    attributes: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    config: Mapped[dict[str, Any]] = mapped_column(
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

    error_code: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Domain helpers
    # ------------------------------------------------------------------

    @property
    def source_kind(self) -> SourceType | None:
        """
        Return the typed source family when recognised.

        Unknown values return None so a newer collector value does not cause
        older application code to fail while reading the row.
        """
        try:
            return SourceType(self.source_type)
        except ValueError:
            return None

    @property
    def source_status(self) -> SourceStatus | None:
        """Return the typed source status when recognised."""
        try:
            return SourceStatus(self.status)
        except ValueError:
            return None

    @property
    def is_available(self) -> bool:
        """
        Return whether OSHA considers the source usable for collection.

        This is an application-level convenience property and does not perform
        a live connectivity check.
        """
        return (
            self.enabled
            and self.status
            in {
                SourceStatus.ACTIVE.value,
                SourceStatus.DISCOVERING.value,
            }
        )

    def mark_seen(
        self,
        *,
        seen_at: datetime,
    ) -> None:
        """Update the source's discovery timestamps."""
        if self.first_seen_at is None:
            self.first_seen_at = seen_at

        self.last_seen_at = seen_at

    def mark_healthy(
        self,
        *,
        checked_at: datetime,
    ) -> None:
        """Mark the source as active and healthy."""
        self.status = SourceStatus.ACTIVE.value
        self.last_healthy_at = checked_at
        self.last_seen_at = checked_at
        self.status_message = None
        self.error_code = None

    def mark_unreachable(
        self,
        *,
        message: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """Mark the source as unreachable."""
        self.status = SourceStatus.UNREACHABLE.value
        self.status_message = message
        self.error_code = error_code

    def mark_disabled(self) -> None:
        """Disable collection from this source."""
        self.enabled = False
        self.status = SourceStatus.DISABLED.value

    def public_identity(self) -> dict[str, Any]:
        """
        Return non-secret source identity information.

        Configuration contents are deliberately excluded.
        """
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "name": self.name,
            "display_name": self.display_name,
            "source_type": self.source_type,
            "external_id": self.external_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "environment": self.environment,
            "region": self.region,
            "zone": self.zone,
            "status": self.status,
            "enabled": self.enabled,
        }


__all__ = [
    "Source",
    "SourceStatus",
    "SourceType",
]
