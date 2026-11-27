"""
OSHA Source Repository.

Provides persistence operations for infrastructure Sources.

A Source represents an infrastructure discovery boundary such as:

    - Linux host/environment
    - Kubernetes cluster
    - OpenStack deployment
    - Ceph cluster
    - Docker host
    - cloud environment

This repository owns Source-specific database queries while transaction
management remains with OSHA's session layer.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.source import Source, SourceStatus
from .base import BaseRepository


class SourceRepository(BaseRepository[Source]):
    """Repository for OSHA infrastructure sources."""

    model = Source

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    # ------------------------------------------------------------------
    # Identity lookup
    # ------------------------------------------------------------------

    def get_by_external_id(
        self,
        *,
        tenant_id: uuid.UUID,
        source_type: str,
        external_id: str,
    ) -> Source | None:
        """
        Return a source by its tenant-scoped external identity.
        """
        statement = (
            select(Source)
            .where(
                Source.tenant_id == tenant_id,
                Source.source_type == source_type,
                Source.external_id == external_id,
            )
            .limit(1)
        )

        return self.session.scalar(statement)

    def get_by_name(
        self,
        *,
        tenant_id: uuid.UUID,
        name: str,
        source_type: str | None = None,
    ) -> Source | None:
        """Return a source by name within a tenant."""
        statement = select(Source).where(
            Source.tenant_id == tenant_id,
            Source.name == name,
        )

        if source_type is not None:
            statement = statement.where(
                Source.source_type == source_type
            )

        return self.session.scalar(
            statement.limit(1)
        )

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_for_tenant(
        self,
        *,
        tenant_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return infrastructure sources belonging to a tenant."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Source)
            .where(
                Source.tenant_id == tenant_id
            )
            .order_by(
                Source.name.asc()
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
        source_type: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return tenant sources of a specific infrastructure type."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Source)
            .where(
                Source.tenant_id == tenant_id,
                Source.source_type == source_type,
            )
            .order_by(
                Source.name.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    def list_active(
        self,
        *,
        tenant_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return active and enabled sources for a tenant."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Source)
            .where(
                Source.tenant_id == tenant_id,
                Source.status == SourceStatus.ACTIVE.value,
                Source.enabled.is_(True),
            )
            .order_by(
                Source.name.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    def list_unreachable(
        self,
        *,
        tenant_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence:
        """Return sources currently marked unreachable."""
        self._validate_pagination(
            offset=offset,
            limit=limit,
        )

        statement = (
            select(Source)
            .where(
                Source.tenant_id == tenant_id,
                Source.status == SourceStatus.UNREACHABLE.value,
            )
            .order_by(
                Source.last_seen_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        return tuple(
            self.session.scalars(statement).all()
        )

    # ------------------------------------------------------------------
    # Discovery persistence
    # ------------------------------------------------------------------

    def upsert_discovered(
        self,
        *,
        tenant_id: uuid.UUID,
        source_type: str,
        external_id: str,
        name: str,
        seen_at: datetime,
        display_name: str | None = None,
        hostname: str | None = None,
        ip_address: str | None = None,
        address: str | None = None,
        port: int | None = None,
        environment: str | None = None,
        region: str | None = None,
        zone: str | None = None,
        collector_id: uuid.UUID | None = None,
        collector_name: str | None = None,
        collector_version: str | None = None,
        labels: dict | None = None,
        attributes: dict | None = None,
    ) -> tuple[Source, bool]:
        """
        Create or update a discovered infrastructure source.

        Returns:
            (source, created)

        ``created`` is True when a new Source was created and False when an
        existing Source was refreshed.

        The method flushes but does not commit. This allows the caller to
        persist the source and its discovered resources in one transaction.
        """
        source = self.get_by_external_id(
            tenant_id=tenant_id,
            source_type=source_type,
            external_id=external_id,
        )

        created = source is None

        if source is None:
            source = Source(
                tenant_id=tenant_id,
                source_type=source_type,
                external_id=external_id,
                name=name,
                display_name=display_name,
                hostname=hostname,
                ip_address=ip_address,
                address=address,
                port=port,
                environment=environment,
                region=region,
                zone=zone,
                collector_id=collector_id,
                collector_name=collector_name,
                collector_version=collector_version,
                labels=labels or {},
                attributes=attributes or {},
                status=SourceStatus.DISCOVERING.value,
                enabled=True,
            )

            source.mark_seen(
                seen_at=seen_at
            )

            self.add(
                source,
                flush=True,
            )

            return source, True

        # Refresh mutable discovery information.
        source.name = name

        if display_name is not None:
            source.display_name = display_name

        if hostname is not None:
            source.hostname = hostname

        if ip_address is not None:
            source.ip_address = ip_address

        if address is not None:
            source.address = address

        if port is not None:
            source.port = port

        if environment is not None:
            source.environment = environment

        if region is not None:
            source.region = region

        if zone is not None:
            source.zone = zone

        if collector_id is not None:
            source.collector_id = collector_id

        if collector_name is not None:
            source.collector_name = collector_name

        if collector_version is not None:
            source.collector_version = collector_version

        if labels is not None:
            source.labels = labels

        if attributes is not None:
            source.attributes = attributes

        source.mark_seen(
            seen_at=seen_at
        )

        if source.status in {
            SourceStatus.PENDING.value,
            SourceStatus.UNREACHABLE.value,
            SourceStatus.ERROR.value,
        }:
            source.status = SourceStatus.DISCOVERING.value

        self.flush()

        return source, created

    # ------------------------------------------------------------------
    # Health state
    # ------------------------------------------------------------------

    def mark_healthy(
        self,
        *,
        source_id: uuid.UUID,
        checked_at: datetime,
    ) -> Source:
        """Mark a source healthy."""
        source = self.get_required(
            source_id
        )

        source.mark_healthy(
            checked_at=checked_at
        )

        self.flush()

        return source

    def mark_unreachable(
        self,
        *,
        source_id: uuid.UUID,
        message: str | None = None,
        error_code: str | None = None,
    ) -> Source:
        """Mark a source unreachable."""
        source = self.get_required(
            source_id
        )

        source.mark_unreachable(
            message=message,
            error_code=error_code,
        )

        self.flush()

        return source

    def disable(
        self,
        *,
        source_id: uuid.UUID,
    ) -> Source:
        """Disable collection from a source."""
        source = self.get_required(
            source_id
        )

        source.mark_disabled()

        self.flush()

        return source

    def enable(
        self,
        *,
        source_id: uuid.UUID,
    ) -> Source:
        """
        Re-enable collection for a source.

        Enabling does not automatically claim the source is healthy.
        The source returns to pending state until discovery/health checking
        confirms its current condition.
        """
        source = self.get_required(
            source_id
        )

        source.enabled = True
        source.status = SourceStatus.PENDING.value

        source.status_message = None
        source.error_code = None

        self.flush()

        return source


__all__ = [
    "SourceRepository",
]
