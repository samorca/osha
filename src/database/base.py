"""
OSHA Database ORM Foundation.

Provides the common SQLAlchemy declarative base and reusable mixins for
OSHA's relational database models.

Database connection and session lifecycle are intentionally kept outside
this module.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, MetaData, String
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    declared_attr,
    mapped_column,
)


# ---------------------------------------------------------------------------
# Constraint and index naming
# ---------------------------------------------------------------------------

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

_CAMEL_CASE_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_CASE_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def generate_uuid() -> uuid.UUID:
    """Generate a UUID for a new OSHA database entity."""
    return uuid.uuid4()


def _snake_case(value: str) -> str:
    """Convert a Python class name into snake_case."""
    value = _CAMEL_CASE_BOUNDARY_1.sub(r"\1_\2", value)
    value = _CAMEL_CASE_BOUNDARY_2.sub(r"\1_\2", value)
    return value.replace("-", "_").lower()


# ---------------------------------------------------------------------------
# ORM mixins
# ---------------------------------------------------------------------------

class TableNameMixin:
    """Automatically derive table names from ORM model names."""

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return f"{_snake_case(cls.__name__)}s"


class UUIDPrimaryKeyMixin:
    """Provide a UUID primary key."""

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=generate_uuid,
        nullable=False,
    )


class TimestampMixin:
    """Provide standard creation and modification timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class VersionMixin:
    """Provide a row version for models requiring concurrency tracking."""

    version: Mapped[int] = mapped_column(
        default=1,
        nullable=False,
    )


class TenantMixin:
    """
    Provide tenant ownership.

    This remains optional because not every OSHA database object needs to
    belong to an individual tenant.
    """

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        nullable=False,
        index=True,
    )


class ExternalIdentityMixin:
    """
    Provide identity information for infrastructure discovered outside OSHA.

    Typical source types can include Linux, Kubernetes, OpenStack and other
    infrastructure providers.
    """

    external_id: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        index=True,
    )

    source_type: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )


# ---------------------------------------------------------------------------
# OSHA declarative base
# ---------------------------------------------------------------------------

class Base(
    TableNameMixin,
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    DeclarativeBase,
):
    """
    Base class for OSHA relational ORM models.

    Standard models automatically receive:

        id
        created_at
        updated_at

    Additional capabilities are supplied through optional mixins.
    """

    __abstract__ = True

    metadata = metadata

    def __repr__(self) -> str:
        """
        Return a minimal representation suitable for debugging.

        Full model contents are deliberately excluded because database
        entities may contain sensitive infrastructure information.
        """
        entity_id = getattr(self, "id", None)

        return (
            f"<{self.__class__.__name__} "
            f"id={entity_id!s}>"
        )

    def identity(self) -> dict[str, Any]:
        """Return the stable identity of the database entity."""
        return {
            "type": self.__class__.__name__,
            "id": str(self.id) if self.id is not None else None,
        }


__all__ = [
    "Base",
    "ExternalIdentityMixin",
    "NAMING_CONVENTION",
    "TableNameMixin",
    "TenantMixin",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "VersionMixin",
    "generate_uuid",
    "metadata",
    "utc_now",
]
