"""Relational model for persisted API decisions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base, VersionMixin


class Decision(VersionMixin, Base):
    """Durable policy decision evaluated for an incident action."""

    __tablename__ = "decisions"

    decision_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
        index=True,
    )

    incident_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    tenant_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    candidate_action: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    target: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    risk: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    approval_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    policy_results: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    dry_run: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    actor_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    correlation_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    idempotency_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
    )

    attributes: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )