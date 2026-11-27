"""
OSHA Database Security Context.

Provides request/task-local database security identity.

The security context carries:

    - tenant identity
    - actor identity
    - actor type
    - request identity
    - correlation identity
    - privileged-operation state

The context uses Python context variables so concurrent requests, workers and
tasks do not rely on mutable process-global tenant state.

IMPORTANT:

This module represents application security context. PostgreSQL-level tenant
isolation should additionally be enforced through database policies.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class DatabaseSecurityContext:
    """Security identity associated with a database operation."""

    tenant_id: uuid.UUID

    actor_id: str | None = None
    actor_type: str = "system"

    request_id: str | None = None
    correlation_id: str | None = None

    privileged: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, uuid.UUID):
            raise TypeError(
                "tenant_id must be a UUID"
            )

        if not self.actor_type.strip():
            raise ValueError(
                "actor_type cannot be empty"
            )

        if self.actor_id is not None and not self.actor_id.strip():
            raise ValueError(
                "actor_id cannot be empty when supplied"
            )


_current_security_context: ContextVar[
    DatabaseSecurityContext | None
] = ContextVar(
    "osha_database_security_context",
    default=None,
)


class SecurityContextError(RuntimeError):
    """Base error for OSHA database security context failures."""


class SecurityContextMissingError(SecurityContextError):
    """Raised when tenant-sensitive work has no security context."""


class TenantContextMismatchError(SecurityContextError):
    """Raised when an operation attempts to cross tenant boundaries."""


class PrivilegedContextRequiredError(SecurityContextError):
    """Raised when a privileged operation lacks privileged context."""


def get_security_context() -> DatabaseSecurityContext | None:
    """Return the active database security context."""
    return _current_security_context.get()


def require_security_context() -> DatabaseSecurityContext:
    """
    Return the active security context or fail closed.

    Tenant-sensitive database operations should prefer this helper instead
    of silently operating without tenant identity.
    """
    context = get_security_context()

    if context is None:
        raise SecurityContextMissingError(
            "database security context is required"
        )

    return context


def require_tenant(
    tenant_id: uuid.UUID,
) -> DatabaseSecurityContext:
    """
    Require the active security context to match tenant_id.
    """
    context = require_security_context()

    if context.tenant_id != tenant_id:
        raise TenantContextMismatchError(
            "database operation attempted to cross tenant boundary"
        )

    return context


def require_privileged_context() -> DatabaseSecurityContext:
    """Require an explicitly privileged database security context."""
    context = require_security_context()

    if not context.privileged:
        raise PrivilegedContextRequiredError(
            "privileged database security context is required"
        )

    return context


def set_security_context(
    context: DatabaseSecurityContext,
) -> Token:
    """
    Set the current database security context.

    Prefer security_context() for normal scoped use.
    """
    return _current_security_context.set(
        context
    )


def reset_security_context(
    token: Token,
) -> None:
    """Restore the previous security context."""
    _current_security_context.reset(
        token
    )


def clear_security_context() -> None:
    """Clear the active security context."""
    _current_security_context.set(
        None
    )


@contextmanager
def security_context(
    *,
    tenant_id: uuid.UUID,
    actor_id: str | None = None,
    actor_type: str = "system",
    request_id: str | None = None,
    correlation_id: str | None = None,
    privileged: bool = False,
) -> Iterator"""
    Establish a scoped OSHA database security context.

    Example:

        with security_context(
            tenant_id=tenant_id,
            actor_id=user_id,
            actor_type="user",
        ):
            ...

    The previous context is restored automatically when the scope exits.
    """
    context = DatabaseSecurityContext(
        tenant_id=tenant_id,
        actor_id=actor_id,
        actor_type=actor_type,
        request_id=request_id,
        correlation_id=correlation_id,
        privileged=privileged,
    )

    token = set_security_context(
        context
    )

    try:
        yield context

    finally:
        reset_security_context(
            token
        )


__all__ = [
    "DatabaseSecurityContext",
    "PrivilegedContextRequiredError",
    "SecurityContextError",
    "SecurityContextMissingError",
    "TenantContextMismatchError",
    "clear_security_context",
    "get_security_context",
    "require_privileged_context",
    "require_security_context",
    "require_tenant",
    "reset_security_context",
    "security_context",
    "set_security_context",
]
