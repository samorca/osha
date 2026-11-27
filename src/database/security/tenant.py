"""
OSHA PostgreSQL Tenant Security.

Bridges OSHA's application security context to the active PostgreSQL
transaction.

The intended security architecture is:

    authenticated request / trusted worker
                |
                v
    DatabaseSecurityContext
                |
                v
       bind_tenant_context()
                |
                v
        PostgreSQL transaction
                |
                v
        OSHA tenant setting
                |
                v
      Row-Level Security policy
                |
                v
          tenant_id rows

The application must not rely on repository filters alone for tenant
isolation.

PostgreSQL Row-Level Security should provide an additional database-level
enforcement boundary.

IMPORTANT:

Normal OSHA application database roles should not be PostgreSQL superusers
and should not have BYPASSRLS when tenant isolation depends on RLS.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .context import (
    DatabaseSecurityContext,
    require_security_context,
    require_tenant,
)
from .validation import (
    validate_tenant_id,
)


# ---------------------------------------------------------------------------
# PostgreSQL setting names
# ---------------------------------------------------------------------------

TENANT_SETTING = "osha.tenant_id"
ACTOR_ID_SETTING = "osha.actor_id"
ACTOR_TYPE_SETTING = "osha.actor_type"
REQUEST_ID_SETTING = "osha.request_id"
CORRELATION_ID_SETTING = "osha.correlation_id"
PRIVILEGED_SETTING = "osha.privileged"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TenantSecurityError(RuntimeError):
    """Base error for OSHA PostgreSQL tenant security operations."""


class TenantBindingError(TenantSecurityError):
    """Raised when tenant context cannot be bound to PostgreSQL."""


class TenantVerificationError(TenantSecurityError):
    """Raised when PostgreSQL tenant state does not match the application."""


class UnsupportedTenantDatabaseError(TenantSecurityError):
    """
    Raised when tenant binding is requested for an unsupported backend.

    OSHA's RLS implementation in this module targets PostgreSQL.
    """


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PostgreSQLTenantState:
    """Tenant/security state read from the PostgreSQL transaction."""

    tenant_id: uuid.UUID

    actor_id: str | None
    actor_type: str | None

    request_id: str | None
    correlation_id: str | None

    privileged: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable representation."""
        return {
            "tenant_id": str(self.tenant_id),
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "privileged": self.privileged,
        }


# ---------------------------------------------------------------------------
# Backend validation
# ---------------------------------------------------------------------------

def _require_postgresql_session(
    session: Session,
) -> None:
    """Require the Session to be bound to PostgreSQL."""
    bind = session.get_bind()

    if bind is None:
        raise UnsupportedTenantDatabaseError(
            "SQLAlchemy Session has no database bind"
        )

    dialect_name = bind.dialect.name

    if dialect_name != "postgresql":
        raise UnsupportedTenantDatabaseError(
            "OSHA PostgreSQL tenant binding requires "
            f"the PostgreSQL dialect, got {dialect_name!r}"
        )


def _require_postgresql_connection(
    connection: Connection,
) -> None:
    """Require a PostgreSQL SQLAlchemy Connection."""
    dialect_name = connection.dialect.name

    if dialect_name != "postgresql":
        raise UnsupportedTenantDatabaseError(
            "OSHA PostgreSQL tenant binding requires "
            f"the PostgreSQL dialect, got {dialect_name!r}"
        )


# ---------------------------------------------------------------------------
# PostgreSQL transaction-local settings
# ---------------------------------------------------------------------------

def _set_local_setting(
    connection: Connection,
    *,
    name: str,
    value: str,
) -> None:
    """
    Set a PostgreSQL transaction-local OSHA configuration value.

    PostgreSQL set_config(..., true) scopes the value locally to the current
    transaction.
    """
    connection.execute(
        text(
            "SELECT set_config(:name, :value, true)"
        ),
        {
            "name": name,
            "value": value,
        },
    )


def _read_setting(
    connection: Connection,
    *,
    name: str,
) -> str | None:
    """
    Read an OSHA PostgreSQL configuration value.

    current_setting(..., true) returns NULL when the setting is unavailable.
    """
    value = connection.scalar(
        text(
            "SELECT current_setting(:name, true)"
        ),
        {
            "name": name,
        },
    )

    if value is None:
        return None

    result = str(value).strip()

    return result or None


# ---------------------------------------------------------------------------
# Bind context
# ---------------------------------------------------------------------------

def bind_tenant_context(
    session: Session,
    *,
    context: DatabaseSecurityContext | None = None,
    verify: bool = True,
) -> PostgreSQLTenantState:
    """
    Bind OSHA security identity to the current PostgreSQL transaction.

    If context is omitted, the active DatabaseSecurityContext is required.

    Transaction-local settings are used so pooled connections do not
    intentionally carry tenant identity as permanent connection state.

    The caller must establish the application security context before calling
    this function.
    """
    if session is None:
        raise TypeError(
            "session cannot be None"
        )

    _require_postgresql_session(
        session
    )

    security = (
        context
        if context is not None
        else require_security_context()
    )

    tenant_id = validate_tenant_id(
        security.tenant_id
    )

    try:
        connection = session.connection()

        _require_postgresql_connection(
            connection
        )

        _set_local_setting(
            connection,
            name=TENANT_SETTING,
            value=str(tenant_id),
        )

        _set_local_setting(
            connection,
            name=ACTOR_TYPE_SETTING,
            value=security.actor_type,
        )

        _set_local_setting(
            connection,
            name=PRIVILEGED_SETTING,
            value=(
                "true"
                if security.privileged
                else "false"
            ),
        )

        _set_local_setting(
            connection,
            name=ACTOR_ID_SETTING,
            value=security.actor_id or "",
        )

        _set_local_setting(
            connection,
            name=REQUEST_ID_SETTING,
            value=security.request_id or "",
        )

        _set_local_setting(
            connection,
            name=CORRELATION_ID_SETTING,
            value=security.correlation_id or "",
        )

        state = read_tenant_state(
            session
        )

    except SQLAlchemyError as exc:
        raise TenantBindingError(
            "unable to bind OSHA tenant context "
            "to PostgreSQL transaction"
        ) from exc

    if verify:
        verify_tenant_state(
            expected=security,
            actual=state,
        )

    return state


# ---------------------------------------------------------------------------
# Read transaction state
# ---------------------------------------------------------------------------

def read_tenant_state(
    session: Session,
) -> PostgreSQLTenantState:
    """
    Read OSHA tenant/security state from the PostgreSQL transaction.

    Missing tenant identity fails closed.
    """
    if session is None:
        raise TypeError(
            "session cannot be None"
        )

    _require_postgresql_session(
        session
    )

    try:
        connection = session.connection()

        _require_postgresql_connection(
            connection
        )

        tenant_value = _read_setting(
            connection,
            name=TENANT_SETTING,
        )

        if tenant_value is None:
            raise TenantVerificationError(
                "PostgreSQL transaction has no OSHA tenant context"
            )

        try:
            tenant_id = validate_tenant_id(
                tenant_value
            )

        except ValueError as exc:
            raise TenantVerificationError(
                "PostgreSQL transaction contains "
                "an invalid OSHA tenant identity"
            ) from exc

        actor_id = _read_setting(
            connection,
            name=ACTOR_ID_SETTING,
        )

        actor_type = _read_setting(
            connection,
            name=ACTOR_TYPE_SETTING,
        )

        request_id = _read_setting(
            connection,
            name=REQUEST_ID_SETTING,
        )

        correlation_id = _read_setting(
            connection,
            name=CORRELATION_ID_SETTING,
        )

        privileged_value = _read_setting(
            connection,
            name=PRIVILEGED_SETTING,
        )

        privileged = (
            privileged_value is not None
            and privileged_value.lower() == "true"
        )

        return PostgreSQLTenantState(
            tenant_id=tenant_id,
            actor_id=actor_id,
            actor_type=actor_type,
            request_id=request_id,
            correlation_id=correlation_id,
            privileged=privileged,
        )

    except TenantSecurityError:
        raise

    except SQLAlchemyError as exc:
        raise TenantVerificationError(
            "unable to read OSHA PostgreSQL tenant state"
        ) from exc


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_tenant_state(
    *,
    expected: DatabaseSecurityContext,
    actual: PostgreSQLTenantState,
) -> None:
    """
    Verify PostgreSQL transaction identity against application context.
    """
    require_tenant(
        expected.tenant_id
    )

    if actual.tenant_id != expected.tenant_id:
        raise TenantVerificationError(
            "PostgreSQL tenant identity does not match "
            "the OSHA application security context"
        )

    if actual.actor_type != expected.actor_type:
        raise TenantVerificationError(
            "PostgreSQL actor type does not match "
            "the OSHA application security context"
        )

    if actual.actor_id != expected.actor_id:
        raise TenantVerificationError(
            "PostgreSQL actor identity does not match "
            "the OSHA application security context"
        )

    if actual.request_id != expected.request_id:
        raise TenantVerificationError(
            "PostgreSQL request identity does not match "
            "the OSHA application security context"
        )

    if actual.correlation_id != expected.correlation_id:
        raise TenantVerificationError(
            "PostgreSQL correlation identity does not match "
            "the OSHA application security context"
        )

    if actual.privileged != expected.privileged:
        raise TenantVerificationError(
            "PostgreSQL privilege state does not match "
            "the OSHA application security context"
        )


# ---------------------------------------------------------------------------
# Tenant guard
# ---------------------------------------------------------------------------

def require_bound_tenant(
    session: Session,
    *,
    tenant_id: uuid.UUID | str,
) -> PostgreSQLTenantState:
    """
    Require PostgreSQL transaction tenant identity to equal tenant_id.

    This provides another fail-closed boundary before tenant-sensitive
    operations.
    """
    expected_tenant = validate_tenant_id(
        tenant_id
    )

    require_tenant(
        expected_tenant
    )

    state = read_tenant_state(
        session
    )

    if state.tenant_id != expected_tenant:
        raise TenantVerificationError(
            "bound PostgreSQL tenant does not match "
            "the requested tenant"
        )

    return state


# ---------------------------------------------------------------------------
# RLS policy expression
# ---------------------------------------------------------------------------

def tenant_rls_expression() -> str:
    """
    Return the canonical OSHA PostgreSQL tenant policy expression.

    This string is intended for Alembic migrations that create tenant RLS
    policies.

    The policy casts the transaction setting to UUID and compares it with
    the row's tenant_id.
    """
    return (
        "tenant_id = "
        "NULLIF(current_setting('osha.tenant_id', true), '')::uuid"
    )


def tenant_rls_check_expression() -> str:
    """
    Return the canonical WITH CHECK expression for tenant writes.
    """
    return tenant_rls_expression()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ACTOR_ID_SETTING",
    "ACTOR_TYPE_SETTING",
    "CORRELATION_ID_SETTING",
    "PostgreSQLTenantState",
    "PRIVILEGED_SETTING",
    "REQUEST_ID_SETTING",
    "TENANT_SETTING",
    "TenantBindingError",
    "TenantSecurityError",
    "TenantVerificationError",
    "UnsupportedTenantDatabaseError",
    "bind_tenant_context",
    "read_tenant_state",
    "require_bound_tenant",
    "tenant_rls_check_expression",
    "tenant_rls_expression",
    "verify_tenant_state",
]
