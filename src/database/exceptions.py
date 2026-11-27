"""
OSHA Database Exception Hierarchy.

Provides application-level exceptions for OSHA's database subsystem.

The goal is to prevent higher layers from depending directly on SQLAlchemy
or database-driver exception types.

Typical propagation:

    PostgreSQL / driver error
            |
            v
       SQLAlchemy error
            |
            v
    OSHA database exception
            |
            v
    API / worker / service layer

Exceptions intentionally do not contain database credentials, SQL statements,
tokens, secrets or other sensitive connection information.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class DatabaseError(RuntimeError):
    """
    Base exception for OSHA database subsystem failures.

    All database-specific application exceptions should derive from this
    class so callers can catch either a specialised error or the complete
    database error family.
    """

    default_message = "OSHA database operation failed"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.code = code
        self.context = context or {}

        super().__init__(self.message)

    def as_dict(self) -> dict[str, Any]:
        """
        Return a serialisable, non-sensitive exception representation.

        Callers remain responsible for ensuring context does not contain
        credentials or other secret material.
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "context": self.context,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class DatabaseConfigurationError(DatabaseError):
    """Raised when database configuration is missing or invalid."""

    default_message = "OSHA database configuration is invalid"


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

class DatabaseUnavailableError(DatabaseError):
    """Raised when the configured database cannot be reached."""

    default_message = "OSHA database is unavailable"


class DatabaseTimeoutError(DatabaseUnavailableError):
    """Raised when a database operation exceeds its allowed timeout."""

    default_message = "OSHA database operation timed out"


class DatabasePoolError(DatabaseUnavailableError):
    """Raised when database connection-pool resources cannot be obtained."""

    default_message = "OSHA database connection pool is unavailable"


# ---------------------------------------------------------------------------
# Sessions / transactions
# ---------------------------------------------------------------------------

class DatabaseTransactionError(DatabaseError):
    """Raised when a database transaction fails."""

    default_message = "OSHA database transaction failed"


class DatabaseCommitError(DatabaseTransactionError):
    """Raised when transaction commit fails."""

    default_message = "OSHA database transaction commit failed"


class DatabaseRollbackError(DatabaseTransactionError):
    """Raised when transaction rollback fails."""

    default_message = "OSHA database transaction rollback failed"


class DatabaseFlushError(DatabaseTransactionError):
    """Raised when pending ORM state cannot be flushed."""

    default_message = "OSHA database flush failed"


# ---------------------------------------------------------------------------
# Persistence / integrity
# ---------------------------------------------------------------------------

class DatabaseIntegrityError(DatabaseError):
    """Base exception for persistence integrity failures."""

    default_message = "OSHA database integrity constraint failed"


class DuplicateEntityError(DatabaseIntegrityError):
    """Raised when a uniqueness constraint would be violated."""

    default_message = "OSHA database entity already exists"


class ForeignKeyViolationError(DatabaseIntegrityError):
    """Raised when a database relationship references an invalid entity."""

    default_message = "OSHA database relationship is invalid"


class ConstraintViolationError(DatabaseIntegrityError):
    """Raised when another database constraint is violated."""

    default_message = "OSHA database constraint was violated"


# ---------------------------------------------------------------------------
# Repository / entity errors
# ---------------------------------------------------------------------------

class RepositoryError(DatabaseError):
    """Base error for repository operations."""

    default_message = "OSHA repository operation failed"


class EntityNotFoundError(RepositoryError):
    """Raised when an expected persistent entity cannot be found."""

    default_message = "OSHA database entity was not found"


class EntityConflictError(RepositoryError):
    """Raised when current persistent state conflicts with an operation."""

    default_message = "OSHA database entity is in a conflicting state"


class EntityValidationError(RepositoryError):
    """Raised when persistence input fails database-domain validation."""

    default_message = "OSHA database entity validation failed"


class PaginationError(RepositoryError):
    """Raised when repository pagination parameters are invalid."""

    default_message = "OSHA repository pagination is invalid"


# ---------------------------------------------------------------------------
# Multi-tenancy
# ---------------------------------------------------------------------------

class TenantIsolationError(DatabaseError):
    """
    Raised when an operation attempts an invalid cross-tenant association.

    This exception should not expose information about resources owned by a
    different tenant.
    """

    default_message = "OSHA database tenant boundary violation"


# ---------------------------------------------------------------------------
# Discovery / topology
# ---------------------------------------------------------------------------

class DiscoveryPersistenceError(DatabaseError):
    """Base database error for persistence of discovered infrastructure."""

    default_message = "OSHA infrastructure discovery persistence failed"


class TopologyPersistenceError(DatabaseError):
    """Base database error for persistence or retrieval of topology."""

    default_message = "OSHA infrastructure topology operation failed"


class TopologyLimitError(TopologyPersistenceError):
    """Raised when a topology request exceeds configured traversal limits."""

    default_message = "OSHA topology traversal limit exceeded"


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------

class IncidentPersistenceError(DatabaseError):
    """Base database error for the incident persistence subsystem."""

    default_message = "OSHA incident database operation failed"


class IncidentStateConflictError(IncidentPersistenceError):
    """Raised for invalid persisted incident lifecycle transitions."""

    default_message = "OSHA incident lifecycle transition is invalid"


# ---------------------------------------------------------------------------
# Predictions / anomalies
# ---------------------------------------------------------------------------

class PredictionPersistenceError(DatabaseError):
    """Raised when prediction persistence fails."""

    default_message = "OSHA prediction persistence operation failed"


class AnomalyPersistenceError(DatabaseError):
    """Raised when anomaly persistence fails."""

    default_message = "OSHA anomaly persistence operation failed"


# ---------------------------------------------------------------------------
# Remediation
# ---------------------------------------------------------------------------

class RemediationPersistenceError(DatabaseError):
    """Base error for remediation persistence."""

    default_message = "OSHA remediation database operation failed"


class RemediationStateConflictError(RemediationPersistenceError):
    """Raised for invalid persisted remediation lifecycle transitions."""

    default_message = "OSHA remediation lifecycle transition is invalid"


class RemediationApprovalError(RemediationPersistenceError):
    """Raised when remediation approval requirements are not satisfied."""

    default_message = "OSHA remediation approval requirements are not satisfied"


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class AuditPersistenceError(DatabaseError):
    """Raised when durable audit persistence fails."""

    default_message = "OSHA audit persistence failed"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class DatabaseHealthCheckError(DatabaseError):
    """Raised when database health cannot be determined reliably."""

    default_message = "OSHA database health check failed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Base
    "DatabaseError",

    # Configuration
    "DatabaseConfigurationError",

    # Connectivity
    "DatabaseUnavailableError",
    "DatabaseTimeoutError",
    "DatabasePoolError",

    # Transactions
    "DatabaseTransactionError",
    "DatabaseCommitError",
    "DatabaseRollbackError",
    "DatabaseFlushError",

    # Integrity
    "DatabaseIntegrityError",
    "DuplicateEntityError",
    "ForeignKeyViolationError",
    "ConstraintViolationError",

    # Repository
    "RepositoryError",
    "EntityNotFoundError",
    "EntityConflictError",
    "EntityValidationError",
    "PaginationError",

    # Tenant isolation
    "TenantIsolationError",

    # Discovery / topology
    "DiscoveryPersistenceError",
    "TopologyPersistenceError",
    "TopologyLimitError",

    # Incident
    "IncidentPersistenceError",
    "IncidentStateConflictError",

    # Intelligence
    "PredictionPersistenceError",
    "AnomalyPersistenceError",

    # Remediation
    "RemediationPersistenceError",
    "RemediationStateConflictError",
    "RemediationApprovalError",

    # Audit
    "AuditPersistenceError",

    # Health
    "DatabaseHealthCheckError",
]
