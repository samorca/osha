from __future__ import annotations

"""
Database-layer exception hierarchy for OSHA
persistence.
"""

from typing import (
    Any,
    Mapping,
    Optional,
)

from src.core.exceptions import (
    ConflictError,
    NotFoundError,
    PersistenceError,
    TimeoutErrorBase,
)


# ---------------------------------------------------------------
# Base database error
# ---------------------------------------------------------------


class DatabaseError(PersistenceError):
    """
    Base class for expected OSHA database-layer
    failures.

    Raw backend/driver exceptions should normally
    be converted into this hierarchy before they
    leave the database boundary.
    """

    default_code = "database_error"


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------


class DatabaseConfigurationError(DatabaseError):
    default_code = "database_configuration_error"


# ---------------------------------------------------------------
# Connection
# ---------------------------------------------------------------


class DatabaseConnectionError(DatabaseError):
    default_code = "database_connection_error"


class DatabaseUnavailableError(DatabaseConnectionError):
    """
    Database availability failure.

    Marked retryable because temporary dependency
    unavailability may recover, but higher-level
    retry policy remains responsible for deciding
    whether to actually retry.
    """

    default_code = "database_unavailable"

    def __init__(
        self,
        message: str = ("Database dependency is unavailable."),
        *,
        details: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        super().__init__(
            message,
            code=self.default_code,
            retryable=True,
            details=details,
        )


# ---------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------


class DatabaseTimeoutError(
    DatabaseError,
    TimeoutErrorBase,
):
    default_code = "database_timeout"

    def __init__(
        self,
        message: str = ("Database operation timed out."),
        *,
        details: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        super().__init__(
            message,
            code=self.default_code,
            retryable=True,
            details=details,
        )


# ---------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------


class TransactionError(DatabaseError):
    default_code = "transaction_error"


class TransactionStateError(TransactionError):
    default_code = "transaction_state_error"


class TransactionCommitError(TransactionError):
    default_code = "transaction_commit_error"


class TransactionRollbackError(TransactionError):
    default_code = "transaction_rollback_error"


# ---------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------


class RepositoryError(DatabaseError):
    default_code = "repository_error"


class RepositoryNotFoundError(
    RepositoryError,
    NotFoundError,
):
    default_code = "repository_record_not_found"


class RepositoryConflictError(
    RepositoryError,
    ConflictError,
):
    default_code = "repository_conflict"


class RepositoryValidationError(RepositoryError):
    default_code = "repository_validation_error"


# ---------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------


class MigrationError(DatabaseError):
    default_code = "migration_error"


class MigrationConflictError(
    MigrationError,
    ConflictError,
):
    default_code = "migration_conflict"


class MigrationChecksumError(MigrationError):
    default_code = "migration_checksum_error"


class MigrationApplyError(MigrationError):
    default_code = "migration_apply_error"


# ---------------------------------------------------------------
# Driver error wrapping
# ---------------------------------------------------------------


def wrap_driver_error(
    error: BaseException,
    *,
    message: str = ("Database driver operation failed."),
    retryable: bool = False,
    code: str = ("database_driver_error"),
) -> DatabaseError:
    """
    Convert a raw driver exception into bounded
    database-layer metadata.

    The raw exception message is deliberately not
    copied because it may contain:

      SQL fragments
      connection strings
      usernames
      database details
      credentials
      provider-specific information

    Only type/module metadata is preserved.
    """

    return DatabaseError(
        message,
        code=code,
        retryable=retryable,
        details={
            "error_type": (type(error).__name__),
            "error_module": (type(error).__module__),
        },
    )


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DatabaseError",
    "DatabaseConfigurationError",
    "DatabaseConnectionError",
    "DatabaseUnavailableError",
    "DatabaseTimeoutError",
    "TransactionError",
    "TransactionStateError",
    "TransactionCommitError",
    "TransactionRollbackError",
    "RepositoryError",
    "RepositoryNotFoundError",
    "RepositoryConflictError",
    "RepositoryValidationError",
    "MigrationError",
    "MigrationConflictError",
    "MigrationChecksumError",
    "MigrationApplyError",
    "wrap_driver_error",
]
