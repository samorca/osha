from __future__ import annotations

"""
Database-neutral persistence contracts for OSHA.
"""

from dataclasses import (
    dataclass,
    field,
)

from enum import Enum

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from src.core.identifiers import (
    random_id,
)

from src.core.serialization import (
    to_json_safe,
)

from src.core.time import (
    utc_now,
)

from src.core.validation import (
    environment_id,
    required_string,
    unique_strings,
)


DATABASE_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------
# Database backend
# ---------------------------------------------------------------

class DatabaseBackend(
    str,
    Enum,
):
    MEMORY = "memory"

    SQLITE = "sqlite"

    POSTGRESQL = "postgresql"

    CUSTOM = "custom"


# ---------------------------------------------------------------
# Connection state
# ---------------------------------------------------------------

class ConnectionState(
    str,
    Enum,
):
    CREATED = "created"

    CONNECTING = "connecting"

    READY = "ready"

    DEGRADED = "degraded"

    CLOSING = "closing"

    CLOSED = "closed"

    FAILED = "failed"


# ---------------------------------------------------------------
# Transaction state
# ---------------------------------------------------------------

class TransactionState(
    str,
    Enum,
):
    CREATED = "created"

    ACTIVE = "active"

    COMMITTED = "committed"

    ROLLED_BACK = "rolled_back"

    FAILED = "failed"


# ---------------------------------------------------------------
# Persistence operation
# ---------------------------------------------------------------

class PersistenceOperation(
    str,
    Enum,
):
    CREATE = "create"

    READ = "read"

    UPDATE = "update"

    DELETE = "delete"

    LIST = "list"


# ---------------------------------------------------------------
# Migration state
# ---------------------------------------------------------------

class MigrationState(
    str,
    Enum,
):
    PENDING = "pending"

    APPLIED = "applied"

    FAILED = "failed"

    SKIPPED = "skipped"


# ---------------------------------------------------------------
# Database identity
# ---------------------------------------------------------------

@dataclass(frozen=True)
class DatabaseIdentity:
    """
    Non-secret identity of a database target.

    Credentials deliberately do not belong here.
    """

    backend: str

    database: str

    environment_id: str

    host: Optional[
        str
    ] = None

    port: Optional[
        int
    ] = None

    namespace: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "DatabaseIdentity":
        if (
            self.backend
            not in {
                item.value
                for item
                in DatabaseBackend
            }
        ):
            raise ValueError(
                "Unknown database backend: "
                f"{self.backend}"
            )

        required_string(
            self.database,
            name="database",
            maximum_length=256,
        )

        environment_id(
            self.environment_id
        )

        if self.host is not None:
            required_string(
                self.host,
                name="database host",
                maximum_length=512,
            )

        if (
            self.port is not None
            and not (
                1
                <= int(
                    self.port
                )
                <= 65535
            )
        ):
            raise ValueError(
                "database port must be "
                "between 1 and 65535"
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "backend": self.backend,
            "database": self.database,
            "environment_id": (
                self.environment_id
            ),
            "host": self.host,
            "port": self.port,
            "namespace": (
                self.namespace
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Connection status
# ---------------------------------------------------------------

@dataclass
class ConnectionStatus:
    """
    Application-level status for one database
    connection/pool abstraction.

    This object must not contain passwords,
    connection credentials or secret values.
    """

    identity: DatabaseIdentity

    state: str = (
        ConnectionState.CREATED.value
    )

    connection_id: Optional[
        str
    ] = None

    opened_at: Optional[
        str
    ] = None

    closed_at: Optional[
        str
    ] = None

    last_error_type: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    updated_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "ConnectionStatus":
        self.identity.finalize()

        if (
            self.state
            not in {
                item.value
                for item
                in ConnectionState
            }
        ):
            raise ValueError(
                "Unknown connection state: "
                f"{self.state}"
            )

        if not self.connection_id:
            self.connection_id = (
                random_id(
                    "database:connection"
                )
            )

        self.updated_at = (
            utc_now()
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "connection_id": (
                self.connection_id
            ),
            "identity": (
                self.identity.to_dict()
            ),
            "state": self.state,
            "opened_at": (
                self.opened_at
            ),
            "closed_at": (
                self.closed_at
            ),
            "last_error_type": (
                self.last_error_type
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
            "updated_at": (
                self.updated_at
            ),
        }


# ---------------------------------------------------------------
# Transaction record
# ---------------------------------------------------------------

@dataclass
class TransactionRecord:
    connection_id: str

    state: str = (
        TransactionState.CREATED.value
    )

    transaction_id: Optional[
        str
    ] = None

    started_at: Optional[
        str
    ] = None

    completed_at: Optional[
        str
    ] = None

    rollback_reason: Optional[
        str
    ] = None

    error_type: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "TransactionRecord":
        required_string(
            self.connection_id,
            name="connection_id",
            maximum_length=512,
        )

        if (
            self.state
            not in {
                item.value
                for item
                in TransactionState
            }
        ):
            raise ValueError(
                "Unknown transaction "
                f"state: {self.state}"
            )

        if not self.transaction_id:
            self.transaction_id = (
                random_id(
                    "database:transaction"
                )
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "transaction_id": (
                self.transaction_id
            ),
            "connection_id": (
                self.connection_id
            ),
            "state": self.state,
            "started_at": (
                self.started_at
            ),
            "completed_at": (
                self.completed_at
            ),
            "rollback_reason": (
                self.rollback_reason
            ),
            "error_type": (
                self.error_type
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Generic persistence metadata
# ---------------------------------------------------------------

@dataclass
class PersistenceRecord:
    """
    Generic metadata describing a record owned by
    a domain repository.

    This is not intended to replace subsystem
    domain schemas.
    """

    repository: str

    record_type: str

    record_id: str

    environment_id: str

    version: int = 1

    reference_ids: List[
        str
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=utc_now
    )

    updated_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "PersistenceRecord":
        required_string(
            self.repository,
            name="repository",
            maximum_length=256,
        )

        required_string(
            self.record_type,
            name="record_type",
            maximum_length=256,
        )

        required_string(
            self.record_id,
            name="record_id",
            maximum_length=512,
        )

        environment_id(
            self.environment_id
        )

        self.version = max(
            1,
            int(
                self.version
            ),
        )

        self.reference_ids = (
            unique_strings(
                self.reference_ids
            )
        )

        self.updated_at = (
            utc_now()
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "repository": (
                self.repository
            ),
            "record_type": (
                self.record_type
            ),
            "record_id": (
                self.record_id
            ),
            "environment_id": (
                self.environment_id
            ),
            "version": (
                self.version
            ),
            "reference_ids": (
                list(
                    self.reference_ids
                )
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
            "created_at": (
                self.created_at
            ),
            "updated_at": (
                self.updated_at
            ),
        }


# ---------------------------------------------------------------
# Generic persistence result
# ---------------------------------------------------------------

@dataclass
class PersistenceResult:
    operation: str

    success: bool

    repository: str

    record_id: Optional[
        str
    ] = None

    reference_ids: List[
        str
    ] = field(
        default_factory=list
    )

    error_type: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    occurred_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "PersistenceResult":
        if (
            self.operation
            not in {
                item.value
                for item
                in PersistenceOperation
            }
        ):
            raise ValueError(
                "Unknown persistence "
                f"operation: {self.operation}"
            )

        required_string(
            self.repository,
            name="repository",
            maximum_length=256,
        )

        self.reference_ids = (
            unique_strings(
                self.reference_ids
            )
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "operation": (
                self.operation
            ),
            "success": self.success,
            "repository": (
                self.repository
            ),
            "record_id": (
                self.record_id
            ),
            "reference_ids": (
                list(
                    self.reference_ids
                )
            ),
            "error_type": (
                self.error_type
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
            "occurred_at": (
                self.occurred_at
            ),
        }


# ---------------------------------------------------------------
# Migration record
# ---------------------------------------------------------------

@dataclass
class MigrationRecord:
    migration_id: str

    name: str

    version: int

    state: str = (
        MigrationState.PENDING.value
    )

    checksum: Optional[
        str
    ] = None

    applied_at: Optional[
        str
    ] = None

    error_type: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "MigrationRecord":
        required_string(
            self.migration_id,
            name="migration_id",
            maximum_length=256,
        )

        required_string(
            self.name,
            name="migration name",
            maximum_length=256,
        )

        self.version = max(
            1,
            int(
                self.version
            ),
        )

        if (
            self.state
            not in {
                item.value
                for item
                in MigrationState
            }
        ):
            raise ValueError(
                "Unknown migration state: "
                f"{self.state}"
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "migration_id": (
                self.migration_id
            ),
            "name": self.name,
            "version": self.version,
            "state": self.state,
            "checksum": (
                self.checksum
            ),
            "applied_at": (
                self.applied_at
            ),
            "error_type": (
                self.error_type
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Database health
# ---------------------------------------------------------------

@dataclass
class DatabaseHealth:
    """
    Health of OSHA's database dependency.

    This does not claim health of infrastructure
    systems OSHA happens to monitor.
    """

    status: str

    connection_id: Optional[
        str
    ] = None

    latency_seconds: Optional[
        float
    ] = None

    message: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    observed_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "DatabaseHealth":
        if (
            self.status
            not in {
                "unknown",
                "healthy",
                "degraded",
                "unhealthy",
            }
        ):
            raise ValueError(
                "Unknown database health "
                f"status: {self.status}"
            )

        if (
            self.latency_seconds
            is not None
            and self.latency_seconds
            < 0
        ):
            raise ValueError(
                "latency_seconds "
                "must be >= 0"
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "status": self.status,
            "connection_id": (
                self.connection_id
            ),
            "latency_seconds": (
                self.latency_seconds
            ),
            "message": self.message,
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
            "observed_at": (
                self.observed_at
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DATABASE_SCHEMA_VERSION",

    "DatabaseBackend",
    "ConnectionState",
    "TransactionState",
    "PersistenceOperation",
    "MigrationState",

    "DatabaseIdentity",
    "ConnectionStatus",
    "TransactionRecord",

    "PersistenceRecord",
    "PersistenceResult",

    "MigrationRecord",

    "DatabaseHealth",
]
