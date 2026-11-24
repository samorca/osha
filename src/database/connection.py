from __future__ import annotations

"""
Database backend/connection lifecycle for OSHA
persistence.
"""

from abc import (
    ABC,
    abstractmethod,
)

from threading import RLock

from typing import (
    Any,
    Dict,
    Optional,
)

from src.core.time import (
    utc_now,
)

from src.database.config import (
    DatabaseConfig,
)

from src.database.errors import (
    DatabaseConnectionError,
    DatabaseUnavailableError,
    wrap_driver_error,
)

from src.database.schema import (
    ConnectionState,
    ConnectionStatus,
    DatabaseBackend,
)

from src.security.secrets import (
    SecretManager,
    SecretValue,
)


# ---------------------------------------------------------------
# Database adapter contract
# ---------------------------------------------------------------


class DatabaseAdapter(ABC):
    """
    Backend adapter contract.

    Raw driver objects should remain behind this
    interface.

    Future implementations might wrap:

      SQLite
      PostgreSQL
      another supported persistence backend
    """

    @abstractmethod
    def connect(
        self,
        *,
        config: DatabaseConfig,
        credential: Optional[SecretValue],
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def close(
        self,
        handle: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def ping(
        self,
        handle: Any,
    ) -> bool:
        raise NotImplementedError


# ---------------------------------------------------------------
# Development memory adapter
# ---------------------------------------------------------------


class MemoryDatabaseAdapter(DatabaseAdapter):
    """
    Development adapter providing a process-local
    database handle.

    It provides no durability and should not be
    treated as a production persistence backend.
    """

    def connect(
        self,
        *,
        config: DatabaseConfig,
        credential: Optional[SecretValue],
    ) -> Dict[str, Any]:
        config.validate()

        # credential is intentionally unused by
        # the memory backend.

        return {
            "backend": (DatabaseBackend.MEMORY.value),
            "database": (config.database),
            "environment_id": (config.environment_id),
            "open": True,
            # repository.py can eventually use
            # this process-local store.
            "store": {},
        }

    def close(
        self,
        handle: Dict[
            str,
            Any,
        ],
    ) -> None:
        handle["open"] = False

    def ping(
        self,
        handle: Dict[
            str,
            Any,
        ],
    ) -> bool:
        return bool(
            handle.get(
                "open",
                False,
            )
        )


# ---------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------


class DatabaseConnection:
    """
    Own one backend handle and enforce its
    application-level connection lifecycle.

    Credential handling:

        DatabaseConfig
             ↓
        SecretReference
             ↓
        SecretManager
             ↓
        SecretValue
             ↓
        DatabaseAdapter.connect()

    The resolved SecretValue is not stored in
    ConnectionStatus.
    """

    def __init__(
        self,
        *,
        config: Optional[DatabaseConfig] = None,
        adapter: Optional[DatabaseAdapter] = None,
        secrets: Optional[SecretManager] = None,
    ) -> None:
        self.config = config or DatabaseConfig()

        self.config.validate()

        # -----------------------------------------------------
        # Default adapter
        # -----------------------------------------------------

        if adapter is None:
            if self.config.backend == DatabaseBackend.MEMORY.value:
                adapter = MemoryDatabaseAdapter()

            else:
                raise (
                    DatabaseConnectionError(
                        "A database adapter "
                        "must be provided for "
                        "the configured "
                        "backend."
                    )
                )

        self.adapter = adapter

        self.secrets = secrets

        self.status = ConnectionStatus(identity=(self.config.identity())).finalize()

        self._handle: Any = None

        self._lock = RLock()

    # ---------------------------------------------------------
    # Handle
    # ---------------------------------------------------------

    @property
    def handle(
        self,
    ) -> Any:
        """
        Return the raw backend handle only while
        the connection is READY.

        Domain code should generally use
        repositories rather than accessing this
        directly.
        """

        with self._lock:
            if self.status.state != ConnectionState.READY.value or self._handle is None:
                raise DatabaseUnavailableError("Database connection is not ready.")

            return self._handle

    # ---------------------------------------------------------
    # Ready state
    # ---------------------------------------------------------

    @property
    def ready(
        self,
    ) -> bool:
        return bool(
            self.status.state == ConnectionState.READY.value
            and self._handle is not None
        )

    # ---------------------------------------------------------
    # Credential resolution
    # ---------------------------------------------------------

    def _credential(
        self,
    ) -> Optional[SecretValue]:
        """
        Resolve credentials only when a reference
        is configured.

        The resolved value is returned directly
        to the adapter and is not stored on
        ConnectionStatus.
        """

        if self.config.credential is None:
            return None

        if self.secrets is None:
            raise DatabaseConnectionError(
                "Database credential "
                "reference is configured "
                "but no SecretManager is "
                "available."
            )

        return self.secrets.resolve(self.config.credential)

    # ---------------------------------------------------------
    # Open
    # ---------------------------------------------------------

    def open(
        self,
    ) -> ConnectionStatus:
        with self._lock:
            if self.status.state == ConnectionState.READY.value:
                return self.status

            if self.status.state not in {
                ConnectionState.CREATED.value,
                ConnectionState.CLOSED.value,
                ConnectionState.FAILED.value,
                ConnectionState.DEGRADED.value,
            }:
                raise (
                    DatabaseConnectionError(
                        "Database connection "
                        "cannot open from "
                        "state "
                        f"{self.status.state}."
                    )
                )

            self.status.state = ConnectionState.CONNECTING.value

            self.status.last_error_type = None

            self.status.finalize()

        # -----------------------------------------------------
        # Resolve credential and connect
        # -----------------------------------------------------

        try:
            credential = self._credential()

            handle = self.adapter.connect(
                config=self.config,
                credential=credential,
            )

            if handle is None:
                raise (
                    DatabaseConnectionError(
                        "Database adapter returned no connection handle."
                    )
                )

            # Initial readiness check.
            if not self.adapter.ping(handle):
                try:
                    self.adapter.close(handle)
                except Exception:
                    pass

                raise (
                    DatabaseUnavailableError(
                        "Database connection did not pass readiness ping."
                    )
                )

        except DatabaseConnectionError as exc:
            with self._lock:
                self._handle = None

                self.status.state = ConnectionState.FAILED.value

                self.status.last_error_type = type(exc).__name__

                self.status.finalize()

            raise

        except Exception as exc:
            wrapped = wrap_driver_error(
                exc,
                message=("Database connection failed."),
                retryable=True,
                code=("database_connection_failed"),
            )

            with self._lock:
                self._handle = None

                self.status.state = ConnectionState.FAILED.value

                self.status.last_error_type = type(exc).__name__

                self.status.finalize()

            raise wrapped from exc

        # -----------------------------------------------------
        # Ready
        # -----------------------------------------------------

        with self._lock:
            self._handle = handle

            self.status.state = ConnectionState.READY.value

            self.status.opened_at = utc_now()

            self.status.closed_at = None

            self.status.last_error_type = None

            self.status.finalize()

            return self.status

    # ---------------------------------------------------------
    # Ping
    # ---------------------------------------------------------

    def ping(
        self,
    ) -> bool:
        """
        Perform a backend health/readiness check.

        Driver exceptions are converted into a
        degraded connection state rather than
        leaking through this lightweight probe.
        """

        with self._lock:
            if self._handle is None:
                return False

            handle = self._handle

        try:
            healthy = bool(self.adapter.ping(handle))

        except Exception as exc:
            with self._lock:
                self.status.state = ConnectionState.DEGRADED.value

                self.status.last_error_type = type(exc).__name__

                self.status.finalize()

            return False

        with self._lock:
            self.status.state = (
                ConnectionState.READY.value
                if healthy
                else (ConnectionState.DEGRADED.value)
            )

            if healthy:
                self.status.last_error_type = None

            self.status.finalize()

        return healthy

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------

    def close(
        self,
    ) -> ConnectionStatus:
        with self._lock:
            if self.status.state == ConnectionState.CLOSED.value:
                return self.status

            handle = self._handle

            self.status.state = ConnectionState.CLOSING.value

            self.status.finalize()

        if handle is not None:
            try:
                self.adapter.close(handle)

            except Exception as exc:
                wrapped = wrap_driver_error(
                    exc,
                    message=("Database connection close failed."),
                    retryable=False,
                    code=("database_close_failed"),
                )

                with self._lock:
                    self.status.state = ConnectionState.FAILED.value

                    self.status.last_error_type = type(exc).__name__

                    self.status.finalize()

                raise wrapped from exc

        with self._lock:
            self._handle = None

            self.status.state = ConnectionState.CLOSED.value

            self.status.closed_at = utc_now()

            self.status.finalize()

            return self.status

    # ---------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------

    def __enter__(
        self,
    ) -> "DatabaseConnection":
        self.open()

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:
        self.close()

        # Never suppress the caller's exception.
        return False

    # ---------------------------------------------------------
    # Safe summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "connection": (self.status.to_dict()),
            "ready": (self.ready),
            "adapter": (type(self.adapter).__name__),
            "credential_reference_configured": (self.config.credential is not None),
            "credential_value_exposed": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DatabaseAdapter",
    "MemoryDatabaseAdapter",
    "DatabaseConnection",
]
