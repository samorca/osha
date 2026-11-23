from __future__ import annotations

"""
Composition facade for OSHA persistence
services.
"""

from threading import RLock

from typing import (
    Any,
    Dict,
    Optional,
)

from src.database.config import (
    DatabaseConfig,
    safe_config_summary,
)

from src.database.connection import (
    DatabaseAdapter,
    DatabaseConnection,
)

from src.database.errors import (
    DatabaseConnectionError,
    RepositoryValidationError,
)

from src.database.health import (
    DatabaseHealthChecker,
)

from src.database.migrations import (
    Migration,
    MigrationRegistry,
    MigrationRunner,
)

from src.database.repository import (
    MemoryRepository,
)

from src.database.schema import (
    DatabaseBackend,
    DatabaseHealth,
)

from src.database.transaction import (
    DatabaseTransaction,
    TransactionAdapter,
)

from src.security.secrets import (
    SecretManager,
)


# ---------------------------------------------------------------
# Database pipeline
# ---------------------------------------------------------------

class DatabasePipeline:
    """
    Unified lifecycle facade for OSHA persistence
    services.

    Responsibilities:

      database connection lifecycle
      credential resolution through security
      transactions
      repositories
      migrations
      database dependency health

    Domain-specific persistence logic should stay
    in domain repositories rather than being
    added directly to this class.
    """

    def __init__(
        self,
        *,
        config: Optional[
            DatabaseConfig
        ] = None,
        secrets: Optional[
            SecretManager
        ] = None,
        adapter: Optional[
            DatabaseAdapter
        ] = None,
        transaction_adapter: Optional[
            TransactionAdapter
        ] = None,
        degraded_latency_seconds: float = 1.0,
    ) -> None:
        self.config = (
            config
            or DatabaseConfig()
        )

        self.config.validate()

        # -----------------------------------------------------
        # Security-backed secret resolution
        # -----------------------------------------------------

        self.secrets = secrets

        # -----------------------------------------------------
        # Connection
        # -----------------------------------------------------

        self.connection = (
            DatabaseConnection(
                config=self.config,
                adapter=adapter,
                secrets=secrets,
            )
        )

        # -----------------------------------------------------
        # Transaction adapter
        # -----------------------------------------------------

        self.transaction_adapter = (
            transaction_adapter
        )

        # -----------------------------------------------------
        # Migrations
        # -----------------------------------------------------

        self.migrations = (
            MigrationRegistry()
        )

        # -----------------------------------------------------
        # Health
        # -----------------------------------------------------

        self.health = (
            DatabaseHealthChecker(
                connection=(
                    self.connection
                ),
                degraded_latency_seconds=(
                    degraded_latency_seconds
                ),
            )
        )

        # -----------------------------------------------------
        # Created repositories
        # -----------------------------------------------------

        self._repositories: Dict[
            str,
            Any,
        ] = {}

        self._lock = RLock()

    # ---------------------------------------------------------
    # Ready
    # ---------------------------------------------------------

    @property
    def ready(
        self,
    ) -> bool:
        return (
            self.connection.ready
        )

    # ---------------------------------------------------------
    # Start
    # ---------------------------------------------------------

    def start(
        self,
    ) -> Dict[str, Any]:
        """
        Start the persistence layer.

        Startup sequence:

            open connection
                ↓
            optional configured migrations
                ↓
            database health check

        Migrations only run automatically when:

            migrations.enabled
            AND
            migrations.apply_on_startup
        """

        status = (
            self.connection.open()
        )

        migration_records = []

        if (
            self.config.
            migrations.enabled
            and self.config.
            migrations.
            apply_on_startup
        ):
            runner = (
                self.migration_runner()
            )

            migration_records = (
                runner.apply_pending()
            )

        health = (
            self.health.check()
        )

        if (
            health.status
            == "unhealthy"
        ):
            raise (
                DatabaseConnectionError(
                    "Database pipeline "
                    "failed its startup "
                    "health check."
                )
            )

        return {
            "connection": (
                status.to_dict()
            ),

            "health": (
                health.to_dict()
            ),

            "applied_startup_migrations": [
                item.to_dict()
                for item
                in migration_records
            ],
        }

    # ---------------------------------------------------------
    # Stop
    # ---------------------------------------------------------

    def stop(
        self,
    ) -> Dict[str, Any]:
        status = (
            self.connection.close()
        )

        return {
            "connection": (
                status.to_dict()
            ),

            "repository_count": (
                len(
                    self._repositories
                )
            ),
        }

    # ---------------------------------------------------------
    # Transaction
    # ---------------------------------------------------------

    def transaction(
        self,
    ) -> DatabaseTransaction:
        """
        Create an unstarted transaction object.

        Typical usage:

            with database.transaction():
                ...
        """

        if not self.connection.ready:
            raise (
                DatabaseConnectionError(
                    "Database pipeline is "
                    "not ready."
                )
            )

        return DatabaseTransaction(
            connection=(
                self.connection
            ),
            adapter=(
                self.
                transaction_adapter
            ),
        )

    # ---------------------------------------------------------
    # Register migration
    # ---------------------------------------------------------

    def register_migration(
        self,
        migration: Migration,
    ) -> Migration:
        return self.migrations.register(
            migration
        )

    # ---------------------------------------------------------
    # Migration runner
    # ---------------------------------------------------------

    def migration_runner(
        self,
    ) -> MigrationRunner:
        if not self.connection.ready:
            raise (
                DatabaseConnectionError(
                    "Database pipeline "
                    "must be started "
                    "before migrations "
                    "run."
                )
            )

        return MigrationRunner(
            connection=(
                self.connection
            ),
            registry=(
                self.migrations
            ),
            config=(
                self.config.migrations
            ),
        )

    # ---------------------------------------------------------
    # Repository
    # ---------------------------------------------------------

    def repository(
        self,
        name: str,
    ) -> MemoryRepository:
        """
        Return or create a generic repository.

        The initial generic implementation is
        intentionally limited to the memory
        backend.

        Production backends should introduce
        backend-specific repository adapters
        rather than pretending MemoryRepository
        works against every driver.
        """

        if not self.connection.ready:
            raise (
                DatabaseConnectionError(
                    "Database pipeline is "
                    "not ready."
                )
            )

        if (
            self.config.backend
            != DatabaseBackend.
            MEMORY.value
        ):
            raise (
                RepositoryValidationError(
                    "Generic repository "
                    "creation currently "
                    "supports only the "
                    "memory backend."
                )
            )

        with self._lock:
            repository = (
                self._repositories.get(
                    name
                )
            )

            if repository is None:
                repository = (
                    MemoryRepository(
                        connection=(
                            self.connection
                        ),
                        name=name,
                    )
                )

                self._repositories[
                    name
                ] = repository

            return repository

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------

    def health_check(
        self,
    ) -> DatabaseHealth:
        return self.health.check()

    # ---------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------

    def __enter__(
        self,
    ) -> "DatabasePipeline":
        self.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:
        self.stop()

        return False

    # ---------------------------------------------------------
    # Safe summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        """
        Return non-secret state for diagnostics.

        No resolved credential value is included.
        """

        with self._lock:
            repositories = {
                name: (
                    repository.
                    summary()
                )
                for (
                    name,
                    repository,
                )
                in sorted(
                    self._repositories.
                    items()
                )
            }

        # Applied migration state requires an
        # active connection. Don't implicitly
        # connect merely to produce a summary.
        if self.connection.ready:
            migration_summary = (
                self.migration_runner().
                summary()
            )

        else:
            migration_summary = {
                "enabled": (
                    self.config.
                    migrations.enabled
                ),

                "apply_on_startup": (
                    self.config.
                    migrations.
                    apply_on_startup
                ),

                "registered_migrations": (
                    len(
                        self.migrations.
                        list()
                    )
                ),

                "connection_required_for_applied_state": (
                    True
                ),
            }

        return {
            "config": (
                safe_config_summary(
                    self.config
                )
            ),

            "connection": (
                self.connection.
                summary()
            ),

            "health": (
                self.health.summary()
            ),

            "migrations": (
                migration_summary
            ),

            "repositories": (
                repositories
            ),

            "repository_count": (
                len(
                    repositories
                )
            ),

            "credential_value_exposed": (
                False
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DatabasePipeline",
]
