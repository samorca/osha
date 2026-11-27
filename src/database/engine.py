"""
OSHA Database Engine Interface.

Provides the high-level lifecycle API for OSHA's relational database engine.

The low-level construction and connectivity logic lives in connection.py.
This module provides a clean interface for application startup, shutdown,
health checks and diagnostics without duplicating engine creation logic.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import Engine

from .connection import (
    DatabaseConnectionError,
    check_database_connection,
    dispose_engine,
    get_engine,
    get_pool_status,
    require_database_connection,
)


logger = logging.getLogger(__name__)


class DatabaseEngineManager:
    """
    Manage the lifecycle of OSHA's process-wide relational database engine.

    The manager does not create a second independent connection pool.
    It delegates to the process-wide engine maintained by connection.py.
    """

    def __init__(self) -> None:
        self._started = False

    @property
    def engine(self) -> Engine:
        """Return the shared SQLAlchemy Engine."""
        return get_engine()

    @property
    def started(self) -> bool:
        """Return whether this manager completed startup successfully."""
        return self._started

    @property
    def ready(self) -> bool:
        """Return whether the manager is started and the database is reachable."""
        return self._started and self.healthy

    @property
    def healthy(self) -> bool:
        """Return whether the database currently responds to a health check."""
        return check_database_connection(self.engine)

    def start(
        self,
        *,
        require_connection: bool = True,
    ) -> Engine:
        """
        Start the database engine lifecycle.

        SQLAlchemy engine creation itself is lazy. When require_connection is
        True, OSHA performs a real connectivity check before startup is
        considered successful.
        """
        engine = self.engine

        if require_connection:
            require_database_connection(engine)

        self._started = True

        logger.info(
            "OSHA database engine started successfully"
        )

        return engine

    def stop(self) -> None:
        """
        Stop the database engine lifecycle and dispose pooled connections.

        Calling stop repeatedly is safe.
        """
        if not self._started:
            return

        dispose_engine()

        self._started = False

        logger.info(
            "OSHA database engine stopped"
        )

    def healthcheck(self) -> bool:
        """Run an explicit database connectivity check."""
        return check_database_connection(self.engine)

    def require_healthy(self) -> None:
        """
        Require database connectivity.

        Raises DatabaseConnectionError if the database cannot be reached.
        """
        require_database_connection(self.engine)

    def diagnostics(self) -> dict[str, Any]:
        """
        Return non-secret engine diagnostics.

        Credentials and database passwords are deliberately excluded.
        """
        engine = self.engine

        return {
            "started": self._started,
            "healthy": check_database_connection(engine),
            "driver": engine.url.drivername,
            "host": engine.url.host,
            "port": engine.url.port,
            "database": engine.url.database,
            "pool": get_pool_status(engine),
        }

    def __enter__(self) -> "DatabaseEngineManager":
        """Support controlled use as a context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Dispose database resources when leaving the context."""
        self.stop()


database_engine_manager = DatabaseEngineManager()


def start_database(
    *,
    require_connection: bool = True,
) -> Engine:
    """
    Start OSHA's process-wide database infrastructure.

    Intended to be called from the application/service startup lifecycle.
    """
    return database_engine_manager.start(
        require_connection=require_connection,
    )


def stop_database() -> None:
    """
    Shut down OSHA's process-wide database infrastructure.

    Intended to be called during graceful application shutdown.
    """
    database_engine_manager.stop()


def database_healthcheck() -> bool:
    """Return the current relational database health state."""
    return database_engine_manager.healthcheck()


def database_diagnostics() -> dict[str, Any]:
    """Return safe database engine diagnostics."""
    return database_engine_manager.diagnostics()


__all__ = [
    "DatabaseConnectionError",
    "DatabaseEngineManager",
    "database_diagnostics",
    "database_engine_manager",
    "database_healthcheck",
    "start_database",
    "stop_database",
]
