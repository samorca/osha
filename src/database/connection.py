"""
OSHA Database Connection Management.

Creates and manages the SQLAlchemy engine used by OSHA's relational
persistence layer.

Responsibilities:
    - create the SQLAlchemy engine
    - configure connection pooling
    - configure PostgreSQL connection parameters
    - expose database health checks
    - expose safe pool diagnostics
    - dispose database resources during shutdown

Session creation and transaction boundaries belong in session.py.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from .config import DatabaseConfig, get_database_config


logger = logging.getLogger(__name__)


class DatabaseConnectionError(RuntimeError):
    """Raised when OSHA cannot communicate with its relational database."""


def _build_connect_args(config: DatabaseConfig) -> dict[str, Any]:
    """
    Build DBAPI connection arguments.

    PostgreSQL-specific options are applied only when a PostgreSQL driver
    is configured.
    """
    if config.driver.startswith("postgresql"):
        return {
            "connect_timeout": config.connect_timeout,
            "application_name": config.application_name,
        }

    return {}


def create_database_engine(
    config: DatabaseConfig | None = None,
) -> Engine:
    """
    Create a configured SQLAlchemy Engine.

    The function does not itself run migrations or create application tables.
    Schema lifecycle belongs to the migration layer.
    """
    settings = config or get_database_config()

    logger.info(
        "Creating OSHA database engine for %s",
        settings.safe_url,
    )

    try:
        engine = create_engine(
            settings.url,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout,
            pool_recycle=settings.pool_recycle,
            pool_pre_ping=settings.pool_pre_ping,
            echo=settings.echo,
            echo_pool=settings.echo_pool,
            connect_args=_build_connect_args(settings),
            hide_parameters=True,
        )
    except (SQLAlchemyError, TypeError, ValueError) as exc:
        logger.exception(
            "Failed to create OSHA database engine for %s",
            settings.safe_url,
        )
        raise DatabaseConnectionError(
            "Failed to create OSHA database engine"
        ) from exc

    return engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    Return the process-wide OSHA SQLAlchemy Engine.

    The engine is created lazily on first use and then reused.
    """
    return create_database_engine()


def check_database_connection(
    engine: Engine | None = None,
) -> bool:
    """
    Check whether the relational database is reachable.

    Executes a minimal SELECT statement through SQLAlchemy.

    Returns:
        True when the database responds successfully.
        False when SQLAlchemy reports a database error.
    """
    database_engine = engine or get_engine()

    try:
        with database_engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return True

    except SQLAlchemyError:
        logger.exception("OSHA database health check failed")
        return False


def require_database_connection(
    engine: Engine | None = None,
) -> None:
    """
    Verify database connectivity and raise on failure.

    Intended for startup checks where continuing without the database would
    leave the service unable to operate correctly.
    """
    if not check_database_connection(engine):
        raise DatabaseConnectionError(
            "OSHA relational database is unavailable"
        )


def get_pool_status(
    engine: Engine | None = None,
) -> dict[str, Any]:
    """
    Return safe connection-pool diagnostics.

    Exact numeric counters depend on the pool implementation. The textual
    status remains available as a general diagnostic representation.
    """
    database_engine = engine or get_engine()
    pool = database_engine.pool

    result: dict[str, Any] = {
        "status": pool.status(),
    }

    for attribute in (
        "size",
        "checkedin",
        "checkedout",
        "overflow",
    ):
        method = getattr(pool, attribute, None)

        if callable(method):
            try:
                result[attribute] = method()
            except (NotImplementedError, TypeError):
                pass

    return result


def dispose_engine(
    engine: Engine | None = None,
) -> None:
    """
    Dispose an SQLAlchemy Engine and its pooled connections.

    When disposing the process-wide cached engine, the cache is also cleared
    so a later call to get_engine() creates a fresh Engine.
    """
    if engine is not None:
        logger.info("Disposing OSHA database engine")
        engine.dispose()
        return

    cached_engine: Engine | None = None

    if get_engine.cache_info().currsize:
        cached_engine = get_engine()

    if cached_engine is not None:
        logger.info("Disposing OSHA process-wide database engine")
        cached_engine.dispose()

    get_engine.cache_clear()


def reset_engine() -> None:
    """
    Dispose and clear the cached process-wide Engine.

    Primarily useful during controlled application reconfiguration and tests.
    """
    dispose_engine()


__all__ = [
    "DatabaseConnectionError",
    "check_database_connection",
    "create_database_engine",
    "dispose_engine",
    "get_engine",
    "get_pool_status",
    "require_database_connection",
    "reset_engine",
]
