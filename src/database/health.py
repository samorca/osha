"""
OSHA Database Health Monitoring.

Provides structured health information for OSHA's relational database layer.

This module can support:

    - application startup checks
    - API health endpoints
    - Docker health checks
    - Kubernetes readiness checks
    - Kubernetes liveness diagnostics
    - operational monitoring
    - database pool diagnostics

Health checks must not expose database passwords, connection credentials,
tokens or other sensitive configuration.

The module reuses the process-wide SQLAlchemy Engine managed by OSHA's
connection layer.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from .config import get_database_config
from .connection import (
    get_engine,
    get_pool_status,
)
from .exceptions import (
    DatabaseHealthCheckError,
    DatabaseUnavailableError,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health status
# ---------------------------------------------------------------------------

class DatabaseHealthStatus(str, enum.Enum):
    """Health states reported by OSHA's database subsystem."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Health result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DatabaseHealthResult:
    """
    Structured result returned by a database health check.

    Sensitive connection information is intentionally excluded.
    """

    status: DatabaseHealthStatus

    reachable: bool

    checked_at: datetime

    latency_ms: float | None

    database: str | None
    host: str | None
    port: int | None
    driver: str | None

    pool: dict[str, Any]

    error: str | None = None

    @property
    def healthy(self) -> bool:
        """Return whether the database is healthy."""
        return self.status == DatabaseHealthStatus.HEALTHY

    @property
    def ready(self) -> bool:
        """
        Return whether OSHA should consider the relational database ready.

        A reachable degraded database remains available for traffic while
        monitoring systems can separately alert on degradation.
        """
        return (
            self.reachable
            and self.status
            in {
                DatabaseHealthStatus.HEALTHY,
                DatabaseHealthStatus.DEGRADED,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable, non-sensitive health representation."""
        return {
            "status": self.status.value,
            "healthy": self.healthy,
            "ready": self.ready,
            "reachable": self.reachable,
            "checked_at": self.checked_at.isoformat(),
            "latency_ms": self.latency_ms,
            "database": self.database,
            "host": self.host,
            "port": self.port,
            "driver": self.driver,
            "pool": self.pool,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Health service
# ---------------------------------------------------------------------------

class DatabaseHealthService:
    """
    Perform structured health checks for OSHA's relational database.

    The service accepts an optional Engine for testing or specialised
    integration scenarios. Otherwise it uses OSHA's process-wide Engine.
    """

    def __init__(
        self,
        engine: Engine | None = None,
    ) -> None:
        self._engine = engine

    @property
    def engine(self) -> Engine:
        """Return the SQLAlchemy Engine used for health checks."""
        return self._engine or get_engine()

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def check(
        self,
    ) -> DatabaseHealthResult:
        """
        Perform a relational database health check.

        The health query intentionally performs the smallest practical
        operation:

            SELECT 1

        The result includes:
            - connectivity
            - query latency
            - safe endpoint identity
            - pool diagnostics

        Database errors are converted into an unhealthy result rather than
        escaping from this method.
        """
        checked_at = datetime.now(
            timezone.utc
        )

        engine = self.engine

        started = time.perf_counter()

        try:
            with engine.connect() as connection:
                value = connection.scalar(
                    text("SELECT 1")
                )

            latency_ms = (
                time.perf_counter() - started
            ) * 1000.0

            if value != 1:
                return DatabaseHealthResult(
                    status=DatabaseHealthStatus.DEGRADED,
                    reachable=True,
                    checked_at=checked_at,
                    latency_ms=round(
                        latency_ms,
                        3,
                    ),
                    database=engine.url.database,
                    host=engine.url.host,
                    port=engine.url.port,
                    driver=engine.url.drivername,
                    pool=self._safe_pool_status(
                        engine
                    ),
                    error=(
                        "database health query returned "
                        "an unexpected result"
                    ),
                )

            return DatabaseHealthResult(
                status=DatabaseHealthStatus.HEALTHY,
                reachable=True,
                checked_at=checked_at,
                latency_ms=round(
                    latency_ms,
                    3,
                ),
                database=engine.url.database,
                host=engine.url.host,
                port=engine.url.port,
                driver=engine.url.drivername,
                pool=self._safe_pool_status(
                    engine
                ),
            )

        except SQLAlchemyError as exc:
            latency_ms = (
                time.perf_counter() - started
            ) * 1000.0

            logger.warning(
                "OSHA database health check failed: %s",
                exc.__class__.__name__,
            )

            return DatabaseHealthResult(
                status=DatabaseHealthStatus.UNHEALTHY,
                reachable=False,
                checked_at=checked_at,
                latency_ms=round(
                    latency_ms,
                    3,
                ),
                database=engine.url.database,
                host=engine.url.host,
                port=engine.url.port,
                driver=engine.url.drivername,
                pool=self._safe_pool_status(
                    engine
                ),
                error="database connection check failed",
            )

    # ------------------------------------------------------------------
    # Startup requirement
    # ------------------------------------------------------------------

    def require_healthy(
        self,
    ) -> DatabaseHealthResult:
        """
        Require the relational database to be healthy.

        This is suitable for service startup when OSHA cannot operate without
        its relational persistence layer.
        """
        result = self.check()

        if not result.healthy:
            raise DatabaseUnavailableError(
                "OSHA relational database is not healthy",
                code="database_unhealthy",
                context={
                    "status": result.status.value,
                    "reachable": result.reachable,
                },
            )

        return result

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def readiness(
        self,
    ) -> DatabaseHealthResult:
        """
        Return database readiness state.

        Readiness determines whether the database is available sufficiently
        for OSHA to accept work.
        """
        return self.check()

    # ------------------------------------------------------------------
    # Liveness
    # ------------------------------------------------------------------

    def liveness(
        self,
    ) -> dict[str, Any]:
        """
        Return process-side database subsystem liveness information.

        Liveness deliberately does not require a successful database query.

        Database unavailability should normally affect readiness rather than
        automatically implying that the OSHA process itself is dead.
        """
        try:
            engine = self.engine

            return {
                "status": "alive",
                "engine_available": engine is not None,
            }

        except Exception:
            logger.exception(
                "OSHA database engine liveness check failed"
            )

            return {
                "status": "unhealthy",
                "engine_available": False,
            }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(
        self,
    ) -> dict[str, Any]:
        """
        Return safe database diagnostics.

        Passwords and complete credential-bearing URLs are never returned.
        """
        try:
            engine = self.engine
            config = get_database_config()

            return {
                "driver": engine.url.drivername,
                "database": engine.url.database,
                "host": engine.url.host,
                "port": engine.url.port,
                "application_name": config.application_name,
                "pool": self._safe_pool_status(
                    engine
                ),
            }

        except Exception as exc:
            logger.exception(
                "Unable to generate OSHA database diagnostics"
            )

            raise DatabaseHealthCheckError(
                "Unable to generate database diagnostics",
                code="database_diagnostics_failed",
            ) from exc

    # ------------------------------------------------------------------
    # Pool diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_pool_status(
        engine: Engine,
    ) -> dict[str, Any]:
        """
        Return pool diagnostics without allowing diagnostics failures to
        break the primary health check.
        """
        try:
            return get_pool_status(
                engine
            )

        except Exception:
            logger.exception(
                "Unable to collect OSHA database pool diagnostics"
            )

            return {
                "status": "unavailable",
            }


# ---------------------------------------------------------------------------
# Process-wide health service
# ---------------------------------------------------------------------------

database_health_service = DatabaseHealthService()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def check_database_health() -> DatabaseHealthResult:
    """Run OSHA's default database health check."""
    return database_health_service.check()


def require_database_health() -> DatabaseHealthResult:
    """Require OSHA's relational database to be healthy."""
    return database_health_service.require_healthy()


def database_readiness() -> DatabaseHealthResult:
    """Return relational database readiness."""
    return database_health_service.readiness()


def database_liveness() -> dict[str, Any]:
    """Return database subsystem liveness information."""
    return database_health_service.liveness()


def database_health_diagnostics() -> dict[str, Any]:
    """Return safe relational database diagnostics."""
    return database_health_service.diagnostics()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DatabaseHealthResult",
    "DatabaseHealthService",
    "DatabaseHealthStatus",
    "check_database_health",
    "database_health_diagnostics",
    "database_health_service",
    "database_liveness",
    "database_readiness",
    "require_database_health",
]
