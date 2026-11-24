from __future__ import annotations

"""
Database dependency health/readiness checks for
OSHA.
"""

from time import (
    perf_counter,
)

from typing import (
    Any,
    Dict,
    Optional,
)

from src.database.connection import (
    DatabaseConnection,
)

from src.database.schema import (
    ConnectionState,
    DatabaseHealth,
)


# ---------------------------------------------------------------
# Database health checker
# ---------------------------------------------------------------


class DatabaseHealthChecker:
    """
    Probe OSHA's configured persistence
    dependency without exposing driver or
    credential details.

    This reports health of OSHA's database
    dependency only.

    It does not represent health of external
    infrastructure monitored by OSHA.
    """

    def __init__(
        self,
        *,
        connection: DatabaseConnection,
        degraded_latency_seconds: float = 1.0,
    ) -> None:
        if degraded_latency_seconds <= 0:
            raise ValueError("degraded_latency_seconds must be > 0")

        self.connection = connection

        self.degraded_latency_seconds = float(degraded_latency_seconds)

        self._last: Optional[DatabaseHealth] = None

    # ---------------------------------------------------------
    # Health check
    # ---------------------------------------------------------

    def check(
        self,
    ) -> DatabaseHealth:
        """
        Run one lightweight readiness probe.

        Results:

            healthy
            degraded
            unhealthy
            unknown
        """

        # -----------------------------------------------------
        # Connection not ready
        # -----------------------------------------------------

        if not self.connection.ready:
            state = self.connection.status.state

            status = (
                "unhealthy"
                if state
                in {
                    ConnectionState.FAILED.value,
                    ConnectionState.CLOSED.value,
                }
                else "unknown"
            )

            self._last = DatabaseHealth(
                status=status,
                connection_id=(self.connection.status.connection_id),
                message=("Database connection is not ready."),
                attributes={
                    "connection_state": (state),
                    "error_type": (self.connection.status.last_error_type),
                },
            ).finalize()

            return self._last

        # -----------------------------------------------------
        # Readiness ping
        # -----------------------------------------------------

        started = perf_counter()

        try:
            healthy = self.connection.ping()

        except Exception as exc:
            latency = max(
                0.0,
                perf_counter() - started,
            )

            self._last = DatabaseHealth(
                status="unhealthy",
                connection_id=(self.connection.status.connection_id),
                latency_seconds=(latency),
                message=("Database health probe failed."),
                attributes={
                    "error_type": (type(exc).__name__),
                    "error_module": (type(exc).__module__),
                },
            ).finalize()

            return self._last

        latency = max(
            0.0,
            perf_counter() - started,
        )

        # -----------------------------------------------------
        # Determine health status
        # -----------------------------------------------------

        if not healthy:
            status = "unhealthy"

            message = "Database readiness probe failed."

        elif latency >= self.degraded_latency_seconds:
            status = "degraded"

            message = (
                "Database readiness probe exceeded the configured latency threshold."
            )

        else:
            status = "healthy"

            message = "Database dependency is ready."

        self._last = DatabaseHealth(
            status=status,
            connection_id=(self.connection.status.connection_id),
            latency_seconds=(latency),
            message=message,
            attributes={
                "connection_state": (self.connection.status.state),
                "degraded_latency_seconds": (self.degraded_latency_seconds),
                "backend": (self.connection.config.backend),
            },
        ).finalize()

        return self._last

    # ---------------------------------------------------------
    # Last observation
    # ---------------------------------------------------------

    @property
    def last(
        self,
    ) -> Optional[DatabaseHealth]:
        return self._last

    # ---------------------------------------------------------
    # Readiness convenience
    # ---------------------------------------------------------

    def ready(
        self,
    ) -> bool:
        """
        Return True only for a healthy result.

        DEGRADED intentionally does not count as
        fully ready in this helper.
        """

        return self.check().status == "healthy"

    # ---------------------------------------------------------
    # Safe summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "connection_ready": (self.connection.ready),
            "connection_state": (self.connection.status.state),
            "degraded_latency_seconds": (self.degraded_latency_seconds),
            "last_health": (self._last.to_dict() if self._last is not None else None),
            "scope": ("osha_database_dependency"),
            "credentials_exposed": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DatabaseHealthChecker",
]
