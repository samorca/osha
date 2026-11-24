from __future__ import annotations

"""
Connectivity health probes for OSHA integration
adapters.
"""

from time import (
    perf_counter,
)

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from src.integrations.base import (
    IntegrationAdapter,
)

from src.integrations.config import (
    HealthConfig,
)

from src.integrations.registry import (
    IntegrationRegistry,
)

from src.integrations.schema import (
    IntegrationHealth,
    IntegrationHealthStatus,
    IntegrationState,
)


# ---------------------------------------------------------------
# Integration health checker
# ---------------------------------------------------------------


class IntegrationHealthChecker:
    """
    Probe adapter connectivity without claiming
    broader infrastructure/workload health.

    Example:

        HEALTHY here means:

            OSHA can communicate with the
            configured provider target.

        It does NOT automatically mean:

            all VMs are healthy
            all Kubernetes workloads are healthy
            Ceph is healthy
            every Linux service is healthy

    Those infrastructure conclusions belong in
    collectors/telemetry/intelligence.
    """

    def __init__(
        self,
        *,
        registry: IntegrationRegistry,
        config: Optional[HealthConfig] = None,
    ) -> None:
        self.registry = registry

        self.config = config or HealthConfig()

        self.config.validate()

        self._last: Dict[
            str,
            IntegrationHealth,
        ] = {}

    # ---------------------------------------------------------
    # Internal target key
    # ---------------------------------------------------------

    @staticmethod
    def _key(
        adapter: IntegrationAdapter,
    ) -> str:
        return "/".join(
            (
                adapter.target.environment_id,
                adapter.target.provider,
                adapter.target.target_id,
            )
        )

    # ---------------------------------------------------------
    # Check one adapter
    # ---------------------------------------------------------

    def check_adapter(
        self,
        adapter: IntegrationAdapter,
    ) -> IntegrationHealth:
        """
        Run one lightweight integration readiness
        check.
        """

        # -----------------------------------------------------
        # Health probing disabled
        # -----------------------------------------------------

        if not self.config.enabled:
            result = IntegrationHealth(
                target=adapter.target,
                status=(IntegrationHealthStatus.UNKNOWN.value),
                integration_id=(adapter.status.integration_id),
                message=("Integration health checks are disabled."),
                attributes={
                    "adapter_state": (adapter.status.state),
                },
            ).finalize()

            self._last[self._key(adapter)] = result

            return result

        # -----------------------------------------------------
        # Adapter not ready
        # -----------------------------------------------------

        if not adapter.ready:
            state = adapter.status.state

            status = (
                IntegrationHealthStatus.UNHEALTHY.value
                if state
                in {
                    IntegrationState.FAILED.value,
                    IntegrationState.CLOSED.value,
                }
                else (IntegrationHealthStatus.UNKNOWN.value)
            )

            result = IntegrationHealth(
                target=adapter.target,
                status=status,
                integration_id=(adapter.status.integration_id),
                message=("Integration adapter is not ready."),
                error_type=(adapter.status.last_error_type),
                attributes={
                    "adapter_state": (state),
                },
            ).finalize()

            self._last[self._key(adapter)] = result

            return result

        # -----------------------------------------------------
        # Provider connectivity probe
        # -----------------------------------------------------

        started = perf_counter()

        try:
            healthy = adapter.ping()

        except Exception as exc:
            latency = max(
                0.0,
                perf_counter() - started,
            )

            result = IntegrationHealth(
                target=adapter.target,
                status=(IntegrationHealthStatus.UNHEALTHY.value),
                integration_id=(adapter.status.integration_id),
                latency_seconds=(latency),
                message=("Integration health probe failed."),
                error_type=(type(exc).__name__),
                attributes={
                    "error_module": (type(exc).__module__),
                    "adapter_state": (adapter.status.state),
                },
            ).finalize()

            self._last[self._key(adapter)] = result

            return result

        latency = max(
            0.0,
            perf_counter() - started,
        )

        # -----------------------------------------------------
        # Determine health
        # -----------------------------------------------------

        if not healthy:
            status = IntegrationHealthStatus.UNHEALTHY.value

            message = "Integration readiness probe failed."

        elif latency >= self.config.degraded_latency_seconds:
            status = IntegrationHealthStatus.DEGRADED.value

            message = "Integration readiness latency exceeded the configured threshold."

        else:
            status = IntegrationHealthStatus.HEALTHY.value

            message = "Integration adapter can reach its configured target."

        result = IntegrationHealth(
            target=adapter.target,
            status=status,
            integration_id=(adapter.status.integration_id),
            latency_seconds=(latency),
            message=message,
            error_type=(adapter.status.last_error_type),
            attributes={
                "adapter_state": (adapter.status.state),
                "degraded_latency_seconds": (self.config.degraded_latency_seconds),
                "scope": ("integration_connectivity"),
                "infrastructure_health_claimed": (False),
            },
        ).finalize()

        self._last[self._key(adapter)] = result

        return result

    # ---------------------------------------------------------
    # Check all registered adapters
    # ---------------------------------------------------------

    def check_all(
        self,
        *,
        environment: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> List[IntegrationHealth]:
        """
        Probe registered integrations matching
        the optional filters.
        """

        return [
            self.check_adapter(adapter)
            for adapter in self.registry.list(
                environment=environment,
                provider=provider,
            )
        ]

    # ---------------------------------------------------------
    # Last health result for a target
    # ---------------------------------------------------------

    def last(
        self,
        *,
        environment: str,
        provider: str,
        target_id: str,
    ) -> Optional[IntegrationHealth]:
        key = "/".join(
            (
                environment,
                provider,
                target_id,
            )
        )

        return self._last.get(key)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        counts: Dict[
            str,
            int,
        ] = {}

        by_provider: Dict[
            str,
            Dict[
                str,
                int,
            ],
        ] = {}

        for health in self._last.values():
            counts[health.status] = (
                counts.get(
                    health.status,
                    0,
                )
                + 1
            )

            provider_counts = by_provider.setdefault(
                health.target.provider,
                {},
            )

            provider_counts[health.status] = (
                provider_counts.get(
                    health.status,
                    0,
                )
                + 1
            )

        return {
            "enabled": (self.config.enabled),
            "observed_targets": (len(self._last)),
            "by_status": (counts),
            "by_provider": (by_provider),
            "degraded_latency_seconds": (self.config.degraded_latency_seconds),
            "scope": ("integration_connectivity"),
            "infrastructure_health_claimed": (False),
            "credential_values_exposed": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "IntegrationHealthChecker",
]
