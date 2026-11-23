from __future__ import annotations

"""
Composition facade for OSHA infrastructure
integrations.
"""

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
    IntegrationsConfig,
    safe_config_summary,
)

from src.integrations.credentials import (
    IntegrationCredentialResolver,
)

from src.integrations.errors import (
    IntegrationOperationBlockedError,
)

from src.integrations.health import (
    IntegrationHealthChecker,
)

from src.integrations.registry import (
    IntegrationRegistry,
)

from src.integrations.schema import (
    IntegrationHealth,
    IntegrationOperation,
    IntegrationResult,
    IntegrationStatus,
)

from src.security.schema import (
    SecurityContext,
)

from src.security.secrets import (
    SecretManager,
)


# ---------------------------------------------------------------
# Integration pipeline
# ---------------------------------------------------------------

class IntegrationPipeline:
    """
    Unified facade for OSHA infrastructure
    integrations.

    Composes:

      integration configuration
      adapter registry
      credential resolution
      adapter lifecycle
      operation dispatch
      integration connectivity health

    Provider-specific implementation remains
    inside provider adapters.
    """

    def __init__(
        self,
        *,
        config: Optional[
            IntegrationsConfig
        ] = None,
        secrets: Optional[
            SecretManager
        ] = None,
        registry: Optional[
            IntegrationRegistry
        ] = None,
    ) -> None:
        self.config = (
            config
            or IntegrationsConfig()
        )

        self.config.validate()

        # -----------------------------------------------------
        # Adapter registry
        # -----------------------------------------------------

        self.registry = (
            registry
            or IntegrationRegistry()
        )

        # -----------------------------------------------------
        # Credential resolution
        # -----------------------------------------------------

        self.credentials = (
            IntegrationCredentialResolver(
                secrets=secrets
            )
        )

        # -----------------------------------------------------
        # Health
        # -----------------------------------------------------

        self.health = (
            IntegrationHealthChecker(
                registry=(
                    self.registry
                ),
                config=(
                    self.config.health
                ),
            )
        )

    # ---------------------------------------------------------
    # Register adapter
    # ---------------------------------------------------------

    def register(
        self,
        adapter: IntegrationAdapter,
        *,
        connect: bool = False,
        credential_required: bool = False,
    ) -> IntegrationAdapter:
        """
        Register one provider adapter.

        Optionally connect immediately.

        If immediate connection fails, the
        registration is rolled back.
        """

        # Keep pipeline-wide integration policy
        # authoritative.
        if (
            adapter.config
            != self.config
        ):
            adapter.config = (
                self.config
            )

            adapter.config.validate()

        self.registry.register(
            adapter
        )

        if connect:
            try:
                self.connect_adapter(
                    adapter,
                    credential_required=(
                        credential_required
                    ),
                )

            except Exception:
                self.registry.remove(
                    environment=(
                        adapter.target.
                        environment_id
                    ),
                    provider=(
                        adapter.target.
                        provider
                    ),
                    target_id=(
                        adapter.target.
                        target_id
                    ),
                    close=False,
                )

                raise

        return adapter

    # ---------------------------------------------------------
    # Connect adapter
    # ---------------------------------------------------------

    def connect_adapter(
        self,
        adapter: IntegrationAdapter,
        *,
        credential_required: bool = False,
    ) -> IntegrationStatus:
        """
        Resolve the target credential reference
        and connect the adapter.

        The SecretValue goes directly from the
        credential resolver into adapter.connect.
        """

        credential = (
            self.credentials.resolve(
                adapter.target,
                required=(
                    credential_required
                ),
            )
        )

        return adapter.connect(
            credential=credential
        )

    # ---------------------------------------------------------
    # Connect target
    # ---------------------------------------------------------

    def connect(
        self,
        *,
        environment: str,
        provider: str,
        target_id: str,
        credential_required: bool = False,
    ) -> IntegrationStatus:
        adapter = (
            self.registry.require(
                environment=environment,
                provider=provider,
                target_id=target_id,
            )
        )

        return self.connect_adapter(
            adapter,
            credential_required=(
                credential_required
            ),
        )

    # ---------------------------------------------------------
    # Connect all registered adapters
    # ---------------------------------------------------------

    def connect_all(
        self,
        *,
        credential_required: bool = False,
    ) -> Dict[
        str,
        str,
    ]:
        """
        Attempt to connect all registered
        adapters.

        One adapter failing does not prevent
        attempts against the remaining adapters.
        """

        results: Dict[
            str,
            str,
        ] = {}

        for adapter in (
            self.registry.list()
        ):
            key = "/".join(
                (
                    adapter.target.
                    environment_id,

                    adapter.target.
                    provider,

                    adapter.target.
                    target_id,
                )
            )

            try:
                status = (
                    self.connect_adapter(
                        adapter,
                        credential_required=(
                            credential_required
                        ),
                    )
                )

                results[
                    key
                ] = status.state

            except Exception as exc:
                # Do not expose provider error
                # messages here.
                results[
                    key
                ] = (
                    "error:"
                    + type(exc).__name__
                )

        return results

    # ---------------------------------------------------------
    # Disconnect
    # ---------------------------------------------------------

    def disconnect(
        self,
        *,
        environment: str,
        provider: str,
        target_id: str,
    ) -> IntegrationStatus:
        adapter = (
            self.registry.require(
                environment=environment,
                provider=provider,
                target_id=target_id,
            )
        )

        return adapter.close()

    # ---------------------------------------------------------
    # Dispatch operation
    # ---------------------------------------------------------

    def dispatch(
        self,
        operation: IntegrationOperation,
        *,
        security_context: Optional[
            SecurityContext
        ] = None,
    ) -> IntegrationResult:
        """
        Locate the exact registered adapter and
        dispatch the operation through the common
        IntegrationAdapter safety boundary.
        """

        operation.finalize()

        if not self.config.enabled:
            raise (
                IntegrationOperationBlockedError(
                    "Infrastructure "
                    "integrations are "
                    "disabled."
                )
            )

        adapter = (
            self.registry.require(
                environment=(
                    operation.target.
                    environment_id
                ),
                provider=(
                    operation.target.
                    provider
                ),
                target_id=(
                    operation.target.
                    target_id
                ),
            )
        )

        return adapter.dispatch(
            operation,
            security_context=(
                security_context
            ),
        )

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------

    def health_check(
        self,
        *,
        environment: Optional[
            str
        ] = None,
        provider: Optional[
            str
        ] = None,
    ) -> List[
        IntegrationHealth
    ]:
        return self.health.check_all(
            environment=environment,
            provider=provider,
        )

    # ---------------------------------------------------------
    # Start
    # ---------------------------------------------------------

    def start(
        self,
        *,
        connect_registered: bool = False,
        credential_required: bool = False,
    ) -> Dict[str, Any]:
        """
        Start the integration facade.

        Adapters are not connected automatically
        unless connect_registered=True.

        This avoids unexpected external
        connections merely from constructing or
        starting the application.
        """

        connections: Dict[
            str,
            str,
        ] = {}

        if connect_registered:
            connections = (
                self.connect_all(
                    credential_required=(
                        credential_required
                    ),
                )
            )

        health = (
            self.health_check()
            if self.config.
            health.enabled
            else []
        )

        return {
            "enabled": (
                self.config.enabled
            ),

            "connections": (
                connections
            ),

            "health": [
                item.to_dict()
                for item
                in health
            ],

            "registry": (
                self.registry.summary()
            ),
        }

    # ---------------------------------------------------------
    # Stop
    # ---------------------------------------------------------

    def stop(
        self,
    ) -> Dict[str, Any]:
        """
        Attempt to close every registered adapter.
        """

        return {
            "closed": (
                self.registry.
                close_all()
            ),

            "registry": (
                self.registry.summary()
            ),
        }

    # ---------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------

    def __enter__(
        self,
    ) -> "IntegrationPipeline":
        # Do not create external connections
        # automatically.
        self.start(
            connect_registered=False
        )

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
        return {
            "config": (
                safe_config_summary(
                    self.config
                )
            ),

            "registry": (
                self.registry.summary()
            ),

            "credentials": (
                self.credentials.
                summary()
            ),

            "health": (
                self.health.summary()
            ),

            "capabilities": (
                self.registry.
                capabilities()
            ),

            "credential_values_exposed": (
                False
            ),

            "infrastructure_health_claimed": (
                False
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "IntegrationPipeline",
]
