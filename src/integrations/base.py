from __future__ import annotations

"""
Base lifecycle and dispatch contract for OSHA
infrastructure integrations.
"""

from abc import (
    ABC,
    abstractmethod,
)

from threading import RLock

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.core.time import (
    utc_now,
)

from src.integrations.config import (
    IntegrationsConfig,
)

from src.integrations.errors import (
    IntegrationCapabilityError,
    IntegrationConnectionError,
    IntegrationOperationBlockedError,
    IntegrationOperationError,
    IntegrationUnavailableError,
    wrap_provider_error,
)

from src.integrations.schema import (
    IntegrationCapability,
    IntegrationOperation,
    IntegrationResult,
    IntegrationState,
    IntegrationStatus,
    IntegrationTarget,
    OperationStatus,
)

from src.security.schema import (
    AuthorizationDecision,
    SecurityContext,
)

from src.security.secrets import (
    SecretValue,
)


# ---------------------------------------------------------------
# Integration adapter
# ---------------------------------------------------------------


class IntegrationAdapter(ABC):
    """
    Provider-neutral integration lifecycle and
    operation-dispatch boundary.

    Every provider adapter must implement:

      define_capabilities()
      connect_client()
      close_client()
      ping_client()
      execute_operation()

    Provider SDK/client objects remain behind
    this boundary.
    """

    def __init__(
        self,
        *,
        target: IntegrationTarget,
        config: Optional[IntegrationsConfig] = None,
    ) -> None:
        self.target = target.finalize()

        self.config = config or IntegrationsConfig()

        self.config.validate()

        if not self.config.provider_enabled(self.target.provider):
            raise (
                IntegrationConnectionError(
                    f"Integration provider is disabled: {self.target.provider}"
                )
            )

        self.status = IntegrationStatus(target=self.target).finalize()

        self._client: Any = None

        self._capabilities: Dict[
            str,
            IntegrationCapability,
        ] = {}

        self._lock = RLock()

        # -----------------------------------------------------
        # Register advertised capabilities
        # -----------------------------------------------------

        for capability in self.define_capabilities():
            capability.finalize()

            if capability.name in self._capabilities:
                raise (
                    IntegrationCapabilityError(
                        f"Duplicate integration capability: {capability.name}"
                    )
                )

            self._capabilities[capability.name] = capability

        self.status.capabilities = sorted(self._capabilities)

        self.status.finalize()

    # ---------------------------------------------------------
    # Provider contracts
    # ---------------------------------------------------------

    @abstractmethod
    def define_capabilities(
        self,
    ) -> Iterable[IntegrationCapability]:
        """
        Return capabilities explicitly supported
        by this adapter.
        """

        raise NotImplementedError

    @abstractmethod
    def connect_client(
        self,
        credential: Optional[SecretValue],
    ) -> Any:
        """
        Create and authenticate the
        provider-specific client.

        credential may be None when the provider
        uses another explicitly supported
        authentication mechanism.
        """

        raise NotImplementedError

    @abstractmethod
    def close_client(
        self,
        client: Any,
    ) -> None:
        """
        Close the provider client and release
        associated resources.
        """

        raise NotImplementedError

    @abstractmethod
    def ping_client(
        self,
        client: Any,
    ) -> bool:
        """
        Perform a lightweight provider
        connectivity/readiness check.
        """

        raise NotImplementedError

    @abstractmethod
    def execute_operation(
        self,
        client: Any,
        operation: IntegrationOperation,
    ) -> Any:
        """
        Translate and execute one operation using
        the provider-specific client.
        """

        raise NotImplementedError

    # ---------------------------------------------------------
    # Ready
    # ---------------------------------------------------------

    @property
    def ready(
        self,
    ) -> bool:
        with self._lock:
            return bool(
                self.status.state == IntegrationState.READY.value
                and self._client is not None
            )

    # ---------------------------------------------------------
    # Client
    # ---------------------------------------------------------

    @property
    def client(
        self,
    ) -> Any:
        with self._lock:
            if not self.ready:
                raise (
                    IntegrationUnavailableError(
                        "Integration adapter is not ready.",
                        details={
                            "provider": (self.target.provider),
                            "target_id": (self.target.target_id),
                        },
                    )
                )

            return self._client

    # ---------------------------------------------------------
    # Capabilities
    # ---------------------------------------------------------

    def capabilities(
        self,
    ) -> List[IntegrationCapability]:
        with self._lock:
            return [self._capabilities[name] for name in sorted(self._capabilities)]

    def capability(
        self,
        name: str,
    ) -> Optional[IntegrationCapability]:
        with self._lock:
            return self._capabilities.get(name)

    # ---------------------------------------------------------
    # Connect
    # ---------------------------------------------------------

    def connect(
        self,
        *,
        credential: Optional[SecretValue] = None,
    ) -> IntegrationStatus:
        with self._lock:
            if self.ready:
                return self.status

            if self.status.state not in {
                IntegrationState.CREATED.value,
                IntegrationState.CLOSED.value,
                IntegrationState.FAILED.value,
                IntegrationState.DEGRADED.value,
            }:
                raise (
                    IntegrationConnectionError(
                        f"Integration cannot connect from state {self.status.state}."
                    )
                )

            self.status.state = IntegrationState.CONNECTING.value

            self.status.last_error_type = None

            self.status.finalize()

        # -----------------------------------------------------
        # Create client and verify readiness
        # -----------------------------------------------------

        try:
            client = self.connect_client(credential)

            if client is None:
                raise (
                    IntegrationConnectionError(
                        "Integration adapter returned no provider client."
                    )
                )

            if not self.ping_client(client):
                try:
                    self.close_client(client)
                except Exception:
                    pass

                raise (
                    IntegrationUnavailableError(
                        "Integration target did not pass readiness validation."
                    )
                )

        except IntegrationConnectionError as exc:
            with self._lock:
                self._client = None

                self.status.state = IntegrationState.FAILED.value

                self.status.last_error_type = type(exc).__name__

                self.status.finalize()

            raise

        except Exception as exc:
            with self._lock:
                self._client = None

                self.status.state = IntegrationState.FAILED.value

                self.status.last_error_type = type(exc).__name__

                self.status.finalize()

            raise wrap_provider_error(
                exc,
                provider=(self.target.provider),
                target_id=(self.target.target_id),
                message=("Integration connection failed."),
                retryable=True,
                code=("integration_connection_failed"),
            ) from exc

        # -----------------------------------------------------
        # Connected
        # -----------------------------------------------------

        with self._lock:
            self._client = client

            self.status.state = IntegrationState.READY.value

            self.status.connected_at = utc_now()

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
        with self._lock:
            client = self._client

        if client is None:
            return False

        try:
            healthy = bool(self.ping_client(client))

        except Exception as exc:
            with self._lock:
                self.status.state = IntegrationState.DEGRADED.value

                self.status.last_error_type = type(exc).__name__

                self.status.finalize()

            return False

        with self._lock:
            self.status.state = (
                IntegrationState.READY.value
                if healthy
                else (IntegrationState.DEGRADED.value)
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
    ) -> IntegrationStatus:
        with self._lock:
            if self.status.state == IntegrationState.CLOSED.value:
                return self.status

            client = self._client

            self.status.state = IntegrationState.CLOSING.value

            self.status.finalize()

        if client is not None:
            try:
                self.close_client(client)

            except Exception as exc:
                with self._lock:
                    self.status.state = IntegrationState.FAILED.value

                    self.status.last_error_type = type(exc).__name__

                    self.status.finalize()

                raise wrap_provider_error(
                    exc,
                    provider=(self.target.provider),
                    target_id=(self.target.target_id),
                    message=("Integration close failed."),
                    retryable=False,
                    code=("integration_close_failed"),
                ) from exc

        with self._lock:
            self._client = None

            self.status.state = IntegrationState.CLOSED.value

            self.status.closed_at = utc_now()

            self.status.finalize()

            return self.status

    # ---------------------------------------------------------
    # Target validation
    # ---------------------------------------------------------

    def _validate_operation_target(
        self,
        operation: IntegrationOperation,
    ) -> None:
        operation.finalize()

        if operation.target.provider != self.target.provider:
            raise (
                IntegrationOperationBlockedError(
                    "Operation provider does not match adapter provider."
                )
            )

        if operation.target.environment_id != self.target.environment_id:
            raise (
                IntegrationOperationBlockedError(
                    "Operation environment does not match adapter environment."
                )
            )

        if operation.target.target_id != self.target.target_id:
            raise (
                IntegrationOperationBlockedError(
                    "Operation target does not match adapter target."
                )
            )

    # ---------------------------------------------------------
    # Capability validation
    # ---------------------------------------------------------

    def _validate_capability(
        self,
        operation: IntegrationOperation,
    ) -> IntegrationCapability:
        capability = self.capability(operation.capability)

        if capability is None:
            raise (
                IntegrationCapabilityError(
                    f"Adapter does not advertise capability: {operation.capability}"
                )
            )

        if operation.action not in capability.actions:
            raise (
                IntegrationCapabilityError(
                    f"Capability "
                    f"{capability.name} "
                    "does not support "
                    f"action "
                    f"{operation.action}."
                )
            )

        return capability

    # ---------------------------------------------------------
    # Security and mutation validation
    # ---------------------------------------------------------

    def _validate_security_context(
        self,
        *,
        capability: IntegrationCapability,
        security_context: Optional[SecurityContext],
    ) -> None:
        # -----------------------------------------------------
        # Read-only operation
        # -----------------------------------------------------

        if capability.read_only:
            if not (self.config.operations.allow_read_operations):
                raise (
                    IntegrationOperationBlockedError(
                        "Read integration operations are disabled."
                    )
                )

            return

        # -----------------------------------------------------
        # Mutation globally disabled
        # -----------------------------------------------------

        if not (self.config.operations.allow_mutating_operations):
            raise (
                IntegrationOperationBlockedError(
                    "Mutating integration operations are disabled."
                )
            )

        # -----------------------------------------------------
        # Require authorized security context
        # -----------------------------------------------------

        if self.config.operations.require_security_context_for_mutations:
            if security_context is None:
                raise (
                    IntegrationOperationBlockedError(
                        "Mutating operation requires a security context."
                    )
                )

            security_context.finalize()

            if security_context.authorization is None:
                raise (
                    IntegrationOperationBlockedError(
                        "Mutating operation requires an authorization result."
                    )
                )

            if (
                security_context.authorization.decision
                != AuthorizationDecision.ALLOW.value
            ):
                raise (
                    IntegrationOperationBlockedError(
                        "Mutating operation was not authorized."
                    )
                )

            if security_context.environment_id != self.target.environment_id:
                raise (
                    IntegrationOperationBlockedError(
                        "Security context "
                        "environment does "
                        "not match "
                        "integration target."
                    )
                )

    # ---------------------------------------------------------
    # Dispatch
    # ---------------------------------------------------------

    def dispatch(
        self,
        operation: IntegrationOperation,
        *,
        security_context: Optional[SecurityContext] = None,
    ) -> IntegrationResult:
        """
        Validate and execute one integration
        operation.

        This base implementation deliberately
        does not automatically retry mutating
        operations.
        """

        operation.finalize()

        try:
            if not self.ready:
                raise (IntegrationUnavailableError("Integration adapter is not ready."))

            self._validate_operation_target(operation)

            capability = self._validate_capability(operation)

            self._validate_security_context(
                capability=capability,
                security_context=(security_context),
            )

            value = self.execute_operation(
                self.client,
                operation,
            )

            return IntegrationResult(
                operation_id=(operation.operation_id),
                status=(OperationStatus.SUCCEEDED.value),
                value=value,
                reference_ids=list(operation.reference_ids),
                attributes={
                    "provider": (self.target.provider),
                    "target_id": (self.target.target_id),
                    "capability": (capability.name),
                    "read_only": (capability.read_only),
                },
            ).finalize()

        # -----------------------------------------------------
        # Deliberately blocked
        # -----------------------------------------------------

        except IntegrationOperationBlockedError as exc:
            return IntegrationResult(
                operation_id=(operation.operation_id),
                status=(OperationStatus.BLOCKED.value),
                error_type=(type(exc).__name__),
                retryable=False,
                reference_ids=list(operation.reference_ids),
                attributes={
                    "provider": (self.target.provider),
                    "target_id": (self.target.target_id),
                },
            ).finalize()

        # -----------------------------------------------------
        # Invalid/unsupported operation
        # -----------------------------------------------------

        except IntegrationCapabilityError:
            # Programming/configuration error.
            # Keep it explicit rather than
            # flattening it into a provider
            # failure result.
            raise

        # -----------------------------------------------------
        # Target unavailable
        # -----------------------------------------------------

        except IntegrationUnavailableError as exc:
            return IntegrationResult(
                operation_id=(operation.operation_id),
                status=(OperationStatus.FAILED.value),
                error_type=(type(exc).__name__),
                retryable=True,
                reference_ids=list(operation.reference_ids),
                attributes={
                    "provider": (self.target.provider),
                    "target_id": (self.target.target_id),
                },
            ).finalize()

        # -----------------------------------------------------
        # Known operation error
        # -----------------------------------------------------

        except IntegrationOperationError as exc:
            return IntegrationResult(
                operation_id=(operation.operation_id),
                status=(OperationStatus.FAILED.value),
                error_type=(type(exc).__name__),
                retryable=bool(
                    getattr(
                        exc,
                        "retryable",
                        False,
                    )
                ),
                reference_ids=list(operation.reference_ids),
                attributes={
                    "provider": (self.target.provider),
                    "target_id": (self.target.target_id),
                },
            ).finalize()

        # -----------------------------------------------------
        # Raw provider/client failure
        # -----------------------------------------------------

        except Exception as exc:
            wrapped = wrap_provider_error(
                exc,
                provider=(self.target.provider),
                target_id=(self.target.target_id),
                message=("Integration operation failed."),
                retryable=False,
                code=("integration_operation_failed"),
            )

            return IntegrationResult(
                operation_id=(operation.operation_id),
                status=(OperationStatus.FAILED.value),
                error_type=(type(wrapped).__name__),
                retryable=(wrapped.retryable),
                reference_ids=list(operation.reference_ids),
                attributes={
                    "provider": (self.target.provider),
                    "target_id": (self.target.target_id),
                    **dict(wrapped.details),
                },
            ).finalize()

    # ---------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------

    def __enter__(
        self,
    ) -> "IntegrationAdapter":
        self.connect()

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:
        self.close()

        return False

    # ---------------------------------------------------------
    # Safe summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "status": (self.status.to_dict()),
            "ready": (self.ready),
            "adapter": (type(self).__name__),
            "capabilities": [item.to_dict() for item in self.capabilities()],
            "mutating_operations_enabled": (
                self.config.operations.allow_mutating_operations
            ),
            "credential_value_exposed": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "IntegrationAdapter",
]
