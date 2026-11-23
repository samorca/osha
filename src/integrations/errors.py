from __future__ import annotations

"""
Provider-neutral exception hierarchy for OSHA
infrastructure integrations.
"""

from typing import (
    Any,
    Mapping,
    Optional,
)

from src.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ExternalServiceError,
    NotFoundError,
    PermissionDeniedError,
    TimeoutErrorBase,
)


# ---------------------------------------------------------------
# Base integration error
# ---------------------------------------------------------------

class IntegrationError(
    ExternalServiceError
):
    """
    Base class for expected failures at OSHA's
    infrastructure integration boundary.
    """

    default_code = (
        "integration_error"
    )


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

class IntegrationConfigurationError(
    IntegrationError,
    ConfigurationError,
):
    default_code = (
        "integration_configuration_error"
    )


# ---------------------------------------------------------------
# Connection
# ---------------------------------------------------------------

class IntegrationConnectionError(
    IntegrationError
):
    default_code = (
        "integration_connection_error"
    )


class IntegrationUnavailableError(
    IntegrationConnectionError
):
    default_code = (
        "integration_unavailable"
    )

    def __init__(
        self,
        message: str = (
            "Integration target is "
            "unavailable."
        ),
        *,
        details: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        super().__init__(
            message,
            code=self.default_code,
            retryable=True,
            details=details,
        )


# ---------------------------------------------------------------
# Provider authentication
# ---------------------------------------------------------------

class IntegrationAuthenticationError(
    IntegrationError,
    AuthenticationError,
):
    """
    Authentication failure between OSHA and an
    external infrastructure provider.

    This is different from authenticating an OSHA
    user, which belongs in security/.
    """

    default_code = (
        "integration_authentication_error"
    )


# ---------------------------------------------------------------
# Provider permission
# ---------------------------------------------------------------

class IntegrationPermissionError(
    IntegrationError,
    PermissionDeniedError,
):
    """
    Provider rejected an already-dispatched
    operation because the integration identity
    lacks permission.
    """

    default_code = (
        "integration_permission_denied"
    )


# ---------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------

class IntegrationTimeoutError(
    IntegrationError,
    TimeoutErrorBase,
):
    default_code = (
        "integration_timeout"
    )

    def __init__(
        self,
        message: str = (
            "Integration operation "
            "timed out."
        ),
        *,
        details: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        super().__init__(
            message,
            code=self.default_code,
            retryable=True,
            details=details,
        )


# ---------------------------------------------------------------
# Capability
# ---------------------------------------------------------------

class IntegrationCapabilityError(
    IntegrationError
):
    """
    Requested action/capability is unsupported or
    wasn't explicitly advertised by the adapter.
    """

    default_code = (
        "integration_capability_error"
    )


# ---------------------------------------------------------------
# Operations
# ---------------------------------------------------------------

class IntegrationOperationError(
    IntegrationError
):
    default_code = (
        "integration_operation_error"
    )


class IntegrationOperationBlockedError(
    IntegrationOperationError
):
    """
    The integration layer deliberately refused to
    dispatch the operation.

    Examples may include an integration policy
    blocking mutation or a missing required
    execution context.
    """

    default_code = (
        "integration_operation_blocked"
    )


class IntegrationResourceNotFoundError(
    IntegrationOperationError,
    NotFoundError,
):
    default_code = (
        "integration_resource_not_found"
    )


# ---------------------------------------------------------------
# Health
# ---------------------------------------------------------------

class IntegrationHealthError(
    IntegrationError
):
    default_code = (
        "integration_health_error"
    )


# ---------------------------------------------------------------
# Registry
# ---------------------------------------------------------------

class IntegrationRegistryError(
    IntegrationError
):
    default_code = (
        "integration_registry_error"
    )


class IntegrationConflictError(
    IntegrationRegistryError
):
    default_code = (
        "integration_conflict"
    )


# ---------------------------------------------------------------
# Safe provider error metadata
# ---------------------------------------------------------------

def safe_provider_details(
    error: BaseException,
    *,
    provider: Optional[
        str
    ] = None,
    target_id: Optional[
        str
    ] = None,
) -> Mapping[str, Any]:
    """
    Build bounded metadata for a provider/client
    failure.

    The raw exception message is deliberately not
    copied.

    Provider exceptions can contain endpoint
    details, request payloads, authentication
    information, or other data we do not want
    leaking through normal application errors.
    """

    details: dict[
        str,
        Any,
    ] = {
        "error_type": (
            type(error).__name__
        ),

        "error_module": (
            type(error).__module__
        ),
    }

    if provider is not None:
        details[
            "provider"
        ] = provider

    if target_id is not None:
        details[
            "target_id"
        ] = target_id

    return details


# ---------------------------------------------------------------
# Provider error wrapper
# ---------------------------------------------------------------

def wrap_provider_error(
    error: BaseException,
    *,
    provider: Optional[
        str
    ] = None,
    target_id: Optional[
        str
    ] = None,
    message: str = (
        "Integration provider "
        "operation failed."
    ),
    retryable: bool = False,
    code: str = (
        "integration_provider_error"
    ),
) -> IntegrationError:
    """
    Convert a raw provider/client exception into
    the shared integration error model.

    Example future boundary:

        try:
            openstack_client.list_servers()
        except Exception as exc:
            raise wrap_provider_error(
                exc,
                provider="openstack",
                target_id="cloud-01",
            ) from exc
    """

    return IntegrationError(
        message,
        code=code,
        retryable=retryable,
        details=(
            safe_provider_details(
                error,
                provider=provider,
                target_id=target_id,
            )
        ),
    )


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "IntegrationError",

    "IntegrationConfigurationError",

    "IntegrationConnectionError",
    "IntegrationUnavailableError",

    "IntegrationAuthenticationError",
    "IntegrationPermissionError",

    "IntegrationTimeoutError",

    "IntegrationCapabilityError",

    "IntegrationOperationError",
    "IntegrationOperationBlockedError",
    "IntegrationResourceNotFoundError",

    "IntegrationHealthError",

    "IntegrationRegistryError",
    "IntegrationConflictError",

    "safe_provider_details",
    "wrap_provider_error",
]
