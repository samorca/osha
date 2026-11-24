from __future__ import annotations

"""
Credential resolution boundary for OSHA
infrastructure integrations.
"""

from dataclasses import (
    dataclass,
)

from typing import (
    Any,
    Dict,
    Optional,
)

from src.integrations.errors import (
    IntegrationAuthenticationError,
    IntegrationConfigurationError,
)

from src.integrations.schema import (
    IntegrationTarget,
)

from src.security.secrets import (
    SecretError,
    SecretManager,
    SecretValue,
)


# ---------------------------------------------------------------
# Credential resolution metadata
# ---------------------------------------------------------------


@dataclass(frozen=True)
class CredentialResolution:
    """
    Non-secret metadata describing one
    credential-resolution attempt.

    This deliberately does NOT contain:

      password
      token
      credential value
      private key
    """

    provider: str

    target_id: str

    configured: bool

    resolved: bool

    secret_provider: Optional[str] = None

    reference_id: Optional[str] = None

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "provider": (self.provider),
            "target_id": (self.target_id),
            "configured": (self.configured),
            "resolved": (self.resolved),
            "secret_provider": (self.secret_provider),
            "reference_id": (self.reference_id),
            "secret_value_exposed": (False),
        }


# ---------------------------------------------------------------
# Credential resolver
# ---------------------------------------------------------------


class IntegrationCredentialResolver:
    """
    Resolve an IntegrationTarget's
    SecretReference through the security
    SecretManager.

    The returned SecretValue should be passed
    directly into the integration adapter's
    connection boundary.

    It should not be copied into integration
    status, logging, telemetry, or operation
    metadata.
    """

    def __init__(
        self,
        *,
        secrets: Optional[SecretManager] = None,
    ) -> None:
        self.secrets = secrets

        self._last: Optional[CredentialResolution] = None

    # ---------------------------------------------------------
    # Last resolution
    # ---------------------------------------------------------

    @property
    def last(
        self,
    ) -> Optional[CredentialResolution]:
        return self._last

    # ---------------------------------------------------------
    # Resolve
    # ---------------------------------------------------------

    def resolve(
        self,
        target: IntegrationTarget,
        *,
        required: bool = False,
    ) -> Optional[SecretValue]:
        """
        Resolve target credentials.

        Returns None when:

          target has no credential reference
          AND
          credentials aren't required

        Raises when a configured/required
        credential cannot be resolved.
        """

        target.finalize()

        reference = target.credential

        # -----------------------------------------------------
        # No credential reference
        # -----------------------------------------------------

        if reference is None:
            self._last = CredentialResolution(
                provider=(target.provider),
                target_id=(target.target_id),
                configured=False,
                resolved=False,
            )

            if required:
                raise (
                    IntegrationConfigurationError(
                        "Integration target "
                        "requires credentials "
                        "but no credential "
                        "reference is "
                        "configured."
                    )
                )

            return None

        # -----------------------------------------------------
        # No SecretManager
        # -----------------------------------------------------

        if self.secrets is None:
            self._last = CredentialResolution(
                provider=(target.provider),
                target_id=(target.target_id),
                configured=True,
                resolved=False,
                secret_provider=(reference.provider),
                reference_id=(reference.reference_id),
            )

            raise (
                IntegrationConfigurationError(
                    "Integration credential "
                    "reference is configured "
                    "but no SecretManager "
                    "is available."
                )
            )

        # -----------------------------------------------------
        # Resolve
        # -----------------------------------------------------

        try:
            value = self.secrets.resolve(reference)

        except SecretError as exc:
            self._last = CredentialResolution(
                provider=(target.provider),
                target_id=(target.target_id),
                configured=True,
                resolved=False,
                secret_provider=(reference.provider),
                reference_id=(reference.reference_id),
            )

            # Do not copy the secret manager's
            # arbitrary error text.
            raise (
                IntegrationAuthenticationError(
                    "Integration credential resolution failed.",
                    details={
                        "provider": (target.provider),
                        "target_id": (target.target_id),
                        "secret_provider": (reference.provider),
                        "reference_id": (reference.reference_id),
                        "error_type": (type(exc).__name__),
                        "error_module": (type(exc).__module__),
                    },
                )
            ) from exc

        # -----------------------------------------------------
        # Resolved
        # -----------------------------------------------------

        self._last = CredentialResolution(
            provider=(target.provider),
            target_id=(target.target_id),
            configured=True,
            resolved=True,
            secret_provider=(reference.provider),
            reference_id=(reference.reference_id),
        )

        return value

    # ---------------------------------------------------------
    # Invalidate cached credential
    # ---------------------------------------------------------

    def invalidate(
        self,
        target: IntegrationTarget,
    ) -> bool:
        """
        Invalidate this target's underlying secret
        cache entry if SecretManager caching is
        enabled.
        """

        target.finalize()

        if target.credential is None or self.secrets is None:
            return False

        return self.secrets.invalidate(target.credential)

    # ---------------------------------------------------------
    # Safe summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "secret_manager_registered": (self.secrets is not None),
            "last_resolution": (self._last.to_dict() if self._last else None),
            "secret_value_exposed": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "CredentialResolution",
    "IntegrationCredentialResolver",
]
