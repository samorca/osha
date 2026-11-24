from __future__ import annotations

"""
Provider-neutral authentication and bounded
session handling for OSHA.
"""

from dataclasses import (
    dataclass,
    field,
)

from datetime import (
    timedelta,
)

from threading import RLock

from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
)

from src.core.serialization import (
    DEFAULT_SENSITIVE_KEYS,
    to_json_safe,
)

from src.core.time import (
    is_expired,
    utc_datetime,
    utc_now,
)

from src.core.validation import (
    required_string,
)

from src.security.config import (
    AuthenticationConfig,
)

from src.security.schema import (
    AuthenticationContext,
    AuthenticationStatus,
    SecurityPrincipal,
)


# ---------------------------------------------------------------
# Errors
# ---------------------------------------------------------------


class AuthenticationErrorBase(RuntimeError):
    pass


class AuthenticationFailedError(AuthenticationErrorBase):
    pass


class AuthenticationMethodError(AuthenticationErrorBase):
    pass


class SessionNotFoundError(AuthenticationErrorBase):
    pass


# ---------------------------------------------------------------
# Provider contract
# ---------------------------------------------------------------
#
# A provider receives opaque credential material.
#
# It returns:
#
#   SecurityPrincipal -> successful authentication
#   None              -> authentication rejected
#
# The credential itself is never stored by this
# module.
#
# ---------------------------------------------------------------

AuthenticationProvider = Callable[
    [Any],
    Optional[SecurityPrincipal],
]


# ---------------------------------------------------------------
# Registered provider
# ---------------------------------------------------------------


@dataclass(frozen=True)
class RegisteredAuthenticationProvider:
    method: str

    provider: AuthenticationProvider

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)


# ---------------------------------------------------------------
# Authentication service
# ---------------------------------------------------------------


class AuthenticationService:
    """
    Coordinates explicitly registered
    authentication providers and bounded
    application sessions.

    It deliberately does NOT define:

      password hashing
      password verification algorithms
      JWT validation
      OAuth flows
      certificate validation
      MFA protocols

    Provider-specific authentication belongs in
    integrations/security adapters.
    """

    def __init__(
        self,
        *,
        config: Optional[AuthenticationConfig] = None,
        redact_sensitive: bool = True,
    ) -> None:
        self.config = config or AuthenticationConfig()

        self.config.validate()

        self.redact_sensitive = bool(redact_sensitive)

        self._providers: Dict[
            str,
            RegisteredAuthenticationProvider,
        ] = {}

        self._sessions: Dict[
            str,
            AuthenticationContext,
        ] = {}

        self._lock = RLock()

    # ---------------------------------------------------------
    # Provider registration
    # ---------------------------------------------------------

    def register_provider(
        self,
        method: str,
        provider: AuthenticationProvider,
        *,
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        normalized = required_string(
            method,
            name=("authentication method"),
            maximum_length=128,
        ).lower()

        if normalized not in set(self.config.accepted_methods):
            raise (
                AuthenticationMethodError(
                    "Authentication method "
                    "is not allowed by "
                    "configuration: "
                    f"{normalized}"
                )
            )

        with self._lock:
            if normalized in self._providers:
                raise (
                    AuthenticationMethodError(
                        f"Authentication provider already registered: {normalized}"
                    )
                )

            self._providers[normalized] = RegisteredAuthenticationProvider(
                method=normalized,
                provider=provider,
                attributes=(
                    to_json_safe(
                        dict(attributes or {}),
                        redact_sensitive=(self.redact_sensitive),
                        sensitive_keys=(DEFAULT_SENSITIVE_KEYS),
                    )
                ),
            )

    # ---------------------------------------------------------
    # Authenticate
    # ---------------------------------------------------------

    def authenticate(
        self,
        *,
        method: str,
        credential: Any,
    ) -> AuthenticationContext:
        """
        Authenticate opaque credential material
        through an explicitly registered
        provider.

        The credential is not persisted.
        """

        if not self.config.enabled:
            if self.config.require_authentication:
                raise (
                    AuthenticationFailedError(
                        "Authentication is "
                        "required but "
                        "authentication "
                        "services are "
                        "disabled."
                    )
                )

            return AuthenticationContext(
                status=(AuthenticationStatus.UNAUTHENTICATED.value),
                method=None,
                attributes={
                    "authentication_required": (False),
                },
            ).finalize()

        normalized = required_string(
            method,
            name=("authentication method"),
            maximum_length=128,
        ).lower()

        if normalized not in set(self.config.accepted_methods):
            raise AuthenticationMethodError(
                f"Authentication method is not allowed: {normalized}"
            )

        with self._lock:
            registered = self._providers.get(normalized)

        if registered is None:
            raise AuthenticationMethodError(
                f"No authentication provider is registered for method: {normalized}"
            )

        try:
            authenticated_principal = registered.provider(credential)

        except Exception as exc:
            # Do not propagate arbitrary provider
            # exception messages into normal
            # authentication output.
            raise (
                AuthenticationFailedError("Authentication provider failed.")
            ) from exc

        if authenticated_principal is None:
            raise AuthenticationFailedError("Authentication failed.")

        authenticated_principal.finalize()

        now = utc_datetime()

        expires = now + timedelta(seconds=(self.config.session_lifetime_seconds))

        context = AuthenticationContext(
            status=(AuthenticationStatus.AUTHENTICATED.value),
            principal=(authenticated_principal),
            method=normalized,
            authenticated_at=(now.isoformat()),
            expires_at=(expires.isoformat()),
            attributes={
                "provider_attributes": (registered.attributes),
            },
        ).finalize()

        with self._lock:
            self._sessions[context.session_id] = context

        return context

    # ---------------------------------------------------------
    # Session lookup
    # ---------------------------------------------------------

    def session(
        self,
        session_id: str,
    ) -> Optional[AuthenticationContext]:
        normalized = required_string(
            session_id,
            name="session_id",
            maximum_length=512,
        )

        with self._lock:
            context = self._sessions.get(normalized)

        if context is None:
            return None

        if context.expires_at and is_expired(context.expires_at):
            context.status = AuthenticationStatus.EXPIRED.value

        return context.finalize()

    # ---------------------------------------------------------
    # Require valid authenticated session
    # ---------------------------------------------------------

    def require_session(
        self,
        session_id: str,
    ) -> AuthenticationContext:
        context = self.session(session_id)

        if context is None:
            raise SessionNotFoundError(f"Unknown session: {session_id}")

        if context.status != AuthenticationStatus.AUTHENTICATED.value:
            raise (
                AuthenticationFailedError(
                    f"Session is not authenticated: {context.status}"
                )
            )

        return context

    # ---------------------------------------------------------
    # Revoke
    # ---------------------------------------------------------

    def revoke(
        self,
        session_id: str,
    ) -> AuthenticationContext:
        context = self.session(session_id)

        if context is None:
            raise SessionNotFoundError(f"Unknown session: {session_id}")

        context.status = AuthenticationStatus.REVOKED.value

        context.attributes["revoked_at"] = utc_now()

        with self._lock:
            self._sessions[context.session_id] = context

        return context.finalize()

    # ---------------------------------------------------------
    # Remove session metadata
    # ---------------------------------------------------------

    def remove(
        self,
        session_id: str,
    ) -> AuthenticationContext:
        normalized = required_string(
            session_id,
            name="session_id",
            maximum_length=512,
        )

        with self._lock:
            context = self._sessions.pop(
                normalized,
                None,
            )

        if context is None:
            raise SessionNotFoundError(f"Unknown session: {session_id}")

        return context

    # ---------------------------------------------------------
    # Purge expired sessions
    # ---------------------------------------------------------

    def purge_expired(
        self,
    ) -> int:
        with self._lock:
            expired = [
                session_id
                for (
                    session_id,
                    context,
                ) in self._sessions.items()
                if (context.expires_at is not None and is_expired(context.expires_at))
            ]

            for session_id in expired:
                self._sessions.pop(
                    session_id,
                    None,
                )

            return len(expired)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            statuses: Dict[
                str,
                int,
            ] = {}

            for context in self._sessions.values():
                status = context.status

                if context.expires_at and is_expired(context.expires_at):
                    status = AuthenticationStatus.EXPIRED.value

                statuses[status] = (
                    statuses.get(
                        status,
                        0,
                    )
                    + 1
                )

            return {
                "enabled": (self.config.enabled),
                "require_authentication": (self.config.require_authentication),
                "accepted_methods": (list(self.config.accepted_methods)),
                "registered_methods": (sorted(self._providers)),
                "session_count": (len(self._sessions)),
                "sessions_by_status": (statuses),
            }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "AuthenticationErrorBase",
    "AuthenticationFailedError",
    "AuthenticationMethodError",
    "SessionNotFoundError",
    "AuthenticationProvider",
    "RegisteredAuthenticationProvider",
    "AuthenticationService",
]
