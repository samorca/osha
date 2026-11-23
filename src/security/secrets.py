from __future__ import annotations

"""
Provider-neutral secret resolution for OSHA
without secret-value serialization.
"""

import os

from dataclasses import (
    dataclass,
)

from threading import RLock

from typing import (
    Callable,
    Dict,
    Mapping,
    Optional,
)

from src.core.time import (
    is_expired,
    timestamp_after,
)

from src.core.validation import (
    required_string,
)

from src.security.config import (
    SecretsConfig,
)

from src.security.schema import (
    SecretReference,
)


# ---------------------------------------------------------------
# Errors
# ---------------------------------------------------------------

class SecretError(
    RuntimeError
):
    pass


class SecretProviderError(
    SecretError
):
    pass


class SecretNotFoundError(
    SecretError
):
    pass


class SecretProviderNotFoundError(
    SecretError
):
    pass


# ---------------------------------------------------------------
# Provider contract
# ---------------------------------------------------------------

SecretProvider = Callable[
    [
        SecretReference
    ],
    Optional[str],
]


# ---------------------------------------------------------------
# Opaque secret value
# ---------------------------------------------------------------

class SecretValue:
    """
    Opaque in-process secret wrapper.

    str() and repr() never return the secret.

    reveal() is deliberately explicit and should
    only be used at the narrow integration
    boundary that actually requires the value.
    """

    __slots__ = (
        "__value",
    )

    def __init__(
        self,
        value: str,
    ) -> None:
        if value is None:
            raise SecretNotFoundError(
                "Secret value is "
                "unavailable."
            )

        self.__value = str(
            value
        )

    def reveal(
        self,
    ) -> str:
        return self.__value

    def __repr__(
        self,
    ) -> str:
        return (
            "SecretValue([REDACTED])"
        )

    def __str__(
        self,
    ) -> str:
        return "[REDACTED]"

    def __bool__(
        self,
    ) -> bool:
        return bool(
            self.__value
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.__value
        )


# ---------------------------------------------------------------
# Cache record
# ---------------------------------------------------------------

@dataclass
class CachedSecret:
    """
    Internal cache record.

    Cached SecretValue objects should never be
    serialized.
    """

    value: SecretValue

    expires_at: str


# ---------------------------------------------------------------
# Environment provider
# ---------------------------------------------------------------

class EnvironmentSecretProvider:
    """
    Resolve SecretReference objects from a
    process-environment mapping.

    This provider does not enumerate or expose
    environment variables.

    reference.key is preferred. If absent,
    reference.name is used.
    """

    def __init__(
        self,
        environ: Optional[
            Mapping[
                str,
                str,
            ]
        ] = None,
    ) -> None:
        self.environ = (
            os.environ
            if environ is None
            else environ
        )

    def __call__(
        self,
        reference: SecretReference,
    ) -> Optional[str]:
        reference.finalize()

        key = (
            reference.key
            or reference.name
        )

        key = required_string(
            key,
            name=(
                "environment secret key"
            ),
            maximum_length=512,
        )

        return self.environ.get(
            key
        )


# ---------------------------------------------------------------
# Secret manager
# ---------------------------------------------------------------

class SecretManager:
    """
    Resolve secret references through explicitly
    registered providers.

    The manager never exposes secret values from
    summary/status methods.
    """

    def __init__(
        self,
        *,
        config: Optional[
            SecretsConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or SecretsConfig()
        )

        self.config.validate()

        self._providers: Dict[
            str,
            SecretProvider,
        ] = {}

        self._cache: Dict[
            str,
            CachedSecret,
        ] = {}

        self._lock = RLock()

    # ---------------------------------------------------------
    # Provider registration
    # ---------------------------------------------------------

    def register_provider(
        self,
        name: str,
        provider: SecretProvider,
    ) -> None:
        normalized = required_string(
            name,
            name="secret provider",
            maximum_length=128,
        ).lower()

        if (
            normalized
            not in set(
                self.config.
                allowed_providers
            )
        ):
            raise SecretProviderError(
                "Secret provider is not "
                "allowed by configuration: "
                f"{normalized}"
            )

        with self._lock:
            if (
                normalized
                in self._providers
            ):
                raise SecretProviderError(
                    "Secret provider already "
                    "registered: "
                    f"{normalized}"
                )

            self._providers[
                normalized
            ] = provider

    # ---------------------------------------------------------
    # Convenience environment provider
    # ---------------------------------------------------------

    def register_default_environment_provider(
        self,
        *,
        environ: Optional[
            Mapping[
                str,
                str,
            ]
        ] = None,
    ) -> None:
        self.register_provider(
            "environment",
            EnvironmentSecretProvider(
                environ
            ),
        )

    # ---------------------------------------------------------
    # Internal cache key
    # ---------------------------------------------------------

    def _cache_key(
        self,
        reference: SecretReference,
    ) -> str:
        reference.finalize()

        return "|".join(
            (
                reference.provider,

                reference.name,

                reference.key
                or "",

                reference.version
                or "",
            )
        )

    # ---------------------------------------------------------
    # Resolve
    # ---------------------------------------------------------

    def resolve(
        self,
        reference: SecretReference,
    ) -> SecretValue:
        """
        Resolve a SecretReference to an opaque
        SecretValue.

        Provider exceptions are wrapped without
        copying arbitrary provider exception
        messages.
        """

        if not self.config.enabled:
            raise SecretProviderError(
                "Secret resolution is "
                "disabled."
            )

        reference.finalize()

        provider_name = (
            required_string(
                (
                    reference.provider
                    or self.config.
                    default_provider
                ),
                name="secret provider",
                maximum_length=128,
            ).lower()
        )

        if (
            provider_name
            not in set(
                self.config.
                allowed_providers
            )
        ):
            raise SecretProviderError(
                "Secret provider is not "
                "allowed by configuration: "
                f"{provider_name}"
            )

        cache_key = (
            self._cache_key(
                reference
            )
        )

        # -----------------------------------------------------
        # Cache lookup
        # -----------------------------------------------------

        if (
            self.config.cache_enabled
        ):
            with self._lock:
                cached = (
                    self._cache.get(
                        cache_key
                    )
                )

            if (
                cached is not None
                and not is_expired(
                    cached.expires_at
                )
            ):
                return cached.value

        # -----------------------------------------------------
        # Provider lookup
        # -----------------------------------------------------

        with self._lock:
            provider = (
                self._providers.get(
                    provider_name
                )
            )

        if provider is None:
            raise (
                SecretProviderNotFoundError(
                    "No secret provider is "
                    "registered for: "
                    f"{provider_name}"
                )
            )

        # -----------------------------------------------------
        # Resolve through provider
        # -----------------------------------------------------

        try:
            raw_value = provider(
                reference
            )

        except Exception as exc:
            raise SecretProviderError(
                "Secret provider failed."
            ) from exc

        if raw_value is None:
            raise SecretNotFoundError(
                "Secret was not found."
            )

        secret = SecretValue(
            raw_value
        )

        # -----------------------------------------------------
        # Optional short-lived cache
        # -----------------------------------------------------

        if (
            self.config.cache_enabled
        ):
            with self._lock:
                self._cache[
                    cache_key
                ] = CachedSecret(
                    value=secret,
                    expires_at=(
                        timestamp_after(
                            self.config.
                            cache_ttl_seconds
                        )
                    ),
                )

        return secret

    # ---------------------------------------------------------
    # Invalidate one cached secret
    # ---------------------------------------------------------

    def invalidate(
        self,
        reference: SecretReference,
    ) -> bool:
        key = self._cache_key(
            reference
        )

        with self._lock:
            return (
                self._cache.pop(
                    key,
                    None,
                )
                is not None
            )

    # ---------------------------------------------------------
    # Clear cache
    # ---------------------------------------------------------

    def clear_cache(
        self,
    ) -> int:
        with self._lock:
            count = len(
                self._cache
            )

            self._cache.clear()

            return count

    # ---------------------------------------------------------
    # Purge expired cache entries
    # ---------------------------------------------------------

    def purge_expired(
        self,
    ) -> int:
        with self._lock:
            expired = [
                key
                for key, cached
                in self._cache.items()
                if is_expired(
                    cached.expires_at
                )
            ]

            for key in expired:
                self._cache.pop(
                    key,
                    None,
                )

            return len(
                expired
            )

    # ---------------------------------------------------------
    # Safe summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[
        str,
        object,
    ]:
        """
        This summary contains no secret names,
        keys or values.
        """

        with self._lock:
            return {
                "enabled": (
                    self.config.enabled
                ),

                "default_provider": (
                    self.config.
                    default_provider
                ),

                "allowed_providers": (
                    list(
                        self.config.
                        allowed_providers
                    )
                ),

                "registered_providers": (
                    sorted(
                        self._providers
                    )
                ),

                "cache_enabled": (
                    self.config.
                    cache_enabled
                ),

                "cached_secret_count": (
                    len(
                        self._cache
                    )
                ),
            }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "SecretError",
    "SecretProviderError",
    "SecretNotFoundError",
    "SecretProviderNotFoundError",

    "SecretProvider",

    "SecretValue",
    "CachedSecret",

    "EnvironmentSecretProvider",

    "SecretManager",
]
