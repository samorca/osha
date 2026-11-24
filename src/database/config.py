from __future__ import annotations

"""
Non-secret database configuration for OSHA
persistence services.
"""

import os

from dataclasses import (
    dataclass,
    field,
)

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

from src.core.validation import (
    boolean,
    environment_id,
    integer,
    number,
    required_string,
)

from src.database.schema import (
    DatabaseBackend,
    DatabaseIdentity,
)

from src.security.schema import (
    SecretReference,
)


# ---------------------------------------------------------------
# Environment prefix
# ---------------------------------------------------------------

DEFAULT_ENV_PREFIX = "OSHA_DATABASE_"


# ---------------------------------------------------------------
# Pool configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class PoolConfig:
    """
    Generic pool policy.

    A backend may map these settings onto its own
    driver/pool implementation later.
    """

    minimum_size: int = 1

    maximum_size: int = 10

    acquire_timeout_seconds: float = 10.0

    idle_timeout_seconds: float = 300.0

    def validate(
        self,
    ) -> None:
        integer(
            self.minimum_size,
            name="minimum_size",
            minimum=0,
        )

        integer(
            self.maximum_size,
            name="maximum_size",
            minimum=1,
        )

        if self.minimum_size > self.maximum_size:
            raise ValueError("minimum_size cannot exceed maximum_size")

        number(
            self.acquire_timeout_seconds,
            name=("acquire_timeout_seconds"),
            minimum=0.001,
        )

        number(
            self.idle_timeout_seconds,
            name=("idle_timeout_seconds"),
            minimum=0.001,
        )


# ---------------------------------------------------------------
# Migration configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class MigrationConfig:
    enabled: bool = True

    # Keep automatic migration disabled until the
    # migration system is explicitly wired.
    apply_on_startup: bool = False

    validate_checksums: bool = True

    target_version: Optional[int] = None

    def validate(
        self,
    ) -> None:
        if self.target_version is not None:
            integer(
                self.target_version,
                name="target_version",
                minimum=1,
            )

        if self.apply_on_startup and not self.enabled:
            raise ValueError("apply_on_startup requires migrations to be enabled")


# ---------------------------------------------------------------
# Main database configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration without embedded
    credentials.

    credential contains a SecretReference only,
    never the actual database password/token.
    """

    backend: str = DatabaseBackend.MEMORY.value

    database: str = "osha"

    environment_id: str = "local-dev"

    host: Optional[str] = None

    port: Optional[int] = None

    namespace: Optional[str] = None

    username: Optional[str] = None

    credential: Optional[SecretReference] = None

    connect_timeout_seconds: float = 10.0

    statement_timeout_seconds: float = 30.0

    pool: PoolConfig = field(default_factory=PoolConfig)

    migrations: MigrationConfig = field(default_factory=MigrationConfig)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        if self.backend not in {item.value for item in DatabaseBackend}:
            raise ValueError(f"Unknown database backend: {self.backend}")

        required_string(
            self.database,
            name="database",
            maximum_length=256,
        )

        environment_id(self.environment_id)

        if self.host is not None:
            required_string(
                self.host,
                name="host",
                maximum_length=512,
            )

        if self.port is not None:
            integer(
                self.port,
                name="port",
                minimum=1,
                maximum=65535,
            )

        if self.username is not None:
            required_string(
                self.username,
                name="username",
                maximum_length=256,
            )

        if self.credential is not None:
            self.credential.finalize()

        number(
            self.connect_timeout_seconds,
            name=("connect_timeout_seconds"),
            minimum=0.001,
        )

        number(
            self.statement_timeout_seconds,
            name=("statement_timeout_seconds"),
            minimum=0.001,
        )

        self.pool.validate()

        self.migrations.validate()

    # ---------------------------------------------------------
    # Non-secret identity
    # ---------------------------------------------------------

    def identity(
        self,
    ) -> DatabaseIdentity:
        self.validate()

        return DatabaseIdentity(
            backend=self.backend,
            database=self.database,
            environment_id=(self.environment_id),
            host=self.host,
            port=self.port,
            namespace=self.namespace,
            attributes={
                "configured_username": (self.username is not None),
            },
        ).finalize()

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Safe configuration representation.

        credential serializes only its
        SecretReference metadata.
        """

        self.validate()

        return {
            "backend": self.backend,
            "database": self.database,
            "environment_id": (self.environment_id),
            "host": self.host,
            "port": self.port,
            "namespace": (self.namespace),
            "username": (self.username),
            "credential": (self.credential.to_dict() if self.credential else None),
            "connect_timeout_seconds": (self.connect_timeout_seconds),
            "statement_timeout_seconds": (self.statement_timeout_seconds),
            "pool": {
                "minimum_size": (self.pool.minimum_size),
                "maximum_size": (self.pool.maximum_size),
                "acquire_timeout_seconds": (self.pool.acquire_timeout_seconds),
                "idle_timeout_seconds": (self.pool.idle_timeout_seconds),
            },
            "migrations": {
                "enabled": (self.migrations.enabled),
                "apply_on_startup": (self.migrations.apply_on_startup),
                "validate_checksums": (self.migrations.validate_checksums),
                "target_version": (self.migrations.target_version),
            },
            "attributes": dict(self.attributes),
        }


# ---------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------


def _get(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Any = None,
) -> Any:
    return values.get(
        prefix + name,
        default,
    )


def _bool(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: bool,
) -> bool:
    value = _get(
        values,
        prefix,
        name,
    )

    if value is None or str(value).strip() == "":
        return default

    return boolean(
        value,
        name=prefix + name,
    )


def _int(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Optional[int],
    *,
    minimum: int = 1,
) -> Optional[int]:
    value = _get(
        values,
        prefix,
        name,
    )

    if value is None or str(value).strip() == "":
        return default

    return integer(
        value,
        name=prefix + name,
        minimum=minimum,
    )


def _float(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: float,
) -> float:
    value = _get(
        values,
        prefix,
        name,
    )

    if value is None or str(value).strip() == "":
        return default

    return number(
        value,
        name=prefix + name,
        minimum=0.001,
    )


def _optional_text(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
) -> Optional[str]:
    value = _get(
        values,
        prefix,
        name,
    )

    if value is None:
        return None

    text = str(value).strip()

    return text or None


# ---------------------------------------------------------------
# Mapping loader
# ---------------------------------------------------------------


def from_mapping(
    values: Mapping[
        str,
        Any,
    ],
    *,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> DatabaseConfig:
    """
    Build database configuration from a mapping.

    Note:
    OSHA_DATABASE_SECRET_* describes where the
    credential can be resolved. It does not
    contain the credential itself.
    """

    secret_name = _optional_text(
        values,
        prefix,
        "SECRET_NAME",
    )

    secret_provider = (
        _optional_text(
            values,
            prefix,
            "SECRET_PROVIDER",
        )
        or "environment"
    )

    secret_key = _optional_text(
        values,
        prefix,
        "SECRET_KEY",
    )

    secret_version = _optional_text(
        values,
        prefix,
        "SECRET_VERSION",
    )

    credential = None

    if secret_name is not None or secret_key is not None:
        credential = SecretReference(
            name=(secret_name or ("database-credential")),
            provider=(secret_provider),
            key=secret_key,
            version=(secret_version),
        ).finalize()

    config = DatabaseConfig(
        backend=str(
            _get(
                values,
                prefix,
                "BACKEND",
                DatabaseBackend.MEMORY.value,
            )
        )
        .strip()
        .lower(),
        database=str(
            _get(
                values,
                prefix,
                "NAME",
                "osha",
            )
        ).strip(),
        environment_id=str(
            _get(
                values,
                prefix,
                "ENVIRONMENT_ID",
                "local-dev",
            )
        ).strip(),
        host=_optional_text(
            values,
            prefix,
            "HOST",
        ),
        port=_int(
            values,
            prefix,
            "PORT",
            None,
        ),
        namespace=_optional_text(
            values,
            prefix,
            "NAMESPACE",
        ),
        username=_optional_text(
            values,
            prefix,
            "USERNAME",
        ),
        credential=credential,
        connect_timeout_seconds=(
            _float(
                values,
                prefix,
                "CONNECT_TIMEOUT_SECONDS",
                10.0,
            )
        ),
        statement_timeout_seconds=(
            _float(
                values,
                prefix,
                "STATEMENT_TIMEOUT_SECONDS",
                30.0,
            )
        ),
        pool=PoolConfig(
            minimum_size=(
                _int(
                    values,
                    prefix,
                    "POOL_MIN_SIZE",
                    1,
                    minimum=0,
                )
                or 0
            ),
            maximum_size=(
                _int(
                    values,
                    prefix,
                    "POOL_MAX_SIZE",
                    10,
                )
                or 10
            ),
            acquire_timeout_seconds=(
                _float(
                    values,
                    prefix,
                    "POOL_ACQUIRE_TIMEOUT_SECONDS",
                    10.0,
                )
            ),
            idle_timeout_seconds=(
                _float(
                    values,
                    prefix,
                    "POOL_IDLE_TIMEOUT_SECONDS",
                    300.0,
                )
            ),
        ),
        migrations=MigrationConfig(
            enabled=_bool(
                values,
                prefix,
                "MIGRATIONS_ENABLED",
                True,
            ),
            apply_on_startup=_bool(
                values,
                prefix,
                "MIGRATIONS_APPLY_ON_STARTUP",
                False,
            ),
            validate_checksums=_bool(
                values,
                prefix,
                "MIGRATIONS_VALIDATE_CHECKSUMS",
                True,
            ),
            target_version=_int(
                values,
                prefix,
                "MIGRATIONS_TARGET_VERSION",
                None,
            ),
        ),
        attributes={
            "config_source": ("mapping"),
            "environment_prefix": (prefix),
        },
    )

    config.validate()

    return config


# ---------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------


def from_environment(
    *,
    environ: Optional[
        Mapping[
            str,
            str,
        ]
    ] = None,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> DatabaseConfig:
    source = os.environ if environ is None else environ

    config = from_mapping(
        source,
        prefix=prefix,
    )

    return DatabaseConfig(
        backend=config.backend,
        database=config.database,
        environment_id=(config.environment_id),
        host=config.host,
        port=config.port,
        namespace=config.namespace,
        username=config.username,
        credential=config.credential,
        connect_timeout_seconds=(config.connect_timeout_seconds),
        statement_timeout_seconds=(config.statement_timeout_seconds),
        pool=config.pool,
        migrations=config.migrations,
        attributes={
            **config.attributes,
            "config_source": ("environment"),
        },
    )


# ---------------------------------------------------------------
# Safe summary
# ---------------------------------------------------------------


def safe_config_summary(
    config: DatabaseConfig,
) -> Dict[str, Any]:
    """
    Return database settings suitable for status
    and diagnostics without exposing credentials.
    """

    config.validate()

    return {
        "backend": config.backend,
        "database": config.database,
        "environment_id": (config.environment_id),
        "host": config.host,
        "port": config.port,
        "namespace": (config.namespace),
        "username_configured": (config.username is not None),
        "credential_reference_configured": (config.credential is not None),
        "credential_provider": (
            config.credential.provider if config.credential else None
        ),
        "connect_timeout_seconds": (config.connect_timeout_seconds),
        "statement_timeout_seconds": (config.statement_timeout_seconds),
        "pool_minimum_size": (config.pool.minimum_size),
        "pool_maximum_size": (config.pool.maximum_size),
        "migrations_enabled": (config.migrations.enabled),
        "migrations_apply_on_startup": (config.migrations.apply_on_startup),
    }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_ENV_PREFIX",
    "PoolConfig",
    "MigrationConfig",
    "DatabaseConfig",
    "from_mapping",
    "from_environment",
    "safe_config_summary",
]
