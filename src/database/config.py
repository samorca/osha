"""
OSHA Database Configuration.

Centralises relational database configuration for OSHA.

Configuration is loaded from environment variables so the same application
code can run in:

    - local development
    - Docker / Docker Compose
    - Kubernetes
    - CI/CD
    - production environments

Secrets such as database passwords must not be hard-coded in source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy.engine import URL


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _get_env(name: str, default: str | None = None) -> str | None:
    """
    Read and normalise an environment variable.

    Empty values are treated as unset.
    """
    value = os.getenv(name)

    if value is None and name.startswith("OSHA_DB_"):
        suffix = name.removeprefix("OSHA_DB_")
        if suffix == "USER":
            suffix = "USERNAME"
        value = os.getenv(f"OSHA_DATABASE_{suffix}")

    if value is None:
        return default

    value = value.strip()

    if not value:
        return default

    return value


def _get_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Read and validate an integer environment variable."""
    raw = _get_env(name)

    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be an integer, got {raw!r}"
            ) from exc

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}, got {value}"
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"{name} must be <= {maximum}, got {value}"
        )

    return value


def _get_bool(name: str, default: bool) -> bool:
    """Read and validate a boolean environment variable."""
    raw = _get_env(name)

    if raw is None:
        return default

    value = raw.lower()

    if value in _TRUE_VALUES:
        return True

    if value in _FALSE_VALUES:
        return False

    raise ValueError(
        f"{name} must be one of "
        f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {raw!r}"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """
    Immutable OSHA relational database configuration.

    The configuration object deliberately contains configuration only.
    Engine and Session creation belong to connection.py and session.py.
    """

    driver: str

    host: str
    port: int

    database: str
    username: str
    password: str | None

    pool_size: int
    max_overflow: int
    pool_timeout: int
    pool_recycle: int

    pool_pre_ping: bool

    echo: bool
    echo_pool: bool

    connect_timeout: int

    application_name: str

    @property
    def url(self) -> URL:
        """
        Build the SQLAlchemy database URL safely.

        URL.create() is used rather than manually concatenating credentials
        into a connection string.
        """
        return URL.create(
            drivername=self.driver,
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )

    @property
    def safe_url(self) -> str:
        """
        Return a database URL with the password hidden.

        Suitable for diagnostics and logging.
        """
        return self.url.render_as_string(hide_password=True)

    def __repr__(self) -> str:
        """Return configuration information without exposing credentials."""
        return (
            "DatabaseConfig("
            f"url={self.safe_url!r}, "
            f"pool_size={self.pool_size}, "
            f"max_overflow={self.max_overflow}, "
            f"pool_timeout={self.pool_timeout}, "
            f"pool_recycle={self.pool_recycle}, "
            f"pool_pre_ping={self.pool_pre_ping}"
            ")"
        )


# ---------------------------------------------------------------------------
# Configuration factory
# ---------------------------------------------------------------------------

def load_database_config() -> DatabaseConfig:
    """
    Load OSHA database configuration from environment variables.

    Supported environment variables use either the OSHA_DB_* or
    OSHA_DATABASE_* prefix:

        OSHA_DB_DRIVER
        OSHA_DB_HOST
        OSHA_DB_PORT
        OSHA_DB_NAME
        OSHA_DB_USER
        OSHA_DB_PASSWORD

        OSHA_DB_POOL_SIZE
        OSHA_DB_MAX_OVERFLOW
        OSHA_DB_POOL_TIMEOUT
        OSHA_DB_POOL_RECYCLE
        OSHA_DB_POOL_PRE_PING

        OSHA_DB_ECHO
        OSHA_DB_ECHO_POOL

        OSHA_DB_CONNECT_TIMEOUT
        OSHA_DB_APPLICATION_NAME
    """

    return DatabaseConfig(
        driver=_get_env(
            "OSHA_DB_DRIVER",
            "postgresql+psycopg",
        )
        or "postgresql+psycopg",

        host=_get_env(
            "OSHA_DB_HOST",
            "localhost",
        )
        or "localhost",

        port=_get_int(
            "OSHA_DB_PORT",
            5432,
            minimum=1,
            maximum=65535,
        ),

        database=_get_env(
            "OSHA_DB_NAME",
            "osha",
        )
        or "osha",

        username=_get_env(
            "OSHA_DB_USER",
            "osha",
        )
        or "osha",

        password=_get_env(
            "OSHA_DB_PASSWORD",
        ),

        pool_size=_get_int(
            "OSHA_DB_POOL_SIZE",
            10,
            minimum=1,
        ),

        max_overflow=_get_int(
            "OSHA_DB_MAX_OVERFLOW",
            20,
            minimum=0,
        ),

        pool_timeout=_get_int(
            "OSHA_DB_POOL_TIMEOUT",
            30,
            minimum=1,
        ),

        pool_recycle=_get_int(
            "OSHA_DB_POOL_RECYCLE",
            1800,
            minimum=1,
        ),

        pool_pre_ping=_get_bool(
            "OSHA_DB_POOL_PRE_PING",
            True,
        ),

        echo=_get_bool(
            "OSHA_DB_ECHO",
            False,
        ),

        echo_pool=_get_bool(
            "OSHA_DB_ECHO_POOL",
            False,
        ),

        connect_timeout=_get_int(
            "OSHA_DB_CONNECT_TIMEOUT",
            10,
            minimum=1,
        ),

        application_name=_get_env(
            "OSHA_DB_APPLICATION_NAME",
            "osha",
        )
        or "osha",
    )


@lru_cache(maxsize=1)
def get_database_config() -> DatabaseConfig:
    """
    Return the process-wide database configuration.

    Environment variables are evaluated once per process.
    """
    return load_database_config()


def reset_database_config() -> None:
    """
    Clear the cached configuration.

    Primarily useful for tests that temporarily change environment variables.
    """
    get_database_config.cache_clear()


__all__ = [
    "DatabaseConfig",
    "get_database_config",
    "load_database_config",
    "reset_database_config",
]
