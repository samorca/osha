from __future__ import annotations

"""
Non-secret Linux integration configuration for
OSHA.
"""

import os

from dataclasses import (
    dataclass,
    field,
)

from enum import Enum

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

from src.core.validation import (
    boolean,
    number,
    required_string,
)

from src.integrations.schema import (
    IntegrationTarget,
    ProviderKind,
)

from src.security.schema import (
    SecretReference,
)


# ---------------------------------------------------------------
# Environment prefix
# ---------------------------------------------------------------

DEFAULT_ENV_PREFIX = "OSHA_LINUX_"


# ---------------------------------------------------------------
# Connection mode
# ---------------------------------------------------------------


class LinuxMode(
    str,
    Enum,
):
    LOCAL = "local"

    REMOTE = "remote"


# ---------------------------------------------------------------
# Linux connection configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class LinuxConnectionConfig:
    """
    Linux host connection metadata.

    LOCAL means OSHA reads the host where the
    integration process is running.

    REMOTE is reserved for a controlled remote
    transport implementation in client.py.

    Credential values never belong here.
    """

    mode: str = LinuxMode.LOCAL.value

    host: Optional[str] = None

    port: int = 22

    username: Optional[str] = None

    credential: Optional[SecretReference] = None

    verify_host_key: bool = True

    def validate(
        self,
    ) -> None:
        if self.mode not in {item.value for item in LinuxMode}:
            raise ValueError(f"Unknown Linux integration mode: {self.mode}")

        if self.mode == LinuxMode.REMOTE.value:
            if not self.host:
                raise ValueError("Remote Linux integration requires host")

            required_string(
                self.host,
                name="host",
                maximum_length=512,
            )

            if not (1 <= int(self.port) <= 65535):
                raise ValueError("Linux remote port must be between 1 and 65535")

            if self.username is not None:
                required_string(
                    self.username,
                    name="username",
                    maximum_length=256,
                )

        if self.mode == LinuxMode.LOCAL.value and self.credential is not None:
            raise ValueError(
                "Local Linux integration does not require a credential reference"
            )

        if self.credential is not None:
            self.credential.finalize()

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "mode": self.mode,
            "host": self.host,
            "port": self.port,
            "username": (self.username),
            "verify_host_key": (self.verify_host_key),
            "credential": (self.credential.to_dict() if self.credential else None),
            "credential_value_exposed": (False),
        }


# ---------------------------------------------------------------
# Linux integration configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class LinuxConfig:
    """
    Configuration for one Linux integration
    target.

    The first implementation will primarily
    support read-only local host inspection.
    """

    environment_id: str

    target_id: str

    connection: LinuxConnectionConfig = field(default_factory=(LinuxConnectionConfig))

    request_timeout_seconds: float = 15.0

    include_processes: bool = True

    include_cpu: bool = True

    include_memory: bool = True

    include_disks: bool = True

    include_network: bool = True

    include_boot_info: bool = True

    # Avoid exposing environmental variables from
    # the Linux host by default.
    include_environment: bool = False

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        required_string(
            self.environment_id,
            name="environment_id",
            maximum_length=256,
        )

        required_string(
            self.target_id,
            name="target_id",
            maximum_length=512,
        )

        number(
            self.request_timeout_seconds,
            name=("request_timeout_seconds"),
            minimum=0.001,
        )

        self.connection.validate()

    # ---------------------------------------------------------
    # Generic target
    # ---------------------------------------------------------

    def target(
        self,
    ) -> IntegrationTarget:
        self.validate()

        endpoint = None

        if self.connection.mode == LinuxMode.REMOTE.value and self.connection.host:
            endpoint = f"ssh://{self.connection.host}:{self.connection.port}"

        return IntegrationTarget(
            provider=(ProviderKind.LINUX.value),
            environment_id=(self.environment_id),
            target_id=(self.target_id),
            endpoint=endpoint,
            credential=(self.connection.credential),
            attributes={
                "mode": (self.connection.mode),
                "remote": (self.connection.mode == LinuxMode.REMOTE.value),
                "verify_host_key": (self.connection.verify_host_key),
            },
        ).finalize()

    # ---------------------------------------------------------
    # Safe serialization
    # ---------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "environment_id": (self.environment_id),
            "target_id": (self.target_id),
            "connection": (self.connection.to_dict()),
            "request_timeout_seconds": (self.request_timeout_seconds),
            "collection": {
                "processes": (self.include_processes),
                "cpu": (self.include_cpu),
                "memory": (self.include_memory),
                "disks": (self.include_disks),
                "network": (self.include_network),
                "boot_info": (self.include_boot_info),
                "environment": (self.include_environment),
            },
            "attributes": dict(self.attributes),
        }


# ---------------------------------------------------------------
# Mapping helpers
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


def _text(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Optional[str] = None,
) -> Optional[str]:
    value = _get(
        values,
        prefix,
        name,
        default,
    )

    if value is None:
        return None

    result = str(value).strip()

    return result or None


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
) -> LinuxConfig:
    mode = (
        _text(
            values,
            prefix,
            "MODE",
            LinuxMode.LOCAL.value,
        )
        or LinuxMode.LOCAL.value
    ).lower()

    secret_key = _text(
        values,
        prefix,
        "SECRET_KEY",
    )

    secret_name = (
        _text(
            values,
            prefix,
            "SECRET_NAME",
        )
        or "linux-credential"
    )

    secret_provider = (
        _text(
            values,
            prefix,
            "SECRET_PROVIDER",
        )
        or "environment"
    )

    credential = (
        SecretReference(
            name=secret_name,
            provider=(secret_provider),
            key=secret_key,
            version=_text(
                values,
                prefix,
                "SECRET_VERSION",
            ),
        ).finalize()
        if secret_key
        else None
    )

    port_text = _text(
        values,
        prefix,
        "PORT",
        "22",
    )

    port = int(port_text or "22")

    config = LinuxConfig(
        environment_id=(
            _text(
                values,
                prefix,
                "ENVIRONMENT_ID",
                "local-dev",
            )
            or "local-dev"
        ),
        target_id=(
            _text(
                values,
                prefix,
                "TARGET_ID",
                "linux-local",
            )
            or "linux-local"
        ),
        connection=(
            LinuxConnectionConfig(
                mode=mode,
                host=_text(
                    values,
                    prefix,
                    "HOST",
                ),
                port=port,
                username=_text(
                    values,
                    prefix,
                    "USERNAME",
                ),
                credential=credential,
                verify_host_key=_bool(
                    values,
                    prefix,
                    "VERIFY_HOST_KEY",
                    True,
                ),
            )
        ),
        request_timeout_seconds=(
            _float(
                values,
                prefix,
                "REQUEST_TIMEOUT_SECONDS",
                15.0,
            )
        ),
        include_processes=_bool(
            values,
            prefix,
            "INCLUDE_PROCESSES",
            True,
        ),
        include_cpu=_bool(
            values,
            prefix,
            "INCLUDE_CPU",
            True,
        ),
        include_memory=_bool(
            values,
            prefix,
            "INCLUDE_MEMORY",
            True,
        ),
        include_disks=_bool(
            values,
            prefix,
            "INCLUDE_DISKS",
            True,
        ),
        include_network=_bool(
            values,
            prefix,
            "INCLUDE_NETWORK",
            True,
        ),
        include_boot_info=_bool(
            values,
            prefix,
            "INCLUDE_BOOT_INFO",
            True,
        ),
        include_environment=_bool(
            values,
            prefix,
            "INCLUDE_ENVIRONMENT",
            False,
        ),
        attributes={
            "config_source": ("mapping"),
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
) -> LinuxConfig:
    source = os.environ if environ is None else environ

    return from_mapping(
        source,
        prefix=prefix,
    )


# ---------------------------------------------------------------
# Safe summary
# ---------------------------------------------------------------


def safe_config_summary(
    config: LinuxConfig,
) -> Dict[str, Any]:
    config.validate()

    return {
        "environment_id": (config.environment_id),
        "target_id": (config.target_id),
        "mode": (config.connection.mode),
        "remote_host_configured": (config.connection.host is not None),
        "remote_port": (config.connection.port),
        "username_configured": (config.connection.username is not None),
        "credential_reference_configured": (config.connection.credential is not None),
        "verify_host_key": (config.connection.verify_host_key),
        "collection": {
            "processes": (config.include_processes),
            "cpu": (config.include_cpu),
            "memory": (config.include_memory),
            "disks": (config.include_disks),
            "network": (config.include_network),
            "boot_info": (config.include_boot_info),
            "environment": (config.include_environment),
        },
        "secret_value_exposed": (False),
        "arbitrary_command_execution_enabled": (False),
    }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_ENV_PREFIX",
    "LinuxMode",
    "LinuxConnectionConfig",
    "LinuxConfig",
    "from_mapping",
    "from_environment",
    "safe_config_summary",
]
