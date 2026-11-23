from __future__ import annotations

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


# ---------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------

_TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "on",
}

_FALSE_VALUES = {
    "0",
    "false",
    "no",
    "off",
}


def _text(
    value: Any,
    *,
    default: str = "",
) -> str:
    if value is None:
        return default

    return str(
        value
    ).strip()


def _bool(
    value: Any,
    *,
    default: bool,
) -> bool:
    if value is None:
        return default

    if isinstance(
        value,
        bool,
    ):
        return value

    normalized = str(
        value
    ).strip().lower()

    if normalized in _TRUE_VALUES:
        return True

    if normalized in _FALSE_VALUES:
        return False

    raise ValueError(
        "Invalid boolean value: "
        f"{value}"
    )


def _int(
    value: Any,
    *,
    default: int,
    minimum: int = 0,
) -> int:
    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    result = int(
        value
    )

    if result < minimum:
        raise ValueError(
            "Integer value must be "
            f">= {minimum}"
        )

    return result


def _float(
    value: Any,
    *,
    default: float,
    minimum: float = 0.0,
) -> float:
    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    result = float(
        value
    )

    if result < minimum:
        raise ValueError(
            "Float value must be "
            f">= {minimum}"
        )

    return result


# ---------------------------------------------------------------
# Service configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class ServiceConfig:
    """
    Runtime configuration for one OSHA internal
    service.

    This does not describe external
    infrastructure services.
    """

    enabled: bool = True

    required: bool = True

    startup_timeout_seconds: float = 30.0

    shutdown_timeout_seconds: float = 30.0

    restart_limit: int = 0

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        if (
            self.startup_timeout_seconds
            <= 0
        ):
            raise ValueError(
                "startup_timeout_seconds "
                "must be > 0"
            )

        if (
            self.shutdown_timeout_seconds
            <= 0
        ):
            raise ValueError(
                "shutdown_timeout_seconds "
                "must be > 0"
            )

        if self.restart_limit < 0:
            raise ValueError(
                "restart_limit "
                "must be >= 0"
            )


# ---------------------------------------------------------------
# Worker configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class WorkerConfig:
    """
    Generic worker-runtime settings.

    workers.py can consume these later.
    """

    enabled: bool = True

    concurrency: int = 1

    poll_interval_seconds: float = 1.0

    graceful_shutdown_seconds: float = 30.0

    def validate(
        self,
    ) -> None:
        if self.concurrency < 1:
            raise ValueError(
                "worker concurrency "
                "must be >= 1"
            )

        if (
            self.poll_interval_seconds
            <= 0
        ):
            raise ValueError(
                "poll_interval_seconds "
                "must be > 0"
            )

        if (
            self.graceful_shutdown_seconds
            <= 0
        ):
            raise ValueError(
                "graceful_shutdown_seconds "
                "must be > 0"
            )


# ---------------------------------------------------------------
# Main runtime configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeConfig:
    """
    Top-level OSHA runtime configuration.

    dry_run defaults to True.

    Secrets should not be stored directly in this
    configuration object. Credential-specific
    integrations should use the security/secrets
    layer.
    """

    environment_id: str = "local-dev"

    instance_name: str = "osha"

    dry_run: bool = True

    startup_timeout_seconds: float = 60.0

    shutdown_timeout_seconds: float = 60.0

    health_interval_seconds: float = 15.0

    fail_fast: bool = True

    workers: WorkerConfig = field(
        default_factory=WorkerConfig
    )

    services: Dict[
        str,
        ServiceConfig,
    ] = field(
        default_factory=dict
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        if not (
            self.environment_id.strip()
        ):
            raise ValueError(
                "environment_id is required"
            )

        if not (
            self.instance_name.strip()
        ):
            raise ValueError(
                "instance_name is required"
            )

        if (
            self.startup_timeout_seconds
            <= 0
        ):
            raise ValueError(
                "startup_timeout_seconds "
                "must be > 0"
            )

        if (
            self.shutdown_timeout_seconds
            <= 0
        ):
            raise ValueError(
                "shutdown_timeout_seconds "
                "must be > 0"
            )

        if (
            self.health_interval_seconds
            <= 0
        ):
            raise ValueError(
                "health_interval_seconds "
                "must be > 0"
            )

        self.workers.validate()

        for name, service in (
            self.services.items()
        ):
            if not str(
                name
            ).strip():
                raise ValueError(
                    "service configuration "
                    "name is required"
                )

            service.validate()

    def service(
        self,
        name: str,
    ) -> ServiceConfig:
        """
        Return configured service settings.

        An unspecified service receives safe
        default settings.
        """

        return self.services.get(
            name,
            ServiceConfig(),
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "environment_id": (
                self.environment_id
            ),
            "instance_name": (
                self.instance_name
            ),
            "dry_run": (
                self.dry_run
            ),
            "startup_timeout_seconds": (
                self.
                startup_timeout_seconds
            ),
            "shutdown_timeout_seconds": (
                self.
                shutdown_timeout_seconds
            ),
            "health_interval_seconds": (
                self.
                health_interval_seconds
            ),
            "fail_fast": (
                self.fail_fast
            ),
            "workers": {
                "enabled": (
                    self.workers.enabled
                ),
                "concurrency": (
                    self.workers.
                    concurrency
                ),
                "poll_interval_seconds": (
                    self.workers.
                    poll_interval_seconds
                ),
                "graceful_shutdown_seconds": (
                    self.workers.
                    graceful_shutdown_seconds
                ),
            },
            "services": {
                name: {
                    "enabled": (
                        service.enabled
                    ),
                    "required": (
                        service.required
                    ),
                    "startup_timeout_seconds": (
                        service.
                        startup_timeout_seconds
                    ),
                    "shutdown_timeout_seconds": (
                        service.
                        shutdown_timeout_seconds
                    ),
                    "restart_limit": (
                        service.
                        restart_limit
                    ),
                    "attributes": dict(
                        service.attributes
                    ),
                }
                for name, service
                in sorted(
                    self.services.items()
                )
            },
            "attributes": dict(
                self.attributes
            ),
        }


# ---------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------

DEFAULT_ENV_PREFIX = "OSHA_"


def from_mapping(
    values: Mapping[
        str,
        Any,
    ],
    *,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> RuntimeConfig:
    """
    Build RuntimeConfig from an arbitrary
    key/value mapping.

    This makes configuration loading easy to
    test without changing os.environ.
    """

    def get(
        name: str,
        default: Any = None,
    ) -> Any:
        return values.get(
            prefix + name,
            default,
        )

    workers = WorkerConfig(
        enabled=_bool(
            get(
                "WORKERS_ENABLED"
            ),
            default=True,
        ),
        concurrency=_int(
            get(
                "WORKER_CONCURRENCY"
            ),
            default=1,
            minimum=1,
        ),
        poll_interval_seconds=_float(
            get(
                "WORKER_POLL_INTERVAL_SECONDS"
            ),
            default=1.0,
            minimum=0.001,
        ),
        graceful_shutdown_seconds=_float(
            get(
                "WORKER_GRACEFUL_SHUTDOWN_SECONDS"
            ),
            default=30.0,
            minimum=0.001,
        ),
    )

    config = RuntimeConfig(
        environment_id=_text(
            get(
                "ENVIRONMENT_ID"
            ),
            default="local-dev",
        ),
        instance_name=_text(
            get(
                "INSTANCE_NAME"
            ),
            default="osha",
        ),
        dry_run=_bool(
            get(
                "DRY_RUN"
            ),
            default=True,
        ),
        startup_timeout_seconds=_float(
            get(
                "STARTUP_TIMEOUT_SECONDS"
            ),
            default=60.0,
            minimum=0.001,
        ),
        shutdown_timeout_seconds=_float(
            get(
                "SHUTDOWN_TIMEOUT_SECONDS"
            ),
            default=60.0,
            minimum=0.001,
        ),
        health_interval_seconds=_float(
            get(
                "HEALTH_INTERVAL_SECONDS"
            ),
            default=15.0,
            minimum=0.001,
        ),
        fail_fast=_bool(
            get(
                "FAIL_FAST"
            ),
            default=True,
        ),
        workers=workers,
        attributes={
            "config_source": (
                "mapping"
            ),
            "environment_prefix": (
                prefix
            ),
        },
    )

    config.validate()

    return config


def from_environment(
    *,
    environ: Optional[
        Mapping[
            str,
            str,
        ]
    ] = None,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> RuntimeConfig:
    """
    Read supported runtime variables from the
    process environment.

    Unrecognized environment variables are not
    copied into RuntimeConfig.
    """

    source = (
        os.environ
        if environ is None
        else environ
    )

    config = from_mapping(
        source,
        prefix=prefix,
    )

    # RuntimeConfig is frozen. Produce a new
    # instance rather than mutating its regular
    # configuration fields.
    return RuntimeConfig(
        environment_id=(
            config.environment_id
        ),
        instance_name=(
            config.instance_name
        ),
        dry_run=config.dry_run,
        startup_timeout_seconds=(
            config.
            startup_timeout_seconds
        ),
        shutdown_timeout_seconds=(
            config.
            shutdown_timeout_seconds
        ),
        health_interval_seconds=(
            config.
            health_interval_seconds
        ),
        fail_fast=(
            config.fail_fast
        ),
        workers=config.workers,
        services=dict(
            config.services
        ),
        attributes={
            **config.attributes,
            "config_source": (
                "environment"
            ),
        },
    )


# ---------------------------------------------------------------
# Service configuration composition
# ---------------------------------------------------------------

def merge_service_configs(
    config: RuntimeConfig,
    services: Mapping[
        str,
        ServiceConfig,
    ],
) -> RuntimeConfig:
    """
    Return a new RuntimeConfig containing the
    supplied per-service settings.
    """

    merged = dict(
        config.services
    )

    merged.update(
        services
    )

    result = RuntimeConfig(
        environment_id=(
            config.environment_id
        ),
        instance_name=(
            config.instance_name
        ),
        dry_run=config.dry_run,
        startup_timeout_seconds=(
            config.
            startup_timeout_seconds
        ),
        shutdown_timeout_seconds=(
            config.
            shutdown_timeout_seconds
        ),
        health_interval_seconds=(
            config.
            health_interval_seconds
        ),
        fail_fast=(
            config.fail_fast
        ),
        workers=config.workers,
        services=merged,
        attributes=dict(
            config.attributes
        ),
    )

    result.validate()

    return result


# ---------------------------------------------------------------
# Safe summary
# ---------------------------------------------------------------

def safe_config_summary(
    config: RuntimeConfig,
) -> Dict[str, Any]:
    """
    Return non-secret configuration metadata for
    logging/readiness output.

    Do not add credentials to this summary.
    """

    config.validate()

    return {
        "environment_id": (
            config.environment_id
        ),
        "instance_name": (
            config.instance_name
        ),
        "dry_run": (
            config.dry_run
        ),
        "startup_timeout_seconds": (
            config.
            startup_timeout_seconds
        ),
        "shutdown_timeout_seconds": (
            config.
            shutdown_timeout_seconds
        ),
        "health_interval_seconds": (
            config.
            health_interval_seconds
        ),
        "fail_fast": (
            config.fail_fast
        ),
        "workers_enabled": (
            config.workers.enabled
        ),
        "worker_concurrency": (
            config.workers.
            concurrency
        ),
        "configured_services": (
            sorted(
                config.services
            )
        ),
    }
