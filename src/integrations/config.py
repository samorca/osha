from __future__ import annotations

"""
Common, non-secret configuration for OSHA
infrastructure integrations.
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
    Tuple,
)

from src.core.validation import (
    boolean,
    integer,
    number,
    required_string,
    unique_strings,
)

from src.integrations.schema import (
    ProviderKind,
)


# ---------------------------------------------------------------
# Environment prefix
# ---------------------------------------------------------------

DEFAULT_ENV_PREFIX = (
    "OSHA_INTEGRATIONS_"
)


# ---------------------------------------------------------------
# Timeout configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class TimeoutConfig:
    connect_seconds: float = 10.0

    operation_seconds: float = 30.0

    health_seconds: float = 5.0

    def validate(
        self,
    ) -> None:
        number(
            self.connect_seconds,
            name="connect_seconds",
            minimum=0.001,
        )

        number(
            self.operation_seconds,
            name="operation_seconds",
            minimum=0.001,
        )

        number(
            self.health_seconds,
            name="health_seconds",
            minimum=0.001,
        )


# ---------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class RetryConfig:
    enabled: bool = True

    maximum_attempts: int = 3

    initial_delay_seconds: float = 0.5

    maximum_delay_seconds: float = 5.0

    multiplier: float = 2.0

    def validate(
        self,
    ) -> None:
        integer(
            self.maximum_attempts,
            name="maximum_attempts",
            minimum=1,
        )

        number(
            self.initial_delay_seconds,
            name=(
                "initial_delay_seconds"
            ),
            minimum=0.0,
        )

        number(
            self.maximum_delay_seconds,
            name=(
                "maximum_delay_seconds"
            ),
            minimum=0.0,
        )

        number(
            self.multiplier,
            name="multiplier",
            minimum=1.0,
        )

        if (
            self.maximum_delay_seconds
            < self.initial_delay_seconds
        ):
            raise ValueError(
                "maximum_delay_seconds "
                "cannot be less than "
                "initial_delay_seconds"
            )


# ---------------------------------------------------------------
# Concurrency configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class ConcurrencyConfig:
    """
    Process-local integration operation limits.

    Real distributed concurrency enforcement may
    later require workers/queues/database-backed
    coordination.
    """

    maximum_operations: int = 16

    maximum_mutating_operations: int = 2

    def validate(
        self,
    ) -> None:
        integer(
            self.maximum_operations,
            name="maximum_operations",
            minimum=1,
        )

        integer(
            self.maximum_mutating_operations,
            name=(
                "maximum_mutating_operations"
            ),
            minimum=1,
        )

        if (
            self.maximum_mutating_operations
            > self.maximum_operations
        ):
            raise ValueError(
                "maximum_mutating_operations "
                "cannot exceed "
                "maximum_operations"
            )


# ---------------------------------------------------------------
# Health configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class HealthConfig:
    enabled: bool = True

    interval_seconds: float = 30.0

    degraded_latency_seconds: float = 1.0

    def validate(
        self,
    ) -> None:
        number(
            self.interval_seconds,
            name="interval_seconds",
            minimum=0.001,
        )

        number(
            self.degraded_latency_seconds,
            name=(
                "degraded_latency_seconds"
            ),
            minimum=0.001,
        )


# ---------------------------------------------------------------
# Operation policy
# ---------------------------------------------------------------

@dataclass(frozen=True)
class OperationPolicyConfig:
    """
    Integration-level execution policy.

    Read operations are enabled by default.

    Mutating operations are disabled by default
    and should only be enabled alongside the
    security/remediation controls we've already
    built.
    """

    allow_read_operations: bool = True

    allow_mutating_operations: bool = False

    require_security_context_for_mutations: bool = True

    require_explicit_capability: bool = True

    def validate(
        self,
    ) -> None:
        if (
            self.allow_mutating_operations
            and not self.
            require_explicit_capability
        ):
            raise ValueError(
                "Mutating integrations "
                "require explicit "
                "capability validation."
            )


# ---------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class IntegrationsConfig:
    enabled: bool = True

    enabled_providers: Tuple[
        str,
        ...,
    ] = (
        ProviderKind.OPENSTACK.value,
        ProviderKind.KUBERNETES.value,
        ProviderKind.LINUX.value,
        ProviderKind.CEPH.value,
        ProviderKind.MINIO.value,
    )

    timeouts: TimeoutConfig = field(
        default_factory=TimeoutConfig
    )

    retries: RetryConfig = field(
        default_factory=RetryConfig
    )

    concurrency: ConcurrencyConfig = field(
        default_factory=ConcurrencyConfig
    )

    health: HealthConfig = field(
        default_factory=HealthConfig
    )

    operations: OperationPolicyConfig = field(
        default_factory=(
            OperationPolicyConfig
        )
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
        providers = unique_strings(
            self.enabled_providers,
            name="enabled_providers",
            allow_empty=False,
        )

        known = {
            item.value
            for item
            in ProviderKind
        }

        unknown = [
            provider
            for provider
            in providers
            if provider
            not in known
        ]

        if unknown:
            raise ValueError(
                "Unknown integration "
                "providers: "
                + ", ".join(
                    unknown
                )
            )

        self.timeouts.validate()
        self.retries.validate()
        self.concurrency.validate()
        self.health.validate()
        self.operations.validate()

    # ---------------------------------------------------------
    # Provider enabled
    # ---------------------------------------------------------

    def provider_enabled(
        self,
        provider: str,
    ) -> bool:
        self.validate()

        normalized = (
            required_string(
                provider,
                name="provider",
                maximum_length=128,
            ).lower()
        )

        return bool(
            self.enabled
            and normalized
            in set(
                self.enabled_providers
            )
        )

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "enabled": (
                self.enabled
            ),

            "enabled_providers": (
                list(
                    self.enabled_providers
                )
            ),

            "timeouts": {
                "connect_seconds": (
                    self.timeouts.
                    connect_seconds
                ),

                "operation_seconds": (
                    self.timeouts.
                    operation_seconds
                ),

                "health_seconds": (
                    self.timeouts.
                    health_seconds
                ),
            },

            "retries": {
                "enabled": (
                    self.retries.enabled
                ),

                "maximum_attempts": (
                    self.retries.
                    maximum_attempts
                ),

                "initial_delay_seconds": (
                    self.retries.
                    initial_delay_seconds
                ),

                "maximum_delay_seconds": (
                    self.retries.
                    maximum_delay_seconds
                ),

                "multiplier": (
                    self.retries.multiplier
                ),
            },

            "concurrency": {
                "maximum_operations": (
                    self.concurrency.
                    maximum_operations
                ),

                "maximum_mutating_operations": (
                    self.concurrency.
                    maximum_mutating_operations
                ),
            },

            "health": {
                "enabled": (
                    self.health.enabled
                ),

                "interval_seconds": (
                    self.health.
                    interval_seconds
                ),

                "degraded_latency_seconds": (
                    self.health.
                    degraded_latency_seconds
                ),
            },

            "operations": {
                "allow_read_operations": (
                    self.operations.
                    allow_read_operations
                ),

                "allow_mutating_operations": (
                    self.operations.
                    allow_mutating_operations
                ),

                "require_security_context_for_mutations": (
                    self.operations.
                    require_security_context_for_mutations
                ),

                "require_explicit_capability": (
                    self.operations.
                    require_explicit_capability
                ),
            },

            "attributes": dict(
                self.attributes
            ),
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

    if (
        value is None
        or str(value).strip() == ""
    ):
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
    default: int,
) -> int:
    value = _get(
        values,
        prefix,
        name,
    )

    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    return integer(
        value,
        name=prefix + name,
        minimum=1,
    )


def _float(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: float,
    *,
    minimum: float = 0.001,
) -> float:
    value = _get(
        values,
        prefix,
        name,
    )

    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    return number(
        value,
        name=prefix + name,
        minimum=minimum,
    )


def _csv(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Tuple[
        str,
        ...,
    ],
) -> Tuple[
    str,
    ...,
]:
    value = _get(
        values,
        prefix,
        name,
    )

    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    return tuple(
        item.lower()
        for item
        in unique_strings(
            str(value).split(
                ","
            ),
            name=prefix + name,
            allow_empty=False,
        )
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
) -> IntegrationsConfig:
    defaults = (
        IntegrationsConfig()
    )

    config = IntegrationsConfig(
        enabled=_bool(
            values,
            prefix,
            "ENABLED",
            True,
        ),

        enabled_providers=_csv(
            values,
            prefix,
            "ENABLED_PROVIDERS",
            defaults.
            enabled_providers,
        ),

        timeouts=TimeoutConfig(
            connect_seconds=_float(
                values,
                prefix,
                "CONNECT_TIMEOUT_SECONDS",
                10.0,
            ),

            operation_seconds=_float(
                values,
                prefix,
                "OPERATION_TIMEOUT_SECONDS",
                30.0,
            ),

            health_seconds=_float(
                values,
                prefix,
                "HEALTH_TIMEOUT_SECONDS",
                5.0,
            ),
        ),

        retries=RetryConfig(
            enabled=_bool(
                values,
                prefix,
                "RETRY_ENABLED",
                True,
            ),

            maximum_attempts=_int(
                values,
                prefix,
                "RETRY_MAX_ATTEMPTS",
                3,
            ),

            initial_delay_seconds=(
                _float(
                    values,
                    prefix,
                    "RETRY_INITIAL_DELAY_SECONDS",
                    0.5,
                    minimum=0.0,
                )
            ),

            maximum_delay_seconds=(
                _float(
                    values,
                    prefix,
                    "RETRY_MAX_DELAY_SECONDS",
                    5.0,
                    minimum=0.0,
                )
            ),

            multiplier=_float(
                values,
                prefix,
                "RETRY_MULTIPLIER",
                2.0,
                minimum=1.0,
            ),
        ),

        concurrency=ConcurrencyConfig(
            maximum_operations=_int(
                values,
                prefix,
                "MAX_OPERATIONS",
                16,
            ),

            maximum_mutating_operations=(
                _int(
                    values,
                    prefix,
                    "MAX_MUTATING_OPERATIONS",
                    2,
                )
            ),
        ),

        health=HealthConfig(
            enabled=_bool(
                values,
                prefix,
                "HEALTH_ENABLED",
                True,
            ),

            interval_seconds=_float(
                values,
                prefix,
                "HEALTH_INTERVAL_SECONDS",
                30.0,
            ),

            degraded_latency_seconds=(
                _float(
                    values,
                    prefix,
                    "HEALTH_DEGRADED_LATENCY_SECONDS",
                    1.0,
                )
            ),
        ),

        operations=OperationPolicyConfig(
            allow_read_operations=(
                _bool(
                    values,
                    prefix,
                    "ALLOW_READ_OPERATIONS",
                    True,
                )
            ),

            allow_mutating_operations=(
                _bool(
                    values,
                    prefix,
                    "ALLOW_MUTATING_OPERATIONS",
                    False,
                )
            ),

            require_security_context_for_mutations=(
                _bool(
                    values,
                    prefix,
                    "REQUIRE_SECURITY_CONTEXT_FOR_MUTATIONS",
                    True,
                )
            ),

            require_explicit_capability=(
                _bool(
                    values,
                    prefix,
                    "REQUIRE_EXPLICIT_CAPABILITY",
                    True,
                )
            ),
        ),

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
) -> IntegrationsConfig:
    source = (
        os.environ
        if environ is None
        else environ
    )

    config = from_mapping(
        source,
        prefix=prefix,
    )

    return IntegrationsConfig(
        enabled=config.enabled,

        enabled_providers=(
            config.enabled_providers
        ),

        timeouts=config.timeouts,

        retries=config.retries,

        concurrency=(
            config.concurrency
        ),

        health=config.health,

        operations=config.operations,

        attributes={
            **config.attributes,
            "config_source": (
                "environment"
            ),
        },
    )


# ---------------------------------------------------------------
# Safe summary
# ---------------------------------------------------------------

def safe_config_summary(
    config: IntegrationsConfig,
) -> Dict[str, Any]:
    """
    Return non-secret integration policy
    configuration for diagnostics.
    """

    config.validate()

    return {
        "enabled": (
            config.enabled
        ),

        "enabled_providers": (
            list(
                config.
                enabled_providers
            )
        ),

        "connect_timeout_seconds": (
            config.timeouts.
            connect_seconds
        ),

        "operation_timeout_seconds": (
            config.timeouts.
            operation_seconds
        ),

        "retry_enabled": (
            config.retries.enabled
        ),

        "retry_maximum_attempts": (
            config.retries.
            maximum_attempts
        ),

        "maximum_operations": (
            config.concurrency.
            maximum_operations
        ),

        "maximum_mutating_operations": (
            config.concurrency.
            maximum_mutating_operations
        ),

        "health_enabled": (
            config.health.enabled
        ),

        "allow_read_operations": (
            config.operations.
            allow_read_operations
        ),

        "allow_mutating_operations": (
            config.operations.
            allow_mutating_operations
        ),

        "require_security_context_for_mutations": (
            config.operations.
            require_security_context_for_mutations
        ),

        "require_explicit_capability": (
            config.operations.
            require_explicit_capability
        ),

        "credentials_present": (
            False
        ),
    }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_ENV_PREFIX",

    "TimeoutConfig",
    "RetryConfig",
    "ConcurrencyConfig",
    "HealthConfig",
    "OperationPolicyConfig",

    "IntegrationsConfig",

    "from_mapping",
    "from_environment",

    "safe_config_summary",
]
