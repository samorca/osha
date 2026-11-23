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

from src.core.validation import (
    boolean,
    integer,
    number,
    one_of,
)

from src.observability.schema import (
    Severity,
)


# ---------------------------------------------------------------
# Environment prefix
# ---------------------------------------------------------------

DEFAULT_ENV_PREFIX = (
    "OSHA_OBSERVABILITY_"
)


# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------

@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool = True

    minimum_severity: str = (
        Severity.INFO.value
    )

    structured: bool = True

    include_context: bool = True

    def validate(
        self,
    ) -> None:
        one_of(
            self.minimum_severity,
            (
                item.value
                for item
                in Severity
            ),
            name="minimum_severity",
        )


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------

@dataclass(frozen=True)
class MetricsConfig:
    enabled: bool = True

    collection_interval_seconds: float = 15.0

    maximum_buffered_samples: int = 5000

    def validate(
        self,
    ) -> None:
        number(
            self.collection_interval_seconds,
            name=(
                "collection_interval_seconds"
            ),
            minimum=0.001,
        )

        integer(
            self.maximum_buffered_samples,
            name=(
                "maximum_buffered_samples"
            ),
            minimum=1,
        )


# ---------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------

@dataclass(frozen=True)
class TracingConfig:
    enabled: bool = True

    # 0.0 = no sampled traces
    # 1.0 = every trace sampled
    sample_rate: float = 1.0

    maximum_buffered_spans: int = 5000

    def validate(
        self,
    ) -> None:
        number(
            self.sample_rate,
            name="sample_rate",
            minimum=0.0,
            maximum=1.0,
        )

        integer(
            self.maximum_buffered_spans,
            name=(
                "maximum_buffered_spans"
            ),
            minimum=1,
        )


# ---------------------------------------------------------------
# Events
# ---------------------------------------------------------------

@dataclass(frozen=True)
class EventsConfig:
    enabled: bool = True

    minimum_severity: str = (
        Severity.INFO.value
    )

    maximum_buffered_events: int = 5000

    def validate(
        self,
    ) -> None:
        one_of(
            self.minimum_severity,
            (
                item.value
                for item
                in Severity
            ),
            name="minimum_severity",
        )

        integer(
            self.maximum_buffered_events,
            name=(
                "maximum_buffered_events"
            ),
            minimum=1,
        )


# ---------------------------------------------------------------
# Health
# ---------------------------------------------------------------

@dataclass(frozen=True)
class HealthConfig:
    enabled: bool = True

    observation_interval_seconds: float = 15.0

    maximum_buffered_observations: int = 1000

    def validate(
        self,
    ) -> None:
        number(
            self.observation_interval_seconds,
            name=(
                "observation_interval_seconds"
            ),
            minimum=0.001,
        )

        integer(
            self.
            maximum_buffered_observations,
            name=(
                "maximum_buffered_observations"
            ),
            minimum=1,
        )


# ---------------------------------------------------------------
# Export
# ---------------------------------------------------------------

@dataclass(frozen=True)
class ExportConfig:
    """
    Non-secret exporter configuration.

    Credentials/endpoints that require secrets
    should eventually be resolved through the
    security/integration layer rather than stored
    directly here.
    """

    enabled: bool = False

    exporter: str = "none"

    batch_size: int = 100

    flush_interval_seconds: float = 5.0

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        one_of(
            self.exporter,
            (
                "none",
                "console",
                "file",
                "custom",
            ),
            name="exporter",
        )

        integer(
            self.batch_size,
            name="batch_size",
            minimum=1,
        )

        number(
            self.flush_interval_seconds,
            name=(
                "flush_interval_seconds"
            ),
            minimum=0.001,
        )

        if (
            self.enabled
            and self.exporter
            == "none"
        ):
            raise ValueError(
                "An enabled observability "
                "exporter cannot use "
                "exporter='none'"
            )


# ---------------------------------------------------------------
# Main observability configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class ObservabilityConfig:
    logging: LoggingConfig = field(
        default_factory=LoggingConfig
    )

    metrics: MetricsConfig = field(
        default_factory=MetricsConfig
    )

    tracing: TracingConfig = field(
        default_factory=TracingConfig
    )

    events: EventsConfig = field(
        default_factory=EventsConfig
    )

    health: HealthConfig = field(
        default_factory=HealthConfig
    )

    export: ExportConfig = field(
        default_factory=ExportConfig
    )

    # Keep redaction enabled unless explicitly
    # overridden by controlled configuration.
    redact_sensitive: bool = True

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        self.logging.validate()
        self.metrics.validate()
        self.tracing.validate()
        self.events.validate()
        self.health.validate()
        self.export.validate()

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "logging": {
                "enabled": (
                    self.logging.enabled
                ),
                "minimum_severity": (
                    self.logging.
                    minimum_severity
                ),
                "structured": (
                    self.logging.
                    structured
                ),
                "include_context": (
                    self.logging.
                    include_context
                ),
            },

            "metrics": {
                "enabled": (
                    self.metrics.enabled
                ),
                "collection_interval_seconds": (
                    self.metrics.
                    collection_interval_seconds
                ),
                "maximum_buffered_samples": (
                    self.metrics.
                    maximum_buffered_samples
                ),
            },

            "tracing": {
                "enabled": (
                    self.tracing.enabled
                ),
                "sample_rate": (
                    self.tracing.
                    sample_rate
                ),
                "maximum_buffered_spans": (
                    self.tracing.
                    maximum_buffered_spans
                ),
            },

            "events": {
                "enabled": (
                    self.events.enabled
                ),
                "minimum_severity": (
                    self.events.
                    minimum_severity
                ),
                "maximum_buffered_events": (
                    self.events.
                    maximum_buffered_events
                ),
            },

            "health": {
                "enabled": (
                    self.health.enabled
                ),
                "observation_interval_seconds": (
                    self.health.
                    observation_interval_seconds
                ),
                "maximum_buffered_observations": (
                    self.health.
                    maximum_buffered_observations
                ),
            },

            "export": {
                "enabled": (
                    self.export.enabled
                ),
                "exporter": (
                    self.export.exporter
                ),
                "batch_size": (
                    self.export.
                    batch_size
                ),
                "flush_interval_seconds": (
                    self.export.
                    flush_interval_seconds
                ),
                "attributes": dict(
                    self.export.
                    attributes
                ),
            },

            "redact_sensitive": (
                self.redact_sensitive
            ),

            "attributes": dict(
                self.attributes
            ),
        }


# ---------------------------------------------------------------
# Mapping-read helpers
# ---------------------------------------------------------------

def _read_bool(
    values: Mapping[
        str,
        Any,
    ],
    key: str,
    default: bool,
) -> bool:
    value = values.get(
        key
    )

    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    return boolean(
        value,
        name=key,
    )


def _read_int(
    values: Mapping[
        str,
        Any,
    ],
    key: str,
    default: int,
) -> int:
    value = values.get(
        key
    )

    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    return integer(
        value,
        name=key,
        minimum=1,
    )


def _read_float(
    values: Mapping[
        str,
        Any,
    ],
    key: str,
    default: float,
    *,
    minimum: float = 0.001,
    maximum: Optional[
        float
    ] = None,
) -> float:
    value = values.get(
        key
    )

    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    return number(
        value,
        name=key,
        minimum=minimum,
        maximum=maximum,
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
) -> ObservabilityConfig:
    """
    Construct observability configuration from
    an arbitrary mapping.

    This makes environment configuration easy to
    test without mutating os.environ.
    """

    def key(
        name: str,
    ) -> str:
        return (
            prefix
            + name
        )

    config = ObservabilityConfig(
        logging=LoggingConfig(
            enabled=_read_bool(
                values,
                key(
                    "LOGGING_ENABLED"
                ),
                True,
            ),

            minimum_severity=str(
                values.get(
                    key(
                        "LOGGING_MINIMUM_SEVERITY"
                    ),
                    Severity.INFO.value,
                )
            ).strip().lower(),

            structured=_read_bool(
                values,
                key(
                    "LOGGING_STRUCTURED"
                ),
                True,
            ),

            include_context=_read_bool(
                values,
                key(
                    "LOGGING_INCLUDE_CONTEXT"
                ),
                True,
            ),
        ),

        metrics=MetricsConfig(
            enabled=_read_bool(
                values,
                key(
                    "METRICS_ENABLED"
                ),
                True,
            ),

            collection_interval_seconds=(
                _read_float(
                    values,
                    key(
                        "METRICS_INTERVAL_SECONDS"
                    ),
                    15.0,
                )
            ),

            maximum_buffered_samples=(
                _read_int(
                    values,
                    key(
                        "METRICS_MAX_BUFFERED"
                    ),
                    5000,
                )
            ),
        ),

        tracing=TracingConfig(
            enabled=_read_bool(
                values,
                key(
                    "TRACING_ENABLED"
                ),
                True,
            ),

            sample_rate=_read_float(
                values,
                key(
                    "TRACING_SAMPLE_RATE"
                ),
                1.0,
                minimum=0.0,
                maximum=1.0,
            ),

            maximum_buffered_spans=(
                _read_int(
                    values,
                    key(
                        "TRACING_MAX_BUFFERED"
                    ),
                    5000,
                )
            ),
        ),

        events=EventsConfig(
            enabled=_read_bool(
                values,
                key(
                    "EVENTS_ENABLED"
                ),
                True,
            ),

            minimum_severity=str(
                values.get(
                    key(
                        "EVENTS_MINIMUM_SEVERITY"
                    ),
                    Severity.INFO.value,
                )
            ).strip().lower(),

            maximum_buffered_events=(
                _read_int(
                    values,
                    key(
                        "EVENTS_MAX_BUFFERED"
                    ),
                    5000,
                )
            ),
        ),

        health=HealthConfig(
            enabled=_read_bool(
                values,
                key(
                    "HEALTH_ENABLED"
                ),
                True,
            ),

            observation_interval_seconds=(
                _read_float(
                    values,
                    key(
                        "HEALTH_INTERVAL_SECONDS"
                    ),
                    15.0,
                )
            ),

            maximum_buffered_observations=(
                _read_int(
                    values,
                    key(
                        "HEALTH_MAX_BUFFERED"
                    ),
                    1000,
                )
            ),
        ),

        export=ExportConfig(
            enabled=_read_bool(
                values,
                key(
                    "EXPORT_ENABLED"
                ),
                False,
            ),

            exporter=str(
                values.get(
                    key(
                        "EXPORTER"
                    ),
                    "none",
                )
            ).strip().lower(),

            batch_size=_read_int(
                values,
                key(
                    "EXPORT_BATCH_SIZE"
                ),
                100,
            ),

            flush_interval_seconds=(
                _read_float(
                    values,
                    key(
                        "EXPORT_FLUSH_INTERVAL_SECONDS"
                    ),
                    5.0,
                )
            ),
        ),

        redact_sensitive=_read_bool(
            values,
            key(
                "REDACT_SENSITIVE"
            ),
            True,
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
) -> ObservabilityConfig:
    source = (
        os.environ
        if environ is None
        else environ
    )

    config = from_mapping(
        source,
        prefix=prefix,
    )

    return ObservabilityConfig(
        logging=config.logging,
        metrics=config.metrics,
        tracing=config.tracing,
        events=config.events,
        health=config.health,
        export=config.export,
        redact_sensitive=(
            config.redact_sensitive
        ),
        attributes={
            **config.attributes,
            "config_source": (
                "environment"
            ),
        },
    )


# ---------------------------------------------------------------
# Safe configuration summary
# ---------------------------------------------------------------

def safe_config_summary(
    config: ObservabilityConfig,
) -> Dict[str, Any]:
    """
    Non-secret observability configuration
    suitable for application status output.
    """

    config.validate()

    return {
        "logging_enabled": (
            config.logging.enabled
        ),
        "logging_minimum_severity": (
            config.logging.
            minimum_severity
        ),

        "metrics_enabled": (
            config.metrics.enabled
        ),
        "metrics_interval_seconds": (
            config.metrics.
            collection_interval_seconds
        ),

        "tracing_enabled": (
            config.tracing.enabled
        ),
        "tracing_sample_rate": (
            config.tracing.
            sample_rate
        ),

        "events_enabled": (
            config.events.enabled
        ),

        "health_enabled": (
            config.health.enabled
        ),

        "export_enabled": (
            config.export.enabled
        ),
        "exporter": (
            config.export.exporter
        ),

        "redact_sensitive": (
            config.redact_sensitive
        ),
    }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_ENV_PREFIX",

    "LoggingConfig",
    "MetricsConfig",
    "TracingConfig",
    "EventsConfig",
    "HealthConfig",
    "ExportConfig",

    "ObservabilityConfig",

    "from_mapping",
    "from_environment",

    "safe_config_summary",
]
