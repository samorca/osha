from __future__ import annotations

"""
Shared observability records for OSHA's own
application components.
"""

from dataclasses import (
    dataclass,
    field,
)

from enum import Enum

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from src.core.identifiers import (
    deterministic_id,
    random_id,
)

from src.core.serialization import (
    to_json_safe,
)

from src.core.time import (
    utc_now,
)

from src.core.validation import (
    environment_id,
    number,
    required_string,
    unique_strings,
)


# ---------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------

OBSERVABILITY_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------
# Signal kinds
# ---------------------------------------------------------------


class SignalKind(
    str,
    Enum,
):
    EVENT = "event"

    LOG = "log"

    METRIC = "metric"

    TRACE = "trace"

    HEALTH = "health"


# ---------------------------------------------------------------
# Severity
# ---------------------------------------------------------------


class Severity(
    str,
    Enum,
):
    DEBUG = "debug"

    INFO = "info"

    WARNING = "warning"

    ERROR = "error"

    CRITICAL = "critical"


# ---------------------------------------------------------------
# Health status
# ---------------------------------------------------------------


class HealthStatus(
    str,
    Enum,
):
    UNKNOWN = "unknown"

    HEALTHY = "healthy"

    DEGRADED = "degraded"

    UNHEALTHY = "unhealthy"


# ---------------------------------------------------------------
# Metric type
# ---------------------------------------------------------------


class MetricType(
    str,
    Enum,
):
    COUNTER = "counter"

    GAUGE = "gauge"

    HISTOGRAM = "histogram"


# ---------------------------------------------------------------
# Trace span status
# ---------------------------------------------------------------


class SpanStatus(
    str,
    Enum,
):
    UNSET = "unset"

    OK = "ok"

    ERROR = "error"


# ---------------------------------------------------------------
# Observability context
# ---------------------------------------------------------------


@dataclass(frozen=True)
class ObservabilityContext:
    """
    Identifies where an OSHA observability signal
    originated.

    These fields correlate OSHA's internal
    runtime/workers/orchestration activity.

    They do not claim health or state of external
    infrastructure.
    """

    environment_id: str

    service: str

    component: Optional[str] = None

    runtime_id: Optional[str] = None

    worker_id: Optional[str] = None

    orchestration_run_id: Optional[str] = None

    job_id: Optional[str] = None

    correlation_id: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "ObservabilityContext":
        environment_id(self.environment_id)

        required_string(
            self.service,
            name="service",
            maximum_length=128,
        )

        if self.component is not None:
            required_string(
                self.component,
                name="component",
                maximum_length=128,
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "environment_id": (self.environment_id),
            "service": self.service,
            "component": (self.component),
            "runtime_id": (self.runtime_id),
            "worker_id": (self.worker_id),
            "orchestration_run_id": (self.orchestration_run_id),
            "job_id": self.job_id,
            "correlation_id": (self.correlation_id),
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Application event
# ---------------------------------------------------------------


@dataclass
class ObservabilityEvent:
    """
    Structured event emitted by an OSHA
    application component.
    """

    name: str

    context: ObservabilityContext

    severity: str = Severity.INFO.value

    message: Optional[str] = None

    event_type: Optional[str] = None

    reference_ids: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    occurred_at: str = field(default_factory=utc_now)

    event_id: Optional[str] = None

    schema_version: str = OBSERVABILITY_SCHEMA_VERSION

    def finalize(
        self,
    ) -> "ObservabilityEvent":
        self.context.finalize()

        required_string(
            self.name,
            name="event name",
            maximum_length=256,
        )

        if self.severity not in {item.value for item in Severity}:
            raise ValueError(f"Unknown event severity: {self.severity}")

        self.reference_ids = unique_strings(self.reference_ids)

        if not self.event_id:
            self.event_id = random_id("observability:event")

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (self.schema_version),
            "event_id": (self.event_id),
            "kind": (SignalKind.EVENT.value),
            "name": self.name,
            "severity": (self.severity),
            "message": (self.message),
            "event_type": (self.event_type),
            "context": (self.context.to_dict()),
            "reference_ids": (self.reference_ids),
            "attributes": (to_json_safe(self.attributes)),
            "occurred_at": (self.occurred_at),
        }


# ---------------------------------------------------------------
# Log record
# ---------------------------------------------------------------


@dataclass
class LogRecord:
    """
    Structured OSHA application log record.

    Do not place raw credentials, tokens, or
    secrets into message fields.
    """

    message: str

    context: ObservabilityContext

    severity: str = Severity.INFO.value

    logger: str = "osha"

    error_type: Optional[str] = None

    reference_ids: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    occurred_at: str = field(default_factory=utc_now)

    log_id: Optional[str] = None

    schema_version: str = OBSERVABILITY_SCHEMA_VERSION

    def finalize(
        self,
    ) -> "LogRecord":
        self.context.finalize()

        required_string(
            self.message,
            name="log message",
            maximum_length=8192,
        )

        required_string(
            self.logger,
            name="logger",
            maximum_length=256,
        )

        if self.severity not in {item.value for item in Severity}:
            raise ValueError(f"Unknown log severity: {self.severity}")

        self.reference_ids = unique_strings(self.reference_ids)

        if not self.log_id:
            self.log_id = random_id("observability:log")

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (self.schema_version),
            "log_id": self.log_id,
            "kind": (SignalKind.LOG.value),
            "message": self.message,
            "severity": (self.severity),
            "logger": self.logger,
            "error_type": (self.error_type),
            "context": (self.context.to_dict()),
            "reference_ids": (self.reference_ids),
            "attributes": (to_json_safe(self.attributes)),
            "occurred_at": (self.occurred_at),
        }


# ---------------------------------------------------------------
# Metric sample
# ---------------------------------------------------------------


@dataclass
class MetricSample:
    """
    One metric value observed from an OSHA
    application component.
    """

    name: str

    value: float

    context: ObservabilityContext

    metric_type: str = MetricType.GAUGE.value

    unit: Optional[str] = None

    labels: Dict[
        str,
        str,
    ] = field(default_factory=dict)

    observed_at: str = field(default_factory=utc_now)

    metric_id: Optional[str] = None

    schema_version: str = OBSERVABILITY_SCHEMA_VERSION

    def finalize(
        self,
    ) -> "MetricSample":
        self.context.finalize()

        required_string(
            self.name,
            name="metric name",
            maximum_length=256,
        )

        self.value = number(
            self.value,
            name="metric value",
        )

        if self.metric_type not in {item.value for item in MetricType}:
            raise ValueError(f"Unknown metric type: {self.metric_type}")

        self.labels = {str(key): str(value) for key, value in self.labels.items()}

        if not self.metric_id:
            self.metric_id = deterministic_id(
                "observability:metric",
                {
                    "name": self.name,
                    "environment_id": (self.context.environment_id),
                    "service": (self.context.service),
                    "component": (self.context.component),
                    "labels": (self.labels),
                    "observed_at": (self.observed_at),
                },
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (self.schema_version),
            "metric_id": (self.metric_id),
            "kind": (SignalKind.METRIC.value),
            "name": self.name,
            "value": self.value,
            "metric_type": (self.metric_type),
            "unit": self.unit,
            "labels": dict(self.labels),
            "context": (self.context.to_dict()),
            "observed_at": (self.observed_at),
        }


# ---------------------------------------------------------------
# Trace span
# ---------------------------------------------------------------


@dataclass
class TraceSpan:
    """
    One tracing span representing an OSHA
    application operation.
    """

    name: str

    context: ObservabilityContext

    trace_id: str

    parent_span_id: Optional[str] = None

    span_id: Optional[str] = None

    status: str = SpanStatus.UNSET.value

    started_at: str = field(default_factory=utc_now)

    ended_at: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    schema_version: str = OBSERVABILITY_SCHEMA_VERSION

    def finalize(
        self,
    ) -> "TraceSpan":
        self.context.finalize()

        required_string(
            self.name,
            name="span name",
            maximum_length=256,
        )

        required_string(
            self.trace_id,
            name="trace_id",
            maximum_length=256,
        )

        if self.status not in {item.value for item in SpanStatus}:
            raise ValueError(f"Unknown span status: {self.status}")

        if not self.span_id:
            self.span_id = random_id("observability:span")

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (self.schema_version),
            "kind": (SignalKind.TRACE.value),
            "trace_id": (self.trace_id),
            "span_id": self.span_id,
            "parent_span_id": (self.parent_span_id),
            "name": self.name,
            "status": self.status,
            "context": (self.context.to_dict()),
            "started_at": (self.started_at),
            "ended_at": (self.ended_at),
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Health observation
# ---------------------------------------------------------------


@dataclass
class HealthObservation:
    """
    Observation about an OSHA application
    component.

    This should not be used as evidence that an
    external cloud/server/network is healthy.
    """

    component: str

    status: str

    context: ObservabilityContext

    message: Optional[str] = None

    required: bool = True

    evidence_reference_ids: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    observed_at: str = field(default_factory=utc_now)

    observation_id: Optional[str] = None

    schema_version: str = OBSERVABILITY_SCHEMA_VERSION

    def finalize(
        self,
    ) -> "HealthObservation":
        self.context.finalize()

        required_string(
            self.component,
            name="health component",
            maximum_length=256,
        )

        if self.status not in {item.value for item in HealthStatus}:
            raise ValueError(f"Unknown health status: {self.status}")

        self.evidence_reference_ids = unique_strings(self.evidence_reference_ids)

        if not self.observation_id:
            self.observation_id = random_id("observability:health")

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (self.schema_version),
            "observation_id": (self.observation_id),
            "kind": (SignalKind.HEALTH.value),
            "component": (self.component),
            "status": self.status,
            "required": self.required,
            "message": self.message,
            "context": (self.context.to_dict()),
            "evidence_reference_ids": (self.evidence_reference_ids),
            "attributes": (to_json_safe(self.attributes)),
            "observed_at": (self.observed_at),
        }


# ---------------------------------------------------------------
# Observability batch
# ---------------------------------------------------------------


@dataclass
class ObservabilityBatch:
    """
    Bounded collection of OSHA observability
    signals.

    Export/storage behavior belongs in later
    observability modules.
    """

    events: List[ObservabilityEvent] = field(default_factory=list)

    logs: List[LogRecord] = field(default_factory=list)

    metrics: List[MetricSample] = field(default_factory=list)

    spans: List[TraceSpan] = field(default_factory=list)

    health: List[HealthObservation] = field(default_factory=list)

    generated_at: str = field(default_factory=utc_now)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    schema_version: str = OBSERVABILITY_SCHEMA_VERSION

    def finalize(
        self,
    ) -> "ObservabilityBatch":
        self.events = [item.finalize() for item in self.events]

        self.logs = [item.finalize() for item in self.logs]

        self.metrics = [item.finalize() for item in self.metrics]

        self.spans = [item.finalize() for item in self.spans]

        self.health = [item.finalize() for item in self.health]

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (self.schema_version),
            "generated_at": (self.generated_at),
            "counts": {
                "events": len(self.events),
                "logs": len(self.logs),
                "metrics": len(self.metrics),
                "spans": len(self.spans),
                "health": len(self.health),
            },
            "events": [item.to_dict() for item in self.events],
            "logs": [item.to_dict() for item in self.logs],
            "metrics": [item.to_dict() for item in self.metrics],
            "spans": [item.to_dict() for item in self.spans],
            "health": [item.to_dict() for item in self.health],
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "OBSERVABILITY_SCHEMA_VERSION",
    "SignalKind",
    "Severity",
    "HealthStatus",
    "MetricType",
    "SpanStatus",
    "ObservabilityContext",
    "ObservabilityEvent",
    "LogRecord",
    "MetricSample",
    "TraceSpan",
    "HealthObservation",
    "ObservabilityBatch",
]
