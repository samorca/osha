from __future__ import annotations

"""
Vendor-neutral, bounded tracing for OSHA
application components.
"""

import random

from collections import deque
from threading import RLock

from typing import (
    Any,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
)

from src.core.identifiers import (
    random_id,
)

from src.core.serialization import (
    DEFAULT_SENSITIVE_KEYS,
    to_json_safe,
)

from src.core.time import (
    duration_seconds,
    utc_now,
)

from src.core.validation import (
    integer,
)

from src.observability.config import (
    TracingConfig,
)

from src.observability.schema import (
    ObservabilityContext,
    SpanStatus,
    TraceSpan,
)


# ---------------------------------------------------------------
# Errors
# ---------------------------------------------------------------


class TracingError(RuntimeError):
    pass


# ---------------------------------------------------------------
# Trace identity
# ---------------------------------------------------------------


def new_trace_id() -> str:
    return random_id("observability:trace")


# ---------------------------------------------------------------
# Trace buffer
# ---------------------------------------------------------------


class TraceBuffer:
    """
    Thread-safe bounded trace-span buffer.

    This is vendor-neutral. It does not depend on
    OpenTelemetry, Azure Monitor, Jaeger, Tempo,
    or another tracing backend.
    """

    def __init__(
        self,
        *,
        config: Optional[TracingConfig] = None,
        redact_sensitive: bool = True,
    ) -> None:
        self.config = config or TracingConfig()

        self.config.validate()

        self.redact_sensitive = bool(redact_sensitive)

        self._spans: Deque[TraceSpan] = deque(
            maxlen=(self.config.maximum_buffered_spans)
        )

        self._active: Dict[
            str,
            TraceSpan,
        ] = {}

        self._lock = RLock()

        self._accepted = 0
        self._sampled_out = 0
        self._evicted = 0

    # ---------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------

    def should_sample(
        self,
    ) -> bool:
        if not self.config.enabled:
            return False

        if self.config.sample_rate <= 0.0:
            return False

        if self.config.sample_rate >= 1.0:
            return True

        return random.random() < self.config.sample_rate

    # ---------------------------------------------------------
    # Start span
    # ---------------------------------------------------------

    def start_span(
        self,
        *,
        name: str,
        context: ObservabilityContext,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        sampled: Optional[bool] = None,
    ) -> Optional[TraceSpan]:
        """
        Start a trace span.

        Returns None if tracing is disabled or
        the trace is not sampled.
        """

        selected = self.should_sample() if sampled is None else bool(sampled)

        if not self.config.enabled or not selected:
            with self._lock:
                self._sampled_out += 1

            return None

        span = TraceSpan(
            name=name,
            context=context,
            trace_id=(trace_id or new_trace_id()),
            parent_span_id=(parent_span_id),
            attributes=(
                to_json_safe(
                    dict(attributes or {}),
                    redact_sensitive=(self.redact_sensitive),
                    sensitive_keys=(DEFAULT_SENSITIVE_KEYS),
                )
            ),
        ).finalize()

        with self._lock:
            if span.span_id in self._active:
                raise TracingError(f"Duplicate active span_id: {span.span_id}")

            self._active[span.span_id] = span

        return span

    # ---------------------------------------------------------
    # Child span
    # ---------------------------------------------------------

    def child_span(
        self,
        parent: TraceSpan,
        *,
        name: str,
        context: Optional[ObservabilityContext] = None,
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> Optional[TraceSpan]:
        """
        Start a child of an already sampled
        parent.

        Child spans preserve the parent's trace
        ID.
        """

        parent.finalize()

        return self.start_span(
            name=name,
            context=(context or parent.context),
            trace_id=(parent.trace_id),
            parent_span_id=(parent.span_id),
            attributes=attributes,
            sampled=True,
        )

    # ---------------------------------------------------------
    # Finish span
    # ---------------------------------------------------------

    def finish_span(
        self,
        span: TraceSpan,
        *,
        status: str = (SpanStatus.OK.value),
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        error: Optional[BaseException] = None,
    ) -> TraceSpan:
        """
        Complete an active span.

        Arbitrary exception messages are not
        copied. Only bounded exception type/module
        metadata is attached.
        """

        span.finalize()

        if status not in {item.value for item in SpanStatus}:
            raise TracingError(f"Unknown span status: {status}")

        if error is not None:
            status = SpanStatus.ERROR.value

        with self._lock:
            if span.span_id not in self._active:
                raise TracingError(f"Span is not active: {span.span_id}")

            merged = dict(span.attributes)

            if attributes:
                merged.update(
                    to_json_safe(
                        dict(attributes),
                        redact_sensitive=(self.redact_sensitive),
                        sensitive_keys=(DEFAULT_SENSITIVE_KEYS),
                    )
                )

            if error is not None:
                merged.update(
                    {
                        "error_type": (type(error).__name__),
                        "error_module": (type(error).__module__),
                    }
                )

            span.attributes = merged

            span.status = status

            span.ended_at = utc_now()

            span.attributes["duration_seconds"] = max(
                0.0,
                duration_seconds(
                    span.started_at,
                    span.ended_at,
                ),
            )

            del self._active[span.span_id]

            was_full = len(self._spans) == self._spans.maxlen

            self._spans.append(span)

            self._accepted += 1

            if was_full:
                self._evicted += 1

        return span.finalize()

    # ---------------------------------------------------------
    # Abandon span
    # ---------------------------------------------------------

    def abandon_span(
        self,
        span: TraceSpan,
        *,
        reason: str = ("Span abandoned."),
    ) -> TraceSpan:
        return self.finish_span(
            span,
            status=(SpanStatus.ERROR.value),
            attributes={
                "abandoned": True,
                "reason": reason,
            },
        )

    # ---------------------------------------------------------
    # Query completed spans
    # ---------------------------------------------------------

    def spans(
        self,
        *,
        trace_id: Optional[str] = None,
        service: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[TraceSpan]:
        if limit is not None:
            limit = integer(
                limit,
                name="limit",
                minimum=1,
            )

        if status is not None and status not in {item.value for item in SpanStatus}:
            raise TracingError(f"Unknown span status: {status}")

        with self._lock:
            values = list(self._spans)

        if newest_first:
            values.reverse()

        output: List[TraceSpan] = []

        for span in values:
            if trace_id is not None and span.trace_id != trace_id:
                continue

            if service is not None and span.context.service != service:
                continue

            if status is not None and span.status != status:
                continue

            output.append(span)

            if limit is not None and len(output) >= limit:
                break

        return output

    # ---------------------------------------------------------
    # Active spans
    # ---------------------------------------------------------

    def active_spans(
        self,
    ) -> List[TraceSpan]:
        with self._lock:
            return list(self._active.values())

    # ---------------------------------------------------------
    # Drain completed spans
    # ---------------------------------------------------------

    def drain(
        self,
        *,
        limit: Optional[int] = None,
    ) -> List[TraceSpan]:
        if limit is not None:
            limit = integer(
                limit,
                name="limit",
                minimum=1,
            )

        with self._lock:
            count = (
                len(self._spans)
                if limit is None
                else min(
                    limit,
                    len(self._spans),
                )
            )

            return [self._spans.popleft() for _ in range(count)]

    # ---------------------------------------------------------
    # Clear completed spans
    # ---------------------------------------------------------

    def clear_completed(
        self,
    ) -> int:
        with self._lock:
            count = len(self._spans)

            self._spans.clear()

            return count

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

            for span in self._spans:
                statuses[span.status] = (
                    statuses.get(
                        span.status,
                        0,
                    )
                    + 1
                )

            return {
                "enabled": (self.config.enabled),
                "sample_rate": (self.config.sample_rate),
                "active_spans": (len(self._active)),
                "buffered_spans": (len(self._spans)),
                "maximum_buffered_spans": (self.config.maximum_buffered_spans),
                "accepted_total": (self._accepted),
                "sampled_out_total": (self._sampled_out),
                "evicted_total": (self._evicted),
                "buffered_by_status": (statuses),
            }


# ---------------------------------------------------------------
# Span context manager
# ---------------------------------------------------------------


class SpanScope:
    """
    Context-manager helper for one TraceSpan.

    It automatically marks normal completion OK
    and exceptional completion ERROR.

    Original exceptions are not suppressed.
    """

    def __init__(
        self,
        buffer: TraceBuffer,
        span: Optional[TraceSpan],
    ) -> None:
        self.buffer = buffer
        self.span = span

    def __enter__(
        self,
    ) -> Optional[TraceSpan]:
        return self.span

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:
        if self.span is None:
            return False

        if exc is None:
            self.buffer.finish_span(
                self.span,
                status=(SpanStatus.OK.value),
            )

        else:
            self.buffer.finish_span(
                self.span,
                error=exc,
            )

        return False


# ---------------------------------------------------------------
# Convenience tracing helper
# ---------------------------------------------------------------


def traced(
    buffer: TraceBuffer,
    *,
    name: str,
    context: ObservabilityContext,
    trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    attributes: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> SpanScope:
    """
    Example:

        with traced(
            traces,
            name="worker.process",
            context=context,
        ):
            process_job()

    If tracing is sampled out, the context
    manager still works and simply contains None.
    """

    return SpanScope(
        buffer,
        buffer.start_span(
            name=name,
            context=context,
            trace_id=trace_id,
            parent_span_id=(parent_span_id),
            attributes=attributes,
        ),
    )


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "TracingError",
    "new_trace_id",
    "TraceBuffer",
    "SpanScope",
    "traced",
]
