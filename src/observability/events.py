from __future__ import annotations

"""
Bounded in-memory recording for OSHA application
events.
"""

from collections import deque
from threading import RLock

from typing import (
    Any,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.core.validation import (
    integer,
)

from src.observability.config import (
    EventsConfig,
)

from src.observability.schema import (
    ObservabilityContext,
    ObservabilityEvent,
    Severity,
)


# ---------------------------------------------------------------
# Severity ordering
# ---------------------------------------------------------------

SEVERITY_RANK = {
    Severity.DEBUG.value: 10,
    Severity.INFO.value: 20,
    Severity.WARNING.value: 30,
    Severity.ERROR.value: 40,
    Severity.CRITICAL.value: 50,
}


# ---------------------------------------------------------------
# Event errors
# ---------------------------------------------------------------


class EventStoreError(RuntimeError):
    pass


# ---------------------------------------------------------------
# Severity comparison
# ---------------------------------------------------------------


def severity_at_least(
    severity: str,
    minimum: str,
) -> bool:
    """
    Return whether severity meets the configured
    minimum threshold.
    """

    if severity not in SEVERITY_RANK:
        raise EventStoreError(f"Unknown severity: {severity}")

    if minimum not in SEVERITY_RANK:
        raise EventStoreError(f"Unknown minimum severity: {minimum}")

    return SEVERITY_RANK[severity] >= SEVERITY_RANK[minimum]


# ---------------------------------------------------------------
# Event buffer
# ---------------------------------------------------------------


class EventBuffer:
    """
    Thread-safe bounded event buffer for OSHA's
    own application events.

    This is an in-process buffer only.

    Durable persistence/export belongs in later
    integration/export layers.
    """

    def __init__(
        self,
        *,
        config: Optional[EventsConfig] = None,
    ) -> None:
        self.config = config or EventsConfig()

        self.config.validate()

        self._events: Deque[ObservabilityEvent] = deque(
            maxlen=(self.config.maximum_buffered_events)
        )

        self._lock = RLock()

        self._accepted = 0
        self._filtered = 0
        self._evicted = 0

    # ---------------------------------------------------------
    # Acceptance
    # ---------------------------------------------------------

    def accepts(
        self,
        event: ObservabilityEvent,
    ) -> bool:
        event.finalize()

        return bool(
            self.config.enabled
            and severity_at_least(
                event.severity,
                self.config.minimum_severity,
            )
        )

    # ---------------------------------------------------------
    # Record existing event
    # ---------------------------------------------------------

    def record(
        self,
        event: ObservabilityEvent,
    ) -> bool:
        """
        Record an event if enabled and its
        severity satisfies the threshold.

        Returns True if recorded.
        """

        event.finalize()

        with self._lock:
            if not self.accepts(event):
                self._filtered += 1

                return False

            was_full = len(self._events) == self._events.maxlen

            self._events.append(event)

            self._accepted += 1

            # deque(maxlen=...) evicts the oldest
            # event automatically when full.
            if was_full:
                self._evicted += 1

            return True

    # ---------------------------------------------------------
    # Emit event
    # ---------------------------------------------------------

    def emit(
        self,
        *,
        name: str,
        context: ObservabilityContext,
        severity: str = (Severity.INFO.value),
        message: Optional[str] = None,
        event_type: Optional[str] = None,
        reference_ids: Iterable[str] = (),
        attributes: Optional[
            Dict[
                str,
                Any,
            ]
        ] = None,
    ) -> Optional[ObservabilityEvent]:
        """
        Construct and record a structured event.

        Returns None when filtering/configuration
        prevents the event from being buffered.
        """

        event = ObservabilityEvent(
            name=name,
            context=context,
            severity=severity,
            message=message,
            event_type=event_type,
            reference_ids=list(reference_ids),
            attributes=dict(attributes or {}),
        ).finalize()

        if self.record(event):
            return event

        return None

    # ---------------------------------------------------------
    # Query buffered events
    # ---------------------------------------------------------

    def list(
        self,
        *,
        minimum_severity: Optional[str] = None,
        service: Optional[str] = None,
        component: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[ObservabilityEvent]:
        """
        Query the current event buffer.

        Querying does not remove events.
        """

        if limit is not None:
            limit = integer(
                limit,
                name="limit",
                minimum=1,
            )

        with self._lock:
            events = list(self._events)

        if newest_first:
            events.reverse()

        output: List[ObservabilityEvent] = []

        for event in events:
            if minimum_severity is not None and not severity_at_least(
                event.severity,
                minimum_severity,
            ):
                continue

            if service is not None and event.context.service != service:
                continue

            if component is not None and event.context.component != component:
                continue

            if event_type is not None and event.event_type != event_type:
                continue

            output.append(event)

            if limit is not None and len(output) >= limit:
                break

        return output

    # ---------------------------------------------------------
    # Drain
    # ---------------------------------------------------------

    def drain(
        self,
        *,
        limit: Optional[int] = None,
    ) -> List[ObservabilityEvent]:
        """
        Remove and return the oldest buffered
        events.

        This can later be used by an exporter.
        """

        if limit is not None:
            limit = integer(
                limit,
                name="limit",
                minimum=1,
            )

        with self._lock:
            count = (
                len(self._events)
                if limit is None
                else min(
                    limit,
                    len(self._events),
                )
            )

            return [self._events.popleft() for _ in range(count)]

    # ---------------------------------------------------------
    # Clear
    # ---------------------------------------------------------

    def clear(
        self,
    ) -> int:
        """
        Clear the event buffer.

        Returns how many events were removed.
        """

        with self._lock:
            count = len(self._events)

            self._events.clear()

            return count

    # ---------------------------------------------------------
    # Length
    # ---------------------------------------------------------

    def __len__(
        self,
    ) -> int:
        with self._lock:
            return len(self._events)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            severities: Dict[
                str,
                int,
            ] = {}

            for event in self._events:
                severities[event.severity] = (
                    severities.get(
                        event.severity,
                        0,
                    )
                    + 1
                )

            return {
                "enabled": (self.config.enabled),
                "minimum_severity": (self.config.minimum_severity),
                "buffered_events": (len(self._events)),
                "maximum_buffered_events": (self.config.maximum_buffered_events),
                "accepted_total": (self._accepted),
                "filtered_total": (self._filtered),
                "evicted_total": (self._evicted),
                "buffered_by_severity": (severities),
            }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "SEVERITY_RANK",
    "EventStoreError",
    "severity_at_least",
    "EventBuffer",
]
