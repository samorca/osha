from __future__ import annotations

"""
Composition facade for OSHA observability
signals.
"""

from dataclasses import (
    dataclass,
    field,
)

from typing import (
    Any,
    Callable,
    Dict,
    Optional,
)

from src.observability.config import (
    ObservabilityConfig,
)

from src.observability.events import (
    EventBuffer,
)

from src.observability.health import (
    HealthRegistry,
    HealthReport,
)

from src.observability.logging import (
    LogBuffer,
)

from src.observability.metrics import (
    MetricsRegistry,
)

from src.observability.schema import (
    ObservabilityBatch,
)

from src.observability.tracing import (
    TraceBuffer,
)


# ---------------------------------------------------------------
# Exporter contract
# ---------------------------------------------------------------

Exporter = Callable[
    [
        ObservabilityBatch
    ],
    Any,
]


# ---------------------------------------------------------------
# Export result
# ---------------------------------------------------------------

@dataclass
class ExportResult:
    """
    Bounded result from one observability export
    operation.

    Arbitrary exporter exception messages are
    not copied into the result.
    """

    attempted: bool = False

    success: bool = False

    exported_count: int = 0

    error_type: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "attempted": (
                self.attempted
            ),
            "success": (
                self.success
            ),
            "exported_count": (
                self.exported_count
            ),
            "error_type": (
                self.error_type
            ),
            "attributes": dict(
                self.attributes
            ),
        }


# ---------------------------------------------------------------
# Observability pipeline
# ---------------------------------------------------------------

class ObservabilityPipeline:
    """
    Unified in-process facade for OSHA
    observability.

    Provides access to:

      events
      logs
      metrics
      traces
      health

    Export is vendor-neutral and requires an
    explicitly injected exporter.
    """

    def __init__(
        self,
        *,
        config: Optional[
            ObservabilityConfig
        ] = None,
        exporter: Optional[
            Exporter
        ] = None,
    ) -> None:
        self.config = (
            config
            or ObservabilityConfig()
        )

        self.config.validate()

        self.events = EventBuffer(
            config=(
                self.config.events
            )
        )

        self.metrics = MetricsRegistry(
            config=(
                self.config.metrics
            )
        )

        self.logs = LogBuffer(
            config=(
                self.config.logging
            ),
            redact_sensitive=(
                self.config.
                redact_sensitive
            ),
        )

        self.traces = TraceBuffer(
            config=(
                self.config.tracing
            ),
            redact_sensitive=(
                self.config.
                redact_sensitive
            ),
        )

        self.health = HealthRegistry(
            config=(
                self.config.health
            ),
            redact_sensitive=(
                self.config.
                redact_sensitive
            ),
        )

        self.exporter = exporter

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------

    def health_check(
        self,
    ) -> HealthReport:
        return (
            self.health.check_all()
        )

    # ---------------------------------------------------------
    # Non-destructive snapshot
    # ---------------------------------------------------------

    def snapshot(
        self,
    ) -> ObservabilityBatch:
        """
        Return currently buffered signals without
        removing them.
        """

        return ObservabilityBatch(
            events=(
                self.events.list()
            ),
            logs=(
                self.logs.list()
            ),
            metrics=(
                self.metrics.samples()
            ),
            spans=(
                self.traces.spans()
            ),
            health=(
                self.health.
                observations()
            ),
            attributes={
                "scope": (
                    "osha_application_"
                    "observability"
                ),
                "destructive": False,
            },
        ).finalize()

    # ---------------------------------------------------------
    # Destructive drain
    # ---------------------------------------------------------

    def drain(
        self,
        *,
        limit_per_signal: Optional[
            int
        ] = None,
    ) -> ObservabilityBatch:
        """
        Remove a batch of currently buffered
        signals.

        This is useful when the caller has already
        arranged reliable persistence/export.

        flush() is usually safer because it only
        removes buffered data after successful
        export.
        """

        if (
            limit_per_signal
            is not None
            and limit_per_signal < 1
        ):
            raise ValueError(
                "limit_per_signal "
                "must be >= 1"
            )

        events = self.events.drain(
            limit=limit_per_signal
        )

        logs = self.logs.drain(
            limit=limit_per_signal
        )

        metrics = (
            self.metrics.drain(
                limit=limit_per_signal
            )
        )

        spans = self.traces.drain(
            limit=limit_per_signal
        )

        health = (
            self.health.drain()
        )

        # health.py currently exposes a full drain
        # rather than a bounded drain. Avoid
        # silently pretending its semantics match
        # the other buffers.
        if (
            limit_per_signal
            is not None
            and len(health)
            > limit_per_signal
        ):
            health = health[
                :limit_per_signal
            ]

        return ObservabilityBatch(
            events=events,
            logs=logs,
            metrics=metrics,
            spans=spans,
            health=health,
            attributes={
                "scope": (
                    "osha_application_"
                    "observability"
                ),
                "destructive": True,
            },
        ).finalize()

    # ---------------------------------------------------------
    # Signal count
    # ---------------------------------------------------------

    @staticmethod
    def signal_count(
        batch: ObservabilityBatch,
    ) -> int:
        batch.finalize()

        return (
            len(
                batch.events
            )
            + len(
                batch.logs
            )
            + len(
                batch.metrics
            )
            + len(
                batch.spans
            )
            + len(
                batch.health
            )
        )

    # ---------------------------------------------------------
    # Export an explicit batch
    # ---------------------------------------------------------

    def export(
        self,
        batch: Optional[
            ObservabilityBatch
        ] = None,
    ) -> ExportResult:
        """
        Export through the explicitly injected
        exporter.

        This method does not remove buffered
        signals.
        """

        if not (
            self.config.
            export.enabled
        ):
            return ExportResult(
                attempted=False,
                success=False,
                attributes={
                    "reason": (
                        "export_disabled"
                    ),
                },
            )

        if self.exporter is None:
            return ExportResult(
                attempted=False,
                success=False,
                attributes={
                    "reason": (
                        "exporter_not_registered"
                    ),
                },
            )

        value = (
            batch
            or self.snapshot()
        )

        count = self.signal_count(
            value
        )

        try:
            self.exporter(
                value
            )

            return ExportResult(
                attempted=True,
                success=True,
                exported_count=count,
            )

        except Exception as exc:
            return ExportResult(
                attempted=True,
                success=False,
                exported_count=0,
                error_type=(
                    type(exc).__name__
                ),
                attributes={
                    "reason": (
                        "export_failed"
                    ),
                },
            )

    # ---------------------------------------------------------
    # Flush safely
    # ---------------------------------------------------------

    def flush(
        self,
    ) -> ExportResult:
        """
        Snapshot -> export -> drain on success.

        Buffered signals are retained when export
        fails.
        """

        if not (
            self.config.
            export.enabled
        ):
            return ExportResult(
                attempted=False,
                success=False,
                attributes={
                    "reason": (
                        "export_disabled"
                    ),
                },
            )

        if self.exporter is None:
            return ExportResult(
                attempted=False,
                success=False,
                attributes={
                    "reason": (
                        "exporter_not_registered"
                    ),
                },
            )

        batch = self.snapshot()

        result = self.export(
            batch
        )

        if result.success:
            self.events.drain()
            self.logs.drain()
            self.metrics.drain()
            self.traces.drain()
            self.health.drain()

        return result

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "events": (
                self.events.summary()
            ),

            "logs": (
                self.logs.summary()
            ),

            "metrics": (
                self.metrics.summary()
            ),

            "traces": (
                self.traces.summary()
            ),

            "health": (
                self.health.summary()
            ),

            "export": {
                "enabled": (
                    self.config.
                    export.enabled
                ),

                "exporter": (
                    self.config.
                    export.exporter
                ),

                "registered": (
                    self.exporter
                    is not None
                ),
            },

            "scope": (
                "osha_application_"
                "observability"
            ),

            "external_infrastructure_health_claimed": (
                False
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "Exporter",

    "ExportResult",

    "ObservabilityPipeline",
]
