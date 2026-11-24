from __future__ import annotations

"""
Bounded in-memory metrics for OSHA application
components.
"""

from collections import (
    defaultdict,
    deque,
)

from threading import RLock

from typing import (
    Any,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

from src.core.validation import (
    integer,
    number,
    required_string,
)

from src.observability.config import (
    MetricsConfig,
)

from src.observability.schema import (
    MetricSample,
    MetricType,
    ObservabilityContext,
)


# ---------------------------------------------------------------
# Errors
# ---------------------------------------------------------------


class MetricsError(RuntimeError):
    pass


# ---------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------


def normalize_labels(
    labels: Optional[
        Mapping[
            str,
            Any,
        ]
    ],
) -> Dict[str, str]:
    """
    Normalize metric labels to deterministic
    string key/value pairs.

    Keep label usage bounded. Avoid placing
    unique job IDs, raw error messages, tokens or
    other high-cardinality/sensitive values in
    metric labels.
    """

    if labels is None:
        return {}

    normalized: Dict[
        str,
        str,
    ] = {}

    for key, value in labels.items():
        name = required_string(
            key,
            name=("metric label name"),
            maximum_length=128,
        )

        label_value = required_string(
            value,
            name=(f"metric label {name}"),
            maximum_length=512,
        )

        normalized[name] = label_value

    return dict(sorted(normalized.items()))


# ---------------------------------------------------------------
# Metric identity key
# ---------------------------------------------------------------


def metric_key(
    name: str,
    labels: Mapping[
        str,
        str,
    ],
) -> Tuple[
    str,
    Tuple[
        Tuple[
            str,
            str,
        ],
        ...,
    ],
]:
    """
    Construct an in-memory identity for one
    metric series.
    """

    metric_name = required_string(
        name,
        name="metric name",
        maximum_length=256,
    )

    return (
        metric_name,
        tuple(sorted(labels.items())),
    )


# ---------------------------------------------------------------
# Metrics registry
# ---------------------------------------------------------------


class MetricsRegistry:
    """
    Thread-safe bounded in-process metrics
    registry.

    Maintains:

      current counter values
      current gauge values
      histogram aggregates
      bounded raw MetricSample buffer

    External metrics export belongs in the
    observability pipeline/integration layer.
    """

    def __init__(
        self,
        *,
        config: Optional[MetricsConfig] = None,
    ) -> None:
        self.config = config or MetricsConfig()

        self.config.validate()

        self._samples: Deque[MetricSample] = deque(
            maxlen=(self.config.maximum_buffered_samples)
        )

        self._counters: Dict[
            Tuple[
                str,
                Tuple[
                    Tuple[
                        str,
                        str,
                    ],
                    ...,
                ],
            ],
            float,
        ] = defaultdict(float)

        self._gauges: Dict[
            Tuple[
                str,
                Tuple[
                    Tuple[
                        str,
                        str,
                    ],
                    ...,
                ],
            ],
            float,
        ] = {}

        self._histograms: Dict[
            Tuple[
                str,
                Tuple[
                    Tuple[
                        str,
                        str,
                    ],
                    ...,
                ],
            ],
            Dict[
                str,
                float,
            ],
        ] = {}

        self._lock = RLock()

        self._accepted = 0
        self._evicted = 0

    # ---------------------------------------------------------
    # Internal sample recording
    # ---------------------------------------------------------

    def _record(
        self,
        sample: MetricSample,
    ) -> Optional[MetricSample]:
        sample.finalize()

        if not self.config.enabled:
            return None

        with self._lock:
            was_full = len(self._samples) == self._samples.maxlen

            self._samples.append(sample)

            self._accepted += 1

            if was_full:
                self._evicted += 1

            return sample

    # ---------------------------------------------------------
    # Counter
    # ---------------------------------------------------------

    def increment(
        self,
        *,
        name: str,
        context: ObservabilityContext,
        amount: float = 1.0,
        labels: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        unit: Optional[str] = None,
    ) -> Optional[MetricSample]:
        """
        Increment a monotonic-style counter.

        Negative increments are rejected.
        """

        if not self.config.enabled:
            return None

        amount = number(
            amount,
            name="counter amount",
            minimum=0.0,
        )

        normalized = normalize_labels(labels)

        key = metric_key(
            name,
            normalized,
        )

        with self._lock:
            self._counters[key] += amount

            value = self._counters[key]

        return self._record(
            MetricSample(
                name=name,
                value=value,
                context=context,
                metric_type=(MetricType.COUNTER.value),
                unit=unit,
                labels=normalized,
            )
        )

    # ---------------------------------------------------------
    # Gauge
    # ---------------------------------------------------------

    def gauge(
        self,
        *,
        name: str,
        value: float,
        context: ObservabilityContext,
        labels: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        unit: Optional[str] = None,
    ) -> Optional[MetricSample]:
        """
        Set a gauge to its latest observed value.
        """

        if not self.config.enabled:
            return None

        value = number(
            value,
            name="gauge value",
        )

        normalized = normalize_labels(labels)

        key = metric_key(
            name,
            normalized,
        )

        with self._lock:
            self._gauges[key] = value

        return self._record(
            MetricSample(
                name=name,
                value=value,
                context=context,
                metric_type=(MetricType.GAUGE.value),
                unit=unit,
                labels=normalized,
            )
        )

    # ---------------------------------------------------------
    # Histogram observation
    # ---------------------------------------------------------

    def observe(
        self,
        *,
        name: str,
        value: float,
        context: ObservabilityContext,
        labels: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
        unit: Optional[str] = None,
    ) -> Optional[MetricSample]:
        """
        Record one histogram-style observation.

        This development implementation stores:

          count
          sum
          min
          max

        It does not calculate production histogram
        buckets or percentiles.
        """

        if not self.config.enabled:
            return None

        value = number(
            value,
            name=("histogram observation"),
        )

        normalized = normalize_labels(labels)

        key = metric_key(
            name,
            normalized,
        )

        with self._lock:
            aggregate = self._histograms.setdefault(
                key,
                {
                    "count": 0.0,
                    "sum": 0.0,
                    "min": value,
                    "max": value,
                },
            )

            aggregate["count"] += 1.0

            aggregate["sum"] += value

            aggregate["min"] = min(
                aggregate["min"],
                value,
            )

            aggregate["max"] = max(
                aggregate["max"],
                value,
            )

        return self._record(
            MetricSample(
                name=name,
                value=value,
                context=context,
                metric_type=(MetricType.HISTOGRAM.value),
                unit=unit,
                labels=normalized,
            )
        )

    # ---------------------------------------------------------
    # Query samples
    # ---------------------------------------------------------

    def samples(
        self,
        *,
        name: Optional[str] = None,
        metric_type: Optional[str] = None,
        service: Optional[str] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[MetricSample]:
        """
        Query buffered samples without removing
        them.
        """

        if limit is not None:
            limit = integer(
                limit,
                name="limit",
                minimum=1,
            )

        if metric_type is not None and metric_type not in {
            item.value for item in MetricType
        }:
            raise MetricsError(f"Unknown metric type: {metric_type}")

        with self._lock:
            values = list(self._samples)

        if newest_first:
            values.reverse()

        output: List[MetricSample] = []

        for sample in values:
            if name is not None and sample.name != name:
                continue

            if metric_type is not None and sample.metric_type != metric_type:
                continue

            if service is not None and sample.context.service != service:
                continue

            output.append(sample)

            if limit is not None and len(output) >= limit:
                break

        return output

    # ---------------------------------------------------------
    # Drain samples
    # ---------------------------------------------------------

    def drain(
        self,
        *,
        limit: Optional[int] = None,
    ) -> List[MetricSample]:
        """
        Remove and return oldest samples.

        Registry aggregate state remains intact.
        """

        if limit is not None:
            limit = integer(
                limit,
                name="limit",
                minimum=1,
            )

        with self._lock:
            count = (
                len(self._samples)
                if limit is None
                else min(
                    limit,
                    len(self._samples),
                )
            )

            return [self._samples.popleft() for _ in range(count)]

    # ---------------------------------------------------------
    # Counter value
    # ---------------------------------------------------------

    def counter_value(
        self,
        name: str,
        *,
        labels: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> float:
        key = metric_key(
            name,
            normalize_labels(labels),
        )

        with self._lock:
            return self._counters.get(
                key,
                0.0,
            )

    # ---------------------------------------------------------
    # Gauge value
    # ---------------------------------------------------------

    def gauge_value(
        self,
        name: str,
        *,
        labels: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> Optional[float]:
        key = metric_key(
            name,
            normalize_labels(labels),
        )

        with self._lock:
            return self._gauges.get(key)

    # ---------------------------------------------------------
    # Histogram summary
    # ---------------------------------------------------------

    def histogram_summary(
        self,
        name: str,
        *,
        labels: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> Optional[
        Dict[
            str,
            float,
        ]
    ]:
        key = metric_key(
            name,
            normalize_labels(labels),
        )

        with self._lock:
            aggregate = self._histograms.get(key)

            if aggregate is None:
                return None

            count = aggregate["count"]

            return {
                **dict(aggregate),
                "average": (aggregate["sum"] / count if count else 0.0),
            }

    # ---------------------------------------------------------
    # Clear buffered samples
    # ---------------------------------------------------------

    def clear_samples(
        self,
    ) -> int:
        """
        Clear raw buffered samples while
        preserving current metric aggregates.
        """

        with self._lock:
            count = len(self._samples)

            self._samples.clear()

            return count

    # ---------------------------------------------------------
    # Full reset
    # ---------------------------------------------------------

    def reset(
        self,
    ) -> None:
        """
        Reset buffered samples and all current
        aggregate values.
        """

        with self._lock:
            self._samples.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            by_type: Dict[
                str,
                int,
            ] = {}

            for sample in self._samples:
                by_type[sample.metric_type] = (
                    by_type.get(
                        sample.metric_type,
                        0,
                    )
                    + 1
                )

            return {
                "enabled": (self.config.enabled),
                "buffered_samples": (len(self._samples)),
                "maximum_buffered_samples": (self.config.maximum_buffered_samples),
                "accepted_total": (self._accepted),
                "evicted_total": (self._evicted),
                "registered_counters": (len(self._counters)),
                "registered_gauges": (len(self._gauges)),
                "registered_histograms": (len(self._histograms)),
                "buffered_by_type": (by_type),
            }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "MetricsError",
    "normalize_labels",
    "metric_key",
    "MetricsRegistry",
]
