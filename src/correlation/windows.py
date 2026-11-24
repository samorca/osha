from __future__ import annotations

from dataclasses import dataclass
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from src.correlation.schema import (
    TimeWindow,
)

from src.telemetry.schema import (
    TelemetryRecord,
)


DEFAULT_WINDOW_SECONDS = 60.0

DEFAULT_BEFORE_SECONDS = 30.0

DEFAULT_AFTER_SECONDS = 30.0


def parse_timestamp(
    value: Any,
) -> Optional[datetime]:
    """
    Parse an ISO-8601 timestamp into an aware
    datetime.

    Invalid timestamps return None rather than
    terminating correlation.
    """

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)

    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def format_timestamp(
    value: datetime,
) -> str:
    """
    Produce a UTC ISO-8601 timestamp.
    """

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc).isoformat()


def record_timestamp(
    telemetry: TelemetryRecord,
) -> Optional[datetime]:
    return parse_timestamp(telemetry.signal.timestamp)


def record_id(
    telemetry: TelemetryRecord,
) -> Optional[str]:
    telemetry.finalize()

    return telemetry.record_id


def ensure_positive_seconds(
    value: float,
    *,
    name: str,
    allow_zero: bool = False,
) -> float:
    try:
        seconds = float(value)

    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(f"{name} must be numeric")

    minimum = 0.0 if allow_zero else 0.000001

    if seconds < minimum:
        if allow_zero:
            raise ValueError(f"{name} must be >= 0")

        raise ValueError(f"{name} must be > 0")

    return seconds


@dataclass(frozen=True)
class WindowedRecord:
    """
    A telemetry record associated with a
    correlation time window.
    """

    window: TimeWindow

    telemetry: TelemetryRecord

    offset_seconds: float

    def to_dict(self) -> dict:
        return {
            "window": (self.window.to_dict()),
            "record_id": (self.telemetry.record_id),
            "resource_id": (self.telemetry.resource.effective_id()),
            "signal_type": (self.telemetry.signal.signal_type),
            "timestamp": (self.telemetry.signal.timestamp),
            "offset_seconds": (self.offset_seconds),
        }


@dataclass
class RecordWindow:
    """
    Concrete TimeWindow plus the telemetry
    records contained within it.
    """

    window: TimeWindow

    records: List[TelemetryRecord]

    def record_ids(
        self,
    ) -> List[str]:
        output = []

        for telemetry in self.records:
            telemetry.finalize()

            if telemetry.record_id:
                output.append(telemetry.record_id)

        return output

    def resource_ids(
        self,
    ) -> List[str]:
        return sorted({telemetry.resource.effective_id() for telemetry in self.records})

    def providers(
        self,
    ) -> List[str]:
        return sorted({telemetry.resource.provider for telemetry in self.records})

    def signal_types(
        self,
    ) -> List[str]:
        return sorted({telemetry.signal.signal_type for telemetry in self.records})

    def to_dict(self) -> dict:
        return {
            "window": (self.window.to_dict()),
            "record_count": (len(self.records)),
            "record_ids": (self.record_ids()),
            "resource_ids": (self.resource_ids()),
            "providers": (self.providers()),
            "signal_types": (self.signal_types()),
        }


def build_anchor_window(
    anchor: TelemetryRecord,
    *,
    before_seconds: float = (DEFAULT_BEFORE_SECONDS),
    after_seconds: float = (DEFAULT_AFTER_SECONDS),
) -> Optional[TimeWindow]:
    """
    Build a time window centered on one
    telemetry observation.

    Example:

        before=30
        after=30

    gives a 60-second search range around the
    anchor timestamp.
    """

    before = ensure_positive_seconds(
        before_seconds,
        name="before_seconds",
        allow_zero=True,
    )

    after = ensure_positive_seconds(
        after_seconds,
        name="after_seconds",
        allow_zero=True,
    )

    anchor_time = record_timestamp(anchor)

    if anchor_time is None:
        return None

    anchor.finalize()

    start = anchor_time - timedelta(seconds=before)

    end = anchor_time + timedelta(seconds=after)

    return TimeWindow(
        start=format_timestamp(start),
        end=format_timestamp(end),
        environment_id=(anchor.resource.environment_id),
        anchor_record_id=(anchor.record_id),
        anchor_resource_id=(anchor.resource.effective_id()),
    )


def records_in_window(
    records: Iterable[TelemetryRecord],
    window: TimeWindow,
    *,
    same_environment: bool = True,
) -> List[TelemetryRecord]:
    """
    Return telemetry records whose timestamps
    fall within an inclusive TimeWindow.
    """

    start = parse_timestamp(window.start)

    end = parse_timestamp(window.end)

    if start is None or end is None:
        return []

    if end < start:
        return []

    output = []

    for telemetry in records:
        if (
            same_environment
            and telemetry.resource.environment_id != window.environment_id
        ):
            continue

        timestamp = record_timestamp(telemetry)

        if timestamp is None:
            continue

        if start <= timestamp <= end:
            output.append(telemetry)

    output.sort(
        key=lambda item: (
            record_timestamp(item) or datetime.min.replace(tzinfo=timezone.utc)
        )
    )

    return output


def records_around(
    anchor: TelemetryRecord,
    records: Iterable[TelemetryRecord],
    *,
    before_seconds: float = (DEFAULT_BEFORE_SECONDS),
    after_seconds: float = (DEFAULT_AFTER_SECONDS),
    include_anchor: bool = False,
    same_environment: bool = True,
) -> List[TelemetryRecord]:
    """
    Return records near one anchor observation.

    This is candidate selection only.
    """

    window = build_anchor_window(
        anchor,
        before_seconds=(before_seconds),
        after_seconds=(after_seconds),
    )

    if window is None:
        return []

    anchor.finalize()

    output = records_in_window(
        records,
        window,
        same_environment=(same_environment),
    )

    if include_anchor:
        return output

    return [
        telemetry for telemetry in output if telemetry.record_id != anchor.record_id
    ]


def windowed_records_around(
    anchor: TelemetryRecord,
    records: Iterable[TelemetryRecord],
    *,
    before_seconds: float = (DEFAULT_BEFORE_SECONDS),
    after_seconds: float = (DEFAULT_AFTER_SECONDS),
    include_anchor: bool = False,
) -> List[WindowedRecord]:
    """
    Return candidate records plus their signed
    offset from the anchor.

    Negative:
        candidate occurred before anchor.

    Positive:
        candidate occurred after anchor.
    """

    window = build_anchor_window(
        anchor,
        before_seconds=(before_seconds),
        after_seconds=(after_seconds),
    )

    if window is None:
        return []

    anchor_time = record_timestamp(anchor)

    if anchor_time is None:
        return []

    nearby = records_around(
        anchor,
        records,
        before_seconds=(before_seconds),
        after_seconds=(after_seconds),
        include_anchor=(include_anchor),
    )

    output = []

    for telemetry in nearby:
        timestamp = record_timestamp(telemetry)

        if timestamp is None:
            continue

        offset = (timestamp - anchor_time).total_seconds()

        output.append(
            WindowedRecord(
                window=window,
                telemetry=telemetry,
                offset_seconds=(offset),
            )
        )

    return output


def fixed_windows(
    records: Iterable[TelemetryRecord],
    *,
    window_seconds: float = (DEFAULT_WINDOW_SECONDS),
    environment_id: Optional[str] = None,
) -> List[RecordWindow]:
    """
    Partition observations into non-overlapping
    fixed windows.

    Each environment is partitioned separately.

    The first valid timestamp in each environment
    starts that environment's first window.
    """

    size = ensure_positive_seconds(
        window_seconds,
        name="window_seconds",
    )

    by_environment: Dict[
        str,
        List[TelemetryRecord],
    ] = {}

    for telemetry in records:
        if (
            environment_id is not None
            and telemetry.resource.environment_id != environment_id
        ):
            continue

        if record_timestamp(telemetry) is None:
            continue

        environment = telemetry.resource.environment_id

        by_environment.setdefault(
            environment,
            [],
        ).append(telemetry)

    windows: List[RecordWindow] = []

    for (
        environment,
        environment_records,
    ) in by_environment.items():
        environment_records.sort(
            key=lambda item: (
                record_timestamp(item) or datetime.min.replace(tzinfo=timezone.utc)
            )
        )

        if not environment_records:
            continue

        first_time = record_timestamp(environment_records[0])

        if first_time is None:
            continue

        buckets: Dict[
            int,
            List[TelemetryRecord],
        ] = {}

        for telemetry in environment_records:
            timestamp = record_timestamp(telemetry)

            if timestamp is None:
                continue

            offset = (timestamp - first_time).total_seconds()

            bucket = int(offset // size)

            buckets.setdefault(
                bucket,
                [],
            ).append(telemetry)

        for bucket in sorted(buckets):
            start = first_time + timedelta(seconds=(bucket * size))

            end = start + timedelta(seconds=size)

            window = TimeWindow(
                start=(format_timestamp(start)),
                end=(format_timestamp(end)),
                window_seconds=size,
                environment_id=(environment),
            )

            windows.append(
                RecordWindow(
                    window=window,
                    records=(buckets[bucket]),
                )
            )

    windows.sort(
        key=lambda item: (
            parse_timestamp(item.window.start)
            or datetime.min.replace(tzinfo=timezone.utc)
        )
    )

    return windows


def sliding_windows(
    records: Iterable[TelemetryRecord],
    *,
    window_seconds: float = 60.0,
    step_seconds: float = 30.0,
    environment_id: Optional[str] = None,
) -> List[RecordWindow]:
    """
    Generate overlapping time windows.

    Sliding windows can later support incident
    candidate discovery where an event may occur
    near the boundary of a fixed bucket.
    """

    size = ensure_positive_seconds(
        window_seconds,
        name="window_seconds",
    )

    step = ensure_positive_seconds(
        step_seconds,
        name="step_seconds",
    )

    if step > size:
        raise ValueError("step_seconds should not exceed window_seconds")

    by_environment: Dict[
        str,
        List[TelemetryRecord],
    ] = {}

    for telemetry in records:
        if (
            environment_id is not None
            and telemetry.resource.environment_id != environment_id
        ):
            continue

        timestamp = record_timestamp(telemetry)

        if timestamp is None:
            continue

        environment = telemetry.resource.environment_id

        by_environment.setdefault(
            environment,
            [],
        ).append(telemetry)

    output: List[RecordWindow] = []

    for (
        environment,
        environment_records,
    ) in by_environment.items():
        environment_records.sort(
            key=lambda item: (
                record_timestamp(item) or datetime.min.replace(tzinfo=timezone.utc)
            )
        )

        if not environment_records:
            continue

        start_time = record_timestamp(environment_records[0])

        final_time = record_timestamp(environment_records[-1])

        if start_time is None or final_time is None:
            continue

        cursor = start_time

        while cursor <= final_time:
            end = cursor + timedelta(seconds=size)

            window = TimeWindow(
                start=(format_timestamp(cursor)),
                end=(format_timestamp(end)),
                window_seconds=size,
                environment_id=(environment),
            )

            members = records_in_window(
                environment_records,
                window,
            )

            if members:
                output.append(
                    RecordWindow(
                        window=window,
                        records=members,
                    )
                )

            cursor = cursor + timedelta(seconds=step)

    output.sort(
        key=lambda item: (
            parse_timestamp(item.window.start)
            or datetime.min.replace(tzinfo=timezone.utc)
        )
    )

    return output


def pair_candidates(
    record_window: RecordWindow,
    *,
    same_record: bool = False,
    cross_provider_only: bool = False,
    maximum_pairs: int = 100_000,
) -> Iterator[
    Tuple[
        TelemetryRecord,
        TelemetryRecord,
    ]
]:
    """
    Generate unique observation pairs inside one
    time window.

    This does not claim correlation. It only
    limits the search space for correlate.py.
    """

    if maximum_pairs < 1:
        raise ValueError("maximum_pairs must be >= 1")

    records = list(record_window.records)

    emitted = 0

    for left_index in range(len(records)):
        right_start = left_index if same_record else left_index + 1

        for right_index in range(
            right_start,
            len(records),
        ):
            left = records[left_index]

            right = records[right_index]

            if not same_record and left is right:
                continue

            if (
                cross_provider_only
                and left.resource.provider == right.resource.provider
            ):
                continue

            yield (
                left,
                right,
            )

            emitted += 1

            if emitted >= maximum_pairs:
                return


def window_summary(
    windows: Sequence[RecordWindow],
) -> Dict[str, Any]:
    total_records = 0

    environments = set()

    providers = set()

    signal_types = set()

    largest_window = 0

    for item in windows:
        count = len(item.records)

        total_records += count

        largest_window = max(
            largest_window,
            count,
        )

        environments.add(item.window.environment_id)

        providers.update(item.providers())

        signal_types.update(item.signal_types())

    return {
        "window_count": (len(windows)),
        "record_memberships": (total_records),
        "environment_count": (len(environments)),
        "providers": sorted(providers),
        "signal_types": sorted(signal_types),
        "largest_window_records": (largest_window),
    }


class WindowManager:
    """
    Convenience API for the future running
    correlation pipeline.
    """

    def __init__(
        self,
        *,
        before_seconds: float = (DEFAULT_BEFORE_SECONDS),
        after_seconds: float = (DEFAULT_AFTER_SECONDS),
        fixed_window_seconds: float = (DEFAULT_WINDOW_SECONDS),
    ) -> None:
        self.before_seconds = ensure_positive_seconds(
            before_seconds,
            name="before_seconds",
            allow_zero=True,
        )

        self.after_seconds = ensure_positive_seconds(
            after_seconds,
            name="after_seconds",
            allow_zero=True,
        )

        self.fixed_window_seconds = ensure_positive_seconds(
            fixed_window_seconds,
            name=("fixed_window_seconds"),
        )

    def around(
        self,
        anchor: TelemetryRecord,
        records: Iterable[TelemetryRecord],
        *,
        include_anchor: bool = False,
    ) -> List[TelemetryRecord]:
        return records_around(
            anchor,
            records,
            before_seconds=(self.before_seconds),
            after_seconds=(self.after_seconds),
            include_anchor=(include_anchor),
        )

    def fixed(
        self,
        records: Iterable[TelemetryRecord],
    ) -> List[RecordWindow]:
        return fixed_windows(
            records,
            window_seconds=(self.fixed_window_seconds),
        )
