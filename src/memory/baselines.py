from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

from src.memory.schema import (
    BaselineMemory,
    VerificationStatus,
    checked_score,
)

from src.memory.store import (
    MemoryQuery,
    MemoryStore,
)

from src.telemetry.schema import (
    SignalType,
    TelemetryRecord,
)


DEFAULT_MINIMUM_SAMPLES = 10

DEFAULT_ANOMALY_Z_SCORE = 3.0


@dataclass(frozen=True)
class BaselineConfig:
    """
    Configuration for incremental metric
    baselines.

    minimum_samples:
        Number of samples required before a
        baseline should be used for anomaly
        assessment.

    anomaly_z_score:
        Absolute z-score at which the current
        sample becomes an anomaly candidate.

    use_sample_variance:
        Use n - 1 as the variance denominator
        when True.
    """

    minimum_samples: int = DEFAULT_MINIMUM_SAMPLES

    anomaly_z_score: float = DEFAULT_ANOMALY_Z_SCORE

    use_sample_variance: bool = True

    require_metric_signal: bool = True

    def validate(
        self,
    ) -> None:
        if self.minimum_samples < 2:
            raise ValueError("minimum_samples must be >= 2")

        if not math.isfinite(self.anomaly_z_score) or self.anomaly_z_score <= 0:
            raise ValueError("anomaly_z_score must be finite and > 0")


@dataclass(frozen=True)
class BaselineAssessment:
    """
    Assessment of one numeric value against an
    existing baseline.

    `anomalous` means the configured statistical
    threshold was exceeded. It does not by itself
    prove an operational incident.
    """

    baseline_memory_id: Optional[str]

    ready: bool

    value: Optional[float]

    sample_count: int

    mean: Optional[float]

    standard_deviation: Optional[float]

    z_score: Optional[float]

    anomaly_score: float

    anomalous: bool

    reason: str

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "baseline_memory_id": (self.baseline_memory_id),
            "ready": self.ready,
            "value": self.value,
            "sample_count": (self.sample_count),
            "mean": self.mean,
            "standard_deviation": (self.standard_deviation),
            "z_score": self.z_score,
            "anomaly_score": (self.anomaly_score),
            "anomalous": (self.anomalous),
            "reason": self.reason,
        }


def optional_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def parse_timestamp(
    value: Any,
) -> Optional[datetime]:
    text = optional_string(value)

    if text is None:
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


def earliest_timestamp(
    left: Optional[str],
    right: Optional[str],
) -> Optional[str]:
    left_time = parse_timestamp(left)

    right_time = parse_timestamp(right)

    if left_time is None:
        return right

    if right_time is None:
        return left

    return left if left_time <= right_time else right


def latest_timestamp(
    left: Optional[str],
    right: Optional[str],
) -> Optional[str]:
    left_time = parse_timestamp(left)

    right_time = parse_timestamp(right)

    if left_time is None:
        return right

    if right_time is None:
        return left

    return left if left_time >= right_time else right


def finite_number(
    value: Any,
) -> Optional[float]:
    """
    Convert a numeric telemetry value into a
    finite float.

    Booleans are excluded because True/False are
    states rather than numeric measurements.
    """

    if isinstance(
        value,
        bool,
    ):
        return None

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(number):
        return None

    return number


def metric_value(
    telemetry: TelemetryRecord,
) -> Optional[float]:
    """
    Extract the numeric value from a canonical
    metric signal.
    """

    direct = finite_number(telemetry.signal.value)

    if direct is not None:
        return direct

    attributes = telemetry.signal.attributes or {}

    for key in (
        "value",
        "metric_value",
        "measurement",
    ):
        if key not in attributes:
            continue

        value = finite_number(attributes.get(key))

        if value is not None:
            return value

    return None


def metric_name(
    telemetry: TelemetryRecord,
) -> Optional[str]:
    name = optional_string(telemetry.signal.signal_name)

    if name:
        return name

    attributes = telemetry.signal.attributes or {}

    return optional_string(attributes.get("metric_name"))


def baseline_key(
    *,
    environment_id: str,
    resource_id: str,
    signal_name: str,
    unit: Optional[str],
) -> str:
    """
    Build a stable, human-readable baseline key.

    Resource and environment remain part of the
    baseline identity so measurements from
    unrelated systems are not mixed.
    """

    unit_component = unit if unit else "unitless"

    return f"{environment_id}::{resource_id}::{signal_name}::{unit_component}"


def baseline_key_for_record(
    telemetry: TelemetryRecord,
) -> Optional[str]:
    name = metric_name(telemetry)

    if name is None:
        return None

    return baseline_key(
        environment_id=(telemetry.resource.environment_id),
        resource_id=(telemetry.resource.effective_id()),
        signal_name=name,
        unit=telemetry.signal.unit,
    )


def baseline_m2(
    baseline: BaselineMemory,
) -> float:
    """
    Retrieve Welford's accumulated M2 statistic.

    M2 is kept under attributes because the
    initial BaselineMemory schema exposes the
    derived variance but not a dedicated M2
    field.
    """

    value = finite_number(baseline.attributes.get("m2"))

    return value if value is not None else 0.0


def calculate_variance(
    *,
    sample_count: int,
    m2: float,
    sample_variance: bool,
) -> Optional[float]:
    if sample_count <= 0:
        return None

    if sample_variance:
        if sample_count < 2:
            return None

        denominator = sample_count - 1

    else:
        denominator = sample_count

    variance = m2 / denominator

    # Small negative values may occur because of
    # floating-point rounding.
    return max(
        0.0,
        variance,
    )


def standard_deviation(
    variance: Optional[float],
) -> Optional[float]:
    if variance is None:
        return None

    if variance < 0:
        return None

    return math.sqrt(variance)


def empty_baseline(
    telemetry: TelemetryRecord,
    *,
    config: BaselineConfig,
) -> Optional[BaselineMemory]:
    """
    Create an empty baseline contract for one
    metric stream.
    """

    name = metric_name(telemetry)

    key = baseline_key_for_record(telemetry)

    if name is None or key is None:
        return None

    telemetry.finalize()

    memory = BaselineMemory(
        environment_id=(telemetry.resource.environment_id),
        baseline_key=key,
        resource_id=(telemetry.resource.effective_id()),
        signal_name=name,
        metric_unit=(telemetry.signal.unit),
        sample_count=0,
        model_type=(
            "welford_sample" if config.use_sample_variance else "welford_population"
        ),
        resource_ids=[telemetry.resource.effective_id()],
        source_ids=([telemetry.record_id] if telemetry.record_id else []),
        tags=[
            "metric_baseline",
            "online_statistics",
        ],
        verification_status=(VerificationStatus.UNVERIFIED.value),
        attributes={
            "m2": 0.0,
            "minimum_samples": (config.minimum_samples),
            "anomaly_z_score": (config.anomaly_z_score),
            "sample_variance": (config.use_sample_variance),
        },
    )

    return memory.finalize()


def update_baseline_value(
    baseline: BaselineMemory,
    value: float,
    *,
    timestamp: Optional[str] = None,
    source_id: Optional[str] = None,
    config: Optional[BaselineConfig] = None,
) -> BaselineMemory:
    """
    Update one baseline using Welford's online
    mean and variance recurrence.

    This function mutates and returns `baseline`.
    """

    configuration = config or BaselineConfig(
        use_sample_variance=(
            baseline.attributes.get(
                "sample_variance",
                True,
            )
            is not False
        )
    )

    configuration.validate()

    number = finite_number(value)

    if number is None:
        raise ValueError("Baseline value must be a finite number")

    old_count = max(
        0,
        int(baseline.sample_count),
    )

    old_mean = float(baseline.mean) if baseline.mean is not None else 0.0

    old_m2 = baseline_m2(baseline)

    new_count = old_count + 1

    delta = number - old_mean

    new_mean = old_mean + delta / new_count

    delta2 = number - new_mean

    new_m2 = old_m2 + delta * delta2

    baseline.sample_count = new_count

    baseline.mean = new_mean

    baseline.minimum = (
        number
        if baseline.minimum is None
        else min(
            float(baseline.minimum),
            number,
        )
    )

    baseline.maximum = (
        number
        if baseline.maximum is None
        else max(
            float(baseline.maximum),
            number,
        )
    )

    variance = calculate_variance(
        sample_count=(new_count),
        m2=new_m2,
        sample_variance=(configuration.use_sample_variance),
    )

    baseline.variance = variance

    baseline.standard_deviation = standard_deviation(variance)

    baseline.attributes["m2"] = new_m2

    baseline.attributes["minimum_samples"] = configuration.minimum_samples

    baseline.attributes["anomaly_z_score"] = configuration.anomaly_z_score

    baseline.attributes["sample_variance"] = configuration.use_sample_variance

    baseline.first_sample_at = earliest_timestamp(
        baseline.first_sample_at,
        timestamp,
    )

    baseline.last_sample_at = latest_timestamp(
        baseline.last_sample_at,
        timestamp,
    )

    if source_id:
        baseline.source_ids = sorted(set(baseline.source_ids) | {source_id})

    baseline.touch()

    return baseline.finalize()


def update_baseline_from_record(
    baseline: BaselineMemory,
    telemetry: TelemetryRecord,
    *,
    config: Optional[BaselineConfig] = None,
) -> BaselineMemory:
    value = metric_value(telemetry)

    if value is None:
        raise ValueError("Telemetry record does not contain a numeric metric value")

    telemetry.finalize()

    return update_baseline_value(
        baseline,
        value,
        timestamp=(telemetry.signal.timestamp),
        source_id=(telemetry.record_id),
        config=config,
    )


def find_baseline(
    store: MemoryStore,
    *,
    environment_id: str,
    baseline_key_value: str,
) -> Optional[BaselineMemory]:
    query = MemoryQuery(
        memory_types=("baseline",),
        newest_first=True,
    )

    for memory in store.query(query):
        if (
            isinstance(
                memory,
                BaselineMemory,
            )
            and memory.baseline_key == baseline_key_value
        ):
            return memory

    return None


def remember_metric(
    store: MemoryStore,
    telemetry: TelemetryRecord,
    *,
    config: Optional[BaselineConfig] = None,
) -> Optional[BaselineMemory]:
    """
    Add one canonical metric observation to its
    persistent baseline.

    Non-metric or non-numeric records return
    None.
    """

    configuration = config or BaselineConfig()

    configuration.validate()

    if (
        configuration.require_metric_signal
        and telemetry.signal.signal_type != SignalType.METRIC.value
    ):
        return None

    value = metric_value(telemetry)

    key = baseline_key_for_record(telemetry)

    if value is None or key is None:
        return None

    baseline = find_baseline(
        store,
        environment_id=(telemetry.resource.environment_id),
        baseline_key_value=key,
    )

    if baseline is None:
        baseline = empty_baseline(
            telemetry,
            config=configuration,
        )

    if baseline is None:
        return None

    baseline = update_baseline_from_record(
        baseline,
        telemetry,
        config=configuration,
    )

    stored = store.put(baseline)

    if not isinstance(
        stored,
        BaselineMemory,
    ):
        raise TypeError("Memory store returned a non-BaselineMemory object")

    return stored


def remember_metrics(
    store: MemoryStore,
    records: Iterable[TelemetryRecord],
    *,
    config: Optional[BaselineConfig] = None,
) -> List[BaselineMemory]:
    """
    Incrementally update baselines for a telemetry
    batch.

    The returned list contains the final state of
    each baseline touched by the batch.
    """

    configuration = config or BaselineConfig()

    configuration.validate()

    touched: Dict[
        str,
        BaselineMemory,
    ] = {}

    for telemetry in records:
        baseline = remember_metric(
            store,
            telemetry,
            config=configuration,
        )

        if baseline is None or not baseline.memory_id:
            continue

        touched[baseline.memory_id] = baseline

    return list(touched.values())


# ---------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------


def baseline_ready(
    baseline: BaselineMemory,
    *,
    config: Optional[BaselineConfig] = None,
) -> bool:
    configuration = config or BaselineConfig()

    configuration.validate()

    return (
        baseline.sample_count >= configuration.minimum_samples
        and baseline.mean is not None
        and baseline.standard_deviation is not None
    )


def z_score(
    value: float,
    baseline: BaselineMemory,
) -> Optional[float]:
    number = finite_number(value)

    if number is None or baseline.mean is None or baseline.standard_deviation is None:
        return None

    deviation = float(baseline.standard_deviation)

    if deviation <= 0:
        if number == float(baseline.mean):
            return 0.0

        # A zero-variance baseline followed by a
        # different value is maximally different,
        # but there is no finite z-score.
        return math.inf

    return (number - float(baseline.mean)) / deviation


def anomaly_score_from_z(
    z_score_value: Optional[float],
    *,
    threshold: float,
) -> float:
    if z_score_value is None:
        return 0.0

    if math.isinf(z_score_value):
        return 1.0

    absolute = abs(z_score_value)

    if absolute <= 0:
        return 0.0

    # Reaches 1.0 at twice the configured
    # threshold.
    return checked_score(absolute / (threshold * 2.0))


def assess_value(
    baseline: BaselineMemory,
    value: Any,
    *,
    config: Optional[BaselineConfig] = None,
) -> BaselineAssessment:
    """
    Assess a value without updating the baseline.

    The caller can therefore compare first and
    only then decide whether the observation is
    appropriate for baseline learning.
    """

    configuration = config or BaselineConfig()

    configuration.validate()

    number = finite_number(value)

    if number is None:
        return BaselineAssessment(
            baseline_memory_id=(baseline.memory_id),
            ready=False,
            value=None,
            sample_count=(baseline.sample_count),
            mean=baseline.mean,
            standard_deviation=(baseline.standard_deviation),
            z_score=None,
            anomaly_score=0.0,
            anomalous=False,
            reason=("Value is not a finite numeric measurement."),
        )

    ready = baseline_ready(
        baseline,
        config=configuration,
    )

    if not ready:
        return BaselineAssessment(
            baseline_memory_id=(baseline.memory_id),
            ready=False,
            value=number,
            sample_count=(baseline.sample_count),
            mean=baseline.mean,
            standard_deviation=(baseline.standard_deviation),
            z_score=None,
            anomaly_score=0.0,
            anomalous=False,
            reason=("Baseline does not yet have enough samples."),
        )

    score_value = z_score(
        number,
        baseline,
    )

    anomaly_score = anomaly_score_from_z(
        score_value,
        threshold=(configuration.anomaly_z_score),
    )

    anomalous = score_value is not None and (
        math.isinf(score_value) or abs(score_value) >= configuration.anomaly_z_score
    )

    return BaselineAssessment(
        baseline_memory_id=(baseline.memory_id),
        ready=True,
        value=number,
        sample_count=(baseline.sample_count),
        mean=baseline.mean,
        standard_deviation=(baseline.standard_deviation),
        z_score=score_value,
        anomaly_score=(anomaly_score),
        anomalous=anomalous,
        reason=(
            "Value exceeded the configured z-score threshold."
            if anomalous
            else ("Value is within the configured baseline range.")
        ),
    )


def assess_record(
    store: MemoryStore,
    telemetry: TelemetryRecord,
    *,
    config: Optional[BaselineConfig] = None,
) -> BaselineAssessment:
    """
    Assess one metric against already persisted
    history without learning from the value.
    """

    configuration = config or BaselineConfig()

    key = baseline_key_for_record(telemetry)

    value = metric_value(telemetry)

    if key is None or value is None:
        return BaselineAssessment(
            baseline_memory_id=None,
            ready=False,
            value=value,
            sample_count=0,
            mean=None,
            standard_deviation=None,
            z_score=None,
            anomaly_score=0.0,
            anomalous=False,
            reason=(
                "Record does not contain a "
                "numeric metric suitable for "
                "baseline assessment."
            ),
        )

    baseline = find_baseline(
        store,
        environment_id=(telemetry.resource.environment_id),
        baseline_key_value=key,
    )

    if baseline is None:
        return BaselineAssessment(
            baseline_memory_id=None,
            ready=False,
            value=value,
            sample_count=0,
            mean=None,
            standard_deviation=None,
            z_score=None,
            anomaly_score=0.0,
            anomalous=False,
            reason=("No historical baseline exists for this metric."),
        )

    return assess_value(
        baseline,
        value,
        config=configuration,
    )


def assess_then_remember(
    store: MemoryStore,
    telemetry: TelemetryRecord,
    *,
    config: Optional[BaselineConfig] = None,
    learn_anomalies: bool = False,
) -> Tuple[
    BaselineAssessment,
    Optional[BaselineMemory],
]:
    """
    Compare a measurement with the existing
    baseline, then optionally update history.

    By default, values classified as anomalous
    are not learned into the baseline. This helps
    reduce immediate baseline contamination.
    """

    configuration = config or BaselineConfig()

    assessment = assess_record(
        store,
        telemetry,
        config=configuration,
    )

    if assessment.anomalous and not learn_anomalies:
        return (
            assessment,
            None,
        )

    updated = remember_metric(
        store,
        telemetry,
        config=configuration,
    )

    return (
        assessment,
        updated,
    )


# ---------------------------------------------------------------
# Queries and summaries
# ---------------------------------------------------------------


def list_baselines(
    store: MemoryStore,
    *,
    environment_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    active_only: bool = False,
    limit: Optional[int] = None,
) -> List[BaselineMemory]:
    query = MemoryQuery(
        memory_types=("baseline",),
        environment_ids=((environment_id,) if environment_id else ()),
        resource_ids=((resource_id,) if resource_id else ()),
        active_only=active_only,
        limit=limit,
    )

    return [
        memory
        for memory in store.query(query)
        if isinstance(
            memory,
            BaselineMemory,
        )
    ]


def baseline_summary(
    baselines: Iterable[BaselineMemory],
    *,
    config: Optional[BaselineConfig] = None,
) -> Dict[str, Any]:
    configuration = config or BaselineConfig()

    baseline_list = list(baselines)

    return {
        "baseline_count": len(baseline_list),
        "ready_baseline_count": sum(
            1
            for baseline in baseline_list
            if baseline_ready(
                baseline,
                config=configuration,
            )
        ),
        "sample_count": sum(baseline.sample_count for baseline in baseline_list),
        "resource_count": len(
            {baseline.resource_id for baseline in baseline_list if baseline.resource_id}
        ),
        "signal_count": len(
            {baseline.signal_name for baseline in baseline_list if baseline.signal_name}
        ),
        "environment_count": len(
            {baseline.environment_id for baseline in baseline_list}
        ),
    }


class BaselineManager:
    """
    Convenience interface used by memory.pipeline
    and future anomaly-history integration.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        config: Optional[BaselineConfig] = None,
    ) -> None:
        self.store = store

        self.config = config or BaselineConfig()

        self.config.validate()

    def remember(
        self,
        telemetry: TelemetryRecord,
    ) -> Optional[BaselineMemory]:
        return remember_metric(
            self.store,
            telemetry,
            config=self.config,
        )

    def remember_many(
        self,
        records: Iterable[TelemetryRecord],
    ) -> List[BaselineMemory]:
        return remember_metrics(
            self.store,
            records,
            config=self.config,
        )

    def assess(
        self,
        telemetry: TelemetryRecord,
    ) -> BaselineAssessment:
        return assess_record(
            self.store,
            telemetry,
            config=self.config,
        )

    def assess_and_remember(
        self,
        telemetry: TelemetryRecord,
        *,
        learn_anomalies: bool = False,
    ) -> Tuple[
        BaselineAssessment,
        Optional[BaselineMemory],
    ]:
        return assess_then_remember(
            self.store,
            telemetry,
            config=self.config,
            learn_anomalies=(learn_anomalies),
        )

    def list(
        self,
        *,
        environment_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        active_only: bool = False,
        limit: Optional[int] = None,
    ) -> List[BaselineMemory]:
        return list_baselines(
            self.store,
            environment_id=(environment_id),
            resource_id=(resource_id),
            active_only=(active_only),
            limit=limit,
        )
