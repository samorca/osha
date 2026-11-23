from __future__ import annotations

"""
CPU workload optimization recommendations for OSHA.

Inputs:

    AcceleratorSelection
    AcceleratorWorkload
    OptimizationConstraints
    CPUDeviceProfile
    CPUMetricsSnapshot

Output:

    OptimizationPlan

This optimizer is advisory. It does not:

    call torch.set_num_threads()
    call torch.set_num_interop_threads()
    change environment variables
    change CPU affinity
    change DataLoader workers
    change batch size
    enable BF16
    compile a model
    execute a workload

Callers explicitly decide which recommendations
to apply.
"""

from dataclasses import (
    dataclass,
)

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from src.accelerators.cpu.devices import (
    CPUDeviceInspector,
    CPUDeviceProfile,
    evaluate_cpu_workload,
)

from src.accelerators.cpu.metrics import (
    CPUMetricsCollector,
    CPUMetricsSnapshot,
    optimization_metrics_attributes,
)

from src.accelerators.optimization import (
    AcceleratorOptimizationProfile,
    AcceleratorWorkload,
    OptimizationConstraints,
    OptimizationLevel,
    OptimizationPlan,
    OptimizationRecommendation,
    RecommendationConfidence,
    RecommendationDisposition,
    RecommendationKind,
    RecommendationPriority,
    WorkloadKind,
    build_optimization_plan,
    require_optimizer_backend,
)

from src.accelerators.schema import (
    AcceleratorCapability,
    AcceleratorKind,
    AcceleratorPrecision,
    AcceleratorSelection,
)


# ------------------------------------------------------------------
# Optimizer identity
# ------------------------------------------------------------------

CPU_OPTIMIZER_NAME = "osha-cpu-optimizer"

CPU_OPTIMIZER_VERSION = "1"


# ------------------------------------------------------------------
# CPU optimization policy
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CPUOptimizationPolicy:
    """
    Tunable recommendation policy.

    These values are recommendation heuristics,
    not hardware guarantees. Callers may provide a
    different policy for their environment.
    """

    high_cpu_utilization_percent: float = 90.0

    low_cpu_utilization_percent: float = 35.0

    high_memory_utilization_percent: float = 85.0

    high_swap_utilization_percent: float = 25.0

    high_normalized_load: float = 1.0

    maximum_default_data_loader_workers: int = 8

    reserve_logical_processors: int = 1

    minimum_available_memory_bytes: int = (
        512
        * 1024**2
    )

    compilation_minimum_parameter_count: int = (
        100_000
    )

    large_model_parameter_count: int = (
        10_000_000
    )

    batch_increase_factor: int = 2

    batch_decrease_divisor: int = 2

    def validate(
        self,
    ) -> None:
        for name, value in (
            (
                "high_cpu_utilization_percent",
                self.
                high_cpu_utilization_percent,
            ),
            (
                "low_cpu_utilization_percent",
                self.
                low_cpu_utilization_percent,
            ),
            (
                "high_memory_utilization_percent",
                self.
                high_memory_utilization_percent,
            ),
            (
                "high_swap_utilization_percent",
                self.
                high_swap_utilization_percent,
            ),
        ):
            if (
                value < 0.0
                or value > 100.0
            ):
                raise ValueError(
                    f"{name} must be between "
                    "0 and 100"
                )

        if (
            self.low_cpu_utilization_percent
            >= self.high_cpu_utilization_percent
        ):
            raise ValueError(
                "low_cpu_utilization_percent "
                "must be lower than "
                "high_cpu_utilization_percent"
            )

        if self.high_normalized_load < 0.0:
            raise ValueError(
                "high_normalized_load must "
                "be >= 0"
            )

        for name, value in (
            (
                "maximum_default_data_loader_workers",
                self.
                maximum_default_data_loader_workers,
            ),
            (
                "reserve_logical_processors",
                self.
                reserve_logical_processors,
            ),
            (
                "minimum_available_memory_bytes",
                self.
                minimum_available_memory_bytes,
            ),
            (
                "compilation_minimum_parameter_count",
                self.
                compilation_minimum_parameter_count,
            ),
            (
                "large_model_parameter_count",
                self.
                large_model_parameter_count,
            ),
        ):
            if value < 0:
                raise ValueError(
                    f"{name} must be >= 0"
                )

        if self.batch_increase_factor < 2:
            raise ValueError(
                "batch_increase_factor must "
                "be >= 2"
            )

        if self.batch_decrease_divisor < 2:
            raise ValueError(
                "batch_decrease_divisor must "
                "be >= 2"
            )


# ------------------------------------------------------------------
# Recommendation helpers
# ------------------------------------------------------------------

def recommendation(
    *,
    kind: str,
    title: str,
    reason: str,
    current_value: Any = None,
    recommended_value: Any = None,
    priority: str = (
        RecommendationPriority.MEDIUM.value
    ),
    confidence: str = (
        RecommendationConfidence.MEDIUM.value
    ),
    disposition: str = (
        RecommendationDisposition.RECOMMENDED.value
    ),
    expected_effect: Optional[str] = None,
    risks: Tuple[str, ...] = (),
    prerequisites: Tuple[str, ...] = (),
    evidence: Tuple[str, ...] = (),
    automatically_applicable: bool = False,
    attributes: Optional[
        Dict[str, Any]
    ] = None,
) -> OptimizationRecommendation:
    result = OptimizationRecommendation(
        kind=kind,

        title=title,

        reason=reason,

        current_value=current_value,

        recommended_value=(
            recommended_value
        ),

        priority=priority,

        confidence=confidence,

        disposition=disposition,

        expected_effect=(
            expected_effect
        ),

        risks=risks,

        prerequisites=prerequisites,

        evidence=evidence,

        automatically_applicable=(
            automatically_applicable
        ),

        attributes=dict(
            attributes
            or {}
        ),
    )

    result.validate()

    return result


def constrained_batch_size(
    value: int,
    constraints: OptimizationConstraints,
) -> int:
    """
    Clamp a proposed batch size to caller-defined
    boundaries.
    """

    result = max(
        1,
        int(
            value
        ),
    )

    if (
        constraints.minimum_batch_size
        is not None
    ):
        result = max(
            result,
            constraints.minimum_batch_size,
        )

    if (
        constraints.maximum_batch_size
        is not None
    ):
        result = min(
            result,
            constraints.maximum_batch_size,
        )

    return result


def recommended_thread_count(
    profile: CPUDeviceProfile,
    policy: CPUOptimizationPolicy,
) -> Optional"""
    Recommend an intra-operation thread ceiling.

    The process-visible affinity count is preferred
    over the host logical processor count.
    """

    available = (
        profile.effective_processors
    )

    if available is None:
        return None

    reserved = min(
        policy.reserve_logical_processors,
        max(
            0,
            available - 1,
        ),
    )

    return max(
        1,
        available - reserved,
    )


def recommended_worker_count(
    profile: CPUDeviceProfile,
    workload: AcceleratorWorkload,
    policy: CPUOptimizationPolicy,
) -> Optional"""
    Recommend a conservative DataLoader worker
    count based on process-visible processors.

    This function does not modify a DataLoader.
    """

    available = (
        profile.effective_processors
    )

    if available is None:
        return None

    reserved = max(
        1,
        policy.reserve_logical_processors,
    )

    candidate = max(
        0,
        available - reserved,
    )

    candidate = min(
        candidate,
        policy.
        maximum_default_data_loader_workers,
    )

    if workload.latency_sensitive:
        candidate = min(
            candidate,
            2,
        )

    return candidate


# ------------------------------------------------------------------
# CPU optimizer
# ------------------------------------------------------------------

class CPUOptimizer:
    """
    Generate advisory CPU optimization plans.
    """

    def __init__(
        self,
        *,
        inspector: Optional[
            CPUDeviceInspector
        ] = None,
        metrics_collector: Optional[
            CPUMetricsCollector
        ] = None,
        policy: Optional[
            CPUOptimizationPolicy
        ] = None,
    ) -> None:
        self.inspector = (
            inspector
            or CPUDeviceInspector()
        )

        self.metrics_collector = (
            metrics_collector
            or CPUMetricsCollector(
                inspector=self.inspector
            )
        )

        self.policy = (
            policy
            or CPUOptimizationPolicy()
        )

        self.policy.validate()

    # ---------------------------------------------------------
    # Threading recommendations
    # ---------------------------------------------------------

    def _threading_recommendations(
        self,
        profile: CPUDeviceProfile,
        workload: AcceleratorWorkload,
        constraints: OptimizationConstraints,
    ) -> List[
        OptimizationRecommendation
    ]:
        if not (
            constraints.
            allow_thread_change
        ):
            return []

        output: List[
            OptimizationRecommendation
        ] = []

        target = recommended_thread_count(
            profile,
            self.policy,
        )

        if target is None:
            return [
                recommendation(
                    kind=(
                        RecommendationKind.
                        WARNING.value
                    ),

                    title=(
                        "CPU thread count "
                        "could not be derived"
                    ),

                    reason=(
                        "The effective CPU count "
                        "visible to this process "
                        "is unknown."
                    ),

                    priority=(
                        RecommendationPriority.
                        LOW.value
                    ),

                    confidence=(
                        RecommendationConfidence.
                        HIGH.value
                    ),

                    disposition=(
                        RecommendationDisposition.
                        OPTIONAL.value
                    ),

                    evidence=(
                        "effective_processors_unknown",
                    ),
                )
            ]

        current = (
            profile.torch_num_threads
        )

        if current != target:
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        THREADING.value
                    ),

                    title=(
                        "Align PyTorch CPU "
                        "threads with process "
                        "CPU visibility"
                    ),

                    reason=(
                        "The recommended thread "
                        "count is based on CPUs "
                        "visible through process "
                        "affinity while reserving "
                        "capacity for the runtime."
                    ),

                    current_value=current,

                    recommended_value={
                        "torch_num_threads": (
                            target
                        ),
                    },

                    priority=(
                        RecommendationPriority.
                        MEDIUM.value
                    ),

                    confidence=(
                        RecommendationConfidence.
                        MEDIUM.value
                    ),

                    expected_effect=(
                        "Reduce oversubscription "
                        "and align compute work "
                        "with process-visible CPUs."
                    ),

                    risks=(
                        "A different thread count "
                        "may perform better for a "
                        "specific model.",
                    ),

                    prerequisites=(
                        "Apply before sustained "
                        "parallel workload execution.",
                    ),

                    evidence=(
                        "process_affinity",
                        "effective_processor_count",
                    ),

                    automatically_applicable=False,

                    attributes={
                        "effective_processors": (
                            profile.
                            effective_processors
                        ),

                        "reserved_processors": (
                            self.policy.
                            reserve_logical_processors
                        ),
                    },
                )
            )

        if (
            profile.
            torch_num_interop_threads
            is not None
            and profile.
            effective_processors
            is not None
            and profile.
            torch_num_interop_threads
            > profile.
            effective_processors
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        THREADING.value
                    ),

                    title=(
                        "Reduce inter-operation "
                        "thread oversubscription"
                    ),

                    reason=(
                        "The current inter-operation "
                        "thread count exceeds the "
                        "number of CPUs visible to "
                        "the process."
                    ),

                    current_value=(
                        profile.
                        torch_num_interop_threads
                    ),

                    recommended_value={
                        "maximum": (
                            profile.
                            effective_processors
                        ),
                    },

                    priority=(
                        RecommendationPriority.
                        HIGH.value
                    ),

                    confidence=(
                        RecommendationConfidence.
                        HIGH.value
                    ),

                    expected_effect=(
                        "Reduce contention between "
                        "parallel operators."
                    ),

                    evidence=(
                        "torch_interop_threads",
                        "effective_processor_count",
                    ),

                    automatically_applicable=False,
                )
            )

        return output

    # ---------------------------------------------------------
    # DataLoader recommendations
    # ---------------------------------------------------------

    def _data_loader_recommendations(
        self,
        profile: CPUDeviceProfile,
        workload: AcceleratorWorkload,
        constraints: OptimizationConstraints,
    ) -> List[
        OptimizationRecommendation
    ]:
        if not (
            constraints.
            allow_data_loader_change
        ):
            return []

        if workload.kind not in {
            WorkloadKind.TRAINING.value,
            WorkloadKind.EVALUATION.value,
            WorkloadKind.DATA_PROCESSING.value,
        }:
            return []

        target = recommended_worker_count(
            profile,
            workload,
            self.policy,
        )

        if target is None:
            return []

        current = (
            workload.
            data_loader_workers
        )

        if current == target:
            return []

        return [
            recommendation(
                kind=(
                    RecommendationKind.
                    DATA_LOADER.value
                ),

                title=(
                    "Evaluate CPU DataLoader "
                    "worker count"
                ),

                reason=(
                    "The recommendation keeps "
                    "DataLoader concurrency within "
                    "the number of CPUs visible to "
                    "the process."
                ),

                current_value=current,

                recommended_value={
                    "num_workers": target,
                },

                priority=(
                    RecommendationPriority.
                    MEDIUM.value
                ),

                confidence=(
                    RecommendationConfidence.
                    MEDIUM.value
                ),

                expected_effect=(
                    "Balance input preparation "
                    "parallelism against model "
                    "execution CPU demand."
                ),

                risks=(
                    "More workers can increase "
                    "memory use and process startup "
                    "cost.",
                ),

                prerequisites=(
                    "Benchmark representative "
                    "input pipelines.",
                ),

                evidence=(
                    "effective_processor_count",
                    "workload_kind",
                ),

                automatically_applicable=False,

                attributes={
                    "effective_processors": (
                        profile.
                        effective_processors
                    ),

                    "worker_limit": (
                        self.policy.
                        maximum_default_data_loader_workers
                    ),
                },
            )
        ]

    # ---------------------------------------------------------
    # Precision recommendations
    # ---------------------------------------------------------

    def _precision_recommendations(
        self,
        profile: CPUDeviceProfile,
        workload: AcceleratorWorkload,
        constraints: OptimizationConstraints,
    ) -> List[
        OptimizationRecommendation
    ]:
        output: List[
            OptimizationRecommendation
        ] = []

        if (
            constraints.required_precision
            is not None
            and not profile.
            supports_precision(
                constraints.
                required_precision
            )
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        PRECISION.value
                    ),

                    title=(
                        "Required precision is "
                        "not supported by the "
                        "CPU profile"
                    ),

                    reason=(
                        "The selected CPU does not "
                        "advertise the precision "
                        "required by optimization "
                        "constraints."
                    ),

                    current_value=(
                        workload.precision
                    ),

                    recommended_value=(
                        constraints.
                        required_precision
                    ),

                    priority=(
                        RecommendationPriority.
                        CRITICAL.value
                    ),

                    confidence=(
                        RecommendationConfidence.
                        HIGH.value
                    ),

                    disposition=(
                        RecommendationDisposition.
                        UNSUPPORTED.value
                    ),

                    evidence=(
                        "required_precision",
                        "device_supported_precisions",
                    ),
                )
            )

            return output

        if not profile.supports_precision(
            workload.precision
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        PRECISION.value
                    ),

                    title=(
                        "Use a CPU-supported "
                        "workload precision"
                    ),

                    reason=(
                        "The current workload "
                        "precision is not advertised "
                        "by the CPU device profile."
                    ),

                    current_value=(
                        workload.precision
                    ),

                    recommended_value=(
                        AcceleratorPrecision.
                        FLOAT32.value
                    ),

                    priority=(
                        RecommendationPriority.
                        CRITICAL.value
                    ),

                    confidence=(
                        RecommendationConfidence.
                        HIGH.value
                    ),

                    disposition=(
                        RecommendationDisposition.
                        RECOMMENDED.value
                    ),

                    evidence=(
                        "workload_precision",
                        "device_supported_precisions",
                    ),
                )
            )

            return output

        if not (
            constraints.
            allow_precision_change
            and constraints.
            allow_mixed_precision
        ):
            return output

        if (
            workload.precision
            == AcceleratorPrecision.
            FLOAT32.value
            and profile.
            supports_precision(
                AcceleratorPrecision.
                BFLOAT16.value
            )
            and not workload.
            mixed_precision_enabled
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        MIXED_PRECISION.value
                    ),

                    title=(
                        "Benchmark CPU BF16 "
                        "autocast"
                    ),

                    reason=(
                        "The runtime recognizes "
                        "BF16 for this CPU profile, "
                        "but operator coverage and "
                        "performance remain workload "
                        "dependent."
                    ),

                    current_value={
                        "precision": (
                            workload.precision
                        ),

                        "mixed_precision": (
                            workload.
                            mixed_precision_enabled
                        ),
                    },

                    recommended_value={
                        "precision_candidate": (
                            AcceleratorPrecision.
                            BFLOAT16.value
                        ),

                        "mixed_precision": True,
                    },

                    priority=(
                        RecommendationPriority.
                        LOW.value
                    ),

                    confidence=(
                        RecommendationConfidence.
                        LOW.value
                    ),

                    disposition=(
                        RecommendationDisposition.
                        OPTIONAL.value
                    ),

                    expected_effect=(
                        "Potentially reduce memory "
                        "bandwidth and improve CPU "
                        "throughput."
                    ),

                    risks=(
                        "Some operations may remain "
                        "in float32.",
                        "Numerical behavior must be "
                        "validated.",
                        "Runtime recognition does "
                        "not guarantee hardware "
                        "acceleration.",
                    ),

                    prerequisites=(
                        "Benchmark representative "
                        "training or inference.",
                        "Validate output quality.",
                    ),

                    evidence=(
                        "bfloat16_runtime_support",
                    ),

                    automatically_applicable=False,
                )
            )

        return output

    # ---------------------------------------------------------
    # Memory recommendations
    # ---------------------------------------------------------

    def _memory_recommendations(
        self,
        profile: CPUDeviceProfile,
        metrics: CPUMetricsSnapshot,
        workload: AcceleratorWorkload,
        constraints: OptimizationConstraints,
    ) -> List[
        OptimizationRecommendation
    ]:
        if not (
            constraints.
            allow_memory_tuning
        ):
            return []

        output: List[
            OptimizationRecommendation
        ] = []

        available = (
            metrics.memory.
            available_bytes
        )

        memory_percent = (
            metrics.memory.
            utilization_percent
        )

        swap_percent = (
            metrics.swap.
            utilization_percent
        )

        high_pressure = bool(
            (
                memory_percent is not None
                and memory_percent
                >= self.policy.
                high_memory_utilization_percent
            )
            or (
                available is not None
                and available
                < self.policy.
                minimum_available_memory_bytes
            )
        )

        if high_pressure:
            recommended_batch = None

            if (
                workload.batch_size
                is not None
                and constraints.
                allow_batch_size_change
            ):
                recommended_batch = (
                    constrained_batch_size(
                        max(
                            1,
                            workload.batch_size
                            // self.policy.
                            batch_decrease_divisor,
                        ),
                        constraints,
                    )
                )

            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        MEMORY.value
                    ),

                    title=(
                        "Reduce CPU memory "
                        "pressure"
                    ),

                    reason=(
                        "Current system memory "
                        "telemetry indicates limited "
                        "available capacity or high "
                        "utilization."
                    ),

                    current_value={
                        "batch_size": (
                            workload.batch_size
                        ),

                        "available_memory_bytes": (
                            available
                        ),

                        "memory_utilization_percent": (
                            memory_percent
                        ),
                    },

                    recommended_value={
                        "batch_size": (
                            recommended_batch
                        ),

                        "reduce_prefetching": True,

                        "reduce_in_memory_caching": (
                            True
                        ),
                    },

                    priority=(
                        RecommendationPriority.
                        HIGH.value
                    ),

                    confidence=(
                        RecommendationConfidence.
                        HIGH.value
                    ),

                    expected_effect=(
                        "Lower the risk of process "
                        "memory exhaustion and "
                        "paging."
                    ),

                    risks=(
                        "Lower batch size may "
                        "reduce throughput.",
                    ),

                    evidence=(
                        "system_memory_utilization",
                        "available_system_memory",
                    ),

                    automatically_applicable=False,
                )
            )

        if (
            swap_percent is not None
            and swap_percent
            >= self.policy.
            high_swap_utilization_percent
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        WARNING.value
                    ),

                    title=(
                        "Investigate active swap "
                        "pressure"
                    ),

                    reason=(
                        "Swap utilization is above "
                        "the configured CPU "
                        "optimization policy level."
                    ),

                    current_value={
                        "swap_utilization_percent": (
                            swap_percent
                        ),
                    },

                    recommended_value={
                        "reduce_workload_memory": (
                            True
                        ),
                    },

                    priority=(
                        RecommendationPriority.
                        HIGH.value
                    ),

                    confidence=(
                        RecommendationConfidence.
                        HIGH.value
                    ),

                    disposition=(
                        RecommendationDisposition.
                        RECOMMENDED.value
                    ),

                    expected_effect=(
                        "Reduce storage-backed "
                        "memory stalls."
                    ),

                    risks=(
                        "Swap utilization alone "
                        "does not prove current "
                        "swapping activity.",
                    ),

                    evidence=(
                        "swap_utilization",
                    ),
                )
            )

        if (
            workload.
            estimated_peak_memory_bytes
            is not None
            and available is not None
            and workload.
            estimated_peak_memory_bytes
            > available
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        ALLOCATION.value
                    ),

                    title=(
                        "Estimated workload memory "
                        "exceeds available system "
                        "memory"
                    ),

                    reason=(
                        "The declared peak workload "
                        "memory is greater than the "
                        "currently available memory."
                    ),

                    current_value={
                        "available_memory_bytes": (
                            available
                        ),
                    },

                    recommended_value={
                        "maximum_peak_memory_bytes": (
                            available
                        ),
                    },

                    priority=(
                        RecommendationPriority.
                        CRITICAL.value
                    ),

                    confidence=(
                        RecommendationConfidence.
                        HIGH.value
                    ),

                    disposition=(
                        RecommendationDisposition.
                        NOT_RECOMMENDED.value
                    ),

                    expected_effect=(
                        "Avoid starting a workload "
                        "whose declared memory "
                        "requirement cannot fit."
                    ),

                    evidence=(
                        "estimated_peak_memory",
                        "available_system_memory",
                    ),
                )
            )

        return output

    # ---------------------------------------------------------
    # Batch-size recommendations
    # ---------------------------------------------------------

    def _batch_recommendations(
        self,
        metrics: CPUMetricsSnapshot,
        workload: AcceleratorWorkload,
        constraints: OptimizationConstraints,
    ) -> List[
        OptimizationRecommendation
    ]:
        if not (
            constraints.
            allow_batch_size_change
        ):
            return []

        if workload.batch_size is None:
            return []

        cpu_percent = (
            metrics.cpu_percent
        )

        memory_percent = (
            metrics.memory.
            utilization_percent
        )

        if (
            cpu_percent is not None
            and cpu_percent
            <= self.policy.
            low_cpu_utilization_percent
            and (
                memory_percent is None
                or memory_percent
                < self.policy.
                high_memory_utilization_percent
            )
            and workload.
            throughput_sensitive
        ):
            target = constrained_batch_size(
                workload.batch_size
                * self.policy.
                batch_increase_factor,
                constraints,
            )

            if target != workload.batch_size:
                return [
                    recommendation(
                        kind=(
                            RecommendationKind.
                            BATCH_SIZE.value
                        ),

                        title=(
                            "Benchmark a larger "
                            "CPU batch size"
                        ),

                        reason=(
                            "Observed CPU and memory "
                            "utilization leave room "
                            "for a throughput-focused "
                            "batch-size experiment."
                        ),

                        current_value=(
                            workload.batch_size
                        ),

                        recommended_value=target,

                        priority=(
                            RecommendationPriority.
                            LOW.value
                        ),

                        confidence=(
                            RecommendationConfidence.
                            LOW.value
                        ),

                        disposition=(
                            RecommendationDisposition.
                            OPTIONAL.value
                        ),

                        expected_effect=(
                            "Potentially improve "
                            "throughput by increasing "
                            "work per iteration."
                        ),

                        risks=(
                            "A larger batch consumes "
                            "more memory.",
                            "A larger training batch "
                            "may change optimization "
                            "behavior.",
                        ),

                        prerequisites=(
                            "Benchmark representative "
                            "workload steps.",
                            "Monitor memory usage.",
                        ),

                        evidence=(
                            "low_cpu_utilization",
                            "available_memory_headroom",
                            "throughput_sensitive_workload",
                        ),

                        automatically_applicable=False,
                    )
                ]

        if (
            memory_percent is not None
            and memory_percent
            >= self.policy.
            high_memory_utilization_percent
        ):
            target = constrained_batch_size(
                max(
                    1,
                    workload.batch_size
                    // self.policy.
                    batch_decrease_divisor,
                ),
                constraints,
            )

            if target != workload.batch_size:
                return [
                    recommendation(
                        kind=(
                            RecommendationKind.
                            BATCH_SIZE.value
                        ),

                        title=(
                            "Reduce CPU batch size "
                            "under memory pressure"
                        ),

                        reason=(
                            "System memory utilization "
                            "is above the configured "
                            "CPU optimization policy "
                            "level."
                        ),

                        current_value=(
                            workload.batch_size
                        ),

                        recommended_value=target,

                        priority=(
                            RecommendationPriority.
                            HIGH.value
                        ),

                        confidence=(
                            RecommendationConfidence.
                            MEDIUM.value
                        ),

                        expected_effect=(
                            "Reduce per-iteration "
                            "memory demand."
                        ),

                        risks=(
                            "Smaller batches may "
                            "reduce throughput.",
                        ),

                        evidence=(
                            "high_memory_utilization",
                        ),

                        automatically_applicable=False,
                    )
                ]

        return []

    # ---------------------------------------------------------
    # Compilation recommendations
    # ---------------------------------------------------------

    def _compilation_recommendations(
        self,
        workload: AcceleratorWorkload,
        constraints: OptimizationConstraints,
    ) -> List[
        OptimizationRecommendation
    ]:
        if not (
            constraints.
            allow_compilation
        ):
            return []

        if workload.compiled:
            return []

        if workload.kind not in {
            WorkloadKind.TRAINING.value,
            WorkloadKind.INFERENCE.value,
            WorkloadKind.EVALUATION.value,
            WorkloadKind.EMBEDDING.value,
        }:
            return []

        parameters = (
            workload.
            model_parameter_count
        )

        if (
            parameters is None
            or parameters
            < self.policy.
            compilation_minimum_parameter_count
        ):
            return []

        if (
            workload.latency_sensitive
            and workload.inference
        ):
            priority = (
                RecommendationPriority.LOW.value
            )

            disposition = (
                RecommendationDisposition.OPTIONAL.value
            )

        else:
            priority = (
                RecommendationPriority.MEDIUM.value
            )

            disposition = (
                RecommendationDisposition.OPTIONAL.value
            )

        return [
            recommendation(
                kind=(
                    RecommendationKind.
                    COMPILATION.value
                ),

                title=(
                    "Benchmark model compilation "
                    "on CPU"
                ),

                reason=(
                    "The workload contains enough "
                    "declared model parameters to "
                    "justify measuring compiled "
                    "execution against eager mode."
                ),

                current_value=False,

                recommended_value={
                    "compile_candidate": True,
                },

                priority=priority,

                confidence=(
                    RecommendationConfidence.LOW.value
                ),

                disposition=disposition,

                expected_effect=(
                    "Potentially reduce repeated "
                    "model execution overhead."
                ),

                risks=(
                    "Initial compilation adds "
                    "startup latency.",
                    "Dynamic behavior may trigger "
                    "recompilation.",
                    "Not every workload becomes "
                    "faster after compilation.",
                ),

                prerequisites=(
                    "Benchmark warm and cold "
                    "execution separately.",
                    "Retain an eager-mode fallback.",
                ),

                evidence=(
                    "model_parameter_count",
                    "repeated_model_workload",
                ),

                automatically_applicable=False,
            )
        ]

    # ---------------------------------------------------------
    # Transfer recommendations
    # ---------------------------------------------------------

    @staticmethod
    def _transfer_recommendations(
        workload: AcceleratorWorkload,
    ) -> List[
        OptimizationRecommendation
    ]:
        output: List[
            OptimizationRecommendation
        ] = []

        if workload.pin_memory_enabled:
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        PIN_MEMORY.value
                    ),

                    title=(
                        "Disable pinned-memory "
                        "loading for CPU-only "
                        "execution"
                    ),

                    reason=(
                        "Pinned host memory is "
                        "primarily useful for "
                        "transfers to accelerator "
                        "devices; this workload is "
                        "executing on CPU."
                    ),

                    current_value=True,

                    recommended_value=False,

                    priority=(
                        RecommendationPriority.
                        MEDIUM.value
                    ),

                    confidence=(
                        RecommendationConfidence.
                        HIGH.value
                    ),

                    expected_effect=(
                        "Avoid unnecessary pinned "
                        "host-memory allocation."
                    ),

                    evidence=(
                        "cpu_selected",
                        "pin_memory_enabled",
                    ),

                    automatically_applicable=False,
                )
            )

        if workload.non_blocking_transfers:
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        NON_BLOCKING_TRANSFER.value
                    ),

                    title=(
                        "Do not depend on "
                        "non-blocking transfer "
                        "behavior for CPU-only "
                        "execution"
                    ),

                    reason=(
                        "This selected workload has "
                        "no CPU-to-accelerator "
                        "transfer path."
                    ),

                    current_value=True,

                    recommended_value=False,

                    priority=(
                        RecommendationPriority.LOW.value
                    ),

                    confidence=(
                        RecommendationConfidence.HIGH.value
                    ),

                    disposition=(
                        RecommendationDisposition.OPTIONAL.value
                    ),

                    evidence=(
                        "cpu_selected",
                    ),
                )
            )

        return output

    # ---------------------------------------------------------
    # Runtime-pressure recommendations
    # ---------------------------------------------------------

    def _runtime_recommendations(
        self,
        metrics: CPUMetricsSnapshot,
    ) -> List[
        OptimizationRecommendation
    ]:
        output: List[
            OptimizationRecommendation
        ] = []

        if (
            metrics.cpu_percent is not None
            and metrics.cpu_percent
            >= self.policy.
            high_cpu_utilization_percent
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        WORKLOAD.value
                    ),

                    title=(
                        "CPU capacity is currently "
                        "highly utilized"
                    ),

                    reason=(
                        "The current CPU utilization "
                        "is above the configured "
                        "optimizer policy level."
                    ),

                    current_value=(
                        metrics.cpu_percent
                    ),

                    recommended_value={
                        "benchmark_lower_parallelism": (
                            True
                        ),
                    },

                    priority=(
                        RecommendationPriority.
                        MEDIUM.value
                    ),

                    confidence=(
                        RecommendationConfidence.
                        HIGH.value
                    ),

                    disposition=(
                        RecommendationDisposition.
                        OPTIONAL.value
                    ),

                    expected_effect=(
                        "Potentially reduce runtime "
                        "contention."
                    ),

                    risks=(
                        "High utilization may be "
                        "expected and efficient.",
                    ),

                    evidence=(
                        "system_cpu_utilization",
                    ),
                )
            )

        if (
            metrics.normalized_load_1m
            is not None
            and metrics.normalized_load_1m
            >= self.policy.
            high_normalized_load
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        WARNING.value
                    ),

                    title=(
                        "Process-visible CPU load "
                        "is elevated"
                    ),

                    reason=(
                        "The one-minute load average "
                        "normalized by process-visible "
                        "CPUs is above the configured "
                        "policy level."
                    ),

                    current_value=(
                        metrics.
                        normalized_load_1m
                    ),

                    recommended_value={
                        "review_concurrent_workloads": (
                            True
                        ),
                    },

                    priority=(
                        RecommendationPriority.
                        MEDIUM.value
                    ),

                    confidence=(
                        RecommendationConfidence.
                        MEDIUM.value
                    ),

                    disposition=(
                        RecommendationDisposition.
                        OPTIONAL.value
                    ),

                    evidence=(
                        "normalized_load_average",
                        "effective_cpu_count",
                    ),
                )
            )

        return output

    # ---------------------------------------------------------
    # Recommend
    # ---------------------------------------------------------

    def recommend(
        self,
        *,
        selection: AcceleratorSelection,
        workload: AcceleratorWorkload,
        constraints: Optional[
            OptimizationConstraints
        ] = None,
        profile: Optional[
            CPUDeviceProfile
        ] = None,
        metrics: Optional[
            CPUMetricsSnapshot
        ] = None,
        metrics_interval: Optional[
            float
        ] = None,
    ) -> OptimizationPlan:
        """
        Build a complete advisory optimization
        plan for a CPU selection.
        """

        require_optimizer_backend(
            selection,
            AcceleratorKind.CPU.value,
        )

        workload.validate()

        resolved_constraints = (
            constraints
            or OptimizationConstraints()
        )

        resolved_constraints.validate()

        if (
            resolved_constraints.level
            == OptimizationLevel.DISABLED.value
        ):
            optimization_profile = (
                self.inspector.
                optimization_profile(
                    workload,
                    device=(
                        selection.device
                    ),
                    refresh_memory=True,
                )
            )

            return build_optimization_plan(
                selection=selection,

                workload=workload,

                constraints=(
                    resolved_constraints
                ),

                recommendations=(),

                profile=(
                    optimization_profile
                ),

                optimizer_name=(
                    CPU_OPTIMIZER_NAME
                ),

                attributes={
                    "optimizer_version": (
                        CPU_OPTIMIZER_VERSION
                    ),

                    "optimization_disabled": (
                        True
                    ),
                },
            )

        resolved_profile = (
            profile
            or self.inspector.inspect(
                selection.device,
                refresh_memory=True,
            )
        )

        resolved_profile.validate()

        resolved_metrics = (
            metrics
            or self.metrics_collector.
            collect(
                interval=(
                    metrics_interval
                ),
                include_process=True,
            )
        )

        resolved_metrics.validate()

        suitability = (
            evaluate_cpu_workload(
                resolved_profile,
                workload,
            )
        )

        recommendations: List[
            OptimizationRecommendation
        ] = []

        if not suitability.suitable:
            recommendations.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        WARNING.value
                    ),

                    title=(
                        "CPU workload suitability "
                        "check failed"
                    ),

                    reason=(
                        "The CPU profile does not "
                        "satisfy one or more declared "
                        "workload properties."
                    ),

                    current_value={
                        "reasons": list(
                            suitability.reasons
                        ),
                    },

                    recommended_value={
                        "review_workload": True,
                    },

                    priority=(
                        RecommendationPriority.
                        CRITICAL.value
                    ),

                    confidence=(
                        RecommendationConfidence.
                        HIGH.value
                    ),

                    disposition=(
                        RecommendationDisposition.
                        NOT_RECOMMENDED.value
                    ),

                    evidence=(
                        suitability.reasons
                    ),
                )
            )

        recommendations.extend(
            self._threading_recommendations(
                resolved_profile,
                workload,
                resolved_constraints,
            )
        )

        recommendations.extend(
            self._data_loader_recommendations(
                resolved_profile,
                workload,
                resolved_constraints,
            )
        )

        recommendations.extend(
            self._precision_recommendations(
                resolved_profile,
                workload,
                resolved_constraints,
            )
        )

        recommendations.extend(
            self._memory_recommendations(
                resolved_profile,
                resolved_metrics,
                workload,
                resolved_constraints,
            )
        )

        recommendations.extend(
            self._batch_recommendations(
                resolved_metrics,
                workload,
                resolved_constraints,
            )
        )

        recommendations.extend(
            self._compilation_recommendations(
                workload,
                resolved_constraints,
            )
        )

        recommendations.extend(
            self._transfer_recommendations(
                workload
            )
        )

        recommendations.extend(
            self._runtime_recommendations(
                resolved_metrics
            )
        )

        optimization_profile = (
            AcceleratorOptimizationProfile(
                device=(
                    resolved_profile.device
                ),

                workload=workload,

                free_memory_bytes=(
                    resolved_metrics.
                    memory.
                    available_bytes
                ),

                total_memory_bytes=(
                    resolved_metrics.
                    memory.total_bytes
                ),

                utilization_percent=(
                    resolved_metrics.
                    cpu_percent
                ),

                temperature_celsius=None,

                capabilities=(
                    resolved_profile.
                    capabilities
                ),

                attributes={
                    **dict(
                        self.inspector.
                        optimization_profile(
                            workload,
                            device=(
                                selection.device
                            ),
                            refresh_memory=False,
                        ).attributes
                    ),

                    "metrics": (
                        optimization_metrics_attributes(
                            resolved_metrics
                        )
                    ),

                    "suitability": (
                        suitability.to_dict()
                    ),

                    "supported_precisions": list(
                        resolved_profile.
                        supported_precisions
                    ),
                },
            )
        )

        optimization_profile.validate()

        return build_optimization_plan(
            selection=selection,

            workload=workload,

            constraints=(
                resolved_constraints
            ),

            recommendations=tuple(
                recommendations
            ),

            profile=(
                optimization_profile
            ),

            optimizer_name=(
                CPU_OPTIMIZER_NAME
            ),

            attributes={
                "optimizer_version": (
                    CPU_OPTIMIZER_VERSION
                ),

                "policy": {
                    "high_cpu_utilization_percent": (
                        self.policy.
                        high_cpu_utilization_percent
                    ),

                    "low_cpu_utilization_percent": (
                        self.policy.
                        low_cpu_utilization_percent
                    ),

                    "high_memory_utilization_percent": (
                        self.policy.
                        high_memory_utilization_percent
                    ),

                    "high_swap_utilization_percent": (
                        self.policy.
                        high_swap_utilization_percent
                    ),

                    "high_normalized_load": (
                        self.policy.
                        high_normalized_load
                    ),
                },

                "automatically_applied": (
                    False
                ),
            },
        )


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------

def recommend_cpu_optimizations(
    *,
    selection: AcceleratorSelection,
    workload: AcceleratorWorkload,
    constraints: Optional[
        OptimizationConstraints
    ] = None,
    profile: Optional[
        CPUDeviceProfile
    ] = None,
    metrics: Optional[
        CPUMetricsSnapshot
    ] = None,
    metrics_interval: Optional[
        float
    ] = None,
    policy: Optional[
        CPUOptimizationPolicy
    ] = None,
) -> OptimizationPlan:
    return CPUOptimizer(
        policy=policy
    ).recommend(
        selection=selection,
        workload=workload,
        constraints=constraints,
        profile=profile,
        metrics=metrics,
        metrics_interval=(
            metrics_interval
        ),
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "CPU_OPTIMIZER_NAME",
    "CPU_OPTIMIZER_VERSION",

    "CPUOptimizationPolicy",

    "recommendation",
    "constrained_batch_size",
    "recommended_thread_count",
    "recommended_worker_count",

    "CPUOptimizer",

    "recommend_cpu_optimizations",
]
