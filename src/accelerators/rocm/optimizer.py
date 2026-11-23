from __future__ import annotations

"""
AMD ROCm/HIP optimization recommendations for OSHA.

Inputs:

    AcceleratorSelection
    AcceleratorWorkload
    OptimizationConstraints
    ROCmDeviceProfile
    ROCmMetricsSnapshot

Output:

    OptimizationPlan

This optimizer is advisory. It does not:

    enable mixed precision
    modify model precision
    change batch size
    change DataLoader configuration
    call torch.compile
    clear ROCm/HIP caches
    reset peak-memory statistics
    synchronize accelerator streams
    execute training or inference

Callers explicitly decide which recommendations
to apply and should benchmark the result.

Identity boundary:

    OSHA logical device:
        rocm:N

    PyTorch HIP runtime device:
        cuda:N
"""

from dataclasses import dataclass

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
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

from src.accelerators.rocm.devices import (
    ROCmDeviceInspector,
    ROCmDeviceProfile,
    evaluate_rocm_workload,
)

from src.accelerators.rocm.metrics import (
    ROCmMetricsCollector,
    ROCmMetricsSnapshot,
    optimization_metrics_attributes,
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

ROCM_OPTIMIZER_NAME = "osha-rocm-optimizer"

ROCM_OPTIMIZER_VERSION = "1"


# ------------------------------------------------------------------
# ROCm optimization policy
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ROCmOptimizationPolicy:
    """
    Adjustable recommendation thresholds for
    ROCm/HIP workloads.

    These are policy heuristics, not performance
    guarantees.
    """

    high_memory_pressure_ratio: float = 0.90

    moderate_memory_pressure_ratio: float = 0.75

    low_memory_pressure_ratio: float = 0.40

    high_device_utilization_percent: float = 90.0

    low_device_utilization_percent: float = 35.0

    high_memory_controller_percent: float = 90.0

    high_temperature_celsius: float = 85.0

    allocator_inactive_ratio: float = 0.40

    minimum_memory_headroom_bytes: int = (
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

    prefer_bfloat16_when_supported: bool = True

    recommend_tf32_for_supported_float32: bool = (
        False
    )

    def validate(
        self,
    ) -> None:
        for name, value in (
            (
                "high_memory_pressure_ratio",
                self.high_memory_pressure_ratio,
            ),
            (
                "moderate_memory_pressure_ratio",
                self.moderate_memory_pressure_ratio,
            ),
            (
                "low_memory_pressure_ratio",
                self.low_memory_pressure_ratio,
            ),
            (
                "allocator_inactive_ratio",
                self.allocator_inactive_ratio,
            ),
        ):
            if (
                value < 0.0
                or value > 1.0
            ):
                raise ValueError(
                    f"{name} must be between "
                    "0 and 1"
                )

        if not (
            self.low_memory_pressure_ratio
            < self.moderate_memory_pressure_ratio
            < self.high_memory_pressure_ratio
        ):
            raise ValueError(
                "ROCm memory-pressure ratios "
                "must increase from low to "
                "moderate to high."
            )

        for name, value in (
            (
                "high_device_utilization_percent",
                self.
                high_device_utilization_percent,
            ),
            (
                "low_device_utilization_percent",
                self.
                low_device_utilization_percent,
            ),
            (
                "high_memory_controller_percent",
                self.
                high_memory_controller_percent,
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
            self.low_device_utilization_percent
            >= self.high_device_utilization_percent
        ):
            raise ValueError(
                "low_device_utilization_percent "
                "must be lower than "
                "high_device_utilization_percent."
            )

        if self.high_temperature_celsius < 0.0:
            raise ValueError(
                "high_temperature_celsius "
                "must be >= 0"
            )

        for name, value in (
            (
                "minimum_memory_headroom_bytes",
                self.
                minimum_memory_headroom_bytes,
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
# Recommendation helper
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
        recommended_value=recommended_value,
        priority=priority,
        confidence=confidence,
        disposition=disposition,
        expected_effect=expected_effect,
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


# ------------------------------------------------------------------
# Batch-size constraint helper
# ------------------------------------------------------------------

def constrained_batch_size(
    value: int,
    constraints: OptimizationConstraints,
) -> int:
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


# ------------------------------------------------------------------
# Allocator reservation helper
# ------------------------------------------------------------------

def allocator_inactive_ratio(
    metrics: ROCmMetricsSnapshot,
) -> Optional[float]:
    reserved = (
        metrics.allocator.reserved_bytes
    )

    inactive = (
        metrics.
        allocator.
        inactive_reserved_bytes
    )

    if (
        reserved is None
        or inactive is None
        or reserved <= 0
    ):
        return None

    return (
        inactive
        / reserved
    )


# ------------------------------------------------------------------
# Maximum memory policy
# ------------------------------------------------------------------

def effective_memory_limit(
    metrics: ROCmMetricsSnapshot,
    constraints: OptimizationConstraints,
) -> Optional[int]:
    total = metrics.memory.total_bytes

    if total is None:
        return None

    fraction = (
        constraints.
        maximum_memory_fraction
    )

    if fraction is None:
        return total

    return int(
        total
        * fraction
    )


# ------------------------------------------------------------------
# ROCm optimizer
# ------------------------------------------------------------------

class ROCmOptimizer:
    """
    Generate advisory ROCm/HIP optimization plans.
    """

    def __init__(
        self,
        *,
        inspector: Optional[
            ROCmDeviceInspector
        ] = None,
        metrics_collector: Optional[
            ROCmMetricsCollector
        ] = None,
        policy: Optional[
            ROCmOptimizationPolicy
        ] = None,
    ) -> None:
        self.inspector = (
            inspector
            or ROCmDeviceInspector()
        )

        self.metrics_collector = (
            metrics_collector
        )

        self.policy = (
            policy
            or ROCmOptimizationPolicy()
        )

        self.policy.validate()

    # ---------------------------------------------------------
    # Precision recommendations
    # ---------------------------------------------------------

    def _precision_recommendations(
        self,
        profile: ROCmDeviceProfile,
        workload: AcceleratorWorkload,
        constraints: OptimizationConstraints,
    ) -> List[
        OptimizationRecommendation
    ]:
        output: List[
            OptimizationRecommendation
        ] = []

        required = (
            constraints.required_precision
        )

        if (
            required is not None
            and not profile.supports_precision(
                required
            )
        ):
            return [
                recommendation(
                    kind=(
                        RecommendationKind.
                        PRECISION.value
                    ),

                    title=(
                        "Required precision is "
                        "not supported by this "
                        "ROCm device"
                    ),

                    reason=(
                        "The selected AMD device "
                        "does not advertise the "
                        "precision required by the "
                        "optimization constraints."
                    ),

                    current_value=(
                        workload.precision
                    ),

                    recommended_value=(
                        required
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
            ]

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
                        "Use a ROCm-supported "
                        "workload precision"
                    ),

                    reason=(
                        "The workload precision "
                        "is not advertised by the "
                        "selected ROCm device."
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

                    evidence=(
                        "workload_precision",
                        "device_supported_precisions",
                    ),
                )
            )

            return output

        if not (
            constraints.allow_precision_change
            and constraints.
            allow_mixed_precision
        ):
            return output

        if workload.mixed_precision_enabled:
            return output

        if (
            workload.precision
            != AcceleratorPrecision.
            FLOAT32.value
        ):
            return output

        if not profile.supports_capability(
            AcceleratorCapability.
            MIXED_PRECISION.value
        ):
            return output

        if (
            self.policy.
            prefer_bfloat16_when_supported
            and profile.bfloat16_supported
        ):
            precision_candidate = (
                AcceleratorPrecision.
                BFLOAT16.value
            )

            confidence = (
                RecommendationConfidence.
                HIGH.value
            )

            evidence = (
                "bfloat16_supported",
                "mixed_precision_capability",
            )

        else:
            precision_candidate = (
                AcceleratorPrecision.
                FLOAT16.value
            )

            confidence = (
                RecommendationConfidence.
                MEDIUM.value
            )

            evidence = (
                "float16_supported",
                "mixed_precision_capability",
            )

        risks = (
            "Validate numerical accuracy and model quality.",
            "Some operations may continue to execute in float32.",
            "Performance depends on ROCm operator and kernel coverage.",
        )

        if workload.training:
            risks = (
                *risks,
                "FP16 training may require gradient scaling.",
            )

        output.append(
            recommendation(
                kind=(
                    RecommendationKind.
                    MIXED_PRECISION.value
                ),

                title=(
                    "Benchmark ROCm automatic "
                    "mixed precision"
                ),

                reason=(
                    "The selected ROCm device "
                    "advertises mixed-precision "
                    "support and the workload "
                    "currently uses float32."
                ),

                current_value={
                    "precision": (
                        workload.precision
                    ),

                    "mixed_precision": (
                        False
                    ),
                },

                recommended_value={
                    "mixed_precision": True,

                    "autocast_precision": (
                        precision_candidate
                    ),

                    "gradient_scaler": bool(
                        workload.training
                        and precision_candidate
                        == AcceleratorPrecision.
                        FLOAT16.value
                    ),

                    "runtime_device_type": (
                        "cuda"
                    ),
                },

                priority=(
                    RecommendationPriority.
                    HIGH.value
                ),

                confidence=confidence,

                disposition=(
                    RecommendationDisposition.
                    RECOMMENDED.value
                ),

                expected_effect=(
                    "Potentially reduce VRAM "
                    "usage and improve ROCm "
                    "workload throughput."
                ),

                risks=risks,

                prerequisites=(
                    "Benchmark representative workload steps.",
                    "Validate numerical quality.",
                    "Use PyTorch's cuda device type for HIP.",
                ),

                evidence=evidence,

                automatically_applicable=False,

                attributes={
                    "architecture": (
                        profile.architecture
                    ),

                    "architecture_family": (
                        profile.
                        architecture_family
                    ),

                    "matrix_acceleration_candidate": (
                        profile.
                        matrix_acceleration_candidate
                    ),

                    "pytorch_uses_cuda_interface": (
                        True
                    ),
                },
            )
        )

        return output

    # ---------------------------------------------------------
    # TF32 recommendations
    # ---------------------------------------------------------

    def _tf32_recommendations(
        self,
        profile: ROCmDeviceProfile,
        workload: AcceleratorWorkload,
        constraints: OptimizationConstraints,
    ) -> List[
        OptimizationRecommendation
    ]:
        if not (
            self.policy.
            recommend_tf32_for_supported_float32
        ):
            return []

        if not (
            constraints.
            allow_precision_change
        ):
            return []

        if not profile.tf32_supported:
            return []

        if (
            workload.precision
            != AcceleratorPrecision.
            FLOAT32.value
        ):
            return []

        return [
            recommendation(
                kind=(
                    RecommendationKind.
                    PRECISION.value
                ),

                title=(
                    "Benchmark ROCm TF32 "
                    "matrix execution"
                ),

                reason=(
                    "The selected ROCm runtime "
                    "reports TF32 support for "
                    "float32 matrix operations."
                ),

                current_value={
                    "precision": (
                        AcceleratorPrecision.
                        FLOAT32.value
                    ),

                    "tf32_enabled": (
                        False
                    ),
                },

                recommended_value={
                    "tf32_candidate": True,
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
                    "Potentially improve supported "
                    "float32 matrix-operation "
                    "throughput."
                ),

                risks=(
                    "TF32 changes numerical behavior.",
                    "Support and benefit are architecture-dependent.",
                ),

                prerequisites=(
                    "Validate numerical quality.",
                    "Benchmark representative matrix workloads.",
                ),

                evidence=(
                    "rocm_tf32_supported",
                    "float32_workload",
                ),

                automatically_applicable=False,
            )
        ]

    # ---------------------------------------------------------
    # Host-device transfer recommendations
    # ---------------------------------------------------------

    @staticmethod
    def _transfer_recommendations(
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
            WorkloadKind.INFERENCE.value,
            WorkloadKind.EVALUATION.value,
            WorkloadKind.EMBEDDING.value,
        }:
            return []

        output: List[
            OptimizationRecommendation
        ] = []

        if workload.pin_memory_enabled is not True:
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        PIN_MEMORY.value
                    ),

                    title=(
                        "Benchmark pinned "
                        "host-memory loading"
                    ),

                    reason=(
                        "ROCm workloads may "
                        "benefit from pinned host "
                        "buffers during CPU-to-HIP "
                        "input transfer."
                    ),

                    current_value=(
                        workload.
                        pin_memory_enabled
                    ),

                    recommended_value=True,

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

                    expected_effect=(
                        "Potentially reduce "
                        "host-to-device transfer "
                        "overhead."
                    ),

                    risks=(
                        "Pinned memory consumes a constrained host resource.",
                        "Performance depends on the input pipeline.",
                    ),

                    prerequisites=(
                        "Benchmark end-to-end throughput.",
                        "Monitor host-memory pressure.",
                    ),

                    evidence=(
                        "rocm_selected",
                        "host_to_device_transfer",
                    ),

                    automatically_applicable=False,
                )
            )

        if (
            workload.
            non_blocking_transfers
            is not True
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        NON_BLOCKING_TRANSFER.value
                    ),

                    title=(
                        "Benchmark non-blocking "
                        "CPU-to-ROCm transfers"
                    ),

                    reason=(
                        "Non-blocking transfer is "
                        "a candidate when tensors "
                        "are moved from host memory "
                        "to the selected HIP device."
                    ),

                    current_value=(
                        workload.
                        non_blocking_transfers
                    ),

                    recommended_value=True,

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

                    expected_effect=(
                        "Potentially overlap "
                        "host-to-device copies "
                        "with other execution."
                    ),

                    risks=(
                        "Correctness requires valid tensor lifetime.",
                        "Synchronization behavior must be verified.",
                    ),

                    prerequisites=(
                        "Use a ROCm/HIP input-transfer path.",
                        "Benchmark representative batches.",
                    ),

                    evidence=(
                        "rocm_selected",
                        "host_to_device_transfer",
                    ),

                    automatically_applicable=False,

                    attributes={
                        "pytorch_device_type": (
                            "cuda"
                        ),
                    },
                )
            )

        return output

    # ---------------------------------------------------------
    # Memory recommendations
    # ---------------------------------------------------------

    def _memory_recommendations(
        self,
        profile: ROCmDeviceProfile,
        metrics: ROCmMetricsSnapshot,
        workload: AcceleratorWorkload,
        constraints: OptimizationConstraints,
    ) -> List[
        OptimizationRecommendation
    ]:
        if not (
            constraints.allow_memory_tuning
        ):
            return []

        output: List[
            OptimizationRecommendation
        ] = []

        pressure = (
            metrics.global_memory_pressure
        )

        free_bytes = (
            metrics.memory.free_bytes
        )

        memory_limit = (
            effective_memory_limit(
                metrics,
                constraints,
            )
        )

        high_pressure = bool(
            (
                pressure is not None
                and pressure
                >= self.policy.
                high_memory_pressure_ratio
            )
            or (
                free_bytes is not None
                and free_bytes
                < self.policy.
                minimum_memory_headroom_bytes
            )
        )

        if high_pressure:
            target_batch = None

            if (
                workload.batch_size
                is not None
                and constraints.
                allow_batch_size_change
            ):
                target_batch = (
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
                        "Reduce ROCm memory "
                        "pressure"
                    ),

                    reason=(
                        "Current device-memory "
                        "telemetry indicates high "
                        "VRAM utilization or limited "
                        "free-memory headroom."
                    ),

                    current_value={
                        "batch_size": (
                            workload.batch_size
                        ),

                        "free_memory_bytes": (
                            free_bytes
                        ),

                        "memory_pressure": (
                            pressure
                        ),

                        "effective_memory_limit": (
                            memory_limit
                        ),
                    },

                    recommended_value={
                        "batch_size": (
                            target_batch
                        ),

                        "gradient_accumulation": (
                            bool(
                                workload.training
                            )
                        ),

                        "release_unused_references": (
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
                        "Reduce the likelihood "
                        "of a ROCm out-of-memory "
                        "failure."
                    ),

                    risks=(
                        "Smaller batches may reduce throughput.",
                        "Gradient accumulation changes optimizer-step frequency.",
                    ),

                    evidence=(
                        "global_vram_pressure",
                        "free_device_memory",
                    ),

                    automatically_applicable=False,
                )
            )

        estimated_peak = (
            workload.
            estimated_peak_memory_bytes
        )

        available_limit = (
            free_bytes
        )

        if (
            memory_limit is not None
            and metrics.memory.used_bytes
            is not None
        ):
            policy_headroom = max(
                0,
                memory_limit
                - metrics.memory.used_bytes,
            )

            if available_limit is None:
                available_limit = (
                    policy_headroom
                )

            else:
                available_limit = min(
                    available_limit,
                    policy_headroom,
                )

        if (
            estimated_peak is not None
            and available_limit is not None
            and estimated_peak
            > available_limit
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        ALLOCATION.value
                    ),

                    title=(
                        "Declared workload "
                        "memory exceeds available "
                        "ROCm headroom"
                    ),

                    reason=(
                        "The workload's estimated "
                        "peak memory is greater "
                        "than the current device "
                        "headroom allowed by memory "
                        "telemetry and constraints."
                    ),

                    current_value={
                        "estimated_peak_memory_bytes": (
                            estimated_peak
                        ),

                        "available_memory_bytes": (
                            available_limit
                        ),
                    },

                    recommended_value={
                        "reduce_peak_memory": True,
                        "select_larger_device": True,
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
                        "Avoid beginning a "
                        "workload whose declared "
                        "memory requirement cannot "
                        "fit."
                    ),

                    evidence=(
                        "estimated_peak_memory",
                        "available_device_memory",
                    ),
                )
            )

        inactive_ratio = (
            allocator_inactive_ratio(
                metrics
            )
        )

        if (
            inactive_ratio is not None
            and inactive_ratio
            >= self.policy.
            allocator_inactive_ratio
            and metrics.
            allocator.reserved_bytes
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        CACHE.value
                    ),

                    title=(
                        "Review inactive ROCm "
                        "allocator reservation"
                    ),

                    reason=(
                        "A substantial portion "
                        "of PyTorch-reserved HIP "
                        "memory is not currently "
                        "occupied by live tensor "
                        "allocations."
                    ),

                    current_value={
                        "allocated_bytes": (
                            metrics.
                            allocator.
                            allocated_bytes
                        ),

                        "reserved_bytes": (
                            metrics.
                            allocator.
                            reserved_bytes
                        ),

                        "inactive_reserved_ratio": (
                            inactive_ratio
                        ),
                    },

                    recommended_value={
                        "release_unused_tensor_references": (
                            True
                        ),

                        "consider_empty_cache_at_safe_boundary": (
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

                    expected_effect=(
                        "Make unused cached "
                        "blocks available to "
                        "other HIP workloads."
                    ),

                    risks=(
                        "Frequent cache clearing can increase allocation overhead.",
                        "Cache clearing does not free live tensors.",
                    ),

                    prerequisites=(
                        "Use only at a safe workload boundary.",
                        "Confirm unused Python references are released.",
                    ),

                    evidence=(
                        "pytorch_allocated_memory",
                        "pytorch_reserved_memory",
                    ),

                    automatically_applicable=False,
                )
            )

        return output

    # ---------------------------------------------------------
    # Batch-size recommendations
    # ---------------------------------------------------------

    def _batch_recommendations(
        self,
        metrics: ROCmMetricsSnapshot,
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

        pressure = (
            metrics.global_memory_pressure
        )

        utilization = (
            metrics.
            activity.
            device_utilization_percent
        )

        if (
            workload.throughput_sensitive
            and pressure is not None
            and pressure
            <= self.policy.
            low_memory_pressure_ratio
            and (
                utilization is None
                or utilization
                <= self.policy.
                low_device_utilization_percent
            )
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
                            "ROCm batch size"
                        ),

                        reason=(
                            "Observed VRAM pressure "
                            "is low and the workload "
                            "is throughput-sensitive."
                        ),

                        current_value=(
                            workload.batch_size
                        ),

                        recommended_value=target,

                        priority=(
                            RecommendationPriority.
                            MEDIUM.value
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
                            "accelerator occupancy "
                            "and throughput."
                        ),

                        risks=(
                            "A larger batch consumes more VRAM.",
                            "Training batch size may affect optimization behavior.",
                        ),

                        prerequisites=(
                            "Benchmark representative workload steps.",
                            "Monitor peak allocated memory.",
                        ),

                        evidence=(
                            "low_global_vram_pressure",
                            "throughput_sensitive_workload",
                        ),

                        automatically_applicable=False,
                    )
                ]

        if (
            pressure is not None
            and pressure
            >= self.policy.
            high_memory_pressure_ratio
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
                            "Reduce batch size "
                            "under ROCm memory "
                            "pressure"
                        ),

                        reason=(
                            "Current VRAM pressure "
                            "is above the configured "
                            "high-pressure policy "
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
                            HIGH.value
                        ),

                        expected_effect=(
                            "Reduce per-step "
                            "device-memory demand."
                        ),

                        risks=(
                            "Smaller batches may reduce throughput.",
                        ),

                        evidence=(
                            "high_global_vram_pressure",
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
        profile: ROCmDeviceProfile,
        workload: AcceleratorWorkload,
        constraints: OptimizationConstraints,
    ) -> List[
        OptimizationRecommendation
    ]:
        if not constraints.allow_compilation:
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

        parameter_count = (
            workload.model_parameter_count
        )

        if (
            parameter_count is None
            or parameter_count
            < self.policy.
            compilation_minimum_parameter_count
        ):
            return []

        mode = (
            "reduce-overhead"
            if workload.latency_sensitive
            else "default"
        )

        return [
            recommendation(
                kind=(
                    RecommendationKind.
                    COMPILATION.value
                ),

                title=(
                    "Benchmark torch.compile "
                    "on ROCm"
                ),

                reason=(
                    "The described model "
                    "workload is a candidate "
                    "for compiled execution, "
                    "but benefit depends on "
                    "graph stability, operator "
                    "coverage, and compilation "
                    "cost."
                ),

                current_value=False,

                recommended_value={
                    "compile_candidate": True,
                    "backend": "inductor",
                    "mode": mode,
                    "runtime_device_type": (
                        "cuda"
                    ),
                },

                priority=(
                    RecommendationPriority.
                    MEDIUM.value
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
                    "Potentially reduce repeated "
                    "ROCm model execution "
                    "overhead and improve kernel "
                    "fusion."
                ),

                risks=(
                    "First execution includes compilation cost.",
                    "Dynamic shapes may cause recompilation.",
                    "Graph breaks reduce optimization opportunities.",
                    "ROCm operator coverage varies by workload.",
                    "Some workloads remain faster in eager mode.",
                ),

                prerequisites=(
                    "Benchmark cold and warm execution separately.",
                    "Test representative input shapes.",
                    "Retain an eager-mode fallback.",
                ),

                evidence=(
                    "model_parameter_count",
                    "rocm_model_workload",
                ),

                automatically_applicable=False,

                attributes={
                    "architecture": (
                        profile.architecture
                    ),

                    "architecture_family": (
                        profile.
                        architecture_family
                    ),
                },
            )
        ]

    # ---------------------------------------------------------
    # Runtime-pressure recommendations
    # ---------------------------------------------------------

    def _runtime_recommendations(
        self,
        metrics: ROCmMetricsSnapshot,
    ) -> List[
        OptimizationRecommendation
    ]:
        output: List[
            OptimizationRecommendation
        ] = []

        utilization = (
            metrics.
            activity.
            device_utilization_percent
        )

        memory_controller = (
            metrics.
            activity.
            memory_utilization_percent
        )

        temperature = (
            metrics.
            activity.
            temperature_celsius
        )

        if (
            utilization is not None
            and utilization
            >= self.policy.
            high_device_utilization_percent
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        WORKLOAD.value
                    ),

                    title=(
                        "ROCm compute "
                        "utilization is high"
                    ),

                    reason=(
                        "The selected device "
                        "is currently reporting "
                        "high compute utilization."
                    ),

                    current_value=utilization,

                    recommended_value={
                        "retain_current_parallelism": (
                            True
                        ),

                        "benchmark_before_increasing_load": (
                            True
                        ),
                    },

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

                    risks=(
                        "High utilization may already represent efficient device use.",
                    ),

                    evidence=(
                        "rocm_device_utilization",
                    ),
                )
            )

        if (
            memory_controller is not None
            and memory_controller
            >= self.policy.
            high_memory_controller_percent
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        MEMORY.value
                    ),

                    title=(
                        "ROCm workload may be "
                        "memory-bandwidth bound"
                    ),

                    reason=(
                        "Memory-controller "
                        "utilization is high in "
                        "the current telemetry "
                        "sample."
                    ),

                    current_value=(
                        memory_controller
                    ),

                    recommended_value={
                        "benchmark_lower_precision": (
                            True
                        ),

                        "reduce_transfer_volume": (
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

                    risks=(
                        "A utilization sample alone does not prove the workload bottleneck.",
                    ),

                    evidence=(
                        "rocm_memory_controller_utilization",
                    ),
                )
            )

        if (
            temperature is not None
            and temperature
            >= self.policy.
            high_temperature_celsius
        ):
            output.append(
                recommendation(
                    kind=(
                        RecommendationKind.
                        WARNING.value
                    ),

                    title=(
                        "ROCm device "
                        "temperature is elevated"
                    ),

                    reason=(
                        "The reported AMD device "
                        "temperature is above the "
                        "configured policy level."
                    ),

                    current_value=temperature,

                    recommended_value={
                        "review_thermal_conditions": (
                            True
                        ),

                        "avoid_increasing_workload": (
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

                    evidence=(
                        "device_temperature",
                    ),
                )
            )

        return output

    # ---------------------------------------------------------
    # Optimization profile
    # ---------------------------------------------------------

    @staticmethod
    def _optimization_profile(
        profile: ROCmDeviceProfile,
        metrics: ROCmMetricsSnapshot,
        workload: AcceleratorWorkload,
        suitability: Any,
    ) -> AcceleratorOptimizationProfile:
        result = AcceleratorOptimizationProfile(
            device=profile.device,

            workload=workload,

            free_memory_bytes=(
                metrics.memory.free_bytes
            ),

            total_memory_bytes=(
                metrics.memory.total_bytes
            ),

            utilization_percent=(
                metrics.
                activity.
                device_utilization_percent
            ),

            temperature_celsius=(
                metrics.
                activity.
                temperature_celsius
            ),

            capabilities=(
                profile.capabilities
            ),

            attributes={
                "logical_device_id": (
                    profile.device.device_id
                ),

                "runtime_device": (
                    profile.runtime_device
                ),

                "architecture": (
                    profile.architecture
                ),

                "architecture_family": (
                    profile.
                    architecture_family
                ),

                "multiprocessor_count": (
                    profile.
                    multiprocessor_count
                ),

                "matrix_acceleration_candidate": (
                    profile.
                    matrix_acceleration_candidate
                ),

                "bfloat16_supported": (
                    profile.
                    bfloat16_supported
                ),

                "tf32_supported": (
                    profile.
                    tf32_supported
                ),

                "supported_precisions": list(
                    profile.
                    supported_precisions
                ),

                "metrics": (
                    optimization_metrics_attributes(
                        metrics
                    )
                ),

                "suitability": (
                    suitability.to_dict()
                ),

                "pytorch_uses_cuda_interface": (
                    True
                ),
            },
        )

        result.validate()

        return result

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
            ROCmDeviceProfile
        ] = None,
        metrics: Optional[
            ROCmMetricsSnapshot
        ] = None,
    ) -> OptimizationPlan:
        """
        Generate a complete advisory ROCm
        optimization plan.
        """

        require_optimizer_backend(
            selection,
            AcceleratorKind.ROCM.value,
        )

        workload.validate()

        resolved_constraints = (
            constraints
            or OptimizationConstraints()
        )

        resolved_constraints.validate()

        resolved_profile = (
            profile
            or self.inspector.inspect(
                selection.device,
                refresh_memory=True,
            )
        )

        resolved_profile.validate()

        collector = (
            self.metrics_collector
            or ROCmMetricsCollector(
                device=selection.device,
                backend=(
                    self.inspector.backend
                ),
                inspector=(
                    self.inspector
                ),
            )
        )

        resolved_metrics = (
            metrics
            or collector.collect(
                refresh_profile=False
            )
        )

        resolved_metrics.validate()

        suitability = (
            evaluate_rocm_workload(
                resolved_profile,
                workload,
            )
        )

        optimization_profile = (
            self._optimization_profile(
                resolved_profile,
                resolved_metrics,
                workload,
                suitability,
            )
        )

        if (
            resolved_constraints.level
            == OptimizationLevel.DISABLED.value
        ):
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
                    ROCM_OPTIMIZER_NAME
                ),

                attributes={
                    "optimizer_version": (
                        ROCM_OPTIMIZER_VERSION
                    ),

                    "optimization_disabled": (
                        True
                    ),

                    "pytorch_uses_cuda_interface": (
                        True
                    ),
                },
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
                        "ROCm workload "
                        "suitability check failed"
                    ),

                    reason=(
                        "The selected ROCm "
                        "profile does not satisfy "
                        "one or more declared "
                        "workload properties."
                    ),

                    current_value={
                        "reasons": list(
                            suitability.reasons
                        ),
                    },

                    recommended_value={
                        "review_workload_or_device": (
                            True
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

                    evidence=(
                        suitability.reasons
                    ),
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
            self._tf32_recommendations(
                resolved_profile,
                workload,
                resolved_constraints,
            )
        )

        recommendations.extend(
            self._transfer_recommendations(
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
                resolved_profile,
                workload,
                resolved_constraints,
            )
        )

        recommendations.extend(
            self._runtime_recommendations(
                resolved_metrics
            )
        )

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
                ROCM_OPTIMIZER_NAME
            ),

            attributes={
                "optimizer_version": (
                    ROCM_OPTIMIZER_VERSION
                ),

                "logical_device_id": (
                    selection.device_id
                ),

                "runtime_device": (
                    selection.runtime_device
                ),

                "pytorch_uses_cuda_interface": (
                    True
                ),

                "policy": {
                    "high_memory_pressure_ratio": (
                        self.policy.
                        high_memory_pressure_ratio
                    ),

                    "moderate_memory_pressure_ratio": (
                        self.policy.
                        moderate_memory_pressure_ratio
                    ),

                    "low_memory_pressure_ratio": (
                        self.policy.
                        low_memory_pressure_ratio
                    ),

                    "high_device_utilization_percent": (
                        self.policy.
                        high_device_utilization_percent
                    ),

                    "low_device_utilization_percent": (
                        self.policy.
                        low_device_utilization_percent
                    ),

                    "high_temperature_celsius": (
                        self.policy.
                        high_temperature_celsius
                    ),
                },

                "automatically_applied": (
                    False
                ),
            },
        )


# ------------------------------------------------------------------
# Convenience API
# ------------------------------------------------------------------

def recommend_rocm_optimizations(
    *,
    selection: AcceleratorSelection,
    workload: AcceleratorWorkload,
    constraints: Optional[
        OptimizationConstraints
    ] = None,
    profile: Optional[
        ROCmDeviceProfile
    ] = None,
    metrics: Optional[
        ROCmMetricsSnapshot
    ] = None,
    policy: Optional[
        ROCmOptimizationPolicy
    ] = None,
) -> OptimizationPlan:
    return ROCmOptimizer(
        policy=policy
    ).recommend(
        selection=selection,
        workload=workload,
        constraints=constraints,
        profile=profile,
        metrics=metrics,
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "ROCM_OPTIMIZER_NAME",
    "ROCM_OPTIMIZER_VERSION",

    "ROCmOptimizationPolicy",

    "recommendation",
    "constrained_batch_size",
    "allocator_inactive_ratio",
    "effective_memory_limit",

    "ROCmOptimizer",

    "recommend_rocm_optimizations",
]
