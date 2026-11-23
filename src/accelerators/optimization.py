from __future__ import annotations

"""
Framework-neutral optimization contracts for OSHA.

Flow:

    workload description
          ↓
    AcceleratorSelection
          ↓
    backend optimizer
          ↓
    OptimizationPlan
          ↓
    training / AI decides what to apply

This module defines optimization contracts only.

It does not:

    modify PyTorch global state
    compile models
    change thread counts
    change batch sizes
    enable mixed precision
    allocate accelerator memory
    execute workloads

Backend optimizers produce recommendations.
Callers explicitly decide whether to apply them.
"""

from dataclasses import (
    dataclass,
    field,
)

from enum import Enum

import math

from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
)

from src.accelerators.schema import (
    AcceleratorDevice,
    AcceleratorKind,
    AcceleratorPrecision,
    AcceleratorSelection,
)

from src.core.serialization import (
    to_json_safe,
)

from src.core.time import (
    utc_now,
)


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

OPTIMIZATION_SCHEMA_VERSION = "1"


# ------------------------------------------------------------------
# Workload kind
# ------------------------------------------------------------------

class WorkloadKind(
    str,
    Enum,
):
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    EMBEDDING = "embedding"
    DATA_PROCESSING = "data_processing"
    CUSTOM = "custom"


# ------------------------------------------------------------------
# Optimization level
# ------------------------------------------------------------------

class OptimizationLevel(
    str,
    Enum,
):
    DISABLED = "disabled"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


# ------------------------------------------------------------------
# Recommendation severity
# ------------------------------------------------------------------

class RecommendationPriority(
    str,
    Enum,
):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ------------------------------------------------------------------
# Recommendation type
# ------------------------------------------------------------------

class RecommendationKind(
    str,
    Enum,
):
    PRECISION = "precision"
    MIXED_PRECISION = "mixed_precision"
    BATCH_SIZE = "batch_size"
    DATA_LOADER = "data_loader"
    PIN_MEMORY = "pin_memory"
    NON_BLOCKING_TRANSFER = (
        "non_blocking_transfer"
    )
    COMPILATION = "compilation"
    THREADING = "threading"
    MEMORY = "memory"
    CACHE = "cache"
    SYNCHRONIZATION = "synchronization"
    ALLOCATION = "allocation"
    WORKLOAD = "workload"
    WARNING = "warning"


# ------------------------------------------------------------------
# Recommendation confidence
# ------------------------------------------------------------------

class RecommendationConfidence(
    str,
    Enum,
):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ------------------------------------------------------------------
# Optimization application state
# ------------------------------------------------------------------

class RecommendationDisposition(
    str,
    Enum,
):
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    NOT_RECOMMENDED = "not_recommended"
    UNSUPPORTED = "unsupported"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _required_text(
    value: Any,
    *,
    name: str,
    maximum_length: int = 2048,
) -> str:
    if value is None:
        raise ValueError(
            f"{name} is required"
        )

    text = str(
        value
    ).strip()

    if not text:
        raise ValueError(
            f"{name} cannot be empty"
        )

    if len(
        text
    ) > maximum_length:
        raise ValueError(
            f"{name} exceeds maximum "
            f"length {maximum_length}"
        )

    return text


def _optional_non_negative_int(
    value: Any,
    *,
    name: str,
) -> Optional[int]:
    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{name} must be an integer"
        )

    try:
        result = int(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{name} must be an integer"
        ) from exc

    if result < 0:
        raise ValueError(
            f"{name} must be >= 0"
        )

    return result


def _optional_positive_int(
    value: Any,
    *,
    name: str,
) -> Optional[int]:
    result = _optional_non_negative_int(
        value,
        name=name,
    )

    if (
        result is not None
        and result < 1
    ):
        raise ValueError(
            f"{name} must be >= 1"
        )

    return result


def _optional_finite_number(
    value: Any,
    *,
    name: str,
    minimum: Optional[
        float
    ] = None,
) -> Optional[float]:
    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{name} must be numeric"
        )

    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(
        result
    ):
        raise ValueError(
            f"{name} must be finite"
        )

    if (
        minimum is not None
        and result < minimum
    ):
        raise ValueError(
            f"{name} must be >= {minimum}"
        )

    return result


def _unique_strings(
    values: Iterable[
        Any
    ],
) -> Tuple[str, ...]:
    output = []
    seen = set()

    for value in values:
        text = str(
            value
        ).strip()

        if not text:
            continue

        if text in seen:
            continue

        seen.add(
            text
        )

        output.append(
            text
        )

    return tuple(
        output
    )


# ------------------------------------------------------------------
# Workload description
# ------------------------------------------------------------------

@dataclass(frozen=True)
class AcceleratorWorkload:
    """
    Framework-neutral description of the workload
    that will run on an accelerator.

    The optimizer uses known workload properties
    to produce recommendations.

    Unknown values remain None.
    """

    kind: str

    batch_size: Optional[
        int
    ] = None

    model_parameter_count: Optional[
        int
    ] = None

    estimated_model_bytes: Optional[
        int
    ] = None

    estimated_input_bytes: Optional[
        int
    ] = None

    estimated_peak_memory_bytes: Optional[
        int
    ] = None

    sequence_length: Optional[
        int
    ] = None

    precision: str = (
        AcceleratorPrecision.FLOAT32.value
    )

    mixed_precision_enabled: bool = False

    gradient_accumulation_steps: int = 1

    data_loader_workers: Optional[
        int
    ] = None

    pin_memory_enabled: Optional[
        bool
    ] = None

    non_blocking_transfers: Optional[
        bool
    ] = None

    compiled: bool = False

    latency_sensitive: bool = False

    throughput_sensitive: bool = False

    deterministic: bool = False

    attributes: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        if (
            self.kind
            not in {
                item.value
                for item
                in WorkloadKind
            }
        ):
            raise ValueError(
                "Unsupported accelerator "
                f"workload kind: {self.kind}"
            )

        _optional_positive_int(
            self.batch_size,
            name="batch_size",
        )

        _optional_non_negative_int(
            self.model_parameter_count,
            name="model_parameter_count",
        )

        _optional_non_negative_int(
            self.estimated_model_bytes,
            name="estimated_model_bytes",
        )

        _optional_non_negative_int(
            self.estimated_input_bytes,
            name="estimated_input_bytes",
        )

        _optional_non_negative_int(
            self.estimated_peak_memory_bytes,
            name=(
                "estimated_peak_memory_bytes"
            ),
        )

        _optional_positive_int(
            self.sequence_length,
            name="sequence_length",
        )

        if (
            self.precision
            not in {
                item.value
                for item
                in AcceleratorPrecision
            }
        ):
            raise ValueError(
                "Unsupported workload "
                f"precision: {self.precision}"
            )

        if (
            isinstance(
                self.gradient_accumulation_steps,
                bool,
            )
            or self.gradient_accumulation_steps
            < 1
        ):
            raise ValueError(
                "gradient_accumulation_steps "
                "must be >= 1"
            )

        if self.data_loader_workers is not None:
            _optional_non_negative_int(
                self.data_loader_workers,
                name="data_loader_workers",
            )

    @property
    def training(
        self,
    ) -> bool:
        return (
            self.kind
            == WorkloadKind.TRAINING.value
        )

    @property
    def inference(
        self,
    ) -> bool:
        return (
            self.kind
            == WorkloadKind.INFERENCE.value
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "kind": self.kind,

            "batch_size": (
                self.batch_size
            ),

            "model_parameter_count": (
                self.model_parameter_count
            ),

            "estimated_model_bytes": (
                self.estimated_model_bytes
            ),

            "estimated_input_bytes": (
                self.estimated_input_bytes
            ),

            "estimated_peak_memory_bytes": (
                self.
                estimated_peak_memory_bytes
            ),

            "sequence_length": (
                self.sequence_length
            ),

            "precision": self.precision,

            "mixed_precision_enabled": (
                self.mixed_precision_enabled
            ),

            "gradient_accumulation_steps": (
                self.
                gradient_accumulation_steps
            ),

            "data_loader_workers": (
                self.data_loader_workers
            ),

            "pin_memory_enabled": (
                self.pin_memory_enabled
            ),

            "non_blocking_transfers": (
                self.non_blocking_transfers
            ),

            "compiled": self.compiled,

            "latency_sensitive": (
                self.latency_sensitive
            ),

            "throughput_sensitive": (
                self.throughput_sensitive
            ),

            "deterministic": (
                self.deterministic
            ),

            "attributes": (
                to_json_safe(
                    dict(
                        self.attributes
                    )
                )
            ),
        }


# ------------------------------------------------------------------
# Optimization constraints
# ------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizationConstraints:
    """
    Safety boundaries supplied to backend
    optimizers.

    Recommendations must remain within these
    caller-controlled limits.
    """

    level: str = (
        OptimizationLevel.BALANCED.value
    )

    allow_precision_change: bool = True

    allow_mixed_precision: bool = True

    allow_batch_size_change: bool = True

    allow_compilation: bool = True

    allow_thread_change: bool = True

    allow_data_loader_change: bool = True

    allow_memory_tuning: bool = True

    minimum_batch_size: Optional[
        int
    ] = None

    maximum_batch_size: Optional[
        int
    ] = None

    maximum_memory_fraction: Optional[
        float
    ] = None

    required_precision: Optional[
        str
    ] = None

    attributes: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        if (
            self.level
            not in {
                item.value
                for item
                in OptimizationLevel
            }
        ):
            raise ValueError(
                "Unknown optimization level: "
                f"{self.level}"
            )

        minimum = _optional_positive_int(
            self.minimum_batch_size,
            name="minimum_batch_size",
        )

        maximum = _optional_positive_int(
            self.maximum_batch_size,
            name="maximum_batch_size",
        )

        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            raise ValueError(
                "minimum_batch_size cannot "
                "exceed maximum_batch_size"
            )

        fraction = _optional_finite_number(
            self.maximum_memory_fraction,
            name="maximum_memory_fraction",
            minimum=0.0,
        )

        if (
            fraction is not None
            and fraction > 1.0
        ):
            raise ValueError(
                "maximum_memory_fraction "
                "must be <= 1"
            )

        if self.required_precision is not None:
            if (
                self.required_precision
                not in {
                    item.value
                    for item
                    in AcceleratorPrecision
                }
            ):
                raise ValueError(
                    "Unsupported required "
                    "precision."
                )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "level": self.level,

            "allow_precision_change": (
                self.allow_precision_change
            ),

            "allow_mixed_precision": (
                self.allow_mixed_precision
            ),

            "allow_batch_size_change": (
                self.allow_batch_size_change
            ),

            "allow_compilation": (
                self.allow_compilation
            ),

            "allow_thread_change": (
                self.allow_thread_change
            ),

            "allow_data_loader_change": (
                self.allow_data_loader_change
            ),

            "allow_memory_tuning": (
                self.allow_memory_tuning
            ),

            "minimum_batch_size": (
                self.minimum_batch_size
            ),

            "maximum_batch_size": (
                self.maximum_batch_size
            ),

            "maximum_memory_fraction": (
                self.maximum_memory_fraction
            ),

            "required_precision": (
                self.required_precision
            ),

            "attributes": (
                to_json_safe(
                    dict(
                        self.attributes
                    )
                )
            ),
        }


# ------------------------------------------------------------------
# Optimization recommendation
# ------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizationRecommendation:
    """
    One backend optimization recommendation.

    current_value and recommended_value are
    deliberately generic so recommendations can
    represent batch size, precision, flags,
    worker counts, memory policy, or other
    backend-neutral values.
    """

    kind: str

    title: str

    reason: str

    recommended_value: Any = None

    current_value: Any = None

    priority: str = (
        RecommendationPriority.MEDIUM.value
    )

    confidence: str = (
        RecommendationConfidence.MEDIUM.value
    )

    disposition: str = (
        RecommendationDisposition.RECOMMENDED.value
    )

    expected_effect: Optional[
        str
    ] = None

    risks: Tuple[
        str,
        ...,
    ] = ()

    prerequisites: Tuple[
        str,
        ...,
    ] = ()

    evidence: Tuple[
        str,
        ...,
    ] = ()

    automatically_applicable: bool = False

    attributes: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        if (
            self.kind
            not in {
                item.value
                for item
                in RecommendationKind
            }
        ):
            raise ValueError(
                "Unsupported recommendation "
                f"kind: {self.kind}"
            )

        _required_text(
            self.title,
            name="recommendation title",
        )

        _required_text(
            self.reason,
            name="recommendation reason",
            maximum_length=4096,
        )

        if (
            self.priority
            not in {
                item.value
                for item
                in RecommendationPriority
            }
        ):
            raise ValueError(
                "Unsupported recommendation "
                f"priority: {self.priority}"
            )

        if (
            self.confidence
            not in {
                item.value
                for item
                in RecommendationConfidence
            }
        ):
            raise ValueError(
                "Unsupported recommendation "
                f"confidence: {self.confidence}"
            )

        if (
            self.disposition
            not in {
                item.value
                for item
                in RecommendationDisposition
            }
        ):
            raise ValueError(
                "Unsupported recommendation "
                "disposition: "
                f"{self.disposition}"
            )

        _unique_strings(
            self.risks
        )

        _unique_strings(
            self.prerequisites
        )

        _unique_strings(
            self.evidence
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "kind": self.kind,

            "title": self.title,

            "reason": self.reason,

            "current_value": (
                to_json_safe(
                    self.current_value
                )
            ),

            "recommended_value": (
                to_json_safe(
                    self.recommended_value
                )
            ),

            "priority": self.priority,

            "confidence": (
                self.confidence
            ),

            "disposition": (
                self.disposition
            ),

            "expected_effect": (
                self.expected_effect
            ),

            "risks": list(
                self.risks
            ),

            "prerequisites": list(
                self.prerequisites
            ),

            "evidence": list(
                self.evidence
            ),

            "automatically_applicable": (
                self.
                automatically_applicable
            ),

            "attributes": (
                to_json_safe(
                    dict(
                        self.attributes
                    )
                )
            ),
        }


# ------------------------------------------------------------------
# Optimization profile
# ------------------------------------------------------------------

@dataclass(frozen=True)
class AcceleratorOptimizationProfile:
    """
    Snapshot of the selected device information
    used while deriving optimization advice.
    """

    device: AcceleratorDevice

    workload: AcceleratorWorkload

    free_memory_bytes: Optional[
        int
    ] = None

    total_memory_bytes: Optional[
        int
    ] = None

    utilization_percent: Optional[
        float
    ] = None

    temperature_celsius: Optional[
        float
    ] = None

    capabilities: Tuple[
        str,
        ...,
    ] = ()

    attributes: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        self.device.validate()

        self.workload.validate()

        _optional_non_negative_int(
            self.free_memory_bytes,
            name="free_memory_bytes",
        )

        _optional_non_negative_int(
            self.total_memory_bytes,
            name="total_memory_bytes",
        )

        utilization = (
            _optional_finite_number(
                self.utilization_percent,
                name="utilization_percent",
                minimum=0.0,
            )
        )

        if (
            utilization is not None
            and utilization > 100.0
        ):
            raise ValueError(
                "utilization_percent "
                "must be <= 100"
            )

        _optional_finite_number(
            self.temperature_celsius,
            name="temperature_celsius",
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "device": (
                self.device.to_dict()
            ),

            "workload": (
                self.workload.to_dict()
            ),

            "free_memory_bytes": (
                self.free_memory_bytes
            ),

            "total_memory_bytes": (
                self.total_memory_bytes
            ),

            "utilization_percent": (
                self.utilization_percent
            ),

            "temperature_celsius": (
                self.temperature_celsius
            ),

            "capabilities": list(
                self.capabilities
            ),

            "attributes": (
                to_json_safe(
                    dict(
                        self.attributes
                    )
                )
            ),
        }


# ------------------------------------------------------------------
# Optimization plan
# ------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizationPlan:
    """
    Backend-generated optimization plan.

    The plan is advisory by default.

    No recommendation is automatically applied
    merely because it appears here.
    """

    selection: AcceleratorSelection

    workload: AcceleratorWorkload

    constraints: OptimizationConstraints

    recommendations: Tuple[
        OptimizationRecommendation,
        ...,
    ]

    profile: Optional[
        AcceleratorOptimizationProfile
    ] = None

    optimizer_name: Optional[
        str
    ] = None

    optimizer_version: str = (
        OPTIMIZATION_SCHEMA_VERSION
    )

    generated_at: str = field(
        default_factory=utc_now
    )

    attributes: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        self.selection.validate()

        self.workload.validate()

        self.constraints.validate()

        if (
            self.profile is not None
        ):
            self.profile.validate()

            if (
                self.profile.device.device_id
                != self.selection.device_id
            ):
                raise ValueError(
                    "Optimization profile "
                    "device does not match "
                    "the selected device."
                )

        for recommendation in (
            self.recommendations
        ):
            recommendation.validate()

        _required_text(
            self.optimizer_version,
            name="optimizer_version",
            maximum_length=128,
        )

        _required_text(
            self.generated_at,
            name="generated_at",
            maximum_length=128,
        )

    @property
    def device_id(
        self,
    ) -> str:
        return self.selection.device_id

    @property
    def backend(
        self,
    ) -> str:
        return self.selection.kind

    @property
    def recommendation_count(
        self,
    ) -> int:
        return len(
            self.recommendations
        )

    @property
    def high_priority_count(
        self,
    ) -> int:
        return sum(
            1
            for recommendation
            in self.recommendations
            if recommendation.priority
            in {
                RecommendationPriority.HIGH.value,
                RecommendationPriority.CRITICAL.value,
            }
        )

    def by_kind(
        self,
        kind: str,
    ) -> Tuple[
        OptimizationRecommendation,
        ...,
    ]:
        return tuple(
            recommendation
            for recommendation
            in self.recommendations
            if recommendation.kind
            == kind
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "schema_version": (
                OPTIMIZATION_SCHEMA_VERSION
            ),

            "device_id": (
                self.device_id
            ),

            "backend": (
                self.backend
            ),

            "selection": (
                self.selection.to_dict()
            ),

            "workload": (
                self.workload.to_dict()
            ),

            "constraints": (
                self.constraints.to_dict()
            ),

            "profile": (
                self.profile.to_dict()
                if self.profile
                is not None
                else None
            ),

            "recommendation_count": (
                self.recommendation_count
            ),

            "high_priority_count": (
                self.high_priority_count
            ),

            "recommendations": [
                recommendation.to_dict()
                for recommendation
                in self.recommendations
            ],

            "optimizer_name": (
                self.optimizer_name
            ),

            "optimizer_version": (
                self.optimizer_version
            ),

            "generated_at": (
                self.generated_at
            ),

            "attributes": (
                to_json_safe(
                    dict(
                        self.attributes
                    )
                )
            ),

            "automatically_applied": (
                False
            ),
        }


# ------------------------------------------------------------------
# Optimization outcome
# ------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizationOutcome:
    """
    Records what a caller actually applied from an
    OptimizationPlan.

    This keeps recommendations separate from
    mutations.
    """

    plan: OptimizationPlan

    applied_recommendations: Tuple[
        str,
        ...,
    ] = ()

    skipped_recommendations: Tuple[
        str,
        ...,
    ] = ()

    failed_recommendations: Tuple[
        str,
        ...,
    ] = ()

    attributes: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    completed_at: str = field(
        default_factory=utc_now
    )

    def validate(
        self,
    ) -> None:
        self.plan.validate()

        _unique_strings(
            self.applied_recommendations
        )

        _unique_strings(
            self.skipped_recommendations
        )

        _unique_strings(
            self.failed_recommendations
        )

        _required_text(
            self.completed_at,
            name="completed_at",
            maximum_length=128,
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "schema_version": (
                OPTIMIZATION_SCHEMA_VERSION
            ),

            "plan": (
                self.plan.to_dict()
            ),

            "applied_recommendations": list(
                self.applied_recommendations
            ),

            "skipped_recommendations": list(
                self.skipped_recommendations
            ),

            "failed_recommendations": list(
                self.failed_recommendations
            ),

            "completed_at": (
                self.completed_at
            ),

            "attributes": (
                to_json_safe(
                    dict(
                        self.attributes
                    )
                )
            ),
        }


# ------------------------------------------------------------------
# Plan builder
# ------------------------------------------------------------------

def build_optimization_plan(
    *,
    selection: AcceleratorSelection,
    workload: AcceleratorWorkload,
    recommendations: Iterable[
        OptimizationRecommendation
    ],
    constraints: Optional[
        OptimizationConstraints
    ] = None,
    profile: Optional[
        AcceleratorOptimizationProfile
    ] = None,
    optimizer_name: Optional[
        str
    ] = None,
    attributes: Optional[
        Mapping[
            str,
            Any,
        ]
    ] = None,
) -> OptimizationPlan:
    plan = OptimizationPlan(
        selection=selection,

        workload=workload,

        constraints=(
            constraints
            or OptimizationConstraints()
        ),

        recommendations=tuple(
            recommendations
        ),

        profile=profile,

        optimizer_name=(
            optimizer_name
        ),

        attributes=dict(
            attributes
            or {}
        ),
    )

    plan.validate()

    return plan


# ------------------------------------------------------------------
# Backend compatibility
# ------------------------------------------------------------------

def require_optimizer_backend(
    selection: AcceleratorSelection,
    expected_kind: str,
) -> None:
    """
    Prevent a backend optimizer from processing a
    selection belonging to another backend.
    """

    selection.validate()

    expected = str(
        expected_kind
    ).strip().lower()

    if expected not in {
        AcceleratorKind.CPU.value,
        AcceleratorKind.CUDA.value,
        AcceleratorKind.ROCM.value,
    }:
        raise ValueError(
            "Unsupported optimizer backend: "
            f"{expected}"
        )

    if selection.kind != expected:
        raise ValueError(
            "Optimization backend mismatch: "
            f"expected {expected}, "
            f"received {selection.kind}"
        )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

__all__ = [
    "OPTIMIZATION_SCHEMA_VERSION",

    "WorkloadKind",
    "OptimizationLevel",

    "RecommendationPriority",
    "RecommendationKind",
    "RecommendationConfidence",
    "RecommendationDisposition",

    "AcceleratorWorkload",
    "OptimizationConstraints",

    "OptimizationRecommendation",
    "AcceleratorOptimizationProfile",

    "OptimizationPlan",
    "OptimizationOutcome",

    "build_optimization_plan",
    "require_optimizer_backend",
]
