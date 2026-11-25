from __future__ import annotations

"""
Configuration for OSHA model training,
evaluation, checkpointing, and artifact
promotion.
"""

import os

from dataclasses import (
    dataclass,
    field,
)

from pathlib import Path

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Tuple,
)

from src.core.validation import (
    boolean,
    integer,
    number,
    unique_strings,
)

from src.training.schema import (
    TrainingTask,
)


# ---------------------------------------------------------------
# Environment prefix
# ---------------------------------------------------------------

DEFAULT_ENV_PREFIX = "OSHA_TRAINING_"


# ---------------------------------------------------------------
# Accelerator policy
# ---------------------------------------------------------------

SUPPORTED_ACCELERATORS = (
    "auto",
    "cuda",
    "rocm",
    "cpu",
)


# ---------------------------------------------------------------
# Training paths
# ---------------------------------------------------------------


@dataclass(frozen=True)
class TrainingPathConfig:
    """
    Filesystem paths used during training.

    These paths are artifact locations, not
    Python source-code directories.
    """

    prepared_data_directory: str = "data/prepared"

    orchestration_data_directory: str = "orchestration-ai-data"

    checkpoint_directory: str = "checkpoints"

    artifact_directory: str = "artifacts"

    model_directory: str = "models"

    log_directory: str = "logs/training"

    def validate(
        self,
    ) -> None:
        for name, value in (
            (
                "prepared_data_directory",
                self.prepared_data_directory,
            ),
            (
                "orchestration_data_directory",
                self.orchestration_data_directory,
            ),
            (
                "checkpoint_directory",
                self.checkpoint_directory,
            ),
            (
                "artifact_directory",
                self.artifact_directory,
            ),
            (
                "model_directory",
                self.model_directory,
            ),
            (
                "log_directory",
                self.log_directory,
            ),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} cannot be empty")

    def resolved(
        self,
    ) -> Dict[str, str]:
        self.validate()

        return {
            "prepared_data": str(
                Path(self.prepared_data_directory).expanduser().resolve()
            ),
            "orchestration_data": str(
                Path(self.orchestration_data_directory).expanduser().resolve()
            ),
            "checkpoints": str(Path(self.checkpoint_directory).expanduser().resolve()),
            "artifacts": str(Path(self.artifact_directory).expanduser().resolve()),
            "models": str(Path(self.model_directory).expanduser().resolve()),
            "logs": str(Path(self.log_directory).expanduser().resolve()),
        }


# ---------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------


@dataclass(frozen=True)
class ReproducibilityConfig:
    """
    Shared random-seed and deterministic-training
    policy.
    """

    random_seed: int = 42

    deterministic: bool = True

    benchmark: bool = False

    def validate(
        self,
    ) -> None:
        integer(
            self.random_seed,
            name="random_seed",
            minimum=0,
        )

        if self.deterministic and self.benchmark:
            raise ValueError(
                "deterministic and "
                "benchmark should not both "
                "be enabled in the common "
                "training policy"
            )


# ---------------------------------------------------------------
# Optimization configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationConfig:
    """
    Common optimizer/training-loop defaults.

    Individual event/anomaly models can override
    these values when their architecture requires
    different settings.
    """

    batch_size: int = 64

    epochs: int = 20

    learning_rate: float = 0.001

    weight_decay: float = 0.0

    gradient_clip_norm: Optional[float] = 1.0

    accumulation_steps: int = 1

    def validate(
        self,
    ) -> None:
        integer(
            self.batch_size,
            name="batch_size",
            minimum=1,
        )

        integer(
            self.epochs,
            name="epochs",
            minimum=1,
        )

        number(
            self.learning_rate,
            name="learning_rate",
            minimum=1e-12,
        )

        number(
            self.weight_decay,
            name="weight_decay",
            minimum=0.0,
        )

        if self.gradient_clip_norm is not None:
            number(
                self.gradient_clip_norm,
                name=("gradient_clip_norm"),
                minimum=0.0,
            )

        integer(
            self.accumulation_steps,
            name=("accumulation_steps"),
            minimum=1,
        )


# ---------------------------------------------------------------
# Accelerator configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class AcceleratorConfig:
    """
    Training accelerator selection.

    src/accelerators/ remains responsible for
    actual hardware detection and backend
    capabilities.
    """

    preferred: str = "auto"

    allow_fallback_to_cpu: bool = True

    mixed_precision: bool = False

    device_index: Optional[int] = None

    def validate(
        self,
    ) -> None:
        normalized = str(self.preferred).strip().lower()

        if normalized not in SUPPORTED_ACCELERATORS:
            raise ValueError(f"Unsupported accelerator: {self.preferred}")

        if self.device_index is not None:
            integer(
                self.device_index,
                name="device_index",
                minimum=0,
            )


# ---------------------------------------------------------------
# Checkpoint policy
# ---------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointConfig:
    enabled: bool = True

    save_every_epochs: int = 1

    keep_last: int = 5

    keep_best: bool = True

    monitor_metric: str = "validation_loss"

    mode: str = "min"

    save_optimizer_state: bool = True

    def validate(
        self,
    ) -> None:
        integer(
            self.save_every_epochs,
            name="save_every_epochs",
            minimum=1,
        )

        integer(
            self.keep_last,
            name="keep_last",
            minimum=1,
        )

        if self.mode not in {
            "min",
            "max",
        }:
            raise ValueError("checkpoint mode must be min or max")

        if not str(self.monitor_metric).strip():
            raise ValueError("monitor_metric cannot be empty")


# ---------------------------------------------------------------
# Evaluation policy
# ---------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Common evaluation behavior.

    Task-specific metrics are supplied by event
    and anomaly evaluators.
    """

    enabled: bool = True

    evaluate_validation: bool = True

    evaluate_test: bool = True

    require_test_before_promotion: bool = True

    fail_on_non_finite_metric: bool = True

    thresholds: Dict[
        str,
        float,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        for name, value in self.thresholds.items():
            if not str(name).strip():
                raise ValueError("Evaluation threshold name cannot be empty")

            float(value)


# ---------------------------------------------------------------
# Promotion policy
# ---------------------------------------------------------------


@dataclass(frozen=True)
class PromotionConfig:
    """
    Controls whether evaluated training artifacts
    can be promoted into models/.

    Promotion is explicit by default.
    """

    enabled: bool = True

    automatic: bool = False

    require_evaluation_pass: bool = True

    require_checksum: bool = True

    allow_replace_promoted_model: bool = False

    def validate(
        self,
    ) -> None:
        if self.automatic and not self.enabled:
            raise ValueError("automatic promotion requires promotion to be enabled")

        if self.automatic and not self.require_evaluation_pass:
            raise ValueError("automatic promotion requires an evaluation pass")


# ---------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class TrainingSourceConfig:
    """
    Controls which categories of prepared and
    operational data are eligible for training.

    Eligibility does not mean that every record
    is automatically selected. sources.py and
    assembly.py still perform explicit loading and
    filtering.
    """

    include_prepared_datasets: bool = True

    include_processed_operational_data: bool = True

    include_normalized_telemetry: bool = True

    include_correlated_data: bool = True

    include_intelligence_data: bool = True

    include_decision_data: bool = False

    include_remediation_data: bool = True

    include_memory_data: bool = False

    include_live_data: bool = False

    def validate(
        self,
    ) -> None:
        # Live production data should not enter
        # training accidentally.
        if self.include_live_data:
            raise ValueError(
                "Direct live-data training "
                "is disabled by the common "
                "training policy. Promote "
                "live observations through "
                "the processed data path "
                "first."
            )


# ---------------------------------------------------------------
# Main training configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    enabled: bool = True

    allowed_tasks: Tuple[
        str,
        ...,
    ] = (
        TrainingTask.EVENT.value,
        TrainingTask.ANOMALY.value,
        TrainingTask.INCIDENT_ROOT_CAUSE.value,
        TrainingTask.DECISION_POLICY_RISK.value,
        TrainingTask.REMEDIATION_RECOMMENDATION.value,
    )

    paths: TrainingPathConfig = field(default_factory=(TrainingPathConfig))

    reproducibility: ReproducibilityConfig = field(
        default_factory=(ReproducibilityConfig)
    )

    optimization: OptimizationConfig = field(default_factory=(OptimizationConfig))

    accelerator: AcceleratorConfig = field(default_factory=(AcceleratorConfig))

    checkpoint: CheckpointConfig = field(default_factory=(CheckpointConfig))

    evaluation: EvaluationConfig = field(default_factory=(EvaluationConfig))

    promotion: PromotionConfig = field(default_factory=(PromotionConfig))

    sources: TrainingSourceConfig = field(default_factory=(TrainingSourceConfig))

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        self.paths.validate()

        self.reproducibility.validate()

        self.optimization.validate()

        self.accelerator.validate()

        self.checkpoint.validate()

        self.evaluation.validate()

        self.promotion.validate()

        self.sources.validate()

        tasks = unique_strings(
            self.allowed_tasks,
            name="allowed_tasks",
            allow_empty=False,
        )

        valid_tasks = {item.value for item in TrainingTask}

        unsupported = [task for task in tasks if task not in valid_tasks]

        if unsupported:
            raise ValueError("Unsupported training tasks: " + ", ".join(unsupported))

    # ---------------------------------------------------------
    # Task allowed
    # ---------------------------------------------------------

    def task_allowed(
        self,
        task: str,
    ) -> bool:
        self.validate()

        return self.enabled and str(task).strip().lower() in set(self.allowed_tasks)

    # ---------------------------------------------------------
    # Dictionary
    # ---------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "enabled": (self.enabled),
            "allowed_tasks": list(self.allowed_tasks),
            "paths": (self.paths.resolved()),
            "reproducibility": {
                "random_seed": (self.reproducibility.random_seed),
                "deterministic": (self.reproducibility.deterministic),
                "benchmark": (self.reproducibility.benchmark),
            },
            "optimization": {
                "batch_size": (self.optimization.batch_size),
                "epochs": (self.optimization.epochs),
                "learning_rate": (self.optimization.learning_rate),
                "weight_decay": (self.optimization.weight_decay),
                "gradient_clip_norm": (self.optimization.gradient_clip_norm),
                "accumulation_steps": (self.optimization.accumulation_steps),
            },
            "accelerator": {
                "preferred": (self.accelerator.preferred),
                "allow_fallback_to_cpu": (self.accelerator.allow_fallback_to_cpu),
                "mixed_precision": (self.accelerator.mixed_precision),
                "device_index": (self.accelerator.device_index),
            },
            "checkpoint": {
                "enabled": (self.checkpoint.enabled),
                "save_every_epochs": (self.checkpoint.save_every_epochs),
                "keep_last": (self.checkpoint.keep_last),
                "keep_best": (self.checkpoint.keep_best),
                "monitor_metric": (self.checkpoint.monitor_metric),
                "mode": (self.checkpoint.mode),
                "save_optimizer_state": (self.checkpoint.save_optimizer_state),
            },
            "evaluation": {
                "enabled": (self.evaluation.enabled),
                "evaluate_validation": (self.evaluation.evaluate_validation),
                "evaluate_test": (self.evaluation.evaluate_test),
                "require_test_before_promotion": (
                    self.evaluation.require_test_before_promotion
                ),
                "fail_on_non_finite_metric": (
                    self.evaluation.fail_on_non_finite_metric
                ),
                "thresholds": dict(self.evaluation.thresholds),
            },
            "promotion": {
                "enabled": (self.promotion.enabled),
                "automatic": (self.promotion.automatic),
                "require_evaluation_pass": (self.promotion.require_evaluation_pass),
                "require_checksum": (self.promotion.require_checksum),
                "allow_replace_promoted_model": (
                    self.promotion.allow_replace_promoted_model
                ),
            },
            "sources": {
                "prepared_datasets": (self.sources.include_prepared_datasets),
                "processed_operational_data": (
                    self.sources.include_processed_operational_data
                ),
                "normalized_telemetry": (self.sources.include_normalized_telemetry),
                "correlated_data": (self.sources.include_correlated_data),
                "intelligence_data": (self.sources.include_intelligence_data),
                "decision_data": (self.sources.include_decision_data),
                "remediation_data": (self.sources.include_remediation_data),
                "memory_data": (self.sources.include_memory_data),
                "live_data": (self.sources.include_live_data),
            },
            "attributes": dict(self.attributes),
        }


# ---------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------


def _get(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Any = None,
) -> Any:
    return values.get(
        prefix + name,
        default,
    )


def _text(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Optional[str] = None,
) -> Optional[str]:
    value = _get(
        values,
        prefix,
        name,
        default,
    )

    if value is None:
        return None

    text = str(value).strip()

    return text or None


def _bool(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: bool,
) -> bool:
    value = _get(
        values,
        prefix,
        name,
    )

    if value is None or str(value).strip() == "":
        return default

    return boolean(
        value,
        name=prefix + name,
    )


def _int(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: int,
    *,
    minimum: int = 0,
) -> int:
    value = _get(
        values,
        prefix,
        name,
    )

    if value is None or str(value).strip() == "":
        return default

    return integer(
        value,
        name=prefix + name,
        minimum=minimum,
    )


def _float(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: float,
    *,
    minimum: float = 0.0,
) -> float:
    value = _get(
        values,
        prefix,
        name,
    )

    if value is None or str(value).strip() == "":
        return default

    return number(
        value,
        name=prefix + name,
        minimum=minimum,
    )


def _optional_float(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Optional[float],
) -> Optional[float]:
    value = _get(
        values,
        prefix,
        name,
    )

    if value is None or str(value).strip() == "":
        return default

    return number(
        value,
        name=prefix + name,
        minimum=0.0,
    )


def _optional_int(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Optional[int],
) -> Optional[int]:
    value = _get(
        values,
        prefix,
        name,
    )

    if value is None or str(value).strip() == "":
        return default

    return integer(
        value,
        name=prefix + name,
        minimum=0,
    )


def _csv(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Tuple[
        str,
        ...,
    ],
) -> Tuple[
    str,
    ...,
]:
    value = _get(
        values,
        prefix,
        name,
    )

    if value is None or str(value).strip() == "":
        return default

    return tuple(
        item.lower()
        for item in unique_strings(
            str(value).split(","),
            name=prefix + name,
            allow_empty=False,
        )
    )


# ---------------------------------------------------------------
# Mapping loader
# ---------------------------------------------------------------


def from_mapping(
    values: Mapping[
        str,
        Any,
    ],
    *,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> TrainingConfig:
    defaults = TrainingConfig()

    paths = TrainingPathConfig(
        prepared_data_directory=(
            _text(
                values,
                prefix,
                "PREPARED_DATA_DIRECTORY",
                defaults.paths.prepared_data_directory,
            )
            or defaults.paths.prepared_data_directory
        ),
        orchestration_data_directory=(
            _text(
                values,
                prefix,
                "ORCHESTRATION_DATA_DIRECTORY",
                defaults.paths.orchestration_data_directory,
            )
            or defaults.paths.orchestration_data_directory
        ),
        checkpoint_directory=(
            _text(
                values,
                prefix,
                "CHECKPOINT_DIRECTORY",
                defaults.paths.checkpoint_directory,
            )
            or defaults.paths.checkpoint_directory
        ),
        artifact_directory=(
            _text(
                values,
                prefix,
                "ARTIFACT_DIRECTORY",
                defaults.paths.artifact_directory,
            )
            or defaults.paths.artifact_directory
        ),
        model_directory=(
            _text(
                values,
                prefix,
                "MODEL_DIRECTORY",
                defaults.paths.model_directory,
            )
            or defaults.paths.model_directory
        ),
        log_directory=(
            _text(
                values,
                prefix,
                "LOG_DIRECTORY",
                defaults.paths.log_directory,
            )
            or defaults.paths.log_directory
        ),
    )

    config = TrainingConfig(
        enabled=_bool(
            values,
            prefix,
            "ENABLED",
            True,
        ),
        allowed_tasks=_csv(
            values,
            prefix,
            "ALLOWED_TASKS",
            defaults.allowed_tasks,
        ),
        paths=paths,
        reproducibility=(
            ReproducibilityConfig(
                random_seed=_int(
                    values,
                    prefix,
                    "RANDOM_SEED",
                    42,
                    minimum=0,
                ),
                deterministic=_bool(
                    values,
                    prefix,
                    "DETERMINISTIC",
                    True,
                ),
                benchmark=_bool(
                    values,
                    prefix,
                    "BENCHMARK",
                    False,
                ),
            )
        ),
        optimization=(
            OptimizationConfig(
                batch_size=_int(
                    values,
                    prefix,
                    "BATCH_SIZE",
                    64,
                    minimum=1,
                ),
                epochs=_int(
                    values,
                    prefix,
                    "EPOCHS",
                    20,
                    minimum=1,
                ),
                learning_rate=_float(
                    values,
                    prefix,
                    "LEARNING_RATE",
                    0.001,
                    minimum=1e-12,
                ),
                weight_decay=_float(
                    values,
                    prefix,
                    "WEIGHT_DECAY",
                    0.0,
                ),
                gradient_clip_norm=(
                    _optional_float(
                        values,
                        prefix,
                        "GRADIENT_CLIP_NORM",
                        1.0,
                    )
                ),
                accumulation_steps=_int(
                    values,
                    prefix,
                    "ACCUMULATION_STEPS",
                    1,
                    minimum=1,
                ),
            )
        ),
        accelerator=(
            AcceleratorConfig(
                preferred=(
                    _text(
                        values,
                        prefix,
                        "ACCELERATOR",
                        "auto",
                    )
                    or "auto"
                ).lower(),
                allow_fallback_to_cpu=(
                    _bool(
                        values,
                        prefix,
                        "ALLOW_CPU_FALLBACK",
                        True,
                    )
                ),
                mixed_precision=_bool(
                    values,
                    prefix,
                    "MIXED_PRECISION",
                    False,
                ),
                device_index=(
                    _optional_int(
                        values,
                        prefix,
                        "DEVICE_INDEX",
                        None,
                    )
                ),
            )
        ),
        checkpoint=CheckpointConfig(
            enabled=_bool(
                values,
                prefix,
                "CHECKPOINT_ENABLED",
                True,
            ),
            save_every_epochs=_int(
                values,
                prefix,
                "CHECKPOINT_EVERY_EPOCHS",
                1,
                minimum=1,
            ),
            keep_last=_int(
                values,
                prefix,
                "CHECKPOINT_KEEP_LAST",
                5,
                minimum=1,
            ),
            keep_best=_bool(
                values,
                prefix,
                "CHECKPOINT_KEEP_BEST",
                True,
            ),
            monitor_metric=(
                _text(
                    values,
                    prefix,
                    "CHECKPOINT_MONITOR_METRIC",
                    "validation_loss",
                )
                or "validation_loss"
            ),
            mode=(
                _text(
                    values,
                    prefix,
                    "CHECKPOINT_MODE",
                    "min",
                )
                or "min"
            ).lower(),
            save_optimizer_state=_bool(
                values,
                prefix,
                "CHECKPOINT_SAVE_OPTIMIZER_STATE",
                True,
            ),
        ),
        evaluation=EvaluationConfig(
            enabled=_bool(
                values,
                prefix,
                "EVALUATION_ENABLED",
                True,
            ),
            evaluate_validation=_bool(
                values,
                prefix,
                "EVALUATE_VALIDATION",
                True,
            ),
            evaluate_test=_bool(
                values,
                prefix,
                "EVALUATE_TEST",
                True,
            ),
            require_test_before_promotion=(
                _bool(
                    values,
                    prefix,
                    "REQUIRE_TEST_BEFORE_PROMOTION",
                    True,
                )
            ),
            fail_on_non_finite_metric=(
                _bool(
                    values,
                    prefix,
                    "FAIL_ON_NON_FINITE_METRIC",
                    True,
                )
            ),
        ),
        promotion=PromotionConfig(
            enabled=_bool(
                values,
                prefix,
                "PROMOTION_ENABLED",
                True,
            ),
            automatic=_bool(
                values,
                prefix,
                "PROMOTION_AUTOMATIC",
                False,
            ),
            require_evaluation_pass=(
                _bool(
                    values,
                    prefix,
                    "PROMOTION_REQUIRE_EVALUATION_PASS",
                    True,
                )
            ),
            require_checksum=_bool(
                values,
                prefix,
                "PROMOTION_REQUIRE_CHECKSUM",
                True,
            ),
            allow_replace_promoted_model=(
                _bool(
                    values,
                    prefix,
                    "PROMOTION_ALLOW_REPLACE",
                    False,
                )
            ),
        ),
        sources=TrainingSourceConfig(
            include_prepared_datasets=(
                _bool(
                    values,
                    prefix,
                    "SOURCE_PREPARED_DATASETS",
                    True,
                )
            ),
            include_processed_operational_data=(
                _bool(
                    values,
                    prefix,
                    "SOURCE_PROCESSED_OPERATIONAL_DATA",
                    True,
                )
            ),
            include_normalized_telemetry=(
                _bool(
                    values,
                    prefix,
                    "SOURCE_NORMALIZED_TELEMETRY",
                    True,
                )
            ),
            include_correlated_data=(
                _bool(
                    values,
                    prefix,
                    "SOURCE_CORRELATED_DATA",
                    True,
                )
            ),
            include_intelligence_data=(
                _bool(
                    values,
                    prefix,
                    "SOURCE_INTELLIGENCE_DATA",
                    True,
                )
            ),
            include_decision_data=_bool(
                values,
                prefix,
                "SOURCE_DECISION_DATA",
                False,
            ),
            include_remediation_data=(
                _bool(
                    values,
                    prefix,
                    "SOURCE_REMEDIATION_DATA",
                    True,
                )
            ),
            include_memory_data=_bool(
                values,
                prefix,
                "SOURCE_MEMORY_DATA",
                False,
            ),
            include_live_data=_bool(
                values,
                prefix,
                "SOURCE_LIVE_DATA",
                False,
            ),
        ),
        attributes={
            "config_source": ("mapping"),
            "environment_prefix": (prefix),
        },
    )

    config.validate()

    return config


# ---------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------


def from_environment(
    *,
    environ: Optional[
        Mapping[
            str,
            str,
        ]
    ] = None,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> TrainingConfig:
    source = os.environ if environ is None else environ

    config = from_mapping(
        source,
        prefix=prefix,
    )

    return TrainingConfig(
        enabled=config.enabled,
        allowed_tasks=(config.allowed_tasks),
        paths=config.paths,
        reproducibility=(config.reproducibility),
        optimization=(config.optimization),
        accelerator=(config.accelerator),
        checkpoint=(config.checkpoint),
        evaluation=(config.evaluation),
        promotion=config.promotion,
        sources=config.sources,
        attributes={
            **config.attributes,
            "config_source": ("environment"),
        },
    )


# ---------------------------------------------------------------
# Safe summary
# ---------------------------------------------------------------


def safe_config_summary(
    config: TrainingConfig,
) -> Dict[str, Any]:
    config.validate()

    return {
        "enabled": (config.enabled),
        "allowed_tasks": list(config.allowed_tasks),
        "paths": (config.paths.resolved()),
        "random_seed": (config.reproducibility.random_seed),
        "deterministic": (config.reproducibility.deterministic),
        "batch_size": (config.optimization.batch_size),
        "epochs": (config.optimization.epochs),
        "learning_rate": (config.optimization.learning_rate),
        "accelerator": (config.accelerator.preferred),
        "cpu_fallback": (config.accelerator.allow_fallback_to_cpu),
        "mixed_precision": (config.accelerator.mixed_precision),
        "checkpoint_enabled": (config.checkpoint.enabled),
        "evaluation_enabled": (config.evaluation.enabled),
        "promotion_enabled": (config.promotion.enabled),
        "automatic_promotion": (config.promotion.automatic),
        "live_data_training_enabled": (config.sources.include_live_data),
    }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_ENV_PREFIX",
    "SUPPORTED_ACCELERATORS",
    "TrainingPathConfig",
    "ReproducibilityConfig",
    "OptimizationConfig",
    "AcceleratorConfig",
    "CheckpointConfig",
    "EvaluationConfig",
    "PromotionConfig",
    "TrainingSourceConfig",
    "TrainingConfig",
    "from_mapping",
    "from_environment",
    "safe_config_summary",
]
