from __future__ import annotations

"""
Shared training orchestration facade for OSHA.

This pipeline coordinates common training
infrastructure:

    source discovery
    dataset assembly
    leakage-safe splitting
    feature preparation
    run lifecycle
    checkpoints
    evaluations
    candidate artifacts
    model promotion
    health/readiness

The task-specific optimization loops remain in:

    train_events.py
    train_anomalies.py
"""

import builtins
import math
import os
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import (
    RLock,
)
from typing import (
    Any,
    Self,
)

from src.core.time import (
    utc_now,
)
from src.training.artifacts import (
    ModelArtifactManager,
    ModelArtifactRegistry,
)
from src.training.assembly import (
    AssemblyResult,
    ExampleBuilder,
    TrainingDatasetAssembler,
)
from src.training.checkpoints import (
    CheckpointManager,
    CheckpointRegistry,
)
from src.training.config import (
    TrainingConfig,
    safe_config_summary,
)
from src.training.errors import (
    EvaluationMetricError,
    EvaluationThresholdError,
    TrainingConfigurationError,
    TrainingNumericalError,
    TrainingStateError,
    TrainingTaskMismatchError,
)
from src.training.features import (
    FeatureBuilder,
    FeatureBuildResult,
    FeatureExtractor,
    FeatureExtractorRegistry,
    FeatureSchema,
)
from src.training.health import (
    TrainingHealth,
    TrainingHealthChecker,
)
from src.training.schema import (
    EvaluationResult,
    EvaluationSplit,
    ModelArtifact,
    TrainingCheckpoint,
    TrainingDataset,
    TrainingMetric,
    TrainingRun,
    TrainingState,
)
from src.training.sources import (
    DiscoveredTrainingSource,
    TrainingSourceManager,
)
from src.training.splits import (
    TrainingSplitResult,
    TrainingSplitter,
)

# ---------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------

try:
    import numpy as np

except ImportError:
    np = None


try:
    import torch

except ImportError:
    torch = None


# ---------------------------------------------------------------
# Training run registry
# ---------------------------------------------------------------


class TrainingRunRegistry:
    """
    Thread-safe in-memory registry of training
    runs.

    Durable run persistence can later be backed
    by src/database/ without changing the common
    training contracts.
    """

    def __init__(
        self,
    ) -> None:
        self._runs: dict[
            str,
            TrainingRun,
        ] = {}

        self._lock = RLock()

    # ---------------------------------------------------------
    # Register
    # ---------------------------------------------------------

    def register(
        self,
        run: TrainingRun,
        *,
        replace: bool = False,
    ) -> TrainingRun:
        run.finalize()

        run_id = str(run.run_id)

        with self._lock:
            if run_id in self._runs and not replace:
                raise TrainingStateError(
                    "Training run is already registered.",
                    details={
                        "run_id": run_id,
                    },
                )

            self._runs[run_id] = run

        return run

    # ---------------------------------------------------------
    # Get
    # ---------------------------------------------------------

    def get(
        self,
        run_id: str,
    ) -> TrainingRun | None:
        with self._lock:
            return self._runs.get(str(run_id))

    # ---------------------------------------------------------
    # Require
    # ---------------------------------------------------------

    def require(
        self,
        run_id: str,
    ) -> TrainingRun:
        run = self.get(run_id)

        if run is None:
            raise TrainingStateError(
                "Training run is not registered.",
                details={
                    "run_id": run_id,
                },
            )

        return run

    # ---------------------------------------------------------
    # List
    # ---------------------------------------------------------

    def list(
        self,
        *,
        task: str | None = None,
        state: str | None = None,
        model_name: str | None = None,
    ) -> builtins.list[TrainingRun]:
        with self._lock:
            values = list(self._runs.values())

        output: list[TrainingRun] = []

        for run in values:
            if task is not None and run.task != task:
                continue

            if state is not None and run.state != state:
                continue

            if model_name is not None and run.model_name != model_name:
                continue

            output.append(run)

        return sorted(
            output,
            key=lambda item: (
                item.started_at or "",
                item.run_id or "",
            ),
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> dict[str, Any]:
        runs = self.list()

        by_state: dict[
            str,
            int,
        ] = {}

        by_task: dict[
            str,
            int,
        ] = {}

        for run in runs:
            by_state[run.state] = (
                by_state.get(
                    run.state,
                    0,
                )
                + 1
            )

            by_task[run.task] = (
                by_task.get(
                    run.task,
                    0,
                )
                + 1
            )

        return {
            "run_count": len(runs),
            "by_state": (by_state),
            "by_task": (by_task),
        }


# ---------------------------------------------------------------
# Evaluation registry
# ---------------------------------------------------------------


class EvaluationRegistry:
    """
    Thread-safe in-memory evaluation registry.
    """

    def __init__(
        self,
    ) -> None:
        self._evaluations: dict[
            str,
            EvaluationResult,
        ] = {}

        self._lock = RLock()

    # ---------------------------------------------------------
    # Register
    # ---------------------------------------------------------

    def register(
        self,
        evaluation: EvaluationResult,
        *,
        replace: bool = False,
    ) -> EvaluationResult:
        evaluation.finalize()

        evaluation_id = str(evaluation.evaluation_id)

        with self._lock:
            if evaluation_id in self._evaluations and not replace:
                raise TrainingStateError(
                    "Evaluation is already registered.",
                    details={
                        "evaluation_id": (evaluation_id),
                    },
                )

            self._evaluations[evaluation_id] = evaluation

        return evaluation

    # ---------------------------------------------------------
    # Get
    # ---------------------------------------------------------

    def get(
        self,
        evaluation_id: str,
    ) -> EvaluationResult | None:
        with self._lock:
            return self._evaluations.get(str(evaluation_id))

    # ---------------------------------------------------------
    # Require
    # ---------------------------------------------------------

    def require(
        self,
        evaluation_id: str,
    ) -> EvaluationResult:
        evaluation = self.get(evaluation_id)

        if evaluation is None:
            raise TrainingStateError(
                "Evaluation is not registered.",
                details={
                    "evaluation_id": (evaluation_id),
                },
            )

        return evaluation

    # ---------------------------------------------------------
    # List
    # ---------------------------------------------------------

    def list(
        self,
        *,
        run_id: str | None = None,
        split: str | None = None,
        passed: bool | None = None,
    ) -> builtins.list[EvaluationResult]:
        with self._lock:
            values = list(self._evaluations.values())

        output: list[EvaluationResult] = []

        for evaluation in values:
            if run_id is not None and evaluation.run_id != run_id:
                continue

            if split is not None and evaluation.split != split:
                continue

            if passed is not None and evaluation.passed is not passed:
                continue

            output.append(evaluation)

        return sorted(
            output,
            key=lambda item: (
                item.evaluated_at,
                item.evaluation_id or "",
            ),
        )

    # ---------------------------------------------------------
    # Latest
    # ---------------------------------------------------------

    def latest(
        self,
        *,
        run_id: str,
        split: str | None = None,
    ) -> EvaluationResult | None:
        values = self.list(
            run_id=run_id,
            split=split,
        )

        if not values:
            return None

        return values[-1]

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> dict[str, Any]:
        values = self.list()

        passed = sum(1 for evaluation in values if evaluation.passed is True)

        failed = sum(1 for evaluation in values if evaluation.passed is False)

        return {
            "evaluation_count": len(values),
            "passed": passed,
            "failed": failed,
            "undetermined": (len(values) - passed - failed),
        }


# ---------------------------------------------------------------
# Training preparation result
# ---------------------------------------------------------------


class TrainingPreparation:
    """
    Container for assembled and leakage-safe
    training data.

    Features are intentionally not fitted here.
    Feature schemas must be fitted on the train
    partition only.
    """

    def __init__(
        self,
        *,
        assembly: AssemblyResult,
        split: TrainingSplitResult,
    ) -> None:
        self.assembly = assembly
        self.split = split

        self.split.validate()

    @property
    def dataset(
        self,
    ) -> TrainingDataset:
        return self.assembly.dataset

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "assembly": (self.assembly.to_dict()),
            "split": (self.split.to_dict()),
        }


# ---------------------------------------------------------------
# Encoded split result
# ---------------------------------------------------------------


class EncodedTrainingSplits:
    """
    Feature-schema result for train, validation,
    and test partitions.

    The schema is fitted from train examples only.
    """

    def __init__(
        self,
        *,
        schema: FeatureSchema,
        train: FeatureBuildResult,
        validation: FeatureBuildResult | None,
        test: FeatureBuildResult | None,
    ) -> None:
        self.schema = schema
        self.train = train
        self.validation = validation
        self.test = test

    def to_dict(
        self,
        *,
        include_examples: bool = False,
    ) -> dict[str, Any]:
        return {
            "schema": (self.schema.to_dict()),
            "train": (self.train.to_dict(include_examples=(include_examples))),
            "validation": (
                self.validation.to_dict(include_examples=(include_examples))
                if self.validation is not None
                else None
            ),
            "test": (
                self.test.to_dict(include_examples=(include_examples))
                if self.test is not None
                else None
            ),
        }


# ---------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------


class TrainingPipeline:
    """
    Common OSHA training facade.

    This class coordinates shared lifecycle and
    persistence boundaries.

    It does not implement task-specific forward,
    backward, optimizer, or loss calculations.
    """

    def __init__(
        self,
        *,
        config: TrainingConfig | None = None,
        sources: TrainingSourceManager | None = None,
        assembler: TrainingDatasetAssembler | None = None,
        splitter: TrainingSplitter | None = None,
        feature_builders: FeatureExtractorRegistry | None = None,
        checkpoints: CheckpointManager | None = None,
        artifacts: ModelArtifactManager | None = None,
        runs: TrainingRunRegistry | None = None,
        evaluations: EvaluationRegistry | None = None,
    ) -> None:
        self.config = config or TrainingConfig()

        self.config.validate()

        # -----------------------------------------------------
        # Shared services
        # -----------------------------------------------------

        self.sources = sources or TrainingSourceManager(config=self.config)

        self.assembler = assembler or TrainingDatasetAssembler(
            config=self.config,
            sources=self.sources,
        )

        self.splitter = splitter or TrainingSplitter(config=self.config)

        self.feature_extractors = feature_builders or FeatureExtractorRegistry()

        self.feature_builder = FeatureBuilder(registry=(self.feature_extractors))

        # -----------------------------------------------------
        # Lifecycle storage
        # -----------------------------------------------------

        self.checkpoints = checkpoints or CheckpointManager(
            config=self.config,
            registry=(CheckpointRegistry()),
        )

        self.artifacts = artifacts or ModelArtifactManager(
            config=self.config,
            registry=(ModelArtifactRegistry()),
        )

        self.runs = runs or TrainingRunRegistry()

        self.evaluations = evaluations or EvaluationRegistry()

        # -----------------------------------------------------
        # Health
        # -----------------------------------------------------

        self.health = TrainingHealthChecker(
            config=self.config,
            source_manager=self.sources,
            checkpoints=(self.checkpoints),
            artifacts=(self.artifacts),
        )

    # ---------------------------------------------------------
    # Enabled/task guard
    # ---------------------------------------------------------

    def _require_enabled(
        self,
    ) -> None:
        if not self.config.enabled:
            raise TrainingConfigurationError("Training subsystem is disabled.")

    def _require_task(
        self,
        task: str,
    ) -> str:
        self._require_enabled()

        normalized = str(task).strip().lower()

        if not self.config.task_allowed(normalized):
            raise TrainingTaskMismatchError(
                "Training task is not enabled.",
                details={
                    "task": normalized,
                },
            )

        return normalized

    # ---------------------------------------------------------
    # Initialize directories
    # ---------------------------------------------------------

    def initialize_directories(
        self,
    ) -> dict[str, str]:
        """
        Create output directories owned by the
        training subsystem.

        Input data directories are not created
        here because they belong to data
        production/collection pipelines.
        """

        self._require_enabled()

        paths = self.config.paths.resolved()

        output_keys = (
            "checkpoints",
            "artifacts",
            "models",
            "logs",
        )

        try:
            for key in output_keys:
                Path(paths[key]).mkdir(
                    parents=True,
                    exist_ok=True,
                )

        except OSError as exc:
            raise TrainingConfigurationError(
                "Training output directories could not be initialized.",
                details={
                    "error_type": (type(exc).__name__),
                },
            ) from exc

        return {key: paths[key] for key in output_keys}

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    def apply_reproducibility(
        self,
    ) -> dict[str, Any]:
        """
        Apply configured random seeds.

        Framework-specific determinism is applied
        when the relevant framework is installed.
        """

        seed = self.config.reproducibility.random_seed

        random.seed(seed)

        numpy_seeded = False
        torch_seeded = False

        if np is not None:
            np.random.seed(seed)

            numpy_seeded = True

        if torch is not None:
            if self.config.reproducibility.deterministic:
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            deterministic = self.config.reproducibility.deterministic

            benchmark = self.config.reproducibility.benchmark

            if hasattr(
                torch.backends,
                "cudnn",
            ):
                torch.backends.cudnn.deterministic = deterministic

                torch.backends.cudnn.benchmark = benchmark

            try:
                torch.use_deterministic_algorithms(deterministic)

            except (
                AttributeError,
                RuntimeError,
            ):
                # Some operations/backends may not
                # support strict deterministic
                # algorithms. The caller can inspect
                # the returned status.
                pass

            torch_seeded = True

        return {
            "random_seed": seed,
            "python_seeded": True,
            "numpy_seeded": (numpy_seeded),
            "torch_seeded": (torch_seeded),
            "deterministic_requested": (self.config.reproducibility.deterministic),
            "benchmark": (self.config.reproducibility.benchmark),
        }

    # ---------------------------------------------------------
    # Register sources/builders
    # ---------------------------------------------------------

    def register_example_builder(
        self,
        builder: ExampleBuilder,
    ) -> None:
        self._require_enabled()

        if not callable(builder):
            raise TrainingConfigurationError("Example builder must be callable.")

        self.assembler.builder = builder

    def register_feature_extractor(
        self,
        *,
        task: str,
        extractor: FeatureExtractor,
        replace: bool = False,
    ) -> None:
        task = self._require_task(task)

        self.feature_extractors.register(
            task=task,
            extractor=extractor,
            replace=replace,
        )

    # ---------------------------------------------------------
    # Source discovery
    # ---------------------------------------------------------

    def discover_sources(
        self,
    ) -> list[DiscoveredTrainingSource]:
        self._require_enabled()

        return self.sources.discover()

    # ---------------------------------------------------------
    # Assemble and split
    # ---------------------------------------------------------

    def prepare_dataset(
        self,
        *,
        task: str,
        dataset_id: str,
        version: str = "1",
        sources: Iterable[DiscoveredTrainingSource] | None = None,
        require_labels: bool = True,
    ) -> TrainingPreparation:
        """
        Assemble source records and perform a
        leakage-safe split.

        No feature schema is fitted here.
        """

        task = self._require_task(task)

        assembly = self.assembler.assemble(
            task=task,
            dataset_id=dataset_id,
            version=version,
            discovered_sources=(sources),
            require_labels=(require_labels),
        )

        split = self.splitter.split(assembly.dataset)

        return TrainingPreparation(
            assembly=assembly,
            split=split,
        )

    # ---------------------------------------------------------
    # Fit/transform structured features
    # ---------------------------------------------------------

    def build_structured_features(
        self,
        preparation: TrainingPreparation,
        *,
        schema_version: str = "1",
    ) -> EncodedTrainingSplits:
        """
        Fit a FeatureSchema on TRAIN only and
        apply it unchanged to all partitions.

        This path is appropriate for structured
        numeric/categorical tasks, particularly
        anomaly training.

        Text/event models may use their dedicated
        vocabulary-based dataset path instead.
        """

        split = preparation.split

        split.validate()

        schema = self.feature_builder.fit(
            split.train,
            task=split.task,
            version=(schema_version),
        )

        train = self.feature_builder.transform(
            split.train,
            schema=schema,
        )

        validation = (
            self.feature_builder.transform(
                split.validation,
                schema=schema,
            )
            if split.validation
            else None
        )

        test = (
            self.feature_builder.transform(
                split.test,
                schema=schema,
            )
            if split.test
            else None
        )

        return EncodedTrainingSplits(
            schema=schema,
            train=train,
            validation=validation,
            test=test,
        )

    # ---------------------------------------------------------
    # Create run
    # ---------------------------------------------------------

    def create_run(
        self,
        *,
        task: str,
        dataset_id: str,
        model_name: str,
        accelerator: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TrainingRun:
        task = self._require_task(task)

        run = TrainingRun(
            task=task,
            dataset_id=(dataset_id),
            state=(TrainingState.CREATED.value),
            model_name=(model_name),
            accelerator=(accelerator or self.config.accelerator.preferred),
            attributes=dict(attributes or {}),
        ).finalize()

        return self.runs.register(run)

    # ---------------------------------------------------------
    # State transition
    # ---------------------------------------------------------

    def transition_run(
        self,
        run: TrainingRun,
        state: str,
        *,
        error: BaseException | None = None,
    ) -> TrainingRun:
        run.finalize()

        allowed_states = {item.value for item in TrainingState}

        if state not in allowed_states:
            raise TrainingStateError(
                "Unknown training run state.",
                details={
                    "state": state,
                },
            )

        transitions = {
            TrainingState.CREATED.value: {
                TrainingState.ASSEMBLING.value,
                TrainingState.READY.value,
                TrainingState.CANCELLED.value,
                TrainingState.FAILED.value,
            },
            TrainingState.ASSEMBLING.value: {
                TrainingState.READY.value,
                TrainingState.FAILED.value,
                TrainingState.CANCELLED.value,
            },
            TrainingState.READY.value: {
                TrainingState.TRAINING.value,
                TrainingState.CANCELLED.value,
                TrainingState.FAILED.value,
            },
            TrainingState.TRAINING.value: {
                TrainingState.EVALUATING.value,
                TrainingState.COMPLETED.value,
                TrainingState.FAILED.value,
                TrainingState.CANCELLED.value,
            },
            TrainingState.EVALUATING.value: {
                TrainingState.COMPLETED.value,
                TrainingState.FAILED.value,
                TrainingState.CANCELLED.value,
            },
            TrainingState.COMPLETED.value: set(),
            TrainingState.FAILED.value: set(),
            TrainingState.CANCELLED.value: set(),
        }

        if state != run.state and state not in transitions.get(
            run.state,
            set(),
        ):
            raise TrainingStateError(
                "Invalid training run state transition.",
                details={
                    "run_id": (run.run_id),
                    "current_state": (run.state),
                    "requested_state": (state),
                },
            )

        run.state = state

        if state == TrainingState.TRAINING.value and run.started_at is None:
            run.started_at = utc_now()

        if state in {
            TrainingState.COMPLETED.value,
            TrainingState.FAILED.value,
            TrainingState.CANCELLED.value,
        }:
            run.completed_at = utc_now()

        if error is not None:
            run.error_type = type(error).__name__

        self.runs.register(
            run,
            replace=True,
        )

        return run

    # ---------------------------------------------------------
    # Add metric
    # ---------------------------------------------------------

    def add_metric(
        self,
        run: TrainingRun,
        *,
        name: str,
        value: float,
        step: int | None = None,
        epoch: int | None = None,
        split: str | None = None,
    ) -> TrainingMetric:
        run.finalize()

        numeric = float(value)

        if not math.isfinite(numeric):
            raise TrainingNumericalError(
                "Training metric must be finite.",
                details={
                    "run_id": (run.run_id),
                    "metric": name,
                },
            )

        metric = TrainingMetric(
            name=name,
            value=numeric,
            step=step,
            epoch=epoch,
            split=split,
        ).finalize()

        run.metrics.append(metric)

        self.runs.register(
            run,
            replace=True,
        )

        return metric

    # ---------------------------------------------------------
    # Save checkpoint
    # ---------------------------------------------------------

    def save_checkpoint(
        self,
        *,
        run: TrainingRun,
        model: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        epoch: int | None = None,
        step: int | None = None,
        score: float | None = None,
        score_name: str | None = None,
        vocabulary: Mapping[str, Any] | None = None,
        labels: Mapping[str, Any] | None = None,
        feature_schema: Mapping[str, Any] | None = None,
        model_config: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> TrainingCheckpoint:
        run.finalize()

        if not (self.config.checkpoint.enabled):
            raise TrainingConfigurationError("Checkpointing is disabled.")

        if run.state not in {
            TrainingState.TRAINING.value,
            TrainingState.EVALUATING.value,
        }:
            raise TrainingStateError(
                "Checkpoint can only be saved while a run is training or evaluating.",
                details={
                    "run_id": (run.run_id),
                    "state": run.state,
                },
            )

        checkpoint = self.checkpoints.save(
            run=run,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            step=step,
            score=score,
            score_name=(score_name),
            vocabulary=vocabulary,
            labels=labels,
            feature_schema=(feature_schema),
            model_config=(model_config),
            extra=extra,
        )

        self.runs.register(
            run,
            replace=True,
        )

        self.checkpoints.apply_retention(run_id=str(run.run_id))

        return checkpoint

    # ---------------------------------------------------------
    # Evaluation threshold direction
    # ---------------------------------------------------------

    @staticmethod
    def _threshold_passed(
        *,
        metric_name: str,
        value: float,
        threshold: float,
        directions: Mapping[str, str] | None = None,
    ) -> bool:
        direction = str((directions or {}).get(metric_name, "")).strip().lower()

        if not direction:
            lowered = metric_name.lower()

            if (
                "loss" in lowered
                or "error" in lowered
                or "false_positive" in lowered
                or "false_negative" in lowered
            ):
                direction = "max"

            else:
                direction = "min"

        if direction == "min":
            # Minimum required value:
            # accuracy >= threshold.
            return value >= threshold

        if direction == "max":
            # Maximum allowed value:
            # loss <= threshold.
            return value <= threshold

        raise EvaluationThresholdError(
            "Evaluation threshold direction must be min or max.",
            details={
                "metric": metric_name,
                "direction": direction,
            },
        )

    # ---------------------------------------------------------
    # Record evaluation
    # ---------------------------------------------------------

    def record_evaluation(
        self,
        *,
        run: TrainingRun,
        split: str,
        metrics: Mapping[
            str,
            float,
        ],
        thresholds: Mapping[str, float] | None = None,
        threshold_directions: Mapping[str, str] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        run.finalize()

        split_name = str(split).strip().lower()

        if split_name not in {item.value for item in EvaluationSplit}:
            raise EvaluationMetricError(
                "Unknown evaluation split.",
                details={
                    "split": split_name,
                },
            )

        if not metrics:
            raise EvaluationMetricError("Evaluation requires at least one metric.")

        metric_values: list[TrainingMetric] = []

        normalized_metrics: dict[
            str,
            float,
        ] = {}

        for name, raw_value in metrics.items():
            value = float(raw_value)

            if self.config.evaluation.fail_on_non_finite_metric and not math.isfinite(
                value
            ):
                raise EvaluationMetricError(
                    "Evaluation metric must be finite.",
                    details={
                        "metric": name,
                    },
                )

            normalized_metrics[str(name)] = value

            metric_values.append(
                TrainingMetric(
                    name=str(name),
                    value=value,
                    split=split_name,
                ).finalize()
            )

        required_thresholds = dict(
            thresholds if thresholds is not None else self.config.evaluation.thresholds
        )

        passed: bool | None

        if required_thresholds:
            checks: list[bool] = []

            for name, threshold in required_thresholds.items():
                if name not in (normalized_metrics):
                    raise (
                        EvaluationThresholdError(
                            "Required evaluation metric is missing.",
                            details={
                                "metric": name,
                            },
                        )
                    )

                checks.append(
                    self._threshold_passed(
                        metric_name=name,
                        value=(normalized_metrics[name]),
                        threshold=float(threshold),
                        directions=(threshold_directions),
                    )
                )

            passed = all(checks)

        else:
            passed = None

        evaluation = EvaluationResult(
            run_id=str(run.run_id),
            model_name=str(run.model_name or "unnamed-model"),
            split=split_name,
            metrics=metric_values,
            passed=passed,
            thresholds={
                str(name): float(threshold)
                for name, threshold in required_thresholds.items()
            },
            attributes={
                **dict(attributes or {}),
                "threshold_directions": dict(threshold_directions or {}),
            },
        ).finalize()

        self.evaluations.register(evaluation)

        return evaluation

    # ---------------------------------------------------------
    # Candidate artifact
    # ---------------------------------------------------------

    def create_candidate(
        self,
        *,
        checkpoint: TrainingCheckpoint,
        task: str,
        version: str,
        evaluation: EvaluationResult | None = None,
        overwrite: bool = False,
    ) -> ModelArtifact:
        task = self._require_task(task)

        return self.artifacts.create_candidate(
            checkpoint=checkpoint,
            task=task,
            version=version,
            evaluation=evaluation,
            overwrite=overwrite,
        )

    # ---------------------------------------------------------
    # Promote artifact
    # ---------------------------------------------------------

    def promote(
        self,
        artifact: ModelArtifact,
        *,
        evaluation: EvaluationResult | None = None,
    ) -> ModelArtifact:
        self._require_enabled()

        return self.artifacts.promote(
            artifact,
            evaluation=evaluation,
        )

    # ---------------------------------------------------------
    # Best checkpoint
    # ---------------------------------------------------------

    def best_checkpoint(
        self,
        *,
        run_id: str,
        mark: bool = True,
    ) -> TrainingCheckpoint:
        self._require_enabled()

        return self.checkpoints.best(
            run_id=run_id,
            mark=mark,
        )

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------

    def health_check(
        self,
    ) -> TrainingHealth:
        return self.health.check()

    # ---------------------------------------------------------
    # Start
    # ---------------------------------------------------------

    def start(
        self,
        *,
        discover_existing: bool = True,
    ) -> dict[str, Any]:
        """
        Initialize shared training infrastructure.

        This does not start model training.
        """

        if not self.config.enabled:
            return {
                "enabled": False,
                "started": False,
                "training_started": (False),
            }

        paths = self.initialize_directories()

        reproducibility = self.apply_reproducibility()

        discovered_checkpoints = 0
        discovered_artifacts = 0
        discovered_models = 0

        if discover_existing:
            discovered_checkpoints = len(self.checkpoints.discover())

            discovered_artifacts = len(self.artifacts.discover(promoted=False))

            discovered_models = len(self.artifacts.discover(promoted=True))

        health = self.health_check()

        return {
            "enabled": True,
            "started": True,
            "paths": paths,
            "reproducibility": (reproducibility),
            "discovered_checkpoints": (discovered_checkpoints),
            "discovered_artifacts": (discovered_artifacts),
            "discovered_models": (discovered_models),
            "health": (health.to_dict()),
            "training_started": (False),
        }

    # ---------------------------------------------------------
    # Stop
    # ---------------------------------------------------------

    def stop(
        self,
    ) -> dict[str, Any]:
        """
        The common training facade owns no
        persistent optimization worker.

        Worker lifecycle remains under
        src/workers/ and task-specific trainers.
        """

        return {
            "stopped": True,
            "active_training_workers": (0),
            "training_cancelled": (False),
        }

    # ---------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------

    def __enter__(
        self,
    ) -> Self:
        self.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:
        self.stop()

        return False

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> dict[str, Any]:
        return {
            "config": (safe_config_summary(self.config)),
            "sources": (self.sources.summary()),
            "assembler": (self.assembler.summary()),
            "splitter": (self.splitter.summary()),
            "features": (self.feature_builder.summary()),
            "runs": (self.runs.summary()),
            "evaluations": (self.evaluations.summary()),
            "checkpoints": (self.checkpoints.summary()),
            "artifacts": (self.artifacts.summary()),
            "health": (self.health.summary()),
            "task_specific_training_loops": {
                "event": ("src.training.train_events"),
                "anomaly": ("src.training.train_anomalies"),
            },
            "optimization_loop_implemented_here": (False),
            "automatic_model_promotion": (self.config.promotion.automatic),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "EncodedTrainingSplits",
    "EvaluationRegistry",
    "TrainingPipeline",
    "TrainingPreparation",
    "TrainingRunRegistry",
]
