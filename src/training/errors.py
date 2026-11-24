from __future__ import annotations

"""
Exception hierarchy for OSHA model training,
evaluation, checkpointing, and artifact
promotion.
"""

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)


# ---------------------------------------------------------------
# Base training error
# ---------------------------------------------------------------


class TrainingError(RuntimeError):
    """
    Base exception for expected failures inside
    the OSHA training subsystem.

    Underlying framework/provider exception
    messages should normally remain behind this
    boundary.
    """

    default_code = "training_error"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        retryable: bool = False,
        details: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        super().__init__(message)

        self.message = message

        self.code = code or self.default_code

        self.retryable = bool(retryable)

        self.details = dict(details or {})


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------


class TrainingConfigurationError(TrainingError):
    default_code = "training_configuration_error"


# ---------------------------------------------------------------
# Sources
# ---------------------------------------------------------------


class TrainingSourceError(TrainingError):
    default_code = "training_source_error"


class TrainingSourceNotFoundError(TrainingSourceError):
    default_code = "training_source_not_found"


class TrainingSourceUnsupportedError(TrainingSourceError):
    default_code = "training_source_unsupported"


class TrainingSourceFormatError(TrainingSourceError):
    default_code = "training_source_format_error"


# ---------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------


class DatasetAssemblyError(TrainingError):
    default_code = "training_dataset_assembly_error"


class EmptyTrainingDatasetError(DatasetAssemblyError):
    default_code = "empty_training_dataset"


class DuplicateTrainingExampleError(DatasetAssemblyError):
    default_code = "duplicate_training_example"


class TrainingTaskMismatchError(DatasetAssemblyError):
    default_code = "training_task_mismatch"


# ---------------------------------------------------------------
# Features
# ---------------------------------------------------------------


class FeatureError(TrainingError):
    default_code = "feature_error"


class FeatureExtractionError(FeatureError):
    default_code = "feature_extraction_error"


class FeatureValidationError(FeatureError):
    default_code = "feature_validation_error"


class FeatureSchemaMismatchError(FeatureError):
    default_code = "feature_schema_mismatch"


# ---------------------------------------------------------------
# Labels
# ---------------------------------------------------------------


class LabelError(TrainingError):
    default_code = "label_error"


class MissingLabelError(LabelError):
    default_code = "missing_label"


class InvalidLabelError(LabelError):
    default_code = "invalid_label"


# ---------------------------------------------------------------
# Dataset splitting / leakage
# ---------------------------------------------------------------


class TrainingSplitError(TrainingError):
    default_code = "training_split_error"


class TrainingLeakageError(TrainingSplitError):
    """
    Evidence suggests related examples appear
    across multiple training/evaluation splits.
    """

    default_code = "training_data_leakage"


class MissingGroupError(TrainingSplitError):
    default_code = "training_group_missing"


class SplitOverlapError(TrainingSplitError):
    default_code = "training_split_overlap"


# ---------------------------------------------------------------
# Accelerator
# ---------------------------------------------------------------


class AcceleratorSelectionError(TrainingError):
    default_code = "accelerator_selection_error"


class AcceleratorUnavailableError(AcceleratorSelectionError):
    default_code = "accelerator_unavailable"


class AcceleratorFallbackError(AcceleratorSelectionError):
    default_code = "accelerator_fallback_error"


# ---------------------------------------------------------------
# Training execution
# ---------------------------------------------------------------


class TrainingRunError(TrainingError):
    default_code = "training_run_error"


class TrainingStateError(TrainingRunError):
    default_code = "training_state_error"


class TrainingNumericalError(TrainingRunError):
    """
    Non-finite loss/gradient/metric or other
    numerical training failure.
    """

    default_code = "training_numerical_error"


class TrainingCancelledError(TrainingRunError):
    default_code = "training_cancelled"


# ---------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------


class CheckpointError(TrainingError):
    default_code = "checkpoint_error"


class CheckpointNotFoundError(CheckpointError):
    default_code = "checkpoint_not_found"


class CheckpointConflictError(CheckpointError):
    default_code = "checkpoint_conflict"


class CheckpointIntegrityError(CheckpointError):
    default_code = "checkpoint_integrity_error"


class CheckpointLoadError(CheckpointError):
    default_code = "checkpoint_load_error"


class CheckpointSaveError(CheckpointError):
    default_code = "checkpoint_save_error"


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------


class EvaluationError(TrainingError):
    default_code = "evaluation_error"


class EvaluationMetricError(EvaluationError):
    default_code = "evaluation_metric_error"


class EvaluationThresholdError(EvaluationError):
    default_code = "evaluation_threshold_error"


class EvaluationFailedError(EvaluationError):
    """
    Evaluation completed successfully from a
    technical perspective but failed the required
    acceptance criteria.
    """

    default_code = "evaluation_failed"


# ---------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------


class ModelArtifactError(TrainingError):
    default_code = "model_artifact_error"


class ModelArtifactNotFoundError(ModelArtifactError):
    default_code = "model_artifact_not_found"


class ModelArtifactConflictError(ModelArtifactError):
    default_code = "model_artifact_conflict"


class ModelArtifactIntegrityError(ModelArtifactError):
    default_code = "model_artifact_integrity_error"


# ---------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------


class ModelPromotionError(ModelArtifactError):
    default_code = "model_promotion_error"


class ModelPromotionBlockedError(ModelPromotionError):
    """
    Promotion was refused because one or more
    configured safety/quality conditions were
    not satisfied.
    """

    default_code = "model_promotion_blocked"


# ---------------------------------------------------------------
# Safe underlying-error metadata
# ---------------------------------------------------------------


def safe_training_error_details(
    error: BaseException,
    *,
    operation: Optional[str] = None,
    task: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Produce bounded metadata for an underlying
    ML framework/filesystem/accelerator failure.

    Raw exception text is intentionally omitted.
    """

    details: Dict[
        str,
        Any,
    ] = {
        "error_type": (type(error).__name__),
        "error_module": (type(error).__module__),
    }

    if operation is not None:
        details["operation"] = operation

    if task is not None:
        details["task"] = task

    if run_id is not None:
        details["run_id"] = run_id

    return details


# ---------------------------------------------------------------
# Generic framework error wrapper
# ---------------------------------------------------------------


def wrap_training_error(
    error: BaseException,
    *,
    message: str = ("Training operation failed."),
    code: str = ("training_operation_failed"),
    retryable: bool = False,
    operation: Optional[str] = None,
    task: Optional[str] = None,
    run_id: Optional[str] = None,
) -> TrainingError:
    """
    Convert an underlying framework/runtime
    exception into the training-layer model.
    """

    return TrainingError(
        message,
        code=code,
        retryable=retryable,
        details=(
            safe_training_error_details(
                error,
                operation=operation,
                task=task,
                run_id=run_id,
            )
        ),
    )


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "TrainingError",
    "TrainingConfigurationError",
    "TrainingSourceError",
    "TrainingSourceNotFoundError",
    "TrainingSourceUnsupportedError",
    "TrainingSourceFormatError",
    "DatasetAssemblyError",
    "EmptyTrainingDatasetError",
    "DuplicateTrainingExampleError",
    "TrainingTaskMismatchError",
    "FeatureError",
    "FeatureExtractionError",
    "FeatureValidationError",
    "FeatureSchemaMismatchError",
    "LabelError",
    "MissingLabelError",
    "InvalidLabelError",
    "TrainingSplitError",
    "TrainingLeakageError",
    "MissingGroupError",
    "SplitOverlapError",
    "AcceleratorSelectionError",
    "AcceleratorUnavailableError",
    "AcceleratorFallbackError",
    "TrainingRunError",
    "TrainingStateError",
    "TrainingNumericalError",
    "TrainingCancelledError",
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointConflictError",
    "CheckpointIntegrityError",
    "CheckpointLoadError",
    "CheckpointSaveError",
    "EvaluationError",
    "EvaluationMetricError",
    "EvaluationThresholdError",
    "EvaluationFailedError",
    "ModelArtifactError",
    "ModelArtifactNotFoundError",
    "ModelArtifactConflictError",
    "ModelArtifactIntegrityError",
    "ModelPromotionError",
    "ModelPromotionBlockedError",
    "safe_training_error_details",
    "wrap_training_error",
]
