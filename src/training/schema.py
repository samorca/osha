from __future__ import annotations

"""
Shared training contracts for OSHA.

These models describe training examples, datasets,
runs, metrics, checkpoints, evaluations, and model
artifacts.

Data preparation remains in src/data/.
Infrastructure interpretation remains in
src/intelligence/.
"""

from dataclasses import (
    dataclass,
    field,
)

from enum import Enum

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from src.core.identifiers import (
    random_id,
)

from src.core.serialization import (
    to_json_safe,
)

from src.core.time import (
    utc_now,
)

from src.core.validation import (
    required_string,
    unique_strings,
)


TRAINING_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------
# Training task
# ---------------------------------------------------------------

class TrainingTask(
    str,
    Enum,
):
    EVENT = "event"

    ANOMALY = "anomaly"

    CLASSIFICATION = "classification"

    REGRESSION = "regression"

    REPRESENTATION = "representation"

    CUSTOM = "custom"


# ---------------------------------------------------------------
# Training source kind
# ---------------------------------------------------------------

class TrainingSourceKind(
    str,
    Enum,
):
    PREPARED_DATASET = (
        "prepared_dataset"
    )

    PROCESSED_OPERATIONAL_DATA = (
        "processed_operational_data"
    )

    NORMALIZED_TELEMETRY = (
        "normalized_telemetry"
    )

    CORRELATED_DATA = (
        "correlated_data"
    )

    INTELLIGENCE_DATA = (
        "intelligence_data"
    )

    DECISION_DATA = (
        "decision_data"
    )

    REMEDIATION_DATA = (
        "remediation_data"
    )

    MEMORY_DATA = "memory_data"

    CUSTOM = "custom"


# ---------------------------------------------------------------
# Training lifecycle
# ---------------------------------------------------------------

class TrainingState(
    str,
    Enum,
):
    CREATED = "created"

    ASSEMBLING = "assembling"

    READY = "ready"

    TRAINING = "training"

    EVALUATING = "evaluating"

    COMPLETED = "completed"

    FAILED = "failed"

    CANCELLED = "cancelled"


# ---------------------------------------------------------------
# Checkpoint state
# ---------------------------------------------------------------

class CheckpointState(
    str,
    Enum,
):
    CREATED = "created"

    SAVED = "saved"

    BEST = "best"

    REJECTED = "rejected"

    PROMOTED = "promoted"


# ---------------------------------------------------------------
# Evaluation split
# ---------------------------------------------------------------

class EvaluationSplit(
    str,
    Enum,
):
    TRAIN = "train"

    VALIDATION = "validation"

    TEST = "test"


# ---------------------------------------------------------------
# Artifact state
# ---------------------------------------------------------------

class ArtifactState(
    str,
    Enum,
):
    CANDIDATE = "candidate"

    ACCEPTED = "accepted"

    REJECTED = "rejected"

    PROMOTED = "promoted"

    RETIRED = "retired"


# ---------------------------------------------------------------
# Training source
# ---------------------------------------------------------------

@dataclass(frozen=True)
class TrainingSource:
    """
    Identifies one source contributing examples
    to a training dataset.

    location points to an artifact/directory. It
    does not contain credentials.
    """

    kind: str

    location: str

    source_id: Optional[
        str
    ] = None

    dataset_id: Optional[
        str
    ] = None

    dataset_version: Optional[
        str
    ] = None

    provider: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "TrainingSource":
        if (
            self.kind
            not in {
                item.value
                for item
                in TrainingSourceKind
            }
        ):
            raise ValueError(
                "Unknown training source "
                f"kind: {self.kind}"
            )

        required_string(
            self.location,
            name="training source location",
            maximum_length=4096,
        )

        if self.source_id is not None:
            required_string(
                self.source_id,
                name="source_id",
                maximum_length=512,
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "kind": self.kind,
            "location": self.location,
            "source_id": self.source_id,
            "dataset_id": self.dataset_id,
            "dataset_version": (
                self.dataset_version
            ),
            "provider": self.provider,
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Training example
# ---------------------------------------------------------------

@dataclass
class TrainingExample:
    """
    One canonical model-learning example.

    group_id is important for leakage-safe
    splitting. Related records from one incident,
    sequence, host session, trace, or similar
    context can be assigned to one group.
    """

    task: str

    features: Dict[
        str,
        Any,
    ]

    label: Any = None

    source_id: Optional[
        str
    ] = None

    dataset_id: Optional[
        str
    ] = None

    provider: Optional[
        str
    ] = None

    environment_id: Optional[
        str
    ] = None

    resource_type: Optional[
        str
    ] = None

    resource_id: Optional[
        str
    ] = None

    incident_id: Optional[
        str
    ] = None

    group_id: Optional[
        str
    ] = None

    timestamp: Optional[
        str
    ] = None

    reference_ids: List[
        str
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    example_id: Optional[
        str
    ] = None

    def finalize(
        self,
    ) -> "TrainingExample":
        if (
            self.task
            not in {
                item.value
                for item
                in TrainingTask
            }
        ):
            raise ValueError(
                "Unknown training task: "
                f"{self.task}"
            )

        if not isinstance(
            self.features,
            dict,
        ):
            raise ValueError(
                "TrainingExample features "
                "must be a dictionary"
            )

        self.reference_ids = (
            unique_strings(
                self.reference_ids
            )
        )

        if not self.example_id:
            self.example_id = (
                random_id(
                    "training:example"
                )
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "example_id": (
                self.example_id
            ),
            "task": self.task,
            "features": (
                to_json_safe(
                    self.features
                )
            ),
            "label": (
                to_json_safe(
                    self.label
                )
            ),
            "source_id": self.source_id,
            "dataset_id": self.dataset_id,
            "provider": self.provider,
            "environment_id": (
                self.environment_id
            ),
            "resource_type": (
                self.resource_type
            ),
            "resource_id": (
                self.resource_id
            ),
            "incident_id": (
                self.incident_id
            ),
            "group_id": self.group_id,
            "timestamp": self.timestamp,
            "reference_ids": list(
                self.reference_ids
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Assembled training dataset
# ---------------------------------------------------------------

@dataclass
class TrainingDataset:
    dataset_id: str

    task: str

    examples: List[
        TrainingExample
    ] = field(
        default_factory=list
    )

    sources: List[
        TrainingSource
    ] = field(
        default_factory=list
    )

    version: str = "1"

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "TrainingDataset":
        required_string(
            self.dataset_id,
            name="training dataset_id",
            maximum_length=512,
        )

        required_string(
            self.version,
            name="training dataset version",
            maximum_length=128,
        )

        if (
            self.task
            not in {
                item.value
                for item
                in TrainingTask
            }
        ):
            raise ValueError(
                "Unknown training task: "
                f"{self.task}"
            )

        seen = set()

        for example in self.examples:
            example.finalize()

            if example.task != self.task:
                raise ValueError(
                    "Training example task "
                    "does not match dataset "
                    "task"
                )

            if (
                example.example_id
                in seen
            ):
                raise ValueError(
                    "Duplicate training "
                    "example_id detected"
                )

            seen.add(
                example.example_id
            )

        for source in self.sources:
            source.finalize()

        return self

    @property
    def example_count(
        self,
    ) -> int:
        return len(
            self.examples
        )

    def to_dict(
        self,
        *,
        include_examples: bool = False,
    ) -> Dict[str, Any]:
        self.finalize()

        result = {
            "schema_version": (
                TRAINING_SCHEMA_VERSION
            ),
            "dataset_id": self.dataset_id,
            "version": self.version,
            "task": self.task,
            "example_count": (
                self.example_count
            ),
            "sources": [
                source.to_dict()
                for source
                in self.sources
            ],
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
            "created_at": (
                self.created_at
            ),
        }

        if include_examples:
            result[
                "examples"
            ] = [
                example.to_dict()
                for example
                in self.examples
            ]

        return result


# ---------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------

@dataclass(frozen=True)
class TrainingMetric:
    name: str

    value: float

    step: Optional[
        int
    ] = None

    epoch: Optional[
        int
    ] = None

    split: Optional[
        str
    ] = None

    observed_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "TrainingMetric":
        required_string(
            self.name,
            name="metric name",
            maximum_length=256,
        )

        if (
            self.split is not None
            and self.split
            not in {
                item.value
                for item
                in EvaluationSplit
            }
        ):
            raise ValueError(
                "Unknown metric split: "
                f"{self.split}"
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "name": self.name,
            "value": float(
                self.value
            ),
            "step": self.step,
            "epoch": self.epoch,
            "split": self.split,
            "observed_at": (
                self.observed_at
            ),
        }


# ---------------------------------------------------------------
# Training run
# ---------------------------------------------------------------

@dataclass
class TrainingRun:
    task: str

    dataset_id: str

    state: str = (
        TrainingState.CREATED.value
    )

    run_id: Optional[
        str
    ] = None

    model_name: Optional[
        str
    ] = None

    accelerator: Optional[
        str
    ] = None

    started_at: Optional[
        str
    ] = None

    completed_at: Optional[
        str
    ] = None

    metrics: List[
        TrainingMetric
    ] = field(
        default_factory=list
    )

    checkpoint_ids: List[
        str
    ] = field(
        default_factory=list
    )

    error_type: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> "TrainingRun":
        if (
            self.task
            not in {
                item.value
                for item
                in TrainingTask
            }
        ):
            raise ValueError(
                "Unknown training task: "
                f"{self.task}"
            )

        required_string(
            self.dataset_id,
            name="dataset_id",
            maximum_length=512,
        )

        if (
            self.state
            not in {
                item.value
                for item
                in TrainingState
            }
        ):
            raise ValueError(
                "Unknown training state: "
                f"{self.state}"
            )

        if not self.run_id:
            self.run_id = (
                random_id(
                    "training:run"
                )
            )

        self.metrics = [
            metric.finalize()
            for metric
            in self.metrics
        ]

        self.checkpoint_ids = (
            unique_strings(
                self.checkpoint_ids
            )
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (
                TRAINING_SCHEMA_VERSION
            ),
            "run_id": self.run_id,
            "task": self.task,
            "dataset_id": (
                self.dataset_id
            ),
            "state": self.state,
            "model_name": (
                self.model_name
            ),
            "accelerator": (
                self.accelerator
            ),
            "started_at": (
                self.started_at
            ),
            "completed_at": (
                self.completed_at
            ),
            "metrics": [
                metric.to_dict()
                for metric
                in self.metrics
            ],
            "checkpoint_ids": list(
                self.checkpoint_ids
            ),
            "error_type": (
                self.error_type
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
        }


# ---------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------

@dataclass
class TrainingCheckpoint:
    run_id: str

    model_name: str

    path: str

    state: str = (
        CheckpointState.CREATED.value
    )

    checkpoint_id: Optional[
        str
    ] = None

    epoch: Optional[
        int
    ] = None

    step: Optional[
        int
    ] = None

    score: Optional[
        float
    ] = None

    score_name: Optional[
        str
    ] = None

    checksum: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "TrainingCheckpoint":
        required_string(
            self.run_id,
            name="run_id",
            maximum_length=512,
        )

        required_string(
            self.model_name,
            name="model_name",
            maximum_length=512,
        )

        required_string(
            self.path,
            name="checkpoint path",
            maximum_length=4096,
        )

        if (
            self.state
            not in {
                item.value
                for item
                in CheckpointState
            }
        ):
            raise ValueError(
                "Unknown checkpoint "
                f"state: {self.state}"
            )

        if not self.checkpoint_id:
            self.checkpoint_id = (
                random_id(
                    "training:checkpoint"
                )
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "checkpoint_id": (
                self.checkpoint_id
            ),
            "run_id": self.run_id,
            "model_name": (
                self.model_name
            ),
            "path": self.path,
            "state": self.state,
            "epoch": self.epoch,
            "step": self.step,
            "score": self.score,
            "score_name": (
                self.score_name
            ),
            "checksum": (
                self.checksum
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
            "created_at": (
                self.created_at
            ),
        }


# ---------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------

@dataclass
class EvaluationResult:
    run_id: str

    model_name: str

    split: str

    metrics: List[
        TrainingMetric
    ] = field(
        default_factory=list
    )

    passed: Optional[
        bool
    ] = None

    thresholds: Dict[
        str,
        float,
    ] = field(
        default_factory=dict
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    evaluation_id: Optional[
        str
    ] = None

    evaluated_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "EvaluationResult":
        required_string(
            self.run_id,
            name="run_id",
            maximum_length=512,
        )

        required_string(
            self.model_name,
            name="model_name",
            maximum_length=512,
        )

        if (
            self.split
            not in {
                item.value
                for item
                in EvaluationSplit
            }
        ):
            raise ValueError(
                "Unknown evaluation "
                f"split: {self.split}"
            )

        self.metrics = [
            metric.finalize()
            for metric
            in self.metrics
        ]

        if not self.evaluation_id:
            self.evaluation_id = (
                random_id(
                    "training:evaluation"
                )
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "evaluation_id": (
                self.evaluation_id
            ),
            "run_id": self.run_id,
            "model_name": (
                self.model_name
            ),
            "split": self.split,
            "metrics": [
                metric.to_dict()
                for metric
                in self.metrics
            ],
            "passed": self.passed,
            "thresholds": dict(
                self.thresholds
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
            "evaluated_at": (
                self.evaluated_at
            ),
        }


# ---------------------------------------------------------------
# Model artifact
# ---------------------------------------------------------------

@dataclass
class ModelArtifact:
    """
    A checkpoint becomes a model artifact only
    after an explicit artifact/promotion step.
    """

    model_name: str

    task: str

    version: str

    path: str

    state: str = (
        ArtifactState.CANDIDATE.value
    )

    artifact_id: Optional[
        str
    ] = None

    source_run_id: Optional[
        str
    ] = None

    checkpoint_id: Optional[
        str
    ] = None

    checksum: Optional[
        str
    ] = None

    metrics: Dict[
        str,
        float,
    ] = field(
        default_factory=dict
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "ModelArtifact":
        required_string(
            self.model_name,
            name="model_name",
            maximum_length=512,
        )

        required_string(
            self.version,
            name="model version",
            maximum_length=128,
        )

        required_string(
            self.path,
            name="model artifact path",
            maximum_length=4096,
        )

        if (
            self.task
            not in {
                item.value
                for item
                in TrainingTask
            }
        ):
            raise ValueError(
                "Unknown model task: "
                f"{self.task}"
            )

        if (
            self.state
            not in {
                item.value
                for item
                in ArtifactState
            }
        ):
            raise ValueError(
                "Unknown model artifact "
                f"state: {self.state}"
            )

        if not self.artifact_id:
            self.artifact_id = (
                random_id(
                    "training:artifact"
                )
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (
                TRAINING_SCHEMA_VERSION
            ),
            "artifact_id": (
                self.artifact_id
            ),
            "model_name": (
                self.model_name
            ),
            "task": self.task,
            "version": self.version,
            "path": self.path,
            "state": self.state,
            "source_run_id": (
                self.source_run_id
            ),
            "checkpoint_id": (
                self.checkpoint_id
            ),
            "checksum": (
                self.checksum
            ),
            "metrics": dict(
                self.metrics
            ),
            "attributes": (
                to_json_safe(
                    self.attributes
                )
            ),
            "created_at": (
                self.created_at
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "TRAINING_SCHEMA_VERSION",

    "TrainingTask",
    "TrainingSourceKind",
    "TrainingState",
    "CheckpointState",
    "EvaluationSplit",
    "ArtifactState",

    "TrainingSource",
    "TrainingExample",
    "TrainingDataset",

    "TrainingMetric",
    "TrainingRun",

    "TrainingCheckpoint",
    "EvaluationResult",

    "ModelArtifact",
]
