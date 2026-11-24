from __future__ import annotations

"""
Training dataset assembly for OSHA.

Combines eligible records discovered by
TrainingSourceManager into canonical
TrainingExample / TrainingDataset objects.

This module does not:
    train models
    build tensors
    select accelerators
    evaluate checkpoints
"""

import hashlib
import json

from collections.abc import (
    Callable,
    Iterable,
    Mapping,
)

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from src.core.serialization import (
    to_json_safe,
)

from src.training.config import (
    TrainingConfig,
)

from src.training.errors import (
    DatasetAssemblyError,
    DuplicateTrainingExampleError,
    EmptyTrainingDatasetError,
    MissingLabelError,
    TrainingTaskMismatchError,
)

from src.training.schema import (
    TrainingDataset,
    TrainingExample,
    TrainingSource,
    TrainingTask,
)

from src.training.sources import (
    DiscoveredTrainingSource,
    TrainingSourceManager,
)


# ---------------------------------------------------------------
# Example builder contract
# ---------------------------------------------------------------

ExampleBuilder = Callable[
    [
        Dict[
            str,
            Any,
        ],
        str,
    ],
    Optional[TrainingExample],
]


# ---------------------------------------------------------------
# Common field aliases
# ---------------------------------------------------------------

LABEL_FIELDS = (
    "label",
    "target",
    "class",
    "event_label",
    "anomaly_label",
)

PROVIDER_FIELDS = (
    "provider",
    "source_provider",
)

ENVIRONMENT_FIELDS = (
    "environment_id",
    "environment",
)

RESOURCE_TYPE_FIELDS = (
    "resource_type",
    "resource",
)

RESOURCE_ID_FIELDS = (
    "resource_id",
    "resource_name",
)

INCIDENT_ID_FIELDS = (
    "incident_id",
    "incident",
)

GROUP_ID_FIELDS = (
    "group_id",
    "sequence_id",
    "session_id",
    "trace_id",
)

TIMESTAMP_FIELDS = (
    "timestamp",
    "time",
    "datetime",
    "@timestamp",
)


# ---------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------


def _field(
    value: Mapping[
        str,
        Any,
    ],
    names: Tuple[
        str,
        ...,
    ],
) -> Any:
    for name in names:
        candidate = value.get(name)

        if candidate is not None:
            return candidate

    return None


def _text_field(
    value: Mapping[
        str,
        Any,
    ],
    names: Tuple[
        str,
        ...,
    ],
) -> Optional[str]:
    candidate = _field(
        value,
        names,
    )

    if candidate is None:
        return None

    text = str(candidate).strip()

    return text or None


# ---------------------------------------------------------------
# Training-source provenance
# ---------------------------------------------------------------


def _source_metadata(
    record: Mapping[
        str,
        Any,
    ],
) -> Dict[str, Any]:
    value = record.get(
        "_training_source",
        {},
    )

    if isinstance(
        value,
        Mapping,
    ):
        return dict(value)

    return {}


# ---------------------------------------------------------------
# Stable example ID
# ---------------------------------------------------------------


def deterministic_example_id(
    *,
    task: str,
    record: Mapping[
        str,
        Any,
    ],
) -> str:
    """
    Generate a stable training-example identity.

    Provenance metadata and normalized record
    content participate in the hash.
    """

    try:
        safe = to_json_safe(dict(record))

        serialized = json.dumps(
            safe,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )

    except Exception as exc:
        raise DatasetAssemblyError(
            "Training record could not be canonicalized for identity.",
            details={
                "error_type": (type(exc).__name__),
            },
        ) from exc

    digest = hashlib.sha256((str(task) + "\n" + serialized).encode("utf-8")).hexdigest()

    return "training-example-" + digest


# ---------------------------------------------------------------
# Feature payload
# ---------------------------------------------------------------


def _feature_payload(
    record: Mapping[
        str,
        Any,
    ],
) -> Dict[str, Any]:
    """
    Produce the initial feature payload carried by
    TrainingExample.

    This is NOT final numeric model feature
    engineering. features.py owns that step.
    """

    if "features" in record and isinstance(
        record["features"],
        Mapping,
    ):
        value = dict(record["features"])

    elif "payload" in record and isinstance(
        record["payload"],
        Mapping,
    ):
        value = dict(record["payload"])

    else:
        excluded = {
            "_training_source",
            *LABEL_FIELDS,
        }

        value = {key: item for key, item in record.items() if key not in excluded}

    safe = to_json_safe(value)

    if not isinstance(
        safe,
        dict,
    ):
        raise DatasetAssemblyError("Training features must normalize to an object.")

    return safe


# ---------------------------------------------------------------
# Default grouping
# ---------------------------------------------------------------


def derive_group_id(
    record: Mapping[
        str,
        Any,
    ],
) -> Optional[str]:
    """
    Prefer explicit grouping, then incident ID.

    We do not automatically group every record by
    resource_id because that could place the
    entire history of a long-lived host/cluster
    into one partition.
    """

    group_id = _text_field(
        record,
        GROUP_ID_FIELDS,
    )

    if group_id:
        return group_id

    incident_id = _text_field(
        record,
        INCIDENT_ID_FIELDS,
    )

    if incident_id:
        return "incident:" + incident_id

    return None


# ---------------------------------------------------------------
# Default example builder
# ---------------------------------------------------------------


def default_example_builder(
    record: Dict[
        str,
        Any,
    ],
    task: str,
) -> Optional[TrainingExample]:
    """
    Convert one loaded source record into a
    task-specific TrainingExample container.

    The features here remain structural/raw.
    Model-specific feature construction happens
    later in features.py.
    """

    if task not in {item.value for item in TrainingTask}:
        raise TrainingTaskMismatchError(
            "Unknown training task.",
            details={
                "task": task,
            },
        )

    source = _source_metadata(record)

    label = _field(
        record,
        LABEL_FIELDS,
    )

    # Anomaly/event supervised training normally
    # requires labels. If a record is not yet a
    # usable supervised example, skip it here.
    if (
        task
        in {
            TrainingTask.EVENT.value,
            TrainingTask.ANOMALY.value,
            TrainingTask.CLASSIFICATION.value,
        }
        and label is None
    ):
        return None

    provider = _text_field(
        record,
        PROVIDER_FIELDS,
    ) or (
        str(source.get("provider")).strip()
        if source.get("provider") is not None
        else None
    )

    incident_id = _text_field(
        record,
        INCIDENT_ID_FIELDS,
    )

    group_id = derive_group_id(record)

    references: List[str] = []

    for candidate in (
        record.get("record_id"),
        record.get("event_id"),
        record.get("signal_id"),
        record.get("correlation_id"),
        incident_id,
    ):
        if candidate is None:
            continue

        text = str(candidate).strip()

        if text and text not in references:
            references.append(text)

    source_id = (
        str(source.get("source_id")).strip()
        if source.get("source_id") is not None
        else None
    )

    dataset_id = (
        str(source.get("dataset_id")).strip()
        if source.get("dataset_id") is not None
        else None
    )

    example = TrainingExample(
        task=task,
        features=(_feature_payload(record)),
        label=label,
        source_id=(source_id),
        dataset_id=(dataset_id),
        provider=provider,
        environment_id=(
            _text_field(
                record,
                ENVIRONMENT_FIELDS,
            )
        ),
        resource_type=(
            _text_field(
                record,
                RESOURCE_TYPE_FIELDS,
            )
        ),
        resource_id=(
            _text_field(
                record,
                RESOURCE_ID_FIELDS,
            )
        ),
        incident_id=(incident_id),
        group_id=(group_id),
        timestamp=(
            _text_field(
                record,
                TIMESTAMP_FIELDS,
            )
        ),
        reference_ids=(references),
        attributes={
            "source_kind": (source.get("kind")),
            "source_file": (source.get("file")),
            "dataset_version": (source.get("dataset_version")),
            "feature_stage": ("assembled"),
        },
        example_id=(
            deterministic_example_id(
                task=task,
                record=record,
            )
        ),
    )

    return example.finalize()


# ---------------------------------------------------------------
# Assembly result
# ---------------------------------------------------------------


class AssemblyResult:
    """
    Bounded summary plus assembled dataset.
    """

    def __init__(
        self,
        *,
        dataset: TrainingDataset,
        source_records: int,
        accepted_examples: int,
        skipped_records: int,
        duplicate_examples: int,
    ) -> None:
        self.dataset = dataset.finalize()

        self.source_records = int(source_records)

        self.accepted_examples = int(accepted_examples)

        self.skipped_records = int(skipped_records)

        self.duplicate_examples = int(duplicate_examples)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "dataset": (self.dataset.to_dict(include_examples=False)),
            "source_records": (self.source_records),
            "accepted_examples": (self.accepted_examples),
            "skipped_records": (self.skipped_records),
            "duplicate_examples": (self.duplicate_examples),
        }


# ---------------------------------------------------------------
# Dataset assembler
# ---------------------------------------------------------------


class TrainingDatasetAssembler:
    """
    Assemble canonical TrainingExample objects
    from discovered training sources.
    """

    def __init__(
        self,
        *,
        config: Optional[TrainingConfig] = None,
        sources: Optional[TrainingSourceManager] = None,
        builder: Optional[ExampleBuilder] = None,
    ) -> None:
        self.config = config or TrainingConfig()

        self.config.validate()

        self.sources = sources or TrainingSourceManager(config=self.config)

        self.builder = builder or default_example_builder

    # ---------------------------------------------------------
    # Task guard
    # ---------------------------------------------------------

    def _require_task(
        self,
        task: str,
    ) -> str:
        normalized = str(task).strip().lower()

        if not self.config.task_allowed(normalized):
            raise TrainingTaskMismatchError(
                "Training task is not enabled.",
                details={
                    "task": (normalized),
                },
            )

        return normalized

    # ---------------------------------------------------------
    # Assemble discovered sources
    # ---------------------------------------------------------

    def assemble(
        self,
        *,
        task: str,
        dataset_id: str,
        version: str = "1",
        discovered_sources: Optional[Iterable[DiscoveredTrainingSource]] = None,
        require_labels: bool = True,
    ) -> AssemblyResult:
        """
        Combine eligible source records into one
        canonical TrainingDataset.

        Duplicate examples are ignored after the
        first occurrence, but counted.
        """

        task = self._require_task(task)

        sources = list(
            discovered_sources
            if discovered_sources is not None
            else self.sources.discover()
        )

        if not sources:
            raise EmptyTrainingDatasetError(
                "No eligible training sources were discovered."
            )

        examples: List[TrainingExample] = []

        source_records = 0
        skipped_records = 0
        duplicate_examples = 0

        example_ids = set()

        source_contracts: List[TrainingSource] = []

        source_keys = set()

        for discovered in sources:
            source = discovered.source.finalize()

            source_key = source.source_id or (source.kind + ":" + source.location)

            if source_key not in source_keys:
                source_keys.add(source_key)

                source_contracts.append(source)

            for record in self.sources.iter_records(discovered):
                source_records += 1

                try:
                    example = self.builder(
                        record,
                        task,
                    )

                except Exception as exc:
                    if isinstance(
                        exc,
                        DatasetAssemblyError,
                    ):
                        raise

                    raise DatasetAssemblyError(
                        "Training example builder failed.",
                        details={
                            "task": task,
                            "source_id": (source.source_id),
                            "error_type": (type(exc).__name__),
                            "error_module": (type(exc).__module__),
                        },
                    ) from exc

                if example is None:
                    skipped_records += 1

                    continue

                example.finalize()

                if example.task != task:
                    raise (
                        TrainingTaskMismatchError(
                            "Assembled example task does not match the requested task."
                        )
                    )

                if require_labels and example.label is None:
                    raise MissingLabelError(
                        "Training example does not contain a required label.",
                        details={
                            "example_id": (example.example_id),
                        },
                    )

                if example.example_id in example_ids:
                    duplicate_examples += 1

                    continue

                example_ids.add(example.example_id)

                examples.append(example)

        if not examples:
            raise EmptyTrainingDatasetError(
                "Training source records produced no usable examples.",
                details={
                    "source_records": (source_records),
                    "skipped_records": (skipped_records),
                },
            )

        dataset = TrainingDataset(
            dataset_id=(dataset_id),
            task=task,
            examples=examples,
            sources=(source_contracts),
            version=version,
            attributes={
                "source_record_count": (source_records),
                "accepted_example_count": (len(examples)),
                "skipped_record_count": (skipped_records),
                "duplicate_example_count": (duplicate_examples),
                "source_count": (len(source_contracts)),
                "assembly_stage": ("canonical_examples"),
            },
        ).finalize()

        return AssemblyResult(
            dataset=dataset,
            source_records=(source_records),
            accepted_examples=len(examples),
            skipped_records=(skipped_records),
            duplicate_examples=(duplicate_examples),
        )

    # ---------------------------------------------------------
    # Assemble explicit records
    # ---------------------------------------------------------

    def assemble_records(
        self,
        records: Iterable[
            Dict[
                str,
                Any,
            ]
        ],
        *,
        task: str,
        dataset_id: str,
        version: str = "1",
        require_labels: bool = True,
    ) -> TrainingDataset:
        """
        Assemble caller-provided records.

        Useful for tests and future specialized
        ingestion pipelines.
        """

        self._require_task(task)

        examples: List[TrainingExample] = []

        seen = set()

        for record in records:
            if not isinstance(
                record,
                dict,
            ):
                raise DatasetAssemblyError(
                    "Training assembly record must be a dictionary."
                )

            example = self.builder(
                record,
                task,
            )

            if example is None:
                continue

            example.finalize()

            if require_labels and example.label is None:
                raise MissingLabelError(
                    "Training example does not contain a required label."
                )

            if example.example_id in seen:
                raise (
                    DuplicateTrainingExampleError(
                        "Duplicate training example detected.",
                        details={
                            "example_id": (example.example_id),
                        },
                    )
                )

            seen.add(example.example_id)

            examples.append(example)

        if not examples:
            raise EmptyTrainingDatasetError(
                "No usable training examples were assembled."
            )

        return TrainingDataset(
            dataset_id=dataset_id,
            task=task,
            examples=examples,
            version=version,
            attributes={
                "assembly_stage": ("canonical_examples"),
                "explicit_records": (True),
            },
        ).finalize()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "allowed_tasks": list(self.config.allowed_tasks),
            "source_manager": (self.sources.summary()),
            "builder": (
                getattr(
                    self.builder,
                    "__name__",
                    type(self.builder).__name__,
                )
            ),
            "feature_stage": ("assembled_not_encoded"),
            "trains_models": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "ExampleBuilder",
    "LABEL_FIELDS",
    "PROVIDER_FIELDS",
    "ENVIRONMENT_FIELDS",
    "RESOURCE_TYPE_FIELDS",
    "RESOURCE_ID_FIELDS",
    "INCIDENT_ID_FIELDS",
    "GROUP_ID_FIELDS",
    "TIMESTAMP_FIELDS",
    "deterministic_example_id",
    "derive_group_id",
    "default_example_builder",
    "AssemblyResult",
    "TrainingDatasetAssembler",
]
