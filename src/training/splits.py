from __future__ import annotations

"""
Leakage-safe training dataset splitting for OSHA.

Important ordering:

    assemble examples
        ↓
    split examples              <- this module
        ↓
    fit feature schema on train only
        ↓
    transform train
    transform validation
    transform test

Related examples sharing a group_id remain in
the same partition.
"""

import hashlib
import random

from dataclasses import (
    dataclass,
)

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from src.data.config import (
    SplitConfig,
)

from src.training.config import (
    TrainingConfig,
)

from src.training.errors import (
    EmptyTrainingDatasetError,
    MissingGroupError,
    SplitOverlapError,
    TrainingLeakageError,
    TrainingSplitError,
    TrainingTaskMismatchError,
)

from src.training.schema import (
    EvaluationSplit,
    TrainingDataset,
    TrainingExample,
)


# ---------------------------------------------------------------
# Training split result
# ---------------------------------------------------------------


@dataclass(frozen=True)
class TrainingSplitResult:
    """
    Non-overlapping training dataset partitions.
    """

    dataset_id: str

    task: str

    train: Tuple[
        TrainingExample,
        ...,
    ]

    validation: Tuple[
        TrainingExample,
        ...,
    ]

    test: Tuple[
        TrainingExample,
        ...,
    ]

    random_seed: int

    grouped: bool

    shuffled: bool

    def validate(
        self,
    ) -> None:
        if not self.dataset_id:
            raise TrainingSplitError("Split result requires dataset_id.")

        if not self.task:
            raise TrainingSplitError("Split result requires task.")

        all_examples = [
            *self.train,
            *self.validation,
            *self.test,
        ]

        if not all_examples:
            raise EmptyTrainingDatasetError("Training split contains no examples.")

        if not self.train:
            raise TrainingSplitError("Training partition cannot be empty.")

        for example in all_examples:
            example.finalize()

            if example.task != self.task:
                raise (
                    TrainingTaskMismatchError(
                        "Split contains examples from a different task."
                    )
                )

        verify_split_integrity(self)

    @property
    def total_count(
        self,
    ) -> int:
        return len(self.train) + len(self.validation) + len(self.test)

    @property
    def counts(
        self,
    ) -> Dict[str, int]:
        return {
            EvaluationSplit.TRAIN.value: len(self.train),
            EvaluationSplit.VALIDATION.value: len(self.validation),
            EvaluationSplit.TEST.value: len(self.test),
        }

    def split(
        self,
        name: str,
    ) -> Tuple[
        TrainingExample,
        ...,
    ]:
        normalized = str(name).strip().lower()

        if normalized == EvaluationSplit.TRAIN.value:
            return self.train

        if normalized == EvaluationSplit.VALIDATION.value:
            return self.validation

        if normalized == EvaluationSplit.TEST.value:
            return self.test

        raise TrainingSplitError(
            "Unknown training split.",
            details={
                "split": normalized,
            },
        )

    def to_dict(
        self,
        *,
        include_example_ids: bool = True,
    ) -> Dict[str, Any]:
        self.validate()

        result: Dict[
            str,
            Any,
        ] = {
            "dataset_id": (self.dataset_id),
            "task": self.task,
            "total_count": (self.total_count),
            "counts": (self.counts),
            "random_seed": (self.random_seed),
            "grouped": (self.grouped),
            "shuffled": (self.shuffled),
        }

        if include_example_ids:
            result["example_ids"] = {
                EvaluationSplit.TRAIN.value: [
                    example.example_id for example in self.train
                ],
                EvaluationSplit.VALIDATION.value: [
                    example.example_id for example in self.validation
                ],
                EvaluationSplit.TEST.value: [
                    example.example_id for example in self.test
                ],
            }

        return result


# ---------------------------------------------------------------
# Stable fallback grouping
# ---------------------------------------------------------------


def fallback_group_id(
    example: TrainingExample,
) -> str:
    """
    Produce a deterministic fallback group for an
    example without explicit group_id.

    The fallback is example-specific. It prevents
    accidental loss of an example while avoiding
    the dangerous assumption that every example
    from one long-lived resource must share a
    single group.
    """

    example.finalize()

    if not example.example_id:
        raise MissingGroupError(
            "Training example requires example_id before fallback grouping."
        )

    digest = hashlib.sha256(str(example.example_id).encode("utf-8")).hexdigest()

    return "example:" + digest


# ---------------------------------------------------------------
# Effective group
# ---------------------------------------------------------------


def effective_group_id(
    example: TrainingExample,
    *,
    require_explicit: bool = False,
) -> str:
    example.finalize()

    if example.group_id:
        return str(example.group_id)

    if require_explicit:
        raise MissingGroupError(
            "Training example has no explicit group_id.",
            details={
                "example_id": (example.example_id),
                "incident_id": (example.incident_id),
            },
        )

    return fallback_group_id(example)


# ---------------------------------------------------------------
# Validate examples
# ---------------------------------------------------------------


def _validate_examples(
    examples: Sequence[TrainingExample],
) -> Tuple[
    str,
    str,
]:
    if not examples:
        raise EmptyTrainingDatasetError("Cannot split an empty training dataset.")

    dataset_ids = set()
    tasks = set()
    example_ids = set()

    for example in examples:
        if not isinstance(
            example,
            TrainingExample,
        ):
            raise TrainingSplitError(
                "Split input contains a non-TrainingExample value."
            )

        example.finalize()

        tasks.add(example.task)

        if example.dataset_id:
            dataset_ids.add(example.dataset_id)

        if not example.example_id:
            raise TrainingSplitError("Training example requires example_id.")

        if example.example_id in example_ids:
            raise TrainingLeakageError(
                "Duplicate training example_id detected.",
                details={
                    "example_id": (example.example_id),
                },
            )

        example_ids.add(example.example_id)

    if len(tasks) != 1:
        raise TrainingTaskMismatchError(
            "Training examples contain multiple tasks.",
            details={
                "tasks": sorted(tasks),
            },
        )

    if len(dataset_ids) > 1:
        raise TrainingSplitError(
            "Training examples contain multiple source dataset IDs.",
            details={
                "dataset_ids": sorted(dataset_ids),
            },
        )

    task = next(iter(tasks))

    dataset_id = next(iter(dataset_ids)) if dataset_ids else "assembled"

    return (
        dataset_id,
        task,
    )


# ---------------------------------------------------------------
# Ratio counts
# ---------------------------------------------------------------


def calculate_partition_counts(
    total: int,
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> Dict[str, int]:
    """
    Calculate partition counts.

    Validation and test use floor allocation.
    Train receives the remainder so every item is
    assigned exactly once.
    """

    if total < 1:
        raise TrainingSplitError("Partition total must be >= 1.")

    total_ratio = float(train_ratio) + float(validation_ratio) + float(test_ratio)

    if abs(total_ratio - 1.0) > 1e-9:
        raise TrainingSplitError("Training split ratios must sum to 1.0.")

    if train_ratio <= 0:
        raise TrainingSplitError("Training ratio must be > 0.")

    validation_count = int(total * validation_ratio)

    test_count = int(total * test_ratio)

    train_count = total - validation_count - test_count

    if train_count < 1:
        raise TrainingSplitError("Split would produce an empty training partition.")

    return {
        EvaluationSplit.TRAIN.value: (train_count),
        EvaluationSplit.VALIDATION.value: (validation_count),
        EvaluationSplit.TEST.value: (test_count),
    }


# ---------------------------------------------------------------
# Example ID sets
# ---------------------------------------------------------------


def split_example_ids(
    result: TrainingSplitResult,
) -> Dict[
    str,
    Set[str],
]:
    return {
        EvaluationSplit.TRAIN.value: {
            str(example.example_id) for example in result.train
        },
        EvaluationSplit.VALIDATION.value: {
            str(example.example_id) for example in result.validation
        },
        EvaluationSplit.TEST.value: {
            str(example.example_id) for example in result.test
        },
    }


# ---------------------------------------------------------------
# Group sets
# ---------------------------------------------------------------


def split_group_ids(
    result: TrainingSplitResult,
) -> Dict[
    str,
    Set[str],
]:
    return {
        EvaluationSplit.TRAIN.value: {
            effective_group_id(example) for example in result.train
        },
        EvaluationSplit.VALIDATION.value: {
            effective_group_id(example) for example in result.validation
        },
        EvaluationSplit.TEST.value: {
            effective_group_id(example) for example in result.test
        },
    }


# ---------------------------------------------------------------
# Integrity / leakage verification
# ---------------------------------------------------------------


def verify_split_integrity(
    result: TrainingSplitResult,
) -> None:
    """
    Verify both example-level overlap and
    group-level leakage.
    """

    ids = split_example_ids(result)

    train_ids = ids[EvaluationSplit.TRAIN.value]

    validation_ids = ids[EvaluationSplit.VALIDATION.value]

    test_ids = ids[EvaluationSplit.TEST.value]

    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise SplitOverlapError("Training partitions contain overlapping example IDs.")

    groups = split_group_ids(result)

    train_groups = groups[EvaluationSplit.TRAIN.value]

    validation_groups = groups[EvaluationSplit.VALIDATION.value]

    test_groups = groups[EvaluationSplit.TEST.value]

    train_validation = train_groups & validation_groups

    train_test = train_groups & test_groups

    validation_test = validation_groups & test_groups

    if train_validation or train_test or validation_test:
        leaked_groups = sorted(train_validation | train_test | validation_test)

        raise TrainingLeakageError(
            "Related training groups appear in multiple partitions.",
            details={
                "group_count": len(leaked_groups),
                "groups": (leaked_groups[:100]),
            },
        )


# ---------------------------------------------------------------
# Grouped split
# ---------------------------------------------------------------


def split_grouped_examples(
    examples: Iterable[TrainingExample],
    *,
    split_config: SplitConfig,
    dataset_id: Optional[str] = None,
    require_explicit_groups: bool = False,
) -> TrainingSplitResult:
    """
    Split examples while ensuring each group
    remains entirely in one partition.

    Ratios are applied to GROUPS, not individual
    examples.

    Therefore example counts may not match the
    target ratios exactly when group sizes vary.
    """

    values = list(examples)

    (
        inferred_dataset_id,
        task,
    ) = _validate_examples(values)

    split_config.validate()

    groups: Dict[
        str,
        List[TrainingExample],
    ] = {}

    for example in values:
        group_id = effective_group_id(
            example,
            require_explicit=(require_explicit_groups),
        )

        groups.setdefault(
            group_id,
            [],
        ).append(example)

    group_ids = sorted(groups)

    if split_config.shuffle:
        randomizer = random.Random(split_config.random_seed)

        randomizer.shuffle(group_ids)

    counts = calculate_partition_counts(
        len(group_ids),
        train_ratio=(split_config.train_ratio),
        validation_ratio=(split_config.validation_ratio),
        test_ratio=(split_config.test_ratio),
    )

    train_end = counts[EvaluationSplit.TRAIN.value]

    validation_end = train_end + counts[EvaluationSplit.VALIDATION.value]

    train_group_ids = set(group_ids[:train_end])

    validation_group_ids = set(group_ids[train_end:validation_end])

    test_group_ids = set(group_ids[validation_end:])

    train: List[TrainingExample] = []

    validation: List[TrainingExample] = []

    test: List[TrainingExample] = []

    # Preserve deterministic group order and
    # within-group example order.
    for group_id in group_ids:
        members = groups[group_id]

        if group_id in train_group_ids:
            train.extend(members)

        elif group_id in validation_group_ids:
            validation.extend(members)

        elif group_id in test_group_ids:
            test.extend(members)

        else:
            raise TrainingSplitError(
                "Training group was not assigned to a partition.",
                details={
                    "group_id": group_id,
                },
            )

    result = TrainingSplitResult(
        dataset_id=(dataset_id or inferred_dataset_id),
        task=task,
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
        random_seed=(split_config.random_seed),
        grouped=True,
        shuffled=(split_config.shuffle),
    )

    if result.total_count != len(values):
        raise TrainingSplitError("Training split lost or duplicated examples.")

    result.validate()

    return result


# ---------------------------------------------------------------
# Training splitter
# ---------------------------------------------------------------


class TrainingSplitter:
    """
    High-level leakage-safe splitter.

    By default, related examples stay together.

    Examples with no explicit group receive an
    example-specific deterministic fallback group.
    """

    def __init__(
        self,
        *,
        config: Optional[TrainingConfig] = None,
        split_config: Optional[SplitConfig] = None,
        require_explicit_groups: bool = False,
    ) -> None:
        self.config = config or TrainingConfig()

        self.config.validate()

        # Training and data preparation should use
        # the same default split semantics unless a
        # training-specific override is provided.
        self.split_config = split_config or SplitConfig(
            random_seed=(self.config.reproducibility.random_seed)
        )

        self.split_config.validate()

        self.require_explicit_groups = bool(require_explicit_groups)

    # ---------------------------------------------------------
    # Split dataset
    # ---------------------------------------------------------

    def split(
        self,
        dataset: TrainingDataset,
    ) -> TrainingSplitResult:
        dataset.finalize()

        if not self.config.task_allowed(dataset.task):
            raise TrainingTaskMismatchError(
                "Training dataset task is not enabled.",
                details={
                    "task": dataset.task,
                },
            )

        return split_grouped_examples(
            dataset.examples,
            split_config=(self.split_config),
            dataset_id=(dataset.dataset_id),
            require_explicit_groups=(self.require_explicit_groups),
        )

    # ---------------------------------------------------------
    # Verify existing result
    # ---------------------------------------------------------

    def verify(
        self,
        result: TrainingSplitResult,
    ) -> None:
        result.validate()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "train_ratio": (self.split_config.train_ratio),
            "validation_ratio": (self.split_config.validation_ratio),
            "test_ratio": (self.split_config.test_ratio),
            "random_seed": (self.split_config.random_seed),
            "shuffle": (self.split_config.shuffle),
            "grouped": True,
            "require_explicit_groups": (self.require_explicit_groups),
            "fallback_grouping": (
                "per-example" if not self.require_explicit_groups else None
            ),
            "example_overlap_allowed": (False),
            "group_leakage_allowed": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "TrainingSplitResult",
    "fallback_group_id",
    "effective_group_id",
    "calculate_partition_counts",
    "split_example_ids",
    "split_group_ids",
    "verify_split_integrity",
    "split_grouped_examples",
    "TrainingSplitter",
]
