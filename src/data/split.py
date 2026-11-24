from __future__ import annotations

"""
Deterministic dataset splitting for OSHA.

Creates non-overlapping:

    train
    validation
    test

partitions from canonical DatasetRecord objects.
"""

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
)

from src.data.config import (
    SplitConfig,
)

from src.data.errors import (
    DataSplitConfigurationError,
    DataSplitError,
)

from src.data.schema import (
    DatasetRecord,
    DatasetSplit,
    SplitName,
)


# ---------------------------------------------------------------
# Split result
# ---------------------------------------------------------------


@dataclass
class SplitResult:
    """
    Complete result of one dataset split.
    """

    dataset_id: str

    train: List[DatasetRecord]

    validation: List[DatasetRecord]

    test: List[DatasetRecord]

    seed: int

    shuffled: bool

    def all_records(
        self,
    ) -> List[DatasetRecord]:
        return [
            *self.train,
            *self.validation,
            *self.test,
        ]

    @property
    def total_count(
        self,
    ) -> int:
        return len(self.train) + len(self.validation) + len(self.test)

    def metadata(
        self,
    ) -> List[DatasetSplit]:
        return [
            DatasetSplit(
                dataset_id=(self.dataset_id),
                name=(SplitName.TRAIN.value),
                record_ids=[
                    record.record_id
                    for record in self.train
                    if record.record_id is not None
                ],
                attributes={
                    "seed": (self.seed),
                    "shuffled": (self.shuffled),
                },
            ).finalize(),
            DatasetSplit(
                dataset_id=(self.dataset_id),
                name=(SplitName.VALIDATION.value),
                record_ids=[
                    record.record_id
                    for record in self.validation
                    if record.record_id is not None
                ],
                attributes={
                    "seed": (self.seed),
                    "shuffled": (self.shuffled),
                },
            ).finalize(),
            DatasetSplit(
                dataset_id=(self.dataset_id),
                name=(SplitName.TEST.value),
                record_ids=[
                    record.record_id
                    for record in self.test
                    if record.record_id is not None
                ],
                attributes={
                    "seed": (self.seed),
                    "shuffled": (self.shuffled),
                },
            ).finalize(),
        ]

    def to_dict(
        self,
        *,
        include_records: bool = False,
    ) -> Dict[str, Any]:
        result = {
            "dataset_id": (self.dataset_id),
            "total_count": (self.total_count),
            "train_count": len(self.train),
            "validation_count": len(self.validation),
            "test_count": len(self.test),
            "seed": self.seed,
            "shuffled": (self.shuffled),
            "splits": [item.to_dict() for item in self.metadata()],
        }

        if include_records:
            result["records"] = {
                "train": [record.to_dict() for record in self.train],
                "validation": [record.to_dict() for record in self.validation],
                "test": [record.to_dict() for record in self.test],
            }

        return result


# ---------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------


def _validate_records(
    records: Sequence[DatasetRecord],
) -> str:
    """
    Ensure all records belong to the same dataset
    and have unique record IDs.

    Returns the dataset_id.
    """

    if not records:
        raise DataSplitError("Cannot split an empty dataset.")

    dataset_ids = set()

    record_ids = set()

    for record in records:
        if not isinstance(
            record,
            DatasetRecord,
        ):
            raise DataSplitError(
                "Dataset split input contains a non-DatasetRecord value."
            )

        record.finalize()

        dataset_ids.add(record.dataset_id)

        if not record.record_id:
            raise DataSplitError("DatasetRecord requires record_id before splitting.")

        if record.record_id in record_ids:
            raise DataSplitError(
                "Duplicate DatasetRecord record_id detected.",
                details={
                    "record_id": (record.record_id),
                },
            )

        record_ids.add(record.record_id)

    if len(dataset_ids) != 1:
        raise DataSplitError(
            "Dataset split input contains records from multiple datasets.",
            details={
                "dataset_ids": sorted(dataset_ids),
            },
        )

    return next(iter(dataset_ids))


# ---------------------------------------------------------------
# Split counts
# ---------------------------------------------------------------


def calculate_split_counts(
    total: int,
    config: SplitConfig,
) -> Dict[str, int]:
    """
    Calculate deterministic split sizes.

    Train receives any remainder after validation
    and test allocations so that every record is
    assigned exactly once.
    """

    config.validate()

    if total < 1:
        raise DataSplitError("Split total must be >= 1.")

    validation_count = int(total * config.validation_ratio)

    test_count = int(total * config.test_ratio)

    train_count = total - validation_count - test_count

    if train_count < 1:
        raise (
            DataSplitConfigurationError(
                "Split configuration produces an empty training set.",
                details={
                    "total_records": (total),
                    "train_count": (train_count),
                    "validation_count": (validation_count),
                    "test_count": (test_count),
                },
            )
        )

    return {
        SplitName.TRAIN.value: (train_count),
        SplitName.VALIDATION.value: (validation_count),
        SplitName.TEST.value: (test_count),
    }


# ---------------------------------------------------------------
# Overlap verification
# ---------------------------------------------------------------


def verify_no_overlap(
    result: SplitResult,
) -> bool:
    """
    Confirm that no record ID appears in multiple
    partitions.
    """

    train_ids = {record.record_id for record in result.train}

    validation_ids = {record.record_id for record in result.validation}

    test_ids = {record.record_id for record in result.test}

    return not (
        train_ids.intersection(validation_ids)
        or train_ids.intersection(test_ids)
        or validation_ids.intersection(test_ids)
    )


# ---------------------------------------------------------------
# Record splitter
# ---------------------------------------------------------------


class DatasetSplitter:
    """
    Split canonical DatasetRecord objects into
    deterministic train/validation/test sets.

    The input list itself is not modified.
    """

    def __init__(
        self,
        *,
        config: Optional[SplitConfig] = None,
    ) -> None:
        self.config = config or SplitConfig()

        self.config.validate()

    # ---------------------------------------------------------
    # Split
    # ---------------------------------------------------------

    def split(
        self,
        records: Iterable[DatasetRecord],
    ) -> SplitResult:
        values = list(records)

        dataset_id = _validate_records(values)

        working = list(values)

        if self.config.shuffle:
            randomizer = random.Random(self.config.random_seed)

            randomizer.shuffle(working)

        counts = calculate_split_counts(
            len(working),
            self.config,
        )

        train_end = counts[SplitName.TRAIN.value]

        validation_end = train_end + counts[SplitName.VALIDATION.value]

        train = working[:train_end]

        validation = working[train_end:validation_end]

        test = working[validation_end:]

        result = SplitResult(
            dataset_id=(dataset_id),
            train=train,
            validation=validation,
            test=test,
            seed=(self.config.random_seed),
            shuffled=(self.config.shuffle),
        )

        # -----------------------------------------------------
        # Internal invariants
        # -----------------------------------------------------

        if result.total_count != len(values):
            raise DataSplitError("Dataset split lost or duplicated records.")

        if not verify_no_overlap(result):
            raise DataSplitError("Dataset split contains overlapping records.")

        return result

    # ---------------------------------------------------------
    # Ordered split
    # ---------------------------------------------------------

    def split_ordered(
        self,
        records: Iterable[DatasetRecord],
    ) -> SplitResult:
        """
        Split records in the existing order,
        regardless of configured shuffle policy.

        Useful for callers that have already
        ordered records intentionally.
        """

        config = SplitConfig(
            train_ratio=(self.config.train_ratio),
            validation_ratio=(self.config.validation_ratio),
            test_ratio=(self.config.test_ratio),
            random_seed=(self.config.random_seed),
            shuffle=False,
        )

        return DatasetSplitter(config=config).split(records)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "train_ratio": (self.config.train_ratio),
            "validation_ratio": (self.config.validation_ratio),
            "test_ratio": (self.config.test_ratio),
            "random_seed": (self.config.random_seed),
            "shuffle": (self.config.shuffle),
            "duplicate_record_ids_allowed": (False),
            "split_overlap_allowed": (False),
        }


# ---------------------------------------------------------------
# Group-aware deterministic split
# ---------------------------------------------------------------


def split_by_group(
    records: Iterable[DatasetRecord],
    groups: Mapping[
        str,
        str,
    ],
    *,
    config: Optional[SplitConfig] = None,
) -> SplitResult:
    """
    Keep records belonging to the same group in
    the same partition.

    groups maps:

        record_id -> group_id

    This is useful for preventing leakage when
    multiple records belong to the same incident,
    sequence, host, session, or other logical
    group.
    """

    values = list(records)

    dataset_id = _validate_records(values)

    split_config = config or SplitConfig()

    split_config.validate()

    grouped: Dict[
        str,
        List[DatasetRecord],
    ] = {}

    for record in values:
        record_id = record.record_id

        if record_id not in groups:
            raise DataSplitError(
                "Dataset record has no group assignment.",
                details={
                    "record_id": (record_id),
                },
            )

        group_id = str(groups[record_id]).strip()

        if not group_id:
            raise DataSplitError(
                "Dataset record group ID cannot be empty.",
                details={
                    "record_id": (record_id),
                },
            )

        grouped.setdefault(
            group_id,
            [],
        ).append(record)

    group_ids = sorted(grouped)

    if split_config.shuffle:
        randomizer = random.Random(split_config.random_seed)

        randomizer.shuffle(group_ids)

    counts = calculate_split_counts(
        len(group_ids),
        split_config,
    )

    train_group_end = counts[SplitName.TRAIN.value]

    validation_group_end = train_group_end + counts[SplitName.VALIDATION.value]

    train_groups = set(group_ids[:train_group_end])

    validation_groups = set(group_ids[train_group_end:validation_group_end])

    test_groups = set(group_ids[validation_group_end:])

    train: List[DatasetRecord] = []

    validation: List[DatasetRecord] = []

    test: List[DatasetRecord] = []

    for group_id in group_ids:
        target = (
            train
            if group_id in train_groups
            else (validation if group_id in validation_groups else test)
        )

        target.extend(grouped[group_id])

    result = SplitResult(
        dataset_id=(dataset_id),
        train=train,
        validation=validation,
        test=test,
        seed=(split_config.random_seed),
        shuffled=(split_config.shuffle),
    )

    if result.total_count != len(values):
        raise DataSplitError("Grouped dataset split lost or duplicated records.")

    if not verify_no_overlap(result):
        raise DataSplitError("Grouped dataset split contains overlapping records.")

    return result


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "SplitResult",
    "calculate_split_counts",
    "verify_no_overlap",
    "DatasetSplitter",
    "split_by_group",
]
