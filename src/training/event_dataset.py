from __future__ import annotations

"""
Dataset utilities for OSHA infrastructure-event
model training.

The event model is sequence/text based.

Pipeline:

    TrainingDataset
        -> TrainingSplitter
        -> TrainingSplitResult
        -> event records
        -> Vocabulary fitted on TRAIN
        -> LabelMaps fitted on TRAIN
        -> EventDataset
        -> InfrastructureEventModel

The PyTorch batch contract remains:

    input_ids
    attention_mask
    source_target
    component_target
    event_target
"""

import hashlib
import json
import re

from collections import Counter

from pathlib import Path

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

import torch

from torch.utils.data import Dataset

from src.training.errors import (
    EmptyTrainingDatasetError,
    InvalidLabelError,
    TrainingSourceFormatError,
)

from src.training.schema import (
    TrainingExample,
    TrainingTask,
)

from src.training.splits import (
    TrainingSplitResult,
)


# ---------------------------------------------------------------
# Reserved IDs
# ---------------------------------------------------------------

PAD_ID = 0

UNK_ID = 1

IGNORE_INDEX = -100


# ---------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------

def tokenize(
    text: str,
) -> List[str]:
    """
    Lightweight tokenizer suitable for
    infrastructure logs.

    Technical punctuation remains available
    because paths, addresses, service names,
    versions, and infrastructure identifiers
    carry useful information.
    """

    if not text:
        return []

    return re.findall(
        r"[A-Za-z_][A-Za-z0-9_.:/-]*|"
        r"0x[0-9A-Fa-f]+|"
        r"\d+(?:\.\d+)?|"
        r"[^\s]",
        str(
            text
        ),
    )


# ---------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------

def _text(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    result = str(
        value
    ).strip()

    return result or None


def _mapping(
    value: Any,
) -> Dict[str, Any]:
    if isinstance(
        value,
        Mapping,
    ):
        return dict(
            value
        )

    return {}


# ---------------------------------------------------------------
# Event record validation
# ---------------------------------------------------------------

def is_event_record(
    record: Mapping[
        str,
        Any,
    ],
) -> bool:
    """
    Return True if a raw event record contains
    the minimum fields required by the existing
    event model.
    """

    record_type = _text(
        record.get(
            "record_type"
        )
    )

    if (
        record_type is not None
        and record_type
        not in {
            "log_event",
            "event",
        }
    ):
        return False

    content = _text(
        record.get(
            "content"
        )
    )

    source = _text(
        record.get(
            "source_dataset"
        )
    )

    event_id = _text(
        record.get(
            "event_id"
        )
    )

    return bool(
        content
        and source
        and event_id
    )


# ---------------------------------------------------------------
# Legacy JSONL loader
# ---------------------------------------------------------------

def load_event_records(
    path: str | Path,
    max_records: Optional[
        int
    ] = None,
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    """
    Load legacy event JSONL records.

    This remains for compatibility with
    train_events.py while the trainer is migrated
    onto TrainingPipeline.
    """

    source = Path(
        path
    ).expanduser()

    if (
        max_records is not None
        and max_records < 1
    ):
        raise ValueError(
            "max_records must be >= 1"
        )

    if not source.exists():
        raise TrainingSourceFormatError(
            "Event training source was "
            "not found.",
            details={
                "path": str(
                    source
                ),
            },
        )

    if not source.is_file():
        raise TrainingSourceFormatError(
            "Event training source is "
            "not a regular file.",
            details={
                "path": str(
                    source
                ),
            },
        )

    records: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    try:
        with source.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                text = line.strip()

                if not text:
                    continue

                try:
                    value = json.loads(
                        text
                    )

                except json.JSONDecodeError as exc:
                    raise TrainingSourceFormatError(
                        "Event training "
                        "JSONL contains "
                        "invalid JSON.",
                        details={
                            "path": str(
                                source
                            ),
                            "line_number": (
                                line_number
                            ),
                            "column": (
                                exc.colno
                            ),
                        },
                    ) from exc

                if not isinstance(
                    value,
                    Mapping,
                ):
                    continue

                record = dict(
                    value
                )

                if not is_event_record(
                    record
                ):
                    continue

                records.append(
                    record
                )

                if (
                    max_records is not None
                    and len(
                        records
                    )
                    >= max_records
                ):
                    break

    except TrainingSourceFormatError:
        raise

    except (
        OSError,
        UnicodeError,
    ) as exc:
        raise TrainingSourceFormatError(
            "Event training source "
            "could not be read.",
            details={
                "path": str(
                    source
                ),
                "error_type": (
                    type(exc).__name__
                ),
            },
        ) from exc

    if not records:
        raise EmptyTrainingDatasetError(
            "No usable event records "
            "were found.",
            details={
                "path": str(
                    source
                ),
            },
        )

    return records


# ---------------------------------------------------------------
# Event grouping
# ---------------------------------------------------------------

def event_group_key(
    record: Mapping[
        str,
        Any,
    ],
) -> str:
    """
    Determine a leakage-aware grouping key.

    Explicit incident/sequence/trace identity
    takes precedence over the legacy record
    identity.
    """

    for field in (
        "group_id",
        "incident_id",
        "sequence_id",
        "session_id",
        "trace_id",
    ):
        value = _text(
            record.get(
                field
            )
        )

        if value:
            return (
                field
                + ":"
                + value
            )

    return "|".join(
        (
            _text(
                record.get(
                    "source_dataset"
                )
            )
            or "",
            _text(
                record.get(
                    "source_file"
                )
            )
            or "",
            _text(
                record.get(
                    "line_id"
                )
            )
            or "",
            _text(
                record.get(
                    "event_id"
                )
            )
            or "",
        )
    )


# ---------------------------------------------------------------
# Legacy deterministic split
# ---------------------------------------------------------------

def split_records(
    records: Iterable[
        Mapping[
            str,
            Any,
        ]
    ],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Backward-compatible deterministic split.

    New training should prefer TrainingSplitter,
    but existing train_events.py can continue
    using this function during migration.
    """

    if not (
        0.0
        < train_ratio
        < 1.0
    ):
        raise ValueError(
            "train_ratio must be "
            "between 0 and 1"
        )

    if not (
        0.0
        <= val_ratio
        < 1.0
    ):
        raise ValueError(
            "val_ratio must be "
            "between 0 and 1"
        )

    if (
        train_ratio
        + val_ratio
        >= 1.0
    ):
        raise ValueError(
            "train_ratio + val_ratio "
            "must be less than 1"
        )

    train: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    validation: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    test: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    train_cutoff = int(
        train_ratio
        * 10000
    )

    validation_cutoff = int(
        (
            train_ratio
            + val_ratio
        )
        * 10000
    )

    for source_record in records:
        record = dict(
            source_record
        )

        key = event_group_key(
            record
        )

        digest = hashlib.sha256(
            key.encode(
                "utf-8"
            )
        ).hexdigest()

        bucket = (
            int(
                digest[
                    :8
                ],
                16,
            )
            % 10000
        )

        if bucket < train_cutoff:
            train.append(
                record
            )

        elif bucket < validation_cutoff:
            validation.append(
                record
            )

        else:
            test.append(
                record
            )

    if not train:
        raise EmptyTrainingDatasetError(
            "Event split produced an "
            "empty training partition."
        )

    return (
        train,
        validation,
        test,
    )


# ---------------------------------------------------------------
# TrainingExample -> event record
# ---------------------------------------------------------------

def training_example_to_event_record(
    example: TrainingExample,
) -> Dict[str, Any]:
    """
    Convert a canonical TrainingExample to the
    record contract expected by the event model.
    """

    example.finalize()

    if (
        example.task
        != TrainingTask.EVENT.value
    ):
        raise InvalidLabelError(
            "Training example is not an "
            "event task.",
            details={
                "example_id": (
                    example.example_id
                ),
                "task": (
                    example.task
                ),
            },
        )

    features = dict(
        example.features
    )

    payload = _mapping(
        features.get(
            "payload"
        )
    )

    content = (
        _text(
            features.get(
                "content"
            )
        )
        or _text(
            features.get(
                "message"
            )
        )
        or _text(
            features.get(
                "text"
            )
        )
        or _text(
            payload.get(
                "content"
            )
        )
        or _text(
            payload.get(
                "message"
            )
        )
        or _text(
            payload.get(
                "text"
            )
        )
    )

    source_dataset = (
        _text(
            features.get(
                "source_dataset"
            )
        )
        or example.dataset_id
        or example.source_id
        or example.provider
    )

    event_id = (
        _text(
            features.get(
                "event_id"
            )
        )
        or _text(
            features.get(
                "event_type"
            )
        )
        or _text(
            payload.get(
                "event_id"
            )
        )
        or _text(
            payload.get(
                "event_type"
            )
        )
        or _text(
            example.label
        )
    )

    if not content:
        raise InvalidLabelError(
            "Event training example has "
            "no textual content.",
            details={
                "example_id": (
                    example.example_id
                ),
            },
        )

    if not source_dataset:
        raise InvalidLabelError(
            "Event training example has "
            "no source identity.",
            details={
                "example_id": (
                    example.example_id
                ),
            },
        )

    if not event_id:
        raise InvalidLabelError(
            "Event training example has "
            "no event identity.",
            details={
                "example_id": (
                    example.example_id
                ),
            },
        )

    metadata = _mapping(
        features.get(
            "metadata"
        )
    )

    component = (
        _text(
            metadata.get(
                "Component"
            )
        )
        or _text(
            metadata.get(
                "component"
            )
        )
        or _text(
            features.get(
                "component"
            )
        )
        or _text(
            payload.get(
                "component"
            )
        )
        or example.resource_type
    )

    if component:
        metadata[
            "Component"
        ] = component

    return {
        "record_type": (
            "log_event"
        ),

        "content": (
            content
        ),

        "source_dataset": (
            str(
                source_dataset
            )
        ),

        "event_id": (
            str(
                event_id
            )
        ),

        "metadata": (
            metadata
        ),

        "source_file": (
            example.attributes.get(
                "source_file"
            )
        ),

        "line_id": (
            example.example_id
        ),

        "group_id": (
            example.group_id
        ),

        "incident_id": (
            example.incident_id
        ),

        "provider": (
            example.provider
        ),

        "environment_id": (
            example.environment_id
        ),

        "resource_id": (
            example.resource_id
        ),

        "reference_ids": list(
            example.reference_ids
        ),
    }


# ---------------------------------------------------------------
# Convert multiple examples
# ---------------------------------------------------------------

def event_records_from_examples(
    examples: Iterable[
        TrainingExample
    ],
    *,
    skip_invalid: bool = False,
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    records: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for example in examples:
        try:
            record = (
                training_example_to_event_record(
                    example
                )
            )

        except InvalidLabelError:
            if skip_invalid:
                continue

            raise

        records.append(
            record
        )

    if not records:
        raise EmptyTrainingDatasetError(
            "No usable event examples "
            "were available."
        )

    return records


# ---------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------

class LabelMaps:
    """
    Multi-head event classification mappings.

    IMPORTANT:
        Fit LabelMaps on TRAIN records only.

    Unknown validation/test targets map to
    IGNORE_INDEX rather than causing KeyError.
    """

    def __init__(
        self,
        records: Iterable[
            Mapping[
                str,
                Any,
            ]
        ],
    ) -> None:
        values = [
            dict(
                record
            )
            for record
            in records
        ]

        if not values:
            raise EmptyTrainingDatasetError(
                "Cannot build event labels "
                "from an empty training "
                "partition."
            )

        sources = sorted(
            {
                str(
                    record[
                        "source_dataset"
                    ]
                )
                for record
                in values
                if _text(
                    record.get(
                        "source_dataset"
                    )
                )
            }
        )

        components = sorted(
            {
                str(
                    (
                        record.get(
                            "metadata"
                        )
                        or {}
                    ).get(
                        "Component"
                    )
                )
                for record
                in values
                if _text(
                    (
                        record.get(
                            "metadata"
                        )
                        or {}
                    ).get(
                        "Component"
                    )
                )
            }
        )

        events = sorted(
            {
                (
                    str(
                        record[
                            "source_dataset"
                        ]
                    )
                    + ":"
                    + str(
                        record[
                            "event_id"
                        ]
                    )
                )
                for record
                in values
                if (
                    _text(
                        record.get(
                            "source_dataset"
                        )
                    )
                    and _text(
                        record.get(
                            "event_id"
                        )
                    )
                )
            }
        )

        if not sources:
            raise InvalidLabelError(
                "Training partition "
                "contains no source "
                "classes."
            )

        if not events:
            raise InvalidLabelError(
                "Training partition "
                "contains no event "
                "classes."
            )

        self.source_to_id = {
            value: index
            for index, value
            in enumerate(
                sources
            )
        }

        self.component_to_id = {
            value: index
            for index, value
            in enumerate(
                components
            )
        }

        self.event_to_id = {
            value: index
            for index, value
            in enumerate(
                events
            )
        }

        self._rebuild_reverse_maps()

    def _rebuild_reverse_maps(
        self,
    ) -> None:
        self.id_to_source = {
            index: value
            for value, index
            in self.source_to_id.items()
        }

        self.id_to_component = {
            index: value
            for value, index
            in self.component_to_id.items()
        }

        self.id_to_event = {
            index: value
            for value, index
            in self.event_to_id.items()
        }

    def source_id(
        self,
        source_name: Optional[
            str
        ],
    ) -> int:
        if source_name is None:
            return IGNORE_INDEX

        return self.source_to_id.get(
            str(
                source_name
            ),
            IGNORE_INDEX,
        )

    def component_id(
        self,
        component_name: Optional[
            str
        ],
    ) -> int:
        if component_name is None:
            return IGNORE_INDEX

        return (
            self.component_to_id.get(
                str(
                    component_name
                ),
                IGNORE_INDEX,
            )
        )

    def event_id(
        self,
        source_name: Optional[
            str
        ],
        event_id: Optional[
            str
        ],
    ) -> int:
        if (
            source_name is None
            or event_id is None
        ):
            return IGNORE_INDEX

        name = (
            str(
                source_name
            )
            + ":"
            + str(
                event_id
            )
        )

        return self.event_to_id.get(
            name,
            IGNORE_INDEX,
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "source_to_id": dict(
                self.source_to_id
            ),

            "component_to_id": dict(
                self.component_to_id
            ),

            "event_to_id": dict(
                self.event_to_id
            ),

            "ignore_index": (
                IGNORE_INDEX
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[
            str,
            Any,
        ],
    ) -> "LabelMaps":
        obj = cls.__new__(
            cls
        )

        obj.source_to_id = {
            str(
                key
            ): int(
                value
            )
            for key, value
            in dict(
                data[
                    "source_to_id"
                ]
            ).items()
        }

        obj.component_to_id = {
            str(
                key
            ): int(
                value
            )
            for key, value
            in dict(
                data.get(
                    "component_to_id",
                    {},
                )
            ).items()
        }

        obj.event_to_id = {
            str(
                key
            ): int(
                value
            )
            for key, value
            in dict(
                data[
                    "event_to_id"
                ]
            ).items()
        }

        obj._rebuild_reverse_maps()

        return obj


# ---------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------

class Vocabulary:
    """
    Infrastructure-event text vocabulary.

    IMPORTANT:
        Fit Vocabulary on TRAIN text only.
    """

    def __init__(
        self,
        records: Iterable[
            Mapping[
                str,
                Any,
            ]
        ],
        min_frequency: int = 2,
        max_size: int = 50000,
    ) -> None:
        if min_frequency < 1:
            raise ValueError(
                "min_frequency must be >= 1"
            )

        if max_size < 2:
            raise ValueError(
                "max_size must be >= 2"
            )

        counter: Counter[str] = (
            Counter()
        )

        for record in records:
            counter.update(
                tokenize(
                    str(
                        record.get(
                            "content",
                            "",
                        )
                    )
                )
            )

        self.token_to_id = {
            "<PAD>": PAD_ID,
            "<UNK>": UNK_ID,
        }

        candidates = [
            (
                token,
                count,
            )
            for token, count
            in counter.items()
            if count >= min_frequency
        ]

        candidates.sort(
            key=lambda item: (
                -item[
                    1
                ],
                item[
                    0
                ],
            )
        )

        available = max(
            0,
            max_size
            - len(
                self.token_to_id
            ),
        )

        for token, _ in candidates[
            :available
        ]:
            self.token_to_id[
                token
            ] = len(
                self.token_to_id
            )

        self._rebuild_reverse_map()

    def _rebuild_reverse_map(
        self,
    ) -> None:
        self.id_to_token = {
            index: token
            for token, index
            in self.token_to_id.items()
        }

    def encode(
        self,
        text: str,
        max_length: int,
    ) -> Tuple[
 [int],
        List[int],
    ]:
        if max_length < 1:
            raise ValueError(
                "max_length must be >= 1"
            )

        tokens = tokenize(
            text
        )[
            :max_length
        ]

        ids = [
            self.token_to_id.get(
                token,
                UNK_ID,
            )
            for token
            in tokens
        ]

        mask = [
            1
        ] * len(
            ids
        )

        if (
            len(
                ids
            )
            < max_length
        ):
            padding = (
                max_length
                - len(
                    ids
                )
            )

            ids.extend(
                [
                    PAD_ID
                ]
                * padding
            )

            mask.extend(
                [
                    0
                ]
                * padding
            )

        return (
            ids,
            mask,
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "token_to_id": dict(
                self.token_to_id
            ),

            "pad_id": PAD_ID,

            "unknown_id": (
                UNK_ID
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[
            str,
            Any,
        ],
    ) -> "Vocabulary":
        obj = cls.__new__(
            cls
        )

        obj.token_to_id = {
            str(
                key
            ): int(
                value
            )
            for key, value
            in dict(
                data[
                    "token_to_id"
                ]
            ).items()
        }

        obj._rebuild_reverse_map()

        return obj

    def __len__(
        self,
    ) -> int:
        return len(
            self.token_to_id
        )


# ---------------------------------------------------------------
# PyTorch event dataset
# ---------------------------------------------------------------

class EventDataset(
    Dataset
):
    """
    Dataset compatible with your existing
    InfrastructureEventModel.

    Returned dictionary:

        input_ids
        attention_mask
        source_target
        component_target
        event_target
    """

    def __init__(
        self,
        records: Iterable[
            Mapping[
                str,
                Any,
            ]
        ],
        vocabulary: Vocabulary,
        labels: LabelMaps,
        max_length: int = 128,
    ) -> None:
        if max_length < 1:
            raise ValueError(
                "max_length must be >= 1"
            )

        self.records = [
            dict(
                record
            )
            for record
            in records
            if is_event_record(
                record
            )
        ]

        if not self.records:
            raise EmptyTrainingDatasetError(
                "EventDataset contains no "
                "usable event records."
            )

        self.vocabulary = (
            vocabulary
        )

        self.labels = labels

        self.max_length = int(
            max_length
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.records
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        record = self.records[
            index
        ]

        content = str(
            record[
                "content"
            ]
        )

        (
            input_ids,
            attention_mask,
        ) = self.vocabulary.encode(
            content,
            self.max_length,
        )

        source_name = str(
            record[
                "source_dataset"
            ]
        )

        raw_event_id = str(
            record[
                "event_id"
            ]
        )

        metadata = _mapping(
            record.get(
                "metadata"
            )
        )

        component_name = _text(
            metadata.get(
                "Component"
            )
        )

        source_target = (
            self.labels.source_id(
                source_name
            )
        )

        component_target = (
            self.labels.component_id(
                component_name
            )
        )

        event_target = (
            self.labels.event_id(
                source_name,
                raw_event_id,
            )
        )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),

            "attention_mask": (
                torch.tensor(
                    attention_mask,
                    dtype=torch.float32,
                )
            ),

            "source_target": (
                torch.tensor(
                    source_target,
                    dtype=torch.long,
                )
            ),

            "component_target": (
                torch.tensor(
                    component_target,
                    dtype=torch.long,
                )
            ),

            "event_target": (
                torch.tensor(
                    event_target,
                    dtype=torch.long,
                )
            ),
        }


# ---------------------------------------------------------------
# Leakage-safe event bundle
# ---------------------------------------------------------------

class EventDatasetBundle:
    """
    Train/validation/test EventDataset objects
    sharing train-fitted vocabulary and labels.
    """

    def __init__(
        self,
        *,
        train: EventDataset,
        validation: Optional[
            EventDataset
        ],
        test: Optional[
            EventDataset
        ],
        vocabulary: Vocabulary,
        labels: LabelMaps,
    ) -> None:
        self.train = train

        self.validation = (
            validation
        )

        self.test = test

        self.vocabulary = (
            vocabulary
        )

        self.labels = labels

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "train_count": (
                len(
                    self.train
                )
            ),

            "validation_count": (
                len(
                    self.validation
                )
                if self.validation
                is not None
                else 0
            ),

            "test_count": (
                len(
                    self.test
                )
                if self.test
                is not None
                else 0
            ),

            "vocabulary_size": (
                len(
                    self.vocabulary
                )
            ),

            "source_classes": (
                len(
                    self.labels.
                    source_to_id
                )
            ),

            "component_classes": (
                len(
                    self.labels.
                    component_to_id
                )
            ),

            "event_classes": (
                len(
                    self.labels.
                    event_to_id
                )
            ),

            "vocabulary_fit_split": (
                "train"
            ),

            "labels_fit_split": (
                "train"
            ),
        }


# ---------------------------------------------------------------
# Build from TrainingSplitResult
# ---------------------------------------------------------------

def build_event_datasets(
    split_result: TrainingSplitResult,
    *,
    max_length: int = 128,
    min_frequency: int = 2,
    max_vocabulary_size: int = 50000,
    skip_invalid_examples: bool = False,
) -> EventDatasetBundle:
    """
    Bridge the shared training architecture to
    the sequence-based event model.

    Vocabulary and label mappings are fitted
    exclusively from the TRAIN partition.
    """

    split_result.validate()

    if (
        split_result.task
        != TrainingTask.EVENT.value
    ):
        raise InvalidLabelError(
            "Event datasets require an "
            "event TrainingSplitResult.",
            details={
                "task": (
                    split_result.task
                ),
            },
        )

    train_records = (
        event_records_from_examples(
            split_result.train,
            skip_invalid=(
                skip_invalid_examples
            ),
        )
    )

    validation_records = (
        event_records_from_examples(
            split_result.validation,
            skip_invalid=(
                skip_invalid_examples
            ),
        )
        if split_result.validation
        else []
    )

    test_records = (
        event_records_from_examples(
            split_result.test,
            skip_invalid=(
                skip_invalid_examples
            ),
        )
        if split_result.test
        else []
    )

    # Critical leakage boundary:
    #
    # These two structures learn from TRAIN only.
    vocabulary = Vocabulary(
        train_records,
        min_frequency=(
            min_frequency
        ),
        max_size=(
            max_vocabulary_size
        ),
    )

    labels = LabelMaps(
        train_records
    )

    train_dataset = EventDataset(
        train_records,
        vocabulary,
        labels,
        max_length=max_length,
    )

    validation_dataset = (
        EventDataset(
            validation_records,
            vocabulary,
            labels,
            max_length=max_length,
        )
        if validation_records
        else None
    )

    test_dataset = (
        EventDataset(
            test_records,
            vocabulary,
            labels,
            max_length=max_length,
        )
        if test_records
        else None
    )

    return EventDatasetBundle(
        train=train_dataset,
        validation=(
            validation_dataset
        ),
        test=test_dataset,
        vocabulary=vocabulary,
        labels=labels,
    )


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "PAD_ID",
    "UNK_ID",
    "IGNORE_INDEX",

    "tokenize",

    "is_event_record",
    "load_event_records",

    "event_group_key",
    "split_records",

    "training_example_to_event_record",
    "event_records_from_examples",

    "LabelMaps",
    "Vocabulary",

    "EventDataset",
    "EventDatasetBundle",

    "build_event_datasets",
]
