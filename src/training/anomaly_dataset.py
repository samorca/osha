from __future__ import annotations

"""
Dataset utilities for OSHA anomaly-model training.

The anomaly branch consumes fixed-width numeric
feature vectors.

New pipeline:

    TrainingDataset
        -> TrainingSplitter
        -> FeatureBuilder.fit(train only)
        -> FeatureSchema
        -> encoded train/validation/test
        -> AnomalyDataset
        -> anomaly model

Legacy JSONL anomaly records remain supported so
existing training workflows can be migrated
incrementally.
"""

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
from torch.utils.data import Dataset

from src.training.errors import (
    EmptyTrainingDatasetError,
    FeatureValidationError,
    InvalidLabelError,
    TrainingSourceFormatError,
)
from src.training.features import (
    EncodedTrainingExample,
    FeatureBuilder,
    FeatureBuildResult,
    FeatureSchema,
)
from src.training.schema import (
    TrainingTask,
)
from src.training.splits import (
    TrainingSplitResult,
)

# ---------------------------------------------------------------
# Anomaly labels
# ---------------------------------------------------------------

NORMAL_LABEL = 0

ANOMALY_LABEL = 1

IGNORE_INDEX = -100

NON_FEATURE_FIELDS = {
    "label",
    "target",
    "anomaly",
    "anomaly_label",
    "is_anomaly",
    "row_id",
    "record_id",
    "example_id",
}


# ---------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------

NORMAL_LABEL_VALUES = {
    "0",
    "normal",
    "healthy",
    "ok",
    "false",
    "negative",
    "non_anomaly",
    "non-anomaly",
}

ANOMALY_LABEL_VALUES = {
    "1",
    "anomaly",
    "anomalous",
    "abnormal",
    "unhealthy",
    "true",
    "positive",
}


def normalize_anomaly_label(
    value: Any,
    *,
    allow_missing: bool = False,
) -> int:
    """
    Normalize common anomaly-label encodings to:

        0 = normal
        1 = anomaly

    Missing labels can optionally map to
    IGNORE_INDEX.
    """

    if value is None:
        if allow_missing:
            return IGNORE_INDEX

        raise InvalidLabelError("Anomaly example requires a label.")

    if isinstance(
        value,
        bool,
    ):
        return ANOMALY_LABEL if value else NORMAL_LABEL

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        numeric = float(value)

        if not math.isfinite(numeric):
            raise InvalidLabelError("Anomaly label must be finite.")

        if numeric == 0.0:
            return NORMAL_LABEL

        if numeric == 1.0:
            return ANOMALY_LABEL

    text = str(value).strip().lower()

    if text in NORMAL_LABEL_VALUES:
        return NORMAL_LABEL

    if text in ANOMALY_LABEL_VALUES:
        return ANOMALY_LABEL

    raise InvalidLabelError(
        "Unsupported anomaly label.",
        details={
            "label_type": (type(value).__name__),
        },
    )


# ---------------------------------------------------------------
# Legacy feature extraction
# ---------------------------------------------------------------


def _numeric_vector(
    value: Any,
) -> Optional[List[float]]:
    if not isinstance(
        value,
        Sequence,
    ) or isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        return None

    output: List[float] = []

    for item in value:
        if isinstance(
            item,
            bool,
        ) or not isinstance(
            item,
            (
                int,
                float,
            ),
        ):
            return None

        number = float(item)

        if not math.isfinite(number):
            return None

        output.append(number)

    return output or None


def extract_legacy_features(
    record: Mapping[
        str,
        Any,
    ],
) -> Optional[List[float]]:
    """
    Extract an already-numeric vector from common
    legacy anomaly-record fields.

    We deliberately do not invent numeric values
    from arbitrary raw dictionaries here.
    """

    for field in (
        "features",
        "feature_vector",
        "values",
        "vector",
        "x",
    ):
        vector = _numeric_vector(record.get(field))

        if vector is not None:
            return vector

    nested = record.get("values")
    if isinstance(nested, Mapping):
        output: List[float] = []
        ordered_values = [
            nested[key]
            for key in sorted(nested, key=str)
            if str(key).strip().lower() not in NON_FEATURE_FIELDS
        ]

        for value in ordered_values:
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                number = float(value)
                if math.isfinite(number):
                    output.append(number)
                continue
            if value is not None:
                digest = hashlib.sha256(
                    str(value).encode("utf-8")
                ).digest()
                output.append(
                    int.from_bytes(
                        digest[:8],
                        "little",
                        signed=False,
                    )
                    % 1_000_000
                    / 1_000_000
                )

        for field in (
            "score",
            "severity",
        ):
            value = record.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                number = float(value)
                if math.isfinite(number):
                    output.append(number)

        if len(output) < 16:
            output.extend([0.0] * (16 - len(output)))

        return output[:16]

    return None


def extract_legacy_label(
    record: Mapping[
        str,
        Any,
    ],
) -> Any:
    for field in (
        "label",
        "target",
        "anomaly_label",
        "is_anomaly",
        "anomaly",
    ):
        if field in record and record[field] is not None:
            return record[field]

    nested = record.get("values")
    if isinstance(nested, Mapping):
        for field in (
            "is_anomaly",
            "anomaly",
            "anomaly_label",
            "label",
            "target",
        ):
            if field in nested and nested[field] is not None:
                return nested[field]

    return None


# ---------------------------------------------------------------
# Legacy anomaly-record validation
# ---------------------------------------------------------------


def is_anomaly_record(
    record: Mapping[
        str,
        Any,
    ],
    *,
    require_label: bool = True,
) -> bool:
    record_type = record.get("record_type")

    if record_type is not None and str(record_type).strip().lower() not in {
        "anomaly",
        "anomaly_record",
        "anomaly_example",
        "metric_window",
        "anomaly_groundtruth",
    }:
        return False

    vector = extract_legacy_features(record)

    if vector is None:
        return False

    if require_label:
        try:
            normalize_anomaly_label(extract_legacy_label(record))

        except InvalidLabelError:
            return False

    return True


# ---------------------------------------------------------------
# Legacy JSONL loading
# ---------------------------------------------------------------


def load_anomaly_records(
    path: str | Path,
    *,
    max_records: Optional[int] = None,
    require_label: bool = True,
) -> List[
    Dict[
        str,
        Any,
    ]
]:
    source = Path(path).expanduser()

    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be >= 1")

    if not source.exists():
        raise TrainingSourceFormatError(
            "Anomaly training source was not found.",
            details={
                "path": str(source),
            },
        )

    if not source.is_file():
        raise TrainingSourceFormatError(
            "Anomaly training source is not a regular file."
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
                    value = json.loads(text)

                except json.JSONDecodeError as exc:
                    raise TrainingSourceFormatError(
                        "Anomaly JSONL contains invalid JSON.",
                        details={
                            "path": str(source),
                            "line_number": (line_number),
                            "column": (exc.colno),
                        },
                    ) from exc

                if not isinstance(
                    value,
                    Mapping,
                ):
                    continue

                record = dict(value)

                if not is_anomaly_record(
                    record,
                    require_label=(require_label),
                ):
                    continue

                records.append(record)

                if max_records is not None and len(records) >= max_records:
                    break

    except TrainingSourceFormatError:
        raise

    except (
        OSError,
        UnicodeError,
    ) as exc:
        raise TrainingSourceFormatError(
            "Anomaly training source could not be read.",
            details={
                "path": str(source),
                "error_type": (type(exc).__name__),
            },
        ) from exc

    if not records:
        raise EmptyTrainingDatasetError(
            "No usable anomaly records were found.",
            details={
                "path": str(source),
            },
        )

    return records


# ---------------------------------------------------------------
# Feature dimension validation
# ---------------------------------------------------------------


def validate_feature_dimensions(
    vectors: Iterable[Sequence[float]],
) -> int:
    dimension: Optional[int] = None

    count = 0

    for vector in vectors:
        count += 1

        current = len(vector)

        if current < 1:
            raise FeatureValidationError("Anomaly feature vector cannot be empty.")

        if dimension is None:
            dimension = current

        elif current != dimension:
            raise FeatureValidationError(
                "Anomaly feature vectors have inconsistent dimensions.",
                details={
                    "expected": (dimension),
                    "actual": (current),
                },
            )

        for value in vector:
            if not math.isfinite(float(value)):
                raise FeatureValidationError(
                    "Anomaly feature vector contains a non-finite value."
                )

    if count == 0 or dimension is None:
        raise EmptyTrainingDatasetError("No anomaly feature vectors were supplied.")

    return dimension


# ---------------------------------------------------------------
# Tensor-ready anomaly sample
# ---------------------------------------------------------------


class AnomalySample:
    """
    Framework-neutral in-memory anomaly sample.
    """

    def __init__(
        self,
        *,
        values: Sequence[float],
        label: int,
        example_id: Optional[str] = None,
        group_id: Optional[str] = None,
        source_id: Optional[str] = None,
        reference_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.values = tuple(float(value) for value in values)

        self.label = int(label)

        self.example_id = example_id

        self.group_id = group_id

        self.source_id = source_id

        self.reference_ids = tuple(str(value) for value in (reference_ids or ()))

        self.validate()

    def validate(
        self,
    ) -> None:
        if not self.values:
            raise FeatureValidationError(
                "Anomaly sample cannot have an empty feature vector."
            )

        for value in self.values:
            if not math.isfinite(value):
                raise FeatureValidationError(
                    "Anomaly sample contains a non-finite feature."
                )

        if self.label not in {
            NORMAL_LABEL,
            ANOMALY_LABEL,
            IGNORE_INDEX,
        }:
            raise InvalidLabelError(
                "Anomaly sample contains an invalid normalized label."
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "values": list(self.values),
            "label": (self.label),
            "example_id": (self.example_id),
            "group_id": (self.group_id),
            "source_id": (self.source_id),
            "reference_ids": list(self.reference_ids),
        }


# ---------------------------------------------------------------
# PyTorch anomaly dataset
# ---------------------------------------------------------------


class AnomalyDataset(Dataset):
    """
    Fixed-width numeric anomaly dataset.

    Batch contract:

        features:
            float32 [feature_dimension]

        target:
            long scalar

    Additional metadata is retained internally
    but is not included in the default batch so
    DataLoader collation remains simple.
    """

    def __init__(
        self,
        samples: Iterable[AnomalySample],
    ) -> None:
        self.samples = list(samples)

        if not self.samples:
            raise EmptyTrainingDatasetError("AnomalyDataset contains no samples.")

        self.feature_dimension = validate_feature_dimensions(
            sample.values for sample in self.samples
        )

        for sample in self.samples:
            sample.validate()

    def __len__(
        self,
    ) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]

        return {
            "features": torch.tensor(
                sample.values,
                dtype=torch.float32,
            ),
            "target": torch.tensor(
                sample.label,
                dtype=torch.long,
            ),
        }

    def sample_metadata(
        self,
        index: int,
    ) -> Dict[str, Any]:
        sample = self.samples[index]

        return {
            "example_id": (sample.example_id),
            "group_id": (sample.group_id),
            "source_id": (sample.source_id),
            "reference_ids": list(sample.reference_ids),
        }

    def summary(
        self,
    ) -> Dict[str, Any]:
        normal_count = sum(
            1 for sample in self.samples if (sample.label == NORMAL_LABEL)
        )

        anomaly_count = sum(
            1 for sample in self.samples if (sample.label == ANOMALY_LABEL)
        )

        ignored_count = sum(
            1 for sample in self.samples if (sample.label == IGNORE_INDEX)
        )

        return {
            "sample_count": len(self.samples),
            "feature_dimension": (self.feature_dimension),
            "normal_count": (normal_count),
            "anomaly_count": (anomaly_count),
            "ignored_count": (ignored_count),
        }


# ---------------------------------------------------------------
# Build from encoded examples
# ---------------------------------------------------------------


def samples_from_encoded(
    result: FeatureBuildResult,
    *,
    allow_missing_labels: bool = False,
) -> List[AnomalySample]:
    result.schema.validate()

    if result.schema.task != TrainingTask.ANOMALY.value:
        raise FeatureValidationError(
            "Anomaly samples require an anomaly feature schema.",
            details={
                "task": (result.schema.task),
            },
        )

    samples: List[AnomalySample] = []

    for example in result.examples:
        example.validate()

        label = normalize_anomaly_label(
            example.label,
            allow_missing=(allow_missing_labels),
        )

        samples.append(
            AnomalySample(
                values=(example.values),
                label=label,
                example_id=(example.example_id),
                group_id=(example.group_id),
                source_id=(example.source_id),
                reference_ids=(example.reference_ids),
            )
        )

    if not samples:
        raise EmptyTrainingDatasetError("Feature encoding produced no anomaly samples.")

    return samples


# ---------------------------------------------------------------
# Build legacy dataset
# ---------------------------------------------------------------


def anomaly_dataset_from_records(
    records: Iterable[
        Mapping[
            str,
            Any,
        ]
    ],
    *,
    allow_missing_labels: bool = False,
) -> AnomalyDataset:
    samples: List[AnomalySample] = []

    for record in records:
        vector = extract_legacy_features(record)

        if vector is None:
            continue

        try:
            label = normalize_anomaly_label(
                extract_legacy_label(record),
                allow_missing=(allow_missing_labels),
            )

        except InvalidLabelError:
            if allow_missing_labels:
                label = IGNORE_INDEX

            else:
                raise

        example_id = (
            str(record["record_id"]) if record.get("record_id") is not None else None
        )

        group_id = None

        for field in (
            "group_id",
            "incident_id",
            "sequence_id",
            "trace_id",
        ):
            if record.get(field) is not None:
                group_id = str(record[field])

                break

        samples.append(
            AnomalySample(
                values=vector,
                label=label,
                example_id=(example_id),
                group_id=(group_id),
            )
        )

    if not samples:
        raise EmptyTrainingDatasetError("No usable anomaly samples were produced.")

    return AnomalyDataset(samples)


# ---------------------------------------------------------------
# Complete anomaly dataset bundle
# ---------------------------------------------------------------


class AnomalyDatasetBundle:
    """
    Leakage-safe anomaly train/validation/test
    datasets sharing one feature schema.

    The schema is fitted on TRAIN examples only.
    """

    def __init__(
        self,
        *,
        train: AnomalyDataset,
        validation: Optional[AnomalyDataset],
        test: Optional[AnomalyDataset],
        schema: FeatureSchema,
        train_features: (FeatureBuildResult),
        validation_features: Optional[FeatureBuildResult],
        test_features: Optional[FeatureBuildResult],
    ) -> None:
        self.train = train

        self.validation = validation

        self.test = test

        self.schema = schema

        self.train_features = train_features

        self.validation_features = validation_features

        self.test_features = test_features

    @property
    def feature_dimension(
        self,
    ) -> int:
        return self.schema.dimension

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "train": (self.train.summary()),
            "validation": (
                self.validation.summary() if self.validation is not None else None
            ),
            "test": (self.test.summary() if self.test is not None else None),
            "feature_schema": (self.schema.to_dict()),
            "feature_dimension": (self.feature_dimension),
            "schema_fit_split": ("train"),
        }


# ---------------------------------------------------------------
# Build new-pipeline anomaly datasets
# ---------------------------------------------------------------


def build_anomaly_datasets(
    split_result: TrainingSplitResult,
    *,
    feature_builder: Optional[FeatureBuilder] = None,
    schema_version: str = "1",
) -> AnomalyDatasetBundle:
    """
    Convert a leakage-safe TrainingSplitResult
    into numeric anomaly datasets.

    Critical boundary:

        FeatureBuilder.fit(train only)

    The same fitted schema is then applied to
    validation and test.
    """

    split_result.validate()

    if split_result.task != TrainingTask.ANOMALY.value:
        raise FeatureValidationError(
            "Anomaly dataset builder requires an anomaly split.",
            details={
                "task": (split_result.task),
            },
        )

    builder = feature_builder or FeatureBuilder()

    schema = builder.fit(
        split_result.train,
        task=(TrainingTask.ANOMALY.value),
        version=(schema_version),
    )

    train_features = builder.transform(
        split_result.train,
        schema=schema,
    )

    validation_features = (
        builder.transform(
            split_result.validation,
            schema=schema,
        )
        if split_result.validation
        else None
    )

    test_features = (
        builder.transform(
            split_result.test,
            schema=schema,
        )
        if split_result.test
        else None
    )

    train_dataset = AnomalyDataset(samples_from_encoded(train_features))

    validation_dataset = (
        AnomalyDataset(samples_from_encoded(validation_features))
        if validation_features is not None
        else None
    )

    test_dataset = (
        AnomalyDataset(samples_from_encoded(test_features))
        if test_features is not None
        else None
    )

    # Defensive contract check.
    for dataset in (
        train_dataset,
        validation_dataset,
        test_dataset,
    ):
        if dataset is None:
            continue

        if dataset.feature_dimension != schema.dimension:
            raise FeatureValidationError(
                "Anomaly dataset feature dimension does not match the fitted schema.",
                details={
                    "dataset_dimension": (dataset.feature_dimension),
                    "schema_dimension": (schema.dimension),
                },
            )

    return AnomalyDatasetBundle(
        train=train_dataset,
        validation=(validation_dataset),
        test=test_dataset,
        schema=schema,
        train_features=(train_features),
        validation_features=(validation_features),
        test_features=(test_features),
    )


# ---------------------------------------------------------------
# Legacy metric-window compatibility
# ---------------------------------------------------------------


class RobustScaler:
    def __init__(
        self,
        *,
        center: Optional[
            Dict[str, float]
        ] = None,
        scale: Optional[
            Dict[str, float]
        ] = None,
    ) -> None:
        self.center = dict(center or {})
        self.scale = dict(scale or {})

    @classmethod
    def from_dict(
        cls,
        value: Mapping[
            str,
            Any,
        ],
    ) -> "RobustScaler":
        if not isinstance(value, Mapping):
            raise TypeError("RobustScaler data must be a mapping.")
        return cls(
            center={
                str(key): float(item)
                for key, item
                in value.get("center", {}).items()
            },
            scale={
                str(key): float(item)
                for key, item
                in value.get("scale", {}).items()
            },
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "center": dict(self.center),
            "scale": dict(self.scale),
        }

    def transform(
        self,
        row: Mapping[
            str,
            Any,
        ],
        feature_names: Iterable[str],
    ) -> List[float]:
        output: List[float] = []
        for name in feature_names:
            value = float(row.get(name, 0.0))
            center = self.center.get(name, 0.0)
            scale = self.scale.get(name, 1.0)
            if not scale:
                scale = 1.0
            output.append(
                (value - center) / scale
            )
        return output


def load_metric_rows(
    path: str | Path,
    split: str,
    feature_names: Iterable[str],
) -> List[Dict[str, Any]]:
    source = Path(path).expanduser()
    if not source.exists():
        raise TrainingSourceFormatError(
            "Metric window source was not found.",
            details={
                "path": str(source),
            },
        )

    names = list(feature_names)
    rows: List[Dict[str, Any]] = []

    with source.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise TrainingSourceFormatError(
                    "Metric window JSONL contains invalid JSON.",
                    details={
                        "path": str(source),
                        "line_number": line_number,
                    },
                ) from exc

            if not isinstance(item, Mapping):
                continue

            values = item.get("values")
            row = dict(values) if isinstance(values, Mapping) else dict(item)
            row["timestamp"] = str(
                item.get(
                    "timestamp",
                    item.get("time", ""),
                )
            )
            row["event_id"] = str(
                item.get(
                    "event_id",
                    item.get("event", ""),
                )
            )
            row["row_id"] = int(
                item.get(
                    "row_id",
                    line_number - 1,
                )
            )
            row["label"] = normalize_anomaly_label(
                extract_legacy_label(item),
                allow_missing=True,
            )

            if names:
                for name in names:
                    if name not in row:
                        row[name] = 0.0

            rows.append(row)

    return rows


class MetricWindowDataset(Dataset):
    def __init__(
        self,
        *,
        rows: Iterable[
            Mapping[str, Any]
        ],
        scaler: RobustScaler,
        window_size: int,
        stride: int,
        expected_interval_seconds: Optional[
            float
        ] = None,
    ) -> None:
        del expected_interval_seconds
        self.rows = list(rows)
        self.scaler = scaler
        self.window_size = int(window_size)
        self.stride = int(stride)
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")

        feature_name_set = set()
        for row in self.rows:
            feature_name_set.update(
                key
                for key in row
                if isinstance(key, str)
                and key not in {
                    "timestamp",
                    "event_id",
                    "row_id",
                    "label",
                }
            )
        feature_names = sorted(feature_name_set)
        self.feature_names = feature_names
        self.windows: List[
            Dict[str, Any]
        ] = []

        for start in range(
            0,
            max(
                0,
                len(self.rows)
                - self.window_size
                + 1,
            ),
            self.stride,
        ):
            window_rows = (
                self.rows[
                    start:
                    start
                    + self.window_size
                ]
            )
            features = [
                self.scaler.transform(
                    row,
                    self.feature_names,
                )
                for row in window_rows
            ]
            labels = [
                int(row.get("label", IGNORE_INDEX))
                for row in window_rows
            ]
            valid_labels = [
                label
                for label in labels
                if label in {
                    NORMAL_LABEL,
                    ANOMALY_LABEL,
                }
            ]
            self.windows.append(
                {
                    "window": features,
                    "label": (
                        valid_labels[
                            0
                        ]
                        if valid_labels
                        else IGNORE_INDEX
                    ),
                    "timestamp": (
                        window_rows[-1].get(
                            "timestamp",
                            "",
                        )
                    ),
                    "event_id": (
                        window_rows[-1].get(
                            "event_id",
                            "",
                        )
                    ),
                    "row_id": int(
                        window_rows[-1].get(
                            "row_id",
                            start,
                        )
                    ),
                }
            )

    def __len__(
        self,
    ) -> int:
        return len(self.windows)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        item = self.windows[index]
        return {
            "window": torch.tensor(
                item["window"],
                dtype=torch.float32,
            ),
            "label": torch.tensor(
                item["label"],
                dtype=torch.long,
            ),
            "timestamp": item["timestamp"],
            "event_id": item["event_id"],
            "row_id": torch.tensor(
                item["row_id"],
                dtype=torch.long,
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "NORMAL_LABEL",
    "ANOMALY_LABEL",
    "IGNORE_INDEX",
    "NORMAL_LABEL_VALUES",
    "ANOMALY_LABEL_VALUES",
    "normalize_anomaly_label",
    "extract_legacy_features",
    "extract_legacy_label",
    "is_anomaly_record",
    "load_anomaly_records",
    "validate_feature_dimensions",
    "AnomalySample",
    "AnomalyDataset",
    "samples_from_encoded",
    "anomaly_dataset_from_records",
    "AnomalyDatasetBundle",
    "build_anomaly_datasets",
    "RobustScaler",
    "load_metric_rows",
    "MetricWindowDataset",
]
