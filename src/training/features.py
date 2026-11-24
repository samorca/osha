from __future__ import annotations

"""
Deterministic feature construction for OSHA
training examples.

Responsibilities:

    flatten structured feature mappings
    infer or apply a feature schema
    encode numeric and boolean values
    encode categorical values deterministically
    handle missing values
    produce ordered numeric feature vectors

This module does not create framework-specific
tensors. event_dataset.py and anomaly_dataset.py
can convert the resulting vectors into PyTorch,
NumPy, or another model-specific representation.
"""

import hashlib
import json
import math

from collections.abc import (
    Callable,
    Iterable,
    Mapping,
)

from dataclasses import (
    dataclass,
    field,
)

from enum import Enum

from threading import (
    RLock,
)

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from src.core.serialization import (
    to_json_safe,
)

from src.training.errors import (
    FeatureExtractionError,
    FeatureSchemaMismatchError,
    FeatureValidationError,
)

from src.training.schema import (
    TrainingDataset,
    TrainingExample,
    TrainingTask,
)


# ---------------------------------------------------------------
# Feature types
# ---------------------------------------------------------------


class FeatureKind(
    str,
    Enum,
):
    NUMERIC = "numeric"

    BOOLEAN = "boolean"

    CATEGORICAL = "categorical"


# ---------------------------------------------------------------
# Missing-value strategy
# ---------------------------------------------------------------


class MissingValueStrategy(
    str,
    Enum,
):
    ZERO = "zero"

    MEAN = "mean"

    ERROR = "error"


# ---------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """
    Description of one encoded feature.

    Categorical values are represented by an
    integer number generated through stable
    hashing. The result is then normalized to the
    range [0, 1].
    """

    name: str

    kind: str

    mean: float = 0.0

    scale: float = 1.0

    missing_value: float = 0.0

    category_buckets: int = 1024

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        if not str(self.name).strip():
            raise FeatureValidationError("Feature name cannot be empty.")

        if self.kind not in {item.value for item in FeatureKind}:
            raise FeatureValidationError(
                "Unknown feature kind.",
                details={
                    "feature": self.name,
                    "kind": self.kind,
                },
            )

        if not math.isfinite(float(self.mean)):
            raise FeatureValidationError(
                "Feature mean must be finite.",
                details={
                    "feature": self.name,
                },
            )

        if not math.isfinite(float(self.scale)):
            raise FeatureValidationError(
                "Feature scale must be finite.",
                details={
                    "feature": self.name,
                },
            )

        if float(self.scale) < 0:
            raise FeatureValidationError(
                "Feature scale must be >= 0.",
                details={
                    "feature": self.name,
                },
            )

        if not math.isfinite(float(self.missing_value)):
            raise FeatureValidationError(
                "Feature missing value must be finite.",
                details={
                    "feature": self.name,
                },
            )

        if self.category_buckets < 2:
            raise FeatureValidationError(
                "category_buckets must be >= 2.",
                details={
                    "feature": self.name,
                },
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "name": self.name,
            "kind": self.kind,
            "mean": float(self.mean),
            "scale": float(self.scale),
            "missing_value": float(self.missing_value),
            "category_buckets": (self.category_buckets),
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSchema:
    """
    Ordered model-facing feature contract.

    The order of features in this schema defines
    the order of values in every encoded vector.
    """

    task: str

    features: Tuple[
        FeatureSpec,
        ...,
    ]

    version: str = "1"

    schema_id: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        if self.task not in {item.value for item in TrainingTask}:
            raise FeatureValidationError(
                "Unknown feature-schema task.",
                details={
                    "task": self.task,
                },
            )

        if not str(self.version).strip():
            raise FeatureValidationError("Feature schema version cannot be empty.")

        if not self.features:
            raise FeatureValidationError("Feature schema cannot be empty.")

        names = set()

        for feature in self.features:
            feature.validate()

            if feature.name in names:
                raise FeatureValidationError(
                    "Feature schema contains a duplicate feature.",
                    details={
                        "feature": (feature.name),
                    },
                )

            names.add(feature.name)

    @property
    def feature_names(
        self,
    ) -> List[str]:
        return [feature.name for feature in self.features]

    @property
    def dimension(
        self,
    ) -> int:
        return len(self.features)

    @property
    def effective_schema_id(
        self,
    ) -> str:
        if self.schema_id:
            return self.schema_id

        material = json.dumps(
            {
                "task": self.task,
                "version": self.version,
                "features": [feature.to_dict() for feature in self.features],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode("utf-8")

        return "feature-schema-" + hashlib.sha256(material).hexdigest()

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "schema_id": (self.effective_schema_id),
            "version": self.version,
            "task": self.task,
            "dimension": (self.dimension),
            "feature_names": (self.feature_names),
            "features": [feature.to_dict() for feature in self.features],
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Encoded training example
# ---------------------------------------------------------------


@dataclass(frozen=True)
class EncodedTrainingExample:
    example_id: str

    task: str

    values: Tuple[
        float,
        ...,
    ]

    label: Any

    feature_schema_id: str

    group_id: Optional[str] = None

    source_id: Optional[str] = None

    reference_ids: Tuple[
        str,
        ...,
    ] = ()

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def validate(
        self,
    ) -> None:
        if not self.example_id:
            raise FeatureValidationError("Encoded example requires example_id.")

        if not self.feature_schema_id:
            raise FeatureValidationError("Encoded example requires feature_schema_id.")

        if not self.values:
            raise FeatureValidationError("Encoded feature vector cannot be empty.")

        for index, value in enumerate(self.values):
            if not math.isfinite(float(value)):
                raise FeatureValidationError(
                    "Encoded feature value must be finite.",
                    details={
                        "example_id": (self.example_id),
                        "index": index,
                    },
                )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "example_id": (self.example_id),
            "task": self.task,
            "values": [float(value) for value in self.values],
            "label": (to_json_safe(self.label)),
            "feature_schema_id": (self.feature_schema_id),
            "group_id": self.group_id,
            "source_id": self.source_id,
            "reference_ids": list(self.reference_ids),
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Feature build result
# ---------------------------------------------------------------


@dataclass(frozen=True)
class FeatureBuildResult:
    schema: FeatureSchema

    examples: Tuple[
        EncodedTrainingExample,
        ...,
    ]

    rejected_example_ids: Tuple[
        str,
        ...,
    ] = ()

    def to_dict(
        self,
        *,
        include_examples: bool = False,
    ) -> Dict[str, Any]:
        result = {
            "schema": (self.schema.to_dict()),
            "example_count": len(self.examples),
            "rejected_example_count": len(self.rejected_example_ids),
            "rejected_example_ids": list(self.rejected_example_ids),
        }

        if include_examples:
            result["examples"] = [example.to_dict() for example in self.examples]

        return result


# ---------------------------------------------------------------
# Feature builder contract
# ---------------------------------------------------------------

FeatureExtractor = Callable[
    [TrainingExample],
    Dict[
        str,
        Any,
    ],
]


# ---------------------------------------------------------------
# Stable categorical encoding
# ---------------------------------------------------------------


def categorical_hash(
    value: Any,
    *,
    buckets: int = 1024,
) -> float:
    """
    Hash a categorical value deterministically
    and normalize it to [0, 1].

    Python's built-in hash() is intentionally not
    used because it is process-randomized.
    """

    if buckets < 2:
        raise FeatureValidationError("Categorical bucket count must be >= 2.")

    try:
        serialized = json.dumps(
            to_json_safe(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )

    except Exception as exc:
        raise FeatureExtractionError(
            "Categorical feature could not be serialized.",
            details={
                "error_type": (type(exc).__name__),
            },
        ) from exc

    digest = hashlib.sha256(serialized.encode("utf-8")).digest()

    integer_value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    bucket = integer_value % buckets

    return float(bucket) / float(buckets - 1)


# ---------------------------------------------------------------
# Flatten structured features
# ---------------------------------------------------------------


def flatten_features(
    value: Mapping[
        str,
        Any,
    ],
    *,
    prefix: str = "",
    separator: str = ".",
) -> Dict[str, Any]:
    """
    Flatten nested mappings into deterministic
    dotted feature names.

    Lists and tuples remain categorical values
    rather than creating variable vector widths.
    """

    output: Dict[
        str,
        Any,
    ] = {}

    for key in sorted(
        value,
        key=lambda item: str(item),
    ):
        name = str(key).strip()

        if not name:
            continue

        qualified = prefix + separator + name if prefix else name

        item = value[key]

        if isinstance(
            item,
            Mapping,
        ):
            output.update(
                flatten_features(
                    item,
                    prefix=qualified,
                    separator=separator,
                )
            )

        else:
            output[qualified] = item

    return output


# ---------------------------------------------------------------
# Feature-kind inference
# ---------------------------------------------------------------


def infer_feature_kind(
    values: Sequence[Any],
) -> str:
    """
    Infer one stable feature kind from observed
    non-missing values.
    """

    present = [value for value in values if value is not None]

    if not present:
        return FeatureKind.NUMERIC.value

    if all(
        isinstance(
            value,
            bool,
        )
        for value in present
    ):
        return FeatureKind.BOOLEAN.value

    if all(
        isinstance(
            value,
            (
                int,
                float,
            ),
        )
        and not isinstance(
            value,
            bool,
        )
        for value in present
    ):
        return FeatureKind.NUMERIC.value

    return FeatureKind.CATEGORICAL.value


# ---------------------------------------------------------------
# Numeric statistics
# ---------------------------------------------------------------


def numeric_statistics(
    values: Sequence[Any],
) -> Tuple[
    float,
    float,
]:
    """
    Return mean and population standard
    deviation.

    A zero standard deviation is represented by
    scale=1.0 to keep encoding finite.
    """

    numeric_values = [
        float(value)
        for value in values
        if (
            value is not None
            and not isinstance(
                value,
                bool,
            )
            and isinstance(
                value,
                (
                    int,
                    float,
                ),
            )
            and math.isfinite(float(value))
        )
    ]

    if not numeric_values:
        return (
            0.0,
            1.0,
        )

    mean = sum(numeric_values) / len(numeric_values)

    variance = sum((value - mean) ** 2 for value in numeric_values) / len(
        numeric_values
    )

    scale = math.sqrt(variance)

    if not math.isfinite(scale) or scale == 0.0:
        scale = 1.0

    return (
        mean,
        scale,
    )


# ---------------------------------------------------------------
# Default extractor
# ---------------------------------------------------------------


def default_feature_extractor(
    example: TrainingExample,
) -> Dict[str, Any]:
    example.finalize()

    flattened = flatten_features(example.features)

    # Add bounded context fields that are often
    # useful across event/anomaly training.
    context = {
        "context.provider": (example.provider),
        "context.environment_id": (example.environment_id),
        "context.resource_type": (example.resource_type),
    }

    for key, value in context.items():
        if value is not None:
            flattened[key] = value

    return flattened


# ---------------------------------------------------------------
# Extractor registry
# ---------------------------------------------------------------


class FeatureExtractorRegistry:
    """
    Thread-safe registry keyed by training task.

    A task-specific extractor can replace the
    generic extractor without changing dataset
    assembly contracts.
    """

    def __init__(
        self,
    ) -> None:
        self._extractors: Dict[
            str,
            FeatureExtractor,
        ] = {}

        self._lock = RLock()

    def register(
        self,
        *,
        task: str,
        extractor: FeatureExtractor,
        replace: bool = False,
    ) -> None:
        task_name = str(task).strip().lower()

        if task_name not in {item.value for item in TrainingTask}:
            raise FeatureValidationError(
                "Unknown extractor task.",
                details={
                    "task": task_name,
                },
            )

        if not callable(extractor):
            raise FeatureValidationError("Feature extractor must be callable.")

        with self._lock:
            if task_name in self._extractors and not replace:
                raise FeatureValidationError(
                    "Feature extractor is already registered.",
                    details={
                        "task": task_name,
                    },
                )

            self._extractors[task_name] = extractor

    def resolve(
        self,
        task: str,
    ) -> Optional[FeatureExtractor]:
        task_name = str(task).strip().lower()

        with self._lock:
            return self._extractors.get(task_name)

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            tasks = sorted(self._extractors)

        return {
            "extractor_count": len(tasks),
            "tasks": tasks,
        }


# ---------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------


class FeatureBuilder:
    """
    Fit and apply deterministic feature schemas.

    Workflow:

        schema = builder.fit(dataset)
        result = builder.transform(
            dataset,
            schema=schema,
        )

    For leakage-safe model evaluation, fit the
    schema on training examples only, then apply
    it unchanged to validation/test examples.
    """

    def __init__(
        self,
        *,
        registry: Optional[FeatureExtractorRegistry] = None,
        missing_strategy: str = (MissingValueStrategy.ZERO.value),
        category_buckets: int = 1024,
        reject_invalid: bool = True,
    ) -> None:
        if missing_strategy not in {item.value for item in MissingValueStrategy}:
            raise FeatureValidationError(
                "Unknown missing-value strategy.",
                details={
                    "strategy": (missing_strategy),
                },
            )

        if category_buckets < 2:
            raise FeatureValidationError("category_buckets must be >= 2.")

        self.registry = registry or FeatureExtractorRegistry()

        self.missing_strategy = missing_strategy

        self.category_buckets = int(category_buckets)

        self.reject_invalid = bool(reject_invalid)

    # ---------------------------------------------------------
    # Extract
    # ---------------------------------------------------------

    def extract(
        self,
        example: TrainingExample,
    ) -> Dict[str, Any]:
        example.finalize()

        extractor = self.registry.resolve(example.task) or default_feature_extractor

        try:
            values = extractor(example)

        except (
            FeatureExtractionError,
            FeatureValidationError,
        ):
            raise

        except Exception as exc:
            raise FeatureExtractionError(
                "Feature extractor failed.",
                details={
                    "example_id": (example.example_id),
                    "task": example.task,
                    "error_type": (type(exc).__name__),
                    "error_module": (type(exc).__module__),
                },
            ) from exc

        if not isinstance(
            values,
            dict,
        ):
            raise FeatureExtractionError(
                "Feature extractor did not return a dictionary.",
                details={
                    "example_id": (example.example_id),
                },
            )

        return values

    # ---------------------------------------------------------
    # Fit schema
    # ---------------------------------------------------------

    def fit(
        self,
        examples: Iterable[TrainingExample],
        *,
        task: Optional[str] = None,
        version: str = "1",
    ) -> FeatureSchema:
        values = list(examples)

        if not values:
            raise FeatureValidationError("Cannot fit a feature schema on no examples.")

        task_name = str(task).strip().lower() if task else values[0].task

        extracted: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        names = set()

        for example in values:
            example.finalize()

            if example.task != task_name:
                raise FeatureSchemaMismatchError(
                    "Training examples contain multiple tasks."
                )

            features = self.extract(example)

            extracted.append(features)

            names.update(features)

        specs: List[FeatureSpec] = []

        for name in sorted(names):
            observed = [feature_values.get(name) for feature_values in extracted]

            kind = infer_feature_kind(observed)

            mean = 0.0
            scale = 1.0
            missing_value = 0.0

            if kind == FeatureKind.NUMERIC.value:
                (
                    mean,
                    scale,
                ) = numeric_statistics(observed)

                if self.missing_strategy == MissingValueStrategy.MEAN.value:
                    missing_value = mean

            specs.append(
                FeatureSpec(
                    name=name,
                    kind=kind,
                    mean=mean,
                    scale=scale,
                    missing_value=(missing_value),
                    category_buckets=(self.category_buckets),
                    attributes={
                        "observed_count": sum(
                            1 for value in observed if value is not None
                        ),
                    },
                )
            )

        schema = FeatureSchema(
            task=task_name,
            features=tuple(specs),
            version=version,
            attributes={
                "missing_strategy": (self.missing_strategy),
                "category_buckets": (self.category_buckets),
                "fit_example_count": len(values),
            },
        )

        schema.validate()

        return schema

    # ---------------------------------------------------------
    # Encode one scalar
    # ---------------------------------------------------------

    def _encode_value(
        self,
        value: Any,
        spec: FeatureSpec,
    ) -> float:
        if value is None:
            if self.missing_strategy == MissingValueStrategy.ERROR.value:
                raise FeatureValidationError(
                    "Required feature is missing.",
                    details={
                        "feature": (spec.name),
                    },
                )

            value = spec.missing_value

        if spec.kind == FeatureKind.BOOLEAN.value:
            if isinstance(
                value,
                bool,
            ):
                return 1.0 if value else 0.0

            if value in {
                0,
                1,
            }:
                return float(value)

            raise FeatureValidationError(
                "Boolean feature contains an invalid value.",
                details={
                    "feature": spec.name,
                },
            )

        if spec.kind == FeatureKind.NUMERIC.value:
            if isinstance(
                value,
                bool,
            ) or not isinstance(
                value,
                (
                    int,
                    float,
                ),
            ):
                raise FeatureValidationError(
                    "Numeric feature contains a non-numeric value.",
                    details={
                        "feature": (spec.name),
                    },
                )

            numeric = float(value)

            if not math.isfinite(numeric):
                raise FeatureValidationError(
                    "Numeric feature must be finite.",
                    details={
                        "feature": (spec.name),
                    },
                )

            return (numeric - spec.mean) / (spec.scale if spec.scale != 0.0 else 1.0)

        return categorical_hash(
            value,
            buckets=(spec.category_buckets),
        )

    # ---------------------------------------------------------
    # Transform one
    # ---------------------------------------------------------

    def transform_one(
        self,
        example: TrainingExample,
        *,
        schema: FeatureSchema,
    ) -> EncodedTrainingExample:
        example.finalize()
        schema.validate()

        if example.task != schema.task:
            raise FeatureSchemaMismatchError(
                "Training example task does not match feature schema task.",
                details={
                    "example_task": (example.task),
                    "schema_task": (schema.task),
                },
            )

        extracted = self.extract(example)

        encoded = tuple(
            self._encode_value(
                extracted.get(spec.name),
                spec,
            )
            for spec in schema.features
        )

        result = EncodedTrainingExample(
            example_id=str(example.example_id),
            task=example.task,
            values=encoded,
            label=example.label,
            feature_schema_id=(schema.effective_schema_id),
            group_id=(example.group_id),
            source_id=(example.source_id),
            reference_ids=tuple(example.reference_ids),
            attributes={
                "provider": (example.provider),
                "environment_id": (example.environment_id),
                "resource_type": (example.resource_type),
                "resource_id": (example.resource_id),
                "incident_id": (example.incident_id),
                "timestamp": (example.timestamp),
                "feature_stage": ("encoded"),
            },
        )

        result.validate()

        return result

    # ---------------------------------------------------------
    # Transform many
    # ---------------------------------------------------------

    def transform(
        self,
        examples: Iterable[TrainingExample],
        *,
        schema: FeatureSchema,
    ) -> FeatureBuildResult:
        encoded: List[EncodedTrainingExample] = []

        rejected: List[str] = []

        for example in examples:
            try:
                encoded.append(
                    self.transform_one(
                        example,
                        schema=schema,
                    )
                )

            except (
                FeatureExtractionError,
                FeatureValidationError,
                FeatureSchemaMismatchError,
            ):
                if not self.reject_invalid:
                    raise

                rejected.append(str(example.example_id))

        if not encoded:
            raise FeatureValidationError(
                "Feature transformation produced no usable examples."
            )

        return FeatureBuildResult(
            schema=schema,
            examples=tuple(encoded),
            rejected_example_ids=tuple(rejected),
        )

    # ---------------------------------------------------------
    # Convenience fit + transform
    # ---------------------------------------------------------

    def fit_transform(
        self,
        dataset: TrainingDataset,
        *,
        version: str = "1",
    ) -> FeatureBuildResult:
        dataset.finalize()

        schema = self.fit(
            dataset.examples,
            task=dataset.task,
            version=version,
        )

        return self.transform(
            dataset.examples,
            schema=schema,
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "missing_strategy": (self.missing_strategy),
            "category_buckets": (self.category_buckets),
            "reject_invalid": (self.reject_invalid),
            "registry": (self.registry.summary()),
            "framework_specific_tensors": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "FeatureKind",
    "MissingValueStrategy",
    "FeatureSpec",
    "FeatureSchema",
    "EncodedTrainingExample",
    "FeatureBuildResult",
    "FeatureExtractor",
    "categorical_hash",
    "flatten_features",
    "infer_feature_kind",
    "numeric_statistics",
    "default_feature_extractor",
    "FeatureExtractorRegistry",
    "FeatureBuilder",
]
