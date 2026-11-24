from __future__ import annotations

"""
Shared data-engineering contracts for OSHA.

These models describe datasets, source files,
validation, splitting, and preparation results.

This layer does not fetch data, train models, or
perform provider-specific telemetry collection.
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


# ---------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------

DATA_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------
# Data source kind
# ---------------------------------------------------------------


class DataSourceKind(
    str,
    Enum,
):
    LOCAL = "local"

    HTTP = "http"

    OBJECT_STORAGE = "object_storage"

    GENERATED = "generated"

    CUSTOM = "custom"


# ---------------------------------------------------------------
# Dataset formats
# ---------------------------------------------------------------


class DatasetFormat(
    str,
    Enum,
):
    TEXT = "text"

    JSON = "json"

    JSONL = "jsonl"

    CSV = "csv"

    PARQUET = "parquet"

    LOG = "log"

    CUSTOM = "custom"


# ---------------------------------------------------------------
# Dataset purpose
# ---------------------------------------------------------------


class DatasetPurpose(
    str,
    Enum,
):
    EVENT_TRAINING = "event_training"

    ANOMALY_TRAINING = "anomaly_training"

    EVALUATION = "evaluation"

    INFERENCE_REFERENCE = "inference_reference"

    GENERAL = "general"


# ---------------------------------------------------------------
# Data lifecycle state
# ---------------------------------------------------------------


class DataState(
    str,
    Enum,
):
    DECLARED = "declared"

    FETCHING = "fetching"

    FETCHED = "fetched"

    VALIDATING = "validating"

    VALIDATED = "validated"

    PREPARING = "preparing"

    PREPARED = "prepared"

    FAILED = "failed"


# ---------------------------------------------------------------
# Validation severity
# ---------------------------------------------------------------


class ValidationSeverity(
    str,
    Enum,
):
    INFO = "info"

    WARNING = "warning"

    ERROR = "error"


# ---------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------


class SplitName(
    str,
    Enum,
):
    TRAIN = "train"

    VALIDATION = "validation"

    TEST = "test"


# ---------------------------------------------------------------
# Data source
# ---------------------------------------------------------------


@dataclass(frozen=True)
class DataSource:
    """
    Description of where a dataset originates.

    location may identify a local path, URL, or
    provider-neutral external reference.

    Credentials must never be stored here.
    """

    kind: str

    location: str

    source_id: Optional[str] = None

    provider: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "DataSource":
        if self.kind not in {item.value for item in DataSourceKind}:
            raise ValueError(f"Unknown data source kind: {self.kind}")

        required_string(
            self.location,
            name="data source location",
            maximum_length=4096,
        )

        if self.source_id is not None:
            required_string(
                self.source_id,
                name="source_id",
                maximum_length=512,
            )

        if self.provider is not None:
            required_string(
                self.provider,
                name="source provider",
                maximum_length=128,
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
            "provider": self.provider,
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Dataset file
# ---------------------------------------------------------------


@dataclass
class DatasetFile:
    """
    One physical/logical file belonging to a
    dataset.
    """

    path: str

    checksum: Optional[str] = None

    checksum_algorithm: str = "sha256"

    size_bytes: Optional[int] = None

    media_type: Optional[str] = None

    record_count: Optional[int] = None

    required: bool = True

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    file_id: Optional[str] = None

    def finalize(
        self,
    ) -> "DatasetFile":
        required_string(
            self.path,
            name="dataset file path",
            maximum_length=4096,
        )

        required_string(
            self.checksum_algorithm,
            name="checksum_algorithm",
            maximum_length=64,
        )

        if self.size_bytes is not None and int(self.size_bytes) < 0:
            raise ValueError("size_bytes must be >= 0")

        if self.record_count is not None and int(self.record_count) < 0:
            raise ValueError("record_count must be >= 0")

        if not self.file_id:
            self.file_id = random_id("data:file")

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "file_id": self.file_id,
            "path": self.path,
            "checksum": self.checksum,
            "checksum_algorithm": (self.checksum_algorithm),
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "record_count": (self.record_count),
            "required": self.required,
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------


@dataclass
class DatasetManifest:
    """
    Authoritative description of one dataset
    version.
    """

    dataset_id: str

    name: str

    version: str

    source: DataSource

    format: str

    purpose: str = DatasetPurpose.GENERAL.value

    files: List[DatasetFile] = field(default_factory=list)

    labels: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    created_at: str = field(default_factory=utc_now)

    schema_version: str = DATA_SCHEMA_VERSION

    def finalize(
        self,
    ) -> "DatasetManifest":
        required_string(
            self.dataset_id,
            name="dataset_id",
            maximum_length=512,
        )

        required_string(
            self.name,
            name="dataset name",
            maximum_length=512,
        )

        required_string(
            self.version,
            name="dataset version",
            maximum_length=128,
        )

        self.source.finalize()

        if self.format not in {item.value for item in DatasetFormat}:
            raise ValueError(f"Unknown dataset format: {self.format}")

        if self.purpose not in {item.value for item in DatasetPurpose}:
            raise ValueError(f"Unknown dataset purpose: {self.purpose}")

        self.files = [item.finalize() for item in self.files]

        self.labels = unique_strings(self.labels)

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (self.schema_version),
            "dataset_id": self.dataset_id,
            "name": self.name,
            "version": self.version,
            "source": (self.source.to_dict()),
            "format": self.format,
            "purpose": self.purpose,
            "files": [item.to_dict() for item in self.files],
            "labels": list(self.labels),
            "attributes": (to_json_safe(self.attributes)),
            "created_at": (self.created_at),
        }


# ---------------------------------------------------------------
# Canonical dataset record
# ---------------------------------------------------------------


@dataclass
class DatasetRecord:
    """
    Provider-neutral ML/data record.

    Raw provider records should be normalized
    into this representation before the training
    layer consumes them.
    """

    dataset_id: str

    record_type: str

    payload: Dict[
        str,
        Any,
    ]

    provider: Optional[str] = None

    environment_id: Optional[str] = None

    resource_type: Optional[str] = None

    resource_id: Optional[str] = None

    event_type: Optional[str] = None

    label: Optional[str] = None

    timestamp: Optional[str] = None

    source_file: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    record_id: Optional[str] = None

    def finalize(
        self,
    ) -> "DatasetRecord":
        required_string(
            self.dataset_id,
            name="dataset_id",
            maximum_length=512,
        )

        required_string(
            self.record_type,
            name="record_type",
            maximum_length=256,
        )

        if not isinstance(
            self.payload,
            dict,
        ):
            raise ValueError("DatasetRecord payload must be a dictionary")

        if not self.record_id:
            self.record_id = random_id("data:record")

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "record_id": self.record_id,
            "dataset_id": self.dataset_id,
            "record_type": (self.record_type),
            "provider": self.provider,
            "environment_id": (self.environment_id),
            "resource_type": (self.resource_type),
            "resource_id": (self.resource_id),
            "event_type": (self.event_type),
            "label": self.label,
            "timestamp": self.timestamp,
            "source_file": (self.source_file),
            "payload": (to_json_safe(self.payload)),
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Validation issue
# ---------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    code: str

    message: str

    severity: str

    file_id: Optional[str] = None

    record_id: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "ValidationIssue":
        required_string(
            self.code,
            name="validation issue code",
            maximum_length=128,
        )

        required_string(
            self.message,
            name=("validation issue message"),
            maximum_length=2048,
        )

        if self.severity not in {item.value for item in ValidationSeverity}:
            raise ValueError(f"Unknown validation severity: {self.severity}")

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "file_id": self.file_id,
            "record_id": self.record_id,
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------


@dataclass
class ValidationResult:
    dataset_id: str

    valid: bool

    records_seen: int = 0

    records_valid: int = 0

    records_invalid: int = 0

    issues: List[ValidationIssue] = field(default_factory=list)

    validated_at: str = field(default_factory=utc_now)

    def finalize(
        self,
    ) -> "ValidationResult":
        required_string(
            self.dataset_id,
            name="dataset_id",
            maximum_length=512,
        )

        for name, value in (
            (
                "records_seen",
                self.records_seen,
            ),
            (
                "records_valid",
                self.records_valid,
            ),
            (
                "records_invalid",
                self.records_invalid,
            ),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} must be >= 0")

        self.issues = [issue.finalize() for issue in self.issues]

        return self

    @property
    def errors(
        self,
    ) -> List[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if (issue.severity == ValidationSeverity.ERROR.value)
        ]

    @property
    def warnings(
        self,
    ) -> List[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if (issue.severity == ValidationSeverity.WARNING.value)
        ]

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "dataset_id": (self.dataset_id),
            "valid": self.valid,
            "records_seen": (self.records_seen),
            "records_valid": (self.records_valid),
            "records_invalid": (self.records_invalid),
            "issues": [issue.to_dict() for issue in self.issues],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "validated_at": (self.validated_at),
        }


# ---------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------


@dataclass
class DatasetSplit:
    """
    Result metadata for one prepared dataset
    partition.
    """

    dataset_id: str

    name: str

    record_ids: List[str] = field(default_factory=list)

    output_path: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def finalize(
        self,
    ) -> "DatasetSplit":
        required_string(
            self.dataset_id,
            name="dataset_id",
            maximum_length=512,
        )

        if self.name not in {item.value for item in SplitName}:
            raise ValueError(f"Unknown dataset split: {self.name}")

        self.record_ids = unique_strings(self.record_ids)

        return self

    @property
    def record_count(
        self,
    ) -> int:
        return len(self.record_ids)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "dataset_id": (self.dataset_id),
            "name": self.name,
            "record_count": (self.record_count),
            "record_ids": list(self.record_ids),
            "output_path": (self.output_path),
            "attributes": (to_json_safe(self.attributes)),
        }


# ---------------------------------------------------------------
# Preparation result
# ---------------------------------------------------------------


@dataclass
class PreparationResult:
    """
    Summary of one dataset preparation run.

    This describes data preparation only. It does
    not contain model-training metrics.
    """

    dataset_id: str

    success: bool

    state: str

    output_directory: Optional[str] = None

    splits: List[DatasetSplit] = field(default_factory=list)

    source_record_count: int = 0

    prepared_record_count: int = 0

    rejected_record_count: int = 0

    reference_ids: List[str] = field(default_factory=list)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    error_type: Optional[str] = None

    completed_at: str = field(default_factory=utc_now)

    preparation_id: Optional[str] = None

    def finalize(
        self,
    ) -> "PreparationResult":
        required_string(
            self.dataset_id,
            name="dataset_id",
            maximum_length=512,
        )

        if self.state not in {item.value for item in DataState}:
            raise ValueError(f"Unknown data state: {self.state}")

        for name, value in (
            (
                "source_record_count",
                self.source_record_count,
            ),
            (
                "prepared_record_count",
                self.prepared_record_count,
            ),
            (
                "rejected_record_count",
                self.rejected_record_count,
            ),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} must be >= 0")

        self.splits = [split.finalize() for split in self.splits]

        self.reference_ids = unique_strings(self.reference_ids)

        if not self.preparation_id:
            self.preparation_id = random_id("data:preparation")

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "schema_version": (DATA_SCHEMA_VERSION),
            "preparation_id": (self.preparation_id),
            "dataset_id": (self.dataset_id),
            "success": self.success,
            "state": self.state,
            "output_directory": (self.output_directory),
            "source_record_count": (self.source_record_count),
            "prepared_record_count": (self.prepared_record_count),
            "rejected_record_count": (self.rejected_record_count),
            "splits": [split.to_dict() for split in self.splits],
            "reference_ids": list(self.reference_ids),
            "attributes": (to_json_safe(self.attributes)),
            "error_type": (self.error_type),
            "completed_at": (self.completed_at),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DATA_SCHEMA_VERSION",
    "DataSourceKind",
    "DatasetFormat",
    "DatasetPurpose",
    "DataState",
    "ValidationSeverity",
    "SplitName",
    "DataSource",
    "DatasetFile",
    "DatasetManifest",
    "DatasetRecord",
    "ValidationIssue",
    "ValidationResult",
    "DatasetSplit",
    "PreparationResult",
]
