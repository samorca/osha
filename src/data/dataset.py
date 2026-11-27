"""


Dataset abstractions for the OSHA orchestration/self-healing platform.

This module provides:

- Dataset lifecycle management
- Dataset metadata
- Record collections
- Manifest integration
- Filtering and mapping
- Dataset fingerprints
- Serialization-friendly representations
- Dataset statistics

Persistence, database access, and durable storage are intentionally handled
by other modules in src/data/.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from uuid import uuid4

from .errors import (
    ManifestError,
)
from .schema import (
    DataSource,
    DatasetFile,
    DatasetFormat,
    DatasetManifest,
    DatasetPurpose,
)


class DatasetError(Exception):
    """Base exception for dataset-related failures."""


class DatasetValidationError(DatasetError):
    """Raised when dataset validation fails."""


class DatasetStateError(DatasetError):
    """Raised when an operation is invalid for the current dataset state."""


class DatasetState(str, Enum):
    """Lifecycle state of a dataset."""

    CREATED = "created"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    ARCHIVED = "archived"


def utc_now() -> datetime:
    """Return current timezone-aware UTC time."""

    return datetime.now(timezone.utc)


def _serialize_value(value: Any) -> Any:
    """
    Convert common Python values into JSON-compatible representations.

    Unknown values are converted to strings instead of allowing serialization
    to fail unexpectedly during fingerprint generation.
    """

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc).isoformat()

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Mapping):
        return {str(key): _serialize_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return str(value)


@dataclass(slots=True)
class DatasetRecord:
    """
    Generic record stored inside a dataset.

    The actual payload remains schema-neutral. This allows collectors and
    pipelines to use the data layer without coupling it to one infrastructure
    provider.
    """

    data: dict[str, Any]

    record_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=utc_now)

    source: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the record."""

        if not self.record_id.strip():
            raise DatasetValidationError("record_id cannot be empty.")

        if not isinstance(self.data, dict):
            raise DatasetValidationError("DatasetRecord.data must be a dictionary.")

        if self.timestamp.tzinfo is None:
            raise DatasetValidationError(
                "DatasetRecord.timestamp must be timezone-aware."
            )

        for key, value in self.labels.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise DatasetValidationError(
                    "Record labels must contain string keys and values."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        self.validate()

        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "source": self.source,
            "data": _serialize_value(self.data),
            "labels": dict(self.labels),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "DatasetRecord":
        """Create a record from serialized data."""

        timestamp_value = value.get("timestamp")

        if timestamp_value:
            try:
                timestamp = datetime.fromisoformat(
                    str(timestamp_value).replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise DatasetValidationError(
                    f"Invalid record timestamp: {timestamp_value!r}"
                ) from exc

            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = utc_now()

        record = cls(
            record_id=str(value.get("record_id") or uuid4()),
            timestamp=timestamp.astimezone(timezone.utc),
            source=(str(value["source"]) if value.get("source") is not None else None),
            data=dict(value.get("data") or {}),
            labels=dict(value.get("labels") or {}),
            metadata=dict(value.get("metadata") or {}),
        )

        record.validate()
        return record


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Basic dataset statistics."""

    record_count: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    source_count: int
    label_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "first_timestamp": (
                self.first_timestamp.isoformat() if self.first_timestamp else None
            ),
            "last_timestamp": (
                self.last_timestamp.isoformat() if self.last_timestamp else None
            ),
            "source_count": self.source_count,
            "label_count": self.label_count,
        }


@dataclass(slots=True)
class Dataset:
    """
    In-memory collection of related records.

    Dataset is intentionally provider-neutral. A dataset can contain telemetry
    collected from Kubernetes, OpenStack, Linux, Ceph, MinIO, or future
    infrastructure integrations.
    """

    name: str

    dataset_id: str = field(default_factory=lambda: str(uuid4()))
    description: str | None = None
    version: str = "1"

    state: DatasetState = DatasetState.CREATED

    records: list[DatasetRecord] = field(default_factory=list)

    source: DataSource | None = None

    schema_name: str | None = None
    schema_version: str | None = None

    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.records)

    def __bool__(self) -> bool:
        return bool(self.records)

    def validate(self) -> None:
        """Validate dataset invariants."""

        if not self.dataset_id.strip():
            raise DatasetValidationError("dataset_id cannot be empty.")

        if not self.name.strip():
            raise DatasetValidationError("Dataset name cannot be empty.")

        if not self.version.strip():
            raise DatasetValidationError("Dataset version cannot be empty.")

        if self.created_at.tzinfo is None:
            raise DatasetValidationError("created_at must be timezone-aware.")

        if self.updated_at.tzinfo is None:
            raise DatasetValidationError("updated_at must be timezone-aware.")

        if self.updated_at < self.created_at:
            raise DatasetValidationError("updated_at cannot occur before created_at.")

        if self.source is not None:
            try:
                self.source.finalize()
            except (ManifestError, TypeError, ValueError) as exc:
                raise DatasetValidationError(f"Invalid dataset source: {exc}") from exc

        for key, value in self.labels.items():
            if not isinstance(key, str) or not isinstance(
                value,
                str,
            ):
                raise DatasetValidationError(
                    "Dataset labels must contain string keys and values."
                )

        seen_ids: set[str] = set()

        for record in self.records:
            if not isinstance(record, DatasetRecord):
                raise DatasetValidationError(
                    "records must contain DatasetRecord instances."
                )

            record.validate()

            if record.record_id in seen_ids:
                raise DatasetValidationError(f"Duplicate record_id: {record.record_id}")

            seen_ids.add(record.record_id)

    def touch(self) -> None:
        """Update the dataset modification timestamp."""

        self.updated_at = utc_now()

    def set_state(
        self,
        state: DatasetState | str,
    ) -> None:
        """Set the dataset lifecycle state."""

        self.state = DatasetState(state)
        self.touch()

    def mark_ready(self) -> None:
        self.set_state(DatasetState.READY)

    def mark_processing(self) -> None:
        self.set_state(DatasetState.PROCESSING)

    def mark_complete(self) -> None:
        self.set_state(DatasetState.COMPLETE)

    def mark_failed(self) -> None:
        self.set_state(DatasetState.FAILED)

    def archive(self) -> None:
        self.set_state(DatasetState.ARCHIVED)

    def _ensure_mutable(self) -> None:
        if self.state == DatasetState.ARCHIVED:
            raise DatasetStateError("Archived datasets cannot be modified.")

    def add(
        self,
        record: DatasetRecord,
    ) -> None:
        """Add one record to the dataset."""

        self._ensure_mutable()

        if not isinstance(record, DatasetRecord):
            raise TypeError("record must be a DatasetRecord instance.")

        record.validate()

        if self.get(record.record_id) is not None:
            raise DatasetValidationError(f"Duplicate record_id: {record.record_id}")

        self.records.append(record)
        self.touch()

    def append(
        self,
        data: Mapping[str, Any],
        *,
        source: str | None = None,
        labels: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> DatasetRecord:
        """
        Construct and add a record.

        Returns the newly created record.
        """

        record = DatasetRecord(
            data=dict(data),
            source=source,
            labels=dict(labels or {}),
            metadata=dict(metadata or {}),
            timestamp=timestamp or utc_now(),
        )

        self.add(record)
        return record

    def extend(
        self,
        records: Iterable[DatasetRecord],
    ) -> None:
        """Add multiple records."""

        self._ensure_mutable()

        new_records = list(records)

        existing_ids = {record.record_id for record in self.records}

        batch_ids: set[str] = set()

        for record in new_records:
            if not isinstance(record, DatasetRecord):
                raise TypeError("records must contain DatasetRecord instances.")

            record.validate()

            if record.record_id in existing_ids or record.record_id in batch_ids:
                raise DatasetValidationError(f"Duplicate record_id: {record.record_id}")

            batch_ids.add(record.record_id)

        self.records.extend(new_records)
        self.touch()

    def get(
        self,
        record_id: str,
    ) -> DatasetRecord | None:
        """Return a record by ID."""

        for record in self.records:
            if record.record_id == record_id:
                return record

        return None

    def require(
        self,
        record_id: str,
    ) -> DatasetRecord:
        """
        Return a record by ID.

        Raises KeyError if the record does not exist.
        """

        record = self.get(record_id)

        if record is None:
            raise KeyError(f"Dataset record not found: {record_id}")

        return record

    def remove(
        self,
        record_id: str,
    ) -> DatasetRecord:
        """Remove and return a record."""

        self._ensure_mutable()

        for index, record in enumerate(self.records):
            if record.record_id == record_id:
                removed = self.records.pop(index)
                self.touch()
                return removed

        raise KeyError(f"Dataset record not found: {record_id}")

    def clear(self) -> None:
        """Remove all records."""

        self._ensure_mutable()

        self.records.clear()
        self.touch()

    def filter(
        self,
        predicate: Callable[[DatasetRecord], bool],
        *,
        name: str | None = None,
    ) -> "Dataset":
        """
        Create a new dataset containing records matching a predicate.

        Record instances are copied through serialization to prevent accidental
        mutations in the derived dataset from changing the original dataset.
        """

        records = [
            DatasetRecord.from_dict(record.to_dict())
            for record in self.records
            if predicate(record)
        ]

        return Dataset(
            name=name or f"{self.name}-filtered",
            description=self.description,
            version=self.version,
            state=DatasetState.READY,
            records=records,
            source=self.source,
            schema_name=self.schema_name,
            schema_version=self.schema_version,
            labels=dict(self.labels),
            metadata={
                **self.metadata,
                "parent_dataset_id": self.dataset_id,
            },
        )

    def map(
        self,
        transform: Callable[
            [DatasetRecord],
            DatasetRecord,
        ],
        *,
        name: str | None = None,
    ) -> "Dataset":
        """Create a transformed dataset."""

        transformed: list[DatasetRecord] = []

        for record in self.records:
            result = transform(record)

            if not isinstance(result, DatasetRecord):
                raise DatasetValidationError(
                    "Dataset transform must return DatasetRecord instances."
                )

            result.validate()
            transformed.append(result)

        return Dataset(
            name=name or f"{self.name}-mapped",
            description=self.description,
            version=self.version,
            state=DatasetState.READY,
            records=transformed,
            source=self.source,
            schema_name=self.schema_name,
            schema_version=self.schema_version,
            labels=dict(self.labels),
            metadata={
                **self.metadata,
                "parent_dataset_id": self.dataset_id,
            },
        )

    def stats(self) -> DatasetStats:
        """Calculate basic statistics for the dataset."""

        if not self.records:
            return DatasetStats(
                record_count=0,
                first_timestamp=None,
                last_timestamp=None,
                source_count=0,
                label_count=0,
            )

        timestamps = [record.timestamp for record in self.records]

        sources = {record.source for record in self.records if record.source}

        label_names: set[str] = set()

        for record in self.records:
            label_names.update(record.labels)

        return DatasetStats(
            record_count=len(self.records),
            first_timestamp=min(timestamps),
            last_timestamp=max(timestamps),
            source_count=len(sources),
            label_count=len(label_names),
        )

    def fingerprint(self) -> str:
        """
        Return a stable SHA-256 fingerprint of dataset records.

        Operational metadata such as dataset state and updated_at is excluded.
        Therefore the fingerprint represents the dataset contents rather than
        its current lifecycle state.
        """

        payload = [record.to_dict() for record in self.records]

        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    def build_manifest(
        self,
        *,
        uri: str | None = None,
        content_type: str = "application/json",
        format: str = "json",
    ) -> DatasetManifest:
        """
        Build a DatasetManifest describing this dataset.

        Note that this does not persist the dataset.
        """

        manifest_uri = uri or f"dataset://{self.dataset_id}"
        manifest = DatasetManifest(
            dataset_id=self.dataset_id,
            name=self.name,
            version=self.version,
            source=self.source
            or DataSource(
                kind="generated",
                location=manifest_uri,
                provider="osha",
            ),
            format=format,
            purpose=DatasetPurpose.GENERAL.value,
            files=[
                DatasetFile(
                    path=manifest_uri,
                    checksum=self.fingerprint(),
                    media_type=content_type,
                    record_count=len(self.records),
                )
            ],
            labels=list(self.labels),
            attributes={
                **self.metadata,
                "dataset_state": self.state.value,
                "schema_name": self.schema_name,
                "schema_version": self.schema_version,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            },
        )

        manifest.finalize()
        return manifest

    def to_dict(
        self,
        *,
        include_records: bool = True,
    ) -> dict[str, Any]:
        """Serialize the dataset."""

        self.validate()

        result: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "state": self.state.value,
            "source": (self.source.to_dict() if self.source else None),
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
            "updated_at": self.updated_at.astimezone(timezone.utc).isoformat(),
            "labels": dict(self.labels),
            "metadata": _serialize_value(self.metadata),
        }

        if include_records:
            result["records"] = [record.to_dict() for record in self.records]

        return result

    def to_json(
        self,
        *,
        include_records: bool = True,
        indent: int | None = 2,
    ) -> str:
        """Serialize the dataset to JSON."""

        return json.dumps(
            self.to_dict(
                include_records=include_records,
            ),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "Dataset":
        """Create a dataset from serialized data."""

        try:
            source_data = value.get("source")

            created_at = datetime.fromisoformat(
                str(value.get("created_at") or utc_now().isoformat()).replace(
                    "Z", "+00:00"
                )
            )

            updated_at = datetime.fromisoformat(
                str(value.get("updated_at") or created_at.isoformat()).replace(
                    "Z", "+00:00"
                )
            )

            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)

            records_value = value.get("records") or []

            if not isinstance(records_value, Sequence):
                raise DatasetValidationError("records must be a sequence.")

            records = [DatasetRecord.from_dict(record) for record in records_value]

            dataset = cls(
                dataset_id=str(value.get("dataset_id") or uuid4()),
                name=str(value["name"]),
                description=value.get("description"),
                version=str(value.get("version", "1")),
                state=DatasetState(
                    value.get(
                        "state",
                        DatasetState.CREATED.value,
                    )
                ),
                records=records,
                source=(DataSource(**source_data) if source_data else None),
                schema_name=value.get("schema_name"),
                schema_version=value.get("schema_version"),
                created_at=created_at.astimezone(timezone.utc),
                updated_at=updated_at.astimezone(timezone.utc),
                labels=dict(value.get("labels") or {}),
                metadata=dict(value.get("metadata") or {}),
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            ManifestError,
        ) as exc:
            raise DatasetValidationError(f"Invalid dataset data: {exc}") from exc

        dataset.validate()
        return dataset

    @classmethod
    def from_json(
        cls,
        payload: str,
    ) -> "Dataset":
        """Deserialize a dataset from JSON."""

        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError("Dataset contains invalid JSON.") from exc

        if not isinstance(value, dict):
            raise DatasetValidationError("Dataset JSON root must be an object.")

        return cls.from_dict(value)


def dataset_from_records(
    name: str,
    records: Iterable[Mapping[str, Any]],
    *,
    source: DataSource | None = None,
    description: str | None = None,
    schema_name: str | None = None,
    schema_version: str | None = None,
    labels: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Dataset:
    """
    Convenience constructor for building a dataset from dictionaries.
    """

    dataset = Dataset(
        name=name,
        description=description,
        source=source,
        schema_name=schema_name,
        schema_version=schema_version,
        labels=dict(labels or {}),
        metadata=dict(metadata or {}),
    )

    for value in records:
        dataset.append(value)

    dataset.mark_ready()

    return dataset


__all__ = [
    "Dataset",
    "DatasetError",
    "DatasetRecord",
    "DatasetState",
    "DatasetStateError",
    "DatasetStats",
    "DatasetValidationError",
    "dataset_from_records",
]
