from __future__ import annotations

"""
Canonical dataset-record normalization for OSHA.

Raw dataset records are converted into the
provider-neutral DatasetRecord contract before
training consumes them.
"""

import hashlib
import json

from dataclasses import (
    dataclass,
)

from threading import (
    RLock,
)

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.core.serialization import (
    to_json_safe,
)

from src.data.errors import (
    DataNormalizationError,
    UnsupportedRecordError,
)

from src.data.schema import (
    DatasetFile,
    DatasetManifest,
    DatasetRecord,
)


# ---------------------------------------------------------------
# Normalizer contract
# ---------------------------------------------------------------

RecordNormalizer = Callable[
    [
        Any,
        DatasetManifest,
        DatasetFile,
        int,
    ],
    DatasetRecord,
]


# ---------------------------------------------------------------
# Normalization result
# ---------------------------------------------------------------

@dataclass
class NormalizationResult:
    dataset_id: str

    source_records: int

    normalized_records: int

    rejected_records: int

    records: List[
        DatasetRecord
    ]

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "dataset_id": (
                self.dataset_id
            ),
            "source_records": (
                self.source_records
            ),
            "normalized_records": (
                self.normalized_records
            ),
            "rejected_records": (
                self.rejected_records
            ),
            "records": [
                record.to_dict()
                for record
                in self.records
            ],
        }


# ---------------------------------------------------------------
# Normalizer registry key
# ---------------------------------------------------------------

def normalizer_key(
    *,
    provider: Optional[str],
    record_type: str,
) -> str:
    provider_value = (
        str(
            provider
        ).strip().lower()
        if provider
        else "*"
    )

    record_type_value = str(
        record_type
    ).strip().lower()

    if not record_type_value:
        raise DataNormalizationError(
            "record_type is required."
        )

    return (
        provider_value
        + ":"
        + record_type_value
    )


# ---------------------------------------------------------------
# Deterministic record ID
# ---------------------------------------------------------------

def deterministic_record_id(
    *,
    dataset_id: str,
    source_file: str,
    record_number: int,
    value: Any,
) -> str:
    """
    Build a repeatable identifier from record
    provenance and canonicalized value.

    Re-running preparation against identical
    source data therefore produces the same
    record identity.
    """

    try:
        safe_value = to_json_safe(
            value
        )

        serialized = json.dumps(
            safe_value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    except Exception as exc:
        raise DataNormalizationError(
            "Dataset record could not be "
            "canonicalized for identity.",
            details={
                "error_type": (
                    type(exc).__name__
                ),
            },
        ) from exc

    material = (
        f"{dataset_id}\n"
        f"{source_file}\n"
        f"{record_number}\n"
        f"{serialized}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        material
    ).hexdigest()

    return (
        "data-record-"
        + digest
    )


# ---------------------------------------------------------------
# Generic payload conversion
# ---------------------------------------------------------------

def _payload(
    value: Any,
) -> Dict[str, Any]:
    """
    Convert common raw record structures into a
    canonical payload dictionary.
    """

    if isinstance(
        value,
        Mapping,
    ):
        safe = to_json_safe(
            dict(
                value
            )
        )

        if not isinstance(
            safe,
            dict,
        ):
            raise DataNormalizationError(
                "Record mapping did not "
                "normalize to an object."
            )

        return safe

    if isinstance(
        value,
        str,
    ):
        return {
            "message": value,
        }

    if isinstance(
        value,
        (
            int,
            float,
            bool,
        ),
    ):
        return {
            "value": value,
        }

    if isinstance(
        value,
        list,
    ):
        return {
            "value": (
                to_json_safe(
                    value
                )
            ),
        }

    raise UnsupportedRecordError(
        "Raw dataset record type is "
        "not supported.",
        details={
            "record_type": (
                type(value).__name__
            ),
        },
    )


# ---------------------------------------------------------------
# Generic field extraction
# ---------------------------------------------------------------

def _field(
    value: Mapping[
        str,
        Any,
    ],
    *names: str,
) -> Optional[str]:
    for name in names:
        item = value.get(
            name
        )

        if item is None:
            continue

        text = str(
            item
        ).strip()

        if text:
            return text

    return None


# ---------------------------------------------------------------
# Default normalizer
# ---------------------------------------------------------------

def default_normalizer(
    value: Any,
    manifest: DatasetManifest,
    dataset_file: DatasetFile,
    record_number: int,
) -> DatasetRecord:
    """
    Generic normalization for text/log/JSON/CSV
    records.

    Provider-specific semantics should be handled
    by a registered provider normalizer rather
    than continuously expanding this function.
    """

    manifest.finalize()
    dataset_file.finalize()

    payload = _payload(
        value
    )

    provider = (
        manifest.source.provider
    )

    environment = None
    resource_type = None
    resource_id = None
    event_type = None
    label = None
    timestamp = None

    if isinstance(
        value,
        Mapping,
    ):
        environment = _field(
            value,
            "environment_id",
            "environment",
        )

        resource_type = _field(
            value,
            "resource_type",
            "resource",
        )

        resource_id = _field(
            value,
            "resource_id",
            "resource_name",
        )

        event_type = _field(
            value,
            "event_type",
            "event",
            "type",
        )

        label = _field(
            value,
            "label",
            "target",
            "class",
        )

        timestamp = _field(
            value,
            "timestamp",
            "time",
            "datetime",
            "@timestamp",
        )

        provider = (
            _field(
                value,
                "provider",
                "source_provider",
            )
            or provider
        )

    record_type = (
        event_type
        or "record"
    )

    record_id = (
        deterministic_record_id(
            dataset_id=(
                manifest.dataset_id
            ),
            source_file=(
                dataset_file.path
            ),
            record_number=(
                record_number
            ),
            value=value,
        )
    )

    return DatasetRecord(
        dataset_id=(
            manifest.dataset_id
        ),

        record_type=(
            record_type
        ),

        payload=payload,

        provider=provider,

        environment_id=(
            environment
        ),

        resource_type=(
            resource_type
        ),

        resource_id=(
            resource_id
        ),

        event_type=(
            event_type
        ),

        label=label,

        timestamp=(
            timestamp
        ),

        source_file=(
            dataset_file.path
        ),

        attributes={
            "source_record_number": (
                record_number
            ),

            "source_format": (
                manifest.format
            ),

            "dataset_version": (
                manifest.version
            ),
        },

        record_id=record_id,
    ).finalize()


# ---------------------------------------------------------------
# Normalizer registry
# ---------------------------------------------------------------

class NormalizerRegistry:
    """
    Thread-safe registry for provider/record-type
    specific normalizers.

    Lookup precedence:

      provider + record_type
      provider + *
      * + record_type
      * + *
    """

    def __init__(
        self,
    ) -> None:
        self._normalizers: Dict[
            str,
            RecordNormalizer,
        ] = {}

        self._lock = RLock()

    # ---------------------------------------------------------
    # Register
    # ---------------------------------------------------------

    def register(
        self,
        *,
        record_type: str,
        normalizer: RecordNormalizer,
        provider: Optional[
            str
        ] = None,
        replace: bool = False,
    ) -> None:
        if not callable(
            normalizer
        ):
            raise DataNormalizationError(
                "normalizer must be "
                "callable."
            )

        key = normalizer_key(
            provider=provider,
            record_type=(
                record_type
            ),
        )

        with self._lock:
            if (
                key
                in self._normalizers
                and not replace
            ):
                raise DataNormalizationError(
                    "Dataset normalizer is "
                    "already registered.",
                    details={
                        "key": key,
                    },
                )

            self._normalizers[
                key
            ] = normalizer

    # ---------------------------------------------------------
    # Resolve
    # ---------------------------------------------------------

    def resolve(
        self,
        *,
        provider: Optional[
            str
        ],
        record_type: str,
    ) -> Optional[
        RecordNormalizer
    ]:
        provider_name = (
            str(
                provider
            ).strip().lower()
            if provider
            else "*"
        )

        record_name = (
            str(
                record_type
            ).strip().lower()
        )

        candidates = (
            f"{provider_name}:{record_name}",
            f"{provider_name}:*",
            f"*:{record_name}",
            "*:*",
        )

        with self._lock:
            for key in candidates:
                normalizer = (
                    self._normalizers.get(
                        key
                    )
                )

                if normalizer is not None:
                    return normalizer

        return None

    # ---------------------------------------------------------
    # List keys
    # ---------------------------------------------------------

    def keys(
        self,
    ) -> List[str]:
        with self._lock:
            return sorted(
                self._normalizers
            )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        keys = self.keys()

        return {
            "normalizer_count": len(
                keys
            ),
            "registered_keys": keys,
        }


# ---------------------------------------------------------------
# Dataset normalizer service
# ---------------------------------------------------------------

class DatasetNormalizer:
    """
    Normalize raw records into DatasetRecord
    objects.

    The service can use explicitly registered
    provider-specific converters while retaining
    a generic fallback.
    """

    def __init__(
        self,
        *,
        registry: Optional[
            NormalizerRegistry
        ] = None,
        use_default_fallback: bool = True,
    ) -> None:
        self.registry = (
            registry
            or NormalizerRegistry()
        )

        self.use_default_fallback = bool(
            use_default_fallback
        )

    # ---------------------------------------------------------
    # Resolve normalizer
    # ---------------------------------------------------------

    def _normalizer(
        self,
        *,
        manifest: DatasetManifest,
        value: Any,
    ) -> RecordNormalizer:
        provider = (
            manifest.source.provider
        )

        raw_record_type = (
            "record"
        )

        if isinstance(
            value,
            Mapping,
        ):
            raw_record_type = (
                _field(
                    value,
                    "record_type",
                    "event_type",
                    "event",
                    "type",
                )
                or "record"
            )

            provider = (
                _field(
                    value,
                    "provider",
                    "source_provider",
                )
                or provider
            )

        normalizer = (
            self.registry.resolve(
                provider=provider,
                record_type=(
                    raw_record_type
                ),
            )
        )

        if normalizer is not None:
            return normalizer

        if self.use_default_fallback:
            return default_normalizer

        raise UnsupportedRecordError(
            "No normalizer is registered "
            "for the dataset record.",
            details={
                "provider": provider,
                "record_type": (
                    raw_record_type
                ),
            },
        )

    # ---------------------------------------------------------
    # Normalize one
    # ---------------------------------------------------------

    def normalize_one(
        self,
        value: Any,
        *,
        manifest: DatasetManifest,
        dataset_file: DatasetFile,
        record_number: int,
    ) -> DatasetRecord:
        if record_number < 1:
            raise DataNormalizationError(
                "record_number must be "
                ">= 1."
            )

        normalizer = self._normalizer(
            manifest=manifest,
            value=value,
        )

        try:
            record = normalizer(
                value,
                manifest,
                dataset_file,
                record_number,
            )

        except (
            DataNormalizationError,
            UnsupportedRecordError,
        ):
            raise

        except Exception as exc:
            raise DataNormalizationError(
                "Dataset normalizer "
                "failed.",
                details={
                    "dataset_id": (
                        manifest.dataset_id
                    ),
                    "file_id": (
                        dataset_file.file_id
                    ),
                    "record_number": (
                        record_number
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error_module": (
                        type(exc).__module__
                    ),
                },
            ) from exc

        if not isinstance(
            record,
            DatasetRecord,
        ):
            raise DataNormalizationError(
                "Dataset normalizer did "
                "not return a "
                "DatasetRecord."
            )

        return record.finalize()

    # ---------------------------------------------------------
    # Normalize iterable
    # ---------------------------------------------------------

    def normalize_many(
        self,
        values: Iterable[
            Any
        ],
        *,
        manifest: DatasetManifest,
        dataset_file: DatasetFile,
        start_record_number: int = 1,
        reject_invalid: bool = True,
    ) -> NormalizationResult:
        """
        Normalize a stream/iterable of values.

        Records aren't duplicated internally
        beyond the returned list.
        """

        if start_record_number < 1:
            raise DataNormalizationError(
                "start_record_number must "
                "be >= 1."
            )

        records: List[
            DatasetRecord
        ] = []

        source_records = 0
        rejected_records = 0

        for offset, value in enumerate(
            values
        ):
            source_records += 1

            record_number = (
                start_record_number
                + offset
            )

            try:
                record = self.normalize_one(
                    value,
                    manifest=manifest,
                    dataset_file=(
                        dataset_file
                    ),
                    record_number=(
                        record_number
                    ),
                )

                records.append(
                    record
                )

            except (
                DataNormalizationError,
                UnsupportedRecordError,
            ):
                rejected_records += 1

                if not reject_invalid:
                    raise

        return NormalizationResult(
            dataset_id=(
                manifest.dataset_id
            ),
            source_records=(
                source_records
            ),
            normalized_records=len(
                records
            ),
            rejected_records=(
                rejected_records
            ),
            records=records,
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "default_fallback_enabled": (
                self.use_default_fallback
            ),
            "registry": (
                self.registry.summary()
            ),
            "canonical_record_type": (
                "DatasetRecord"
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "RecordNormalizer",

    "NormalizationResult",

    "normalizer_key",
    "deterministic_record_id",

    "default_normalizer",

    "NormalizerRegistry",
    "DatasetNormalizer",
]
