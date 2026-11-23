from __future__ import annotations

"""
Exception hierarchy for OSHA data engineering.
"""

from typing import (
    Any,
    Mapping,
    Optional,
)

from src.core.exceptions import (
    ConflictError,
    NotFoundError,
    ValidationError,
)


# ---------------------------------------------------------------
# Base data error
# ---------------------------------------------------------------

class DataError(
    RuntimeError
):
    """
    Base exception for the OSHA data subsystem.
    """

    default_code = "data_error"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        retryable: bool = False,
        details: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        super().__init__(
            message
        )

        self.message = message

        self.code = (
            code
            or self.default_code
        )

        self.retryable = bool(
            retryable
        )

        self.details = dict(
            details
            or {}
        )


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

class DataConfigurationError(
    DataError
):
    default_code = (
        "data_configuration_error"
    )


# ---------------------------------------------------------------
# Source errors
# ---------------------------------------------------------------

class DataSourceError(
    DataError
):
    default_code = (
        "data_source_error"
    )


class DataSourceNotFoundError(
    DataSourceError,
    NotFoundError,
):
    default_code = (
        "data_source_not_found"
    )


class DataSourceUnsupportedError(
    DataSourceError
):
    default_code = (
        "data_source_unsupported"
    )


# ---------------------------------------------------------------
# Fetch errors
# ---------------------------------------------------------------

class DataFetchError(
    DataError
):
    default_code = (
        "data_fetch_error"
    )


class DataFetchUnavailableError(
    DataFetchError
):
    """
    Temporary source availability failure.

    Classification as retryable does not mean
    callers must automatically retry.
    """

    default_code = (
        "data_fetch_unavailable"
    )

    def __init__(
        self,
        message: str = (
            "Dataset source is "
            "temporarily unavailable."
        ),
        *,
        details: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        super().__init__(
            message,
            code=self.default_code,
            retryable=True,
            details=details,
        )


class DataFetchLimitError(
    DataFetchError
):
    default_code = (
        "data_fetch_limit"
    )


# ---------------------------------------------------------------
# Integrity/checksum errors
# ---------------------------------------------------------------

class DataIntegrityError(
    DataError
):
    default_code = (
        "data_integrity_error"
    )


class ChecksumError(
    DataIntegrityError
):
    default_code = (
        "checksum_error"
    )


class ChecksumMismatchError(
    ChecksumError
):
    default_code = (
        "checksum_mismatch"
    )


class ChecksumAlgorithmError(
    ChecksumError
):
    default_code = (
        "checksum_algorithm_error"
    )


class DatasetFileSizeError(
    DataIntegrityError
):
    default_code = (
        "dataset_file_size_error"
    )


# ---------------------------------------------------------------
# Manifest errors
# ---------------------------------------------------------------

class ManifestError(
    DataError
):
    default_code = (
        "manifest_error"
    )


class ManifestNotFoundError(
    ManifestError,
    NotFoundError,
):
    default_code = (
        "manifest_not_found"
    )


class ManifestConflictError(
    ManifestError,
    ConflictError,
):
    default_code = (
        "manifest_conflict"
    )


class ManifestValidationError(
    ManifestError,
    ValidationError,
):
    default_code = (
        "manifest_validation_error"
    )


# ---------------------------------------------------------------
# Record/dataset validation errors
# ---------------------------------------------------------------

class DataValidationError(
    DataError,
    ValidationError,
):
    default_code = (
        "data_validation_error"
    )


class DatasetEmptyError(
    DataValidationError
):
    default_code = (
        "dataset_empty"
    )


class DatasetFormatError(
    DataValidationError
):
    default_code = (
        "dataset_format_error"
    )


class DatasetRecordError(
    DataValidationError
):
    default_code = (
        "dataset_record_error"
    )


class RecordSizeError(
    DatasetRecordError
):
    default_code = (
        "record_size_error"
    )


# ---------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------

class DataNormalizationError(
    DataError
):
    default_code = (
        "data_normalization_error"
    )


class UnsupportedRecordError(
    DataNormalizationError
):
    default_code = (
        "unsupported_record"
    )


# ---------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------

class DataSplitError(
    DataError
):
    default_code = (
        "data_split_error"
    )


class DataSplitConfigurationError(
    DataSplitError
):
    default_code = (
        "data_split_configuration_error"
    )


# ---------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------

class DataPreparationError(
    DataError
):
    default_code = (
        "data_preparation_error"
    )


class PreparedDatasetConflictError(
    DataPreparationError,
    ConflictError,
):
    default_code = (
        "prepared_dataset_conflict"
    )


# ---------------------------------------------------------------
# Serialization / parsing
# ---------------------------------------------------------------

class DataParseError(
    DataError
):
    default_code = (
        "data_parse_error"
    )


class DataSerializationError(
    DataError
):
    default_code = (
        "data_serialization_error"
    )


# ---------------------------------------------------------------
# Safe external/filesystem error details
# ---------------------------------------------------------------

def safe_error_details(
    error: BaseException,
    *,
    operation: Optional[
        str
    ] = None,
    source_kind: Optional[
        str
    ] = None,
) -> Dict[str, Any]:
    """
    Convert an underlying exception into bounded
    metadata.

    Arbitrary exception messages are deliberately
    not copied.
    """

    details: Dict[
        str,
        Any,
    ] = {
        "error_type": (
            type(error).__name__
        ),

        "error_module": (
            type(error).__module__
        ),
    }

    if operation is not None:
        details[
            "operation"
        ] = operation

    if source_kind is not None:
        details[
            "source_kind"
        ] = source_kind

    return details


# ---------------------------------------------------------------
# Error wrapper
# ---------------------------------------------------------------

def wrap_data_error(
    error: BaseException,
    *,
    message: str = (
        "Data operation failed."
    ),
    code: str = (
        "data_operation_failed"
    ),
    retryable: bool = False,
    operation: Optional[
        str
    ] = None,
    source_kind: Optional[
        str
    ] = None,
) -> DataError:
    """
    Wrap a filesystem/parser/source exception
    without copying its raw message.
    """

    return DataError(
        message,
        code=code,
        retryable=retryable,
        details=safe_error_details(
            error,
            operation=operation,
            source_kind=source_kind,
        ),
    )


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DataError",

    "DataConfigurationError",

    "DataSourceError",
    "DataSourceNotFoundError",
    "DataSourceUnsupportedError",

    "DataFetchError",
    "DataFetchUnavailableError",
    "DataFetchLimitError",

    "DataIntegrityError",
    "ChecksumError",
    "ChecksumMismatchError",
    "ChecksumAlgorithmError",
    "DatasetFileSizeError",

    "ManifestError",
    "ManifestNotFoundError",
    "ManifestConflictError",
    "ManifestValidationError",

    "DataValidationError",
    "DatasetEmptyError",
    "DatasetFormatError",
    "DatasetRecordError",
    "RecordSizeError",

    "DataNormalizationError",
    "UnsupportedRecordError",

    "DataSplitError",
    "DataSplitConfigurationError",

    "DataPreparationError",
    "PreparedDatasetConflictError",

    "DataParseError",
    "DataSerializationError",

    "safe_error_details",
    "wrap_data_error",
]
