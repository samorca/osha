from __future__ import annotations

"""
Dataset and record validation for OSHA data
engineering.

Validation reports problems but does not silently
repair or rewrite source data.
"""

import csv
import json

from collections.abc import (
    Callable,
    Iterator,
    Mapping,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from src.data.checksums import (
    FileIntegrityResult,
    manifest_integrity_ok,
    verify_manifest_files,
)

from src.data.config import (
    DataConfig,
)

from src.data.errors import (
    DataValidationError,
    DatasetEmptyError,
    DatasetFormatError,
    DatasetRecordError,
    RecordSizeError,
)

from src.data.manifests import (
    resolve_dataset_file,
)

from src.data.schema import (
    DatasetFile,
    DatasetFormat,
    DatasetManifest,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)


# ---------------------------------------------------------------
# Record validator contract
# ---------------------------------------------------------------

RecordValidator = Callable[
    [
        Any,
        DatasetFile,
        int,
    ],
    Optional[
        ValidationIssue
    ],
]


# ---------------------------------------------------------------
# Parsed raw record
# ---------------------------------------------------------------

RawRecord = Dict[
    str,
    Any,
]


# ---------------------------------------------------------------
# Issue helper
# ---------------------------------------------------------------

def validation_issue(
    *,
    code: str,
    message: str,
    severity: str,
    dataset_file: Optional[
        DatasetFile
    ] = None,
    record_number: Optional[
        int
    ] = None,
    attributes: Optional[
        Dict[
            str,
            Any,
        ]
    ] = None,
) -> ValidationIssue:
    issue_attributes = dict(
        attributes
        or {}
    )

    if record_number is not None:
        issue_attributes[
            "record_number"
        ] = record_number

    return ValidationIssue(
        code=code,
        message=message,
        severity=severity,
        file_id=(
            dataset_file.file_id
            if dataset_file
            is not None
            else None
        ),
        attributes=(
            issue_attributes
        ),
    ).finalize()


# ---------------------------------------------------------------
# Record-size check
# ---------------------------------------------------------------

def _check_record_size(
    raw: bytes,
    *,
    maximum_record_bytes: int,
    dataset_file: DatasetFile,
    record_number: int,
) -> None:
    if (
        len(
            raw
        )
        > maximum_record_bytes
    ):
        raise RecordSizeError(
            "Dataset record exceeds "
            "the configured size limit.",
            details={
                "file_id": (
                    dataset_file.file_id
                ),
                "record_number": (
                    record_number
                ),
                "record_size_bytes": (
                    len(
                        raw
                    )
                ),
                "maximum_record_bytes": (
                    maximum_record_bytes
                ),
            },
        )


# ---------------------------------------------------------------
# Text/log streaming
# ---------------------------------------------------------------

def _iter_text_records(
    path: Path,
    dataset_file: DatasetFile,
    *,
    maximum_record_bytes: int,
) -> Iterator[RawRecord]:
    record_number = 0

    try:
        with path.open(
            "rb"
        ) as stream:
            for raw in stream:
                record_number += 1

                _check_record_size(
                    raw,
                    maximum_record_bytes=(
                        maximum_record_bytes
                    ),
                    dataset_file=(
                        dataset_file
                    ),
                    record_number=(
                        record_number
                    ),
                )

                try:
                    text = raw.decode(
                        "utf-8"
                    ).rstrip(
                        "\r\n"
                    )

                except UnicodeDecodeError as exc:
                    raise DatasetRecordError(
                        "Dataset record is "
                        "not valid UTF-8.",
                        details={
                            "file_id": (
                                dataset_file.
                                file_id
                            ),
                            "record_number": (
                                record_number
                            ),
                        },
                    ) from exc

                yield {
                    "record_number": (
                        record_number
                    ),
                    "value": text,
                }

    except DataValidationError:
        raise

    except OSError as exc:
        raise DataValidationError(
            "Dataset text file could "
            "not be read.",
            details={
                "file_id": (
                    dataset_file.file_id
                ),
                "error_type": (
                    type(exc).__name__
                ),
            },
        ) from exc


# ---------------------------------------------------------------
# JSONL streaming
# ---------------------------------------------------------------

def _iter_jsonl_records(
    path: Path,
    dataset_file: DatasetFile,
    *,
    maximum_record_bytes: int,
) -> Iterator[RawRecord]:
    for item in _iter_text_records(
        path,
        dataset_file,
        maximum_record_bytes=(
            maximum_record_bytes
        ),
    ):
        record_number = int(
            item[
                "record_number"
            ]
        )

        text = str(
            item[
                "value"
            ]
        )

        if not text.strip():
            yield {
                "record_number": (
                    record_number
                ),
                "value": None,
                "empty": True,
            }

            continue

        try:
            value = json.loads(
                text
            )

        except json.JSONDecodeError as exc:
            raise DatasetRecordError(
                "JSONL record contains "
                "invalid JSON.",
                details={
                    "file_id": (
                        dataset_file.file_id
                    ),
                    "record_number": (
                        record_number
                    ),
                    "column": (
                        exc.colno
                    ),
                },
            ) from exc

        yield {
            "record_number": (
                record_number
            ),
            "value": value,
        }


# ---------------------------------------------------------------
# CSV streaming
# ---------------------------------------------------------------

def _iter_csv_records(
    path: Path,
    dataset_file: DatasetFile,
    *,
    maximum_record_bytes: int,
) -> Iterator[RawRecord]:
    """
    Stream CSV rows through DictReader.

    The configured record limit is enforced on
    the encoded field content of each parsed row.
    """

    try:
        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:
            reader = csv.DictReader(
                stream
            )

            if reader.fieldnames is None:
                raise DatasetRecordError(
                    "CSV dataset has no "
                    "header.",
                    details={
                        "file_id": (
                            dataset_file.
                            file_id
                        ),
                    },
                )

            for record_number, row in enumerate(
                reader,
                start=2,
            ):
                raw_size = sum(
                    len(
                        (
                            value
                            or ""
                        ).encode(
                            "utf-8"
                        )
                    )
                    for value
                    in row.values()
                )

                if (
                    raw_size
                    > maximum_record_bytes
                ):
                    raise RecordSizeError(
                        "CSV record exceeds "
                        "the configured size "
                        "limit.",
                        details={
                            "file_id": (
                                dataset_file.
                                file_id
                            ),
                            "record_number": (
                                record_number
                            ),
                            "record_size_bytes": (
                                raw_size
                            ),
                            "maximum_record_bytes": (
                                maximum_record_bytes
                            ),
                        },
                    )

                yield {
                    "record_number": (
                        record_number
                    ),
                    "value": dict(
                        row
                    ),
                }

    except DataValidationError:
        raise

    except (
        OSError,
        UnicodeError,
        csv.Error,
    ) as exc:
        raise DataValidationError(
            "CSV dataset could not "
            "be parsed.",
            details={
                "file_id": (
                    dataset_file.file_id
                ),
                "error_type": (
                    type(exc).__name__
                ),
            },
        ) from exc


# ---------------------------------------------------------------
# JSON records
# ---------------------------------------------------------------

def _iter_json_records(
    path: Path,
    dataset_file: DatasetFile,
    *,
    maximum_record_bytes: int,
) -> Iterator[RawRecord]:
    """
    Read one JSON document.

    JSON arrays produce one record per element.
    Other JSON values produce a single record.

    Unlike JSONL, a standard JSON document may
    require loading the complete document.
    """

    try:
        size = int(
            path.stat().st_size
        )

        if (
            size
            > maximum_record_bytes
        ):
            raise RecordSizeError(
                "JSON document exceeds "
                "the configured record "
                "size limit.",
                details={
                    "file_id": (
                        dataset_file.file_id
                    ),
                    "record_size_bytes": (
                        size
                    ),
                    "maximum_record_bytes": (
                        maximum_record_bytes
                    ),
                    "suggestion": (
                        "Use JSONL for large "
                        "datasets."
                    ),
                },
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as stream:
            value = json.load(
                stream
            )

    except DataValidationError:
        raise

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise DataValidationError(
            "JSON dataset could not "
            "be parsed.",
            details={
                "file_id": (
                    dataset_file.file_id
                ),
                "error_type": (
                    type(exc).__name__
                ),
            },
        ) from exc

    if isinstance(
        value,
        list,
    ):
        for index, item in enumerate(
            value,
            start=1,
        ):
            yield {
                "record_number": index,
                "value": item,
            }

    else:
        yield {
            "record_number": 1,
            "value": value,
        }


# ---------------------------------------------------------------
# Record iterator selection
# ---------------------------------------------------------------

def iter_dataset_records(
    path: str | Path,
    dataset_file: DatasetFile,
    *,
    dataset_format: str,
    maximum_record_bytes: int,
) -> Iterator[RawRecord]:
    """
    Yield raw records from one supported dataset
    file.

    Supported here:

      text
      log
      jsonl
      json
      csv

    Parquet requires an explicit tabular adapter
    and is not silently treated as another
    format.
    """

    dataset_file.finalize()

    source = Path(
        path
    ).expanduser()

    if (
        dataset_format
        in {
            DatasetFormat.TEXT.value,
            DatasetFormat.LOG.value,
        }
    ):
        yield from _iter_text_records(
            source,
            dataset_file,
            maximum_record_bytes=(
                maximum_record_bytes
            ),
        )

        return

    if (
        dataset_format
        == DatasetFormat.JSONL.value
    ):
        yield from _iter_jsonl_records(
            source,
            dataset_file,
            maximum_record_bytes=(
                maximum_record_bytes
            ),
        )

        return

    if (
        dataset_format
        == DatasetFormat.JSON.value
    ):
        yield from _iter_json_records(
            source,
            dataset_file,
            maximum_record_bytes=(
                maximum_record_bytes
            ),
        )

        return

    if (
        dataset_format
        == DatasetFormat.CSV.value
    ):
        yield from _iter_csv_records(
            source,
            dataset_file,
            maximum_record_bytes=(
                maximum_record_bytes
            ),
        )

        return

    raise DatasetFormatError(
        "Dataset format does not have "
        "a validation reader.",
        details={
            "format": (
                dataset_format
            ),
        },
    )


# ---------------------------------------------------------------
# Default record validation
# ---------------------------------------------------------------

def default_record_validator(
    value: Any,
    dataset_file: DatasetFile,
    record_number: int,
) -> Optional[
    ValidationIssue
]:
    """
    Generic validation shared across formats.

    Dataset-specific required fields and label
    rules should be supplied through an explicit
    RecordValidator.
    """

    if value is None:
        return validation_issue(
            code="empty_record",
            message=(
                "Dataset record is empty."
            ),
            severity=(
                ValidationSeverity.
                ERROR.value
            ),
            dataset_file=(
                dataset_file
            ),
            record_number=(
                record_number
            ),
        )

    if isinstance(
        value,
        str,
    ) and not value.strip():
        return validation_issue(
            code="blank_record",
            message=(
                "Dataset record contains "
                "no text."
            ),
            severity=(
                ValidationSeverity.
                ERROR.value
            ),
            dataset_file=(
                dataset_file
            ),
            record_number=(
                record_number
            ),
        )

    if isinstance(
        value,
        Mapping,
    ) and not value:
        return validation_issue(
            code="empty_object",
            message=(
                "Dataset record is an "
                "empty object."
            ),
            severity=(
                ValidationSeverity.
                WARNING.value
            ),
            dataset_file=(
                dataset_file
            ),
            record_number=(
                record_number
            ),
        )

    return None


# ---------------------------------------------------------------
# Dataset validator
# ---------------------------------------------------------------

class DatasetValidator:
    """
    Validate dataset manifests and raw dataset
    files without altering their contents.
    """

    def __init__(
        self,
        *,
        config: Optional[
            DataConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or DataConfig()
        )

        self.config.validate()

    # ---------------------------------------------------------
    # Raw directory
    # ---------------------------------------------------------

    @property
    def raw_directory(
        self,
    ) -> Path:
        return Path(
            self.config.
            paths.resolved()[
                "raw"
            ]
        )

    # ---------------------------------------------------------
    # Manifest validation
    # ---------------------------------------------------------

    def validate_manifest(
        self,
        manifest: DatasetManifest,
    ) -> List[
        ValidationIssue
    ]:
        issues: List[
            ValidationIssue
        ] = []

        try:
            manifest.finalize()

        except Exception as exc:
            return [
                validation_issue(
                    code=(
                        "invalid_manifest"
                    ),
                    message=(
                        "Dataset manifest "
                        "failed schema "
                        "validation."
                    ),
                    severity=(
                        ValidationSeverity.
                            ERROR.value
                    ),
                    attributes={
                        "error_type": (
                            type(exc).__name__
                        ),
                    },
                )
            ]

        if not self.config.format_allowed(
            manifest.format
        ):
            issues.append(
                validation_issue(
                    code=(
                        "format_not_allowed"
                    ),
                    message=(
                        "Dataset format is "
                        "not allowed by "
                        "configuration."
                    ),
                    severity=(
                        ValidationSeverity.
                        ERROR.value
                    ),
                    attributes={
                        "format": (
                            manifest.format
                        ),
                    },
                )
            )

        if (
            not manifest.files
            and not self.config.
            validation.
            allow_empty_dataset
        ):
            issues.append(
                validation_issue(
                    code=(
                        "manifest_has_no_files"
                    ),
                    message=(
                        "Dataset manifest "
                        "declares no files."
                    ),
                    severity=(
                        ValidationSeverity.
                        ERROR.value
                    ),
                )
            )

        paths = set()

        for item in manifest.files:
            if item.path in paths:
                issues.append(
                    validation_issue(
                        code=(
                            "duplicate_file_path"
                        ),
                        message=(
                            "Dataset manifest "
                            "contains a "
                            "duplicate file "
                            "path."
                        ),
                        severity=(
                            ValidationSeverity.
                            ERROR.value
                        ),
                        dataset_file=item,
                    )
                )

            paths.add(
                item.path
            )

            if (
                self.config.
                integrity.
                require_checksums
                and not item.checksum
            ):
                issues.append(
                    validation_issue(
                        code=(
                            "checksum_missing"
                        ),
                        message=(
                            "Dataset file "
                            "does not declare "
                            "a required "
                            "checksum."
                        ),
                        severity=(
                            ValidationSeverity.
                            ERROR.value
                        ),
                        dataset_file=item,
                    )
                )

        return issues

    # ---------------------------------------------------------
    # Integrity validation
    # ---------------------------------------------------------

    def validate_integrity(
        self,
        manifest: DatasetManifest,
    ) -> tuple[
        List[
            FileIntegrityResult
        ],
        List[
            ValidationIssue
        ],
    ]:
        try:
            results = (
                verify_manifest_files(
                    manifest,
                    self.raw_directory,
                    config=(
                        self.config.
                        integrity
                    ),
                    raise_on_mismatch=False,
                )
            )

        except Exception as exc:
            return (
                [],
                [
                    validation_issue(
                        code=(
                            "integrity_check_failed"
                        ),
                        message=(
                            "Dataset file "
                            "integrity could "
                            "not be checked."
                        ),
                        severity=(
                            ValidationSeverity.
                            ERROR.value
                        ),
                        attributes={
                            "error_type": (
                                type(exc).__name__
                            ),
                        },
                    )
                ],
            )

        issues: List[
            ValidationIssue
        ] = []

        for result in results:
            if (
                result.required
                and not result.present
            ):
                issues.append(
                    ValidationIssue(
                        code=(
                            "required_file_missing"
                        ),
                        message=(
                            "Required dataset "
                            "file is missing."
                        ),
                        severity=(
                            ValidationSeverity.
                            ERROR.value
                        ),
                        file_id=(
                            result.file_id
                        ),
                        attributes={
                            "path": (
                                result.path
                            ),
                        },
                    ).finalize()
                )

                continue

            if (
                result.present
                and not result.
                size_verified
            ):
                issues.append(
                    ValidationIssue(
                        code=(
                            "file_size_mismatch"
                        ),
                        message=(
                            "Dataset file "
                            "size does not "
                            "match the "
                            "manifest."
                        ),
                        severity=(
                            ValidationSeverity.
                            ERROR.value
                        ),
                        file_id=(
                            result.file_id
                        ),
                        attributes={
                            "path": (
                                result.path
                            ),
                            "size_bytes": (
                                result.
                                size_bytes
                            ),
                        },
                    ).finalize()
                )

            if (
                result.present
                and not result.
                checksum_verified
            ):
                issues.append(
                    ValidationIssue(
                        code=(
                            "checksum_mismatch"
                        ),
                        message=(
                            "Dataset file "
                            "checksum was not "
                            "verified."
                        ),
                        severity=(
                            ValidationSeverity.
                            ERROR.value
                        ),
                        file_id=(
                            result.file_id
                        ),
                        attributes={
                            "path": (
                                result.path
                            ),
                            "algorithm": (
                                result.algorithm
                            ),
                        },
                    ).finalize()
                )

        if (
            not manifest_integrity_ok(
                results
            )
            and not issues
        ):
            issues.append(
                validation_issue(
                    code=(
                        "manifest_integrity_failed"
                    ),
                    message=(
                        "Dataset manifest "
                        "integrity validation "
                        "failed."
                    ),
                    severity=(
                        ValidationSeverity.
                        ERROR.value
                    ),
                )
            )

        return (
            results,
            issues,
        )

    # ---------------------------------------------------------
    # Complete validation
    # ---------------------------------------------------------

    def validate(
        self,
        manifest: DatasetManifest,
        *,
        record_validator: Optional[
            RecordValidator
        ] = None,
    ) -> ValidationResult:
        """
        Validate manifest, file integrity, and raw
        records.

        The caller may supply one additional
        dataset-specific record validator.
        """

        manifest_issues = (
            self.validate_manifest(
                manifest
            )
        )

        issues = list(
            manifest_issues
        )

        records_seen = 0
        records_valid = 0
        records_invalid = 0

        maximum_issues = (
            self.config.
            validation.
            maximum_validation_issues
        )

        if any(
            issue.severity
            == ValidationSeverity.
            ERROR.value
            for issue
            in manifest_issues
        ):
            return ValidationResult(
                dataset_id=(
                    manifest.dataset_id
                    if getattr(
                        manifest,
                        "dataset_id",
                        None,
                    )
                    else "unknown"
                ),
                valid=False,
                issues=issues[
                    :maximum_issues
                ],
            ).finalize()

        (
            _,
            integrity_issues,
        ) = self.validate_integrity(
            manifest
        )

        issues.extend(
            integrity_issues
        )

        if any(
            issue.severity
            == ValidationSeverity.
            ERROR.value
            for issue
            in integrity_issues
        ):
            return ValidationResult(
                dataset_id=(
                    manifest.dataset_id
                ),
                valid=False,
                issues=issues[
                    :maximum_issues
                ],
            ).finalize()

        validators: List[
            RecordValidator
        ] = [
            default_record_validator
        ]

        if record_validator is not None:
            validators.append(
                record_validator
            )

        for dataset_file in (
            manifest.files
        ):
            path = resolve_dataset_file(
                self.raw_directory,
                dataset_file,
            )

            if (
                not path.exists()
                and not dataset_file.required
            ):
                continue

            if (
                path.stat().st_size == 0
            ):
                severity = (
                    ValidationSeverity.
                    WARNING.value
                    if self.config.
                    validation.
                    allow_empty_files
                    else (
                        ValidationSeverity.
                        ERROR.value
                    )
                )

                issues.append(
                    validation_issue(
                        code="empty_file",
                        message=(
                            "Dataset file "
                            "is empty."
                        ),
                        severity=severity,
                        dataset_file=(
                            dataset_file
                        ),
                    )
                )

                if (
                    not self.config.
                    validation.
                    allow_empty_files
                ):
                    continue

            try:
                iterator = (
                    iter_dataset_records(
                        path,
                        dataset_file,
                        dataset_format=(
                            manifest.format
                        ),
                        maximum_record_bytes=(
                            self.config.
                            validation.
                            maximum_record_bytes
                        ),
                    )
                )

                for record in iterator:
                    records_seen += 1

                    record_number = int(
                        record[
                            "record_number"
                        ]
                    )

                    value = record.get(
                        "value"
                    )

                    record_issues: List[
                        ValidationIssue
                    ] = []

                    for validator in validators:
                        issue = validator(
                            value,
                            dataset_file,
                            record_number,
                        )

                        if issue is not None:
                            record_issues.append(
                                issue.finalize()
                            )

                    has_error = any(
                        issue.severity
                        == ValidationSeverity.
                        ERROR.value
                        for issue
                        in record_issues
                    )

                    if has_error:
                        records_invalid += 1

                    else:
                        records_valid += 1

                    remaining = max(
                        0,
                        maximum_issues
                        - len(
                            issues
                        ),
                    )

                    if remaining:
                        issues.extend(
                            record_issues[
                                :remaining
                            ]
                        )

                    if (
                        has_error
                        and self.config.
                        validation.
                        fail_on_invalid_record
                    ):
                        raise DatasetRecordError(
                            "Dataset contains "
                            "an invalid record.",
                            details={
                                "file_id": (
                                    dataset_file.
                                    file_id
                                ),
                                "record_number": (
                                    record_number
                                ),
                            },
                        )

            except DataValidationError as exc:
                records_invalid += 1

                if (
                    len(
                        issues
                    )
                    < maximum_issues
                ):
                    issues.append(
                        validation_issue(
                            code=(
                                "record_read_failed"
                            ),
                            message=(
                                "Dataset record "
                                "stream could "
                                "not be "
                                "validated."
                            ),
                            severity=(
                                ValidationSeverity.
                                ERROR.value
                            ),
                            dataset_file=(
                                dataset_file
                            ),
                            attributes={
                                "error_type": (
                                    type(exc).__name__
                                ),
                            },
                        )
                    )

                if (
                    self.config.
                    validation.
                    fail_on_invalid_record
                ):
                    raise

        if (
            records_seen == 0
            and not self.config.
            validation.
            allow_empty_dataset
        ):
            if (
                len(
                    issues
                )
                < maximum_issues
            ):
                issues.append(
                    validation_issue(
                        code=(
                            "dataset_has_no_records"
                        ),
                        message=(
                            "Dataset contains "
                            "no records."
                        ),
                        severity=(
                            ValidationSeverity.
                            ERROR.value
                        ),
                    )
                )

        valid = not any(
            issue.severity
            == ValidationSeverity.
            ERROR.value
            for issue
            in issues
        )

        return ValidationResult(
            dataset_id=(
                manifest.dataset_id
            ),
            valid=valid,
            records_seen=(
                records_seen
            ),
            records_valid=(
                records_valid
            ),
            records_invalid=(
                records_invalid
            ),
            issues=issues[
                :maximum_issues
            ],
        ).finalize()

    # ---------------------------------------------------------
    # Require valid
    # ---------------------------------------------------------

    def require_valid(
        self,
        manifest: DatasetManifest,
        *,
        record_validator: Optional[
            RecordValidator
        ] = None,
    ) -> ValidationResult:
        result = self.validate(
            manifest,
            record_validator=(
                record_validator
            ),
        )

        if not result.valid:
            raise DatasetEmptyError(
                "Dataset validation "
                "failed.",
                details={
                    "dataset_id": (
                        result.dataset_id
                    ),
                    "error_count": (
                        len(
                            result.errors
                        )
                    ),
                    "warning_count": (
                        len(
                            result.warnings
                        )
                    ),
                },
            )

        return result

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "raw_directory": str(
                self.raw_directory
            ),
            "allowed_formats": list(
                self.config.
                allowed_formats
            ),
            "maximum_record_bytes": (
                self.config.
                validation.
                maximum_record_bytes
            ),
            "maximum_validation_issues": (
                self.config.
                validation.
                maximum_validation_issues
            ),
            "allow_empty_dataset": (
                self.config.
                validation.
                allow_empty_dataset
            ),
            "allow_empty_files": (
                self.config.
                validation.
                allow_empty_files
            ),
            "mutates_source_data": (
                False
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "RecordValidator",
    "RawRecord",

    "validation_issue",

    "iter_dataset_records",
    "default_record_validator",

    "DatasetValidator",
]
