from __future__ import annotations

"""
Training-corpus preparation for OSHA.

Pipeline:

    validated raw files
        -> parse records
        -> normalize
        -> deduplicate
        -> split
        -> write JSONL artifacts

This module prepares datasets only. It does not
train or evaluate models.
"""

import json
import os
import shutil
import tempfile

from pathlib import Path

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

from src.core.serialization import (
    to_json_safe,
)

from src.data.config import (
    DataConfig,
)

from src.data.errors import (
    DataNormalizationError,
    DataPreparationError,
    PreparedDatasetConflictError,
    UnsupportedRecordError,
)

from src.data.manifests import (
    resolve_dataset_file,
)

from src.data.normalize import (
    DatasetNormalizer,
    NormalizerRegistry,
)

from src.data.schema import (
    DataState,
    DatasetFile,
    DatasetManifest,
    DatasetRecord,
    DatasetSplit,
    PreparationResult,
    SplitName,
)

from src.data.split import (
    DatasetSplitter,
    SplitResult,
)

from src.data.validate import (
    DatasetValidator,
    RecordValidator,
    iter_dataset_records,
)


# ---------------------------------------------------------------
# Safe artifact name
# ---------------------------------------------------------------

def safe_artifact_name(
    value: str,
) -> str:
    """
    Convert a dataset identifier or version into
    a bounded filesystem-safe name.
    """

    output: List[str] = []

    for character in str(
        value
    ):
        if (
            character.isalnum()
            or character
            in {
                "-",
                "_",
                ".",
            }
        ):
            output.append(
                character
            )

        else:
            output.append(
                "_"
            )

    normalized = "".join(
        output
    ).strip(
        "._"
    )

    if not normalized:
        raise DataPreparationError(
            "Dataset identity cannot "
            "produce a safe artifact name."
        )

    return normalized


# ---------------------------------------------------------------
# Prepared dataset directory
# ---------------------------------------------------------------

def prepared_dataset_directory(
    prepared_root: str | Path,
    manifest: DatasetManifest,
) -> Path:
    manifest.finalize()

    return (
        Path(
            prepared_root
        ).expanduser().resolve()
        / safe_artifact_name(
            manifest.dataset_id
        )
        / safe_artifact_name(
            manifest.version
        )
    )


# ---------------------------------------------------------------
# Atomic JSONL writer
# ---------------------------------------------------------------

def write_jsonl(
    path: str | Path,
    records: Iterable[
        DatasetRecord
    ],
) -> int:
    """
    Write canonical records as UTF-8 JSONL.

    Returns the number of records written.
    """

    destination = Path(
        path
    )

    try:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as exc:
        raise DataPreparationError(
            "Prepared-data directory "
            "could not be created.",
            details={
                "error_type": (
                    type(exc).__name__
                ),
            },
        ) from exc

    temporary = destination.with_name(
        destination.name
        + ".writing"
    )

    count = 0

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            for record in records:
                record.finalize()

                stream.write(
                    json.dumps(
                        record.to_dict(),
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":",
                        ),
                    )
                )

                stream.write(
                    "\n"
                )

                count += 1

            stream.flush()

            os.fsync(
                stream.fileno()
            )

        temporary.replace(
            destination
        )

    except (
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        try:
            temporary.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise DataPreparationError(
            "Prepared JSONL artifact "
            "could not be written.",
            details={
                "error_type": (
                    type(exc).__name__
                ),
            },
        ) from exc

    return count


# ---------------------------------------------------------------
# Rejected-record writer
# ---------------------------------------------------------------

def write_rejected_jsonl(
    path: str | Path,
    rejected: Iterable[
        Dict[
            str,
            Any,
        ]
    ],
) -> int:
    """
    Write bounded rejected-record metadata.

    Raw exception messages are not included.
    """

    destination = Path(
        path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        destination.name
        + ".writing"
    )

    count = 0

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            for item in rejected:
                stream.write(
                    json.dumps(
                        to_json_safe(
                            item
                        ),
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":",
                        ),
                    )
                )

                stream.write(
                    "\n"
                )

                count += 1

            stream.flush()

            os.fsync(
                stream.fileno()
            )

        temporary.replace(
            destination
        )

    except (
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        try:
            temporary.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise DataPreparationError(
            "Rejected-record artifact "
            "could not be written.",
            details={
                "error_type": (
                    type(exc).__name__
                ),
            },
        ) from exc

    return count


# ---------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------

def deduplicate_records(
    records: Iterable[
        DatasetRecord
    ],
) -> Tuple[
    List[
        DatasetRecord
    ],
    int,
]:
    """
    Deduplicate canonical records by record_id.

    The first occurrence is preserved.
    """

    output: List[
        DatasetRecord
    ] = []

    seen = set()

    duplicate_count = 0

    for record in records:
        record.finalize()

        if not record.record_id:
            raise DataPreparationError(
                "Canonical record has no "
                "record_id."
            )

        if record.record_id in seen:
            duplicate_count += 1

            continue

        seen.add(
            record.record_id
        )

        output.append(
            record
        )

    return (
        output,
        duplicate_count,
    )


# ---------------------------------------------------------------
# Corpus preparer
# ---------------------------------------------------------------

class CorpusPreparer:
    """
    Prepare one validated dataset version for the
    training layer.

    This implementation materializes canonical
    records in memory before splitting. That
    keeps splitting deterministic. A future
    large-corpus backend can provide external
    shuffle and partitioning through the same
    output contracts.
    """

    def __init__(
        self,
        *,
        config: Optional[
            DataConfig
        ] = None,
        validator: Optional[
            DatasetValidator
        ] = None,
        normalizer: Optional[
            DatasetNormalizer
        ] = None,
        splitter: Optional[
            DatasetSplitter
        ] = None,
        normalizers: Optional[
            NormalizerRegistry
        ] = None,
    ) -> None:
        self.config = (
            config
            or DataConfig()
        )

        self.config.validate()

        self.validator = (
            validator
            or DatasetValidator(
                config=self.config
            )
        )

        self.normalizer = (
            normalizer
            or DatasetNormalizer(
                registry=normalizers
            )
        )

        self.splitter = (
            splitter
            or DatasetSplitter(
                config=(
                    self.config.split
                )
            )
        )

    # ---------------------------------------------------------
    # Paths
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

    @property
    def prepared_directory(
        self,
    ) -> Path:
        return Path(
            self.config.
            paths.resolved()[
                "prepared"
            ]
        )

    @property
    def rejected_directory(
        self,
    ) -> Path:
        return Path(
            self.config.
            paths.resolved()[
                "rejected"
            ]
        )

    # ---------------------------------------------------------
    # Normalize all source files
    # ---------------------------------------------------------

    def _normalize_manifest(
        self,
        manifest: DatasetManifest,
    ) -> Tuple[
        List[
            DatasetRecord
        ],
        List[
            Dict[
                str,
                Any,
            ]
        ],
        int,
    ]:
        records: List[
            DatasetRecord
        ] = []

        rejected: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        source_record_count = 0

        for dataset_file in manifest.files:
            path = resolve_dataset_file(
                self.raw_directory,
                dataset_file,
            )

            if (
                not path.exists()
                and not dataset_file.required
            ):
                continue

            iterator = iter_dataset_records(
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

            for raw_record in iterator:
                source_record_count += 1

                record_number = int(
                    raw_record[
                        "record_number"
                    ]
                )

                value = raw_record.get(
                    "value"
                )

                try:
                    record = (
                        self.normalizer.
                        normalize_one(
                            value,
                            manifest=manifest,
                            dataset_file=(
                                dataset_file
                            ),
                            record_number=(
                                record_number
                            ),
                        )
                    )

                    records.append(
                        record
                    )

                except (
                    DataNormalizationError,
                    UnsupportedRecordError,
                ) as exc:
                    rejected.append(
                        {
                            "dataset_id": (
                                manifest.
                                dataset_id
                            ),
                            "dataset_version": (
                                manifest.version
                            ),
                            "file_id": (
                                dataset_file.
                                file_id
                            ),
                            "source_file": (
                                dataset_file.path
                            ),
                            "record_number": (
                                record_number
                            ),
                            "error_type": (
                                type(exc).__name__
                            ),
                        }
                    )

                    if not (
                        self.config.
                        validation.
                        reject_invalid_records
                    ):
                        raise

        return (
            records,
            rejected,
            source_record_count,
        )

    # ---------------------------------------------------------
    # Write split
    # ---------------------------------------------------------

    @staticmethod
    def _write_split(
        staging_directory: Path,
        *,
        dataset_id: str,
        split_name: str,
        records: Iterable[
            DatasetRecord
        ],
    ) -> DatasetSplit:
        output = (
            staging_directory
            / (
                split_name
                + ".jsonl"
            )
        )

        values = list(
            records
        )

        write_jsonl(
            output,
            values,
        )

        return DatasetSplit(
            dataset_id=(
                dataset_id
            ),
            name=split_name,
            record_ids=[
                record.record_id
                for record
                in values
                if record.record_id
                is not None
            ],
            output_path=(
                output.name
            ),
        ).finalize()

    # ---------------------------------------------------------
    # Write corpus metadata
    # ---------------------------------------------------------

    @staticmethod
    def _write_metadata(
        staging_directory: Path,
        *,
        manifest: DatasetManifest,
        source_record_count: int,
        prepared_record_count: int,
        rejected_record_count: int,
        duplicate_record_count: int,
        splits: List[
            DatasetSplit
        ],
    ) -> None:
        path = (
            staging_directory
            / "dataset.json"
        )

        value = {
            "dataset_id": (
                manifest.dataset_id
            ),
            "name": manifest.name,
            "version": (
                manifest.version
            ),
            "format": (
                manifest.format
            ),
            "purpose": (
                manifest.purpose
            ),
            "source_record_count": (
                source_record_count
            ),
            "prepared_record_count": (
                prepared_record_count
            ),
            "rejected_record_count": (
                rejected_record_count
            ),
            "duplicate_record_count": (
                duplicate_record_count
            ),
            "splits": [
                split.to_dict()
                for split
                in splits
            ],
        }

        temporary = path.with_name(
            path.name
            + ".writing"
        )

        try:
            temporary.write_text(
                json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            temporary.replace(
                path
            )

        except (
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            try:
                temporary.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

            raise DataPreparationError(
                "Prepared dataset "
                "metadata could not be "
                "written.",
                details={
                    "error_type": (
                        type(exc).__name__
                    ),
                },
            ) from exc

    # ---------------------------------------------------------
    # Prepare
    # ---------------------------------------------------------

    def prepare(
        self,
        manifest: DatasetManifest,
        *,
        record_validator: Optional[
            RecordValidator
        ] = None,
    ) -> PreparationResult:
        manifest.finalize()

        if not self.config.enabled:
            raise DataPreparationError(
                "Data pipeline is "
                "disabled."
            )

        if (
            self.config.
            preparation.
            output_format
            != "jsonl"
        ):
            raise DataPreparationError(
                "The current corpus "
                "writer supports JSONL "
                "output only.",
                details={
                    "configured_format": (
                        self.config.
                        preparation.
                        output_format
                    ),
                },
            )

        # -----------------------------------------------------
        # Validate before transformation
        # -----------------------------------------------------

        validation = (
            self.validator.
            require_valid(
                manifest,
                record_validator=(
                    record_validator
                ),
            )
        )

        # -----------------------------------------------------
        # Normalize
        # -----------------------------------------------------

        (
            records,
            rejected,
            source_record_count,
        ) = self._normalize_manifest(
            manifest
        )

        if not records:
            raise DataPreparationError(
                "Dataset produced no "
                "canonical records.",
                details={
                    "dataset_id": (
                        manifest.dataset_id
                    ),
                    "rejected_count": (
                        len(
                            rejected
                        )
                    ),
                },
            )

        # -----------------------------------------------------
        # Deduplicate
        # -----------------------------------------------------

        duplicate_count = 0

        if (
            self.config.
            preparation.
            deduplicate_records
        ):
            (
                records,
                duplicate_count,
            ) = deduplicate_records(
                records
            )

        # -----------------------------------------------------
        # Destination
        # -----------------------------------------------------

        destination = (
            prepared_dataset_directory(
                self.prepared_directory,
                manifest,
            )
        )

        if (
            destination.exists()
            and not self.config.
            preparation.
            overwrite_existing
        ):
            raise (
                PreparedDatasetConflictError(
                    "Prepared dataset "
                    "already exists.",
                    details={
                        "dataset_id": (
                            manifest.
                            dataset_id
                        ),
                        "version": (
                            manifest.version
                        ),
                        "destination": (
                            str(
                                destination
                            )
                        ),
                    },
                )
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        staging_directory = Path(
            tempfile.mkdtemp(
                prefix=(
                    "."
                    + safe_artifact_name(
                        manifest.version
                    )
                    + "-"
                ),
                suffix=".preparing",
                dir=str(
                    destination.parent
                ),
            )
        )

        splits: List[
            DatasetSplit
        ] = []

        try:
            # -------------------------------------------------
            # Split and write
            # -------------------------------------------------

            if (
                self.config.
                preparation.
                create_splits
            ):
                split_result = (
                    self.splitter.split(
                        records
                    )
                )

                splits = [
                    self._write_split(
                        staging_directory,
                        dataset_id=(
                            manifest.
                            dataset_id
                        ),
                        split_name=(
                            SplitName.
                            TRAIN.value
                        ),
                        records=(
                            split_result.train
                        ),
                    ),

                    self._write_split(
                        staging_directory,
                        dataset_id=(
                            manifest.
                            dataset_id
                        ),
                        split_name=(
                            SplitName.
                            VALIDATION.value
                        ),
                        records=(
                            split_result.
                            validation
                        ),
                    ),

                    self._write_split(
                        staging_directory,
                        dataset_id=(
                            manifest.
                            dataset_id
                        ),
                        split_name=(
                            SplitName.
                            TEST.value
                        ),
                        records=(
                            split_result.test
                        ),
                    ),
                ]

            else:
                splits = [
                    self._write_split(
                        staging_directory,
                        dataset_id=(
                            manifest.
                            dataset_id
                        ),
                        split_name=(
                            SplitName.
                            TRAIN.value
                        ),
                        records=records,
                    )
                ]

            # -------------------------------------------------
            # Rejected metadata
            # -------------------------------------------------

            if (
                rejected
                and self.config.
                preparation.
                preserve_rejected_records
            ):
                write_rejected_jsonl(
                    (
                        staging_directory
                        / "rejected.jsonl"
                    ),
                    rejected,
                )

            # -------------------------------------------------
            # Dataset metadata
            # -------------------------------------------------

            self._write_metadata(
                staging_directory,
                manifest=manifest,
                source_record_count=(
                    source_record_count
                ),
                prepared_record_count=(
                    len(
                        records
                    )
                ),
                rejected_record_count=(
                    len(
                        rejected
                    )
                ),
                duplicate_record_count=(
                    duplicate_count
                ),
                splits=splits,
            )

            # -------------------------------------------------
            # Publish prepared directory
            # -------------------------------------------------

            if destination.exists():
                shutil.rmtree(
                    destination
                )

            staging_directory.replace(
                destination
            )

        except Exception:
            try:
                shutil.rmtree(
                    staging_directory,
                    ignore_errors=True,
                )
            finally:
                pass

            raise

        # -----------------------------------------------------
        # Final paths
        # -----------------------------------------------------

        final_splits: List[
            DatasetSplit
        ] = []

        for split in splits:
            final_splits.append(
                DatasetSplit(
                    dataset_id=(
                        split.dataset_id
                    ),
                    name=split.name,
                    record_ids=list(
                        split.record_ids
                    ),
                    output_path=str(
                        destination
                        / str(
                            split.
                            output_path
                        )
                    ),
                    attributes={
                        **split.attributes,
                        "output_format": (
                            "jsonl"
                        ),
                    },
                ).finalize()
            )

        return PreparationResult(
            dataset_id=(
                manifest.dataset_id
            ),
            success=True,
            state=(
                DataState.PREPARED.value
            ),
            output_directory=str(
                destination
            ),
            splits=final_splits,
            source_record_count=(
                source_record_count
            ),
            prepared_record_count=len(
                records
            ),
            rejected_record_count=len(
                rejected
            ),
            reference_ids=[
                dataset_file.file_id
                for dataset_file
                in manifest.files
                if dataset_file.file_id
                is not None
            ],
            attributes={
                "dataset_version": (
                    manifest.version
                ),
                "validation_records_seen": (
                    validation.
                    records_seen
                ),
                "validation_records_valid": (
                    validation.
                    records_valid
                ),
                "duplicate_record_count": (
                    duplicate_count
                ),
                "output_format": (
                    "jsonl"
                ),
                "training_performed": (
                    False
                ),
            },
        ).finalize()

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
            "prepared_directory": str(
                self.prepared_directory
            ),
            "rejected_directory": str(
                self.rejected_directory
            ),
            "output_format": (
                self.config.
                preparation.
                output_format
            ),
            "deduplicate_records": (
                self.config.
                preparation.
                deduplicate_records
            ),
            "create_splits": (
                self.config.
                preparation.
                create_splits
            ),
            "overwrite_existing": (
                self.config.
                preparation.
                overwrite_existing
            ),
            "preserve_rejected_records": (
                self.config.
                preparation.
                preserve_rejected_records
            ),
            "training_performed": (
                False
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "safe_artifact_name",
    "prepared_dataset_directory",

    "write_jsonl",
    "write_rejected_jsonl",

    "deduplicate_records",

    "CorpusPreparer",
]
