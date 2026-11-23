from __future__ import annotations

"""
Configuration for OSHA dataset acquisition,
validation, normalization, splitting, and corpus
preparation.
"""

import os

from dataclasses import (
    dataclass,
    field,
)

from pathlib import Path

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Tuple,
)

from src.core.validation import (
    boolean,
    integer,
    number,
    unique_strings,
)

from src.data.schema import (
    DatasetFormat,
)


# ---------------------------------------------------------------
# Environment prefix
# ---------------------------------------------------------------

DEFAULT_ENV_PREFIX = (
    "OSHA_DATA_"
)


# ---------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class DataPathConfig:
    """
    Filesystem locations used by the data
    pipeline.

    These paths describe dataset artifacts, not
    application database storage.
    """

    root_directory: str = "data"

    raw_directory: str = "data/raw"

    prepared_directory: str = (
        "data/prepared"
    )

    cache_directory: str = "data/cache"

    manifest_directory: str = (
        "data/manifests"
    )

    rejected_directory: str = (
        "data/rejected"
    )

    def validate(
        self,
    ) -> None:
        for name, value in (
            (
                "root_directory",
                self.root_directory,
            ),
            (
                "raw_directory",
                self.raw_directory,
            ),
            (
                "prepared_directory",
                self.prepared_directory,
            ),
            (
                "cache_directory",
                self.cache_directory,
            ),
            (
                "manifest_directory",
                self.manifest_directory,
            ),
            (
                "rejected_directory",
                self.rejected_directory,
            ),
        ):
            if not str(
                value
            ).strip():
                raise ValueError(
                    f"{name} cannot be empty"
                )

    def resolved(
        self,
    ) -> Dict[str, str]:
        """
        Return normalized absolute paths without
        creating directories.
        """

        self.validate()

        return {
            "root": str(
                Path(
                    self.root_directory
                ).expanduser().resolve()
            ),

            "raw": str(
                Path(
                    self.raw_directory
                ).expanduser().resolve()
            ),

            "prepared": str(
                Path(
                    self.prepared_directory
                ).expanduser().resolve()
            ),

            "cache": str(
                Path(
                    self.cache_directory
                ).expanduser().resolve()
            ),

            "manifests": str(
                Path(
                    self.manifest_directory
                ).expanduser().resolve()
            ),

            "rejected": str(
                Path(
                    self.rejected_directory
                ).expanduser().resolve()
            ),
        }


# ---------------------------------------------------------------
# Integrity policy
# ---------------------------------------------------------------

@dataclass(frozen=True)
class IntegrityConfig:
    """
    Dataset/file integrity policy.
    """

    require_checksums: bool = True

    checksum_algorithm: str = "sha256"

    verify_after_fetch: bool = True

    verify_before_prepare: bool = True

    maximum_file_size_bytes: int = (
        10
        * 1024
        * 1024
        * 1024
    )

    def validate(
        self,
    ) -> None:
        if (
            self.checksum_algorithm
            not in {
                "sha256",
                "sha512",
            }
        ):
            raise ValueError(
                "checksum_algorithm must "
                "be sha256 or sha512"
            )

        integer(
            self.maximum_file_size_bytes,
            name=(
                "maximum_file_size_bytes"
            ),
            minimum=1,
        )


# ---------------------------------------------------------------
# Validation policy
# ---------------------------------------------------------------

@dataclass(frozen=True)
class DataValidationConfig:
    """
    Generic validation behavior.

    Dataset-specific schema rules still belong in
    validate.py or caller-provided validators.
    """

    reject_invalid_records: bool = True

    fail_on_invalid_record: bool = False

    allow_empty_dataset: bool = False

    allow_empty_files: bool = False

    maximum_validation_issues: int = 1000

    maximum_record_bytes: int = (
        4
        * 1024
        * 1024
    )

    def validate(
        self,
    ) -> None:
        integer(
            self.maximum_validation_issues,
            name=(
                "maximum_validation_issues"
            ),
            minimum=1,
        )

        integer(
            self.maximum_record_bytes,
            name="maximum_record_bytes",
            minimum=1,
        )


# ---------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class SplitConfig:
    """
    Deterministic train/validation/test splitting.

    The initial implementation uses these ratios
    and random_seed. split.py can later support
    additional strategies such as temporal or
    grouped splitting.
    """

    train_ratio: float = 0.80

    validation_ratio: float = 0.10

    test_ratio: float = 0.10

    random_seed: int = 42

    shuffle: bool = True

    def validate(
        self,
    ) -> None:
        train = number(
            self.train_ratio,
            name="train_ratio",
            minimum=0.0,
            maximum=1.0,
        )

        validation = number(
            self.validation_ratio,
            name="validation_ratio",
            minimum=0.0,
            maximum=1.0,
        )

        test = number(
            self.test_ratio,
            name="test_ratio",
            minimum=0.0,
            maximum=1.0,
        )

        total = (
            train
            + validation
            + test
        )

        if abs(
            total - 1.0
        ) > 1e-9:
            raise ValueError(
                "train_ratio + "
                "validation_ratio + "
                "test_ratio must equal 1.0"
            )

        if train <= 0:
            raise ValueError(
                "train_ratio must be > 0"
            )

        integer(
            self.random_seed,
            name="random_seed",
            minimum=0,
        )


# ---------------------------------------------------------------
# Preparation configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class PreparationConfig:
    """
    Corpus preparation behavior.
    """

    deduplicate_records: bool = True

    normalize_records: bool = True

    create_splits: bool = True

    overwrite_existing: bool = False

    preserve_rejected_records: bool = True

    output_format: str = (
        DatasetFormat.JSONL.value
    )

    def validate(
        self,
    ) -> None:
        if (
            self.output_format
            not in {
                item.value
                for item
                in DatasetFormat
            }
        ):
            raise ValueError(
                "Unknown preparation "
                "output format: "
                f"{self.output_format}"
            )


# ---------------------------------------------------------------
# Main data configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """
    Top-level configuration for OSHA data
    engineering.
    """

    enabled: bool = True

    paths: DataPathConfig = field(
        default_factory=DataPathConfig
    )

    integrity: IntegrityConfig = field(
        default_factory=IntegrityConfig
    )

    validation: DataValidationConfig = field(
        default_factory=(
            DataValidationConfig
        )
    )

    split: SplitConfig = field(
        default_factory=SplitConfig
    )

    preparation: PreparationConfig = field(
        default_factory=(
            PreparationConfig
        )
    )

    allowed_formats: Tuple[
        str,
        ...,
    ] = (
        DatasetFormat.TEXT.value,
        DatasetFormat.JSON.value,
        DatasetFormat.JSONL.value,
        DatasetFormat.CSV.value,
        DatasetFormat.PARQUET.value,
        DatasetFormat.LOG.value,
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        self.paths.validate()
        self.integrity.validate()
        self.validation.validate()
        self.split.validate()
        self.preparation.validate()

        formats = unique_strings(
            self.allowed_formats,
            name="allowed_formats",
            allow_empty=False,
        )

        if not formats:
            raise ValueError(
                "allowed_formats "
                "cannot be empty"
            )

        valid_formats = {
            item.value
            for item
            in DatasetFormat
        }

        unsupported = [
            value
            for value
            in formats
            if value
            not in valid_formats
        ]

        if unsupported:
            raise ValueError(
                "Unsupported dataset "
                "formats: "
                + ", ".join(
                    unsupported
                )
            )

    # ---------------------------------------------------------
    # Format check
    # ---------------------------------------------------------

    def format_allowed(
        self,
        dataset_format: str,
    ) -> bool:
        self.validate()

        return (
            str(
                dataset_format
            ).lower()
            in set(
                self.allowed_formats
            )
        )

    # ---------------------------------------------------------
    # Safe dictionary
    # ---------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "enabled": self.enabled,

            "paths": (
                self.paths.resolved()
            ),

            "integrity": {
                "require_checksums": (
                    self.integrity.
                    require_checksums
                ),

                "checksum_algorithm": (
                    self.integrity.
                    checksum_algorithm
                ),

                "verify_after_fetch": (
                    self.integrity.
                    verify_after_fetch
                ),

                "verify_before_prepare": (
                    self.integrity.
                    verify_before_prepare
                ),

                "maximum_file_size_bytes": (
                    self.integrity.
                    maximum_file_size_bytes
                ),
            },

            "validation": {
                "reject_invalid_records": (
                    self.validation.
                    reject_invalid_records
                ),

                "fail_on_invalid_record": (
                    self.validation.
                    fail_on_invalid_record
                ),

                "allow_empty_dataset": (
                    self.validation.
                    allow_empty_dataset
                ),

                "allow_empty_files": (
                    self.validation.
                    allow_empty_files
                ),

                "maximum_validation_issues": (
                    self.validation.
                    maximum_validation_issues
                ),

                "maximum_record_bytes": (
                    self.validation.
                    maximum_record_bytes
                ),
            },

            "split": {
                "train_ratio": (
                    self.split.
                    train_ratio
                ),

                "validation_ratio": (
                    self.split.
                    validation_ratio
                ),

                "test_ratio": (
                    self.split.
                    test_ratio
                ),

                "random_seed": (
                    self.split.
                    random_seed
                ),

                "shuffle": (
                    self.split.shuffle
                ),
            },

            "preparation": {
                "deduplicate_records": (
                    self.preparation.
                    deduplicate_records
                ),

                "normalize_records": (
                    self.preparation.
                    normalize_records
                ),

                "create_splits": (
                    self.preparation.
                    create_splits
                ),

                "overwrite_existing": (
                    self.preparation.
                    overwrite_existing
                ),

                "preserve_rejected_records": (
                    self.preparation.
                    preserve_rejected_records
                ),

                "output_format": (
                    self.preparation.
                    output_format
                ),
            },

            "allowed_formats": list(
                self.allowed_formats
            ),

            "attributes": dict(
                self.attributes
            ),
        }


# ---------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------

def _get(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Any = None,
) -> Any:
    return values.get(
        prefix + name,
        default,
    )


def _text(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Optional[
        str
    ] = None,
) -> Optional[str]:
    value = _get(
        values,
        prefix,
        name,
        default,
    )

    if value is None:
        return None

    text = str(
        value
    ).strip()

    return (
        text
        or None
    )


def _bool(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: bool,
) -> bool:
    value = _get(
        values,
        prefix,
        name,
    )

    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    return boolean(
        value,
        name=prefix + name,
    )


def _int(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: int,
    *,
    minimum: int = 0,
) -> int:
    value = _get(
        values,
        prefix,
        name,
    )

    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    return integer(
        value,
        name=prefix + name,
        minimum=minimum,
    )


def _float(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: float,
) -> float:
    value = _get(
        values,
        prefix,
        name,
    )

    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    return number(
        value,
        name=prefix + name,
        minimum=0.0,
        maximum=1.0,
    )


def _csv(
    values: Mapping[
        str,
        Any,
    ],
    prefix: str,
    name: str,
    default: Tuple[
        str,
        ...,
    ],
) -> Tuple[
    str,
    ...,
]:
    value = _get(
        values,
        prefix,
        name,
    )

    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    return tuple(
        item.lower()
        for item
        in unique_strings(
            str(value).split(
                ","
            ),
            name=prefix + name,
            allow_empty=False,
        )
    )


# ---------------------------------------------------------------
# Mapping loader
# ---------------------------------------------------------------

def from_mapping(
    values: Mapping[
        str,
        Any,
    ],
    *,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> DataConfig:
    defaults = DataConfig()

    root = (
        _text(
            values,
            prefix,
            "ROOT_DIRECTORY",
            defaults.paths.
            root_directory,
        )
        or defaults.paths.
        root_directory
    )

    paths = DataPathConfig(
        root_directory=root,

        raw_directory=(
            _text(
                values,
                prefix,
                "RAW_DIRECTORY",
                defaults.paths.
                raw_directory,
            )
            or defaults.paths.
            raw_directory
        ),

        prepared_directory=(
            _text(
                values,
                prefix,
                "PREPARED_DIRECTORY",
                defaults.paths.
                prepared_directory,
            )
            or defaults.paths.
            prepared_directory
        ),

        cache_directory=(
            _text(
                values,
                prefix,
                "CACHE_DIRECTORY",
                defaults.paths.
                cache_directory,
            )
            or defaults.paths.
            cache_directory
        ),

        manifest_directory=(
            _text(
                values,
                prefix,
                "MANIFEST_DIRECTORY",
                defaults.paths.
                manifest_directory,
            )
            or defaults.paths.
            manifest_directory
        ),

        rejected_directory=(
            _text(
                values,
                prefix,
                "REJECTED_DIRECTORY",
                defaults.paths.
                rejected_directory,
            )
            or defaults.paths.
            rejected_directory
        ),
    )

    config = DataConfig(
        enabled=_bool(
            values,
            prefix,
            "ENABLED",
            True,
        ),

        paths=paths,

        integrity=IntegrityConfig(
            require_checksums=_bool(
                values,
                prefix,
                "REQUIRE_CHECKSUMS",
                True,
            ),

            checksum_algorithm=(
                _text(
                    values,
                    prefix,
                    "CHECKSUM_ALGORITHM",
                    "sha256",
                )
                or "sha256"
            ).lower(),

            verify_after_fetch=_bool(
                values,
                prefix,
                "VERIFY_AFTER_FETCH",
                True,
            ),

            verify_before_prepare=_bool(
                values,
                prefix,
                "VERIFY_BEFORE_PREPARE",
                True,
            ),

            maximum_file_size_bytes=(
                _int(
                    values,
                    prefix,
                    "MAX_FILE_SIZE_BYTES",
                    defaults.integrity.
                    maximum_file_size_bytes,
                    minimum=1,
                )
            ),
        ),

        validation=DataValidationConfig(
            reject_invalid_records=_bool(
                values,
                prefix,
                "REJECT_INVALID_RECORDS",
                True,
            ),

            fail_on_invalid_record=_bool(
                values,
                prefix,
                "FAIL_ON_INVALID_RECORD",
                False,
            ),

            allow_empty_dataset=_bool(
                values,
                prefix,
                "ALLOW_EMPTY_DATASET",
                False,
            ),

            allow_empty_files=_bool(
                values,
                prefix,
                "ALLOW_EMPTY_FILES",
                False,
            ),

            maximum_validation_issues=(
                _int(
                    values,
                    prefix,
                    "MAX_VALIDATION_ISSUES",
                    1000,
                    minimum=1,
                )
            ),

            maximum_record_bytes=_int(
                values,
                prefix,
                "MAX_RECORD_BYTES",
                4
                * 1024
                * 1024,
                minimum=1,
            ),
        ),

        split=SplitConfig(
            train_ratio=_float(
                values,
                prefix,
                "TRAIN_RATIO",
                0.80,
            ),

            validation_ratio=_float(
                values,
                prefix,
                "VALIDATION_RATIO",
                0.10,
            ),

            test_ratio=_float(
                values,
                prefix,
                "TEST_RATIO",
                0.10,
            ),

            random_seed=_int(
                values,
                prefix,
                "RANDOM_SEED",
                42,
                minimum=0,
            ),

            shuffle=_bool(
                values,
                prefix,
                "SHUFFLE",
                True,
            ),
        ),

        preparation=PreparationConfig(
            deduplicate_records=_bool(
                values,
                prefix,
                "DEDUPLICATE_RECORDS",
                True,
            ),

            normalize_records=_bool(
                values,
                prefix,
                "NORMALIZE_RECORDS",
                True,
            ),

            create_splits=_bool(
                values,
                prefix,
                "CREATE_SPLITS",
                True,
            ),

            overwrite_existing=_bool(
                values,
                prefix,
                "OVERWRITE_EXISTING",
                False,
            ),

            preserve_rejected_records=(
                _bool(
                    values,
                    prefix,
                    "PRESERVE_REJECTED_RECORDS",
                    True,
                )
            ),

            output_format=(
                _text(
                    values,
                    prefix,
                    "OUTPUT_FORMAT",
                    DatasetFormat.
                    JSONL.value,
                )
                or DatasetFormat.
                JSONL.value
            ).lower(),
        ),

        allowed_formats=_csv(
            values,
            prefix,
            "ALLOWED_FORMATS",
            defaults.allowed_formats,
        ),

        attributes={
            "config_source": (
                "mapping"
            ),
            "environment_prefix": (
                prefix
            ),
        },
    )

    config.validate()

    return config


# ---------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------

def from_environment(
    *,
    environ: Optional[
        Mapping[
            str,
            str,
        ]
    ] = None,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> DataConfig:
    source = (
        os.environ
        if environ is None
        else environ
    )

    config = from_mapping(
        source,
        prefix=prefix,
    )

    return DataConfig(
        enabled=config.enabled,
        paths=config.paths,
        integrity=config.integrity,
        validation=config.validation,
        split=config.split,
        preparation=config.preparation,
        allowed_formats=(
            config.allowed_formats
        ),
        attributes={
            **config.attributes,
            "config_source": (
                "environment"
            ),
        },
    )


# ---------------------------------------------------------------
# Safe summary
# ---------------------------------------------------------------

def safe_config_summary(
    config: DataConfig,
) -> Dict[str, Any]:
    config.validate()

    paths = config.paths.resolved()

    return {
        "enabled": config.enabled,

        "paths": paths,

        "require_checksums": (
            config.integrity.
            require_checksums
        ),

        "checksum_algorithm": (
            config.integrity.
            checksum_algorithm
        ),

        "maximum_file_size_bytes": (
            config.integrity.
            maximum_file_size_bytes
        ),

        "reject_invalid_records": (
            config.validation.
            reject_invalid_records
        ),

        "fail_on_invalid_record": (
            config.validation.
            fail_on_invalid_record
        ),

        "split": {
            "train": (
                config.split.
                train_ratio
            ),

            "validation": (
                config.split.
                validation_ratio
            ),

            "test": (
                config.split.
                test_ratio
            ),

            "random_seed": (
                config.split.
                random_seed
            ),

            "shuffle": (
                config.split.shuffle
            ),
        },

        "deduplicate_records": (
            config.preparation.
            deduplicate_records
        ),

        "normalize_records": (
            config.preparation.
            normalize_records
        ),

        "output_format": (
            config.preparation.
            output_format
        ),

        "allowed_formats": list(
            config.allowed_formats
        ),
    }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_ENV_PREFIX",

    "DataPathConfig",
    "IntegrityConfig",
    "DataValidationConfig",
    "SplitConfig",
    "PreparationConfig",
    "DataConfig",

    "from_mapping",
    "from_environment",

    "safe_config_summary",
]
