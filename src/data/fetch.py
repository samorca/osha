from __future__ import annotations

"""
Controlled dataset acquisition for OSHA.

Initial implementation:

    local file      -> controlled raw directory
    local directory -> controlled raw directory

Remote HTTP/object-storage acquisition should be
added through explicit source adapters later.
"""

import os
import shutil

from dataclasses import (
    dataclass,
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
    checksum_file,
    checksum_matches,
)

from src.data.config import (
    DataConfig,
)

from src.data.errors import (
    ChecksumMismatchError,
    DataFetchError,
    DataFetchLimitError,
    DataSourceNotFoundError,
    DataSourceUnsupportedError,
)

from src.data.schema import (
    DataSource,
    DataSourceKind,
    DatasetFile,
    DatasetManifest,
)


# ---------------------------------------------------------------
# Copy block size
# ---------------------------------------------------------------

DEFAULT_COPY_BLOCK_SIZE = (
    1024
    * 1024
)


# ---------------------------------------------------------------
# Fetch result
# ---------------------------------------------------------------

@dataclass(frozen=True)
class FetchResult:
    """
    Result of acquiring one dataset file.
    """

    source: str

    destination: str

    size_bytes: int

    checksum: Optional[
        str
    ] = None

    checksum_algorithm: Optional[
        str
    ] = None

    verified: bool = False

    overwritten: bool = False

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "source": self.source,

            "destination": (
                self.destination
            ),

            "size_bytes": (
                self.size_bytes
            ),

            "checksum": (
                self.checksum
            ),

            "checksum_algorithm": (
                self.checksum_algorithm
            ),

            "verified": (
                self.verified
            ),

            "overwritten": (
                self.overwritten
            ),
        }


# ---------------------------------------------------------------
# Path guards
# ---------------------------------------------------------------

def _resolved_root(
    root: str | Path,
) -> Path:
    return Path(
        root
    ).expanduser().resolve()


def _safe_destination(
    root: str | Path,
    relative_path: str | Path,
) -> Path:
    """
    Resolve a destination below a controlled
    root.

    Absolute destination declarations and path
    traversal outside the root are rejected.
    """

    root_path = _resolved_root(
        root
    )

    declared = Path(
        relative_path
    )

    if declared.is_absolute():
        raise (
            DataFetchError(
                "Dataset destination must "
                "be relative to the raw "
                "data directory."
            )
        )

    destination = (
        root_path
        / declared
    ).resolve()

    try:
        destination.relative_to(
            root_path
        )

    except ValueError as exc:
        raise (
            DataFetchError(
                "Dataset destination "
                "escapes the controlled "
                "raw data directory."
            )
        ) from exc

    return destination


# ---------------------------------------------------------------
# Local source validation
# ---------------------------------------------------------------

def _local_source(
    source: str | Path,
) -> Path:
    path = Path(
        source
    ).expanduser().resolve()

    if not path.exists():
        raise (
            DataSourceNotFoundError(
                "Dataset source was not "
                "found.",
                details={
                    "source": str(
                        path
                    ),
                },
            )
        )

    return path


# ---------------------------------------------------------------
# Streaming local copy
# ---------------------------------------------------------------

def _copy_file(
    source: Path,
    destination: Path,
    *,
    maximum_size_bytes: int,
    overwrite: bool,
    block_size: int = DEFAULT_COPY_BLOCK_SIZE,
) -> tuple[
    int,
    bool,
]:
    """
    Copy a local file into the controlled raw
    directory using a temporary file followed by
    replacement.

    Returns:

        size_bytes
        overwritten
    """

    if block_size < 1:
        raise ValueError(
            "block_size must be >= 1"
        )

    if not source.is_file():
        raise (
            DataFetchError(
                "Dataset source is not "
                "a regular file.",
                details={
                    "source": str(
                        source
                    ),
                },
            )
        )

    try:
        source_size = int(
            source.stat().st_size
        )

    except OSError as exc:
        raise (
            DataFetchError(
                "Dataset source metadata "
                "could not be read.",
                details={
                    "error_type": (
                        type(exc).__name__
                    ),
                },
            )
        ) from exc

    if (
        source_size
        > maximum_size_bytes
    ):
        raise (
            DataFetchLimitError(
                "Dataset source exceeds "
                "the configured file-size "
                "limit.",
                details={
                    "size_bytes": (
                        source_size
                    ),
                    "maximum_size_bytes": (
                        maximum_size_bytes
                    ),
                },
            )
        )

    existed = destination.exists()

    if (
        existed
        and not overwrite
    ):
        raise (
            DataFetchError(
                "Dataset destination "
                "already exists.",
                details={
                    "destination": str(
                        destination
                    ),
                },
            )
        )

    try:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as exc:
        raise (
            DataFetchError(
                "Dataset destination "
                "directory could not be "
                "created.",
                details={
                    "error_type": (
                        type(exc).__name__
                    ),
                },
            )
        ) from exc

    temporary = (
        destination.with_name(
            destination.name
            + ".fetching"
        )
    )

    bytes_written = 0

    try:
        with source.open(
            "rb"
        ) as source_stream:
            with temporary.open(
                "wb"
            ) as destination_stream:
                while True:
                    block = source_stream.read(
                        block_size
                    )

                    if not block:
                        break

                    bytes_written += len(
                        block
                    )

                    if (
                        bytes_written
                        > maximum_size_bytes
                    ):
                        raise (
                            DataFetchLimitError(
                                "Dataset source "
                                "exceeded the "
                                "configured "
                                "file-size limit "
                                "while copying."
                            )
                        )

                    destination_stream.write(
                        block
                    )

                destination_stream.flush()

                os.fsync(
                    destination_stream.
                    fileno()
                )

        temporary.replace(
            destination
        )

    except DataFetchLimitError:
        try:
            temporary.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise

    except OSError as exc:
        try:
            temporary.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise (
            DataFetchError(
                "Dataset file could not "
                "be copied.",
                details={
                    "error_type": (
                        type(exc).__name__
                    ),
                },
            )
        ) from exc

    return (
        bytes_written,
        existed,
    )


# ---------------------------------------------------------------
# Fetch one local file
# ---------------------------------------------------------------

def fetch_local_file(
    source: str | Path,
    *,
    raw_directory: str | Path,
    destination: str | Path,
    maximum_size_bytes: int,
    overwrite: bool = False,
    checksum: Optional[
        str
    ] = None,
    checksum_algorithm: str = (
        "sha256"
    ),
    verify_checksum: bool = True,
) -> FetchResult:
    """
    Copy one local source file to the controlled
    raw dataset directory.
    """

    source_path = _local_source(
        source
    )

    destination_path = (
        _safe_destination(
            raw_directory,
            destination,
        )
    )

    (
        size_bytes,
        overwritten,
    ) = _copy_file(
        source_path,
        destination_path,
        maximum_size_bytes=(
            maximum_size_bytes
        ),
        overwrite=overwrite,
    )

    calculated_checksum = None
    verified = False

    if (
        verify_checksum
        or checksum is not None
    ):
        calculated = checksum_file(
            destination_path,
            algorithm=(
                checksum_algorithm
            ),
            maximum_size_bytes=(
                maximum_size_bytes
            ),
        )

        calculated_checksum = (
            calculated.checksum
        )

        if checksum is not None:
            verified = checksum_matches(
                calculated_checksum,
                checksum,
            )

            if not verified:
                try:
                    destination_path.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass

                raise (
                    ChecksumMismatchError(
                        "Fetched dataset "
                        "file checksum does "
                        "not match the "
                        "expected value.",
                        details={
                            "algorithm": (
                                checksum_algorithm
                            ),
                            "expected_checksum": (
                                checksum
                            ),
                            "actual_checksum": (
                                calculated_checksum
                            ),
                        },
                    )
                )

        else:
            verified = True

    return FetchResult(
        source=str(
            source_path
        ),

        destination=str(
            destination_path
        ),

        size_bytes=(
            size_bytes
        ),

        checksum=(
            calculated_checksum
        ),

        checksum_algorithm=(
            checksum_algorithm
            if calculated_checksum
            is not None
            else None
        ),

        verified=verified,

        overwritten=(
            overwritten
        ),
    )


# ---------------------------------------------------------------
# Dataset fetcher
# ---------------------------------------------------------------

class DataFetcher:
    """
    Controlled dataset acquisition service.

    The initial implementation supports LOCAL
    manifests only.

    HTTP and object-storage sources intentionally
    raise DataSourceUnsupportedError until
    explicit fetch adapters are added.
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
    # Fetch one declared file
    # ---------------------------------------------------------

    def fetch_file(
        self,
        *,
        source_root: str | Path,
        dataset_file: DatasetFile,
        destination: Optional[
            str | Path
        ] = None,
        overwrite: Optional[
            bool
        ] = None,
    ) -> FetchResult:
        """
        Acquire one file from a local source root.

        dataset_file.path is interpreted relative
        to source_root unless destination is
        explicitly provided.
        """

        dataset_file.finalize()

        source_root_path = (
            _local_source(
                source_root
            )
        )

        if not source_root_path.is_dir():
            raise (
                DataFetchError(
                    "Local dataset source "
                    "root must be a "
                    "directory."
                )
            )

        source = (
            source_root_path
            / dataset_file.path
        ).resolve()

        try:
            source.relative_to(
                source_root_path
            )

        except ValueError as exc:
            raise (
                DataFetchError(
                    "Dataset source file "
                    "escapes the configured "
                    "local source root."
                )
            ) from exc

        destination_path = (
            destination
            if destination
            is not None
            else dataset_file.path
        )

        overwrite_value = (
            self.config.
            preparation.
            overwrite_existing
            if overwrite is None
            else overwrite
        )

        return fetch_local_file(
            source,

            raw_directory=(
                self.raw_directory
            ),

            destination=(
                destination_path
            ),

            maximum_size_bytes=(
                self.config.
                integrity.
                maximum_file_size_bytes
            ),

            overwrite=(
                overwrite_value
            ),

            checksum=(
                dataset_file.checksum
            ),

            checksum_algorithm=(
                dataset_file.
                checksum_algorithm
                or self.config.
                integrity.
                checksum_algorithm
            ),

            verify_checksum=(
                self.config.
                integrity.
                verify_after_fetch
            ),
        )

    # ---------------------------------------------------------
    # Fetch manifest
    # ---------------------------------------------------------

    def fetch_manifest(
        self,
        manifest: DatasetManifest,
        *,
        overwrite: Optional[
            bool
        ] = None,
    ) -> List[
        FetchResult
    ]:
        """
        Acquire all files declared by a LOCAL
        manifest.

        Required file failures abort the fetch.

        Missing optional files are skipped.
        """

        manifest.finalize()

        if not self.config.enabled:
            raise (
                DataFetchError(
                    "Data pipeline is "
                    "disabled."
                )
            )

        if (
            manifest.source.kind
            != DataSourceKind.LOCAL.value
        ):
            raise (
                DataSourceUnsupportedError(
                    "DataFetcher currently "
                    "supports local "
                    "dataset sources only.",
                    details={
                        "source_kind": (
                            manifest.
                            source.kind
                        ),
                    },
                )
            )

        source_root = _local_source(
            manifest.source.location
        )

        if not source_root.is_dir():
            raise (
                DataFetchError(
                    "Local manifest source "
                    "must identify a "
                    "directory."
                )
            )

        results: List[
            FetchResult
        ] = []

        for dataset_file in (
            manifest.files
        ):
            source_path = (
                source_root
                / dataset_file.path
            ).resolve()

            try:
                source_path.relative_to(
                    source_root
                )

            except ValueError as exc:
                raise (
                    DataFetchError(
                        "Manifest source "
                        "path escapes the "
                        "declared source "
                        "directory."
                    )
                ) from exc

            if not source_path.exists():
                if dataset_file.required:
                    raise (
                        DataSourceNotFoundError(
                            "Required dataset "
                            "source file is "
                            "missing.",
                            details={
                                "file_id": (
                                    dataset_file.
                                    file_id
                                ),
                                "path": str(
                                    source_path
                                ),
                            },
                        )
                    )

                continue

            result = self.fetch_file(
                source_root=source_root,
                dataset_file=(
                    dataset_file
                ),
                overwrite=overwrite,
            )

            results.append(
                result
            )

        return results

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "enabled": (
                self.config.enabled
            ),

            "supported_source_kinds": [
                DataSourceKind.
                LOCAL.value
            ],

            "raw_directory": str(
                self.raw_directory
            ),

            "maximum_file_size_bytes": (
                self.config.
                integrity.
                maximum_file_size_bytes
            ),

            "verify_after_fetch": (
                self.config.
                integrity.
                verify_after_fetch
            ),

            "overwrite_existing": (
                self.config.
                preparation.
                overwrite_existing
            ),

            "arbitrary_remote_download_enabled": (
                False
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DEFAULT_COPY_BLOCK_SIZE",

    "FetchResult",

    "fetch_local_file",

    "DataFetcher",
]
