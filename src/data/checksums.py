from __future__ import annotations

"""
Streaming checksum and dataset-integrity utilities
for OSHA data engineering.
"""

import hashlib

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

from src.data.config import (
    IntegrityConfig,
)

from src.data.errors import (
    ChecksumAlgorithmError,
    ChecksumMismatchError,
    DataIntegrityError,
    DataSourceNotFoundError,
    DatasetFileSizeError,
)

from src.data.manifests import (
    resolve_dataset_file,
)

from src.data.schema import (
    DatasetFile,
    DatasetManifest,
)


# ---------------------------------------------------------------
# Supported algorithms
# ---------------------------------------------------------------

SUPPORTED_ALGORITHMS = {
    "sha256",
    "sha512",
}


# ---------------------------------------------------------------
# Default streaming block size
# ---------------------------------------------------------------

DEFAULT_BLOCK_SIZE = 1024 * 1024


# ---------------------------------------------------------------
# Checksum result
# ---------------------------------------------------------------


@dataclass(frozen=True)
class ChecksumResult:
    """
    Result of calculating one file checksum.
    """

    path: str

    algorithm: str

    checksum: str

    size_bytes: int

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "path": self.path,
            "algorithm": (self.algorithm),
            "checksum": (self.checksum),
            "size_bytes": (self.size_bytes),
        }


# ---------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------


@dataclass(frozen=True)
class FileIntegrityResult:
    """
    Bounded integrity result for one declared
    DatasetFile.
    """

    file_id: Optional[str]

    path: str

    present: bool

    required: bool

    checksum_required: bool

    checksum_verified: bool

    size_verified: bool

    valid: bool

    algorithm: Optional[str] = None

    calculated_checksum: Optional[str] = None

    expected_checksum: Optional[str] = None

    size_bytes: Optional[int] = None

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "file_id": (self.file_id),
            "path": self.path,
            "present": self.present,
            "required": self.required,
            "checksum_required": (self.checksum_required),
            "checksum_verified": (self.checksum_verified),
            "size_verified": (self.size_verified),
            "valid": self.valid,
            "algorithm": (self.algorithm),
            "calculated_checksum": (self.calculated_checksum),
            "expected_checksum": (self.expected_checksum),
            "size_bytes": (self.size_bytes),
        }


# ---------------------------------------------------------------
# Algorithm normalization
# ---------------------------------------------------------------


def normalize_algorithm(
    algorithm: str,
) -> str:
    normalized = str(algorithm).strip().lower()

    if normalized not in SUPPORTED_ALGORITHMS:
        raise (
            ChecksumAlgorithmError(
                "Unsupported checksum algorithm.",
                details={
                    "algorithm": (normalized),
                    "supported": (sorted(SUPPORTED_ALGORITHMS)),
                },
            )
        )

    return normalized


# ---------------------------------------------------------------
# Hash implementation
# ---------------------------------------------------------------


def _hasher(
    algorithm: str,
) -> Any:
    algorithm = normalize_algorithm(algorithm)

    if algorithm == "sha256":
        return hashlib.sha256()

    if algorithm == "sha512":
        return hashlib.sha512()

    raise (ChecksumAlgorithmError("Unsupported checksum algorithm."))


# ---------------------------------------------------------------
# Calculate checksum
# ---------------------------------------------------------------


def checksum_file(
    path: str | Path,
    *,
    algorithm: str = "sha256",
    maximum_size_bytes: Optional[int] = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> ChecksumResult:
    """
    Calculate a checksum incrementally.

    The complete dataset file is never loaded
    into memory at once.
    """

    source = Path(path).expanduser()

    algorithm = normalize_algorithm(algorithm)

    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    if maximum_size_bytes is not None and maximum_size_bytes < 1:
        raise ValueError("maximum_size_bytes must be >= 1")

    if not source.exists():
        raise (
            DataSourceNotFoundError(
                "Dataset file was not found.",
                details={
                    "path": str(source),
                },
            )
        )

    if not source.is_file():
        raise (
            DataIntegrityError(
                "Dataset path is not a regular file.",
                details={
                    "path": str(source),
                },
            )
        )

    try:
        stat = source.stat()

    except OSError as exc:
        raise (
            DataIntegrityError(
                "Dataset file metadata could not be read.",
                details={
                    "path": str(source),
                    "error_type": (type(exc).__name__),
                },
            )
        ) from exc

    size_bytes = int(stat.st_size)

    if maximum_size_bytes is not None and size_bytes > maximum_size_bytes:
        raise (
            DatasetFileSizeError(
                "Dataset file exceeds the configured size limit.",
                details={
                    "path": str(source),
                    "size_bytes": (size_bytes),
                    "maximum_size_bytes": (maximum_size_bytes),
                },
            )
        )

    digest = _hasher(algorithm)

    bytes_read = 0

    try:
        with source.open("rb") as stream:
            while True:
                block = stream.read(block_size)

                if not block:
                    break

                bytes_read += len(block)

                if maximum_size_bytes is not None and bytes_read > maximum_size_bytes:
                    raise (
                        DatasetFileSizeError(
                            "Dataset file "
                            "exceeded the "
                            "configured size "
                            "limit while "
                            "reading.",
                            details={
                                "path": str(source),
                                "maximum_size_bytes": (maximum_size_bytes),
                            },
                        )
                    )

                digest.update(block)

    except DatasetFileSizeError:
        raise

    except OSError as exc:
        raise (
            DataIntegrityError(
                "Dataset file could not be read for checksum calculation.",
                details={
                    "path": str(source),
                    "error_type": (type(exc).__name__),
                },
            )
        ) from exc

    return ChecksumResult(
        path=str(source.resolve()),
        algorithm=algorithm,
        checksum=(digest.hexdigest()),
        size_bytes=(bytes_read),
    )


# ---------------------------------------------------------------
# Convenience checksums
# ---------------------------------------------------------------


def sha256_file(
    path: str | Path,
    *,
    maximum_size_bytes: Optional[int] = None,
) -> str:
    return checksum_file(
        path,
        algorithm="sha256",
        maximum_size_bytes=(maximum_size_bytes),
    ).checksum


def sha512_file(
    path: str | Path,
    *,
    maximum_size_bytes: Optional[int] = None,
) -> str:
    return checksum_file(
        path,
        algorithm="sha512",
        maximum_size_bytes=(maximum_size_bytes),
    ).checksum


# ---------------------------------------------------------------
# Compare checksum
# ---------------------------------------------------------------


def checksum_matches(
    actual: str,
    expected: str,
) -> bool:
    """
    Checksums are hex strings and are compared
    after whitespace/case normalization.
    """

    return str(actual).strip().lower() == str(expected).strip().lower()


# ---------------------------------------------------------------
# Verify one DatasetFile
# ---------------------------------------------------------------


def verify_dataset_file(
    root_directory: str | Path,
    dataset_file: DatasetFile,
    *,
    config: Optional[IntegrityConfig] = None,
    raise_on_mismatch: bool = True,
) -> FileIntegrityResult:
    """
    Verify existence, size policy, declared size,
    and checksum for one manifest file.
    """

    integrity = config or IntegrityConfig()

    integrity.validate()

    dataset_file.finalize()

    path = resolve_dataset_file(
        root_directory,
        dataset_file,
    )

    # ---------------------------------------------------------
    # Missing file
    # ---------------------------------------------------------

    if not path.exists():
        valid_missing = not dataset_file.required

        result = FileIntegrityResult(
            file_id=(dataset_file.file_id),
            path=str(path),
            present=False,
            required=(dataset_file.required),
            checksum_required=(integrity.require_checksums),
            checksum_verified=False,
            size_verified=False,
            valid=(valid_missing),
            algorithm=(dataset_file.checksum_algorithm),
            expected_checksum=(dataset_file.checksum),
        )

        if dataset_file.required and raise_on_mismatch:
            raise (
                DataSourceNotFoundError(
                    "Required dataset file is missing.",
                    details={
                        "file_id": (dataset_file.file_id),
                        "path": str(path),
                    },
                )
            )

        return result

    if not path.is_file():
        if raise_on_mismatch:
            raise (
                DataIntegrityError(
                    "Dataset path is not a regular file.",
                    details={
                        "file_id": (dataset_file.file_id),
                        "path": str(path),
                    },
                )
            )

        return FileIntegrityResult(
            file_id=(dataset_file.file_id),
            path=str(path),
            present=True,
            required=(dataset_file.required),
            checksum_required=(integrity.require_checksums),
            checksum_verified=False,
            size_verified=False,
            valid=False,
            algorithm=(dataset_file.checksum_algorithm),
            expected_checksum=(dataset_file.checksum),
        )

    # ---------------------------------------------------------
    # Determine algorithm
    # ---------------------------------------------------------

    algorithm = dataset_file.checksum_algorithm or integrity.checksum_algorithm

    algorithm = normalize_algorithm(algorithm)

    # ---------------------------------------------------------
    # Missing required checksum
    # ---------------------------------------------------------

    if integrity.require_checksums and not dataset_file.checksum:
        if raise_on_mismatch:
            raise (
                ChecksumMismatchError(
                    "Dataset file has no required checksum.",
                    details={
                        "file_id": (dataset_file.file_id),
                        "path": str(path),
                    },
                )
            )

        return FileIntegrityResult(
            file_id=(dataset_file.file_id),
            path=str(path),
            present=True,
            required=(dataset_file.required),
            checksum_required=True,
            checksum_verified=False,
            size_verified=False,
            valid=False,
            algorithm=algorithm,
            expected_checksum=None,
        )

    # ---------------------------------------------------------
    # Calculate digest
    # ---------------------------------------------------------

    calculated = checksum_file(
        path,
        algorithm=algorithm,
        maximum_size_bytes=(integrity.maximum_file_size_bytes),
    )

    # ---------------------------------------------------------
    # Declared size
    # ---------------------------------------------------------

    size_verified = True

    if dataset_file.size_bytes is not None:
        size_verified = int(dataset_file.size_bytes) == calculated.size_bytes

        if not size_verified and raise_on_mismatch:
            raise (
                DatasetFileSizeError(
                    "Dataset file size does not match the manifest.",
                    details={
                        "file_id": (dataset_file.file_id),
                        "expected_size_bytes": (dataset_file.size_bytes),
                        "actual_size_bytes": (calculated.size_bytes),
                    },
                )
            )

    # ---------------------------------------------------------
    # Checksum comparison
    # ---------------------------------------------------------

    checksum_verified = (
        True
        if dataset_file.checksum is None
        else checksum_matches(
            calculated.checksum,
            dataset_file.checksum,
        )
    )

    valid = bool(
        size_verified
        and checksum_verified
        and (dataset_file.checksum is not None or not integrity.require_checksums)
    )

    if not checksum_verified and raise_on_mismatch:
        raise (
            ChecksumMismatchError(
                "Dataset file checksum does not match the manifest.",
                details={
                    "file_id": (dataset_file.file_id),
                    "algorithm": (algorithm),
                    "expected_checksum": (dataset_file.checksum),
                    "actual_checksum": (calculated.checksum),
                },
            )
        )

    return FileIntegrityResult(
        file_id=(dataset_file.file_id),
        path=str(path),
        present=True,
        required=(dataset_file.required),
        checksum_required=(integrity.require_checksums),
        checksum_verified=(checksum_verified),
        size_verified=(size_verified),
        valid=valid,
        algorithm=algorithm,
        calculated_checksum=(calculated.checksum),
        expected_checksum=(dataset_file.checksum),
        size_bytes=(calculated.size_bytes),
    )


# ---------------------------------------------------------------
# Verify manifest
# ---------------------------------------------------------------


def verify_manifest_files(
    manifest: DatasetManifest,
    root_directory: str | Path,
    *,
    config: Optional[IntegrityConfig] = None,
    raise_on_mismatch: bool = True,
) -> List[FileIntegrityResult]:
    """
    Verify all files declared by a manifest.

    Verification is performed one file at a time
    so large training corpora are not loaded into
    memory.
    """

    manifest.finalize()

    integrity = config or IntegrityConfig()

    integrity.validate()

    results: List[FileIntegrityResult] = []

    for dataset_file in manifest.files:
        result = verify_dataset_file(
            root_directory,
            dataset_file,
            config=integrity,
            raise_on_mismatch=(raise_on_mismatch),
        )

        results.append(result)

    return results


# ---------------------------------------------------------------
# Manifest verification status
# ---------------------------------------------------------------


def manifest_integrity_ok(
    results: List[FileIntegrityResult],
) -> bool:
    """
    Return True when every required file is valid
    and every present optional file is valid.
    """

    for result in results:
        if result.required and not result.valid:
            return False

        if result.present and not result.valid:
            return False

    return True


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "SUPPORTED_ALGORITHMS",
    "DEFAULT_BLOCK_SIZE",
    "ChecksumResult",
    "FileIntegrityResult",
    "normalize_algorithm",
    "checksum_file",
    "sha256_file",
    "sha512_file",
    "checksum_matches",
    "verify_dataset_file",
    "verify_manifest_files",
    "manifest_integrity_ok",
]
