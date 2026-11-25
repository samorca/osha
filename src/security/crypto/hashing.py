"""
Cryptographic hashing utilities for OSHA.

Provides secure hashing for:
- telemetry and event payloads
- datasets
- model artefacts
- manifests
- audit records
- remediation envelopes

This module intentionally exposes only approved cryptographic hash
algorithms. MD5 and SHA-1 are not supported.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Final, Mapping, TypeAlias


DEFAULT_CHUNK_SIZE: Final[int] = 1024 * 1024
MIN_CHUNK_SIZE: Final[int] = 4 * 1024
MAX_CHUNK_SIZE: Final[int] = 64 * 1024 * 1024

_HEX_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]+$")

JSONPrimitive: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = (
    JSONPrimitive
    | list["JSONValue"]
    | tuple["JSONValue", ...]
    | Mapping[str, "JSONValue"]
)


class HashingError(Exception):
    """Base exception for hashing failures."""


class UnsupportedHashAlgorithmError(HashingError, ValueError):
    """Requested hashing algorithm is not permitted."""


class InvalidDigestError(HashingError, ValueError):
    """Digest is malformed or invalid."""


class InvalidHashInputError(HashingError, TypeError):
    """Input cannot be safely or deterministically hashed."""


class FileHashingError(HashingError, OSError):
    """File cannot be safely hashed."""


class FileChangedDuringHashError(FileHashingError):
    """File changed while hashing was in progress."""


class HashAlgorithm(str, Enum):
    """Cryptographic hashing algorithms permitted by OSHA."""

    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"

    SHA3_256 = "sha3_256"
    SHA3_384 = "sha3_384"
    SHA3_512 = "sha3_512"


DEFAULT_ALGORITHM: Final[HashAlgorithm] = HashAlgorithm.SHA256


_DIGEST_SIZES: Final[dict[HashAlgorithm, int]] = {
    HashAlgorithm.SHA256: 32,
    HashAlgorithm.SHA384: 48,
    HashAlgorithm.SHA512: 64,
    HashAlgorithm.SHA3_256: 32,
    HashAlgorithm.SHA3_384: 48,
    HashAlgorithm.SHA3_512: 64,
}


@dataclass(frozen=True, slots=True)
class Digest:
    """
    Immutable cryptographic digest.

    String format:

        algorithm:hexadecimal-digest
    """

    algorithm: HashAlgorithm
    hexdigest: str

    def __post_init__(self) -> None:
        algorithm = normalize_algorithm(self.algorithm)
        digest = _normalize_hexdigest(self.hexdigest)

        expected_length = _DIGEST_SIZES[algorithm] * 2

        if len(digest) != expected_length:
            raise InvalidDigestError(
                f"Invalid {algorithm.value} digest length: "
                f"expected {expected_length}, got {len(digest)}."
            )

        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "hexdigest", digest)

    @property
    def raw(self) -> bytes:
        """Return digest as raw bytes."""
        return bytes.fromhex(self.hexdigest)

    def as_dict(self) -> dict[str, str]:
        """Return JSON-compatible representation."""
        return {
            "algorithm": self.algorithm.value,
            "digest": self.hexdigest,
        }

    def verify(
        self,
        candidate: bytes | bytearray | memoryview,
    ) -> bool:
        """Verify bytes against this digest."""
        return verify_bytes(candidate, self)

    def __str__(self) -> str:
        return f"{self.algorithm.value}:{self.hexdigest}"

    @classmethod
    def parse(cls, value: str) -> "Digest":
        """Parse algorithm:hexdigest representation."""

        if not isinstance(value, str):
            raise InvalidDigestError(
                "Encoded digest must be a string."
            )

        algorithm_name, separator, digest = value.strip().partition(":")

        if not separator or not algorithm_name or not digest:
            raise InvalidDigestError(
                "Digest must have format 'algorithm:hexdigest'."
            )

        return cls(
            algorithm=normalize_algorithm(algorithm_name),
            hexdigest=digest,
        )


@dataclass(frozen=True, slots=True)
class FileDigest:
    """
    Digest and filesystem metadata for an artefact.
    """

    path: str
    digest: Digest
    size_bytes: int
    modified_time_ns: int
    device: int
    inode: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "algorithm": self.digest.algorithm.value,
            "digest": self.digest.hexdigest,
            "size_bytes": self.size_bytes,
            "modified_time_ns": self.modified_time_ns,
            "device": self.device,
            "inode": self.inode,
        }


def normalize_algorithm(
    algorithm: HashAlgorithm | str,
) -> HashAlgorithm:
    """
    Validate and normalize a hashing algorithm.

    Only explicitly allowed algorithms are accepted.
    """

    if isinstance(algorithm, HashAlgorithm):
        return algorithm

    if not isinstance(algorithm, str):
        raise UnsupportedHashAlgorithmError(
            "Algorithm must be a HashAlgorithm or string."
        )

    name = algorithm.strip().lower().replace("-", "_")

    aliases = {
        "sha256": HashAlgorithm.SHA256,
        "sha_256": HashAlgorithm.SHA256,

        "sha384": HashAlgorithm.SHA384,
        "sha_384": HashAlgorithm.SHA384,

        "sha512": HashAlgorithm.SHA512,
        "sha_512": HashAlgorithm.SHA512,

        "sha3_256": HashAlgorithm.SHA3_256,
        "sha3_384": HashAlgorithm.SHA3_384,
        "sha3_512": HashAlgorithm.SHA3_512,
    }

    try:
        return aliases[name]

    except KeyError as exc:
        allowed = ", ".join(x.value for x in HashAlgorithm)

        raise UnsupportedHashAlgorithmError(
            f"Unsupported hashing algorithm {algorithm!r}. "
            f"Allowed: {allowed}."
        ) from exc


def canonical_json_bytes(value: JSONValue) -> bytes:
    """
    Convert JSON-compatible data into deterministic UTF-8 bytes.

    Object keys are sorted and insignificant whitespace is removed.
    Non-finite floats are rejected.
    """

    _validate_json_value(value, "$")

    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidHashInputError(
            "Unable to deterministically serialize JSON value."
        ) from exc

    return serialized.encode("utf-8")


def hash_bytes(
    data: bytes | bytearray | memoryview,
    algorithm: HashAlgorithm | str = DEFAULT_ALGORITHM,
) -> Digest:
    """Hash bytes."""

    selected = normalize_algorithm(algorithm)

    hasher = _new_hasher(selected)
    hasher.update(_as_byte_view(data))

    return Digest(
        algorithm=selected,
        hexdigest=hasher.hexdigest(),
    )


def hash_text(
    text: str,
    algorithm: HashAlgorithm | str = DEFAULT_ALGORITHM,
    *,
    encoding: str = "utf-8",
) -> Digest:
    """Hash text."""

    if not isinstance(text, str):
        raise InvalidHashInputError(
            "Text input must be a string."
        )

    if not isinstance(encoding, str) or not encoding:
        raise InvalidHashInputError(
            "Encoding must be a non-empty string."
        )

    try:
        encoded = text.encode(
            encoding,
            errors="strict",
        )

    except (LookupError, UnicodeEncodeError) as exc:
        raise InvalidHashInputError(
            f"Unable to encode text using {encoding!r}."
        ) from exc

    return hash_bytes(
        encoded,
        algorithm,
    )


def hash_json(
    value: JSONValue,
    algorithm: HashAlgorithm | str = DEFAULT_ALGORITHM,
) -> Digest:
    """Hash deterministic JSON."""

    return hash_bytes(
        canonical_json_bytes(value),
        algorithm,
    )


def hash_stream(
    stream: BinaryIO,
    algorithm: HashAlgorithm | str = DEFAULT_ALGORITHM,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[Digest, int]:
    """
    Incrementally hash a binary stream.

    Returns:

        (Digest, bytes_read)
    """

    chunk_size = _validate_chunk_size(chunk_size)
    selected = normalize_algorithm(algorithm)

    if not hasattr(stream, "read"):
        raise InvalidHashInputError(
            "Stream must provide read()."
        )

    hasher = _new_hasher(selected)

    total_bytes = 0

    while True:
        chunk = stream.read(chunk_size)

        if chunk == b"":
            break

        if not isinstance(
            chunk,
            (bytes, bytearray, memoryview),
        ):
            raise InvalidHashInputError(
                "Binary stream returned non-bytes data."
            )

        view = _as_byte_view(chunk)

        hasher.update(view)

        total_bytes += view.nbytes

    return (
        Digest(
            algorithm=selected,
            hexdigest=hasher.hexdigest(),
        ),
        total_bytes,
    )


def hash_file(
    path: str | os.PathLike[str],
    algorithm: HashAlgorithm | str = DEFAULT_ALGORITHM,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    reject_symlinks: bool = True,
    require_stable_file: bool = True,
) -> FileDigest:
    """
    Incrementally hash a regular file.

    Symbolic links are rejected by default.

    The function can also detect filesystem metadata changes that
    occur while the file is being hashed.
    """

    file_path = _normalize_path(path)

    chunk_size = _validate_chunk_size(chunk_size)
    selected = normalize_algorithm(algorithm)

    try:
        first_stat = file_path.lstat()

    except OSError as exc:
        raise FileHashingError(
            f"Unable to inspect file: {file_path}"
        ) from exc

    if reject_symlinks and stat.S_ISLNK(first_stat.st_mode):
        raise FileHashingError(
            f"Refusing symbolic link: {file_path}"
        )

    if not stat.S_ISREG(first_stat.st_mode):
        raise FileHashingError(
            f"Not a regular file: {file_path}"
        )

    flags = os.O_RDONLY

    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    if reject_symlinks and hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    fd: int | None = None

    try:
        fd = os.open(
            file_path,
            flags,
        )

        before = os.fstat(fd)

        if not stat.S_ISREG(before.st_mode):
            raise FileHashingError(
                f"Opened object is not a regular file: {file_path}"
            )

        if reject_symlinks:
            if (
                before.st_dev != first_stat.st_dev
                or before.st_ino != first_stat.st_ino
            ):
                raise FileHashingError(
                    f"File identity changed before hashing: {file_path}"
                )

        with os.fdopen(
            fd,
            mode="rb",
            closefd=True,
        ) as stream:

            fd = None

            digest, bytes_read = hash_stream(
                stream,
                selected,
                chunk_size=chunk_size,
            )

            after = os.fstat(
                stream.fileno()
            )

    except FileHashingError:
        raise

    except OSError as exc:
        raise FileHashingError(
            f"Unable to hash file: {file_path}"
        ) from exc

    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    if bytes_read != after.st_size:
        raise FileChangedDuringHashError(
            f"File size changed while hashing: {file_path}"
        )

    if require_stable_file and _file_changed(
        before,
        after,
    ):
        raise FileChangedDuringHashError(
            f"File changed while hashing: {file_path}"
        )

    return FileDigest(
        path=str(file_path),
        digest=digest,
        size_bytes=after.st_size,
        modified_time_ns=after.st_mtime_ns,
        device=after.st_dev,
        inode=after.st_ino,
    )


def verify_bytes(
    data: bytes | bytearray | memoryview,
    expected: Digest | str,
) -> bool:
    """
    Verify bytes with constant-time digest comparison.
    """

    expected_digest = _coerce_digest(expected)

    actual = hash_bytes(
        data,
        expected_digest.algorithm,
    )

    return hmac.compare_digest(
        actual.raw,
        expected_digest.raw,
    )


def verify_text(
    text: str,
    expected: Digest | str,
    *,
    encoding: str = "utf-8",
) -> bool:
    """Verify text against a digest."""

    expected_digest = _coerce_digest(expected)

    actual = hash_text(
        text,
        expected_digest.algorithm,
        encoding=encoding,
    )

    return hmac.compare_digest(
        actual.raw,
        expected_digest.raw,
    )


def verify_json(
    value: JSONValue,
    expected: Digest | str,
) -> bool:
    """Verify deterministic JSON against a digest."""

    expected_digest = _coerce_digest(expected)

    actual = hash_json(
        value,
        expected_digest.algorithm,
    )

    return hmac.compare_digest(
        actual.raw,
        expected_digest.raw,
    )


def verify_file(
    path: str | os.PathLike[str],
    expected: Digest | str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    reject_symlinks: bool = True,
    require_stable_file: bool = True,
) -> bool:
    """Verify a file against an expected digest."""

    expected_digest = _coerce_digest(expected)

    actual = hash_file(
        path,
        expected_digest.algorithm,
        chunk_size=chunk_size,
        reject_symlinks=reject_symlinks,
        require_stable_file=require_stable_file,
    )

    return hmac.compare_digest(
        actual.digest.raw,
        expected_digest.raw,
    )


def _new_hasher(
    algorithm: HashAlgorithm,
) -> Any:
    """
    Create a cryptographic hashlib instance.

    usedforsecurity=True explicitly marks this as security-sensitive use.
    """

    try:
        return hashlib.new(
            algorithm.value,
            usedforsecurity=True,
        )

    except (TypeError, ValueError) as exc:
        raise UnsupportedHashAlgorithmError(
            f"Hash algorithm {algorithm.value!r} "
            "is unavailable in this security environment."
        ) from exc


def _coerce_digest(
    value: Digest | str,
) -> Digest:

    if isinstance(value, Digest):
        return value

    if isinstance(value, str):
        return Digest.parse(value)

    raise InvalidDigestError(
        "Expected digest must be Digest or string."
    )


def _normalize_hexdigest(
    value: str,
) -> str:

    if not isinstance(value, str):
        raise InvalidDigestError(
            "Digest must be a string."
        )

    digest = value.strip().lower()

    if not digest:
        raise InvalidDigestError(
            "Digest cannot be empty."
        )

    if _HEX_PATTERN.fullmatch(digest) is None:
        raise InvalidDigestError(
            "Digest contains non-hexadecimal characters."
        )

    return digest


def _as_byte_view(
    data: bytes | bytearray | memoryview,
) -> memoryview:

    if not isinstance(
        data,
        (bytes, bytearray, memoryview),
    ):
        raise InvalidHashInputError(
            "Input must be bytes-like."
        )

    try:
        view = memoryview(data)

        if not view.contiguous:
            raise InvalidHashInputError(
                "Memoryview must be contiguous."
            )

        return view.cast("B")

    except (TypeError, ValueError) as exc:
        raise InvalidHashInputError(
            "Unable to interpret input as bytes."
        ) from exc


def _normalize_path(
    path: str | os.PathLike[str],
) -> Path:

    if isinstance(path, str) and "\x00" in path:
        raise FileHashingError(
            "Path contains a null byte."
        )

    try:
        result = Path(path)

    except (TypeError, ValueError) as exc:
        raise FileHashingError(
            "Invalid file path."
        ) from exc

    if not str(result):
        raise FileHashingError(
            "File path cannot be empty."
        )

    return result


def _validate_chunk_size(
    chunk_size: int,
) -> int:

    if isinstance(chunk_size, bool) or not isinstance(
        chunk_size,
        int,
    ):
        raise InvalidHashInputError(
            "Chunk size must be an integer."
        )

    if not MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE:
        raise InvalidHashInputError(
            f"Chunk size must be between "
            f"{MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE} bytes."
        )

    return chunk_size


def _file_changed(
    before: os.stat_result,
    after: os.stat_result,
) -> bool:

    return any(
        (
            before.st_dev != after.st_dev,
            before.st_ino != after.st_ino,
            before.st_size != after.st_size,
            before.st_mtime_ns != after.st_mtime_ns,
            before.st_ctime_ns != after.st_ctime_ns,
        )
    )


def _validate_json_value(
    value: JSONValue,
    path: str,
) -> None:

    if value is None:
        return

    if isinstance(value, (str, bool)):
        return

    if isinstance(value, int):
        return

    if isinstance(value, float):

        if not math.isfinite(value):
            raise InvalidHashInputError(
                f"Non-finite float at {path}."
            )

        return

    if isinstance(value, Mapping):

        for key, child in value.items():

            if not isinstance(key, str):
                raise InvalidHashInputError(
                    f"JSON key at {path} must be a string."
                )

            _validate_json_value(
                child,
                f"{path}.{key}",
            )

        return

    if isinstance(value, (list, tuple)):

        for index, child in enumerate(value):

            _validate_json_value(
                child,
                f"{path}[{index}]",
            )

        return

    raise InvalidHashInputError(
        f"Unsupported JSON type at {path}: "
        f"{type(value).__name__}"
    )


__all__ = [
    "DEFAULT_ALGORITHM",
    "DEFAULT_CHUNK_SIZE",
    "Digest",
    "FileChangedDuringHashError",
    "FileDigest",
    "FileHashingError",
    "HashAlgorithm",
    "HashingError",
    "InvalidDigestError",
    "InvalidHashInputError",
    "UnsupportedHashAlgorithmError",
    "canonical_json_bytes",
    "hash_bytes",
    "hash_file",
    "hash_json",
    "hash_stream",
    "hash_text",
    "normalize_algorithm",
    "verify_bytes",
    "verify_file",
    "verify_json",
    "verify_text",
]
