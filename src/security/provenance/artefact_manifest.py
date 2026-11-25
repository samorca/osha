"""
Artifact provenance manifests for OSHA.

This module creates deterministic manifests describing security-sensitive
OSHA artifacts.

Examples:

- trained models
- datasets
- model checkpoints
- training outputs
- configuration snapshots
- packaged artifacts
- evaluation reports
- deployment bundles
- provenance records

A manifest records WHAT an artifact is and WHERE it came from.

It does NOT establish authenticity by itself.

Authenticity is added later by:

    model_signing.py
    dataset_signing.py
    signatures.py

Security model
--------------
1. Every artifact is cryptographically hashed.
2. Artifact size is recorded.
3. Artifact type and version are explicit.
4. Producer workload identity is recorded.
5. Input/material lineage can be recorded.
6. Tool/pipeline identity can be recorded.
7. Arbitrary secret values must not be inserted into metadata.
8. Verification recalculates the file digest and size.
9. Symbolic links are rejected by default through hashing.py.
10. Files modified during hashing are rejected by default.

A valid manifest means the artifact matches the recorded bytes and metadata.
It does NOT mean the artifact is safe, accurate, licensed, unbiased, or
authorized for deployment.
"""

from __future__ import annotations

import json
import re
import time
import uuid

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, Mapping

from ..crypto.hashing import (
    Digest,
    FileDigest,
    HashAlgorithm,
    HashingError,
    canonical_json_bytes,
    hash_file,
    hash_json,
    verify_file,
)
from ..identity.workload_identity import (
    WorkloadIdentity,
    workload_identity_from_id,
)


ARTEFACT_MANIFEST_FORMAT: Final[str] = (
    "osha-artefact-manifest"
)

ARTEFACT_MANIFEST_VERSION: Final[int] = 1

MAX_ID_LENGTH: Final[int] = 128
MAX_NAME_LENGTH: Final[int] = 256
MAX_VERSION_LENGTH: Final[int] = 128
MAX_LOCATION_LENGTH: Final[int] = 4096
MAX_METADATA_ITEMS: Final[int] = 128
MAX_METADATA_VALUE_LENGTH: Final[int] = 2048
MAX_MATERIALS: Final[int] = 1024

_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"
)


class ArtefactManifestError(Exception):
    """Base exception for OSHA artifact-manifest failures."""


class InvalidArtefactManifestError(
    ArtefactManifestError,
    ValueError,
):
    """Artifact manifest is malformed."""


class ArtefactVerificationError(
    ArtefactManifestError,
):
    """Artifact does not match its manifest."""


class ArtefactType(str, Enum):
    """Artifact categories understood by OSHA provenance."""

    MODEL = "model"
    DATASET = "dataset"
    CHECKPOINT = "checkpoint"
    CONFIGURATION = "configuration"
    EVALUATION = "evaluation"
    REPORT = "report"
    DEPLOYMENT_BUNDLE = "deployment_bundle"
    PROVENANCE = "provenance"
    GENERIC = "generic"


class ArtefactVerificationStatus(str, Enum):
    """Result of artifact-manifest verification."""

    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class MaterialReference:
    """
    Reference to an input used to produce an artifact.

    Example inputs:

    - training dataset
    - source configuration
    - base model
    - previous checkpoint
    - prepared JSONL dataset

    location is descriptive metadata. Trust comes from the digest, not from
    the location string alone.
    """

    material_id: str
    artefact_type: ArtefactType
    digest: Digest
    location: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(
            self.material_id,
            field="material_id",
        )

        if not isinstance(
            self.artefact_type,
            ArtefactType,
        ):
            raise InvalidArtefactManifestError(
                "artefact_type must be ArtefactType."
            )

        if not isinstance(
            self.digest,
            Digest,
        ):
            raise InvalidArtefactManifestError(
                "digest must be Digest."
            )

        if self.location is not None:
            _validate_text(
                self.location,
                field="location",
                max_length=MAX_LOCATION_LENGTH,
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "artefact_type": (
                self.artefact_type.value
            ),
            "digest": str(self.digest),
            "location": self.location,
            "material_id": self.material_id,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "MaterialReference":

        if not isinstance(value, Mapping):
            raise InvalidArtefactManifestError(
                "Material reference must be a mapping."
            )

        expected = {
            "artefact_type",
            "digest",
            "location",
            "material_id",
        }

        _require_exact_fields(
            value,
            expected,
            name="material reference",
        )

        try:
            artefact_type = ArtefactType(
                value["artefact_type"]
            )
        except (TypeError, ValueError) as exc:
            raise InvalidArtefactManifestError(
                "Invalid material artifact type."
            ) from exc

        try:
            digest = Digest.parse(
                value["digest"]
            )
        except Exception as exc:
            raise InvalidArtefactManifestError(
                "Invalid material digest."
            ) from exc

        return cls(
            material_id=value["material_id"],
            artefact_type=artefact_type,
            digest=digest,
            location=value["location"],
        )


@dataclass(frozen=True, slots=True)
class ArtefactManifest:
    """
    Canonical OSHA artifact provenance manifest.

    The manifest itself is not a digital signature.

    manifest_digest can be calculated with digest() and then signed by the
    later model_signing.py or dataset_signing.py modules.
    """

    manifest_id: str

    artefact_id: str
    artefact_name: str
    artefact_type: ArtefactType
    artefact_version: str

    digest: Digest
    size_bytes: int

    produced_at_ns: int
    producer_workload_id: str

    pipeline_id: str | None = None
    run_id: str | None = None

    location: str | None = None

    materials: tuple[MaterialReference, ...] = ()

    metadata: Mapping[str, str] | None = None

    format_version: int = ARTEFACT_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if (
            self.format_version
            != ARTEFACT_MANIFEST_VERSION
        ):
            raise InvalidArtefactManifestError(
                f"Unsupported manifest version: "
                f"{self.format_version!r}."
            )

        _validate_identifier(
            self.manifest_id,
            field="manifest_id",
        )

        _validate_identifier(
            self.artefact_id,
            field="artefact_id",
        )

        _validate_text(
            self.artefact_name,
            field="artefact_name",
            max_length=MAX_NAME_LENGTH,
        )

        if not isinstance(
            self.artefact_type,
            ArtefactType,
        ):
            raise InvalidArtefactManifestError(
                "artefact_type must be ArtefactType."
            )

        _validate_text(
            self.artefact_version,
            field="artefact_version",
            max_length=MAX_VERSION_LENGTH,
        )

        if not isinstance(
            self.digest,
            Digest,
        ):
            raise InvalidArtefactManifestError(
                "digest must be Digest."
            )

        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(
                self.size_bytes,
                int,
            )
            or self.size_bytes < 0
        ):
            raise InvalidArtefactManifestError(
                "size_bytes must be a non-negative integer."
            )

        if (
            isinstance(self.produced_at_ns, bool)
            or not isinstance(
                self.produced_at_ns,
                int,
            )
            or self.produced_at_ns <= 0
        ):
            raise InvalidArtefactManifestError(
                "produced_at_ns must be positive."
            )

        _validate_workload_id(
            self.producer_workload_id
        )

        if self.pipeline_id is not None:
            _validate_identifier(
                self.pipeline_id,
                field="pipeline_id",
            )

        if self.run_id is not None:
            _validate_identifier(
                self.run_id,
                field="run_id",
            )

        if self.location is not None:
            _validate_text(
                self.location,
                field="location",
                max_length=MAX_LOCATION_LENGTH,
            )

        if not isinstance(
            self.materials,
            tuple,
        ):
            raise InvalidArtefactManifestError(
                "materials must be a tuple."
            )

        if len(self.materials) > MAX_MATERIALS:
            raise InvalidArtefactManifestError(
                f"materials cannot exceed "
                f"{MAX_MATERIALS} entries."
            )

        material_ids: set[str] = set()

        for material in self.materials:
            if not isinstance(
                material,
                MaterialReference,
            ):
                raise InvalidArtefactManifestError(
                    "Every material must be "
                    "MaterialReference."
                )

            if material.material_id in material_ids:
                raise InvalidArtefactManifestError(
                    f"Duplicate material_id: "
                    f"{material.material_id!r}."
                )

            material_ids.add(
                material.material_id
            )

        normalized_metadata = (
            _normalize_metadata(
                self.metadata
            )
        )

        object.__setattr__(
            self,
            "metadata",
            normalized_metadata,
        )

    def as_dict(self) -> dict[str, Any]:
        """
        Return canonical JSON-compatible manifest.
        """

        return {
            "artefact": {
                "digest": str(self.digest),
                "id": self.artefact_id,
                "location": self.location,
                "name": self.artefact_name,
                "size_bytes": self.size_bytes,
                "type": self.artefact_type.value,
                "version": self.artefact_version,
            },
            "format": ARTEFACT_MANIFEST_FORMAT,
            "format_version": self.format_version,
            "manifest_id": self.manifest_id,
            "materials": [
                material.as_dict()
                for material in self.materials
            ],
            "metadata": dict(
                sorted(
                    (self.metadata or {}).items()
                )
            ),
            "production": {
                "pipeline_id": self.pipeline_id,
                "producer_workload_id": (
                    self.producer_workload_id
                ),
                "produced_at_ns": self.produced_at_ns,
                "run_id": self.run_id,
            },
        }

    def canonical_bytes(self) -> bytes:
        """
        Return deterministic bytes for hashing/signing.
        """

        return canonical_json_bytes(
            self.as_dict()
        )

    def manifest_digest(
        self,
        algorithm: HashAlgorithm | str = (
            HashAlgorithm.SHA256
        ),
    ) -> Digest:
        """
        Hash the complete canonical manifest.
        """

        return hash_json(
            self.as_dict(),
            algorithm=algorithm,
        )

    def to_json(self) -> str:
        """
        Serialize the manifest as deterministic compact JSON.
        """

        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "ArtefactManifest":
        """
        Parse and strictly validate an artifact manifest.
        """

        if not isinstance(value, Mapping):
            raise InvalidArtefactManifestError(
                "Artifact manifest must be a mapping."
            )

        expected = {
            "artefact",
            "format",
            "format_version",
            "manifest_id",
            "materials",
            "metadata",
            "production",
        }

        _require_exact_fields(
            value,
            expected,
            name="artifact manifest",
        )

        if (
            value["format"]
            != ARTEFACT_MANIFEST_FORMAT
        ):
            raise InvalidArtefactManifestError(
                f"Unsupported manifest format: "
                f"{value['format']!r}."
            )

        artefact = value["artefact"]
        production = value["production"]

        if not isinstance(
            artefact,
            Mapping,
        ):
            raise InvalidArtefactManifestError(
                "artefact must be a mapping."
            )

        if not isinstance(
            production,
            Mapping,
        ):
            raise InvalidArtefactManifestError(
                "production must be a mapping."
            )

        _require_exact_fields(
            artefact,
            {
                "digest",
                "id",
                "location",
                "name",
                "size_bytes",
                "type",
                "version",
            },
            name="artefact",
        )

        _require_exact_fields(
            production,
            {
                "pipeline_id",
                "producer_workload_id",
                "produced_at_ns",
                "run_id",
            },
            name="production",
        )

        try:
            artefact_type = ArtefactType(
                artefact["type"]
            )
        except (TypeError, ValueError) as exc:
            raise InvalidArtefactManifestError(
                "Unsupported artifact type."
            ) from exc

        try:
            digest = Digest.parse(
                artefact["digest"]
            )
        except Exception as exc:
            raise InvalidArtefactManifestError(
                "Invalid artifact digest."
            ) from exc

        materials_value = value["materials"]

        if not isinstance(
            materials_value,
            list,
        ):
            raise InvalidArtefactManifestError(
                "materials must be a list."
            )

        materials = tuple(
            MaterialReference.from_dict(
                material
            )
            for material in materials_value
        )

        return cls(
            manifest_id=value["manifest_id"],
            artefact_id=artefact["id"],
            artefact_name=artefact["name"],
            artefact_type=artefact_type,
            artefact_version=artefact["version"],
            digest=digest,
            size_bytes=artefact["size_bytes"],
            produced_at_ns=(
                production["produced_at_ns"]
            ),
            producer_workload_id=(
                production[
                    "producer_workload_id"
                ]
            ),
            pipeline_id=(
                production["pipeline_id"]
            ),
            run_id=production["run_id"],
            location=artefact["location"],
            materials=materials,
            metadata=value["metadata"],
            format_version=value[
                "format_version"
            ],
        )

    @classmethod
    def from_json(
        cls,
        encoded: str | bytes,
    ) -> "ArtefactManifest":
        """Parse a JSON artifact manifest."""

        if isinstance(encoded, bytes):
            try:
                encoded = encoded.decode(
                    "utf-8",
                    errors="strict",
                )
            except UnicodeDecodeError as exc:
                raise InvalidArtefactManifestError(
                    "Manifest is not valid UTF-8."
                ) from exc

        if not isinstance(encoded, str):
            raise InvalidArtefactManifestError(
                "Encoded manifest must be string "
                "or UTF-8 bytes."
            )

        try:
            value = json.loads(
                encoded
            )
        except json.JSONDecodeError as exc:
            raise InvalidArtefactManifestError(
                "Manifest contains invalid JSON."
            ) from exc

        return cls.from_dict(
            value
        )


@dataclass(frozen=True, slots=True)
class ArtefactVerificationResult:
    """
    Result of comparing an artifact with its manifest.
    """

    status: ArtefactVerificationStatus

    artefact_id: str
    manifest_id: str

    expected_digest: Digest
    actual_digest: Digest | None

    expected_size_bytes: int
    actual_size_bytes: int | None

    reason: str

    verified_at_ns: int

    @property
    def valid(self) -> bool:
        return (
            self.status
            is ArtefactVerificationStatus.VALID
        )

    def require_valid(self) -> None:
        if not self.valid:
            raise ArtefactVerificationError(
                self.reason
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "actual_digest": (
                None
                if self.actual_digest is None
                else str(self.actual_digest)
            ),
            "actual_size_bytes": (
                self.actual_size_bytes
            ),
            "artefact_id": self.artefact_id,
            "expected_digest": str(
                self.expected_digest
            ),
            "expected_size_bytes": (
                self.expected_size_bytes
            ),
            "manifest_id": self.manifest_id,
            "reason": self.reason,
            "status": self.status.value,
            "verified_at_ns": (
                self.verified_at_ns
            ),
        }


def create_file_manifest(
    path: str | Path,
    *,
    artefact_id: str,
    artefact_name: str,
    artefact_type: ArtefactType,
    artefact_version: str,
    producer: WorkloadIdentity,
    pipeline_id: str | None = None,
    run_id: str | None = None,
    materials: tuple[MaterialReference, ...] = (),
    metadata: Mapping[str, str] | None = None,
    algorithm: HashAlgorithm | str = (
        HashAlgorithm.SHA256
    ),
) -> ArtefactManifest:
    """
    Hash a file and construct its provenance manifest.

    The file must be stable while hashing.
    Symbolic links are rejected by hashing.py.
    """

    if not isinstance(
        producer,
        WorkloadIdentity,
    ):
        raise InvalidArtefactManifestError(
            "producer must be WorkloadIdentity."
        )

    try:
        file_digest = hash_file(
            path,
            algorithm=algorithm,
            reject_symlinks=True,
            require_stable_file=True,
        )
    except (
        HashingError,
        OSError,
    ) as exc:
        raise ArtefactManifestError(
            f"Unable to hash artifact: {path!s}."
        ) from exc

    return ArtefactManifest(
        manifest_id=str(
            uuid.uuid4()
        ),
        artefact_id=artefact_id,
        artefact_name=artefact_name,
        artefact_type=artefact_type,
        artefact_version=artefact_version,
        digest=file_digest.digest,
        size_bytes=file_digest.size_bytes,
        produced_at_ns=time.time_ns(),
        producer_workload_id=(
            producer.workload_id
        ),
        pipeline_id=pipeline_id,
        run_id=run_id,
        location=str(
            Path(path)
        ),
        materials=materials,
        metadata=metadata,
    )


def material_from_file(
    path: str | Path,
    *,
    material_id: str,
    artefact_type: ArtefactType,
    algorithm: HashAlgorithm | str = (
        HashAlgorithm.SHA256
    ),
) -> MaterialReference:
    """
    Hash an input/material file and create a lineage reference.
    """

    try:
        result = hash_file(
            path,
            algorithm=algorithm,
            reject_symlinks=True,
            require_stable_file=True,
        )
    except (
        HashingError,
        OSError,
    ) as exc:
        raise ArtefactManifestError(
            f"Unable to hash material: {path!s}."
        ) from exc

    return MaterialReference(
        material_id=material_id,
        artefact_type=artefact_type,
        digest=result.digest,
        location=str(
            Path(path)
        ),
    )


def verify_artefact(
    path: str | Path,
    manifest: ArtefactManifest,
) -> ArtefactVerificationResult:
    """
    Verify an artifact file against its manifest.

    Verification checks:

    - file exists and can be safely hashed
    - file size matches
    - cryptographic digest matches
    """

    if not isinstance(
        manifest,
        ArtefactManifest,
    ):
        raise InvalidArtefactManifestError(
            "manifest must be ArtefactManifest."
        )

    artefact_path = Path(path)

    if not artefact_path.exists():
        return ArtefactVerificationResult(
            status=(
                ArtefactVerificationStatus.MISSING
            ),
            artefact_id=manifest.artefact_id,
            manifest_id=manifest.manifest_id,
            expected_digest=manifest.digest,
            actual_digest=None,
            expected_size_bytes=(
                manifest.size_bytes
            ),
            actual_size_bytes=None,
            reason="Artifact file does not exist.",
            verified_at_ns=time.time_ns(),
        )

    try:
        actual = hash_file(
            artefact_path,
            algorithm=(
                manifest.digest.algorithm
            ),
            reject_symlinks=True,
            require_stable_file=True,
        )
    except (
        HashingError,
        OSError,
    ):
        return ArtefactVerificationResult(
            status=(
                ArtefactVerificationStatus.ERROR
            ),
            artefact_id=manifest.artefact_id,
            manifest_id=manifest.manifest_id,
            expected_digest=manifest.digest,
            actual_digest=None,
            expected_size_bytes=(
                manifest.size_bytes
            ),
            actual_size_bytes=None,
            reason=(
                "Artifact could not be safely hashed."
            ),
            verified_at_ns=time.time_ns(),
        )

    if (
        actual.size_bytes
        != manifest.size_bytes
    ):
        return ArtefactVerificationResult(
            status=(
                ArtefactVerificationStatus.INVALID
            ),
            artefact_id=manifest.artefact_id,
            manifest_id=manifest.manifest_id,
            expected_digest=manifest.digest,
            actual_digest=actual.digest,
            expected_size_bytes=(
                manifest.size_bytes
            ),
            actual_size_bytes=(
                actual.size_bytes
            ),
            reason=(
                "Artifact size does not match manifest."
            ),
            verified_at_ns=time.time_ns(),
        )

    try:
        digest_matches = verify_file(
            artefact_path,
            manifest.digest,
            reject_symlinks=True,
            require_stable_file=True,
        )
    except (
        HashingError,
        OSError,
    ):
        digest_matches = False

    if not digest_matches:
        return ArtefactVerificationResult(
            status=(
                ArtefactVerificationStatus.INVALID
            ),
            artefact_id=manifest.artefact_id,
            manifest_id=manifest.manifest_id,
            expected_digest=manifest.digest,
            actual_digest=actual.digest,
            expected_size_bytes=(
                manifest.size_bytes
            ),
            actual_size_bytes=(
                actual.size_bytes
            ),
            reason=(
                "Artifact digest does not match manifest."
            ),
            verified_at_ns=time.time_ns(),
        )

    return ArtefactVerificationResult(
        status=ArtefactVerificationStatus.VALID,
        artefact_id=manifest.artefact_id,
        manifest_id=manifest.manifest_id,
        expected_digest=manifest.digest,
        actual_digest=actual.digest,
        expected_size_bytes=(
            manifest.size_bytes
        ),
        actual_size_bytes=(
            actual.size_bytes
        ),
        reason=(
            "Artifact matches manifest digest and size."
        ),
        verified_at_ns=time.time_ns(),
    )


def _validate_workload_id(
    value: str,
) -> None:
    if not isinstance(
        value,
        str,
    ):
        raise InvalidArtefactManifestError(
            "producer_workload_id must be a string."
        )

    try:
        workload_identity_from_id(
            value
        )
    except Exception as exc:
        raise InvalidArtefactManifestError(
            "producer_workload_id is not a valid "
            "OSHA workload identity."
        ) from exc


def _normalize_metadata(
    metadata: Mapping[str, str] | None,
) -> Mapping[str, str]:

    if metadata is None:
        return {}

    if not isinstance(
        metadata,
        Mapping,
    ):
        raise InvalidArtefactManifestError(
            "metadata must be a mapping."
        )

    if len(metadata) > MAX_METADATA_ITEMS:
        raise InvalidArtefactManifestError(
            f"metadata cannot exceed "
            f"{MAX_METADATA_ITEMS} entries."
        )

    normalized: dict[str, str] = {}

    for key, value in metadata.items():
        if (
            not isinstance(key, str)
            or not key.strip()
        ):
            raise InvalidArtefactManifestError(
                "Metadata keys must be "
                "non-empty strings."
            )

        if not isinstance(
            value,
            str,
        ):
            raise InvalidArtefactManifestError(
                "Metadata values must be strings."
            )

        normalized_key = (
            key.strip().lower()
        )

        normalized_value = (
            value.strip()
        )

        if (
            len(normalized_value)
            > MAX_METADATA_VALUE_LENGTH
        ):
            raise InvalidArtefactManifestError(
                "Metadata value exceeds "
                "maximum length."
            )

        normalized[
            normalized_key
        ] = normalized_value

    return normalized


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:

    actual = set(
        value.keys()
    )

    missing = expected - actual
    unknown = actual - expected

    if missing:
        raise InvalidArtefactManifestError(
            f"{name} is missing fields: "
            + ", ".join(
                sorted(missing)
            )
            + "."
        )

    if unknown:
        raise InvalidArtefactManifestError(
            f"{name} contains unknown fields: "
            + ", ".join(
                sorted(
                    str(item)
                    for item in unknown
                )
            )
            + "."
        )


def _validate_identifier(
    value: str,
    *,
    field: str,
) -> None:

    if (
        not isinstance(value, str)
        or not value
    ):
        raise InvalidArtefactManifestError(
            f"{field} must be a non-empty string."
        )

    if len(value) > MAX_ID_LENGTH:
        raise InvalidArtefactManifestError(
            f"{field} exceeds maximum length "
            f"of {MAX_ID_LENGTH}."
        )

    if (
        _ID_PATTERN.fullmatch(
            value
        )
        is None
    ):
        raise InvalidArtefactManifestError(
            f"{field} contains invalid characters."
        )


def _validate_text(
    value: str,
    *,
    field: str,
    max_length: int,
) -> None:

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise InvalidArtefactManifestError(
            f"{field} must be non-empty text."
        )

    if len(value) > max_length:
        raise InvalidArtefactManifestError(
            f"{field} exceeds maximum length "
            f"of {max_length}."
        )


__all__ = [
    "ARTEFACT_MANIFEST_FORMAT",
    "ARTEFACT_MANIFEST_VERSION",
    "ArtefactManifest",
    "ArtefactManifestError",
    "ArtefactType",
    "ArtefactVerificationError",
    "ArtefactVerificationResult",
    "ArtefactVerificationStatus",
    "InvalidArtefactManifestError",
    "MaterialReference",
    "create_file_manifest",
    "material_from_file",
    "verify_artefact",
]
