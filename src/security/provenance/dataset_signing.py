"""
Dataset-manifest signing and verification for OSHA.

This module cryptographically protects dataset provenance before data is
accepted into training, evaluation, or other trusted OSHA pipelines.

Security flow
-------------

Dataset preparation:

    dataset artifact
        |
        v
    ArtefactManifest
        |
        v
    canonical manifest
        |
        v
    trusted Ed25519 signature
        |
        v
    SignedDatasetManifest


Training/evaluation:

    SignedDatasetManifest
        |
        +--> trusted signing-key check
        +--> signature verification
        +--> producer/preparation-pipeline policy
        +--> dataset ID/version policy
        +--> provenance metadata requirements
        +--> artifact digest/size verification
        |
        v
    ACCEPT or REJECT


Important boundaries
--------------------
A valid dataset signature proves only that the trusted signer signed the
specified provenance manifest and that the dataset bytes match that manifest.

It does NOT prove:

- dataset accuracy
- absence of poisoned records
- absence of bias
- completeness
- legal ownership
- copyright permission
- training permission
- privacy compliance
- absence of personal information
- suitability for a particular model

Those concerns require separate governance, review, validation, and rights
management controls.

This module never parses or executes dataset contents during cryptographic
verification.
"""

from __future__ import annotations

import json
import time

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, Mapping

from ..crypto.hashing import canonical_json_bytes
from ..crypto.key_provider import KeyReference
from ..crypto.signatures import (
    InvalidSignatureEnvelopeError,
    SignatureEnvelope,
    SignatureError,
    SignatureService,
)
from .artefact_manifest import (
    ArtefactManifest,
    ArtefactType,
    ArtefactVerificationResult,
    InvalidArtefactManifestError,
    verify_artefact,
)


DATASET_SIGNATURE_FORMAT: Final[str] = (
    "osha-signed-dataset-manifest"
)

DATASET_SIGNATURE_VERSION: Final[int] = 1

DATASET_SIGNATURE_CONTEXT: Final[bytes] = (
    b"osha-dataset-manifest-v1"
)

MAX_SIGNED_DATASET_MANIFEST_SIZE: Final[int] = (
    16 * 1024 * 1024
)


class DatasetSigningError(Exception):
    """Base exception for OSHA dataset-signing failures."""


class InvalidSignedDatasetManifestError(
    DatasetSigningError,
    ValueError,
):
    """Signed dataset manifest is malformed."""


class DatasetSignatureVerificationError(
    DatasetSigningError,
):
    """Dataset-manifest signature verification failed."""


class DatasetArtefactVerificationError(
    DatasetSigningError,
):
    """Dataset artifact does not match the signed manifest."""


class DatasetTrustPolicyError(
    DatasetSigningError,
):
    """Dataset does not satisfy configured provenance policy."""


class DatasetVerificationStatus(str, Enum):
    """Complete dataset verification outcome."""

    VALID = "valid"

    INVALID_SIGNATURE = "invalid_signature"
    INVALID_ARTEFACT = "invalid_artefact"

    UNTRUSTED_SIGNER = "untrusted_signer"

    POLICY_REJECTED = "policy_rejected"

    MALFORMED = "malformed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SignedDatasetManifest:
    """
    Dataset provenance manifest and its digital signature.

    The signature covers the complete canonical manifest plus the signed
    envelope format and signing timestamp.
    """

    manifest: ArtefactManifest

    signature: SignatureEnvelope

    signed_at_ns: int

    format_version: int = (
        DATASET_SIGNATURE_VERSION
    )

    def __post_init__(self) -> None:
        if self.format_version != (
            DATASET_SIGNATURE_VERSION
        ):
            raise InvalidSignedDatasetManifestError(
                f"Unsupported signed-dataset version: "
                f"{self.format_version!r}."
            )

        if not isinstance(
            self.manifest,
            ArtefactManifest,
        ):
            raise InvalidSignedDatasetManifestError(
                "manifest must be ArtefactManifest."
            )

        if (
            self.manifest.artefact_type
            is not ArtefactType.DATASET
        ):
            raise InvalidSignedDatasetManifestError(
                "Signed dataset manifest must describe "
                "an artifact of type DATASET."
            )

        if not isinstance(
            self.signature,
            SignatureEnvelope,
        ):
            raise InvalidSignedDatasetManifestError(
                "signature must be SignatureEnvelope."
            )

        if (
            isinstance(self.signed_at_ns, bool)
            or not isinstance(
                self.signed_at_ns,
                int,
            )
            or self.signed_at_ns <= 0
        ):
            raise InvalidSignedDatasetManifestError(
                "signed_at_ns must be a positive integer."
            )

        if (
            self.signed_at_ns
            < self.manifest.produced_at_ns
        ):
            raise InvalidSignedDatasetManifestError(
                "Dataset manifest cannot be signed before "
                "its production timestamp."
            )

    @property
    def signing_key_reference(
        self,
    ) -> KeyReference:
        """Return the signing-key reference."""

        return self.signature.key_reference

    def unsigned_dict(
        self,
    ) -> dict[str, Any]:
        """
        Return exactly the content covered by the signature.
        """

        return {
            "format": (
                DATASET_SIGNATURE_FORMAT
            ),
            "format_version": (
                self.format_version
            ),
            "manifest": (
                self.manifest.as_dict()
            ),
            "signed_at_ns": (
                self.signed_at_ns
            ),
        }

    def signing_bytes(
        self,
    ) -> bytes:
        """Return canonical signed bytes."""

        return canonical_json_bytes(
            self.unsigned_dict()
        )

    def as_dict(
        self,
    ) -> dict[str, Any]:
        """Return JSON-compatible signed dataset manifest."""

        return {
            **self.unsigned_dict(),
            "signature": (
                self.signature.as_dict()
            ),
        }

    def to_json(
        self,
    ) -> str:
        """Serialize signed dataset manifest."""

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
    ) -> "SignedDatasetManifest":
        """
        Parse and strictly validate a signed dataset manifest.
        """

        if not isinstance(
            value,
            Mapping,
        ):
            raise InvalidSignedDatasetManifestError(
                "Signed dataset manifest must be a mapping."
            )

        expected_fields = {
            "format",
            "format_version",
            "manifest",
            "signature",
            "signed_at_ns",
        }

        _require_exact_fields(
            value,
            expected_fields,
            object_name=(
                "signed dataset manifest"
            ),
        )

        if (
            value["format"]
            != DATASET_SIGNATURE_FORMAT
        ):
            raise InvalidSignedDatasetManifestError(
                f"Unsupported signed-dataset format: "
                f"{value['format']!r}."
            )

        format_version = (
            value["format_version"]
        )

        if (
            isinstance(format_version, bool)
            or not isinstance(
                format_version,
                int,
            )
        ):
            raise InvalidSignedDatasetManifestError(
                "format_version must be an integer."
            )

        signed_at_ns = value[
            "signed_at_ns"
        ]

        if (
            isinstance(signed_at_ns, bool)
            or not isinstance(
                signed_at_ns,
                int,
            )
        ):
            raise InvalidSignedDatasetManifestError(
                "signed_at_ns must be an integer."
            )

        try:
            manifest = (
                ArtefactManifest.from_dict(
                    value["manifest"]
                )
            )

        except InvalidArtefactManifestError as exc:
            raise InvalidSignedDatasetManifestError(
                "Embedded artifact manifest is invalid."
            ) from exc

        try:
            signature = (
                SignatureEnvelope.from_dict(
                    value["signature"]
                )
            )

        except InvalidSignatureEnvelopeError as exc:
            raise InvalidSignedDatasetManifestError(
                "Embedded signature envelope is invalid."
            ) from exc

        return cls(
            manifest=manifest,
            signature=signature,
            signed_at_ns=signed_at_ns,
            format_version=format_version,
        )

    @classmethod
    def from_json(
        cls,
        encoded: str | bytes,
    ) -> "SignedDatasetManifest":
        """Parse JSON encoded signed dataset manifest."""

        if isinstance(
            encoded,
            bytes,
        ):
            try:
                encoded = encoded.decode(
                    "utf-8",
                    errors="strict",
                )

            except UnicodeDecodeError as exc:
                raise InvalidSignedDatasetManifestError(
                    "Signed dataset manifest is not "
                    "valid UTF-8."
                ) from exc

        if not isinstance(
            encoded,
            str,
        ):
            raise InvalidSignedDatasetManifestError(
                "Encoded signed dataset manifest must "
                "be string or UTF-8 bytes."
            )

        if len(
            encoded.encode(
                "utf-8"
            )
        ) > MAX_SIGNED_DATASET_MANIFEST_SIZE:
            raise InvalidSignedDatasetManifestError(
                "Signed dataset manifest exceeds "
                "supported size."
            )

        try:
            value = json.loads(
                encoded
            )

        except json.JSONDecodeError as exc:
            raise InvalidSignedDatasetManifestError(
                "Signed dataset manifest contains "
                "invalid JSON."
            ) from exc

        return cls.from_dict(
            value
        )


@dataclass(frozen=True, slots=True)
class DatasetTrustPolicy:
    """
    Dataset provenance expectations.

    trusted_signing_keys:
        Signing key references trusted by the consumer.

    allowed_producers:
        OSHA workload IDs permitted to produce datasets.

    allowed_pipeline_ids:
        Dataset preparation pipelines trusted by this consumer.

    allowed_dataset_ids:
        Dataset IDs permitted by this training/evaluation pipeline.

    allowed_versions:
        Dataset versions accepted by this consumer.

    required_metadata_keys:
        Provenance/governance metadata that must exist.

        Examples may include organization-specific keys such as:

            source
            platform
            rights_record
            license
            review_status

        This class verifies only that required metadata fields are present
        and non-empty. It does NOT validate whether a rights claim is true.

    require_run_id:
        Require preparation-run identity.

    require_pipeline_id:
        Require preparation-pipeline identity.

    minimum_materials:
        Minimum number of lineage inputs required.
    """

    trusted_signing_keys: frozenset[
        KeyReference
    ]

    allowed_producers: frozenset[
        str
    ] = frozenset()

    allowed_pipeline_ids: frozenset[
        str
    ] = frozenset()

    allowed_dataset_ids: frozenset[
        str
    ] = frozenset()

    allowed_versions: frozenset[
        str
    ] = frozenset()

    required_metadata_keys: frozenset[
        str
    ] = frozenset()

    minimum_materials: int = 0

    require_run_id: bool = True
    require_pipeline_id: bool = True

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.trusted_signing_keys,
            frozenset,
        ):
            raise DatasetTrustPolicyError(
                "trusted_signing_keys must be frozenset."
            )

        if not self.trusted_signing_keys:
            raise DatasetTrustPolicyError(
                "At least one trusted dataset-signing "
                "key is required."
            )

        for reference in (
            self.trusted_signing_keys
        ):
            if not isinstance(
                reference,
                KeyReference,
            ):
                raise DatasetTrustPolicyError(
                    "Every trusted signing key must "
                    "be KeyReference."
                )

        string_sets = (
            (
                "allowed_producers",
                self.allowed_producers,
            ),
            (
                "allowed_pipeline_ids",
                self.allowed_pipeline_ids,
            ),
            (
                "allowed_dataset_ids",
                self.allowed_dataset_ids,
            ),
            (
                "allowed_versions",
                self.allowed_versions,
            ),
            (
                "required_metadata_keys",
                self.required_metadata_keys,
            ),
        )

        for field_name, values in (
            string_sets
        ):
            if not isinstance(
                values,
                frozenset,
            ):
                raise DatasetTrustPolicyError(
                    f"{field_name} must be frozenset."
                )

            for value in values:
                if (
                    not isinstance(
                        value,
                        str,
                    )
                    or not value
                ):
                    raise DatasetTrustPolicyError(
                        f"{field_name} must contain "
                        "non-empty strings."
                    )

        if (
            isinstance(
                self.minimum_materials,
                bool,
            )
            or not isinstance(
                self.minimum_materials,
                int,
            )
            or self.minimum_materials < 0
        ):
            raise DatasetTrustPolicyError(
                "minimum_materials must be a "
                "non-negative integer."
            )

        if not isinstance(
            self.require_run_id,
            bool,
        ):
            raise DatasetTrustPolicyError(
                "require_run_id must be boolean."
            )

        if not isinstance(
            self.require_pipeline_id,
            bool,
        ):
            raise DatasetTrustPolicyError(
                "require_pipeline_id must be boolean."
            )

    def evaluate(
        self,
        signed_manifest: SignedDatasetManifest,
    ) -> tuple[
        bool,
        str,
    ]:
        """
        Evaluate dataset provenance expectations.
        """

        if (
            signed_manifest.signing_key_reference
            not in self.trusted_signing_keys
        ):
            return (
                False,
                "Dataset was not signed by a trusted "
                "signing-key reference.",
            )

        manifest = (
            signed_manifest.manifest
        )

        if (
            self.allowed_producers
            and manifest.producer_workload_id
            not in self.allowed_producers
        ):
            return (
                False,
                "Dataset producer is not trusted "
                "by policy.",
            )

        if (
            self.allowed_pipeline_ids
            and manifest.pipeline_id
            not in self.allowed_pipeline_ids
        ):
            return (
                False,
                "Dataset preparation pipeline is "
                "not trusted by policy.",
            )

        if (
            self.allowed_dataset_ids
            and manifest.artefact_id
            not in self.allowed_dataset_ids
        ):
            return (
                False,
                "Dataset ID is not allowed "
                "by policy.",
            )

        if (
            self.allowed_versions
            and manifest.artefact_version
            not in self.allowed_versions
        ):
            return (
                False,
                "Dataset version is not allowed "
                "by policy.",
            )

        if (
            self.require_run_id
            and manifest.run_id is None
        ):
            return (
                False,
                "Dataset manifest does not identify "
                "the preparation run.",
            )

        if (
            self.require_pipeline_id
            and manifest.pipeline_id is None
        ):
            return (
                False,
                "Dataset manifest does not identify "
                "the preparation pipeline.",
            )

        if (
            len(
                manifest.materials
            )
            < self.minimum_materials
        ):
            return (
                False,
                "Dataset manifest does not contain "
                "the required material lineage.",
            )

        metadata = (
            manifest.metadata or {}
        )

        missing_metadata = sorted(
            key
            for key
            in self.required_metadata_keys
            if (
                key not in metadata
                or not metadata[key]
            )
        )

        if missing_metadata:
            return (
                False,
                "Dataset manifest is missing required "
                "metadata fields: "
                + ", ".join(
                    missing_metadata
                )
                + ".",
            )

        return (
            True,
            "Dataset satisfies configured "
            "provenance policy.",
        )


@dataclass(frozen=True, slots=True)
class DatasetVerificationResult:
    """
    Result of complete dataset verification.
    """

    status: DatasetVerificationStatus

    dataset_id: str | None
    dataset_version: str | None
    manifest_id: str | None

    signing_key_reference: (
        KeyReference | None
    )

    signature_valid: bool
    artefact_valid: bool
    policy_valid: bool

    reason: str

    verified_at_ns: int

    artefact_verification: (
        ArtefactVerificationResult
        | None
    ) = None

    @property
    def valid(
        self,
    ) -> bool:
        return (
            self.status
            is DatasetVerificationStatus.VALID
            and self.signature_valid
            and self.artefact_valid
            and self.policy_valid
        )

    def require_valid(
        self,
    ) -> None:
        """
        Fail closed unless every dataset verification check succeeds.
        """

        if not self.valid:
            raise DatasetSignatureVerificationError(
                self.reason
            )

    def as_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "artefact_valid": (
                self.artefact_valid
            ),
            "artefact_verification": (
                None
                if self.artefact_verification
                is None
                else self
                .artefact_verification
                .as_dict()
            ),
            "dataset_id": self.dataset_id,
            "dataset_version": (
                self.dataset_version
            ),
            "format_version": (
                DATASET_SIGNATURE_VERSION
            ),
            "manifest_id": self.manifest_id,
            "policy_valid": self.policy_valid,
            "reason": self.reason,
            "signature_valid": (
                self.signature_valid
            ),
            "signing_key_reference": (
                None
                if self.signing_key_reference
                is None
                else str(
                    self.signing_key_reference
                )
            ),
            "status": self.status.value,
            "verified_at_ns": (
                self.verified_at_ns
            ),
        }


class DatasetSigningService:
    """
    Sign dataset artifact manifests.

    signature_service must possess a private signing key.
    """

    def __init__(
        self,
        signature_service: SignatureService,
    ) -> None:
        if not isinstance(
            signature_service,
            SignatureService,
        ):
            raise TypeError(
                "signature_service must be "
                "SignatureService."
            )

        if not (
            signature_service.can_sign
        ):
            raise DatasetSigningError(
                "Dataset signing requires a "
                "SignatureService with a private key."
            )

        self._signature_service = (
            signature_service
        )

    def sign_manifest(
        self,
        manifest: ArtefactManifest,
        *,
        signed_at_ns: int | None = None,
    ) -> SignedDatasetManifest:
        """
        Sign one dataset artifact manifest.
        """

        if not isinstance(
            manifest,
            ArtefactManifest,
        ):
            raise DatasetSigningError(
                "manifest must be ArtefactManifest."
            )

        if (
            manifest.artefact_type
            is not ArtefactType.DATASET
        ):
            raise DatasetSigningError(
                "Only DATASET artifact manifests "
                "may be signed by "
                "DatasetSigningService."
            )

        resolved_time = (
            time.time_ns()
            if signed_at_ns is None
            else signed_at_ns
        )

        if (
            isinstance(
                resolved_time,
                bool,
            )
            or not isinstance(
                resolved_time,
                int,
            )
            or resolved_time <= 0
        ):
            raise DatasetSigningError(
                "signed_at_ns must be a "
                "positive integer."
            )

        if (
            resolved_time
            < manifest.produced_at_ns
        ):
            raise DatasetSigningError(
                "Dataset cannot be signed before "
                "its production timestamp."
            )

        unsigned = {
            "format": (
                DATASET_SIGNATURE_FORMAT
            ),
            "format_version": (
                DATASET_SIGNATURE_VERSION
            ),
            "manifest": manifest.as_dict(),
            "signed_at_ns": resolved_time,
        }

        try:
            signature = (
                self._signature_service.sign(
                    canonical_json_bytes(
                        unsigned
                    ),
                    context=(
                        DATASET_SIGNATURE_CONTEXT
                    ),
                )
            )

        except SignatureError as exc:
            raise DatasetSigningError(
                "Unable to sign dataset manifest."
            ) from exc

        return SignedDatasetManifest(
            manifest=manifest,
            signature=signature,
            signed_at_ns=resolved_time,
        )


class DatasetVerificationService:
    """
    Verify signed datasets before trusted use.

    This should be called before a dataset enters a training/evaluation
    pipeline that requires trusted provenance.
    """

    def __init__(
        self,
        *,
        signature_service: SignatureService,
        trust_policy: DatasetTrustPolicy,
    ) -> None:
        if not isinstance(
            signature_service,
            SignatureService,
        ):
            raise TypeError(
                "signature_service must be "
                "SignatureService."
            )

        if not isinstance(
            trust_policy,
            DatasetTrustPolicy,
        ):
            raise TypeError(
                "trust_policy must be "
                "DatasetTrustPolicy."
            )

        self._signature_service = (
            signature_service
        )

        self._trust_policy = (
            trust_policy
        )

    def verify(
        self,
        dataset_path: str | Path,
        signed_manifest: SignedDatasetManifest,
    ) -> DatasetVerificationResult:
        """
        Perform complete dataset verification.

        Order:

        1. validate signed-manifest object
        2. verify trusted signing-key reference
        3. verify digital signature
        4. evaluate dataset provenance policy
        5. verify actual dataset digest and size
        """

        if not isinstance(
            signed_manifest,
            SignedDatasetManifest,
        ):
            return _verification_result(
                status=(
                    DatasetVerificationStatus
                    .MALFORMED
                ),
                reason=(
                    "signed_manifest must be "
                    "SignedDatasetManifest."
                ),
            )

        signing_reference = (
            signed_manifest
            .signing_key_reference
        )

        if (
            signing_reference
            not in self._trust_policy
            .trusted_signing_keys
        ):
            return _verification_result(
                status=(
                    DatasetVerificationStatus
                    .UNTRUSTED_SIGNER
                ),
                reason=(
                    "Dataset signing-key reference "
                    "is not trusted."
                ),
                signed_manifest=(
                    signed_manifest
                ),
            )

        try:
            signature_valid = (
                self._signature_service
                .verify(
                    signed_manifest
                    .signing_bytes(),
                    signed_manifest.signature,
                    context=(
                        DATASET_SIGNATURE_CONTEXT
                    ),
                )
            )

        except SignatureError:
            signature_valid = False

        if not signature_valid:
            return _verification_result(
                status=(
                    DatasetVerificationStatus
                    .INVALID_SIGNATURE
                ),
                reason=(
                    "Dataset-manifest signature "
                    "verification failed."
                ),
                signed_manifest=(
                    signed_manifest
                ),
                signature_valid=False,
            )

        policy_valid, policy_reason = (
            self._trust_policy.evaluate(
                signed_manifest
            )
        )

        if not policy_valid:
            return _verification_result(
                status=(
                    DatasetVerificationStatus
                    .POLICY_REJECTED
                ),
                reason=policy_reason,
                signed_manifest=(
                    signed_manifest
                ),
                signature_valid=True,
                policy_valid=False,
            )

        artefact_verification = (
            verify_artefact(
                dataset_path,
                signed_manifest.manifest,
            )
        )

        if not (
            artefact_verification.valid
        ):
            return _verification_result(
                status=(
                    DatasetVerificationStatus
                    .INVALID_ARTEFACT
                ),
                reason=(
                    "Dataset artifact does not match "
                    "the signed manifest."
                ),
                signed_manifest=(
                    signed_manifest
                ),
                signature_valid=True,
                policy_valid=True,
                artefact_valid=False,
                artefact_verification=(
                    artefact_verification
                ),
            )

        return _verification_result(
            status=(
                DatasetVerificationStatus.VALID
            ),
            reason=(
                "Dataset signature, provenance "
                "policy, digest, and size verified "
                "successfully."
            ),
            signed_manifest=(
                signed_manifest
            ),
            signature_valid=True,
            artefact_valid=True,
            policy_valid=True,
            artefact_verification=(
                artefact_verification
            ),
        )

    def require_verified(
        self,
        dataset_path: str | Path,
        signed_manifest: SignedDatasetManifest,
    ) -> DatasetVerificationResult:
        """
        Require full dataset verification before trusted use.
        """

        result = self.verify(
            dataset_path,
            signed_manifest,
        )

        result.require_valid()

        return result


def _verification_result(
    *,
    status: DatasetVerificationStatus,
    reason: str,
    signed_manifest: (
        SignedDatasetManifest | None
    ) = None,
    signature_valid: bool = False,
    artefact_valid: bool = False,
    policy_valid: bool = False,
    artefact_verification: (
        ArtefactVerificationResult
        | None
    ) = None,
) -> DatasetVerificationResult:

    manifest = (
        None
        if signed_manifest is None
        else signed_manifest.manifest
    )

    return DatasetVerificationResult(
        status=status,
        dataset_id=(
            None
            if manifest is None
            else manifest.artefact_id
        ),
        dataset_version=(
            None
            if manifest is None
            else manifest.artefact_version
        ),
        manifest_id=(
            None
            if manifest is None
            else manifest.manifest_id
        ),
        signing_key_reference=(
            None
            if signed_manifest is None
            else signed_manifest
            .signing_key_reference
        ),
        signature_valid=signature_valid,
        artefact_valid=artefact_valid,
        policy_valid=policy_valid,
        reason=reason,
        verified_at_ns=time.time_ns(),
        artefact_verification=(
            artefact_verification
        ),
    )


def _require_exact_fields(
    value: Mapping[str, Any],
    expected_fields: set[str],
    *,
    object_name: str,
) -> None:

    actual_fields = set(
        value.keys()
    )

    missing = (
        expected_fields - actual_fields
    )

    unknown = (
        actual_fields - expected_fields
    )

    if missing:
        raise InvalidSignedDatasetManifestError(
            f"{object_name} is missing fields: "
            + ", ".join(
                sorted(missing)
            )
            + "."
        )

    if unknown:
        raise InvalidSignedDatasetManifestError(
            f"{object_name} contains "
            "unknown fields: "
            + ", ".join(
                sorted(
                    str(item)
                    for item in unknown
                )
            )
            + "."
        )


__all__ = [
    "DATASET_SIGNATURE_CONTEXT",
    "DATASET_SIGNATURE_FORMAT",
    "DATASET_SIGNATURE_VERSION",
    "DatasetArtefactVerificationError",
    "DatasetSignatureVerificationError",
    "DatasetSigningError",
    "DatasetSigningService",
    "DatasetTrustPolicy",
    "DatasetTrustPolicyError",
    "DatasetVerificationResult",
    "DatasetVerificationService",
    "DatasetVerificationStatus",
    "InvalidSignedDatasetManifestError",
    "SignedDatasetManifest",
]
