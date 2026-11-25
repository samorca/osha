"""
Model-manifest signing and verification for OSHA.

This module connects:

    artefact_manifest.py
        -> records model digest, size, version, producer, and lineage

    signatures.py
        -> signs and verifies the canonical manifest

    model_signing.py
        -> requires both a trusted signature and matching model artifact

Security flow
-------------

Training pipeline:

    model file
        |
        v
    ArtefactManifest
        |
        v
    canonical manifest bytes
        |
        v
    Ed25519 signature
        |
        v
    SignedModelManifest

Runtime/model registry:

    SignedModelManifest
        |
        +--> validate manifest type
        +--> verify expected signer identity
        +--> verify trusted signature
        +--> verify model digest
        +--> verify model size
        +--> verify expected artifact/version/producer constraints
        |
        v
    ACCEPT or REJECT

Security boundaries
-------------------
1. Signing a manifest does not prove model quality or safety.
2. Verification does not deserialize or execute the model.
3. Model loading must occur only after successful verification.
4. A public key embedded in a signature envelope is not trusted by itself.
5. The verifier must use a SignatureService configured with a trusted key.
6. Model signatures do not establish dataset usage rights or licensing.
7. Model files are never loaded with pickle or another executable format here.
8. The signing key should eventually be managed through an approved KMS/HSM.
"""

from __future__ import annotations

import json
import time

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, Mapping

from ..crypto.hashing import (
    Digest,
    canonical_json_bytes,
)
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


MODEL_SIGNATURE_FORMAT: Final[str] = (
    "osha-signed-model-manifest"
)

MODEL_SIGNATURE_VERSION: Final[int] = 1

MODEL_SIGNATURE_CONTEXT: Final[bytes] = (
    b"osha-model-manifest-v1"
)

MAX_SIGNED_MANIFEST_SIZE: Final[int] = (
    16 * 1024 * 1024
)


class ModelSigningError(Exception):
    """Base exception for OSHA model-signing failures."""


class InvalidSignedModelManifestError(
    ModelSigningError,
    ValueError,
):
    """Signed model manifest is malformed or unsupported."""


class ModelSignatureVerificationError(
    ModelSigningError,
):
    """Model-manifest signature verification failed."""


class ModelArtefactVerificationError(
    ModelSigningError,
):
    """Model artifact does not match the signed manifest."""


class ModelTrustPolicyError(
    ModelSigningError,
):
    """Model does not satisfy configured trust expectations."""


class ModelVerificationStatus(str, Enum):
    """Outcome of complete model verification."""

    VALID = "valid"
    INVALID_SIGNATURE = "invalid_signature"
    INVALID_ARTEFACT = "invalid_artefact"
    UNTRUSTED_SIGNER = "untrusted_signer"
    POLICY_REJECTED = "policy_rejected"
    MALFORMED = "malformed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SignedModelManifest:
    """
    Model provenance manifest plus digital signature.

    The signature covers:

    - envelope format
    - envelope format version
    - complete canonical artifact manifest

    The model file is bound through the digest and size stored in the signed
    manifest.
    """

    manifest: ArtefactManifest
    signature: SignatureEnvelope
    signed_at_ns: int
    format_version: int = MODEL_SIGNATURE_VERSION

    def __post_init__(self) -> None:
        if (
            self.format_version
            != MODEL_SIGNATURE_VERSION
        ):
            raise InvalidSignedModelManifestError(
                f"Unsupported signed-model version: "
                f"{self.format_version!r}."
            )

        if not isinstance(
            self.manifest,
            ArtefactManifest,
        ):
            raise InvalidSignedModelManifestError(
                "manifest must be ArtefactManifest."
            )

        if (
            self.manifest.artefact_type
            is not ArtefactType.MODEL
        ):
            raise InvalidSignedModelManifestError(
                "Signed model manifest must describe "
                "an artifact of type MODEL."
            )

        if not isinstance(
            self.signature,
            SignatureEnvelope,
        ):
            raise InvalidSignedModelManifestError(
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
            raise InvalidSignedModelManifestError(
                "signed_at_ns must be a positive integer."
            )

        if (
            self.signed_at_ns
            < self.manifest.produced_at_ns
        ):
            raise InvalidSignedModelManifestError(
                "Manifest cannot be signed before the "
                "artifact production time."
            )

    @property
    def signing_key_reference(
        self,
    ) -> KeyReference:
        """Return the signing-key reference."""

        return self.signature.key_reference

    def unsigned_dict(self) -> dict[str, Any]:
        """
        Return exactly the object covered by the signature.
        """

        return {
            "format": MODEL_SIGNATURE_FORMAT,
            "format_version": self.format_version,
            "manifest": self.manifest.as_dict(),
            "signed_at_ns": self.signed_at_ns,
        }

    def signing_bytes(self) -> bytes:
        """
        Return deterministic bytes covered by the signature.
        """

        return canonical_json_bytes(
            self.unsigned_dict()
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible signed manifest."""

        return {
            **self.unsigned_dict(),
            "signature": self.signature.as_dict(),
        }

    def to_json(self) -> str:
        """Serialize signed model manifest as compact JSON."""

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
    ) -> "SignedModelManifest":
        """Parse and strictly validate a signed model manifest."""

        if not isinstance(value, Mapping):
            raise InvalidSignedModelManifestError(
                "Signed model manifest must be a mapping."
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
            object_name="signed model manifest",
        )

        if value["format"] != MODEL_SIGNATURE_FORMAT:
            raise InvalidSignedModelManifestError(
                f"Unsupported signed-model format: "
                f"{value['format']!r}."
            )

        format_version = value["format_version"]

        if (
            isinstance(format_version, bool)
            or not isinstance(
                format_version,
                int,
            )
        ):
            raise InvalidSignedModelManifestError(
                "format_version must be an integer."
            )

        signed_at_ns = value["signed_at_ns"]

        if (
            isinstance(signed_at_ns, bool)
            or not isinstance(
                signed_at_ns,
                int,
            )
        ):
            raise InvalidSignedModelManifestError(
                "signed_at_ns must be an integer."
            )

        try:
            manifest = ArtefactManifest.from_dict(
                value["manifest"]
            )
        except InvalidArtefactManifestError as exc:
            raise InvalidSignedModelManifestError(
                "Embedded artifact manifest is invalid."
            ) from exc

        try:
            signature = SignatureEnvelope.from_dict(
                value["signature"]
            )
        except InvalidSignatureEnvelopeError as exc:
            raise InvalidSignedModelManifestError(
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
    ) -> "SignedModelManifest":
        """Parse a JSON-encoded signed model manifest."""

        if isinstance(encoded, bytes):
            try:
                encoded = encoded.decode(
                    "utf-8",
                    errors="strict",
                )
            except UnicodeDecodeError as exc:
                raise InvalidSignedModelManifestError(
                    "Signed model manifest is not "
                    "valid UTF-8."
                ) from exc

        if not isinstance(encoded, str):
            raise InvalidSignedModelManifestError(
                "Encoded signed manifest must be "
                "string or UTF-8 bytes."
            )

        if len(encoded.encode("utf-8")) > (
            MAX_SIGNED_MANIFEST_SIZE
        ):
            raise InvalidSignedModelManifestError(
                "Signed model manifest exceeds "
                "the supported size."
            )

        try:
            value = json.loads(
                encoded
            )
        except json.JSONDecodeError as exc:
            raise InvalidSignedModelManifestError(
                "Signed model manifest contains "
                "invalid JSON."
            ) from exc

        return cls.from_dict(
            value
        )


@dataclass(frozen=True, slots=True)
class ModelTrustPolicy:
    """
    Expectations that a model must satisfy before acceptance.

    Empty optional sets mean that the field is unrestricted.

    trusted_signing_keys:
        Signing-key references trusted by this verifier.

    allowed_producers:
        OSHA workload IDs authorized as model producers.

    allowed_pipeline_ids:
        Training-pipeline IDs accepted by this runtime/registry.

    allowed_model_ids:
        Model artifact IDs accepted by this consumer.

    allowed_versions:
        Model versions accepted by this consumer.

    minimum_materials:
        Minimum lineage/material entries required in the manifest.

    require_run_id:
        Require a training-run identifier.

    require_pipeline_id:
        Require a training-pipeline identifier.
    """

    trusted_signing_keys: frozenset[KeyReference]

    allowed_producers: frozenset[str] = frozenset()
    allowed_pipeline_ids: frozenset[str] = frozenset()
    allowed_model_ids: frozenset[str] = frozenset()
    allowed_versions: frozenset[str] = frozenset()

    minimum_materials: int = 0

    require_run_id: bool = True
    require_pipeline_id: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.trusted_signing_keys,
            frozenset,
        ):
            raise ModelTrustPolicyError(
                "trusted_signing_keys must be frozenset."
            )

        if not self.trusted_signing_keys:
            raise ModelTrustPolicyError(
                "At least one trusted signing key "
                "is required."
            )

        for reference in self.trusted_signing_keys:
            if not isinstance(
                reference,
                KeyReference,
            ):
                raise ModelTrustPolicyError(
                    "Every trusted signing key must be "
                    "KeyReference."
                )

        for field_name, values in (
            (
                "allowed_producers",
                self.allowed_producers,
            ),
            (
                "allowed_pipeline_ids",
                self.allowed_pipeline_ids,
            ),
            (
                "allowed_model_ids",
                self.allowed_model_ids,
            ),
            (
                "allowed_versions",
                self.allowed_versions,
            ),
        ):
            if not isinstance(
                values,
                frozenset,
            ):
                raise ModelTrustPolicyError(
                    f"{field_name} must be frozenset."
                )

            for value in values:
                if (
                    not isinstance(value, str)
                    or not value
                ):
                    raise ModelTrustPolicyError(
                        f"{field_name} must contain "
                        "non-empty strings."
                    )

        if (
            isinstance(self.minimum_materials, bool)
            or not isinstance(
                self.minimum_materials,
                int,
            )
            or self.minimum_materials < 0
        ):
            raise ModelTrustPolicyError(
                "minimum_materials must be a "
                "non-negative integer."
            )

        if not isinstance(
            self.require_run_id,
            bool,
        ):
            raise ModelTrustPolicyError(
                "require_run_id must be boolean."
            )

        if not isinstance(
            self.require_pipeline_id,
            bool,
        ):
            raise ModelTrustPolicyError(
                "require_pipeline_id must be boolean."
            )

    def evaluate(
        self,
        signed_manifest: SignedModelManifest,
    ) -> tuple[bool, str]:
        """Evaluate configured model trust expectations."""

        if (
            signed_manifest.signing_key_reference
            not in self.trusted_signing_keys
        ):
            return (
                False,
                "Model was not signed by a trusted "
                "signing-key reference.",
            )

        manifest = signed_manifest.manifest

        if (
            self.allowed_producers
            and manifest.producer_workload_id
            not in self.allowed_producers
        ):
            return (
                False,
                "Model producer is not trusted by policy.",
            )

        if (
            self.allowed_pipeline_ids
            and manifest.pipeline_id
            not in self.allowed_pipeline_ids
        ):
            return (
                False,
                "Training pipeline is not trusted by policy.",
            )

        if (
            self.allowed_model_ids
            and manifest.artefact_id
            not in self.allowed_model_ids
        ):
            return (
                False,
                "Model artifact ID is not allowed by policy.",
            )

        if (
            self.allowed_versions
            and manifest.artefact_version
            not in self.allowed_versions
        ):
            return (
                False,
                "Model version is not allowed by policy.",
            )

        if (
            self.require_run_id
            and manifest.run_id is None
        ):
            return (
                False,
                "Model manifest does not identify "
                "the training run.",
            )

        if (
            self.require_pipeline_id
            and manifest.pipeline_id is None
        ):
            return (
                False,
                "Model manifest does not identify "
                "the training pipeline.",
            )

        if (
            len(manifest.materials)
            < self.minimum_materials
        ):
            return (
                False,
                "Model manifest does not contain "
                "the required material lineage.",
            )

        return (
            True,
            "Model satisfies configured trust policy.",
        )


@dataclass(frozen=True, slots=True)
class ModelVerificationResult:
    """
    Result of complete signed-model verification.
    """

    status: ModelVerificationStatus

    model_id: str | None
    model_version: str | None
    manifest_id: str | None

    signing_key_reference: KeyReference | None

    signature_valid: bool
    artefact_valid: bool
    policy_valid: bool

    reason: str
    verified_at_ns: int

    artefact_verification: (
        ArtefactVerificationResult | None
    ) = None

    @property
    def valid(self) -> bool:
        return (
            self.status
            is ModelVerificationStatus.VALID
            and self.signature_valid
            and self.artefact_valid
            and self.policy_valid
        )

    def require_valid(self) -> None:
        """Raise unless every model trust check succeeded."""

        if not self.valid:
            raise ModelSignatureVerificationError(
                self.reason
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "artefact_valid": self.artefact_valid,
            "artefact_verification": (
                None
                if self.artefact_verification is None
                else self.artefact_verification.as_dict()
            ),
            "format_version": MODEL_SIGNATURE_VERSION,
            "manifest_id": self.manifest_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "policy_valid": self.policy_valid,
            "reason": self.reason,
            "signature_valid": self.signature_valid,
            "signing_key_reference": (
                None
                if self.signing_key_reference is None
                else str(
                    self.signing_key_reference
                )
            ),
            "status": self.status.value,
            "verified_at_ns": self.verified_at_ns,
        }


class ModelSigningService:
    """
    Sign model artifact manifests.

    The supplied SignatureService must contain a private signing key.
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
                "signature_service must be SignatureService."
            )

        if not signature_service.can_sign:
            raise ModelSigningError(
                "Model signing requires a SignatureService "
                "with a private key."
            )

        self._signature_service = signature_service

    def sign_manifest(
        self,
        manifest: ArtefactManifest,
        *,
        signed_at_ns: int | None = None,
    ) -> SignedModelManifest:
        """
        Digitally sign a model artifact manifest.
        """

        if not isinstance(
            manifest,
            ArtefactManifest,
        ):
            raise ModelSigningError(
                "manifest must be ArtefactManifest."
            )

        if (
            manifest.artefact_type
            is not ArtefactType.MODEL
        ):
            raise ModelSigningError(
                "Only MODEL artifact manifests may be "
                "signed by ModelSigningService."
            )

        resolved_time = (
            time.time_ns()
            if signed_at_ns is None
            else signed_at_ns
        )

        if (
            isinstance(resolved_time, bool)
            or not isinstance(
                resolved_time,
                int,
            )
            or resolved_time <= 0
        ):
            raise ModelSigningError(
                "signed_at_ns must be a positive integer."
            )

        unsigned = {
            "format": MODEL_SIGNATURE_FORMAT,
            "format_version": MODEL_SIGNATURE_VERSION,
            "manifest": manifest.as_dict(),
            "signed_at_ns": resolved_time,
        }

        try:
            signature = self._signature_service.sign(
                canonical_json_bytes(
                    unsigned
                ),
                context=MODEL_SIGNATURE_CONTEXT,
            )
        except SignatureError as exc:
            raise ModelSigningError(
                "Unable to sign model manifest."
            ) from exc

        return SignedModelManifest(
            manifest=manifest,
            signature=signature,
            signed_at_ns=resolved_time,
        )


class ModelVerificationService:
    """
    Verify model signatures, trust expectations, and artifact bytes.

    The supplied SignatureService must contain a trusted public key. An
    embedded public key is not accepted as a trust root.
    """

    def __init__(
        self,
        *,
        signature_service: SignatureService,
        trust_policy: ModelTrustPolicy,
    ) -> None:
        if not isinstance(
            signature_service,
            SignatureService,
        ):
            raise TypeError(
                "signature_service must be SignatureService."
            )

        if not isinstance(
            trust_policy,
            ModelTrustPolicy,
        ):
            raise TypeError(
                "trust_policy must be ModelTrustPolicy."
            )

        self._signature_service = signature_service
        self._trust_policy = trust_policy

    def verify(
        self,
        model_path: str | Path,
        signed_manifest: SignedModelManifest,
    ) -> ModelVerificationResult:
        """
        Perform complete model verification.

        Check order:

        1. Validate signed-manifest object.
        2. Verify trusted signing-key reference.
        3. Verify Ed25519 signature.
        4. Evaluate producer/pipeline/version policy.
        5. Verify the model file digest and size.
        """

        if not isinstance(
            signed_manifest,
            SignedModelManifest,
        ):
            return _verification_result(
                status=(
                    ModelVerificationStatus.MALFORMED
                ),
                reason=(
                    "signed_manifest must be "
                    "SignedModelManifest."
                ),
            )

        manifest = signed_manifest.manifest
        signing_reference = (
            signed_manifest.signing_key_reference
        )

        if (
            signing_reference
            not in self._trust_policy
            .trusted_signing_keys
        ):
            return _verification_result(
                status=(
                    ModelVerificationStatus
                    .UNTRUSTED_SIGNER
                ),
                reason=(
                    "Model signing-key reference is "
                    "not trusted."
                ),
                signed_manifest=signed_manifest,
            )

        try:
            signature_valid = (
                self._signature_service.verify(
                    signed_manifest.signing_bytes(),
                    signed_manifest.signature,
                    context=MODEL_SIGNATURE_CONTEXT,
                )
            )
        except SignatureError:
            signature_valid = False

        if not signature_valid:
            return _verification_result(
                status=(
                    ModelVerificationStatus
                    .INVALID_SIGNATURE
                ),
                reason=(
                    "Model-manifest signature "
                    "verification failed."
                ),
                signed_manifest=signed_manifest,
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
                    ModelVerificationStatus
                    .POLICY_REJECTED
                ),
                reason=policy_reason,
                signed_manifest=signed_manifest,
                signature_valid=True,
                policy_valid=False,
            )

        artefact_verification = verify_artefact(
            model_path,
            manifest,
        )

        if not artefact_verification.valid:
            return _verification_result(
                status=(
                    ModelVerificationStatus
                    .INVALID_ARTEFACT
                ),
                reason=(
                    "Model artifact does not match "
                    "the signed manifest."
                ),
                signed_manifest=signed_manifest,
                signature_valid=True,
                policy_valid=True,
                artefact_valid=False,
                artefact_verification=(
                    artefact_verification
                ),
            )

        return _verification_result(
            status=ModelVerificationStatus.VALID,
            reason=(
                "Model signature, provenance policy, "
                "digest, and size verified successfully."
            ),
            signed_manifest=signed_manifest,
            signature_valid=True,
            artefact_valid=True,
            policy_valid=True,
            artefact_verification=(
                artefact_verification
            ),
        )

    def require_verified(
        self,
        model_path: str | Path,
        signed_manifest: SignedModelManifest,
    ) -> ModelVerificationResult:
        """
        Verify model and fail closed on any failure.

        Call this before the runtime deserializes or loads a model.
        """

        result = self.verify(
            model_path,
            signed_manifest,
        )

        result.require_valid()

        return result


def _verification_result(
    *,
    status: ModelVerificationStatus,
    reason: str,
    signed_manifest: SignedModelManifest | None = None,
    signature_valid: bool = False,
    artefact_valid: bool = False,
    policy_valid: bool = False,
    artefact_verification: (
        ArtefactVerificationResult | None
    ) = None,
) -> ModelVerificationResult:

    manifest = (
        None
        if signed_manifest is None
        else signed_manifest.manifest
    )

    return ModelVerificationResult(
        status=status,
        model_id=(
            None
            if manifest is None
            else manifest.artefact_id
        ),
        model_version=(
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
        raise InvalidSignedModelManifestError(
            f"{object_name} is missing fields: "
            + ", ".join(
                sorted(missing)
            )
            + "."
        )

    if unknown:
        raise InvalidSignedModelManifestError(
            f"{object_name} contains unknown fields: "
            + ", ".join(
                sorted(
                    str(item)
                    for item in unknown
                )
            )
            + "."
        )


__all__ = [
    "InvalidSignedModelManifestError",
    "MODEL_SIGNATURE_CONTEXT",
    "MODEL_SIGNATURE_FORMAT",
    "MODEL_SIGNATURE_VERSION",
    "ModelArtefactVerificationError",
    "ModelSignatureVerificationError",
    "ModelSigningError",
    "ModelSigningService",
    "ModelTrustPolicy",
    "ModelTrustPolicyError",
    "ModelVerificationResult",
    "ModelVerificationService",
    "ModelVerificationStatus",
    "SignedModelManifest",
]
