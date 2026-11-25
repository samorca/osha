"""
Digital signature services for OSHA.

This module provides Ed25519 digital signatures for security-sensitive OSHA
objects such as:

- model manifests
- dataset manifests
- provenance records
- AI decision records
- remediation authorizations
- audit events
- configuration attestations
- artifact metadata

Digital signatures provide authenticity and integrity. They do NOT provide
confidentiality. Use encryption.py when content must remain secret.

Security design
---------------
1. Ed25519 is the only signature algorithm currently accepted.
2. Private keys are used only for signing.
3. Public keys are used for verification.
4. Signatures are domain-separated to reduce cross-protocol misuse.
5. JSON signing uses deterministic canonical serialization.
6. Serialized signature envelopes contain no private key material.
7. Verification fails closed when signatures, metadata, or messages differ.

This module deliberately does not persist cryptographic keys. Production key
storage belongs behind OSHA's KeyProvider/KMS/HSM boundary.
"""

from __future__ import annotations

import base64
import binascii
import json
import struct

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Mapping

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except ImportError as exc:
    raise RuntimeError(
        "The 'cryptography' package is required by "
        "src.security.crypto.signatures. "
        "Install it with: python3 -m pip install cryptography"
    ) from exc

from .hashing import canonical_json_bytes
from .key_provider import KeyReference


SIGNATURE_FORMAT: Final[str] = "osha-signature"
SIGNATURE_VERSION: Final[int] = 1

ED25519_PRIVATE_KEY_SIZE: Final[int] = 32
ED25519_PUBLIC_KEY_SIZE: Final[int] = 32
ED25519_SIGNATURE_SIZE: Final[int] = 64

_CONTEXT_DOMAIN: Final[bytes] = b"OSHA-SIGNATURE-V1"
_MAX_CONTEXT_SIZE: Final[int] = 4096
_MAX_SERIALIZED_SIGNATURE_SIZE: Final[int] = 1024 * 1024


class SignatureError(Exception):
    """Base exception for OSHA signature failures."""


class SignatureGenerationError(SignatureError):
    """A signature could not be generated."""


class SignatureVerificationError(SignatureError):
    """A digital signature failed verification."""


class InvalidSignatureEnvelopeError(SignatureError, ValueError):
    """A signature envelope is malformed or unsupported."""


class InvalidSigningInputError(SignatureError, TypeError):
    """Input cannot be safely signed."""


class InvalidSigningKeyError(SignatureError, ValueError):
    """Private signing key is malformed or unsupported."""


class InvalidVerificationKeyError(SignatureError, ValueError):
    """Public verification key is malformed or unsupported."""


class SignatureAlgorithm(str, Enum):
    """Signature algorithms approved by OSHA."""

    ED25519 = "ed25519"


DEFAULT_SIGNATURE_ALGORITHM: Final[
    SignatureAlgorithm
] = SignatureAlgorithm.ED25519


@dataclass(frozen=True, slots=True)
class SignatureEnvelope:
    """
    Portable representation of an OSHA digital signature.

    key_reference identifies the signing-key version without exposing
    private key material.

    public_key may be transported with the signature for development and
    provenance workflows, but production trust must not be established
    merely because a public key arrived inside the same envelope.

    Production verification should resolve the trusted public key through
    an approved trust store, KMS, HSM, certificate system, or KeyProvider.
    """

    algorithm: SignatureAlgorithm
    format_version: int
    key_reference: KeyReference
    signature: bytes
    public_key: bytes

    def __post_init__(self) -> None:
        algorithm = _normalize_algorithm(self.algorithm)

        if self.format_version != SIGNATURE_VERSION:
            raise InvalidSignatureEnvelopeError(
                f"Unsupported signature format version: "
                f"{self.format_version!r}."
            )

        if not isinstance(self.key_reference, KeyReference):
            raise InvalidSignatureEnvelopeError(
                "key_reference must be a KeyReference."
            )

        signature = _require_bytes(
            self.signature,
            field="signature",
            exception_type=InvalidSignatureEnvelopeError,
        )

        public_key = _require_bytes(
            self.public_key,
            field="public_key",
            exception_type=InvalidSignatureEnvelopeError,
        )

        if len(signature) != ED25519_SIGNATURE_SIZE:
            raise InvalidSignatureEnvelopeError(
                f"Ed25519 signature must contain exactly "
                f"{ED25519_SIGNATURE_SIZE} bytes."
            )

        if len(public_key) != ED25519_PUBLIC_KEY_SIZE:
            raise InvalidSignatureEnvelopeError(
                f"Ed25519 public key must contain exactly "
                f"{ED25519_PUBLIC_KEY_SIZE} bytes."
            )

        object.__setattr__(
            self,
            "algorithm",
            algorithm,
        )

        object.__setattr__(
            self,
            "signature",
            signature,
        )

        object.__setattr__(
            self,
            "public_key",
            public_key,
        )

    def metadata(self) -> dict[str, Any]:
        """Return non-secret signature metadata."""

        return {
            "algorithm": self.algorithm.value,
            "format": SIGNATURE_FORMAT,
            "format_version": self.format_version,
            "key_id": self.key_reference.key_id,
            "key_version": self.key_reference.version,
        }

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible signature representation."""

        return {
            **self.metadata(),
            "public_key": _base64_encode(
                self.public_key
            ),
            "signature": _base64_encode(
                self.signature
            ),
        }

    def to_json(self) -> str:
        """Serialize the signature envelope."""

        return json.dumps(
            self.as_dict(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "SignatureEnvelope":
        """Parse and validate a signature envelope."""

        if not isinstance(value, Mapping):
            raise InvalidSignatureEnvelopeError(
                "Signature envelope must be a mapping."
            )

        expected_fields = {
            "algorithm",
            "format",
            "format_version",
            "key_id",
            "key_version",
            "public_key",
            "signature",
        }

        actual_fields = set(value.keys())

        missing = expected_fields - actual_fields
        unknown = actual_fields - expected_fields

        if missing:
            raise InvalidSignatureEnvelopeError(
                "Signature envelope is missing fields: "
                + ", ".join(sorted(missing))
                + "."
            )

        if unknown:
            raise InvalidSignatureEnvelopeError(
                "Signature envelope contains unknown fields: "
                + ", ".join(
                    sorted(str(item) for item in unknown)
                )
                + "."
            )

        if value["format"] != SIGNATURE_FORMAT:
            raise InvalidSignatureEnvelopeError(
                f"Unsupported signature format: "
                f"{value['format']!r}."
            )

        format_version = value["format_version"]

        if (
            isinstance(format_version, bool)
            or not isinstance(format_version, int)
        ):
            raise InvalidSignatureEnvelopeError(
                "format_version must be an integer."
            )

        key_id = value["key_id"]
        key_version = value["key_version"]

        if not isinstance(key_id, str):
            raise InvalidSignatureEnvelopeError(
                "key_id must be a string."
            )

        if not isinstance(key_version, str):
            raise InvalidSignatureEnvelopeError(
                "key_version must be a string."
            )

        return cls(
            algorithm=_normalize_algorithm(
                value["algorithm"]
            ),
            format_version=format_version,
            key_reference=KeyReference(
                key_id=key_id,
                version=key_version,
            ),
            public_key=_base64_decode(
                value["public_key"],
                field="public_key",
            ),
            signature=_base64_decode(
                value["signature"],
                field="signature",
            ),
        )

    @classmethod
    def from_json(
        cls,
        encoded: str | bytes,
    ) -> "SignatureEnvelope":
        """Parse a JSON signature envelope."""

        if isinstance(encoded, bytes):
            try:
                encoded = encoded.decode(
                    "utf-8",
                    errors="strict",
                )
            except UnicodeDecodeError as exc:
                raise InvalidSignatureEnvelopeError(
                    "Signature envelope is not valid UTF-8."
                ) from exc

        if not isinstance(encoded, str):
            raise InvalidSignatureEnvelopeError(
                "Encoded signature must be a string or UTF-8 bytes."
            )

        if len(encoded.encode("utf-8")) > (
            _MAX_SERIALIZED_SIGNATURE_SIZE
        ):
            raise InvalidSignatureEnvelopeError(
                "Serialized signature envelope exceeds supported size."
            )

        try:
            value = json.loads(encoded)

        except json.JSONDecodeError as exc:
            raise InvalidSignatureEnvelopeError(
                "Signature envelope contains invalid JSON."
            ) from exc

        return cls.from_dict(value)


class SignatureService:
    """
    OSHA Ed25519 signature service.

    A service instance represents one signing identity.

    Production systems should obtain private-key operations from a trusted
    KeyProvider/KMS/HSM implementation rather than creating long-lived raw
    private keys inside application code.
    """

    def __init__(
        self,
        *,
        key_reference: KeyReference,
        private_key: Ed25519PrivateKey | None = None,
        public_key: Ed25519PublicKey | None = None,
    ) -> None:

        if not isinstance(
            key_reference,
            KeyReference,
        ):
            raise InvalidSigningKeyError(
                "key_reference must be a KeyReference."
            )

        if (
            private_key is None
            and public_key is None
        ):
            raise InvalidSigningKeyError(
                "A private key or public key is required."
            )

        if (
            private_key is not None
            and not isinstance(
                private_key,
                Ed25519PrivateKey,
            )
        ):
            raise InvalidSigningKeyError(
                "private_key must be an Ed25519PrivateKey."
            )

        if (
            public_key is not None
            and not isinstance(
                public_key,
                Ed25519PublicKey,
            )
        ):
            raise InvalidVerificationKeyError(
                "public_key must be an Ed25519PublicKey."
            )

        derived_public: Ed25519PublicKey | None = None

        if private_key is not None:
            derived_public = private_key.public_key()

        if (
            public_key is not None
            and derived_public is not None
        ):
            supplied = _public_key_bytes(
                public_key
            )

            derived = _public_key_bytes(
                derived_public
            )

            if supplied != derived:
                raise InvalidSigningKeyError(
                    "Provided private and public keys "
                    "do not form the same Ed25519 key pair."
                )

        self._key_reference = key_reference
        self._private_key = private_key
        self._public_key = (
            public_key
            if public_key is not None
            else derived_public
        )

    @property
    def key_reference(self) -> KeyReference:
        return self._key_reference

    @property
    def can_sign(self) -> bool:
        return self._private_key is not None

    @property
    def public_key_bytes(self) -> bytes:
        if self._public_key is None:
            raise InvalidVerificationKeyError(
                "No public verification key is available."
            )

        return _public_key_bytes(
            self._public_key
        )

    def sign(
        self,
        message: bytes | bytearray | memoryview,
        *,
        context: bytes | bytearray | memoryview | None = None,
    ) -> SignatureEnvelope:
        """
        Sign arbitrary bytes.

        context provides OSHA protocol/domain separation.

        Examples:

            b"model-manifest"
            b"dataset-manifest"
            b"remediation-command"
            b"audit-event"

        The context is authenticated as part of the signed message.
        """

        if self._private_key is None:
            raise SignatureGenerationError(
                "This SignatureService has no private signing key."
            )

        message_bytes = _require_bytes(
            message,
            field="message",
            exception_type=InvalidSigningInputError,
        )

        context_bytes = _normalize_context(
            context
        )

        signing_input = _build_signing_input(
            message_bytes,
            context_bytes,
        )

        try:
            signature = self._private_key.sign(
                signing_input
            )

        except (ValueError, TypeError) as exc:
            raise SignatureGenerationError(
                "Unable to generate digital signature."
            ) from exc

        return SignatureEnvelope(
            algorithm=SignatureAlgorithm.ED25519,
            format_version=SIGNATURE_VERSION,
            key_reference=self._key_reference,
            signature=signature,
            public_key=self.public_key_bytes,
        )

    def verify(
        self,
        message: bytes | bytearray | memoryview,
        envelope: SignatureEnvelope,
        *,
        context: bytes | bytearray | memoryview | None = None,
    ) -> bool:
        """
        Verify an Ed25519 signature.

        Returns True only when verification succeeds.

        A malformed envelope raises an input exception. Cryptographically
        invalid signatures return False.
        """

        if not isinstance(
            envelope,
            SignatureEnvelope,
        ):
            raise InvalidSignatureEnvelopeError(
                "envelope must be a SignatureEnvelope."
            )

        if (
            envelope.algorithm
            is not SignatureAlgorithm.ED25519
        ):
            raise InvalidSignatureEnvelopeError(
                "Unsupported signature algorithm."
            )

        if (
            envelope.key_reference
            != self._key_reference
        ):
            return False

        message_bytes = _require_bytes(
            message,
            field="message",
            exception_type=InvalidSigningInputError,
        )

        context_bytes = _normalize_context(
            context
        )

        verification_input = _build_signing_input(
            message_bytes,
            context_bytes,
        )

        trusted_key = self._public_key

        if trusted_key is None:
            raise InvalidVerificationKeyError(
                "No trusted public verification key is available."
            )

        trusted_public_bytes = _public_key_bytes(
            trusted_key
        )

        if envelope.public_key != trusted_public_bytes:
            return False

        try:
            trusted_key.verify(
                envelope.signature,
                verification_input,
            )

            return True

        except InvalidSignature:
            return False

        except (ValueError, TypeError) as exc:
            raise SignatureVerificationError(
                "Unable to perform signature verification."
            ) from exc

    def sign_text(
        self,
        text: str,
        *,
        context: bytes | bytearray | memoryview | None = None,
        encoding: str = "utf-8",
    ) -> SignatureEnvelope:
        """Encode and sign text."""

        if not isinstance(text, str):
            raise InvalidSigningInputError(
                "text must be a string."
            )

        if (
            not isinstance(encoding, str)
            or not encoding
        ):
            raise InvalidSigningInputError(
                "encoding must be a non-empty string."
            )

        try:
            data = text.encode(
                encoding,
                errors="strict",
            )

        except (
            LookupError,
            UnicodeEncodeError,
        ) as exc:
            raise InvalidSigningInputError(
                f"Unable to encode text using {encoding!r}."
            ) from exc

        return self.sign(
            data,
            context=context,
        )

    def verify_text(
        self,
        text: str,
        envelope: SignatureEnvelope,
        *,
        context: bytes | bytearray | memoryview | None = None,
        encoding: str = "utf-8",
    ) -> bool:
        """Verify signed text."""

        if not isinstance(text, str):
            raise InvalidSigningInputError(
                "text must be a string."
            )

        try:
            encoded = text.encode(
                encoding,
                errors="strict",
            )

        except (
            LookupError,
            UnicodeEncodeError,
        ) as exc:
            raise InvalidSigningInputError(
                f"Unable to encode text using {encoding!r}."
            ) from exc

        return self.verify(
            encoded,
            envelope,
            context=context,
        )

    def sign_json(
        self,
        value: Any,
        *,
        context: bytes | bytearray | memoryview | None = None,
    ) -> SignatureEnvelope:
        """
        Canonicalize and sign a JSON-compatible object.

        canonical_json_bytes comes from OSHA hashing.py so hashing and
        signatures use the same deterministic representation.
        """

        try:
            encoded = canonical_json_bytes(
                value
            )

        except Exception as exc:
            raise InvalidSigningInputError(
                "Value cannot be canonicalized for signing."
            ) from exc

        return self.sign(
            encoded,
            context=context,
        )

    def verify_json(
        self,
        value: Any,
        envelope: SignatureEnvelope,
        *,
        context: bytes | bytearray | memoryview | None = None,
    ) -> bool:
        """Canonicalize and verify a JSON-compatible object."""

        try:
            encoded = canonical_json_bytes(
                value
            )

        except Exception as exc:
            raise InvalidSigningInputError(
                "Value cannot be canonicalized for verification."
            ) from exc

        return self.verify(
            encoded,
            envelope,
            context=context,
        )


def generate_ed25519_key_pair(
) -> tuple[
    Ed25519PrivateKey,
    Ed25519PublicKey,
]:
    """
    Generate an Ed25519 signing key pair.

    Development/testing utility.

    Production systems should normally delegate key generation and private-key
    protection to their approved KMS/HSM/key-management implementation.
    """

    try:
        private_key = (
            Ed25519PrivateKey.generate()
        )

    except Exception as exc:
        raise InvalidSigningKeyError(
            "Unable to generate Ed25519 private key."
        ) from exc

    return (
        private_key,
        private_key.public_key(),
    )


def load_ed25519_private_key(
    private_key_bytes: bytes | bytearray | memoryview,
) -> Ed25519PrivateKey:
    """
    Load a raw 32-byte Ed25519 private key.

    Raw private-key bytes should not be stored in OSHA source code,
    configuration files, logs, or model artifacts.
    """

    raw = _require_bytes(
        private_key_bytes,
        field="private_key",
        exception_type=InvalidSigningKeyError,
    )

    if len(raw) != ED25519_PRIVATE_KEY_SIZE:
        raise InvalidSigningKeyError(
            f"Ed25519 private key must contain exactly "
            f"{ED25519_PRIVATE_KEY_SIZE} bytes."
        )

    try:
        return Ed25519PrivateKey.from_private_bytes(
            raw
        )

    except ValueError as exc:
        raise InvalidSigningKeyError(
            "Invalid Ed25519 private key."
        ) from exc


def load_ed25519_public_key(
    public_key_bytes: bytes | bytearray | memoryview,
) -> Ed25519PublicKey:
    """Load a raw 32-byte Ed25519 public key."""

    raw = _require_bytes(
        public_key_bytes,
        field="public_key",
        exception_type=InvalidVerificationKeyError,
    )

    if len(raw) != ED25519_PUBLIC_KEY_SIZE:
        raise InvalidVerificationKeyError(
            f"Ed25519 public key must contain exactly "
            f"{ED25519_PUBLIC_KEY_SIZE} bytes."
        )

    try:
        return Ed25519PublicKey.from_public_bytes(
            raw
        )

    except ValueError as exc:
        raise InvalidVerificationKeyError(
            "Invalid Ed25519 public key."
        ) from exc


def export_ed25519_private_key(
    private_key: Ed25519PrivateKey,
) -> bytes:
    """
    Export raw private-key bytes.

    Use only when necessary for integration with an approved key provider.
    Never log the returned bytes.
    """

    if not isinstance(
        private_key,
        Ed25519PrivateKey,
    ):
        raise InvalidSigningKeyError(
            "private_key must be an Ed25519PrivateKey."
        )

    return private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def export_ed25519_public_key(
    public_key: Ed25519PublicKey,
) -> bytes:
    """Export raw Ed25519 public-key bytes."""

    if not isinstance(
        public_key,
        Ed25519PublicKey,
    ):
        raise InvalidVerificationKeyError(
            "public_key must be an Ed25519PublicKey."
        )

    return _public_key_bytes(
        public_key
    )


def _public_key_bytes(
    public_key: Ed25519PublicKey,
) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _build_signing_input(
    message: bytes,
    context: bytes,
) -> bytes:
    """
    Build unambiguous domain-separated bytes for signing.

    Length prefixes prevent ambiguity between context and message.
    """

    return b"".join(
        (
            _CONTEXT_DOMAIN,
            struct.pack(
                ">I",
                SIGNATURE_VERSION,
            ),
            struct.pack(
                ">I",
                len(context),
            ),
            context,
            struct.pack(
                ">Q",
                len(message),
            ),
            message,
        )
    )


def _normalize_context(
    context: bytes | bytearray | memoryview | None,
) -> bytes:

    if context is None:
        return b""

    result = _require_bytes(
        context,
        field="context",
        exception_type=InvalidSigningInputError,
    )

    if len(result) > _MAX_CONTEXT_SIZE:
        raise InvalidSigningInputError(
            f"Signature context exceeds "
            f"{_MAX_CONTEXT_SIZE} bytes."
        )

    return result


def _normalize_algorithm(
    algorithm: SignatureAlgorithm | str,
) -> SignatureAlgorithm:

    if isinstance(
        algorithm,
        SignatureAlgorithm,
    ):
        return algorithm

    if not isinstance(algorithm, str):
        raise InvalidSignatureEnvelopeError(
            "Signature algorithm must be a string."
        )

    normalized = (
        algorithm
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
    )

    if normalized != "ed25519":
        raise InvalidSignatureEnvelopeError(
            f"Unsupported signature algorithm: "
            f"{algorithm!r}."
        )

    return SignatureAlgorithm.ED25519


def _require_bytes(
    value: bytes | bytearray | memoryview,
    *,
    field: str,
    exception_type: type[Exception],
) -> bytes:

    if not isinstance(
        value,
        (
            bytes,
            bytearray,
            memoryview,
        ),
    ):
        raise exception_type(
            f"{field} must be bytes-like."
        )

    try:
        view = memoryview(value)

        if not view.contiguous:
            raise exception_type(
                f"{field} must be contiguous."
            )

        return bytes(
            view.cast("B")
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise exception_type(
            f"{field} cannot be interpreted as bytes."
        ) from exc


def _base64_encode(
    value: bytes,
) -> str:

    return base64.b64encode(
        value
    ).decode("ascii")


def _base64_decode(
    value: Any,
    *,
    field: str,
) -> bytes:

    if not isinstance(value, str):
        raise InvalidSignatureEnvelopeError(
            f"{field} must be a Base64 string."
        )

    try:
        return base64.b64decode(
            value.encode("ascii"),
            validate=True,
        )

    except (
        UnicodeEncodeError,
        binascii.Error,
        ValueError,
    ) as exc:
        raise InvalidSignatureEnvelopeError(
            f"{field} is not valid Base64."
        ) from exc


__all__ = [
    "DEFAULT_SIGNATURE_ALGORITHM",
    "ED25519_PRIVATE_KEY_SIZE",
    "ED25519_PUBLIC_KEY_SIZE",
    "ED25519_SIGNATURE_SIZE",
    "InvalidSignatureEnvelopeError",
    "InvalidSigningInputError",
    "InvalidSigningKeyError",
    "InvalidVerificationKeyError",
    "SIGNATURE_FORMAT",
    "SIGNATURE_VERSION",
    "SignatureAlgorithm",
    "SignatureEnvelope",
    "SignatureError",
    "SignatureGenerationError",
    "SignatureService",
    "SignatureVerificationError",
    "export_ed25519_private_key",
    "export_ed25519_public_key",
    "generate_ed25519_key_pair",
    "load_ed25519_private_key",
    "load_ed25519_public_key",
]
