"""
Authenticated encryption services for OSHA.

This module protects sensitive OSHA data using AES-256-GCM.

AES-GCM provides:

- confidentiality for plaintext
- integrity verification for ciphertext
- authentication of associated metadata
- tamper detection during decryption

Keys are obtained exclusively through the KeyProvider interface. This
module does not read keys directly from source code, environment variables,
configuration files, or secret files.

Security requirements
---------------------
1. Every encryption operation generates a fresh 96-bit nonce.
2. Nonces must never be manually reused with the same key.
3. Only 256-bit symmetric encryption keys are accepted.
4. Key identity and envelope metadata are authenticated.
5. Decryption fails closed when authentication fails.
6. Secret keys and plaintext are never included in exception messages.
7. This module is intended for bounded messages and records. Large model
   files and datasets should later use chunked or envelope encryption.

The cryptography package is required:

    python3 -m pip install cryptography
"""

from __future__ import annotations

import base64
import binascii
import json
import secrets
import struct

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Mapping

try:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError as exc:
    raise RuntimeError(
        "The 'cryptography' package is required by "
        "src.security.crypto.encryption. "
        "Install it with: python3 -m pip install cryptography"
    ) from exc

from .key_provider import (
    InvalidKeyError,
    KeyMaterial,
    KeyPolicyError,
    KeyProvider,
    KeyProviderError,
    KeyPurpose,
    KeyReference,
    KeyState,
    KeyType,
)


AES_256_KEY_SIZE: Final[int] = 32
AES_GCM_NONCE_SIZE: Final[int] = 12
AES_GCM_TAG_SIZE: Final[int] = 16

ENCRYPTION_FORMAT: Final[str] = "osha-aead"
ENCRYPTION_VERSION: Final[int] = 1

_AAD_DOMAIN: Final[bytes] = b"OSHA-AEAD-V1"
_MAX_ENCODED_PAYLOAD_SIZE: Final[int] = 256 * 1024 * 1024


class EncryptionError(Exception):
    """Base exception for OSHA encryption failures."""


class InvalidPlaintextError(EncryptionError, TypeError):
    """Plaintext is not valid bytes-like input."""


class InvalidAssociatedDataError(EncryptionError, TypeError):
    """Associated authenticated data is invalid."""


class InvalidEncryptedPayloadError(EncryptionError, ValueError):
    """Encrypted payload is malformed or unsupported."""


class EncryptionKeyError(EncryptionError):
    """Encryption key is unavailable, invalid, or prohibited."""


class DecryptionError(EncryptionError):
    """
    Decryption failed.

    This exception intentionally does not distinguish between a wrong key,
    modified nonce, modified ciphertext, modified metadata, or modified
    associated data. Returning a generic failure avoids exposing unnecessary
    cryptographic failure details to callers.
    """


class EncryptionAlgorithm(str, Enum):
    """Encryption algorithms approved by this module."""

    AES_256_GCM = "aes-256-gcm"


DEFAULT_ENCRYPTION_ALGORITHM: Final[
    EncryptionAlgorithm
] = EncryptionAlgorithm.AES_256_GCM


@dataclass(frozen=True, slots=True)
class EncryptedPayload:
    """
    Portable authenticated-encryption result.

    The nonce is not secret and may be stored with the ciphertext.

    AESGCM returns ciphertext with its authentication tag appended. Therefore,
    ciphertext contains both encrypted bytes and the 16-byte GCM tag.
    """

    algorithm: EncryptionAlgorithm
    format_version: int
    key_reference: KeyReference
    nonce: bytes
    ciphertext: bytes

    def __post_init__(self) -> None:
        algorithm = _normalize_algorithm(self.algorithm)

        if self.format_version != ENCRYPTION_VERSION:
            raise InvalidEncryptedPayloadError(
                f"Unsupported encryption format version: "
                f"{self.format_version!r}."
            )

        if not isinstance(self.key_reference, KeyReference):
            raise InvalidEncryptedPayloadError(
                "key_reference must be a KeyReference."
            )

        nonce = _require_bytes(
            self.nonce,
            field="nonce",
            exception_type=InvalidEncryptedPayloadError,
        )

        ciphertext = _require_bytes(
            self.ciphertext,
            field="ciphertext",
            exception_type=InvalidEncryptedPayloadError,
        )

        if len(nonce) != AES_GCM_NONCE_SIZE:
            raise InvalidEncryptedPayloadError(
                f"AES-GCM nonce must contain exactly "
                f"{AES_GCM_NONCE_SIZE} bytes."
            )

        if len(ciphertext) < AES_GCM_TAG_SIZE:
            raise InvalidEncryptedPayloadError(
                "Ciphertext is too short to contain an AES-GCM "
                "authentication tag."
            )

        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "nonce", nonce)
        object.__setattr__(self, "ciphertext", ciphertext)

    def metadata(self) -> dict[str, Any]:
        """
        Return non-secret metadata authenticated by AES-GCM.
        """
        return {
            "algorithm": self.algorithm.value,
            "format": ENCRYPTION_FORMAT,
            "format_version": self.format_version,
            "key_id": self.key_reference.key_id,
            "key_version": self.key_reference.version,
        }

    def as_dict(self) -> dict[str, Any]:
        """
        Return a JSON-compatible representation.

        Base64 is transport encoding only. It is not encryption.
        """
        return {
            **self.metadata(),
            "nonce": _base64_encode(self.nonce),
            "ciphertext": _base64_encode(self.ciphertext),
        }

    def to_json(self) -> str:
        """Serialize this encrypted payload as compact JSON."""
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
    ) -> "EncryptedPayload":
        """Parse and validate a JSON-compatible encrypted payload."""
        if not isinstance(value, Mapping):
            raise InvalidEncryptedPayloadError(
                "Encrypted payload must be a mapping."
            )

        expected_fields = {
            "algorithm",
            "format",
            "format_version",
            "key_id",
            "key_version",
            "nonce",
            "ciphertext",
        }

        actual_fields = set(value.keys())

        missing = expected_fields - actual_fields
        unknown = actual_fields - expected_fields

        if missing:
            raise InvalidEncryptedPayloadError(
                "Encrypted payload is missing required fields: "
                + ", ".join(sorted(missing))
                + "."
            )

        if unknown:
            raise InvalidEncryptedPayloadError(
                "Encrypted payload contains unknown fields: "
                + ", ".join(sorted(str(item) for item in unknown))
                + "."
            )

        if value["format"] != ENCRYPTION_FORMAT:
            raise InvalidEncryptedPayloadError(
                f"Unsupported encryption format: {value['format']!r}."
            )

        format_version = value["format_version"]

        if isinstance(format_version, bool) or not isinstance(
            format_version,
            int,
        ):
            raise InvalidEncryptedPayloadError(
                "format_version must be an integer."
            )

        key_id = value["key_id"]
        key_version = value["key_version"]

        if not isinstance(key_id, str):
            raise InvalidEncryptedPayloadError(
                "key_id must be a string."
            )

        if not isinstance(key_version, str):
            raise InvalidEncryptedPayloadError(
                "key_version must be a string."
            )

        nonce = _base64_decode(
            value["nonce"],
            field="nonce",
        )

        ciphertext = _base64_decode(
            value["ciphertext"],
            field="ciphertext",
        )

        return cls(
            algorithm=_normalize_algorithm(value["algorithm"]),
            format_version=format_version,
            key_reference=KeyReference(
                key_id=key_id,
                version=key_version,
            ),
            nonce=nonce,
            ciphertext=ciphertext,
        )

    @classmethod
    def from_json(
        cls,
        encoded: str | bytes,
    ) -> "EncryptedPayload":
        """Parse and validate a JSON-encoded encrypted payload."""
        if isinstance(encoded, bytes):
            try:
                encoded = encoded.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise InvalidEncryptedPayloadError(
                    "Encrypted payload JSON is not valid UTF-8."
                ) from exc

        if not isinstance(encoded, str):
            raise InvalidEncryptedPayloadError(
                "Encoded payload must be a string or UTF-8 bytes."
            )

        if len(encoded.encode("utf-8")) > _MAX_ENCODED_PAYLOAD_SIZE:
            raise InvalidEncryptedPayloadError(
                "Encoded encrypted payload exceeds the supported size."
            )

        try:
            value = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise InvalidEncryptedPayloadError(
                "Encrypted payload contains invalid JSON."
            ) from exc

        return cls.from_dict(value)


class EncryptionService:
    """
    OSHA authenticated-encryption service.

    The service depends on KeyProvider rather than a concrete storage system.
    This allows development to use InMemoryKeyProvider while production can
    later use an approved KMS, secret manager, or HSM-backed provider.
    """

    def __init__(
        self,
        key_provider: KeyProvider,
    ) -> None:
        if not isinstance(key_provider, KeyProvider):
            raise TypeError(
                "key_provider must implement KeyProvider."
            )

        self._key_provider = key_provider

    def encrypt(
        self,
        plaintext: bytes | bytearray | memoryview,
        *,
        key_reference: KeyReference,
        associated_data: bytes | bytearray | memoryview | None = None,
    ) -> EncryptedPayload:
        """
        Encrypt and authenticate plaintext using AES-256-GCM.

        A fresh cryptographically random 96-bit nonce is generated internally.
        Callers cannot supply a nonce, which reduces accidental nonce reuse.

        associated_data is authenticated but remains unencrypted. Appropriate
        examples include an event ID, tenant ID, resource ID, schema version,
        or incident ID that must be cryptographically bound to the ciphertext.
        """
        plaintext_bytes = _require_bytes(
            plaintext,
            field="plaintext",
            exception_type=InvalidPlaintextError,
        )

        user_aad = _normalize_associated_data(associated_data)

        material = self._get_encryption_key(key_reference)

        payload_metadata = {
            "algorithm": DEFAULT_ENCRYPTION_ALGORITHM.value,
            "format": ENCRYPTION_FORMAT,
            "format_version": ENCRYPTION_VERSION,
            "key_id": key_reference.key_id,
            "key_version": key_reference.version,
        }

        authenticated_data = _build_authenticated_data(
            payload_metadata,
            user_aad,
        )

        nonce = secrets.token_bytes(AES_GCM_NONCE_SIZE)

        try:
            cipher = AESGCM(material.secret)
            ciphertext = cipher.encrypt(
                nonce,
                plaintext_bytes,
                authenticated_data,
            )
        except (ValueError, OverflowError) as exc:
            raise EncryptionError(
                "Unable to encrypt the supplied plaintext."
            ) from exc

        return EncryptedPayload(
            algorithm=DEFAULT_ENCRYPTION_ALGORITHM,
            format_version=ENCRYPTION_VERSION,
            key_reference=key_reference,
            nonce=nonce,
            ciphertext=ciphertext,
        )

    def decrypt(
        self,
        payload: EncryptedPayload,
        *,
        associated_data: bytes | bytearray | memoryview | None = None,
    ) -> bytes:
        """
        Authenticate and decrypt an encrypted payload.

        The exact same associated_data supplied during encryption must be
        supplied during decryption. Any change to the key reference, nonce,
        ciphertext, authentication tag, metadata, or associated data causes
        decryption to fail.
        """
        if not isinstance(payload, EncryptedPayload):
            raise InvalidEncryptedPayloadError(
                "payload must be an EncryptedPayload."
            )

        user_aad = _normalize_associated_data(associated_data)

        material = self._get_encryption_key(
            payload.key_reference,
        )

        authenticated_data = _build_authenticated_data(
            payload.metadata(),
            user_aad,
        )

        try:
            cipher = AESGCM(material.secret)

            return cipher.decrypt(
                payload.nonce,
                payload.ciphertext,
                authenticated_data,
            )

        except InvalidTag as exc:
            raise DecryptionError(
                "Encrypted payload authentication failed."
            ) from exc

        except (ValueError, OverflowError) as exc:
            raise DecryptionError(
                "Unable to decrypt the encrypted payload."
            ) from exc

    def encrypt_text(
        self,
        plaintext: str,
        *,
        key_reference: KeyReference,
        associated_data: bytes | bytearray | memoryview | None = None,
        encoding: str = "utf-8",
    ) -> EncryptedPayload:
        """Encode and encrypt text."""
        if not isinstance(plaintext, str):
            raise InvalidPlaintextError(
                "Text plaintext must be a string."
            )

        if not isinstance(encoding, str) or not encoding:
            raise InvalidPlaintextError(
                "encoding must be a non-empty string."
            )

        try:
            encoded = plaintext.encode(
                encoding,
                errors="strict",
            )
        except (LookupError, UnicodeEncodeError) as exc:
            raise InvalidPlaintextError(
                f"Unable to encode plaintext using {encoding!r}."
            ) from exc

        return self.encrypt(
            encoded,
            key_reference=key_reference,
            associated_data=associated_data,
        )

    def decrypt_text(
        self,
        payload: EncryptedPayload,
        *,
        associated_data: bytes | bytearray | memoryview | None = None,
        encoding: str = "utf-8",
    ) -> str:
        """Authenticate, decrypt, and decode text."""
        plaintext = self.decrypt(
            payload,
            associated_data=associated_data,
        )

        try:
            return plaintext.decode(
                encoding,
                errors="strict",
            )
        except (LookupError, UnicodeDecodeError) as exc:
            raise DecryptionError(
                f"Decrypted plaintext is not valid {encoding!r} text."
            ) from exc

    def encrypt_json(
        self,
        value: Any,
        *,
        key_reference: KeyReference,
        associated_data: bytes | bytearray | memoryview | None = None,
    ) -> EncryptedPayload:
        """
        Deterministically serialize and encrypt a JSON-compatible value.
        """
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidPlaintextError(
                "Value cannot be serialized as valid JSON."
            ) from exc

        return self.encrypt(
            encoded,
            key_reference=key_reference,
            associated_data=associated_data,
        )

    def decrypt_json(
        self,
        payload: EncryptedPayload,
        *,
        associated_data: bytes | bytearray | memoryview | None = None,
    ) -> Any:
        """Authenticate, decrypt, and parse JSON data."""
        plaintext = self.decrypt(
            payload,
            associated_data=associated_data,
        )

        try:
            return json.loads(
                plaintext.decode("utf-8", errors="strict")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DecryptionError(
                "Decrypted plaintext is not valid JSON."
            ) from exc

    def _get_encryption_key(
        self,
        reference: KeyReference,
    ) -> KeyMaterial:
        if not isinstance(reference, KeyReference):
            raise EncryptionKeyError(
                "key_reference must be a KeyReference."
            )

        try:
            material = self._key_provider.get_key(
                reference,
                expected_purpose=KeyPurpose.ENCRYPTION,
            )
        except (
            InvalidKeyError,
            KeyPolicyError,
            KeyProviderError,
        ) as exc:
            raise EncryptionKeyError(
                f"Unable to obtain encryption key {reference}."
            ) from exc

        metadata = material.metadata

        if metadata.state is not KeyState.ACTIVE:
            raise EncryptionKeyError(
                f"Encryption key {reference} is not active."
            )

        if metadata.key_type is not KeyType.SYMMETRIC:
            raise EncryptionKeyError(
                f"Encryption key {reference} is not symmetric."
            )

        if metadata.purpose is not KeyPurpose.ENCRYPTION:
            raise EncryptionKeyError(
                f"Key {reference} is not authorized for encryption."
            )

        if len(material.secret) != AES_256_KEY_SIZE:
            raise EncryptionKeyError(
                f"AES-256-GCM requires a {AES_256_KEY_SIZE}-byte key."
            )

        return material


def _normalize_algorithm(
    algorithm: EncryptionAlgorithm | str,
) -> EncryptionAlgorithm:
    if isinstance(algorithm, EncryptionAlgorithm):
        return algorithm

    if not isinstance(algorithm, str):
        raise InvalidEncryptedPayloadError(
            "Encryption algorithm must be a string."
        )

    normalized = algorithm.strip().lower().replace("_", "-")

    if normalized != EncryptionAlgorithm.AES_256_GCM.value:
        raise InvalidEncryptedPayloadError(
            f"Unsupported encryption algorithm: {algorithm!r}."
        )

    return EncryptionAlgorithm.AES_256_GCM


def _normalize_associated_data(
    associated_data: bytes | bytearray | memoryview | None,
) -> bytes:
    if associated_data is None:
        return b""

    return _require_bytes(
        associated_data,
        field="associated_data",
        exception_type=InvalidAssociatedDataError,
    )


def _build_authenticated_data(
    metadata: Mapping[str, Any],
    user_aad: bytes,
) -> bytes:
    """
    Build unambiguous associated authenticated data.

    The length prefix separates OSHA-controlled metadata from caller-provided
    associated data without relying on a delimiter that might appear in the
    caller's bytes.
    """
    try:
        metadata_bytes = json.dumps(
            metadata,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidAssociatedDataError(
            "Encryption metadata cannot be serialized."
        ) from exc

    return b"".join(
        (
            _AAD_DOMAIN,
            struct.pack(">I", len(metadata_bytes)),
            metadata_bytes,
            struct.pack(">Q", len(user_aad)),
            user_aad,
        )
    )


def _require_bytes(
    value: bytes | bytearray | memoryview,
    *,
    field: str,
    exception_type: type[Exception],
) -> bytes:
    if not isinstance(
        value,
        (bytes, bytearray, memoryview),
    ):
        raise exception_type(
            f"{field} must be bytes-like."
        )

    try:
        view = memoryview(value)

        if not view.contiguous:
            raise exception_type(
                f"{field} memoryview must be contiguous."
            )

        return bytes(view.cast("B"))
    except (TypeError, ValueError) as exc:
        raise exception_type(
            f"{field} cannot be interpreted as bytes."
        ) from exc


def _base64_encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _base64_decode(
    value: Any,
    *,
    field: str,
) -> bytes:
    if not isinstance(value, str):
        raise InvalidEncryptedPayloadError(
            f"{field} must be a Base64 string."
        )

    if len(value.encode("ascii", errors="ignore")) > (
        _MAX_ENCODED_PAYLOAD_SIZE
    ):
        raise InvalidEncryptedPayloadError(
            f"{field} exceeds the supported size."
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
        raise InvalidEncryptedPayloadError(
            f"{field} is not valid Base64."
        ) from exc


__all__ = [
    "AES_256_KEY_SIZE",
    "AES_GCM_NONCE_SIZE",
    "AES_GCM_TAG_SIZE",
    "DEFAULT_ENCRYPTION_ALGORITHM",
    "DecryptionError",
    "EncryptedPayload",
    "EncryptionAlgorithm",
    "EncryptionError",
    "EncryptionKeyError",
    "EncryptionService",
    "InvalidAssociatedDataError",
    "InvalidEncryptedPayloadError",
    "InvalidPlaintextError",
]
