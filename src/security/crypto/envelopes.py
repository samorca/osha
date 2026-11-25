"""
Protected message envelopes for OSHA.

This module combines OSHA's existing cryptographic components into a common
transport structure:

- hashing.py provides deterministic canonical JSON serialization
- key_provider.py identifies encryption and signing keys
- encryption.py protects payload confidentiality and integrity
- signatures.py authenticates the complete protected envelope

The envelope format can be used by:

- collectors and telemetry ingestion
- training and provenance pipelines
- AI inference workers
- correlation and intelligence pipelines
- decision services
- remediation executors
- audit services
- internal service-to-service messages

Security model
--------------
1. Payload data is encrypted with AES-256-GCM through EncryptionService.
2. Envelope metadata is bound to the encrypted payload as associated data.
3. The encrypted envelope can be signed using Ed25519.
4. Signature verification occurs before payload decryption.
5. The signature covers metadata and encrypted-payload representation.
6. Expiration is checked independently of cryptographic validity.
7. Unknown fields, malformed values, and unsupported versions are rejected.
8. An embedded public key does not establish trust by itself. Verification
   must use a SignatureService initialized with a trusted public key.
"""

from __future__ import annotations

import json
import re
import time
import uuid

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Mapping

from .encryption import (
    DecryptionError,
    EncryptedPayload,
    EncryptionError,
    EncryptionService,
    InvalidEncryptedPayloadError,
)
from .hashing import (
    Digest,
    HashAlgorithm,
    canonical_json_bytes,
    hash_json,
)
from .key_provider import KeyReference
from .signatures import (
    InvalidSignatureEnvelopeError,
    SignatureEnvelope,
    SignatureError,
    SignatureService,
)


ENVELOPE_FORMAT: Final[str] = "osha-secure-envelope"
ENVELOPE_VERSION: Final[int] = 1

DEFAULT_TTL_SECONDS: Final[int] = 300
MAX_TTL_SECONDS: Final[int] = 86_400
MAX_CLOCK_SKEW_SECONDS: Final[int] = 60

MAX_ID_LENGTH: Final[int] = 128
MAX_NAME_LENGTH: Final[int] = 256
MAX_CONTENT_TYPE_LENGTH: Final[int] = 128
MAX_SERIALIZED_ENVELOPE_SIZE: Final[int] = 256 * 1024 * 1024

_ENVELOPE_SIGNATURE_CONTEXT: Final[bytes] = (
    b"osha-secure-envelope-v1"
)

_IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"
)


class EnvelopeError(Exception):
    """Base exception for OSHA protected-envelope failures."""


class InvalidEnvelopeError(EnvelopeError, ValueError):
    """The protected envelope is malformed or unsupported."""


class EnvelopeEncryptionError(EnvelopeError):
    """The envelope payload could not be encrypted."""


class EnvelopeDecryptionError(EnvelopeError):
    """The protected payload could not be authenticated or decrypted."""


class EnvelopeSigningError(EnvelopeError):
    """The envelope could not be digitally signed."""


class EnvelopeVerificationError(EnvelopeError):
    """The envelope failed signature or integrity verification."""


class EnvelopeExpiredError(EnvelopeVerificationError):
    """The envelope is expired."""


class EnvelopeNotYetValidError(EnvelopeVerificationError):
    """The envelope creation time is unacceptably far in the future."""


class EnvelopeReplayError(EnvelopeVerificationError):
    """The envelope was identified as a replay."""


class EnvelopePayloadType(str, Enum):
    """Supported logical payload representations."""

    BYTES = "bytes"
    TEXT = "text"
    JSON = "json"


class EnvelopePurpose(str, Enum):
    """Security-sensitive OSHA envelope purposes."""

    TELEMETRY = "telemetry"
    TRAINING = "training"
    INFERENCE = "inference"
    CORRELATION = "correlation"
    DECISION = "decision"
    REMEDIATION = "remediation"
    AUDIT = "audit"
    PROVENANCE = "provenance"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class EnvelopeMetadata:
    """
    Non-secret metadata attached to an OSHA protected envelope.

    This metadata is authenticated twice:

    1. It is included as AES-GCM associated data.
    2. It is covered by the optional Ed25519 envelope signature.
    """

    envelope_id: str
    correlation_id: str
    source: str
    destination: str
    purpose: EnvelopePurpose
    payload_type: EnvelopePayloadType
    content_type: str
    created_at_ns: int
    expires_at_ns: int
    schema_version: str

    def __post_init__(self) -> None:
        _validate_identifier(
            self.envelope_id,
            field="envelope_id",
            max_length=MAX_ID_LENGTH,
        )

        _validate_identifier(
            self.correlation_id,
            field="correlation_id",
            max_length=MAX_ID_LENGTH,
        )

        _validate_identifier(
            self.source,
            field="source",
            max_length=MAX_NAME_LENGTH,
        )

        _validate_identifier(
            self.destination,
            field="destination",
            max_length=MAX_NAME_LENGTH,
        )

        if not isinstance(self.purpose, EnvelopePurpose):
            raise InvalidEnvelopeError(
                "purpose must be an EnvelopePurpose."
            )

        if not isinstance(
            self.payload_type,
            EnvelopePayloadType,
        ):
            raise InvalidEnvelopeError(
                "payload_type must be an EnvelopePayloadType."
            )

        _validate_text(
            self.content_type,
            field="content_type",
            max_length=MAX_CONTENT_TYPE_LENGTH,
        )

        _validate_text(
            self.schema_version,
            field="schema_version",
            max_length=64,
        )

        if (
            isinstance(self.created_at_ns, bool)
            or not isinstance(self.created_at_ns, int)
            or self.created_at_ns <= 0
        ):
            raise InvalidEnvelopeError(
                "created_at_ns must be a positive integer."
            )

        if (
            isinstance(self.expires_at_ns, bool)
            or not isinstance(self.expires_at_ns, int)
            or self.expires_at_ns <= self.created_at_ns
        ):
            raise InvalidEnvelopeError(
                "expires_at_ns must be greater than created_at_ns."
            )

        lifetime_ns = (
            self.expires_at_ns - self.created_at_ns
        )

        if lifetime_ns > MAX_TTL_SECONDS * 1_000_000_000:
            raise InvalidEnvelopeError(
                f"Envelope lifetime exceeds the maximum of "
                f"{MAX_TTL_SECONDS} seconds."
            )

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible metadata."""

        return {
            "content_type": self.content_type,
            "correlation_id": self.correlation_id,
            "created_at_ns": self.created_at_ns,
            "destination": self.destination,
            "envelope_id": self.envelope_id,
            "expires_at_ns": self.expires_at_ns,
            "payload_type": self.payload_type.value,
            "purpose": self.purpose.value,
            "schema_version": self.schema_version,
            "source": self.source,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "EnvelopeMetadata":
        """Parse and strictly validate envelope metadata."""

        if not isinstance(value, Mapping):
            raise InvalidEnvelopeError(
                "Envelope metadata must be a mapping."
            )

        expected_fields = {
            "content_type",
            "correlation_id",
            "created_at_ns",
            "destination",
            "envelope_id",
            "expires_at_ns",
            "payload_type",
            "purpose",
            "schema_version",
            "source",
        }

        _validate_exact_fields(
            value,
            expected_fields,
            object_name="envelope metadata",
        )

        try:
            purpose = EnvelopePurpose(value["purpose"])
        except (TypeError, ValueError) as exc:
            raise InvalidEnvelopeError(
                f"Unsupported envelope purpose: "
                f"{value['purpose']!r}."
            ) from exc

        try:
            payload_type = EnvelopePayloadType(
                value["payload_type"]
            )
        except (TypeError, ValueError) as exc:
            raise InvalidEnvelopeError(
                f"Unsupported payload type: "
                f"{value['payload_type']!r}."
            ) from exc

        return cls(
            envelope_id=value["envelope_id"],
            correlation_id=value["correlation_id"],
            source=value["source"],
            destination=value["destination"],
            purpose=purpose,
            payload_type=payload_type,
            content_type=value["content_type"],
            created_at_ns=value["created_at_ns"],
            expires_at_ns=value["expires_at_ns"],
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class SecureEnvelope:
    """
    Complete OSHA protected envelope.

    payload_digest hashes the canonical JSON representation of the encrypted
    payload. It supports explicit integrity checks before signature
    verification and decryption.

    signature may be None only when an unsigned envelope was intentionally
    created. Production decision, remediation, provenance, and audit flows
    should normally require signatures.
    """

    format_version: int
    metadata: EnvelopeMetadata
    encrypted_payload: EncryptedPayload
    payload_digest: Digest
    signature: SignatureEnvelope | None = None

    def __post_init__(self) -> None:
        if self.format_version != ENVELOPE_VERSION:
            raise InvalidEnvelopeError(
                f"Unsupported envelope version: "
                f"{self.format_version!r}."
            )

        if not isinstance(
            self.metadata,
            EnvelopeMetadata,
        ):
            raise InvalidEnvelopeError(
                "metadata must be EnvelopeMetadata."
            )

        if not isinstance(
            self.encrypted_payload,
            EncryptedPayload,
        ):
            raise InvalidEnvelopeError(
                "encrypted_payload must be EncryptedPayload."
            )

        if not isinstance(self.payload_digest, Digest):
            raise InvalidEnvelopeError(
                "payload_digest must be a Digest."
            )

        if (
            self.signature is not None
            and not isinstance(
                self.signature,
                SignatureEnvelope,
            )
        ):
            raise InvalidEnvelopeError(
                "signature must be SignatureEnvelope or None."
            )

    @property
    def is_signed(self) -> bool:
        """Return whether the envelope contains a signature."""

        return self.signature is not None

    def unsigned_dict(self) -> dict[str, Any]:
        """
        Return all signed content except the signature itself.

        This representation is used as the input to SignatureService.
        """

        return {
            "encrypted_payload": (
                self.encrypted_payload.as_dict()
            ),
            "format": ENVELOPE_FORMAT,
            "format_version": self.format_version,
            "metadata": self.metadata.as_dict(),
            "payload_digest": str(self.payload_digest),
        }

    def signing_bytes(self) -> bytes:
        """Return canonical bytes covered by the signature."""

        return canonical_json_bytes(
            self.unsigned_dict()
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible envelope."""

        result = self.unsigned_dict()

        result["signature"] = (
            None
            if self.signature is None
            else self.signature.as_dict()
        )

        return result

    def to_json(self) -> str:
        """Serialize the complete envelope as compact JSON."""

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
    ) -> "SecureEnvelope":
        """Parse and strictly validate an envelope."""

        if not isinstance(value, Mapping):
            raise InvalidEnvelopeError(
                "Secure envelope must be a mapping."
            )

        expected_fields = {
            "encrypted_payload",
            "format",
            "format_version",
            "metadata",
            "payload_digest",
            "signature",
        }

        _validate_exact_fields(
            value,
            expected_fields,
            object_name="secure envelope",
        )

        if value["format"] != ENVELOPE_FORMAT:
            raise InvalidEnvelopeError(
                f"Unsupported envelope format: "
                f"{value['format']!r}."
            )

        format_version = value["format_version"]

        if (
            isinstance(format_version, bool)
            or not isinstance(format_version, int)
        ):
            raise InvalidEnvelopeError(
                "format_version must be an integer."
            )

        try:
            payload_digest = Digest.parse(
                value["payload_digest"]
            )
        except Exception as exc:
            raise InvalidEnvelopeError(
                "payload_digest is invalid."
            ) from exc

        signature_value = value["signature"]

        try:
            signature = (
                None
                if signature_value is None
                else SignatureEnvelope.from_dict(
                    signature_value
                )
            )
        except InvalidSignatureEnvelopeError as exc:
            raise InvalidEnvelopeError(
                "Envelope signature is invalid."
            ) from exc

        try:
            encrypted_payload = (
                EncryptedPayload.from_dict(
                    value["encrypted_payload"]
                )
            )
        except InvalidEncryptedPayloadError as exc:
            raise InvalidEnvelopeError(
                "Encrypted envelope payload is invalid."
            ) from exc

        return cls(
            format_version=format_version,
            metadata=EnvelopeMetadata.from_dict(
                value["metadata"]
            ),
            encrypted_payload=encrypted_payload,
            payload_digest=payload_digest,
            signature=signature,
        )

    @classmethod
    def from_json(
        cls,
        encoded: str | bytes,
    ) -> "SecureEnvelope":
        """Parse and validate a JSON envelope."""

        if isinstance(encoded, bytes):
            try:
                encoded = encoded.decode(
                    "utf-8",
                    errors="strict",
                )
            except UnicodeDecodeError as exc:
                raise InvalidEnvelopeError(
                    "Envelope is not valid UTF-8."
                ) from exc

        if not isinstance(encoded, str):
            raise InvalidEnvelopeError(
                "Encoded envelope must be a string "
                "or UTF-8 bytes."
            )

        if len(encoded.encode("utf-8")) > (
            MAX_SERIALIZED_ENVELOPE_SIZE
        ):
            raise InvalidEnvelopeError(
                "Serialized envelope exceeds supported size."
            )

        try:
            value = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise InvalidEnvelopeError(
                "Envelope contains invalid JSON."
            ) from exc

        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class OpenedEnvelope:
    """Successfully authenticated and decrypted envelope."""

    metadata: EnvelopeMetadata
    plaintext: bytes
    signed: bool
    signature_key_reference: KeyReference | None

    def text(
        self,
        *,
        encoding: str = "utf-8",
    ) -> str:
        """Decode a text envelope payload."""

        if self.metadata.payload_type is not (
            EnvelopePayloadType.TEXT
        ):
            raise InvalidEnvelopeError(
                "Envelope payload type is not text."
            )

        try:
            return self.plaintext.decode(
                encoding,
                errors="strict",
            )
        except (LookupError, UnicodeDecodeError) as exc:
            raise EnvelopeDecryptionError(
                f"Decrypted payload is not valid "
                f"{encoding!r} text."
            ) from exc

    def json(self) -> Any:
        """Parse a JSON envelope payload."""

        if self.metadata.payload_type is not (
            EnvelopePayloadType.JSON
        ):
            raise InvalidEnvelopeError(
                "Envelope payload type is not JSON."
            )

        try:
            return json.loads(
                self.plaintext.decode(
                    "utf-8",
                    errors="strict",
                )
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise EnvelopeDecryptionError(
                "Decrypted payload is not valid JSON."
            ) from exc


class EnvelopeService:
    """
    Create, sign, verify, and decrypt OSHA secure envelopes.

    Separate signing and verification services may be supplied. A producer
    normally has a signing service with a private key. A consumer normally
    has a verification service containing only the trusted public key.
    """

    def __init__(
        self,
        *,
        encryption_service: EncryptionService,
        signing_service: SignatureService | None = None,
        verification_service: SignatureService | None = None,
    ) -> None:
        if not isinstance(
            encryption_service,
            EncryptionService,
        ):
            raise TypeError(
                "encryption_service must be EncryptionService."
            )

        if (
            signing_service is not None
            and not isinstance(
                signing_service,
                SignatureService,
            )
        ):
            raise TypeError(
                "signing_service must be SignatureService or None."
            )

        if (
            verification_service is not None
            and not isinstance(
                verification_service,
                SignatureService,
            )
        ):
            raise TypeError(
                "verification_service must be "
                "SignatureService or None."
            )

        self._encryption_service = encryption_service
        self._signing_service = signing_service
        self._verification_service = verification_service

    @property
    def signing_key_reference(self) -> KeyReference | None:
        if self._signing_service is None:
            return None
        return self._signing_service.key_reference

    @property
    def verification_key_reference(self) -> KeyReference | None:
        if self._verification_service is None:
            return None
        return self._verification_service.key_reference

    def seal_bytes(
        self,
        plaintext: bytes | bytearray | memoryview,
        *,
        encryption_key: KeyReference,
        source: str,
        destination: str,
        purpose: EnvelopePurpose,
        correlation_id: str | None = None,
        content_type: str = "application/octet-stream",
        schema_version: str = "1",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        sign: bool = True,
    ) -> SecureEnvelope:
        """Encrypt bytes and optionally sign the envelope."""

        plaintext_bytes = _require_bytes(
            plaintext,
            field="plaintext",
        )

        metadata = _new_metadata(
            source=source,
            destination=destination,
            purpose=purpose,
            payload_type=EnvelopePayloadType.BYTES,
            content_type=content_type,
            schema_version=schema_version,
            correlation_id=correlation_id,
            ttl_seconds=ttl_seconds,
        )

        return self._seal(
            plaintext_bytes,
            encryption_key=encryption_key,
            metadata=metadata,
            sign=sign,
        )

    def seal_text(
        self,
        plaintext: str,
        *,
        encryption_key: KeyReference,
        source: str,
        destination: str,
        purpose: EnvelopePurpose,
        correlation_id: str | None = None,
        content_type: str = "text/plain; charset=utf-8",
        schema_version: str = "1",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        sign: bool = True,
        encoding: str = "utf-8",
    ) -> SecureEnvelope:
        """Encode, encrypt, and optionally sign text."""

        if not isinstance(plaintext, str):
            raise InvalidEnvelopeError(
                "plaintext must be a string."
            )

        try:
            plaintext_bytes = plaintext.encode(
                encoding,
                errors="strict",
            )
        except (LookupError, UnicodeEncodeError) as exc:
            raise InvalidEnvelopeError(
                f"Unable to encode plaintext using "
                f"{encoding!r}."
            ) from exc

        metadata = _new_metadata(
            source=source,
            destination=destination,
            purpose=purpose,
            payload_type=EnvelopePayloadType.TEXT,
            content_type=content_type,
            schema_version=schema_version,
            correlation_id=correlation_id,
            ttl_seconds=ttl_seconds,
        )

        return self._seal(
            plaintext_bytes,
            encryption_key=encryption_key,
            metadata=metadata,
            sign=sign,
        )

    def seal_json(
        self,
        value: Any,
        *,
        encryption_key: KeyReference,
        source: str,
        destination: str,
        purpose: EnvelopePurpose,
        correlation_id: str | None = None,
        content_type: str = "application/json",
        schema_version: str = "1",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        sign: bool = True,
    ) -> SecureEnvelope:
        """Canonicalize, encrypt, and optionally sign JSON."""

        try:
            plaintext_bytes = canonical_json_bytes(
                value
            )
        except Exception as exc:
            raise InvalidEnvelopeError(
                "Payload cannot be canonicalized as JSON."
            ) from exc

        metadata = _new_metadata(
            source=source,
            destination=destination,
            purpose=purpose,
            payload_type=EnvelopePayloadType.JSON,
            content_type=content_type,
            schema_version=schema_version,
            correlation_id=correlation_id,
            ttl_seconds=ttl_seconds,
        )

        return self._seal(
            plaintext_bytes,
            encryption_key=encryption_key,
            metadata=metadata,
            sign=sign,
        )

    def open(
        self,
        envelope: SecureEnvelope,
        *,
        require_signature: bool = True,
        expected_source: str | None = None,
        expected_destination: str | None = None,
        expected_purpose: EnvelopePurpose | None = None,
        now_ns: int | None = None,
    ) -> OpenedEnvelope:
        """
        Validate, verify, and decrypt an envelope.

        Validation order:

        1. Validate expected routing metadata.
        2. Validate creation/expiration time.
        3. Recalculate encrypted-payload digest.
        4. Verify the digital signature.
        5. Decrypt and authenticate the payload.
        """

        if not isinstance(envelope, SecureEnvelope):
            raise InvalidEnvelopeError(
                "envelope must be SecureEnvelope."
            )

        self._validate_expectations(
            envelope.metadata,
            expected_source=expected_source,
            expected_destination=expected_destination,
            expected_purpose=expected_purpose,
        )

        self._validate_time(
            envelope.metadata,
            now_ns=now_ns,
        )

        calculated_digest = _encrypted_payload_digest(
            envelope.encrypted_payload
        )

        if calculated_digest != envelope.payload_digest:
            raise EnvelopeVerificationError(
                "Encrypted payload digest verification failed."
            )

        signature_key_reference: KeyReference | None = None

        if envelope.signature is None:
            if require_signature:
                raise EnvelopeVerificationError(
                    "A signed envelope is required."
                )
        else:
            if self._verification_service is None:
                raise EnvelopeVerificationError(
                    "No trusted signature verification service "
                    "is configured."
                )

            try:
                verified = self._verification_service.verify(
                    envelope.signing_bytes(),
                    envelope.signature,
                    context=_ENVELOPE_SIGNATURE_CONTEXT,
                )
            except SignatureError as exc:
                raise EnvelopeVerificationError(
                    "Envelope signature verification failed."
                ) from exc

            if not verified:
                raise EnvelopeVerificationError(
                    "Envelope signature is not valid."
                )

            signature_key_reference = (
                envelope.signature.key_reference
            )

        associated_data = _metadata_aad(
            envelope.metadata
        )

        try:
            plaintext = self._encryption_service.decrypt(
                envelope.encrypted_payload,
                associated_data=associated_data,
            )
        except (
            DecryptionError,
            EncryptionError,
        ) as exc:
            raise EnvelopeDecryptionError(
                "Envelope payload authentication or "
                "decryption failed."
            ) from exc

        return OpenedEnvelope(
            metadata=envelope.metadata,
            plaintext=plaintext,
            signed=envelope.signature is not None,
            signature_key_reference=signature_key_reference,
        )

    def _seal(
        self,
        plaintext: bytes,
        *,
        encryption_key: KeyReference,
        metadata: EnvelopeMetadata,
        sign: bool,
    ) -> SecureEnvelope:
        associated_data = _metadata_aad(
            metadata
        )

        try:
            encrypted_payload = (
                self._encryption_service.encrypt(
                    plaintext,
                    key_reference=encryption_key,
                    associated_data=associated_data,
                )
            )
        except EncryptionError as exc:
            raise EnvelopeEncryptionError(
                "Unable to encrypt envelope payload."
            ) from exc

        payload_digest = _encrypted_payload_digest(
            encrypted_payload
        )

        unsigned_envelope = SecureEnvelope(
            format_version=ENVELOPE_VERSION,
            metadata=metadata,
            encrypted_payload=encrypted_payload,
            payload_digest=payload_digest,
            signature=None,
        )

        if not sign:
            return unsigned_envelope

        if self._signing_service is None:
            raise EnvelopeSigningError(
                "Envelope signing was requested but no "
                "signing service is configured."
            )

        try:
            signature = self._signing_service.sign(
                unsigned_envelope.signing_bytes(),
                context=_ENVELOPE_SIGNATURE_CONTEXT,
            )
        except SignatureError as exc:
            raise EnvelopeSigningError(
                "Unable to sign secure envelope."
            ) from exc

        return SecureEnvelope(
            format_version=unsigned_envelope.format_version,
            metadata=unsigned_envelope.metadata,
            encrypted_payload=(
                unsigned_envelope.encrypted_payload
            ),
            payload_digest=unsigned_envelope.payload_digest,
            signature=signature,
        )

    @staticmethod
    def _validate_expectations(
        metadata: EnvelopeMetadata,
        *,
        expected_source: str | None,
        expected_destination: str | None,
        expected_purpose: EnvelopePurpose | None,
    ) -> None:
        if (
            expected_source is not None
            and metadata.source != expected_source
        ):
            raise EnvelopeVerificationError(
                "Envelope source does not match the "
                "expected source."
            )

        if (
            expected_destination is not None
            and metadata.destination != expected_destination
        ):
            raise EnvelopeVerificationError(
                "Envelope destination does not match the "
                "expected destination."
            )

        if (
            expected_purpose is not None
            and metadata.purpose is not expected_purpose
        ):
            raise EnvelopeVerificationError(
                "Envelope purpose does not match the "
                "expected purpose."
            )

    @staticmethod
    def _validate_time(
        metadata: EnvelopeMetadata,
        *,
        now_ns: int | None,
    ) -> None:
        current_ns = (
            time.time_ns()
            if now_ns is None
            else now_ns
        )

        if (
            isinstance(current_ns, bool)
            or not isinstance(current_ns, int)
            or current_ns <= 0
        ):
            raise InvalidEnvelopeError(
                "now_ns must be a positive integer."
            )

        allowed_future_ns = (
            MAX_CLOCK_SKEW_SECONDS * 1_000_000_000
        )

        if (
            metadata.created_at_ns
            > current_ns + allowed_future_ns
        ):
            raise EnvelopeNotYetValidError(
                "Envelope creation time is too far "
                "in the future."
            )

        if current_ns > metadata.expires_at_ns:
            raise EnvelopeExpiredError(
                "Envelope has expired."
            )


class ReplayCache:
    """
    Thread-independent replay-detection interface.

    This base implementation defines the expected API. A production runtime
    should use a shared atomic store when multiple OSHA workers or replicas
    process envelopes concurrently.
    """

    def check_and_record(
        self,
        envelope_id: str,
        *,
        expires_at_ns: int,
    ) -> None:
        raise NotImplementedError


class InMemoryReplayCache(ReplayCache):
    """
    Simple process-local replay cache for development and tests.

    This implementation is not sufficient for multiple processes, pods, or
    distributed workers because its state is local to one Python process.
    """

    def __init__(self) -> None:
        import threading

        self._lock = threading.RLock()
        self._entries: dict[str, int] = {}

    def check_and_record(
        self,
        envelope_id: str,
        *,
        expires_at_ns: int,
    ) -> None:
        _validate_identifier(
            envelope_id,
            field="envelope_id",
            max_length=MAX_ID_LENGTH,
        )

        now_ns = time.time_ns()

        with self._lock:
            expired_ids = [
                item_id
                for item_id, expiry in self._entries.items()
                if expiry < now_ns
            ]

            for item_id in expired_ids:
                del self._entries[item_id]

            if envelope_id in self._entries:
                raise EnvelopeReplayError(
                    f"Envelope {envelope_id!r} was already processed."
                )

            self._entries[envelope_id] = expires_at_ns


def open_once(
    service: EnvelopeService,
    envelope: SecureEnvelope,
    replay_cache: ReplayCache,
    *,
    require_signature: bool = True,
    expected_source: str | None = None,
    expected_destination: str | None = None,
    expected_purpose: EnvelopePurpose | None = None,
    now_ns: int | None = None,
) -> OpenedEnvelope:
    """
    Open an envelope and reject an already-recorded envelope ID.

    The cache records the envelope only after successful cryptographic
    verification and decryption.
    """

    if not isinstance(service, EnvelopeService):
        raise TypeError(
            "service must be EnvelopeService."
        )

    if not isinstance(replay_cache, ReplayCache):
        raise TypeError(
            "replay_cache must implement ReplayCache."
        )

    opened = service.open(
        envelope,
        require_signature=require_signature,
        expected_source=expected_source,
        expected_destination=expected_destination,
        expected_purpose=expected_purpose,
        now_ns=now_ns,
    )

    replay_cache.check_and_record(
        envelope.metadata.envelope_id,
        expires_at_ns=envelope.metadata.expires_at_ns,
    )

    return opened


def _new_metadata(
    *,
    source: str,
    destination: str,
    purpose: EnvelopePurpose,
    payload_type: EnvelopePayloadType,
    content_type: str,
    schema_version: str,
    correlation_id: str | None,
    ttl_seconds: int,
) -> EnvelopeMetadata:
    if not isinstance(purpose, EnvelopePurpose):
        raise InvalidEnvelopeError(
            "purpose must be EnvelopePurpose."
        )

    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, int)
    ):
        raise InvalidEnvelopeError(
            "ttl_seconds must be an integer."
        )

    if not 1 <= ttl_seconds <= MAX_TTL_SECONDS:
        raise InvalidEnvelopeError(
            f"ttl_seconds must be between 1 and "
            f"{MAX_TTL_SECONDS}."
        )

    created_at_ns = time.time_ns()

    envelope_id = str(uuid.uuid4())

    resolved_correlation_id = (
        correlation_id
        if correlation_id is not None
        else envelope_id
    )

    return EnvelopeMetadata(
        envelope_id=envelope_id,
        correlation_id=resolved_correlation_id,
        source=source,
        destination=destination,
        purpose=purpose,
        payload_type=payload_type,
        content_type=content_type,
        created_at_ns=created_at_ns,
        expires_at_ns=(
            created_at_ns
            + ttl_seconds * 1_000_000_000
        ),
        schema_version=schema_version,
    )


def _metadata_aad(
    metadata: EnvelopeMetadata,
) -> bytes:
    """
    Produce associated data for AES-GCM.

    The encryption service also authenticates its own format and key metadata.
    """

    return canonical_json_bytes(
        {
            "envelope_format": ENVELOPE_FORMAT,
            "envelope_version": ENVELOPE_VERSION,
            "metadata": metadata.as_dict(),
        }
    )


def _encrypted_payload_digest(
    encrypted_payload: EncryptedPayload,
) -> Digest:
    return hash_json(
        encrypted_payload.as_dict(),
        algorithm=HashAlgorithm.SHA256,
    )


def _require_bytes(
    value: bytes | bytearray | memoryview,
    *,
    field: str,
) -> bytes:
    if not isinstance(
        value,
        (bytes, bytearray, memoryview),
    ):
        raise InvalidEnvelopeError(
            f"{field} must be bytes-like."
        )

    try:
        view = memoryview(value)

        if not view.contiguous:
            raise InvalidEnvelopeError(
                f"{field} must be contiguous."
            )

        return bytes(view.cast("B"))
    except (TypeError, ValueError) as exc:
        raise InvalidEnvelopeError(
            f"{field} cannot be interpreted as bytes."
        ) from exc


def _validate_exact_fields(
    value: Mapping[str, Any],
    expected_fields: set[str],
    *,
    object_name: str,
) -> None:
    actual_fields = set(value.keys())

    missing = expected_fields - actual_fields
    unknown = actual_fields - expected_fields

    if missing:
        raise InvalidEnvelopeError(
            f"{object_name} is missing fields: "
            + ", ".join(sorted(missing))
            + "."
        )

    if unknown:
        raise InvalidEnvelopeError(
            f"{object_name} contains unknown fields: "
            + ", ".join(
                sorted(str(item) for item in unknown)
            )
            + "."
        )


def _validate_identifier(
    value: str,
    *,
    field: str,
    max_length: int,
) -> None:
    _validate_text(
        value,
        field=field,
        max_length=max_length,
    )

    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise InvalidEnvelopeError(
            f"{field} contains prohibited characters."
        )


def _validate_text(
    value: str,
    *,
    field: str,
    max_length: int,
) -> None:
    if not isinstance(value, str):
        raise InvalidEnvelopeError(
            f"{field} must be a string."
        )

    if not value:
        raise InvalidEnvelopeError(
            f"{field} cannot be empty."
        )

    if value != value.strip():
        raise InvalidEnvelopeError(
            f"{field} cannot start or end with whitespace."
        )

    if len(value) > max_length:
        raise InvalidEnvelopeError(
            f"{field} exceeds the maximum length of "
            f"{max_length} characters."
        )


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "ENVELOPE_FORMAT",
    "ENVELOPE_VERSION",
    "EnvelopeDecryptionError",
    "EnvelopeEncryptionError",
    "EnvelopeError",
    "EnvelopeExpiredError",
    "EnvelopeMetadata",
    "EnvelopeNotYetValidError",
    "EnvelopePayloadType",
    "EnvelopePurpose",
    "EnvelopeReplayError",
    "EnvelopeService",
    "EnvelopeSigningError",
    "EnvelopeVerificationError",
    "InMemoryReplayCache",
    "InvalidEnvelopeError",
    "OpenedEnvelope",
    "ReplayCache",
    "SecureEnvelope",
    "open_once",
]
