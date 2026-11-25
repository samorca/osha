"""
Central cryptographic verification services for OSHA.

This module provides a common verification boundary for cryptographically
protected OSHA data and artifacts.

It coordinates the primitives implemented by:

    hashing.py
    key_provider.py
    encryption.py
    signatures.py
    envelopes.py

This module does NOT implement new cryptographic algorithms.

Its job is to answer questions such as:

- Does this byte sequence match its trusted digest?
- Does this file match the expected artifact digest?
- Is this digital signature valid for the expected signer?
- Is this secure envelope authentic and decryptable?
- Did envelope routing metadata match what the consumer expected?
- Is the envelope expired?
- Has the envelope already been processed?
- Can a protected object be accepted by the next OSHA subsystem?

Security model
--------------
Verification fails closed.

Malformed cryptographic objects are not treated as successful verification.
Exceptions are converted into structured verification results unless the
caller explicitly requests an exception through require_* methods.

A cryptographically valid object is not automatically semantically safe.
Policy, authorization, model-quality assessment, and remediation safety remain
the responsibility of higher OSHA layers.
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .envelopes import (
    EnvelopeDecryptionError,
    EnvelopeError,
    EnvelopeExpiredError,
    EnvelopeNotYetValidError,
    EnvelopePurpose,
    EnvelopeReplayError,
    EnvelopeService,
    EnvelopeVerificationError,
    OpenedEnvelope,
    ReplayCache,
    SecureEnvelope,
    open_once,
)
from .hashing import (
    Digest,
    HashingError,
    InvalidDigestError,
    verify_bytes,
    verify_file,
    verify_json,
    verify_text,
)
from .key_provider import (
    KeyReference,
)
from .signatures import (
    InvalidSignatureEnvelopeError,
    InvalidSigningInputError,
    SignatureEnvelope,
    SignatureError,
    SignatureService,
)


VERIFICATION_FORMAT_VERSION: Final[int] = 1


class VerificationError(Exception):
    """Base exception for OSHA verification failures."""


class VerificationFailedError(VerificationError):
    """Raised when a required verification does not succeed."""


class InvalidVerificationInputError(
    VerificationError,
    ValueError,
):
    """Verification input is malformed or unsupported."""


class VerificationKind(str, Enum):
    """Types of verification performed by OSHA."""

    BYTES_DIGEST = "bytes_digest"
    TEXT_DIGEST = "text_digest"
    JSON_DIGEST = "json_digest"
    FILE_DIGEST = "file_digest"
    SIGNATURE = "signature"
    ENVELOPE = "envelope"


class VerificationStatus(str, Enum):
    """Outcome of a verification operation."""

    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    REPLAYED = "replayed"
    MALFORMED = "malformed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """
    Structured result of an OSHA verification operation.

    reason is intentionally human-readable but should not contain secrets,
    plaintext, private keys, or raw credentials.
    """

    kind: VerificationKind
    status: VerificationStatus
    reason: str
    verified_at_ns: int
    key_reference: KeyReference | None = None
    object_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.kind,
            VerificationKind,
        ):
            raise InvalidVerificationInputError(
                "kind must be VerificationKind."
            )

        if not isinstance(
            self.status,
            VerificationStatus,
        ):
            raise InvalidVerificationInputError(
                "status must be VerificationStatus."
            )

        if (
            not isinstance(self.reason, str)
            or not self.reason
        ):
            raise InvalidVerificationInputError(
                "reason must be a non-empty string."
            )

        if (
            isinstance(self.verified_at_ns, bool)
            or not isinstance(
                self.verified_at_ns,
                int,
            )
            or self.verified_at_ns <= 0
        ):
            raise InvalidVerificationInputError(
                "verified_at_ns must be a positive integer."
            )

        if (
            self.key_reference is not None
            and not isinstance(
                self.key_reference,
                KeyReference,
            )
        ):
            raise InvalidVerificationInputError(
                "key_reference must be KeyReference or None."
            )

        if (
            self.object_id is not None
            and (
                not isinstance(self.object_id, str)
                or not self.object_id
            )
        ):
            raise InvalidVerificationInputError(
                "object_id must be a non-empty string or None."
            )

    @property
    def valid(self) -> bool:
        """Return True only for successful verification."""

        return self.status is VerificationStatus.VALID

    def require_valid(self) -> None:
        """
        Raise VerificationFailedError unless verification succeeded.
        """

        if not self.valid:
            raise VerificationFailedError(
                f"{self.kind.value} verification failed: "
                f"{self.status.value}: {self.reason}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible non-secret representation."""

        return {
            "format_version": VERIFICATION_FORMAT_VERSION,
            "key_reference": (
                None
                if self.key_reference is None
                else str(self.key_reference)
            ),
            "kind": self.kind.value,
            "object_id": self.object_id,
            "reason": self.reason,
            "status": self.status.value,
            "verified_at_ns": self.verified_at_ns,
        }


@dataclass(frozen=True, slots=True)
class EnvelopeVerificationResult:
    """
    Verification result for a protected envelope.

    opened is available only when verification and decryption succeeded.
    """

    verification: VerificationResult
    opened: OpenedEnvelope | None

    @property
    def valid(self) -> bool:
        return (
            self.verification.valid
            and self.opened is not None
        )

    def require_valid(self) -> OpenedEnvelope:
        """
        Return the opened envelope or raise VerificationFailedError.
        """

        self.verification.require_valid()

        if self.opened is None:
            raise VerificationFailedError(
                "Envelope verification succeeded without "
                "an opened payload."
            )

        return self.opened


class VerificationService:
    """
    Central verification coordinator for OSHA.

    Optional services are injected because not every OSHA process needs every
    verification capability.

    For example:

    - artifact registries may only verify hashes
    - model loaders may verify file digests and signatures
    - remediation workers may verify secure envelopes and replay state
    """

    def __init__(
        self,
        *,
        signature_service: SignatureService | None = None,
        envelope_service: EnvelopeService | None = None,
        replay_cache: ReplayCache | None = None,
    ) -> None:

        if (
            signature_service is not None
            and not isinstance(
                signature_service,
                SignatureService,
            )
        ):
            raise TypeError(
                "signature_service must be "
                "SignatureService or None."
            )

        if (
            envelope_service is not None
            and not isinstance(
                envelope_service,
                EnvelopeService,
            )
        ):
            raise TypeError(
                "envelope_service must be "
                "EnvelopeService or None."
            )

        if (
            replay_cache is not None
            and not isinstance(
                replay_cache,
                ReplayCache,
            )
        ):
            raise TypeError(
                "replay_cache must implement ReplayCache."
            )

        self._signature_service = signature_service
        self._envelope_service = envelope_service
        self._replay_cache = replay_cache

    def verify_bytes_digest(
        self,
        data: bytes | bytearray | memoryview,
        expected: Digest | str,
    ) -> VerificationResult:
        """Verify bytes against a trusted digest."""

        try:
            valid = verify_bytes(
                data,
                expected,
            )

        except (
            HashingError,
            InvalidDigestError,
            TypeError,
            ValueError,
        ):
            return _result(
                kind=VerificationKind.BYTES_DIGEST,
                status=VerificationStatus.MALFORMED,
                reason=(
                    "Digest or byte input is malformed."
                ),
            )

        if not valid:
            return _result(
                kind=VerificationKind.BYTES_DIGEST,
                status=VerificationStatus.INVALID,
                reason=(
                    "Byte content does not match the expected digest."
                ),
            )

        return _result(
            kind=VerificationKind.BYTES_DIGEST,
            status=VerificationStatus.VALID,
            reason="Byte digest verified successfully.",
        )

    def verify_text_digest(
        self,
        text: str,
        expected: Digest | str,
        *,
        encoding: str = "utf-8",
    ) -> VerificationResult:
        """Verify text against a trusted digest."""

        try:
            valid = verify_text(
                text,
                expected,
                encoding=encoding,
            )

        except (
            HashingError,
            InvalidDigestError,
            TypeError,
            ValueError,
        ):
            return _result(
                kind=VerificationKind.TEXT_DIGEST,
                status=VerificationStatus.MALFORMED,
                reason=(
                    "Digest or text input is malformed."
                ),
            )

        if not valid:
            return _result(
                kind=VerificationKind.TEXT_DIGEST,
                status=VerificationStatus.INVALID,
                reason=(
                    "Text does not match the expected digest."
                ),
            )

        return _result(
            kind=VerificationKind.TEXT_DIGEST,
            status=VerificationStatus.VALID,
            reason="Text digest verified successfully.",
        )

    def verify_json_digest(
        self,
        value: Any,
        expected: Digest | str,
    ) -> VerificationResult:
        """
        Verify canonical JSON data against a trusted digest.
        """

        try:
            valid = verify_json(
                value,
                expected,
            )

        except (
            HashingError,
            InvalidDigestError,
            TypeError,
            ValueError,
        ):
            return _result(
                kind=VerificationKind.JSON_DIGEST,
                status=VerificationStatus.MALFORMED,
                reason=(
                    "Digest or JSON input is malformed."
                ),
            )

        if not valid:
            return _result(
                kind=VerificationKind.JSON_DIGEST,
                status=VerificationStatus.INVALID,
                reason=(
                    "JSON content does not match the expected digest."
                ),
            )

        return _result(
            kind=VerificationKind.JSON_DIGEST,
            status=VerificationStatus.VALID,
            reason="JSON digest verified successfully.",
        )

    def verify_file_digest(
        self,
        path: str | Path,
        expected: Digest | str,
        *,
        chunk_size: int = 1024 * 1024,
        reject_symlinks: bool = True,
        require_stable_file: bool = True,
    ) -> VerificationResult:
        """
        Verify an artifact or other regular file against a trusted digest.

        Symbolic links and files modified during hashing are rejected by
        default through hashing.py.
        """

        object_id = str(path)

        try:
            valid = verify_file(
                path,
                expected,
                chunk_size=chunk_size,
                reject_symlinks=reject_symlinks,
                require_stable_file=require_stable_file,
            )

        except (
            HashingError,
            InvalidDigestError,
            OSError,
            TypeError,
            ValueError,
        ):
            return _result(
                kind=VerificationKind.FILE_DIGEST,
                status=VerificationStatus.MALFORMED,
                reason=(
                    "File or expected digest could not be "
                    "safelyy verified."
                ),
                object_id=object_id,
           )

        if not valid:
            return _result(
                kind=VerificationKind.FILE_DIGEST,
                status=VerificationStatus.INVALID,
                reason=(
                    "File does not match the expected digest."
                ),
                object_id=object_id,
            )

        return _result(
            kind=VerificationKind.FILE_DIGEST,
            status=VerificationStatus.VALID,
            reason="File digest verified successfully.",
            object_id=object_id,
        )

    def verify_signature(
        self,
        message: bytes | bytearray | memoryview,
        envelope: SignatureEnvelope,
        *,
        context: (
            bytes
            | bytearray
            | memoryview
            | None
        ) = None,
    ) -> VerificationResult:
        """
        Verify a digital signature using the configured trusted verifier.

        The public key embedded in SignatureEnvelope does not establish trust
        by itself. SignatureService compares it against its trusted public key.
        """

        if self._signature_service is None:
            return _result(
                kind=VerificationKind.SIGNATURE,
                status=VerificationStatus.ERROR,
                reason=(
                    "No trusted signature verification service "
                    "is configured."
                ),
            )

        key_reference: KeyReference | None = None

        if isinstance(
            envelope,
            SignatureEnvelope,
        ):
            key_reference = envelope.key_reference

        try:
            valid = self._signature_service.verify(
                message,
                envelope,
                context=context,
            )

        except (
            SignatureError,
            InvalidSignatureEnvelopeError,
            InvalidSigningInputError,
            TypeError,
            ValueError,
        ):
            return _result(
                kind=VerificationKind.SIGNATURE,
                status=VerificationStatus.MALFORMED,
                reason=(
                    "Signature object or signing input "
                    "is malformed."
                ),
                key_reference=key_reference,
            )

        if not valid:
            return _result(
                kind=VerificationKind.SIGNATURE,
                status=VerificationStatus.INVALID,
                reason=(
                    "Digital signature verification failed."
                ),
                key_reference=key_reference,
            )

        return _result(
            kind=VerificationKind.SIGNATURE,
            status=VerificationStatus.VALID,
            reason=(
                "Digital signature verified successfully."
            ),
            key_reference=key_reference,
        )

    def verify_json_signature(
        self,
        value: Any,
        envelope: SignatureEnvelope,
        *,
        context: (
            bytes
            | bytearray
            | memoryview
            | None
        ) = None,
    ) -> VerificationResult:
        """Verify a signed canonical JSON value."""

        if self._signature_service is None:
            return _result(
                kind=VerificationKind.SIGNATURE,
                status=VerificationStatus.ERROR,
                reason=(
                    "No trusted signature verification service "
                    "is configured."
                ),
            )

        key_reference: KeyReference | None = None

        if isinstance(
            envelope,
            SignatureEnvelope,
        ):
            key_reference = envelope.key_reference

        try:
            valid = (
                self._signature_service.verify_json(
                    value,
                    envelope,
                    context=context,
                )
            )

        except (
            SignatureError,
            InvalidSignatureEnvelopeError,
            InvalidSigningInputError,
            TypeError,
            ValueError,
        ):
            return _result(
                kind=VerificationKind.SIGNATURE,
                status=VerificationStatus.MALFORMED,
                reason=(
                    "Signature or JSON input is malformed."
                ),
                key_reference=key_reference,
            )

        if not valid:
            return _result(
                kind=VerificationKind.SIGNATURE,
                status=VerificationStatus.INVALID,
                reason=(
                    "JSON digital signature verification failed."
                ),
                key_reference=key_reference,
            )

        return _result(
            kind=VerificationKind.SIGNATURE,
            status=VerificationStatus.VALID,
            reason=(
                "JSON digital signature verified successfully."
            ),
            key_reference=key_reference,
        )

    def verify_envelope(
        self,
        envelope: SecureEnvelope,
        *,
        require_signature: bool = True,
        reject_replay: bool = True,
        expected_source: str | None = None,
        expected_destination: str | None = None,
        expected_purpose: EnvelopePurpose | None = None,
        now_ns: int | None = None,
    ) -> EnvelopeVerificationResult:
        """
        Verify and decrypt an OSHA secure envelope.

        When reject_replay is True, a ReplayCache must be configured.

        Successful verification means:

        - envelope structure was accepted
        - routing expectations matched
        - the envelope is within its accepted lifetime
        - encrypted-payload digest matched
        - required signature verified
        - AES-GCM authentication succeeded
        - payload decrypted successfully
        - replay protection succeeded when requested
        """

        if self._envelope_service is None:
            result = _result(
                kind=VerificationKind.ENVELOPE,
                status=VerificationStatus.ERROR,
                reason=(
                    "No envelope verification service "
                    "is configured."
                ),
            )

            return EnvelopeVerificationResult(
                verification=result,
                opened=None,
            )

        object_id: str | None = None

        if isinstance(
            envelope,
            SecureEnvelope,
        ):
            object_id = (
                envelope.metadata.envelope_id
            )

        if reject_replay and self._replay_cache is None:
            result = _result(
                kind=VerificationKind.ENVELOPE,
                status=VerificationStatus.ERROR,
                reason=(
                    "Replay rejection was requested but "
                    "no ReplayCache is configured."
                ),
                object_id=object_id,
            )

            return EnvelopeVerificationResult(
                verification=result,
                opened=None,
            )

        try:
            if reject_replay:
                assert self._replay_cache is not None

                opened = open_once(
                    self._envelope_service,
                    envelope,
                    self._replay_cache,
                    require_signature=require_signature,
                    expected_source=expected_source,
                    expected_destination=expected_destination,
                    expected_purpose=expected_purpose,
                    now_ns=now_ns,
                )

            else:
                opened = self._envelope_service.open(
                    envelope,
                    require_signature=require_signature,
                    expected_source=expected_source,
                    expected_destination=expected_destination,
                    expected_purpose=expected_purpose,
                    now_ns=now_ns,
                )

        except EnvelopeExpiredError:
            result = _result(
                kind=VerificationKind.ENVELOPE,
                status=VerificationStatus.EXPIRED,
                reason="Secure envelope has expired.",
                object_id=object_id,
            )

            return EnvelopeVerificationResult(
                verification=result,
                opened=None,
            )

        except EnvelopeNotYetValidError:
            result = _result(
                kind=VerificationKind.ENVELOPE,
                status=VerificationStatus.NOT_YET_VALID,
                reason=(
                    "Secure envelope creation time is "
                    "not yet acceptable."
                ),
                object_id=object_id,
            )

            return EnvelopeVerificationResult(
                verification=result,
                opened=None,
            )

        except EnvelopeReplayError:
            result = _result(
                kind=VerificationKind.ENVELOPE,
                status=VerificationStatus.REPLAYED,
                reason=(
                    "Secure envelope has already been processed."
                ),
                object_id=object_id,
            )

            return EnvelopeVerificationResult(
                verification=result,
                opened=None,
            )

        except (
            EnvelopeVerificationError,
            EnvelopeDecryptionError,
        ):
            result = _result(
                kind=VerificationKind.ENVELOPE,
                status=VerificationStatus.INVALID,
                reason=(
                    "Secure envelope cryptographic "
                    "verification failed."
                ),
                object_id=object_id,
            )

            return EnvelopeVerificationResult(
                verification=result,
                opened=None,
            )

        except (
            EnvelopeError,
            TypeError,
            ValueError,
        ):
            result = _result(
                kind=VerificationKind.ENVELOPE,
                status=VerificationStatus.MALFORMED,
                reason=(
                    "Secure envelope is malformed or unsupported."
                ),
                object_id=object_id,
            )

            return EnvelopeVerificationResult(
                verification=result,
                opened=None,
            )

        key_reference = (
            opened.signature_key_reference
        )

        result = _result(
            kind=VerificationKind.ENVELOPE,
            status=VerificationStatus.VALID,
            reason=(
                "Secure envelope verified and decrypted "
                "successfully."
            ),
            key_reference=key_reference,
            object_id=object_id,
        )

        return EnvelopeVerificationResult(
            verification=result,
            opened=opened,
        )

    def require_bytes_digest(
        self,
        data: bytes | bytearray | memoryview,
        expected: Digest | str,
    ) -> None:
        """Require byte-digest verification to succeed."""

        result = self.verify_bytes_digest(
            data,
            expected,
        )

        result.require_valid()

    def require_file_digest(
        self,
        path: str | Path,
        expected: Digest | str,
        *,
        chunk_size: int = 1024 * 1024,
        reject_symlinks: bool = True,
        require_stable_file: bool = True,
    ) -> None:
        """Require file-digest verification to succeed."""

        result = self.verify_file_digest(
            path,
            expected,
            chunk_size=chunk_size,
            reject_symlinks=reject_symlinks,
            require_stable_file=require_stable_file,
        )

        result.require_valid()

    def require_signature(
        self,
        message: bytes | bytearray | memoryview,
        envelope: SignatureEnvelope,
        *,
        context: (
            bytes
            | bytearray
            | memoryview
            | None
        ) = None,
    ) -> None:
        """Require digital-signature verification."""

        result = self.verify_signature(
            message,
            envelope,
            context=context,
        )

        result.require_valid()

    def require_envelope(
        self,
        envelope: SecureEnvelope,
        *,
        require_signature: bool = True,
        reject_replay: bool = True,
        expected_source: str | None = None,
        expected_destination: str | None = None,
        expected_purpose: EnvelopePurpose | None = None,
        now_ns: int | None = None,
    ) -> OpenedEnvelope:
        """
        Require secure-envelope verification and return plaintext.

        Higher-risk OSHA components such as remediation workers should prefer
        this fail-closed interface instead of manually checking boolean flags.
        """

        result = self.verify_envelope(
            envelope,
            require_signature=require_signature,
            reject_replay=reject_replay,
            expected_source=expected_source,
            expected_destination=expected_destination,
            expected_purpose=expected_purpose,
            now_ns=now_ns,
        )

        return result.require_valid()


def _result(
    *,
    kind: VerificationKind,
    status: VerificationStatus,
    reason: str,
    key_reference: KeyReference | None = None,
    object_id: str | None = None,
) -> VerificationResult:

    return VerificationResult(
        kind=kind,
        status=status,
        reason=reason,
        verified_at_ns=time.time_ns(),
        key_reference=key_reference,
        object_id=object_id,
    )


__all__ = [
    "EnvelopeVerificationResult",
    "InvalidVerificationInputError",
    "VERIFICATION_FORMAT_VERSION",
    "VerificationError",
    "VerificationFailedError",
    "VerificationKind",
    "VerificationResult",
    "VerificationService",
    "VerificationStatus",
]
