"""
Cryptographically signed security audit events for OSHA.

This module provides a canonical audit-event format and Ed25519 signing/
verification for security-sensitive OSHA operations.

Examples of events:

- workload authentication
- permission decisions
- authorization decisions
- remediation decisions
- dataset verification
- model verification
- key lifecycle events
- certificate validation
- provenance verification
- security failures

Security properties
-------------------
1. Events use deterministic canonical JSON before signing.
2. Event identity, timestamp, actor, action, result, target, correlation ID,
   severity, and structured details are covered by the signature.
3. Verification uses a trusted SignatureService rather than trusting the
   public key embedded in the signature envelope.
4. Event details must contain JSON-compatible, non-secret data.
5. Audit events must never contain private keys, credentials, raw tokens,
   passwords, or other secrets.
6. Signed events are immutable value objects.
7. Signing establishes event authenticity and integrity.
8. Signing alone does NOT detect event deletion or reordering.
   integrity_chain.py adds that property.
"""

from __future__ import annotations

import json
import re
import time
import uuid

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Mapping

from ..crypto.hashing import (
    Digest,
    HashAlgorithm,
    canonical_json_bytes,
    hash_json,
)
from ..crypto.key_provider import KeyReference
from ..crypto.signatures import (
    InvalidSignatureEnvelopeError,
    SignatureEnvelope,
    SignatureError,
    SignatureService,
)
from ..identity.workload_identity import (
    InvalidWorkloadIdentityError,
    workload_identity_from_id,
)


AUDIT_EVENT_FORMAT: Final[str] = "osha-audit-event"
AUDIT_EVENT_VERSION: Final[int] = 1

SIGNED_AUDIT_EVENT_FORMAT: Final[str] = (
    "osha-signed-audit-event"
)
SIGNED_AUDIT_EVENT_VERSION: Final[int] = 1

AUDIT_SIGNATURE_CONTEXT: Final[bytes] = (
    b"osha-audit-event-v1"
)

MAX_ID_LENGTH: Final[int] = 128
MAX_ACTION_LENGTH: Final[int] = 256
MAX_TARGET_LENGTH: Final[int] = 1024
MAX_MESSAGE_LENGTH: Final[int] = 4096
MAX_DETAIL_ITEMS: Final[int] = 128
MAX_DETAIL_DEPTH: Final[int] = 8
MAX_DETAIL_STRING_LENGTH: Final[int] = 8192
MAX_SERIALIZED_EVENT_SIZE: Final[int] = (
    8 * 1024 * 1024
)

_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"
)


class AuditEventError(Exception):
    """Base exception for OSHA signed-audit-event failures."""


class InvalidAuditEventError(
    AuditEventError,
    ValueError,
):
    """Audit event is malformed."""


class AuditEventSigningError(
    AuditEventError,
):
    """Audit event could not be signed."""


class AuditEventVerificationError(
    AuditEventError,
):
    """Audit-event signature verification failed."""


class AuditSeverity(str, Enum):
    """Security audit severity."""

    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditOutcome(str, Enum):
    """Outcome represented by an audit event."""

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    REQUIRES_APPROVAL = "requires_approval"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class AuditEventType(str, Enum):
    """Known OSHA security audit-event categories."""

    WORKLOAD_AUTHENTICATED = (
        "workload_authenticated"
    )

    WORKLOAD_AUTHENTICATION_FAILED = (
        "workload_authentication_failed"
    )

    PERMISSION_DECISION = (
        "permission_decision"
    )

    AUTHORIZATION_DECISION = (
        "authorization_decision"
    )

    REMEDIATION_DECISION = (
        "remediation_decision"
    )

    REMEDIATION_EXECUTION = (
        "remediation_execution"
    )

    DATASET_VERIFICATION = (
        "dataset_verification"
    )

    MODEL_VERIFICATION = (
        "model_verification"
    )

    PROVENANCE_VERIFICATION = (
        "provenance_verification"
    )

    CERTIFICATE_VALIDATION = (
        "certificate_validation"
    )

    KEY_CREATED = "key_created"
    KEY_ROTATED = "key_rotated"
    KEY_DISABLED = "key_disabled"
    KEY_DESTROYED = "key_destroyed"

    SECURITY_FAILURE = "security_failure"

    SYSTEM = "system"


class AuditVerificationStatus(str, Enum):
    """Signed audit-event verification result."""

    VALID = "valid"
    INVALID_SIGNATURE = "invalid_signature"
    UNTRUSTED_SIGNER = "untrusted_signer"
    MALFORMED = "malformed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """
    Canonical unsigned OSHA security event.

    actor_workload_id identifies the workload responsible for producing or
    initiating the operation.

    target is a logical target identifier rather than necessarily a path,
    hostname, or URL.

    details must be non-secret JSON-compatible metadata.
    """

    event_id: str
    event_type: AuditEventType

    timestamp_ns: int

    actor_workload_id: str

    action: str
    outcome: AuditOutcome
    severity: AuditSeverity

    target: str

    correlation_id: str

    message: str

    details: Mapping[str, Any] | None = None

    format_version: int = AUDIT_EVENT_VERSION

    def __post_init__(self) -> None:
        if (
            self.format_version
            != AUDIT_EVENT_VERSION
        ):
            raise InvalidAuditEventError(
                f"Unsupported audit-event version: "
                f"{self.format_version!r}."
            )

        _validate_identifier(
            self.event_id,
            field="event_id",
            max_length=MAX_ID_LENGTH,
        )

        if not isinstance(
            self.event_type,
            AuditEventType,
        ):
            raise InvalidAuditEventError(
                "event_type must be AuditEventType."
            )

        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(
                self.timestamp_ns,
                int,
            )
            or self.timestamp_ns <= 0
        ):
            raise InvalidAuditEventError(
                "timestamp_ns must be a positive integer."
            )

        _validate_workload_id(
            self.actor_workload_id
        )

        _validate_text(
            self.action,
            field="action",
            max_length=MAX_ACTION_LENGTH,
        )

        if not isinstance(
            self.outcome,
            AuditOutcome,
        ):
            raise InvalidAuditEventError(
                "outcome must be AuditOutcome."
            )

        if not isinstance(
            self.severity,
            AuditSeverity,
        ):
            raise InvalidAuditEventError(
                "severity must be AuditSeverity."
            )

        _validate_text(
            self.target,
            field="target",
            max_length=MAX_TARGET_LENGTH,
        )

        _validate_identifier(
            self.correlation_id,
            field="correlation_id",
            max_length=MAX_ID_LENGTH,
        )

        _validate_text(
            self.message,
            field="message",
            max_length=MAX_MESSAGE_LENGTH,
        )

        normalized_details = _normalize_details(
            self.details
        )

        object.__setattr__(
            self,
            "details",
            normalized_details,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return canonical JSON-compatible event data."""

        return {
            "action": self.action,
            "actor_workload_id": (
                self.actor_workload_id
            ),
            "correlation_id": self.correlation_id,
            "details": self.details or {},
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "format": AUDIT_EVENT_FORMAT,
            "format_version": self.format_version,
            "message": self.message,
            "outcome": self.outcome.value,
            "severity": self.severity.value,
            "target": self.target,
            "timestamp_ns": self.timestamp_ns,
        }

    def canonical_bytes(self) -> bytes:
        """Return deterministic event bytes."""

        return canonical_json_bytes(
            self.as_dict()
        )

    def digest(
        self,
        algorithm: HashAlgorithm | str = (
            HashAlgorithm.SHA256
        ),
    ) -> Digest:
        """Hash the entire canonical unsigned event."""

        return hash_json(
            self.as_dict(),
            algorithm=algorithm,
        )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "AuditEvent":
        """Parse a strict audit event."""

        if not isinstance(value, Mapping):
            raise InvalidAuditEventError(
                "Audit event must be a mapping."
            )

        expected = {
            "action",
            "actor_workload_id",
            "correlation_id",
            "details",
            "event_id",
            "event_type",
            "format",
            "format_version",
            "message",
            "outcome",
            "severity",
            "target",
            "timestamp_ns",
        }

        _require_exact_fields(
            value,
            expected,
            object_name="audit event",
        )

        if (
            value["format"]
            != AUDIT_EVENT_FORMAT
        ):
            raise InvalidAuditEventError(
                f"Unsupported audit-event format: "
                f"{value['format']!r}."
            )

        try:
            event_type = AuditEventType(
                value["event_type"]
            )
        except (TypeError, ValueError) as exc:
            raise InvalidAuditEventError(
                "Unsupported audit event type."
            ) from exc

        try:
            outcome = AuditOutcome(
                value["outcome"]
            )
        except (TypeError, ValueError) as exc:
            raise InvalidAuditEventError(
                "Unsupported audit outcome."
            ) from exc

        try:
            severity = AuditSeverity(
                value["severity"]
            )
        except (TypeError, ValueError) as exc:
            raise InvalidAuditEventError(
                "Unsupported audit severity."
            ) from exc

        return cls(
            event_id=value["event_id"],
            event_type=event_type,
            timestamp_ns=value["timestamp_ns"],
            actor_workload_id=(
                value["actor_workload_id"]
            ),
            action=value["action"],
            outcome=outcome,
            severity=severity,
            target=value["target"],
            correlation_id=(
                value["correlation_id"]
            ),
            message=value["message"],
            details=value["details"],
            format_version=value[
                "format_version"
            ],
        )


@dataclass(frozen=True, slots=True)
class SignedAuditEvent:
    """
    Digitally signed OSHA audit event.

    signed_at_ns is covered by the signature along with the complete event.
    """

    event: AuditEvent

    signature: SignatureEnvelope

    signed_at_ns: int

    format_version: int = (
        SIGNED_AUDIT_EVENT_VERSION
    )

    def __post_init__(self) -> None:
        if (
            self.format_version
            != SIGNED_AUDIT_EVENT_VERSION
        ):
            raise InvalidAuditEventError(
                f"Unsupported signed-event version: "
                f"{self.format_version!r}."
            )

        if not isinstance(
            self.event,
            AuditEvent,
        ):
            raise InvalidAuditEventError(
                "event must be AuditEvent."
            )

        if not isinstance(
            self.signature,
            SignatureEnvelope,
        ):
            raise InvalidAuditEventError(
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
            raise InvalidAuditEventError(
                "signed_at_ns must be positive."
            )

        if (
            self.signed_at_ns
            < self.event.timestamp_ns
        ):
            raise InvalidAuditEventError(
                "Audit event cannot be signed before "
                "its event timestamp."
            )

    @property
    def signing_key_reference(
        self,
    ) -> KeyReference:
        return self.signature.key_reference

    def unsigned_dict(
        self,
    ) -> dict[str, Any]:
        """Return exactly the object covered by the signature."""

        return {
            "event": self.event.as_dict(),
            "format": SIGNED_AUDIT_EVENT_FORMAT,
            "format_version": self.format_version,
            "signed_at_ns": self.signed_at_ns,
        }

    def signing_bytes(self) -> bytes:
        """Return the deterministic bytes that were signed."""

        return canonical_json_bytes(
            self.unsigned_dict()
        )

    def digest(
        self,
    ) -> Digest:
        """
        Hash the full signed event.

        integrity_chain.py will use this value to link records.
        """

        return hash_json(
            self.as_dict(),
            algorithm=HashAlgorithm.SHA256,
        )

    def as_dict(
        self,
    ) -> dict[str, Any]:
        """Return serialized signed-event structure."""

        return {
            **self.unsigned_dict(),
            "signature": (
                self.signature.as_dict()
            ),
        }

    def to_json(
        self,
    ) -> str:
        """Serialize signed event as deterministic JSON."""

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
    ) -> "SignedAuditEvent":
        """Parse a strict signed audit event."""

        if not isinstance(value, Mapping):
            raise InvalidAuditEventError(
                "Signed audit event must be a mapping."
            )

        expected = {
            "event",
            "format",
            "format_version",
            "signature",
            "signed_at_ns",
        }

        _require_exact_fields(
            value,
            expected,
            object_name="signed audit event",
        )

        if (
            value["format"]
            != SIGNED_AUDIT_EVENT_FORMAT
        ):
            raise InvalidAuditEventError(
                f"Unsupported signed-event format: "
                f"{value['format']!r}."
            )

        try:
            event = AuditEvent.from_dict(
                value["event"]
            )
        except InvalidAuditEventError:
            raise
        except Exception as exc:
            raise InvalidAuditEventError(
                "Embedded audit event is invalid."
            ) from exc

        try:
            signature = (
                SignatureEnvelope.from_dict(
                    value["signature"]
                )
            )
        except InvalidSignatureEnvelopeError as exc:
            raise InvalidAuditEventError(
                "Embedded event signature is invalid."
            ) from exc

        return cls(
            event=event,
            signature=signature,
            signed_at_ns=value["signed_at_ns"],
            format_version=value[
                "format_version"
            ],
        )

    @classmethod
    def from_json(
        cls,
        encoded: str | bytes,
    ) -> "SignedAuditEvent":
        """Parse JSON-encoded signed audit event."""

        if isinstance(encoded, bytes):
            try:
                encoded = encoded.decode(
                    "utf-8",
                    errors="strict",
                )
            except UnicodeDecodeError as exc:
                raise InvalidAuditEventError(
                    "Signed audit event is not "
                    "valid UTF-8."
                ) from exc

        if not isinstance(encoded, str):
            raise InvalidAuditEventError(
                "Encoded event must be string "
                "or UTF-8 bytes."
            )

        if len(encoded.encode("utf-8")) > (
            MAX_SERIALIZED_EVENT_SIZE
        ):
            raise InvalidAuditEventError(
                "Signed audit event exceeds "
                "supported size."
            )

        try:
            value = json.loads(
                encoded
            )
        except json.JSONDecodeError as exc:
            raise InvalidAuditEventError(
                "Signed audit event contains "
                "invalid JSON."
            ) from exc

        return cls.from_dict(
            value
        )


@dataclass(frozen=True, slots=True)
class AuditVerificationResult:
    """Result of trusted audit-event verification."""

    status: AuditVerificationStatus

    event_id: str | None
    correlation_id: str | None

    signing_key_reference: (
        KeyReference | None
    )

    signature_valid: bool

    reason: str

    verified_at_ns: int

    @property
    def valid(self) -> bool:
        return (
            self.status
            is AuditVerificationStatus.VALID
            and self.signature_valid
        )

    def require_valid(self) -> None:
        if not self.valid:
            raise AuditEventVerificationError(
                self.reason
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "correlation_id": (
                self.correlation_id
            ),
            "event_id": self.event_id,
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


class AuditSigningService:
    """Create signed OSHA security audit events."""

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

        if not signature_service.can_sign:
            raise AuditEventSigningError(
                "Audit signing requires a "
                "SignatureService with private key."
            )

        self._signature_service = (
            signature_service
        )

    def sign(
        self,
        event: AuditEvent,
        *,
        signed_at_ns: int | None = None,
    ) -> SignedAuditEvent:
        """Digitally sign one audit event."""

        if not isinstance(
            event,
            AuditEvent,
        ):
            raise AuditEventSigningError(
                "event must be AuditEvent."
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
            raise AuditEventSigningError(
                "signed_at_ns must be positive."
            )

        if resolved_time < event.timestamp_ns:
            raise AuditEventSigningError(
                "Event cannot be signed before "
                "its timestamp."
            )

        unsigned = {
            "event": event.as_dict(),
            "format": (
                SIGNED_AUDIT_EVENT_FORMAT
            ),
            "format_version": (
                SIGNED_AUDIT_EVENT_VERSION
            ),
            "signed_at_ns": resolved_time,
        }

        try:
            signature = (
                self._signature_service.sign(
                    canonical_json_bytes(
                        unsigned
                    ),
                    context=(
                        AUDIT_SIGNATURE_CONTEXT
                    ),
                )
            )
        except SignatureError as exc:
            raise AuditEventSigningError(
                "Unable to sign audit event."
            ) from exc

        return SignedAuditEvent(
            event=event,
            signature=signature,
            signed_at_ns=resolved_time,
        )


class AuditVerificationService:
    """
    Verify audit events against a trusted signing service.

    trusted_signing_keys limits which key references may sign accepted
    audit events.
    """

    def __init__(
        self,
        *,
        signature_service: SignatureService,
        trusted_signing_keys: frozenset[
            KeyReference
        ],
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
            trusted_signing_keys,
            frozenset,
        ):
            raise TypeError(
                "trusted_signing_keys must be frozenset."
            )

        if not trusted_signing_keys:
            raise AuditEventVerificationError(
                "At least one trusted audit signing "
                "key is required."
            )

        for reference in trusted_signing_keys:
            if not isinstance(
                reference,
                KeyReference,
            ):
                raise TypeError(
                    "Every trusted signing key must "
                    "be KeyReference."
                )

        self._signature_service = (
            signature_service
        )

        self._trusted_signing_keys = (
            trusted_signing_keys
        )

    def verify(
        self,
        signed_event: SignedAuditEvent,
    ) -> AuditVerificationResult:
        """Verify signer trust and digital signature."""

        if not isinstance(
            signed_event,
            SignedAuditEvent,
        ):
            return _verification_result(
                status=(
                    AuditVerificationStatus
                    .MALFORMED
                ),
                reason=(
                    "signed_event must be "
                    "SignedAuditEvent."
                ),
            )

        reference = (
            signed_event
            .signing_key_reference
        )

        if reference not in (
            self._trusted_signing_keys
        ):
            return _verification_result(
                status=(
                    AuditVerificationStatus
                    .UNTRUSTED_SIGNER
                ),
                reason=(
                    "Audit signing-key reference "
                    "is not trusted."
                ),
                signed_event=signed_event,
            )

        try:
            valid = (
                self._signature_service.verify(
                    signed_event.signing_bytes(),
                    signed_event.signature,
                    context=(
                        AUDIT_SIGNATURE_CONTEXT
                    ),
                )
            )

        except SignatureError:
            valid = False

        if not valid:
            return _verification_result(
                status=(
                    AuditVerificationStatus
                    .INVALID_SIGNATURE
                ),
                reason=(
                    "Audit-event signature "
                    "verification failed."
                ),
                signed_event=signed_event,
                signature_valid=False,
            )

        return _verification_result(
            status=AuditVerificationStatus.VALID,
            reason=(
                "Audit-event signature verified "
                "successfully."
            ),
            signed_event=signed_event,
            signature_valid=True,
        )

    def require_verified(
        self,
        signed_event: SignedAuditEvent,
    ) -> AuditVerificationResult:
        """Require trusted audit signature."""

        result = self.verify(
            signed_event
        )

        result.require_valid()

        return result


def new_audit_event(
    *,
    event_type: AuditEventType,
    actor_workload_id: str,
    action: str,
    outcome: AuditOutcome,
    severity: AuditSeverity,
    target: str,
    correlation_id: str,
    message: str,
    details: Mapping[str, Any] | None = None,
    timestamp_ns: int | None = None,
) -> AuditEvent:
    """Create a new unsigned audit event."""

    resolved_time = (
        time.time_ns()
        if timestamp_ns is None
        else timestamp_ns
    )

    return AuditEvent(
        event_id=str(
            uuid.uuid4()
        ),
        event_type=event_type,
        timestamp_ns=resolved_time,
        actor_workload_id=(
            actor_workload_id
        ),
        action=action,
        outcome=outcome,
        severity=severity,
        target=target,
        correlation_id=correlation_id,
        message=message,
        details=details,
    )


def _verification_result(
    *,
    status: AuditVerificationStatus,
    reason: str,
    signed_event: SignedAuditEvent | None = None,
    signature_valid: bool = False,
) -> AuditVerificationResult:

    return AuditVerificationResult(
        status=status,
        event_id=(
            None
            if signed_event is None
            else signed_event.event.event_id
        ),
        correlation_id=(
            None
            if signed_event is None
            else signed_event
            .event
            .correlation_id
        ),
        signing_key_reference=(
            None
            if signed_event is None
            else signed_event
            .signing_key_reference
        ),
        signature_valid=signature_valid,
        reason=reason,
        verified_at_ns=time.time_ns(),
    )


def _normalize_details(
    details: Mapping[str, Any] | None,
) -> Mapping[str, Any]:

    if details is None:
        return {}

    if not isinstance(
        details,
        Mapping,
    ):
        raise InvalidAuditEventError(
            "details must be a mapping."
        )

    if len(details) > MAX_DETAIL_ITEMS:
        raise InvalidAuditEventError(
            f"details cannot contain more than "
            f"{MAX_DETAIL_ITEMS} entries."
        )

    normalized: dict[str, Any] = {}

    for key, value in details.items():
        if (
            not isinstance(key, str)
            or not key.strip()
        ):
            raise InvalidAuditEventError(
                "Detail keys must be "
                "non-empty strings."
            )

        normalized_key = (
            key.strip().lower()
        )

        normalized[
            normalized_key
        ] = _validate_detail_value(
            value,
            depth=0,
        )

    return normalized


def _validate_detail_value(
    value: Any,
    *,
    depth: int,
) -> Any:

    if depth > MAX_DETAIL_DEPTH:
        raise InvalidAuditEventError(
            "Audit detail nesting exceeds "
            "the maximum depth."
        )

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if (
            value != value
            or value in (
                float("inf"),
                float("-inf"),
            )
        ):
            raise InvalidAuditEventError(
                "Audit details cannot contain "
                "NaN or infinity."
            )

        return value

    if isinstance(
        value,
        str,
    ):
        if len(value) > (
            MAX_DETAIL_STRING_LENGTH
        ):
            raise InvalidAuditEventError(
                "Audit detail string exceeds "
                "maximum length."
            )

        return value

    if isinstance(
        value,
        Mapping,
    ):
        if len(value) > MAX_DETAIL_ITEMS:
            raise InvalidAuditEventError(
                "Nested audit details contain "
                "too many entries."
            )

        result: dict[str, Any] = {}

        for key, child in value.items():
            if (
                not isinstance(key, str)
                or not key
            ):
                raise InvalidAuditEventError(
                    "Nested detail keys must "
                    "be non-empty strings."
                )

            result[key] = (
                _validate_detail_value(
                    child,
                    depth=depth + 1,
                )
            )

        return result

    if isinstance(
        value,
        (list, tuple),
    ):
        if len(value) > MAX_DETAIL_ITEMS:
            raise InvalidAuditEventError(
                "Audit detail list contains "
                "too many entries."
            )

        return [
            _validate_detail_value(
                child,
                depth=depth + 1,
            )
            for child in value
        ]

    raise InvalidAuditEventError(
        f"Unsupported audit detail type: "
        f"{type(value).__name__}."
    )


def _validate_workload_id(
    value: str,
) -> None:

    try:
        workload_identity_from_id(
            value
        )
    except InvalidWorkloadIdentityError as exc:
        raise InvalidAuditEventError(
            "actor_workload_id is not a valid "
            "OSHA workload identity."
        ) from exc


def _validate_identifier(
    value: str,
    *,
    field: str,
    max_length: int,
) -> None:

    if (
        not isinstance(value, str)
        or not value
    ):
        raise InvalidAuditEventError(
            f"{field} must be a non-empty string."
        )

    if len(value) > max_length:
        raise InvalidAuditEventError(
            f"{field} exceeds maximum length "
            f"of {max_length}."
        )

    if (
        _ID_PATTERN.fullmatch(
            value
        )
        is None
    ):
        raise InvalidAuditEventError(
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
        raise InvalidAuditEventError(
            f"{field} must be non-empty text."
        )

    if len(value) > max_length:
        raise InvalidAuditEventError(
            f"{field} exceeds maximum length "
            f"of {max_length}."
        )


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    object_name: str,
) -> None:

    actual = set(
        value.keys()
    )

    missing = expected - actual
    unknown = actual - expected

    if missing:
        raise InvalidAuditEventError(
            f"{object_name} is missing fields: "
            + ", ".join(
                sorted(missing)
            )
            + "."
        )

    if unknown:
        raise InvalidAuditEventError(
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
    "AUDIT_EVENT_FORMAT",
    "AUDIT_EVENT_VERSION",
    "AUDIT_SIGNATURE_CONTEXT",
    "AuditEvent",
    "AuditEventError",
    "AuditEventSigningError",
    "AuditEventType",
    "AuditEventVerificationError",
    "AuditOutcome",
    "AuditSeverity",
    "AuditSigningService",
    "AuditVerificationResult",
    "AuditVerificationService",
    "AuditVerificationStatus",
    "InvalidAuditEventError",
    "SIGNED_AUDIT_EVENT_FORMAT",
    "SIGNED_AUDIT_EVENT_VERSION",
    "SignedAuditEvent",
    "new_audit_event",
]
