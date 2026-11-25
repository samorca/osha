"""
Tamper-evident audit integrity chains for OSHA.

This module links individually signed OSHA audit events into an ordered
cryptographic chain.

Each chain record commits to:

- chain identity
- sequence number
- previous record digest
- complete signed audit event
- append timestamp

Security properties
-------------------
1. Each event already has an Ed25519 signature from signed_events.py.
2. Each chain record contains the previous record digest.
3. Each chain-record digest covers all record fields.
4. Sequence numbers must be continuous and start at zero.
5. The first record uses a fixed genesis digest.
6. Verification recalculates every record digest.
7. Verification can also validate each embedded event signature.
8. Modification, deletion, insertion, duplication, or reordering breaks
   chain verification unless all following records are rebuilt.
9. External anchoring of the chain head is required if the threat model
   includes an attacker capable of rewriting the complete chain and its
   locally stored head.

This module is storage-agnostic.

The in-memory implementation is intended for development and testing.
Production storage should provide atomic append semantics and durable,
appropriately protected storage.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Iterable, Mapping

from ..crypto.hashing import (
    Digest,
    HashAlgorithm,
    hash_json,
)
from .signed_events import (
    AuditVerificationService,
    SignedAuditEvent,
)


INTEGRITY_CHAIN_FORMAT: Final[str] = (
    "osha-audit-integrity-chain"
)

INTEGRITY_CHAIN_VERSION: Final[int] = 1

CHAIN_RECORD_FORMAT: Final[str] = (
    "osha-audit-chain-record"
)

CHAIN_RECORD_VERSION: Final[int] = 1

CHAIN_HASH_ALGORITHM: Final[
    HashAlgorithm
] = HashAlgorithm.SHA256

GENESIS_DIGEST: Final[Digest] = Digest(
    algorithm=CHAIN_HASH_ALGORITHM,
    hexdigest="0" * 64,
)

MAX_CHAIN_ID_LENGTH: Final[int] = 128

MAX_RECORDS_ON_IMPORT: Final[int] = 10_000_000

MAX_SERIALIZED_CHAIN_SIZE: Final[int] = (
    512 * 1024 * 1024
)

_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"
)


class IntegrityChainError(Exception):
    """Base exception for OSHA integrity-chain failures."""


class InvalidChainRecordError(
    IntegrityChainError,
    ValueError,
):
    """A chain record is malformed."""


class InvalidIntegrityChainError(
    IntegrityChainError,
    ValueError,
):
    """An audit integrity chain is malformed."""


class IntegrityChainVerificationError(
    IntegrityChainError,
):
    """Audit integrity-chain verification failed."""


class DuplicateAuditEventError(
    IntegrityChainError,
):
    """An audit event already exists in the chain."""


class ChainSequenceError(
    IntegrityChainError,
):
    """A record has an invalid sequence number."""


class ChainLinkError(
    IntegrityChainError,
):
    """A record does not correctly link to its predecessor."""


class ChainVerificationStatus(str, Enum):
    """Outcome of complete chain verification."""

    VALID = "valid"

    EMPTY = "empty"

    INVALID_SEQUENCE = "invalid_sequence"

    INVALID_LINK = "invalid_link"

    INVALID_RECORD_DIGEST = (
        "invalid_record_digest"
    )

    INVALID_EVENT_SIGNATURE = (
        "invalid_event_signature"
    )

    DUPLICATE_EVENT = "duplicate_event"

    MALFORMED = "malformed"

    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ChainRecord:
    """
    One immutable record in an OSHA audit chain.

    record_digest is calculated from record_content_dict(),
    which intentionally excludes record_digest itself.
    """

    chain_id: str

    sequence_number: int

    previous_digest: Digest

    signed_event: SignedAuditEvent

    appended_at_ns: int

    record_digest: Digest

    format_version: int = (
        CHAIN_RECORD_VERSION
    )

    def __post_init__(
        self,
    ) -> None:

        _validate_chain_id(
            self.chain_id
        )

        if (
            isinstance(
                self.sequence_number,
                bool,
            )
            or not isinstance(
                self.sequence_number,
                int,
            )
            or self.sequence_number < 0
        ):
            raise InvalidChainRecordError(
                "sequence_number must be a "
                "non-negative integer."
            )

        if not isinstance(
            self.previous_digest,
            Digest,
        ):
            raise InvalidChainRecordError(
                "previous_digest must be Digest."
            )

        if not isinstance(
            self.signed_event,
            SignedAuditEvent,
        ):
            raise InvalidChainRecordError(
                "signed_event must be SignedAuditEvent."
            )

        if (
            isinstance(
                self.appended_at_ns,
                bool,
            )
            or not isinstance(
                self.appended_at_ns,
                int,
            )
            or self.appended_at_ns <= 0
        ):
            raise InvalidChainRecordError(
                "appended_at_ns must be a "
                "positive integer."
            )

        if (
            self.appended_at_ns
            < self.signed_event.signed_at_ns
        ):
            raise InvalidChainRecordError(
                "Chain record cannot be appended before "
                "the audit event was signed."
            )

        if not isinstance(
            self.record_digest,
            Digest,
        ):
            raise InvalidChainRecordError(
                "record_digest must be Digest."
            )

        if (
            self.previous_digest.algorithm
            is not CHAIN_HASH_ALGORITHM
        ):
            raise InvalidChainRecordError(
                "previous_digest must use SHA-256."
            )

        if (
            self.record_digest.algorithm
            is not CHAIN_HASH_ALGORITHM
        ):
            raise InvalidChainRecordError(
                "record_digest must use SHA-256."
            )

        if (
            self.format_version
            != CHAIN_RECORD_VERSION
        ):
            raise InvalidChainRecordError(
                f"Unsupported chain-record version: "
                f"{self.format_version!r}."
            )

    @property
    def event_id(
        self,
    ) -> str:
        """Return embedded audit-event ID."""

        return (
            self.signed_event
            .event
            .event_id
        )

    def record_content_dict(
        self,
    ) -> dict[str, Any]:
        """
        Return exactly the fields committed to by record_digest.
        """

        return {
            "appended_at_ns": (
                self.appended_at_ns
            ),
            "chain_id": self.chain_id,
            "format": CHAIN_RECORD_FORMAT,
            "format_version": (
                self.format_version
            ),
            "previous_digest": str(
                self.previous_digest
            ),
            "sequence_number": (
                self.sequence_number
            ),
            "signed_event": (
                self.signed_event.as_dict()
            ),
        }

    def calculated_digest(
        self,
    ) -> Digest:
        """Recalculate this record's digest."""

        return hash_json(
            self.record_content_dict(),
            algorithm=CHAIN_HASH_ALGORITHM,
        )

    def verify_digest(
        self,
    ) -> bool:
        """Verify this record's stored digest."""

        return (
            self.calculated_digest()
            == self.record_digest
        )

    def as_dict(
        self,
    ) -> dict[str, Any]:
        """Return JSON-compatible chain record."""

        return {
            **self.record_content_dict(),
            "record_digest": str(
                self.record_digest
            ),
        }

    @classmethod
    def create(
        cls,
        *,
        chain_id: str,
        sequence_number: int,
        previous_digest: Digest,
        signed_event: SignedAuditEvent,
        appended_at_ns: int | None = None,
    ) -> "ChainRecord":
        """
        Construct a record and calculate its digest.
        """

        _validate_chain_id(
            chain_id
        )

        if not isinstance(
            previous_digest,
            Digest,
        ):
            raise InvalidChainRecordError(
                "previous_digest must be Digest."
            )

        if not isinstance(
            signed_event,
            SignedAuditEvent,
        ):
            raise InvalidChainRecordError(
                "signed_event must be SignedAuditEvent."
            )

        resolved_time = (
            time.time_ns()
            if appended_at_ns is None
            else appended_at_ns
        )

        content = {
            "appended_at_ns": resolved_time,
            "chain_id": chain_id,
            "format": CHAIN_RECORD_FORMAT,
            "format_version": (
                CHAIN_RECORD_VERSION
            ),
            "previous_digest": str(
                previous_digest
            ),
            "sequence_number": (
                sequence_number
            ),
            "signed_event": (
                signed_event.as_dict()
            ),
        }

        digest = hash_json(
            content,
            algorithm=CHAIN_HASH_ALGORITHM,
        )

        return cls(
            chain_id=chain_id,
            sequence_number=sequence_number,
            previous_digest=previous_digest,
            signed_event=signed_event,
            appended_at_ns=resolved_time,
            record_digest=digest,
        )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "ChainRecord":
        """
        Parse one strict chain record.
        """

        if not isinstance(
            value,
            Mapping,
        ):
            raise InvalidChainRecordError(
                "Chain record must be a mapping."
            )

        expected_fields = {
            "appended_at_ns",
            "chain_id",
            "format",
            "format_version",
            "previous_digest",
            "record_digest",
            "sequence_number",
            "signed_event",
        }

        _require_exact_fields(
            value,
            expected_fields,
            object_name="chain record",
        )

        if (
            value["format"]
            != CHAIN_RECORD_FORMAT
        ):
            raise InvalidChainRecordError(
                f"Unsupported chain-record format: "
                f"{value['format']!r}."
            )

        try:
            previous_digest = Digest.parse(
                value["previous_digest"]
            )

            record_digest = Digest.parse(
                value["record_digest"]
            )

        except Exception as exc:
            raise InvalidChainRecordError(
                "Chain record contains "
                "an invalid digest."
            ) from exc

        try:
            signed_event = (
                SignedAuditEvent.from_dict(
                    value["signed_event"]
                )
            )

        except Exception as exc:
            raise InvalidChainRecordError(
                "Chain record contains an invalid "
                "signed audit event."
            ) from exc

        return cls(
            chain_id=value["chain_id"],
            sequence_number=(
                value["sequence_number"]
            ),
            previous_digest=(
                previous_digest
            ),
            signed_event=signed_event,
            appended_at_ns=(
                value["appended_at_ns"]
            ),
            record_digest=record_digest,
            format_version=(
                value["format_version"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ChainHead:
    """
    Portable chain-head checkpoint.

    A production system should periodically protect this value outside the
    primary audit store. For example, it can later be signed and persisted
    in a separately protected checkpoint store.
    """

    chain_id: str

    record_count: int

    sequence_number: int | None

    record_digest: Digest

    generated_at_ns: int

    def __post_init__(
        self,
    ) -> None:

        _validate_chain_id(
            self.chain_id
        )

        if (
            isinstance(
                self.record_count,
                bool,
            )
            or not isinstance(
                self.record_count,
                int,
            )
            or self.record_count < 0
        ):
            raise InvalidIntegrityChainError(
                "record_count must be "
                "a non-negative integer."
            )

        if not isinstance(
            self.record_digest,
            Digest,
        ):
            raise InvalidIntegrityChainError(
                "record_digest must be Digest."
            )

        if (
            self.record_digest.algorithm
            is not CHAIN_HASH_ALGORITHM
        ):
            raise InvalidIntegrityChainError(
                "Chain-head digest must use SHA-256."
            )

        if self.record_count == 0:

            if self.sequence_number is not None:
                raise InvalidIntegrityChainError(
                    "Empty chain head cannot have "
                    "a sequence number."
                )

            if (
                self.record_digest
                != GENESIS_DIGEST
            ):
                raise InvalidIntegrityChainError(
                    "Empty chain head must use "
                    "GENESIS_DIGEST."
                )

        else:

            if (
                isinstance(
                    self.sequence_number,
                    bool,
                )
                or not isinstance(
                    self.sequence_number,
                    int,
                )
                or self.sequence_number
                != self.record_count - 1
            ):
                raise InvalidIntegrityChainError(
                    "Chain-head sequence_number "
                    "does not match record_count."
                )

        if (
            isinstance(
                self.generated_at_ns,
                bool,
            )
            or not isinstance(
                self.generated_at_ns,
                int,
            )
            or self.generated_at_ns <= 0
        ):
            raise InvalidIntegrityChainError(
                "generated_at_ns must be "
                "a positive integer."
            )

    def as_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "chain_id": self.chain_id,
            "generated_at_ns": (
                self.generated_at_ns
            ),
            "record_count": (
                self.record_count
            ),
            "record_digest": str(
                self.record_digest
            ),
            "sequence_number": (
                self.sequence_number
            ),
        }


@dataclass(frozen=True, slots=True)
class ChainVerificationResult:
    """
    Result of complete audit-chain verification.
    """

    status: ChainVerificationStatus

    chain_id: str | None

    records_checked: int

    failure_sequence: int | None

    failure_event_id: str | None

    head_digest: Digest | None

    reason: str

    verified_at_ns: int

    @property
    def valid(
        self,
    ) -> bool:

        return self.status in {
            ChainVerificationStatus.VALID,
            ChainVerificationStatus.EMPTY,
        }

    def require_valid(
        self,
    ) -> None:
        """Raise unless chain verification succeeds."""

        if not self.valid:
            raise IntegrityChainVerificationError(
                self.reason
            )

    def as_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "chain_id": self.chain_id,
            "failure_event_id": (
                self.failure_event_id
            ),
            "failure_sequence": (
                self.failure_sequence
            ),
            "head_digest": (
                None
                if self.head_digest is None
                else str(
                    self.head_digest
                )
            ),
            "reason": self.reason,
            "records_checked": (
                self.records_checked
            ),
            "status": self.status.value,
            "verified_at_ns": (
                self.verified_at_ns
            ),
        }


class IntegrityChain:
    """
    Thread-safe process-local audit chain.

    The RLock protects appends made by threads inside one Python process.

    Production deployments with multiple processes, containers, pods, or
    replicas require a storage backend providing atomic sequence allocation
    and atomic compare-and-append semantics.
    """

    def __init__(
        self,
        *,
        chain_id: str,
        records: Iterable[
            ChainRecord
        ] = (),
    ) -> None:

        _validate_chain_id(
            chain_id
        )

        resolved_records = list(
            records
        )

        if len(resolved_records) > (
            MAX_RECORDS_ON_IMPORT
        ):
            raise InvalidIntegrityChainError(
                "Imported chain contains "
                "too many records."
            )

        event_ids: set[str] = set()

        for record in resolved_records:

            if not isinstance(
                record,
                ChainRecord,
            ):
                raise InvalidIntegrityChainError(
                    "Every chain entry must be "
                    "ChainRecord."
                )

            if record.chain_id != chain_id:
                raise InvalidIntegrityChainError(
                    "Imported record belongs to "
                    "another chain."
                )

            if record.event_id in event_ids:
                raise InvalidIntegrityChainError(
                    "Imported chain contains a "
                    "duplicate event ID."
                )

            event_ids.add(
                record.event_id
            )

        self._chain_id = chain_id

        self._records = (
            resolved_records
        )

        self._event_ids = event_ids

        self._lock = (
            threading.RLock()
        )

    @property
    def chain_id(
        self,
    ) -> str:

        return self._chain_id

    @property
    def records(
        self,
    ) -> tuple[
        ChainRecord,
        ...,
    ]:
        """Return immutable record snapshot."""

        with self._lock:
            return tuple(
                self._records
            )

    @property
    def record_count(
        self,
    ) -> int:

        with self._lock:
            return len(
                self._records
            )

    @property
    def current_digest(
        self,
    ) -> Digest:
        """Return current chain-head digest."""

        with self._lock:

            if not self._records:
                return GENESIS_DIGEST

            return (
                self._records[-1]
                .record_digest
            )

    def head(
        self,
    ) -> ChainHead:
        """
        Create a chain-head checkpoint.
        """

        with self._lock:

            count = len(
                self._records
            )

            if count == 0:

                sequence_number = None

                digest = (
                    GENESIS_DIGEST
                )

            else:

                sequence_number = (
                    count - 1
                )

                digest = (
                    self._records[-1]
                    .record_digest
                )

            return ChainHead(
                chain_id=self._chain_id,
                record_count=count,
                sequence_number=(
                    sequence_number
                ),
                record_digest=digest,
                generated_at_ns=(
                    time.time_ns()
                ),
            )

    def append(
        self,
        signed_event: SignedAuditEvent,
        *,
        appended_at_ns: int | None = None,
    ) -> ChainRecord:
        """
        Atomically append one signed audit event
        within this process.
        """

        if not isinstance(
            signed_event,
            SignedAuditEvent,
        ):
            raise InvalidChainRecordError(
                "signed_event must be "
                "SignedAuditEvent."
            )

        event_id = (
            signed_event
            .event
            .event_id
        )

        with self._lock:

            if event_id in self._event_ids:

                raise DuplicateAuditEventError(
                    f"Audit event {event_id!r} "
                    "already exists in the chain."
                )

            sequence_number = len(
                self._records
            )

            previous_digest = (
                GENESIS_DIGEST
                if not self._records
                else (
                    self._records[-1]
                    .record_digest
                )
            )

            record = ChainRecord.create(
                chain_id=self._chain_id,
                sequence_number=(
                    sequence_number
                ),
                previous_digest=(
                    previous_digest
                ),
                signed_event=signed_event,
                appended_at_ns=(
                    appended_at_ns
                ),
            )

            self._records.append(
                record
            )

            self._event_ids.add(
                event_id
            )

            return record

    def verify(
        self,
        *,
        audit_verifier: (
            AuditVerificationService
            | None
        ) = None,
        expected_head: (
            ChainHead | None
        ) = None,
    ) -> ChainVerificationResult:
        """
        Verify complete chain continuity and integrity.
        """

        with self._lock:

            snapshot = tuple(
                self._records
            )

        return verify_integrity_chain(
            snapshot,
            chain_id=self._chain_id,
            audit_verifier=(
                audit_verifier
            ),
            expected_head=(
                expected_head
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Return complete JSON-compatible chain."""

        with self._lock:

            records = [
                record.as_dict()
                for record in self._records
            ]

        return {
            "chain_id": self._chain_id,
            "format": (
                INTEGRITY_CHAIN_FORMAT
            ),
            "format_version": (
                INTEGRITY_CHAIN_VERSION
            ),
            "records": records,
        }

    def to_json(
        self,
    ) -> str:
        """Serialize complete chain."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "IntegrityChain":
        """
        Parse an integrity chain.
        """

        if not isinstance(
            value,
            Mapping,
        ):
            raise InvalidIntegrityChainError(
                "Integrity chain must be "
                "a mapping."
            )

        expected_fields = {
            "chain_id",
            "format",
            "format_version",
            "records",
        }

        _require_exact_fields(
            value,
            expected_fields,
            object_name=(
                "integrity chain"
            ),
        )

        if (
            value["format"]
            != INTEGRITY_CHAIN_FORMAT
        ):
            raise InvalidIntegrityChainError(
                f"Unsupported chain format: "
                f"{value['format']!r}."
            )

        if (
            value["format_version"]
            != INTEGRITY_CHAIN_VERSION
        ):
            raise InvalidIntegrityChainError(
                f"Unsupported chain version: "
                f"{value['format_version']!r}."
            )

        records_value = (
            value["records"]
        )

        if not isinstance(
            records_value,
            list,
        ):
            raise InvalidIntegrityChainError(
                "records must be a list."
            )

        if len(records_value) > (
            MAX_RECORDS_ON_IMPORT
        ):
            raise InvalidIntegrityChainError(
                "Imported chain contains "
                "too many records."
            )

        records = tuple(
            ChainRecord.from_dict(
                record
            )
            for record in records_value
        )

        return cls(
            chain_id=value["chain_id"],
            records=records,
        )

    @classmethod
    def from_json(
        cls,
        encoded: str | bytes,
    ) -> "IntegrityChain":
        """
        Parse JSON-encoded integrity chain.
        """

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
                raise InvalidIntegrityChainError(
                    "Integrity chain is not "
                    "valid UTF-8."
                ) from exc

        if not isinstance(
            encoded,
            str,
        ):
            raise InvalidIntegrityChainError(
                "Encoded chain must be a "
                "string or UTF-8 bytes."
            )

        if len(
            encoded.encode(
                "utf-8"
            )
        ) > MAX_SERIALIZED_CHAIN_SIZE:
            raise InvalidIntegrityChainError(
                "Serialized chain exceeds "
                "supported size."
            )

        try:
            value = json.loads(
                encoded
            )

        except json.JSONDecodeError as exc:
            raise InvalidIntegrityChainError(
                "Integrity chain contains "
                "invalid JSON."
            ) from exc

        return cls.from_dict(
            value
        )


def verify_integrity_chain(
    records: Iterable[
        ChainRecord
    ],
    *,
    chain_id: str,
    audit_verifier: (
        AuditVerificationService
        | None
    ) = None,
    expected_head: (
        ChainHead | None
    ) = None,
) -> ChainVerificationResult:
    """
    Verify an ordered collection of audit-chain records.

    Checks:

    - object validity
    - chain ID consistency
    - exact sequence continuity
    - unique event IDs
    - genesis link
    - predecessor links
    - chain-record digest integrity
    - optional signed-event verification
    - optional externally retained chain-head comparison
    """

    try:
        _validate_chain_id(
            chain_id
        )

    except Exception:

        return _chain_result(
            status=(
                ChainVerificationStatus
                .MALFORMED
            ),
            chain_id=None,
            records_checked=0,
            reason=(
                "chain_id is malformed."
            ),
        )

    if (
        audit_verifier is not None
        and not isinstance(
            audit_verifier,
            AuditVerificationService,
        )
    ):
        return _chain_result(
            status=(
                ChainVerificationStatus
                .MALFORMED
            ),
            chain_id=chain_id,
            records_checked=0,
            reason=(
                "audit_verifier has an "
                "unsupported type."
            ),
        )

    if (
        expected_head is not None
        and not isinstance(
            expected_head,
            ChainHead,
        )
    ):
        return _chain_result(
            status=(
                ChainVerificationStatus
                .MALFORMED
            ),
            chain_id=chain_id,
            records_checked=0,
            reason=(
                "expected_head must be "
                "ChainHead or None."
            ),
        )

    try:
        resolved = tuple(
            records
        )

    except TypeError:

        return _chain_result(
            status=(
                ChainVerificationStatus
                .MALFORMED
            ),
            chain_id=chain_id,
            records_checked=0,
            reason=(
                "records is not iterable."
            ),
        )

    if len(resolved) > (
        MAX_RECORDS_ON_IMPORT
    ):
        return _chain_result(
            status=(
                ChainVerificationStatus
                .MALFORMED
            ),
            chain_id=chain_id,
            records_checked=0,
            reason=(
                "Chain exceeds maximum "
                "supported record count."
            ),
        )

    if not resolved:

        if expected_head is not None:

            if (
                expected_head.chain_id
                != chain_id
                or expected_head.record_count
                != 0
                or expected_head.sequence_number
                is not None
                or expected_head.record_digest
                != GENESIS_DIGEST
            ):
                return _chain_result(
                    status=(
                        ChainVerificationStatus
                        .INVALID_LINK
                    ),
                    chain_id=chain_id,
                    records_checked=0,
                    reason=(
                        "Empty chain does not "
                        "match expected chain head."
                    ),
                    head_digest=(
                        GENESIS_DIGEST
                    ),
                )

        return _chain_result(
            status=(
                ChainVerificationStatus
                .EMPTY
            ),
            chain_id=chain_id,
            records_checked=0,
            reason=(
                "Integrity chain is empty."
            ),
            head_digest=GENESIS_DIGEST,
        )

    expected_previous = (
        GENESIS_DIGEST
    )

    seen_event_ids: set[str] = (
        set()
    )

    records_checked = 0

    for (
        expected_sequence,
        record,
    ) in enumerate(
        resolved
    ):

        if not isinstance(
            record,
            ChainRecord,
        ):
            return _chain_result(
                status=(
                    ChainVerificationStatus
                    .MALFORMED
                ),
                chain_id=chain_id,
                records_checked=(
                    records_checked
                ),
                failure_sequence=(
                    expected_sequence
                ),
                reason=(
                    "Chain contains a malformed "
                    "record object."
                ),
            )

        if (
            record.chain_id
            != chain_id
        ):
            return _chain_result(
                status=(
                    ChainVerificationStatus
                    .INVALID_LINK
                ),
                chain_id=chain_id,
                records_checked=(
                    records_checked
                ),
                failure_sequence=(
                    expected_sequence
                ),
                failure_event_id=(
                    record.event_id
                ),
                reason=(
                    "Record belongs to "
                    "a different chain."
                ),
            )

        if (
            record.sequence_number
            != expected_sequence
        ):
            return _chain_result(
                status=(
                    ChainVerificationStatus
                    .INVALID_SEQUENCE
                ),
                chain_id=chain_id,
                records_checked=(
                    records_checked
                ),
                failure_sequence=(
                    expected_sequence
                ),
                failure_event_id=(
                    record.event_id
                ),
                reason=(
                    "Chain sequence is "
                    "not continuous."
                ),
            )

        if (
            record.event_id
            in seen_event_ids
        ):
            return _chain_result(
                status=(
                    ChainVerificationStatus
                    .DUPLICATE_EVENT
                ),
                chain_id=chain_id,
                records_checked=(
                    records_checked
                ),
                failure_sequence=(
                    expected_sequence
                ),
                failure_event_id=(
                    record.event_id
                ),
                reason=(
                    "Chain contains a "
                    "duplicate audit event."
                ),
            )

        if (
            record.previous_digest
            != expected_previous
        ):
            return _chain_result(
                status=(
                    ChainVerificationStatus
                    .INVALID_LINK
                ),
                chain_id=chain_id,
                records_checked=(
                    records_checked
                ),
                failure_sequence=(
                    expected_sequence
                ),
                failure_event_id=(
                    record.event_id
                ),
                reason=(
                    "Record does not reference "
                    "the expected predecessor."
                ),
            )

        try:
            calculated = (
                record.calculated_digest()
            )

        except Exception:

            return _chain_result(
                status=(
                    ChainVerificationStatus
                    .ERROR
                ),
                chain_id=chain_id,
                records_checked=(
                    records_checked
                ),
                failure_sequence=(
                    expected_sequence
                ),
                failure_event_id=(
                    record.event_id
                ),
                reason=(
                    "Unable to calculate "
                    "chain-record digest."
                ),
            )

        if (
            calculated
            != record.record_digest
        ):
            return _chain_result(
                status=(
                    ChainVerificationStatus
                    .INVALID_RECORD_DIGEST
                ),
                chain_id=chain_id,
                records_checked=(
                    records_checked
                ),
                failure_sequence=(
                    expected_sequence
                ),
                failure_event_id=(
                    record.event_id
                ),
                reason=(
                    "Chain-record digest does "
                    "not match canonical content."
                ),
            )

        if (
            audit_verifier
            is not None
        ):
            try:
                verification = (
                    audit_verifier
                    .verify(
                        record.signed_event
                    )
                )

            except Exception:

                return _chain_result(
                    status=(
                        ChainVerificationStatus
                        .ERROR
                    ),
                    chain_id=chain_id,
                    records_checked=(
                        records_checked
                    ),
                    failure_sequence=(
                        expected_sequence
                    ),
                    failure_event_id=(
                        record.event_id
                    ),
                    reason=(
                        "Audit signature verifier "
                        "failed unexpectedly."
                    ),
                )

            if not verification.valid:

                return _chain_result(
                    status=(
                        ChainVerificationStatus
                        .INVALID_EVENT_SIGNATURE
                    ),
                    chain_id=chain_id,
                    records_checked=(
                        records_checked
                    ),
                    failure_sequence=(
                        expected_sequence
                    ),
                    failure_event_id=(
                        record.event_id
                    ),
                    reason=(
                        "Embedded audit event has "
                        "an invalid or untrusted "
                        "signature."
                    ),
                )

        seen_event_ids.add(
            record.event_id
        )

        expected_previous = (
            record.record_digest
        )

        records_checked += 1

    final_digest = (
        resolved[-1]
        .record_digest
    )

    if expected_head is not None:

        if (
            expected_head.chain_id
            != chain_id
            or expected_head.record_count
            != len(resolved)
            or expected_head.sequence_number
            != len(resolved) - 1
            or expected_head.record_digest
            != final_digest
        ):
            return _chain_result(
                status=(
                    ChainVerificationStatus
                    .INVALID_LINK
                ),
                chain_id=chain_id,
                records_checked=(
                    records_checked
                ),
                failure_sequence=(
                    len(resolved) - 1
                ),
                failure_event_id=(
                    resolved[-1]
                    .event_id
                ),
                reason=(
                    "Calculated chain state does "
                    "not match the expected "
                    "chain-head checkpoint."
                ),
                head_digest=(
                    final_digest
                ),
            )

    return _chain_result(
        status=(
            ChainVerificationStatus
            .VALID
        ),
        chain_id=chain_id,
        records_checked=(
            records_checked
        ),
        reason=(
            "Audit integrity chain, record "
            "digests, links, sequence, and "
            "configured event signatures "
            "verified successfully."
        ),
        head_digest=final_digest,
    )


def new_integrity_chain(
    *,
    prefix: str = "audit",
) -> IntegrityChain:
    """
    Create a new integrity chain with a
    random chain identifier.
    """

    if (
        not isinstance(prefix, str)
        or not prefix
    ):
        raise InvalidIntegrityChainError(
            "prefix must be non-empty."
        )

    return IntegrityChain(
        chain_id=(
            f"{prefix}:{uuid.uuid4()}"
        )
    )


def _chain_result(
    *,
    status: ChainVerificationStatus,
    chain_id: str | None,
    records_checked: int,
    reason: str,
    failure_sequence: int | None = None,
    failure_event_id: str | None = None,
    head_digest: Digest | None = None,
) -> ChainVerificationResult:

    return ChainVerificationResult(
        status=status,
        chain_id=chain_id,
        records_checked=records_checked,
        failure_sequence=(
            failure_sequence
        ),
        failure_event_id=(
            failure_event_id
        ),
        head_digest=head_digest,
        reason=reason,
        verified_at_ns=(
            time.time_ns()
        ),
    )


def _validate_chain_id(
    value: str,
) -> None:

    if (
        not isinstance(value, str)
        or not value
    ):
        raise InvalidIntegrityChainError(
            "chain_id must be a "
            "non-empty string."
        )

    if len(value) > (
        MAX_CHAIN_ID_LENGTH
    ):
        raise InvalidIntegrityChainError(
            "chain_id exceeds "
            "maximum length."
        )

    if (
        _ID_PATTERN.fullmatch(
            value
        )
        is None
    ):
        raise InvalidIntegrityChainError(
            "chain_id contains "
            "invalid characters."
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

    missing = (
        expected - actual
    )

    unknown = (
        actual - expected
    )

    if missing:

        raise InvalidIntegrityChainError(
            f"{object_name} is missing fields: "
            + ", ".join(
                sorted(missing)
            )
            + "."
        )

    if unknown:

        raise InvalidIntegrityChainError(
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
    "CHAIN_HASH_ALGORITHM",
    "CHAIN_RECORD_FORMAT",
    "CHAIN_RECORD_VERSION",
    "ChainHead",
    "ChainLinkError",
    "ChainRecord",
    "ChainSequenceError",
    "ChainVerificationResult",
    "ChainVerificationStatus",
    "DuplicateAuditEventError",
    "GENESIS_DIGEST",
    "INTEGRITY_CHAIN_FORMAT",
    "INTEGRITY_CHAIN_VERSION",
    "IntegrityChain",
    "IntegrityChainError",
    "IntegrityChainVerificationError",
    "InvalidChainRecordError",
    "InvalidIntegrityChainError",
    "new_integrity_chain",
    "verify_integrity_chain",
]
