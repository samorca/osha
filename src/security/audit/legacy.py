from __future__ import annotations

"""
Bounded, redacted security audit recording for
OSHA.
"""

from collections import deque
from threading import RLock

from typing import (
    Any,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.core.serialization import (
    DEFAULT_SENSITIVE_KEYS,
    to_json_safe,
)

from src.core.validation import (
    integer,
)

from src.security.config import (
    AuditConfig,
)

from src.security.schema import (
    AuditOutcome,
    SecurityAuditRecord,
)


# ---------------------------------------------------------------
# Errors
# ---------------------------------------------------------------


class SecurityAuditError(RuntimeError):
    pass


# ---------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------


def sanitize_audit_attributes(
    attributes: Optional[
        Mapping[
            str,
            Any,
        ]
    ],
) -> Dict[str, Any]:
    """
    Convert audit attributes to JSON-compatible,
    sensitivity-redacted metadata.

    Security audit attributes must never be used
    to store credentials or secret values.
    """

    value = to_json_safe(
        dict(attributes or {}),
        redact_sensitive=True,
        sensitive_keys=(DEFAULT_SENSITIVE_KEYS),
    )

    return (
        value
        if isinstance(
            value,
            dict,
        )
        else {}
    )


# ---------------------------------------------------------------
# Security audit log
# ---------------------------------------------------------------


class SecurityAuditLog:
    """
    Thread-safe bounded in-process security audit
    recorder.

    This provides audit records for later durable
    persistence/export. The in-memory deque is
    not itself a production audit database.
    """

    def __init__(
        self,
        *,
        config: Optional[AuditConfig] = None,
    ) -> None:
        self.config = config or AuditConfig()

        self.config.validate()

        self._records: Deque[SecurityAuditRecord] = deque(
            maxlen=(self.config.maximum_buffered_records)
        )

        self._lock = RLock()

        self._accepted = 0
        self._filtered = 0
        self._evicted = 0

    # ---------------------------------------------------------
    # Outcome filter
    # ---------------------------------------------------------

    def _should_record(
        self,
        outcome: str,
    ) -> bool:
        if not self.config.enabled:
            return False

        if outcome == AuditOutcome.SUCCESS.value:
            return self.config.record_success

        if outcome == AuditOutcome.FAILURE.value:
            return self.config.record_failures

        if outcome == AuditOutcome.BLOCKED.value:
            return self.config.record_policy_blocks

        return True

    # ---------------------------------------------------------
    # Record
    # ---------------------------------------------------------

    def record(
        self,
        record: SecurityAuditRecord,
    ) -> bool:
        record.finalize()

        with self._lock:
            if not self._should_record(record.outcome):
                self._filtered += 1

                return False

            record.attributes = sanitize_audit_attributes(record.attributes)

            was_full = len(self._records) == self._records.maxlen

            self._records.append(record)

            self._accepted += 1

            if was_full:
                self._evicted += 1

            return True

    # ---------------------------------------------------------
    # Generic emit
    # ---------------------------------------------------------

    def emit(
        self,
        *,
        event: str,
        outcome: str,
        principal_id: Optional[str] = None,
        environment_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        reference_ids: Iterable[str] = (),
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> Optional[SecurityAuditRecord]:
        record = SecurityAuditRecord(
            event=event,
            outcome=outcome,
            principal_id=(principal_id),
            environment_id=(environment_id),
            resource=resource,
            action=action,
            reference_ids=list(reference_ids),
            attributes=(sanitize_audit_attributes(attributes)),
        ).finalize()

        if self.record(record):
            return record

        return None

    # ---------------------------------------------------------
    # Authentication audit
    # ---------------------------------------------------------

    def authentication(
        self,
        *,
        success: bool,
        principal_id: Optional[str] = None,
        method: Optional[str] = None,
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> Optional[SecurityAuditRecord]:
        """
        Record authentication outcome.

        Credential material must not be supplied
        in attributes.
        """

        return self.emit(
            event="authentication",
            outcome=(
                AuditOutcome.SUCCESS.value if success else (AuditOutcome.FAILURE.value)
            ),
            principal_id=(principal_id),
            attributes={
                **dict(attributes or {}),
                "method": method,
            },
        )

    # ---------------------------------------------------------
    # Authorization audit
    # ---------------------------------------------------------

    def authorization(
        self,
        *,
        decision: str,
        principal_id: Optional[str],
        environment_id: Optional[str],
        resource: str,
        action: str,
        reference_ids: Iterable[str] = (),
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> Optional[SecurityAuditRecord]:
        """
        Map authorization decisions onto audit
        outcomes.

        allow              -> success
        deny               -> blocked
        require_approval   -> blocked
        other              -> unknown
        """

        normalized = str(decision).strip().lower()

        if normalized == "allow":
            outcome = AuditOutcome.SUCCESS.value

        elif normalized in {
            "deny",
            "require_approval",
        }:
            outcome = AuditOutcome.BLOCKED.value

        else:
            outcome = AuditOutcome.UNKNOWN.value

        return self.emit(
            event="authorization",
            outcome=outcome,
            principal_id=(principal_id),
            environment_id=(environment_id),
            resource=resource,
            action=action,
            reference_ids=(reference_ids),
            attributes={
                **dict(attributes or {}),
                "decision": (normalized),
            },
        )

    # ---------------------------------------------------------
    # Secret access audit
    # ---------------------------------------------------------

    def secret_access(
        self,
        *,
        principal_id: Optional[str],
        provider: str,
        reference_id: Optional[str] = None,
        success: bool,
        attributes: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> Optional[SecurityAuditRecord]:
        """
        Record secret-reference access.

        Deliberately does NOT accept:

          secret value
          secret key
          secret name
          credential
          token

        Only the provider and optional opaque
        reference ID are recorded here.
        """

        references = [reference_id] if reference_id else []

        return self.emit(
            event="secret_access",
            outcome=(
                AuditOutcome.SUCCESS.value if success else (AuditOutcome.FAILURE.value)
            ),
            principal_id=(principal_id),
            resource="secret",
            action="resolve",
            reference_ids=(references),
            attributes={
                **dict(attributes or {}),
                "provider": (provider),
            },
        )

    # ---------------------------------------------------------
    # Query
    # ---------------------------------------------------------

    def list(
        self,
        *,
        outcome: Optional[str] = None,
        principal_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[SecurityAuditRecord]:
        if limit is not None:
            limit = integer(
                limit,
                name="limit",
                minimum=1,
            )

        if outcome is not None and outcome not in {item.value for item in AuditOutcome}:
            raise SecurityAuditError(f"Unknown audit outcome: {outcome}")

        with self._lock:
            values = list(self._records)

        if newest_first:
            values.reverse()

        output: List[SecurityAuditRecord] = []

        for record in values:
            if outcome is not None and record.outcome != outcome:
                continue

            if principal_id is not None and record.principal_id != principal_id:
                continue

            if resource is not None and record.resource != resource:
                continue

            if action is not None and record.action != action:
                continue

            output.append(record)

            if limit is not None and len(output) >= limit:
                break

        return output

    # ---------------------------------------------------------
    # Drain
    # ---------------------------------------------------------

    def drain(
        self,
        *,
        limit: Optional[int] = None,
    ) -> List[SecurityAuditRecord]:
        if limit is not None:
            limit = integer(
                limit,
                name="limit",
                minimum=1,
            )

        with self._lock:
            count = (
                len(self._records)
                if limit is None
                else min(
                    limit,
                    len(self._records),
                )
            )

            return [self._records.popleft() for _ in range(count)]

    # ---------------------------------------------------------
    # Clear
    # ---------------------------------------------------------

    def clear(
        self,
    ) -> int:
        with self._lock:
            count = len(self._records)

            self._records.clear()

            return count

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            outcomes: Dict[
                str,
                int,
            ] = {}

            events: Dict[
                str,
                int,
            ] = {}

            for record in self._records:
                outcomes[record.outcome] = (
                    outcomes.get(
                        record.outcome,
                        0,
                    )
                    + 1
                )

                events[record.event] = (
                    events.get(
                        record.event,
                        0,
                    )
                    + 1
                )

            return {
                "enabled": (self.config.enabled),
                "buffered_records": (len(self._records)),
                "maximum_buffered_records": (self.config.maximum_buffered_records),
                "accepted_total": (self._accepted),
                "filtered_total": (self._filtered),
                "evicted_total": (self._evicted),
                "buffered_by_outcome": (outcomes),
                "buffered_by_event": (events),
                "secret_values_recorded": (False),
            }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "SecurityAuditError",
    "sanitize_audit_attributes",
    "SecurityAuditLog",
]
