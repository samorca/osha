from __future__ import annotations

import hashlib
import json
import re
import threading
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import (
    asdict,
    dataclass,
    field,
)
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.remediation.schema import (
    AuditEvent,
    ExecutionResult,
    RemediationRequest,
    RemediationResult,
    RollbackResult,
    ValidationResult,
    VerificationResult,
    json_safe,
    utc_now,
)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class AuditConfig:
    """
    OSHA remediation audit configuration.

    Audit logging must remain metadata-oriented.
    Raw unrestricted adapter output should not be
    copied into the audit store.
    """

    maximum_text_chars: int = 2048

    maximum_attribute_depth: int = 8

    include_parameters: bool = False

    include_execution_output: bool = False

    hash_chain_enabled: bool = True

    fsync_jsonl: bool = False

    def validate(
        self,
    ) -> None:
        if self.maximum_text_chars < 128:
            raise ValueError("maximum_text_chars must be >= 128")

        if self.maximum_attribute_depth < 1:
            raise ValueError("maximum_attribute_depth must be >= 1")


# ---------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------

SENSITIVE_KEYWORDS = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "authorization",
    "credential",
    "credentials",
}


REDACTION_PATTERNS = (
    re.compile(
        r"(?i)"
        r"(password|passwd|secret|token|"
        r"api[_-]?key|access[_-]?key|"
        r"private[_-]?key)"
        r"\s*[:=]\s*"
        r"([^\s,;]+)"
    ),
    re.compile(
        r"(?i)"
        r"(authorization\s*:\s*bearer)"
        r"\s+([^\s]+)"
    ),
)


def sensitive_key(
    value: Any,
) -> bool:
    key = str(value).strip().lower()

    normalized = key.replace(
        "-",
        "_",
    )

    return any(keyword in normalized for keyword in SENSITIVE_KEYWORDS)


def redact_text(
    value: Any,
    *,
    maximum_chars: int,
) -> str:
    text = str(value)

    for pattern in REDACTION_PATTERNS:
        text = pattern.sub(
            lambda match: f"{match.group(1)}=[REDACTED]",
            text,
        )

    if len(text) <= maximum_chars:
        return text

    suffix = "[TRUNCATED]"

    limit = max(
        0,
        maximum_chars - len(suffix),
    )

    return text[:limit] + suffix


def sanitize_value(
    value: Any,
    *,
    config: AuditConfig,
    depth: int = 0,
) -> Any:
    """
    Recursively sanitize audit attributes.

    Secret-looking keys are removed regardless
    of their value.
    """

    if depth > config.maximum_attribute_depth:
        return "[MAX_DEPTH_REACHED]"

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
        return json_safe(value)

    if isinstance(
        value,
        str,
    ):
        return redact_text(
            value,
            maximum_chars=(config.maximum_text_chars),
        )

    if isinstance(
        value,
        Mapping,
    ):
        result = {}

        for key, item in value.items():
            if sensitive_key(key):
                result[str(key)] = "[REDACTED]"

                continue

            result[str(key)] = sanitize_value(
                item,
                config=config,
                depth=depth + 1,
            )

        return result

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        return [
            sanitize_value(
                item,
                config=config,
                depth=depth + 1,
            )
            for item in value
        ]

    return redact_text(
        value,
        maximum_chars=(config.maximum_text_chars),
    )


# ---------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------


@dataclass(frozen=True)
class AuditRecord:
    """
    Stored audit record.

    event:
        Sanitized logical AuditEvent.

    previous_hash:
        Hash of the previous record when hash
        chaining is enabled.

    event_hash:
        Digest over the current sanitized event
        and previous_hash.

    This provides tamper evidence only. It does
    not make mutable storage tamper-proof.
    """

    event_id: str

    event: AuditEvent

    previous_hash: Optional[str] = None

    event_hash: Optional[str] = None

    recorded_at: str = field(default_factory=utc_now)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "event_id": (self.event_id),
            "event": (self.event.to_dict()),
            "previous_hash": (self.previous_hash),
            "event_hash": (self.event_hash),
            "recorded_at": (self.recorded_at),
        }


def audit_digest(
    event: AuditEvent,
    previous_hash: Optional[str],
) -> str:
    payload = {
        "event": (event.to_dict()),
        "previous_hash": (previous_hash),
    }

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def event_identifier(
    event: AuditEvent,
    previous_hash: Optional[str],
) -> str:
    digest = audit_digest(
        event,
        previous_hash,
    )

    return "audit:" + digest


# ---------------------------------------------------------------
# Audit store
# ---------------------------------------------------------------


class AuditStore(ABC):
    """
    Append-only logical audit interface.
    """

    @abstractmethod
    def append(
        self,
        event: AuditEvent,
    ) -> AuditRecord:
        pass

    @abstractmethod
    def records(
        self,
    ) -> List[AuditRecord]:
        pass

    def last_hash(
        self,
    ) -> Optional[str]:
        values = self.records()

        if not values:
            return None

        return values[-1].event_hash


class InMemoryAuditStore(AuditStore):
    """
    Development audit store.

    Append-only through this interface.
    """

    def __init__(
        self,
        *,
        config: Optional[AuditConfig] = None,
    ) -> None:
        self.config = config or AuditConfig()

        self.config.validate()

        self._records: List[AuditRecord] = []

        self._lock = threading.RLock()

    def append(
        self,
        event: AuditEvent,
    ) -> AuditRecord:
        with self._lock:
            previous_hash = self.last_hash() if self.config.hash_chain_enabled else None

            sanitized = sanitize_event(
                event,
                config=self.config,
            )

            digest = (
                audit_digest(
                    sanitized,
                    previous_hash,
                )
                if self.config.hash_chain_enabled
                else None
            )

            event_id = "audit:" + (
                digest
                or audit_digest(
                    sanitized,
                    None,
                )
            )

            record = AuditRecord(
                event_id=event_id,
                event=sanitized,
                previous_hash=(previous_hash),
                event_hash=digest,
            )

            self._records.append(record)

            return record

    def records(
        self,
    ) -> List[AuditRecord]:
        with self._lock:
            return list(self._records)


class JsonlAuditStore(AuditStore):
    """
    JSONL append-oriented audit store.

    This improves operational durability relative
    to an in-memory store.

    File append semantics alone do not make the
    file immutable or tamper-proof.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        config: Optional[AuditConfig] = None,
    ) -> None:
        self.path = Path(path)

        self.config = config or AuditConfig()

        self.config.validate()

        self._lock = threading.RLock()

    def records(
        self,
    ) -> List[AuditRecord]:
        if not self.path.exists():
            return []

        output = []

        with self._lock:
            with self.path.open(
                "r",
                encoding="utf-8",
            ) as source:
                for line in source:
                    line = line.strip()

                    if not line:
                        continue

                    data = json.loads(line)

                    event_data = data["event"]

                    event = AuditEvent(
                        event_type=(event_data["event_type"]),
                        description=(event_data["description"]),
                        remediation_id=(event_data.get("remediation_id")),
                        request_id=(event_data.get("request_id")),
                        action_id=(event_data.get("action_id")),
                        actor_id=(event_data.get("actor_id")),
                        occurred_at=(event_data["occurred_at"]),
                        attributes=dict(
                            event_data.get(
                                "attributes",
                                {},
                            )
                        ),
                    )

                    output.append(
                        AuditRecord(
                            event_id=(data["event_id"]),
                            event=event,
                            previous_hash=(data.get("previous_hash")),
                            event_hash=(data.get("event_hash")),
                            recorded_at=(data["recorded_at"]),
                        )
                    )

        return output

    def append(
        self,
        event: AuditEvent,
    ) -> AuditRecord:
        import os

        with self._lock:
            values = self.records()

            previous_hash = (
                values[-1].event_hash
                if (values and self.config.hash_chain_enabled)
                else None
            )

            sanitized = sanitize_event(
                event,
                config=self.config,
            )

            digest = (
                audit_digest(
                    sanitized,
                    previous_hash,
                )
                if self.config.hash_chain_enabled
                else None
            )

            event_id = "audit:" + (
                digest
                or audit_digest(
                    sanitized,
                    None,
                )
            )

            record = AuditRecord(
                event_id=event_id,
                event=sanitized,
                previous_hash=(previous_hash),
                event_hash=digest,
            )

            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with self.path.open(
                "a",
                encoding="utf-8",
            ) as destination:
                destination.write(
                    json.dumps(
                        record.to_dict(),
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":",
                        ),
                        default=str,
                    )
                )

                destination.write("\n")

                destination.flush()

                if self.config.fsync_jsonl:
                    os.fsync(destination.fileno())

            return record


# ---------------------------------------------------------------
# Event sanitization
# ---------------------------------------------------------------


def sanitize_event(
    event: AuditEvent,
    *,
    config: AuditConfig,
) -> AuditEvent:
    return AuditEvent(
        event_type=(
            redact_text(
                event.event_type,
                maximum_chars=(config.maximum_text_chars),
            )
        ),
        description=(
            redact_text(
                event.description,
                maximum_chars=(config.maximum_text_chars),
            )
        ),
        remediation_id=(event.remediation_id),
        request_id=(event.request_id),
        action_id=(event.action_id),
        actor_id=(
            redact_text(
                event.actor_id,
                maximum_chars=(config.maximum_text_chars),
            )
            if event.actor_id
            else None
        ),
        occurred_at=(event.occurred_at),
        attributes=(
            sanitize_value(
                event.attributes,
                config=config,
            )
        ),
    )


# ---------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------


def request_event(
    request: RemediationRequest,
    *,
    actor_id: Optional[str] = None,
    remediation_id: Optional[str] = None,
    config: Optional[AuditConfig] = None,
) -> AuditEvent:
    configuration = config or AuditConfig()

    attributes: Dict[
        str,
        Any,
    ] = {
        "action_type": (request.action_type),
        "environment_id": (request.environment_id),
        "mode": request.mode,
        "target_resource_ids": list(request.target_resource_ids),
        "target_service_ids": list(request.target_service_ids),
        "risk_level": (request.risk_level),
        "risk_score": (request.risk_score),
        "approval_status": (request.approval_status),
        "approval_count": len(request.approval_ids),
        "decision_id": (request.decision_id),
        "plan_id": (request.plan_id),
        "incident_id": (request.incident_id),
    }

    if configuration.include_parameters:
        attributes["parameters"] = dict(request.parameters)

    return AuditEvent(
        event_type=("remediation.requested"),
        description=("Remediation request entered the controlled lifecycle."),
        remediation_id=(remediation_id),
        request_id=(request.request_id),
        action_id=(request.action_id),
        actor_id=(actor_id or request.requested_by),
        attributes=attributes,
    )


def validation_event(
    request: RemediationRequest,
    validation: ValidationResult,
    *,
    remediation_id: Optional[str] = None,
) -> AuditEvent:
    return AuditEvent(
        event_type=("remediation.validation"),
        description=(f"Remediation request validation status: {validation.status}."),
        remediation_id=(remediation_id),
        request_id=(request.request_id),
        action_id=(request.action_id),
        attributes={
            "status": (validation.status),
            "valid": (validation.valid),
            "blocking_reasons": (list(validation.blocking_reasons)),
            "warnings": list(validation.warnings),
            "check_count": len(validation.checks),
        },
    )


def execution_event(
    request: RemediationRequest,
    execution: ExecutionResult,
    *,
    remediation_id: Optional[str] = None,
    config: Optional[AuditConfig] = None,
) -> AuditEvent:
    configuration = config or AuditConfig()

    attributes: Dict[
        str,
        Any,
    ] = {
        "status": execution.status,
        "success": execution.success,
        "adapter_name": (execution.adapter_name),
        "operation": (execution.operation),
        "changed": (execution.changed),
        "duration_seconds": (execution.duration_seconds),
        "exit_code": (execution.exit_code),
    }

    if configuration.include_execution_output:
        attributes["stdout"] = execution.stdout

        attributes["stderr"] = execution.stderr

    return AuditEvent(
        event_type=("remediation.execution"),
        description=(f"Remediation execution status: {execution.status}."),
        remediation_id=(remediation_id),
        request_id=(request.request_id),
        action_id=(request.action_id),
        attributes=attributes,
    )


def verification_event(
    request: RemediationRequest,
    verification: VerificationResult,
    *,
    remediation_id: Optional[str] = None,
) -> AuditEvent:
    return AuditEvent(
        event_type=("remediation.verification"),
        description=(f"Post-action verification status: {verification.status}."),
        remediation_id=(remediation_id),
        request_id=(request.request_id),
        action_id=(request.action_id),
        attributes={
            "status": (verification.status),
            "passed": (verification.passed),
            "health_before": (verification.health_before),
            "health_after": (verification.health_after),
            "checks": list(verification.checks),
            "failed_checks": list(verification.failed_checks),
        },
    )


def rollback_event(
    request: RemediationRequest,
    rollback: RollbackResult,
    *,
    remediation_id: Optional[str] = None,
) -> AuditEvent:
    return AuditEvent(
        event_type=("remediation.rollback"),
        description=(f"Rollback lifecycle status: {rollback.status}."),
        remediation_id=(remediation_id),
        request_id=(request.request_id),
        action_id=(request.action_id),
        attributes={
            "status": (rollback.status),
            "success": (rollback.success),
            "action_type": (rollback.action_type),
            "adapter_name": (rollback.adapter_name),
            "reason": rollback.reason,
        },
    )


def completion_event(
    result: RemediationResult,
) -> AuditEvent:
    result.finalize()

    return AuditEvent(
        event_type=("remediation.completed"),
        description=(f"Remediation lifecycle reached status: {result.status}."),
        remediation_id=(result.remediation_id),
        request_id=(result.request.request_id),
        action_id=(result.request.action_id),
        attributes={
            "status": result.status,
            "validation_status": (result.validation.status),
            "execution_status": (result.execution.status),
            "verification_status": (result.verification.status),
            "rollback_status": (result.rollback.status),
            "error_present": bool(result.error),
        },
    )


# ---------------------------------------------------------------
# Audit manager
# ---------------------------------------------------------------


class RemediationAudit:
    """
    Lifecycle audit coordinator.

    This component records events only.
    It has no execution capability.
    """

    def __init__(
        self,
        *,
        store: Optional[AuditStore] = None,
        config: Optional[AuditConfig] = None,
    ) -> None:
        self.config = config or AuditConfig()

        self.config.validate()

        self.store = store or InMemoryAuditStore(config=self.config)

    def record(
        self,
        event: AuditEvent,
    ) -> AuditRecord:
        return self.store.append(event)

    def record_request(
        self,
        request: RemediationRequest,
        *,
        actor_id: Optional[str] = None,
        remediation_id: Optional[str] = None,
    ) -> AuditRecord:
        return self.record(
            request_event(
                request,
                actor_id=actor_id,
                remediation_id=(remediation_id),
                config=self.config,
            )
        )

    def record_validation(
        self,
        request: RemediationRequest,
        validation: ValidationResult,
        *,
        remediation_id: Optional[str] = None,
    ) -> AuditRecord:
        return self.record(
            validation_event(
                request,
                validation,
                remediation_id=(remediation_id),
            )
        )

    def record_execution(
        self,
        request: RemediationRequest,
        execution: ExecutionResult,
        *,
        remediation_id: Optional[str] = None,
    ) -> AuditRecord:
        return self.record(
            execution_event(
                request,
                execution,
                remediation_id=(remediation_id),
                config=self.config,
            )
        )

    def record_verification(
        self,
        request: RemediationRequest,
        verification: VerificationResult,
        *,
        remediation_id: Optional[str] = None,
    ) -> AuditRecord:
        return self.record(
            verification_event(
                request,
                verification,
                remediation_id=(remediation_id),
            )
        )

    def record_rollback(
        self,
        request: RemediationRequest,
        rollback: RollbackResult,
        *,
        remediation_id: Optional[str] = None,
    ) -> AuditRecord:
        return self.record(
            rollback_event(
                request,
                rollback,
                remediation_id=(remediation_id),
            )
        )

    def record_completion(
        self,
        result: RemediationResult,
    ) -> AuditRecord:
        return self.record(completion_event(result))

    def records(
        self,
    ) -> List[AuditRecord]:
        return self.store.records()


# ---------------------------------------------------------------
# Hash-chain verification
# ---------------------------------------------------------------


def verify_audit_chain(
    records: Iterable[AuditRecord],
) -> tuple[
    bool,
    List[str],
]:
    """
    Validate this module's logical hash chain.

    This can detect modifications to chained
    record content when hashes are retained.

    It does not establish storage immutability or
    authenticity against an attacker capable of
    rewriting the full chain.
    """

    values = list(records)

    errors = []

    previous_hash = None

    for index, record in enumerate(values):
        if record.previous_hash != previous_hash:
            errors.append(f"Audit record {index} has an invalid previous_hash.")

        expected_hash = audit_digest(
            record.event,
            previous_hash,
        )

        if record.event_hash != expected_hash:
            errors.append(f"Audit record {index} has an invalid event_hash.")

        expected_id = "audit:" + expected_hash

        if record.event_id != expected_id:
            errors.append(f"Audit record {index} has an invalid event_id.")

        previous_hash = record.event_hash

    return (
        not errors,
        errors,
    )


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------


def audit_summary(
    records: Iterable[AuditRecord],
) -> Dict[str, Any]:
    values = list(records)

    event_types: Dict[
        str,
        int,
    ] = {}

    for record in values:
        event_type = record.event.event_type

        event_types[event_type] = (
            event_types.get(
                event_type,
                0,
            )
            + 1
        )

    chain_valid, errors = (
        verify_audit_chain(values)
        if values and all(record.event_hash is not None for record in values)
        else (
            False,
            [("Hash chain is disabled or unavailable.")],
        )
    )

    return {
        "record_count": len(values),
        "event_types": (event_types),
        "hash_chain_valid": (chain_valid),
        "hash_chain_errors": (errors),
    }
