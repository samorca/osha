from __future__ import annotations

import uuid

from pathlib import Path
from typing import Any, Callable, Mapping

from src.security.audit.integrity_chain import IntegrityChain
from src.security.audit.signed_events import (
    AuditEventType,
    AuditOutcome,
    AuditSeverity,
    AuditSigningService,
    SignedAuditEvent,
    new_audit_event,
)
from src.security.crypto.envelopes import (
    EnvelopePurpose,
    EnvelopeService,
    InMemoryReplayCache,
    ReplayCache,
    SecureEnvelope,
    open_once,
)
from src.security.crypto.key_provider import KeyReference
from src.security.identity.workload_identity import AuthenticatedWorkload
from src.security.policy.authorisation import RiskLevel
from src.security.policy.remediation_policy import (
    InfrastructurePlatform,
    RemediationAction as SecurityRemediationAction,
    RemediationDecision,
    RemediationPolicyEngine,
    RemediationRequest as SecurityRemediationRequest,
)
from src.security.provenance.dataset_signing import (
    DatasetVerificationService,
    SignedDatasetManifest,
)
from src.security.provenance.model_signing import (
    ModelVerificationService,
    SignedModelManifest,
)


SIGNED_MANIFEST_SUFFIX = ".osha-signed.json"


class SecurityBoundaryError(RuntimeError):
    pass


class RemediationAuthorizationGateError(SecurityBoundaryError):
    pass


def signed_manifest_path(artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    return path.with_name(path.name + SIGNED_MANIFEST_SUFFIX)


def read_signed_dataset_manifest(path: str | Path) -> SignedDatasetManifest:
    manifest_path = signed_manifest_path(path)
    try:
        encoded = manifest_path.read_bytes()
    except OSError as exc:
        raise SecurityBoundaryError(
            "Signed dataset manifest could not be read."
        ) from exc
    return SignedDatasetManifest.from_json(encoded)


def write_signed_dataset_manifest(
    artifact_path: str | Path,
    manifest: SignedDatasetManifest,
) -> Path:
    if not isinstance(manifest, SignedDatasetManifest):
        raise TypeError("manifest must be SignedDatasetManifest.")
    destination = signed_manifest_path(artifact_path)
    temporary = destination.with_name(destination.name + ".writing")
    try:
        temporary.write_text(manifest.to_json() + "\n", encoding="utf-8")
        temporary.replace(destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise SecurityBoundaryError(
            "Signed dataset manifest could not be written."
        ) from exc
    return destination


def read_signed_model_manifest(path: str | Path) -> SignedModelManifest:
    manifest_path = signed_manifest_path(path)
    try:
        encoded = manifest_path.read_bytes()
    except OSError as exc:
        raise SecurityBoundaryError(
            "Signed model manifest could not be read."
        ) from exc
    return SignedModelManifest.from_json(encoded)


def write_signed_model_manifest(
    artifact_path: str | Path,
    manifest: SignedModelManifest,
) -> Path:
    if not isinstance(manifest, SignedModelManifest):
        raise TypeError("manifest must be SignedModelManifest.")
    destination = signed_manifest_path(artifact_path)
    temporary = destination.with_name(destination.name + ".writing")
    try:
        temporary.write_text(manifest.to_json() + "\n", encoding="utf-8")
        temporary.replace(destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise SecurityBoundaryError(
            "Signed model manifest could not be written."
        ) from exc
    return destination


class SecurityAuditRecorder:
    """Sign audit events and append them to a tamper-evident chain."""

    def __init__(
        self,
        *,
        signer: AuditSigningService,
        chain: IntegrityChain,
        actor: AuthenticatedWorkload,
    ) -> None:
        if not isinstance(signer, AuditSigningService):
            raise TypeError("signer must be AuditSigningService.")
        if not isinstance(chain, IntegrityChain):
            raise TypeError("chain must be IntegrityChain.")
        if not isinstance(actor, AuthenticatedWorkload):
            raise TypeError("actor must be AuthenticatedWorkload.")
        actor.require_verified()
        self._signer = signer
        self._chain = chain
        self._actor = actor

    def record(
        self,
        *,
        event_type: AuditEventType,
        action: str,
        outcome: AuditOutcome,
        severity: AuditSeverity,
        target: str,
        correlation_id: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> SignedAuditEvent:
        identity = self._actor.require_verified()
        event = new_audit_event(
            event_type=event_type,
            actor_workload_id=identity.workload_id,
            action=action,
            outcome=outcome,
            severity=severity,
            target=target,
            correlation_id=correlation_id,
            message=message,
            details=details,
        )
        signed = self._signer.sign(event)
        self._chain.append(signed)
        return signed


def require_verified_dataset(
    path: str | Path,
    verifier: DatasetVerificationService,
    *,
    audit: SecurityAuditRecorder | None = None,
    correlation_id: str | None = None,
) -> Any:
    if not isinstance(verifier, DatasetVerificationService):
        raise TypeError("verifier must be DatasetVerificationService.")
    if not isinstance(audit, SecurityAuditRecorder):
        raise ValueError(
            "Dataset verification requires a signed security audit recorder."
        )
    source = Path(path)
    correlation = correlation_id or str(uuid.uuid4())
    try:
        manifest = read_signed_dataset_manifest(source)
        result = verifier.require_verified(source, manifest)
    except Exception as exc:
        if audit is not None:
            audit.record(
                event_type=AuditEventType.DATASET_VERIFICATION,
                action="verify_dataset",
                outcome=AuditOutcome.REJECTED,
                severity=AuditSeverity.ERROR,
                target=source.name,
                correlation_id=correlation,
                message="Dataset verification was rejected.",
                details={"error_type": type(exc).__name__},
            )
        raise
    if audit is not None:
        audit.record(
            event_type=AuditEventType.DATASET_VERIFICATION,
            action="verify_dataset",
            outcome=AuditOutcome.ACCEPTED,
            severity=AuditSeverity.NOTICE,
            target=source.name,
            correlation_id=correlation,
            message="Dataset signature, provenance, and digest verified.",
            details={"manifest_id": result.manifest_id},
        )
    return result


def require_verified_model(
    path: str | Path,
    verifier: ModelVerificationService,
    *,
    audit: SecurityAuditRecorder | None = None,
    correlation_id: str | None = None,
) -> Any:
    if not isinstance(verifier, ModelVerificationService):
        raise TypeError("verifier must be ModelVerificationService.")
    if not isinstance(audit, SecurityAuditRecorder):
        raise ValueError(
            "Model verification requires a signed security audit recorder."
        )
    source = Path(path)
    correlation = correlation_id or str(uuid.uuid4())
    try:
        manifest = read_signed_model_manifest(source)
        result = verifier.require_verified(source, manifest)
    except Exception as exc:
        if audit is not None:
            audit.record(
                event_type=AuditEventType.MODEL_VERIFICATION,
                action="verify_model",
                outcome=AuditOutcome.REJECTED,
                severity=AuditSeverity.ERROR,
                target=source.name,
                correlation_id=correlation,
                message="Model verification was rejected.",
                details={"error_type": type(exc).__name__},
            )
        raise
    if audit is not None:
        audit.record(
            event_type=AuditEventType.MODEL_VERIFICATION,
            action="verify_model",
            outcome=AuditOutcome.ACCEPTED,
            severity=AuditSeverity.NOTICE,
            target=source.name,
            correlation_id=correlation,
            message="Model signature, provenance, and digest verified.",
            details={"manifest_id": result.manifest_id},
        )
    return result


class SecureTelemetryTransport:
    """Seal normalized telemetry and authenticate it before correlation."""

    def __init__(
        self,
        *,
        envelopes: EnvelopeService,
        workload: AuthenticatedWorkload,
        destination: str,
        expected_source: str | None = None,
        expected_source_signing_key: KeyReference | None = None,
        encryption_key: KeyReference | None = None,
        replay_cache: ReplayCache | None = None,
        audit: SecurityAuditRecorder | None = None,
    ) -> None:
        if not isinstance(envelopes, EnvelopeService):
            raise TypeError("envelopes must be EnvelopeService.")
        if not isinstance(workload, AuthenticatedWorkload):
            raise TypeError("workload must be AuthenticatedWorkload.")
        if not isinstance(audit, SecurityAuditRecorder):
            raise ValueError(
                "Secure telemetry requires a signed security audit recorder."
            )
        identity = workload.require_verified()
        if expected_source_signing_key is not None and not isinstance(
            expected_source_signing_key,
            KeyReference,
        ):
            raise TypeError(
                "expected_source_signing_key must be KeyReference or None."
            )
        if (expected_source is None) != (expected_source_signing_key is None):
            raise ValueError(
                "Expected source identity and its trusted signing key must be "
                "configured together."
            )
        key = encryption_key or identity.encryption_key
        if not isinstance(key, KeyReference):
            raise SecurityBoundaryError(
                "Authenticated collector identity has no encryption key."
            )
        self._envelopes = envelopes
        self._workload = workload
        self._destination = destination
        self._expected_source = expected_source
        self._expected_source_signing_key = expected_source_signing_key
        self._encryption_key = key
        self._replay_cache = replay_cache or InMemoryReplayCache()
        self._audit = audit

    def seal(
        self,
        telemetry: Any,
        *,
        correlation_id: str | None = None,
    ) -> SecureEnvelope:
        identity = self._workload.require_verified()
        if (
            identity.signing_key is None
            or self._envelopes.signing_key_reference != identity.signing_key
        ):
            raise SecurityBoundaryError(
                "Envelope signing key is not bound to the authenticated workload."
            )
        payload = telemetry.to_dict() if hasattr(telemetry, "to_dict") else telemetry
        if not isinstance(payload, Mapping):
            raise SecurityBoundaryError("Normalized telemetry must be a mapping.")
        correlation = correlation_id or str(uuid.uuid4())
        envelope = self._envelopes.seal_json(
            dict(payload),
            encryption_key=self._encryption_key,
            source=identity.workload_id,
            destination=self._destination,
            purpose=EnvelopePurpose.TELEMETRY,
            correlation_id=correlation,
            sign=True,
        )
        if self._audit is not None:
            self._audit.record(
                event_type=AuditEventType.SYSTEM,
                action="seal_telemetry",
                outcome=AuditOutcome.SUCCESS,
                severity=AuditSeverity.INFO,
                target=self._destination,
                correlation_id=correlation,
                message="Normalized telemetry was sealed for transport.",
                details={"envelope_id": envelope.metadata.envelope_id},
            )
        return envelope

    def open(self, value: Mapping[str, Any]) -> dict[str, Any]:
        receiver = self._workload.require_verified()
        correlation = "unknown"
        try:
            if (
                self._expected_source is None
                or self._expected_source_signing_key is None
            ):
                raise SecurityBoundaryError(
                    "Expected collector identity and signing key are not configured."
                )
            if (
                self._envelopes.verification_key_reference
                != self._expected_source_signing_key
            ):
                raise SecurityBoundaryError(
                    "Envelope verification key is not trusted for the expected collector."
                )
            if not isinstance(value, Mapping):
                raise SecurityBoundaryError(
                    "Protected telemetry envelope must be a mapping."
                )
            metadata = value.get("metadata")
            if isinstance(metadata, Mapping):
                correlation = str(metadata.get("correlation_id", "unknown"))
            envelope = SecureEnvelope.from_dict(value)
            opened = open_once(
                self._envelopes,
                envelope,
                self._replay_cache,
                require_signature=True,
                expected_source=self._expected_source,
                expected_destination=receiver.workload_id,
                expected_purpose=EnvelopePurpose.TELEMETRY,
            )
            payload = opened.json()
            if not isinstance(payload, Mapping):
                raise SecurityBoundaryError(
                    "Protected telemetry payload must be a JSON object."
                )
        except Exception as exc:
            if self._audit is not None:
                self._audit.record(
                    event_type=AuditEventType.SYSTEM,
                    action="open_telemetry",
                    outcome=AuditOutcome.REJECTED,
                    severity=AuditSeverity.ERROR,
                    target=receiver.workload_id,
                    correlation_id=correlation,
                    message="Protected telemetry was rejected.",
                    details={"error_type": type(exc).__name__},
                )
            raise SecurityBoundaryError(
                "Protected telemetry failed authentication or validation."
            ) from exc
        if self._audit is not None:
            self._audit.record(
                event_type=AuditEventType.SYSTEM,
                action="open_telemetry",
                outcome=AuditOutcome.ACCEPTED,
                severity=AuditSeverity.NOTICE,
                target=receiver.workload_id,
                correlation_id=opened.metadata.correlation_id,
                message="Protected telemetry was authenticated and opened.",
                details={"envelope_id": opened.metadata.envelope_id},
            )
        return dict(payload)


class RemediationAuthorizationGate:
    """Require authenticated workload authorization and safety policy before execution."""

    def __init__(
        self,
        *,
        engine: RemediationPolicyEngine,
        workload: AuthenticatedWorkload,
        risk_resolver: Callable[[Any], RiskLevel] | None = None,
        platform_resolver: Callable[[Any], InfrastructurePlatform] | None = None,
        protected_resource_resolver: Callable[[Any], bool] | None = None,
        rollback_checker: Callable[[Any], bool] | None = None,
        approval_verifier: Callable[[Any], bool] | None = None,
        audit: SecurityAuditRecorder | None = None,
    ) -> None:
        if not isinstance(engine, RemediationPolicyEngine):
            raise TypeError("engine must be RemediationPolicyEngine.")
        if not isinstance(workload, AuthenticatedWorkload):
            raise TypeError("workload must be AuthenticatedWorkload.")
        if not isinstance(audit, SecurityAuditRecorder):
            raise ValueError(
                "Remediation authorization requires a signed security audit recorder."
            )
        workload.require_verified()
        self._engine = engine
        self._workload = workload
        self._risk_resolver = risk_resolver
        self._platform_resolver = platform_resolver
        self._protected_resource_resolver = protected_resource_resolver
        self._rollback_checker = rollback_checker
        self._approval_verifier = approval_verifier
        self._audit = audit

    def __call__(self, request: Any) -> Any:
        try:
            return self._authorize(request)
        except RemediationAuthorizationGateError:
            raise
        except Exception as exc:
            self._audit.record(
                event_type=AuditEventType.REMEDIATION_DECISION,
                action="authorize_remediation",
                outcome=AuditOutcome.DENIED,
                severity=AuditSeverity.ERROR,
                target="remediation",
                correlation_id=str(uuid.uuid4()),
                message="Remediation authorization failed closed.",
                details={"error_type": type(exc).__name__},
            )
            raise

    def _authorize(self, request: Any) -> Any:
        from src.remediation.schema import RemediationMode

        self._workload.require_verified()
        targets = sorted(
            set(request.target_resource_ids) | set(request.target_service_ids)
        )
        target = targets[0] if targets else request.action_id
        action_text = str(request.action_type).strip().lower()
        try:
            action = SecurityRemediationAction(action_text)
        except ValueError:
            action = SecurityRemediationAction.CUSTOM
        platform = (
            self._platform_resolver(request)
            if self._platform_resolver is not None
            else InfrastructurePlatform.GENERIC
        )
        risk = (
            self._risk_resolver(request)
            if self._risk_resolver is not None
            else RiskLevel.CRITICAL
        )
        attributes = request.attributes if isinstance(request.attributes, Mapping) else {}
        confidence_value = attributes.get("confidence", 0.0)
        confidence = (
            float(confidence_value)
            if isinstance(confidence_value, (int, float))
            and not isinstance(confidence_value, bool)
            else 0.0
        )
        evidence_ids = tuple(
            str(value)
            for value in (request.decision_id, request.plan_id, request.incident_id)
            if value
        )
        policy_request = SecurityRemediationRequest(
            request_id=str(request.request_id or request.action_id),
            correlation_id=str(request.incident_id or request.request_id or request.action_id),
            platform=platform,
            action=action,
            target=str(target),
            target_count=max(1, len(targets)),
            risk_level=risk,
            confidence=confidence,
            evidence_ids=evidence_ids,
            rollback_available=(
                bool(self._rollback_checker(request))
                if self._rollback_checker is not None
                else False
            ),
            protected_resource=(
                bool(self._protected_resource_resolver(request))
                if self._protected_resource_resolver is not None
                else True
            ),
            dry_run=request.mode == RemediationMode.DRY_RUN.value,
            human_approved=(
                bool(self._approval_verifier(request))
                if self._approval_verifier is not None
                else False
            ),
        )
        result = self._engine.evaluate(self._workload, policy_request)
        if self._audit is not None:
            outcome = {
                RemediationDecision.ALLOWED: AuditOutcome.SUCCESS,
                RemediationDecision.REQUIRE_APPROVAL: AuditOutcome.REQUIRES_APPROVAL,
                RemediationDecision.DENIED: AuditOutcome.DENIED,
            }[result.decision]
            self._audit.record(
                event_type=AuditEventType.REMEDIATION_DECISION,
                action=action.value,
                outcome=outcome,
                severity=(
                    AuditSeverity.NOTICE
                    if result.decision is RemediationDecision.ALLOWED
                    else AuditSeverity.WARNING
                ),
                target=target,
                correlation_id=policy_request.correlation_id,
                message=result.reason,
                details={
                    "decision_id": result.decision_id,
                    "checks": list(result.checks),
                },
            )
        if result.decision is not RemediationDecision.ALLOWED:
            raise RemediationAuthorizationGateError(result.reason)
        return result


__all__ = [
    "RemediationAuthorizationGate",
    "RemediationAuthorizationGateError",
    "SIGNED_MANIFEST_SUFFIX",
    "SecurityAuditRecorder",
    "SecurityBoundaryError",
    "SecureTelemetryTransport",
    "read_signed_dataset_manifest",
    "read_signed_model_manifest",
    "require_verified_dataset",
    "require_verified_model",
    "signed_manifest_path",
    "write_signed_model_manifest",
    "write_signed_dataset_manifest",
]
