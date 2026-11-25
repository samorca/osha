from __future__ import annotations

import json
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.correlation.pipeline import load_telemetry_file
from src.remediation.executor import (
    ExecutorConfig,
    execute_request,
)
from src.remediation.schema import (
    ExecutionStatus,
    RemediationRequest,
    ValidationResult,
)
from src.security.audit.integrity_chain import IntegrityChain
from src.security.audit.signed_events import AuditSigningService
from src.security.crypto.envelopes import EnvelopeService
from src.security.crypto.encryption import EncryptionService
from src.security.crypto.key_provider import (
    InMemoryKeyProvider,
    KeyPurpose,
    KeyReference,
)
from src.security.crypto.signatures import SignatureService
from src.security.identity.workload_identity import (
    AuthenticatedWorkload,
    DeploymentEnvironment,
    IdentityAuthenticationState,
    WorkloadIdentity,
    WorkloadType,
)
from src.security.integration import (
    SecurityBoundaryError,
    SecurityAuditRecorder,
    SecureTelemetryTransport,
    require_verified_dataset,
    require_verified_model,
    signed_manifest_path,
    write_signed_dataset_manifest,
    write_signed_model_manifest,
)
from src.security.provenance.artefact_manifest import (
    ArtefactType,
    create_file_manifest,
)
from src.security.provenance.dataset_signing import (
    DatasetSignatureVerificationError,
    DatasetSigningService,
    DatasetTrustPolicy,
    DatasetVerificationService,
)
from src.security.provenance.model_signing import (
    ModelSignatureVerificationError,
    ModelSigningService,
    ModelTrustPolicy,
    ModelVerificationService,
)


def _signature_services(
    key_id: str,
) -> tuple[SignatureService, SignatureService]:
    reference = KeyReference(key_id=key_id, version="1")
    private_key = Ed25519PrivateKey.generate()
    return (
        SignatureService(
            key_reference=reference,
            private_key=private_key,
        ),
        SignatureService(
            key_reference=reference,
            public_key=private_key.public_key(),
        ),
    )


def _workload(
    name: str,
    workload_type: WorkloadType,
    *,
    signing_key: KeyReference | None = None,
    encryption_key: KeyReference | None = None,
) -> AuthenticatedWorkload:
    identity = WorkloadIdentity(
        trust_domain="osha.test",
        environment=DeploymentEnvironment.TEST,
        workload_type=workload_type,
        namespace="unit",
        name=name,
        signing_key=signing_key,
        encryption_key=encryption_key,
    )
    return AuthenticatedWorkload(
        identity=identity,
        state=IdentityAuthenticationState.VERIFIED,
        authenticated_at_ns=time.time_ns(),
        authentication_method="unit-test",
    )


def _audit_recorder(
    actor: AuthenticatedWorkload,
    name: str,
) -> tuple[SecurityAuditRecorder, IntegrityChain]:
    signer, _ = _signature_services(name)
    chain = IntegrityChain(chain_id=f"audit-{name}")
    return (
        SecurityAuditRecorder(
            signer=AuditSigningService(signer),
            chain=chain,
            actor=actor,
        ),
        chain,
    )


def test_dataset_verification_precedes_use_and_audits_rejection(tmp_path) -> None:
    producer = _workload("trainer", WorkloadType.TRAINING)
    signer, verifier_key = _signature_services("dataset-signer")
    audit, chain = _audit_recorder(producer, "dataset-audit")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"label":"ok"}\n', encoding="utf-8")

    manifest = create_file_manifest(
        dataset,
        artefact_id="dataset:test",
        artefact_name="test-dataset",
        artefact_type=ArtefactType.DATASET,
        artefact_version="1",
        producer=producer.identity,
        pipeline_id="pipeline:training",
        run_id="run:test",
    )
    signed = DatasetSigningService(signer).sign_manifest(manifest)
    write_signed_dataset_manifest(dataset, signed)

    service = DatasetVerificationService(
        signature_service=verifier_key,
        trust_policy=DatasetTrustPolicy(
            trusted_signing_keys=frozenset({signer.key_reference}),
            allowed_producers=frozenset({producer.identity.workload_id}),
            allowed_pipeline_ids=frozenset({"pipeline:training"}),
        ),
    )

    result = require_verified_dataset(dataset, service, audit=audit)
    assert result.valid
    assert chain.record_count == 1
    assert signed_manifest_path(dataset).is_file()

    dataset.write_text('{"label":"tampered"}\n', encoding="utf-8")
    with pytest.raises(DatasetSignatureVerificationError):
        require_verified_dataset(dataset, service, audit=audit)
    assert chain.record_count == 2


def test_training_source_requires_dataset_verification_before_parsing(tmp_path) -> None:
    pytest.importorskip("torch")
    from src.training.schema import TrainingSource, TrainingSourceKind
    from src.training.sources import (
        DiscoveredTrainingSource,
        TrainingSourceManager,
    )

    producer = _workload("trainer", WorkloadType.TRAINING)
    signer, verifier_key = _signature_services("training-source-signer")
    audit, chain = _audit_recorder(producer, "training-source-audit")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{invalid json\n", encoding="utf-8")
    verifier = DatasetVerificationService(
        signature_service=verifier_key,
        trust_policy=DatasetTrustPolicy(
            trusted_signing_keys=frozenset({signer.key_reference}),
            allowed_producers=frozenset({producer.identity.workload_id}),
            allowed_pipeline_ids=frozenset({"pipeline:training"}),
        ),
    )
    manager = TrainingSourceManager(
        dataset_verifier=verifier,
        require_signed_datasets=True,
        security_audit_recorder=audit,
    )
    source = DiscoveredTrainingSource(
        source=TrainingSource(
            kind=TrainingSourceKind.PREPARED_DATASET.value,
            location=str(tmp_path),
        ),
        files=(str(dataset),),
    )

    with pytest.raises(SecurityBoundaryError):
        next(manager.iter_records(source))
    assert chain.record_count == 1


def test_model_verification_preserves_unqualified_status(tmp_path) -> None:
    producer = _workload("trainer", WorkloadType.TRAINING)
    signer, verifier_key = _signature_services("model-signer")
    audit, chain = _audit_recorder(producer, "model-audit")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model bytes")

    manifest = create_file_manifest(
        model,
        artefact_id="model:test",
        artefact_name="test-model",
        artefact_type=ArtefactType.MODEL,
        artefact_version="1",
        producer=producer.identity,
        pipeline_id="pipeline:training",
        run_id="run:test",
        metadata={
            "evaluation_passed": "unknown",
            "qualification_status": "user_accepted_unqualified_candidate",
        },
    )
    signed = ModelSigningService(signer).sign_manifest(manifest)
    write_signed_model_manifest(model, signed)

    service = ModelVerificationService(
        signature_service=verifier_key,
        trust_policy=ModelTrustPolicy(
            trusted_signing_keys=frozenset({signer.key_reference}),
            allowed_producers=frozenset({producer.identity.workload_id}),
            allowed_pipeline_ids=frozenset({"pipeline:training"}),
            allowed_model_ids=frozenset({"model:test"}),
            allowed_versions=frozenset({"1"}),
        ),
    )
    result = require_verified_model(model, service, audit=audit)
    assert result.valid
    assert signed.manifest.metadata["evaluation_passed"] == "unknown"
    assert (
        signed.manifest.metadata["qualification_status"]
        == "user_accepted_unqualified_candidate"
    )
    assert chain.record_count == 1

    model.write_bytes(b"tampered model bytes")
    with pytest.raises(ModelSignatureVerificationError):
        require_verified_model(model, service, audit=audit)
    assert chain.record_count == 2


def test_secure_telemetry_is_opened_before_correlation(tmp_path) -> None:
    signing_key = KeyReference(key_id="collector-signing", version="1")
    sender_signer, receiver_verifier = _signature_services("collector-signing")
    key_provider = InMemoryKeyProvider()
    encryption_key = key_provider.generate_symmetric_key(
        key_id="telemetry-encryption",
        purpose=KeyPurpose.ENCRYPTION,
    )
    encryption = EncryptionService(key_provider)
    sender = _workload(
        "collector",
        WorkloadType.COLLECTOR,
        signing_key=signing_key,
        encryption_key=encryption_key,
    )
    receiver = _workload(
        "correlation",
        WorkloadType.CORRELATION,
        encryption_key=encryption_key,
    )
    sender_audit, sender_chain = _audit_recorder(sender, "telemetry-sender-audit")
    receiver_audit, receiver_chain = _audit_recorder(
        receiver,
        "telemetry-receiver-audit",
    )
    sender_transport = SecureTelemetryTransport(
        envelopes=EnvelopeService(
            encryption_service=encryption,
            signing_service=sender_signer,
        ),
        workload=sender,
        destination=receiver.identity.workload_id,
        audit=sender_audit,
    )
    receiver_transport = SecureTelemetryTransport(
        envelopes=EnvelopeService(
            encryption_service=encryption,
            verification_service=receiver_verifier,
        ),
        workload=receiver,
        destination=receiver.identity.workload_id,
        expected_source=sender.identity.workload_id,
        expected_source_signing_key=signing_key,
        audit=receiver_audit,
    )
    telemetry = {
        "record_id": "record:test",
        "resource": {
            "resource_type": "host",
            "provider": "linux",
            "provider_id": "host-1",
            "environment_id": "test",
        },
        "signal": {
            "signal_type": "log",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "provider": "linux",
            "resource_id": "host-1",
            "environment_id": "test",
        },
        "relationships": [
            {
                "source_resource_id": "host-1",
                "target_resource_id": "service-1",
                "relationship_type": "depends_on",
                "provider": "linux",
            }
        ],
        "native": {},
    }
    envelope = sender_transport.seal(telemetry)
    input_path = tmp_path / "secure-telemetry.jsonl"
    input_path.write_text(
        json.dumps(envelope.as_dict()) + "\n",
        encoding="utf-8",
    )

    records, issues = load_telemetry_file(
        input_path,
        secure_transport=receiver_transport,
        require_secure=True,
    )
    assert not issues
    assert len(records) == 1
    assert len(records[0].relationships) == 1
    assert sender_chain.record_count == 1
    assert receiver_chain.record_count == 1

    with pytest.raises(SecurityBoundaryError):
        receiver_transport.open(envelope.as_dict())
    assert receiver_chain.record_count == 2


def test_execute_mode_without_security_gate_never_dispatches() -> None:
    class AlwaysValidValidator:
        def validate(self, request):
            return ValidationResult(valid=True).finalize()

    class DispatchTracker:
        def __init__(self):
            self.dispatches = 0

        def capabilities(self):
            return ()

        def resolve(self, request, *, preferred_adapter=None):
            self.dispatches += 1
            raise AssertionError("adapter dispatch must not occur")

    request = RemediationRequest(
        action_id="action:test",
        action_type="restart_service",
        environment_id="test",
        mode="execute",
        target_resource_ids=["service:test"],
    )
    registry = DispatchTracker()
    result = execute_request(
        request,
        validator=AlwaysValidValidator(),
        registry=registry,
        config=ExecutorConfig(execution_enabled=True),
    )

    assert result.status == ExecutionStatus.BLOCKED.value
    assert registry.dispatches == 0
