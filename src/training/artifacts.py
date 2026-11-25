from __future__ import annotations

"""
Model artifact and promotion management for OSHA.

Lifecycle:

    TrainingCheckpoint
        -> evaluated candidate
        -> ModelArtifact
        -> explicit promotion
        -> models/

Creating an artifact does not automatically make
it a production model.
"""

import json
import shutil
import tempfile

from pathlib import Path
from threading import RLock

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.data.checksums import (
    sha256_file,
)

from src.training.config import (
    TrainingConfig,
)

from src.training.errors import (
    ModelArtifactConflictError,
    ModelArtifactError,
    ModelArtifactIntegrityError,
    ModelArtifactNotFoundError,
    ModelPromotionBlockedError,
    ModelPromotionError,
)

from src.training.schema import (
    ArtifactState,
    EvaluationResult,
    ModelArtifact,
    TrainingCheckpoint,
)


# ---------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------

MODEL_FILE_NAME = "model.pt"

ARTIFACT_METADATA_FILE_NAME = "artifact.json"


# ---------------------------------------------------------------
# Safe filesystem identity
# ---------------------------------------------------------------


def safe_artifact_name(
    value: str,
) -> str:
    output: List[str] = []

    for character in str(value):
        if character.isalnum() or character in {
            "-",
            "_",
            ".",
        }:
            output.append(character)

        else:
            output.append("_")

    normalized = "".join(output).strip("._")

    if not normalized:
        raise ModelArtifactError("Artifact identity cannot produce a safe path.")

    return normalized


# ---------------------------------------------------------------
# Path guards
# ---------------------------------------------------------------


def _inside(
    root: Path,
    path: Path,
) -> bool:
    try:
        path.resolve().relative_to(root.resolve())

        return True

    except ValueError:
        return False


def _require_inside(
    root: Path,
    path: Path,
) -> Path:
    resolved = path.expanduser().resolve()

    if not _inside(
        root,
        resolved,
    ):
        raise ModelArtifactError("Model artifact path escapes the configured root.")

    return resolved


# ---------------------------------------------------------------
# Candidate directory
# ---------------------------------------------------------------


def artifact_directory(
    root: str | Path,
    *,
    task: str,
    model_name: str,
    version: str,
) -> Path:
    root_path = Path(root).expanduser().resolve()

    return (
        root_path
        / safe_artifact_name(task)
        / safe_artifact_name(model_name)
        / safe_artifact_name(version)
    )


# ---------------------------------------------------------------
# Promoted model directory
# ---------------------------------------------------------------


def promoted_model_directory(
    root: str | Path,
    *,
    task: str,
    model_name: str,
    version: str,
) -> Path:
    root_path = Path(root).expanduser().resolve()

    return (
        root_path
        / safe_artifact_name(task)
        / safe_artifact_name(model_name)
        / safe_artifact_name(version)
    )


# ---------------------------------------------------------------
# Metadata serialization
# ---------------------------------------------------------------


def artifact_to_json(
    artifact: ModelArtifact,
) -> str:
    artifact.finalize()

    try:
        return json.dumps(
            artifact.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ModelArtifactError(
            "Model artifact metadata could not be serialized.",
            details={
                "artifact_id": (artifact.artifact_id),
                "error_type": (type(exc).__name__),
            },
        ) from exc


# ---------------------------------------------------------------
# Metadata parser
# ---------------------------------------------------------------


def artifact_from_dict(
    value: Mapping[
        str,
        Any,
    ],
) -> ModelArtifact:
    try:
        artifact = ModelArtifact(
            model_name=str(value["model_name"]),
            task=str(value["task"]),
            version=str(value["version"]),
            path=str(value["path"]),
            state=str(
                value.get(
                    "state",
                    ArtifactState.CANDIDATE.value,
                )
            ),
            artifact_id=(
                str(value["artifact_id"])
                if value.get("artifact_id") is not None
                else None
            ),
            source_run_id=(
                str(value["source_run_id"])
                if value.get("source_run_id") is not None
                else None
            ),
            checkpoint_id=(
                str(value["checkpoint_id"])
                if value.get("checkpoint_id") is not None
                else None
            ),
            checksum=(
                str(value["checksum"]) if value.get("checksum") is not None else None
            ),
            metrics={
                str(key): float(metric)
                for key, metric in dict(
                    value.get(
                        "metrics",
                        {},
                    )
                ).items()
            },
            attributes=dict(
                value.get(
                    "attributes",
                    {},
                )
            ),
            created_at=str(value.get("created_at", "")),
        )

        return artifact.finalize()

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ModelArtifactError(
            "Model artifact metadata is invalid.",
            details={
                "error_type": (type(exc).__name__),
            },
        ) from exc


# ---------------------------------------------------------------
# Load artifact metadata
# ---------------------------------------------------------------


def load_artifact_metadata(
    path: str | Path,
) -> ModelArtifact:
    source = Path(path).expanduser()

    if source.is_dir():
        source = source / ARTIFACT_METADATA_FILE_NAME

    if not source.exists():
        raise ModelArtifactNotFoundError(
            "Model artifact metadata was not found.",
            details={
                "path": str(source),
            },
        )

    try:
        value = json.loads(source.read_text(encoding="utf-8"))

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ModelArtifactError(
            "Model artifact metadata could not be read.",
            details={
                "error_type": (type(exc).__name__),
            },
        ) from exc

    if not isinstance(
        value,
        Mapping,
    ):
        raise ModelArtifactError("Model artifact metadata root must be an object.")

    return artifact_from_dict(value)


# ---------------------------------------------------------------
# Artifact registry
# ---------------------------------------------------------------


class ModelArtifactRegistry:
    """
    Thread-safe in-memory artifact registry.
    """

    def __init__(
        self,
    ) -> None:
        self._artifacts: Dict[
            str,
            ModelArtifact,
        ] = {}

        self._lock = RLock()

    def register(
        self,
        artifact: ModelArtifact,
        *,
        replace: bool = False,
    ) -> ModelArtifact:
        artifact.finalize()

        artifact_id = str(artifact.artifact_id)

        with self._lock:
            if artifact_id in self._artifacts and not replace:
                raise (
                    ModelArtifactConflictError(
                        "Model artifact is already registered.",
                        details={
                            "artifact_id": (artifact_id),
                        },
                    )
                )

            self._artifacts[artifact_id] = artifact

        return artifact

    def get(
        self,
        artifact_id: str,
    ) -> Optional[ModelArtifact]:
        with self._lock:
            return self._artifacts.get(str(artifact_id))

    def require(
        self,
        artifact_id: str,
    ) -> ModelArtifact:
        artifact = self.get(artifact_id)

        if artifact is None:
            raise ModelArtifactNotFoundError(
                "Model artifact is not registered.",
                details={
                    "artifact_id": (artifact_id),
                },
            )

        return artifact

    def list(
        self,
        *,
        model_name: Optional[str] = None,
        task: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[ModelArtifact]:
        with self._lock:
            values = list(self._artifacts.values())

        output: List[ModelArtifact] = []

        for artifact in values:
            if model_name is not None and artifact.model_name != model_name:
                continue

            if task is not None and artifact.task != task:
                continue

            if state is not None and artifact.state != state:
                continue

            output.append(artifact)

        return sorted(
            output,
            key=lambda item: (
                item.model_name,
                item.version,
                item.created_at,
            ),
        )

    def summary(
        self,
    ) -> Dict[str, Any]:
        values = self.list()

        by_state: Dict[
            str,
            int,
        ] = {}

        for artifact in values:
            by_state[artifact.state] = (
                by_state.get(
                    artifact.state,
                    0,
                )
                + 1
            )

        return {
            "artifact_count": len(values),
            "by_state": (by_state),
        }


# ---------------------------------------------------------------
# Artifact manager
# ---------------------------------------------------------------


class ModelArtifactManager:
    """
    Convert evaluated checkpoints into candidate
    artifacts and explicitly promote accepted
    artifacts into models/.
    """

    def __init__(
        self,
        *,
        config: Optional[TrainingConfig] = None,
        registry: Optional[ModelArtifactRegistry] = None,
        model_signer: Any = None,
        model_producer: Any = None,
        model_pipeline_id: Optional[str] = None,
        security_audit_recorder: Any = None,
        require_signed_models: bool = False,
    ) -> None:
        self.config = config or TrainingConfig()

        self.config.validate()

        self.registry = registry or ModelArtifactRegistry()
        if require_signed_models and (
            model_signer is None
            or model_producer is None
            or not model_pipeline_id
            or security_audit_recorder is None
        ):
            raise ValueError(
                "Signed model promotion requires a signer, verified producer, "
                "pipeline ID, and signed security audit recorder."
            )
        if model_signer is not None and (
            model_producer is None
            or not model_pipeline_id
            or security_audit_recorder is None
        ):
            raise ValueError(
                "Model signing requires a verified producer, pipeline ID, "
                "and signed security audit recorder."
            )
        if model_signer is not None or model_producer is not None:
            from src.security.identity.workload_identity import AuthenticatedWorkload
            from src.security.integration import SecurityAuditRecorder
            from src.security.provenance.model_signing import ModelSigningService

            if model_signer is not None and not isinstance(
                model_signer,
                ModelSigningService,
            ):
                raise TypeError("model_signer must be ModelSigningService.")
            if model_producer is not None and not isinstance(
                model_producer,
                AuthenticatedWorkload,
            ):
                raise TypeError("model_producer must be AuthenticatedWorkload.")
            if model_signer is not None and not isinstance(
                security_audit_recorder,
                SecurityAuditRecorder,
            ):
                raise TypeError(
                    "security_audit_recorder must be SecurityAuditRecorder."
                )
        if model_producer is not None:
            model_producer.require_verified()
        self.model_signer = model_signer
        self.model_producer = model_producer
        self.model_pipeline_id = model_pipeline_id
        self.security_audit_recorder = security_audit_recorder
        self.require_signed_models = require_signed_models

    # ---------------------------------------------------------
    # Roots
    # ---------------------------------------------------------

    @property
    def artifact_root(
        self,
    ) -> Path:
        return Path(self.config.paths.resolved()["artifacts"])

    @property
    def model_root(
        self,
    ) -> Path:
        return Path(self.config.paths.resolved()["models"])

    # ---------------------------------------------------------
    # Rewrite metadata
    # ---------------------------------------------------------

    @staticmethod
    def _write_metadata(
        artifact: ModelArtifact,
        directory: Path,
    ) -> None:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = directory / ARTIFACT_METADATA_FILE_NAME

        temporary = destination.with_name(destination.name + ".writing")

        try:
            temporary.write_text(
                artifact_to_json(artifact) + "\n",
                encoding="utf-8",
            )

            temporary.replace(destination)

        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

            raise ModelArtifactError(
                "Model artifact metadata could not be written.",
                details={
                    "artifact_id": (artifact.artifact_id),
                    "error_type": (type(exc).__name__),
                },
            ) from exc

    # ---------------------------------------------------------
    # Evaluation metrics
    # ---------------------------------------------------------

    @staticmethod
    def _metrics(
        evaluation: Optional[EvaluationResult],
    ) -> Dict[str, float]:
        if evaluation is None:
            return {}

        evaluation.finalize()

        return {metric.name: float(metric.value) for metric in evaluation.metrics}

    # ---------------------------------------------------------
    # Create candidate
    # ---------------------------------------------------------

    def create_candidate(
        self,
        *,
        checkpoint: TrainingCheckpoint,
        task: str,
        version: str,
        evaluation: Optional[EvaluationResult] = None,
        overwrite: bool = False,
    ) -> ModelArtifact:
        """
        Copy one verified checkpoint state file
        into artifacts/ as a versioned candidate.

        Evaluation status is recorded but
        candidate creation itself is not model
        promotion.
        """

        checkpoint.finalize()

        checkpoint_path = Path(checkpoint.path).expanduser().resolve()

        if not checkpoint_path.exists():
            raise ModelArtifactNotFoundError(
                "Source checkpoint file was not found.",
                details={
                    "checkpoint_id": (checkpoint.checkpoint_id),
                },
            )

        if not checkpoint_path.is_file():
            raise ModelArtifactError("Source checkpoint path is not a regular file.")

        actual_checkpoint_checksum = sha256_file(checkpoint_path)

        if (
            checkpoint.checksum
            and actual_checkpoint_checksum.lower() != checkpoint.checksum.lower()
        ):
            raise (
                ModelArtifactIntegrityError(
                    "Source checkpoint failed checksum verification.",
                    details={
                        "checkpoint_id": (checkpoint.checkpoint_id),
                    },
                )
            )

        if self.config.promotion.require_checksum and not checkpoint.checksum:
            raise (
                ModelPromotionBlockedError(
                    "Candidate creation requires a checksummed checkpoint."
                )
            )

        destination = artifact_directory(
            self.artifact_root,
            task=task,
            model_name=(checkpoint.model_name),
            version=version,
        )

        if destination.exists():
            if not overwrite:
                raise (
                    ModelArtifactConflictError(
                        "Model artifact version already exists.",
                        details={
                            "model_name": (checkpoint.model_name),
                            "version": (version),
                        },
                    )
                )

            shutil.rmtree(destination)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        staging = Path(
            tempfile.mkdtemp(
                prefix=("." + safe_artifact_name(version) + "-"),
                suffix=".artifact",
                dir=str(destination.parent),
            )
        )

        try:
            candidate_model_path = staging / MODEL_FILE_NAME

            shutil.copy2(
                checkpoint_path,
                candidate_model_path,
            )

            checksum = sha256_file(candidate_model_path)

            state = (
                ArtifactState.ACCEPTED.value
                if (evaluation is not None and evaluation.passed is True)
                else (ArtifactState.CANDIDATE.value)
            )

            artifact = ModelArtifact(
                model_name=(checkpoint.model_name),
                task=task,
                version=version,
                path=str(destination / MODEL_FILE_NAME),
                state=state,
                source_run_id=(checkpoint.run_id),
                checkpoint_id=(checkpoint.checkpoint_id),
                checksum=checksum,
                metrics=(self._metrics(evaluation)),
                attributes={
                    "evaluation_id": (evaluation.evaluation_id if evaluation else None),
                    "evaluation_passed": (evaluation.passed if evaluation else None),
                    "source_checkpoint_score": (checkpoint.score),
                    "source_checkpoint_metric": (checkpoint.score_name),
                },
            ).finalize()

            self._write_metadata(
                artifact,
                staging,
            )

            staging.replace(destination)

        except ModelArtifactError:
            shutil.rmtree(
                staging,
                ignore_errors=True,
            )

            raise

        except Exception as exc:
            shutil.rmtree(
                staging,
                ignore_errors=True,
            )

            raise ModelArtifactError(
                "Candidate model artifact could not be created.",
                details={
                    "checkpoint_id": (checkpoint.checkpoint_id),
                    "error_type": (type(exc).__name__),
                    "error_module": (type(exc).__module__),
                },
            ) from exc

        self.registry.register(
            artifact,
            replace=overwrite,
        )

        return artifact

    # ---------------------------------------------------------
    # Verify artifact
    # ---------------------------------------------------------

    def verify(
        self,
        artifact: ModelArtifact,
    ) -> bool:
        artifact.finalize()

        path = Path(artifact.path).expanduser().resolve()

        valid_root = _inside(
            self.artifact_root,
            path,
        ) or _inside(
            self.model_root,
            path,
        )

        if not valid_root:
            raise (
                ModelArtifactIntegrityError(
                    "Model artifact path is outside configured artifact/model roots."
                )
            )

        if not path.exists():
            raise (
                ModelArtifactNotFoundError(
                    "Model artifact file was not found.",
                    details={
                        "artifact_id": (artifact.artifact_id),
                    },
                )
            )

        if not path.is_file():
            raise (
                ModelArtifactIntegrityError(
                    "Model artifact path is not a regular file."
                )
            )

        if not artifact.checksum:
            raise (
                ModelArtifactIntegrityError("Model artifact has no recorded checksum.")
            )

        actual = sha256_file(path)

        if actual.lower() != artifact.checksum.lower():
            raise (
                ModelArtifactIntegrityError(
                    "Model artifact checksum verification failed.",
                    details={
                        "artifact_id": (artifact.artifact_id),
                        "expected_checksum": (artifact.checksum),
                        "actual_checksum": (actual),
                    },
                )
            )

        return True

    # ---------------------------------------------------------
    # Promotion guard
    # ---------------------------------------------------------

    def _validate_promotion(
        self,
        artifact: ModelArtifact,
        evaluation: Optional[EvaluationResult],
    ) -> None:
        if not (self.config.promotion.enabled):
            raise (ModelPromotionBlockedError("Model promotion is disabled."))

        artifact.finalize()

        self.verify(artifact)

        if self.config.promotion.require_evaluation_pass:
            passed = (
                evaluation.passed
                if evaluation is not None
                else artifact.attributes.get("evaluation_passed")
            )

            if passed is not True:
                raise (
                    ModelPromotionBlockedError(
                        "Model artifact has not passed required evaluation."
                    )
                )

        if self.config.evaluation.require_test_before_promotion:
            if evaluation is None or evaluation.split != "test":
                raise (
                    ModelPromotionBlockedError(
                        "Model promotion requires a passing test evaluation."
                    )
                )

            if evaluation.passed is not True:
                raise (ModelPromotionBlockedError("Test evaluation did not pass."))

    # ---------------------------------------------------------
    # Promote
    # ---------------------------------------------------------

    def promote(
        self,
        artifact: ModelArtifact,
        *,
        evaluation: Optional[EvaluationResult] = None,
    ) -> ModelArtifact:
        """
        Explicitly promote an accepted candidate
        into models/.

        Existing promoted versions aren't
        replaced unless configuration explicitly
        permits it.
        """

        self._validate_promotion(
            artifact,
            evaluation,
        )

        source = Path(artifact.path).resolve()

        destination_directory = promoted_model_directory(
            self.model_root,
            task=artifact.task,
            model_name=(artifact.model_name),
            version=(artifact.version),
        )

        destination = destination_directory / MODEL_FILE_NAME

        if destination_directory.exists():
            if not (self.config.promotion.allow_replace_promoted_model):
                raise (
                    ModelArtifactConflictError(
                        "Promoted model version already exists.",
                        details={
                            "model_name": (artifact.model_name),
                            "version": (artifact.version),
                        },
                    )
                )

            shutil.rmtree(destination_directory)

        destination_directory.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        staging = Path(
            tempfile.mkdtemp(
                prefix=("." + safe_artifact_name(artifact.version) + "-"),
                suffix=".promoting",
                dir=str(destination_directory.parent),
            )
        )

        try:
            staged_model = staging / MODEL_FILE_NAME

            shutil.copy2(
                source,
                staged_model,
            )

            promoted_checksum = sha256_file(staged_model)

            if (
                artifact.checksum
                and promoted_checksum.lower() != artifact.checksum.lower()
            ):
                raise (
                    ModelArtifactIntegrityError(
                        "Promoted model copy failed checksum verification."
                    )
                )

            promoted = ModelArtifact(
                model_name=(artifact.model_name),
                task=artifact.task,
                version=artifact.version,
                path=str(destination),
                state=(ArtifactState.PROMOTED.value),
                artifact_id=(artifact.artifact_id),
                source_run_id=(artifact.source_run_id),
                checkpoint_id=(artifact.checkpoint_id),
                checksum=(promoted_checksum),
                metrics=dict(artifact.metrics),
                attributes={
                    **artifact.attributes,
                    "promoted": True,
                    "evaluation_id": (
                        evaluation.evaluation_id
                        if evaluation
                        else artifact.attributes.get("evaluation_id")
                    ),
                },
                created_at=(artifact.created_at),
            ).finalize()

            self._write_metadata(
                promoted,
                staging,
            )

            if self.model_signer is not None:
                from src.security.audit.signed_events import (
                    AuditEventType,
                    AuditOutcome,
                    AuditSeverity,
                )
                from src.security.integration import write_signed_model_manifest
                from src.security.provenance.artefact_manifest import (
                    ArtefactType,
                    create_file_manifest,
                )

                producer = self.model_producer.require_verified()
                if not promoted.source_run_id:
                    raise ModelPromotionError(
                        "Signed model promotion requires a source training run ID."
                    )

                manual_authorization = artifact.attributes.get(
                    "manual_install_authorization",
                    {},
                )
                qualification_status = (
                    manual_authorization.get("qualification_status", "unspecified")
                    if isinstance(manual_authorization, Mapping)
                    else "unspecified"
                )
                evaluation_passed = (
                    evaluation.passed
                    if evaluation is not None
                    else artifact.attributes.get("evaluation_passed")
                )
                manifest = create_file_manifest(
                    staged_model,
                    artefact_id=str(promoted.artifact_id),
                    artefact_name=promoted.model_name,
                    artefact_type=ArtefactType.MODEL,
                    artefact_version=promoted.version,
                    producer=producer,
                    pipeline_id=self.model_pipeline_id,
                    run_id=promoted.source_run_id,
                    metadata={
                        "evaluation_passed": (
                            "unknown"
                            if evaluation_passed is None
                            else str(evaluation_passed).lower()
                        ),
                        "qualification_status": str(qualification_status),
                    },
                )
                signed_manifest = self.model_signer.sign_manifest(manifest)
                write_signed_model_manifest(staged_model, signed_manifest)
                self.security_audit_recorder.record(
                    event_type=AuditEventType.PROVENANCE_VERIFICATION,
                    action="sign_promoted_model",
                    outcome=AuditOutcome.ACCEPTED,
                    severity=AuditSeverity.NOTICE,
                    target=f"{promoted.model_name}:{promoted.version}",
                    correlation_id=str(promoted.artifact_id),
                    message="Promoted model manifest was signed.",
                    details={
                        "evaluation_passed": evaluation_passed,
                        "qualification_status": str(qualification_status),
                    },
                )
            elif self.require_signed_models:
                raise ModelPromotionError(
                    "Model promotion requires a signed manifest."
                )

            staging.replace(destination_directory)

        except ModelArtifactError:
            shutil.rmtree(
                staging,
                ignore_errors=True,
            )

            raise

        except Exception as exc:
            shutil.rmtree(
                staging,
                ignore_errors=True,
            )

            raise ModelPromotionError(
                "Model artifact could not be promoted.",
                details={
                    "artifact_id": (artifact.artifact_id),
                    "error_type": (type(exc).__name__),
                    "error_module": (type(exc).__module__),
                },
            ) from exc

        artifact.state = ArtifactState.PROMOTED.value

        artifact.path = str(destination)

        artifact.checksum = promoted.checksum

        artifact.attributes.update(promoted.attributes)

        self.registry.register(
            artifact,
            replace=True,
        )

        return artifact

    # ---------------------------------------------------------
    # Reject candidate
    # ---------------------------------------------------------

    def reject(
        self,
        artifact: ModelArtifact,
        *,
        reason_code: Optional[str] = None,
    ) -> ModelArtifact:
        artifact.finalize()

        if artifact.state == ArtifactState.PROMOTED.value:
            raise (
                ModelArtifactError(
                    "A promoted artifact cannot be rejected by the candidate path."
                )
            )

        artifact.state = ArtifactState.REJECTED.value

        if reason_code:
            artifact.attributes["rejection_reason_code"] = str(reason_code)

        directory = Path(artifact.path).parent

        self._write_metadata(
            artifact,
            directory,
        )

        self.registry.register(
            artifact,
            replace=True,
        )

        return artifact

    # ---------------------------------------------------------
    # Discover candidate artifacts
    # ---------------------------------------------------------

    def discover(
        self,
        *,
        promoted: bool = False,
    ) -> List[ModelArtifact]:
        root = self.model_root if promoted else self.artifact_root

        if not root.exists():
            return []

        output: List[ModelArtifact] = []

        for metadata_path in sorted(
            root.rglob(ARTIFACT_METADATA_FILE_NAME),
            key=lambda item: str(item),
        ):
            artifact = load_artifact_metadata(metadata_path)

            artifact.path = str(metadata_path.parent / MODEL_FILE_NAME)

            if promoted:
                artifact.state = ArtifactState.PROMOTED.value

            artifact.finalize()

            self.registry.register(
                artifact,
                replace=True,
            )

            output.append(artifact)

        return output

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "artifact_root": str(self.artifact_root),
            "model_root": str(self.model_root),
            "promotion_enabled": (self.config.promotion.enabled),
            "automatic_promotion": (self.config.promotion.automatic),
            "require_evaluation_pass": (self.config.promotion.require_evaluation_pass),
            "require_test_before_promotion": (
                self.config.evaluation.require_test_before_promotion
            ),
            "require_checksum": (self.config.promotion.require_checksum),
            "allow_replace_promoted_model": (
                self.config.promotion.allow_replace_promoted_model
            ),
            "registry": (self.registry.summary()),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "MODEL_FILE_NAME",
    "ARTIFACT_METADATA_FILE_NAME",
    "safe_artifact_name",
    "artifact_directory",
    "promoted_model_directory",
    "artifact_to_json",
    "artifact_from_dict",
    "load_artifact_metadata",
    "ModelArtifactRegistry",
    "ModelArtifactManager",
]
