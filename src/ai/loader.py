from __future__ import annotations

"""
Promoted-model discovery and registration for
OSHA AI serving.

Expected promoted model layout:

    models/
    └── <family>/
        └── <model_name>/
            └── <version>/
                ├── model.pt
                └── artifact.json

The loader:

    discovers promoted artifacts
    validates metadata
    verifies SHA-256 integrity
    constructs AIModelDescriptor objects
    registers descriptors
    optionally loads runtime adapters

Training checkpoints and candidate artifacts are
not normal AI-serving inputs.
"""

import json

from pathlib import Path

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

from src.ai.config import (
    AIConfig,
)

from src.ai.errors import (
    AIModelArtifactInvalidError,
    AIModelArtifactNotFoundError,
    AIModelChecksumMismatchError,
    AIModelChecksumMissingError,
    AIModelDiscoveryError,
    AIModelNotPromotedError,
    safe_ai_error_details,
)

from src.ai.registry import (
    AIModelRegistry,
    AIModelRegistryEntry,
)

from src.ai.schema import (
    AIDeviceFamily,
    AIModelDescriptor,
    AIModelFamily,
    AIModelIdentity,
    AIModelState,
)

from src.data.checksums import (
    sha256_file,
)

from src.training.artifacts import (
    ARTIFACT_METADATA_FILE_NAME,
    MODEL_FILE_NAME,
)

from src.training.schema import (
    ArtifactState,
)


# ---------------------------------------------------------------
# Supported artifact metadata version
# ---------------------------------------------------------------

SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = {
    "1",
}


# ---------------------------------------------------------------
# Discovery result
# ---------------------------------------------------------------


class AIModelDiscoveryResult:
    """
    Result of discovering promoted models.

    Invalid artifacts can either abort discovery
    or be collected as bounded failures.
    """

    def __init__(
        self,
        *,
        descriptors: Iterable[AIModelDescriptor] = (),
        failures: Iterable[
            Mapping[
                str,
                Any,
            ]
        ] = (),
    ) -> None:
        self.descriptors = tuple(descriptors)

        self.failures = tuple(dict(failure) for failure in failures)

    @property
    def model_count(
        self,
    ) -> int:
        return len(self.descriptors)

    @property
    def failure_count(
        self,
    ) -> int:
        return len(self.failures)

    @property
    def successful(
        self,
    ) -> bool:
        return self.failure_count == 0

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "model_count": (self.model_count),
            "failure_count": (self.failure_count),
            "successful": (self.successful),
            "models": [descriptor.to_dict() for descriptor in self.descriptors],
            "failures": [dict(failure) for failure in self.failures],
        }


# ---------------------------------------------------------------
# Path protection
# ---------------------------------------------------------------


def _resolved_root(
    value: str | Path,
) -> Path:
    return Path(value).expanduser().resolve()


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
        raise AIModelArtifactInvalidError(
            "Promoted model path escapes the configured models root.",
            details={
                "models_root": str(root),
            },
        )

    return resolved


# ---------------------------------------------------------------
# Metadata reader
# ---------------------------------------------------------------


def read_artifact_metadata(
    path: str | Path,
    *,
    models_root: str | Path,
) -> Dict[str, Any]:
    """
    Read one artifact.json under the controlled
    promoted-model root.
    """

    root = _resolved_root(models_root)

    source = _require_inside(
        root,
        Path(path),
    )

    if source.is_dir():
        source = source / ARTIFACT_METADATA_FILE_NAME

        source = _require_inside(
            root,
            source,
        )

    if not source.exists():
        raise AIModelArtifactNotFoundError(
            "Promoted model metadata was not found.",
            details={
                "path": str(source),
            },
        )

    if not source.is_file():
        raise AIModelArtifactInvalidError(
            "Promoted model metadata path is not a regular file."
        )

    try:
        value = json.loads(source.read_text(encoding="utf-8"))

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise AIModelArtifactInvalidError(
            "Promoted model metadata could not be read.",
            details=safe_ai_error_details(
                exc,
                operation=("read_artifact_metadata"),
            ),
        ) from exc

    if not isinstance(
        value,
        Mapping,
    ):
        raise AIModelArtifactInvalidError(
            "Promoted model metadata root must be an object."
        )

    return dict(value)


# ---------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------


def validate_artifact_metadata(
    value: Mapping[
        str,
        Any,
    ],
    *,
    require_promoted: bool = True,
) -> None:
    required = (
        "model_name",
        "task",
        "version",
        "state",
    )

    missing = [
        field
        for field in required
        if (field not in value or value[field] is None or not str(value[field]).strip())
    ]

    if missing:
        raise AIModelArtifactInvalidError(
            "Promoted model metadata is incomplete.",
            details={
                "missing": missing,
            },
        )

    schema_version = str(
        value.get(
            "schema_version",
            "1",
        )
    ).strip()

    if schema_version not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise AIModelArtifactInvalidError(
            "Promoted model artifact schema version is unsupported.",
            details={
                "schema_version": (schema_version),
            },
        )

    family = str(value["task"]).strip().lower()

    if family not in {item.value for item in AIModelFamily}:
        raise AIModelArtifactInvalidError(
            "Promoted model family is unsupported.",
            details={
                "family": family,
            },
        )

    state = str(value["state"]).strip().lower()

    if require_promoted and state != ArtifactState.PROMOTED.value:
        raise AIModelNotPromotedError(
            "AI serving accepts promoted model artifacts only.",
            details={
                "state": state,
                "model_name": str(value["model_name"]),
                "version": str(value["version"]),
            },
        )


# ---------------------------------------------------------------
# Canonical model path
# ---------------------------------------------------------------


def promoted_model_path(
    metadata_path: str | Path,
    *,
    models_root: str | Path,
) -> Path:
    root = _resolved_root(models_root)

    metadata = _require_inside(
        root,
        Path(metadata_path),
    )

    if metadata.is_dir():
        directory = metadata

    else:
        directory = metadata.parent

    model_path = _require_inside(
        root,
        directory / MODEL_FILE_NAME,
    )

    if not model_path.exists():
        raise AIModelArtifactNotFoundError(
            "Promoted model state file was not found.",
            details={
                "path": str(model_path),
            },
        )

    if not model_path.is_file():
        raise AIModelArtifactInvalidError(
            "Promoted model state path is not a regular file."
        )

    return model_path


# ---------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------


def verify_promoted_model(
    model_path: str | Path,
    *,
    expected_checksum: Optional[str],
    require_checksum: bool = True,
) -> str:
    """
    Verify the promoted model's SHA-256 checksum.

    Returns the calculated checksum.
    """

    source = Path(model_path).expanduser().resolve()

    if not source.exists():
        raise AIModelArtifactNotFoundError(
            "Promoted model file was not found.",
            details={
                "path": str(source),
            },
        )

    if require_checksum and not expected_checksum:
        raise AIModelChecksumMissingError(
            "Promoted model metadata contains no checksum."
        )

    try:
        actual = sha256_file(source)

    except Exception as exc:
        raise AIModelDiscoveryError(
            "Promoted model checksum could not be calculated.",
            details=safe_ai_error_details(
                exc,
                operation=("verify_promoted_model"),
            ),
        ) from exc

    if expected_checksum and actual.lower() != str(expected_checksum).strip().lower():
        raise AIModelChecksumMismatchError(
            "Promoted model checksum verification failed.",
            details={
                "expected_checksum": str(expected_checksum),
                "actual_checksum": (actual),
            },
        )

    return actual


# ---------------------------------------------------------------
# Descriptor construction
# ---------------------------------------------------------------


def descriptor_from_artifact(
    metadata_path: str | Path,
    *,
    models_root: str | Path,
    config: Optional[AIConfig] = None,
) -> AIModelDescriptor:
    """
    Construct one runtime descriptor from a
    promoted artifact directory.
    """

    serving_config = config or AIConfig()

    serving_config.validate()

    value = read_artifact_metadata(
        metadata_path,
        models_root=models_root,
    )

    validate_artifact_metadata(
        value,
        require_promoted=(serving_config.loading.require_promoted_state),
    )

    model_path = promoted_model_path(
        metadata_path,
        models_root=models_root,
    )

    expected_checksum = value.get("checksum")

    if serving_config.loading.verify_checksum:
        actual_checksum = verify_promoted_model(
            model_path,
            expected_checksum=(
                str(expected_checksum) if expected_checksum is not None else None
            ),
            require_checksum=True,
        )

    else:
        actual_checksum = (
            str(expected_checksum) if expected_checksum is not None else None
        )

    family = str(value["task"]).strip().lower()

    model_name = str(value["model_name"]).strip()

    version = str(value["version"]).strip()

    identity = AIModelIdentity(
        model_name=model_name,
        family=family,
        version=version,
        artifact_id=(
            str(value["artifact_id"]) if value.get("artifact_id") is not None else None
        ),
        checkpoint_id=(
            str(value["checkpoint_id"])
            if value.get("checkpoint_id") is not None
            else None
        ),
    )

    capabilities = [
        "inference",
    ]

    if family == AIModelFamily.EVENT.value:
        capabilities.extend(
            (
                "event_classification",
                "source_classification",
                "component_classification",
                "text_encoding",
            )
        )

    elif family == AIModelFamily.ANOMALY.value:
        capabilities.extend(
            (
                "anomaly_scoring",
                "anomaly_classification",
                "representation_encoding",
            )
        )

    elif family == AIModelFamily.INCIDENT_ROOT_CAUSE.value:
        capabilities.extend(
            (
                "incident_inference",
                "root_cause_ranking",
                "severity_classification",
            )
        )

    elif family == AIModelFamily.DECISION_POLICY_RISK.value:
        capabilities.extend(
            (
                "decision_risk_scoring",
                "policy_risk_classification",
            )
        )

    elif family == AIModelFamily.REMEDIATION_RECOMMENDATION.value:
        capabilities.extend(
            (
                "remediation_ranking",
                "safe_action_scoring",
                "recommendation_ranking",
            )
        )

    artifact_attributes = value.get(
        "attributes",
        {},
    )

    if not isinstance(
        artifact_attributes,
        Mapping,
    ):
        artifact_attributes = {}

    descriptor = AIModelDescriptor(
        identity=identity,
        path=str(model_path),
        state=(AIModelState.REGISTERED.value),
        device=(serving_config.device.preferred),
        checksum=(actual_checksum),
        capabilities=(capabilities),
        model_config={},
        attributes={
            **dict(artifact_attributes),
            "artifact_metadata_path": str(
                Path(metadata_path).resolve()
                if Path(metadata_path).is_file()
                else (Path(metadata_path).resolve() / ARTIFACT_METADATA_FILE_NAME)
            ),
            "artifact_state": (str(value["state"])),
            "source_run_id": (value.get("source_run_id")),
            "metrics": (
                dict(
                    value.get(
                        "metrics",
                        {},
                    )
                )
                if isinstance(
                    value.get(
                        "metrics",
                        {},
                    ),
                    Mapping,
                )
                else {}
            ),
        },
    )

    return descriptor.finalize()


# ---------------------------------------------------------------
# Metadata discovery
# ---------------------------------------------------------------


def discover_metadata_files(
    models_root: str | Path,
) -> List[Path]:
    root = _resolved_root(models_root)

    if not root.exists():
        return []

    if not root.is_dir():
        raise AIModelDiscoveryError(
            "Configured promoted-model root is not a directory.",
            details={
                "path": str(root),
            },
        )

    output: List[Path] = []

    for path in root.rglob(ARTIFACT_METADATA_FILE_NAME):
        if not path.is_file():
            continue

        try:
            resolved = _require_inside(
                root,
                path,
            )

        except AIModelArtifactInvalidError:
            continue

        output.append(resolved)

    return sorted(
        output,
        key=lambda item: str(item),
    )


# ---------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------


class AIModelLoader:
    """
    Discover and register promoted model
    artifacts.

    Descriptor discovery does not allocate model
    runtime memory unless eager loading is
    requested or lazy loading is disabled.
    """

    def __init__(
        self,
        *,
        config: Optional[AIConfig] = None,
        registry: Optional[AIModelRegistry] = None,
    ) -> None:
        self.config = config or AIConfig()

        self.config.validate()

        self.registry = registry or AIModelRegistry(config=self.config)

    # ---------------------------------------------------------
    # Root
    # ---------------------------------------------------------

    @property
    def models_root(
        self,
    ) -> Path:
        return Path(self.config.paths.resolved()["models"])

    # ---------------------------------------------------------
    # Discover descriptors
    # ---------------------------------------------------------

    def discover(
        self,
        *,
        fail_fast: bool = False,
    ) -> AIModelDiscoveryResult:
        """
        Discover promoted descriptors.

        fail_fast=True raises on the first invalid
        artifact.

        fail_fast=False collects bounded failure
        metadata and continues.
        """

        metadata_files = discover_metadata_files(self.models_root)

        descriptors: List[AIModelDescriptor] = []

        failures: List[
            Dict[
                str,
                Any,
            ]
        ] = []

        model_ids = set()

        for metadata_path in metadata_files:
            try:
                descriptor = descriptor_from_artifact(
                    metadata_path,
                    models_root=(self.models_root),
                    config=self.config,
                )

                model_id = descriptor.identity.model_id

                if model_id in model_ids:
                    raise (
                        AIModelArtifactInvalidError(
                            "Multiple promoted artifacts have the same model identity.",
                            details={
                                "model_id": (model_id),
                            },
                        )
                    )

                model_ids.add(model_id)

                descriptors.append(descriptor)

            except Exception as exc:
                if fail_fast:
                    raise

                failures.append(
                    {
                        "metadata_path": str(metadata_path),
                        "error_type": (type(exc).__name__),
                        "error_code": (
                            getattr(
                                exc,
                                "code",
                                "model_discovery_error",
                            )
                        ),
                    }
                )

        return AIModelDiscoveryResult(
            descriptors=descriptors,
            failures=failures,
        )

    # ---------------------------------------------------------
    # Register discovered models
    # ---------------------------------------------------------

    def discover_and_register(
        self,
        *,
        fail_fast: bool = False,
        replace: bool = False,
        eager_load: Optional[bool] = None,
    ) -> Tuple[
        AIModelDiscoveryResult,
        List[AIModelRegistryEntry],
    ]:
        result = self.discover(fail_fast=fail_fast)

        load_models = (
            not self.config.loading.lazy_loading
            if eager_load is None
            else bool(eager_load)
        )

        entries: List[AIModelRegistryEntry] = []

        for descriptor in result.descriptors:
            entries.append(
                self.registry.register(
                    descriptor,
                    replace=replace,
                    create_adapter=(load_models),
                    load=load_models,
                )
            )

        return (
            result,
            entries,
        )

    # ---------------------------------------------------------
    # Register explicit artifact
    # ---------------------------------------------------------

    def register_artifact(
        self,
        path: str | Path,
        *,
        replace: bool = False,
        load: bool = False,
    ) -> AIModelRegistryEntry:
        """
        Register one explicit promoted artifact
        directory or artifact.json file.
        """

        descriptor = descriptor_from_artifact(
            path,
            models_root=(self.models_root),
            config=self.config,
        )

        return self.registry.register(
            descriptor,
            replace=replace,
            create_adapter=load,
            load=load,
        )

    # ---------------------------------------------------------
    # Re-scan
    # ---------------------------------------------------------

    def refresh(
        self,
        *,
        fail_fast: bool = False,
        replace: bool = False,
    ) -> AIModelDiscoveryResult:
        result = self.discover(fail_fast=fail_fast)

        existing_ids = {
            descriptor.identity.model_id for descriptor in self.registry.descriptors()
        }

        for descriptor in result.descriptors:
            model_id = descriptor.identity.model_id

            if model_id in existing_ids:
                if not replace:
                    continue

            self.registry.register(
                descriptor,
                replace=replace,
                create_adapter=False,
                load=(not self.config.loading.lazy_loading),
            )

        return result

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        metadata_files = discover_metadata_files(self.models_root)

        return {
            "models_root": str(self.models_root),
            "artifact_metadata_files": len(metadata_files),
            "verify_checksum": (self.config.loading.verify_checksum),
            "require_promoted_state": (self.config.loading.require_promoted_state),
            "lazy_loading": (self.config.loading.lazy_loading),
            "registry": (self.registry.summary()),
            "training_checkpoints_loaded": (False),
            "candidate_artifacts_loaded": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "SUPPORTED_ARTIFACT_SCHEMA_VERSIONS",
    "AIModelDiscoveryResult",
    "read_artifact_metadata",
    "validate_artifact_metadata",
    "promoted_model_path",
    "verify_promoted_model",
    "descriptor_from_artifact",
    "discover_metadata_files",
    "AIModelLoader",
]
