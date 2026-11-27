from __future__ import annotations

"""
Training-source discovery and safe record loading
for OSHA.

Training inputs may come from:

    data/prepared/
    dataset/processed/
    dataset/normalized/
    dataset/correlated/
    dataset/intelligence/
    dataset/decision/
    dataset/remediation/
    dataset/memory/

Direct live and quarantine inputs are excluded
from normal training discovery.
"""

import json

from dataclasses import (
    dataclass,
)

from pathlib import Path

from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from src.training.config import (
    TrainingConfig,
)

from src.training.errors import (
    TrainingSourceError,
    TrainingSourceFormatError,
    TrainingSourceNotFoundError,
    TrainingSourceUnsupportedError,
)

from src.training.schema import (
    TrainingSource,
    TrainingSourceKind,
)


# ---------------------------------------------------------------
# Supported artifact types
# ---------------------------------------------------------------

SUPPORTED_SUFFIXES = {
    ".json",
    ".jsonl",
}


# ---------------------------------------------------------------
# Explicitly excluded operational directories
# ---------------------------------------------------------------

EXCLUDED_OPERATIONAL_DIRECTORIES = {
    "live",
    "quarantine",
    "raw",
    "audit",
}


# ---------------------------------------------------------------
# Operational source mapping
# ---------------------------------------------------------------

OPERATIONAL_SOURCE_DIRECTORIES = {
    TrainingSourceKind.PROCESSED_OPERATIONAL_DATA.value: ("processed"),
    TrainingSourceKind.NORMALIZED_TELEMETRY.value: ("normalized"),
    TrainingSourceKind.CORRELATED_DATA.value: ("correlated"),
    TrainingSourceKind.INTELLIGENCE_DATA.value: ("intelligence"),
    TrainingSourceKind.DECISION_DATA.value: ("decision"),
    TrainingSourceKind.REMEDIATION_DATA.value: ("remediation"),
    TrainingSourceKind.MEMORY_DATA.value: ("memory"),
}


# ---------------------------------------------------------------
# Discovered source
# ---------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredTrainingSource:
    source: TrainingSource

    files: Tuple[
        str,
        ...,
    ]

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "source": (self.source.to_dict()),
            "file_count": len(self.files),
            "files": list(self.files),
        }


# ---------------------------------------------------------------
# Safe root/path resolution
# ---------------------------------------------------------------


def _root(
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


def _require_file_inside(
    root: Path,
    path: Path,
) -> Path:
    resolved = path.expanduser().resolve()

    if not _inside(
        root,
        resolved,
    ):
        raise TrainingSourceError("Training source path escapes its configured root.")

    if not resolved.exists():
        raise TrainingSourceNotFoundError(
            "Training source file was not found.",
            details={
                "path": str(resolved),
            },
        )

    if not resolved.is_file():
        raise TrainingSourceError("Training source path is not a regular file.")

    return resolved


# ---------------------------------------------------------------
# Discover compatible files
# ---------------------------------------------------------------


def discover_files(
    root: str | Path,
    *,
    recursive: bool = True,
) -> List[Path]:
    root_path = _root(root)

    if not root_path.exists():
        return []

    if not root_path.is_dir():
        raise TrainingSourceError(
            "Training source root is not a directory.",
            details={
                "path": str(root_path),
            },
        )

    iterator = root_path.rglob("*") if recursive else root_path.glob("*")

    output: List[Path] = []

    for path in iterator:
        if not path.is_file():
            continue

        if path.name.endswith(".osha-signed.json"):
            continue

        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        if not _inside(
            root_path,
            path,
        ):
            continue

        output.append(path.resolve())

    return sorted(
        output,
        key=lambda path: str(path),
    )


# ---------------------------------------------------------------
# JSONL streaming
# ---------------------------------------------------------------


def _iter_jsonl(
    path: Path,
) -> Iterator[
    Dict[
        str,
        Any,
    ]
]:
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as stream:
            for line_number, line in enumerate(
                stream,
                start=1,
            ):
                text = line.strip()

                if not text:
                    continue

                try:
                    value = json.loads(text)

                except json.JSONDecodeError as exc:
                    raise (
                        TrainingSourceFormatError(
                            "Training JSONL contains invalid JSON.",
                            details={
                                "path": str(path),
                                "line_number": (line_number),
                            },
                        )
                    ) from exc

                if not isinstance(
                    value,
                    Mapping,
                ):
                    raise (
                        TrainingSourceFormatError(
                            "Training JSONL record must be an object.",
                            details={
                                "path": str(path),
                                "line_number": (line_number),
                            },
                        )
                    )

                yield dict(value)

    except TrainingSourceError:
        raise

    except (
        OSError,
        UnicodeError,
    ) as exc:
        raise TrainingSourceError(
            "Training JSONL source could not be read.",
            details={
                "path": str(path),
                "error_type": (type(exc).__name__),
            },
        ) from exc


# ---------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------


def _iter_json(
    path: Path,
) -> Iterator[
    Dict[
        str,
        Any,
    ]
]:
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as stream:
            value = json.load(stream)

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise TrainingSourceFormatError(
            "Training JSON source could not be parsed.",
            details={
                "path": str(path),
                "error_type": (type(exc).__name__),
            },
        ) from exc

    if isinstance(
        value,
        Mapping,
    ):
        yield dict(value)

        return

    if isinstance(
        value,
        list,
    ):
        for index, item in enumerate(
            value,
            start=1,
        ):
            if not isinstance(
                item,
                Mapping,
            ):
                raise (
                    TrainingSourceFormatError(
                        "Training JSON array entry must be an object.",
                        details={
                            "path": str(path),
                            "record_number": (index),
                        },
                    )
                )

            yield dict(item)

        return

    raise TrainingSourceFormatError("Training JSON root must be an object or array.")


# ---------------------------------------------------------------
# Generic source reader
# ---------------------------------------------------------------


def iter_source_file(
    path: str | Path,
    *,
    root: str | Path,
) -> Iterator[
    Dict[
        str,
        Any,
    ]
]:
    root_path = _root(root)

    source = _require_file_inside(
        root_path,
        Path(path),
    )

    suffix = source.suffix.lower()

    if suffix == ".jsonl":
        yield from _iter_jsonl(source)

        return

    if suffix == ".json":
        yield from _iter_json(source)

        return

    raise TrainingSourceUnsupportedError(
        "Training source file type is unsupported.",
        details={
            "suffix": suffix,
        },
    )


# ---------------------------------------------------------------
# Training source manager
# ---------------------------------------------------------------


class TrainingSourceManager:
    """
    Discover eligible training artifacts without
    silently mixing every OSHA data directory.

    Source inclusion remains controlled through
    TrainingSourceConfig.
    """

    def __init__(
        self,
        *,
        config: Optional[TrainingConfig] = None,
        dataset_verifier: Any = None,
        require_signed_datasets: bool = False,
        security_audit_recorder: Any = None,
    ) -> None:
        self.config = config or TrainingConfig()

        self.config.validate()

        if require_signed_datasets and dataset_verifier is None:
            raise ValueError(
                "require_signed_datasets requires a dataset verifier."
            )
        if dataset_verifier is not None:
            from src.security.integration import SecurityAuditRecorder
            from src.security.provenance.dataset_signing import (
                DatasetVerificationService,
            )

            if not isinstance(dataset_verifier, DatasetVerificationService):
                raise TypeError(
                    "dataset_verifier must be DatasetVerificationService."
                )
            if not isinstance(security_audit_recorder, SecurityAuditRecorder):
                raise ValueError(
                    "Dataset verification requires a signed security audit recorder."
                )
        self.dataset_verifier = dataset_verifier
        self.require_signed_datasets = require_signed_datasets
        self.security_audit_recorder = security_audit_recorder

    # ---------------------------------------------------------
    # Resolved roots
    # ---------------------------------------------------------

    @property
    def prepared_root(
        self,
    ) -> Path:
        return Path(self.config.paths.resolved()["prepared_data"])

    @property
    def operational_root(
        self,
    ) -> Path:
        return Path(self.config.paths.resolved()["orchestration_data"])

    # ---------------------------------------------------------
    # Prepared data discovery
    # ---------------------------------------------------------

    def discover_prepared(
        self,
    ) -> List[DiscoveredTrainingSource]:
        if not (self.config.sources.include_prepared_datasets):
            return []

        root = self.prepared_root

        if not root.exists():
            return []

        sources: List[DiscoveredTrainingSource] = []

        # Expected data layout:
        #
        # data/prepared/
        #   <dataset_id>/
        #     <version>/
        #       dataset.json
        #       train.jsonl
        #       validation.jsonl
        #       test.jsonl

        for dataset_directory in sorted(
            root.iterdir(),
            key=lambda path: path.name,
        ):
            if not dataset_directory.is_dir():
                continue

            for version_directory in sorted(
                dataset_directory.iterdir(),
                key=lambda path: path.name,
            ):
                if not (version_directory.is_dir()):
                    continue

                files = discover_files(
                    version_directory,
                    recursive=False,
                )

                training_files = [
                    path
                    for path in files
                    if path.name
                    in {
                        "train.jsonl",
                        "validation.jsonl",
                        "test.jsonl",
                    }
                ]

                if not training_files:
                    continue

                source = TrainingSource(
                    kind=(TrainingSourceKind.PREPARED_DATASET.value),
                    location=str(version_directory.resolve()),
                    source_id=(
                        "prepared:"
                        + dataset_directory.name
                        + ":"
                        + version_directory.name
                    ),
                    dataset_id=(dataset_directory.name),
                    dataset_version=(version_directory.name),
                    attributes={
                        "origin": ("data_pipeline"),
                    },
                ).finalize()

                sources.append(
                    DiscoveredTrainingSource(
                        source=source,
                        files=tuple(str(path) for path in training_files),
                    )
                )

        return sources

    # ---------------------------------------------------------
    # Whether operational source is enabled
    # ---------------------------------------------------------

    def _operational_enabled(
        self,
        kind: str,
    ) -> bool:
        source_config = self.config.sources

        mapping = {
            TrainingSourceKind.PROCESSED_OPERATIONAL_DATA.value: (
                source_config.include_processed_operational_data
            ),
            TrainingSourceKind.NORMALIZED_TELEMETRY.value: (
                source_config.include_normalized_telemetry
            ),
            TrainingSourceKind.CORRELATED_DATA.value: (
                source_config.include_correlated_data
            ),
            TrainingSourceKind.INTELLIGENCE_DATA.value: (
                source_config.include_intelligence_data
            ),
            TrainingSourceKind.DECISION_DATA.value: (
                source_config.include_decision_data
            ),
            TrainingSourceKind.REMEDIATION_DATA.value: (
                source_config.include_remediation_data
            ),
            TrainingSourceKind.MEMORY_DATA.value: (source_config.include_memory_data),
        }

        return bool(
            mapping.get(
                kind,
                False,
            )
        )

    # ---------------------------------------------------------
    # Operational discovery
    # ---------------------------------------------------------

    def discover_operational(
        self,
    ) -> List[DiscoveredTrainingSource]:
        root = self.operational_root

        if not root.exists():
            return []

        if not root.is_dir():
            raise TrainingSourceError(
                "Configured orchestration data root is not a directory."
            )

        sources: List[DiscoveredTrainingSource] = []

        for (
            kind,
            directory_name,
        ) in OPERATIONAL_SOURCE_DIRECTORIES.items():
            if not self._operational_enabled(kind):
                continue

            if directory_name in EXCLUDED_OPERATIONAL_DIRECTORIES:
                continue

            directory = root / directory_name

            if not directory.exists():
                continue

            files = discover_files(
                directory,
                recursive=True,
            )

            if not files:
                continue

            source = TrainingSource(
                kind=kind,
                location=str(directory.resolve()),
                source_id=("operational:" + directory_name),
                attributes={
                    "origin": ("dataset"),
                    "directory": (directory_name),
                },
            ).finalize()

            sources.append(
                DiscoveredTrainingSource(
                    source=source,
                    files=tuple(str(path) for path in files),
                )
            )

        return sources

    # ---------------------------------------------------------
    # Discover all eligible sources
    # ---------------------------------------------------------

    def discover(
        self,
    ) -> List[DiscoveredTrainingSource]:
        return [
            *self.discover_prepared(),
            *self.discover_operational(),
        ]

    # ---------------------------------------------------------
    # Read source
    # ---------------------------------------------------------

    def iter_records(
        self,
        source: DiscoveredTrainingSource,
    ) -> Iterator[
        Dict[
            str,
            Any,
        ]
    ]:
        """
        Iterate records belonging to one
        discovered source.

        Provenance metadata is added under a
        reserved training-source key rather than
        replacing source fields.
        """

        root = Path(source.source.location).resolve()

        for file_path in source.files:
            path = Path(file_path).resolve()

            if self.dataset_verifier is not None:
                from src.security.integration import require_verified_dataset

                require_verified_dataset(
                    path,
                    self.dataset_verifier,
                    audit=self.security_audit_recorder,
                )
            elif self.require_signed_datasets:
                raise TrainingSourceError(
                    "Training source has no configured dataset verifier."
                )

            for value in iter_source_file(
                path,
                root=root,
            ):
                record = dict(value)

                record["_training_source"] = {
                    "kind": (source.source.kind),
                    "source_id": (source.source.source_id),
                    "dataset_id": (source.source.dataset_id),
                    "dataset_version": (source.source.dataset_version),
                    "provider": (source.source.provider),
                    "file": str(path),
                }

                yield record

    # ---------------------------------------------------------
    # Read all configured sources
    # ---------------------------------------------------------

    def iter_all_records(
        self,
    ) -> Iterator[
        Dict[
            str,
            Any,
        ]
    ]:
        for source in self.discover():
            yield from self.iter_records(source)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        discovered = self.discover()

        by_kind: Dict[
            str,
            int,
        ] = {}

        file_count = 0

        for item in discovered:
            kind = item.source.kind

            by_kind[kind] = (
                by_kind.get(
                    kind,
                    0,
                )
                + 1
            )

            file_count += len(item.files)

        return {
            "prepared_root": str(self.prepared_root),
            "operational_root": str(self.operational_root),
            "source_count": len(discovered),
            "file_count": (file_count),
            "sources_by_kind": (by_kind),
            "supported_file_types": (sorted(SUPPORTED_SUFFIXES)),
            "excluded_operational_directories": (
                sorted(EXCLUDED_OPERATIONAL_DIRECTORIES)
            ),
            "direct_live_training": (False),
            "quarantine_training": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "SUPPORTED_SUFFIXES",
    "EXCLUDED_OPERATIONAL_DIRECTORIES",
    "OPERATIONAL_SOURCE_DIRECTORIES",
    "DiscoveredTrainingSource",
    "discover_files",
    "iter_source_file",
    "TrainingSourceManager",
]
