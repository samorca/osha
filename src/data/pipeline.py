from __future__ import annotations

"""
Composition facade for OSHA data engineering.

The DataPipeline coordinates:

    manifests
    fetching
    integrity
    validation
    normalization
    splitting
    corpus preparation

Model training remains in src/training/.
"""

import argparse
import json

from pathlib import Path

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from src.data.checksums import (
    FileIntegrityResult,
    manifest_integrity_ok,
    verify_manifest_files,
)

from src.data.config import (
    DataConfig,
    safe_config_summary,
)

from src.data.errors import (
    DataError,
    DataIntegrityError,
    DataPreparationError,
)

from src.data.fetch import (
    DataFetcher,
    FetchResult,
)

from src.data.manifests import (
    ManifestRegistry,
    load_manifest,
    save_manifest,
)

from src.data.normalize import (
    DatasetNormalizer,
    NormalizerRegistry,
    RecordNormalizer,
)

from src.data.prepare_corpus import (
    CorpusPreparer,
)

from src.data.schema import (
    DatasetManifest,
    PreparationResult,
    ValidationResult,
)

from src.data.split import (
    DatasetSplitter,
)

from src.data.validate import (
    DatasetValidator,
    RecordValidator,
)


# ---------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------


class DataPipeline:
    """
    Unified data-engineering facade for OSHA.

    Main workflow:

        register manifest
            ↓
        fetch raw files
            ↓
        verify integrity
            ↓
        validate
            ↓
        normalize
            ↓
        split
            ↓
        prepare corpus

    This class does not train AI/ML models.
    """

    def __init__(
        self,
        *,
        config: Optional[DataConfig] = None,
        manifests: Optional[ManifestRegistry] = None,
        normalizers: Optional[NormalizerRegistry] = None,
    ) -> None:
        self.config = config or DataConfig()

        self.config.validate()

        # -----------------------------------------------------
        # Registries
        # -----------------------------------------------------

        self.manifests = manifests or ManifestRegistry()

        self.normalizers = normalizers or NormalizerRegistry()

        # -----------------------------------------------------
        # Services
        # -----------------------------------------------------

        self.fetcher = DataFetcher(config=self.config)

        self.validator = DatasetValidator(config=self.config)

        self.normalizer = DatasetNormalizer(registry=(self.normalizers))

        self.splitter = DatasetSplitter(config=(self.config.split))

        self.preparer = CorpusPreparer(
            config=self.config,
            validator=(self.validator),
            normalizer=(self.normalizer),
            splitter=(self.splitter),
        )

    # ---------------------------------------------------------
    # Enabled guard
    # ---------------------------------------------------------

    def _require_enabled(
        self,
    ) -> None:
        if not self.config.enabled:
            raise DataError("Data pipeline is disabled.")

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------

    @property
    def paths(
        self,
    ) -> Dict[str, str]:
        return self.config.paths.resolved()

    # ---------------------------------------------------------
    # Initialize directories
    # ---------------------------------------------------------

    def initialize_directories(
        self,
    ) -> Dict[str, str]:
        """
        Create configured dataset directories.

        This does not fetch or modify datasets.
        """

        self._require_enabled()

        paths = self.paths

        try:
            for path in paths.values():
                Path(path).mkdir(
                    parents=True,
                    exist_ok=True,
                )

        except OSError as exc:
            raise DataPreparationError(
                "Data directory layout could not be initialized.",
                details={
                    "error_type": (type(exc).__name__),
                },
            ) from exc

        return paths

    # ---------------------------------------------------------
    # Register manifest
    # ---------------------------------------------------------

    def register_manifest(
        self,
        manifest: DatasetManifest,
        *,
        persist: bool = False,
        overwrite: bool = False,
    ) -> DatasetManifest:
        self._require_enabled()

        manifest.finalize()

        if overwrite:
            registered = self.manifests.upsert(manifest)

        else:
            registered = self.manifests.register(manifest)

        if persist:
            save_manifest(
                registered,
                self.paths["manifests"],
                overwrite=(overwrite),
            )

        return registered

    # ---------------------------------------------------------
    # Load manifest
    # ---------------------------------------------------------

    def load_manifest(
        self,
        path: str | Path,
        *,
        register: bool = True,
        replace: bool = False,
    ) -> DatasetManifest:
        self._require_enabled()

        manifest = load_manifest(path)

        if register:
            if replace:
                self.manifests.upsert(manifest)

            else:
                self.manifests.register(manifest)

        return manifest

    # ---------------------------------------------------------
    # Resolve registered manifest
    # ---------------------------------------------------------

    def manifest(
        self,
        dataset_id: str,
        version: str,
    ) -> DatasetManifest:
        self._require_enabled()

        return self.manifests.require(
            dataset_id,
            version,
        )

    # ---------------------------------------------------------
    # Register normalizer
    # ---------------------------------------------------------

    def register_normalizer(
        self,
        *,
        record_type: str,
        normalizer: RecordNormalizer,
        provider: Optional[str] = None,
        replace: bool = False,
    ) -> None:
        self._require_enabled()

        self.normalizers.register(
            record_type=(record_type),
            normalizer=normalizer,
            provider=provider,
            replace=replace,
        )

    # ---------------------------------------------------------
    # Fetch
    # ---------------------------------------------------------

    def fetch(
        self,
        dataset_id: str,
        version: str,
        *,
        overwrite: Optional[bool] = None,
    ) -> List[FetchResult]:
        self._require_enabled()

        manifest = self.manifest(
            dataset_id,
            version,
        )

        return self.fetcher.fetch_manifest(
            manifest,
            overwrite=overwrite,
        )

    # ---------------------------------------------------------
    # Integrity
    # ---------------------------------------------------------

    def verify_integrity(
        self,
        dataset_id: str,
        version: str,
        *,
        raise_on_mismatch: bool = True,
    ) -> List[FileIntegrityResult]:
        self._require_enabled()

        manifest = self.manifest(
            dataset_id,
            version,
        )

        results = verify_manifest_files(
            manifest,
            self.paths["raw"],
            config=(self.config.integrity),
            raise_on_mismatch=(raise_on_mismatch),
        )

        if raise_on_mismatch and not manifest_integrity_ok(results):
            raise DataIntegrityError(
                "Dataset integrity verification failed.",
                details={
                    "dataset_id": (dataset_id),
                    "version": (version),
                },
            )

        return results

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    def validate(
        self,
        dataset_id: str,
        version: str,
        *,
        record_validator: Optional[RecordValidator] = None,
    ) -> ValidationResult:
        self._require_enabled()

        manifest = self.manifest(
            dataset_id,
            version,
        )

        return self.validator.validate(
            manifest,
            record_validator=(record_validator),
        )

    # ---------------------------------------------------------
    # Prepare
    # ---------------------------------------------------------

    def prepare(
        self,
        dataset_id: str,
        version: str,
        *,
        record_validator: Optional[RecordValidator] = None,
    ) -> PreparationResult:
        """
        Prepare an already-fetched dataset.

        Validation, normalization, deduplication
        and splitting are performed by
        CorpusPreparer.
        """

        self._require_enabled()

        manifest = self.manifest(
            dataset_id,
            version,
        )

        return self.preparer.prepare(
            manifest,
            record_validator=(record_validator),
        )

    # ---------------------------------------------------------
    # Fetch and prepare
    # ---------------------------------------------------------

    def fetch_and_prepare(
        self,
        dataset_id: str,
        version: str,
        *,
        fetch_overwrite: Optional[bool] = None,
        record_validator: Optional[RecordValidator] = None,
    ) -> PreparationResult:
        """
        Convenience workflow for a registered
        dataset:

            fetch
              ↓
            integrity verification
              ↓
            prepare

        CorpusPreparer performs full validation
        before transformation.
        """

        self._require_enabled()

        self.fetch(
            dataset_id,
            version,
            overwrite=(fetch_overwrite),
        )

        if self.config.integrity.verify_after_fetch:
            self.verify_integrity(
                dataset_id,
                version,
                raise_on_mismatch=True,
            )

        return self.prepare(
            dataset_id,
            version,
            record_validator=(record_validator),
        )

    # ---------------------------------------------------------
    # Start
    # ---------------------------------------------------------

    def start(
        self,
    ) -> Dict[str, Any]:
        """
        Initialize filesystem layout.

        Starting the data pipeline does not fetch,
        prepare, or train anything automatically.
        """

        if not self.config.enabled:
            return {
                "enabled": False,
                "initialized": False,
            }

        paths = self.initialize_directories()

        return {
            "enabled": True,
            "initialized": True,
            "paths": paths,
        }

    # ---------------------------------------------------------
    # Stop
    # ---------------------------------------------------------

    def stop(
        self,
    ) -> Dict[str, Any]:
        """
        The current data pipeline owns no
        persistent network connection or worker.
        """

        return {
            "stopped": True,
            "active_connections": 0,
        }

    # ---------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------

    def __enter__(
        self,
    ) -> "DataPipeline":
        self.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:
        self.stop()

        return False

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "config": (safe_config_summary(self.config)),
            "manifests": (self.manifests.summary()),
            "normalizers": (self.normalizers.summary()),
            "fetcher": (self.fetcher.summary()),
            "validator": (self.validator.summary()),
            "normalizer": (self.normalizer.summary()),
            "splitter": (self.splitter.summary()),
            "preparer": (self.preparer.summary()),
            "training_performed": (False),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "DataPipeline",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OSHA data engineering pipeline",
    )
    parser.add_argument(
        "--wrapper-info",
        action="store_true",
        help="Print pipeline subsystem information.",
    )
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="Create configured data directories.",
    )
    parser.add_argument(
        "--manifest",
        help="Path to a JSON dataset manifest.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the manifest in the in-memory registry.",
    )
    arguments = parser.parse_args()

    pipeline = DataPipeline()

    if arguments.wrapper_info:
        print(
            json.dumps(
                {
                    "module": "src.data.pipeline",
                    "training_performed": False,
                },
                sort_keys=True,
            )
        )
        return

    if arguments.initialize:
        print(
            json.dumps(
                pipeline.initialize_directories(),
                sort_keys=True,
            )
        )
        return

    if arguments.manifest:
        manifest = load_manifest(arguments.manifest)

        if arguments.register:
            pipeline.manifests.register(manifest)

        validation = pipeline.validator.validate(manifest)
        print(
            json.dumps(
                {
                    "manifest": manifest.to_dict(),
                    "validation": validation.to_dict(),
                    "registered": arguments.register,
                },
                sort_keys=True,
            )
        )

        if not validation.valid:
            raise SystemExit(1)
        return

    print(
        json.dumps(
            pipeline.summary(),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
