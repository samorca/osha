from __future__ import annotations

"""
Dataset manifest persistence and registry for
OSHA data engineering.
"""

import json

from pathlib import (
    Path,
)

from threading import (
    RLock,
)

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

from src.data.errors import (
    ManifestConflictError,
    ManifestNotFoundError,
    ManifestValidationError,
)

from src.data.schema import (
    DataSource,
    DatasetFile,
    DatasetManifest,
)


# ---------------------------------------------------------------
# Manifest identity
# ---------------------------------------------------------------

ManifestKey = Tuple[
    str,
    str,
]


def manifest_key(
    dataset_id: str,
    version: str,
) -> ManifestKey:
    dataset_id = str(
        dataset_id
    ).strip()

    version = str(
        version
    ).strip()

    if not dataset_id:
        raise (
            ManifestValidationError(
                "dataset_id is required."
            )
        )

    if not version:
        raise (
            ManifestValidationError(
                "dataset version is "
                "required."
            )
        )

    return (
        dataset_id,
        version,
    )


# ---------------------------------------------------------------
# Manifest filename
# ---------------------------------------------------------------

def manifest_filename(
    manifest: DatasetManifest,
) -> str:
    """
    Produce a deterministic manifest filename.

    Dataset IDs and versions are normalized for
    filesystem use while their original values
    remain inside the manifest.
    """

    manifest.finalize()

    def safe(
        value: str,
    ) -> str:
        output = []

        for character in value:
            if (
                character.isalnum()
                or character in {
                    "-",
                    "_",
                    ".",
                }
            ):
                output.append(
                    character
                )

            else:
                output.append(
                    "_"
                )

        normalized = "".join(
            output
        ).strip(
            "._"
        )

        if not normalized:
            raise (
                ManifestValidationError(
                    "Manifest identity "
                    "cannot produce a safe "
                    "filename."
                )
            )

        return normalized

    return (
        safe(
            manifest.dataset_id
        )
        + "-"
        + safe(
            manifest.version
        )
        + ".json"
    )


# ---------------------------------------------------------------
# Manifest serialization
# ---------------------------------------------------------------

def manifest_to_json(
    manifest: DatasetManifest,
    *,
    indent: int = 2,
) -> str:
    manifest.finalize()

    try:
        return json.dumps(
            manifest.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise (
            ManifestValidationError(
                "Dataset manifest "
                "could not be serialized.",
                details={
                    "dataset_id": (
                        manifest.dataset_id
                    ),
                    "version": (
                        manifest.version
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                },
            )
        ) from exc


# ---------------------------------------------------------------
# Manifest parser
# ---------------------------------------------------------------

def manifest_from_dict(
    value: Mapping[
        str,
        Any,
    ],
) -> DatasetManifest:
    """
    Construct a DatasetManifest from an ordinary
    mapping.

    Unknown top-level fields are not silently
    interpreted.
    """

    try:
        source_value = value.get(
            "source"
        )

        if not isinstance(
            source_value,
            Mapping,
        ):
            raise ValueError(
                "Manifest source must "
                "be an object."
            )

        source = DataSource(
            kind=str(
                source_value[
                    "kind"
                ]
            ),

            location=str(
                source_value[
                    "location"
                ]
            ),

            source_id=(
                str(
                    source_value[
                        "source_id"
                    ]
                )
                if source_value.get(
                    "source_id"
                )
                is not None
                else None
            ),

            provider=(
                str(
                    source_value[
                        "provider"
                    ]
                )
                if source_value.get(
                    "provider"
                )
                is not None
                else None
            ),

            attributes=dict(
                source_value.get(
                    "attributes",
                    {},
                )
            ),
        ).finalize()

        files_value = value.get(
            "files",
            [],
        )

        if not isinstance(
            files_value,
            list,
        ):
            raise ValueError(
                "Manifest files must "
                "be a list."
            )

        files: List[
            DatasetFile
        ] = []

        for item in files_value:
            if not isinstance(
                item,
                Mapping,
            ):
                raise ValueError(
                    "Manifest file entry "
                    "must be an object."
                )

            files.append(
                DatasetFile(
                    path=str(
                        item[
                            "path"
                        ]
                    ),

                    checksum=(
                        str(
                            item[
                                "checksum"
                            ]
                        )
                        if item.get(
                            "checksum"
                        )
                        is not None
                        else None
                    ),

                    checksum_algorithm=str(
                        item.get(
                            "checksum_algorithm",
                            "sha256",
                        )
                    ),

                    size_bytes=(
                        int(
                            item[
                                "size_bytes"
                            ]
                        )
                        if item.get(
                            "size_bytes"
                        )
                        is not None
                        else None
                    ),

                    media_type=(
                        str(
                            item[
                                "media_type"
                            ]
                        )
                        if item.get(
                            "media_type"
                        )
                        is not None
                        else None
                    ),

                    record_count=(
                        int(
                            item[
                                "record_count"
                            ]
                        )
                        if item.get(
                            "record_count"
                        )
                        is not None
                        else None
                    ),

                    required=bool(
                        item.get(
                            "required",
                            True,
                        )
                    ),

                    attributes=dict(
                        item.get(
                            "attributes",
                            {},
                        )
                    ),

                    file_id=(
                        str(
                            item[
                                "file_id"
                            ]
                        )
                        if item.get(
                            "file_id"
                        )
                        is not None
                        else None
                    ),
                ).finalize()
            )

        labels_value = value.get(
            "labels",
            [],
        )

        if not isinstance(
            labels_value,
            list,
        ):
            raise ValueError(
                "Manifest labels must "
                "be a list."
            )

        manifest = DatasetManifest(
            dataset_id=str(
                value[
                    "dataset_id"
                ]
            ),

            name=str(
                value[
                    "name"
                ]
            ),

            version=str(
                value[
                    "version"
                ]
            ),

            source=source,

            format=str(
                value[
                    "format"
                ]
            ),

            purpose=str(
                value.get(
                    "purpose",
                    "general",
                )
            ),

            files=files,

            labels=[
                str(
                    label
                )
                for label
                in labels_value
            ],

            attributes=dict(
                value.get(
                    "attributes",
                    {},
                )
            ),

            created_at=str(
                value.get(
                    "created_at",
                    ""
                )
            )
            or DatasetManifest(
                dataset_id="temporary",
                name="temporary",
                version="1",
                source=source,
                format=str(
                    value[
                        "format"
                    ]
                ),
            ).created_at,

            schema_version=str(
                value.get(
                    "schema_version",
                    "1",
                )
            ),
        )

        return manifest.finalize()

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise (
            ManifestValidationError(
                "Dataset manifest is "
                "invalid.",
                details={
                    "error_type": (
                        type(exc).__name__
                    ),
                },
            )
        ) from exc


# ---------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------

def load_manifest(
    path: str | Path,
) -> DatasetManifest:
    manifest_path = Path(
        path
    ).expanduser()

    if not manifest_path.exists():
        raise (
            ManifestNotFoundError(
                "Dataset manifest was "
                "not found.",
                details={
                    "path": str(
                        manifest_path
                    ),
                },
            )
        )

    if not manifest_path.is_file():
        raise (
            ManifestValidationError(
                "Dataset manifest path "
                "is not a file.",
                details={
                    "path": str(
                        manifest_path
                    ),
                },
            )
        )

    try:
        raw = manifest_path.read_text(
            encoding="utf-8"
        )

        value = json.loads(
            raw
        )

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise (
            ManifestValidationError(
                "Dataset manifest could "
                "not be read.",
                details={
                    "path": str(
                        manifest_path
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                },
            )
        ) from exc

    if not isinstance(
        value,
        Mapping,
    ):
        raise (
            ManifestValidationError(
                "Dataset manifest root "
                "must be an object."
            )
        )

    return manifest_from_dict(
        value
    )


# ---------------------------------------------------------------
# Save manifest
# ---------------------------------------------------------------

def save_manifest(
    manifest: DatasetManifest,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Persist one manifest atomically enough for
    normal local dataset preparation.

    Dataset contents are not written here.
    """

    manifest.finalize()

    directory_path = Path(
        directory
    ).expanduser()

    try:
        directory_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as exc:
        raise (
            ManifestValidationError(
                "Manifest directory "
                "could not be created.",
                details={
                    "error_type": (
                        type(exc).__name__
                    ),
                },
            )
        ) from exc

    destination = (
        directory_path
        / manifest_filename(
            manifest
        )
    )

    if (
        destination.exists()
        and not overwrite
    ):
        raise (
            ManifestConflictError(
                "Dataset manifest already "
                "exists.",
                details={
                    "dataset_id": (
                        manifest.dataset_id
                    ),
                    "version": (
                        manifest.version
                    ),
                },
            )
        )

    temporary = (
        destination.with_suffix(
            destination.suffix
            + ".tmp"
        )
    )

    try:
        temporary.write_text(
            manifest_to_json(
                manifest
            )
            + "\n",
            encoding="utf-8",
        )

        temporary.replace(
            destination
        )

    except OSError as exc:
        try:
            temporary.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise (
            ManifestValidationError(
                "Dataset manifest "
                "could not be saved.",
                details={
                    "error_type": (
                        type(exc).__name__
                    ),
                },
            )
        ) from exc

    return destination


# ---------------------------------------------------------------
# Controlled file path resolution
# ---------------------------------------------------------------

def resolve_dataset_file(
    root_directory: str | Path,
    dataset_file: DatasetFile,
) -> Path:
    """
    Resolve a declared dataset file against a
    controlled dataset root.

    Relative manifest paths may not escape the
    configured root using '..' traversal.
    """

    dataset_file.finalize()

    root = Path(
        root_directory
    ).expanduser().resolve()

    declared = Path(
        dataset_file.path
    ).expanduser()

    if declared.is_absolute():
        resolved = (
            declared.resolve()
        )

    else:
        resolved = (
            root
            / declared
        ).resolve()

        try:
            resolved.relative_to(
                root
            )

        except ValueError as exc:
            raise (
                ManifestValidationError(
                    "Dataset file path "
                    "escapes the configured "
                    "data root.",
                    details={
                        "file_id": (
                            dataset_file.
                            file_id
                        ),
                    },
                )
            ) from exc

    return resolved


# ---------------------------------------------------------------
# Manifest registry
# ---------------------------------------------------------------

class ManifestRegistry:
    """
    Thread-safe in-memory registry of dataset
    manifests.

    Identity is:

        dataset_id + version
    """

    def __init__(
        self,
    ) -> None:
        self._manifests: Dict[
            ManifestKey,
            DatasetManifest,
        ] = {}

        self._lock = RLock()

    # ---------------------------------------------------------
    # Register
    # ---------------------------------------------------------

    def register(
        self,
        manifest: DatasetManifest,
    ) -> DatasetManifest:
        manifest.finalize()

        key = manifest_key(
            manifest.dataset_id,
            manifest.version,
        )

        with self._lock:
            if (
                key
                in self._manifests
            ):
                raise (
                    ManifestConflictError(
                        "Dataset manifest "
                        "already registered.",
                        details={
                            "dataset_id": (
                                manifest.
                                dataset_id
                            ),
                            "version": (
                                manifest.
                                version
                            ),
                        },
                    )
                )

            self._manifests[
                key
            ] = manifest

        return manifest

    # ---------------------------------------------------------
    # Upsert
    # ---------------------------------------------------------

    def upsert(
        self,
        manifest: DatasetManifest,
    ) -> DatasetManifest:
        manifest.finalize()

        key = manifest_key(
            manifest.dataset_id,
            manifest.version,
        )

        with self._lock:
            self._manifests[
                key
            ] = manifest

        return manifest

    # ---------------------------------------------------------
    # Get
    # ---------------------------------------------------------

    def get(
        self,
        dataset_id: str,
        version: str,
    ) -> Optional[
        DatasetManifest
    ]:
        key = manifest_key(
            dataset_id,
            version,
        )

        with self._lock:
            return self._manifests.get(
                key
            )

    # ---------------------------------------------------------
    # Require
    # ---------------------------------------------------------

    def require(
        self,
        dataset_id: str,
        version: str,
    ) -> DatasetManifest:
        manifest = self.get(
            dataset_id,
            version,
        )

        if manifest is None:
            raise (
                ManifestNotFoundError(
                    "Dataset manifest is "
                    "not registered.",
                    details={
                        "dataset_id": (
                            dataset_id
                        ),
                        "version": (
                            version
                        ),
                    },
                )
            )

        return manifest

    # ---------------------------------------------------------
    # Remove
    # ---------------------------------------------------------

    def remove(
        self,
        dataset_id: str,
        version: str,
    ) -> DatasetManifest:
        key = manifest_key(
            dataset_id,
            version,
        )

        with self._lock:
            manifest = (
                self._manifests.pop(
                    key,
                    None,
                )
            )

        if manifest is None:
            raise (
                ManifestNotFoundError(
                    "Dataset manifest is "
                    "not registered."
                )
            )

        return manifest

    # ---------------------------------------------------------
    # List
    # ---------------------------------------------------------

    def list(
        self,
        *,
        dataset_id: Optional[
            str
        ] = None,
    ) -> List[
        DatasetManifest
    ]:
        with self._lock:
            values = list(
                self._manifests.
                values()
            )

        if dataset_id is not None:
            values = [
                manifest
                for manifest
                in values
                if (
                    manifest.dataset_id
                    == dataset_id
                )
            ]

        return sorted(
            values,
            key=lambda manifest: (
                manifest.dataset_id,
                manifest.version,
            ),
        )

    # ---------------------------------------------------------
    # Load and register
    # ---------------------------------------------------------

    def load(
        self,
        path: str | Path,
        *,
        replace: bool = False,
    ) -> DatasetManifest:
        manifest = load_manifest(
            path
        )

        if replace:
            return self.upsert(
                manifest
            )

        return self.register(
            manifest
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        manifests = self.list()

        dataset_ids = {
            manifest.dataset_id
            for manifest
            in manifests
        }

        return {
            "manifest_count": (
                len(
                    manifests
                )
            ),

            "dataset_count": (
                len(
                    dataset_ids
                )
            ),

            "versions_by_dataset": {
                dataset_id: [
                    manifest.version
                    for manifest
                    in manifests
                    if (
                        manifest.dataset_id
                        == dataset_id
                    )
                ]
                for dataset_id
                in sorted(
                    dataset_ids
                )
            },
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "ManifestKey",

    "manifest_key",
    "manifest_filename",

    "manifest_to_json",
    "manifest_from_dict",

    "load_manifest",
    "save_manifest",

    "resolve_dataset_file",

    "ManifestRegistry",
]
