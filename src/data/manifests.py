from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    enabled: bool
    url: str
    archive: str
    destination: str


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    version: str
    repository: str
    destination_root: Path
    datasets: list[DatasetEntry]


def load_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = Path(path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest does not exist: {manifest_path}"
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}

    dataset = data.get("dataset", {})
    source = data.get("source", {})
    destination = data.get("destination", {})
    raw_entries = data.get("datasets", [])

    dataset_id = str(dataset.get("id", "")).strip()
    version = str(dataset.get("version", "")).strip()
    repository = str(source.get("repository", "")).strip()
    destination_root = str(destination.get("root", "")).strip()

    if not dataset_id:
        raise ValueError("dataset.id is required")

    if not version:
        raise ValueError("dataset.version is required")

    if not repository:
        raise ValueError("source.repository is required")

    if not destination_root:
        raise ValueError("destination.root is required")

    entries: list[DatasetEntry] = []

    for index, item in enumerate(raw_entries):
        name = str(item.get("name", "")).strip()
        url = str(item.get("url", "")).strip()
        archive = str(item.get("archive", "")).strip()
        dest = str(item.get("destination", name)).strip()

        if not name:
            raise ValueError(
                f"datasets[{index}].name is required"
            )

        if item.get("enabled", False) and not url:
            raise ValueError(
                f"datasets[{index}].url is required "
                f"when {name} is enabled"
            )

        entries.append(
            DatasetEntry(
                name=name,
                enabled=bool(item.get("enabled", False)),
                url=url,
                archive=archive,
                destination=dest,
            )
        )

    return DatasetManifest(
        dataset_id=dataset_id,
        version=version,
        repository=repository,
        destination_root=Path(destination_root),
        datasets=entries,
    )
