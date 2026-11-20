from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from .manifests import DatasetEntry, load_manifest


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] {url}")
    print(f"[DEST]     {destination}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "OSHA-Dataset-Fetcher/1.0"},
    )

    with urllib.request.urlopen(request) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def extract_archive(
    archive_path: Path,
    output_dir: Path,
    archive_type: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXTRACT]  {archive_path}")
    print(f"[TO]       {output_dir}")

    if archive_type == "zip":
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(output_dir)
        return

    if archive_type in {"tar.gz", "tgz"}:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(output_dir)
        return

    raise ValueError(
        f"Unsupported archive type: {archive_type}"
    )


def archive_filename(entry: DatasetEntry) -> str:
    if entry.archive == "zip":
        return f"{entry.name}.zip"

    if entry.archive in {"tar.gz", "tgz"}:
        return f"{entry.name}.tar.gz"

    raise ValueError(
        f"Unsupported archive type for {entry.name}: "
        f"{entry.archive}"
    )


def fetch_dataset(
    entry: DatasetEntry,
    root: Path,
    downloads: Path,
    force: bool,
) -> None:
    output_dir = root / entry.destination

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        print(f"[SKIP]     {entry.name}: already exists")
        return

    filename = archive_filename(entry)
    archive_path = downloads / filename

    if not archive_path.exists() or force:
        download_file(entry.url, archive_path)
    else:
        print(f"[CACHE]    {archive_path}")

    if force and output_dir.exists():
        shutil.rmtree(output_dir)

    extract_archive(
        archive_path=archive_path,
        output_dir=output_dir,
        archive_type=entry.archive,
    )

    print(f"[READY]    {entry.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch datasets declared in an OSHA manifest."
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the YAML dataset manifest.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Download and extract datasets again.",
    )

    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archive files after extraction.",
    )

    args = parser.parse_args()

    manifest = load_manifest(args.manifest)

    root = manifest.destination_root
    downloads = Path("orchestration-ai-data/downloads")

    root.mkdir(parents=True, exist_ok=True)
    downloads.mkdir(parents=True, exist_ok=True)

    enabled = [
        dataset
        for dataset in manifest.datasets
        if dataset.enabled
    ]

    print(f"Dataset:    {manifest.dataset_id}")
    print(f"Version:    {manifest.version}")
    print(f"Upstream:   {manifest.repository}")
    print(f"Destination:{root}")
    print(f"Enabled:    {len(enabled)}")
    print()

    if not enabled:
        print("No datasets are enabled.")
        return

    for entry in enabled:
        print(f"=== {entry.name} ===")

        fetch_dataset(
            entry=entry,
            root=root,
            downloads=downloads,
            force=args.force,
        )

        print()

    if not args.keep_archives:
        for entry in enabled:
            archive_path = downloads / archive_filename(entry)

            if archive_path.exists():
                archive_path.unlink()
                print(f"[CLEAN]    {archive_path}")

    print()
    print("Dataset fetch completed.")


if __name__ == "__main__":
    main()
