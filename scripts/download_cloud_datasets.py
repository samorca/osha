from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path.home() / "osha" / "orchestration-ai-data"
RAW = ROOT / "raw"
DOWNLOADS = ROOT / "downloads"


DATASETS = [
    {
        "name": "alibaba-2018-processed",
        "url": (
            "https://zenodo.org/records/14564935/files/"
            "machine_usage_days_1_to_8_grouped_300_seconds.csv"
            "?download=1"
        ),
        "filename": "machine_usage_days_1_to_8_grouped_300_seconds.csv",
        "destination": RAW / "alibaba-cluster",
        "archive": None,
        "md5": "6c2312c01f1c4dc4b6863c4402976dee",
    },
    {
        "name": "google-2019-processed",
        "url": (
            "https://zenodo.org/records/14564935/files/"
            "instance_usage_grouped_300_seconds_month.csv"
            "?download=1"
        ),
        "filename": "instance_usage_grouped_300_seconds_month.csv",
        "destination": RAW / "google-cluster",
        "archive": None,
        "md5": "08d31b15f3203cd40407bdfb6ca902ea",
    },
    {
        "name": "azure-v2-processed",
        "url": (
            "https://zenodo.org/records/14564935/files/"
            "vm_cpu_readings_month_aggregated_cpu_mem.csv"
            "?download=1"
        ),
        "filename": "vm_cpu_readings_month_aggregated_cpu_mem.csv",
        "destination": RAW / "azure",
        "archive": None,
        "md5": "b14d5002a7d2f7e0033a0deae63d2ced",
    },
    {
        "name": "cloud-native-anomaly",
        "url": (
            "https://zenodo.org/records/19926114/files/"
            "AIClusterKPI.zip?download=1"
        ),
        "filename": "AIClusterKPI.zip",
        "destination": RAW / "cloud-native-anomaly",
        "archive": "zip",
        "md5": "6786283f55ac25b59e56cc8f22faa21e",
    },
]


def md5sum(path: Path) -> str:
    digest = hashlib.md5()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    print(f"[DOWNLOAD] {url}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "OSHA-Dataset-Manager/1.0"},
    )

    with urllib.request.urlopen(request) as response:
        with target.open("wb") as output:
            shutil.copyfileobj(response, output)


def process_dataset(dataset: dict) -> None:
    name = dataset["name"]
    destination = dataset["destination"]

    destination.mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)

    archive_file = DOWNLOADS / dataset["filename"]

    print()
    print("=" * 70)
    print(f"Dataset: {name}")
    print("=" * 70)

    if not archive_file.exists():
        download(dataset["url"], archive_file)
    else:
        print(f"[CACHE] {archive_file}")

    expected = dataset.get("md5")

    if expected:
        actual = md5sum(archive_file)

        if actual != expected:
            raise RuntimeError(
                f"Checksum failed for {name}\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )

        print(f"[CHECKSUM OK] {actual}")

    if dataset["archive"] == "zip":
        print(f"[EXTRACT] {archive_file} -> {destination}")

        with zipfile.ZipFile(archive_file, "r") as archive:
            archive.extractall(destination)

    else:
        target = destination / dataset["filename"]

        print(f"[COPY] {archive_file} -> {target}")

        shutil.copy2(archive_file, target)

    print(f"[READY] {destination}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        process_dataset(dataset)

    print()
    print("All selected cloud datasets are ready.")
    print(f"Raw dataset root: {RAW}")


if __name__ == "__main__":
    main()
