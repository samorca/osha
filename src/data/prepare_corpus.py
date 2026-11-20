from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger("osha.data.prepare")

STRUCTURED_SUFFIX = "_2k.log_structured.csv"
TEMPLATE_SUFFIXES = (
    "_2k.log_templates.csv",
    "_templates.csv",
)

METRIC_SOURCES = {
    "alibaba-cluster": "alibaba",
    "azure": "azure",
    "google-cluster": "google",
}

TEXT_COLUMNS = {
    "Content",
    "EventTemplate",
    "Component",
    "Level",
    "Label",
    "Pid",
    "Date",
    "Time",
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    if isinstance(value, str):
        cleaned = re.sub(r"\s+", " ", value).strip()
        return cleaned or None

    if hasattr(value, "item"):
        return value.item()

    return value


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )


def read_csv_robust(path: Path) -> pd.DataFrame:
    attempts = [
        {"encoding": "utf-8"},
        {"encoding": "utf-8-sig"},
        {"encoding": "latin-1"},
    ]

    last_error: Exception | None = None

    for options in attempts:
        try:
            return pd.read_csv(
                path,
                low_memory=False,
                on_bad_lines="skip",
                **options,
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Unable to read CSV {path}: {last_error}"
    )


def detect_loghub_dataset(path: Path) -> str:
    parts = list(path.parts)

    if "loghub" in parts:
        index = parts.index("loghub")

        if index + 1 < len(parts):
            return parts[index + 1]

    return path.parent.name


def normalize_loghub_structured(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = detect_loghub_dataset(path)
    frame = read_csv_robust(path)

    required = {"Content", "EventId", "EventTemplate"}
    missing = sorted(required - set(frame.columns))

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}"
        )

    records: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        metadata = {}

        for column in frame.columns:
            if column in {
                "LineId",
                "Content",
                "EventId",
                "EventTemplate",
                "Label",
            }:
                continue

            value = clean_value(row.get(column))

            if value is not None:
                metadata[column] = value

        label = clean_value(row.get("Label"))

        record = {
            "record_type": "log_event",
            "source_family": "loghub",
            "source_dataset": dataset,
            "source_file": str(path),
            "line_id": clean_value(row.get("LineId")),
            "content": clean_value(row.get("Content")),
            "event_id": clean_value(row.get("EventId")),
            "event_template": clean_value(
                row.get("EventTemplate")
            ),
            "label": label,
            "metadata": metadata,
        }

        records.append(record)

    report = {
        "path": str(path),
        "kind": "loghub_structured",
        "dataset": dataset,
        "rows": len(frame),
        "columns": list(frame.columns),
        "sha256": sha256sum(path),
        "status": "processed",
    }

    return records, report


def normalize_templates(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = detect_loghub_dataset(path)
    frame = read_csv_robust(path)

    if "EventTemplate" not in frame.columns:
        raise ValueError(
            f"{path} has no EventTemplate column"
        )

    records: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        records.append(
            {
                "record_type": "log_template",
                "source_family": "loghub",
                "source_dataset": dataset,
                "source_file": str(path),
                "event_id": clean_value(row.get("EventId")),
                "event_template": clean_value(
                    row.get("EventTemplate")
                ),
                "occurrences": clean_value(
                    row.get("Occurrences")
                ),
            }
        )

    report = {
        "path": str(path),
        "kind": "loghub_templates",
        "dataset": dataset,
        "rows": len(frame),
        "columns": list(frame.columns),
        "sha256": sha256sum(path),
        "status": "processed",
    }

    return records, report


def normalize_metric_csv(
    path: Path,
    provider: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = read_csv_robust(path)
    frame = frame.dropna(how="all")

    records: list[dict[str, Any]] = []

    for row_index, row in frame.iterrows():
        timestamp = None
        metrics = {}
        metadata = {}

        for column in frame.columns:
            value = clean_value(row.get(column))
            lower = column.lower()

            if value is None:
                continue

            if lower in {
                "timestamp",
                "time",
                "datetime",
                "date",
            }:
                timestamp = value
                continue

            if isinstance(value, (int, float)):
                metrics[column] = value
            else:
                metadata[column] = value

        records.append(
            {
                "record_type": "metric_snapshot",
                "source_family": "cloud_trace",
                "source_dataset": provider,
                "source_file": str(path),
                "row_id": int(row_index),
                "timestamp": timestamp,
                "metrics": metrics,
                "metadata": metadata,
            }
        )

    report = {
        "path": str(path),
        "kind": "cloud_metrics",
        "dataset": provider,
        "rows": len(frame),
        "columns": list(frame.columns),
        "sha256": sha256sum(path),
        "status": "processed",
    }

    return records, report


def normalize_anomaly_file(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = read_csv_robust(path)
    frame = frame.dropna(how="all")

    split_name = path.stem.lower()
    dataset = "cloud-native-anomaly"
    records: list[dict[str, Any]] = []

    for row_index, row in frame.iterrows():
        values = {
            column: clean_value(row.get(column))
            for column in frame.columns
        }

        record_type = (
            "anomaly_groundtruth"
            if "groundtruth" in split_name
            else "anomaly_metric_snapshot"
        )

        records.append(
            {
                "record_type": record_type,
                "source_family": "cloud_native_kpi",
                "source_dataset": dataset,
                "source_file": str(path),
                "split": split_name,
                "row_id": int(row_index),
                "values": values,
            }
        )

    report = {
        "path": str(path),
        "kind": "cloud_native_anomaly",
        "dataset": dataset,
        "split": split_name,
        "rows": len(frame),
        "columns": list(frame.columns),
        "sha256": sha256sum(path),
        "status": "processed",
    }

    return records, report


def inventory_raw_log(path: Path) -> dict[str, Any]:
    line_count = 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for _ in handle:
            line_count += 1

    return {
        "path": str(path),
        "kind": "raw_log",
        "dataset": detect_loghub_dataset(path),
        "rows": line_count,
        "size_bytes": path.stat().st_size,
        "sha256": sha256sum(path),
        "status": "inventoried",
        "note": (
            "Raw log retained for future parser evaluation. "
            "Structured CSV is used for the initial corpus."
        ),
    }


def process_corpus(raw_dir: Path, output_dir: Path) -> None:
    inventory: list[dict[str, Any]] = []
    log_events: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []
    cloud_metrics: list[dict[str, Any]] = []
    anomaly_records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    files = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".log"}
    )

    LOGGER.info("Discovered %s input files", len(files))

    for path in files:
        relative = path.relative_to(raw_dir)
        LOGGER.info("Processing %s", relative)

        try:
            if path.suffix.lower() == ".log":
                inventory.append(inventory_raw_log(path))
                continue

            filename = path.name

            if filename.endswith(STRUCTURED_SUFFIX):
                records, report = normalize_loghub_structured(
                    path
                )
                log_events.extend(records)
                inventory.append(report)
                continue

            if filename.endswith(TEMPLATE_SUFFIXES):
                records, report = normalize_templates(path)
                templates.extend(records)
                inventory.append(report)
                continue

            relative_parts = relative.parts

            if (
                relative_parts
                and relative_parts[0] in METRIC_SOURCES
            ):
                provider = METRIC_SOURCES[relative_parts[0]]
                records, report = normalize_metric_csv(
                    path,
                    provider,
                )
                cloud_metrics.extend(records)
                inventory.append(report)
                continue

            if (
                relative_parts
                and relative_parts[0]
                == "cloud-native-anomaly"
            ):
                records, report = normalize_anomaly_file(
                    path
                )
                anomaly_records.extend(records)
                inventory.append(report)
                continue

            inventory.append(
                {
                    "path": str(path),
                    "kind": "unclassified_csv",
                    "rows": None,
                    "sha256": sha256sum(path),
                    "status": "skipped",
                }
            )

        except Exception as exc:
            LOGGER.exception("Failed to process %s", path)
            errors.append(
                {
                    "path": str(path),
                    "error": str(exc),
                }
            )

    write_jsonl(
        log_events,
        output_dir / "logs" / "events.jsonl",
    )
    write_jsonl(
        templates,
        output_dir / "logs" / "templates.jsonl",
    )
    write_jsonl(
        cloud_metrics,
        output_dir / "metrics" / "snapshots.jsonl",
    )
    write_jsonl(
        anomaly_records,
        output_dir / "anomalies" / "records.jsonl",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "raw_directory": str(raw_dir),
        "output_directory": str(output_dir),
        "discovered_files": len(files),
        "log_events": len(log_events),
        "log_templates": len(templates),
        "cloud_metric_snapshots": len(cloud_metrics),
        "anomaly_records": len(anomaly_records),
        "errors": errors,
        "inventory": inventory,
    }

    report_path = output_dir / "preparation_report.json"

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    LOGGER.info("Preparation complete")
    LOGGER.info("Log events: %s", len(log_events))
    LOGGER.info("Templates: %s", len(templates))
    LOGGER.info(
        "Cloud metric snapshots: %s",
        len(cloud_metrics),
    )
    LOGGER.info(
        "Anomaly records: %s",
        len(anomaly_records),
    )
    LOGGER.info("Errors: %s", len(errors))
    LOGGER.info("Report: %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and normalize OSHA raw datasets."
        )
    )

    parser.add_argument(
        "--raw-dir",
        default="orchestration-ai-data/raw",
        help="Directory containing raw datasets.",
    )

    parser.add_argument(
        "--output-dir",
        default="orchestration-ai-data/processed",
        help="Directory for normalized outputs.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=(
            "%(asctime)s | %(levelname)s | %(message)s"
        ),
    )

    raw_dir = Path(args.raw_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset directory does not exist: {raw_dir}"
        )

    if raw_dir == output_dir:
        raise ValueError(
            "Raw and processed directories must be different."
        )

    process_corpus(raw_dir, output_dir)


if __name__ == "__main__":
    main()
