from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd


CORE_COLUMNS = {
    "LineId",
    "Content",
    "EventId",
    "EventTemplate",
}

SENSITIVE_PATTERNS = [
    re.compile(
        r"(?i)(password|passwd|secret|token|api[_-]?key)"
        r"\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(
        r"(?i)authorization:\s*bearer\s+[A-Za-z0-9._~+/=-]+"
    ),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
        r".*?"
        r"-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
]

EXCLUDED_DIRECTORIES = {
    "features",
    "labels",
    "processed",
    "synthetic",
    ".git",
    ".venv",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def require_parquet_engine() -> None:
    if not any(
        importlib.util.find_spec(engine) is not None
        for engine in ("pyarrow", "fastparquet")
    ):
        raise RuntimeError(
            "Writing Parquet output requires pyarrow or fastparquet. "
            "Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        )


def hash_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def redact_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value)

    for pattern in SENSITIVE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)

    return text


def discover_structured_files(
    data_root: Path,
) -> list[Path]:
    files: list[Path] = []

    for path in data_root.rglob("*_structured.csv"):
        if any(
            part in EXCLUDED_DIRECTORIES
            for part in path.parts
        ):
            continue

        files.append(path)

    return sorted(files)


def validate_dataframe(
    dataframe: pd.DataFrame,
    source_file: Path,
) -> list[str]:
    errors: list[str] = []

    missing_columns = CORE_COLUMNS - set(
        dataframe.columns
    )

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if dataframe.empty:
        errors.append("Dataset contains no records")

    for column in CORE_COLUMNS & set(dataframe.columns):
        missing_count = int(
            dataframe[column].isna().sum()
        )

        if missing_count:
            errors.append(
                f"{column} contains "
                f"{missing_count} missing values"
            )

    if errors:
        logging.error(
            "Validation failed for %s: %s",
            source_file,
            errors,
        )

    return errors


def _parse_timestamps(values: pd.Series) -> pd.Series:
    raw_values = values.astype(str)
    time_only = raw_values.str.fullmatch(
        r"\d{1,2}:\d{2}:\d{2}(?:\.\d+)?"
    )
    if time_only.any():
        values = values.astype(object)
        values.loc[time_only] = "2000-01-01 " + raw_values.loc[time_only]
    parsed = pd.to_datetime(
        values,
        format="mixed",
        errors="coerce",
        utc=True,
    )
    missing_year = (
        parsed.dt.year.eq(1)
        & ~raw_values.str.contains(
            r"(?<!\d)\d{4}(?!\d)",
            regex=True,
        )
    )
    if missing_year.any():
        parsed.loc[missing_year] = parsed.loc[missing_year].map(
            lambda value: value.replace(year=2000)
        )

    return parsed.astype("datetime64[ns, UTC]")


def build_timestamp(
    dataframe: pd.DataFrame,
) -> pd.Series:
    if {"Month", "Date", "Time"}.issubset(dataframe.columns):
        combined = (
            dataframe["Month"].astype(str)
            + " "
            + dataframe["Date"].astype(str)
            + " "
            + dataframe["Time"].astype(str)
        )
        return _parse_timestamps(combined)

    if {"Date", "Day", "Time"}.issubset(dataframe.columns):
        combined = (
            dataframe["Date"].astype(str)
            + " "
            + dataframe["Day"].astype(str)
            + " "
            + dataframe["Time"].astype(str)
        )
        return _parse_timestamps(combined)

    if {"Date", "Time"}.issubset(dataframe.columns):
        combined = (
            dataframe["Date"].astype(str)
            + " "
            + dataframe["Time"].astype(str)
        )

        return _parse_timestamps(combined)

    if "Timestamp" in dataframe.columns:
        return _parse_timestamps(dataframe["Timestamp"])

    if "Time" in dataframe.columns:
        return _parse_timestamps(dataframe["Time"])

    return pd.Series(
        [pd.NaT] * len(dataframe),
        index=dataframe.index,
        dtype="datetime64[ns, UTC]",
    )


def normalize_dataframe(
    dataframe: pd.DataFrame,
    dataset_name: str,
    source_file: Path,
) -> pd.DataFrame:
    normalized = pd.DataFrame()

    normalized["dataset"] = dataset_name
    normalized["source_file"] = str(source_file)
    normalized["line_id"] = dataframe[
        "LineId"
    ].astype(str)

    normalized["timestamp"] = build_timestamp(
        dataframe
    )

    normalized["level"] = dataframe.get(
        "Level",
        pd.Series(
            ["UNKNOWN"] * len(dataframe),
            index=dataframe.index,
        ),
    ).fillna("UNKNOWN").astype(str).str.upper()

    normalized["component"] = dataframe.get(
        "Component",
        pd.Series(
            ["unknown"] * len(dataframe),
            index=dataframe.index,
        ),
    ).fillna("unknown").astype(str)

    normalized["event_id"] = dataframe[
        "EventId"
    ].fillna("UNKNOWN").astype(str)

    normalized["event_template"] = dataframe[
        "EventTemplate"
    ].map(redact_text)

    normalized["content"] = dataframe[
        "Content"
    ].map(redact_text)

    normalized["content_hash"] = normalized[
        "content"
    ].map(hash_text)

    normalized["is_error_level"] = (
        normalized["level"].isin(
            {
                "ERROR",
                "CRITICAL",
                "FATAL",
                "ALERT",
                "EMERGENCY",
            }
        )
    )

    extra_columns = [
        column
        for column in dataframe.columns
        if column not in CORE_COLUMNS
    ]

    normalized["source_metadata"] = dataframe[
        extra_columns
    ].apply(
        lambda row: json.dumps(
            {
                key: (
                    None
                    if pd.isna(value)
                    else str(value)
                )
                for key, value in row.items()
            },
            ensure_ascii=False,
        ),
        axis=1,
    )

    normalized["record_id"] = normalized.apply(
        lambda row: hash_text(
            "|".join(
                [
                    str(row["dataset"]),
                    str(row["source_file"]),
                    str(row["line_id"]),
                    str(row["event_id"]),
                    str(row["content_hash"]),
                ]
            )
        ),
        axis=1,
    )

    return normalized


def process_file(
    source_file: Path,
    data_root: Path,
    processed_root: Path,
) -> dict[str, Any]:
    dataset_name = source_file.parent.name

    logging.info(
        "Processing %s",
        source_file,
    )

    try:
        dataframe = pd.read_csv(
            source_file,
            low_memory=False,
        )
    except Exception as error:
        return {
            "dataset": dataset_name,
            "source_file": str(source_file),
            "status": "read_failed",
            "error": str(error),
        }

    validation_errors = validate_dataframe(
        dataframe,
        source_file,
    )

    if validation_errors:
        return {
            "dataset": dataset_name,
            "source_file": str(source_file),
            "status": "validation_failed",
            "rows": len(dataframe),
            "errors": validation_errors,
        }

    normalized = normalize_dataframe(
        dataframe=dataframe,
        dataset_name=dataset_name,
        source_file=source_file.relative_to(
            data_root
        ),
    )

    duplicates = int(
        normalized.duplicated(
            subset=["record_id"]
        ).sum()
    )

    normalized = normalized.drop_duplicates(
        subset=["record_id"]
    ).reset_index(drop=True)

    output_directory = (
        processed_root
        / dataset_name.lower()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path = (
        output_directory
        / "events.parquet"
    )

    csv_path = (
        output_directory
        / "events.csv"
    )

    normalized.to_parquet(
        parquet_path,
        index=False,
    )

    normalized.to_csv(
        csv_path,
        index=False,
    )

    return {
        "dataset": dataset_name,
        "source_file": str(source_file),
        "status": "processed",
        "input_rows": len(dataframe),
        "output_rows": len(normalized),
        "duplicates_removed": duplicates,
        "missing_timestamps": int(
            normalized["timestamp"].isna().sum()
        ),
        "csv_output": str(csv_path),
        "parquet_output": str(parquet_path),
    }


def build_combined_dataset(
    processed_root: Path,
) -> Path | None:
    parquet_files = sorted(
        processed_root.glob("*/events.parquet")
    )

    if not parquet_files:
        return None

    dataframes = [
        pd.read_parquet(path)
        for path in parquet_files
    ]

    combined = pd.concat(
        dataframes,
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=["record_id"]
    ).reset_index(drop=True)

    output_path = (
        processed_root
        / "all_loghub_events.parquet"
    )

    combined.to_parquet(
        output_path,
        index=False,
    )

    combined.to_csv(
        processed_root
        / "all_loghub_events.csv",
        index=False,
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, normalize, redact, and combine "
            "OSHA Loghub datasets."
        )
    )

    parser.add_argument(
        "--data-root",
        default="dataset",
    )

    parser.add_argument(
        "--processed-root",
        default=(
            "dataset/processed"
        ),
    )

    arguments = parser.parse_args()

    configure_logging()
    require_parquet_engine()

    data_root = Path(
        arguments.data_root
    ).resolve()

    processed_root = Path(
        arguments.processed_root
    ).resolve()

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root not found: {data_root}"
        )

    processed_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = discover_structured_files(
        data_root
    )

    if not files:
        raise RuntimeError(
            "No *_structured.csv files were found."
        )

    logging.info(
        "Discovered %d structured datasets",
        len(files),
    )

    report = [
        process_file(
            source_file=path,
            data_root=data_root,
            processed_root=processed_root,
        )
        for path in files
    ]

    combined_path = build_combined_dataset(
        processed_root
    )

    report_path = (
        processed_root
        / "preparation_report.json"
    )

    report_path.write_text(
        json.dumps(
            {
                "datasets_discovered": len(files),
                "combined_output": (
                    str(combined_path)
                    if combined_path
                    else None
                ),
                "results": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    successful = sum(
        item["status"] == "processed"
        for item in report
    )

    failed = len(report) - successful

    print()
    print("=" * 60)
    print("OSHA DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Datasets discovered: {len(files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed validation/read: {failed}")
    print(f"Report: {report_path}")

    if combined_path:
        print(f"Combined dataset: {combined_path}")


if __name__ == "__main__":
    main()