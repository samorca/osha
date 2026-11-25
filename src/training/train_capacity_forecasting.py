from __future__ import annotations

import argparse
import json
import math

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from sklearn.linear_model import Ridge

from src.data.checksums import sha256_file
from src.forecasting.trained import (
    CAPACITY_FORECAST_MODEL_NAME,
    CAPACITY_FORECAST_MODEL_TYPE,
    CAPACITY_FORECAST_SCHEMA_VERSION,
)
from src.training.artifacts import ModelArtifactManager
from src.training.config import TrainingConfig, TrainingPathConfig
from src.training.schema import (
    CheckpointState,
    EvaluationResult,
    TrainingCheckpoint,
    TrainingMetric,
    TrainingTask,
)


INTERVAL_SECONDS = 300
LOOKBACK_CANDIDATES = (3, 6, 12, 24)
RIDGE_ALPHA_CANDIDATES = (0.01, 0.1, 1.0, 10.0, 100.0)
BLEND_CANDIDATES = tuple(index / 20 for index in range(21))
METRIC_FIELDS: dict[str, dict[str, tuple[str, float]]] = {
    "alibaba": {
        "cpu": ("cpu_util_percent", 1.0),
        "memory": ("mem_util_percent", 1.0),
        "disk": ("disk_io_percent", 1.0),
    },
    "google": {
        "cpu": ("avg_cpu", 100.0),
        "memory": ("avg_mem", 100.0),
    },
}


def load_capacity_series(
    path: str | Path,
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Load percentage-labelled, 5-minute series from prepared snapshots."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Forecasting training data not found: {source_path}")

    grouped: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with source_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {source_path}:{line_number}."
                ) from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"Expected a JSON object at line {line_number}.")
            dataset = record.get("source_dataset")
            if dataset not in METRIC_FIELDS:
                continue
            source_file = str(record.get("source_file", ""))
            if "300_seconds" not in Path(source_file).name:
                raise ValueError(
                    f"Expected a 300-second grouped source at line {line_number}."
                )
            row_id = record.get("row_id")
            if isinstance(row_id, bool):
                raise ValueError(f"Invalid row_id at line {line_number}.")
            try:
                index = int(row_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row_id at line {line_number}.") from exc
            if index < 0:
                raise ValueError(f"Invalid row_id at line {line_number}.")

            metrics = record.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            for metric, (field, scale) in METRIC_FIELDS[dataset].items():
                if field not in metrics:
                    continue
                try:
                    value = float(metrics[field]) * scale
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid {field} value at line {line_number}."
                    ) from exc
                if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                    raise ValueError(
                        f"{field} at line {line_number} is not a percentage."
                    )
                previous = grouped[metric][dataset].get(index)
                if previous is not None:
                    raise ValueError(
                        f"Duplicate {dataset} row_id {index} for {metric}."
                    )
                grouped[metric][dataset][index] = value

    output: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for metric, sources in grouped.items():
        output[metric] = {
            dataset: sorted(rows.items())
            for dataset, rows in sources.items()
            if rows
        }
    if not output:
        raise ValueError("No supported capacity telemetry series were found.")
    return output


def _windows(
    rows: Sequence[tuple[int, float]],
    lookback: int,
) -> tuple[list[list[float]], list[float], list[list[float]], list[float], list[list[float]], list[float]]:
    if len(rows) < lookback + 5:
        return [], [], [], [], [], []

    train_end = int(len(rows) * 0.6)
    validation_end = int(len(rows) * 0.8)
    train_x: list[list[float]] = []
    train_y: list[float] = []
    validation_x: list[list[float]] = []
    validation_y: list[float] = []
    test_x: list[list[float]] = []
    test_y: list[float] = []

    for target in range(lookback, len(rows)):
        window = rows[target - lookback : target + 1]
        if any(
            right[0] - left[0] != 1
            for left, right in zip(window, window[1:])
        ):
            continue
        features = [value for _, value in reversed(window[:-1])]
        target_value = window[-1][1]
        if target < train_end:
            train_x.append(features)
            train_y.append(target_value)
        elif target < validation_end:
            validation_x.append(features)
            validation_y.append(target_value)
        else:
            test_x.append(features)
            test_y.append(target_value)

    return (
        train_x,
        train_y,
        validation_x,
        validation_y,
        test_x,
        test_y,
    )


def _fit_metric(
    metric: str,
    source_series: Mapping[str, Sequence[tuple[int, float]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: list[tuple[float, int, float, float]] = []
    for lookback in LOOKBACK_CANDIDATES:
        train_x: list[list[float]] = []
        train_y: list[float] = []
        validation_x: list[list[float]] = []
        validation_y: list[float] = []
        for rows in source_series.values():
            (
                source_train_x,
                source_train_y,
                source_validation_x,
                source_validation_y,
                _,
                _,
            ) = _windows(rows, lookback)
            train_x.extend(source_train_x)
            train_y.extend(source_train_y)
            validation_x.extend(source_validation_x)
            validation_y.extend(source_validation_y)

        if not train_x or not validation_x:
            continue
        feature_matrix = np.asarray(train_x, dtype=np.float64)
        target_vector = np.asarray(train_y, dtype=np.float64)
        validation_matrix = np.asarray(validation_x, dtype=np.float64)
        validation_targets = np.asarray(validation_y, dtype=np.float64)

        for alpha in RIDGE_ALPHA_CANDIDATES:
            model = Ridge(alpha=alpha).fit(feature_matrix, target_vector)
            raw_predictions = model.predict(validation_matrix)
            persistence_predictions = validation_matrix[:, 0]
            for blend in BLEND_CANDIDATES:
                predictions = (
                    blend * raw_predictions
                    + (1.0 - blend) * persistence_predictions
                )
                mae = float(np.mean(np.abs(predictions - validation_targets)))
                candidates.append((mae, lookback, alpha, blend))

    if not candidates:
        raise ValueError(f"Not enough contiguous data to train {metric} model.")
    validation_mae, lookback, alpha, blend = min(
        candidates,
        key=lambda candidate: (
            candidate[0],
            candidate[1],
            candidate[2],
            candidate[3],
        ),
    )

    train_x: list[list[float]] = []
    train_y: list[float] = []
    validation_x: list[list[float]] = []
    validation_y: list[float] = []
    test_x: list[list[float]] = []
    test_y: list[float] = []
    source_counts: dict[str, dict[str, int]] = {}
    for dataset, rows in source_series.items():
        (
            source_train_x,
            source_train_y,
            source_validation_x,
            source_validation_y,
            source_test_x,
            source_test_y,
        ) = _windows(rows, lookback)
        train_x.extend(source_train_x)
        train_y.extend(source_train_y)
        validation_x.extend(source_validation_x)
        validation_y.extend(source_validation_y)
        test_x.extend(source_test_x)
        test_y.extend(source_test_y)
        source_counts[dataset] = {
            "rows": len(rows),
            "train_windows": len(source_train_x),
            "validation_windows": len(source_validation_x),
            "test_windows": len(source_test_x),
        }
    if not test_x:
        raise ValueError(f"Not enough temporal holdout data to evaluate {metric}.")

    validation_model = Ridge(alpha=alpha).fit(
        np.asarray(train_x, dtype=np.float64),
        np.asarray(train_y, dtype=np.float64),
    )
    validation_targets = np.asarray(validation_y, dtype=np.float64)
    validation_matrix = np.asarray(validation_x, dtype=np.float64)
    validation_predictions = (
        blend * validation_model.predict(validation_matrix)
        + (1.0 - blend) * validation_matrix[:, 0]
    )
    residuals = validation_targets - validation_predictions
    residual_std = float(np.std(residuals, ddof=1))
    total_variance = float(
        np.sum((validation_targets - np.mean(validation_targets)) ** 2)
    )
    residual_variance = float(np.sum(residuals**2))
    validation_r_squared = (
        1.0 - residual_variance / total_variance
        if total_variance > 0
        else 0.0
    )

    final_x = train_x + validation_x
    final_y = train_y + validation_y
    final_model = Ridge(alpha=alpha).fit(
        np.asarray(final_x, dtype=np.float64),
        np.asarray(final_y, dtype=np.float64),
    )
    test_matrix = np.asarray(test_x, dtype=np.float64)
    test_targets = np.asarray(test_y, dtype=np.float64)
    test_predictions = (
        blend * final_model.predict(test_matrix)
        + (1.0 - blend) * test_matrix[:, 0]
    )
    baseline_predictions = test_matrix[:, 0]
    test_mae = float(np.mean(np.abs(test_predictions - test_targets)))
    test_rmse = float(np.sqrt(np.mean((test_predictions - test_targets) ** 2)))
    baseline_test_mae = float(
        np.mean(np.abs(baseline_predictions - test_targets))
    )

    coefficients = np.asarray(final_model.coef_, dtype=np.float64) * blend
    coefficients[0] += 1.0 - blend
    model_parameters = {
        "coefficients": coefficients.tolist(),
        "intercept": float(final_model.intercept_) * blend,
        "residual_std": residual_std,
        "validation_r_squared": validation_r_squared,
        "validation_mae": validation_mae,
        "lookback_points": lookback,
        "ridge_alpha": alpha,
        "autoregressive_blend": blend,
        "training_samples": len(final_y),
    }
    metrics = {
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "baseline_test_mae": baseline_test_mae,
        "test_samples": len(test_y),
        "validation_mae": validation_mae,
        "lookback_points": lookback,
        "ridge_alpha": alpha,
        "autoregressive_blend": blend,
        "source_counts": source_counts,
    }
    return model_parameters, metrics


def train_capacity_model(
    series: Mapping[str, Mapping[str, Sequence[tuple[int, float]]]],
    *,
    dataset_checksum: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required_metrics = {"cpu", "memory", "disk"}
    missing_metrics = required_metrics.difference(series)
    if missing_metrics:
        raise ValueError(
            "Capacity model training data is missing required metrics: "
            + ", ".join(sorted(missing_metrics))
        )

    models: dict[str, Any] = {}
    evaluation: dict[str, Any] = {}
    for metric in sorted(required_metrics):
        model_parameters, metric_evaluation = _fit_metric(
            metric,
            series[metric],
        )
        models[metric] = model_parameters
        evaluation[metric] = metric_evaluation

    passed = all(
        item["test_mae"] < item["baseline_test_mae"]
        for item in evaluation.values()
    )
    payload = {
        "model_type": CAPACITY_FORECAST_MODEL_TYPE,
        "schema_version": CAPACITY_FORECAST_SCHEMA_VERSION,
        "model_name": CAPACITY_FORECAST_MODEL_NAME,
        "interval_seconds": INTERVAL_SECONDS,
        "trained_metrics": sorted(models),
        "metrics": models,
        "training_data": {
            "dataset_id": "osha_processed_metric_snapshots_v1",
            "sha256": dataset_checksum,
            "source_datasets": sorted(
                {
                    source
                    for metric_sources in series.values()
                    for source in metric_sources
                }
            ),
            "split": {
                "training_fraction": 0.6,
                "validation_fraction": 0.2,
                "test_fraction": 0.2,
                "temporal_order_preserved": True,
                "row_id_interval_seconds": INTERVAL_SECONDS,
            },
        },
    }
    report = {
        "evaluation_split": "test",
        "passed": passed,
        "metrics": evaluation,
        "trained_metrics": sorted(models),
        "dataset_checksum": dataset_checksum,
    }
    return payload, report


def train_and_promote_capacity_model(
    *,
    data_path: str | Path,
    version: str,
    checkpoint_directory: str | Path = "checkpoints",
    artifact_directory: str | Path = "artifacts",
    model_directory: str | Path = "models",
) -> dict[str, Any]:
    dataset_path = Path(data_path).expanduser().resolve()
    checksum = sha256_file(dataset_path)
    series = load_capacity_series(dataset_path)
    payload, report = train_capacity_model(
        series,
        dataset_checksum=checksum,
    )
    if report["passed"] is not True:
        raise ValueError(
            "Capacity model failed held-out evaluation against the persistence "
            "baseline; promotion was blocked."
        )

    run_id = uuid4().hex
    config = TrainingConfig(
        paths=TrainingPathConfig(
            checkpoint_directory=str(checkpoint_directory),
            artifact_directory=str(artifact_directory),
            model_directory=str(model_directory),
        )
    )
    manager = ModelArtifactManager(config=config)
    roots = config.paths.resolved()
    checkpoint_path = (
        Path(roots["checkpoints"])
        / TrainingTask.CAPACITY_FORECASTING.value
        / CAPACITY_FORECAST_MODEL_NAME
        / run_id
        / "model-state.json"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(checkpoint_path)

    metric_records: list[TrainingMetric] = []
    thresholds: dict[str, float] = {}
    for metric, result in report["metrics"].items():
        metric_records.extend(
            (
                TrainingMetric(
                    name=f"{metric}_test_mae",
                    value=result["test_mae"],
                    split="test",
                ),
                TrainingMetric(
                    name=f"{metric}_baseline_test_mae",
                    value=result["baseline_test_mae"],
                    split="test",
                ),
            )
        )
        thresholds[f"{metric}_test_mae"] = result["baseline_test_mae"]

    evaluation = EvaluationResult(
        run_id=run_id,
        model_name=CAPACITY_FORECAST_MODEL_NAME,
        split="test",
        metrics=metric_records,
        passed=True,
        thresholds=thresholds,
        attributes={
            "evaluation_protocol": "temporal_train_validation_test",
            "baseline": "last_observation_persistence",
            "metrics": report["metrics"],
        },
    ).finalize()
    weighted_test_mae = sum(
        result["test_mae"] * result["test_samples"]
        for result in report["metrics"].values()
    ) / sum(result["test_samples"] for result in report["metrics"].values())
    checkpoint = TrainingCheckpoint(
        run_id=run_id,
        model_name=CAPACITY_FORECAST_MODEL_NAME,
        path=str(checkpoint_path),
        state=CheckpointState.BEST.value,
        score=weighted_test_mae,
        score_name="weighted_test_mae",
        checksum=sha256_file(checkpoint_path),
        attributes={
            "training_task": TrainingTask.CAPACITY_FORECASTING.value,
            "dataset_id": "osha_processed_metric_snapshots_v1",
            "dataset_sha256": checksum,
            "training_metrics": report["metrics"],
        },
    ).finalize()

    artifact = manager.create_candidate(
        checkpoint=checkpoint,
        task=TrainingTask.CAPACITY_FORECASTING.value,
        version=version,
        evaluation=evaluation,
    )
    promoted = manager.promote(artifact, evaluation=evaluation)
    report_path = (
        Path(roots["logs"])
        / TrainingTask.CAPACITY_FORECASTING.value
        / f"{run_id}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                **report,
                "run_id": run_id,
                "artifact_id": promoted.artifact_id,
                "promoted_model_path": promoted.path,
                "model_checksum": promoted.checksum,
                "version": promoted.version,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "version": promoted.version,
        "artifact_id": promoted.artifact_id,
        "model_path": promoted.path,
        "checksum": promoted.checksum,
        "report_path": str(report_path),
        "evaluation": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and promote the OSHA capacity model."
    )
    parser.add_argument(
        "--data",
        default="data/processed/metrics/snapshots.jsonl",
        help="Processed 5-minute metric snapshot JSONL file.",
    )
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--models", default="models")
    arguments = parser.parse_args()
    result = train_and_promote_capacity_model(
        data_path=arguments.data,
        version=arguments.version,
        checkpoint_directory=arguments.checkpoints,
        artifact_directory=arguments.artifacts,
        model_directory=arguments.models,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
