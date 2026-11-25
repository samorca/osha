from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from src.data.checksums import sha256_file
from src.training.artifacts import (
    ModelArtifactManager,
    artifact_directory,
    promoted_model_directory,
)
from src.training.config import (
    TrainingPathConfig,
    from_environment,
)
from src.training.schema import (
    CheckpointState,
    EvaluationResult,
    TrainingCheckpoint,
    TrainingMetric,
    TrainingTask,
)
from src.training.train_capacity_forecasting import (
    load_capacity_series,
    train_capacity_model,
)


MODEL_SPECS = {
    TrainingTask.EVENT.value: "infrastructure-event-model",
    TrainingTask.ANOMALY.value: "infrastructure-anomaly-model",
    TrainingTask.INCIDENT_ROOT_CAUSE.value: "incident-root-cause-v1",
    TrainingTask.DECISION_POLICY_RISK.value: "decision-risk-v1",
    TrainingTask.REMEDIATION_RECOMMENDATION.value: "remediation-ranking-v1",
    TrainingTask.CAPACITY_FORECASTING.value: "osha-capacity-autoregressive",
}


@dataclass(frozen=True)
class PromotionInput:
    task: str
    model_name: str
    checkpoint_path: Path
    run_id: str
    dataset_id: str
    metrics: Mapping[str, float]
    test_samples: int
    score_name: str
    score: float
    report_path: str


def _read_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read training report: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Training report root must be an object: {path}")
    return dict(value)


def _test_metrics(value: Any, *, report_path: Path) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Training report has no test metrics: {report_path}")
    metrics: dict[str, float] = {}
    for name, raw_value in value.items():
        if raw_value is None or isinstance(raw_value, bool):
            continue
        if not isinstance(raw_value, (int, float)):
            continue
        numeric_value = float(raw_value)
        if not math.isfinite(numeric_value):
            raise ValueError(
                f"Training report contains a non-finite test metric: {name}"
            )
        metrics[str(name)] = numeric_value
    if not metrics:
        raise ValueError(f"Training report contains no numeric test metrics: {report_path}")
    return metrics


def _test_sample_count(value: Any, *, report_path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"Training report must contain at least one test sample: {report_path}"
        )
    return value


def _promotion_input(
    *,
    task: str,
    checkpoint_path: Path,
    report_path: Path,
    run_id: str,
    dataset_id: str,
    test_values: Any,
    test_sample_count: Any,
    preferred_score: str,
) -> PromotionInput:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint_path}")
    report_path = report_path.resolve()
    metrics = _test_metrics(test_values, report_path=report_path)
    sample_count = _test_sample_count(test_sample_count, report_path=report_path)
    score_name = preferred_score if preferred_score in metrics else next(iter(metrics))
    return PromotionInput(
        task=task,
        model_name=MODEL_SPECS[task],
        checkpoint_path=checkpoint_path.resolve(),
        run_id=run_id,
        dataset_id=dataset_id,
        metrics=metrics,
        test_samples=sample_count,
        score_name=score_name,
        score=metrics[score_name],
        report_path=str(report_path),
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _training_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    paths = {
        "PREPARED_DATA_DIRECTORY": args.data_root,
        "CHECKPOINT_DIRECTORY": args.checkpoints,
        "ARTIFACT_DIRECTORY": args.artifacts,
        "MODEL_DIRECTORY": args.models,
        "LOG_DIRECTORY": args.logs,
    }
    for name, value in paths.items():
        environment[f"OSHA_TRAINING_{name}"] = str(
            Path(value).expanduser().resolve()
        )
    return environment


def _run_module(
    module: str,
    arguments: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> None:
    command = [sys.executable, "-m", module, *arguments]
    subprocess.run(command, cwd=cwd, env=dict(environment), check=True)


def _capacity_promotion_input(
    *,
    data_path: Path,
    checkpoint_root: Path,
    run_id: str,
) -> PromotionInput:
    payload, report = train_capacity_model(
        load_capacity_series(data_path),
        dataset_checksum=sha256_file(data_path),
    )
    if report.get("passed") is not True:
        raise ValueError("Capacity model failed its held-out persistence evaluation.")
    test_samples = 0
    metrics: dict[str, float] = {}
    weighted_error = 0.0
    for metric, result in report["metrics"].items():
        count = _test_sample_count(result.get("test_samples"), report_path=data_path)
        test_samples += count
        test_mae = float(result["test_mae"])
        baseline_mae = float(result["baseline_test_mae"])
        if not math.isfinite(test_mae) or not math.isfinite(baseline_mae):
            raise ValueError(f"Capacity evaluation for {metric} is non-finite.")
        if test_mae >= baseline_mae:
            raise ValueError(
                f"Capacity model failed to beat persistence for {metric}; "
                "promotion blocked."
            )
        metrics[f"{metric}_test_mae"] = test_mae
        metrics[f"{metric}_baseline_test_mae"] = baseline_mae
        weighted_error += test_mae * count

    checkpoint_path = (
        checkpoint_root
        / TrainingTask.CAPACITY_FORECASTING.value
        / MODEL_SPECS[TrainingTask.CAPACITY_FORECASTING.value]
        / run_id
        / "model-state.json"
    )
    _write_json_atomic(checkpoint_path, payload)
    report_path = checkpoint_path.with_name("evaluation.json")
    _write_json_atomic(report_path, report)
    return PromotionInput(
        task=TrainingTask.CAPACITY_FORECASTING.value,
        model_name=MODEL_SPECS[TrainingTask.CAPACITY_FORECASTING.value],
        checkpoint_path=checkpoint_path,
        run_id=run_id,
        dataset_id="osha_processed_metric_snapshots_v1",
        metrics=metrics,
        test_samples=test_samples,
        score_name="weighted_test_mae",
        score=weighted_error / test_samples,
        report_path=str(report_path.resolve()),
    )


def _prepare_inputs(args: argparse.Namespace) -> dict[str, Path]:
    data_root = Path(args.data_root).expanduser().resolve()
    inputs = {
        "data_root": data_root,
        "event_data": Path(args.event_data or data_root / "processed/logs/events.jsonl")
        .expanduser()
        .resolve(),
        "anomaly_data": Path(
            args.anomaly_data or data_root / "processed/anomalies/records.jsonl"
        )
        .expanduser()
        .resolve(),
        "capacity_data": Path(
            args.capacity_data or data_root / "processed/metrics/snapshots.jsonl"
        )
        .expanduser()
        .resolve(),
    }
    if not data_root.is_dir():
        raise FileNotFoundError(f"Training data directory was not found: {data_root}")
    for name in ("event_data", "anomaly_data", "capacity_data"):
        if not inputs[name].is_file():
            raise FileNotFoundError(f"{name} input was not found: {inputs[name]}")
    return inputs


def _new_version() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


def train_all(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _prepare_inputs(args)
    cwd = Path.cwd().resolve()
    run_id = uuid4().hex
    version = args.version or _new_version()
    if not version or len(version) > 128 or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in version
    ):
        raise ValueError("Model version must be 1-128 safe filename characters.")

    base_config = from_environment()
    config = replace(
        base_config,
        paths=TrainingPathConfig(
            checkpoint_directory=args.checkpoints,
            artifact_directory=args.artifacts,
            model_directory=args.models,
            log_directory=args.logs,
            prepared_data_directory=str(inputs["data_root"]),
        ),
    )
    config.validate()
    roots = config.paths.resolved()
    checkpoint_root = Path(roots["checkpoints"])
    artifact_root = Path(roots["artifacts"])
    model_root = Path(roots["models"])
    log_root = Path(roots["logs"])
    manager = ModelArtifactManager(config=config)

    for task, model_name in MODEL_SPECS.items():
        candidate_path = artifact_directory(
            artifact_root,
            task=task,
            model_name=model_name,
            version=version,
        )
        promoted_path = promoted_model_directory(
            model_root,
            task=task,
            model_name=model_name,
            version=version,
        )
        if candidate_path.exists() or promoted_path.exists():
            raise FileExistsError(
                f"Version {version} already exists for {task}; choose a new version."
            )

    run_artifacts = artifact_root / "training-runs" / run_id
    event_output = run_artifacts / "event"
    anomaly_output = run_artifacts / "anomaly"
    operational_output = run_artifacts / "operational"
    training_environment = _training_environment(args)
    promotions: list[PromotionInput] = []

    _run_module(
        "src.training.train_events",
        [
            "--data",
            str(inputs["event_data"]),
            "--output",
            str(event_output),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--accelerator",
            args.accelerator,
            "--num-workers",
            str(args.num_workers),
        ],
        cwd=cwd,
        environment=training_environment,
    )
    event_report_path = event_output / "training_report.json"
    event_report = _read_report(event_report_path)
    event_run = event_report.get("run")
    event_run_id = (
        str(event_run.get("run_id"))
        if isinstance(event_run, Mapping) and event_run.get("run_id")
        else run_id
    )
    event_records = event_report.get("records")
    if not isinstance(event_records, Mapping):
        raise ValueError(f"Event report has no record counts: {event_report_path}")
    promotions.append(
        _promotion_input(
            task=TrainingTask.EVENT.value,
            checkpoint_path=event_output / "best.pt",
            report_path=event_report_path,
            run_id=event_run_id,
            dataset_id="osha-infrastructure-events",
            test_values=event_report.get("test"),
            test_sample_count=event_records.get("test"),
            preferred_score="loss",
        )
    )

    _run_module(
        "src.training.train_anomalies",
        [
            "--data",
            str(inputs["anomaly_data"]),
            "--output",
            str(anomaly_output),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--accelerator",
            args.accelerator,
            "--num-workers",
            str(args.num_workers),
        ],
        cwd=cwd,
        environment=training_environment,
    )
    anomaly_report_path = anomaly_output / "training_report.json"
    anomaly_report = _read_report(anomaly_report_path)
    anomaly_run = anomaly_report.get("run")
    anomaly_run_id = (
        str(anomaly_run.get("run_id"))
        if isinstance(anomaly_run, Mapping) and anomaly_run.get("run_id")
        else run_id
    )
    anomaly_records = anomaly_report.get("records")
    if not isinstance(anomaly_records, Mapping):
        raise ValueError(f"Anomaly report has no record counts: {anomaly_report_path}")
    promotions.append(
        _promotion_input(
            task=TrainingTask.ANOMALY.value,
            checkpoint_path=anomaly_output / "best.pt",
            report_path=anomaly_report_path,
            run_id=anomaly_run_id,
            dataset_id="osha-infrastructure-anomalies",
            test_values=anomaly_report.get("test"),
            test_sample_count=anomaly_records.get("test"),
            preferred_score="f1",
        )
    )

    operational_accelerator = (
        "cuda" if args.accelerator == "rocm" else args.accelerator
    )
    _run_module(
        "src.training.operational_models",
        [
            "--data-root",
            str(inputs["data_root"]),
            "--output-root",
            str(operational_output),
            "--checkpoint-root",
            str(checkpoint_root),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--accelerator",
            operational_accelerator,
        ],
        cwd=cwd,
        environment=training_environment,
    )
    operational_reports = (
        (
            TrainingTask.INCIDENT_ROOT_CAUSE.value,
            "incident-root-cause-v1",
            "root_cause_accuracy",
        ),
        (
            TrainingTask.DECISION_POLICY_RISK.value,
            "decision-risk-v1",
            "risk_accuracy",
        ),
        (
            TrainingTask.REMEDIATION_RECOMMENDATION.value,
            "remediation-ranking-v1",
            "accuracy",
        ),
    )
    for task, directory, preferred_score in operational_reports:
        report_path = operational_output / directory / "training_report.json"
        report = _read_report(report_path)
        promotions.append(
            _promotion_input(
                task=task,
                checkpoint_path=operational_output / directory / "best.pt",
                report_path=report_path,
                run_id=run_id,
                dataset_id="operational-models-v1",
                test_values=report.get("evaluation"),
                test_sample_count=report.get("test_samples"),
                preferred_score=preferred_score,
            )
        )

    promotions.append(
        _capacity_promotion_input(
            data_path=inputs["capacity_data"],
            checkpoint_root=checkpoint_root,
            run_id=run_id,
        )
    )

    candidates: list[tuple[PromotionInput, Any, EvaluationResult]] = []
    for item in promotions:
        evaluation = EvaluationResult(
            run_id=item.run_id,
            model_name=item.model_name,
            split="test",
            metrics=[
                TrainingMetric(name=name, value=value, split="test")
                for name, value in item.metrics.items()
            ],
            passed=True,
            attributes={
                "test_samples": item.test_samples,
                "evaluation_policy": "finite_metrics_and_nonempty_held_out_test",
                "source_report": item.report_path,
            },
        ).finalize()
        checkpoint = TrainingCheckpoint(
            run_id=item.run_id,
            model_name=item.model_name,
            path=str(item.checkpoint_path),
            state=CheckpointState.BEST.value,
            score=item.score,
            score_name=item.score_name,
            checksum=sha256_file(item.checkpoint_path),
            attributes={
                "training_task": item.task,
                "dataset_id": item.dataset_id,
                "suite_run_id": run_id,
            },
        ).finalize()
        candidate = manager.create_candidate(
            checkpoint=checkpoint,
            task=item.task,
            version=version,
            evaluation=evaluation,
        )
        candidates.append((item, candidate, evaluation))

    promoted_models = {}
    for item, candidate, evaluation in candidates:
        promoted = manager.promote(candidate, evaluation=evaluation)
        promoted_models[item.task] = {
            "model_name": item.model_name,
            "version": version,
            "path": promoted.path,
            "checksum": promoted.checksum,
            "test_samples": item.test_samples,
            "metrics": dict(item.metrics),
        }

    summary = {
        "status": "promoted",
        "suite_run_id": run_id,
        "version": version,
        "data": {name: str(path) for name, path in inputs.items()},
        "models": promoted_models,
    }
    report_path = log_root / "full-model-suite" / f"{run_id}.json"
    _write_json_atomic(report_path, summary)
    summary["report_path"] = str(report_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train, test, and promote all six OSHA model families as one suite."
        )
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--event-data")
    parser.add_argument("--anomaly-data")
    parser.add_argument("--capacity-data")
    parser.add_argument("--version")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cuda", "rocm", "cpu"),
        default="auto",
    )
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--models", default="models")
    parser.add_argument("--logs", default="logs/training")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    return args


def main() -> None:
    train_all(parse_args())


if __name__ == "__main__":
    main()
