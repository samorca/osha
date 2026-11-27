from __future__ import annotations

"""
Train the OSHA structured anomaly classifier.

Input records must contain a numeric feature
vector under one of:

    features
    feature_vector
    values
    vector
    x

and a binary anomaly label under one of:

    label
    target
    anomaly_label
    is_anomaly
    anomaly

The trainer provides:

    deterministic train/validation/test splits
    binary anomaly classification
    class-imbalance weighting
    validation-based checkpoint selection
    managed SHA-256 checkpoints
    final held-out test evaluation
    atomic JSON reporting

The anomaly decision threshold is not permanently
embedded into the model. Threshold calibration
belongs to analyze_anomaly_thresholds.py.
"""

import argparse
import hashlib
import json
import math
import random
import shutil

from pathlib import Path

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch

from torch import nn

from torch.optim import (
    AdamW,
)

from torch.utils.data import (
    DataLoader,
)

from src.training.anomaly_dataset import (
    ANOMALY_LABEL,
    IGNORE_INDEX,
    NORMAL_LABEL,
    AnomalyDataset,
    anomaly_dataset_from_records,
    extract_legacy_features,
    extract_legacy_label,
    load_anomaly_records,
    normalize_anomaly_label,
)

from src.training.anomaly_model import (
    InfrastructureAnomalyModel,
)

from src.training.config import (
    AcceleratorConfig,
    CheckpointConfig,
    OptimizationConfig,
    ReproducibilityConfig,
    TrainingConfig,
    from_environment,
)

from src.training.errors import (
    EmptyTrainingDatasetError,
    TrainingNumericalError,
    TrainingRunError,
)

from src.training.pipeline import (
    TrainingPipeline,
)

from src.training.schema import (
    EvaluationSplit,
    TrainingState,
    TrainingTask,
)


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Train the OSHA structured infrastructure anomaly model.")
    )

    parser.add_argument(
        "--data",
        default=("dataset/processed/anomalies.jsonl"),
        help=("Input JSONL containing numeric anomaly records."),
    )

    parser.add_argument(
        "--output",
        default=("artifacts/models/anomaly-detection-v1"),
        help=("Directory for compatibility checkpoints and the training report."),
    )

    parser.add_argument(
        "--model-name",
        default=("infrastructure-anomaly-model"),
    )

    parser.add_argument(
        "--dataset-id",
        default=("osha-infrastructure-anomalies"),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--representation-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help=(
            "Temporary evaluation threshold. "
            "Use threshold analysis before "
            "production promotion."
        ),
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--accelerator",
        choices=(
            "auto",
            "cuda",
            "rocm",
            "cpu",
        ),
        default="auto",
    )

    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
    )

    parser.add_argument(
        "--pin-memory",
        action="store_true",
    )

    parser.add_argument(
        "--keep-last",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--disable-class-weight",
        action="store_true",
        help=("Disable positive-class weighting in BCEWithLogitsLoss."),
    )

    return parser.parse_args()


# ---------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------


def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0")

    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")

    if args.hidden_dim < 1:
        raise ValueError("--hidden-dim must be >= 1")

    if args.representation_dim < 1:
        raise ValueError("--representation-dim must be >= 1")

    if not (0.0 <= args.dropout < 1.0):
        raise ValueError("--dropout must be in [0, 1)")

    if args.gradient_clip_norm <= 0:
        raise ValueError("--gradient-clip-norm must be > 0")

    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("--threshold must be in [0, 1]")

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")

    if not (0.0 <= args.validation_ratio < 1.0):
        raise ValueError("--validation-ratio must be between 0 and 1")

    if args.train_ratio + args.validation_ratio >= 1.0:
        raise ValueError("--train-ratio plus --validation-ratio must be less than 1")

    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    if args.keep_last < 1:
        raise ValueError("--keep-last must be >= 1")

    if args.max_records is not None and args.max_records < 1:
        raise ValueError("--max-records must be >= 1")


# ---------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------


def build_training_config(
    args: argparse.Namespace,
) -> TrainingConfig:
    config = TrainingConfig(
        enabled=True,
        allowed_tasks=(
            TrainingTask.EVENT.value,
            TrainingTask.ANOMALY.value,
        ),
        paths=from_environment().paths,
        reproducibility=(
            ReproducibilityConfig(
                random_seed=args.seed,
                deterministic=True,
                benchmark=False,
            )
        ),
        optimization=(
            OptimizationConfig(
                batch_size=(args.batch_size),
                epochs=args.epochs,
                learning_rate=(args.learning_rate),
                weight_decay=(args.weight_decay),
                gradient_clip_norm=(args.gradient_clip_norm),
                accumulation_steps=1,
            )
        ),
        accelerator=AcceleratorConfig(
            preferred=args.accelerator,
            allow_fallback_to_cpu=(not args.no_cpu_fallback),
            mixed_precision=False,
            device_index=(args.device_index),
        ),
        checkpoint=CheckpointConfig(
            enabled=True,
            save_every_epochs=1,
            keep_last=args.keep_last,
            keep_best=True,
            monitor_metric=("validation_loss"),
            mode="min",
            save_optimizer_state=True,
        ),
        attributes={
            "trainer": ("train_anomalies"),
        },
    )

    config.validate()

    return config


# ---------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------


def select_device(
    config: TrainingConfig,
) -> torch.device:
    preferred = config.accelerator.preferred.lower()

    fallback = config.accelerator.allow_fallback_to_cpu

    device_index = config.accelerator.device_index

    if preferred == "cpu":
        return torch.device("cpu")

    cuda_available = torch.cuda.is_available()

    if preferred == "cuda":
        if cuda_available:
            index = device_index if device_index is not None else 0

            if index >= torch.cuda.device_count():
                raise TrainingRunError(
                    "Configured CUDA device index is unavailable.",
                    details={
                        "device_index": (index),
                        "visible_devices": (torch.cuda.device_count()),
                    },
                )

            return torch.device(f"cuda:{index}")

        if fallback:
            return torch.device("cpu")

        raise TrainingRunError("CUDA was requested but is not available.")

    if preferred == "rocm":
        hip_version = getattr(
            torch.version,
            "hip",
            None,
        )

        if hip_version and cuda_available:
            index = device_index if device_index is not None else 0

            return torch.device(f"cuda:{index}")

        if fallback:
            return torch.device("cpu")

        raise TrainingRunError("ROCm was requested but is not available.")

    if cuda_available:
        index = device_index if device_index is not None else 0

        return torch.device(f"cuda:{index}")

    return torch.device("cpu")


# ---------------------------------------------------------------
# Stable record identity
# ---------------------------------------------------------------


def anomaly_record_key(
    record: Mapping[
        str,
        Any,
    ],
) -> str:
    """
    Derive a stable, leakage-aware split key.

    Explicit grouping fields take precedence.
    Otherwise the canonicalized record content is
    used as its own deterministic group.
    """

    for field in (
        "group_id",
        "incident_id",
        "sequence_id",
        "session_id",
        "trace_id",
        "window_id",
    ):
        value = record.get(field)

        if value is not None:
            text = str(value).strip()

            if text:
                return field + ":" + text

    for field in (
        "record_id",
        "example_id",
    ):
        value = record.get(field)

        if value is not None:
            text = str(value).strip()

            if text:
                return field + ":" + text

    serialized = json.dumps(
        dict(record),
        sort_keys=True,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )

    return "record:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------
# Deterministic grouped split
# ---------------------------------------------------------------


def split_records(
    records: Iterable[
        Mapping[
            str,
            Any,
        ]
    ],
    *,
    train_ratio: float,
    validation_ratio: float,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    train: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    validation: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    test: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    train_cutoff = int(train_ratio * 10000)

    validation_cutoff = int((train_ratio + validation_ratio) * 10000)

    for source_record in records:
        record = dict(source_record)

        key = anomaly_record_key(record)

        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()

        bucket = (
            int(
                digest[:8],
                16,
            )
            % 10000
        )

        if bucket < train_cutoff:
            train.append(record)

        elif bucket < validation_cutoff:
            validation.append(record)

        else:
            test.append(record)

    if not train:
        raise EmptyTrainingDatasetError(
            "Anomaly split produced an empty training partition."
        )

    return (
        train,
        validation,
        test,
    )


# ---------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------


def build_loader(
    dataset: Optional[AnomalyDataset],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> Optional[DataLoader]:
    if dataset is None:
        return None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


# ---------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------


def positive_class_weight(
    dataset: AnomalyDataset,
) -> float:
    normal_count = sum(1 for sample in dataset.samples if sample.label == NORMAL_LABEL)

    anomaly_count = sum(
        1 for sample in dataset.samples if sample.label == ANOMALY_LABEL
    )

    if anomaly_count == 0:
        raise EmptyTrainingDatasetError("Training data contains no anomaly examples.")

    if normal_count == 0:
        raise EmptyTrainingDatasetError("Training data contains no normal examples.")

    return float(normal_count) / float(anomaly_count)


# ---------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------


def binary_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float,
) -> Dict[str, int]:
    valid = targets != IGNORE_INDEX

    if int(valid.sum().item()) == 0:
        return {
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "count": 0,
        }

    predictions = (probabilities[valid] >= threshold).to(torch.long)

    expected = targets[valid].to(torch.long)

    true_positive = int(
        ((predictions == ANOMALY_LABEL) & (expected == ANOMALY_LABEL)).sum().item()
    )

    true_negative = int(
        ((predictions == NORMAL_LABEL) & (expected == NORMAL_LABEL)).sum().item()
    )

    false_positive = int(
        ((predictions == ANOMALY_LABEL) & (expected == NORMAL_LABEL)).sum().item()
    )

    false_negative = int(
        ((predictions == NORMAL_LABEL) & (expected == ANOMALY_LABEL)).sum().item()
    )

    return {
        "true_positive": (true_positive),
        "true_negative": (true_negative),
        "false_positive": (false_positive),
        "false_negative": (false_negative),
        "count": int(valid.sum().item()),
    }


def finalize_metrics(
    *,
    total_loss: float,
    batches: int,
    examples: int,
    confusion: Mapping[
        str,
        int,
    ],
    threshold: float,
) -> Dict[str, Any]:
    tp = int(confusion["true_positive"])

    tn = int(confusion["true_negative"])

    fp = int(confusion["false_positive"])

    fn = int(confusion["false_negative"])

    count = int(confusion["count"])

    accuracy = (tp + tn) / count if count else None

    precision = tp / (tp + fp) if (tp + fp) else None

    recall = tp / (tp + fn) if (tp + fn) else None

    specificity = tn / (tn + fp) if (tn + fp) else None

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision is not None and recall is not None and (precision + recall) > 0)
        else None
    )

    false_positive_rate = fp / (fp + tn) if (fp + tn) else None

    false_negative_rate = fn / (fn + tp) if (fn + tp) else None

    return {
        "loss": (total_loss / batches if batches else None),
        "example_count": (examples),
        "evaluated_count": (count),
        "batch_count": (batches),
        "threshold": (threshold),
        "accuracy": (accuracy),
        "precision": (precision),
        "recall": recall,
        "specificity": (specificity),
        "f1": f1,
        "false_positive_rate": (false_positive_rate),
        "false_negative_rate": (false_negative_rate),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------


def evaluate(
    model: InfrastructureAnomalyModel,
    loader: Optional[DataLoader],
    *,
    device: torch.device,
    criterion: nn.Module,
    threshold: float,
) -> Dict[str, Any]:
    if loader is None:
        return finalize_metrics(
            total_loss=0.0,
            batches=0,
            examples=0,
            confusion={
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0,
                "count": 0,
            },
            threshold=threshold,
        )

    model.eval()

    total_loss = 0.0
    batches = 0
    examples = 0

    confusion = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "count": 0,
    }

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(
                device,
                non_blocking=True,
            )

            targets = batch["target"].to(
                device,
                non_blocking=True,
            )

            valid = targets != IGNORE_INDEX

            if int(valid.sum().item()) == 0:
                continue

            output = model(features)

            logits = output["anomaly_logits"]

            loss = criterion(
                logits[valid],
                targets[valid].float(),
            )

            if not torch.isfinite(loss):
                raise TrainingNumericalError(
                    "Anomaly evaluation produced non-finite loss."
                )

            total_loss += float(loss.item())

            batches += 1

            examples += int(features.shape[0])

            values = binary_metrics(
                output["anomaly_probability"],
                targets,
                threshold=threshold,
            )

            for key in confusion:
                confusion[key] += values[key]

    return finalize_metrics(
        total_loss=total_loss,
        batches=batches,
        examples=examples,
        confusion=confusion,
        threshold=threshold,
    )


# ---------------------------------------------------------------
# Atomic JSON
# ---------------------------------------------------------------


def write_json_atomic(
    path: Path,
    value: Mapping[
        str,
        Any,
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(path.name + ".writing")

    try:
        temporary.write_text(
            json.dumps(
                dict(value),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary.replace(path)

    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass

        raise


# ---------------------------------------------------------------
# Compatibility checkpoint
# ---------------------------------------------------------------


def copy_checkpoint_file(
    source: str | Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(destination.name + ".copying")

    shutil.copy2(
        source,
        temporary,
    )

    temporary.replace(destination)


# ---------------------------------------------------------------
# Register evaluation
# ---------------------------------------------------------------


def register_evaluation(
    pipeline: TrainingPipeline,
    run,
    *,
    split: str,
    metrics: Mapping[
        str,
        Any,
    ],
):
    numeric = {
        str(name): float(value)
        for name, value in metrics.items()
        if (
            value is not None
            and isinstance(
                value,
                (
                    int,
                    float,
                ),
            )
            and name
            not in {
                "example_count",
                "evaluated_count",
                "batch_count",
                "true_positive",
                "true_negative",
                "false_positive",
                "false_negative",
                "threshold",
            }
            and math.isfinite(float(value))
        )
    }

    if not numeric:
        return None

    return pipeline.record_evaluation(
        run=run,
        split=split,
        metrics=numeric,
    )


# ---------------------------------------------------------------
# Show dataset summary
# ---------------------------------------------------------------


def print_dataset_summary(
    name: str,
    dataset: Optional[AnomalyDataset],
) -> None:
    if dataset is None:
        print(f"{name}: not available")

        return

    summary = dataset.summary()

    print(
        f"{name}: "
        f"samples="
        f"{summary['sample_count']} "
        f"normal="
        f"{summary['normal_count']} "
        f"anomaly="
        f"{summary['anomaly_count']} "
        f"dimension="
        f"{summary['feature_dimension']}"
    )


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------


def main() -> None:
    args = parse_args()

    validate_args(args)

    config = build_training_config(args)

    pipeline = TrainingPipeline(config=config)

    pipeline.start(discover_existing=True)

    pipeline.apply_reproducibility()

    random.seed(args.seed)

    device = select_device(config)

    output_directory = Path(args.output).expanduser().resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading anomaly records from {args.data}")

    records = load_anomaly_records(
        args.data,
        max_records=(args.max_records),
        require_label=True,
    )

    (
        train_records,
        validation_records,
        test_records,
    ) = split_records(
        records,
        train_ratio=(args.train_ratio),
        validation_ratio=(args.validation_ratio),
    )

    train_dataset = anomaly_dataset_from_records(train_records)

    validation_dataset = (
        anomaly_dataset_from_records(validation_records) if validation_records else None
    )

    test_dataset = anomaly_dataset_from_records(test_records) if test_records else None

    feature_dimension = train_dataset.feature_dimension

    for (
        name,
        dataset,
    ) in (
        (
            "validation",
            validation_dataset,
        ),
        (
            "test",
            test_dataset,
        ),
    ):
        if dataset is not None and dataset.feature_dimension != feature_dimension:
            raise TrainingRunError(
                "Anomaly dataset feature dimensions do not match.",
                details={
                    "split": name,
                    "train_dimension": (feature_dimension),
                    "split_dimension": (dataset.feature_dimension),
                },
            )

    pin_memory = bool(args.pin_memory and device.type == "cuda")

    train_loader = build_loader(
        train_dataset,
        batch_size=(args.batch_size),
        shuffle=True,
        num_workers=(args.num_workers),
        pin_memory=pin_memory,
    )

    validation_loader = build_loader(
        validation_dataset,
        batch_size=(args.batch_size),
        shuffle=False,
        num_workers=(args.num_workers),
        pin_memory=pin_memory,
    )

    test_loader = build_loader(
        test_dataset,
        batch_size=(args.batch_size),
        shuffle=False,
        num_workers=(args.num_workers),
        pin_memory=pin_memory,
    )

    model = InfrastructureAnomalyModel(
        input_dim=(feature_dimension),
        hidden_dim=(args.hidden_dim),
        representation_dim=(args.representation_dim),
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=(args.weight_decay),
    )

    class_weight = (
        1.0 if args.disable_class_weight else positive_class_weight(train_dataset)
    )

    pos_weight = torch.tensor(
        class_weight,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model_config = {
        **model.model_config(),
        "temporary_threshold": (args.threshold),
        "positive_class_weight": (class_weight),
        "feature_source": ("legacy_numeric_jsonl"),
    }

    run = pipeline.create_run(
        task=(TrainingTask.ANOMALY.value),
        dataset_id=(args.dataset_id),
        model_name=(args.model_name),
        accelerator=str(device),
        attributes={
            "data_path": str(Path(args.data).expanduser().resolve()),
            "output_directory": (str(output_directory)),
            "record_count": len(records),
            "feature_dimension": (feature_dimension),
        },
    )

    pipeline.transition_run(
        run,
        TrainingState.READY.value,
    )

    pipeline.transition_run(
        run,
        TrainingState.TRAINING.value,
    )

    print(f"Device: {device}")

    print_dataset_summary(
        "train",
        train_dataset,
    )

    print_dataset_summary(
        "validation",
        validation_dataset,
    )

    print_dataset_summary(
        "test",
        test_dataset,
    )

    print(f"Positive class weight: {class_weight:.6f}")

    history: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    best_validation_loss = math.inf

    best_checkpoint = None

    global_step = 0

    try:
        for epoch in range(
            1,
            args.epochs + 1,
        ):
            model.train()

            total_loss = 0.0
            batches = 0

            for (
                batch_number,
                batch,
            ) in enumerate(
                train_loader,
                start=1,
            ):
                optimizer.zero_grad(set_to_none=True)

                features = batch["features"].to(
                    device,
                    non_blocking=True,
                )

                targets = batch["target"].to(
                    device,
                    non_blocking=True,
                )

                valid = targets != IGNORE_INDEX

                if int(valid.sum().item()) == 0:
                    continue

                output = model(features)

                logits = output["anomaly_logits"]

                loss = criterion(
                    logits[valid],
                    targets[valid].float(),
                )

                if not torch.isfinite(loss):
                    raise (
                        TrainingNumericalError(
                            "Anomaly training produced a non-finite loss.",
                            details={
                                "epoch": epoch,
                                "batch": (batch_number),
                            },
                        )
                    )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(args.gradient_clip_norm),
                )

                optimizer.step()

                total_loss += float(loss.item())

                batches += 1
                global_step += 1

                if batch_number % 100 == 0:
                    print(f"Epoch {epoch} batch {batch_number} loss={loss.item():.4f}")

            if batches == 0:
                raise TrainingRunError(
                    "Training epoch processed no usable anomaly batches."
                )

            train_loss = total_loss / batches

            pipeline.add_metric(
                run,
                name="loss",
                value=train_loss,
                epoch=epoch,
                step=global_step,
                split=(EvaluationSplit.TRAIN.value),
            )

            validation_metrics = evaluate(
                model,
                validation_loader,
                device=device,
                criterion=criterion,
                threshold=(args.threshold),
            )

            validation_loss = validation_metrics["loss"]

            checkpoint_score = (
                float(validation_loss) if validation_loss is not None else train_loss
            )

            epoch_result = {
                "epoch": epoch,
                "global_step": (global_step),
                "train_loss": (train_loss),
                **{
                    ("validation_" + name): value
                    for name, value in validation_metrics.items()
                },
            }

            history.append(epoch_result)

            print(f"\nEpoch {epoch}/{args.epochs}")

            print(f"  train loss: {train_loss:.6f}")

            if validation_loss is not None:
                print(f"  validation loss: {validation_loss:.6f}")

                for metric_name in (
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "false_positive_rate",
                    "false_negative_rate",
                ):
                    value = validation_metrics[metric_name]

                    if value is not None:
                        print(f"  {metric_name}: {value:.4f}")

            checkpoint = pipeline.save_checkpoint(
                run=run,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                score=(checkpoint_score),
                score_name=("validation_loss"),
                model_config=(model_config),
                extra={
                    "history": list(history),
                    "data_path": (args.data),
                    "temporary_threshold": (args.threshold),
                },
            )

            copy_checkpoint_file(
                checkpoint.path,
                output_directory / "last.pt",
            )

            if checkpoint_score < best_validation_loss:
                best_validation_loss = checkpoint_score

                best_checkpoint = checkpoint

                copy_checkpoint_file(
                    checkpoint.path,
                    output_directory / "best.pt",
                )

                print("  Saved new best checkpoint.")

        best_checkpoint = pipeline.best_checkpoint(
            run_id=str(run.run_id),
            mark=True,
        )

        pipeline.transition_run(
            run,
            TrainingState.EVALUATING.value,
        )

        pipeline.checkpoints.restore(
            best_checkpoint,
            model=model,
            map_location=device,
            strict=True,
            verify=True,
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            device=device,
            criterion=criterion,
            threshold=args.threshold,
        )

        test_metrics = evaluate(
            model,
            test_loader,
            device=device,
            criterion=criterion,
            threshold=args.threshold,
        )

        validation_evaluation = (
            register_evaluation(
                pipeline,
                run,
                split=(EvaluationSplit.VALIDATION.value),
                metrics=(validation_metrics),
            )
            if validation_loader is not None
            else None
        )

        test_evaluation = (
            register_evaluation(
                pipeline,
                run,
                split=(EvaluationSplit.TEST.value),
                metrics=test_metrics,
            )
            if test_loader is not None
            else None
        )

        pipeline.transition_run(
            run,
            TrainingState.COMPLETED.value,
        )

    except Exception as exc:
        if run.state not in {
            TrainingState.COMPLETED.value,
            TrainingState.FAILED.value,
            TrainingState.CANCELLED.value,
        }:
            pipeline.transition_run(
                run,
                TrainingState.FAILED.value,
                error=exc,
            )

        raise

    print("\nFinal test metrics:")

    for name, value in test_metrics.items():
        if isinstance(
            value,
            float,
        ):
            print(f"  {name}: {value:.6f}")

        else:
            print(f"  {name}: {value}")

    report = {
        "schema_version": "1",
        "run": run.to_dict(),
        "data": str(Path(args.data).expanduser().resolve()),
        "device": str(device),
        "records": {
            "total": len(records),
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "datasets": {
            "train": (train_dataset.summary()),
            "validation": (
                validation_dataset.summary() if validation_dataset is not None else None
            ),
            "test": (test_dataset.summary() if test_dataset is not None else None),
        },
        "model_config": (model_config),
        "class_weighting": {
            "enabled": (not args.disable_class_weight),
            "positive_class_weight": (class_weight),
        },
        "history": history,
        "best_checkpoint": (best_checkpoint.to_dict()),
        "validation": (validation_metrics),
        "test": test_metrics,
        "evaluations": {
            "validation": (
                validation_evaluation.to_dict()
                if validation_evaluation is not None
                else None
            ),
            "test": (
                test_evaluation.to_dict() if test_evaluation is not None else None
            ),
        },
        "threshold": {
            "temporary_value": (args.threshold),
            "calibrated": False,
            "embedded_in_model": (False),
        },
        "candidate_artifact_created": (False),
        "automatic_promotion": (False),
    }

    report_path = output_directory / "training_report.json"

    write_json_atomic(
        report_path,
        report,
    )

    print("\nAnomaly training complete.")

    print(f"Managed best checkpoint: {best_checkpoint.path}")

    print(f"Compatibility checkpoint: {output_directory / 'best.pt'}")

    print(f"Report: {report_path}")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    main()
