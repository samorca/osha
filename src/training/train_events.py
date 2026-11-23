from __future__ import annotations

"""
Train the OSHA infrastructure event model.

Hardware discovery and device selection are delegated to:

    src.accelerators.AcceleratorManager

This trainer owns only the task-specific PyTorch optimization loop.
Vocabulary and label maps are fitted on the training partition only.
"""

import argparse
import json
import math
import shutil

from pathlib import Path

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

import torch

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.accelerators import (
    AcceleratorConfig,
    AcceleratorManager,
    AcceleratorPrecision,
    AcceleratorSelection,
    AcceleratorSelectionConfig,
)

from src.training.config import (
    AcceleratorConfig as TrainingAcceleratorConfig,
    CheckpointConfig,
    OptimizationConfig,
    ReproducibilityConfig,
    TrainingConfig,
    TrainingPathConfig,
)

from src.training.errors import (
    EmptyTrainingDatasetError,
    TrainingNumericalError,
    TrainingRunError,
)

from src.training.event_dataset import (
    EventDataset,
    IGNORE_INDEX,
    LabelMaps,
    Vocabulary,
    load_event_records,
    split_records,
)

from src.training.event_model import (
    InfrastructureEventModel,
)

from src.training.pipeline import (
    TrainingPipeline,
)

from src.training.schema import (
    EvaluationSplit,
    TrainingState,
    TrainingTask,
)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the OSHA infrastructure "
            "event model."
        )
    )

    parser.add_argument(
        "--data",
        default=(
            "orchestration-ai-data/"
            "processed/logs/events.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "artifacts/models/"
            "event-understanding-v1"
        ),
    )

    parser.add_argument(
        "--model-name",
        default=(
            "infrastructure-event-model"
        ),
    )

    parser.add_argument(
        "--dataset-id",
        default=(
            "osha-infrastructure-events"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
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
        "--min-token-frequency",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-vocabulary-size",
        type=int,
        default=50000,
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
        "--mixed-precision",
        action="store_true",
    )

    parser.add_argument(
        "--minimum-free-memory-gb",
        type=float,
        default=None,
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

    return parser.parse_args()


# ------------------------------------------------------------------
# Argument validation
# ------------------------------------------------------------------

def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.epochs < 1:
        raise ValueError(
            "--epochs must be >= 1"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be > 0"
        )

    if args.weight_decay < 0:
        raise ValueError(
            "--weight-decay must be >= 0"
        )

    if args.max_length < 1:
        raise ValueError(
            "--max-length must be >= 1"
        )

    if args.embedding_dim < 1:
        raise ValueError(
            "--embedding-dim must be >= 1"
        )

    if args.hidden_dim < 1:
        raise ValueError(
            "--hidden-dim must be >= 1"
        )

    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(
            "--dropout must be in [0, 1)"
        )

    if args.gradient_clip_norm <= 0:
        raise ValueError(
            "--gradient-clip-norm must be > 0"
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers must be >= 0"
        )

    if args.keep_last < 1:
        raise ValueError(
            "--keep-last must be >= 1"
        )

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError(
            "--train-ratio must be between 0 and 1"
        )

    if not 0.0 <= args.validation_ratio < 1.0:
        raise ValueError(
            "--validation-ratio must be in [0, 1)"
        )

    if (
        args.train_ratio
        + args.validation_ratio
        >= 1.0
    ):
        raise ValueError(
            "--train-ratio plus "
            "--validation-ratio must be less than 1"
        )

    if (
        args.max_records is not None
        and args.max_records < 1
    ):
        raise ValueError(
            "--max-records must be >= 1"
        )

    if (
        args.minimum_free_memory_gb is not None
        and args.minimum_free_memory_gb < 0
    ):
        raise ValueError(
            "--minimum-free-memory-gb must be >= 0"
        )

    if (
        args.mixed_precision
        and args.accelerator == "cpu"
    ):
        raise ValueError(
            "--mixed-precision cannot be used "
            "with an explicitly selected CPU"
        )


# ------------------------------------------------------------------
# Shared training configuration
# ------------------------------------------------------------------

def build_training_config(
    args: argparse.Namespace,
) -> TrainingConfig:
    config = TrainingConfig(
        enabled=True,

        allowed_tasks=(
            TrainingTask.EVENT.value,
            TrainingTask.ANOMALY.value,
        ),

        paths=TrainingPathConfig(),

        reproducibility=(
            ReproducibilityConfig(
                random_seed=args.seed,
                deterministic=True,
                benchmark=False,
            )
        ),

        optimization=(
            OptimizationConfig(
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=(
                    args.learning_rate
                ),
                weight_decay=(
                    args.weight_decay
                ),
                gradient_clip_norm=(
                    args.gradient_clip_norm
                ),
                accumulation_steps=1,
            )
        ),

        accelerator=(
            TrainingAcceleratorConfig(
                preferred=args.accelerator,
                allow_fallback_to_cpu=(
                    not args.no_cpu_fallback
                ),
                mixed_precision=(
                    args.mixed_precision
                ),
                device_index=(
                    args.device_index
                ),
            )
        ),

        checkpoint=CheckpointConfig(
            enabled=True,
            save_every_epochs=1,
            keep_last=args.keep_last,
            keep_best=True,
            monitor_metric=(
                "validation_loss"
            ),
            mode="min",
            save_optimizer_state=True,
        ),

        attributes={
            "trainer": "train_events",
            "accelerator_provider": (
                "src.accelerators"
            ),
        },
    )

    config.validate()

    return config


# ------------------------------------------------------------------
# Accelerator manager
# ------------------------------------------------------------------

def build_accelerator_manager(
    args: argparse.Namespace,
) -> AcceleratorManager:
    config = AcceleratorConfig(
        selection=(
            AcceleratorSelectionConfig(
                preferred_kind=(
                    args.accelerator
                ),
                device_index=(
                    args.device_index
                ),
                allow_cpu_fallback=(
                    not args.no_cpu_fallback
                ),
                allow_degraded_devices=True,
                require_accelerator=(
                    args.no_cpu_fallback
                    and args.accelerator
                    in {
                        "cuda",
                        "rocm",
                    }
                ),
                prefer_highest_free_memory=True,
            )
        ),
        attributes={
            "consumer": (
                "training.train_events"
            ),
        },
    )

    config.validate()

    return AcceleratorManager(
        config=config
    )


def minimum_free_memory_bytes(
    args: argparse.Namespace,
) -> Optional[int]:
    if args.minimum_free_memory_gb is None:
        return None

    return int(
        args.minimum_free_memory_gb
        * 1024**3
    )


def select_accelerator(
    manager: AcceleratorManager,
    args: argparse.Namespace,
) -> AcceleratorSelection:
    required_precisions = ()

    if args.mixed_precision:
        required_precisions = (
            AcceleratorPrecision.
            FLOAT16.value,
        )

    return manager.select_for_training(
        preferred_kind=args.accelerator,
        device_index=args.device_index,
        minimum_free_memory_bytes=(
            minimum_free_memory_bytes(
                args
            )
        ),
        require_accelerator=(
            args.no_cpu_fallback
            and args.accelerator
            in {
                "cuda",
                "rocm",
            }
        ),
        allow_cpu_fallback=(
            not args.no_cpu_fallback
        ),
        mixed_precision=(
            args.mixed_precision
        ),
        required_precisions=(
            required_precisions
        ),
        refresh=True,
    )


# ------------------------------------------------------------------
# Atomic file helpers
# ------------------------------------------------------------------

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

    temporary = path.with_name(
        path.name + ".writing"
    )

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

        temporary.replace(
            path
        )

    except Exception:
        temporary.unlink(
            missing_ok=True
        )

        raise


def copy_checkpoint_file(
    source: str | Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        destination.name + ".copying"
    )

    shutil.copy2(
        source,
        temporary,
    )

    temporary.replace(
        destination
    )


# ------------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------------

def build_loader(
    dataset: Optional[
        EventDataset
    ],
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
        persistent_workers=(
            num_workers > 0
        ),
    )


# ------------------------------------------------------------------
# Safe loss and accuracy
# ------------------------------------------------------------------

def safe_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> tuple[
    torch.Tensor,
    int,
]:
    valid = (
        targets != IGNORE_INDEX
    )

    count = int(
        valid.sum().item()
    )

    if count == 0:
        return (
            logits.sum() * 0.0,
            0,
        )

    loss = criterion(
        logits,
        targets,
    )

    if not torch.isfinite(
        loss
    ):
        raise TrainingNumericalError(
            "Classification loss is not finite."
        )

    return (
        loss,
        count,
    )


def accuracy_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[int, int]:
    valid = (
        targets != IGNORE_INDEX
    )

    count = int(
        valid.sum().item()
    )

    if count == 0:
        return (
            0,
            0,
        )

    predictions = logits.argmax(
        dim=1
    )

    correct = int(
        (
            predictions[valid]
            == targets[valid]
        ).sum().item()
    )

    return (
        correct,
        count,
    )


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(
    model: InfrastructureEventModel,
    loader: Optional[DataLoader],
    *,
    device: Any,
    criterion: nn.Module,
) -> Dict[str, Any]:
    if loader is None:
        return {
            "loss": None,
            "batch_count": 0,
            "example_count": 0,
            "source_accuracy": None,
            "source_count": 0,
            "component_accuracy": None,
            "component_count": 0,
            "event_accuracy": None,
            "event_count": 0,
        }

    model.eval()

    total_loss = 0.0
    batch_count = 0
    example_count = 0

    correct_counts = {
        "source": 0,
        "component": 0,
        "event": 0,
    }

    target_counts = {
        "source": 0,
        "component": 0,
        "event": 0,
    }

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch[
                "input_ids"
            ].to(
                device,
                non_blocking=True,
            )

            attention_mask = batch[
                "attention_mask"
            ].to(
                device,
                non_blocking=True,
            )

            targets = {
                "source": batch[
                    "source_target"
                ].to(
                    device,
                    non_blocking=True,
                ),

                "component": batch[
                    "component_target"
                ].to(
                    device,
                    non_blocking=True,
                ),

                "event": batch[
                    "event_target"
                ].to(
                    device,
                    non_blocking=True,
                ),
            }

            output = model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
            )

            losses = []
            active_heads = 0

            for name in (
                "source",
                "component",
                "event",
            ):
                logits = output[
                    name + "_logits"
                ]

                loss, valid_count = (
                    safe_classification_loss(
                        logits,
                        targets[name],
                        criterion,
                    )
                )

                if valid_count > 0:
                    losses.append(
                        loss
                    )
                    active_heads += 1

                correct, count = (
                    accuracy_counts(
                        logits,
                        targets[name],
                    )
                )

                correct_counts[name] += (
                    correct
                )

                target_counts[name] += (
                    count
                )

            if active_heads == 0:
                continue

            loss = sum(
                losses
            ) / active_heads

            total_loss += float(
                loss.item()
            )

            batch_count += 1

            example_count += int(
                input_ids.shape[0]
            )

    return {
        "loss": (
            total_loss / batch_count
            if batch_count
            else None
        ),

        "batch_count": batch_count,

        "example_count": (
            example_count
        ),

        "source_accuracy": (
            correct_counts["source"]
            / target_counts["source"]
            if target_counts["source"]
            else None
        ),

        "source_count": (
            target_counts["source"]
        ),

        "component_accuracy": (
            correct_counts[
                "component"
            ]
            / target_counts[
                "component"
            ]
            if target_counts[
                "component"
            ]
            else None
        ),

        "component_count": (
            target_counts[
                "component"
            ]
        ),

        "event_accuracy": (
            correct_counts["event"]
            / target_counts["event"]
            if target_counts["event"]
            else None
        ),

        "event_count": (
            target_counts["event"]
        ),
    }


# ------------------------------------------------------------------
# Evaluation registration
# ------------------------------------------------------------------

def register_evaluation(
    pipeline: TrainingPipeline,
    run: Any,
    *,
    split: str,
    metrics: Mapping[
        str,
        Any,
    ],
):
    numeric = {
        str(name): float(value)
        for name, value
        in metrics.items()
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
                "batch_count",
                "example_count",
                "source_count",
                "component_count",
                "event_count",
            }
            and math.isfinite(
                float(value)
            )
        )
    }

    if not numeric:
        return None

    return pipeline.record_evaluation(
        run=run,
        split=split,
        metrics=numeric,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    validate_args(
        args
    )

    training_config = (
        build_training_config(
            args
        )
    )

    pipeline = TrainingPipeline(
        config=training_config
    )

    pipeline.start(
        discover_existing=True
    )

    pipeline.apply_reproducibility()

    accelerator_manager = (
        build_accelerator_manager(
            args
        )
    )

    accelerator_selection = (
        select_accelerator(
            accelerator_manager,
            args,
        )
    )

    device = (
        accelerator_manager.
        runtime_device(
            accelerator_selection
        )
    )

    output_directory = Path(
        args.output
    ).expanduser().resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    run = None

    try:
        records = load_event_records(
            args.data,
            max_records=(
                args.max_records
            ),
        )

        (
            train_records,
            validation_records,
            test_records,
        ) = split_records(
            records,
            train_ratio=(
                args.train_ratio
            ),
            val_ratio=(
                args.validation_ratio
            ),
        )

        if not train_records:
            raise EmptyTrainingDatasetError(
                "No training event records "
                "were produced."
            )

        vocabulary = Vocabulary(
            train_records,
            min_frequency=(
                args.min_token_frequency
            ),
            max_size=(
                args.max_vocabulary_size
            ),
        )

        labels = LabelMaps(
            train_records
        )

        train_dataset = EventDataset(
            train_records,
            vocabulary,
            labels,
            max_length=args.max_length,
        )

        validation_dataset = (
            EventDataset(
                validation_records,
                vocabulary,
                labels,
                max_length=(
                    args.max_length
                ),
            )
            if validation_records
            else None
        )

        test_dataset = (
            EventDataset(
                test_records,
                vocabulary,
                labels,
                max_length=(
                    args.max_length
                ),
            )
            if test_records
            else None
        )

        pin_memory = bool(
            args.pin_memory
            and accelerator_selection.
            accelerator
        )

        train_loader = build_loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=(
                args.num_workers
            ),
            pin_memory=pin_memory,
        )

        validation_loader = build_loader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=(
                args.num_workers
            ),
            pin_memory=pin_memory,
        )

        test_loader = build_loader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=(
                args.num_workers
            ),
            pin_memory=pin_memory,
        )

        component_classes = max(
            1,
            len(
                labels.component_to_id
            ),
        )

        model_config = {
            "vocab_size": len(
                vocabulary
            ),

            "num_sources": len(
                labels.source_to_id
            ),

            "num_components": (
                component_classes
            ),

            "num_events": len(
                labels.event_to_id
            ),

            "embedding_dim": (
                args.embedding_dim
            ),

            "hidden_dim": (
                args.hidden_dim
            ),

            "dropout": (
                args.dropout
            ),

            "max_length": (
                args.max_length
            ),
        }

        model = InfrastructureEventModel(
            vocab_size=(
                model_config[
                    "vocab_size"
                ]
            ),
            num_sources=(
                model_config[
                    "num_sources"
                ]
            ),
            num_components=(
                model_config[
                    "num_components"
                ]
            ),
            num_events=(
                model_config[
                    "num_events"
                ]
            ),
            embedding_dim=(
                args.embedding_dim
            ),
            hidden_dim=(
                args.hidden_dim
            ),
            dropout=args.dropout,
        ).to(
            device
        )

        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=(
                args.weight_decay
            ),
        )

        criterion = nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX
        )

        use_amp = bool(
            args.mixed_precision
            and accelerator_selection.
            accelerator
        )

        scaler = torch.amp.GradScaler(
            device=(
                "cuda"
                if accelerator_selection.
                accelerator
                else "cpu"
            ),
            enabled=use_amp,
        )

        run = pipeline.create_run(
            task=(
                TrainingTask.EVENT.value
            ),
            dataset_id=(
                args.dataset_id
            ),
            model_name=(
                args.model_name
            ),
            accelerator=(
                accelerator_selection.
                runtime_device
            ),
            attributes={
                "data_path": str(
                    Path(
                        args.data
                    ).expanduser().
                    resolve()
                ),
                "output_directory": str(
                    output_directory
                ),
                "record_count": len(
                    records
                ),
                "accelerator_selection": (
                    accelerator_selection.
                    to_dict()
                ),
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

        print(
            "Accelerator:",
            accelerator_selection.
            device_id,
        )

        print(
            "Runtime device:",
            device,
        )

        print(
            "Selection reason:",
            accelerator_selection.
            reason,
        )

        print(
            "CPU fallback used:",
            accelerator_selection.
            fallback_used,
        )

        print(
            "Records:",
            f"total={len(records)}",
            f"train={len(train_records)}",
            (
                "validation="
                f"{len(validation_records)}"
            ),
            f"test={len(test_records)}",
        )

        history = []
        best_validation_loss = (
            math.inf
        )
        global_step = 0

        for epoch in range(
            1,
            args.epochs + 1,
        ):
            model.train()

            total_loss = 0.0
            batch_count = 0

            for batch_number, batch in (
                enumerate(
                    train_loader,
                    start=1,
                )
            ):
                optimizer.zero_grad(
                    set_to_none=True
                )

                input_ids = batch[
                    "input_ids"
                ].to(
                    device,
                    non_blocking=True,
                )

                attention_mask = batch[
                    "attention_mask"
                ].to(
                    device,
                    non_blocking=True,
                )

                targets = {
                    "source": batch[
                        "source_target"
                    ].to(
                        device,
                        non_blocking=True,
                    ),

                    "component": batch[
                        "component_target"
                    ].to(
                        device,
                        non_blocking=True,
                    ),

                    "event": batch[
                        "event_target"
                    ].to(
                        device,
                        non_blocking=True,
                    ),
                }

                autocast_device = (
                    "cuda"
                    if accelerator_selection.
                    accelerator
                    else "cpu"
                )

                with torch.autocast(
                    device_type=(
                        autocast_device
                    ),
                    dtype=(
                        torch.float16
                        if autocast_device
                        == "cuda"
                        else torch.bfloat16
                    ),
                    enabled=use_amp,
                ):
                    output = model(
                        input_ids=input_ids,
                        attention_mask=(
                            attention_mask
                        ),
                    )

                    losses = []
                    active_heads = 0

                    for name in (
                        "source",
                        "component",
                        "event",
                    ):
                        loss_value, count = (
                            safe_classification_loss(
                                output[
                                    name
                                    + "_logits"
                                ],
                                targets[name],
                                criterion,
                            )
                        )

                        if count > 0:
                            losses.append(
                                loss_value
                            )
                            active_heads += 1

                    if active_heads == 0:
                        continue

                    loss = sum(
                        losses
                    ) / active_heads

                if not torch.isfinite(
                    loss
                ):
                    raise TrainingNumericalError(
                        "Training produced a "
                        "non-finite loss.",
                        details={
                            "epoch": epoch,
                            "batch": (
                                batch_number
                            ),
                        },
                    )

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(
                        args.
                        gradient_clip_norm
                    ),
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                total_loss += float(
                    loss.item()
                )

                batch_count += 1
                global_step += 1

            if batch_count == 0:
                raise TrainingRunError(
                    "Training epoch processed "
                    "no usable batches."
                )

            train_loss = (
                total_loss
                / batch_count
            )

            pipeline.add_metric(
                run,
                name="loss",
                value=train_loss,
                epoch=epoch,
                step=global_step,
                split=(
                    EvaluationSplit.
                    TRAIN.value
                ),
            )

            validation_metrics = evaluate(
                model,
                validation_loader,
                device=device,
                criterion=criterion,
            )

            validation_loss = (
                validation_metrics[
                    "loss"
                ]
            )

            checkpoint_score = (
                float(
                    validation_loss
                )
                if validation_loss
                is not None
                else train_loss
            )

            epoch_result = {
                "epoch": epoch,
                "global_step": (
                    global_step
                ),
                "train_loss": (
                    train_loss
                ),
                **{
                    (
                        "validation_"
                        + name
                    ): value
                    for name, value
                    in (
                        validation_metrics.
                        items()
                    )
                },
            }

            history.append(
                epoch_result
            )

            checkpoint = (
                pipeline.save_checkpoint(
                    run=run,
                    model=model,
                    optimizer=optimizer,
                    scaler=(
                        scaler
                        if use_amp
                        else None
                    ),
                    epoch=epoch,
                    step=global_step,
                    score=(
                        checkpoint_score
                    ),
                    score_name=(
                        "validation_loss"
                    ),
                    vocabulary=(
                        vocabulary.to_dict()
                    ),
                    labels=(
                        labels.to_dict()
                    ),
                    model_config=(
                        model_config
                    ),
                    extra={
                        "history": list(
                            history
                        ),
                        "accelerator": (
                            accelerator_selection.
                            to_dict()
                        ),
                        "mixed_precision": (
                            use_amp
                        ),
                    },
                )
            )

            copy_checkpoint_file(
                checkpoint.path,
                output_directory
                / "last.pt",
            )

            if (
                checkpoint_score
                < best_validation_loss
            ):
                best_validation_loss = (
                    checkpoint_score
                )

                copy_checkpoint_file(
                    checkpoint.path,
                    output_directory
                    / "best.pt",
                )

                print(
                    f"Epoch {epoch}: "
                    "saved new best "
                    "checkpoint."
                )

            print(
                f"Epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.6f} "
                f"validation_loss="
                f"{validation_loss}"
            )

        best_checkpoint = (
            pipeline.best_checkpoint(
                run_id=str(
                    run.run_id
                ),
                mark=True,
            )
        )

        pipeline.transition_run(
            run,
            TrainingState.
            EVALUATING.value,
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
        )

        test_metrics = evaluate(
            model,
            test_loader,
            device=device,
            criterion=criterion,
        )

        validation_evaluation = (
            register_evaluation(
                pipeline,
                run,
                split=(
                    EvaluationSplit.
                    VALIDATION.value
                ),
                metrics=(
                    validation_metrics
                ),
            )
            if validation_loader
            is not None
            else None
        )

        test_evaluation = (
            register_evaluation(
                pipeline,
                run,
                split=(
                    EvaluationSplit.
                    TEST.value
                ),
                metrics=test_metrics,
            )
            if test_loader
            is not None
            else None
        )

        pipeline.transition_run(
            run,
            TrainingState.
            COMPLETED.value,
        )

        report = {
            "schema_version": "1",
            "run": run.to_dict(),
            "data": str(
                Path(
                    args.data
                ).expanduser().resolve()
            ),
            "accelerator": (
                accelerator_selection.
                to_dict()
            ),
            "runtime_device": str(
                device
            ),
            "mixed_precision": (
                use_amp
            ),
            "records": {
                "total": len(
                    records
                ),
                "train": len(
                    train_records
                ),
                "validation": len(
                    validation_records
                ),
                "test": len(
                    test_records
                ),
            },
            "model_config": (
                model_config
            ),
            "vocabulary_size": len(
                vocabulary
            ),
            "classes": {
                "source": len(
                    labels.source_to_id
                ),
                "component": len(
                    labels.
                    component_to_id
                ),
                "event": len(
                    labels.event_to_id
                ),
            },
            "history": history,
            "best_checkpoint": (
                best_checkpoint.to_dict()
            ),
            "validation": (
                validation_metrics
            ),
            "test": test_metrics,
            "evaluations": {
                "validation": (
                    validation_evaluation.
                    to_dict()
                    if validation_evaluation
                    is not None
                    else None
                ),
                "test": (
                    test_evaluation.
                    to_dict()
                    if test_evaluation
                    is not None
                    else None
                ),
            },
            "automatic_promotion": (
                False
            ),
        }

        report_path = (
            output_directory
            / "training_report.json"
        )

        write_json_atomic(
            report_path,
            report,
        )

        print(
            "\nEvent training complete."
        )

        print(
            "Best checkpoint:",
            best_checkpoint.path,
        )

        print(
            "Report:",
            report_path,
        )

    except Exception as exc:
        if (
            run is not None
            and run.state
            not in {
                TrainingState.
                COMPLETED.value,
                TrainingState.
                FAILED.value,
                TrainingState.
                CANCELLED.value,
            }
        ):
            pipeline.transition_run(
                run,
                TrainingState.
                FAILED.value,
                error=exc,
            )

        raise

    finally:
        try:
            accelerator_manager.release(
                accelerator_selection
            )

        except Exception as release_error:
            print(
                "Warning: accelerator "
                "release failed:",
                type(
                    release_error
                ).__name__,
            )


if __name__ == "__main__":
    main()
