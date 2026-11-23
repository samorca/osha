from __future__ import annotations

"""
Evaluate a trained OSHA infrastructure event
model.

Supports checkpoints produced by train_events.py,
including the compatibility best.pt/last.pt files
and managed checkpoint payloads.

Evaluation reuses the training-fitted vocabulary
and label mappings stored in the checkpoint.
"""

import argparse
import json
import math

from pathlib import Path

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

import torch

from torch import nn

from torch.utils.data import (
    DataLoader,
)

from src.training.config import (
    TrainingConfig,
)

from src.training.errors import (
    CheckpointLoadError,
    EvaluationError,
    EvaluationMetricError,
    TrainingSourceNotFoundError,
)

from src.training.event_dataset import (
    EventDataset,
    IGNORE_INDEX,
    LabelMaps,
    Vocabulary,
    load_event_records,
)

from src.training.event_model import (
    InfrastructureEventModel,
)

from src.training.pipeline import (
    TrainingPipeline,
)

from src.training.schema import (
    EvaluationSplit,
)


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an OSHA infrastructure "
            "event model checkpoint."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to best.pt, last.pt, or a "
            "managed checkpoint.pt file."
        ),
    )

    parser.add_argument(
        "--data",
        required=True,
        help=(
            "JSONL event data to evaluate."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "artifacts/evaluations/"
            "event-evaluation.json"
        ),
        help=(
            "Destination JSON evaluation report."
        ),
    )

    parser.add_argument(
        "--split",
        choices=(
            "validation",
            "test",
        ),
        default="test",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cuda",
            "cpu",
        ),
        default="auto",
    )

    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--register",
        action="store_true",
        help=(
            "Register the evaluation with a "
            "TrainingPipeline when run metadata "
            "is available in the checkpoint."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------

def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers must be >= 0"
        )

    if (
        args.max_records is not None
        and args.max_records < 1
    ):
        raise ValueError(
            "--max-records must be >= 1"
        )

    if args.device_index < 0:
        raise ValueError(
            "--device-index must be >= 0"
        )


# ---------------------------------------------------------------
# Device
# ---------------------------------------------------------------

def select_device(
    requested: str,
    *,
    device_index: int = 0,
) -> torch.device:
    requested = (
        requested.strip().lower()
    )

    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise EvaluationError(
                "CUDA was requested but "
                "is unavailable."
            )

        if (
            device_index
            >= torch.cuda.device_count()
        ):
            raise EvaluationError(
                "Requested CUDA device "
                "index is unavailable.",
                details={
                    "device_index": (
                        device_index
                    ),
                    "visible_devices": (
                        torch.cuda.
                        device_count()
                    ),
                },
            )

        return torch.device(
            f"cuda:{device_index}"
        )

    if torch.cuda.is_available():
        return torch.device(
            f"cuda:{device_index}"
        )

    return torch.device(
        "cpu"
    )


# ---------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------

def load_checkpoint(
    path: str | Path,
    *,
    map_location: Any,
) -> Dict[str, Any]:
    source = Path(
        path
    ).expanduser().resolve()

    if not source.exists():
        raise TrainingSourceNotFoundError(
            "Event checkpoint was not found.",
            details={
                "path": str(
                    source
                ),
            },
        )

    if source.is_dir():
        source = (
            source
            / "checkpoint.pt"
        )

    if not source.is_file():
        raise CheckpointLoadError(
            "Event checkpoint path is not "
            "a regular file.",
            details={
                "path": str(
                    source
                ),
            },
        )

    try:
        try:
            value = torch.load(
                source,
                map_location=map_location,
                weights_only=False,
            )

        except TypeError:
            value = torch.load(
                source,
                map_location=map_location,
            )

    except Exception as exc:
        raise CheckpointLoadError(
            "Event checkpoint could not "
            "be loaded.",
            details={
                "path": str(
                    source
                ),
                "error_type": (
                    type(exc).__name__
                ),
                "error_module": (
                    type(exc).__module__
                ),
            },
        ) from exc

    if not isinstance(
        value,
        Mapping,
    ):
        raise CheckpointLoadError(
            "Event checkpoint root must "
            "be an object."
        )

    payload = dict(
        value
    )

    required = (
        "model_state_dict",
        "model_config",
        "vocabulary",
        "labels",
    )

    missing = [
        key
        for key in required
        if key not in payload
    ]

    if missing:
        raise CheckpointLoadError(
            "Event checkpoint is missing "
            "required fields.",
            details={
                "missing": missing,
            },
        )

    return payload


# ---------------------------------------------------------------
# Reconstruct vocabulary
# ---------------------------------------------------------------

def load_vocabulary(
    payload: Mapping[
        str,
        Any,
    ],
) -> Vocabulary:
    value = payload.get(
        "vocabulary"
    )

    if not isinstance(
        value,
        Mapping,
    ):
        raise CheckpointLoadError(
            "Checkpoint vocabulary is invalid."
        )

    return Vocabulary.from_dict(
        value
    )


# ---------------------------------------------------------------
# Reconstruct labels
# ---------------------------------------------------------------

def load_labels(
    payload: Mapping[
        str,
        Any,
    ],
) -> LabelMaps:
    value = payload.get(
        "labels"
    )

    if not isinstance(
        value,
        Mapping,
    ):
        raise CheckpointLoadError(
            "Checkpoint label maps are invalid."
        )

    return LabelMaps.from_dict(
        value
    )


# ---------------------------------------------------------------
# Reconstruct model
# ---------------------------------------------------------------

def load_model(
    payload: Mapping[
        str,
        Any,
    ],
    *,
    device: torch.device,
) -> InfrastructureEventModel:
    config = payload.get(
        "model_config"
    )

    if not isinstance(
        config,
        Mapping,
    ):
        raise CheckpointLoadError(
            "Checkpoint model_config "
            "is invalid."
        )

    required = (
        "vocab_size",
        "num_sources",
        "num_components",
        "num_events",
    )

    missing = [
        name
        for name in required
        if name not in config
    ]

    if missing:
        raise CheckpointLoadError(
            "Checkpoint model configuration "
            "is incomplete.",
            details={
                "missing": missing,
            },
        )

    model = InfrastructureEventModel(
        vocab_size=int(
            config[
                "vocab_size"
            ]
        ),

        num_sources=int(
            config[
                "num_sources"
            ]
        ),

        num_components=int(
            config[
                "num_components"
            ]
        ),

        num_events=int(
            config[
                "num_events"
            ]
        ),

        embedding_dim=int(
            config.get(
                "embedding_dim",
                192,
            )
        ),

        hidden_dim=int(
            config.get(
                "hidden_dim",
                256,
            )
        ),

        dropout=float(
            config.get(
                "dropout",
                0.20,
            )
        ),
    )

    state = payload.get(
        "model_state_dict"
    )

    if not isinstance(
        state,
        Mapping,
    ):
        raise CheckpointLoadError(
            "Checkpoint contains no valid "
            "model_state_dict."
        )

    try:
        model.load_state_dict(
            state,
            strict=True,
        )

    except Exception as exc:
        raise CheckpointLoadError(
            "Event model state is not "
            "compatible with the model "
            "configuration.",
            details={
                "error_type": (
                    type(exc).__name__
                ),
            },
        ) from exc

    model.to(
        device
    )

    model.eval()

    return model


# ---------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------

def accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[
    int,
    int,
]:
    predictions = logits.argmax(
        dim=1
    )

    valid = (
        targets
        != IGNORE_INDEX
    )

    count = int(
        valid.sum().item()
    )

    if count == 0:
        return (
            0,
            0,
        )

    correct = int(
        (
            predictions[
                valid
            ]
            == targets[
                valid
            ]
        ).sum().item()
    )

    return (
        correct,
        count,
    )


# ---------------------------------------------------------------
# Safe loss
# ---------------------------------------------------------------

def safe_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> tuple[
    torch.Tensor,
    int,
]:
    valid = (
        targets
        != IGNORE_INDEX
    )

    count = int(
        valid.sum().item()
    )

    if count == 0:
        return (
            logits.sum()
            * 0.0,
            0,
        )

    value = criterion(
        logits,
        targets,
    )

    if not torch.isfinite(
        value
    ):
        raise EvaluationMetricError(
            "Evaluation produced "
            "non-finite loss."
        )

    return (
        value,
        count,
    )


# ---------------------------------------------------------------
# Evaluate loader
# ---------------------------------------------------------------

def evaluate_loader(
    model: InfrastructureEventModel,
    loader: DataLoader,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss(
        ignore_index=(
            IGNORE_INDEX
        )
    )

    total_loss = 0.0
    batches = 0
    examples = 0

    source_correct = 0
    source_count = 0

    component_correct = 0
    component_count = 0

    event_correct = 0
    event_count = 0

    model.eval()

    with torch.no_grad():
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

            source_target = batch[
                "source_target"
            ].to(
                device,
                non_blocking=True,
            )

            component_target = batch[
                "component_target"
            ].to(
                device,
                non_blocking=True,
            )

            event_target = batch[
                "event_target"
            ].to(
                device,
                non_blocking=True,
            )

            output = model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
            )

            (
                source_loss,
                source_valid,
            ) = safe_loss(
                output[
                    "source_logits"
                ],
                source_target,
                criterion,
            )

            (
                component_loss,
                component_valid,
            ) = safe_loss(
                output[
                    "component_logits"
                ],
                component_target,
                criterion,
            )

            (
                event_loss,
                event_valid,
            ) = safe_loss(
                output[
                    "event_logits"
                ],
                event_target,
                criterion,
            )

            active_heads = sum(
                (
                    source_valid > 0,
                    component_valid > 0,
                    event_valid > 0,
                )
            )

            if active_heads > 0:
                loss = (
                    source_loss
                    + component_loss
                    + event_loss
                ) / active_heads

                total_loss += float(
                    loss.item()
                )

                batches += 1

            examples += int(
                input_ids.shape[
                    0
                ]
            )

            correct, count = accuracy(
                output[
                    "source_logits"
                ],
                source_target,
            )

            source_correct += correct
            source_count += count

            correct, count = accuracy(
                output[
                    "component_logits"
                ],
                component_target,
            )

            component_correct += (
                correct
            )

            component_count += (
                count
            )

            correct, count = accuracy(
                output[
                    "event_logits"
                ],
                event_target,
            )

            event_correct += (
                correct
            )

            event_count += count

    return {
        "loss": (
            total_loss
            / batches
            if batches
            else None
        ),

        "example_count": (
            examples
        ),

        "batch_count": (
            batches
        ),

        "source_accuracy": (
            source_correct
            / source_count
            if source_count
            else None
        ),

        "source_count": (
            source_count
        ),

        "component_accuracy": (
            component_correct
            / component_count
            if component_count
            else None
        ),

        "component_count": (
            component_count
        ),

        "event_accuracy": (
            event_correct
            / event_count
            if event_count
            else None
        ),

        "event_count": (
            event_count
        ),
    }


# ---------------------------------------------------------------
# Atomic report
# ---------------------------------------------------------------

def write_report(
    path: str | Path,
    value: Mapping[
        str,
        Any,
    ],
) -> Path:
    destination = Path(
        path
    ).expanduser().resolve()

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        destination.with_name(
            destination.name
            + ".writing"
        )
    )

    try:
        temporary.write_text(
            json.dumps(
                dict(
                    value
                ),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary.replace(
            destination
        )

    except Exception:
        try:
            temporary.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise

    return destination


# ---------------------------------------------------------------
# Register evaluation
# ---------------------------------------------------------------

def register_evaluation(
    pipeline: TrainingPipeline,
    payload: Mapping[
        str,
        Any,
    ],
    *,
    split: str,
    metrics: Mapping[
        str,
        Any,
    ],
):
    """
    Register when a checkpoint contains run
    metadata matching TrainingRun's structure.

    Standalone legacy checkpoints without run
    metadata remain fully evaluable.
    """

    run_value = payload.get(
        "run"
    )

    if not isinstance(
        run_value,
        Mapping,
    ):
        return None

    run_id = run_value.get(
        "run_id"
    )

    if not run_id:
        return None

    run = pipeline.runs.get(
        str(
            run_id
        )
    )

    if run is None:
        return None

    numeric = {
        str(
            name
        ): float(
            value
        )
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
            and not str(
                name
            ).endswith(
                "_count"
            )
            and name
            not in {
                "batch_count",
                "example_count",
            }
            and math.isfinite(
                float(
                    value
                )
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


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    args = parse_args()

    validate_args(
        args
    )

    device = select_device(
        args.device,
        device_index=(
            args.device_index
        ),
    )

    print(
        f"Device: {device}"
    )

    payload = load_checkpoint(
        args.checkpoint,
        map_location=(
            device
        ),
    )

    vocabulary = load_vocabulary(
        payload
    )

    labels = load_labels(
        payload
    )

    model = load_model(
        payload,
        device=device,
    )

    records = load_event_records(
        args.data,
        max_records=(
            args.max_records
        ),
    )

    model_config = dict(
        payload[
            "model_config"
        ]
    )

    max_length = int(
        model_config.get(
            "max_length",
            128,
        )
    )

    dataset = EventDataset(
        records,
        vocabulary,
        labels,
        max_length=max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    metrics = evaluate_loader(
        model,
        loader,
        device=device,
    )

    print(
        "\nEvent evaluation:"
    )

    for name, value in (
        metrics.items()
    ):
        if isinstance(
            value,
            float,
        ):
            print(
                f"  {name}: "
                f"{value:.4f}"
            )

        else:
            print(
                f"  {name}: {value}"
            )

    evaluation = None

    if args.register:
        pipeline = TrainingPipeline(
            config=TrainingConfig()
        )

        pipeline.start(
            discover_existing=True
        )

        evaluation = (
            register_evaluation(
                pipeline,
                payload,
                split=args.split,
                metrics=metrics,
            )
        )

    report = {
        "schema_version": "1",

        "checkpoint": str(
            Path(
                args.checkpoint
            ).expanduser().
            resolve()
        ),

        "data": str(
            Path(
                args.data
            ).expanduser().
            resolve()
        ),

        "split": (
            args.split
        ),

        "device": str(
            device
        ),

        "record_count": len(
            records
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

        "model_config": (
            model_config
        ),

        "metrics": (
            metrics
        ),

        "evaluation": (
            evaluation.to_dict()
            if evaluation
            is not None
            else None
        ),

        "unknown_targets_ignored": (
            True
        ),

        "ignore_index": (
            IGNORE_INDEX
        ),
    }

    report_path = write_report(
        args.output,
        report,
    )

    print(
        f"\nReport: "
        f"{report_path}"
    )


# ---------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------

if __name__ == "__main__":
    main()
