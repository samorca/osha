from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import torch

from src.training.event_dataset import (
    LabelMaps,
    Vocabulary,
)
from src.training.event_model import (
    InfrastructureEventModel,
)


def read_jsonl(
    path: Path,
) -> Iterator[dict]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc


def load_event_runtime(
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["model_config"]

    vocabulary = Vocabulary.from_dict(checkpoint["vocabulary"])

    labels = LabelMaps.from_dict(checkpoint["labels"])

    model = InfrastructureEventModel(
        vocab_size=config["vocab_size"],
        num_sources=config["num_sources"],
        num_components=config["num_components"],
        num_events=config["num_events"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return (
        model,
        vocabulary,
        labels,
        config,
    )


def top_predictions(
    logits: torch.Tensor,
    lookup: dict[int, str],
    *,
    top_k: int,
) -> list[dict]:
    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    count = min(
        top_k,
        probabilities.shape[-1],
    )

    values, indices = torch.topk(
        probabilities,
        k=count,
        dim=-1,
    )

    output = []

    for probability, index in zip(
        values[0].tolist(),
        indices[0].tolist(),
    ):
        output.append(
            {
                "name": lookup[index],
                "probability": float(probability),
            }
        )

    return output


def log_text_from_window(
    window: dict,
    *,
    maximum_logs: int,
) -> str:
    logs = window.get(
        "logs",
        [],
    )

    if not isinstance(logs, list):
        return ""

    selected = logs[-maximum_logs:]

    messages = []

    for item in selected:
        payload = item.get(
            "payload",
            {},
        )

        message = payload.get("message")

        if isinstance(message, str):
            text = message.strip()

            if text:
                messages.append(text)

    return "\n".join(messages)


def infer_window(
    window: dict,
    *,
    model,
    vocabulary,
    labels,
    config,
    device,
    top_k: int,
    maximum_logs: int,
) -> dict:
    text = log_text_from_window(
        window,
        maximum_logs=maximum_logs,
    )

    base = {
        "schema_version": "1",
        "record_type": "runtime_inference",
        "window_id": window.get("window_id"),
        "window_start": window.get("window_start"),
        "window_end": window.get("window_end"),
        "environment_id": window.get("environment_id"),
        "cluster_id": window.get("cluster_id"),
        "namespace": window.get("namespace"),
        "service_name": window.get("service_name"),
        "resource_key": window.get("resource_key"),
        "pod_name": window.get("pod_name"),
        "node_name": window.get("node_name"),
        "container_name": window.get("container_name"),
        "signal_counts": window.get(
            "signal_counts",
            {},
        ),
        "log_count": len(
            window.get(
                "logs",
                [],
            )
        ),
        "status": "processed",
    }

    if not text:
        base.update(
            {
                "status": "no_logs",
                "source_predictions": [],
                "component_predictions": [],
                "event_predictions": [],
                "event_embedding": None,
            }
        )

        return base

    input_ids, attention_mask = vocabulary.encode(
        text,
        config["max_length"],
    )

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    mask_tensor = torch.tensor(
        [attention_mask],
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        output = model(
            input_tensor,
            mask_tensor,
        )

    representation = output["representation"][0].cpu().tolist()

    source_predictions = top_predictions(
        output["source_logits"],
        labels.id_to_source,
        top_k=top_k,
    )

    component_predictions = (
        top_predictions(
            output["component_logits"],
            labels.id_to_component,
            top_k=top_k,
        )
        if labels.id_to_component
        else []
    )

    event_predictions = top_predictions(
        output["event_logits"],
        labels.id_to_event,
        top_k=top_k,
    )

    # This is confidence, not correctness. Live Kubernetes
    # telemetry may differ substantially from LogHub.
    top_event_confidence = (
        event_predictions[0]["probability"] if event_predictions else 0.0
    )

    novelty_score = 1.0 - top_event_confidence

    base.update(
        {
            "source_predictions": (source_predictions),
            "component_predictions": (component_predictions),
            "event_predictions": (event_predictions),
            "top_event_confidence": (top_event_confidence),
            "novelty_score": novelty_score,
            "event_embedding": (representation),
            "model_checkpoint": (
                str(
                    config.get(
                        "checkpoint_name",
                        "event-understanding-v1",
                    )
                )
            ),
        }
    )

    return base


def run_inference(
    *,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    top_k: int,
    maximum_logs: int,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        model,
        vocabulary,
        labels,
        config,
    ) = load_event_runtime(
        checkpoint_path,
        device,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows = 0
    processed = 0
    no_logs = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        for window in read_jsonl(input_path):
            if window.get("record_type") != "correlated_window":
                continue

            windows += 1

            result = infer_window(
                window,
                model=model,
                vocabulary=vocabulary,
                labels=labels,
                config=config,
                device=device,
                top_k=top_k,
                maximum_logs=maximum_logs,
            )

            if result["status"] == "no_logs":
                no_logs += 1
            else:
                processed += 1

            output_handle.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "device": str(device),
        "windows_read": windows,
        "windows_processed": processed,
        "windows_without_logs": no_logs,
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Run OSHA event inference on correlated telemetry windows.")
    )

    parser.add_argument(
        "--input",
        default=("orchestration-ai-data/processed/live/kubernetes/windows.jsonl"),
    )

    parser.add_argument(
        "--output",
        default=("orchestration-ai-data/processed/live/kubernetes/predictions.jsonl"),
    )

    parser.add_argument(
        "--checkpoint",
        default=("artifacts/models/event-understanding-v1/best.pt"),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--maximum-logs",
        type=int,
        default=50,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.top_k < 1:
        raise ValueError("top-k must be positive")

    if args.maximum_logs < 1:
        raise ValueError("maximum-logs must be positive")

    report = run_inference(
        input_path=Path(args.input),
        output_path=Path(args.output),
        checkpoint_path=Path(args.checkpoint),
        top_k=args.top_k,
        maximum_logs=args.maximum_logs,
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
