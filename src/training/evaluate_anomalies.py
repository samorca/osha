from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.anomaly_dataset import (
    MetricWindowDataset,
    RobustScaler,
    load_metric_rows,
)
from src.training.anomaly_model import (
    GRUAutoencoder,
    reconstruction_scores,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the OSHA anomaly detector "
            "against labelled test telemetry."
        )
    )

    parser.add_argument(
        "--data",
        default=(
            "orchestration-ai-data/processed/"
            "anomalies/records.jsonl"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "artifacts/models/"
            "anomaly-understanding-v1/best.pt"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "artifacts/models/"
            "anomaly-understanding-v1/"
            "evaluation"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--number-workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def binary_metrics(labels, predictions):
    true_positive = sum(
        1
        for label, prediction
        in zip(labels, predictions)
        if label == 1 and prediction == 1
    )

    true_negative = sum(
        1
        for label, prediction
        in zip(labels, predictions)
        if label == 0 and prediction == 0
    )

    false_positive = sum(
        1
        for label, prediction
        in zip(labels, predictions)
        if label == 0 and prediction == 1
    )

    false_negative = sum(
        1
        for label, prediction
        in zip(labels, predictions)
        if label == 1 and prediction == 0
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    specificity = safe_divide(
        true_negative,
        true_negative + false_positive,
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    accuracy = safe_divide(
        true_positive + true_negative,
        len(labels),
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2.0

    false_positive_rate = safe_divide(
        false_positive,
        false_positive + true_negative,
    )

    false_negative_rate = safe_divide(
        false_negative,
        false_negative + true_positive,
    )

    return {
        "examples": len(labels),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": (
            false_positive_rate
        ),
        "false_negative_rate": (
            false_negative_rate
        ),
    }


def main():
    args = parse_args()

    output_directory = Path(args.output)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["model_config"]
    feature_names = checkpoint["feature_names"]
    threshold = checkpoint.get("threshold")

    if threshold is None:
        raise ValueError(
            "Checkpoint has no calibrated threshold"
        )

    scaler = RobustScaler.from_dict(
        checkpoint["scaler"]
    )

    test_rows = load_metric_rows(
        args.data,
        "test",
        feature_names,
    )

    test_dataset = MetricWindowDataset(
        rows=test_rows,
        scaler=scaler,
        window_size=config["window_size"],
        stride=config["stride"],
        expected_interval_seconds=(
            config[
                "expected_interval_seconds"
            ]
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.number_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = GRUAutoencoder(
        number_features=config[
            "number_features"
        ],
        hidden_size=config["hidden_size"],
        latent_size=config["latent_size"],
        number_layers=config["number_layers"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    labels = []
    predictions = []
    scores = []
    timestamps = []
    event_ids = []
    row_ids = []

    with torch.no_grad():
        for batch in test_loader:
            windows = batch["window"].to(
                device,
                non_blocking=True,
            )

            output = model(windows)

            batch_scores = reconstruction_scores(
                windows,
                output["reconstruction"],
            ).cpu()

            batch_predictions = (
                batch_scores > threshold
            ).long()

            labels.extend(
                batch["label"].tolist()
            )

            predictions.extend(
                batch_predictions.tolist()
            )

            scores.extend(
                batch_scores.tolist()
            )

            timestamps.extend(
                list(batch["timestamp"])
            )

            event_ids.extend(
                list(batch["event_id"])
            )

            row_ids.extend(
                batch["row_id"].tolist()
            )

    valid = [
        index
        for index, label in enumerate(labels)
        if label in (0, 1)
    ]

    labels = [labels[index] for index in valid]
    predictions = [
        predictions[index]
        for index in valid
    ]
    scores = [scores[index] for index in valid]
    timestamps = [
        timestamps[index]
        for index in valid
    ]
    event_ids = [
        event_ids[index]
        for index in valid
    ]
    row_ids = [row_ids[index] for index in valid]

    overall = binary_metrics(
        labels,
        predictions,
    )

    event_groups = defaultdict(
        lambda: {
            "labels": [],
            "predictions": [],
        }
    )

    for label, prediction, event_id in zip(
        labels,
        predictions,
        event_ids,
    ):
        event_groups[event_id][
            "labels"
        ].append(label)

        event_groups[event_id][
            "predictions"
        ].append(prediction)

    per_event = {}

    for event_id, values in sorted(
        event_groups.items()
    ):
        per_event[event_id] = binary_metrics(
            values["labels"],
            values["predictions"],
        )

    report = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "device": str(device),
        "threshold": threshold,
        "test_rows": len(test_rows),
        "test_windows": len(test_dataset),
        "overall": overall,
        "per_event": per_event,
    }

    with (
        output_directory / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    with (
        output_directory / "predictions.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "row_id",
                "timestamp",
                "event_id",
                "label",
                "prediction",
                "anomaly_score",
                "threshold",
            ]
        )

        for values in zip(
            row_ids,
            timestamps,
            event_ids,
            labels,
            predictions,
            scores,
        ):
            writer.writerow(
                [
                    *values,
                    threshold,
                ]
            )

    print("\nAnomaly evaluation")
    print("==================")
    print("Device:", device)
    print("Threshold:", threshold)

    for name, value in overall.items():
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")

    print("\nPer-event results:")

    for event_id, metrics in per_event.items():
        print(f"\n{event_id}")

        for name in (
            "examples",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
        ):
            value = metrics[name]

            if isinstance(value, float):
                print(f"  {name}: {value:.6f}")
            else:
                print(f"  {name}: {value}")

    print(
        "\nMetrics saved to:",
        output_directory / "metrics.json",
    )

    print(
        "Predictions saved to:",
        output_directory
        / "predictions.csv",
    )


if __name__ == "__main__":
    main()
