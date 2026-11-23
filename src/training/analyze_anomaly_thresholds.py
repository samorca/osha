from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyse anomaly-score distributions and "
            "threshold tradeoffs."
        )
    )

    parser.add_argument(
        "--predictions",
        default=(
            "artifacts/models/"
            "anomaly-understanding-smoke/"
            "evaluation/predictions.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "artifacts/models/"
            "anomaly-understanding-smoke/"
            "threshold-analysis"
        ),
    )

    parser.add_argument(
        "--current-threshold",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--threshold-count",
        type=int,
        default=1000,
    )

    return parser.parse_args()


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (
        scores > threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    false_positive_rate = safe_divide(
        fp,
        fp + tn,
    )

    false_negative_rate = safe_divide(
        fn,
        fn + tp,
    )

    accuracy = safe_divide(
        tp + tn,
        len(labels),
    )

    balanced_accuracy = balanced_accuracy_score(
        labels,
        predictions,
    )

    return {
        "threshold": float(threshold),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "false_negative_rate": float(
            false_negative_rate
        ),
        "predicted_anomalies": int(
            predictions.sum()
        ),
    }


def distribution_summary(
    values: np.ndarray,
) -> dict:
    percentiles = [
        0,
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        100,
    ]

    result = {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }

    result["percentiles"] = {
        str(percentile): float(
            np.percentile(
                values,
                percentile,
            )
        )
        for percentile in percentiles
    }

    return result


def find_best(
    rows: list[dict],
    field: str,
) -> dict:
    return max(
        rows,
        key=lambda row: (
            row[field],
            row["recall"],
            -row["false_positive_rate"],
        ),
    )


def best_at_fpr(
    rows: list[dict],
    maximum_fpr: float,
) -> dict | None:
    eligible = [
        row
        for row in rows
        if (
            row["false_positive_rate"]
            <= maximum_fpr
        )
    ]

    if not eligible:
        return None

    return max(
        eligible,
        key=lambda row: (
            row["recall"],
            row["precision"],
            row["f1"],
        ),
    )


def main():
    args = parse_args()

    input_path = Path(
        args.predictions
    )

    output_directory = Path(
        args.output
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        input_path
    )

    required_columns = {
        "label",
        "anomaly_score",
        "event_id",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Predictions file is missing columns: "
            + ", ".join(sorted(missing))
        )

    dataframe = dataframe[
        dataframe["label"].isin([0, 1])
    ].copy()

    dataframe["label"] = dataframe[
        "label"
    ].astype(int)

    dataframe["anomaly_score"] = pd.to_numeric(
        dataframe["anomaly_score"],
        errors="raise",
    )

    labels = dataframe[
        "label"
    ].to_numpy(dtype=int)

    scores = dataframe[
        "anomaly_score"
    ].to_numpy(dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError(
            "Anomaly scores contain NaN or infinity"
        )

    if len(np.unique(labels)) != 2:
        raise ValueError(
            "Both normal and anomaly labels "
            "are required"
        )

    normal_scores = scores[
        labels == 0
    ]

    anomaly_scores = scores[
        labels == 1
    ]

    minimum_score = float(
        np.min(scores)
    )

    maximum_score = float(
        np.max(scores)
    )

    quantile_thresholds = np.quantile(
        scores,
        np.linspace(
            0.0,
            1.0,
            args.threshold_count,
        ),
    )

    unique_scores = np.unique(scores)

    if len(unique_scores) <= 10000:
        thresholds = unique_scores
    else:
        thresholds = quantile_thresholds

    thresholds = np.unique(
        np.concatenate(
            [
                thresholds,
                quantile_thresholds,
                np.array(
                    [
                        np.nextafter(
                            minimum_score,
                            -np.inf,
                        ),
                        np.nextafter(
                            maximum_score,
                            np.inf,
                        ),
                    ]
                ),
            ]
        )
    )

    sweep_rows = [
        calculate_metrics(
            labels,
            scores,
            threshold,
        )
        for threshold in thresholds
    ]

    best_f1 = find_best(
        sweep_rows,
        "f1",
    )

    best_balanced = find_best(
        sweep_rows,
        "balanced_accuracy",
    )

    controlled_fpr = {}

    for target in (
        0.01,
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
    ):
        controlled_fpr[
            f"{int(target * 100)}_percent"
        ] = best_at_fpr(
            sweep_rows,
            target,
        )

    roc_auc = roc_auc_score(
        labels,
        scores,
    )

    pr_auc = average_precision_score(
        labels,
        scores,
    )

    fpr_curve, tpr_curve, roc_thresholds = (
        roc_curve(
            labels,
            scores,
        )
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            labels,
            scores,
        )
    )

    if args.current_threshold is None:
        if "threshold" in dataframe.columns:
            current_threshold = float(
                dataframe[
                    "threshold"
                ].iloc[0]
            )
        else:
            current_threshold = None
    else:
        current_threshold = (
            args.current_threshold
        )

    current_metrics = None

    if current_threshold is not None:
        current_metrics = calculate_metrics(
            labels,
            scores,
            current_threshold,
        )

    per_event = {}

    for event_id, group in dataframe.groupby(
        "event_id"
    ):
        event_labels = group[
            "label"
        ].to_numpy(dtype=int)

        event_scores = group[
            "anomaly_score"
        ].to_numpy(dtype=float)

        event_result = {
            "examples": int(
                len(group)
            ),
            "normal_examples": int(
                (event_labels == 0).sum()
            ),
            "anomaly_examples": int(
                (event_labels == 1).sum()
            ),
            "score_distribution": {
                "all": distribution_summary(
                    event_scores
                )
            },
        }

        if (
            event_labels == 0
        ).any():
            event_result[
                "score_distribution"
            ]["normal"] = (
                distribution_summary(
                    event_scores[
                        event_labels == 0
                    ]
                )
            )

        if (
            event_labels == 1
        ).any():
            event_result[
                "score_distribution"
            ]["anomaly"] = (
                distribution_summary(
                    event_scores[
                        event_labels == 1
                    ]
                )
            )

        if (
            len(np.unique(event_labels))
            == 2
        ):
            event_result["roc_auc"] = float(
                roc_auc_score(
                    event_labels,
                    event_scores,
                )
            )

            event_result["pr_auc"] = float(
                average_precision_score(
                    event_labels,
                    event_scores,
                )
            )

        per_event[str(event_id)] = (
            event_result
        )

    report = {
        "predictions_file": str(
            input_path
        ),
        "examples": int(
            len(dataframe)
        ),
        "normal_examples": int(
            (labels == 0).sum()
        ),
        "anomaly_examples": int(
            (labels == 1).sum()
        ),
        "score_distribution": {
            "normal": distribution_summary(
                normal_scores
            ),
            "anomaly": distribution_summary(
                anomaly_scores
            ),
        },
        "roc_auc": float(
            roc_auc
        ),
        "pr_auc": float(
            pr_auc
        ),
        "current_threshold": (
            current_metrics
        ),
        "best_f1_threshold": best_f1,
        "best_balanced_accuracy_threshold": (
            best_balanced
        ),
        "recall_at_controlled_fpr": (
            controlled_fpr
        ),
        "per_event": per_event,
        "warning": (
            "Thresholds in this report were analysed "
            "using final test labels. They are diagnostic "
            "and must not be reported as unbiased final "
            "production performance."
        ),
    }

    with (
        output_directory
        / "threshold_analysis.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    sweep_frame = pd.DataFrame(
        sweep_rows
    )

    sweep_frame.to_csv(
        output_directory
        / "threshold_sweep.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))

    plt.hist(
        normal_scores,
        bins=80,
        alpha=0.60,
        density=True,
        label="Normal",
    )

    plt.hist(
        anomaly_scores,
        bins=80,
        alpha=0.60,
        density=True,
        label="Anomaly",
    )

    if current_threshold is not None:
        plt.axvline(
            current_threshold,
            linestyle="--",
            label=(
                "Current threshold "
                f"{current_threshold:.4f}"
            ),
        )

    plt.axvline(
        best_f1["threshold"],
        linestyle=":",
        label=(
            "Diagnostic best F1 "
            f"{best_f1['threshold']:.4f}"
        ),
    )

    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title(
        "Normal vs anomaly score distribution"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory
        / "score_distribution.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall_curve,
        precision_curve,
        label=f"PR-AUC={pr_auc:.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory
        / "precision_recall_curve.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr_curve,
        tpr_curve,
        label=f"ROC-AUC={roc_auc:.4f}",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random baseline",
    )
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory
        / "roc_curve.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(
        sweep_frame["threshold"],
        sweep_frame["precision"],
        label="Precision",
    )
    plt.plot(
        sweep_frame["threshold"],
        sweep_frame["recall"],
        label="Recall",
    )
    plt.plot(
        sweep_frame["threshold"],
        sweep_frame["f1"],
        label="F1",
    )
    plt.plot(
        sweep_frame["threshold"],
        sweep_frame[
            "balanced_accuracy"
        ],
        label="Balanced accuracy",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Threshold tradeoffs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory
        / "threshold_tradeoffs.png",
        dpi=160,
    )
    plt.close()

    print("\nThreshold analysis")
    print("==================")
    print("Examples:", len(labels))
    print(
        "Normal:",
        int((labels == 0).sum()),
    )
    print(
        "Anomaly:",
        int((labels == 1).sum()),
    )
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC:  {pr_auc:.6f}")

    if current_metrics is not None:
        print("\nCurrent threshold:")
        print(
            json.dumps(
                current_metrics,
                indent=2,
            )
        )

    print("\nDiagnostic best F1:")
    print(
        json.dumps(
            best_f1,
            indent=2,
        )
    )

    print(
        "\nDiagnostic best balanced accuracy:"
    )
    print(
        json.dumps(
            best_balanced,
            indent=2,
        )
    )

    print(
        "\nRecall at controlled "
        "false-positive rates:"
    )

    print(
        json.dumps(
            controlled_fpr,
            indent=2,
        )
    )

    print(
        "\nSaved:",
        output_directory
        / "threshold_analysis.json",
    )
    print(
        "Saved:",
        output_directory
        / "threshold_sweep.csv",
    )
    print(
        "Saved:",
        output_directory
        / "score_distribution.png",
    )
    print(
        "Saved:",
        output_directory
        / "precision_recall_curve.png",
    )
    print(
        "Saved:",
        output_directory
        / "roc_curve.png",
    )
    print(
        "Saved:",
        output_directory
        / "threshold_tradeoffs.png",
    )

    print(
        "\nWARNING: These thresholds use test "
        "labels and are diagnostic only."
    )


if __name__ == "__main__":
    main()
