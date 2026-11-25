from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Sequence,
)

import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from src.training.checkpoints import CheckpointManager
from src.training.config import (
    CheckpointConfig,
    TrainingConfig,
    TrainingPathConfig,
)
from src.training.errors import EmptyTrainingDatasetError
from src.training.decision_risk_model import DecisionRiskModel
from src.training.incident_root_cause_model import IncidentRootCauseModel
from src.training.remediation_ranker import RemediationRanker
from src.training.schema import (
    TrainingRun,
    TrainingState,
    TrainingTask,
)

FEATURE_DIMENSION = 16


def stable_number(
    value: Any,
) -> float:
    text = str(value)
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()
    return (
        int.from_bytes(
            digest[:8],
            "little",
            signed=False,
        )
        % 10000
        / 10000.0
    )


def feature_vector(
    value: Any,
) -> List[float]:
    if isinstance(value, Sequence) and not isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        values: List[float] = []
        for item in value[:FEATURE_DIMENSION]:
            if isinstance(item, (int, float)):
                number = float(item)
                if torch.isfinite(torch.tensor(number)).item():
                    values.append(number)
        if values:
            return (values + [0.0] * FEATURE_DIMENSION)[
                :FEATURE_DIMENSION
            ]

    if isinstance(value, Mapping):
        values = []
        for key in sorted(value, key=str):
            item = value[key]
            if isinstance(item, (int, float)):
                values.append(float(item))
            elif key in {
                "severity",
                "risk",
                "confidence",
                "score",
                "novelty",
            }:
                values.append(
                    stable_number(item)
                )
        if values:
            return (values + [0.0] * FEATURE_DIMENSION)[
                :FEATURE_DIMENSION
            ]

    return [
        stable_number(value)
        for _ in range(FEATURE_DIMENSION)
    ]


@dataclass(frozen=True)
class OperationalSample:
    features: List[float]
    target: int
    score: float = 0.0


@dataclass(frozen=True)
class OperationalCheckpointSpec:
    task: str
    model_name: str
    directory: str
    score_key: str
    labels: Mapping[str, Sequence[str]]
    model_factory: Callable[[Mapping[str, Any]], nn.Module]


class OperationalDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[
            OperationalSample
        ],
    ) -> None:
        self.samples = list(samples)

    def __len__(
        self,
    ) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "features": torch.tensor(
                sample.features,
                dtype=torch.float32,
            ),
            "target": torch.tensor(
                sample.target,
                dtype=torch.long,
            ),
            "score": torch.tensor(
                sample.score,
                dtype=torch.float32,
            ),
        }


def read_jsonl(
    paths: Iterable[
        Path
    ],
) -> List[Mapping[str, Any]]:
    records: List[
        Mapping[str, Any]
    ] = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, Mapping):
                    records.append(value)
    return records


def read_json_payload(
    paths: Iterable[
        Path
    ],
) -> List[Mapping[str, Any]]:
    records: List[
        Mapping[str, Any]
    ] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            value = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            json.JSONDecodeError,
            OSError,
            UnicodeError,
        ):
            continue
        if isinstance(value, Mapping):
            records.append(value)
        elif isinstance(value, list):
            records.extend(
                item
                for item in value
                if isinstance(item, Mapping)
            )
    return records


INCIDENT_SEVERITIES = (
    "info",
    "low",
    "medium",
    "high",
    "critical",
)

ROOT_CAUSES = (
    "capacity",
    "configuration",
    "network",
    "storage",
    "service",
    "security",
    "software",
    "unknown",
)

RISK_LEVELS = (
    "none",
    "low",
    "medium",
    "high",
    "critical",
)


def severity_target(
    record: Mapping[
        str,
        Any,
    ],
) -> int:
    severity = str(
        record.get(
            "severity",
            record.get(
                "risk_level",
                "",
            ),
        )
    ).lower()
    if severity in INCIDENT_SEVERITIES:
        return INCIDENT_SEVERITIES.index(
            severity
        )
    score = float(
        record.get(
            "score",
            record.get(
                "confidence",
                0.0,
            )
        )
        if isinstance(
            record.get(
                "score",
                record.get(
                    "confidence",
                    0.0,
                ),
            ),
            (int, float),
        )
        else 0.0
    )
    if score >= 0.90:
        return 4
    if score >= 0.75:
        return 3
    if score >= 0.50:
        return 2
    if score >= 0.20:
        return 1
    return 0


def root_cause_target(
    record: Mapping[
        str,
        Any,
    ],
) -> int:
    text = " ".join(
        str(record.get(key, "")).lower()
        for key in (
            "title",
            "description",
            "anomaly_type",
            "action_type",
            "category",
        )
    )
    for index, cause in enumerate(
        ROOT_CAUSES[:-1]
    ):
        if cause in text:
            return index
    return len(ROOT_CAUSES) - 1


def risk_target(
    record: Mapping[
        str,
        Any,
    ],
) -> int:
    level = str(
        record.get(
            "risk_level",
            record.get(
                "severity",
                "",
            ),
        )
    ).lower()
    if level in RISK_LEVELS:
        return RISK_LEVELS.index(level)
    score = float(
        record.get(
            "risk_score",
            record.get(
                "score",
                0.0,
            )
        )
        if isinstance(
            record.get(
                "risk_score",
                record.get(
                    "score",
                    0.0,
                ),
            ),
            (int, float),
        )
        else 0.0
    )
    if score >= 0.90:
        return 4
    if score >= 0.75:
        return 3
    if score >= 0.50:
        return 2
    if score >= 0.20:
        return 1
    return 0


def build_incident_samples(
    root: Path,
) -> List[
    OperationalSample
]:
    records = read_jsonl(
        (
            root
            / "intelligence/anomalies.jsonl",
            root
            / "intelligence/incidents.jsonl",
        )
    )
    samples = []
    for record in records:
        if not record:
            continue
        samples.append(
            OperationalSample(
                features=feature_vector(record),
                target=severity_target(record),
            )
        )

    anomaly_records = read_jsonl(
        (
            root
            / "processed/anomalies/records.jsonl",
            root
            / "processed/anomalies/records_with_internal_external.jsonl",
        )
    )
    for record in anomaly_records[:2000]:
        nested = record.get("values")
        features = feature_vector(
            nested
            if nested is not None
            else record
        )
        is_anomaly = str(
            nested.get("is_anomaly")
            if isinstance(nested, Mapping)
            else record.get("is_anomaly", 0)
        ).lower()
        anomaly = is_anomaly in {
            "1",
            "true",
            "anomaly",
            "positive",
        }
        samples.append(
            OperationalSample(
                features=features,
                target=(3 if anomaly else 1),
                score=(1.0 if anomaly else 0.0),
            )
        )
    if not samples:
        raise EmptyTrainingDatasetError(
            "Incident training requires real labeled incident or anomaly data."
        )
    return samples


def build_decision_samples(
    root: Path,
) -> List[
    OperationalSample
]:
    records = read_json_payload(
        (
            root
            / "decision/result.json",
        )
    )
    records.extend(
        read_jsonl(
            (
                root
                / "intelligence/incidents.jsonl",
            )
        )
    )
    records.extend(
        read_json_payload(
            (
                root
                / "remediation/request.json",
            )
        )
    )
    samples = []
    for record in records:
        values = record
        if isinstance(record.get("decision"), Mapping):
            values = record["decision"]
        if isinstance(record.get("contexts"), list):
            for context in record["contexts"]:
                if not isinstance(context, Mapping):
                    continue
                samples.append(
                    OperationalSample(
                        features=feature_vector(context),
                        target=risk_target(context),
                        score=float(
                            context.get(
                                "impact_score",
                                context.get(
                                    "incident_confidence",
                                    0.0,
                                ),
                            )
                            if isinstance(
                                context.get(
                                    "impact_score",
                                    context.get(
                                        "incident_confidence",
                                        0.0,
                                    ),
                                ),
                                (int, float),
                            )
                            else 0.0
                        ),
                    )
                )
        if isinstance(record.get("plans"), list):
            for plan in record["plans"]:
                if not isinstance(plan, Mapping):
                    continue
                samples.append(
                    OperationalSample(
                        features=feature_vector(plan),
                        target=risk_target(plan),
                        score=float(
                            plan.get(
                                "risk_score",
                                0.0,
                            )
                            if isinstance(
                                plan.get(
                                    "risk_score",
                                    0.0,
                                ),
                                (int, float),
                            )
                            else 0.0
                        ),
                    )
                )
        if isinstance(
            record.get("candidates"),
            list,
        ):
            for candidate in record["candidates"]:
                if isinstance(candidate, Mapping):
                    raw_score = candidate.get(
                        "risk_score",
                        candidate.get(
                            "score",
                            0.0,
                        ),
                    )
                    samples.append(
                        OperationalSample(
                            features=feature_vector(candidate),
                            target=risk_target(candidate),
                            score=(
                                float(raw_score)
                                if isinstance(
                                    raw_score,
                                    (int, float),
                                )
                                else 0.0
                            ),
                        )
                    )
        else:
            raw_score = values.get(
                "risk_score",
                values.get(
                    "score",
                    0.0,
                ),
            )
            samples.append(
                OperationalSample(
                    features=feature_vector(values),
                    target=risk_target(values),
                    score=(
                        float(raw_score)
                        if isinstance(
                            raw_score,
                            (int, float),
                        )
                        else 0.0
                    ),
                )
            )
    if not samples:
        raise EmptyTrainingDatasetError(
            "Decision-risk training requires real decision, incident, or remediation data."
        )
    operational_records = read_jsonl(
        (
            root
            / "processed/anomalies/records_with_internal_external.jsonl",
            root
            / "processed/logs/events_with_external.jsonl",
        )
    )
    for record in operational_records[:1000]:
        nested = record.get("values")
        label_value = (
            nested.get("is_anomaly")
            if isinstance(nested, Mapping)
            else record.get("is_anomaly", 0)
        )
        try:
            score = float(label_value)
        except (TypeError, ValueError):
            score = 0.0
        risk_index = (
            4
            if score >= 0.75
            else 3
            if score >= 0.50
            else 2
            if score >= 0.25
            else 1
        )
        samples.append(
            OperationalSample(
                features=feature_vector(record),
                target=risk_index,
                score=score,
            )
        )
    return samples


def build_remediation_samples(
    root: Path,
) -> List[
    OperationalSample
]:
    records = read_json_payload(
        (
            root
            / "remediation/result.json",
        )
    )
    records.extend(
        read_json_payload(
            (
                root
                / "decision/result.json",
            )
        )
    )
    records.extend(
        read_jsonl(
            (
                root
                / "audit/remediation.jsonl",
            )
        )
    )
    samples = []
    for record in records:
        target = 1 if str(
            record.get(
                "status",
                record.get(
                    "outcome",
                    "",
                ),
            )
        ).lower() in {
            "completed",
            "verified",
            "succeeded",
            "executed",
            "approved",
        } else 0
        risk = float(
            record.get(
                "risk_score",
                0.0,
            )
            if isinstance(
                record.get(
                    "risk_score",
                    0.0,
                ),
                (int, float),
            )
            else 0.0
        )
        score = (
            1.0
            if target == 1
            and risk < 0.75
            else 0.0
        )
        samples.append(
            OperationalSample(
                features=feature_vector(record),
                target=target,
                score=score,
            )
        )
    if not samples:
        raise EmptyTrainingDatasetError(
            "Remediation training requires real remediation outcome records."
        )
    return samples


def split_samples(
    samples: Sequence[
        OperationalSample
    ],
) -> tuple[
    List[OperationalSample],
    List[OperationalSample],
]:
    values = list(samples)
    random.Random(42).shuffle(values)
    split = max(
        1,
        int(
            len(values)
            * 0.8
        ),
    )
    if len(values) == 1:
        return values, values
    return values[:split], values[split:]


def split_training_validation(
    samples: Sequence[OperationalSample],
    *,
    validation_ratio: float = 0.10,
    seed: int = 43,
) -> tuple[List[OperationalSample], List[OperationalSample]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    values = list(samples)
    if len(values) < 2:
        raise ValueError("At least two training samples are required")
    random.Random(seed).shuffle(values)
    validation_count = min(
        len(values) - 1,
        max(1, int(len(values) * validation_ratio)),
    )
    return values[validation_count:], values[:validation_count]


def select_training_device(accelerator: str) -> torch.device:
    if accelerator == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. "
                "Install the GPU training requirements and verify the NVIDIA driver."
            )
        return torch.device("cuda")
    if accelerator == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters(), torch.empty(0)).device


def train_classifier(
    model: nn.Module,
    train_dataset: Dataset,
    *,
    epochs: int,
    batch_size: int,
    output_key: str,
    validation_samples: Sequence[OperationalSample],
    validation_metric_names: Sequence[str],
    incident: bool = False,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    history = []
    best_score = float("-inf")
    best_validation_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in loader:
            optimizer.zero_grad(
                set_to_none=True
            )
            features = batch[
                "features"
            ].to(device)
            output = model(features)
            logits = output[output_key]
            loss = nn.functional.cross_entropy(
                logits,
                batch[
                    "target"
                ].to(device),
            )
            loss.backward()
            optimizer.step()
            total_loss += float(
                loss.detach()
            )
            batch_count += 1
        validation = evaluate_classifier(
            model,
            validation_samples,
            incident=incident,
        )
        score = min(
            float(validation[name])
            for name in validation_metric_names
        )
        validation_loss = float(validation["loss"])
        history.append({
            "epoch": epoch,
            "loss": total_loss / max(1, batch_count),
            "validation": validation,
        })
        if score > best_score or (
            score == best_score
            and validation_loss < best_validation_loss
        ):
            best_score = score
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_loss": best_validation_loss,
    }


def train_ranker(
    model: RemediationRanker,
    train_dataset: Dataset,
    *,
    epochs: int,
    batch_size: int,
    validation_samples: Sequence[OperationalSample],
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    history = []
    best_score = float("-inf")
    best_validation_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in loader:
            optimizer.zero_grad(
                set_to_none=True
            )
            output = model(
                batch[
                    "features"
                ].to(device)
            )
            loss = nn.functional.binary_cross_entropy(
                output[
                    "recommendation_score"
                ],
                batch["target"].to(device, dtype=torch.float32),
            )
            loss.backward()
            optimizer.step()
            total_loss += float(
                loss.detach()
            )
            batch_count += 1
        validation = evaluate_ranker(
            model,
            validation_samples,
        )
        score = float(validation["accuracy"])
        validation_loss = float(validation["loss"])
        history.append({
            "epoch": epoch,
            "loss": total_loss / max(1, batch_count),
            "validation": validation,
        })
        if score > best_score or (
            score == best_score
            and validation_loss < best_validation_loss
        ):
            best_score = score
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_loss": best_validation_loss,
    }


def train_incident_model(
    model: IncidentRootCauseModel,
    train_dataset: Dataset,
    *,
    epochs: int,
    batch_size: int,
    validation_samples: Sequence[OperationalSample],
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    history = []
    best_score = float("-inf")
    best_validation_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in loader:
            optimizer.zero_grad(
                set_to_none=True
            )
            features = batch[
                "features"
            ].to(device)
            severity_targets = batch[
                "target"
            ].to(device)
            output = model(features)

            severity_loss = nn.functional.cross_entropy(
                output["severity_logits"],
                severity_targets,
            )

            root_cause_targets = severity_targets.remainder(
                len(ROOT_CAUSES)
            )
            root_cause_loss = nn.functional.cross_entropy(
                output["root_cause_logits"],
                root_cause_targets,
            )

            loss = (
                severity_loss
                + root_cause_loss
            )
            loss.backward()
            optimizer.step()
            total_loss += float(
                loss.detach()
            )
            batch_count += 1
        validation = evaluate_classifier(
            model,
            validation_samples,
            incident=True,
        )
        score = min(
            float(validation["severity_accuracy"]),
            float(validation["root_cause_accuracy"]),
        )
        validation_loss = float(validation["loss"])
        history.append({
            "epoch": epoch,
            "loss": total_loss / max(1, batch_count),
            "validation": validation,
        })
        if score > best_score or (
            score == best_score
            and validation_loss < best_validation_loss
        ):
            best_score = score
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_loss": best_validation_loss,
    }


def atomic_save(
    path: Path,
    payload: Mapping[
        str,
        Any,
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_name(
        path.name + ".tmp"
    )
    torch.save(
        payload,
        temporary,
    )
    temporary.replace(path)


def write_json(
    path: Path,
    payload: Mapping[
        str,
        Any,
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def evaluate_classifier(
    model: nn.Module,
    samples: Sequence[
        OperationalSample
    ],
    *,
    incident: bool = False,
) -> Dict[str, Any]:
    if not samples:
        return {"accuracy": 0.0, "samples": 0}

    model.eval()
    device = model_device(model)
    correct_severity = 0
    correct_root_cause = 0
    total_loss = 0.0
    with torch.no_grad():
        for sample in samples:
            features = torch.tensor(
                [sample.features],
                dtype=torch.float32,
            ).to(device)
            output = model(features)
            if incident:
                total_loss += float(
                    nn.functional.cross_entropy(
                        output["severity_logits"],
                        torch.tensor(
                            [sample.target],
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                    + nn.functional.cross_entropy(
                        output["root_cause_logits"],
                        torch.tensor(
                            [sample.target % len(ROOT_CAUSES)],
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                )
                severity_prediction = int(
                    output[
                        "severity_logits"
                    ].argmax(dim=1).item()
                )
                root_cause_prediction = int(
                    output[
                        "root_cause_logits"
                    ].argmax(dim=1).item()
                )
                correct_severity += int(
                    severity_prediction
                    == sample.target
                )
                correct_root_cause += int(
                    root_cause_prediction
                    == (sample.target % len(ROOT_CAUSES))
                )
            else:
                total_loss += float(
                    nn.functional.cross_entropy(
                        output["risk_logits"],
                        torch.tensor(
                            [sample.target],
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                )
                prediction = int(
                    output[
                        "risk_logits"
                    ].argmax(dim=1).item()
                )
                correct_severity += int(
                    prediction
                    == sample.target
                )

    if incident:
        count = len(samples)
        return {
            "samples": count,
            "loss": total_loss / count,
            "severity_accuracy": (
                correct_severity / count
            ),
            "root_cause_accuracy": (
                correct_root_cause / count
            ),
        }

    return {
        "samples": len(samples),
        "loss": total_loss / len(samples),
        "risk_accuracy": (
            correct_severity
            / len(samples)
        ),
    }


def evaluate_ranker(
    model: RemediationRanker,
    samples: Sequence[
        OperationalSample
    ],
) -> Dict[str, Any]:
    if not samples:
        return {
            "samples": 0,
            "accuracy": 0.0,
            "mean_score": 0.0,
        }

    model.eval()
    device = model_device(model)
    correct = 0
    scores = []
    total_loss = 0.0
    with torch.no_grad():
        for sample in samples:
            features = torch.tensor(
                [sample.features],
                dtype=torch.float32,
            ).to(device)
            score = float(
                model(
                    features
                )[
                    "recommendation_score"
                ].item()
            )
            scores.append(score)
            total_loss += float(
                nn.functional.binary_cross_entropy(
                    torch.tensor(score, device=device),
                    torch.tensor(float(sample.target), device=device),
                )
            )
            prediction = 1 if score >= 0.5 else 0
            correct += int(
                prediction
                == sample.target
            )

    return {
        "samples": len(samples),
        "loss": total_loss / len(samples),
        "accuracy": (
            correct
            / len(samples)
        ),
        "mean_score": (
            sum(scores)
            / len(scores)
        ),
    }


def checkpoint_manager(
    checkpoint_root: Path,
) -> CheckpointManager:
    config = TrainingConfig(
        paths=TrainingPathConfig(
            checkpoint_directory=str(checkpoint_root),
        ),
        checkpoint=CheckpointConfig(
            monitor_metric="test_accuracy",
            mode="max",
            save_optimizer_state=False,
        ),
    )
    return CheckpointManager(config=config)


def save_operational_checkpoint(
    manager: CheckpointManager,
    *,
    task: str,
    model_name: str,
    model: nn.Module,
    score: float | None,
    model_config: Mapping[str, Any],
    labels: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> str:
    run = TrainingRun(
        task=task,
        dataset_id="operational-models-v1",
        state=TrainingState.COMPLETED.value,
        model_name=model_name,
        attributes={
            "model_family": task,
            "checkpoint_source": "operational_training",
        },
    ).finalize()
    checkpoint = manager.save(
        run=run,
        model=model,
        model_name=model_name,
        task=task,
        score=score,
        score_name="test_accuracy",
        model_config=model_config,
        labels=labels,
        feature_schema={
            "dimension": FEATURE_DIMENSION,
            "encoding": "operational_feature_vector_v1",
        },
        extra=dict(extra),
    )
    return checkpoint.path


def register_existing_operational_checkpoints(
    output_root: Path,
    manager: CheckpointManager,
) -> Dict[str, str]:
    specifications = (
        OperationalCheckpointSpec(
            task=TrainingTask.INCIDENT_ROOT_CAUSE.value,
            model_name="incident-root-cause-v1",
            directory="incident-root-cause-v1",
            score_key="root_cause_accuracy",
            labels={
                "severity": list(INCIDENT_SEVERITIES),
                "root_cause": list(ROOT_CAUSES),
            },
            model_factory=lambda config: IncidentRootCauseModel(
                input_dim=int(config["input_dim"]),
                severity_count=int(config["severity_count"]),
                root_cause_count=int(config["root_cause_count"]),
            ),
        ),
        OperationalCheckpointSpec(
            task=TrainingTask.DECISION_POLICY_RISK.value,
            model_name="decision-risk-v1",
            directory="decision-risk-v1",
            score_key="risk_accuracy",
            labels={
                "risk": list(RISK_LEVELS),
            },
            model_factory=lambda config: DecisionRiskModel(
                input_dim=int(config["input_dim"]),
                risk_count=int(config["risk_count"]),
            ),
        ),
        OperationalCheckpointSpec(
            task=TrainingTask.REMEDIATION_RECOMMENDATION.value,
            model_name="remediation-ranking-v1",
            directory="remediation-ranking-v1",
            score_key="accuracy",
            labels={},
            model_factory=lambda config: RemediationRanker(
                input_dim=int(config["input_dim"]),
            ),
        ),
    )
    registered: Dict[str, str] = {}

    for specification in specifications:
        family = specification.task
        artifact_directory = output_root / specification.directory
        artifact_path = artifact_directory / "best.pt"
        report_path = artifact_directory / "training_report.json"
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Operational model checkpoint is missing: {artifact_path}"
            )
        if not report_path.is_file():
            raise FileNotFoundError(
                f"Operational training report is missing: {report_path}"
            )

        payload = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, Mapping):
            raise ValueError(f"Invalid operational checkpoint: {artifact_path}")
        model_config = payload.get("model_config")
        model_state = payload.get("model_state_dict")
        if not isinstance(model_config, Mapping) or not isinstance(
            model_state,
            Mapping,
        ):
            raise ValueError(f"Invalid operational checkpoint: {artifact_path}")

        model = specification.model_factory(model_config)
        model.load_state_dict(model_state, strict=True)
        model.eval()

        report = json.loads(report_path.read_text(encoding="utf-8"))
        evaluation = report.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError(f"Invalid evaluation report: {report_path}")
        score = evaluation.get(specification.score_key)
        if score is not None:
            score = float(score)

        registered[family] = save_operational_checkpoint(
            manager,
            task=family,
            model_name=specification.model_name,
            model=model,
            score=score,
            model_config=model_config,
            labels=specification.labels,
            extra={
                "artifact_path": str(artifact_path.resolve()),
                "evaluation": dict(evaluation),
                "training_report_path": str(report_path.resolve()),
            },
        )

    return registered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train OSHA incident, decision, "
            "and remediation models."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--data-root",
        default="orchestration-ai-data",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/models",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints",
    )
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--register-existing",
        action="store_true",
        help="Create managed checkpoints from existing operational best.pt files.",
    )
    arguments = parser.parse_args()
    if arguments.epochs < 1:
        parser.error("--epochs must be >= 1")
    if arguments.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    return arguments


def main() -> int:
    arguments = parse_args()
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    root = Path(
        arguments.data_root
    ).expanduser().resolve()
    output_root = Path(
        arguments.output_root
    ).expanduser().resolve()
    checkpoint_root = Path(
        arguments.checkpoint_root
    ).expanduser().resolve()
    manager = checkpoint_manager(checkpoint_root)

    if arguments.register_existing:
        registered = register_existing_operational_checkpoints(
            output_root,
            manager,
        )
        print(
            json.dumps(
                {"registered_checkpoints": registered},
                sort_keys=True,
            )
        )
        return 0

    device = select_training_device(arguments.accelerator)
    print(f"Training device: {device}")

    incident_source_samples = build_incident_samples(root)
    decision_source_samples = build_decision_samples(root)
    remediation_source_samples = build_remediation_samples(root)

    incident_train, incident_test = split_samples(
        incident_source_samples
    )
    incident_train, incident_validation = split_training_validation(
        incident_train,
        seed=arguments.seed,
    )
    decision_train, decision_test = split_samples(
        decision_source_samples
    )
    decision_train, decision_validation = split_training_validation(
        decision_train,
        seed=arguments.seed,
    )
    remediation_train, remediation_test = split_samples(
        remediation_source_samples
    )
    remediation_train, remediation_validation = split_training_validation(
        remediation_train,
        seed=arguments.seed,
    )

    incident_model = IncidentRootCauseModel()
    incident_training = train_incident_model(
        incident_model,
        OperationalDataset(incident_train),
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        validation_samples=incident_validation,
        device=device,
    )

    decision_model = DecisionRiskModel()
    decision_training = train_classifier(
        decision_model,
        OperationalDataset(decision_train),
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        output_key="risk_logits",
        validation_samples=decision_validation,
        validation_metric_names=("risk_accuracy",),
        device=device,
    )

    remediation_model = RemediationRanker()
    remediation_training = train_ranker(
        remediation_model,
        OperationalDataset(remediation_train),
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        validation_samples=remediation_validation,
        device=device,
    )

    incident_evaluation = evaluate_classifier(
        incident_model,
        incident_test,
        incident=True,
    )
    decision_evaluation = evaluate_classifier(
        decision_model,
        decision_test,
    )
    remediation_evaluation = evaluate_ranker(
        remediation_model,
        remediation_test,
    )
    incident_model.to("cpu")
    decision_model.to("cpu")
    remediation_model.to("cpu")

    incident_dir = output_root / "incident-root-cause-v1"
    decision_dir = output_root / "decision-risk-v1"
    remediation_dir = output_root / "remediation-ranking-v1"

    incident_model_config = {
        "input_dim": FEATURE_DIMENSION,
        "severity_count": len(INCIDENT_SEVERITIES),
        "root_cause_count": len(ROOT_CAUSES),
    }
    decision_model_config = {
        "input_dim": FEATURE_DIMENSION,
        "risk_count": len(RISK_LEVELS),
    }
    remediation_model_config = {
        "input_dim": FEATURE_DIMENSION,
    }

    atomic_save(
        incident_dir / "best.pt",
        {
            "schema_version": "1",
            "model_family": "incident_root_cause",
            "model_config": incident_model_config,
            "severity_labels": list(INCIDENT_SEVERITIES),
            "root_cause_labels": list(ROOT_CAUSES),
            "model_state_dict": incident_model.state_dict(),
        },
    )

    atomic_save(
        decision_dir / "best.pt",
        {
            "schema_version": "1",
            "model_family": "decision_policy_risk",
            "model_config": decision_model_config,
            "risk_labels": list(RISK_LEVELS),
            "model_state_dict": decision_model.state_dict(),
        },
    )

    atomic_save(
        remediation_dir / "best.pt",
        {
            "schema_version": "1",
            "model_family": "remediation_recommendation_ranking",
            "model_config": remediation_model_config,
            "model_state_dict": remediation_model.state_dict(),
        },
    )

    write_json(
        incident_dir / "training_report.json",
        {
            "family": "incident_root_cause",
            "device": str(device),
            "train_samples": len(incident_train),
            "validation_samples": len(incident_validation),
            "test_samples": len(incident_test),
            "feature_dimension": FEATURE_DIMENSION,
            "labels": {
                "severity": list(INCIDENT_SEVERITIES),
                "root_cause": list(ROOT_CAUSES),
            },
            "evaluation": incident_evaluation,
            **incident_training,
        },
    )
    write_json(
        decision_dir / "training_report.json",
        {
            "family": "decision_policy_risk",
            "device": str(device),
            "train_samples": len(decision_train),
            "validation_samples": len(decision_validation),
            "test_samples": len(decision_test),
            "feature_dimension": FEATURE_DIMENSION,
            "risk_labels": list(RISK_LEVELS),
            "evaluation": decision_evaluation,
            **decision_training,
        },
    )
    write_json(
        remediation_dir / "training_report.json",
        {
            "family": "remediation_recommendation_ranking",
            "device": str(device),
            "train_samples": len(remediation_train),
            "validation_samples": len(remediation_validation),
            "test_samples": len(remediation_test),
            "feature_dimension": FEATURE_DIMENSION,
            "ranking_type": "successful_remediation_outcome",
            "evaluation": remediation_evaluation,
            **remediation_training,
        },
    )

    incident_checkpoint = save_operational_checkpoint(
        manager,
        task=TrainingTask.INCIDENT_ROOT_CAUSE.value,
        model_name="incident-root-cause-v1",
        model=incident_model,
        score=incident_evaluation.get("root_cause_accuracy"),
        model_config=incident_model_config,
        labels={
            "severity": list(INCIDENT_SEVERITIES),
            "root_cause": list(ROOT_CAUSES),
        },
        extra={
            "evaluation": incident_evaluation,
            "training": incident_training,
            "train_samples": len(incident_train),
            "test_samples": len(incident_test),
        },
    )
    decision_checkpoint = save_operational_checkpoint(
        manager,
        task=TrainingTask.DECISION_POLICY_RISK.value,
        model_name="decision-risk-v1",
        model=decision_model,
        score=decision_evaluation.get("risk_accuracy"),
        model_config=decision_model_config,
        labels={"risk": list(RISK_LEVELS)},
        extra={
            "evaluation": decision_evaluation,
            "training": decision_training,
            "train_samples": len(decision_train),
            "test_samples": len(decision_test),
        },
    )
    remediation_checkpoint = save_operational_checkpoint(
        manager,
        task=TrainingTask.REMEDIATION_RECOMMENDATION.value,
        model_name="remediation-ranking-v1",
        model=remediation_model,
        score=remediation_evaluation.get("accuracy"),
        model_config=remediation_model_config,
        labels={},
        extra={
            "evaluation": remediation_evaluation,
            "training": remediation_training,
            "train_samples": len(remediation_train),
            "test_samples": len(remediation_test),
        },
    )

    print(
        json.dumps(
            {
                "incident_root_cause": {
                    "train_samples": len(incident_train),
                    "test_samples": len(incident_test),
                    "artifact": str(
                        incident_dir / "best.pt"
                    ),
                    "checkpoint": incident_checkpoint,
                },
                "decision_policy_risk": {
                    "train_samples": len(decision_train),
                    "test_samples": len(decision_test),
                    "artifact": str(
                        decision_dir / "best.pt"
                    ),
                    "checkpoint": decision_checkpoint,
                },
                "remediation_recommendation_ranking": {
                    "train_samples": len(remediation_train),
                    "test_samples": len(remediation_test),
                    "artifact": str(
                        remediation_dir / "best.pt"
                    ),
                    "checkpoint": remediation_checkpoint,
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
