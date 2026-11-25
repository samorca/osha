from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import os
import subprocess

from collections.abc import Callable

import pytest

from src.training.checkpoints import CheckpointManager
from src.training.config import TrainingConfig, TrainingPathConfig
from src.training.errors import CheckpointLoadError, EmptyTrainingDatasetError
from src.training.operational_models import (
    build_decision_samples,
    build_incident_samples,
    build_remediation_samples,
)
from src.training.train_all import (
    _prepare_inputs,
    _promotion_input,
    _training_environment,
)
from src.training.schema import TrainingTask
from src.training.operational_models import OperationalSample


def test_legacy_capacity_state_does_not_break_checkpoint_discovery(
    tmp_path: Path,
) -> None:
    metadata = (
        tmp_path
        / "capacity_forecasting"
        / "osha-capacity-autoregressive"
        / "legacy"
        / "checkpoint.json"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        '{"model_type":"osha.capacity.autoregressive-ridge","schema_version":"1"}',
        encoding="utf-8",
    )
    manager = CheckpointManager(
        config=TrainingConfig(
            paths=TrainingPathConfig(checkpoint_directory=str(tmp_path))
        )
    )

    assert manager.discover() == []


def test_checkpoint_discovery_still_rejects_malformed_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "event" / "broken" / "checkpoint.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"not_a_checkpoint":true}', encoding="utf-8")
    manager = CheckpointManager(
        config=TrainingConfig(
            paths=TrainingPathConfig(checkpoint_directory=str(tmp_path))
        )
    )

    with pytest.raises(CheckpointLoadError):
        manager.discover()


def test_model_seed_is_atomic_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "image-models"
    destination = tmp_path / "volume-models"
    version_directory = source / "capacity_forecasting" / "model" / "1.0.0"
    version_directory.mkdir(parents=True)
    (version_directory / "model.pt").write_bytes(b"model")
    (version_directory / "artifact.json").write_text(
        '{"state":"promoted"}',
        encoding="utf-8",
    )
    script = Path("scripts/models/seed_models.sh").resolve()
    environment = {
        **os.environ,
        "OSHA_MODEL_SEED_DIRECTORY": str(source),
        "OSHA_AI_MODELS_DIRECTORY": str(destination),
    }

    subprocess.run(["bash", str(script)], check=True, env=environment)
    (destination / "capacity_forecasting/model/1.0.0/model.pt").write_bytes(
        b"volume-version"
    )
    subprocess.run(["bash", str(script)], check=True, env=environment)

    assert (
        destination / "capacity_forecasting/model/1.0.0/model.pt"
    ).read_bytes() == b"volume-version"


def test_training_suite_checks_every_input_before_starting(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="event_data"):
        _prepare_inputs(
            Namespace(
                data_root=str(tmp_path),
                event_data=None,
                anomaly_data=None,
                capacity_data=None,
            )
        )


def test_training_environment_sets_deterministic_cuda_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    environment = _training_environment(
        Namespace(
            data_root=str(tmp_path / "data"),
            checkpoints=str(tmp_path / "checkpoints"),
            artifacts=str(tmp_path / "artifacts"),
            models=str(tmp_path / "models"),
            logs=str(tmp_path / "logs"),
        )
    )

    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_training_environment_preserves_cuda_workspace_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    environment = _training_environment(
        Namespace(
            data_root=str(tmp_path / "data"),
            checkpoints=str(tmp_path / "checkpoints"),
            artifacts=str(tmp_path / "artifacts"),
            models=str(tmp_path / "models"),
            logs=str(tmp_path / "logs"),
        )
    )

    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


@pytest.mark.parametrize(
    "builder",
    (
        build_incident_samples,
        build_decision_samples,
        build_remediation_samples,
    ),
)
def test_operational_training_rejects_empty_real_data(
    tmp_path: Path,
    builder: Callable[[Path], list[OperationalSample]],
) -> None:
    with pytest.raises(EmptyTrainingDatasetError):
        builder(tmp_path)


def test_promotion_requires_finite_test_metrics_and_nonempty_holdout(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = tmp_path / "training_report.json"
    report.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one test sample"):
        _promotion_input(
            task=TrainingTask.EVENT.value,
            checkpoint_path=checkpoint,
            report_path=report,
            run_id="run-1",
            dataset_id="dataset-1",
            test_values={"loss": 0.1},
            test_sample_count=0,
            preferred_score="loss",
        )

    with pytest.raises(ValueError, match="non-finite"):
        _promotion_input(
            task=TrainingTask.EVENT.value,
            checkpoint_path=checkpoint,
            report_path=report,
            run_id="run-1",
            dataset_id="dataset-1",
            test_values={"loss": float("nan")},
            test_sample_count=1,
            preferred_score="loss",
        )
