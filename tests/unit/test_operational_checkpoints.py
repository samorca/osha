import pytest
import torch

from src.training.operational_models import (
    OperationalSample,
    checkpoint_manager,
    save_operational_checkpoint,
    select_training_device,
    split_training_validation,
)
from src.training.schema import TrainingTask


@pytest.mark.parametrize(
    "task",
    [
        TrainingTask.INCIDENT_ROOT_CAUSE.value,
        TrainingTask.DECISION_POLICY_RISK.value,
        TrainingTask.REMEDIATION_RECOMMENDATION.value,
    ],
)
def test_operational_checkpoint_is_discoverable(tmp_path, task):
    manager = checkpoint_manager(tmp_path)
    model = torch.nn.Linear(16, 2)

    saved_path = save_operational_checkpoint(
        manager,
        task=task,
        model_name=task,
        model=model,
        score=0.75,
        model_config={"input_dim": 16},
        labels={},
        extra={"test": True},
    )

    discovered = manager.discover()

    assert len(discovered) == 1
    assert discovered[0].path == saved_path
    assert discovered[0].state == "saved"
    assert task in saved_path


def test_training_validation_split_is_disjoint_and_deterministic():
    samples = [
        OperationalSample(
            features=[float(index)] * 16,
            target=index,
        )
        for index in range(20)
    ]

    training, validation = split_training_validation(
        samples,
        seed=17,
    )
    repeated_training, repeated_validation = split_training_validation(
        samples,
        seed=17,
    )

    training_targets = {sample.target for sample in training}
    validation_targets = {sample.target for sample in validation}
    sample_targets = {sample.target for sample in samples}

    assert training_targets.isdisjoint(validation_targets)
    assert training_targets | validation_targets == sample_targets
    assert training == repeated_training
    assert validation == repeated_validation


def test_cuda_selection_fails_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        select_training_device("cuda")

    assert select_training_device("cpu") == torch.device("cpu")
