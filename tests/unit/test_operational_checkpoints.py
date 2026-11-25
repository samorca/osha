import pytest
import torch

from src.training.operational_models import (
    checkpoint_manager,
    save_operational_checkpoint,
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
