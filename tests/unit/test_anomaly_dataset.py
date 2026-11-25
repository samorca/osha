import pytest

from src.training.anomaly_dataset import (
    extract_legacy_features,
    extract_legacy_label,
)


@pytest.mark.parametrize(
    "label_field",
    (
        "label",
        "target",
        "anomaly",
        "anomaly_label",
        "is_anomaly",
    ),
)
def test_legacy_features_exclude_target_and_row_id(label_field: str) -> None:
    record = {
        "row_id": 17,
        "values": {
            "cpu_utilization": 0.75,
            label_field: 1,
        },
    }

    features = extract_legacy_features(record)

    assert features is not None
    assert features[0] == 0.75
    assert features[1:] == [0.0] * 15
    assert extract_legacy_label(record) == 1
