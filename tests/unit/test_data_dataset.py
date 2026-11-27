from src.data.dataset import Dataset, DatasetRecord


def test_dataset_builds_manifest_using_data_schema_contract() -> None:
    dataset = Dataset(
        name="telemetry-samples",
        records=[DatasetRecord(data={"cpu_percent": 42.0})],
    )

    manifest = dataset.build_manifest()

    assert manifest.source.kind == "generated"
    assert manifest.source.location == f"dataset://{dataset.dataset_id}"
    assert manifest.files[0].record_count == 1
    assert manifest.files[0].checksum == dataset.fingerprint()
