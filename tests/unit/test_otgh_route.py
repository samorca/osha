from __future__ import annotations

import base64
from datetime import datetime, timezone

from src.api.routes.otgh import OTGHRecord, normalize_record


def test_normalize_otgh_json_record_preserves_payload_and_transport_metadata() -> None:
    record = OTGHRecord(
        id="record-1",
        type="event",
        priority=1,
        source={"system": "kubernetes", "component": "collector"},
        content_type="application/json",
        compression="none",
        created_at=datetime.now(timezone.utc),
        payload=base64.b64encode(b'{"message":"pod restarted"}').decode("ascii"),
        metadata={"cluster": "test"},
    )

    normalized = normalize_record(record)

    assert normalized["message"] == "pod restarted"
    assert normalized["otgh"]["record_id"] == "record-1"
    assert normalized["otgh"]["record_type"] == "event"
    assert normalized["otgh"]["source"]["system"] == "kubernetes"
    assert normalized["otgh"]["metadata"] == {"cluster": "test"}