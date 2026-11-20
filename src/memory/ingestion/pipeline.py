from .redactors.secrets import redact_sensitive_data
from .validators.telemetry import validate_telemetry
from .normalizers.telemetry import normalize_telemetry

from ..schemas.telemetry import Telemetry


class TelemetryIngestionPipeline:

    def process(self, raw_record: dict) -> Telemetry:
        if not isinstance(raw_record, dict):
            raise TypeError("raw_record must be a dictionary")
        validated = validate_telemetry(raw_record)
        redacted = redact_sensitive_data(validated)
        return normalize_telemetry(redacted)

    def process_batch(self, raw_records: list[dict]) -> list[Telemetry]:
        if not isinstance(raw_records, list):
            raise TypeError("raw_records must be a list")
        return [self.process(record) for record in raw_records]