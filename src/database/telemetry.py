from __future__ import annotations

import hashlib
import json

from datetime import datetime, timezone
from threading import RLock
from typing import Any, Mapping, Sequence

from src.database.connection import (
    DatabaseConnection,
    PostgreSQLDatabaseHandle,
)
from src.database.errors import (
    DatabaseConnectionError,
    RepositoryValidationError,
    wrap_driver_error,
)


class TelemetryRepository:
    """Persist collector JSONL records in the configured database."""

    def __init__(self, connection: DatabaseConnection) -> None:
        self.connection = connection
        self._memory_lock = RLock()

    def ensure_schema(self) -> None:
        handle = self.connection.handle
        if isinstance(handle, dict):
            handle.setdefault("store", {}).setdefault("telemetry_records", {})
            return
        if not isinstance(handle, PostgreSQLDatabaseHandle):
            raise DatabaseConnectionError(
                "Telemetry persistence does not support this database adapter."
            )

        statement = """
            CREATE TABLE IF NOT EXISTS osha_telemetry_records (
                record_id TEXT PRIMARY KEY,
                environment_id TEXT NOT NULL,
                source TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                collected_at TIMESTAMPTZ NOT NULL,
                signal_type TEXT NOT NULL,
                record JSONB NOT NULL,
                inserted_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """
        with handle.lock:
            with handle.connection.transaction():
                handle.connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext('osha_telemetry_schema_v1'))"
                )
                handle.connection.execute(statement)
                handle.connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS osha_telemetry_lookup_idx
                    ON osha_telemetry_records (environment_id, source, collected_at)
                    """
                )

    def store_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        source: str,
        worker_id: str,
        environment_id: str,
    ) -> int:
        if not source.strip() or not worker_id.strip() or not environment_id.strip():
            raise RepositoryValidationError(
                "Telemetry source, worker_id, and environment_id are required."
            )

        handle = self.connection.handle
        normalized = [
            self._normalize(record, source, worker_id, environment_id)
            for record in records
        ]
        if not normalized:
            return 0

        if isinstance(handle, dict):
            with self._memory_lock:
                store = handle.setdefault("store", {}).setdefault(
                    "telemetry_records", {}
                )
                before = len(store)
                store.update({item[0]: item[1] for item in normalized})
                return len(store) - before

        if not isinstance(handle, PostgreSQLDatabaseHandle):
            raise DatabaseConnectionError(
                "Telemetry persistence does not support this database adapter."
            )

        statement = """
            INSERT INTO osha_telemetry_records (
                record_id, environment_id, source, worker_id,
                collected_at, signal_type, record
            )
            VALUES (%s, %s, %s, %s, %s::timestamptz, %s, %s::jsonb)
            ON CONFLICT (record_id) DO NOTHING
        """
        values = [
            (
                record_id,
                item["environment_id"],
                item["source"],
                item["worker_id"],
                item["collected_at"],
                item["signal_type"],
                json.dumps(item["record"], separators=(",", ":"), ensure_ascii=False),
            )
            for record_id, item in normalized
        ]
        try:
            with handle.lock:
                with handle.connection.transaction():
                    cursor = handle.connection.cursor()
                    cursor.executemany(statement, values)
                    return cursor.rowcount
        except Exception as exc:
            raise wrap_driver_error(
                exc,
                message="Telemetry batch could not be persisted.",
                retryable=True,
            ) from exc

    @staticmethod
    def _normalize(
        record: Mapping[str, Any],
        source: str,
        worker_id: str,
        environment_id: str,
    ) -> tuple[str, dict[str, Any]]:
        if not isinstance(record, Mapping):
            raise RepositoryValidationError("Telemetry records must be JSON objects.")
        try:
            encoded = json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            normalized_record = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise RepositoryValidationError(
                "Telemetry records must contain JSON-compatible values."
            ) from exc

        timestamp = (
            normalized_record.get("collected_at")
            or normalized_record.get("timestamp")
            or datetime.now(timezone.utc).isoformat()
        )
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise RepositoryValidationError(
                "Telemetry record timestamps must be non-empty strings."
            )
        try:
            parsed_timestamp = datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise RepositoryValidationError(
                "Telemetry record timestamps must use ISO 8601 format."
            ) from exc
        if parsed_timestamp.tzinfo is None:
            parsed_timestamp = parsed_timestamp.replace(tzinfo=timezone.utc)

        item = {
            "environment_id": environment_id,
            "source": source,
            "worker_id": worker_id,
            "collected_at": parsed_timestamp.isoformat(),
            "signal_type": str(normalized_record.get("signal_type") or "unknown"),
            "record": normalized_record,
        }
        identity = json.dumps(
            [environment_id, source, worker_id, normalized_record],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        record_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return record_id, item
