from __future__ import annotations

import json
import subprocess

from pathlib import Path

import pytest

from src.frameworks import FrameworkApplication
from src.memory.store import create_memory_store
from src.workers import main as telemetry_worker


def test_default_sources_detects_in_cluster_kubernetes(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OSHA_COLLECTOR_SOURCES", raising=False)
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.delenv("KUBECONFIG", raising=False)

    assert telemetry_worker.default_sources() == ("linux", "kubernetes")


def test_default_sources_uses_linux_only_without_kubernetes_config(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OSHA_COLLECTOR_SOURCES", raising=False)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.delenv("KUBECONFIG", raising=False)

    assert telemetry_worker.default_sources() == ("linux",)


def test_build_kubernetes_command_targets_namespace_and_keeps_logs(tmp_path):
    command = telemetry_worker.build_command(
        "kubernetes",
        output_directory=tmp_path,
        worker_id="worker-1",
        namespace="osha",
        tail_lines=200,
        include_pod_logs=True,
    )

    assert command[command.index("--namespace") + 1] == "osha"
    assert command[command.index("--tail-lines") + 1] == "200"
    assert "--skip-logs" not in command
    assert command[command.index("--output") + 1] == str(
        tmp_path / "live" / "kubernetes" / "worker-1.jsonl"
    )


def test_collect_cycle_reports_failure_and_continues(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        exit_code = 1 if "linux" in command[2] else 0
        if exit_code == 0:
            output_path = Path(command[command.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            args=command,
            returncode=exit_code,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(telemetry_worker.subprocess, "run", fake_run)

    result = telemetry_worker.collect_cycle(
        ("linux", "kubernetes"),
        output_directory=Path(tmp_path),
        worker_id="worker-1",
        namespace="osha",
        tail_lines=200,
        include_pod_logs=True,
        timeout_seconds=5,
        emit_output=False,
    )

    assert result["status"] == "degraded"
    assert [item["status"] for item in result["sources"]] == ["failed", "success"]
    assert len(commands) == 2


def test_collect_cycle_reports_collector_errors_as_degraded(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "signal_type": "collector_error",
                    "payload": {
                        "collector": "linux_services",
                        "error": "systemctl_not_found",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(telemetry_worker.subprocess, "run", fake_run)

    result = telemetry_worker.collect_cycle(
        ("linux",),
        output_directory=Path(tmp_path),
        worker_id="worker-1",
        namespace="osha",
        tail_lines=200,
        include_pod_logs=True,
        timeout_seconds=5,
        emit_output=False,
    )

    assert result["status"] == "degraded"
    assert result["sources"][0]["status"] == "degraded"
    assert result["sources"][0]["collector_errors"] == [
        {
            "collector": "linux_services",
            "error": "systemctl_not_found",
        }
    ]


def test_collect_cycle_persists_new_records(monkeypatch, tmp_path):
    class Repository:
        def __init__(self):
            self.batches = []

        def store_batch(self, records, **kwargs):
            self.batches.append((records, kwargs))
            return len(records)

    repository = Repository()

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "signal_type": "resource_state",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "payload": {"cpu_percent": 12.5},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(telemetry_worker.subprocess, "run", fake_run)

    result = telemetry_worker.collect_cycle(
        ("linux",),
        output_directory=Path(tmp_path),
        worker_id="worker-1",
        namespace="osha",
        tail_lines=200,
        include_pod_logs=True,
        timeout_seconds=5,
        emit_output=False,
        telemetry_repository=repository,
        environment_id="test",
    )

    assert result["status"] == "success"
    assert result["sources"][0]["records_persisted"] == 1
    assert repository.batches[0][1] == {
        "source": "linux",
        "worker_id": "worker-1",
        "environment_id": "test",
    }


def test_collect_cycle_reports_persistence_failure(monkeypatch, tmp_path):
    persisted_offsets = {}

    class Repository:
        def store_batch(self, records, **kwargs):
            raise RuntimeError("database unavailable")

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"signal_type":"metric"}\n', encoding="utf-8")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(telemetry_worker.subprocess, "run", fake_run)
    result = telemetry_worker.collect_cycle(
        ("linux",),
        output_directory=Path(tmp_path),
        worker_id="worker-1",
        namespace="osha",
        tail_lines=200,
        include_pod_logs=True,
        timeout_seconds=5,
        emit_output=False,
        telemetry_repository=Repository(),
        persisted_offsets=persisted_offsets,
    )

    assert result["status"] == "degraded"
    assert result["sources"][0]["status"] == "failed"
    assert result["sources"][0]["persistence_error"] == "RuntimeError"
    assert persisted_offsets == {}


def test_collect_cycle_processes_collected_records_with_frameworks(
    monkeypatch, tmp_path
):
    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "record_type": "linux_telemetry",
                    "provider": "linux",
                    "signal_type": "metric",
                    "timestamp": "2026-11-26T00:00:00Z",
                    "environment_id": "test",
                    "host_name": "node-1",
                    "payload": {
                        "name": "cpu_usage",
                        "value": 92.0,
                        "unit": "percent",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(telemetry_worker.subprocess, "run", fake_run)
    application = FrameworkApplication(
        process_role="test",
        memory_store=create_memory_store(
            backend="json",
            path=tmp_path / "memory.json",
        ),
    )
    try:
        result = telemetry_worker.collect_cycle(
            ("linux",),
            output_directory=tmp_path,
            worker_id="worker-1",
            namespace="osha",
            tail_lines=200,
            include_pod_logs=True,
            timeout_seconds=5,
            emit_output=False,
            environment_id="test",
            framework_application=application,
        )
    finally:
        application.close()

    assert result["status"] == "success"
    assert result["framework_processing"]["status"] == "completed"
    assert result["framework_processing"]["summary"]["input_record_count"] == 1


def test_parse_sources_rejects_unknown_sources():
    with pytest.raises(ValueError, match="unknown"):
        telemetry_worker.parse_sources("linux,unknown")
