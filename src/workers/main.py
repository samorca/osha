from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time

from pathlib import Path
from typing import Any, Optional, Sequence

from src.database.runtime import create_database_pipeline


SUPPORTED_SOURCES = ("linux", "kubernetes")
shutdown_requested = False


def request_shutdown(signum: int, frame: object) -> None:
    del signum, frame
    global shutdown_requested
    shutdown_requested = True


def _environment_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def default_sources() -> tuple[str, ...]:
    configured = os.getenv("OSHA_COLLECTOR_SOURCES", "").strip()
    if configured:
        return parse_sources(configured)

    sources = ["linux"]
    kubeconfig = os.getenv("KUBECONFIG")
    default_kubeconfig = Path.home() / ".kube" / "config"
    if (
        os.getenv("KUBERNETES_SERVICE_HOST")
        or kubeconfig
        or default_kubeconfig.is_file()
    ):
        sources.append("kubernetes")
    return tuple(sources)


def parse_sources(value: str) -> tuple[str, ...]:
    sources = tuple(
        dict.fromkeys(
            part.strip().lower()
            for part in value.split(",")
            if part.strip()
        )
    )
    if not sources:
        raise ValueError("at least one collector source is required")

    unsupported = sorted(set(sources) - set(SUPPORTED_SOURCES))
    if unsupported:
        raise ValueError(
            "unsupported collector source(s): " + ", ".join(unsupported)
        )
    return sources


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return safe or "worker"


def build_command(
    source: str,
    *,
    output_directory: Path,
    worker_id: str,
    namespace: str,
    tail_lines: int,
    include_pod_logs: bool,
) -> list[str]:
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported collector source: {source}")

    output = output_directory / "live" / source / f"{_safe_component(worker_id)}.jsonl"
    command = [
        sys.executable,
        "-m",
        f"src.collectors.{source}.collector",
        "--output",
        str(output),
    ]

    if source == "kubernetes":
        if namespace.lower() != "all":
            command.extend(("--namespace", namespace))
        command.extend(("--tail-lines", str(tail_lines)))
        if not include_pod_logs:
            command.append("--skip-logs")

    return command


def _read_collector_output(
    output_path: Path,
    offset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records = []
    errors = []
    with output_path.open("rb") as output_file:
        output_file.seek(offset)
        for line in output_file:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("JSONL records must be objects")
            records.append(record)
            if record.get("signal_type") != "collector_error":
                continue

            payload = record.get("payload", {})
            if isinstance(payload, dict):
                errors.append(
                    {
                        key: payload[key]
                        for key in ("collector", "error", "message")
                        if isinstance(payload.get(key), str)
                    }
                )
    return records, errors


def collect_cycle(
    sources: Sequence[str],
    *,
    output_directory: Path,
    worker_id: str,
    namespace: str,
    tail_lines: int,
    include_pod_logs: bool,
    timeout_seconds: float,
    emit_output: bool = True,
    telemetry_repository: Any = None,
    environment_id: str = "local-dev",
    persisted_offsets: Optional[dict[str, int]] = None,
) -> dict:
    results = []
    for source in sources:
        command = build_command(
            source,
            output_directory=output_directory,
            worker_id=worker_id,
            namespace=namespace,
            tail_lines=tail_lines,
            include_pod_logs=include_pod_logs,
        )
        output_path = Path(command[command.index("--output") + 1])
        try:
            current_size = output_path.stat().st_size
            if telemetry_repository is not None and persisted_offsets is not None:
                output_offset = persisted_offsets.get(source, 0)
            else:
                output_offset = current_size
        except FileNotFoundError:
            current_size = 0
            output_offset = 0
        except OSError as exc:
            results.append(
                {
                    "source": source,
                    "status": "failed",
                    "error": type(exc).__name__,
                }
            )
            continue

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            results.append(
                {
                    "source": source,
                    "status": "failed",
                    "error": "timeout",
                }
            )
            continue
        except OSError as exc:
            results.append(
                {
                    "source": source,
                    "status": "failed",
                    "error": type(exc).__name__,
                }
            )
            continue

        if emit_output:
            if completed.stdout:
                print(completed.stdout, end="", flush=True)
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr, flush=True)

        if completed.returncode != 0:
            results.append(
                {
                    "source": source,
                    "status": "failed",
                    "exit_code": completed.returncode,
                }
            )
            continue

        try:
            records, collector_errors = _read_collector_output(
                output_path,
                output_offset,
            )
        except FileNotFoundError:
            results.append(
                {
                    "source": source,
                    "status": "failed",
                    "error": "output_not_created",
                }
            )
            continue
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            results.append(
                {
                    "source": source,
                    "status": "failed",
                    "error": "invalid_jsonl_output",
                    "message": str(exc),
                }
            )
            continue

        records_persisted = 0
        persistence_error = None
        if telemetry_repository is not None:
            try:
                records_persisted = telemetry_repository.store_batch(
                    records,
                    source=source,
                    worker_id=worker_id,
                    environment_id=environment_id,
                )
                if persisted_offsets is not None:
                    persisted_offsets[source] = output_path.stat().st_size
            except Exception as exc:
                persistence_error = type(exc).__name__

        source_result = {
            "source": source,
            "status": (
                "failed"
                if persistence_error
                else "degraded"
                if collector_errors
                else "success"
            ),
            "exit_code": completed.returncode,
            "records_collected": len(records),
        }
        if telemetry_repository is not None:
            source_result["records_persisted"] = records_persisted
        if persistence_error:
            source_result["persistence_error"] = persistence_error
        if collector_errors:
            source_result["collector_errors"] = collector_errors
        results.append(source_result)

    return {
        "worker_id": worker_id,
        "status": (
            "success"
            if all(item["status"] == "success" for item in results)
            else "degraded"
        ),
        "sources": results,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect read-only OSHA Linux and Kubernetes telemetry."
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.getenv("OSHA_COLLECTOR_INTERVAL_SECONDS", "30")),
    )
    parser.add_argument(
        "--sources",
        default=os.getenv("OSHA_COLLECTOR_SOURCES") or None,
        help=(
            "Comma-separated sources: linux,kubernetes "
            "(default: detect available Kubernetes credentials)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("OSHA_DATA_DIR", "orchestration-ai-data"),
    )
    parser.add_argument(
        "--worker-id",
        default=(
            os.getenv("OSHA_POD_NAME")
            or os.getenv("OSHA_WORKER_ID")
            or socket.gethostname()
        ),
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("OSHA_KUBERNETES_NAMESPACE", "osha"),
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=int(os.getenv("OSHA_KUBERNETES_TAIL_LINES", "200")),
    )
    parser.add_argument(
        "--skip-pod-logs",
        action="store_true",
        default=not _environment_bool(
            "OSHA_KUBERNETES_INCLUDE_POD_LOGS",
            default=True,
        ),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", action="store_true")

    arguments = parser.parse_args(argv)
    if arguments.interval < 0.001:
        parser.error("--interval must be >= 0.001")
    if arguments.tail_lines < 1:
        parser.error("--tail-lines must be >= 1")
    if arguments.timeout < 0.001:
        parser.error("--timeout must be >= 0.001")

    try:
        arguments.sources = (
            parse_sources(arguments.sources)
            if arguments.sources
            else default_sources()
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not arguments.namespace.strip():
        parser.error("--namespace cannot be empty; use 'all' for all namespaces")
    arguments.data_dir = Path(arguments.data_dir).expanduser().resolve()
    arguments.worker_id = _safe_component(arguments.worker_id)
    return arguments


def main(argv: Optional[Sequence[str]] = None) -> int:
    global shutdown_requested
    shutdown_requested = False
    arguments = parse_args(argv)

    if not _environment_bool("OSHA_COLLECTORS_ENABLED", default=True):
        print("Collector worker disabled by OSHA_COLLECTORS_ENABLED=false", flush=True)
        return 0
    if not _environment_bool("OSHA_TELEMETRY_ENABLED", default=True):
        print("Collector worker disabled by OSHA_TELEMETRY_ENABLED=false", flush=True)
        return 0

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    database = create_database_pipeline()
    try:
        database.start()
        telemetry_repository = database.telemetry_repository()
        telemetry_repository.ensure_schema()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": "database_start_failed",
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        database.stop()
        return 1

    try:
        cycle = 0
        persisted_offsets: dict[str, int] = {}
        while not shutdown_requested:
            cycle += 1
            started = time.monotonic()
            result = collect_cycle(
                arguments.sources,
                output_directory=arguments.data_dir,
                worker_id=arguments.worker_id,
                namespace=arguments.namespace,
                tail_lines=arguments.tail_lines,
                include_pod_logs=not arguments.skip_pod_logs,
                timeout_seconds=arguments.timeout,
                emit_output=not arguments.json,
                telemetry_repository=telemetry_repository,
                environment_id=os.getenv("OSHA_ENVIRONMENT_ID", "local-dev"),
                persisted_offsets=persisted_offsets,
            )
            result["cycle"] = cycle
            print(json.dumps(result, sort_keys=True), flush=True)

            if arguments.once:
                return 0 if result["status"] == "success" else 1

            remaining = arguments.interval - (time.monotonic() - started)
            if remaining > 0 and not shutdown_requested:
                time.sleep(remaining)

        return 0
    finally:
        database.stop()


if __name__ == "__main__":
    raise SystemExit(main())
