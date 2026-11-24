#!/usr/bin/env python3
"""
OSHA runtime health checker.

Performs read-only health checks for the local OSHA installation.

This script is suitable for:
    - manual health checks
    - Docker HEALTHCHECK commands
    - service supervisors
    - CI/CD smoke checks
    - operational diagnostics

It does NOT:
    - modify infrastructure
    - execute remediation
    - install packages
    - modify configuration
    - expose credentials

Exit codes:
    0 - healthy
    1 - degraded
    2 - unhealthy
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"


# =============================================================================
# Constants
# =============================================================================

REQUIRED_CONFIGS = (
    "base.yaml",
    "production.yaml",
    "logging.yaml",
    "models.yaml",
    "collectors.yaml",
    "correlation.yaml",
    "ai.yaml",
    "remediation.yaml",
    "security.yaml",
    "observability.yaml",
    "infrastructure.yaml",
)

RUNTIME_DIRECTORIES = (
    ARTIFACTS_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    MODELS_DIR,
)

MINIMUM_DISK_WARNING_BYTES = 5 * 1024**3
MINIMUM_DISK_CRITICAL_BYTES = 1 * 1024**3


# =============================================================================
# Health status
# =============================================================================

class ExitCode(IntEnum):
    HEALTHY = 0
    DEGRADED = 1
    UNHEALTHY = 2


@dataclass
class HealthCheck:
    name: str
    status: str
    message: str
    critical: bool = False
    duration_ms: float = 0.0

    @property
    def healthy(self) -> bool:
        return self.status == "healthy"

    @property
    def degraded(self) -> bool:
        return self.status == "degraded"

    @property
    def unhealthy(self) -> bool:
        return self.status == "unhealthy"


class HealthReport:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.checks: list[HealthCheck] = []

    def add(
        self,
        name: str,
        status: str,
        message: str,
        *,
        critical: bool = False,
        duration_ms: float = 0.0,
    ) -> None:
        self.checks.append(
            HealthCheck(
                name=name,
                status=status,
                message=message,
                critical=critical,
                duration_ms=round(duration_ms, 3),
            )
        )

    def healthy(
        self,
        name: str,
        message: str,
        *,
        critical: bool = False,
        duration_ms: float = 0.0,
    ) -> None:
        self.add(
            name,
            "healthy",
            message,
            critical=critical,
            duration_ms=duration_ms,
        )

    def degraded(
        self,
        name: str,
        message: str,
        *,
        critical: bool = False,
        duration_ms: float = 0.0,
    ) -> None:
        self.add(
            name,
            "degraded",
            message,
            critical=critical,
            duration_ms=duration_ms,
        )

    def unhealthy(
        self,
        name: str,
        message: str,
        *,
        critical: bool = True,
        duration_ms: float = 0.0,
    ) -> None:
        self.add(
            name,
            "unhealthy",
            message,
            critical=critical,
            duration_ms=duration_ms,
        )

    @property
    def duration_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def overall_status(self) -> str:
        if any(
            check.unhealthy and check.critical
            for check in self.checks
        ):
            return "unhealthy"

        if any(check.unhealthy for check in self.checks):
            return "degraded"

        if any(check.degraded for check in self.checks):
            return "degraded"

        return "healthy"

    @property
    def exit_code(self) -> ExitCode:
        status = self.overall_status

        if status == "healthy":
            return ExitCode.HEALTHY

        if status == "degraded":
            return ExitCode.DEGRADED

        return ExitCode.UNHEALTHY


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check OSHA runtime health."
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output health information as JSON.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only overall status.",
    )

    parser.add_argument(
        "--readiness",
        action="store_true",
        help="Run readiness-oriented health checks.",
    )

    parser.add_argument(
        "--liveness",
        action="store_true",
        help="Run minimal liveness-oriented checks.",
    )

    parser.add_argument(
        "--require-models",
        action="store_true",
        help="Treat absence of model files as unhealthy.",
    )

    return parser.parse_args()


# =============================================================================
# Timing helper
# =============================================================================

class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.elapsed_ms = (
            time.perf_counter() - self.start
        ) * 1000.0


# =============================================================================
# Generic helpers
# =============================================================================

def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def format_bytes(value: int) -> str:
    size = float(value)

    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"

        size /= 1024.0

    return f"{value} B"


def count_files(path: Path) -> int:
    if not path.is_dir():
        return 0

    try:
        return sum(
            1
            for entry in path.rglob("*")
            if entry.is_file()
        )
    except OSError:
        return 0


# =============================================================================
# Process health
# =============================================================================

def check_process(report: HealthReport) -> None:
    with Timer() as timer:
        pid = os.getpid()

        executable = sys.executable

    report.healthy(
        "process",
        f"Health checker running with PID {pid} using {executable}.",
        critical=True,
        duration_ms=timer.elapsed_ms,
    )


# =============================================================================
# Source health
# =============================================================================

def check_source(report: HealthReport) -> None:
    with Timer() as timer:
        exists = SRC_DIR.is_dir()

    if exists:
        report.healthy(
            "source",
            f"Source directory available: {SRC_DIR}",
            critical=True,
            duration_ms=timer.elapsed_ms,
        )
    else:
        report.unhealthy(
            "source",
            f"Source directory missing: {SRC_DIR}",
            critical=True,
            duration_ms=timer.elapsed_ms,
        )


# =============================================================================
# Configuration health
# =============================================================================

def check_configuration(report: HealthReport) -> None:
    with Timer() as timer:
        if not CONFIG_DIR.is_dir():
            report.unhealthy(
                "configuration",
                f"Configuration directory missing: {CONFIG_DIR}",
                critical=True,
                duration_ms=timer.elapsed_ms,
            )
            return

        missing = [
            filename
            for filename in REQUIRED_CONFIGS
            if not (CONFIG_DIR / filename).is_file()
        ]

    if missing:
        report.unhealthy(
            "configuration",
            "Missing configuration files: "
            + ", ".join(missing),
            critical=True,
            duration_ms=timer.elapsed_ms,
        )
        return

    if yaml is None:
        report.degraded(
            "configuration",
            "Configuration files exist but PyYAML is unavailable.",
            critical=False,
            duration_ms=timer.elapsed_ms,
        )
        return

    errors: list[str] = []

    with Timer() as parse_timer:
        for filename in REQUIRED_CONFIGS:
            path = CONFIG_DIR / filename

            try:
                with path.open(
                    "r",
                    encoding="utf-8",
                ) as handle:
                    data = yaml.safe_load(handle)

                if not isinstance(data, dict):
                    errors.append(
                        f"{filename}: root must be a mapping"
                    )

            except Exception as exc:
                errors.append(
                    f"{filename}: {exc}"
                )

    if errors:
        report.unhealthy(
            "configuration",
            "Invalid configuration: "
            + "; ".join(errors),
            critical=True,
            duration_ms=parse_timer.elapsed_ms,
        )
        return

    report.healthy(
        "configuration",
        (
            f"{len(REQUIRED_CONFIGS)} required configuration "
            "files loaded successfully."
        ),
        critical=True,
        duration_ms=parse_timer.elapsed_ms,
    )


# =============================================================================
# Runtime directories
# =============================================================================

def check_runtime_storage(
    report: HealthReport,
) -> None:
    failures: list[str] = []
    degraded: list[str] = []

    with Timer() as timer:
        for path in RUNTIME_DIRECTORIES:
            if not path.exists():
                degraded.append(
                    f"{path.name}: missing"
                )
                continue

            if not path.is_dir():
                failures.append(
                    f"{path.name}: not a directory"
                )
                continue

            if not os.access(path, os.W_OK):
                failures.append(
                    f"{path.name}: not writable"
                )

    if failures:
        report.unhealthy(
            "runtime.storage",
            "; ".join(failures),
            critical=True,
            duration_ms=timer.elapsed_ms,
        )
        return

    if degraded:
        report.degraded(
            "runtime.storage",
            "; ".join(degraded),
            duration_ms=timer.elapsed_ms,
        )
        return

    report.healthy(
        "runtime.storage",
        "Runtime directories are available and writable.",
        duration_ms=timer.elapsed_ms,
    )


# =============================================================================
# Disk health
# =============================================================================

def check_disk(report: HealthReport) -> None:
    with Timer() as timer:
        try:
            usage = shutil.disk_usage(PROJECT_ROOT)
        except OSError as exc:
            report.degraded(
                "disk",
                f"Unable to inspect disk usage: {exc}",
                duration_ms=timer.elapsed_ms,
            )
            return

    message = (
        f"{format_bytes(usage.free)} free of "
        f"{format_bytes(usage.total)}."
    )

    if usage.free < MINIMUM_DISK_CRITICAL_BYTES:
        report.unhealthy(
            "disk",
            message + " Free disk space is critically low.",
            critical=True,
            duration_ms=timer.elapsed_ms,
        )

    elif usage.free < MINIMUM_DISK_WARNING_BYTES:
        report.degraded(
            "disk",
            message + " Free disk space is low.",
            duration_ms=timer.elapsed_ms,
        )

    else:
        report.healthy(
            "disk",
            message,
            duration_ms=timer.elapsed_ms,
        )


# =============================================================================
# Model health
# =============================================================================

def check_models(
    report: HealthReport,
    *,
    require_models: bool,
) -> None:
    with Timer() as timer:
        model_count = count_files(MODELS_DIR)
        checkpoint_count = count_files(CHECKPOINTS_DIR)

    message = (
        f"{model_count} model file(s), "
        f"{checkpoint_count} checkpoint file(s) detected."
    )

    if model_count == 0 and checkpoint_count == 0:
        if require_models:
            report.unhealthy(
                "models",
                message + " Models were explicitly required.",
                critical=True,
                duration_ms=timer.elapsed_ms,
            )
        else:
            report.degraded(
                "models",
                message,
                duration_ms=timer.elapsed_ms,
            )

        return

    report.healthy(
        "models",
        message,
        duration_ms=timer.elapsed_ms,
    )


# =============================================================================
# Python dependency health
# =============================================================================

def check_runtime_dependencies(
    report: HealthReport,
) -> None:
    required = {
        "yaml": "PyYAML",
    }

    missing: list[str] = []

    with Timer() as timer:
        for module, package in required.items():
            if not module_available(module):
                missing.append(package)

    if missing:
        report.unhealthy(
            "dependencies",
            "Missing required runtime package(s): "
            + ", ".join(missing),
            critical=True,
            duration_ms=timer.elapsed_ms,
        )
        return

    report.healthy(
        "dependencies",
        "Required health-check dependencies are importable.",
        duration_ms=timer.elapsed_ms,
    )


# =============================================================================
# Provider tooling
# =============================================================================

def check_provider_tools(
    report: HealthReport,
) -> None:
    providers = {
        "kubectl": "Kubernetes CLI",
        "openstack": "OpenStack CLI",
        "ceph": "Ceph CLI",
        "mc": "MinIO client",
    }

    available: list[str] = []
    missing: list[str] = []

    with Timer() as timer:
        for command, description in providers.items():
            if shutil.which(command):
                available.append(description)
            else:
                missing.append(description)

    if available:
        message = "Available: " + ", ".join(available)

        if missing:
            message += ". Unavailable: " + ", ".join(missing)

        report.healthy(
            "provider.tooling",
            message,
            duration_ms=timer.elapsed_ms,
        )
    else:
        report.degraded(
            "provider.tooling",
            (
                "No optional provider CLI tools detected. "
                "Python/API integrations may still be used."
            ),
            duration_ms=timer.elapsed_ms,
        )


# =============================================================================
# GPU/model runtime
# =============================================================================

def check_inference_device(
    report: HealthReport,
) -> None:
    if not module_available("torch"):
        report.degraded(
            "inference.device",
            "PyTorch is not importable.",
        )
        return

    with Timer() as timer:
        probe = (
            "import json, torch\n"
            "try:\n"
            "    available = bool(torch.cuda.is_available())\n"
            "    count = int(torch.cuda.device_count()) if available else 0\n"
            "    names = []\n"
            "    for index in range(count):\n"
            "        try:\n"
            "            names.append(str(torch.cuda.get_device_name(index)))\n"
            "        except Exception:\n"
            "            names.append(f'device-{index}')\n"
            "    print(json.dumps({'available': available, 'count': count, 'names': names}))\n"
            "except BaseException as exc:\n"
            "    print(json.dumps({'available': False, 'count': 0, 'names': [], 'error': type(exc).__name__}))\n"
        )

        try:
            completed = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except subprocess.TimeoutExpired:
            report.degraded(
                "inference.device",
                "PyTorch CUDA probe timed out after 15 seconds.",
                duration_ms=timer.elapsed_ms,
            )
            return

        if completed.returncode != 0:
            report.degraded(
                "inference.device",
                (
                    "PyTorch CUDA native probe aborted "
                    f"with exit code {completed.returncode}; "
                    "CPU fallback may still be usable."
                ),
                duration_ms=timer.elapsed_ms,
            )
            return

        try:
            result = json.loads(completed.stdout.strip())
        except json.JSONDecodeError:
            report.degraded(
                "inference.device",
                "PyTorch CUDA probe returned an invalid result.",
                duration_ms=timer.elapsed_ms,
            )
            return

        if bool(result.get("available")):
            devices = int(result.get("count", 0))
            names = result.get("names", ())
            report.healthy(
                "inference.device",
                (
                    f"CUDA available with {devices} device(s): "
                    + ", ".join(names)
                ),
                duration_ms=timer.elapsed_ms,
            )
        else:
            report.healthy(
                "inference.device",
                "PyTorch available; using CPU-capable runtime.",
                duration_ms=timer.elapsed_ms,
            )


# =============================================================================
# Main runtime artifact health
# =============================================================================

def check_artifacts(
    report: HealthReport,
) -> None:
    with Timer() as timer:
        if not ARTIFACTS_DIR.exists():
            report.degraded(
                "artifacts",
                f"Artifact directory missing: {ARTIFACTS_DIR}",
                duration_ms=timer.elapsed_ms,
            )
            return

        count = count_files(ARTIFACTS_DIR)

    report.healthy(
        "artifacts",
        f"Artifact storage accessible; {count} file(s) detected.",
        duration_ms=timer.elapsed_ms,
    )


# =============================================================================
# Liveness
# =============================================================================

def perform_liveness_checks(
    report: HealthReport,
) -> None:
    check_process(report)
    check_source(report)
    check_disk(report)


# =============================================================================
# Readiness
# =============================================================================

def perform_readiness_checks(
    report: HealthReport,
    *,
    require_models: bool,
) -> None:
    check_process(report)
    check_source(report)
    check_configuration(report)
    check_runtime_dependencies(report)
    check_runtime_storage(report)
    check_disk(report)
    check_models(
        report,
        require_models=require_models,
    )
    check_inference_device(report)
    check_provider_tools(report)
    check_artifacts(report)


# =============================================================================
# Full health check
# =============================================================================

def perform_full_checks(
    report: HealthReport,
    *,
    require_models: bool,
) -> None:
    perform_readiness_checks(
        report,
        require_models=require_models,
    )


# =============================================================================
# Reporting
# =============================================================================

def summary_counts(
    report: HealthReport,
) -> dict[str, int]:
    return {
        "healthy": sum(
            1
            for check in report.checks
            if check.healthy
        ),
        "degraded": sum(
            1
            for check in report.checks
            if check.degraded
        ),
        "unhealthy": sum(
            1
            for check in report.checks
            if check.unhealthy
        ),
    }


def print_human_report(
    report: HealthReport,
) -> None:
    print()
    print("OSHA Health Check")
    print("=" * 78)

    for check in report.checks:
        status = check.status.upper()

        print(
            f"[{status:<9}] "
            f"{check.name:<24} "
            f"{check.message} "
            f"({check.duration_ms:.1f} ms)"
        )

    counts = summary_counts(report)

    print()
    print("=" * 78)
    print(f"Overall   : {report.overall_status.upper()}")
    print(f"Healthy   : {counts['healthy']}")
    print(f"Degraded  : {counts['degraded']}")
    print(f"Unhealthy : {counts['unhealthy']}")
    print(
        f"Duration  : "
        f"{report.duration_seconds:.3f} seconds"
    )
    print(
        f"Exit code : {int(report.exit_code)}"
    )


def print_json_report(
    report: HealthReport,
) -> None:
    payload: dict[str, Any] = {
        "service": "osha",
        "status": report.overall_status,
        "exit_code": int(report.exit_code),
        "duration_seconds": round(
            report.duration_seconds,
            6,
        ),
        "summary": summary_counts(report),
        "checks": [
            asdict(check)
            for check in report.checks
        ],
    }

    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )


def print_quiet_report(
    report: HealthReport,
) -> None:
    print(report.overall_status)


# =============================================================================
# Entry point
# =============================================================================

def main() -> int:
    args = parse_args()

    report = HealthReport()

    if args.liveness:
        perform_liveness_checks(report)

    elif args.readiness:
        perform_readiness_checks(
            report,
            require_models=args.require_models,
        )

    else:
        perform_full_checks(
            report,
            require_models=args.require_models,
        )

    if args.json_output:
        print_json_report(report)

    elif args.quiet:
        print_quiet_report(report)

    else:
        print_human_report(report)

    return int(report.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())