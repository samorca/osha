#!/usr/bin/env python3
"""
OSHA environment preflight checker.

This script performs read-only checks against the local execution environment.

It does NOT:
    - install dependencies
    - modify configuration
    - create infrastructure resources
    - execute remediation
    - expose credential values

Usage:
    python scripts/check_environment.py

    python scripts/check_environment.py --strict

    python scripts/check_environment.py --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VENV_DIR = PROJECT_ROOT / ".venv"
CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"

REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"


# =============================================================================
# Project expectations
# =============================================================================

REQUIRED_DIRECTORIES = (
    "src",
    "config",
    "scripts",
)

RUNTIME_DIRECTORIES = (
    "artifacts",
    "checkpoints",
    "logs",
    "models",
)

REQUIRED_CONFIG_FILES = (
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

REQUIRED_PROJECT_FILES = (
    "pyproject.toml",
    "requirements.txt",
)

# Keep the dependency checks relatively conservative.
# These packages are useful indicators, but pyproject.toml /
# requirements.txt remain the authoritative dependency definitions.
PYTHON_PACKAGES = {
    "yaml": "PyYAML",
}

OPTIONAL_PYTHON_PACKAGES = {
    "torch": "PyTorch",
    "kubernetes": "Kubernetes Python client",
    "openstack": "OpenStack SDK",
    "minio": "MinIO Python client",
    "psutil": "psutil",
}

OPTIONAL_COMMANDS = (
    "git",
    "docker",
    "kubectl",
    "openstack",
    "ceph",
    "mc",
    "nvidia-smi",
)

SENSITIVE_ENVIRONMENT_MARKERS = (
    "PASSWORD",
    "PASSWD",
    "TOKEN",
    "SECRET",
    "PRIVATE_KEY",
    "API_KEY",
    "ACCESS_KEY",
    "AUTHORIZATION",
)

INFRASTRUCTURE_ENVIRONMENT_PREFIXES = (
    "OS_",
    "KUBECONFIG",
    "MINIO_",
    "CEPH_",
    "AWS_",
)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    critical: bool = False

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"

    @property
    def warning(self) -> bool:
        return self.status == "WARN"


class EnvironmentReport:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def pass_check(
        self,
        name: str,
        message: str,
        *,
        critical: bool = False,
    ) -> None:
        self.results.append(
            CheckResult(
                name=name,
                status="PASS",
                message=message,
                critical=critical,
            )
        )

    def warn(
        self,
        name: str,
        message: str,
        *,
        critical: bool = False,
    ) -> None:
        self.results.append(
            CheckResult(
                name=name,
                status="WARN",
                message=message,
                critical=critical,
            )
        )

    def fail(
        self,
        name: str,
        message: str,
        *,
        critical: bool = True,
    ) -> None:
        self.results.append(
            CheckResult(
                name=name,
                status="FAIL",
                message=message,
                critical=critical,
            )
        )

    @property
    def failures(self) -> list:
        return [item for item in self.results if item.failed]

    @property
    def warnings(self) -> list:
        return [item for item in self.results if item.warning]

    @property
    def critical_failures(self) -> list:
        return [
            item
            for item in self.results
            if item.failed and item.critical
        ]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the local environment is ready for OSHA."
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failure.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON.",
    )

    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip GPU/CUDA inspection.",
    )

    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")

    size = float(value)

    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"

        size /= 1024.0

    return f"{value} B"


def command_path(name: str) -> str | None:
    return shutil.which(name)


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def is_path_writable(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)

    parent = path.parent

    while not parent.exists() and parent != parent.parent:
        parent = parent.parent

    return parent.exists() and os.access(parent, os.W_OK)


def running_in_project_venv() -> bool:
    try:
        executable = Path(sys.executable).resolve()
        venv = VENV_DIR.resolve()
        executable.relative_to(venv)
        return True

    except ValueError:
        return False


# =============================================================================
# Python checks
# =============================================================================

def check_python(report: EnvironmentReport) -> None:
    version = sys.version_info

    report.pass_check(
        "python.version",
        (
            f"Python {version.major}.{version.minor}.{version.micro} "
            f"at {sys.executable}"
        ),
        critical=True,
    )

    if sys.implementation.name != "cpython":
        report.warn(
            "python.implementation",
            f"Interpreter is {sys.implementation.name}, not CPython.",
        )
    else:
        report.pass_check(
            "python.implementation",
            "CPython detected.",
        )


def check_virtual_environment(report: EnvironmentReport) -> None:
    if not VENV_DIR.exists():
        report.fail(
            "python.venv",
            f"Project virtual environment does not exist: {VENV_DIR}",
        )
        return

    if running_in_project_venv():
        report.pass_check(
            "python.venv",
            f"Project virtual environment is active: {VENV_DIR}",
            critical=True,
        )
        return

    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        report.warn(
            "python.venv",
            (
                "A virtual environment is active, but it is not the project's "
                f".venv. Interpreter: {sys.executable}"
            ),
        )
        return

    report.warn(
        "python.venv",
        (
            "Project .venv exists but is not active. "
            "Run: source .venv/bin/activate"
        ),
    )


def check_pip(report: EnvironmentReport) -> None:
    if module_available("pip"):
        report.pass_check(
            "python.pip",
            "pip is available.",
        )
    else:
        report.fail(
            "python.pip",
            "pip is unavailable in the current Python environment.",
        )


def check_python_packages(report: EnvironmentReport) -> None:
    for import_name, display_name in PYTHON_PACKAGES.items():
        if module_available(import_name):
            report.pass_check(
                f"package.{import_name}",
                f"{display_name} is installed.",
            )
        else:
            report.fail(
                f"package.{import_name}",
                f"{display_name} is not installed.",
            )

    for import_name, display_name in OPTIONAL_PYTHON_PACKAGES.items():
        if module_available(import_name):
            report.pass_check(
                f"package.{import_name}",
                f"{display_name} is installed.",
            )
        else:
            report.warn(
                f"package.{import_name}",
                f"{display_name} is not installed or not importable.",
            )


# =============================================================================
# Project structure
# =============================================================================

def check_project_structure(report: EnvironmentReport) -> None:
    for directory_name in REQUIRED_DIRECTORIES:
        path = PROJECT_ROOT / directory_name

        if path.is_dir():
            report.pass_check(
                f"directory.{directory_name}",
                f"{path} exists.",
            )
        else:
            report.fail(
                f"directory.{directory_name}",
                f"Required directory is missing: {path}",
            )

    for filename in REQUIRED_PROJECT_FILES:
        path = PROJECT_ROOT / filename

        if path.is_file():
            report.pass_check(
                f"file.{filename}",
                f"{filename} exists.",
            )
        else:
            report.fail(
                f"file.{filename}",
                f"Required project file is missing: {path}",
            )


# =============================================================================
# Configuration
# =============================================================================

def check_configuration_files(report: EnvironmentReport) -> None:
    if not CONFIG_DIR.is_dir():
        report.fail(
            "config.directory",
            f"Configuration directory is missing: {CONFIG_DIR}",
        )
        return

    for filename in REQUIRED_CONFIG_FILES:
        path = CONFIG_DIR / filename

        if not path.is_file():
            report.fail(
                f"config.{filename}",
                f"Missing configuration file: {path}",
            )
            continue

        try:
            if path.stat().st_size == 0:
                report.fail(
                    f"config.{filename}",
                    f"Configuration file is empty: {path}",
                )
            else:
                report.pass_check(
                    f"config.{filename}",
                    f"{filename} exists and is non-empty.",
                )

        except OSError as exc:
            report.fail(
                f"config.{filename}",
                f"Cannot inspect {path}: {exc}",
            )


# =============================================================================
# Runtime directories
# =============================================================================

def check_runtime_directories(report: EnvironmentReport) -> None:
    for directory_name in RUNTIME_DIRECTORIES:
        path = PROJECT_ROOT / directory_name

        if not path.exists():
            report.warn(
                f"runtime.{directory_name}",
                f"Runtime directory does not exist yet: {path}",
            )
            continue

        if not path.is_dir():
            report.fail(
                f"runtime.{directory_name}",
                f"Expected directory but found another file type: {path}",
            )
            continue

        if is_path_writable(path):
            report.pass_check(
                f"runtime.{directory_name}",
                f"{path} is writable.",
            )
        else:
            report.fail(
                f"runtime.{directory_name}",
                f"{path} is not writable.",
            )


# =============================================================================
# Storage
# =============================================================================

def check_disk(report: EnvironmentReport) -> None:
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)

    except OSError as exc:
        report.warn(
            "system.disk",
            f"Unable to inspect disk usage: {exc}",
        )
        return

    free_percentage = (
        (usage.free / usage.total) * 100.0
        if usage.total
        else 0.0
    )

    message = (
        f"Free {format_bytes(usage.free)} of "
        f"{format_bytes(usage.total)} ({free_percentage:.1f}% free)."
    )

    if usage.free < 1 * 1024**3:
        report.fail(
            "system.disk",
            message + " Less than 1 GiB remains.",
            critical=False,
        )

    elif free_percentage < 10.0:
        report.warn(
            "system.disk",
            message + " Free space is below 10%.",
        )

    else:
        report.pass_check(
            "system.disk",
            message,
        )


# =============================================================================
# Memory
# =============================================================================

def check_memory(report: EnvironmentReport) -> None:
    if not module_available("psutil"):
        report.warn(
            "system.memory",
            "psutil unavailable; detailed memory inspection skipped.",
        )
        return

    try:
        import psutil

        memory = psutil.virtual_memory()

        message = (
            f"Total {format_bytes(memory.total)}, "
            f"available {format_bytes(memory.available)} "
            f"({memory.available / memory.total * 100:.1f}% available)."
        )

        if memory.available < 512 * 1024**2:
            report.warn(
                "system.memory",
                message + " Available memory is low.",
            )
        else:
            report.pass_check(
                "system.memory",
                message,
            )

    except Exception as exc:
        report.warn(
            "system.memory",
            f"Unable to inspect memory: {exc}",
        )


# =============================================================================
# System information
# =============================================================================

def check_platform(report: EnvironmentReport) -> None:
    report.pass_check(
        "system.platform",
        (
            f"{platform.system()} "
            f"{platform.release()} "
            f"{platform.machine()}"
        ),
    )

    hostname = platform.node()

    if hostname:
        report.pass_check(
            "system.hostname",
            hostname,
        )
    else:
        report.warn(
            "system.hostname",
            "Unable to determine hostname.",
        )


# =============================================================================
# External commands
# =============================================================================

def check_commands(report: EnvironmentReport) -> None:
    for command in OPTIONAL_COMMANDS:
        path = command_path(command)

        if path:
            report.pass_check(
                f"command.{command}",
                f"{command} available at {path}.",
            )
        else:
            report.warn(
                f"command.{command}",
                f"{command} is not installed or not on PATH.",
            )


# =============================================================================
# GPU
# =============================================================================

def check_gpu(report: EnvironmentReport) -> None:
    if not module_available("torch"):
        report.warn(
            "gpu.torch",
            "PyTorch is unavailable; CUDA inspection skipped.",
        )
        return

    try:
        import torch

    except Exception as exc:
        report.warn(
            "gpu.torch",
            f"PyTorch could not be imported: {exc}",
        )
        return

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
        "            names.append('unknown')\n"
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
        report.warn(
            "gpu.cuda",
            "CUDA probe timed out after 15 seconds.",
        )
        return

    if completed.returncode != 0:
        report.warn(
            "gpu.cuda",
            f"PyTorch CUDA probe aborted with exit code {completed.returncode}.",
        )
        return

    try:
        result = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        report.warn(
            "gpu.cuda",
            "PyTorch CUDA probe returned an invalid result.",
        )
        return

    cuda_available = bool(result.get("available"))
    if not cuda_available:
        report.warn(
            "gpu.cuda",
            (
                "CUDA is not available to PyTorch or the native "
                "CUDA probe could not complete. "
                "CPU inference/training may still be possible."
            ),
        )
        return

    count = int(result.get("count", 0))
    report.pass_check(
        "gpu.cuda",
        f"CUDA available with {count} visible device(s).",
    )

    names = result.get("names", ())
    for index in range(count):
        name = names[index] if index < len(names) else "unknown"
        report.pass_check(
            f"gpu.cuda.{index}",
            f"GPU {index}: {name}",
        )


# =============================================================================
# Environment variables
# =============================================================================

def is_sensitive_environment_name(name: str) -> bool:
    upper_name = name.upper()

    return any(
        marker in upper_name
        for marker in SENSITIVE_ENVIRONMENT_MARKERS
    )


def check_environment_variables(
    report: EnvironmentReport,
) -> None:
    detected: list[str] = []

    for name in os.environ:
        if any(
            name.startswith(prefix)
            for prefix in INFRASTRUCTURE_ENVIRONMENT_PREFIXES
        ):
            detected.append(name)

    if detected:
        safe_names = sorted(
            name
            for name in detected
            if not is_sensitive_environment_name(name)
        )

        sensitive_count = len(detected) - len(safe_names)

        description_parts: list[str] = []

        if safe_names:
            description_parts.append(
                "non-sensitive variable names: "
                + ", ".join(safe_names)
            )

        if sensitive_count:
            description_parts.append(
                f"{sensitive_count} sensitive-named variable(s) detected "
                "and intentionally hidden"
            )

        report.pass_check(
            "environment.infrastructure",
            "; ".join(description_parts),
        )
    else:
        report.warn(
            "environment.infrastructure",
            (
                "No recognized infrastructure environment variables "
                "were detected. This is acceptable if credentials and "
                "provider configuration come from another mechanism."
            ),
        )


# =============================================================================
# Source import visibility
# =============================================================================

def check_source_visibility(report: EnvironmentReport) -> None:
    if not SRC_DIR.is_dir():
        report.fail(
            "python.src",
            f"Source directory does not exist: {SRC_DIR}",
        )
        return

    source_path = str(SRC_DIR.resolve())

    if source_path in sys.path:
        report.pass_check(
            "python.src",
            "src directory is present on Python import path.",
        )
        return

    project_path = str(PROJECT_ROOT.resolve())

    if project_path in sys.path:
        report.pass_check(
            "python.src",
            "Project root is present on Python import path.",
        )
        return

    report.warn(
        "python.src",
        (
            "Neither project root nor src is explicitly present on "
            "the Python import path. Editable project installation may "
            "be required depending on package layout."
        ),
    )


# =============================================================================
# Git
# =============================================================================

def check_git_repository(report: EnvironmentReport) -> None:
    git_directory = PROJECT_ROOT / ".git"

    if git_directory.exists():
        report.pass_check(
            "git.repository",
            "Git repository metadata detected.",
        )
    else:
        report.warn(
            "git.repository",
            "No .git directory detected at project root.",
        )


# =============================================================================
# Report output
# =============================================================================

def print_human_report(report: EnvironmentReport) -> None:
    print()
    print("OSHA Environment Check")
    print("=" * 72)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print()

    for result in report.results:
        print(
            f"[{result.status:<4}] "
            f"{result.name:<36} "
            f"{result.message}"
        )

    print()
    print("=" * 72)

    passed = sum(
        1 for result in report.results
        if result.status == "PASS"
    )

    warnings = len(report.warnings)
    failures = len(report.failures)

    print(f"PASS : {passed}")
    print(f"WARN : {warnings}")
    print(f"FAIL : {failures}")

    print()

    if report.critical_failures:
        print("[FAILED] Critical environment checks failed.")

    elif failures:
        print("[FAILED] Environment checks failed.")

    elif warnings:
        print(
            "[OK] Environment passed required checks "
            "with warnings."
        )

    else:
        print("[OK] Environment is ready.")


def print_json_report(report: EnvironmentReport) -> None:
    payload: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "python": sys.executable,
        "platform": platform.platform(),
        "summary": {
            "passed": sum(
                1
                for item in report.results
                if item.status == "PASS"
            ),
            "warnings": len(report.warnings),
            "failures": len(report.failures),
            "critical_failures": len(
                report.critical_failures
            ),
        },
        "checks": [
            asdict(item)
            for item in report.results
        ],
    }

    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()

    report = EnvironmentReport()

    check_python(report)
    check_virtual_environment(report)
    check_pip(report)

    check_platform(report)

    check_project_structure(report)
    check_configuration_files(report)
    check_runtime_directories(report)

    check_source_visibility(report)

    check_python_packages(report)

    check_disk(report)
    check_memory(report)

    check_commands(report)

    if not args.skip_gpu:
        check_gpu(report)

    check_environment_variables(report)
    check_git_repository(report)

    if args.json_output:
        print_json_report(report)
    else:
        print_human_report(report)

    if report.failures:
        return 1

    if args.strict and report.warnings:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())