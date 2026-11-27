#!/usr/bin/env bash

#
# OSHA development/runtime environment diagnostics.
#
# Usage:
#
#   ./scripts/diagnostics/environment.sh
#   ./scripts/diagnostics/environment.sh --verbose
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This script is read-only.
#
# It intentionally does not print the complete
# process environment because environment variables
# may contain passwords, tokens, credentials, or
# connection strings.
#

set -euo pipefail


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" \
        >/dev/null 2>&1
    pwd -P
)"

PROJECT_ROOT="$(
    cd -- "${SCRIPT_DIR}/../.." \
        >/dev/null 2>&1
    pwd -P
)"

VENV_DIR="${PROJECT_ROOT}/.venv"


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

VERBOSE=false

START_SECONDS="${SECONDS}"


# -------------------------------------------------------------------
# Colours
# -------------------------------------------------------------------

if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    RESET=$'\033[0m'
    GREEN=$'\033[32m'
    RED=$'\033[31m'
    YELLOW=$'\033[33m'
    BLUE=$'\033[34m'
    BOLD=$'\033[1m'
else
    RESET=""
    GREEN=""
    RED=""
    YELLOW=""
    BLUE=""
    BOLD=""
fi


# -------------------------------------------------------------------
# Output
# -------------------------------------------------------------------

info() {
    printf '%s[INFO]%s %s\n' \
        "${BLUE}" \
        "${RESET}" \
        "$*"
}


success() {
    printf '%s[PASS]%s %s\n' \
        "${GREEN}" \
        "${RESET}" \
        "$*"
}


warn() {
    printf '%s[WARN]%s %s\n' \
        "${YELLOW}" \
        "${RESET}" \
        "$*" >&2
}


failure() {
    printf '%s[FAIL]%s %s\n' \
        "${RED}" \
        "${RESET}" \
        "$*" >&2
}


section() {
    printf '\n%s%s== %s ==%s\n' \
        "${BOLD}" \
        "${BLUE}" \
        "$*" \
        "${RESET}"
}


# -------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  scripts/diagnostics/environment.sh [options]

Options:
  --verbose
      Show additional safe environment details.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override the Python interpreter.

Examples:
  ./scripts/diagnostics/environment.sh
  ./scripts/diagnostics/environment.sh --verbose
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --verbose)
            VERBOSE=true
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac

    shift
done


# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


printf '\n%sOSHA Environment Diagnostics%s\n' \
    "${BOLD}" \
    "${RESET}"


# -------------------------------------------------------------------
# Repository
# -------------------------------------------------------------------

section "Repository"

printf 'Project root: %s\n' \
    "${PROJECT_ROOT}"

printf 'Working dir:  %s\n' \
    "$(pwd)"


if [[ -d "${PROJECT_ROOT}/src" ]]; then
    success "src/ exists"
else
    failure "src/ is missing"
fi


if [[ -d "${PROJECT_ROOT}/scripts" ]]; then
    success "scripts/ exists"
else
    failure "scripts/ is missing"
fi


if [[ -d "${PROJECT_ROOT}/tests" ]]; then
    success "tests/ exists"
else
    warn "tests/ does not currently exist"
fi


# -------------------------------------------------------------------
# Git
# -------------------------------------------------------------------

section "Git"


if command -v git >/dev/null 2>&1; then
    success "git available"

    GIT_VERSION="$(
        git --version
    )"

    printf 'Version: %s\n' \
        "${GIT_VERSION}"

    if git rev-parse \
        --is-inside-work-tree \
        >/dev/null 2>&1
    then
        BRANCH="$(
            git branch \
                --show-current \
                2>/dev/null \
            || true
        )"

        COMMIT="$(
            git rev-parse \
                --short HEAD \
                2>/dev/null \
            || true
        )"

        if [[ -n "${BRANCH}" ]]; then
            printf 'Branch:  %s\n' \
                "${BRANCH}"
        fi

        if [[ -n "${COMMIT}" ]]; then
            printf 'Commit:  %s\n' \
                "${COMMIT}"
        fi

        if git diff \
            --quiet \
            --ignore-submodules \
            HEAD \
            -- \
            2>/dev/null
        then
            printf 'Tree:    clean\n'
        else
            printf 'Tree:    modified\n'
        fi

    else
        warn "Project root is not inside a Git work tree"
    fi

else
    warn "git is not installed"
fi


# -------------------------------------------------------------------
# Python selection
# -------------------------------------------------------------------

section "Python"


if [[ -n "${OSHA_PYTHON:-}" ]]; then
    PYTHON="${OSHA_PYTHON}"
    PYTHON_SOURCE="OSHA_PYTHON"

elif [[ -x "${VENV_DIR}/bin/python3" ]]; then
    PYTHON="${VENV_DIR}/bin/python3"
    PYTHON_SOURCE=".venv/bin/python3"

elif [[ -x "${VENV_DIR}/bin/python" ]]; then
    PYTHON="${VENV_DIR}/bin/python"
    PYTHON_SOURCE=".venv/bin/python"

elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(
        command -v python3
    )"

    PYTHON_SOURCE="PATH"

else
    failure "No Python interpreter found"
    exit 2
fi


if [[ ! -x "${PYTHON}" ]]; then
    failure \
        "Python is not executable: ${PYTHON}"

    exit 2
fi


printf 'Interpreter: %s\n' \
    "${PYTHON}"

printf 'Selected by: %s\n' \
    "${PYTHON_SOURCE}"


"${PYTHON}" - <<'PY'
from __future__ import annotations

import os
import platform
import sys


print(
    "Version:     "
    + platform.python_version()
)

print(
    "Executable:  "
    + sys.executable
)

print(
    "Implementation: "
    + platform.python_implementation()
)

print(
    "Prefix:      "
    + sys.prefix
)

print(
    "Base prefix: "
    + sys.base_prefix
)

print(
    "Virtualenv:  "
    + str(
        sys.prefix != sys.base_prefix
    )
)

print(
    "Byte order:  "
    + sys.byteorder
)

print(
    "Optimization: "
    + str(sys.flags.optimize)
)
PY


# -------------------------------------------------------------------
# Virtual environment
# -------------------------------------------------------------------

section "Virtual environment"


if [[ -d "${VENV_DIR}" ]]; then
    success ".venv directory exists"

    printf 'Path: %s\n' \
        "${VENV_DIR}"

    if [[ -x "${VENV_DIR}/bin/python" ]]; then
        VENV_VERSION="$(
            "${VENV_DIR}/bin/python" \
                -c \
                'import platform; print(platform.python_version())'
        )"

        printf 'Python: %s\n' \
            "${VENV_VERSION}"
    fi

else
    warn ".venv does not exist"
fi


if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    printf 'Activated: yes\n'
    printf 'VIRTUAL_ENV: %s\n' \
        "${VIRTUAL_ENV}"

    if [[ "${VIRTUAL_ENV}" != "${VENV_DIR}" ]]; then
        warn \
            "Activated virtualenv differs from project .venv"
    fi

else
    printf 'Activated: no\n'
fi


# -------------------------------------------------------------------
# Platform
# -------------------------------------------------------------------

section "Platform"


"${PYTHON}" - <<'PY'
from __future__ import annotations

import platform


print(
    "System:       "
    + platform.system()
)

print(
    "Release:      "
    + platform.release()
)

print(
    "Machine:      "
    + platform.machine()
)

print(
    "Architecture: "
    + platform.architecture()[0]
)

print(
    "Platform:     "
    + platform.platform()
)
PY


# -------------------------------------------------------------------
# Python package locations
# -------------------------------------------------------------------

section "Python package environment"


"${PYTHON}" - <<'PY'
from __future__ import annotations

import site
import sys


print(
    "sys.path entries: "
    + str(len(sys.path))
)


site_packages = []

try:
    site_packages.extend(
        site.getsitepackages()
    )
except AttributeError:
    pass


user_site = site.getusersitepackages()

if user_site:
    site_packages.append(
        user_site
    )


for value in dict.fromkeys(
    site_packages
):
    print(
        "site-packages: "
        + str(value)
    )
PY


# -------------------------------------------------------------------
# Core project dependencies
# -------------------------------------------------------------------

section "Core Python dependencies"


"${PYTHON}" - <<'PY'
from __future__ import annotations

import importlib
import importlib.util


PACKAGES = (
    "torch",
    "fastapi",
    "pydantic",
    "uvicorn",
    "pytest",
    "ruff",
    "mypy",
)


for name in PACKAGES:
    spec = importlib.util.find_spec(
        name
    )

    if spec is None:
        print(
            f"{name:<12} missing"
        )

        continue

    try:
        module = importlib.import_module(
            name
        )

        version = getattr(
            module,
            "__version__",
            "installed",
        )

    except Exception as exc:
        version = (
            "import-error:"
            + type(exc).__name__
        )

    print(
        f"{name:<12} {version}"
    )
PY


# -------------------------------------------------------------------
# PyTorch runtime
# -------------------------------------------------------------------

section "PyTorch runtime"


"${PYTHON}" - <<'PY'
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys


if importlib.util.find_spec(
    "torch"
) is None:
    print(
        "PyTorch: unavailable"
    )

    raise SystemExit(0)


import torch


print(
    "PyTorch version: "
    + str(torch.__version__)
)

print(
    "CUDA runtime:    "
    + str(
        torch.version.cuda
    )
)

print(
    "ROCm/HIP runtime:"
    " "
    + str(
        getattr(
            torch.version,
            "hip",
            None,
        )
    )
)

probe = (
    "import json, torch\n"
    "try:\n"
    "    available = bool(torch.cuda.is_available())\n"
    "    count = int(torch.cuda.device_count()) if available else 0\n"
    "    names = []\n"
    "    for index in range(count):\n"
    "        try:\n"
    "            names.append(str(torch.cuda.get_device_name(index)))\n"
    "        except Exception as exc:\n"
    "            names.append('<unavailable:' + type(exc).__name__ + '>')\n"
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
    print("torch.cuda available: <probe timed out>")
    print("Runtime device count: unavailable")
else:
    if completed.returncode != 0:
        print("torch.cuda available: <native probe aborted>")
        print("Runtime device count: unavailable")
    else:
        try:
            result = json.loads(completed.stdout.strip())
        except json.JSONDecodeError:
            print("torch.cuda available: <invalid probe result>")
            print("Runtime device count: unavailable")
        else:
            print(
                "torch.cuda available: "
                + str(bool(result.get("available")))
            )
            print(
                "Runtime device count: "
                + str(result.get("count", 0))
            )
            for index, name in enumerate(result.get("names", ())):
                print(f"runtime device {index}: {name}")
PY


# -------------------------------------------------------------------
# OSHA package imports
# -------------------------------------------------------------------

section "OSHA package boundaries"


export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"


"${PYTHON}" - <<'PY'
from __future__ import annotations

import importlib


MODULES = (
    "src",
    "src.accelerators",
    "src.training",
    "src.ai",
    "src.api",
)


failures = 0


for name in MODULES:
    try:
        importlib.import_module(
            name
        )

        print(
            f"{name:<24} OK"
        )

    except Exception as exc:
        failures += 1

        print(
            f"{name:<24} "
            f"FAILED "
            f"({type(exc).__name__})"
        )


if failures:
    raise SystemExit(1)
PY


success "OSHA package boundaries"


# -------------------------------------------------------------------
# API configuration
# -------------------------------------------------------------------

section "OSHA API configuration"


if [[ -f "${PROJECT_ROOT}/src/api/config.py" ]]; then
    "${PYTHON}" - <<'PY'
from __future__ import annotations

from src.api.config import (
    from_environment,
)


try:
    config = from_environment()

except Exception as exc:
    print(
        "API configuration: INVALID"
    )

    print(
        "Error type: "
        + type(exc).__name__
    )

    raise SystemExit(1)


print(
    "Service:     "
    + str(config.service_name)
)

print(
    "Environment: "
    + str(config.environment)
)

print(
    "API prefix:  "
    + str(config.versioned_prefix)
)

print(
    "Host:        "
    + str(config.server.host)
)

print(
    "Port:        "
    + str(config.server.port)
)

print(
    "CORS:        "
    + str(config.cors.enabled)
)

print(
    "Inference:   "
    + str(
        config.features.
        inference_enabled
    )
)

print(
    "Live remediation: "
    + str(
        config.features.
        remediation_live_execution_enabled
    )
)
PY

    success "API configuration"

else
    warn "src/api/config.py not found"
fi


# -------------------------------------------------------------------
# Project directories
# -------------------------------------------------------------------

section "Project directories"


DIRECTORIES=(
    "src"
    "scripts"
    "artifacts"
    "configs"
    "tests"
    "dataset"
)


for directory in "${DIRECTORIES[@]}"; do
    if [[ -d "${PROJECT_ROOT}/${directory}" ]]; then
        printf '%-24s present\n' \
            "${directory}/"

    else
        printf '%-24s absent\n' \
            "${directory}/"
    fi
done


# -------------------------------------------------------------------
# Safe environment information
# -------------------------------------------------------------------

section "Selected environment variables"


safe_env() {
    local name="$1"
    local value="${!name:-}"

    if [[ -n "${value}" ]]; then
        printf '%-24s %s\n' \
            "${name}" \
            "${value}"

    else
        printf '%-24s %s\n' \
            "${name}" \
            "<unset>"
    fi
}


safe_env "VIRTUAL_ENV"
safe_env "PYTHONPATH"
safe_env "CUDA_VISIBLE_DEVICES"
safe_env "HIP_VISIBLE_DEVICES"
safe_env "ROCR_VISIBLE_DEVICES"


if [[ "${VERBOSE}" == true ]]; then
    safe_env "PATH"
fi


printf '\n'
info \
    "Sensitive environment variables are intentionally not displayed."


# -------------------------------------------------------------------
# Shell tools
# -------------------------------------------------------------------

section "Developer commands"


COMMANDS=(
    git
    bash
    curl
    docker
    kubectl
    openstack
    nvidia-smi
    rocm-smi
)


for command_name in "${COMMANDS[@]}"; do
    if command -v "${command_name}" \
        >/dev/null 2>&1
    then
        printf '%-16s %s\n' \
            "${command_name}" \
            "$(
                command -v \
                    "${command_name}"
            )"

    else
        printf '%-16s %s\n' \
            "${command_name}" \
            "<not found>"
    fi
done


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Environment Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Project:  %s\n' \
    "${PROJECT_ROOT}"

printf 'Python:   %s\n' \
    "$(
        "${PYTHON}" -c \
            'import platform; print(platform.python_version())'
    )"

printf 'Venv:     %s\n' \
    "$(
        if [[ -d "${VENV_DIR}" ]]; then
            printf 'present'
        else
            printf 'absent'
        fi
    )"

printf 'Duration: %ss\n' \
    "${elapsed}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


printf '\n%sENVIRONMENT DIAGNOSTICS COMPLETE%s\n' \
    "${GREEN}" \
    "${RESET}"

exit 0
