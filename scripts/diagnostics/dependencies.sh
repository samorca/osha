#!/usr/bin/env bash

#
# OSHA dependency diagnostics.
#
# Usage:
#
#   ./scripts/diagnostics/dependencies.sh
#   ./scripts/diagnostics/dependencies.sh --verbose
#   ./scripts/diagnostics/dependencies.sh --strict
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This script is read-only.
#
# It validates:
#
#   Python/package manager availability
#   pip dependency consistency
#   required Python imports
#   optional Python imports
#   development tooling
#   project dependency metadata
#   important external CLI availability
#   OSHA package-boundary imports
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
STRICT=false

REQUIRED_FAILURES=0
OPTIONAL_MISSING=0
CHECKS=0

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
# Logging
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
  scripts/diagnostics/dependencies.sh [options]

Options:
  --verbose
      Show additional dependency information.

  --strict
      Treat missing optional development dependencies
      as failures.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override Python interpreter.

Examples:
  ./scripts/diagnostics/dependencies.sh

  ./scripts/diagnostics/dependencies.sh --verbose

  ./scripts/diagnostics/dependencies.sh --strict
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

        --strict)
            STRICT=true
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
# Project
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -d "${PROJECT_ROOT}/src" ]]; then
    failure \
        "src/ directory not found: ${PROJECT_ROOT}/src"

    exit 2
fi


# -------------------------------------------------------------------
# Python discovery
# -------------------------------------------------------------------

if [[ -n "${OSHA_PYTHON:-}" ]]; then
    PYTHON="${OSHA_PYTHON}"

elif [[ -x "${VENV_DIR}/bin/python3" ]]; then
    PYTHON="${VENV_DIR}/bin/python3"

elif [[ -x "${VENV_DIR}/bin/python" ]]; then
    PYTHON="${VENV_DIR}/bin/python"

elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(
        command -v python3
    )"

else
    failure "No Python interpreter found."

    exit 2
fi


if [[ ! -x "${PYTHON}" ]]; then
    failure \
        "Python interpreter is not executable: ${PYTHON}"

    exit 2
fi


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

printf '\n%sOSHA Dependency Diagnostics%s\n' \
    "${BOLD}" \
    "${RESET}"

info "Project root: ${PROJECT_ROOT}"
info "Python: ${PYTHON}"
info "Python version: ${PYTHON_VERSION}"

if [[ "${STRICT}" == true ]]; then
    info "Mode: strict"
else
    info "Mode: normal"
fi


# -------------------------------------------------------------------
# Virtual environment
# -------------------------------------------------------------------

section "Python environment"

CHECKS=$((CHECKS + 1))


if "${PYTHON}" - <<'PY'
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit(1)
PY
then
    success "Python is running inside a virtual environment"

else
    warn "Selected Python is not running inside a virtual environment"

    if [[ "${STRICT}" == true ]]; then
        REQUIRED_FAILURES=$((REQUIRED_FAILURES + 1))
    fi
fi


printf 'sys.executable: '

"${PYTHON}" -c \
    'import sys; print(sys.executable)'


printf 'sys.prefix:     '

"${PYTHON}" -c \
    'import sys; print(sys.prefix)'


# -------------------------------------------------------------------
# pip
# -------------------------------------------------------------------

section "pip"

CHECKS=$((CHECKS + 1))


if "${PYTHON}" -m pip \
    --version \
    >/dev/null 2>&1
then
    success "pip available"

    "${PYTHON}" -m pip \
        --version

else
    failure "pip is unavailable"

    REQUIRED_FAILURES=$((REQUIRED_FAILURES + 1))
fi


# -------------------------------------------------------------------
# pip dependency consistency
# -------------------------------------------------------------------

section "Dependency consistency"

CHECKS=$((CHECKS + 1))


set +e

PIP_CHECK_OUTPUT="$(
    "${PYTHON}" -m pip check 2>&1
)"

PIP_CHECK_EXIT=$?

set -e


if ((PIP_CHECK_EXIT == 0)); then
    success "pip dependency consistency"

    if [[ -n "${PIP_CHECK_OUTPUT}" ]]; then
        printf '%s\n' \
            "${PIP_CHECK_OUTPUT}"
    fi

else
    failure "pip dependency consistency"

    printf '%s\n' \
        "${PIP_CHECK_OUTPUT}" >&2

    REQUIRED_FAILURES=$((REQUIRED_FAILURES + 1))
fi


# -------------------------------------------------------------------
# Installed distribution count
# -------------------------------------------------------------------

section "Installed Python distributions"


PACKAGE_COUNT="$(
    "${PYTHON}" -m pip list \
        --format=json \
    | "${PYTHON}" -c \
        'import json, sys; print(len(json.load(sys.stdin)))'
)"


printf 'Installed distributions: %s\n' \
    "${PACKAGE_COUNT}"


if [[ "${VERBOSE}" == true ]]; then
    printf '\n'

    "${PYTHON}" -m pip list
fi


# -------------------------------------------------------------------
# Required Python packages
# -------------------------------------------------------------------

section "Required Python packages"


REQUIRED_MODULES=(
    torch
    fastapi
    pydantic
    uvicorn
)


for module in "${REQUIRED_MODULES[@]}"; do
    CHECKS=$((CHECKS + 1))

    set +e

    RESULT="$(
        OSHA_DEPENDENCY_MODULE="${module}" \
        "${PYTHON}" - <<'PY' 2>&1
from __future__ import annotations

import importlib
import importlib.metadata
import os


name = os.environ[
    "OSHA_DEPENDENCY_MODULE"
]


module = importlib.import_module(
    name
)


version = getattr(
    module,
    "__version__",
    None,
)


if version is None:
    try:
        version = (
            importlib.metadata.version(
                name
            )
        )

    except importlib.metadata.PackageNotFoundError:
        version = "installed"


print(
    str(version)
)
PY
    )"

    EXIT_CODE=$?

    set -e


    if ((EXIT_CODE == 0)); then
        printf '%s[PASS]%s %-18s %s\n' \
            "${GREEN}" \
            "${RESET}" \
            "${module}" \
            "${RESULT}"

    else
        printf '%s[FAIL]%s %-18s import failed\n' \
            "${RED}" \
            "${RESET}" \
            "${module}" >&2

        if [[ "${VERBOSE}" == true ]]; then
            printf '%s\n' \
                "${RESULT}" >&2
        fi

        REQUIRED_FAILURES=$((REQUIRED_FAILURES + 1))
    fi
done


# -------------------------------------------------------------------
# Development dependencies
# -------------------------------------------------------------------

section "Development Python packages"


DEVELOPMENT_MODULES=(
    pytest
    ruff
    mypy
)


for module in "${DEVELOPMENT_MODULES[@]}"; do
    CHECKS=$((CHECKS + 1))

    set +e

    RESULT="$(
        OSHA_DEPENDENCY_MODULE="${module}" \
        "${PYTHON}" - <<'PY' 2>&1
from __future__ import annotations

import importlib
import importlib.metadata
import os


name = os.environ[
    "OSHA_DEPENDENCY_MODULE"
]


module = importlib.import_module(
    name
)


version = getattr(
    module,
    "__version__",
    None,
)


if version is None:
    try:
        version = importlib.metadata.version(
            name
        )

    except importlib.metadata.PackageNotFoundError:
        version = "installed"


print(
    str(version)
)
PY
    )"

    EXIT_CODE=$?

    set -e


    if ((EXIT_CODE == 0)); then
        printf '%s[PASS]%s %-18s %s\n' \
            "${GREEN}" \
            "${RESET}" \
            "${module}" \
            "${RESULT}"

    else
        printf '%s[WARN]%s %-18s missing/import failure\n' \
            "${YELLOW}" \
            "${RESET}" \
            "${module}" >&2

        OPTIONAL_MISSING=$((OPTIONAL_MISSING + 1))

        if [[ "${STRICT}" == true ]]; then
            REQUIRED_FAILURES=$((REQUIRED_FAILURES + 1))
        fi
    fi
done


# -------------------------------------------------------------------
# Common optional runtime integrations
# -------------------------------------------------------------------

section "Optional runtime integrations"


OPTIONAL_MODULES=(
    numpy
    psutil
    yaml
    httpx
    requests
    kubernetes
    openstack
)


for module in "${OPTIONAL_MODULES[@]}"; do
    if OSHA_DEPENDENCY_MODULE="${module}" \
        "${PYTHON}" - <<'PY' \
        >/dev/null 2>&1
import importlib
import os

importlib.import_module(
    os.environ[
        "OSHA_DEPENDENCY_MODULE"
    ]
)
PY
    then
        printf '%s[INFO]%s %-18s available\n' \
            "${BLUE}" \
            "${RESET}" \
            "${module}"

    else
        printf '%s[INFO]%s %-18s not installed\n' \
            "${BLUE}" \
            "${RESET}" \
            "${module}"
    fi
done


# -------------------------------------------------------------------
# PyTorch backend dependencies
# -------------------------------------------------------------------

section "PyTorch backend"


set +e

TORCH_INFO="$(
    "${PYTHON}" - <<'PY' 2>&1
from __future__ import annotations

import torch


print(
    "version="
    + str(
        torch.__version__
    )
)

print(
    "cuda_runtime="
    + str(
        torch.version.cuda
    )
)

print(
    "hip_runtime="
    + str(
        getattr(
            torch.version,
            "hip",
            None,
        )
    )
)
PY
)"

TORCH_EXIT=$?

set -e


if ((TORCH_EXIT == 0)); then
    success "PyTorch runtime import"

    printf '%s\n' \
        "${TORCH_INFO}"

else
    failure "PyTorch runtime import"

    printf '%s\n' \
        "${TORCH_INFO}" >&2

    REQUIRED_FAILURES=$((REQUIRED_FAILURES + 1))
fi


# -------------------------------------------------------------------
# OSHA package imports
# -------------------------------------------------------------------

section "OSHA package dependencies"


OSHA_MODULES=(
    src
    src.accelerators
    src.training
    src.ai
    src.api
)


for module in "${OSHA_MODULES[@]}"; do
    CHECKS=$((CHECKS + 1))

    if OSHA_DEPENDENCY_MODULE="${module}" \
        "${PYTHON}" - <<'PY' \
        >/dev/null 2>&1
import importlib
import os


importlib.import_module(
    os.environ[
        "OSHA_DEPENDENCY_MODULE"
    ]
)
PY
    then
        printf '%s[PASS]%s %-24s importable\n' \
            "${GREEN}" \
            "${RESET}" \
            "${module}"

    else
        printf '%s[FAIL]%s %-24s import failed\n' \
            "${RED}" \
            "${RESET}" \
            "${module}" >&2

        REQUIRED_FAILURES=$((REQUIRED_FAILURES + 1))
    fi
done


# -------------------------------------------------------------------
# Project dependency files
# -------------------------------------------------------------------

section "Dependency definition files"


DEPENDENCY_FILES_FOUND=0


for file in \
    pyproject.toml \
    requirements.txt \
    requirements-dev.txt \
    requirements/dev.txt \
    requirements/test.txt \
    setup.cfg \
    setup.py
do
    if [[ -f "${PROJECT_ROOT}/${file}" ]]; then
        printf '%-28s present\n' \
            "${file}"

        DEPENDENCY_FILES_FOUND=$((DEPENDENCY_FILES_FOUND + 1))
    fi
done


if ((DEPENDENCY_FILES_FOUND == 0)); then
    warn \
        "No standard Python dependency-definition file was found"
fi


# -------------------------------------------------------------------
# Editable installation
# -------------------------------------------------------------------

section "Editable installations"


EDITABLE_JSON="$(
    "${PYTHON}" -m pip list \
        --editable \
        --format=json
)"


EDITABLE_COUNT="$(
    printf '%s' "${EDITABLE_JSON}" \
    | "${PYTHON}" -c \
        'import json, sys; print(len(json.load(sys.stdin)))'
)"


printf 'Editable distributions: %s\n' \
    "${EDITABLE_COUNT}"


if [[ "${VERBOSE}" == true ]] \
    && ((EDITABLE_COUNT > 0))
then
    printf '%s' "${EDITABLE_JSON}" \
    | "${PYTHON}" -m json.tool
fi


# -------------------------------------------------------------------
# External command dependencies
# -------------------------------------------------------------------

section "External command dependencies"


REQUIRED_COMMANDS=(
    bash
    git
)


OPTIONAL_COMMANDS=(
    curl
    docker
    kubectl
    openstack
    nvidia-smi
    rocm-smi
)


for command_name in "${REQUIRED_COMMANDS[@]}"; do
    CHECKS=$((CHECKS + 1))

    if command -v "${command_name}" \
        >/dev/null 2>&1
    then
        printf '%s[PASS]%s %-16s %s\n' \
            "${GREEN}" \
            "${RESET}" \
            "${command_name}" \
            "$(
                command -v \
                    "${command_name}"
            )"

    else
        printf '%s[FAIL]%s %-16s missing\n' \
            "${RED}" \
            "${RESET}" \
            "${command_name}" >&2

        REQUIRED_FAILURES=$((REQUIRED_FAILURES + 1))
    fi
done


for command_name in "${OPTIONAL_COMMANDS[@]}"; do
    if command -v "${command_name}" \
        >/dev/null 2>&1
    then
        printf '%s[INFO]%s %-16s %s\n' \
            "${BLUE}" \
            "${RESET}" \
            "${command_name}" \
            "$(
                command -v \
                    "${command_name}"
            )"

    else
        printf '%s[INFO]%s %-16s %s\n' \
            "${BLUE}" \
            "${RESET}" \
            "${command_name}" \
            "<not found>"
    fi
done


# -------------------------------------------------------------------
# Existing CI validation scripts
# -------------------------------------------------------------------

section "OSHA validation tools"


VALIDATION_SCRIPTS=(
    scripts/ci/compile.sh
    scripts/ci/check_imports.sh
    scripts/ci/test.sh
    scripts/ci/verify.sh
    scripts/dev/lint.sh
    scripts/dev/format.sh
    scripts/dev/typecheck.sh
)


for script in "${VALIDATION_SCRIPTS[@]}"; do
    if [[ -x "${PROJECT_ROOT}/${script}" ]]; then
        printf '%s[PASS]%s %s\n' \
            "${GREEN}" \
            "${RESET}" \
            "${script}"

    elif [[ -f "${PROJECT_ROOT}/${script}" ]]; then
        printf '%s[WARN]%s %s exists but is not executable\n' \
            "${YELLOW}" \
            "${RESET}" \
            "${script}"

        OPTIONAL_MISSING=$((OPTIONAL_MISSING + 1))

    else
        printf '%s[INFO]%s %s not implemented\n' \
            "${BLUE}" \
            "${RESET}" \
            "${script}"
    fi
done


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Dependency Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Python:            %s\n' \
    "${PYTHON_VERSION}"

printf 'Packages:          %s\n' \
    "${PACKAGE_COUNT}"

printf 'Checks:            %s\n' \
    "${CHECKS}"

printf 'Required failures: %s\n' \
    "${REQUIRED_FAILURES}"

printf 'Optional missing:  %s\n' \
    "${OPTIONAL_MISSING}"

printf 'Mode:              %s\n' \
    "$(
        if [[ "${STRICT}" == true ]]; then
            printf 'strict'
        else
            printf 'normal'
        fi
    )"

printf 'Duration:          %ss\n' \
    "${elapsed}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


# -------------------------------------------------------------------
# Final result
# -------------------------------------------------------------------

if ((REQUIRED_FAILURES > 0)); then
    printf '\n%sDEPENDENCY CHECK FAILED%s\n' \
        "${RED}" \
        "${RESET}"

    exit 1
fi


if ((OPTIONAL_MISSING > 0)); then
    printf '\n%sDEPENDENCY CHECK PASSED WITH WARNINGS%s\n' \
        "${YELLOW}" \
        "${RESET}"

else
    printf '\n%sDEPENDENCY CHECK PASSED%s\n' \
        "${GREEN}" \
        "${RESET}"
fi


exit 0
