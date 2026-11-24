#!/usr/bin/env bash

#
# OSHA anomaly-model evaluation entrypoint.
#
# Expected authoritative implementation:
#
#     src/training/evaluate_anomalies.py
#
# Usage:
#
#   ./scripts/training/evaluate_anomaly.sh --help
#   ./scripts/training/evaluate_anomaly.sh --wrapper-info
#   ./scripts/training/evaluate_anomaly.sh <evaluator arguments...>
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This wrapper does not implement:
#
#   checkpoint loading
#   model construction
#   dataset loading
#   accelerator selection
#   anomaly metrics
#   threshold evaluation
#   report generation
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

EVALUATION_MODULE="src.training.evaluate_anomalies"

EVALUATION_FILE="${PROJECT_ROOT}/src/training/evaluate_anomalies.py"


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


# -------------------------------------------------------------------
# Project
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -d "${PROJECT_ROOT}/src/training" ]]; then
    failure \
        "src/training directory does not exist."

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
    failure \
        "No Python interpreter found."

    exit 2
fi


if [[ ! -x "${PYTHON}" ]]; then
    failure \
        "Python interpreter is not executable: ${PYTHON}"

    exit 2
fi


export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


# -------------------------------------------------------------------
# Wrapper inspection
# -------------------------------------------------------------------

if (($# == 1)) \
    && [[ "$1" == "--wrapper-info" ]]; then
    printf '\n%sOSHA Anomaly Evaluation Wrapper%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Project root:      %s\n' \
        "${PROJECT_ROOT}"

    printf 'Python:            %s\n' \
        "${PYTHON}"

    printf 'Python version:    %s\n' \
        "${PYTHON_VERSION}"

    printf 'Expected module:   %s\n' \
        "${EVALUATION_MODULE}"

    printf 'Expected source:   %s\n' \
        "${EVALUATION_FILE}"

    printf 'Source exists:     %s\n' \
        "$(
            if [[ -f "${EVALUATION_FILE}" ]]; then
                printf 'yes'
            else
                printf 'no'
            fi
        )"

    if [[ -f "${EVALUATION_FILE}" ]]; then
        set +e

        HAS_MAIN="$(
            "${PYTHON}" - <<'PY'
from __future__ import annotations

import src.training.evaluate_anomalies as module


print(
    "yes"
    if callable(
        getattr(
            module,
            "main",
            None,
        )
    )
    else "no"
)
PY
        )"

        IMPORT_EXIT=$?

        set -e

        if ((IMPORT_EXIT == 0)); then
            printf 'CLI main():        %s\n' \
                "${HAS_MAIN}"
        else
            printf 'CLI main():        import failed\n'
        fi

    else
        printf 'CLI main():        unavailable\n'
    fi

    exit 0
fi


# -------------------------------------------------------------------
# Evaluator existence
# -------------------------------------------------------------------

if [[ ! -f "${EVALUATION_FILE}" ]]; then
    failure \
        "Anomaly evaluation implementation does not exist: ${EVALUATION_FILE}"

    cat >&2 <<'EOF'

The evaluation implementation belongs in:

    src/training/evaluate_anomalies.py

The shell wrapper must not implement model loading,
thresholding, metrics, or checkpoint interpretation.

Expected Python entrypoint:

    def main() -> None:
        ...

    if __name__ == "__main__":
        main()

Once implemented, this wrapper executes:

    python -m src.training.evaluate_anomalies

EOF

    exit 2
fi


# -------------------------------------------------------------------
# Compile
# -------------------------------------------------------------------

if ! "${PYTHON}" -m py_compile \
    "${EVALUATION_FILE}"
then
    failure \
        "src/training/evaluate_anomalies.py failed Python compilation."

    exit 1
fi


# -------------------------------------------------------------------
# Import
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import src.training.evaluate_anomalies' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.training.evaluate_anomalies."

    printf '\nRun for the full traceback:\n\n' >&2

    printf '  %q -c %q\n\n' \
        "${PYTHON}" \
        'import src.training.evaluate_anomalies'

    exit 1
fi


# -------------------------------------------------------------------
# Dependency checks
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import torch' \
    >/dev/null 2>&1
then
    failure \
        "PyTorch cannot be imported."

    exit 1
fi


if ! "${PYTHON}" -c \
    'from src.accelerators import AcceleratorManager' \
    >/dev/null 2>&1
then
    failure \
        "AcceleratorManager cannot be imported."

    exit 1
fi


if ! "${PYTHON}" -c \
    'from src.training.anomaly_dataset import AnomalyDataset' \
    >/dev/null 2>&1
then
    failure \
        "AnomalyDataset cannot be imported."

    exit 1
fi


if ! "${PYTHON}" -c \
    'from src.training.anomaly_model import InfrastructureAnomalyModel' \
    >/dev/null 2>&1
then
    failure \
        "InfrastructureAnomalyModel cannot be imported."

    exit 1
fi


# -------------------------------------------------------------------
# CLI entrypoint
# -------------------------------------------------------------------

HAS_MAIN="$(
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import src.training.evaluate_anomalies as module


print(
    "1"
    if callable(
        getattr(
            module,
            "main",
            None,
        )
    )
    else "0"
)
PY
)"


if [[ "${HAS_MAIN}" != "1" ]]; then
    failure \
        "src.training.evaluate_anomalies does not expose a callable main()."

    exit 2
fi


# -------------------------------------------------------------------
# Runtime summary
# -------------------------------------------------------------------

if [[ -t 1 ]]; then
    RUNTIME_SUMMARY="$(
        "${PYTHON}" - <<'PY'
from __future__ import annotations

import torch


hip = getattr(
    torch.version,
    "hip",
    None,
)

cuda = torch.version.cuda


if hip:
    backend = (
        "rocm:"
        + str(hip)
    )

elif cuda:
    backend = (
        "cuda:"
        + str(cuda)
    )

else:
    backend = "cpu"


try:
    available = bool(
        torch.cuda.is_available()
    )

except Exception:
    available = False


try:
    count = int(
        torch.cuda.device_count()
    )

except Exception:
    count = 0


print(
    f"{backend}; "
    f"accelerator_available={available}; "
    f"runtime_devices={count}"
)
PY
    )"

    printf '\n%sOSHA Anomaly Model Evaluation%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Python: ${PYTHON_VERSION}"
    info "Runtime: ${RUNTIME_SUMMARY}"
    info "Evaluator: ${EVALUATION_MODULE}"

    success \
        "Anomaly evaluator preflight checks passed"

    info \
        "Delegating arguments to ${EVALUATION_MODULE}"
fi


# -------------------------------------------------------------------
# Execute
# -------------------------------------------------------------------

exec \
    "${PYTHON}" \
    -m "${EVALUATION_MODULE}" \
    "$@"
