#!/usr/bin/env bash

#
# OSHA anomaly-model training entrypoint.
#
# Authoritative implementation:
#
#     src/training/train_anomalies.py
#
# Execution path:
#
#     train_anomaly.sh
#          |
#          v
#     src.training.train_anomalies
#          |
#          v
#     TrainingPipeline
#          |
#          +--> anomaly dataset
#          +--> anomaly model
#          +--> AcceleratorManager
#          +--> optimizer
#          +--> checkpoints
#          +--> evaluation
#
# Usage:
#
#   ./scripts/training/train_anomaly.sh --help
#   ./scripts/training/train_anomaly.sh <trainer arguments...>
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# The shell wrapper does not own:
#
#   dataset loading
#   accelerator selection
#   PyTorch training loops
#   class weighting
#   checkpointing
#   evaluation
#   artifact generation
#
# Those responsibilities remain in:
#
#     src.training.train_anomalies
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

TRAIN_MODULE="src.training.train_anomalies"

TRAIN_FILE="${PROJECT_ROOT}/src/training/train_anomalies.py"


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


# -------------------------------------------------------------------
# Project validation
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -d "${PROJECT_ROOT}/src/training" ]]; then
    failure \
        "src/training directory does not exist."

    exit 2
fi


if [[ ! -f "${TRAIN_FILE}" ]]; then
    failure \
        "Anomaly trainer does not exist: ${TRAIN_FILE}"

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
# Compile trainer
# -------------------------------------------------------------------

if ! "${PYTHON}" -m py_compile \
    "${TRAIN_FILE}"
then
    failure \
        "src/training/train_anomalies.py failed Python compilation."

    printf '\nThe anomaly trainer must compile before training starts.\n' >&2
    printf 'Run:\n\n' >&2

    printf '  %q -m py_compile %q\n\n' \
        "${PYTHON}" \
        "${TRAIN_FILE}" >&2

    exit 1
fi


# -------------------------------------------------------------------
# Import trainer
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import src.training.train_anomalies' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.training.train_anomalies."

    printf '\nRun this for the complete traceback:\n\n' >&2

    printf '  %q -c %q\n\n' \
        "${PYTHON}" \
        'import src.training.train_anomalies'

    exit 1
fi


# -------------------------------------------------------------------
# Validate critical dependencies
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
    'from src.training.pipeline import TrainingPipeline' \
    >/dev/null 2>&1
then
    failure \
        "TrainingPipeline cannot be imported."

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
# Validate trainer CLI
# -------------------------------------------------------------------

HAS_MAIN="$(
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import src.training.train_anomalies as module


main = getattr(
    module,
    "main",
    None,
)


print(
    "1"
    if callable(main)
    else "0"
)
PY
)"


if [[ "${HAS_MAIN}" != "1" ]]; then
    failure \
        "src.training.train_anomalies does not expose a callable main()."

    cat >&2 <<'EOF'

Expected in src/training/train_anomalies.py:

    def main() -> None:
        ...

    if __name__ == "__main__":
        main()

Training behavior belongs in Python. The shell
wrapper must remain an operator entrypoint only.

EOF

    exit 2
fi


# -------------------------------------------------------------------
# Wrapper information
# -------------------------------------------------------------------

if (($# == 1)) \
    && [[ "$1" == "--wrapper-info" ]]; then
    TORCH_VERSION="$(
        "${PYTHON}" -c \
            'import torch; print(torch.__version__)'
    )"

    printf '\n%sOSHA Anomaly Training Wrapper%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Project root:    %s\n' \
        "${PROJECT_ROOT}"

    printf 'Python:          %s\n' \
        "${PYTHON}"

    printf 'Python version:  %s\n' \
        "${PYTHON_VERSION}"

    printf 'PyTorch:         %s\n' \
        "${TORCH_VERSION}"

    printf 'Module:          %s\n' \
        "${TRAIN_MODULE}"

    printf 'Source:          %s\n' \
        "${TRAIN_FILE}"

    printf 'CLI main():      yes\n'

    exit 0
fi


# -------------------------------------------------------------------
# Runtime summary
# -------------------------------------------------------------------

if [[ -t 1 ]]; then
    TORCH_SUMMARY="$(
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
    device_count = int(
        torch.cuda.device_count()
    )

except Exception:
    device_count = 0


print(
    f"{backend}; "
    f"accelerator_available={available}; "
    f"runtime_devices={device_count}"
)
PY
    )"

    printf '\n%sOSHA Anomaly Training%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Python: ${PYTHON_VERSION}"
    info "Runtime: ${TORCH_SUMMARY}"
    info "Trainer: ${TRAIN_MODULE}"

    success \
        "Anomaly trainer preflight checks passed"

    info \
        "Delegating training arguments to ${TRAIN_MODULE}"
fi


# -------------------------------------------------------------------
# Execute trainer
# -------------------------------------------------------------------
#
# exec is intentional:
#
#   Python receives SIGINT/SIGTERM directly
#   Ctrl+C reaches the trainer
#   Python determines the training exit status
#   arguments are preserved exactly
#

exec \
    "${PYTHON}" \
    -m "${TRAIN_MODULE}" \
    "$@"
