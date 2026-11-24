#!/usr/bin/env bash

#
# OSHA event-model training entrypoint.
#
# Authoritative implementation:
#
#     src/training/train_events.py
#
# Execution path:
#
#     train_event.sh
#          |
#          v
#     src.training.train_events
#          |
#          v
#     TrainingPipeline
#          |
#          +--> event dataset
#          +--> event model
#          +--> AcceleratorManager
#          +--> optimizer
#          +--> checkpoints
#          +--> evaluation
#
# Usage:
#
#   ./scripts/training/train_event.sh --help
#   ./scripts/training/train_event.sh <trainer arguments...>
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This wrapper does not implement training logic.
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

TRAIN_MODULE="src.training.train_events"

TRAIN_FILE="${PROJECT_ROOT}/src/training/train_events.py"


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
# Logging helpers
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
        "Event trainer does not exist: ${TRAIN_FILE}"

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
# Compile trainer before execution
# -------------------------------------------------------------------

if ! "${PYTHON}" -m py_compile \
    "${TRAIN_FILE}"
then
    failure \
        "src/training/train_events.py failed Python compilation."

    printf '\nFix the training source before starting a run.\n' >&2

    exit 1
fi


# -------------------------------------------------------------------
# Import trainer
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import src.training.train_events' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.training.train_events."

    printf '\nRun this for the complete traceback:\n\n' >&2

    printf '  %q -c %q\n\n' \
        "${PYTHON}" \
        'import src.training.train_events'

    exit 1
fi


# -------------------------------------------------------------------
# Validate critical training dependencies
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


# -------------------------------------------------------------------
# Check trainer CLI
# -------------------------------------------------------------------

HAS_MAIN="$(
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import src.training.train_events as module


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
        "src.training.train_events does not expose a callable main()."

    cat >&2 <<'EOF'

The training implementation should own its CLI.

Expected structure inside:

    src/training/train_events.py

is:

    def main() -> None:
        ...

    if __name__ == "__main__":
        main()

The shell wrapper should not reproduce the
training loop.

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

    printf '\n%sOSHA Event Training Wrapper%s\n' \
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
# Training environment summary
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
    build = (
        "rocm:"
        + str(hip)
    )

elif cuda:
    build = (
        "cuda:"
        + str(cuda)
    )

else:
    build = "cpu"


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
    f"{build}; "
    f"accelerator_available={available}; "
    f"runtime_devices={count}"
)
PY
    )"

    printf '\n%sOSHA Event Training%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Python: ${PYTHON_VERSION}"
    info "Runtime: ${TORCH_SUMMARY}"
    info "Trainer: ${TRAIN_MODULE}"
fi


# -------------------------------------------------------------------
# Preflight acknowledgement
# -------------------------------------------------------------------
#
# We deliberately do not:
#
#   require a GPU
#   inspect trainer-specific --data arguments
#   guess checkpoint paths
#   guess model options
#
# train_events.py owns those policies and validates
# its CLI arguments itself.
#

if [[ -t 1 ]]; then
    success \
        "Event trainer preflight checks passed"

    info \
        "Delegating training arguments to ${TRAIN_MODULE}"
fi


# -------------------------------------------------------------------
# Execute trainer
# -------------------------------------------------------------------
#
# exec is intentional:
#
#   * Python directly receives SIGINT/SIGTERM
#   * Ctrl+C reaches the training process
#   * Python's exit code is preserved
#   * no shell process remains between the operator
#     and the training implementation
#

exec \
    "${PYTHON}" \
    -m "${TRAIN_MODULE}" \
    "$@"
