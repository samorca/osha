#!/usr/bin/env bash

#
# OSHA event-model evaluation entrypoint.
#
# Expected authoritative implementation:
#
#     src/training/evaluate_events.py
#
# Execution path:
#
#     evaluate_event.sh
#          |
#          v
#     src.training.evaluate_events
#          |
#          +--> trained checkpoint
#          +--> event dataset
#          +--> accelerator selection
#          +--> model evaluation
#          +--> metrics
#          +--> evaluation report
#
# Usage:
#
#   ./scripts/training/evaluate_event.sh --help
#   ./scripts/training/evaluate_event.sh <evaluator arguments...>
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This wrapper does not implement model evaluation
# or metric calculations itself.
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

EVALUATION_MODULE="src.training.evaluate_events"

EVALUATION_FILE="${PROJECT_ROOT}/src/training/evaluate_events.py"


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
# Wrapper information
# -------------------------------------------------------------------
#
# Handle this before requiring the evaluator so the
# wrapper can also tell us whether implementation is
# currently available.
#

if (($# == 1)) \
    && [[ "$1" == "--wrapper-info" ]]; then
    printf '\n%sOSHA Event Evaluation Wrapper%s\n' \
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
        HAS_MAIN="$(
            "${PYTHON}" - <<'PY' 2>/dev/null || true
from __future__ import annotations

import src.training.evaluate_events as module


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

        printf 'CLI main():        %s\n' \
            "${HAS_MAIN:-import failed}"

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
        "Event evaluation implementation does not exist: ${EVALUATION_FILE}"

    cat >&2 <<'EOF'

The evaluation logic should be implemented in:

    src/training/evaluate_events.py

Do not put checkpoint loading, dataset evaluation,
or metrics logic into evaluate_event.sh.

The Python module should expose:

    def main() -> None:
        ...

    if __name__ == "__main__":
        main()

After that this wrapper will execute:

    python -m src.training.evaluate_events

EOF

    exit 2
fi


# -------------------------------------------------------------------
# Compile evaluator
# -------------------------------------------------------------------

if ! "${PYTHON}" -m py_compile \
    "${EVALUATION_FILE}"
then
    failure \
        "src/training/evaluate_events.py failed Python compilation."

    exit 1
fi


# -------------------------------------------------------------------
# Import evaluator
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import src.training.evaluate_events' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.training.evaluate_events."

    printf '\nRun this for the complete traceback:\n\n' >&2

    printf '  %q -c %q\n\n' \
        "${PYTHON}" \
        'import src.training.evaluate_events'

    exit 1
fi


# -------------------------------------------------------------------
# Critical dependencies
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
# CLI validation
# -------------------------------------------------------------------

HAS_MAIN="$(
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import src.training.evaluate_events as module


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
        "src.training.evaluate_events does not expose a callable main()."

    cat >&2 <<'EOF'

The Python evaluator should own command-line
parsing and evaluation orchestration.

Expected:

    def main() -> None:
        ...

    if __name__ == "__main__":
        main()

EOF

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

    printf '\n%sOSHA Event Model Evaluation%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Python: ${PYTHON_VERSION}"
    info "Runtime: ${RUNTIME_SUMMARY}"
    info "Evaluator: ${EVALUATION_MODULE}"

    success \
        "Event evaluator preflight checks passed"

    info \
        "Delegating evaluation arguments to ${EVALUATION_MODULE}"
fi


# -------------------------------------------------------------------
# Execute evaluator
# -------------------------------------------------------------------
#
# exec preserves:
#
#   Ctrl+C / SIGINT
#   SIGTERM
#   Python exit status
#   exact arguments
#

exec \
    "${PYTHON}" \
    -m "${EVALUATION_MODULE}" \
    "$@"
