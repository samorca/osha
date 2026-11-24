#!/usr/bin/env bash

#
# OSHA training subsystem smoke test.
#
# Default behavior:
#
#   Compile and import the training subsystem,
#   validate AcceleratorManager, inspect trainer
#   entrypoints, and run the accelerator smoke test.
#
# Optional execution:
#
#   --event -- <trainer arguments...>
#
#       delegates to:
#
#           scripts/training/train_event.sh
#
#   --anomaly -- <trainer arguments...>
#
#       delegates to:
#
#           scripts/training/train_anomaly.sh
#
# A real training run happens ONLY when --event or
# --anomaly is explicitly supplied.
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

ACCELERATOR_SMOKE="${
    PROJECT_ROOT
}/scripts/accelerators/smoke_test.sh"

EVENT_TRAINER="${
    PROJECT_ROOT
}/scripts/training/train_event.sh"

ANOMALY_TRAINER="${
    PROJECT_ROOT
}/scripts/training/train_anomaly.sh"


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

MODE="preflight"

VERBOSE=false

TRAINER_ARGS=()

START_SECONDS="${SECONDS}"


# -------------------------------------------------------------------
# Colours
# -------------------------------------------------------------------

if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    RESET=$'\033[0m'
    GREEN=$'\033[32m'
    RED=$'\033[31m'
    BLUE=$'\033[34m'
    BOLD=$'\033[1m'
else
    RESET=""
    GREEN=""
    RED=""
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


failure() {
    printf '%s[FAIL]%s %s\n' \
        "${RED}" \
        "${RESET}" \
        "$*" >&2
}


# -------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  scripts/training/smoke_test.sh [options]

Preflight only:

  ./scripts/training/smoke_test.sh

Run event trainer explicitly:

  ./scripts/training/smoke_test.sh \
      --event \
      -- <event trainer arguments...>

Run anomaly trainer explicitly:

  ./scripts/training/smoke_test.sh \
      --anomaly \
      -- <anomaly trainer arguments...>

Options:
  --event
      Delegate to train_event.sh after preflight.

  --anomaly
      Delegate to train_anomaly.sh after preflight.

  --verbose
      Show additional preflight information.

  -h, --help
      Show this help.

Important:
  The default command performs no model training.

  --event and --anomaly initiate a real trainer,
  so provide appropriately small trainer arguments
  yourself if smoke-training is intended.
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --event)
            if [[ "${MODE}" != "preflight" ]]; then
                failure \
                    "Choose either --event or --anomaly."

                exit 2
            fi

            MODE="event"
            ;;

        --anomaly)
            if [[ "${MODE}" != "preflight" ]]; then
                failure \
                    "Choose either --event or --anomaly."

                exit 2
            fi

            MODE="anomaly"
            ;;

        --verbose)
            VERBOSE=true
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        --)
            shift

            while (($# > 0)); do
                TRAINER_ARGS+=(
                    "$1"
                )

                shift
            done

            break
            ;;

        *)
            failure \
                "Unknown argument: $1"

            printf '\n' >&2
            usage >&2

            exit 2
            ;;
    esac

    shift
done


# -------------------------------------------------------------------
# Normalize paths
# -------------------------------------------------------------------

ACCELERATOR_SMOKE="${PROJECT_ROOT}/scripts/accelerators/smoke_test.sh"

EVENT_TRAINER="${PROJECT_ROOT}/scripts/training/train_event.sh"

ANOMALY_TRAINER="${PROJECT_ROOT}/scripts/training/train_anomaly.sh"


# -------------------------------------------------------------------
# Project
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


# -------------------------------------------------------------------
# Python
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


export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

printf '\n%sOSHA Training Smoke Test%s\n' \
    "${BOLD}" \
    "${RESET}"

info "Mode: ${MODE}"

info "Python: $(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


# -------------------------------------------------------------------
# Compile training sources
# -------------------------------------------------------------------

info "Compiling training modules"


if "${PYTHON}" -m compileall \
    -q \
    -f \
    "${PROJECT_ROOT}/src/training"
then
    success \
        "Training source compilation"
else
    failure \
        "Training source compilation"

    exit 1
fi


# -------------------------------------------------------------------
# Import training boundaries
# -------------------------------------------------------------------

info "Importing training subsystem"


if ! "${PYTHON}" - <<'PY'
from __future__ import annotations

import importlib


modules = (
    "src.training",
    "src.training.config",
    "src.training.schema",
    "src.training.pipeline",
    "src.training.event_dataset",
    "src.training.anomaly_dataset",
    "src.training.event_model",
    "src.training.anomaly_model",
    "src.training.train_events",
    "src.training.train_anomalies",
)


for module in modules:
    importlib.import_module(
        module
    )

    print(
        "  OK "
        + module
    )
PY
then
    failure \
        "Training subsystem imports"

    exit 1
fi


success \
    "Training subsystem imports"


# -------------------------------------------------------------------
# Entrypoint validation
# -------------------------------------------------------------------

info "Validating training entrypoints"


if ! "${PYTHON}" - <<'PY'
from __future__ import annotations

import src.training.train_events as events
import src.training.train_anomalies as anomalies


for name, module in (
    (
        "train_events",
        events,
    ),
    (
        "train_anomalies",
        anomalies,
    ),
):
    main = getattr(
        module,
        "main",
        None,
    )

    if not callable(main):
        raise RuntimeError(
            name
            + " does not expose main()"
        )

    print(
        "  OK "
        + name
        + ".main"
    )
PY
then
    failure \
        "Training entrypoint validation"

    exit 1
fi


success \
    "Training entrypoints"


# -------------------------------------------------------------------
# Accelerator smoke test
# -------------------------------------------------------------------

if [[ ! -x "${ACCELERATOR_SMOKE}" ]]; then
    failure \
        "Accelerator smoke test is unavailable: ${ACCELERATOR_SMOKE}"

    exit 2
fi


info "Running accelerator smoke test"


if OSHA_PYTHON="${PYTHON}" \
    "${ACCELERATOR_SMOKE}" \
    --kind auto
then
    success \
        "Accelerator smoke test"
else
    failure \
        "Accelerator smoke test"

    exit 1
fi


# -------------------------------------------------------------------
# Optional verbose summary
# -------------------------------------------------------------------

if [[ "${VERBOSE}" == true ]]; then
    printf '\n'

    "${PYTHON}" - <<'PY'
from __future__ import annotations

import torch


print(
    "PyTorch:",
    torch.__version__,
)

print(
    "CUDA runtime:",
    torch.version.cuda,
)

print(
    "HIP runtime:",
    getattr(
        torch.version,
        "hip",
        None,
    ),
)

print(
    "Accelerator available:",
    torch.cuda.is_available(),
)

print(
    "Runtime devices:",
    torch.cuda.device_count(),
)
PY
fi


# -------------------------------------------------------------------
# Pure preflight mode
# -------------------------------------------------------------------

if [[ "${MODE}" == "preflight" ]]; then
    elapsed=$(
        (
            SECONDS
            - START_SECONDS
        )
    )

    printf '\n%s========================================%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf '%sOSHA Training Smoke Summary%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Mode:     preflight\n'

    printf 'Training: not executed\n'

    printf 'Duration: %ss\n' \
        "${elapsed}"

    printf '%s========================================%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf '\n%sTRAINING PREFLIGHT PASSED%s\n' \
        "${GREEN}" \
        "${RESET}"

    exit 0
fi


# -------------------------------------------------------------------
# Real trainer delegation
# -------------------------------------------------------------------

if [[ "${MODE}" == "event" ]]; then
    TRAINER="${EVENT_TRAINER}"

else
    TRAINER="${ANOMALY_TRAINER}"
fi


if [[ ! -x "${TRAINER}" ]]; then
    failure \
        "Training wrapper is unavailable: ${TRAINER}"

    exit 2
fi


if ((${#TRAINER_ARGS[@]} == 0)); then
    failure \
        "Real training was requested but no trainer arguments were supplied."

    printf '\nInspect the trainer first:\n\n' >&2

    printf '  %q --help\n\n' \
        "${TRAINER}" >&2

    exit 2
fi


printf '\n'

info \
    "Preflight complete; delegating to ${MODE} trainer"

printf '\n'


exec \
    env OSHA_PYTHON="${PYTHON}" \
    "${TRAINER}" \
    "${TRAINER_ARGS[@]}"
