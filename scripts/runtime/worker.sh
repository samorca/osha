#!/usr/bin/env bash

#
# OSHA runtime worker launcher.
#
# Worker implementations live under:
#
#     src/workers/
#
# Runtime lifecycle lives separately under:
#
#     src/runtime/
#
# Usage:
#
#   ./scripts/runtime/worker.sh --wrapper-info
#
#   ./scripts/runtime/worker.sh \
#       --module src.workers.worker
#
#   ./scripts/runtime/worker.sh \
#       --worker telemetry
#
#   ./scripts/runtime/worker.sh \
#       --worker remediation \
#       -- --help
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
#   OSHA_WORKER_MODULE=src.workers.worker
#
#   OSHA_WORKER_TYPE=telemetry
#
# This shell script only launches a worker.
#
# Worker polling, scheduling, queues, telemetry,
# retries, correlation, decision-making, remediation,
# observability and shutdown behavior remain in
# Python.
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

WORKERS_DIR="${PROJECT_ROOT}/src/workers"


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

WORKER_MODULE="${OSHA_WORKER_MODULE:-}"

WORKER_TYPE="${OSHA_WORKER_TYPE:-}"

VERBOSE=false

WRAPPER_INFO=false

WORKER_ARGS=()


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
# Usage
# -------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  scripts/runtime/worker.sh [options] [-- worker-options...]

Worker selection:

  --module MODULE
      Explicit Python worker entrypoint.

      Example:

          src.workers.worker

  --worker TYPE
      Resolve a worker by type/name.

      Candidate modules:

          src.workers.TYPE
          src.workers.TYPE_worker

      Example:

          --worker telemetry

      attempts:

          src.workers.telemetry
          src.workers.telemetry_worker

Default discovery:

  If neither --module nor --worker is provided,
  these common entrypoints are attempted:

      src.workers.main
      src.workers.worker

Options:

  --verbose
      Show worker resolution attempts and the
      complete execution command.

  --wrapper-info
      Resolve and validate the worker without
      starting it.

  -h, --help
      Show this help.

Environment:

  OSHA_PYTHON
      Override Python interpreter.

  OSHA_WORKER_MODULE
      Explicit Python worker module.

      Example:

          src.workers.worker

  OSHA_WORKER_TYPE
      Default worker type.

      Example:

          telemetry

Argument forwarding:

  Arguments after -- are forwarded unchanged to
  the resolved Python worker.

Examples:

  ./scripts/runtime/worker.sh \
      --wrapper-info

  ./scripts/runtime/worker.sh \
      --module src.workers.worker

  ./scripts/runtime/worker.sh \
      --worker telemetry

  ./scripts/runtime/worker.sh \
      --worker telemetry \
      -- --help

  ./scripts/runtime/worker.sh \
      --worker telemetry \
      -- --config configs/runtime.yaml

  OSHA_WORKER_TYPE=telemetry \
      ./scripts/runtime/worker.sh

Important:

  src.runtime.main is NOT a worker entrypoint.

  The whole application runtime is launched with:

      ./scripts/runtime/run.sh
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --module)
            if (($# < 2)); then
                printf \
                    'ERROR: --module requires a value.\n' \
                    >&2

                exit 2
            fi

            WORKER_MODULE="$2"

            shift
            ;;

        --worker)
            if (($# < 2)); then
                printf \
                    'ERROR: --worker requires a value.\n' \
                    >&2

                exit 2
            fi

            WORKER_TYPE="$2"

            shift
            ;;

        --verbose)
            VERBOSE=true
            ;;

        --wrapper-info)
            WRAPPER_INFO=true
            ;;

        -h|--help)
            usage

            exit 0
            ;;

        --)
            shift

            while (($# > 0)); do
                WORKER_ARGS+=(
                    "$1"
                )

                shift
            done

            break
            ;;

        *)
            #
            # Preserve convenient argument forwarding.
            #
            # Wrapper-specific options should therefore
            # appear before worker-specific arguments.
            #
            WORKER_ARGS+=(
                "$1"
            )
            ;;
    esac

    shift
done


# -------------------------------------------------------------------
# Worker selector validation
# -------------------------------------------------------------------

if [[ -n "${WORKER_MODULE}" ]] \
    && [[ -n "${WORKER_TYPE}" ]]; then
    failure \
        "--module and --worker cannot be used together."

    exit 2
fi


# -------------------------------------------------------------------
# Module-name validation
# -------------------------------------------------------------------

if [[ -n "${WORKER_MODULE}" ]]; then
    if [[ ! "${WORKER_MODULE}" =~ ^[A-Za-z_][A-Za-z0-9_.]*$ ]]; then
        failure \
            "Invalid Python module name: ${WORKER_MODULE}"

        exit 2
    fi


    #
    # Do not accidentally allow the complete runtime
    # executable to masquerade as a worker.
    #
    if [[ "${WORKER_MODULE}" == "src.runtime.main" ]]; then
        failure \
            "src.runtime.main is the application runtime, not a worker."

        printf '\nUse:\n\n' >&2

        printf '  ./scripts/runtime/run.sh\n\n' >&2

        exit 2
    fi
fi


# -------------------------------------------------------------------
# Worker-name validation
# -------------------------------------------------------------------

if [[ -n "${WORKER_TYPE}" ]]; then
    if [[ ! "${WORKER_TYPE}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        failure \
            "Invalid worker type: ${WORKER_TYPE}"

        exit 2
    fi
fi


# -------------------------------------------------------------------
# Project validation
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -d "${WORKERS_DIR}" ]]; then
    failure \
        "Worker package does not exist: ${WORKERS_DIR}"

    exit 2
fi


if [[ ! -f "${WORKERS_DIR}/__init__.py" ]]; then
    warn \
        "src/workers/__init__.py was not found."

    warn \
        "Python namespace-package behavior may still work, but verify the intended package structure."
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

export PYTHONUNBUFFERED=1


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


# -------------------------------------------------------------------
# Compile worker package
# -------------------------------------------------------------------

if ! "${PYTHON}" -m compileall \
    -q \
    -f \
    "${WORKERS_DIR}"
then
    failure \
        "Worker Python compilation failed."

    exit 1
fi


# -------------------------------------------------------------------
# Import package boundary
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import src.workers' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.workers."

    printf '\nRun this for the complete traceback:\n\n' >&2

    printf '  %q -c %q\n\n' \
        "${PYTHON}" \
        'import src.workers'

    exit 1
fi


# -------------------------------------------------------------------
# Candidate worker modules
# -------------------------------------------------------------------

CANDIDATES=()


#
# Explicit module has highest priority.
#
if [[ -n "${WORKER_MODULE}" ]]; then
    CANDIDATES+=(
        "${WORKER_MODULE}"
    )


#
# Named worker.
#
elif [[ -n "${WORKER_TYPE}" ]]; then
    CANDIDATES+=(
        "src.workers.${WORKER_TYPE}"
        "src.workers.${WORKER_TYPE}_worker"
    )


#
# Automatic worker entrypoint discovery.
#
# Do NOT fall back to src.runtime.main.
#
else
    CANDIDATES+=(
        "src.workers.main"
        "src.workers.worker"
    )
fi


# -------------------------------------------------------------------
# Candidate list for Python resolver
# -------------------------------------------------------------------

CANDIDATE_TEXT="$(
    printf '%s\n' \
        "${CANDIDATES[@]}"
)"


export OSHA_WORKER_CANDIDATES="${CANDIDATE_TEXT}"


# -------------------------------------------------------------------
# Resolve executable worker
# -------------------------------------------------------------------

set +e


RESOLUTION="$(
    "${PYTHON}" - <<'PY' 2>&1
from __future__ import annotations

import importlib
import importlib.util
import json
import os

from pathlib import Path


candidates = [
    value.strip()
    for value in os.environ.get(
        "OSHA_WORKER_CANDIDATES",
        "",
    ).splitlines()
    if value.strip()
]


attempts = []


for name in candidates:
    # ---------------------------------------------------------------
    # Locate module
    # ---------------------------------------------------------------

    try:
        spec = importlib.util.find_spec(
            name
        )

    except (
        ImportError,
        ModuleNotFoundError,
        AttributeError,
    ) as exc:
        attempts.append(
            {
                "module": name,

                "found": False,

                "error_type": (
                    type(exc).__name__
                ),

                "error": str(exc),
            }
        )

        continue


    if spec is None:
        attempts.append(
            {
                "module": name,
                "found": False,
            }
        )

        continue


    # ---------------------------------------------------------------
    # Import module
    # ---------------------------------------------------------------

    try:
        module = importlib.import_module(
            name
        )

    except Exception as exc:
        attempts.append(
            {
                "module": name,

                "found": True,

                "importable": False,

                "error_type": (
                    type(exc).__name__
                ),

                "error": str(exc),
            }
        )

        continue


    # ---------------------------------------------------------------
    # Require an executable main()
    # ---------------------------------------------------------------

    main = getattr(
        module,
        "main",
        None,
    )


    source = getattr(
        module,
        "__file__",
        None,
    )


    result = {
        "module": name,

        "found": True,

        "importable": True,

        "has_main": callable(
            main
        ),

        "source": (
            str(
                Path(source).resolve()
            )
            if source
            else None
        ),
    }


    attempts.append(
        result
    )


    if not callable(main):
        continue


    # ---------------------------------------------------------------
    # Successful resolution
    # ---------------------------------------------------------------

    print(
        json.dumps(
            {
                "success": True,

                "module": name,

                "source": (
                    result[
                        "source"
                    ]
                ),

                "attempts": (
                    attempts
                ),
            },
            sort_keys=True,
        )
    )


    raise SystemExit(0)


# -------------------------------------------------------------------
# No executable worker
# -------------------------------------------------------------------

print(
    json.dumps(
        {
            "success": False,

            "attempts": (
                attempts
            ),
        },
        sort_keys=True,
    )
)


raise SystemExit(3)
PY
)"


RESOLUTION_EXIT=$?


set -e


# -------------------------------------------------------------------
# Resolution failure
# -------------------------------------------------------------------

if ((RESOLUTION_EXIT != 0)); then
    failure \
        "No executable OSHA worker module was found."


    if [[ "${VERBOSE}" == true ]]; then
        printf '\nResolution attempts:\n\n' >&2

        printf '%s\n' \
            "${RESOLUTION}" \
        | "${PYTHON}" -m json.tool \
        >&2 \
        || printf '%s\n' \
            "${RESOLUTION}" >&2
    fi


    printf '\nAvailable worker Python files:\n\n' >&2


    find src/workers \
        -maxdepth 3 \
        -type f \
        -name '*.py' \
        -print \
        | sort \
        >&2


    printf '\nA worker entry module must expose:\n\n' >&2


    cat >&2 <<'EOF'
    def main() -> None:
        ...

    if __name__ == "__main__":
        main()

Example explicit invocation:

    ./scripts/runtime/worker.sh \
        --module src.workers.YOUR_MODULE

Example named worker invocation:

    ./scripts/runtime/worker.sh \
        --worker telemetry

Important:

    src.runtime.main

is deliberately not used as a worker fallback.

Launch the complete runtime with:

    ./scripts/runtime/run.sh
EOF


    exit "${RESOLUTION_EXIT}"
fi


# -------------------------------------------------------------------
# Extract resolved module
# -------------------------------------------------------------------

RESOLVED_MODULE="$(
    RESOLUTION="${RESOLUTION}" \
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import json
import os


payload = json.loads(
    os.environ[
        "RESOLUTION"
    ]
)


print(
    payload[
        "module"
    ]
)
PY
)"


RESOLVED_SOURCE="$(
    RESOLUTION="${RESOLUTION}" \
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import json
import os


payload = json.loads(
    os.environ[
        "RESOLUTION"
    ]
)


print(
    payload.get(
        "source"
    )
    or ""
)
PY
)"


# -------------------------------------------------------------------
# Build worker command
# -------------------------------------------------------------------

COMMAND=(
    "${PYTHON}"
    -m
    "${RESOLVED_MODULE}"
)


if ((${#WORKER_ARGS[@]} > 0)); then
    COMMAND+=(
        "${WORKER_ARGS[@]}"
    )
fi


# -------------------------------------------------------------------
# Wrapper information
# -------------------------------------------------------------------

if [[ "${WRAPPER_INFO}" == true ]]; then
    printf '\n%sOSHA Runtime Worker Launcher%s\n' \
        "${BOLD}" \
        "${RESET}"


    printf 'Project root:   %s\n' \
        "${PROJECT_ROOT}"


    printf 'Worker root:    %s\n' \
        "${WORKERS_DIR}"


    printf 'Python:         %s\n' \
        "${PYTHON}"


    printf 'Python version: %s\n' \
        "${PYTHON_VERSION}"


    printf 'Worker type:    %s\n' \
        "$(
            if [[ -n "${WORKER_TYPE}" ]]; then
                printf '%s' \
                    "${WORKER_TYPE}"
            else
                printf '<not specified>'
            fi
        )"


    printf 'Module:         %s\n' \
        "${RESOLVED_MODULE}"


    printf 'Source:         %s\n' \
        "${RESOLVED_SOURCE}"


    printf 'CLI main():     yes\n'


    printf 'Unbuffered:     %s\n' \
        "${PYTHONUNBUFFERED}"


    if [[ "${VERBOSE}" == true ]]; then
        printf '\nResolution attempts:\n\n'

        printf '%s\n' \
            "${RESOLUTION}" \
        | "${PYTHON}" -m json.tool
    fi


    printf '\nCommand:\n\n  '


    printf '%q ' \
        "${COMMAND[@]}"


    printf '\n\n'


    exit 0
fi


# -------------------------------------------------------------------
# Startup summary
# -------------------------------------------------------------------

printf '\n%sOSHA Runtime Worker%s\n' \
    "${BOLD}" \
    "${RESET}"


info \
    "Project root: ${PROJECT_ROOT}"


info \
    "Python: ${PYTHON_VERSION}"


if [[ -n "${WORKER_TYPE}" ]]; then
    info \
        "Worker type: ${WORKER_TYPE}"
fi


info \
    "Module: ${RESOLVED_MODULE}"


info \
    "Source: ${RESOLVED_SOURCE}"


if [[ "${VERBOSE}" == true ]]; then
    printf '\n[INFO] Resolution:\n\n'


    printf '%s\n' \
        "${RESOLUTION}" \
    | "${PYTHON}" -m json.tool


    printf '\n[INFO] Command:\n\n  '


    printf '%q ' \
        "${COMMAND[@]}"


    printf '\n'
fi


printf '\n'


success \
    "Worker preflight checks passed"


info \
    "Starting worker in foreground"


printf '\n'


# -------------------------------------------------------------------
# Execute worker
# -------------------------------------------------------------------
#
# exec is intentional:
#
#   Python becomes the foreground process.
#
#   Ctrl+C reaches the worker directly.
#
#   SIGTERM reaches the worker directly.
#
#   Python owns worker lifecycle/shutdown.
#
#   The worker's exit status is preserved.
#

exec \
    "${COMMAND[@]}"
