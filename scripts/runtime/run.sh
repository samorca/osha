#!/usr/bin/env bash

#
# OSHA orchestration runtime launcher.
#
# Authoritative executable:
#
#     src.runtime.main
#
# Architecture:
#
#     scripts/runtime/run.sh
#              |
#              v
#     src.runtime.main
#              |
#              v
#     OshaRuntime
#              |
#              +--> RuntimeConfig
#              +--> RuntimeServiceRegistry
#              +--> RuntimeBootstrap
#              +--> RuntimeStateManager
#              +--> RuntimeShutdown
#
# Usage:
#
#   ./scripts/runtime/run.sh
#
#   ./scripts/runtime/run.sh --wrapper-info
#
#   ./scripts/runtime/run.sh --once
#
#   ./scripts/runtime/run.sh --json
#
#   ./scripts/runtime/run.sh \
#       --status-interval 30
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# Runtime configuration continues to come from
# supported OSHA_* environment variables.
#
# This shell wrapper does not implement runtime
# lifecycle behavior itself.
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

RUNTIME_MODULE="src.runtime.main"

RUNTIME_FILE="${PROJECT_ROOT}/src/runtime/main.py"


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

VERBOSE=false
WRAPPER_INFO=false

RUNTIME_ARGS=()


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
# Usage
# -------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  scripts/runtime/run.sh [options] [-- runtime-options...]

Wrapper options:

  --wrapper-info
      Validate the runtime environment and display
      the resolved command without starting OSHA.

  --verbose
      Print additional preflight information.

  -h, --help
      Show this wrapper help.

Runtime options:

  Runtime arguments may be supplied directly or
  after --.

  Common runtime options include:

      --once

      --host-id HOST_ID

      --started-by IDENTITY

      --status-interval SECONDS

      --json

      --log-level LEVEL

      --no-signal-handlers

Examples:

  Inspect the launcher:

      ./scripts/runtime/run.sh \
          --wrapper-info

  Start OSHA:

      ./scripts/runtime/run.sh

  Safe one-shot lifecycle test:

      ./scripts/runtime/run.sh \
          --once

  JSON one-shot test:

      ./scripts/runtime/run.sh \
          --once \
          --json

  Periodic runtime snapshots:

      ./scripts/runtime/run.sh \
          --status-interval 30

  Explicit runtime argument separator:

      ./scripts/runtime/run.sh -- \
          --once \
          --json

Environment:

  OSHA_PYTHON
      Override the Python interpreter.

Runtime configuration is loaded by:

  src.runtime.config.from_environment()

using its supported OSHA_* variables.

Worker processes are NOT started through this
wrapper.

Use:

  scripts/runtime/worker.sh

for individual workers.
EOF
}


# -------------------------------------------------------------------
# Wrapper arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --wrapper-info)
            WRAPPER_INFO=true
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
                RUNTIME_ARGS+=(
                    "$1"
                )

                shift
            done

            break
            ;;

        *)
            #
            # The remaining options belong to
            # src.runtime.main.
            #
            RUNTIME_ARGS+=(
                "$1"
            )
            ;;
    esac

    shift
done


# -------------------------------------------------------------------
# Project validation
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -d "${PROJECT_ROOT}/src/runtime" ]]; then
    failure \
        "src/runtime does not exist."

    exit 2
fi


if [[ ! -f "${RUNTIME_FILE}" ]]; then
    failure \
        "Runtime entrypoint does not exist: ${RUNTIME_FILE}"

    cat >&2 <<'EOF'

The OSHA runtime executable must exist at:

    src/runtime/main.py

Expected:

    def main() -> None:
        ...

    if __name__ == "__main__":
        main()

EOF

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
        "Python is not executable: ${PYTHON}"

    exit 2
fi


export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTHONUNBUFFERED=1


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


# -------------------------------------------------------------------
# Compile runtime
# -------------------------------------------------------------------

if ! "${PYTHON}" -m compileall \
    -q \
    -f \
    "${PROJECT_ROOT}/src/runtime"
then
    failure \
        "Runtime Python compilation failed."

    exit 1
fi


# -------------------------------------------------------------------
# Runtime package
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import src.runtime' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.runtime."

    printf '\nRun for the full traceback:\n\n' >&2

    printf '  %q -c %q\n\n' \
        "${PYTHON}" \
        'import src.runtime'

    exit 1
fi


# -------------------------------------------------------------------
# Runtime entrypoint
# -------------------------------------------------------------------

set +e

RUNTIME_INFO="$(
    "${PYTHON}" - <<'PY' 2>&1
from __future__ import annotations

import json
from pathlib import Path

import src.runtime.main as module


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


payload = {
    "module": (
        module.__name__
    ),

    "source": (
        str(
            Path(source).resolve()
        )
        if source
        else None
    ),

    "has_main": callable(
        main
    ),
}


print(
    json.dumps(
        payload,
        sort_keys=True,
    )
)


raise SystemExit(
    0
    if callable(main)
    else 1
)
PY
)"

RUNTIME_INFO_EXIT=$?

set -e


if ((RUNTIME_INFO_EXIT != 0)); then
    failure \
        "src.runtime.main does not expose an executable main()."

    printf '%s\n' \
        "${RUNTIME_INFO}" >&2

    exit 1
fi


RUNTIME_SOURCE="$(
    RUNTIME_INFO="${RUNTIME_INFO}" \
    "${PYTHON}" - <<'PY'
import json
import os


payload = json.loads(
    os.environ[
        "RUNTIME_INFO"
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
# Runtime lifecycle contracts
# -------------------------------------------------------------------

set +e

CONTRACT_INFO="$(
    "${PYTHON}" - <<'PY' 2>&1
from __future__ import annotations

import json

from src.runtime.bootstrap import (
    RuntimeBootstrap,
)

from src.runtime.config import (
    RuntimeConfig,
    from_environment,
)

from src.runtime.runtime import (
    OshaRuntime,
    runtime_from_environment,
)

from src.runtime.services import (
    RuntimeServiceManager,
    RuntimeServiceRegistry,
    registry_from_config,
)

from src.runtime.shutdown import (
    RuntimeShutdown,
)

from src.runtime.state import (
    RuntimeStateManager,
)


config = from_environment()


payload = {
    "config_class": (
        RuntimeConfig.__name__
    ),

    "runtime_class": (
        OshaRuntime.__name__
    ),

    "bootstrap_class": (
        RuntimeBootstrap.__name__
    ),

    "registry_class": (
        RuntimeServiceRegistry.
        __name__
    ),

    "service_manager_class": (
        RuntimeServiceManager.
        __name__
    ),

    "state_manager_class": (
        RuntimeStateManager.
        __name__
    ),

    "shutdown_class": (
        RuntimeShutdown.__name__
    ),

    "has_runtime_from_environment": (
        callable(
            runtime_from_environment
        )
    ),

    "has_registry_from_config": (
        callable(
            registry_from_config
        )
    ),

    "environment_id": (
        config.environment_id
    ),

    "instance_name": (
        config.instance_name
    ),

    "dry_run": (
        config.dry_run
    ),

    "configured_services": sorted(
        config.services.keys()
    ),
}


print(
    json.dumps(
        payload,
        sort_keys=True,
    )
)
PY
)"

CONTRACT_EXIT=$?

set -e


if ((CONTRACT_EXIT != 0)); then
    failure \
        "Runtime lifecycle contract validation failed."

    printf '%s\n' \
        "${CONTRACT_INFO}" >&2

    exit 1
fi


# -------------------------------------------------------------------
# Extract safe configuration summary
# -------------------------------------------------------------------

ENVIRONMENT_ID="$(
    CONTRACT_INFO="${CONTRACT_INFO}" \
    "${PYTHON}" - <<'PY'
import json
import os


print(
    json.loads(
        os.environ[
            "CONTRACT_INFO"
        ]
    )[
        "environment_id"
    ]
)
PY
)"


INSTANCE_NAME="$(
    CONTRACT_INFO="${CONTRACT_INFO}" \
    "${PYTHON}" - <<'PY'
import json
import os


print(
    json.loads(
        os.environ[
            "CONTRACT_INFO"
        ]
    )[
        "instance_name"
    ]
)
PY
)"


DRY_RUN="$(
    CONTRACT_INFO="${CONTRACT_INFO}" \
    "${PYTHON}" - <<'PY'
import json
import os


value = json.loads(
    os.environ[
        "CONTRACT_INFO"
    ]
)[
    "dry_run"
]


print(
    str(value).lower()
)
PY
)"


SERVICE_COUNT="$(
    CONTRACT_INFO="${CONTRACT_INFO}" \
    "${PYTHON}" - <<'PY'
import json
import os


payload = json.loads(
    os.environ[
        "CONTRACT_INFO"
    ]
)


print(
    len(
        payload[
            "configured_services"
        ]
    )
)
PY
)"


# -------------------------------------------------------------------
# Safety notice
# -------------------------------------------------------------------

if [[ "${DRY_RUN}" != "true" ]]; then
    warn \
        "Runtime configuration reports dry_run=false."

    warn \
        "The launcher itself performs no remediation, but registered runtime services may have operational behavior."
fi


# -------------------------------------------------------------------
# Build command
# -------------------------------------------------------------------

COMMAND=(
    "${PYTHON}"
    -m
    "${RUNTIME_MODULE}"
)


if ((${#RUNTIME_ARGS[@]} > 0)); then
    COMMAND+=(
        "${RUNTIME_ARGS[@]}"
    )
fi


# -------------------------------------------------------------------
# Wrapper information
# -------------------------------------------------------------------

if [[ "${WRAPPER_INFO}" == true ]]; then
    printf '\n%sOSHA Runtime Launcher%s\n' \
        "${BOLD}" \
        "${RESET}"


    printf 'Project root:        %s\n' \
        "${PROJECT_ROOT}"


    printf 'Python:              %s\n' \
        "${PYTHON}"


    printf 'Python version:      %s\n' \
        "${PYTHON_VERSION}"


    printf 'Runtime module:      %s\n' \
        "${RUNTIME_MODULE}"


    printf 'Runtime source:      %s\n' \
        "${RUNTIME_SOURCE}"


    printf 'CLI main():          yes\n'


    printf 'Environment ID:      %s\n' \
        "${ENVIRONMENT_ID}"


    printf 'Instance name:       %s\n' \
        "${INSTANCE_NAME}"


    printf 'Dry run:             %s\n' \
        "${DRY_RUN}"


    printf 'Configured services: %s\n' \
        "${SERVICE_COUNT}"


    printf 'Python unbuffered:   %s\n' \
        "${PYTHONUNBUFFERED}"


    if [[ "${VERBOSE}" == true ]]; then
        printf '\nRuntime contracts:\n\n'

        printf '%s\n' \
            "${CONTRACT_INFO}" \
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

printf '\n%sOSHA Orchestration Runtime%s\n' \
    "${BOLD}" \
    "${RESET}"


info \
    "Project root: ${PROJECT_ROOT}"


info \
    "Python: ${PYTHON_VERSION}"


info \
    "Runtime module: ${RUNTIME_MODULE}"


info \
    "Environment: ${ENVIRONMENT_ID}"


info \
    "Instance: ${INSTANCE_NAME}"


info \
    "Dry run: ${DRY_RUN}"


info \
    "Configured services: ${SERVICE_COUNT}"


if [[ "${VERBOSE}" == true ]]; then
    printf '\n[INFO] Runtime contracts:\n\n'

    printf '%s\n' \
        "${CONTRACT_INFO}" \
    | "${PYTHON}" -m json.tool


    printf '\n[INFO] Command:\n\n  '


    printf '%q ' \
        "${COMMAND[@]}"


    printf '\n'
fi


printf '\n'


success \
    "Runtime preflight checks passed"


info \
    "Starting OSHA runtime"


printf '\n'


# -------------------------------------------------------------------
# Execute runtime
# -------------------------------------------------------------------
#
# exec is intentional:
#
#   src.runtime.main becomes the foreground process.
#
#   Ctrl+C reaches RuntimeSignalController.
#
#   SIGTERM reaches RuntimeSignalController.
#
#   OshaRuntime owns lifecycle shutdown.
#
#   Python's process exit status is preserved.
#

exec \
    "${COMMAND[@]}"
