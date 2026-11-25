#!/usr/bin/env bash

#
# OSHA API development server.
#
# Application:
#
#     src.api.app:app
#
# Usage:
#
#   ./scripts/api/run_dev.sh
#   ./scripts/api/run_dev.sh --host 127.0.0.1
#   ./scripts/api/run_dev.sh --port 8088
#   ./scripts/api/run_dev.sh --no-reload
#   ./scripts/api/run_dev.sh --log-level debug
#   ./scripts/api/run_dev.sh --verbose
#
# Additional Uvicorn arguments can be supplied after:
#
#   --
#
# Example:
#
#   ./scripts/api/run_dev.sh -- \
#       --access-log
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#   OSHA_API_HOST=127.0.0.1
#   OSHA_API_PORT=8088
#
# Development only.
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

APP_MODULE="src.api.app:app"

APP_FILE="${PROJECT_ROOT}/src/api/app.py"


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

HOST="${OSHA_API_HOST:-127.0.0.1}"

PORT="${OSHA_API_PORT:-8088}"

LOG_LEVEL="${OSHA_API_LOG_LEVEL:-info}"

RELOAD=true

VERBOSE=false

UVICORN_EXTRA_ARGS=()


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
  scripts/api/run_dev.sh [options] [-- uvicorn-options...]

Options:
  --host HOST
      Development bind address.

      Default:
          127.0.0.1

  --port PORT
      Development HTTP port.

      Default:
          8088

  --log-level LEVEL
      Uvicorn log level.

      Supported:
          critical
          error
          warning
          info
          debug
          trace

  --no-reload
      Disable automatic source reload.

  --verbose
      Print the complete Uvicorn command.

  --wrapper-info
      Show the resolved development-server
      configuration without starting the server.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override the Python interpreter.

  OSHA_API_HOST
      Override the default development host.

  OSHA_API_PORT
      Override the default development port.

  OSHA_API_LOG_LEVEL
      Override the default log level.

Examples:
  ./scripts/api/run_dev.sh

  ./scripts/api/run_dev.sh \
      --port 8088

  ./scripts/api/run_dev.sh \
      --host 0.0.0.0 \
      --port 8088

  ./scripts/api/run_dev.sh \
      --log-level debug

  ./scripts/api/run_dev.sh \
      --no-reload

  ./scripts/api/run_dev.sh -- \
      --access-log
EOF
}


# -------------------------------------------------------------------
# Wrapper-info handling
# -------------------------------------------------------------------

WRAPPER_INFO=false


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --host)
            if (($# < 2)); then
                printf \
                    'ERROR: --host requires a value.\n' \
                    >&2

                exit 2
            fi

            HOST="$2"
            shift
            ;;

        --port)
            if (($# < 2)); then
                printf \
                    'ERROR: --port requires a value.\n' \
                    >&2

                exit 2
            fi

            PORT="$2"
            shift
            ;;

        --log-level)
            if (($# < 2)); then
                printf \
                    'ERROR: --log-level requires a value.\n' \
                    >&2

                exit 2
            fi

            LOG_LEVEL="$2"
            shift
            ;;

        --no-reload)
            RELOAD=false
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
                UVICORN_EXTRA_ARGS+=(
                    "$1"
                )

                shift
            done

            break
            ;;

        *)
            printf \
                'Unknown option: %s\n\n' \
                "$1" >&2

            usage >&2

            exit 2
            ;;
    esac

    shift
done


# -------------------------------------------------------------------
# Argument validation
# -------------------------------------------------------------------

if [[ ! "${PORT}" =~ ^[0-9]+$ ]]; then
    failure \
        "Port must be an integer: ${PORT}"

    exit 2
fi


if ((PORT < 1)) \
    || ((PORT > 65535)); then
    failure \
        "Port must be between 1 and 65535."

    exit 2
fi


case "${LOG_LEVEL}" in
    critical|error|warning|info|debug|trace)
        ;;

    *)
        failure \
            "Unsupported log level: ${LOG_LEVEL}"

        printf \
            'Expected: critical, error, warning, info, debug, trace\n' \
            >&2

        exit 2
        ;;
esac


if [[ -z "${HOST}" ]]; then
    failure \
        "Host cannot be empty."

    exit 2
fi


# -------------------------------------------------------------------
# Project
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -f "${APP_FILE}" ]]; then
    failure \
        "API application does not exist: ${APP_FILE}"

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
# Dependency checks
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import fastapi' \
    >/dev/null 2>&1
then
    failure \
        "FastAPI cannot be imported."

    printf '\nRun:\n\n' >&2
    printf '  ./scripts/dev/setup.sh\n\n' >&2

    exit 1
fi


if ! "${PYTHON}" -c \
    'import uvicorn' \
    >/dev/null 2>&1
then
    failure \
        "Uvicorn cannot be imported."

    printf '\nRun:\n\n' >&2
    printf '  ./scripts/dev/setup.sh\n\n' >&2

    exit 1
fi


UVICORN_VERSION="$(
    "${PYTHON}" -c \
        'import uvicorn; print(uvicorn.__version__)'
)"


# -------------------------------------------------------------------
# Compile API before startup
# -------------------------------------------------------------------

if ! "${PYTHON}" -m compileall \
    -q \
    -f \
    "${PROJECT_ROOT}/src/api"
then
    failure \
        "API Python compilation failed."

    exit 1
fi


# -------------------------------------------------------------------
# Import composition root
# -------------------------------------------------------------------

if ! "${PYTHON}" - <<'PY' >/dev/null
from src.api.app import app


if app is None:
    raise RuntimeError(
        "src.api.app.app is None"
    )


routes = getattr(
    app,
    "routes",
    None,
)


if routes is None:
    raise RuntimeError(
        "FastAPI application exposes no route collection"
    )
PY
then
    failure \
        "Unable to import the OSHA API composition root."

    printf '\nRun for the full traceback:\n\n' >&2

    printf '  %q -c %q\n\n' \
        "${PYTHON}" \
        'from src.api.app import app; print(app)'

    exit 1
fi


# -------------------------------------------------------------------
# Application information
# -------------------------------------------------------------------

readarray -t APP_INFO < <(
    "${PYTHON}" - <<'PY'
from src.api.app import app


print(
    getattr(
        app,
        "title",
        "OSHA API",
    )
)

print(
    getattr(
        app,
        "version",
        "",
    )
)

print(
    len(
        getattr(
            app,
            "routes",
            (),
        )
    )
)
PY
)


APP_TITLE="${APP_INFO-OSHA API}"

APP_VERSION="${APP_INFO-}"

APP_ROUTE_COUNT="${APP_INFO-0}"


# -------------------------------------------------------------------
# Development safety warning
# -------------------------------------------------------------------

if [[ "${HOST}" == "0.0.0.0" ]] \
    || [[ "${HOST}" == "::" ]]
then
    warn \
        "Development server will listen on all interfaces."

    warn \
        "Use this only when network exposure is intentional."
fi


# -------------------------------------------------------------------
# Build Uvicorn command
# -------------------------------------------------------------------

COMMAND=(
    "${PYTHON}"
    -m
    uvicorn
    "${APP_MODULE}"
    --host
    "${HOST}"
    --port
    "${PORT}"
    --log-level
    "${LOG_LEVEL}"
)


if [[ "${RELOAD}" == true ]]; then
    COMMAND+=(
        --reload
        --reload-dir
        "${PROJECT_ROOT}/src"
    )
fi


if ((${#UVICORN_EXTRA_ARGS[@]} > 0)); then
    COMMAND+=(
        "${UVICORN_EXTRA_ARGS[@]}"
    )
fi


# -------------------------------------------------------------------
# Wrapper information
# -------------------------------------------------------------------

if [[ "${WRAPPER_INFO}" == true ]]; then
    printf '\n%sOSHA API Development Server%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Project root: %s\n' \
        "${PROJECT_ROOT}"

    printf 'Python:       %s\n' \
        "${PYTHON}"

    printf 'Python ver:   %s\n' \
        "${PYTHON_VERSION}"

    printf 'Uvicorn:      %s\n' \
        "${UVICORN_VERSION}"

    printf 'Application:  %s\n' \
        "${APP_MODULE}"

    printf 'Title:        %s\n' \
        "${APP_TITLE}"

    printf 'Version:      %s\n' \
        "${APP_VERSION}"

    printf 'Routes:       %s\n' \
        "${APP_ROUTE_COUNT}"

    printf 'Host:         %s\n' \
        "${HOST}"

    printf 'Port:         %s\n' \
        "${PORT}"

    printf 'Reload:       %s\n' \
        "${RELOAD}"

    printf 'Log level:    %s\n' \
        "${LOG_LEVEL}"

    printf '\nCommand:\n\n  '

    printf '%q ' \
        "${COMMAND[@]}"

    printf '\n\n'

    exit 0
fi


# -------------------------------------------------------------------
# Startup summary
# -------------------------------------------------------------------

printf '\n%sOSHA API Development Server%s\n' \
    "${BOLD}" \
    "${RESET}"

info "Application: ${APP_TITLE}"

if [[ -n "${APP_VERSION}" ]]; then
    info "Version: ${APP_VERSION}"
fi

info "Registered routes: ${APP_ROUTE_COUNT}"
info "Host: ${HOST}"
info "Port: ${PORT}"
info "Reload: ${RELOAD}"
info "Log level: ${LOG_LEVEL}"
info "Uvicorn: ${UVICORN_VERSION}"


if [[ "${VERBOSE}" == true ]]; then
    printf '\n[INFO] Command:\n\n  '

    printf '%q ' \
        "${COMMAND[@]}"

    printf '\n'
fi


printf '\n'

success \
    "API preflight checks passed"

info \
    "Starting development server"

printf '\n'


# -------------------------------------------------------------------
# Execute
# -------------------------------------------------------------------
#
# exec is intentional:
#
#   Uvicorn becomes the foreground process
#   Ctrl+C reaches Uvicorn directly
#   SIGTERM reaches the server directly
#   exit status is preserved
#

exec \
    "${COMMAND[@]}"
