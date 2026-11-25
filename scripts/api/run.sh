#!/usr/bin/env bash

#
# OSHA API service launcher.
#
# Application:
#
#     src.api.app:app
#
# This is the non-reloading API launcher.
#
# Usage:
#
#   ./scripts/api/run.sh
#   ./scripts/api/run.sh --workers 4
#   ./scripts/api/run.sh --host 0.0.0.0
#   ./scripts/api/run.sh --port 8088
#   ./scripts/api/run.sh --wrapper-info
#
# Additional Uvicorn arguments may be supplied after:
#
#   --
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
#   OSHA_API_HOST=0.0.0.0
#   OSHA_API_PORT=8088
#   OSHA_API_WORKERS=1
#
#   OSHA_API_LOG_LEVEL=info
#
#   OSHA_API_GRACEFUL_TIMEOUT=30
#
#   OSHA_API_LIMIT_CONCURRENCY=
#   OSHA_API_LIMIT_MAX_REQUESTS=
#
#   OSHA_FORWARDED_ALLOW_IPS=
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

HOST="${OSHA_API_HOST:-0.0.0.0}"

PORT="${OSHA_API_PORT:-8088}"

WORKERS="${OSHA_API_WORKERS:-1}"

LOG_LEVEL="${OSHA_API_LOG_LEVEL:-info}"

GRACEFUL_TIMEOUT="${OSHA_API_GRACEFUL_TIMEOUT:-30}"

LIMIT_CONCURRENCY="${OSHA_API_LIMIT_CONCURRENCY:-500}"

LIMIT_MAX_REQUESTS="${OSHA_API_LIMIT_MAX_REQUESTS:-10000}"

FORWARDED_ALLOW_IPS="${OSHA_FORWARDED_ALLOW_IPS:-127.0.0.1}"

ACCESS_LOG=true

PROXY_HEADERS=true

VERBOSE=false

WRAPPER_INFO=false

UVICORN_EXTRA_ARGS=()


# -------------------------------------------------------------------
# Normalize multiline assignments
# -------------------------------------------------------------------

GRACEFUL_TIMEOUT="${OSHA_API_GRACEFUL_TIMEOUT:-30}"

LIMIT_CONCURRENCY="${OSHA_API_LIMIT_CONCURRENCY:-500}"

LIMIT_MAX_REQUESTS="${OSHA_API_LIMIT_MAX_REQUESTS:-10000}"

FORWARDED_ALLOW_IPS="${OSHA_FORWARDED_ALLOW_IPS:-127.0.0.1}"


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
  scripts/api/run.sh [options] [-- uvicorn-options...]

Options:
  --host HOST
      Bind address.

  --port PORT
      TCP port.

  --workers COUNT
      Number of Uvicorn worker processes.

  --log-level LEVEL
      critical
      error
      warning
      info
      debug
      trace

  --graceful-timeout SECONDS
      Graceful shutdown timeout.

  --limit-concurrency COUNT
      Maximum concurrent connections/tasks accepted
      by Uvicorn.

  --limit-max-requests COUNT
      Recycle workers after this many requests.

  --forwarded-allow-ips VALUE
      Trusted IP addresses/networks for forwarded
      proxy headers.

  --no-proxy-headers
      Disable proxy-header processing.

  --no-access-log
      Disable Uvicorn access logging.

  --verbose
      Print the complete command.

  --wrapper-info
      Resolve and display configuration without
      starting the server.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
  OSHA_API_HOST
  OSHA_API_PORT
  OSHA_API_WORKERS
  OSHA_API_LOG_LEVEL
  OSHA_API_GRACEFUL_TIMEOUT
  OSHA_API_LIMIT_CONCURRENCY
  OSHA_API_LIMIT_MAX_REQUESTS
  OSHA_FORWARDED_ALLOW_IPS

Examples:
  ./scripts/api/run.sh

  ./scripts/api/run.sh \
      --workers 4

  ./scripts/api/run.sh \
      --host 0.0.0.0 \
      --port 8088 \
      --workers 4

  ./scripts/api/run.sh \
      --forwarded-allow-ips 127.0.0.1

  ./scripts/api/run.sh \
      --no-access-log

  ./scripts/api/run.sh \
      --wrapper-info
EOF
}


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

        --workers)
            if (($# < 2)); then
                printf \
                    'ERROR: --workers requires a value.\n' \
                    >&2

                exit 2
            fi

            WORKERS="$2"
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

        --graceful-timeout)
            if (($# < 2)); then
                printf \
                    'ERROR: --graceful-timeout requires a value.\n' \
                    >&2

                exit 2
            fi

            GRACEFUL_TIMEOUT="$2"
            shift
            ;;

        --limit-concurrency)
            if (($# < 2)); then
                printf \
                    'ERROR: --limit-concurrency requires a value.\n' \
                    >&2

                exit 2
            fi

            LIMIT_CONCURRENCY="$2"
            shift
            ;;

        --limit-max-requests)
            if (($# < 2)); then
                printf \
                    'ERROR: --limit-max-requests requires a value.\n' \
                    >&2

                exit 2
            fi

            LIMIT_MAX_REQUESTS="$2"
            shift
            ;;

        --forwarded-allow-ips)
            if (($# < 2)); then
                printf \
                    'ERROR: --forwarded-allow-ips requires a value.\n' \
                    >&2

                exit 2
            fi

            FORWARDED_ALLOW_IPS="$2"
            shift
            ;;

        --no-proxy-headers)
            PROXY_HEADERS=false
            ;;

        --no-access-log)
            ACCESS_LOG=false
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
# Numeric validation
# -------------------------------------------------------------------

validate_positive_integer() {
    local name="$1"
    local value="$2"

    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        failure \
            "${name} must be an integer: ${value}"

        exit 2
    fi

    if ((value < 1)); then
        failure \
            "${name} must be >= 1."

        exit 2
    fi
}


validate_nonnegative_integer() {
    local name="$1"
    local value="$2"

    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        failure \
            "${name} must be an integer: ${value}"

        exit 2
    fi
}


validate_positive_integer \
    "Port" \
    "${PORT}"


if ((PORT > 65535)); then
    failure \
        "Port must be <= 65535."

    exit 2
fi


validate_positive_integer \
    "Workers" \
    "${WORKERS}"


validate_nonnegative_integer \
    "Graceful timeout" \
    "${GRACEFUL_TIMEOUT}"


validate_positive_integer \
    "Concurrency limit" \
    "${LIMIT_CONCURRENCY}"


validate_positive_integer \
    "Maximum requests" \
    "${LIMIT_MAX_REQUESTS}"


case "${LOG_LEVEL}" in
    critical|error|warning|info|debug|trace)
        ;;

    *)
        failure \
            "Unsupported log level: ${LOG_LEVEL}"

        exit 2
        ;;
esac


if [[ -z "${HOST}" ]]; then
    failure \
        "Host cannot be empty."

    exit 2
fi


# -------------------------------------------------------------------
# Proxy trust safety
# -------------------------------------------------------------------

if [[ "${PROXY_HEADERS}" == true ]] \
    && [[ "${FORWARDED_ALLOW_IPS}" == "*" ]]; then
    warn \
        "All proxy addresses are trusted."

    warn \
        "Do not use '*' if clients can reach Uvicorn directly."
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
        "Python is not executable: ${PYTHON}"

    exit 2
fi


export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


# -------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import fastapi, uvicorn' \
    >/dev/null 2>&1
then
    failure \
        "FastAPI and/or Uvicorn cannot be imported."

    printf '\nRun:\n\n' >&2
    printf '  ./scripts/dev/setup.sh\n\n' >&2

    exit 1
fi


UVICORN_VERSION="$(
    "${PYTHON}" -c \
        'import uvicorn; print(uvicorn.__version__)'
)"


# -------------------------------------------------------------------
# API compile check
# -------------------------------------------------------------------

if ! "${PYTHON}" -m compileall \
    -q \
    -f \
    "${PROJECT_ROOT}/src/api"
then
    failure \
        "API compilation failed."

    exit 1
fi


# -------------------------------------------------------------------
# Composition-root preflight
# -------------------------------------------------------------------

APP_INFO="$(
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import json

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
        "Application has no route collection"
    )


payload = {
    "title": getattr(
        app,
        "title",
        "OSHA API",
    ),

    "version": getattr(
        app,
        "version",
        "",
    ),

    "route_count": len(
        routes
    ),
}


print(
    json.dumps(
        payload
    )
)
PY
)"


APP_TITLE="$(
    APP_INFO="${APP_INFO}" \
    "${PYTHON}" - <<'PY'
import json
import os

print(
    json.loads(
        os.environ["APP_INFO"]
    )["title"]
)
PY
)"


APP_VERSION="$(
    APP_INFO="${APP_INFO}" \
    "${PYTHON}" - <<'PY'
import json
import os

print(
    json.loads(
        os.environ["APP_INFO"]
    )["version"]
)
PY
)"


APP_ROUTE_COUNT="$(
    APP_INFO="${APP_INFO}" \
    "${PYTHON}" - <<'PY'
import json
import os

print(
    json.loads(
        os.environ["APP_INFO"]
    )["route_count"]
)
PY
)"


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
    --workers
    "${WORKERS}"
    --log-level
    "${LOG_LEVEL}"
    --timeout-graceful-shutdown
    "${GRACEFUL_TIMEOUT}"
    --limit-concurrency
    "${LIMIT_CONCURRENCY}"
    --limit-max-requests
    "${LIMIT_MAX_REQUESTS}"
)


# -------------------------------------------------------------------
# Proxy headers
# -------------------------------------------------------------------

if [[ "${PROXY_HEADERS}" == true ]]; then
    COMMAND+=(
        --proxy-headers
        --forwarded-allow-ips
        "${FORWARDED_ALLOW_IPS}"
    )

else
    COMMAND+=(
        --no-proxy-headers
    )
fi


# -------------------------------------------------------------------
# Access logging
# -------------------------------------------------------------------

if [[ "${ACCESS_LOG}" == true ]]; then
    COMMAND+=(
        --access-log
    )

else
    COMMAND+=(
        --no-access-log
    )
fi


# -------------------------------------------------------------------
# Extra arguments
# -------------------------------------------------------------------

if ((${#UVICORN_EXTRA_ARGS[@]} > 0)); then
    COMMAND+=(
        "${UVICORN_EXTRA_ARGS[@]}"
    )
fi


# -------------------------------------------------------------------
# Wrapper information
# -------------------------------------------------------------------

if [[ "${WRAPPER_INFO}" == true ]]; then
    printf '\n%sOSHA API Service Launcher%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Project root:      %s\n' \
        "${PROJECT_ROOT}"

    printf 'Python:            %s\n' \
        "${PYTHON}"

    printf 'Python version:    %s\n' \
        "${PYTHON_VERSION}"

    printf 'Uvicorn:           %s\n' \
        "${UVICORN_VERSION}"

    printf 'Application:       %s\n' \
        "${APP_MODULE}"

    printf 'Title:             %s\n' \
        "${APP_TITLE}"

    printf 'Version:           %s\n' \
        "${APP_VERSION}"

    printf 'Routes:            %s\n' \
        "${APP_ROUTE_COUNT}"

    printf 'Host:              %s\n' \
        "${HOST}"

    printf 'Port:              %s\n' \
        "${PORT}"

    printf 'Workers:           %s\n' \
        "${WORKERS}"

    printf 'Log level:         %s\n' \
        "${LOG_LEVEL}"

    printf 'Access log:        %s\n' \
        "${ACCESS_LOG}"

    printf 'Proxy headers:     %s\n' \
        "${PROXY_HEADERS}"

    printf 'Forwarded IPs:     %s\n' \
        "${FORWARDED_ALLOW_IPS}"

    printf 'Concurrency limit: %s\n' \
        "${LIMIT_CONCURRENCY}"

    printf 'Max requests:      %s\n' \
        "${LIMIT_MAX_REQUESTS}"

    printf 'Graceful timeout:  %ss\n' \
        "${GRACEFUL_TIMEOUT}"

    printf '\nCommand:\n\n  '

    printf '%q ' \
        "${COMMAND[@]}"

    printf '\n\n'

    exit 0
fi


# -------------------------------------------------------------------
# Service summary
# -------------------------------------------------------------------

printf '\n%sOSHA API Service%s\n' \
    "${BOLD}" \
    "${RESET}"

info "Application: ${APP_TITLE}"

if [[ -n "${APP_VERSION}" ]]; then
    info "Version: ${APP_VERSION}"
fi

info "Routes: ${APP_ROUTE_COUNT}"
info "Host: ${HOST}"
info "Port: ${PORT}"
info "Workers: ${WORKERS}"
info "Log level: ${LOG_LEVEL}"
info "Access log: ${ACCESS_LOG}"
info "Proxy headers: ${PROXY_HEADERS}"

if [[ "${PROXY_HEADERS}" == true ]]; then
    info \
        "Forwarded allow IPs: ${FORWARDED_ALLOW_IPS}"
fi

info \
    "Concurrency limit: ${LIMIT_CONCURRENCY}"

info \
    "Maximum requests/worker: ${LIMIT_MAX_REQUESTS}"

info \
    "Graceful shutdown timeout: ${GRACEFUL_TIMEOUT}s"


if [[ "${VERBOSE}" == true ]]; then
    printf '\n[INFO] Command:\n\n  '

    printf '%q ' \
        "${COMMAND[@]}"

    printf '\n'
fi


printf '\n'

success \
    "API service preflight checks passed"

info \
    "Starting OSHA API"

printf '\n'


# -------------------------------------------------------------------
# Execute
# -------------------------------------------------------------------
#
# exec is intentional:
#
#   Uvicorn becomes the foreground process
#   signals reach the server directly
#   graceful shutdown remains Uvicorn's responsibility
#   server exit status is preserved
#

exec \
    "${COMMAND[@]}"
