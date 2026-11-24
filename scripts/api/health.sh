#!/usr/bin/env bash

#
# OSHA API health probe.
#
# Endpoints:
#
#   /health/live
#       Is the API process alive?
#
#   /health/ready
#       Is the worker ready to receive traffic?
#
#   /health
#       What is the state of API components?
#
# Usage:
#
#   ./scripts/api/health.sh
#   ./scripts/api/health.sh --live
#   ./scripts/api/health.sh --ready
#   ./scripts/api/health.sh --full
#   ./scripts/api/health.sh --all
#   ./scripts/api/health.sh --json
#
# Environment:
#
#   OSHA_API_URL=http://127.0.0.1:8000
#   OSHA_API_HEALTH_TIMEOUT=5
#
# This script is read-only.
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

BASE_URL="${OSHA_API_URL:-http://127.0.0.1:8000}"

TIMEOUT="${OSHA_API_HEALTH_TIMEOUT:-5}"

MODE="all"

JSON_OUTPUT=false
VERBOSE=false
QUIET=false

REQUEST_ID=""

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
# Output
# -------------------------------------------------------------------

info() {
    if [[ "${JSON_OUTPUT}" == false ]] \
        && [[ "${QUIET}" == false ]]; then
        printf '%s[INFO]%s %s\n' \
            "${BLUE}" \
            "${RESET}" \
            "$*"
    fi
}


success() {
    if [[ "${JSON_OUTPUT}" == false ]] \
        && [[ "${QUIET}" == false ]]; then
        printf '%s[PASS]%s %s\n' \
            "${GREEN}" \
            "${RESET}" \
            "$*"
    fi
}


warn() {
    if [[ "${JSON_OUTPUT}" == false ]] \
        && [[ "${QUIET}" == false ]]; then
        printf '%s[WARN]%s %s\n' \
            "${YELLOW}" \
            "${RESET}" \
            "$*" >&2
    fi
}


failure() {
    if [[ "${JSON_OUTPUT}" == false ]] \
        && [[ "${QUIET}" == false ]]; then
        printf '%s[FAIL]%s %s\n' \
            "${RED}" \
            "${RESET}" \
            "$*" >&2
    fi
}


# -------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  scripts/api/health.sh [options]

Probe selection:
  --live
      Probe /health/live only.

  --ready
      Probe /health/ready only.

  --full
      Probe /health only.

  --all
      Probe all three health endpoints.

      This is the default.

Connection:
  --url URL
      API base URL.

      Default:
          http://127.0.0.1:8000

  --timeout SECONDS
      Maximum request time for each probe.

      Default:
          5

Output:
  --json
      Emit one machine-readable JSON result.

  --verbose
      Include response headers and response details.

  --quiet
      Suppress normal human-readable output.

      Exit status still indicates success/failure.

  --request-id VALUE
      Send an explicit X-Request-ID.

  --wrapper-info
      Show resolved configuration without probing.

  -h, --help
      Show this help.

Environment:
  OSHA_API_URL
  OSHA_API_HEALTH_TIMEOUT

Examples:
  ./scripts/api/health.sh

  ./scripts/api/health.sh --live

  ./scripts/api/health.sh --ready

  ./scripts/api/health.sh --full

  ./scripts/api/health.sh \
      --url http://127.0.0.1:8080

  ./scripts/api/health.sh --json

  ./scripts/api/health.sh --quiet
EOF
}


# -------------------------------------------------------------------
# Wrapper info
# -------------------------------------------------------------------

WRAPPER_INFO=false


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --live)
            MODE="live"
            ;;

        --ready)
            MODE="ready"
            ;;

        --full)
            MODE="full"
            ;;

        --all)
            MODE="all"
            ;;

        --url)
            if (($# < 2)); then
                printf \
                    'ERROR: --url requires a value.\n' \
                    >&2

                exit 2
            fi

            BASE_URL="$2"
            shift
            ;;

        --timeout)
            if (($# < 2)); then
                printf \
                    'ERROR: --timeout requires a value.\n' \
                    >&2

                exit 2
            fi

            TIMEOUT="$2"
            shift
            ;;

        --json)
            JSON_OUTPUT=true
            ;;

        --verbose)
            VERBOSE=true
            ;;

        --quiet)
            QUIET=true
            ;;

        --request-id)
            if (($# < 2)); then
                printf \
                    'ERROR: --request-id requires a value.\n' \
                    >&2

                exit 2
            fi

            REQUEST_ID="$2"
            shift
            ;;

        --wrapper-info)
            WRAPPER_INFO=true
            ;;

        -h|--help)
            usage
            exit 0
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
# Validation
# -------------------------------------------------------------------

if [[ -z "${BASE_URL}" ]]; then
    printf \
        'ERROR: API URL cannot be empty.\n' \
        >&2

    exit 2
fi


# Remove trailing slash so endpoint concatenation is stable.
BASE_URL="${BASE_URL%/}"


if [[ ! "${TIMEOUT}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf \
        'ERROR: timeout must be numeric: %s\n' \
        "${TIMEOUT}" >&2

    exit 2
fi


if [[ "${TIMEOUT}" == "0" ]] \
    || [[ "${TIMEOUT}" == "0.0" ]]
then
    printf \
        'ERROR: timeout must be greater than zero.\n' \
        >&2

    exit 2
fi


case "${BASE_URL}" in
    http://*|https://*)
        ;;

    *)
        printf \
            'ERROR: URL must start with http:// or https://\n' \
            >&2

        exit 2
        ;;
esac


# -------------------------------------------------------------------
# Project
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


# -------------------------------------------------------------------
# Python discovery
# -------------------------------------------------------------------
#
# Python is used to parse and normalize JSON responses.
#

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
    printf \
        'ERROR: no Python interpreter found.\n' \
        >&2

    exit 2
fi


if [[ ! -x "${PYTHON}" ]]; then
    printf \
        'ERROR: Python is not executable: %s\n' \
        "${PYTHON}" >&2

    exit 2
fi


# -------------------------------------------------------------------
# curl
# -------------------------------------------------------------------

if ! command -v curl >/dev/null 2>&1; then
    failure \
        "curl is required for HTTP health probes."

    exit 2
fi


CURL_VERSION="$(
    curl --version \
    | head -n 1
)"


# -------------------------------------------------------------------
# Wrapper information
# -------------------------------------------------------------------

if [[ "${WRAPPER_INFO}" == true ]]; then
    printf '\n%sOSHA API Health Probe%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Project root: %s\n' \
        "${PROJECT_ROOT}"

    printf 'Base URL:     %s\n' \
        "${BASE_URL}"

    printf 'Mode:         %s\n' \
        "${MODE}"

    printf 'Timeout:      %ss\n' \
        "${TIMEOUT}"

    printf 'JSON:         %s\n' \
        "${JSON_OUTPUT}"

    printf 'Quiet:        %s\n' \
        "${QUIET}"

    printf 'Verbose:      %s\n' \
        "${VERBOSE}"

    printf 'curl:         %s\n' \
        "${CURL_VERSION}"

    exit 0
fi


# -------------------------------------------------------------------
# Temporary working files
# -------------------------------------------------------------------

TEMP_DIR="$(
    mktemp -d \
        "${TMPDIR:-/tmp}/osha-health.XXXXXX"
)"


cleanup() {
    rm -rf -- \
        "${TEMP_DIR}"
}


trap cleanup EXIT INT TERM


# -------------------------------------------------------------------
# Probe definitions
# -------------------------------------------------------------------

PROBE_NAMES=()
PROBE_PATHS=()


case "${MODE}" in
    live)
        PROBE_NAMES+=(
            "liveness"
        )

        PROBE_PATHS+=(
            "/health/live"
        )
        ;;

    ready)
        PROBE_NAMES+=(
            "readiness"
        )

        PROBE_PATHS+=(
            "/health/ready"
        )
        ;;

    full)
        PROBE_NAMES+=(
            "health"
        )

        PROBE_PATHS+=(
            "/health"
        )
        ;;

    all)
        PROBE_NAMES+=(
            "liveness"
            "readiness"
            "health"
        )

        PROBE_PATHS+=(
            "/health/live"
            "/health/ready"
            "/health"
        )
        ;;

    *)
        printf \
            'ERROR: unsupported probe mode: %s\n' \
            "${MODE}" >&2

        exit 2
        ;;
esac


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == false ]] \
    && [[ "${QUIET}" == false ]]; then
    printf '\n%sOSHA API Health Probe%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Base URL: ${BASE_URL}"
    info "Mode: ${MODE}"
    info "Timeout: ${TIMEOUT}s"

    printf '\n'
fi


# -------------------------------------------------------------------
# Probe individual endpoint
# -------------------------------------------------------------------

RESULT_FILES=()

OVERALL_FAILURE=0


for index in "${!PROBE_NAMES[@]}"; do
    name="${PROBE_NAMES[$index]}"

    path="${PROBE_PATHS[$index]}"

    url="${BASE_URL}${path}"

    body_file="${TEMP_DIR}/${name}.body"

    headers_file="${TEMP_DIR}/${name}.headers"

    error_file="${TEMP_DIR}/${name}.error"

    result_file="${TEMP_DIR}/${name}.json"

    CURL_ARGS=(
        --silent
        --show-error
        --location
        --max-time
        "${TIMEOUT}"
        --connect-timeout
        "${TIMEOUT}"
        --output
        "${body_file}"
        --dump-header
        "${headers_file}"
        --write-out
        '%{http_code}|%{time_total}|%{size_download}'
        --header
        'Accept: application/json'
    )


    if [[ -n "${REQUEST_ID}" ]]; then
        CURL_ARGS+=(
            --header
            "X-Request-ID: ${REQUEST_ID}"
        )
    fi


    set +e

    CURL_META="$(
        curl \
            "${CURL_ARGS[@]}" \
            "${url}" \
            2>"${error_file}"
    )"

    CURL_EXIT=$?

    set -e


    HTTP_CODE="000"
    TOTAL_TIME="0"
    SIZE_DOWNLOAD="0"


    if [[ -n "${CURL_META}" ]]; then
        IFS='|' read -r \
            HTTP_CODE \
            TOTAL_TIME \
            SIZE_DOWNLOAD \
            <<< "${CURL_META}"
    fi


    # ---------------------------------------------------------------
    # Valid HTTP health status
    # ---------------------------------------------------------------

    HTTP_SUCCESS=false


    if [[ "${HTTP_CODE}" =~ ^2[0-9][0-9]$ ]]; then
        HTTP_SUCCESS=true
    fi


    if [[ "${CURL_EXIT}" -ne 0 ]]; then
        OVERALL_FAILURE=1

    elif [[ "${HTTP_SUCCESS}" == false ]]; then
        OVERALL_FAILURE=1
    fi


    # ---------------------------------------------------------------
    # Build normalized JSON result
    # ---------------------------------------------------------------

    OSHA_HEALTH_NAME="${name}" \
    OSHA_HEALTH_PATH="${path}" \
    OSHA_HEALTH_URL="${url}" \
    OSHA_HEALTH_HTTP="${HTTP_CODE}" \
    OSHA_HEALTH_CURL_EXIT="${CURL_EXIT}" \
    OSHA_HEALTH_TIME="${TOTAL_TIME}" \
    OSHA_HEALTH_SIZE="${SIZE_DOWNLOAD}" \
    OSHA_HEALTH_BODY="${body_file}" \
    OSHA_HEALTH_HEADERS="${headers_file}" \
    OSHA_HEALTH_ERROR="${error_file}" \
    "${PYTHON}" - <<'PY' \
        > "${result_file}"
from __future__ import annotations

import json
import os
from pathlib import Path


name = os.environ[
    "OSHA_HEALTH_NAME"
]

path = os.environ[
    "OSHA_HEALTH_PATH"
]

url = os.environ[
    "OSHA_HEALTH_URL"
]

http_code_text = os.environ.get(
    "OSHA_HEALTH_HTTP",
    "000",
)

curl_exit = int(
    os.environ.get(
        "OSHA_HEALTH_CURL_EXIT",
        "1",
    )
)

time_text = os.environ.get(
    "OSHA_HEALTH_TIME",
    "0",
)

size_text = os.environ.get(
    "OSHA_HEALTH_SIZE",
    "0",
)


body_path = Path(
    os.environ[
        "OSHA_HEALTH_BODY"
    ]
)

headers_path = Path(
    os.environ[
        "OSHA_HEALTH_HEADERS"
    ]
)

error_path = Path(
    os.environ[
        "OSHA_HEALTH_ERROR"
    ]
)


body_text = (
    body_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    if body_path.exists()
    else ""
)


headers_text = (
    headers_path.read_text(
        encoding="latin-1",
        errors="replace",
    )
    if headers_path.exists()
    else ""
)


error_text = (
    error_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()
    if error_path.exists()
    else ""
)


try:
    http_code = int(
        http_code_text
    )

except ValueError:
    http_code = 0


try:
    duration_seconds = float(
        time_text
    )

except ValueError:
    duration_seconds = 0.0


try:
    size_bytes = int(
        float(
            size_text
        )
    )

except ValueError:
    size_bytes = 0


parsed_body = None


if body_text:
    try:
        parsed_body = json.loads(
            body_text
        )

    except json.JSONDecodeError:
        parsed_body = None


response_headers: dict[
    str,
    str,
] = {}


# dump-header may contain more than one response when
# redirects occur. Keep values from the final block.
current: dict[
    str,
    str,
] = {}


for raw_line in headers_text.splitlines():
    line = raw_line.strip(
        "\r"
    )

    if line.startswith(
        "HTTP/"
    ):
        current = {}

        continue

    if not line:
        if current:
            response_headers = (
                current
            )

        continue

    if ":" not in line:
        continue

    key, value = line.split(
        ":",
        1,
    )

    current[
        key.strip().lower()
    ] = value.strip()


if current:
    response_headers = current


http_success = (
    200
    <= http_code
    < 300
)


success = (
    curl_exit == 0
    and http_success
)


result = {
    "name": name,

    "path": path,

    "url": url,

    "success": success,

    "transport": {
        "success": (
            curl_exit == 0
        ),

        "curl_exit_code": (
            curl_exit
        ),

        "error": (
            error_text
            or None
        ),
    },

    "response": {
        "http_status": (
            http_code
        ),

        "duration_seconds": (
            duration_seconds
        ),

        "duration_ms": (
            duration_seconds
            * 1000.0
        ),

        "size_bytes": (
            size_bytes
        ),

        "content_type": (
            response_headers.get(
                "content-type"
            )
        ),

        "request_id": (
            response_headers.get(
                "x-request-id"
            )
        ),

        "body": (
            parsed_body
            if parsed_body
            is not None
            else (
                body_text
                if body_text
                else None
            )
        ),
    },
}


print(
    json.dumps(
        result,
        sort_keys=True,
    )
)
PY


    RESULT_FILES+=(
        "${result_file}"
    )


    # ---------------------------------------------------------------
    # Human-readable output
    # ---------------------------------------------------------------

    if [[ "${JSON_OUTPUT}" == false ]] \
        && [[ "${QUIET}" == false ]]; then
        if [[ "${CURL_EXIT}" -ne 0 ]]; then
            failure \
                "${name}: connection/request failure"

            if [[ -s "${error_file}" ]]; then
                printf '       '

                cat "${error_file}" >&2
            fi

        elif [[ "${HTTP_SUCCESS}" == true ]]; then
            success \
                "${name}: HTTP ${HTTP_CODE} (${TOTAL_TIME}s)"

        else
            failure \
                "${name}: HTTP ${HTTP_CODE} (${TOTAL_TIME}s)"
        fi


        if [[ "${VERBOSE}" == true ]]; then
            printf '\n'

            printf 'URL: %s\n' \
                "${url}"

            printf 'HTTP status: %s\n' \
                "${HTTP_CODE}"

            printf 'Duration: %ss\n' \
                "${TOTAL_TIME}"

            printf 'Downloaded: %s bytes\n' \
                "${SIZE_DOWNLOAD}"


            if [[ -s "${headers_file}" ]]; then
                printf '\nResponse headers:\n'

                cat "${headers_file}"
            fi


            if [[ -s "${body_file}" ]]; then
                printf '\nResponse body:\n'

                if "${PYTHON}" -m json.tool \
                    "${body_file}" \
                    2>/dev/null
                then
                    :
                else
                    cat "${body_file}"
                    printf '\n'
                fi
            fi


            printf '\n'
        fi
    fi
done


# -------------------------------------------------------------------
# JSON output
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == true ]]; then
    OSHA_HEALTH_MODE="${MODE}" \
    OSHA_HEALTH_BASE_URL="${BASE_URL}" \
    OSHA_HEALTH_TIMEOUT="${TIMEOUT}" \
    OSHA_HEALTH_OVERALL_FAILURE="${OVERALL_FAILURE}" \
    "${PYTHON}" - \
        "${RESULT_FILES[@]}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


probes = []


for value in sys.argv[1:]:
    path = Path(value)

    probes.append(
        json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    )


success = all(
    probe.get(
        "success",
        False,
    )
    for probe in probes
)


payload = {
    "schema_version": "1",

    "success": (
        success
    ),

    "base_url": os.environ[
        "OSHA_HEALTH_BASE_URL"
    ],

    "mode": os.environ[
        "OSHA_HEALTH_MODE"
    ],

    "timeout_seconds": float(
        os.environ[
            "OSHA_HEALTH_TIMEOUT"
        ]
    ),

    "probe_count": len(
        probes
    ),

    "probes": probes,
}


print(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
)
PY
fi


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


if [[ "${JSON_OUTPUT}" == false ]] \
    && [[ "${QUIET}" == false ]]; then
    printf '\n%s========================================%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf '%sOSHA API Health Summary%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Base URL: %s\n' \
        "${BASE_URL}"

    printf 'Mode:     %s\n' \
        "${MODE}"

    printf 'Probes:   %s\n' \
        "${#PROBE_NAMES[@]}"

    printf 'Duration: %ss\n' \
        "${elapsed}"

    printf 'Result:   %s\n' \
        "$(
            if ((OVERALL_FAILURE == 0)); then
                printf 'healthy'
            else
                printf 'unhealthy'
            fi
        )"

    printf '%s========================================%s\n' \
        "${BOLD}" \
        "${RESET}"


    if ((OVERALL_FAILURE == 0)); then
        printf '\n%sAPI HEALTH CHECK PASSED%s\n' \
            "${GREEN}" \
            "${RESET}"

    else
        printf '\n%sAPI HEALTH CHECK FAILED%s\n' \
            "${RED}" \
            "${RESET}"
    fi
fi


exit "${OVERALL_FAILURE}"
