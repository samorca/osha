#!/usr/bin/env bash

#
# OSHA API HTTP smoke test.
#
# Usage:
#
#   ./scripts/api/smoke_test.sh
#   ./scripts/api/smoke_test.sh --url http://127.0.0.1:8088
#   ./scripts/api/smoke_test.sh --verbose
#   ./scripts/api/smoke_test.sh --json
#   ./scripts/api/smoke_test.sh --skip-ready
#
# Environment:
#
#   OSHA_API_URL=http://127.0.0.1:8088
#   OSHA_API_SMOKE_TIMEOUT=10
#   OSHA_PYTHON=/path/to/python
#
# The smoke test is deliberately read-only.
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

HEALTH_SCRIPT="${PROJECT_ROOT}/scripts/api/health.sh"


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

BASE_URL="${OSHA_API_URL:-http://127.0.0.1:8088}"

TIMEOUT="${OSHA_API_SMOKE_TIMEOUT:-10}"

VERBOSE=false
JSON_OUTPUT=false
SKIP_READY=false
WRAPPER_INFO=false

FAILURES=0
CHECKS=0

START_SECONDS="${SECONDS}"

REQUEST_ID="osha-api-smoke-$$"


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
    if [[ "${JSON_OUTPUT}" == false ]]; then
        printf '%s[INFO]%s %s\n' \
            "${BLUE}" \
            "${RESET}" \
            "$*"
    fi
}


success() {
    if [[ "${JSON_OUTPUT}" == false ]]; then
        printf '%s[PASS]%s %s\n' \
            "${GREEN}" \
            "${RESET}" \
            "$*"
    fi
}


warn() {
    if [[ "${JSON_OUTPUT}" == false ]]; then
        printf '%s[WARN]%s %s\n' \
            "${YELLOW}" \
            "${RESET}" \
            "$*" >&2
    fi
}


failure() {
    FAILURES=$((FAILURES + 1))

    if [[ "${JSON_OUTPUT}" == false ]]; then
        printf '%s[FAIL]%s %s\n' \
            "${RED}" \
            "${RESET}" \
            "$*" >&2
    fi
}


section() {
    if [[ "${JSON_OUTPUT}" == false ]]; then
        printf '\n%s%s== %s ==%s\n' \
            "${BOLD}" \
            "${BLUE}" \
            "$*" \
            "${RESET}"
    fi
}


# -------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  scripts/api/smoke_test.sh [options]

Options:
  --url URL
      OSHA API base URL.

      Default:
          http://127.0.0.1:8088

  --timeout SECONDS
      Maximum duration of each HTTP request.

      Default:
          10

  --skip-ready
      Do not require /health/ready to return 2xx.

      Useful while intentionally testing an API
      whose optional/critical service dependencies
      are still being configured.

      Liveness remains mandatory.

  --json
      Emit machine-readable JSON summary.

  --verbose
      Show response details.

  --wrapper-info
      Show resolved smoke-test configuration
      without contacting the API.

  -h, --help
      Show this help.

Environment:
  OSHA_API_URL
  OSHA_API_SMOKE_TIMEOUT
  OSHA_PYTHON

Examples:
  ./scripts/api/smoke_test.sh

  ./scripts/api/smoke_test.sh \
      --verbose

  ./scripts/api/smoke_test.sh \
      --url       http://127.0.0.1:8088

  ./scripts/api/smoke_test.sh \
      --skip-ready

  ./scripts/api/smoke_test.sh \
      --json
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
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

        --skip-ready)
            SKIP_READY=true
            ;;

        --json)
            JSON_OUTPUT=true
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
# Input validation
# -------------------------------------------------------------------

BASE_URL="${BASE_URL%/}"


case "${BASE_URL}" in
    http://*|https://*)
        ;;

    *)
        printf \
            'ERROR: --url must begin with http:// or https://\n' \
            >&2

        exit 2
        ;;
esac


if [[ ! "${TIMEOUT}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf \
        'ERROR: timeout must be numeric.\n' \
        >&2

    exit 2
fi


# -------------------------------------------------------------------
# Project
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


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
    printf \
        'ERROR: curl is required for API smoke testing.\n' \
        >&2

    exit 2
fi


# -------------------------------------------------------------------
# health.sh
# -------------------------------------------------------------------

if [[ ! -x "${HEALTH_SCRIPT}" ]]; then
    printf \
        'ERROR: health probe is unavailable: %s\n' \
        "${HEALTH_SCRIPT}" >&2

    exit 2
fi


# -------------------------------------------------------------------
# Wrapper information
# -------------------------------------------------------------------

if [[ "${WRAPPER_INFO}" == true ]]; then
    printf '\n%sOSHA API Smoke Test%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Project root: %s\n' \
        "${PROJECT_ROOT}"

    printf 'Base URL:     %s\n' \
        "${BASE_URL}"

    printf 'Timeout:      %ss\n' \
        "${TIMEOUT}"

    printf 'Skip ready:   %s\n' \
        "${SKIP_READY}"

    printf 'Health probe: %s\n' \
        "${HEALTH_SCRIPT}"

    printf 'Python:       %s\n' \
        "${PYTHON}"

    exit 0
fi


# -------------------------------------------------------------------
# Temporary files
# -------------------------------------------------------------------

TEMP_DIR="$(
    mktemp -d \
        "${TMPDIR:-/tmp}/osha-api-smoke.XXXXXX"
)"


cleanup() {
    rm -rf -- "${TEMP_DIR}"
}


trap cleanup EXIT INT TERM


RESULT_FILE="${TEMP_DIR}/results.jsonl"

touch "${RESULT_FILE}"


# -------------------------------------------------------------------
# Record result
# -------------------------------------------------------------------

record_result() {
    local name="$1"
    local required="$2"
    local passed="$3"
    local message="$4"
    local details="${5:-}"

    OSHA_RESULT_NAME="${name}" \
    OSHA_RESULT_REQUIRED="${required}" \
    OSHA_RESULT_PASSED="${passed}" \
    OSHA_RESULT_MESSAGE="${message}" \
    OSHA_RESULT_DETAILS="${details}" \
    "${PYTHON}" - <<'PY' >> "${RESULT_FILE}"
from __future__ import annotations

import json
import os


details_text = os.environ.get(
    "OSHA_RESULT_DETAILS",
    "",
)


details = None


if details_text:
    try:
        details = json.loads(
            details_text
        )

    except json.JSONDecodeError:
        details = details_text


payload = {
    "name": os.environ[
        "OSHA_RESULT_NAME"
    ],

    "required": (
        os.environ[
            "OSHA_RESULT_REQUIRED"
        ]
        == "true"
    ),

    "passed": (
        os.environ[
            "OSHA_RESULT_PASSED"
        ]
        == "true"
    ),

    "message": os.environ[
        "OSHA_RESULT_MESSAGE"
    ],

    "details": details,
}


print(
    json.dumps(
        payload,
        sort_keys=True,
    )
)
PY
}


# -------------------------------------------------------------------
# HTTP GET helper
# -------------------------------------------------------------------

http_get() {
    local url="$1"
    local output="$2"
    local headers="$3"
    local error="$4"

    set +e

    HTTP_META="$(
        curl \
            --silent \
            --show-error \
            --location \
            --connect-timeout "${TIMEOUT}" \
            --max-time "${TIMEOUT}" \
            --header 'Accept: application/json' \
            --header "X-Request-ID: ${REQUEST_ID}" \
            --output "${output}" \
            --dump-header "${headers}" \
            --write-out '%{http_code}|%{time_total}' \
            "${url}" \
            2>"${error}"
    )"

    HTTP_EXIT=$?

    set -e


    HTTP_STATUS="000"
    HTTP_TIME="0"


    if [[ -n "${HTTP_META}" ]]; then
        IFS='|' read -r \
            HTTP_STATUS \
            HTTP_TIME \
            <<< "${HTTP_META}"
    fi
}


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

printf '\n'

if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '%sOSHA API HTTP Smoke Test%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Base URL: ${BASE_URL}"
    info "Request ID: ${REQUEST_ID}"
    info "Timeout: ${TIMEOUT}s"
fi


# -------------------------------------------------------------------
# 1. Liveness
# -------------------------------------------------------------------

section "Liveness"

CHECKS=$((CHECKS + 1))


if OSHA_API_URL="${BASE_URL}" \
   OSHA_API_HEALTH_TIMEOUT="${TIMEOUT}" \
   OSHA_PYTHON="${PYTHON}" \
   "${HEALTH_SCRIPT}" \
       --live \
       --quiet
then
    success \
        "API liveness"

    record_result \
        "liveness" \
        "true" \
        "true" \
        "API liveness endpoint returned success."
else
    failure \
        "API liveness"

    record_result \
        "liveness" \
        "true" \
        "false" \
        "API liveness endpoint failed."
fi


# -------------------------------------------------------------------
# Liveness is foundational
# -------------------------------------------------------------------

if ((FAILURES > 0)); then
    if [[ "${JSON_OUTPUT}" == false ]]; then
        printf '\n'

        failure_message=(
            "API is not live; remaining HTTP "
            "smoke tests cannot run reliably."
        )

        printf '%s\n' \
            "${failure_message}" >&2
    fi


    OSHA_SMOKE_BASE_URL="${BASE_URL}" \
    OSHA_SMOKE_RESULT_FILE="${RESULT_FILE}" \
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path


results = [
    json.loads(line)
    for line in Path(
        os.environ[
            "OSHA_SMOKE_RESULT_FILE"
        ]
    ).read_text().splitlines()
    if line.strip()
]


payload = {
    "schema_version": "1",
    "success": False,
    "base_url": os.environ[
        "OSHA_SMOKE_BASE_URL"
    ],
    "checks": results,
}


if os.environ.get(
    "OSHA_SMOKE_JSON",
    "0",
) == "1":
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )
PY

    exit 1
fi


# -------------------------------------------------------------------
# 2. Readiness
# -------------------------------------------------------------------

section "Readiness"

CHECKS=$((CHECKS + 1))


set +e

OSHA_API_URL="${BASE_URL}" \
OSHA_API_HEALTH_TIMEOUT="${TIMEOUT}" \
OSHA_PYTHON="${PYTHON}" \
"${HEALTH_SCRIPT}" \
    --ready \
    --quiet

READY_EXIT=$?

set -e


if ((READY_EXIT == 0)); then
    success \
        "API readiness"

    record_result \
        "readiness" \
        "$(
            if [[ "${SKIP_READY}" == true ]]; then
                printf 'false'
            else
                printf 'true'
            fi
        )" \
        "true" \
        "API readiness endpoint returned success."

else
    if [[ "${SKIP_READY}" == true ]]; then
        warn \
            "API is not ready; continuing because --skip-ready was supplied"

        record_result \
            "readiness" \
            "false" \
            "false" \
            "API readiness failed but was configured as optional."

    else
        failure \
            "API readiness"

        record_result \
            "readiness" \
            "true" \
            "false" \
            "API readiness endpoint failed."
    fi
fi


# -------------------------------------------------------------------
# 3. Full component health
# -------------------------------------------------------------------

section "Component health"

CHECKS=$((CHECKS + 1))


set +e

OSHA_API_URL="${BASE_URL}" \
OSHA_API_HEALTH_TIMEOUT="${TIMEOUT}" \
OSHA_PYTHON="${PYTHON}" \
"${HEALTH_SCRIPT}" \
    --full \
    --quiet

HEALTH_EXIT=$?

set -e


if ((HEALTH_EXIT == 0)); then
    success \
        "Component health"

    record_result \
        "component_health" \
        "true" \
        "true" \
        "Component health endpoint returned success."
else
    failure \
        "Component health"

    record_result \
        "component_health" \
        "true" \
        "false" \
        "Component health endpoint failed."
fi


# -------------------------------------------------------------------
# 4. OpenAPI
# -------------------------------------------------------------------

section "OpenAPI contract"

CHECKS=$((CHECKS + 1))


OPENAPI_BODY="${TEMP_DIR}/openapi.json"
OPENAPI_HEADERS="${TEMP_DIR}/openapi.headers"
OPENAPI_ERROR="${TEMP_DIR}/openapi.error"


http_get \
    "${BASE_URL}/openapi.json" \
    "${OPENAPI_BODY}" \
    "${OPENAPI_HEADERS}" \
    "${OPENAPI_ERROR}"


if (
    ((HTTP_EXIT == 0))
    && [[ "${HTTP_STATUS}" =~ ^2[0-9][0-9]$ ]]
); then
    if "${PYTHON}" - "${OPENAPI_BODY}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


path = Path(
    sys.argv[1]
)


document = json.loads(
    path.read_text(
        encoding="utf-8"
    )
)


if not isinstance(
    document,
    dict,
):
    raise RuntimeError(
        "OpenAPI document is not an object."
    )


if "openapi" not in document:
    raise RuntimeError(
        "OpenAPI document has no version field."
    )


paths = document.get(
    "paths"
)


if not isinstance(
    paths,
    dict,
):
    raise RuntimeError(
        "OpenAPI document has no paths map."
    )


print(
    len(paths)
)
PY
    then
        success \
            "OpenAPI document"

        record_result \
            "openapi" \
            "true" \
            "true" \
            "OpenAPI document is valid JSON and contains a paths map."

    else
        failure \
            "OpenAPI document is malformed"

        record_result \
            "openapi" \
            "true" \
            "false" \
            "OpenAPI endpoint returned an invalid document."
    fi

else
    failure \
        "OpenAPI endpoint: HTTP ${HTTP_STATUS}"

    record_result \
        "openapi" \
        "true" \
        "false" \
        "OpenAPI endpoint could not be retrieved."
fi


# -------------------------------------------------------------------
# 5. Expected routes in OpenAPI
# -------------------------------------------------------------------

section "Route contracts"

CHECKS=$((CHECKS + 1))


if [[ -s "${OPENAPI_BODY}" ]]; then
    set +e

    ROUTE_RESULT="$(
        "${PYTHON}" - "${OPENAPI_BODY}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


document = json.loads(
    Path(
        sys.argv[1]
    ).read_text(
        encoding="utf-8"
    )
)


paths = document.get(
    "paths",
    {}
)


expected = (
    "/health",
    "/health/live",
    "/health/ready",

    "/api/v1/accelerators",
    "/api/v1/accelerators/select",
    "/api/v1/accelerators/optimize",
    "/api/v1/accelerators/{device_id}",
    "/api/v1/accelerators/{device_id}/metrics",

    "/api/v1/inference/events",
    "/api/v1/inference/anomalies",
    "/api/v1/inference/batch",

    "/api/v1/models",
    "/api/v1/models/refresh",
    "/api/v1/models/{model_id}",
    "/api/v1/models/{model_id}/load",
    "/api/v1/models/{model_id}/unload",

    "/api/v1/incidents",
    "/api/v1/incidents/{incident_id}",
    "/api/v1/incidents/{incident_id}/status",

    "/api/v1/decisions",
    "/api/v1/decisions/evaluate",
    "/api/v1/decisions/{decision_id}",

    "/api/v1/remediation/plans",
    "/api/v1/remediation/plans/{plan_id}",
    "/api/v1/remediation/plans/{plan_id}/approve",
    "/api/v1/remediation/plans/{plan_id}/execute",
)


missing = [
    path
    for path in expected
    if path not in paths
]


result = {
    "expected_count": len(
        expected
    ),

    "registered_count": len(
        paths
    ),

    "missing": missing,
}


print(
    json.dumps(
        result,
        sort_keys=True,
    )
)


raise SystemExit(
    1
    if missing
    else 0
)
PY
    )"

    ROUTE_EXIT=$?

    set -e


    if ((ROUTE_EXIT == 0)); then
        success \
            "Expected API routes are registered"

        record_result \
            "route_contracts" \
            "true" \
            "true" \
            "Expected API routes are present in OpenAPI." \
            "${ROUTE_RESULT}"

    else
        failure \
            "Expected API routes are missing"

        record_result \
            "route_contracts" \
            "true" \
            "false" \
            "One or more expected API routes are absent." \
            "${ROUTE_RESULT}"

        if [[ "${VERBOSE}" == true ]]; then
            printf '%s\n' \
                "${ROUTE_RESULT}" \
            | "${PYTHON}" -m json.tool
        fi
    fi

else
    failure \
        "Route verification skipped because OpenAPI was unavailable"

    record_result \
        "route_contracts" \
        "true" \
        "false" \
        "Route verification could not inspect OpenAPI."
fi


# -------------------------------------------------------------------
# 6. Safe GET /api/v1/accelerators
# -------------------------------------------------------------------

section "Accelerator HTTP endpoint"

CHECKS=$((CHECKS + 1))


ACCELERATORS_BODY="${TEMP_DIR}/accelerators.json"
ACCELERATORS_HEADERS="${TEMP_DIR}/accelerators.headers"
ACCELERATORS_ERROR="${TEMP_DIR}/accelerators.error"


http_get \
    "${BASE_URL}/api/v1/accelerators" \
    "${ACCELERATORS_BODY}" \
    "${ACCELERATORS_HEADERS}" \
    "${ACCELERATORS_ERROR}"


if (
    ((HTTP_EXIT == 0))
    && [[ "${HTTP_STATUS}" =~ ^2[0-9][0-9]$ ]]
); then
    if "${PYTHON}" -m json.tool \
        "${ACCELERATORS_BODY}" \
        >/dev/null 2>&1
    then
        success \
            "Accelerator inventory endpoint"

        record_result \
            "accelerator_inventory" \
            "true" \
            "true" \
            "Accelerator inventory returned JSON."
    else
        failure \
            "Accelerator endpoint returned invalid JSON"

        record_result \
            "accelerator_inventory" \
            "true" \
            "false" \
            "Accelerator endpoint returned invalid JSON."
    fi

else
    failure \
        "Accelerator inventory endpoint: HTTP ${HTTP_STATUS}"

    record_result \
        "accelerator_inventory" \
        "true" \
        "false" \
        "Accelerator inventory HTTP request failed."
fi


# -------------------------------------------------------------------
# 7. Safe GET /api/v1/models
# -------------------------------------------------------------------

section "Model HTTP endpoint"

CHECKS=$((CHECKS + 1))


MODELS_BODY="${TEMP_DIR}/models.json"
MODELS_HEADERS="${TEMP_DIR}/models.headers"
MODELS_ERROR="${TEMP_DIR}/models.error"


http_get \
    "${BASE_URL}/api/v1/models" \
    "${MODELS_BODY}" \
    "${MODELS_HEADERS}" \
    "${MODELS_ERROR}"


if (
    ((HTTP_EXIT == 0))
    && [[ "${HTTP_STATUS}" =~ ^2[0-9][0-9]$ ]]
); then
    if "${PYTHON}" -m json.tool \
        "${MODELS_BODY}" \
        >/dev/null 2>&1
    then
        success \
            "Model-list endpoint"

        record_result \
            "model_list" \
            "true" \
            "true" \
            "Model-list endpoint returned JSON."
    else
        failure \
            "Model-list endpoint returned invalid JSON"

        record_result \
            "model_list" \
            "true" \
            "false" \
            "Model-list endpoint returned invalid JSON."
    fi

else
    failure \
        "Model-list endpoint: HTTP ${HTTP_STATUS}"

    record_result \
        "model_list" \
        "true" \
        "false" \
        "Model-list HTTP request failed."
fi


# -------------------------------------------------------------------
# Verbose response information
# -------------------------------------------------------------------

if (
    [[ "${VERBOSE}" == true ]]
    && [[ "${JSON_OUTPUT}" == false ]]
); then
    section "Response samples"

    if [[ -s "${ACCELERATORS_BODY}" ]]; then
        printf 'Accelerators:\n'

        "${PYTHON}" -m json.tool \
            "${ACCELERATORS_BODY}" \
            2>/dev/null \
        || cat \
            "${ACCELERATORS_BODY}"

        printf '\n'
    fi


    if [[ -s "${MODELS_BODY}" ]]; then
        printf 'Models:\n'

        "${PYTHON}" -m json.tool \
            "${MODELS_BODY}" \
            2>/dev/null \
        || cat \
            "${MODELS_BODY}"

        printf '\n'
    fi
fi


# -------------------------------------------------------------------
# Build final JSON
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


export OSHA_SMOKE_JSON="$(
    if [[ "${JSON_OUTPUT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"


if [[ "${JSON_OUTPUT}" == true ]]; then
    OSHA_SMOKE_BASE_URL="${BASE_URL}" \
    OSHA_SMOKE_RESULT_FILE="${RESULT_FILE}" \
    OSHA_SMOKE_DURATION="${elapsed}" \
    OSHA_SMOKE_SKIP_READY="${SKIP_READY}" \
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path


results = [
    json.loads(line)
    for line in Path(
        os.environ[
            "OSHA_SMOKE_RESULT_FILE"
        ]
    ).read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]


required_failures = [
    result
    for result in results
    if (
        result.get(
            "required",
            False,
        )
        and not result.get(
            "passed",
            False,
        )
    )
]


payload = {
    "schema_version": "1",

    "success": (
        len(
            required_failures
        )
        == 0
    ),

    "base_url": os.environ[
        "OSHA_SMOKE_BASE_URL"
    ],

    "skip_ready": (
        os.environ.get(
            "OSHA_SMOKE_SKIP_READY"
        )
        == "true"
    ),

    "summary": {
        "checks": len(
            results
        ),

        "passed": sum(
            1
            for result in results
            if result.get(
                "passed",
                False,
            )
        ),

        "required_failures": len(
            required_failures
        ),

        "duration_seconds": int(
            os.environ[
                "OSHA_SMOKE_DURATION"
            ]
        ),
    },

    "checks": results,
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
# Human summary
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%s========================================%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf '%sOSHA API Smoke Test Summary%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Base URL: %s\n' \
        "${BASE_URL}"

    printf 'Checks:   %s\n' \
        "${CHECKS}"

    printf 'Failures: %s\n' \
        "${FAILURES}"

    printf 'Duration: %ss\n' \
        "${elapsed}"

    printf '%s========================================%s\n' \
        "${BOLD}" \
        "${RESET}"


    if ((FAILURES == 0)); then
        printf '\n%sAPI SMOKE TEST PASSED%s\n' \
            "${GREEN}" \
            "${RESET}"

    else
        printf '\n%sAPI SMOKE TEST FAILED%s\n' \
            "${RED}" \
            "${RESET}"
    fi
fi


# -------------------------------------------------------------------
# Exit
# -------------------------------------------------------------------

if ((FAILURES > 0)); then
    exit 1
fi


exit 0
