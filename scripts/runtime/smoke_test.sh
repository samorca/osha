#!/usr/bin/env bash

#
# OSHA orchestration runtime smoke test.
#
# Default behavior:
#
#   compile src/runtime/
#   import runtime package boundaries
#   validate the runtime main() entrypoint
#   inspect runtime public modules
#   run the accelerator smoke test
#   optionally validate a worker entrypoint
#
# This command does NOT:
#
#   start the long-running orchestration runtime
#   start a long-running worker loop
#   execute remediation
#   modify infrastructure
#   create incidents
#   promote or load models
#
# Usage:
#
#   ./scripts/runtime/smoke_test.sh
#
#   ./scripts/runtime/smoke_test.sh \
#       --module src.runtime.main
#
#   ./scripts/runtime/smoke_test.sh \
#       --worker telemetry
#
#   ./scripts/runtime/smoke_test.sh \
#       --worker-module src.runtime.workers.telemetry
#
#   ./scripts/runtime/smoke_test.sh \
#       --skip-accelerator
#
#   ./scripts/runtime/smoke_test.sh \
#       --json
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#   OSHA_RUNTIME_MODULE=src.runtime.main
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

ACCELERATOR_SMOKE="${PROJECT_ROOT}/scripts/accelerators/smoke_test.sh"


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

RUNTIME_MODULE="${OSHA_RUNTIME_MODULE:-src.runtime.main}"

WORKER_MODULE=""

WORKER_TYPE=""

CHECK_WORKER=false
SKIP_ACCELERATOR=false
JSON_OUTPUT=false
VERBOSE=false
STRICT=false
WRAPPER_INFO=false

CHECKS=0
FAILURES=0
WARNINGS=0

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
    WARNINGS=$((WARNINGS + 1))

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
  scripts/runtime/smoke_test.sh [options]

Options:
  --module MODULE
      Runtime entry module.

      Default:
          src.runtime.main

  --worker TYPE
      Validate a named worker entrypoint.

      Candidate modules:

          src.runtime.workers.TYPE
          src.runtime.TYPE_worker
          src.runtime.TYPE

  --worker-module MODULE
      Validate an explicit worker entry module.

  --skip-accelerator
      Skip the AcceleratorManager execution smoke
      test.

  --strict
      Treat optional subsystem import failures as
      smoke-test failures.

  --json
      Emit machine-readable JSON.

  --verbose
      Include module inventory and detailed results.

  --wrapper-info
      Show resolved configuration without running
      the checks.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override the Python interpreter.

  OSHA_RUNTIME_MODULE
      Override the runtime entry module.

Examples:
  ./scripts/runtime/smoke_test.sh

  ./scripts/runtime/smoke_test.sh \
      --module src.runtime.main

  ./scripts/runtime/smoke_test.sh \
      --worker telemetry

  ./scripts/runtime/smoke_test.sh \
      --worker-module src.runtime.worker

  ./scripts/runtime/smoke_test.sh \
      --strict \
      --verbose

  ./scripts/runtime/smoke_test.sh \
      --json

Exit status:
  0
      All required checks passed.

  1
      One or more required runtime checks failed.

  2
      Invalid script configuration or missing
      prerequisites.
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

            RUNTIME_MODULE="$2"
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
            CHECK_WORKER=true
            shift
            ;;

        --worker-module)
            if (($# < 2)); then
                printf \
                    'ERROR: --worker-module requires a value.\n' \
                    >&2

                exit 2
            fi

            WORKER_MODULE="$2"
            CHECK_WORKER=true
            shift
            ;;

        --skip-accelerator)
            SKIP_ACCELERATOR=true
            ;;

        --strict)
            STRICT=true
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

if [[ -z "${RUNTIME_MODULE}" ]]; then
    printf \
        'ERROR: runtime module cannot be empty.\n' \
        >&2

    exit 2
fi


if [[ ! "${RUNTIME_MODULE}" =~ ^[A-Za-z_][A-Za-z0-9_.]*$ ]]; then
    printf \
        'ERROR: invalid runtime module: %s\n' \
        "${RUNTIME_MODULE}" >&2

    exit 2
fi


if [[ -n "${WORKER_TYPE}" ]] \
    && [[ -n "${WORKER_MODULE}" ]]; then
    printf \
        'ERROR: --worker and --worker-module cannot be combined.\n' \
        >&2

    exit 2
fi


if [[ -n "${WORKER_TYPE}" ]] \
    && [[ ! "${WORKER_TYPE}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]
then
    printf \
        'ERROR: invalid worker type: %s\n' \
        "${WORKER_TYPE}" >&2

    exit 2
fi


if [[ -n "${WORKER_MODULE}" ]] \
    && [[ ! "${WORKER_MODULE}" =~ ^[A-Za-z_][A-Za-z0-9_.]*$ ]]
then
    printf \
        'ERROR: invalid worker module: %s\n' \
        "${WORKER_MODULE}" >&2

    exit 2
fi


# -------------------------------------------------------------------
# Project validation
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -d "${PROJECT_ROOT}/src/runtime" ]]; then
    printf \
        'ERROR: src/runtime does not exist.\n' \
        >&2

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


export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTHONUNBUFFERED=1


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


# -------------------------------------------------------------------
# Wrapper information
# -------------------------------------------------------------------

if [[ "${WRAPPER_INFO}" == true ]]; then
    printf '\n%sOSHA Runtime Smoke Test%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Project root:      %s\n' \
        "${PROJECT_ROOT}"

    printf 'Python:            %s\n' \
        "${PYTHON}"

    printf 'Python version:    %s\n' \
        "${PYTHON_VERSION}"

    printf 'Runtime module:    %s\n' \
        "${RUNTIME_MODULE}"

    printf 'Check worker:      %s\n' \
        "${CHECK_WORKER}"

    printf 'Worker type:       %s\n' \
        "${WORKER_TYPE:-<not specified>}"

    printf 'Worker module:     %s\n' \
        "${WORKER_MODULE:-<not specified>}"

    printf 'Skip accelerator:  %s\n' \
        "${SKIP_ACCELERATOR}"

    printf 'Strict:            %s\n' \
        "${STRICT}"

    printf 'JSON:              %s\n' \
        "${JSON_OUTPUT}"

    printf 'Verbose:           %s\n' \
        "${VERBOSE}"

    exit 0
fi


# -------------------------------------------------------------------
# Result storage
# -------------------------------------------------------------------

TEMP_DIR="$(
    mktemp -d \
        "${TMPDIR:-/tmp}/osha-runtime-smoke.XXXXXX"
)"

RESULT_FILE="${TEMP_DIR}/results.jsonl"


cleanup() {
    rm -rf -- "${TEMP_DIR}"
}


trap cleanup EXIT INT TERM


touch "${RESULT_FILE}"


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


print(
    json.dumps(
        {
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
        },
        sort_keys=True,
    )
)
PY
}


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%sOSHA Runtime Smoke Test%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Project root: ${PROJECT_ROOT}"
    info "Python: ${PYTHON_VERSION}"
    info "Runtime module: ${RUNTIME_MODULE}"
    info "Strict: ${STRICT}"
fi


# -------------------------------------------------------------------
# 1. Compile runtime package
# -------------------------------------------------------------------

section "Runtime compilation"

CHECKS=$((CHECKS + 1))


if "${PYTHON}" -m compileall \
    -q \
    -f \
    "${PROJECT_ROOT}/src/runtime"
then
    success \
        "Runtime source compilation"

    record_result \
        "runtime_compilation" \
        "true" \
        "true" \
        "src/runtime compiled successfully."

else
    failure \
        "Runtime source compilation"

    record_result \
        "runtime_compilation" \
        "true" \
        "false" \
        "src/runtime failed Python compilation."
fi


# -------------------------------------------------------------------
# 2. Runtime package import
# -------------------------------------------------------------------

section "Runtime package"

CHECKS=$((CHECKS + 1))


set +e

RUNTIME_PACKAGE_RESULT="$(
    "${PYTHON}" - <<'PY' 2>&1
from __future__ import annotations

import json

import src.runtime as runtime


print(
    json.dumps(
        {
            "module": runtime.__name__,

            "source": getattr(
                runtime,
                "__file__",
                None,
            ),
        },
        sort_keys=True,
    )
)
PY
)"

RUNTIME_PACKAGE_EXIT=$?

set -e


if ((RUNTIME_PACKAGE_EXIT == 0)); then
    success \
        "Import src.runtime"

    record_result \
        "runtime_package" \
        "true" \
        "true" \
        "src.runtime imported successfully." \
        "${RUNTIME_PACKAGE_RESULT}"

else
    failure \
        "Import src.runtime"

    record_result \
        "runtime_package" \
        "true" \
        "false" \
        "src.runtime import failed." \
        "${RUNTIME_PACKAGE_RESULT}"
fi


# -------------------------------------------------------------------
# 3. Runtime entrypoint
# -------------------------------------------------------------------

section "Runtime entrypoint"

CHECKS=$((CHECKS + 1))


export OSHA_SMOKE_RUNTIME_MODULE="${RUNTIME_MODULE}"


set +e

RUNTIME_ENTRY_RESULT="$(
    "${PYTHON}" - <<'PY' 2>&1
from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path


name = os.environ[
    "OSHA_SMOKE_RUNTIME_MODULE"
]


spec = importlib.util.find_spec(
    name
)


if spec is None:
    print(
        json.dumps(
            {
                "module": name,
                "found": False,
            },
            sort_keys=True,
        )
    )

    raise SystemExit(1)


module = importlib.import_module(
    name
)


source = getattr(
    module,
    "__file__",
    None,
)


has_main = callable(
    getattr(
        module,
        "main",
        None,
    )
)


payload = {
    "module": name,

    "found": True,

    "importable": True,

    "source": (
        str(
            Path(source).resolve()
        )
        if source
        else None
    ),

    "has_main": has_main,
}


print(
    json.dumps(
        payload,
        sort_keys=True,
    )
)


raise SystemExit(
    0
    if has_main
    else 1
)
PY
)"

RUNTIME_ENTRY_EXIT=$?

set -e


if ((RUNTIME_ENTRY_EXIT == 0)); then
    success \
        "Runtime entrypoint"

    record_result \
        "runtime_entrypoint" \
        "true" \
        "true" \
        "Runtime module exists, imports and exposes main()." \
        "${RUNTIME_ENTRY_RESULT}"

else
    failure \
        "Runtime entrypoint"

    record_result \
        "runtime_entrypoint" \
        "true" \
        "false" \
        "Runtime module is missing, unimportable or has no main()." \
        "${RUNTIME_ENTRY_RESULT}"
fi


# -------------------------------------------------------------------
# 4. Runtime subsystem modules
# -------------------------------------------------------------------

section "Runtime subsystem boundaries"

CHECKS=$((CHECKS + 1))


STRICT_VALUE=0


if [[ "${STRICT}" == true ]]; then
    STRICT_VALUE=1
fi


set +e

SUBSYSTEM_RESULT="$(
    OSHA_RUNTIME_STRICT="${STRICT_VALUE}" \
    "${PYTHON}" - <<'PY' 2>&1
from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path


runtime_root = Path(
    "src/runtime"
)


required = [
    "src.runtime",
]


preferred = [
    "src.accelerators",
    "src.ai",
    "src.api",
]


# Discover implemented runtime modules rather than
# guessing that every possible subsystem exists.
discovered = []


for path in sorted(
    runtime_root.rglob(
        "*.py"
    )
):
    if path.name == "__init__.py":
        relative = path.parent

    else:
        relative = path.with_suffix(
            ""
        )

    parts = relative.parts

    if not parts:
        continue

    module = ".".join(
        parts
    )

    if module:
        discovered.append(
            module
        )


modules = []


for module in (
    required
    + preferred
    + discovered
):
    if module not in modules:
        modules.append(
            module
        )


results = []


for module in modules:
    preferred_module = (
        module in preferred
    )

    required_module = (
        module in required
        or module in discovered
    )

    spec = importlib.util.find_spec(
        module
    )


    if spec is None:
        results.append(
            {
                "module": module,

                "implemented": False,

                "required": (
                    required_module
                ),

                "preferred": (
                    preferred_module
                ),

                "imported": False,

                "error": (
                    "module_not_found"
                ),
            }
        )

        continue


    try:
        importlib.import_module(
            module
        )

        results.append(
            {
                "module": module,

                "implemented": True,

                "required": (
                    required_module
                ),

                "preferred": (
                    preferred_module
                ),

                "imported": True,
            }
        )

    except Exception as exc:
        results.append(
            {
                "module": module,

                "implemented": True,

                "required": (
                    required_module
                ),

                "preferred": (
                    preferred_module
                ),

                "imported": False,

                "error_type": (
                    type(exc).__name__
                ),

                "error": str(
                    exc
                ),
            }
        )


strict = (
    os.environ.get(
        "OSHA_RUNTIME_STRICT",
        "0",
    )
    == "1"
)


required_failures = [
    result
    for result in results
    if (
        result[
            "required"
        ]
        and not result[
            "imported"
        ]
    )
]


preferred_failures = [
    result
    for result in results
    if (
        result[
            "preferred"
        ]
        and not result[
            "imported"
        ]
    )
]


payload = {
    "discovered_runtime_modules": len(
        discovered
    ),

    "required_failures": len(
        required_failures
    ),

    "preferred_failures": len(
        preferred_failures
    ),

    "modules": results,
}


print(
    json.dumps(
        payload,
        sort_keys=True,
    )
)


failed = bool(
    required_failures
    or (
        strict
        and preferred_failures
    )
)


raise SystemExit(
    1
    if failed
    else 0
)
PY
)"

SUBSYSTEM_EXIT=$?

set -e


if ((SUBSYSTEM_EXIT == 0)); then
    success \
        "Runtime subsystem imports"

    record_result \
        "runtime_subsystems" \
        "true" \
        "true" \
        "Implemented runtime modules imported successfully." \
        "${SUBSYSTEM_RESULT}"

else
    failure \
        "Runtime subsystem imports"

    record_result \
        "runtime_subsystems" \
        "true" \
        "false" \
        "One or more runtime subsystem modules failed to import." \
        "${SUBSYSTEM_RESULT}"
fi


# -------------------------------------------------------------------
# 5. Optional worker validation
# -------------------------------------------------------------------

if [[ "${CHECK_WORKER}" == true ]]; then
    section "Worker entrypoint"

    CHECKS=$((CHECKS + 1))


    CANDIDATES=()


    if [[ -n "${WORKER_MODULE}" ]]; then
        CANDIDATES+=(
            "${WORKER_MODULE}"
        )

    else
        CANDIDATES+=(
            "src.runtime.workers.${WORKER_TYPE}"
            "src.runtime.${WORKER_TYPE}_worker"
            "src.runtime.${WORKER_TYPE}"
        )
    fi


    export OSHA_SMOKE_WORKER_CANDIDATES="$(
        printf '%s\n' \
            "${CANDIDATES[@]}"
    )"


    set +e

    WORKER_RESULT="$(
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
        "OSHA_SMOKE_WORKER_CANDIDATES",
        "",
    ).splitlines()
    if value.strip()
]


attempts = []


for name in candidates:
    try:
        spec = importlib.util.find_spec(
            name
        )

    except Exception as exc:
        attempts.append(
            {
                "module": name,
                "found": False,
                "error_type": (
                    type(exc).__name__
                ),
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


    try:
        module = importlib.import_module(
            name
        )

    except Exception as exc:
        attempts.append(
            {
                "module": name,
                "found": True,
                "imported": False,
                "error_type": (
                    type(exc).__name__
                ),
                "error": str(
                    exc
                ),
            }
        )

        continue


    source = getattr(
        module,
        "__file__",
        None,
    )

    has_main = callable(
        getattr(
            module,
            "main",
            None,
        )
    )


    attempts.append(
        {
            "module": name,
            "found": True,
            "imported": True,
            "has_main": has_main,
            "source": (
                str(
                    Path(source).resolve()
                )
                if source
                else None
            ),
        }
    )


    if has_main:
        print(
            json.dumps(
                {
                    "success": True,
                    "resolved_module": name,
                    "attempts": attempts,
                },
                sort_keys=True,
            )
        )

        raise SystemExit(0)


print(
    json.dumps(
        {
            "success": False,
            "attempts": attempts,
        },
        sort_keys=True,
    )
)


raise SystemExit(1)
PY
    )"

    WORKER_EXIT=$?

    set -e


    if ((WORKER_EXIT == 0)); then
        success \
            "Worker entrypoint"

        record_result \
            "worker_entrypoint" \
            "true" \
            "true" \
            "Worker module exists, imports and exposes main()." \
            "${WORKER_RESULT}"

    else
        failure \
            "Worker entrypoint"

        record_result \
            "worker_entrypoint" \
            "true" \
            "false" \
            "No matching executable worker module was found." \
            "${WORKER_RESULT}"
    fi
fi


# -------------------------------------------------------------------
# 6. Accelerator execution path
# -------------------------------------------------------------------

if [[ "${SKIP_ACCELERATOR}" == false ]]; then
    section "Accelerator execution"

    CHECKS=$((CHECKS + 1))


    if [[ ! -x "${ACCELERATOR_SMOKE}" ]]; then
        failure \
            "Accelerator smoke script is unavailable"

        record_result \
            "accelerator_execution" \
            "true" \
            "false" \
            "scripts/accelerators/smoke_test.sh is unavailable."

    else
        set +e

        OSHA_PYTHON="${PYTHON}" \
            "${ACCELERATOR_SMOKE}" \
            --kind auto \
            >/dev/null 2>&1

        ACCELERATOR_EXIT=$?

        set -e


        if ((ACCELERATOR_EXIT == 0)); then
            success \
                "Accelerator execution path"

            record_result \
                "accelerator_execution" \
                "true" \
                "true" \
                "AcceleratorManager selected a runtime device and completed a computation."

        else
            failure \
                "Accelerator execution path"

            record_result \
                "accelerator_execution" \
                "true" \
                "false" \
                "Accelerator smoke test failed."
        fi
    fi

else
    section "Accelerator execution"

    info \
        "Accelerator smoke test skipped"

    record_result \
        "accelerator_execution" \
        "false" \
        "false" \
        "Accelerator smoke test was skipped."
fi


# -------------------------------------------------------------------
# Verbose details
# -------------------------------------------------------------------

if [[ "${VERBOSE}" == true ]] \
    && [[ "${JSON_OUTPUT}" == false ]]; then
    section "Detailed results"

    if [[ -s "${RESULT_FILE}" ]]; then
        "${PYTHON}" - "${RESULT_FILE}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


for line in Path(
    sys.argv[1]
).read_text(
    encoding="utf-8"
).splitlines():
    if not line.strip():
        continue

    result = json.loads(
        line
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print()
PY
    fi
fi


# -------------------------------------------------------------------
# Final summary data
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


SUMMARY="$(
    OSHA_RESULT_FILE="${RESULT_FILE}" \
    OSHA_RUNTIME_MODULE="${RUNTIME_MODULE}" \
    OSHA_RUNTIME_DURATION="${elapsed}" \
    OSHA_RUNTIME_STRICT="$(
        if [[ "${STRICT}" == true ]]; then
            printf '1'
        else
            printf '0'
        fi
    )" \
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path


results = [
    json.loads(line)
    for line in Path(
        os.environ[
            "OSHA_RESULT_FILE"
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


warnings = [
    result
    for result in results
    if (
        not result.get(
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

    "runtime_module": os.environ[
        "OSHA_RUNTIME_MODULE"
    ],

    "strict": (
        os.environ[
            "OSHA_RUNTIME_STRICT"
        ]
        == "1"
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

        "warnings": len(
            warnings
        ),

        "duration_seconds": int(
            os.environ[
                "OSHA_RUNTIME_DURATION"
            ]
        ),
    },

    "checks": results,
}


print(
    json.dumps(
        payload,
        sort_keys=True,
    )
)
PY
)"


# -------------------------------------------------------------------
# JSON output
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == true ]]; then
    printf '%s\n' \
        "${SUMMARY}" \
    | "${PYTHON}" -m json.tool
fi


# -------------------------------------------------------------------
# Human summary
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == false ]]; then
    REQUIRED_FAILURE_COUNT="$(
        SUMMARY="${SUMMARY}" \
        "${PYTHON}" - <<'PY'
import json
import os


print(
    json.loads(
        os.environ[
            "SUMMARY"
        ]
    )[
        "summary"
    ][
        "required_failures"
    ]
)
PY
    )"


    WARNING_COUNT="$(
        SUMMARY="${SUMMARY}" \
        "${PYTHON}" - <<'PY'
import json
import os


print(
    json.loads(
        os.environ[
            "SUMMARY"
        ]
    )[
        "summary"
    ][
        "warnings"
    ]
)
PY
    )"


    printf '\n%s========================================%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf '%sOSHA Runtime Smoke Test Summary%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Runtime module:    %s\n' \
        "${RUNTIME_MODULE}"

    printf 'Checks:            %s\n' \
        "${CHECKS}"

    printf 'Required failures: %s\n' \
        "${REQUIRED_FAILURE_COUNT}"

    printf 'Warnings:          %s\n' \
        "${WARNING_COUNT}"

    printf 'Strict:            %s\n' \
        "${STRICT}"

    printf 'Duration:          %ss\n' \
        "${elapsed}"

    printf '%s========================================%s\n' \
        "${BOLD}" \
        "${RESET}"


    if ((REQUIRED_FAILURE_COUNT == 0)); then
        printf '\n%sRUNTIME SMOKE TEST PASSED%s\n' \
            "${GREEN}" \
            "${RESET}"

    else
        printf '\n%sRUNTIME SMOKE TEST FAILED%s\n' \
            "${RED}" \
            "${RESET}"
    fi
fi


# -------------------------------------------------------------------
# Exit status
# -------------------------------------------------------------------

SUCCESS="$(
    SUMMARY="${SUMMARY}" \
    "${PYTHON}" - <<'PY'
import json
import os


print(
    "1"
    if json.loads(
        os.environ[
            "SUMMARY"
        ]
    )[
        "success"
    ]
    else "0"
)
PY
)"


if [[ "${SUCCESS}" == "1" ]]; then
    exit 0
fi


exit 1
