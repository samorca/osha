#!/usr/bin/env bash

#
# OSHA Python linting.
#
# Usage:
#
#   ./scripts/dev/lint.sh
#   ./scripts/dev/lint.sh --fix
#   ./scripts/dev/lint.sh --diff
#   ./scripts/dev/lint.sh --verbose
#   ./scripts/dev/lint.sh src/api
#
# Default:
#
#   Ruff lint checks src/ and tests/ when tests/
#   exists.
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
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

FIX=false
DIFF=false
VERBOSE=false

TARGETS=()

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
  scripts/dev/lint.sh [options] [path...]

Options:
  --fix        Apply Ruff's available lint fixes.
  --diff       Preview available fixes as a diff.
  --verbose    Enable verbose Ruff output.
  -h, --help   Show this help.

Default targets:
  src/
  tests/       when present

Environment:
  OSHA_PYTHON  Override Python interpreter.

Examples:
  ./scripts/dev/lint.sh
  ./scripts/dev/lint.sh --fix
  ./scripts/dev/lint.sh --diff
  ./scripts/dev/lint.sh src/api
  ./scripts/dev/lint.sh src/api src/accelerators
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --fix)
            FIX=true
            ;;

        --diff)
            DIFF=true
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
                TARGETS+=(
                    "$1"
                )

                shift
            done

            break
            ;;

        -*)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;

        *)
            TARGETS+=(
                "$1"
            )
            ;;
    esac

    shift
done


# -------------------------------------------------------------------
# Argument validation
# -------------------------------------------------------------------

if [[ "${FIX}" == true ]] \
    && [[ "${DIFF}" == true ]]; then
    failure \
        "--fix and --diff cannot be used together"

    exit 2
fi


# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -d "${PROJECT_ROOT}/src" ]]; then
    failure \
        "src/ directory not found: ${PROJECT_ROOT}/src"

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
    failure "No Python interpreter found."
    exit 2
fi


if [[ ! -x "${PYTHON}" ]]; then
    failure \
        "Python interpreter is not executable: ${PYTHON}"

    exit 2
fi


# -------------------------------------------------------------------
# Ruff discovery
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("ruff") else 1)' \
    >/dev/null 2>&1
then
    failure \
        "Ruff is not installed in the selected environment."

    printf '\nRun:\n\n' >&2
    printf '  ./scripts/dev/setup.sh\n\n' >&2

    exit 2
fi


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"

RUFF_VERSION="$(
    "${PYTHON}" -m ruff \
        --version
)"


# -------------------------------------------------------------------
# Default targets
# -------------------------------------------------------------------

if ((${#TARGETS[@]} == 0)); then
    TARGETS+=(
        "src"
    )

    if [[ -d "${PROJECT_ROOT}/tests" ]]; then
        TARGETS+=(
            "tests"
        )
    fi
fi


# -------------------------------------------------------------------
# Validate targets
# -------------------------------------------------------------------

for target in "${TARGETS[@]}"; do
    if [[ ! -e "${PROJECT_ROOT}/${target}" ]] \
        && [[ ! -e "${target}" ]]
    then
        failure \
            "Lint target does not exist: ${target}"

        exit 2
    fi
done


# -------------------------------------------------------------------
# Ruff arguments
# -------------------------------------------------------------------

RUFF_ARGS=(
    check
)


if [[ "${FIX}" == true ]]; then
    RUFF_ARGS+=(
        --fix
    )

elif [[ "${DIFF}" == true ]]; then
    RUFF_ARGS+=(
        --diff
    )
fi


if [[ "${VERBOSE}" == true ]]; then
    RUFF_ARGS+=(
        --verbose
    )
fi


RUFF_ARGS+=(
    "${TARGETS[@]}"
)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

printf '\n%sOSHA Python Lint%s\n' \
    "${BOLD}" \
    "${RESET}"

info "Project root: ${PROJECT_ROOT}"
info "Python: ${PYTHON}"
info "Python version: ${PYTHON_VERSION}"
info "Ruff: ${RUFF_VERSION}"

info "Mode: $(
    if [[ "${FIX}" == true ]]; then
        printf 'fix'
    elif [[ "${DIFF}" == true ]]; then
        printf 'diff'
    else
        printf 'check'
    fi
)"


printf '[INFO] Targets:'

for target in "${TARGETS[@]}"; do
    printf ' %s' "${target}"
done

printf '\n'


# -------------------------------------------------------------------
# Execute Ruff
# -------------------------------------------------------------------

printf '\n%sRunning Ruff%s\n' \
    "${BOLD}" \
    "${RESET}"


set +e

"${PYTHON}" -m ruff \
    "${RUFF_ARGS[@]}"

RUFF_EXIT_CODE=$?

set -e


# -------------------------------------------------------------------
# Result
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Lint Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Ruff:      %s\n' \
    "${RUFF_VERSION}"

printf 'Mode:      %s\n' \
    "$(
        if [[ "${FIX}" == true ]]; then
            printf 'fix'
        elif [[ "${DIFF}" == true ]]; then
            printf 'diff'
        else
            printf 'check'
        fi
    )"

printf 'Exit code: %s\n' \
    "${RUFF_EXIT_CODE}"

printf 'Duration:  %ss\n' \
    "${elapsed}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


if ((RUFF_EXIT_CODE != 0)); then
    printf '\n%sLINT FAILED%s\n' \
        "${RED}" \
        "${RESET}"

    if [[ "${FIX}" == false ]]; then
        printf '\nInspect the violations above.\n'

        printf 'Preview automatic fixes with:\n\n'

        printf '  ./scripts/dev/lint.sh --diff\n\n'

        printf 'Apply available fixes with:\n\n'

        printf '  ./scripts/dev/lint.sh --fix\n\n'
    fi

    exit "${RUFF_EXIT_CODE}"
fi


printf '\n%sLINT PASSED%s\n' \
    "${GREEN}" \
    "${RESET}"

exit 0
