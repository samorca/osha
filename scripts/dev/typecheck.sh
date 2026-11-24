#!/usr/bin/env bash

#
# OSHA static type checking.
#
# Usage:
#
#   ./scripts/dev/typecheck.sh
#   ./scripts/dev/typecheck.sh --strict
#   ./scripts/dev/typecheck.sh --verbose
#   ./scripts/dev/typecheck.sh src/api
#
# Default targets:
#
#   src/
#   tests/ when present
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

STRICT=false
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
  scripts/dev/typecheck.sh [options] [path...]

Options:
  --strict      Enable MyPy strict mode.
  --verbose     Enable verbose MyPy output.
  -h, --help    Show this help.

Default targets:
  src/
  tests/ when present

Environment:
  OSHA_PYTHON   Override Python interpreter.

Examples:
  ./scripts/dev/typecheck.sh
  ./scripts/dev/typecheck.sh --strict
  ./scripts/dev/typecheck.sh src/api
  ./scripts/dev/typecheck.sh src/training
  ./scripts/dev/typecheck.sh --strict src/api
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --strict)
            STRICT=true
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
# Project validation
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
# MyPy discovery
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("mypy") else 1)' \
    >/dev/null 2>&1
then
    failure \
        "MyPy is not installed in the selected environment."

    printf '\nRun:\n\n' >&2
    printf '  ./scripts/dev/setup.sh\n\n' >&2

    exit 2
fi


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


MYPY_VERSION="$(
    "${PYTHON}" -m mypy \
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
            "Type-check target does not exist: ${target}"

        exit 2
    fi
done


# -------------------------------------------------------------------
# Count Python files
# -------------------------------------------------------------------

PYTHON_FILE_COUNT=0


for target in "${TARGETS[@]}"; do
    if [[ -f "${target}" ]]; then
        if [[ "${target}" == *.py ]]; then
            PYTHON_FILE_COUNT=$((PYTHON_FILE_COUNT + 1))
        fi

        continue
    fi

    if [[ -d "${target}" ]]; then
        count="$(
            find "${target}" \
                -type f \
                -name '*.py' \
                -print \
            | wc -l
        )"

        count="$(
            printf '%s' "${count}" \
            | tr -d '[:space:]'
        )"

        PYTHON_FILE_COUNT=$((PYTHON_FILE_COUNT + count))
    fi
done


if ((PYTHON_FILE_COUNT == 0)); then
    warn "No Python files found in selected targets."

    exit 0
fi


# -------------------------------------------------------------------
# Detect MyPy configuration
# -------------------------------------------------------------------

CONFIG_FILE=""


if [[ -f "${PROJECT_ROOT}/mypy.ini" ]]; then
    CONFIG_FILE="mypy.ini"

elif [[ -f "${PROJECT_ROOT}/.mypy.ini" ]]; then
    CONFIG_FILE=".mypy.ini"

elif [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    if grep -Eq \
        '^[[:space:]]*\[tool\.mypy\][[:space:]]*$' \
        "${PROJECT_ROOT}/pyproject.toml"
    then
        CONFIG_FILE="pyproject.toml"
    fi

elif [[ -f "${PROJECT_ROOT}/setup.cfg" ]]; then
    if grep -Eq \
        '^[[:space:]]*\[mypy\][[:space:]]*$' \
        "${PROJECT_ROOT}/setup.cfg"
    then
        CONFIG_FILE="setup.cfg"
    fi
fi


# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"


# -------------------------------------------------------------------
# Build MyPy command
# -------------------------------------------------------------------

MYPY_ARGS=()


if [[ -n "${CONFIG_FILE}" ]]; then
    MYPY_ARGS+=(
        --config-file
        "${CONFIG_FILE}"
    )
fi


if [[ "${STRICT}" == true ]]; then
    MYPY_ARGS+=(
        --strict
    )
fi


if [[ "${VERBOSE}" == true ]]; then
    MYPY_ARGS+=(
        --verbose
    )
fi


# Keep output usable both locally and in CI.
MYPY_ARGS+=(
    --explicit-package-bases
    --show-error-codes
    --show-column-numbers
)


MYPY_ARGS+=(
    "${TARGETS[@]}"
)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

printf '\n%sOSHA Static Type Check%s\n' \
    "${BOLD}" \
    "${RESET}"

info "Project root: ${PROJECT_ROOT}"
info "Python: ${PYTHON}"
info "Python version: ${PYTHON_VERSION}"
info "MyPy: ${MYPY_VERSION}"
info "Python files: ${PYTHON_FILE_COUNT}"


if [[ -n "${CONFIG_FILE}" ]]; then
    info "Configuration: ${CONFIG_FILE}"
else
    info "Configuration: MyPy defaults"
fi


if [[ "${STRICT}" == true ]]; then
    info "Mode: strict"
else
    info "Mode: normal"
fi


printf '[INFO] Targets:'

for target in "${TARGETS[@]}"; do
    printf ' %s' "${target}"
done

printf '\n'


# -------------------------------------------------------------------
# Run MyPy
# -------------------------------------------------------------------

printf '\n%sRunning MyPy%s\n' \
    "${BOLD}" \
    "${RESET}"


set +e

"${PYTHON}" -m mypy \
    "${MYPY_ARGS[@]}"

MYPY_EXIT_CODE=$?

set -e


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Type Check Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'MyPy:        %s\n' \
    "${MYPY_VERSION}"

printf 'Mode:        %s\n' \
    "$(
        if [[ "${STRICT}" == true ]]; then
            printf 'strict'
        else
            printf 'normal'
        fi
    )"

printf 'Config:      %s\n' \
    "$(
        if [[ -n "${CONFIG_FILE}" ]]; then
            printf '%s' "${CONFIG_FILE}"
        else
            printf 'defaults'
        fi
    )"

printf 'Python files: %s\n' \
    "${PYTHON_FILE_COUNT}"

printf 'Exit code:    %s\n' \
    "${MYPY_EXIT_CODE}"

printf 'Duration:     %ss\n' \
    "${elapsed}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


# -------------------------------------------------------------------
# Result
# -------------------------------------------------------------------

if ((MYPY_EXIT_CODE != 0)); then
    printf '\n%sTYPE CHECK FAILED%s\n' \
        "${RED}" \
        "${RESET}"

    printf '\nFix the errors above and rerun:\n\n'

    printf '  ./scripts/dev/typecheck.sh\n\n'

    if [[ "${STRICT}" == true ]]; then
        printf 'To compare against normal mode:\n\n'

        printf '  ./scripts/dev/typecheck.sh\n\n'
    fi

    exit "${MYPY_EXIT_CODE}"
fi


printf '\n%sTYPE CHECK PASSED%s\n' \
    "${GREEN}" \
    "${RESET}"

exit 0
