#!/usr/bin/env bash

#
# OSHA test-suite runner.
#
# Usage:
#
#   ./scripts/ci/test.sh
#   ./scripts/ci/test.sh --verbose
#   ./scripts/ci/test.sh --fail-fast
#   ./scripts/ci/test.sh --unit
#
# Additional arguments after -- are forwarded
# directly to pytest:
#
#   ./scripts/ci/test.sh -- -k accelerator
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#   OSHA_TEST_PATH=tests
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

VERBOSE=false
FAIL_FAST=false
UNIT_ONLY=false

TEST_PATH="${OSHA_TEST_PATH:-tests}"

PYTEST_EXTRA_ARGS=()

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
  scripts/ci/test.sh [options] [-- pytest-arguments...]

Options:
  --verbose     Verbose pytest output.
  --fail-fast   Stop on first test failure.
  --unit        Run tests/unit when available.
  -h, --help    Show this help.

Environment:
  OSHA_PYTHON     Override Python interpreter.
  OSHA_TEST_PATH  Override default test directory.

Examples:
  ./scripts/ci/test.sh
  ./scripts/ci/test.sh --verbose
  ./scripts/ci/test.sh --fail-fast
  ./scripts/ci/test.sh --unit
  ./scripts/ci/test.sh -- -k accelerator
  ./scripts/ci/test.sh -- tests/api
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --verbose)
            VERBOSE=true
            shift
            ;;

        --fail-fast)
            FAIL_FAST=true
            shift
            ;;

        --unit)
            UNIT_ONLY=true
            shift
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        --)
            shift

            while (($# > 0)); do
                PYTEST_EXTRA_ARGS+=(
                    "$1"
                )

                shift
            done
            ;;

        *)
            PYTEST_EXTRA_ARGS+=(
                "$1"
            )

            shift
            ;;
    esac
done


# -------------------------------------------------------------------
# Project root
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
    failure "No Python interpreter found."
    exit 2
fi


if [[ ! -x "${PYTHON}" ]]; then
    failure "Python interpreter is not executable: ${PYTHON}"
    exit 2
fi


# -------------------------------------------------------------------
# Pytest check
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import pytest' \
    >/dev/null 2>&1
then
    failure "pytest is not installed in ${PYTHON}"
    exit 2
fi


PYTEST_VERSION="$(
    "${PYTHON}" -c \
        'import pytest; print(pytest.__version__)'
)"

PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


# -------------------------------------------------------------------
# Select test target
# -------------------------------------------------------------------

if [[ "${UNIT_ONLY}" == true ]]; then
    if [[ -d "${PROJECT_ROOT}/tests/unit" ]]; then
        TEST_TARGET="tests/unit"

    else
        failure \
            "--unit requested but tests/unit does not exist"

        exit 2
    fi

else
    TEST_TARGET="${TEST_PATH}"
fi


if [[ ! -e "${PROJECT_ROOT}/${TEST_TARGET}" ]]; then
    failure "Test target does not exist: ${TEST_TARGET}"
    exit 2
fi


# -------------------------------------------------------------------
# Count test files
# -------------------------------------------------------------------

if [[ -d "${PROJECT_ROOT}/${TEST_TARGET}" ]]; then
    TEST_FILE_COUNT="$(
        find "${PROJECT_ROOT}/${TEST_TARGET}" \
            -type f \
            \( \
                -name 'test_*.py' \
                -o \
                -name '*_test.py' \
            \) \
            -print \
        | wc -l
    )"

    TEST_FILE_COUNT="$(
        printf '%s' "${TEST_FILE_COUNT}" \
        | tr -d '[:space:]'
    )"

else
    TEST_FILE_COUNT=1
fi


if ((TEST_FILE_COUNT == 0)); then
    failure \
        "No pytest test files found under ${TEST_TARGET}"

    exit 5
fi


# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTHONUNBUFFERED=1


# -------------------------------------------------------------------
# Build pytest command
# -------------------------------------------------------------------

PYTEST_ARGS=()


if [[ "${VERBOSE}" == true ]]; then
    PYTEST_ARGS+=(
        -vv
    )

else
    PYTEST_ARGS+=(
        -q
    )
fi


if [[ "${FAIL_FAST}" == true ]]; then
    PYTEST_ARGS+=(
        -x
    )
fi


PYTEST_ARGS+=(
    "${TEST_TARGET}"
)


if ((${#PYTEST_EXTRA_ARGS[@]} > 0)); then
    PYTEST_ARGS+=(
        "${PYTEST_EXTRA_ARGS[@]}"
    )
fi


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

printf '\n%sOSHA Test Suite%s\n' \
    "${BOLD}" \
    "${RESET}"

info "Project root: ${PROJECT_ROOT}"
info "Python: ${PYTHON}"
info "Python version: ${PYTHON_VERSION}"
info "pytest version: ${PYTEST_VERSION}"
info "Test target: ${TEST_TARGET}"
info "Test files: ${TEST_FILE_COUNT}"


printf '\n%sRunning tests%s\n' \
    "${BOLD}" \
    "${RESET}"


# -------------------------------------------------------------------
# Run pytest
# -------------------------------------------------------------------

set +e

"${PYTHON}" -m pytest \
    "${PYTEST_ARGS[@]}"

PYTEST_EXIT_CODE=$?

set -e


# -------------------------------------------------------------------
# Result handling
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Test Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Python:      %s\n' \
    "${PYTHON_VERSION}"

printf 'pytest:      %s\n' \
    "${PYTEST_VERSION}"

printf 'Target:      %s\n' \
    "${TEST_TARGET}"

printf 'Test files:  %s\n' \
    "${TEST_FILE_COUNT}"

printf 'Exit code:   %s\n' \
    "${PYTEST_EXIT_CODE}"

printf 'Duration:    %ss\n' \
    "${elapsed}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


case "${PYTEST_EXIT_CODE}" in
    0)
        printf '\n%sTESTS PASSED%s\n' \
            "${GREEN}" \
            "${RESET}"

        exit 0
        ;;

    1)
        printf '\n%sTESTS FAILED%s\n' \
            "${RED}" \
            "${RESET}"

        exit 1
        ;;

    2)
        failure "pytest was interrupted"
        exit 2
        ;;

    3)
        failure "pytest encountered an internal error"
        exit 3
        ;;

    4)
        failure "pytest command-line/configuration error"
        exit 4
        ;;

    5)
        failure "pytest collected no tests"
        exit 5
        ;;

    6)
        failure "pytest warning limit was exceeded"
        exit 6
        ;;

    *)
        failure \
            "pytest returned unexpected exit code ${PYTEST_EXIT_CODE}"

        exit "${PYTEST_EXIT_CODE}"
        ;;
esac
