#!/usr/bin/env bash

#
# OSHA developer test runner.
#
# Thin developer wrapper around:
#
#     scripts/ci/test.sh
#
# The CI test script remains the authoritative
# pytest implementation.
#
# Usage:
#
#   ./scripts/dev/test.sh
#   ./scripts/dev/test.sh --verbose
#   ./scripts/dev/test.sh --fail-fast
#   ./scripts/dev/test.sh --unit
#   ./scripts/dev/test.sh --api
#   ./scripts/dev/test.sh --accelerators
#   ./scripts/dev/test.sh --training
#   ./scripts/dev/test.sh --keyword NAME
#   ./scripts/dev/test.sh --marker NAME
#
# Arbitrary pytest arguments can be supplied after:
#
#   --
#
# Examples:
#
#   ./scripts/dev/test.sh -- -k accelerator
#   ./scripts/dev/test.sh -- tests/api/test_health.py
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

CI_TEST_SCRIPT="${PROJECT_ROOT}/scripts/ci/test.sh"


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

VERBOSE=false
FAIL_FAST=false
UNIT_ONLY=false

SUBSYSTEM=""

KEYWORD=""

MARKER=""

EXTRA_ARGS=()


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
  scripts/dev/test.sh [options] [-- pytest-arguments...]

Options:
  --verbose
      Verbose pytest output.

  --fail-fast
      Stop after the first test failure.

  --unit
      Run tests/unit.

  --api
      Run tests/api when available.

  --accelerators
      Run tests/accelerators when available.

  --training
      Run tests/training when available.

  --keyword EXPRESSION
      Pass -k EXPRESSION to pytest.

  --marker EXPRESSION
      Pass -m EXPRESSION to pytest.

  -h, --help
      Show this help.

Examples:
  ./scripts/dev/test.sh

  ./scripts/dev/test.sh --verbose

  ./scripts/dev/test.sh --fail-fast

  ./scripts/dev/test.sh --unit

  ./scripts/dev/test.sh --api

  ./scripts/dev/test.sh --accelerators

  ./scripts/dev/test.sh --training

  ./scripts/dev/test.sh \
      --keyword accelerator

  ./scripts/dev/test.sh \
      --marker slow

  ./scripts/dev/test.sh -- \
      tests/api/test_health.py

  ./scripts/dev/test.sh -- \
      tests/api/test_health.py::test_ready
EOF
}


# -------------------------------------------------------------------
# Subsystem selection
# -------------------------------------------------------------------

select_subsystem() {
    local name="$1"

    if [[ -n "${SUBSYSTEM}" ]]; then
        failure \
            "Only one subsystem selector may be used."

        exit 2
    fi

    SUBSYSTEM="${name}"
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

        --api)
            select_subsystem "api"
            shift
            ;;

        --accelerators)
            select_subsystem "accelerators"
            shift
            ;;

        --training)
            select_subsystem "training"
            shift
            ;;

        --keyword)
            if (($# < 2)); then
                failure \
                    "--keyword requires an argument"

                exit 2
            fi

            KEYWORD="$2"

            shift 2
            ;;

        --marker)
            if (($# < 2)); then
                failure \
                    "--marker requires an argument"

                exit 2
            fi

            MARKER="$2"

            shift 2
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        --)
            shift

            while (($# > 0)); do
                EXTRA_ARGS+=(
                    "$1"
                )

                shift
            done

            break
            ;;

        *)
            EXTRA_ARGS+=(
                "$1"
            )

            shift
            ;;
    esac
done


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -f "${CI_TEST_SCRIPT}" ]]; then
    failure \
        "CI test runner is missing: ${CI_TEST_SCRIPT}"

    exit 2
fi


if [[ ! -x "${CI_TEST_SCRIPT}" ]]; then
    failure \
        "CI test runner is not executable: ${CI_TEST_SCRIPT}"

    printf '\nRun:\n\n' >&2

    printf '  chmod +x scripts/ci/test.sh\n\n' >&2

    exit 2
fi


if [[ "${UNIT_ONLY}" == true ]] \
    && [[ -n "${SUBSYSTEM}" ]]; then
    failure \
        "--unit cannot be combined with a subsystem selector"

    exit 2
fi


# -------------------------------------------------------------------
# Build delegated command
# -------------------------------------------------------------------

CI_ARGS=()


if [[ "${VERBOSE}" == true ]]; then
    CI_ARGS+=(
        --verbose
    )
fi


if [[ "${FAIL_FAST}" == true ]]; then
    CI_ARGS+=(
        --fail-fast
    )
fi


if [[ "${UNIT_ONLY}" == true ]]; then
    CI_ARGS+=(
        --unit
    )
fi


PYTEST_ARGS=()


# -------------------------------------------------------------------
# Subsystem path
# -------------------------------------------------------------------

if [[ -n "${SUBSYSTEM}" ]]; then
    SUBSYSTEM_PATH="tests/${SUBSYSTEM}"

    if [[ ! -d "${PROJECT_ROOT}/${SUBSYSTEM_PATH}" ]]; then
        failure \
            "Subsystem test directory does not exist: ${SUBSYSTEM_PATH}"

        exit 2
    fi

    PYTEST_ARGS+=(
        "${SUBSYSTEM_PATH}"
    )
fi


# -------------------------------------------------------------------
# Keyword expression
# -------------------------------------------------------------------

if [[ -n "${KEYWORD}" ]]; then
    PYTEST_ARGS+=(
        -k
        "${KEYWORD}"
    )
fi


# -------------------------------------------------------------------
# Marker expression
# -------------------------------------------------------------------

if [[ -n "${MARKER}" ]]; then
    PYTEST_ARGS+=(
        -m
        "${MARKER}"
    )
fi


# -------------------------------------------------------------------
# Caller-provided pytest arguments
# -------------------------------------------------------------------

if ((${#EXTRA_ARGS[@]} > 0)); then
    PYTEST_ARGS+=(
        "${EXTRA_ARGS[@]}"
    )
fi


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

printf '\n%sOSHA Developer Tests%s\n' \
    "${BOLD}" \
    "${RESET}"

info "Authoritative runner: scripts/ci/test.sh"


if [[ "${UNIT_ONLY}" == true ]]; then
    info "Selection: unit tests"

elif [[ -n "${SUBSYSTEM}" ]]; then
    info "Selection: ${SUBSYSTEM}"

else
    info "Selection: complete test suite"
fi


if [[ -n "${KEYWORD}" ]]; then
    info "Keyword: ${KEYWORD}"
fi


if [[ -n "${MARKER}" ]]; then
    info "Marker: ${MARKER}"
fi


if [[ "${FAIL_FAST}" == true ]]; then
    info "Fail fast: enabled"
fi


# -------------------------------------------------------------------
# Show delegated command
# -------------------------------------------------------------------

printf '[INFO] Command:'

printf ' %q' \
    "${CI_TEST_SCRIPT}"

for value in "${CI_ARGS[@]}"; do
    printf ' %q' \
        "${value}"
done


if ((${#PYTEST_ARGS[@]} > 0)); then
    printf ' --'

    for value in "${PYTEST_ARGS[@]}"; do
        printf ' %q' \
            "${value}"
    done
fi

printf '\n\n'


# -------------------------------------------------------------------
# Execute authoritative test runner
# -------------------------------------------------------------------

if ((${#PYTEST_ARGS[@]} > 0)); then
    exec \
        "${CI_TEST_SCRIPT}" \
        "${CI_ARGS[@]}" \
        -- \
        "${PYTEST_ARGS[@]}"

else
    exec \
        "${CI_TEST_SCRIPT}" \
        "${CI_ARGS[@]}"
fi
