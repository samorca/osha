#!/usr/bin/env bash

#
# OSHA complete CI verification.
#
# Pipeline:
#
#   shell syntax
#       ↓
#   Python compilation
#       ↓
#   Python imports
#       ↓
#   tests
#
# Usage:
#
#   ./scripts/ci/verify.sh
#   ./scripts/ci/verify.sh --quick
#   ./scripts/ci/verify.sh --verbose
#   ./scripts/ci/verify.sh --no-tests
#   ./scripts/ci/verify.sh --all-imports
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


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

QUICK=false
VERBOSE=false
RUN_TESTS=true
ALL_IMPORTS=false

CHECK_COUNT=0

START_SECONDS="${SECONDS}"


# -------------------------------------------------------------------
# Colours
# -------------------------------------------------------------------

if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    RESET=$'\033[0m'
    GREEN=$'\033[32m'
    RED=$'\033[31m'
    BLUE=$'\033[34m'
    BOLD=$'\033[1m'
else
    RESET=""
    GREEN=""
    RED=""
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


failure() {
    printf '%s[FAIL]%s %s\n' \
        "${RED}" \
        "${RESET}" \
        "$*" >&2
}


section() {
    printf '\n%s%s== %s ==%s\n' \
        "${BOLD}" \
        "${BLUE}" \
        "$*" \
        "${RESET}"
}


# -------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  scripts/ci/verify.sh [options]

Options:
  --quick        Compile and validate imports only.
  --verbose      Enable verbose child checks.
  --no-tests     Skip pytest.
  --all-imports  Import every module under src/.
  -h, --help     Show this help.

Examples:
  ./scripts/ci/verify.sh
  ./scripts/ci/verify.sh --quick
  ./scripts/ci/verify.sh --all-imports
  ./scripts/ci/verify.sh --no-tests
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --quick)
            QUICK=true
            RUN_TESTS=false
            ;;

        --verbose)
            VERBOSE=true
            ;;

        --no-tests)
            RUN_TESTS=false
            ;;

        --all-imports)
            ALL_IMPORTS=true
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac

    shift
done


# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


# -------------------------------------------------------------------
# Required scripts
# -------------------------------------------------------------------

COMPILE_SCRIPT="${SCRIPT_DIR}/compile.sh"

IMPORT_SCRIPT="${SCRIPT_DIR}/check_imports.sh"

TEST_SCRIPT="${SCRIPT_DIR}/test.sh"


for script in \
    "${COMPILE_SCRIPT}" \
    "${IMPORT_SCRIPT}"
do
    if [[ ! -f "${script}" ]]; then
        failure \
            "Required CI script is missing: ${script}"

        exit 2
    fi

    if [[ ! -x "${script}" ]]; then
        failure \
            "Required CI script is not executable: ${script}"

        exit 2
    fi
done


if [[ "${RUN_TESTS}" == true ]] \
    && [[ ! -x "${TEST_SCRIPT}" ]]; then
    failure \
        "Test script is missing or not executable: ${TEST_SCRIPT}"

    exit 2
fi


# -------------------------------------------------------------------
# Child arguments
# -------------------------------------------------------------------

COMPILE_ARGS=()

IMPORT_ARGS=()

TEST_ARGS=()


if [[ "${VERBOSE}" == true ]]; then
    COMPILE_ARGS+=(
        --verbose
    )

    IMPORT_ARGS+=(
        --verbose
    )

    TEST_ARGS+=(
        --verbose
    )
fi


if [[ "${ALL_IMPORTS}" == true ]]; then
    IMPORT_ARGS+=(
        --all
    )
fi


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

printf '\n%sOSHA CI Verification%s\n' \
    "${BOLD}" \
    "${RESET}"

info "Project root: ${PROJECT_ROOT}"

if [[ "${QUICK}" == true ]]; then
    info "Mode: quick"
else
    info "Mode: full"
fi


# -------------------------------------------------------------------
# 1. Shell syntax
# -------------------------------------------------------------------

section "Shell syntax"

SHELL_FILE_COUNT=0


while IFS= read -r -d '' script; do
    SHELL_FILE_COUNT=$((SHELL_FILE_COUNT + 1))

    if ! bash -n "${script}"; then
        failure \
            "Shell syntax failed: ${script#${PROJECT_ROOT}/}"

        exit 1
    fi

done < <(
    find scripts \
        -type f \
        -name '*.sh' \
        -print0
)


success \
    "Shell syntax (${SHELL_FILE_COUNT} scripts)"

CHECK_COUNT=$((CHECK_COUNT + 1))


# -------------------------------------------------------------------
# 2. Compilation
# -------------------------------------------------------------------

section "Python compilation"

if "${COMPILE_SCRIPT}" \
    "${COMPILE_ARGS[@]}"
then
    success "Compilation check"
else
    failure "Compilation check"
    exit 1
fi


CHECK_COUNT=$((CHECK_COUNT + 1))


# -------------------------------------------------------------------
# 3. Imports
# -------------------------------------------------------------------

section "Import validation"

if "${IMPORT_SCRIPT}" \
    "${IMPORT_ARGS[@]}"
then
    success "Import validation"
else
    failure "Import validation"
    exit 1
fi


CHECK_COUNT=$((CHECK_COUNT + 1))


# -------------------------------------------------------------------
# 4. Tests
# -------------------------------------------------------------------

if [[ "${RUN_TESTS}" == true ]]; then
    section "Tests"

    if "${TEST_SCRIPT}" \
        "${TEST_ARGS[@]}"
    then
        success "Test suite"
    else
        exit_code=$?

        failure \
            "Test suite returned exit code ${exit_code}"

        exit "${exit_code}"
    fi

    CHECK_COUNT=$((CHECK_COUNT + 1))

else
    section "Tests"
    info "Test suite skipped"
fi


# -------------------------------------------------------------------
# Final summary
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA CI Verification Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Checks:   %s\n' \
    "${CHECK_COUNT}"

printf 'Tests:    %s\n' \
    "$(
        if [[ "${RUN_TESTS}" == true ]]; then
            printf 'enabled'
        else
            printf 'skipped'
        fi
    )"

printf 'Imports:  %s\n' \
    "$(
        if [[ "${ALL_IMPORTS}" == true ]]; then
            printf 'all'
        else
            printf 'boundaries'
        fi
    )"

printf 'Duration: %ss\n' \
    "${elapsed}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


printf '\n%sCI VERIFICATION PASSED%s\n' \
    "${GREEN}" \
    "${RESET}"

exit 0
