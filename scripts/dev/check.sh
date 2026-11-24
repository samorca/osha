#!/usr/bin/env bash

#
# OSHA developer validation entrypoint.
#
# Usage:
#
#   ./scripts/dev/check.sh
#
# Optional:
#
#   ./scripts/dev/check.sh --quick
#   ./scripts/dev/check.sh --no-tests
#   ./scripts/dev/check.sh --verbose
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# Purpose:
#
#   Validate the repository before committing,
#   pushing, opening a merge request, or running
#   larger training/runtime workflows.
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

SCRIPTS_DIR="${PROJECT_ROOT}/scripts"

VENV_DIR="${PROJECT_ROOT}/.venv"


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

QUICK=false
RUN_TESTS=true
VERBOSE=false

FAILURES=0
WARNINGS=0
CHECKS=0

START_SECONDS="${SECONDS}"


# -------------------------------------------------------------------
# Colours
# -------------------------------------------------------------------

if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    COLOR_RESET=$'\033[0m'
    COLOR_BLUE=$'\033[34m'
    COLOR_GREEN=$'\033[32m'
    COLOR_YELLOW=$'\033[33m'
    COLOR_RED=$'\033[31m'
    COLOR_BOLD=$'\033[1m'
else
    COLOR_RESET=""
    COLOR_BLUE=""
    COLOR_GREEN=""
    COLOR_YELLOW=""
    COLOR_RED=""
    COLOR_BOLD=""
fi


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

info() {
    printf '%s[INFO]%s %s\n' \
        "${COLOR_BLUE}" \
        "${COLOR_RESET}" \
        "$*"
}


success() {
    printf '%s[PASS]%s %s\n' \
        "${COLOR_GREEN}" \
        "${COLOR_RESET}" \
        "$*"
}


warn() {
    WARNINGS=$((WARNINGS + 1))

    printf '%s[WARN]%s %s\n' \
        "${COLOR_YELLOW}" \
        "${COLOR_RESET}" \
        "$*" >&2
}


failure() {
    FAILURES=$((FAILURES + 1))

    printf '%s[FAIL]%s %s\n' \
        "${COLOR_RED}" \
        "${COLOR_RESET}" \
        "$*" >&2
}


section() {
    printf '\n%s%s== %s ==%s\n' \
        "${COLOR_BOLD}" \
        "${COLOR_BLUE}" \
        "$*" \
        "${COLOR_RESET}"
}


# -------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  scripts/dev/check.sh [options]

Options:
  --quick       Run only essential fast checks.
  --no-tests    Skip the test suite.
  --verbose     Show commands before execution.
  -h, --help    Show this help.

Environment:
  OSHA_PYTHON   Override the Python interpreter.

Examples:
  ./scripts/dev/check.sh
  ./scripts/dev/check.sh --quick
  ./scripts/dev/check.sh --no-tests
  OSHA_PYTHON=/usr/bin/python3 ./scripts/dev/check.sh
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --quick)
            QUICK=true
            ;;

        --no-tests)
            RUN_TESTS=false
            ;;

        --verbose)
            VERBOSE=true
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
# Failure trap
# -------------------------------------------------------------------

on_error() {
    local exit_code=$?
    local line_number="${1:-unknown}"

    printf '\n%s[ERROR]%s Unexpected failure at line %s (exit %s).\n' \
        "${COLOR_RED}" \
        "${COLOR_RESET}" \
        "${line_number}" \
        "${exit_code}" >&2

    exit "${exit_code}"
}

trap 'on_error "${LINENO}"' ERR


# -------------------------------------------------------------------
# Project validation
# -------------------------------------------------------------------

if [[ ! -d "${PROJECT_ROOT}/src" ]]; then
    printf 'ERROR: src/ not found under %s\n' \
        "${PROJECT_ROOT}" >&2

    exit 2
fi

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
    printf 'ERROR: No Python interpreter found.\n' >&2
    exit 2
fi


if [[ ! -x "${PYTHON}" ]]; then
    printf 'ERROR: Python is not executable: %s\n' \
        "${PYTHON}" >&2

    exit 2
fi


# -------------------------------------------------------------------
# Command execution helpers
# -------------------------------------------------------------------

run_required() {
    local name="$1"
    shift

    CHECKS=$((CHECKS + 1))

    info "${name}"

    if [[ "${VERBOSE}" == true ]]; then
        printf '       '
        printf '%q ' "$@"
        printf '\n'
    fi

    if "$@"; then
        success "${name}"
        return 0
    fi

    failure "${name}"

    return 0
}


run_script_if_present() {
    local name="$1"
    local script="$2"
    shift 2

    if [[ ! -f "${script}" ]]; then
        warn "${name}: script not found: ${script}"
        return 0
    fi

    if [[ ! -x "${script}" ]]; then
        warn "${name}: script is not executable: ${script}"

        run_required \
            "${name}" \
            bash "${script}" "$@"

        return 0
    fi

    run_required \
        "${name}" \
        "${script}" "$@"
}


run_optional_tool() {
    local name="$1"
    local command_name="$2"
    shift 2

    if ! command -v "${command_name}" >/dev/null 2>&1; then
        warn "${name}: ${command_name} is not installed"
        return 0
    fi

    run_required \
        "${name}" \
        "${command_name}" "$@"
}


python_module_available() {
    local module="$1"

    "${PYTHON}" -c \
        "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('${module}') else 1)" \
        >/dev/null 2>&1
}


run_python_module_if_available() {
    local name="$1"
    local module="$2"
    shift 2

    if ! python_module_available "${module}"; then
        warn "${name}: Python module '${module}' is not installed"
        return 0
    fi

    run_required \
        "${name}" \
        "${PYTHON}" -m "${module}" "$@"
}


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

printf '\n%sOSHA Developer Check%s\n' \
    "${COLOR_BOLD}" \
    "${COLOR_RESET}"

printf 'Root:   %s\n' "${PROJECT_ROOT}"
printf 'Python: %s\n' "${PYTHON}"
printf 'Mode:   %s\n' "$(
    if [[ "${QUICK}" == true ]]; then
        printf 'quick'
    else
        printf 'full'
    fi
)"

"${PYTHON}" --version


# -------------------------------------------------------------------
# 1. Repository structure
# -------------------------------------------------------------------

section "Repository"

run_required \
    "src directory exists" \
    test -d "${PROJECT_ROOT}/src"

run_required \
    "scripts directory exists" \
    test -d "${PROJECT_ROOT}/scripts"

if [[ -d "${PROJECT_ROOT}/tests" ]]; then
    success "tests directory exists"
else
    warn "tests/ directory does not currently exist"
fi


# -------------------------------------------------------------------
# 2. Shell syntax
# -------------------------------------------------------------------

section "Shell syntax"

shell_files_found=0

while IFS= read -r -d '' script; do
    shell_files_found=$((shell_files_found + 1))

    run_required \
        "bash syntax: ${script#${PROJECT_ROOT}/}" \
        bash -n "${script}"

done < <(
    find scripts \
        -type f \
        -name '*.sh' \
        -print0
)

if ((shell_files_found == 0)); then
    warn "No shell scripts found"
fi


# -------------------------------------------------------------------
# 3. Python bytecode compilation
# -------------------------------------------------------------------

section "Python compilation"

if [[ -f "${SCRIPTS_DIR}/ci/compile.sh" ]]; then
    run_script_if_present \
        "Python compilation" \
        "${SCRIPTS_DIR}/ci/compile.sh"

else
    run_required \
        "Compile src/" \
        "${PYTHON}" -m compileall \
        -q \
        -f \
        "${PROJECT_ROOT}/src"
fi


# -------------------------------------------------------------------
# 4. Import validation
# -------------------------------------------------------------------

section "Imports"

if [[ -f "${SCRIPTS_DIR}/ci/check_imports.sh" ]]; then
    run_script_if_present \
        "Import validation" \
        "${SCRIPTS_DIR}/ci/check_imports.sh"

else
    run_required \
        "Import src package" \
        "${PYTHON}" -c \
        'import src'

    if [[ -f "${PROJECT_ROOT}/src/api/app.py" ]]; then
        run_required \
            "Import API application" \
            "${PYTHON}" -c \
            'from src.api.app import app; assert app is not None'
    fi
fi


# -------------------------------------------------------------------
# Quick mode stops after essential checks
# -------------------------------------------------------------------

if [[ "${QUICK}" == true ]]; then
    section "Quick check complete"

else

    # ---------------------------------------------------------------
    # 5. Ruff lint
    # ---------------------------------------------------------------

    section "Lint"

    if python_module_available "ruff"; then
        run_required \
            "Ruff lint" \
            "${PYTHON}" -m ruff check \
            src \
            scripts
    else
        warn "Ruff is not installed; lint check skipped"
    fi


    # ---------------------------------------------------------------
    # 6. Formatting verification
    # ---------------------------------------------------------------

    section "Formatting"

    if python_module_available "ruff"; then
        run_required \
            "Ruff format check" \
            "${PYTHON}" -m ruff format \
            --check \
            src
    else
        warn "Ruff is not installed; format check skipped"
    fi


    # ---------------------------------------------------------------
    # 7. Static typing
    # ---------------------------------------------------------------

    section "Type checking"

    if [[ -x "${SCRIPTS_DIR}/dev/typecheck.sh" ]] \
        && [[ "${SCRIPTS_DIR}/dev/typecheck.sh" != "${SCRIPT_DIR}/check.sh" ]]; then

        run_script_if_present \
            "Type checking" \
            "${SCRIPTS_DIR}/dev/typecheck.sh"

    elif python_module_available "mypy"; then
        run_required \
            "Mypy" \
            "${PYTHON}" -m mypy src

    else
        warn "No configured type checker found"
    fi


    # ---------------------------------------------------------------
    # 8. Tests
    # ---------------------------------------------------------------

    if [[ "${RUN_TESTS}" == true ]]; then
        section "Tests"

        if [[ -x "${SCRIPTS_DIR}/ci/test.sh" ]]; then
            run_script_if_present \
                "Test suite" \
                "${SCRIPTS_DIR}/ci/test.sh"

        elif [[ -x "${SCRIPTS_DIR}/dev/test.sh" ]]; then
            run_script_if_present \
                "Test suite" \
                "${SCRIPTS_DIR}/dev/test.sh"

        elif python_module_available "pytest"; then
            if [[ -d "${PROJECT_ROOT}/tests" ]]; then
                run_required \
                    "Pytest" \
                    "${PYTHON}" -m pytest \
                    -q \
                    tests
            else
                warn "pytest exists but tests/ does not"
            fi

        else
            warn "No test runner is currently available"
        fi

    else
        section "Tests"
        info "Tests skipped (--no-tests)"
    fi

fi


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))

printf '\n%s========================================%s\n' \
    "${COLOR_BOLD}" \
    "${COLOR_RESET}"

printf '%sOSHA Developer Check Summary%s\n' \
    "${COLOR_BOLD}" \
    "${COLOR_RESET}"

printf 'Checks:   %s\n' "${CHECKS}"
printf 'Warnings: %s\n' "${WARNINGS}"
printf 'Failures: %s\n' "${FAILURES}"
printf 'Duration: %ss\n' "${elapsed}"

printf '%s========================================%s\n' \
    "${COLOR_BOLD}" \
    "${COLOR_RESET}"


if ((FAILURES > 0)); then
    printf '\n%sCHECK FAILED%s\n' \
        "${COLOR_RED}" \
        "${COLOR_RESET}"

    exit 1
fi


if ((WARNINGS > 0)); then
    printf '\n%sCHECK PASSED WITH WARNINGS%s\n' \
        "${COLOR_YELLOW}" \
        "${COLOR_RESET}"

else
    printf '\n%sALL CHECKS PASSED%s\n' \
        "${COLOR_GREEN}" \
        "${COLOR_RESET}"
fi

exit 0
