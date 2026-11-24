#!/usr/bin/env bash

#
# OSHA developer environment bootstrap.
#
# Usage:
#
#   ./scripts/dev/setup.sh
#   ./scripts/dev/setup.sh --skip-verify
#   ./scripts/dev/setup.sh --recreate
#   ./scripts/dev/setup.sh --verbose
#
# Environment:
#
#   OSHA_BOOTSTRAP_PYTHON=/usr/bin/python3
#   OSHA_VENV=/custom/path/.venv
#
# This script:
#
#   validates the repository
#   selects a Python interpreter
#   creates/reuses the virtual environment
#   upgrades packaging tools
#   installs project dependencies
#   installs developer tools when necessary
#   makes project shell scripts executable
#   validates the resulting environment
#
# It does not:
#
#   install system packages with sudo
#   modify GPU drivers
#   install CUDA or ROCm
#   modify system Python
#   delete an existing venv unless --recreate is used
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

VENV_DIR="${OSHA_VENV:-${PROJECT_ROOT}/.venv}"


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

RECREATE=false
SKIP_VERIFY=false
VERBOSE=false

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


section() {
    printf '\n%s%s== %s ==%s\n' \
        "${BOLD}" \
        "${BLUE}" \
        "$*" \
        "${RESET}"
}


# -------------------------------------------------------------------
# Command helper
# -------------------------------------------------------------------

run() {
    if [[ "${VERBOSE}" == true ]]; then
        printf '%s[CMD]%s ' \
            "${BLUE}" \
            "${RESET}"

        printf '%q ' "$@"
        printf '\n'
    fi

    "$@"
}


# -------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  scripts/dev/setup.sh [options]

Options:
  --recreate     Delete and recreate .venv.
  --skip-verify  Skip final compile/import checks.
  --verbose      Print commands before execution.
  -h, --help     Show this help.

Environment:
  OSHA_BOOTSTRAP_PYTHON
      Python interpreter used to create the virtual environment.

  OSHA_VENV
      Override the virtual-environment location.

Examples:
  ./scripts/dev/setup.sh
  ./scripts/dev/setup.sh --recreate
  ./scripts/dev/setup.sh --skip-verify
  OSHA_BOOTSTRAP_PYTHON=/usr/bin/python3 ./scripts/dev/setup.sh
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --recreate)
            RECREATE=true
            ;;

        --skip-verify)
            SKIP_VERIFY=true
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
# Repository validation
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"

section "Repository"

if [[ ! -d "${PROJECT_ROOT}/src" ]]; then
    failure "src/ does not exist."
    exit 2
fi

success "src/ found"


if [[ ! -d "${PROJECT_ROOT}/scripts" ]]; then
    failure "scripts/ does not exist."
    exit 2
fi

success "scripts/ found"


# -------------------------------------------------------------------
# Bootstrap Python
# -------------------------------------------------------------------

section "Python"

if [[ -n "${OSHA_BOOTSTRAP_PYTHON:-}" ]]; then
    BOOTSTRAP_PYTHON="${OSHA_BOOTSTRAP_PYTHON}"

elif command -v python3 >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="$(
        command -v python3
    )"

else
    failure "python3 is not installed or is not on PATH."
    exit 2
fi


if [[ ! -x "${BOOTSTRAP_PYTHON}" ]]; then
    failure \
        "Bootstrap Python is not executable: ${BOOTSTRAP_PYTHON}"

    exit 2
fi


PYTHON_VERSION="$(
    "${BOOTSTRAP_PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"

info "Bootstrap Python: ${BOOTSTRAP_PYTHON}"
info "Python version: ${PYTHON_VERSION}"


# -------------------------------------------------------------------
# Check venv support
# -------------------------------------------------------------------

if ! "${BOOTSTRAP_PYTHON}" -c \
    'import venv' \
    >/dev/null 2>&1
then
    failure \
        "The selected Python installation does not provide the venv module."

    printf '\nInstall Python virtual-environment support for your OS, then rerun setup.\n' >&2

    exit 2
fi

success "Python venv support"


# -------------------------------------------------------------------
# Virtual environment
# -------------------------------------------------------------------

section "Virtual environment"

if [[ "${RECREATE}" == true ]] && [[ -e "${VENV_DIR}" ]]; then
    info "Removing existing virtual environment: ${VENV_DIR}"

    run rm -rf -- "${VENV_DIR}"
fi


if [[ ! -d "${VENV_DIR}" ]]; then
    info "Creating virtual environment: ${VENV_DIR}"

    run "${BOOTSTRAP_PYTHON}" \
        -m venv \
        "${VENV_DIR}"

    success "Virtual environment created"

else
    info "Using existing virtual environment: ${VENV_DIR}"
fi


VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"


if [[ ! -x "${VENV_PYTHON}" ]]; then
    failure \
        "Virtual-environment Python not found: ${VENV_PYTHON}"

    exit 2
fi


VENV_VERSION="$(
    "${VENV_PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"

info "Venv Python version: ${VENV_VERSION}"

success "Virtual environment ready"


# -------------------------------------------------------------------
# Packaging tools
# -------------------------------------------------------------------

section "Packaging tools"

info "Upgrading pip, setuptools, and wheel"

run "${VENV_PYTHON}" \
    -m pip install \
    --upgrade \
    pip \
    setuptools \
    wheel

success "Packaging tools ready"


# -------------------------------------------------------------------
# Dependency installation
# -------------------------------------------------------------------

section "Project dependencies"

DEPENDENCIES_INSTALLED=false


# Prefer pyproject-based installation when the project declares one.
if [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    info "Found pyproject.toml"

    # First try an editable install with the conventional dev extra.
    if "${VENV_PYTHON}" -m pip install \
        -e '.[dev]'
    then
        success "Installed editable project with dev dependencies"

        DEPENDENCIES_INSTALLED=true

    else
        warn \
            "The project does not expose a usable [dev] extra; trying editable base install."

        run "${VENV_PYTHON}" \
            -m pip install \
            -e .

        success "Installed editable project"

        DEPENDENCIES_INSTALLED=true
    fi

elif [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
    info "Found requirements.txt"

    run "${VENV_PYTHON}" \
        -m pip install \
        -r requirements.txt

    success "Installed requirements.txt"

    DEPENDENCIES_INSTALLED=true

else
    warn \
        "No pyproject.toml or requirements.txt found; project dependency installation skipped."
fi


# -------------------------------------------------------------------
# Development requirement files
# -------------------------------------------------------------------

for requirement_file in \
    requirements-dev.txt \
    requirements/dev.txt \
    requirements/test.txt
do
    if [[ -f "${PROJECT_ROOT}/${requirement_file}" ]]; then
        info "Installing ${requirement_file}"

        run "${VENV_PYTHON}" \
            -m pip install \
            -r "${requirement_file}"

        success "Installed ${requirement_file}"

        DEPENDENCIES_INSTALLED=true
    fi
done


# -------------------------------------------------------------------
# Developer tooling
# -------------------------------------------------------------------

section "Developer tooling"

DEV_TOOLS=(
    pytest
    ruff
    mypy
)


for module in "${DEV_TOOLS[@]}"; do
    if "${VENV_PYTHON}" -c \
        "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('${module}') else 1)" \
        >/dev/null 2>&1
    then
        success "${module} already installed"

    else
        info "Installing ${module}"

        if run "${VENV_PYTHON}" \
            -m pip install \
            "${module}"
        then
            success "${module} installed"
        else
            failure "Unable to install ${module}"
            exit 1
        fi
    fi
done


# -------------------------------------------------------------------
# Runtime package checks
# -------------------------------------------------------------------

section "Core package checks"

CORE_PACKAGES=(
    fastapi
    pydantic
    torch
)


for module in "${CORE_PACKAGES[@]}"; do
    if "${VENV_PYTHON}" -c \
        "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('${module}') else 1)" \
        >/dev/null 2>&1
    then
        version="$(
            "${VENV_PYTHON}" -c \
                "import ${module}; print(getattr(${module}, '__version__', 'installed'))" \
                2>/dev/null \
            || printf 'installed'
        )"

        success "${module}: ${version}"

    else
        failure \
            "Required project package is missing: ${module}"

        printf '\nThe project dependency definition should install this package.\n' >&2

        exit 1
    fi
done


# -------------------------------------------------------------------
# Script permissions
# -------------------------------------------------------------------

section "Script permissions"

while IFS= read -r -d '' script; do
    chmod +x "${script}"

done < <(
    find "${PROJECT_ROOT}/scripts" \
        -type f \
        -name '*.sh' \
        -print0
)

success "Shell scripts marked executable"


# -------------------------------------------------------------------
# Bash syntax check
# -------------------------------------------------------------------

section "Shell syntax"

SHELL_COUNT=0


while IFS= read -r -d '' script; do
    SHELL_COUNT=$((SHELL_COUNT + 1))

    if ! bash -n "${script}"; then
        failure \
            "Invalid shell syntax: ${script#${PROJECT_ROOT}/}"

        exit 1
    fi

done < <(
    find "${PROJECT_ROOT}/scripts" \
        -type f \
        -name '*.sh' \
        -print0
)

success "Shell syntax (${SHELL_COUNT} scripts)"


# -------------------------------------------------------------------
# Verification
# -------------------------------------------------------------------

if [[ "${SKIP_VERIFY}" == false ]]; then
    section "Repository verification"

    COMPILE_SCRIPT="${PROJECT_ROOT}/scripts/ci/compile.sh"
    IMPORT_SCRIPT="${PROJECT_ROOT}/scripts/ci/check_imports.sh"

    if [[ ! -x "${COMPILE_SCRIPT}" ]]; then
        failure \
            "Missing executable compile check: ${COMPILE_SCRIPT}"

        exit 1
    fi

    if [[ ! -x "${IMPORT_SCRIPT}" ]]; then
        failure \
            "Missing executable import check: ${IMPORT_SCRIPT}"

        exit 1
    fi

    info "Running Python compilation"

    OSHA_PYTHON="${VENV_PYTHON}" \
        "${COMPILE_SCRIPT}"

    success "Compilation verification"

    info "Running import validation"

    OSHA_PYTHON="${VENV_PYTHON}" \
        "${IMPORT_SCRIPT}"

    success "Import verification"

else
    section "Repository verification"

    warn "Verification skipped (--skip-verify)"
fi


# -------------------------------------------------------------------
# Dependency snapshot
# -------------------------------------------------------------------

section "Environment snapshot"

info "Installed package count"

PACKAGE_COUNT="$(
    "${VENV_PYTHON}" -m pip list \
        --format=json \
    | "${VENV_PYTHON}" -c \
        'import json, sys; print(len(json.load(sys.stdin)))'
)"

printf 'Packages: %s\n' \
    "${PACKAGE_COUNT}"


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Developer Setup Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Project:       %s\n' \
    "${PROJECT_ROOT}"

printf 'Virtual env:   %s\n' \
    "${VENV_DIR}"

printf 'Python:        %s\n' \
    "${VENV_VERSION}"

printf 'Packages:      %s\n' \
    "${PACKAGE_COUNT}"

printf 'Dependencies:  %s\n' \
    "$(
        if [[ "${DEPENDENCIES_INSTALLED}" == true ]]; then
            printf 'installed'
        else
            printf 'existing/manual'
        fi
    )"

printf 'Verification:  %s\n' \
    "$(
        if [[ "${SKIP_VERIFY}" == true ]]; then
            printf 'skipped'
        else
            printf 'passed'
        fi
    )"

printf 'Duration:      %ss\n' \
    "${elapsed}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


printf '\n%sDEVELOPER ENVIRONMENT READY%s\n' \
    "${GREEN}" \
    "${RESET}"

printf '\nActivate with:\n\n'
printf '  source %q\n\n' \
    "${VENV_DIR}/bin/activate"

printf 'Then validate with:\n\n'
printf '  ./scripts/dev/check.sh --quick\n\n'

exit 0
