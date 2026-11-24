#!/usr/bin/env bash

# ==============================================================================
# OSHA - Project Bootstrap
# ==============================================================================
#
# Prepares a local OSHA environment.
#
# Usage:
#   ./scripts/bootstrap.sh
#
# Options:
#   --skip-install     Do not install Python dependencies
#   --recreate-venv   Remove and recreate .venv
#   --help            Show help
#
# This script is intended to be safe to run multiple times.
# ==============================================================================

set -Eeuo pipefail

IFS=$'\n\t'


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${PROJECT_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
PYPROJECT_FILE="${PROJECT_ROOT}/pyproject.toml"

SKIP_INSTALL=false
RECREATE_VENV=false


# ------------------------------------------------------------------------------
# Colors
# ------------------------------------------------------------------------------

if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    BOLD=''
    RESET=''
fi


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

info() {
    printf "%b[INFO]%b %s\n" "${BLUE}" "${RESET}" "$*"
}

success() {
    printf "%b[OK]%b %s\n" "${GREEN}" "${RESET}" "$*"
}

warning() {
    printf "%b[WARN]%b %s\n" "${YELLOW}" "${RESET}" "$*" >&2
}

error() {
    printf "%b[ERROR]%b %s\n" "${RED}" "${RESET}" "$*" >&2
}

die() {
    error "$*"
    exit 1
}


# ------------------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------------------

on_error() {
    local exit_code=$?
    local line_number=${BASH_LINENO[0]:-unknown}

    error "Bootstrap failed at line ${line_number}."
    error "Exit code: ${exit_code}"

    exit "${exit_code}"
}

trap on_error ERR


# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------

show_help() {
    cat <<'EOF'
OSHA project bootstrap

Usage:
    ./scripts/bootstrap.sh [options]

Options:
    --skip-install
        Skip Python dependency installation.

    --recreate-venv
        Delete the existing .venv and create it again.

    --help
        Show this help message.

Examples:
    ./scripts/bootstrap.sh

    ./scripts/bootstrap.sh --skip-install

    ./scripts/bootstrap.sh --recreate-venv
EOF
}


# ------------------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------------------

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-install)
                SKIP_INSTALL=true
                shift
                ;;

            --recreate-venv)
                RECREATE_VENV=true
                shift
                ;;

            --help|-h)
                show_help
                exit 0
                ;;

            *)
                die "Unknown argument: $1"
                ;;
        esac
    done
}


# ------------------------------------------------------------------------------
# Command helpers
# ------------------------------------------------------------------------------

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

require_command() {
    local command_name="$1"

    if ! command_exists "${command_name}"; then
        die "Required command not found: ${command_name}"
    fi
}


# ------------------------------------------------------------------------------
# Project validation
# ------------------------------------------------------------------------------

validate_project_root() {
    info "Validating project root..."

    [[ -d "${PROJECT_ROOT}/src" ]] ||
        die "Missing src directory: ${PROJECT_ROOT}/src"

    [[ -d "${PROJECT_ROOT}/config" ]] ||
        die "Missing config directory: ${PROJECT_ROOT}/config"

    if [[ ! -f "${PYPROJECT_FILE}" && ! -f "${REQUIREMENTS_FILE}" ]]; then
        die "Neither pyproject.toml nor requirements.txt exists."
    fi

    success "Project root validated: ${PROJECT_ROOT}"
}


# ------------------------------------------------------------------------------
# Python discovery
# ------------------------------------------------------------------------------

find_python() {
    if command_exists python3; then
        SYSTEM_PYTHON="$(command -v python3)"
    elif command_exists python; then
        SYSTEM_PYTHON="$(command -v python)"
    else
        die "Python was not found."
    fi

    local version

    version="$("${SYSTEM_PYTHON}" --version 2>&1)"

    success "Found ${version}"
}


# ------------------------------------------------------------------------------
# Virtual environment
# ------------------------------------------------------------------------------

create_virtual_environment() {
    if [[ "${RECREATE_VENV}" == true && -d "${VENV_DIR}" ]]; then
        warning "Removing existing virtual environment."
        rm -rf -- "${VENV_DIR}"
    fi

    if [[ ! -d "${VENV_DIR}" ]]; then
        info "Creating virtual environment at ${VENV_DIR}..."

        "${SYSTEM_PYTHON}" -m venv "${VENV_DIR}"

        success "Virtual environment created."
    else
        success "Virtual environment already exists."
    fi

    [[ -x "${VENV_PYTHON}" ]] ||
        die "Virtual environment Python is unavailable."

    [[ -f "${VENV_DIR}/bin/activate" ]] ||
        die "Virtual environment activation script is unavailable."
}


activate_virtual_environment() {
    info "Selecting virtual environment..."

    export VIRTUAL_ENV="${VENV_DIR}"
    export PATH="${VENV_DIR}/bin:${PATH}"
    hash -r

    success "Virtual environment activated."

    info "Python: ${VENV_PYTHON}"
    info "Python version: $("${VENV_PYTHON}" --version 2>&1)"
}


# ------------------------------------------------------------------------------
# Runtime directories
# ------------------------------------------------------------------------------

create_runtime_directories() {
    info "Creating runtime directories..."

    local directories=(
        "${PROJECT_ROOT}/artifacts"
        "${PROJECT_ROOT}/artifacts/runtime"
        "${PROJECT_ROOT}/artifacts/incidents"
        "${PROJECT_ROOT}/artifacts/decisions"
        "${PROJECT_ROOT}/artifacts/remediation"
        "${PROJECT_ROOT}/artifacts/evaluations"
        "${PROJECT_ROOT}/artifacts/reports"
        "${PROJECT_ROOT}/artifacts/correlation"
        "${PROJECT_ROOT}/artifacts/correlation/quarantine"
        "${PROJECT_ROOT}/artifacts/ai"
        "${PROJECT_ROOT}/artifacts/ai/memory"
        "${PROJECT_ROOT}/artifacts/ai/results"
        "${PROJECT_ROOT}/artifacts/infrastructure"
        "${PROJECT_ROOT}/artifacts/observability"
        "${PROJECT_ROOT}/checkpoints"
        "${PROJECT_ROOT}/logs"
        "${PROJECT_ROOT}/logs/audit"
        "${PROJECT_ROOT}/models"
    )

    local directory

    for directory in "${directories[@]}"; do
        mkdir -p -- "${directory}"
    done

    success "Runtime directories ready."
}


# ------------------------------------------------------------------------------
# Python tooling
# ------------------------------------------------------------------------------

prepare_python_tooling() {
    info "Checking pip..."

    "${VENV_PYTHON}" -m pip --version >/dev/null 2>&1 ||
        die "pip is unavailable inside the virtual environment."

    success "pip is available."
}


# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

install_dependencies() {
    if [[ "${SKIP_INSTALL}" == true ]]; then
        warning "Dependency installation skipped."
        return
    fi

    if [[ -f "${REQUIREMENTS_FILE}" ]]; then
        info "Installing requirements.txt dependencies..."

        "${VENV_PYTHON}" -m pip install -r "${REQUIREMENTS_FILE}"

        success "requirements.txt dependencies installed."
    else
        warning "requirements.txt does not exist."
    fi

    # Install the project itself only when pyproject.toml declares
    # a Python package that pip can install.
    if [[ -f "${PYPROJECT_FILE}" ]]; then
        info "Checking project installation metadata..."

        if "${VENV_PYTHON}" -m pip install --dry-run "${PROJECT_ROOT}" \
            >/dev/null 2>&1; then

            info "Installing OSHA project..."

            "${VENV_PYTHON}" -m pip install -e "${PROJECT_ROOT}"

            success "OSHA project installed in editable mode."
        else
            warning "Project is not currently installable through pyproject.toml."
            warning "Skipping editable installation."
        fi
    fi
}


# ------------------------------------------------------------------------------
# Configuration validation
# ------------------------------------------------------------------------------

validate_yaml_syntax() {
    info "Checking configuration files..."

    if ! "${VENV_PYTHON}" - <<PY
from pathlib import Path

config_dir = Path(r"${PROJECT_ROOT}") / "config"

files = sorted(config_dir.glob("*.yaml"))

if not files:
    raise SystemExit("No YAML configuration files found.")

try:
    import yaml
except ImportError:
    print("PyYAML is not installed; skipping YAML syntax validation.")
    raise SystemExit(0)

errors = []

for path in files:
    try:
        with path.open("r", encoding="utf-8") as handle:
            yaml.safe_load(handle)
    except Exception as exc:
        errors.append(f"{path.name}: {exc}")

if errors:
    print("Invalid configuration:")
    for error in errors:
        print(f"  - {error}")
    raise SystemExit(1)

print(f"Validated {len(files)} YAML configuration file(s).")
PY
    then
        die "Configuration validation failed."
    fi

    success "Configuration check completed."
}


# ------------------------------------------------------------------------------
# Source compilation
# ------------------------------------------------------------------------------

validate_python_source() {
    info "Checking Python source syntax..."

    "${VENV_PYTHON}" -m compileall -q "${PROJECT_ROOT}/src"

    success "Python source syntax check passed."
}


# ------------------------------------------------------------------------------
# Environment summary
# ------------------------------------------------------------------------------

print_summary() {
    printf "\n"
    printf "%b%s%b\n" "${BOLD}" "OSHA bootstrap complete" "${RESET}"
    printf "\n"

    printf "Project root : %s\n" "${PROJECT_ROOT}"
    printf "Virtual env  : %s\n" "${VENV_DIR}"
    printf "Python       : %s\n" "$("${VENV_PYTHON}" --version 2>&1)"
    printf "Interpreter  : %s\n" "${VENV_PYTHON}"
    printf "Environment  : production\n"

    printf "\n"

    printf "Next commands:\n"
    printf "  .venv/bin/python scripts/validate_config.py\n"

    printf "\n"
}


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

main() {
    parse_arguments "$@"

    printf "\n"
    printf "%b%s%b\n" "${BOLD}" "OSHA Project Bootstrap" "${RESET}"
    printf "\n"

    cd "${PROJECT_ROOT}"

    validate_project_root

    require_command uname

    find_python

    create_virtual_environment
    activate_virtual_environment

    create_runtime_directories

    prepare_python_tooling
    install_dependencies

    validate_yaml_syntax
    validate_python_source

    print_summary
}


main "$@"