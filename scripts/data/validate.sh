#!/usr/bin/env bash

#
# OSHA data validation entrypoint.
#
# Authoritative implementation:
#
#     src/data/validate.py
#
# Usage:
#
#   ./scripts/data/validate.sh --help
#   ./scripts/data/validate.sh <validate.py arguments...>
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This shell wrapper does NOT implement:
#
#   schema validation
#   record validation
#   checksum validation
#   manifest validation
#   duplicate detection
#   split validation
#   corpus integrity rules
#
# Those responsibilities remain in src/data/.
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

VALIDATE_MODULE="src.data.validate"

VALIDATE_FILE="${PROJECT_ROOT}/src/data/validate.py"


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
# Project validation
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -d "${PROJECT_ROOT}/src/data" ]]; then
    failure "src/data directory does not exist."
    exit 2
fi


if [[ ! -f "${VALIDATE_FILE}" ]]; then
    failure \
        "Data validation implementation does not exist: ${VALIDATE_FILE}"

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


export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


# -------------------------------------------------------------------
# Source validation
# -------------------------------------------------------------------

if ! "${PYTHON}" -m py_compile \
    "${VALIDATE_FILE}"
then
    failure \
        "src/data/validate.py failed Python compilation."

    exit 1
fi


# -------------------------------------------------------------------
# Import validation
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import src.data.validate' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.data.validate."

    printf '\nRun for the complete traceback:\n\n' >&2

    printf '  %q -c %q\n\n' \
        "${PYTHON}" \
        'import src.data.validate'

    exit 1
fi


# -------------------------------------------------------------------
# Determine whether a CLI entrypoint exists
# -------------------------------------------------------------------

HAS_MAIN="$(
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import src.data.validate as module


main = getattr(
    module,
    "main",
    None,
)


print(
    "1"
    if callable(main)
    else "0"
)
PY
)"


# -------------------------------------------------------------------
# Wrapper information
# -------------------------------------------------------------------

if (($# == 1)) \
    && [[ "$1" == "--wrapper-info" ]]; then
    printf '\n%sOSHA Data Validation Wrapper%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Project root: %s\n' \
        "${PROJECT_ROOT}"

    printf 'Python:       %s\n' \
        "${PYTHON}"

    printf 'Version:      %s\n' \
        "${PYTHON_VERSION}"

    printf 'Module:       %s\n' \
        "${VALIDATE_MODULE}"

    printf 'Source:       %s\n' \
        "${VALIDATE_FILE}"

    printf 'CLI main():   %s\n' \
        "$(
            if [[ "${HAS_MAIN}" == "1" ]]; then
                printf 'yes'
            else
                printf 'no'
            fi
        )"

    exit 0
fi


# -------------------------------------------------------------------
# Require the implementation to own the CLI
# -------------------------------------------------------------------

if [[ "${HAS_MAIN}" != "1" ]]; then
    failure \
        "src.data.validate does not expose a callable main()."

    cat >&2 <<'EOF'

scripts/data/validate.sh will not reimplement the
Python validation subsystem in Bash.

The appropriate next step is to expose a CLI from:

    src/data/validate.py

For example, the module should own:

    def main() -> None:
        ...

    if __name__ == "__main__":
        main()

The existing validation functions should remain the
source of truth. main() should only parse arguments,
call those functions, format the validation result,
and choose the process exit status.

After that, this wrapper delegates with:

    python -m src.data.validate

EOF

    exit 2
fi


# -------------------------------------------------------------------
# Operator information
# -------------------------------------------------------------------

if [[ -t 1 ]]; then
    info \
        "Delegating validation to ${VALIDATE_MODULE}"
fi


# -------------------------------------------------------------------
# Execute validation
# -------------------------------------------------------------------
#
# exec is intentional:
#
#   signals reach Python directly
#   Python controls exit status
#   command arguments are preserved exactly
#   there is only one validation implementation
#

exec \
    "${PYTHON}" \
    -m "${VALIDATE_MODULE}" \
    "$@"
