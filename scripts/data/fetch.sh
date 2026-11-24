#!/usr/bin/env bash

#
# OSHA data acquisition entrypoint.
#
# Authoritative implementation:
#
#     src/data/fetch.py
#
# Usage:
#
#   ./scripts/data/fetch.sh --help
#   ./scripts/data/fetch.sh <fetch.py arguments...>
#
# Examples:
#
#   ./scripts/data/fetch.sh --help
#   ./scripts/data/fetch.sh --wrapper-info
#   ./scripts/data/fetch.sh --source-root /path/to/dataset
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This wrapper intentionally does not implement
# downloading, checksums, manifests, retries, or
# source validation itself.
#
# Those responsibilities remain under src/data/.
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

FETCH_MODULE="src.data.fetch"

FETCH_FILE="${PROJECT_ROOT}/src/data/fetch.py"


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
    failure \
        "src/data directory does not exist."

    exit 2
fi


if [[ ! -f "${FETCH_FILE}" ]]; then
    failure \
        "Data fetch implementation does not exist: ${FETCH_FILE}"

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


# -------------------------------------------------------------------
# Validate source before invoking it
# -------------------------------------------------------------------

if ! "${PYTHON}" -m py_compile \
    "${FETCH_FILE}"
then
    failure \
        "src/data/fetch.py failed Python compilation."

    exit 1
fi


if ! "${PYTHON}" -c \
    'import src.data.fetch' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.data.fetch."

    printf '\nRun this for the complete traceback:\n\n' >&2

    printf '  %q -c %q\n\n' \
        "${PYTHON}" \
        'import src.data.fetch'

    exit 1
fi


# -------------------------------------------------------------------
# Determine whether fetch.py exposes a module CLI
# -------------------------------------------------------------------

HAS_MAIN="$(
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import inspect

import src.data.fetch as module


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
# No CLI main()
# -------------------------------------------------------------------

if [[ "${HAS_MAIN}" != "1" ]]; then
    failure \
        "src.data.fetch does not expose a callable main()."

    cat >&2 <<'EOF'

scripts/data/fetch.sh is intentionally only an
operator wrapper. It will not guess how the Python
fetch subsystem should be called.

Implement a CLI entrypoint in src/data/fetch.py:

    def main() -> None:
        ...

    if __name__ == "__main__":
        main()

Then this wrapper can delegate to:

    python -m src.data.fetch

EOF

    exit 2
fi


# -------------------------------------------------------------------
# Special wrapper information option
# -------------------------------------------------------------------

if (($# == 1)) \
    && [[ "$1" == "--wrapper-info" ]]; then
    printf '\n%sOSHA Data Fetch Wrapper%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Project root: %s\n' \
        "${PROJECT_ROOT}"

    printf 'Python:       %s\n' \
        "${PYTHON}"

    printf 'Module:       %s\n' \
        "${FETCH_MODULE}"

    printf 'Source:       %s\n' \
        "${FETCH_FILE}"

    printf 'CLI main():   yes\n'

    exit 0
fi


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------

if [[ -t 1 ]]; then
    info \
        "Delegating to ${FETCH_MODULE}"
fi


# exec is intentional:
#
#   * the Python command receives signals directly
#   * exit codes are preserved exactly
#   * there is no duplicate fetch implementation
#
exec \
    "${PYTHON}" \
    -m "${FETCH_MODULE}" \
    "$@"
