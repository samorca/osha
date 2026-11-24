#!/usr/bin/env bash

#
# OSHA corpus preparation entrypoint.
#
# Authoritative implementation:
#
#     src/data/prepare_corpus.py
#
# Usage:
#
#   ./scripts/data/prepare.sh --help
#   ./scripts/data/prepare.sh <prepare_corpus.py arguments...>
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This wrapper intentionally does not implement:
#
#   record normalization
#   schema conversion
#   corpus generation
#   deduplication
#   data splitting
#   manifest generation
#   checksum logic
#
# Those responsibilities belong to src/data/.
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

PREPARE_MODULE="src.data.prepare_corpus"

PREPARE_FILE="$PROJECT_ROOT/src/data/prepare_corpus.py"


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


if [[ ! -f "${PREPARE_FILE}" ]]; then
    failure \
        "Corpus preparation implementation does not exist: ${PREPARE_FILE}"

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
# Python version
# -------------------------------------------------------------------

PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


# -------------------------------------------------------------------
# Compilation validation
# -------------------------------------------------------------------

if ! "${PYTHON}" -m py_compile \
    "${PREPARE_FILE}"
then
    failure \
        "src/data/prepare_corpus.py failed Python compilation."

    exit 1
fi


# -------------------------------------------------------------------
# Import validation
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import src.data.prepare_corpus' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.data.prepare_corpus."

    printf '\nRun this for the full traceback:\n\n' >&2

    printf '  %q -c %q\n\n' \
        "${PYTHON}" \
        'import src.data.prepare_corpus'

    exit 1
fi


# -------------------------------------------------------------------
# Inspect module entrypoint
# -------------------------------------------------------------------

HAS_MAIN="$(
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import src.data.prepare_corpus as module


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


if [[ "${HAS_MAIN}" != "1" ]]; then
    failure \
        "src.data.prepare_corpus does not expose a callable main()."

    cat >&2 <<'EOF'

scripts/data/prepare.sh is deliberately only an
operator wrapper.

The data transformation and corpus preparation
logic must stay in src/data/prepare_corpus.py.

Add a CLI entrypoint there:

    def main() -> None:
        ...

    if __name__ == "__main__":
        main()

After that, prepare.sh can delegate using:

    python -m src.data.prepare_corpus

EOF

    exit 2
fi


# -------------------------------------------------------------------
# Wrapper information
# -------------------------------------------------------------------

if (($# == 1)) \
    && [[ "$1" == "--wrapper-info" ]]; then
    printf '\n%sOSHA Data Preparation Wrapper%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Project root: %s\n' \
        "${PROJECT_ROOT}"

    printf 'Python:       %s\n' \
        "${PYTHON}"

    printf 'Version:      %s\n' \
        "${PYTHON_VERSION}"

    printf 'Module:       %s\n' \
        "${PREPARE_MODULE}"

    printf 'Source:       %s\n' \
        "${PREPARE_FILE}"

    printf 'CLI main():   yes\n'

    exit 0
fi


# -------------------------------------------------------------------
# Operator information
# -------------------------------------------------------------------

if [[ -t 1 ]]; then
    info \
        "Delegating corpus preparation to ${PREPARE_MODULE}"
fi


# -------------------------------------------------------------------
# Execute
# -------------------------------------------------------------------
#
# exec is intentional:
#
#   * Python receives signals directly
#   * Python's exit status is preserved
#   * shell does not duplicate data logic
#   * all arguments are forwarded unchanged
#

exec \
    "${PYTHON}" \
    -m "${PREPARE_MODULE}" \
    "$@"
