#!/usr/bin/env bash

#
# OSHA training-dataset build entrypoint.
#
# Authoritative implementation:
#
#     src/data/pipeline.py
#
# Intended flow:
#
#     source data
#         |
#         v
#     normalization
#         |
#         v
#     validation
#         |
#         v
#     deterministic splitting
#         |
#         v
#     manifests / checksums
#         |
#         v
#     training-ready datasets
#
# Usage:
#
#   ./scripts/data/build_datasets.sh --help
#   ./scripts/data/build_datasets.sh <pipeline.py arguments...>
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This wrapper does not implement dataset logic.
# All data semantics remain inside src/data/.
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

PIPELINE_MODULE="src.data.pipeline"

PIPELINE_FILE="${PROJECT_ROOT}/src/data/pipeline.py"


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


if [[ ! -f "${PIPELINE_FILE}" ]]; then
    failure \
        "Data pipeline implementation does not exist: ${PIPELINE_FILE}"

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
    failure \
        "No Python interpreter found."

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
# Compile pipeline
# -------------------------------------------------------------------

if ! "${PYTHON}" -m py_compile \
    "${PIPELINE_FILE}"
then
    failure \
        "src/data/pipeline.py failed Python compilation."

    exit 1
fi


# -------------------------------------------------------------------
# Import pipeline
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import src.data.pipeline' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.data.pipeline."

    printf '\nRun for the full traceback:\n\n' >&2

    printf '  %q -c %q\n\n' \
        "${PYTHON}" \
        'import src.data.pipeline'

    exit 1
fi


# -------------------------------------------------------------------
# Check CLI entrypoint
# -------------------------------------------------------------------

HAS_MAIN="$(
    "${PYTHON}" - <<'PY'
from __future__ import annotations

import src.data.pipeline as module


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
    printf '\n%sOSHA Dataset Build Wrapper%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Project root: %s\n' \
        "${PROJECT_ROOT}"

    printf 'Python:       %s\n' \
        "${PYTHON}"

    printf 'Version:      %s\n' \
        "${PYTHON_VERSION}"

    printf 'Module:       %s\n' \
        "${PIPELINE_MODULE}"

    printf 'Source:       %s\n' \
        "${PIPELINE_FILE}"

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
# Require Python-owned CLI
# -------------------------------------------------------------------

if [[ "${HAS_MAIN}" != "1" ]]; then
    failure \
        "src.data.pipeline does not expose a callable main()."

    cat >&2 <<'EOF'

scripts/data/build_datasets.sh intentionally does
not reconstruct the dataset pipeline in Bash.

The orchestration belongs in:

    src/data/pipeline.py

Expose a small CLI entrypoint there:

    def main() -> None:
        ...

    if __name__ == "__main__":
        main()

main() should parse command-line arguments and call
the existing pipeline implementation.

After that this wrapper can execute:

    python -m src.data.pipeline

EOF

    exit 2
fi


# -------------------------------------------------------------------
# Operator information
# -------------------------------------------------------------------

if [[ -t 1 ]]; then
    info \
        "Delegating dataset construction to ${PIPELINE_MODULE}"
fi


# -------------------------------------------------------------------
# Execute
# -------------------------------------------------------------------
#
# exec ensures:
#
#   Python receives termination signals directly
#   Python's exit code becomes this script's exit code
#   arguments are forwarded exactly
#   no dataset logic is duplicated here
#

exec \
    "${PYTHON}" \
    -m "${PIPELINE_MODULE}" \
    "$@"
