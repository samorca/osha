#!/usr/bin/env bash

#
# OSHA Python formatting.
#
# Usage:
#
#   ./scripts/dev/format.sh
#   ./scripts/dev/format.sh --write
#   ./scripts/dev/format.sh --diff
#   ./scripts/dev/format.sh --verbose
#   ./scripts/dev/format.sh src/api
#
# Default behavior:
#
#   Read-only formatting check.
#
# --write:
#
#   Apply Ruff formatting in-place.
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

VENV_DIR="${PROJECT_ROOT}/.venv"


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

WRITE=false
DIFF=false
VERBOSE=false

TARGETS=()

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
  scripts/dev/format.sh [options] [path...]

Options:
  --write      Format files in-place.
  --diff       Show the formatting diff without modifying files.
  --verbose    Enable verbose Ruff output.
  -h, --help   Show this help.

Default:
  Performs a read-only formatting check.

Default targets:
  src/
  tests/       when present

Environment:
  OSHA_PYTHON  Override Python interpreter.

Examples:
  ./scripts/dev/format.sh
  ./scripts/dev/format.sh --diff
  ./scripts/dev/format.sh --write
  ./scripts/dev/format.sh src/api
  ./scripts/dev/format.sh --write src/api
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --write)
            WRITE=true
            ;;

        --diff)
            DIFF=true
            ;;

        --verbose)
            VERBOSE=true
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        --)
            shift

            while (($# > 0)); do
                TARGETS+=(
                    "$1"
                )

                shift
            done

            break
            ;;

        -*)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;

        *)
            TARGETS+=(
                "$1"
            )
            ;;
    esac

    shift
done


# -------------------------------------------------------------------
# Argument validation
# -------------------------------------------------------------------

if [[ "${WRITE}" == true ]] \
    && [[ "${DIFF}" == true ]]; then
    failure \
        "--write and --diff cannot be used together"

    exit 2
fi


# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -d "${PROJECT_ROOT}/src" ]]; then
    failure \
        "src/ directory not found: ${PROJECT_ROOT}/src"

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


# -------------------------------------------------------------------
# Ruff discovery
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("ruff") else 1)' \
    >/dev/null 2>&1
then
    failure \
        "Ruff is not installed in the selected environment."

    printf '\nRun:\n\n' >&2
    printf '  ./scripts/dev/setup.sh\n\n' >&2

    exit 2
fi


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"

RUFF_VERSION="$(
    "${PYTHON}" -m ruff \
        --version
)"


# -------------------------------------------------------------------
# Default targets
# -------------------------------------------------------------------

if ((${#TARGETS[@]} == 0)); then
    TARGETS+=(
        "src"
    )

    if [[ -d "${PROJECT_ROOT}/tests" ]]; then
        TARGETS+=(
            "tests"
        )
    fi
fi


# -------------------------------------------------------------------
# Validate targets
# -------------------------------------------------------------------

for target in "${TARGETS[@]}"; do
    if [[ ! -e "${PROJECT_ROOT}/${target}" ]] \
        && [[ ! -e "${target}" ]]
    then
        failure \
            "Format target does not exist: ${target}"

        exit 2
    fi
done


# -------------------------------------------------------------------
# Count Python files
# -------------------------------------------------------------------

PYTHON_FILE_COUNT=0


for target in "${TARGETS[@]}"; do
    if [[ -f "${target}" ]]; then
        if [[ "${target}" == *.py ]]; then
            PYTHON_FILE_COUNT=$((PYTHON_FILE_COUNT + 1))
        fi

    elif [[ -d "${target}" ]]; then
        count="$(
            find "${target}" \
                -type f \
                -name '*.py' \
                -print \
            | wc -l
        )"

        count="$(
            printf '%s' "${count}" \
            | tr -d '[:space:]'
        )"

        PYTHON_FILE_COUNT=$((PYTHON_FILE_COUNT + count))
    fi
done


if ((PYTHON_FILE_COUNT == 0)); then
    warn "No Python files found in selected targets"

    exit 0
fi


# -------------------------------------------------------------------
# Ruff formatter arguments
# -------------------------------------------------------------------

RUFF_ARGS=(
    format
)


if [[ "${WRITE}" == false ]]; then
    RUFF_ARGS+=(
        --check
    )
fi


if [[ "${DIFF}" == true ]]; then
    RUFF_ARGS+=(
        --diff
    )
fi


if [[ "${VERBOSE}" == true ]]; then
    RUFF_ARGS+=(
        --verbose
    )
fi


RUFF_ARGS+=(
    "${TARGETS[@]}"
)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

printf '\n%sOSHA Python Format%s\n' \
    "${BOLD}" \
    "${RESET}"

info "Project root: ${PROJECT_ROOT}"
info "Python: ${PYTHON}"
info "Python version: ${PYTHON_VERSION}"
info "Ruff: ${RUFF_VERSION}"
info "Python files: ${PYTHON_FILE_COUNT}"

if [[ "${WRITE}" == true ]]; then
    MODE="write"

elif [[ "${DIFF}" == true ]]; then
    MODE="diff"

else
    MODE="check"
fi

info "Mode: ${MODE}"

printf '[INFO] Targets:'

for target in "${TARGETS[@]}"; do
    printf ' %s' "${target}"
done

printf '\n'


# -------------------------------------------------------------------
# Execute formatter
# -------------------------------------------------------------------

printf '\n%sRunning Ruff formatter%s\n' \
    "${BOLD}" \
    "${RESET}"


set +e

"${PYTHON}" -m ruff \
    "${RUFF_ARGS[@]}"

RUFF_EXIT_CODE=$?

set -e


# -------------------------------------------------------------------
# Result summary
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Format Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Mode:         %s\n' \
    "${MODE}"

printf 'Python files: %s\n' \
    "${PYTHON_FILE_COUNT}"

printf 'Exit code:    %s\n' \
    "${RUFF_EXIT_CODE}"

printf 'Duration:     %ss\n' \
    "${elapsed}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


# -------------------------------------------------------------------
# Failure
# -------------------------------------------------------------------

if ((RUFF_EXIT_CODE != 0)); then
    if [[ "${MODE}" == "check" ]]; then
        printf '\n%sFORMAT CHECK FAILED%s\n' \
            "${RED}" \
            "${RESET}"

        printf '\nSome Python files need formatting.\n'

        printf '\nPreview the changes with:\n\n'

        printf '  ./scripts/dev/format.sh --diff\n\n'

        printf 'Apply the changes with:\n\n'

        printf '  ./scripts/dev/format.sh --write\n\n'

    elif [[ "${MODE}" == "diff" ]]; then
        printf '\n%sFORMAT DIFFERENCES FOUND%s\n' \
            "${YELLOW}" \
            "${RESET}"

        printf '\nApply them with:\n\n'

        printf '  ./scripts/dev/format.sh --write\n\n'

    else
        printf '\n%sFORMATTING FAILED%s\n' \
            "${RED}" \
            "${RESET}"
    fi

    exit "${RUFF_EXIT_CODE}"
fi


# -------------------------------------------------------------------
# Success
# -------------------------------------------------------------------

if [[ "${MODE}" == "write" ]]; then
    printf '\n%sFORMATTING COMPLETE%s\n' \
        "${GREEN}" \
        "${RESET}"

    printf '\nValidate the changed source with:\n\n'

    printf '  ./scripts/ci/verify.sh --quick\n\n'

else
    printf '\n%sFORMAT CHECK PASSED%s\n' \
        "${GREEN}" \
        "${RESET}"
fi


exit 0
