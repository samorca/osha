#!/usr/bin/env bash

#
# OSHA Python compilation check.
#
# Usage:
#
#   ./scripts/ci/compile.sh
#
#   ./scripts/ci/compile.sh --verbose
#
#   ./scripts/ci/compile.sh --all
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# Default:
#
#   compile src/
#
# --all:
#
#   compile src/ and tests/ when tests exists.
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

VERBOSE=false
COMPILE_ALL=false


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
# Usage
# -------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  scripts/ci/compile.sh [options]

Options:
  --all       Compile src/ and tests/ when tests/ exists.
  --verbose   Show individual files as they compile.
  -h, --help  Show this help.

Environment:
  OSHA_PYTHON Override the Python interpreter.

Examples:
  ./scripts/ci/compile.sh
  ./scripts/ci/compile.sh --verbose
  ./scripts/ci/compile.sh --all
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --all)
            COMPILE_ALL=true
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
# Root validation
# -------------------------------------------------------------------

if [[ ! -d "${PROJECT_ROOT}/src" ]]; then
    failure "src/ directory not found: ${PROJECT_ROOT}/src"
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
    failure "No Python interpreter found."
    exit 2
fi


if [[ ! -x "${PYTHON}" ]]; then
    failure "Python interpreter is not executable: ${PYTHON}"
    exit 2
fi


# -------------------------------------------------------------------
# Interpreter information
# -------------------------------------------------------------------

PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"

info "Project root: ${PROJECT_ROOT}"
info "Python: ${PYTHON}"
info "Python version: ${PYTHON_VERSION}"


# -------------------------------------------------------------------
# Compile targets
# -------------------------------------------------------------------

TARGETS=(
    "${PROJECT_ROOT}/src"
)


if [[ "${COMPILE_ALL}" == true ]]; then
    if [[ -d "${PROJECT_ROOT}/tests" ]]; then
        TARGETS+=(
            "${PROJECT_ROOT}/tests"
        )
    else
        warn "tests/ does not exist; compiling src/ only"
    fi
fi


# -------------------------------------------------------------------
# Count Python files
# -------------------------------------------------------------------

PYTHON_FILE_COUNT=0

for target in "${TARGETS[@]}"; do
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
done


if ((PYTHON_FILE_COUNT == 0)); then
    failure "No Python source files found."
    exit 1
fi


info "Python files: ${PYTHON_FILE_COUNT}"


# -------------------------------------------------------------------
# Compilation
# -------------------------------------------------------------------

printf '\n%sCompiling OSHA Python sources%s\n' \
    "${BOLD}" \
    "${RESET}"


COMPILE_ARGS=(
    -m
    compileall
    -f
)


if [[ "${VERBOSE}" == false ]]; then
    COMPILE_ARGS+=(
        -q
    )
fi


# Use checked-hash pyc files when supported.
#
# This makes the bytecode validation content-based
# rather than relying only on source timestamps.
COMPILE_ARGS+=(
    --invalidation-mode
    checked-hash
)


if "${PYTHON}" \
    "${COMPILE_ARGS[@]}" \
    "${TARGETS[@]}"
then
    success "Python bytecode compilation"
else
    failure "Python bytecode compilation"
    exit 1
fi


# -------------------------------------------------------------------
# Explicit source syntax validation
# -------------------------------------------------------------------
#
# compileall already catches syntax errors.
#
# This second pass uses Python's compile() directly
# against each source file. It gives this CI check a
# deterministic source-level validation independent
# of existing __pycache__ state.
#

info "Running source syntax validation"


if "${PYTHON}" - "${TARGETS[@]}" <<'PY'
from __future__ import annotations

import pathlib
import sys


targets = [
    pathlib.Path(value)
    for value in sys.argv[1:]
]


files: list[pathlib.Path] = []

for target in targets:
    if target.is_file():
        if target.suffix == ".py":
            files.append(target)
        continue

    files.extend(
        path
        for path in target.rglob("*.py")
        if path.is_file()
    )


files.sort(
    key=lambda path: str(path)
)


failures: list[
    tuple[
        pathlib.Path,
        BaseException,
    ]
] = []


for path in files:
    try:
        source = path.read_bytes()

        compile(
            source,
            str(path),
            "exec",
            dont_inherit=True,
        )

    except (
        SyntaxError,
        UnicodeError,
        ValueError,
    ) as exc:
        failures.append(
            (
                path,
                exc,
            )
        )


if failures:
    print(
        "",
        file=sys.stderr,
    )

    print(
        "Python source validation failures:",
        file=sys.stderr,
    )

    for path, error in failures:
        print(
            f"\n  {path}",
            file=sys.stderr,
        )

        if isinstance(
            error,
            SyntaxError,
        ):
            line = (
                error.lineno
                if error.lineno
                is not None
                else "?"
            )

            column = (
                error.offset
                if error.offset
                is not None
                else "?"
            )

            print(
                (
                    "    "
                    f"line {line}, "
                    f"column {column}: "
                    f"{error.msg}"
                ),
                file=sys.stderr,
            )

            if error.text:
                print(
                    "    "
                    + error.text.rstrip(),
                    file=sys.stderr,
                )

        else:
            print(
                (
                    "    "
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
                file=sys.stderr,
            )

    raise SystemExit(1)


print(
    f"Validated {len(files)} Python source files."
)
PY
then
    success "Python source syntax validation"
else
    failure "Python source syntax validation"
    exit 1
fi


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Compilation Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Python: %s\n' \
    "${PYTHON_VERSION}"

printf 'Files:  %s\n' \
    "${PYTHON_FILE_COUNT}"

printf 'Targets:\n'

for target in "${TARGETS[@]}"; do
    printf '  - %s\n' \
        "${target#${PROJECT_ROOT}/}"
done

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '\n%sCOMPILATION PASSED%s\n' \
    "${GREEN}" \
    "${RESET}"

exit 0
