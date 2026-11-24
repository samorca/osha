#!/usr/bin/env bash

#
# OSHA Python import validation.
#
# Usage:
#
#   ./scripts/ci/check_imports.sh
#   ./scripts/ci/check_imports.sh --verbose
#   ./scripts/ci/check_imports.sh --all
#
# Default mode validates important package boundaries.
#
# --all imports every Python module discovered under src/.
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

VERBOSE=false
IMPORT_ALL=false

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
  scripts/ci/check_imports.sh [options]

Options:
  --all       Import every Python module under src/.
  --verbose   Print each imported module.
  -h, --help  Show this help.

Environment:
  OSHA_PYTHON Override the Python interpreter.

Examples:
  ./scripts/ci/check_imports.sh
  ./scripts/ci/check_imports.sh --verbose
  ./scripts/ci/check_imports.sh --all
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --all)
            IMPORT_ALL=true
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
# Project validation
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


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

printf '\n%sOSHA Import Validation%s\n' \
    "${BOLD}" \
    "${RESET}"

info "Project root: ${PROJECT_ROOT}"
info "Python: ${PYTHON}"
info "Python version: ${PYTHON_VERSION}"

if [[ "${IMPORT_ALL}" == true ]]; then
    info "Mode: all modules"
else
    info "Mode: package boundaries"
fi


# -------------------------------------------------------------------
# Ensure repository root is importable
# -------------------------------------------------------------------

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"


# -------------------------------------------------------------------
# Import validation
# -------------------------------------------------------------------

VERBOSE_VALUE=0
IMPORT_ALL_VALUE=0

if [[ "${VERBOSE}" == true ]]; then
    VERBOSE_VALUE=1
fi

if [[ "${IMPORT_ALL}" == true ]]; then
    IMPORT_ALL_VALUE=1
fi


if OSHA_IMPORT_VERBOSE="${VERBOSE_VALUE}" \
   OSHA_IMPORT_ALL="${IMPORT_ALL_VALUE}" \
   "${PYTHON}" - <<'PY'
from __future__ import annotations

import importlib
import os
import pathlib
import sys
import traceback
from dataclasses import dataclass


ROOT = pathlib.Path.cwd()
SRC = ROOT / "src"

VERBOSE = (
    os.environ.get(
        "OSHA_IMPORT_VERBOSE",
        "0",
    )
    == "1"
)

IMPORT_ALL = (
    os.environ.get(
        "OSHA_IMPORT_ALL",
        "0",
    )
    == "1"
)


@dataclass(frozen=True)
class ImportFailure:
    module: str
    error_type: str
    message: str
    traceback_text: str


def module_name_from_path(
    path: pathlib.Path,
) -> str | None:
    relative = path.relative_to(ROOT)

    if relative.suffix != ".py":
        return None

    parts = list(relative.with_suffix("").parts)

    if not parts:
        return None

    if parts[-1] == "__init__":
        parts = parts[:-1]

    if not parts:
        return None

    if any(
        not part.isidentifier()
        for part in parts
    ):
        return None

    return ".".join(parts)


def discover_all_modules() -> list[str]:
    modules: set[str] = set()

    for path in SRC.rglob("*.py"):
        if not path.is_file():
            continue

        module = module_name_from_path(path)

        if module:
            modules.add(module)

    return sorted(
        modules,
        key=lambda value: (
            value.count("."),
            value,
        ),
    )


def existing_modules(
    candidates: list[str],
) -> list[str]:
    result: list[str] = []

    for module in candidates:
        path = ROOT.joinpath(
            *module.split(".")
        )

        if (
            path.with_suffix(".py").is_file()
            or (
                path.is_dir()
                and (
                    path / "__init__.py"
                ).is_file()
            )
        ):
            result.append(module)

    return result


BOUNDARY_CANDIDATES = [
    # Root
    "src",

    # Core
    "src.core",

    # Data
    "src.data",

    # Training
    "src.training",

    # Accelerator public surface
    "src.accelerators",
    "src.accelerators.schema",
    "src.accelerators.optimization",

    # CPU
    "src.accelerators.cpu",
    "src.accelerators.cpu.backend",
    "src.accelerators.cpu.devices",
    "src.accelerators.cpu.metrics",
    "src.accelerators.cpu.optimizer",

    # CUDA
    "src.accelerators.cuda",
    "src.accelerators.cuda.backend",
    "src.accelerators.cuda.devices",
    "src.accelerators.cuda.metrics",
    "src.accelerators.cuda.optimizer",

    # ROCm
    "src.accelerators.rocm",
    "src.accelerators.rocm.backend",
    "src.accelerators.rocm.devices",
    "src.accelerators.rocm.metrics",
    "src.accelerators.rocm.optimizer",

    # AI
    "src.ai",
    "src.ai.config",
    "src.ai.schema",
    "src.ai.registry",
    "src.ai.loader",
    "src.ai.pipeline",

    # API foundation
    "src.api",
    "src.api.schema",
    "src.api.config",
    "src.api.errors",
    "src.api.middleware",
    "src.api.dependencies",

    # API services
    "src.api.services",
    "src.api.services.accelerators",
    "src.api.services.inference",
    "src.api.services.models",
    "src.api.services.incidents",
    "src.api.services.decisions",
    "src.api.services.remediation",

    # API routes
    "src.api.routes",
    "src.api.routes.health",
    "src.api.routes.accelerators",
    "src.api.routes.inference",
    "src.api.routes.models",
    "src.api.routes.incidents",
    "src.api.routes.decisions",
    "src.api.routes.remediation",

    # API composition root last
    "src.api.app",
]


if IMPORT_ALL:
    modules = discover_all_modules()
else:
    modules = existing_modules(
        BOUNDARY_CANDIDATES
    )


if not modules:
    print(
        "No importable modules were discovered.",
        file=sys.stderr,
    )

    raise SystemExit(1)


failures: list[ImportFailure] = []

passed = 0


for module in modules:
    if VERBOSE:
        print(
            f"[IMPORT] {module}"
        )

    try:
        importlib.import_module(
            module
        )

        passed += 1

        if VERBOSE:
            print(
                f"[PASS]   {module}"
            )

    except BaseException as exc:
        failures.append(
            ImportFailure(
                module=module,

                error_type=(
                    type(exc).__name__
                ),

                message=str(exc),

                traceback_text=(
                    "".join(
                        traceback.format_exception(
                            type(exc),
                            exc,
                            exc.__traceback__,
                        )
                    )
                ),
            )
        )

        print(
            (
                "[FAIL] "
                f"{module}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            file=sys.stderr,
        )


print()

print(
    "Import validation summary:"
)

print(
    f"  Discovered: {len(modules)}"
)

print(
    f"  Passed:     {passed}"
)

print(
    f"  Failed:     {len(failures)}"
)


if failures:
    print(
        "\nDetailed import failures:",
        file=sys.stderr,
    )

    for failure in failures:
        print(
            (
                "\n"
                "========================================\n"
                f"MODULE: {failure.module}\n"
                f"ERROR:  {failure.error_type}: "
                f"{failure.message}\n"
                "----------------------------------------"
            ),
            file=sys.stderr,
        )

        print(
            failure.traceback_text.rstrip(),
            file=sys.stderr,
        )

    raise SystemExit(1)


raise SystemExit(0)
PY
then
    success "Python import validation"
else
    failure "Python import validation"
    exit 1
fi


# -------------------------------------------------------------------
# Explicit API composition check
# -------------------------------------------------------------------

if [[ -f "${PROJECT_ROOT}/src/api/app.py" ]]; then
    info "Validating API composition root"

    if "${PYTHON}" - <<'PY'
from src.api.app import (
    app,
)

if app is None:
    raise RuntimeError(
        "src.api.app.app is None"
    )

if not getattr(
    app,
    "title",
    None,
):
    raise RuntimeError(
        "FastAPI application has no title"
    )

routes = getattr(
    app,
    "routes",
    None,
)

if routes is None:
    raise RuntimeError(
        "FastAPI application exposes no routes"
    )

print(
    f"API application: {app.title}"
)

print(
    f"Registered routes: {len(routes)}"
)
PY
    then
        success "API composition import"
    else
        failure "API composition import"
        exit 1
    fi
fi


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))

printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Import Check Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Python:   %s\n' \
    "${PYTHON_VERSION}"

printf 'Mode:     %s\n' \
    "$(
        if [[ "${IMPORT_ALL}" == true ]]; then
            printf 'all modules'
        else
            printf 'package boundaries'
        fi
    )"

printf 'Duration: %ss\n' \
    "${elapsed}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '\n%sIMPORT CHECK PASSED%s\n' \
    "${GREEN}" \
    "${RESET}"

exit 0
