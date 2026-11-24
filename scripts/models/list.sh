#!/usr/bin/env bash

#
# OSHA model inventory.
#
# Usage:
#
#   ./scripts/models/list.sh
#   ./scripts/models/list.sh --json
#   ./scripts/models/list.sh --verbose
#   ./scripts/models/list.sh --family event
#   ./scripts/models/list.sh --family anomaly
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This command is read-only.
#
# It does not:
#
#   load model weights onto an accelerator
#   promote models
#   delete models
#   change serving state
#   modify checkpoints
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

JSON_OUTPUT=false
VERBOSE=false
FAMILY=""


# -------------------------------------------------------------------
# Colours
# -------------------------------------------------------------------

if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    RESET=$'\033[0m'
    GREEN=$'\033[32m'
    RED=$'\033[31m'
    BLUE=$'\033[34m'
    BOLD=$'\033[1m'
else
    RESET=""
    GREEN=""
    RED=""
    BLUE=""
    BOLD=""
fi


# -------------------------------------------------------------------
# Output
# -------------------------------------------------------------------

info() {
    if [[ "${JSON_OUTPUT}" == false ]]; then
        printf '%s[INFO]%s %s\n' \
            "${BLUE}" \
            "${RESET}" \
            "$*"
    fi
}


failure() {
    if [[ "${JSON_OUTPUT}" == false ]]; then
        printf '%s[FAIL]%s %s\n' \
            "${RED}" \
            "${RESET}" \
            "$*" >&2
    fi
}


# -------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  scripts/models/list.sh [options]

Options:
  --json
      Emit machine-readable JSON.

  --verbose
      Include additional model metadata.

  --family FAMILY
      Filter models by family/type.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override the Python interpreter.

Examples:
  ./scripts/models/list.sh

  ./scripts/models/list.sh \
      --family event

  ./scripts/models/list.sh \
      --family anomaly

  ./scripts/models/list.sh \
      --verbose

  ./scripts/models/list.sh \
      --json
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --json)
            JSON_OUTPUT=true
            ;;

        --verbose)
            VERBOSE=true
            ;;

        --family)
            if (($# < 2)); then
                printf \
                    'ERROR: --family requires a value\n' \
                    >&2

                exit 2
            fi

            FAMILY="$2"
            shift
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
# Project
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -d "${PROJECT_ROOT}/src/ai" ]]; then
    failure \
        "src/ai does not exist."

    exit 2
fi


# -------------------------------------------------------------------
# Python
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
        "Python is not executable: ${PYTHON}"

    exit 2
fi


export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


# -------------------------------------------------------------------
# Registry import
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import src.ai.registry' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.ai.registry."

    printf '\nRun for the full traceback:\n\n' >&2

    printf '  %q -c %q\n\n' \
        "${PYTHON}" \
        'import src.ai.registry'

    exit 1
fi


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%sOSHA Model Inventory%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Project root: ${PROJECT_ROOT}"
    info "Python: ${PYTHON_VERSION}"

    if [[ -n "${FAMILY}" ]]; then
        info "Family filter: ${FAMILY}"
    else
        info "Family filter: all"
    fi
fi


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

export OSHA_MODELS_JSON="$(
    if [[ "${JSON_OUTPUT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_MODELS_VERBOSE="$(
    if [[ "${VERBOSE}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_MODELS_FAMILY="${FAMILY}"


# -------------------------------------------------------------------
# Model registry inspection
# -------------------------------------------------------------------

set +e

"${PYTHON}" - <<'PY'
from __future__ import annotations

import inspect
import json
import os

from dataclasses import (
    asdict,
    is_dataclass,
)

from enum import Enum

from pathlib import Path

from typing import (
    Any,
    Iterable,
    Mapping,
)

import src.ai.registry as registry_module


JSON_OUTPUT = (
    os.environ.get(
        "OSHA_MODELS_JSON",
        "0",
    )
    == "1"
)

VERBOSE = (
    os.environ.get(
        "OSHA_MODELS_VERBOSE",
        "0",
    )
    == "1"
)

FAMILY = (
    os.environ.get(
        "OSHA_MODELS_FAMILY",
        "",
    )
    .strip()
    .lower()
)


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------

def serialize(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        Enum,
    ):
        return value.value

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): serialize(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        return [
            serialize(item)
            for item in value
        ]

    if is_dataclass(value):
        return serialize(
            asdict(value)
        )

    converter = getattr(
        value,
        "to_dict",
        None,
    )

    if callable(converter):
        try:
            return serialize(
                converter()
            )

        except Exception:
            pass

    dumper = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(dumper):
        try:
            return serialize(
                dumper()
            )

        except Exception:
            pass

    return str(value)


def value_of(
    value: Any,
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if isinstance(
            value,
            Mapping,
        ):
            if name in value:
                return value[name]

        elif hasattr(
            value,
            name,
        ):
            return getattr(
                value,
                name,
            )

    return default


def enum_text(
    value: Any,
) -> str:
    if isinstance(value, Enum):
        return str(value.value)

    if value is None:
        return ""

    return str(value)


# ------------------------------------------------------------------
# Find registry class
# ------------------------------------------------------------------

REGISTRY_CLASSES = (
    "AIModelRegistry",
    "ModelRegistry",
    "Registry",
)


registry_class = None


for name in REGISTRY_CLASSES:
    candidate = getattr(
        registry_module,
        name,
        None,
    )

    if inspect.isclass(candidate):
        registry_class = candidate
        break


if registry_class is None:
    available_classes = [
        name
        for name, value
        in vars(
            registry_module
        ).items()
        if (
            inspect.isclass(value)
            and value.__module__
            == registry_module.__name__
        )
    ]

    raise RuntimeError(
        "No supported model registry class "
        "was found in src.ai.registry. "
        "Available classes: "
        + ", ".join(
            sorted(
                available_classes
            )
        )
    )


# ------------------------------------------------------------------
# Construct registry
# ------------------------------------------------------------------

try:
    registry = registry_class()

except TypeError as exc:
    signature = inspect.signature(
        registry_class
    )

    raise RuntimeError(
        "The model registry requires explicit "
        "constructor parameters. "
        f"Signature: {signature}"
    ) from exc


# ------------------------------------------------------------------
# Start/discover when supported
# ------------------------------------------------------------------

starter = getattr(
    registry,
    "start",
    None,
)


if callable(starter):
    try:
        signature = inspect.signature(
            starter
        )

        if (
            "discover_existing"
            in signature.parameters
        ):
            starter(
                discover_existing=True
            )

        else:
            starter()

    except TypeError:
        starter()


for method_name in (
    "discover",
    "refresh",
    "scan",
):
    method = getattr(
        registry,
        method_name,
        None,
    )

    if callable(method):
        try:
            method()

        except TypeError:
            pass

        break


# ------------------------------------------------------------------
# Retrieve models
# ------------------------------------------------------------------

result = None
used_method = None


for method_name in (
    "list_models",
    "models",
    "list",
    "all",
    "entries",
):
    candidate = getattr(
        registry,
        method_name,
        None,
    )

    if candidate is None:
        continue

    if callable(candidate):
        try:
            signature = inspect.signature(
                candidate
            )

            arguments = {}

            if (
                FAMILY
                and "family"
                in signature.parameters
            ):
                arguments[
                    "family"
                ] = FAMILY

            result = candidate(
                **arguments
            )

            used_method = method_name
            break

        except TypeError:
            continue

    elif isinstance(
        candidate,
        (
            Mapping,
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        result = candidate
        used_method = method_name
        break


if result is None:
    for attribute in (
        "_models",
        "_entries",
        "_registry",
    ):
        candidate = getattr(
            registry,
            attribute,
            None,
        )

        if isinstance(
            candidate,
            (
                Mapping,
                list,
                tuple,
                set,
                frozenset,
            ),
        ):
            result = candidate
            used_method = attribute
            break


if result is None:
    methods = [
        name
        for name in dir(registry)
        if (
            not name.startswith("_")
            and callable(
                getattr(
                    registry,
                    name,
                    None,
                )
            )
        )
    ]

    raise RuntimeError(
        "Unable to locate a model listing API "
        "on the model registry. Public methods: "
        + ", ".join(
            methods
        )
    )


# ------------------------------------------------------------------
# Normalize result
# ------------------------------------------------------------------

if isinstance(
    result,
    Mapping,
):
    source_models = list(
        result.values()
    )

elif isinstance(
    result,
    Iterable,
) and not isinstance(
    result,
    (
        str,
        bytes,
    ),
):
    source_models = list(result)

else:
    source_models = [
        result
    ]


models = []


for model in source_models:
    family = enum_text(
        value_of(
            model,
            "family",
            "task",
            "model_type",
            "type",
            default="",
        )
    ).lower()

    if (
        FAMILY
        and family != FAMILY
    ):
        continue

    model_id = value_of(
        model,
        "model_id",
        "id",
        "name",
        default=None,
    )

    name = value_of(
        model,
        "name",
        "model_name",
        default=model_id,
    )

    version = value_of(
        model,
        "version",
        "model_version",
        default=None,
    )

    state = enum_text(
        value_of(
            model,
            "state",
            "status",
            default="unknown",
        )
    )

    path = value_of(
        model,
        "path",
        "checkpoint_path",
        "artifact_path",
        default=None,
    )

    promoted = value_of(
        model,
        "promoted",
        "is_promoted",
        default=None,
    )

    loaded = value_of(
        model,
        "loaded",
        "is_loaded",
        default=None,
    )

    ready = value_of(
        model,
        "ready",
        "is_ready",
        default=None,
    )

    entry = {
        "model_id": (
            str(model_id)
            if model_id is not None
            else None
        ),

        "name": (
            str(name)
            if name is not None
            else None
        ),

        "family": (
            family
            or None
        ),

        "version": (
            str(version)
            if version is not None
            else None
        ),

        "state": state,

        "promoted": (
            promoted
        ),

        "loaded": (
            loaded
        ),

        "ready": (
            ready
        ),

        "path": (
            str(path)
            if path is not None
            else None
        ),
    }

    if VERBOSE:
        entry[
            "metadata"
        ] = serialize(
            model
        )

    models.append(
        entry
    )


models.sort(
    key=lambda item: (
        item.get(
            "family"
        )
        or "",

        item.get(
            "name"
        )
        or "",

        item.get(
            "version"
        )
        or "",

        item.get(
            "model_id"
        )
        or "",
    )
)


family_counts: dict[
    str,
    int,
] = {}


for model in models:
    family = (
        model[
            "family"
        ]
        or "unknown"
    )

    family_counts[
        family
    ] = (
        family_counts.get(
            family,
            0,
        )
        + 1
    )


payload = {
    "schema_version": "1",

    "registry": (
        registry_class.__name__
    ),

    "listing_method": (
        used_method
    ),

    "filters": {
        "family": (
            FAMILY
            or None
        ),
    },

    "summary": {
        "model_count": len(
            models
        ),

        "families": (
            family_counts
        ),
    },

    "models": models,
}


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

if JSON_OUTPUT:
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

else:
    print()

    print(
        "Summary"
    )

    print(
        "-------"
    )

    print(
        (
            "Registry: "
            f"{registry_class.__name__}"
        )
    )

    print(
        (
            "Models:   "
            f"{len(models)}"
        )
    )

    if family_counts:
        print(
            "Families:"
        )

        for family in sorted(
            family_counts
        ):
            print(
                (
                    f"  {family:<16} "
                    f"{family_counts[family]}"
                )
            )

    print()

    print(
        "Models"
    )

    print(
        "------"
    )

    if not models:
        print(
            "No models matched."
        )

    for index, model in enumerate(
        models,
        start=1,
    ):
        print()

        print(
            (
                f"[{index}] "
                f"{model['model_id']}"
            )
        )

        print(
            (
                "    Name:     "
                f"{model['name']}"
            )
        )

        print(
            (
                "    Family:   "
                f"{model['family']}"
            )
        )

        print(
            (
                "    Version:  "
                f"{model['version']}"
            )
        )

        print(
            (
                "    State:    "
                f"{model['state']}"
            )
        )

        if (
            model[
                "promoted"
            ]
            is not None
        ):
            print(
                (
                    "    Promoted: "
                    f"{model['promoted']}"
                )
            )

        if (
            model[
                "loaded"
            ]
            is not None
        ):
            print(
                (
                    "    Loaded:   "
                    f"{model['loaded']}"
                )
            )

        if (
            model[
                "ready"
            ]
            is not None
        ):
            print(
                (
                    "    Ready:    "
                    f"{model['ready']}"
                )
            )

        if model[
            "path"
        ]:
            print(
                (
                    "    Path:     "
                    f"{model['path']}"
                )
            )

        if (
            VERBOSE
            and model.get(
                "metadata"
            )
        ):
            print(
                "    Metadata:"
            )

            metadata = json.dumps(
                model[
                    "metadata"
                ],
                indent=2,
                sort_keys=True,
                default=str,
            )

            for line in (
                metadata.splitlines()
            ):
                print(
                    "        " + line
                )


# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------

closer = getattr(
    registry,
    "close",
    None,
)


if callable(closer):
    try:
        closer()

    except Exception:
        pass
PY

LIST_EXIT_CODE=$?

set -e


# -------------------------------------------------------------------
# Final result
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == true ]]; then
    exit "${LIST_EXIT_CODE}"
fi


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Model Inventory Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Python:    %s\n' \
    "${PYTHON_VERSION}"

printf 'Family:    %s\n' \
    "$(
        if [[ -n "${FAMILY}" ]]; then
            printf '%s' "${FAMILY}"
        else
            printf 'all'
        fi
    )"

printf 'Exit code: %s\n' \
    "${LIST_EXIT_CODE}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


if ((LIST_EXIT_CODE == 0)); then
    printf '\n%sMODEL INVENTORY COMPLETE%s\n' \
        "${GREEN}" \
        "${RESET}"
else
    printf '\n%sMODEL INVENTORY FAILED%s\n' \
        "${RED}" \
        "${RESET}"
fi


exit "${LIST_EXIT_CODE}"
