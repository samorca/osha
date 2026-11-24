#!/usr/bin/env bash

#
# OSHA model metadata inspection.
#
# Usage:
#
#   ./scripts/models/inspect.sh MODEL_ID
#   ./scripts/models/inspect.sh MODEL_ID --json
#   ./scripts/models/inspect.sh MODEL_ID --verbose
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This command is read-only.
#
# It intentionally does NOT:
#
#   load model weights
#   allocate an accelerator
#   run inference
#   promote a model
#   modify registry state
#   alter checkpoints
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

MODEL_ID=""

JSON_OUTPUT=false
VERBOSE=false


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
  scripts/models/inspect.sh MODEL_ID [options]

Options:
  --json
      Emit machine-readable JSON.

  --verbose
      Include the complete registry metadata
      representation for the selected model.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override Python interpreter.

Examples:
  ./scripts/models/inspect.sh event-model-v1

  ./scripts/models/inspect.sh \
      anomaly-model-v1 \
      --verbose

  ./scripts/models/inspect.sh \
      anomaly-model-v1 \
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

        -h|--help)
            usage
            exit 0
            ;;

        -*)
            printf \
                'Unknown option: %s\n\n' \
                "$1" >&2

            usage >&2
            exit 2
            ;;

        *)
            if [[ -n "${MODEL_ID}" ]]; then
                printf \
                    'ERROR: multiple model IDs supplied.\n\n' \
                    >&2

                usage >&2
                exit 2
            fi

            MODEL_ID="$1"
            ;;
    esac

    shift
done


# -------------------------------------------------------------------
# Required model ID
# -------------------------------------------------------------------

if [[ -z "${MODEL_ID}" ]]; then
    printf \
        'ERROR: MODEL_ID is required.\n\n' \
        >&2

    usage >&2

    exit 2
fi


# -------------------------------------------------------------------
# Project
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -d "${PROJECT_ROOT}/src/ai" ]]; then
    failure \
        "src/ai does not exist."

    exit 2
fi


if [[ ! -f "${PROJECT_ROOT}/src/ai/registry.py" ]]; then
    failure \
        "src/ai/registry.py does not exist."

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
# Compile registry
# -------------------------------------------------------------------

if ! "${PYTHON}" -m py_compile \
    "${PROJECT_ROOT}/src/ai/registry.py"
then
    failure \
        "src/ai/registry.py failed compilation."

    exit 1
fi


# -------------------------------------------------------------------
# Import registry
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
    printf '\n%sOSHA Model Inspection%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Model ID: ${MODEL_ID}"
    info "Python: ${PYTHON_VERSION}"
fi


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

export OSHA_INSPECT_MODEL_ID="${MODEL_ID}"

export OSHA_INSPECT_JSON="$(
    if [[ "${JSON_OUTPUT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_INSPECT_VERBOSE="$(
    if [[ "${VERBOSE}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"


# -------------------------------------------------------------------
# Registry inspection
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


MODEL_ID = os.environ[
    "OSHA_INSPECT_MODEL_ID"
]

JSON_OUTPUT = (
    os.environ.get(
        "OSHA_INSPECT_JSON",
        "0",
    )
    == "1"
)

VERBOSE = (
    os.environ.get(
        "OSHA_INSPECT_VERBOSE",
        "0",
    )
    == "1"
)


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------

def serialize(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
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

    if isinstance(value, Mapping):
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
        if isinstance(value, Mapping):
            if name in value:
                return value[name]

        elif hasattr(value, name):
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
# Registry class discovery
# ------------------------------------------------------------------

registry_class = None


for name in (
    "AIModelRegistry",
    "ModelRegistry",
    "Registry",
):
    candidate = getattr(
        registry_module,
        name,
        None,
    )

    if inspect.isclass(candidate):
        registry_class = candidate
        break


if registry_class is None:
    classes = [
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
        "No supported registry class found. "
        "Available classes: "
        + ", ".join(
            sorted(classes)
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
        "Registry requires constructor "
        "arguments. Signature: "
        f"{signature}"
    ) from exc


# ------------------------------------------------------------------
# Discover existing models if supported
# ------------------------------------------------------------------

starter = getattr(
    registry,
    "start",
    None,
)


if callable(starter):
    try:
        parameters = (
            inspect.signature(
                starter
            ).parameters
        )

        if (
            "discover_existing"
            in parameters
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
# Direct lookup
# ------------------------------------------------------------------

model = None
lookup_method = None


for method_name in (
    "get_model",
    "get",
    "find_model",
    "lookup",
    "resolve",
):
    method = getattr(
        registry,
        method_name,
        None,
    )

    if not callable(method):
        continue

    try:
        signature = inspect.signature(
            method
        )

        parameters = signature.parameters

        if "model_id" in parameters:
            model = method(
                model_id=MODEL_ID
            )

        elif "identifier" in parameters:
            model = method(
                identifier=MODEL_ID
            )

        elif "name" in parameters:
            model = method(
                name=MODEL_ID
            )

        else:
            model = method(
                MODEL_ID
            )

        lookup_method = method_name

        if model is not None:
            break

    except (
        KeyError,
        LookupError,
        ValueError,
    ):
        model = None

    except TypeError:
        continue


# ------------------------------------------------------------------
# Listing fallback
# ------------------------------------------------------------------

if model is None:
    listing = None

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
                listing = candidate()

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
            listing = candidate

        if listing is not None:
            lookup_method = (
                method_name
                + ":scan"
            )

            break


    if isinstance(listing, Mapping):
        if MODEL_ID in listing:
            model = listing[
                MODEL_ID
            ]

        values = list(
            listing.values()
        )

    elif isinstance(
        listing,
        Iterable,
    ) and not isinstance(
        listing,
        (
            str,
            bytes,
        ),
    ):
        values = list(listing)

    else:
        values = []


    if model is None:
        for candidate in values:
            identifiers = {
                str(value)
                for value in (
                    value_of(
                        candidate,
                        "model_id",
                        default=None,
                    ),

                    value_of(
                        candidate,
                        "id",
                        default=None,
                    ),

                    value_of(
                        candidate,
                        "name",
                        default=None,
                    ),
                )
                if value is not None
            }

            if MODEL_ID in identifiers:
                model = candidate
                break


# ------------------------------------------------------------------
# Not found
# ------------------------------------------------------------------

if model is None:
    if JSON_OUTPUT:
        print(
            json.dumps(
                {
                    "schema_version": "1",

                    "found": False,

                    "model_id": (
                        MODEL_ID
                    ),

                    "registry": (
                        registry_class.__name__
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )

    else:
        print(
            (
                "Model not found: "
                f"{MODEL_ID}"
            )
        )

    raise SystemExit(3)


# ------------------------------------------------------------------
# Normalize selected model
# ------------------------------------------------------------------

raw = serialize(
    model
)


model_id = value_of(
    model,
    "model_id",
    "id",
    default=MODEL_ID,
)

name = value_of(
    model,
    "name",
    "model_name",
    default=model_id,
)

family = enum_text(
    value_of(
        model,
        "family",
        "task",
        "model_type",
        "type",
        default=None,
    )
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

checkpoint = value_of(
    model,
    "checkpoint",
    "checkpoint_path",
    default=None,
)

created_at = value_of(
    model,
    "created_at",
    "created",
    default=None,
)

updated_at = value_of(
    model,
    "updated_at",
    "modified_at",
    "updated",
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

metrics = serialize(
    value_of(
        model,
        "metrics",
        "evaluation",
        default=None,
    )
)

config = serialize(
    value_of(
        model,
        "model_config",
        "config",
        default=None,
    )
)

attributes = serialize(
    value_of(
        model,
        "attributes",
        "metadata",
        default=None,
    )
)


payload = {
    "schema_version": "1",

    "found": True,

    "registry": (
        registry_class.__name__
    ),

    "lookup_method": (
        lookup_method
    ),

    "model": {
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

        "checkpoint": (
            str(checkpoint)
            if checkpoint is not None
            else None
        ),

        "created_at": serialize(
            created_at
        ),

        "updated_at": serialize(
            updated_at
        ),

        "metrics": metrics,

        "config": config,

        "attributes": attributes,
    },
}


if VERBOSE:
    payload[
        "raw_registry_metadata"
    ] = raw


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
    item = payload[
        "model"
    ]

    print()

    print(
        "Model"
    )

    print(
        "-----"
    )

    print(
        (
            "ID:          "
            f"{item['model_id']}"
        )
    )

    print(
        (
            "Name:        "
            f"{item['name']}"
        )
    )

    print(
        (
            "Family:      "
            f"{item['family']}"
        )
    )

    print(
        (
            "Version:     "
            f"{item['version']}"
        )
    )

    print(
        (
            "State:       "
            f"{item['state']}"
        )
    )

    if item[
        "promoted"
    ] is not None:
        print(
            (
                "Promoted:    "
                f"{item['promoted']}"
            )
        )

    if item[
        "loaded"
    ] is not None:
        print(
            (
                "Loaded:      "
                f"{item['loaded']}"
            )
        )

    if item[
        "ready"
    ] is not None:
        print(
            (
                "Ready:       "
                f"{item['ready']}"
            )
        )

    if item[
        "path"
    ]:
        print(
            (
                "Path:        "
                f"{item['path']}"
            )
        )

    if item[
        "checkpoint"
    ]:
        print(
            (
                "Checkpoint:  "
                f"{item['checkpoint']}"
            )
        )

    if item[
        "created_at"
    ]:
        print(
            (
                "Created:     "
                f"{item['created_at']}"
            )
        )

    if item[
        "updated_at"
    ]:
        print(
            (
                "Updated:     "
                f"{item['updated_at']}"
            )
        )


    if item[
        "metrics"
    ] is not None:
        print()

        print(
            "Metrics"
        )

        print(
            "-------"
        )

        print(
            json.dumps(
                item[
                    "metrics"
                ],
                indent=2,
                sort_keys=True,
                default=str,
            )
        )


    if item[
        "config"
    ] is not None:
        print()

        print(
            "Model configuration"
        )

        print(
            "-------------------"
        )

        print(
            json.dumps(
                item[
                    "config"
                ],
                indent=2,
                sort_keys=True,
                default=str,
            )
        )


    if item[
        "attributes"
    ] is not None:
        print()

        print(
            "Attributes"
        )

        print(
            "----------"
        )

        print(
            json.dumps(
                item[
                    "attributes"
                ],
                indent=2,
                sort_keys=True,
                default=str,
            )
        )


    if VERBOSE:
        print()

        print(
            "Raw registry metadata"
        )

        print(
            "---------------------"
        )

        print(
            json.dumps(
                payload[
                    "raw_registry_metadata"
                ],
                indent=2,
                sort_keys=True,
                default=str,
            )
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

INSPECT_EXIT_CODE=$?

set -e


# -------------------------------------------------------------------
# Final result
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == true ]]; then
    exit "${INSPECT_EXIT_CODE}"
fi


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Model Inspection Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Model ID:  %s\n' \
    "${MODEL_ID}"

printf 'Python:    %s\n' \
    "${PYTHON_VERSION}"

printf 'Exit code: %s\n' \
    "${INSPECT_EXIT_CODE}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


case "${INSPECT_EXIT_CODE}" in
    0)
        printf '\n%sMODEL INSPECTION COMPLETE%s\n' \
            "${GREEN}" \
            "${RESET}"
        ;;

    3)
        printf '\n%sMODEL NOT FOUND%s\n' \
            "${RED}" \
            "${RESET}"
        ;;

    *)
        printf '\n%sMODEL INSPECTION FAILED%s\n' \
            "${RED}" \
            "${RESET}"
        ;;
esac


exit "${INSPECT_EXIT_CODE}"
