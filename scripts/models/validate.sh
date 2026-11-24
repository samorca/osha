#!/usr/bin/env bash

#
# OSHA model artifact validation.
#
# Usage:
#
#   ./scripts/models/validate.sh MODEL_ID
#   ./scripts/models/validate.sh MODEL_ID --json
#   ./scripts/models/validate.sh MODEL_ID --verbose
#   ./scripts/models/validate.sh MODEL_ID --strict
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This command is read-only.
#
# It validates:
#
#   registry presence
#   model descriptor
#   artifact/checkpoint path
#   regular-file status
#   file size
#   SHA-256 digest
#   checkpoint readability
#   checkpoint metadata/configuration
#
# It does NOT:
#
#   promote a model
#   load a model for serving
#   allocate an accelerator
#   run inference
#   modify checkpoints
#   modify registry state
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
STRICT=false


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
  scripts/models/validate.sh MODEL_ID [options]

Options:
  --json
      Emit machine-readable JSON.

  --verbose
      Include additional checkpoint metadata.

  --strict
      Treat missing optional metadata/configuration
      as validation failures.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override Python interpreter.

Examples:
  ./scripts/models/validate.sh MODEL_ID

  ./scripts/models/validate.sh \
      MODEL_ID \
      --verbose

  ./scripts/models/validate.sh \
      MODEL_ID \
      --strict

  ./scripts/models/validate.sh \
      MODEL_ID \
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

        --strict)
            STRICT=true
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
                    'ERROR: multiple MODEL_ID values supplied.\n\n' \
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
# Model ID
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
# Core imports
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import src.ai.registry' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.ai.registry."

    exit 1
fi


if ! "${PYTHON}" -c \
    'import torch' \
    >/dev/null 2>&1
then
    failure \
        "PyTorch cannot be imported."

    exit 1
fi


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%sOSHA Model Validation%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Model ID: ${MODEL_ID}"
    info "Python: ${PYTHON_VERSION}"
    info "Strict: ${STRICT}"
fi


# -------------------------------------------------------------------
# Environment passed to Python
# -------------------------------------------------------------------

export OSHA_VALIDATE_MODEL_ID="${MODEL_ID}"

export OSHA_VALIDATE_JSON="$(
    if [[ "${JSON_OUTPUT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_VALIDATE_VERBOSE="$(
    if [[ "${VERBOSE}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_VALIDATE_STRICT="$(
    if [[ "${STRICT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

set +e

"${PYTHON}" - <<'PY'
from __future__ import annotations

import hashlib
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

import torch

import src.ai.registry as registry_module


MODEL_ID = os.environ[
    "OSHA_VALIDATE_MODEL_ID"
]

JSON_OUTPUT = (
    os.environ.get(
        "OSHA_VALIDATE_JSON",
        "0",
    )
    == "1"
)

VERBOSE = (
    os.environ.get(
        "OSHA_VALIDATE_VERBOSE",
        "0",
    )
    == "1"
)

STRICT = (
    os.environ.get(
        "OSHA_VALIDATE_STRICT",
        "0",
    )
    == "1"
)


# ------------------------------------------------------------------
# Helpers
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


def add_result(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    required: bool,
    message: str,
    details: Any = None,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": passed,
            "required": required,
            "message": message,
            "details": serialize(
                details
            ),
        }
    )


# ------------------------------------------------------------------
# Registry construction
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
    raise RuntimeError(
        "No supported registry class found."
    )


try:
    registry = registry_class()

except TypeError as exc:
    raise RuntimeError(
        "Registry requires explicit constructor "
        f"parameters: {inspect.signature(registry_class)}"
    ) from exc


# ------------------------------------------------------------------
# Registry startup/discovery
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
# Locate model
# ------------------------------------------------------------------

model = None


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

        parameters = (
            signature.parameters
        )

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
# List fallback
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
        value = getattr(
            registry,
            method_name,
            None,
        )

        if value is None:
            continue

        if callable(value):
            try:
                listing = value()

            except TypeError:
                continue

        else:
            listing = value

        if listing is not None:
            break


    if isinstance(
        listing,
        Mapping,
    ):
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
        values = list(
            listing
        )

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
# Initial validation
# ------------------------------------------------------------------

checks: list[
    dict[
        str,
        Any,
    ]
] = []


if model is None:
    add_result(
        checks,
        name="registry_lookup",
        passed=False,
        required=True,
        message=(
            "Model was not found in registry."
        ),
    )

    payload = {
        "schema_version": "1",
        "model_id": MODEL_ID,
        "valid": False,
        "checks": checks,
    }

    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        if JSON_OUTPUT
        else (
            "Model not found: "
            + MODEL_ID
        )
    )

    raise SystemExit(3)


add_result(
    checks,
    name="registry_lookup",
    passed=True,
    required=True,
    message="Model found in registry.",
)


# ------------------------------------------------------------------
# Model metadata
# ------------------------------------------------------------------

raw_metadata = serialize(
    model
)


model_id = value_of(
    model,
    "model_id",
    "id",
    default=MODEL_ID,
)

family = value_of(
    model,
    "family",
    "task",
    "model_type",
    "type",
    default=None,
)

model_config = value_of(
    model,
    "model_config",
    "config",
    default=None,
)

path_value = value_of(
    model,
    "checkpoint_path",
    "artifact_path",
    "path",
    default=None,
)


add_result(
    checks,
    name="model_identifier",
    passed=(
        model_id is not None
    ),
    required=True,
    message=(
        "Model identifier is present."
        if model_id is not None
        else "Model identifier is missing."
    ),
)


add_result(
    checks,
    name="model_family",
    passed=(
        family is not None
    ),
    required=STRICT,
    message=(
        "Model family is present."
        if family is not None
        else "Model family is not declared."
    ),
)


add_result(
    checks,
    name="model_config",
    passed=(
        model_config is not None
    ),
    required=STRICT,
    message=(
        "Model configuration is present."
        if model_config is not None
        else "Model configuration is not declared."
    ),
)


# ------------------------------------------------------------------
# Artifact path
# ------------------------------------------------------------------

artifact_path: Path | None = None


if path_value is not None:
    artifact_path = Path(
        str(path_value)
    ).expanduser()

    if not artifact_path.is_absolute():
        artifact_path = (
            Path.cwd()
            / artifact_path
        )

    artifact_path = (
        artifact_path.resolve()
    )


add_result(
    checks,
    name="artifact_path",
    passed=(
        artifact_path is not None
    ),
    required=True,
    message=(
        (
            "Artifact path resolved to "
            f"{artifact_path}"
        )
        if artifact_path is not None
        else "Model has no artifact/checkpoint path."
    ),
)


# ------------------------------------------------------------------
# File checks
# ------------------------------------------------------------------

artifact_info: dict[
    str,
    Any,
] | None = None


if artifact_path is not None:
    exists = artifact_path.exists()

    add_result(
        checks,
        name="artifact_exists",
        passed=exists,
        required=True,
        message=(
            "Artifact exists."
            if exists
            else "Artifact does not exist."
        ),
    )


    regular = (
        artifact_path.is_file()
        if exists
        else False
    )

    add_result(
        checks,
        name="artifact_regular_file",
        passed=regular,
        required=True,
        message=(
            "Artifact is a regular file."
            if regular
            else "Artifact is not a regular file."
        ),
    )


    if regular:
        size_bytes = (
            artifact_path.stat().
            st_size
        )

        size_valid = (
            size_bytes > 0
        )

        add_result(
            checks,
            name="artifact_nonempty",
            passed=size_valid,
            required=True,
            message=(
                (
                    "Artifact contains "
                    f"{size_bytes} bytes."
                )
                if size_valid
                else "Artifact is empty."
            ),
        )


        digest = hashlib.sha256()

        with artifact_path.open(
            "rb"
        ) as stream:
            while True:
                block = stream.read(
                    1024 * 1024
                )

                if not block:
                    break

                digest.update(
                    block
                )


        sha256 = digest.hexdigest()


        artifact_info = {
            "path": str(
                artifact_path
            ),

            "size_bytes": (
                size_bytes
            ),

            "sha256": (
                sha256
            ),
        }


# ------------------------------------------------------------------
# Checkpoint structural inspection
# ------------------------------------------------------------------

checkpoint_metadata = None


if (
    artifact_path is not None
    and artifact_path.is_file()
    and artifact_path.stat().
    st_size > 0
):
    try:
        # weights_only=False is intentional here:
        # this operates only on a locally registered
        # OSHA checkpoint and allows inspection of
        # training checkpoint metadata.
        checkpoint = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=False,
        )

        add_result(
            checks,
            name="checkpoint_readable",
            passed=True,
            required=True,
            message=(
                "Checkpoint is readable by PyTorch."
            ),
        )


        if isinstance(
            checkpoint,
            Mapping,
        ):
            checkpoint_keys = sorted(
                str(key)
                for key in checkpoint.keys()
            )

            state_keys = (
                "model_state_dict",
                "state_dict",
                "model",
            )

            state_present = any(
                key in checkpoint
                for key in state_keys
            )

            add_result(
                checks,
                name="model_state",
                passed=state_present,
                required=True,
                message=(
                    "Checkpoint contains model state."
                    if state_present
                    else (
                        "Checkpoint does not contain "
                        "a recognized model-state key."
                    )
                ),
                details={
                    "recognized_keys": (
                        state_keys
                    ),
                },
            )


            checkpoint_model_config = (
                checkpoint.get(
                    "model_config"
                )
            )

            add_result(
                checks,
                name="checkpoint_model_config",
                passed=(
                    checkpoint_model_config
                    is not None
                ),
                required=STRICT,
                message=(
                    "Checkpoint contains model_config."
                    if checkpoint_model_config
                    is not None
                    else (
                        "Checkpoint does not contain "
                        "model_config."
                    )
                ),
            )


            checkpoint_metadata = {
                "keys": (
                    checkpoint_keys
                ),

                "epoch": (
                    checkpoint.get(
                        "epoch"
                    )
                ),

                "step": (
                    checkpoint.get(
                        "step"
                    )
                ),

                "score": (
                    checkpoint.get(
                        "score"
                    )
                ),

                "score_name": (
                    checkpoint.get(
                        "score_name"
                    )
                ),

                "model_config": serialize(
                    checkpoint_model_config
                ),
            }


        else:
            add_result(
                checks,
                name="model_state",
                passed=False,
                required=True,
                message=(
                    "Checkpoint root object is not "
                    "a mapping."
                ),
            )


    except Exception as exc:
        add_result(
            checks,
            name="checkpoint_readable",
            passed=False,
            required=True,
            message=(
                "Checkpoint could not be read."
            ),
            details={
                "error_type": (
                    type(exc).__name__
                ),

                "error": str(
                    exc
                ),
            },
        )


# ------------------------------------------------------------------
# Registry-native validator
# ------------------------------------------------------------------

native_validation = None


for method_name in (
    "validate_model",
    "validate",
    "verify_model",
    "verify",
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

        parameters = (
            signature.parameters
        )

        if "model_id" in parameters:
            native_validation = method(
                model_id=MODEL_ID
            )

        elif "model" in parameters:
            native_validation = method(
                model=model
            )

        else:
            native_validation = method(
                MODEL_ID
            )


        normalized = serialize(
            native_validation
        )


        if isinstance(
            native_validation,
            bool,
        ):
            native_passed = (
                native_validation
            )

        else:
            native_passed = (
                value_of(
                    native_validation,
                    "valid",
                    "passed",
                    "success",
                    default=True,
                )
            )

            native_passed = bool(
                native_passed
            )


        add_result(
            checks,
            name="registry_native_validation",
            passed=native_passed,
            required=True,
            message=(
                "Registry-native validation passed."
                if native_passed
                else (
                    "Registry-native validation "
                    "reported failure."
                )
            ),
            details=normalized,
        )

        break


    except TypeError:
        continue

    except Exception as exc:
        add_result(
            checks,
            name="registry_native_validation",
            passed=False,
            required=True,
            message=(
                "Registry-native validation failed."
            ),
            details={
                "error_type": (
                    type(exc).__name__
                ),

                "error": str(exc),
            },
        )

        break


# ------------------------------------------------------------------
# Final result
# ------------------------------------------------------------------

required_failures = [
    check
    for check in checks
    if (
        check[
            "required"
        ]
        and not check[
            "passed"
        ]
    )
]


warnings = [
    check
    for check in checks
    if (
        not check[
            "required"
        ]
        and not check[
            "passed"
        ]
    )
]


valid = (
    len(
        required_failures
    )
    == 0
)


payload = {
    "schema_version": "1",

    "model_id": (
        str(model_id)
    ),

    "valid": valid,

    "strict": STRICT,

    "summary": {
        "check_count": len(
            checks
        ),

        "required_failures": len(
            required_failures
        ),

        "warnings": len(
            warnings
        ),
    },

    "artifact": (
        artifact_info
    ),

    "checkpoint": (
        checkpoint_metadata
    ),

    "checks": (
        checks
    ),
}


if VERBOSE:
    payload[
        "registry_metadata"
    ] = raw_metadata


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
        "Validation checks"
    )

    print(
        "-----------------"
    )


    for check in checks:
        if check[
            "passed"
        ]:
            status = "PASS"

        elif check[
            "required"
        ]:
            status = "FAIL"

        else:
            status = "WARN"


        print(
            (
                f"[{status}] "
                f"{check['name']}: "
                f"{check['message']}"
            )
        )


    if artifact_info:
        print()

        print(
            "Artifact"
        )

        print(
            "--------"
        )

        print(
            (
                "Path:   "
                f"{artifact_info['path']}"
            )
        )

        print(
            (
                "Size:   "
                f"{artifact_info['size_bytes']} bytes"
            )
        )

        print(
            (
                "SHA256: "
                f"{artifact_info['sha256']}"
            )
        )


    if (
        VERBOSE
        and checkpoint_metadata
    ):
        print()

        print(
            "Checkpoint metadata"
        )

        print(
            "-------------------"
        )

        print(
            json.dumps(
                checkpoint_metadata,
                indent=2,
                sort_keys=True,
                default=str,
            )
        )


    print()

    print(
        "Summary"
    )

    print(
        "-------"
    )

    print(
        (
            "Valid:             "
            f"{valid}"
        )
    )

    print(
        (
            "Checks:            "
            f"{len(checks)}"
        )
    )

    print(
        (
            "Required failures: "
            f"{len(required_failures)}"
        )
    )

    print(
        (
            "Warnings:          "
            f"{len(warnings)}"
        )
    )

    print(
        (
            "Strict:            "
            f"{STRICT}"
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


raise SystemExit(
    0
    if valid
    else 1
)
PY

VALIDATE_EXIT_CODE=$?

set -e


# -------------------------------------------------------------------
# Final result
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == true ]]; then
    exit "${VALIDATE_EXIT_CODE}"
fi


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Model Validation Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Model ID:  %s\n' \
    "${MODEL_ID}"

printf 'Python:    %s\n' \
    "${PYTHON_VERSION}"

printf 'Strict:    %s\n' \
    "${STRICT}"

printf 'Exit code: %s\n' \
    "${VALIDATE_EXIT_CODE}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


if ((VALIDATE_EXIT_CODE == 0)); then
    printf '\n%sMODEL VALIDATION PASSED%s\n' \
        "${GREEN}" \
        "${RESET}"

else
    printf '\n%sMODEL VALIDATION FAILED%s\n' \
        "${RED}" \
        "${RESET}"
fi


exit "${VALIDATE_EXIT_CODE}"
