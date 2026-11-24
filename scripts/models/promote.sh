#!/usr/bin/env bash

#
# OSHA guarded model promotion.
#
# Usage:
#
#   Dry run:
#
#       ./scripts/models/promote.sh MODEL_ID
#
#   Apply:
#
#       ./scripts/models/promote.sh MODEL_ID \
#           --apply \
#           --confirm MODEL_ID
#
#   JSON:
#
#       ./scripts/models/promote.sh MODEL_ID \
#           --json
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# Safety:
#
#   Promotion is never applied by default.
#
#   A real promotion requires:
#
#       --apply
#       --confirm MODEL_ID
#
#   The confirmation value must exactly match the
#   requested model ID.
#
# This script does not copy checkpoints manually.
# Registry mutation remains owned by src.ai.registry.
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

VALIDATE_SCRIPT="${PROJECT_ROOT}/scripts/models/validate.sh"


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

MODEL_ID=""
CONFIRMATION=""
REASON=""

APPLY=false
JSON_OUTPUT=false
VERBOSE=false
SKIP_VALIDATION=false


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


warn() {
    if [[ "${JSON_OUTPUT}" == false ]]; then
        printf '%s[WARN]%s %s\n' \
            "${YELLOW}" \
            "${RESET}" \
            "$*" >&2
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
  scripts/models/promote.sh MODEL_ID [options]

Options:
  --apply
      Apply the promotion.

      Without this option, the command performs a
      dry run and makes no registry changes.

  --confirm MODEL_ID
      Required with --apply.

      The confirmation value must exactly match the
      positional MODEL_ID.

  --reason TEXT
      Record an operator-supplied promotion reason
      when supported by the registry API.

  --skip-validation
      Skip validate.sh.

      This option is intentionally rejected when
      --apply is used. It is only available for
      diagnostic dry runs.

  --json
      Emit machine-readable JSON.

  --verbose
      Include additional registry metadata.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override Python interpreter.

Examples:
  Dry run:

      ./scripts/models/promote.sh MODEL_ID

  Dry run with JSON:

      ./scripts/models/promote.sh MODEL_ID --json

  Apply promotion:

      ./scripts/models/promote.sh MODEL_ID \
          --apply \
          --confirm MODEL_ID \
          --reason "Passed validation and evaluation"

Important:
  --apply without an exact --confirm value fails.
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
        --apply)
            APPLY=true
            ;;

        --confirm)
            if (($# < 2)); then
                printf \
                    'ERROR: --confirm requires MODEL_ID\n' \
                    >&2

                exit 2
            fi

            CONFIRMATION="$2"
            shift
            ;;

        --reason)
            if (($# < 2)); then
                printf \
                    'ERROR: --reason requires text\n' \
                    >&2

                exit 2
            fi

            REASON="$2"
            shift
            ;;

        --skip-validation)
            SKIP_VALIDATION=true
            ;;

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
# Input validation
# -------------------------------------------------------------------

if [[ -z "${MODEL_ID}" ]]; then
    printf \
        'ERROR: MODEL_ID is required.\n\n' \
        >&2

    usage >&2
    exit 2
fi


if [[ "${APPLY}" == true ]] \
    && [[ "${SKIP_VALIDATION}" == true ]]; then
    printf \
        'ERROR: --skip-validation cannot be used with --apply.\n' \
        >&2

    exit 2
fi


if [[ "${APPLY}" == true ]]; then
    if [[ -z "${CONFIRMATION}" ]]; then
        printf \
            'ERROR: --apply requires --confirm MODEL_ID.\n' \
            >&2

        exit 2
    fi

    if [[ "${CONFIRMATION}" != "${MODEL_ID}" ]]; then
        printf \
            'ERROR: confirmation does not match MODEL_ID.\n' \
            >&2

        printf \
            'Expected: %s\nReceived: %s\n' \
            "${MODEL_ID}" \
            "${CONFIRMATION}" >&2

        exit 2
    fi
fi


# -------------------------------------------------------------------
# Project
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -f "${PROJECT_ROOT}/src/ai/registry.py" ]]; then
    failure \
        "src/ai/registry.py does not exist."

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
# Registry import
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import src.ai.registry' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.ai.registry."

    exit 1
fi


# -------------------------------------------------------------------
# Validation precondition
# -------------------------------------------------------------------

VALIDATION_STATUS="skipped"


if [[ "${SKIP_VALIDATION}" == false ]]; then
    if [[ ! -x "${VALIDATE_SCRIPT}" ]]; then
        failure \
            "Model validator is missing or not executable: ${VALIDATE_SCRIPT}"

        exit 2
    fi

    if [[ "${JSON_OUTPUT}" == false ]]; then
        info \
            "Validating model before promotion"
    fi

    set +e

    OSHA_PYTHON="${PYTHON}" \
        "${VALIDATE_SCRIPT}" \
        "${MODEL_ID}" \
        >/tmp/osha_model_promotion_validation.out \
        2>/tmp/osha_model_promotion_validation.err

    VALIDATION_EXIT=$?

    set -e


    if ((VALIDATION_EXIT != 0)); then
        if [[ "${JSON_OUTPUT}" == true ]]; then
            OSHA_PROMOTION_MODEL_ID="${MODEL_ID}" \
            OSHA_PROMOTION_VALIDATION_EXIT="${VALIDATION_EXIT}" \
            "${PYTHON}" - <<'PY'
from __future__ import annotations

import json
import os


print(
    json.dumps(
        {
            "schema_version": "1",
            "model_id": os.environ[
                "OSHA_PROMOTION_MODEL_ID"
            ],
            "success": False,
            "applied": False,
            "validation": {
                "passed": False,
                "exit_code": int(
                    os.environ[
                        "OSHA_PROMOTION_VALIDATION_EXIT"
                    ]
                ),
            },
            "error": {
                "code": (
                    "model_validation_failed"
                ),
                "message": (
                    "Model validation failed; "
                    "promotion was not attempted."
                ),
            },
        },
        indent=2,
        sort_keys=True,
    )
)
PY

        else
            cat \
                /tmp/osha_model_promotion_validation.out \
                || true

            cat \
                /tmp/osha_model_promotion_validation.err \
                >&2 \
                || true

            failure \
                "Model validation failed; promotion was not attempted."
        fi

        rm -f \
            /tmp/osha_model_promotion_validation.out \
            /tmp/osha_model_promotion_validation.err

        exit 1
    fi


    VALIDATION_STATUS="passed"

    if [[ "${JSON_OUTPUT}" == false ]]; then
        cat \
            /tmp/osha_model_promotion_validation.out \
            || true

        success_message="Model validation passed"

        printf '%s[PASS]%s %s\n' \
            "${GREEN}" \
            "${RESET}" \
            "${success_message}"
    fi


    rm -f \
        /tmp/osha_model_promotion_validation.out \
        /tmp/osha_model_promotion_validation.err
fi


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%sOSHA Guarded Model Promotion%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Model ID: ${MODEL_ID}"
    info "Python: ${PYTHON_VERSION}"
    info "Validation: ${VALIDATION_STATUS}"

    if [[ "${APPLY}" == true ]]; then
        info "Mode: apply"
    else
        info "Mode: dry-run"
    fi

    if [[ -n "${REASON}" ]]; then
        info "Reason: ${REASON}"
    fi
fi


# -------------------------------------------------------------------
# Configuration for registry operation
# -------------------------------------------------------------------

export OSHA_PROMOTE_MODEL_ID="${MODEL_ID}"

export OSHA_PROMOTE_REASON="${REASON}"

export OSHA_PROMOTE_APPLY="$(
    if [[ "${APPLY}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_PROMOTE_JSON="$(
    if [[ "${JSON_OUTPUT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_PROMOTE_VERBOSE="$(
    if [[ "${VERBOSE}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_PROMOTE_VALIDATION="${VALIDATION_STATUS}"


# -------------------------------------------------------------------
# Registry operation
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
    "OSHA_PROMOTE_MODEL_ID"
]

REASON = os.environ.get(
    "OSHA_PROMOTE_REASON",
    "",
).strip()

APPLY = (
    os.environ.get(
        "OSHA_PROMOTE_APPLY",
        "0",
    )
    == "1"
)

JSON_OUTPUT = (
    os.environ.get(
        "OSHA_PROMOTE_JSON",
        "0",
    )
    == "1"
)

VERBOSE = (
    os.environ.get(
        "OSHA_PROMOTE_VERBOSE",
        "0",
    )
    == "1"
)

VALIDATION = os.environ.get(
    "OSHA_PROMOTE_VALIDATION",
    "unknown",
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
            str(key): serialize(
                item
            )
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


def invoke_lookup(
    registry: Any,
    model_id: str,
) -> tuple[
    Any,
    str | None,
]:
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
                result = method(
                    model_id=model_id
                )

            elif "identifier" in parameters:
                result = method(
                    identifier=model_id
                )

            elif "name" in parameters:
                result = method(
                    name=model_id
                )

            else:
                result = method(
                    model_id
                )

            if result is not None:
                return (
                    result,
                    method_name,
                )

        except (
            KeyError,
            LookupError,
            ValueError,
            TypeError,
        ):
            continue

    return (
        None,
        None,
    )


# ------------------------------------------------------------------
# Registry class
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

    if inspect.isclass(
        candidate
    ):
        registry_class = candidate
        break


if registry_class is None:
    raise RuntimeError(
        "No supported registry class was found."
    )


try:
    registry = registry_class()

except TypeError as exc:
    raise RuntimeError(
        "Registry requires explicit constructor "
        f"parameters: {inspect.signature(registry_class)}"
    ) from exc


# ------------------------------------------------------------------
# Start/discover
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
# Lookup and current state
# ------------------------------------------------------------------

model, lookup_method = invoke_lookup(
    registry,
    MODEL_ID,
)


if model is None:
    payload = {
        "schema_version": "1",
        "success": False,
        "applied": False,
        "model_id": MODEL_ID,
        "validation": VALIDATION,
        "error": {
            "code": "model_not_found",
            "message": (
                "Model was not found in registry."
            ),
        },
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


current_state = serialize(
    model
)


current_promoted = value_of(
    model,
    "promoted",
    "is_promoted",
    default=None,
)

current_status = value_of(
    model,
    "state",
    "status",
    default=None,
)


# ------------------------------------------------------------------
# Discover promotion API
# ------------------------------------------------------------------

promotion_method = None
promotion_method_name = None
promotion_signature = None


for method_name in (
    "promote_model",
    "promote",
    "set_promoted",
    "activate_model",
    "activate",
):
    candidate = getattr(
        registry,
        method_name,
        None,
    )

    if callable(candidate):
        promotion_method = candidate
        promotion_method_name = (
            method_name
        )

        promotion_signature = (
            inspect.signature(
                candidate
            )
        )

        break


if promotion_method is None:
    public_methods = [
        name
        for name in dir(
            registry
        )
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

    payload = {
        "schema_version": "1",
        "success": False,
        "applied": False,
        "model_id": MODEL_ID,
        "validation": VALIDATION,
        "error": {
            "code": (
                "promotion_api_unavailable"
            ),
            "message": (
                "Registry exposes no supported "
                "promotion method."
            ),
            "public_methods": (
                public_methods
            ),
        },
    }

    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        if JSON_OUTPUT
        else (
            "Registry exposes no supported "
            "promotion method.\nPublic methods: "
            + ", ".join(
                public_methods
            )
        )
    )

    raise SystemExit(2)


# ------------------------------------------------------------------
# Build supported method arguments
# ------------------------------------------------------------------

parameters = (
    promotion_signature.parameters
)


arguments: dict[
    str,
    Any,
] = {}


if "model_id" in parameters:
    arguments[
        "model_id"
    ] = MODEL_ID

elif "identifier" in parameters:
    arguments[
        "identifier"
    ] = MODEL_ID

elif "name" in parameters:
    arguments[
        "name"
    ] = MODEL_ID

elif "model" in parameters:
    arguments[
        "model"
    ] = model


if "reason" in parameters:
    arguments[
        "reason"
    ] = (
        REASON
        or None
    )


if "promoted" in parameters:
    arguments[
        "promoted"
    ] = True


if "active" in parameters:
    arguments[
        "active"
    ] = True


if "enabled" in parameters:
    arguments[
        "enabled"
    ] = True


if "validate" in parameters:
    arguments[
        "validate"
    ] = True


if "dry_run" in parameters:
    arguments[
        "dry_run"
    ] = (
        not APPLY
    )


# Determine whether the model must be supplied
# positionally.
positional_model_id = (
    not any(
        name in parameters
        for name in (
            "model_id",
            "identifier",
            "name",
            "model",
        )
    )
)


# ------------------------------------------------------------------
# Dry run
# ------------------------------------------------------------------

if not APPLY:
    payload = {
        "schema_version": "1",

        "success": True,

        "applied": False,

        "mode": "dry-run",

        "model_id": MODEL_ID,

        "validation": VALIDATION,

        "registry": (
            registry_class.__name__
        ),

        "lookup_method": (
            lookup_method
        ),

        "promotion_method": (
            promotion_method_name
        ),

        "promotion_signature": str(
            promotion_signature
        ),

        "current": {
            "promoted": (
                current_promoted
            ),

            "state": serialize(
                current_status
            ),
        },

        "planned_arguments": (
            serialize(
                arguments
            )
        ),

        "reason": (
            REASON
            or None
        ),
    }


    if VERBOSE:
        payload[
            "registry_metadata"
        ] = current_state


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
            "Promotion dry run"
        )

        print(
            "-----------------"
        )

        print(
            (
                "Model ID:         "
                f"{MODEL_ID}"
            )
        )

        print(
            (
                "Registry:         "
                f"{registry_class.__name__}"
            )
        )

        print(
            (
                "Validation:       "
                f"{VALIDATION}"
            )
        )

        print(
            (
                "Current promoted: "
                f"{current_promoted}"
            )
        )

        print(
            (
                "Current state:    "
                f"{current_status}"
            )
        )

        print(
            (
                "Promotion method: "
                f"{promotion_method_name}"
            )
        )

        print(
            (
                "Method signature: "
                f"{promotion_signature}"
            )
        )

        print(
            (
                "Reason:           "
                f"{REASON or None}"
            )
        )

        print()

        print(
            "No registry state was changed."
        )

        print()

        print(
            "To apply:"
        )

        print(
            (
                "  ./scripts/models/promote.sh "
                f"{MODEL_ID} "
                "--apply "
                f"--confirm {MODEL_ID}"
            )
        )


    raise SystemExit(0)


# ------------------------------------------------------------------
# Apply promotion
# ------------------------------------------------------------------

if positional_model_id:
    result = promotion_method(
        MODEL_ID,
        **arguments,
    )

else:
    result = promotion_method(
        **arguments
    )


# Refresh registry after mutation if supported.
for method_name in (
    "refresh",
    "discover",
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


updated_model, _ = invoke_lookup(
    registry,
    MODEL_ID,
)


updated_state = (
    serialize(
        updated_model
    )
    if updated_model is not None
    else None
)


updated_promoted = (
    value_of(
        updated_model,
        "promoted",
        "is_promoted",
        default=None,
    )
    if updated_model is not None
    else None
)


# A result of False is an explicit failure.
if isinstance(
    result,
    bool,
) and not result:
    raise RuntimeError(
        "Registry promotion method returned False."
    )


payload = {
    "schema_version": "1",

    "success": True,

    "applied": True,

    "mode": "apply",

    "model_id": MODEL_ID,

    "validation": VALIDATION,

    "registry": (
        registry_class.__name__
    ),

    "promotion_method": (
        promotion_method_name
    ),

    "reason": (
        REASON
        or None
    ),

    "result": serialize(
        result
    ),

    "before": {
        "promoted": (
            current_promoted
        ),

        "state": serialize(
            current_status
        ),
    },

    "after": {
        "promoted": (
            updated_promoted
        ),
    },
}


if VERBOSE:
    payload[
        "updated_registry_metadata"
    ] = updated_state


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
        "Promotion applied"
    )

    print(
        "-----------------"
    )

    print(
        (
            "Model ID:         "
            f"{MODEL_ID}"
        )
    )

    print(
        (
            "Registry:         "
            f"{registry_class.__name__}"
        )
    )

    print(
        (
            "Promotion method: "
            f"{promotion_method_name}"
        )
    )

    print(
        (
            "Before promoted:  "
            f"{current_promoted}"
        )
    )

    print(
        (
            "After promoted:   "
            f"{updated_promoted}"
        )
    )

    print(
        (
            "Reason:           "
            f"{REASON or None}"
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

PROMOTE_EXIT_CODE=$?

set -e


# -------------------------------------------------------------------
# Final result
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == true ]]; then
    exit "${PROMOTE_EXIT_CODE}"
fi


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Model Promotion Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Model ID:   %s\n' \
    "${MODEL_ID}"

printf 'Mode:       %s\n' \
    "$(
        if [[ "${APPLY}" == true ]]; then
            printf 'apply'
        else
            printf 'dry-run'
        fi
    )"

printf 'Validation: %s\n' \
    "${VALIDATION_STATUS}"

printf 'Python:     %s\n' \
    "${PYTHON_VERSION}"

printf 'Exit code:  %s\n' \
    "${PROMOTE_EXIT_CODE}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


if ((PROMOTE_EXIT_CODE == 0)); then
    if [[ "${APPLY}" == true ]]; then
        printf '\n%sMODEL PROMOTION APPLIED%s\n' \
            "${GREEN}" \
            "${RESET}"
    else
        printf '\n%sMODEL PROMOTION DRY RUN COMPLETE%s\n' \
            "${GREEN}" \
            "${RESET}"
    fi

else
    printf '\n%sMODEL PROMOTION FAILED%s\n' \
        "${RED}" \
        "${RESET}"
fi


exit "${PROMOTE_EXIT_CODE}"
