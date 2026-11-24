#!/usr/bin/env bash

#
# OSHA accelerator inventory.
#
# Usage:
#
#   ./scripts/accelerators/inventory.sh
#   ./scripts/accelerators/inventory.sh --json
#   ./scripts/accelerators/inventory.sh --verbose
#   ./scripts/accelerators/inventory.sh --kind cpu
#   ./scripts/accelerators/inventory.sh --kind cuda
#   ./scripts/accelerators/inventory.sh --kind rocm
#   ./scripts/accelerators/inventory.sh --available
#   ./scripts/accelerators/inventory.sh --strict
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# The authoritative inventory is produced by:
#
#   src.accelerators.AcceleratorManager
#
# This script does not independently decide which
# accelerator should be used for a workload.
#
# It is read-only and does not:
#
#   reserve a device
#   select a training device
#   select an inference device
#   reset a GPU
#   modify clocks
#   clear caches
#   kill GPU processes
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
STRICT=false
AVAILABLE_ONLY=false

KIND=""

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
    if [[ "${JSON_OUTPUT}" == false ]]; then
        printf '%s[INFO]%s %s\n' \
            "${BLUE}" \
            "${RESET}" \
            "$*"
    fi
}


success() {
    if [[ "${JSON_OUTPUT}" == false ]]; then
        printf '%s[PASS]%s %s\n' \
            "${GREEN}" \
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
  scripts/accelerators/inventory.sh [options]

Options:
  --json
      Emit machine-readable JSON.

  --verbose
      Include additional normalized device fields.

  --kind KIND
      Filter by:
          cpu
          cuda
          rocm

  --available
      Show only available devices.

  --strict
      Fail when accelerator discovery reports backend failures.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override Python interpreter.

Examples:
  ./scripts/accelerators/inventory.sh

  ./scripts/accelerators/inventory.sh \
      --kind cuda

  ./scripts/accelerators/inventory.sh \
      --available

  ./scripts/accelerators/inventory.sh \
      --json

  ./scripts/accelerators/inventory.sh \
      --strict
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

        --available)
            AVAILABLE_ONLY=true
            ;;

        --strict)
            STRICT=true
            ;;

        --kind)
            if (($# < 2)); then
                printf \
                    'ERROR: --kind requires a value\n' \
                    >&2

                exit 2
            fi

            KIND="$2"

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
# Validate kind
# -------------------------------------------------------------------

case "${KIND}" in
    "")
        ;;

    cpu|cuda|rocm)
        ;;

    *)
        printf \
            'ERROR: unsupported accelerator kind: %s\n' \
            "${KIND}" >&2

        printf \
            'Expected one of: cpu, cuda, rocm\n' \
            >&2

        exit 2
        ;;
esac


# -------------------------------------------------------------------
# Project
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -d "${PROJECT_ROOT}/src/accelerators" ]]; then
    failure \
        "src/accelerators does not exist."

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


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


# -------------------------------------------------------------------
# Validate AcceleratorManager import
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'from src.accelerators import AcceleratorManager' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import src.accelerators.AcceleratorManager"

    exit 1
fi


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%sOSHA Accelerator Inventory%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Project root: ${PROJECT_ROOT}"
    info "Python: ${PYTHON}"
    info "Python version: ${PYTHON_VERSION}"

    if [[ -n "${KIND}" ]]; then
        info "Kind filter: ${KIND}"
    fi

    if [[ "${AVAILABLE_ONLY}" == true ]]; then
        info "Availability filter: available only"
    fi

    if [[ "${STRICT}" == true ]]; then
        info "Mode: strict"
    else
        info "Mode: normal"
    fi

    printf '\n'
fi


# -------------------------------------------------------------------
# Export script configuration
# -------------------------------------------------------------------

export OSHA_INVENTORY_JSON="$(
    if [[ "${JSON_OUTPUT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_INVENTORY_VERBOSE="$(
    if [[ "${VERBOSE}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_INVENTORY_AVAILABLE="$(
    if [[ "${AVAILABLE_ONLY}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_INVENTORY_STRICT="$(
    if [[ "${STRICT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_INVENTORY_KIND="${KIND}"


# -------------------------------------------------------------------
# Run authoritative inventory
# -------------------------------------------------------------------

set +e

"${PYTHON}" - <<'PY'
from __future__ import annotations

import json
import os
import sys

from dataclasses import (
    asdict,
    is_dataclass,
)

from enum import Enum

from typing import (
    Any,
    Mapping,
)


from src.accelerators import (
    AcceleratorManager,
)


# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

JSON_OUTPUT = (
    os.environ.get(
        "OSHA_INVENTORY_JSON",
        "0",
    )
    == "1"
)

VERBOSE = (
    os.environ.get(
        "OSHA_INVENTORY_VERBOSE",
        "0",
    )
    == "1"
)

AVAILABLE_ONLY = (
    os.environ.get(
        "OSHA_INVENTORY_AVAILABLE",
        "0",
    )
    == "1"
)

STRICT = (
    os.environ.get(
        "OSHA_INVENTORY_STRICT",
        "0",
    )
    == "1"
)

KIND_FILTER = (
    os.environ.get(
        "OSHA_INVENTORY_KIND",
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
            serialize(
                item
            )
            for item in value
        ]

    if is_dataclass(
        value
    ):
        return serialize(
            asdict(
                value
            )
        )

    converter = getattr(
        value,
        "to_dict",
        None,
    )

    if callable(
        converter
    ):
        try:
            converted = converter()

            return serialize(
                converted
            )

        except Exception:
            pass

    return str(
        value
    )


# ------------------------------------------------------------------
# Attribute helpers
# ------------------------------------------------------------------

def get_value(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(
        value,
        Mapping,
    ):
        return value.get(
            name,
            default,
        )

    return getattr(
        value,
        name,
        default,
    )


def enum_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        Enum,
    ):
        return str(
            value.value
        )

    return str(
        value
    )


def kind_of(
    device: Any,
) -> str:
    return enum_text(
        get_value(
            device,
            "kind",
            "",
        )
    ).lower()


def available_of(
    device: Any,
) -> bool:
    value = get_value(
        device,
        "available",
        None,
    )

    if value is not None:
        return bool(
            value
        )

    state = enum_text(
        get_value(
            device,
            "state",
            "",
        )
    ).lower()

    return state in {
        "available",
        "healthy",
        "ready",
    }


def accelerator_of(
    device: Any,
) -> bool:
    value = get_value(
        device,
        "accelerator",
        None,
    )

    if value is not None:
        return bool(
            value
        )

    return kind_of(
        device
    ) in {
        "cuda",
        "rocm",
    }


def memory_of(
    device: Any,
) -> dict[str, Any]:
    memory = get_value(
        device,
        "memory",
        None,
    )

    if memory is None:
        return {}

    mapped = serialize(
        memory
    )

    if isinstance(
        mapped,
        dict,
    ):
        return mapped

    return {}


def bytes_human(
    value: Any,
) -> str:
    if value is None:
        return "-"

    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return "-"

    units = (
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    )

    index = 0

    while (
        number >= 1024.0
        and index
        < len(units) - 1
    ):
        number /= 1024.0
        index += 1

    if index == 0:
        return (
            f"{int(number)} "
            f"{units[index]}"
        )

    return (
        f"{number:.2f} "
        f"{units[index]}"
    )


# ------------------------------------------------------------------
# Failure normalization
# ------------------------------------------------------------------

def normalize_failure(
    failure: Any,
) -> dict[str, Any]:
    data = serialize(
        failure
    )

    if not isinstance(
        data,
        dict,
    ):
        return {
            "backend": None,
            "error_type": (
                type(
                    failure
                ).__name__
            ),
            "message": str(
                failure
            ),
        }

    return {
        "backend": data.get(
            "backend"
        ),

        "error_type": data.get(
            "error_type"
        ),

        "message": data.get(
            "message"
        ),
    }


# ------------------------------------------------------------------
# Device normalization
# ------------------------------------------------------------------

def normalize_device(
    device: Any,
) -> dict[str, Any]:
    memory = memory_of(
        device
    )

    capabilities = serialize(
        get_value(
            device,
            "capabilities",
            (),
        )
    )

    supported_precisions = serialize(
        get_value(
            device,
            "supported_precisions",
            (),
        )
    )

    data = {
        "device_id": get_value(
            device,
            "device_id",
            None,
        ),

        "kind": kind_of(
            device
        ),

        "runtime_device": get_value(
            device,
            "runtime_device",
            None,
        ),

        "index": get_value(
            device,
            "index",
            None,
        ),

        "name": get_value(
            device,
            "name",
            None,
        ),

        "vendor": get_value(
            device,
            "vendor",
            None,
        ),

        "state": enum_text(
            get_value(
                device,
                "state",
                "unknown",
            )
        ),

        "available": available_of(
            device
        ),

        "accelerator": accelerator_of(
            device
        ),

        "architecture": get_value(
            device,
            "architecture",
            None,
        ),

        "compute_capability": get_value(
            device,
            "compute_capability",
            None,
        ),

        "runtime_version": get_value(
            device,
            "runtime_version",
            None,
        ),

        "memory": memory,

        "capabilities": (
            capabilities
            if isinstance(
                capabilities,
                list,
            )
            else []
        ),

        "supported_precisions": (
            supported_precisions
            if isinstance(
                supported_precisions,
                list,
            )
            else []
        ),
    }

    if VERBOSE:
        attributes = serialize(
            get_value(
                device,
                "attributes",
                {},
            )
        )

        if isinstance(
            attributes,
            dict,
        ):
            data[
                "attributes"
            ] = attributes

    return data


# ------------------------------------------------------------------
# Inventory
# ------------------------------------------------------------------

manager = AcceleratorManager()


try:
    inventory = manager.inventory(
        refresh=True
    )

    source_devices = tuple(
        get_value(
            inventory,
            "devices",
            (),
        )
        or ()
    )

    failures = [
        normalize_failure(
            item
        )
        for item in (
            get_value(
                inventory,
                "failures",
                (),
            )
            or ()
        )
    ]

    devices = [
        normalize_device(
            device
        )
        for device in source_devices
    ]

    if KIND_FILTER:
        devices = [
            device
            for device in devices
            if device[
                "kind"
            ] == KIND_FILTER
        ]

    if AVAILABLE_ONLY:
        devices = [
            device
            for device in devices
            if device[
                "available"
            ]
        ]

    available_count = sum(
        1
        for device in devices
        if device[
            "available"
        ]
    )

    accelerator_count = sum(
        1
        for device in devices
        if (
            device[
                "available"
            ]
            and device[
                "accelerator"
            ]
        )
    )

    kind_counts: dict[
        str,
        int,
    ] = {}

    for device in devices:
        kind = (
            device[
                "kind"
            ]
            or "unknown"
        )

        kind_counts[
            kind
        ] = (
            kind_counts.get(
                kind,
                0,
            )
            + 1
        )

    payload = {
        "schema_version": "1",

        "filters": {
            "kind": (
                KIND_FILTER
                or None
            ),

            "available_only": (
                AVAILABLE_ONLY
            ),
        },

        "summary": {
            "device_count": len(
                devices
            ),

            "available_count": (
                available_count
            ),

            "accelerator_count": (
                accelerator_count
            ),

            "backend_failure_count": len(
                failures
            ),

            "kinds": kind_counts,
        },

        "devices": devices,

        "failures": failures,
    }


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
        print(
            "Summary"
        )

        print(
            "-------"
        )

        print(
            (
                "Devices:             "
                f"{len(devices)}"
            )
        )

        print(
            (
                "Available:           "
                f"{available_count}"
            )
        )

        print(
            (
                "Accelerator devices: "
                f"{accelerator_count}"
            )
        )

        print(
            (
                "Backend failures:    "
                f"{len(failures)}"
            )
        )

        if kind_counts:
            print(
                "Kinds:"
            )

            for kind in sorted(
                kind_counts
            ):
                print(
                    (
                        f"  {kind:<10} "
                        f"{kind_counts[kind]}"
                    )
                )

        print()

        print(
            "Devices"
        )

        print(
            "-------"
        )

        if not devices:
            print(
                "No devices matched "
                "the selected filters."
            )

        for number, device in enumerate(
            devices,
            start=1,
        ):
            memory = device[
                "memory"
            ]

            total_memory = (
                memory.get(
                    "total_bytes"
                )
            )

            free_memory = (
                memory.get(
                    "free_bytes"
                )
            )

            used_memory = (
                memory.get(
                    "used_bytes"
                )
            )

            print(
                (
                    f"[{number}] "
                    f"{device['device_id']}"
                )
            )

            print(
                (
                    "    Kind:           "
                    f"{device['kind']}"
                )
            )

            print(
                (
                    "    Name:           "
                    f"{device['name']}"
                )
            )

            print(
                (
                    "    Runtime device: "
                    f"{device['runtime_device']}"
                )
            )

            print(
                (
                    "    State:          "
                    f"{device['state']}"
                )
            )

            print(
                (
                    "    Available:      "
                    f"{device['available']}"
                )
            )

            print(
                (
                    "    Accelerator:    "
                    f"{device['accelerator']}"
                )
            )

            if device[
                "vendor"
            ]:
                print(
                    (
                        "    Vendor:         "
                        f"{device['vendor']}"
                    )
                )

            if device[
                "architecture"
            ]:
                print(
                    (
                        "    Architecture:   "
                        f"{device['architecture']}"
                    )
                )

            if device[
                "compute_capability"
            ]:
                print(
                    (
                        "    Compute:        "
                        f"{device['compute_capability']}"
                    )
                )

            if device[
                "runtime_version"
            ]:
                print(
                    (
                        "    Runtime:        "
                        f"{device['runtime_version']}"
                    )
                )

            if total_memory is not None:
                print(
                    (
                        "    Memory total:   "
                        f"{bytes_human(total_memory)}"
                    )
                )

            if free_memory is not None:
                print(
                    (
                        "    Memory free:    "
                        f"{bytes_human(free_memory)}"
                    )
                )

            if used_memory is not None:
                print(
                    (
                        "    Memory used:    "
                        f"{bytes_human(used_memory)}"
                    )
                )

            if device[
                "supported_precisions"
            ]:
                precisions = ", ".join(
                    str(value)
                    for value in device[
                        "supported_precisions"
                    ]
                )

                print(
                    (
                        "    Precisions:     "
                        f"{precisions}"
                    )
                )

            if device[
                "capabilities"
            ]:
                capabilities = ", ".join(
                    str(value)
                    for value in device[
                        "capabilities"
                    ]
                )

                print(
                    (
                        "    Capabilities:   "
                        f"{capabilities}"
                    )
                )

            if (
                VERBOSE
                and device.get(
                    "attributes"
                )
            ):
                print(
                    "    Attributes:"
                )

                formatted = json.dumps(
                    device[
                        "attributes"
                    ],
                    indent=2,
                    sort_keys=True,
                    default=str,
                )

                for line in (
                    formatted.splitlines()
                ):
                    print(
                        "        " + line
                    )

            print()

        if failures:
            print(
                "Backend discovery failures"
            )

            print(
                "--------------------------"
            )

            for failure in failures:
                print(
                    (
                        "Backend: "
                        f"{failure.get('backend')}"
                    )
                )

                print(
                    (
                        "Type:    "
                        f"{failure.get('error_type')}"
                    )
                )

                print(
                    (
                        "Message: "
                        f"{failure.get('message')}"
                    )
                )

                print()

    if (
        STRICT
        and failures
    ):
        raise SystemExit(1)

    raise SystemExit(0)

finally:
    release_all = getattr(
        manager,
        "release_all",
        None,
    )

    if callable(
        release_all
    ):
        try:
            release_all()

        except Exception:
            pass
PY

INVENTORY_EXIT_CODE=$?

set -e


# -------------------------------------------------------------------
# Result
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


if [[ "${JSON_OUTPUT}" == true ]]; then
    exit "${INVENTORY_EXIT_CODE}"
fi


printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Accelerator Inventory Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Python:   %s\n' \
    "${PYTHON_VERSION}"

printf 'Filter:   %s\n' \
    "$(
        if [[ -n "${KIND}" ]]; then
            printf '%s' "${KIND}"
        else
            printf 'all'
        fi
    )"

printf 'Available only: %s\n' \
    "${AVAILABLE_ONLY}"

printf 'Strict:         %s\n' \
    "${STRICT}"

printf 'Duration:       %ss\n' \
    "${elapsed}"

printf 'Exit code:      %s\n' \
    "${INVENTORY_EXIT_CODE}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


if ((INVENTORY_EXIT_CODE == 0)); then
    printf '\n%sINVENTORY COMPLETE%s\n' \
        "${GREEN}" \
        "${RESET}"

else
    printf '\n%sINVENTORY FAILED%s\n' \
        "${RED}" \
        "${RESET}"
fi


exit "${INVENTORY_EXIT_CODE}"
