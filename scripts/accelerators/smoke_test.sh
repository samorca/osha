#!/usr/bin/env bash

#
# OSHA accelerator smoke test.
#
# Usage:
#
#   ./scripts/accelerators/smoke_test.sh
#   ./scripts/accelerators/smoke_test.sh --kind cpu
#   ./scripts/accelerators/smoke_test.sh --kind cuda
#   ./scripts/accelerators/smoke_test.sh --kind rocm
#   ./scripts/accelerators/smoke_test.sh --kind auto
#   ./scripts/accelerators/smoke_test.sh --device 0
#   ./scripts/accelerators/smoke_test.sh --training
#   ./scripts/accelerators/smoke_test.sh --inference
#   ./scripts/accelerators/smoke_test.sh --mixed-precision
#   ./scripts/accelerators/smoke_test.sh --no-cpu-fallback
#   ./scripts/accelerators/smoke_test.sh --json
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This script:
#
#   discovers accelerator inventory
#   selects a device through AcceleratorManager
#   resolves the PyTorch runtime device
#   performs a small deterministic computation
#   verifies the result
#   releases the selection in a finally block
#
# It does not:
#
#   benchmark hardware
#   reserve large accelerator memory
#   train a model
#   reset a GPU
#   change clocks or power limits
#   clear accelerator caches
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

KIND="auto"
DEVICE_INDEX=""
WORKLOAD="inference"

MIXED_PRECISION=false
ALLOW_CPU_FALLBACK=true
JSON_OUTPUT=false
VERBOSE=false
STRICT=false

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
  scripts/accelerators/smoke_test.sh [options]

Options:
  --kind KIND
      Select:
          auto
          cpu
          cuda
          rocm

      Default: auto

  --device INDEX
      Request a specific accelerator device index.

  --training
      Use the training-selection path.

  --inference
      Use the inference-selection path.

      This is the default.

  --mixed-precision
      Request mixed-precision capability.

  --no-cpu-fallback
      Do not fall back to CPU when the requested
      accelerator is unavailable.

  --json
      Emit machine-readable JSON.

  --verbose
      Include inventory and tensor details.

  --strict
      Require the selected backend to match the
      requested non-auto backend exactly.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override the Python interpreter.

Examples:
  ./scripts/accelerators/smoke_test.sh

  ./scripts/accelerators/smoke_test.sh \
      --kind cpu

  ./scripts/accelerators/smoke_test.sh \
      --kind cuda \
      --device 0 \
      --no-cpu-fallback

  ./scripts/accelerators/smoke_test.sh \
      --kind rocm \
      --training \
      --no-cpu-fallback

  ./scripts/accelerators/smoke_test.sh \
      --kind auto \
      --mixed-precision

  ./scripts/accelerators/smoke_test.sh \
      --json
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
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

        --device)
            if (($# < 2)); then
                printf \
                    'ERROR: --device requires an index\n' \
                    >&2

                exit 2
            fi

            DEVICE_INDEX="$2"
            shift
            ;;

        --training)
            WORKLOAD="training"
            ;;

        --inference)
            WORKLOAD="inference"
            ;;

        --mixed-precision)
            MIXED_PRECISION=true
            ;;

        --no-cpu-fallback)
            ALLOW_CPU_FALLBACK=false
            ;;

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

        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac

    shift
done


# -------------------------------------------------------------------
# Validate arguments
# -------------------------------------------------------------------

case "${KIND}" in
    auto|cpu|cuda|rocm)
        ;;

    *)
        printf \
            'ERROR: unsupported accelerator kind: %s\n' \
            "${KIND}" >&2

        printf \
            'Expected one of: auto, cpu, cuda, rocm\n' \
            >&2

        exit 2
        ;;
esac


if [[ -n "${DEVICE_INDEX}" ]] \
    && [[ ! "${DEVICE_INDEX}" =~ ^[0-9]+$ ]]
then
    printf \
        'ERROR: --device must be a non-negative integer\n' \
        >&2

    exit 2
fi


if [[ "${MIXED_PRECISION}" == true ]] \
    && [[ "${KIND}" == "cpu" ]]; then
    printf \
        'ERROR: mixed precision cannot be requested with an explicit CPU selection\n' \
        >&2

    exit 2
fi


# -------------------------------------------------------------------
# Project validation
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
# Import validation
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'import torch; from src.accelerators import AcceleratorManager' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import PyTorch and/or AcceleratorManager"

    exit 1
fi


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%sOSHA Accelerator Smoke Test%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Project root: ${PROJECT_ROOT}"
    info "Python: ${PYTHON}"
    info "Python version: ${PYTHON_VERSION}"
    info "Requested kind: ${KIND}"
    info "Workload: ${WORKLOAD}"
    info "Mixed precision: ${MIXED_PRECISION}"
    info "CPU fallback: ${ALLOW_CPU_FALLBACK}"
    info "Strict: ${STRICT}"

    if [[ -n "${DEVICE_INDEX}" ]]; then
        info "Requested device index: ${DEVICE_INDEX}"
    fi
fi


# -------------------------------------------------------------------
# Export test configuration
# -------------------------------------------------------------------

export OSHA_SMOKE_KIND="${KIND}"

export OSHA_SMOKE_DEVICE_INDEX="${DEVICE_INDEX}"

export OSHA_SMOKE_WORKLOAD="${WORKLOAD}"

export OSHA_SMOKE_MIXED_PRECISION="$(
    if [[ "${MIXED_PRECISION}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_SMOKE_CPU_FALLBACK="$(
    if [[ "${ALLOW_CPU_FALLBACK}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_SMOKE_JSON="$(
    if [[ "${JSON_OUTPUT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_SMOKE_VERBOSE="$(
    if [[ "${VERBOSE}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_SMOKE_STRICT="$(
    if [[ "${STRICT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"


# -------------------------------------------------------------------
# Run smoke test
# -------------------------------------------------------------------

set +e

"${PYTHON}" - <<'PY'
from __future__ import annotations

import inspect
import json
import os
import time

from dataclasses import (
    asdict,
    is_dataclass,
)

from enum import Enum

from typing import (
    Any,
    Callable,
    Mapping,
)

import torch

from src.accelerators import (
    AcceleratorManager,
)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

REQUESTED_KIND = os.environ.get(
    "OSHA_SMOKE_KIND",
    "auto",
).strip().lower()

DEVICE_INDEX_TEXT = os.environ.get(
    "OSHA_SMOKE_DEVICE_INDEX",
    "",
).strip()

DEVICE_INDEX = (
    int(DEVICE_INDEX_TEXT)
    if DEVICE_INDEX_TEXT
    else None
)

WORKLOAD = os.environ.get(
    "OSHA_SMOKE_WORKLOAD",
    "inference",
).strip().lower()

MIXED_PRECISION = (
    os.environ.get(
        "OSHA_SMOKE_MIXED_PRECISION",
        "0",
    )
    == "1"
)

ALLOW_CPU_FALLBACK = (
    os.environ.get(
        "OSHA_SMOKE_CPU_FALLBACK",
        "1",
    )
    == "1"
)

JSON_OUTPUT = (
    os.environ.get(
        "OSHA_SMOKE_JSON",
        "0",
    )
    == "1"
)

VERBOSE = (
    os.environ.get(
        "OSHA_SMOKE_VERBOSE",
        "0",
    )
    == "1"
)

STRICT = (
    os.environ.get(
        "OSHA_SMOKE_STRICT",
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

    return str(value)


def value_of(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(value, Mapping):
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
    if isinstance(value, Enum):
        return str(value.value)

    if value is None:
        return ""

    return str(value)


def available_of(
    device: Any,
) -> bool:
    available = value_of(
        device,
        "available",
        None,
    )

    if available is not None:
        return bool(available)

    return enum_text(
        value_of(
            device,
            "state",
            "",
        )
    ).lower() in {
        "available",
        "healthy",
        "ready",
    }


# ------------------------------------------------------------------
# Supported-argument invocation
# ------------------------------------------------------------------

def invoke_supported(
    method: Callable[..., Any],
    values: Mapping[str, Any],
) -> Any:
    try:
        signature = inspect.signature(
            method
        )

    except (
        TypeError,
        ValueError,
    ):
        return method(
            **dict(values)
        )

    parameters = signature.parameters

    accepts_kwargs = any(
        parameter.kind
        == inspect.Parameter.VAR_KEYWORD
        for parameter
        in parameters.values()
    )

    if accepts_kwargs:
        arguments = dict(values)

    else:
        arguments = {
            name: value
            for name, value
            in values.items()
            if name in parameters
        }

    return method(
        **arguments
    )


# ------------------------------------------------------------------
# Device-kind normalization
# ------------------------------------------------------------------

def kind_of(
    value: Any,
) -> str:
    if value is None:
        return ""

    device = value_of(
        value,
        "device",
        value,
    )

    kind = value_of(
        value,
        "kind",
        None,
    )

    if kind is None:
        kind = value_of(
            device,
            "kind",
            None,
        )

    return enum_text(
        kind
    ).lower()


def selected_device_of(
    selection: Any,
) -> Any:
    return value_of(
        selection,
        "device",
        selection,
    )


def runtime_device_text(
    manager: Any,
    selection: Any,
) -> str:
    resolver = getattr(
        manager,
        "runtime_device",
        None,
    )

    if callable(resolver):
        try:
            resolved = resolver(
                selection
            )

            return str(resolved)

        except TypeError:
            device = selected_device_of(
                selection
            )

            resolved = resolver(
                device
            )

            return str(resolved)

    device = selected_device_of(
        selection
    )

    value = value_of(
        selection,
        "runtime_device",
        value_of(
            device,
            "runtime_device",
            None,
        ),
    )

    if value is not None:
        return str(value)

    kind = kind_of(selection)

    index = value_of(
        selection,
        "index",
        value_of(
            device,
            "index",
            0,
        ),
    )

    if kind == "cpu":
        return "cpu"

    if kind in {
        "cuda",
        "rocm",
    }:
        return f"cuda:{index or 0}"

    raise RuntimeError(
        "Unable to resolve runtime device."
    )


# ------------------------------------------------------------------
# Inventory normalization
# ------------------------------------------------------------------

def inventory_payload(
    inventory: Any,
) -> dict[str, Any]:
    devices = tuple(
        value_of(
            inventory,
            "devices",
            (),
        )
        or ()
    )

    failures = tuple(
        value_of(
            inventory,
            "failures",
            (),
        )
        or ()
    )

    return {
        "device_count": len(devices),

        "available_device_count": sum(
            1
            for device in devices
            if available_of(device)
        ),

        "devices": [
            {
                "device_id": value_of(
                    device,
                    "device_id",
                    None,
                ),

                "kind": enum_text(
                    value_of(
                        device,
                        "kind",
                        None,
                    )
                ).lower(),

                "index": value_of(
                    device,
                    "index",
                    None,
                ),

                "name": value_of(
                    device,
                    "name",
                    None,
                ),

                "runtime_device": value_of(
                    device,
                    "runtime_device",
                    None,
                ),

                "state": enum_text(
                    value_of(
                        device,
                        "state",
                        "unknown",
                    )
                ),

                "available": available_of(
                    device
                ),
            }
            for device in devices
        ],

        "backend_failures": [
            serialize(failure)
            for failure in failures
        ],
    }


# ------------------------------------------------------------------
# Selection
# ------------------------------------------------------------------

def select_device(
    manager: AcceleratorManager,
) -> Any:
    if WORKLOAD == "training":
        selector = getattr(
            manager,
            "select_for_training",
            None,
        )

    else:
        selector = getattr(
            manager,
            "select_for_inference",
            None,
        )

    if not callable(selector):
        raise RuntimeError(
            "AcceleratorManager exposes no "
            f"{WORKLOAD} selection method."
        )

    preferred_kind = (
        None
        if REQUESTED_KIND == "auto"
        else REQUESTED_KIND
    )

    arguments = {
        "preferred_kind": preferred_kind,

        "device_index": (
            DEVICE_INDEX
        ),

        "require_accelerator": (
            REQUESTED_KIND
            in {
                "cuda",
                "rocm",
            }
            and not ALLOW_CPU_FALLBACK
        ),

        "allow_cpu_fallback": (
            ALLOW_CPU_FALLBACK
        ),

        "mixed_precision": (
            MIXED_PRECISION
        ),

        "refresh": True,
    }

    return invoke_supported(
        selector,
        arguments,
    )


# ------------------------------------------------------------------
# Precision selection
# ------------------------------------------------------------------

def test_dtype(
    selected_kind: str,
) -> torch.dtype:
    if not MIXED_PRECISION:
        return torch.float32

    if selected_kind == "cpu":
        return torch.float32

    # Float16 is broadly appropriate for the small
    # accelerator smoke operation. This test does
    # not claim model-level mixed-precision support.
    return torch.float16


# ------------------------------------------------------------------
# Computation
# ------------------------------------------------------------------

def run_computation(
    runtime_text: str,
    selected_kind: str,
) -> dict[str, Any]:
    device = torch.device(
        runtime_text
    )

    dtype = test_dtype(
        selected_kind
    )

    started = time.perf_counter()

    left = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=dtype,
        device=device,
    )

    right = torch.tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=dtype,
        device=device,
    )

    result = torch.matmul(
        left,
        right,
    )

    expected = torch.tensor(
        [
            [19.0, 22.0],
            [43.0, 50.0],
        ],
        dtype=dtype,
        device=device,
    )

    valid = bool(
        torch.allclose(
            result,
            expected,
            rtol=1e-3,
            atol=1e-3,
        )
    )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    duration_ms = (
        time.perf_counter()
        - started
    ) * 1000.0

    if not valid:
        raise RuntimeError(
            "Accelerator smoke computation "
            "returned an unexpected result."
        )

    return {
        "success": True,

        "runtime_device": str(device),

        "device_type": device.type,

        "dtype": str(dtype),

        "operation": "matrix_multiplication",

        "shape": [
            2,
            2,
        ],

        "result": (
            result.
            detach().
            to(
                dtype=torch.float32,
                device="cpu",
            ).
           tolist()
        ),

        "expected": (
            expected.
            detach().
            to(
                dtype=torch.float32,
                device="cpu",
            ).
           tolist()
        ),

        "duration_ms": (
            duration_ms
        ),
    }


# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------

manager = AcceleratorManager()
selection = None

exit_code = 0


try:
    inventory = manager.inventory(
        refresh=True
    )

    normalized_inventory = (
        inventory_payload(
            inventory
        )
    )

    if (
        normalized_inventory[
            "available_device_count"
        ]
        < 1
    ):
        raise RuntimeError(
            "OSHA discovered no available "
            "runtime devices."
        )

    selection = select_device(
        manager
    )

    device = selected_device_of(
        selection
    )

    selected_kind = kind_of(
        selection
    )

    selected_device_id = value_of(
        selection,
        "device_id",
        value_of(
            device,
            "device_id",
            None,
        ),
    )

    selected_index = value_of(
        selection,
        "index",
        value_of(
            device,
            "index",
            None,
        ),
    )

    selected_name = value_of(
        device,
        "name",
        None,
    )

    fallback_used = bool(
        value_of(
            selection,
            "fallback_used",
            False,
        )
    )

    selection_reason = value_of(
        selection,
        "reason",
        None,
    )

    runtime_text = runtime_device_text(
        manager,
        selection,
    )

    if (
        STRICT
        and REQUESTED_KIND
        != "auto"
        and selected_kind
        != REQUESTED_KIND
    ):
        raise RuntimeError(
            "Selected accelerator kind does not "
            "match the strict requested kind: "
            f"requested={REQUESTED_KIND}, "
            f"selected={selected_kind}"
        )

    if (
        not ALLOW_CPU_FALLBACK
        and REQUESTED_KIND
        in {
            "cuda",
            "rocm",
        }
        and selected_kind
        == "cpu"
    ):
        raise RuntimeError(
            "CPU fallback occurred even though "
            "fallback was disabled."
        )

    computation = run_computation(
        runtime_text,
        selected_kind,
    )

    payload = {
        "schema_version": "1",

        "success": True,

        "request": {
            "kind": REQUESTED_KIND,

            "device_index": (
                DEVICE_INDEX
            ),

            "workload": (
                WORKLOAD
            ),

            "mixed_precision": (
                MIXED_PRECISION
            ),

            "allow_cpu_fallback": (
                ALLOW_CPU_FALLBACK
            ),

            "strict": STRICT,
        },

        "selection": {
            "device_id": (
                selected_device_id
            ),

            "kind": (
                selected_kind
            ),

            "index": (
                selected_index
            ),

            "name": (
                selected_name
            ),

            "runtime_device": (
                runtime_text
            ),

            "fallback_used": (
                fallback_used
            ),

            "reason": (
                selection_reason
            ),
        },

        "computation": (
            computation
        ),

        "inventory": (
            normalized_inventory
            if VERBOSE
            else {
                "device_count": (
                    normalized_inventory[
                        "device_count"
                    ]
                ),

                "available_device_count": (
                    normalized_inventory[
                        "available_device_count"
                    ]
                ),

                "backend_failure_count": len(
                    normalized_inventory[
                        "backend_failures"
                    ]
                ),
            }
        ),
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
        print()

        print(
            "Inventory"
        )

        print(
            "---------"
        )

        print(
            (
                "Devices:             "
                f"{normalized_inventory['device_count']}"
            )
        )

        print(
            (
                "Available devices:   "
                f"{normalized_inventory['available_device_count']}"
            )
        )

        print(
            (
                "Backend failures:    "
                f"{len(normalized_inventory['backend_failures'])}"
            )
        )

        if VERBOSE:
            for current in normalized_inventory[
                "devices"
            ]:
                print(
                    (
                        "  "
                        f"{current['device_id']}: "
                        f"kind={current['kind']} "
                        f"state={current['state']} "
                        f"available={current['available']}"
                    )
                )


        print()

        print(
            "Selection"
        )

        print(
            "---------"
        )

        print(
            (
                "Device ID:           "
                f"{selected_device_id}"
            )
        )

        print(
            (
                "Kind:                "
                f"{selected_kind}"
            )
        )

        print(
            (
                "Index:               "
                f"{selected_index}"
            )
        )

        print(
            (
                "Name:                "
                f"{selected_name}"
            )
        )

        print(
            (
                "Runtime device:      "
                f"{runtime_text}"
            )
        )

        print(
            (
                "CPU fallback used:   "
                f"{fallback_used}"
            )
        )

        print(
            (
                "Selection reason:    "
                f"{selection_reason}"
            )
        )


        print()

        print(
            "Computation"
        )

        print(
            "-----------"
        )

        print(
            (
                "Operation:           "
                f"{computation['operation']}"
            )
        )

        print(
            (
                "Device:              "
                f"{computation['runtime_device']}"
            )
        )

        print(
            (
                "Data type:           "
                f"{computation['dtype']}"
            )
        )

        print(
            (
                "Result:              "
                f"{computation['result']}"
            )
        )

        print(
            (
                "Expected:            "
                f"{computation['expected']}"
            )
        )

        print(
            (
                "Duration:            "
                f"{computation['duration_ms']:.3f} ms"
            )
        )

        print()

        print(
            "Accelerator smoke test passed."
        )


except Exception as exc:
    exit_code = 1

    error_payload = {
        "schema_version": "1",

        "success": False,

        "request": {
            "kind": REQUESTED_KIND,

            "device_index": (
                DEVICE_INDEX
            ),

            "workload": (
                WORKLOAD
            ),

            "mixed_precision": (
                MIXED_PRECISION
            ),

            "allow_cpu_fallback": (
                ALLOW_CPU_FALLBACK
            ),

            "strict": STRICT,
        },

        "error": {
            "type": (
                type(exc).__name__
            ),

            "message": str(exc),
        },
    }

    if JSON_OUTPUT:
        print(
            json.dumps(
                error_payload,
                indent=2,
                sort_keys=True,
                default=str,
            )
        )

    else:
        print(
            (
                "Accelerator smoke test failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            file=os.sys.stderr,
        )


finally:
    if selection is not None:
        release = getattr(
            manager,
            "release",
            None,
        )

        if callable(release):
            try:
                release(
                    selection
                )

            except Exception as exc:
                if exit_code == 0:
                    exit_code = 1

                if not JSON_OUTPUT:
                    print(
                        (
                            "Accelerator release failed: "
                            f"{type(exc).__name__}"
                        ),
                        file=os.sys.stderr,
                    )

    release_all = getattr(
        manager,
        "release_all",
        None,
    )

    if callable(release_all):
        try:
            release_all()

        except Exception as exc:
            if exit_code == 0:
                exit_code = 1

            if not JSON_OUTPUT:
                print(
                    (
                        "Accelerator release-all failed: "
                        f"{type(exc).__name__}"
                    ),
                    file=os.sys.stderr,
                )


raise SystemExit(
    exit_code
)
PY

SMOKE_EXIT_CODE=$?

set -e


# -------------------------------------------------------------------
# Final result
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


if [[ "${JSON_OUTPUT}" == true ]]; then
    exit "${SMOKE_EXIT_CODE}"
fi


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA Accelerator Smoke Test Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Python:           %s\n' \
    "${PYTHON_VERSION}"

printf 'Requested kind:   %s\n' \
    "${KIND}"

printf 'Device index:     %s\n' \
    "$(
        if [[ -n "${DEVICE_INDEX}" ]]; then
            printf '%s' "${DEVICE_INDEX}"
        else
            printf 'automatic'
        fi
    )"

printf 'Workload:         %s\n' \
    "${WORKLOAD}"

printf 'Mixed precision:  %s\n' \
    "${MIXED_PRECISION}"

printf 'CPU fallback:     %s\n' \
    "${ALLOW_CPU_FALLBACK}"

printf 'Strict:           %s\n' \
    "${STRICT}"

printf 'Duration:         %ss\n' \
    "${elapsed}"

printf 'Exit code:        %s\n' \
    "${SMOKE_EXIT_CODE}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


if ((SMOKE_EXIT_CODE == 0)); then
    printf '\n%sACCELERATOR SMOKE TEST PASSED%s\n' \
        "${GREEN}" \
        "${RESET}"

else
    printf '\n%sACCELERATOR SMOKE TEST FAILED%s\n' \
        "${RED}" \
        "${RESET}"
fi


exit "${SMOKE_EXIT_CODE}"
