#!/usr/bin/env bash

#
# OSHA CUDA accelerator diagnostics.
#
# Usage:
#
#   ./scripts/accelerators/cuda.sh
#   ./scripts/accelerators/cuda.sh --device 0
#   ./scripts/accelerators/cuda.sh --metrics
#   ./scripts/accelerators/cuda.sh --smoke
#   ./scripts/accelerators/cuda.sh --json
#   ./scripts/accelerators/cuda.sh --verbose
#   ./scripts/accelerators/cuda.sh --strict
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This script is read-only.
#
# It does NOT:
#
#   reset GPUs
#   modify clocks
#   change power limits
#   change persistence mode
#   kill GPU processes
#   clear CUDA caches
#   reserve an OSHA accelerator
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
METRICS=false
SMOKE=false
STRICT=false

DEVICE_INDEX=""

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
  scripts/accelerators/cuda.sh [options]

Options:
  --device INDEX
      Inspect one CUDA device index.

  --metrics
      Include NVIDIA runtime metrics when
      nvidia-smi is available.

  --smoke
      Run a tiny PyTorch CUDA tensor operation.

  --json
      Emit machine-readable JSON.

  --verbose
      Include additional runtime/device details.

  --strict
      Fail when a usable CUDA device cannot be
      confirmed across required layers.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override Python interpreter.

Examples:
  ./scripts/accelerators/cuda.sh

  ./scripts/accelerators/cuda.sh \
      --device 0

  ./scripts/accelerators/cuda.sh \
      --metrics

  ./scripts/accelerators/cuda.sh \
      --smoke

  ./scripts/accelerators/cuda.sh \
      --metrics \
      --smoke \
      --verbose

  ./scripts/accelerators/cuda.sh \
      --json
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
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

        --metrics)
            METRICS=true
            ;;

        --smoke)
            SMOKE=true
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
# Validate device
# -------------------------------------------------------------------

if [[ -n "${DEVICE_INDEX}" ]]; then
    if [[ ! "${DEVICE_INDEX}" =~ ^[0-9]+$ ]]; then
        printf \
            'ERROR: --device must be a non-negative integer\n' \
            >&2

        exit 2
    fi
fi


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
# Required imports
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
    printf '\n%sOSHA CUDA Diagnostics%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Project root: ${PROJECT_ROOT}"
    info "Python: ${PYTHON}"
    info "Python version: ${PYTHON_VERSION}"

    if [[ -n "${DEVICE_INDEX}" ]]; then
        info "Device index: ${DEVICE_INDEX}"
    else
        info "Device index: all"
    fi

    info "Metrics: ${METRICS}"
    info "Smoke test: ${SMOKE}"
    info "Strict: ${STRICT}"
fi


# -------------------------------------------------------------------
# Visibility environment
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%sCUDA visibility%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'CUDA_VISIBLE_DEVICES: %s\n' \
        "${CUDA_VISIBLE_DEVICES:-<unset>}"

    printf 'NVIDIA_VISIBLE_DEVICES: %s\n' \
        "${NVIDIA_VISIBLE_DEVICES:-<unset>}"
fi


# -------------------------------------------------------------------
# NVIDIA driver view
# -------------------------------------------------------------------

NVIDIA_SMI_FOUND=false
NVIDIA_SMI_OK=false
NVIDIA_JSON='null'


if command -v nvidia-smi >/dev/null 2>&1; then
    NVIDIA_SMI_FOUND=true

    set +e

    NVIDIA_QUERY="$(
        nvidia-smi \
            --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
            --format=csv,noheader,nounits \
            2>&1
    )"

    NVIDIA_EXIT=$?

    set -e


    if ((NVIDIA_EXIT == 0)); then
        NVIDIA_SMI_OK=true

        NVIDIA_JSON="$(
            NVIDIA_QUERY="${NVIDIA_QUERY}" \
            "${PYTHON}" - <<'PY'
from __future__ import annotations

import json
import os


text = os.environ.get(
    "NVIDIA_QUERY",
    "",
)


devices = []


for line in text.splitlines():
    line = line.strip()

    if not line:
        continue

    fields = [
        value.strip()
        for value in line.split(",")
    ]

    if len(fields) < 9:
        continue

    (
        index,
        name,
        uuid,
        driver,
        total_mb,
        used_mb,
        free_mb,
        utilization,
        temperature,
    ) = fields[:9]


    def number(
        value: str,
    ):
        try:
            return float(value)

        except ValueError:
            return None


    devices.append(
        {
            "index": int(index),

            "name": name,

            "uuid": uuid,

            "driver_version": driver,

            "memory_total_mib": number(
                total_mb
            ),

            "memory_used_mib": number(
                used_mb
            ),

            "memory_free_mib": number(
                free_mb
            ),

            "utilization_percent": number(
                utilization
            ),

            "temperature_c": number(
                temperature
            ),
        }
    )


print(
    json.dumps(
        devices,
        sort_keys=True,
    )
)
PY
        )"

        if [[ "${JSON_OUTPUT}" == false ]]; then
            printf '\n%sNVIDIA driver view%s\n' \
                "${BOLD}" \
                "${RESET}"

            success "nvidia-smi"

            if [[ "${METRICS}" == true ]]; then
                printf '%s' \
                    "${NVIDIA_JSON}" \
                | "${PYTHON}" -m json.tool

            else
                NVIDIA_JSON="${NVIDIA_JSON}" \
                "${PYTHON}" - <<'PY'
import json
import os


devices = json.loads(
    os.environ["NVIDIA_JSON"]
)


if not devices:
    print(
        "No NVIDIA GPUs reported."
    )


for device in devices:
    print(
        (
            f"[{device['index']}] "
            f"{device['name']}"
        )
    )

    print(
        (
            "    Driver: "
            f"{device['driver_version']}"
        )
    )

    print(
        (
            "    UUID:   "
            f"{device['uuid']}"
        )
    )
PY
            fi
        fi

    else
        if [[ "${JSON_OUTPUT}" == false ]]; then
            warn "nvidia-smi failed"

            if [[ "${VERBOSE}" == true ]]; then
                printf '%s\n' \
                    "${NVIDIA_QUERY}" >&2
            fi
        fi
    fi

else
    if [[ "${JSON_OUTPUT}" == false ]]; then
        info "nvidia-smi is not installed"
    fi
fi


# -------------------------------------------------------------------
# Export runtime controls
# -------------------------------------------------------------------

export OSHA_CUDA_JSON="$(
    if [[ "${JSON_OUTPUT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_CUDA_VERBOSE="$(
    if [[ "${VERBOSE}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_CUDA_SMOKE="$(
    if [[ "${SMOKE}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_CUDA_STRICT="$(
    if [[ "${STRICT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_CUDA_DEVICE="${DEVICE_INDEX}"

export OSHA_NVIDIA_SMI_FOUND="$(
    if [[ "${NVIDIA_SMI_FOUND}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_NVIDIA_SMI_OK="$(
    if [[ "${NVIDIA_SMI_OK}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_NVIDIA_JSON="${NVIDIA_JSON}"


# -------------------------------------------------------------------
# PyTorch + OSHA CUDA inspection
# -------------------------------------------------------------------

set +e

"${PYTHON}" - <<'PY'
from __future__ import annotations

import json
import os

from dataclasses import (
    asdict,
    is_dataclass,
)

from enum import Enum

from typing import (
    Any,
    Mapping,
)


import torch


from src.accelerators import (
    AcceleratorManager,
)


JSON_OUTPUT = (
    os.environ.get(
        "OSHA_CUDA_JSON",
        "0",
    )
    == "1"
)

VERBOSE = (
    os.environ.get(
        "OSHA_CUDA_VERBOSE",
        "0",
    )
    == "1"
)

RUN_SMOKE = (
    os.environ.get(
        "OSHA_CUDA_SMOKE",
        "0",
    )
    == "1"
)

STRICT = (
    os.environ.get(
        "OSHA_CUDA_STRICT",
        "0",
    )
    == "1"
)

DEVICE_TEXT = os.environ.get(
    "OSHA_CUDA_DEVICE",
    "",
).strip()

DEVICE_INDEX = (
    int(DEVICE_TEXT)
    if DEVICE_TEXT
    else None
)

NVIDIA_SMI_FOUND = (
    os.environ.get(
        "OSHA_NVIDIA_SMI_FOUND",
        "0",
    )
    == "1"
)

NVIDIA_SMI_OK = (
    os.environ.get(
        "OSHA_NVIDIA_SMI_OK",
        "0",
    )
    == "1"
)


try:
    NVIDIA_DEVICES = json.loads(
        os.environ.get(
            "OSHA_NVIDIA_JSON",
            "null",
        )
    )

except json.JSONDecodeError:
    NVIDIA_DEVICES = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def serialize(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(value, Enum):
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


def human_bytes(
    value: Any,
) -> str:
    if value is None:
        return "-"

    number = float(value)

    units = (
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    )

    unit = 0

    while (
        number >= 1024
        and unit < len(units) - 1
    ):
        number /= 1024
        unit += 1

    return (
        f"{number:.2f} "
        f"{units[unit]}"
    )


# ------------------------------------------------------------------
# PyTorch runtime
# ------------------------------------------------------------------

cuda_version = torch.version.cuda

hip_version = getattr(
    torch.version,
    "hip",
    None,
)


# A HIP build uses torch.cuda interfaces but is
# not an NVIDIA CUDA build.
is_cuda_build = bool(
    cuda_version
    and not hip_version
)


try:
    torch_available = bool(
        torch.cuda.is_available()
    )

except Exception:
    torch_available = False


try:
    torch_device_count = int(
        torch.cuda.device_count()
    )

except Exception:
    torch_device_count = 0


torch_devices = []


if is_cuda_build:
    for index in range(
        torch_device_count
    ):
        if (
            DEVICE_INDEX is not None
            and index != DEVICE_INDEX
        ):
            continue

        device = {
            "index": index,
        }

        try:
            device[
                "name"
            ] = torch.cuda.get_device_name(
                index
            )

        except Exception as exc:
            device[
                "name_error"
            ] = type(exc).__name__

        try:
            properties = (
                torch.cuda.
                get_device_properties(
                    index
                )
            )

            device.update(
                {
                    "total_memory_bytes": int(
                        properties.total_memory
                    ),

                    "compute_capability": (
                        f"{properties.major}."
                        f"{properties.minor}"
                    ),

                    "multi_processor_count": int(
                        getattr(
                            properties,
                            "multi_processor_count",
                            0,
                        )
                    ),
                }
            )

        except Exception as exc:
            device[
                "properties_error"
            ] = type(exc).__name__

        try:
            free_bytes, total_bytes = (
                torch.cuda.mem_get_info(
                    index
                )
            )

            device[
                "free_memory_bytes"
            ] = int(
                free_bytes
            )

            device[
                "runtime_total_memory_bytes"
            ] = int(
                total_bytes
            )

        except Exception as exc:
            device[
                "memory_error"
            ] = type(exc).__name__

        try:
            device[
                "bf16_supported"
            ] = bool(
                torch.cuda.
                is_bf16_supported()
            )

        except Exception:
            device[
                "bf16_supported"
            ] = None

        torch_devices.append(
            device
        )


# ------------------------------------------------------------------
# Tiny smoke operation
# ------------------------------------------------------------------

smoke = None


if RUN_SMOKE:
    smoke = {
        "requested": True,
        "success": False,
    }

    if (
        not is_cuda_build
        or not torch_available
        or torch_device_count < 1
    ):
        smoke[
            "reason"
        ] = (
            "CUDA runtime is unavailable."
        )

    elif (
        DEVICE_INDEX is not None
        and DEVICE_INDEX
        >= torch_device_count
    ):
        smoke[
            "reason"
        ] = (
            "Requested CUDA device index "
            "does not exist."
        )

    else:
        index = (
            DEVICE_INDEX
            if DEVICE_INDEX is not None
            else 0
        )

        try:
            device = torch.device(
                f"cuda:{index}"
            )

            left = torch.tensor(
                [
                    1.0,
                    2.0,
                    3.0,
                ],
                device=device,
            )

            right = torch.tensor(
                [
                    4.0,
                    5.0,
                    6.0,
                ],
                device=device,
            )

            result = left + right

            torch.cuda.synchronize(
                device
            )

            actual = (
                result.cpu().tolist()
            )

            if actual != [
                5.0,
                7.0,
                9.0,
            ]:
                raise RuntimeError(
                    "Unexpected CUDA tensor result."
                )

            smoke.update(
                {
                    "success": True,
                    "device": str(
                        device
                    ),
                    "result": actual,
                }
            )

        except Exception as exc:
            smoke.update(
                {
                    "success": False,
                    "error_type": (
                        type(exc).__name__
                    ),
                }
            )


# ------------------------------------------------------------------
# OSHA inventory
# ------------------------------------------------------------------

manager = AcceleratorManager()


try:
    inventory = manager.inventory(
        refresh=True
    )

    source_devices = tuple(
        value_of(
            inventory,
            "devices",
            (),
        )
        or ()
    )

    cuda_devices = []


    for device in source_devices:
        kind = enum_text(
            value_of(
                device,
                "kind",
                "",
            )
        ).lower()

        if kind != "cuda":
            continue

        index = value_of(
            device,
            "index",
            None,
        )

        if (
            DEVICE_INDEX is not None
            and index != DEVICE_INDEX
        ):
            continue

        memory = serialize(
            value_of(
                device,
                "memory",
                None,
            )
        )

        if not isinstance(
            memory,
            dict,
        ):
            memory = {}

        item = {
            "device_id": value_of(
                device,
                "device_id",
                None,
            ),

            "index": index,

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

            "vendor": value_of(
                device,
                "vendor",
                None,
            ),

            "architecture": value_of(
                device,
                "architecture",
                None,
            ),

            "compute_capability": value_of(
                device,
                "compute_capability",
                None,
            ),

            "runtime_version": value_of(
                device,
                "runtime_version",
                None,
            ),

            "memory": memory,

            "capabilities": serialize(
                value_of(
                    device,
                    "capabilities",
                    (),
                )
            ),

            "supported_precisions": serialize(
                value_of(
                    device,
                    "supported_precisions",
                    (),
                )
            ),
        }

        if VERBOSE:
            item[
                "attributes"
            ] = serialize(
                value_of(
                    device,
                    "attributes",
                    {},
                )
            )

        cuda_devices.append(
            item
        )


    failures = serialize(
        value_of(
            inventory,
            "failures",
            (),
        )
        or ()
    )

    if not isinstance(
        failures,
        list,
    ):
        failures = []


    available_cuda_count = sum(
        1
        for device
        in cuda_devices
        if device[
            "available"
        ]
    )


    payload = {
        "schema_version": "1",

        "requested_device": (
            DEVICE_INDEX
        ),

        "nvidia": {
            "nvidia_smi_found": (
                NVIDIA_SMI_FOUND
            ),

            "nvidia_smi_ok": (
                NVIDIA_SMI_OK
            ),

            "devices": (
                NVIDIA_DEVICES
                if isinstance(
                    NVIDIA_DEVICES,
                    list,
                )
                else []
            ),
        },

        "pytorch": {
            "version": str(
                torch.__version__
            ),

            "cuda_version": (
                cuda_version
            ),

            "hip_version": (
                hip_version
            ),

            "is_cuda_build": (
                is_cuda_build
            ),

            "available": (
                torch_available
            ),

            "device_count": (
                torch_device_count
            ),

            "devices": torch_devices,

            "smoke": smoke,
        },

        "osha": {
            "device_count": len(
                cuda_devices
            ),

            "available_device_count": (
                available_cuda_count
            ),

            "devices": cuda_devices,

            "backend_failures": failures,
        },
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
            "PyTorch CUDA runtime"
        )

        print(
            "--------------------"
        )

        print(
            (
                "PyTorch:        "
                f"{torch.__version__}"
            )
        )

        print(
            (
                "CUDA build:     "
                f"{is_cuda_build}"
            )
        )

        print(
            (
                "CUDA version:   "
                f"{cuda_version}"
            )
        )

        print(
            (
                "HIP version:    "
                f"{hip_version}"
            )
        )

        print(
            (
                "CUDA available: "
                f"{torch_available}"
            )
        )

        print(
            (
                "Runtime devices:"
                f" {torch_device_count}"
            )
        )


        for device in torch_devices:
            print()

            print(
                (
                    f"[{device['index']}] "
                    f"{device.get('name')}"
                )
            )

            if (
                device.get(
                    "compute_capability"
                )
                is not None
            ):
                print(
                    (
                        "    Compute capability: "
                        f"{device['compute_capability']}"
                    )
                )

            if (
                device.get(
                    "total_memory_bytes"
                )
                is not None
            ):
                print(
                    (
                        "    Total memory:       "
                        f"{human_bytes(device['total_memory_bytes'])}"
                    )
                )

            if (
                device.get(
                    "free_memory_bytes"
                )
                is not None
            ):
                print(
                    (
                        "    Free memory:        "
                        f"{human_bytes(device['free_memory_bytes'])}"
                    )
                )

            print(
                (
                    "    BF16 supported:     "
                    f"{device.get('bf16_supported')}"
                )
            )


        if smoke is not None:
            print()

            print(
                "CUDA smoke test"
            )

            print(
                "---------------"
            )

            print(
                json.dumps(
                    smoke,
                    indent=2,
                    sort_keys=True,
                )
            )


        print()

        print(
            "OSHA CUDA inventory"
        )

        print(
            "-------------------"
        )

        print(
            (
                "CUDA devices:      "
                f"{len(cuda_devices)}"
            )
        )

        print(
            (
                "Available devices: "
                f"{available_cuda_count}"
            )
        )

        print(
            (
                "Backend failures:  "
                f"{len(failures)}"
            )
        )


        for device in cuda_devices:
            print()

            print(
                (
                    f"[{device['index']}] "
                    f"{device['device_id']}"
                )
            )

            print(
                (
                    "    Name:      "
                    f"{device['name']}"
                )
            )

            print(
                (
                    "    Runtime:   "
                    f"{device['runtime_device']}"
                )
            )

            print(
                (
                    "    State:     "
                    f"{device['state']}"
                )
            )

            print(
                (
                    "    Available: "
                    f"{device['available']}"
                )
            )

            if device[
                "architecture"
            ]:
                print(
                    (
                        "    Architecture: "
                        f"{device['architecture']}"
                    )
                )

            if device[
                "compute_capability"
            ]:
                print(
                    (
                        "    Compute:      "
                        f"{device['compute_capability']}"
                    )
                )

            memory = device[
                "memory"
            ]

            if (
                memory.get(
                    "total_bytes"
                )
                is not None
            ):
                print(
                    (
                        "    Memory total: "
                        f"{human_bytes(memory['total_bytes'])}"
                    )
                )

            if (
                memory.get(
                    "free_bytes"
                )
                is not None
            ):
                print(
                    (
                        "    Memory free:  "
                        f"{human_bytes(memory['free_bytes'])}"
                    )
                )


    strict_failures = []


    if STRICT:
        if not NVIDIA_SMI_FOUND:
            strict_failures.append(
                "nvidia-smi is unavailable"
            )

        elif not NVIDIA_SMI_OK:
            strict_failures.append(
                "nvidia-smi failed"
            )

        if not is_cuda_build:
            strict_failures.append(
                "PyTorch is not a CUDA build"
            )

        if not torch_available:
            strict_failures.append(
                "PyTorch CUDA runtime is unavailable"
            )

        if torch_device_count < 1:
            strict_failures.append(
                "PyTorch reports zero CUDA devices"
            )

        if available_cuda_count < 1:
            strict_failures.append(
                "OSHA reports no available CUDA device"
            )

        if (
            DEVICE_INDEX is not None
            and not any(
                device.get(
                    "index"
                )
                == DEVICE_INDEX
                for device
                in cuda_devices
            )
        ):
            strict_failures.append(
                "Requested CUDA device is not present "
                "in OSHA inventory"
            )

        if (
            RUN_SMOKE
            and (
                smoke is None
                or not smoke.get(
                    "success",
                    False,
                )
            )
        ):
            strict_failures.append(
                "CUDA smoke operation failed"
            )


    if (
        strict_failures
        and not JSON_OUTPUT
    ):
        print()

        print(
            "Strict validation failures"
        )

        print(
            "--------------------------"
        )

        for value in (
            strict_failures
        ):
            print(
                "  - " + value
            )


    raise SystemExit(
        1
        if strict_failures
        else 0
    )


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

CUDA_EXIT_CODE=$?

set -e


# -------------------------------------------------------------------
# Final result
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


if [[ "${JSON_OUTPUT}" == true ]]; then
    exit "${CUDA_EXIT_CODE}"
fi


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA CUDA Diagnostic Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Python:      %s\n' \
    "${PYTHON_VERSION}"

printf 'Device:      %s\n' \
    "$(
        if [[ -n "${DEVICE_INDEX}" ]]; then
            printf '%s' "${DEVICE_INDEX}"
        else
            printf 'all'
        fi
    )"

printf 'nvidia-smi:  %s\n' \
    "${NVIDIA_SMI_FOUND}"

printf 'Metrics:     %s\n' \
    "${METRICS}"

printf 'Smoke test:  %s\n' \
    "${SMOKE}"

printf 'Strict:      %s\n' \
    "${STRICT}"

printf 'Duration:    %ss\n' \
    "${elapsed}"

printf 'Exit code:   %s\n' \
    "${CUDA_EXIT_CODE}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


if ((CUDA_EXIT_CODE == 0)); then
    printf '\n%sCUDA DIAGNOSTICS COMPLETE%s\n' \
        "${GREEN}" \
        "${RESET}"
else
    printf '\n%sCUDA DIAGNOSTICS FAILED%s\n' \
        "${RED}" \
        "${RESET}"
fi


exit "${CUDA_EXIT_CODE}"
