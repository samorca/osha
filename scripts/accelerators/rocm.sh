#!/usr/bin/env bash

#
# OSHA ROCm accelerator diagnostics.
#
# Usage:
#
#   ./scripts/accelerators/rocm.sh
#   ./scripts/accelerators/rocm.sh --device 0
#   ./scripts/accelerators/rocm.sh --metrics
#   ./scripts/accelerators/rocm.sh --smoke
#   ./scripts/accelerators/rocm.sh --json
#   ./scripts/accelerators/rocm.sh --verbose
#   ./scripts/accelerators/rocm.sh --strict
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This script is read-only.
#
# It does not:
#
#   reset AMD GPUs
#   modify clocks or power limits
#   modify compute/memory partitions
#   change performance levels
#   kill GPU processes
#   clear PyTorch caches
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
  scripts/accelerators/rocm.sh [options]

Options:
  --device INDEX
      Inspect one ROCm/HIP runtime device.

  --metrics
      Include metrics from amd-smi or rocm-smi
      when one of those tools is available.

  --smoke
      Run a tiny PyTorch HIP tensor operation.

  --json
      Emit machine-readable JSON.

  --verbose
      Include additional runtime and device details.

  --strict
      Fail unless a usable ROCm device is confirmed
      through PyTorch and OSHA.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override the Python interpreter.

Examples:
  ./scripts/accelerators/rocm.sh

  ./scripts/accelerators/rocm.sh \
      --device 0

  ./scripts/accelerators/rocm.sh \
      --metrics

  ./scripts/accelerators/rocm.sh \
      --smoke

  ./scripts/accelerators/rocm.sh \
      --metrics \
      --smoke \
      --verbose

  ./scripts/accelerators/rocm.sh \
      --strict \
      --smoke

  ./scripts/accelerators/rocm.sh \
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
# Argument validation
# -------------------------------------------------------------------

if [[ -n "${DEVICE_INDEX}" ]] \
    && [[ ! "${DEVICE_INDEX}" =~ ^[0-9]+$ ]]
then
    printf \
        'ERROR: --device must be a non-negative integer\n' \
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
    printf '\n%sOSHA ROCm Diagnostics%s\n' \
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
# Visibility variables
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%sROCm visibility%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'HIP_VISIBLE_DEVICES:  %s\n' \
        "${HIP_VISIBLE_DEVICES:-<unset>}"

    printf 'ROCR_VISIBLE_DEVICES: %s\n' \
        "${ROCR_VISIBLE_DEVICES:-<unset>}"

    printf 'GPU_DEVICE_ORDINAL:   %s\n' \
        "${GPU_DEVICE_ORDINAL:-<unset>}"

    printf 'CUDA_VISIBLE_DEVICES: %s\n' \
        "${CUDA_VISIBLE_DEVICES:-<unset>}"
fi


# -------------------------------------------------------------------
# Device nodes
# -------------------------------------------------------------------

KFD_PRESENT=false
RENDER_NODE_COUNT=0


if [[ -e /dev/kfd ]]; then
    KFD_PRESENT=true
fi


while IFS= read -r node; do
    if [[ -n "${node}" ]]; then
        RENDER_NODE_COUNT=$((RENDER_NODE_COUNT + 1))
    fi
done < <(
    compgen -G '/dev/dri/renderD*' \
    || true
)


if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%sLinux device nodes%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf '/dev/kfd:         %s\n' \
        "${KFD_PRESENT}"

    printf 'DRI render nodes: %s\n' \
        "${RENDER_NODE_COUNT}"

    if [[ "${VERBOSE}" == true ]]; then
        if [[ -e /dev/kfd ]]; then
            ls -l /dev/kfd
        fi

        for node in /dev/dri/renderD*; do
            if [[ -e "${node}" ]]; then
                ls -l "${node}"
            fi
        done
    fi
fi


# -------------------------------------------------------------------
# AMD tooling
# -------------------------------------------------------------------

AMD_TOOL="none"
AMD_TOOL_OK=false
AMD_LIST_JSON='null'
AMD_METRICS_JSON='null'
AMD_TOOL_ERROR=""


if command -v amd-smi >/dev/null 2>&1; then
    AMD_TOOL="amd-smi"

    set +e

    AMD_LIST_OUTPUT="$(
        amd-smi list \
            --json \
            2>&1
    )"

    AMD_LIST_EXIT=$?

    set -e


    if ((AMD_LIST_EXIT == 0)); then
        AMD_TOOL_OK=true
        AMD_LIST_JSON="${AMD_LIST_OUTPUT}"

    else
        AMD_TOOL_ERROR="${AMD_LIST_OUTPUT}"
    fi


    if [[ "${METRICS}" == true ]] \
        && [[ "${AMD_TOOL_OK}" == true ]]; then
        AMD_METRICS_ARGS=(
            metric
            --json
        )

        if [[ -n "${DEVICE_INDEX}" ]]; then
            AMD_METRICS_ARGS+=(
                --gpu
                "${DEVICE_INDEX}"
            )
        fi

        set +e

        AMD_METRICS_OUTPUT="$(
            amd-smi \
                "${AMD_METRICS_ARGS[@]}" \
                2>&1
        )"

        AMD_METRICS_EXIT=$?

        set -e


        if ((AMD_METRICS_EXIT == 0)); then
            AMD_METRICS_JSON="$(
                printf '%s' \
                    "${AMD_METRICS_OUTPUT}"
            )"
        else
            AMD_TOOL_ERROR="$(
                printf '%s\n%s' \
                    "${AMD_TOOL_ERROR}" \
                    "${AMD_METRICS_OUTPUT}"
            )"
        fi
    fi


elif command -v rocm-smi >/dev/null 2>&1; then
    AMD_TOOL="rocm-smi"

    ROCM_ARGS=(
        --showproductname
        --showuniqueid
        --showmeminfo
        vram
    )

    if [[ "${METRICS}" == true ]]; then
        ROCM_ARGS+=(
            --showuse
            --showtemp
            --showpower
            --showclocks
        )
    fi

    if [[ -n "${DEVICE_INDEX}" ]]; then
        ROCM_ARGS+=(
            --device
            "${DEVICE_INDEX}"
        )
    fi

    ROCM_ARGS+=(
        --json
    )

    set +e

    ROCM_OUTPUT="$(
        rocm-smi \
            "${ROCM_ARGS[@]}" \
            2>&1
    )"

    ROCM_EXIT=$?

    set -e


    if ((ROCM_EXIT == 0)); then
        AMD_TOOL_OK=true

        AMD_LIST_JSON="$(
            printf '%s' \
                "${ROCM_OUTPUT}"
        )"

    else
        AMD_TOOL_ERROR="${ROCM_OUTPUT}"
    fi
fi


if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%sAMD runtime tooling%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'Selected tool: %s\n' \
        "${AMD_TOOL}"

    printf 'Tool success:  %s\n' \
        "${AMD_TOOL_OK}"

    if [[ "${AMD_TOOL}" == "none" ]]; then
        info \
            "Neither amd-smi nor rocm-smi is installed"

    elif [[ "${AMD_TOOL_OK}" == true ]]; then
        success "${AMD_TOOL}"

        if [[ "${VERBOSE}" == true ]] \
            || [[ "${METRICS}" == true ]]
        then
            printf '\nStatic/list output:\n'

            printf '%s\n' \
                "${AMD_LIST_JSON}"

            if [[ "${AMD_METRICS_JSON}" != "null" ]]; then
                printf '\nMetrics output:\n'

                printf '%s\n' \
                    "${AMD_METRICS_JSON}"
            fi
        fi

    else
        warn "${AMD_TOOL} failed"

        if [[ "${VERBOSE}" == true ]]; then
            printf '%s\n' \
                "${AMD_TOOL_ERROR}" >&2
        fi
    fi
fi


# -------------------------------------------------------------------
# Additional ROCm tools
# -------------------------------------------------------------------

ROCMINFO_FOUND=false
HIPCONFIG_FOUND=false
ROCM_GPU_AGENT_COUNT=0
HIPCONFIG_VERSION=""


if command -v rocminfo >/dev/null 2>&1; then
    ROCMINFO_FOUND=true

    set +e

    ROCM_GPU_AGENT_COUNT="$(
        rocminfo 2>/dev/null \
        | grep -Ec \
            '^[[:space:]]*Name:[[:space:]]*gfx' \
        || true
    )"

    set -e
fi


if command -v hipconfig >/dev/null 2>&1; then
    HIPCONFIG_FOUND=true

    HIPCONFIG_VERSION="$(
        hipconfig --version \
            2>/dev/null \
        || true
    )"
fi


if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%sROCm installation tools%s\n' \
        "${BOLD}" \
        "${RESET}"

    printf 'rocminfo:       %s\n' \
        "${ROCMINFO_FOUND}"

    printf 'ROCm GPU agents:%s\n' \
        " ${ROCM_GPU_AGENT_COUNT}"

    printf 'hipconfig:      %s\n' \
        "${HIPCONFIG_FOUND}"

    if [[ -n "${HIPCONFIG_VERSION}" ]]; then
        printf 'HIP version:    %s\n' \
            "${HIPCONFIG_VERSION}"
    fi
fi


# -------------------------------------------------------------------
# Export values to the Python diagnostic
# -------------------------------------------------------------------

export OSHA_ROCM_JSON="$(
    if [[ "${JSON_OUTPUT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_ROCM_VERBOSE="$(
    if [[ "${VERBOSE}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_ROCM_SMOKE="$(
    if [[ "${SMOKE}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_ROCM_STRICT="$(
    if [[ "${STRICT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_ROCM_DEVICE="${DEVICE_INDEX}"

export OSHA_ROCM_KFD="${KFD_PRESENT}"

export OSHA_ROCM_RENDER_NODES="${RENDER_NODE_COUNT}"

export OSHA_ROCM_TOOL="${AMD_TOOL}"

export OSHA_ROCM_TOOL_OK="${AMD_TOOL_OK}"

export OSHA_ROCM_TOOL_ERROR="$(
    printf '%s' \
        "${AMD_TOOL_ERROR}"
)"

export OSHA_ROCM_LIST_JSON="$(
    printf '%s' \
        "${AMD_LIST_JSON}"
)"

export OSHA_ROCM_METRICS_JSON="$(
    printf '%s' \
        "${AMD_METRICS_JSON}"
)"

export OSHA_ROCMINFO_FOUND="${ROCMINFO_FOUND}"

export OSHA_ROCM_AGENT_COUNT="${ROCM_GPU_AGENT_COUNT}"

export OSHA_HIPCONFIG_FOUND="${HIPCONFIG_FOUND}"

export OSHA_HIPCONFIG_VERSION="${HIPCONFIG_VERSION}"


# -------------------------------------------------------------------
# PyTorch + OSHA ROCm inspection
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


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

JSON_OUTPUT = (
    os.environ.get(
        "OSHA_ROCM_JSON",
        "0",
    )
    == "1"
)

VERBOSE = (
    os.environ.get(
        "OSHA_ROCM_VERBOSE",
        "0",
    )
    == "1"
)

RUN_SMOKE = (
    os.environ.get(
        "OSHA_ROCM_SMOKE",
        "0",
    )
    == "1"
)

STRICT = (
    os.environ.get(
        "OSHA_ROCM_STRICT",
        "0",
    )
    == "1"
)

DEVICE_TEXT = os.environ.get(
    "OSHA_ROCM_DEVICE",
    "",
).strip()

DEVICE_INDEX = (
    int(DEVICE_TEXT)
    if DEVICE_TEXT
    else None
)

KFD_PRESENT = (
    os.environ.get(
        "OSHA_ROCM_KFD",
        "false",
    )
    == "true"
)

RENDER_NODES = int(
    os.environ.get(
        "OSHA_ROCM_RENDER_NODES",
        "0",
    )
)

AMD_TOOL = os.environ.get(
    "OSHA_ROCM_TOOL",
    "none",
)

AMD_TOOL_OK = (
    os.environ.get(
        "OSHA_ROCM_TOOL_OK",
        "false",
    )
    == "true"
)

AMD_TOOL_ERROR = os.environ.get(
    "OSHA_ROCM_TOOL_ERROR",
    "",
)

ROCMINFO_FOUND = (
    os.environ.get(
        "OSHA_ROCMINFO_FOUND",
        "false",
    )
    == "true"
)

ROCM_AGENT_COUNT = int(
    os.environ.get(
        "OSHA_ROCM_AGENT_COUNT",
        "0",
    )
)

HIPCONFIG_FOUND = (
    os.environ.get(
        "OSHA_HIPCONFIG_FOUND",
        "false",
    )
    == "true"
)

HIPCONFIG_VERSION = os.environ.get(
    "OSHA_HIPCONFIG_VERSION",
    "",
)


def parse_json_or_text(
    value: str,
) -> Any:
    text = value.strip()

    if not text or text == "null":
        return None

    try:
        return json.loads(
            text
        )

    except json.JSONDecodeError:
        return {
            "raw": text,
        }


AMD_LIST = parse_json_or_text(
    os.environ.get(
        "OSHA_ROCM_LIST_JSON",
        "null",
    )
)

AMD_METRICS = parse_json_or_text(
    os.environ.get(
        "OSHA_ROCM_METRICS_JSON",
        "null",
    )
)


# ------------------------------------------------------------------
# Generic helpers
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
            return serialize(
                converter()
            )

        except Exception:
            pass

    return str(
        value
    )


def value_of(
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
    if isinstance(
        value,
        Enum,
    ):
        return str(
            value.value
        )

    if value is None:
        return ""

    return str(
        value
    )


def available_of(
    device: Any,
) -> bool:
    available = value_of(
        device,
        "available",
        None,
    )

    if available is not None:
        return bool(
            available
        )

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

    return (
        f"{number:.2f} "
        f"{units[index]}"
    )


# ------------------------------------------------------------------
# PyTorch HIP runtime
# ------------------------------------------------------------------

hip_version = getattr(
    torch.version,
    "hip",
    None,
)

cuda_version = torch.version.cuda

is_rocm_build = bool(
    hip_version
)


try:
    runtime_available = bool(
        torch.cuda.is_available()
    )

except Exception:
    runtime_available = False


try:
    runtime_device_count = int(
        torch.cuda.device_count()
    )

except Exception:
    runtime_device_count = 0


runtime_devices = []


if is_rocm_build:
    for index in range(
        runtime_device_count
    ):
        if (
            DEVICE_INDEX is not None
            and index != DEVICE_INDEX
        ):
            continue

        item = {
            "index": index,

            # PyTorch HIP intentionally uses
            # cuda:<index> runtime device names.
            "runtime_device": (
                f"cuda:{index}"
            ),
        }

        try:
            item[
                "name"
            ] = torch.cuda.get_device_name(
                index
            )

        except Exception as exc:
            item[
                "name_error"
            ] = type(
                exc
            ).__name__

        try:
            properties = (
                torch.cuda.
                get_device_properties(
                    index
                )
            )

            item.update(
                {
                    "total_memory_bytes": int(
                        properties.total_memory
                    ),

                    "architecture": getattr(
                        properties,
                        "gcnArchName",
                        None,
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
            item[
                "properties_error"
            ] = type(
                exc
            ).__name__

        try:
            free_bytes, total_bytes = (
                torch.cuda.mem_get_info(
                    index
                )
            )

            item[
                "free_memory_bytes"
            ] = int(
                free_bytes
            )

            item[
                "runtime_total_memory_bytes"
            ] = int(
                total_bytes
            )

        except Exception as exc:
            item[
                "memory_error"
            ] = type(
                exc
            ).__name__

        try:
            item[
                "bf16_supported"
            ] = bool(
                torch.cuda.
                is_bf16_supported()
            )

        except Exception:
            item[
                "bf16_supported"
            ] = None

        runtime_devices.append(
            item
        )


# ------------------------------------------------------------------
# Tiny HIP smoke operation
# ------------------------------------------------------------------

smoke = None


if RUN_SMOKE:
    smoke = {
        "requested": True,
        "success": False,
    }

    if not is_rocm_build:
        smoke[
            "reason"
        ] = (
            "Installed PyTorch is not "
            "a ROCm/HIP build."
        )

    elif not runtime_available:
        smoke[
            "reason"
        ] = (
            "PyTorch HIP runtime "
            "is unavailable."
        )

    elif runtime_device_count < 1:
        smoke[
            "reason"
        ] = (
            "PyTorch reports zero "
            "HIP runtime devices."
        )

    elif (
        DEVICE_INDEX is not None
        and DEVICE_INDEX
        >= runtime_device_count
    ):
        smoke[
            "reason"
        ] = (
            "Requested device index "
            "does not exist."
        )

    else:
        index = (
            DEVICE_INDEX
            if DEVICE_INDEX is not None
            else 0
        )

        try:
            # HIP devices use the cuda device
            # namespace in PyTorch.
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

            expected = [
                5.0,
                7.0,
                9.0,
            ]

            if actual != expected:
                raise RuntimeError(
                    "Unexpected HIP tensor "
                    "operation result."
                )

            smoke.update(
                {
                    "success": True,

                    "runtime_device": str(
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
                        type(
                            exc
                        ).__name__
                    ),

                    "message": str(
                        exc
                    ),
                }
            )


# ------------------------------------------------------------------
# OSHA ROCm inventory
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

    rocm_devices = []


    for device in source_devices:
        kind = enum_text(
            value_of(
                device,
                "kind",
                "",
            )
        ).lower()

        if kind != "rocm":
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

        rocm_devices.append(
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


    available_rocm_count = sum(
        1
        for device
        in rocm_devices
        if device[
            "available"
        ]
    )


    payload = {
        "schema_version": "1",

        "requested_device": (
            DEVICE_INDEX
        ),

        "host": {
            "kfd_present": (
                KFD_PRESENT
            ),

            "render_node_count": (
                RENDER_NODES
            ),

            "amd_tool": (
                AMD_TOOL
            ),

            "amd_tool_ok": (
                AMD_TOOL_OK
            ),

            "amd_tool_error": (
                AMD_TOOL_ERROR
                if VERBOSE
                else None
            ),

            "amd_static": (
                AMD_LIST
            ),

            "amd_metrics": (
                AMD_METRICS
            ),

            "rocminfo_found": (
                ROCMINFO_FOUND
            ),

            "rocminfo_gpu_agents": (
                ROCM_AGENT_COUNT
            ),

            "hipconfig_found": (
                HIPCONFIG_FOUND
            ),

            "hipconfig_version": (
                HIPCONFIG_VERSION
            ),
        },

        "pytorch": {
            "version": str(
                torch.__version__
            ),

            "hip_version": (
                hip_version
            ),

            "cuda_version": (
                cuda_version
            ),

            "is_rocm_build": (
                is_rocm_build
            ),

            "runtime_available": (
                runtime_available
            ),

            "device_count": (
                runtime_device_count
            ),

            "devices": (
                runtime_devices
            ),

            "smoke": smoke,
        },

        "osha": {
            "device_count": len(
                rocm_devices
            ),

            "available_device_count": (
                available_rocm_count
            ),

            "devices": (
                rocm_devices
            ),

            "backend_failures": (
                failures
            ),
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
            "PyTorch ROCm/HIP runtime"
        )

        print(
            "------------------------"
        )

        print(
            (
                "PyTorch:          "
                f"{torch.__version__}"
            )
        )

        print(
            (
                "ROCm/HIP build:   "
                f"{is_rocm_build}"
            )
        )

        print(
            (
                "HIP version:      "
                f"{hip_version}"
            )
        )

        print(
            (
                "CUDA version:     "
                f"{cuda_version}"
            )
        )

        print(
            (
                "Runtime available:"
                f" {runtime_available}"
            )
        )

        print(
            (
                "Runtime devices:  "
                f"{runtime_device_count}"
            )
        )


        for device in runtime_devices:
            print()

            print(
                (
                    f"[{device['index']}] "
                    f"{device.get('name')}"
                )
            )

            print(
                (
                    "    Runtime device: "
                    f"{device['runtime_device']}"
                )
            )

            if device.get(
                "architecture"
            ):
                print(
                    (
                        "    Architecture:   "
                        f"{device['architecture']}"
                    )
                )

            if device.get(
                "total_memory_bytes"
            ) is not None:
                print(
                    (
                        "    Total memory:   "
                        f"{human_bytes(device['total_memory_bytes'])}"
                    )
                )

            if device.get(
                "free_memory_bytes"
            ) is not None:
                print(
                    (
                        "    Free memory:    "
                        f"{human_bytes(device['free_memory_bytes'])}"
                    )
                )

            print(
                (
                    "    BF16 supported: "
                    f"{device.get('bf16_supported')}"
                )
            )


        if smoke is not None:
            print()

            print(
                "ROCm smoke test"
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
            "OSHA ROCm inventory"
        )

        print(
            "-------------------"
        )

        print(
            (
                "ROCm devices:      "
                f"{len(rocm_devices)}"
            )
        )

        print(
            (
                "Available devices: "
                f"{available_rocm_count}"
            )
        )

        print(
            (
                "Backend failures:  "
                f"{len(failures)}"
            )
        )


        for device in rocm_devices:
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

            if device.get(
                "architecture"
            ):
                print(
                    (
                        "    Architecture: "
                        f"{device['architecture']}"
                    )
                )

            if device.get(
                "runtime_version"
            ):
                print(
                    (
                        "    Runtime:      "
                        f"{device['runtime_version']}"
                    )
                )

            memory = device[
                "memory"
            ]

            if memory.get(
                "total_bytes"
            ) is not None:
                print(
                    (
                        "    Memory total: "
                        f"{human_bytes(memory['total_bytes'])}"
                    )
                )

            if memory.get(
                "free_bytes"
            ) is not None:
                print(
                    (
                        "    Memory free:  "
                        f"{human_bytes(memory['free_bytes'])}"
                    )
                )


    strict_failures = []


    if STRICT:
        if not KFD_PRESENT:
            strict_failures.append(
                "/dev/kfd is unavailable"
            )

        if RENDER_NODES < 1:
            strict_failures.append(
                "No DRI render nodes are visible"
            )

        if AMD_TOOL == "none":
            strict_failures.append(
                "Neither amd-smi nor rocm-smi "
                "is installed"
            )

        elif not AMD_TOOL_OK:
            strict_failures.append(
                f"{AMD_TOOL} failed"
            )

        if not is_rocm_build:
            strict_failures.append(
                "Installed PyTorch is not "
                "a ROCm/HIP build"
            )

        if not runtime_available:
            strict_failures.append(
                "PyTorch HIP runtime "
                "is unavailable"
            )

        if runtime_device_count < 1:
            strict_failures.append(
                "PyTorch reports zero "
                "HIP runtime devices"
            )

        if available_rocm_count < 1:
            strict_failures.append(
                "OSHA reports no available "
                "ROCm device"
            )

        if (
            DEVICE_INDEX is not None
            and not any(
                device.get(
                    "index"
                )
                == DEVICE_INDEX
                for device
                in rocm_devices
            )
        ):
            strict_failures.append(
                "Requested ROCm device "
                "is not present in OSHA inventory"
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
                "ROCm smoke operation failed"
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

        for value in strict_failures:
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

ROCM_EXIT_CODE=$?

set -e


# -------------------------------------------------------------------
# Final result
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


if [[ "${JSON_OUTPUT}" == true ]]; then
    exit "${ROCM_EXIT_CODE}"
fi


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA ROCm Diagnostic Summary%s\n' \
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

printf 'AMD tool:    %s\n' \
    "${AMD_TOOL}"

printf '/dev/kfd:    %s\n' \
    "${KFD_PRESENT}"

printf 'Render nodes:%s\n' \
    " ${RENDER_NODE_COUNT}"

printf 'Metrics:     %s\n' \
    "${METRICS}"

printf 'Smoke test:  %s\n' \
    "${SMOKE}"

printf 'Strict:      %s\n' \
    "${STRICT}"

printf 'Duration:    %ss\n' \
    "${elapsed}"

printf 'Exit code:   %s\n' \
    "${ROCM_EXIT_CODE}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


if ((ROCM_EXIT_CODE == 0)); then
    printf '\n%sROCM DIAGNOSTICS COMPLETE%s\n' \
        "${GREEN}" \
        "${RESET}"
else
    printf '\n%sROCM DIAGNOSTICS FAILED%s\n' \
        "${RED}" \
        "${RESET}"
fi


exit "${ROCM_EXIT_CODE}"
