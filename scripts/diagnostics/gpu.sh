#!/usr/bin/env bash

#
# OSHA GPU and accelerator diagnostics.
#
# Usage:
#
#   ./scripts/diagnostics/gpu.sh
#   ./scripts/diagnostics/gpu.sh --verbose
#   ./scripts/diagnostics/gpu.sh --strict
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This script is read-only.
#
# It does not:
#
#   install or modify GPU drivers
#   reset GPUs
#   clear accelerator caches
#   allocate large tensors
#   change device visibility
#   modify persistence or power settings
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
STRICT=false

FAILURES=0
WARNINGS=0

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
    WARNINGS=$((WARNINGS + 1))

    printf '%s[WARN]%s %s\n' \
        "${YELLOW}" \
        "${RESET}" \
        "$*" >&2
}


failure() {
    FAILURES=$((FAILURES + 1))

    printf '%s[FAIL]%s %s\n' \
        "${RED}" \
        "${RESET}" \
        "$*" >&2
}


section() {
    printf '\n%s%s== %s ==%s\n' \
        "${BOLD}" \
        "${BLUE}" \
        "$*" \
        "${RESET}"
}


# -------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  scripts/diagnostics/gpu.sh [options]

Options:
  --verbose
      Show additional vendor and runtime output.

  --strict
      Treat detected GPU/runtime mismatches as failures.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override the Python interpreter.

Examples:
  ./scripts/diagnostics/gpu.sh
  ./scripts/diagnostics/gpu.sh --verbose
  ./scripts/diagnostics/gpu.sh --strict
EOF
}


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

while (($# > 0)); do
    case "$1" in
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
# Project
# -------------------------------------------------------------------

cd "${PROJECT_ROOT}"


if [[ ! -d "${PROJECT_ROOT}/src" ]]; then
    failure \
        "src/ directory not found: ${PROJECT_ROOT}/src"

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


PYTHON_VERSION="$(
    "${PYTHON}" -c \
        'import platform; print(platform.python_version())'
)"


export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

printf '\n%sOSHA GPU Diagnostics%s\n' \
    "${BOLD}" \
    "${RESET}"

info "Project root: ${PROJECT_ROOT}"
info "Python: ${PYTHON}"
info "Python version: ${PYTHON_VERSION}"

if [[ "${STRICT}" == true ]]; then
    info "Mode: strict"
else
    info "Mode: normal"
fi


# -------------------------------------------------------------------
# Visibility environment
# -------------------------------------------------------------------

section "Device visibility environment"


print_visibility_variable() {
    local name="$1"
    local value="${!name:-}"

    if [[ -n "${value}" ]]; then
        printf '%-24s %s\n' \
            "${name}" \
            "${value}"
    else
        printf '%-24s %s\n' \
            "${name}" \
            "<unset>"
    fi
}


print_visibility_variable \
    "CUDA_VISIBLE_DEVICES"

print_visibility_variable \
    "HIP_VISIBLE_DEVICES"

print_visibility_variable \
    "ROCR_VISIBLE_DEVICES"

print_visibility_variable \
    "GPU_DEVICE_ORDINAL"


# -------------------------------------------------------------------
# Host PCI devices
# -------------------------------------------------------------------

section "Host PCI display devices"


HOST_NVIDIA=false
HOST_AMD=false
HOST_GPU_LINES=""


if command -v lspci >/dev/null 2>&1; then
    set +e

    HOST_GPU_LINES="$(
        lspci 2>/dev/null \
        | grep -Ei \
            'VGA compatible controller|3D controller|Display controller'
    )"

    LSPCI_RESULT=$?

    set -e


    if [[ -n "${HOST_GPU_LINES}" ]]; then
        printf '%s\n' \
            "${HOST_GPU_LINES}"

        if printf '%s\n' "${HOST_GPU_LINES}" \
            | grep -Eqi 'NVIDIA'
        then
            HOST_NVIDIA=true
        fi

        if printf '%s\n' "${HOST_GPU_LINES}" \
            | grep -Eqi \
                'AMD|Advanced Micro Devices|ATI'
        then
            HOST_AMD=true
        fi

    else
        warn \
            "lspci found no display-class PCI devices"
    fi

else
    warn \
        "lspci is unavailable; PCI GPU detection skipped"
fi


printf '\nHost NVIDIA detected: %s\n' \
    "${HOST_NVIDIA}"

printf 'Host AMD detected:    %s\n' \
    "${HOST_AMD}"


# -------------------------------------------------------------------
# Device nodes
# -------------------------------------------------------------------

section "Linux accelerator device nodes"


DEVICE_NODES_FOUND=false


for pattern in \
    '/dev/nvidia*' \
    '/dev/dri/renderD*' \
    '/dev/kfd'
do
    matches=()

    while IFS= read -r value; do
        if [[ -n "${value}" ]]; then
            matches+=(
                "${value}"
            )
        fi
    done < <(
        compgen -G "${pattern}" \
        || true
    )

    if ((${#matches[@]} > 0)); then
        DEVICE_NODES_FOUND=true

        for value in "${matches[@]}"; do
            if [[ -e "${value}" ]]; then
                ls -l -- "${value}"
            fi
        done
    fi
done


if [[ "${DEVICE_NODES_FOUND}" == false ]]; then
    info "No NVIDIA/DRI/KFD device nodes found"
fi


# -------------------------------------------------------------------
# NVIDIA
# -------------------------------------------------------------------

section "NVIDIA driver visibility"


NVIDIA_SMI_AVAILABLE=false
NVIDIA_DEVICE_COUNT=0


if command -v nvidia-smi >/dev/null 2>&1; then
    NVIDIA_SMI_AVAILABLE=true

    success \
        "nvidia-smi available: $(command -v nvidia-smi)"

    set +e

    NVIDIA_QUERY_OUTPUT="$(
        nvidia-smi \
            --query-gpu=index,name,uuid,driver_version,memory.total,memory.free,temperature.gpu,utilization.gpu \
            --format=csv,noheader,nounits \
            2>&1
    )"

    NVIDIA_QUERY_EXIT=$?

    set -e


    if ((NVIDIA_QUERY_EXIT == 0)); then
        if [[ -n "${NVIDIA_QUERY_OUTPUT}" ]]; then
            NVIDIA_DEVICE_COUNT="$(
                printf '%s\n' \
                    "${NVIDIA_QUERY_OUTPUT}" \
                | sed '/^[[:space:]]*$/d' \
                | wc -l \
                | tr -d '[:space:]'
            )"

            printf 'NVIDIA devices: %s\n\n' \
                "${NVIDIA_DEVICE_COUNT}"

            printf '%s\n' \
                "${NVIDIA_QUERY_OUTPUT}" \
            | while IFS= read -r line; do
                printf '  %s\n' "${line}"
            done

        else
            warn \
                "nvidia-smi returned no devices"
        fi

    else
        if [[ "${HOST_NVIDIA}" == true ]]; then
            if [[ "${STRICT}" == true ]]; then
                failure \
                    "NVIDIA hardware is visible but nvidia-smi failed"
            else
                warn \
                    "NVIDIA hardware is visible but nvidia-smi failed"
            fi

        else
            warn "nvidia-smi failed"
        fi

        if [[ "${VERBOSE}" == true ]]; then
            printf '%s\n' \
                "${NVIDIA_QUERY_OUTPUT}" >&2
        fi
    fi


    if [[ "${VERBOSE}" == true ]]; then
        printf '\n'

        nvidia-smi \
            || true
    fi

else
    if [[ "${HOST_NVIDIA}" == true ]]; then
        if [[ "${STRICT}" == true ]]; then
            failure \
                "NVIDIA hardware detected but nvidia-smi is unavailable"
        else
            warn \
                "NVIDIA hardware detected but nvidia-smi is unavailable"
        fi

    else
        info "nvidia-smi not installed"
    fi
fi


# -------------------------------------------------------------------
# AMD / ROCm
# -------------------------------------------------------------------

section "AMD and ROCm visibility"


AMD_TOOL_AVAILABLE=false
AMD_TOOL_SUCCEEDED=false


if command -v amd-smi >/dev/null 2>&1; then
    AMD_TOOL_AVAILABLE=true

    success \
        "amd-smi available: $(command -v amd-smi)"

    set +e

    AMD_SMI_OUTPUT="$(
        amd-smi static 2>&1
    )"

    AMD_SMI_EXIT=$?

    set -e


    if ((AMD_SMI_EXIT == 0)); then
        AMD_TOOL_SUCCEEDED=true

        printf '%s\n' \
            "${AMD_SMI_OUTPUT}"
    else
        warn "amd-smi failed"

        if [[ "${VERBOSE}" == true ]]; then
            printf '%s\n' \
                "${AMD_SMI_OUTPUT}" >&2
        fi
    fi


elif command -v rocm-smi >/dev/null 2>&1; then
    AMD_TOOL_AVAILABLE=true

    success \
        "rocm-smi available: $(command -v rocm-smi)"

    set +e

    ROCM_SMI_OUTPUT="$(
        rocm-smi \
            --showproductname \
            --showmeminfo vram \
            --showuse \
            --showtemp \
            2>&1
    )"

    ROCM_SMI_EXIT=$?

    set -e


    if ((ROCM_SMI_EXIT == 0)); then
        AMD_TOOL_SUCCEEDED=true

        printf '%s\n' \
            "${ROCM_SMI_OUTPUT}"
    else
        warn "rocm-smi failed"

        if [[ "${VERBOSE}" == true ]]; then
            printf '%s\n' \
                "${ROCM_SMI_OUTPUT}" >&2
        fi
    fi

else
    if [[ "${HOST_AMD}" == true ]]; then
        if [[ "${STRICT}" == true ]]; then
            failure \
                "AMD hardware detected but amd-smi/rocm-smi is unavailable"
        else
            warn \
                "AMD hardware detected but amd-smi/rocm-smi is unavailable"
        fi

    else
        info "amd-smi and rocm-smi not installed"
    fi
fi


# -------------------------------------------------------------------
# ROCm installation
# -------------------------------------------------------------------

section "ROCm installation"


if command -v rocminfo >/dev/null 2>&1; then
    success \
        "rocminfo available: $(command -v rocminfo)"

    if [[ "${VERBOSE}" == true ]]; then
        rocminfo 2>/dev/null \
        | sed -n '1,160p' \
        || true
    else
        set +e

        ROCM_AGENT_COUNT="$(
            rocminfo 2>/dev/null \
            | grep -c \
                '^[[:space:]]*Name:[[:space:]]*gfx' \
            || true
        )"

        set -e

        printf 'ROCm GPU agents: %s\n' \
            "${ROCM_AGENT_COUNT}"
    fi

else
    info "rocminfo not installed"
fi


if command -v hipconfig >/dev/null 2>&1; then
    success \
        "hipconfig available: $(command -v hipconfig)"

    set +e

    HIP_VERSION="$(
        hipconfig --version 2>/dev/null
    )"

    HIP_EXIT=$?

    set -e

    if ((HIP_EXIT == 0)); then
        printf 'HIP version: %s\n' \
            "${HIP_VERSION}"
    fi

else
    info "hipconfig not installed"
fi


# -------------------------------------------------------------------
# PyTorch runtime
# -------------------------------------------------------------------

section "PyTorch accelerator runtime"


if ! "${PYTHON}" -c \
    'import torch' \
    >/dev/null 2>&1
then
    failure \
        "PyTorch cannot be imported by ${PYTHON}"

    PYTORCH_GPU_AVAILABLE=false
    PYTORCH_DEVICE_COUNT=0
    PYTORCH_BACKEND="unavailable"

else
    set +e

    PYTORCH_OUTPUT="$(
        "${PYTHON}" - <<'PY' 2>&1
from __future__ import annotations

import json

import torch


cuda_version = torch.version.cuda

hip_version = getattr(
    torch.version,
    "hip",
    None,
)


if hip_version:
    backend = "rocm"

elif cuda_version:
    backend = "cuda"

else:
    backend = "cpu"


available = bool(
    torch.cuda.is_available()
)


try:
    count = int(
        torch.cuda.device_count()
    )

except Exception:
    count = 0


devices = []


for index in range(count):
    item = {
        "index": index,
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
        ] = type(exc).__name__

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

                "major": int(
                    getattr(
                        properties,
                        "major",
                        0,
                    )
                ),

                "minor": int(
                    getattr(
                        properties,
                        "minor",
                        0,
                    )
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
        ] = type(exc).__name__

    try:
        free_bytes, total_bytes = (
            torch.cuda.mem_get_info(
                index
            )
        )

        item[
            "runtime_free_memory_bytes"
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
        ] = type(exc).__name__

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

    devices.append(
        item
    )


result = {
    "torch_version": str(
        torch.__version__
    ),

    "backend": backend,

    "cuda_version": cuda_version,

    "hip_version": hip_version,

    "available": available,

    "initialized": bool(
        torch.cuda.is_initialized()
    ),

    "device_count": count,

    "devices": devices,
}


print(
    json.dumps(
        result,
        sort_keys=True,
    )
)
PY
    )"

    PYTORCH_EXIT=$?

    set -e


    if ((PYTORCH_EXIT != 0)); then
        failure \
            "PyTorch accelerator inspection failed"

        printf '%s\n' \
            "${PYTORCH_OUTPUT}" >&2

        PYTORCH_GPU_AVAILABLE=false
        PYTORCH_DEVICE_COUNT=0
        PYTORCH_BACKEND="error"

    else
        PYTORCH_BACKEND="$(
            printf '%s' \
                "${PYTORCH_OUTPUT}" \
            | "${PYTHON}" -c \
                'import json, sys; print(json.load(sys.stdin)["backend"])'
        )"

        PYTORCH_GPU_AVAILABLE="$(
            printf '%s' \
                "${PYTORCH_OUTPUT}" \
            | "${PYTHON}" -c \
                'import json, sys; print(str(json.load(sys.stdin)["available"]).lower())'
        )"

        PYTORCH_DEVICE_COUNT="$(
            printf '%s' \
                "${PYTORCH_OUTPUT}" \
            | "${PYTHON}" -c \
                'import json, sys; print(json.load(sys.stdin)["device_count"])'
        )"

        printf '%s' \
            "${PYTORCH_OUTPUT}" \
        | "${PYTHON}" -m json.tool

        if [[ "${PYTORCH_GPU_AVAILABLE}" == "true" ]]; then
            success \
                "PyTorch accelerator runtime available"
        else
            info \
                "PyTorch reports no available GPU runtime"
        fi
    fi
fi


# -------------------------------------------------------------------
# Runtime smoke operation
# -------------------------------------------------------------------

section "PyTorch smoke operation"


if [[ "${PYTORCH_GPU_AVAILABLE:-false}" == "true" ]]; then
    set +e

    SMOKE_OUTPUT="$(
        "${PYTHON}" - <<'PY' 2>&1
from __future__ import annotations

import torch


device = torch.device(
    "cuda:0"
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


expected = [
    5.0,
    7.0,
    9.0,
]


actual = result.cpu().tolist()


if actual != expected:
    raise RuntimeError(
        f"Unexpected accelerator result: {actual}"
    )


print(
    "device="
    + str(device)
)

print(
    "result="
    + str(actual)
)
PY
    )"

    SMOKE_EXIT=$?

    set -e


    if ((SMOKE_EXIT == 0)); then
        success \
            "Small PyTorch accelerator operation"

        printf '%s\n' \
            "${SMOKE_OUTPUT}"
    else
        if [[ "${STRICT}" == true ]]; then
            failure \
                "Small PyTorch accelerator operation failed"
        else
            warn \
                "Small PyTorch accelerator operation failed"
        fi

        printf '%s\n' \
            "${SMOKE_OUTPUT}" >&2
    fi

else
    info \
        "GPU smoke operation skipped because no PyTorch GPU runtime is available"
fi


# -------------------------------------------------------------------
# OSHA AcceleratorManager
# -------------------------------------------------------------------

section "OSHA AcceleratorManager discovery"


set +e

OSHA_OUTPUT="$(
    "${PYTHON}" - <<'PY' 2>&1
from __future__ import annotations

import json

from src.accelerators import (
    AcceleratorManager,
)


manager = AcceleratorManager()


try:
    inventory = manager.inventory(
        refresh=True
    )

    devices = getattr(
        inventory,
        "devices",
        (),
    )

    payload = {
        "device_count": len(
            devices
        ),

        "devices": [
            (
                device.to_dict()
                if callable(
                    getattr(
                        device,
                        "to_dict",
                        None,
                    )
                )
                else {
                    "device_id": getattr(
                        device,
                        "device_id",
                        None,
                    ),

                    "kind": str(
                        getattr(
                            device,
                            "kind",
                            None,
                        )
                    ),

                    "name": getattr(
                        device,
                        "name",
                        None,
                    ),

                    "available": getattr(
                        device,
                        "available",
                        None,
                    ),

                    "runtime_device": getattr(
                        device,
                        "runtime_device",
                        None,
                    ),
                }
            )
            for device in devices
        ],

        "failures": [
            (
                failure.to_dict()
                if callable(
                    getattr(
                        failure,
                        "to_dict",
                        None,
                    )
                )
                else {
                    "backend": getattr(
                        failure,
                        "backend",
                        None,
                    ),

                    "error_type": getattr(
                        failure,
                        "error_type",
                        None,
                    ),
                }
            )
            for failure in (
                getattr(
                    inventory,
                    "failures",
                    (),
                )
                or ()
            )
        ],
    }

    print(
        json.dumps(
            payload,
            sort_keys=True,
            default=str,
        )
    )

finally:
    try:
        manager.release_all()

    except Exception:
        pass
PY
)"

OSHA_EXIT=$?

set -e


if ((OSHA_EXIT == 0)); then
    success \
        "OSHA accelerator inventory"

    printf '%s' \
        "${OSHA_OUTPUT}" \
    | "${PYTHON}" -m json.tool

    OSHA_DEVICE_COUNT="$(
        printf '%s' \
            "${OSHA_OUTPUT}" \
        | "${PYTHON}" -c \
            'import json, sys; print(json.load(sys.stdin)["device_count"])'
    )"

else
    failure \
        "OSHA AcceleratorManager discovery failed"

    printf '%s\n' \
        "${OSHA_OUTPUT}" >&2

    OSHA_DEVICE_COUNT=0
fi


# -------------------------------------------------------------------
# Consistency checks
# -------------------------------------------------------------------

section "Consistency checks"


if [[ "${HOST_NVIDIA}" == true ]] \
    && [[ "${NVIDIA_SMI_AVAILABLE}" == false ]]; then
    if [[ "${STRICT}" == true ]]; then
        failure \
            "Host NVIDIA hardware exists but NVIDIA driver tooling is unavailable"
    else
        warn \
            "Host NVIDIA hardware exists but NVIDIA driver tooling is unavailable"
    fi
fi


if [[ "${HOST_AMD}" == true ]] \
    && [[ "${AMD_TOOL_AVAILABLE}" == false ]]; then
    if [[ "${STRICT}" == true ]]; then
        failure \
            "Host AMD display hardware exists but AMD/ROCm tooling is unavailable"
    else
        warn \
            "Host AMD display hardware exists but AMD/ROCm tooling is unavailable"
    fi
fi


if ((NVIDIA_DEVICE_COUNT > 0)) \
    && [[ "${PYTORCH_BACKEND:-cpu}" == "cpu" ]]; then
    if [[ "${STRICT}" == true ]]; then
        failure \
            "NVIDIA driver sees GPUs but installed PyTorch is CPU-only"
    else
        warn \
            "NVIDIA driver sees GPUs but installed PyTorch is CPU-only"
    fi
fi


if [[ "${AMD_TOOL_SUCCEEDED}" == true ]] \
    && [[ "${PYTORCH_BACKEND:-cpu}" != "rocm" ]]; then
    if [[ "${STRICT}" == true ]]; then
        failure \
            "AMD/ROCm tooling is available but PyTorch is not a ROCm build"
    else
        warn \
            "AMD/ROCm tooling is available but PyTorch is not a ROCm build"
    fi
fi


if [[ "${PYTORCH_GPU_AVAILABLE:-false}" == "true" ]] \
    && ((PYTORCH_DEVICE_COUNT == 0)); then
    failure \
        "PyTorch reports accelerator availability but zero devices"
fi


if ((OSHA_DEVICE_COUNT > 0)); then
    success \
        "OSHA discovered ${OSHA_DEVICE_COUNT} accelerator/runtime devices"
else
    warn \
        "OSHA discovered no devices"
fi


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA GPU Diagnostic Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Host NVIDIA:       %s\n' \
    "${HOST_NVIDIA}"

printf 'Host AMD:          %s\n' \
    "${HOST_AMD}"

printf 'nvidia-smi GPUs:   %s\n' \
    "${NVIDIA_DEVICE_COUNT}"

printf 'PyTorch backend:   %s\n' \
    "${PYTORCH_BACKEND:-unknown}"

printf 'PyTorch available: %s\n' \
    "${PYTORCH_GPU_AVAILABLE:-false}"

printf 'PyTorch devices:   %s\n' \
    "${PYTORCH_DEVICE_COUNT:-0}"

printf 'OSHA devices:      %s\n' \
    "${OSHA_DEVICE_COUNT}"

printf 'Warnings:          %s\n' \
    "${WARNINGS}"

printf 'Failures:          %s\n' \
    "${FAILURES}"

printf 'Mode:              %s\n' \
    "$(
        if [[ "${STRICT}" == true ]]; then
            printf 'strict'
        else
            printf 'normal'
        fi
    )"

printf 'Duration:          %ss\n' \
    "${elapsed}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


# -------------------------------------------------------------------
# Final result
# -------------------------------------------------------------------

if ((FAILURES > 0)); then
    printf '\n%sGPU DIAGNOSTICS FAILED%s\n' \
        "${RED}" \
        "${RESET}"

    exit 1
fi


if ((WARNINGS > 0)); then
    printf '\n%sGPU DIAGNOSTICS COMPLETED WITH WARNINGS%s\n' \
        "${YELLOW}" \
        "${RESET}"

else
    printf '\n%sGPU DIAGNOSTICS PASSED%s\n' \
        "${GREEN}" \
        "${RESET}"
fi


exit 0
