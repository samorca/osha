#!/usr/bin/env bash

#
# OSHA CPU accelerator diagnostics.
#
# Usage:
#
#   ./scripts/accelerators/cpu.sh
#   ./scripts/accelerators/cpu.sh --json
#   ./scripts/accelerators/cpu.sh --verbose
#   ./scripts/accelerators/cpu.sh --metrics
#   ./scripts/accelerators/cpu.sh --strict
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# Read-only:
#
#   This script does not change CPU affinity,
#   governors, NUMA placement, thread counts,
#   process priority, or kernel settings.
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
  scripts/accelerators/cpu.sh [options]

Options:
  --json
      Emit machine-readable JSON.

  --verbose
      Include additional normalized CPU details.

  --metrics
      Collect a short live CPU utilization sample.

  --strict
      Fail if OSHA cannot discover an available CPU device.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override Python interpreter.

Examples:
  ./scripts/accelerators/cpu.sh

  ./scripts/accelerators/cpu.sh --metrics

  ./scripts/accelerators/cpu.sh --verbose

  ./scripts/accelerators/cpu.sh --json

  ./scripts/accelerators/cpu.sh --strict
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

        --metrics)
            METRICS=true
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


if [[ ! -d src/accelerators ]]; then
    failure "src/accelerators does not exist."
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
# Validate core imports
# -------------------------------------------------------------------

if ! "${PYTHON}" -c \
    'from src.accelerators import AcceleratorManager' \
    >/dev/null 2>&1
then
    failure \
        "Unable to import AcceleratorManager"

    exit 1
fi


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%sOSHA CPU Accelerator Diagnostics%s\n' \
        "${BOLD}" \
        "${RESET}"

    info "Project root: ${PROJECT_ROOT}"
    info "Python: ${PYTHON}"
    info "Python version: ${PYTHON_VERSION}"

    if [[ "${METRICS}" == true ]]; then
        info "Live metrics: enabled"
    fi

    if [[ "${STRICT}" == true ]]; then
        info "Mode: strict"
    else
        info "Mode: normal"
    fi
fi


# -------------------------------------------------------------------
# Host CPU summary
# -------------------------------------------------------------------

if [[ "${JSON_OUTPUT}" == false ]]; then
    printf '\n%sHost CPU%s\n' \
        "${BOLD}"

    printf '%s' \
        "${RESET}"


    if command -v lscpu >/dev/null 2>&1; then
        if [[ "${VERBOSE}" == true ]]; then
            lscpu
        else
            lscpu \
            | grep -E \
                '^(Architecture|CPU\(s\)|On-line CPU|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|Model name|NUMA node\(s\)|Vendor ID):' \
            || true
        fi

    else
        warn "lscpu is not installed"
    fi
fi


# -------------------------------------------------------------------
# Export controls for Python
# -------------------------------------------------------------------

export OSHA_CPU_JSON="$(
    if [[ "${JSON_OUTPUT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_CPU_VERBOSE="$(
    if [[ "${VERBOSE}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_CPU_METRICS="$(
    if [[ "${METRICS}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"

export OSHA_CPU_STRICT="$(
    if [[ "${STRICT}" == true ]]; then
        printf '1'
    else
        printf '0'
    fi
)"


# -------------------------------------------------------------------
# OSHA CPU inspection
# -------------------------------------------------------------------

set +e

"${PYTHON}" - <<'PY'
from __future__ import annotations

import importlib.util
import json
import os
import platform

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


JSON_OUTPUT = (
    os.environ.get(
        "OSHA_CPU_JSON",
        "0",
    )
    == "1"
)

VERBOSE = (
    os.environ.get(
        "OSHA_CPU_VERBOSE",
        "0",
    )
    == "1"
)

COLLECT_METRICS = (
    os.environ.get(
        "OSHA_CPU_METRICS",
        "0",
    )
    == "1"
)

STRICT = (
    os.environ.get(
        "OSHA_CPU_STRICT",
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
    if isinstance(value, Enum):
        return str(value.value)

    if value is None:
        return ""

    return str(value)


def available_of(
    device: Any,
) -> bool:
    value = value_of(
        device,
        "available",
        None,
    )

    if value is not None:
        return bool(value)

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


def bytes_human(
    value: Any,
) -> str:
    if value is None:
        return "-"

    try:
        number = float(value)

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

    unit = 0

    while (
        number >= 1024
        and unit < len(units) - 1
    ):
        number /= 1024
        unit += 1

    if unit == 0:
        return (
            f"{int(number)} "
            f"{units[unit]}"
        )

    return (
        f"{number:.2f} "
        f"{units[unit]}"
    )


# ------------------------------------------------------------------
# Host information
# ------------------------------------------------------------------

host = {
    "architecture": (
        platform.machine()
    ),

    "processor": (
        platform.processor()
    ),

    "logical_cpu_count": (
        os.cpu_count()
    ),

    "load_average": None,

    "affinity_cpu_count": None,
}


try:
    host["load_average"] = list(
        os.getloadavg()
    )

except (
    AttributeError,
    OSError,
):
    pass


try:
    host[
        "affinity_cpu_count"
    ] = len(
        os.sched_getaffinity(0)
    )

except (
    AttributeError,
    OSError,
):
    pass


# ------------------------------------------------------------------
# psutil metrics
# ------------------------------------------------------------------

metrics = None


if COLLECT_METRICS:
    if importlib.util.find_spec(
        "psutil"
    ) is None:
        metrics = {
            "available": False,
            "reason": (
                "psutil is not installed"
            ),
        }

    else:
        import psutil

        logical = psutil.cpu_count(
            logical=True
        )

        physical = psutil.cpu_count(
            logical=False
        )

        frequency = psutil.cpu_freq()

        stats = psutil.cpu_stats()

        # A non-zero interval provides an observed
        # utilization sample rather than relying on
        # the first immediate call.
        utilization = psutil.cpu_percent(
            interval=1.0,
            percpu=False,
        )

        per_cpu = psutil.cpu_percent(
            interval=0.25,
            percpu=True,
        )

        metrics = {
            "available": True,

            "logical_cpu_count": (
                logical
            ),

            "physical_cpu_count": (
                physical
            ),

            "utilization_percent": (
                utilization
            ),

            "per_cpu_percent": (
                per_cpu
            ),

            "frequency_mhz": (
                {
                    "current": (
                        frequency.current
                    ),

                    "minimum": (
                        frequency.min
                    ),

                    "maximum": (
                        frequency.max
                    ),
                }
                if frequency is not None
                else None
            ),

            "stats": (
                stats._asdict()
                if stats is not None
                else None
            ),

            "load_average": (
                list(
                    psutil.getloadavg()
                )
                if hasattr(
                    psutil,
                    "getloadavg",
                )
                else None
            ),
        }


# ------------------------------------------------------------------
# OSHA inventory
# ------------------------------------------------------------------

manager = AcceleratorManager()


try:
    inventory = manager.inventory(
        refresh=True
    )

    all_devices = tuple(
        value_of(
            inventory,
            "devices",
            (),
        )
        or ()
    )

    cpu_devices = [
        device
        for device in all_devices
        if enum_text(
            value_of(
                device,
                "kind",
                "",
            )
        ).lower()
        == "cpu"
    ]

    normalized = []


    for device in cpu_devices:
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

        entry = {
            "device_id": value_of(
                device,
                "device_id",
                None,
            ),

            "name": value_of(
                device,
                "name",
                None,
            ),

            "kind": "cpu",

            "runtime_device": (
                value_of(
                    device,
                    "runtime_device",
                    None,
                )
            ),

            "state": enum_text(
                value_of(
                    device,
                    "state",
                    "unknown",
                )
            ),

            "available": (
                available_of(
                    device
                )
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

            "memory": memory,

            "capabilities": serialize(
                value_of(
                    device,
                    "capabilities",
                    (),
                )
            ),

            "supported_precisions": (
                serialize(
                    value_of(
                        device,
                        "supported_precisions",
                        (),
                    )
                )
            ),
        }

        if VERBOSE:
            attributes = serialize(
                value_of(
                    device,
                    "attributes",
                    {},
                )
            )

            entry[
                "attributes"
            ] = (
                attributes
                if isinstance(
                    attributes,
                    dict,
                )
                else {}
            )

        normalized.append(
            entry
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


    payload = {
        "schema_version": "1",

        "host": host,

        "summary": {
            "cpu_device_count": len(
                normalized
            ),

            "available_cpu_device_count": sum(
                1
                for device
                in normalized
                if device[
                    "available"
                ]
            ),

            "backend_failure_count": len(
                failures
            ),
        },

        "devices": normalized,

        "metrics": metrics,

        "backend_failures": failures,
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
            "OSHA CPU inventory"
        )

        print(
            "------------------"
        )

        print(
            (
                "CPU devices:       "
                f"{len(normalized)}"
            )
        )

        print(
            (
                "Available devices: "
                f"{payload['summary']['available_cpu_device_count']}"
            )
        )

        print(
            (
                "Logical CPUs:      "
                f"{host['logical_cpu_count']}"
            )
        )

        print(
            (
                "Affinity CPUs:     "
                f"{host['affinity_cpu_count']}"
            )
        )

        print(
            (
                "Architecture:      "
                f"{host['architecture']}"
            )
        )

        print(
            (
                "Backend failures:  "
                f"{len(failures)}"
            )
        )


        for index, device in enumerate(
            normalized,
            start=1,
        ):
            print()

            print(
                (
                    f"[{index}] "
                    f"{device['device_id']}"
                )
            )

            print(
                (
                    "    Name:        "
                    f"{device['name']}"
                )
            )

            print(
                (
                    "    Runtime:     "
                    f"{device['runtime_device']}"
                )
            )

            print(
                (
                    "    State:       "
                    f"{device['state']}"
                )
            )

            print(
                (
                    "    Available:   "
                    f"{device['available']}"
                )
            )

            if device[
                "vendor"
            ]:
                print(
                    (
                        "    Vendor:      "
                        f"{device['vendor']}"
                    )
                )

            if device[
                "architecture"
            ]:
                print(
                    (
                        "    Architecture:"
                        f" {device['architecture']}"
                    )
                )

            memory = device[
                "memory"
            ]

            total = memory.get(
                "total_bytes"
            )

            free = memory.get(
                "free_bytes"
            )

            if total is not None:
                print(
                    (
                        "    Memory total:"
                        f" {bytes_human(total)}"
                    )
                )

            if free is not None:
                print(
                    (
                        "    Memory free: "
                        f"{bytes_human(free)}"
                    )
                )

            if device[
                "supported_precisions"
            ]:
                print(
                    (
                        "    Precisions:  "
                        + ", ".join(
                            str(value)
                            for value
                            in device[
                                "supported_precisions"
                            ]
                        )
                    )
                )

            if device[
                "capabilities"
            ]:
                print(
                    (
                        "    Capabilities:"
                        " "
                        + ", ".join(
                            str(value)
                            for value
                            in device[
                                "capabilities"
                            ]
                        )
                    )
                )


        if metrics is not None:
            print()

            print(
                "Live CPU metrics"
            )

            print(
                "----------------"
            )

            if not metrics.get(
                "available",
                False,
            ):
                print(
                    metrics.get(
                        "reason",
                        "Metrics unavailable.",
                    )
                )

            else:
                print(
                    (
                        "Logical CPUs:      "
                        f"{metrics['logical_cpu_count']}"
                    )
                )

                print(
                    (
                        "Physical CPUs:     "
                        f"{metrics['physical_cpu_count']}"
                    )
                )

                print(
                    (
                        "CPU utilization:   "
                        f"{metrics['utilization_percent']}%"
                    )
                )

                frequency = metrics.get(
                    "frequency_mhz"
                )

                if frequency:
                    print(
                        (
                            "Frequency current:"
                            " "
                            f"{frequency['current']} MHz"
                        )
                    )

                    print(
                        (
                            "Frequency min:    "
                            f"{frequency['minimum']} MHz"
                        )
                    )

                    print(
                        (
                            "Frequency max:    "
                            f"{frequency['maximum']} MHz"
                        )
                    )

                if VERBOSE:
                    print(
                        (
                            "Per-CPU percent:  "
                            f"{metrics['per_cpu_percent']}"
                        )
                    )

                    print(
                        (
                            "CPU stats:        "
                            f"{metrics['stats']}"
                        )
                    )


        if failures:
            print()

            print(
                "Backend failures"
            )

            print(
                "----------------"
            )

            print(
                json.dumps(
                    failures,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )


    available_cpu_count = (
        payload[
            "summary"
        ][
            "available_cpu_device_count"
        ]
    )


    if (
        STRICT
        and available_cpu_count < 1
    ):
        raise SystemExit(1)


    raise SystemExit(0)


finally:
    release_all = getattr(
        manager,
        "release_all",
        None,
    )

    if callable(release_all):
        try:
            release_all()

        except Exception:
            pass
PY

CPU_EXIT_CODE=$?

set -e


# -------------------------------------------------------------------
# Result
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


if [[ "${JSON_OUTPUT}" == true ]]; then
    exit "${CPU_EXIT_CODE}"
fi


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA CPU Diagnostic Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Python:   %s\n' \
    "${PYTHON_VERSION}"

printf 'Metrics:  %s\n' \
    "${METRICS}"

printf 'Strict:   %s\n' \
    "${STRICT}"

printf 'Duration: %ss\n' \
    "${elapsed}"

printf 'Exit code:%s\n' \
    " ${CPU_EXIT_CODE}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


if ((CPU_EXIT_CODE == 0)); then
    printf '\n%sCPU DIAGNOSTICS COMPLETE%s\n' \
        "${GREEN}" \
        "${RESET}"
else
    printf '\n%sCPU DIAGNOSTICS FAILED%s\n' \
        "${RED}" \
        "${RESET}"
fi


exit "${CPU_EXIT_CODE}"
