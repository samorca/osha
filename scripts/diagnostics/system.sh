#!/usr/bin/env bash

#
# OSHA host-system diagnostics.
#
# Usage:
#
#   ./scripts/diagnostics/system.sh
#   ./scripts/diagnostics/system.sh --verbose
#   ./scripts/diagnostics/system.sh --strict
#
# Environment:
#
#   OSHA_PYTHON=/path/to/python
#
# This script is read-only.
#
# It inspects:
#
#   operating system
#   kernel
#   CPU topology
#   memory and swap
#   load
#   pressure stall information
#   filesystems
#   project filesystem
#   cgroups
#   virtualization/container environment
#   process limits
#   file descriptors
#   clock/timezone
#   selected kernel/runtime limits
#
# It does NOT:
#
#   modify sysctl
#   change ulimits
#   mount filesystems
#   modify swap
#   change cgroups
#   restart services
#   install packages
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

WARNINGS=0
FAILURES=0

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
  scripts/diagnostics/system.sh [options]

Options:
  --verbose
      Show additional host-level diagnostics.

  --strict
      Treat important host inconsistencies as failures.

  -h, --help
      Show this help.

Environment:
  OSHA_PYTHON
      Override Python interpreter.

Examples:
  ./scripts/diagnostics/system.sh
  ./scripts/diagnostics/system.sh --verbose
  ./scripts/diagnostics/system.sh --strict
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
    PYTHON=""
fi


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

printf '\n%sOSHA System Diagnostics%s\n' \
    "${BOLD}" \
    "${RESET}"

info "Project root: ${PROJECT_ROOT}"

if [[ -n "${PYTHON}" ]]; then
    info "Python: ${PYTHON}"
fi

if [[ "${STRICT}" == true ]]; then
    info "Mode: strict"
else
    info "Mode: normal"
fi


# -------------------------------------------------------------------
# Operating system
# -------------------------------------------------------------------

section "Operating system"


if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release

    printf 'Distribution: %s\n' \
        "${PRETTY_NAME:-unknown}"

    printf 'ID:           %s\n' \
        "${ID:-unknown}"

    printf 'Version ID:   %s\n' \
        "${VERSION_ID:-unknown}"

else
    warn "/etc/os-release is unavailable"
fi


printf 'Kernel:       %s\n' \
    "$(uname -r)"

printf 'Kernel name:  %s\n' \
    "$(uname -s)"

printf 'Machine:      %s\n' \
    "$(uname -m)"

printf 'Hostname:     %s\n' \
    "$(hostname 2>/dev/null || printf 'unknown')"


if [[ "${VERBOSE}" == true ]]; then
    printf '\n'

    uname -a
fi


# -------------------------------------------------------------------
# Boot and uptime
# -------------------------------------------------------------------

section "Uptime and load"


if command -v uptime >/dev/null 2>&1; then
    uptime
else
    warn "uptime command unavailable"
fi


if [[ -r /proc/uptime ]]; then
    UPTIME_SECONDS="$(
        awk '{
            printf "%.0f", $1
        }' /proc/uptime
    )"

    printf 'Uptime seconds: %s\n' \
        "${UPTIME_SECONDS}"
fi


if [[ -r /proc/loadavg ]]; then
    printf 'Load average:   '

    cat /proc/loadavg
fi


# -------------------------------------------------------------------
# CPU
# -------------------------------------------------------------------

section "CPU"


if command -v nproc >/dev/null 2>&1; then
    printf 'Available CPUs: %s\n' \
        "$(nproc)"

    printf 'Installed CPUs: %s\n' \
        "$(nproc --all)"
fi


if command -v lscpu >/dev/null 2>&1; then
    if [[ "${VERBOSE}" == true ]]; then
        lscpu

    else
        lscpu \
        | grep -E \
            '^(Architecture|CPU\(s\)|On-line CPU|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|Model name|NUMA node\(s\)|Virtualization|Hypervisor vendor):' \
        || true
    fi

else
    warn "lscpu unavailable"
fi


if [[ -r /proc/cpuinfo ]]; then
    CPU_MODEL="$(
        grep -m1 \
            -E '^model name[[:space:]]*:' \
            /proc/cpuinfo \
        | cut -d: -f2- \
        | sed 's/^[[:space:]]*//' \
        || true
    )"

    if [[ -n "${CPU_MODEL}" ]]; then
        printf 'CPU model: %s\n' \
            "${CPU_MODEL}"
    fi
fi


# -------------------------------------------------------------------
# Memory
# -------------------------------------------------------------------

section "Memory"


if command -v free >/dev/null 2>&1; then
    free -h
else
    warn "free command unavailable"
fi


if [[ -r /proc/meminfo ]]; then
    printf '\nSelected /proc/meminfo values:\n'

    grep -E \
        '^(MemTotal|MemAvailable|SwapTotal|SwapFree|HugePages_Total|HugePages_Free|Hugepagesize):' \
        /proc/meminfo \
    || true
fi


# -------------------------------------------------------------------
# Swap
# -------------------------------------------------------------------

section "Swap"


if command -v swapon >/dev/null 2>&1; then
    set +e

    SWAP_OUTPUT="$(
        swapon \
            --show \
            --bytes \
            2>/dev/null
    )"

    SWAP_EXIT=$?

    set -e

    if ((SWAP_EXIT == 0)) \
        && [[ -n "${SWAP_OUTPUT}" ]]; then
        printf '%s\n' \
            "${SWAP_OUTPUT}"

    else
        info "No active swap reported"
    fi

else
    info "swapon command unavailable"
fi


# -------------------------------------------------------------------
# Filesystems
# -------------------------------------------------------------------

section "Filesystems"


if command -v df >/dev/null 2>&1; then
    df -hT
else
    warn "df command unavailable"
fi


# -------------------------------------------------------------------
# Project filesystem
# -------------------------------------------------------------------

section "OSHA project filesystem"


if command -v df >/dev/null 2>&1; then
    df -hT "${PROJECT_ROOT}"
fi


if command -v stat >/dev/null 2>&1; then
    printf '\nProject filesystem metadata:\n'

    stat \
        -f \
        -c \
        'Filesystem type: %T
Block size:      %S
Total blocks:    %b
Free blocks:     %f
Available:       %a' \
        "${PROJECT_ROOT}" \
        2>/dev/null \
    || true
fi


if [[ -w "${PROJECT_ROOT}" ]]; then
    success "Project root is writable"
else
    if [[ "${STRICT}" == true ]]; then
        failure "Project root is not writable"
    else
        warn "Project root is not writable"
    fi
fi


# -------------------------------------------------------------------
# Inodes
# -------------------------------------------------------------------

section "Filesystem inodes"


if command -v df >/dev/null 2>&1; then
    df -ih "${PROJECT_ROOT}"
else
    info "df unavailable"
fi


# -------------------------------------------------------------------
# Disk/block devices
# -------------------------------------------------------------------

section "Block devices"


if command -v lsblk >/dev/null 2>&1; then
    lsblk \
        -o \
        NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS \
        2>/dev/null \
    || lsblk

else
    info "lsblk unavailable"
fi


# -------------------------------------------------------------------
# Pressure stall information
# -------------------------------------------------------------------

section "Linux pressure"


PSI_AVAILABLE=false


for resource in \
    cpu \
    memory \
    io
do
    path="/proc/pressure/${resource}"

    if [[ -r "${path}" ]]; then
        PSI_AVAILABLE=true

        printf '%s:\n' \
            "${resource}"

        cat "${path}"

        printf '\n'
    fi
done


if [[ "${PSI_AVAILABLE}" == false ]]; then
    info \
        "Linux pressure stall information is unavailable"
fi


# -------------------------------------------------------------------
# Cgroups
# -------------------------------------------------------------------

section "Control groups"


CGROUP_VERSION="unknown"


if [[ -f /sys/fs/cgroup/cgroup.controllers ]]; then
    CGROUP_VERSION="v2"

elif [[ -d /sys/fs/cgroup ]]; then
    CGROUP_VERSION="v1-or-hybrid"
fi


printf 'Cgroup version: %s\n' \
    "${CGROUP_VERSION}"


if [[ -r /proc/self/cgroup ]]; then
    printf '\nCurrent process cgroups:\n'

    cat /proc/self/cgroup
fi


if [[ "${CGROUP_VERSION}" == "v2" ]]; then
    printf '\nControllers:\n'

    cat /sys/fs/cgroup/cgroup.controllers \
        2>/dev/null \
    || true
fi


# -------------------------------------------------------------------
# Container / virtualization
# -------------------------------------------------------------------

section "Virtualization and container context"


VIRTUALIZATION="unknown"


if command -v systemd-detect-virt \
    >/dev/null 2>&1
then
    set +e

    VIRTUALIZATION="$(
        systemd-detect-virt 2>/dev/null
    )"

    VIRT_EXIT=$?

    set -e

    if ((VIRT_EXIT != 0)); then
        VIRTUALIZATION="none"
    fi

    printf 'Detected: %s\n' \
        "${VIRTUALIZATION}"

else
    info "systemd-detect-virt unavailable"
fi


CONTAINER_HINT="false"


if [[ -f /.dockerenv ]]; then
    CONTAINER_HINT="docker"
fi


if [[ -r /proc/1/cgroup ]] \
    && grep -Eq \
        'docker|containerd|kubepods|podman|lxc' \
        /proc/1/cgroup
then
    CONTAINER_HINT="true"
fi


printf 'Container hint: %s\n' \
    "${CONTAINER_HINT}"


# -------------------------------------------------------------------
# Process limits
# -------------------------------------------------------------------

section "Process limits"


printf 'Open files soft: '

ulimit -Sn


printf 'Open files hard: '

ulimit -Hn


printf 'Processes soft:  '

ulimit -Su 2>/dev/null \
    || printf 'unavailable\n'


printf 'Processes hard:  '

ulimit -Hu 2>/dev/null \
    || printf 'unavailable\n'


printf 'Locked memory:   '

ulimit -l 2>/dev/null \
    || printf 'unavailable\n'


printf 'Stack size:      '

ulimit -s 2>/dev/null \
    || printf 'unavailable\n'


if [[ "${VERBOSE}" == true ]]; then
    printf '\nFull shell limits:\n'

    ulimit -a
fi


# -------------------------------------------------------------------
# Kernel file descriptor capacity
# -------------------------------------------------------------------

section "File descriptors"


if [[ -r /proc/sys/fs/file-max ]]; then
    printf 'System file-max: '

    cat /proc/sys/fs/file-max
fi


if [[ -r /proc/sys/fs/file-nr ]]; then
    printf 'System file-nr:  '

    cat /proc/sys/fs/file-nr
fi


if [[ -r /proc/self/limits ]]; then
    printf '\nCurrent process:\n'

    grep -E \
        'Max open files|Max processes|Max locked memory|Max stack size' \
        /proc/self/limits \
    || true
fi


# -------------------------------------------------------------------
# Kernel memory/runtime parameters
# -------------------------------------------------------------------

section "Selected kernel runtime limits"


KERNEL_PARAMETERS=(
    "/proc/sys/vm/max_map_count"
    "/proc/sys/vm/swappiness"
    "/proc/sys/kernel/pid_max"
    "/proc/sys/kernel/threads-max"
    "/proc/sys/fs/inotify/max_user_watches"
    "/proc/sys/fs/inotify/max_user_instances"
)


for parameter in "${KERNEL_PARAMETERS[@]}"; do
    if [[ -r "${parameter}" ]]; then
        printf '%-48s ' \
            "${parameter}"

        cat "${parameter}"
    fi
done


# -------------------------------------------------------------------
# Clock and timezone
# -------------------------------------------------------------------

section "Clock"


printf 'UTC:   %s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

printf 'Local: %s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')"


if command -v timedatectl >/dev/null 2>&1; then
    set +e

    TIME_OUTPUT="$(
        timedatectl \
            show \
            --property=Timezone \
            --property=NTPSynchronized \
            --property=NTP \
            --property=LocalRTC \
            2>/dev/null
    )"

    TIME_EXIT=$?

    set -e

    if ((TIME_EXIT == 0)); then
        printf '%s\n' \
            "${TIME_OUTPUT}"
    else
        info \
            "timedatectl information unavailable"
    fi
fi


# -------------------------------------------------------------------
# Python host interpretation
# -------------------------------------------------------------------

if [[ -n "${PYTHON}" ]]; then
    section "Python system view"

    "${PYTHON}" - <<'PY'
from __future__ import annotations

import os
import platform
import shutil


print(
    "platform="
    + platform.platform()
)

print(
    "machine="
    + platform.machine()
)

print(
    "processor="
    + str(
        platform.processor()
    )
)

print(
    "cpu_count="
    + str(
        os.cpu_count()
    )
)


for path in (
    "/",
    ".",
):
    usage = shutil.disk_usage(
        path
    )

    print(
        (
            "disk_usage "
            f"path={path} "
            f"total={usage.total} "
            f"used={usage.used} "
            f"free={usage.free}"
        )
    )
PY
fi


# -------------------------------------------------------------------
# Basic consistency checks
# -------------------------------------------------------------------

section "Consistency checks"


AVAILABLE_CPU_COUNT=0


if command -v nproc >/dev/null 2>&1; then
    AVAILABLE_CPU_COUNT="$(
        nproc
    )"

elif [[ -n "${PYTHON}" ]]; then
    AVAILABLE_CPU_COUNT="$(
        "${PYTHON}" -c \
            'import os; print(os.cpu_count() or 0)'
    )"
fi


if ((AVAILABLE_CPU_COUNT < 1)); then
    failure \
        "System reports no available CPUs"
else
    success \
        "Available CPU count: ${AVAILABLE_CPU_COUNT}"
fi


MEM_AVAILABLE_KB=0


if [[ -r /proc/meminfo ]]; then
    MEM_AVAILABLE_KB="$(
        awk \
            '/^MemAvailable:/ {
                print $2
                exit
            }' \
            /proc/meminfo
    )"

    MEM_AVAILABLE_KB="${MEM_AVAILABLE_KB:-0}"
fi


if ((MEM_AVAILABLE_KB <= 0)); then
    warn \
        "Unable to determine available system memory"
else
    success \
        "Available memory reported by kernel"
fi


PROJECT_AVAILABLE_KB=0


if command -v df >/dev/null 2>&1; then
    PROJECT_AVAILABLE_KB="$(
        df -Pk "${PROJECT_ROOT}" \
        | awk \
            'NR == 2 {
                print $4
            }'
    )"

    PROJECT_AVAILABLE_KB="${PROJECT_AVAILABLE_KB:-0}"
fi


if ((PROJECT_AVAILABLE_KB <= 0)); then
    if [[ "${STRICT}" == true ]]; then
        failure \
            "Project filesystem reports no available space"
    else
        warn \
            "Unable to determine project filesystem free space"
    fi
else
    success \
        "Project filesystem has available space"
fi


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

elapsed=$((SECONDS - START_SECONDS))


printf '\n%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"

printf '%sOSHA System Diagnostic Summary%s\n' \
    "${BOLD}" \
    "${RESET}"

printf 'Kernel:           %s\n' \
    "$(uname -r)"

printf 'Architecture:     %s\n' \
    "$(uname -m)"

printf 'Available CPUs:   %s\n' \
    "${AVAILABLE_CPU_COUNT}"

printf 'Memory available: %s KiB\n' \
    "${MEM_AVAILABLE_KB}"

printf 'Disk available:   %s KiB\n' \
    "${PROJECT_AVAILABLE_KB}"

printf 'Cgroups:          %s\n' \
    "${CGROUP_VERSION}"

printf 'Virtualization:   %s\n' \
    "${VIRTUALIZATION}"

printf 'Warnings:         %s\n' \
    "${WARNINGS}"

printf 'Failures:         %s\n' \
    "${FAILURES}"

printf 'Mode:             %s\n' \
    "$(
        if [[ "${STRICT}" == true ]]; then
            printf 'strict'
        else
            printf 'normal'
        fi
    )"

printf 'Duration:         %ss\n' \
    "${elapsed}"

printf '%s========================================%s\n' \
    "${BOLD}" \
    "${RESET}"


# -------------------------------------------------------------------
# Result
# -------------------------------------------------------------------

if ((FAILURES > 0)); then
    printf '\n%sSYSTEM DIAGNOSTICS FAILED%s\n' \
        "${RED}" \
        "${RESET}"

    exit 1
fi


if ((WARNINGS > 0)); then
    printf '\n%sSYSTEM DIAGNOSTICS COMPLETED WITH WARNINGS%s\n' \
        "${YELLOW}" \
        "${RESET}"

else
    printf '\n%sSYSTEM DIAGNOSTICS PASSED%s\n' \
        "${GREEN}" \
        "${RESET}"
fi


exit 0
