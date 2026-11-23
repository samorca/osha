from __future__ import annotations

"""
Shared, dependency-free constants for OSHA.

Keep this module deliberately small. It contains
stable values that may be imported across
subsystems without creating cross-layer
dependencies.
"""

from typing import Final


# ---------------------------------------------------------------
# Application identity
# ---------------------------------------------------------------

APPLICATION_NAME: Final[
    str
] = "osha"

APPLICATION_DISPLAY_NAME: Final[
    str
] = "Orchestration Self-Healing AI"

APPLICATION_NAMESPACE: Final[
    str
] = "osha"


# ---------------------------------------------------------------
# Core schema version
# ---------------------------------------------------------------

CORE_SCHEMA_VERSION: Final[
    str
] = "1"


# ---------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------

DEFAULT_ENVIRONMENT_ID: Final[
    str
] = "local-dev"

DEFAULT_INSTANCE_NAME: Final[
    str
] = APPLICATION_NAME


# ---------------------------------------------------------------
# Safety defaults
# ---------------------------------------------------------------

DEFAULT_DRY_RUN: Final[
    bool
] = True

DEFAULT_FAIL_FAST: Final[
    bool
] = True


# ---------------------------------------------------------------
# Runtime timing defaults
# ---------------------------------------------------------------

DEFAULT_STARTUP_TIMEOUT_SECONDS: Final[
    float
] = 60.0

DEFAULT_SHUTDOWN_TIMEOUT_SECONDS: Final[
    float
] = 60.0

DEFAULT_HEALTH_INTERVAL_SECONDS: Final[
    float
] = 15.0

DEFAULT_POLL_INTERVAL_SECONDS: Final[
    float
] = 1.0

DEFAULT_GRACEFUL_SHUTDOWN_SECONDS: Final[
    float
] = 30.0


# ---------------------------------------------------------------
# Worker / queue defaults
# ---------------------------------------------------------------

DEFAULT_WORKER_CONCURRENCY: Final[
    int
] = 1

DEFAULT_JOB_PRIORITY: Final[
    int
] = 100

DEFAULT_MAX_ATTEMPTS: Final[
    int
] = 3


# ---------------------------------------------------------------
# Retry defaults
# ---------------------------------------------------------------

DEFAULT_RETRY_INITIAL_DELAY_SECONDS: Final[
    float
] = 2.0

DEFAULT_RETRY_MAXIMUM_DELAY_SECONDS: Final[
    float
] = 60.0

DEFAULT_RETRY_BACKOFF_MULTIPLIER: Final[
    float
] = 2.0


# ---------------------------------------------------------------
# Lease defaults
# ---------------------------------------------------------------

DEFAULT_LEASE_DURATION_SECONDS: Final[
    float
] = 60.0

DEFAULT_LEASE_RENEWAL_INTERVAL_SECONDS: Final[
    float
] = 20.0


# ---------------------------------------------------------------
# Queue defaults
# ---------------------------------------------------------------

DEFAULT_MAXIMUM_QUEUE_ITEMS: Final[
    int
] = 10_000


# ---------------------------------------------------------------
# Serialization defaults
# ---------------------------------------------------------------

DEFAULT_MAX_SERIALIZATION_DEPTH: Final[
    int
] = 20

MAX_DEPTH_REACHED_MARKER: Final[
    str
] = "[MAX_DEPTH_REACHED]"


# ---------------------------------------------------------------
# Identifier namespaces
# ---------------------------------------------------------------
#
# These are prefixes only.
#
# The actual identifier-generation implementation
# belongs in:
#
#     src/core/identifiers.py
#
# ---------------------------------------------------------------

ID_PREFIX_RUNTIME_CONTEXT: Final[
    str
] = "runtime:context"

ID_PREFIX_RUNTIME_INSTANCE: Final[
    str
] = "runtime:instance"


ID_PREFIX_ORCHESTRATOR_CONTEXT: Final[
    str
] = "orchestrator:context"

ID_PREFIX_ORCHESTRATOR_RUN: Final[
    str
] = "orchestrator:run"

ID_PREFIX_ORCHESTRATOR_SCHEDULE: Final[
    str
] = "orchestrator:schedule"


ID_PREFIX_WORKER_JOB: Final[
    str
] = "worker:job"

ID_PREFIX_WORKER_LEASE: Final[
    str
] = "worker:lease"


# ---------------------------------------------------------------
# Environment-variable prefixes
# ---------------------------------------------------------------

ENV_PREFIX_RUNTIME: Final[
    str
] = "OSHA_"

ENV_PREFIX_WORKER: Final[
    str
] = "OSHA_WORKER_"


# ---------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------

__all__ = [
    "APPLICATION_NAME",
    "APPLICATION_DISPLAY_NAME",
    "APPLICATION_NAMESPACE",

    "CORE_SCHEMA_VERSION",

    "DEFAULT_ENVIRONMENT_ID",
    "DEFAULT_INSTANCE_NAME",

    "DEFAULT_DRY_RUN",
    "DEFAULT_FAIL_FAST",

    "DEFAULT_STARTUP_TIMEOUT_SECONDS",
    "DEFAULT_SHUTDOWN_TIMEOUT_SECONDS",
    "DEFAULT_HEALTH_INTERVAL_SECONDS",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "DEFAULT_GRACEFUL_SHUTDOWN_SECONDS",

    "DEFAULT_WORKER_CONCURRENCY",
    "DEFAULT_JOB_PRIORITY",
    "DEFAULT_MAX_ATTEMPTS",

    "DEFAULT_RETRY_INITIAL_DELAY_SECONDS",
    "DEFAULT_RETRY_MAXIMUM_DELAY_SECONDS",
    "DEFAULT_RETRY_BACKOFF_MULTIPLIER",

    "DEFAULT_LEASE_DURATION_SECONDS",
    "DEFAULT_LEASE_RENEWAL_INTERVAL_SECONDS",

    "DEFAULT_MAXIMUM_QUEUE_ITEMS",

    "DEFAULT_MAX_SERIALIZATION_DEPTH",
    "MAX_DEPTH_REACHED_MARKER",

    "ID_PREFIX_RUNTIME_CONTEXT",
    "ID_PREFIX_RUNTIME_INSTANCE",

    "ID_PREFIX_ORCHESTRATOR_CONTEXT",
    "ID_PREFIX_ORCHESTRATOR_RUN",
    "ID_PREFIX_ORCHESTRATOR_SCHEDULE",

    "ID_PREFIX_WORKER_JOB",
    "ID_PREFIX_WORKER_LEASE",

    "ENV_PREFIX_RUNTIME",
    "ENV_PREFIX_WORKER",
]
