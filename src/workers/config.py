from __future__ import annotations

import os
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Tuple,
)

from src.workers.schema import (
    JobType,
)


# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

_TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "on",
}

_FALSE_VALUES = {
    "0",
    "false",
    "no",
    "off",
}

DEFAULT_ENV_PREFIX = (
    "OSHA_WORKER_"
)


# ---------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------

def _bool(
    value: Any,
    *,
    default: bool,
) -> bool:
    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    if isinstance(
        value,
        bool,
    ):
        return value

    normalized = str(
        value
    ).strip().lower()

    if normalized in _TRUE_VALUES:
        return True

    if normalized in _FALSE_VALUES:
        return False

    raise ValueError(
        "Invalid boolean value: "
        f"{value}"
    )


def _int(
    value: Any,
    *,
    default: int,
    minimum: int,
) -> int:
    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    result = int(
        value
    )

    if result < minimum:
        raise ValueError(
            "Integer value must be "
            f">= {minimum}"
        )

    return result


def _float(
    value: Any,
    *,
    default: float,
    minimum: float,
) -> float:
    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    result = float(
        value
    )

    if result < minimum:
        raise ValueError(
            "Float value must be "
            f">= {minimum}"
        )

    return result


def _csv(
    value: Any,
    *,
    default: Tuple[
        str,
        ...,
    ],
) -> Tuple[
    str,
    ...,
]:
    if (
        value is None
        or str(value).strip() == ""
    ):
        return default

    return tuple(
        dict.fromkeys(
            part.strip()
            for part
            in str(value).split(",")
            if part.strip()
        )
    )


# ---------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class RetryConfig:
    """
    Worker retry policy.

    The worker layer may retry processing a job,
    but it must not use retries to bypass
    decision/remediation safety controls.
    """

    max_attempts: int = 3

    initial_delay_seconds: float = 2.0

    maximum_delay_seconds: float = 60.0

    backoff_multiplier: float = 2.0

    def validate(
        self,
    ) -> None:
        if self.max_attempts < 1:
            raise ValueError(
                "max_attempts "
                "must be >= 1"
            )

        if (
            self.initial_delay_seconds
            < 0
        ):
            raise ValueError(
                "initial_delay_seconds "
                "must be >= 0"
            )

        if (
            self.maximum_delay_seconds
            < self.initial_delay_seconds
        ):
            raise ValueError(
                "maximum_delay_seconds "
                "must be >= "
                "initial_delay_seconds"
            )

        if (
            self.backoff_multiplier
            < 1.0
        ):
            raise ValueError(
                "backoff_multiplier "
                "must be >= 1.0"
            )

    def delay_for_attempt(
        self,
        attempt: int,
    ) -> float:
        """
        Calculate bounded exponential retry
        delay.

        attempt=1 returns the initial delay.
        """

        attempt_number = max(
            1,
            int(
                attempt
            ),
        )

        delay = (
            self.initial_delay_seconds
            * (
                self.backoff_multiplier
                ** (
                    attempt_number
                    - 1
                )
            )
        )

        return min(
            self.maximum_delay_seconds,
            delay,
        )


# ---------------------------------------------------------------
# Lease configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class LeaseConfig:
    """
    Worker lease timing.

    renewal_interval_seconds must be shorter than
    the lease duration so a future worker loop
    has an opportunity to renew ownership.
    """

    duration_seconds: float = 60.0

    renewal_interval_seconds: float = 20.0

    def validate(
        self,
    ) -> None:
        if (
            self.duration_seconds
            <= 0
        ):
            raise ValueError(
                "lease duration_seconds "
                "must be > 0"
            )

        if (
            self.
            renewal_interval_seconds
            <= 0
        ):
            raise ValueError(
                "renewal_interval_seconds "
                "must be > 0"
            )

        if (
            self.
            renewal_interval_seconds
            >= self.duration_seconds
        ):
            raise ValueError(
                "renewal_interval_seconds "
                "must be less than lease "
                "duration_seconds"
            )


# ---------------------------------------------------------------
# Worker configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class WorkerConfig:
    """
    Configuration for one OSHA worker.

    The default worker handles orchestration
    jobs and remains dry-run-only.
    """

    worker_id: str = "worker-1"

    concurrency: int = 1

    poll_interval_seconds: float = 1.0

    graceful_shutdown_seconds: float = 30.0

    drain_on_shutdown: bool = True

    dry_run_only: bool = True

    accepted_job_types: Tuple[
        str,
        ...,
    ] = (
        JobType.
        ORCHESTRATION.value,
    )

    retry: RetryConfig = field(
        default_factory=RetryConfig
    )

    lease: LeaseConfig = field(
        default_factory=LeaseConfig
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:
        if not (
            self.worker_id.strip()
        ):
            raise ValueError(
                "worker_id is required"
            )

        if self.concurrency < 1:
            raise ValueError(
                "concurrency "
                "must be >= 1"
            )

        if (
            self.poll_interval_seconds
            <= 0
        ):
            raise ValueError(
                "poll_interval_seconds "
                "must be > 0"
            )

        if (
            self.
            graceful_shutdown_seconds
            <= 0
        ):
            raise ValueError(
                "graceful_shutdown_seconds "
                "must be > 0"
            )

        valid_types = {
            item.value
            for item in JobType
        }

        unknown = sorted(
            set(
                self.accepted_job_types
            )
            - valid_types
        )

        if unknown:
            raise ValueError(
                "Unknown accepted job "
                f"types: {unknown}"
            )

        if not (
            self.accepted_job_types
        ):
            raise ValueError(
                "accepted_job_types "
                "cannot be empty"
            )

        self.retry.validate()

        self.lease.validate()

    def accepts(
        self,
        job_type: str,
    ) -> bool:
        return (
            job_type
            in set(
                self.accepted_job_types
            )
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.validate()

        return {
            "worker_id": (
                self.worker_id
            ),
            "concurrency": (
                self.concurrency
            ),
            "poll_interval_seconds": (
                self.
                poll_interval_seconds
            ),
            "graceful_shutdown_seconds": (
                self.
                graceful_shutdown_seconds
            ),
            "drain_on_shutdown": (
                self.drain_on_shutdown
            ),
            "dry_run_only": (
                self.dry_run_only
            ),
            "accepted_job_types": (
                list(
                    self.
                    accepted_job_types
                )
            ),
            "retry": {
                "max_attempts": (
                    self.retry.
                    max_attempts
                ),
                "initial_delay_seconds": (
                    self.retry.
                    initial_delay_seconds
                ),
                "maximum_delay_seconds": (
                    self.retry.
                    maximum_delay_seconds
                ),
                "backoff_multiplier": (
                    self.retry.
                    backoff_multiplier
                ),
            },
            "lease": {
                "duration_seconds": (
                    self.lease.
                    duration_seconds
                ),
                "renewal_interval_seconds": (
                    self.lease.
                    renewal_interval_seconds
                ),
            },
            "attributes": dict(
                self.attributes
            ),
        }


# ---------------------------------------------------------------
# Mapping loader
# ---------------------------------------------------------------

def from_mapping(
    values: Mapping[
        str,
        Any,
    ],
    *,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> WorkerConfig:
    """
    Build worker configuration from a key/value
    mapping.

    Keeping this separate from os.environ makes
    configuration testing straightforward.
    """

    def get(
        name: str,
        default: Any = None,
    ) -> Any:
        return values.get(
            prefix + name,
            default,
        )

    retry = RetryConfig(
        max_attempts=_int(
            get(
                "MAX_ATTEMPTS"
            ),
            default=3,
            minimum=1,
        ),
        initial_delay_seconds=_float(
            get(
                "RETRY_INITIAL_DELAY_SECONDS"
            ),
            default=2.0,
            minimum=0.0,
        ),
        maximum_delay_seconds=_float(
            get(
                "RETRY_MAXIMUM_DELAY_SECONDS"
            ),
            default=60.0,
            minimum=0.0,
        ),
        backoff_multiplier=_float(
            get(
                "RETRY_BACKOFF_MULTIPLIER"
            ),
            default=2.0,
            minimum=1.0,
        ),
    )

    lease = LeaseConfig(
        duration_seconds=_float(
            get(
                "LEASE_DURATION_SECONDS"
            ),
            default=60.0,
            minimum=0.001,
        ),
        renewal_interval_seconds=_float(
            get(
                "LEASE_RENEWAL_INTERVAL_SECONDS"
            ),
            default=20.0,
            minimum=0.001,
        ),
    )

    config = WorkerConfig(
        worker_id=str(
            get(
                "ID",
                "worker-1",
            )
        ).strip(),

        concurrency=_int(
            get(
                "CONCURRENCY"
            ),
            default=1,
            minimum=1,
        ),

        poll_interval_seconds=_float(
            get(
                "POLL_INTERVAL_SECONDS"
            ),
            default=1.0,
            minimum=0.001,
        ),

        graceful_shutdown_seconds=(
            _float(
                get(
                    "GRACEFUL_SHUTDOWN_SECONDS"
                ),
                default=30.0,
                minimum=0.001,
            )
        ),

        drain_on_shutdown=_bool(
            get(
                "DRAIN_ON_SHUTDOWN"
            ),
            default=True,
        ),

        dry_run_only=_bool(
            get(
                "DRY_RUN_ONLY"
            ),
            default=True,
        ),

        accepted_job_types=_csv(
            get(
                "ACCEPTED_JOB_TYPES"
            ),
            default=(
                JobType.
                ORCHESTRATION.value,
            ),
        ),

        retry=retry,

        lease=lease,

        attributes={
            "config_source": (
                "mapping"
            ),
            "environment_prefix": (
                prefix
            ),
        },
    )

    config.validate()

    return config


# ---------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------

def from_environment(
    *,
    environ: Optional[
        Mapping[
            str,
            str,
        ]
    ] = None,
    prefix: str = DEFAULT_ENV_PREFIX,
) -> WorkerConfig:
    source = (
        os.environ
        if environ is None
        else environ
    )

    config = from_mapping(
        source,
        prefix=prefix,
    )

    return WorkerConfig(
        worker_id=(
            config.worker_id
        ),
        concurrency=(
            config.concurrency
        ),
        poll_interval_seconds=(
            config.
            poll_interval_seconds
        ),
        graceful_shutdown_seconds=(
            config.
            graceful_shutdown_seconds
        ),
        drain_on_shutdown=(
            config.
            drain_on_shutdown
        ),
        dry_run_only=(
            config.dry_run_only
        ),
        accepted_job_types=(
            config.
            accepted_job_types
        ),
        retry=config.retry,
        lease=config.lease,
        attributes={
            **config.attributes,
            "config_source": (
                "environment"
            ),
        },
    )


# ---------------------------------------------------------------
# Safe summary
# ---------------------------------------------------------------

def safe_config_summary(
    config: WorkerConfig,
) -> Dict[str, Any]:
    """
    Return non-secret worker configuration for
    diagnostics and status output.
    """

    config.validate()

    return {
        "worker_id": (
            config.worker_id
        ),
        "concurrency": (
            config.concurrency
        ),
        "poll_interval_seconds": (
            config.
            poll_interval_seconds
        ),
        "graceful_shutdown_seconds": (
            config.
            graceful_shutdown_seconds
        ),
        "drain_on_shutdown": (
            config.
            drain_on_shutdown
        ),
        "dry_run_only": (
            config.dry_run_only
        ),
        "accepted_job_types": (
            list(
                config.
                accepted_job_types
            )
        ),
        "max_attempts": (
            config.retry.
            max_attempts
        ),
        "lease_duration_seconds": (
            config.lease.
            duration_seconds
        ),
    }
