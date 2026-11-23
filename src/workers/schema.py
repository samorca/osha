from __future__ import annotations

import hashlib
import json
import math
from dataclasses import (
    asdict,
    dataclass,
    field,
)
from datetime import (
    datetime,
    timezone,
)
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)


WORKERS_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------

class WorkerState(
    str,
    Enum,
):
    CREATED = "created"

    STARTING = "starting"

    IDLE = "idle"

    BUSY = "busy"

    DRAINING = "draining"

    STOPPING = "stopping"

    STOPPED = "stopped"

    FAILED = "failed"


# ---------------------------------------------------------------
# Job state
# ---------------------------------------------------------------

class JobState(
    str,
    Enum,
):
    PENDING = "pending"

    READY = "ready"

    LEASED = "leased"

    RUNNING = "running"

    SUCCEEDED = "succeeded"

    RETRY_WAIT = "retry_wait"

    FAILED = "failed"

    CANCELLED = "cancelled"

    EXPIRED = "expired"


# ---------------------------------------------------------------
# Job type
# ---------------------------------------------------------------

class JobType(
    str,
    Enum,
):
    """
    Type of work being processed.

    ORCHESTRATION should eventually be the
    primary path for controlled infrastructure
    workflows.
    """

    ORCHESTRATION = (
        "orchestration"
    )

    TELEMETRY = "telemetry"

    CORRELATION = "correlation"

    INTELLIGENCE = "intelligence"

    TRAINING = "training"

    INTERNAL = "internal"


# ---------------------------------------------------------------
# Lease state
# ---------------------------------------------------------------

class LeaseState(
    str,
    Enum,
):
    ACTIVE = "active"

    RELEASED = "released"

    EXPIRED = "expired"

    COMPLETED = "completed"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def json_safe(
    value: Any,
    *,
    depth: int = 0,
    maximum_depth: int = 20,
) -> Any:
    if depth > maximum_depth:
        return (
            "[MAX_DEPTH_REACHED]"
        )

    if (
        value is None
        or isinstance(
            value,
            (
                bool,
                int,
                str,
            ),
        )
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        return (
            value
            if math.isfinite(
                value
            )
            else None
        )

    if isinstance(
        value,
        datetime,
    ):
        if value.tzinfo is None:
            value = value.replace(
                tzinfo=timezone.utc
            )

        return value.isoformat()

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): json_safe(
                item,
                depth=depth + 1,
                maximum_depth=(
                    maximum_depth
                ),
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
            json_safe(
                item,
                depth=depth + 1,
                maximum_depth=(
                    maximum_depth
                ),
            )
            for item in value
        ]

    if hasattr(
        value,
        "__dataclass_fields__",
    ):
        return json_safe(
            asdict(value),
            depth=depth + 1,
            maximum_depth=(
                maximum_depth
            ),
        )

    return str(value)


def stable_hash(
    value: Mapping[
        str,
        Any,
    ],
) -> str:
    encoded = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


# ---------------------------------------------------------------
# Job lease
# ---------------------------------------------------------------

@dataclass
class JobLease:
    """
    Ownership metadata for an actively leased
    worker job.

    A lease helps prevent several workers from
    intentionally processing the same queue item
    at the same time.
    """

    worker_id: str

    acquired_at: str

    expires_at: str

    state: str = (
        LeaseState.ACTIVE.value
    )

    lease_id: Optional[
        str
    ] = None

    def build_lease_id(
        self,
    ) -> str:
        return (
            "worker:lease:"
            + stable_hash(
                {
                    "worker_id": (
                        self.worker_id
                    ),
                    "acquired_at": (
                        self.acquired_at
                    ),
                    "expires_at": (
                        self.expires_at
                    ),
                }
            )
        )

    def finalize(
        self,
    ) -> "JobLease":
        self.worker_id = (
            self.worker_id.strip()
        )

        if not self.worker_id:
            raise ValueError(
                "worker_id is required"
            )

        if (
            self.state
            not in {
                item.value
                for item
                in LeaseState
            }
        ):
            raise ValueError(
                "Unknown lease state: "
                f"{self.state}"
            )

        if not self.lease_id:
            self.lease_id = (
                self.build_lease_id()
            )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return json_safe(
            asdict(self)
        )


# ---------------------------------------------------------------
# Worker job
# ---------------------------------------------------------------

@dataclass
class WorkerJob:
    """
    One unit of background work.

    payload should contain enough information for
    an appropriate job handler to process the
    request.

    WorkerJob itself performs no action.
    """

    job_type: str

    environment_id: str

    payload: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    state: str = (
        JobState.PENDING.value
    )

    priority: int = 100

    max_attempts: int = 1

    attempt: int = 0

    scheduled_for: Optional[
        str
    ] = None

    expires_at: Optional[
        str
    ] = None

    orchestration_context_id: Optional[
        str
    ] = None

    lease: Optional[
        JobLease
    ] = None

    error: Optional[
        str
    ] = None

    result_reference_ids: List[
        str
    ] = field(
        default_factory=list
    )

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=utc_now
    )

    updated_at: str = field(
        default_factory=utc_now
    )

    completed_at: Optional[
        str
    ] = None

    job_id: Optional[
        str
    ] = None

    def build_job_id(
        self,
    ) -> str:
        return (
            "worker:job:"
            + stable_hash(
                {
                    "job_type": (
                        self.job_type
                    ),
                    "environment_id": (
                        self.environment_id
                    ),
                    "orchestration_context_id": (
                        self.
                        orchestration_context_id
                    ),
                    "created_at": (
                        self.created_at
                    ),
                }
            )
        )

    def finalize(
        self,
    ) -> "WorkerJob":
        self.environment_id = (
            self.environment_id.strip()
        )

        if not self.environment_id:
            raise ValueError(
                "environment_id is required"
            )

        if (
            self.job_type
            not in {
                item.value
                for item
                in JobType
            }
        ):
            raise ValueError(
                "Unknown job type: "
                f"{self.job_type}"
            )

        if (
            self.state
            not in {
                item.value
                for item
                in JobState
            }
        ):
            raise ValueError(
                "Unknown job state: "
                f"{self.state}"
            )

        self.priority = int(
            self.priority
        )

        self.max_attempts = max(
            1,
            int(
                self.max_attempts
            ),
        )

        self.attempt = max(
            0,
            int(
                self.attempt
            ),
        )

        if (
            self.attempt
            > self.max_attempts
        ):
            raise ValueError(
                "attempt cannot exceed "
                "max_attempts"
            )

        if self.lease is not None:
            self.lease.finalize()

        self.result_reference_ids = (
            sorted(
                set(
                    self.
                    result_reference_ids
                )
            )
        )

        if not self.job_id:
            self.job_id = (
                self.build_job_id()
            )

        self.updated_at = (
            utc_now()
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "job_id": self.job_id,
            "job_type": (
                self.job_type
            ),
            "environment_id": (
                self.environment_id
            ),
            "state": self.state,
            "priority": self.priority,
            "max_attempts": (
                self.max_attempts
            ),
            "attempt": self.attempt,
            "scheduled_for": (
                self.scheduled_for
            ),
            "expires_at": (
                self.expires_at
            ),
            "orchestration_context_id": (
                self.
                orchestration_context_id
            ),
            "lease": (
                self.lease.to_dict()
                if self.lease
                else None
            ),
            "payload": (
                json_safe(
                    self.payload
                )
            ),
            "error": self.error,
            "result_reference_ids": (
                self.result_reference_ids
            ),
            "attributes": (
                json_safe(
                    self.attributes
                )
            ),
            "created_at": (
                self.created_at
            ),
            "updated_at": (
                self.updated_at
            ),
            "completed_at": (
                self.completed_at
            ),
        }


# ---------------------------------------------------------------
# Worker status
# ---------------------------------------------------------------

@dataclass
class WorkerStatus:
    """
    Runtime status for one OSHA worker.
    """

    worker_id: str

    state: str = (
        WorkerState.CREATED.value
    )

    concurrency: int = 1

    active_job_ids: List[
        str
    ] = field(
        default_factory=list
    )

    completed_jobs: int = 0

    failed_jobs: int = 0

    last_error: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=utc_now
    )

    started_at: Optional[
        str
    ] = None

    stopped_at: Optional[
        str
    ] = None

    updated_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "WorkerStatus":
        self.worker_id = (
            self.worker_id.strip()
        )

        if not self.worker_id:
            raise ValueError(
                "worker_id is required"
            )

        if (
            self.state
            not in {
                item.value
                for item
                in WorkerState
            }
        ):
            raise ValueError(
                "Unknown worker state: "
                f"{self.state}"
            )

        self.concurrency = max(
            1,
            int(
                self.concurrency
            ),
        )

        self.active_job_ids = (
            sorted(
                set(
                    self.active_job_ids
                )
            )
        )

        if (
            len(
                self.active_job_ids
            )
            > self.concurrency
        ):
            raise ValueError(
                "active_job_ids exceeds "
                "worker concurrency"
            )

        self.completed_jobs = max(
            0,
            int(
                self.completed_jobs
            ),
        )

        self.failed_jobs = max(
            0,
            int(
                self.failed_jobs
            ),
        )

        self.updated_at = (
            utc_now()
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return json_safe(
            asdict(self)
        )


# ---------------------------------------------------------------
# Job result
# ---------------------------------------------------------------

@dataclass
class JobResult:
    """
    Bounded result metadata for processing one
    worker job.

    Detailed subsystem results should remain in
    their authoritative subsystem and be
    referenced using result_reference_ids.
    """

    job_id: str

    worker_id: str

    success: bool

    state: str

    result_reference_ids: List[
        str
    ] = field(
        default_factory=list
    )

    error: Optional[
        str
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    completed_at: str = field(
        default_factory=utc_now
    )

    def finalize(
        self,
    ) -> "JobResult":
        if (
            self.state
            not in {
                item.value
                for item
                in JobState
            }
        ):
            raise ValueError(
                "Unknown job result "
                f"state: {self.state}"
            )

        self.result_reference_ids = (
            sorted(
                set(
                    self.
                    result_reference_ids
                )
            )
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return json_safe(
            asdict(self)
        )


# ---------------------------------------------------------------
# Aggregate worker result
# ---------------------------------------------------------------

@dataclass
class WorkersResult:
    workers: List[
        WorkerStatus
    ] = field(
        default_factory=list
    )

    jobs: List[
        WorkerJob
    ] = field(
        default_factory=list
    )

    generated_at: str = field(
        default_factory=utc_now
    )

    metadata: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    schema_version: str = (
        WORKERS_SCHEMA_VERSION
    )

    def finalize(
        self,
    ) -> "WorkersResult":
        self.workers = [
            worker.finalize()
            for worker
            in self.workers
        ]

        self.jobs = [
            job.finalize()
            for job
            in self.jobs
        ]

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        worker_states: Dict[
            str,
            int,
        ] = {}

        job_states: Dict[
            str,
            int,
        ] = {}

        for worker in self.workers:
            worker_states[
                worker.state
            ] = (
                worker_states.get(
                    worker.state,
                    0,
                )
                + 1
            )

        for job in self.jobs:
            job_states[
                job.state
            ] = (
                job_states.get(
                    job.state,
                    0,
                )
                + 1
            )

        return {
            "schema_version": (
                self.schema_version
            ),
            "generated_at": (
                self.generated_at
            ),
            "worker_count": len(
                self.workers
            ),
            "job_count": len(
                self.jobs
            ),
            "worker_states": (
                worker_states
            ),
            "job_states": (
                job_states
            ),
            "workers": [
                worker.to_dict()
                for worker
                in self.workers
            ],
            "jobs": [
                job.to_dict()
                for job
                in self.jobs
            ],
            "metadata": (
                json_safe(
                    self.metadata
                )
            ),
        }
