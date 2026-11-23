from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from threading import RLock
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
)

from src.workers.config import (
    LeaseConfig,
    RetryConfig,
)

from src.workers.schema import (
    JobLease,
    JobState,
    LeaseState,
    WorkerJob,
)

from src.workers.state import (
    InvalidJobTransition,
    transition_job,
    transition_lease,
)


# ---------------------------------------------------------------
# Terminal states
# ---------------------------------------------------------------

TERMINAL_JOB_STATES = {
    JobState.SUCCEEDED.value,
    JobState.FAILED.value,
    JobState.CANCELLED.value,
    JobState.EXPIRED.value,
}


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------

class QueueError(
    RuntimeError
):
    pass


class JobNotFoundError(
    QueueError
):
    pass


class LeaseOwnershipError(
    QueueError
):
    pass


# ---------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------

def as_utc(
    value: Optional[
        datetime
    ] = None,
) -> datetime:
    current = (
        value
        or datetime.now(
            timezone.utc
        )
    )

    if current.tzinfo is None:
        current = current.replace(
            tzinfo=timezone.utc
        )

    return current.astimezone(
        timezone.utc
    )


def parse_timestamp(
    value: Optional[
        str
    ],
) -> Optional[
    datetime
]:
    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        parsed = (
            datetime.fromisoformat(
                text
            )
        )

    except ValueError as exc:
        raise QueueError(
            f"Invalid timestamp: {value}"
        ) from exc

    return as_utc(
        parsed
    )


def ready_at(
    job: WorkerJob,
) -> datetime:
    return (
        parse_timestamp(
            job.scheduled_for
        )
        or parse_timestamp(
            job.created_at
        )
        or as_utc()
    )


def job_expired(
    job: WorkerJob,
    *,
    now: datetime,
) -> bool:
    expires_at = parse_timestamp(
        job.expires_at
    )

    return bool(
        expires_at is not None
        and now >= expires_at
    )


def lease_expired(
    job: WorkerJob,
    *,
    now: datetime,
) -> bool:
    if (
        job.lease is None
        or job.lease.state
        != LeaseState.ACTIVE.value
    ):
        return False

    expires_at = parse_timestamp(
        job.lease.expires_at
    )

    return bool(
        expires_at is not None
        and now >= expires_at
    )


# ---------------------------------------------------------------
# In-memory queue
# ---------------------------------------------------------------

class InMemoryJobQueue:
    """
    Thread-safe development queue with
    in-process atomic lease acquisition.

    This queue:

      stores job metadata
      determines readiness
      grants ownership leases
      renews/releases leases
      handles retries
      handles expiration

    It never executes job payloads.

    A future durable/distributed backend must
    supply its own transactional lease behavior.
    """

    def __init__(
        self,
        *,
        lease_config: Optional[
            LeaseConfig
        ] = None,
        retry_config: Optional[
            RetryConfig
        ] = None,
        maximum_jobs: int = 10000,
    ) -> None:
        self.lease_config = (
            lease_config
            or LeaseConfig()
        )

        self.retry_config = (
            retry_config
            or RetryConfig()
        )

        self.lease_config.validate()

        self.retry_config.validate()

        if maximum_jobs < 1:
            raise ValueError(
                "maximum_jobs must be >= 1"
            )

        self.maximum_jobs = (
            maximum_jobs
        )

        self._jobs: Dict[
            str,
            WorkerJob,
        ] = {}

        self._lock = RLock()

    # ---------------------------------------------------------
    # Submit
    # ---------------------------------------------------------

    def submit(
        self,
        job: WorkerJob,
    ) -> WorkerJob:
        with self._lock:
            job.finalize()

            if (
                len(
                    self._jobs
                )
                >= self.maximum_jobs
            ):
                raise QueueError(
                    "Queue capacity has "
                    "been reached"
                )

            if (
                job.job_id
                in self._jobs
            ):
                raise QueueError(
                    "Duplicate job_id: "
                    f"{job.job_id}"
                )

            if (
                job.state
                != JobState.PENDING.value
            ):
                raise QueueError(
                    "Only pending jobs "
                    "can be submitted"
                )

            # A job using the schema default can
            # inherit the queue retry policy.
            if job.max_attempts == 1:
                job.max_attempts = (
                    self.retry_config.
                    max_attempts
                )

            self._jobs[
                job.job_id
            ] = job

            self._refresh_job(
                job,
                now=as_utc(),
            )

            return job

    # ---------------------------------------------------------
    # Lookup
    # ---------------------------------------------------------

    def get(
        self,
        job_id: str,
    ) -> Optional[
        WorkerJob
    ]:
        with self._lock:
            return self._jobs.get(
                job_id
            )

    def require(
        self,
        job_id: str,
    ) -> WorkerJob:
        job = self.get(
            job_id
        )

        if job is None:
            raise JobNotFoundError(
                "Unknown job_id: "
                f"{job_id}"
            )

        return job

    # ---------------------------------------------------------
    # List
    # ---------------------------------------------------------

    def list(
        self,
        *,
        states: Optional[
            Iterable[str]
        ] = None,
    ) -> List[
        WorkerJob
    ]:
        with self._lock:
            allowed = (
                set(states)
                if states is not None
                else None
            )

            jobs = [
                job
                for job
                in self._jobs.values()
                if (
                    allowed is None
                    or job.state
                    in allowed
                )
            ]

            return sorted(
                jobs,
                key=lambda item: (
                    ready_at(item),
                    item.priority,
                    item.created_at,
                    item.job_id
                    or "",
                ),
            )

    # ---------------------------------------------------------
    # Refresh one job
    # ---------------------------------------------------------

    def _refresh_job(
        self,
        job: WorkerJob,
        *,
        now: datetime,
    ) -> None:
        if (
            job.state
            in TERMINAL_JOB_STATES
        ):
            return

        # -----------------------------------------------------
        # Job expiration
        # -----------------------------------------------------

        if job_expired(
            job,
            now=now,
        ):
            if (
                job.lease is not None
                and job.lease.state
                == LeaseState.ACTIVE.value
            ):
                transition_lease(
                    job.lease,
                    LeaseState.
                    EXPIRED.value,
                    reason=(
                        "The job expired "
                        "while its lease "
                        "was active."
                    ),
                )

            transition_job(
                job,
                JobState.EXPIRED.value,
                reason=(
                    "The job expiration "
                    "time was reached."
                ),
            )

            return

        # -----------------------------------------------------
        # Lease expiration
        # -----------------------------------------------------

        if lease_expired(
            job,
            now=now,
        ):
            if job.lease is None:
                return

            transition_lease(
                job.lease,
                LeaseState.EXPIRED.value,
                reason=(
                    "The worker lease "
                    "expired."
                ),
            )

            job.attributes[
                "expired_lease_id"
            ] = job.lease.lease_id

            job.lease = None

            # The job was leased but processing
            # had not started, so it can return
            # directly to READY.
            if (
                job.state
                == JobState.LEASED.value
            ):
                transition_job(
                    job,
                    JobState.READY.value,
                    reason=(
                        "Expired lease "
                        "returned the "
                        "unstarted job to "
                        "the ready queue."
                    ),
                )

            # A running job consumed an attempt.
            # It needs retry delay or final
            # failure.
            elif (
                job.state
                == JobState.RUNNING.value
            ):
                if (
                    job.attempt
                    < job.max_attempts
                ):
                    transition_job(
                        job,
                        JobState.
                        RETRY_WAIT.value,
                        reason=(
                            "The active lease "
                            "expired while "
                            "the job was "
                            "running."
                        ),
                        error=(
                            "Worker lease "
                            "expired during "
                            "processing."
                        ),
                    )

                    job.scheduled_for = (
                        now
                        + timedelta(
                            seconds=(
                                self.
                                retry_config.
                                delay_for_attempt(
                                    job.attempt
                                )
                            )
                        )
                    ).isoformat()

                else:
                    transition_job(
                        job,
                        JobState.FAILED.value,
                        reason=(
                            "The lease expired "
                            "and no retry "
                            "attempts remain."
                        ),
                        error=(
                            "Worker lease "
                            "expired during "
                            "the final "
                            "processing "
                            "attempt."
                        ),
                    )

            return

        # -----------------------------------------------------
        # Scheduled job readiness
        # -----------------------------------------------------

        scheduled = ready_at(
            job
        )

        if (
            job.state
            == JobState.PENDING.value
            and now >= scheduled
        ):
            transition_job(
                job,
                JobState.READY.value,
                reason=(
                    "The job reached its "
                    "scheduled processing "
                    "time."
                ),
            )

        elif (
            job.state
            == JobState.
            RETRY_WAIT.value
            and now >= scheduled
        ):
            job.lease = None

            transition_job(
                job,
                JobState.READY.value,
                reason=(
                    "The configured retry "
                    "delay elapsed."
                ),
            )

    # ---------------------------------------------------------
    # Refresh queue
    # ---------------------------------------------------------

    def refresh(
        self,
        *,
        now: Optional[
            datetime
        ] = None,
    ) -> List[
        WorkerJob
    ]:
        current = as_utc(
            now
        )

        with self._lock:
            for job in (
                self._jobs.values()
            ):
                self._refresh_job(
                    job,
                    now=current,
                )

            return self.list()

    # ---------------------------------------------------------
    # Ready jobs
    # ---------------------------------------------------------

    def ready(
        self,
        *,
        accepted_job_types: Optional[
            Iterable[str]
        ] = None,
        now: Optional[
            datetime
        ] = None,
    ) -> List[
        WorkerJob
    ]:
        current = as_utc(
            now
        )

        accepted = (
            set(
                accepted_job_types
            )
            if (
                accepted_job_types
                is not None
            )
            else None
        )

        with self._lock:
            for job in (
                self._jobs.values()
            ):
                self._refresh_job(
                    job,
                    now=current,
                )

            return [
                job
                for job
                in self.list(
                    states=(
                        JobState.
                        READY.value,
                    )
                )
                if (
                    accepted is None
                    or job.job_type
                    in accepted
                )
            ]

    # ---------------------------------------------------------
    # Acquire lease
    # ---------------------------------------------------------

    def lease(
        self,
        *,
        worker_id: str,
        accepted_job_types: Optional[
            Sequence[str]
        ] = None,
        now: Optional[
            datetime
        ] = None,
    ) -> Optional[
        WorkerJob
    ]:
        """
        Atomically acquire one ready job within
        this in-memory queue instance.

        This does not start job execution.
        """

        owner = worker_id.strip()

        if not owner:
            raise QueueError(
                "worker_id is required "
                "for lease acquisition"
            )

        current = as_utc(
            now
        )

        with self._lock:
            available = self.ready(
                accepted_job_types=(
                    accepted_job_types
                ),
                now=current,
            )

            if not available:
                return None

            job = available[
                0
            ]

            job.lease = JobLease(
                worker_id=owner,
                acquired_at=(
                    current.isoformat()
                ),
                expires_at=(
                    current
                    + timedelta(
                        seconds=(
                            self.
                            lease_config.
                            duration_seconds
                        )
                    )
                ).isoformat(),
            ).finalize()

            transition_job(
                job,
                JobState.LEASED.value,
                reason=(
                    "Job lease acquired "
                    f"by worker {owner}."
                ),
            )

            return job

    # ---------------------------------------------------------
    # Renew lease
    # ---------------------------------------------------------

    def renew_lease(
        self,
        job_id: str,
        *,
        worker_id: str,
        now: Optional[
            datetime
        ] = None,
    ) -> JobLease:
        current = as_utc(
            now
        )

        with self._lock:
            job = self.require(
                job_id
            )

            if job.lease is None:
                raise LeaseOwnershipError(
                    "Job has no lease"
                )

            if (
                job.lease.worker_id
                != worker_id
            ):
                raise LeaseOwnershipError(
                    "Lease is owned by "
                    "another worker"
                )

            if (
                job.lease.state
                != LeaseState.ACTIVE.value
            ):
                raise LeaseOwnershipError(
                    "Lease is not active"
                )

            if lease_expired(
                job,
                now=current,
            ):
                self._refresh_job(
                    job,
                    now=current,
                )

                raise LeaseOwnershipError(
                    "Lease has expired"
                )

            job.lease.expires_at = (
                current
                + timedelta(
                    seconds=(
                        self.
                        lease_config.
                        duration_seconds
                    )
                )
            ).isoformat()

            job.attributes[
                "lease_renewed_at"
            ] = current.isoformat()

            job.finalize()

            return job.lease

    # ---------------------------------------------------------
    # Release unstarted lease
    # ---------------------------------------------------------

    def release_lease(
        self,
        job_id: str,
        *,
        worker_id: str,
        return_to_ready: bool = True,
    ) -> WorkerJob:
        """
        Release a job whose processing has not
        started.

        Running jobs must finish through the
        worker state manager instead.
        """

        with self._lock:
            job = self.require(
                job_id
            )

            if (
                job.lease is None
                or job.lease.worker_id
                != worker_id
            ):
                raise LeaseOwnershipError(
                    "Lease is not owned by "
                    "this worker"
                )

            if (
                job.lease.state
                != LeaseState.ACTIVE.value
            ):
                raise LeaseOwnershipError(
                    "Lease is not active"
                )

            if (
                job.state
                != JobState.LEASED.value
            ):
                raise (
                    InvalidJobTransition(
                        "Only a leased, "
                        "unstarted job can "
                        "be manually released"
                    )
                )

            transition_lease(
                job.lease,
                LeaseState.
                RELEASED.value,
                reason=(
                    "Worker released the "
                    "job before processing "
                    "began."
                ),
            )

            job.attributes[
                "released_lease_id"
            ] = job.lease.lease_id

            job.lease = None

            if return_to_ready:
                transition_job(
                    job,
                    JobState.READY.value,
                    reason=(
                        "Released job "
                        "returned to the "
                        "ready queue."
                    ),
                )

            else:
                transition_job(
                    job,
                    JobState.
                    CANCELLED.value,
                    reason=(
                        "Released job was "
                        "cancelled."
                    ),
                )

            return job

    # ---------------------------------------------------------
    # Schedule retry
    # ---------------------------------------------------------

    def schedule_retry(
        self,
        job_id: str,
        *,
        now: Optional[
            datetime
        ] = None,
    ) -> WorkerJob:
        current = as_utc(
            now
        )

        with self._lock:
            job = self.require(
                job_id
            )

            if (
                job.state
                != JobState.
                RETRY_WAIT.value
            ):
                raise QueueError(
                    "Only retry-wait jobs "
                    "can be scheduled "
                    "for retry"
                )

            if (
                job.attempt
                >= job.max_attempts
            ):
                transition_job(
                    job,
                    JobState.FAILED.value,
                    reason=(
                        "No retry attempts "
                        "remain."
                    ),
                )

                return job

            job.lease = None

            job.scheduled_for = (
                current
                + timedelta(
                    seconds=(
                        self.retry_config.
                        delay_for_attempt(
                            job.attempt
                        )
                    )
                )
            ).isoformat()

            job.attributes[
                "retry_scheduled_at"
            ] = current.isoformat()

            job.finalize()

            return job

    # ---------------------------------------------------------
    # Cancel job
    # ---------------------------------------------------------

    def cancel(
        self,
        job_id: str,
        *,
        reason: str = (
            "Job cancellation requested."
        ),
    ) -> WorkerJob:
        with self._lock:
            job = self.require(
                job_id
            )

            if (
                job.state
                in TERMINAL_JOB_STATES
            ):
                raise QueueError(
                    "Job is already "
                    f"terminal: {job.state}"
                )

            if (
                job.state
                == JobState.RUNNING.value
            ):
                raise QueueError(
                    "Running jobs must be "
                    "cancelled by their "
                    "owning worker"
                )

            if (
                job.lease is not None
                and job.lease.state
                == LeaseState.ACTIVE.value
            ):
                transition_lease(
                    job.lease,
                    LeaseState.
                    RELEASED.value,
                    reason=(
                        "Job was cancelled "
                        "before processing "
                        "completed."
                    ),
                )

            transition_job(
                job,
                JobState.CANCELLED.value,
                reason=reason,
            )

            return job

    # ---------------------------------------------------------
    # Remove terminal job
    # ---------------------------------------------------------

    def remove(
        self,
        job_id: str,
    ) -> WorkerJob:
        with self._lock:
            job = self.require(
                job_id
            )

            if (
                job.state
                not in TERMINAL_JOB_STATES
            ):
                raise QueueError(
                    "Only terminal jobs "
                    "can be removed"
                )

            return self._jobs.pop(
                job_id
            )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            states: Dict[
                str,
                int,
            ] = {}

            for job in (
                self._jobs.values()
            ):
                states[
                    job.state
                ] = (
                    states.get(
                        job.state,
                        0,
                    )
                    + 1
                )

            return {
                "job_count": len(
                    self._jobs
                ),
                "states": states,
                "maximum_jobs": (
                    self.maximum_jobs
                ),
                "lease_duration_seconds": (
                    self.lease_config.
                    duration_seconds
                ),
            }
