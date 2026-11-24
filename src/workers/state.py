from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Optional,
    Set,
)

from src.workers.schema import (
    JobLease,
    JobState,
    LeaseState,
    WorkerJob,
    WorkerState,
    WorkerStatus,
    utc_now,
)


# ---------------------------------------------------------------
# Worker transitions
# ---------------------------------------------------------------

WORKER_TRANSITIONS: Dict[
    str,
    Set[str],
] = {
    WorkerState.CREATED.value: {
        WorkerState.STARTING.value,
        WorkerState.STOPPED.value,
        WorkerState.FAILED.value,
    },
    WorkerState.STARTING.value: {
        WorkerState.IDLE.value,
        WorkerState.FAILED.value,
        WorkerState.STOPPING.value,
    },
    WorkerState.IDLE.value: {
        WorkerState.BUSY.value,
        WorkerState.DRAINING.value,
        WorkerState.STOPPING.value,
        WorkerState.FAILED.value,
    },
    WorkerState.BUSY.value: {
        WorkerState.IDLE.value,
        WorkerState.DRAINING.value,
        WorkerState.STOPPING.value,
        WorkerState.FAILED.value,
    },
    WorkerState.DRAINING.value: {
        WorkerState.IDLE.value,
        WorkerState.STOPPING.value,
        WorkerState.FAILED.value,
    },
    WorkerState.STOPPING.value: {
        WorkerState.STOPPED.value,
        WorkerState.FAILED.value,
    },
    WorkerState.STOPPED.value: (set()),
    WorkerState.FAILED.value: {
        WorkerState.STOPPING.value,
        WorkerState.STOPPED.value,
    },
}


# ---------------------------------------------------------------
# Job transitions
# ---------------------------------------------------------------

JOB_TRANSITIONS: Dict[
    str,
    Set[str],
] = {
    JobState.PENDING.value: {
        JobState.READY.value,
        JobState.CANCELLED.value,
        JobState.EXPIRED.value,
    },
    JobState.READY.value: {
        JobState.LEASED.value,
        JobState.CANCELLED.value,
        JobState.EXPIRED.value,
    },
    JobState.LEASED.value: {
        JobState.RUNNING.value,
        JobState.READY.value,
        JobState.CANCELLED.value,
        JobState.EXPIRED.value,
    },
    JobState.RUNNING.value: {
        JobState.SUCCEEDED.value,
        JobState.RETRY_WAIT.value,
        JobState.FAILED.value,
        JobState.CANCELLED.value,
    },
    JobState.RETRY_WAIT.value: {
        JobState.READY.value,
        JobState.FAILED.value,
        JobState.CANCELLED.value,
        JobState.EXPIRED.value,
    },
    JobState.SUCCEEDED.value: (set()),
    JobState.FAILED.value: (set()),
    JobState.CANCELLED.value: (set()),
    JobState.EXPIRED.value: (set()),
}


# ---------------------------------------------------------------
# Lease transitions
# ---------------------------------------------------------------

LEASE_TRANSITIONS: Dict[
    str,
    Set[str],
] = {
    LeaseState.ACTIVE.value: {
        LeaseState.RELEASED.value,
        LeaseState.EXPIRED.value,
        LeaseState.COMPLETED.value,
    },
    LeaseState.RELEASED.value: (set()),
    LeaseState.EXPIRED.value: (set()),
    LeaseState.COMPLETED.value: (set()),
}


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------


class InvalidWorkerTransition(ValueError):
    pass


class InvalidJobTransition(ValueError):
    pass


class InvalidLeaseTransition(ValueError):
    pass


# ---------------------------------------------------------------
# Transition record
# ---------------------------------------------------------------


@dataclass(frozen=True)
class StateTransition:
    subject: str

    previous_state: str

    next_state: str

    reason: str

    occurred_at: str = field(default_factory=utc_now)

    attributes: Dict[
        str,
        Any,
    ] = field(default_factory=dict)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "previous_state": (self.previous_state),
            "next_state": (self.next_state),
            "reason": self.reason,
            "occurred_at": (self.occurred_at),
            "attributes": dict(self.attributes),
        }


# ---------------------------------------------------------------
# Worker transition
# ---------------------------------------------------------------


def transition_worker(
    worker: WorkerStatus,
    next_state: str,
    *,
    reason: str,
) -> StateTransition:
    worker.finalize()

    previous = worker.state

    if next_state != previous and next_state not in WORKER_TRANSITIONS.get(
        previous,
        set(),
    ):
        raise (
            InvalidWorkerTransition(
                f"Invalid worker transition: {previous} -> {next_state}"
            )
        )

    # An IDLE worker must have no jobs.
    if next_state == WorkerState.IDLE.value and worker.active_job_ids:
        raise (
            InvalidWorkerTransition("Worker cannot become idle while jobs are active")
        )

    # A fully stopped worker must have
    # relinquished all active jobs.
    if next_state == WorkerState.STOPPED.value and worker.active_job_ids:
        raise (InvalidWorkerTransition("Worker cannot stop while jobs are active"))

    event = StateTransition(
        subject=(f"worker:{worker.worker_id}"),
        previous_state=previous,
        next_state=next_state,
        reason=reason,
    )

    worker.state = next_state

    if next_state == WorkerState.STARTING.value and worker.started_at is None:
        worker.started_at = utc_now()

    if next_state == WorkerState.STOPPED.value:
        worker.stopped_at = utc_now()

    worker.attributes.setdefault(
        "state_transitions",
        [],
    ).append(event.to_dict())

    worker.finalize()

    return event


# ---------------------------------------------------------------
# Job transition
# ---------------------------------------------------------------


def transition_job(
    job: WorkerJob,
    next_state: str,
    *,
    reason: str,
    error: Optional[str] = None,
) -> StateTransition:
    job.finalize()

    previous = job.state

    if next_state != previous and next_state not in JOB_TRANSITIONS.get(
        previous,
        set(),
    ):
        raise (
            InvalidJobTransition(f"Invalid job transition: {previous} -> {next_state}")
        )

    # LEASED/RUNNING require ownership.
    if (
        next_state
        in {
            JobState.LEASED.value,
            JobState.RUNNING.value,
        }
        and job.lease is None
    ):
        raise (InvalidJobTransition(f"Job cannot become {next_state} without a lease"))

    # ---------------------------------------------------------
    # Starting an attempt
    # ---------------------------------------------------------

    if next_state == JobState.RUNNING.value:
        if job.lease is None or job.lease.state != LeaseState.ACTIVE.value:
            raise (InvalidJobTransition("Job requires an active lease before running"))

        job.attempt += 1

        if job.attempt > job.max_attempts:
            job.attempt -= 1

            raise (InvalidJobTransition("Job attempt limit has been reached"))

    if error is not None:
        job.error = error

    # ---------------------------------------------------------
    # Terminal jobs
    # ---------------------------------------------------------

    if next_state in {
        JobState.SUCCEEDED.value,
        JobState.FAILED.value,
        JobState.CANCELLED.value,
        JobState.EXPIRED.value,
    }:
        job.completed_at = utc_now()

    event = StateTransition(
        subject=(f"job:{job.job_id}"),
        previous_state=previous,
        next_state=next_state,
        reason=reason,
    )

    job.state = next_state

    job.attributes.setdefault(
        "state_transitions",
        [],
    ).append(event.to_dict())

    job.finalize()

    return event


# ---------------------------------------------------------------
# Lease transition
# ---------------------------------------------------------------


def transition_lease(
    lease: JobLease,
    next_state: str,
    *,
    reason: str,
) -> StateTransition:
    lease.finalize()

    previous = lease.state

    if next_state != previous and next_state not in LEASE_TRANSITIONS.get(
        previous,
        set(),
    ):
        raise (
            InvalidLeaseTransition(
                f"Invalid lease transition: {previous} -> {next_state}"
            )
        )

    event = StateTransition(
        subject=(f"lease:{lease.lease_id}"),
        previous_state=previous,
        next_state=next_state,
        reason=reason,
    )

    lease.state = next_state

    lease.finalize()

    return event


# ---------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------


def worker_has_capacity(
    worker: WorkerStatus,
) -> bool:
    worker.finalize()

    return len(worker.active_job_ids) < worker.concurrency and worker.state in {
        WorkerState.IDLE.value,
        WorkerState.BUSY.value,
    }


# ---------------------------------------------------------------
# Assign job
# ---------------------------------------------------------------


def assign_job(
    worker: WorkerStatus,
    job: WorkerJob,
) -> None:
    """
    Attach a leased job to its owning worker and
    move it into RUNNING.

    Lease acquisition itself belongs in
    queue.py.
    """

    worker.finalize()

    job.finalize()

    if not worker_has_capacity(worker):
        raise (InvalidWorkerTransition("Worker has no available job capacity"))

    if job.lease is None or job.lease.worker_id != worker.worker_id:
        raise (InvalidLeaseTransition("Job lease does not belong to this worker"))

    if job.lease.state != LeaseState.ACTIVE.value:
        raise (InvalidLeaseTransition("Job lease is not active"))

    if job.state != JobState.LEASED.value:
        raise (InvalidJobTransition("Only leased jobs can be assigned to a worker"))

    if job.job_id not in worker.active_job_ids:
        worker.active_job_ids.append(job.job_id)

    transition_job(
        job,
        JobState.RUNNING.value,
        reason=("Worker began processing the leased job."),
    )

    if worker.state == WorkerState.IDLE.value:
        transition_worker(
            worker,
            WorkerState.BUSY.value,
            reason=("Worker accepted an active job."),
        )

    worker.finalize()


# ---------------------------------------------------------------
# Release worker capacity
# ---------------------------------------------------------------


def release_job(
    worker: WorkerStatus,
    job: WorkerJob,
    *,
    success: bool,
) -> None:
    """
    Release worker capacity after an attempt
    finishes.

    This handles worker accounting only.
    """

    if job.job_id in worker.active_job_ids:
        worker.active_job_ids.remove(job.job_id)

    if success:
        worker.completed_jobs += 1

    else:
        worker.failed_jobs += 1

    if not worker.active_job_ids and worker.state == WorkerState.BUSY.value:
        transition_worker(
            worker,
            WorkerState.IDLE.value,
            reason=("Worker has no remaining active jobs."),
        )

    worker.finalize()


# ---------------------------------------------------------------
# Worker state manager
# ---------------------------------------------------------------


class WorkerStateManager:
    """
    Controlled facade for worker/job lifecycle
    transitions.

    It does not dequeue or execute jobs.
    """

    # ---------------------------------------------------------
    # Worker lifecycle
    # ---------------------------------------------------------

    def start(
        self,
        worker: WorkerStatus,
    ) -> WorkerStatus:
        transition_worker(
            worker,
            WorkerState.STARTING.value,
            reason=("Worker startup began."),
        )

        transition_worker(
            worker,
            WorkerState.IDLE.value,
            reason=("Worker startup completed."),
        )

        return worker

    def drain(
        self,
        worker: WorkerStatus,
    ) -> WorkerStatus:
        if worker.state in {
            WorkerState.IDLE.value,
            WorkerState.BUSY.value,
        }:
            transition_worker(
                worker,
                WorkerState.DRAINING.value,
                reason=("Worker entered drain mode."),
            )

        return worker

    def stop(
        self,
        worker: WorkerStatus,
    ) -> WorkerStatus:
        if worker.active_job_ids:
            raise (
                InvalidWorkerTransition("Worker cannot stop while jobs remain active")
            )

        if worker.state == WorkerState.DRAINING.value:
            transition_worker(
                worker,
                WorkerState.STOPPING.value,
                reason=("Worker drain completed; shutdown began."),
            )

        elif worker.state in {
            WorkerState.IDLE.value,
            WorkerState.STARTING.value,
            WorkerState.FAILED.value,
        }:
            transition_worker(
                worker,
                WorkerState.STOPPING.value,
                reason=("Worker shutdown began."),
            )

        elif worker.state == WorkerState.CREATED.value:
            transition_worker(
                worker,
                WorkerState.STOPPED.value,
                reason=("Worker stopped before startup."),
            )

            return worker

        elif worker.state == WorkerState.STOPPED.value:
            return worker

        else:
            raise (
                InvalidWorkerTransition(f"Worker cannot stop from state {worker.state}")
            )

        transition_worker(
            worker,
            WorkerState.STOPPED.value,
            reason=("Worker shutdown completed."),
        )

        return worker

    def fail(
        self,
        worker: WorkerStatus,
        *,
        reason: str,
        error: Optional[str] = None,
    ) -> WorkerStatus:
        worker.last_error = error

        transition_worker(
            worker,
            WorkerState.FAILED.value,
            reason=reason,
        )

        return worker

    # ---------------------------------------------------------
    # Job lifecycle
    # ---------------------------------------------------------

    def mark_ready(
        self,
        job: WorkerJob,
    ) -> WorkerJob:
        transition_job(
            job,
            JobState.READY.value,
            reason=("Job is ready for leasing."),
        )

        return job

    def start_job(
        self,
        worker: WorkerStatus,
        job: WorkerJob,
    ) -> WorkerJob:
        assign_job(
            worker,
            job,
        )

        return job

    def succeed_job(
        self,
        worker: WorkerStatus,
        job: WorkerJob,
    ) -> WorkerJob:
        transition_job(
            job,
            JobState.SUCCEEDED.value,
            reason=("Job completed successfully."),
        )

        if job.lease is not None and job.lease.state == LeaseState.ACTIVE.value:
            transition_lease(
                job.lease,
                LeaseState.COMPLETED.value,
                reason=("Job completed under this lease."),
            )

        release_job(
            worker,
            job,
            success=True,
        )

        return job

    def fail_job(
        self,
        worker: WorkerStatus,
        job: WorkerJob,
        *,
        retry: bool,
        error: Optional[str] = None,
    ) -> WorkerJob:
        """
        Finish an unsuccessful processing
        attempt.

        RETRY_WAIT is used only when:
          retry was requested
          AND attempts remain.
        """

        next_state = (
            JobState.RETRY_WAIT.value
            if (retry and job.attempt < job.max_attempts)
            else (JobState.FAILED.value)
        )

        transition_job(
            job,
            next_state,
            reason=(
                "Job will wait before retry."
                if (next_state == JobState.RETRY_WAIT.value)
                else ("Job processing failed.")
            ),
            error=error,
        )

        if job.lease is not None and job.lease.state == LeaseState.ACTIVE.value:
            transition_lease(
                job.lease,
                LeaseState.RELEASED.value,
                reason=("Worker released the lease after unsuccessful processing."),
            )

        release_job(
            worker,
            job,
            success=False,
        )

        return job
