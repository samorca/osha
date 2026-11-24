from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
    replace,
)
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)

from src.workers.config import (
    WorkerConfig,
)

from src.workers.queue import (
    InMemoryJobQueue,
)

from src.workers.schema import (
    JobResult,
    WorkerState,
    WorkersResult,
)

from src.workers.worker import (
    JobHandler,
    Worker,
)


# ---------------------------------------------------------------
# Errors
# ---------------------------------------------------------------


class WorkerManagerError(RuntimeError):
    pass


# ---------------------------------------------------------------
# Manager configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class WorkerManagerConfig:
    """
    Configuration for a local OSHA worker pool.

    worker_count is separate from each worker's
    internal concurrency setting.
    """

    worker_count: int = 1

    stop_on_worker_failure: bool = True

    def validate(
        self,
    ) -> None:
        if self.worker_count < 1:
            raise ValueError("worker_count must be >= 1")


# ---------------------------------------------------------------
# Processing cycle result
# ---------------------------------------------------------------


@dataclass
class WorkerCycleResult:
    """
    Result of one bounded pass across the worker
    pool.

    Each worker processes at most one job during
    a cycle.
    """

    results: List[JobResult] = field(default_factory=list)

    workers_polled: int = 0

    jobs_processed: int = 0

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "workers_polled": (self.workers_polled),
            "jobs_processed": (self.jobs_processed),
            "results": [item.to_dict() for item in self.results],
        }


# ---------------------------------------------------------------
# Worker manager
# ---------------------------------------------------------------


class WorkerManager:
    """
    Coordinate multiple workers against one
    shared job queue.

    The manager itself does not execute
    orchestration/remediation logic. Workers
    invoke registered job handlers.
    """

    def __init__(
        self,
        *,
        queue: Optional[InMemoryJobQueue] = None,
        worker_config: Optional[WorkerConfig] = None,
        manager_config: Optional[WorkerManagerConfig] = None,
        handlers: Optional[
            Mapping[
                str,
                JobHandler,
            ]
        ] = None,
    ) -> None:
        self.worker_config = worker_config or WorkerConfig()

        self.worker_config.validate()

        self.manager_config = manager_config or WorkerManagerConfig()

        self.manager_config.validate()

        self.queue = queue or InMemoryJobQueue(
            lease_config=(self.worker_config.lease),
            retry_config=(self.worker_config.retry),
        )

        self.handlers: Dict[
            str,
            JobHandler,
        ] = dict(handlers or {})

        self._workers: Dict[
            str,
            Worker,
        ] = {}

        self._build_workers()

    # ---------------------------------------------------------
    # Worker IDs
    # ---------------------------------------------------------

    def _worker_id(
        self,
        index: int,
    ) -> str:
        """
        Preserve the configured worker ID when
        only one worker exists.

        Multi-worker pools receive deterministic
        numeric suffixes.
        """

        if self.manager_config.worker_count == 1:
            return self.worker_config.worker_id

        return f"{self.worker_config.worker_id}-{index + 1}"

    # ---------------------------------------------------------
    # Build worker pool
    # ---------------------------------------------------------

    def _build_workers(
        self,
    ) -> None:
        for index in range(self.manager_config.worker_count):
            worker_id = self._worker_id(index)

            if worker_id in self._workers:
                raise WorkerManagerError(f"Duplicate worker_id: {worker_id}")

            # WorkerConfig is frozen, so create a
            # new instance with a distinct worker
            # ID.
            config = replace(
                self.worker_config,
                worker_id=worker_id,
            )

            self._workers[worker_id] = Worker(
                config=config,
                queue=self.queue,
                handlers=self.handlers,
            )

    # ---------------------------------------------------------
    # Accessors
    # ---------------------------------------------------------

    def worker(
        self,
        worker_id: str,
    ) -> Optional[Worker]:
        return self._workers.get(worker_id)

    def workers(
        self,
    ) -> List[Worker]:
        return [self._workers[name] for name in sorted(self._workers)]

    # ---------------------------------------------------------
    # Handler registration
    # ---------------------------------------------------------

    def register_handler(
        self,
        job_type: str,
        handler: JobHandler,
    ) -> None:
        """
        Register the same logical handler across
        all workers in this manager.
        """

        if job_type in self.handlers:
            raise WorkerManagerError(
                f"Handler already registered for job type: {job_type}"
            )

        self.handlers[job_type] = handler

        for worker in self._workers.values():
            worker.register_handler(
                job_type,
                handler,
            )

    # ---------------------------------------------------------
    # Start
    # ---------------------------------------------------------

    def start(
        self,
    ) -> List[Any]:
        statuses = []

        for worker in self.workers():
            statuses.append(worker.start())

        return statuses

    # ---------------------------------------------------------
    # Run one bounded cycle
    # ---------------------------------------------------------

    def run_cycle(
        self,
    ) -> WorkerCycleResult:
        """
        Give each runnable worker one opportunity
        to lease/process a job.

        This method does not run continuously.
        """

        results: List[JobResult] = []

        polled = 0

        for worker in self.workers():
            if worker.status.state in {
                WorkerState.DRAINING.value,
                WorkerState.STOPPING.value,
                WorkerState.STOPPED.value,
                WorkerState.FAILED.value,
            }:
                continue

            polled += 1

            result = worker.run_once()

            if result is not None:
                results.append(result)

            if self.manager_config.stop_on_worker_failure and (
                worker.status.state == WorkerState.FAILED.value
            ):
                break

        return WorkerCycleResult(
            results=results,
            workers_polled=polled,
            jobs_processed=len(results),
        )

    # ---------------------------------------------------------
    # Run bounded number of cycles
    # ---------------------------------------------------------

    def run_cycles(
        self,
        maximum_cycles: int,
    ) -> List[WorkerCycleResult]:
        """
        Process up to maximum_cycles.

        The loop exits early when an entire cycle
        processes no jobs.

        This deliberately avoids an unbounded
        daemon loop.
        """

        if maximum_cycles < 1:
            raise ValueError("maximum_cycles must be >= 1")

        cycles: List[WorkerCycleResult] = []

        for _ in range(maximum_cycles):
            result = self.run_cycle()

            cycles.append(result)

            if result.jobs_processed == 0:
                break

        return cycles

    # ------------------------------------------
    # Drain worker pool
    # ---------------------------------------------------------

    def drain(
        self,
    ) -> None:
        """
        Prevent workers from accepting additional
        jobs while allowing already active jobs
        to finish.
        """

        for worker in self.workers():
            if worker.status.state in {
                WorkerState.IDLE.value,
                WorkerState.BUSY.value,
            }:
                worker.drain()

    # ---------------------------------------------------------
    # Stop
    # ---------------------------------------------------------

    def stop(
        self,
    ) -> None:
        """
        Ask each worker to stop.

        Workers with active jobs and draining
        enabled may remain in DRAINING until their
        active jobs are finished.
        """

        for worker in self.workers():
            if worker.status.state == WorkerState.STOPPED.value:
                continue

            worker.stop()

    # ---------------------------------------------------------
    # Pool state
    # ---------------------------------------------------------

    def all_stopped(
        self,
    ) -> bool:
        return all(
            (worker.status.state == WorkerState.STOPPED.value)
            for worker in self._workers.values()
        )

    # ---------------------------------------------------------
    # Aggregate result
    # ---------------------------------------------------------

    def result(
        self,
    ) -> WorkersResult:
        return WorkersResult(
            workers=[worker.status for worker in self.workers()],
            jobs=self.queue.list(),
            metadata={
                "worker_count": (len(self._workers)),
                "all_stopped": (self.all_stopped()),
                "shared_queue": True,
            },
        ).finalize()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        states: Dict[
            str,
            int,
        ] = {}

        active_jobs = 0
        completed_jobs = 0
        failed_jobs = 0

        for worker in self.workers():
            states[worker.status.state] = (
                states.get(
                    worker.status.state,
                    0,
                )
                + 1
            )

            active_jobs += len(worker.status.active_job_ids)

            completed_jobs += worker.status.completed_jobs

            failed_jobs += worker.status.failed_jobs

        return {
            "worker_count": len(self._workers),
            "worker_states": states,
            "active_jobs": (active_jobs),
            "completed_jobs": (completed_jobs),
            "failed_jobs": (failed_jobs),
            "queue": (self.queue.summary()),
        }
