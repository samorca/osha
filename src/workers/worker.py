from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
)

from src.workers.config import (
    WorkerConfig,
    safe_config_summary,
)

from src.workers.queue import (
    InMemoryJobQueue,
)

from src.workers.schema import (
    JobResult,
    JobState,
    JobType,
    WorkerJob,
    WorkerState,
    WorkerStatus,
)

from src.workers.state import (
    WorkerStateManager,
    worker_has_capacity,
)


# ---------------------------------------------------------------
# Job handler contract
# ---------------------------------------------------------------

JobHandler = Callable[
    [WorkerJob],
    Any,
]


# ---------------------------------------------------------------
# Handler result
# ---------------------------------------------------------------

@dataclass
class HandlerResult:
    """
    Normalized result returned by one worker job
    handler.

    retryable indicates a transient processing
    failure. It must not be used to bypass policy
    or approval blocks.
    """

    success: bool

    retryable: bool = False

    result_reference_ids: list[
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

    error: Optional[
        str
    ] = None

    def finalize(
        self,
    ) -> "HandlerResult":
        self.result_reference_ids = (
            sorted(
                set(
                    self.
                    result_reference_ids
                )
            )
        )

        return self


# ---------------------------------------------------------------
# Worker errors
# ---------------------------------------------------------------

class WorkerError(
    RuntimeError
):
    pass


class MissingJobHandlerError(
    WorkerError
):
    pass


class WorkerPolicyError(
    WorkerError
):
    pass


# ---------------------------------------------------------------
# Dry-run policy
# ---------------------------------------------------------------

def payload_requests_live_execution(
    job: WorkerJob,
) -> bool:
    """
    Inspect standard worker payload policy
    fields.

    The worker does not inspect provider-specific
    commands. Downstream remediation guardrails
    remain independently responsible for
    enforcement.
    """

    payload = job.payload

    if bool(
        payload.get(
            "execute",
            False,
        )
    ):
        return True

    if (
        payload.get(
            "dry_run"
        )
        is False
    ):
        return True

    mode = str(
        payload.get(
            "mode",
            "",
        )
    ).strip().lower()

    return mode in {
        "execute",
        "live",
        "apply",
    }


# ---------------------------------------------------------------
# Handler-result normalization
# ---------------------------------------------------------------

def normalize_handler_result(
    value: Any,
) -> HandlerResult:
    """
    Normalize common handler return forms.

    Preferred return type is HandlerResult.
    """

    if isinstance(
        value,
        HandlerResult,
    ):
        return value.finalize()

    if isinstance(
        value,
        JobResult,
    ):
        return HandlerResult(
            success=value.success,
            retryable=False,
            result_reference_ids=list(
                value.
                result_reference_ids
            ),
            attributes=dict(
                value.attributes
            ),
            error=value.error,
        ).finalize()

    if isinstance(
        value,
        Mapping,
    ):
        return HandlerResult(
            success=bool(
                value.get(
                    "success",
                    True,
                )
            ),
            retryable=bool(
                value.get(
                    "retryable",
                    False,
                )
            ),
            result_reference_ids=list(
                value.get(
                    "result_reference_ids",
                    [],
                )
            ),
            attributes=dict(
                value.get(
                    "attributes",
                    {},
                )
            ),
            error=value.get(
                "error"
            ),
        ).finalize()

    if isinstance(
        value,
        bool,
    ):
        return HandlerResult(
            success=value
        ).finalize()

    # A non-exceptional handler return is treated
    # as successful, but its arbitrary value is
    # not copied into job state.
    return HandlerResult(
        success=True,
        attributes={
            "handler_returned_value": (
                value is not None
            ),
        },
    ).finalize()


# ---------------------------------------------------------------
# Worker
# ---------------------------------------------------------------

class Worker:
    """
    Lease and process jobs through explicitly
    registered handlers.

    The Worker:

      leases compatible queue jobs
      enforces worker-level dry-run policy
      invokes one registered handler
      records success/failure
      schedules retryable failures
      tracks worker capacity

    It does not decide infrastructure actions.
    """

    def __init__(
        self,
        *,
        config: Optional[
            WorkerConfig
        ] = None,
        queue: Optional[
            InMemoryJobQueue
        ] = None,
        handlers: Optional[
            Mapping[
                str,
                JobHandler,
            ]
        ] = None,
        state: Optional[
            WorkerStateManager
        ] = None,
    ) -> None:
        self.config = (
            config
            or WorkerConfig()
        )

        self.config.validate()

        self.queue = (
            queue
            or InMemoryJobQueue(
                lease_config=(
                    self.config.lease
                ),
                retry_config=(
                    self.config.retry
                ),
            )
        )

        self.handlers: Dict[
            str,
            JobHandler,
        ] = dict(
            handlers
            or {}
        )

        self.state = (
            state
            or WorkerStateManager()
        )

        self.status = (
            WorkerStatus(
                worker_id=(
                    self.config.worker_id
                ),
                concurrency=(
                    self.config.concurrency
                ),
                attributes={
                    "config": (
                        safe_config_summary(
                            self.config
                        )
                    ),
                },
            ).finalize()
        )

        self._validate_handlers()

    # ---------------------------------------------------------
    # Handler validation
    # ---------------------------------------------------------

    def _validate_handlers(
        self,
    ) -> None:
        valid_types = {
            item.value
            for item in JobType
        }

        unknown = sorted(
            set(
                self.handlers
            )
            - valid_types
        )

        if unknown:
            raise ValueError(
                "Handlers registered for "
                "unknown job types: "
                f"{unknown}"
            )

    def register_handler(
        self,
        job_type: str,
        handler: JobHandler,
    ) -> None:
        if (
            job_type
            not in {
                item.value
                for item in JobType
            }
        ):
            raise ValueError(
                "Unknown job type: "
                f"{job_type}"
            )

        if (
            job_type
            in self.handlers
        ):
            raise ValueError(
                "Handler already "
                "registered for job "
                f"type: {job_type}"
            )

        self.handlers[
            job_type
        ] = handler

    # ---------------------------------------------------------
    # Start
    # ---------------------------------------------------------

    def start(
        self,
    ) -> WorkerStatus:
        if (
            self.status.state
            == WorkerState.CREATED.value
        ):
            self.state.start(
                self.status
            )

        elif (
            self.status.state
            not in {
                WorkerState.IDLE.value,
                WorkerState.BUSY.value,
            }
        ):
            raise WorkerError(
                "Worker cannot start "
                "from state "
                f"{self.status.state}"
            )

        return self.status

    # ---------------------------------------------------------
    # Lease next job
    # ---------------------------------------------------------

    def lease_next(
        self,
    ) -> Optional[
        WorkerJob
    ]:
        """
        Lease one compatible job when the worker
        has capacity.

        No job execution happens here.
        """

        if (
            self.status.state
            in {
                WorkerState.
                DRAINING.value,
                WorkerState.
                STOPPING.value,
                WorkerState.
                STOPPED.value,
                WorkerState.
                FAILED.value,
            }
        ):
            return None

        if not worker_has_capacity(
            self.status
        ):
            return None

        return self.queue.lease(
            worker_id=(
                self.status.worker_id
            ),
            accepted_job_types=(
                self.config.
                accepted_job_types
            ),
        )

    # ---------------------------------------------------------
    # Process job
    # ---------------------------------------------------------

    def process(
        self,
        job: WorkerJob,
    ) -> JobResult:
        """
        Process one job that is already leased to
        this worker.
        """

        job.finalize()

        # -----------------------------------------------------
        # Job type policy
        # -----------------------------------------------------

        if not self.config.accepts(
            job.job_type
        ):
            raise WorkerPolicyError(
                "Worker does not accept "
                f"job type: {job.job_type}"
            )

        # -----------------------------------------------------
        # Dry-run policy
        # -----------------------------------------------------

        if (
            self.config.dry_run_only
            and payload_requests_live_execution(
                job
            )
        ):
            # Policy block is not retryable.
            self.queue.release_lease(
                job.job_id,
                worker_id=(
                    self.status.worker_id
                ),
                return_to_ready=False,
            )

            return JobResult(
                job_id=job.job_id,
                worker_id=(
                    self.status.worker_id
                ),
                success=False,
                state=(
                    JobState.
                    CANCELLED.value
                ),
                error=(
                    "Worker dry-run policy "
                    "rejected live "
                    "execution."
                ),
                attributes={
                    "policy_block": (
                        "dry_run_only"
                    ),
                },
            ).finalize()

        # -----------------------------------------------------
        # Handler lookup
        # -----------------------------------------------------

        handler = self.handlers.get(
            job.job_type
        )

        if handler is None:
            # Return the job to READY because the
            # worker didn't begin processing it.
            self.queue.release_lease(
                job.job_id,
                worker_id=(
                    self.status.worker_id
                ),
                return_to_ready=True,
            )

            raise (
                MissingJobHandlerError(
                    "No handler registered "
                    "for job type: "
                    f"{job.job_type}"
                )
            )

        # -----------------------------------------------------
        # Start processing attempt
        # -----------------------------------------------------

        self.state.start_job(
            self.status,
            job,
        )

        try:
            handler_result = (
                normalize_handler_result(
                    handler(
                        job
                    )
                )
            )

        except Exception as exc:
            # Unexpected handler failures may be
            # retried while attempts remain.
            retry = (
                job.attempt
                < job.max_attempts
            )

            self.state.fail_job(
                self.status,
                job,
                retry=retry,
                error=(
                    "Job handler raised "
                    "an unexpected "
                    "exception."
                ),
            )

            if (
                job.state
                == JobState.
                RETRY_WAIT.value
            ):
                self.queue.schedule_retry(
                    job.job_id
                )

            return JobResult(
                job_id=job.job_id,
                worker_id=(
                    self.status.worker_id
                ),
                success=False,
                state=job.state,
                error=(
                    "Job handler raised "
                    "an unexpected "
                    "exception."
                ),
                attributes={
                    "error_type": (
                        type(exc).__name__
                    ),
                    "retry_scheduled": (
                        job.state
                        == JobState.
                        RETRY_WAIT.value
                    ),
                },
            ).finalize()

        # -----------------------------------------------------
        # Successful result
        # -----------------------------------------------------

        if handler_result.success:
            job.result_reference_ids = list(
                handler_result.
                result_reference_ids
            )

            job.attributes.update(
                handler_result.attributes
            )

            self.state.succeed_job(
                self.status,
                job,
            )

            return JobResult(
                job_id=job.job_id,
                worker_id=(
                    self.status.worker_id
                ),
                success=True,
                state=(
                    JobState.
                    SUCCEEDED.value
                ),
                result_reference_ids=list(
                    job.
                    result_reference_ids
                ),
                attributes=dict(
                    handler_result.
                    attributes
                ),
            ).finalize()

        # -----------------------------------------------------
        # Handler-reported failure
        # -----------------------------------------------------

        retry = bool(
            handler_result.retryable
            and job.attempt
            < job.max_attempts
        )

        self.state.fail_job(
            self.status,
            job,
            retry=retry,
            error=(
                handler_result.error
                or (
                    "Job handler "
                    "reported failure."
                )
            ),
        )

        if (
            job.state
            == JobState.
            RETRY_WAIT.value
        ):
            self.queue.schedule_retry(
                job.job_id
            )

        return JobResult(
            job_id=job.job_id,
            worker_id=(
                self.status.worker_id
            ),
            success=False,
            state=job.state,
            result_reference_ids=list(
                handler_result.
                result_reference_ids
            ),
            error=(
                handler_result.error
                or (
                    "Job handler "
                    "reported failure."
                )
            ),
            attributes={
                **handler_result.
                attributes,
                "retry_scheduled": (
                    job.state
                    == JobState.
                    RETRY_WAIT.value
                ),
            },
        ).finalize()

    # ---------------------------------------------------------
    # Run one queue cycle
    # ---------------------------------------------------------

    def run_once(
        self,
    ) -> Optional[
        JobResult
    ]:
        """
        Lease and process at most one job.

        This method does not contain an infinite
        polling loop.
        """

        if (
            self.status.state
            == WorkerState.CREATED.value
        ):
            self.start()

        job = self.lease_next()

        if job is None:
            return None

        return self.process(
            job
        )

    # ---------------------------------------------------------
    # Drain and stop
    # ---------------------------------------------------------

    def drain(
        self,
    ) -> WorkerStatus:
        self.state.drain(
            self.status
        )

        return self.status

    def stop(
        self,
    ) -> WorkerStatus:
        """
        Stop immediately when no jobs are active.

        If active jobs exist and draining is
        enabled, enter DRAINING and let the
        manager/runtime call stop again after
        those jobs finish.
        """

        if (
            self.config.
            drain_on_shutdown
            and self.status.
            active_job_ids
        ):
            self.drain()

            return self.status

        self.state.stop(
            self.status
        )

        return self.status

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "worker": (
                self.status.to_dict()
            ),
            "queue": (
                self.queue.summary()
            ),
            "registered_job_types": (
                sorted(
                    self.handlers
                )
            ),
            "dry_run_only": (
                self.config.
                dry_run_only
            ),
        }
