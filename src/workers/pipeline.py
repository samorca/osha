from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

from src.orchestrator.pipeline import (
    OrchestratorPipeline,
    PipelineInput,
    PipelineResult,
)

from src.orchestrator.schema import (
    OrchestrationContext,
    TriggerType,
)

from src.workers.config import (
    WorkerConfig,
)

from src.workers.manager import (
    WorkerManager,
    WorkerManagerConfig,
)

from src.workers.queue import (
    InMemoryJobQueue,
)

from src.workers.schema import (
    JobType,
    WorkerJob,
    WorkersResult,
)

from src.workers.worker import (
    HandlerResult,
    JobHandler,
)


# ---------------------------------------------------------------
# Workers-pipeline configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class WorkerPipelineConfig:
    """
    Configuration for the worker composition
    layer.

    require_dry_run remains enabled by default.
    """

    worker_count: int = 1

    maximum_cycles: int = 100

    require_dry_run: bool = True

    def validate(
        self,
    ) -> None:
        if self.worker_count < 1:
            raise ValueError("worker_count must be >= 1")

        if self.maximum_cycles < 1:
            raise ValueError("maximum_cycles must be >= 1")


# ---------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------


def _mapping(
    value: Any,
    name: str,
) -> Dict[str, Any]:
    raw = (
        value.get(
            name,
            {},
        )
        if isinstance(
            value,
            Mapping,
        )
        else {}
    )

    if raw is None:
        return {}

    if not isinstance(
        raw,
        Mapping,
    ):
        raise ValueError(f"{name} must be a mapping")

    return dict(raw)


def _strings(
    value: Any,
    name: str,
) -> list[str]:
    raw = (
        value.get(
            name,
            [],
        )
        if isinstance(
            value,
            Mapping,
        )
        else []
    )

    if raw is None:
        return []

    if isinstance(
        raw,
        str,
    ):
        return [raw]

    return [str(item) for item in raw]


# ---------------------------------------------------------------
# Worker job -> orchestrator input
# ---------------------------------------------------------------


def pipeline_input_from_job(
    job: WorkerJob,
) -> PipelineInput:
    """
    Convert one orchestration WorkerJob into an
    Orchestrator PipelineInput.

    The worker does not directly invoke
    remediation or self-healing.
    """

    job.finalize()

    if job.job_type != JobType.ORCHESTRATION.value:
        raise ValueError("Only orchestration jobs can be translated into PipelineInput")

    payload = job.payload

    dry_run = bool(
        payload.get(
            "dry_run",
            True,
        )
    )

    context = OrchestrationContext(
        environment_id=(job.environment_id),
        trigger_type=str(
            payload.get(
                "trigger_type",
                TriggerType.WORKER.value,
            )
        ),
        incident_id=(payload.get("incident_id")),
        correlation_id=(payload.get("correlation_id")),
        resource_ids=_strings(
            payload,
            "resource_ids",
        ),
        service_ids=_strings(
            payload,
            "service_ids",
        ),
        requested_by=(payload.get("requested_by")),
        dry_run=dry_run,
        attributes={
            **_mapping(
                payload,
                "context_attributes",
            ),
            "worker_job_id": (job.job_id),
            # job.attempt increments when the
            # Worker moves LEASED -> RUNNING.
            # Handler conversion happens
            # after that transition.
            "worker_attempt": (job.attempt),
        },
    ).finalize()

    initial_artifacts = _mapping(
        payload,
        "initial_artifacts",
    )

    metadata = {
        **_mapping(
            payload,
            "metadata",
        ),
        "worker_job_id": (job.job_id),
        "worker_type": (job.job_type),
    }

    return PipelineInput(
        context=context,
        initial_artifacts=(initial_artifacts),
        metadata=metadata,
    ).finalize()


# ---------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------


def orchestration_references(
    result: PipelineResult,
) -> list[str]:
    """
    Build bounded references to authoritative
    orchestration outputs.

    Full subsystem artifacts are not copied into
    worker job results.
    """

    references: list[str] = []

    if result.run.run_id:
        references.append(result.run.run_id)

    references.extend(result.run.remediation_ids)

    references.extend(result.run.healing_session_ids)

    references.extend(result.run.outcome_ids)

    return sorted(set(references))


# ---------------------------------------------------------------
# Orchestration worker handler
# ---------------------------------------------------------------


def orchestration_handler(
    orchestrator: OrchestratorPipeline,
    *,
    require_dry_run: bool = True,
) -> JobHandler:
    """
    Build the handler used by workers for
    ORCHESTRATION jobs.

    Infrastructure-changing logic remains behind
    the orchestrator/remediation layers.
    """

    def handle(
        job: WorkerJob,
    ) -> HandlerResult:
        pipeline_input = pipeline_input_from_job(job)

        # -----------------------------------------------------
        # Independent dry-run boundary
        # -----------------------------------------------------

        if require_dry_run and not pipeline_input.context.dry_run:
            return HandlerResult(
                success=False,
                retryable=False,
                error=("Worker pipeline rejected non-dry-run orchestration input."),
                attributes={
                    "policy_block": ("require_dry_run"),
                },
            )

        # -----------------------------------------------------
        # Run through orchestrator
        # -----------------------------------------------------

        result = orchestrator.run(pipeline_input)

        failed = result.run.state == "failed"

        # Only a recorded orchestration failure
        # is considered potentially retryable
        # here. Approval, safety, escalation and
        # other controlled stop reasons are not
        # retried by the worker.
        retryable = bool(failed and result.run.stop_reason == "failure")

        return HandlerResult(
            success=not failed,
            retryable=retryable,
            result_reference_ids=(orchestration_references(result)),
            attributes={
                "orchestration_state": (result.run.state),
                "orchestration_stop_reason": (result.run.stop_reason),
            },
            error=("Orchestration run failed." if failed else None),
        ).finalize()

    return handle


# ---------------------------------------------------------------
# Workers pipeline
# ---------------------------------------------------------------


class WorkersPipeline:
    """
    Composition root for OSHA background workers.

    Main flow:

        WorkerJob
            ↓
        shared queue
            ↓
        worker lease
            ↓
        ORCHESTRATION handler
            ↓
        OrchestratorPipeline
            ↓
        reference-based JobResult

    Real infrastructure actions remain governed
    by downstream decision/remediation controls.
    """

    def __init__(
        self,
        *,
        orchestrator: OrchestratorPipeline,
        worker_config: Optional[WorkerConfig] = None,
        config: Optional[WorkerPipelineConfig] = None,
        queue: Optional[InMemoryJobQueue] = None,
        handlers: Optional[
            Mapping[
                str,
                JobHandler,
            ]
        ] = None,
    ) -> None:
        self.config = config or WorkerPipelineConfig()

        self.config.validate()

        self.worker_config = worker_config or WorkerConfig()

        self.worker_config.validate()

        if self.config.require_dry_run and not self.worker_config.dry_run_only:
            raise ValueError(
                "Worker pipeline requires a dry-run-only worker configuration"
            )

        self.queue = queue or InMemoryJobQueue(
            lease_config=(self.worker_config.lease),
            retry_config=(self.worker_config.retry),
        )

        registered: Dict[
            str,
            JobHandler,
        ] = dict(handlers or {})

        # WorkersPipeline owns the integration
        # between ORCHESTRATION jobs and
        # OrchestratorPipeline.
        if JobType.ORCHESTRATION.value in registered:
            raise ValueError("orchestration handler is owned by WorkersPipeline")

        registered[JobType.ORCHESTRATION.value] = orchestration_handler(
            orchestrator,
            require_dry_run=(self.config.require_dry_run),
        )

        self.manager = WorkerManager(
            queue=self.queue,
            worker_config=(self.worker_config),
            manager_config=(
                WorkerManagerConfig(worker_count=(self.config.worker_count))
            ),
            handlers=registered,
        )

    # ---------------------------------------------------------
    # Submit
    # ---------------------------------------------------------

    def submit(
        self,
        job: WorkerJob,
    ) -> WorkerJob:
        """
        Submit work through the workers pipeline.

        ORCHESTRATION jobs are independently
        checked for the pipeline dry-run policy.
        """

        if self.config.require_dry_run and job.job_type == JobType.ORCHESTRATION.value:
            if job.payload.get("dry_run") is False:
                raise ValueError(
                    "WorkersPipeline refuses non-dry-run orchestration jobs"
                )

            mode = (
                str(
                    job.payload.get(
                        "mode",
                        "",
                    )
                )
                .strip()
                .lower()
            )

            if mode in {
                "execute",
                "live",
                "apply",
            }:
                raise ValueError("WorkersPipeline refuses live execution mode")

        return self.queue.submit(job)

    # ---------------------------------------------------------
    # Start
    # ---------------------------------------------------------

    def start(
        self,
    ) -> None:
        self.manager.start()

    # ---------------------------------------------------------
    # Processing
    # ---------------------------------------------------------

    def run_once(
        self,
    ):
        return self.manager.run_cycle()

    def run(
        self,
        *,
        maximum_cycles: Optional[int] = None,
    ):
        limit = (
            maximum_cycles
            if maximum_cycles is not None
            else (self.config.maximum_cycles)
        )

        if limit < 1:
            raise ValueError("maximum_cycles must be >= 1")

        return self.manager.run_cycles(limit)

    # ---------------------------------------------------------
    # Shutdown lifecycle
    # ---------------------------------------------------------

    def drain(
        self,
    ) -> None:
        self.manager.drain()

    def stop(
        self,
    ) -> None:
        self.manager.stop()

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    def result(
        self,
    ) -> WorkersResult:
        return self.manager.result()

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            **self.manager.summary(),
            "require_dry_run": (self.config.require_dry_run),
            "maximum_cycles": (self.config.maximum_cycles),
        }
