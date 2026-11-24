from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
    Set,
)

from src.orchestrator.schema import (
    OrchestrationEvidence,
    OrchestrationRun,
    OrchestrationState,
    StageName,
    StageResult,
    StageStatus,
    StopReason,
    utc_now,
)


# ---------------------------------------------------------------
# Run transitions
# ---------------------------------------------------------------

RUN_TRANSITIONS: Dict[
    str,
    Set[str],
] = {
    OrchestrationState.CREATED.value: {
        OrchestrationState.QUEUED.value,
        OrchestrationState.RUNNING.value,
        OrchestrationState.CANCELLED.value,
        OrchestrationState.FAILED.value,
    },
    OrchestrationState.QUEUED.value: {
        OrchestrationState.RUNNING.value,
        OrchestrationState.CANCELLED.value,
        OrchestrationState.FAILED.value,
    },
    OrchestrationState.RUNNING.value: {
        OrchestrationState.WAITING.value,
        OrchestrationState.PAUSED.value,
        OrchestrationState.COMPLETED.value,
        OrchestrationState.PARTIAL.value,
        OrchestrationState.FAILED.value,
        OrchestrationState.CANCELLED.value,
        OrchestrationState.ESCALATED.value,
    },
    OrchestrationState.WAITING.value: {
        OrchestrationState.RUNNING.value,
        OrchestrationState.PAUSED.value,
        OrchestrationState.FAILED.value,
        OrchestrationState.CANCELLED.value,
        OrchestrationState.ESCALATED.value,
    },
    OrchestrationState.PAUSED.value: {
        OrchestrationState.RUNNING.value,
        OrchestrationState.CANCELLED.value,
        OrchestrationState.FAILED.value,
    },
    OrchestrationState.PARTIAL.value: set(),
    OrchestrationState.COMPLETED.value: set(),
    OrchestrationState.FAILED.value: set(),
    OrchestrationState.CANCELLED.value: set(),
    OrchestrationState.ESCALATED.value: set(),
}


TERMINAL_RUN_STATES = {
    OrchestrationState.COMPLETED.value,
    OrchestrationState.PARTIAL.value,
    OrchestrationState.FAILED.value,
    OrchestrationState.CANCELLED.value,
    OrchestrationState.ESCALATED.value,
}


TERMINAL_STAGE_STATUSES = {
    StageStatus.SUCCEEDED.value,
    StageStatus.SKIPPED.value,
    StageStatus.BLOCKED.value,
    StageStatus.FAILED.value,
    StageStatus.CANCELLED.value,
}


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------


class InvalidOrchestrationTransition(ValueError):
    pass


class InvalidStageTransition(ValueError):
    pass


# ---------------------------------------------------------------
# Transition record
# ---------------------------------------------------------------


@dataclass(frozen=True)
class RunTransition:
    """
    Immutable description of one orchestration
    state change.
    """

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
            "previous_state": (self.previous_state),
            "next_state": (self.next_state),
            "reason": self.reason,
            "occurred_at": (self.occurred_at),
            "attributes": dict(self.attributes),
        }


# ---------------------------------------------------------------
# Run-state helpers
# ---------------------------------------------------------------


def valid_run_state(
    value: str,
) -> bool:
    return value in {item.value for item in OrchestrationState}


def can_transition(
    run: OrchestrationRun,
    next_state: str,
) -> bool:
    if run.state == next_state:
        return True

    return next_state in RUN_TRANSITIONS.get(
        run.state,
        set(),
    )


def transition_run(
    run: OrchestrationRun,
    next_state: str,
    *,
    reason: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> RunTransition:
    """
    Move the orchestration run through a valid
    lifecycle transition.
    """

    run.finalize()

    if not valid_run_state(next_state):
        raise (
            InvalidOrchestrationTransition(f"Unknown orchestration state: {next_state}")
        )

    previous = run.state

    if not can_transition(
        run,
        next_state,
    ):
        raise (
            InvalidOrchestrationTransition(
                f"Invalid orchestration transition: {previous} -> {next_state}"
            )
        )

    event = RunTransition(
        previous_state=previous,
        next_state=next_state,
        reason=reason,
        attributes=dict(attributes or {}),
    )

    run.state = next_state

    if next_state == OrchestrationState.RUNNING.value and run.started_at is None:
        run.started_at = utc_now()

    if next_state in TERMINAL_RUN_STATES:
        run.completed_at = utc_now()

        run.current_stage = None

    run.attributes.setdefault(
        "state_transitions",
        [],
    ).append(event.to_dict())

    run.evidence.append(
        OrchestrationEvidence(
            evidence_type=("orchestration_transition"),
            description=(f"{previous} -> {next_state}: {reason}"),
            score=1.0,
            source_id=run.run_id,
            resource_ids=list(run.context.resource_ids),
            attributes={
                "previous_state": (previous),
                "next_state": (next_state),
            },
        )
    )

    run.finalize()

    return event


def ensure_running(
    run: OrchestrationRun,
    *,
    reason: str = ("Orchestration run started."),
) -> None:
    """
    Ensure a non-terminal run is RUNNING.
    """

    if run.state == OrchestrationState.RUNNING.value:
        return

    if run.state in {
        OrchestrationState.CREATED.value,
        OrchestrationState.QUEUED.value,
        OrchestrationState.WAITING.value,
        OrchestrationState.PAUSED.value,
    }:
        transition_run(
            run,
            OrchestrationState.RUNNING.value,
            reason=reason,
        )

        return

    raise (
        InvalidOrchestrationTransition(
            f"Run cannot enter running state from {run.state}"
        )
    )


# ---------------------------------------------------------------
# Stage lookup
# ---------------------------------------------------------------


def get_stage(
    run: OrchestrationRun,
    stage: str,
) -> Optional[StageResult]:
    return run.stage(stage)


# ---------------------------------------------------------------
# Start stage
# ---------------------------------------------------------------


def start_stage(
    run: OrchestrationRun,
    stage: str,
    *,
    input_count: int = 0,
    attributes: Optional[Dict[str, Any]] = None,
) -> StageResult:
    """
    Start one orchestration stage.

    Only one occurrence of each high-level stage
    exists in the initial orchestrator model.
    """

    run.finalize()

    if stage not in {item.value for item in StageName}:
        raise InvalidStageTransition(f"Unknown orchestration stage: {stage}")

    ensure_running(run)

    existing = get_stage(
        run,
        stage,
    )

    if existing is not None:
        if existing.status == StageStatus.RUNNING.value:
            return existing

        raise InvalidStageTransition(
            f"Stage {stage} already exists with status {existing.status}"
        )

    result = StageResult(
        stage=stage,
        status=(StageStatus.RUNNING.value),
        started_at=utc_now(),
        input_count=input_count,
        attributes=dict(attributes or {}),
    ).finalize()

    run.stages.append(result)

    run.current_stage = stage

    run.finalize()

    return result


# ---------------------------------------------------------------
# Complete stage
# ---------------------------------------------------------------


def complete_stage(
    run: OrchestrationRun,
    stage: str,
    *,
    output_count: int = 0,
    output_ids: Iterable[str] = (),
    attributes: Optional[Dict[str, Any]] = None,
) -> StageResult:
    result = get_stage(
        run,
        stage,
    )

    if result is None:
        raise InvalidStageTransition(f"Stage {stage} has not been started")

    if result.status != StageStatus.RUNNING.value:
        raise InvalidStageTransition(
            f"Stage {stage} cannot complete from status {result.status}"
        )

    result.status = StageStatus.SUCCEEDED.value

    result.completed_at = utc_now()

    result.output_count = max(
        0,
        int(output_count),
    )

    result.output_ids = sorted(set(output_ids))

    if attributes:
        result.attributes.update(attributes)

    if run.current_stage == stage:
        run.current_stage = None

    run.finalize()

    return result.finalize()


# ---------------------------------------------------------------
# Generic terminal stage handling
# ---------------------------------------------------------------


def end_stage(
    run: OrchestrationRun,
    stage: str,
    *,
    status: str,
    reason: str,
    stop_reason: str = (StopReason.NONE.value),
    error: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> StageResult:
    """
    Finish a stage with a terminal status other
    than the normal success path.
    """

    if status not in TERMINAL_STAGE_STATUSES:
        raise InvalidStageTransition(f"Status {status} is not terminal for a stage")

    result = get_stage(
        run,
        stage,
    )

    if result is None:
        raise InvalidStageTransition(f"Stage {stage} has not been started")

    if result.status != StageStatus.RUNNING.value:
        raise InvalidStageTransition(
            f"Stage {stage} cannot end from status {result.status}"
        )

    result.status = status

    result.completed_at = utc_now()

    result.error = error

    if attributes:
        result.attributes.update(attributes)

    result.evidence.append(
        OrchestrationEvidence(
            evidence_type=("stage_end"),
            description=reason,
            score=(
                1.0
                if status
                in {
                    StageStatus.SUCCEEDED.value,
                    StageStatus.SKIPPED.value,
                }
                else 0.0
            ),
            source_id=(run.run_id),
            resource_ids=list(run.context.resource_ids),
            attributes={
                "stage": stage,
                "status": status,
            },
        )
    )

    if run.current_stage == stage:
        run.current_stage = None

    if stop_reason != StopReason.NONE.value:
        run.stop_reason = stop_reason

    run.finalize()

    return result.finalize()


# ---------------------------------------------------------------
# Convenience stage operations
# ---------------------------------------------------------------


def block_stage(
    run: OrchestrationRun,
    stage: str,
    *,
    reason: str,
    stop_reason: str = (StopReason.BLOCKED.value),
) -> StageResult:
    return end_stage(
        run,
        stage,
        status=(StageStatus.BLOCKED.value),
        reason=reason,
        stop_reason=(stop_reason),
    )


def fail_stage(
    run: OrchestrationRun,
    stage: str,
    *,
    reason: str,
    error: Optional[str] = None,
) -> StageResult:
    return end_stage(
        run,
        stage,
        status=(StageStatus.FAILED.value),
        reason=reason,
        stop_reason=(StopReason.FAILURE.value),
        error=error,
    )


def skip_stage(
    run: OrchestrationRun,
    stage: str,
    *,
    reason: str,
) -> StageResult:
    start_stage(
        run,
        stage,
    )

    return end_stage(
        run,
        stage,
        status=(StageStatus.SKIPPED.value),
        reason=reason,
    )


# ---------------------------------------------------------------
# Finish run
# ---------------------------------------------------------------


def finish_run(
    run: OrchestrationRun,
) -> OrchestrationRun:
    """
    Derive a final run state from stage results.
    """

    run.finalize()

    if run.state in TERMINAL_RUN_STATES:
        return run

    if run.current_stage is not None:
        raise (
            InvalidOrchestrationTransition(
                f"Cannot finish run while stage {run.current_stage} is active"
            )
        )

    failed = [item for item in run.stages if (item.status == StageStatus.FAILED.value)]

    blocked = [
        item for item in run.stages if (item.status == StageStatus.BLOCKED.value)
    ]

    if failed:
        run.stop_reason = StopReason.FAILURE.value

        transition_run(
            run,
            OrchestrationState.FAILED.value,
            reason=("One or more orchestration stages failed."),
        )

    elif blocked:
        if run.stop_reason == StopReason.ESCALATION_REQUIRED.value:
            transition_run(
                run,
                OrchestrationState.ESCALATED.value,
                reason=("The orchestration run requires human escalation."),
            )

        else:
            if run.stop_reason == StopReason.NONE.value:
                run.stop_reason = StopReason.BLOCKED.value

            transition_run(
                run,
                OrchestrationState.PARTIAL.value,
                reason=("One or more orchestration stages were blocked."),
            )

    else:
        run.stop_reason = StopReason.COMPLETED.value

        transition_run(
            run,
            OrchestrationState.COMPLETED.value,
            reason=("All executed orchestration stages completed."),
        )

    return run.finalize()


# ---------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------


def cancel_run(
    run: OrchestrationRun,
    *,
    reason: str,
) -> OrchestrationRun:
    run.finalize()

    if run.current_stage:
        active_stage = run.current_stage

        result = get_stage(
            run,
            active_stage,
        )

        if result and result.status == StageStatus.RUNNING.value:
            end_stage(
                run,
                active_stage,
                status=(StageStatus.CANCELLED.value),
                reason=reason,
                stop_reason=(StopReason.CANCELLED.value),
            )

    run.stop_reason = StopReason.CANCELLED.value

    transition_run(
        run,
        OrchestrationState.CANCELLED.value,
        reason=reason,
    )

    return run.finalize()


# ---------------------------------------------------------------
# State manager
# ---------------------------------------------------------------


class OrchestrationStateManager:
    """
    Reusable orchestration state manager.

    This class changes orchestration metadata
    only. It cannot operate infrastructure.
    """

    def start(
        self,
        run: OrchestrationRun,
        *,
        reason: str = ("Orchestration run started."),
    ) -> OrchestrationRun:
        ensure_running(
            run,
            reason=reason,
        )

        return run.finalize()

    def start_stage(
        self,
        run: OrchestrationRun,
        stage: str,
        *,
        input_count: int = 0,
    ) -> StageResult:
        return start_stage(
            run,
            stage,
            input_count=input_count,
        )

    def complete_stage(
        self,
        run: OrchestrationRun,
        stage: str,
        *,
        output_count: int = 0,
        output_ids: Iterable[str] = (),
    ) -> StageResult:
        return complete_stage(
            run,
            stage,
            output_count=(output_count),
            output_ids=output_ids,
        )

    def block_stage(
        self,
        run: OrchestrationRun,
        stage: str,
        *,
        reason: str,
        stop_reason: str = (StopReason.BLOCKED.value),
    ) -> StageResult:
        return block_stage(
            run,
            stage,
            reason=reason,
            stop_reason=(stop_reason),
        )

    def fail_stage(
        self,
        run: OrchestrationRun,
        stage: str,
        *,
        reason: str,
        error: Optional[str] = None,
    ) -> StageResult:
        return fail_stage(
            run,
            stage,
            reason=reason,
            error=error,
        )

    def finish(
        self,
        run: OrchestrationRun,
    ) -> OrchestrationRun:
        return finish_run(run)

    def cancel(
        self,
        run: OrchestrationRun,
        *,
        reason: str,
    ) -> OrchestrationRun:
        return cancel_run(
            run,
            reason=reason,
        )
