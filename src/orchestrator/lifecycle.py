from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from src.orchestrator.schema import (
    OrchestrationRun,
    StageName,
    StageStatus,
    StopReason,
)


# ---------------------------------------------------------------
# Default orchestration order
# ---------------------------------------------------------------

DEFAULT_STAGE_ORDER: Tuple[
    str,
    ...,
] = (
    StageName.COLLECTION.value,
    StageName.TELEMETRY.value,
    StageName.CORRELATION.value,
    StageName.INTELLIGENCE.value,
    StageName.MEMORY.value,
    StageName.DECISION.value,
    StageName.REMEDIATION.value,
    StageName.SELF_HEALING.value,
    StageName.OUTCOME.value,
)


# ---------------------------------------------------------------
# Stage dependencies
# ---------------------------------------------------------------

STAGE_DEPENDENCIES: Dict[
    str,
    Set[str],
] = {
    StageName.COLLECTION.value: (set()),
    StageName.TELEMETRY.value: {
        StageName.COLLECTION.value,
    },
    StageName.CORRELATION.value: {
        StageName.TELEMETRY.value,
    },
    StageName.INTELLIGENCE.value: {
        StageName.CORRELATION.value,
    },
    StageName.MEMORY.value: {
        StageName.INTELLIGENCE.value,
    },
    StageName.DECISION.value: {
        StageName.INTELLIGENCE.value,
    },
    StageName.REMEDIATION.value: {
        StageName.DECISION.value,
    },
    StageName.SELF_HEALING.value: {
        StageName.REMEDIATION.value,
    },
    StageName.OUTCOME.value: {
        StageName.SELF_HEALING.value,
    },
}


SATISFIED_STAGE_STATUSES = {
    StageStatus.SUCCEEDED.value,
    StageStatus.SKIPPED.value,
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


class LifecycleError(ValueError):
    pass


# ---------------------------------------------------------------
# Lifecycle decision
# ---------------------------------------------------------------


@dataclass(frozen=True)
class LifecycleDecision:
    """
    Result from evaluating progression through
    the orchestration lifecycle.

    allowed:
        The requested stage may proceed.

    stop:
        Automated orchestration must stop.

    stop_reason:
        Why the orchestration loop stops.

    missing_dependencies:
        Required stages that have not completed
        successfully or been intentionally
        skipped.
    """

    allowed: bool

    stage: Optional[str]

    stop: bool

    stop_reason: str

    reason: str

    missing_dependencies: Tuple[
        str,
        ...,
    ] = ()

    def to_dict(
        self,
    ) -> Dict[
        str,
        object,
    ]:
        return {
            "allowed": self.allowed,
            "stage": self.stage,
            "stop": self.stop,
            "stop_reason": (self.stop_reason),
            "reason": self.reason,
            "missing_dependencies": (list(self.missing_dependencies)),
        }


# ---------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------


def valid_stage(
    stage: str,
) -> bool:
    return stage in {item.value for item in StageName}


def stage_position(
    stage: str,
) -> int:
    if stage not in DEFAULT_STAGE_ORDER:
        raise LifecycleError(f"Unknown stage: {stage}")

    return DEFAULT_STAGE_ORDER.index(stage)


def stage_completed(
    run: OrchestrationRun,
    stage: str,
) -> bool:
    """
    Return whether the stage reached any
    terminal state.
    """

    result = run.stage(stage)

    return bool(result and result.status in TERMINAL_STAGE_STATUSES)


def stage_satisfied(
    run: OrchestrationRun,
    stage: str,
) -> bool:
    """
    Return whether a stage satisfies downstream
    dependencies.

    FAILED/BLOCKED/CANCELLED are terminal but do
    not satisfy downstream dependencies.
    """

    result = run.stage(stage)

    return bool(result and result.status in SATISFIED_STAGE_STATUSES)


def missing_dependencies(
    run: OrchestrationRun,
    stage: str,
) -> List[str]:
    if not valid_stage(stage):
        raise LifecycleError(f"Unknown stage: {stage}")

    return sorted(
        dependency
        for dependency in STAGE_DEPENDENCIES.get(
            stage,
            set(),
        )
        if not stage_satisfied(
            run,
            dependency,
        )
    )


# ---------------------------------------------------------------
# Stage-entry gate
# ---------------------------------------------------------------


def can_enter_stage(
    run: OrchestrationRun,
    stage: str,
) -> LifecycleDecision:
    """
    Determine whether a stage's lifecycle
    dependencies are satisfied.
    """

    run.finalize()

    if not valid_stage(stage):
        raise LifecycleError(f"Unknown stage: {stage}")

    existing = run.stage(stage)

    if existing is not None:
        return LifecycleDecision(
            allowed=False,
            stage=stage,
            stop=False,
            stop_reason=(StopReason.NONE.value),
            reason=(f"Stage {stage} already exists with status {existing.status}."),
        )

    missing = missing_dependencies(
        run,
        stage,
    )

    if missing:
        return LifecycleDecision(
            allowed=False,
            stage=stage,
            stop=False,
            stop_reason=(StopReason.NONE.value),
            reason=("Required lifecycle dependencies are not satisfied."),
            missing_dependencies=(tuple(missing)),
        )

    return LifecycleDecision(
        allowed=True,
        stage=stage,
        stop=False,
        stop_reason=(StopReason.NONE.value),
        reason=("Lifecycle dependencies are satisfied."),
    )


# ---------------------------------------------------------------
# Decision -> remediation gate
# ---------------------------------------------------------------


def decision_gate(
    *,
    approval_required: bool,
    action_available: bool,
    mutating_action: bool,
) -> LifecycleDecision:
    """
    Determine whether orchestration should
    continue from the decision layer into
    remediation.

    This does not approve an action.
    """

    if approval_required:
        return LifecycleDecision(
            allowed=False,
            stage=(StageName.REMEDIATION.value),
            stop=True,
            stop_reason=(StopReason.APPROVAL_REQUIRED.value),
            reason=("Decision requires approval before remediation can proceed."),
        )

    if not action_available:
        return LifecycleDecision(
            allowed=False,
            stage=(StageName.REMEDIATION.value),
            stop=False,
            stop_reason=(StopReason.NONE.value),
            reason=(
                "No remediation action is available; remediation should be skipped."
            ),
        )

    if mutating_action:
        reason = "Decision permits remediation evaluation."

    else:
        reason = (
            "Decision contains a "
            "non-mutating action that may "
            "pass through remediation in "
            "dry-run/evaluation mode."
        )

    return LifecycleDecision(
        allowed=True,
        stage=(StageName.REMEDIATION.value),
        stop=False,
        stop_reason=(StopReason.NONE.value),
        reason=reason,
    )


# ---------------------------------------------------------------
# Remediation -> self-healing gate
# ---------------------------------------------------------------


def remediation_gate(
    *,
    blocked: bool,
    failed: bool,
    changed: bool,
    dry_run: bool,
) -> LifecycleDecision:
    """
    Determine whether orchestration may enter
    the recovery/self-healing feedback layer.
    """

    if blocked:
        return LifecycleDecision(
            allowed=False,
            stage=(StageName.SELF_HEALING.value),
            stop=True,
            stop_reason=(StopReason.SAFETY_GUARDRAIL.value),
            reason=("Remediation was blocked by the controlled remediation layer."),
        )

    if failed:
        return LifecycleDecision(
            allowed=False,
            stage=(StageName.SELF_HEALING.value),
            stop=True,
            stop_reason=(StopReason.FAILURE.value),
            reason=(
                "Remediation failed before "
                "a normal healing-feedback "
                "path could continue."
            ),
        )

    if dry_run or not changed:
        return LifecycleDecision(
            allowed=True,
            stage=(StageName.SELF_HEALING.value),
            stop=False,
            stop_reason=(StopReason.NONE.value),
            reason=(
                "Self-healing may evaluate "
                "the lifecycle, but recovery "
                "must not be claimed from "
                "a dry-run or unchanged "
                "state."
            ),
        )

    return LifecycleDecision(
        allowed=True,
        stage=(StageName.SELF_HEALING.value),
        stop=False,
        stop_reason=(StopReason.NONE.value),
        reason=(
            "Remediation changed "
            "infrastructure; recovery and "
            "stabilization evaluation may "
            "proceed."
        ),
    )


# ---------------------------------------------------------------
# Self-healing -> outcome gate
# ---------------------------------------------------------------


def healing_gate(
    *,
    escalation_required: bool,
    outcome_available: bool,
) -> LifecycleDecision:
    """
    Determine whether orchestration should
    continue past self-healing.
    """

    if escalation_required:
        return LifecycleDecision(
            allowed=False,
            stage=(StageName.OUTCOME.value),
            stop=True,
            stop_reason=(StopReason.ESCALATION_REQUIRED.value),
            reason=(
                "Self-healing requires "
                "human escalation; automated "
                "progression must stop."
            ),
        )

    if not outcome_available:
        return LifecycleDecision(
            allowed=False,
            stage=(StageName.OUTCOME.value),
            stop=False,
            stop_reason=(StopReason.NONE.value),
            reason=("Self-healing has not produced a final outcome yet."),
        )

    return LifecycleDecision(
        allowed=True,
        stage=(StageName.OUTCOME.value),
        stop=False,
        stop_reason=(StopReason.NONE.value),
        reason=("Self-healing produced an outcome that may be recorded."),
    )


# ---------------------------------------------------------------
# Next stage
# ---------------------------------------------------------------


def next_stage(
    run: OrchestrationRun,
) -> LifecycleDecision:
    """
    Return the next structurally eligible stage.

    Conditional business gates such as approval
    are evaluated separately.
    """

    run.finalize()

    if run.stop_reason != StopReason.NONE.value:
        return LifecycleDecision(
            allowed=False,
            stage=None,
            stop=True,
            stop_reason=(run.stop_reason),
            reason=(f"Run already carries stop reason {run.stop_reason}."),
        )

    if run.current_stage is not None:
        return LifecycleDecision(
            allowed=False,
            stage=(run.current_stage),
            stop=False,
            stop_reason=(StopReason.NONE.value),
            reason=(
                "The current orchestration stage must finish before another begins."
            ),
        )

    for stage in DEFAULT_STAGE_ORDER:
        if run.stage(stage) is not None:
            continue

        decision = can_enter_stage(
            run,
            stage,
        )

        if decision.allowed:
            return decision

    return LifecycleDecision(
        allowed=False,
        stage=None,
        stop=False,
        stop_reason=(StopReason.COMPLETED.value),
        reason=("No remaining lifecycle stage is ready to run."),
    )


# ---------------------------------------------------------------
# Lifecycle facade
# ---------------------------------------------------------------


class OrchestrationLifecycle:
    """
    Reusable orchestration lifecycle evaluator.

    It evaluates progression only. It does not
    invoke any OSHA subsystem.
    """

    def can_enter(
        self,
        run: OrchestrationRun,
        stage: str,
    ) -> LifecycleDecision:
        return can_enter_stage(
            run,
            stage,
        )

    def next(
        self,
        run: OrchestrationRun,
    ) -> LifecycleDecision:
        return next_stage(run)

    def decision_gate(
        self,
        *,
        approval_required: bool,
        action_available: bool,
        mutating_action: bool,
    ) -> LifecycleDecision:
        return decision_gate(
            approval_required=(approval_required),
            action_available=(action_available),
            mutating_action=(mutating_action),
        )

    def remediation_gate(
        self,
        *,
        blocked: bool,
        failed: bool,
        changed: bool,
        dry_run: bool,
    ) -> LifecycleDecision:
        return remediation_gate(
            blocked=blocked,
            failed=failed,
            changed=changed,
            dry_run=dry_run,
        )

    def healing_gate(
        self,
        *,
        escalation_required: bool,
        outcome_available: bool,
    ) -> LifecycleDecision:
        return healing_gate(
            escalation_required=(escalation_required),
            outcome_available=(outcome_available),
        )
