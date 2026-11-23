from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.orchestrator.lifecycle import (
    LifecycleDecision,
    OrchestrationLifecycle,
)

from src.orchestrator.schema import (
    OrchestrationContext,
    OrchestrationRun,
    StageName,
    StageStatus,
    StopReason,
)

from src.orchestrator.state import (
    OrchestrationStateManager,
    end_stage,
    skip_stage,
)


StageHandler = Callable[
    ["CoordinatorContext"],
    "StageExecution",
]


# ---------------------------------------------------------------
# Stage execution result
# ---------------------------------------------------------------

@dataclass
class StageExecution:
    """
    Generic result returned by a coordinator
    stage handler.

    output:
        Authoritative subsystem output.

    output_ids:
        IDs that should be referenced from the
        orchestration StageResult.

    skip:
        Stage was intentionally skipped.

    blocked:
        Stage reached a controlled block rather
        than an unexpected failure.

    stop_reason:
        Structured reason for stopping further
        orchestration.
    """

    output: Any = None

    output_ids: List[
        str
    ] = field(
        default_factory=list
    )

    output_count: Optional[
        int
    ] = None

    attributes: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    skip: bool = False

    blocked: bool = False

    stop_reason: str = (
        StopReason.NONE.value
    )

    reason: Optional[
        str
    ] = None

    def count(
        self,
    ) -> int:
        if (
            self.output_count
            is not None
        ):
            return max(
                0,
                int(
                    self.output_count
                ),
            )

        if self.output is None:
            return 0

        if isinstance(
            self.output,
            (
                list,
                tuple,
                set,
                frozenset,
                dict,
            ),
        ):
            return len(
                self.output
            )

        return 1


# ---------------------------------------------------------------
# Coordinator context
# ---------------------------------------------------------------

@dataclass
class CoordinatorContext:
    """
    Mutable in-process context for one run.

    artifacts contains subsystem outputs keyed by
    stage name.

    Detailed outputs are not copied into
    OrchestrationRun. Only references, counts and
    stage metadata are recorded there.
    """

    run: OrchestrationRun

    artifacts: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    metadata: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def put(
        self,
        stage: str,
        value: Any,
    ) -> None:
        self.artifacts[
            stage
        ] = value

    def get(
        self,
        stage: str,
        default: Any = None,
    ) -> Any:
        return self.artifacts.get(
            stage,
            default,
        )


# ---------------------------------------------------------------
# Coordinator configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class CoordinatorConfig:
    """
    Coordinator failure policy.

    redact_exception_messages:
        Avoid placing unexpected exception text
        into orchestration state by default.
    """

    fail_fast: bool = True

    stop_on_blocked_stage: bool = True

    redact_exception_messages: bool = True


# ---------------------------------------------------------------
# Coordinator result
# ---------------------------------------------------------------

@dataclass
class CoordinationResult:
    run: OrchestrationRun

    artifacts: Dict[
        str,
        Any,
    ]

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "artifact_stages": sorted(
                self.artifacts
            ),
        }


# ---------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------

class OrchestrationCoordinator:
    """
    Coordinate registered OSHA stage handlers.

    The coordinator does not implement provider
    commands and does not replace the underlying
    subsystem pipelines.
    """

    def __init__(
        self,
        *,
        handlers: Optional[
            Mapping[
                str,
                StageHandler,
            ]
        ] = None,
        lifecycle: Optional[
            OrchestrationLifecycle
        ] = None,
        state: Optional[
            OrchestrationStateManager
        ] = None,
        config: Optional[
            CoordinatorConfig
        ] = None,
    ) -> None:
        self.handlers: Dict[
            str,
            StageHandler,
        ] = dict(
            handlers
            or {}
        )

        self.lifecycle = (
            lifecycle
            or OrchestrationLifecycle()
        )

        self.state = (
            state
            or OrchestrationStateManager()
        )

        self.config = (
            config
            or CoordinatorConfig()
        )

    # ---------------------------------------------------------
    # Registration
    # ---------------------------------------------------------

    def register(
        self,
        stage: str,
        handler: StageHandler,
    ) -> None:
        if (
            stage
            not in {
                item.value
                for item in StageName
            }
        ):
            raise ValueError(
                "Unknown orchestration "
                f"stage: {stage}"
            )

        if stage in self.handlers:
            raise ValueError(
                "Handler already registered "
                f"for stage: {stage}"
            )

        self.handlers[
            stage
        ] = handler

    # ---------------------------------------------------------
    # Run creation
    # ---------------------------------------------------------

    def create_run(
        self,
        context: OrchestrationContext,
    ) -> OrchestrationRun:
        return OrchestrationRun(
            context=context
        ).finalize()

    # ---------------------------------------------------------
    # Lifecycle gates
    # ---------------------------------------------------------

    def _apply_gate(
        self,
        run: OrchestrationRun,
        gate: LifecycleDecision,
    ) -> bool:
        if gate.allowed:
            return True

        if gate.stop:
            run.stop_reason = (
                gate.stop_reason
            )

            run.attributes[
                "stop_gate"
            ] = gate.to_dict()

        return False

    # ---------------------------------------------------------
    # Input count
    # ---------------------------------------------------------

    def _input_count(
        self,
        context: CoordinatorContext,
        stage: str,
    ) -> int:
        dependencies = {
            StageName.
            COLLECTION.value: None,

            StageName.
            TELEMETRY.value: (
                StageName.
                COLLECTION.value
            ),

            StageName.
            CORRELATION.value: (
                StageName.
                TELEMETRY.value
            ),

            StageName.
            INTELLIGENCE.value: (
                StageName.
                CORRELATION.value
            ),

            StageName.
            MEMORY.value: (
                StageName.
                INTELLIGENCE.value
            ),

            StageName.
            DECISION.value: (
                StageName.
                INTELLIGENCE.value
            ),

            StageName.
            REMEDIATION.value: (
                StageName.
                DECISION.value
            ),

            StageName.
            SELF_HEALING.value: (
                StageName.
                REMEDIATION.value
            ),

            StageName.
            OUTCOME.value: (
                StageName.
                SELF_HEALING.value
            ),
        }

        source = dependencies.get(
            stage
        )

        value = (
            context.get(source)
            if source
            else None
        )

        if value is None:
            return 0

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
                frozenset,
                dict,
            ),
        ):
            return len(
                value
            )

        return 1

    # ---------------------------------------------------------
    # Execute one stage
    # ---------------------------------------------------------

    def run_stage(
        self,
        context: CoordinatorContext,
        stage: str,
    ) -> Optional[
        StageExecution
    ]:
        """
        Execute one registered stage handler.

        The handler result is translated into the
        orchestration state model.
        """

        gate = (
            self.lifecycle.can_enter(
                context.run,
                stage,
            )
        )

        if not self._apply_gate(
            context.run,
            gate,
        ):
            return None

        handler = self.handlers.get(
            stage
        )

        self.state.start_stage(
            context.run,
            stage,
            input_count=(
                self._input_count(
                    context,
                    stage,
                )
            ),
        )

        # Missing handlers are treated as
        # intentional skips in this generic
        # coordinator. pipeline.py can register
        # required handlers and enforce stricter
        # behavior.
        if handler is None:
            skip_stage(
                context.run,
                stage,
                reason=(
                    "No coordinator handler "
                    "is registered for this "
                    "optional stage."
                ),
            )

            return StageExecution(
                skip=True,
                reason=(
                    "No handler registered."
                ),
            )

        try:
            execution = handler(
                context
            )

            if not isinstance(
                execution,
                StageExecution,
            ):
                execution = (
                    StageExecution(
                        output=execution
                    )
                )

            # -----------------------------------------------
            # Controlled block
            # -----------------------------------------------

            if execution.blocked:
                stop_reason = (
                    execution.stop_reason
                )

                if (
                    stop_reason
                    == StopReason.NONE.value
                ):
                    stop_reason = (
                        StopReason.
                        BLOCKED.value
                    )

                self.state.block_stage(
                    context.run,
                    stage,
                    reason=(
                        execution.reason
                        or (
                            "Stage was "
                            "blocked."
                        )
                    ),
                    stop_reason=(
                        stop_reason
                    ),
                )

                return execution

            # -----------------------------------------------
            # Intentional skip
            # -----------------------------------------------

            if execution.skip:
                end_stage(
                    context.run,
                    stage,
                    status=(
                        StageStatus.
                        SKIPPED.value
                    ),
                    reason=(
                        execution.reason
                        or (
                            "Stage was "
                            "intentionally "
                            "skipped."
                        )
                    ),
                    attributes=(
                        execution.attributes
                    ),
                )

                return execution

            # -----------------------------------------------
            # Successful completion
            # -----------------------------------------------

            context.put(
                stage,
                execution.output,
            )

            self.state.complete_stage(
                context.run,
                stage,
                output_count=(
                    execution.count()
                ),
                output_ids=(
                    execution.output_ids
                ),
            )

            result = (
                context.run.stage(
                    stage
                )
            )

            if result is not None:
                result.attributes.update(
                    execution.attributes
                )

            return execution

        except Exception as exc:
            error = (
                "Unexpected stage "
                "execution failure."
                if self.config.
                redact_exception_messages
                else (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

            self.state.fail_stage(
                context.run,
                stage,
                reason=(
                    f"Stage {stage} failed."
                ),
                error=error,
            )

            if not self.config.fail_fast:
                return StageExecution(
                    reason=error
                )

            return None

    # ---------------------------------------------------------
    # Coordinate stages
    # ---------------------------------------------------------

    def coordinate(
        self,
        context: OrchestrationContext,
        *,
        stages: Optional[
            Iterable[str]
        ] = None,
    ) -> CoordinationResult:
        """
        Coordinate one orchestration run.

        By default all declared stage names are
        considered in schema order.

        Stage handlers must be registered by the
        composition root or pipeline.
        """

        run = self.create_run(
            context
        )

        coordinator_context = (
            CoordinatorContext(
                run=run
            )
        )

        self.state.start(
            run
        )

        selected = (
            list(stages)
            if stages is not None
            else [
                item.value
                for item in StageName
            ]
        )

        for stage in selected:
            if (
                run.stop_reason
                != StopReason.NONE.value
            ):
                break

            self.run_stage(
                coordinator_context,
                stage,
            )

            stage_result = (
                run.stage(stage)
            )

            if stage_result is None:
                continue

            if (
                stage_result.status
                == StageStatus.
                FAILED.value
                and self.config.
                fail_fast
            ):
                break

            if (
                stage_result.status
                == StageStatus.
                BLOCKED.value
                and self.config.
                stop_on_blocked_stage
            ):
                break

        self.state.finish(
            run
        )

        return CoordinationResult(
            run=run,
            artifacts=(
                coordinator_context.
                artifacts
            ),
        )

    # ---------------------------------------------------------
    # Conditional gates for stage handlers
    # ---------------------------------------------------------

    def gate_decision(
        self,
        *,
        approval_required: bool,
        action_available: bool,
        mutating_action: bool,
    ) -> LifecycleDecision:
        return (
            self.lifecycle.
            decision_gate(
                approval_required=(
                    approval_required
                ),
                action_available=(
                    action_available
                ),
                mutating_action=(
                    mutating_action
                ),
            )
        )

    def gate_remediation(
        self,
        *,
        blocked: bool,
        failed: bool,
        changed: bool,
        dry_run: bool,
    ) -> LifecycleDecision:
        return (
            self.lifecycle.
            remediation_gate(
                blocked=blocked,
                failed=failed,
                changed=changed,
                dry_run=dry_run,
            )
        )

    def gate_healing(
        self,
        *,
        escalation_required: bool,
        outcome_available: bool,
    ) -> LifecycleDecision:
        return (
            self.lifecycle.
            healing_gate(
                escalation_required=(
                    escalation_required
                ),
                outcome_available=(
                    outcome_available
                ),
            )
        )
