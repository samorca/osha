from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
)

from src.orchestrator.coordinator import (
    CoordinationResult,
    CoordinatorConfig,
    CoordinatorContext,
    OrchestrationCoordinator,
    StageExecution,
    StageHandler,
)

from src.orchestrator.lifecycle import (
    DEFAULT_STAGE_ORDER,
    OrchestrationLifecycle,
)

from src.orchestrator.schema import (
    OrchestrationContext,
    OrchestrationResult,
    StageName,
    StopReason,
)

from src.orchestrator.state import (
    OrchestrationStateManager,
    skip_stage,
)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """
    Composition-root configuration.

    The pipeline remains dry-run-only by default.

    required_handlers:
        Handlers that must exist when constructing
        the pipeline.

    stage_order:
        Ordered stages considered during a run.
    """

    required_handlers: tuple[
        str,
        ...,
    ] = (
        StageName.
        CORRELATION.value,

        StageName.
        INTELLIGENCE.value,

        StageName.
        DECISION.value,

        StageName.
        REMEDIATION.value,

        StageName.
        SELF_HEALING.value,

        StageName.
        OUTCOME.value,
    )

    stage_order: tuple[
        str,
        ...,
    ] = DEFAULT_STAGE_ORDER

    require_dry_run: bool = True

    fail_fast: bool = True

    stop_on_blocked_stage: bool = True

    redact_exception_messages: bool = True

    def validate(
        self,
    ) -> None:
        valid = {
            item.value
            for item in StageName
        }

        unknown_required = sorted(
            set(
                self.required_handlers
            )
            - valid
        )

        unknown_order = sorted(
            set(
                self.stage_order
            )
            - valid
        )

        if unknown_required:
            raise ValueError(
                "Unknown required handlers: "
                f"{unknown_required}"
            )

        if unknown_order:
            raise ValueError(
                "Unknown stages in "
                f"stage_order: {unknown_order}"
            )

        if (
            len(
                set(
                    self.stage_order
                )
            )
            != len(
                self.stage_order
            )
        ):
            raise ValueError(
                "stage_order contains "
                "duplicate stages"
            )


# ---------------------------------------------------------------
# Pipeline input
# ---------------------------------------------------------------

@dataclass
class PipelineInput:
    """
    Input for one orchestration run.

    initial_artifacts supports alternate entry
    points.

    Examples:

      already collected telemetry
      existing correlation result
      existing incident/intelligence result

    Initial artifacts remain authoritative
    outputs from their original subsystem.
    """

    context: OrchestrationContext

    initial_artifacts: Dict[
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

    def finalize(
        self,
    ) -> "PipelineInput":
        self.context.finalize()

        valid = {
            item.value
            for item in StageName
        }

        unknown = sorted(
            set(
                self.initial_artifacts
            )
            - valid
        )

        if unknown:
            raise ValueError(
                "Unknown initial artifact "
                f"stages: {unknown}"
            )

        return self


# ---------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------

@dataclass
class PipelineResult:
    coordination: CoordinationResult

    metadata: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    @property
    def run(
        self,
    ):
        return self.coordination.run

    @property
    def artifacts(
        self,
    ) -> Dict[str, Any]:
        return (
            self.coordination.artifacts
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "coordination": (
                self.coordination.to_dict()
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# ---------------------------------------------------------------
# Generic object access
# ---------------------------------------------------------------

def read_value(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    """
    Read a field from either a mapping or an
    object.

    This lets the composition root inspect
    subsystem contracts without serializing
    everything into dictionaries first.
    """

    if value is None:
        return default

    if isinstance(
        value,
        Mapping,
    ):
        return value.get(
            name,
            default,
        )

    return getattr(
        value,
        name,
        default,
    )


# ---------------------------------------------------------------
# Decision interpretation
# ---------------------------------------------------------------

def selected_action(
    decision_output: Any,
) -> Any:
    """
    Resolve the action selected by the decision
    subsystem.

    Supported contract shapes:

      selected_action
      action
      first item in actions
    """

    action = read_value(
        decision_output,
        "selected_action",
    )

    if action is not None:
        return action

    action = read_value(
        decision_output,
        "action",
    )

    if action is not None:
        return action

    actions = read_value(
        decision_output,
        "actions",
        [],
    )

    if actions:
        return list(
            actions
        )[0]

    return None


def approval_required(
    decision_output: Any,
    action: Any,
) -> bool:
    direct = read_value(
        decision_output,
        "approval_required",
    )

    if direct is not None:
        return bool(
            direct
        )

    approval = read_value(
        action,
        "approval",
    )

    required = read_value(
        approval,
        "required",
    )

    if required is not None:
        return bool(
            required
        )

    status = str(
        read_value(
            approval,
            "status",
            "",
        )
    ).lower()

    return status in {
        "required",
        "pending",
        "awaiting_approval",
    }


def mutating_action(
    action: Any,
) -> bool:
    """
    Determine whether an action is expected to
    change infrastructure.

    An explicit 'mutating' field wins.

    The fallback action types are OSHA policy,
    not a universal classification.
    """

    explicit = read_value(
        action,
        "mutating",
    )

    if explicit is not None:
        return bool(
            explicit
        )

    action_type = str(
        read_value(
            action,
            "action_type",
            "",
        )
    )

    return action_type not in {
        "",
        "observe_resource",
        "collect_diagnostics",
        "notify_operator",
    }


# ---------------------------------------------------------------
# Remediation interpretation
# ---------------------------------------------------------------

def remediation_signals(
    output: Any,
) -> Dict[str, bool]:
    request = read_value(
        output,
        "request",
    )

    execution = read_value(
        output,
        "execution",
    )

    status = str(
        read_value(
            output,
            "status",
            "",
        )
    ).lower()

    execution_status = str(
        read_value(
            execution,
            "status",
            "",
        )
    ).lower()

    mode = str(
        read_value(
            request,
            "mode",
            "",
        )
    ).lower()

    return {
        "blocked": (
            status == "blocked"
            or execution_status
            == "blocked"
        ),
        "failed": (
            status == "failed"
            or execution_status
            in {
                "failed",
                "timed_out",
                "cancelled",
            }
        ),
        "changed": bool(
            read_value(
                execution,
                "changed",
                False,
            )
        ),
        "dry_run": (
            mode == "dry_run"
            or execution_status
            == "dry_run"
        ),
    }


# ---------------------------------------------------------------
# Healing interpretation
# ---------------------------------------------------------------

def healing_signals(
    output: Any,
) -> Dict[str, bool]:
    session = read_value(
        output,
        "session",
        output,
    )

    escalation = read_value(
        session,
        "escalation",
    )

    outcome = read_value(
        session,
        "outcome",
    )

    return {
        "escalation_required": bool(
            read_value(
                escalation,
                "required",
                False,
            )
        ),
        "outcome_available": (
            outcome is not None
        ),
    }


# ---------------------------------------------------------------
# Orchestrator pipeline
# ---------------------------------------------------------------

class OrchestratorPipeline:
    """
    Composition root for OSHA orchestration
    handlers.

    The pipeline:

      validates required handler registration
      supports alternate entry points
      enforces decision/remediation/healing gates
      maintains dry-run as the default boundary
      returns orchestration state plus artifacts

    It does not implement infrastructure provider
    operations.
    """

    def __init__(
        self,
        *,
        handlers: Mapping[
            str,
            StageHandler,
        ],
        config: Optional[
            PipelineConfig
        ] = None,
        lifecycle: Optional[
            OrchestrationLifecycle
        ] = None,
        state: Optional[
            OrchestrationStateManager
        ] = None,
    ) -> None:
        self.config = (
            config
            or PipelineConfig()
        )

        self.config.validate()

        self.handlers = dict(
            handlers
        )

        missing = sorted(
            set(
                self.config.
                required_handlers
            )
            - set(
                self.handlers
            )
        )

        if missing:
            raise ValueError(
                "Missing required "
                "orchestrator handlers: "
                f"{missing}"
            )

        self.lifecycle = (
            lifecycle
            or OrchestrationLifecycle()
        )

        self.state = (
            state
            or OrchestrationStateManager()
        )

        self.coordinator = (
            OrchestrationCoordinator(
                handlers=(
                    self._wrapped_handlers()
                ),
                lifecycle=self.lifecycle,
                state=self.state,
                config=CoordinatorConfig(
                    fail_fast=(
                        self.config.
                        fail_fast
                    ),
                    stop_on_blocked_stage=(
                        self.config.
                        stop_on_blocked_stage
                    ),
                    redact_exception_messages=(
                        self.config.
                        redact_exception_messages
                    ),
                ),
            )
        )

    # ---------------------------------------------------------
    # Gate-aware handlers
    # ---------------------------------------------------------

    def _wrapped_handlers(
        self,
    ) -> Dict[
        str,
        StageHandler,
    ]:
        wrapped: Dict[
            str,
            StageHandler,
        ] = {}

        for stage, handler in (
            self.handlers.items()
        ):
            if (
                stage
                == StageName.
                REMEDIATION.value
            ):
                wrapped[
                    stage
                ] = (
                    self.
                    _remediation_handler(
                        handler
                    )
                )

            elif (
                stage
                == StageName.
                SELF_HEALING.value
            ):
                wrapped[
                    stage
                ] = (
                    self.
                    _healing_handler(
                        handler
                    )
                )

            elif (
                stage
                == StageName.
                OUTCOME.value
            ):
                wrapped[
                    stage
                ] = (
                    self.
                    _outcome_handler(
                        handler
                    )
                )

            else:
                wrapped[
                    stage
                ] = handler

        return wrapped

    def _remediation_handler(
        self,
        handler: StageHandler,
    ) -> StageHandler:
        def wrapped(
            context: CoordinatorContext,
        ) -> StageExecution:
            decision_output = (
                context.get(
                    StageName.
                    DECISION.value
                )
            )

            action = selected_action(
                decision_output
            )

            gate = (
                self.lifecycle.
                decision_gate(
                    approval_required=(
                        approval_required(
                            decision_output,
                            action,
                        )
                    ),
                    action_available=(
                        action is not None
                    ),
                    mutating_action=(
                        mutating_action(
                            action
                        )
                    ),
                )
            )

            if gate.stop:
                return StageExecution(
                    blocked=True,
                    stop_reason=(
                        gate.stop_reason
                    ),
                    reason=gate.reason,
                    attributes={
                        "gate": (
                            gate.to_dict()
                        ),
                    },
                )

            if not gate.allowed:
                return StageExecution(
                    skip=True,
                    reason=gate.reason,
                    attributes={
                        "gate": (
                            gate.to_dict()
                        ),
                    },
                )

            result = handler(
                context
            )

            if isinstance(
                result,
                StageExecution,
            ):
                return result

            return StageExecution(
                output=result
            )

        return wrapped

    def _healing_handler(
        self,
        handler: StageHandler,
    ) -> StageHandler:
        def wrapped(
            context: CoordinatorContext,
        ) -> StageExecution:
            signals = (
                remediation_signals(
                    context.get(
                        StageName.
                        REMEDIATION.value
                    )
                )
            )

            gate = (
                self.lifecycle.
                remediation_gate(
                    **signals
                )
            )

            if gate.stop:
                return StageExecution(
                    blocked=True,
                    stop_reason=(
                        gate.stop_reason
                    ),
                    reason=gate.reason,
                    attributes={
                        "gate": (
                            gate.to_dict()
                        ),
                        **signals,
                    },
                )

            result = handler(
                context
            )

            execution = (
                result
                if isinstance(
                    result,
                    StageExecution,
                )
                else StageExecution(
                    output=result
                )
            )

            execution.attributes.setdefault(
                "gate",
                gate.to_dict(),
            )

            return execution

        return wrapped

    def _outcome_handler(
        self,
        handler: StageHandler,
    ) -> StageHandler:
        def wrapped(
            context: CoordinatorContext,
        ) -> StageExecution:
            signals = healing_signals(
                context.get(
                    StageName.
                    SELF_HEALING.value
                )
            )

            gate = (
                self.lifecycle.
                healing_gate(
                    **signals
                )
            )

            if gate.stop:
                return StageExecution(
                    blocked=True,
                    stop_reason=(
                        gate.stop_reason
                    ),
                    reason=gate.reason,
                    attributes={
                        "gate": (
                            gate.to_dict()
                        ),
                        **signals,
                    },
                )

            if not gate.allowed:
                return StageExecution(
                    skip=True,
                    reason=gate.reason,
                    attributes={
                        "gate": (
                            gate.to_dict()
                        ),
                        **signals,
                    },
                )

            result = handler(
                context
            )

            if isinstance(
                result,
                StageExecution,
            ):
                return result

            return StageExecution(
                output=result
            )

        return wrapped

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------

    def run(
        self,
        pipeline_input: PipelineInput,
    ) -> PipelineResult:
        pipeline_input.finalize()

        if (
            self.config.require_dry_run
            and not pipeline_input.
            context.dry_run
        ):
            raise ValueError(
                "Orchestrator pipeline "
                "currently requires "
                "dry_run context"
            )

        run = (
            self.coordinator.
            create_run(
                pipeline_input.context
            )
        )

        context = CoordinatorContext(
            run=run,
            artifacts=dict(
                pipeline_input.
                initial_artifacts
            ),
            metadata=dict(
                pipeline_input.metadata
            ),
        )

        self.state.start(
            run
        )

        for stage in (
            self.config.stage_order
        ):
            if (
                run.stop_reason
                != StopReason.NONE.value
            ):
                break

            # An initial artifact means the stage
            # was completed outside this run.
            # Mark it SKIPPED with provenance so
            # downstream dependencies are still
            # satisfied.
            if stage in context.artifacts:
                skip_stage(
                    run,
                    stage,
                    reason=(
                        "Stage output was "
                        "supplied as an "
                        "initial artifact."
                    ),
                )

                stage_result = (
                    run.stage(stage)
                )

                if stage_result is not None:
                    stage_result.output_count = 1

                    stage_result.attributes[
                        "initial_artifact"
                    ] = True

                continue

            self.coordinator.run_stage(
                context,
                stage,
            )

        self.state.finish(
            run
        )

        return PipelineResult(
            coordination=(
                CoordinationResult(
                    run=run,
                    artifacts=(
                        context.artifacts
                    ),
                )
            ),
            metadata={
                "dry_run": (
                    pipeline_input.
                    context.dry_run
                ),
                "configured_stage_order": (
                    list(
                        self.config.
                        stage_order
                    )
                ),
            },
        )

    # ---------------------------------------------------------
    # Batch
    # ---------------------------------------------------------

    def run_many(
        self,
        inputs: Iterable[
            PipelineInput
        ],
    ) -> OrchestrationResult:
        results = [
            self.run(
                item
            )
            for item in inputs
        ]

        return OrchestrationResult(
            runs=[
                item.run
                for item in results
            ],
            metadata={
                "dry_run_required": (
                    self.config.
                    require_dry_run
                ),
                "run_count": len(
                    results
                ),
            },
        ).finalize()
