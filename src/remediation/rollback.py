from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.remediation.adapters import (
    AdapterError,
    AdapterNotFoundError,
    AdapterRegistry,
    default_adapter_registry,
)

from src.remediation.schema import (
    ExecutionResult,
    ExecutionStatus,
    RemediationEvidence,
    RemediationMode,
    RemediationRequest,
    RollbackResult,
    RollbackStatus,
    VerificationResult,
    VerificationStatus,
    utc_now,
)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class RollbackConfig:
    """
    Rollback decision configuration.

    Rollback execution remains disabled unless
    every surrounding execution boundary also
    permits it.
    """

    rollback_enabled: bool = False

    rollback_on_execution_failure: bool = True

    rollback_on_verification_failure: bool = True

    rollback_on_inconclusive_verification: bool = False

    require_changed_state: bool = True

    require_declared_rollback: bool = True

    preferred_adapter: Optional[str] = None


# ---------------------------------------------------------------
# Rollback decision
# ---------------------------------------------------------------


@dataclass(frozen=True)
class RollbackDecision:
    """
    Explanation of whether rollback should be
    attempted.

    required:
        Whether the current lifecycle state
        indicates rollback should happen.

    available:
        Whether the request and adapter provide
        a rollback path.

    allowed:
        Whether this component permits actually
        invoking rollback.

    Dry-run may report required/available while
    still remaining non-mutating.
    """

    required: bool

    available: bool

    allowed: bool

    reason: str

    adapter_name: Optional[str] = None

    rollback_action_type: Optional[str] = None

    attributes: Dict[
        str,
        Any,
    ] = None

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "required": (self.required),
            "available": (self.available),
            "allowed": (self.allowed),
            "reason": self.reason,
            "adapter_name": (self.adapter_name),
            "rollback_action_type": (self.rollback_action_type),
            "attributes": dict(self.attributes or {}),
        }


# ---------------------------------------------------------------
# State analysis
# ---------------------------------------------------------------


def rollback_required(
    execution: ExecutionResult,
    verification: Optional[VerificationResult] = None,
    *,
    config: Optional[RollbackConfig] = None,
) -> tuple[
    bool,
    str,
]:
    """
    Decide whether lifecycle state requires
    rollback.

    The e decision is independent of whether
    rollback is actually available.
    """

    configuration = config or RollbackConfig()

    execution.finalize()

    if configuration.require_changed_state and not execution.changed:
        return (
            False,
            ("Execution did not report an infrastructure state change."),
        )

    if configuration.rollback_on_execution_failure and execution.status in {
        ExecutionStatus.FAILED.value,
        ExecutionStatus.TIMED_OUT.value,
    }:
        return (
            True,
            ("Execution failed after a reported state change."),
        )

    if verification is None:
        return (
            False,
            ("No post-execution verification result is available."),
        )

    verification.finalize()

    if (
        configuration.rollback_on_verification_failure
        and verification.status == VerificationStatus.FAILED.value
    ):
        return (
            True,
            ("Post-execution verification failed."),
        )

    if (
        configuration.rollback_on_inconclusive_verification
        and verification.status == VerificationStatus.INCONCLUSIVE.value
    ):
        return (
            True,
            ("Post-execution verification was inconclusive."),
        )

    return (
        False,
        ("Current execution and verification state does not require rollback."),
    )


# ---------------------------------------------------------------
# Availability
# ---------------------------------------------------------------


def resolve_rollback_adapter(
    request: RemediationRequest,
    registry: AdapterRegistry,
    *,
    preferred_adapter: Optional[str] = None,
):
    """
    Resolve the adapter associated with rollback.

    Adapter resolution alone does not execute
    rollback.
    """

    request.finalize()

    adapter = registry.resolve(
        request,
        preferred_adapter=(preferred_adapter),
    )

    if not (adapter.supports_rollback(request)):
        raise AdapterNotFoundError(
            "Resolved adapter does not "
            "support rollback action "
            f"{request.rollback_action_type}"
        )

    return adapter


def evaluate_rollback(
    request: RemediationRequest,
    execution: ExecutionResult,
    verification: Optional[VerificationResult] = None,
    *,
    registry: Optional[AdapterRegistry] = None,
    config: Optional[RollbackConfig] = None,
) -> RollbackDecision:
    """
    Evaluate rollback requirement, availability
    and permission without executing anything.
    """

    configuration = config or RollbackConfig()

    adapters = registry or default_adapter_registry()

    request.finalize()

    required, reason = rollback_required(
        execution,
        verification,
        config=configuration,
    )

    if not (request.rollback_action_type):
        return RollbackDecision(
            required=required,
            available=False,
            allowed=False,
            reason=(reason + " No rollback action is declared."),
            rollback_action_type=None,
            attributes={
                "rollback_enabled": (configuration.rollback_enabled),
                "mode": request.mode,
            },
        )

    adapter_name = None

    available = False

    try:
        adapter = resolve_rollback_adapter(
            request,
            adapters,
            preferred_adapter=(configuration.preferred_adapter),
        )

        adapter_name = adapter.capability.adapter_name

        available = True

    except (
        AdapterNotFoundError,
        AdapterError,
    ):
        available = False

    # Dry-run can evaluate rollback capability but
    # never performs infrastructure mutation.
    if request.mode == RemediationMode.DRY_RUN.value:
        allowed = required and available

    else:
        allowed = required and available and configuration.rollback_enabled

    if not required:
        final_reason = reason

    elif not available:
        final_reason = reason + " No compatible rollback adapter is available."

    elif (
        request.mode == RemediationMode.EXECUTE.value
        and not configuration.rollback_enabled
    ):
        final_reason = reason + " Rollback execution is disabled."

    elif request.mode == RemediationMode.DRY_RUN.value:
        final_reason = reason + " Rollback will be simulated only."

    else:
        final_reason = reason + " Rollback is permitted."

    return RollbackDecision(
        required=required,
        available=available,
        allowed=allowed,
        reason=final_reason,
        adapter_name=(adapter_name),
        rollback_action_type=(request.rollback_action_type),
        attributes={
            "rollback_enabled": (configuration.rollback_enabled),
            "mode": request.mode,
            "execution_status": (execution.status),
            "execution_changed": (execution.changed),
            "verification_status": (
                verification.status if verification is not None else None
            ),
        },
    )


# ---------------------------------------------------------------
# Rollback handling
# ---------------------------------------------------------------


def no_rollback_result(
    request: RemediationRequest,
    decision: RollbackDecision,
) -> RollbackResult:
    """
    Represent a non-executed rollback state.
    """

    if not decision.required:
        status = RollbackStatus.NOT_REQUIRED.value

    elif not decision.available:
        status = RollbackStatus.UNAVAILABLE.value

    elif not decision.allowed:
        status = RollbackStatus.REQUIRED.value

    else:
        status = RollbackStatus.AVAILABLE.value

    return RollbackResult(
        status=status,
        success=False,
        action_type=(request.rollback_action_type),
        adapter_name=(decision.adapter_name),
        reason=decision.reason,
        evidence=[
            RemediationEvidence(
                evidence_type=("rollback_decision"),
                description=(decision.reason),
                score=(1.0 if decision.available else 0.0),
                source_id=(request.request_id),
                resource_ids=list(request.target_resource_ids),
                attributes=(decision.to_dict()),
            )
        ],
    ).finalize()


def perform_rollback(
    request: RemediationRequest,
    execution: ExecutionResult,
    verification: Optional[VerificationResult] = None,
    *,
    registry: Optional[AdapterRegistry] = None,
    config: Optional[RollbackConfig] = None,
) -> RollbackResult:
    """
    Evaluate and, when permitted, dispatch
    rollback through the original adapter
    boundary.

    With the current default registry no real
    rollback execution is possible.
    """

    configuration = config or RollbackConfig()

    adapters = registry or default_adapter_registry()

    decision = evaluate_rollback(
        request,
        execution,
        verification,
        registry=adapters,
        config=configuration,
    )

    if not decision.allowed:
        return no_rollback_result(
            request,
            decision,
        )

    try:
        adapter = resolve_rollback_adapter(
            request,
            adapters,
            preferred_adapter=(configuration.preferred_adapter),
        )

        # In dry-run this routes to the
        # non-mutating DryRunAdapter.rollback().
        result = adapter.rollback(request)

        result.attributes["rollback_decision"] = decision.to_dict()

        return result.finalize()

    except (
        AdapterNotFoundError,
        AdapterError,
    ) as exc:
        return RollbackResult(
            status=(RollbackStatus.FAILED.value),
            success=False,
            action_type=(request.rollback_action_type),
            adapter_name=(decision.adapter_name),
            started_at=utc_now(),
            finished_at=utc_now(),
            reason=("Rollback adapter failed or could not be resolved."),
            evidence=[
                RemediationEvidence(
                    evidence_type=("rollback_failure"),
                    description=(
                        "Rollback could not be completed through the adapter boundary."
                    ),
                    score=0.0,
                    source_id=(request.request_id),
                    resource_ids=list(request.target_resource_ids),
                    attributes={
                        "error_type": (type(exc).__name__),
                    },
                )
            ],
        ).finalize()

    except Exception as exc:
        # Avoid propagating provider internals or
        # arbitrary exception content.
        return RollbackResult(
            status=(RollbackStatus.FAILED.value),
            success=False,
            action_type=(request.rollback_action_type),
            adapter_name=(decision.adapter_name),
            started_at=utc_now(),
            finished_at=utc_now(),
            reason=("Unexpected rollback failure."),
            evidence=[
                RemediationEvidence(
                    evidence_type=("rollback_failure"),
                    description=("Unexpected rollback adapter failure."),
                    score=0.0,
                    source_id=(request.request_id),
                    resource_ids=list(request.target_resource_ids),
                    attributes={
                        "error_type": (type(exc).__name__),
                    },
                )
            ],
        ).finalize()


# ---------------------------------------------------------------
# Batch
# ---------------------------------------------------------------


def perform_rollbacks(
    items: Iterable[
        tuple[
            RemediationRequest,
            ExecutionResult,
            Optional[VerificationResult],
        ]
    ],
    *,
    registry: Optional[AdapterRegistry] = None,
    config: Optional[RollbackConfig] = None,
) -> List[RollbackResult]:
    adapters = registry or default_adapter_registry()

    configuration = config or RollbackConfig()

    return [
        perform_rollback(
            request,
            execution,
            verification,
            registry=adapters,
            config=configuration,
        )
        for (
            request,
            execution,
            verification,
        ) in items
    ]


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------


def rollback_summary(
    results: Iterable[RollbackResult],
) -> Dict[str, Any]:
    values = list(results)

    statuses: Dict[
        str,
        int,
    ] = {}

    for result in values:
        statuses[result.status] = (
            statuses.get(
                result.status,
                0,
            )
            + 1
        )

    return {
        "rollback_count": len(values),
        "successful_count": sum(1 for result in values if result.success),
        "required_count": sum(
            1 for result in values if result.status == RollbackStatus.REQUIRED.value
        ),
        "unavailable_count": sum(
            1 for result in values if result.status == RollbackStatus.UNAVAILABLE.value
        ),
        "failed_count": sum(
            1 for result in values if result.status == RollbackStatus.FAILED.value
        ),
        "statuses": statuses,
    }


class RollbackManager:
    """
    Controlled rollback coordinator.

    The default configuration cannot perform real
    rollback execution.
    """

    def __init__(
        self,
        *,
        registry: Optional[AdapterRegistry] = None,
        config: Optional[RollbackConfig] = None,
    ) -> None:
        self.registry = registry or default_adapter_registry()

        self.config = config or RollbackConfig()

    def evaluate(
        self,
        request: RemediationRequest,
        execution: ExecutionResult,
        verification: Optional[VerificationResult] = None,
    ) -> RollbackDecision:
        return evaluate_rollback(
            request,
            execution,
            verification,
            registry=self.registry,
            config=self.config,
        )

    def rollback(
        self,
        request: RemediationRequest,
        execution: ExecutionResult,
        verification: Optional[VerificationResult] = None,
    ) -> RollbackResult:
        return perform_rollback(
            request,
            execution,
            verification,
            registry=self.registry,
            config=self.config,
        )
