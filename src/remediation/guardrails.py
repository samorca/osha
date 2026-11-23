from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.remediation.schema import (
    GuardrailCheck,
    GuardrailStatus,
    RemediationEvidence,
    RemediationMode,
    RemediationRequest,
)


@dataclass(frozen=True)
class GuardrailConfig:
    """
    Hard remediation safety boundaries.

    The default configuration permits dry-run
    evaluation only. Actual execution remains
    disabled until explicitly enabled by a
    trusted runtime configuration.
    """

    execution_enabled: bool = False

    allowed_environments: tuple[
        str,
        ...,
    ] = (
        "development",
        "dev",
        "test",
        "staging",
        "local-dev",
    )

    protected_environments: tuple[
        str,
        ...,
    ] = (
        "production",
        "prod",
    )

    allowed_action_types: tuple[
        str,
        ...,
    ] = (
        "observe_resource",
        "collect_diagnostics",
        "notify_operator",
        "restart_service",
        "reload_service",
        "rollback_configuration",
        "isolate_resource",
        "scale_capacity",
    )

    maximum_target_resources: int = 1

    maximum_target_services: int = 1

    maximum_risk_score: float = 0.70

    require_approved_status: bool = True

    require_approval_ids: bool = True

    require_request_expiry: bool = True

    require_rollback_for_execution: bool = (
        True
    )

    forbid_unknown_parameters: bool = True

    allowed_parameter_names: tuple[
        str,
        ...,
    ] = (
        "maximum_targets",
        "timeout_seconds",
        "expected_state",
        "configuration_version",
        "capacity_delta",
        "reason",
    )

    def validate(
        self,
    ) -> None:
        if (
            self.maximum_target_resources
            < 1
        ):
            raise ValueError(
                "maximum_target_resources "
                "must be >= 1"
            )

        if (
            self.maximum_target_services
            < 1
        ):
            raise ValueError(
                "maximum_target_services "
                "must be >= 1"
            )

        if not (
            0.0
            <= self.maximum_risk_score
            <= 1.0
        ):
            raise ValueError(
                "maximum_risk_score must "
                "be between 0 and 1"
            )


@dataclass
class GuardrailEvaluation:
    """
    Combined guardrail result for one request.
    """

    allowed: bool = False

    checks: List[
        GuardrailCheck
    ] = field(
        default_factory=list
    )

    blocking_reasons: List[
        str
    ] = field(
        default_factory=list
    )

    warnings: List[
        str
    ] = field(
        default_factory=list
    )

    evaluated_at: str = field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    def finalize(
        self,
    ) -> "GuardrailEvaluation":
        self.blocking_reasons = list(
            dict.fromkeys(
                self.blocking_reasons
            )
        )

        self.warnings = list(
            dict.fromkeys(
                self.warnings
            )
        )

        self.allowed = not any(
            (
                check.blocking
                and check.status
                == GuardrailStatus.
                FAILED.value
            )
            for check in self.checks
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        self.finalize()

        return {
            "allowed": self.allowed,
            "checks": [
                check.to_dict()
                for check in self.checks
            ],
            "blocking_reasons": (
                self.blocking_reasons
            ),
            "warnings": (
                self.warnings
            ),
            "evaluated_at": (
                self.evaluated_at
            ),
        }


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def parse_timestamp(
    value: Optional[str],
) -> Optional[
    datetime
]:
    if not value:
        return None

    text = value.strip()

    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            text
        )

    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )


def make_check(
    guardrail_id: str,
    name: str,
    passed: bool,
    success_description: str,
    failure_description: str,
    *,
    blocking: bool = True,
    attributes: Optional[
        Dict[str, Any]
    ] = None,
) -> GuardrailCheck:
    description = (
        success_description
        if passed
        else failure_description
    )

    return GuardrailCheck(
        guardrail_id=guardrail_id,
        name=name,
        status=(
            GuardrailStatus.
            PASSED.value
            if passed
            else GuardrailStatus.
            FAILED.value
        ),
        description=description,
        blocking=blocking,
        evidence=[
            RemediationEvidence(
                evidence_type=(
                    "guardrail"
                ),
                description=(
                    description
                ),
                score=(
                    1.0
                    if passed
                    else 0.0
                ),
                attributes=dict(
                    attributes
                    or {}
                ),
            )
        ],
        attributes=dict(
            attributes
            or {}
        ),
    )


# ---------------------------------------------------------------
# Individual guardrails
# ---------------------------------------------------------------

def execution_mode_check(
    request: RemediationRequest,
    config: GuardrailConfig,
) -> GuardrailCheck:
    """
    Dry-run is always permitted by this check.

    Execute mode requires execution_enabled=True.
    """

    passed = (
        request.mode
        == RemediationMode.
        DRY_RUN.value
        or config.execution_enabled
    )

    return make_check(
        (
            "remediation."
            "execution-enabled"
        ),
        "Execution authority",
        passed,
        (
            "Dry-run mode is safe or "
            "execution is explicitly enabled."
        ),
        (
            "Execute mode was requested "
            "while execution is disabled."
        ),
        attributes={
            "mode": request.mode,
            "execution_enabled": (
                config.execution_enabled
            ),
        },
    )


def environment_check(
    request: RemediationRequest,
    config: GuardrailConfig,
) -> GuardrailCheck:
    """
    Require an explicit environment allowlist.

    Protected environments remain blocked even
    if they are accidentally also added to the
    allowlist.
    """

    environment = (
        request.environment_id
        .strip()
        .lower()
    )

    protected = {
        item.lower()
        for item
        in config.
        protected_environments
    }

    allowed = {
        item.lower()
        for item
        in config.
        allowed_environments
    }

    passed = (
        environment in allowed
        and environment
        not in protected
    )

    return make_check(
        "remediation.environment",
        "Environment restriction",
        passed,
        (
            "Environment is explicitly "
            "allowed for remediation "
            "evaluation."
        ),
        (
            "Environment is protected or "
            "is not in the execution "
            "allowlist."
        ),
        attributes={
            "environment": (
                request.environment_id
            ),
        },
    )


def action_type_check(
    request: RemediationRequest,
    config: GuardrailConfig,
) -> GuardrailCheck:
    passed = (
        request.action_type
        in set(
            config.
            allowed_action_types
        )
    )

    return make_check(
        "remediation.action-type",
        "Action allowlist",
        passed,
        (
            "Action type is explicitly "
            "allowlisted."
        ),
        (
            "Action type is not explicitly "
            "allowlisted."
        ),
        attributes={
            "action_type": (
                request.action_type
            ),
        },
    )


def approval_check(
    request: RemediationRequest,
    config: GuardrailConfig,
) -> GuardrailCheck:
    """
    Dry-run requests don't require execution
    approval.

    Execute requests require the expected
    approval handoff metadata.
    """

    if (
        request.mode
        == RemediationMode.
        DRY_RUN.value
    ):
        passed = True

        description = (
            "Dry-run mode does not require "
            "execution approval."
        )

    else:
        status_ok = (
            not config.
            require_approved_status
            or request.approval_status
            == "approved"
        )

        ids_ok = (
            not config.
            require_approval_ids
            or bool(
                request.approval_ids
            )
        )

        passed = (
            status_ok
            and ids_ok
        )

        description = (
            "Execution approval metadata "
            "is present."
            if passed
            else (
                "Execute mode requires "
                "approved status and "
                "approval identifiers."
            )
        )

    return make_check(
        "remediation.approval",
        "Approval handoff",
        passed,
        description,
        description,
        attributes={
            "approval_status": (
                request.approval_status
            ),
            "approval_count": len(
                request.approval_ids
            ),
        },
    )


def expiry_check(
    request: RemediationRequest,
    config: GuardrailConfig,
    *,
    now: Optional[
        datetime
    ] = None,
) -> GuardrailCheck:
    """
    Block stale execute requests.

    The request timestamp and target state may no
    longer be relevant after expiration.
    """

    if (
        request.mode
        == RemediationMode.
        DRY_RUN.value
        and not request.expires_at
    ):
        passed = True

        description = (
            "Dry-run request has no "
            "mandatory expiry requirement."
        )

    else:
        expiry = parse_timestamp(
            request.expires_at
        )

        current = (
            now
            or datetime.now(
                timezone.utc
            )
        )

        if current.tzinfo is None:
            current = current.replace(
                tzinfo=timezone.utc
            )

        passed = (
            expiry is not None
            and current.astimezone(
                timezone.utc
            )
            < expiry
        )

        description = (
            "Remediation request has not "
            "expired."
            if passed
            else (
                "Remediation request is "
                "expired or has no valid "
                "expiry timestamp."
            )
        )

        if (
            not config.
            require_request_expiry
            and expiry is None
        ):
            passed = True

            description = (
                "Request expiry is not "
                "required by configuration."
            )

    return make_check(
        (
            "remediation."
            "request-expiry"
        ),
        "Request freshness",
        passed,
        description,
        description,
        attributes={
            "expires_at": (
                request.expires_at
            ),
        },
    )


def target_scope_check(
    request: RemediationRequest,
    config: GuardrailConfig,
) -> GuardrailCheck:
    """
    Require explicit and bounded targets.
    """

    resource_count = len(
        set(
            request.
            target_resource_ids
        )
    )

    service_count = len(
        set(
            request.
            target_service_ids
        )
    )

    has_target = (
        resource_count > 0
        or service_count > 0
    )

    within_limits = (
        resource_count
        <= config.
        maximum_target_resources
        and service_count
        <= config.
        maximum_target_services
    )

    passed = (
        has_target
        and within_limits
    )

    return make_check(
        "remediation.target-scope",
        "Target scope",
        passed,
        (
            "Request has explicit targets "
            "within configured limits."
        ),
        (
            "Request has no target or "
            "exceeds configured target "
            "limits."
        ),
        attributes={
            "resource_count": (
                resource_count
            ),
            "service_count": (
                service_count
            ),
            "maximum_resources": (
                config.
                maximum_target_resources
            ),
            "maximum_services": (
                config.
                maximum_target_services
            ),
        },
    )


def risk_check(
    request: RemediationRequest,
    config: GuardrailConfig,
) -> GuardrailCheck:
    passed = (
        request.risk_score
        <= config.maximum_risk_score
    )

    return make_check(
        "remediation.risk-limit",
        "Risk tolerance",
        passed,
        (
            "Action risk is within the "
            "remediation limit."
        ),
        (
            "Action risk exceeds the "
            "remediation limit."
        ),
        attributes={
            "risk_score": (
                request.risk_score
            ),
            "risk_level": (
                request.risk_level
            ),
            "maximum_risk_score": (
                config.
                maximum_risk_score
            ),
        },
    )


def rollback_check(
    request: RemediationRequest,
    config: GuardrailConfig,
) -> GuardrailCheck:
    """
    Execute mode requires a rollback action when
    configured.

    A rollback name alone doesn't prove the
    rollback implementation works. Later
    validator/adapters modules must verify that
    independently.
    """

    if (
        request.mode
        == RemediationMode.
        DRY_RUN.value
    ):
        passed = True

        description = (
            "Dry-run mode does not require "
            "an executable rollback."
        )

    else:
        passed = (
            not config.
            require_rollback_for_execution
            or bool(
                request.
                rollback_action_type
            )
        )

        description = (
            "Rollback action is defined."
            if passed
            else (
                "Execute mode requires a "
                "rollback action."
            )
        )

    return make_check(
        "remediation.rollback",
        "Rollback availability",
        passed,
        description,
        description,
        attributes={
            "rollback_action_type": (
                request.
                rollback_action_type
            ),
        },
    )


def parameter_check(
    request: RemediationRequest,
    config: GuardrailConfig,
) -> GuardrailCheck:
    """
    Reject unexpected parameter names before an
    adapter receives the request.
    """

    unknown = sorted(
        set(
            request.parameters
        )
        - set(
            config.
            allowed_parameter_names
        )
    )

    passed = not (
        config.
        forbid_unknown_parameters
        and unknown
    )

    return make_check(
        "remediation.parameters",
        "Parameter allowlist",
        passed,
        (
            "All remediation parameters are "
            "allowlisted."
        ),
        (
            "Request contains parameters "
            "that are not allowlisted."
        ),
        attributes={
            "unknown_parameters": (
                unknown
            ),
        },
    )


# ---------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------

def evaluate_guardrails(
    request: RemediationRequest,
    *,
    config: Optional[
        GuardrailConfig
    ] = None,
    now: Optional[
        datetime
    ] = None,
) -> GuardrailEvaluation:
    configuration = (
        config
        or GuardrailConfig()
    )

    configuration.validate()

    request.finalize()

    checks = [
        execution_mode_check(
            request,
            configuration,
        ),
        environment_check(
            request,
            configuration,
        ),
        action_type_check(
            request,
            configuration,
        ),
        approval_check(
            request,
            configuration,
        ),
        expiry_check(
            request,
            configuration,
            now=now,
        ),
        target_scope_check(
            request,
            configuration,
        ),
        risk_check(
            request,
            configuration,
        ),
        rollback_check(
            request,
            configuration,
        ),
        parameter_check(
            request,
            configuration,
        ),
    ]

    blocking_reasons = [
        check.description
        for check in checks
        if (
            check.blocking
            and check.status
            == GuardrailStatus.
            FAILED.value
        )
    ]

    warnings = [
        check.description
        for check in checks
        if (
            check.status
            == GuardrailStatus.
            WARNING.value
        )
    ]

    return GuardrailEvaluation(
        checks=checks,
        blocking_reasons=(
            blocking_reasons
        ),
        warnings=warnings,
    ).finalize()


def guardrails_pass(
    request: RemediationRequest,
    *,
    config: Optional[
        GuardrailConfig
    ] = None,
) -> bool:
    return evaluate_guardrails(
        request,
        config=config,
    ).allowed


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------

def guardrail_summary(
    evaluations: Iterable[
        GuardrailEvaluation
    ],
) -> Dict[str, Any]:
    values = list(
        evaluations
    )

    checks = [
        check
        for evaluation in values
        for check in evaluation.checks
    ]

    return {
        "evaluation_count": len(
            values
        ),
        "allowed_count": sum(
            1
            for item in values
            if item.allowed
        ),
        "blocked_count": sum(
            1
            for item in values
            if not item.allowed
        ),
        "check_count": len(
            checks
        ),
        "failed_check_count": sum(
            1
            for check in checks
            if (
                check.status
                == GuardrailStatus.
                FAILED.value
            )
        ),
    }


class GuardrailEngine:
    """
    Reusable remediation guardrail evaluator.

    This class has no execution capability.
    """

    def __init__(
        self,
        *,
        config: Optional[
            GuardrailConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or GuardrailConfig()
        )

        self.config.validate()

    def evaluate(
        self,
        request: RemediationRequest,
        *,
        now: Optional[
            datetime
        ] = None,
    ) -> GuardrailEvaluation:
        return evaluate_guardrails(
            request,
            config=self.config,
            now=now,
        )

    def allows(
        self,
        request: RemediationRequest,
    ) -> bool:
        return self.evaluate(
            request
        ).allowed
