from __future__ import annotations

import math
from dataclasses import (
    dataclass,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.remediation.guardrails import (
    GuardrailConfig,
    GuardrailEngine,
)

from src.remediation.schema import (
    GuardrailCheck,
    GuardrailStatus,
    RemediationEvidence,
    RemediationMode,
    RemediationRequest,
    ValidationResult,
)


# ---------------------------------------------------------------
# Adapter capability contract
# ---------------------------------------------------------------

@dataclass(frozen=True)
class AdapterCapability:
    """
    Declarative description of a future execution
    adapter.

    This does not execute anything.

    adapters.py will later expose these
    capabilities through an adapter registry.
    """

    adapter_name: str

    supported_action_types: tuple[
        str,
        ...,
    ]

    supported_rollback_types: tuple[
        str,
        ...,
    ] = ()

    supported_environments: tuple[
        str,
        ...,
    ] = ()

    execution_supported: bool = False

    dry_run_supported: bool = True

    enabled: bool = True

    def validate(
        self,
    ) -> None:
        if not self.adapter_name.strip():
            raise ValueError(
                "adapter_name is required"
            )

        if not (
            self.supported_action_types
        ):
            raise ValueError(
                "Adapter must declare at "
                "least one supported "
                "action type"
            )


# ---------------------------------------------------------------
# Validator configuration
# ---------------------------------------------------------------

@dataclass(frozen=True)
class ValidatorConfig:
    """
    Additional remediation validation.

    Guardrails remain a separate first-line
    boundary.
    """

    maximum_timeout_seconds: int = 300

    minimum_timeout_seconds: int = 1

    maximum_absolute_capacity_delta: int = 10

    require_decision_id_for_execution: bool = (
        True
    )

    require_plan_id_for_execution: bool = True

    require_incident_id_for_execution: bool = (
        True
    )

    require_action_id: bool = True

    require_adapter_for_execution: bool = True

    require_rollback_support: bool = True

    require_single_environment_scope: bool = (
        True
    )

    def validate(
        self,
    ) -> None:
        if (
            self.minimum_timeout_seconds
            < 1
        ):
            raise ValueError(
                "minimum_timeout_seconds "
                "must be >= 1"
            )

        if (
            self.maximum_timeout_seconds
            < self.
            minimum_timeout_seconds
        ):
            raise ValueError(
                "maximum_timeout_seconds "
                "must not be less than "
                "minimum_timeout_seconds"
            )

        if (
            self.
            maximum_absolute_capacity_delta
            < 1
        ):
            raise ValueError(
                "maximum_absolute_capacity_"
                "delta must be >= 1"
            )


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def finite_number(
    value: Any,
) -> Optional[float]:
    if isinstance(
        value,
        bool,
    ):
        return None

    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        result
    ):
        return None

    return result


def validation_check(
    check_id: str,
    name: str,
    passed: bool,
    success: str,
    failure: str,
    *,
    blocking: bool = True,
    attributes: Optional[
        Dict[str, Any]
    ] = None,
) -> GuardrailCheck:
    """
    Reuse GuardrailCheck as the common
    check/evidence structure for validation.
    """

    description = (
        success
        if passed
        else failure
    )

    return GuardrailCheck(
        guardrail_id=check_id,
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
                    "validation"
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
# Structural validation
# ---------------------------------------------------------------

def identity_check(
    request: RemediationRequest,
    config: ValidatorConfig,
) -> GuardrailCheck:
    passed = bool(
        request.action_id.strip()
    )

    if not (
        config.require_action_id
    ):
        passed = True

    return validation_check(
        "validation.action-id",
        "Action identity",
        passed,
        (
            "Request contains an action ID."
        ),
        (
            "Remediation request is missing "
            "its action ID."
        ),
    )


def decision_handoff_check(
    request: RemediationRequest,
    config: ValidatorConfig,
) -> GuardrailCheck:
    """
    Execute mode must be traceable to its
    upstream decision objects.

    Dry-run may be used to evaluate a partial
    request during development.
    """

    if (
        request.mode
        == RemediationMode.
        DRY_RUN.value
    ):
        return validation_check(
            "validation.decision-handoff",
            "Decision handoff",
            True,
            (
                "Dry-run request may be "
                "validated without complete "
                "decision handoff metadata."
            ),
            "",
        )

    missing = []

    if (
        config.
        require_decision_id_for_execution
        and not request.decision_id
    ):
        missing.append(
            "decision_id"
        )

    if (
        config.
        require_plan_id_for_execution
        and not request.plan_id
    ):
        missing.append(
            "plan_id"
        )

    if (
        config.
        require_incident_id_for_execution
        and not request.incident_id
    ):
        missing.append(
            "incident_id"
        )

    passed = not missing

    return validation_check(
        "validation.decision-handoff",
        "Decision handoff",
        passed,
        (
            "Execution request is traceable "
            "to decision, plan and incident "
            "context."
        ),
        (
            "Execution request is missing "
            "required upstream decision "
            "identifiers."
        ),
        attributes={
            "missing": missing,
        },
    )


def target_identity_check(
    request: RemediationRequest,
) -> GuardrailCheck:
    """
    Reject blank target identifiers and duplicate
    target representations.
    """

    resource_values = [
        value.strip()
        for value
        in request.target_resource_ids
    ]

    service_values = [
        value.strip()
        for value
        in request.target_service_ids
    ]

    blank_resources = [
        value
        for value
        in resource_values
        if not value
    ]

    blank_services = [
        value
        for value
        in service_values
        if not value
    ]

    passed = not (
        blank_resources
        or blank_services
    )

    return validation_check(
        "validation.target-identity",
        "Target identity",
        passed,
        (
            "All target identifiers are "
            "non-empty."
        ),
        (
            "One or more target identifiers "
            "are empty."
        ),
    )


# ---------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------

def timeout_parameter_check(
    request: RemediationRequest,
    config: ValidatorConfig,
) -> GuardrailCheck:
    if (
        "timeout_seconds"
        not in request.parameters
    ):
        return validation_check(
            "validation.timeout",
            "Timeout parameter",
            True,
            (
                "No explicit timeout was "
                "provided."
            ),
            "",
            blocking=False,
        )

    value = finite_number(
        request.parameters[
            "timeout_seconds"
        ]
    )

    passed = (
        value is not None
        and value.is_integer()
        and (
            config.
            minimum_timeout_seconds
            <= value
            <= config.
            maximum_timeout_seconds
        )
    )

    return validation_check(
        "validation.timeout",
        "Timeout parameter",
        passed,
        (
            "Timeout is within configured "
            "bounds."
        ),
        (
            "timeout_seconds must be an "
            "integer inside configured "
            "bounds."
        ),
        attributes={
            "value": value,
            "minimum": (
                config.
                minimum_timeout_seconds
            ),
            "maximum": (
                config.
                maximum_timeout_seconds
            ),
        },
    )


def maximum_targets_check(
    request: RemediationRequest,
) -> GuardrailCheck:
    if (
        "maximum_targets"
        not in request.parameters
    ):
        return validation_check(
            "validation.maximum-targets",
            "Maximum target parameter",
            True,
            (
                "No per-action target bound "
                "was supplied."
            ),
            "",
            blocking=False,
        )

    value = finite_number(
        request.parameters[
            "maximum_targets"
        ]
    )

    target_count = len(
        set(
            request.
            target_resource_ids
        )
        | set(
            request.
            target_service_ids
        )
    )

    passed = (
        value is not None
        and value.is_integer()
        and value >= 1
        and target_count <= value
    )

    return validation_check(
        "validation.maximum-targets",
        "Maximum target parameter",
        passed,
        (
            "Actual target count is within "
            "the action-specific bound."
        ),
        (
            "maximum_targets is invalid or "
            "smaller than the actual target "
            "count."
        ),
        attributes={
            "value": value,
            "target_count": (
                target_count
            ),
        },
    )


def capacity_parameter_check(
    request: RemediationRequest,
    config: ValidatorConfig,
) -> GuardrailCheck:
    if (
        "capacity_delta"
        not in request.parameters
    ):
        return validation_check(
            "validation.capacity-delta",
            "Capacity parameter",
            True,
            (
                "No capacity adjustment was "
                "provided."
            ),
            "",
            blocking=False,
        )

    value = finite_number(
        request.parameters[
            "capacity_delta"
        ]
    )

    passed = (
        value is not None
        and value.is_integer()
        and value != 0
        and abs(value)
        <= config.
        maximum_absolute_capacity_delta
    )

    return validation_check(
        "validation.capacity-delta",
        "Capacity parameter",
        passed,
        (
            "Capacity delta is within the "
            "configured bound."
        ),
        (
            "capacity_delta must be a "
            "non-zero integer inside the "
            "configured bound."
        ),
        attributes={
            "value": value,
            "maximum_absolute_delta": (
                config.
                maximum_absolute_capacity_delta
            ),
        },
    )


def string_parameter_check(
    request: RemediationRequest,
    parameter_name: str,
) -> GuardrailCheck:
    if (
        parameter_name
        not in request.parameters
    ):
        return validation_check(
            (
                "validation."
                f"{parameter_name}"
            ),
            (
                f"{parameter_name} "
                "parameter"
            ),
            True,
            (
                f"No {parameter_name} "
                "parameter was supplied."
            ),
            "",
            blocking=False,
        )

    value = (
        request.parameters[
            parameter_name
        ]
    )

    passed = (
        isinstance(
            value,
            str,
        )
        and bool(
            value.strip()
        )
        and len(
            value
        )
        <= 512
    )

    return validation_check(
        (
            "validation."
            f"{parameter_name}"
        ),
        (
            f"{parameter_name} "
            "parameter"
        ),
        passed,
        (
            f"{parameter_name} is a "
            "bounded non-empty string."
        ),
        (
            f"{parameter_name} must be "
            "a bounded non-empty string."
        ),
    )


# ---------------------------------------------------------------
# Adapter capability
# ---------------------------------------------------------------

def matching_adapters(
    request: RemediationRequest,
    capabilities: Iterable[
        AdapterCapability
    ],
) -> List[
    AdapterCapability
]:
    output = []

    environment = (
        request.environment_id
        .strip()
        .lower()
    )

    for capability in capabilities:
        capability.validate()

        if not capability.enabled:
            continue

        if (
            request.action_type
            not in capability.
            supported_action_types
        ):
            continue

        allowed_environments = {
            value.lower()
            for value
            in capability.
            supported_environments
        }

        if (
            allowed_environments
            and environment
            not in allowed_environments
        ):
            continue

        output.append(
            capability
        )

    return output


def adapter_support_check(
    request: RemediationRequest,
    capabilities: Iterable[
        AdapterCapability
    ],
    config: ValidatorConfig,
) -> GuardrailCheck:
    matches = matching_adapters(
        request,
        capabilities,
    )

    if (
        request.mode
        == RemediationMode.
        DRY_RUN.value
    ):
        compatible = [
            item
            for item in matches
            if item.dry_run_supported
        ]

    else:
        compatible = [
            item
            for item in matches
            if em.execution_supported
        ]

    passed = bool(
        compatible
    )

    if (
        request.mode
        == RemediationMode.
        DRY_RUN.value
        and not capabilities
    ):
        # Early development can validate dry-run
        # requests before adapters.py exists.
        passed = True

    elif (
        request.mode
        == RemediationMode.
        EXECUTE.value
        and config.
        require_adapter_for_execution
    ):
        passed = bool(
            compatible
        )

    return validation_check(
        "validation.adapter-support",
        "Adapter support",
        passed,
        (
            "A compatible adapter capability "
            "is available, or no adapter is "
            "required for this dry-run."
        ),
        (
            "No compatible adapter supports "
            "the requested execution mode."
        ),
        attributes={
            "matching_adapters": [
                item.adapter_name
                for item
                in compatible
            ],
        },
    )


def rollback_support_check(
    request: RemediationRequest,
    capabilities: Iterable[
        AdapterCapability
    ],
    config: ValidatorConfig,
) -> GuardrailCheck:
    if (
        request.mode
        == RemediationMode.
        DRY_RUN.value
    ):
        return validation_check(
            "validation.rollback-support",
            "Rollback adapter support",
            True,
            (
                "Dry-run does not require "
                "executable rollback "
                "support."
            ),
            "",
        )

    if not (
        config.
        require_rollback_support
    ):
        return validation_check(
            "validation.rollback-support",
            "Rollback adapter support",
            True,
            (
                "Rollback adapter support "
                "is not required by "
                "configuration."
            ),
            "",
        )

    matches = matching_adapters(
        request,
        capabilities,
    )

    supported = any(
        (
            request.rollback_action_type
            in capability.
            supported_rollback_types
        )
        for capability in matches
        if (
            request.
            rollback_action_type
        )
    )

    return validation_check(
        "validation.rollback-support",
        "Rollback adapter support",
        supported,
        (
            "A matching adapter declares "
            "support for the rollback action."
        ),
        (
            "No matching adapter declares "
            "support for the rollback action."
        ),
        attributes={
            "rollback_action_type": (
                request.
                rollback_action_type
            ),
        },
    )


# ---------------------------------------------------------------
# Combined validator
# ---------------------------------------------------------------

def validate_request(
    request: RemediationRequest,
    *,
    guardrail_engine: Optional[
        GuardrailEngine
    ] = None,
    guardrail_config: Optional[
        GuardrailConfig
    ] = None,
    validator_config: Optional[
        ValidatorConfig
    ] = None,
    adapter_capabilities: Iterable[
        AdapterCapability
    ] = (),
) -> ValidationResult:
    """
    Fully validate a remediation request.

    Execution remains impossible from this
    module.
    """

    request.finalize()

    configuration = (
        validator_config
        or ValidatorConfig()
    )

    configuration.validate()

    guardrails = (
        guardrail_engine
        or GuardrailEngine(
            config=(
                guardrail_config
                or GuardrailConfig()
            )
        )
    )

    guardrail_result = (
        guardrails.evaluate(
            request
        )
    )

    capabilities = list(
        adapter_capabilities
    )

    checks = list(
        guardrail_result.checks
    )

    checks.extend(
        [
            identity_check(
                request,
                configuration,
            ),
            decision_handoff_check(
                request,
                configuration,
            ),
            target_identity_check(
                request
            ),
            timeout_parameter_check(
                request,
                configuration,
            ),
            maximum_targets_check(
                request
            ),
            capacity_parameter_check(
                request,
                configuration,
            ),
            string_parameter_check(
                request,
                "expected_state",
            ),
            string_parameter_check(
                request,
                "configuration_version",
            ),
            string_parameter_check(
                request,
                "reason",
            ),
            adapter_support_check(
                request,
                capabilities,
                configuration,
            ),
            rollback_support_check(
                request,
                capabilities,
                configuration,
            ),
        ]
    )

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
            not check.blocking
            and check.status
            == GuardrailStatus.
            FAILED.value
        )
    ]

    result = ValidationResult(
        checks=checks,
        blocking_reasons=(
            blocking_reasons
        ),
        warnings=warnings,
        attributes={
            "request_id": (
                request.request_id
            ),
            "mode": request.mode,
            "adapter_capability_count": (
                len(capabilities)
            ),
            "guardrails_allowed": (
                guardrail_result.allowed
            ),
        },
    )

    return result.finalize()


def validate_requests(
    requests: Iterable[
        RemediationRequest
    ],
    *,
    guardrail_engine: Optional[
        GuardrailEngine
    ] = None,
    validator_config: Optional[
        ValidatorConfig
    ] = None,
    adapter_capabilities: Iterable[
        AdapterCapability
    ] = (),
) -> List[
    ValidationResult
]:
    capabilities = list(
        adapter_capabilities
    )

    guardrails = (
        guardrail_engine
        or GuardrailEngine()
    )

    return [
        validate_request(
            request,
            guardrail_engine=(
                guardrails
            ),
            validator_config=(
                validator_config
            ),
            adapter_capabilities=(
                capabilities
            ),
        )
        for request
        in requests
    ]


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------

def validation_summary(
    results: Iterable[
        ValidationResult
    ],
) -> Dict[str, Any]:
    values = list(
        results
    )

    return {
        "validation_count": len(
            values
        ),
        "valid_count": sum(
            1
            for item in values
            if item.valid
        ),
        "invalid_count": sum(
            1
            for item in values
            if not item.valid
        ),
        "conditional_count": sum(
            1
            for item in values
            if (
                item.status
                == "conditional"
            )
        ),
        "blocking_reason_count": sum(
            len(
                item.blocking_reasons
            )
            for item in values
        ),
        "warning_count": sum(
            len(
                item.warnings
            )
            for item in values
        ),
    }


class RemediationValidator:
    """
    Reusable pre-execution validator.

    There is deliberately no execute() method.
    """

    def __init__(
        self,
        *,
        guardrail_engine: Optional[
            GuardrailEngine
        ] = None,
        config: Optional[
            ValidatorConfig
        ] = None,
        adapter_capabilities: Iterable[
            AdapterCapability
        ] = (),
    ) -> None:
        self.guardrail_engine = (
            guardrail_engine
            or GuardrailEngine()
        )

        self.config = (
            config
            or ValidatorConfig()
        )

        self.config.validate()

        self.adapter_capabilities = list(
            adapter_capabilities
        )

        for capability in (
            self.adapter_capabilities
        ):
            capability.validate()

    def validate(
        self,
        request: RemediationRequest,
    ) -> ValidationResult:
        return validate_request(
            request,
            guardrail_engine=(
                self.guardrail_engine
            ),
            validator_config=(
                self.config
            ),
            adapter_capabilities=(
                self.adapter_capabilities
            ),
        )

    def validate_many(
        self,
        requests: Iterable[
            RemediationRequest
        ],
    ) -> List[
        ValidationResult
    ]:
        return [
            self.validate(
                request
            )
            for request
            in requests
        ]
