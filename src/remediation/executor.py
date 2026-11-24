from __future__ import annotations

import re
import time
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
    AdapterExecutionDisabledError,
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
    ValidationResult,
    utc_now,
)

from src.remediation.validator import (
    RemediationValidator,
    ValidatorConfig,
)


# ---------------------------------------------------------------
# Executor configuration
# ---------------------------------------------------------------


@dataclass(frozen=True)
class ExecutorConfig:
    """
    Execution-orchestration configuration.

    Actual infrastructure execution remains
    disabled by default.

    Setting execution_enabled=True here alone is
    not sufficient. Guardrails, validation and an
    execution-capable adapter must also permit
    the request.
    """

    execution_enabled: bool = False

    maximum_output_chars: int = 4096

    redact_output: bool = True

    preferred_adapter: Optional[str] = None

    catch_adapter_exceptions: bool = True

    def validate(
        self,
    ) -> None:
        if self.maximum_output_chars < 128:
            raise ValueError("maximum_output_chars must be >= 128")


# ---------------------------------------------------------------
# Output protection
# ---------------------------------------------------------------

REDACTION_PATTERNS = (
    re.compile(
        r"(?i)"
        r"(password|passwd|secret|token|"
        r"api[_-]?key|access[_-]?key)"
        r"\s*[:=]\s*"
        r"([^\s,;]+)"
    ),
    re.compile(
        r"(?i)"
        r"(authorization\s*:\s*bearer)"
        r"\s+([^\s]+)"
    ),
)


def redact_text(
    value: Optional[str],
) -> Optional[str]:
    """
    Apply basic defensive redaction.

    This should complement upstream telemetry
    redaction rather than replace it.
    """

    if value is None:
        return None

    text = str(value)

    for pattern in REDACTION_PATTERNS:
        text = pattern.sub(
            lambda match: f"{match.group(1)}=[REDACTED]",
            text,
        )

    return text


def truncate_text(
    value: Optional[str],
    *,
    maximum_chars: int,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value)

    if len(text) <= maximum_chars:
        return text

    suffix = "\n[OUTPUT TRUNCATED]"

    remaining = max(
        0,
        maximum_chars - len(suffix),
    )

    return text[:remaining] + suffix


def sanitize_output(
    value: Optional[str],
    *,
    config: ExecutorConfig,
) -> Optional[str]:
    result = value

    if config.redact_output:
        result = redact_text(result)

    return truncate_text(
        result,
        maximum_chars=(config.maximum_output_chars),
    )


def sanitize_execution_result(
    result: ExecutionResult,
    *,
    config: ExecutorConfig,
) -> ExecutionResult:
    """
    Ensure adapter output is bounded before it is
    passed into audit or memory.
    """

    result.stdout = sanitize_output(
        result.stdout,
        config=config,
    )

    result.stderr = sanitize_output(
        result.stderr,
        config=config,
    )

    return result.finalize()


# ---------------------------------------------------------------
# Blocked/failure results
# ---------------------------------------------------------------


def blocked_result(
    request: RemediationRequest,
    validation: ValidationResult,
) -> ExecutionResult:
    """
    Represent refusal to execute as structured
    data rather than raising through the pipeline.
    """

    return ExecutionResult(
        status=(ExecutionStatus.BLOCKED.value),
        success=False,
        operation=(request.action_type),
        started_at=utc_now(),
        finished_at=utc_now(),
        changed=False,
        evidence=[
            RemediationEvidence(
                evidence_type=("execution_blocked"),
                description=("Remediation execution was blocked by validation."),
                score=0.0,
                source_id=(request.request_id),
                resource_ids=list(request.target_resource_ids),
                attributes={
                    "validation_status": (validation.status),
                    "blocking_reasons": (list(validation.blocking_reasons)),
                },
            )
        ],
        attributes={
            "validation": (validation.to_dict()),
        },
    ).finalize()


def failed_result(
    request: RemediationRequest,
    *,
    adapter_name: Optional[str],
    error_type: str,
    error_message: str,
    started_at: str,
    duration_seconds: float,
) -> ExecutionResult:
    return ExecutionResult(
        status=(ExecutionStatus.FAILED.value),
        success=False,
        adapter_name=adapter_name,
        operation=(request.action_type),
        started_at=started_at,
        finished_at=utc_now(),
        duration_seconds=(duration_seconds),
        changed=False,
        stderr=error_message,
        evidence=[
            RemediationEvidence(
                evidence_type=("adapter_failure"),
                description=("The remediation adapter failed or refused the request."),
                score=0.0,
                source_id=(request.request_id),
                resource_ids=list(request.target_resource_ids),
                attributes={
                    "error_type": (error_type),
                    "adapter_name": (adapter_name),
                },
            )
        ],
        attributes={
            "error_type": error_type,
        },
    ).finalize()


# ---------------------------------------------------------------
# Execution
# ---------------------------------------------------------------


def execute_request(
    request: RemediationRequest,
    *,
    validator: Optional[RemediationValidator] = None,
    registry: Optional[AdapterRegistry] = None,
    config: Optional[ExecutorConfig] = None,
) -> ExecutionResult:
    """
    Validate and dispatch one remediation
    request.

    Default behavior can only reach a
    DryRunAdapter.

    No fallback from execute -> dry_run occurs.
    """

    configuration = config or ExecutorConfig()

    configuration.validate()

    adapters = registry or default_adapter_registry()

    active_validator = validator or RemediationValidator(
        config=(ValidatorConfig()),
        adapter_capabilities=(adapters.capabilities()),
    )

    request.finalize()

    validation = active_validator.validate(request)

    if not validation.valid:
        return blocked_result(
            request,
            validation,
        )

    # Independent executor kill-switch.
    #
    # Guardrails also check execution authority,
    # but the executor itself must independently
    # refuse execute mode when disabled.
    if (
        request.mode == RemediationMode.EXECUTE.value
        and not configuration.execution_enabled
    ):
        return ExecutionResult(
            status=(ExecutionStatus.BLOCKED.value),
            success=False,
            operation=(request.action_type),
            started_at=utc_now(),
            finished_at=utc_now(),
            changed=False,
            evidence=[
                RemediationEvidence(
                    evidence_type=("execution_kill_switch"),
                    description=(
                        "Executor refused "
                        "execute mode because "
                        "execution_enabled "
                        "is false."
                    ),
                    score=0.0,
                    source_id=(request.request_id),
                    resource_ids=list(request.target_resource_ids),
                )
            ],
            attributes={
                "execution_enabled": False,
            },
        ).finalize()

    started_at = utc_now()

    monotonic_started = time.monotonic()

    adapter_name = None

    try:
        adapter = adapters.resolve(
            request,
            preferred_adapter=(configuration.preferred_adapter),
        )

        adapter_name = adapter.capability.adapter_name

        result = adapter.execute(request)

        result.attributes["validation"] = validation.to_dict()

        result.attributes["request_id"] = request.request_id

        return sanitize_execution_result(
            result,
            config=configuration,
        )

    except (
        AdapterNotFoundError,
        AdapterExecutionDisabledError,
        AdapterError,
    ) as exc:
        duration = time.monotonic() - monotonic_started

        return sanitize_execution_result(
            failed_result(
                request,
                adapter_name=(adapter_name),
                error_type=(type(exc).__name__),
                error_message=str(exc),
                started_at=(started_at),
                duration_seconds=(duration),
            ),
            config=configuration,
        )

    except Exception as exc:
        if not (configuration.catch_adapter_exceptions):
            raise

        duration = time.monotonic() - monotonic_started

        # Do not expose traceback or exception
        # internals in the result.
        return sanitize_execution_result(
            failed_result(
                request,
                adapter_name=(adapter_name),
                error_type=(type(exc).__name__),
                error_message=("Unexpected adapter execution failure."),
                started_at=(started_at),
                duration_seconds=(duration),
            ),
            config=configuration,
        )


def execute_requests(
    requests: Iterable[RemediationRequest],
    *,
    validator: Optional[RemediationValidator] = None,
    registry: Optional[AdapterRegistry] = None,
    config: Optional[ExecutorConfig] = None,
) -> List[ExecutionResult]:
    adapters = registry or default_adapter_registry()

    configuration = config or ExecutorConfig()

    configuration.validate()

    active_validator = validator or RemediationValidator(
        adapter_capabilities=(adapters.capabilities())
    )

    return [
        execute_request(
            request,
            validator=(active_validator),
            registry=adapters,
            config=configuration,
        )
        for request in requests
    ]


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------


def execution_summary(
    results: Iterable[ExecutionResult],
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
        "execution_count": len(values),
        "success_count": sum(1 for result in values if result.success),
        "changed_count": sum(1 for result in values if result.changed),
        "blocked_count": sum(
            1 for result in values if (result.status == ExecutionStatus.BLOCKED.value)
        ),
        "failed_count": sum(
            1
            for result in values
            if (
                result.status
                in {
                    ExecutionStatus.FAILED.value,
                    ExecutionStatus.TIMED_OUT.value,
                }
            )
        ),
        "statuses": statuses,
    }


class RemediationExecutor:
    """
    Controlled adapter dispatcher.

    Default construction is dry-run-only.
    """

    def __init__(
        self,
        *,
        registry: Optional[AdapterRegistry] = None,
        validator: Optional[RemediationValidator] = None,
        config: Optional[ExecutorConfig] = None,
    ) -> None:
        self.registry = registry or default_adapter_registry()

        self.config = config or ExecutorConfig()

        self.config.validate()

        self.validator = validator or RemediationValidator(
            adapter_capabilities=(self.registry.capabilities())
        )

    def execute(
        self,
        request: RemediationRequest,
    ) -> ExecutionResult:
        return execute_request(
            request,
            validator=self.validator,
            registry=self.registry,
            config=self.config,
        )

    def execute_many(
        self,
        requests: Iterable[RemediationRequest],
    ) -> List[ExecutionResult]:
        return [self.execute(request) for request in requests]
