from __future__ import annotations

"""
Shared exception hierarchy for OSHA.

This module is intentionally dependency-free.

Subsystems can inherit from these exceptions
without core importing subsystem code.
"""

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)


# ---------------------------------------------------------------
# Base error
# ---------------------------------------------------------------

class OshaError(
    Exception
):
    """
    Base class for expected OSHA application
    errors.

    retryable describes error semantics only.
    Retry policy still belongs to the subsystem
    processing the operation.
    """

    default_code = "osha_error"

    def __init__(
        self,
        message: str = (
            "OSHA operation failed."
        ),
        *,
        code: Optional[
            str
        ] = None,
        retryable: bool = False,
        details: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        self.message = (
            str(message).strip()
            or "OSHA operation failed."
        )

        self.code = (
            str(
                code
                or self.default_code
            ).strip()
            or self.default_code
        )

        self.retryable = bool(
            retryable
        )

        self.details: Dict[
            str,
            Any,
        ] = dict(
            details
            or {}
        )

        super().__init__(
            self.message
        )

    def to_dict(
        self,
        *,
        include_details: bool = False,
    ) -> Dict[str, Any]:
        """
        Return bounded error metadata.

        Details are excluded unless explicitly
        requested because they may contain
        subsystem-specific information.
        """

        value: Dict[
            str,
            Any,
        ] = {
            "type": (
                type(self).__name__
            ),
            "code": self.code,
            "message": self.message,
            "retryable": (
                self.retryable
            ),
        }

        if (
            include_details
            and self.details
        ):
            value[
                "details"
            ] = dict(
                self.details
            )

        return value


# ---------------------------------------------------------------
# Configuration / validation
# ---------------------------------------------------------------

class ConfigurationError(
    OshaError
):
    default_code = (
        "configuration_error"
    )


class ValidationError(
    OshaError
):
    default_code = (
        "validation_error"
    )


class IdentifierError(
    ValidationError
):
    default_code = (
        "identifier_error"
    )


class SerializationError(
    OshaError
):
    default_code = (
        "serialization_error"
    )


# ---------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------

class LifecycleError(
    OshaError
):
    default_code = (
        "lifecycle_error"
    )


class StateTransitionError(
    LifecycleError
):
    default_code = (
        "state_transition_error"
    )


# ---------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------

class DependencyError(
    OshaError
):
    default_code = (
        "dependency_error"
    )


class DependencyUnavailableError(
    DependencyError
):
    default_code = (
        "dependency_unavailable"
    )


# ---------------------------------------------------------------
# Policy
# ---------------------------------------------------------------

class PolicyError(
    OshaError
):
    default_code = (
        "policy_error"
    )


class PolicyBlockedError(
    PolicyError
):
    """
    A controlled policy block.

    Policy blocks are non-retryable by default.
    """

    default_code = (
        "policy_blocked"
    )

    def __init__(
        self,
        message: str = (
            "Operation blocked by policy."
        ),
        *,
        code: Optional[
            str
        ] = None,
        details: Optional[
            Mapping[
                str,
                Any,
            ]
        ] = None,
    ) -> None:
        super().__init__(
            message,
            code=(
                code
                or self.default_code
            ),
            retryable=False,
            details=details,
        )


class ApprovalRequiredError(
    PolicyBlockedError
):
    default_code = (
        "approval_required"
    )


class SafetyGuardrailError(
    PolicyBlockedError
):
    default_code = (
        "safety_guardrail"
    )


# ---------------------------------------------------------------
# Queue / lease
# ---------------------------------------------------------------

class QueueError(
    OshaError
):
    default_code = (
        "queue_error"
    )


class JobNotFoundError(
    QueueError
):
    default_code = (
        "job_not_found"
    )


class LeaseError(
    QueueError
):
    default_code = (
        "lease_error"
    )


class LeaseOwnershipError(
    LeaseError
):
    default_code = (
        "lease_ownership_error"
    )


class LeaseExpiredError(
    LeaseError
):
    default_code = (
        "lease_expired"
    )


# ---------------------------------------------------------------
# Workers
# ---------------------------------------------------------------

class WorkerError(
    OshaError
):
    default_code = (
        "worker_error"
    )


# ---------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------

class OrchestrationError(
    OshaError
):
    default_code = (
        "orchestration_error"
    )


# ---------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------

class RuntimeErrorBase(
    OshaError
):
    """
    OSHA runtime-layer error.

    Named RuntimeErrorBase to avoid confusion
    with Python's built-in RuntimeError.
    """

    default_code = (
        "runtime_error"
    )


# ---------------------------------------------------------------
# Integration
# ---------------------------------------------------------------

class IntegrationError(
    OshaError
):
    default_code = (
        "integration_error"
    )


class ExternalServiceError(
    IntegrationError
):
    default_code = (
        "external_service_error"
    )


# ---------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------

class PersistenceError(
    OshaError
):
    default_code = (
        "persistence_error"
    )


# ---------------------------------------------------------------
# Security
# ---------------------------------------------------------------

class SecurityError(
    OshaError
):
    default_code = (
        "security_error"
    )


class AuthorizationError(
    SecurityError
):
    default_code = (
        "authorization_error"
    )


class AuthenticationError(
    SecurityError
):
    default_code = (
        "authentication_error"
    )


# ---------------------------------------------------------------
# Generic resource errors
# ---------------------------------------------------------------

class NotFoundError(
    OshaError
):
    default_code = (
        "not_found"
    )


class ConflictError(
    OshaError
):
    default_code = (
        "conflict"
    )


class CapacityError(
    OshaError
):
    default_code = (
        "capacity_error"
    )


class TimeoutErrorBase(
    OshaError
):
    """
    OSHA timeout error.

    Named TimeoutErrorBase to distinguish it
    from Python's built-in TimeoutError.
    """

    default_code = "timeout"


# ---------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------

__all__ = [
    "OshaError",

    "ConfigurationError",
    "ValidationError",
    "IdentifierError",
    "SerializationError",

    "LifecycleError",
    "StateTransitionError",

    "DependencyError",
    "DependencyUnavailableError",

    "PolicyError",
    "PolicyBlockedError",
    "ApprovalRequiredError",
    "SafetyGuardrailError",

    "QueueError",
    "JobNotFoundError",
    "LeaseError",
    "LeaseOwnershipError",
    "LeaseExpiredError",

    "WorkerError",
    "OrchestrationError",
    "RuntimeErrorBase",

    "IntegrationError",
    "ExternalServiceError",

    "PersistenceError",

    "SecurityError",
    "AuthorizationError",
    "AuthenticationError",

    "NotFoundError",
    "ConflictError",
    "CapacityError",
    "TimeoutErrorBase",
]
